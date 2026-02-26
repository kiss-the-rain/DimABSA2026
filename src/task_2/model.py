from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np, pandas as pd
from contextlib import nullcontext

import torch, os, json,random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers.utils.logging import set_verbosity_error
from torch.amp import GradScaler
from collections import defaultdict as _dd

from sklearn.model_selection import GroupShuffleSplit, train_test_split

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 项目里的路径工具
from src.utils.paths import output_path

# ================= 配置 =================
@dataclass
class Config:
    # —— 模型/数据基础 ——
    local_model_dir: str = "models/roberta-base"  # 本地权重路径（离线加载）；换模型时指到对应目录
    max_len: int = 224                            # tokenizer 截断长度；长句可提到 256/320，显存↑ 训练慢
    batch_size: int = 16                          # 训练批大小；显存不够可减小并配合梯度累积
    epochs: int = 30                              # 训练轮数；配合 early stop，一般 10–20 足够
    seed: int = 42                                # 随机种子；复现用
    num_workers: int = 14

    # —— 学习率与优化 ——
    lr_encoder: float = 1e-5                      # 编码器（RoBERTa/BERT）学习率；解冻后也会用到
    lr_heads: float   = 8e-5                      # 任务头（BIO/配对/VA）的学习率；过大易抖动、过小收敛慢
    weight_decay: float = 0.02                    # L2 权重衰减（正则）；0.01–0.05 常用，过大欠拟合
    warmup_ratio: float = 0.12                    # 线性 warmup 比例；小数据/大 LR 建议 0.06–0.1
    warmup_min_steps: int = 100                   # warmup 的最小步数下限；总步数太少时的兜底
    freeze_epochs: int = 4                        # 前几轮只训头部（冻结 encoder）；可提升早期稳定与召回
    enc_lr_after_unfreeze: float = 5e-6           # 解冻后 encoder 的 LR；越小越把学习量留给任务头
    patience: int = 2                             # 验证无提升的容忍轮数（早停）；曲线抖动大可增至 8–10

    # —— 解码/候选（控制召回与误报） ——
    pair_window: int = 160                        # A/O 起点最大距离（token）；增大→召回↑但误配风险↑
    topk_pairs: int  = 16                         # 每句最多输出配对候选；过大可能引入重复/噪声
    pair_thresh: float = 0.25                     # 非 None 类的最小概率阈；降→召回↑ 精度↓，建议 0.22–0.40
    max_decode_len: int = 256                     # 推理阶段对极长句的保护性上限；避免极端长文本拖慢
    use_amp: bool = True                          # 混合精度（AMP）；显存省、速度快，数值不稳时可关

    # —— VA 回归（情感强度/唤醒度） ——
    va_head_dim: int = 64                         # VA 头的隐藏维度；32/64/128 常用，过大易过拟合
    lambda_pair: float = 0.5                     # 总损失中“配对分类”权重；↑能抑制判 None 倾向，但太大压 BIO/VA
    gamma_focal: float = 2.0                      # focal loss 的聚焦参数；2.0 常用，↑更关注难样本
    lambda_va: float = 0.5                        # VA 回归损失权重；调平分类与回归的相对重要性

    # —— 类不平衡校正（配对分类的 focal α 系数） ——
    alpha_none: float = 0.99                      # None 类的权重（越小→越不偏向判 None，召回↑）
    alpha_pos:  float = 0.01                          # 非 None 类权重（与上相对）；两者一般和≈1.0



# ================ 随机种子/设备 ================
def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
torch.set_float32_matmul_precision("high")

# ================ BIO 标签 ================
BIO_LABELS = ["O","B","I"]
BIO2ID = {t:i for i,t in enumerate(BIO_LABELS)} # {"O": 0, "B": 1, "I": 2}

# ================ 工具：字符→token 对齐 ================
def find_span(text: str, surface: str) -> Optional[Tuple[int,int]]:
    '''
    在原始字符串 text 里寻找子串 surface 的字符级区间，返回 (start, end)（start 含、end 不含）
    :param text:
    :param surface:
    :return:
    '''
    if not surface or surface == "NULL":
        return None
    st = text.find(surface)
    if st < 0:
        return None
    return st, st + len(surface)

def char_to_token_span(offsets: List[Tuple[int,int]], ch_st: int, ch_ed: int) -> Optional[Tuple[int,int]]:
    '''
    把一个字符区间 [ch_st, ch_ed)（左闭右开）映射成 token 区间 (i, j)（两端都包含）

    :param offsets: 由 tokenizer 生成的每个 token 对应的字符偏移量列表。offsets[t] = (s, e) 表示第 t 个 token 在原始文本中覆盖的字符范围是 [s, e)（左闭右开）。
        例如：文本 "Good battery!" → tokens ["Good", "battery", "!"] → offsets 可能是 [(0,4), (5,12), (12,13)]
    :param ch_st,ch_ed: 目标字符区间，左闭右开，即 [ch_st, ch_ed)
    :return: 对应的 token 区间 (i, j)，表示从第 i 个 token 到第 j 个 token（包含两端），或 None（无法映射）
    '''
    i = j = None
    for tid,(s,e) in enumerate(offsets):
        # tid 表示index
        # offsets[t] = (s, e) 表示第 t 个 token 对应原文的字符范围 [s, e)
        if s==e:  # 特殊/CLS/SEP
            continue
        if i is None and not (e<=ch_st or s>=ch_ed):
            i = tid
        if not (e<=ch_st or s>=ch_ed):
            j = tid
    if i is None or j is None:
        return None
    return i, j

# ===== 起点法 BIO 解码（带阈值） =====

def _decode_spans_B(prob, mask, tau=0.30):
    '''
    在一条序列里找出所有满足条件的 BIO 片段，并返回这些片段的token 下标区间 (start, end)（两端都包含）。它可能返回多个片段，也可能一个都没有。
    :param prob: 形状 [B, L, 3] 的概率张量（不是 logits）。最后一维 3 对应 BIO_LABELS=["O","B","I"]
    :param mask:形状 [B, L] 的有效位置掩码 :为True 代表第 B 个样本的第 L 个 token 是有效的
    :param tau:起点阈值（只对 B 标签的概率做阈值），默认 0.30。低于该阈值的 B 会被忽略。
    :return:返回这些片段的token 下标区间 (start, end)（两端都包含）。它可能返回多个片段，也可能一个都没有
    '''
    #? prob:形状 [B, L, 3] 的概率张量（不是 logits）。最后一维 3 对应 BIO_LABELS=["O","B","I"]
    #? mask: 形状 [B, L] 的有效位置掩码
    #? tau: 起点阈值（只对 B 标签的概率做阈值），默认 0.30。低于该阈值的 B 会被忽略。
    tag = prob.argmax(-1)   # 获取最可能的标签序列
    #todo [B,L] tag[b, t] 是该 token 在句子中的BIO编码 tag[b, t]的取值只会是 0/1/2，对应 O/B/I 三类中的索引。
    #todo tag[b, l] 代表的是：第 b 条样本里第 l 个 token 的“预测标签索引”（哪一类概率最大），
    #todo B = batch size（一次送进模型的样本条数）L = 序列长度（每条样本的 token 数），也就是 tokenizer 编码后、pad 对齐后的长度。
    outs = []
    for i in range(tag.size(0)):
        cur=[] # cur 收集该样本的所有片段 (st, ed)（包含端点的 token 下标）
        if not torch.is_floating_point(mask[i]): #? 返回一个 Python 布尔值，表示 mask[i] 这个张量是否是“浮点类型”
            L = int(mask[i].sum().item()) # mask[i]是一个一维数组,L是当前样本的实际有效长度
        else:
            L = int((mask[i] > 0.5).sum().item())
        t = 0
        while t < L:
            #? tag[b, t] 是该 token 在句子中的BIO 编码
            if tag[i,t].item()==BIO2ID["B"] and prob[i,t,BIO2ID["B"]].item()>=tau:
                # 满足tag[i,t].item()==BIO2ID["B"],则表示找到样本的起始点
                st=t
                ed=t
                t+=1
                while t<L and tag[i,t].item()==BIO2ID["I"]:
                    ed=t
                    t+=1
                cur.append((st,ed)) #? 每个 outs[i] 是若干 (st, ed) 片段
            else:
                t+=1
        outs.append(cur)
    return outs

WEAK_OPINIONS = {
    "very","really","finally","quite","somewhat","kinda","sorta","rather","so","too","enough","extremely","basically"
}

def _is_weak_opinion(opn: str) -> bool:
    '''
    定义一个弱意见词表（全小写）.如果 opinion 片段正好是这些词之一，倾向于判定为“弱”，从而在候选中丢弃或降权
    :param opn:
    :return:
    '''
    t = (opn or "").strip().lower()
    if t in WEAK_OPINIONS:
        return True
    letters = sum(ch.isalpha() for ch in t) # 统计 t 中字母字符的个数（isalpha()），得到 letters。
    if letters <= 1:
        return True
    if letters < 3 and len(t) <= 3: # 非常短的片段（总长度 ≤ 3，且其中字母数 < 3） → 弱意见。
        return True
    return False

# ================ 数据集 ================
class TrainEx:
    __slots__ = ("text","a_span","o_span","cat","id","va_v","va_a")
    #? __slots__ 是 Python 类的一个“属性布局清单”。给类定义了 __slots__ 之后，解释器会为这些名字预先分配固定的槽位
    def __init__(self, text:str, a_span:Tuple[int,int], o_span:Tuple[int,int],cat:str, rid:str, va_v: float, va_a: float):
        self.text = text
        self.a_span = a_span
        self.o_span = o_span
        self.cat = cat
        self.id = rid
        self.va_v = va_v
        self.va_a = va_a

def build_label_map(cats: List[str]) -> Dict[str,int]:
    '''
    类别到整数ID的映射字典
    :param cats:
    :return:
    '''
    uniq = sorted({c for c in cats if c and c!="NULL"})
    id_map = {"None":0}
    for i,c in enumerate(uniq, start=1): # 从 1 开始给每个真实类别编号（避免与 0:None 冲突）。
        id_map[c]=i
    return id_map

class ASTETrainDataset(Dataset):
    '''
    把一行 DataFrame 记录加工成可训练的特征字典
    '''

    def __init__(self, df: pd.DataFrame, tok: AutoTokenizer, max_len: int, cat2id: Dict[str,int]):
        self.tok = tok
        self.max_len = max_len
        self.cat2id = cat2id
        rows: List[TrainEx] = []
        for _,r in df.iterrows(): # iterrows 是 Pandas DataFrame 提供的一个方法，用于按行遍历 DataFrame
            text = str(r["text"])
            a = str(r["aspect"])
            o = str(r["opinion"])
            c = str(r["category"])
            rid = str(r["id"])
            va_raw = str(r.get("VA", "5.0#5.0"))
            try:
                v_str, a_str = va_raw.split("#")
                va_v, va_a = float(v_str), float(a_str)
            except Exception:
                va_v, va_a = 5.0, 5.0

            a_span = find_span(text, a)
            o_span = find_span(text, o) # 返回Aspect和Opinion
            if a_span is None or o_span is None:
                continue
            rows.append(TrainEx(text, a_span, o_span, c, rid, va_v, va_a))
        self.rows = rows

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        ex = self.rows[i]
        enc = self.tok(ex.text, truncation=True, max_length=self.max_len, return_offsets_mapping=True)
        #? 用 HuggingFace fast 分词器对原句编码。
        offsets = enc.pop("offset_mapping")
        #? 取出 (s,e) 列表（长度 L），并从 enc 移除，避免后续 pad 干扰。offsets[t] = (s,e)
        a_tok = char_to_token_span(offsets, *ex.a_span)
        o_tok = char_to_token_span(offsets, *ex.o_span)# 把字符串的起始和结束位置,转换为token中的起始和结束位置
        L = len(enc["input_ids"])
        # 初始化两套 BIO 标签（Aspect/Opinion），长度 L，默认全 O
        a_tags = [BIO2ID["O"]]*L
        o_tags = [BIO2ID["O"]]*L
        if a_tok is not None:
            # 若 Aspect 映射成功,在 ai 打 B，(ai+1..aj) 打 I；
            ai,aj = a_tok
            a_tags[ai] = BIO2ID["B"]
            for t in range(ai+1,aj+1):
                a_tags[t]=BIO2ID["I"]
        else:
            ai = -1
        if o_tok is not None:
            oi,oj = o_tok
            o_tags[oi] = BIO2ID["B"]
            for t in range(oi+1,oj+1): o_tags[t]=BIO2ID["I"]
        else:
            oi = -1
        #todo 至此，a_tags/o_tags 是未 pad的 BIO 标签序列，长度与当前样本 token 数一致
        pair = (max(ai,0), max(oi,0))
        #? 配对监督（pair_ij）：用 A/O 的起点 token 作为监督坐标 (i,j)
        cat = self.cat2id.get(ex.cat, 0) # ex.cat：当前三元组的情感类别（如 "POS"）
        #? 类别监督：把字符串类别映射成整数 ID。若找不到（理论上不该），回退为 0（"None" 类）。

        enc["a_tags"]   = torch.tensor(a_tags, dtype=torch.long)
        enc["o_tags"]   = torch.tensor(o_tags, dtype=torch.long)
        enc["pair_ij"]  = torch.tensor(pair, dtype=torch.long)
        enc["pair_cat"] = torch.tensor(cat, dtype=torch.long)
        enc["va_target"] = torch.tensor([ex.va_v, ex.va_a], dtype=torch.float)  # [2]
        # 所有监督信号塞回 enc（分词器返回的字典里）并转成张量，交给 collator 统一 pad
        return enc

class ASTEDevDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tok: AutoTokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.max_len = max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(r["text"], truncation=True, max_length=min(self.max_len, Config.max_len),return_offsets_mapping=True)
        enc["offsets"] = enc.pop("offset_mapping")
        enc["id"] = str(r["id"])
        enc["text"] = r["text"]
        return enc

# ================ Collator ================
class TrainCollator:
    '''
    使用 tok.pad() 对 batch 内的 token IDs、attention masks 等进行填充（padding）；
    pad_to_multiple_of=8：将序列长度填充至 8 的倍数，提升 GPU 计算效率（尤其在 Tensor Core 上）；
    同时处理标签（如 span 标签、pair 标签、VA 值）的对齐与填充；
    返回一个包含所有张量的字典或命名元组，供模型直接使用
    '''
    def __init__(self, tok, pad_to_multiple_of=8):
        self.tok = tok
        self.mult = pad_to_multiple_of   # pad_to_multiple_of=8：把序列长度 pad 到 8 的倍数，
    def __call__(self, feats: List[Dict[str,Any]]):
        a_tags = [f.pop("a_tags") for f in feats]
        o_tags = [f.pop("o_tags") for f in feats] # 从每个样本字典里取出两套 BIO 标签序列（Aspect/Opinion）
        pair_ij = torch.stack([f.pop("pair_ij") for f in feats],0) # [B, 2]（每条样本的 A/O 起点 token 坐标 (i,j)）
        pair_cat= torch.stack([f.pop("pair_cat") for f in feats],0) # [B]（每条样本的配对类别 id）
        va_targets = torch.stack([f.pop("va_target") for f in feats],0)  # [B,2] （每条样本的 (V,A) 回归目标）
        # 先 pop 掉它们，是为了让后面的 tok.pad(...) 只处理“分词器认识的键”，避免报错/干扰。
        batch = self.tok.pad(feats, padding=True, pad_to_multiple_of=self.mult, return_tensors="pt")
        # 用分词器自带的 pad 对剩下的输入键做自动补齐。return_tensors="pt"：直接返回 PyTorch 张量。
        maxL = batch["input_ids"].size(1)

        # 把一批变长的一维序列（比如每条样本的 BIO 标签序列）按当前 batch 的统一长度 maxL 补齐到同一长度，并返回一个形状为 [B, maxL] 的 LongTensor
        def pad1d(arr, fill=0):
            out = torch.full((len(arr), maxL), fill, dtype=torch.long)
            # 先创建一个 [B, L_pad] 的整型二维张量 out，用 fill 值（默认 0）填充。
            for i, t in enumerate(arr):
                L = min(len(t), maxL)
                out[i, :L] = t[:L] # 把arr的内容拷贝到out中
            return out
        batch["a_tags"] = pad1d(a_tags, fill=BIO2ID["O"])
        batch["o_tags"] = pad1d(o_tags, fill=BIO2ID["O"])
        batch["pair_ij"] = pair_ij
        batch["pair_cat"]= pair_cat
        batch["va_target"] = va_targets
        return batch

class DevCollator:
    def __init__(self, tok, pad_to_multiple_of=8):
        self.tok = tok
        self.mult = pad_to_multiple_of
    def __call__(self, feats: List[Dict[str,Any]]):
        ids = [f.pop("id") for f in feats]
        texts = [f.pop("text") for f in feats]
        offsets = [f.pop("offsets") for f in feats]
        batch = self.tok.pad(feats, padding=True, pad_to_multiple_of=self.mult, return_tensors="pt")
        batch["id"] = ids
        batch["text"] = texts
        batch["offsets"] = offsets
        return batch

# ================ 模型 ================
#! ??
class Biaffine(nn.Module):
    '''
    双仿射（Biaffine）注意力层
    '''
    def __init__(self, in1, in2, out):
        super().__init__()
        self.U = nn.Parameter(torch.empty(out, in1, in2))
        # U 是一个三维可学习参数张量，形状为 [out, in1, in2]
            # 可理解为：为每个输出维度 k ∈ [0, out) 维护一个双线性矩阵 U_k ∈ ℝ^{in1 × in2}
        nn.init.xavier_uniform_(self.U) # 使用 Xavier 初始化
    def forward(self, H1, H2):  # [B,L,H1],[B,L,H2] -> [B,L,L,out]
        T = torch.einsum("blh,ohk->blok", H1, self.U)     # [B,L,out,H2]
        S = torch.einsum("blok, bmk -> blom", T, H2)      # [B,L,out,L]
        return S.permute(0,1,3,2)                         # [B,L,L,out]

class ASTEModel(nn.Module):
    def __init__(self, enc: AutoModel, n_cat: int, drop: float = 0.2, va_head_dim: int = 64):
        super().__init__()
        self.enc = enc # 预训练编码器
        H = enc.config.hidden_size #? 隐层维度（如 768/1024），后面所有头都会基于它。
        self.drop = nn.Dropout(drop)
        self.a_cls = nn.Linear(H, 3) # 两个独立的 token 分类头（每个 token → 3 类：O/B/I）。输出形状均为 [B, L, 3]
        self.o_cls = nn.Linear(H, 3)

        self.va_head = nn.Sequential( # 对 (token_i, token_j) 拼接后的表示（维度 2H）做回归，输出 2 维连续值 (V, A)
            nn.Linear(2*H, va_head_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(va_head_dim, 2)  # (V,A)
        )
        self.biaff = Biaffine(H, H, n_cat) # 双仿射打分器，把每个 (i, j) 的组合映到 n_cat 个类别分数（含 None 类 0）

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        # 调用预训练语言模型编码器，将输入的 token 序列（input_ids）通过多层 Transformer 编码器，输出上下文感知的 token 表示
        # out.last_hidden_state 形状 [B, L, H]
        H = self.drop(out.last_hidden_state)            # [B,L,H]
        B, Lh, _ = H.size()
        # dropout 后的 token 表示；B=batch，Lh=序列长度（与 L 等价，这里取名 Lh）。
        a_logits = self.a_cls(H)                        # [B,L,3]
        o_logits = self.o_cls(H)                        # [B,L,3]
        # 两套 BIO 头的未归一化 logits
        pair_scores = self.biaff(H, H)                  # [B,L,L,C] 双仿射对所有 (i, j) 组合打分，C = n_cat（类别数，含 None）
        _, Lr, Lc, C = pair_scores.shape                # Lr/Lc 分别是“行/列”的序列长度（此处与 Lh 相同）

        # VA（i,j）
        Hi = H.unsqueeze(2).expand(-1, -1, Lh, -1)      # [B,L,L,H]
        Hj = H.unsqueeze(1).expand(-1, Lh, -1, -1)      # [B,L,L,H]
        H_pair = torch.cat([Hi, Hj], dim=-1)     # [B,L,L,2H]
        # 构造笛卡尔积特征：把每个 i 的向量与每个 j 的向量拼接，得到每个 (i,j) 的 2H 表示。
        # expand 仅视图扩展，不复制内存（高效）
        va_scores = self.va_head(H_pair)                # [B,L,L,2]
        va_scores = torch.sigmoid(va_scores) * 8.0 + 1.0
        # 通过 MLP 输出 (V, A)，再用 sigmoid * 8 + 1 把范围约束到 [1, 9]，与标注区间一致。

        # mask
        m = attention_mask
        if m.dim()!=2:
            m = m.view(B,-1)
        # 保障 mask 为 [B, L]
        def fit_mask(m, L):
            # 对齐函数：若当前 mask 列数与 pair_scores 的 Lr/Lc 不一致
            if m.size(1) >= L:
                return m[:, :L] # 列数多 → 截断
            pad = torch.ones(B, L - m.size(1), dtype=m.dtype, device=m.device)
            return torch.cat([m, pad], dim=1) # 列数少 → 右侧补 1
        row_m = fit_mask(m, Lr).bool()
        col_m = fit_mask(m, Lc).bool()

        # 扩展row和col
        row4 = row_m.unsqueeze(2).unsqueeze(3)          # [B,L,1,1]
        col4 = col_m.unsqueeze(1).unsqueeze(3)          # [B,1,L,1]
        mask4 = row4 & col4                             # [B,L,L,1]
        # 构造二元 mask：只有当行位与列位都有效时，(i,j) 才有效
        # mask4[b,i,j,0] = True 表示第 b 条样本的第 i、j 个 token 都在真实段内
        pair_scores = pair_scores.masked_fill(~mask4.expand_as(pair_scores), -1e4)
        va_scores   = va_scores.masked_fill(~mask4.expand_as(va_scores),   0.0)
        # 对 无效 (i,j)：pair_scores 填充一个极小值 -1e4，避免 softmax 后引入伪概率；
        # va_scores 置 0（无意义处不参与监督/解码）。
        return a_logits, o_logits, pair_scores, va_scores

# ================ 优化器与调度器 ================
def _build_param_groups(model, lr_enc, lr_heads, wd):
    '''
    把模型参数按“编码器 vs 任务头、是否做权重衰减（weight decay）”分成 4 组，以便用 AdamW（或别的优化器）对不同子网施加不同学习率和是否衰减的策略

    model：你的 ASTEModel 实例，里面有 enc（预训练主干）和若干头（a_cls/o_cls/biaff/va_head/...）。
    lr_enc：encoder（model.enc.*）的学习率。
    lr_heads：heads（除了 enc. 开头的其它参数）学习率。
    wd：对需要衰减的参数组使用的 weight_decay 系数。
    '''
    def no_decay(n):
        # 名字里含 bias 或 LayerNorm.weight 的参数标记为“不做衰减（no decay）”
        return ("bias" in n) or ("LayerNorm.weight" in n)
    enc_decay, enc_nodecay, head_decay, head_nodecay = [], [], [], []
    # enc_decay：编码器里要衰减的参数   enc_nodecay：编码器里不衰减的参数
    # head_decay：头里要衰减的参数      head_nodecay：头里不衰减的参数
    for n,p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("enc."): # 根据名称前缀 判断参数的归属是属于编码器还是头:主干用小 LR、头用大 LR；
            (enc_nodecay if no_decay(n) else enc_decay).append(p)
        else:
            (head_nodecay if no_decay(n) else head_decay).append(p)
    # 返回4组param group 给优化器
    return [
        {"params": enc_decay,   "lr": lr_enc,   "weight_decay": wd},
        {"params": enc_nodecay, "lr": lr_enc,   "weight_decay": 0.0},
        {"params": head_decay,  "lr": lr_heads, "weight_decay": wd},
        {"params": head_nodecay,"lr": lr_heads, "weight_decay": 0.0},
    ]

def build_optim_sched(model, steps_total: int,lr_enc: float, lr_heads: float, wd: float,warmup_ratio: float, warmup_min_steps: int):
    '''
    构建 优化器和调度器

    model：包含编码器 enc.* 与各个任务头的模型。
    steps_total：训练总步数（optimizer 的更新次数）。通常是 len(train_loader) * epochs（若以后引入梯度累积，应该改为 len(train_loader) * epochs / accum_steps）。
    lr_enc / lr_heads：主干和头部的基础学习率（两组会被 scheduler 一起缩放，但基数不同）。
    wd：需要权重衰减的参数组的 weight_decay。
    warmup_ratio：线性 warmup 比例（0~1 之间的一个小数）。
    warmup_min_steps：warmup 的最小步数兜底，防止总步数太少时 warmup 几乎为 0 步。
    '''
    opt = torch.optim.AdamW(_build_param_groups(model, lr_enc, lr_heads, wd))
    # 用 AdamW 优化器，并把参数按四组喂进去（编码器/头 × 衰减/不衰减）。
    # 作用：主干用小 LR、头用大 LR；bias/LayerNorm.weight 不做衰减。这样做能在微调时更稳（主干不被破坏、头部快速适配）。
    # 此时opt是一个优化器实例,已经用 4 组参数（enc/heads × decay/no-decay）初始化好
    warmup_by_ratio = int(steps_total * max(0.0, min(0.2, warmup_ratio)))
    # 按比例计算warmup步数,但把比例缩放到了 [0, 0.2]
    warmup = max(warmup_min_steps, warmup_by_ratio) # 下界兜底
    warmup = min(warmup, max(1, steps_total // 3))  # 上界兜底
    sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=steps_total)
    # 构造 线性 warmup + 线性衰减 的学习率调度器
    # 第 0 ~ warmup-1 步：学习率从 0 线性升到 设定的基础 LR（每个 param group 的基础 LR 不同，但缩放曲线一致）。
    # 第 warmup ~ steps_total-1 步：线性下降到 0。
    return opt, sch # 返回优化器与调度器

# ================ 损失 ================
def loss_fn(a_logits, o_logits, pair_scores, va_scores,
            a_tags, o_tags, pair_ij, pair_cat, va_targets,
            lambda_pair: float = 0.8, gamma: float = 2.0, lambda_va: float = 0.5,
            alpha_none: float = 0.10, alpha_pos: float = 0.90):
    '''
    a_logits, o_logits: [B, L, 3]（3 对应 O/B/I）。

    pair_scores: [B, L, L, C]（C=类别数，含 None 类 0）。

    va_scores: [B, L, L, 2]（V、A），模型中已 sigmoid*8+1 → [1,9]。

    a_tags, o_tags: [B, L]（值在 {0,1,2}）。

    pair_ij: [B, 2]（每条样本监督的 A 起点、O 起点 token 下标，负值表示缺失时被 clamp 到 0）。

    pair_cat: [B]（配对的类别 id，0=“None”）。

    va_targets: [B, 2]（监督的 V 和 A，标注范围应在 [1,9]）。

    lambda_pair：配对多分类损失的权重。

    gamma：Focal Loss 的 γ，聚焦难样本（常用 2.0）。

    lambda_va：VA 回归损失的权重。

    alpha_none/alpha_pos：Focal Loss 的类别权重（None 类 vs 非 None 类），用来对抗“判 None 过多”的不平衡。
    你这里设为 None=0.10, Pos=0.90，是鼓励非 None、抑制“全判 None”的正确方向。
    '''

    B, L, _ = a_logits.size()
    # 取 batch 和序列长度，下面把 [B,L,3] 展平成 [B·L,3] 来算交叉熵。

    a_loss = F.cross_entropy(a_logits.reshape(B*L, -1), a_tags.view(-1), reduction="mean", label_smoothing=0.05)
    o_loss = F.cross_entropy(o_logits.reshape(B*L, -1), o_tags.view(-1), reduction="mean", label_smoothing=0.05)
    # BIO序列标注损失

    idx_b = torch.arange(B, device=pair_scores.device)
    # 作为batch 维的索引，等会与每个样本自己的 (i,j) 搭配做逐样本选取
    ii = torch.clamp(pair_ij[:, 0], min=0)
    jj = torch.clamp(pair_ij[:, 1], min=0)
    # pair_ij 形状是 [B, 2]，每条样本给出一个Aspect 起点 token 下标 i 和 Opinion 起点 token 下标 j
    # clamp(min=0) 把可能出现的 -1（找不到 span 的占位）截到 0，防止负索引崩溃
    #todo ii 和 jj 就是与 idx_b 一一对应的、来自 pair_ij 的每条样本的 (i, j) 下标（分别是该样本的 Aspect 起点 token 下标、Opinion 起点 token 下标）。
    #todo 三者按 batch 维度对齐，用来在 pair_scores 的四维张量里做逐样本取点索引。
    logits = pair_scores[idx_b, ii, jj, :]  # [B, C]
    # pair_scores 形状是 [B, L, L, C]
    # L, L 是所有 token 两两组合的位置（行是候选 aspect 位置，列是候选 opinion 位置）
    # C 是配对的类别数（包含 None 类）
    # 传入三个长度同为 B 的 1D 索引 idx_b、ii、jj，得到的就是逐样本取 (b, ii[b], jj[b], :) 那一条
    # 返回的 logits 形状为 [B, C]，表示每条样本在其监督的 (i,j) 配对位置上的类别打分向量

    logp = F.log_softmax(logits, dim=-1) # logits → logp/p：把 [B,C] 分类得分变成概率分布
    p    = logp.exp()
    y    = pair_cat # y 就是真实类别（含 None=0）
    pt   = p[torch.arange(B, device=p.device), y] # pt 是真类的预测概率

    alpha_vec = torch.full_like(pt, fill_value=alpha_pos)
    # 创建一个与 pt 形状、dtype、device 完全一致的张量，并把所有元素填成 alpha_pos
    alpha_vec = torch.where(y==0, torch.full_like(alpha_vec, alpha_none), alpha_vec)
    # 构造每条样本的 α：若该样本真类是 None，则用 alpha_none；否则用 alpha_pos。体现“少判 None、多关注正类”的不平衡校正
    #todo torch.where(cond, x, y)：在 cond 为 True 的位置取 x，为 False 的位置取 y（逐元素选择，不会原地修改，需要接收返回值）
    #? alpha_vec = 按真实类选的常数权重 为每个样本构造 Focal Loss 的 α 权重向量，让“None 类(0类)”与“非 None 类(正类)”使用不同的类别权重，从而缓解类别不平衡

    focal = alpha_vec * ((1.0 - pt) ** gamma) * F.nll_loss(logp, y, reduction="none") # focal焦点损失
    # F.nll_loss(logp, y, reduction="none"): 形状 [B] 的逐样本交叉熵，等于 -logp[b, y[b]]
    pair_loss = focal.mean()

    pred_va = va_scores[idx_b, ii, jj, :]                   # [B,2]  对每条样本，在它的监督配对位置 (i,j) 取出 (V,A) 的预测，得到 [B,2]
    va_targets = torch.clamp(va_targets, 1.0, 9.0)          # [B,2] 重新现指va_targets的范围
    va_loss = F.mse_loss(pred_va, va_targets) # 计算预测值与真实值之间的误差

    total = a_loss + o_loss + lambda_pair * pair_loss + lambda_va * va_loss
    return total, (a_loss.item(), o_loss.item(), pair_loss.item(), va_loss.item())
    # 返回 total（参与反传）和四项标量（.item() 取数便于日志打印）。

# ================ EMA（新增，最小实现） ================
# ================ EMA（完整：含 load_weights） ================
# ema_init 是在为**“用滑动平均权重评估/保存”**打基础：
# 它让你在训练过程中持续维护一份更平滑、更稳健的权重副本，常常带来更好、更稳定的验证与最终提交结果。
def ema_init(model):
    """
    初始化 EMA 字典为【全部参数】，避免解冻后出现缺键。
    键名和 model.named_parameters() 完全一致。
    """
    return {n: p.detach().clone() for n,p in model.named_parameters()}

@torch.no_grad()
def ema_update(model, ema_dict, decay=0.999):
    """
    仅对当前 requires_grad=True 的参数做指数滑动平均；
    如遇偶发缺键（热加载/重构），即时补齐。
    """
    for n,p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n not in ema_dict: # 若 ema_dict 没有这个参数名 则添加
            ema_dict[n] = p.detach().clone()
        ema_dict[n].mul_(decay).add_(p.detach(), alpha=1-decay) # 更新参数

@torch.no_grad()
def load_weights(model, state_dict):
    """
    将 state_dict（通常来自 EMA 字典）拷回到模型参数上。
    仅覆盖【模型中存在且需要训练】的参数，避免不必要的覆盖。
    """
    for n,p in model.named_parameters():
        if n in state_dict and p.requires_grad:
            p.copy_(state_dict[n])
            #? 把 EMA 权重拷回模型会改变参数，自然会改变准确率。在你的任务与代码流程中，这种改变通常是正向且稳定的；

def state_dict_from_ema(ema_dict):
    """
    返回 EMA 字典的深拷贝，用于保存到 checkpoint。
    """
    return {k: v.clone() for k,v in ema_dict.items()}


# ================ 训练/验证 ================
def train_one_epoch(model, loader, opt, sch, cfg: Config, ema_dict=None, ema_decay=0.999):
    model.train()
    scaler = GradScaler(enabled=cfg.use_amp and device.type=="cuda")
    # 在 PyTorch 中启用自动混合精度（Automatic Mixed Precision, AMP）
    total, n = 0.0, 0
    logs = [0.0,0.0,0.0,0.0]
    #todo total 累加 loss.item()*bs；logs[4] 内各项同理（a/o/pair/va 四分量）
    amp_ctx = torch.autocast("cuda", dtype=torch.float16) if (cfg.use_amp and device.type=="cuda") else nullcontext()

    for batch in loader: # 取 batch 并移动到设备
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        a_tags = batch["a_tags"].to(device)
        o_tags = batch["o_tags"].to(device)
        pair_ij= batch["pair_ij"].to(device)
        pair_cat=batch["pair_cat"].to(device)
        va_targets = batch["va_target"].to(device)  # [B,2]

        opt.zero_grad(set_to_none=True)
        with amp_ctx:
            a_logits, o_logits, pair_scores, va_scores = model(ids, att)
            loss, parts = loss_fn(
                a_logits, o_logits, pair_scores, va_scores,
                a_tags, o_tags, pair_ij, pair_cat, va_targets,
                lambda_pair=cfg.lambda_pair, gamma=cfg.gamma_focal, lambda_va=cfg.lambda_va,
                alpha_none=cfg.alpha_none, alpha_pos=cfg.alpha_pos
            )
            # loss：总损失 = BIO(a)+BIO(o)+ λ_pair·Focal(pair) + λ_va·MSE(VA)
            # parts：分量损失 (a_loss, o_loss, pair_loss, va_loss)
        if scaler.is_enabled(): # 反向传播与优化（兼容 AMP/非 AMP）
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        # —— EMA 更新（新增） ——
        if ema_dict is not None: # 用更新后的模型权重去更新 EMA 影子副本
            ema_update(model, ema_dict, decay=ema_decay)


        bs = ids.size(0) # 当前这个 batch 的样本数（batch size）
        total += loss.item()*bs
        n += bs
        for k in range(4):
            logs[k]+=parts[k]*bs
    return total/n, [x/n for x in logs]
    # total/n：本 epoch 的平均总损失
    # [x/n for x in logs]：四个子损失的平均值（a/o/pair/va）

@torch.no_grad()
def evaluate(model, loader, cfg: Config, ema_dict=None):
    # 验证时若提供 EMA，则临时切换到 EMA 权重
    backup = None
    if ema_dict is not None: # 如果传进来的是 EMA 字典，就用 EMA 权重做验证；否则用当前参数做验证。
        backup = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        # backup 备份当前模型参数
        load_weights(model, ema_dict)
        # 把 ema_dict 中的权重拷贝到模型的可训练参数上

    model.eval()
    total, n = 0.0, 0
    logs=[0.0,0.0,0.0,0.0]
    for batch in loader:
        ids = batch["input_ids"].to(device) # [B, L] token id
        att = batch["attention_mask"].to(device) # [B, L] attention_mask
        a_tags = batch["a_tags"].to(device)      # [B, L] BIO 标签
        o_tags = batch["o_tags"].to(device)
        pair_ij= batch["pair_ij"].to(device)     # [B, 2] 每条样本监督的 (A,O) 起点
        pair_cat=batch["pair_cat"].to(device)    # [B] 配对类别（含 None = 0）
        va_targets = batch["va_target"].to(device) # [B, 2] V/A 回归目标

        a_logits, o_logits, pair_scores, va_scores = model(ids, att)
        # a_logits/o_logits：[B, L, 3] BIO logits
        # pair_scores：[B, L, L, C] 配对分类 logits
        # va_scores：[B, L, L, 2] (V,A) 预测映射到 [1,9]
        loss, parts = loss_fn(
            # loss 总损失(加权组合各分支)
            # parts：一个四元组 (a_loss, o_loss, pair_loss, va_loss)，都是标量
            a_logits, o_logits, pair_scores, va_scores,
            a_tags, o_tags, pair_ij, pair_cat, va_targets,
            lambda_pair=cfg.lambda_pair, gamma=cfg.gamma_focal, lambda_va=cfg.lambda_va,
            alpha_none=cfg.alpha_none, alpha_pos=cfg.alpha_pos
        )
        bs = ids.size(0) # 当前 batch 的样本数（可能不是固定的 batch_size，比如最后一批）。
        total += loss.item()*bs # 把该 batch 的平均损失乘以样本数，累计成“总损失和”；
        n += bs
        for k in range(4):
            logs[k]+=parts[k]*bs # 同样对四个子损失做“加权求和”；

    # 还原权重
    if backup is not None:
        load_weights(model, backup)
        # 如果一开始传入了 ema_dict，我们前面把模型切成 EMA 权重进行验证，这里就要把原来的参数还原
    return total/n, [x/n for x in logs]
    # total / n：这一整个验证集上的总平均损失。
    # [x/n for x in logs]：四个分支的平均损失：[a_loss, o_loss, pair_loss, va_loss]。

# ===================== 推理解码 =====================
@torch.no_grad()
def predict_dev(model, dev_df, tok, cfg: Config, id2cat: Dict[int, str], ema_dict=None):
    """
    model: 已训练的模型。
    dev_df: 开发集数据，Pandas DataFrame，包含 "id" 和 "text" 列。
    tok: HuggingFace 的 tokenizer（如 RoBERTaTokenizer）。
    cfg: 配置对象，包含超参数（如 batch_size、max_len 等）。
    id2cat: 类别 ID 到类别名的映射字典（虽然函数中未直接使用，但可能为后续扩展预留）。
    ema_dict: 可选，指数移动平均（EMA）权重字典，用于更稳定的推理。
    仅解码侧增强：更严的 Opinion 起点阈值 + 片段文本过滤 + 配对阈值下限 0.30。
    """
    # —— 可移植的小工具（局部定义，避免污染全局命名空间） ——
    EN_STOP = {
        "a","an","the","and","or","but","if","so","for","to","of","in","on","at","by","with","as",
        "is","are","was","were","be","been","being","do","does","did","have","has","had",
        "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","its","our","their",
        "this","that","these","those","here","there","very","really","quite","somewhat","too","enough","well"
    }
    # 判断一个字符串是否是有效的 Aspect（方面词）
    def _is_valid_aspect(s: str) -> bool:
        t = (s or "").strip() # 去除首尾空格
        if len(t) < 2:
            return False
        low = t.lower()
        if low in {"i","it","this","that","the"}:  # 排除常见无实际语义的代词或限定词
            return False
        if all(not c.isalpha() for c in t):  # 若全为非字母字符（如标点）视为无效
            return False
        if t in {"-","–","—"}:  # 仅为破折号，也视为无效
            return False
        return True

    def _is_valid_opinion(s: str) -> bool: # 判断一个字符串是否是有效的 Opinion（观点词）
        t = (s or "").strip().lower()
        if len(t) < 2:
            return False
        if t in EN_STOP:   # 过滤功能词
            return False
        if sum(ch.isalpha() for ch in t) <= 1:        # 基本无字母
            return False
        return True

    # —— 若提供 EMA，则临时切换到 EMA 权重 ——
    backup = None
    if ema_dict is not None:
        backup = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        load_weights(model, ema_dict)

    # —— Dev dataset & collator（与原实现一致，仅键名保持 "offsets"） ——
    # 构建开发集DataLoader
    class _DevDS(Dataset):
        def __init__(self, df, tok, max_len):
            self.df = df.reset_index(drop=True)
            self.tok = tok
            self.max_len = max_len
        def __len__(self):
            return len(self.df)
        def __getitem__(self, i):# 仅对第 i 行文本进行 tokenization
            r = self.df.iloc[i]
            enc = self.tok(
                r["text"],
                truncation=True, # 启用截断
                max_length=min(self.max_len, cfg.max_decode_len),
                return_offsets_mapping=True
            )
            off = enc.pop("offset_mapping")
            enc["offset_mapping"] = off
            enc["id"] = str(r["id"])
            enc["text"] = r["text"]
            return enc

    # 自定义 collate 函数，用于将多个样本打包成 batch
    def _coll(feats):
        ids = [f.pop("id") for f in feats]
        texts = [f.pop("text") for f in feats]
        offsets = [f.pop("offset_mapping") for f in feats] # 先取出非张量字段
        batch = tok.pad(feats, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        # 再用 tokenizer 的 pad 方法对输入进行批处理填充
        batch["id"] = ids
        batch["text"] = texts
        batch["offsets"] = offsets # 最后将非张量字段加回batch字典
        return batch

    ds = _DevDS(dev_df, tok, cfg.max_len)
    num_workers = cfg.num_workers if device.type == "cuda" else 0
    pin = True if device.type == "cuda" else False
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin, collate_fn=_coll)

    # —— 解码阈值：Opinion 更严；配对阈值设置下限 0.30 ——
    tau_a = 0.18 # Aspect span 解码阈值（较低，允许更多候选）
    tau_o = 0.28 # Opinion span 解码阈值（更高，更严格）
    pair_thresh_main = max(cfg.pair_thresh, 0.30)# triplet 分类置信度阈值，至少为 0.30（防止过低阈值引入噪声）

    model.eval()
    lines = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device).bool()
        tt  = None  # RoBERTa 无 token_type_ids，设为 None

        a_logits, o_logits, pair_scores, va_scores = model(ids, att.to(ids.dtype), tt)
        # a_logits: Aspect 边界分类 logits（[B, L, 3]，BIO 标签）；
        # o_logits: Opinion 边界分类 logits；
        # pair_scores: 所有 token 对的 triplet 类别分数（[B, L, L, C]）；
        # va_scores: 对应的 Valence-Arousal（情感强度）回归值（[B, L, L, 2]）。
        probs = F.softmax(pair_scores, dim=-1)  # 对 pair_scores 做 softmax 得到概率 probs [B,L,L,C]
        B, L, _ = a_logits.size()

        for b in range(B): # 单样本处理循环
            text = batch["text"][b]
            offsets = batch["offsets"][b]
            mask_b = att[b].tolist() # attention mask（用于忽略 padding）

            # 1) A/O span 解码（B-起点法，带阈值）
            a_prob_b = a_logits[b].softmax(-1).unsqueeze(0)
            # a_logits[b]：取出 batch 中第 b 个样本的 Aspect logits，形状变为 [L, 3]
            # .softmax(-1)：在最后一个维度（即 3 个标签维度）上应用 softmax 函数，将 logits 转换为概率分布。结果形状仍是 [L, 3]，每个位置的三个值之和为 1。
            # .unsqueeze(0)：在最前面增加一个 batch 维度，形状变为 [1, L, 3]
            # 为什么加 batch 维？ 因为后续调用的 _decode_spans_B 函数设计为处理 batch 输入（即输入形状为 [B, L, 3]）。即使只处理一个样本，也要伪装成 batch size=1 的形式。
            # 结果：a_prob_b 是一个形状为 [1, L, 3] 的张量，表示第 b 个样本每个 token 属于 O/B/I 的概率。
            o_prob_b = o_logits[b].softmax(-1).unsqueeze(0)
            mask_tensor = torch.tensor(mask_b, device=a_prob_b.device).unsqueeze(0)
            # .unsqueeze(0)：同样增加 batch 维度，使其形状变为 [1, L]，以匹配 _decode_spans_B 对 mask 的输入要求（[B, L]）。
            # 结果：mask_tensor 是一个形状为 [1, L] 的布尔或整型张量，指示哪些 token 是有效的（非 padding）

            A_spans = _decode_spans_B(a_prob_b, mask_tensor, tau=tau_a)[0]
            O_spans = _decode_spans_B(o_prob_b, mask_tensor, tau=tau_o)[0]
            # 调用_decode_spans_B,寻找所有满足条件的连续 "B-I..." 片段
            # 结果：A_spans 是一个 Python 列表，每个元素是一个二元组 (start_token_index, end_token_index)，代表一个预测出的 Aspect 片段（两端都包含）
            A_spans = [(i,j) for (i,j) in A_spans if (j-i+1) >= 1]
            O_spans = [(i,j) for (i,j) in O_spans if (j-i+1) >= 1] # 过滤掉长度小于 1 的 span（片段）

            # 2) 候选配对：span×span 最大化 + 类阈值
            cand = []
            for (ai, aj) in A_spans:
                for (oi, oj) in O_spans:
                    if abs(ai - oi) > cfg.pair_window: # 若 token 距离超过 pair_window（如 100），跳过（局部性假设）
                        continue
                    block = probs[b, ai:(aj+1), oi:(oj+1), :]   # [lenA,lenO,C]
                    #? block：从完整概率张量中切出一个子立方体，覆盖整个 Aspect span（从 ai 到 aj）和 Opinion span（从 oi 到 oj）
                    # lenA:代表 Aspect span 中的每一个 token（方面词内部的 token 位置）
                    # lenO:代表 Opinion span 中的每一个 token（观点词内部的 token 位置）
                    # C:代表情感类别（sentiment category）的概率分布
                    flat  = block.reshape(-1, block.size(-1))   # [lenA*lenO,C]
                    #? flat：将前两维展平，变成 [lenA * lenO, C]，每一行代表一个具体的 (aspect_token, opinion_token) 配对的概率分布
                    # 每类的最大概率，选非 None 类
                    p_per_class = flat.max(dim=0).values # 在所有 token 对上，对每个类别 c 取最大概率 → 得到长度为 C 的向量
                    cat_id = int(torch.argmax(p_per_class).item()) # 选出非 None 类中概率最高的类别？⚠️ 注意：这里包含类别 0（None）！
                    p_max  = float(p_per_class.max().item()) # 所有类别中的最大概率值
                    if cat_id == 0 or p_max < pair_thresh_main:
                        continue

                    # 在该 span×span 内找总体最大处的 (vi,vj) 来取 VA
                    max_idx = torch.argmax(flat) # 在 flat（形状 [N, C]）中，找出哪个位置 (n, c) 的概率最大
                    lenA, lenO, Cc = block.size(0), block.size(1), block.size(2) # 各个维度的长度
                    ri = int((max_idx // Cc) // lenO) # 在 Aspect span 内的相对偏移（0 ≤ ri < lenA）
                    rj = int((max_idx // Cc) %  lenO) # 在 Opinion span 内的相对偏移（0 ≤ rj < lenO）
                    vi, vj = ai + ri, oi + rj # 得到原始序列中具体的 token 索引，用于查询 VA 值

                    # 获取并裁剪 VA 值
                    v_raw = float(va_scores[b, vi, vj, 0].item())
                    a_raw = float(va_scores[b, vi, vj, 1].item())
                    v_fin = round(max(1.0, min(9.0, v_raw)), 2)
                    a_fin = round(max(1.0, min(9.0, a_raw)), 2)

                    cand.append((ai, aj, oi, oj, cat_id, p_max, v_fin, a_fin, vi, vj))
                    # ai, aj	Aspect 的 token 区间
                    # oi, oj	Opinion 的 token 区间
                    # cat_id	情感类别（1=POS, 2=NEG, 3=NEU）
                    # p_max	该配对的最大置信度
                    # v_fin, a_fin	裁剪后的 Valence 和 Arousal
                    # vi, vj	用于预测 VA 的具体 token 位置（可用于调试或后处理

            # 3) 互为最近（允许轻微 margin）
            MARGIN = 0.02
            bestO = _dd(lambda: (-1, -1.0))  # ai -> (idx, p)
            # 对于 Aspect 起点 ai，记录它在 cand 中置信度最高的候选索引 idx 和对应的概率 p
            bestA = _dd(lambda: (-1, -1.0))  # oi -> (idx, p)
            # 对于 Opinion 起点 oi，记录它在 cand 中置信度最高的候选索引 idx 和对应的概率 p
            for idx, (ai, aj, oi, oj, cat_id, p, v_fin, a_fin, vi, vj) in enumerate(cand):
                if p > bestO[ai][1]:
                    bestO[ai] = (idx, p)
                if p > bestA[oi][1]:
                    bestA[oi] = (idx, p)
            keep_idx = set()
            for ai, (io, p) in bestO.items():
                oi = cand[io][2] # 从候选 io 中取出对应的 opinion 起点 oi
                j_best, p_b = bestA[oi] # # 查找这个 oi 的最佳配对是谁（j_best 是 cand 中的索引）
                if j_best == io or abs(p_b - p) <= MARGIN:
                    # j_best == io：
                        # 表示：ai 认为 io 是它最好的配对；
                        # 同时，oi（即 cand[io] 中的 opinion 起点）也认为 io 是它最好的配对。
                    # abs(p_b - p) <= MARGIN
                        # 即使 j_best != io（比如 oi 的最佳配对是另一个候选 j_best），
                        # 但如果 oi 对 j_best 的置信度 p_b 和对 io 的置信度 p 相差不超过 0.02，
                        # 则认为这两个配对“几乎一样好”，可以接受 io 作为有效配对。
                        # ✅ 近似互为最佳，保留。
                    keep_idx.add(io)

            # 4) 回填 + 文本过滤 + 去重
            trips = []
            seen = set()
            for k in sorted(keep_idx, key=lambda i: -cand[i][5])[:cfg.topk_pairs]:
            # 首先根据每个候选的置信度 p（即 cand[i][5]）对 keep_idx 进行降序排序，然后只取前 cfg.topk_pairs 个最高的配对进行处理。
            # 这里 cfg.topk_pairs 应该是一个配置项，指定了希望保留的最高置信度配对的数量。
                ai, aj, oi, oj, cat_id, p, v_fin, a_fin, vi, vj = cand[k]
                a_st = offsets[ai][0]
                a_ed = offsets[aj][1]
                o_st = offsets[oi][0]
                o_ed = offsets[oj][1]
                # 使用 offsets 数组来获取 Aspect (asp) 和 Opinion (opn) 在原始文本中的起始和结束位置
                asp = text[a_st:a_ed].strip()
                opn = text[o_st:o_ed].strip() # 提取在原始文本中对应的Aspect和Opinion字符串

                # —— 新增：过滤无效 Aspect/Opinion ——
                if not _is_valid_aspect(asp):
                    continue
                if not _is_valid_opinion(opn):
                    continue

                key = (asp, opn)
                if key in seen: # 如果该组合已经出现过（key in seen），则跳过，避免重复添加
                    continue
                seen.add(key)
                trips.append({"Aspect": asp, "Opinion": opn, "VA": f"{v_fin:.2f}#{a_fin:.2f}"})
                # 构建最终三元组并加入结果列表

            # 5) 兜底分支（也加上过滤）
            if len(trips) == 0: # 当没有产生任何三元组时
                pair_thresh_fb = max(0.20, cfg.pair_thresh - 0.08) #  放宽配对置信度阈值
                # 以下为重复执行第2,3,4步操作
                margin_fb = 0.05 # 放宽容差范围（MARGIN）
                keep_idx_fb = set()
                bestO = _dd(lambda: (-1, -1.0))
                bestA = _dd(lambda: (-1, -1.0))
                cand_fb = []
                for (ai, aj) in A_spans:
                    for (oi, oj) in O_spans:
                        if abs(ai - oi) > max(cfg.pair_window, 200):
                            continue
                        block = probs[b, ai:(aj+1), oi:(oj+1), :]
                        flat  = block.reshape(-1, block.size(-1))
                        p_per_class = flat.max(dim=0).values
                        cat_id = int(torch.argmax(p_per_class).item())
                        p_max  = float(p_per_class.max().item())
                        if cat_id == 0 or p_max < pair_thresh_fb:
                            continue
                        v_raw = float(va_scores[b, ai, oi, 0].item())
                        a_raw = float(va_scores[b, ai, oi, 1].item())
                        v_fin = round(max(1.0, min(9.0, v_raw)), 2)
                        a_fin = round(max(1.0, min(9.0, a_raw)), 2)
                        cand_fb.append((ai, aj, oi, oj, cat_id, p_max, v_fin, a_fin))
                for idx, (ai, aj, oi, oj, cat_id, p, v_fin, a_fin) in enumerate(cand_fb):
                    if p > bestO[ai][1]:
                        bestO[ai] = (idx, p)
                    if p > bestA[oi][1]:
                        bestA[oi] = (idx, p)
                for ai, (io, p) in bestO.items():
                    oi = cand_fb[io][2] if len(cand_fb)>0 else -1
                    if oi != -1:
                        j_best, p_b = bestA[oi]
                        if j_best == io or abs(p_b - p) <= margin_fb:
                            keep_idx_fb.add(io)
                if len(keep_idx_fb) > 0:
                    k = sorted(keep_idx_fb, key=lambda i: -cand_fb[i][5])[0]
                    ai, aj, oi, oj, cat_id, p, v_fin, a_fin = cand_fb[k]
                    a_st = offsets[ai][0]
                    a_ed = offsets[aj][1]
                    o_st = offsets[oi][0]
                    o_ed = offsets[oj][1]
                    asp = text[a_st:a_ed].strip()
                    opn = text[o_st:o_ed].strip()
                    if _is_valid_aspect(asp) and _is_valid_opinion(opn):
                        trips.append({"Aspect": asp, "Opinion": opn, "VA": f"{v_fin: .2f}#{a_fin:.2f}"})

            # 6) 兜底兜底：极高置信度的单 token（也加过滤）
            if len(trips) == 0:
                P = probs[b] # # shape: [L, L, C]
                P_nonzero = P[..., 1:] # # shape: [L, L, C-1]
                # [..., 1:] 切片跳过第 0 类（"None"），只保留 POS/NEG/NEU 的概率
                # 这样后续找最大值时，不会被高概率的 "None" 干扰
                p_max_val = float(P_nonzero.max().item()) #P_nonzero.max() 找出所有元素中的最大值
                if p_max_val >= 0.65:
                    idx = torch.argmax(P_nonzero) # 找到其中最大值的索引（从 0 开始计数）。
                    C_ = P_nonzero.size(-1) # 情感类别数
                    L_ = P_nonzero.size(0) # 序列长度
                    ijc = np.unravel_index(int(idx), (L_, L_, C_))
                    # np.unravel_index 将扁平化的一维索引 idx 还原为 (i, j, c) 三维坐标
                    i_best, j_best = int(ijc[0]), int(ijc[1]) # 获得Aspect token 的位置（在序列中的索引）
                    ai = max(0, min(i_best, len(offsets)-1))
                    oi = max(0, min(j_best, len(offsets)-1)) # 防止越界
                    a_st = offsets[ai][0]; a_ed = offsets[ai][1]
                    o_st = offsets[oi][0]; o_ed = offsets[oi][1]
                    asp = text[a_st:a_ed].strip()
                    opn = text[o_st:o_ed].strip() # 获得字符范围后,提取字符串
                    v_raw = float(va_scores[b, ai, oi, 0].item())
                    a_raw = float(va_scores[b, ai, oi, 1].item())
                    # 从 va_scores 中取出对应 token 对 (ai, oi) 的 Valence 和 Arousal
                    v_fin = round(max(1.0, min(9.0, v_raw)), 2)
                    a_fin = round(max(1.0, min(9.0, a_raw)), 2)
                    # 裁剪到合法范围 [1.0, 9.0] 并保留两位小数
                    if _is_valid_aspect(asp) and _is_valid_opinion(opn):
                        trips.append({"Aspect": asp, "Opinion": opn, "VA": f"{v_fin:.2f}#{a_fin:.2f}"})

            lines.append({"ID": str(batch["id"][b]), "Triplet": trips})
            # 将第 b 个样本的预测三元组列表 trips 与对应的样本 ID 绑定，构造成一个字典，并添加到最终输出列表 lines 中
    if backup is not None: # backup 是在函数开头对原始模型权重的备份
        load_weights(model, backup) #恢复模型权重
    return lines

# 保存最终提交的json文件
def save_jsonl(objs, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for o in objs: w.write(json.dumps(o, ensure_ascii=False)+"\n")

# ================ 主流程 ================
def main():
    set_verbosity_error() # 将日志级别设为仅显示错误（ERROR），抑制警告和信息性输出，使控制台更干净，常用于生产或提交环境
    cfg = Config()
    set_seed(cfg.seed)

    # 模型与分词器
    local_dir = (Path(__file__).resolve().parents[2] / cfg.local_model_dir).resolve()
    # 使用 pathlib.Path 构建本地模型目录的绝对路径：
        # Path(__file__)：获取当前脚本的路径；
        # .resolve()：解析为绝对路径（消除 .. 和符号链接）；
        # .parents[2]：向上回溯两级目录（例如：project/src/utils/script.py → project/）；
        # / cfg.local_model_dir：拼接配置中指定的模型子目录（如 "roberta-base"）；
        # 最后再 .resolve() 确保结果是规范化的绝对路径。
    assert local_dir.exists(), f"本地模型目录不存在：{local_dir}"
    tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True, local_files_only=True)
    enc = AutoModel.from_pretrained(local_dir, local_files_only=True).to(device)
    # 使用 Hugging Face 的 AutoTokenizer 从本地目录加载分词器和基础编码器模型

    # 数据路径
    data_dir = output_path("output", "track_a", "subtask_2")
    train_path = (data_dir / "train_pairs.parquet")
    dev_path   = (data_dir / "dev_pairs.parquet")
    if not train_path.exists():
        train_path = (data_dir / "eng" / "subtask2_processed" / "train_subtask2.parquet")
        dev_path   = (data_dir / "eng" / "subtask2_processed" / "dev_subtask2.parquet")

    train_df = pd.read_parquet(train_path)
    dev_df   = pd.read_parquet(dev_path)

    cat2id = build_label_map(train_df["category"].astype(str).tolist()) #构建类别到整数 ID 的映射字典
    id2cat = {v:k for k,v in cat2id.items()} # 构建反向映射：从类别 ID 到类别名称的字典，便于后续预测结果的可读性展示或评估

    if "id" in train_df.columns:
        gss = GroupShuffleSplit(test_size=0.1, random_state=cfg.seed)
        # 创建一个 GroupShuffleSplit 分割器（来自 sklearn.model_selection）：
            # test_size=0.1：将 10% 的组（groups） 划分为验证集；
            # random_state=cfg.seed：固定随机种子，保证划分可复现。
        tr_idx, va_idx = next(gss.split(train_df, groups=train_df["id"]))
        # 执行一次划分：
            # gss.split(...) 返回一个生成器；
            # next(...) 获取第一个（也是唯一一个）划分结果；
            # 返回两个索引数组：tr_idx（训练集行索引）、va_idx（验证集行索引）；
            # groups=train_df["id"]：指定分组依据
        tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        # 根据索引切片，得到无数据泄露的训练子集 tr_df 和验证子集 va_df
    else:
        tr_df, va_df = train_test_split(train_df, test_size=0.1, random_state=cfg.seed)

    tr_ds = ASTETrainDataset(tr_df, tok, cfg.max_len, cat2id)
    va_ds = ASTETrainDataset(va_df, tok, cfg.max_len, cat2id)
    # 原始 DataFrame 转换为模型可训练的样本格式。
    n_workers = cfg.num_workers if device.type=="cuda" else 0
    pin = True if device.type=="cuda" else False
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=n_workers, pin_memory=pin,
                           collate_fn=TrainCollator(tok,8))
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=n_workers, pin_memory=pin,
                           collate_fn=TrainCollator(tok,8))
    # 创建数据加载器

    model = ASTEModel(enc, n_cat=len(cat2id), va_head_dim=cfg.va_head_dim).to(device)

    # 先冻结 encoder
    for p in model.enc.parameters():
        p.requires_grad=False
    steps_total = max(1, len(tr_loader)*cfg.epochs)
    opt, sch = build_optim_sched(
        model, steps_total,
        lr_enc=cfg.lr_encoder, lr_heads=cfg.lr_heads, wd=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio, warmup_min_steps=cfg.warmup_min_steps
    )

    # —— 初始化 EMA（新增） ——
    ema_dict = ema_init(model)
    ema_decay = 0.999

    best, bad, best_path = 1e18, 0, "aste_pa_best.pt"
    best_state_from_ema = None

    for ep in range(cfg.epochs):
        if ep == cfg.freeze_epochs: # 解冻
            for p in model.enc.parameters():
                p.requires_grad=True
            steps_left = max(1, len(tr_loader)*(cfg.epochs-ep))# 解冻后的剩余步数
            opt, sch = build_optim_sched(
                model, steps_left,
                lr_enc=cfg.enc_lr_after_unfreeze, lr_heads=cfg.lr_heads, wd=cfg.weight_decay,
                warmup_ratio=cfg.warmup_ratio, warmup_min_steps=cfg.warmup_min_steps
            )

        tl,(al,ol,pl,vl) = train_one_epoch(model, tr_loader, opt, sch, cfg, ema_dict=ema_dict, ema_decay=ema_decay)
        # —— 用 EMA 权重做验证（新增） ——
        vl_eval,(val_a,val_o,val_p,val_v) = evaluate(model, va_loader, cfg, ema_dict=ema_dict)
        print(f"Epoch {ep+1:02d} | train={tl:.4f} (a:{al:.3f} o:{ol:.3f} p:{pl:.3f} va:{vl:.3f}) "
              f"|| val={vl_eval:.4f} (a:{val_a:.3f} o:{val_o:.3f} p:{val_p:.3f} va:{val_v:.3f})")
        # 保存最佳模型
        if vl_eval < best:
            best = vl_eval; bad = 0
            best_state_from_ema = state_dict_from_ema(ema_dict)  # 保存 EMA 权重（新增）
            torch.save({"state_dict": best_state_from_ema, "cat2id": cat2id}, best_path)
            print(f"[BEST] saved (EMA) -> {best_path}")
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[EARLY STOP] no improvement for {cfg.patience} epochs."); break

    # 推理：加载 EMA 最优
    try:
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(best_path, map_location="cpu")

    enc2 = AutoModel.from_pretrained(local_dir, local_files_only=True).to(device)
    model = ASTEModel(enc2, n_cat=len(ckpt["cat2id"]), va_head_dim=cfg.va_head_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])

    preds = predict_dev(model, dev_df, tok, cfg, {}, ema_dict=None)  # ckpt 已是 EMA，无需再传
    out_file = output_path("submit","task2","pred_dev.jsonl")
    save_jsonl(preds, out_file)
    print("Dev 提交文件 ->", out_file)

if __name__ == "__main__":
    main()