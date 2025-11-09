import os, math, json
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np, pandas as pd
from collections import defaultdict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers.utils.logging import set_verbosity_error
from torch.amp import GradScaler

from sklearn.model_selection import GroupShuffleSplit, train_test_split

# 项目里的路径工具
from src.utils.paths import output_path

# ================= 配置 =================
@dataclass
class Config:
    # —— 模型/数据基础 ——
    local_model_dir: str = "models/roberta-base"  # 本地权重路径（离线加载）；换模型时指到对应目录
    max_len: int = 224                            # tokenizer 截断长度；长句可提到 256/320，显存↑ 训练慢
    batch_size: int = 12                          # 训练批大小；显存不够可减小并配合梯度累积
    epochs: int = 16                              # 训练轮数；配合 early stop，一般 10–20 足够
    seed: int = 42                                # 随机种子；复现用

    # —— 学习率与优化 ——
    lr_encoder: float = 1e-5                      # 编码器（RoBERTa/BERT）学习率；解冻后也会用到
    lr_heads: float   = 8e-5                      # 任务头（BIO/配对/VA）的学习率；过大易抖动、过小收敛慢
    weight_decay: float = 0.02                    # L2 权重衰减（正则）；0.01–0.05 常用，过大欠拟合
    warmup_ratio: float = 0.10                    # 线性 warmup 比例；小数据/大 LR 建议 0.06–0.1
    warmup_min_steps: int = 100                   # warmup 的最小步数下限；总步数太少时的兜底
    freeze_epochs: int = 4                        # 前几轮只训头部（冻结 encoder）；可提升早期稳定与召回
    enc_lr_after_unfreeze: float = 4.5e-6           # 解冻后 encoder 的 LR；越小越把学习量留给任务头
    patience: int = 4                             # 验证无提升的容忍轮数（早停）；曲线抖动大可增至 8–10

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
    alpha_none: float = 0.15                      # None 类的权重（越小→越不偏向判 None，召回↑）
    alpha_pos:  float = 0.85                      # 非 None 类权重（与上相对）；两者一般和≈1.0


# ================ 随机种子/设备 ================
def set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
torch.set_float32_matmul_precision("high")

# ================ BIO 标签 ================
BIO_LABELS = ["O","B","I"]
BIO2ID = {t:i for i,t in enumerate(BIO_LABELS)}

# ================ 工具：字符→token 对齐 ================
def find_span(text: str, surface: str) -> Optional[Tuple[int,int]]:
    if not surface or surface == "NULL": return None
    st = text.find(surface)
    if st < 0: return None
    return st, st + len(surface)

def char_to_token_span(offsets: List[Tuple[int,int]], ch_st: int, ch_ed: int) -> Optional[Tuple[int,int]]:
    i = j = None
    for tid,(s,e) in enumerate(offsets):
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
    tag = prob.argmax(-1)  # [B,L]
    outs = []
    for i in range(tag.size(0)):
        cur=[]
        if not torch.is_floating_point(mask[i]):
            L = int(mask[i].sum().item())
        else:
            L = int((mask[i] > 0.5).sum().item())
        t = 0
        while t < L:
            if tag[i,t].item()==BIO2ID["B"] and prob[i,t,BIO2ID["B"]].item()>=tau:
                st=t; ed=t
                t+=1
                while t<L and tag[i,t].item()==BIO2ID["I"]:
                    ed=t; t+=1
                cur.append((st,ed))
            else:
                t+=1
        outs.append(cur)
    return outs

WEAK_OPINIONS = {
    "very","really","finally","quite","somewhat","kinda","sorta","rather","so","too","enough","extremely","basically"
}
def _is_weak_opinion(opn: str) -> bool:
    t = (opn or "").strip().lower()
    if t in WEAK_OPINIONS:
        return True
    letters = sum(ch.isalpha() for ch in t)
    if letters <= 1:
        return True
    if letters < 3 and len(t) <= 3:
        return True
    return False

# ================ 数据集 ================
class TrainEx:
    __slots__ = ("text","a_span","o_span","cat","id","va_v","va_a")
    def __init__(self, text:str, a_span:Tuple[int,int], o_span:Tuple[int,int],
                 cat:str, rid:str, va_v: float, va_a: float):
        self.text = text; self.a_span = a_span; self.o_span = o_span
        self.cat = cat; self.id = rid
        self.va_v = va_v; self.va_a = va_a

def build_label_map(cats: List[str]) -> Dict[str,int]:
    uniq = sorted({c for c in cats if c and c!="NULL"})
    id_map = {"None":0}
    for i,c in enumerate(uniq, start=1):
        id_map[c]=i
    return id_map

class ASTETrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tok: AutoTokenizer, max_len: int, cat2id: Dict[str,int]):
        self.tok = tok; self.max_len = max_len; self.cat2id = cat2id
        rows: List[TrainEx] = []
        for _,r in df.iterrows():
            text = str(r["text"])
            a = str(r["aspect"]); o = str(r["opinion"]); c = str(r["category"])
            rid = str(r["id"])
            va_raw = str(r.get("VA", "5.0#5.0"))
            try:
                v_str, a_str = va_raw.split("#")
                va_v, va_a = float(v_str), float(a_str)
            except Exception:
                va_v, va_a = 5.0, 5.0

            a_span = find_span(text, a); o_span = find_span(text, o)
            if a_span is None or o_span is None:
                continue
            rows.append(TrainEx(text, a_span, o_span, c, rid, va_v, va_a))
        self.rows = rows

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        ex = self.rows[i]
        enc = self.tok(ex.text, truncation=True, max_length=self.max_len, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        a_tok = char_to_token_span(offsets, *ex.a_span)
        o_tok = char_to_token_span(offsets, *ex.o_span)
        L = len(enc["input_ids"])
        a_tags = [BIO2ID["O"]]*L
        o_tags = [BIO2ID["O"]]*L
        if a_tok is not None:
            ai,aj = a_tok
            a_tags[ai] = BIO2ID["B"]
            for t in range(ai+1,aj+1): a_tags[t]=BIO2ID["I"]
        else:
            ai = -1
        if o_tok is not None:
            oi,oj = o_tok
            o_tags[oi] = BIO2ID["B"]
            for t in range(oi+1,oj+1): o_tags[t]=BIO2ID["I"]
        else:
            oi = -1

        pair = (max(ai,0), max(oi,0))
        cat = self.cat2id.get(ex.cat, 0)

        enc["a_tags"]   = torch.tensor(a_tags, dtype=torch.long)
        enc["o_tags"]   = torch.tensor(o_tags, dtype=torch.long)
        enc["pair_ij"]  = torch.tensor(pair, dtype=torch.long)
        enc["pair_cat"] = torch.tensor(cat, dtype=torch.long)
        enc["va_target"] = torch.tensor([ex.va_v, ex.va_a], dtype=torch.float)  # [2]
        return enc

class ASTEDevDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tok: AutoTokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tok = tok; self.max_len = max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(r["text"], truncation=True, max_length=min(self.max_len, Config.max_len),
                       return_offsets_mapping=True)
        enc["offsets"] = enc.pop("offset_mapping")
        enc["id"] = str(r["id"])
        enc["text"] = r["text"]
        return enc

# ================ Collator ================
class TrainCollator:
    def __init__(self, tok, pad_to_multiple_of=8):
        self.tok = tok; self.mult = pad_to_multiple_of
    def __call__(self, feats: List[Dict[str,Any]]):
        a_tags = [f.pop("a_tags") for f in feats]
        o_tags = [f.pop("o_tags") for f in feats]
        pair_ij = torch.stack([f.pop("pair_ij") for f in feats],0)
        pair_cat= torch.stack([f.pop("pair_cat") for f in feats],0)
        va_targets = torch.stack([f.pop("va_target") for f in feats],0)  # [B,2]
        batch = self.tok.pad(feats, padding=True, pad_to_multiple_of=self.mult, return_tensors="pt")
        maxL = batch["input_ids"].size(1)
        def pad1d(arr, fill=0):
            out = torch.full((len(arr), maxL), fill, dtype=torch.long)
            for i, t in enumerate(arr):
                L = min(len(t), maxL)
                out[i, :L] = t[:L]
            return out
        batch["a_tags"] = pad1d(a_tags, fill=BIO2ID["O"])
        batch["o_tags"] = pad1d(o_tags, fill=BIO2ID["O"])
        batch["pair_ij"] = pair_ij
        batch["pair_cat"]= pair_cat
        batch["va_target"] = va_targets
        return batch

class DevCollator:
    def __init__(self, tok, pad_to_multiple_of=8):
        self.tok = tok; self.mult = pad_to_multiple_of
    def __call__(self, feats: List[Dict[str,Any]]):
        ids = [f.pop("id") for f in feats]
        texts = [f.pop("text") for f in feats]
        offsets = [f.pop("offsets") for f in feats]
        batch = self.tok.pad(feats, padding=True, pad_to_multiple_of=self.mult, return_tensors="pt")
        batch["id"] = ids; batch["text"] = texts; batch["offsets"] = offsets
        return batch

# ================ 模型 ================
class Biaffine(nn.Module):
    def __init__(self, in1, in2, out):
        super().__init__()
        self.U = nn.Parameter(torch.empty(out, in1, in2))
        nn.init.xavier_uniform_(self.U)
    def forward(self, H1, H2):  # [B,L,H1],[B,L,H2] -> [B,L,L,out]
        T = torch.einsum("blh,ohk->blok", H1, self.U)     # [B,L,out,H2]
        S = torch.einsum("blok, bmk -> blom", T, H2)      # [B,L,out,L]
        return S.permute(0,1,3,2)                         # [B,L,L,out]

class ASTEModel(nn.Module):
    def __init__(self, enc: AutoModel, n_cat: int, drop: float = 0.15, va_head_dim: int = 64):
        super().__init__()
        self.enc = enc
        H = enc.config.hidden_size
        self.drop = nn.Dropout(drop)
        self.a_cls = nn.Linear(H, 3)
        self.o_cls = nn.Linear(H, 3)

        self.va_head = nn.Sequential(
            nn.Linear(2*H, va_head_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(va_head_dim, 2)  # (V,A)
        )
        self.biaff = Biaffine(H, H, n_cat)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        H = self.drop(out.last_hidden_state)            # [B,L,H]
        B, Lh, _ = H.size()

        a_logits = self.a_cls(H)                        # [B,L,3]
        o_logits = self.o_cls(H)                        # [B,L,3]

        pair_scores = self.biaff(H, H)                  # [B,L,L,C]
        _, Lr, Lc, C = pair_scores.shape

        # VA（i,j）
        Hi = H.unsqueeze(2).expand(-1, -1, Lh, -1)      # [B,L,L,H]
        Hj = H.unsqueeze(1).expand(-1, Lh, -1, -1)      # [B,L,L,H]
        H_pair = torch.cat([Hi, Hj], dim=-1)            # [B,L,L,2H]
        va_scores = self.va_head(H_pair)                # [B,L,L,2]
        va_scores = torch.sigmoid(va_scores) * 8.0 + 1.0

        # mask
        m = attention_mask
        if m.dim()!=2: m = m.view(B,-1)
        def fit_mask(m, L):
            if m.size(1) >= L: return m[:, :L]
            pad = torch.ones(B, L - m.size(1), dtype=m.dtype, device=m.device)
            return torch.cat([m, pad], dim=1)
        row_m = fit_mask(m, Lr).bool()
        col_m = fit_mask(m, Lc).bool()
        row4 = row_m.unsqueeze(2).unsqueeze(3)          # [B,L,1,1]
        col4 = col_m.unsqueeze(1).unsqueeze(3)          # [B,1,L,1]
        mask4 = row4 & col4                             # [B,L,L,1]

        pair_scores = pair_scores.masked_fill(~mask4.expand_as(pair_scores), -1e4)
        va_scores   = va_scores.masked_fill(~mask4.expand_as(va_scores),   0.0)
        return a_logits, o_logits, pair_scores, va_scores

# ================ 优化器与调度器 ================
def _build_param_groups(model, lr_enc, lr_heads, wd):
    def no_decay(n): return ("bias" in n) or ("LayerNorm.weight" in n)
    enc_decay, enc_nodecay, head_decay, head_nodecay = [], [], [], []
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("enc."):
            (enc_nodecay if no_decay(n) else enc_decay).append(p)
        else:
            (head_nodecay if no_decay(n) else head_decay).append(p)
    return [
        {"params": enc_decay,   "lr": lr_enc,   "weight_decay": wd},
        {"params": enc_nodecay, "lr": lr_enc,   "weight_decay": 0.0},
        {"params": head_decay,  "lr": lr_heads, "weight_decay": wd},
        {"params": head_nodecay,"lr": lr_heads, "weight_decay": 0.0},
    ]

def build_optim_sched(model, steps_total: int,
                      lr_enc: float, lr_heads: float, wd: float,
                      warmup_ratio: float, warmup_min_steps: int):
    opt = torch.optim.AdamW(_build_param_groups(model, lr_enc, lr_heads, wd))
    warmup_by_ratio = int(steps_total * max(0.0, min(0.2, warmup_ratio)))
    warmup = max(warmup_min_steps, warmup_by_ratio)
    warmup = min(warmup, max(1, steps_total // 3))
    sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=steps_total)
    return opt, sch

# ================ 损失 ================
def loss_fn(a_logits, o_logits, pair_scores, va_scores,
            a_tags, o_tags, pair_ij, pair_cat, va_targets,
            lambda_pair: float = 0.8, gamma: float = 2.0, lambda_va: float = 0.5,
            alpha_none: float = 0.10, alpha_pos: float = 0.90):
    B, L, _ = a_logits.size()

    a_loss = F.cross_entropy(a_logits.reshape(B*L, -1), a_tags.view(-1), reduction="mean", label_smoothing=0.05)
    o_loss = F.cross_entropy(o_logits.reshape(B*L, -1), o_tags.view(-1), reduction="mean", label_smoothing=0.05)

    idx_b = torch.arange(B, device=pair_scores.device)
    ii = torch.clamp(pair_ij[:, 0], min=0)
    jj = torch.clamp(pair_ij[:, 1], min=0)
    logits = pair_scores[idx_b, ii, jj, :]  # [B, C]

    logp = F.log_softmax(logits, dim=-1)
    p    = logp.exp()
    y    = pair_cat
    pt   = p[torch.arange(B, device=p.device), y]

    alpha_vec = torch.full_like(pt, fill_value=alpha_pos)
    alpha_vec = torch.where(y==0, torch.full_like(alpha_vec, alpha_none), alpha_vec)

    focal = alpha_vec * ((1.0 - pt) ** gamma) * F.nll_loss(logp, y, reduction="none")
    pair_loss = focal.mean()

    pred_va = va_scores[idx_b, ii, jj, :]                   # [B,2]
    va_targets = torch.clamp(va_targets, 1.0, 9.0)          # [B,2]
    va_loss = F.mse_loss(pred_va, va_targets)

    total = a_loss + o_loss + lambda_pair * pair_loss + lambda_va * va_loss
    return total, (a_loss.item(), o_loss.item(), pair_loss.item(), va_loss.item())

# ================ EMA（新增，最小实现） ================
# ================ EMA（完整：含 load_weights） ================
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
        if n not in ema_dict:
            ema_dict[n] = p.detach().clone()
        ema_dict[n].mul_(decay).add_(p.detach(), alpha=1-decay)

@torch.no_grad()
def load_weights(model, state_dict):
    """
    将 state_dict（通常来自 EMA 字典）拷回到模型参数上。
    仅覆盖【模型中存在且需要训练】的参数，避免不必要的覆盖。
    """
    for n,p in model.named_parameters():
        if n in state_dict and p.requires_grad:
            p.copy_(state_dict[n])

def state_dict_from_ema(ema_dict):
    """
    返回 EMA 字典的深拷贝，用于保存到 checkpoint。
    """
    return {k: v.clone() for k,v in ema_dict.items()}


# ================ 训练/验证 ================
def train_one_epoch(model, loader, opt, sch, cfg: Config, ema_dict=None, ema_decay=0.999):
    model.train()
    scaler = GradScaler(enabled=cfg.use_amp and device.type=="cuda")
    total, n = 0.0, 0
    logs = [0.0,0.0,0.0,0.0]
    amp_ctx = torch.autocast("cuda", dtype=torch.float16) if (cfg.use_amp and device.type=="cuda") else nullcontext()

    for batch in loader:
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
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        # —— EMA 更新（新增） ——
        if ema_dict is not None:
            ema_update(model, ema_dict, decay=ema_decay)

        bs = ids.size(0); total += loss.item()*bs; n += bs
        for k in range(4): logs[k]+=parts[k]*bs
    return total/n, [x/n for x in logs]

@torch.no_grad()
def evaluate(model, loader, cfg: Config, ema_dict=None):
    # 验证时若提供 EMA，则临时切换到 EMA 权重
    backup = None
    if ema_dict is not None:
        backup = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        load_weights(model, ema_dict)

    model.eval()
    total, n = 0.0, 0
    logs=[0.0,0.0,0.0,0.0]
    for batch in loader:
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        a_tags = batch["a_tags"].to(device)
        o_tags = batch["o_tags"].to(device)
        pair_ij= batch["pair_ij"].to(device)
        pair_cat=batch["pair_cat"].to(device)
        va_targets = batch["va_target"].to(device)

        a_logits, o_logits, pair_scores, va_scores = model(ids, att)
        loss, parts = loss_fn(
            a_logits, o_logits, pair_scores, va_scores,
            a_tags, o_tags, pair_ij, pair_cat, va_targets,
            lambda_pair=cfg.lambda_pair, gamma=cfg.gamma_focal, lambda_va=cfg.lambda_va,
            alpha_none=cfg.alpha_none, alpha_pos=cfg.alpha_pos
        )
        bs = ids.size(0); total += loss.item()*bs; n += bs
        for k in range(4): logs[k]+=parts[k]*bs

    # 还原权重
    if backup is not None:
        load_weights(model, backup)
    return total/n, [x/n for x in logs]

# ===================== 推理解码 =====================
@torch.no_grad()
def predict_dev(model, dev_df, tok, cfg: Config, id2cat: Dict[int, str], ema_dict=None):
    # 推理时若提供 EMA，则临时切换到 EMA 权重
    backup = None
    if ema_dict is not None:
        backup = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        load_weights(model, ema_dict)

    class _DevDS(Dataset):
        def __init__(self, df, tok, max_len):
            self.df = df.reset_index(drop=True); self.tok = tok; self.max_len = max_len
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            r = self.df.iloc[i]
            enc = self.tok(r["text"], truncation=True, max_length=min(self.max_len, cfg.max_decode_len),
                           return_offsets_mapping=True)
            off = enc.pop("offset_mapping")
            enc["offset_mapping"] = off
            enc["id"] = str(r["id"])
            enc["text"] = r["text"]
            return enc

    def _coll(feats):
        ids = [f.pop("id") for f in feats]
        texts = [f.pop("text") for f in feats]
        offsets = [f.pop("offset_mapping") for f in feats]  # 修复键名
        batch = tok.pad(feats, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        batch["id"] = ids; batch["text"] = texts; batch["offsets"] = offsets
        return batch

    ds = _DevDS(dev_df, tok, cfg.max_len)
    num_workers = 4 if device.type == "cuda" else 0
    pin = True if device.type == "cuda" else False
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin, collate_fn=_coll)

    model.eval()
    lines = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device).bool()

        a_logits, o_logits, pair_scores, va_scores = model(ids, att.to(ids.dtype))
        probs = F.softmax(pair_scores, dim=-1)  # [B,L,L,C]
        B, L, _ = a_logits.size()

        for b in range(B):
            text = batch["text"][b]
            offsets = batch["offsets"][b]
            mask_b = att[b].tolist()

            # 1) 解码 A/O spans —— 带阈值的起点法（略收紧）
            a_prob_b = a_logits[b].softmax(-1).unsqueeze(0)
            o_prob_b = o_logits[b].softmax(-1).unsqueeze(0)
            mask_tensor = torch.tensor(mask_b, device=a_prob_b.device).unsqueeze(0)
            A_spans = _decode_spans_B(a_prob_b, mask_tensor, tau=0.16)[0]
            O_spans = _decode_spans_B(o_prob_b, mask_tensor, tau=0.22)[0]
            A_spans = [(i,j) for (i,j) in A_spans if (j-i+1) >= 1]
            O_spans = [(i,j) for (i,j) in O_spans if (j-i+1) >= 1]

            # 2) 候选配对：span×span 最大化 + 阈值
            cand = []
            for (ai, aj) in A_spans:
                for (oi, oj) in O_spans:
                    if abs(ai - oi) > cfg.pair_window:
                        continue
                    block = probs[b, ai:(aj+1), oi:(oj+1), :]         # [lenA,lenO,C]
                    flat  = block.reshape(-1, block.size(-1))         # [lenA*lenO, C]
                    max_idx = torch.argmax(flat)                      # —— 新增：取最大处索引
                    lenA, lenO, Cc = block.size(0), block.size(1), block.size(2)
                    ri, rj, c_best = (max_idx // Cc) // lenO, (max_idx // Cc) % lenO, int(max_idx % Cc)
                    p_per_class = flat.max(dim=0).values              # 每类最大概率
                    cat_id = int(torch.argmax(p_per_class).item())
                    p_max  = float(p_per_class.max().item())
                    if cat_id == 0 or p_max < cfg.pair_thresh:
                        continue

                    # —— 用最大类概率所在 token 对 (vi,vj) 读取 VA（关键修正）
                    vi, vj = ai + int(ri), oi + int(rj)
                    v_raw = float(va_scores[b, vi, vj, 0].item())
                    a_raw = float(va_scores[b, vi, vj, 1].item())
                    v_fin = round(max(1.0, min(9.0, v_raw)), 2)
                    a_fin = round(max(1.0, min(9.0, a_raw)), 2)

                    cand.append((ai, aj, oi, oj, cat_id, p_max, v_fin, a_fin, vi, vj))

            # 3) 互为最近（保留 margin 松弛）
            MARGIN = 0.02
            bestO = defaultdict(lambda: (-1, -1.0))  # ai -> (idx, p)
            bestA = defaultdict(lambda: (-1, -1.0))  # oi -> (idx, p)
            for idx, (ai, aj, oi, oj, cat_id, p, v_fin, a_fin, vi, vj) in enumerate(cand):
                if p > bestO[ai][1]: bestO[ai] = (idx, p)
                if p > bestA[oi][1]: bestA[oi] = (idx, p)
            keep_idx = set()
            for ai, (io, p) in bestO.items():
                oi = cand[io][2]
                j_best, p_b = bestA[oi]
                if j_best == io or abs(p_b - p) <= MARGIN:
                    keep_idx.add(io)

            # 4) 回填 + 句内去重（弱意见过滤默认关闭）
            trips = []
            seen = set()
            for k in sorted(keep_idx, key=lambda i: -cand[i][5])[:cfg.topk_pairs]:
                ai, aj, oi, oj, cat_id, p, v_fin, a_fin, vi, vj = cand[k]
                a_st = offsets[ai][0]; a_ed = offsets[aj][1]
                o_st = offsets[oi][0]; o_ed = offsets[oj][1]
                asp = text[a_st:a_ed].strip()
                opn = text[o_st:o_ed].strip()
                key = (asp, opn)
                if key in seen: continue
                seen.add(key)
                trips.append({"Aspect": asp, "Opinion": opn, "VA": f"{v_fin:.2f}#{a_fin:.2f}"})

            # ===================== 兜底策略（保持不变，阈值略收紧）=====================
            if len(trips) == 0:
                pair_thresh_fb = max(0.15, cfg.pair_thresh - 0.10)
                margin_fb = 0.05
                keep_idx_fb = set()
                bestO = defaultdict(lambda: (-1, -1.0))
                bestA = defaultdict(lambda: (-1, -1.0))
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
                        # VA 用起点-起点即可（保持最小改动）
                        v_raw = float(va_scores[b, ai, oi, 0].item())
                        a_raw = float(va_scores[b, ai, oi, 1].item())
                        v_fin = round(max(1.0, min(9.0, v_raw)), 2)
                        a_fin = round(max(1.0, min(9.0, a_raw)), 2)
                        cand_fb.append((ai, aj, oi, oj, cat_id, p_max, v_fin, a_fin))
                for idx, (ai, aj, oi, oj, cat_id, p, v_fin, a_fin) in enumerate(cand_fb):
                    if p > bestO[ai][1]: bestO[ai] = (idx, p)
                    if p > bestA[oi][1]: bestA[oi] = (idx, p)
                for ai, (io, p) in bestO.items():
                    oi = cand_fb[io][2] if len(cand_fb)>0 else -1
                    if oi != -1:
                        j_best, p_b = bestA[oi]
                        if j_best == io or abs(p_b - p) <= margin_fb:
                            keep_idx_fb.add(io)
                if len(keep_idx_fb) > 0:
                    k = sorted(keep_idx_fb, key=lambda i: -cand_fb[i][5])[0]
                    ai, aj, oi, oj, cat_id, p, v_fin, a_fin = cand_fb[k]
                    a_st = offsets[ai][0]; a_ed = offsets[aj][1]
                    o_st = offsets[oi][0]; o_ed = offsets[oj][1]
                    asp = text[a_st:a_ed].strip()
                    opn = text[o_st:o_ed].strip()
                    trips.append({"Aspect": asp, "Opinion": opn, "VA": f"{v_fin:.2f}#{a_fin:.2f}"})
            if len(trips) == 0:
                P = probs[b]
                P_nonzero = P[..., 1:]
                p_max_val = float(P_nonzero.max().item())
                if p_max_val >= 0.65:   # 0.60 -> 0.65
                    idx = torch.argmax(P_nonzero)
                    C_ = P_nonzero.size(-1)
                    L_ = P_nonzero.size(0)
                    ijc = np.unravel_index(int(idx), (L_, L_, C_))
                    i_best, j_best, c_best = int(ijc[0]), int(ijc[1]), int(ijc[2]+1)
                    ai = max(0, min(i_best, len(offsets)-1))
                    oi = max(0, min(j_best, len(offsets)-1))
                    a_st = offsets[ai][0]; a_ed = offsets[ai][1]
                    o_st = offsets[oi][0]; o_ed = offsets[oi][1]
                    asp = text[a_st:a_ed].strip()
                    opn = text[o_st:o_ed].strip()
                    v_raw = float(va_scores[b, ai, oi, 0].item())
                    a_raw = float(va_scores[b, ai, oi, 1].item())
                    v_fin = round(max(1.0, min(9.0, v_raw)), 2)
                    a_fin = round(max(1.0, min(9.0, a_raw)), 2)
                    if asp and opn:
                        trips.append({"Aspect": asp, "Opinion": opn, "VA": f"{v_fin:.2f}#{a_fin:.2f}"})
            # ===================== 兜底结束 ======================

            lines.append({"ID": str(batch["id"][b]), "Triplet": trips})

    # 还原权重
    if backup is not None:
        load_weights(model, backup)
    return lines

def save_jsonl(objs, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for o in objs: w.write(json.dumps(o, ensure_ascii=False)+"\n")

# ================ 主流程 ================
def main():
    set_verbosity_error()
    cfg = Config(); set_seed(cfg.seed)

    # 模型与分词器
    local_dir = (Path(__file__).resolve().parents[2] / cfg.local_model_dir).resolve()
    assert local_dir.exists(), f"本地模型目录不存在：{local_dir}"
    tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True, local_files_only=True)
    enc = AutoModel.from_pretrained(local_dir, local_files_only=True).to(device)

    # 数据路径
    data_dir = output_path("output", "track_a", "subtask_2")
    train_path = (data_dir / "train_pairs.parquet")
    dev_path   = (data_dir / "dev_pairs.parquet")
    if not train_path.exists():
        train_path = (data_dir / "eng" / "subtask2_processed" / "train_subtask2.parquet")
        dev_path   = (data_dir / "eng" / "subtask2_processed" / "dev_subtask2.parquet")

    train_df = pd.read_parquet(train_path)
    dev_df   = pd.read_parquet(dev_path)

    cat2id = build_label_map(train_df["category"].astype(str).tolist())
    id2cat = {v:k for k,v in cat2id.items()}

    if "id" in train_df.columns:
        gss = GroupShuffleSplit(test_size=0.1, random_state=cfg.seed)
        tr_idx, va_idx = next(gss.split(train_df, groups=train_df["id"]))
        tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    else:
        tr_df, va_df = train_test_split(train_df, test_size=0.1, random_state=cfg.seed)

    tr_ds = ASTETrainDataset(tr_df, tok, cfg.max_len, cat2id)
    va_ds = ASTETrainDataset(va_df, tok, cfg.max_len, cat2id)
    n_workers = 4 if device.type=="cuda" else 0
    pin = True if device.type=="cuda" else False
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=n_workers, pin_memory=pin,
                           collate_fn=TrainCollator(tok,8))
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=n_workers, pin_memory=pin,
                           collate_fn=TrainCollator(tok,8))

    model = ASTEModel(enc, n_cat=len(cat2id), va_head_dim=cfg.va_head_dim).to(device)

    # 先冻结 encoder
    for p in model.enc.parameters(): p.requires_grad=False
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
        if ep == cfg.freeze_epochs:
            for p in model.enc.parameters(): p.requires_grad=True
            steps_left = max(1, len(tr_loader)*(cfg.epochs-ep))
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