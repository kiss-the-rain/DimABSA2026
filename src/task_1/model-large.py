# -*- coding: utf-8 -*-
# 离线版：本地加载 roberta-base（不访问外网），含早停/冻结解冻/动态 padding/AMP 自适应
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from dataclasses import dataclass
from pathlib import Path
import json
import sys
import random, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import output_path, data_path
from transformers.utils.logging import set_verbosity_error


# ================= 配置 =================
@dataclass
# 存储参数
class Config:
    local_model_dir: str = "/home/cuizhibin/projects/Models/roberta-large"
    max_len: int = 512
    # 优化与正则
    lr_encoder: float = 1e-5
    lr_head: float = 1.7e-4
    weight_decay: float = 0.03
    warmup_ratio: float = 0.06
    warmup_min_steps: int = 400  # warmup 下限

    # 训练策略
    epochs: int = 20
    batch_size: int = 32
    freeze_epochs: int = 1
    enc_lr_after_unfreeze: float = 6e-6  # 解冻后 encoder 更小 LR
    patience: int = 3
    use_huber: bool = True

    seed: int = 42


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
torch.set_float32_matmul_precision("high")


# 提高运算速度，轻微降低精度


# ================= 分词与输入 =================
# 把样本在分词阶段统一成同一种字典结构（样本级数据处理）
def build_inputs(tok, text: str, aspect: str, max_len: int):
    # tok = tokenizer 分词器
    """不返回 tensor；让自定义 collator 统一做 pad + tensor 化"""
    prompt = f"Aspect: {aspect}"
    return tok(
        text,
        prompt,
        truncation="only_first",
        # truncation="only_first" 是 Hugging Face 分词器在双序列输入（text_pair 存在）时的截断策略
        # 当text长度超过 max_length时，text会被截断 only_first 只会截断第一个序列，即text，（同理，第二个序列为 prompt）
        max_length=max_len,
        # 不在这里 padding/tensor 化
    )
    # 返回一个经过分词器处理后的BatchEncoding字典，后续在collator中统一处理


# ================= 数据集 =================
class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int, tok):
        '''
        df：你的训练表，至少含 text, aspect, v, a 四列。
        reset_index(drop=True)：把 DataFrame 索引整理成 0..N-1 升序，避免花式索引造成 iloc 慢或报错。
        tok：HuggingFace 的 tokenizer 对象。
        max_len：编码后的最大长度（包含特殊符号）。
        '''
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tok = tok

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]  # 取df(训练集)的第i行
        # 调用build_inputs 把样本在分词阶段统一成同一种字典结构
        enc = build_inputs(self.tok, r["text"], r["aspect"], self.max_len)
        # 标签从[1,9]缩放到 [0,1]
        y = torch.tensor([
            (float(r["v"]) - 1.0) / 8.0,
            (float(r["a"]) - 1.0) / 8.0
        ], dtype=torch.float32)
        enc["labels"] = y
        return enc


class DevDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int, tok):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tok = tok

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        enc = build_inputs(self.tok, r["text"], r["aspect"], self.max_len)
        enc["id"] = str(r["id"])
        enc["aspect"] = r["aspect"]
        return enc


# ================= 自定义 Collator =================
# 批次级训练数据处理  统一“一个批次的长度/张量形状”
# 使一个batch的数据的长度对齐
# 若在build_inputs中对齐，则会选择所有数据中最大的一个，对其他长度更小的batch的数据，造成空间浪费
class TrainCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tok = tokenizer  # 选择分词器
        self.pad_to_multiple_of = pad_to_multiple_of  # 设置补齐后的长度

    def __call__(self, features):
        # features 是TrainDataset中返回的enc
        # 1) 取出 labels，避免被 tokenizer.pad 处理
        labels = [f.pop("labels") for f in features]
        # 2) 只 pad 模型输入
        batch = self.tok.pad(
            features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
            # 一次性把所有字段堆叠成 PyTorch 张量，维度为 [B, L]
        )
        # 3) 把 labels 手动堆叠回去
        batch["labels"] = torch.stack(labels, dim=0)  # [B,2]
        # 把先前存下的标签列表堆叠成 [B, 2] 的 FloatTensor
        return batch


class DevCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        ids = [f.pop("id") for f in features]
        asps = [f.pop("aspect") for f in features]
        batch = self.tok.pad(
            features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        batch["id"] = ids
        batch["aspect"] = asps
        return batch


# ================= 模型 =================
class MeanPooling(nn.Module):
    '''
    last_hidden_state: [B, L, H]，编码器最后一层每个 token 的隐向量（B=batch，L=序列长度，H=隐藏维度(每个 token 的向量长度/特征数)）。
    #? 相当于是last_hidden中存放的是文字向量信息,而mask是权重信息,通过把向量相加,再除以序列的长度,就得到了平均向量
    attention_mask: [B, L]，有效位置为 1，padding 为 0（通常特殊符号也是 1，见下方提示）。

    unsqueeze(-1): 得到 [B, L, 1]，便于与 [B, L, H] 做逐 token 逐维度相乘（广播）。

    type_as(last_hidden_state): 把 mask 转成与隐向量相同的浮点类型（fp16/bf16/fp32），这样在 AMP 下不会触发不必要的类型提升。

    (last_hidden_state * mask).sum(1): 把 padding 对应的位置乘 0，再沿着 token 维度求和，得到加权和 [B, H]。

    mask.sum(1): 每个样本有效 token 个数（可能包含特殊符号，见下）。

    clamp(min=1e-6): 防止全 0 时除以 0（即完全是 padding 的极端情况）。

    /: 加权和除以有效计数 ⇒ 均值池化，输出 [B, H]。
    '''

    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


class VARegressor(nn.Module):
    def __init__(self, encoder: AutoModel, dropout: float = 0.20):
        super().__init__()
        self.enc = encoder  # 加载编码器
        hidden = self.enc.config.hidden_size
        # hidden 指的是编码器（BERT/RoBERTa 等）隐向量的维度 H
        self.pool = MeanPooling()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),  # 非线性激活
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # inputs_ids是每个字词对对应的词表的id;mask是有效信息标识;token_type_ids是对不同句子的标识
        # out.last_hidden_state: [B, L, H]
        feat = self.pool(out.last_hidden_state, attention_mask)
        logits = self.head(feat)
        pred01 = torch.sigmoid(logits)  # in [0,1]
        return pred01


# ================= 损失与指标 =================
def loss_fn(pred01, target01, huber=False):
    return F.smooth_l1_loss(pred01, target01) if huber else F.mse_loss(pred01, target01)


# 计算的是整体 RMSE：把 V 与 A 两个维度、以及 整个 batch 的误差一起平均，再开根号。
def rmse_va(pred, target):
    # pred/target: [B, 2], 已在 [1,9] 空间
    err2 = (pred - target) ** 2  # [B, 2]
    per_sample = err2.sum(dim=1)  # [B] —— V/A 两维相加
    mse = per_sample.mean()  # 对样本数 N 取平均
    return torch.sqrt(mse)


# 线性反缩放函数：把模型在 [0,1] 区间的输出 X01 映射回任务原标尺 [1,9]。

def _to_19(x01):  # [0,1] -> [1,9]
    return 1.0 + 8.0 * x01


# ================= 优化器与调度器 =================
# 定义一个辅助函数，同时构建优化器和学习率调度器
def _build_param_groups(model, lr_enc, lr_head, wd):
    def no_decay(n):  # bias / LayerNorm 不做 weight decay
        return ("bias" in n) or ("LayerNorm.weight" in n)

    enc_decay, enc_nodecay, head_decay, head_nodecay = [], [], [], []
    # enc_decay：编码器里需要 L2 的参数
    # enc_nodecay：编码器里不需要 L2 的参数（bias/LN）
    # head_decay：回归头里需要 L2 的参数
    # head_nodecay：回归头里不需要 L2 的参数
    for n, p in model.named_parameters():  # n表示parameters的名字，p表示对应的参数对象
        if not p.requires_grad:
            continue
        if "enc." in n:  # 名字包含 "enc." 来判断是否属于编码器
            (enc_nodecay if no_decay(n) else enc_decay).append(p)
        else:
            (head_nodecay if no_decay(n) else head_decay).append(p)

    return [  # 返回四个优化器参数组（给 torch.optim.AdamW([...])）
        {"params": enc_decay, "lr": lr_enc, "weight_decay": wd},  # Encoder 里需要做 weight decay(L2)
        {"params": enc_nodecay, "lr": lr_enc, "weight_decay": 0.0},  # Encoder 里不做 weight decay
        {"params": head_decay, "lr": lr_head, "weight_decay": wd},  # 下游 回归头(head) 中需要做 weight decay 的参数
        {"params": head_nodecay, "lr": lr_head, "weight_decay": 0.0},  # 回归头里 不做 weight decay 的参数
    ]


# 构建优化器 + 学习率调度器，并把它们返回给训练循环使用
# build_optim_sched = “给训练循环准备好怎么更新（优化器）和怎么变速（调度器）”
# 但不做“初始化权重”和“实际更新权重”的工作。
def build_optim_sched(model, num_train_steps, lr_enc, lr_head, wd, warmup_ratio, warmup_min_steps):
    param_groups = _build_param_groups(model, lr_enc, lr_head, wd)
    optimizer = torch.optim.AdamW(param_groups)
    num_warmup = max(warmup_min_steps, int(num_train_steps * warmup_ratio))
    # 计算预热步数
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_train_steps)
    # 创建 线性预热 + 线性衰减 的调度器：
    return optimizer, scheduler


# ================= 训练 / 验证 =================
def train_one_epoch(model, loader, optimizer, scheduler, device, use_amp=True, huber=False, max_grad_norm=1.0):
    '''
    model：你的 VARegressor
    loader：训练 DataLoader
    optimizer / scheduler：优化器与学习率调度器
    device：cpu/cuda/mps
    use_amp：是否启用自动混合精度
    huber：损失是否用 Huber（SmoothL1）
    max_grad_norm：梯度裁剪阈值
    '''
    model.train()
    scaler = GradScaler(enabled=use_amp)
    # AMP 的梯度缩放器；开启后会动态放大/缩小 loss，避免半精度下的 underflow（数值变 0）。
    total_loss = total_rmse = 0.0
    n = 0
    # autocast 会在合适算子上用 FP16/BF16，提高吞吐、节省显存。
    amp_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float16
    for batch in loader:  # 开始提取数据
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ttids = batch.get("token_type_ids")
        ttids = ttids.to(device) if ttids is not None else None
        labels01 = batch["labels"].to(device)  # [B,2] in [0,1]

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            pred01 = model(input_ids, attn, ttids)  # [0,1]
            loss = loss_fn(pred01, labels01, huber=huber)

        scaler.scale(loss).backward()  # 反向传播 对 loss 缩放后反传。此处不会立刻更新参数，只是把梯度累在 .grad 上。
        # ?等价于对 loss * scale 反传，所有梯度都会被乘以 scale，下溢风险显著降低。
        # ?之后更新参数前需要把这层“放大”去掉，所以要 unscale_。
        scaler.unscale_(optimizer)  # 把之前缩放过的梯度还原回真实尺度；为了下一步能正确做梯度裁剪或检查梯度值。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        # 对所有参数的梯度做 L2 范数裁剪，限制到 max_grad_norm。防止梯度爆炸，稳定训练。
        scaler.step(optimizer)  # 用当前参数上的梯度，按优化算法（AdamW/SGD 等）更新参数值
        scaler.update()
        # scaler.step(optimizer)：用缩放安全的方式执行 optimizer.step()（内部会根据梯度是否为 NaN/Inf 决定跳过或执行）。
        scheduler.step()

        # === 日志 RMSE：反缩放到 [1,9] 再算（与官方一致）===
        pred19 = _to_19(pred01.detach())
        lab19 = _to_19(labels01)  # 将预测和样本数据缩放到[1,9]
        rmse_b = rmse_va(pred19, lab19)  # 计算当前batch的RMSE
        bs = input_ids.size(0)  # 取当前batch的样本数
        total_loss += loss.item() * bs
        total_rmse += rmse_b.item() * bs  # 用样本数加权累加 loss 与 RMSE，确保最终是按样本平均
        n += bs  # 计算所有样本数

    return total_loss / n, total_rmse / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = total_rmse = 0.0
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ttids = batch.get("token_type_ids")
        ttids = ttids.to(device) if ttids is not None else None
        labels01 = batch["labels"].to(device)  # [B,2] in [0,1]

        pred01 = model(input_ids, attn, ttids)  # [0,1]
        loss = loss_fn(pred01, labels01)

        # === 验证 RMSE：反缩放到 [1,9] 再算（与线上一致）===
        pred19 = _to_19(pred01)
        lab19 = _to_19(labels01)
        rmse_b = rmse_va(pred19, lab19)

        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        total_rmse += rmse_b.item() * bs
        n += bs

    return total_loss / n, total_rmse / n


# ================= 推理与保存 =================
def format_va(v, a):
    v = float(max(1.0, min(9.0, v)))
    a = float(max(1.0, min(9.0, a)))
    return f"{v:.2f}#{a:.2f}"


@torch.no_grad()
def predict_dev(model, dev_df, tok, cfg: Config):
    ds = DevDataset(dev_df, cfg.max_len, tok)
    # 用你的 DevDataset 封装样本，负责把 (text, aspect) 编码成定长序列

    # 服务器优化操作
    num_workers = 4 if device.type == "cuda" else 0
    pin_mem = True if device.type == "cuda" else False

    collate = DevCollator(tok, pad_to_multiple_of=8)
    # 批处理阶段的 collate_fn：按本批最长序列动态 padding，并把长度对齐到 8 的倍数（利于 Tensor Cores/AMP）。
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_mem,
                        collate_fn=collate)
    # 批量加载，不打乱顺序
    model.eval()
    records = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ttids = batch.get("token_type_ids")
        ttids = ttids.to(device) if ttids is not None else None

        pred01 = model(input_ids, attn, ttids).cpu().numpy()  # [B,2] in [0,1]
        # 前向得到 [B,2] 的归一化预测（0~1），立刻搬回 CPU 并转 numpy 做后处理。
        pred19 = 1.0 + 8.0 * pred01  # 回缩到 [1,9]

        # 逐条把该 batch 的 ID、Aspect 与对应的 V/A 取出，存成一行
        ids = batch["id"]
        asps = batch["aspect"]
        for i, (vid, asp) in enumerate(zip(ids, asps)):
            v, a = float(pred19[i][0]), float(pred19[i][1])
            records.append((vid, asp, v, a))

    bag = defaultdict(list)
    for rid, asp, v, a in records:
        bag[rid].append({"Aspect": asp, "VA": format_va(v, a)})
        # 组装成 {"Aspect": 原样的 aspect, "VA": "V#A"} 放入该 ID 的列表中。

    lines = [{"ID": rid, "Aspect_VA": items} for rid, items in bag.items()]
    # 最终输出为一个行对象列表：{"ID": "R001", "Aspect_VA": [{"Aspect":"thai food","VA":"6.75#6.38"}, ...]}
    return lines


def save_jsonl(objs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for o in objs:
            w.write(json.dumps(o, ensure_ascii=False) + "\n")

def _load_domain_ids(path: Path, domain_name: str) -> dict:
    if not path.exists():
        return {}
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = obj.get("ID") or obj.get("id") or obj.get("Id")
            if rid is None:
                continue
            mapping[str(rid)] = domain_name
    return mapping

def _infer_domain_from_id(rid: str) -> str | None:
    s = str(rid).lower()
    if "lap" in s:
        return "laptop"
    if "rest" in s or "res" in s:
        return "restaurant"
    return None

def build_domain_map(dev_df: pd.DataFrame) -> dict:
    if "domain" in dev_df.columns:
        return (
            dev_df[["id", "domain"]]
            .drop_duplicates("id")
            .set_index("id")["domain"]
            .astype(str)
            .str.strip()
            .str.lower()
            .to_dict()
        )
    domain_map = {}
    for lang in ("eng", "zho"):
        for domain_name in ("laptop", "restaurant"):
            path = data_path("track_a", "subtask_1", lang, f"{lang}_{domain_name}_dev_task1.jsonl")
            domain_map.update(_load_domain_ids(path, domain_name))
    if domain_map:
        return domain_map
    if "id" not in dev_df.columns:
        return {}
    ids = dev_df["id"].astype(str).tolist()
    return {rid: d for rid in ids if (d := _infer_domain_from_id(rid)) is not None}


# ================= 主流程 =================
def main():
    print("ROOT ->", os.getenv("DIMABSA_ROOT"))
    print("EXPECT ->", output_path("output", "track_a", "subtask_1"))
    cfg = Config()
    set_seed(cfg.seed)
    set_verbosity_error()
    # 1) 离线加载分词器与编码器（不联网）
    local_dir = (Path(__file__).resolve().parents[2] / cfg.local_model_dir).resolve()
    assert local_dir.exists(), f"本地模型目录不存在：{local_dir}"
    tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True, local_files_only=True)
    enc = AutoModel.from_pretrained(local_dir, local_files_only=True).to(device)

    # 2) 读取 dataset.py 预处理后的数据
    data_dir = output_path("output", "track_a", "subtask_1")
    train_df = pd.read_parquet(data_dir / "train_pairs.parquet")
    dev_df = pd.read_parquet(data_dir / "dev_pairs.parquet")

    # 3) 分组切分（按同一句子的 id 分组，避免信息泄漏）
    if "id" in train_df.columns:
        groups = train_df["id"]
        gss = GroupShuffleSplit(test_size=0.1, random_state=cfg.seed)
        # 构造一个“按组随机划分”的拆分器：
        tr_idx, va_idx = next(gss.split(train_df, groups=groups))
        # 让拆分器基于 groups 产生一次划分，得到下标，同一个 id 的样本必须被划到同一侧
        tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    else:
        tr_df, va_df = train_test_split(train_df, test_size=0.1, random_state=cfg.seed)

    # 4) DataLoader（动态 padding + 8 对齐）
    # 服务器优化
    num_workers = 4 if device.type == "cuda" else 0
    pin_mem = True if device.type == "cuda" else False

    # 加载数据集
    tr_ds = TrainDataset(tr_df, cfg.max_len, tok)
    va_ds = TrainDataset(va_df, cfg.max_len, tok)
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=pin_mem,
                           collate_fn=TrainCollator(tok, pad_to_multiple_of=8))
    # collate 是从dataload取出一个batch的数据后要执行的函数
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_mem,
                           collate_fn=TrainCollator(tok, pad_to_multiple_of=8))

    # 5) 构建模型/优化器/调度器
    model = VARegressor(enc, dropout=0.20).to(device)
    steps_total = len(tr_loader) * cfg.epochs
    optimizer, scheduler = build_optim_sched(
        model, steps_total, cfg.lr_encoder, cfg.lr_head,
        cfg.weight_decay, cfg.warmup_ratio, cfg.warmup_min_steps
    )

    # 6) 训练循环：冻结→解冻 + 早停
    for p in model.enc.parameters():
        p.requires_grad = False

    best_rmse, best_path = 1e9, "best_model.pt"
    use_amp = (device.type != "cpu")
    # 在 非 CPU（CUDA/MPS）设备上启用 AMP（自动混合精度），以提升吞吐、降显存
    bad = 0  # 早停计数器

    for ep in range(cfg.epochs):
        if ep == cfg.freeze_epochs:  # 解冻
            for p in model.enc.parameters():
                p.requires_grad = True
            steps_left = len(tr_loader) * (cfg.epochs - ep)  # 剩余训练步数

            # 重建优化器与 LR 调度器，使用新的学习率
            optimizer, scheduler = build_optim_sched(
                model, steps_left, cfg.enc_lr_after_unfreeze, cfg.lr_head,
                cfg.weight_decay, cfg.warmup_ratio, cfg.warmup_min_steps
            )
        # 执行一个epoch的训练，并返回相应的参数
        tl, tr = train_one_epoch(model, tr_loader, optimizer, scheduler, device,
                                 use_amp=use_amp, huber=cfg.use_huber)
        vl, vr = evaluate(model, va_loader, device)
        print(f"Epoch {ep + 1:02d} | train_loss={tl:.4f} rmse={tr:.3f} || val_loss={vl:.4f} rmse={vr:.3f}")
        # 最优 则保存模型
        if vr < best_rmse:
            best_rmse, bad = vr, 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, best_path)
            print(f"[BEST] saved -> {best_path} (rmse={best_rmse:.3f})")
        else:  # 早停机制
            bad += 1
            if bad >= cfg.patience:
                print(f"[EARLY STOP] no improvement for {cfg.patience} epochs.")
                break

    # 7) 推理（开发集）
    # 加载最优模型
    try:
        ckpt = torch.load("best_model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load("best_model.pt", map_location="cpu")

    enc2 = AutoModel.from_pretrained(local_dir, local_files_only=True).to(device)
    model = VARegressor(enc2, dropout=0.20).to(device)
    model.load_state_dict(ckpt["state_dict"])
    # 调用预测函数，输出JSONL列表
    preds = predict_dev(model, dev_df, tok, cfg)
    domain_map = build_domain_map(dev_df)
    if not domain_map:
        raise KeyError("dev_pairs.parquet 缺少 domain 字段，且无法从原始 dev JSONL 或 ID 推断")

    for domain_name, file_name in [
        ("laptop", "pred_eng_laptop.jsonl"),
        ("restaurant", "pred_eng_restaurant.jsonl"),
    ]:
        lines = [
            p for p in preds if domain_map.get(p.get("ID")) == domain_name
        ]
        out_file = output_path("submit", "task1", file_name)
        save_jsonl(lines, out_file)
        print(f"Dev 提交文件 -> {out_file} ({domain_name}: {len(lines)})")

    missing = [p for p in preds if domain_map.get(p.get("ID")) is None]
    if missing:
        print(f"[WARN] 无法判定 domain 的样本数: {len(missing)}，已跳过写入")


if __name__ == "__main__":
    main()
