# model.py
# =========================================================
# DimABSA Track A / Subtask-1 (DimASR)
# Encoder + 双回归头 (V, A)
# 服务器友好：CUDA/MPS/CPU 自适应、条件 AMP、梯度累积、可写输出路径
# =========================================================

from dataclasses import dataclass
from pathlib import Path
import os, random, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from collections import defaultdict
from src.utils.paths import output_path  # 你的 paths.py 工具

# ---------- 环境变量（服务器缓存/并行友好） ----------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 如服务器有专用写权限目录，设置 HF 缓存，避免默认写到 ~/.cache
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/.cache/huggingface/hub")
# 如需离线加载模型，外部设置：export HF_HUB_OFFLINE=1

# -------------------- 基础配置 --------------------
@dataclass
class Config:
    model_name: str = "roberta-base"  # 中文：hfl/chinese-roberta-wwm-ext
    max_len: int = 192

    lr_encoder: float = 1e-5
    lr_head: float   = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06

    epochs:   int = 5
    batch_size: int = 16
    grad_accum: int = 1           # 梯度累积（显存不够时 >1）
    seed:     int = 42

    dropout: float = 0.1
    num_workers: int = 4          # 服务器上可调大些（2/4/8）
    use_grad_ckpt: bool = False   # 是否开启 encoder 的梯度检查点（省显存，略慢）
    try_compile: bool = False     # PyTorch 2.3+ 可尝试 torch.compile

# -------------------- 设备/随机种子/AMP --------------------
def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    # 推荐：允许非完全确定性以换取速度；若要绝对复现可改 True 并关闭 cudnn.benchmark
    torch.use_deterministic_algorithms(False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
torch.set_float32_matmul_precision("high")

def amp_setup(device: torch.device):
    """
    返回 (AMP 是否启用, autocast dtype, GradScaler 是否启用).
    保险起见：仅 CUDA 开 AMP + GradScaler；MPS/CPU 关闭 AMP。
    """
    if device.type == "cuda":
        # Amp dtype: 新卡建议 bfloat16；如遇不支持可换 float16
        return True, torch.bfloat16, True
    return False, torch.float32, False

AMP_ENABLED, AMP_DTYPE, SCALER_ENABLED = amp_setup(device)

# -------------------- 分词与构造输入 --------------------
def build_inputs(tokenizer, text: str, aspect: str, max_len: int):
    """
    双序列输入：text + prompt("Aspect: {aspect}")
    只截断第一序列(text)，尽量保留提示词完整。
    """
    prompt = f"Aspect: {aspect}"
    return tokenizer(
        text, prompt,
        truncation="only_first",
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )

# -------------------- Dataset --------------------
class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = build_inputs(self.tok, r["text"], r["aspect"], self.max_len)
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor([float(r["v"]), float(r["a"])], dtype=torch.float32),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item

class DevDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = build_inputs(self.tok, r["text"], r["aspect"], self.max_len)
        item = {
            "id":      str(r["id"]),
            "aspect":  r["aspect"],
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item

# -------------------- 模型 --------------------
class VARegressor(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1, use_grad_ckpt: bool = False):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        if use_grad_ckpt and hasattr(self.enc, "gradient_checkpointing_enable"):
            try:
                self.enc.gradient_checkpointing_enable()
            except Exception:
                pass
        hidden = self.enc.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2)  # [V, A]
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.enc(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0]  # [B, H]
        logits = self.head(cls)            # [B, 2]
        return logits

# -------------------- 损失与指标 --------------------
def loss_fn(pred, target, huber=False):
    return F.smooth_l1_loss(pred, target) if huber else F.mse_loss(pred, target)

def rmse_va(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return torch.sqrt(mse)

# -------------------- 优化器与调度器 --------------------
def build_optim_sched(model, num_train_steps, lr_enc, lr_head, wd, warmup_ratio):
    enc_params, head_params = [], []
    for n, p in model.named_parameters():
        if "enc." in n:
            enc_params.append(p)
        else:
            head_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": lr_enc, "weight_decay": wd},
        {"params": head_params, "lr": lr_head, "weight_decay": wd},
    ])
    num_warmup = int(num_train_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_train_steps)
    return optimizer, scheduler

# -------------------- 训练与验证 --------------------
def train_one_epoch(model, loader, optimizer, scheduler, device,
                    use_huber=False, max_grad_norm=1.0, grad_accum=1):
    model.train()
    # 仅 CUDA 启用 GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=SCALER_ENABLED)

    total_loss, total_rmse, n = 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ttids = batch.get("token_type_ids")
        ttids = ttids.to(device) if ttids is not None else None
        labels = batch["labels"].to(device)

        # 条件 AMP（仅 CUDA）
        if AMP_ENABLED:
            with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                pred = model(input_ids, attn, ttids)
                loss = loss_fn(pred, labels, huber=use_huber)
                loss = loss / grad_accum
        else:
            pred = model(input_ids, attn, ttids)
            loss = loss_fn(pred, labels, huber=use_huber) / grad_accum

        # 反传
        if SCALER_ENABLED:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 累积到位才更新
        if (step + 1) % grad_accum == 0:
            if SCALER_ENABLED:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if SCALER_ENABLED:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # 统计
        bs = input_ids.size(0)
        total_loss += loss.item() * bs * grad_accum
        total_rmse += rmse_va(pred.detach(), labels).item() * bs
        n += bs

    return total_loss / n, total_rmse / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_rmse, n = 0.0, 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ttids = batch.get("token_type_ids")
        ttids = ttids.to(device) if ttids is not None else None
        labels = batch["labels"].to(device)

        pred = model(input_ids, attn, ttids)
        loss = loss_fn(pred, labels)
        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        total_rmse += rmse_va(pred, labels).item() * bs
        n += bs
    return total_loss / n, total_rmse / n

# -------------------- 训练主流程 --------------------
def fit(train_df: pd.DataFrame, cfg: Config, tokenizer: AutoTokenizer, ckpt_path: Path):
    set_seed(cfg.seed)

    tr_df, va_df = train_test_split(train_df, test_size=0.1, random_state=cfg.seed)

    tr_ds = TrainDataset(tr_df, tokenizer, cfg.max_len)
    va_ds = TrainDataset(va_df, tokenizer, cfg.max_len)

    pin = (device.type == "cuda")
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=True)

    model = VARegressor(cfg.model_name, dropout=cfg.dropout, use_grad_ckpt=cfg.use_grad_ckpt).to(device)

    # 可选：torch.compile 加速（PyTorch 2.3+）
    if cfg.try_compile:
        try:
            model = torch.compile(model)
        except Exception:
            pass

    steps = (len(tr_loader) // max(1, cfg.grad_accum)) * cfg.epochs
    optimizer, scheduler = build_optim_sched(model, steps, cfg.lr_encoder, cfg.lr_head, cfg.weight_decay, cfg.warmup_ratio)

    best_rmse = 1e9
    for ep in range(cfg.epochs):
        tl, tr = train_one_epoch(model, tr_loader, optimizer, scheduler, device,
                                 use_huber=False, max_grad_norm=1.0, grad_accum=cfg.grad_accum)
        vl, vr = evaluate(model, va_loader, device)
        print(f"Epoch {ep+1:02d} | train_loss={tl:.4f} rmse={tr:.3f} || val_loss={vl:.4f} rmse={vr:.3f}")
        if vr < best_rmse:
            best_rmse = vr
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
            print(f"[BEST] saved -> {ckpt_path} (rmse={best_rmse:.3f})")
    return ckpt_path

# -------------------- 推理与导出 --------------------
def format_va(v, a):
    v = float(max(1.0, min(9.0, v)))
    a = float(max(1.0, min(9.0, a)))
    return f"{v:.2f}#{a:.2f}"

@torch.no_grad()
def predict_dev(model, dev_df: pd.DataFrame, cfg: Config, tokenizer: AutoTokenizer):
    ds = DevDataset(dev_df, tokenizer, cfg.max_len)
    pin = (device.type == "cuda")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=True)

    model.eval()
    records = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ttids = batch.get("token_type_ids")
        ttids = ttids.to(device) if ttids is not None else None

        pred = model(input_ids, attn, ttids).cpu().numpy()  # [B,2]
        for i, (vid, asp) in enumerate(zip(batch["id"], batch["aspect"])):
            v, a = pred[i].tolist()
            records.append((vid, asp, v, a))

    bag = defaultdict(list)
    for rid, asp, v, a in records:
        bag[rid].append({"Aspect": asp, "VA": format_va(v, a)})

    lines = []
    for rid, items in bag.items():
        lines.append({"ID": rid, "Aspect_VA": items})
    return lines

def save_jsonl(objs, path: Path):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as w:
        for o in objs:
            w.write(json.dumps(o, ensure_ascii=False) + "\n")

# -------------------- 主入口 --------------------
def main():
    cfg = Config()

    # 在 main 里根据 cfg 实例化 tokenizer，方便日后换模型名
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    # 读取 dataset.py 产物（保持与你那边的保存路径一致）
    tr_path = output_path("output", "track_a", "subtask_1") / "train_pairs.parquet"
    dv_path = output_path("output", "track_a", "subtask_1") / "dev_pairs.parquet"
    assert tr_path.exists() and dv_path.exists(), f"找不到训练/开发数据：{tr_path} / {dv_path}"

    train_df = pd.read_parquet(tr_path)
    dev_df   = pd.read_parquet(dv_path)

    # 训练 + 保存最优权重
    ckpt_path = output_path("models", "task1", "best_model.pt")
    ckpt_path = fit(train_df, cfg, tokenizer, ckpt_path)

    # 加载最佳权重并推理 dev -> 导出提交文件
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = VARegressor(cfg.model_name, dropout=cfg.dropout, use_grad_ckpt=False)
    model.load_state_dict(ckpt["state_dict"]); model.to(device)

    preds = predict_dev(model, dev_df, cfg, tokenizer)
    out_file = output_path("submit", "task1", "pred_dev.jsonl")
    save_jsonl(preds, out_file)
    print("Dev 提交文件 ->", out_file)

if __name__ == "__main__":
    main()
