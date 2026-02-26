# -*- coding: utf-8 -*-
# LLM 版：本地加载 HuggingFace LLM（不访问外网），保留原有训练/早停/预测流程
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from dataclasses import dataclass
from pathlib import Path
import random, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from collections import defaultdict
from transformers.utils.logging import set_verbosity_error

from src.utils.paths import output_path


# ================= 配置 =================
@dataclass
class Config:
    local_model_dir: str = "models/roberta-base"

    max_len: int = 256

    # 优化与正则（LLM 更大，可以把 encoder lr 设小一点）
    lr_encoder: float = 5e-6
    lr_head: float = 1e-4
    weight_decay: float = 0.02
    warmup_ratio: float = 0.06
    warmup_min_steps: int = 500

    # 训练策略（LLM 训练成本高，先保守一点）
    epochs: int = 10
    batch_size: int = 8
    freeze_epochs: int = 1
    enc_lr_after_unfreeze: float = 3e-6
    patience: int = 2
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


# ================= 分词与输入 =================
def build_inputs(tok, text: str, aspect: str, max_len: int):
    """
    对 LLM 更友好一点的指令式 prompt。
    这里只做编码，不做 padding/tensor 化，保持和原先 collator 设计一致。
    """
    prompt = (
        "You are an expert for dimensional sentiment analysis.\n"
        f"Text: {text}\n"
        f"Aspect: {aspect}\n"
        "Please understand the sentiment towards the aspect."
    )
    return tok(
        prompt,
        truncation=True,
        max_length=max_len,
    )


# ================= 数据集 =================
class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int, tok):
        """
        df：包含 text, aspect, v, a 列
        tok：HuggingFace tokenizer
        """
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tok = tok

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        enc = build_inputs(self.tok, r["text"], r["aspect"], self.max_len)

        # 标签从 [1,9] 缩放到 [0,1]
        y = torch.tensor(
            [
                (float(r["v"]) - 1.0) / 8.0,
                (float(r["a"]) - 1.0) / 8.0,
            ],
            dtype=torch.float32,
        )
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
class TrainCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # 1) 取出 labels，避免被 tokenizer.pad 处理
        labels = [f.pop("labels") for f in features]

        # 2) 对输入做动态 padding
        batch = self.tok.pad(
            features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # 3) 把 labels 堆叠回去
        batch["labels"] = torch.stack(labels, dim=0)
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
            return_tensors="pt",
        )
        batch["id"] = ids
        batch["aspect"] = asps
        return batch


# ================= 模型 =================
class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: [B, L, H]
        # attention_mask:    [B, L]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        # padding 位置为 0，有效 token 为 1
        summed = (last_hidden_state * mask).sum(1)  # [B, H]
        counts = mask.sum(1).clamp(min=1e-6)        # [B, 1]
        return summed / counts


class VARegressor(nn.Module):
    def __init__(self, encoder: AutoModel, dropout: float = 0.20):
        super().__init__()
        self.enc = encoder
        hidden = self.enc.config.hidden_size
        self.pool = MeanPooling()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 对于 BERT 类模型可以传 token_type_ids；
        # 对于大多数 decoder-only LLM，必须去掉 token_type_ids。
        if token_type_ids is not None:
            out = self.enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            out = self.enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        feat = self.pool(out.last_hidden_state, attention_mask)
        logits = self.head(feat)
        pred01 = torch.sigmoid(logits)  # 映射到 [0,1]
        return pred01


# ================= 损失与指标 =================
def loss_fn(pred01, target01, huber=False):
    return (
        F.smooth_l1_loss(pred01, target01)
        if huber
        else F.mse_loss(pred01, target01)
    )


def rmse_va(pred, target):
    err2 = (pred - target) ** 2     # [B, 2]
    per_sample = err2.sum(dim=1)    # [B]
    mse = per_sample.mean()
    return torch.sqrt(mse)


def _to_19(x01):
    # [0,1] -> [1,9]
    return 1.0 + 8.0 * x01


# ================= 优化器与调度器 =================
def _build_param_groups(model, lr_enc, lr_head, wd):
    """
    更健壮的分组方式：按“参数属于不属于 model.enc”来区分 encoder / head。
    """
    def no_decay(name: str) -> bool:
        name = name.lower()
        return (
            "bias" in name
            or "layernorm.weight" in name
            or "layer_norm.weight" in name
            or "rmsnorm.weight" in name
        )

    enc_params = list(model.enc.parameters())
    enc_param_ids = {id(p) for p in enc_params}

    enc_decay, enc_nodecay, head_decay, head_nodecay = [], [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # 判断是不是 encoder 参数
        is_enc = id(p) in enc_param_ids
        if is_enc:
            if no_decay(name):
                enc_nodecay.append(p)
            else:
                enc_decay.append(p)
        else:
            if no_decay(name):
                head_nodecay.append(p)
            else:
                head_decay.append(p)

    return [
        {"params": enc_decay, "lr": lr_enc, "weight_decay": wd},
        {"params": enc_nodecay, "lr": lr_enc, "weight_decay": 0.0},
        {"params": head_decay, "lr": lr_head, "weight_decay": wd},
        {"params": head_nodecay, "lr": lr_head, "weight_decay": 0.0},
    ]


def build_optim_sched(
    model,
    num_train_steps,
    lr_enc,
    lr_head,
    wd,
    warmup_ratio,
    warmup_min_steps,
):
    param_groups = _build_param_groups(model, lr_enc, lr_head, wd)
    optimizer = torch.optim.AdamW(param_groups)

    num_warmup = max(warmup_min_steps, int(num_train_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup, num_train_steps
    )
    return optimizer, scheduler


# ================= 训练 / 验证 =================
def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    use_amp=True,
    huber=False,
    max_grad_norm=1.0,
):
    model.train()
    scaler = GradScaler(enabled=use_amp)
    total_loss = total_rmse = 0.0
    n = 0

    amp_dtype = (
        torch.float16 if device.type in ("cuda", "mps") else torch.float32
    )

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ttids = batch.get("token_type_ids")
        ttids = ttids.to(device) if ttids is not None else None
        labels01 = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(
            device_type=device.type, enabled=use_amp, dtype=amp_dtype
        ):
            pred01 = model(input_ids, attn, ttids)
            loss = loss_fn(pred01, labels01, huber=huber)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # 统计 loss / RMSE
        pred19 = _to_19(pred01.detach())
        lab19 = _to_19(labels01)
        rmse_b = rmse_va(pred19, lab19)
        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        total_rmse += rmse_b.item() * bs
        n += bs

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
        labels01 = batch["labels"].to(device)

        pred01 = model(input_ids, attn, ttids)
        loss = loss_fn(pred01, labels01)

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

    num_workers = 4 if device.type == "cuda" else 0
    pin_mem = True if device.type == "cuda" else False

    collate = DevCollator(tok, pad_to_multiple_of=8)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=collate,
    )

    model.eval()
    records = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ttids = batch.get("token_type_ids")
        ttids = ttids.to(device) if ttids is not None else None

        pred01 = model(input_ids, attn, ttids).cpu().numpy()
        pred19 = 1.0 + 8.0 * pred01

        ids = batch["id"]
        asps = batch["aspect"]
        for i, (vid, asp) in enumerate(zip(ids, asps)):
            v, a = float(pred19[i][0]), float(pred19[i][1])
            records.append((vid, asp, v, a))

    bag = defaultdict(list)
    for rid, asp, v, a in records:
        bag[rid].append({"Aspect": asp, "VA": format_va(v, a)})

    lines = [
        {"ID": rid, "Aspect_VA": items} for rid, items in bag.items()
    ]
    return lines


def save_jsonl(objs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for o in objs:
            w.write(json.dumps(o, ensure_ascii=False) + "\n")


# ================= 主流程 =================
def main():
    print("ROOT ->", os.getenv("DIMABSA_ROOT"))
    print("EXPECT ->", output_path("output", "track_a", "subtask_1"))

    cfg = Config()
    set_seed(cfg.seed)
    set_verbosity_error()

    # 1) 离线加载 tokenizer 与 LLM
    local_dir = (Path(__file__).resolve().parents[2] / cfg.local_model_dir).resolve()
    assert local_dir.exists(), f"本地模型目录不存在：{local_dir}"

    tok = AutoTokenizer.from_pretrained(
        local_dir,
        use_fast=False,          # 很多 LLM 没有 fast tokenizer
        local_files_only=True,
        trust_remote_code=True,  # Qwen/LLaMA 等常用
    )

    # 很多 LLM 没有 pad_token，需要手动指定为 eos_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    enc = AutoModel.from_pretrained(
        local_dir,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)

    # 2) 读取 dataset.py 预处理后的数据
    data_dir = output_path("output", "track_a", "subtask_1")
    train_df = pd.read_parquet(data_dir / "train_pairs.parquet")
    dev_df = pd.read_parquet(data_dir / "dev_pairs.parquet")

    # 3) 分组切分（按同一句子的 id 分组）
    if "id" in train_df.columns:
        groups = train_df["id"]
        gss = GroupShuffleSplit(test_size=0.1, random_state=cfg.seed)
        tr_idx, va_idx = next(gss.split(train_df, groups=groups))
        tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    else:
        tr_df, va_df = train_test_split(
            train_df, test_size=0.1, random_state=cfg.seed
        )

    # 4) DataLoader
    num_workers = 4 if device.type == "cuda" else 0
    pin_mem = True if device.type == "cuda" else False

    tr_ds = TrainDataset(tr_df, cfg.max_len, tok)
    va_ds = TrainDataset(va_df, cfg.max_len, tok)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=TrainCollator(tok, pad_to_multiple_of=8),
    )

    va_loader = DataLoader(
        va_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=TrainCollator(tok, pad_to_multiple_of=8),
    )

    # 5) 构建模型/优化器/调度器
    model = VARegressor(enc, dropout=0.20).to(device)
    steps_total = len(tr_loader) * cfg.epochs
    optimizer, scheduler = build_optim_sched(
        model,
        steps_total,
        cfg.lr_encoder,
        cfg.lr_head,
        cfg.weight_decay,
        cfg.warmup_ratio,
        cfg.warmup_min_steps,
    )

    # 6) 训练循环：先冻结 encoder，再解冻
    for p in model.enc.parameters():
        p.requires_grad = False

    best_rmse, best_path = 1e9, "best_model.pt"
    use_amp = device.type != "cpu"
    bad = 0

    for ep in range(cfg.epochs):
        # 到达解冻轮次，重新构建优化器/调度器
        if ep == cfg.freeze_epochs:
            for p in model.enc.parameters():
                p.requires_grad = True

            steps_left = len(tr_loader) * (cfg.epochs - ep)
            optimizer, scheduler = build_optim_sched(
                model,
                steps_left,
                cfg.enc_lr_after_unfreeze,
                cfg.lr_head,
                cfg.weight_decay,
                cfg.warmup_ratio,
                cfg.warmup_min_steps,
            )

        tl, tr = train_one_epoch(
            model,
            tr_loader,
            optimizer,
            scheduler,
            device,
            use_amp=use_amp,
            huber=cfg.use_huber,
        )
        vl, vr = evaluate(model, va_loader, device)
        print(
            f"Epoch {ep + 1:02d} | "
            f"train_loss={tl:.4f} rmse={tr:.3f} || "
            f"val_loss={vl:.4f} rmse={vr:.3f}"
        )

        if vr < best_rmse:
            best_rmse, bad = vr, 0
            torch.save(
                {"state_dict": model.state_dict(), "cfg": cfg.__dict__},
                best_path,
            )
            print(f"[BEST] saved -> {best_path} (rmse={best_rmse:.3f})")
        else:
            bad += 1
            if bad >= cfg.patience:
                print(
                    f"[EARLY STOP] no improvement for {cfg.patience} epochs."
                )
                break

    # 7) 推理（开发集）
    try:
        ckpt = torch.load("best_model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load("best_model.pt", map_location="cpu")

    # 推理阶段重新构建 encoder 和 VARegressor，保持和训练时相同结构
    enc2 = AutoModel.from_pretrained(
        local_dir,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    model = VARegressor(enc2, dropout=0.20).to(device)
    model.load_state_dict(ckpt["state_dict"])

    preds = predict_dev(model, dev_df, tok, cfg)
    out_file = output_path("submit", "task1", "pred_dev.jsonl")
    save_jsonl(preds, out_file)
    print("Dev 提交文件 ->", out_file)


if __name__ == "__main__":
    main()
