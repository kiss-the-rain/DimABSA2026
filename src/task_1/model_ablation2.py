# -*- coding: utf-8 -*-
"""
model_ablation.py
- 一个脚本跑 base / ablation（MeanPooling vs CLS, Huber vs MSE）
- 自动 tag：best_{tag}.pt, pred_{tag}_eng_laptop.jsonl, pred_{tag}_eng_restaurant.jsonl, eval_{tag}.json
- 推理后用官方 gold 离线评测（RMSE(VA) 同口径）
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 如需固定 GPU，可在命令行传 --cuda_visible_devices ；不在代码里硬编码

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import sys
import random
import math
from typing import Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    DebertaV2Tokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.utils.logging import set_verbosity_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# ====== 你的工程路径 ======
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import output_path, data_path


# ================= Config =================
@dataclass
class TrainConfig:
    local_model_dir: str
    max_len: int = 512

    lr_encoder: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.5
    warmup_ratio: float = 0.06
    warmup_min_steps: int = 500

    epochs: int = 20
    batch_size: int = 16
    freeze_epochs: int = 1
    enc_lr_after_unfreeze: float = 4.5e-6
    patience: int = 3

    seed: int = 42

    # ablation knobs
    pooling: str = "mean"     # "mean" | "cls"
    loss: str = "huber"       # "huber" | "mse"


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


def load_tokenizer(local_dir: Path):
    model_cfg = AutoConfig.from_pretrained(local_dir, local_files_only=True)
    if model_cfg.model_type in ("deberta-v2", "deberta-v3", "deberta"):
        return DebertaV2Tokenizer.from_pretrained(local_dir, local_files_only=True)
    return AutoTokenizer.from_pretrained(local_dir, use_fast=False, local_files_only=True)


def build_inputs(tok, text: str, aspect: str, max_len: int):
    prompt = f"Aspect: {aspect}"
    return tok(text, prompt, truncation="only_first", max_length=max_len)


# ================= Dataset / Collator =================
class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int, tok):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tok = tok

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        enc = build_inputs(self.tok, r["text"], r["aspect"], self.max_len)
        # labels: [1,9] -> [0,1]
        y = torch.tensor([(float(r["v"]) - 1.0) / 8.0, (float(r["a"]) - 1.0) / 8.0], dtype=torch.float32)
        enc["labels"] = y
        return enc


class DevDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int, tok):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tok = tok

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        enc = build_inputs(self.tok, r["text"], r["aspect"], self.max_len)
        enc["id"] = str(r["id"])
        enc["aspect_raw"] = r["aspect_raw"] if "aspect_raw" in self.df.columns else r["aspect"]
        return enc


class TrainCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        labels = [f.pop("labels") for f in features]
        batch = self.tok.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.stack(labels, dim=0)
        return batch


class DevCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        ids = [f.pop("id") for f in features]
        asps = [f.pop("aspect_raw") for f in features]
        batch = self.tok.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["id"] = ids
        batch["aspect_raw"] = asps
        return batch


# ================= Model =================
class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


class VARegressor(nn.Module):
    """
    pooling:
      - mean: MeanPooling(last_hidden_state, attention_mask)
      - cls : last_hidden_state[:, 0, :]
    """
    def __init__(self, encoder: AutoModel, dropout: float = 0.20, pooling: str = "mean"):
        super().__init__()
        self.enc = encoder
        self.pooling = pooling
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
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.pooling == "cls":
            feat = out.last_hidden_state[:, 0, :]
        else:
            feat = self.pool(out.last_hidden_state, attention_mask)
        logits = self.head(feat)
        pred01 = torch.sigmoid(logits)
        return pred01


# ================= Loss / Metric =================
def loss_fn(pred01, target01, loss_type: str):
    if loss_type == "huber":
        return F.smooth_l1_loss(pred01, target01)
    return F.mse_loss(pred01, target01)


def rmse_va(pred19, target19):
    # pred19/target19: [B,2] in [1,9]
    err2 = (pred19 - target19) ** 2
    per_sample = err2.sum(dim=1)
    return torch.sqrt(per_sample.mean())


def to_19(x01):
    return 1.0 + 8.0 * x01


# ================= Optim / Scheduler =================
def _build_param_groups(model, lr_enc, lr_head, wd):
    def no_decay(n):
        return ("bias" in n) or ("LayerNorm.weight" in n)

    enc_decay, enc_nodecay, head_decay, head_nodecay = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "enc." in n:
            (enc_nodecay if no_decay(n) else enc_decay).append(p)
        else:
            (head_nodecay if no_decay(n) else head_decay).append(p)

    return [
        {"params": enc_decay, "lr": lr_enc, "weight_decay": wd},
        {"params": enc_nodecay, "lr": lr_enc, "weight_decay": 0.0},
        {"params": head_decay, "lr": lr_head, "weight_decay": wd},
        {"params": head_nodecay, "lr": lr_head, "weight_decay": 0.0},
    ]


def build_optim_sched(model, num_train_steps, lr_enc, lr_head, wd, warmup_ratio, warmup_min_steps):
    optimizer = torch.optim.AdamW(_build_param_groups(model, lr_enc, lr_head, wd))
    num_warmup = max(warmup_min_steps, int(num_train_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_train_steps)
    return optimizer, scheduler


# ================= Train / Eval =================
def train_one_epoch(model, loader, optimizer, scheduler, device, loss_type: str, use_amp=True, max_grad_norm=1.0):
    model.train()
    scaler = GradScaler(enabled=use_amp)
    total_loss = total_rmse = 0.0
    n = 0
    amp_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float16

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ttids = batch.get("token_type_ids")
        ttids = ttids.to(device) if ttids is not None else None
        labels01 = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            pred01 = model(input_ids, attn, ttids)
            loss = loss_fn(pred01, labels01, loss_type)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        pred19 = to_19(pred01.detach())
        lab19 = to_19(labels01)
        rmse_b = rmse_va(pred19, lab19)

        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        total_rmse += rmse_b.item() * bs
        n += bs

    return total_loss / n, total_rmse / n


@torch.no_grad()
def evaluate(model, loader, device, loss_type_for_log: str = "mse"):
    """
    说明：RMSE 是官方口径，loss 仅用于你训练日志观察。
    默认保持你原版做法：val loss 用 MSE；如你想一致，可传 loss_type_for_log=cfg.loss
    """
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
        loss = loss_fn(pred01, labels01, loss_type_for_log)

        pred19 = to_19(pred01)
        lab19 = to_19(labels01)
        rmse_b = rmse_va(pred19, lab19)

        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        total_rmse += rmse_b.item() * bs
        n += bs

    return total_loss / n, total_rmse / n


# ================= Predict / Save =================
def format_va(v, a):
    # 官方要求：[1,9] + 两位小数
    v = float(max(1.0, min(9.0, v)))
    a = float(max(1.0, min(9.0, a)))
    return f"{v:.2f}#{a:.2f}"


@torch.no_grad()
def predict_dev(model, dev_df, tok, cfg: TrainConfig, device: torch.device):
    ds = DevDataset(dev_df, cfg.max_len, tok)
    num_workers = 4 if device.type == "cuda" else 0
    pin_mem = True if device.type == "cuda" else False

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=DevCollator(tok, pad_to_multiple_of=8),
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
        asps = batch["aspect_raw"]
        for i, (rid, asp) in enumerate(zip(ids, asps)):
            v, a = float(pred19[i][0]), float(pred19[i][1])
            records.append((rid, asp, v, a))

    bag = defaultdict(list)
    for rid, asp, v, a in records:
        bag[rid].append({"Aspect": asp, "VA": format_va(v, a)})

    seen = set()
    id_order = []
    for rid, _, _, _ in records:
        if rid not in seen:
            seen.add(rid)
            id_order.append(rid)

    lines = [{"ID": rid, "Aspect_VA": bag[rid]} for rid in id_order]
    return lines


def save_jsonl(objs, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for o in objs:
            w.write(json.dumps(o, ensure_ascii=False) + "\n")


# ================= Domain map (与你原版逻辑一致) =================
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


def _infer_domain_from_id(rid: str):
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
            path = data_path("track_a", "subtask_1", lang, f"{lang}_{domain_name}_test_task1.jsonl")
            domain_map.update(_load_domain_ids(path, domain_name))

    if domain_map:
        return domain_map

    if "id" not in dev_df.columns:
        return {}
    ids = dev_df["id"].astype(str).tolist()
    return {rid: d for rid in ids if (d := _infer_domain_from_id(rid)) is not None}


# ================= Official-like offline evaluation =================
def _parse_va_str(va: str) -> Tuple[float, float]:
    v_str, a_str = va.split("#")
    return float(v_str), float(a_str)


def _clip_round(x: float, lo: float = 1.0, hi: float = 9.0, nd: int = 2) -> float:
    x = max(lo, min(hi, x))
    return round(x, nd)


def _load_task1_jsonl_as_map(path: str) -> Dict[Tuple[str, str], Tuple[float, float]]:
    m: Dict[Tuple[str, str], Tuple[float, float]] = {}
    df = pd.read_json(path, lines=True)

    id_col = "ID" if "ID" in df.columns else "id"
    av_col = "Aspect_VA" if "Aspect_VA" in df.columns else "aspect_va"

    for _, row in df.iterrows():
        rid = str(row[id_col])
        av_list = row[av_col]
        if isinstance(av_list, str):
            av_list = json.loads(av_list)

        for item in av_list:
            asp = item["Aspect"]
            v, a = _parse_va_str(item["VA"])
            v = _clip_round(v)
            a = _clip_round(a)
            m[(rid, asp)] = (v, a)
    return m


def evaluate_task1_with_gold(pred_path: str, gold_path: str, strict: bool = False) -> Dict[str, float]:
    gold = _load_task1_jsonl_as_map(gold_path)
    pred = _load_task1_jsonl_as_map(pred_path)

    gold_keys = set(gold.keys())
    pred_keys = set(pred.keys())

    missing = sorted(list(gold_keys - pred_keys))
    extra = sorted(list(pred_keys - gold_keys))
    common = sorted(list(gold_keys & pred_keys))

    if strict and (missing or extra):
        raise ValueError(
            f"[STRICT] pred/gold keys mismatch. missing={len(missing)}, extra={len(extra)}. "
            f"Example missing={missing[:3]}, extra={extra[:3]}"
        )
    if len(common) == 0:
        raise ValueError("pred 与 gold 没有任何可对齐的 (ID, Aspect)。请检查输出格式/字段大小写/是否用对文件。")

    # RMSE(VA) = sqrt( (1/N) * sum( (dv)^2 + (da)^2 ) )
    sse = 0.0
    for k in common:
        pv, pa = pred[k]
        gv, ga = gold[k]
        dv = pv - gv
        da = pa - ga
        sse += dv * dv + da * da

    rmse = math.sqrt(sse / len(common))
    return {
        "rmse_va": rmse,
        "n_gold": float(len(gold_keys)),
        "n_pred": float(len(pred_keys)),
        "n_matched": float(len(common)),
        "missing": float(len(missing)),
        "extra": float(len(extra)),
    }


def pretty_print_report(name: str, rep: Dict[str, float]):
    print(f"\n===== Task1 Eval: {name} =====")
    print(f"RMSE(VA)   : {rep['rmse_va']:.6f}")
    print(f"Gold tuples: {int(rep['n_gold'])}")
    print(f"Pred tuples: {int(rep['n_pred'])}")
    print(f"Matched    : {int(rep['n_matched'])}")
    print(f"Missing    : {int(rep['missing'])}")
    print(f"Extra      : {int(rep['extra'])}")


# ================= Tag / CLI =================
def make_tag(cfg: TrainConfig) -> str:
    # 你也可以把更多超参编码进 tag（lr、seed 等），这里先把消融核心写进去
    return f"pool-{cfg.pooling}_loss-{cfg.loss}_seed-{cfg.seed}"


def parse_args():
    p = argparse.ArgumentParser("DimABSA Subtask1 - ablation runner + gold offline eval")
    p.add_argument("--local_model_dir", type=str, required=True, help="本地 HF 模型目录（相对 PROJECT_ROOT 或绝对路径）")

    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"], help="mean=MeanPooling, cls=[CLS]")
    p.add_argument("--loss", type=str, default="huber", choices=["huber", "mse"], help="huber=SmoothL1, mse=MSELoss")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=512)

    p.add_argument("--cuda_visible_devices", type=str, default=None, help="如 '0' 或 '1'，可覆盖 CUDA_VISIBLE_DEVICES")
    p.add_argument("--val_loss_type", type=str, default="mse", choices=["mse", "same"],
                   help="val loss 记录方式：mse=保持原版；same=与训练 loss 一致")
    p.add_argument("--strict_eval", action="store_true", help="严格对齐 pred/gold keys，不一致直接报错")

    return p.parse_args()


# ================= Main =================
def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    device = get_device()
    torch.set_float32_matmul_precision("high")
    set_verbosity_error()

    cfg = TrainConfig(
        local_model_dir=args.local_model_dir,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        pooling=args.pooling,
        loss=args.loss,
    )
    set_seed(cfg.seed)

    tag = make_tag(cfg)

    # === paths with tag ===
    ckpt_path = output_path("checkpoints", "task1", f"best_{tag}.pt")
    pred_lap_path = output_path("submit", "task1", f"pred_{tag}_eng_laptop.jsonl")
    pred_res_path = output_path("submit", "task1", f"pred_{tag}_eng_restaurant.jsonl")
    eval_path = output_path("submit", "task1", f"eval_{tag}.json")

    print(f"[TAG] {tag}")
    print(f"[CKPT] {ckpt_path}")
    print(f"[PRED] {pred_lap_path}")
    print(f"[PRED] {pred_res_path}")
    print(f"[EVAL] {eval_path}")
    print(f"[DEVICE] {device}")

    # resolve local model dir
    local_dir = Path(cfg.local_model_dir)
    if not local_dir.is_absolute():
        local_dir = (PROJECT_ROOT / local_dir).resolve()
    assert local_dir.exists(), f"本地模型目录不存在：{local_dir}"

    tok = load_tokenizer(local_dir)
    enc = AutoModel.from_pretrained(local_dir, local_files_only=True).to(device)

    # load train/dev pairs
    pair_dir = output_path("output", "track_a", "subtask_1")
    train_df = pd.read_parquet(pair_dir / "train_pairs.parquet")
    dev_df = pd.read_parquet(pair_dir / "dev_pairs.parquet")

    # split train/val
    if "id" in train_df.columns:
        groups = train_df["id"]
        gss = GroupShuffleSplit(test_size=0.1, random_state=cfg.seed)
        tr_idx, va_idx = next(gss.split(train_df, groups=groups))
        tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    else:
        tr_df, va_df = train_test_split(train_df, test_size=0.1, random_state=cfg.seed)

    num_workers = 4 if device.type == "cuda" else 0
    pin_mem = True if device.type == "cuda" else False

    tr_loader = DataLoader(
        TrainDataset(tr_df, cfg.max_len, tok),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=TrainCollator(tok, pad_to_multiple_of=8),
    )
    va_loader = DataLoader(
        TrainDataset(va_df, cfg.max_len, tok),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=TrainCollator(tok, pad_to_multiple_of=8),
    )

    # build model with ablation knobs
    model = VARegressor(enc, dropout=0.20, pooling=cfg.pooling).to(device)

    # optimizer/scheduler
    steps_total = len(tr_loader) * cfg.epochs
    optimizer, scheduler = build_optim_sched(
        model, steps_total, cfg.lr_encoder, cfg.lr_head, cfg.weight_decay, cfg.warmup_ratio, cfg.warmup_min_steps
    )

    # freeze encoder then unfreeze
    for p in model.enc.parameters():
        p.requires_grad = False

    best_rmse = 1e9
    bad = 0
    use_amp = (device.type != "cpu")
    val_loss_type = cfg.loss if args.val_loss_type == "same" else "mse"

    for ep in range(cfg.epochs):
        if ep == cfg.freeze_epochs:
            for p in model.enc.parameters():
                p.requires_grad = True
            steps_left = len(tr_loader) * (cfg.epochs - ep)
            optimizer, scheduler = build_optim_sched(
                model, steps_left, cfg.enc_lr_after_unfreeze, cfg.lr_head,
                cfg.weight_decay, cfg.warmup_ratio, cfg.warmup_min_steps
            )

        tl, tr = train_one_epoch(model, tr_loader, optimizer, scheduler, device, loss_type=cfg.loss, use_amp=use_amp)
        vl, vr = evaluate(model, va_loader, device, loss_type_for_log=val_loss_type)

        print(f"Epoch {ep+1:02d} | train_loss={tl:.4f} rmse={tr:.3f} || val_loss={vl:.4f} rmse={vr:.3f}")

        if vr < best_rmse:
            best_rmse, bad = vr, 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
            print(f"[BEST] saved -> {ckpt_path} (rmse={best_rmse:.3f})")
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[EARLY STOP] no improvement for {cfg.patience} epochs.")
                break

    # reload best ckpt for prediction
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    enc2 = AutoModel.from_pretrained(local_dir, local_files_only=True).to(device)
    model2 = VARegressor(enc2, dropout=0.20, pooling=cfg.pooling).to(device)
    model2.load_state_dict(ckpt["state_dict"])

    preds = predict_dev(model2, dev_df, tok, cfg, device)
    domain_map = build_domain_map(dev_df)
    if not domain_map:
        raise KeyError("dev_pairs.parquet 缺少 domain 字段，且无法从原始 dev JSONL 或 ID 推断 domain。")

    lines_lap = [p for p in preds if domain_map.get(p.get("ID")) == "laptop"]
    lines_res = [p for p in preds if domain_map.get(p.get("ID")) == "restaurant"]

    save_jsonl(lines_lap, Path(pred_lap_path))
    save_jsonl(lines_res, Path(pred_res_path))
    print(f"[WRITE] {pred_lap_path} (laptop: {len(lines_lap)})")
    print(f"[WRITE] {pred_res_path} (restaurant: {len(lines_res)})")

    # ===== Offline eval with official gold =====
    gold_lap = data_path("track_a", "subtask_1", "eng", "eng_laptop_test_gold.jsonl")
    gold_res = data_path("track_a", "subtask_1", "eng", "eng_restaurant_test_gold.jsonl")

    rep_lap = evaluate_task1_with_gold(str(pred_lap_path), str(gold_lap), strict=args.strict_eval)
    rep_res = evaluate_task1_with_gold(str(pred_res_path), str(gold_res), strict=args.strict_eval)
    pretty_print_report("eng-laptop", rep_lap)
    pretty_print_report("eng-restaurant", rep_res)

    overall_sse = (rep_lap["rmse_va"] ** 2) * rep_lap["n_matched"] + (rep_res["rmse_va"] ** 2) * rep_res["n_matched"]
    overall_n = rep_lap["n_matched"] + rep_res["n_matched"]
    overall_rmse = math.sqrt(overall_sse / overall_n)

    summary = {
        "tag": tag,
        "cfg": cfg.__dict__,
        "pred_laptop": str(pred_lap_path),
        "pred_restaurant": str(pred_res_path),
        "gold_laptop": str(gold_lap),
        "gold_restaurant": str(gold_res),
        "eval_laptop": rep_lap,
        "eval_restaurant": rep_res,
        "eval_overall_weighted_rmse_va": overall_rmse,
    }

    Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(eval_path).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n===== Task1 Overall (weighted) =====\nRMSE(VA): {overall_rmse:.6f}")
    print(f"[EVAL SAVED] {eval_path}\n")


if __name__ == "__main__":
    main()