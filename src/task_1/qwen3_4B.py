# -*- coding: utf-8 -*-
# LLM 版：本地加载 HuggingFace LLM（不访问外网），保留原有训练/早停/预测流程
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
import random, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from collections import defaultdict
from transformers.utils.logging import set_verbosity_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import output_path
from datasets import Dataset as HFDataset

try:
    from transformers import EarlyStoppingCallback
except Exception:
    EarlyStoppingCallback = None


# ================= 配置 =================
@dataclass
class Config:
    # 改成你的 Qwen3-4B 本地目录（建议放在 models/ 下）
    local_model_dir: str = "/home/cuizhibin/projects/Models/Qwen3-4B-Instruct-2507-bnb-4bit"

    max_len: int = 512

    # 优化与正则（LLM 更大，可以把 encoder lr 设小一点）
    lr_encoder: float = 5e-6
    lr_head: float = 1e-4
    weight_decay: float = 0.02
    warmup_ratio: float = 0.06
    warmup_min_steps: int = 500

    # 训练策略（LLM 训练成本高，先保守一点）
    epochs: int = 4
    batch_size: int = 16
    freeze_epochs: int = 1          # Trainer 简化版不再使用
    enc_lr_after_unfreeze: float = 3e-6  # Trainer 简化版不再使用
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
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True


# ================= 分词与输入 =================
def build_inputs(tok, text: str, aspect: str, max_len: int):
    """
    对 LLM 更友好一点的指令式 prompt。
    这里只做编码，不做 padding/tensor 化，保持和原先 collator 设计一致。
    """
    system_prompt = "You are an expert for dimensional sentiment analysis."
    user_prompt = (
        f"Text: {text}\n"
        f"Aspect: {aspect}\n"
        "Please predict the valence and arousal for the aspect."
    )
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        prompt = system_prompt + "\n" + user_prompt
    return tok(
        prompt,
        truncation=True,
        max_length=max_len,
    )


# ================= 数据集（HF Datasets） =================
def build_hf_dataset(df: pd.DataFrame, tok, max_len: int, with_labels: bool):
    df = df.reset_index(drop=True)
    ds = HFDataset.from_pandas(df)

    def _tokenize(batch):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        has_ttids = False

        for text, aspect in zip(batch["text"], batch["aspect"]):
            enc = build_inputs(tok, text, aspect, max_len)
            input_ids.append(enc["input_ids"])
            attention_mask.append(enc["attention_mask"])
            if "token_type_ids" in enc:
                token_type_ids.append(enc["token_type_ids"])
                has_ttids = True

        outputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if has_ttids:
            outputs["token_type_ids"] = token_type_ids

        if with_labels:
            labels = [
                [(float(v) - 1.0) / 8.0, (float(a) - 1.0) / 8.0]
                for v, a in zip(batch["v"], batch["a"])
            ]
            outputs["labels"] = labels

        return outputs

    return ds.map(_tokenize, batched=True, remove_columns=ds.column_names)


class LabelCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.base = DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def __call__(self, features):
        labels = [f.pop("labels") for f in features if "labels" in f]
        batch = self.base(features)
        if labels:
            batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


# ================= 模型 =================
# ================= 模型 =================
class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-6)
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

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
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
        pred01 = torch.sigmoid(logits)
        return {"logits": pred01}


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
    更健壮的分组方式：按“参数属于不属于 base_model”来区分 encoder / head。
    """
    def no_decay(name: str) -> bool:
        name = name.lower()
        return (
            "bias" in name
            or "layernorm.weight" in name
            or "layer_norm.weight" in name
            or "rmsnorm.weight" in name
        )

    base_model = getattr(model, "base_model", None)
    if base_model is None:
        base_model = getattr(model, "model", None)
    if base_model is None:
        base_model = model

    enc_params = list(base_model.parameters())
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


# ================= Trainer 训练 =================
class VATrainer(Trainer):
    def __init__(self, *args, huber=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_huber = huber
        # 旧版本 Trainer 不支持 label_names 参数，手动指定
        if hasattr(self, "label_names"):
            self.label_names = ["labels"]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        if isinstance(outputs, dict):
            pred01 = outputs.get("logits")
        else:
            pred01 = outputs
        if labels is not None:
            labels = labels.to(pred01.dtype)
        loss = loss_fn(pred01, labels, huber=self.use_huber)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = torch.as_tensor(preds)
    labels = torch.as_tensor(labels)
    if preds.ndim > 2:
        preds = preds[..., 0]
    rmse = rmse_va(_to_19(preds), _to_19(labels)).item()
    return {"rmse": rmse}


# ================= 推理与保存 =================
def format_va(v, a):
    v = float(max(1.0, min(9.0, v)))
    a = float(max(1.0, min(9.0, a)))
    return f"{v:.2f}#{a:.2f}"


@torch.no_grad()
def predict_dev(trainer: Trainer, dev_df: pd.DataFrame, tok, cfg: Config):
    pred_ids = dev_df["id"].astype(str).tolist()
    pred_asps = dev_df["aspect"].tolist()

    pred_ds = build_hf_dataset(dev_df, tok, cfg.max_len, with_labels=False)
    outputs = trainer.predict(pred_ds)
    pred01 = outputs.predictions
    if isinstance(pred01, (tuple, list)):
        pred01 = pred01[0]

    pred19 = 1.0 + 8.0 * pred01

    records = []
    for vid, asp, (v, a) in zip(pred_ids, pred_asps, pred19):
        records.append((vid, asp, float(v), float(a)))

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
    tok.padding_side = "right"

    is_4bit = "4bit" in str(local_dir).lower() or "bnb" in str(local_dir).lower()
    enc_dtype = torch.float16 if is_4bit else (
        torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    load_kwargs = dict(
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=enc_dtype,
    )
    if is_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = None
        load_kwargs["low_cpu_mem_usage"] = False
    else:
        load_kwargs["device_map"] = None
        load_kwargs["low_cpu_mem_usage"] = False

    enc = AutoModel.from_pretrained(
        local_dir,
        **load_kwargs,
    )
    enc = enc.to(device)

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

    # 4) Dataset / Collator
    num_workers = 8 if device.type == "cuda" else 0
    pin_mem = True if device.type == "cuda" else False

    tr_ds = build_hf_dataset(tr_df, tok, cfg.max_len, with_labels=True)
    va_ds = build_hf_dataset(va_df, tok, cfg.max_len, with_labels=True)
    collator = LabelCollator(tok, pad_to_multiple_of=8)

    # 5) 构建模型（4bit 时默认冻结 encoder，避免梯度不稳定）
    model = VARegressor(enc, dropout=0.20).to(device)
    if is_4bit:
        for p in model.enc.parameters():
            p.requires_grad = False

    steps_per_epoch = math.ceil(len(tr_ds) / cfg.batch_size)
    num_update_steps = math.ceil(steps_per_epoch / 1) * cfg.epochs
    num_warmup = max(cfg.warmup_min_steps, int(num_update_steps * cfg.warmup_ratio))

    optimizer = torch.optim.AdamW(
        _build_param_groups(model, cfg.lr_encoder, cfg.lr_head, cfg.weight_decay)
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup, num_update_steps
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and (not bf16)

    training_kwargs = dict(
        output_dir=str(output_path("output", "track_a", "subtask_1", "trainer")),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr_head,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=pin_mem,
        dataloader_num_workers=num_workers,
        seed=cfg.seed,
    )
    if "evaluation_strategy" in TrainingArguments.__init__.__code__.co_varnames:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"
    if "dataloader_persistent_workers" in TrainingArguments.__init__.__code__.co_varnames:
        training_kwargs["dataloader_persistent_workers"] = num_workers > 0
    if "prediction_loss_only" in TrainingArguments.__init__.__code__.co_varnames:
        training_kwargs["prediction_loss_only"] = False

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tr_ds,
        eval_dataset=va_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )
    if "processing_class" in Trainer.__init__.__code__.co_varnames:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok

    callbacks = []
    if EarlyStoppingCallback is not None:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.patience)
        )
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = VATrainer(huber=cfg.use_huber, **trainer_kwargs)
    trainer.train()

    best_path = "best_model.pt"
    torch.save(
        {"state_dict": trainer.model.state_dict(), "cfg": cfg.__dict__},
        best_path,
    )

    # 7) 推理（开发集）
    preds = predict_dev(trainer, dev_df, tok, cfg)
    out_file = output_path("submit", "task1", "pred_dev.jsonl")
    save_jsonl(preds, out_file)
    print("Dev 提交文件 ->", out_file)


if __name__ == "__main__":
    main()
