# -*- coding: utf-8 -*-
"""
Task3 Ablation: w/o Span Recovery
- 基于你的 qwen3_task3_sft.py 改造（Trainer 训练逻辑保持一致）
- 增加 --no_span_recovery：extract_quadruplets_from_text 不再 recover_span（消融）
- 增加 “官方同口径”离线评测：cP/cR/cF1（continuous F1）
- 输出文件名自动加 tag，避免覆盖原文件
"""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import json
import random
import re
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, DatasetDict

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import data_path, output_path


# ===================== 配置 =====================

@dataclass
class Config:
    base_model: str = "/home/cuizhibin/projects/Models/Qwen3-4B-Instruct-2507-bnb-4bit"
    max_seq_len: int = 512

    use_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    epochs: int = 3
    train_batch_size: int = 4
    eval_batch_size: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 6e-5
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_workers: int = 10
    logging_steps: int = 50

    gen_max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 50

    dev_ratio: float = 0.1
    seed: int = 42

    output_dir: str = "output/qwen3_task3"


cfg = Config()


# ===================== 随机种子 / 设备 =====================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()


# ===================== 提示模板 =====================

SYSTEM_PROMPT = (
    "You are an expert system for Dimensional Aspect Sentiment Quad Prediction (DimASQP).\n"
    "Given a subjective sentence [Text], you must extract all opinion quadruplets.\n"
    "\n"
    "Each quadruplet has four fields:\n"
    "- Aspect (A): an aspect term explicitly mentioned in [Text];\n"
    "- Category (C): an aspect category label in the format ENTITY#ATTRIBUTE, written in UPPERCASE;\n"
    "- Opinion (O): an opinion expression about that aspect;\n"
    "- VA: a valence–arousal score in the format \"V#A\".\n"
    "\n"
    "Valence (V) and Arousal (A) are real numbers between 1.00 and 9.00.\n"
    "Both must be written with exactly two digits after the decimal point (e.g. 6.75#6.38).\n"
    "\n"
    "Output format requirement:\n"
    "- Output a single line starting with \"[Quadruplet]\".\n"
    "- Then list all quadruplets in the form: (Aspect, Category, Opinion, V#A).\n"
    "- Separate multiple quadruplets with comma and space.\n"
    "- If there is no valid quadruplet, output \"[Quadruplet]\" alone.\n"
)

USER_TEMPLATE = (
    "Below is a sentence.\n"
    "[Text] {text}\n\n"
    "Please extract ALL (Aspect, Category, Opinion, VA) quadruplets from [Text].\n"
    "Format your answer as:\n"
    "[Quadruplet] (aspect_1, CATEGORY_1#ATTRIBUTE_1, opinion_1, V1#A1), "
    "(aspect_2, CATEGORY_2#ATTRIBUTE_2, opinion_2, V2#A2), ...\n"
    "If no valid quadruplets can be found, just output \"[Quadruplet]\".\n"
)


# ===================== IO =====================

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)

def save_jsonl(objs: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for o in objs:
            w.write(json.dumps(o, ensure_ascii=False) + "\n")


# ===================== Span & VA 规范化 =====================

def recover_span(text: str, span: str) -> str:
    if not span:
        return span
    s = span.strip()
    if not s:
        return s
    text_lower = text.lower()
    s_lower = s.lower()
    idx = text_lower.find(s_lower)
    return text[idx: idx + len(s)] if idx != -1 else s

def normalize_va(va_str: str) -> str | None:
    m = re.match(r'^\s*([0-9.]+)\s*#\s*([0-9.]+)\s*$', va_str)
    if not m:
        return None
    try:
        v = float(m.group(1))
        a = float(m.group(2))
    except ValueError:
        return None
    v = min(9.0, max(1.0, v))
    a = min(9.0, max(1.0, a))
    return f"{v:.2f}#{a:.2f}"


# ===================== SFT 构建（与你原版一致） =====================

def _format_quad_answer(quads: List[Dict[str, str]]) -> str:
    parts = []
    for q in quads:
        parts.append(f"({q['Aspect']}, {q['Category']}, {q['Opinion']}, {q['VA']})")
    return "[Quadruplet] " + ", ".join(parts) if parts else "[Quadruplet]"

def build_sft_examples() -> List[Dict[str, Any]]:
    lp_train = data_path("track_a", "subtask_3", "eng", "eng_laptop_train_alltasks.jsonl")
    rs_train = data_path("track_a", "subtask_3", "eng", "eng_restaurant_train_alltasks.jsonl")

    examples: List[Dict[str, Any]] = []
    for path in [lp_train, rs_train]:
        assert Path(path).exists(), f"未找到训练文件: {path}"
        for obj in read_jsonl(Path(path)):
            sent_id = obj.get("ID", "")
            text = obj.get("Text", "")

            quads = obj.get("Quadruplet", []) or []
            new_quads = []
            for q in quads:
                asp = q.get("Aspect", "") or ""
                cat = q.get("Category", "") or ""
                opn = q.get("Opinion", "") or ""
                va  = q.get("VA", "") or ""
                if not (asp and cat and opn and va):
                    continue
                new_quads.append({"Aspect": asp, "Category": cat, "Opinion": opn, "VA": va})

            if not new_quads:
                continue

            target = _format_quad_answer(new_quads)
            user_content = USER_TEMPLATE.format(text=text)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
                {"role": "assistant", "content": target},
            ]
            examples.append({"id": sent_id, "text": text, "messages": messages, "target": target})
    return examples

def build_dataset_dict(cfg: Config) -> DatasetDict:
    examples = build_sft_examples()
    print(f"[INFO] 总训练样本数（两领域合并）: {len(examples)}")
    set_seed(cfg.seed)
    random.shuffle(examples)

    n_total = len(examples)
    n_dev = max(1, int(n_total * cfg.dev_ratio))
    dev_examples = examples[:n_dev]
    train_examples = examples[n_dev:]
    print(f"[INFO] 划分: train={len(train_examples)}, dev={len(dev_examples)}")
    return DatasetDict({"train": Dataset.from_list(train_examples), "dev": Dataset.from_list(dev_examples)})

def tokenize_sft_dataset(dsd: DatasetDict, tokenizer, cfg: Config) -> DatasetDict:
    def _preprocess(example):
        messages = example["messages"]
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        prompt_tok = tokenizer(prompt_text, max_length=cfg.max_seq_len, truncation=True, padding=False)
        full_tok = tokenizer(full_text, max_length=cfg.max_seq_len, truncation=True, padding=False)

        input_ids = full_tok["input_ids"]
        attention_mask = full_tok["attention_mask"]
        prompt_len = min(len(prompt_tok["input_ids"]), len(input_ids))

        labels = [-100] * prompt_len + input_ids[prompt_len:]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    out = DatasetDict()
    for split in dsd:
        d = dsd[split].map(_preprocess, remove_columns=dsd[split].column_names, desc=f"Tokenizing {split}")
        out[split] = d
    return out

def make_sft_collator(tokenizer):
    def _collate(batch):
        batch_for_pad = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch]
        padded = tokenizer.pad(batch_for_pad, return_tensors="pt")
        max_len = padded["input_ids"].shape[1]
        labels = []
        for b in batch:
            lab = b["labels"]
            lab = (lab + [-100] * (max_len - len(lab)))[:max_len]
            labels.append(lab)
        padded["labels"] = torch.tensor(labels, dtype=torch.long)
        return padded
    return _collate


# ===================== 模型加载 / 训练 =====================

def load_qwen3_and_tokenizer(cfg: Config):
    _ = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    tokenizer.padding_side = "right"

    if cfg.use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype, trust_remote_code=True)
        model.to(device)

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

def train_qwen3_task3(cfg: Config, tag: str) -> Path:
    set_seed(cfg.seed)
    dsd_raw = build_dataset_dict(cfg)
    model, tokenizer = load_qwen3_and_tokenizer(cfg)
    tokenizer.model_max_length = cfg.max_seq_len
    if not cfg.use_4bit:
        model.to(device)

    dsd = tokenize_sft_dataset(dsd_raw, tokenizer, cfg)
    train_ds, dev_ds = dsd["train"], dsd["dev"]

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and (not bf16)
    if cfg.use_4bit:
        bf16 = False
        fp16 = torch.cuda.is_available()

    training_kwargs = dict(
        output_dir=str(output_path(cfg.output_dir, f"trainer_{tag}")),
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )
    if "evaluation_strategy" in TrainingArguments.__init__.__code__.co_varnames:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"

    if "gradient_checkpointing" in TrainingArguments.__init__.__code__.co_varnames:
        training_kwargs["gradient_checkpointing"] = True

    args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=make_sft_collator(tokenizer),
        tokenizer=tokenizer,
    )
    trainer.train()

    out_dir = Path(output_path(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / f"qwen3_task3_lora_{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[INFO] LoRA 权重已保存到: {save_dir}")
    return save_dir


# ===================== 推理：Span Recovery 开关（消融核心） =====================

def load_lora_model(save_dir: Path, cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype, trust_remote_code=True)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = True

    model = PeftModel.from_pretrained(base_model, save_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def extract_quadruplets_from_text(raw: str, text: str, use_span_recovery: bool) -> List[Dict[str, str]]:
    pattern = r'\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, raw)

    quadruplets: List[Dict[str, str]] = []
    for aspect, category, opinion, va in matches:
        asp = aspect.strip()
        cat = category.strip()
        opn = opinion.strip()
        va_norm = normalize_va(va)
        if not (asp and cat and opn and va_norm):
            continue

        # ===== 消融核心：关闭 span recovery =====
        if use_span_recovery:
            asp = recover_span(text, asp)
            opn = recover_span(text, opn)
        else:
            asp = asp.strip()
            opn = opn.strip()

        quadruplets.append({"Aspect": asp, "Category": cat, "Opinion": opn, "VA": va_norm})
    return quadruplets


@torch.no_grad()
def predict_for_split(model, tokenizer, split_name: str, cfg: Config,
                      tag: str, use_span_recovery: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path]:
    lp_dev = data_path("track_a", "subtask_3", "eng", f"eng_laptop_{split_name}.jsonl")
    rs_dev = data_path("track_a", "subtask_3", "eng", f"eng_restaurant_{split_name}.jsonl")
    for path in [lp_dev, rs_dev]:
        assert Path(path).exists(), f"未找到输入文件: {path}"

    inputs: List[Dict[str, Any]] = []
    domains: List[str] = []
    for path in [lp_dev, rs_dev]:
        domain_name = "laptop" if "laptop" in str(path).lower() else "restaurant"
        for obj in read_jsonl(Path(path)):
            inputs.append(obj)
            domains.append(domain_name)

    debug_path = Path(output_path("debug", f"task3_raw_generations_{tag}.jsonl"))
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_f = debug_path.open("w", encoding="utf-8")

    lp_results, rs_results = [], []
    for idx, (obj, domain_name) in enumerate(zip(inputs, domains), 1):
        text_id = obj.get("ID", "")
        text = obj.get("Text", "")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=text)},
        ]

        enc = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if isinstance(enc, torch.Tensor):
            input_ids = enc.to(device)
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        gen_kwargs = dict(max_new_tokens=cfg.gen_max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        if cfg.temperature > 0.0:
            gen_kwargs.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)
        else:
            gen_kwargs.update(do_sample=False)

        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        gen_ids = out[0, input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        debug_f.write(json.dumps({"ID": text_id, "raw": gen_text}, ensure_ascii=False) + "\n")

        quads = extract_quadruplets_from_text(gen_text, text, use_span_recovery=use_span_recovery)
        line = {"ID": text_id, "Quadruplet": quads}
        (lp_results if domain_name == "laptop" else rs_results).append(line)

        if idx % 50 == 0 or idx == len(inputs):
            print(f"[PRED] {idx}/{len(inputs)}")

    debug_f.close()
    return lp_results, rs_results, debug_path


# ===================== 官方同口径：continuous F1 (cF1) 评测 =====================

D_MAX = math.sqrt(8.0 * 8.0 + 8.0 * 8.0)  # √128

def _parse_va(va: str) -> Tuple[float, float]:
    v_str, a_str = va.split("#")
    return float(v_str), float(a_str)

def _dist_norm(vp: float, ap: float, vg: float, ag: float) -> float:
    return math.sqrt((vp - vg) ** 2 + (ap - ag) ** 2) / D_MAX

def _key_quad(q: Dict[str, str]) -> Tuple[str, str, str]:
    return (q["Aspect"], q["Category"], q["Opinion"])

def _load_gold_or_pred_task3(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """
    {"ID": "...", "Quadruplet":[{"Aspect":..,"Category":..,"Opinion":..,"VA":..}, ...]}
    """
    mp: Dict[str, List[Dict[str, str]]] = {}
    for obj in read_jsonl(path):
        rid = str(obj.get("ID", ""))
        quads = obj.get("Quadruplet", []) or []
        out = []
        for q in quads:
            asp = (q.get("Aspect") or "").strip()
            cat = (q.get("Category") or "").strip()
            opn = (q.get("Opinion") or "").strip()
            va = (q.get("VA") or "").strip()
            va2 = normalize_va(va)
            if asp and cat and opn and va2:
                out.append({"Aspect": asp, "Category": cat, "Opinion": opn, "VA": va2})
        mp[rid] = out
    return mp

def eval_task3_cF1(pred_path: Path, gold_path: Path) -> Dict[str, float]:
    pred = _load_gold_or_pred_task3(pred_path)
    gold = _load_gold_or_pred_task3(gold_path)

    n_pred = 0
    n_gold = 0
    ctp_sum = 0.0

    for rid, gold_list in gold.items():
        pred_list = pred.get(rid, [])

        gmap: DefaultDict[Tuple[str, str, str], List[str]] = defaultdict(list)
        pmap: DefaultDict[Tuple[str, str, str], List[str]] = defaultdict(list)

        for g in gold_list:
            gmap[_key_quad(g)].append(g["VA"])
        for p in pred_list:
            pmap[_key_quad(p)].append(p["VA"])

        n_gold += len(gold_list)
        n_pred += len(pred_list)

        for key, gvas in gmap.items():
            pvas = pmap.get(key, [])
            if not pvas:
                continue
            unused = pvas[:]
            for gva in gvas:
                if not unused:
                    break
                gv, ga = _parse_va(gva)
                best_j = -1
                best_dist = 1e9
                for j, pva in enumerate(unused):
                    pv, pa = _parse_va(pva)
                    d = _dist_norm(pv, pa, gv, ga)
                    if d < best_dist:
                        best_dist = d
                        best_j = j
                ctp_sum += max(0.0, 1.0 - best_dist)
                unused.pop(best_j)

    cP = ctp_sum / max(1, n_pred)
    cR = ctp_sum / max(1, n_gold)
    cF1 = 0.0 if (cP + cR) == 0 else (2 * cP * cR / (cP + cR))
    return {"cP": cP, "cR": cR, "cF1": cF1, "n_pred": float(n_pred), "n_gold": float(n_gold), "cTP": ctp_sum}


# ===================== CLI / tag =====================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no_span_recovery", action="store_true", help="Ablation: disable span recovery for Aspect/Opinion.")
    p.add_argument("--split", type=str, default="test_task3", help="input split name like dev_task3/test_task3")
    p.add_argument("--gold_split", type=str, default="test_gold", help="gold split suffix: test_gold or dev_gold (if exists)")
    p.add_argument("--cuda_visible_devices", type=str, default=None)
    return p.parse_args()

def make_tag(args, cfg: Config) -> str:
    sr = "spanrec-off" if args.no_span_recovery else "spanrec-on"
    return f"task3_{sr}_seed-{cfg.seed}"

def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cfg = Config()
    set_seed(cfg.seed)
    tag = make_tag(args, cfg)

    # 1) train
    save_dir = train_qwen3_task3(cfg, tag)

    # 2) predict
    model, tokenizer = load_lora_model(save_dir, cfg)
    use_span_recovery = (not args.no_span_recovery)

    lp_pred, rs_pred, debug_path = predict_for_split(
        model, tokenizer, split_name=args.split, cfg=cfg, tag=tag, use_span_recovery=use_span_recovery
    )

    pred_lp_path = Path(output_path("submit", "task3", f"pred_{tag}_eng_laptop.jsonl"))
    pred_rs_path = Path(output_path("submit", "task3", f"pred_{tag}_eng_restaurant.jsonl"))
    save_jsonl(lp_pred, pred_lp_path)
    save_jsonl(rs_pred, pred_rs_path)
    print(f"[WRITE] {pred_lp_path}")
    print(f"[WRITE] {pred_rs_path}")
    print(f"[DEBUG] {debug_path}")

    # 3) eval with official gold
    gold_lp = Path(data_path("track_a", "subtask_3", "eng", f"eng_laptop_{args.gold_split}.jsonl"))
    gold_rs = Path(data_path("track_a", "subtask_3", "eng", f"eng_restaurant_{args.gold_split}.jsonl"))

    if gold_lp.exists() and gold_rs.exists():
        rep_lp = eval_task3_cF1(pred_lp_path, gold_lp)
        rep_rs = eval_task3_cF1(pred_rs_path, gold_rs)

        overall_ctp = rep_lp["cTP"] + rep_rs["cTP"]
        overall_pred = rep_lp["n_pred"] + rep_rs["n_pred"]
        overall_gold = rep_lp["n_gold"] + rep_rs["n_gold"]
        cP = overall_ctp / max(1.0, overall_pred)
        cR = overall_ctp / max(1.0, overall_gold)
        cF1 = 0.0 if (cP + cR) == 0 else (2 * cP * cR / (cP + cR))

        print("\n===== Task3 Official-like Eval (cF1) =====")
        print(f"[Laptop] cP={rep_lp['cP']:.4f} cR={rep_lp['cR']:.4f} cF1={rep_lp['cF1']:.4f}")
        print(f"[Rest  ] cP={rep_rs['cP']:.4f} cR={rep_rs['cR']:.4f} cF1={rep_rs['cF1']:.4f}")
        print(f"[OVERALL] cP={cP:.4f} cR={cR:.4f} cF1={cF1:.4f}")

        eval_path = Path(output_path("submit", "task3", f"eval_{tag}.json"))
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with eval_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"tag": tag, "split": args.split, "gold_split": args.gold_split,
                 "laptop": rep_lp, "restaurant": rep_rs, "overall": {"cP": cP, "cR": cR, "cF1": cF1}},
                f, ensure_ascii=False, indent=2
            )
        print(f"[EVAL SAVED] {eval_path}")
    else:
        print(f"[WARN] gold 文件不存在，跳过评测：{gold_lp} / {gold_rs}")


if __name__ == "__main__":
    main()