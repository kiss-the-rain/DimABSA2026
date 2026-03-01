# -*- coding: utf-8 -*-
"""
Task2 Ablation: w/o Span Recovery
- 基于你的 qwen3_task2_base.py 改造（训练循环不变）
- 增加 --no_span_recovery：推理时不再 recover_span（消融）
- 增加 “官方同口径”离线评测：cP/cR/cF1（continuous F1）
- 输出文件名自动加 tag，避免覆盖原文件
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import json
import random
import math
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_linear_schedule_with_warmup
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
    train_batch_size: int = 2
    eval_batch_size: int = 2
    grad_accum_steps: int = 4
    learning_rate: float = 8e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    logging_steps: int = 50
    num_workers: int = 10

    gen_max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 50

    dev_ratio: float = 0.1
    seed: int = 42

    output_dir: str = "output/qwen3_task2"


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


# ===================== 任务提示模板 =====================

SYSTEM_PROMPT = (
    "You are an expert system for Dimensional Aspect Sentiment Analysis.\n"
    "Given a subjective sentence [Text], you must extract all opinion triplets.\n"
    "\n"
    "Each triplet has three fields:\n"
    "- Aspect: an aspect term (a phrase describing an entity explicitly mentioned in [Text]);\n"
    "- Opinion: an opinion expression about that aspect;\n"
    "- VA: a valence–arousal score in the format \"V#A\".\n"
    "\n"
    "Valence (V) ranges from 1.00 (very negative) to 9.00 (very positive).\n"
    "Arousal (A) ranges from 1.00 (very calm) to 9.00 (very excited).\n"
    "Both V and A must be decimals with two digits after the decimal point.\n"
    "\n"
    "Output format requirement:\n"
    "- Output a single line starting with \"[Triplet]\".\n"
    "- Then list all triplets in the form: (Aspect, Opinion, V#A).\n"
    "- Separate multiple triplets with comma and space.\n"
    "- If there is no valid triplet, output \"[Triplet]\" alone.\n"
    "\n"
    "Example:\n"
    "Input [Text]: average to good thai food, but terrible delivery.\n"
    "Output:\n"
    "[Triplet] (thai food, average to good, 6.75#6.38), (delivery, terrible, 2.88#6.62)\n"
)

USER_TEMPLATE = (
    "Below is a sentence.\n"
    "[Text] {text}\n\n"
    "Please extract ALL (Aspect, Opinion, VA) triplets from [Text].\n"
    "Format your answer as:\n"
    "[Triplet] (aspect_1, opinion_1, V1#A1), (aspect_2, opinion_2, V2#A2), ...\n"
    "If no valid triplets can be found, just output \"[Triplet]\".\n"
)


# ===================== IO 工具 =====================

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


# ===================== 训练数据构建（与你原版一致） =====================

def _format_triplet_answer(triplets: List[Dict[str, str]]) -> str:
    parts = []
    for t in triplets:
        parts.append(f"({t['Aspect']}, {t['Opinion']}, {t['VA']})")
    return "[Triplet] " + ", ".join(parts) if parts else "[Triplet]"


def build_sft_examples() -> List[Dict[str, Any]]:
    lp_train = data_path("track_a", "subtask_2", "eng", "eng_laptop_train_alltasks.jsonl")
    rs_train = data_path("track_a", "subtask_2", "eng", "eng_restaurant_train_alltasks.jsonl")

    examples: List[Dict[str, Any]] = []
    for path in [lp_train, rs_train]:
        assert Path(path).exists(), f"未找到训练文件: {path}"

        for obj in read_jsonl(Path(path)):
            sent_id = obj.get("ID", "")
            text = obj.get("Text", "")

            quads = obj.get("Quadruplet", []) or []
            triplets = []
            for q in quads:
                asp = q.get("Aspect", "") or ""
                opn = q.get("Opinion", "") or ""
                va  = q.get("VA", "") or ""
                if not asp or not opn or not va:
                    continue
                triplets.append({"Aspect": asp, "Opinion": opn, "VA": va})

            if not triplets:
                continue

            target = _format_triplet_answer(triplets)
            user_content = USER_TEMPLATE.format(text=text)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
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
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tok = tokenizer(text, max_length=cfg.max_seq_len, truncation=True, padding="max_length")

        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]
        labels = [tid if m == 1 else -100 for tid, m in zip(input_ids, attention_mask)]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    out = DatasetDict()
    for split in dsd:
        d = dsd[split].map(_preprocess, remove_columns=dsd[split].column_names, desc=f"Tokenizing {split}")
        d.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        out[split] = d
    return out


# ===================== 加载 Qwen3 + LoRA（与你原版一致） =====================

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
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
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


def train_qwen3_task2(cfg: Config, tag: str) -> Path:
    set_seed(cfg.seed)
    dsd_raw = build_dataset_dict(cfg)
    model, tokenizer = load_qwen3_and_tokenizer(cfg)
    if not cfg.use_4bit:
        model.to(device)

    dsd = tokenize_sft_dataset(dsd_raw, tokenizer, cfg)
    train_ds, dev_ds = dsd["train"], dsd["dev"]

    num_workers = max(0, int(cfg.num_workers))
    pin_mem = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem, persistent_workers=(num_workers > 0)
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem, persistent_workers=(num_workers > 0)
    )

    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(opt_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    max_train_steps = cfg.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(cfg.warmup_ratio * max_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
    )

    print("==((====))==  Manual SFT training loop (NO unsloth)")
    print(f"   Num examples = {len(train_ds)}, Num epochs = {cfg.epochs}, Total steps = {max_train_steps}")

    global_step = 0
    model.train()

    for epoch in range(cfg.epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / cfg.grad_accum_steps
            loss.backward()
            running_loss += loss.item()

            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(opt_params, max_norm=cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.logging_steps == 0:
                    avg_loss = running_loss / cfg.logging_steps
                    print(f"[Epoch {epoch+1}] step {global_step}/{max_train_steps} | loss={avg_loss:.4f}")
                    running_loss = 0.0

        model.eval()
        dev_loss, dev_count = 0.0, 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                dev_loss += outputs.loss.item() * batch["input_ids"].size(0)
                dev_count += batch["input_ids"].size(0)
        print(f"[DEV] Epoch {epoch+1} | avg loss = {dev_loss / max(1, dev_count):.4f}")
        model.train()

    out_dir = Path(output_path(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / f"qwen3_task2_lora_{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
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
    model = PeftModel.from_pretrained(base_model, save_dir)
    model.config.use_cache = True
    model.to(device)
    model.eval()
    return model, tokenizer


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


def recover_span(orig_text: str, predicted_span: str) -> str:
    if not predicted_span:
        return predicted_span
    orig_lower = orig_text.lower()
    span_lower = predicted_span.lower().strip()
    idx = orig_lower.find(span_lower)
    return predicted_span.strip() if idx == -1 else orig_text[idx: idx + len(span_lower)]


def extract_triplets_from_text(raw: str) -> List[Dict[str, str]]:
    # 仍按你原正则格式抽取
    pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, raw)
    triplets = []
    for aspect, opinion, va in matches:
        triplets.append({"Aspect": aspect.strip(), "Opinion": opinion.strip(), "VA": va.strip()})
    return triplets


@torch.no_grad()
def predict_for_split(model, tokenizer, split_name: str, cfg: Config,
                      tag: str, use_span_recovery: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path]:
    lp_dev = data_path("track_a", "subtask_2", "eng", f"eng_laptop_{split_name}.jsonl")
    rs_dev = data_path("track_a", "subtask_2", "eng", f"eng_restaurant_{split_name}.jsonl")
    for path in [lp_dev, rs_dev]:
        assert Path(path).exists(), f"未找到输入文件: {path}"

    inputs: List[Tuple[Dict[str, Any], str]] = []
    for domain_name, path in [("laptop", lp_dev), ("restaurant", rs_dev)]:
        for obj in read_jsonl(Path(path)):
            inputs.append((obj, domain_name))

    debug_path = Path(output_path("debug", f"task2_raw_generations_{tag}.jsonl"))
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_f = debug_path.open("w", encoding="utf-8")

    lp_results, rs_results = [], []
    for idx, (obj, domain_name) in enumerate(inputs, 1):
        text_id = obj.get("ID", "")
        text = obj.get("Text", "")

        user_content = USER_TEMPLATE.format(text=text)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]

        enc = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
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

        raw_triplets = extract_triplets_from_text(gen_text)

        fixed_triplets: List[Dict[str, str]] = []
        for t in raw_triplets:
            asp_raw = t.get("Aspect", "")
            opn_raw = t.get("Opinion", "")
            va_raw = t.get("VA", "")

            # ===== 消融核心：关闭 span recovery =====
            if use_span_recovery:
                asp = recover_span(text, asp_raw)
                opn = recover_span(text, opn_raw)
            else:
                asp = asp_raw.strip()
                opn = opn_raw.strip()

            va_norm = normalize_va(va_raw)  # 仍保留 VA 规范化（不是 span recovery）
            if not (asp and opn and va_norm):
                continue

            fixed_triplets.append({"Aspect": asp, "Opinion": opn, "VA": va_norm})

        line = {"ID": text_id, "Triplet": fixed_triplets}
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

def _key_triplet(t: Dict[str, str]) -> Tuple[str, str]:
    return (t["Aspect"], t["Opinion"])

def _load_gold_or_pred_task2(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """
    返回 dict: id -> list of triplets
    gold/pred 的结构都是：
      {"ID": "...", "Triplet":[{"Aspect":..,"Opinion":..,"VA":..}, ...]}
    """
    mp: Dict[str, List[Dict[str, str]]] = {}
    for obj in read_jsonl(path):
        rid = str(obj.get("ID", ""))
        trips = obj.get("Triplet", []) or []
        out = []
        for t in trips:
            asp = (t.get("Aspect") or "").strip()
            opn = (t.get("Opinion") or "").strip()
            va = (t.get("VA") or "").strip()
            va2 = normalize_va(va)
            if asp and opn and va2:
                out.append({"Aspect": asp, "Opinion": opn, "VA": va2})
        mp[rid] = out
    return mp

def eval_task2_cF1(pred_path: Path, gold_path: Path) -> Dict[str, float]:
    pred = _load_gold_or_pred_task2(pred_path)
    gold = _load_gold_or_pred_task2(gold_path)

    # counts
    n_pred = 0
    n_gold = 0
    ctp_sum = 0.0

    # 对每个样本：按 categorical key=(Aspect,Opinion) 做匹配；同 key 多个时按最小 dist 贪心配对
    for rid, gold_list in gold.items():
        pred_list = pred.get(rid, [])

        gmap: DefaultDict[Tuple[str, str], List[str]] = defaultdict(list)
        pmap: DefaultDict[Tuple[str, str], List[str]] = defaultdict(list)

        for g in gold_list:
            gmap[_key_triplet(g)].append(g["VA"])
        for p in pred_list:
            pmap[_key_triplet(p)].append(p["VA"])

        n_gold += len(gold_list)
        n_pred += len(pred_list)

        # categorical match 才有 cTP
        for key, gvas in gmap.items():
            pvas = pmap.get(key, [])
            if not pvas:
                continue

            # 贪心：对每个 gold，找 dist 最小的 pred
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
                # 1 - dist
                ctp_sum += max(0.0, 1.0 - best_dist)
                unused.pop(best_j)

    cP = ctp_sum / max(1, n_pred)
    cR = ctp_sum / max(1, n_gold)
    cF1 = 0.0 if (cP + cR) == 0 else (2 * cP * cR / (cP + cR))
    return {"cP": cP, "cR": cR, "cF1": cF1, "n_pred": float(n_pred), "n_gold": float(n_gold), "cTP": ctp_sum}


# ===================== CLI / tag =====================

def make_tag(args, cfg: Config) -> str:
    # 只把影响实验的关键因素编码进 tag（你也可以把 lr/epoch 等加进去）
    sr = "spanrec-off" if args.no_span_recovery else "spanrec-on"
    return f"task2_{sr}_seed-{cfg.seed}"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no_span_recovery", action="store_true", help="Ablation: disable span recovery for Aspect/Opinion.")
    p.add_argument("--split", type=str, default="test_task2", help="input split name like dev_task2/test_task2")
    p.add_argument("--gold_split", type=str, default="test_gold", help="gold split suffix: test_gold or dev_gold (if exists)")
    p.add_argument("--cuda_visible_devices", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cfg = Config()
    set_seed(cfg.seed)
    tag = make_tag(args, cfg)

    # 1) train
    save_dir = train_qwen3_task2(cfg, tag)

    # 2) predict
    model, tokenizer = load_lora_model(save_dir, cfg)

    use_span_recovery = (not args.no_span_recovery)
    lp_pred, rs_pred, debug_path = predict_for_split(
        model, tokenizer, split_name=args.split, cfg=cfg, tag=tag, use_span_recovery=use_span_recovery
    )

    pred_lp_path = Path(output_path("submit", "task2", f"pred_{tag}_eng_laptop.jsonl"))
    pred_rs_path = Path(output_path("submit", "task2", f"pred_{tag}_eng_restaurant.jsonl"))
    save_jsonl(lp_pred, pred_lp_path)
    save_jsonl(rs_pred, pred_rs_path)
    print(f"[WRITE] {pred_lp_path}")
    print(f"[WRITE] {pred_rs_path}")
    print(f"[DEBUG] {debug_path}")

    # 3) eval with official gold
    gold_lp = Path(data_path("track_a", "subtask_2", "eng", f"eng_laptop_{args.gold_split}.jsonl"))
    gold_rs = Path(data_path("track_a", "subtask_2", "eng", f"eng_restaurant_{args.gold_split}.jsonl"))
    if gold_lp.exists() and gold_rs.exists():
        rep_lp = eval_task2_cF1(pred_lp_path, gold_lp)
        rep_rs = eval_task2_cF1(pred_rs_path, gold_rs)

        # overall by pooling counts/ctp
        overall_ctp = rep_lp["cTP"] + rep_rs["cTP"]
        overall_pred = rep_lp["n_pred"] + rep_rs["n_pred"]
        overall_gold = rep_lp["n_gold"] + rep_rs["n_gold"]
        cP = overall_ctp / max(1.0, overall_pred)
        cR = overall_ctp / max(1.0, overall_gold)
        cF1 = 0.0 if (cP + cR) == 0 else (2 * cP * cR / (cP + cR))

        print("\n===== Task2 Official-like Eval (cF1) =====")
        print(f"[Laptop] cP={rep_lp['cP']:.4f} cR={rep_lp['cR']:.4f} cF1={rep_lp['cF1']:.4f}")
        print(f"[Rest  ] cP={rep_rs['cP']:.4f} cR={rep_rs['cR']:.4f} cF1={rep_rs['cF1']:.4f}")
        print(f"[OVERALL] cP={cP:.4f} cR={cR:.4f} cF1={cF1:.4f}")

        eval_path = Path(output_path("submit", "task2", f"eval_{tag}.json"))
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