import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import json, random, math, os, re
import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

# 你项目里的路径工具
from src.utils.paths import data_path, output_path


# ===================== 配置 =====================

@dataclass
class Config:
    # —— 模型与 LoRA ——
    base_model: str = "/home/cuizhibin/projects/Models/Qwen3-4B-Instruct-2507-bnb-4bit"
    max_seq_len: int = 512                # 指令 + 文本 + 标签 的最大长度

    # 如果你想先用 FP16/BF16 全精度，不想折腾 4bit，设为 False 即可
    use_4bit: bool = True                 # True = QLoRA (bitsandbytes), False = 常规 LoRA

    lora_r: int = 16                       # LoRA rank（低秩分解维度）
    lora_alpha: int = 32                   # LoRA 缩放系数
    lora_dropout: float = 0.05             # LoRA dropout

    # —— 训练超参 ——
    epochs: int = 3
    train_batch_size: int = 8              # per-device batch size
    eval_batch_size: int = 4
    grad_accum_steps: int = 4              # 梯度累积步数
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    logging_steps: int = 50

    # —— 解码 ——（预测 dev_task2 用）
    gen_max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 50

    # —— 数据 ——
    dev_ratio: float = 0.1
    seed: int = 42

    # —— 输出目录 ——
    output_dir: str = "output/qwen3_task2"


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


# ===================== 工具函数：读取 jsonl =====================

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ===================== 构建 SFT 样本（messages 形式） =====================

def _format_triplet_answer(triplets: List[Dict[str, str]]) -> str:
    """
    把 gold triplets 变成一行文本：
    [Triplet] (asp1, opn1, V1#A1), (asp2, opn2, V2#A2)
    """
    parts = []
    for t in triplets:
        asp = t["Aspect"]
        opn = t["Opinion"]
        va  = t["VA"]
        parts.append(f"({asp}, {opn}, {va})")
    if parts:
        return "[Triplet] " + ", ".join(parts)
    else:
        return "[Triplet]"


def build_sft_examples() -> List[Dict[str, Any]]:
    """
    从 eng_*_train_alltasks.jsonl 里抽 Subtask2 的三元组，
    每个句子对应一条 SFT 样本（system + user + assistant）。
    """
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
                triplets.append(
                    {"Aspect": asp, "Opinion": opn, "VA": va}
                )

            # 没有三元组就不训练这句
            if len(triplets) == 0:
                continue

            # gold 作为 assistant 回复（括号风格）
            target = _format_triplet_answer(triplets)

            user_content = USER_TEMPLATE.format(text=text)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target},
            ]

            examples.append(
                {
                    "id": sent_id,
                    "text": text,
                    "messages": messages,
                    "target": target,
                }
            )

    return examples


def build_dataset_dict(cfg: Config) -> DatasetDict:
    """
    返回一个 DatasetDict:
    - train: 90%
    - dev  : 10%
    """
    examples = build_sft_examples()
    print(f"[INFO] 总训练样本数（两领域合并）: {len(examples)}")

    set_seed(cfg.seed)
    random.shuffle(examples)

    n_total = len(examples)
    n_dev = max(1, int(n_total * cfg.dev_ratio))
    dev_examples = examples[:n_dev]
    train_examples = examples[n_dev:]

    print(f"[INFO] 划分: train={len(train_examples)}, dev={len(dev_examples)}")

    ds_train = Dataset.from_list(train_examples)
    ds_dev   = Dataset.from_list(dev_examples)
    return DatasetDict({"train": ds_train, "dev": ds_dev})


# ===================== 加载 Qwen3 + LoRA =====================

def load_qwen3_and_tokenizer(cfg: Config):
    """
    用 transformers + peft 加载 Qwen3-1.7B，并挂 LoRA。
    """

    # 读一下 config，确认本地模型没问题
    _ = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True, local_files_only=True)

    # 1) 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
    )

    # Qwen3 默认没有 pad_token，这里用 eos_token 兜底
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"

    # 2) 加载 base model
    if cfg.use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_compute_dtype = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = "nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config = bnb_config,
            device_map = "auto",
            trust_remote_code = True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype = dtype,
            trust_remote_code = True,
        )
        model.to(device)

    model.config.use_cache = False   # 训练时关闭 cache
    model.config.pad_token_id = tokenizer.pad_token_id

    # 3) LoRA 配置
    lora_config = LoraConfig(
        r = cfg.lora_r,
        lora_alpha = cfg.lora_alpha,
        lora_dropout = cfg.lora_dropout,
        bias = "none",
        task_type = "CAUSAL_LM",
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ===================== 文本 -> token 序列（SFT 预处理） =====================

def tokenize_sft_dataset(dsd: DatasetDict, tokenizer, cfg: Config) -> DatasetDict:
    """
    把 messages 用 chat_template 展开成字符串，再用 tokenizer 编码，
    得到 input_ids / attention_mask / labels。
    labels 中 padding 位置为 -100。
    """

    def _preprocess(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False,
        )
        tok = tokenizer(
            text,
            max_length = cfg.max_seq_len,
            truncation = True,
            padding = "max_length",
        )
        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]

        labels = []
        for tid, m in zip(input_ids, attention_mask):
            if m == 0:
                labels.append(-100)
            else:
                labels.append(tid)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dsd_tok = DatasetDict()
    for split in dsd:
        d = dsd[split].map(
            _preprocess,
            remove_columns = dsd[split].column_names,
            desc = f"Tokenizing {split}",
        )
        d.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        dsd_tok[split] = d

    return dsd_tok


# ===================== 手写训练循环 =====================

def train_qwen3_task2(cfg: Config):
    set_seed(cfg.seed)

    # 1) 构建原始 DatasetDict（里面是 messages）
    dsd_raw = build_dataset_dict(cfg)

    # 2) 加载模型 & tokenizer（带 LoRA）
    model, tokenizer = load_qwen3_and_tokenizer(cfg)

    # 如果不是 4bit，这里统一放到 device 上
    if not cfg.use_4bit:
        model.to(device)

    # 3) 把 messages -> token 序列
    dsd = tokenize_sft_dataset(dsd_raw, tokenizer, cfg)

    train_ds = dsd["train"]
    dev_ds   = dsd["dev"]

    train_loader = DataLoader(
        train_ds,
        batch_size = cfg.train_batch_size,
        shuffle = True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size = cfg.eval_batch_size,
        shuffle = False,
    )

    # 4) 优化器 & scheduler（只训练 LoRA 参数）
    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        opt_params,
        lr = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    max_train_steps = cfg.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(cfg.warmup_ratio * max_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = num_warmup_steps,
        num_training_steps = max_train_steps,
    )

    print("==((====))==  Manual SFT training loop (NO unsloth)")
    print(f"   Num examples = {len(train_ds)}, Num epochs = {cfg.epochs}, "
          f"Total steps = {max_train_steps}")
    print(f"   Batch per device = {cfg.train_batch_size}, "
          f"Grad accum steps = {cfg.grad_accum_steps}")

    global_step = 0
    model.train()

    for epoch in range(cfg.epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                labels = batch["labels"],
            )
            loss = outputs.loss

            # 梯度累积
            loss = loss / cfg.grad_accum_steps
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

        # 每个 epoch 后简单做下 dev loss
        model.eval()
        dev_loss, dev_count = 0.0, 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    labels = batch["labels"],
                )
                dev_loss += outputs.loss.item() * batch["input_ids"].size(0)
                dev_count += batch["input_ids"].size(0)
        avg_dev = dev_loss / max(1, dev_count)
        print(f"[DEV] Epoch {epoch+1} | avg loss = {avg_dev:.4f}")
        model.train()

    # 5) 保存 LoRA 权重（adapter）
    out_dir = Path(output_path(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / "qwen3_task2_lora"
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_dir)      # 这里只保存 LoRA adapter
    tokenizer.save_pretrained(save_dir)  # tokenizer 也存一份，方便推理时直接用
    print(f"[INFO] LoRA 权重已保存到: {save_dir}")

    return save_dir


# ===================== 推理 =====================

def load_lora_model(save_dir: Path, cfg: Config):
    """
    加载基础 Qwen3-1.7B + 已训练好的 LoRA 适配器。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype = dtype,
        trust_remote_code = True,
    )

    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, save_dir)

    # 推理阶段打开 KV cache，加速生成
    model.config.use_cache = True

    model.to(device)
    model.eval()
    return model, tokenizer


def extract_triplets_from_text(raw: str) -> List[Dict[str, str]]:
    """
    用正则在任意字符串里抽取 (Aspect, Opinion, V#A) 形式的三元组。
    e.g. [Triplet] (thai food, average to good, 6.75#6.38), (delivery, terrible, 2.88#6.62)
    """
    pattern = r'\(([^,]+),\s*([^,]+),\s*([\d.]+#[\d.]+)\)'
    matches = re.findall(pattern, raw)

    triplets: List[Dict[str, str]] = []
    for aspect, opinion, va in matches:
        triplets.append({
            "Aspect": aspect.strip(),
            "Opinion": opinion.strip(),
            "VA": va.strip(),
        })
    return triplets


def recover_span(orig_text: str, predicted_span: str) -> str:
    """
    使用原始句子 orig_text，恢复 predicted_span 在原文中的 substring，
    保留原始大小写和空格。

    参数:
    - orig_text: 原始句子，例如 "Battery life is bad enough as it is"
    - predicted_span: 模型输出的 span，例如 "battery life"（可能是小写）

    返回:
    - 在 orig_text 中找到的 substring，例如 "Battery life"
      找不到则返回 strip 后的 predicted_span，避免崩溃。
    """
    if not predicted_span:
        return predicted_span

    orig_lower = orig_text.lower()
    span_lower = predicted_span.lower().strip()

    idx = orig_lower.find(span_lower)
    if idx == -1:
        # 找不到就先原样返回，你也可以在这里打 log 调试
        return predicted_span.strip()

    return orig_text[idx: idx + len(span_lower)]


@torch.no_grad()
def predict_for_split(model, tokenizer, split_name: str, cfg: Config):
    """
    对 A 组英文 dev_task2 做预测。
    输出 JSONL：
    {"ID": "...", "Triplet": [ {"Aspect": "...", "Opinion": "...", "VA": "V#A"}, ... ]}
    """
    # 1) 读官方 dev 文件
    lp_dev = data_path("track_a", "subtask_2", "eng", f"eng_laptop_{split_name}.jsonl")
    rs_dev = data_path("track_a", "subtask_2", "eng", f"eng_restaurant_{split_name}.jsonl")

    for path in [lp_dev, rs_dev]:
        assert Path(path).exists(), f"未找到 dev 文件: {path}"

    inputs: List[Dict[str, Any]] = []
    for path in [lp_dev, rs_dev]:
        for obj in read_jsonl(Path(path)):
            inputs.append(obj)

    print(f"[INFO] 预测样本数: {len(inputs)}")

    results: List[Dict[str, Any]] = []

    # 调试文件：记录模型原始输出
    debug_path = Path(output_path("debug", "task2_raw_generations.jsonl"))
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_f = debug_path.open("w", encoding="utf-8")

    for idx, obj in enumerate(inputs, 1):
        text_id = obj.get("ID", "")
        text = obj.get("Text", "")

        # ===== 构造对话 =====
        user_content = USER_TEMPLATE.format(text=text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        # ===== 编码 prompt，拿 input_ids + attention_mask =====
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
            return_attention_mask = True,
        )

        if isinstance(enc, torch.Tensor):
            input_ids = enc.to(device)
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = enc["input_ids"].to(device)
            if "attention_mask" in enc:
                attention_mask = enc["attention_mask"].to(device)
            else:
                attention_mask = torch.ones_like(input_ids)

        # ===== 生成参数 =====
        gen_kwargs = dict(
            max_new_tokens = cfg.gen_max_new_tokens,
            pad_token_id   = tokenizer.eos_token_id,
        )
        if cfg.temperature > 0.0:
            gen_kwargs.update(
                do_sample   = True,
                temperature = cfg.temperature,
                top_p       = cfg.top_p,
                top_k       = cfg.top_k,
            )
        else:
            gen_kwargs.update(do_sample = False)

        # ===== 调用 generate =====
        out = model.generate(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            **gen_kwargs,
        )

        # 只取新生成部分
        gen_ids  = out[0, input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # —— 调试：把原始输出存下来 —— #
        debug_obj = {"ID": text_id, "raw": gen_text}
        debug_f.write(json.dumps(debug_obj, ensure_ascii=False) + "\n")

        # ========== 正则抽取三元组 ==========
        raw_triplets = extract_triplets_from_text(gen_text)

        fixed_triplets: List[Dict[str, str]] = []
        for t in raw_triplets:
            asp_raw = t.get("Aspect", "")
            opn_raw = t.get("Opinion", "")
            va_raw  = t.get("VA", "")

            # 用原句 text 恢复 Aspect/Opinion 的原始大小写与 span
            asp_fix = recover_span(text, asp_raw)
            opn_fix = recover_span(text, opn_raw)

            # 规范 VA：防止越界，统一到两位小数
            va_fix = va_raw
            try:
                v_str, a_str = va_raw.split("#")
                v = float(v_str)
                a = float(a_str)
                v = min(9.0, max(1.0, v))
                a = min(9.0, max(1.0, a))
                va_fix = f"{v:.2f}#{a:.2f}"
            except Exception:
                # 解析失败就用原值
                va_fix = va_raw

            fixed_triplets.append(
                {
                    "Aspect": asp_fix,
                    "Opinion": opn_fix,
                    "VA": va_fix,
                }
            )

        results.append({"ID": text_id, "Triplet": fixed_triplets})

        if idx % 20 == 0 or idx == len(inputs):
            print(f"[PRED] 已完成 {idx}/{len(inputs)} 样本")

    debug_f.close()
    print(f"[DEBUG] 原始生成结果已写入: {debug_path}")
    return results


def save_jsonl(objs: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for o in objs:
            w.write(json.dumps(o, ensure_ascii=False) + "\n")


# ===================== main =====================

def main():
    set_seed(cfg.seed)

    # 1) 训练 LoRA
    save_dir = train_qwen3_task2(cfg)

    # 2) 加载 LoRA + 做 dev_task2 预测
    model, tokenizer = load_lora_model(save_dir, cfg)

    preds = predict_for_split(model, tokenizer, split_name="dev_task2", cfg=cfg)

    out_path = Path(output_path("submit", "task2", "qwen3_task2_dev_pred.jsonl"))
    save_jsonl(preds, out_path)
    print("[INFO] Dev 提交文件已生成:", out_path)


if __name__ == "__main__":
    main()
