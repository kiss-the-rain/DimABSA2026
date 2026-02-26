from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import json, random, re
import numpy as np
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if torch.cuda.is_available():
    # 启用 TF32 提升矩阵乘速度（对 40 系通常有效）
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
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
    use_4bit: bool = True                # True = QLoRA (bitsandbytes), False = 常规 LoRA

    lora_r: int = 16                      # LoRA rank（低秩分解维度）
    lora_alpha: int = 32                  # LoRA 缩放系数
    lora_dropout: float = 0.05            # LoRA dropout

    # —— 训练超参 ——
    epochs: int = 3
    train_batch_size: int = 4             # per-device batch size
    eval_batch_size: int = 4
    grad_accum_steps: int = 1             # 梯度累积步数
    learning_rate: float = 6e-5
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_workers: int = 10
    logging_steps: int = 50

    # —— 解码 ——（预测 dev_task3 用）
    gen_max_new_tokens: int = 512          # Quad 输出不长，没必要 256
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 50

    # —— 数据 ——
    dev_ratio: float = 0.1
    seed: int = 42

    # —— 输出目录 ——
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


# ===================== 任务提示模板（Quadruplet 版：括号格式） =====================

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
    "\n"
    "Example:\n"
    "Input [Text]: average to good thai food, but terrible delivery.\n"
    "Output:\n"
    "[Quadruplet] (thai food, FOOD#QUALITY, average to good, 6.75#6.38), "
    "(delivery, SERVICE#GENERAL, terrible, 2.88#6.62)\n"
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


# ===================== 工具函数：读取 jsonl =====================

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ===================== 文本 span & VA 规范化 =====================

def recover_span(text: str, span: str) -> str:
    """
    尝试在原始 text 里恢复 span 的真实大小写形式：
    - 忽略大小写做一次查找；
    - 找到的话，用原文片段替换；找不到就原样返回 strip 后的 span。
    """
    if not span:
        return span
    s = span.strip()
    if not s:
        return s

    text_lower = text.lower()
    s_lower = s.lower()
    idx = text_lower.find(s_lower)
    if idx != -1:
        return text[idx: idx + len(s)]
    return s


def normalize_va(va_str: str) -> str | None:
    """
    把任意类似 '7.5#6.2' 的字符串规范成 '7.50#6.20'，
    并且裁剪到 [1.00, 9.00] 范围；解析失败返回 None。
    """
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


# ===================== 构建 SFT 样本（messages 形式） =====================

def _format_quad_answer(quads: List[Dict[str, str]]) -> str:
    """
    gold 四元组 -> 一行文本：
    [Quadruplet] (asp1, cat1, opn1, V1#A1), (asp2, cat2, opn2, V2#A2)
    """
    parts = []
    for q in quads:
        asp = q["Aspect"]
        cat = q["Category"]
        opn = q["Opinion"]
        va  = q["VA"]
        parts.append(f"({asp}, {cat}, {opn}, {va})")
    if parts:
        return "[Quadruplet] " + ", ".join(parts)
    else:
        return "[Quadruplet]"


def build_sft_examples() -> List[Dict[str, Any]]:
    """
    从 eng_*_train_alltasks.jsonl 里抽 Subtask3 的四元组，
    每个句子对应一条 SFT 样本（system + user + assistant）。
    """
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
                new_quads.append(
                    {"Aspect": asp, "Category": cat, "Opinion": opn, "VA": va}
                )

            # 没有四元组就不训练这句
            if len(new_quads) == 0:
                continue

            # gold 作为 assistant 回复（括号风格）
            target = _format_quad_answer(new_quads)

            user_content = USER_TEMPLATE.format(text=text)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
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


# ===================== 加载 Qwen3 + LoRA（不依赖 unsloth） =====================

def load_qwen3_and_tokenizer(cfg: Config):
    """
    用 transformers + peft 加载 Qwen3-1.7B，并挂 LoRA，不使用 unsloth。
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
        # 4bit 在部分环境下不支持 bf16 GEMM，默认用 fp16 更稳
        bnb_compute_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_compute_dtype = bnb_compute_dtype,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = "nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config = bnb_config,
            torch_dtype = torch.float16,
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

    model.config.use_cache = False   # 训练时关掉 cache
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
    labels 会把 prompt（system+user）部分屏蔽为 -100，仅训练 assistant 回复。
    """

    def _preprocess(example):
        messages = example["messages"]
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize = False,
            add_generation_prompt = True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False,
        )

        prompt_tok = tokenizer(
            prompt_text,
            max_length = cfg.max_seq_len,
            truncation = True,
            padding = False,
        )
        full_tok = tokenizer(
            full_text,
            max_length = cfg.max_seq_len,
            truncation = True,
            padding = False,
        )

        input_ids = full_tok["input_ids"]
        attention_mask = full_tok["attention_mask"]
        prompt_len = len(prompt_tok["input_ids"])
        if prompt_len > len(input_ids):
            prompt_len = len(input_ids)

        labels = [-100] * prompt_len + input_ids[prompt_len:]

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
        dsd_tok[split] = d

    return dsd_tok


def make_sft_collator(tokenizer):
    def _collate(batch):
        batch_for_pad = [
            {
                "input_ids": b["input_ids"],
                "attention_mask": b["attention_mask"],
            }
            for b in batch
        ]
        padded = tokenizer.pad(batch_for_pad, return_tensors="pt")
        max_len = padded["input_ids"].shape[1]

        labels = []
        for b in batch:
            lab = b["labels"]
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            labels.append(lab)
        padded["labels"] = torch.tensor(labels, dtype=torch.long)
        return padded
    return _collate


# ===================== Trainer 训练流程 =====================

def train_qwen3_task3(cfg: Config):
    set_seed(cfg.seed)

    # 1) 构建原始 DatasetDict（里面是 messages）
    dsd_raw = build_dataset_dict(cfg)

    # 2) 加载模型 & tokenizer（带 LoRA）
    model, tokenizer = load_qwen3_and_tokenizer(cfg)
    tokenizer.model_max_length = cfg.max_seq_len

    # 如果不是 4bit，这里统一放到 device 上
    if not cfg.use_4bit:
        model.to(device)

    # 3) 把 messages -> token 序列
    dsd = tokenize_sft_dataset(dsd_raw, tokenizer, cfg)

    train_ds = dsd["train"]
    dev_ds   = dsd["dev"]

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and (not bf16)
    if cfg.use_4bit:
        bf16 = False
        fp16 = torch.cuda.is_available()

    training_kwargs = dict(
        output_dir = str(output_path(cfg.output_dir, "trainer")),
        per_device_train_batch_size = cfg.train_batch_size,
        per_device_eval_batch_size = cfg.eval_batch_size,
        gradient_accumulation_steps = cfg.grad_accum_steps,
        num_train_epochs = cfg.epochs,
        learning_rate = cfg.learning_rate,
        warmup_ratio = cfg.warmup_ratio,
        weight_decay = cfg.weight_decay,
        max_grad_norm = cfg.max_grad_norm,
        logging_steps = cfg.logging_steps,
        save_strategy = "epoch",
        save_total_limit = 2,
        bf16 = bf16,
        fp16 = fp16,
        report_to = "none",
        remove_unused_columns = False,
        dataloader_pin_memory = True,
    )
    if "evaluation_strategy" in TrainingArguments.__init__.__code__.co_varnames:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"
    if "gradient_checkpointing" in TrainingArguments.__init__.__code__.co_varnames:
        training_kwargs["gradient_checkpointing"] = True
    if "group_by_length" in TrainingArguments.__init__.__code__.co_varnames:
        training_kwargs["group_by_length"] = True
    if "dataloader_num_workers" in TrainingArguments.__init__.__code__.co_varnames:
        training_kwargs["dataloader_num_workers"] = cfg.num_workers

    training_args = TrainingArguments(**training_kwargs)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    print("==((====))==  Trainer SFT (Task3, NO unsloth)")
    print(f"   Num examples = {len(train_ds)}, Num epochs = {cfg.epochs}")
    print(f"   Batch per device = {cfg.train_batch_size}, "
          f"Grad accum steps = {cfg.grad_accum_steps}")

    trainer_kwargs = dict(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = dev_ds,
        tokenizer = tokenizer,
        data_collator = make_sft_collator(tokenizer),
    )
    if "processing_class" in Trainer.__init__.__code__.co_varnames:
        trainer_kwargs.pop("tokenizer", None)
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    # 5) 保存 LoRA 权重（adapter）
    out_dir = Path(output_path(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / "qwen3_task3_lora"
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_dir)      # 这里只保存 LoRA adapter
    tokenizer.save_pretrained(save_dir)  # tokenizer 也存一份，方便推理时直接用
    print(f"[INFO] LoRA 权重已保存到: {save_dir}")

    return save_dir


# ===================== 推理：括号 -> Quadruplet =====================

def load_lora_model(save_dir: Path, cfg: Config):
    """
    加载基础 Qwen3-1.7B + 已训练好的 LoRA 适配器（不依赖 unsloth）。
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
    base_model.config.use_cache = True      # 推理时打开 KV cache 加速

    model = PeftModel.from_pretrained(base_model, save_dir)

    model.to(device)
    model.eval()
    return model, tokenizer


def extract_quadruplets_from_text(raw: str, text: str) -> List[Dict[str, str]]:
    """
    用正则在任意字符串里抽取 (Aspect, Category, Opinion, V#A) 形式的四元组。
    形如：
    [Quadruplet] (thai food, FOOD#QUALITY, average to good, 6.75#6.38),
                 (delivery, SERVICE#GENERAL, terrible, 2.88#6.62)
    """
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

        # 在原文中恢复 Aspect / Opinion 的大小写
        asp_rec = recover_span(text, asp)
        opn_rec = recover_span(text, opn)

        quadruplets.append(
            {
                "Aspect": asp_rec,
                "Category": cat,   # Category 本身就是 schema label，不从 text 截取
                "Opinion": opn_rec,
                "VA": va_norm,
            }
        )

    return quadruplets


@torch.no_grad()
def predict_for_split(model, tokenizer, split_name: str, cfg: Config):
    """
    对 A 组英文 dev_task3 做预测。
    输出 JSONL，每行：
    {"ID": "...", "Quadruplet": [ {"Aspect": "...", "Category": "...",
                                  "Opinion": "...", "VA": "V#A"}, ... ]}
    """
    lp_dev = data_path("track_a", "subtask_3", "eng", f"eng_laptop_{split_name}.jsonl")
    rs_dev = data_path("track_a", "subtask_3", "eng", f"eng_restaurant_{split_name}.jsonl")

    for path in [lp_dev, rs_dev]:
        assert Path(path).exists(), f"未找到 dev 文件: {path}"

    inputs: List[Dict[str, Any]] = []
    input_domains: List[str] = []
    for path in [lp_dev, rs_dev]:
        domain_name = "laptop" if "laptop" in str(path).lower() else "restaurant"
        for obj in read_jsonl(Path(path)):
            inputs.append(obj)
            input_domains.append(domain_name)

    print(f"[INFO] 预测样本数: {len(inputs)}")

    results: List[Dict[str, Any]] = []

    # 调试：把原始生成结果也存一下
    debug_path = Path(output_path("debug", "task3_raw_generations.jsonl"))
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_f = debug_path.open("w", encoding="utf-8")

    for idx, (obj, domain_name) in enumerate(zip(inputs, input_domains), 1):
        text_id = obj.get("ID", "")
        text = obj.get("Text", "")

        user_content = USER_TEMPLATE.format(text=text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        # 这里我们自己构造 attention_mask，避免 pad=EOS 的 warning
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
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

        out = model.generate(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            **gen_kwargs,
        )

        gen_ids  = out[0, input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # 调试写一份原始输出
        debug_obj = {"ID": text_id, "raw": gen_text}
        debug_f.write(json.dumps(debug_obj, ensure_ascii=False) + "\n")

        # 用正则抽取四元组 + span / VA 规范化
        quadruplets = extract_quadruplets_from_text(gen_text, text=text)

        results.append({
            "ID": text_id,
            "Quadruplet": quadruplets,
            "domain": domain_name,
        })

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
    save_dir = train_qwen3_task3(cfg)

    # 2) 加载 LoRA + 做 dev_task3 预测
    model, tokenizer = load_lora_model(save_dir, cfg)

    preds = predict_for_split(model, tokenizer, split_name="test_task3", cfg=cfg)

    for domain_name, file_name in [
        ("laptop", "pred_eng_laptop.jsonl"),
        ("restaurant", "pred_eng_restaurant.jsonl"),
    ]:
        domain_preds = [
            {"ID": p["ID"], "Quadruplet": p["Quadruplet"]}
            for p in preds if p.get("domain") == domain_name
        ]
        out_path = Path(output_path("submit", "task3", file_name))
        save_jsonl(domain_preds, out_path)
        print(f"[INFO] Dev 提交文件已生成: {out_path} ({domain_name}: {len(domain_preds)})")


if __name__ == "__main__":
    main()
