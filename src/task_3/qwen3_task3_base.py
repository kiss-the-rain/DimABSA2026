from __future__ import annotations
# ===================== FIX 1: Unsloth 必须放在最前面 =====================
from unsloth import FastLanguageModel, is_bfloat16_supported

import os
import re
import json
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from datasets import Dataset
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer, SFTConfig

# ===================== 1. 路径配置 =====================
try:
    from src.utils.paths import data_path, output_path
except ImportError:
    def data_path(*args):
        return os.path.join("data", *args)


    def output_path(*args):
        return os.path.join("output", *args)


@dataclass
class Config:
    base_model: str = "/home/cuizhibin/projects/Models/Qwen3-4B-Instruct-2507-bnb-4bit"
    output_dir: str = "output/qwen3_task3_unsloth"

    max_seq_length: int = 512
    load_in_4bit: bool = True

    seed: int = 42
    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 4
    learning_rate: float = 2e-4


cfg = Config()

# ===================== 2. 官方 Prompt 模板与领域定义 =====================

# 定义领域约束 (从您的 Prompt 复制)
DOMAIN_CONSTRAINTS = {
    'restaurant': (
        'RESTAURANT, FOOD, DRINKS, AMBIENCE, SERVICE, LOCATION',
        'GENERAL, PRICES, QUALITY, STYLE_OPTIONS, MISCELLANEOUS'
    ),
    'laptop': (
        'LAPTOP, DISPLAY, KEYBOARD, MOUSE, MOTHERBOARD, CPU, FANS_COOLING, PORTS, MEMORY, POWER_SUPPLY, OPTICAL_DRIVES, BATTERY, GRAPHICS, HARD_DISK, MULTIMEDIA_DEVICES, HARDWARE, SOFTWARE, OS, WARRANTY, SHIPPING, SUPPORT, COMPANY',
        'GENERAL, PRICE, QUALITY, DESIGN_FEATURES, OPERATION_PERFORMANCE, USABILITY, PORTABILITY, CONNECTIVITY, MISCELLANEOUS'
    ),
    'hotel': (
        'HOTEL, ROOMS, FACILITIES, ROOM_AMENITIES, SERVICE, LOCATION, FOOD_DRINKS',
        'GENERAL, PRICE, COMFORT, CLEANLINESS, QUALITY, DESIGN_FEATURES, STYLE_OPTIONS, MISCELLANEOUS'
    ),
    'finance': (
        'MARKET, COMPANY, BUSINESS, PRODUCT',
        'GENERAL, SALES, PROFIT, AMOUNT, PRICE, COST'
    )
}

# 基础指令模板
BASE_INSTRUCTION = """Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

### Instruction:
Given a textual instance [Text], extract all (A, C, O, VA) quadruplets, where:
- A is an Aspect term (a phrase describing an entity mentioned in [Text])
- C is a Category label (e.g. FOOD#QUALITY)
- O is an Opinion term
- VA is a Valence–Arousal score in the format (valence#arousal)

Valence ranges from 1 (negative) to 9 (positive),
Arousal ranges from 1 (calm) to 9 (excited).

### Label constraints:
[Entity Labels] ({entity_label})
[Attribute Labels] ({attribute_label})

### Example:
Input:
[Text] average to good thai food, but terrible delivery.

Output:
[Quadruplet] (thai food, FOOD#QUALITY, average to good, 6.75#6.38),
             (delivery, SERVICE#GENERAL, terrible, 2.88#6.62)

### Question:
Now complete the following example:
Input:
[Text] {text}

Output:
"""


# ===================== 3. 数据处理 =====================

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            yield json.loads(line)


def format_quad_target(quads: List[Dict[str, str]]) -> str:
    """将四元组列表格式化为: [Quadruplet] (A, C, O, V#A), ..."""
    parts = []
    for q in quads:
        parts.append(f"({q['Aspect']}, {q['Category']}, {q['Opinion']}, {q['VA']})")

    # 注意：为了匹配 Example 的输出格式，这里加上 [Quadruplet] 前缀
    if parts:
        return "[Quadruplet] " + ", ".join(parts)
    else:
        return "[Quadruplet]"


def get_domain_from_path(path: Path) -> str:
    """根据文件名判断领域"""
    filename = path.name.lower()
    if "restaurant" in filename:
        return "restaurant"
    elif "laptop" in filename:
        return "laptop"
    elif "hotel" in filename:
        return "hotel"
    elif "finance" in filename:
        return "finance"
    else:
        # 默认 fallback，防止报错，或者您可以抛出异常
        return "restaurant"


def prepare_dataset(tokenizer):
    # 定义训练文件
    train_files = [
        data_path("track_a", "subtask_3", "eng", "eng_laptop_train_alltasks.jsonl"),
        data_path("track_a", "subtask_3", "eng", "eng_restaurant_train_alltasks.jsonl")
    ]

    data_items = []
    print(f"[INFO] Loading data from: {train_files}")

    for file_path in train_files:
        p = Path(file_path)
        if not p.exists():
            print(f"[WARN] File not found: {p}")
            continue

        # 1. 确定领域 (laptop 或 restaurant)
        domain = get_domain_from_path(p)
        entity_label, attribute_label = DOMAIN_CONSTRAINTS[domain]

        for obj in read_jsonl(p):
            text = obj.get("Text", "")
            quads = obj.get("Quadruplet", []) or []

            # 过滤无效四元组
            valid_quads = [q for q in quads if
                           q.get("Aspect") and q.get("Category") and q.get("Opinion") and q.get("VA")]

            # 2. 构造 Assistant 回复 (Target)
            target_str = format_quad_target(valid_quads)

            # 3. 构造 User 输入 (动态填入领域约束)
            # 使用 .format 填充 {entity_label}, {attribute_label} 和 {text}
            user_content = BASE_INSTRUCTION.format(
                entity_label=entity_label,
                attribute_label=attribute_label,
                text=text
            )

            # 4. 构造 Chat Messages
            # 注意：System Prompt 已被整合进 User Instruction 中，因此 role 列表可以简化
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target_str}
            ]

            # 5. 应用 Chat 模板
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            data_items.append({"text": formatted_text})

    # 简单划分 Dev 集 (10%)
    import random
    random.seed(cfg.seed)
    random.shuffle(data_items)

    split_idx = int(len(data_items) * 0.9)
    train_data = data_items[:split_idx]
    dev_data = data_items[split_idx:]

    print(f"[INFO] Dataset loaded. Train: {len(train_data)}, Dev: {len(dev_data)}")
    return Dataset.from_list(train_data), Dataset.from_list(dev_data)


# ===================== 4. 主流程 =====================

def main():
    # 4.1 加载模型
    print("===> Loading Model (Unsloth)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )

    # 4.2 LoRA 配置
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    # 4.3 数据准备 (已集成新 Prompt 逻辑)
    train_ds, dev_ds = prepare_dataset(tokenizer)

    # 4.4 初始化 Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=SFTConfig(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum_steps,
            warmup_ratio=0.03,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            optim="adamw_8bit",
            seed=cfg.seed,
        ),
    )

    # 4.5 训练
    print("===> Starting Training...")
    trainer.train()

    # 4.6 保存
    save_path = Path(output_path(cfg.output_dir, "lora_model"))
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"===> Model saved to: {save_path}")

    # ===================== 5. 推理 (Inference) =====================
    print("===> Starting Inference on Dev Set...")
    FastLanguageModel.for_inference(model)

    dev_files = [
        data_path("track_a", "subtask_3", "eng", "eng_laptop_dev_task3.jsonl"),
        data_path("track_a", "subtask_3", "eng", "eng_restaurant_dev_task3.jsonl")
    ]

    inputs_data = []
    # 预测时也需要知道来源文件以确定 Domain
    for fp in dev_files:
        p = Path(fp)
        if p.exists():
            # 将 domain 信息暂存到对象中，方便后续构造 Prompt
            domain = get_domain_from_path(p)
            for item in read_jsonl(p):
                item["_domain"] = domain  # 临时标记
                inputs_data.append(item)

    print(f"[INFO] Predicting {len(inputs_data)} samples...")
    results = []

    for idx, obj in enumerate(inputs_data):
        text = obj.get("Text", "")
        domain = obj.get("_domain", "restaurant")  # 获取之前标记的 domain

        # 获取对应领域的约束
        entity_label, attribute_label = DOMAIN_CONSTRAINTS[domain]

        # 构造推理时的 User Prompt (与训练保持一致)
        user_content = BASE_INSTRUCTION.format(
            entity_label=entity_label,
            attribute_label=attribute_label,
            text=text
        )

        messages = [{"role": "user", "content": user_content}]

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_ids = outputs[0][input_ids.shape[1]:]
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        quads = extract_quadruplets(decoded_text, text)
        results.append({"ID": obj.get("ID"), "Quadruplet": quads})

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(inputs_data)}")

    pred_file = Path(output_path("submit", "task3", "qwen3_task3_unsloth_pred.jsonl"))
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    with pred_file.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"===> Prediction saved to: {pred_file}")


# ===================== 辅助函数 =====================
def recover_span(text: str, span: str) -> str:
    if not span: return span
    s = span.strip()
    idx = text.lower().find(s.lower())
    if idx != -1: return text[idx: idx + len(s)]
    return s


def normalize_va(va_str: str) -> str | None:
    m = re.match(r'^\s*([0-9.]+)\s*#\s*([0-9.]+)\s*$', va_str)
    if not m: return None
    try:
        v = min(9.0, max(1.0, float(m.group(1))))
        a = min(9.0, max(1.0, float(m.group(2))))
        return f"{v:.2f}#{a:.2f}"
    except:
        return None


def extract_quadruplets(raw: str, text: str) -> List[Dict[str, str]]:
    pattern = r'\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, raw)
    quads = []
    for asp, cat, opn, va in matches:
        asp_rec = recover_span(text, asp)
        opn_rec = recover_span(text, opn)
        va_norm = normalize_va(va)
        if asp_rec and cat and opn_rec and va_norm:
            quads.append({
                "Aspect": asp_rec,
                "Category": cat.strip(),
                "Opinion": opn_rec,
                "VA": va_norm
            })
    return quads


if __name__ == "__main__":
    main()