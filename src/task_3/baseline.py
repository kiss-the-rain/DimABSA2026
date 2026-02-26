#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DimABSA2026 Track-A Subtask3 本地离线版 baseline（最小改动 + 更稳）
- 只读本地模型（local_files_only=True / HF_HUB_OFFLINE=1）
- 只读本地数据（jsonl 路径）
- 训练前把数据集压成只含 text，避免 DataCollator 把 ID/Text 等字符串列也拿去 pad 导致报错
- 用 HuggingFace Trainer + DataCollatorForLanguageModeling 稳定地产生 labels，避免 int.mean / batch_size mismatch
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import json
import zipfile
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from unsloth import FastLanguageModel
from trl import SFTTrainer


# =========================
# 0) 推荐：离线/本地模式
# =========================
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 你之前遇到过 torchdynamo/fake tensor 的编译期报错，先直接禁用更稳
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# =========================
# 1) 任务配置
# =========================
subtask = "subtask_3"   # subtask_2 or subtask_3
task    = "task3"       # task2 or task3
lang    = "eng"
domain  = "restaurant"  # restaurant / laptop / hotel / finance

# 本地模型目录（你自己的）
MODEL_ID = Path("/home/cuizhibin/projects/Models/Qwen3-4B-Instruct-2507-bnb-4bit")  # <- 改成你的本地模型目录

# 输出目录
OUT_DIR = Path("./outputs_task3_local")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2) 路径工具（兼容你项目结构）
# =========================
try:
    from src.utils.paths import data_path, output_path  # 你项目里已有
except Exception:
    def data_path(*parts):
        return os.path.join("data", *parts)
    def output_path(*parts):
        return os.path.join("output", *parts)

train_url   = str(data_path("track_a", "subtask_3", "eng", f"eng_{domain}_train_alltasks.jsonl"))
predict_url = str(data_path("track_a", "subtask_3", "eng", f"eng_{domain}_dev_task3.jsonl"))

assert os.path.exists(train_url),   f"Train file not found: {train_url}"
assert os.path.exists(predict_url), f"Dev file not found: {predict_url}"
assert MODEL_ID.exists(),           f"MODEL_ID not found: {MODEL_ID}"

# =========================
# 3) 读数据
# =========================
dataset = load_dataset("json", data_files={"train": train_url})

# =========================
# 4) prompt 模板（官方 baseline 原样）
# =========================
if task == "task2":
    instruction = """Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

### Instruction:
Given a textual instance [Text], extract all (A, O, VA) triplets, where:
- A is an Aspect term (a phrase describing an entity mentioned in [Text])
- O is an Opinion term
- VA is a Valence–Arousal score in the format (valence#arousal)

Valence ranges from 1 (negative) to 9 (positive),
Arousal ranges from 1 (calm) to 9 (excited).

### Example:
Input:
[Text] average to good thai food, but terrible delivery.

Output:
[Triplet] (thai food, average to good, 6.75#6.38), (delivery, terrible, 2.88#6.62)

### Question:
Now complete the following example:
Input:
"""
    def convert(x):
        text = x["Text"]
        quads = x.get("Quadruplet", [])
        answer = ", ".join([f"({q['Aspect']}, {q['Opinion']}, {q['VA']})" for q in quads])
        prompt = instruction + "[Text] " + text + "\n\nOutput:"
        return {"text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}"}

elif task == "task3":
    rest_entity = "RESTAURANT, FOOD, DRINKS, AMBIENCE, SERVICE, LOCATION"
    rest_attribute = "GENERAL, PRICES, QUALITY, STYLE_OPTIONS, MISCELLANEOUS"

    laptop_entity = "LAPTOP, DISPLAY, KEYBOARD, MOUSE, MOTHERBOARD, CPU, FANS_COOLING, PORTS, MEMORY, POWER_SUPPLY, OPTICAL_DRIVES, BATTERY, GRAPHICS, HARD_DISK, MULTIMEDIA_DEVICES, HARDWARE, SOFTWARE, OS, WARRANTY, SHIPPING, SUPPORT, COMPANY"
    laptop_attribute = "GENERAL, PRICE, QUALITY, DESIGN_FEATURES, OPERATION_PERFORMANCE, USABILITY, PORTABILITY, CONNECTIVITY, MISCELLANEOUS"

    hotel_entity = "HOTEL, ROOMS, FACILITIES, ROOM_AMENITIES, SERVICE, LOCATION, FOOD_DRINKS"
    hotel_attribute = "GENERAL, PRICE, COMFORT, CLEANLINESS, QUALITY, DESIGN_FEATURES, STYLE_OPTIONS, MISCELLANEOUS"

    finance_entity = "MARKET, COMPANY, BUSINESS, PRODUCT"
    finance_attribute = "GENERAL, SALES, PROFIT, AMOUNT, PRICE, COST"

    entity_attribute_map = {
        "restaurant": (rest_entity, rest_attribute),
        "laptop": (laptop_entity, laptop_attribute),
        "hotel": (hotel_entity, hotel_attribute),
        "finance": (finance_entity, finance_attribute),
    }
    entity_label, attribute_label = entity_attribute_map[domain]

    instruction = f"""Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

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
"""
    def convert(x):
        text = x["Text"]
        quads = x.get("Quadruplet", [])
        answer = ", ".join([f"({q['Aspect']}, {q['Category']}, {q['Opinion']}, {q['VA']})" for q in quads])
        prompt = instruction + "[Text] " + text + "\n\nOutput:"
        return {"text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}"}
else:
    raise ValueError("task must be task2 or task3")

# =========================
# 5) 只保留 text 列（关键：避免 collator 把 ID/Text 等字符串列也当成 tensor 来 pad）
# =========================
train_dataset = dataset["train"].map(
    convert,
    remove_columns=dataset["train"].column_names,  # ✅把原列都删掉，只留下 convert 的返回
)

# =========================
# 6) 加载本地模型 + tokenizer
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(MODEL_ID),
    max_seq_length=512,
    load_in_4bit=True,
    local_files_only=True,   # ✅只读本地
)

# Qwen 系列经常没有 pad_token，给它一个，避免 padding 报错
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA（官方 baseline：只打 attention；你要更强可以把 MLP 的 gate/up/down 也加上）
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","up_proj","down_proj"],
)

# =========================
# 7) 训练：SFTTrainer + 显式 tokenize，避免 logits/labels 长度不一致
# 4090 支持 bf16，优先 bf16；不要同时开 fp16 & bf16

max_seq_length = 512
tokenizer.model_max_length = max_seq_length  # 确保 SFTTrainer 不会产出超过上限的序列

def tokenize_fn(ex):
    return tokenizer(
        ex["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

tokenized_train = train_dataset.map(
    tokenize_fn,
    remove_columns=train_dataset.column_names,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_train,
    data_collator=data_collator,
    max_seq_length=max_seq_length,
    packing=False,  # ✅先关掉，稳；想提速再开
    args=TrainingArguments(
        output_dir=str(OUT_DIR / "lora_ckpt"),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        warmup_steps=20,
        num_train_epochs=2,
        learning_rate=5e-5,
        logging_steps=100,
        save_steps=400,
        bf16=True,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=8,
    ),
)


trainer.train()
# 保存 LoRA（adapter）
save_dir = OUT_DIR / "lora_adapter"
save_dir.mkdir(parents=True, exist_ok=True)
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# =========================
# 8) 推理 + 写 jsonl
# =========================
# predict_dataset = load_dataset("json", data_files={"train": predict_url})["train"]
predict_dataset = load_dataset("json", data_files={"train": predict_url})

# convert text to prompt
def format_dataset(x):
    text = x["Text"]
    final_prompt = instruction + "[Text] " + text + "\n\nOutput:"
    return [{"role": "user", "content": final_prompt}]

# ========== 更稳的抽取：只匹配合法格式 ==========
def extract_answer(text, task):
    result = []
    if task == "task2":
        # (aspect, opinion, 7.50#7.33)
        pattern = r"\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([0-9]+(?:\.[0-9]+)?#[0-9]+(?:\.[0-9]+)?)\s*\)"
        matches = re.findall(pattern, text)
        for aspect, opinion, va in matches:
            # 过滤占位符
            if aspect.strip() in {"A", "Aspect"}:
                continue
            result.append({
                "Aspect": aspect.strip(),
                "Opinion": opinion.strip(),
                "VA": va.strip(),
            })

    elif task == "task3":
        # (aspect, LAPTOP#PRICE, opinion, 7.50#7.33)
        pattern = r"\(\s*([^,()]+?)\s*,\s*([A-Z_]+#[A-Z_]+)\s*,\s*([^,()]+?)\s*,\s*([0-9]+(?:\.[0-9]+)?#[0-9]+(?:\.[0-9]+)?)\s*\)"
        matches = re.findall(pattern, text)
        for aspect, category, opinion, va in matches:
            # 过滤占位符/垃圾项
            if aspect.strip() in {"A", "Aspect"}:
                continue
            if category.strip() in {"C", "Category"}:
                continue
            result.append({
                "Aspect": aspect.strip(),
                "Category": category.strip(),
                "Opinion": opinion.strip(),
                "VA": va.strip(),
            })
    else:
        raise ValueError("Invalid task")
    return result


# ========== 推理：只 decode 新生成 token，别把 prompt 也 decode 进去 ==========
predict_dataset = load_dataset("json", data_files=predict_url)
pred_ds = predict_dataset["train"] if "train" in predict_dataset else predict_dataset

# Perform inference
results = []
model.eval()

for i, sample in enumerate(pred_ds):
    messages = format_dataset(sample)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = generated[0][input_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if "Output:" in gen_text:
        gen_text = gen_text.split("Output:", 1)[-1].strip()

    key = "Triplet" if task == "task2" else "Quadruplet"
    dump_data = {
        "ID": sample.get("ID", f"sample_{i}"),
        key: extract_answer(gen_text, task),
    }

    print(dump_data)
    results.append(dump_data)

# 写 JSONL
jsonl_path = Path(output_path("submit", "task3", f"pred_{lang}_{domain}_local.jsonl"))
jsonl_path.parent.mkdir(parents=True, exist_ok=True)
with open(jsonl_path, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 可选：打包成 zip，方便提交/传输
zip_path = jsonl_path.with_suffix(".zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(jsonl_path, arcname=jsonl_path.name)

print(f"[DONE] wrote: {jsonl_path}")
print(f"[DONE] zipped: {zip_path}")
