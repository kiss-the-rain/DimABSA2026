import os
import re
import json
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import json,torch
import os
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, DataCollatorWithPadding


os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["UNSLOTH_DISABLE_TRAINING_PATCH"] = "1"  # ✅关键：禁用训练步patch

#task config
task = "task3"
domain = "restaurant"

try:
    from src.utils.paths import data_path, output_path
except ImportError:
    def data_path(*args):
        return os.path.join("data", *args)


    def output_path(*args):
        return os.path.join("output", *args)

train_url = str(data_path("track_a", "subtask_3", "eng", "eng_restaurant_train_alltasks.jsonl"))
predict_url = str(data_path("track_a", "subtask_3", "eng", "eng_restaurant_dev_task3.jsonl"))


#load train data from url
dataset = load_dataset("json", data_files=train_url)

# task 2 prompt template covert
if task == "task2":
  instruction = '''Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

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
'''

  def convert(x):
      text = x["Text"]
      quads = x.get("Quadruplet", [])
      answer = ", ".join([
          f"({q['Aspect']}, {q['Opinion']}, {q['VA']})"
          for q in quads
      ])
      prompt = instruction + "[Text] " + text + "\n\nOutput:"
      return {"text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}"}

# task 3 prompt template covert, with task3 predefine entity and attribute labels.
elif task == "task3":
  rest_entity = 'RESTAURANT, FOOD, DRINKS, AMBIENCE, SERVICE, LOCATION'
  rest_attribute = 'GENERAL, PRICES, QUALITY, STYLE_OPTIONS, MISCELLANEOUS'

  laptop_entity = 'LAPTOP, DISPLAY, KEYBOARD, MOUSE, MOTHERBOARD, CPU, FANS_COOLING, PORTS, MEMORY, POWER_SUPPLY, OPTICAL_DRIVES, BATTERY, GRAPHICS, HARD_DISK, MULTIMEDIA_DEVICES, HARDWARE, SOFTWARE, OS, WARRANTY, SHIPPING, SUPPORT, COMPANY'
  laptop_attribute = 'GENERAL, PRICE, QUALITY, DESIGN_FEATURES, OPERATION_PERFORMANCE, USABILITY, PORTABILITY, CONNECTIVITY, MISCELLANEOUS'

  hotel_entity = 'HOTEL, ROOMS, FACILITIES, ROOM_AMENITIES, SERVICE, LOCATION, FOOD_DRINKS'
  hotel_attribute = 'GENERAL, PRICE, COMFORT, CLEANLINESS, QUALITY, DESIGN_FEATURES, STYLE_OPTIONS, MISCELLANEOUS'

  finance_entity = 'MARKET, COMPANY, BUSINESS, PRODUCT'
  finance_attribute = 'GENERAL, SALES, PROFIT, AMOUNT, PRICE, COST'

  entity_attribute_map = {
      'restaurant': (rest_entity, rest_attribute),
      'laptop': (laptop_entity, laptop_attribute),
      'hotel': (hotel_entity, hotel_attribute),
      'finance': (finance_entity, finance_attribute),
  }

  entity_label, attribute_label = entity_attribute_map[domain]

  instruction = f'''Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

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
'''

  def convert(x):
      text = x["Text"]
      quads = x.get("Quadruplet", [])
      answer = ", ".join([
          f"({q['Aspect']}, {q['Category']}, {q['Opinion']}, {q['VA']})"
          for q in quads
      ])
      prompt = instruction + "[Text] " + text + "\n\nOutput:"
      return {"text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}"}


# covert dataset to train template
train_dataset = dataset["train"].map(convert)



model_id = "/home/cuizhibin/projects/Models/Qwen3-4B-Instruct-2507-bnb-4bit"

# tokenizer and model setting
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = 512, # DimASBA Task usually less then 512 tokens.
    load_in_4bit = True,
    local_files_only = True,
)

# lora setting
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0.05,
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],

)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# ====== 关键：显式 tokenize，并构造 labels，保证 loss 一定是 Tensor ======
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

# ====== 训练模式：非常关键 ======
FastLanguageModel.for_training(model)
model.train()
model.config.use_cache = False
torch.set_grad_enabled(True)

# ====== tokenize：把 text 变成 input_ids ======
max_len = 512
def tokenize_fn(ex):
    return tokenizer(
        ex["text"],
        truncation=True,
        max_length=max_len,
        padding=False,
    )

tokenized_train = train_dataset.map(
    tokenize_fn,
    remove_columns=train_dataset.column_names,
)

# ====== collator：自动 pad + 自动 labels（mlm=False => causal LM）=====
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
loader = DataLoader(tokenized_train, batch_size=1, shuffle=True, collate_fn=collator)

# ====== 优化器 & 学习率调度 ======
lr = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

num_epochs = 2
grad_accum = 4
warmup_steps = 20

total_updates = (len(loader) * num_epochs + grad_accum - 1) // grad_accum
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_updates,
)

# ====== mixed precision：4090D 优先 bf16 ======
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and (not use_bf16)
scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

print(f"[INFO] steps={len(loader)} epochs={num_epochs} grad_accum={grad_accum} "
      f"total_updates={total_updates} bf16={use_bf16} fp16={use_fp16}")

# ====== 训练循环 ======
global_update = 0
optimizer.zero_grad(set_to_none=True)

for epoch in range(num_epochs):
    for it, batch in enumerate(loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # autocast：bf16/fp16 任选其一
        if use_fp16:
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        elif use_bf16:
            ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            # CPU or no AMP
            class _Dummy:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            ctx = _Dummy()

        with ctx:
            out = model(**batch)
            loss = out.loss
            # ✅ 一步到位：如果 loss 不是 tensor，立刻报更清晰的错
            if not torch.is_tensor(loss):
                raise TypeError(f"loss is not tensor: {type(loss)}; out keys: {out.keys() if hasattr(out,'keys') else 'N/A'}")
            loss = loss / grad_accum

        if use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (it + 1) % grad_accum == 0:
            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1

            if global_update % 50 == 0:
                # 乘回 grad_accum 还原真实 loss
                print(f"[train] epoch={epoch} update={global_update}/{total_updates} "
                      f"loss={loss.item()*grad_accum:.4f} lr={scheduler.get_last_lr()[0]:.2e}")





# load dev json to predict
predict_dataset = load_dataset("json", data_files=predict_url)

# convert text to prompt
def format_dataset(x):
    text = x["Text"]
    final_prompt = instruction + '[Text] ' + text + '\n\nOutput:'
    return [
        {"role": "user", "content": final_prompt}
    ]

# extract answer
def extract_answer(text,task):
  result = []
  if task == "task2":
    pattern = r'\(([^,]+),\s*([^,]+),\s*([\d.]+#[\d.]+)\)'
    matches = re.findall(pattern, text)

    for aspect, opinion, va in matches:
        meta_triplet = {}
        meta_triplet["Aspect"] = aspect.strip()
        meta_triplet["Opinion"] = opinion.strip()
        meta_triplet["VA"] = va
        result.append(meta_triplet)

  elif task == "task3":
    pattern = r'\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, text)

    for aspect, category, opinion, va in matches:
        meta_quadra = {}
        meta_quadra["Aspect"] = aspect.strip()
        meta_quadra["Category"] = category.strip()
        meta_quadra["Opinion"] = opinion.strip()
        meta_quadra["VA"] = va
        result.append(meta_quadra)
  else:
    raise ValueError("Invalid task")

  return result

# Perform inference
results = []
for i, sample in enumerate(predict_dataset["train"]):
    messages = format_dataset(sample)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    result = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        max_new_tokens=1024,
        temperature=0.7, top_p=0.8, top_k=20,
    )

    decoded = tokenizer.decode(result[0])
    extracted_text = decoded.split("\n")[-1]

    key = "Triplet" if task == "task2" else "Quadruplet"

    dump_data = {
        "ID": sample.get("ID", f"sample_{i}"),
        "Text": sample["Text"],
        key: extract_answer(extracted_text, task),
    }

    print(dump_data)
    results.append(dump_data)

save_dir = "./Lora_adapter"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("[INFO] LoRA adapter saved to", save_dir)


# JSONL file path
jsonl_path = Path(output_path("submit", "task3", "qwen3_task3_unsloth_pred.jsonl"))

# write JSONL
with open(jsonl_path, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

