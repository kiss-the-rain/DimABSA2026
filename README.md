# DimABSA2026

本项目用于 DimABSA 竞赛的三个子任务：
- Subtask 1：Aspect 级 VA 回归（`Aspect_VA`）
- Subtask 2：Triplet 抽取（`Aspect, Opinion, VA`）
- Subtask 3：Quadruplet 抽取（`Aspect, Category, Opinion, VA`）

当前代码包含两类路线：
- `src/task_1/model.py`：基于 DeBERTa/RoBERTa 编码器的回归模型（手写训练循环）
- `src/task_2/qwen3_task2_base.py`、`src/task_3/qwen3_task3_sft.py`：基于 Qwen3 + LoRA 的生成式 SFT

## 1. 项目结构

```text
src/
  task_1/
    dataset.py          # Task1 数据预处理（jsonl -> parquet/csv/npz）
    model.py            # Task1 训练+推理
  task_2/
    qwen3_task2_base.py # Task2 Qwen3 LoRA 训练+推理
    qwen3_task2_nospanrec_eval.py
  task_3/
    qwen3_task3_sft.py  # Task3 Qwen3 LoRA 训练+推理（Trainer）
    qwen3_task3_nospanrec_eval.py
  utils/
    paths.py            # 路径工具（DIMABSA_ROOT / DIMABSA_OUT）
data/
  track_a/...           # 官方原始数据
  output/...            # 中间产物
  submit/...            # 提交文件
evaluation_script/
  metrics_subtask_1_2_3.py
```

## 2. 环境准备

建议 Python 3.10，GPU 环境（4090D 可用）。

常用依赖（按脚本导入）：
- `torch`
- `transformers`
- `datasets`
- `peft`
- `bitsandbytes`（4bit QLoRA 时）
- `sentencepiece`（DeBERTa 慢分词器需要）
- `pandas`, `numpy`, `scikit-learn`

建议环境变量：

```bash
export DIMABSA_ROOT=/home/cuizhibin/projects/DimABSA2026
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

## 3. 数据放置约定

按当前代码，官方数据放在：
- `data/track_a/subtask_1/eng/...`
- `data/track_a/subtask_2/eng/...`
- `data/track_a/subtask_3/eng/...`

`src/utils/paths.py` 会优先使用 `DIMABSA_ROOT` 解析项目根目录。

## 4. Task1（VA 回归）

### 4.1 预处理

```bash
python src/task_1/dataset.py
```

产物：
- `data/output/track_a/subtask_1/train_pairs.parquet`
- `data/output/track_a/subtask_1/dev_pairs.parquet`

说明：
- `dev_pairs` 会保留 `aspect_raw`，用于提交时严格回写原始 Aspect（避免大小写/引号不一致报错）。

### 4.2 训练 + 推理

```bash
python src/task_1/model.py
```

产物：
- `best_model.pt`
- `data/submit/task1/pred_eng_laptop.jsonl`
- `data/submit/task1/pred_eng_restaurant.jsonl`

## 5. Task2（Triplet，Qwen3 + LoRA）

```bash
python src/task_2/qwen3_task2_base.py
```

当前脚本流程：
1. 从 `eng_*_train_alltasks.jsonl` 构建 SFT 样本  
2. LoRA 训练（手写训练循环）  
3. 在 `test_task2` 上生成预测并拆分领域输出

产物：
- LoRA：`data/output/qwen3_task2/qwen3_task2_lora/`
- 提交：
  - `data/submit/task2/qwen3_task2_dev_pred_laptop.jsonl`
  - `data/submit/task2/qwen3_task2_dev_pred_restaurant.jsonl`
- 调试原始生成：
  - `data/debug/task2_raw_generations.jsonl`

## 6. Task3（Quadruplet，Qwen3 + LoRA）

```bash
python src/task_3/qwen3_task3_sft.py
```

当前脚本流程：
1. 从 `eng_*_train_alltasks.jsonl` 构建 SFT 样本  
2. 使用 `Trainer` 训练 LoRA  
3. 在 `test_task3` 上生成预测并拆分领域输出

产物：
- LoRA：`data/output/qwen3_task3/qwen3_task3_lora/`
- 提交：
  - `data/submit/task3/pred_eng_laptop.jsonl`
  - `data/submit/task3/pred_eng_restaurant.jsonl`
- 调试原始生成：
  - `data/debug/task3_raw_generations.jsonl`

## 7. 评测

官方脚本示例：

```bash
python evaluation_script/metrics_subtask_1_2_3.py \
  -t 1 \
  -p data/submit/task1/pred_eng_laptop.jsonl \
  -g data/track_a/subtask_1/eng/eng_laptop_train_alltasks.jsonl
```

`-t` 含义：
- `1`：Subtask1（Aspect_VA）
- `2`：Subtask2（Triplet）
- `3`：Subtask3（Quadruplet）

## 8. 常见问题

1. `ModuleNotFoundError: No module named 'src'`  
   优先在项目根目录运行脚本，或使用 `python -m src.xxx` 方式。

2. DeBERTa tokenizer 报错/警告  
   安装 `sentencepiece`，并优先使用慢分词器。

3. 提交提示 `Missing aspect ...`  
   输出时必须严格使用原始 Aspect 字符串（大小写、标点、弯引号都要一致）。

4. 4bit + CUDA 报 GEMM 不支持  
   可尝试 `bnb_4bit_compute_dtype=torch.float16`，并关闭不兼容混合精度组合。
