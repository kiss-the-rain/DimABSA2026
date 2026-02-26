# dataset.py
# -*- coding: utf-8 -*-
"""
DimASQP 子任务 3 的数据读取与清洗代码。

核心思想：
- 读取官方 JSONL（train_alltasks + dev_task3 等）；
- 对带标签的数据，把 Quadruplet 线性化为一个目标字符串；
- 构造 seq2seq 训练样本：source = 原始 Text，target = 线性化后的四元组描述；
- 使用 HuggingFace 的 tokenizer 做编码。

你需要做的改动：
1. 把 DEFAULT_TRAIN_FILES / DEFAULT_DEV_FILES 换成你项目里的真实路径；
2. 在 model.py 里导入本文件，并传入合适的路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# ======================== 配置 ========================

@dataclass
class DataConfig:
    """
    与“数据处理/编码”相关的超参数。

    参数说明：
    - model_name_or_path: 用来初始化 tokenizer（以及后面模型）的名称或本地权重路径，
      例如 "t5-base"、"google/mt5-small"、"facebook/bart-base" 等。
    - max_source_len: 编码输入 Text 时的最大长度（token 数），过长会被截断。
    - max_target_len: 编码目标字符串（线性化四元组）时的最大长度。
    """
    model_name_or_path: str = "t5-base"
    max_source_len: int = 128
    max_target_len: int = 128


# 你可以在 model.py 里覆盖这些默认路径
DEFAULT_TRAIN_FILES = [
    # 注意：这里是示例路径，换成你项目中的实际路径
    # "data/track_a/subtask_3/eng/eng_laptop_train_alltasks.jsonl",
    # "data/track_a/subtask_3/eng/eng_restaurant_train_alltasks.jsonl",
]
DEFAULT_DEV_FILES = [
    # 如果你自己从 train_alltasks 划分 dev，可以在 model.py 中重新设置
]


# ======================== 工具函数 ========================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件，返回一个字典列表。"""
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def linearize_quadruples(quads: List[Dict[str, Any]]) -> str:
    """
    把一个样本里的所有四元组线性化为一个字符串。

    设计目标：
    - 人能看懂；
    - 结构尽量规整，方便后续你写解析脚本把生成文本再还原回 (A, C, O, VA)。

    目前使用的模板：
      "Aspect: A, Category: C, Opinion: O, VA: V#A; Aspect: ..."

    例子：
      [
        {"Aspect": "thai food", "Category": "FOOD#QUALITY",
         "Opinion": "average to good", "VA": "6.75#6.38"},
        {"Aspect": "delivery", "Category": "SERVICE#GENERAL",
         "Opinion": "terrible", "VA": "2.88#6.62"}
      ]

    -> "Aspect: thai food, Category: FOOD#QUALITY, Opinion: average to good, VA: 6.75#6.38;
        Aspect: delivery, Category: SERVICE#GENERAL, Opinion: terrible, VA: 2.88#6.62"
    """
    parts: List[str] = []
    for q in quads:
        a = q.get("Aspect", "NULL")
        c = q.get("Category", "NULL#NULL")
        o = q.get("Opinion", "NULL")
        va = q.get("VA", "5.00#5.00")
        part = f"Aspect: {a}, Category: {c}, Opinion: {o}, VA: {va}"
        parts.append(part)
    return "; ".join(parts)


# ======================== Dataset 类 ========================

class DimASQPDataset(Dataset):
    """
    子任务 3 的 Dataset。

    - use_labels=True: 训练/有标签场景，会同时返回 labels（target）。
    - use_labels=False: 推理/测试场景，只编码 Text，不需要 labels。
    """

    def __init__(
        self,
        data_files: List[str],
        tokenizer: PreTrainedTokenizerBase,
        config: DataConfig,
        use_labels: bool = True,
    ) -> None:
        """
        参数说明：
        - data_files: JSONL 文件路径列表，可以同时传入 laptop + restaurant 的 train_alltasks。
        - tokenizer: transformers 的 tokenizer 对象，例如 AutoTokenizer.from_pretrained("t5-base")。
        - config: DataConfig 实例，提供 max_source_len / max_target_len 等。
        - use_labels: 是否需要读取 Quadruplet 并生成 target 文本。
        """
        self.tokenizer = tokenizer
        self.config = config
        self.use_labels = use_labels

        self.examples: List[Dict[str, Any]] = []
        for fp in data_files:
            path = Path(fp)
            assert path.exists(), f"文件不存在: {path}"
            records = read_jsonl(path)
            for r in records:
                # 只保留 ID + Text + Quadruplet
                ex = {
                    "id": r.get("ID"),
                    "text": r.get("Text", ""),
                }
                if use_labels and "Quadruplet" in r and r["Quadruplet"]:
                    ex["quads"] = r["Quadruplet"]
                self.examples.append(ex)

        if use_labels:
            # 过滤掉没有 Quadruplet 的样本（以防将来有混合文件）
            self.examples = [e for e in self.examples if "quads" in e and e["quads"]]

        print(
            f"[DimASQPDataset] 加载样本数: {len(self.examples)} | "
            f"use_labels={self.use_labels}"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        text: str = ex["text"]

        # ---------- 编码输入 Text ----------
        # 关键参数说明：
        # - max_length: 最长长度，超过就截断；
        # - padding="max_length": 不同样本统一 pad 到同一长度，方便 DataLoader 默认 collate；
        # - truncation=True: 开启截断；
        # - return_tensors="pt": 返回 PyTorch 张量。
        encoded_input = self.tokenizer(
            text,
            max_length=self.config.max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item: Dict[str, Any] = {
            "id": ex["id"],
            "input_ids": encoded_input["input_ids"].squeeze(0),      # (L,)
            "attention_mask": encoded_input["attention_mask"].squeeze(0),
        }

        # ---------- 编码目标（线性化四元组） ----------
        if self.use_labels:
            target_str = linearize_quadruples(ex["quads"])

            encoded_tgt = self.tokenizer(
                target_str,
                max_length=self.config.max_target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = encoded_tgt["input_ids"].squeeze(0)  # (T,)

            # 把 padding 的 token id 置为 -100，loss 计算时会被忽略
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100

            item["labels"] = labels
            item["target_text"] = target_str  # 方便调试打印

        return item


# ======================== 简单测试 ========================

if __name__ == "__main__":
    # 简单自测，确认能跑通
    cfg = DataConfig(
        model_name_or_path="t5-base",
        max_source_len=128,
        max_target_len=128,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

    # 把下面路径改成你本地的
    train_files = DEFAULT_TRAIN_FILES

    if not train_files:
        print("请在 dataset.py 中设置 DEFAULT_TRAIN_FILES 后再运行自测。")
    else:
        ds = DimASQPDataset(train_files, tokenizer, cfg, use_labels=True)
        print("样本数:", len(ds))
        sample = ds[0]
        print("ID:", sample["id"])
        print("原始文本:", tokenizer.decode(sample["input_ids"], skip_special_tokens=True))
        print("目标文本:", sample["target_text"])
