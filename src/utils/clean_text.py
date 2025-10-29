# -*- coding: utf-8 -*-
# utils/clean_text.py
import re
import math
from typing import Any, Iterable
import pandas as pd

# --- 1) 先把分开的缩写合并：i ' m -> i'm, i ' ll -> i'll 等 ---
_CONTRACTIONS = {
    r"\b([A-Za-z])\s*'\s*s\b":   r"\1's",
    r"\b([A-Za-z])\s*'\s*re\b":  r"\1're",
    r"\b([A-Za-z])\s*'\s*ve\b":  r"\1've",
    r"\b([A-Za-z])\s*'\s*d\b":   r"\1'd",
    r"\b([A-Za-z])\s*'\s*ll\b":  r"\1'll",
    r"\b([A-Za-z])\s*'\s*m\b":   r"\1'm",
    r"\b([A-Za-z])\s*n\s*'\s*t\b": r"\1n't",
}

# 允许调用方自定义：是否把 & 还原为 and（仅在空白包夹时）
DEFAULT_RESTORE_AMP = True


def _to_text(x: Any) -> str:
    """把输入转为字符串；None/NaN -> 空串。"""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def detokenize_ptb_A(s: Any, restore_amp: bool = DEFAULT_RESTORE_AMP) -> str:
    """
    将 PTB 风格 token 修复为常规英文文本：
    - 合并缩写：i ' m -> i'm
    - 修复引号：`` pretty ''、` pretty `、' pretty ' -> "pretty"
    - 收紧标点空格：, . ! ? % 前不留空；左括号后不留空
    - 连字符：只在“词-词”之间收紧，保留行首“- ”
    - 省略号：. . . -> ...
    - &（空白包夹） -> and（可选）
    - 压缩多余空白
    """
    s = _to_text(s)

    # === A. 合并缩写（优先，避免把单词内的 ' 误当引号） ===
    for pat, rep in _CONTRACTIONS.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)

    # === B. 处理 PTB 引号 ===
    # 1) 连续 token：`` / '' -> "
    s = re.sub(r"\s*``\s*", '"', s)
    s = re.sub(r"\s*''\s*", '"', s)

    # 2) 被拆开的成对 token：` <空白*> `、' <空白*> ' -> "
    s = re.sub(r"`\s*`", '"', s)
    s = re.sub(r"'\s*'", '"', s)

    # 3) 片段被反/单引号包围：` foo `、' bar ' -> "foo"/"bar"
    s = re.sub(r"`\s+([^`']+?)\s+`", r'"\1"', s)
    s = re.sub(r"'\s+([^`']+?)\s+'", r'"\1"', s)

    # 4) 把 " foo " 收紧为 "foo"
    s = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', s)

    # === C. 标点与括号空格规整 ===
    s = re.sub(r"\s+([,.;:!?%])", r"\1", s)   # 去掉标点前空格
    s = re.sub(r"([\[(])\s+", r"\1", s)       # 去掉左括号后空格

    # === D. 连字符：仅在“词-词”之间收紧 ===
    s = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", s)

    # === E. 省略号 ===
    s = re.sub(r"\.\s*\.\s*\.", "...", s)

    # === F. 仅当 & 被空白包夹时还原为 and（防止误伤 URL）===
    if restore_amp:
        s = re.sub(r"(?<=\s)&(?=\s)", " and ", s)

    # === G. 压缩空白 ===
    s = re.sub(r"\s+", " ", s).strip()

    return s


def clean_series(texts: Iterable[Any], restore_amp: bool = DEFAULT_RESTORE_AMP) -> pd.Series:
    """
    对 pandas Series/可迭代对象批量清洗；保持 NaN -> "" 的语义。
    """
    if isinstance(texts, pd.Series):
        return texts.fillna("").map(lambda t: detokenize_ptb_A(t, restore_amp=restore_amp))
    # 其他可迭代
    return pd.Series([detokenize_ptb_A(t, restore_amp=restore_amp) for t in texts])


# --- 可选：简单自检护栏（开发期很有用，线上可关） ---
def sanity_checks(series: pd.Series) -> None:
    """
    若检测到明显残留/异常，抛出 AssertionError 方便定位。
    """
    assert not series.str.contains(r"`|'").any(), "Detected leftover PTB quotes (` or ')"
    assert not series.str.contains(r'"\s+"').any(), 'Detected empty quotes "" with spaces'
