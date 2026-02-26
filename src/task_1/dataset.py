"""
DimABSA Subtask1 数据转换与存储（使用 data_path + pd.read_json(lines=True)）
- Train:  *_train_alltasks.jsonl  -> (text, aspect, v, a, id, category, opinion)
- Dev:    *_dev_task1.jsonl       -> (id, text, aspect)

运行：python dataset.py
输出：output/track_a/subtask_1/eng/subtask1_processed/{train_pairs,dev_pairs}.{parquet,csv,npz}
"""

from __future__ import annotations
from pathlib import Path
import math, re, unicodedata
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
# 你项目里的路径工具
from src.utils.paths import data_path, output_path


# ========= 读取（按你给的方式） =========
lp_dev   = data_path("track_a", "subtask_1", "eng", "eng_laptop_test_task1.jsonl")
lp_train = data_path("track_a", "subtask_1", "eng", "eng_laptop_train_alltasks.jsonl")
rs_dev   = data_path("track_a", "subtask_1", "eng", "eng_restaurant_test_task1.jsonl")
rs_train = data_path("track_a", "subtask_1", "eng", "eng_restaurant_train_alltasks.jsonl")

for p in [lp_dev, lp_train, rs_dev, rs_train]:
    assert p.exists(), f"未找到数据文件：{p}. 如在云端，请先设置环境变量 DIMABSA_ROOT=你的项目根目录"

laptop_dev      = pd.read_json(lp_dev, lines=True)
restaurant_dev  = pd.read_json(rs_dev, lines=True)
laptop_train    = pd.read_json(lp_train, lines=True)
restaurant_train= pd.read_json(rs_train, lines=True)


# ========= 工具 =========
# === 去掉强调符号，但保留词面（大小写不变） ===
_EM_DLM = re.compile(r"`+([^`]+?)`+")            # `like this`
_AST_EM = re.compile(r"\*(\S(?:.*?\S)?)\*")      # *italic*
_UND_EM = re.compile(r"_(\S(?:.*?\S)?)_")        # _italic_
MULTI_SPACE = re.compile(r"\s+")
ELLIPSIS = re.compile(r"\.{3,}")
MULTI_PUNC = re.compile(r"([!?])[!?]{1,}")       # !!!?? → ! / ?
SMART_QUOTES = {                                 # 智能引号归一
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u00b4": "'", "\u2032": "'",
}

def _unify_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for k, v in SMART_QUOTES.items():
        s = s.replace(k, v)
    return s

def _strip_markdown_like(text: str) -> str:
    # 去掉强调外壳：`...` / *...* / _..._
    text = _EM_DLM.sub(lambda m: m.group(1), text)
    text = _AST_EM.sub(lambda m: m.group(1), text)
    text = _UND_EM.sub(lambda m: m.group(1), text)
    return text

def normalize_keep_emphasis(text: Any) -> str:
    """删除强调标记（`,*,_），保留内部词面；统一引号/省略号/连写标点与多空格。"""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    s = _unify_unicode(text)
    s = _strip_markdown_like(s)
    s = ELLIPSIS.sub("...", s)
    s = MULTI_PUNC.sub(lambda m: m.group(1), s)
    s = MULTI_SPACE.sub(" ", s)
    return s.strip()



def _first_scalar(x):
    """
    把 list/ndarray/Series 压成“第一个非空标量”；否则原样返回。
    规则：跳过 None、NaN、空串""、"NULL"。
    """
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        for t in x:
            if t is None:
                continue
            if isinstance(t, float) and np.isnan(t):
                continue
            if isinstance(t, str) and (t == "" or t == "NULL"):
                continue
            return t
        return None
    return x

# 划分VA
def parse_va(va: Any) -> Optional[Tuple[float, float]]: 
    """'5.33#5.17' -> (5.33, 5.17)，非法返回 None。"""
    if not isinstance(va, str) or "#" not in va:
        return None
    try:
        v_str, a_str = va.split("#", 1) # maxsplit=1：最多只切一次（从左往右）
        v = float(v_str.strip()); a = float(a_str.strip())
        if any([math.isnan(v), math.isnan(a), math.isinf(v), math.isinf(a)]):
            return None
        return v, a
    except Exception:
        return None


def save_table(df: pd.DataFrame, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True) # 确保“输出文件所在的文件夹”已经创建好。
    # parquet / csv / npz
    df.to_parquet(out_prefix.with_suffix(".parquet"), index=False)
    df.to_csv(out_prefix.with_suffix(".csv"), index=False, encoding="utf-8")
    np.savez(out_prefix.with_suffix(".npz"), **{c: df[c].to_numpy() for c in df.columns})


# ========= 构建训练表 =========
def build_train_table(dfs: List[pd.DataFrame], source_names: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for df, src in zip(dfs, source_names):
        # 用 zip 把两个列表一一配对，每次循环同时拿到一个 DataFrame（df）和它的来源名（src）。
        # 关键点：zip 会以最短的列表为准截断。如果 dfs 和 source_names 长度不一样，超出的那部分会被静默丢掉
        for _, r in df.iterrows():
            text = normalize_keep_emphasis(r.get("Text", ""))
            rid = r.get("ID", None)
            quads = r.get("Quadruplet", []) or []
            for q in quads:
                aspect = normalize_keep_emphasis(q.get("Aspect", "NULL") or "NULL")
                parsed = parse_va(q.get("VA", None))
                if parsed is None:
                    continue
                v, a = parsed
                rows.append({
                    "text": text,
                    "aspect": aspect,
                    "v": float(v),
                    "a": float(a),
                    "id": rid,
                    "category": normalize_keep_emphasis(q.get("Category", None)) if q.get("Category", None) is not None else None,
                    "opinion": normalize_keep_emphasis(q.get("Opinion", None)) if q.get("Opinion", None) is not None else None,
                    "source": src,
                })
    df_out = pd.DataFrame(rows)
    cols = ["text", "aspect", "v", "a", "id", "category", "opinion", "source"]
    return df_out[cols].reset_index(drop=True)


# ========= 构建开发/测试表 =========
def build_dev_table(dfs: List[pd.DataFrame], source_names: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for df, src in zip(dfs, source_names):
        for _, r in df.iterrows():
            rid  = r.get("ID", None)
            text = normalize_keep_emphasis(r.get("Text", ""))  # 去掉强调符号、统一空白/引号等
            emitted = False

            # 1) 若有显式的 Aspect 字段（可能是 str 或 list），逐个输出
            val = r["Aspect"] if ("Aspect" in r) else None
            val = _first_scalar(val)
            if isinstance(val, str) and val not in ("", "NULL"):
                aspects = r["Aspect"]
                if isinstance(aspects, str):
                    aspects = [aspects]
                if isinstance(aspects, list):
                    for asp in aspects:
                        asp_raw = "" if asp is None else str(asp)
                        asp = normalize_keep_emphasis(asp_raw)  # 去掉强调符号，但不改大小写/词形
                        rows.append(
                            {
                                "id": rid,
                                "text": text,
                                "aspect": asp,
                                "aspect_raw": asp_raw,
                                "source": src,
                            }
                        )
                    emitted = True

            # 2) 否则尝试从 Quadruplet 推断（兼容 dict/list）
            if not emitted:
                quads = r.get("Quadruplet", []) or []
                if isinstance(quads, dict):
                    quads = [quads]
                if quads:
                    for q in quads:
                        asp_raw = q.get("Aspect", "NULL") or "NULL"
                        asp = normalize_keep_emphasis(asp_raw)
                        rows.append(
                            {
                                "id": rid,
                                "text": text,
                                "aspect": asp,
                                "aspect_raw": asp_raw,
                                "source": src,
                            }
                        )
                    emitted = True

            # 3) 仍无可用 Aspect，则填充占位
            if not emitted:
                rows.append(
                    {
                        "id": rid,
                        "text": text,
                        "aspect": "NULL",
                        "aspect_raw": "NULL",
                        "source": src,
                    }
                )

    df_out = pd.DataFrame(rows)
    cols = ["id", "text", "aspect", "aspect_raw", "source"]
    return df_out[cols].reset_index(drop=True)



# ========= 主流程 =========
def main():
    # 输出目录用你的工具拼：output/track_a/subtask_1/eng/subtask1_processed
    out_dir = output_path("output","track_a", "subtask_1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 训练集合并（laptop + restaurant）
    df_train = build_train_table(
        [laptop_train, restaurant_train],
        ["eng_laptop_train_alltasks", "eng_restaurant_train_alltasks"]
    )
    save_table(df_train, out_dir / "train_pairs")
    print(f"[OK] Train 转换完成：{len(df_train)} 行 -> {out_dir/'train_pairs'}.[parquet|csv|npz]")

    # 开发集合并（laptop + restaurant）
    df_dev = build_dev_table(
        [laptop_dev, restaurant_dev],
        ["eng_laptop_dev_task1", "eng_restaurant_dev_task1"]
    )
    save_table(df_dev, out_dir / "dev_pairs")
    print(f"[OK] Dev   转换完成：{len(df_dev)} 行 -> {out_dir/'dev_pairs'}.[parquet|csv|npz]")


if __name__ == "__main__":
    main()
