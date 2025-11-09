from __future__ import annotations
from pathlib import Path
import math, re, unicodedata
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.utils.paths import data_path, output_path

# ========= 读取 =========
lp_dev   = data_path("track_a", "subtask_2", "eng", "eng_laptop_dev_task2.jsonl")
lp_train = data_path("track_a", "subtask_2", "eng", "eng_laptop_train_alltasks.jsonl")
rs_dev   = data_path("track_a", "subtask_2", "eng", "eng_restaurant_dev_task2.jsonl")
rs_train = data_path("track_a", "subtask_2", "eng", "eng_restaurant_train_alltasks.jsonl")

for p in [lp_dev, lp_train, rs_dev, rs_train]:
    assert p.exists(), f"未找到数据文件：{p}. 如在云端，请先设置环境变量 DIMABSA_ROOT=你的项目根目录"

laptop_dev       = pd.read_json(lp_dev, lines=True)
restaurant_dev   = pd.read_json(rs_dev, lines=True)
laptop_train     = pd.read_json(lp_train, lines=True)
restaurant_train = pd.read_json(rs_train, lines=True)


# ===================== 文本标准化：normalize_keep_emphasis =====================
_EM_DLM = re.compile(r"`+([^`]+?)`+")  # 处理用反引号包裹的强调：`like this`
_AST_EM = re.compile(r"\*(\S(?:.*?\S)?)\*")  # *italic* 风格
_UND_EM = re.compile(r"_(\S(?:.*?\S)?)_")      # _italic_ 风格

MULTI_SPACE = re.compile(r"\s+")
ELLIPSIS = re.compile(r"\.{3,}")
MULTI_PUNC = re.compile(r"([!?])[!?]{1,}")   # 合并 !!!?? → ! / ?

SMART_QUOTES = {
    "\u2018": "'", "\u2019": "'",  # ‘ ’
    "\u201c": '"', "\u201d": '"',  # “ ”
    "\u00b4": "'", "\u2032": "'",
}


# ========= 工具 =========
def _unify_unicode(s: str) -> str:
    # NFKC 归一化 + 智能引号替换
    s = unicodedata.normalize("NFKC", s)
    for k, v in SMART_QUOTES.items():
        s = s.replace(k, v)
    return s


def _strip_markdown_like(text: str) -> str:
    """去掉部分 Markdown 强调标记，但保留被强调词面本身。
    - `code` → code
    - *em*   → em
    - _em_   → em
    """
    # 反引号强调
    text = _EM_DLM.sub(lambda m: m.group(1), text)
    # 星号/下划线强调
    text = _AST_EM.sub(lambda m: m.group(1), text)
    text = _UND_EM.sub(lambda m: m.group(1), text)
    return text


def _norm_str(text: Any) -> Optional[str]:
    """标准化文本，但“保留强调的词面”——即删除强调标记（`, *, _），不改变其中词的大小写/形态。
    其他规则：
      - Unicode 归一化/引号统一
      - 合并多空格
      - 归一省略号与连写标点（!!!?? → !/?）
      - 去掉首尾空白
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    s = _unify_unicode(text)
    s = _strip_markdown_like(s)
    s = ELLIPSIS.sub("...", s)
    s = MULTI_PUNC.sub(lambda m: m.group(1), s)
    s = MULTI_SPACE.sub(" ", s)
    return s.strip()

def _clip19(v: float) -> float:
    return float(max(1.0, min(9.0, v)))

def _fmt_va_str(v: float, a: float) -> str:
    # 约束到 [1.00, 9.00] 并四舍五入两位；返回 "V#A"
    v = max(1.0, min(9.0, float(v)))
    a = max(1.0, min(9.0, float(a)))
    return f"{v:.2f}#{a:.2f}"

def parse_va(va: Any) -> Optional[Tuple[float, float]]:
    """'5.33#5.17' -> (5.33, 5.17)，非法返回 None，并裁剪到 [1,9]."""
    if not isinstance(va, str) or "#" not in va:
        return None
    try:
        v_str, a_str = va.split("#", 1)
        v = float(v_str.strip()); a = float(a_str.strip())
        if any([math.isnan(v), math.isnan(a), math.isinf(v), math.isinf(a)]):
            return None
        return _clip19(v), _clip19(a)
    except Exception:
        return None

def save_table(df: pd.DataFrame, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_prefix.with_suffix(".parquet"), index=False)
    df.to_csv(out_prefix.with_suffix(".csv"), index=False, encoding="utf-8")
    # 额外存 jsonl 便于抽样检查
    with out_prefix.with_suffix(".jsonl").open("w", encoding="utf-8") as w:
        for _, r in df.iterrows():
            w.write(r.to_json(force_ascii=False) + "\n")

# ========= 构建训练表 =========
def build_train_table(dfs: List[pd.DataFrame], source_names: List[str], domains: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    drop_va, total = 0, 0
    for df, src, dom in zip(dfs, source_names, domains):
        for _, r in df.iterrows():
            text = _norm_str(r.get("Text", ""))
            rid  = r.get("ID", None)
            quads = r.get("Quadruplet", []) or []
            # quad 可能是 dict 或 list
            if isinstance(quads, dict):
                quads = [quads]
            for q in quads:
                total += 1
                aspect   = _norm_str(q.get("Aspect", None))
                opinion  = _norm_str(q.get("Opinion", None))
                category = _norm_str(q.get("Category", None))
                if category is not None:
                    category = category.upper()
                parsed = parse_va(q.get("VA", None))
                if parsed is None:
                    drop_va += 1
                    continue
                v, a = parsed
                rows.append({
                    "id": rid,
                    "text": text or "",
                    "aspect": aspect,             # 允许 None（对抽取器是有效监督）
                    "opinion": opinion,           # 允许 None
                    "category": category,         # 允许 None
                    "v": v, "a": a,               # 已裁剪到 [1,9]
                    "VA": _fmt_va_str(v, a),
                    "source": src,
                    "domain": dom,                # laptop / restaurant
                })
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        # 去重（完全相同的行）
        df_out = df_out.drop_duplicates(
            subset=["id","text","aspect","opinion","category","VA","v","a","source","domain"],
            keep="first"    
        ).reset_index(drop=True)
    print(f"[Train] 总四元组: {total} | 无效VA丢弃: {drop_va} | 保留: {len(df_out)}")
    cols = ["id","text","aspect","opinion","category","VA","v","a","source","domain"]
    return df_out[cols]

# ========= 构建开发表 =========
def build_dev_table(dfs: List[pd.DataFrame], source_names: List[str], domains: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for df, src, dom in zip(dfs, source_names, domains):
        for _, r in df.iterrows():
            rows.append({
                "id":     r.get("ID", None),
                "text":   _norm_str(r.get("Text", "")) or "",
                "source": src,
                "domain": dom,
            })
    df_out = pd.DataFrame(rows).drop_duplicates(subset=["id","text","source","domain"]).reset_index(drop=True)
    cols = ["id","text","source","domain"]
    return df_out[cols]

# ========= 主流程 =========
def main():
    out_dir = output_path("output","track_a","subtask_2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 训练集合并（laptop + restaurant）
    df_train = build_train_table(
        [laptop_train, restaurant_train],
        ["eng_laptop_train_alltasks", "eng_restaurant_train_alltasks"],
        ["laptop", "restaurant"]
    )
    save_table(df_train, out_dir / "train_pairs")
    print(f"[OK] Train 转换完成：{len(df_train)} 行 -> {out_dir/'train_pairs'}.[parquet|csv|jsonl]")

    # 开发集合并（laptop + restaurant）
    df_dev = build_dev_table(
        [laptop_dev, restaurant_dev],
        ["eng_laptop_dev_task2", "eng_restaurant_dev_task2"],
        ["laptop", "restaurant"]
    )
    save_table(df_dev, out_dir / "dev_pairs")
    print(f"[OK] Dev   转换完成：{len(df_dev)} 行 -> {out_dir/'dev_pairs'}.[parquet|csv|jsonl]")

if __name__ == "__main__":
    main()
