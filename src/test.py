# test.py
import pandas as pd
import numpy as np
from pathlib import Path


def report_text_len(jsonl_path: str, text_col: str = "Text"):
    """
    统计 JSONL 文件中某一文本列的【字符长度】分布，并打印关键分位点。
    
    参数说明：
    ----------
    jsonl_path : str
        JSONL 文件路径（每行一个 JSON 对象），例如：
        ../data/track_a/subtask_3/eng/eng_laptop_train_alltasks.jsonl
    text_col : str
        要统计的文本列名，DimABSA 里默认是 "Text"（注意大小写）。
    
    返回：
    ----------
    stats : dict
        各种统计量的字典，比如 count、min、max、p90 等。
    """
    jsonl_path = str(jsonl_path)
    p = Path(jsonl_path)
    assert p.exists(), f"文件不存在：{p}"

    # 关键点1：JSONL 要加 lines=True
    df = pd.read_json(p, lines=True)

    assert text_col in df.columns, f"列不存在：{text_col}，现有列：{list(df.columns)}"

    # 关键点2：用 str.len() 计算字符长度，None/NaN 会先被转成 "nan"，
    # 如果你希望空值长度=0，可以先 fillna("")。
    # 这里选择把 NaN 当成空串处理：
    lens = (
        df[text_col]
        .fillna("")          # 把 NaN 变成 ""
        .astype(str)         # 统一成字符串
        .str.len()           # 计算每个字符串的字符数
        .to_numpy(dtype=np.int32)
    )

    def pct(a: np.ndarray, p: float) -> int:
        """最近邻分位：返回整数长度，数组为空时返回 0。"""
        if a.size == 0:
            return 0
        # numpy>=1.22 支持 method="nearest"
        return int(np.percentile(a, p, method="nearest"))

    if lens.size == 0:
        stats = {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "max": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p98": 0,
            "p99": 0,
            "p100": 0,
        }
    else:
        stats = {
            "count": int(lens.size),
            "min": int(lens.min()),
            "mean": float(lens.mean()),
            "max": int(lens.max()),
            "p50": pct(lens, 50),
            "p75": pct(lens, 75),
            "p90": pct(lens, 90),
            "p95": pct(lens, 95),
            "p98": pct(lens, 98),
            "p99": pct(lens, 99),
            "p100": pct(lens, 100),
        }

    # 打印报告
    print(f"[{p}] 列: {text_col}")
    print("count={count} min={min} mean={mean:.1f} max={max}".format(**stats))
    print(
        "p50={p50} p75={p75} p90={p90} p95={p95} p98={p98} p99={p99} p100={p100}".format(
            **stats
        )
    )

    return stats


if __name__ == "__main__":
    # 用法示例：python test.py
    # 你也可以改成从 sys.argv 读取路径，这里先写死方便快速测试
    report_text_len("../data/track_a/subtask_3/eng/eng_laptop_train_alltasks.jsonl", text_col="Text")
    report_text_len("../data/track_a/subtask_3/eng/eng_restaurant_train_alltasks.jsonl", text_col="Text")
