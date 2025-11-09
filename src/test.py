# test.py
import pandas as pd
import numpy as np

def report_text_len(csv_path: str, text_col: str = "text"):
    """
    统计 CSV 中 text 列的字符长度分布，并打印关键分位点。
    返回一个包含统计量的 dict。
    """
    df = pd.read_csv(csv_path)
    assert text_col in df.columns, f"列不存在：{text_col}"
    # 转字符串并计算长度（None/NaN -> ""）
    lens = df[text_col].astype(str).map(len).to_numpy(np.int32)

    def pct(a, p):  # 最近邻分位
        return int(np.percentile(a, p, method="nearest")) if len(a) else 0

    stats = {
        "count": int(lens.size),
        "min": int(lens.min()) if lens.size else 0,
        "mean": float(lens.mean()) if lens.size else 0.0,
        "max": int(lens.max()) if lens.size else 0,
        "p50": pct(lens, 50),
        "p75": pct(lens, 75),
        "p90": pct(lens, 90),
        "p95": pct(lens, 95),
        "p98": pct(lens, 98),
        "p99": pct(lens, 99),
        "p100": pct(lens, 100),
    }

    # 打印报告
    print(f"[{csv_path}] 列: {text_col}")
    print("count={count} min={min} mean={mean:.1f} max={max}".format(**stats))
    print("p50={p50} p75={p75} p90={p90} p95={p95} p98={p98} p99={p99} p100={p100}".format(**stats))

    return stats

if __name__ == "__main__":
    # 用法示例：python test.py 时自行修改路径
    report_text_len("../data/output/track_a/subtask_2/dev_pairs.csv")
    report_text_len("../data/output/track_a/subtask_2/train_pairs.csv")
