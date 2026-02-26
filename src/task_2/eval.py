# src/task_2/eval.py
# 离线评测：DimASTE(Subtask2)  — 三元组(A,O,VA) 严格匹配 + 诊断
# 运行示例：
#   python -m src.task_2.eval \
#     --gold data/track_a/subtask_2/eng/eng_laptop_dev_task2.jsonl \
#     --pred data/submit/task2/pred_dev.jsonl \
#     --lower --strip-punct

from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

# 允许直接 python src/task_2/eval.py 运行
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def read_jsonl(path: Path) -> List[dict]:
    data = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

_punct_re = re.compile(r"[^\w\s]")  # 非字母数字与空白

def norm_text(s: str, lower: bool, strip_punct: bool) -> str:
    if s is None:
        return ""
    t = s.strip()
    if lower:
        t = t.lower()
    if strip_punct:
        t = _punct_re.sub(" ", t)
    # 折叠多空格
    t = re.sub(r"\s+", " ", t)
    return t

def parse_va(va_str: str) -> Tuple[float, float]:
    # 期望 "v#a"（可能带空格）
    if not isinstance(va_str, str):
        return (float("nan"), float("nan"))
    parts = va_str.split("#")
    if len(parts) != 2:
        return (float("nan"), float("nan"))
    try:
        v = float(parts[0])
        a = float(parts[1])
        return (v, a)
    except Exception:
        return (float("nan"), float("nan"))

def ints_from_va(va_str: str) -> Tuple[int, int]:
    v, a = parse_va(va_str)
    # 四舍五入到最近整数（1..9）
    def clamp_int(x):
        if math.isnan(x):
            return -999
        r = int(round(x))
        return max(1, min(9, r))
    return clamp_int(v), clamp_int(a)

def triples_from_lines(lines: List[dict], lower: bool, strip_punct: bool) -> Dict[str, List[Tuple[str,str,str]]]:
    """返回: id -> [(A_norm, O_norm, VA_raw_str)]"""
    out: Dict[str, List[Tuple[str,str,str]]] = {}
    for r in lines:
        rid = str(r.get("ID", r.get("id", "")))
        trips = r.get("Triplet", []) or r.get("Triplets", []) or []
        cur = []
        for t in trips:
            a = norm_text(str(t.get("Aspect","")), lower, strip_punct)
            o = norm_text(str(t.get("Opinion","")), lower, strip_punct)
            va = str(t.get("VA",""))
            if a and o:
                cur.append((a,o,va))
        out[rid] = cur
    return out

def pair_sets_for_f1(pred: Dict[str, List[Tuple[str,str,str]]],
                     gold: Dict[str, List[Tuple[str,str,str]]]) -> Tuple[int,int,int,int,int,int]:
    """返回六个计数：TPao, FPao, FNao, TPfull, FPfull, FNfull"""
    TPao = FPao = FNao = 0
    TPfull = FPfull = FNfull = 0
    ids = set(pred.keys()) | set(gold.keys())
    for rid in ids:
        p = pred.get(rid, [])
        g = gold.get(rid, [])
        # A/O 集合（忽略 VA）
        p_ao = {(a,o) for a,o,_ in p}
        g_ao = {(a,o) for a,o,_ in g}
        TPao += len(p_ao & g_ao)
        FPao += len(p_ao - g_ao)
        FNao += len(g_ao - p_ao)
        # 严格集合（A,O, v_int, a_int）
        def _strict_set(lst):
            s = set()
            for a,o,va in lst:
                vi, ai = ints_from_va(va)
                s.add((a,o,vi,ai))
            return s
        p_full = _strict_set(p)
        g_full = _strict_set(g)
        TPfull += len(p_full & g_full)
        FPfull += len(p_full - g_full)
        FNfull += len(g_full - p_full)
    return TPao, FPao, FNao, TPfull, FPfull, FNfull

def precision_recall_f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return prec, rec, f1

def va_metrics_on_intersection(pred: Dict[str, List[Tuple[str,str,str]]],
                               gold: Dict[str, List[Tuple[str,str,str]]]) -> Tuple[float, float, int]:
    """仅在 A/O 匹配的对上，计算整数 RMSE 与整数命中率（v_int,a_int都相等算命中）"""
    errs = []
    hits = 0
    cnt  = 0
    ids = set(pred.keys()) & set(gold.keys())
    for rid in ids:
        p_map = {}
        for a,o,va in pred.get(rid, []):
            p_map.setdefault((a,o), []).append(va)
        for a,o,va_g in gold.get(rid, []):
            key = (a,o)
            if key not in p_map:
                continue
            # 允许一对多：按“就近”策略（与 gold 整数差之和最小）
            vi_g, ai_g = ints_from_va(va_g)
            best = None
            best_err = 1e9
            for va_p in p_map[key]:
                vi_p, ai_p = ints_from_va(va_p)
                e = (vi_p-vi_g)**2 + (ai_p-ai_g)**2
                if e < best_err:
                    best_err = e
                    best = (vi_p, ai_p)
            if best is not None:
                cnt += 1
                rmse_pair = math.sqrt(best_err)  # 欧氏距离（整数空间）
                errs.append(rmse_pair)
                hits += int(best == (vi_g, ai_g))
    if cnt == 0:
        return float("nan"), float("nan"), 0
    rmse = sum(errs)/len(errs)
    acc  = hits / cnt
    return rmse, acc, cnt

def per_id_confusion(pred: Dict[str, List[Tuple[str,str,str]]],
                     gold: Dict[str, List[Tuple[str,str,str]]]) -> List[Tuple[str,int,int,int]]:
    rows = []
    ids = sorted(set(pred.keys()) | set(gold.keys()))
    for rid in ids:
        p_ao = {(a,o) for a,o,_ in pred.get(rid, [])}
        g_ao = {(a,o) for a,o,_ in gold.get(rid, [])}
        tp = len(p_ao & g_ao)
        fp = len(p_ao - g_ao)
        fn = len(g_ao - p_ao)
        rows.append((rid,tp,fp,fn))
    rows.sort(key=lambda x:(x[2]+x[3], -x[1]), reverse=True)
    return rows

def main():
    DEFAULT_GOLD = ROOT / "data/track_a/subtask_2/eng/eng_laptop_dev_task2.jsonl"
    DEFAULT_PRED = ROOT / "data/submit/task2/pred_dev.jsonl"
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=DEFAULT_GOLD, help="gold jsonl 路径（dev_task2.jsonl）")
    ap.add_argument("--pred", default=DEFAULT_PRED, help="你的提交文件 jsonl 路径")
    ap.add_argument("--lower", action="store_true", help="统一转小写后匹配")
    ap.add_argument("--strip-punct", action="store_true", help="去除标点后匹配")
    ap.add_argument("--topk", type=int, default=15, help="诊断中最多显示多少条样本")
    args = ap.parse_args()

    gold_lines = read_jsonl(Path(args.gold))
    pred_lines = read_jsonl(Path(args.pred))

    gold = triples_from_lines(gold_lines, args.lower, args.strip_punct)
    pred = triples_from_lines(pred_lines, args.lower, args.strip_punct)

    print("\n========================================================================")
    print("Subtask2 离线评测（严格匹配 + 诊断）")
    print(f"- 设置: lower={args.lower}, strip_punct={args.strip_punct}")
    print("========================================================================")

    TPao, FPao, FNao, TPfull, FPfull, FNfull = pair_sets_for_f1(pred, gold)
    p_full, r_full, f1_full = precision_recall_f1(TPfull, FPfull, FNfull)
    p_ao,   r_ao,   f1_ao   = precision_recall_f1(TPao, FPao, FNao)

    print("【严格三元组 F1】(A+O+Vint+Aint 完全匹配)")
    print(f" Precision={p_full:.4f}  Recall={r_full:.4f}  F1={f1_full:.4f}   (TP={TPfull} FP={FPfull} FN={FNfull})")
    print("------------------------------------------------------------------------")
    print("【A/O F1】(忽略 VA，只看抽取准确性)")
    print(f" Precision={p_ao:.4f}  Recall={r_ao:.4f}  F1={f1_ao:.4f}   (TP={TPao} FP={FPao} FN={FNao})")
    print("------------------------------------------------------------------------")

    rmse_int, acc_int, n_pairs = va_metrics_on_intersection(pred, gold)
    if n_pairs == 0 or math.isnan(rmse_int):
        print("【VA 评测】A/O 没有交集，无法计算 RMSE/整数命中率")
    else:
        print(f"【VA 评测】在 A/O 命中对上，整数RMSE={rmse_int:.3f}，整数命中率(=1即完全相等)={acc_int:.3f}（对数={n_pairs}）")

    print("========================================================================")
    print("【诊断】错误较多的样本（按 FP+FN 降序）：")
    rows = per_id_confusion(pred, gold)
    for rid,tp,fp,fn in rows[:args.topk]:
        print(f" - ID={rid}  TP={tp}  FP={fp}  FN={fn}")
    print("========================================================================\n")

if __name__ == "__main__":
    main()
