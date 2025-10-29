# src/utils/paths.py
from pathlib import Path
import os

def project_root() -> Path:
    """
    返回项目根目录的 Path。兼容本地源码、云端 notebook、以及不同工作目录。
    优先级：
      1) 环境变量 DIMABSA_ROOT
      2) 以 __file__ 为锚（源码运行）
      3) Notebook/未知环境下退化为 cwd
      4) 额外：向上搜索含有 'data' 子目录的父路径（可选）
    """
    env = os.getenv("DIMABSA_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    # 尝试用 __file__（源码/模块方式）
    try:
        here = Path(__file__).resolve()
        root = here.parents[2]  # .../DimABSA2026/
        if (root / "data").exists():
            return root
        # 防御：向上搜索直到根
        cur = here
        while cur != cur.parent:
            if (cur / "data").exists() and (cur / "src").exists():
                return cur
            cur = cur.parent
        # 最后兜底
        return Path.cwd().resolve()
    except NameError:
        # __file__ 不存在（Notebook）
        cwd = Path.cwd().resolve()
        # 若当前目录就是项目根（含 data 与 src），用它
        if (cwd / "data").exists() and (cwd / "src").exists():
            return cwd
        return cwd  # 或者在 notebook 里手动设置 DIMABSA_ROOT

def data_path(*parts) -> Path:
    return project_root() / "data" / Path(*parts)

def output_path(*parts) -> Path:
    # 云端写路径：优先环境变量 OUT_DIR，其次项目内 reports/，再退到 /tmp
    out_dir = os.getenv("DIMABSA_OUT") or (project_root() / "data")
    p = Path(out_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    fp = p / Path(*parts)
    fp.parent.mkdir(parents=True, exist_ok=True)
    return fp
