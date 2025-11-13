import random, numpy as np, pandas as pd, torch
from pathlib import Path

def load_csvs(path_like: str) -> pd.DataFrame:
    p = Path(path_like)
    if p.is_dir():
        files = sorted([*p.glob("*.csv")])
        if not files: raise FileNotFoundError(f"No CSVs under {p}")
        dfs = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(p, encoding="utf-8-sig")
    return df

def get_feature_cols(df: pd.DataFrame):
    """
    改成固定只抓 ap0 ~ ap255 共 256 維
    """
    all_cols = set(df.columns)
    ap_cols = [f"ap{i}" for i in range(256)]
    missing = [c for c in ap_cols if c not in all_cols]
    if missing:
        raise ValueError(f"資料集中缺少欄位（ap0~ap255）：{missing[:10]} ...")
    return ap_cols

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
