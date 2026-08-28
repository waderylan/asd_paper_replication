#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd
from typing import List, Tuple, Dict

SKIP_FILE_SUBSTR = ["feature", "final dataset", "_2d", " 2d", "-2d", "2d.xlsx", "~$", ".tmp"]
SKIP_DIR_SUBSTR  = ["augment", "augmentation", "__macosx"]

N_JOINTS, N_COORDS = 25, 3
NEEDED = N_JOINTS * N_COORDS  # 75

# If headers include joint names, we can prioritize them:
JOINT_HINTS = ["spine", "neck", "head", "shoulder", "elbow", "wrist", "hand", "thumb",
               "hip", "knee", "ankle", "foot"]

def is_excel(fn): return fn.lower().endswith((".xlsx", ".xls"))
def skip_file(fn): return any(s in fn.lower() for s in SKIP_FILE_SUBSTR)
def skip_dir(dn):  return any(s in dn.lower() for s in SKIP_DIR_SUBSTR)

def collect_xlsx(root: str) -> List[str]:
    paths = []
    for cur, dirs, files in os.walk(root):
        # prune unwanted dirs
        dirs[:] = [d for d in dirs if not skip_dir(d)]
        for f in files:
            if is_excel(f) and not skip_file(f):
                full = os.path.join(cur, f)
                # hard-skip any "severe levels of asd" (or any 'severe' mention)
                if "severe" in full.lower():
                    continue
                paths.append(full)
    return sorted(paths)

def coerce_numeric(df: pd.DataFrame, min_numeric_ratio: float = 0.8) -> pd.DataFrame:
    """
    Coerce every column to numeric; keep columns where >= min_numeric_ratio of entries are numeric.
    """
    keep = []
    for col in df.columns:
        col_series = pd.to_numeric(df[col], errors="coerce")
        ratio = col_series.notna().mean()  # fraction numeric
        if ratio >= min_numeric_ratio:
            keep.append(col_series)
    if not keep:
        return pd.DataFrame()
    out = pd.concat(keep, axis=1)
    out = out.fillna(0.0).astype(np.float32)  # zeros for non-numeric cells
    return out

def choose_75_columns(df_num: pd.DataFrame, prefer: str = "first") -> np.ndarray:
    """
    df_num is fully numeric (float32). Select 75 columns:
      1) If headers contain joint-like names, prioritize those columns first.
      2) If not enough, use remaining columns by 'prefer' policy.
    Returns (T,75) float32 or raises ValueError.
    """
    T, K = df_num.shape
    if K < NEEDED:
        raise ValueError(f"only {K} numeric-ish columns after coercion, need >= {NEEDED}")

    # Try priority by header names:
    colnames = [str(c).lower() for c in df_num.columns]
    pri_idx = [i for i, n in enumerate(colnames) if any(h in n for h in JOINT_HINTS)]
    pri_idx = pri_idx[:NEEDED]
    picked = []
    used = set()
    for i in pri_idx:
        picked.append(i); used.add(i)

    # Fill the rest by order:
    if prefer == "last":
        rng = range(K-1, -1, -1)
    else:
        rng = range(K)
    for i in rng:
        if len(picked) >= NEEDED: break
        if i in used: continue
        picked.append(i); used.add(i)

    picked = picked[:NEEDED]
    arr = df_num.iloc[:, picked].to_numpy(dtype=np.float32)  # (T,75)
    return arr

def load_trial(path: str, prefer: str, min_numeric_ratio: float) -> np.ndarray:
    # let pandas infer header; if it fails, fallback no header
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception:
        df = pd.read_excel(path, header=None, engine="openpyxl")

    # Coerce every column to numeric with tolerance for occasional text
    df_num = coerce_numeric(df, min_numeric_ratio=min_numeric_ratio)
    if df_num.shape[1] < NEEDED:
        raise ValueError(f"{path}: numeric-ish cols={df_num.shape[1]} < {NEEDED}")

    arr75 = choose_75_columns(df_num, prefer=prefer)  # (T,75)
    xyz = arr75.reshape(arr75.shape[0], N_JOINTS, N_COORDS)  # (T,25,3)
    return np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def detect_label_and_index(path: str) -> Tuple[int, str]:
    """
    Returns:
      label: 0 for Typical, 1 for Autism; -1 if unknown
      idx: the folder name immediately under 'Autism/' or 'Typical/' (used in filename)
    """
    parts = os.path.normpath(path).split(os.sep)
    lowparts = [p.lower() for p in parts]

    # Find 'autism' or 'typical' anchor and take the next component as index
    label, idx = -1, None
    if "autism" in lowparts:
        anchor = lowparts.index("autism")
        if anchor + 1 < len(parts):
            idx = parts[anchor + 1]
            label = 1
    if "typical" in lowparts:
        anchor = lowparts.index("typical")
        if anchor + 1 < len(parts):
            idx = parts[anchor + 1]
            label = 0

    return label, (idx or "unknown")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="GAIT dataset root (has Autism/ and Typical/)")
    ap.add_argument("--out", required=True, help="Output folder for .npz")
    ap.add_argument("--numeric-slice", choices=["first","last"], default="first")
    ap.add_argument("--min-numeric-ratio", type=float, default=0.8,
                    help="Keep a column if >= this fraction is numeric")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = collect_xlsx(args.root)
    print(f"Found {len(files)} candidate xlsx files")

    wrote = skipped = 0
    reasons: Dict[str, int] = {}

    for p in files:
        # skip anything with 'severe' anywhere in path (safety net)
        if "severe" in p.lower():
            continue

        label, idx = detect_label_and_index(p)
        if label not in (0, 1):
            reasons["unknown_group"] = reasons.get("unknown_group", 0) + 1
            continue

        try:
            xyz = load_trial(p, args.numeric_slice, args.min_numeric_ratio)
            base = f"autism_{idx}.npz" if label == 1 else f"typical_{idx}.npz"
            out = os.path.join(args.out, base).replace(" ", "_")
            np.savez_compressed(out,
                                xyz=xyz,
                                label=np.int64(label))
            print(f"[ok] {os.path.basename(out)}  shape={xyz.shape}  label={label}")
            wrote += 1
        except Exception as e:
            key = str(e).split(":")[0]
            print(f"[skip] {p} -> {e}")
            reasons[key] = reasons.get(key, 0) + 1
            skipped += 1

    print(f"\nDone. wrote={wrote}, skipped={skipped}")
    if reasons:
        print("Skip reasons:")
        for k, v in reasons.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
