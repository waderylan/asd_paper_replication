#!/usr/bin/env python3
"""
Convert DREAM JSON files into simple .npz samples.

Input (per file):
  JSON with at least:
    data['skeleton'][joint]['x'|'y'|'z'] -> lists of length T
  Optionally:
    data['ados'], data['module'], data['age_months']
    data['start_frame'], data['end_frame']

Output (per file): <out>/<user>_<stem>.npz with:
  - xyz: float32 array (T, U, 3)   # U = number of joints present in the JSON
  - names: array of joint names (len U)
  - ados, module, age_months, start_frame, end_frame: int64 (or -1 if missing)
  - meta: dict with source path, user, stem
"""

import os
import json
import argparse
from typing import Dict, List, Tuple
import numpy as np

SKIP_DIRS = {"dataset_tools", "__pycache__", ".git", ".ipynb_checkpoints"}

def is_json(path: str) -> bool:
    return path.lower().endswith(".json")

def collect_jsons(root: str) -> List[str]:
    files = []
    for cur, dirs, fnames in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in fnames:
            if is_json(f):
                files.append(os.path.join(cur, f))
    files.sort()
    return files

def _to_ndarray(v):
    # robust conversion to ndarray of float32
    a = np.asarray(v, dtype=np.float32)
    return a

def extract_skeleton(data: Dict) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      xyz: (T, U, 3) float32
      names: list[str] length U
    Expects schema like data['skeleton'][joint]['x'|'y'|'z'] -> list length T
    """
    if "skeleton" not in data or not isinstance(data["skeleton"], dict):
        raise ValueError("missing 'skeleton' dict in JSON")

    joints = list(data["skeleton"].keys())
    if len(joints) == 0:
        raise ValueError("no joints inside 'skeleton'")

    # Determine T from the first joint's x
    first = joints[0]
    try:
        T = len(data["skeleton"][first]["x"])
    except Exception:
        raise ValueError("skeleton joint does not have x/y/z arrays")

    names: List[str] = []
    cols: List[np.ndarray] = []  # each will be (T,3)

    for j in joints:
        jdict = data["skeleton"][j]
        # Some dumps might nest differently; we assume x,y,z lists:
        x = _to_ndarray(jdict.get("x", []))
        y = _to_ndarray(jdict.get("y", []))
        z = _to_ndarray(jdict.get("z", []))
        if not (len(x) == len(y) == len(z) == T):
            # If inconsistent length, try to clamp to min length
            tmin = min(len(x), len(y), len(z))
            if tmin == 0:
                # skip this joint entirely
                continue
            x, y, z = x[:tmin], y[:tmin], z[:tmin]
            T = tmin  # shrink T to consistent length across joints

        col = np.stack([x, y, z], axis=1)  # (T,3)
        cols.append(col)
        names.append(j)

    if len(cols) == 0:
        raise ValueError("no usable joints with x/y/z")

    # Stack to (T, U, 3)
    U = len(cols)
    xyz = np.stack(cols, axis=1)  # (T, U, 3)
    # Clean NaNs/Infs
    xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return xyz, names

def read_int_field(data: Dict, key: str, default: int = -1) -> int:
    v = data.get(key, default)
    try:
        return int(v)
    except Exception:
        return default

def convert_one(path: str, out_dir: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    xyz, names = extract_skeleton(data)

    meta = {}
    # common DREAM metadata, if present
    ados = read_int_field(data, "ados", -1)
    module = read_int_field(data, "module", -1)
    age_m = read_int_field(data, "age_months", -1)
    start_f = read_int_field(data, "start_frame", 0)
    end_f   = read_int_field(data, "end_frame", xyz.shape[0]-1)

    # Build output filename: <user>_<stem>.npz
    parts = os.path.normpath(path).split(os.sep)
    user = parts[-2] if len(parts) >= 2 else "User"
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(out_dir, f"{user}_{stem}.npz").replace(" ", "_")

    np.savez_compressed(
        out,
        xyz=xyz,                       # (T,U,3)
        names=np.array(names, dtype=object),
        ados=np.int64(ados),
        module=np.int64(module),
        age_months=np.int64(age_m),
        start_frame=np.int64(start_f),
        end_frame=np.int64(end_f),
        meta=np.array(
            {"source_path": path, "user": user, "stem": stem},
            dtype=object
        ),
    )
    return out, xyz.shape, len(names)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder with DREAM JSONs (e.g., raw_data/dream_dataset/)")
    ap.add_argument("--out",  required=True, help="Output folder for .npz files")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = collect_jsons(args.root)
    if not files:
        print("No JSONs found. Check --root.")
        return

    n_ok, n_err = 0, 0
    for p in files:
        try:
            out, shape, U = convert_one(p, args.out)
            print(f"[ok] {out}  shape={shape} (U={U})")
            n_ok += 1
        except Exception as e:
            print(f"[skip] {p} -> {e}")
            n_err += 1

    print(f"\nDone. wrote={n_ok}, errors={n_err}")

if __name__ == "__main__":
    main()
