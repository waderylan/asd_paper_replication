import os, glob, numpy as np
from torch.utils.data import Dataset

# Kinect v2 (25) canonical order we’ll target
K2_ORDER = [
    "spine_base","spine_mid","neck","head",
    "shoulder_left","elbow_left","wrist_left","hand_left","handtip_left","thumb_left",
    "shoulder_right","elbow_right","wrist_right","hand_right","handtip_right","thumb_right",
    "hip_left","knee_left","ankle_left","foot_left",
    "hip_right","knee_right","ankle_right","foot_right",
    "spine_shoulder",
]
# common DREAM upper-body names → k2 names (extend if your JSON uses other spellings)
NAME_MAP = {
    "head":"head","neck":"neck","spine_shoulder":"spine_shoulder","spine_mid":"spine_mid",
    "shoulder_left":"shoulder_left","elbow_left":"elbow_left","wrist_left":"wrist_left","hand_left":"hand_left",
    "shoulder_right":"shoulder_right","elbow_right":"elbow_right","wrist_right":"wrist_right","hand_right":"hand_right",
    # if DREAM has 'midspine' or 'spine_mid' variants, normalize them here:
    "midspine":"spine_mid","midspain":"spine_mid",  # just in case
}

def pad_to_25(xyz, names):
    """
    xyz: (T,U,3) from DREAM .npz, names: list[str] of length U
    returns (T,25,3) in Kinect v2 order; missing joints filled with 0
    """
    T = xyz.shape[0]
    out = np.zeros((T, 25, 3), dtype=np.float32)
    name2idx = {n.lower(): i for i, n in enumerate(names)}
    for j, k2name in enumerate(K2_ORDER):
        # try exact, then mapped
        src = name2idx.get(k2name, None)
        if src is None:
            # try alias
            for k, v in NAME_MAP.items():
                if v == k2name and k in name2idx:
                    src = name2idx[k]
                    break
        if src is not None:
            out[:, j, :] = xyz[:, src, :]
        # else stays zero (lower body typically)
    return out

def time_to_64(xyz25):
    """
    xyz25: (T,25,3)  -> (64,25,3), by uniform sampling or zero-pad
    """
    T = xyz25.shape[0]
    if T == 64:
        return xyz25
    if T > 64:
        idx = np.linspace(0, T-1, 64).round().astype(int)
        return xyz25[idx]
    # T < 64: pad
    out = np.zeros((64, 25, 3), dtype=np.float32)
    out[:T] = xyz25
    return out

def ados_to_label(ados: int, module: int, age_months: int) -> int:
    """
    Map ADOS + module + age (in months) -> class label
    Returns: 0 = NS, 1 = ASD, 2 = AUT
    """
    # ---- Module 1 (younger group) ----
    if module == 1:
        if ados <= 10:
            return 0  # NS
        elif 10 < ados <= 15:
            return 1  # ASD
        else:  # ados >= 15
            return 2  # AUT

    # ---- Module 2 (older group) ----
    elif module == 2:
        # Special tweak: 3–4 years (36–48 months)
        if 36 <= age_months <= 48:
            if ados == 6:
                return 0  # NS (narrowed down)
            elif 8 <= ados <= 9:
                return 1  # ASD
            elif ados > 9:
                return 2  # AUT
            else:
                return 1  # fallback: treat as ASD if ambiguous
        else:
            if ados <= 6:
                return 0  # NS
            elif 8 <= ados <= 9:
                return 1  # ASD
            elif ados > 9:
                return 2  # AUT
            else:
                return 1  # fallback for ambiguous case

    else:
        raise ValueError(f"Unknown module {module}, expected 1 or 2")
                         # otherwise ASD

class DREAM(Dataset):
    def __init__(self, root, split_indices=None, train=True, return_modal="both"):
        """
        root: folder with *.npz produced by convert_dream_to_npz.py
        split_indices: list of indices for this split (10-fold CV can pass these)
        return_modal: "both" | "skepxels" | "skeleton"
        """
        paths = sorted(glob.glob(os.path.join(root, "*.npz")))
        if split_indices is None:
            self.paths = paths
        else:
            self.paths = [paths[i] for i in split_indices]
        self.return_modal = return_modal

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        rec = np.load(self.paths[i], allow_pickle=True)
        xyz, names = rec["xyz"].astype(np.float32), list(rec["names"])
        ados, module, age = int(rec["ados"]), int(rec["module"]), int(rec["age_months"])
        # 1) joints → 25
        xyz25 = pad_to_25(xyz, names)                # (T,25,3)
        # 2) frames → 64
        xyz64 = time_to_64(xyz25)                    # (64,25,3)
        # 3) label
        y = ados_to_label(ados, module, age)         # 0/1/2

        # Package for the two streams (your training loop may build skepxels on the fly)
        sample = {
            "skeleton": np.transpose(xyz64, (2,0,1)).astype(np.float32),  # (3,64,25) for graph stream
            "y": np.int64(y),
        }
        # If you prebuild Skepxels, put it here as (3,224,224); otherwise generate in your skepxel transform
        return sample
