import os, numpy as np, torch
from torch.utils.data import Dataset
from preprocessing.skepxels import build_skepxels
from preprocessing.angle_embed import embed_with_angles

class GaitFullbody(Dataset):
    """
    Assumes you have pre-extracted per-sample arrays:
      sample['xyz']: (T, 25, 3) float32, T>=64 (we will downsample/pad to 64)
      sample['label']: 0=TD, 1=ASD
    You can write a small script to convert Dryad CSVs to these npz files.
    """
    def __init__(self, root, fold_indices, train=True, n_frames=64, return_modal="both"):
        self.return_modal = return_modal  # "both" | "skepxels" | "skeleton"
        self.files = sorted([os.path.join(root,f) for f in os.listdir(root) if f.endswith(".npz")])
        self.indices = fold_indices["train" if train else "test"]
        self.n_frames = n_frames

    def __len__(self): return len(self.indices)

    def _fix_len(self, x):
        T = x.shape[0]
        if T == self.n_frames: return x
        if T > self.n_frames:
            idx = np.linspace(0, T-1, self.n_frames).astype(int)
            return x[idx]
        out = np.zeros((self.n_frames, x.shape[1], x.shape[2]), dtype=x.dtype)
        out[:T] = x
        return out

    def __getitem__(self, idx):
        path = self.files[self.indices[idx]]
        arr = np.load(path, allow_pickle=True)
        xyz = arr["xyz"].astype(np.float32)           # (T,25,3)
        y   = int(arr["label"])
        xyz = self._fix_len(xyz)

        if self.return_modal in ("both","skepxels"):
            sk = build_skepxels(xyz)                  # (3,224,224)
        if self.return_modal in ("both","skeleton"):
            ae = embed_with_angles(xyz)               # (64,25,3)
            ae = ae.transpose(0, 2, 1)                # (3,64,25)  -> (C,T,J)
        out = {}
        if self.return_modal in ("both","skepxels"): out["skepxels"] = torch.from_numpy(sk).float()/255.0
        if self.return_modal in ("both","skeleton"): out["skeleton"] = torch.from_numpy(ae).float()
        out["y"] = torch.tensor(y, dtype=torch.long)
        return out
