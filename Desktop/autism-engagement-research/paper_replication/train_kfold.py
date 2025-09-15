import os
import math
import json
import time
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from config import Config
from utils.seed import set_seed
from utils.metrics import accuracy
from utils.kinect_v2 import adjacency
from utils.folds import make_kfold_indices
from datasets.gait_fullbody import GaitFullbody
from datasets.dream import DREAM
from models.two_stream import TwoStream, alignment_loss


def get_dataset(cfg, return_modal="both"):
    if cfg.dataset_name == "gait":
        root = cfg.gait_root
        base = GaitFullbody
        num_classes = 2
    else:
        root = cfg.dream_root
        base = DREAM
        num_classes = 3

    # We need total sample count to form folds
    files = sorted([f for f in os.listdir(root) if f.endswith(".npz")])
    n = len(files)
    folds = make_kfold_indices(n, n_splits=10, seed=cfg.seed)  # expects dicts with 'train'/'test'
    return base, root, folds, num_classes


def run_fold(cfg, base, root, fold_indices, num_classes, fold_id):
    """
    Train/eval one fold; save best (by fused val acc) to checkpoints/.
    Returns best accs dict: {'img': ..., 'skel': ..., 'fuse': ...}
    """
    # Datasets
    ds_train = base(root, fold_indices, train=True,  n_frames=cfg.n_frames, return_modal="both")
    ds_test  = base(root, fold_indices, train=False, n_frames=cfg.n_frames, return_modal="both")

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=cfg.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    # Model
    A_hat = adjacency(cfg.n_joints)
    model = TwoStream(
        num_classes,
        A_hat,
        convnext_dim=cfg.convnext_embed_dim,
        gcn_dim=cfg.gcn_embed_dim
    ).to(cfg.device)

    ce  = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    os.makedirs("checkpoints", exist_ok=True)
    best_accs = {"img": -1.0, "skel": -1.0, "fuse": -1.0}
    best_state = None

    # -------- Joint training with alignment loss ----------
    for epoch in range(cfg.epochs_joint):
        model.train()
        pbar = tqdm(dl_train, desc=f"[joint] epoch {epoch+1}/{cfg.epochs_joint}")
        for b in pbar:
            skep = b["skepxels"].to(cfg.device)   # (B,3,224,224)
            ske  = b["skeleton"].to(cfg.device)   # layout normalized inside model if needed
            y    = b["y"].to(cfg.device)

            (li, fi), (ls, fs), lf = model(skep, ske)

            loss_img = ce(li, y)
            loss_ske = ce(ls, y)
            loss_fus = ce(lf, y)
            loss_aln = cfg.align_lambda * alignment_loss(fi, fs)

            loss = loss_img + loss_ske + loss_fus + loss_aln

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                acc_i = accuracy(li, y)
                acc_s = accuracy(ls, y)
                acc_f = accuracy(lf, y)

            # detach for clean tqdm printing
            pbar.set_postfix(
                loss=float(loss.detach()),
                img=float(acc_i),
                skel=float(acc_s),
                fuse=float(acc_f),
                aln=float(loss_aln.detach())
            )

        # ---- quick eval on this epoch ----
        model.eval()
        tot = {"img": [0.0, 0], "skel": [0.0, 0], "fuse": [0.0, 0]}
        with torch.no_grad():
            for b in dl_test:
                skep = b["skepxels"].to(cfg.device)
                ske  = b["skeleton"].to(cfg.device)
                y    = b["y"].to(cfg.device)

                (li, fi), (ls, fs), lf = model(skep, ske)
                n = y.size(0)

                tot["img"][0]  += float(accuracy(li, y)) * n;  tot["img"][1]  += n
                tot["skel"][0] += float(accuracy(ls, y)) * n;  tot["skel"][1] += n
                tot["fuse"][0] += float(accuracy(lf, y)) * n;  tot["fuse"][1] += n

        accs = {k: (tot[k][0] / max(1, tot[k][1])) for k in tot}
        print(f"VAL img={accs['img']:.4f} skel={accs['skel']:.4f} fuse={accs['fuse']:.4f}")

        # Save best by fused accuracy
        if accs["fuse"] > best_accs["fuse"]:
            best_accs = accs.copy()
            best_state = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "accs": best_accs,
                "cfg": vars(cfg),
            }
            ckpt_path = os.path.join("checkpoints", f"{cfg.dataset_name}_fold{fold_id}_best.pt")
            torch.save(best_state, ckpt_path)

    return best_accs


def main():
    cfg = Config()
    set_seed(cfg.seed)
    base, root, folds, num_classes = get_dataset(cfg)

    fold_metrics = []
    for i, fold in enumerate(folds):
        print(f"\n========== Fold {i+1}/{len(folds)} ==========")
        accs = run_fold(cfg, base, root, fold, num_classes, fold_id=i+1)
        fold_metrics.append(accs)

    # Report mean±std over folds (fused and per-stream) + save summary
    import numpy as np
    os.makedirs("results", exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")

    summary = {"dataset": cfg.dataset_name, "seed": cfg.seed, "timestamp": ts, "folds": fold_metrics}
    for key in ("img", "skel", "fuse"):
        arr = np.array([m[key] for m in fold_metrics], dtype=np.float32)
        mean, std = float(arr.mean()), float(arr.std())
        print(f"{key}: mean={mean:.4f}  std={std:.4f}")
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std

    # Save JSON summary
    json_path = os.path.join("results", f"summary_{cfg.dataset_name}_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {json_path}")

    # Save per-fold CSV
    csv_path = os.path.join("results", f"folds_{cfg.dataset_name}_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "img", "skel", "fuse"])
        w.writeheader()
        for i, m in enumerate(fold_metrics, start=1):
            w.writerow({"fold": i, "img": m["img"], "skel": m["skel"], "fuse": m["fuse"]})
    print(f"Saved per-fold CSV -> {csv_path}")


if __name__ == "__main__":
    main()
