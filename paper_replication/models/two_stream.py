import torch, torch.nn as nn
from .convnext_stream import ConvNeXtStream
from .gcn_tcn import GCNTCN

class TwoStream(nn.Module):
    def __init__(self, num_classes, A_hat, convnext_dim=1024, gcn_dim=512):
        super().__init__()
        self.img = ConvNeXtStream(num_classes, embed_dim=convnext_dim)
        self.skel = GCNTCN(num_classes, A_hat, in_ch=3, embed_dim=gcn_dim)
        # Optional fusion head if you want a fused prediction:
        self.fuse = nn.Sequential(
            nn.Linear(convnext_dim + gcn_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, skepxels, skeleton):
        li, fi = self.img(skepxels)
        ls, fs = self.skel(skeleton)
        lf = self.fuse(torch.cat([fi, fs], dim=1))
        return (li, fi), (ls, fs), lf

def alignment_loss(feat_img, feat_skel, reduction="mean"):
    # Euclidean distance between stream embeddings
    d = torch.norm(feat_img - feat_skel, dim=1)
    if reduction == "mean": return d.mean()
    return d
