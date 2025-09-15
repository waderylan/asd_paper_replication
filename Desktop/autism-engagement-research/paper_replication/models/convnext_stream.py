import torch, torch.nn as nn
import torchvision.models as tvm

class ConvNeXtStream(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 1024):
        super().__init__()
        m = tvm.convnext_base(weights=None)  # or ConvNeXt-Tiny if you want faster
        self.backbone = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(1024, embed_dim)
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: (B,3,224,224)
        returns: logits (B,C), feat (B,D)
        """
        h = self.backbone(x)          # (B,1024,H',W')
        h = self.pool(h).flatten(1)   # (B,1024)
        feat = self.embed(h)          # (B,D)
        feat = torch.relu(feat)
        logits = self.cls(feat)
        return logits, feat
