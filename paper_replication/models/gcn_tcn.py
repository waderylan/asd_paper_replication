import torch, torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_ch, out_ch, A_hat):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(A_hat, dtype=torch.float32), requires_grad=False)
        self.theta = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T, J)
        B,C,T,J = x.shape
        x = x.permute(0,2,1,3)             # (B,T,C,J)
        x = torch.einsum('b t c j, j k -> b t c k', x, self.A)  # graph message passing
        x = x.permute(0,2,1,3)             # (B,C,T,J)
        x = self.theta(x)
        return x

class TemporalConv(nn.Module):
    def __init__(self, ch, ks=9):
        super().__init__()
        pad = ks//2
        self.conv = nn.Conv2d(ch, ch, kernel_size=(ks,1), padding=(pad,0), groups=ch)
        self.pw   = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x):
        h = self.conv(x)
        h = F.relu(self.pw(h))
        return h

class GCNTCN(nn.Module):
    def __init__(self, num_classes, A_hat, in_ch=3, embed_dim=512):
        super().__init__()
        self.g1 = GraphConv(in_ch, 64, A_hat)
        self.t1 = TemporalConv(64)
        self.g2 = GraphConv(64, 128, A_hat)
        self.t2 = TemporalConv(128)
        self.g3 = GraphConv(128, 256, A_hat)
        self.t3 = TemporalConv(256)
        self.pool_t = nn.AdaptiveAvgPool2d((1,1))
        self.embed = nn.Linear(256, embed_dim)
        self.cls   = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x arrives from the dataset as (B, T, C, J). GraphConv expects (B, C, T, J).
        Normalize once here.
        """
        # If joints aren't last, move them to last (safety for any caller)
        J = self.g1.A.size(0)
        if x.shape[-1] != J:
            # find dim that equals J and move it to the end
            jd = [d for d in range(1,4) if x.shape[d] == J][0]
            order = [0,1,2,3]; order.remove(jd); order.append(jd)
            x = x.permute(*order).contiguous()      # now joints last

        # If layout is (B, T, C, J), swap to (B, C, T, J)
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3).contiguous()

        # Now x is (B, C, T, J)
        h = F.relu(self.g1(x)); h = self.t1(h)
        h = F.relu(self.g2(h)); h = self.t2(h)
        h = F.relu(self.g3(h)); h = self.t3(h)
        h = self.pool_t(h).squeeze(-1).squeeze(-1)  # (B,256)
        feat = torch.relu(self.embed(h))
        logits = self.cls(feat)
        return logits, feat
