import math

import torch
import torch.nn as nn

from .registry import MODELS, register


@register(MODELS, "mlp")
class VelocityMLP(nn.Module):
    def __init__(self, dim, hidden=128, depth=3):
        super().__init__()
        layers = []
        in_dim = dim + 1
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden), nn.SiLU()])
            in_dim = hidden
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_expand = t.expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_expand], dim=-1))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Block1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))

    def forward(self, x, t_emb):
        return self.conv(x) + self.time_mlp(t_emb)[:, :, None]


@register(MODELS, "unet1d")
class UNet1D(nn.Module):
    def __init__(self, dim, hidden=64, time_dim=256):
        super().__init__()
        del dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.down1 = Block1D(1, hidden, time_dim)
        self.pool1 = nn.MaxPool1d(2)
        self.down2 = Block1D(hidden, hidden * 2, time_dim)
        self.pool2 = nn.MaxPool1d(2)
        self.mid = Block1D(hidden * 2, hidden * 2, time_dim)
        self.up1 = nn.ConvTranspose1d(hidden * 2, hidden * 2, 2, stride=2)
        self.block_up1 = Block1D(hidden * 4, hidden, time_dim)
        self.up2 = nn.ConvTranspose1d(hidden, hidden, 2, stride=2)
        self.block_up2 = Block1D(hidden * 2, hidden, time_dim)
        self.out = nn.Conv1d(hidden, 1, 1)

    def forward(self, x, t):
        x = x.unsqueeze(1)
        t_emb = self.time_mlp(t.expand(x.shape[0], 1))
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)
        xm = self.mid(self.pool2(x2), t_emb)
        u1 = self.up1(xm)
        u1 = self.block_up1(torch.cat([u1, x2], dim=1), t_emb)
        u2 = self.up2(u1)
        u2 = self.block_up2(torch.cat([u2, x1], dim=1), t_emb)
        return self.out(u2).squeeze(1)


def build_model(name, dim, config):
    cls = MODELS[name]
    kwargs = dict(config.get("model_kwargs", {}))
    return cls(dim=dim, **kwargs)
