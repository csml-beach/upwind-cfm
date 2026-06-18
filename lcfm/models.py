import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _group_count(channels, requested):
    groups = min(requested, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_ch, groups), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.norm2 = nn.GroupNorm(_group_count(out_ch, groups), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attention2D(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels, groups), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels**-0.5

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(dim=1)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


@register(MODELS, "unet2d")
class UNet2D(nn.Module):
    def __init__(
        self,
        dim,
        image_shape=(3, 32, 32),
        base_channels=128,
        channel_mults=(1, 2, 2, 4),
        num_res_blocks=2,
        time_dim=256,
        attention_resolutions=(16,),
        groups=8,
        num_classes=None,
    ):
        super().__init__()
        self.image_shape = tuple(image_shape)
        channels, height, width = self.image_shape
        if dim != channels * height * width:
            raise ValueError("UNet2D dim must equal product(image_shape).")
        self.dim = dim
        self.channels = channels
        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.label_emb = nn.Embedding(int(num_classes), time_dim) if num_classes is not None else None

        self.init_conv = nn.Conv2d(channels, base_channels, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.attention_resolutions = set(attention_resolutions or [])

        in_ch = base_channels
        resolution = height
        skip_channels = []
        level_channels = [base_channels * int(mult) for mult in channel_mults]
        for level, out_ch in enumerate(level_channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock2D(in_ch, out_ch, time_dim, groups))
                in_ch = out_ch
            attn = Attention2D(in_ch, groups) if resolution in self.attention_resolutions else nn.Identity()
            self.down_blocks.append(nn.ModuleDict({"blocks": blocks, "attention": attn}))
            skip_channels.append(in_ch)
            if level != len(level_channels) - 1:
                self.downsamples.append(nn.Conv2d(in_ch, in_ch, 4, stride=2, padding=1))
                resolution //= 2
            else:
                self.downsamples.append(nn.Identity())

        self.mid1 = ResBlock2D(in_ch, in_ch, time_dim, groups)
        self.mid_attn = Attention2D(in_ch, groups)
        self.mid2 = ResBlock2D(in_ch, in_ch, time_dim, groups)

        for out_ch, skip_ch in zip(reversed(level_channels), reversed(skip_channels)):
            blocks = nn.ModuleList([ResBlock2D(in_ch + skip_ch, out_ch, time_dim, groups)])
            for _ in range(max(0, num_res_blocks - 1)):
                blocks.append(ResBlock2D(out_ch, out_ch, time_dim, groups))
            attn = Attention2D(out_ch, groups) if resolution in self.attention_resolutions else nn.Identity()
            self.up_blocks.append(nn.ModuleDict({"blocks": blocks, "attention": attn}))
            in_ch = out_ch
            resolution *= 2

        self.final_norm = nn.GroupNorm(_group_count(in_ch, groups), in_ch)
        self.final_conv = nn.Conv2d(in_ch, channels, 3, padding=1)

    def forward(self, x, t, y=None):
        b = x.shape[0]
        h = x.reshape(b, *self.image_shape)
        t_emb = self.time_mlp(t.expand(b, 1))
        if self.label_emb is not None:
            if y is None:
                raise ValueError("UNet2D was configured with num_classes and requires labels.")
            t_emb = t_emb + self.label_emb(y.long())
        h = self.init_conv(h)
        skips = []
        for block_group, downsample in zip(self.down_blocks, self.downsamples):
            for block in block_group["blocks"]:
                h = block(h, t_emb)
            h = block_group["attention"](h)
            skips.append(h)
            h = downsample(h)

        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        for block_group in self.up_blocks:
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in block_group["blocks"]:
                h = block(h, t_emb)
            h = block_group["attention"](h)

        h = self.final_conv(F.silu(self.final_norm(h)))
        return h.reshape(b, self.dim)


def build_model(name, dim, config):
    cls = MODELS[name]
    kwargs = dict(config.get("model_kwargs", {}))
    return cls(dim=dim, **kwargs)
