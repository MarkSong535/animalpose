"""Frame-wise behavior classification from pose sequences.

:class:`EgocentricFeatures` turns raw keypoints into view- and scale-robust
features inside the graph (no Python-side preprocessing in deployment):
coordinates are centred on the animal, rotated so the body axis
(posterior -> anterior group) points along +x, and scaled by the sequence's
mean body length. Per-frame apparent body length, centroid velocity, turning
rate and per-keypoint speeds are appended. ``D = 3`` input (triangulated,
z up) keeps the height channel, which makes rearing directly observable.

:class:`BehaviorTCN` is a dilated temporal convolution stack producing
per-frame logits, so it runs on any sequence length and exports cleanly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from animalpose.labels import BEHAVIORS
from animalpose.skeleton import CANONICAL


def _shift(x: torch.Tensor) -> torch.Tensor:
    """x[t-1] with x[-1] := x[0], along dim 1."""
    return torch.cat([x[:, :1], x[:, :-1]], dim=1)


class EgocentricFeatures(nn.Module):
    def __init__(self, num_keypoints: int = 25, dims: int = 2, anterior=None, posterior=None):
        super().__init__()
        self.k, self.d = num_keypoints, dims
        ant = torch.zeros(num_keypoints)
        post = torch.zeros(num_keypoints)
        ant[anterior or CANONICAL.anterior] = 1.0
        post[posterior or CANONICAL.posterior] = 1.0
        self.register_buffer("ant", ant)
        self.register_buffer("post", post)

    @property
    def out_dim(self) -> int:
        return self.k * self.d + 2 * self.k + 6

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(x.dtype)
        x = torch.nan_to_num(x) * m.unsqueeze(-1)
        eps = 1e-6

        def group(w):
            ww = m * w
            return (x * ww.unsqueeze(-1)).sum(2) / ww.sum(2, keepdim=True).clamp(min=eps)

        centroid = (x * m.unsqueeze(-1)).sum(2) / m.sum(2, keepdim=True).clamp(min=eps)
        axis = group(self.ant) - group(self.post)
        length = torch.sqrt((axis[..., :2] ** 2).sum(-1, keepdim=True) + eps)  # (B,T,1)
        u = axis[..., :2] / length
        ref = length.mean(dim=1, keepdim=True).clamp(min=eps)  # (B,1,1)
        d = x - centroid.unsqueeze(2)
        ux, uy = u[..., 0:1], u[..., 1:2]
        ex = d[..., 0] * ux + d[..., 1] * uy
        ey = -d[..., 0] * uy + d[..., 1] * ux
        ego = [ex, ey] + ([d[..., 2]] if self.d == 3 else [])
        ego = torch.stack(ego, dim=-1) / ref.unsqueeze(-1) * m.unsqueeze(-1)  # (B,T,K,D)
        vel = (centroid - _shift(centroid))[..., :2]
        vx = vel[..., 0:1] * ux + vel[..., 1:2] * uy
        vy = -vel[..., 0:1] * uy + vel[..., 1:2] * ux
        u_prev = _shift(u)
        turn_sin = u_prev[..., 0:1] * uy - u_prev[..., 1:2] * ux
        turn_cos = (u_prev * u).sum(-1, keepdim=True)
        kp_speed = torch.sqrt(((ego - _shift(ego)) ** 2).sum(-1) + eps) * m * _shift(m)
        speed = torch.sqrt(vx**2 + vy**2 + eps) / ref
        b, t = x.shape[:2]
        return torch.cat([ego.reshape(b, t, -1), m, kp_speed, vx / ref, vy / ref, speed,
                          length / ref, turn_sin, turn_cos], dim=-1)


class TCNBlock(nn.Module):
    def __init__(self, c: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(c, c, 1)
        self.norm = nn.BatchNorm1d(c)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.conv2(F.relu(self.norm(self.conv1(x)))))


class BehaviorTCN(nn.Module):
    def __init__(self, num_keypoints: int = 25, dims: int = 2, classes: tuple[str, ...] = BEHAVIORS,
                 hidden: int = 96, dilations=(1, 2, 4, 8, 16), dropout: float = 0.1):
        super().__init__()
        self.classes = tuple(classes)
        self.features = EgocentricFeatures(num_keypoints, dims)
        self.inp = nn.Conv1d(self.features.out_dim, hidden, 1)
        self.blocks = nn.Sequential(*[TCNBlock(hidden, d, dropout) for d in dilations])
        self.out = nn.Conv1d(hidden, len(classes), 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``x (B, T, K, D)``, ``mask (B, T, K)`` -> logits ``(B, T, C)``."""
        f = self.features(x, mask).transpose(1, 2)
        return self.out(self.blocks(self.inp(f))).transpose(1, 2)


class BehaviorExport(nn.Module):
    def __init__(self, model: BehaviorTCN):
        super().__init__()
        self.model = model

    def forward(self, x, mask):
        return torch.softmax(self.model(x, mask), dim=-1)


def to_bouts(labels, min_len: int = 5) -> list[tuple[int, int, int]]:
    """Per-frame labels -> ``[(label, start, end_exclusive)]``, absorbing bouts
    shorter than ``min_len`` into the preceding bout."""
    bouts: list[list[int]] = []
    for i, lab in enumerate(list(labels)):
        if bouts and bouts[-1][0] == lab:
            bouts[-1][2] = i + 1
        else:
            bouts.append([int(lab), i, i + 1])
    merged: list[list[int]] = []
    for b in bouts:
        if merged and (b[2] - b[1] < min_len or merged[-1][0] == b[0]):
            merged[-1][2] = b[2]
        else:
            merged.append(b)
    return [tuple(b) for b in merged]
