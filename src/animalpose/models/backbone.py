"""Backbones producing stride-4 feature maps through a light FPN.

Only standard convolution, batch-norm, ReLU, resize and add ops are used so the
graph exports cleanly to TorchScript and ONNX.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1):
        super().__init__(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, 3, stride)
        self.conv2 = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = (
            nn.Identity() if stride == 1 and cin == cout
            else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        )

    def forward(self, x):
        return F.relu(self.conv2(self.conv1(x)) + self.skip(x))


class Normalize(nn.Module):
    """Maps RGB float images in [0, 1] to ImageNet-normalized input."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class FPN(nn.Module):
    """Top-down merge of strides (4, 8, 16, 32) into one stride-4 map."""

    def __init__(self, in_channels: list[int], out_channels: int):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in in_channels)
        self.smooth = ConvBNAct(out_channels, out_channels, 3)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = self.lateral[-1](feats[-1])
        for lat, f in zip(reversed(self.lateral[:-1]), reversed(feats[:-1])):
            x = lat(f) + F.interpolate(x, size=f.shape[-2:], mode="nearest")
        return self.smooth(x)


class TinyResNet(nn.Module):
    """Small residual encoder (~0.4-6M params depending on width). Trains from scratch."""

    def __init__(self, width: int = 32, blocks: tuple[int, ...] = (1, 2, 2, 1)):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(3, width, 3, 2), ConvBNAct(width, width, 3, 1))
        chans = [width, width * 2, width * 4, width * 8]
        stages, cin = [], width
        for c, n in zip(chans, blocks):
            layers = [ResBlock(cin, c, 2)] + [ResBlock(c, c) for _ in range(n - 1)]
            stages.append(nn.Sequential(*layers))
            cin = c
        self.stages = nn.ModuleList(stages)
        self.channels = chans

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for s in self.stages:
            x = s(x)
            feats.append(x)
        return feats


class TorchvisionResNet(nn.Module):
    """torchvision ResNet encoder. ``pretrained=True`` downloads ImageNet weights."""

    def __init__(self, name: str = "resnet18", pretrained: bool = False):
        super().__init__()
        import torchvision

        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        exp = 4 if name in ("resnet50", "resnet101", "resnet152") else 1
        self.channels = [64 * exp, 128 * exp, 256 * exp, 512 * exp]

    def forward(self, x):
        x = self.stem(x)  # stride 4
        feats = []
        for s in self.stages:
            x = s(x)
            feats.append(x)
        return feats  # strides 4, 8, 16, 32


class Backbone(nn.Module):
    """Encoder + FPN. ``forward`` returns (stride-4 map, deepest map)."""

    def __init__(self, name: str = "tiny", width: int = 32, out_channels: int = 64, pretrained: bool = False):
        super().__init__()
        self.norm = Normalize()
        # Both encoders return stage outputs at strides 4, 8, 16, 32.
        self.encoder = TinyResNet(width) if name == "tiny" else TorchvisionResNet(name, pretrained)
        in_ch = self.encoder.channels
        self.fpn = FPN(in_ch, out_channels)
        self.out_channels = out_channels
        self.deep_channels = in_ch[-1]
        self.stride = 4

    def forward(self, x):
        feats = self.encoder(self.norm(x))
        return self.fpn(feats), feats[-1]


def build_backbone(cfg: dict | None = None) -> Backbone:
    cfg = dict(cfg or {})
    return Backbone(cfg.get("name", "tiny"), cfg.get("width", 32), cfg.get("out_channels", 64),
                    cfg.get("pretrained", False))
