"""Stage 1: anchor-free, class-agnostic animal detector (CenterNet-style).

The detector predicts a single "animal" centre heatmap so it is not tied to a
species list; an optional species head classifies each detection for routing
(e.g. per-species body-length priors) without affecting localisation.

``forward`` returns raw maps for training. :class:`DetectorExport` wraps the
model with in-graph peak extraction (max-pool NMS + top-k) so exported graphs
emit fixed-size ``(B, max_det, 4)`` boxes and ``(B, max_det)`` scores; score
thresholding happens in the caller.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from animalpose.models.backbone import ConvBNAct, build_backbone


def _head(cin: int, cout: int, bias: float = 0.0) -> nn.Sequential:
    head = nn.Sequential(ConvBNAct(cin, cin, 3), nn.Conv2d(cin, cout, 1))
    nn.init.constant_(head[-1].bias, bias)
    return head


class CenterDetector(nn.Module):
    def __init__(self, backbone: dict | None = None, num_species: int = 0):
        super().__init__()
        self.backbone = build_backbone(backbone)
        c = self.backbone.out_channels
        self.stride = self.backbone.stride
        self.heat = _head(c, 1, bias=-math.log((1 - 0.01) / 0.01))  # prior p=0.01
        self.size = _head(c, 2)  # log(w, h) in stride units
        self.offset = _head(c, 2)  # sub-cell centre offset
        self.num_species = num_species
        self.species = _head(c, num_species) if num_species > 0 else None

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """``images``: RGB float in [0, 1], ``(B, 3, H, W)`` with H, W divisible by 32."""
        feat, _ = self.backbone(images)
        out = {"heat": self.heat(feat), "size": self.size(feat), "offset": self.offset(feat)}
        if self.species is not None:
            out["species"] = self.species(feat)
        return out


def decode(heat_logits: torch.Tensor, size: torch.Tensor, offset: torch.Tensor, stride: int, max_det: int = 20,
           *, for_export: bool = False):
    """Peak extraction with export-friendly ops. Returns boxes, scores, flat indices."""
    b, _, h, w = heat_logits.shape
    heat = torch.sigmoid(heat_logits)
    peaks = F.max_pool2d(heat, 3, stride=1, padding=1)
    heat = heat * (peaks == heat).to(heat.dtype)
    flat = heat.reshape(b, -1)
    if for_export:
        # ONNX and Torch can break ties differently. Rank exact ties by flat index in
        # float64, with a perturbation below the spacing of distinct float32 scores.
        # MPS has no float64, so rank on the CPU there; export runs on the CPU anyway.
        device = torch.device("cpu") if heat.device.type == "mps" else heat.device
        positions = torch.arange(h * w, device=device, dtype=torch.float64)
        ranking = flat.to(device=device, dtype=torch.float64) * (1 - (positions / (h * w)) * 2**-26)
        idx = torch.topk(ranking, max_det, dim=1).indices.to(heat.device)
        scores = flat.gather(1, idx)
    else:
        scores, idx = torch.topk(flat, max_det, dim=1)
    ys = torch.div(idx, w, rounding_mode="floor").to(heat.dtype)
    xs = (idx - ys.to(idx.dtype) * w).to(heat.dtype)
    gather = idx.unsqueeze(1).expand(-1, 2, -1)
    off = torch.gather(offset.reshape(b, 2, -1), 2, gather)
    wh = torch.exp(torch.gather(size.reshape(b, 2, -1), 2, gather).clamp(max=10.0))
    cx = (xs + off[:, 0]) * stride
    cy = (ys + off[:, 1]) * stride
    bw, bh = wh[:, 0] * stride, wh[:, 1] * stride
    boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)
    if for_export:
        boxes = torch.where(scores.unsqueeze(-1) > 0, boxes, 0.0)
    return boxes, scores, idx


class DetectorExport(nn.Module):
    """Inference/export wrapper: images -> (boxes (B,N,4), scores (B,N))."""

    def __init__(self, model: CenterDetector, max_det: int = 20):
        super().__init__()
        self.model = model
        self.max_det = max_det

    def forward(self, images: torch.Tensor):
        out = self.model(images)
        boxes, scores, _ = decode(out["heat"], out["size"], out["offset"], self.model.stride, self.max_det, for_export=True)
        return boxes, scores


# ----------------------------------------------------------------------------- training

def draw_gaussian(heat: torch.Tensor, cx: float, cy: float, sx: float, sy: float) -> None:
    h, w = heat.shape
    x0, x1 = max(0, int(cx - 3 * sx)), min(w, int(cx + 3 * sx) + 2)
    y0, y1 = max(0, int(cy - 3 * sy)), min(h, int(cy + 3 * sy) + 2)
    if x0 >= x1 or y0 >= y1:
        return
    ys = torch.arange(y0, y1, dtype=torch.float32)[:, None]
    xs = torch.arange(x0, x1, dtype=torch.float32)[None, :]
    g = torch.exp(-((xs - cx) ** 2) / (2 * sx * sx) - ((ys - cy) ** 2) / (2 * sy * sy))
    heat[y0:y1, x0:x1] = torch.maximum(heat[y0:y1, x0:x1], g)


def build_targets(boxes: list[torch.Tensor], out_hw: tuple[int, int], stride: int,
                  species: list[torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """Dense targets from per-image xyxy boxes (input pixel units)."""
    b = len(boxes)
    h, w = out_hw
    heat = torch.zeros(b, 1, h, w)
    size = torch.zeros(b, 2, h, w)
    offset = torch.zeros(b, 2, h, w)
    reg_mask = torch.zeros(b, 1, h, w)
    spc = torch.full((b, h, w), -100, dtype=torch.long)
    for i, bx in enumerate(boxes):
        for j, (x0, y0, x1, y1) in enumerate(bx.tolist()):
            bw, bh = (x1 - x0) / stride, (y1 - y0) / stride
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = (x0 + x1) / 2 / stride, (y0 + y1) / 2 / stride
            ix, iy = min(max(int(cx), 0), w - 1), min(max(int(cy), 0), h - 1)
            # TTFNet-style elliptical Gaussian, centred on the integer peak cell.
            draw_gaussian(heat[i, 0], ix, iy, max(0.8, 0.54 * bw / 6), max(0.8, 0.54 * bh / 6))
            heat[i, 0, iy, ix] = 1.0
            size[i, :, iy, ix] = torch.tensor([math.log(bw), math.log(bh)])
            offset[i, :, iy, ix] = torch.tensor([cx - ix, cy - iy])
            reg_mask[i, 0, iy, ix] = 1.0
            if species is not None:
                spc[i, iy, ix] = int(species[i][j])
    return {"heat": heat, "size": size, "offset": offset, "reg_mask": reg_mask, "species": spc}


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = -torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = -torch.log(1 - p) * p**alpha * (1 - target) ** beta * neg
    n = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n


def detector_loss(out: dict, tgt: dict, size_weight: float = 0.1, offset_weight: float = 1.0) -> dict:
    tgt = {k: v.to(out["heat"].device) for k, v in tgt.items()}
    m = tgt["reg_mask"]
    n = m.sum().clamp(min=1.0)
    losses = {
        "heat": focal_loss(out["heat"], tgt["heat"]),
        "size": size_weight * (F.l1_loss(out["size"], tgt["size"], reduction="none") * m).sum() / n,
        "offset": offset_weight * (F.l1_loss(out["offset"], tgt["offset"], reduction="none") * m).sum() / n,
    }
    if "species" in out:
        losses["species"] = F.cross_entropy(out["species"], tgt["species"], ignore_index=-100)
    losses["total"] = sum(losses.values())
    return losses
