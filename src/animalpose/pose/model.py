"""Stage 2: top-down keypoint network over the canonical vocabulary.

Heatmaps are produced by *keypoint queries*: each canonical keypoint owns a
learned embedding, and its heatmap is the dot product of that embedding with
the shared feature map (implemented as a 1x1 convolution so it exports
anywhere). Consequences:

* one network serves every species; datasets that label different subsets
  are pooled with per-keypoint loss masks;
* extending the vocabulary later only adds query rows, which can be fitted
  while the backbone stays frozen (see :meth:`PoseNet.add_keypoints`).

Locations are decoded as the heatmap argmax plus a learned sub-pixel offset
(DeepLabCut-style location refinement); confidence is the sigmoid peak value.
A pooled appearance embedding is emitted for tracking re-identification.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from animalpose.models.backbone import ConvBNAct, build_backbone
from animalpose.multiview.symmetry import bilateral_pairs


class PoseNet(nn.Module):
    def __init__(self, num_keypoints: int = 25, backbone: dict | None = None, embed_dim: int = 64,
                 context: bool = False):
        super().__init__()
        self.backbone = build_backbone(backbone)
        c = self.backbone.out_channels
        self.stride = self.backbone.stride
        self.heat_feat = nn.Sequential(ConvBNAct(c, c, 3), ConvBNAct(c, c, 3), nn.Conv2d(c, c, 1))
        self.off_feat = nn.Sequential(ConvBNAct(c, c, 3), nn.Conv2d(c, c, 1))
        self.heat_queries = nn.Parameter(torch.randn(num_keypoints, c) / math.sqrt(c))
        self.heat_bias = nn.Parameter(torch.full((num_keypoints,), -4.0))
        self.off_queries = nn.Parameter(torch.zeros(num_keypoints * 2, c))
        self.embed = nn.Sequential(nn.Linear(self.backbone.deep_channels, 128), nn.ReLU(inplace=True),
                                   nn.Linear(128, embed_dim))
        # Optional whole-crop context (FiLM): heading and dorsal/ventral orientation decide
        # which side of the body faces the camera, which local features alone cannot see.
        # Zero-initialised, so it starts as the identity and older checkpoints load unchanged.
        self.context = None
        if context:
            self.context = nn.Sequential(nn.Linear(self.backbone.deep_channels, 128), nn.ReLU(inplace=True),
                                         nn.Linear(128, 2 * c))
            nn.init.zeros_(self.context[-1].weight), nn.init.zeros_(self.context[-1].bias)

    @property
    def num_keypoints(self) -> int:
        return self.heat_queries.shape[0]

    def forward(self, crops: torch.Tensor) -> dict[str, torch.Tensor]:
        """``crops``: RGB float in [0, 1], ``(B, 3, S, S)``, S divisible by 32."""
        feat, deep = self.backbone(crops)
        c = feat.shape[1]
        pooled = deep.mean(dim=(2, 3))
        hf = self.heat_feat(feat)
        if self.context is not None:
            scale, shift = self.context(pooled).view(-1, 2 * c, 1, 1).chunk(2, dim=1)
            hf = hf * (1 + scale) + shift
        heat = F.conv2d(hf, self.heat_queries.view(-1, c, 1, 1), self.heat_bias)
        off = F.conv2d(self.off_feat(feat), self.off_queries.view(-1, c, 1, 1))
        emb = self.embed(pooled)
        return {"heat": heat, "offset": off, "embedding": F.normalize(emb, dim=1)}

    @torch.no_grad()
    def add_keypoints(self, n_new: int, init_from: list[int] | None = None) -> None:
        """Append ``n_new`` keypoint queries (optionally initialised from existing ones)."""
        c = self.heat_queries.shape[1]
        new_q = torch.randn(n_new, c, device=self.heat_queries.device) / math.sqrt(c)
        if init_from is not None:
            new_q = self.heat_queries[init_from].clone()
        self.heat_queries = nn.Parameter(torch.cat([self.heat_queries, new_q]))
        self.heat_bias = nn.Parameter(torch.cat([self.heat_bias, torch.full((n_new,), -4.0, device=new_q.device)]))
        self.off_queries = nn.Parameter(torch.cat([self.off_queries, torch.zeros(2 * n_new, c, device=new_q.device)]))


def decode(heat_logits: torch.Tensor, offset: torch.Tensor, stride: int):
    """Argmax + offset decoding with export-friendly ops.

    Returns keypoints ``(B, K, 2)`` in crop pixels and confidence ``(B, K)``.
    """
    b, k, h, w = heat_logits.shape
    idx = heat_logits.reshape(b, k, h * w).max(dim=2)[1]
    return _decode_at(heat_logits, offset, stride, idx)


def _decode_at(heat_logits: torch.Tensor, offset: torch.Tensor, stride: int, idx: torch.Tensor):
    """Keypoints and confidence at flat heatmap cells ``idx`` ``(B, K)``."""
    b, k, h, w = heat_logits.shape
    peak = torch.gather(heat_logits.reshape(b, k, h * w), 2, idx.unsqueeze(2)).squeeze(2)
    ys = torch.div(idx, w, rounding_mode="floor")
    xs = idx - ys * w
    off = offset.reshape(b, k, 2, h * w)
    off = torch.gather(off, 3, idx.view(b, k, 1, 1).expand(-1, -1, 2, -1)).squeeze(3)
    x = (xs.to(heat_logits.dtype) + off[..., 0]) * stride
    y = (ys.to(heat_logits.dtype) + off[..., 1]) * stride
    return torch.stack([x, y], dim=-1), torch.sigmoid(peak)


def decode_pairs(heat_logits: torch.Tensor, offset: torch.Tensor, stride: int, pairs: torch.Tensor,
                 radius: float = 2.5, min_peak: float = 0.1, return_margin: bool = False):
    """Argmax decoding that keeps bilateral pairs on distinct peaks.

    When near and far limbs look alike (side views), the left and right
    heatmaps often both peak on the more salient paw, so independent argmax
    puts both keypoints on one paw. For each ``(left, right)`` row of
    ``pairs`` whose argmaxes fall within ``radius`` cells of each other, the
    two strongest separated peaks of ``max(p_left, p_right)`` are found
    (non-maximum suppression of ``radius``); if the second one reaches
    ``min_peak`` probability the pair is split across the two peaks, in the
    left/right order the two heatmaps themselves prefer. Pairs whose second
    peak is weak (genuinely occluded or overlapping limbs) are left as they
    are. Export-friendly ops only.

    With ``return_margin`` also returns ``(B, P)`` left/right evidence for each
    decoded pair, ``p_l(x_l) + p_r(x_r) - p_l(x_r) - p_r(x_l)``: positive when the
    heatmaps prefer the decoded labelling over its swap, near zero when they
    cannot tell (used by :func:`animalpose.temporal.pairs.resolve_pair_identity`).
    """
    b, k, h, w = heat_logits.shape
    flat = heat_logits.reshape(b, k, h * w)
    idx = flat.max(dim=2)[1]
    if pairs.numel() == 0:
        kp, conf = _decode_at(heat_logits, offset, stride, idx)
        return (kp, conf, heat_logits.new_zeros(b, 0)) if return_margin else (kp, conf)
    prob = torch.sigmoid(flat)
    cell = torch.arange(h * w, device=flat.device)
    cy, cx = torch.div(cell, w, rounding_mode="floor"), cell % w
    lft, rgt = pairs[:, 0], pairs[:, 1]
    pl, pr = prob[:, lft], prob[:, rgt]  # (B, P, HW)
    il, ir = idx[:, lft], idx[:, rgt]
    comb = torch.maximum(pl, pr)
    i1 = comb.max(dim=2)[1]
    far = (cx.view(1, 1, -1) - cx[i1].unsqueeze(2)).pow(2) + (cy.view(1, 1, -1) - cy[i1].unsqueeze(2)).pow(2)
    v2, i2 = torch.where(far > radius * radius, comb, torch.zeros_like(comb)).max(dim=2)
    near = (cx[il] - cx[ir]).pow(2) + (cy[il] - cy[ir]).pow(2) <= radius * radius
    split = near & (v2 >= min_peak)

    def at(p, i):
        return torch.gather(p, 2, i.unsqueeze(2)).squeeze(2)

    keep = at(pl, i1) + at(pr, i2) >= at(pl, i2) + at(pr, i1)
    il = torch.where(split, torch.where(keep, i1, i2), il)
    ir = torch.where(split, torch.where(keep, i2, i1), ir)
    idx = idx.scatter(1, lft.unsqueeze(0).expand(b, -1), il).scatter(1, rgt.unsqueeze(0).expand(b, -1), ir)
    kp, conf = _decode_at(heat_logits, offset, stride, idx)
    if not return_margin:
        return kp, conf
    return kp, conf, at(pl, il) + at(pr, ir) - at(pl, ir) - at(pr, il)


class PoseExport(nn.Module):
    """Inference/export wrapper: crops -> (keypoints, confidence, embedding).

    ``pair_decoding`` switches to :func:`decode_pairs` over every bilateral
    pair of the vocabulary.
    """

    def __init__(self, model: PoseNet, pair_decoding: bool = False, pair_radius: float = 2.5,
                 pair_min_peak: float = 0.1):
        super().__init__()
        self.model = model
        self.pair_decoding = pair_decoding
        self.pair_radius, self.pair_min_peak = pair_radius, pair_min_peak
        self.register_buffer("pairs", torch.tensor(bilateral_pairs(), dtype=torch.long).reshape(-1, 2), persistent=False)

    def forward(self, crops: torch.Tensor):
        out = self.model(crops)
        if self.pair_decoding:
            kpts, conf = decode_pairs(out["heat"], out["offset"], self.model.stride, self.pairs,
                                      self.pair_radius, self.pair_min_peak)
        else:
            kpts, conf = decode(out["heat"], out["offset"], self.model.stride)
        return kpts, conf, out["embedding"]

    def with_margins(self, crops: torch.Tensor):
        """Eager-only: pair-decoded keypoints plus the ``(B, P)`` pair margins of :func:`decode_pairs`."""
        out = self.model(crops)
        kpts, conf, margin = decode_pairs(out["heat"], out["offset"], self.model.stride, self.pairs,
                                          self.pair_radius, self.pair_min_peak, return_margin=True)
        return kpts, conf, out["embedding"], margin


# ----------------------------------------------------------------------------- training

def build_targets(kpts: torch.Tensor, state: torch.Tensor, out_hw: tuple[int, int], stride: int,
                  sigma: float = 1.5, occluded_weight: float = 0.5, offset_radius: float = 2.0):
    """Heatmap/offset targets.

    ``kpts``: ``(B, K, 2)`` crop pixels. ``state``: ``(B, K)`` with
    2 = visible, 1 = labeled but occluded (trained as "not visible": zero
    heatmap, so confidence drops and the temporal model fills the point),
    0 = not labeled for this sample/dataset (no loss).
    """
    b, k, _ = kpts.shape
    h, w = out_hw
    g = kpts / stride
    ys = torch.arange(h, dtype=torch.float32).view(1, 1, h, 1)
    xs = torch.arange(w, dtype=torch.float32).view(1, 1, 1, w)
    # Peak on the rounded cell so argmax decoding + offset recovers the location.
    cx = torch.round(g[..., 0]).view(b, k, 1, 1)
    cy = torch.round(g[..., 1]).view(b, k, 1, 1)
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    vis = (state == 2).float().view(b, k, 1, 1)
    heat = torch.exp(-d2 / (2 * sigma**2)) * vis
    weight = torch.where(state == 2, 1.0, torch.where(state == 1, occluded_weight, 0.0)).view(b, k, 1, 1)
    near = (d2 <= offset_radius**2).float() * vis
    off_x = (g[..., 0].view(b, k, 1, 1) - xs) * near
    off_y = (g[..., 1].view(b, k, 1, 1) - ys) * near
    offset = torch.stack([off_x, off_y], dim=2).view(b, 2 * k, h, w)
    return {"heat": heat, "weight": weight, "offset": offset, "offset_mask": near.repeat_interleave(2, dim=1)}


def supcon_loss(emb: torch.Tensor, ids: torch.Tensor, temperature: float = 0.1,
                groups: torch.Tensor | None = None) -> torch.Tensor:
    """Supervised contrastive loss over identity labels (ignores id < 0).

    With ``groups`` (e.g. video/scene id), each anchor is contrasted only
    against samples of its own group: re-identification has to separate
    animals that share a video, not every animal ever seen.
    """
    keep = ids >= 0
    emb, ids = emb[keep], ids[keep]
    if emb.shape[0] < 2:
        return emb.sum() * 0.0
    sim = emb @ emb.t() / temperature
    eye = torch.eye(len(ids), device=emb.device, dtype=torch.bool)
    pos = (ids[:, None] == ids[None, :]) & ~eye
    excluded = eye
    if groups is not None:
        groups = groups[keep]
        excluded = excluded | (groups[:, None] != groups[None, :])
        pos = pos & ~excluded
    # Anchors need a positive and at least one negative to contribute.
    has_pos = pos.any(1) & (~excluded & ~pos).any(1)
    if not has_pos.any():
        return emb.sum() * 0.0
    sim = sim.masked_fill(excluded, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -(log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)
    return loss[has_pos].mean()


def pose_loss(out: dict, tgt: dict, ids: torch.Tensor | None = None, offset_weight: float = 0.05,
              embed_weight: float = 0.1, pos_weight: float = 10.0, groups: torch.Tensor | None = None) -> dict:
    dev = out["heat"].device
    tgt = {k: v.to(dev) for k, v in tgt.items()}
    bce = F.binary_cross_entropy_with_logits(out["heat"], tgt["heat"], reduction="none")
    bce = bce * (1 + pos_weight * tgt["heat"]) * tgt["weight"]
    denom = (tgt["weight"].sum() * out["heat"].shape[-1] * out["heat"].shape[-2] / 64).clamp(min=1.0)
    losses = {"heat": bce.sum() / denom}
    m = tgt["offset_mask"]
    off = F.smooth_l1_loss(out["offset"], tgt["offset"], reduction="none") * m
    losses["offset"] = offset_weight * off.sum() / m.sum().clamp(min=1.0)
    if ids is not None and embed_weight > 0:
        losses["embed"] = embed_weight * supcon_loss(out["embedding"], ids.to(dev),
                                                     groups=groups.to(dev) if groups is not None else None)
    losses["total"] = sum(losses.values())
    return losses
