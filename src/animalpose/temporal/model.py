"""Temporal LSTM over keypoint tracks: smoothing, outlier correction, gap filling.

Input per track window: coordinates ``(B, T, K, D)`` (D = 2 for image tracks,
3 for triangulated tracks), detector confidence ``(B, T, K)`` and an
``observed`` mask ``(B, T, K)`` (0 where the point is missing: occluded, out
of frame, or below the confidence threshold).

Output: refined coordinates for *every* frame and keypoint, a per-point
confidence in (0, 1] derived from the predicted uncertainty, and an
``imputed`` flag that is 1 where the value was not observed and had to be
predicted. Normalisation (masked centring and scaling) happens inside the
graph so exported models take raw pixel/world coordinates.

``bidirectional=True`` is for offline analysis (interpolates through gaps);
``bidirectional=False`` is causal and suitable for online use (extrapolates).

With ``prior_bandwidth > 0`` (offline models) the network predicts a residual
on top of a kernel-interpolation prior: a Laplacian-weighted average of the
observed frames, ``exp(-|t - s| / h)``, computed with two matrix products.
The prior already denoises observed points and bridges gaps roughly like
linear interpolation, so the LSTM only has to learn the correction
(posture dynamics, outliers, swaps). ``log`` of the prior's support is an
extra input, telling the network how far each point is from an observation.
For causal models the kernel is one-sided, so outputs never depend on future frames.
"""

from __future__ import annotations

import torch
from torch import nn

RESIDUAL_GUARD = 4.0  # in units of the window's keypoint spread


class TemporalLSTM(nn.Module):
    def __init__(self, num_keypoints: int = 25, dims: int = 2, hidden: int = 128, layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.1, prior_bandwidth: float = 0.0,
                 prior: str = "kernel"):
        super().__init__()
        self.k, self.d = num_keypoints, dims
        self.bidirectional = bidirectional
        self.prior_bandwidth = prior_bandwidth
        self.prior = prior  # "kernel" (Laplacian average), "linear" (exact interpolation) or "hybrid"
        n_in = num_keypoints * (dims + 2) + (num_keypoints * (dims + 1) if prior_bandwidth > 0 else 0)
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if layers > 1 else 0.0)
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Linear(out, hidden), nn.ReLU(), nn.Linear(hidden, num_keypoints * (dims + 1)))

    @staticmethod
    def normalizer(x: torch.Tensor, observed: torch.Tensor):
        """Masked centre ``(B, 1, 1, D)`` and isotropic scale ``(B, 1, 1, 1)``."""
        m = observed.unsqueeze(-1)
        n = m.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        center = (x * m).sum(dim=(1, 2), keepdim=True) / n
        var = (((x - center) ** 2) * m).sum(dim=(1, 2, 3), keepdim=True) / (n * x.shape[-1])
        return center, torch.sqrt(var).clamp(min=1e-3)

    def linear_prior(self, xn: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """Per-keypoint linear interpolation between the nearest observed frames
        (hold the nearest value at the ends; zero if never observed). Causal
        models hold the last observation instead. Built from masked max/min and
        gather over a (T, T) lag grid, so it exports to ONNX."""
        b, t, k, d = xn.shape
        steps = torch.arange(t, device=xn.device)
        obs = observed > 0.5  # (B, T, K)
        grid = steps.view(1, 1, t, 1).expand(b, t, t, k)  # [.., query t, source s, k] = s
        before = (steps.view(1, 1, t, 1) <= steps.view(1, t, 1, 1)) & obs.unsqueeze(1)
        after = (steps.view(1, 1, t, 1) >= steps.view(1, t, 1, 1)) & obs.unsqueeze(1)
        prev = torch.where(before, grid, torch.full_like(grid, -1)).max(dim=2).values  # (B, T, K)
        nxt = torch.where(after, grid, torch.full_like(grid, t)).min(dim=2).values
        has_prev, has_next = prev >= 0, nxt < t
        if not self.bidirectional:
            has_next = torch.zeros_like(has_next)
        p_idx = torch.where(has_prev, prev, torch.where(has_next, nxt, torch.zeros_like(prev)))
        n_idx = torch.where(has_next, nxt, p_idx)
        p_val = torch.gather(xn, 1, p_idx.unsqueeze(-1).expand(-1, -1, -1, d))
        n_val = torch.gather(xn, 1, n_idx.unsqueeze(-1).expand(-1, -1, -1, d))
        span = (n_idx - p_idx).clamp(min=1).to(xn.dtype)
        frac = ((steps.view(1, t, 1) - p_idx).to(xn.dtype) / span).clamp(0.0, 1.0)
        frac = torch.where(has_prev & has_next, frac, torch.zeros_like(frac))
        out = p_val + frac.unsqueeze(-1) * (n_val - p_val)
        return out * (has_prev | has_next).unsqueeze(-1).to(xn.dtype)

    def kernel_prior(self, xn: torch.Tensor, observed: torch.Tensor):
        """Laplacian-kernel average over observed frames -> ``(prior (B,T,K,D), support (B,T,K))``."""
        b, t, k, d = xn.shape
        steps = torch.arange(t, dtype=xn.dtype, device=xn.device)
        lag = steps[:, None] - steps[None, :]
        w = torch.exp(-torch.abs(lag) / self.prior_bandwidth)  # (T, T)
        if not self.bidirectional:
            w = w * (lag >= 0).to(w.dtype)  # causal models may only look back
        num = torch.matmul(w, (xn * observed.unsqueeze(-1)).reshape(b, t, k * d)).view(b, t, k, d)
        support = torch.matmul(w, observed)  # (B, T, K)
        return num / support.clamp(min=1e-6).unsqueeze(-1), support

    def forward(self, x: torch.Tensor, conf: torch.Tensor, observed: torch.Tensor):
        b, t, k, d = x.shape
        observed = observed.to(x.dtype)
        x = torch.nan_to_num(x) * observed.unsqueeze(-1)
        center, scale = self.normalizer(x, observed)
        xn = (x - center) / scale * observed.unsqueeze(-1)
        feats = [xn.reshape(b, t, k * d), observed, conf * observed]
        if self.prior_bandwidth > 0:
            base, support = self.kernel_prior(xn, observed)
            if self.prior == "linear":
                base = self.linear_prior(xn, observed)
            elif self.prior == "hybrid":  # kernel average where observed, linear interpolation in gaps
                obs = observed.unsqueeze(-1)
                base = obs * base + (1 - obs) * self.linear_prior(xn, observed)
            feats += [base.reshape(b, t, k * d), torch.log(support + 1e-6)]
        else:
            base = xn  # residual on observed points, absolute where xn == 0
        h, _ = self.lstm(self.inp(torch.cat(feats, dim=-1)))
        out = self.head(h).view(b, t, k, d + 1)
        delta = out[..., :d]
        log_sigma = out[..., d].clamp(-7.0, 3.0)
        if not self.training:
            # Guard: a correction larger than RESIDUAL_GUARD window-scales means the input is far
            # outside the training distribution (e.g. a sparse real skeleton fed to a synthetic-only
            # model). Fall back to the prior there and report maximal uncertainty.
            bad = (delta.abs() > RESIDUAL_GUARD).any(-1)
            delta = torch.where(bad.unsqueeze(-1), torch.zeros_like(delta), delta)
            log_sigma = torch.where(bad, torch.full_like(log_sigma, 3.0), log_sigma)
        pred_n = base + delta
        return {"coords": pred_n * scale + center, "coords_norm": pred_n, "log_sigma": log_sigma,
                "center": center, "scale": scale}


class TemporalExport(nn.Module):
    """Export wrapper: (coords, conf, observed) -> (coords, confidence, imputed)."""

    def __init__(self, model: TemporalLSTM, sigma_ref: float = 0.05):
        super().__init__()
        self.model = model
        self.sigma_ref = sigma_ref

    def forward(self, x, conf, observed):
        out = self.model(x, conf, observed)
        confidence = 1.0 / (1.0 + torch.exp(out["log_sigma"]) / self.sigma_ref)
        imputed = 1.0 - observed.to(x.dtype)
        return out["coords"], confidence, imputed


def temporal_loss(out: dict, target: torch.Tensor, target_mask: torch.Tensor, mse_weight: float = 1.0) -> dict:
    """Gaussian NLL (isotropic, per keypoint) in normalised units + MSE anchor."""
    tn = (torch.nan_to_num(target) - out["center"]) / out["scale"]
    m = target_mask.to(tn.dtype)
    err2 = ((out["coords_norm"] - tn) ** 2).sum(-1)
    ls = out["log_sigma"]
    d = tn.shape[-1]
    nll = 0.5 * err2 * torch.exp(-2 * ls) + d * ls
    n = m.sum().clamp(min=1.0)
    losses = {"nll": (nll * m).sum() / n, "mse": mse_weight * (err2 * m).sum() / n}
    losses["total"] = losses["nll"] + losses["mse"]
    return losses


@torch.no_grad()
def smooth_sequence(model: TemporalLSTM, coords, conf, observed, window: int = 64, hop: int = 32,
                    sigma_ref: float = 0.05):
    """Run on a full track ``(T, K, D)`` with overlapping windows blended by a
    triangular weight. Returns numpy ``(coords, confidence, imputed)``."""
    import numpy as np

    dev = next(model.parameters()).device
    coords = np.asarray(coords, dtype=np.float32)
    t = coords.shape[0]
    x = torch.from_numpy(np.nan_to_num(coords)).to(dev)
    c = torch.from_numpy(np.nan_to_num(np.asarray(conf, dtype=np.float32))).to(dev)
    o = torch.from_numpy(np.asarray(observed, dtype=np.float32)).to(dev)
    if t <= window:
        starts = [0]
        window = t
    else:
        starts = list(range(0, t - window + 1, hop))
        if starts[-1] != t - window:
            starts.append(t - window)
    acc = torch.zeros_like(x)
    acc_s = torch.zeros(x.shape[:2], device=dev)
    wsum = torch.zeros(t, device=dev)
    ramp = torch.minimum(torch.arange(1, window + 1, device=dev), torch.arange(window, 0, -1, device=dev)).float()
    batch = [(s, x[s:s + window], c[s:s + window], o[s:s + window]) for s in starts]
    xs = torch.stack([b[1] for b in batch])
    cs = torch.stack([b[2] for b in batch])
    os_ = torch.stack([b[3] for b in batch])
    model.eval()
    out = model(xs, cs, os_)
    for i, s in enumerate(starts):
        acc[s:s + window] += out["coords"][i] * ramp.view(-1, 1, 1)
        acc_s[s:s + window] += torch.exp(out["log_sigma"][i]) * ramp.view(-1, 1)
        wsum[s:s + window] += ramp
    coords_out = (acc / wsum.view(-1, 1, 1)).cpu().numpy()
    sigma = (acc_s / wsum.view(-1, 1)).cpu().numpy()
    confidence = 1.0 / (1.0 + sigma / sigma_ref)
    imputed = 1.0 - o.cpu().numpy()
    return coords_out, confidence, imputed
