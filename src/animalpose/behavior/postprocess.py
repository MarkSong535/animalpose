"""Post-processing of classifier probabilities: bout smoothing and prior adaptation.

The behavior TCN is frame-wise and trained on one lab's sessions; a new recording usually
has a different behavior *distribution* (a corner camera sees rearing far more often than a
top camera does) and noisier keypoints, which shows up as a flickering argmax and a class
prior that no longer matches. Neither needs new labels:

* :func:`median_smooth` filters the probability sequence with a per-class moving median,
  which removes single-frame flicker while keeping bout edges sharp (a moving *mean* blurs
  the transition of a 1 s bout into the neighbouring bout);
* :func:`adapt_prior` re-weights classes by the Saerens-Latinne-Decaestecker expectation
  step: the class priors ``pi`` are iterated until the re-weighted posteriors have the same
  marginals as ``pi`` itself, i.e. until the model, seen on *this* recording, is internally
  consistent. ``--prior-max-ratio`` caps how far a prior may move, which stops the EM from
  switching a class off entirely when a recording simply lacks it.
* :func:`fit_temperature` / :func:`temperature_scale` sharpen or flatten the probabilities
  on labelled data; they do not change an argmax over the same class set, but the values are
  what :func:`animalpose.behavior.catalog.full_probabilities` reports.
"""

from __future__ import annotations

import numpy as np

__all__ = ["adapt_prior", "fit_temperature", "median_smooth", "moving_median", "temperature_scale"]


def moving_median(p: np.ndarray, win: int) -> np.ndarray:
    """Per-class median over a window of ``win`` frames, edges replicated. ``(T, C)`` -> ``(T, C)``."""
    p = np.asarray(p, np.float32)
    if win <= 1 or len(p) < 3:
        return p.copy()
    win = int(win) + 1 - int(win) % 2  # odd, so the centre frame is the current one
    pad = np.pad(p, [(win // 2, win - 1 - win // 2), (0, 0)], mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, win, axis=0), axis=2).astype(np.float32)


def median_smooth(probs: np.ndarray, win: int = 5, renormalize: bool = True) -> np.ndarray:
    """Bout-level smoothing of ``(T, C)`` (or ``(B, T, C)``) probabilities; see :func:`moving_median`."""
    probs = np.asarray(probs, np.float32)
    if probs.ndim == 2:
        out = moving_median(probs, win)
    else:
        out = np.stack([moving_median(b, win) for b in probs])
    if renormalize and win > 1:
        out = out / np.maximum(out.sum(-1, keepdims=True), 1e-9)
    return out


def adapt_prior(probs: np.ndarray, prior_source: np.ndarray | None = None, iters: int = 25,
                max_ratio: float = 4.0, tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Prior-shift ``(T, C)`` or ``(B, T, C)`` probabilities; returns ``(adjusted, prior)``.

    ``prior_source`` is the class distribution the model was trained on; when omitted it is
    the model's own average prediction over these frames, so the correction only removes
    *shape* mismatch (a class the target rarely shows keeps being predicted at its training
    rate) rather than assuming a lab's label balance. The EM step is
    ``p~(c|x) ~ p(c|x) * pi[c] / prior_source[c]``, with ``pi`` the mean of ``p~``.
    """
    flat = np.asarray(probs, np.float64).reshape(-1, np.asarray(probs).shape[-1])
    flat = flat / np.maximum(flat.sum(-1, keepdims=True), 1e-12)
    n, c = flat.shape
    if c < 2:
        return np.asarray(probs, np.float32), np.asarray(prior_source if prior_source is not None
                                                         else flat.mean(0) if n else np.full(c, 1.0), np.float32)
    if n == 0:
        return np.asarray(probs, np.float32), np.full(c, 1.0 / c, np.float32)
    src = flat.mean(0) if prior_source is None else np.asarray(prior_source, np.float64)
    src = np.clip(src / max(src.sum(), 1e-12), 1e-6, None)
    ratio = np.ones(c)
    for _ in range(max(0, int(iters))):
        w = flat * ratio
        post = w / np.maximum(w.sum(-1, keepdims=True), 1e-12)
        new = np.clip(post.mean(0), 1e-6, None)
        new /= new.sum()
        step = np.clip(new / src, 1.0 / max(max_ratio, 1.0), max(max_ratio, 1.0))
        if np.abs(step - ratio).max() < tol:
            ratio = step
            break
        ratio = step
    w = flat * ratio
    post = (w / np.maximum(w.sum(-1, keepdims=True), 1e-12)).astype(np.float32)
    shape = np.asarray(probs).shape
    return (post.reshape(shape) if len(shape) > 2 else post), (ratio * src).astype(np.float32)


def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax rescaling of probabilities (``probs ** (1/T)`` renormalized)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    p = np.clip(np.asarray(probs, np.float64), 1e-12, None)
    w = p ** (1.0 / temperature)
    return (w / w.sum(-1, keepdims=True)).astype(np.float32)


def fit_temperature(probs: np.ndarray, labels: np.ndarray, lo: float = 0.2, hi: float = 10.0,
                    iters: int = 40) -> float:
    """Temperature minimizing cross-entropy on labelled frames (golden-section search)."""
    p = np.clip(np.asarray(probs, np.float64), 1e-12, None)
    lab = np.asarray(labels).reshape(-1)
    keep = lab >= 0
    p, lab = p[keep], lab[keep]
    if len(lab) < 2:
        return 1.0

    def nll(t: float) -> float:
        q = p ** (1.0 / t)
        q = q / q.sum(-1, keepdims=True)
        return float(-np.log(q[np.arange(len(lab)), lab]).mean())

    phi = (5 ** 0.5 - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(d)
    return float((a + b) / 2)
