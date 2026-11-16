"""Held-out synthetic benchmark for every stage.

All numbers come from simulations with seeds disjoint from training, so they
measure whether each component works and how the parts interact. They are
NOT estimates of accuracy on real rodent video; see README for that caveat.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from animalpose.behavior.data import synthetic_batch as behavior_batch
from animalpose.data.synthetic import (
    SimulationPool,
    camera_from_view,
    make_cameras,
    random_camera,
    render,
    render_video,
    simulate,
)
from animalpose.eval.metrics import auroc, average_precision, f1_scores, linear_interpolate, pck, rmse, tracking_metrics
from animalpose.pipeline import MultiViewPipeline, PosePipeline
from animalpose.skeleton import CANONICAL
from animalpose.temporal.data import synthetic_batch as temporal_batch

VIEWS = {"top": (90.0, -90.0), "angled": (45.0, -135.0), "side": (12.0, -90.0)}
SEED = 10_000  # training uses small seeds; evaluation starts far away


def _frames(n, seed, size, view=None, species=("mouse", "rat")):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        sim = simulate(int(rng.integers(1, 5)), 60, str(rng.choice(species)), seed=rng)
        cam = (camera_from_view(sim.arena_cm, *VIEWS[view], size=size) if view else random_camera(sim.arena_cm, rng, size))
        yield sim, render(sim, cam, int(rng.integers(30, 60)), rng)


def eval_detection(pipe: PosePipeline, n: int = 200, size=(320, 256)) -> dict:
    out = {}
    for view in [None, *VIEWS]:
        boxes, scores, gts = [], [], []
        for _, f in _frames(n if view is None else n // 2, SEED + 1, size, view):
            b, s = pipe.detect(f.image)
            boxes.append(b), scores.append(s), gts.append(f.boxes[f.present])
        out[view or "random_views"] = {"ap50": average_precision(boxes, scores, gts, 0.5),
                                       "ap75": average_precision(boxes, scores, gts, 0.75)}
    return out


def eval_pose(pipe: PosePipeline, n: int = 200, size=(320, 256)) -> dict:
    out = {}
    for view in [None, *VIEWS]:
        acc = {"mouse": [], "rat": []}
        for sim, f in _frames(n if view is None else n // 2, SEED + 2, size, view):
            idx = np.nonzero(f.present)[0]
            if len(idx) == 0:
                continue
            kp, cf, _ = pipe.estimate(f.image, f.boxes[idx])
            for i, a in enumerate(idx):
                b = f.boxes[a]
                acc[sim.species].append((kp[i], f.keypoints[a], f.visible[a], max(b[2] - b[0], b[3] - b[1]),
                                         cf[i]))
        res = {}
        for sp, items in list(acc.items()) + [("all", acc["mouse"] + acc["rat"])]:
            if not items:
                continue
            p, g, v, s, c = (np.stack(x) for x in zip(*items))
            res[sp] = {"pck@0.1": pck(p, g, v, s, 0.1), "pck@0.05": pck(p, g, v, s, 0.05), "rmse_px": rmse(p, g, v),
                       "n_instances": len(items),
                       # Visibility calibration: mean confidence on visible vs hidden points.
                       "conf_visible": float(c[v].mean()), "conf_hidden": float(c[~v].mean()) if (~v).any() else None}
        out[view or "random_views"] = res
    return out


def eval_temporal(model, dims: int = 2, batches: int = 8, window: int = 64) -> dict:
    rng = np.random.default_rng(SEED + 3)
    pool = SimulationPool(16, 400, seed=SEED + 3)
    dev = next(model.parameters()).device
    raw_o, sm_o, lin_g, sm_g, gt_o, gt_g = [], [], [], [], [], []
    cover = []
    for _ in range(batches):
        x, conf, obs, target, _ = temporal_batch(16, window, rng, dims, pool=pool)
        with torch.no_grad():
            out = model.eval()(*(torch.from_numpy(a).to(dev) for a in (x, conf, obs)))["coords"].cpu().numpy()
        for i in range(len(x)):
            o = obs[i] > 0
            lin = linear_interpolate(x[i], o)
            raw_o.append(x[i][o]), sm_o.append(out[i][o]), gt_o.append(target[i][o])
            g = ~o & np.isfinite(lin).all(-1)
            lin_g.append(lin[g]), sm_g.append(out[i][g]), gt_g.append(target[i][g])
            cover.append(np.isfinite(lin).all(-1).mean())

    def r(a, b):
        a, b = np.concatenate(a), np.concatenate(b)
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, -1))))

    unit = "px" if dims == 2 else "cm"
    return {f"observed_raw_rmse_{unit}": r(raw_o, gt_o), f"observed_smoothed_rmse_{unit}": r(sm_o, gt_o),
            f"gap_linear_interp_rmse_{unit}": r(lin_g, gt_g), f"gap_lstm_rmse_{unit}": r(sm_g, gt_g),
            "note": "gap metrics only over gaps that linear interpolation can fill (both ends observed or held)"}


def eval_behavior(model, dims: int = 2, batches: int = 8, window: int = 128) -> dict:
    rng = np.random.default_rng(SEED + 4)
    pool = SimulationPool(24, 600, seed=SEED + 4)
    dev = next(model.parameters()).device
    preds, gts = [], []
    for _ in range(batches):
        x, m, y = behavior_batch(16, window, rng, dims, pool=pool)
        with torch.no_grad():
            logits = model.eval()(torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev))
        preds.append(logits.argmax(-1).cpu().numpy()), gts.append(y)
    res = f1_scores(np.concatenate(preds), np.concatenate(gts), len(model.classes))
    res["classes"] = list(model.classes)
    return res


def _gt_track_lists(frames):
    return ([np.nonzero(f.present)[0] for f in frames], [f.boxes[f.present] for f in frames])


def eval_tracking(pipe: PosePipeline, videos: int = 4, frames: int = 300, n_animals: int = 3) -> dict:
    out = {}
    for view in ("top", "angled"):
        agg = []
        for v in range(videos):
            sim = simulate(n_animals, frames, "mouse", seed=SEED + 50 + v)
            cam = make_cameras(sim.arena_cm, (view,))[0]
            fr = render_video(sim, cam, seed=v)
            res = pipe.track([f.image for f in fr])
            gt_ids, gt_boxes = _gt_track_lists(fr)
            pr_ids = [[] for _ in fr]
            pr_boxes = [[] for _ in fr]
            for tid, tr in res.tracks.items():
                for i, t in enumerate(tr.frames):
                    if np.isfinite(tr.boxes[i]).all():
                        pr_ids[t].append(tid), pr_boxes[t].append(tr.boxes[i])
            agg.append(tracking_metrics(gt_ids, gt_boxes, [np.array(p) for p in pr_ids], [np.array(b) for b in pr_boxes]))
        out[view] = {k: float(np.mean([a[k] for a in agg])) for k in ("mota", "idf1", "id_switches", "n_pred_ids")}
        out[view]["n_gt_ids"] = n_animals
    return out


def eval_multiview(pipe: PosePipeline, temporal3d=None, frames: int = 150, n_animals: int = 2) -> dict:
    sim = simulate(n_animals, frames, "rat", seed=SEED + 77, coats=["black", "white", "agouti"][:n_animals])
    cams = make_cameras(sim.arena_cm, ("top", "side", "angled"), size=(320, 256))
    videos = [[f.image for f in render_video(sim, c, seed=i)] for i, c in enumerate(cams)]
    mv = MultiViewPipeline(pipe, cams, temporal3d)
    res = mv.run(videos)
    animals = res["animals"]
    gids = sorted(animals)
    cost = np.full((n_animals, len(gids)), 1e6)
    for a in range(n_animals):
        for j, g in enumerate(gids):
            d = np.linalg.norm(animals[g]["keypoints3d"] - sim.kpts3d[a], axis=-1)
            if np.isfinite(d).sum() > 50:
                cost[a, j] = np.nanmedian(d)
    rows, cols = linear_sum_assignment(cost)
    out = {"n_global_ids": len(gids), "n_animals": n_animals}
    errs_raw, errs_s, coverage = [], [], []
    for a, j in zip(rows, cols):
        X = animals[gids[j]]["keypoints3d"]
        d = np.linalg.norm(X - sim.kpts3d[a], axis=-1)
        errs_raw.append(d[np.isfinite(d)])
        coverage.append(np.isfinite(d).mean())
        if "smoothed3d" in animals[gids[j]]:
            errs_s.append(np.linalg.norm(animals[gids[j]]["smoothed3d"] - sim.kpts3d[a], axis=-1).ravel())
    e = np.concatenate(errs_raw) if errs_raw else np.empty(0)
    out.update({"mpjpe_cm": float(e.mean()) if e.size else None,
                "median_error_cm": float(np.median(e)) if e.size else None,
                "triangulated_fraction": float(np.mean(coverage)) if coverage else 0.0,
                "body_length_cm": float(np.mean([ap.size for ap in sim.appearance]))})
    if errs_s:
        out["mpjpe_smoothed_all_frames_cm"] = float(np.concatenate(errs_s).mean())
    return out


def throughput(pipe: PosePipeline, n: int = 30, size=(320, 256)) -> dict:
    frames = [f for _, f in _frames(n, SEED + 9, size)]
    sync = (torch.cuda.synchronize if pipe.device.type == "cuda"
            else torch.mps.synchronize if pipe.device.type == "mps" else (lambda: None))
    for f in frames[:3]:  # warm-up
        pipe.estimate(f.image, pipe.detect(f.image)[0])
    t0 = time.perf_counter()
    for f in frames:
        pipe.detect(f.image)
    sync()
    t1 = time.perf_counter()
    crops = 0
    for f in frames:
        crops += len(f.boxes[f.present])
        pipe.estimate(f.image, f.boxes[f.present])
    sync()
    t2 = time.perf_counter()
    pipe.track([f.image for f in frames])
    t3 = time.perf_counter()
    return {"device": str(pipe.device), "detector_ms_per_frame": 1e3 * (t1 - t0) / n,
            "pose_ms_per_animal": 1e3 * (t2 - t1) / max(crops, 1), "detect_pose_track_fps": n / (t3 - t2)}


# ----------------------------------------------------------------------------- real data

def eval_coco(pipe: PosePipeline, source, max_images: int = 400, detect: bool = True) -> dict:
    """Held-out real images (COCO, canonical names): pose with GT boxes, detection AP,
    and whether confidence separates visible from labeled-occluded points."""
    n = min(len(source), max_images)
    idx = np.linspace(0, len(source) - 1, n).astype(int)
    preds, gts, vis, norms, occl_conf, vis_conf = [], [], [], [], [], []
    det_b, det_s, det_g = [], [], []
    for i in idx:
        ann = source.sample(int(i))
        if len(ann.boxes) == 0:
            continue
        kp, cf, _ = pipe.estimate(ann.image, ann.boxes)
        for a in range(len(ann.boxes)):
            b = ann.boxes[a]
            preds.append(kp[a]), gts.append(ann.keypoints[a]), vis.append(ann.state[a] == 2)
            norms.append(max(b[2] - b[0], b[3] - b[1]))
            vis_conf.append(cf[a][ann.state[a] == 2]), occl_conf.append(cf[a][ann.state[a] == 1])
        if detect:
            b, s = pipe.detect(ann.image)
            det_b.append(b), det_s.append(s), det_g.append(ann.boxes)
    p, g, v, s = np.stack(preds), np.stack(gts), np.stack(vis), np.asarray(norms)
    d = np.linalg.norm(p - g, axis=-1)
    per_kp = {CANONICAL.names[k]: float((d[:, k][v[:, k]] <= 0.1 * s[v[:, k]]).mean())
              for k in range(d.shape[1]) if v[:, k].sum() >= 20}
    out = {"n_instances": int(len(p)), "pck@0.1": pck(p, g, v, s, 0.1), "pck@0.05": pck(p, g, v, s, 0.05),
           "rmse_px": rmse(p, g, v), "median_err_px": float(np.median(d[v])),
           "median_box_px": float(np.median(s)), "per_keypoint_pck@0.1": per_kp}
    oc, vc = np.concatenate(occl_conf), np.concatenate(vis_conf)
    if len(oc):
        out["conf_auroc_visible_vs_occluded"] = auroc(vc, oc)
    if detect:
        out["det_ap50"] = average_precision(det_b, det_s, det_g, 0.5)
        out["det_ap75"] = average_precision(det_b, det_s, det_g, 0.75)
    return out
