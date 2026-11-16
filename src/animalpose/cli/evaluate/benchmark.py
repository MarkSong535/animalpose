"""Evaluate trained checkpoints on held-out synthetic data and write results/benchmark.json and results/benchmark.md.

    animalpose eval benchmark --runs runs --out results
"""

from __future__ import annotations

import json
from pathlib import Path

from animalpose.eval import benchmark as B
from animalpose.pipeline import PipelineConfig, PosePipeline
from animalpose.tracking.tracker import TrackerConfig
from animalpose.train.checkpoint import load_model

NAME = "benchmark"


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def to_markdown(r: dict) -> str:
    lines = ["# Synthetic benchmark", "",
             "Held-out procedural rodent scenes (seeds disjoint from training). These numbers show the",
             "components work together; they are not real-video accuracy.", ""]
    if "detection" in r:
        lines += ["## Detection", "", "| view | AP50 | AP75 |", "|---|---|---|"]
        lines += [f"| {k} | {_fmt(v['ap50'])} | {_fmt(v['ap75'])} |" for k, v in r["detection"].items()]
    if "pose" in r:
        lines += ["", "## Pose (GT boxes, visible keypoints)", "",
                  "| view | species | PCK@0.1 | PCK@0.05 | RMSE px | conf visible / hidden |", "|---|---|---|---|---|---|"]
        for view, d in r["pose"].items():
            for sp, m in d.items():
                lines.append(f"| {view} | {sp} | {_fmt(m['pck@0.1'])} | {_fmt(m['pck@0.05'])} | {_fmt(m['rmse_px'])} | "
                             f"{_fmt(m['conf_visible'])} / {_fmt(m['conf_hidden'])} |")
    for key, title in (("temporal2d", "Temporal LSTM, 2D"), ("temporal3d", "Temporal LSTM, 3D")):
        if key in r:
            lines += ["", f"## {title}", "", "| metric | value |", "|---|---|"]
            lines += [f"| {k} | {_fmt(v)} |" for k, v in r[key].items() if k != "note"]
    for key, title in (("behavior2d", "Behavior, 2D random views"), ("behavior3d", "Behavior, 3D")):
        if key in r:
            m = r[key]
            lines += ["", f"## {title}", "", f"macro-F1 {_fmt(m['macro_f1'])}, accuracy {_fmt(m['accuracy'])}", "",
                      "| class | F1 |", "|---|---|"]
            lines += [f"| {c} | {_fmt(f)} |" for c, f in zip(m["classes"], m["per_class_f1"])]
    for key in [k for k in r if k.startswith("tracking")]:
        lines += ["", f"## {key.replace('_', ' ').capitalize()} (full detect+pose+track, 3 mice, 300 frames)", "",
                  "| view | MOTA | IDF1 | ID switches | predicted ids (gt 3) |", "|---|---|---|---|---|"]
        lines += [f"| {k} | {_fmt(v['mota'])} | {_fmt(v['idf1'])} | {_fmt(v['id_switches'])} | {_fmt(v['n_pred_ids'])} |"
                  for k, v in r[key].items()]
    if "multiview" in r:
        lines += ["", "## Multi-camera (3 views, 2 rats, full pipeline + fusion + triangulation)", "",
                  "| metric | value |", "|---|---|"] + [f"| {k} | {_fmt(v)} |" for k, v in r["multiview"].items()]
    if "throughput" in r:
        lines += ["", "## Throughput (320x256 frames)", "",
                  "| device | detector ms/frame | pose ms/animal | detect+pose+track FPS |", "|---|---|---|---|"]
        lines += [f"| {t['device']} | {_fmt(t['detector_ms_per_frame'])} | {_fmt(t['pose_ms_per_animal'])} | "
                  f"{_fmt(t['detect_pose_track_fps'])} |" for t in r["throughput"]]
    return "\n".join(lines) + "\n"


def add_arguments(parser) -> None:
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--quick", action="store_true", help="fewer samples")
    parser.add_argument("--devices", nargs="+", default=["auto", "cpu"])


def run(args) -> None:
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 60 if args.quick else 200
    load = lambda name: load_model(runs / f"{name}.pt") if (runs / f"{name}.pt").exists() else None  # noqa: E731
    det, pose, tmp2, tmp3, beh2, beh3 = (load(k) for k in
                                         ("detector", "pose", "temporal", "temporal3d", "behavior", "behavior3d"))
    r: dict = {}
    pipe = PosePipeline(det, pose, tmp2, beh2, PipelineConfig(tracker=TrackerConfig(max_animals=None)))
    print("detection..."), r.update(detection=B.eval_detection(pipe, n))
    print("pose..."), r.update(pose=B.eval_pose(pipe, n))
    if tmp2 is not None:
        print("temporal 2d..."), r.update(temporal2d=B.eval_temporal(tmp2.to(pipe.device), 2))
    if tmp3 is not None:
        print("temporal 3d..."), r.update(temporal3d=B.eval_temporal(tmp3.to(pipe.device), 3))
    if beh2 is not None:
        print("behavior 2d..."), r.update(behavior2d=B.eval_behavior(beh2.to(pipe.device), 2))
    if beh3 is not None:
        print("behavior 3d..."), r.update(behavior3d=B.eval_behavior(beh3.to(pipe.device), 3))
    videos = 2 if args.quick else 4
    print("tracking..."), r.update(tracking_model_embedding=B.eval_tracking(pipe, videos))
    pipe.cfg.embedder = "color"
    r.update(tracking_color_histogram=B.eval_tracking(pipe, videos))
    pipe.cfg.embedder = "model"
    print("multiview..."), r.update(multiview=B.eval_multiview(pipe, tmp3.to(pipe.device) if tmp3 is not None else None))
    r["throughput"] = []
    for dev in args.devices:
        p = PosePipeline(det, pose, config=PipelineConfig(), device=dev)
        if str(p.device) not in [t["device"] for t in r["throughput"]]:
            r["throughput"].append(B.throughput(p))
    (out / "benchmark.json").write_text(json.dumps(r, indent=2))
    (out / "benchmark.md").write_text(to_markdown(r))
    print(to_markdown(r))
