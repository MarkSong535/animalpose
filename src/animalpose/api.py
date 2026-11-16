"""The library surface: five calls, the same ones the command line runs.

    import animalpose

    animalpose.train("pose", "configs/pose.yaml", ["train.steps=200"])
    animalpose.finetune("detector", "configs/detector.yaml", init="weights/detector_general.pt")
    animalpose.label("data/raw/videos", include=["my_video"], boxes="blob", out="data/processed/my_video_pseudo")
    animalpose.predict("my_video.mp4", config="configs/pipeline.yaml", out="outputs/my_video")
    animalpose.behavior("cluster", tracks="outputs/my_video/tracks.npz", video="my_video.mp4", out="outputs/my_video/clusters")
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from animalpose.behavior import clusters as _clusters
from animalpose.config import TRAIN_SCHEMA, load_config, pipeline_schema
from animalpose.frames import (
    frame_timestamps,
    iter_frames,
    read_frames,
    sampled_background,
    source_fps,
    timing_report,
)
from animalpose.multiview.camera import Camera
from animalpose.pipeline import MultiViewPipeline, PipelineConfig, PosePipeline
from animalpose.selftrain import LabelOptions, label_videos
from animalpose.tracking.tracker import TrackerConfig
from animalpose.train.loops import TRAINERS
from animalpose.viz import draw_frame

__all__ = ["Prediction", "behavior", "build_pipeline", "cluster_tracks", "finetune", "label", "label_clusters",
           "predict", "train", "write_naming", "write_video"]

MODELS = tuple(sorted(TRAINERS))


@dataclass(frozen=True)
class Prediction:
    """What :func:`predict` left in ``out``, plus the tracks still in memory.

    A single recording fills ``frames``, ``tracks`` and ``overlay_frames`` (``tracks.npz``, ``tracks.csv``,
    ``overlay.mp4``); a multi-camera run fills ``animals``, ``views`` and ``per_view`` (``animals3d.npz`` and
    one ``view<i>_tracks.csv`` per view). ``result`` is the tracking result itself, for code that would rather
    read the tracks than re-open the file; ``behaviors`` is the classifier's catalogue, if it ran.
    """

    out: Path
    frames: int = 0
    tracks: int = 0
    overlay_frames: int = 0
    behaviors: tuple[str, ...] = ()
    animals: int = 0
    views: int = 0
    result: Any = None
    per_view: tuple[Any, ...] = ()


def train(model: str, config: str | Path, overrides: Sequence[str] = ()) -> tuple[Any, list[dict]]:
    """Train one network from a YAML config; returns ``(model, history)``, saving ``output`` and its log."""
    if model not in TRAINERS:
        raise ValueError(f"unknown model {model!r}; choose from {', '.join(MODELS)}")
    cfg = load_config(config, list(overrides), TRAIN_SCHEMA)
    trained, history = TRAINERS[model](cfg)
    if cfg.get("output"):
        log = Path(cfg["output"]).with_suffix(".history.json")
        log.write_text(json.dumps(history))
        print(f"saved {cfg['output']} and {log}")
    return trained, history


def finetune(model: str, config: str | Path, init: str | Path, overrides: Sequence[str] = ()) -> tuple[Any, list[dict]]:
    """Fine-tune an existing checkpoint: :func:`train` with ``init: <checkpoint>`` set in the config."""
    return train(model, config, [f"init={init}", *overrides])


def label(root: str | Path, *, options: LabelOptions | None = None, **changes: Any) -> dict:
    """Pseudo-label boxes and keypoints for every recording under ``root``.

    ``options`` is an :class:`animalpose.selftrain.LabelOptions` (teacher checkpoints and thresholds);
    ``changes`` override single fields, so ``label("videos", boxes="blob", out="labels")`` needs no
    options object. See :mod:`animalpose.selftrain` for what a label has to survive.
    """
    return label_videos(root, options, **changes)


def build_pipeline(cfg: dict) -> PosePipeline:
    """A :class:`PosePipeline` from a pipeline config's ``checkpoints``, ``pipeline`` and ``tracker``."""
    ck = cfg["checkpoints"]
    pc = PipelineConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.get("pipeline", {}).items()},
                        tracker=TrackerConfig(**cfg.get("tracker", {})))
    return PosePipeline.from_checkpoints(ck["detector"], ck["pose"], ck.get("temporal"), ck.get("behavior"),
                                         pc, cfg.get("device", "auto"))


def write_video(path: str | Path, frames, fps: float = 30.0) -> int:
    """Write an iterable of frames; returns how many were written."""
    vw, n = None, 0
    for f in frames:
        if vw is None:
            h, w = f.shape[:2]
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        vw.write(f)
        n += 1
    if vw is not None:
        vw.release()
    return n


def predict(video: str | Path | None = None, *, config: str | Path = "configs/pipeline.yaml",
            out: str | Path = "outputs/run", overrides: Sequence[str] = (), views: Sequence[str] | None = None,
            cameras: str | Path | None = None, synthetic: bool = False, max_frames: int | None = None,
            fps: float | None = None, overlay: bool = True) -> Prediction:
    """Detect, track and classify one recording; writes ``tracks.npz``, ``tracks.csv`` and ``overlay.mp4``.

    ``views`` + ``cameras`` (a JSON list of ``Camera.to_dict()``) run the multi-camera pipeline and also
    write ``animals3d.npz``; ``synthetic`` renders a two-mouse demo instead of reading a file.
    """
    cfg = load_config(config, list(overrides), pipeline_schema())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(cfg)

    if views:
        cams = [Camera.from_dict(d) for d in json.loads(Path(cameras).read_text())]
        res = MultiViewPipeline(pipe, cams).run([read_frames(v, max_frames)[0] for v in views])
        np.savez_compressed(out / "animals3d.npz", **{f"animal{g}_{k}": v for g, a in res["animals"].items()
                                                      for k, v in a.items() if isinstance(v, np.ndarray)})
        for v, r in enumerate(res["per_view"]):
            r.to_csv(out / f"view{v}_tracks.csv")
        return Prediction(out=out, animals=len(res["animals"]), views=len(cams), result=res,
                          per_view=tuple(res["per_view"]))

    if synthetic:
        from animalpose.data.synthetic import make_cameras, render_video, simulate

        sim = simulate(2, max_frames or 300, "mouse", seed=7, coats=["black", "white"])
        frames = [f.image for f in render_video(sim, make_cameras(sim.arena_cm, ("angled",))[0])]
        result = pipe.run(frames)
        source_frames, fps = frames, fps or 30.0
    else:  # stream: long recordings never sit in memory
        times = frame_timestamps(video, max_frames)
        timing = timing_report(times)
        (out / "timing.json").write_text(json.dumps(timing, indent=1))
        if timing.get("variable"):
            print(f"variable frame rate / dropped frames: {timing}; using real timestamps")
        fps = fps or timing.get("mean_fps") or source_fps(video)
        need_bg = pipe.cfg.background_filter or "blob" in pipe.cfg.detector_candidates
        bg = sampled_background(video, max_frames=max_frames) if need_bg else None
        result = pipe.run(iter_frames(video, max_frames), background=bg, fps=fps, times=times)
        source_frames = iter_frames(video, max_frames)

    result.to_npz(out / "tracks.npz")
    result.to_csv(out / "tracks.csv")
    info = None
    if result.behavior_info:
        result.to_behavior_csv(out / "behavior.csv", pipe.cfg.behavior_top_k)
        (out / "behavior_catalog.json").write_text(json.dumps(result.behavior_info, indent=1))
        info = [entry["name"] for entry in result.behavior_info]  # describe(): one entry per catalogue class
    frames_written = write_video(out / "overlay.mp4",
                                 (draw_frame(f, result, t) for t, f in enumerate(source_frames)), fps) if overlay else 0
    return Prediction(out=out, frames=result.n_frames, overlay_frames=frames_written,
                      tracks=len(result.tracks), behaviors=tuple(info or ()), result=result)


cluster_tracks = _clusters.cluster_tracks
write_naming = _clusters.write_naming
label_clusters = _clusters.label_clusters

BEHAVIOR_TASKS = {"cluster": cluster_tracks, "name": write_naming, "label": label_clusters}


def behavior(task: str, /, **options: Any) -> Any:
    """Bootstrap behavior labels for a new recording: ``cluster``, ``name`` or ``label``.

    The three steps are :func:`cluster_tracks`, :func:`write_naming` and :func:`label_clusters`; call those
    directly for their signatures, or this one with the task name — which is what ``animalpose behavior`` does.
    """
    step = BEHAVIOR_TASKS.get(task)
    if step is None:
        raise ValueError(f"behavior task must be one of {', '.join(BEHAVIOR_TASKS)}") from None
    return step(**options)
