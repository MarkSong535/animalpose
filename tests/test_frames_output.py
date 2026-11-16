"""Frame sources on generated media: video files and image folders, checked at pixel, index and
timestamp level (ranges, strides, resampling, mono/16-bit/alpha conversion, median backgrounds)."""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from animalpose.frames import (
    estimate_fps,
    find_sources,
    frame_timestamps,
    image_files,
    iter_frames,
    open_source,
    read_frames,
    sampled_background,
    source_fps,
    timing_report,
    to_bgr8,
)


def _avi(path: Path, n: int, fps: float = 10.0, hw=(24, 32), step: int = 20) -> list[np.ndarray]:
    """A brightness-ramp MJPG video; returns the frames written, for pixel comparison."""
    frames = [np.full((hw[0], hw[1], 3), step * i, np.uint8) for i in range(n)]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (hw[1], hw[0]))
    for f in frames:
        vw.write(f)
    vw.release()
    return frames


def _folder(folder: Path, frames: list[np.ndarray], names: list[str] | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(folder / (names[i] if names else f"f{i:03d}.png")), f)


def test_video_and_image_folder_are_read_alike(tmp_path):
    frames = _avi(tmp_path / "clip.avi", 8)
    _folder(tmp_path / "frames", frames)
    from_video, fps_video = read_frames(tmp_path / "clip.avi")
    from_folder, fps_folder = read_frames(tmp_path / "frames")
    assert fps_video == pytest.approx(10.0) and fps_folder is None
    assert len(from_video) == len(from_folder) == 8
    for got, want in zip(from_video, from_folder):
        assert got.shape == (24, 32, 3) and got.dtype == np.uint8
        assert abs(float(got.mean()) - float(want.mean())) < 3.0
    assert [float(f.mean()) for f in from_video] == pytest.approx([20.0 * i for i in range(8)], abs=3.0)
    assert from_folder[0].shape == (24, 32, 3) and from_folder[0].dtype == np.uint8
    # A folder without timestamps in its names has no known rate; the video does.
    assert source_fps(tmp_path / "clip.avi") == pytest.approx(10.0) and source_fps(tmp_path / "frames") is None


def test_strides_and_limits_select_the_expected_frames(tmp_path):
    _avi(tmp_path / "clip.avi", 8)
    _folder(tmp_path / "frames", [np.full((24, 32, 3), 20 * i, np.uint8) for i in range(8)])
    every2, fps2 = read_frames(tmp_path / "clip.avi", stride=2)
    assert len(every2) == 4 and fps2 == pytest.approx(5.0)
    assert [float(f.mean()) for f in every2] == pytest.approx([0.0, 40.0, 80.0, 120.0], abs=3.0)
    first3, fps3 = read_frames(tmp_path / "clip.avi", max_frames=3)
    assert len(first3) == 3 and fps3 == pytest.approx(10.0)
    strided, fps4 = read_frames(tmp_path / "frames", stride=3)
    assert len(strided) == 3 and fps4 is None
    assert [float(f.mean()) for f in strided] == pytest.approx([0.0, 60.0, 120.0])
    src = open_source(tmp_path / "clip.avi")
    assert src.kind == "video" and src.n_frames == 8
    got = src.frames([7, 0])  # videos are read in index order
    assert [int(i) for i, _ in got] == [0, 7] and float(got[1][1].mean()) > float(got[0][1].mean())
    imgs = open_source(tmp_path / "frames")
    assert imgs.kind == "images" and imgs.n_frames == 8 and [int(i) for i, _ in imgs.frames([5, 2])] == [5, 2]
    assert len(list(iter_frames(tmp_path / "clip.avi", 5))) == 5
    assert len(list(iter_frames(tmp_path / "clip.avi", 40))) == 8  # the capture runs dry, it does not pad
    assert [float(f.mean()) for f in iter_frames(tmp_path / "frames", 2)] == pytest.approx([0.0, 20.0])
    assert len(list(iter_frames(tmp_path / "frames"))) == 8
    kinds = {s.kind for s in find_sources(tmp_path)}
    assert kinds == {"video", "images"}


def test_a_frame_that_cannot_be_decoded_is_skipped_not_invented(tmp_path):
    frames = _avi(tmp_path / "clip.avi", 8)
    folder = tmp_path / "frames"
    _folder(folder, frames)
    (folder / "f008.png").write_bytes(b"\x89PNG\r\n\x1a\nthe rest of the capture is missing")
    imgs = open_source(folder)
    assert imgs.n_frames == 9  # the name is still on disk
    assert [int(i) for i, _ in imgs.frames(range(9))] == list(range(8))
    got, fps = read_frames(folder)
    assert fps is None and len(got) == 8 and [f.shape for f in got] == [(24, 32, 3)] * 8
    assert [float(f.mean()) for f in got] == pytest.approx([20.0 * i for i in range(8)], abs=3.0)
    assert len(list(iter_frames(folder))) == 8
    video = open_source(tmp_path / "clip.avi")
    assert [int(i) for i, _ in video.frames([1, 99, 3])] == [1, 3]  # an index past the end reads nothing


def test_video_timestamps_come_from_the_container(tmp_path, monkeypatch):
    _avi(tmp_path / "clip.avi", 8)
    times = frame_timestamps(tmp_path / "clip.avi")
    assert len(times) == 8 and times[0] == pytest.approx(0.0)
    assert np.all(np.diff(times) > 0) and np.diff(times) == pytest.approx([0.1] * 7, abs=0.02)
    monkeypatch.setattr(shutil, "which", lambda name: None)  # no ffprobe: OpenCV's own positions
    fallback = frame_timestamps(tmp_path / "clip.avi")
    assert len(fallback) == 8 and np.allclose(fallback, times, atol=0.02)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=""))
    empty = frame_timestamps(tmp_path / "clip.avi")  # a probe that lists no packets falls back too
    assert len(empty) == 8 and np.allclose(empty, times, atol=0.02)
    assert len(frame_timestamps(tmp_path / "clip.avi", 3)) == 3


def test_image_folder_timestamps_come_from_names_or_the_estimated_rate(tmp_path):
    names = ["a_23_59_58_900.tif", "a_23_59_59_000.tif", "b_00_00_00_100.tif", "b_00_00_01_100.tif"]
    _folder(tmp_path / "wrap", [np.zeros((8, 8), np.uint8)] * 4, names)
    times = frame_timestamps(tmp_path / "wrap")  # the last two frames are past midnight
    assert times == pytest.approx([0.0, 0.1, 1.2, 2.2])
    assert np.all(np.diff(times) > 0)
    _folder(tmp_path / "plain", [np.zeros((8, 8), np.uint8)] * 5)
    assert frame_timestamps(tmp_path / "plain") == pytest.approx(np.arange(5) / 30.0)
    assert frame_timestamps(tmp_path / "plain", 2) == pytest.approx([0.0, 1 / 30])
    assert estimate_fps(image_files(tmp_path / "plain")) is None
    assert estimate_fps(image_files(tmp_path / "plain")[:2]) is None
    stamps = [f"cam_00_00_00_{100 * i:03d}.tif" for i in range(5)]
    _folder(tmp_path / "stamps", [np.zeros((8, 8), np.uint8)] * 5, stamps)
    assert estimate_fps(image_files(tmp_path / "stamps")) == pytest.approx(10.0)
    assert frame_timestamps(tmp_path / "stamps") == pytest.approx(np.arange(5) * 0.1)


def test_timing_report_sees_dropped_frames():
    assert timing_report(np.array([0.0])) == {"frames": 1}
    steady = np.arange(8) * 0.1
    report = timing_report(steady)
    assert report["frames"] == 8 and report["gaps"] == 0 and not report["variable"]
    assert report["mean_fps"] == pytest.approx(10.0) and report["duration_s"] == pytest.approx(0.7)
    dropped = timing_report(np.r_[steady, steady[-1] + 0.7, steady[-1] + 0.8])
    assert dropped["gaps"] == 1 and dropped["variable"] and dropped["longest_gap_s"] == pytest.approx(0.7)
    assert dropped["interval_ms_p1_p99"][0] == pytest.approx(100.0, abs=1.0)


def test_greyscale_16bit_and_alpha_become_8bit_bgr(tmp_path):
    grey = np.arange(48, dtype=np.uint8).reshape(6, 8)
    bgr = to_bgr8(grey)
    assert bgr.shape == (6, 8, 3) and bgr.dtype == np.uint8
    assert np.array_equal(bgr[..., 0], bgr[..., 2]) and np.array_equal(bgr[..., 1], grey)
    ramp = (np.arange(20, dtype=np.uint16) * 1000).reshape(4, 5)
    auto = to_bgr8(ramp)
    assert auto[0, 0].tolist() == [0, 0, 0] and auto[-1, -1].tolist() == [255, 255, 255]
    assert auto[2, 2, 0] == pytest.approx(255 * 12 / 19, abs=2)
    windowed = to_bgr8(ramp, 4000.0, 12000.0)
    assert windowed[0, 4].tolist() == [0, 0, 0] and windowed[-1, -1].tolist() == [255, 255, 255]
    assert windowed[1, 0, 0] == pytest.approx(255 / 8, abs=2)
    bgra = np.dstack([np.full((4, 5), 30, np.uint8), np.full((4, 5), 90, np.uint8),
                      np.full((4, 5), 150, np.uint8), np.full((4, 5), 7, np.uint8)])
    out = to_bgr8(bgra)
    assert out.shape == (4, 5, 3) and out[0, 0].tolist() == [30, 90, 150]
    for i in range(4):  # one intensity range for the whole 16-bit sequence
        cv2.imwrite(str(tmp_path / f"seq_{i}.tif"), np.full((10, 12), 1000 * (i + 1), np.uint16))
    frames, fps = read_frames(tmp_path)
    assert fps is None and len(frames) == 4 and frames[0].shape == (10, 12, 3) and frames[0].dtype == np.uint8
    assert [float(f.mean()) for f in frames] == pytest.approx([0.0, 85.0, 170.0, 255.0], abs=2.0)
    (tmp_path / "notes.txt").write_text("ignored")
    assert len(image_files(tmp_path)) == 4


def test_sampled_background_drops_the_moving_animal_and_keeps_the_room(tmp_path):
    folder = tmp_path / "cam"
    folder.mkdir()
    for i in range(8):
        f = np.full((40, 60), 180, np.uint8)
        f[4:12, 6:14] = 40  # a static cup
        f[20:30, 4 * i:4 * i + 10] = 20  # the animal, elsewhere in every frame
        cv2.imwrite(str(folder / f"g{i:03d}.png"), f)
    bg = sampled_background(folder)
    assert bg.shape == (40, 60) and np.isfinite(bg).all()
    assert np.abs(bg[20:30, 24:34] - 180).max() < 1e-3  # every place the animal visited is background
    assert abs(float(np.median(bg)) - 180.0) < 1e-3 and float(np.mean(bg[4:12, 6:14])) < 60.0
    one = sampled_background(folder, n=1, max_frames=1)
    assert float(np.mean(one[20:30, 0:10])) < 60.0  # sampling one frame keeps the animal in the background
    assert float(np.mean(bg[20:30, 0:10])) == pytest.approx(180.0)
