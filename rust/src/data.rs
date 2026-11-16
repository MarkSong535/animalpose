//! Rust port of the data side of `animalpose`: the procedural simulator and its renderer, camera
//! degradations, the detector / pose datasets, the config loader the Python trainers read,
//! behavior session files and frame input.
//!
//! | Python | Rust |
//! | --- | --- |
//! | `src/animalpose/config.py:load_config` | [`crate::data::config::Config::load`] |
//! | `src/animalpose/data/synthetic.py:simulate` / `body_pose` / `random_camera` | [`crate::data::synthetic::simulate`] / [`crate::data::synthetic::body_pose`] / [`crate::data::synthetic::random_camera`] |
//! | `src/animalpose/data/synthetic.py:SimulationPool` | [`crate::data::synthetic::SimulationPool`] |
//! | `src/animalpose/temporal/data.py:corrupt` / `synthetic_batch` | [`crate::data::synthetic::corrupt`] / [`crate::data::synthetic::temporal_batch`] |
//! | `src/animalpose/behavior/data.py:synthetic_batch` | [`crate::data::synthetic::behavior_batch`] |
//! | `src/animalpose/data/tracks.py:behavior_windows` | [`crate::data::sessions::behavior_windows`] |
//! | `src/animalpose/data/behavior_sets.py:load_session_dir` / `remap_labels` | [`crate::data::sessions::load_session_dir`] / [`crate::data::sessions::remap_labels`] |
//! | `src/animalpose/frames.py:read_frames` (image folders, `.npy` frames, videos) | [`crate::data::images::read_frames`] |
//! | `src/animalpose/frames.py:Source` / `iter_frames` / `sampled_background` / `frame_timestamps` / `timing_report` | [`crate::data::frames`] |
//! | `cv2.VideoCapture` / `cv2.VideoWriter` | [`crate::data::video`] (the `ffmpeg` / `ffprobe` executables) |
//! | `src/animalpose/data/synthetic.py:render` / `render_video` / `Frame` / `_draw_animal` / `_background` / `make_cameras` | [`crate::data::render::render`] / [`crate::data::render::render_video`] / [`crate::data::render::Frame`] / [`crate::data::render::draw_animal`] / [`crate::data::render::background`] / [`crate::data::render::make_cameras`] |
//! | `src/animalpose/data/degrade.py:degrade` / `Profile` / `load_profile` / `zoom_out` / `edge_width` / `motion_smear` | [`crate::data::degrade::degrade`] / [`crate::data::degrade::Profile`] / [`crate::data::degrade::load_profile`] / [`crate::data::degrade::zoom_out`] / [`crate::data::degrade::edge_width`] / [`crate::data::degrade::motion_smear`] |
//! | `src/animalpose/data/datasets.py:SyntheticScenes` / `CocoKeypoints` / `MixedSource` / `photometric` | [`crate::data::datasets::SyntheticScenes`] / [`crate::data::datasets::CocoKeypoints`] / [`crate::data::datasets::MixedSource`] / [`crate::data::datasets::photometric`] |
//! | `src/animalpose/data/datasets.py:PoseCropDataset` (`degraded_crop`) / `DetectionDataset` | [`crate::data::datasets::PoseCropDataset`] / [`crate::data::datasets::DetectionDataset`] |
//! | `src/animalpose/train/loops.py:_source` | [`crate::data::datasets::source_from_config`] |
//! | the `cv2` drawing, resampling and filtering those call | [`crate::data::raster`] |
//!
//! A config may inherit from others with `extends:`: one path or a list of them, resolved relative
//! to the declaring file and merged under the declaring file's own keys before the overrides run.
//!
//! Rendering, degradation and the two image datasets feed `animalpose train pose|detector` and
//! `animalpose predict --synthetic`; the OpenCV primitives they use are re-implemented in
//! [`crate::data::raster`] (see its table of fitted rasteriser offsets), so a rendered frame matches
//! Python's in sizes, colours, depth order and annotations but not pixel for pixel. Not ported from
//! `data/`: the dataset converters of `data/real.py` / `data/dlc.py` (their COCO output is what
//! [`crate::data::datasets::CocoKeypoints`] reads) and `data/tracks.py:load_sessions` (HDF5).

pub mod config;
pub mod datasets;
pub mod degrade;
pub mod dlc;
pub mod frames;
pub mod images;
pub mod mat5;
pub mod measure;
pub mod np_random;
pub mod pyjson;
pub mod raster;
pub mod real;
pub mod render;
pub mod sessions;
pub mod synthetic;
pub mod video;

use std::path::{Path, PathBuf};

/// The repository root, found by walking up from this crate's directory; `None` when the crate is
/// used outside a checkout (which is how the fixture tests find `tests/fixtures/` and `configs/`).
pub fn repo_root() -> Option<PathBuf> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut at = manifest.to_path_buf();
    // `rust` -> one level up is the repository root.
    for _ in 0..4 {
        if at.join("src/animalpose").is_dir() && at.join("rust").is_dir() {
            return Some(at);
        }
        if !at.pop() {
            break;
        }
    }
    None
}

/// The fixture directory: `$ANIMALPOSE_FIXTURES` when set, else `<repo>/tests/fixtures`.
///
/// The same convention the `tensor` and `models` fixture tests use, so one variable
/// points the whole package at a generated fixture tree.
pub fn fixtures_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("ANIMALPOSE_FIXTURES") {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            return Some(path);
        }
    }
    repo_root().map(|root| root.join("tests").join("fixtures"))
}
