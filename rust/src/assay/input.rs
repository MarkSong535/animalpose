//! What the rules read, taken from a finished analysis folder so a rerun needs no network and no
//! GPU: `tracks.npz` of a single view, or for a multiview folder `animals3d.npz`,
//! `view<N>_tracks.npz`, `identities.json`, `views.json` and `cameras.json`.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use super::config::AssayConfig;
use crate::data::sessions::{read_npz, NpyData, Npz};
use crate::multiview::camera::{load_cameras, Camera};
use crate::multiview::dense::Vec3;

/// One tracked animal in one view, in that view's pixels.
#[derive(Clone, Debug, Default)]
pub struct Track2d {
    pub id: i64,
    /// First frame; rows cover `first..first + rows`.
    pub first: usize,
    pub rows: usize,
    /// `(rows, K)` keypoints, NaN where unobserved.
    pub keypoints: Vec<[f64; 2]>,
    /// `(rows, K)` confidences.
    pub conf: Vec<f64>,
    /// `(rows,)` boxes xyxy.
    pub boxes: Vec<[f64; 4]>,
}

impl Track2d {
    fn row(&self, frame: usize) -> Option<usize> {
        let row = frame.checked_sub(self.first)?;
        (row < self.rows).then_some(row)
    }

    /// Keypoint `k` at `frame` with its confidence, if observed.
    pub fn point(&self, frame: usize, k: usize, kpts: usize) -> Option<([f64; 2], f64)> {
        let i = self.row(frame)? * kpts + k;
        let p = *self.keypoints.get(i)?;
        (p[0].is_finite() && p[1].is_finite())
            .then(|| (p, self.conf.get(i).copied().unwrap_or(1.0)))
    }

    pub fn bbox(&self, frame: usize) -> Option<[f64; 4]> {
        let b = *self.boxes.get(self.row(frame)?)?;
        b.iter().all(|v| v.is_finite()).then_some(b)
    }

    /// Frames with any observed keypoint.
    pub fn seen(&self, kpts: usize) -> usize {
        (0..self.rows)
            .filter(|r| {
                self.keypoints[r * kpts..(r + 1) * kpts]
                    .iter()
                    .any(|p| p[0].is_finite())
            })
            .count()
    }
}

/// The cameras of a multiview rig, by role.
#[derive(Clone, Debug)]
pub struct Rig {
    pub top: Camera,
    pub side: Option<Camera>,
}

/// Everything the modules read.
#[derive(Clone, Debug, Default)]
pub struct Input {
    pub n_frames: usize,
    pub fps: f64,
    pub names: Vec<String>,
    /// The test animal in the top (or only) view.
    pub top: Option<Track2d>,
    /// The test animal's 3D keypoints `(n_frames, K)` in box units, NaN where missing.
    pub world: Option<Vec<Vec3>>,
    /// The test animal in the side view.
    pub side_test: Option<Track2d>,
    /// Every other side-view track: caged-animal candidates.
    pub side_others: Vec<Track2d>,
    pub rig: Option<Rig>,
    /// The top view's video, for the motion feature.
    pub top_video: Option<PathBuf>,
}

impl Input {
    pub fn index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    pub fn kpts(&self) -> usize {
        self.names.len()
    }
}

fn floats(npz: &Npz, key: &str) -> Option<Vec<f64>> {
    match &npz.get(key)?.data {
        NpyData::Float(v) => Some(v.iter().map(|x| f64::from(*x)).collect()),
        NpyData::Int(v) => Some(v.iter().map(|x| *x as f64).collect()),
        _ => None,
    }
}

fn texts(npz: &Npz, key: &str) -> Option<Vec<String>> {
    match &npz.get(key)?.data {
        NpyData::Text(v) => Some(v.clone()),
        _ => None,
    }
}

/// The tracks of one view's `tracks.npz`, its frame times and keypoint names.
pub fn read_tracks(path: &Path) -> Result<(Vec<Track2d>, Vec<f64>, Vec<String>)> {
    let npz = read_npz(path)?;
    let names = texts(&npz, "keypoint_names").unwrap_or_else(|| {
        crate::data::synthetic::CANONICAL_NAMES
            .map(str::to_owned)
            .to_vec()
    });
    let k = names.len();
    let times = floats(&npz, "times").unwrap_or_default();
    let mut tracks = Vec::new();
    for key in npz.keys() {
        let Some(id) = key
            .strip_prefix("track")
            .and_then(|r| r.strip_suffix("_frames"))
            .and_then(|id| id.parse::<i64>().ok())
        else {
            continue;
        };
        let p = format!("track{id}");
        let frames = floats(&npz, key).unwrap_or_default();
        let rows = frames.len();
        let first = frames.first().map_or(0, |f| *f as usize);
        let kp = floats(&npz, &format!("{p}_smoothed"))
            .or_else(|| floats(&npz, &format!("{p}_keypoints")))
            .context("a track without keypoints")?;
        let conf = floats(&npz, &format!("{p}_smoothed_conf"))
            .or_else(|| floats(&npz, &format!("{p}_conf")))
            .unwrap_or_else(|| vec![1.0; rows * k]);
        let boxes = floats(&npz, &format!("{p}_boxes")).unwrap_or_else(|| vec![f64::NAN; rows * 4]);
        if kp.len() != rows * k * 2 || conf.len() != rows * k || boxes.len() != rows * 4 {
            bail!("{}: track {id} has inconsistent arrays", path.display());
        }
        tracks.push(Track2d {
            id,
            first,
            rows,
            keypoints: kp.chunks(2).map(|c| [c[0], c[1]]).collect(),
            conf,
            boxes: boxes.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect(),
        });
    }
    tracks.sort_by_key(|t| t.id);
    Ok((tracks, times, names))
}

fn fps_of(times: &[f64]) -> Option<f64> {
    let mut d: Vec<f64> = times
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 0.0)
        .collect();
    if d.is_empty() {
        return None;
    }
    d.sort_by(f64::total_cmp);
    Some(1.0 / d[d.len() / 2])
}

/// The longest track: the test animal when a view holds one free animal.
fn longest(tracks: &[Track2d], k: usize) -> Option<&Track2d> {
    tracks.iter().max_by_key(|t| t.seen(k))
}

/// The view whose camera looks most straight down, and the one that looks most level.
pub fn roles(cameras: &[Camera], cfg: &AssayConfig) -> (usize, Option<usize>) {
    let named = |name: &str| {
        (!name.is_empty())
            .then(|| {
                cameras
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(name))
            })
            .flatten()
    };
    // The optical axis in world coordinates is the third row of R; world z is up.
    let down = |c: &Camera| -c.r[2][2];
    let top = named(&cfg.arena.top_view).unwrap_or_else(|| {
        (0..cameras.len())
            .max_by(|a, b| down(&cameras[*a]).total_cmp(&down(&cameras[*b])))
            .unwrap_or(0)
    });
    let side = named(&cfg.arena.side_view).or_else(|| {
        (0..cameras.len()).filter(|i| *i != top).min_by(|a, b| {
            down(&cameras[*a])
                .abs()
                .total_cmp(&down(&cameras[*b]).abs())
        })
    });
    (top, side)
}

/// Reads an analysis folder.
pub fn read_dir(dir: &Path, cfg: &AssayConfig, video: Option<&Path>) -> Result<Input> {
    if dir.join("animals3d.npz").is_file() {
        return read_multiview(dir, cfg);
    }
    let (tracks, times, names) = read_tracks(&dir.join("tracks.npz"))?;
    let k = names.len();
    let top = longest(&tracks, k).cloned();
    let n_frames = times
        .len()
        .max(tracks.iter().map(|t| t.first + t.rows).max().unwrap_or(0));
    Ok(Input {
        n_frames,
        fps: fps_of(&times).unwrap_or(30.0),
        names,
        top,
        top_video: video.map(Path::to_path_buf),
        ..Input::default()
    })
}

fn read_multiview(dir: &Path, cfg: &AssayConfig) -> Result<Input> {
    let cameras = load_cameras(dir.join("cameras.json"))
        .context("this multiview folder has no cameras.json; analyze it again")?;
    let (top_i, side_i) = roles(&cameras, cfg);
    let views: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("views.json"))?)?;
    let identities: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("identities.json"))?)?;
    let animal_of = |view: usize, track: i64| {
        identities.as_array()?.iter().find_map(|e| {
            (e["view"].as_u64() == Some(view as u64) && e["track"].as_i64() == Some(track))
                .then(|| e["animal"].as_u64())
                .flatten()
        })
    };
    let (top_tracks, times, names) = read_tracks(&dir.join(format!("view{top_i}_tracks.npz")))?;
    let k = names.len();
    let top = longest(&top_tracks, k).cloned();
    let gid = top.as_ref().and_then(|t| animal_of(top_i, t.id));
    let npz = read_npz(dir.join("animals3d.npz"))?;
    let world = gid
        .and_then(|g| floats(&npz, &format!("animal{g}_keypoints3d")))
        .map(|v| {
            v.chunks(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect::<Vec<Vec3>>()
        });
    let n_frames = times
        .len()
        .max(world.as_ref().map_or(0, |w| w.len() / k.max(1)));
    let (mut side_test, mut side_others) = (None, Vec::new());
    if let Some(s) = side_i {
        let (tracks, _, _) = read_tracks(&dir.join(format!("view{s}_tracks.npz")))?;
        let mut mine: Vec<Track2d> = Vec::new();
        for t in tracks {
            if gid.is_some() && animal_of(s, t.id) == gid {
                mine.push(t);
            } else {
                side_others.push(t);
            }
        }
        side_test = longest(&mine, k).cloned();
    }
    let top_video = views
        .as_array()
        .and_then(|v| v.get(top_i))
        .and_then(|v| v["video"].as_str())
        .map(PathBuf::from);
    Ok(Input {
        n_frames,
        fps: fps_of(&times).unwrap_or(30.0),
        names,
        top,
        world,
        side_test,
        side_others,
        rig: Some(Rig {
            top: cameras[top_i].clone(),
            side: side_i.map(|s| cameras[s].clone()),
        }),
        top_video,
    })
}
