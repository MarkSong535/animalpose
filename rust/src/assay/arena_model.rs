//! Arena detectors: the arena of a video found by a trained network (`scripts/train_arena.py`).
//!
//! The network is the pose network over the whole top-view frame, one heatmap and offset per arena
//! point. Its checkpoint (kind `arena`) carries the arena definition — `configs/arenas/<type>.yaml`
//! as `animalpose.arena` writes it into the model config's `arena` entry: the point names, the
//! input size, and the floor and zones as point references. Nothing here knows an arena type, so a
//! new setup is a new definition and a trained model.
//!
//! A video's arena is the per-point median over [`FRAMES`] frames sampled across it (a mouse or a
//! hand over a corner in a few frames does not move it). The result is checked before it is
//! trusted: each point found with confidence, the frames agreeing on it, the floor a convex
//! quadrilateral inside the image. A layout that fails a check is `needs_review`, with the reason.
//! Open and closed maze arms are confirmed against the walls seen in the video.

use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

use super::arena::{Layout, Polygon, Pt, Zone};
use super::config::Maze;
use crate::algos::decode::{pose_decode, HeatMaps};
use crate::algos::image::Image;
use crate::models::{self, Model, ModelKind};
use crate::tensor::{Ctx, Graph, NodeId};

/// Frames sampled across a video to find its arena.
pub const FRAMES: usize = 16;
/// A point found with lower confidence (median over the frames) makes the layout `needs_review`.
pub const MIN_CONF: f64 = 0.3;
/// A point that moves more than this share of the image width between frames (median distance
/// from its median position) makes the layout `needs_review`.
pub const MAX_SPREAD: f64 = 0.02;

/// A point of the arena: a point number (1-based), or `"a-b x c-d"`, where the line through
/// points a and b crosses the line through c and d.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum PointRef {
    Index(usize),
    Cross(String),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ZoneDef {
    pub name: String,
    pub kind: String,
    pub polygon: Vec<PointRef>,
}

/// The arena definition a model carries (`animalpose.arena.Arena.as_model_cfg`).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ArenaDef {
    pub name: String,
    #[serde(default)]
    pub experiment: String,
    pub points: Vec<String>,
    #[serde(default)]
    pub floor: Vec<PointRef>,
    #[serde(default)]
    pub zones: Vec<ZoneDef>,
    /// Network input `[width, height]`.
    pub input_size: [usize; 2],
}

impl PointRef {
    /// The point numbers it uses (1-based), checked against `n` points.
    fn indices(&self, n: usize) -> Result<Vec<usize>> {
        let idx = match self {
            Self::Index(i) => vec![*i],
            Self::Cross(text) => {
                let parts: Vec<&str> = text.split(['x', 'X']).collect();
                let pair = |s: &str| -> Option<[usize; 2]> {
                    let (a, b) = s.split_once('-')?;
                    Some([a.trim().parse().ok()?, b.trim().parse().ok()?])
                };
                match parts.as_slice() {
                    [l, r] => match (pair(l), pair(r)) {
                        (Some([a, b]), Some([c, d])) => vec![a, b, c, d],
                        _ => bail!("{text:?} is not \"a-b x c-d\""),
                    },
                    _ => bail!("{text:?} is not \"a-b x c-d\""),
                }
            }
        };
        ensure!(
            idx.iter().all(|i| (1..=n).contains(i)),
            "{self:?} names a point outside 1..={n}"
        );
        Ok(idx)
    }

    /// The point, from points in saved order (references are checked by [`ArenaDef::check`]).
    pub fn resolve(&self, pts: &[Pt]) -> Pt {
        let idx = self.indices(pts.len()).unwrap_or_default();
        match idx.as_slice() {
            [i] => pts[i - 1],
            [a, b, c, d] => cross(pts[a - 1], pts[b - 1], pts[c - 1], pts[d - 1]),
            _ => [f64::NAN, f64::NAN],
        }
    }
}

/// Where the line a–b crosses the line c–d; the four points' mean when they are parallel.
fn cross(a: Pt, b: Pt, c: Pt, d: Pt) -> Pt {
    let (u, v) = ([b[0] - a[0], b[1] - a[1]], [d[0] - c[0], d[1] - c[1]]);
    let det = u[0] * -v[1] - u[1] * -v[0];
    if det.abs() < 1e-9 {
        return [
            (a[0] + b[0] + c[0] + d[0]) / 4.0,
            (a[1] + b[1] + c[1] + d[1]) / 4.0,
        ];
    }
    let s = ((c[0] - a[0]) * -v[1] - (c[1] - a[1]) * -v[0]) / det;
    [a[0] + s * u[0], a[1] + s * u[1]]
}

impl ArenaDef {
    pub fn check(&self) -> Result<()> {
        let n = self.points.len();
        ensure!(n >= 3, "arena {}: {n} points, at least 3 needed", self.name);
        ensure!(
            self.floor.is_empty() || self.floor.len() == 4,
            "arena {}: the floor needs 4 points",
            self.name
        );
        ensure!(
            self.input_size[0] % 32 == 0 && self.input_size[1] % 32 == 0,
            "arena {}: input {:?} is not a multiple of 32",
            self.name,
            self.input_size
        );
        for r in self
            .floor
            .iter()
            .chain(self.zones.iter().flat_map(|z| &z.polygon))
        {
            r.indices(n)
                .with_context(|| format!("arena {}", self.name))?;
        }
        for z in &self.zones {
            ensure!(
                z.polygon.len() >= 3,
                "arena {}: zone {} has under 3 points",
                self.name,
                z.name
            );
        }
        Ok(())
    }

    pub fn floor(&self, pts: &[Pt]) -> Vec<Pt> {
        self.floor.iter().map(|r| r.resolve(pts)).collect()
    }

    pub fn zones(&self, pts: &[Pt]) -> Vec<Zone> {
        self.zones
            .iter()
            .map(|z| Zone {
                name: z.name.clone(),
                kind: z.kind.clone(),
                polygon: Polygon(z.polygon.iter().map(|r| r.resolve(pts)).collect()),
            })
            .collect()
    }
}

/// The arena found in a video: each point's median position and confidence over the frames, and
/// how far it moved between them (median distance from the median, frame pixels).
#[derive(Clone, Debug)]
pub struct Detection {
    pub points: Vec<Pt>,
    pub conf: Vec<f64>,
    pub spread: Vec<f64>,
    pub frames: usize,
}

pub struct ArenaModel {
    pub def: ArenaDef,
    model: Model,
    stride: usize,
    /// The recorded GPU graph: not tried yet, failed (`Some(None)`), or ready.
    #[cfg(all(feature = "mps", target_os = "macos"))]
    gpu: std::sync::Mutex<Option<Option<crate::tensor::mpsgraph::GpuForward>>>,
}

impl ArenaModel {
    pub fn load(path: &Path) -> Result<Self> {
        let (cfg, kind, weights) =
            models::load(path).with_context(|| format!("cannot read {}", path.display()))?;
        ensure!(
            kind == ModelKind::Arena,
            "{} holds a {kind} network, not an arena detector",
            path.display()
        );
        let def: ArenaDef = serde_json::from_value(
            cfg.extra
                .get("arena")
                .cloned()
                .with_context(|| format!("{} carries no arena definition", path.display()))?,
        )
        .with_context(|| format!("the arena definition in {} is not valid", path.display()))?;
        def.check()?;
        ensure!(
            def.points.len() == cfg.num_keypoints(),
            "{}: {} arena points but {} network outputs",
            path.display(),
            def.points.len(),
            cfg.num_keypoints()
        );
        let model = Model::build(ModelKind::Arena, &cfg, &weights)
            .with_context(|| format!("cannot build the arena network of {}", path.display()))?;
        let Model::Pose(net) = &model else {
            bail!("an arena network is a pose network");
        };
        let stride = net.stride();
        Ok(Self {
            def,
            model,
            stride,
            #[cfg(all(feature = "mps", target_os = "macos"))]
            gpu: std::sync::Mutex::new(None),
        })
    }

    /// The arena points in one grey frame (frame pixels) and their confidences. The frame is
    /// scaled to the input size as the trainer does (`cv2.warpAffine` with a pure scale: input
    /// pixel `(x, y)` samples frame `(x·sx, y·sy)`), as RGB in [0, 1].
    pub fn locate(&self, frame: &Image) -> Result<(Vec<Pt>, Vec<f64>)> {
        let [w, h] = self.def.input_size;
        let (fw, fh) = (frame.width as usize, frame.height as usize);
        ensure!(fw > 1 && fh > 1, "the frame is empty");
        let (sx, sy) = (fw as f64 / w as f64, fh as f64 / h as f64);
        let mut plane = vec![0f32; w * h];
        for y in 0..h {
            let fy = (y as f64 * sy).min((fh - 1) as f64);
            let (y0, ty) = (fy.floor() as usize, (fy - fy.floor()) as f32);
            let y1 = (y0 + 1).min(fh - 1);
            for x in 0..w {
                let fx = (x as f64 * sx).min((fw - 1) as f64);
                let (x0, tx) = (fx.floor() as usize, (fx - fx.floor()) as f32);
                let x1 = (x0 + 1).min(fw - 1);
                let p = |yy: usize, xx: usize| f32::from(frame.pixels[yy * fw + xx]);
                let top = p(y0, x0) * (1.0 - tx) + p(y0, x1) * tx;
                let bottom = p(y1, x0) * (1.0 - tx) + p(y1, x1) * tx;
                plane[y * w + x] = (top * (1.0 - ty) + bottom * ty) / 255.0;
            }
        }
        let input: Vec<f32> = plane.iter().chain(&plane).chain(&plane).copied().collect();
        let (heat, offset) = self.forward(input)?;
        let k = self.def.points.len();
        let maps = HeatMaps::new(&heat, &offset, k, w / self.stride, h / self.stride)?;
        let decoded = pose_decode(&maps, self.stride as f64);
        let points = decoded
            .keypoints
            .iter()
            .map(|p| [p[0] * sx, p[1] * sy])
            .collect();
        Ok((points, decoded.conf))
    }

    /// The network's `heat` and `offset` for one input. With the GPU active (`--features mps` on
    /// macOS), the first input runs on the CPU and records the graph, and later ones replay it on
    /// the GPU, as the pose pipeline does; a graph the GPU cannot take stays on the CPU.
    fn forward(&self, input: Vec<f32>) -> Result<(Vec<f32>, Vec<f32>)> {
        #[cfg(all(feature = "mps", target_os = "macos"))]
        if crate::tensor::gpu_active() {
            let mut gpu = self.gpu.lock().expect("the GPU lock is never poisoned");
            if let Some(Some(net)) = gpu.as_mut() {
                let mut out = net.run(&input)?;
                ensure!(
                    out.len() == 2,
                    "the GPU arena graph returned {} outputs",
                    out.len()
                );
                let offset = out.pop().unwrap_or_default();
                return Ok((out.pop().unwrap_or_default(), offset));
            }
            let (g, x, ids) = self.cpu_graph(input)?;
            let values = (g.to_vec(ids[0]), g.to_vec(ids[1]));
            if gpu.is_none() {
                *gpu = Some(
                    crate::tensor::mpsgraph::GpuForward::capture(&g, x, &ids)
                        .map_err(|err| eprintln!("arena detector stays on the CPU: {err:#}"))
                        .ok(),
                );
            }
            return Ok(values);
        }
        let (g, _, ids) = self.cpu_graph(input)?;
        Ok((g.to_vec(ids[0]), g.to_vec(ids[1])))
    }

    /// The forward pass on the CPU: the graph, its input and its `[heat, offset]` outputs.
    fn cpu_graph(&self, input: Vec<f32>) -> Result<(Graph, NodeId, [NodeId; 2])> {
        let [w, h] = self.def.input_size;
        let mut g = Graph::new();
        let x = g.from_data(input, &[1, 3, h, w]);
        let out = self.model.forward_named(&mut Ctx::eval(), &mut g, &[x])?;
        let get = |name: &str| -> Result<NodeId> {
            out.iter()
                .find(|(n, _)| *n == name)
                .map(|(_, id)| *id)
                .with_context(|| format!("the arena network has no {name:?} output"))
        };
        let ids = [get("heat")?, get("offset")?];
        Ok((g, x, ids))
    }

    /// The arena over several frames of one video.
    pub fn detect(&self, frames: &[Image]) -> Result<Detection> {
        ensure!(!frames.is_empty(), "no frames to find the arena in");
        let found: Vec<(Vec<Pt>, Vec<f64>)> = frames
            .iter()
            .map(|f| self.locate(f))
            .collect::<Result<_>>()?;
        let k = self.def.points.len();
        let median = |v: Vec<f64>| super::features::median(&v).unwrap_or(f64::NAN);
        let mut det = Detection {
            points: Vec::with_capacity(k),
            conf: Vec::with_capacity(k),
            spread: Vec::with_capacity(k),
            frames: frames.len(),
        };
        for i in 0..k {
            let p = [
                median(found.iter().map(|f| f.0[i][0]).collect()),
                median(found.iter().map(|f| f.0[i][1]).collect()),
            ];
            det.points.push(p);
            det.conf
                .push(median(found.iter().map(|f| f.1[i]).collect()));
            det.spread.push(median(
                found
                    .iter()
                    .map(|f| (f.0[i][0] - p[0]).hypot(f.0[i][1] - p[1]))
                    .collect(),
            ));
        }
        Ok(det)
    }

    /// The layout of a detection: floor and zones from the definition, checked; open and closed
    /// arms confirmed against the walls in `background` (the median of the frames).
    pub fn layout(&self, det: &Detection, rig: &str, background: &Image, maze: &Maze) -> Layout {
        let (w, h) = (f64::from(background.width), f64::from(background.height));
        let mut doubts = Vec::new();
        for (i, name) in self.def.points.iter().enumerate() {
            if det.conf[i] < MIN_CONF {
                doubts.push(format!("{name} found with confidence {:.2}", det.conf[i]));
            }
            if det.spread[i] > MAX_SPREAD * w {
                doubts.push(format!(
                    "{name} moves {:.0} px between frames",
                    det.spread[i]
                ));
            }
            let p = det.points[i];
            let margin = 0.02 * w;
            if !(-margin..=w + margin).contains(&p[0]) || !(-margin..=h + margin).contains(&p[1]) {
                doubts.push(format!("{name} is outside the image"));
            }
        }
        let floor = self.def.floor(&det.points);
        if floor.len() == 4 && !convex(&floor, 0.02 * w * h) {
            doubts.push("the floor is not a convex quadrilateral".into());
        }
        let mut zones = self.def.zones(&det.points);
        let mut wall_scores = std::collections::BTreeMap::new();
        if let Some(note) = confirm_arms(&mut zones, background, maze, &mut wall_scores) {
            doubts.push(note);
        }
        Layout {
            rig: rig.to_owned(),
            method: 0,
            source: "model".into(),
            status: if doubts.is_empty() {
                "ok"
            } else {
                "needs_review"
            }
            .into(),
            reason: doubts.join("; "),
            image_size: [background.width, background.height],
            floor,
            zones,
            wall_scores,
            finder: format!("arena model {} ({} frames)", self.def.name, det.frames),
        }
    }
}

/// A convex quadrilateral of at least `min_area`.
fn convex(q: &[Pt], min_area: f64) -> bool {
    let crosses: Vec<f64> = (0..4)
        .map(|i| {
            let (a, b, c) = (q[i], q[(i + 1) % 4], q[(i + 2) % 4]);
            (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        })
        .collect();
    let area = 0.5
        * (0..4)
            .map(|i| q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1])
            .sum::<f64>()
            .abs();
    (crosses.iter().all(|c| *c > 0.0) || crosses.iter().all(|c| *c < 0.0)) && area >= min_area
}

/// Open and closed arms (zones of kind `open_arm` / `closed_arm`, each `[inner, end, end, inner]`
/// around a `center` zone) checked against the walls in the video: closed arms show wall edges
/// along them. When the arms the definition calls open have the stronger walls by
/// `maze.wall_ratio`, the kinds are swapped; when neither side wins, the definition's kinds stay
/// and the returned note says the walls were unclear.
fn confirm_arms(
    zones: &mut [Zone],
    background: &Image,
    maze: &Maze,
    scores: &mut std::collections::BTreeMap<String, f64>,
) -> Option<String> {
    let arms: Vec<usize> = (0..zones.len())
        .filter(|i| matches!(zones[*i].kind.as_str(), "open_arm" | "closed_arm"))
        .collect();
    let center = zones.iter().find(|z| z.kind == "center")?;
    if arms.is_empty() || center.polygon.0.is_empty() {
        return None;
    }
    let n = center.polygon.0.len() as f64;
    let c = [
        center.polygon.0.iter().map(|p| p[0]).sum::<f64>() / n,
        center.polygon.0.iter().map(|p| p[1]).sum::<f64>() / n,
    ];
    let enhanced = super::maze::blur5(&super::maze::clahe(
        background,
        maze.clahe_clip,
        maze.clahe_tiles,
    ));
    let values: Vec<f64> = enhanced.pixels.iter().map(|p| f64::from(*p)).collect();
    let (gx, gy) = super::maze::sobel(&values, enhanced.width as usize, enhanced.height as usize);
    let (mut open, mut closed) = (Vec::new(), Vec::new());
    for &i in &arms {
        let p = &zones[i].polygon.0;
        if p.len() != 4 {
            return Some(format!(
                "arm {} is not a quadrilateral; its kind is unchecked",
                zones[i].name
            ));
        }
        let end = [(p[1][0] + p[2][0]) / 2.0, (p[1][1] + p[2][1]) / 2.0];
        let inner = [(p[0][0] + p[3][0]) / 2.0, (p[0][1] + p[3][1]) / 2.0];
        let len = (end[0] - c[0]).hypot(end[1] - c[1]).max(1e-9);
        let along = [(end[0] - c[0]) / len, (end[1] - c[1]) / len];
        let from = (inner[0] - c[0]) * along[0] + (inner[1] - c[1]) * along[1];
        let width = (p[1][0] - p[2][0]).hypot(p[1][1] - p[2][1]);
        let s = super::maze::wall_score(&enhanced, &gx, &gy, c, along, from, len, width);
        scores.insert(zones[i].name.clone(), s);
        if zones[i].kind == "closed_arm" {
            closed.push(s);
        } else {
            open.push(s);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let (o, cl) = (mean(&open), mean(&closed));
    let evidence = o.max(cl);
    if evidence.is_nan() || evidence <= 0.0 {
        Some("no wall evidence along the arms; arms as defined".into())
    } else if cl >= maze.wall_ratio * o {
        None
    } else if o >= maze.wall_ratio * cl {
        for &i in &arms {
            zones[i].kind = if zones[i].kind == "open_arm" {
                "closed_arm"
            } else {
                "open_arm"
            }
            .into();
        }
        None
    } else {
        Some(format!(
            "the walls do not tell open from closed arms (scores {cl:.2} closed vs {o:.2} open); arms as defined"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plus_def() -> ArenaDef {
        serde_json::from_str(
            r#"{"name": "plus_maze", "points": ["1","2","3","4","5","6","7","8"],
                "input_size": [384, 288], "floor": [],
                "zones": [
                  {"name": "center", "kind": "center", "polygon": ["2-5 x 8-3", "2-5 x 7-4", "1-6 x 7-4", "1-6 x 8-3"]},
                  {"name": "arm_north", "kind": "closed_arm", "polygon": ["1-6 x 8-3", 1, 2, "2-5 x 8-3"]}
                ]}"#,
        )
        .unwrap()
    }

    #[test]
    fn point_references_resolve_and_are_checked() {
        let def = plus_def();
        def.check().unwrap();
        let pts = [
            [250.0, 70.0],
            [275.0, 70.0],
            [420.0, 235.0],
            [420.0, 262.0],
            [275.0, 420.0],
            [250.0, 420.0],
            [112.0, 262.0],
            [112.0, 235.0],
        ];
        let zones = def.zones(&pts);
        assert_eq!(
            zones[0].polygon.0,
            vec![
                [275.0, 235.0],
                [275.0, 262.0],
                [250.0, 262.0],
                [250.0, 235.0]
            ]
        );
        assert_eq!(zones[1].polygon.0[1], [250.0, 70.0]);
        let mut bad = def.clone();
        bad.zones[0].polygon[0] = PointRef::Cross("2-9 x 8-3".into());
        assert!(bad.check().is_err());
        bad.zones[0].polygon[0] = PointRef::Cross("2 x 8".into());
        assert!(bad.check().is_err());
    }

    /// Prints the points `ARENA_MODEL` finds in the grey image `ARENA_FRAME`, for comparing with
    /// the Python network (`scripts/train_arena.py` decodes the same way).
    #[test]
    #[ignore = "needs a trained arena model and a frame"]
    fn real_frame() {
        let net = ArenaModel::load(Path::new(&std::env::var("ARENA_MODEL").unwrap())).unwrap();
        let img = image::open(std::env::var("ARENA_FRAME").unwrap())
            .unwrap()
            .to_luma8();
        let frame = Image {
            width: img.width(),
            height: img.height(),
            pixels: img.into_raw(),
        };
        crate::tensor::set_gpu(true);
        let (pts, conf) = net.locate(&frame).unwrap();
        for ((name, p), c) in net.def.points.iter().zip(&pts).zip(&conf) {
            println!("{name}: {:.2} {:.2} conf {c:.3}", p[0], p[1]);
        }
        // Again: with the GPU active the second pass replays the recorded graph there.
        let (again, _) = net.locate(&frame).unwrap();
        let worst = pts
            .iter()
            .zip(&again)
            .map(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]))
            .fold(0.0, f64::max);
        println!(
            "second pass (GPU: {}) differs by at most {worst:.4} px",
            crate::tensor::gpu_active()
        );
    }

    #[test]
    fn a_floor_must_be_a_convex_quadrilateral() {
        let square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert!(convex(&square, 50.0));
        assert!(!convex(&square, 500.0));
        assert!(!convex(
            &[[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]],
            1.0
        ));
    }
}
