//! Converters from the public rodent datasets to canonical COCO keypoint JSON — the port of
//! `src/animalpose/data/real.py` (and the dataset tables of `skeleton/species.py` they use).
//!
//! | Python | Rust |
//! | --- | --- |
//! | `CocoWriter` | [`CocoWriter`] |
//! | `_box` / `_canonical` | [`keypoint_box`] / [`canonical`] |
//! | `convert_mirror_mouse` | [`convert_mirror_mouse`] |
//! | `_lp_crim13_mapping` / `convert_lp_crim13` | [`lp_crim13_mapping`] / [`convert_lp_crim13`] |
//! | `detect_mapping` / `convert_coco_keypoints` | [`detect_mapping`] / [`convert_coco_keypoints`] |
//! | `skeleton/species.py:DATASET_MAPPINGS` | [`dataset_mappings`] |
//! | `pd.read_csv(path, header=[0, 1, 2], index_col=0)` (Lightning Pose label files) | [`LpTable::read`] |
//! | `rat7m_cameras` / `rat7m_visibility` / `convert_rat7m` | [`rat7m_cameras`] / [`rat7m_visibility`] / [`convert_rat7m`] |
//! | `multiview/camera.py:Camera.project` / `in_image` / `center` / `to_dict` (what Rat 7M needs) | [`Rat7mCamera`] |
//!
//! `// differs from Python:`
//! * Images are written with the `image` crate's JPEG encoder at the same quality as
//!   `cv2.imwrite(..., IMWRITE_JPEG_QUALITY)`; the pixels of two JPEG encoders differ slightly, the
//!   JSON (names, sizes, keypoints, boxes) does not.
//! * Rat 7M video chunks are decoded through [`crate::data::images::read_frames`] (every `every`-th
//!   frame), which hands back grey frames: the JPEGs are grey where `cv2.VideoCapture` keeps colour.
//!   The MATLAB mocap is read by [`crate::data::mat5`].

use crate::algos::image::Bgr;
use crate::data::dlc::read_csv;
use crate::data::mat5::{read_mat, MatValue};
use crate::data::pyjson::{loads, PyVal};
use crate::data::synthetic::{index, CANONICAL_NAMES, NUM_KEYPOINTS};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// `real.py:MIRROR_MOUSE`: Lightning Pose mirror-mouse names (suffix stripped) to canonical ones.
pub const MIRROR_MOUSE: [(&str, &str); 7] = [
    ("paw1LH", "left_hind_paw"),
    ("paw2LF", "left_front_paw"),
    ("paw3RF", "right_front_paw"),
    ("paw4RH", "right_hind_paw"),
    ("tailBase", "tail_base"),
    ("tailMid", "tail_mid"),
    ("nose", "nose"),
];
/// `real.py:MIRROR_SPLIT_Y`: the side view is above this row, the mirrored bottom view below.
pub const MIRROR_SPLIT_Y: u32 = 160;

/// `skeleton/species.py:DATASET_MAPPINGS`, every registered dataset in registration order.
pub fn dataset_mappings() -> Vec<(&'static str, Vec<(&'static str, &'static str)>)> {
    let synthetic: Vec<(&str, &str)> = CANONICAL_NAMES.iter().map(|n| (*n, *n)).collect();
    vec![
        ("synthetic", synthetic),
        (
            "superanimal_topviewmouse",
            vec![
                ("nose", "nose"),
                ("left_eye", "left_eye"),
                ("right_eye", "right_eye"),
                ("left_ear", "left_ear"),
                ("right_ear", "right_ear"),
                ("head_midpoint", "head_center"),
                ("neck", "neck"),
                ("mid_back", "spine_upper"),
                ("mouse_center", "spine_mid"),
                ("mid_backend2", "spine_lower"),
                ("tail_base", "tail_base"),
                ("tail3", "tail_mid"),
                ("tail_end", "tail_end"),
                ("left_shoulder", "left_shoulder"),
                ("right_shoulder", "right_shoulder"),
                ("left_hip", "left_hip"),
                ("right_hip", "right_hip"),
            ],
        ),
        (
            "superanimal_quadruped",
            vec![
                ("nose", "nose"),
                ("left_eye", "left_eye"),
                ("right_eye", "right_eye"),
                ("left_earbase", "left_ear"),
                ("right_earbase", "right_ear"),
                ("neck_base", "neck"),
                ("back_base", "spine_upper"),
                ("back_middle", "spine_mid"),
                ("back_end", "spine_lower"),
                ("tail_base", "tail_base"),
                ("tail_end", "tail_end"),
                ("front_left_thai", "left_shoulder"),
                ("front_right_thai", "right_shoulder"),
                ("front_left_knee", "left_front_knee"),
                ("front_right_knee", "right_front_knee"),
                ("front_left_paw", "left_front_paw"),
                ("front_right_paw", "right_front_paw"),
                ("back_left_thai", "left_hip"),
                ("back_right_thai", "right_hip"),
                ("back_left_knee", "left_hind_knee"),
                ("back_right_knee", "right_hind_knee"),
                ("back_left_paw", "left_hind_paw"),
                ("back_right_paw", "right_hind_paw"),
            ],
        ),
        (
            "ap10k",
            vec![
                ("Left Eye", "left_eye"),
                ("Right Eye", "right_eye"),
                ("Nose", "nose"),
                ("Neck", "neck"),
                ("Root of Tail", "tail_base"),
                ("Left Shoulder", "left_shoulder"),
                ("Right Shoulder", "right_shoulder"),
                ("Left Elbow", "left_front_knee"),
                ("Right Elbow", "right_front_knee"),
                ("Left Front Paw", "left_front_paw"),
                ("Right Front Paw", "right_front_paw"),
                ("Left Hip", "left_hip"),
                ("Right Hip", "right_hip"),
                ("Left Knee", "left_hind_knee"),
                ("Right Knee", "right_hind_knee"),
                ("Left Back Paw", "left_hind_paw"),
                ("Right Back Paw", "right_hind_paw"),
            ],
        ),
        (
            "calms21",
            vec![
                ("nose", "nose"),
                ("left_ear", "left_ear"),
                ("right_ear", "right_ear"),
                ("neck", "neck"),
                ("left_hip", "left_hip"),
                ("right_hip", "right_hip"),
                ("tail_base", "tail_base"),
            ],
        ),
        (
            "rat7m",
            vec![
                ("HeadB", "head_center"),
                ("SpineF", "spine_upper"),
                ("SpineM", "spine_mid"),
                ("SpineL", "spine_lower"),
                ("ShoulderL", "left_shoulder"),
                ("ShoulderR", "right_shoulder"),
                ("ElbowL", "left_front_knee"),
                ("ElbowR", "right_front_knee"),
                ("HipL", "left_hip"),
                ("HipR", "right_hip"),
                ("KneeL", "left_hind_knee"),
                ("KneeR", "right_hind_knee"),
            ],
        ),
        ("lp_mirror_mouse", MIRROR_MOUSE.to_vec()),
        (
            "lp_crim13",
            vec![
                ("nose", "nose"),
                ("left_ear", "left_ear"),
                ("right_ear", "right_ear"),
                ("top_of_neck", "neck"),
                ("left_rear_knee", "left_hind_knee"),
                ("right_rear_knee", "right_hind_knee"),
                ("base_of_tail", "tail_base"),
            ],
        ),
        (
            "homecage_dlc",
            vec![
                ("nose", "nose"),
                ("leftear", "left_ear"),
                ("rightear", "right_ear"),
                ("headstage", "head_center"),
                ("midbody", "spine_mid"),
                ("tailbase", "tail_base"),
            ],
        ),
    ]
}

/// One animal handed to [`CocoWriter::add`].
#[derive(Clone, Debug)]
pub struct Animal {
    /// `(K, 2)` canonical keypoints, NaN unlabeled.
    pub keypoints: Vec<[f64; 2]>,
    /// `(K,)` visibility 0/1/2.
    pub visibility: Vec<i64>,
    /// xyxy box.
    pub bbox: [f64; 4],
    /// Track id (`-1` when the dataset has none).
    pub track_id: i64,
}

/// `CocoWriter`: images and annotations accumulated, then saved with the canonical category.
#[derive(Clone, Debug, Default)]
pub struct CocoWriter {
    images: Vec<PyVal>,
    annotations: Vec<PyVal>,
}

impl CocoWriter {
    /// An empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// `add(file_name, size_hw, animals)`.
    pub fn add(&mut self, file_name: &str, size_hw: (i64, i64), animals: &[Animal]) {
        let img_id = self.images.len();
        self.images.push(
            PyVal::dict()
                .with("id", img_id)
                .with("file_name", file_name)
                .with("height", size_hw.0)
                .with("width", size_hw.1),
        );
        for a in animals {
            let mut flat = Vec::with_capacity(a.keypoints.len() * 3);
            for (p, v) in a.keypoints.iter().zip(&a.visibility) {
                if *v > 0 && p[0].is_finite() && p[1].is_finite() {
                    flat.extend([PyVal::Float(p[0]), PyVal::Float(p[1]), PyVal::Int(*v)]);
                } else {
                    flat.extend([PyVal::Float(0.0), PyVal::Float(0.0), PyVal::Int(0)]);
                }
            }
            let [x0, y0, x1, y1] = a.bbox;
            self.annotations.push(
                PyVal::dict()
                    .with("id", self.annotations.len())
                    .with("image_id", img_id)
                    .with("category_id", 1i64)
                    .with("keypoints", PyVal::List(flat))
                    .with("num_keypoints", a.visibility.iter().filter(|v| **v > 0).count())
                    .with("bbox", vec![x0, y0, x1 - x0, y1 - y0])
                    .with("area", (x1 - x0) * (y1 - y0))
                    .with("iscrowd", 0i64)
                    .with("track_id", a.track_id),
            );
        }
    }

    /// The COCO dict `save` writes.
    pub fn to_py(&self) -> PyVal {
        let category = PyVal::dict()
            .with("id", 1i64)
            .with("name", "animal")
            .with("keypoints", CANONICAL_NAMES.to_vec())
            .with("skeleton", PyVal::List(Vec::new()));
        PyVal::dict()
            .with("images", PyVal::List(self.images.clone()))
            .with("annotations", PyVal::List(self.annotations.clone()))
            .with("categories", PyVal::List(vec![category]))
    }

    /// `save(path)`: writes `json.dumps(coco)` and returns `{"images": n, "animals": n}`.
    pub fn save(&self, path: &Path) -> Result<PyVal> {
        std::fs::write(path, self.to_py().dumps())
            .with_context(|| format!("cannot write {}", path.display()))?;
        Ok(PyVal::dict()
            .with("images", self.images.len())
            .with("animals", self.annotations.len()))
    }
}

/// `_box(kp, pad, size_hw, min_side)`: the labelled points' box grown by `pad` of its extent (at
/// least `min_side`), clipped to the image; `None` with fewer than two labelled points.
pub fn keypoint_box(kp: &[[f64; 2]], pad: f64, size_hw: (f64, f64), min_side: f64) -> Option<[f64; 4]> {
    let ok: Vec<&[f64; 2]> = kp.iter().filter(|p| p[0].is_finite() && p[1].is_finite()).collect();
    if ok.len() < 2 {
        return None;
    }
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in ok {
        for d in 0..2 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    let dd = [(hi[0] - lo[0]).max(min_side) * pad, (hi[1] - lo[1]).max(min_side) * pad];
    let (h, w) = size_hw;
    Some([
        (lo[0] - dd[0]).max(0.0),
        (lo[1] - dd[1]).max(0.0),
        w.min(hi[0] + dd[0]),
        h.min(hi[1] + dd[1]),
    ])
}

/// `_canonical(names, xy, mapping)`: dataset points placed into the canonical order.
pub fn canonical(names: &[&str], xy: &[[f64; 2]], mapping: &[(&str, &str)]) -> Vec<[f64; 2]> {
    let mut out = vec![[f64::NAN; 2]; NUM_KEYPOINTS];
    for (n, p) in names.iter().zip(xy) {
        if let Some((_, c)) = mapping.iter().find(|(k, _)| k == n) {
            out[index(c)] = *p;
        }
    }
    out
}

fn visibility(kp: &[[f64; 2]]) -> Vec<i64> {
    kp.iter()
        .map(|p| if p[0].is_finite() && p[1].is_finite() { 2 } else { 0 })
        .collect()
}

/// A Lightning Pose / DeepLabCut label table (`scorer` / `bodyparts` / `coords` header rows, the
/// image path in the first column), the subset of `pd.read_csv(header=[0, 1, 2], index_col=0)`
/// the converters use.
#[derive(Clone, Debug)]
pub struct LpTable {
    /// The scorer of the first column (`df.columns[0][0]`).
    pub scorer: String,
    /// `(scorer, bodypart, coord)` per value column.
    pub columns: Vec<(String, String, String)>,
    /// `(index, values)` per row; empty cells are NaN.
    pub rows: Vec<(String, Vec<f64>)>,
}

impl LpTable {
    /// Reads a label CSV.
    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("[Errno 2] No such file or directory: '{}'", path.display()))?;
        let rows = read_csv(&text);
        if rows.len() < 3 {
            bail!("{} has fewer than three header rows", path.display());
        }
        let width = rows[0].len();
        let col = |r: usize, c: usize| rows[r].get(c).cloned().unwrap_or_default();
        let columns: Vec<(String, String, String)> =
            (1..width).map(|c| (col(0, c), col(1, c), col(2, c))).collect();
        let mut body = Vec::new();
        for (i, r) in rows.iter().enumerate().skip(3) {
            if r.is_empty() {
                continue;
            }
            // pandas reads a row of index names (`file_name,,,`) right under the header.
            if i == 3 && r.iter().skip(1).all(String::is_empty) {
                continue;
            }
            let values = (1..width)
                .map(|c| {
                    let v = r.get(c).map_or("", |s| s.trim());
                    if v.is_empty() {
                        Ok(f64::NAN)
                    } else {
                        v.parse::<f64>()
                            .with_context(|| format!("{}: bad number {v:?}", path.display()))
                    }
                })
                .collect::<Result<Vec<f64>>>()?;
            body.push((r[0].clone(), values));
        }
        Ok(Self {
            scorer: columns.first().map(|c| c.0.clone()).unwrap_or_default(),
            columns,
            rows: body,
        })
    }

    /// The column of `(scorer, bodypart, coord)`.
    pub fn column(&self, bodypart: &str, coord: &str) -> Result<usize> {
        self.columns
            .iter()
            .position(|(s, b, c)| *s == self.scorer && b == bodypart && c == coord)
            .with_context(|| format!("({:?}, {bodypart:?}, {coord:?})", self.scorer))
    }

    /// The bodyparts in column order, each once (`dict.fromkeys(columns.get_level_values(1))`).
    pub fn bodyparts(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (_, b, _) in &self.columns {
            if !out.contains(b) {
                out.push(b.clone());
            }
        }
        out
    }
}

/// `cv2.imwrite(path, img, [IMWRITE_JPEG_QUALITY, quality])`.
pub fn write_jpeg(path: &Path, img: &Bgr, quality: u8) -> Result<()> {
    let rgb: Vec<u8> = img
        .pixels
        .chunks_exact(3)
        .flat_map(|p| [p[2], p[1], p[0]])
        .collect();
    let file = std::fs::File::create(path).with_context(|| format!("cannot write {}", path.display()))?;
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(file), quality);
    enc.encode(&rgb, img.width, img.height, image::ExtendedColorType::Rgb8)
        .with_context(|| format!("cannot encode {}", path.display()))
}

fn rows_of(img: &Bgr, y0: u32, y1: u32) -> Bgr {
    let (y0, y1) = (y0.min(img.height), y1.min(img.height));
    let w = img.width as usize;
    Bgr {
        width: img.width,
        height: y1 - y0,
        pixels: img.pixels[y0 as usize * w * 3..y1 as usize * w * 3].to_vec(),
    }
}

/// `convert_mirror_mouse(root, out)`: every labelled frame split into its side and (mirrored)
/// bottom view, `labels_InD.csv` -> `*_train.json`, `labels_OOD.csv` -> `*_val.json`.
pub fn convert_mirror_mouse(root: &Path, out: &Path) -> Result<PyVal> {
    std::fs::create_dir_all(out.join("images"))?;
    let mut stats = PyVal::dict();
    let names: Vec<&str> = MIRROR_MOUSE.iter().map(|(k, _)| *k).collect();
    for (split, csv) in [("train", "labels_InD.csv"), ("val", "labels_OOD.csv")] {
        let df = LpTable::read(&root.join(csv))?;
        let mut writers = [("side", CocoWriter::new()), ("bottom", CocoWriter::new())];
        for (rel, row) in &df.rows {
            let Ok(img) = crate::data::measure::read_bgr(&root.join(rel)) else {
                continue;
            };
            let views = [
                ("side", "_top", 0, MIRROR_SPLIT_Y),
                ("bottom", "_bot", MIRROR_SPLIT_Y, img.height),
            ];
            for (v, (view, suffix, y0, y1)) in views.into_iter().enumerate() {
                let mut xy = Vec::new();
                for bp in &names {
                    let x = row[df.column(&format!("{bp}{suffix}"), "x")?];
                    let y = row[df.column(&format!("{bp}{suffix}"), "y")?];
                    xy.push([x, y - f64::from(y0)]);
                }
                let kp = canonical(&names, &xy, &MIRROR_MOUSE);
                let part = rows_of(&img, y0, y1);
                let size = (f64::from(part.height), f64::from(part.width));
                let Some(bbox) = keypoint_box(&kp, 0.25, size, 16.0) else {
                    continue;
                };
                let name = format!("{view}/{}", rel.replace('/', "_").replace(".png", ".jpg"));
                std::fs::create_dir_all(out.join("images").join(view))?;
                write_jpeg(&out.join("images").join(&name), &part, 95)?;
                let vis = visibility(&kp);
                writers[v].1.add(
                    &name,
                    (i64::from(part.height), i64::from(part.width)),
                    &[Animal {
                        keypoints: kp,
                        visibility: vis,
                        bbox,
                        track_id: -1,
                    }],
                );
            }
        }
        for (view, w) in &writers {
            stats.set(
                format!("{view}_{split}"),
                w.save(&out.join(format!("{view}_{split}.json")))?,
            );
        }
    }
    Ok(stats)
}

/// `_lp_crim13_mapping(bodypart)`: `black_mouse_nose`-style names to `(animal, canonical name)`.
pub fn lp_crim13_mapping(bodypart: &str) -> Option<(usize, &'static str)> {
    let (color, part) = if let Some(rest) = bodypart
        .strip_prefix("black_mouse_")
        .map(|r| ("black", r))
        .or_else(|| bodypart.strip_prefix("white_mouse_").map(|r| ("white", r)))
    {
        rest
    } else if let Some(p) = bodypart.strip_suffix("_black").filter(|p| !p.is_empty()) {
        ("black", p)
    } else if let Some(p) = bodypart.strip_suffix("_white").filter(|p| !p.is_empty()) {
        ("white", p)
    } else {
        return None;
    };
    if part.is_empty() {
        return None;
    }
    let canon = match part {
        "nose" => "nose",
        "right_ear" => "right_ear",
        "left_ear" => "left_ear",
        "top_of_neck" | "neck" => "neck",
        "right_rear_knee" => "right_hind_knee",
        "left_rear_knee" => "left_hind_knee",
        "base_tail" | "tail_base" | "base_of_tail" => "tail_base",
        "right_side" => "right_hip",
        "left_side" => "left_hip",
        _ => return None,
    };
    Some((usize::from(color != "black"), canon))
}

/// `convert_lp_crim13(root, out)`: two mice (black = track 0, white = 1) per labelled frame.
pub fn convert_lp_crim13(root: &Path, out: &Path) -> Result<PyVal> {
    std::fs::create_dir_all(out.join("images"))?;
    let mut stats = PyVal::dict();
    for (split, csv) in [("train", "labels_InD.csv"), ("val", "labels_OOD.csv")] {
        if !root.join(csv).exists() {
            continue;
        }
        let df = LpTable::read(&root.join(csv))?;
        let parts = df.bodyparts();
        let unmapped: Vec<String> = parts
            .iter()
            .filter(|p| lp_crim13_mapping(p).is_none())
            .cloned()
            .collect();
        let mut w = CocoWriter::new();
        for (rel, row) in &df.rows {
            let Ok(img) = crate::data::measure::read_bgr(&root.join(rel)) else {
                continue;
            };
            let size = (f64::from(img.height), f64::from(img.width));
            let session = Path::new(rel)
                .parent()
                .and_then(|p| p.file_name())
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let mut animals = Vec::new();
            for a in 0..2usize {
                let mut kp = vec![[f64::NAN; 2]; NUM_KEYPOINTS];
                for p in &parts {
                    if let Some((animal, c)) = lp_crim13_mapping(p) {
                        if animal == a {
                            kp[index(c)] = [row[df.column(p, "x")?], row[df.column(p, "y")?]];
                        }
                    }
                }
                if let Some(bbox) = keypoint_box(&kp, 0.3, size, 16.0) {
                    let vis = visibility(&kp);
                    animals.push(Animal {
                        keypoints: kp,
                        visibility: vis,
                        bbox,
                        track_id: i64::from(crate::data::measure::crc32(session.as_bytes()) % 100_000) * 16
                            + a as i64,
                    });
                }
            }
            let name = rel.replace('/', "_").replace(".png", ".jpg");
            write_jpeg(&out.join("images").join(&name), &img, 95)?;
            w.add(&name, (i64::from(img.height), i64::from(img.width)), &animals);
        }
        let mut s = w.save(&out.join(format!("{split}.json")))?;
        s.set("unmapped_bodyparts", unmapped);
        stats.set(split, s);
    }
    Ok(stats)
}

// ------------------------------------------------------------------------------------- Rat 7M

/// `real.py:_RAT7M_LATERAL`: markers on the left / right side of the body.
pub const RAT7M_LEFT: [&str; 7] = ["HeadL", "HipL", "ElbowL", "ArmL", "ShoulderL", "KneeL", "ShinL"];
/// The right-side markers.
pub const RAT7M_RIGHT: [&str; 6] = ["HipR", "ElbowR", "ArmR", "ShoulderR", "KneeR", "ShinR"];

/// The pinhole + Brown-Conrady camera of `multiview/camera.py:Camera`, as far as the Rat 7M
/// converter uses it: `x_cam = R X + t`, pixels `K distort(x_cam / z)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Rat7mCamera {
    /// Intrinsics.
    pub k: [[f64; 3]; 3],
    /// World-to-camera rotation.
    pub r: [[f64; 3]; 3],
    /// World-to-camera translation.
    pub t: [f64; 3],
    /// `k1, k2, p1, p2, k3`.
    pub dist: [f64; 5],
    /// `(width, height)`.
    pub size: (u32, u32),
    /// The camera's name.
    pub name: String,
}

impl Rat7mCamera {
    /// `Camera.center`: `-R^T t`.
    pub fn center(&self) -> [f64; 3] {
        let mut c = [0.0; 3];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = -(0..3).map(|j| self.r[j][i] * self.t[j]).sum::<f64>();
        }
        c
    }

    /// `Camera.project` of one world point.
    pub fn project(&self, x: [f64; 3]) -> [f64; 2] {
        let xc: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| x[j] * self.r[i][j]).sum::<f64>() + self.t[i])
            .collect();
        let z = if xc[2].abs() < 1e-9 { 1e-9 } else { xc[2] };
        let (x, y) = (xc[0] / z, xc[1] / z);
        let [k1, k2, p1, p2, k3] = self.dist;
        let r2 = x * x + y * y;
        let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        [
            xd * self.k[0][0] + yd * self.k[0][1] + self.k[0][2],
            xd * self.k[1][0] + yd * self.k[1][1] + self.k[1][2],
        ]
    }

    /// `Camera.in_image(uv)` (margin 0); NaN is outside.
    pub fn in_image(&self, uv: [f64; 2]) -> bool {
        let (w, h) = (f64::from(self.size.0), f64::from(self.size.1));
        uv[0] >= 0.0 && uv[0] < w && uv[1] >= 0.0 && uv[1] < h
    }

    /// `Camera.to_dict()`.
    pub fn to_py(&self) -> PyVal {
        let m = |a: &[[f64; 3]; 3]| PyVal::List(a.iter().map(|r| PyVal::from(r.to_vec())).collect());
        PyVal::dict()
            .with("name", self.name.as_str())
            .with("K", m(&self.k))
            .with("R", m(&self.r))
            .with("t", self.t.to_vec())
            .with("dist", self.dist.to_vec())
            .with("size", vec![i64::from(self.size.0), i64::from(self.size.1)])
    }
}

fn mat3(v: Option<&MatValue>, what: &str) -> Result<[[f64; 3]; 3]> {
    let rows = v.and_then(MatValue::rows).with_context(|| format!("{what} is missing"))?;
    let mut out = [[0.0; 3]; 3];
    for (i, row) in rows.iter().take(3).enumerate() {
        for (j, x) in row.iter().take(3).enumerate() {
            out[i][j] = *x;
        }
    }
    Ok(out)
}

/// `rat7m_cameras(mat)`: the MATLAB camera structs (row-vector convention, 1-based pixels) as
/// cameras, in the struct's field order.
pub fn rat7m_cameras(cameras: &MatValue) -> Result<Vec<Rat7mCamera>> {
    let mut out = Vec::new();
    for name in cameras.field_names() {
        let c = cameras.field(name).expect("listed field");
        let im = mat3(c.field("IntrinsicMatrix"), "IntrinsicMatrix")?;
        let mut k = [[0.0; 3]; 3];
        let rm = mat3(c.field("rotationMatrix"), "rotationMatrix")?;
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                k[i][j] = im[j][i];
                r[i][j] = rm[j][i];
            }
        }
        k[0][2] -= 1.0;
        k[1][2] -= 1.0;
        let vals = |f: &str| -> Vec<f64> { c.field(f).and_then(MatValue::values).map(<[f64]>::to_vec).unwrap_or_default() };
        let (rad, tan, tv) = (vals("RadialDistortion"), vals("TangentialDistortion"), vals("translationVector"));
        let at = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(0.0);
        out.push(Rat7mCamera {
            k,
            r,
            t: [at(&tv, 0), at(&tv, 1), at(&tv, 2)],
            dist: [at(&rad, 0), at(&rad, 1), at(&tan, 0), at(&tan, 1), at(&rad, 2)],
            size: (1328, 1048),
            name: name.clone(),
        });
    }
    Ok(out)
}

/// `rat7m_visibility(markers, cam)`: 2 for midline markers and lateral markers facing the camera,
/// 1 for lateral markers on the far side of the body.
pub fn rat7m_visibility(markers: &[(String, [f64; 3])], cam: &Rat7mCamera) -> Vec<i64> {
    let get = |n: &str| markers.iter().find(|(k, _)| k == n).map(|(_, x)| *x);
    let (Some(f), Some(l)) = (get("SpineF"), get("SpineL")) else {
        return vec![2; markers.len()];
    };
    let spine = [f[0] - l[0], f[1] - l[1], f[2] - l[2]];
    let left = [-spine[1], spine[0], 0.0]; // [0, 0, 1] x spine
    let usable = left.iter().all(|v| v.is_finite()) && left[0].hypot(left[1]) >= 1e-6;
    let c = cam.center();
    markers
        .iter()
        .map(|(name, x)| {
            let side = if RAT7M_LEFT.contains(&name.as_str()) {
                1.0
            } else if RAT7M_RIGHT.contains(&name.as_str()) {
                -1.0
            } else {
                0.0
            };
            if side == 0.0 || !usable {
                2
            } else {
                let dot: f64 = (0..3).map(|i| side * left[i] * (c[i] - x[i])).sum();
                if dot > 0.0 {
                    2
                } else {
                    1
                }
            }
        })
        .collect()
}

/// `camera<N>-<start>.mp4` -> `(N, start)`.
fn rat7m_video_name(name: &str) -> Option<(String, i64)> {
    let stem = name.strip_suffix(".mp4")?;
    let (head, start) = stem.rsplit_once('-')?;
    let start: i64 = start.parse().ok()?;
    let digits: String = head.chars().rev().take_while(char::is_ascii_digit).collect::<Vec<_>>().into_iter().rev().collect();
    if digits.is_empty() || !head[..head.len() - digits.len()].ends_with("camera") {
        return None;
    }
    Some((digits, start))
}

/// `convert_rat7m(mocap_mat, video_dir, out, every, val_starts)`: the mocap projected into every
/// camera's video chunks; `val_starts = None` puts every chunk in `val` (a held-out subject).
pub fn convert_rat7m(mocap_mat: &Path, video_dir: &Path, out: &Path, every: usize, val_starts: Option<&[i64]>) -> Result<PyVal> {
    let vars = read_mat(mocap_mat)?;
    let var = |n: &str| vars.iter().find(|(k, _)| k == n).map(|(_, v)| v);
    let cameras = var("cameras").context("the MAT-file has no `cameras`")?;
    let cams = rat7m_cameras(cameras)?;
    let mocap_v = var("mocap").context("the MAT-file has no `mocap`")?;
    let mocap: Vec<(String, Vec<Vec<f64>>)> = mocap_v
        .field_names()
        .iter()
        .map(|n| (n.clone(), mocap_v.field(n).and_then(MatValue::rows).unwrap_or_default()))
        .collect();
    let maps = dataset_mappings();
    let mapping = &maps.iter().find(|(k, _)| *k == "rat7m").expect("registered").1;
    std::fs::create_dir_all(out.join("images"))?;
    let mut writers = [("train", CocoWriter::new()), ("val", CocoWriter::new())];
    let mut videos: Vec<PathBuf> = std::fs::read_dir(video_dir)
        .with_context(|| format!("cannot list {}", video_dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "mp4"))
        .collect();
    videos.sort();
    for video in videos {
        let fname = video.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let (num, start) = rat7m_video_name(&fname).with_context(|| format!("{fname}: not a camera<N>-<start>.mp4 chunk"))?;
        let cam_name = format!("Camera{num}");
        let cam = cams.iter().find(|c| c.name == cam_name).with_context(|| format!("no {cam_name} in the mocap file"))?;
        let frame_of: Vec<f64> = cameras
            .field(&cam_name)
            .and_then(|c| c.field("frame"))
            .and_then(MatValue::values)
            .map(<[f64]>::to_vec)
            .unwrap_or_default();
        let mut first_index = std::collections::HashMap::new();
        for (i, f) in frame_of.iter().enumerate() {
            first_index.entry(*f as i64).or_insert(i);
        }
        let split = usize::from(val_starts.map_or(true, |v| v.contains(&start)));
        let (frames, _) = crate::data::images::read_frames(&video, None, every.max(1))?;
        let stem = video.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        for (i, frame) in frames.iter().enumerate() {
            let j = i * every.max(1);
            let Some(&idx) = first_index.get(&(start + j as i64)) else {
                continue;
            };
            let valid: Vec<(String, [f64; 3])> = mocap
                .iter()
                .filter_map(|(n, rows)| {
                    let r = rows.get(idx)?;
                    let x = [r[0], r[1], r[2]];
                    x.iter().all(|v| v.is_finite()).then(|| (n.clone(), x))
                })
                .collect();
            let has = |n: &str| valid.iter().any(|(k, _)| k == n);
            if valid.len() < 6 || !has("SpineF") || !has("SpineL") {
                continue;
            }
            let vis = rat7m_visibility(&valid, cam);
            let mut kp = vec![[f64::NAN; 2]; NUM_KEYPOINTS];
            let mut v = vec![0i64; NUM_KEYPOINTS];
            for ((n, x), vv) in valid.iter().zip(&vis) {
                if let Some((_, c)) = mapping.iter().find(|(k, _)| k == n) {
                    kp[index(c)] = cam.project(*x);
                    v[index(c)] = *vv;
                }
            }
            for (p, vv) in kp.iter().zip(v.iter_mut()) {
                if !cam.in_image(*p) {
                    *vv = 0;
                }
            }
            let all_uv: Vec<[f64; 2]> = valid.iter().map(|(_, x)| cam.project(*x)).filter(|p| cam.in_image(*p)).collect();
            let size = (f64::from(frame.height), f64::from(frame.width));
            let Some(bbox) = keypoint_box(&all_uv, 0.35, size, 16.0) else {
                continue;
            };
            if v.iter().filter(|x| **x > 0).count() < 4 {
                continue;
            }
            let name = format!("{stem}_{j:05}.jpg");
            write_jpeg(&out.join("images").join(&name), &crate::data::images::to_bgr8(frame), 92)?;
            writers[split].1.add(
                &name,
                (i64::from(frame.height), i64::from(frame.width)),
                &[Animal {
                    keypoints: kp,
                    visibility: v,
                    bbox,
                    track_id: 0,
                }],
            );
        }
    }
    std::fs::write(
        out.join("cameras.json"),
        PyVal::List(cams.iter().map(Rat7mCamera::to_py).collect()).dumps_indent(1),
    )?;
    let mut stats = PyVal::dict();
    for (split, w) in &writers {
        stats.set(*split, w.save(&out.join(format!("{split}.json")))?);
    }
    Ok(stats)
}

fn norm(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        .collect()
}

/// `detect_mapping(names)`: the registered mapping whose source names best match `names`
/// (case- and punctuation-insensitive), and `{source name: canonical}` for the hits.
pub fn detect_mapping(names: &[String]) -> (String, Vec<(String, String)>) {
    let mut best = (String::new(), Vec::new());
    for (key, mapping) in dataset_mappings() {
        let hit: Vec<(String, String)> = names
            .iter()
            .filter_map(|n| {
                let k = norm(n);
                // A later duplicate normalised key wins, as in the dict comprehension.
                mapping
                    .iter()
                    .rev()
                    .find(|(src, _)| norm(src) == k)
                    .map(|(_, c)| (n.clone(), c.to_string()))
            })
            .collect();
        let mut unique: Vec<(String, String)> = Vec::new();
        for h in hit {
            if let Some(slot) = unique.iter_mut().find(|u| u.0 == h.0) {
                *slot = h;
            } else {
                unique.push(h);
            }
        }
        if unique.len() > best.1.len() {
            best = (key.to_string(), unique);
        }
    }
    best
}

/// `convert_coco_keypoints(coco_json, out_json, mapping)`: any COCO keypoint file remapped onto
/// the canonical vocabulary (file names kept, so the images stay where they are). Returns the
/// writer's counts plus `mapping`, `mapped`, `unmapped` and `species`.
pub fn convert_coco_keypoints(coco_json: &Path, out_json: &Path, mapping: Option<&str>) -> Result<PyVal> {
    let text = std::fs::read_to_string(coco_json)
        .with_context(|| format!("cannot read {}", coco_json.display()))?;
    let src = loads(&text)?;
    let list = |v: Option<&PyVal>| -> Vec<PyVal> { v.and_then(PyVal::as_list).map(<[PyVal]>::to_vec).unwrap_or_default() };
    let int = |v: Option<&PyVal>| -> i64 { v.and_then(PyVal::as_f64).unwrap_or(0.0) as i64 };
    let strs = |v: Option<&PyVal>| -> Vec<String> {
        list(v).iter().filter_map(|s| s.as_str().map(str::to_string)).collect()
    };
    let cats = list(src.get("categories"));
    let names = strs(cats.first().and_then(|c| c.get("keypoints")));
    let (used, table): (String, Vec<(String, String)>) = match mapping {
        Some(m) => {
            let maps = dataset_mappings();
            let entry = &maps
                .iter()
                .find(|(k, _)| *k == m)
                .with_context(|| format!("unknown mapping {m:?}"))?
                .1;
            (
                m.to_string(),
                names
                    .iter()
                    .filter_map(|n| entry.iter().find(|(k, _)| k == n).map(|(_, c)| (n.clone(), c.to_string())))
                    .collect(),
            )
        }
        None => detect_mapping(&names),
    };
    let cat_of = |id: i64| cats.iter().find(|c| int(c.get("id")) == id);
    let images = list(src.get("images"));
    let mut per_image: Vec<(i64, Vec<(Animal, String)>)> = Vec::new();
    for a in list(src.get("annotations")) {
        let nk = a.get("num_keypoints").and_then(PyVal::as_f64).unwrap_or(1.0);
        let crowd = a.get("iscrowd").and_then(PyVal::as_f64).unwrap_or(0.0);
        if nk == 0.0 || crowd != 0.0 {
            continue;
        }
        let kps: Vec<f64> = list(a.get("keypoints")).iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect();
        let cat = cat_of(int(a.get("category_id"))).context("annotation of an unknown category")?;
        let cat_names = strs(cat.get("keypoints"));
        let mut kp = vec![[f64::NAN; 2]; NUM_KEYPOINTS];
        let mut vis = vec![0i64; NUM_KEYPOINTS];
        for (triple, n) in kps.chunks_exact(3).zip(&cat_names) {
            if let Some((_, c)) = table.iter().find(|(k, _)| k == n) {
                if triple[2] > 0.0 {
                    kp[index(c)] = [triple[0], triple[1]];
                    vis[index(c)] = if triple[2] == 2.0 { 2 } else { 1 };
                }
            }
        }
        let b: Vec<f64> = list(a.get("bbox")).iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
        if b.len() < 4 {
            bail!("annotation without a bbox in {}", coco_json.display());
        }
        let species = cat.get("name").and_then(PyVal::as_str).unwrap_or("").to_string();
        let animal = Animal {
            keypoints: kp,
            visibility: vis,
            bbox: [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            track_id: -1,
        };
        let img_id = int(a.get("image_id"));
        match per_image.iter_mut().find(|(id, _)| *id == img_id) {
            Some(slot) => slot.1.push((animal, species)),
            None => per_image.push((img_id, vec![(animal, species)])),
        }
    }
    let mut writer = CocoWriter::new();
    let mut species: Vec<(String, i64)> = Vec::new();
    for (img_id, animals) in &per_image {
        let im = images
            .iter()
            .find(|im| int(im.get("id")) == *img_id)
            .with_context(|| format!("annotation of an unknown image {img_id}"))?;
        let only: Vec<Animal> = animals.iter().map(|(a, _)| a.clone()).collect();
        writer.add(
            im.get("file_name").and_then(PyVal::as_str).unwrap_or(""),
            (int(im.get("height")), int(im.get("width"))),
            &only,
        );
        for (_, s) in animals {
            match species.iter_mut().find(|(k, _)| k == s) {
                Some(slot) => slot.1 += 1,
                None => species.push((s.clone(), 1)),
            }
        }
    }
    let mut stats = writer.save(out_json)?;
    let mut mapped: Vec<String> = table.iter().map(|(k, _)| k.clone()).collect();
    mapped.sort();
    let mut unmapped: Vec<String> = names
        .iter()
        .filter(|n| !table.iter().any(|(k, _)| k == *n))
        .cloned()
        .collect();
    unmapped.sort();
    unmapped.dedup();
    stats.set("mapping", used);
    stats.set("mapped", mapped);
    stats.set("unmapped", unmapped);
    stats.set(
        "species",
        PyVal::Dict(species.into_iter().map(|(k, v)| (k, PyVal::Int(v))).collect()),
    );
    Ok(stats)
}

/// `prepare._image_root(ann)`: the folder the annotation's `file_name` entries are relative to.
pub fn image_root(ann: &Path) -> Result<PathBuf> {
    let doc = loads(&std::fs::read_to_string(ann)?)?;
    let first = doc
        .get("images")
        .and_then(PyVal::as_list)
        .and_then(|l| l.first())
        .and_then(|im| im.get("file_name"))
        .and_then(PyVal::as_str)
        .context("the annotation lists no images")?
        .to_string();
    let base = Path::new(&first)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut dirs: Vec<&Path> = Vec::new();
    let mut at = ann.parent();
    while let Some(d) = at {
        dirs.push(d);
        at = d.parent();
    }
    for d in dirs {
        let d = if d.as_os_str().is_empty() { Path::new(".") } else { d };
        if d.join(&first).exists() {
            return Ok(std::fs::canonicalize(d)?);
        }
        if let Some(hit) = find_file(d, &base) {
            return Ok(if base == first {
                std::fs::canonicalize(hit.parent().unwrap_or(d))?
            } else {
                std::fs::canonicalize(d)?
            });
        }
    }
    bail!("[Errno 2] No such file or directory: {first:?}")
}

fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in &entries {
        if p.is_file() && p.file_name().is_some_and(|f| f.to_string_lossy() == name) {
            return Some(p.clone());
        }
    }
    entries
        .iter()
        .filter(|p| p.is_dir())
        .find_map(|p| find_file(p, name))
}

/// All `*.json` files under `dir`, sorted (`sorted(dir.rglob("*.json"))`).
pub fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "json") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crim13_names_map_to_animal_and_canonical_point() {
        assert_eq!(lp_crim13_mapping("black_mouse_nose"), Some((0, "nose")));
        assert_eq!(lp_crim13_mapping("white_mouse_base_tail"), Some((1, "tail_base")));
        assert_eq!(lp_crim13_mapping("top_of_neck_white"), Some((1, "neck")));
        assert_eq!(lp_crim13_mapping("black_mouse_whisker_pad"), None);
        assert_eq!(lp_crim13_mapping("nose"), None);
    }

    #[test]
    fn calms21_names_are_detected_as_the_synthetic_vocabulary() {
        let names: Vec<String> = [
            "nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base", "whisker_pad",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let (used, table) = detect_mapping(&names);
        assert_eq!(used, "synthetic");
        assert_eq!(table.len(), 7);
        let (used, table) = detect_mapping(&["Nose".into(), "Root of Tail".into(), "Left Eye".into()]);
        assert_eq!((used.as_str(), table.len()), ("ap10k", 3)); // beats the 2 hits of "synthetic"
    }

    #[test]
    fn rat7m_cameras_apply_the_1_based_origin_and_distortion_defaults() {
        use crate::data::mat5::parse_mat;
        use crate::data::mat5::tests::{double, file, structure};
        // MATLAB's row-vector convention: K and R are stored transposed, the principal point 1-based.
        let cam = structure(
            "",
            &[
                ("frame", double("", &[vec![0.0, 1.0, 2.0]])),
                (
                    "IntrinsicMatrix",
                    double("", &[vec![800.0, 0.0, 0.0], vec![0.0, 810.0, 0.0], vec![665.0, 525.0, 1.0]]),
                ),
                (
                    "rotationMatrix",
                    double("", &[vec![1.0, 0.0, 0.0], vec![0.0, -1.0, 0.0], vec![0.0, 0.0, -1.0]]),
                ),
                ("translationVector", double("", &[vec![0.0, 0.0, 1000.0]])),
                ("TangentialDistortion", double("", &[vec![0.0, 0.0]])),
                ("RadialDistortion", double("", &[vec![-0.2, 0.05]])),
            ],
        );
        let vars = parse_mat(&file(&[structure("cameras", &[("Camera1", cam)])], true)).unwrap();
        let cams = rat7m_cameras(&vars[0].1).unwrap();
        assert_eq!(cams.len(), 1);
        let c = &cams[0];
        assert_eq!(c.k, [[800.0, 0.0, 664.0], [0.0, 810.0, 524.0], [0.0, 0.0, 1.0]]);
        assert_eq!(c.dist, [-0.2, 0.05, 0.0, 0.0, 0.0]);
        assert_eq!(c.center(), [0.0, 0.0, 1000.0]);
        // A point on the optical axis lands on the principal point.
        assert_eq!(c.project([0.0, 0.0, 0.0]), [664.0, 524.0]);
        assert!(c.in_image([664.0, 524.0]) && !c.in_image([f64::NAN, 1.0]) && !c.in_image([1328.0, 0.0]));
        // Spine along +x, so the body's left is +y. Seen from (0, 0, 1000), a lateral marker is
        // visible when the camera lies on that marker's side: dot(side * left, center - X) > 0.
        let markers = vec![
            ("SpineF".to_string(), [10.0, 0.0, 0.0]),
            ("SpineL".to_string(), [0.0, 0.0, 0.0]),
            ("HipL".to_string(), [5.0, -2000.0, 0.0]),
            ("HipR".to_string(), [5.0, -3.0, 0.0]),
        ];
        assert_eq!(rat7m_visibility(&markers, c), [2, 2, 2, 1]);
        assert_eq!(rat7m_video_name("s1-d1-camera3-105000.mp4"), Some(("3".into(), 105_000)));
        assert_eq!(rat7m_video_name("s1-d1-cam3-0.mp4"), None);
    }

    #[test]
    fn boxes_are_padded_and_clipped() {
        let kp = [[10.0, 20.0], [30.0, 25.0], [f64::NAN, f64::NAN]];
        // python: _box(kp, 0.25, (120, 160)): each side grows by max(extent, 16) / 4
        let b = keypoint_box(&kp, 0.25, (120.0, 160.0), 16.0).unwrap();
        assert_eq!(b, [5.0, 16.0, 35.0, 29.0]);
        assert!(keypoint_box(&kp[..1], 0.25, (120.0, 160.0), 16.0).is_none());
    }
}
