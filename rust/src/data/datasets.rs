//! `src/animalpose/data/datasets.py`: the training datasets of the detector and the pose network.
//!
//! | Python | Rust |
//! | --- | --- |
//! | `Annotation` | [`Annotation`] |
//! | `SyntheticScenes` | [`SyntheticScenes`] |
//! | `CocoKeypoints` (the COCO JSON `data/real.py:CocoWriter` and `data/dlc.py:to_coco` write) | [`CocoKeypoints`] |
//! | `MixedSource` | [`MixedSource`] |
//! | `train/loops.py:_source` | [`source_from_config`] |
//! | `letterbox` / `to_tensor` / `photometric` | [`crate::algos::image::letterbox`] / [`crate::data::images::to_tensor`] / [`photometric`] |
//! | `DetectionDataset` / `detection_collate` | [`DetectionDataset`] / [`DetectionItem`] |
//! | `PoseCropDataset` (incl. `degraded_crop`) | [`PoseCropDataset`] / [`PoseItem`] |
//! | `skeleton/species.py:DATASET_MAPPINGS` / `remap_keypoints` | [`dataset_mapping`] / [`remap_keypoints`] |
//!
//! Items are pure functions of their index: `numpy.random.default_rng((a, b, ...))` becomes
//! [`keyed_rng`] over the same key, so item `i` is the same whichever thread builds it and in
//! whatever order, exactly the property the Python `DataLoader` workers rely on.
//!
//! `// differs from Python:` the generators are [`Rng`]s seeded by hashing the key tuple, not
//! numpy's `SeedSequence`, so an index draws different (identically distributed) augmentations.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};

use crate::algos::geometry::{apply_affine, box_xyxy_to_cs_default, crop_affine};
use crate::algos::image::{letterbox, Bgr};
use crate::data::config::Value;
use crate::data::degrade::{degrade, zoom_out, Profile};
use crate::data::images::to_tensor;
use crate::data::raster::{gaussian_blur, read_bgr, warp_affine, ImageF};
use crate::data::render::{random_camera, render, simulate_scene, Scene};
use crate::data::synthetic::{Rng, CANONICAL_NAMES, FLIP_INDEX};

/// `CANONICAL.num_keypoints`.
pub const K: usize = CANONICAL_NAMES.len();

/// `np.random.default_rng((a, b, ...))`: a generator that is a pure function of the key tuple.
pub fn keyed_rng(key: &[u64]) -> Rng {
    let mut h: u64 = 0x243F_6A88_85A3_08D3;
    for &part in key {
        h ^= part;
        h = h.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = h;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        h = z ^ (z >> 31);
    }
    Rng::new(h)
}

/// `Annotation`: one image with its animals, keypoints in canonical order.
#[derive(Clone, Debug)]
pub struct Annotation {
    /// 8-bit BGR.
    pub image: Bgr,
    /// `(N, 4)` xyxy.
    pub boxes: Vec<[f32; 4]>,
    /// `(N, K, 2)` pixels.
    pub keypoints: Vec<[f32; 2]>,
    /// `(N, K)`: 2 visible, 1 labeled but occluded, 0 unlabeled.
    pub state: Vec<i64>,
    /// `(N,)` identity for the embedding loss, -1 unknown.
    pub ids: Vec<i64>,
    /// `(N,)` index into the species list, -1 unknown.
    pub species: Vec<i64>,
}

impl Annotation {
    /// Animals in the image.
    pub fn len(&self) -> usize {
        self.boxes.len()
    }

    /// `len(ann.boxes) == 0`.
    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }
}

// ------------------------------------------------------------------------------------ mappings

/// `DATASET_MAPPINGS[name]`: dataset keypoint name → canonical name, for every mapping
/// `skeleton/species.py` registers.
pub fn dataset_mapping(name: &str) -> Option<Vec<(&'static str, &'static str)>> {
    let table: &[(&str, &str)] = match name {
        "superanimal_quadruped" => &[
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
        "ap10k" => &[
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
        "calms21" => &[
            ("nose", "nose"),
            ("left_ear", "left_ear"),
            ("right_ear", "right_ear"),
            ("neck", "neck"),
            ("left_hip", "left_hip"),
            ("right_hip", "right_hip"),
            ("tail_base", "tail_base"),
        ],
        "rat7m" => &[
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
        "lp_mirror_mouse" => &[
            ("paw1LH", "left_hind_paw"),
            ("paw2LF", "left_front_paw"),
            ("paw3RF", "right_front_paw"),
            ("paw4RH", "right_hind_paw"),
            ("tailBase", "tail_base"),
            ("tailMid", "tail_mid"),
            ("nose", "nose"),
        ],
        "lp_crim13" => &[
            ("nose", "nose"),
            ("left_ear", "left_ear"),
            ("right_ear", "right_ear"),
            ("top_of_neck", "neck"),
            ("left_rear_knee", "left_hind_knee"),
            ("right_rear_knee", "right_hind_knee"),
            ("base_of_tail", "tail_base"),
        ],
        other => return crate::data::synthetic::dataset_mapping(other),
    };
    Some(table.to_vec())
}

/// A keypoint mapping: a registered dataset name or an explicit `{source: canonical}` dict.
#[derive(Clone, Debug, PartialEq)]
pub enum Mapping {
    /// A key of [`dataset_mapping`].
    Named(String),
    /// An explicit table (`data.mapping: {nose: nose, ...}` in a config).
    Explicit(Vec<(String, String)>),
}

impl Mapping {
    /// `data.mapping` of a config: a name or a mapping; `None` is `"synthetic"`.
    pub fn from_value(value: Option<&Value>) -> Result<Self> {
        match value {
            None | Some(Value::Null) => Ok(Self::Named("synthetic".to_string())),
            Some(Value::Str(s)) => Ok(Self::Named(s.clone())),
            Some(Value::Map(m)) => Ok(Self::Explicit(
                m.iter()
                    .map(|(k, v)| Ok((k.clone(), v.as_str().context("mapping values are names")?.to_string())))
                    .collect::<Result<_>>()?,
            )),
            Some(other) => bail!("data.mapping must be a dataset name or a mapping, got {other:?}"),
        }
    }

    /// Source name → canonical index.
    pub fn table(&self) -> Result<HashMap<String, usize>> {
        let pairs: Vec<(String, String)> = match self {
            Self::Named(name) => dataset_mapping(name)
                .with_context(|| format!("unknown keypoint mapping {name:?}"))?
                .into_iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
            Self::Explicit(pairs) => pairs.clone(),
        };
        pairs
            .into_iter()
            .map(|(src, canon)| {
                let Some(i) = CANONICAL_NAMES.iter().position(|n| *n == canon) else {
                    bail!("mapping target {canon:?} is not in the canonical vocabulary");
                };
                Ok((src, i))
            })
            .collect()
    }
}

/// `remap_keypoints(names, keypoints, mapping)` for `(J, C)` rows: `(K, C)` in canonical order
/// (`NaN` where undefined) and the `(K,)` mask of canonical points the dataset defines.
pub fn remap_keypoints(
    names: &[String],
    keypoints: &[Vec<f32>],
    table: &HashMap<String, usize>,
) -> (Vec<Vec<f32>>, Vec<bool>) {
    let c = keypoints.first().map_or(3, Vec::len);
    let mut out = vec![vec![f32::NAN; c]; K];
    let mut labeled = vec![false; K];
    for (j, name) in names.iter().enumerate() {
        if let (Some(&i), Some(row)) = (table.get(name), keypoints.get(j)) {
            out[i] = row.clone();
            labeled[i] = true;
        }
    }
    (out, labeled)
}

// ------------------------------------------------------------------------------------- sources

/// `SyntheticScenes`: random access to rendered frames from a fixed pool of simulations.
pub struct SyntheticScenes {
    /// `(width, height)` of the rendered frames.
    pub size: (usize, usize),
    /// The species names; `Annotation.species` indexes this list.
    pub species: Vec<String>,
    /// The simulation pool.
    pub scenes: Vec<Scene>,
    /// `seed`.
    pub seed: u64,
    /// Extra share of 5–25° (side-like) cameras.
    pub low_view_fraction: f64,
}

impl SyntheticScenes {
    /// `SyntheticScenes(n_sims, frames, size, species, max_animals, seed, low_view_fraction)`.
    pub fn new(
        n_sims: usize,
        frames: usize,
        size: (usize, usize),
        species: &[String],
        max_animals: usize,
        seed: u64,
        low_view_fraction: f64,
    ) -> Result<Self> {
        ensure!(!species.is_empty(), "the simulator needs at least one species");
        ensure!(frames > 20, "data.frames must exceed 20 (frames are sampled from t >= 20)");
        let mut rng = Rng::new(seed);
        let mut scenes = Vec::with_capacity(n_sims);
        for _ in 0..n_sims {
            let n = rng.integers(1, max_animals.max(1) as i64 + 1) as usize;
            let sp = rng.choose(species).clone();
            scenes.push(simulate_scene(n, frames, &sp, &mut rng, None)?);
        }
        Ok(Self {
            size,
            species: species.to_vec(),
            scenes,
            seed,
            low_view_fraction,
        })
    }

    /// `SyntheticScenes.sample(index, variant)`; `variant` re-renders the same simulation at another
    /// time and viewpoint (the positive pairs of the identity embedding).
    pub fn sample(&self, index: u64, variant: u64) -> Annotation {
        let s = keyed_rng(&[self.seed, index]).integers(0, self.scenes.len() as i64) as usize;
        let mut rng = keyed_rng(&[self.seed, index, variant]);
        let scene = &self.scenes[s];
        let t = rng.integers(20, scene.sim.n_frames as i64) as usize;
        let low = rng.random() < self.low_view_fraction;
        let elevation = if low { (5.0, 25.0) } else { (10.0, 90.0) };
        let cam = random_camera(scene.sim.arena_cm, &mut rng, self.size, elevation);
        let f = render(scene, &cam, t, &mut rng, None, true);
        let species = self
            .species
            .iter()
            .position(|n| *n == scene.sim.species)
            .map_or(-1, |i| i as i64);
        let mut ann = Annotation {
            image: f.image,
            boxes: Vec::new(),
            keypoints: Vec::new(),
            state: Vec::new(),
            ids: Vec::new(),
            species: Vec::new(),
        };
        for a in 0..f.present.len() {
            if !f.present[a] {
                continue;
            }
            ann.boxes.push(f.boxes[a]);
            for k in 0..K {
                let p = f.keypoints[a * K + k];
                ann.keypoints.push(p);
                let inside = f64::from(p[0]) >= 0.0
                    && f64::from(p[0]) < cam.width as f64
                    && f64::from(p[1]) >= 0.0
                    && f64::from(p[1]) < cam.height as f64;
                // Out-of-frame points carry no label.
                ann.state.push(match (inside, f.visible[a * K + k]) {
                    (false, _) => 0,
                    (true, true) => 2,
                    (true, false) => 1,
                });
            }
            ann.ids.push(s as i64 * 16 + a as i64);
            ann.species.push(species);
        }
        ann
    }
}

/// One COCO keypoint annotation as read from the JSON.
#[derive(Clone, Debug)]
struct CocoAnn {
    category: i64,
    keypoints: Vec<f32>,
    bbox: [f64; 4],
    track_id: i64,
}

/// `CocoKeypoints`: COCO keypoint JSON → canonical annotations through a dataset mapping. COCO
/// visibility `v = 2` is visible, `v = 1` occluded-labeled, `v = 0` unlabeled.
pub struct CocoKeypoints {
    /// Image directory the `file_name`s are relative to.
    pub root: PathBuf,
    /// `image id -> file_name`.
    files: HashMap<i64, String>,
    /// The annotations of each image that has keypoints.
    by_image: HashMap<i64, Vec<CocoAnn>>,
    /// `sorted(by_image)`.
    pub ids: Vec<i64>,
    /// `category id -> keypoint names`.
    cats: HashMap<i64, Vec<String>>,
    /// Source name → canonical index.
    table: HashMap<String, usize>,
    /// Index of `species` in the species list, or -1.
    pub species_idx: i64,
}

impl CocoKeypoints {
    /// `CocoKeypoints(json_path, image_root, mapping, species, species_list)`.
    pub fn new(
        json_path: &Path,
        image_root: &Path,
        mapping: &Mapping,
        species: Option<&str>,
        species_list: &[&str],
    ) -> Result<Self> {
        let text = std::fs::read_to_string(json_path)
            .with_context(|| format!("cannot read {}", json_path.display()))?;
        let data: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("cannot parse {}", json_path.display()))?;
        let id_of = |v: &serde_json::Value, key: &str| -> Result<i64> {
            v.get(key)
                .and_then(serde_json::Value::as_i64)
                .with_context(|| format!("COCO entry without an integer {key:?}"))
        };
        let mut cats = HashMap::new();
        for c in data["categories"].as_array().context("COCO JSON without categories")? {
            let names = c["keypoints"]
                .as_array()
                .context("COCO category without keypoints")?
                .iter()
                .map(|n| n.as_str().unwrap_or_default().to_string())
                .collect();
            cats.insert(id_of(c, "id")?, names);
        }
        let mut files = HashMap::new();
        for im in data["images"].as_array().context("COCO JSON without images")? {
            files.insert(
                id_of(im, "id")?,
                im["file_name"].as_str().context("image without file_name")?.to_string(),
            );
        }
        let mut by_image: HashMap<i64, Vec<CocoAnn>> = HashMap::new();
        for ann in data["annotations"].as_array().context("COCO JSON without annotations")? {
            if ann.get("num_keypoints").and_then(serde_json::Value::as_i64).unwrap_or(1) <= 0 {
                continue;
            }
            let bbox: Vec<f64> = ann["bbox"]
                .as_array()
                .context("annotation without bbox")?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();
            ensure!(bbox.len() == 4, "a COCO bbox has four numbers");
            let keypoints = ann["keypoints"]
                .as_array()
                .context("annotation without keypoints")?
                .iter()
                .map(|v| v.as_f64().map_or(f32::NAN, |x| x as f32))
                .collect();
            by_image.entry(id_of(ann, "image_id")?).or_default().push(CocoAnn {
                category: id_of(ann, "category_id")?,
                keypoints,
                bbox: [bbox[0], bbox[1], bbox[2], bbox[3]],
                track_id: ann.get("track_id").and_then(serde_json::Value::as_i64).unwrap_or(-1),
            });
        }
        let mut ids: Vec<i64> = by_image.keys().copied().collect();
        ids.sort_unstable();
        let species_idx = species
            .and_then(|s| species_list.iter().position(|n| *n == s))
            .map_or(-1, |i| i as i64);
        Ok(Self {
            root: image_root.to_path_buf(),
            files,
            by_image,
            ids,
            cats,
            table: mapping.table()?,
            species_idx,
        })
    }

    /// Images with at least one labelled animal.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// No labelled image at all.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// `CocoKeypoints.sample(index)`: image `ids[index % len]` and its animals.
    pub fn sample(&self, index: u64) -> Result<Annotation> {
        ensure!(!self.ids.is_empty(), "the COCO file has no annotated image");
        let id = self.ids[(index % self.ids.len() as u64) as usize];
        let file = self.files.get(&id).context("annotation of an unknown image id")?;
        let image = read_bgr(&self.root.join(file))?;
        let anns = &self.by_image[&id];
        let mut out = Annotation {
            image,
            boxes: Vec::new(),
            keypoints: Vec::new(),
            state: Vec::new(),
            ids: Vec::new(),
            species: Vec::new(),
        };
        for a in anns {
            let names = self
                .cats
                .get(&a.category)
                .with_context(|| format!("unknown category {}", a.category))?;
            let rows: Vec<Vec<f32>> = a.keypoints.chunks(3).map(<[f32]>::to_vec).collect();
            let (canon, labeled) = remap_keypoints(names, &rows, &self.table);
            for k in 0..K {
                let row = &canon[k];
                let v = row.get(2).copied().map_or(0.0, crate::data::synthetic::nan_to_num);
                out.state.push(if labeled[k] && v >= 2.0 {
                    2
                } else if labeled[k] && v >= 1.0 {
                    1
                } else {
                    0
                });
                out.keypoints.push([
                    crate::data::synthetic::nan_to_num(row[0]),
                    crate::data::synthetic::nan_to_num(row.get(1).copied().unwrap_or(f32::NAN)),
                ]);
            }
            let [x, y, w, h] = a.bbox;
            out.boxes.push([x as f32, y as f32, (x + w) as f32, (y + h) as f32]);
            out.ids.push(a.track_id);
            out.species.push(self.species_idx);
        }
        Ok(out)
    }
}

/// `MixedSource`: a weighted mixture of annotation sources.
pub struct MixedSource {
    /// The mixed sources.
    pub sources: Vec<Source>,
    /// Normalised weights.
    pub p: Vec<f64>,
    /// `seed`.
    pub seed: u64,
}

impl MixedSource {
    /// `MixedSource(sources, weights, seed)`.
    pub fn new(sources: Vec<Source>, weights: &[f64], seed: u64) -> Result<Self> {
        ensure!(
            !sources.is_empty() && sources.len() == weights.len(),
            "data.sources needs one weight per source"
        );
        let total: f64 = weights.iter().sum();
        ensure!(total > 0.0, "data.sources weights must not all be zero");
        Ok(Self {
            sources,
            p: weights.iter().map(|w| w / total).collect(),
            seed,
        })
    }

    /// `MixedSource.sample(index, variant)`: the source from `default_rng((seed, 99, index))`, its
    /// item index from `default_rng((seed, 98, index)).integers(1 << 30)`.
    pub fn sample(&self, index: u64, variant: u64) -> Result<Annotation> {
        let idx: Vec<usize> = (0..self.sources.len()).collect();
        let k = keyed_rng(&[self.seed, 99, index]).choose_weighted(&idx, &self.p);
        let sub = keyed_rng(&[self.seed, 98, index]).integers(0, 1 << 30) as u64;
        self.sources[k].sample(sub, variant)
    }
}

/// Any annotation source the two datasets read.
pub enum Source {
    /// [`SyntheticScenes`].
    Synthetic(SyntheticScenes),
    /// [`CocoKeypoints`].
    Coco(CocoKeypoints),
    /// [`MixedSource`].
    Mixed(MixedSource),
}

impl Source {
    /// `source.sample(index, variant)` (`CocoKeypoints` ignores `variant`, as in Python).
    pub fn sample(&self, index: u64, variant: u64) -> Result<Annotation> {
        match self {
            Source::Synthetic(s) => Ok(s.sample(index, variant)),
            Source::Coco(c) => c.sample(index),
            Source::Mixed(m) => m.sample(index, variant),
        }
    }
}

/// A config path: as given when it exists (relative to the working directory, as Python reads
/// it), else relative to the repository root.
pub fn resolve_path(path: &str) -> PathBuf {
    let p = PathBuf::from(path);
    if p.is_absolute() || p.exists() {
        return p;
    }
    match crate::data::repo_root() {
        Some(root) if root.join(&p).exists() => root.join(p),
        _ => p,
    }
}

/// `train/loops.py:_source(data_cfg, seed_offset)`: `data.source` = `synthetic` | `coco`, or
/// `data.sources` = a list of such mappings (each merged over the parent block) with a `weight`.
pub fn source_from_config(data: &Value, seed_offset: u64) -> Result<Source> {
    let empty = Value::Map(Default::default());
    let data = if data.is_null() { &empty } else { data };
    let num = |key: &str, default: f64| data.get(key).and_then(Value::as_f64).unwrap_or(default);
    let seed = data.get("seed").and_then(Value::as_usize).unwrap_or(0) as u64;
    if let Some(list) = data.get("sources") {
        let items = list.as_list().context("data.sources must be a list")?;
        let parent = data.map().context("data must be a mapping")?;
        let mut subs = Vec::new();
        let mut weights = Vec::new();
        for item in items {
            let mut merged = parent.clone();
            merged.remove("sources");
            for (k, v) in item.map().context("each data.sources entry is a mapping")? {
                merged.insert(k.clone(), v.clone());
            }
            weights.push(item.get("weight").and_then(Value::as_f64).unwrap_or(1.0));
            subs.push(source_from_config(&Value::Map(merged), seed_offset)?);
        }
        return Ok(Source::Mixed(MixedSource::new(subs, &weights, seed + seed_offset)?));
    }
    let kind = data.get("source").and_then(Value::as_str).unwrap_or("synthetic");
    match kind {
        "synthetic" => {
            let size = data.get("image_size").map(Value::f64_list).unwrap_or_default();
            let size = if size.len() == 2 {
                (size[0] as usize, size[1] as usize)
            } else {
                (320, 256)
            };
            let species = match data.get("species") {
                Some(v) if !v.str_list().is_empty() => v.str_list(),
                _ => vec!["mouse".to_string(), "rat".to_string()],
            };
            Ok(Source::Synthetic(SyntheticScenes::new(
                num("n_sims", 64.0) as usize,
                num("frames", 120.0) as usize,
                size,
                &species,
                num("max_animals", 4.0) as usize,
                seed + seed_offset,
                num("low_view_fraction", 0.0),
            )?))
        }
        "coco" => {
            let json = data.get("json").and_then(Value::as_str).context("data.json is required for source: coco")?;
            let images = data.get("images").and_then(Value::as_str).context("data.images is required for source: coco")?;
            // `data_cfg.get("species", "mouse")`; a list (inherited from a synthetic block) matches
            // no species, as `list in tuple` is False in Python.
            let species = match data.get("species") {
                None => Some("mouse"),
                Some(v) => v.as_str(),
            };
            Ok(Source::Coco(CocoKeypoints::new(
                &resolve_path(json),
                &resolve_path(images),
                &Mapping::from_value(data.get("mapping"))?,
                species,
                &["mouse", "rat"],
            )?))
        }
        other => bail!("unknown data source {other}"),
    }
}

// --------------------------------------------------------------------------------- augmentation

/// `photometric(image, rng)`: grey conversion (20 %), gain and offset, per-channel gain, Gaussian
/// blur (30 %) and noise (30 %).
pub fn photometric(image: &Bgr, rng: &mut Rng) -> Bgr {
    let mut img = ImageF::from_bgr(image);
    if rng.random() < 0.2 {
        img = img.to_mono();
    }
    let gain = rng.uniform(0.7, 1.3) as f32;
    let offset = rng.uniform(-25.0, 25.0) as f32;
    for v in img.data.iter_mut() {
        *v = *v * gain + offset;
    }
    let per: Vec<f32> = (0..3).map(|_| rng.uniform(0.9, 1.1) as f32).collect();
    for px in img.data.chunks_exact_mut(3) {
        for (v, g) in px.iter_mut().zip(&per) {
            *v *= g;
        }
    }
    if rng.random() < 0.3 {
        let sigma = rng.uniform(0.5, 1.5);
        img = gaussian_blur(&img, 0, sigma);
    }
    if rng.random() < 0.3 {
        let sigma = rng.uniform(2.0, 8.0);
        for v in img.data.iter_mut() {
            *v += rng.normal(0.0, sigma) as f32;
        }
    }
    img.to_bgr_trunc()
}

/// `geometry.warp(image, m, out_size)` on an 8-bit image: bilinear `warpAffine`, zero border,
/// rounded back to 8 bits.
pub fn warp(image: &Bgr, m: &[[f64; 3]; 2], out_size: usize) -> Bgr {
    warp_affine(&ImageF::from_bgr(image), m, out_size, out_size).to_bgr_round()
}

/// `np.median(np.maximum(w, h))` over boxes.
fn median_side(boxes: &[[f32; 4]]) -> f64 {
    let sides: Vec<f64> = boxes
        .iter()
        .map(|b| f64::from((b[2] - b[0]).max(b[3] - b[1])))
        .collect();
    crate::data::raster::median(&sides)
}

/// One `DetectionDataset` item, the tuple `detection_collate` stacks.
#[derive(Clone, Debug)]
pub struct DetectionItem {
    /// `(3, H, W)` RGB in `[0, 1]`.
    pub image: Vec<f32>,
    /// `(N, 4)` xyxy in input pixels.
    pub boxes: Vec<[f32; 4]>,
    /// `(N,)` species indices.
    pub species: Vec<i64>,
    /// Whether [`degrade`] ran.
    pub degraded: bool,
}

/// `DetectionDataset`: letterboxed frames with boxes, flip / zoom-out / photometric / degradation
/// augmentation.
pub struct DetectionDataset {
    /// The annotation source.
    pub source: Source,
    /// `len(dataset)`.
    pub length: usize,
    /// Network input `(H, W)`.
    pub size: (usize, usize),
    /// Whether to augment.
    pub augment: bool,
    /// Probability of [`degrade`].
    pub degrade: f64,
    /// Probability of [`zoom_out`].
    pub zoom_out: f64,
    /// The camera profile of the degradations.
    pub profile: Profile,
}

impl DetectionDataset {
    /// `DetectionDataset.__getitem__(i)`.
    pub fn get(&self, i: u64) -> Result<DetectionItem> {
        let ann = self.source.sample(i, 0)?;
        let mut rng = keyed_rng(&[7, i]);
        let mut img = ann.image;
        let mut boxes = ann.boxes;
        if self.augment && rng.random() < 0.5 {
            img = ImageF::from_bgr(&img).flip_horizontal().to_bgr_round();
            let w = img.width as f32;
            for b in boxes.iter_mut() {
                *b = [w - b[2], b[1], w - b[0], b[3]];
            }
        }
        if self.augment && !boxes.is_empty() && rng.random() < self.zoom_out {
            let (z, zb) = zoom_out(&img, &boxes, &mut rng, 0.35);
            img = z;
            boxes = zb;
        }
        if self.augment {
            img = photometric(&img, &mut rng);
        }
        let mut flag = false;
        if self.augment && !boxes.is_empty() && rng.random() < self.degrade {
            let size = median_side(&boxes);
            img = degrade(&img, &mut rng, size, 1.0, None, &self.profile)?;
            flag = true;
        }
        let (lb, s, pad) = letterbox(&img, (self.size.0 as u32, self.size.1 as u32))?;
        let boxes = boxes
            .iter()
            .map(|b| {
                [
                    (f64::from(b[0]) * s + pad[0]) as f32,
                    (f64::from(b[1]) * s + pad[1]) as f32,
                    (f64::from(b[2]) * s + pad[0]) as f32,
                    (f64::from(b[3]) * s + pad[1]) as f32,
                ]
            })
            .collect();
        Ok(DetectionItem {
            image: to_tensor(&lb)?,
            boxes,
            species: ann.species,
            degraded: flag,
        })
    }
}

/// One `PoseCropDataset` item.
#[derive(Clone, Debug)]
pub struct PoseItem {
    /// `(3, S, S)` RGB in `[0, 1]`.
    pub crop: Vec<f32>,
    /// `(K, 2)` crop pixels.
    pub keypoints: Vec<[f32; 2]>,
    /// `(K,)` 2 / 1 / 0.
    pub state: Vec<i64>,
    /// Identity, -1 unknown.
    pub id: i64,
    /// Whether the degraded crop was used.
    pub degraded: bool,
}

/// `PoseCropDataset`: top-down crops with rotation / scale / shift / flip augmentation, the crop
/// box jittered like a detector's.
pub struct PoseCropDataset {
    /// The annotation source.
    pub source: Source,
    /// `len(dataset)`.
    pub length: usize,
    /// Crop side in pixels.
    pub crop: usize,
    /// Whether to augment.
    pub augment: bool,
    /// Maximum rotation in radians.
    pub rotation: f64,
    /// Side scale range.
    pub scale: (f64, f64),
    /// Centre shift as a fraction of the side.
    pub shift: f64,
    /// Probability of the degraded crop.
    pub degrade: f64,
    /// The camera profile of the degradations.
    pub profile: Profile,
}

impl PoseCropDataset {
    /// Degrading happens on a native-scale patch of the frame of at most this many pixels.
    pub const MAX_BLUR_BUFFER: f64 = 256.0;

    /// `PoseCropDataset(source, length, crop_size, degrade=..., profile=...)` with the Python
    /// augmentation defaults (`rotation = pi`, `scale = (0.8, 1.25)`, `shift = 0.1`).
    pub fn new(source: Source, length: usize, crop: usize, degrade: f64, profile: Profile) -> Self {
        Self {
            source,
            length,
            crop,
            augment: true,
            rotation: std::f64::consts::PI,
            scale: (0.8, 1.25),
            shift: 0.1,
            degrade,
            profile,
        }
    }

    /// `PoseCropDataset.__getitem__(i)`.
    ///
    /// Items `4n .. 4n + 3` come from one scene at different times / viewpoints: two crops of
    /// animal A, then two of animal B (when the scene has one), so every batch holds same-scene
    /// positives and negatives for the identity embedding.
    pub fn get(&self, i: u64) -> Result<PoseItem> {
        let mut rng = keyed_rng(&[11, i]);
        let (mut base, variant) = (i / 4, i % 4);
        let mut ann = self.source.sample(base, variant)?;
        while ann.is_empty() {
            base += 1_000_003;
            ann = self.source.sample(base, variant)?;
        }
        let mut scene_ids = if variant > 0 {
            self.source.sample(base, 0)?.ids
        } else {
            ann.ids.clone()
        };
        scene_ids.sort_unstable();
        scene_ids.dedup();
        let order = {
            let mut prng = keyed_rng(&[13, base]);
            let n = scene_ids.len();
            prng.sample_without_replacement(&scene_ids, n)
        };
        let target = if order.is_empty() {
            -1
        } else {
            order[((variant / 2) as usize).min(order.len() - 1)]
        };
        let j = match ann.ids.iter().position(|id| *id == target) {
            Some(j) => j,
            None => rng.integers(0, ann.len() as i64) as usize,
        };
        let b = ann.boxes[j];
        let (mut center, mut side) = box_xyxy_to_cs_default([
            f64::from(b[0]),
            f64::from(b[1]),
            f64::from(b[2]),
            f64::from(b[3]),
        ]);
        let mut rot = 0.0;
        let mut kpts: Vec<[f32; 2]> = ann.keypoints[j * K..(j + 1) * K].to_vec();
        let mut state: Vec<i64> = ann.state[j * K..(j + 1) * K].to_vec();
        if self.augment {
            let dx = rng.uniform(-self.shift, self.shift);
            let dy = rng.uniform(-self.shift, self.shift);
            center = [center[0] + dx * side, center[1] + dy * side];
            side *= rng.uniform(self.scale.0, self.scale.1);
            rot = if rng.random() < 0.8 {
                rng.uniform(-self.rotation, self.rotation)
            } else {
                0.0
            };
        }
        let m = crop_affine(center, side, self.crop as u32, rot);
        for p in kpts.iter_mut() {
            let q = apply_affine(m, [f64::from(p[0]), f64::from(p[1])]);
            *p = [q[0] as f32, q[1] as f32];
        }
        let mut flag = false;
        let mut crop = if self.augment && rng.random() < self.degrade {
            flag = true;
            self.degraded_crop(&ann.image, center, side, rot, &mut rng)?
        } else {
            warp(&ann.image, &m, self.crop)
        };
        if self.augment && rng.random() < 0.5 {
            crop = ImageF::from_bgr(&crop).flip_horizontal().to_bgr_round();
            let s = self.crop as f32;
            let flipped: Vec<[f32; 2]> = FLIP_INDEX.iter().map(|&f| kpts[f]).collect();
            kpts = flipped.iter().map(|p| [s - 1.0 - p[0], p[1]]).collect();
            state = FLIP_INDEX.iter().map(|&f| state[f]).collect();
        }
        if self.augment {
            crop = photometric(&crop, &mut rng);
        }
        let hi = (self.crop - 1) as f32;
        for (s, p) in state.iter_mut().zip(&kpts) {
            let inside = p[0] >= 0.0 && p[1] >= 0.0 && p[0] <= hi && p[1] <= hi;
            if !inside {
                *s = 0;
            }
        }
        Ok(PoseItem {
            crop: to_tensor(&crop)?,
            keypoints: kpts,
            state,
            id: ann.ids[j],
            degraded: flag,
        })
    }

    /// `PoseCropDataset.degraded_crop`: the camera degradations applied at the camera's own pixel
    /// scale — a native-scale patch slightly larger than the crop box is degraded, then resampled
    /// into the crop; animals larger than [`Self::MAX_BLUR_BUFFER`] are shrunk into the patch with
    /// the PSF scaled along.
    pub fn degraded_crop(
        &self,
        img: &Bgr,
        center: [f64; 2],
        side: f64,
        rot: f64,
        rng: &mut Rng,
    ) -> Result<Bgr> {
        let side_buf = side * 1.12;
        let scale = (Self::MAX_BLUR_BUFFER / side_buf.max(1.0)).min(1.0);
        let buf = 8usize.max((side_buf * scale).round_ties_even() as usize);
        let src = warp(img, &crop_affine(center, side_buf, buf as u32, 0.0), buf);
        let psf = self.profile.psf_px.map(|p| p * scale);
        let src = degrade(&src, rng, side * scale, 1.0, psf, &self.profile)?;
        let half = buf as f64 / 2.0;
        Ok(warp(
            &src,
            &crop_affine([half, half], side * scale, self.crop as u32, rot),
            self.crop,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::synthetic::index as kp;

    fn scenes() -> Source {
        Source::Synthetic(
            SyntheticScenes::new(
                3,
                40,
                (160, 128),
                &["mouse".to_string(), "rat".to_string()],
                3,
                0,
                0.0,
            )
            .unwrap(),
        )
    }

    #[test]
    fn synthetic_samples_are_deterministic_and_consistent() {
        let src = scenes();
        let a = src.sample(5, 0).unwrap();
        let b = src.sample(5, 0).unwrap();
        assert_eq!(a.image, b.image);
        assert_eq!(a.state, b.state);
        let n = a.len();
        assert_eq!(a.keypoints.len(), n * K);
        assert_eq!(a.state.len(), n * K);
        assert_eq!(a.ids.len(), n);
        assert!(a.state.iter().all(|s| (0..=2).contains(s)));
        // A different variant views the same simulation (ids share the scene prefix).
        let c = src.sample(5, 1).unwrap();
        for id in a.ids.iter().chain(&c.ids) {
            assert_eq!(id / 16, a.ids.first().or(c.ids.first()).unwrap() / 16);
        }
    }

    #[test]
    fn pose_items_are_pure_functions_of_the_index() {
        let ds = PoseCropDataset::new(scenes(), 16, 64, 0.5, Profile::default());
        let a = ds.get(6).unwrap();
        let b = ds.get(6).unwrap();
        assert_eq!(a.crop, b.crop);
        assert_eq!(a.keypoints, b.keypoints);
        assert_eq!(a.crop.len(), 3 * 64 * 64);
        assert_eq!(a.keypoints.len(), K);
        for (s, p) in a.state.iter().zip(&a.keypoints) {
            if *s > 0 {
                assert!(p[0] >= 0.0 && p[0] <= 63.0 && p[1] >= 0.0 && p[1] <= 63.0);
            }
        }
        // Items 4n..4n+3 share one scene.
        let ids: Vec<i64> = (8..12).map(|i| ds.get(i).unwrap().id).collect();
        assert!(ids.iter().all(|id| id / 16 == ids[0] / 16), "{ids:?}");
    }

    #[test]
    fn unaugmented_crop_keeps_keypoints_on_the_animal() {
        let mut ds = PoseCropDataset::new(scenes(), 4, 64, 0.0, Profile::default());
        ds.augment = false;
        let item = ds.get(0).unwrap();
        // Without augmentation the crop is centred on the box: the visible points sit inside.
        let vis: Vec<&[f32; 2]> = item
            .keypoints
            .iter()
            .zip(&item.state)
            .filter(|(_, s)| **s == 2)
            .map(|(p, _)| p)
            .collect();
        assert!(!vis.is_empty());
        for p in vis {
            assert!(p[0] > 2.0 && p[0] < 61.0 && p[1] > 2.0 && p[1] < 61.0, "{p:?}");
        }
    }

    #[test]
    fn flip_swaps_left_and_right() {
        // `FLIP_INDEX` swaps the two eyes; mirror an item by hand and compare.
        assert_eq!(FLIP_INDEX[kp("left_eye")], kp("right_eye"));
        assert_eq!(FLIP_INDEX[kp("nose")], kp("nose"));
    }

    #[test]
    fn detection_items_letterbox_the_boxes() {
        let ds = DetectionDataset {
            source: scenes(),
            length: 4,
            size: (128, 160),
            augment: true,
            degrade: 0.5,
            zoom_out: 0.5,
            profile: Profile::default(),
        };
        let a = ds.get(3).unwrap();
        let b = ds.get(3).unwrap();
        assert_eq!(a.image, b.image);
        assert_eq!(a.image.len(), 3 * 128 * 160);
        for bx in &a.boxes {
            assert!(bx[0] < bx[2] && bx[1] < bx[3]);
            assert!(bx[0] >= -1.0 && bx[2] <= 161.0);
        }
        assert_eq!(a.boxes.len(), a.species.len());
    }

    #[test]
    fn mixed_source_draws_from_both() {
        let mixed = MixedSource::new(vec![scenes(), scenes()], &[1.0, 3.0], 0).unwrap();
        let picks: Vec<usize> = (0..200u64)
            .map(|i| keyed_rng(&[0, 99, i]).choose_weighted(&[0usize, 1], &mixed.p))
            .collect();
        let ones = picks.iter().filter(|k| **k == 1).count();
        assert!((110..190).contains(&ones), "{ones}");
        assert!(mixed.sample(3, 0).is_ok());
    }

    #[test]
    fn remap_and_mapping_tables() {
        let table = Mapping::Named("lp_crim13".into()).table().unwrap();
        assert_eq!(table["top_of_neck"], kp("neck"));
        let names = vec!["nose".to_string(), "unknown".to_string()];
        let rows = vec![vec![1.0, 2.0, 2.0], vec![3.0, 4.0, 2.0]];
        let (out, labeled) = remap_keypoints(&names, &rows, &table);
        assert_eq!(out[kp("nose")], vec![1.0, 2.0, 2.0]);
        assert!(labeled[kp("nose")]);
        assert_eq!(labeled.iter().filter(|v| **v).count(), 1);
        assert!(out[kp("neck")][0].is_nan());
        assert!(Mapping::Named("nope".into()).table().is_err());
    }
}
