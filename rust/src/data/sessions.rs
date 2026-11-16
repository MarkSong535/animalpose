//! Behavior session files: the `.npz` archives `data/behavior_sets.py:save_session` writes.
//!
//! An `.npz` is a zip archive of `.npy` members, so reading one needs a zip reader (the `zip` crate,
//! deflate only — no C library) and a small reader for the `.npy` format itself: the `\x93NUMPY`
//! magic, a little-endian header length and a python-dict header naming the element type and shape.
//! That covers every member `save_session` writes:
//!
//! ```text
//! coords       float32 (T, 25, 2)   canonical keypoints, NaN where undefined
//! conf         float32 (T, 25)
//! labels       int64   (T,)         an index into `label_names`, or IGNORE
//! fps          float64 ()
//! animal       <U9     ()
//! view         <U3     ()
//! label_names  <U8     (6,)         the classes `labels` indexes
//! ```
//!
//! ```
//! use animalpose::data::sessions::remap_labels;
//!
//! let src = ["sniffing", "other", "rearing", "resting", "grooming", "walking"];
//! let dst = ["resting", "sniffing", "walking", "turning", "rearing", "grooming", "other"];
//! // `data/behavior_sets.py:remap_labels` sends a name `dst` lacks to `IGNORE`, and keeps -100.
//! assert_eq!(
//!     remap_labels(&[0, 1, 3, -100], &src, &dst),
//!     [1, 6, 0, -100]
//! );
//! ```

use crate::data::synthetic::Rng;
use anyhow::{bail, ensure, Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// `data/tracks.py:IGNORE`, the label of a frame no annotator labelled.
pub const IGNORE: i64 = -100;

/// The magic every `.npy` file starts with (`\x93NUMPY`).
const NPY_MAGIC: &[u8] = b"\x93NUMPY";

/// The element type of a `.npy` array, as its `descr` spells it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// `<f4`, the type `save_session` stores coordinates in.
    F32,
    /// `<f8`, the type of the stored `fps`.
    F64,
    /// `<i8`, the type of `labels`.
    I64,
    /// `<i4`.
    I32,
    /// `<u1` / `|b1`, the type of a boolean or byte array.
    U8,
    /// `<U{n}`, a fixed-width UTF-32-LE string (numpy's unicode dtype).
    Unicode,
}

impl Dtype {
    /// Bytes per element.
    fn width(self, unicode_chars: usize) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::U8 => 1,
            Self::Unicode => 4 * unicode_chars.max(1),
        }
    }
}

/// The values of one `.npy` array.
#[derive(Clone, Debug, PartialEq)]
pub enum NpyData {
    /// Floating point values (`F32` / `F64`).
    Float(Vec<f32>),
    /// Integers (`I64` / `I32`).
    Int(Vec<i64>),
    /// Bytes or booleans (`U8` / `b1`).
    Bytes(Vec<u8>),
    /// One or more strings (`<U..`).
    Text(Vec<String>),
}

/// One `.npy` array: its shape and its values.
#[derive(Clone, Debug, PartialEq)]
pub struct NpyArray {
    /// Element count along every axis.
    pub shape: Vec<usize>,
    /// Row-major values.
    pub data: NpyData,
}

impl NpyArray {
    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// `true` when the array holds no element.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The values as `f32`, `None` for a text or byte array.
    pub fn floats(&self) -> Option<Vec<f32>> {
        match &self.data {
            NpyData::Float(v) => Some(v.clone()),
            NpyData::Int(v) => Some(v.iter().map(|i| *i as f32).collect()),
            NpyData::Bytes(_) | NpyData::Text(_) => None,
        }
    }

    /// The values as `i64`.
    pub fn ints(&self) -> Option<Vec<i64>> {
        match &self.data {
            NpyData::Int(v) => Some(v.clone()),
            NpyData::Float(v) => Some(v.iter().map(|f| *f as i64).collect()),
            NpyData::Bytes(v) => Some(v.iter().map(|b| i64::from(*b)).collect()),
            NpyData::Text(_) => None,
        }
    }

    /// A scalar string member (`animal`, `view`).
    pub fn text(&self) -> Option<&str> {
        match &self.data {
            NpyData::Text(v) => v.first().map(String::as_str),
            _ => None,
        }
    }

    /// A scalar float member (`fps`).
    pub fn scalar_f32(&self) -> Option<f32> {
        self.floats().filter(|v| v.len() == 1).map(|v| v[0])
    }

    /// The array reshaped to a `(T, n)` view of floats.
    pub fn matrix(&self) -> Option<(usize, usize, Vec<f32>)> {
        let data = self.floats()?;
        match self.shape.as_slice() {
            [rows, cols] => Some((*rows, *cols, data)),
            _ => None,
        }
    }
}

/// `data/behavior_sets.py:save_session`'s member names, read back from one file.
pub type Npz = BTreeMap<String, NpyArray>;

/// Reads one `.npy` file from disk.
pub fn read_npy(path: impl AsRef<Path>) -> Result<NpyArray> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    parse_npy(&bytes).with_context(|| format!("cannot parse {}", path.display()))
}

/// Parses an in-memory `.npy` image: magic, version, header dict, then the raw values.
pub fn parse_npy(bytes: &[u8]) -> Result<NpyArray> {
    ensure!(
        bytes.len() > 10 && bytes.starts_with(NPY_MAGIC),
        "not a .npy file (magic is {})",
        bytes
            .iter()
            .take(6)
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let (major, minor) = (bytes[6], bytes[7]);
    let (header_len, data_at) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        other => bail!("unsupported .npy format version {other}.{minor}"),
    };
    ensure!(
        bytes.len() >= data_at + header_len,
        ".npy header is truncated"
    );
    let header = String::from_utf8_lossy(&bytes[data_at..data_at + header_len]).into_owned();
    let descr = extract(&header, "descr").context(".npy header has no 'descr'")?;
    let fortran = extract(&header, "fortran_order")
        .map(|v| v.trim() == "True")
        .unwrap_or(false);
    ensure!(
        !fortran,
        "column-major (.npy fortran_order=True) arrays are not supported"
    );
    let shape = parse_shape(
        &extract(&header, "shape").with_context(|| format!(".npy header of a {descr} array"))?,
    )?;
    let (dtype, chars) = parse_descr(&descr)?;
    let raw = &bytes[data_at + header_len..];
    let element = dtype.width(chars);
    let count = shape.iter().product::<usize>();
    ensure!(
        raw.len() >= count * element,
        ".npy body holds {} bytes, {} x {element} expected",
        raw.len(),
        count
    );
    let data = decode_values(dtype, chars, &raw[..count * element])?;
    Ok(NpyArray { shape, data })
}

/// Reads an `.npz` archive (deflate or stored members) into its arrays, keyed by member name
/// without the `.npy` suffix.
pub fn read_npz(path: impl AsRef<Path>) -> Result<Npz> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut archive =
        zip::ZipArchive::new(file).with_context(|| format!("cannot open {}", path.display()))?;
    let mut out = Npz::new();
    for i in 0..archive.len() {
        let mut member = archive.by_index(i)?;
        let name = member.name().trim_end_matches(".npy").to_string();
        if member.is_dir() {
            continue;
        }
        ensure!(
            matches!(
                member.compression(),
                zip::CompressionMethod::Deflated | zip::CompressionMethod::Stored
            ),
            "{}: member {name} uses {}, which this port cannot read",
            path.display(),
            member.compression()
        );
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut member, &mut bytes)?;
        out.insert(
            name,
            parse_npy(&bytes).with_context(|| format!("{}", path.display()))?,
        );
    }
    Ok(out)
}

/// The `'key': value` entry of a `.npy` header dict, or `None` when the key is absent.
fn extract(header: &str, key: &str) -> Option<String> {
    let needle = format!("'{key}':");
    let at = header.find(&needle)? + needle.len();
    let rest = &header[at..];
    // The value runs to the comma that closes it, ignoring the one inside a shape tuple.
    let end = if key == "shape" {
        rest.find(')')? + 1
    } else {
        rest.find(',')
            .or_else(|| rest.find('}'))
            .unwrap_or(rest.len())
    };
    Some(rest[..end].trim().to_string())
}

/// The shape tuple `(14400, 25, 2)` / `(14400,)` / `()`.
fn parse_shape(text: &str) -> Result<Vec<usize>> {
    let inner = text
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim()
        .trim_end_matches(',')
        .trim()
        .to_string();
    let inner = inner.trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|p| {
            let p = p.trim();
            if p.is_empty() {
                Ok(0)
            } else {
                p.parse::<usize>()
                    .map_err(|err| anyhow::anyhow!("bad .npy shape {text:?}: {err}"))
            }
        })
        .collect()
}

/// `'<f4'`-style descriptors to a [`Dtype`] and, for unicode, its fixed character width.
fn parse_descr(descr: &str) -> Result<(Dtype, usize)> {
    let descr = descr.trim().trim_matches('\'');
    let (order, rest) = descr.split_at(1);
    if order == ">" {
        bail!("big-endian .npy dtype {descr:?} is not supported");
    }
    Ok(match rest {
        "f4" => (Dtype::F32, 0),
        "f8" => (Dtype::F64, 0),
        "i8" => (Dtype::I64, 0),
        "i4" => (Dtype::I32, 0),
        "u1" | "b1" => (Dtype::U8, 0),
        other if other.starts_with('U') => (
            Dtype::Unicode,
            other[1..]
                .parse::<usize>()
                .with_context(|| format!("bad .npy dtype {descr:?}"))?,
        ),
        other if other.starts_with('S') => (Dtype::U8, 0),
        other => bail!("unsupported .npy dtype {other:?}"),
    })
}

/// Decodes `raw` as `dtype`.
fn decode_values(dtype: Dtype, chars: usize, raw: &[u8]) -> Result<NpyData> {
    Ok(match dtype {
        Dtype::F32 => NpyData::Float(
            raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        ),
        Dtype::F64 => NpyData::Float(
            raw.chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect(),
        ),
        Dtype::I64 => NpyData::Int(
            raw.chunks_exact(8)
                .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                .collect(),
        ),
        Dtype::I32 => NpyData::Int(
            raw.chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
                .collect(),
        ),
        Dtype::U8 => NpyData::Bytes(raw.to_vec()),
        Dtype::Unicode => {
            let stride = 4 * chars.max(1);
            let mut items = Vec::with_capacity(raw.len() / stride);
            for chunk in raw.chunks_exact(stride) {
                let text = decode_utf32(chunk)?;
                items.push(text.trim_end_matches('\0').to_string());
            }
            NpyData::Text(items)
        }
    })
}

/// Decodes one fixed-width UTF-32-LE field.
fn decode_utf32(chunk: &[u8]) -> Result<String> {
    let mut out = String::new();
    let mut buf = [0u8; 4];
    for code in chunk.chunks_exact(4) {
        buf.copy_from_slice(code);
        let value = u32::from_le_bytes(buf);
        let Some(ch) = char::from_u32(value) else {
            bail!("invalid UTF-32 code point {value:#x} in a .npy string");
        };
        out.push(ch);
    }
    Ok(out)
}

/// One behavior session: a keypoint track with frame-wise labels.
///
/// Port of `data/tracks.py:Session`, whose fields `save_session` stores next to the label vocabulary
/// `labels` indexes (`label_names`), so any label set can be remapped later.
#[derive(Clone, Debug, PartialEq)]
pub struct Session {
    /// The animal group, e.g. `cai_train`.
    pub animal: String,
    /// The camera the recording came from (`top`, `side`, `cage`).
    pub view: String,
    /// Frames of the track.
    pub frames: usize,
    /// Keypoints per frame (25 for the canonical vocabulary).
    pub keypoints: usize,
    /// Coordinates, `(frames, keypoints, 2)` row-major; `NaN` where undefined.
    pub coords: Vec<f32>,
    /// Keypoint confidences, `(frames, keypoints)` row-major.
    pub conf: Vec<f32>,
    /// Per-frame class, an index into [`Session::label_names`] or [`IGNORE`].
    pub labels: Vec<i64>,
    /// Frame rate of the stored frames.
    pub fps: f64,
    /// The classes [`Session::labels`] indexes.
    pub label_names: Vec<String>,
}

impl Session {
    /// `load_session_dir`'s per-file read: the members `save_session` wrote.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let npz = read_npz(path)?;
        let want = |name: &str| -> Result<&NpyArray> {
            npz.get(name)
                .with_context(|| format!("{} has no {name} member", path.display()))
        };
        let coords = want("coords")?;
        let shape = &coords.shape;
        ensure!(
            shape.len() == 3 && shape[2] == 2,
            "{}: coords has shape {shape:?}, expected (T, K, 2)",
            path.display()
        );
        let conf = want("conf")?;
        ensure!(
            conf.shape == vec![shape[0], shape[1]],
            "{}: conf has shape {:?}, expected ({}, {})",
            path.display(),
            conf.shape,
            shape[0],
            shape[1]
        );
        let labels = want("labels")?.ints().context("labels is not numeric")?;
        ensure!(
            labels.len() == shape[0],
            "{}: {} labels for {} frames",
            path.display(),
            labels.len(),
            shape[0]
        );
        Ok(Self {
            animal: want("animal")?.text().unwrap_or("").to_string(),
            view: want("view")?.text().unwrap_or("").to_string(),
            frames: shape[0],
            keypoints: shape[1],
            coords: coords.floats().context("coords is not numeric")?,
            conf: conf.floats().context("conf is not numeric")?,
            labels,
            fps: want("fps")?.scalar_f32().context("fps is not a scalar")? as f64,
            label_names: match npz.get("label_names") {
                Some(names) => match &names.data {
                    NpyData::Text(items) => items.clone(),
                    _ => bail!("{}: label_names is not a text array", path.display()),
                },
                // Files written before the vocabulary was stored default to `BEHAVIORS_EXT`.
                None => vec![
                    "resting".into(),
                    "walking".into(),
                    "rearing".into(),
                    "grooming".into(),
                    "eating".into(),
                    "other".into(),
                ],
            },
        })
    }

    /// `remap_labels(s.labels, s.label_names, classes)`: the labels as indices of `classes`.
    ///
    /// Port of the loop `train/loops.py:train_behavior` runs over every session.
    pub fn with_classes(&self, classes: &[&str]) -> Session {
        let names: Vec<&str> = self.label_names.iter().map(String::as_str).collect();
        let mut out = self.clone();
        out.labels = remap_labels(&self.labels, &names, classes);
        out.label_names = classes.iter().map(|s| (*s).to_string()).collect();
        out
    }

    /// Every `step`-th frame, as `load_session_dir` resamples a 30 fps session to `target_fps`.
    pub fn stride(&self, step: usize) -> Session {
        let step = step.max(1);
        let mut out = self.clone();
        out.frames = self.frames.div_ceil(step);
        out.coords = (0..out.frames)
            .flat_map(|i| {
                let at = i * step;
                self.coords[at * self.keypoints * 2..][..self.keypoints * 2].to_vec()
            })
            .collect();
        out.conf = (0..out.frames)
            .flat_map(|i| {
                let at = i * step;
                self.conf[at * self.keypoints..][..self.keypoints].to_vec()
            })
            .collect();
        out.labels = (0..out.frames).map(|i| self.labels[i * step]).collect();
        out.fps = self.fps / step as f64;
        out
    }
}

/// `data/behavior_sets.py:remap_labels`: class indices in `src` become indices in `dst`.
///
/// A name `dst` does not list becomes [`IGNORE`]; an already-ignored label stays ignored. The
/// out-of-range clamp (`np.clip(lab, 0, len(src))`) maps the extra `src.len()` slot of numpy's
/// lookup table to `IGNORE` as well.
pub fn remap_labels(labels: &[i64], src: &[&str], dst: &[&str]) -> Vec<i64> {
    let table: Vec<i64> = src
        .iter()
        .map(|name| {
            dst.iter()
                .position(|d| d == name)
                .map_or(IGNORE, |i| i as i64)
        })
        .chain(std::iter::once(IGNORE))
        .collect();
    labels
        .iter()
        .map(|&lab| {
            if lab == IGNORE {
                IGNORE
            } else {
                let at = lab.clamp(0, src.len() as i64) as usize;
                table[at]
            }
        })
        .collect()
}

/// `data/behavior_sets.py:load_session_dir`: every `*.npz` under `root`, resampled to `target_fps`
/// and with its labels remapped onto `classes`.
///
/// `groups` keeps only the listed animal groups (`data.session_groups`).
pub fn load_session_dir(
    root: impl AsRef<Path>,
    target_fps: f64,
    groups: Option<&[String]>,
    classes: &[&str],
) -> Result<Vec<Session>> {
    let root = root.as_ref();
    let mut files = Vec::new();
    collect_npz(root, &mut files)?;
    files.sort();
    let mut out = Vec::new();
    for path in files {
        let session = Session::read(&path)?;
        if let Some(groups) = groups {
            if !groups.contains(&session.animal) {
                continue;
            }
        }
        let step = (session.fps / target_fps).round().max(1.0) as usize;
        out.push(session.with_classes(classes).stride(step));
    }
    Ok(out)
}

/// Every `.npz` file under `root`, recursively (`Path.rglob("*.npz")`).
fn collect_npz(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("cannot list {}", dir.display()))?
        .flatten()
    {
        let path = entry.path();
        if path.is_dir() {
            collect_npz(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "npz") {
            out.push(path);
        }
    }
    Ok(())
}

/// `data/tracks.py:behavior_windows`: random labeled windows from real sessions.
///
/// Returns `(coords, mask, labels)` for `batch` windows of `window` frames; a window needs at least
/// `min_labeled` labelled frames, and the loop gives up after `batch * 50` tries exactly like
/// Python, so a short batch is possible and reported by the returned length.
pub fn behavior_windows(
    sessions: &[Session],
    batch: usize,
    window: usize,
    rng: &mut Rng,
    conf_thresh: f64,
    min_labeled: f64,
) -> Option<(Vec<f32>, Vec<f32>, Vec<i64>)> {
    if sessions.is_empty() || batch == 0 || window == 0 {
        return None;
    }
    let mut xs = Vec::new();
    let mut ms = Vec::new();
    let mut ys = Vec::new();
    let mut tries = 0usize;
    while xs.len() < batch && tries < batch * 50 {
        tries += 1;
        let s = &sessions[rng.integers(0, sessions.len() as i64) as usize];
        if s.frames <= window {
            continue;
        }
        let t0 = rng.integers(0, (s.frames - window) as i64) as usize;
        let labels = &s.labels[t0..t0 + window];
        let labeled = labels.iter().filter(|y| **y != IGNORE).count();
        if (labeled as f64 / window as f64) < min_labeled {
            continue;
        }
        let mut coords = vec![0.0f32; window * s.keypoints * 2];
        let mut mask = vec![0.0f32; window * s.keypoints];
        for t in 0..window {
            for k in 0..s.keypoints {
                let src = (t0 + t) * s.keypoints + k;
                let observed = (f64::from(s.conf[src]) >= conf_thresh)
                    && s.coords[src * 2].is_finite()
                    && s.coords[src * 2 + 1].is_finite();
                mask[t * s.keypoints + k] = f32::from(observed);
                for d in 0..2 {
                    // `np.nan_to_num`: an undefined point enters the batch as 0.
                    coords[(t * s.keypoints + k) * 2 + d] =
                        crate::data::synthetic::nan_to_num(s.coords[src * 2 + d]);
                }
            }
        }
        xs.push(coords);
        ms.push(mask);
        ys.push(labels.to_vec());
    }
    if xs.is_empty() {
        return None;
    }
    let labels = ys.into_iter().flatten().collect();
    Some((xs.concat(), ms.concat(), labels))
}

/// Serialises an array back to the `.npy` layout `numpy.save` writes (version 1.0 header).
///
/// The inverse of [`parse_npy`]; the dtype is chosen to match the variant, exactly as
/// `data/tracks.py:save_session` lets `numpy.save` infer it from the array it is handed.
pub fn encode_npy(array: &NpyArray) -> Vec<u8> {
    let (descr, body, element) = match &array.data {
        NpyData::Float(v) => {
            let body: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            ("<f4".to_string(), body, 4)
        }
        NpyData::Int(v) => {
            let body: Vec<u8> = v.iter().flat_map(|i| i.to_le_bytes()).collect();
            ("<i8".to_string(), body, 8)
        }
        NpyData::Bytes(v) => ("<u1".to_string(), v.clone(), 1),
        NpyData::Text(v) => {
            let chars = v.iter().map(|s| s.chars().count()).max().unwrap_or(1);
            let mut body = Vec::with_capacity(v.len() * chars * 4);
            for s in v {
                for ch in s.chars() {
                    body.extend_from_slice(&(ch as u32).to_le_bytes());
                }
                for _ in s.chars().count()..chars {
                    body.extend_from_slice(&0u32.to_le_bytes());
                }
            }
            (format!("<U{chars}"), body, 4 * chars)
        }
    };
    let shape = if array.shape.len() == 1 {
        format!("({},)", array.shape[0])
    } else {
        format!(
            "({})",
            array
                .shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    // The 10-byte preamble plus the header must be a multiple of 64 bytes, as numpy pads it.
    let pad = 64 - (10 + header.len() + 1) % 64;
    for _ in 0..pad {
        header.push(' ');
    }
    let header_len = header.len() + 1; // + the trailing newline
    let mut out = Vec::with_capacity(10 + header_len + body.len());
    out.extend_from_slice(NPY_MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header_len as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(&body);
    let _ = element;
    out
}

/// Writes an `.npz` archive: one deflate-compressed `.npy` member per entry, in the order given.
///
/// The writer counterpart of [`read_npz`], matching `numpy.savez_compressed`, which is what
/// `pipeline.py:VideoResult.to_npz` and `data/tracks.py:save_session` produce.
pub fn write_npz(path: impl AsRef<Path>, entries: &[(&str, &NpyArray)]) -> Result<()> {
    use std::io::Write as _;
    use zip::write::SimpleFileOptions;

    let path = path.as_ref();
    let file =
        std::fs::File::create(path).with_context(|| format!("cannot write {}", path.display()))?;
    let mut zip = zip::ZipWriter::new(std::io::BufWriter::new(file));
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (name, array) in entries {
        zip.start_file(format!("{name}.npy"), options)
            .with_context(|| format!("cannot add {name}.npy to {}", path.display()))?;
        zip.write_all(&encode_npy(array))?;
    }
    zip.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An auto-labelled session's six classes onto the seven classes the general model head has.
    #[test]
    fn remap_matches_the_general_classes() {
        let src = [
            "sniffing", "other", "rearing", "resting", "grooming", "walking",
        ];
        let dst = [
            "resting", "sniffing", "walking", "turning", "rearing", "grooming", "other",
        ];
        assert_eq!(
            remap_labels(&[0, 1, 2, 3, 4, 5, IGNORE], &src, &dst),
            [1, 6, 4, 0, 5, 2, IGNORE]
        );
        // A class the target list lacks (here `eating`, index 4 of BEHAVIORS_EXT) is ignored.
        let ext = [
            "resting", "walking", "rearing", "grooming", "eating", "other",
        ];
        let dst2 = ["resting", "walking"];
        assert_eq!(
            remap_labels(&[0, 1, 4, 5], &ext, &dst2),
            [0, 1, IGNORE, IGNORE]
        );
        // `np.clip(lab, 0, len(src))` sends an out-of-range label to the table's last entry.
        assert_eq!(remap_labels(&[99], &ext, &dst2), [IGNORE]);
    }

    /// A stored header round-trips: the members `save_session` writes come back with their values.
    #[test]
    fn npy_round_trip() {
        let bytes = npy_f32(&[1.0, -2.5, f32::NAN], &[3]);
        let array = parse_npy(&bytes).unwrap();
        assert_eq!(array.shape, [3]);
        let values = array.floats().unwrap();
        assert_eq!(values[0], 1.0);
        assert!(values[2].is_nan());
    }

    /// The dtype spellings a session file uses all decode.
    #[test]
    fn npy_dtypes() {
        assert_eq!(parse_descr("'<f4'").unwrap(), (Dtype::F32, 0));
        assert_eq!(parse_descr("<f8").unwrap(), (Dtype::F64, 0));
        assert_eq!(parse_descr("'<U9'").unwrap(), (Dtype::Unicode, 9));
        assert_eq!(parse_descr("|b1").unwrap(), (Dtype::U8, 0));
        assert!(parse_descr("'>f4'").is_err());
        assert!(parse_descr("'<c8'").is_err());
        assert_eq!(parse_shape("(14400, 25, 2)").unwrap(), [14400, 25, 2]);
        assert_eq!(parse_shape("()").unwrap(), Vec::<usize>::new());
        assert_eq!(parse_shape("(6,)").unwrap(), [6]);
    }

    /// `read_session` on a file this test writes: the same members `save_session` stores.
    #[test]
    fn session_reads_what_save_session_writes() {
        let dir = std::env::temp_dir().join("animalpose-session-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("abc.npz");
        write_fixture_npz(&path);
        let s = Session::read(&path).unwrap();
        assert_eq!(s.animal, "cai_train");
        assert_eq!(s.view, "top");
        assert_eq!((s.frames, s.keypoints), (3, 2));
        assert_eq!(s.fps, 30.0);
        assert_eq!(s.label_names, ["sniffing", "resting", "walking"]);
        assert_eq!(s.conf, vec![0.9, 0.1, 0.8, 0.7, 0.0, 0.5]);
        assert_eq!(s.labels, [0, 1, IGNORE]);
        assert!(s.coords[4].is_nan(), "an undefined point stays NaN");
        // `train_behavior`'s remap: stored class names are looked up in the head's `classes`, and
        // a name the head does not list becomes `IGNORE`.
        assert_eq!(
            s.with_classes(&["resting", "sniffing"]).labels,
            [1, 0, IGNORE]
        );
        assert_eq!(s.with_classes(&["sniffing"]).labels, [0, IGNORE, IGNORE]);
        // `load_session_dir` resamples 30 fps to 15 fps (`step = 2`).
        let sessions =
            load_session_dir(&dir, 15.0, None, &["resting", "sniffing", "walking"]).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].frames, 2);
        assert_eq!(sessions[0].fps, 15.0);
        assert_eq!(sessions[0].labels, [1, IGNORE]); // `sniffing` is class 1 of the three
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `behavior_windows` needs a labeled window and never returns an ignored-only batch.
    #[test]
    fn behavior_windows_never_labels_everything() {
        let dir = std::env::temp_dir().join("animalpose-windows-test");
        std::fs::create_dir_all(&dir).unwrap();
        write_fixture_npz(&dir.join("abc.npz"));
        let mut sessions =
            load_session_dir(&dir, 30.0, None, &["sniffing", "resting", "walking"]).unwrap();
        sessions[0].frames = 6;
        sessions[0].labels = vec![0, 1, IGNORE, 2, 0, 1];
        sessions[0].coords = vec![1.0f32; 6 * 2 * 2];
        sessions[0].conf = vec![0.9f32; 6 * 2];
        let mut rng = Rng::new(7);
        let (x, m, y) = behavior_windows(&sessions, 4, 4, &mut rng, 0.6, 0.25).unwrap();
        assert_eq!(x.len(), 4 * 4 * 2 * 2);
        assert_eq!(m.len(), 4 * 4 * 2);
        assert_eq!(y.len(), 4 * 4);
        assert!(m.iter().all(|v| *v == 1.0));
        // A window that is mostly unlabeled is rejected: with only one labeled class left the loop
        // can still fill the batch, so ask for more than the file can give.
        assert!(behavior_windows(&sessions, 4, 5, &mut rng, 0.6, 0.99).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One `.npy` image of an `f32` array, as numpy writes it (v1.0 header).
    fn npy_f32(values: &[f32], shape: &[usize]) -> Vec<u8> {
        let dims = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let dims = if shape.len() == 1 {
            format!("{dims},")
        } else {
            dims
        };
        let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}), }}");
        let pad = (10 + header.len() + 1).next_multiple_of(64) - (10 + header.len());
        header.push_str(&" ".repeat(pad));
        header.push('\n');
        let mut out = Vec::new();
        out.extend_from_slice(NPY_MAGIC);
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Writes an `.npz` with the member names and dtypes `save_session` produces, deflate-compressed.
    fn write_fixture_npz(path: &Path) {
        use zip::write::SimpleFileOptions;
        let file = std::fs::File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let opts =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        let mut add = |name: &str, bytes: &[u8]| {
            zip.start_file(format!("{name}.npy"), opts).unwrap();
            std::io::Write::write_all(&mut zip, bytes).unwrap();
        };
        // coords: (3, 2, 2) float32, with one NaN pair.
        let mut coords: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        coords[4] = f32::NAN;
        coords[5] = f32::NAN;
        add("coords", &npy_f32(&coords, &[3, 2, 2]));
        add("conf", &npy_f32(&[0.9, 0.1, 0.8, 0.7, 0.0, 0.5], &[3, 2]));
        add("labels", &npy_i64(&[0, 1, IGNORE], &[3]));
        add("fps", &npy_f64(&[30.0], &[]));
        add("animal", &npy_text("cai_train", &[]));
        add("view", &npy_text("top", &[]));
        add(
            "label_names",
            &npy_texts(&["sniffing", "resting", "walking"], &[3]),
        );
        zip.finish().unwrap();
    }

    /// The `.npy` header of one dtype and shape.
    fn npy_header(descr: &str, shape: &[usize], body_len: usize) -> Vec<u8> {
        let dims = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let dims = if shape.len() == 1 {
            format!("{dims},")
        } else {
            dims
        };
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({dims}), }}");
        let pad = (10 + header.len() + 1).next_multiple_of(64) - (10 + header.len());
        header.push_str(&" ".repeat(pad));
        header.push('\n');
        let _ = body_len;
        let mut out = Vec::new();
        out.extend_from_slice(NPY_MAGIC);
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out
    }

    fn npy_i64(values: &[i64], shape: &[usize]) -> Vec<u8> {
        let mut out = npy_header("<i8", shape, values.len() * 8);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn npy_f64(values: &[f64], shape: &[usize]) -> Vec<u8> {
        let mut out = npy_header("<f8", shape, values.len() * 8);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// One `<U{n}` scalar: `n` UTF-32-LE code points, zero padded.
    fn npy_text(text: &str, shape: &[usize]) -> Vec<u8> {
        let chars = text.chars().count();
        let mut out = npy_header(&format!("<U{chars}"), shape, chars * 4);
        for ch in text.chars() {
            out.extend_from_slice(&(ch as u32).to_le_bytes());
        }
        out
    }

    fn npy_texts(items: &[&str], shape: &[usize]) -> Vec<u8> {
        let chars = items.iter().map(|s| s.chars().count()).max().unwrap_or(1);
        let mut out = npy_header(&format!("<U{chars}"), shape, items.len() * chars * 4);
        for item in items {
            for i in 0..chars {
                let code = item.chars().nth(i).map_or(0, |c| c as u32);
                out.extend_from_slice(&code.to_le_bytes());
            }
        }
        out
    }
    /// `write_npz` is the inverse of `read_npz`: the member names, shapes and dtypes survive.
    #[test]
    fn npz_round_trips() {
        let dir = std::env::temp_dir().join(format!("animalpose-npz-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("round.npz");
        let arrays = [
            (
                "coords",
                NpyArray {
                    shape: vec![2, 3, 2],
                    data: NpyData::Float((0..12).map(|v| v as f32).collect()),
                },
            ),
            (
                "frames",
                NpyArray {
                    shape: vec![3],
                    data: NpyData::Int(vec![0, 5, 9]),
                },
            ),
            (
                "keypoint_names",
                NpyArray {
                    shape: vec![2],
                    data: NpyData::Text(vec!["nose".to_string(), "tail_base".to_string()]),
                },
            ),
        ];
        let refs: Vec<(&str, &NpyArray)> = arrays.iter().map(|(n, a)| (*n, a)).collect();
        write_npz(&path, &refs).expect("the archive is written");
        let back = read_npz(&path).expect("the archive is read");
        assert_eq!(back["coords"].shape, vec![2, 3, 2]);
        match &back["coords"].data {
            NpyData::Float(v) => assert_eq!(v.len(), 12),
            other => panic!("coords must stay float, got {other:?}"),
        }
        assert_eq!(back["frames"].data, NpyData::Int(vec![0, 5, 9]));
        assert_eq!(
            back["keypoint_names"].data,
            NpyData::Text(vec!["nose".to_string(), "tail_base".to_string()])
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
