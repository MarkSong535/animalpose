//! Unlabeled collections: `src/animalpose/frames.py:find_sources`, `is_held_out`, `Source` and the
//! `zlib.crc32` path hash that fixes the held-out split, so Rust and Python agree on which
//! recordings are held out and on the image folder names `label` writes.
//!
//! `// differs from Python:` frames are grey levels ([`crate::data::images::read_gray`]); TIF/BMP
//! frames (part of `frames.py:IMAGE_EXT`, so a folder of them is still listed as a recording and
//! still hashed into the split) cannot be decoded and are skipped like an unreadable image. A video
//! file is decoded through [`crate::data::images::read_frames`], whole, the first time it is used;
//! a video that cannot be decoded counts as zero frames, as `cv2.VideoCapture` reports for a file
//! it cannot open. Relative paths use `/` on every platform.

use std::path::{Component, Path, PathBuf};

use crate::algos::image::GrayF;
use crate::data::images::{read_frames, read_gray};

/// `frames.py:IMAGE_EXT`.
pub const IMAGE_EXT: [&str; 6] = [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"];

/// `frames.py:VIDEO_EXT`.
pub const VIDEO_EXT: [&str; 8] = [
    ".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv",
];

/// `zlib.crc32` (the IEEE polynomial, reflected, initial and final XOR `0xFFFFFFFF`).
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for b in bytes {
        crc ^= u32::from(*b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// What a recording is made of.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceKind {
    /// A video file.
    Video,
    /// A folder of image frames.
    Images,
}

/// `frames.py:Source`: one recording, plus its path relative to the walked root.
#[derive(Clone, Debug)]
pub struct Source {
    /// The video file or image folder.
    pub path: PathBuf,
    /// `os.path.relpath(path, root)` with `/` separators (`.` for the root itself).
    pub rel: String,
    /// Video or image folder.
    pub kind: SourceKind,
    /// The image files in order (`Images` only).
    pub files: Vec<PathBuf>,
    decoded: Option<Vec<GrayF>>,
}

impl Source {
    /// A video file or an image folder (`frames.py:open_source`), `rel` being its name.
    pub fn open(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        let rel = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());
        if path.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(&path)
                .map(|it| {
                    it.filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.is_file() && has_ext(p, &IMAGE_EXT))
                        .collect()
                })
                .unwrap_or_default();
            files.sort();
            Self::new(path, rel, SourceKind::Images, files)
        } else {
            Self::new(path, rel, SourceKind::Video, Vec::new())
        }
    }

    fn new(path: PathBuf, rel: String, kind: SourceKind, files: Vec<PathBuf>) -> Self {
        Self {
            path,
            rel,
            kind,
            files,
            decoded: None,
        }
    }

    fn video(&mut self) -> &[GrayF] {
        if self.decoded.is_none() {
            let frames = match read_frames(&self.path, None, 1) {
                Ok((frames, _)) => frames,
                Err(err) => {
                    eprintln!("cannot read {}: {err:#}", self.path.display());
                    Vec::new()
                }
            };
            self.decoded = Some(frames);
        }
        self.decoded.as_deref().unwrap_or_default()
    }

    /// `Source.n_frames`.
    pub fn n_frames(&mut self) -> usize {
        match self.kind {
            SourceKind::Images => self.files.len(),
            SourceKind::Video => self.video().len(),
        }
    }

    /// `Source.frames(indices)`: the readable frames at `indices` (a video's in sorted order, as
    /// the Python reader seeks them).
    pub fn frames(&mut self, indices: &[usize]) -> Vec<(usize, GrayF)> {
        match self.kind {
            SourceKind::Images => indices
                .iter()
                .filter_map(|i| {
                    let file = self.files.get(*i)?;
                    read_gray(file).ok().map(|f| (*i, f))
                })
                .collect(),
            SourceKind::Video => {
                let mut sorted = indices.to_vec();
                sorted.sort_unstable();
                let frames = self.video();
                sorted
                    .into_iter()
                    .filter_map(|i| frames.get(i).map(|f| (i, f.clone())))
                    .collect()
            }
        }
    }
}

fn has_ext(path: &Path, exts: &[&str]) -> bool {
    let ext = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default();
    exts.contains(&ext.as_str())
}

/// `frames.py:find_sources`: every video file and every folder that directly holds image files,
/// anywhere under `root`, in the order of `sorted(os.walk(root))` (by directory path string; in a
/// directory the image folder comes before its videos).
pub fn find_sources(root: impl AsRef<Path>) -> Vec<Source> {
    let root = root.as_ref();
    // (dirpath string as os.walk spells it, the path, its rel, its file names)
    let mut dirs: Vec<(String, PathBuf, String, Vec<String>)> = Vec::new();
    walk(root, &root.to_string_lossy(), ".", &mut dirs);
    dirs.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = Vec::new();
    for (_, dir, rel, mut names) in dirs {
        names.sort();
        let images: Vec<PathBuf> = names
            .iter()
            .map(|n| dir.join(n))
            .filter(|p| has_ext(p, &IMAGE_EXT))
            .collect();
        if !images.is_empty() {
            out.push(Source::new(
                dir.clone(),
                rel.clone(),
                SourceKind::Images,
                images,
            ));
        }
        for n in &names {
            let p = dir.join(n);
            if has_ext(&p, &VIDEO_EXT) {
                out.push(Source::new(p, join_rel(&rel, n), SourceKind::Video, vec![]));
            }
        }
    }
    out
}

fn join_rel(rel: &str, name: &str) -> String {
    if rel == "." {
        name.to_string()
    } else {
        format!("{rel}/{name}")
    }
}

fn walk(
    dir: &Path,
    spelled: &str,
    rel: &str,
    out: &mut Vec<(String, PathBuf, String, Vec<String>)>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return; // os.walk ignores directories it cannot list
    };
    let mut files = Vec::new();
    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let path = entry.path();
        if path.is_dir() {
            // os.walk lists a symlinked folder but does not descend into it.
            let link = entry.file_type().map(|t| t.is_symlink()).unwrap_or(false);
            if !link {
                subdirs.push(name);
            }
        } else {
            files.push(name);
        }
    }
    out.push((
        spelled.to_string(),
        dir.to_path_buf(),
        rel.to_string(),
        files,
    ));
    for name in subdirs {
        let child = if spelled.ends_with('/') {
            format!("{spelled}{name}")
        } else {
            format!("{spelled}/{name}")
        };
        walk(&dir.join(&name), &child, &join_rel(rel, &name), out);
    }
}

/// `frames.py:is_held_out`: about 1 in `every` recordings, by the CRC-32 of its relative path.
pub fn is_held_out(rel: &str, every: u32) -> bool {
    crc32(rel.as_bytes()) % every == 0
}

/// `selftrain.py:source_id`: `<crc32 of rel>_<last 40 characters of its sanitised stem>`.
pub fn source_id(rel: &str) -> String {
    let name = rel.rsplit('/').next().unwrap_or("");
    let name = if name == "." || name == ".." {
        ""
    } else {
        name
    };
    // `PurePath.stem`: drop the last suffix only when the dot is neither first nor last.
    let stem = match name.rfind('.') {
        Some(i) if i > 0 && i < name.len() - 1 => &name[..i],
        _ => name,
    };
    let mut clean = String::new();
    let mut in_run = false;
    for c in stem.chars() {
        if c.is_ascii_alphanumeric() {
            clean.push(c);
            in_run = false;
        } else if !in_run {
            clean.push('_');
            in_run = true;
        }
    }
    let tail = &clean[clean.len().saturating_sub(40)..];
    format!("{:08x}_{tail}", crc32(rel.as_bytes()))
}

/// `os.path.abspath`: joined with the working directory and normalised lexically.
pub fn abspath(path: &Path) -> String {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|d| d.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let mut parts: Vec<String> = Vec::new();
    let mut prefix = String::new();
    for c in joined.components() {
        match c {
            Component::Prefix(p) => prefix = p.as_os_str().to_string_lossy().to_string(),
            Component::RootDir | Component::CurDir => {}
            Component::ParentDir => {
                parts.pop();
            }
            Component::Normal(n) => parts.push(n.to_string_lossy().to_string()),
        }
    }
    format!("{prefix}/{}", parts.join("/"))
}

/// `np.linspace(lo, hi, num).astype(int)`: `num` evenly spaced values, truncated towards zero.
pub fn linspace_int(lo: f64, hi: f64, num: usize) -> Vec<usize> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![lo.max(0.0) as usize];
    }
    let step = (hi - lo) / (num - 1) as f64;
    (0..num)
        .map(|i| {
            let v = if i == num - 1 {
                hi
            } else {
                i as f64 * step + lo
            };
            v.max(0.0) as usize
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_zlib() {
        // zlib.crc32(b"cage_a"), zlib.crc32(b"123456789"), zlib.crc32(b"")
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn the_held_out_split_and_the_source_ids_follow_python() {
        // Python: [f"{zlib.crc32(n.encode()):08x}" for n in names], is_held_out = crc % 5 == 0
        let held: Vec<bool> = ["cage_a", "cage_b", "cage_d", "cage_e"]
            .iter()
            .map(|n| is_held_out(n, 5))
            .collect();
        assert_eq!(held, [false, false, true, true]);
        assert_eq!(
            source_id("cage_a"),
            format!("{:08x}_cage_a", crc32(b"cage_a"))
        );
        assert_eq!(
            source_id("sub dir/my video.v2.mp4"),
            format!("{:08x}_my_video_v2", crc32(b"sub dir/my video.v2.mp4"))
        );
        assert_eq!(source_id("."), format!("{:08x}_", crc32(b".")));
        let long = "a".repeat(50) + ".avi";
        assert_eq!(
            source_id(&long),
            format!("{:08x}_{}", crc32(long.as_bytes()), "a".repeat(40))
        );
    }

    #[test]
    fn linspace_truncates_like_numpy() {
        // np.linspace(0, 9, 4).astype(int) == [0, 3, 6, 9]; np.linspace(0, 23, 12).astype(int)
        assert_eq!(linspace_int(0.0, 9.0, 4), [0, 3, 6, 9]);
        assert_eq!(
            linspace_int(0.0, 23.0, 12),
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]
        );
        assert_eq!(linspace_int(5.0, 5.0, 1), [5]);
        assert!(linspace_int(0.0, 3.0, 0).is_empty());
    }

    #[test]
    fn walk_order_matches_sorted_os_walk() {
        let root = std::env::temp_dir().join(format!("ap-sources-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for d in ["b", "a/x", "a-b"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        for f in [
            "b/f0.png",
            "a/x/f0.jpg",
            "a-b/f.png",
            "z.mp4",
            "a/m.avi",
            "a/notes.txt",
        ] {
            std::fs::write(root.join(f), b"").unwrap();
        }
        let rels: Vec<String> = find_sources(&root).into_iter().map(|s| s.rel).collect();
        // sorted(os.walk): "<root>" < "<root>/a" < "<root>/a-b" < "<root>/a/x" < "<root>/b"
        assert_eq!(rels, ["z.mp4", "a/m.avi", "a-b", "a/x", "b"]);
        let _ = std::fs::remove_dir_all(&root);
    }
}
