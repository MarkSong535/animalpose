//! `animalpose data download`: the public datasets into `data/raw/`, every file checked against
//! its published MD5 — the port of `src/animalpose/cli/data/download.py`.
//!
//! Sources are resolved through the figshare and Zenodo APIs at run time; files already present
//! with the right checksum are skipped, so the command can be re-run after an interruption.
//! Archives are extracted into `<dataset>/extracted/` (`.zip` through the `zip` crate, `.tar.gz`
//! through `flate2` and the tar reader below).
//!
//! `// differs from Python:` HTTP goes through the `curl` executable (shipped with macOS, Windows
//! 10+ and every Linux distribution) instead of `urllib`: a pure-Rust TLS stack needs `ring` or
//! `aws-lc`, which compile C and assembly, and this crate must build with nothing but a Rust
//! toolchain. MD5 and SHA-256 are implemented here for the same reason.

use anyhow::{bail, Context, Result};
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command as Process;

/// `RAT7M_MOCAP`: the figshare article with the motion capture and camera parameters.
pub const RAT7M_MOCAP: u64 = 13_739_233;
/// `RAT7M_VIDEOS`: figshare articles of the videos per subject/day.
pub const RAT7M_VIDEOS: [(&str, u64); 2] = [("s1-d1", 13_751_023), ("s2-d1", 13_753_417)];
/// `RAT7M_CHUNKS`: the chunk start frames used here.
pub const RAT7M_CHUNKS: [u64; 8] = [0, 17500, 35000, 52500, 70000, 87500, 105000, 122500];

/// Where one dataset comes from (`DATASETS[name][0:2]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A figshare article; the file whose name `pick` accepts.
    Figshare(u64),
    /// A Zenodo record.
    Zenodo(u64),
    /// Rat 7M: mocap from [`RAT7M_MOCAP`], video chunks of one subject/day.
    Rat7m(&'static str),
    /// Direct URLs without published checksums ([`direct_files`]).
    Direct,
}

/// `DATASETS`: `(name, kind, folder under data/raw)`, in the order `--all` downloads them.
pub const DATASETS: [(&str, Kind, &str); 7] = [
    ("crim13", Kind::Figshare(24_993_384), "lp_crim13"),
    (
        "mirror_mouse",
        Kind::Figshare(24_993_315),
        "lp_mirror_mouse",
    ),
    ("rat7m_s1", Kind::Rat7m("s1-d1"), "rat7m"),
    ("rat7m_s2", Kind::Rat7m("s2-d1"), "rat7m"),
    ("irodent", Kind::Zenodo(8_250_392), "irodent"),
    ("mit_homecage", Kind::Direct, "mit_homecage"),
    ("abode", Kind::Direct, "abode"),
];

/// `DATASETS[name][3]`: which files of an article/record are wanted.
fn pick(name: &str, file: &str) -> bool {
    match name {
        "crim13" => file == "crim13.zip",
        "mirror_mouse" => file == "mirror-mouse.zip",
        "irodent" => file.ends_with(".tar.gz"),
        _ => true,
    }
}

/// One remote file.
#[derive(Clone, Debug, PartialEq)]
pub struct Remote {
    /// File name.
    pub name: String,
    /// Download URL.
    pub url: String,
    /// Published MD5 (hex), empty when the source publishes none.
    pub md5: String,
    /// Size in bytes (`None` when unknown).
    pub size: Option<u64>,
}

/// `DIRECT`: sources without published checksums; the size is checked and the SHA-256 printed.
pub fn direct_files(name: &str) -> Vec<Remote> {
    let r = |name: &str, size: Option<u64>, url: &str| Remote {
        name: name.into(),
        url: url.into(),
        md5: String::new(),
        size,
    };
    match name {
        // Jhuang et al. 2010, cbmm.mit.edu/mouse-dataset. No license stated; cite the paper.
        "mit_homecage" => vec![r(
            "full_database.zip",
            Some(886_000_000),
            "https://www.dropbox.com/s/1vo4jcdbslgsfzb/full_database.zip?dl=1",
        )],
        // ABODe, Edinburgh DataShare 10.7488/ds/3833, CC BY 4.0.
        "abode" => vec![
            r(
                "Annotations.df",
                Some(14_247),
                "https://datashare.ed.ac.uk/bitstreams/ea0c5254-000d-42fe-81a3-692bd16bd64d/download",
            ),
            r(
                "Datasplits.df",
                None,
                "https://datashare.ed.ac.uk/bitstreams/4ab36ed7-44d4-4cef-aa67-d2f2fc6f9eb0/download",
            ),
            r(
                "README.md",
                None,
                "https://datashare.ed.ac.uk/bitstreams/c205c586-8a63-4a69-a22a-34b77dd25d44/download",
            ),
            r(
                "Snippets.zip",
                Some(5_049_000_000),
                "https://datashare.ed.ac.uk/bitstreams/1d40c69a-427c-4153-9806-c5984fbfb8ec/download",
            ),
        ],
        _ => Vec::new(),
    }
}

/// `curl -fsSL url`: the body as bytes.
fn http_get(url: &str) -> Result<Vec<u8>> {
    let out = Process::new("curl")
        .args(["-fsSL", "--retry", "3", "--max-time", "60", url])
        .output()
        .context("cannot run curl (needed for downloads)")?;
    if !out.status.success() {
        bail!("{url}: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(out.stdout)
}

/// `curl -fL -o dest url`, with curl's progress meter on stderr.
fn http_download(url: &str, dest: &Path) -> Result<()> {
    let status = Process::new("curl")
        .args(["-fL", "--retry", "3", "-#", "-o"])
        .arg(dest)
        .arg(url)
        .status()
        .context("cannot run curl (needed for downloads)")?;
    if !status.success() {
        bail!("download of {url} failed ({status})");
    }
    Ok(())
}

fn json(url: &str) -> Result<serde_json::Value> {
    serde_json::from_slice(&http_get(url)?).with_context(|| format!("{url} did not return JSON"))
}

/// `figshare_files(article)`.
pub fn figshare_files(article: u64) -> Result<Vec<Remote>> {
    let files = json(&format!(
        "https://api.figshare.com/v2/articles/{article}/files?page_size=1000"
    ))?;
    Ok(files
        .as_array()
        .context("figshare: expected a file list")?
        .iter()
        .map(|f| Remote {
            name: f["name"].as_str().unwrap_or_default().into(),
            url: f["download_url"].as_str().unwrap_or_default().into(),
            md5: f["computed_md5"].as_str().unwrap_or_default().into(),
            size: f["size"].as_u64(),
        })
        .collect())
}

/// `zenodo_files(record)`.
pub fn zenodo_files(record: u64) -> Result<Vec<Remote>> {
    let rec = json(&format!("https://zenodo.org/api/records/{record}"))?;
    Ok(rec["files"]
        .as_array()
        .context("zenodo: expected a file list")?
        .iter()
        .map(|f| Remote {
            name: f["key"].as_str().unwrap_or_default().into(),
            url: f["links"]["self"].as_str().unwrap_or_default().into(),
            md5: f["checksum"]
                .as_str()
                .unwrap_or_default()
                .split_once(':')
                .map(|(_, h)| h.to_string())
                .unwrap_or_default(),
            size: f["size"].as_u64(),
        })
        .collect())
}

fn hash_file<H: Hasher>(path: &Path, mut h: H) -> Result<String> {
    let mut f =
        std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finish()))
}

/// `md5(path)`: the hex MD5 of a file.
pub fn md5_file(path: &Path) -> Result<String> {
    hash_file(path, Md5::new())
}

fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

/// `fetch(f, dest)`: download unless present with the right MD5; `true` when the file is good.
pub fn fetch(f: &Remote, dest: &Path) -> Result<bool> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if dest.exists() && md5_file(dest)? == f.md5 {
        return Ok(true);
    }
    let tmp = part_path(dest);
    http_download(&f.url, &tmp)?;
    if md5_file(&tmp)? != f.md5 {
        println!(
            "MD5 MISMATCH {}",
            dest.file_name().unwrap_or_default().to_string_lossy()
        );
        return Ok(false);
    }
    std::fs::rename(&tmp, dest)?;
    println!(
        "ok {} ({:.0} MB)",
        dest.display(),
        f.size.unwrap_or(0) as f64 / 1e6
    );
    Ok(true)
}

/// `fetch_unverified(f, dest)`: for sources without MD5s, the size within 1 % of the expected
/// one and the SHA-256 printed.
pub fn fetch_unverified(f: &Remote, dest: &Path) -> Result<bool> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if !dest.exists() {
        let tmp = part_path(dest);
        http_download(&f.url, &tmp)?;
        std::fs::rename(&tmp, dest)?;
    }
    let size = std::fs::metadata(dest)?.len();
    if let Some(expected) = f.size.filter(|s| *s > 0) {
        if (size as f64 - expected as f64).abs() > 0.01 * expected as f64 {
            println!(
                "SIZE MISMATCH {}: {size} bytes, expected about {expected}",
                dest.file_name().unwrap_or_default().to_string_lossy()
            );
            return Ok(false);
        }
    }
    let sha = hash_file(dest, Sha256::new())?;
    println!(
        "ok {} ({:.0} MB) sha256 {sha}",
        dest.display(),
        size as f64 / 1e6
    );
    Ok(true)
}

/// `extract(archive, out)`: unpack a `.zip` or a (gzipped) tar unless `out` already holds files.
pub fn extract(archive: &Path, out: &Path) -> Result<()> {
    if out.exists() && std::fs::read_dir(out)?.next().is_some() {
        return Ok(());
    }
    std::fs::create_dir_all(out)?;
    let file = std::fs::File::open(archive)
        .with_context(|| format!("cannot open {}", archive.display()))?;
    if archive.extension().is_some_and(|e| e == "zip") {
        let mut z = zip::ZipArchive::new(file)
            .with_context(|| format!("{} is not a zip archive", archive.display()))?;
        z.extract(out)
            .with_context(|| format!("cannot extract {}", archive.display()))?;
    } else {
        let name = archive.to_string_lossy();
        let reader: Box<dyn Read> = if name.ends_with(".gz") || name.ends_with(".tgz") {
            Box::new(flate2::read::GzDecoder::new(std::io::BufReader::new(file)))
        } else {
            Box::new(std::io::BufReader::new(file))
        };
        untar(reader, out).with_context(|| format!("cannot extract {}", archive.display()))?;
    }
    Ok(())
}

/// Unpacks a tar stream (ustar, GNU long names, pax `path`) into `out`, regular files and
/// directories only; entries that would land outside `out` are refused (tarfile's `"data"`
/// filter).
pub fn untar(mut r: impl Read, out: &Path) -> Result<()> {
    let mut header = [0u8; 512];
    let mut long_name: Option<String> = None;
    loop {
        if read_full(&mut r, &mut header)? < 512 || header.iter().all(|b| *b == 0) {
            return Ok(());
        }
        let field = |a: usize, b: usize| -> String {
            let s = &header[a..b];
            let end = s.iter().position(|c| *c == 0).unwrap_or(s.len());
            String::from_utf8_lossy(&s[..end]).into_owned()
        };
        let size = u64::from_str_radix(field(124, 136).trim(), 8).unwrap_or(0);
        let kind = header[156];
        let mut data = vec![0u8; size as usize];
        read_full(&mut r, &mut data)?;
        let pad = (512 - (size % 512)) % 512;
        std::io::copy(&mut (&mut r).take(pad), &mut std::io::sink())?;
        let mut name = field(0, 100);
        let prefix = field(345, 500);
        if &header[257..262] == b"ustar" && !prefix.is_empty() {
            name = format!("{prefix}/{name}");
        }
        match kind {
            b'L' => {
                long_name = Some(
                    String::from_utf8_lossy(&data)
                        .trim_end_matches('\0')
                        .to_string(),
                );
                continue;
            }
            b'x' => {
                for line in String::from_utf8_lossy(&data).lines() {
                    if let Some((_, kv)) = line.split_once(' ') {
                        if let Some(p) = kv.strip_prefix("path=") {
                            long_name = Some(p.to_string());
                        }
                    }
                }
                continue;
            }
            b'g' => continue,
            _ => {}
        }
        let name = long_name.take().unwrap_or(name);
        let rel = Path::new(&name);
        if rel.is_absolute()
            || rel
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            bail!("refusing to extract {name:?} outside the target folder");
        }
        let dest = out.join(rel);
        match kind {
            b'5' => std::fs::create_dir_all(&dest)?,
            b'0' | 0 | b'7' => {
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&dest, &data)?;
            }
            _ => {} // links and devices are skipped, as the "data" filter refuses them
        }
    }
}

fn read_full(r: &mut impl Read, buf: &mut [u8]) -> Result<usize> {
    let mut at = 0;
    while at < buf.len() {
        let n = r.read(&mut buf[at..])?;
        if n == 0 {
            break;
        }
        at += n;
    }
    Ok(at)
}

/// The `download` subcommand's arguments.
pub fn command() -> Command {
    Command::new("download")
        .about("Download the public datasets into data/raw/, verifying every file's published MD5.")
        .arg(Arg::new("datasets").num_args(0..).help(
            "any of crim13, mirror_mouse, rat7m_s1, rat7m_s2, irodent, mit_homecage, abode \
             (default: all)",
        ))
        .arg(Arg::new("all").long("all").action(ArgAction::SetTrue))
        .arg(Arg::new("raw").long("raw").default_value("data/raw"))
}

/// `download.run(args)`: returns `Ok(false)` when any file failed (`SystemExit(1)` in Python).
pub fn run(m: &ArgMatches) -> Result<bool> {
    let given: Vec<String> = m
        .get_many::<String>("datasets")
        .into_iter()
        .flatten()
        .cloned()
        .collect();
    let names: Vec<String> = if m.get_flag("all") || given.is_empty() {
        DATASETS.iter().map(|d| d.0.to_string()).collect()
    } else {
        given
    };
    let mut unknown: Vec<&String> = names
        .iter()
        .filter(|n| !DATASETS.iter().any(|d| d.0 == n.as_str()))
        .collect();
    if !unknown.is_empty() {
        unknown.sort();
        bail!("unknown datasets: {unknown:?}");
    }
    let raw = PathBuf::from(
        m.get_one::<String>("raw")
            .map_or("data/raw", String::as_str),
    );
    let mut bad: Vec<String> = Vec::new();
    for name in &names {
        let (_, kind, folder) = *DATASETS
            .iter()
            .find(|d| d.0 == name.as_str())
            .expect("checked");
        let out = raw.join(folder);
        match kind {
            Kind::Rat7m(ident) => {
                let mocap: Vec<Remote> = figshare_files(RAT7M_MOCAP)?
                    .into_iter()
                    .filter(|f| f.name == format!("mocap-{ident}.mat"))
                    .collect();
                let article = RAT7M_VIDEOS
                    .iter()
                    .find(|v| v.0 == ident)
                    .expect("listed")
                    .1;
                let videos: Vec<Remote> = figshare_files(article)?
                    .into_iter()
                    .filter(|f| {
                        RAT7M_CHUNKS
                            .iter()
                            .any(|c| f.name.ends_with(&format!("-{c}.mp4")))
                    })
                    .collect();
                for f in &mocap {
                    if !fetch(f, &out.join(&f.name))? {
                        bad.push(f.name.clone());
                    }
                }
                for f in &videos {
                    if !fetch(f, &out.join(ident).join(&f.name))? {
                        bad.push(f.name.clone());
                    }
                }
            }
            Kind::Direct => {
                for f in direct_files(name) {
                    if !fetch_unverified(&f, &out.join(&f.name))? {
                        bad.push(f.name.clone());
                    } else if f.name.ends_with(".zip") {
                        extract(&out.join(&f.name), &out.join("extracted"))?;
                    }
                }
            }
            Kind::Figshare(_) | Kind::Zenodo(_) => {
                let files = match kind {
                    Kind::Figshare(id) => figshare_files(id)?,
                    Kind::Zenodo(id) => zenodo_files(id)?,
                    _ => unreachable!(),
                };
                for f in files.iter().filter(|f| pick(name, &f.name)) {
                    if fetch(f, &out.join(&f.name))? {
                        extract(&out.join(&f.name), &out.join("extracted"))?;
                    } else {
                        bad.push(f.name.clone());
                    }
                }
            }
        }
    }
    if bad.is_empty() {
        println!("all downloads verified");
    } else {
        println!("FAILED: {}", bad.join(", "));
    }
    Ok(bad.is_empty())
}

// ----------------------------------------------------------------------------------- hashing

/// A streaming digest.
pub trait Hasher {
    /// Feeds bytes.
    fn update(&mut self, data: &[u8]);
    /// The digest.
    fn finish(self) -> Vec<u8>;
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `floor(abs(sin(i + 1)) * 2^32)`, the MD5 round constants.
const MD5_K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

/// MD5 (RFC 1321).
pub struct Md5 {
    state: [u32; 4],
    buf: Vec<u8>,
    len: u64,
}

impl Default for Md5 {
    fn default() -> Self {
        Self::new()
    }
}

impl Md5 {
    /// A fresh digest.
    pub fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            buf: Vec::with_capacity(64),
            len: 0,
        }
    }

    fn block(&mut self, chunk: &[u8]) {
        const S: [u32; 64] = [
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20,
            5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        ];
        let m: Vec<u32> = chunk
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let [mut a, mut b, mut c, mut d] = self.state;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(MD5_K[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i]));
        }
        for (s, v) in self.state.iter_mut().zip([a, b, c, d]) {
            *s = s.wrapping_add(v);
        }
    }
}

impl Hasher for Md5 {
    fn update(&mut self, data: &[u8]) {
        self.len += data.len() as u64;
        let mut data = data;
        if !self.buf.is_empty() {
            let take = (64 - self.buf.len()).min(data.len());
            self.buf.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.buf.len() == 64 {
                let b = std::mem::take(&mut self.buf);
                self.block(&b);
            }
        }
        let mut chunks = data.chunks_exact(64);
        for c in &mut chunks {
            self.block(c);
        }
        self.buf.extend_from_slice(chunks.remainder());
    }

    fn finish(mut self) -> Vec<u8> {
        let bits = self.len.wrapping_mul(8);
        let mut pad = vec![0x80u8];
        while (self.buf.len() + pad.len()) % 64 != 56 {
            pad.push(0);
        }
        pad.extend_from_slice(&bits.to_le_bytes());
        let len = self.len;
        self.update(&pad);
        self.len = len;
        self.state.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

/// SHA-256 (FIPS 180-4).
pub struct Sha256 {
    state: [u32; 8],
    buf: Vec<u8>,
    len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

const SHA_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    /// A fresh digest.
    pub fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: Vec::with_capacity(64),
            len: 0,
        }
    }

    fn block(&mut self, chunk: &[u8]) {
        let mut w = [0u32; 64];
        for (i, c) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut h = self.state;
        for i in 0..64 {
            let s1 = h[4].rotate_right(6) ^ h[4].rotate_right(11) ^ h[4].rotate_right(25);
            let ch = (h[4] & h[5]) ^ (!h[4] & h[6]);
            let t1 = h[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA_K[i])
                .wrapping_add(w[i]);
            let s0 = h[0].rotate_right(2) ^ h[0].rotate_right(13) ^ h[0].rotate_right(22);
            let maj = (h[0] & h[1]) ^ (h[0] & h[2]) ^ (h[1] & h[2]);
            let t2 = s0.wrapping_add(maj);
            h = [
                t1.wrapping_add(t2),
                h[0],
                h[1],
                h[2],
                h[3].wrapping_add(t1),
                h[4],
                h[5],
                h[6],
            ];
        }
        for (s, v) in self.state.iter_mut().zip(h) {
            *s = s.wrapping_add(v);
        }
    }
}

impl Hasher for Sha256 {
    fn update(&mut self, data: &[u8]) {
        self.len += data.len() as u64;
        let mut data = data;
        if !self.buf.is_empty() {
            let take = (64 - self.buf.len()).min(data.len());
            self.buf.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.buf.len() == 64 {
                let b = std::mem::take(&mut self.buf);
                self.block(&b);
            }
        }
        let mut chunks = data.chunks_exact(64);
        for c in &mut chunks {
            self.block(c);
        }
        self.buf.extend_from_slice(chunks.remainder());
    }

    fn finish(mut self) -> Vec<u8> {
        let bits = self.len.wrapping_mul(8);
        let mut pad = vec![0x80u8];
        while (self.buf.len() + pad.len()) % 64 != 56 {
            pad.push(0);
        }
        pad.extend_from_slice(&bits.to_be_bytes());
        self.update(&pad);
        self.state.iter().flat_map(|v| v.to_be_bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn md5(data: &[u8]) -> String {
        let mut h = Md5::new();
        h.update(data);
        hex(&h.finish())
    }

    fn sha(data: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(data);
        hex(&h.finish())
    }

    #[test]
    fn digests_match_hashlib() {
        assert_eq!(md5(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        let long: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        // hashlib.md5(bytes(i % 251 for i in range(1000))).hexdigest()
        assert_eq!(md5(&long), MD5_LONG);
        assert_eq!(
            sha(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(sha(&long), SHA_LONG);
        // Feeding in pieces gives the same digest.
        let mut h = Md5::new();
        for c in long.chunks(37) {
            h.update(c);
        }
        assert_eq!(hex(&h.finish()), MD5_LONG);
    }

    const MD5_LONG: &str = "a24f1e3ef66950e1327f210e3997ba2c";
    const SHA_LONG: &str = "4e4c294b331f7a2099a379bec34b9f9fc03dc46ab465d998f4d683da53487e6d";

    #[test]
    fn tar_entries_are_unpacked_and_escapes_refused() {
        let dir = std::env::temp_dir().join(format!("animalpose-untar-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let entry = |name: &str, body: &[u8]| -> Vec<u8> {
            let mut h = [0u8; 512];
            h[..name.len()].copy_from_slice(name.as_bytes());
            let size = format!("{:011o}\0", body.len());
            h[124..136].copy_from_slice(size.as_bytes());
            h[156] = b'0';
            h[257..263].copy_from_slice(b"ustar\0");
            let mut out = h.to_vec();
            out.extend_from_slice(body);
            out.resize(out.len() + (512 - body.len() % 512) % 512, 0);
            out
        };
        let mut tar = entry("a/b.txt", b"hello");
        tar.extend(vec![0u8; 1024]);
        untar(&tar[..], &dir).unwrap();
        assert_eq!(std::fs::read(dir.join("a/b.txt")).unwrap(), b"hello");
        let evil = entry("../x.txt", b"no");
        assert!(untar(&evil[..], &dir).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
