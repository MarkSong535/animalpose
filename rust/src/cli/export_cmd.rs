//! `animalpose export <what>`: checkpoints as `.safetensors` and back — the port of
//! `src/animalpose/cli/export/` and `src/animalpose/export/safetensors.py`.
//!
//! | Python | Rust |
//! | --- | --- |
//! | `export safetensors` (`split_checkpoint`, `metadata_for`, `write_safetensors`) | [`export_safetensors`] |
//! | `export import-safetensors` (`checkpoint_from_safetensors`, `torch.save`, `compare`) | [`import_safetensors`] |
//! | `export models` (TorchScript / ONNX through `torch.onnx`) | refused: Python only |
//!
//! The header's `__metadata__` keeps `"animalpose"` (the JSON model config), `"kind"` and, when the
//! checkpoint had other entries, `"extra"` — the layout [`crate::models::load`] reads.
//!
//! `// differs from Python:`
//! * A `.safetensors` input to `export safetensors` is accepted and rewritten with its own dtypes;
//!   any other checkpoint goes through [`crate::models::load`], which hands back `float32` values,
//!   so integer buffers (`num_batches_tracked`) come out as `F32` there.
//! * `import-safetensors` writes the `.pt` zip archive itself (pickle protocol 2, one storage per
//!   tensor, the dtype the safetensors file stored) instead of calling `torch.save`; `--check`
//!   compares values and shapes against the original read through [`crate::models::load`], which
//!   cannot see the original dtypes.

use crate::data::pyjson::{loads, PyVal};
use crate::models::weights::{CONFIG_KEY, KIND_KEY};
use anyhow::{bail, Context, Result};
use clap::{Arg, ArgMatches, Command};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The metadata key holding the other checkpoint entries (`"extra"`).
pub const EXTRA_KEY: &str = "extra";

/// The `export` command group.
pub fn command() -> Command {
    Command::new("export")
        .about("export TorchScript / ONNX / safetensors, and import them back")
        .subcommand_required(true)
        .subcommand(
            Command::new("models")
                .about("Python only: TorchScript and ONNX export (src/animalpose/export/export.py)")
                .arg(
                    Arg::new("args")
                        .num_args(0..)
                        .trailing_var_arg(true)
                        .allow_hyphen_values(true),
                ),
        )
        .subcommand(
            Command::new("safetensors")
                .about("A checkpoint as a .safetensors file.")
                .arg(Arg::new("checkpoint").long("checkpoint").required(true))
                .arg(
                    Arg::new("out")
                        .long("out")
                        .help("default: exported/<checkpoint stem>.safetensors"),
                ),
        )
        .subcommand(
            Command::new("import-safetensors")
                .about("Rebuild a .pt checkpoint from a .safetensors file.")
                .arg(Arg::new("safetensors").long("safetensors").required(true))
                .arg(
                    Arg::new("out")
                        .long("out")
                        .help("default: exported/<name>.pt"),
                )
                .arg(
                    Arg::new("check")
                        .long("check")
                        .help("original checkpoint to compare against"),
                ),
        )
}

/// Runs `animalpose export <what>`.
pub fn run(m: &ArgMatches) -> Result<()> {
    match m.subcommand() {
        Some(("safetensors", a)) => {
            let ckpt = PathBuf::from(a.get_one::<String>("checkpoint").context("--checkpoint")?);
            let out = a
                .get_one::<String>("out")
                .map_or_else(|| export_dir().join(format!("{}.safetensors", stem(&ckpt))), PathBuf::from);
            export_safetensors(&ckpt, &out)
        }
        Some(("import-safetensors", a)) => {
            let src = PathBuf::from(a.get_one::<String>("safetensors").context("--safetensors")?);
            let out = a
                .get_one::<String>("out")
                .map_or_else(|| export_dir().join(format!("{}.pt", stem(&src))), PathBuf::from);
            import_safetensors(&src, &out, a.get_one::<String>("check").map(Path::new))
        }
        Some(("models", _)) => bail!(
            "unsupported: `animalpose export models` traces the networks with torch.jit / torch.onnx \
             (src/animalpose/export/export.py); run it from the Python package"
        ),
        Some((other, _)) => bail!("unknown export target {other:?}"),
        None => bail!("animalpose export needs a target"),
    }
}

/// `paths.py:EXPORT_DIR`: `<repo>/exported`.
fn export_dir() -> PathBuf {
    crate::data::repo_root().map_or_else(|| PathBuf::from("exported"), |r| r.join("exported"))
}

fn stem(p: &Path) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// One tensor with its stored dtype and little-endian bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct RawTensor {
    /// The safetensors dtype.
    pub dtype: Dtype,
    /// The shape.
    pub shape: Vec<usize>,
    /// Row-major little-endian bytes.
    pub data: Vec<u8>,
}

/// A checkpoint as the Python dict holds it: `kind`, `model_cfg`, `state_dict` and the rest.
#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    /// `ckpt["kind"]`.
    pub kind: String,
    /// `ckpt["model_cfg"]`.
    pub model_cfg: PyVal,
    /// `ckpt["state_dict"]`, in file order.
    pub tensors: Vec<(String, RawTensor)>,
    /// Every other entry (`window`, `crop_size`, ...).
    pub extra: PyVal,
}

/// `read_safetensors` + `checkpoint_from_safetensors`.
pub fn read_checkpoint(path: &Path) -> Result<Checkpoint> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    let (_, header) =
        SafeTensors::read_metadata(&bytes).context("safetensors header is malformed")?;
    let meta = header.metadata().clone().unwrap_or_default();
    let file = SafeTensors::deserialize(&bytes).context("safetensors file is malformed")?;
    let mut tensors: Vec<(String, RawTensor, usize)> = file
        .tensors()
        .into_iter()
        .map(|(name, v)| {
            let offset = v.data().as_ptr() as usize - bytes.as_ptr() as usize;
            (
                name,
                RawTensor {
                    dtype: v.dtype(),
                    shape: v.shape().to_vec(),
                    data: v.data().to_vec(),
                },
                offset,
            )
        })
        .collect();
    tensors.sort_by_key(|t| t.2);
    let json = |k: &str| -> Result<PyVal> {
        match meta.get(k) {
            Some(text) => loads(text).with_context(|| format!("metadata {k:?} is not JSON")),
            None => Ok(PyVal::dict()),
        }
    };
    Ok(Checkpoint {
        kind: meta.get(KIND_KEY).cloned().unwrap_or_else(|| "None".into()),
        model_cfg: json(CONFIG_KEY)?,
        tensors: tensors.into_iter().map(|(n, t, _)| (n, t)).collect(),
        extra: json(EXTRA_KEY)?,
    })
}

/// `metadata_for(kind, cfg, extra)`: string values only, as the format requires.
pub fn metadata_for(ckpt: &Checkpoint) -> std::collections::HashMap<String, String> {
    let mut meta = std::collections::HashMap::new();
    meta.insert(CONFIG_KEY.to_string(), ckpt.model_cfg.dumps());
    meta.insert(KIND_KEY.to_string(), ckpt.kind.clone());
    if matches!(&ckpt.extra, PyVal::Dict(d) if !d.is_empty()) {
        meta.insert(EXTRA_KEY.to_string(), ckpt.extra.dumps());
    }
    meta
}

/// `write_safetensors(path, tensors, metadata)`.
pub fn write_checkpoint(path: &Path, ckpt: &Checkpoint) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let views: Vec<(String, TensorView<'_>)> = ckpt
        .tensors
        .iter()
        .map(|(n, t)| {
            Ok((
                n.clone(),
                TensorView::new(t.dtype, t.shape.clone(), &t.data)
                    .with_context(|| format!("tensor {n}"))?,
            ))
        })
        .collect::<Result<_>>()?;
    let bytes =
        safetensors::serialize(views, Some(metadata_for(ckpt))).context("cannot serialise")?;
    std::fs::write(path, bytes).with_context(|| format!("cannot write {}", path.display()))
}

/// `export safetensors --checkpoint <path> --out <out>`.
pub fn export_safetensors(checkpoint: &Path, out: &Path) -> Result<()> {
    let is_st = checkpoint
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
    let ckpt = if is_st {
        read_checkpoint(checkpoint)?
    } else {
        let (cfg, kind, weights) = crate::models::load(checkpoint)
            .with_context(|| format!("cannot read {}", checkpoint.display()))?;
        let model_cfg = match weights.metadata_value(CONFIG_KEY) {
            Some(text) => loads(text)?,
            None => PyVal::from_json(&serde_json::from_str(&cfg.to_json_string()?)?),
        };
        let extra = match weights.metadata_value(EXTRA_KEY) {
            Some(text) => loads(text)?,
            None => PyVal::dict(),
        };
        Checkpoint {
            kind: kind.as_str().to_string(),
            model_cfg,
            tensors: weights
                .tensors()
                .iter()
                .map(|(n, t)| {
                    (
                        n.clone(),
                        RawTensor {
                            dtype: Dtype::F32,
                            shape: t.shape.clone(),
                            data: t.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                        },
                    )
                })
                .collect(),
            extra,
        }
    };
    if !["pose", "detector", "behavior", "temporal"].contains(&ckpt.kind.as_str()) {
        bail!(
            "{}: unknown model kind {:?}",
            checkpoint.display(),
            ckpt.kind
        );
    }
    write_checkpoint(out, &ckpt)?;
    let size = std::fs::metadata(out)?.len();
    println!(
        "wrote {} ({} tensors, {:.2} MB, kind={})",
        out.display(),
        ckpt.tensors.len(),
        size as f64 / 1e6,
        ckpt.kind
    );
    println!("  model config: {}", ckpt.model_cfg.dumps());
    Ok(())
}

/// `export import-safetensors --safetensors <src> --out <out> [--check <original>]`.
pub fn import_safetensors(src: &Path, out: &Path, check: Option<&Path>) -> Result<()> {
    let ckpt = read_checkpoint(src)?;
    write_torch(out, &ckpt)?;
    println!(
        "wrote {} ({} tensors, kind={})",
        out.display(),
        ckpt.tensors.len(),
        ckpt.kind
    );
    println!("  model config: {}", ckpt.model_cfg.dumps());
    if let Some(original) = check {
        let (cfg, kind, weights) = crate::models::load(original)
            .with_context(|| format!("cannot read {}", original.display()))?;
        let mut worst = 0.0f64;
        let mut checked = 0usize;
        for (name, t) in &ckpt.tensors {
            let Some(o) = weights.get(name) else {
                bail!("tensor {name} is not in the original checkpoint");
            };
            if o.shape != t.shape {
                bail!("tensor {name}: shape {:?} != {:?}", o.shape, t.shape);
            }
            let ours = values_f64(t)?;
            for (a, b) in o.data.iter().zip(&ours) {
                worst = worst.max((f64::from(*a) - b).abs());
            }
            checked += 1;
        }
        for name in weights.names() {
            if !ckpt.tensors.iter().any(|(n, _)| n == name) {
                bail!("tensor {name} is missing from the rebuilt checkpoint");
            }
        }
        if kind.as_str() != ckpt.kind {
            bail!(
                "checkpoint entry kind differs: {:?} != {:?}",
                kind.as_str(),
                ckpt.kind
            );
        }
        let _ = cfg;
        println!(
            "round trip vs {}: {checked} tensors, max abs diff {}",
            original.display(),
            format_g(worst)
        );
        if worst != 0.0 {
            bail!("round trip is not bit-exact");
        }
    }
    Ok(())
}

/// Python's `f"{x:g}"`.
pub fn format_g(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    if !x.is_finite() {
        return crate::data::pyjson::float_repr(x);
    }
    let exp = x.abs().log10().floor() as i32;
    if (-4..6).contains(&exp) {
        let digits = (5 - exp).max(0) as usize;
        let s = format!("{x:.digits$}");
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    } else {
        let s = format!("{x:.5e}");
        let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
        let m = if m.contains('.') {
            m.trim_end_matches('0').trim_end_matches('.')
        } else {
            m
        };
        let e: i32 = e.parse().unwrap_or(0);
        format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    }
}

fn values_f64(t: &RawTensor) -> Result<Vec<f64>> {
    let d = &t.data;
    Ok(match t.dtype {
        Dtype::F32 => d
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        Dtype::F64 => d
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
            .collect(),
        Dtype::I64 => d
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as f64)
            .collect(),
        Dtype::I32 => d
            .chunks_exact(4)
            .map(|c| f64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        Dtype::U8 | Dtype::BOOL => d.iter().map(|v| f64::from(*v)).collect(),
        Dtype::I8 => d.iter().map(|v| f64::from(*v as i8)).collect(),
        other => bail!("dtype {other:?} is not compared"),
    })
}

// ------------------------------------------------------------------------- torch .pt writer

/// The `torch.<X>Storage` class and element size of a dtype.
fn storage_of(dtype: Dtype) -> Result<(&'static str, usize)> {
    Ok(match dtype {
        Dtype::F32 => ("FloatStorage", 4),
        Dtype::F64 => ("DoubleStorage", 8),
        Dtype::F16 => ("HalfStorage", 2),
        Dtype::BF16 => ("BFloat16Storage", 2),
        Dtype::I64 => ("LongStorage", 8),
        Dtype::I32 => ("IntStorage", 4),
        Dtype::I16 => ("ShortStorage", 2),
        Dtype::I8 => ("CharStorage", 1),
        Dtype::U8 => ("ByteStorage", 1),
        Dtype::BOOL => ("BoolStorage", 1),
        other => bail!("dtype {other:?} has no torch storage"),
    })
}

/// A pickle (protocol 2) emitter for the values a checkpoint holds.
struct Pickler {
    out: Vec<u8>,
}

impl Pickler {
    fn op(&mut self, b: u8) {
        self.out.push(b);
    }

    fn global(&mut self, module: &str, name: &str) {
        self.op(b'c');
        self.out
            .extend_from_slice(format!("{module}\n{name}\n").as_bytes());
    }

    fn str(&mut self, s: &str) {
        self.op(b'X');
        self.out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        self.out.extend_from_slice(s.as_bytes());
    }

    fn int(&mut self, v: i64) {
        if (0..256).contains(&v) {
            self.op(b'K');
            self.out.push(v as u8);
        } else if (0..65536).contains(&v) {
            self.op(b'M');
            self.out.extend_from_slice(&(v as u16).to_le_bytes());
        } else if i32::try_from(v).is_ok() {
            self.op(b'J');
            self.out.extend_from_slice(&(v as i32).to_le_bytes());
        } else {
            // LONG1: little-endian two's complement.
            let bytes = v.to_le_bytes();
            self.op(0x8a);
            self.out.push(8);
            self.out.extend_from_slice(&bytes);
        }
    }

    fn value(&mut self, v: &PyVal) {
        match v {
            PyVal::None => self.op(b'N'),
            PyVal::Bool(b) => self.op(if *b { 0x88 } else { 0x89 }),
            PyVal::Int(i) => self.int(*i),
            PyVal::Float(x) => {
                self.op(b'G');
                self.out.extend_from_slice(&x.to_be_bytes());
            }
            PyVal::Str(s) => self.str(s),
            PyVal::List(items) => {
                self.op(b']');
                if !items.is_empty() {
                    self.op(b'(');
                    for i in items {
                        self.value(i);
                    }
                    self.op(b'e');
                }
            }
            PyVal::Dict(items) => {
                self.op(b'}');
                if !items.is_empty() {
                    self.op(b'(');
                    for (k, i) in items {
                        self.str(k.strip_prefix(crate::data::pyjson::FLOAT_KEY).unwrap_or(k));
                        self.value(i);
                    }
                    self.op(b'u');
                }
            }
        }
    }

    /// `torch._utils._rebuild_tensor_v2(storage, 0, shape, stride, False, OrderedDict())` with
    /// `storage` the persistent id `('storage', torch.<X>Storage, key, 'cpu', numel)`.
    fn tensor(&mut self, key: usize, t: &RawTensor) -> Result<()> {
        let (storage, size) = storage_of(t.dtype)?;
        let numel = t.data.len() / size;
        self.global("torch._utils", "_rebuild_tensor_v2");
        self.op(b'(');
        // persistent id tuple
        self.op(b'(');
        self.str("storage");
        self.global("torch", storage);
        self.str(&key.to_string());
        self.str("cpu");
        self.int(numel as i64);
        self.op(b't');
        self.op(b'Q'); // BINPERSID
        self.int(0);
        let tuple = |p: &mut Self, items: &[usize]| {
            p.op(b'(');
            for v in items {
                p.int(*v as i64);
            }
            p.op(b't');
        };
        tuple(self, &t.shape);
        let mut strides = vec![1usize; t.shape.len()];
        for i in (0..t.shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * t.shape[i + 1];
        }
        tuple(self, &strides);
        self.op(0x89); // requires_grad=False
        self.global("collections", "OrderedDict");
        self.op(b')');
        self.op(b'R');
        self.op(b't');
        self.op(b'R');
        Ok(())
    }
}

/// `torch.save(ckpt, path)`: the zip archive `torch.load` reads (`<stem>/data.pkl`, one
/// `<stem>/data/<key>` storage per tensor, 64-byte aligned, plus the version records).
pub fn write_torch(path: &Path, ckpt: &Checkpoint) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut p = Pickler { out: vec![0x80, 2] };
    p.op(b'}');
    p.op(b'(');
    p.str("kind");
    p.str(&ckpt.kind);
    p.str("model_cfg");
    p.value(&ckpt.model_cfg);
    p.str("state_dict");
    p.global("collections", "OrderedDict");
    p.op(b')');
    p.op(b'R');
    if !ckpt.tensors.is_empty() {
        p.op(b'(');
        for (i, (name, t)) in ckpt.tensors.iter().enumerate() {
            p.str(name);
            p.tensor(i, t)?;
        }
        p.op(b'u');
    }
    if let PyVal::Dict(items) = &ckpt.extra {
        for (k, v) in items {
            p.str(k);
            p.value(v);
        }
    }
    p.op(b'u');
    p.op(b'.');

    let prefix = stem(path);
    let file =
        std::fs::File::create(path).with_context(|| format!("cannot write {}", path.display()))?;
    let mut z = zip::ZipWriter::new(file);
    let stored = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .with_alignment(64);
    let mut put = |name: String, data: &[u8]| -> Result<()> {
        z.start_file(name, stored)?;
        z.write_all(data)?;
        Ok(())
    };
    put(format!("{prefix}/data.pkl"), &p.out)?;
    put(format!("{prefix}/.format_version"), b"1")?;
    put(format!("{prefix}/.storage_alignment"), b"64")?;
    put(format!("{prefix}/byteorder"), b"little")?;
    for (i, (_, t)) in ckpt.tensors.iter().enumerate() {
        put(format!("{prefix}/data/{i}"), &t.data)?;
    }
    put(format!("{prefix}/version"), b"3\n")?;
    z.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_g_matches_python() {
        assert_eq!(format_g(0.0), "0");
        assert_eq!(format_g(1.5e-7), "1.5e-07");
        assert_eq!(format_g(0.000123), "0.000123");
        assert_eq!(format_g(123456.0), "123456");
        assert_eq!(format_g(1234567.0), "1.23457e+06");
        assert_eq!(format_g(0.5), "0.5");
    }

    #[test]
    fn safetensors_round_trip_keeps_dtypes_and_metadata() {
        let dir = std::env::temp_dir().join(format!("animalpose-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt = Checkpoint {
            kind: "behavior".into(),
            model_cfg: PyVal::dict()
                .with("hidden", 16i64)
                .with("classes", vec!["a", "b"]),
            tensors: vec![
                (
                    "w".into(),
                    RawTensor {
                        dtype: Dtype::F32,
                        shape: vec![2],
                        data: [1.5f32, -2.0]
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    },
                ),
                (
                    "n".into(),
                    RawTensor {
                        dtype: Dtype::I64,
                        shape: vec![],
                        data: 7i64.to_le_bytes().to_vec(),
                    },
                ),
            ],
            extra: PyVal::dict().with("window", 64i64),
        };
        let path = dir.join("m.safetensors");
        write_checkpoint(&path, &ckpt).unwrap();
        let back = read_checkpoint(&path).unwrap();
        assert_eq!(back.kind, "behavior");
        assert_eq!(back.model_cfg, ckpt.model_cfg);
        assert_eq!(back.extra, ckpt.extra);
        let n = back.tensors.iter().find(|t| t.0 == "n").unwrap();
        assert_eq!(n.1.dtype, Dtype::I64);
        write_torch(&dir.join("m.pt"), &back).unwrap();
        let mut z = zip::ZipArchive::new(std::fs::File::open(dir.join("m.pt")).unwrap()).unwrap();
        assert!(z.by_name("m/data.pkl").is_ok() && z.by_name("m/version").is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
