//! Reads PyTorch `.pt` checkpoints (`torch.save` zip archives) without PyTorch.
//!
//! A checkpoint written by `src/animalpose/train/checkpoint.py:save_checkpoint` is a zip archive
//! holding `<prefix>/data.pkl` (a pickle of `{"kind", "model_cfg", "state_dict", ...extra}`) and one
//! raw little-endian storage per tensor under `<prefix>/data/<key>`. This module runs the small part
//! of the pickle virtual machine those files use, rebuilds every tensor from its storage, offset,
//! size and stride, and returns the same [`Weights`] a `.safetensors` export of the checkpoint gives:
//! the state dict as `f32` tensors and the `"animalpose"` / `"kind"` / `"extra"` metadata entries of
//! `src/animalpose/export/safetensors.py:metadata_for`.
//!
//! Only data is reconstructed: a `GLOBAL` names a callable but nothing is ever imported or run, and
//! any callable other than the tensor/`OrderedDict` rebuilders becomes an opaque value.

use crate::models::weights::{Weights, CONFIG_KEY, KIND_KEY};
use anyhow::{bail, Context, Result};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::rc::Rc;

/// The metadata key holding the checkpoint's other top-level entries (`input_size`, `crop_size`,
/// `window`), as JSON.
pub const EXTRA_KEY: &str = "extra";

/// The first bytes of a zip archive, which every `torch.save` file since PyTorch 1.6 is.
const ZIP_MAGIC: &[u8] = b"PK\x03\x04";

/// `true` when `bytes` look like a `torch.save` zip archive.
pub fn is_torch_zip(bytes: &[u8]) -> bool {
    bytes.starts_with(ZIP_MAGIC)
}

/// `true` when `bytes` are a git-lfs pointer file rather than the object it points at.
pub fn is_lfs_pointer(bytes: &[u8]) -> bool {
    bytes.starts_with(b"version https://git-lfs.github.com/spec/")
}

/// One unpickled value. Containers are shared, so a memoised dict filled after `BINPUT` is seen
/// filled through every later `BINGET`, as in Python.
#[derive(Clone, Debug)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes,
    Tuple(Vec<Value>),
    List(Rc<RefCell<Vec<Value>>>),
    Dict(Rc<RefCell<Vec<(Value, Value)>>>),
    Global(String, String),
    Storage { dtype: StorageType, key: String },
    Tensor(Rc<TensorRef>),
    /// A callable applied to arguments that this reader does not interpret.
    Opaque,
    Mark,
}

/// A tensor view onto one storage of the archive.
#[derive(Debug)]
struct TensorRef {
    dtype: StorageType,
    key: String,
    offset: usize,
    size: Vec<usize>,
    stride: Vec<usize>,
}

/// The element type of a storage, from its `torch.<Type>Storage` global.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StorageType {
    F32,
    F64,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl StorageType {
    fn from_global(name: &str) -> Result<Self> {
        Ok(match name {
            "FloatStorage" => Self::F32,
            "DoubleStorage" => Self::F64,
            "HalfStorage" => Self::F16,
            "BFloat16Storage" => Self::BF16,
            "LongStorage" => Self::I64,
            "IntStorage" => Self::I32,
            "ShortStorage" => Self::I16,
            "CharStorage" => Self::I8,
            "ByteStorage" => Self::U8,
            "BoolStorage" => Self::Bool,
            other => bail!("unsupported torch storage type {other}"),
        })
    }

    fn width(self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    fn read(self, b: &[u8]) -> f32 {
        match self {
            Self::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Self::F64 => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
            Self::F16 => f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
            Self::BF16 => f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16),
            Self::I64 => i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
            Self::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            Self::I16 => f32::from(i16::from_le_bytes([b[0], b[1]])),
            Self::I8 => f32::from(b[0] as i8),
            Self::U8 => f32::from(b[0]),
            Self::Bool => {
                if b[0] != 0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// IEEE half to single precision.
fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, _) => {
            // subnormal: renormalise
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3ff) << 13)
        }
        (31, 0) => sign | 0x7f80_0000,
        (31, _) => sign | 0x7fc0_0000,
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// Reads a `.pt` checkpoint into the same [`Weights`] its safetensors export would give.
pub fn read(path: &Path) -> Result<Weights> {
    let bytes =
        std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    from_bytes(&bytes).with_context(|| format!("cannot parse the torch checkpoint {}", path.display()))
}

/// Parses an in-memory `torch.save` archive.
pub fn from_bytes(bytes: &[u8]) -> Result<Weights> {
    if is_lfs_pointer(bytes) {
        bail!("this is a git-lfs pointer, not the checkpoint: run `git lfs pull`");
    }
    if !is_torch_zip(bytes) {
        bail!("not a torch.save zip archive (legacy pre-1.6 pickles are not supported)");
    }
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
        .context("the checkpoint zip is malformed")?;
    let pkl_name = (0..archive.len())
        .filter_map(|i| archive.name_for_index(i).map(str::to_string))
        .find(|n| n.ends_with("data.pkl") && n.matches('/').count() == 1)
        .context("no <prefix>/data.pkl in the archive")?;
    let prefix = pkl_name.trim_end_matches("data.pkl").to_string();
    let mut pkl = Vec::new();
    archive
        .by_name(&pkl_name)
        .context("cannot open data.pkl")?
        .read_to_end(&mut pkl)?;
    let root = Unpickler::new(&pkl).run()?;
    let Value::Dict(items) = root else {
        bail!("the checkpoint is not a dict");
    };

    let mut weights = Weights::new();
    let mut extra = serde_json::Map::new();
    let mut storages: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut saw_state = false;
    for (k, v) in items.borrow().iter() {
        let Value::Str(key) = k else { continue };
        match key.as_str() {
            "kind" => {
                if let Value::Str(kind) = v {
                    weights.set_metadata(KIND_KEY, kind.clone());
                }
            }
            "model_cfg" => {
                let json = to_json(v).context("model_cfg holds a value JSON cannot carry")?;
                weights.set_metadata(CONFIG_KEY, serde_json::to_string(&json)?);
            }
            "state_dict" => {
                saw_state = true;
                let Value::Dict(state) = v else {
                    bail!("state_dict is not a dict");
                };
                for (name, tensor) in state.borrow().iter() {
                    let (Value::Str(name), Value::Tensor(t)) = (name, tensor) else {
                        continue;
                    };
                    if !storages.contains_key(&t.key) {
                        let mut raw = Vec::new();
                        archive
                            .by_name(&format!("{prefix}data/{}", t.key))
                            .with_context(|| format!("storage {} of {name} is missing", t.key))?
                            .read_to_end(&mut raw)?;
                        storages.insert(t.key.clone(), raw);
                    }
                    let data = gather(t, &storages[&t.key])
                        .with_context(|| format!("tensor {name}"))?;
                    weights.insert(name.clone(), &t.size, data);
                }
            }
            _ => {
                if let Some(json) = to_json(v) {
                    extra.insert(key.clone(), json);
                }
            }
        }
    }
    if !saw_state {
        bail!("the checkpoint has no state_dict");
    }
    if !extra.is_empty() {
        weights.set_metadata(EXTRA_KEY, serde_json::Value::Object(extra).to_string());
    }
    Ok(weights)
}

/// The values of one strided view, row-major, as `f32`.
fn gather(t: &TensorRef, raw: &[u8]) -> Result<Vec<f32>> {
    let w = t.dtype.width();
    let numel: usize = t.size.iter().product();
    let mut out = Vec::with_capacity(numel);
    if numel == 0 {
        return Ok(out);
    }
    let mut index = vec![0usize; t.size.len()];
    for _ in 0..numel {
        let at = t.offset + index.iter().zip(&t.stride).map(|(i, s)| i * s).sum::<usize>();
        let b = raw
            .get(at * w..(at + 1) * w)
            .context("a tensor reaches past the end of its storage")?;
        out.push(t.dtype.read(b));
        for d in (0..index.len()).rev() {
            index[d] += 1;
            if index[d] < t.size[d] {
                break;
            }
            index[d] = 0;
        }
    }
    Ok(out)
}

/// A JSON view of a plain value (dicts with string keys, lists, tuples, numbers, strings).
fn to_json(v: &Value) -> Option<serde_json::Value> {
    Some(match v {
        Value::None => serde_json::Value::Null,
        Value::Bool(b) => (*b).into(),
        Value::Int(i) => (*i).into(),
        Value::Float(f) => serde_json::Number::from_f64(*f)?.into(),
        Value::Str(s) => s.clone().into(),
        Value::Tuple(items) => items.iter().map(to_json).collect::<Option<Vec<_>>>()?.into(),
        Value::List(items) => items
            .borrow()
            .iter()
            .map(to_json)
            .collect::<Option<Vec<_>>>()?
            .into(),
        Value::Dict(items) => {
            let mut map = serde_json::Map::new();
            for (k, v) in items.borrow().iter() {
                let Value::Str(k) = k else { return None };
                map.insert(k.clone(), to_json(v)?);
            }
            map.into()
        }
        _ => return None,
    })
}

/// The pickle virtual machine, protocols 2-5, restricted to the opcodes `torch.save` emits.
struct Unpickler<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    memo: BTreeMap<u32, Value>,
}

impl<'a> Unpickler<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            stack: Vec::new(),
            memo: BTreeMap::new(),
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("pickle length overflow")?;
        let bytes = self
            .data
            .get(self.pos..end)
            .context("the pickle ends early")?;
        self.pos = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn line(&mut self) -> Result<String> {
        let rest = &self.data[self.pos..];
        let end = rest
            .iter()
            .position(|&b| b == b'\n')
            .context("unterminated pickle text line")?;
        let s = String::from_utf8_lossy(&rest[..end]).into_owned();
        self.pos += end + 1;
        Ok(s)
    }

    fn pop(&mut self) -> Result<Value> {
        self.stack.pop().context("pickle stack underflow")
    }

    fn top(&mut self) -> Result<&mut Value> {
        self.stack.last_mut().context("pickle stack underflow")
    }

    /// Everything above the topmost mark, which is removed.
    fn pop_mark(&mut self) -> Result<Vec<Value>> {
        let at = self
            .stack
            .iter()
            .rposition(|v| matches!(v, Value::Mark))
            .context("pickle MARK missing")?;
        let items = self.stack.split_off(at + 1);
        self.stack.pop();
        Ok(items)
    }

    fn string(&mut self, n: usize) -> Result<()> {
        let s = String::from_utf8(self.take(n)?.to_vec()).context("pickle string is not UTF-8")?;
        self.stack.push(Value::Str(s));
        Ok(())
    }

    fn memo_put(&mut self, key: u32) -> Result<()> {
        let v = self.stack.last().context("pickle stack underflow")?.clone();
        self.memo.insert(key, v);
        Ok(())
    }

    fn memo_get(&mut self, key: u32) -> Result<()> {
        let v = self
            .memo
            .get(&key)
            .with_context(|| format!("pickle memo {key} is missing"))?
            .clone();
        self.stack.push(v);
        Ok(())
    }

    fn run(mut self) -> Result<Value> {
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    self.u8()?; // PROTO
                }
                0x95 => {
                    self.u64()?; // FRAME
                }
                b'.' => return self.pop(), // STOP
                b'(' => self.stack.push(Value::Mark),
                b'N' => self.stack.push(Value::None),
                0x88 => self.stack.push(Value::Bool(true)),
                0x89 => self.stack.push(Value::Bool(false)),
                b'J' => {
                    let v = self.u32()? as i32;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                b'K' => {
                    let v = self.u8()?;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                b'M' => {
                    let v = self.u16()?;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                0x8a => {
                    // LONG1: little-endian two's complement of n bytes
                    let n = self.u8()? as usize;
                    let b = self.take(n)?;
                    let mut v: i64 = 0;
                    for (i, byte) in b.iter().enumerate().take(8) {
                        v |= i64::from(*byte) << (8 * i);
                    }
                    if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                        v -= 1i64 << (8 * n);
                    }
                    self.stack.push(Value::Int(v));
                }
                b'G' => {
                    let b = self.take(8)?;
                    let v = f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
                    self.stack.push(Value::Float(v));
                }
                b'X' => {
                    let n = self.u32()? as usize;
                    self.string(n)?;
                }
                0x8c => {
                    let n = self.u8()? as usize;
                    self.string(n)?;
                }
                0x8d => {
                    let n = self.u64()? as usize;
                    self.string(n)?;
                }
                b'U' => {
                    let n = self.u8()? as usize;
                    self.string(n)?;
                }
                b'T' => {
                    let n = self.u32()? as usize;
                    self.string(n)?;
                }
                b'C' => {
                    let n = self.u8()? as usize;
                    self.take(n)?;
                    self.stack.push(Value::Bytes);
                }
                b'B' => {
                    let n = self.u32()? as usize;
                    self.take(n)?;
                    self.stack.push(Value::Bytes);
                }
                b'}' => self
                    .stack
                    .push(Value::Dict(Rc::new(RefCell::new(Vec::new())))),
                b']' => self
                    .stack
                    .push(Value::List(Rc::new(RefCell::new(Vec::new())))),
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(items));
                }
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a]));
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b]));
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b, c]));
                }
                b'l' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::List(Rc::new(RefCell::new(items))));
                }
                b'd' => {
                    let items = self.pop_mark()?;
                    let pairs = pairs(items)?;
                    self.stack.push(Value::Dict(Rc::new(RefCell::new(pairs))));
                }
                b'a' => {
                    let v = self.pop()?;
                    match self.top()? {
                        Value::List(l) => l.borrow_mut().push(v),
                        _ => bail!("APPEND onto a non-list"),
                    }
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    match self.top()? {
                        Value::List(l) => l.borrow_mut().extend(items),
                        _ => bail!("APPENDS onto a non-list"),
                    }
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    match self.top()? {
                        Value::Dict(d) => d.borrow_mut().push((k, v)),
                        Value::Opaque => {}
                        _ => bail!("SETITEM onto a non-dict"),
                    }
                }
                b'u' => {
                    let items = pairs(self.pop_mark()?)?;
                    match self.top()? {
                        Value::Dict(d) => d.borrow_mut().extend(items),
                        Value::Opaque => {}
                        _ => bail!("SETITEMS onto a non-dict"),
                    }
                }
                b'q' => {
                    let k = u32::from(self.u8()?);
                    self.memo_put(k)?;
                }
                b'r' => {
                    let k = self.u32()?;
                    self.memo_put(k)?;
                }
                0x94 => {
                    let k = self.memo.len() as u32;
                    self.memo_put(k)?;
                }
                b'h' => {
                    let k = u32::from(self.u8()?);
                    self.memo_get(k)?;
                }
                b'j' => {
                    let k = self.u32()?;
                    self.memo_get(k)?;
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Value::Global(module, name));
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    match (module, name) {
                        (Value::Str(m), Value::Str(n)) => self.stack.push(Value::Global(m, n)),
                        _ => bail!("STACK_GLOBAL needs two strings"),
                    }
                }
                b'Q' => {
                    let pid = self.pop()?;
                    self.stack.push(persistent_load(pid)?);
                }
                b'R' => {
                    let args = self.pop()?;
                    let func = self.pop()?;
                    self.stack.push(reduce(func, args)?);
                }
                0x81 => {
                    // NEWOBJ: cls.__new__(cls, *args)
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    self.stack.push(reduce(cls, args)?);
                }
                b'b' => {
                    // BUILD: state for the object below; tensors and dicts ignore it.
                    self.pop()?;
                }
                other => bail!(
                    "unsupported pickle opcode 0x{other:02x} at byte {}",
                    self.pos - 1
                ),
            }
        }
    }
}

/// `[k0, v0, k1, v1, ...]` as pairs.
fn pairs(items: Vec<Value>) -> Result<Vec<(Value, Value)>> {
    if items.len() % 2 != 0 {
        bail!("odd number of dict items");
    }
    let mut out = Vec::with_capacity(items.len() / 2);
    let mut it = items.into_iter();
    while let (Some(k), Some(v)) = (it.next(), it.next()) {
        out.push((k, v));
    }
    Ok(out)
}

/// `torch.serialization`'s `persistent_load`: `("storage", <Type>Storage, key, location, numel)`.
fn persistent_load(pid: Value) -> Result<Value> {
    let Value::Tuple(items) = pid else {
        bail!("persistent id is not a tuple");
    };
    match items.as_slice() {
        [Value::Str(tag), Value::Global(_, ty), Value::Str(key), ..] if tag == "storage" => {
            Ok(Value::Storage {
                dtype: StorageType::from_global(ty)?,
                key: key.clone(),
            })
        }
        _ => bail!("unsupported persistent id {items:?}"),
    }
}

/// Applies the few callables a checkpoint needs; anything else is kept as an opaque value.
fn reduce(func: Value, args: Value) -> Result<Value> {
    let Value::Global(module, name) = &func else {
        return Ok(Value::Opaque);
    };
    let args = match args {
        Value::Tuple(a) => a,
        _ => return Ok(Value::Opaque),
    };
    match (module.as_str(), name.as_str()) {
        ("collections", "OrderedDict") | ("builtins", "dict") => {
            Ok(Value::Dict(Rc::new(RefCell::new(Vec::new()))))
        }
        ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
            let [Value::Storage { dtype, key }, Value::Int(offset), size, stride, ..] =
                args.as_slice()
            else {
                bail!("unexpected _rebuild_tensor_v2 arguments");
            };
            Ok(Value::Tensor(Rc::new(TensorRef {
                dtype: *dtype,
                key: key.clone(),
                offset: usize::try_from(*offset).context("negative storage offset")?,
                size: ints(size)?,
                stride: ints(stride)?,
            })))
        }
        ("torch._utils", "_rebuild_parameter")
        | ("torch._utils", "_rebuild_parameter_with_state") => args
            .into_iter()
            .next()
            .context("_rebuild_parameter without data"),
        _ => Ok(Value::Opaque),
    }
}

/// A tuple of non-negative integers.
fn ints(v: &Value) -> Result<Vec<usize>> {
    let Value::Tuple(items) = v else {
        bail!("expected a tuple of sizes");
    };
    items
        .iter()
        .map(|i| match i {
            Value::Int(n) => usize::try_from(*n).context("negative size"),
            _ => bail!("expected an integer size"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_decodes() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
        assert!(f16_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn rejects_lfs_pointers_and_non_archives() {
        let err = from_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:00\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("git lfs pull"), "{err}");
        assert!(from_bytes(b"\x80\x02}q\x00.").is_err());
    }

    #[test]
    fn strided_views_gather_in_row_major_order() {
        // a 2x3 storage read transposed (3x2, strides (1, 3)) from offset 0
        let raw: Vec<u8> = [0f32, 1., 2., 3., 4., 5.]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = TensorRef {
            dtype: StorageType::F32,
            key: "0".into(),
            offset: 0,
            size: vec![3, 2],
            stride: vec![1, 3],
        };
        assert_eq!(gather(&t, &raw).unwrap(), [0., 3., 1., 4., 2., 5.]);
    }

    /// Every general checkpoint loads and matches the safetensors fixture exported from it.
    #[test]
    fn general_checkpoints_match_their_safetensors_exports() {
        let Some(root) = crate::data::repo_root() else {
            return;
        };
        for kind in ["behavior", "detector", "pose", "temporal"] {
            let pt = root.join("weights").join(format!("{kind}_general.pt"));
            let Ok(bytes) = std::fs::read(&pt) else {
                eprintln!("SKIP: {} is absent", pt.display());
                continue;
            };
            if is_lfs_pointer(&bytes) {
                eprintln!("SKIP: {} is a git-lfs pointer", pt.display());
                continue;
            }
            let w = from_bytes(&bytes).unwrap();
            assert_eq!(w.metadata_value(KIND_KEY), Some(kind));
            assert!(w.metadata_value(CONFIG_KEY).is_some());
            assert!(w.metadata_value(EXTRA_KEY).is_some(), "{kind}: no extra");
            let st = root
                .join("tests/fixtures/models")
                .join(format!("{kind}_general.safetensors"));
            if let Ok(reference) = Weights::read(&st) {
                assert_eq!(
                    w.names().collect::<Vec<_>>(),
                    reference.names().collect::<Vec<_>>()
                );
                for name in reference.names() {
                    assert_eq!(w.shape(name), reference.shape(name), "{name}");
                    assert_eq!(w.data(name), reference.data(name), "{name}");
                }
                let a: serde_json::Value =
                    serde_json::from_str(w.metadata_value(CONFIG_KEY).unwrap()).unwrap();
                let b: serde_json::Value =
                    serde_json::from_str(reference.metadata_value(CONFIG_KEY).unwrap()).unwrap();
                assert_eq!(a, b, "{kind}: model_cfg");
            }
        }
    }
}
