//! Checkpoint interchange: a `.safetensors` file in, a name-for-name map of tensors out.
//!
//! The names are exactly the keys of the Python `state_dict` (`backbone.encoder.stem.0.0.weight`,
//! `heat_queries`, `blocks.0.conv1.bias`, ...), so [`crate::models::layers::load_into`] can fill a model built
//! from the config without any renaming. The model architecture travels with the weights in the
//! header's `__metadata__` block, which [`crate::models::load`] reads back: `"animalpose"` holds the JSON of
//! the `model_cfg` dict of `src/animalpose/train/checkpoint.py` and `"kind"` the model kind. The
//! producer of those files is `animalpose export safetensors`.
//!
//! ```
//! use animalpose::models::weights::Weights;
//!
//! let mut w = Weights::new();
//! w.insert("heat_bias", &[2], vec![-4.0, -4.0]);
//! w.set_metadata("kind", "pose");
//! let bytes = w.to_bytes().unwrap();
//! let back = Weights::from_bytes(&bytes).unwrap();
//! assert_eq!(back.data("heat_bias").unwrap(), [-4.0, -4.0]);
//! assert_eq!(back.metadata_value("kind"), Some("pose"));
//! ```

use crate::tensor::Shape;
use anyhow::{bail, Context, Result};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use std::collections::BTreeMap;
use std::path::Path;

/// The metadata key holding the JSON model config (`"animalpose"`).
pub const CONFIG_KEY: &str = "animalpose";

/// The metadata key holding the model kind (`"kind"`).
pub const KIND_KEY: &str = "kind";

/// One tensor of a checkpoint: its PyTorch shape and its row-major `f32` values.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightTensor {
    /// PyTorch order: conv weights are `(out, in, kH, kW)`, linear weights `(out, in)`.
    pub shape: Shape,
    /// Row-major values.
    pub data: Vec<f32>,
}

impl WeightTensor {
    /// A tensor of `shape` with `data` values.
    pub fn new(shape: &[usize], data: Vec<f32>) -> Self {
        Self {
            shape: shape.to_vec(),
            data,
        }
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Every tensor of a checkpoint, keyed by its Python state-dict name, plus the header metadata.
#[derive(Clone, Debug, Default)]
pub struct Weights {
    tensors: BTreeMap<String, WeightTensor>,
    metadata: BTreeMap<String, String>,
}

impl Weights {
    /// An empty checkpoint.
    pub fn new() -> Self {
        Self::default()
    }

    /// The tensors, sorted by name.
    pub fn tensors(&self) -> &BTreeMap<String, WeightTensor> {
        &self.tensors
    }

    /// The tensor names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn get(&self, name: &str) -> Option<&WeightTensor> {
        self.tensors.get(name)
    }

    /// The values of one tensor.
    pub fn data(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|t| t.data.as_slice())
    }

    /// The shape of one tensor.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }

    /// Adds (or replaces) one tensor.
    pub fn insert(&mut self, name: impl Into<String>, shape: &[usize], data: Vec<f32>) {
        self.tensors
            .insert(name.into(), WeightTensor::new(shape, data));
    }

    /// The header's `__metadata__` block.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// One metadata entry, e.g. [`CONFIG_KEY`].
    pub fn metadata_value(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Sets one metadata entry.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Reads a checkpoint from disk: `.safetensors`, or a PyTorch `.pt` archive (recognised by its
    /// zip header and read by [`crate::models::torch_pt`]).
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("cannot read the checkpoint {}", path.display()))?;
        if crate::models::torch_pt::is_torch_zip(&bytes)
            || crate::models::torch_pt::is_lfs_pointer(&bytes)
        {
            return crate::models::torch_pt::from_bytes(&bytes)
                .with_context(|| format!("cannot parse the torch checkpoint {}", path.display()));
        }
        Self::from_bytes(&bytes).with_context(|| format!("cannot parse {}", path.display()))
    }

    /// Parses an in-memory `.safetensors` image.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (_, header) =
            SafeTensors::read_metadata(bytes).context("safetensors header is malformed")?;
        let metadata = header
            .metadata()
            .clone()
            .unwrap_or_default()
            .into_iter()
            .collect();
        let file = SafeTensors::deserialize(bytes).context("safetensors file is malformed")?;
        let mut tensors = BTreeMap::new();
        for (name, view) in file.iter() {
            let data =
                values_f32(view.dtype(), view.data()).with_context(|| format!("tensor {name}"))?;
            tensors.insert(
                name.to_string(),
                WeightTensor {
                    shape: view.shape().to_vec(),
                    data,
                },
            );
        }
        Ok(Self { tensors, metadata })
    }

    /// Serialises the tensors and metadata back to the safetensors format.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let buffers: Vec<(String, Vec<u8>, Shape)> = self
            .tensors
            .iter()
            .map(|(name, t)| (name.clone(), to_le_bytes(&t.data), t.shape.clone()))
            .collect();
        let views: Vec<(String, TensorView<'_>)> = buffers
            .iter()
            .map(|(name, bytes, shape)| {
                let view = TensorView::new(Dtype::F32, shape.to_vec(), bytes)
                    .with_context(|| format!("tensor {name}"))?;
                Ok((name.clone(), view))
            })
            .collect::<Result<_>>()?;
        let metadata = (!self.metadata.is_empty()).then(|| {
            self.metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        });
        safetensors::serialize(views, metadata).context("cannot write the safetensors image")
    }

    /// Writes the checkpoint to `path`.
    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("cannot create {}", parent.display()))?;
            }
        }
        std::fs::write(path, self.to_bytes()?)
            .with_context(|| format!("cannot write {}", path.display()))
    }

    /// Copies every matching tensor into one [`crate::tensor::Layer`], returning how many landed.
    pub fn load_layer<L: crate::tensor::Layer + ?Sized>(&self, layer: &mut L) -> usize {
        let mut hit = 0;
        for (name, tensor) in &self.tensors {
            if layer.load(name, &tensor.data) {
                hit += 1;
            }
        }
        hit
    }
}

/// Decodes one tensor's bytes as `f32`, widening the integer types a state dict can hold.
fn values_f32(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect(),
        Dtype::F64 => bytes
            .chunks_exact(8)
            .map(|b| f64::from_bits(le8(b)) as f32)
            .collect(),
        Dtype::I64 => bytes
            .chunks_exact(8)
            .map(|b| (le8(b) as i64) as f32)
            .collect(),
        Dtype::I32 => bytes
            .chunks_exact(4)
            .map(|b| (u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i32) as f32)
            .collect(),
        Dtype::I16 => bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32)
            .collect(),
        Dtype::I8 => bytes.iter().map(|b| (*b as i8) as f32).collect(),
        Dtype::U8 => bytes.iter().map(|b| *b as f32).collect(),
        Dtype::BOOL => bytes
            .iter()
            .map(|b| if *b != 0 { 1.0 } else { 0.0 })
            .collect(),
        other => bail!("unsupported safetensors dtype {other:?}; export the checkpoint as f32"),
    })
}

/// The little-endian `u64` of the first eight bytes.
fn le8(b: &[u8]) -> u64 {
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// The little-endian bytes of an `f32` slice.
fn to_le_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
