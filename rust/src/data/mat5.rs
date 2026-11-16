//! MATLAB level-5 MAT-files (`.mat`, versions 5 to 7.2): the subset `scipy.io.loadmat` reads for
//! the Rat 7M motion capture — numeric arrays, character arrays, structs and cells, compressed
//! (`miCOMPRESSED`, zlib) or not. Version 7.3 files are HDF5 and are refused.
//!
//! [`MatValue`] keeps MATLAB's column-major order; [`MatValue::rows`] hands a 2-D numeric array back
//! row by row the way numpy indexes it (`a[i]`), and `squeeze_me=True` is what
//! [`MatValue::values`] amounts to for vectors.

use anyhow::{bail, ensure, Context, Result};
use std::io::Read;
use std::path::Path;

/// One MATLAB value.
#[derive(Clone, Debug, PartialEq)]
pub enum MatValue {
    /// A numeric or logical array of any class, converted to `f64`, column-major.
    Num {
        /// MATLAB dimensions.
        dims: Vec<usize>,
        /// Column-major values.
        data: Vec<f64>,
    },
    /// A character array.
    Char(String),
    /// A struct array; `elems[i]` holds the fields of element `i`, in `fields` order.
    Struct {
        /// MATLAB dimensions.
        dims: Vec<usize>,
        /// Field names in declaration order (`_fieldnames`).
        fields: Vec<String>,
        /// Per element, one value per field.
        elems: Vec<Vec<MatValue>>,
    },
    /// A cell array, column-major.
    Cell {
        /// MATLAB dimensions.
        dims: Vec<usize>,
        /// The cells.
        items: Vec<MatValue>,
    },
    /// A class this reader does not decode (sparse, objects, function handles).
    Other,
}

impl MatValue {
    /// Field `name` of a 1x1 struct (`getattr(s, name)` with `struct_as_record=False`).
    pub fn field(&self, name: &str) -> Option<&MatValue> {
        match self {
            MatValue::Struct { fields, elems, .. } => {
                let i = fields.iter().position(|f| f == name)?;
                elems.first().map(|e| &e[i])
            }
            _ => None,
        }
    }

    /// The field names of a struct (`s._fieldnames`).
    pub fn field_names(&self) -> &[String] {
        match self {
            MatValue::Struct { fields, .. } => fields,
            _ => &[],
        }
    }

    /// All values of a numeric array, column-major (a squeezed vector's values).
    pub fn values(&self) -> Option<&[f64]> {
        match self {
            MatValue::Num { data, .. } => Some(data),
            _ => None,
        }
    }

    /// A 2-D numeric array `(r, c)` as `r` rows of `c` values (`np.asarray(a)[i]`).
    pub fn rows(&self) -> Option<Vec<Vec<f64>>> {
        let MatValue::Num { dims, data } = self else {
            return None;
        };
        let r = dims.first().copied().unwrap_or(0);
        let c: usize = dims.iter().skip(1).product();
        Some((0..r).map(|i| (0..c).map(|j| data[j * r + i]).collect()).collect())
    }
}

/// `scipy.io.loadmat(path)`: the top-level variables in file order.
pub fn read_mat(path: &Path) -> Result<Vec<(String, MatValue)>> {
    let bytes = std::fs::read(path).with_context(|| format!("[Errno 2] No such file or directory: '{}'", path.display()))?;
    parse_mat(&bytes).with_context(|| format!("cannot read the MAT-file {}", path.display()))
}

/// [`read_mat`] on the bytes of a file.
pub fn parse_mat(bytes: &[u8]) -> Result<Vec<(String, MatValue)>> {
    ensure!(bytes.len() >= 128, "shorter than a MAT-file header");
    let version = u16::from_le_bytes([bytes[124], bytes[125]]);
    ensure!(&bytes[126..128] == b"IM", "big-endian MAT-files are not supported");
    if version == 0x0200 {
        bail!("unsupported: a MATLAB 7.3 (HDF5) file");
    }
    let mut out = Vec::new();
    let mut at = 128;
    while at + 8 <= bytes.len() {
        let (ty, data, next) = element(bytes, at)?;
        at = next;
        let owned;
        let (ty, data) = if ty == MI_COMPRESSED {
            let mut z = flate2::read::ZlibDecoder::new(data);
            let mut buf = Vec::new();
            z.read_to_end(&mut buf).context("bad compressed element")?;
            owned = buf;
            let (t, d, _) = element(&owned, 0)?;
            (t, d)
        } else {
            (ty, data)
        };
        if ty == MI_MATRIX {
            let (name, value) = matrix(data)?;
            out.push((name, value));
        }
    }
    Ok(out)
}

const MI_INT8: u32 = 1;
const MI_UINT8: u32 = 2;
const MI_INT16: u32 = 3;
const MI_UINT16: u32 = 4;
const MI_INT32: u32 = 5;
const MI_UINT32: u32 = 6;
const MI_SINGLE: u32 = 7;
const MI_DOUBLE: u32 = 9;
const MI_INT64: u32 = 12;
const MI_UINT64: u32 = 13;
const MI_MATRIX: u32 = 14;
const MI_COMPRESSED: u32 = 15;
const MI_UTF8: u32 = 16;
const MI_UTF16: u32 = 17;

/// One data element at `at`: `(type, data, offset of the next element)`.
fn element(b: &[u8], at: usize) -> Result<(u32, &[u8], usize)> {
    ensure!(at + 8 <= b.len(), "truncated element tag");
    let first = u32::from_le_bytes(b[at..at + 4].try_into()?);
    if first >> 16 != 0 {
        // Small data element: type and size share the first word, data in the next four bytes.
        let (ty, n) = (first & 0xffff, (first >> 16) as usize);
        ensure!(n <= 4, "malformed small data element");
        return Ok((ty, &b[at + 4..at + 4 + n], at + 8));
    }
    let n = u32::from_le_bytes(b[at + 4..at + 8].try_into()?) as usize;
    let start = at + 8;
    ensure!(start + n <= b.len(), "truncated element of {n} bytes");
    let next = if first == MI_COMPRESSED { start + n } else { start + n.div_ceil(8) * 8 };
    Ok((first, &b[start..start + n], next))
}

fn numbers(ty: u32, d: &[u8]) -> Result<Vec<f64>> {
    Ok(match ty {
        MI_INT8 => d.iter().map(|v| f64::from(*v as i8)).collect(),
        MI_UINT8 => d.iter().map(|v| f64::from(*v)).collect(),
        MI_INT16 => d.chunks_exact(2).map(|c| f64::from(i16::from_le_bytes([c[0], c[1]]))).collect(),
        MI_UINT16 => d.chunks_exact(2).map(|c| f64::from(u16::from_le_bytes([c[0], c[1]]))).collect(),
        MI_INT32 => d
            .chunks_exact(4)
            .map(|c| f64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        MI_UINT32 => d
            .chunks_exact(4)
            .map(|c| f64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        MI_SINGLE => d
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        MI_DOUBLE => d.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8]))).collect(),
        MI_INT64 => d
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as f64)
            .collect(),
        MI_UINT64 => d
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as f64)
            .collect(),
        other => bail!("unexpected numeric element type {other}"),
    })
}

/// A `miMATRIX` body: `(name, value)`.
fn matrix(b: &[u8]) -> Result<(String, MatValue)> {
    if b.is_empty() {
        return Ok((String::new(), MatValue::Other));
    }
    let (_, flags, at) = element(b, 0)?;
    let class = flags.first().copied().unwrap_or(0);
    let complex = flags.get(1).is_some_and(|f| f & 0x08 != 0);
    let (_, dims, at) = element(b, at)?;
    let dims: Vec<usize> = numbers(MI_INT32, dims)?.iter().map(|v| *v as usize).collect();
    let (_, name, mut at) = element(b, at)?;
    let name = String::from_utf8_lossy(name).into_owned();
    let count: usize = dims.iter().product();
    let value = match class {
        // mxCELL_CLASS
        1 => {
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                let (_, d, next) = element(b, at)?;
                at = next;
                items.push(matrix(d)?.1);
            }
            MatValue::Cell { dims, items }
        }
        // mxSTRUCT_CLASS
        2 => {
            let (_, len, next) = element(b, at)?;
            let len = numbers(MI_INT32, len)?.first().copied().unwrap_or(0.0) as usize;
            let (_, names, next) = element(b, next)?;
            at = next;
            let fields: Vec<String> = if len == 0 {
                Vec::new()
            } else {
                names
                    .chunks(len)
                    .map(|c| {
                        let end = c.iter().position(|v| *v == 0).unwrap_or(c.len());
                        String::from_utf8_lossy(&c[..end]).into_owned()
                    })
                    .collect()
            };
            let mut elems = Vec::with_capacity(count);
            for _ in 0..count {
                let mut values = Vec::with_capacity(fields.len());
                for _ in &fields {
                    let (_, d, next) = element(b, at)?;
                    at = next;
                    values.push(matrix(d)?.1);
                }
                elems.push(values);
            }
            MatValue::Struct { dims, fields, elems }
        }
        // mxCHAR_CLASS
        4 => {
            let (ty, d, _) = element(b, at)?;
            let text = match ty {
                MI_UTF8 | MI_UINT8 | MI_INT8 => String::from_utf8_lossy(d).into_owned(),
                MI_UTF16 | MI_UINT16 => {
                    let units: Vec<u16> = d.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                    String::from_utf16_lossy(&units)
                }
                _ => String::from_utf8_lossy(d).into_owned(),
            };
            MatValue::Char(text)
        }
        // mxDOUBLE .. mxUINT64, and logical arrays stored as uint8.
        6..=15 => {
            let (ty, d, _) = element(b, at)?;
            let _ = complex; // the imaginary part, when present, follows and is dropped
            MatValue::Num {
                dims,
                data: numbers(ty, d)?,
            }
        }
        _ => MatValue::Other,
    };
    Ok((name, value))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tiny MAT-file writer (uncompressed or zlib-compressed `miMATRIX` elements), enough to
    /// build the files `scipy.io.savemat` writes for the tests.
    pub(crate) fn tag(ty: u32, data: &[u8]) -> Vec<u8> {
        let mut out = ty.to_le_bytes().to_vec();
        out.extend((data.len() as u32).to_le_bytes());
        out.extend(data);
        while out.len() % 8 != 0 {
            out.push(0);
        }
        out
    }

    pub(crate) fn matrix_bytes(name: &str, class: u8, dims: &[usize], body: &[u8]) -> Vec<u8> {
        let mut m = tag(MI_UINT32, &[class, 0, 0, 0, 0, 0, 0, 0]);
        let d: Vec<u8> = dims.iter().flat_map(|v| (*v as i32).to_le_bytes()).collect();
        m.extend(tag(MI_INT32, &d));
        m.extend(tag(MI_INT8, name.as_bytes()));
        m.extend(body);
        tag(MI_MATRIX, &m)
    }

    /// A double matrix given row by row.
    pub(crate) fn double(name: &str, rows: &[Vec<f64>]) -> Vec<u8> {
        let (r, c) = (rows.len(), rows.first().map_or(0, Vec::len));
        let data: Vec<u8> = (0..c)
            .flat_map(|j| (0..r).map(move |i| (i, j)))
            .flat_map(|(i, j)| rows[i][j].to_le_bytes())
            .collect();
        matrix_bytes(name, 6, &[r, c], &tag(MI_DOUBLE, &data))
    }

    /// A 1x1 struct of already-encoded (unnamed) fields.
    pub(crate) fn structure(name: &str, fields: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let len = fields.iter().map(|f| f.0.len()).max().unwrap_or(0) + 1;
        let mut body = tag(MI_INT32, &(len as i32).to_le_bytes());
        let mut names = Vec::new();
        for (f, _) in fields {
            let mut n = f.as_bytes().to_vec();
            n.resize(len, 0);
            names.extend(n);
        }
        body.extend(tag(MI_INT8, &names));
        for (_, v) in fields {
            body.extend(v);
        }
        matrix_bytes(name, 2, &[1, 1], &body)
    }

    pub(crate) fn file(vars: &[Vec<u8>], compress: bool) -> Vec<u8> {
        let mut out = vec![b' '; 116];
        out[..10].copy_from_slice(b"MATLAB 5.0");
        out.extend([0u8; 8]);
        out.extend([0x00, 0x01, b'I', b'M']);
        for v in vars {
            if compress {
                let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
                std::io::Write::write_all(&mut z, v).unwrap();
                let c = z.finish().unwrap();
                out.extend(MI_COMPRESSED.to_le_bytes());
                out.extend((c.len() as u32).to_le_bytes());
                out.extend(c);
            } else {
                out.extend(v);
            }
        }
        out
    }

    #[test]
    fn structs_of_matrices_read_back_in_numpy_order() {
        for compress in [false, true] {
            let inner = structure(
                "",
                &[
                    ("K", double("", &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])),
                    ("frame", double("", &[vec![10.0, 11.0, 12.0]])),
                ],
            );
            let bytes = file(&[structure("cameras", &[("Camera1", inner)]), double("x", &[vec![7.0]])], compress);
            let vars = parse_mat(&bytes).unwrap();
            assert_eq!(vars.iter().map(|v| v.0.as_str()).collect::<Vec<_>>(), ["cameras", "x"]);
            let cam = vars[0].1.field("Camera1").unwrap();
            assert_eq!(cam.field_names(), ["K", "frame"]);
            assert_eq!(
                cam.field("K").unwrap().rows().unwrap(),
                vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
            );
            assert_eq!(cam.field("frame").unwrap().values().unwrap(), [10.0, 11.0, 12.0]);
        }
    }
}
