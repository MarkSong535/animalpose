//! `.npz` reading and writing with `float64` members, which the multi-view outputs and inputs use
//! (`animals3d.npz` holds `np.float64` keypoints; `--refine obs.npz` is usually `float64`).
//!
//! [`crate::data::sessions`] writes `float32` only; `float64` members are encoded here and every
//! other dtype goes through [`crate::data::sessions::encode_npy`].

use std::io::{Read as _, Write as _};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};

use crate::data::sessions::{encode_npy, NpyArray};

/// One `.npz` member.
#[derive(Clone, Debug)]
pub enum Member {
    /// A `<f8` array: shape and row-major values.
    F64(Vec<usize>, Vec<f64>),
    /// Any dtype [`encode_npy`] writes.
    Npy(NpyArray),
}

fn shape_text(shape: &[usize]) -> String {
    if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// `np.save` of a `float64` array (format 1.0, header padded to 64 bytes).
pub fn encode_npy_f64(shape: &[usize], values: &[f64]) -> Vec<u8> {
    let mut header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': {}, }}",
        shape_text(shape)
    );
    let pad = 64 - (10 + header.len() + 1) % 64;
    header.push_str(&" ".repeat(pad % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + values.len() * 8);
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// `np.savez_compressed(path, **members)`, members in the order given.
pub fn write_npz(path: impl AsRef<Path>, members: &[(String, Member)]) -> Result<()> {
    use zip::write::SimpleFileOptions;
    let path = path.as_ref();
    let file =
        std::fs::File::create(path).with_context(|| format!("cannot write {}", path.display()))?;
    let mut zip = zip::ZipWriter::new(std::io::BufWriter::new(file));
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (name, member) in members {
        zip.start_file(format!("{name}.npy"), options)
            .with_context(|| format!("cannot add {name}.npy to {}", path.display()))?;
        let bytes = match member {
            Member::F64(shape, values) => encode_npy_f64(shape, values),
            Member::Npy(array) => encode_npy(array),
        };
        zip.write_all(&bytes)?;
    }
    zip.finish()?;
    Ok(())
}

/// `np.load(path)[key]` as `float64` (from `<f8`, `<f4`, `<i8` or `<i4` members): shape and
/// row-major values.
pub fn read_npz_f64(path: impl AsRef<Path>, key: &str) -> Result<(Vec<usize>, Vec<f64>)> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut zip =
        zip::ZipArchive::new(file).with_context(|| format!("{} is not an .npz", path.display()))?;
    let mut member = zip
        .by_name(&format!("{key}.npy"))
        .with_context(|| format!("{} has no array {key:?}", path.display()))?;
    let mut bytes = Vec::new();
    member.read_to_end(&mut bytes)?;
    parse_npy_f64(&bytes).with_context(|| format!("{}[{key:?}]", path.display()))
}

/// Parses one `.npy` member into `float64`.
pub fn parse_npy_f64(bytes: &[u8]) -> Result<(Vec<usize>, Vec<f64>)> {
    ensure!(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "not a .npy array"
    );
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            ensure!(bytes.len() >= 12, "truncated .npy header");
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        v => bail!("unsupported .npy version {v}"),
    };
    ensure!(bytes.len() >= start + header_len, "truncated .npy header");
    let header = std::str::from_utf8(&bytes[start..start + header_len])?;
    let field = |name: &str| -> Result<&str> {
        let at = header
            .find(&format!("'{name}'"))
            .with_context(|| format!("no {name} in the .npy header"))?;
        Ok(header[at + name.len() + 2..].trim_start_matches([':', ' ']))
    };
    let descr = field("descr")?;
    let descr = &descr[1..descr[1..].find('\'').map_or(0, |i| i + 1)];
    ensure!(
        field("fortran_order")?.starts_with("False"),
        "Fortran-ordered arrays are not supported"
    );
    let shape_text = field("shape")?;
    let close = shape_text
        .find(')')
        .context("bad shape in the .npy header")?;
    let shape: Vec<usize> = shape_text[1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()
        .context("bad shape in the .npy header")?;
    let count: usize = shape.iter().product();
    let body = &bytes[start + header_len..];
    let values: Vec<f64> = match descr {
        "<f8" => body
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect(),
        "<f4" => body
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes(c.try_into().expect("4 bytes"))))
            .collect(),
        "<i8" => body
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")) as f64)
            .collect(),
        "<i4" => body
            .chunks_exact(4)
            .map(|c| f64::from(i32::from_le_bytes(c.try_into().expect("4 bytes"))))
            .collect(),
        other => bail!("unsupported dtype {other:?} (want float64, float32 or an integer)"),
    };
    ensure!(values.len() >= count, "truncated .npy body");
    Ok((shape, values[..count].to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float64_members_round_trip() {
        let dir = std::env::temp_dir().join(format!("animalpose-mv-npz-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("x.npz");
        let values = vec![1.5, f64::NAN, -2.25, 1e-300, 7.0, 8.0];
        write_npz(
            &path,
            &[(
                "obs".to_string(),
                Member::F64(vec![1, 3, 2], values.clone()),
            )],
        )
        .expect("writes");
        let (shape, back) = read_npz_f64(&path, "obs").expect("reads");
        assert_eq!(shape, vec![1, 3, 2]);
        assert!(back[1].is_nan());
        assert_eq!(
            (back[0], back[2], back[3], back[5]),
            (1.5, -2.25, 1e-300, 8.0)
        );
        let bytes = encode_npy_f64(&[3], &[1.0, 2.0, 3.0]);
        assert_eq!((bytes.len() - 24) % 64, 0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
