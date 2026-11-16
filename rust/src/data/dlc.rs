//! DeepLabCut exports (`CollectedData_<scorer>.csv`) as COCO keypoint annotations: the port of
//! `src/animalpose/data/dlc.py`.
//!
//! | Python | Rust |
//! | --- | --- |
//! | `data/dlc.py:read_dlc_csv` | [`read_dlc_csv`] |
//! | `data/dlc.py:to_coco` | [`to_coco`] |
//! | `csv.reader` | [`read_csv`] |

use crate::data::pyjson::PyVal;
use anyhow::{bail, Context, Result};
use std::path::Path;

/// What [`read_dlc_csv`] returns: `coords` is `(N_images, N_individuals, N_bodyparts, 2)`
/// row-major with NaN for unlabeled points.
#[derive(Clone, Debug, PartialEq)]
pub struct DlcLabels {
    /// One relative image path per row (`labeled-data/video/img.png`).
    pub images: Vec<String>,
    /// Individual names in first-seen order (`["single"]` for a single-animal export).
    pub individuals: Vec<String>,
    /// Body-part names of the first individual, in first-seen order.
    pub bodyparts: Vec<String>,
    /// The coordinates, `images.len() * individuals.len() * bodyparts.len() * 2` values.
    pub coords: Vec<f64>,
}

impl DlcLabels {
    /// The `(x, y)` of one point.
    pub fn point(&self, image: usize, individual: usize, bodypart: usize) -> [f64; 2] {
        let at = ((image * self.individuals.len() + individual) * self.bodyparts.len() + bodypart) * 2;
        [self.coords[at], self.coords[at + 1]]
    }
}

/// Python's `csv.reader` over a whole file: comma-separated, `"`-quoted fields with `""` escapes,
/// `\n` or `\r\n` line ends. A blank line is an empty row.
pub fn read_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    let mut row_started = false;
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => {
                quoted = true;
                row_started = true;
            }
            ',' => {
                row.push(std::mem::take(&mut field));
                row_started = true;
            }
            '\r' => {}
            '\n' => {
                if row_started || !field.is_empty() {
                    row.push(std::mem::take(&mut field));
                }
                rows.push(std::mem::take(&mut row));
                row_started = false;
            }
            c => {
                field.push(c);
                row_started = true;
            }
        }
    }
    if row_started || !field.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

fn unique(items: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for item in items {
        if !out.contains(&item) {
            out.push(item);
        }
    }
    out
}

/// `read_dlc_csv(path)`: single-animal (3 header rows: scorer/bodyparts/coords) and multi-animal
/// (4 header rows: scorer/individuals/bodyparts/coords) exports; the image path takes one column
/// (old format) or three (`labeled-data/video/img.png` split over columns).
pub fn read_dlc_csv(path: impl AsRef<Path>) -> Result<DlcLabels> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read the DeepLabCut CSV {}", path.display()))?;
    parse_dlc_csv(&text).with_context(|| format!("{} is not a DeepLabCut CSV", path.display()))
}

/// [`read_dlc_csv`] on the text of the file.
pub fn parse_dlc_csv(text: &str) -> Result<DlcLabels> {
    let rows = read_csv(text);
    let cell = |r: usize, c: usize| -> &str { rows.get(r).and_then(|row| row.get(c)).map_or("", String::as_str) };
    if rows.len() < 3 {
        bail!("fewer than three header rows");
    }
    let multi = cell(1, 0).trim().to_lowercase() == "individuals";
    let n_header = if multi { 4 } else { 3 };
    if rows.len() < n_header {
        bail!("fewer than {n_header} header rows");
    }
    let header = &rows[..n_header];
    let first = header[n_header - 1]
        .iter()
        .position(|v| v == "x" || v == "y")
        .context("no x/y coordinate column in the header")?;
    let n_cols = header[0].len().saturating_sub(first);
    let tail = |r: &Vec<String>| -> Vec<String> { r.iter().skip(first).cloned().collect() };
    let individuals: Vec<String> = if multi {
        tail(&header[1])
    } else {
        vec!["single".to_string(); n_cols]
    };
    let bodyparts = tail(&header[n_header - 2]);
    let kinds = tail(&header[n_header - 1]);
    let ind_names = unique(individuals.iter().cloned());
    let first_ind = ind_names.first().cloned().unwrap_or_default();
    let bp_names = unique(
        bodyparts
            .iter()
            .zip(&individuals)
            .filter(|(_, i)| **i == first_ind)
            .map(|(b, _)| b.clone()),
    );
    // `zip(cols, individuals, bodyparts, coords_kind)` stops at the shortest.
    let n_zip = n_cols
        .min(individuals.len())
        .min(bodyparts.len())
        .min(kinds.len());
    let (ni, nb) = (ind_names.len(), bp_names.len());
    let mut images = Vec::new();
    let mut coords = Vec::new();
    for r in &rows[n_header..] {
        if r.is_empty() {
            continue;
        }
        let parts: Vec<&str> = r
            .iter()
            .take(first)
            .map(String::as_str)
            .filter(|v| !v.is_empty())
            .collect();
        images.push(parts.join("/"));
        let mut arr = vec![f64::NAN; ni * nb * 2];
        for z in 0..n_zip {
            let c = first + z;
            let bp = &bodyparts[z];
            let Some(b) = bp_names.iter().position(|n| n == bp) else {
                continue;
            };
            let value = r.get(c).with_context(|| format!("row {:?} is shorter than the header", r.first()))?;
            if value.is_empty() {
                continue;
            }
            let i = ind_names
                .iter()
                .position(|n| *n == individuals[z])
                .unwrap_or(0);
            let d = usize::from(kinds[z] != "x");
            arr[(i * nb + b) * 2 + d] = value
                .trim()
                .parse::<f64>()
                .with_context(|| format!("could not convert string to float: {value:?}"))?;
        }
        coords.extend(arr);
    }
    if images.is_empty() {
        bail!("no labelled rows");
    }
    Ok(DlcLabels {
        images,
        individuals: ind_names,
        bodyparts: bp_names,
        coords,
    })
}

/// `to_coco(images, individuals, bodyparts, coords, image_root, pad)`: one COCO dict, boxes from the
/// labelled points (padded by `pad` of their extent plus 5 px), names kept as they are. With an
/// `image_root`, each image's `height` / `width` is read from the file when it can be opened.
pub fn to_coco(labels: &DlcLabels, image_root: Option<&Path>, pad: f64) -> PyVal {
    let mut images = Vec::new();
    let mut annotations = Vec::new();
    let (ni, nb) = (labels.individuals.len(), labels.bodyparts.len());
    for (i, name) in labels.images.iter().enumerate() {
        let mut entry = PyVal::dict().with("id", i).with("file_name", name.as_str());
        if let Some(root) = image_root {
            if let Ok((w, h)) = image::image_dimensions(root.join(name)) {
                entry.set("height", i64::from(h));
                entry.set("width", i64::from(w));
            }
        }
        images.push(entry);
        for a in 0..ni {
            let kp: Vec<[f64; 2]> = (0..nb).map(|b| labels.point(i, a, b)).collect();
            let ok: Vec<bool> = kp.iter().map(|p| p[0].is_finite() && p[1].is_finite()).collect();
            let n_ok = ok.iter().filter(|v| **v).count();
            if n_ok < 2 {
                continue;
            }
            let mut lo = [f64::INFINITY; 2];
            let mut hi = [f64::NEG_INFINITY; 2];
            for (p, _) in kp.iter().zip(&ok).filter(|(_, v)| **v) {
                for d in 0..2 {
                    lo[d] = lo[d].min(p[d]);
                    hi[d] = hi[d].max(p[d]);
                }
            }
            let dd = [(hi[0] - lo[0]) * pad + 5.0, (hi[1] - lo[1]) * pad + 5.0];
            let (x0, y0) = (lo[0] - dd[0], lo[1] - dd[1]);
            let (w, h) = (hi[0] - lo[0] + 2.0 * dd[0], hi[1] - lo[1] + 2.0 * dd[1]);
            let mut flat = Vec::with_capacity(nb * 3);
            for (p, v) in kp.iter().zip(&ok) {
                if *v {
                    flat.extend([PyVal::Float(p[0]), PyVal::Float(p[1]), PyVal::Int(2)]);
                } else {
                    flat.extend([PyVal::Float(0.0), PyVal::Float(0.0), PyVal::Int(0)]);
                }
            }
            annotations.push(
                PyVal::dict()
                    .with("id", annotations.len())
                    .with("image_id", i)
                    .with("category_id", 1i64)
                    .with("keypoints", PyVal::List(flat))
                    .with("num_keypoints", n_ok)
                    .with("bbox", vec![x0, y0, w, h])
                    .with("area", w * h)
                    .with("iscrowd", 0i64)
                    .with("track_id", a),
            );
        }
    }
    let category = PyVal::dict()
        .with("id", 1i64)
        .with("name", "animal")
        .with("keypoints", labels.bodyparts.clone())
        .with("skeleton", PyVal::List(Vec::new()));
    PyVal::dict()
        .with("images", PyVal::List(images))
        .with("annotations", PyVal::List(annotations))
        .with("categories", PyVal::List(vec![category]))
}

/// Python's `repr(list_of_str)`: `['a', 'b']`.
pub fn repr_str_list(items: &[String]) -> String {
    let inner: Vec<String> = items
        .iter()
        .map(|s| {
            if s.contains('\'') && !s.contains('"') {
                format!("\"{s}\"")
            } else {
                format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
            }
        })
        .collect();
    format!("[{}]", inner.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xy(names: &[&str]) -> Vec<String> {
        names
            .iter()
            .flat_map(|n| [n.to_string(), n.to_string()])
            .collect()
    }

    /// `tests/test_cli_behavior.py:dlc_csv(multi=True)`.
    fn multi_csv() -> String {
        let parts = ["nose", "neck", "tail_base"];
        let mut rows: Vec<Vec<String>> = vec![
            [vec!["scorer".to_string(); 3], vec!["Me".to_string(); 12]].concat(),
            [
                vec!["individuals".to_string(); 3],
                vec!["mouse1".to_string(); 6],
                vec!["mouse2".to_string(); 6],
            ]
            .concat(),
            [vec!["bodyparts".to_string(); 3], xy(&parts), xy(&parts)].concat(),
            [
                vec!["coords".to_string(); 3],
                ["x", "y"].repeat(6).iter().map(|s| s.to_string()).collect(),
            ]
            .concat(),
        ];
        let data = [
            "labeled-data,vid1,img0.png,10,20,25,22,40,28,60,20,75,22,90,28",
            "labeled-data,vid1,img1.png,12,22,27,24,42,30,,,,,,",
            "labeled-data,vid1,img2.png,,,,,,,,,,,,",
        ];
        for d in data {
            rows.push(d.split(',').map(str::to_string).collect());
        }
        rows.iter().map(|r| r.join(",")).collect::<Vec<_>>().join("\n") + "\n"
    }

    #[test]
    fn a_multi_animal_export_becomes_coco_with_padded_boxes() {
        let labels = parse_dlc_csv(&multi_csv()).unwrap();
        assert_eq!(labels.individuals, ["mouse1", "mouse2"]);
        assert_eq!(labels.bodyparts, ["nose", "neck", "tail_base"]);
        assert_eq!(labels.images[0], "labeled-data/vid1/img0.png");
        assert!(labels.point(1, 1, 0)[0].is_nan());
        let coco = to_coco(&labels, None, 0.1);
        let anns = coco.get("annotations").unwrap().as_list().unwrap();
        assert_eq!(anns.len(), 3);
        let tracks: Vec<f64> = anns
            .iter()
            .map(|a| a.get("track_id").unwrap().as_f64().unwrap())
            .collect();
        assert_eq!(tracks, [0.0, 1.0, 0.0]);
        let bbox: Vec<f64> = anns[0]
            .get("bbox")
            .unwrap()
            .as_list()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        // pytest.approx([2.0, 14.2, 46.0, 19.6])
        for (got, want) in bbox.iter().zip([2.0, 14.2, 46.0, 19.6]) {
            assert!((got - want).abs() < 1e-9, "{bbox:?}");
        }
        // The JSON text itself: python -c 'json.dumps(to_coco(*read_dlc_csv(p)))' on the same file.
        let want = concat!(
            r#"{"images": [{"id": 0, "file_name": "labeled-data/vid1/img0.png"}, {"id": 1, "file_name": "labeled-data/vid1/img1.png"}, {"id": 2, "file_name": "labeled-data/vid1/img2.png"}], "#,
            r#""annotations": [{"id": 0, "image_id": 0, "category_id": 1, "keypoints": [10.0, 20.0, 2, 25.0, 22.0, 2, 40.0, 28.0, 2], "num_keypoints": 3, "bbox": [2.0, 14.2, 46.0, 19.6], "area": 901.6, "iscrowd": 0, "track_id": 0}, "#,
            r#"{"id": 1, "image_id": 0, "category_id": 1, "keypoints": [60.0, 20.0, 2, 75.0, 22.0, 2, 90.0, 28.0, 2], "num_keypoints": 3, "bbox": [52.0, 14.2, 46.0, 19.6], "area": 901.6, "iscrowd": 0, "track_id": 1}, "#,
            r#"{"id": 2, "image_id": 1, "category_id": 1, "keypoints": [12.0, 22.0, 2, 27.0, 24.0, 2, 42.0, 30.0, 2], "num_keypoints": 3, "bbox": [4.0, 16.2, 46.0, 19.6], "area": 901.6, "iscrowd": 0, "track_id": 0}], "#,
            r#""categories": [{"id": 1, "name": "animal", "keypoints": ["nose", "neck", "tail_base"], "skeleton": []}]}"#
        );
        // `json.dumps(to_coco(*read_dlc_csv(p)))` on the same file, byte for byte.
        assert_eq!(coco.dumps(), want);
    }

    #[test]
    fn a_single_animal_export_has_one_individual() {
        let text = "scorer,Me,Me,Me,Me,Me,Me\nbodyparts,snout,snout,tailbase,tailbase,leftear,leftear\n\
                    coords,x,y,x,y,x,y\nimg0.png,10,20,25,22,40,28\nimg1.png,12,22,27,24,42,30\n";
        let labels = parse_dlc_csv(text).unwrap();
        assert_eq!(labels.individuals, ["single"]);
        assert_eq!(labels.bodyparts, ["snout", "tailbase", "leftear"]);
        assert_eq!(labels.images, ["img0.png", "img1.png"]);
        assert_eq!(
            repr_str_list(&labels.bodyparts),
            "['snout', 'tailbase', 'leftear']"
        );
    }

    #[test]
    fn csv_quotes_and_blank_lines() {
        let rows = read_csv("a,\"b,c\",\"d\"\"e\"\n\nx,,\r\n");
        assert_eq!(rows[0], ["a", "b,c", "d\"e"]);
        assert!(rows[1].is_empty());
        assert_eq!(rows[2], ["x", "", ""]);
    }
}
