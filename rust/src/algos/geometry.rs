//! Box and affine-crop helpers shared by the datasets, inference and tracking.
//!
//! Port of `animalpose.geometry` plus the box rescaling that
//! `animalpose.pipeline.PosePipeline.detect` applies after `letterbox`.

/// Axis-aligned box as `[x0, y0, x1, y1]`.
///
/// // differs from Python: the Python code mixes `float32` and `float64`
/// // boxes; every box here is `f64`.
pub type Box4 = [f64; 4];

/// Default `padding` of `animalpose.geometry.box_xyxy_to_cs`.
pub const DEFAULT_CROP_PADDING: f64 = 1.25;

/// Default `aspect` of `animalpose.geometry.box_xyxy_to_cs`.
pub const DEFAULT_CROP_ASPECT: f64 = 1.0;

/// Box to `(center, side)` of the square crop region.
///
/// Port of `animalpose.geometry.box_xyxy_to_cs`.
pub fn box_xyxy_to_cs(bbox: Box4, aspect: f64, padding: f64) -> ([f64; 2], f64) {
    let [x0, y0, x1, y1] = bbox;
    let c = [(x0 + x1) / 2.0, (y0 + y1) / 2.0];
    let (w, h) = ((x1 - x0).max(1.0), (y1 - y0).max(1.0));
    let side = if w > aspect * h { w } else { h * aspect };
    (c, side * padding)
}

/// [`box_xyxy_to_cs`] with the Python defaults (`aspect = 1.0`, `padding = 1.25`).
pub fn box_xyxy_to_cs_default(bbox: Box4) -> ([f64; 2], f64) {
    box_xyxy_to_cs(bbox, DEFAULT_CROP_ASPECT, DEFAULT_CROP_PADDING)
}

/// Square crop region back to an xyxy box: the inverse of
/// [`box_xyxy_to_cs`] with `padding = 1.0`.
///
/// // differs from Python: `geometry.py` has no such helper; it is the
/// // closed-form inverse that `PosePipeline.estimate` needs when it maps
/// // decoded crop keypoints back to image pixels.
pub fn box_cs_to_xyxy(center: [f64; 2], side: f64) -> Box4 {
    let [cx, cy] = center;
    let half = side / 2.0;
    [cx - half, cy - half, cx + half, cy + half]
}

/// IoU of two xyxy boxes.
///
/// Port of `animalpose.pipeline._iou` (exposed here as `box_iou`). Note the
/// `+ 1e-9` in the denominator, which keeps disjoint degenerate boxes finite.
pub fn box_iou(a: Box4, b: Box4) -> f64 {
    let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    inter / (area_a + area_b - inter + 1e-9)
}

/// Pairwise IoU of `N` boxes against `M` boxes, returned row-major `(N, M)`.
///
/// Port of `animalpose.geometry.box_iou` (the broadcasting version); its
/// denominator is clamped at `1e-9` instead of adding it.
pub fn box_iou_pairwise(a: &[Box4], b: &[Box4]) -> Vec<f64> {
    let area = |r: &Box4| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    let mut out = vec![0.0; a.len() * b.len()];
    for (i, ra) in a.iter().enumerate() {
        for (j, rb) in b.iter().enumerate() {
            let tlx = ra[0].max(rb[0]);
            let tly = ra[1].max(rb[1]);
            let brx = ra[2].min(rb[2]);
            let bry = ra[3].min(rb[3]);
            let inter = (brx - tlx).max(0.0) * (bry - tly).max(0.0);
            let union = (area(ra) + area(rb) - inter).max(1e-9);
            out[i * b.len() + j] = inter / union;
        }
    }
    out
}

/// Tight xyxy box around valid keypoints, expanded by `pad` of its size.
///
/// Port of `animalpose.geometry.keypoints_to_box`; `valid = None` means "every
/// finite point" exactly as in Python.
pub fn keypoints_to_box(kpts: &[[f64; 2]], valid: Option<&[bool]>, pad: f64) -> Option<Box4> {
    let keep: Vec<bool> = match valid {
        Some(v) => v.to_vec(),
        None => kpts
            .iter()
            .map(|p| p[0].is_finite() && p[1].is_finite())
            .collect(),
    };
    if !keep.iter().any(|v| *v) {
        return None;
    }
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for (p, k) in kpts.iter().zip(keep.iter()) {
        if !*k {
            continue;
        }
        for d in 0..2 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    let d = [
        (hi[0] - lo[0]).max(4.0) * pad,
        (hi[1] - lo[1]).max(4.0) * pad,
    ];
    Some([lo[0] - d[0], lo[1] - d[1], hi[0] + d[0], hi[1] + d[1]])
}

/// 2x3 matrix mapping image pixels to an `out_size` square crop.
///
/// Port of `animalpose.geometry.crop_affine` (`rotation` in radians).
pub fn crop_affine(center: [f64; 2], side: f64, out_size: u32, rotation: f64) -> [[f64; 3]; 2] {
    let s = f64::from(out_size) / side;
    let (cos, sin) = (rotation.cos() * s, rotation.sin() * s);
    let half = (f64::from(out_size) - 1.0) / 2.0;
    let [cx, cy] = center;
    [
        [cos, sin, half - cos * cx - sin * cy],
        [-sin, cos, half + sin * cx - cos * cy],
    ]
}

/// Inverse of a 2x3 affine map (port of `animalpose.geometry.invert_affine`,
/// i.e. `cv2.invertAffineTransform` with default flags).
pub fn invert_affine(m: [[f64; 3]; 2]) -> [[f64; 3]; 2] {
    let [[a, b, c], [d, e, f]] = m;
    let det = a * e - b * d;
    let inv = 1.0 / det;
    [
        [e * inv, -b * inv, (b * f - e * c) * inv],
        [-d * inv, a * inv, (d * c - a * f) * inv],
    ]
}

/// Applies an affine map to a point (port of `animalpose.geometry.apply_affine`).
pub fn apply_affine(m: [[f64; 3]; 2], pt: [f64; 2]) -> [f64; 2] {
    [
        m[0][0] * pt[0] + m[0][1] * pt[1] + m[0][2],
        m[1][0] * pt[0] + m[1][1] * pt[1] + m[1][2],
    ]
}

/// Maps letterboxed boxes back to image pixels and clips them to the frame.
///
/// Port of the rescaling in `animalpose.pipeline.PosePipeline.detect`
/// (`(boxes - pad) / scale`, then per-axis clipping).
pub fn unletterbox_boxes(
    boxes: &[Box4],
    scale: f64,
    pad: [f64; 2],
    image_size: (u32, u32),
) -> Vec<Box4> {
    let (w, h) = (f64::from(image_size.0), f64::from(image_size.1));
    boxes
        .iter()
        .map(|b| {
            let mut r = [
                (b[0] - pad[0]) / scale,
                (b[1] - pad[1]) / scale,
                (b[2] - pad[0]) / scale,
                (b[3] - pad[1]) / scale,
            ];
            r[0] = r[0].clamp(0.0, w);
            r[2] = r[2].clamp(0.0, w);
            r[1] = r[1].clamp(0.0, h);
            r[3] = r[3].clamp(0.0, h);
            r
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn box_to_center_side_matches_python() {
        // box_xyxy_to_cs(np.array([0., 0., 10., 4.])) -> ([5., 2.], 12.5)
        let (c, side) = box_xyxy_to_cs_default([0.0, 0.0, 10.0, 4.0]);
        assert!(close(c[0], 5.0) && close(c[1], 2.0));
        assert!(close(side, 12.5));
        // A tall box is widened to a square first.
        let (c, side) = box_xyxy_to_cs([0.0, 0.0, 4.0, 10.0], 1.0, 1.25);
        assert!(close(c[0], 2.0) && close(c[1], 5.0));
        assert!(close(side, 12.5));
        // Degenerate boxes still get side 1 * padding.
        let (_, side) = box_xyxy_to_cs([5.0, 5.0, 5.0, 5.0], 1.0, 1.25);
        assert!(close(side, 1.25));
    }

    #[test]
    fn cs_to_xyxy_inverts_the_square_region() {
        let box_sq = box_cs_to_xyxy([5.0, 2.0], 10.0);
        assert_eq!(box_sq, [0.0, -3.0, 10.0, 7.0]);
        let (c, side) = box_xyxy_to_cs(box_sq, 1.0, 1.0);
        assert!(close(c[0], 5.0) && close(c[1], 2.0) && close(side, 10.0));
    }

    #[test]
    fn box_iou_matches_the_pipeline_helper() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 0.0, 15.0, 10.0];
        // inter 50, union 150 -> 1/3 (up to the 1e-9 epsilon)
        assert!((box_iou(a, b) - 1.0 / 3.0).abs() < 1e-9);
        assert!(box_iou(a, [20.0, 0.0, 30.0, 10.0]) < 1e-9);
        assert!(close(box_iou(a, a), 1.0));
        let pairwise = box_iou_pairwise(&[a, b], &[a, [20.0, 0.0, 30.0, 10.0]]);
        assert!(close(pairwise[0], 1.0));
        assert!(pairwise[1] < 1e-9);
        assert!((pairwise[2] - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn keypoints_to_box_pads_and_ignores_invalid_points() {
        let kpts = [[0.0, 0.0], [10.0, 0.0], [f64::NAN, 5.0]];
        let b = keypoints_to_box(&kpts, None, 0.1).unwrap();
        // d = max(10, 4) * 0.1 = 1 on x, max(0, 4) * 0.1 = 0.4 on y
        assert!(close(b[0], -1.0) && close(b[1], -0.4) && close(b[2], 11.0) && close(b[3], 0.4));
        assert!(keypoints_to_box(&[[f64::NAN, 0.0]], None, 0.1).is_none());
        let b = keypoints_to_box(&[[0.0, 0.0], [4.0, 4.0]], Some(&[true, false]), 0.5).unwrap();
        assert!(close(b[0], -2.0) && close(b[2], 2.0));
    }

    #[test]
    fn crop_affine_maps_the_center_to_the_crop_centre() {
        let (c, side) = box_xyxy_to_cs_default([10.0, 10.0, 30.0, 20.0]);
        let m = crop_affine(c, side, 128, 0.0);
        let p = apply_affine(m, c);
        assert!((p[0] - 63.5).abs() < 1e-9, "{p:?}");
        assert!((p[1] - 63.5).abs() < 1e-9);
        let back = apply_affine(invert_affine(m), p);
        assert!((back[0] - c[0]).abs() < 1e-9 && (back[1] - c[1]).abs() < 1e-9);
    }

    #[test]
    fn affine_inverse_round_trips_arbitrary_maps() {
        let m = [[1.5, -0.25, 7.0], [0.4, 2.0, -3.0]];
        let inv = invert_affine(m);
        for p in [[0.0, 0.0], [13.5, -4.25], [64.0, 64.0]] {
            let q = apply_affine(inv, apply_affine(m, p));
            assert!((q[0] - p[0]).abs() < 1e-9 && (q[1] - p[1]).abs() < 1e-9);
        }
    }

    #[test]
    fn unletterbox_removes_padding_and_clips_to_the_frame() {
        // letterbox scale 1.6, pad (0, 16) as in the image tests.
        let out = unletterbox_boxes(&[[0.0, 16.0, 64.0, 48.0]], 1.6, [0.0, 16.0], (40, 20));
        assert!(close(out[0][0], 0.0) && close(out[0][1], 0.0));
        assert!(close(out[0][2], 40.0) && close(out[0][3], 20.0));
        let out = unletterbox_boxes(&[[0.0, 0.0, 100.0, 100.0]], 1.6, [0.0, 16.0], (40, 20));
        assert!(close(out[0][1], 0.0) && close(out[0][3], 20.0));
    }
}
