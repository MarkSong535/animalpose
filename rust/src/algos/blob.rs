//! Moving-blob detection candidates for a fixed camera.
//!
//! Port of `largest_moving_blob`, `blob_box` and `BlobProposer` from
//! `animalpose.pipeline`, together with the OpenCV primitives they call
//! (`cv2.morphologyEx`, `cv2.connectedComponentsWithStats`).

use crate::algos::geometry::Box4;
use crate::algos::image::{GrayF, Image, ToGray};

/// Morphological erosion with a `k x k` rectangle of ones.
///
/// Port of `cv2.erode(m, np.ones((k, k), np.uint8))` with the default anchor
/// (the kernel centre) and the default `BORDER_CONSTANT`.
///
/// // differs from Python: OpenCV fills the border with the type maximum, so a
/// // pixel outside the image never lowers an erosion (it acts like "set" for a
/// // `0`/`1` mask); that is what is reproduced here.
pub fn erode_rect(mask: &Image, k: u32) -> Image {
    if k == 0 {
        return mask.clone();
    }
    let anchor = (k - 1) / 2;
    let mut out = Image::filled(mask.width, mask.height, 0);
    for y in 0..mask.height {
        for x in 0..mask.width {
            let mut v = 1;
            for dy in 0..k {
                let sy = y as i64 + i64::from(dy) - i64::from(anchor);
                if sy < 0 || sy >= i64::from(mask.height) {
                    continue; // border acts as "set"
                }
                for dx in 0..k {
                    let sx = x as i64 + i64::from(dx) - i64::from(anchor);
                    if sx < 0 || sx >= i64::from(mask.width) {
                        continue;
                    }
                    if mask.pixel(sx as u32, sy as u32) == 0 {
                        v = 0;
                        break;
                    }
                }
                if v == 0 {
                    break;
                }
            }
            out.set_pixel(x, y, v);
        }
    }
    out
}

/// Morphological dilation with a `k x k` rectangle of ones
/// (port of `cv2.dilate(m, np.ones((k, k), np.uint8))`; outside the image acts
/// like "unset").
pub fn dilate_rect(mask: &Image, k: u32) -> Image {
    if k == 0 {
        return mask.clone();
    }
    let anchor = (k - 1) / 2;
    let mut out = Image::filled(mask.width, mask.height, 0);
    for y in 0..mask.height {
        for x in 0..mask.width {
            let mut v = 0;
            'kernel: for dy in 0..k {
                let sy = y as i64 + i64::from(dy) - i64::from(anchor);
                if sy < 0 || sy >= i64::from(mask.height) {
                    continue;
                }
                for dx in 0..k {
                    let sx = x as i64 + i64::from(dx) - i64::from(anchor);
                    if sx < 0 || sx >= i64::from(mask.width) {
                        continue;
                    }
                    if mask.pixel(sx as u32, sy as u32) != 0 {
                        v = 1;
                        break 'kernel;
                    }
                }
            }
            out.set_pixel(x, y, v);
        }
    }
    out
}

/// `cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k)))`: erode then dilate.
pub fn morph_open(mask: &Image, k: u32) -> Image {
    dilate_rect(&erode_rect(mask, k), k)
}

/// `cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k)))`: dilate then erode.
pub fn morph_close(mask: &Image, k: u32) -> Image {
    erode_rect(&dilate_rect(mask, k), k)
}

/// `m &= roi` on two `0`/`1` masks (port of the numpy in-place `&=`).
pub fn mask_and(a: &Image, b: &Image) -> Image {
    let pixels = a
        .pixels
        .iter()
        .zip(b.pixels.iter())
        .map(|(x, y)| u8::from(*x != 0 && *y != 0))
        .collect();
    Image {
        width: a.width,
        height: a.height,
        pixels,
    }
}

/// `a - b` on grey levels (port of the `bg - g` subtraction).
pub fn signed_diff(a: &GrayF, b: &GrayF) -> GrayF {
    let pixels = a
        .pixels
        .iter()
        .zip(b.pixels.iter())
        .map(|(x, y)| x - y)
        .collect();
    GrayF {
        width: a.width,
        height: a.height,
        pixels,
    }
}

/// Bounding box and pixel count of one component, mirroring a row of the
/// `stats` array of `cv2.connectedComponentsWithStats`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Component {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width.
    pub w: u32,
    /// Height.
    pub h: u32,
    /// Pixel count.
    pub area: u32,
}

/// 8-connected components of a `0`/`1` mask.
///
/// Port of `cv2.connectedComponentsWithStats(m)` (which defaults to
/// `connectivity=8`) keeping only the foreground labels, ordered the way
/// OpenCV orders them: by the raster position of their first pixel.
pub fn connected_components(mask: &Image) -> Vec<Component> {
    let w = mask.width as usize;
    let h = mask.height as usize;
    let n = w * h;
    let mut parent: Vec<usize> = (0..n).collect();
    let mut size = vec![1usize; n];
    for y in 0..h {
        for x in 0..w {
            if mask.pixels[y * w + x] == 0 {
                continue;
            }
            for (dx, dy) in [(-1i64, -1i64), (0, -1), (1, -1), (-1, 0)] {
                let sx = x as i64 + dx;
                let sy = y as i64 + dy;
                if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
                    continue;
                }
                let j = sy as usize * w + sx as usize;
                if mask.pixels[j] != 0 {
                    union(&mut parent, &mut size, y * w + x, j);
                }
            }
        }
    }
    // Labels in raster order of their first pixel, like OpenCV's.
    let mut labels: Vec<usize> = vec![usize::MAX; n];
    // (x_min, y_min, x_max, y_max, area) per label.
    let mut spans: Vec<(u32, u32, u32, u32, u32)> = Vec::new();
    for (i, pixel) in mask.pixels.iter().enumerate() {
        if *pixel == 0 {
            continue;
        }
        let (x, y) = ((i % w) as u32, (i / w) as u32);
        let root = find(&mut parent, i);
        if labels[root] == usize::MAX {
            labels[root] = spans.len();
            spans.push((x, y, x, y, 0));
        }
        let s = spans[labels[root]];
        spans[labels[root]] = (s.0.min(x), s.1.min(y), s.2.max(x), s.3.max(y), s.4 + 1);
    }
    spans
        .into_iter()
        .map(|(x0, y0, x1, y1, area)| Component {
            x: x0,
            y: y0,
            w: x1 - x0 + 1,
            h: y1 - y0 + 1,
            area,
        })
        .collect()
}

fn find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    let mut cur = x;
    while parent[cur] != root {
        let next = parent[cur];
        parent[cur] = root;
        cur = next;
    }
    root
}

fn union(parent: &mut [usize], size: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra == rb {
        return;
    }
    if size[ra] < size[rb] {
        parent[ra] = rb;
        size[rb] += size[ra];
    } else {
        parent[rb] = ra;
        size[ra] += size[rb];
    }
}

/// Options of [`largest_moving_blob`], mirroring the Python keyword arguments
/// `dark`, `open_px`, `area` and `roi` (defaults: `dark = true`,
/// `open_px = 5`, `area = None`, `roi = None`).
#[derive(Clone, Debug)]
pub struct BlobOptions {
    /// Look for pixels *darker* than the background instead of any change.
    pub dark: bool,
    /// Size of the square opening that removes thin structures such as a tail.
    pub open_px: u32,
    /// Plausible blob area range in px^2; rejects sizes outside it.
    pub area: Option<(f64, f64)>,
    /// Restricts the search to the arena floor.
    pub roi: Option<Image>,
}

impl Default for BlobOptions {
    fn default() -> Self {
        Self {
            dark: true,
            open_px: 5,
            area: None,
            roi: None,
        }
    }
}

/// `(xyxy box, pixel area)` of the largest moving blob, or `(None, 0)` when
/// absent, `(None, area)` when the blob has an implausible size, and
/// `(None, area)` when a second blob of over 30% of the largest one makes the
/// frame ambiguous.
///
/// Port of `animalpose.pipeline.largest_moving_blob`.
///
/// // differs from Python: `np.argsort(-areas)` uses an unstable sort, so ties
/// // between equal areas resolve arbitrarily; ties here keep the lower OpenCV
/// // label. The box is returned as `f64` instead of `float32`.
pub fn largest_moving_blob<I: ToGray + ?Sized>(
    img: &I,
    bg: &GrayF,
    diff: f32,
    opts: &BlobOptions,
) -> (Option<Box4>, f64) {
    let g = img.to_gray();
    let d = if opts.dark {
        signed_diff(bg, &g)
    } else {
        g.abs_diff(bg)
    };
    let m = d.threshold_gt(diff);
    let m = match &opts.roi {
        Some(roi) => mask_and(&m, roi),
        None => m,
    };
    let m = morph_open(&m, opts.open_px);
    let comps = connected_components(&m);
    if comps.is_empty() {
        // Python: `if k < 2: return None, 0.0` (label 0 is the background).
        return (None, 0.0);
    }
    let mut order: Vec<usize> = (0..comps.len()).collect();
    order.sort_by(|a, b| comps[*b].area.cmp(&comps[*a].area).then(a.cmp(b)));
    let largest = comps[order[0]];
    let a0 = f64::from(largest.area);
    if let Some((lo, hi)) = opts.area {
        if !(lo <= a0 && a0 <= hi) {
            return (None, a0);
        }
    }
    if order.len() > 1 && f64::from(comps[order[1]].area) > 0.3 * a0 {
        return (None, a0);
    }
    let pad = 0.15 * f64::from(largest.w.max(largest.h));
    let width = f64::from(g.width);
    let height = f64::from(g.height);
    let (x, y) = (f64::from(largest.x), f64::from(largest.y));
    let (w, h) = (f64::from(largest.w), f64::from(largest.h));
    let box4 = [
        (x - pad).max(0.0),
        (y - pad).max(0.0),
        (x + w + pad).min(width),
        (y + h + pad).min(height),
    ];
    (Some(box4), a0)
}

/// The largest moving blob as an xyxy box, or `None` when absent or ambiguous.
///
/// Port of `animalpose.pipeline.blob_box`.
pub fn blob_box<I: ToGray + ?Sized>(
    img: &I,
    bg: &GrayF,
    diff: f32,
    opts: &BlobOptions,
) -> Option<Box4> {
    largest_moving_blob(img, bg, diff, opts).0
}

/// Moving-blob detection candidates for a fixed camera, one [`crate::algos::blob::BlobProposer::propose`] call
/// per frame.
///
/// Port of `animalpose.pipeline.BlobProposer`: built from the recording's
/// median background; when `area` is not given it is learned from the first
/// `calibrate` accepted blobs (0.15x to 2.5x their median area).
#[derive(Clone, Debug)]
pub struct BlobProposer {
    /// Median background the blobs are measured against.
    pub bg: GrayF,
    /// Grey levels that count as "different from background".
    pub diff: f32,
    /// Score given to proposed boxes.
    pub score: f32,
    /// Size of the morphological opening.
    pub open_px: u32,
    /// Prefer dark blobs (a dark animal on a bright floor).
    pub dark: bool,
    /// Arena floor mask (see [`crate::algos::background::arena_roi`]).
    pub roi: Option<Image>,
    /// Plausible blob area range; learned when `None`.
    pub area: Option<(f64, f64)>,
    /// Number of accepted blobs used to learn `area`; `0` keeps `area` as-is.
    pub calibrate: usize,
    /// Blob areas seen so far while calibrating.
    pub sizes: Vec<f64>,
}

impl BlobProposer {
    /// A proposer for `bg` with the Python defaults: `diff = 25.0`,
    /// `score = 0.9`, `open_px = 5`, `area = None`, `calibrate = 64`,
    /// `roi = None`, `dark = true`.
    pub fn new(bg: GrayF) -> Self {
        Self {
            bg,
            diff: 25.0,
            score: 0.9,
            open_px: 5,
            dark: true,
            roi: None,
            area: None,
            calibrate: 64,
            sizes: Vec::new(),
        }
    }

    /// Sets `diff` (Python's `diff=`).
    pub fn with_diff(mut self, diff: f32) -> Self {
        self.diff = diff;
        self
    }

    /// Sets the score proposed boxes get (Python's `score=`).
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = score;
        self
    }

    /// Sets the opening size (Python's `open_px=`).
    pub fn with_open_px(mut self, open_px: u32) -> Self {
        self.open_px = open_px;
        self
    }

    /// Sets a fixed plausible area (Python's `area=`).
    pub fn with_area(mut self, area: (f64, f64)) -> Self {
        self.area = Some(area);
        self
    }

    /// Sets how many blobs calibrate the area; `0` disables calibration
    /// (Python's `calibrate=`).
    pub fn with_calibrate(mut self, calibrate: usize) -> Self {
        self.calibrate = calibrate;
        self
    }

    /// Restricts proposals to a mask (Python's `roi=`).
    pub fn with_roi(mut self, roi: Image) -> Self {
        self.roi = Some(roi);
        self
    }

    /// Proposes at most one box, as `(boxes, scores)`; empty when the blob is
    /// absent, ambiguous or implausible.
    ///
    /// Port of `animalpose.pipeline.BlobProposer.__call__`.
    pub fn propose<I: ToGray + ?Sized>(&mut self, img: &I) -> (Vec<Box4>, Vec<f32>) {
        let opts = BlobOptions {
            dark: self.dark,
            open_px: self.open_px,
            area: self.area,
            roi: self.roi.clone(),
        };
        let (box4, a0) = largest_moving_blob(img, &self.bg, self.diff, &opts);
        let Some(box4) = box4 else {
            return (Vec::new(), Vec::new());
        };
        if self.area.is_none() && self.calibrate > 0 {
            self.sizes.push(a0);
            if self.sizes.len() >= self.calibrate {
                let med = median(&self.sizes);
                self.area = Some((0.15 * med, 2.5 * med));
            }
        }
        (vec![box4], vec![self.score])
    }
}

/// Median of a slice (port of `np.median` on a 1-D array).
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::background::median_background;
    use crate::algos::image::Bgr;

    fn mask(width: u32, height: u32, rect: (u32, u32, u32, u32)) -> Image {
        let mut m = Image::filled(width, height, 0);
        m.fill_rect(rect.0, rect.1, rect.2, rect.3, 1);
        m
    }

    #[test]
    fn opening_removes_a_one_pixel_tail_but_keeps_the_body() {
        let mut m = Image::filled(20, 20, 0);
        m.fill_rect(5, 5, 12, 12, 1); // 7x7 body
        m.fill_rect(12, 8, 18, 9, 1); // 1 px tall tail
        let opened = morph_open(&m, 5);
        let comps = connected_components(&opened);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].x, 5);
        assert_eq!(comps[0].y, 5);
        assert_eq!(comps[0].w, 7);
        assert_eq!(comps[0].h, 7);
        assert_eq!(comps[0].area, 49);
        assert!(comps[0].area < connected_components(&m)[0].area);
    }

    #[test]
    fn mask_and_intersects_the_two_masks() {
        let a = mask(10, 10, (0, 0, 6, 10));
        let b = mask(10, 10, (4, 0, 10, 10));
        let both = mask_and(&a, &b);
        assert_eq!(connected_components(&both).len(), 1);
        assert_eq!(connected_components(&both)[0].area, 2 * 10);
        assert_eq!(
            connected_components(&mask_and(&a, &mask(10, 10, (7, 0, 9, 2)))).len(),
            0
        );
    }

    #[test]
    fn closing_fills_a_one_pixel_gap() {
        let mut m = Image::filled(16, 16, 0);
        m.fill_rect(2, 2, 14, 8, 1);
        m.fill_rect(2, 9, 14, 15, 1);
        assert_eq!(connected_components(&m).len(), 2);
        assert_eq!(connected_components(&morph_close(&m, 3)).len(), 1);
    }

    #[test]
    fn connected_components_is_eight_connected_and_ordered() {
        // Two blobs touching only through a diagonal stay one component.
        let mut m = Image::filled(6, 6, 0);
        m.set_pixel(0, 0, 1);
        m.set_pixel(1, 1, 1);
        m.set_pixel(4, 4, 1);
        let comps = connected_components(&m);
        assert_eq!(comps.len(), 2);
        assert_eq!(
            comps[0],
            Component {
                x: 0,
                y: 0,
                w: 2,
                h: 2,
                area: 2
            }
        );
        assert_eq!(
            comps[1],
            Component {
                x: 4,
                y: 4,
                w: 1,
                h: 1,
                area: 1
            }
        );
    }

    #[test]
    fn largest_moving_blob_finds_the_mouse_and_never_the_cup() {
        // Port of tests/test_degrade_frames.py::test_blob_candidates_...
        let frames: Vec<Bgr> = (0..20)
            .map(|t| {
                let mut f = Bgr::filled(80, 60, [200, 200, 200]);
                f.fill_rect(10, 10, 20, 20, [20, 20, 20]);
                let x = (4 * t) % 70;
                f.fill_rect(x, 40, x + 10, 48, [20, 20, 20]);
                f
            })
            .collect();
        let bg = median_background(&frames, 16);
        let opts = BlobOptions::default();
        let (box4, area) = largest_moving_blob(&frames[5], &bg, 25.0, &opts);
        // cv2 reports the opened blob at x=20 y=40 w=10 h=8 (area 80);
        // pad = 0.15 * 10 = 1.5.
        assert_eq!(area, 80.0);
        assert_eq!(box4, Some([18.5, 38.5, 31.5, 49.5]));
        let cx = (box4.unwrap()[0] + box4.unwrap()[2]) / 2.0;
        let cy = (box4.unwrap()[1] + box4.unwrap()[3]) / 2.0;
        assert!((20.0..=30.0).contains(&cx) && (40.0..=48.0).contains(&cy));
    }

    #[test]
    fn blob_rejects_ambiguous_frames_and_implausible_sizes() {
        let bg = GrayF::from_pixels(40, 20, vec![200.0; 800]).unwrap();
        let mut frame = Image::filled(40, 20, 200);
        frame.fill_rect(2, 2, 12, 12, 20); // 10x10 blob A
        frame.fill_rect(25, 2, 35, 12, 20); // 10x10 blob B -> ambiguous
        let (box4, area) = largest_moving_blob(&frame, &bg, 25.0, &BlobOptions::default());
        assert_eq!(box4, None);
        assert_eq!(area, 100.0);
        // A single blob whose area is outside the plausible range is rejected too.
        frame.fill_rect(25, 2, 35, 12, 200);
        let (box4, area) = largest_moving_blob(
            &frame,
            &bg,
            25.0,
            &BlobOptions {
                area: Some((500.0, 2000.0)),
                ..BlobOptions::default()
            },
        );
        assert_eq!(box4, None);
        assert_eq!(area, 100.0);
        let (box4, _) = largest_moving_blob(
            &frame,
            &bg,
            25.0,
            &BlobOptions {
                area: Some((50.0, 200.0)),
                ..BlobOptions::default()
            },
        );
        assert!(box4.is_some());
    }

    #[test]
    fn blob_box_and_empty_foreground_behave_like_python() {
        let bg = GrayF::from_pixels(20, 20, vec![100.0; 400]).unwrap();
        let frame = Image::filled(20, 20, 100);
        assert_eq!(blob_box(&frame, &bg, 25.0, &BlobOptions::default()), None);
        // `dark = false` reacts to a bright blob as well.
        let mut bright = Image::filled(20, 20, 100);
        bright.fill_rect(4, 4, 14, 14, 250);
        assert!(blob_box(
            &bright,
            &bg,
            25.0,
            &BlobOptions {
                dark: false,
                ..BlobOptions::default()
            }
        )
        .is_some());
        assert!(blob_box(&bright, &bg, 25.0, &BlobOptions::default()).is_none());
    }

    #[test]
    fn proposer_outranks_the_detector_and_learns_a_plausible_area() {
        use crate::algos::nms::merge_candidates;

        let frames: Vec<Bgr> = (0..20)
            .map(|t| {
                let mut f = Bgr::filled(80, 60, [200, 200, 200]);
                f.fill_rect(10, 10, 20, 20, [20, 20, 20]);
                let x = (4 * t) % 70;
                f.fill_rect(x, 40, x + 10, 48, [20, 20, 20]);
                f
            })
            .collect();
        let bg = median_background(&frames, 16);
        let mut proposer = BlobProposer::new(bg.clone())
            .with_diff(25.0)
            .with_calibrate(4);
        let boxes: Vec<Vec<Box4>> = frames
            .iter()
            .map(|f| proposer.propose(f))
            .map(|(b, _)| b)
            .collect();
        assert!(boxes.iter().all(|b| b.len() == 1));
        let blob = &boxes[5][0];
        let net = vec![*blob, [10.0, 10.0, 20.0, 20.0]];
        let (merged, scores) = merge_candidates(&net, &[0.6, 0.4], &boxes[5], &[0.9], 0.5, 20);
        assert_eq!(merged.len(), 2);
        assert!((scores[0] - 0.9).abs() < 1e-6);
        assert_eq!(merged[1], [10.0, 10.0, 20.0, 20.0]);
        // 0.15 / 2.5 x the median area of the calibration blobs; Python stops
        // appending once the range is learned, so `sizes` freezes at
        // `calibrate` entries.
        assert_eq!(proposer.area, Some((12.0, 200.0)));
        assert_eq!(proposer.sizes.len(), 4);
    }

    #[test]
    fn proposer_stops_proposing_once_the_blob_is_ambiguous() {
        let bg = GrayF::from_pixels(40, 20, vec![200.0; 800]).unwrap();
        let mut frame = Image::filled(40, 20, 200);
        frame.fill_rect(2, 2, 12, 12, 20);
        let mut proposer = BlobProposer::new(bg).with_calibrate(0);
        assert_eq!(proposer.propose(&frame).0.len(), 1);
        frame.fill_rect(25, 2, 35, 12, 20);
        let (boxes, scores) = proposer.propose(&frame);
        assert!(boxes.is_empty() && scores.is_empty());
        assert!(proposer.area.is_none());
    }

    #[test]
    fn median_matches_numpy_on_odd_and_even_lengths() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(median(&[]).is_nan());
    }
}
