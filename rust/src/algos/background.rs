//! Static-background helpers: the median background of a recording, the
//! foreground share of a box, and the bright-floor arena mask.
//!
//! Port of `median_background`, `foreground_fraction` and `arena_roi` from
//! `animalpose.pipeline`.

use crate::algos::blob::dilate_rect;
use crate::algos::geometry::Box4;
use crate::algos::image::{GrayF, Image, ToGray};

/// Per-pixel median of up to `n` frames spread over the recording (grey levels).
///
/// Port of `animalpose.pipeline.median_background`.
///
/// // differs from Python: an empty frame list yields an empty image instead of
/// // the `ValueError` `np.stack([])` raises, and `n = 0` samples every frame
/// // instead of raising `ZeroDivisionError`. `numpy` propagates `NaN` through
/// // the median; here `NaN` sorts last.
pub fn median_background<I: ToGray>(frames: &[I], n: usize) -> GrayF {
    if frames.is_empty() {
        return GrayF::default();
    }
    let step = frames.len().checked_div(n).unwrap_or(frames.len()).max(1);
    let picked: Vec<GrayF> = frames.iter().step_by(step).map(|f| f.to_gray()).collect();
    let (width, height) = (picked[0].width, picked[0].height);
    let mut out = vec![0.0f32; picked[0].pixels.len()];
    let mut window = vec![0.0f32; picked.len()];
    for (i, o) in out.iter_mut().enumerate() {
        for (j, frame) in picked.iter().enumerate() {
            window[j] = frame.pixels[i];
        }
        window.sort_by(f32::total_cmp);
        let m = window.len();
        *o = if m % 2 == 1 {
            window[m / 2]
        } else {
            (window[m / 2 - 1] + window[m / 2]) / 2.0
        };
    }
    GrayF {
        width,
        height,
        pixels: out,
    }
}

/// Share of the pixels inside `bbox` (xyxy) that differ from the background by
/// more than `diff`.
///
/// Port of `animalpose.pipeline.foreground_fraction`.
///
/// // differs from Python: box coordinates are rounded half away from zero,
/// // where Python's `round` rounds half to even.
pub fn foreground_fraction<I: ToGray>(img: &I, bg: &GrayF, bbox: Box4, diff: f32) -> f64 {
    let (w, h) = (bg.width as i64, bg.height as i64);
    let x0 = (bbox[0].round() as i64).max(0);
    let y0 = (bbox[1].round() as i64).max(0);
    let x1 = (bbox[2].round() as i64).min(w);
    let y1 = (bbox[3].round() as i64).min(h);
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let gray = img.to_gray();
    let mut hits = 0usize;
    let mut total = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * bg.width as i64 + x) as usize;
            if (gray.pixels[i] - bg.pixels[i]).abs() > diff {
                hits += 1;
            }
            total += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

/// Otsu threshold of an 8-bit image, the value `cv2.threshold` picks with
/// `cv2.THRESH_OTSU` (the smallest level maximising between-class variance).
pub fn otsu_threshold(img: &Image) -> u8 {
    let mut hist = [0u64; 256];
    for v in &img.pixels {
        hist[usize::from(*v)] += 1;
    }
    let total = img.pixels.len() as f64;
    if total == 0.0 {
        return 0;
    }
    let mut sum = 0.0;
    for (i, count) in hist.iter().enumerate() {
        sum += i as f64 * (*count as f64) / total;
    }
    let mut sum_b = 0.0;
    let mut w_b = 0.0;
    let mut best = -1.0;
    let mut threshold = 0u8;
    for (i, count) in hist.iter().enumerate().take(255) {
        w_b += (*count as f64) / total;
        if w_b == 0.0 {
            continue;
        }
        let w_f = 1.0 - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += i as f64 * (*count as f64) / total;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;
        let between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if between > best {
            best = between;
            threshold = i as u8;
        }
    }
    threshold
}

/// Bright floor region of the background (where a dark animal can be seen),
/// slightly grown.
///
/// Port of `animalpose.pipeline.arena_roi`: `cv2.threshold(bg.astype(uint8), 0,
/// 1, THRESH_BINARY + THRESH_OTSU)` followed by `cv2.dilate` with a
/// `grow x grow` rectangle. The returned mask holds `0`/`1`.
pub fn arena_roi(bg: &GrayF, grow: u32) -> Image {
    let u8_bg = bg.as_u8();
    let threshold = otsu_threshold(&u8_bg);
    let mask = Image {
        width: bg.width,
        height: bg.height,
        pixels: u8_bg
            .pixels
            .iter()
            .map(|v| u8::from(*v > threshold))
            .collect(),
    };
    dilate_rect(&mask, grow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::image::Bgr;

    /// The frames of `tests/test_degrade_frames.py::test_background_filter_...`
    /// built directly: a static cup and a mouse sweeping across the floor.
    fn frames() -> Vec<Bgr> {
        (0..20)
            .map(|t| {
                let mut f = Bgr::filled(80, 60, [200, 200, 200]);
                f.fill_rect(10, 10, 20, 20, [20, 20, 20]); // static cup
                let x = (4 * t) % 70;
                f.fill_rect(x, 40, x + 10, 48, [20, 20, 20]); // moving mouse
                f
            })
            .collect()
    }

    #[test]
    fn median_background_ignores_the_moving_mouse_but_keeps_the_cup() {
        let frames = frames();
        let bg = median_background(&frames, 16);
        assert_eq!((bg.width, bg.height), (80, 60));
        assert_eq!(bg.pixel(15, 15), 20.0); // the cup stays in the background
        assert_eq!(bg.pixel(25, 44), 200.0); // the mouse path averages out
        assert_eq!(bg.pixel(75, 55), 200.0);
    }

    #[test]
    fn sampling_step_matches_the_python_frame_count() {
        let frames = frames();
        // median_background(frames, 4) with 20 frames steps by 5.
        let coarse = median_background(&frames, 4);
        assert_eq!((coarse.width, coarse.height), (80, 60));
        assert_eq!(coarse.pixel(15, 15), 20.0);
        let empty: Vec<Bgr> = Vec::new();
        assert_eq!(median_background(&empty, 16).pixels.len(), 0);
        assert_eq!(median_background(&frames, 0).pixel(15, 15), 20.0);
    }

    #[test]
    fn foreground_fraction_separates_static_objects_from_the_animal() {
        // Port of the assertions in tests/test_degrade_frames.py.
        let frames = frames();
        let bg = median_background(&frames, 16);
        let cup = foreground_fraction(&frames[5], &bg, [10.0, 10.0, 20.0, 20.0], 25.0);
        let mouse = foreground_fraction(&frames[5], &bg, [20.0, 40.0, 30.0, 48.0], 25.0);
        assert!(cup < 0.05, "{cup}");
        assert!(mouse > 0.5, "{mouse}");
        assert_eq!(cup, 0.0);
        assert_eq!(mouse, 1.0);
    }

    #[test]
    fn foreground_fraction_clips_boxes_outside_the_frame() {
        let frames = frames();
        let bg = median_background(&frames, 16);
        assert_eq!(
            foreground_fraction(&frames[5], &bg, [70.0, 55.0, 200.0, 200.0], 25.0),
            0.0
        );
        assert_eq!(
            foreground_fraction(&frames[5], &bg, [30.0, 40.0, 20.0, 48.0], 25.0),
            0.0
        );
    }

    #[test]
    fn arena_roi_masks_the_bright_floor_and_grows_it() {
        let frames = frames();
        let bg = median_background(&frames, 16);
        let roi = arena_roi(&bg, 15);
        assert_eq!((roi.width, roi.height), (80, 60));
        assert!(roi.pixels.iter().all(|v| *v <= 1));
        // Python: the threshold lands between the cup (20) and the floor (200);
        // growing by 15 swallows the 10x10 cup, so the mask covers everything.
        assert_eq!(roi.pixel(75, 15), 1);
        assert_eq!(roi.pixels.iter().map(|v| *v as u32).sum::<u32>(), 80 * 60);
    }

    #[test]
    fn otsu_splits_a_two_level_histogram() {
        let mut img = Image::filled(10, 1, 0);
        for i in 0..5 {
            img.set_pixel(i, 0, 200);
        }
        // OpenCV keeps the first level that maximises the between-class variance.
        assert_eq!(otsu_threshold(&img), 0);
        let mut shifted = Image::filled(10, 1, 50);
        for i in 0..5 {
            shifted.set_pixel(i, 0, 200);
        }
        assert_eq!(otsu_threshold(&shifted), 50);
        assert_eq!(otsu_threshold(&Image::filled(4, 4, 7)), 0);
    }
}
