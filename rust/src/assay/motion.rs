//! Motion inside the animal's outline, for grooming: the share of outline pixels whose grey level
//! changed by more than `motion.frame_diff` over `motion.lag_s`. The outline is where the frame
//! differs from the median background by more than `motion.bg_diff`, inside the pose box.
//!
//! The video is read in place and streamed twice (once to sample the background, once to
//! measure); memory holds the sampled background frames and `lag` frames, never the whole video.

use std::collections::VecDeque;
use std::path::Path;

use anyhow::Result;

use super::arena::median_u8;
use super::config::Motion;
use super::input::Track2d;
use crate::algos::image::{Image, ToGray};
use crate::data::video::{frames_evenly, VideoReader};

fn grey(frame: &crate::algos::image::Bgr) -> Image {
    super::arena::to_u8(&frame.to_gray())
}

/// `n` grey frames sampled evenly across the video, in one decode pass that hands over only the
/// chosen frames.
pub fn sample(video: &Path, n: usize) -> Result<Vec<Image>> {
    Ok(frames_evenly(video, n)?
        .iter()
        .map(|(_, frame)| grey(frame))
        .collect())
}

/// The median of `n` frames sampled evenly across the video.
pub fn background(video: &Path, n: usize) -> Result<Image> {
    median_u8(&sample(video, n)?)
        .ok_or_else(|| anyhow::anyhow!("{} has no frames", video.display()))
}

/// Per-frame motion of `track` (top view).
pub fn measure(
    video: &Path,
    track: &Track2d,
    frames: usize,
    fps: f64,
    cfg: &Motion,
) -> Result<Vec<Option<f64>>> {
    let bg = background(video, cfg.bg_frames)?;
    let lag = ((cfg.lag_s * fps).round() as usize).max(1);
    let (w, h) = (bg.width as i64, bg.height as i64);
    let mut out = vec![None; frames];
    let mut past: VecDeque<Image> = VecDeque::with_capacity(lag + 1);
    for (t, frame) in VideoReader::open(video, Some(frames))?.enumerate() {
        let now = grey(&frame?);
        if past.len() == lag {
            if let Some(b) = track.bbox(t) {
                let before = &past[0];
                let (x0, y0) = ((b[0].floor() as i64).max(0), (b[1].floor() as i64).max(0));
                let (x1, y1) = ((b[2].ceil() as i64).min(w), (b[3].ceil() as i64).min(h));
                let (mut outline, mut moving) = (0usize, 0usize);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let i = (y * w + x) as usize;
                        let v = f64::from(now.pixels[i]);
                        if (v - f64::from(bg.pixels[i])).abs() > cfg.bg_diff {
                            outline += 1;
                            if (v - f64::from(before.pixels[i])).abs() > cfg.frame_diff {
                                moving += 1;
                            }
                        }
                    }
                }
                if outline > 0 {
                    out[t] = Some(moving as f64 / outline as f64);
                }
            }
            past.pop_front();
        }
        past.push_back(now);
    }
    Ok(out)
}
