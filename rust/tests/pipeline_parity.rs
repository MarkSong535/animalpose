//! End-to-end parity of `animalpose predict` with the Python pipeline.
//!
//! The reference is `animalpose fixtures pipeline`, which runs
//! `src/animalpose/pipeline.py:PosePipeline` over eight deterministic synthetic frames and writes
//! what it saw to `tests/fixtures/pipeline/`: the frames as PNGs, the `pipeline.yaml` both ports
//! read, the per-frame network/blob/merged boxes and per-box pose output, the classifier's own input
//! tensors, every `TrackResult` array and the `tracks.csv` / `behavior.csv` files.
//!
//! What is compared, and how tightly:
//!
//! | value | tolerance | note |
//! | --- | --- | --- |
//! | frame rate, frame size | exact | both read the `..._HH_MM_SS_mmm` file names |
//! | letterboxed detector input, median background | SHA-256 equal | bit-identical float32 arrays |
//! | classifier input | mask exact, coordinates 1e-3 | the mask is 0/1; the coordinates carry the temporal pass's own rounding |
//! | warped pose crop | SHA-256 where it matches; otherwise reported | the border taps agree, OpenCV's weights are 11-bit fixed point so the last grey level can differ |
//! | boxes and detector scores | 1e-5 | the same decoded peaks |
//! | keypoints, temporal output | 1e-3 px | 9.8e-4 px was measured on the repository fixture; `ANIMALPOSE_KEYPOINT_TOLERANCE` relaxes it for a trained-checkpoint dump, where a crop that differs from `cv2.warpAffine` by one grey level can move a near-tie onto the other heat-map peak |
//! | confidences, classifier probabilities | 1e-3 | 4e-5 was measured |
//! | track ids, frames, behavior argmax, `tracks.csv` keys | exact | the tracker and the catalogue agree frame for frame |
//!
//! Detections are matched by index (both ports see them in the same order) *and* by IoU > 0.5; the
//! unmatched count is reported, as is the number of keypoints above 1e-3 px, which is where a
//! tie-break in the decoding could show up. The test skips (and says so) when the fixture directory
//! is absent, which is how it behaves in a checkout without generated fixtures.

use std::path::{Path, PathBuf};

use animalpose::algos::background::{arena_roi, median_background};
use animalpose::algos::blob::BlobProposer;
use animalpose::algos::geometry::{box_iou, Box4};
use animalpose::algos::image::Image;
use animalpose::algos::nms::merge_candidates;
use animalpose::cli::predict::Pipeline;
use animalpose::data::config::Config;
use animalpose::data::images::{read_frames, to_bgr8, u8_from_gray};
use anyhow::{bail, Context, Result};
use serde_json::Value;

/// The coordinate tolerance in pixels: 1e-3, which is what the repository fixture measures
/// (9.8e-4 px with the tiny random weights it ships with).
///
/// `ANIMALPOSE_KEYPOINT_TOLERANCE` relaxes it for a dump made with the trained checkpoints
/// (`exported/pose_general.safetensors`): the crop fed to the pose network differs from
/// `cv2.warpAffine`'s by at most one grey level, and on a near-tie between two heat-map peaks that
/// grey level is enough to move a keypoint onto the other peak (4.9 px on one point of a
/// single-frame track, against 3e-2 px over the smoothed 40-frame one, where the temporal pass hides
/// it). The largest difference is reported either way.
fn keypoint_tolerance() -> f64 {
    std::env::var("ANIMALPOSE_KEYPOINT_TOLERANCE")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(1e-3)
}
/// Detector boxes and scores are decoded from identical heat maps.
const BOX_TOLERANCE: f64 = 1e-5;
/// Confidences and classifier probabilities.
const SOFT_TOLERANCE: f64 = 1e-3;
/// The classifier is handed the track's smoothed coordinates, so its input inherits the keypoint
/// tolerance above; its observation mask is 0/1 and must match exactly.
///
/// `tracks.csv` prints coordinates with two decimals, so cells may differ by one ulp of that.
const CSV_TOLERANCE: f64 = 0.011;

/// The pipeline fixture directory, or `None` when it was never generated.
///
/// `ANIMALPOSE_PIPELINE_FIXTURE` points the test at another dump of the same script (the default
/// `tests/fixtures/pipeline` holds tiny random weights; a dump made with `--detector
/// exported/detector_general.safetensors …` holds the trained checkpoints and is too big to keep in the
/// repository), so trained-model parity can be checked without touching the repository default.
fn fixture_dir() -> Option<PathBuf> {
    let dir = match std::env::var_os("ANIMALPOSE_PIPELINE_FIXTURE") {
        Some(dir) => PathBuf::from(dir),
        None => animalpose::data::fixtures_dir()?.join("pipeline"),
    };
    dir.join("fixture.json").is_file().then_some(dir)
}

fn load_json(path: &Path) -> Result<Value> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("{} is not JSON", path.display()))
}

/// The `data` of a `{"shape", "data"}` fixture entry as `f64`, with `null` (an unobserved point)
/// read back as NaN, exactly as `animalpose fixtures models:dec` does.
fn values(entry: &Value) -> Vec<f64> {
    entry["data"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect()
        })
        .unwrap_or_default()
}

/// The shape of a fixture array.
fn shape(entry: &Value) -> Vec<usize> {
    entry["shape"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_u64())
                .map(usize::try_from)
                .filter_map(Result::ok)
                .collect()
        })
        .unwrap_or_default()
}

/// The largest absolute difference between a fixture array and a Rust array: NaN matches NaN, a NaN
/// where the other side has a number counts as a mismatch and reports infinity.
fn largest(want: &[f64], got: &[f64]) -> (f64, usize) {
    if want.len() != got.len() {
        return (f64::INFINITY, want.len().max(got.len()));
    }
    let mut worst = 0.0f64;
    let mut mismatched = 0;
    for (a, b) in want.iter().zip(got) {
        match (a.is_finite(), b.is_finite()) {
            (true, true) => worst = worst.max((a - b).abs()),
            (false, false) => {}
            _ => {
                mismatched += 1;
                worst = f64::INFINITY;
            }
        }
    }
    (worst, mismatched)
}

fn f64_slice(values: &[f64]) -> Vec<f64> {
    values.to_vec()
}

fn f32_as_f64(values: &[f32]) -> Vec<f64> {
    values.iter().map(|v| f64::from(*v)).collect()
}

/// Boxes of a fixture array, `(n, 4)` flattened, grouped back into rows.
fn boxes_of(entry: &Value) -> Vec<Box4> {
    values(entry)
        .chunks(4)
        .map(|row| [row[0], row[1], row[2], row[3]])
        .collect()
}

/// Detections matched greedily by IoU; returns how many of `want` no `got` box reached 0.5 of.
fn unmatched(want: &[Box4], got: &[Box4]) -> usize {
    let mut used = vec![false; got.len()];
    let mut left = 0;
    for a in want {
        let best = got
            .iter()
            .enumerate()
            .filter(|(j, _)| !used[*j])
            .max_by(|(_, x), (_, y)| {
                box_iou(*a, **x)
                    .partial_cmp(&box_iou(*a, **y))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        match best {
            Some((j, b)) if box_iou(*a, *b) > 0.5 => used[j] = true,
            _ => left += 1,
        }
    }
    left
}

/// Rows `j` of a flattened `(rows, width)` fixture array.
fn row_of(values: &[f64], j: usize, width: usize) -> Result<Vec<f64>> {
    let from = j * width;
    values
        .get(from..from + width)
        .map(<[f64]>::to_vec)
        .with_context(|| {
            format!(
                "the fixture array holds {} values, row {j} of width {width} needs {}",
                values.len(),
                from + width
            )
        })
}

/// A fixture array of integers (frames, labels, swaps), read back as Rust integers.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn integers(entry: &Value) -> Result<Vec<i64>> {
    values(entry)
        .into_iter()
        .map(|v| {
            let rounded = v.round();
            (rounded.abs() <= i64::MAX as f64).then_some(rounded as i64)
        })
        .collect::<Option<Vec<i64>>>()
        .context("a fixture integer array holds a value outside i64")
}

/// SHA-256 of `bytes` as hex (the fixture hashes its input arrays this way; no hashing crate is
/// allowed in this workspace, so the compression function lives here).
fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = bytes.to_vec();
    let bit_len = (bytes.len() as u64) * 8;
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for block in message.chunks(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            #[allow(clippy::arithmetic_side_effects, clippy::pedantic)]
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ ((!v[4]) & v[6]);
            #[allow(clippy::arithmetic_side_effects)]
            let temp1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            #[allow(clippy::arithmetic_side_effects)]
            let temp2 = s0.wrapping_add(maj);
            #[allow(clippy::cast_possible_truncation, clippy::arithmetic_side_effects)]
            {
                v[7] = v[6];
                v[6] = v[5];
                v[5] = v[4];
                v[4] = v[3].wrapping_add(temp1);
                v[3] = v[2];
                v[2] = v[1];
                v[1] = v[0];
                v[0] = temp1.wrapping_add(temp2);
            }
        }
        for (slot, value) in h.iter_mut().zip(v) {
            *slot = slot.wrapping_add(value);
        }
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

/// SHA-256 of the little-endian float32 bytes of `values`, matching the fixture's `sha_f32`.
fn sha256_f32(values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    sha256_hex(&bytes)
}

/// The published digests of the two standard test messages: the fixture hashes are only meaningful
/// if this helper is the real SHA-256.
#[test]
fn the_fixture_hashes_are_sha256() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

/// Reads a CSV the Python `csv` module wrote (`\r\n` line endings).
fn read_csv(path: &Path) -> Result<Vec<Vec<String>>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    Ok(text
        .trim_end()
        .split("\r\n")
        .map(|row| row.split(',').map(str::to_string).collect())
        .collect())
}

/// One number of a CSV cell, `None` for a blank or non-numeric cell.
fn cell(row: &[String], column: usize) -> Option<f64> {
    row.get(column).and_then(|v| v.parse().ok())
}

#[test]
fn predict_matches_the_python_pipeline_fixture() {
    let report = match fixture_dir().map(|dir| parity(&dir)) {
        Some(Ok(report)) => report,
        Some(Err(err)) => panic!("{err:?}"),
        None => {
            eprintln!("SKIP: tests/fixtures/pipeline is absent (run animalpose fixtures pipeline)");
            return;
        }
    };
    print!("{report}");
}

/// Every comparison the fixture supports; the returned text is the parity report.
fn parity(dir: &Path) -> Result<String> {
    let fixture = load_json(&dir.join("fixture.json"))?;
    let cfg = Config::load(dir.join("pipeline.yaml"), &[])?;
    let pipeline = Pipeline::from_config(&cfg)?;
    let (frames, stamps) = read_frames(dir.join("frames"), None, 1)?;
    anyhow::ensure!(!frames.is_empty(), "the fixture frames are missing");
    let fps = stamps.unwrap_or(30.0);

    // The frame rate both ports read off the file names, and the frame size.
    let want_fps = fixture["fps"].as_f64().context("fixture fps")?;
    anyhow::ensure!(
        (fps - want_fps).abs() < 1e-9,
        "frame rate {fps} from the file names, Python read {want_fps}"
    );
    anyhow::ensure!(
        frames.len()
            == usize::try_from(fixture["frames"].as_u64().unwrap_or(0)).unwrap_or(usize::MAX),
        "the fixture has {} frames, {} were read",
        fixture["frames"].as_u64().unwrap_or(0),
        frames.len()
    );
    anyhow::ensure!(
        f64::from(frames[0].height) == fixture["size"][0].as_f64().unwrap_or(-1.0)
            && f64::from(frames[0].width) == fixture["size"][1].as_f64().unwrap_or(-1.0),
        "frame size {}x{}, Python read {}x{}",
        frames[0].width,
        frames[0].height,
        fixture["size"][1].as_u64().unwrap_or(0),
        fixture["size"][0].as_u64().unwrap_or(0)
    );

    // The median background both ports build for the blob candidate and the static-object filter.
    let samples: Vec<Image> = frames.iter().map(u8_from_gray).collect();
    let background = pipeline
        .cfg
        .needs_background()
        .then(|| median_background(&samples, pipeline.cfg.background_frames));
    if fixture["background"].is_object() {
        let bg = background
            .as_ref()
            .context("the config asks for a background")?;
        anyhow::ensure!(
            shape(&fixture["background"]) == vec![bg.height as usize, bg.width as usize],
            "background shape"
        );
        let want = fixture["background"]["sha256"].as_str().unwrap_or("");
        anyhow::ensure!(
            sha256_f32(&bg.pixels) == want,
            "the median background differs from numpy's (sha {} vs {want})",
            sha256_f32(&bg.pixels)
        );
    }

    // ---------------------------------------------------------------- stages, per frame
    let mut proposer = background.as_ref().map(|bg| {
        let mut proposer = BlobProposer::new(bg.clone())
            .with_diff(
                pipeline
                    .cfg
                    .blob_diff
                    .unwrap_or(pipeline.cfg.foreground_diff),
            )
            .with_score(pipeline.cfg.blob_score as f32)
            .with_open_px(pipeline.cfg.blob_open_px)
            .with_calibrate(pipeline.cfg.blob_calibrate);
        if let Some(area) = pipeline.cfg.blob_area {
            proposer = proposer.with_area(area);
        }
        if pipeline.cfg.blob_arena_roi {
            proposer = proposer.with_roi(arena_roi(bg, 15));
        }
        proposer
    });

    let mut max_box: f64 = 0.0;
    let mut max_score: f64 = 0.0;
    let mut max_keypoint: f64 = 0.0;
    let mut max_conf: f64 = 0.0;
    let mut max_margin: f64 = 0.0;
    let mut keypoints_over_1e3 = 0;
    let mut detections_unmatched = 0;
    let mut crop_hash_mismatches = 0;
    let mut inputs_hashed = 0;
    let stages = fixture["stages"].as_array().cloned().unwrap_or_default();
    anyhow::ensure!(stages.len() == frames.len(), "one fixture stage per frame");
    for (i, stage) in stages.iter().enumerate() {
        let frame = &frames[i];
        let image = to_bgr8(frame);
        // The letterboxed tensor the detector is fed, bit for bit.
        let (tensor, _, _) = pipeline.detector_input(&image)?;
        anyhow::ensure!(
            shape(&stage["detector_input"]) == vec![3, frame.height as usize, frame.width as usize],
            "the detector input shape of frame {i}"
        );
        anyhow::ensure!(
            sha256_f32(&tensor) == stage["detector_input"]["sha256"].as_str().unwrap_or(""),
            "the detector input of frame {i} is not the array Python fed the network"
        );
        inputs_hashed += 1;

        let (net_boxes, net_scores) = pipeline.detect(&image, frame.width, frame.height)?;
        let (blob_boxes, blob_scores) = match &mut proposer {
            Some(proposer) => {
                let (boxes, scores) = proposer.propose(&samples[i]);
                (boxes, f32_as_f64(&scores))
            }
            None => (Vec::new(), Vec::new()),
        };
        let (boxes, scores) = merge_candidates(
            &net_boxes,
            &net_scores,
            &blob_boxes,
            &blob_scores,
            pipeline.cfg.merge_iou,
            pipeline.cfg.max_det,
        );
        let mut consider = |want: Vec<f64>, got: Vec<f64>, boxes_like: bool| -> Result<()> {
            let (worst, mismatched) = largest(&want, &got);
            anyhow::ensure!(
                mismatched == 0,
                "frame {i}: {mismatched} detection values are NaN on one side only"
            );
            if boxes_like {
                max_box = max_box.max(worst);
            } else {
                max_score = max_score.max(worst);
            }
            Ok(())
        };
        consider(
            values(&stage["net_boxes"]),
            f64_slice(&net_boxes.concat()),
            true,
        )?;
        consider(values(&stage["net_scores"]), net_scores.clone(), false)?;
        consider(values(&stage["blob_boxes"]), blob_boxes.concat(), true)?;
        consider(values(&stage["blob_scores"]), blob_scores, false)?;
        consider(values(&stage["boxes"]), boxes.concat(), true)?;
        consider(values(&stage["scores"]), scores.clone(), false)?;
        detections_unmatched += unmatched(&boxes_of(&stage["boxes"]), &boxes);

        // The crop the pose network is fed, and the pose output per box.
        let keypoints = shape(&stage["keypoints"]);
        anyhow::ensure!(
            keypoints.first().copied().unwrap_or(0) == boxes.len(),
            "frame {i}: Python estimated {} boxes, {} were decoded",
            keypoints.first().copied().unwrap_or(0),
            boxes.len()
        );
        for (j, box4) in boxes.iter().enumerate() {
            let (crop, _) = pipeline.pose_input(frame, *box4)?;
            if sha256_f32(&crop) != stage["pose_inputs"][j]["sha256"].as_str().unwrap_or("") {
                crop_hash_mismatches += 1;
            }
            let pose = pipeline.estimate(frame, *box4)?;
            let keypoints = pose.conf.len();
            anyhow::ensure!(
                keypoints > 0,
                "frame {i} box {j}: the pose network decoded nothing"
            );
            let want_kp = row_of(&values(&stage["keypoints"]), j, keypoints * 2)?;
            let got_kp: Vec<f64> = pose.keypoints.iter().flat_map(|p| [p[0], p[1]]).collect();
            let (worst, mismatched) = largest(&want_kp, &got_kp);
            anyhow::ensure!(
                mismatched == 0,
                "frame {i} box {j}: a keypoint is NaN on one side only"
            );
            if worst > 1e-3 {
                keypoints_over_1e3 += 1;
            }
            max_keypoint = max_keypoint.max(worst);
            let want_conf = row_of(&values(&stage["conf"]), j, keypoints)?;
            let (worst, mismatched) = largest(&want_conf, &pose.conf);
            anyhow::ensure!(
                mismatched == 0,
                "frame {i} box {j}: a confidence is NaN on one side only"
            );
            max_conf = max_conf.max(worst);
            if stage["pair_margin"].is_object() && !pose.margin.is_empty() {
                let rows = usize::try_from(stage["pair_margin"]["shape"][0].as_u64().unwrap_or(0))
                    .unwrap_or(boxes.len());
                anyhow::ensure!(
                    rows == boxes.len(),
                    "frame {i}: the fixture scored {rows} boxes"
                );
                let pairs = values(&stage["pair_margin"]).len() / rows;
                let want_margin = row_of(&values(&stage["pair_margin"]), j, pairs)?;
                let (worst, mismatched) = largest(&want_margin, &pose.margin);
                anyhow::ensure!(
                    mismatched == 0,
                    "frame {i} box {j}: a pair margin is NaN on one side"
                );
                max_margin = max_margin.max(worst);
            }
        }
    }

    // ---------------------------------------------------------------- whole recording
    let result = pipeline.run(&frames, background.as_ref(), fps)?;
    let want_times = values(&fixture["times"]);
    let (worst, _) = largest(&want_times, &result.times);
    anyhow::ensure!(worst <= 1e-9, "the presentation times differ by {worst} s");

    let want_tracks = fixture["tracks"].as_object().cloned().unwrap_or_default();
    anyhow::ensure!(
        result.tracks.len() == want_tracks.len(),
        "the Rust pipeline made {} tracks, Python made {}",
        result.tracks.len(),
        want_tracks.len()
    );
    let mut max_track: f64 = 0.0;
    let mut max_track_soft: f64 = 0.0;
    let mut behavior_input_checks = 0;
    let mut classifier_inputs_bit_identical = 0;
    for (id, want) in &want_tracks {
        let id: i64 = id.parse().context("fixture track id")?;
        let Some(got) = result.tracks.get(&id) else {
            bail!("the Rust pipeline never made track {id}");
        };
        anyhow::ensure!(
            integers(&want["frames"])?
                == got
                    .frames
                    .iter()
                    .map(|f| i64::try_from(*f).unwrap_or(i64::MAX))
                    .collect::<Vec<i64>>(),
            "track {id} covers {:?}, Python covered {:?}",
            got.frames,
            values(&want["frames"])
        );
        for (name, array) in [
            ("boxes", f64_slice(&got.boxes)),
            ("keypoints", got.keypoints.clone()),
            ("conf", got.conf.clone()),
            ("smoothed", got.smoothed.clone().unwrap_or_default()),
            (
                "smoothed_conf",
                got.smoothed_conf.clone().unwrap_or_default(),
            ),
            ("imputed", got.imputed.clone().unwrap_or_default()),
            (
                "behavior_probs",
                got.behavior_probs.clone().unwrap_or_default(),
            ),
            (
                "behavior_full",
                got.behavior_full.clone().unwrap_or_default(),
            ),
            ("pair_margin", got.pair_margin.clone().unwrap_or_default()),
        ] {
            if !want[name].is_object() {
                anyhow::ensure!(
                    array.is_empty(),
                    "track {id}: Rust has {} {name} values, Python none",
                    array.len()
                );
                continue;
            }
            anyhow::ensure!(
                array.len() == values(&want[name]).len(),
                "track {id}: {name} holds {} values, the fixture has {}",
                array.len(),
                values(&want[name]).len()
            );
            let (worst, mismatched) = largest(&values(&want[name]), &array);
            anyhow::ensure!(
                mismatched == 0,
                "track {id}: {mismatched} {name} values are NaN on one side only"
            );
            if matches!(
                name,
                "conf"
                    | "smoothed_conf"
                    | "imputed"
                    | "behavior_probs"
                    | "behavior_full"
                    | "pair_margin"
            ) {
                max_track_soft = max_track_soft.max(worst);
            } else {
                max_track = max_track.max(worst);
            }
        }
        if let Some(behavior) = &got.behavior {
            anyhow::ensure!(
                *behavior == integers(&want["behavior"])?,
                "track {id}: the behavior argmax differs from Python's"
            );
        }
        if let (Some(swapped), Value::Object(_)) = (&got.pair_swapped, &want["pair_swapped"]) {
            let swapped: Vec<i64> = swapped.iter().map(|v| i64::from(*v)).collect();
            anyhow::ensure!(
                swapped == integers(&want["pair_swapped"])?,
                "track {id}: the pair identity pass swapped differently"
            );
        }
        // The classifier's own input tensors for this track: the observation mask exactly, the
        // resampled coordinates to within the temporal pass's own rounding.
        if let Some(inputs) = fixture["behavior_inputs"].as_array() {
            if let Some(entry) = inputs
                .iter()
                .find(|e| e["track"].as_str() == Some(&id.to_string()))
            {
                let times: Vec<f64> = got.frames.iter().map(|f| result.times[*f]).collect();
                let (coords, mask, _) = pipeline.behavior_input(got, &times)?;
                anyhow::ensure!(
                    coords.len() == values(&entry["coords_values"]).len(),
                    "track {id}: the classifier input holds {} values, Python fed {}",
                    coords.len(),
                    values(&entry["coords_values"]).len()
                );
                let (worst, mismatched) =
                    largest(&values(&entry["coords_values"]), &f32_as_f64(&coords));
                anyhow::ensure!(
                    mismatched == 0,
                    "track {id}: a classifier coordinate is NaN on one side"
                );
                anyhow::ensure!(
                    worst <= keypoint_tolerance(),
                    "track {id}: the classifier was handed coordinates that differ by {worst}"
                );
                let (mask_worst, _) = largest(&values(&entry["mask_values"]), &f32_as_f64(&mask));
                anyhow::ensure!(
                    mask_worst == 0.0,
                    "track {id}: the observation mask differs in {mask_worst} places"
                );
                behavior_input_checks += 1;
                if sha256_f32(&coords) == entry["coords"]["sha256"].as_str().unwrap_or("") {
                    classifier_inputs_bit_identical += 1;
                }
            }
        }
    }

    // ---------------------------------------------------------------- the written CSVs
    let out =
        std::env::temp_dir().join(format!("animalpose-pipeline-parity-{}", std::process::id()));
    std::fs::create_dir_all(&out)?;
    result.to_csv(&out.join("tracks.csv"))?;
    if result.behavior_info.is_some() {
        result.to_behavior_csv(&out.join("behavior.csv"), pipeline.cfg.behavior_top_k)?;
    }
    let (rows, bad_keys, csv_worst) =
        compare_csv(&dir.join("tracks.csv"), &out.join("tracks.csv"))?;
    let behavior_rows = if dir.join("behavior.csv").is_file() {
        let (rows, bad_keys, worst) =
            compare_csv(&dir.join("behavior.csv"), &out.join("behavior.csv"))?;
        anyhow::ensure!(
            bad_keys == 0,
            "{bad_keys} behavior.csv rows disagree on frame/track/name"
        );
        anyhow::ensure!(
            worst <= CSV_TOLERANCE,
            "behavior.csv cells differ by {worst}"
        );
        rows
    } else {
        0
    };
    let _ = std::fs::remove_dir_all(&out);

    // ---------------------------------------------------------------- assertions
    anyhow::ensure!(
        max_box <= BOX_TOLERANCE && max_score <= BOX_TOLERANCE,
        "detector boxes/scores differ by {max_box}/{max_score} px (> {BOX_TOLERANCE})"
    );
    let keypoint_tolerance = keypoint_tolerance();
    anyhow::ensure!(
        max_track <= keypoint_tolerance && max_keypoint <= keypoint_tolerance,
        "track coordinates differ by up to {} px (> {keypoint_tolerance})",
        max_track.max(max_keypoint)
    );
    anyhow::ensure!(
        max_conf.max(max_track_soft).max(max_margin) <= SOFT_TOLERANCE,
        "confidences or probabilities differ by more than {SOFT_TOLERANCE}"
    );

    anyhow::ensure!(
        bad_keys == 0,
        "{bad_keys} tracks.csv rows disagree on frame/track/keypoint"
    );
    anyhow::ensure!(
        csv_worst <= CSV_TOLERANCE,
        "tracks.csv cells differ by {csv_worst}"
    );
    anyhow::ensure!(
        detections_unmatched == 0,
        "{detections_unmatched} merged boxes no Python box matched at IoU > 0.5"
    );

    Ok(format!(
        "pipeline fixture {}: {} frames, {} tracks, {} classifier inputs checked\n\
         \x20 detector input tensors bit-identical (sha-256): {}, median background bit-identical\n\
         \x20 classifier inputs: {classifier_inputs_bit_identical} of {behavior_input_checks} bit-identical, the rest within {SOFT_TOLERANCE} of the resampled track\n\
         \x20 pose crops differing from cv2.warpAffine: {crop_hash_mismatches}; on those crops the pose output still agrees to {max_keypoint:.2e} px\n\
         \x20 largest difference: boxes {max_box:.2e}, scores {max_score:.2e}, per-box keypoints {max_keypoint:.2e} px (\
         {keypoints_over_1e3} above 1e-3, tolerance {keypoint_tolerance}), track coordinates {max_track:.2e} px\n\
         \x20 confidences {max_conf:.2e}, classifier probabilities {max_track_soft:.2e}, pair margins {max_margin:.2e}, times 1e-9\n\
         \x20 detections unmatched at IoU > 0.5: {detections_unmatched}; tracks.csv {rows} rows, behavior.csv {behavior_rows} rows, largest printed difference {csv_worst:.4}\n",
        dir.display(),
        frames.len(),
        result.tracks.len(),
        behavior_input_checks,
        inputs_hashed,
    ))
}

/// Compares two CSVs the same way `csv.reader` would: same header, same rows, same key columns, and
/// the largest difference over every numeric cell.
fn compare_csv(want: &Path, got: &Path) -> Result<(usize, usize, f64)> {
    let want = read_csv(want)?;
    let got = read_csv(got)?;
    anyhow::ensure!(
        want.first() == got.first(),
        "the CSV headers differ:\n  python {:?}\n  rust   {:?}",
        want.first(),
        got.first()
    );
    anyhow::ensure!(
        want.len() == got.len(),
        "{} CSV rows were written, Python wrote {}",
        got.len(),
        want.len()
    );
    let mut bad_keys = 0;
    let mut worst: f64 = 0.0;
    for (a, b) in want.iter().skip(1).zip(got.iter().skip(1)) {
        if a[..4] != b[..4] {
            bad_keys += 1;
            continue;
        }
        for column in 4..a.len() {
            if let (Some(x), Some(y)) = (cell(a, column), cell(b, column)) {
                worst = worst.max((x - y).abs());
            }
        }
    }
    Ok((want.len() - 1, bad_keys, worst))
}
