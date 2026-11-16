//! End-to-end `animalpose predict` with the four general checkpoints of `weights/` (read straight
//! from their `.pt` files) on a short `.mp4`, or on a PNG folder when ffmpeg is not installed.
//!
//! Skipped in debug builds, where the `resnet18` pose net is slow enough to dominate the suite;
//! `cargo test --release` runs it.

use std::path::Path;

use animalpose::data::sessions::{read_npz, NpyArray, NpyData};

/// The frames the pipeline reads: a bright blob walking across a grey floor.
fn write_frames(
    dir: &Path,
    frames: usize,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    for t in 0..frames {
        let mut image = image::GrayImage::from_pixel(width, height, image::Luma([40u8]));
        let cx = 12 + (t as u32 * 8) % (width - 24);
        let cy = height / 2;
        for y in (cy - 8)..(cy + 8) {
            for x in (cx - 6)..(cx + 6) {
                image.put_pixel(x, y, image::Luma([220u8]));
            }
        }
        image.save(dir.join(format!("frame_{t:04}.png")))?;
    }
    Ok(())
}

/// The same clip as an `.mp4`, through the crate's own writer.
fn write_video(path: &Path, frames: usize, width: u32, height: u32) -> anyhow::Result<()> {
    let mut writer = animalpose::data::video::VideoWriter::new(path, 30.0);
    for t in 0..frames {
        let mut image = animalpose::algos::image::Bgr::filled(width, height, [40, 40, 40]);
        let cx = 12 + (t as u32 * 8) % (width - 24);
        let cy = height / 2;
        image.fill_rect(cx - 6, cy - 8, cx + 6, cy + 8, [220, 220, 220]);
        writer.write(&image)?;
    }
    writer.finish()?;
    Ok(())
}

/// `weights/<kind>_general.pt`, or `None` when it is absent or a git-lfs pointer.
fn general(kind: &str) -> Option<String> {
    let path = animalpose::data::repo_root()?
        .join("weights")
        .join(format!("{kind}_general.pt"));
    let bytes = std::fs::read(&path).ok()?;
    (!animalpose::models::torch_pt::is_lfs_pointer(&bytes)).then(|| path.display().to_string())
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "resnet18 is slow in debug builds; run with --release"
)]
fn predict_writes_every_output() {
    let (Some(detector), Some(pose), Some(temporal), Some(behavior)) = (
        general("detector"),
        general("pose"),
        general("temporal"),
        general("behavior"),
    ) else {
        eprintln!("SKIP: weights/*_general.pt are absent (git lfs pull)");
        return;
    };
    let base = std::env::temp_dir().join(format!("animalpose-predict-e2e-{}", std::process::id()));
    let video = animalpose::data::video::ffmpeg_available();
    let frames = if video {
        std::fs::create_dir_all(&base).expect("the temporary directory is writable");
        let path = base.join("clip.mp4");
        write_video(&path, 8, 96, 96).expect("the video is written");
        path
    } else {
        let dir = base.join("frames");
        write_frames(&dir, 8, 96, 96).expect("the frames are written");
        dir
    };
    let out = base.join("out");

    let config = base.join("pipeline.yaml");
    std::fs::create_dir_all(&base).expect("the temporary directory is writable");
    std::fs::write(
        &config,
        format!(
            concat!(
                "checkpoints:\n",
                "  detector: {}\n",
                "  pose: {}\n",
                "  temporal: {}\n",
                "  behavior: {}\n",
                "pipeline:\n",
                "  det_size: [96, 96]\n",
                "  det_thresh: 0.0\n",
                "  crop_size: 48\n",
                "  max_det: 2\n",
                "  embedder: color\n",
                "  behavior_conf: 0.0\n",
                "  behavior_smooth: 1\n",
                "  behavior_prior_adapt: false\n",
                "tracker:\n",
                "  high_thresh: 0.0\n",
                "  low_thresh: 0.0\n",
                "  max_animals: 1\n",
            ),
            detector, pose, temporal, behavior,
        ),
    )
    .expect("the config is written");

    let args: Vec<String> = [
        "predict",
        "--config",
        &config.display().to_string(),
        "--frames",
        &frames.display().to_string(),
        "--out",
        &out.display().to_string(),
        "--fps",
        "30",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    animalpose::cli::run(&args).expect("`predict` runs end to end");

    let csv = std::fs::read_to_string(out.join("tracks.csv")).expect("tracks.csv is written");
    let rows: Vec<&str> = csv.trim_end().split("\r\n").collect();
    assert_eq!(
        rows[0],
        "frame,time_s,track,keypoint,x,y,confidence,imputed,behavior"
    );
    assert!(
        rows.len() > 1 + 25,
        "one row per keypoint: {} rows",
        rows.len()
    );

    let arrays = read_npz(out.join("tracks.npz")).expect("tracks.npz is written");
    assert!(arrays.contains_key("keypoint_names"));
    assert!(arrays.contains_key("times"));
    assert!(arrays.contains_key("behaviors"));
    let track = arrays
        .keys()
        .find(|name| name.starts_with("track") && name.ends_with("_keypoints"))
        .cloned()
        .expect("a track was written");
    let array: &NpyArray = &arrays[&track];
    assert_eq!(array.shape.len(), 3, "keypoints are (T, K, 2)");
    assert_eq!(array.shape[1], 25);
    match &array.data {
        NpyData::Float(_) => {}
        other => panic!("keypoints must be float, got {other:?}"),
    }

    // The classifier was configured, so the catalogue outputs come with it.
    assert!(
        out.join("behavior_catalog.json").is_file(),
        "the catalogue is written"
    );
    let behavior =
        std::fs::read_to_string(out.join("behavior.csv")).expect("behavior.csv is written");
    assert!(
        behavior.starts_with("frame,time_s,track,top1,top1_p"),
        "{behavior}"
    );
    let info: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("behavior_catalog.json")).unwrap())
            .expect("the catalogue is JSON");
    assert_eq!(info.as_array().map(Vec::len), Some(133));

    assert!(out.join("timing.json").is_file(), "timing.json is written");
    if video {
        let info = animalpose::data::video::probe(&out.join("overlay.mp4"))
            .expect("overlay.mp4 is a readable video");
        assert_eq!((info.width, info.height, info.frames), (96, 96, 8));
    }

    let _ = std::fs::remove_dir_all(&base);
}
