//! The acceptance rules of `label_videos`, ported from `tests/test_selftrain.py`: a stand-in teacher
//! (boxes from the dark pixels, keypoints at fixed fractions of each box) on hand-drawn arenas, so
//! every rule can be switched on and off from the options.

use super::*;
use serde_json::Value;

const CUP: (u32, u32, u32, u32) = (12, 20, 26, 34);
const PAD_X: f64 = 20.0;
const PAD_Y: f64 = 14.0;

fn base(out: &Path) -> LabelOptions {
    LabelOptions {
        out: out.to_path_buf(),
        hi: 0.5,
        lo: 0.2,
        kp_conf: 0.4,
        flip_tol: 0.3,
        min_kps: 4,
        min_seq: 8,
        crop_size: 64,
        det_size: (256, 320),
        progress: false,
        ..LabelOptions::default()
    }
}

struct Stub {
    score: f64,
    conf: f64,
    cup: bool,
    sliver: bool,
    layout: Vec<(&'static str, (f64, f64))>,
}

impl Stub {
    fn new(score: f64) -> Self {
        Self {
            score,
            conf: 0.9,
            cup: false,
            sliver: false,
            layout: vec![],
        }
    }
}

impl Teacher for Stub {
    fn detect(&self, frame: &GrayF) -> Result<(Vec<Box4>, Vec<f64>)> {
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
        for y in 50..frame.height {
            for x in 0..frame.width {
                if frame.pixel(x, y) < 100.0 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        let (x0, y0, x1, y1) = (f64::from(x0), f64::from(y0), f64::from(x1), f64::from(y1));
        let mut boxes = vec![[x0 - PAD_X, y0 - PAD_Y, x1 + 1.0 + PAD_X, y1 + 1.0 + PAD_Y]];
        if self.cup {
            let (a, b, c, d) = CUP;
            boxes.push([
                f64::from(a) - PAD_X,
                f64::from(b) - PAD_Y,
                f64::from(c) + PAD_X,
                f64::from(d) + PAD_Y,
            ]);
        }
        if self.sliver {
            boxes.push([x0, y0, x0 + 19.0, y0 + 1.0]);
        }
        let n = boxes.len();
        Ok((boxes, vec![self.score; n]))
    }

    fn estimate(&self, _frame: &GrayF, boxes: &[Box4]) -> Result<Poses> {
        let mut kps = Vec::new();
        for b in boxes {
            let row = CANONICAL_NAMES
                .iter()
                .map(|name| {
                    let side = if name.starts_with("left_") {
                        0.25
                    } else if name.starts_with("right_") {
                        0.75
                    } else {
                        0.5
                    };
                    let (fx, fy) = self
                        .layout
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, f)| *f)
                        .unwrap_or((side, 0.5));
                    [b[0] + fx * (b[2] - b[0]), b[1] + fy * (b[3] - b[1])]
                })
                .collect();
            kps.push(row);
        }
        let confs = vec![vec![self.conf; K]; boxes.len()];
        Ok((kps, confs))
    }
}

fn temp(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ap-selftrain-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn save(path: &Path, img: &image::GrayImage) {
    img.save(path).unwrap();
}

/// `_flat_arena`: a 24x14 dark body walking right across a bright floor, and a static cup.
fn flat_arena(folder: &Path, n: u32) {
    std::fs::create_dir_all(folder).unwrap();
    for t in 0..n {
        let mut img = image::GrayImage::from_pixel(160, 96, image::Luma([200]));
        for y in CUP.1..CUP.3 {
            for x in CUP.0..CUP.2 {
                img.put_pixel(x, y, image::Luma([60]));
            }
        }
        let x = 24 + 5 * t;
        for y in 60..74 {
            for xx in x..x + 24 {
                img.put_pixel(xx, y, image::Luma([30]));
            }
        }
        save(&folder.join(format!("f{t:04}.png")), &img);
    }
}

/// `_tailed_arena`: a 30x16 body with a 2 px tail off its lower right corner, in a walled arena.
fn tailed_arena(folder: &Path, n: u32) {
    std::fs::create_dir_all(folder).unwrap();
    for t in 0..n {
        let mut img = image::GrayImage::from_pixel(320, 120, image::Luma([90]));
        for y in 20..110 {
            for x in 20..300 {
                img.put_pixel(x, y, image::Luma([200]));
            }
        }
        let x = 40 + 5 * t;
        for y in 60..76 {
            for xx in x..x + 30 {
                img.put_pixel(xx, y, image::Luma([30]));
            }
        }
        for y in 74..76 {
            for xx in x + 30..x + 44 {
                img.put_pixel(xx, y, image::Luma([30]));
            }
        }
        save(&folder.join(format!("f{t:04}.png")), &img);
    }
}

struct Run {
    stats: LabelStats,
    coco: Value,
    per_image: std::collections::BTreeMap<i64, Vec<Value>>,
    out: PathBuf,
}

fn run(root: &Path, out: PathBuf, teacher: &Stub, change: impl FnOnce(&mut LabelOptions)) -> Run {
    let mut opts = base(&out);
    change(&mut opts);
    let stats = label_videos(root, teacher, &opts).unwrap();
    let coco: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("train.json")).unwrap()).unwrap();
    let mut per_image: std::collections::BTreeMap<i64, Vec<Value>> = Default::default();
    for ann in coco["annotations"].as_array().unwrap() {
        per_image
            .entry(ann["image_id"].as_i64().unwrap())
            .or_default()
            .push(ann.clone());
    }
    Run {
        stats,
        coco,
        per_image,
        out,
    }
}

fn keypoints(ann: &Value) -> Vec<[f64; 3]> {
    let flat: Vec<f64> = ann["keypoints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    flat.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn bbox(ann: &Value) -> [f64; 4] {
    let b: Vec<f64> = ann["bbox"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    [b[0], b[1], b[2], b[3]]
}

fn stem(im: &Value) -> usize {
    let name = im["file_name"].as_str().unwrap();
    Path::new(name)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .parse()
        .unwrap()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn count_jpgs(dir: &Path) -> usize {
    let mut n = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                n += count_jpgs(&p);
            } else if p.extension().is_some_and(|x| x == "jpg") {
                n += 1;
            }
        }
    }
    n
}

#[test]
fn the_requested_range_labels_the_first_half_of_the_recording() {
    let tmp = temp("range");
    let root = tmp.join("root");
    flat_arena(&root.join("cage_a"), 20);
    let teacher = Stub::new(1.0);
    let whole = run(&root, tmp.join("whole"), &teacher, |o| o.per_source = 20);
    let half = run(&root, tmp.join("half"), &teacher, |o| {
        o.per_source = 20;
        o.time_range = (0.0, 0.5);
    });
    assert_eq!((whole.stats.frames, whole.stats.labeled), (20, 20));
    assert_eq!((half.stats.frames, half.stats.labeled), (10, 10));
    let images = half.coco["images"].as_array().unwrap();
    assert_eq!(
        images.iter().map(stem).collect::<Vec<_>>(),
        (0..10).collect::<Vec<_>>()
    );
    for (im, ann) in images
        .iter()
        .zip(half.coco["annotations"].as_array().unwrap())
    {
        let t = stem(im) as f64;
        assert!(close(bbox(ann)[0], 24.0 + 5.0 * t - 6.0, 0.5), "{ann}");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn only_confident_detections_teach_an_animal() {
    let tmp = temp("confident");
    let root = tmp.join("root");
    flat_arena(&root.join("cage_a"), 18);
    let uncertain = run(&root, tmp.join("uncertain"), &Stub::new(0.5), |o| {
        o.per_source = 18;
        o.hi = 0.9;
    });
    assert_eq!(uncertain.stats.frames, 18);
    assert_eq!(uncertain.stats.skipped_uncertain, 18);
    assert_eq!(uncertain.stats.labeled, 0);
    assert!(uncertain.coco["images"].as_array().unwrap().is_empty());
    assert!(uncertain.coco["annotations"].as_array().unwrap().is_empty());
    assert_eq!(count_jpgs(&uncertain.out.join("images")), 0);

    let accepted = run(&root, tmp.join("accepted"), &Stub::new(0.5), |o| {
        o.per_source = 18;
        o.hi = 0.4;
    });
    let s = &accepted.stats;
    assert_eq!((s.labeled, s.animals, s.keypoints), (18, 18, 25 * 18));
    assert_eq!((s.skipped_uncertain, s.no_animal), (0, 0));
    assert_eq!(accepted.coco["images"].as_array().unwrap().len(), 18);
    assert!(accepted.per_image.values().all(|v| v.len() == 1));
    let images = accepted.coco["images"].as_array().unwrap();
    for (im, ann) in images
        .iter()
        .zip(accepted.coco["annotations"].as_array().unwrap())
    {
        let t = stem(im) as f64;
        let (x0, y0) = (24.0 + 5.0 * t, 60.0);
        let (x1, y1) = (x0 + 24.0, y0 + 14.0);
        assert_eq!(ann["num_keypoints"], 25);
        let [bx, by, bw, bh] = bbox(ann);
        let (pad_x, pad_y) = (0.1 * (x1 - x0 + 6.0), 0.1 * (y1 - y0 + 6.0));
        assert!(bw < 0.75 * (x1 - x0 + 2.0 * PAD_X) && bh < 0.75 * (y1 - y0 + 2.0 * PAD_Y));
        assert!(
            close(bx, x0 - 3.0 - pad_x, 0.5) && close(by, y0 - 3.0 - pad_y, 0.5),
            "{ann}"
        );
        assert!(close(bw, x1 - x0 + 6.0 + 2.0 * pad_x, 0.5));
        assert!(close(bh, y1 - y0 + 6.0 + 2.0 * pad_y, 0.5));
        let kp = keypoints(ann);
        assert_eq!(kp[0][2], 2.0);
        assert!(close(kp[0][0], bx + bw / 2.0, 1.0));
    }
    let missed = run(&root, tmp.join("missed"), &Stub::new(0.05), |o| {
        o.per_source = 18;
        o.hi = 0.4;
    });
    assert_eq!((missed.stats.no_animal, missed.stats.labeled), (18, 0));
    assert!(missed.coco["annotations"].as_array().unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn the_background_filter_decides_what_counts_as_an_animal() {
    let tmp = temp("background");
    let root = tmp.join("root");
    flat_arena(&root.join("cage_a"), 18);
    let teacher = Stub {
        cup: true,
        ..Stub::new(1.0)
    };
    let filtered = run(&root, tmp.join("filtered"), &teacher, |o| o.per_source = 18);
    assert_eq!((filtered.stats.labeled, filtered.stats.animals), (18, 18));
    assert!(filtered.per_image.values().all(|v| v.len() == 1));
    let anns = filtered.coco["annotations"].as_array().unwrap();
    assert!(anns.iter().all(|a| bbox(a)[1] > f64::from(CUP.3)));
    let taught = run(&root, tmp.join("taught"), &teacher, |o| {
        o.per_source = 18;
        o.min_seq = 10_000_000;
    });
    assert_eq!((taught.stats.labeled, taught.stats.animals), (18, 36));
    assert!(taught.per_image.values().all(|v| v.len() == 2));
    let cups: Vec<[f64; 4]> = taught.coco["annotations"]
        .as_array()
        .unwrap()
        .iter()
        .map(bbox)
        .filter(|b| b[1] < f64::from(CUP.3))
        .collect();
    assert_eq!(cups.len(), 18);
    let want = [
        f64::from(CUP.0) - PAD_X,
        f64::from(CUP.1) - PAD_Y,
        f64::from(CUP.2 - CUP.0) + 2.0 * PAD_X,
        f64::from(CUP.3 - CUP.1) + 2.0 * PAD_Y,
    ];
    assert!(cups
        .iter()
        .all(|b| b.iter().zip(&want).all(|(a, w)| close(*a, *w, 1e-6))));
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn keypoints_are_rejected_when_off_the_motion_or_off_the_mirror() {
    let tmp = temp("mirror");
    let root = tmp.join("root");
    flat_arena(&root.join("cage_a"), 18);
    let teacher = Stub {
        layout: vec![("nose", (0.5, -0.1)), ("spine_lower", (0.72, 0.5))],
        ..Stub::new(1.0)
    };
    let masked = run(&root, tmp.join("masked"), &teacher, |o| o.per_source = 18);
    let unmasked = run(&root, tmp.join("unmasked"), &teacher, |o| {
        o.per_source = 18;
        o.min_seq = 10_000_000;
    });
    let (nose, lower, neck) = (index("nose"), index("spine_lower"), index("neck"));
    for (r, expect, is_masked) in [(&masked, 23, true), (&unmasked, 24, false)] {
        assert_eq!(r.per_image.len(), 18);
        for ann in r.coco["annotations"].as_array().unwrap() {
            let kp = keypoints(ann);
            assert_eq!(ann["num_keypoints"], expect);
            assert_eq!(kp.iter().filter(|p| p[2] == 2.0).count(), expect);
            assert_eq!(kp[lower], [0.0, 0.0, 0.0]); // not reproduced by the mirrored estimate
            assert_eq!(kp[neck][2], 2.0);
            if is_masked {
                assert_eq!(kp[nose], [0.0, 0.0, 0.0]); // confident, but not on a moving pixel
            } else {
                let [_, by, _, bh] = bbox(ann);
                assert_eq!(kp[nose][2], 2.0);
                assert!(close(kp[nose][1], by - 0.1 * bh, 1.0));
            }
        }
    }
    assert_eq!(masked.stats.keypoints, 23 * 18);
    assert_eq!(unmasked.stats.keypoints, 24 * 18);
    let first = &masked.per_image[&0][0]["keypoints"];
    assert_eq!(
        first.as_array().unwrap()[..3],
        serde_json::json!([0.0, 0.0, 0]).as_array().unwrap()[..]
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn a_nose_on_the_tail_side_is_thrown_out_wholesale() {
    let tmp = temp("tail");
    let root = tmp.join("root");
    tailed_arena(&root.join("cage_a"), 24);
    let flipped_teacher = Stub {
        layout: vec![("nose", (0.5, 0.8)), ("tail_base", (0.5, 0.2))],
        ..Stub::new(1.0)
    };
    let flipped = run(&root, tmp.join("flipped"), &flipped_teacher, |o| {
        o.per_source = 24;
        o.boxes = "blob".into();
    });
    let sound_teacher = Stub {
        layout: vec![("nose", (0.5, 0.2)), ("tail_base", (0.5, 0.8))],
        ..Stub::new(1.0)
    };
    let sound = run(&root, tmp.join("sound"), &sound_teacher, |o| {
        o.per_source = 24;
        o.boxes = "blob".into();
    });
    assert_eq!((sound.stats.frames, flipped.stats.frames), (24, 24));
    assert_eq!(sound.stats.no_animal, 0);
    assert_eq!(
        (
            sound.stats.labeled,
            sound.stats.head_tail_flip,
            sound.stats.keypoints
        ),
        (24, 0, 25 * 24)
    );
    assert_eq!(
        (flipped.stats.head_tail_flip, flipped.stats.labeled),
        (24, 0)
    );
    assert!(flipped.coco["images"].as_array().unwrap().is_empty());
    assert_eq!(sound.coco["images"].as_array().unwrap().len(), 24);
    let split: Value =
        serde_json::from_str(&std::fs::read_to_string(flipped.out.join("split.json")).unwrap())
            .unwrap();
    assert_eq!(split["train"], serde_json::json!(["cage_a"]));
    for ann in sound.coco["annotations"].as_array().unwrap() {
        let kp = keypoints(ann);
        assert_eq!(
            (kp[index("nose")][2], kp[index("tail_base")][2]),
            (2.0, 2.0)
        );
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn one_row_of_a_box_rejects_the_whole_frame() {
    let tmp = temp("sliver");
    let root = tmp.join("root");
    flat_arena(&root.join("cage_a"), 18);
    let teacher = Stub {
        sliver: true,
        ..Stub::new(1.0)
    };
    let dropped = run(&root, tmp.join("sliver"), &teacher, |o| o.per_source = 18);
    let s = &dropped.stats;
    assert_eq!(
        (s.frames, s.labeled, s.animals, s.no_animal, s.keypoints),
        (18, 0, 0, 0, 0)
    );
    assert!(dropped.coco["images"].as_array().unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn recordings_without_frames_or_without_a_background_are_left_out() {
    let tmp = temp("empty");
    let root = tmp.join("root");
    flat_arena(&root.join("cage_a"), 6);
    std::fs::write(root.join("empty.avi"), b"").unwrap();
    let teacher = Stub::new(1.0);
    let short = run(&root, tmp.join("short"), &teacher, |o| {
        o.per_source = 4;
        o.min_seq = 10_000_000;
        o.holdout = false;
    });
    assert_eq!(short.stats.sources, 2); // the video in the root is listed before the folder
    let split: Value =
        serde_json::from_str(&std::fs::read_to_string(short.out.join("split.json")).unwrap())
            .unwrap();
    assert_eq!(split["train"], serde_json::json!(["empty.avi", "cage_a"]));
    assert_eq!(split["root"].as_str().unwrap(), abspath(&root));
    assert_eq!((short.stats.frames, short.stats.labeled), (4, 4));
    assert_eq!(short.stats.keypoints, 25 * 4);
    let sid = source_id("cage_a");
    let images = short.coco["images"].as_array().unwrap();
    assert_eq!(images.len(), 4);
    assert!(images.iter().all(|im| im["file_name"]
        .as_str()
        .unwrap()
        .starts_with(&format!("{sid}/"))));
    // the four frames of np.linspace(0, 5, 4).astype(int)
    assert_eq!(images.iter().map(stem).collect::<Vec<_>>(), [0, 1, 3, 5]);
    for im in images {
        let path = short
            .out
            .join("images")
            .join(im["file_name"].as_str().unwrap());
        let decoded = image::open(&path).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (160, 96));
        assert_eq!(
            (im["width"].as_u64(), im["height"].as_u64()),
            (Some(160), Some(96))
        );
    }
    let blind = run(&root, tmp.join("blobless"), &teacher, |o| {
        o.per_source = 4;
        o.min_seq = 10_000_000;
        o.holdout = false;
        o.boxes = "blob".into();
    });
    assert_eq!(blind.stats.sources, 2);
    assert_eq!((blind.stats.frames, blind.stats.no_animal), (0, 0));
    assert!(blind.coco["images"].as_array().unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn the_written_labels_are_canonical_coco_and_the_split_holds_out_by_hash() {
    let tmp = temp("coco");
    let root = tmp.join("cages");
    for name in ["cage_a", "cage_b", "cage_d", "cage_e"] {
        flat_arena(&root.join(name), 12);
    }
    let teacher = Stub::new(1.0);
    let r = run(&root, tmp.join("pseudo"), &teacher, |o| o.per_source = 2);
    let split: Value =
        serde_json::from_str(&std::fs::read_to_string(r.out.join("split.json")).unwrap()).unwrap();
    assert_eq!(split["train"], serde_json::json!(["cage_a", "cage_b"]));
    assert_eq!(split["held_out"], serde_json::json!(["cage_d", "cage_e"]));
    assert_eq!((r.stats.sources, r.stats.frames), (4, 4));
    assert_eq!(r.stats.labeled, r.stats.animals);
    assert_eq!(Some(r.stats.labeled), r.stats.images);
    let cat = &r.coco["categories"];
    assert_eq!(
        cat,
        &serde_json::json!([{"id": 1, "name": "animal", "keypoints": CANONICAL_NAMES, "skeleton": []}])
    );
    let images = r.coco["images"].as_array().unwrap();
    let ids: Vec<i64> = images.iter().map(|im| im["id"].as_i64().unwrap()).collect();
    assert_eq!(ids, (0..images.len() as i64).collect::<Vec<_>>());
    for ann in r.coco["annotations"].as_array().unwrap() {
        assert_eq!(
            (ann["category_id"].as_i64(), ann["iscrowd"].as_i64()),
            (Some(1), Some(0))
        );
        let kp = keypoints(ann);
        assert!(kp.iter().all(|p| p[2] == 0.0 || p[2] == 2.0));
        assert!(kp
            .iter()
            .filter(|p| p[2] == 0.0)
            .all(|p| p[0] == 0.0 && p[1] == 0.0));
        let [_, _, w, h] = bbox(ann);
        assert!(close(ann["area"].as_f64().unwrap(), w * h, 1e-9));
    }
    let sids: std::collections::BTreeSet<String> = images
        .iter()
        .map(|im| {
            im["file_name"]
                .as_str()
                .unwrap()
                .split('/')
                .next()
                .unwrap()
                .to_string()
        })
        .collect();
    let want: std::collections::BTreeSet<String> =
        ["cage_a", "cage_b"].iter().map(|n| source_id(n)).collect();
    assert_eq!(sids, want);
    // the --include filter and --no-holdout
    let only = run(&root, tmp.join("only"), &teacher, |o| {
        o.per_source = 2;
        o.include = Some(vec!["cage_a".into()]);
    });
    assert_eq!(only.stats.sources, 1);
    let none = run(&root, tmp.join("none"), &teacher, |o| {
        o.per_source = 2;
        o.include = Some(vec!["atlas".into()]);
    });
    assert_eq!((none.stats.sources, none.stats.frames), (0, 0));
    let every = run(&root, tmp.join("every"), &teacher, |o| {
        o.per_source = 2;
        o.holdout = false;
    });
    assert_eq!((every.stats.sources, every.stats.frames), (4, 8));
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn stats_print_in_python_key_order() {
    let stats = LabelStats {
        sources: 1,
        images: Some(0),
        ..LabelStats::default()
    };
    assert_eq!(
        stats.to_json().dumps(),
        r#"{"sources": 1, "frames": 0, "labeled": 0, "skipped_uncertain": 0, "no_animal": 0, "animals": 0, "keypoints": 0, "head_tail_flip": 0, "images": 0}"#
    );
}
