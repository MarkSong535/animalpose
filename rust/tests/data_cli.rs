//! `animalpose data convert|audit|prepare` end to end, on inputs written here — the assertions of
//! `tests/test_cli_behavior.py` (convert, prepare) and `tests/test_data_audit.py` (audit), with the
//! numbers of the Python package on the very same files where the test says so.

use animalpose::algos::image::Bgr;
use animalpose::cli::audit::{audit, calibrate_psf, profile_from, surrogate_crops, Source};
use animalpose::data::measure::{bgr_to_gray, degrade_defocus, edge_width};
use animalpose::data::pyjson::{loads, PyVal};
use animalpose::data::real::{convert_coco_keypoints, convert_lp_crim13, convert_mirror_mouse};
use std::path::{Path, PathBuf};

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("animalpose-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(args: &[&str]) -> anyhow::Result<()> {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    animalpose::cli::run(&args)
}

fn read_json(path: &Path) -> PyVal {
    loads(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn list(v: &PyVal, key: &str) -> Vec<PyVal> {
    v.get(key).and_then(PyVal::as_list).unwrap().to_vec()
}

fn f(v: &PyVal) -> f64 {
    v.as_f64().unwrap()
}

fn png(path: &Path, w: u32, h: u32, value: u8) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    image::RgbImage::from_pixel(w, h, image::Rgb([value, value, value]))
        .save(path)
        .unwrap();
}

// --------------------------------------------------------------------------------- data convert

fn xy(names: &[&str]) -> Vec<String> {
    names
        .iter()
        .flat_map(|n| [n.to_string(), n.to_string()])
        .collect()
}

#[test]
fn data_convert_turns_a_deeplabcut_csv_into_coco() {
    let dir = tmp("convert");
    let csv = dir.join("labeled-data/vid1/CollectedData_Me.csv");
    let parts = ["nose", "neck", "tail_base"];
    let mut rows: Vec<Vec<String>> = vec![
        [vec!["scorer".into(); 3], vec!["Me".into(); 12]].concat(),
        [
            vec!["individuals".into(); 3],
            vec!["mouse1".into(); 6],
            vec!["mouse2".into(); 6],
        ]
        .concat(),
        [vec!["bodyparts".into(); 3], xy(&parts), xy(&parts)].concat(),
        [
            vec!["coords".to_string(); 3],
            ["x", "y"].repeat(6).iter().map(|s| s.to_string()).collect(),
        ]
        .concat(),
    ];
    for r in [
        "labeled-data,vid1,img0.png,10,20,25,22,40,28,60,20,75,22,90,28",
        "labeled-data,vid1,img1.png,12,22,27,24,42,30,,,,,,",
        "labeled-data,vid1,img2.png,,,,,,,,,,,,",
    ] {
        rows.push(r.split(',').map(str::to_string).collect());
    }
    std::fs::create_dir_all(csv.parent().unwrap()).unwrap();
    let text: Vec<String> = rows.iter().map(|r| r.join(",")).collect();
    std::fs::write(&csv, text.join("\n") + "\n").unwrap();
    for name in ["img0.png", "img1.png", "img2.png"] {
        png(&dir.join("labeled-data/vid1").join(name), 120, 96, 0);
    }
    let out = dir.join("vid1.json");
    run(&[
        "data",
        "convert",
        csv.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--image-root",
        dir.to_str().unwrap(),
    ])
    .unwrap();
    let coco = read_json(&out);
    let cats = list(&coco, "categories");
    assert_eq!(cats[0].get("name").unwrap().as_str(), Some("animal"));
    assert_eq!(
        cats[0].get("keypoints").unwrap().dumps(),
        r#"["nose", "neck", "tail_base"]"#
    );
    let images = list(&coco, "images");
    for (i, im) in images.iter().enumerate() {
        assert_eq!(
            im.get("file_name").unwrap().as_str().unwrap(),
            format!("labeled-data/vid1/img{i}.png")
        );
        assert_eq!(
            (f(im.get("height").unwrap()), f(im.get("width").unwrap())),
            (96.0, 120.0)
        );
    }
    let anns = list(&coco, "annotations");
    assert_eq!(anns.len(), 3);
    let tracks: Vec<f64> = anns.iter().map(|a| f(a.get("track_id").unwrap())).collect();
    assert_eq!(tracks, [0.0, 1.0, 0.0]);
    assert_eq!(
        anns[0].get("bbox").unwrap().dumps(),
        "[2.0, 14.2, 46.0, 19.6]"
    );
    assert_eq!(f(anns[2].get("image_id").unwrap()), 1.0);

    let single = dir.join("CollectedData_Single.csv");
    std::fs::write(
        &single,
        "scorer,Me,Me,Me,Me,Me,Me\nbodyparts,snout,snout,tailbase,tailbase,leftear,leftear\n\
         coords,x,y,x,y,x,y\nimg0.png,10,20,25,22,40,28\nimg1.png,12,22,27,24,42,30\n",
    )
    .unwrap();
    let out = dir.join("single.json");
    run(&[
        "data",
        "convert",
        single.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ])
    .unwrap();
    let doc = read_json(&out);
    assert_eq!(list(&doc, "images").len(), 2);
    assert!(list(&doc, "images")[0].get("height").is_none()); // no --image-root, no sizes
    std::fs::remove_dir_all(&dir).unwrap();
}

// --------------------------------------------------------------------------------- data prepare

/// `lp_csv`: the label table pandas writes (`scorer`/`bodyparts`/`coords`, then `file_name`).
fn lp_csv(path: &Path, bodyparts: &[String], rows: &[Vec<f64>], index: &[String]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut text = String::new();
    let cols: Vec<(String, String)> = bodyparts
        .iter()
        .flat_map(|b| [(b.clone(), "x".to_string()), (b.clone(), "y".to_string())])
        .collect();
    text += &format!("scorer,{}\n", vec!["Skeleton5"; cols.len()].join(","));
    text += &format!(
        "bodyparts,{}\n",
        cols.iter()
            .map(|c| c.0.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    text += &format!(
        "coords,{}\n",
        cols.iter()
            .map(|c| c.1.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    text += &format!("file_name{}\n", ",".repeat(cols.len()));
    for (name, r) in index.iter().zip(rows) {
        let vals: Vec<String> = r.iter().map(|v| format!("{v:?}")).collect();
        text += &format!("{name},{}\n", vals.join(","));
    }
    std::fs::write(path, text).unwrap();
}

#[test]
fn data_prepare_converts_every_fake_download() {
    let dir = tmp("prepare");
    let raw = dir.join("raw");
    let out = dir.join("processed");
    // CRIM13: two animals, one bodypart without a counterpart.
    let crim = raw.join("lp_crim13/extracted/crim13");
    let parts: Vec<String> = [
        "black_mouse_nose",
        "black_mouse_top_of_neck",
        "black_mouse_whisker_pad",
        "white_mouse_nose",
        "white_mouse_base_tail",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let index: Vec<String> = [1, 2]
        .iter()
        .flat_map(|a| (0..2).map(move |b| format!("s{a}/frame{b:02}.png")))
        .collect();
    let mut rows = Vec::new();
    let mut y = 10.0;
    for _ in &index {
        rows.push(vec![
            20.0,
            y,
            60.0,
            y,
            15.0,
            y + 5.0,
            40.0,
            y + 10.0,
            80.0,
            y + 12.0,
        ]);
        y += 30.0;
    }
    for split in ["labels_InD.csv", "labels_OOD.csv"] {
        lp_csv(&crim.join(split), &parts, &rows, &index);
    }
    for rel in &index {
        png(&crim.join(rel), 160, 120, 90);
    }
    let stats = convert_lp_crim13(&crim, &out.join("crim13")).unwrap();
    assert_eq!(
        stats.dumps(),
        r#"{"train": {"images": 4, "animals": 8, "unmapped_bodyparts": ["black_mouse_whisker_pad"]}, "val": {"images": 4, "animals": 8, "unmapped_bodyparts": ["black_mouse_whisker_pad"]}}"#
    );
    let doc = read_json(&out.join("crim13/train.json"));
    assert_eq!(list(&list(&doc, "categories")[0], "keypoints").len(), 25);
    assert!(list(&doc, "annotations")
        .iter()
        .all(|a| f(a.get("num_keypoints").unwrap()) > 0.0));
    // `(zlib.crc32(b"s1") % 100000) * 16 + animal`
    let ids: Vec<f64> = list(&doc, "annotations")
        .iter()
        .map(|a| f(a.get("track_id").unwrap()))
        .collect();
    assert_eq!(ids[..2], [PY_S1_TRACK, PY_S1_TRACK + 1.0]);

    // mirror-mouse: side above row 160, bottom below.
    let mirror = raw.join("lp_mirror_mouse/extracted/mirror-mouse");
    let mp: Vec<String> = [
        "paw1LH", "paw2LF", "paw3RF", "paw4RH", "tailBase", "tailMid", "nose",
    ]
    .iter()
    .flat_map(|bp| [format!("{bp}_top"), format!("{bp}_bot")])
    .collect();
    let mut mrows = Vec::new();
    let mut idx = Vec::new();
    for fr in 0..3 {
        let mut row = Vec::new();
        for i in 0..mp.len() / 2 {
            let i = i as f64;
            row.extend([
                20.0 + 6.0 * i,
                30.0 + 4.0 * i,
                20.0 + 6.0 * i,
                200.0 + 4.0 * i,
            ]);
        }
        mrows.push(row);
        idx.push(format!("sess/frame{fr:02}.png"));
        png(&mirror.join(&idx[fr]), 160, 320, 120);
    }
    lp_csv(&mirror.join("labels_InD.csv"), &mp, &mrows, &idx);
    lp_csv(
        &mirror.join("labels_OOD.csv"),
        &mp,
        &mrows[..1],
        &["other/frame00.png".to_string()],
    );
    png(&mirror.join("other/frame00.png"), 160, 320, 60);
    let stats = convert_mirror_mouse(&mirror, &out.join("mirror_mouse")).unwrap();
    assert_eq!(
        stats.dumps(),
        r#"{"side_train": {"images": 3, "animals": 3}, "bottom_train": {"images": 3, "animals": 3}, "side_val": {"images": 1, "animals": 1}, "bottom_val": {"images": 1, "animals": 1}}"#
    );
    let side = read_json(&out.join("mirror_mouse/side_train.json"));
    let names: Vec<String> = list(&side, "images")
        .iter()
        .map(|i| i.get("file_name").unwrap().as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        names,
        [
            "side/sess_frame00.jpg",
            "side/sess_frame01.jpg",
            "side/sess_frame02.jpg"
        ]
    );
    let im0 = &list(&side, "images")[0];
    assert_eq!(
        (f(im0.get("height").unwrap()), f(im0.get("width").unwrap())),
        (160.0, 160.0)
    );
    assert!(out
        .join("mirror_mouse/images/side/sess_frame00.jpg")
        .is_file());

    // iRodent: a COCO file whose names are the CalMS21 (= canonical) ones.
    let iro = raw.join("irodent/extracted");
    let names = [
        "nose",
        "left_ear",
        "right_ear",
        "neck",
        "left_hip",
        "right_hip",
        "tail_base",
        "whisker_pad",
    ];
    let kps: Vec<String> = (0..names.len())
        .flat_map(|i| {
            [
                format!("{}", 20 + 3 * i),
                format!("{}", 30 + 2 * i),
                "2".into(),
            ]
        })
        .collect();
    std::fs::create_dir_all(iro.join("annotations")).unwrap();
    std::fs::write(
        iro.join("annotations/irodent_coco.json"),
        format!(
            r#"{{"images": [{{"id": 0, "file_name": "img0.png", "height": 120, "width": 160}}], "categories": [{{"id": 1, "name": "mouse", "keypoints": {}, "skeleton": []}}], "annotations": [{{"id": 0, "image_id": 0, "category_id": 1, "iscrowd": 0, "num_keypoints": 7, "bbox": [10.0, 10.0, 60.0, 40.0], "keypoints": [{}]}}]}}"#,
            PyVal::from(names.to_vec()).dumps(),
            kps.join(", ")
        ),
    )
    .unwrap();
    png(&iro.join("images/img0.png"), 160, 120, 30);
    run(&[
        "data",
        "prepare",
        "--raw",
        raw.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--irodent",
    ])
    .unwrap();
    let doc = read_json(&out.join("irodent/test.json"));
    assert_eq!(list(&doc, "annotations").len(), 1);
    assert_eq!(
        std::fs::read_to_string(out.join("irodent/IMAGE_ROOT")).unwrap(),
        std::fs::canonicalize(iro.join("images"))
            .unwrap()
            .display()
            .to_string()
    );
    let stats = convert_coco_keypoints(
        &iro.join("annotations/irodent_coco.json"),
        &dir.join("again.json"),
        None,
    )
    .unwrap();
    assert_eq!(stats.get("mapping").unwrap().as_str(), Some("synthetic"));
    assert_eq!(stats.get("unmapped").unwrap().dumps(), r#"["whisker_pad"]"#);
    assert_eq!(stats.get("mapped").unwrap().as_list().unwrap().len(), 7);
    assert_eq!(stats.get("species").unwrap().dumps(), r#"{"mouse": 1}"#);

    // Rat 7M without its download is an error naming the missing file.
    let err = run(&[
        "data",
        "prepare",
        "--raw",
        dir.join("nothing").to_str().unwrap(),
        "--rat7m",
    ])
    .unwrap_err();
    assert!(format!("{err}").contains("mocap-s1-d1.mat"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// `(zlib.crc32(b"s1") % 100000) * 16`, from Python.
const PY_S1_TRACK: f64 = 562_432.0;

// ----------------------------------------------------------------------------------- data audit

/// `img_HH_MM_SS_mmm.png`, the frame stamps `estimate_fps` reads (`stamp(t, step_ms)` in Python).
fn stamp(t: u32, step_ms: u32) -> String {
    let ms = t * step_ms;
    format!(
        "img_{:02}_{:02}_{:02}_{:03}.png",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

/// A dark mouse-sized ellipse walking over a bright, faintly textured floor inside darker walls.
fn frame(t: u32) -> Bgr {
    let (w, h) = (200u32, 160u32);
    let mut img = Bgr::filled(w, h, [70, 80, 90]);
    let (cx, cy) = (
        50.0 + 2.5 * f64::from(t),
        80.0 + 12.0 * (f64::from(t) / 6.0).sin(),
    );
    for y in 12..h - 12 {
        for x in 12..w - 12 {
            let tex = ((x * 7 + y * 13) % 5) as u8;
            let (dx, dy) = ((f64::from(x) - cx) / 16.0, (f64::from(y) - cy) / 8.0);
            let px = if dx * dx + dy * dy <= 1.0 {
                [35 + tex, 35 + tex, 40 + tex]
            } else {
                [150 + tex, 170 + tex, 190 + tex]
            };
            img.set_pixel(x, y, px);
        }
    }
    img
}

fn write_png(path: &Path, img: &Bgr) {
    let rgb: Vec<u8> = img
        .pixels
        .chunks_exact(3)
        .flat_map(|p| [p[2], p[1], p[0]])
        .collect();
    image::RgbImage::from_raw(img.width, img.height, rgb)
        .unwrap()
        .save(path)
        .unwrap();
}

fn recording(dir: &Path, blur: Option<f64>) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    for t in 0..40 {
        let img = frame(t);
        let img = blur.map_or(img.clone(), |r| degrade_defocus(&img, r));
        write_png(&dir.join(stamp(t, 100)), &img);
    }
    dir.to_path_buf()
}

#[test]
fn the_audit_measures_the_recording_it_is_given() {
    let dir = tmp("audit");
    let sharp = recording(&dir.join("sharp"), None);
    let blurred = recording(&dir.join("blurred"), Some(3.0));
    let src = Source::open(&sharp).unwrap();
    let m = audit(&src, 24, 0.5, 2, 8, 25.0);
    let get = |a: &str, b: &str| f(m.get(a).unwrap().get(b).unwrap());
    assert_eq!(f(m.get("frames").unwrap()), 40.0);
    assert!((f(m.get("fps").unwrap()) - 10.0).abs() < 1e-9); // from the file-name stamps
    assert_eq!(f(m.get("duration_s").unwrap()), 4.0);
    assert_eq!(m.get("colour"), Some(&PyVal::Bool(true)));
    assert_eq!(get("edge_width_px", "blocks"), 4.0);
    assert_eq!(f(m.get("frames_with_blob").unwrap()), 24.0);
    assert_eq!(get("displacement_px", "windows"), 16.0);
    // The ellipse is 32 px long; the opened blob mask measures 30 px of it, as in Python.
    assert_eq!(get("animal_px", "median"), 30.0, "{}", m.dumps());
    assert!((f(m.get("motion_smear_body_frac").unwrap()) - 0.047).abs() < 1e-12);
    let smear = f(m.get("motion_smear_px").unwrap());
    assert!((smear - get("displacement_px", "median") * 0.5).abs() < 0.01);
    // The numbers the Python package prints for these files (`animalpose data audit <dir>
    // --frames 24 --windows 2 --window 8 --no-calibrate`).
    assert_eq!(m.get("edge_width_px").unwrap().dumps(), PY_SHARP_EDGE);
    assert_eq!(f(m.get("motion_smear_px").unwrap()), PY_SHARP_SMEAR);
    assert_eq!(m.get("background_grey").unwrap().dumps(), PY_SHARP_BG);

    let b = audit(&Source::open(&blurred).unwrap(), 24, 0.5, 2, 8, 25.0);
    let edge = |m: &PyVal| f(m.get("edge_width_px").unwrap().get("median").unwrap());
    assert!(edge(&b) > edge(&m) + 1.0, "{} vs {}", edge(&b), edge(&m));

    // The PSF calibration on clean crops of the sharp frames is monotone and picks the blur.
    let crops: Vec<Bgr> = (0..6).map(|t| frame(4 * t)).collect();
    let (psf_sharp, _) = calibrate_psf(edge(&m), &crops);
    let (psf_blur, curve) = calibrate_psf(edge(&b), &crops);
    assert!(psf_blur > psf_sharp, "{psf_blur} vs {psf_sharp}");
    let widths: Vec<f64> = match curve.get("edge_width_by_psf_px").unwrap() {
        PyVal::Dict(items) => items.iter().map(|(_, v)| f(v)).collect(),
        _ => unreachable!(),
    };
    assert!(
        widths.windows(2).all(|w| w[1] >= w[0]) && widths[10] > widths[0],
        "{widths:?}"
    );
    let (p, notes) = profile_from(&m, Some(psf_sharp));
    assert!(notes.iter().any(|n| n.contains("jpeg quality is assumed")));
    assert_eq!(p.get("jpeg").unwrap().dumps(), "[40.0, 80.0]");
    let g: Vec<_> = crops.iter().map(bgr_to_gray).collect();
    assert!(edge_width(&g, 2, 95.0).0 > 0.0);

    // The CLI: a calibrated profile YAML and the measurements as JSON.
    let surrogate = dir.join("surrogate");
    assert!(surrogate_crops(&surrogate, 4, 64).unwrap().is_empty()); // not a prepared dataset
    let out = dir.join("camera.yaml");
    let js = dir.join("measured.json");
    run(&[
        "data",
        "audit",
        blurred.to_str().unwrap(),
        "--frames",
        "16",
        "--windows",
        "2",
        "--window",
        "6",
        "--no-calibrate",
        "--out",
        out.to_str().unwrap(),
        "--json",
        js.to_str().unwrap(),
    ])
    .unwrap();
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.starts_with(&format!(
        "# animalpose data audit on {} (40 frames, 10.00 fps, 4 s, colour).",
        blurred.display()
    )));
    assert!(text.contains("# assumed: psf_px not calibrated"));
    let measured = loads(&std::fs::read_to_string(&js).unwrap()).unwrap();
    assert_eq!(measured.as_list().unwrap().len(), 1);

    // --root finds frame folders; one with three frames is too short.
    let root = dir.join("root");
    recording(&root.join("full"), None);
    std::fs::create_dir_all(root.join("short")).unwrap();
    for t in 0..3 {
        write_png(&root.join("short").join(stamp(t, 100)), &frame(t));
    }
    let js = dir.join("all.json");
    run(&[
        "data",
        "audit",
        "--root",
        root.to_str().unwrap(),
        "--frames",
        "12",
        "--windows",
        "1",
        "--window",
        "4",
        "--surrogate",
        dir.join("unprepared").to_str().unwrap(),
        "--json",
        js.to_str().unwrap(),
    ])
    .unwrap();
    let all = loads(&std::fs::read_to_string(&js).unwrap()).unwrap();
    let all = all.as_list().unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(
        all[1].get("error").and_then(PyVal::as_str),
        Some("too few frames")
    );
    assert_eq!(f(all[0].get("frames").unwrap()), 40.0);

    let err = run(&["data", "audit", "--frames", "8"]).unwrap_err();
    assert!(format!("{err}").contains("give a path"));
    std::fs::remove_dir_all(&dir).unwrap();
}

const PY_SHARP_EDGE: &str =
    r#"{"median": 1.54, "p10": 1.54, "p90": 1.54, "silhouette_spread": 0.04, "blocks": 4}"#;
const PY_SHARP_SMEAR: f64 = 1.4;
const PY_SHARP_BG: &str = "[82.0, 175.0, 178.0]";
