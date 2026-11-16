//! Image rendering of the procedural simulator: `src/animalpose/data/synthetic.py:render`,
//! `render_video`, `Frame`, `_draw_animal`, `_background`, `Appearance` / `COAT_COLORS`,
//! `make_cameras` / `VIEW_PRESETS` and the `elevation` range of `random_camera`.
//!
//! | Python | Rust |
//! | --- | --- |
//! | `simulate(..., coats=...)` (coat colours, food pellets) | [`simulate_scene`] → [`Scene`] |
//! | `Appearance` / `COAT_COLORS` / `_lighter` | [`Appearance`] / [`COAT_COLORS`] / [`lighter`] |
//! | `random_camera(arena, rng, size, elevation)` | [`random_camera`] |
//! | `make_cameras` / `VIEW_PRESETS` | [`make_cameras`] / [`VIEW_PRESETS`] |
//! | `_background` / `_draw_animal` | [`background`] / [`draw_animal`] |
//! | `Frame` / `render` / `render_video` | [`Frame`] / [`render`] / [`render_video`] |
//!
//! Animals are drawn far to near, each into its own canvas and mask and pasted where its mask is
//! set, so a nearer animal hides a farther one; a keypoint is visible when the label map at its
//! (rounded) pixel belongs to its own animal, and the box is the extent of the animal's visible
//! pixels. The far-side legs go behind the trunk and are shaded darker, the near-side ones in
//! front — the depth cue that tells near paws from far ones in side views.
//!
//! `// differs from Python:`
//! * The rasterisers are [`crate::data::raster`]'s, not OpenCV's: edges differ by single pixels
//!   (see that module for the fitted offsets), sizes, colours and ordering are the same.
//! * [`simulate_scene`] draws the coat colours and food pellets first and then calls
//!   [`crate::data::synthetic::simulate`], which draws each body length at the start of its animal,
//!   where Python draws all of them before the food: same distributions, different stream
//!   (the [`Rng`] is not numpy's anyway).

use crate::algos::image::Bgr;
use crate::data::raster::{
    convex_hull, draw_aa, draw_hard, draw_hard_f, gaussian_blur, resize_linear, ImageF, Shape,
};
use crate::data::synthetic::{
    body_length_cm, camera_from_view, index, simulate, Camera, Rng, Simulation, NUM_KEYPOINTS,
};
use anyhow::{bail, Result};

/// `COAT_COLORS` (BGR), in Python's dict order.
pub const COAT_COLORS: [(&str, [u8; 3]); 6] = [
    ("black", [40, 38, 36]),
    ("white", [225, 228, 230]),
    ("brown", [50, 85, 130]),
    ("agouti", [70, 100, 120]),
    ("grey", [125, 125, 125]),
    ("cream", [150, 190, 215]),
];

/// The skin colour of tails, paws, noses and ears (`skin` in `_draw_animal`, BGR).
pub const SKIN: [u8; 3] = [150, 160, 215];

/// `VIEW_PRESETS`: `(elevation, azimuth)` in degrees.
pub const VIEW_PRESETS: [(&str, f64, f64); 3] = [
    ("top", 90.0, -90.0),
    ("side", 8.0, -90.0),
    ("angled", 45.0, -135.0),
];

/// `Appearance`: coat colour (BGR), body length in cm and the coat's name.
#[derive(Clone, Debug, PartialEq)]
pub struct Appearance {
    /// BGR coat colour.
    pub coat: [u8; 3],
    /// Body length in cm (`Simulation.appearance[a].size`).
    pub size: f64,
    /// Key of [`COAT_COLORS`].
    pub coat_name: String,
}

/// A [`Simulation`] plus what only the renderer reads: `Simulation.appearance` and `Simulation.food`.
#[derive(Clone, Debug)]
pub struct Scene {
    /// The keypoint trajectories and behaviors.
    pub sim: Simulation,
    /// One per animal.
    pub appearance: Vec<Appearance>,
    /// Food pellet positions on the floor, `(F, 2)` in cm.
    pub food: Vec<[f64; 2]>,
}

/// The BGR colour of a [`COAT_COLORS`] name.
pub fn coat_color(name: &str) -> Option<[u8; 3]> {
    COAT_COLORS.iter().find(|(n, _)| *n == name).map(|(_, c)| *c)
}

/// `simulate(n_animals, n_frames, species, fps, seed=rng, coats=coats)` with the appearance the
/// renderer needs: coats drawn without replacement while the cohort fits the palette (as
/// `rng.choice(names, size=n, replace=n > 6)`), two food pellets in the central 80 % of the arena.
pub fn simulate_scene(
    n_animals: usize,
    n_frames: usize,
    species: &str,
    rng: &mut Rng,
    coats: Option<&[&str]>,
) -> Result<Scene> {
    let Some(length) = body_length_cm(species) else {
        bail!("unknown species {species:?}: the simulator knows mouse, rat and quadruped");
    };
    let names: Vec<&str> = COAT_COLORS.iter().map(|(n, _)| *n).collect();
    let chosen: Vec<String> = match coats {
        Some(c) => {
            anyhow::ensure!(
                c.len() == n_animals,
                "{} coats for {n_animals} animals",
                c.len()
            );
            c.iter().map(|s| s.to_string()).collect()
        }
        None if n_animals > names.len() => (0..n_animals)
            .map(|_| rng.choose(&names).to_string())
            .collect(),
        None => rng
            .sample_without_replacement(&names, n_animals)
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    let arena = 5.0 * length;
    let food: Vec<[f64; 2]> = (0..2)
        .map(|_| {
            [
                rng.uniform(-0.4 * arena, 0.4 * arena),
                rng.uniform(-0.4 * arena, 0.4 * arena),
            ]
        })
        .collect();
    let sim = simulate(n_animals, n_frames, species, 30.0, None, rng);
    let mut appearance = Vec::with_capacity(n_animals);
    for (a, name) in chosen.iter().enumerate() {
        let Some(coat) = coat_color(name) else {
            bail!("unknown coat {name:?}: known are {names:?}");
        };
        appearance.push(Appearance {
            coat,
            size: sim.sizes[a],
            coat_name: name.clone(),
        });
    }
    Ok(Scene {
        sim,
        appearance,
        food,
    })
}

/// `random_camera(arena_cm, rng, size, elevation)`: elevation uniform in `elevation` (degrees),
/// azimuth uniform in `[-180, 180)`, distance `1.3..2.2` arenas and a fill of `0.75..1.1`.
pub fn random_camera(
    arena_cm: f64,
    rng: &mut Rng,
    size: (usize, usize),
    elevation: (f64, f64),
) -> Camera {
    let el = rng.uniform(elevation.0, elevation.1);
    let az = rng.uniform(-180.0, 180.0);
    let name = format!("rand_el{el:.0}_az{az:.0}");
    let distance = rng.uniform(1.3, 2.2);
    let fill = rng.uniform(0.75, 1.1);
    camera_from_view(arena_cm, el, az, size, distance, &name, fill)
}

/// `make_cameras(arena_cm, views, size)` over [`VIEW_PRESETS`] names.
pub fn make_cameras(arena_cm: f64, views: &[&str], size: (usize, usize)) -> Result<Vec<Camera>> {
    views
        .iter()
        .map(|v| {
            let Some((_, el, az)) = VIEW_PRESETS.iter().find(|(n, _, _)| n == v) else {
                bail!("unknown view {v:?}: choose top, side or angled");
            };
            Ok(camera_from_view(arena_cm, *el, *az, size, 1.6, v, 0.9))
        })
        .collect()
}

/// `Frame`: one rendered image and its annotations.
#[derive(Clone, Debug)]
pub struct Frame {
    /// `(H, W, 3)` 8-bit BGR.
    pub image: Bgr,
    /// `(N, K, 2)` pixels, projected even when hidden.
    pub keypoints: Vec<[f32; 2]>,
    /// `(N, K)`: the keypoint's pixel shows its own animal.
    pub visible: Vec<bool>,
    /// `(N, 4)` `xyxy` of the visible extent (zeros when absent).
    pub boxes: Vec<[f32; 4]>,
    /// `(N,)`: the animal is visible enough to annotate.
    pub present: Vec<bool>,
    /// `(N,)` animal indices.
    pub ids: Vec<i64>,
}

impl Frame {
    /// Animals in the frame.
    pub fn n_animals(&self) -> usize {
        self.present.len()
    }
}

/// `_lighter(c, f)`: mixes `f` of grey level 230 into a colour.
pub fn lighter(c: [u8; 3], f: f64) -> [u8; 3] {
    c.map(|v| (f64::from(v) * (1.0 - f) + 230.0 * f).clamp(0.0, 255.0) as u8)
}

/// `_draw_animal(canvas, mask, uv, px_len, coat, depth, size)`: one animal into an 8-bit canvas
/// (held as floats) and its mask.
///
/// `uv` is `(K, 2)` pixels; with per-keypoint camera `depth` the limbs are depth-ordered (far legs
/// behind the trunk, shaded by up to 40 %, near legs in front). Coordinates go through cv2's
/// `shift=2` fixed point, i.e. are rounded to quarter pixels.
pub fn draw_animal(
    canvas: &mut ImageF,
    mask: &mut [u8],
    uv: &[[f64; 2]],
    px_len: f64,
    coat: [u8; 3],
    depth: Option<&[f64]>,
    size: f64,
) {
    let mut p = Painter { canvas, mask, uv };
    let width = 2i64.max((0.26 * px_len) as i64);
    let limb = 1i64.max((0.06 * px_len) as i64);
    let tail = 1i64.max((0.05 * px_len) as i64);

    let mut far_side = "left";
    let mut shade = 0.0;
    if let Some(depth) = depth {
        let lat = |side: &str| {
            (depth[index(&format!("{side}_shoulder"))] + depth[index(&format!("{side}_hip"))]) / 2.0
        };
        let (l, r) = (lat("left"), lat("right"));
        far_side = if l > r { "left" } else { "right" };
        shade = 0.4 * ((l - r).abs() / (0.18 * size)).clamp(0.0, 1.0);
    }
    let near_side = if far_side == "left" { "right" } else { "left" };
    let legs = |p: &mut Painter, side: &str, hind: bool, front: bool, factor: f64| {
        let c = coat.map(|v| (f64::from(v) * factor) as u8);
        let sk = SKIN.map(|v| (f64::from(v) * factor) as u8);
        for (limb_name, on) in [("hind", hind), ("front", front)] {
            if !on {
                continue;
            }
            let top = if limb_name == "hind" {
                format!("{side}_hip")
            } else {
                format!("{side}_shoulder")
            };
            let knee = index(&format!("{side}_{limb_name}_knee"));
            let paw = index(&format!("{side}_{limb_name}_paw"));
            p.line(index(&top), knee, c, limb);
            p.line(knee, paw, c, limb);
            p.circle(paw, 0.035 * px_len, sk);
        }
    };

    p.line(index("tail_base"), index("tail_mid"), SKIN, tail);
    p.line(index("tail_mid"), index("tail_end"), SKIN, 1.max(tail - 1));
    legs(&mut p, far_side, true, depth.is_some(), 1.0 - shade);
    legs(&mut p, near_side, depth.is_none(), false, 1.0);
    let spine = ["tail_base", "spine_lower", "spine_mid", "spine_upper", "neck"];
    for pair in spine.windows(2) {
        p.line(index(pair[0]), index(pair[1]), coat, width);
    }
    for s in &spine[1..] {
        p.circle(index(s), width as f64 / 2.0, coat);
    }
    // `cv2.convexHull(np.round(uv[trunk]).astype(np.int32))`, then `fillConvexPoly`.
    let trunk: Vec<[f64; 2]> = ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]
        .iter()
        .map(|n| {
            let q = uv[index(n)];
            [q[0].round_ties_even(), q[1].round_ties_even()]
        })
        .collect();
    p.fill(
        &Shape::Polygon {
            points: convex_hull(&trunk),
        },
        coat,
    );
    if depth.is_none() {
        legs(&mut p, "left", false, true, 1.0);
        legs(&mut p, "right", false, true, 1.0);
    } else {
        legs(&mut p, near_side, true, true, 1.0);
    }
    p.line(index("neck"), index("head_center"), coat, (width as f64 * 0.9) as i64);
    p.line(index("head_center"), index("nose"), coat, (width as f64 * 0.6) as i64);
    p.circle(index("head_center"), width as f64 * 0.45, coat);
    p.circle(index("nose"), 0.04 * px_len, SKIN);
    for side in ["left", "right"] {
        p.circle(index(&format!("{side}_ear")), 0.075 * px_len, lighter(SKIN, 0.1));
        p.circle(index(&format!("{side}_eye")), 0.028 * px_len, [10, 10, 10]);
    }
}

/// The canvas/mask pair `_draw_animal`'s `line` and `circle` helpers draw into.
struct Painter<'a> {
    canvas: &'a mut ImageF,
    mask: &'a mut [u8],
    uv: &'a [[f64; 2]],
}

impl Painter<'_> {
    /// `tuple(int(round(v)) for v in uv[i] * 4)` read back with `shift=2`: quarter pixels.
    fn pt(&self, i: usize) -> [f64; 2] {
        [
            (self.uv[i][0] * 4.0).round_ties_even() / 4.0,
            (self.uv[i][1] * 4.0).round_ties_even() / 4.0,
        ]
    }

    /// `LINE_AA` into the canvas, `LINE_8` into the mask.
    fn fill(&mut self, shape: &Shape, color: [u8; 3]) {
        draw_aa(self.canvas, shape, &color.map(f32::from));
        let (w, h) = (self.canvas.width, self.canvas.height);
        draw_hard(self.mask, w, h, shape, 255);
    }

    fn line(&mut self, a: usize, b: usize, color: [u8; 3], th: i64) {
        let shape = Shape::line(self.pt(a), self.pt(b), th);
        self.fill(&shape, color);
    }

    fn circle(&mut self, a: usize, r: f64, color: [u8; 3]) {
        let r = 1i64.max(r as i64);
        let shape = Shape::Disk {
            center: self.pt(a),
            radius: r as f64,
        };
        self.fill(&shape, color);
    }
}

/// `_background(cam, sim, rng)`: a wall colour, the projected floor in a brighter colour,
/// low-frequency noise, 2 % speckle and the food pellets.
pub fn background(cam: &Camera, scene: &Scene, rng: &mut Rng) -> ImageF {
    let (w, h) = (cam.width, cam.height);
    let wall: Vec<f32> = (0..3).map(|_| rng.uniform(40.0, 110.0) as f32).collect();
    let mut img = ImageF::new(w, h, 3, 0.0);
    for px in img.data.chunks_exact_mut(3) {
        px.copy_from_slice(&wall);
    }
    let half = scene.sim.arena_cm / 2.0;
    let floor_poly: Vec<[f64; 2]> = [
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
    ]
    .iter()
    .map(|p| {
        let uv = cam.project(*p);
        [uv[0].round_ties_even(), uv[1].round_ties_even()]
    })
    .collect();
    let base: Vec<f64> = (0..3).map(|_| rng.uniform(120.0, 200.0)).collect();
    let floor: Vec<f32> = base
        .iter()
        .map(|b| (b * rng.uniform(0.85, 1.15)) as f32)
        .collect();
    let mut fmask = vec![0u8; w * h];
    draw_hard(
        &mut fmask,
        w,
        h,
        &Shape::Polygon { points: floor_poly },
        255,
    );
    for (i, m) in fmask.iter().enumerate() {
        if *m > 0 {
            img.data[i * 3..i * 3 + 3].copy_from_slice(&floor);
        }
    }
    // `cv2.resize(rng.normal(0, 12, (max(2, h // 32), max(2, w // 32), 1)), (w, h))`.
    let (lh, lw) = ((h / 32).max(2), (w / 32).max(2));
    let low: Vec<f32> = (0..lh * lw).map(|_| rng.normal(0.0, 12.0) as f32).collect();
    let low = ImageF::from_data(lw, lh, 1, low).expect("sized above");
    let low = resize_linear(&low, w, h);
    for (i, v) in low.data.iter().enumerate() {
        for c in 0..3 {
            img.data[i * 3 + c] += v;
        }
    }
    let speckle: Vec<bool> = (0..w * h).map(|_| rng.random() < 0.02).collect();
    let factor = rng.uniform(0.7, 1.2) as f32;
    for (i, s) in speckle.iter().enumerate() {
        if *s {
            for c in 0..3 {
                img.data[i * 3 + c] *= factor;
            }
        }
    }
    let center = cam.center();
    let dist = (center[0] * center[0] + center[1] * center[1] + center[2] * center[2])
        .sqrt()
        .max(1e-6);
    for f in &scene.food {
        let uv = cam.project([f[0], f[1], 0.0]);
        let r = 1i64.max((0.06 * scene.appearance[0].size * cam.focal / dist) as i64);
        // `tuple(int(v) for v in uv)`: truncation towards zero; LINE_AA is a hard fill on floats.
        let shape = Shape::Disk {
            center: [uv[0].trunc(), uv[1].trunc()],
            radius: r as f64,
        };
        draw_hard_f(&mut img, &shape, &[30.0, 60.0, 100.0]);
    }
    img
}

/// `render(sim, cam, t, rng, background, noise)`: frame `t` of `scene` from `cam`.
pub fn render(
    scene: &Scene,
    cam: &Camera,
    t: usize,
    rng: &mut Rng,
    background_img: Option<&ImageF>,
    noise: bool,
) -> Frame {
    let (w, h) = (cam.width, cam.height);
    let sim = &scene.sim;
    let mut img = match background_img {
        Some(bg) => bg.clone(),
        None => background(cam, scene, rng),
    };
    let n = sim.n_animals;
    let mut uvs: Vec<Vec<[f64; 2]>> = Vec::with_capacity(n);
    let mut depths: Vec<Vec<f64>> = Vec::with_capacity(n);
    for a in 0..n {
        let pts = sim.kpts_at(a, t);
        let mut uv = Vec::with_capacity(NUM_KEYPOINTS);
        let mut dz = Vec::with_capacity(NUM_KEYPOINTS);
        for p in &pts {
            uv.push(cam.project(*p));
            dz.push(cam.r[2][0] * p[0] + cam.r[2][1] * p[1] + cam.r[2][2] * p[2] + cam.t[2]);
        }
        uvs.push(uv);
        depths.push(dz);
    }
    let mean_depth: Vec<f64> = depths
        .iter()
        .map(|d| d.iter().sum::<f64>() / d.len() as f64)
        .collect();
    let mut label = vec![-1i32; w * h];
    // `np.argsort(-depths.mean(1))`: far to near.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| mean_depth[*b].total_cmp(&mean_depth[*a]));
    for &a in &order {
        if depths[a].iter().any(|d| *d <= 1e-3) {
            continue;
        }
        let app = &scene.appearance[a];
        let px_len = cam.focal * app.size / mean_depth[a];
        let mut canvas = ImageF::new(w, h, 3, 0.0);
        let mut mask = vec![0u8; w * h];
        let coat = app
            .coat
            .map(|c| (f64::from(c) * rng.uniform(0.95, 1.05)).clamp(0.0, 255.0) as u8);
        draw_animal(
            &mut canvas,
            &mut mask,
            &uvs[a],
            px_len,
            coat,
            Some(&depths[a]),
            app.size,
        );
        for (i, m) in mask.iter().enumerate() {
            if *m > 0 {
                img.data[i * 3..i * 3 + 3].copy_from_slice(&canvas.data[i * 3..i * 3 + 3]);
                label[i] = a as i32;
            }
        }
    }
    let mut visible = vec![false; n * NUM_KEYPOINTS];
    let mut boxes = vec![[0.0f32; 4]; n];
    let mut present = vec![false; n];
    for a in 0..n {
        let inside = cam.in_image(&uvs[a]);
        for k in 0..NUM_KEYPOINTS {
            let xi = (uvs[a][k][0].round_ties_even()).clamp(0.0, w as f64 - 1.0) as usize;
            let yi = (uvs[a][k][1].round_ties_even()).clamp(0.0, h as f64 - 1.0) as usize;
            visible[a * NUM_KEYPOINTS + k] =
                inside[k] && depths[a][k] > 0.0 && label[yi * w + xi] == a as i32;
        }
        let (mut x0, mut y0, mut x1, mut y1, mut count) = (usize::MAX, usize::MAX, 0, 0, 0usize);
        for (i, l) in label.iter().enumerate() {
            if *l == a as i32 {
                let (x, y) = (i % w, i / w);
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                count += 1;
            }
        }
        let full_px = cam.focal * scene.appearance[a].size / mean_depth[a].max(1e-3);
        let full_area = (full_px * full_px * 0.15).max(1.0);
        let n_vis = visible[a * NUM_KEYPOINTS..(a + 1) * NUM_KEYPOINTS]
            .iter()
            .filter(|v| **v)
            .count();
        if count as f64 >= (0.1 * full_area).max(20.0) && n_vis >= 3 {
            boxes[a] = [x0 as f32, y0 as f32, (x1 + 1) as f32, (y1 + 1) as f32];
            present[a] = true;
        }
    }
    if noise {
        let gain = rng.uniform(0.8, 1.2) as f32;
        let offset = rng.uniform(-20.0, 20.0) as f32;
        for v in img.data.iter_mut() {
            *v = *v * gain + offset;
        }
        let sigma = rng.uniform(1.0, 6.0);
        for v in img.data.iter_mut() {
            *v += rng.normal(0.0, sigma) as f32;
        }
        if rng.random() < 0.3 {
            img = gaussian_blur(&img, 3, 0.0);
        }
    }
    let keypoints = uvs
        .iter()
        .flat_map(|uv| uv.iter().map(|p| [p[0] as f32, p[1] as f32]))
        .collect();
    Frame {
        image: img.to_bgr_trunc(),
        keypoints,
        visible,
        boxes,
        present,
        ids: (0..n as i64).collect(),
    }
}

/// `render_video(sim, cam, seed)`: every frame over one fixed background, as from a static camera.
pub fn render_video(scene: &Scene, cam: &Camera, seed: u64) -> Vec<Frame> {
    let mut rng = Rng::new(seed);
    let bg = background(cam, scene, &mut rng);
    (0..scene.sim.n_frames)
        .map(|t| render(scene, cam, t, &mut rng, Some(&bg), true))
        .collect()
}

/// The two-mouse demo clip of `animalpose predict --synthetic` (`api.py:predict`):
/// `simulate(2, n_frames, "mouse", seed=7, coats=["black", "white"])` rendered from the `angled`
/// preset at 320x256 by `render_video(sim, cam)`.
pub fn demo_clip(n_frames: usize) -> Result<Vec<Frame>> {
    let mut rng = Rng::new(7);
    let scene = simulate_scene(2, n_frames, "mouse", &mut rng, Some(&["black", "white"]))?;
    let cam = &make_cameras(scene.sim.arena_cm, &["angled"], (320, 256))?[0];
    Ok(render_video(&scene, cam, 0))
}

/// Writes [`demo_clip`] into `dir` as grey PNG frames (`frame_00000.png`, ...), converted the way
/// the pipeline's `_gray` converts a BGR frame (`cv2.COLOR_BGR2GRAY`, 8 bits), so that
/// [`crate::data::images::read_frames`] reads back exactly what the pipeline would see.
pub fn write_demo_frames(dir: &std::path::Path, n_frames: usize) -> Result<usize> {
    use crate::algos::image::ToGray;
    std::fs::create_dir_all(dir)?;
    let frames = demo_clip(n_frames)?;
    for (t, f) in frames.iter().enumerate() {
        let gray = f.image.to_gray();
        let pixels = gray.pixels.iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect();
        let img = image::GrayImage::from_raw(gray.width, gray.height, pixels)
            .ok_or_else(|| anyhow::anyhow!("frame {t} has the wrong size"))?;
        img.save(dir.join(format!("frame_{t:05}.png")))?;
    }
    Ok(frames.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> Scene {
        let mut rng = Rng::new(3);
        simulate_scene(2, 40, "mouse", &mut rng, Some(&["black", "white"])).unwrap()
    }

    /// `_draw_animal` against OpenCV: the same uv / depth (the template seen from above) drawn by
    /// `data/synthetic.py` gives these mask areas, mean colours and counts of bright (skin) pixels.
    #[test]
    fn draw_animal_matches_cv2_areas() {
        use crate::data::synthetic::TEMPLATE;
        let cases = [
            (60.0, 1086.0, [67.2, 68.21, 80.07], 101.0),
            (25.0, 174.0, [62.01, 62.82, 73.4], 3.0),
        ];
        for (px, area, color, bright) in cases {
            let uv: Vec<[f64; 2]> = TEMPLATE
                .iter()
                .map(|t| [160.0 + px * t[0], 128.0 - px * t[1] + 0.3 * px * t[2]])
                .collect();
            let depth: Vec<f64> = TEMPLATE.iter().map(|t| 100.0 - px * t[2] + 10.0 * t[1]).collect();
            let mut canvas = ImageF::new(320, 256, 3, 0.0);
            let mut mask = vec![0u8; 320 * 256];
            draw_animal(&mut canvas, &mut mask, &uv, px, [40, 38, 36], Some(&depth), 8.5);
            let on: Vec<usize> = (0..mask.len()).filter(|i| mask[*i] > 0).collect();
            let n = on.len() as f64;
            assert!((n - area).abs() <= 0.05 * area, "px {px}: area {n} vs cv2 {area}");
            for (c, want) in color.iter().enumerate() {
                let mean = on.iter().map(|i| f64::from(canvas.data[i * 3 + c])).sum::<f64>() / n;
                assert!((mean - want).abs() < 6.0, "px {px} channel {c}: {mean} vs {want}");
            }
            let skin = on
                .iter()
                .filter(|i| (0..3).all(|c| canvas.data[*i * 3 + c] > 150.0))
                .count() as f64;
            assert!((skin - bright).abs() <= 0.35 * bright + 3.0, "px {px}: {skin} vs {bright}");
        }
    }

    #[test]
    fn frames_have_the_python_shapes() {
        let scene = scene();
        let cams = make_cameras(scene.sim.arena_cm, &["top", "side", "angled"], (320, 256)).unwrap();
        for cam in &cams {
            let mut rng = Rng::new(1);
            let f = render(&scene, cam, 30, &mut rng, None, true);
            assert_eq!((f.image.width, f.image.height), (320, 256));
            assert_eq!(f.image.pixels.len(), 320 * 256 * 3);
            assert_eq!(f.keypoints.len(), 2 * NUM_KEYPOINTS);
            assert_eq!(f.visible.len(), 2 * NUM_KEYPOINTS);
            assert_eq!(f.boxes.len(), 2);
            assert_eq!(f.ids, vec![0, 1]);
            for a in 0..2 {
                if f.present[a] {
                    let b = f.boxes[a];
                    assert!(b[0] < b[2] && b[1] < b[3]);
                    // Every visible keypoint lies inside (or on the edge of) its visible box.
                    for k in 0..NUM_KEYPOINTS {
                        if f.visible[a * NUM_KEYPOINTS + k] {
                            let p = f.keypoints[a * NUM_KEYPOINTS + k];
                            assert!(p[0] >= b[0] - 1.0 && p[0] <= b[2] + 1.0, "{p:?} {b:?}");
                            assert!(p[1] >= b[1] - 1.0 && p[1] <= b[3] + 1.0, "{p:?} {b:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn top_view_animals_are_present_and_mostly_visible() {
        let scene = scene();
        let cam = &make_cameras(scene.sim.arena_cm, &["top"], (320, 256)).unwrap()[0];
        let mut rng = Rng::new(5);
        let f = render(&scene, cam, 20, &mut rng, None, false);
        assert!(f.present.iter().all(|p| *p), "{:?}", f.present);
        let vis = f.visible.iter().filter(|v| **v).count();
        assert!(vis >= 2 * 12, "{vis} visible keypoints");
    }

    #[test]
    fn a_nearer_animal_hides_a_farther_one() {
        // Two animals stacked exactly on top of each other: in a top view the upper one
        // (raised by 1 cm, so nearer to the camera) owns every pixel.
        let mut scene = scene();
        let k3 = NUM_KEYPOINTS * 3;
        let n_frames = scene.sim.n_frames;
        for t in 0..n_frames {
            for k in 0..k3 {
                let v = scene.sim.kpts3d[t * k3 + k];
                scene.sim.kpts3d[(n_frames + t) * k3 + k] = v + if k % 3 == 2 { 1.0 } else { 0.0 };
            }
        }
        scene.appearance[1].size = scene.appearance[0].size;
        let cam = &make_cameras(scene.sim.arena_cm, &["top"], (320, 256)).unwrap()[0];
        let mut rng = Rng::new(9);
        let f = render(&scene, cam, 10, &mut rng, None, false);
        assert!(!f.present[0]);
        assert!(f.present[1]);
        assert!(f.visible[..NUM_KEYPOINTS].iter().all(|v| !*v));
        // The visible animal is the white one.
        let b = f.boxes[1];
        let (cx, cy) = (((b[0] + b[2]) / 2.0) as u32, ((b[1] + b[3]) / 2.0) as u32);
        let mut bright = 0;
        for dy in 0..3 {
            for dx in 0..3 {
                let px = f.image.pixel(cx + dx - 1, cy + dy - 1);
                bright += u32::from(px[1] > 150);
            }
        }
        assert!(bright >= 5, "the near animal should be white");
    }

    #[test]
    fn render_video_keeps_one_background() {
        let mut rng = Rng::new(3);
        let scene = simulate_scene(1, 3, "rat", &mut rng, None).unwrap();
        assert_eq!(scene.appearance.len(), 1);
        let cam = &make_cameras(scene.sim.arena_cm, &["angled"], (96, 64)).unwrap()[0];
        let frames = render_video(&scene, cam, 0);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].image.pixels.len(), 96 * 64 * 3);
    }
}
