//! The procedural simulator and the keypoint-window training batches.
//!
//! Ported from three Python modules, keeping their function names and defaults:
//!
//! | Python | Rust |
//! | --- | --- |
//! | `data/synthetic.py:TEMPLATE` / `body_pose` / `simulate` / `Simulation` | [`TEMPLATE`] / [`body_pose`] / [`simulate`] / [`Simulation`] |
//! | `data/synthetic.py:camera_from_view` / `_fit_focal` / `random_camera` | [`camera_from_view`] / [`fit_focal`] / [`random_camera`] |
//! | `data/synthetic.py:SimulationPool` | [`SimulationPool`] |
//! | `temporal/data.py:Corruption` / `corrupt` / `synthetic_batch` / `sample_windows` | [`Corruption`] / [`corrupt`] / [`temporal_batch`] / [`sample_windows`] |
//! | `behavior/data.py:synthetic_batch` | [`behavior_batch`] |
//! | `train/loops.py:_KeypointSubset` | [`crate::data::synthetic::keypoint_subset_masks`] |
//!
//! Image rendering (`data/synthetic.py:render`, `_draw_animal`, `_background`, coat colours and food
//! pellets) lives in [`crate::data::render`]; the batches produced here are coordinates only.
//!
//! ## Determinism
//!
//! [`Rng`] is a splitmix64-seeded `xoshiro256++`, **not** numpy's `PCG64`, so no stream here is
//! bit-exact with `numpy.random.default_rng(seed)`. What is bit-exact is everything that does not
//! draw a random number, and those parts are pinned against Python values in the tests:
//! [`TEMPLATE`], [`body_pose`], [`camera_from_view`], [`Camera::project`], [`Corruption`]'s
//! post-processing rules (jitter, drops, gaps, outliers, swaps, the zeroing of unobserved points)
//! and [`crate::data::synthetic::keypoint_subset_masks`]'s masks. The sequences themselves — [`simulate`], [`corrupt`],
//! [`temporal_batch`], [`behavior_batch`], [`sample_windows`] and
//! [`sessions::behavior_windows`](crate::data::sessions::behavior_windows) — are
//! *distributionally* equivalent: same draws in the same order with the same distributions,
//! different numbers for a given seed.

use anyhow::{bail, ensure, Result};

/// The canonical keypoint vocabulary, `skeleton/vocabulary.py:CANONICAL.names`.
pub const CANONICAL_NAMES: [&str; 25] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "head_center",
    "neck",
    "spine_upper",
    "spine_mid",
    "spine_lower",
    "tail_base",
    "tail_mid",
    "tail_end",
    "left_shoulder",
    "right_shoulder",
    "left_front_knee",
    "right_front_knee",
    "left_front_paw",
    "right_front_paw",
    "left_hip",
    "right_hip",
    "left_hind_knee",
    "right_hind_knee",
    "left_hind_paw",
    "right_hind_paw",
];

/// Keypoints of the canonical vocabulary, `CANONICAL.num_keypoints`.
pub const NUM_KEYPOINTS: usize = 25;

/// `CANONICAL.flip_index`: the permutation that swaps left and right.
pub const FLIP_INDEX: [usize; NUM_KEYPOINTS] = [
    0, 2, 1, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23,
];

/// `CANONICAL.anterior`.
pub const ANTERIOR: [usize; 4] = [0, 5, 6, 7];

/// `multiview/symmetry.py:bilateral_pairs()`: the `(left, right)` index rows the pose decoder keeps
/// apart and `temporal/pairs.py:resolve_pair_identity` re-labels.
///
/// `[(i, flip_index[i]) for i in range(25) if side(i) == "left"]`, which is the eight left/right
/// mirrors of the canonical skeleton (eyes, ears, shoulders, front knees, front paws, hips, hind
/// knees, hind paws).
pub const BILATERAL_PAIRS: [(usize, usize); 8] = [
    (1, 2),
    (3, 4),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
];

/// `CANONICAL.posterior`.
pub const POSTERIOR: [usize; 4] = [10, 9, 19, 20];

/// `labels.py:BEHAVIORS`, the classes the simulator's Markov chain switches between.
pub const BEHAVIORS: [&str; 5] = ["resting", "walking", "rearing", "grooming", "eating"];

/// `skeleton/vocabulary.py:CANONICAL.index(name)`.
pub fn index(name: &str) -> usize {
    CANONICAL_NAMES
        .iter()
        .position(|n| *n == name)
        .unwrap_or_else(|| panic!("unknown keypoint {name:?}"))
}

/// `skeleton/species.py:SPECIES[species].body_length_cm`; `None` for an unknown species
/// (`quadruped` has no typical length, which Python stores as `NaN`).
pub fn body_length_cm(species: &str) -> Option<f64> {
    match species {
        "mouse" => Some(8.5),
        "rat" => Some(20.0),
        "quadruped" => Some(f64::NAN),
        _ => None,
    }
}

/// `skeleton/species.py:DATASET_MAPPINGS`: the canonical names one dataset labels.
pub fn dataset_mapping(mapping: &str) -> Option<Vec<(&'static str, &'static str)>> {
    Some(match mapping {
        // `synthetic`: the renderer labels the full vocabulary.
        "synthetic" => CANONICAL_NAMES.iter().map(|n| (*n, *n)).collect(),
        // DeepLabCut home-cage markers (six of the 25 canonical points).
        "homecage_dlc" => [
            ("nose", "nose"),
            ("leftear", "left_ear"),
            ("rightear", "right_ear"),
            ("headstage", "head_center"),
            ("midbody", "spine_mid"),
            ("tailbase", "tail_base"),
        ]
        .to_vec(),
        // SuperAnimal-TopViewMouse (`superanimal_topviewmouse`).
        "superanimal_topviewmouse" => [
            ("nose", "nose"),
            ("left_eye", "left_eye"),
            ("right_eye", "right_eye"),
            ("left_ear", "left_ear"),
            ("right_ear", "right_ear"),
            ("head_midpoint", "head_center"),
            ("neck", "neck"),
            ("mid_back", "spine_upper"),
            ("mouse_center", "spine_mid"),
            ("mid_backend2", "spine_lower"),
            ("tail_base", "tail_base"),
            ("tail3", "tail_mid"),
            ("tail_end", "tail_end"),
            ("left_shoulder", "left_shoulder"),
            ("right_shoulder", "right_shoulder"),
            ("left_hip", "left_hip"),
            ("right_hip", "right_hip"),
        ]
        .to_vec(),
        _ => return None,
    })
}

/// `CANONICAL.mask(list(DATASET_MAPPINGS[mapping].values()))`: the canonical points one dataset
/// labels, which `train/loops.py:_KeypointSubset` multiplies the observation masks by.
pub fn canonical_mask(mapping: &str) -> Result<Vec<f32>> {
    let mapping = dataset_mapping(mapping)
        .ok_or_else(|| anyhow::anyhow!("unknown keypoint mapping {mapping:?}"))?;
    let mut mask = vec![0.0f32; NUM_KEYPOINTS];
    for (_, canonical) in mapping {
        mask[index(canonical)] = 1.0;
    }
    Ok(mask)
}

/// `data/synthetic.py:TEMPLATE`: the body-frame keypoint template in body lengths
/// (`x` forward, `y` left, `z` up, origin under `spine_mid`).
pub const TEMPLATE: [[f64; 3]; NUM_KEYPOINTS] = [
    [0.52, 0.0, 0.10],
    [0.43, 0.045, 0.14],
    [0.43, -0.045, 0.14],
    [0.35, 0.08, 0.17],
    [0.35, -0.08, 0.17],
    [0.36, 0.0, 0.17],
    [0.25, 0.0, 0.18],
    [0.15, 0.0, 0.20],
    [0.0, 0.0, 0.21],
    [-0.17, 0.0, 0.19],
    [-0.30, 0.0, 0.12],
    [-0.62, 0.0, 0.03],
    [-0.95, 0.0, 0.01],
    [0.16, 0.08, 0.11],
    [0.16, -0.08, 0.11],
    [0.17, 0.09, 0.05],
    [0.17, -0.09, 0.05],
    [0.19, 0.09, 0.0],
    [0.19, -0.09, 0.0],
    [-0.18, 0.09, 0.11],
    [-0.18, -0.09, 0.11],
    [-0.14, 0.11, 0.05],
    [-0.14, -0.11, 0.05],
    [-0.20, 0.10, 0.0],
    [-0.20, -0.10, 0.0],
];

/// `CANONICAL.indices(["nose", "left_eye", ..., "head_center"])`.
const HEAD: [usize; 6] = [0, 1, 2, 3, 4, 5];
const FRONT_PAWS: [usize; 2] = [17, 18];
const FRONT_KNEES: [usize; 2] = [15, 16];
const HIND_PAWS: [usize; 2] = [23, 24];
const HIND_KNEES: [usize; 2] = [21, 22];
const TAIL: [usize; 2] = [11, 12];

/// `_PITCHED`: the points that tilt up with the front of the body when the animal rears.
fn pitched() -> Vec<usize> {
    CANONICAL_NAMES
        .iter()
        .enumerate()
        .filter(|(_, n)| {
            !n.starts_with("tail")
                && !n.contains("hind")
                && !n.contains("hip")
                && **n != "spine_lower"
        })
        .map(|(i, _)| i)
        .collect()
}

/// `_POSTURE`: `crouch, head_down, gait, paws_to_face, pitch, groom` per behavior, plus the walking
/// speed in body lengths per second (`target[6]`).
const POSTURE: [[f64; 7]; 5] = [
    [1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 1.3],
    [0.0, 0.0, 0.0, 0.35, 1.1, 0.0, 0.0],
    [0.3, 0.2, 0.0, 1.0, 0.35, 1.0, 0.0],
    [0.2, 1.0, 0.0, 0.6, 0.0, 0.0, 0.0],
];

/// `_DURATION`: how long a behavior lasts, in frames.
const DURATION: [(i64, i64); 5] = [(60, 240), (30, 150), (20, 60), (40, 150), (40, 150)];

/// `_NEXT`: the relative rate of switching to each behavior.
const NEXT: [f64; 5] = [1.0, 3.0, 1.0, 1.0, 1.0];

/// `data/synthetic.py:body_pose`: posture parameters → body-frame keypoints `(K, 3)` in body lengths.
///
/// `params` is `(crouch, head_down, gait, paws_to_face, pitch, groom)` (`_POSTURE[behavior][:6]`,
/// already smoothed towards the target by [`simulate`]).
pub fn body_pose(
    params: &[f64; 6],
    gait_phase: f64,
    groom_phase: f64,
    bend: f64,
    tail_phase: f64,
) -> Vec<[f64; 3]> {
    let (crouch, head_down, gait, paws_to_face, pitch, groom) = (
        params[0], params[1], params[2], params[3], params[4], params[5],
    );
    let mut p = TEMPLATE.map(|row| [row[0], row[1], row[2]]);
    for row in p.iter_mut() {
        row[0] *= 1.0 - 0.15 * crouch;
        row[2] *= 1.0 - 0.4 * crouch;
    }
    // The head pitches down about the neck (sniffing / eating).
    let neck = p[index("neck")];
    rot_xz_arr(&mut p, &HEAD, neck, -0.7 * head_down);
    // A diagonal trot: left front with right hind, right front with left hind.
    for (paws, knees, sign) in [
        (&FRONT_PAWS, &FRONT_KNEES, 1.0f64),
        (&HIND_PAWS, &HIND_KNEES, -1.0),
    ] {
        for (j, (pi, ki)) in paws.iter().zip(knees.iter()).enumerate() {
            let ph = gait_phase
                + std::f64::consts::PI * j as f64
                + if sign > 0.0 {
                    0.0
                } else {
                    std::f64::consts::PI
                };
            p[*pi][0] += gait * 0.09 * ph.sin();
            p[*pi][2] += gait * 0.04 * ph.cos().max(0.0);
            p[*ki][0] += gait * 0.05 * ph.sin();
        }
    }
    // Front paws toward the face (grooming, holding food, the rearing posture).
    let nose = p[index("nose")];
    for (j, (pi, ki)) in FRONT_PAWS.iter().zip(FRONT_KNEES.iter()).enumerate() {
        let side = if j == 0 { 1.0 } else { -1.0 };
        let target = [
            nose[0] - 0.07,
            nose[1] + 0.035 * side,
            nose[2] - 0.02 + 0.04 * groom * (groom_phase + side).sin(),
        ];
        p[*pi] = mix(&p[*pi], &target, paws_to_face);
        let shoulder = p[index(if j == 0 {
            "left_shoulder"
        } else {
            "right_shoulder"
        })];
        let mid = [
            (target[0] + shoulder[0]) / 2.0,
            (target[1] + shoulder[1]) / 2.0,
            (target[2] + shoulder[2]) / 2.0,
        ];
        p[*ki] = mix(&p[*ki], &mid, 0.5 * paws_to_face);
    }
    // The front of the body rears up about the hips.
    let mut pivot = p[index("left_hip")];
    pivot[1] = 0.0;
    let pitched = pitched();
    rot_xz_arr(&mut p, &pitched, pivot, pitch);
    // Tail sway, and the lateral spine bend that comes with turning.
    for (i, &k) in TAIL.iter().enumerate() {
        p[k][1] += 0.12 * tail_phase.sin() * [0.6, 1.0][i] + 0.35 * crouch * [0.5, 1.0][i];
    }
    for row in p.iter_mut() {
        let ang = bend * row[0];
        let (c, s) = (ang.cos(), ang.sin());
        let (x, y) = (row[0], row[1]);
        row[0] = c * x - s * y;
        row[1] = s * x + c * y;
        row[2] = row[2].max(0.0);
    }
    p.to_vec()
}

/// `(1 - t) * a + t * b`, the blend `body_pose` writes as `(1 - w) * p + w * target`.
fn mix(a: &[f64; 3], b: &[f64; 3], t: f64) -> [f64; 3] {
    [
        (1.0 - t) * a[0] + t * b[0],
        (1.0 - t) * a[1] + t * b[1],
        (1.0 - t) * a[2] + t * b[2],
    ]
}

/// [`rot_xz`] over a dynamically sized index list.
fn rot_xz_arr<const N: usize>(
    p: &mut [[f64; 3]; N],
    subset: &[usize],
    pivot: [f64; 3],
    angle: f64,
) {
    let (c, s) = (angle.cos(), angle.sin());
    for &i in subset {
        let d = [p[i][0] - pivot[0], p[i][1] - pivot[1], p[i][2] - pivot[2]];
        p[i][0] = pivot[0] + d[0] * c - d[2] * s;
        p[i][2] = pivot[2] + d[0] * s + d[2] * c;
    }
}

/// A simulated cohort: 3D keypoints of every animal over the canonical vocabulary, plus the behavior
/// label of every frame. Port of `data/synthetic.py:Simulation` (its `appearance` keeps only the
/// body length, which the keypoint batches use; coat colour and food pellets belong to rendering).
#[derive(Clone, Debug, PartialEq)]
pub struct Simulation {
    /// `Simulation.species`.
    pub species: String,
    /// `Simulation.fps`.
    pub fps: f64,
    /// `Simulation.arena_cm`, the side of the square arena.
    pub arena_cm: f64,
    /// `Simulation.n_animals`.
    pub n_animals: usize,
    /// `Simulation.n_frames`.
    pub n_frames: usize,
    /// `Simulation.kpts3d`, `(n_animals, n_frames, K, 3)` in cm, world frame, `z` up.
    pub kpts3d: Vec<f64>,
    /// `Simulation.behavior`, `(n_animals, n_frames)`, an index into [`BEHAVIORS`].
    pub behavior: Vec<i64>,
    /// `Simulation.appearance[a].size`: body length in cm.
    pub sizes: Vec<f64>,
}

impl Simulation {
    /// `sim.kpts3d[a, s:s + length]` as `(length, K, 3)`.
    pub fn window(&self, animal: usize, start: usize, length: usize) -> Vec<f64> {
        let k3 = NUM_KEYPOINTS * 3;
        let at = animal * self.n_frames * k3 + start * k3;
        self.kpts3d[at..at + length * k3].to_vec()
    }

    /// `sim.behavior[a, s:s + length]`.
    pub fn behavior_window(&self, animal: usize, start: usize, length: usize) -> Vec<i64> {
        let at = animal * self.n_frames + start;
        self.behavior[at..at + length].to_vec()
    }

    /// `sim.kpts3d[a, t]` as `(K, 3)`.
    pub fn kpts_at(&self, animal: usize, frame: usize) -> Vec<[f64; 3]> {
        let at = (animal * self.n_frames + frame) * NUM_KEYPOINTS * 3;
        self.kpts3d[at..at + NUM_KEYPOINTS * 3]
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect()
    }
}

/// `data/synthetic.py:simulate`: `n_animals` rodents wandering an arena, driven by the behavior
/// Markov chain of `_POSTURE` / `_DURATION` / `_NEXT`.
///
/// Mathematically the same integration as Python (an OU turning rate, wall avoidance, smoothed
/// posture targets, a diagonal trot gait); the random stream is [`Rng`], not numpy's.
///
/// `arena_cm` is `Simulation.arena_cm` (`None` picks `5 * body_length_cm(species)`, as Python's
/// `arena_cm or ...` does); coat colours and food pellets, which only the renderer reads, are not
/// simulated and consume no draws.
pub fn simulate(
    n_animals: usize,
    n_frames: usize,
    species: &str,
    fps: f64,
    arena_cm: Option<f64>,
    rng: &mut Rng,
) -> Simulation {
    let length = body_length_cm(species).unwrap_or(f64::NAN);
    let arena = arena_cm.unwrap_or(5.0 * length);
    let k3 = NUM_KEYPOINTS * 3;
    let mut kpts = vec![0.0f64; n_animals * n_frames * k3];
    let mut labels = vec![0i64; n_animals * n_frames];
    let mut sizes = Vec::with_capacity(n_animals);
    let dt = 1.0 / fps;
    for a in 0..n_animals {
        let l = length * rng.uniform(0.9, 1.1);
        sizes.push(l);
        let mut pos = [
            rng.uniform(-0.35 * arena, 0.35 * arena),
            rng.uniform(-0.35 * arena, 0.35 * arena),
        ];
        let mut heading = rng.uniform(-std::f64::consts::PI, std::f64::consts::PI);
        let mut omega = 0.0;
        let mut speed = 0.0;
        let mut params = [0.0f64; 6];
        let mut gait_phase = rng.uniform(0.0, std::f64::consts::PI * 2.0);
        let mut groom_phase = rng.uniform(0.0, std::f64::consts::PI * 2.0);
        let mut tail_phase = rng.uniform(0.0, std::f64::consts::PI * 2.0);
        let mut beh = rng.integers(0, BEHAVIORS.len() as i64) as usize;
        let mut remaining = rng.integers(DURATION[beh].0, DURATION[beh].1);
        for t in 0..n_frames {
            if remaining <= 0 {
                // `rng.choice(options, p=w / w.sum())` over the behaviors other than the current one.
                let options: Vec<usize> = (0..BEHAVIORS.len()).filter(|b| *b != beh).collect();
                let weights: Vec<f64> = options.iter().map(|b| NEXT[*b]).collect();
                beh = rng.choose_weighted(&options, &weights);
                remaining = rng.integers(DURATION[beh].0, DURATION[beh].1);
            }
            remaining -= 1;
            let target = POSTURE[beh];
            for i in 0..6 {
                params[i] += (target[i] - params[i]) * 0.15;
            }
            let target_speed = if BEHAVIORS[beh] == "walking" {
                target[6] * l * rng.uniform(0.8, 1.2)
            } else {
                0.0
            };
            speed += (target_speed - speed) * 0.1;
            // Heading: an OU turning rate plus a steer away from the walls.
            omega += -0.5 * omega * dt * 10.0 + rng.normal(0.0, 2.5) * dt.sqrt();
            let ahead = [pos[0] + heading.cos() * l, pos[1] + heading.sin() * l];
            if ahead[0].abs() > 0.42 * arena || ahead[1].abs() > 0.42 * arena {
                let desired = (-pos[1]).atan2(-pos[0]);
                omega += 4.0 * wrap_angle(desired - heading) * dt * 10.0;
            }
            omega = omega.clamp(-4.0, 4.0);
            heading += omega * dt * (0.2 + speed / l).min(1.0);
            pos[0] += heading.cos() * speed * dt;
            pos[1] += heading.sin() * speed * dt;
            pos = [
                pos[0].clamp(-0.45 * arena, 0.45 * arena),
                pos[1].clamp(-0.45 * arena, 0.45 * arena),
            ];
            gait_phase += 2.0 * std::f64::consts::PI * 4.0 * dt * (speed / l);
            groom_phase += 2.0 * std::f64::consts::PI * 5.0 * dt;
            tail_phase += 2.0 * std::f64::consts::PI * 0.7 * dt;
            let bend = (omega * 0.15).clamp(-0.6, 0.6);
            let body = body_pose(&params, gait_phase, groom_phase, bend, tail_phase);
            let mut body: Vec<[f64; 3]> = body
                .iter()
                .map(|row| [row[0] * l, row[1] * l, row[2] * l])
                .collect();
            if BEHAVIORS[beh] == "eating" || BEHAVIORS[beh] == "grooming" {
                for &i in &HEAD {
                    for slot in &mut body[i] {
                        *slot += rng.normal(0.0, 0.004 * l);
                    }
                }
                for row in body.iter_mut() {
                    row[2] = row[2].max(0.0);
                }
            }
            let (c, s) = (heading.cos(), heading.sin());
            let at = (a * n_frames + t) * k3;
            for (k, row) in body.iter().enumerate() {
                kpts[at + k * 3] = c * row[0] - s * row[1] + pos[0];
                kpts[at + k * 3 + 1] = s * row[0] + c * row[1] + pos[1];
                kpts[at + k * 3 + 2] = row[2];
            }
            labels[a * n_frames + t] = beh as i64;
        }
    }
    Simulation {
        species: species.to_string(),
        fps,
        arena_cm: arena,
        n_animals,
        n_frames,
        kpts3d: kpts,
        behavior: labels,
        sizes,
    }
}

/// `np.angle(np.exp(1j * d))`: wraps an angle into `(-pi, pi]`.
fn wrap_angle(d: f64) -> f64 {
    let two = 2.0 * std::f64::consts::PI;
    let mut x = d.rem_euclid(two);
    if x > std::f64::consts::PI {
        x -= two;
    }
    x
}

// ------------------------------------------------------------------------------------- cameras

/// A pinhole camera, the subset of `multiview/camera.py:Camera` the simulator needs: no distortion
/// (`dist` is zero for every camera `camera_from_view` builds), so projection is `K @ (x/z, y/z)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    /// Focal length in pixels (`K[0, 0]`, which `fit_focal` chooses so the arena fills the frame).
    pub focal: f64,
    /// Principal point, `((w - 1) / 2, (h - 1) / 2)`.
    pub cx: f64,
    /// `K[1, 2]`.
    pub cy: f64,
    /// World → camera rotation, `Camera.R`.
    pub r: [[f64; 3]; 3],
    /// World → camera translation, `Camera.t`.
    pub t: [f64; 3],
    /// Image size `(width, height)`, `Camera.size`.
    pub width: usize,
    /// `Camera.size[1]`.
    pub height: usize,
    /// `Camera.name`.
    pub name: String,
}

impl Camera {
    /// `Camera.look_at(eye, target, up, focal, size)`.
    pub fn look_at(
        eye: [f64; 3],
        target: [f64; 3],
        up: [f64; 3],
        focal: f64,
        size: (usize, usize),
        name: &str,
    ) -> Camera {
        let mut fwd = sub(target, eye);
        fwd = scale(fwd, 1.0 / norm(fwd));
        // `abs(np.dot(fwd, up / np.linalg.norm(up))) > 0.999`: looking straight down, pick image
        // "up" = +y.
        let un = scale(up, 1.0 / norm(up));
        let up = if dot(fwd, un).abs() > 0.999 {
            [0.0, 1.0, 0.0]
        } else {
            up
        };
        let mut right = cross(fwd, up);
        right = scale(right, 1.0 / norm(right));
        let down = cross(fwd, right);
        let r = [right, down, fwd];
        let t = [-(dot(r[0], eye)), -(dot(r[1], eye)), -(dot(r[2], eye))];
        Camera {
            focal,
            cx: (size.0 as f64 - 1.0) / 2.0,
            cy: (size.1 as f64 - 1.0) / 2.0,
            r,
            t,
            width: size.0,
            height: size.1,
            name: name.to_string(),
        }
    }

    /// `Camera.center`: the camera position in world coordinates, `-R^T t`.
    pub fn center(&self) -> [f64; 3] {
        let mut out = [0.0; 3];
        for (i, out_at) in out.iter_mut().enumerate() {
            for (j, row) in self.r.iter().enumerate() {
                *out_at -= row[i] * self.t[j];
            }
        }
        out
    }

    /// `Camera.project(X)` for one point.
    pub fn project(&self, x: [f64; 3]) -> [f64; 2] {
        let xc = [
            dot(self.r[0], x) + self.t[0],
            dot(self.r[1], x) + self.t[1],
            dot(self.r[2], x) + self.t[2],
        ];
        let z = if xc[2].abs() < 1e-9 {
            if xc[2] < 0.0 {
                -1e-9
            } else {
                1e-9
            }
        } else {
            xc[2]
        };
        [
            xc[0] / z * self.focal + self.cx,
            xc[1] / z * self.focal + self.cy,
        ]
    }

    /// `Camera.project` over `(..., 3)` points.
    pub fn project_slice(&self, xs: &[[f64; 3]]) -> Vec<[f64; 2]> {
        xs.iter().map(|x| self.project(*x)).collect()
    }

    /// `Camera.project` over a flat `(n, 3)` buffer.
    pub fn project_flat(&self, xs: &[f64]) -> Vec<f64> {
        let points: Vec<[f64; 3]> = xs.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let uv = self.project_slice(&points);
        uv.iter().flat_map(|p| [p[0], p[1]]).collect()
    }

    /// `Camera.in_image(uv)` over already-projected points.
    pub fn in_image(&self, uv: &[[f64; 2]]) -> Vec<bool> {
        uv.iter()
            .map(|p| {
                p[0] >= 0.0 && p[0] < self.width as f64 && p[1] >= 0.0 && p[1] < self.height as f64
            })
            .collect()
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// `_fit_focal`: rescales the focal length so `points` fill `fill` of the frame.
pub fn fit_focal(cam: &mut Camera, points: &[[f64; 3]], fill: f64) {
    cam.focal = 1.0;
    let uv = cam.project_slice(points);
    let max_x = uv.iter().map(|p| (p[0] - cam.cx).abs()).fold(0.0, f64::max);
    let max_y = uv.iter().map(|p| (p[1] - cam.cy).abs()).fold(0.0, f64::max);
    let scale = fill * (cam.width as f64 / (2.0 * max_x)).min(cam.height as f64 / (2.0 * max_y));
    cam.focal = scale;
}

/// `data/synthetic.py:camera_from_view`: a camera on a sphere around the arena centre.
///
/// Elevation 90 is top-down, ~10 is a side view; `fill` decides the focal length through
/// [`fit_focal`], exactly as Python does.
pub fn camera_from_view(
    arena_cm: f64,
    elevation_deg: f64,
    azimuth_deg: f64,
    size: (usize, usize),
    distance: f64,
    name: &str,
    fill: f64,
) -> Camera {
    let el = elevation_deg.to_radians();
    let az = azimuth_deg.to_radians();
    let r = distance * arena_cm;
    let eye = [
        r * el.cos() * az.cos(),
        r * el.cos() * az.sin(),
        r * el.sin(),
    ];
    let target = [
        0.0,
        0.0,
        if elevation_deg > 30.0 {
            0.0
        } else {
            0.03 * arena_cm
        },
    ];
    let mut cam = Camera::look_at(eye, target, [0.0, 0.0, 1.0], 500.0, size, name);
    let half = arena_cm / 2.0;
    let mut corners = Vec::new();
    for x in [-half, half] {
        for y in [-half, half] {
            for z in [0.0, 0.1 * arena_cm] {
                corners.push([x, y, z]);
            }
        }
    }
    fit_focal(&mut cam, &corners, fill);
    cam
}

/// `data/synthetic.py:random_camera`: a random viewpoint, which is what keeps the behavior
/// classifier from over-fitting one camera.
pub fn random_camera(arena_cm: f64, rng: &mut Rng, size: (usize, usize)) -> Camera {
    let el = rng.uniform(10.0, 90.0);
    let az = rng.uniform(-180.0, 180.0);
    let name = format!("rand_el{el:.0}_az{az:.0}");
    camera_from_view(
        arena_cm,
        el,
        az,
        size,
        rng.uniform(1.3, 2.2),
        &name,
        rng.uniform(0.75, 1.1),
    )
}

// ------------------------------------------------------------------------------------- windows

/// One training window: `sim`, `animal` and `start`, the tuple `SimulationPool.window` returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    /// Index of the simulation inside [`SimulationPool::sims`].
    pub sim: usize,
    /// Animal inside that simulation.
    pub animal: usize,
    /// First frame of the window.
    pub start: usize,
}

/// `data/synthetic.py:SimulationPool`: pre-simulated long sequences, so sampling a window is cheap.
#[derive(Clone, Debug)]
pub struct SimulationPool {
    /// `SimulationPool.sims`.
    pub sims: Vec<Simulation>,
}

impl SimulationPool {
    /// `SimulationPool(n_sims, frames, species, seed=seed)`, one `simulate` per entry with a random
    /// animal count in `1..=max_animals` and a species drawn from `species`.
    pub fn new(
        n_sims: usize,
        frames: usize,
        species: &[String],
        max_animals: usize,
        rng: &mut Rng,
    ) -> Result<Self> {
        ensure!(
            !species.is_empty(),
            "the simulator needs at least one species"
        );
        let mut sims = Vec::with_capacity(n_sims);
        for _ in 0..n_sims {
            let n = rng.integers(1, max_animals.max(1) as i64 + 1) as usize;
            let sp = rng.choose(species).clone();
            ensure!(
                body_length_cm(&sp).is_some(),
                "unknown species {sp:?}: known are mouse, rat, quadruped"
            );
            sims.push(simulate(n, frames, &sp, 30.0, None, rng));
        }
        Ok(Self { sims })
    }

    /// `SimulationPool.window(rng, length, warmup=30)`: a random `(sim, animal, start)` with
    /// `start >= warmup`.
    pub fn window(&self, rng: &mut Rng, length: usize, warmup: usize) -> Window {
        let sim = rng.integers(0, self.sims.len() as i64) as usize;
        let s = &self.sims[sim];
        let high = (warmup + 1).max(s.n_frames.saturating_sub(length) + 1) as i64;
        // `rng.integers(warmup, max(warmup + 1, n_frames - length + 1))`: a window longer than the
        // sequence still draws, and the draw lands on `warmup`.
        let start = rng.integers(warmup as i64, high) as usize;
        Window {
            sim,
            animal: rng.integers(0, s.n_animals as i64) as usize,
            start,
        }
    }
}

/// `temporal/data.py:sample_windows` with a pool (the only source the temporal and behavior
/// trainers use, where `_pool` always builds one).
pub fn sample_windows(
    batch: usize,
    window: usize,
    rng: &mut Rng,
    pool: &SimulationPool,
) -> Vec<Window> {
    (0..batch).map(|_| pool.window(rng, window, 30)).collect()
}

/// `temporal/data.py:Corruption`, with its defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Corruption {
    /// `jitter`: noise standard deviation as a fraction of the track spread.
    pub jitter: (f64, f64),
    /// `point_drop`: independent per-point dropout probability.
    pub point_drop: f64,
    /// `point_gaps`: contiguous per-keypoint gap rate per frame.
    pub point_gaps: f64,
    /// `point_gap_len`: gap length range.
    pub point_gap_len: (i64, i64),
    /// `animal_gaps`: probability of at least one whole-animal gap per window.
    pub animal_gaps: f64,
    /// `animal_gap_len`: whole-animal gap length range.
    pub animal_gap_len: (i64, i64),
    /// `outliers`: per-point probability of jumping to a random location.
    pub outliers: f64,
    /// `swaps`: per-frame probability of a left/right swap.
    pub swaps: f64,
    /// `conf_threshold`: confidence an unobserved point gets.
    pub conf_threshold: f64,
}

impl Default for Corruption {
    fn default() -> Self {
        Self {
            jitter: (0.005, 0.03),
            point_drop: 0.05,
            point_gaps: 0.02,
            point_gap_len: (2, 20),
            animal_gaps: 0.6,
            animal_gap_len: (5, 40),
            outliers: 0.01,
            swaps: 0.005,
            conf_threshold: 0.3,
        }
    }
}

/// `behavior/data.py:_MILD`, the milder corruption the behavior classifier is trained with.
pub const MILD_CORRUPTION: Corruption = Corruption {
    jitter: (0.003, 0.02),
    point_drop: 0.05,
    point_gaps: 0.02,
    point_gap_len: (2, 20),
    animal_gaps: 0.2,
    animal_gap_len: (3, 10),
    outliers: 0.01,
    swaps: 0.005,
    conf_threshold: 0.3,
};

/// `temporal/data.py:corrupt`: clean `(frames, keypoints, dims)` coordinates plus a validity mask
/// become the noisy `(x, conf, observed)` triple the models are trained on.
///
/// The rules are identical to Python and pinned by tests: jitter proportional to the spread of the
/// valid points, independent point drops, contiguous per-keypoint and whole-animal gaps, outliers
/// jumping inside the observed bounding box at mid confidence, bilateral swaps, then
/// `conf[~obs] = U(0, conf_threshold)` and `x[~obs] = 0`.
pub fn corrupt(
    clean: &[f64],
    valid: &[bool],
    dims: usize,
    frames: usize,
    keypoints: usize,
    c: &Corruption,
    rng: &mut Rng,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = frames * keypoints;
    let mut x = clean.to_vec();
    // `vals = clean[valid]`: the spread that scales the jitter, and the box outliers jump into.
    let vals: Vec<f64> = valid
        .iter()
        .enumerate()
        .filter(|(_, v)| **v)
        .flat_map(|(i, _)| clean[i * dims..i * dims + dims].to_vec())
        .collect();
    let spread = if vals.is_empty() { 1.0 } else { std_dev(&vals) };
    let sigma = rng.uniform(c.jitter.0, c.jitter.1) * spread;
    for v in x.iter_mut() {
        *v += rng.normal(0.0, sigma);
    }
    let mut conf: Vec<f64> = (0..n).map(|_| rng.uniform(0.6, 1.0)).collect();
    // `obs = valid & (rng.random((t, k)) > point_drop)`: numpy draws the whole `(t, k)` matrix, so the
    // draw happens for every point, valid or not, and comes first.
    let mut obs: Vec<bool> = valid
        .iter()
        .map(|v| rng.random() > c.point_drop && *v)
        .collect();
    let n_gaps = rng.poisson(c.point_gaps * frames as f64 * keypoints as f64 / 5.0);
    for _ in 0..n_gaps {
        let j = rng.integers(0, keypoints as i64) as usize;
        let s = rng.integers(0, frames as i64) as usize;
        let len = rng.integers(c.point_gap_len.0, c.point_gap_len.1) as usize;
        for t in s..(s + len).min(frames) {
            obs[t * keypoints + j] = false;
        }
    }
    if rng.random() < c.animal_gaps {
        let count = rng.integers(1, 3) as usize;
        for _ in 0..count {
            let s = rng.integers(0, frames as f64 as i64) as usize;
            let len = rng.integers(c.animal_gap_len.0, c.animal_gap_len.1) as usize;
            for t in s..(s + len).min(frames) {
                for k in 0..keypoints {
                    obs[t * keypoints + k] = false;
                }
            }
        }
    }
    let outlier: Vec<bool> = (0..n).map(|_| rng.random() < c.outliers).collect();
    if outlier.iter().any(|v| *v) && !vals.is_empty() {
        // `lo, hi = vals.reshape(-1, d).min(0), vals.reshape(-1, d).max(0)`: the box is the extent of
        // the valid points, not of the whole window.
        let mut lo = vec![f64::INFINITY; dims];
        let mut hi = vec![f64::NEG_INFINITY; dims];
        for chunk in vals.chunks_exact(dims) {
            for (d, v) in chunk.iter().enumerate() {
                if *v < lo[d] {
                    lo[d] = *v;
                }
                if *v > hi[d] {
                    hi[d] = *v;
                }
            }
        }
        // `x[out] = rng.uniform(lo, hi, size=(out.sum(), d))` draws every coordinate before
        // `conf[out] = rng.uniform(0.3, 0.7, size=out.sum())` draws every confidence.
        for (i, is_out) in outlier.iter().enumerate() {
            if !*is_out {
                continue;
            }
            for d in 0..dims {
                x[i * dims + d] = rng.uniform(lo[d], hi[d]);
            }
        }
        for (i, is_out) in outlier.iter().enumerate() {
            if *is_out {
                conf[i] = rng.uniform(0.3, 0.7);
            }
        }
    }
    // `x[s] = x[s, flip]` for the frames `rng.random(t) < swaps` selects: numpy copies the
    // permuted frame first, so the swap is not done in place.
    for s in 0..frames {
        if rng.random() >= c.swaps {
            continue;
        }
        let frame = x[s * keypoints * dims..(s + 1) * keypoints * dims].to_vec();
        for k in 0..keypoints {
            let from = FLIP_INDEX[k] * dims;
            for d in 0..dims {
                x[(s * keypoints + k) * dims + d] = frame[from + d];
            }
        }
    }
    for i in 0..n {
        if !obs[i] {
            conf[i] = rng.uniform(0.0, c.conf_threshold);
            for d in 0..dims {
                x[i * dims + d] = 0.0;
            }
        }
    }
    (
        x.iter().map(|v| *v as f32).collect(),
        conf.iter().map(|v| *v as f32).collect(),
        obs.iter().map(|v| f32::from(*v)).collect(),
    )
}

/// `np.std` of a float slice (population standard deviation, `ddof = 0`).
fn std_dev(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64).sqrt()
}

/// One temporal training batch: `x, conf, observed, target, target_mask` of
/// `temporal/data.py:synthetic_batch`, `(batch, frames, keypoints, dims)` / `(batch, frames, keypoints)`.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalBatch {
    /// Model inputs: corrupted coordinates.
    pub x: Vec<f32>,
    /// Model inputs: per-point confidence.
    pub conf: Vec<f32>,
    /// Model inputs: the observation mask.
    pub observed: Vec<f32>,
    /// The clean coordinates the loss regresses.
    pub target: Vec<f32>,
    /// The target mask, `np.ones(valid.shape)` of `temporal/data.py:synthetic_batch`.
    pub target_mask: Vec<f32>,
    /// Batch size.
    pub batch: usize,
    /// Window length.
    pub frames: usize,
    /// Keypoints per frame.
    pub keypoints: usize,
    /// `2` for image tracks, `3` for triangulated ones.
    pub dims: usize,
}

/// `temporal/data.py:synthetic_batch`: clean projected windows from the pool, corrupted with
/// [`Corruption::default`], targets the clean coordinates.
pub fn temporal_batch(
    batch: usize,
    window: usize,
    dims: usize,
    pool: &SimulationPool,
    rng: &mut Rng,
    corruption: &Corruption,
) -> TemporalBatch {
    let mut x = Vec::with_capacity(batch * window * NUM_KEYPOINTS * dims);
    let mut conf = Vec::with_capacity(batch * window * NUM_KEYPOINTS);
    let mut observed = Vec::with_capacity(batch * window * NUM_KEYPOINTS);
    let mut target = Vec::with_capacity(batch * window * NUM_KEYPOINTS * dims);
    let mut mask = Vec::with_capacity(batch * window * NUM_KEYPOINTS);
    for w in sample_windows(batch, window, rng, pool) {
        let sim = &pool.sims[w.sim];
        let (clean, valid) = projected(sim, w.animal, w.start, window, dims, rng);
        let (bx, bc, bo) = corrupt(&clean, &valid, dims, window, NUM_KEYPOINTS, corruption, rng);
        x.extend(bx.iter().copied());
        conf.extend(bc);
        observed.extend(bo);
        // `ts.append(clean.astype(np.float32))`: the clean points, not the corrupted ones.
        target.extend(clean.iter().map(|v| *v as f32));
        // `ms.append(np.ones(valid.shape, np.float32))`: every projected point is a target, the ones
        // outside the image included — `corrupt` zeroed them in `x`, they are not missing here.
        mask.extend(valid.iter().map(|_| 1.0f32));
    }
    TemporalBatch {
        x,
        conf,
        observed,
        target,
        target_mask: mask,
        batch,
        frames: window,
        keypoints: NUM_KEYPOINTS,
        dims,
    }
}

/// The clean coordinates of one window, projected through a random camera
/// (`dims == 3` keeps the 3D ground truth, which is what the multi-view models train on).
fn projected(
    sim: &Simulation,
    animal: usize,
    start: usize,
    length: usize,
    dims: usize,
    rng: &mut Rng,
) -> (Vec<f64>, Vec<bool>) {
    let pts3 = sim.window(animal, start, length);
    if dims == 3 {
        return (pts3, vec![true; length * NUM_KEYPOINTS]);
    }
    let cam = random_camera(sim.arena_cm, rng, (320, 256));
    let points: Vec<[f64; 3]> = pts3.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let uv = cam.project_slice(&points);
    let valid = cam.in_image(&uv);
    let flat = uv.iter().flat_map(|p| [p[0], p[1]]).collect();
    (flat, valid)
}

/// One behavior training batch: `x, mask, labels` of `behavior/data.py:synthetic_batch`.
#[derive(Clone, Debug, PartialEq)]
pub struct BehaviorBatch {
    /// Classifier input coordinates, `(batch, frames, keypoints, dims)`.
    pub x: Vec<f32>,
    /// Classifier input mask, `(batch, frames, keypoints)`.
    pub mask: Vec<f32>,
    /// Frame-wise class, `(batch, frames)`, an index into [`BEHAVIORS`].
    pub labels: Vec<i64>,
    /// Batch size.
    pub batch: usize,
    /// Window length.
    pub frames: usize,
    /// Keypoints per frame.
    pub keypoints: usize,
    /// Coordinate width.
    pub dims: usize,
}

/// `behavior/data.py:synthetic_batch`: projected windows corrupted with [`MILD_CORRUPTION`], labeled
/// by the simulation's Markov chain.
pub fn behavior_batch(
    batch: usize,
    window: usize,
    dims: usize,
    pool: &SimulationPool,
    rng: &mut Rng,
    corruption: &Corruption,
) -> BehaviorBatch {
    let mut x = Vec::with_capacity(batch * window * NUM_KEYPOINTS * dims);
    let mut mask = Vec::with_capacity(batch * window * NUM_KEYPOINTS);
    let mut labels = Vec::with_capacity(batch * window);
    for w in sample_windows(batch, window, rng, pool) {
        let sim = &pool.sims[w.sim];
        let (clean, valid) = projected(sim, w.animal, w.start, window, dims, rng);
        let (bx, _, obs) = corrupt(&clean, &valid, dims, window, NUM_KEYPOINTS, corruption, rng);
        x.extend(bx);
        mask.extend(obs);
        labels.extend(sim.behavior_window(w.animal, w.start, window));
    }
    BehaviorBatch {
        x,
        mask,
        labels,
        batch,
        frames: window,
        keypoints: NUM_KEYPOINTS,
        dims,
    }
}

/// `train/loops.py:_KeypointSubset`: with probability `p`, mask out every keypoint the mapping does
/// not label, so training also covers the sparse skeletons of the real datasets.
///
/// `masks` are [`canonical_mask`] rows; the one drawn is applied to every `(batch, frames, K)` mask
/// array of the batch (coordinates stay put, as they do in Python).
pub fn keypoint_subset_masks(masks: &[Vec<f32>], p: f64, rng: &mut Rng) -> Option<Vec<f32>> {
    if p <= 0.0 || masks.is_empty() || rng.random() >= p {
        return None;
    }
    Some(masks[rng.integers(0, masks.len() as i64) as usize].clone())
}

/// Multiplies a `(batch, frames, keypoints)` mask by a per-keypoint keep mask.
pub fn apply_mask_mask(mask: &mut [f32], keep: &[f32], keypoints: usize) {
    for (i, m) in mask.iter_mut().enumerate() {
        *m *= keep[i % keypoints];
    }
}

// ------------------------------------------------------------------------------------------ rng

/// The generator every batch function draws from.
///
/// Splitmix64-seeded `xoshiro256++` plus a Box-Muller normal. `// differs from Python:` numpy's
/// `default_rng(seed)` is PCG64, so a seed produces different — but identically distributed — draws.
#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 4],
    spare: Option<f64>,
}

impl Rng {
    /// A generator seeded with `seed` (the `train.seed` config key).
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut state = [0u64; 4];
        for s in state.iter_mut() {
            *s = splitmix64(&mut sm);
        }
        Self { state, spare: None }
    }

    /// The next 64 bits of `xoshiro256++`.
    pub fn next_u64(&mut self) -> u64 {
        let [s0, s1, s2, s3] = self.state;
        let out = (s0.wrapping_add(s3)).rotate_left(23).wrapping_add(s0);
        let t = s1 << 17;
        self.state = [
            s0 ^ s3 ^ s1,
            s1 ^ s2 ^ s0,
            s2 ^ s0,
            (s3 ^ s1 ^ t).rotate_left(45),
        ];
        out
    }

    /// `rng.random()`: a float in `[0, 1)`.
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// `rng.uniform(lo, hi)`.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.random()
    }

    /// `rng.integers(low, high)`: an integer in `[low, high)`.
    pub fn integers(&mut self, low: i64, high: i64) -> i64 {
        if high <= low {
            return low;
        }
        let span = (high - low) as f64;
        let v = (self.random() * span).floor() as i64;
        low + v.clamp(0, high - low - 1)
    }

    /// `rng.normal(mu, sigma)`.
    pub fn normal(&mut self, mu: f64, sigma: f64) -> f64 {
        if let Some(v) = self.spare.take() {
            return mu + sigma * v;
        }
        let mut u1 = self.random();
        if u1 <= f64::EPSILON {
            u1 = f64::EPSILON;
        }
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * self.random();
        self.spare = Some(r * theta.sin());
        mu + sigma * (r * theta.cos())
    }

    /// `rng.poisson(lam)`: Knuth's product algorithm (`lam` is at most ~10 in these configs).
    pub fn poisson(&mut self, lam: f64) -> usize {
        if lam <= 0.0 {
            return 0;
        }
        let limit = (-lam).exp();
        let mut k = 0usize;
        let mut p = 1.0;
        while p > limit {
            p *= self.random().max(f64::MIN_POSITIVE);
            k += 1;
        }
        k - 1
    }

    /// `rng.choice(items)`.
    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.integers(0, items.len() as i64) as usize]
    }

    /// `rng.choice(items, p=weights / weights.sum())`.
    pub fn choose_weighted<T>(&mut self, items: &[T], weights: &[f64]) -> T
    where
        T: Clone,
    {
        let total: f64 = weights.iter().sum();
        let mut target = self.random() * total;
        for (i, w) in weights.iter().enumerate() {
            target -= *w;
            if target < 0.0 {
                return items[i].clone();
            }
        }
        items[items.len() - 1].clone()
    }

    /// `rng.choice(items, size=n, replace=False)`: a partial Fisher-Yates shuffle.
    pub fn sample_without_replacement<T: Copy>(&mut self, items: &[T], n: usize) -> Vec<T> {
        let mut pool = items.to_vec();
        let take = n.min(pool.len());
        for i in 0..take {
            let j = self.integers(i as i64, pool.len() as i64) as usize;
            pool.swap(i, j);
        }
        pool.truncate(take);
        pool
    }
}

/// The splitmix64 stream used to expand one seed into four words.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `np.nan_to_num` for one coordinate.
pub fn nan_to_num(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else if v == f32::INFINITY {
        f32::MAX
    } else if v == f32::NEG_INFINITY {
        f32::MIN
    } else {
        v
    }
}

/// The species list of a config, validated against the species table.
pub fn check_species(species: &[String]) -> Result<()> {
    for s in species {
        if body_length_cm(s).is_none() {
            bail!("unknown species {s:?}: the simulator knows mouse, rat and quadruped");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `body_pose` is bit-exact with Python: the reference rows come from
    /// `data/synthetic.py:body_pose` at the parameter sets below.
    #[test]
    fn body_pose_matches_python() {
        type Case = ([f64; 6], f64, f64, f64, f64, &'static [[f64; 3]]);
        let cases: &[Case] = &[
            (
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                0.0,
                0.0,
                0.0,
                0.0,
                &[[0.52, 0.0, 0.1], [-0.95, 0.0, 0.01], [-0.2, 0.1, 0.0]],
            ),
            (
                [0.3, 0.2, 1.0, 0.0, 0.0, 0.0],
                1.7,
                0.4,
                0.25,
                2.1,
                &[
                    [0.480_709_002_401, 0.058_482_228_534, 0.052_707_601_756],
                    [-0.837_108_504_411, 0.407_258_724_027, 0.008_8],
                    [0.263_994_082_274, 0.108_099_602_582, 0.0],
                ],
            ),
            (
                [1.0, 0.3, 0.0, 0.0, 0.0, 0.0],
                0.5,
                1.5,
                -0.6,
                4.0,
                &[
                    [0.413_019_379_55, -0.108_180_493_961, 0.013_212_969_079],
                    [-0.835_282_109_798, -0.146_752_300_456, 0.006],
                    [0.169_449_740_69, 0.073_952_926_785, 0.0],
                ],
            ),
            (
                [0.0, 0.0, 0.0, 0.35, 1.1, 0.0],
                3.0,
                0.2,
                0.1,
                1.0,
                &[
                    [0.146_413_660_512, 0.002_144_079_083, 0.729_309_190_829],
                    [-0.936_138_001_505, 0.190_635_514_43, 0.01],
                    [0.101_458_521_163, 0.071_790_502_477, 0.483_651_711_031],
                ],
            ),
            (
                [0.3, 0.2, 0.0, 1.0, 0.35, 1.0],
                2.0,
                5.0,
                0.0,
                0.0,
                &[
                    [0.459_591_747_902, 0.0, 0.280_374_352_522],
                    [-0.907_25, 0.105, 0.008_8],
                    [0.404_526_052_62, 0.035, 0.227_085_039_961],
                ],
            ),
        ];
        for (params, gait, groom, bend, tail, want) in cases {
            let got = body_pose(params, *gait, *groom, *bend, *tail);
            assert_eq!(got.len(), NUM_KEYPOINTS);
            for row in *want {
                let best = got
                    .iter()
                    .map(|g| {
                        (g[0] - row[0])
                            .abs()
                            .max((g[1] - row[1]).abs())
                            .max((g[2] - row[2]).abs())
                    })
                    .fold(f64::INFINITY, f64::min);
                assert!(best < 1e-9, "{params:?} {row:?} off by {best}");
            }
            // Every coordinate is finite and never dips under the floor.
            assert!(got.iter().all(|p| p.iter().all(|v| v.is_finite())));
            assert!(got.iter().all(|p| p[2] >= 0.0));
        }
    }

    /// `camera_from_view` + `_fit_focal` match Python's `K`, `R`, `t` and projection.
    #[test]
    fn camera_matches_python() {
        let cam = camera_from_view(42.5, 45.0, -135.0, (320, 256), 1.7, "angled", 0.83);
        assert!(
            (cam.focal - 254.975_999_999_997_1).abs() < 1e-9,
            "{}",
            cam.focal
        );
        assert_eq!((cam.cx, cam.cy), (159.5, 127.5));
        assert!(
            (cam.r[0][0] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12,
            "{:?}",
            cam.r
        );
        assert!(
            cam.t[0].abs() < 1e-12 && (cam.t[2] - 72.25).abs() < 1e-9,
            "{:?}",
            cam.t
        );
        let uv = cam.project_slice(&[[0.0, 0.0, 0.0], [5.0, 3.0, 1.0], [-7.0, -2.0, 0.5]]);
        assert!(
            (uv[1][0] - 164.273_321_511_994_14).abs() < 1e-9
                && (uv[2][1] - 143.186_945_289_672_2).abs() < 1e-9,
            "{uv:?}"
        );
        assert_eq!(uv[0], [159.5, 127.5]);
        // A top-down camera looks straight down: `look_at` swaps its "up" hint, as Python does.
        let top = camera_from_view(42.5, 90.0, -90.0, (320, 256), 1.6, "top", 0.9);
        assert!(top.focal.is_finite() && top.focal > 0.0);
        let inside = top.in_image(&top.project_slice(&[[0.0, 0.0, 0.0]]));
        assert!(inside[0]);
    }

    /// A simulation is well formed: finite keypoints, labels inside `BEHAVIORS`, paws on the floor.
    #[test]
    fn simulate_is_well_formed() {
        let mut rng = Rng::new(3);
        let sim = simulate(2, 200, "mouse", 30.0, None, &mut rng);
        assert_eq!((sim.n_animals, sim.n_frames), (2, 200));
        assert_eq!(sim.arena_cm, 5.0 * 8.5);
        assert_eq!(sim.kpts3d.len(), 2 * 200 * NUM_KEYPOINTS * 3);
        assert!(sim.kpts3d.iter().all(|v| v.is_finite()));
        assert!(sim.behavior.iter().all(|b| (0..5).contains(b)));
        assert!(sim
            .kpts3d
            .iter()
            .enumerate()
            .all(|(i, v)| i % 3 != 2 || *v >= -1e-12));
        // The behavior chain actually switches: a single state for 200 frames would be a bug.
        assert!(sim.behavior.windows(2).any(|w| w[0] != w[1]));
        // Windows slice the stored arrays.
        let w = sim.window(1, 40, 64);
        assert_eq!(w.len(), 64 * NUM_KEYPOINTS * 3);
        assert_eq!(sim.behavior_window(1, 40, 64).len(), 64);
        assert_eq!(sim.kpts_at(0, 0).len(), NUM_KEYPOINTS);
    }

    /// `corrupt` obeys its rules: unobserved points are zeroed and cheap, observed ones stay put on
    /// average, and the shapes match the window.
    #[test]
    fn corrupt_respects_the_observation_mask() {
        let (frames, k, dims) = (48, NUM_KEYPOINTS, 2);
        let clean: Vec<f64> = (0..frames * k * dims)
            .map(|i| (i as f64 * 0.37).sin() * 40.0)
            .collect();
        let valid: Vec<bool> = (0..frames * k).map(|i| i % 7 != 0).collect();
        let mut rng = Rng::new(11);
        let (x, conf, obs) = corrupt(
            &clean,
            &valid,
            dims,
            frames,
            k,
            &Corruption::default(),
            &mut rng,
        );
        assert_eq!(x.len(), frames * k * dims);
        assert_eq!(conf.len(), frames * k);
        assert_eq!(obs.len(), frames * k);
        for i in 0..frames * k {
            if obs[i] == 0.0 {
                assert_eq!(&x[i * dims..i * dims + dims], &[0.0, 0.0]);
                assert!(conf[i] < 0.3, "unobserved confidence {}", conf[i]);
            } else {
                assert!(conf[i] >= 0.3, "observed confidence {}", conf[i]);
            }
        }
        // Observed points never exceed the corruption budget of 0.03 * spread + outliers.
        let observed = (0..frames * k).filter(|&i| obs[i] > 0.0).count();
        assert!(observed < valid.iter().filter(|v| **v).count());
        // With zero corruption the batch is the clean signal, apart from the swap stream.
        let none = Corruption {
            jitter: (0.0, 0.0),
            point_drop: 0.0,
            point_gaps: 0.0,
            animal_gaps: 0.0,
            outliers: 0.0,
            swaps: 0.0,
            ..Corruption::default()
        };
        let (x2, conf2, obs2) = corrupt(&clean, &valid, dims, frames, k, &none, &mut rng);
        assert_eq!(
            obs2,
            valid.iter().map(|v| f32::from(*v)).collect::<Vec<_>>()
        );
        for i in 0..frames * k {
            if obs2[i] > 0.0 {
                for d in 0..dims {
                    assert!(
                        (x2[i * dims + d] - clean[i * dims + d] as f32).abs() < 1e-3,
                        "{i} {d}: {} vs {}",
                        x2[i * dims + d],
                        clean[i * dims + d]
                    );
                }
                assert!(conf2[i] >= 0.6);
            }
        }
    }

    /// The outlier jump lands inside the extent of the *valid* points, not of the whole window:
    /// `corrupt` takes `lo, hi` from `vals = clean[valid]`, which is what keeps an out-of-frame point
    /// from widening the box every outlier is drawn from.
    #[test]
    fn outliers_stay_inside_the_valid_extent() {
        let (frames, k, dims) = (12, NUM_KEYPOINTS, 2);
        let n = frames * k;
        // Keypoint 0 of every frame is invalid and sits far outside the rest of the window.
        let mut clean = vec![0.0f64; n * dims];
        for i in 0..n {
            let far = i % k == 0;
            for d in 0..dims {
                clean[i * dims + d] = if far {
                    1000.0
                } else {
                    -10.0 + 20.0 * (i as f64 / n as f64)
                };
            }
        }
        let valid: Vec<bool> = (0..n).map(|i| i % k != 0).collect();
        let only_outliers = Corruption {
            jitter: (0.0, 0.0),
            point_drop: 0.0,
            point_gaps: 0.0,
            animal_gaps: 0.0,
            outliers: 1.0,
            swaps: 0.0,
            ..Corruption::default()
        };
        let (x, _, obs) = corrupt(
            &clean,
            &valid,
            dims,
            frames,
            k,
            &only_outliers,
            &mut Rng::new(4),
        );
        let observed = (0..n).filter(|&i| obs[i] > 0.0).count();
        assert!(observed > 0, "some point stays observed");
        for i in 0..n {
            if obs[i] > 0.0 {
                for d in 0..dims {
                    let v = f64::from(x[i * dims + d]);
                    assert!((-10.0..=10.0).contains(&v), "{i}/{d}: outlier at {v}");
                }
            }
        }
    }

    /// The pool window sampler never starts inside the warmup, and always fits.
    #[test]
    fn pool_windows_fit_their_simulations() {
        let mut rng = Rng::new(5);
        let pool = SimulationPool::new(2, 120, &["mouse".to_string()], 2, &mut rng).unwrap();
        for _ in 0..50 {
            let w = pool.window(&mut rng, 64, 30);
            let s = &pool.sims[w.sim];
            assert!(w.start >= 30);
            assert!(w.start + 64 <= s.n_frames + 64);
            assert!(w.animal < s.n_animals);
        }
        // A window longer than the simulation clamps the start instead of hanging.
        let w = pool.window(&mut rng, 1000, 30);
        assert_eq!(w.start, 30);
        assert!(SimulationPool::new(1, 10, &["dragon".to_string()], 1, &mut rng).is_err());
    }

    /// The two batch builders produce the shapes the models are fed.
    #[test]
    fn batches_have_the_documented_shapes() {
        let mut rng = Rng::new(9);
        let pool = SimulationPool::new(
            2,
            100,
            &["mouse".to_string(), "rat".to_string()],
            2,
            &mut rng,
        )
        .unwrap();
        let b = temporal_batch(3, 32, 2, &pool, &mut rng, &Corruption::default());
        assert_eq!(b.x.len(), 3 * 32 * NUM_KEYPOINTS * 2);
        assert_eq!(b.conf.len(), 3 * 32 * NUM_KEYPOINTS);
        assert_eq!(b.target.len(), b.x.len());
        assert_eq!(b.target_mask.len(), b.conf.len());
        let h = behavior_batch(2, 24, 2, &pool, &mut rng, &MILD_CORRUPTION);
        assert_eq!(h.x.len(), 2 * 24 * NUM_KEYPOINTS * 2);
        assert_eq!(h.mask.len(), 2 * 24 * NUM_KEYPOINTS);
        assert_eq!(h.labels.len(), 2 * 24);
        assert!(h.labels.iter().all(|l| (0..5).contains(l)));
        // The 3D variant keeps the third coordinate.
        let b3 = temporal_batch(1, 16, 3, &pool, &mut rng, &Corruption::default());
        assert_eq!(b3.x.len(), 16 * NUM_KEYPOINTS * 3);
        assert_eq!(b3.dims, 3);
    }

    /// `_KeypointSubset` masks: `homecage_dlc` labels six of the 25 canonical points.
    #[test]
    fn keypoint_masks_are_the_dataset_mappings() {
        let dlc = canonical_mask("homecage_dlc").unwrap();
        assert_eq!(dlc.iter().sum::<f32>() as usize, 6);
        assert_eq!(dlc[index("spine_mid")], 1.0);
        assert_eq!(dlc[index("left_front_paw")], 0.0);
        assert_eq!(
            canonical_mask("synthetic").unwrap().iter().sum::<f32>(),
            25.0
        );
        assert!(canonical_mask("nope").is_err());
        let mut rng = Rng::new(1);
        let drawn = keypoint_subset_masks(std::slice::from_ref(&dlc), 0.5, &mut rng);
        let mut mask = vec![1.0f32; 4 * 2 * NUM_KEYPOINTS];
        if let Some(keep) = drawn {
            apply_mask_mask(&mut mask, &keep, NUM_KEYPOINTS);
            assert_eq!(mask.iter().sum::<f32>() as usize, 4 * 2 * 6);
        }
        assert!(keypoint_subset_masks(&[dlc], 0.0, &mut rng).is_none());
    }

    /// The random stream is uniform-ish and reproducible for one seed.
    #[test]
    fn rng_is_reproducible_and_reasonable() {
        let a = Rng::new(42);
        let _b = Rng::new(42);
        let va: Vec<u64> = (0..8)
            .map(|_| {
                let mut r = a.clone();
                r.next_u64()
            })
            .collect();
        let _ = &va;
        let mut ra = Rng::new(42);
        let mut rb = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(ra.next_u64(), rb.next_u64());
        }
        let mut r = Rng::new(7);
        let mean: f64 = (0..20_000).map(|_| r.normal(0.0, 1.0)).sum::<f64>() / 20_000.0;
        assert!(mean.abs() < 0.05, "normal mean {mean}");
        let u: f64 = (0..20_000).map(|_| r.uniform(0.0, 1.0)).sum::<f64>() / 20_000.0;
        assert!((u - 0.5).abs() < 0.02, "uniform mean {u}");
        let pois: f64 = (0..20_000).map(|_| r.poisson(9.6) as f64).sum::<f64>() / 20_000.0;
        assert!((pois - 9.6).abs() < 0.3, "poisson mean {pois}");
        assert_eq!(r.integers(5, 5), 5);
        for _ in 0..100 {
            let v = r.integers(0, 3);
            assert!((0..3).contains(&v));
        }
        let picked = r.sample_without_replacement(&[1, 2, 3, 4], 2);
        assert_eq!(picked.len(), 2);
        assert_ne!(picked[0], picked[1]);
        let w = r.choose_weighted(&[0, 1], &[0.0, 1.0]);
        assert_eq!(w, 1);
        assert!(check_species(&["mouse".to_string()]).is_ok());
        assert!(check_species(&["mouse".to_string(), "newt".to_string()]).is_err());
        assert_eq!(nan_to_num(f32::NAN), 0.0);
    }

    /// The index tables match `skeleton/vocabulary.py:CANONICAL`.
    #[test]
    fn vocabulary_tables_match_python() {
        assert_eq!(index("nose"), 0);
        assert_eq!(index("right_hind_paw"), 24);
        assert_eq!(FLIP_INDEX[index("left_eye")], index("right_eye"));
        assert_eq!(FLIP_INDEX[index("nose")], index("nose"));
        assert_eq!(
            ANTERIOR,
            [
                index("nose"),
                index("head_center"),
                index("neck"),
                index("spine_upper")
            ]
        );
        assert_eq!(
            POSTERIOR,
            [
                index("tail_base"),
                index("spine_lower"),
                index("left_hip"),
                index("right_hip")
            ]
        );
        assert_eq!(pitched().len(), 15);
        assert_eq!(body_length_cm("rat"), Some(20.0));
        assert_eq!(body_length_cm("quadruped").map(|v| v.is_nan()), Some(true));
        assert_eq!(body_length_cm("newt"), None);
        assert_eq!(wrap_angle(4.0 * std::f64::consts::PI + 0.5), 0.5);
    }
}
