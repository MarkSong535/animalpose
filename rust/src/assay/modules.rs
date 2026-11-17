//! The rules, one module per behavior or location. A module reads the shared features and the
//! arena, and adds columns to the per-frame table, behavior states for the priority resolution,
//! a location, and summary fields. New experiments reuse these or add a module to [`create`].

use std::collections::BTreeMap;

use anyhow::Result;
use serde_json::{json, Value};

use super::arena::{box_uv, Floor, Layout, Polygon, Pt};
use super::config::{AssayConfig, Cup};
use super::features::{angle, dist, median, window_median, Features, Pose};
use super::input::{Input, Track2d};
use super::temporal::{self, clean, dwell, dwell_label, frames, hysteresis, totals};
use crate::multiview::camera::Camera;
use crate::multiview::dense::{mulv3, transpose3, Vec3};

/// What every module reads.
pub struct Context<'a> {
    pub input: &'a Input,
    pub cfg: &'a AssayConfig,
    pub pose: &'a Pose,
    pub features: &'a Features,
    /// The cached layout of the setup (zones in top-image pixels, the floor's corners).
    pub layout: Option<&'a Layout>,
}

impl Context<'_> {
    fn n(&self) -> usize {
        self.input.n_frames
    }

    fn fps(&self) -> f64 {
        self.input.fps
    }

    fn at(&self, t: usize, name: &str) -> Option<Vec3> {
        self.pose.at(t, self.input.index(name))
    }

    /// Keypoint `name` of the test animal in the top image, with its confidence.
    fn top(&self, t: usize, name: &str) -> Option<(Pt, f64)> {
        let k = self.input.kpts();
        self.input
            .top
            .as_ref()?
            .point(t, self.input.index(name)?, k)
    }
}

/// One column of the per-frame table.
#[derive(Clone, Debug)]
pub enum Column {
    Bool(String, Vec<Option<bool>>),
    Num(String, Vec<Option<f64>>),
    Text(String, Vec<Option<String>>),
}

impl Column {
    pub fn name(&self) -> &str {
        match self {
            Column::Bool(n, _) | Column::Num(n, _) | Column::Text(n, _) => n,
        }
    }

    pub fn cell(&self, t: usize) -> String {
        match self {
            Column::Bool(_, v) => match v.get(t).copied().flatten() {
                Some(b) => u8::from(b).to_string(),
                None => String::new(),
            },
            Column::Num(_, v) => v
                .get(t)
                .copied()
                .flatten()
                .map(|x| format!("{x:.4}"))
                .unwrap_or_default(),
            Column::Text(_, v) => v
                .get(t)
                .cloned()
                .flatten()
                .unwrap_or_else(|| "unknown".into()),
        }
    }
}

/// What a module adds.
#[derive(Default)]
pub struct Outcome {
    pub columns: Vec<Column>,
    /// Behavior states, for the priority resolution.
    pub states: Vec<(String, Vec<Option<bool>>)>,
    /// Where the animal is, if this module places it.
    pub location: Option<Vec<Option<String>>>,
    pub summary: Vec<(String, Value)>,
    /// Anything a reader of the results should know (a rule that could not apply, …).
    pub notes: Vec<String>,
}

pub trait Module {
    fn name(&self) -> &'static str;
    /// The layout this module needs found for the setup, if any: `floor`, `plus_maze` or `zones`
    /// (named regions; optional — the module runs without them).
    fn needs_layout(&self) -> Option<&'static str> {
        None
    }
    fn run(&self, ctx: &Context) -> Result<Outcome>;
}

/// The module called `name`.
pub fn create(name: &str) -> Option<Box<dyn Module>> {
    Some(match name {
        "open_field" => Box::new(OpenField),
        "plus_maze" => Box::new(PlusMaze),
        "contact" => Box::new(Contact),
        "grooming" => Box::new(Grooming),
        "rearing" => Box::new(Rearing),
        "zones" => Box::new(Zones),
        _ => return None,
    })
}

fn and(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

fn or(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

fn state_summary(name: &str, states: &[Option<bool>], fps: f64) -> Vec<(String, Value)> {
    let t = totals(states, fps);
    vec![
        (format!("{name}_s"), json!(t.seconds)),
        (format!("{name}_bouts"), json!(t.bouts)),
        (format!("{name}_latency_s"), json!(t.latency)),
    ]
}

/// Entries into each label, and the seconds spent in it.
fn label_summary(
    labels: &[Option<String>],
    fps: f64,
) -> BTreeMap<String, (f64, usize, Option<f64>)> {
    let mut out: BTreeMap<String, (f64, usize, Option<f64>)> = BTreeMap::new();
    let mut prev: Option<&String> = None;
    for (t, l) in labels.iter().enumerate() {
        let Some(l) = l else {
            prev = None;
            continue;
        };
        let e = out.entry(l.clone()).or_insert((0.0, 0, None));
        e.0 += 1.0 / fps;
        if prev != Some(l) {
            e.1 += 1;
            e.2.get_or_insert(t as f64 / fps);
        }
        prev = Some(l);
    }
    out
}

// MARK: Floor positions

/// Positions on the floor: `(u, v)` in [0, 1]² along its length and width, from the 3D points (box
/// units; `floor` is `None`) or from the top view through a floor homography, plus the body length
/// and wall distances in the floor's metric units (box units, or pixels along its sides).
struct OnFloor<'a> {
    ctx: &'a Context<'a>,
    floor: Option<Floor>,
    size: [f64; 2],
    bl: f64,
}

impl<'a> OnFloor<'a> {
    fn new(ctx: &'a Context<'a>, floor: Option<Floor>) -> Self {
        let size = floor
            .as_ref()
            .map_or([1.0, ctx.cfg.arena.width], |f| f.size);
        let mut on = Self {
            ctx,
            floor,
            size,
            bl: f64::NAN,
        };
        on.bl = if on.floor.is_none() {
            ctx.features.bl
        } else {
            let lengths: Vec<f64> = (0..ctx.n())
                .filter_map(|t| {
                    let (a, b) = (
                        on.metric(on.uv(t, "nose")?),
                        on.metric(on.uv(t, "tail_base")?),
                    );
                    Some((a[0] - b[0]).hypot(a[1] - b[1]))
                })
                .collect();
            median(&lengths).unwrap_or(f64::NAN)
        };
        on
    }

    fn uv(&self, t: usize, name: &str) -> Option<Pt> {
        match &self.floor {
            None => self
                .ctx
                .at(t, name)
                .map(|p| box_uv([p[0], p[1]], self.ctx.cfg.arena.width)),
            Some(f) => self.ctx.top(t, name).map(|(p, _)| f.uv(p)),
        }
    }

    fn metric(&self, p: Pt) -> Pt {
        [p[0] * self.size[0], p[1] * self.size[1]]
    }

    /// Distance from `p` (floor `(u, v)`) to the nearest wall, in body lengths.
    fn wall_bl(&self, p: Pt) -> f64 {
        let m = self.metric(p);
        [m[0], self.size[0] - m[0], m[1], self.size[1] - m[1]]
            .into_iter()
            .fold(f64::INFINITY, f64::min)
            / self.bl
    }
}

// MARK: Open field

/// `center` (the middle `openfield.center_frac` of the floor's width and height) or `periphery`.
pub struct OpenField;

impl Module for OpenField {
    fn name(&self) -> &'static str {
        "open_field"
    }

    fn needs_layout(&self) -> Option<&'static str> {
        Some("floor")
    }

    fn run(&self, ctx: &Context) -> Result<Outcome> {
        let cfg = &ctx.cfg.openfield;
        let n = ctx.n();
        let width = ctx.cfg.arena.width;
        // Floor positions in metric floor units (box units, or pixels along the floor's sides).
        let mut notes = Vec::new();
        let floor: Option<Floor> = match (&ctx.input.rig, ctx.pose.dims3) {
            (_, true) => None,
            (Some(rig), false) => {
                let heights: Vec<f64> = ctx
                    .input
                    .world
                    .iter()
                    .flatten()
                    .skip(ctx.input.index("spine_mid").unwrap_or(0))
                    .step_by(ctx.input.kpts().max(1))
                    .map(|p| p[2])
                    .filter(|z| z.is_finite())
                    .collect();
                Some(Floor::from_camera(
                    &rig.top,
                    width,
                    median(&heights).unwrap_or(0.0),
                )?)
            }
            (None, false) => match ctx.layout.filter(|l| l.floor.len() == 4) {
                Some(l) => Some(Floor::from_corners(&l.floor)?),
                None => {
                    anyhow::bail!("the open field needs a camera model or a floor in the layout")
                }
            },
        };
        if floor.is_some() {
            notes.push("open field: positions from the top view (no valid 3D)".into());
        }
        let on = OnFloor::new(ctx, floor);
        let (uv, metric, bl) = (|t, name| on.uv(t, name), |p| on.metric(p), on.bl);
        let lo = 0.5 - cfg.center_frac / 2.0;
        let hi = 0.5 + cfg.center_frac / 2.0;
        let inside = |p: Pt| (lo..=hi).contains(&p[0]) && (lo..=hi).contains(&p[1]);
        let spine: Vec<Option<Pt>> = (0..n).map(|t| uv(t, "spine_mid")).collect();
        let raw: Vec<Option<bool>> = (0..n)
            .map(|t| {
                let s = spine[t]?;
                let head = if cfg.strict {
                    uv(t, "head_center").map(inside)
                } else {
                    Some(true)
                };
                and(Some(inside(s)), head)
            })
            .collect();
        let center = dwell(&raw, frames(cfg.dwell_s, ctx.fps()));
        let zone: Vec<Option<String>> = center
            .iter()
            .map(|c| c.map(|c| if c { "center" } else { "periphery" }.to_owned()))
            .collect();
        let wall: Vec<Option<f64>> = spine.iter().map(|s| s.map(|p| on.wall_bl(p))).collect();
        let (mut d_center, mut d_periphery) = (0.0, 0.0);
        for t in 1..n {
            if let (Some(a), Some(b), Some(c)) = (spine[t - 1], spine[t], center[t]) {
                let (a, b) = (metric(a), metric(b));
                let d = (a[0] - b[0]).hypot(a[1] - b[1]) / bl;
                if c {
                    d_center += d;
                } else {
                    d_periphery += d;
                }
            }
        }
        let t = totals(&center, ctx.fps());
        let walls: Vec<f64> = wall.iter().flatten().copied().collect();
        Ok(Outcome {
            columns: vec![
                Column::Text("zone".into(), zone.clone()),
                Column::Num("u".into(), spine.iter().map(|s| s.map(|s| s[0])).collect()),
                Column::Num("v".into(), spine.iter().map(|s| s.map(|s| s[1])).collect()),
                Column::Num("wall_dist_BL".into(), wall),
            ],
            location: Some(zone),
            summary: vec![
                ("center_s".into(), json!(t.seconds)),
                ("center_entries".into(), json!(t.bouts)),
                ("center_latency_s".into(), json!(t.latency)),
                ("distance_center_BL".into(), json!(d_center)),
                ("distance_periphery_BL".into(), json!(d_periphery)),
                (
                    "mean_wall_dist_BL".into(),
                    json!(
                        (!walls.is_empty()).then(|| walls.iter().sum::<f64>() / walls.len() as f64)
                    ),
                ),
            ],
            notes,
            ..Outcome::default()
        })
    }
}

// MARK: Plus maze

/// `open_arm` / `closed_arm` / `center`, with the four-paw entry rule and head dips.
pub struct PlusMaze;

impl Module for PlusMaze {
    fn name(&self) -> &'static str {
        "plus_maze"
    }

    fn needs_layout(&self) -> Option<&'static str> {
        Some("plus_maze")
    }

    fn run(&self, ctx: &Context) -> Result<Outcome> {
        let cfg = &ctx.cfg.maze;
        let layout = ctx
            .layout
            .ok_or_else(|| anyhow::anyhow!("the plus maze has no zones"))?;
        anyhow::ensure!(
            layout.status == "ok",
            "the plus maze zones need a review ({}); edit zones/{}.json and set \"source\": \"manual\"",
            layout.reason,
            layout.rig
        );
        let n = ctx.n();
        let fps = ctx.fps();
        // Body length in top-image pixels, where the zones are.
        let lengths: Vec<f64> = (0..n)
            .filter_map(|t| {
                let (a, b) = (ctx.top(t, "nose")?.0, ctx.top(t, "tail_base")?.0);
                Some((a[0] - b[0]).hypot(a[1] - b[1]))
            })
            .collect();
        let bl = median(&lengths).unwrap_or(f64::NAN);
        let arms: Vec<_> = layout
            .zones
            .iter()
            .filter(|z| z.kind.ends_with("_arm"))
            .collect();
        let center = layout.zones.iter().find(|z| z.kind == "center");
        let paws = [
            "left_front_paw",
            "right_front_paw",
            "left_hind_paw",
            "right_hind_paw",
        ];
        let mut zone_raw: Vec<Option<String>> = Vec::with_capacity(n);
        let mut prev: Option<String> = None;
        for t in 0..n {
            let observed: Vec<Option<(Pt, f64)>> = paws.iter().map(|p| ctx.top(t, p)).collect();
            let confident = observed
                .iter()
                .all(|p| p.is_some_and(|(_, c)| c >= cfg.paw_conf));
            let arm_of = |pts: &[Pt]| {
                arms.iter()
                    .find(|a| pts.iter().all(|p| a.polygon.contains(*p)))
                    .map(|a| a.name.clone())
            };
            let spine = ctx.top(t, "spine_mid").map(|p| p.0);
            let found = if confident {
                arm_of(&observed.iter().flatten().map(|p| p.0).collect::<Vec<_>>())
            } else {
                match (spine, ctx.top(t, "tail_base")) {
                    (Some(s), Some((b, _))) => arm_of(&[s, b]),
                    _ => None,
                }
            }
            .or_else(|| {
                spine
                    .filter(|s| center.is_some_and(|c| c.polygon.contains(*s)))
                    .map(|_| "center".to_owned())
            });
            // Halfway across a boundary the previous zone is kept.
            let now = found.or_else(|| prev.clone());
            prev = now.clone();
            zone_raw.push(now);
        }
        let kind_of = |name: &str| {
            layout
                .zones
                .iter()
                .find(|z| z.name == name)
                .map(|z| z.kind.clone())
        };
        let zone: Vec<Option<String>> = zone_raw
            .iter()
            .map(|z| z.as_deref().and_then(kind_of))
            .collect();
        let head_dip: Vec<Option<bool>> = (0..n)
            .map(|t| {
                if zone[t].as_deref() != Some("open_arm") {
                    return Some(false);
                }
                let nose = ctx.top(t, "nose")?.0;
                Some(layout.outside_by(nose) > cfg.head_dip_bl * bl)
            })
            .collect();
        let per_kind = label_summary(&zone, fps);
        let per_arm = label_summary(&zone_raw, fps);
        let get = |k: &str| per_kind.get(k).copied().unwrap_or((0.0, 0, None));
        let entries = |kind: &str| {
            per_arm
                .iter()
                .filter(|(name, _)| kind_of(name).as_deref() == Some(kind))
                .map(|(_, v)| v.1)
                .sum::<usize>()
        };
        let (open, closed) = (get("open_arm"), get("closed_arm"));
        let (open_in, closed_in) = (entries("open_arm"), entries("closed_arm"));
        let mut distance = 0.0;
        for t in 1..n {
            if let (Some(a), Some(b)) = (ctx.top(t - 1, "spine_mid"), ctx.top(t, "spine_mid")) {
                distance += (a.0[0] - b.0[0]).hypot(a.0[1] - b.0[1]) / bl;
            }
        }
        let share = |a: f64, b: f64| (a + b > 0.0).then(|| 100.0 * a / (a + b));
        let dips = temporal::bouts(&head_dip).len();
        Ok(Outcome {
            columns: vec![
                Column::Text("zone".into(), zone.clone()),
                Column::Text("zone_raw".into(), zone_raw),
                Column::Bool("head_dip".into(), head_dip),
            ],
            location: Some(zone),
            summary: vec![
                ("open_arm_s".into(), json!(open.0)),
                ("closed_arm_s".into(), json!(closed.0)),
                ("center_s".into(), json!(get("center").0)),
                ("open_arm_entries".into(), json!(open_in)),
                ("closed_arm_entries".into(), json!(closed_in)),
                ("open_time_pct".into(), json!(share(open.0, closed.0))),
                (
                    "open_entries_pct".into(),
                    json!(share(open_in as f64, closed_in as f64)),
                ),
                ("open_arm_latency_s".into(), json!(open.2)),
                ("head_dips".into(), json!(dips)),
                ("distance_BL".into(), json!(distance)),
            ],
            ..Outcome::default()
        })
    }
}

// MARK: Generic zones

/// Where the animal is in the setup's arena (a hand-made layout, or the arena detector's): the
/// named zone its body is in (`zone`, kept for `zones.dwell_s` before it changes) and, when the
/// arena has a floor, its floor position `floor_u`, `floor_v` (0-1 along the floor's sides) and
/// distance to the nearest wall `floor_wall_BL`. Without an arena it adds nothing and says so.
pub struct Zones;

impl Module for Zones {
    fn name(&self) -> &'static str {
        "zones"
    }

    fn needs_layout(&self) -> Option<&'static str> {
        Some("zones")
    }

    fn run(&self, ctx: &Context) -> Result<Outcome> {
        let Some(layout) = ctx.layout else {
            return Ok(Outcome {
                notes: vec!["zones: no arena for this setup (add an arena detector to the profile, or a zones file)".into()],
                ..Outcome::default()
            });
        };
        let mut out = Outcome::default();
        if !layout.zones.is_empty() {
            let raw: Vec<Option<String>> = (0..ctx.n())
                .map(|t| {
                    let p = ctx.top(t, "spine_mid")?.0;
                    Some(
                        layout
                            .zone(p)
                            .map_or_else(|| "outside".to_owned(), |z| z.name.clone()),
                    )
                })
                .collect();
            let zone = dwell_label(&raw, frames(ctx.cfg.zones.dwell_s, ctx.fps()));
            for (name, (s, entries, latency)) in label_summary(&zone, ctx.fps()) {
                out.summary.push((format!("{name}_s"), json!(s)));
                out.summary
                    .push((format!("{name}_entries"), json!(entries)));
                out.summary
                    .push((format!("{name}_latency_s"), json!(latency)));
            }
            out.columns.push(Column::Text("zone".into(), zone.clone()));
            out.location = Some(zone);
        }
        if layout.floor.len() == 4 {
            match Floor::from_corners(&layout.floor) {
                Ok(floor) => {
                    let on = OnFloor::new(ctx, Some(floor));
                    let spine: Vec<Option<Pt>> =
                        (0..ctx.n()).map(|t| on.uv(t, "spine_mid")).collect();
                    let column = |f: &dyn Fn(Pt) -> f64| -> Vec<Option<f64>> {
                        spine.iter().map(|p| p.map(f)).collect()
                    };
                    out.columns
                        .push(Column::Num("floor_u".into(), column(&|p| p[0])));
                    out.columns
                        .push(Column::Num("floor_v".into(), column(&|p| p[1])));
                    out.columns.push(Column::Num(
                        "floor_wall_BL".into(),
                        column(&|p| on.wall_bl(p)),
                    ));
                }
                Err(err) => out
                    .notes
                    .push(format!("zones: no floor positions ({err:#})")),
            }
        }
        Ok(out)
    }
}

// MARK: Grooming

/// The grooming gates of section 9.1 over a centred window, then bouts and grooming type.
pub struct Grooming;

impl Module for Grooming {
    fn name(&self) -> &'static str {
        "grooming"
    }

    fn run(&self, ctx: &Context) -> Result<Outcome> {
        let g = &ctx.cfg.groom;
        let f = ctx.features;
        let n = ctx.n();
        let fps = ctx.fps();
        let half = frames(g.window_s, fps) / 2;
        let med = |v: &[Option<f64>], t: usize| window_median(v, t, half);
        let use_motion = fps >= g.min_motion_fps;
        // Motion inside the outline needs the video; without it the paw speed decides alone.
        let have_motion = f.motion.iter().any(Option::is_some);
        let mut notes = Vec::new();
        if !use_motion {
            notes.push(format!(
                "grooming: {fps:.0} fps is too slow for stroke motion; only the posture gates apply"
            ));
        } else if !have_motion {
            notes.push(
                "grooming: no motion inside the outline (video not at hand); paw speed only".into(),
            );
        }
        if !ctx.pose.dims3 {
            notes.push("grooming: 2D, the not-rearing gate uses no height".into());
        }
        let cmp = |v: Option<f64>, test: &dyn Fn(f64) -> bool| v.map(test);
        let raw: Vec<Option<bool>> = (0..n)
            .map(|t| {
                let still = cmp(med(&f.speed, t), &|s| s < g.speed_bl_s);
                let hunched = cmp(med(&f.compact, t), &|c| c < g.compact);
                let paws_moving = cmp(med(&f.paw_speed, t), &|s| s > g.paw_speed_bl_s);
                let moving = if use_motion && have_motion {
                    or(cmp(med(&f.motion, t), &|m| m > g.motion), paws_moving)
                } else if use_motion {
                    paws_moving
                } else {
                    Some(true)
                };
                let touching = or(
                    cmp(med(&f.paw_d, t), &|d| d <= g.paw_bl),
                    or(
                        cmp(med(&f.nose_body, t), &|d| d <= g.nose_body_bl),
                        cmp(med(&f.nose_tail, t), &|d| d <= g.nose_body_bl),
                    ),
                );
                let not_rearing = if ctx.pose.dims3 {
                    let rearing = and(
                        cmp(med(&f.nose_z, t), &|z| z >= ctx.cfg.rear.nose_z_bl),
                        cmp(med(&f.paw_d, t), &|d| d > ctx.cfg.rear.paw_bl),
                    );
                    rearing.map(|r| !r)
                } else {
                    Some(true)
                };
                [hunched, moving, touching, not_rearing]
                    .into_iter()
                    .fold(still, and)
            })
            .collect();
        let grooming = clean(raw, fps, g.merge_gap_s, g.min_bout_s);
        let kind: Vec<Option<String>> = (0..n)
            .map(|t| {
                if grooming[t] != Some(true) {
                    return Some(String::new());
                }
                let near = |v: Option<f64>, lim: f64| v.is_some_and(|x| x <= lim);
                let paws_at_nose = {
                    let both = ["left_front_paw", "right_front_paw"].iter().all(|p| {
                        match (ctx.at(t, p), ctx.at(t, "nose")) {
                            (Some(a), Some(b)) => dist(a, b) / f.bl <= g.type_paw_bl,
                            _ => false,
                        }
                    });
                    both && f.nose_z[t].map_or(true, |z| z >= g.type_sit_z_bl)
                };
                Some(
                    if paws_at_nose {
                        "paw_face"
                    } else if near(f.paw_ear[t], g.type_paw_bl)
                        && f.paw_d_rate[t].is_some_and(|r| r < 0.0)
                    {
                        "head"
                    } else if near(f.nose_body[t], g.nose_body_bl)
                        && f.compact[t].is_some_and(|c| c < g.type_body_compact)
                    {
                        "body"
                    } else if near(f.nose_tail[t], g.nose_body_bl) {
                        "tail_genital"
                    } else {
                        "unclassified"
                    }
                    .to_owned(),
                )
            })
            .collect();
        let mut summary = state_summary("grooming", &grooming, fps);
        for (name, (s, _, _)) in label_summary(&kind, fps) {
            if !name.is_empty() {
                summary.push((format!("groom_{name}_s"), json!(s)));
            }
        }
        notes.push("grooming: no trained classifier (section 11); grooming = grooming_rule".into());
        Ok(Outcome {
            columns: vec![
                Column::Bool("grooming_rule".into(), grooming.clone()),
                Column::Bool("grooming".into(), grooming.clone()),
                Column::Text("groom_type".into(), kind),
            ],
            states: vec![("grooming".into(), grooming)],
            summary,
            notes,
            ..Outcome::default()
        })
    }
}

// MARK: Rearing

/// Nose high with the front paws away from the face (on the wall); 3D only.
pub struct Rearing;

impl Module for Rearing {
    fn name(&self) -> &'static str {
        "rearing"
    }

    fn run(&self, ctx: &Context) -> Result<Outcome> {
        if !ctx.pose.dims3 {
            return Ok(Outcome {
                notes: vec!["rearing: needs 3D, not scored".into()],
                ..Outcome::default()
            });
        }
        let r = &ctx.cfg.rear;
        let f = ctx.features;
        let raw: Vec<Option<bool>> = (0..ctx.n())
            .map(|t| {
                and(
                    f.nose_z[t].map(|z| z >= r.nose_z_bl),
                    f.paw_d[t].map(|d| d > r.paw_bl),
                )
            })
            .collect();
        let rearing = clean(raw, ctx.fps(), r.merge_gap_s, r.min_bout_s);
        Ok(Outcome {
            columns: vec![Column::Bool("rearing".into(), rearing.clone())],
            summary: state_summary("rearing", &rearing, ctx.fps()),
            states: vec![("rearing".into(), rearing)],
            ..Outcome::default()
        })
    }
}

// MARK: Three-chamber contact

/// Both animals facing each other with their noses within `contact.dist_bl`, per cup.
pub struct Contact;

/// The side image outline of a cup: the convex hull of its rim and base circles.
pub fn cup_outline(cam: &Camera, cup: &Cup) -> Polygon {
    let mut pts: Vec<Pt> = Vec::new();
    for i in 0..48 {
        let a = i as f64 / 48.0 * std::f64::consts::TAU;
        for z in [0.0, cup.height] {
            pts.push(cam.project(&[
                cup.center[0] + cup.radius * a.cos(),
                cup.center[1] + cup.radius * a.sin(),
                z,
            ]));
        }
    }
    Polygon(hull(pts))
}

fn hull(mut p: Vec<Pt>) -> Vec<Pt> {
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let cross = |o: Pt, a: Pt, b: Pt| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut lower: Vec<Pt> = Vec::new();
    for &q in &p {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], q) <= 0.0 {
            lower.pop();
        }
        lower.push(q);
    }
    let mut upper: Vec<Pt> = Vec::new();
    for &q in p.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], q) <= 0.0 {
            upper.pop();
        }
        upper.push(q);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// The 3D point a side-camera pixel shows on the cup's wire: of the two crossings of its ray with
/// the cylinder, the one nearer `towards`. A ray that misses the wire (the animal is inside the
/// cup) lands on the vertical plane through the cup axis facing the camera; `false` says so.
pub fn on_cup(cam: &Camera, px: Pt, cup: &Cup, towards: Vec3) -> Option<(Vec3, bool)> {
    let xn = cam.undistort_point(px);
    let o = cam.center();
    let d = mulv3(&transpose3(&cam.r), &[xn[0], xn[1], 1.0]);
    let (ox, oy) = (o[0] - cup.center[0], o[1] - cup.center[1]);
    let a = d[0] * d[0] + d[1] * d[1];
    let b = 2.0 * (ox * d[0] + oy * d[1]);
    let c = ox * ox + oy * oy - cup.radius * cup.radius;
    let disc = b * b - 4.0 * a * c;
    let at = |s: f64| [o[0] + s * d[0], o[1] + s * d[1], o[2] + s * d[2]];
    if a > 0.0 && disc >= 0.0 {
        let hits: Vec<Vec3> = [
            (-b - disc.sqrt()) / (2.0 * a),
            (-b + disc.sqrt()) / (2.0 * a),
        ]
        .into_iter()
        .filter(|s| *s > 0.0)
        .map(at)
        .filter(|p| (0.0..=cup.height).contains(&p[2]))
        .collect();
        if let Some(best) = hits
            .into_iter()
            .min_by(|p, q| dist(*p, towards).total_cmp(&dist(*q, towards)))
        {
            return Some((best, true));
        }
    }
    // The vertical plane through the axis, normal toward the camera.
    let nrm = [o[0] - cup.center[0], o[1] - cup.center[1]];
    let denom = nrm[0] * d[0] + nrm[1] * d[1];
    if denom.abs() < 1e-12 {
        return None;
    }
    let s = -(ox * nrm[0] + oy * nrm[1]) / denom;
    (s > 0.0).then(|| (at(s), false))
}

impl Module for Contact {
    fn name(&self) -> &'static str {
        "contact"
    }

    fn run(&self, ctx: &Context) -> Result<Outcome> {
        let cfg = ctx.cfg;
        let side = ctx.input.rig.as_ref().and_then(|r| r.side.as_ref());
        let (Some(side), true, false) = (side, ctx.pose.dims3, cfg.cups.is_empty()) else {
            return Ok(Outcome {
                notes: vec![
                    "contact: needs 3D, a side view and the cups (assay.cups); not scored".into(),
                ],
                ..Outcome::default()
            });
        };
        let n = ctx.n();
        let fps = ctx.fps();
        let k = ctx.input.kpts();
        let idx = |name: &str| ctx.input.index(name);
        let (Some(nose), Some(le), Some(re), Some(spine)) = (
            idx("nose"),
            idx("left_ear"),
            idx("right_ear"),
            idx("spine_mid"),
        ) else {
            anyhow::bail!("contact needs nose, ears and spine_mid keypoints");
        };
        let mut out = Outcome::default();
        let mut any: Vec<Option<bool>> = vec![Some(false); n];
        for cup in &cfg.cups {
            let outline = cup_outline(side, cup);
            let mine: Vec<&Track2d> = ctx
                .input
                .side_others
                .iter()
                .filter(|t| {
                    let seen: Vec<bool> = (t.first..t.first + t.rows)
                        .filter_map(|f| t.point(f, spine, k).map(|(p, _)| outline.contains(p)))
                        .collect();
                    !seen.is_empty()
                        && seen.iter().filter(|s| **s).count() as f64
                            >= cfg.cage.inside_frac * seen.len() as f64
                })
                .collect();
            let seen = (0..n)
                .filter(|t| {
                    mine.iter().any(|tr| {
                        tr.point(*t, nose, k).is_some() || tr.point(*t, spine, k).is_some()
                    })
                })
                .count();
            let occupied = n > 0 && seen as f64 >= cfg.cage.occupied_frac * n as f64;
            let name = &cup.name;
            let mut caged: Vec<Option<Vec3>> = vec![None; n];
            let mut test_faces: Vec<Option<bool>> = vec![None; n];
            let mut cage_faces: Vec<Option<bool>> = vec![None; n];
            let mut gap: Vec<Option<f64>> = vec![None; n];
            // Facing on both sides, and the nose-to-nose distance (∞ when not facing).
            let mut facing_dist: Vec<Option<f64>> = vec![None; n];
            for t in 0..n {
                if !occupied {
                    facing_dist[t] = Some(f64::INFINITY);
                    continue;
                }
                let Some(nt) = ctx.pose.at(t, Some(nose)) else {
                    continue;
                };
                let best = mine
                    .iter()
                    .filter_map(|tr| {
                        let (p, c) = tr.point(t, nose, k)?;
                        let (l, lc) = tr.point(t, le, k)?;
                        let (r, rc) = tr.point(t, re, k)?;
                        (c >= cfg.cage.kp_conf
                            && lc >= cfg.cage.ear_conf
                            && rc >= cfg.cage.ear_conf)
                            .then_some((p, c, l, r))
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1));
                let Some((pc, _, l, r)) = best else { continue };
                let Some((nc, on_wire)) = on_cup(side, pc, cup, nt) else {
                    continue;
                };
                caged[t] = Some(nc);
                let head_t = ctx.features.head[t];
                let tf = head_t.and_then(|h| {
                    angle([h[0], h[1], 0.0], [nc[0] - nt[0], nc[1] - nt[1], 0.0])
                        .map(|a| a < cfg.contact.facing_deg)
                });
                let ptn = side.project(&nt);
                let head_c = [
                    pc[0] - (l[0] + r[0]) / 2.0,
                    pc[1] - (l[1] + r[1]) / 2.0,
                    0.0,
                ];
                let same_side = (nc[0] - cup.center[0]) * (nt[0] - cup.center[0])
                    + (nc[1] - cup.center[1]) * (nt[1] - cup.center[1])
                    > 0.0;
                let cf = angle(head_c, [ptn[0] - pc[0], ptn[1] - pc[1], 0.0])
                    .map(|a| a < cfg.contact.facing_deg && same_side);
                test_faces[t] = tf;
                cage_faces[t] = cf;
                let d = dist(nt, nc) / ctx.features.bl;
                gap[t] = Some(d);
                facing_dist[t] = match and(tf, cf) {
                    Some(true) if on_wire => Some(d),
                    Some(_) => Some(f64::INFINITY),
                    None => None,
                };
            }
            let contact_at = |enter: f64| {
                clean(
                    hysteresis(&facing_dist, enter, enter * cfg.contact.exit_factor),
                    fps,
                    cfg.contact.merge_gap_s,
                    cfg.contact.min_bout_s,
                )
            };
            let contact = contact_at(cfg.contact.dist_bl);
            for t in 0..n {
                any[t] = or(any[t], contact[t]);
            }
            out.columns.extend([
                Column::Bool(format!("cup_{name}_occupied"), vec![Some(occupied); n]),
                Column::Num(
                    format!("caged_{name}_x"),
                    caged.iter().map(|p| p.map(|p| p[0])).collect(),
                ),
                Column::Num(
                    format!("caged_{name}_y"),
                    caged.iter().map(|p| p.map(|p| p[1])).collect(),
                ),
                Column::Num(
                    format!("caged_{name}_z"),
                    caged.iter().map(|p| p.map(|p| p[2])).collect(),
                ),
                Column::Bool(format!("test_faces_{name}"), test_faces),
                Column::Bool(format!("caged_faces_{name}"), cage_faces),
                Column::Num(format!("nose_nose_{name}_BL"), gap),
                Column::Bool(format!("mutual_contact_{name}"), contact.clone()),
            ]);
            out.summary.push((
                format!("cup_{name}"),
                json!(if occupied { "occupied" } else { "empty" }),
            ));
            out.summary.extend(state_summary(
                &format!("mutual_contact_{name}"),
                &contact,
                fps,
            ));
            for s in &cfg.contact.sensitivity_bl {
                let seconds = totals(&contact_at(*s), fps).seconds;
                out.summary
                    .push((format!("contact_{name}_at_{s:.2}BL_s"), json!(seconds)));
            }
        }
        out.summary
            .extend(state_summary("mutual_contact", &any, fps));
        out.states.push(("mutual_contact".into(), any));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ray_lands_on_the_near_side_of_the_cup() {
        // A side camera 2 units out along -y, level, looking along +y.
        let cam = Camera::new(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [0.0, 0.05, 2.0],
            &[],
            (640, 480),
            "side",
        );
        let cup = Cup {
            name: "L".into(),
            center: [0.0, 0.0],
            radius: 0.1,
            height: 0.2,
        };
        let px = cam.project(&[0.0, -0.1, 0.05]);
        let (p, wire) = on_cup(&cam, px, &cup, [0.0, -0.5, 0.05]).unwrap();
        assert!(wire);
        assert!((p[1] + 0.1).abs() < 1e-6, "{p:?}");
        // Asked for the far side, the far crossing is taken.
        let (far, _) = on_cup(&cam, px, &cup, [0.0, 0.5, 0.05]).unwrap();
        assert!((far[1] - 0.1).abs() < 1e-6, "{far:?}");
        // A ray past the cylinder lands on the axis plane, and cannot be in contact.
        let (_, wire) = on_cup(
            &cam,
            cam.project(&[0.5, 0.0, 0.05]),
            &cup,
            [0.0, -0.5, 0.05],
        )
        .unwrap();
        assert!(!wire);
        let outline = cup_outline(&cam, &cup);
        assert!(outline.contains(cam.project(&[0.0, -0.1, 0.1])));
        assert!(!outline.contains(cam.project(&[0.4, -0.1, 0.1])));
    }
}
