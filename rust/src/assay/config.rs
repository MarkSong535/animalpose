//! Every threshold of the experiment layer, read from `assay.*` in the pipeline config. Nothing is
//! a constant in the code: a key that is not set takes the default listed here.

use serde::Serialize;

use crate::data::config::Config;

/// A cup of the three-chamber box (box units).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Cup {
    /// Column suffix: `L`, `R`, …
    pub name: String,
    /// Centre of the cup on the floor `(x, y)`.
    pub center: [f64; 2],
    /// Radius of the wire cylinder.
    pub radius: f64,
    /// Height of the rim.
    pub height: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Contact {
    /// Nose-to-nose distance that starts contact (BL).
    pub dist_bl: f64,
    /// Contact ends above `dist_bl * exit_factor`.
    pub exit_factor: f64,
    /// Largest head-to-target angle, for both animals (degrees).
    pub facing_deg: f64,
    /// Extra distances reported in the summary (BL).
    pub sensitivity_bl: Vec<f64>,
    pub min_bout_s: f64,
    pub merge_gap_s: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Cage {
    /// Share of side frames a caged animal must be seen in for its cup to count as occupied.
    pub occupied_frac: f64,
    /// Smallest caged-nose confidence.
    pub kp_conf: f64,
    /// Smallest confidence of each caged ear.
    pub ear_conf: f64,
    /// Share of its frames a side track must spend inside a cup to be that cup's animal.
    pub inside_frac: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Maze {
    /// Frames in the median background.
    pub bg_frames: usize,
    /// Allowed deviation from 90° between the arms (degrees).
    pub axis_tol_deg: f64,
    /// Closed/open wall-score ratio needed to decide which axis is closed.
    pub wall_ratio: f64,
    /// Paw confidence for the four-paw entry rule.
    pub paw_conf: f64,
    /// Nose beyond the arm edge for a head dip (BL).
    pub head_dip_bl: f64,
    /// CLAHE clip limit and tiles per side.
    pub clahe_clip: f64,
    pub clahe_tiles: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenField {
    /// Centre width and height as a share of the floor.
    pub center_frac: f64,
    /// Time inside or outside before the zone changes (s).
    pub dwell_s: f64,
    /// Also require `head_center` inside the centre.
    pub strict: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Groom {
    /// Not travelling: median speed below this (BL/s).
    pub speed_bl_s: f64,
    /// Hunched: median `compact` below this.
    pub compact: f64,
    /// Share of outline pixels changing.
    pub motion: f64,
    /// Or front-paw speed relative to the body above this (BL/s).
    pub paw_speed_bl_s: f64,
    /// Paw-to-nose distance (BL).
    pub paw_bl: f64,
    /// Nose-to-body distance (BL).
    pub nose_body_bl: f64,
    /// The centred window the gates are evaluated over (s).
    pub window_s: f64,
    pub min_bout_s: f64,
    pub merge_gap_s: f64,
    /// Below this frame rate only the posture gates apply.
    pub min_motion_fps: f64,
    /// Grooming types: paw distance to nose or ear (BL), nose height for sitting up (BL), and
    /// compactness for body grooming.
    pub type_paw_bl: f64,
    pub type_sit_z_bl: f64,
    pub type_body_compact: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Rear {
    /// Nose height for rearing (BL).
    pub nose_z_bl: f64,
    /// Front paws at least this far from the nose (BL): on the wall, not at the face.
    pub paw_bl: f64,
    pub min_bout_s: f64,
    pub merge_gap_s: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Features {
    /// Savitzky–Golay window of the body speed (s).
    pub speed_window_s: f64,
    /// Window of the paw speed (s): short, so grooming strokes (3–8 Hz) are not smoothed away.
    pub paw_window_s: f64,
    /// BL is measured over frames whose speed is above this percentile…
    pub walk_speed_pct: f64,
    /// …and whose nose height is below this percentile.
    pub walk_nose_pct: f64,
    /// Longest keypoint gap that is filled (s).
    pub max_fill_s: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Motion {
    /// Grey levels from the background that make a pixel part of the outline.
    pub bg_diff: f64,
    /// Grey levels of change that make an outline pixel moving.
    pub frame_diff: f64,
    /// Interval the change is measured over (s).
    pub lag_s: f64,
    /// Frames in the median background.
    pub bg_frames: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Arena {
    /// The camera setup; its zones are cached as `zones/<rig>.json`.
    pub rig: String,
    /// Box width as a share of its length (box units).
    pub width: f64,
    /// Camera names of the top and side views; empty picks them from the camera directions.
    pub top_view: String,
    pub side_view: String,
    /// The arena detector (`checkpoints.arena`, kind `arena`) that finds this setup's floor and
    /// zones in each video; without one, the calibration or classical vision does.
    pub model: Option<std::path::PathBuf>,
    /// The SHA-256 the arena detector must have (`checksums.arena`), checked before it loads.
    pub model_sha256: Option<String>,
}

/// The experiment setup: its modules, their priorities and every threshold.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AssayConfig {
    /// The preset, or `custom`.
    pub experiment: String,
    /// Modules in evaluation order.
    pub modules: Vec<String>,
    /// Behavior states, highest priority first.
    pub priority: Vec<String>,
    /// Valid 3D share needed for 3D-based states.
    pub min_valid3d: f64,
    pub contact: Contact,
    pub cage: Cage,
    pub maze: Maze,
    pub openfield: OpenField,
    pub groom: Groom,
    pub rear: Rear,
    pub features: Features,
    pub motion: Motion,
    pub arena: Arena,
    pub zones: Zones,
    pub cups: Vec<Cup>,
}

/// The `zones` module: named regions of the arena.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Zones {
    /// Time in a zone (or out of it) before the zone changes (s).
    pub dwell_s: f64,
}

/// The experiments the app offers, and the modules each one runs.
pub const PRESETS: [(&str, &[&str]); 4] = [
    ("open_field", &["open_field", "grooming", "rearing"]),
    ("plus_maze", &["plus_maze", "grooming", "rearing"]),
    (
        "three_chamber",
        &["contact", "zones", "grooming", "rearing"],
    ),
    ("grooming", &["zones", "grooming", "rearing"]),
];

impl AssayConfig {
    /// The experiment in `cfg`, or `None` when the config sets none.
    pub fn from_config(cfg: &Config) -> Option<Self> {
        let experiment = cfg.str_at("assay.experiment", "");
        let experiment = experiment.trim();
        if experiment.is_empty() || experiment == "none" {
            return None;
        }
        Some(Self::read(cfg, experiment))
    }

    fn read(cfg: &Config, experiment: &str) -> Self {
        let f = |key: &str, default: f64| cfg.f64_at(&format!("assay.{key}"), default);
        let modules = cfg.opt_str_list_at("assay.modules").unwrap_or_else(|| {
            PRESETS
                .iter()
                .find(|(name, _)| *name == experiment)
                .map(|(_, modules)| modules.iter().map(|m| (*m).to_owned()).collect())
                .unwrap_or_default()
        });
        let priority = cfg.opt_str_list_at("assay.priority").unwrap_or_else(|| {
            ["mutual_contact", "social", "grooming", "rearing"]
                .map(str::to_owned)
                .to_vec()
        });
        let cups = cfg
            .get("assay")
            .and_then(|a| a.get("cups"))
            .and_then(|c| c.as_list())
            .unwrap_or_default()
            .iter()
            .enumerate()
            .filter_map(|(i, cup)| {
                let center = cup.get("center")?.f64_list();
                (center.len() == 2).then(|| Cup {
                    name: cup.get("name").and_then(|n| n.as_str()).map_or_else(
                        || ["L", "R", "C"].get(i).unwrap_or(&"X").to_string(),
                        str::to_owned,
                    ),
                    center: [center[0], center[1]],
                    radius: cup.get("radius").and_then(|v| v.as_f64()).unwrap_or(0.08),
                    height: cup.get("height").and_then(|v| v.as_f64()).unwrap_or(0.15),
                })
            })
            .collect();
        Self {
            experiment: experiment.to_owned(),
            modules,
            priority,
            min_valid3d: f("qa.min_valid3d", 0.8),
            contact: Contact {
                dist_bl: f("contact.dist_bl", 0.05),
                exit_factor: f("contact.exit_factor", 1.5),
                facing_deg: f("contact.facing_deg", 60.0),
                sensitivity_bl: cfg
                    .f64_list_at("assay.contact.sensitivity_bl", &[0.10, 0.20, 0.30]),
                min_bout_s: f("contact.min_bout_s", 0.2),
                merge_gap_s: f("contact.merge_gap_s", 0.2),
            },
            cage: Cage {
                occupied_frac: f("cage.occupied_frac", 0.20),
                kp_conf: f("cage.kp_conf", 0.5),
                ear_conf: f("cage.ear_conf", 0.3),
                inside_frac: f("cage.inside_frac", 0.5),
            },
            maze: Maze {
                bg_frames: cfg.usize_at("assay.maze.bg_frames", 200),
                axis_tol_deg: f("maze.axis_tol_deg", 10.0),
                wall_ratio: f("maze.wall_ratio", 1.5),
                paw_conf: f("maze.paw_conf", 0.5),
                head_dip_bl: f("maze.head_dip_bl", 0.1),
                clahe_clip: f("maze.clahe_clip", 2.0),
                clahe_tiles: cfg.usize_at("assay.maze.clahe_tiles", 8),
            },
            openfield: OpenField {
                center_frac: f("openfield.center_frac", 0.5),
                dwell_s: f("openfield.dwell_s", 0.2),
                strict: cfg.bool_at("assay.openfield.strict", false),
            },
            groom: Groom {
                speed_bl_s: f("groom.speed_bl_s", 0.25),
                compact: f("groom.compact", 0.8),
                motion: f("groom.motion", 0.08),
                paw_speed_bl_s: f("groom.paw_speed_bl_s", 0.5),
                paw_bl: f("groom.paw_bl", 0.25),
                nose_body_bl: f("groom.nose_body_bl", 0.3),
                window_s: f("groom.window_s", 1.0),
                min_bout_s: f("groom.min_bout_s", 1.5),
                merge_gap_s: f("groom.merge_gap_s", 0.5),
                min_motion_fps: f("groom.min_motion_fps", 15.0),
                type_paw_bl: f("groom.type_paw_bl", 0.2),
                type_sit_z_bl: f("groom.type_sit_z_bl", 0.3),
                type_body_compact: f("groom.type_body_compact", 0.6),
            },
            rear: Rear {
                nose_z_bl: f("rear.nose_z_bl", 0.65),
                paw_bl: f("rear.paw_bl", 0.4),
                min_bout_s: f("rear.min_bout_s", 0.3),
                merge_gap_s: f("rear.merge_gap_s", 0.2),
            },
            features: Features {
                speed_window_s: f("features.speed_window_s", 0.3),
                paw_window_s: f("features.paw_window_s", 0.1),
                walk_speed_pct: f("features.walk_speed_pct", 60.0),
                walk_nose_pct: f("features.walk_nose_pct", 50.0),
                max_fill_s: f("gap.max_fill_s", 0.5),
            },
            motion: Motion {
                bg_diff: f("motion.bg_diff", 25.0),
                frame_diff: f("motion.frame_diff", 15.0),
                lag_s: f("motion.lag_s", 0.2),
                bg_frames: cfg.usize_at("assay.motion.bg_frames", 200),
            },
            arena: Arena {
                rig: cfg.str_at("assay.arena.rig", experiment),
                width: f("arena.width", 1.0),
                top_view: cfg.str_at("assay.arena.top_view", ""),
                side_view: cfg.str_at("assay.arena.side_view", ""),
                model: cfg
                    .get("checkpoints.arena")
                    .and_then(|v| v.as_str())
                    .filter(|p| !p.trim().is_empty())
                    .map(|p| crate::cli::predict::checkpoint_path(cfg, p)),
                model_sha256: cfg
                    .get("checksums.arena")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_owned()),
            },
            zones: Zones {
                dwell_s: f("zones.dwell_s", 0.2),
            },
            cups,
        }
    }

    pub fn runs(&self, module: &str) -> bool {
        self.modules.iter().any(|m| m == module)
    }
}

/// The defaults, for documentation and the config editor.
pub fn defaults(experiment: &str) -> AssayConfig {
    AssayConfig::read(
        &Config::parse("{}", &[]).expect("an empty config parses"),
        experiment,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_and_overrides() {
        let cfg = Config::parse(
            "assay:\n  experiment: open_field\n  groom:\n    min_bout_s: 2.0\n  cups:\n    - {center: [-0.3, 0.0]}\n    - {name: Right, center: [0.3, 0.0], radius: 0.1}\n",
            &[],
        )
        .unwrap();
        let a = AssayConfig::from_config(&cfg).unwrap();
        assert_eq!(a.modules, ["open_field", "grooming", "rearing"]);
        assert_eq!(a.groom.min_bout_s, 2.0);
        assert_eq!(a.contact.dist_bl, 0.05);
        assert_eq!(a.cups[0].name, "L");
        assert_eq!(a.cups[1].name, "Right");
        assert_eq!(a.cups[1].radius, 0.1);
        assert!(AssayConfig::from_config(&Config::parse("{}", &[]).unwrap()).is_none());
        let custom = Config::parse(
            "assay:\n  experiment: home_cage\n  modules: [grooming]\n",
            &[],
        )
        .unwrap();
        assert_eq!(
            AssayConfig::from_config(&custom).unwrap().modules,
            ["grooming"]
        );
    }
}
