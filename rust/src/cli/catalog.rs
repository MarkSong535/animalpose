//! The full behavior catalogue and how the classifier's classes map into it.
//!
//! The port of `src/animalpose/behavior/catalog.py`: the 132 ethogram entries of
//! `behaviors.json` plus `other`, the alias table that maps a training class name (`sniffing`,
//! `grooming`, a dataset synonym) onto a catalogue name, the soft social rules of
//! `pair_probabilities`, and the probability mixing of `full_probabilities`.

use std::collections::HashMap;

use crate::algos::kinematics::Kinematics;
use serde::Deserialize;
use serde_json::json;

/// `catalog.py:CATALOG` minus `other`, as `behaviors.json` writes it.
const BEHAVIORS_JSON: &str = include_str!("../../../src/animalpose/behavior/behaviors.json");

/// `catalog.py:RUN_SPEED`, body lengths per second above which walking becomes running.
pub const RUN_SPEED: f64 = 2.0;

/// `catalog.py:PAIR_RULES`, in the order `full_probabilities` fills them.
pub const PAIR_RULES: [&str; 9] = [
    "sniff_face",
    "sniff_anogenital",
    "sniff_body",
    "approach",
    "follow",
    "chase",
    "flee",
    "huddle",
    "attack",
];

/// One catalogue entry (`catalog.py:Behavior`).
#[derive(Clone, Debug, Deserialize)]
pub struct Behavior {
    /// The canonical name, and the column `behavior.csv` prints.
    pub name: String,
    /// The ethogram group (`locomotion`, `social`, ...).
    pub group: String,
    /// Whether the entry needs a second animal.
    #[serde(default)]
    pub social: bool,
    /// The published definition.
    #[serde(default)]
    pub definition: String,
    /// Dataset spellings of the same behavior.
    #[serde(default)]
    pub synonyms: Vec<String>,
    /// Where the entry came from.
    #[serde(default)]
    pub sources: Vec<String>,
    /// How well a top view can see it.
    #[serde(default)]
    pub detectable_top: String,
    /// How well a side view can see it.
    #[serde(default)]
    pub detectable_side: String,
}

/// The catalogue: `CATALOG`, `NAMES`, `SOCIAL` and `ALIASES` of `catalog.py`.
#[derive(Clone, Debug)]
pub struct Catalog {
    /// Every entry, `other` last.
    pub behaviors: Vec<Behavior>,
    /// `NAMES`: one column per entry, in `behaviors` order.
    pub names: Vec<String>,
    /// `ALIASES`: a classifier class or dataset label to its catalogue name.
    pub aliases: HashMap<String, String>,
}

#[derive(Deserialize)]
struct RawBehavior {
    name: String,
    #[serde(default)]
    group: String,
    #[serde(default)]
    social: bool,
    #[serde(default)]
    definition: String,
    #[serde(default)]
    synonyms: Vec<String>,
    #[serde(default)]
    sources: Vec<String>,
    #[serde(default)]
    detectable_top: String,
    #[serde(default)]
    detectable_side: String,
}

impl Default for Catalog {
    fn default() -> Self {
        Self::load()
    }
}

impl Catalog {
    /// `catalog.py:_load()`: `behaviors.json` plus the `other` entry.
    pub fn load() -> Self {
        let raw: Vec<RawBehavior> = serde_json::from_str(BEHAVIORS_JSON)
            .expect("behaviors.json is compiled into the binary and must parse");
        let mut behaviors: Vec<Behavior> = raw
            .into_iter()
            .map(|r| Behavior {
                name: r.name,
                group: r.group,
                social: r.social,
                definition: r.definition,
                synonyms: r.synonyms,
                sources: r.sources,
                detectable_top: r.detectable_top,
                detectable_side: r.detectable_side,
            })
            .collect();
        behaviors.push(Behavior {
            name: "other".to_string(),
            group: "other".to_string(),
            social: false,
            definition: "labelled activity outside the catalogue (object play, ...)".to_string(),
            synonyms: Vec::new(),
            sources: Vec::new(),
            detectable_top: String::new(),
            detectable_side: String::new(),
        });

        // `ALIASES` starts from the hand-written table, then every synonym word of the catalogue is
        // added in file order with `setdefault`, so the first entry that mentions a word wins.
        let mut aliases: HashMap<String, String> = [
            ("resting", "rest"),
            ("immobile", "rest"),
            ("sniffing", "sniff_environment"),
            ("exploring", "sniff_environment"),
            ("walking", "walk"),
            ("locomotion", "walk"),
            ("running", "run"),
            ("turning", "turn"),
            ("rearing", "rear"),
            ("grooming", "self_groom"),
            ("self_grooming", "self_groom"),
            ("eating", "eat"),
            ("drinking", "drink"),
            ("digging", "dig"),
            ("nesting", "nest_build"),
            ("climbing", "climb"),
            ("scratching", "scratch"),
            ("freezing", "freeze"),
            ("jumping", "jump"),
            ("stretch_attend", "stretched_attend_posture"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        for b in &behaviors {
            for synonym in &b.synonyms {
                for alt in synonym.split('(').next().unwrap_or("").split('/') {
                    let alt = alt.trim();
                    if alt.is_empty() {
                        continue;
                    }
                    aliases.entry(key(alt)).or_insert_with(|| b.name.clone());
                }
            }
        }
        let names = behaviors.iter().map(|b| b.name.clone()).collect();
        Self {
            behaviors,
            names,
            aliases,
        }
    }

    /// `catalog.py:canonical`: the catalogue name of a classifier class, or `None`.
    pub fn canonical(&self, name: &str) -> Option<usize> {
        if let Some(at) = self.names.iter().position(|n| n == name) {
            return Some(at);
        }
        self.aliases
            .get(&key(name))
            .and_then(|n| self.names.iter().position(|m| m == n))
    }

    /// The column of a catalogue name.
    pub fn column(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// `catalog.py:full_probabilities`: `(frames, len(NAMES))` probabilities summing to 1 per frame,
    /// plus which catalogue entries this run models.
    ///
    /// * `probs`: the classifier's `(frames, classes.len())` probabilities.
    /// * `partners`: the kinematics of the other tracked animals, aligned to these frames (NaN rows
    ///   where that animal is absent).
    pub fn full_probabilities(
        &self,
        classes: &[String],
        probs: &[f64],
        frames: usize,
        kin: &Kinematics,
        partners: &[&Kinematics],
    ) -> (Vec<f64>, Vec<bool>) {
        let width = self.names.len();
        let mut full = vec![0.0f64; frames * width];
        let mut modelled = vec![false; width];
        for (j, class) in classes.iter().enumerate() {
            if let Some(at) = self.canonical(class) {
                for t in 0..frames {
                    full[t * width + at] += probs[t * classes.len() + j];
                }
                modelled[at] = true;
            }
        }
        // `run` is split off `walk` by the speed: a walking animal that moves at more than
        // RUN_SPEED body lengths per second is running.
        if self.column("walk").is_some_and(|c| modelled[c]) {
            let (walk, run) = (self.column("walk").unwrap(), self.column("run").unwrap());
            for t in 0..frames {
                let moved = full[t * width + walk] * sigmoid((kin.speed[t] - RUN_SPEED) / 0.3);
                full[t * width + run] += moved;
                full[t * width + walk] -= moved;
            }
            modelled[run] = true;
        }
        let solitary: Vec<usize> = (0..width).filter(|b| !self.behaviors[*b].social).collect();
        for t in 0..frames {
            let sum: f64 = solitary.iter().map(|b| full[t * width + b]).sum();
            for b in &solitary {
                full[t * width + *b] /= sum.max(1e-9);
            }
        }
        // Social rules: each partner votes, the strongest vote per rule wins per frame.
        let mut social = vec![0.0f64; frames * PAIR_RULES.len()];
        for partner in partners {
            let rules = pair_probabilities(kin, partner);
            for p in 0..PAIR_RULES.len() {
                for t in 0..frames {
                    social[t * PAIR_RULES.len() + p] =
                        social[t * PAIR_RULES.len() + p].max(rules[p][t]);
                }
            }
        }
        for rule in PAIR_RULES {
            if let Some(at) = self.column(rule) {
                modelled[at] = true;
            }
        }
        for t in 0..frames {
            // `s = 1 - prod(1 - social)`: the chance that any social rule fired.
            let s: f64 = 1.0
                - (0..PAIR_RULES.len())
                    .map(|p| 1.0 - social[t * PAIR_RULES.len() + p])
                    .product::<f64>();
            let total: f64 = (0..PAIR_RULES.len())
                .map(|p| social[t * PAIR_RULES.len() + p])
                .sum();
            for b in &solitary {
                full[t * width + *b] *= 1.0 - s;
            }
            for (p, name) in PAIR_RULES.iter().enumerate() {
                if let Some(at) = self.column(name) {
                    full[t * width + at] = social[t * PAIR_RULES.len() + p] / total.max(1e-9) * s;
                }
            }
        }
        (full, modelled)
    }

    /// `catalog.py:top_k`: the `k` most likely behaviors per frame, most likely first.
    pub fn top_k(&self, full: &[f64], frames: usize, k: usize) -> (Vec<usize>, Vec<f64>) {
        let width = self.names.len();
        let mut idx = Vec::with_capacity(frames * k);
        let mut probs = Vec::with_capacity(frames * k);
        for t in 0..frames {
            let mut order: Vec<usize> = (0..width).collect();
            // Python's `argsort(-full)` is a quicksort, which leaves ties in an arbitrary order;
            // the stable index tie-break is what makes this reproducible.
            order.sort_by(|a, b| {
                full[t * width + *b]
                    .partial_cmp(&full[t * width + *a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(b))
            });
            for at in order.into_iter().take(k) {
                idx.push(at);
                probs.push(full[t * width + at]);
            }
        }
        (idx, probs)
    }

    /// `catalog.py:describe`: what this run models and where each entry's probability comes from.
    pub fn describe(&self, modelled: &[bool]) -> serde_json::Value {
        let entries: Vec<serde_json::Value> = self
            .behaviors
            .iter()
            .zip(modelled)
            .map(|(b, m)| {
                let how = |name: &str| -> Option<&'static str> {
                    if PAIR_RULES.contains(&name) {
                        Some("pair rule")
                    } else if name == "run" {
                        Some("rule (fast walk)")
                    } else {
                        None
                    }
                };
                json!({
                    "name": b.name,
                    "group": b.group,
                    "social": b.social,
                    "definition": b.definition,
                    "synonyms": b.synonyms,
                    "sources": b.sources,
                    "detectable_top": b.detectable_top,
                    "detectable_side": b.detectable_side,
                    "modelled": *m,
                    "source": if *m { how(&b.name).unwrap_or("classifier") } else { "not modelled (no training data)" },
                })
            })
            .collect();
        serde_json::Value::Array(entries)
    }
}

/// `catalog.py:_key`.
fn key(name: &str) -> String {
    name.to_lowercase().replace([' ', '-'], "_")
}

/// `catalog.py:_sig`.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-30.0, 30.0)).exp())
}

fn dist(a: &[[f64; 2]], b: &[[f64; 2]], body_length: f64) -> Vec<f64> {
    a.iter()
        .zip(b)
        .map(|(p, q)| {
            (((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1])).sqrt()) / body_length
        })
        .collect()
}

/// `catalog.py:pair_probabilities`: the soft social-behavior probabilities of `a` towards `b`.
///
/// Returns one `(frames,)` vector per [`PAIR_RULES`] entry, in that order.
pub fn pair_probabilities(a: &Kinematics, b: &Kinematics) -> Vec<Vec<f64>> {
    let bl = a.length;
    let frames = a.speed.len();
    let d_nn = dist(&a.nose, &b.nose, bl);
    let d_ng = dist(&a.nose, &b.tail, bl);
    let d_nb = dist(&a.nose, &b.body, bl);
    let d_c = dist(&a.centre, &b.centre, bl);
    let sa = &a.speed;
    let sb = &b.speed;
    let rules = [
        // sniff_face
        (0..frames)
            .map(|t| sigmoid((0.5 - d_nn[t]) / 0.1))
            .collect::<Vec<_>>(),
        // sniff_anogenital
        (0..frames)
            .map(|t| sigmoid((0.5 - d_ng[t]) / 0.1) * sigmoid((d_nn[t] - 0.6) / 0.1))
            .collect::<Vec<_>>(),
        // sniff_body
        (0..frames)
            .map(|t| {
                sigmoid((0.4 - d_nb[t]) / 0.1)
                    * sigmoid((d_nn[t] - 0.6) / 0.1)
                    * sigmoid((d_ng[t] - 0.6) / 0.1)
            })
            .collect::<Vec<_>>(),
        // approach
        (0..frames)
            .map(|t| {
                sigmoid((3.0 - d_c[t]) / 0.5)
                    * sigmoid((d_c[t] - 1.0) / 0.2)
                    * sigmoid((-away(a, b, t) - 0.8) / 0.2)
            })
            .collect::<Vec<_>>(),
        // follow
        (0..frames)
            .map(|t| {
                trailing(a, b, t, bl)
                    * sigmoid((sa[t] - 0.8) / 0.2)
                    * sigmoid((sb[t] - 0.8) / 0.2)
                    * sigmoid((RUN_SPEED - sa[t]) / 0.4)
            })
            .collect::<Vec<_>>(),
        // chase
        (0..frames)
            .map(|t| {
                trailing(a, b, t, bl)
                    * sigmoid((sa[t] - RUN_SPEED) / 0.4)
                    * sigmoid((sb[t] - RUN_SPEED) / 0.4)
            })
            .collect::<Vec<_>>(),
        // flee
        (0..frames)
            .map(|t| {
                sigmoid((2.0 - d_c[t]) / 0.4)
                    * sigmoid((away(a, b, t) - RUN_SPEED) / 0.4)
                    * sigmoid((towards(b, a, t) - 0.5) / 0.3)
            })
            .collect::<Vec<_>>(),
        // huddle
        (0..frames)
            .map(|t| {
                sigmoid((0.9 - d_c[t]) / 0.15)
                    * sigmoid((0.2 - sa[t]) / 0.1)
                    * sigmoid((0.2 - sb[t]) / 0.1)
            })
            .collect::<Vec<_>>(),
        // attack
        (0..frames)
            .map(|t| {
                sigmoid((1.0 - d_c[t]) / 0.2)
                    * sigmoid((sa[t] - RUN_SPEED) / 0.4)
                    * sigmoid((sb[t] - RUN_SPEED) / 0.4)
                    * sigmoid((0.3 - align(a, b, t)) / 0.3)
            })
            .collect::<Vec<_>>(),
    ];
    rules
        .into_iter()
        .map(|v| v.into_iter().map(nan_to_num).collect())
        .collect()
}

/// `(a.axis . b.axis)`, the alignment of two body axes.
fn align(a: &Kinematics, b: &Kinematics, t: usize) -> f64 {
    a.axis[t][0] * b.axis[t][0] + a.axis[t][1] * b.axis[t][1]
}

/// `a`'s velocity along the centre-to-centre direction: positive when `a` moves away from `b`.
fn away(a: &Kinematics, b: &Kinematics, t: usize) -> f64 {
    let gap = [
        a.centre[t][0] - b.centre[t][0],
        a.centre[t][1] - b.centre[t][1],
    ];
    let dist = (gap[0] * gap[0] + gap[1] * gap[1]).sqrt() + 1e-6;
    (a.velocity[t][0] * gap[0] + a.velocity[t][1] * gap[1]) / dist
}

/// `b`'s velocity along the centre-to-centre direction: positive when `b` moves towards `a`.
fn towards(b: &Kinematics, a: &Kinematics, t: usize) -> f64 {
    -away(b, a, t)
}

/// `_sig((1.5 - d_c) / 0.3) * _sig((align - 0.5) / 0.15) * behind`: `a` trails `b` and faces it.
fn trailing(a: &Kinematics, b: &Kinematics, t: usize, bl: f64) -> f64 {
    let d_c = ((a.centre[t][0] - b.centre[t][0]).powi(2)
        + (a.centre[t][1] - b.centre[t][1]).powi(2))
    .sqrt()
        / bl;
    let behind = sigmoid(
        -((a.centre[t][0] - b.centre[t][0]) * b.axis[t][0]
            + (a.centre[t][1] - b.centre[t][1]) * b.axis[t][1])
            / bl
            / 0.2,
    );
    sigmoid((1.5 - d_c) / 0.3) * sigmoid((align(a, b, t) - 0.5) / 0.15) * behind
}

/// `np.nan_to_num` for one scalar.
fn nan_to_num(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v == f64::INFINITY {
        f64::MAX
    } else if v == f64::NEG_INFINITY {
        f64::MIN
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::kinematics::Kinematics;

    /// `CATALOG` is `behaviors.json` plus `other`, and the aliases resolve dataset labels.
    #[test]
    fn the_catalogue_has_132_entries_plus_other() {
        let catalog = Catalog::load();
        assert_eq!(catalog.names.len(), 133);
        assert_eq!(catalog.names[0], "walk");
        assert_eq!(catalog.names[132], "other");
        assert_eq!(
            catalog.canonical("sniffing"),
            catalog.column("sniff_environment")
        );
        assert_eq!(catalog.canonical("grooming"), catalog.column("self_groom"));
        assert_eq!(catalog.canonical("resting"), catalog.column("rest"));
        assert_eq!(catalog.canonical("nonsense-behavior"), None);
        assert_eq!(catalog.column("huddle").map(|_| true), Some(true));
    }

    fn kinematics(offset: f64, frames: usize) -> Kinematics {
        let centre: Vec<[f64; 2]> = (0..frames)
            .map(|i| [offset + f64::from(i as u32), 0.0])
            .collect();
        Kinematics {
            centre: centre.clone(),
            axis: vec![[1.0, 0.0]; frames],
            length: 10.0,
            speed: vec![1.2; frames],
            velocity: vec![[0.5, 0.0]; frames],
            nose: centre.iter().map(|c| [c[0] + 5.0, 0.0]).collect(),
            tail: centre.iter().map(|c| [c[0] - 5.0, 0.0]).collect(),
            body: centre.clone(),
        }
    }

    /// `catalog.py:pair_probabilities` for two animals 0.15 body lengths apart.
    #[test]
    fn pair_rules_match_python() {
        let rules = pair_probabilities(&kinematics(0.0, 3), &kinematics(1.5, 3));
        assert_eq!(rules.len(), PAIR_RULES.len());
        let want: [(&str, f64); 9] = [
            ("sniff_face", 0.970_688),
            ("sniff_anogenital", 0.000_322),
            ("sniff_body", 0.006_32),
            ("approach", 0.002_557),
            ("follow", 0.443_191),
            ("chase", 0.009_216),
            ("flee", 6.6e-5),
            ("huddle", 0.0),
            ("attack", 0.001_238),
        ];
        for (p, (name, value)) in want.into_iter().enumerate() {
            assert_eq!(PAIR_RULES[p], name);
            for (t, got) in rules[p].iter().enumerate() {
                assert!(
                    (got - value).abs() < 1e-6,
                    "{name} frame {t}: {got} vs {value}"
                );
            }
        }
    }

    /// `full_probabilities` normalises the solitary columns, splits `run` off `walk` by speed and
    /// fills the social columns from the rules.
    #[test]
    fn full_probabilities_are_a_distribution() {
        let catalog = Catalog::load();
        let width = catalog.names.len();
        let classes = [
            "resting".to_string(),
            "walking".to_string(),
            "rearing".to_string(),
            "grooming".to_string(),
            "eating".to_string(),
            "other".to_string(),
        ];
        let frames = 4;
        let probs: Vec<f64> = (0..frames * classes.len())
            .map(|i| 0.02 + 0.01 * f64::from(i as u32))
            .collect();
        let kin = kinematics(0.0, frames);
        let partners = [kinematics(1.5, frames)];
        let (full, modelled) = catalog.full_probabilities(
            &classes,
            &probs,
            frames,
            &kin,
            &partners.iter().collect::<Vec<_>>(),
        );
        assert_eq!(full.len(), frames * width);
        assert!(modelled[catalog.column("walk").unwrap()]);
        assert!(modelled[catalog.column("run").unwrap()]);
        assert!(modelled[catalog.column("chase").unwrap()]);
        assert!(modelled[catalog.column("other").unwrap()]);
        for t in 0..frames {
            let sum: f64 = full[t * width..(t + 1) * width].iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "frame {t} sums to {sum}");
        }
        // The classifier classes land on their catalogue columns: `resting` is `rest`.
        let (rest, walk) = (
            catalog.column("rest").unwrap(),
            catalog.column("walk").unwrap(),
        );
        assert!(full[rest] > 0.0);
        assert!(full[walk] > 0.0);
        // A fast walking animal (`speed = 1.2 < RUN_SPEED`) keeps most of its probability on `walk`.
        let run = catalog.column("run").unwrap();
        assert!(full[walk] > full[run]);
    }

    /// `top_k` orders by probability and breaks ties by catalogue index.
    #[test]
    fn top_k_orders_by_probability_then_index() {
        let catalog = Catalog::load();
        let width = catalog.names.len();
        let mut full = vec![0.0; width];
        full[3] = 0.5;
        full[1] = 0.5;
        full[0] = 0.1;
        let (idx, probs) = catalog.top_k(&full, 1, 3);
        assert_eq!(idx, [1, 3, 0]);
        assert_eq!(probs, [0.5, 0.5, 0.1]);
    }

    /// `describe` records where each column's probability comes from.
    #[test]
    fn describe_labels_the_source_of_each_column() {
        let catalog = Catalog::load();
        let width = catalog.names.len();
        let mut modelled = vec![false; width];
        modelled[catalog.column("walk").unwrap()] = true;
        modelled[catalog.column("run").unwrap()] = true;
        modelled[catalog.column("chase").unwrap()] = true;
        let info = catalog.describe(&modelled);
        let entries = info.as_array().expect("describe writes a list");
        assert_eq!(entries.len(), width);
        let source = |name: &str| {
            entries
                .iter()
                .find(|e| e["name"] == name)
                .and_then(|e| e["source"].as_str())
                .map(str::to_string)
        };
        assert_eq!(source("walk").as_deref(), Some("classifier"));
        assert_eq!(source("run").as_deref(), Some("rule (fast walk)"));
        assert_eq!(source("chase").as_deref(), Some("pair rule"));
        assert_eq!(
            source("huddle").as_deref(),
            Some("not modelled (no training data)")
        );
    }

    /// The run rule splits `walk` at `RUN_SPEED` body lengths per second.
    #[test]
    fn fast_walk_becomes_run() {
        let catalog = Catalog::load();
        let classes = ["walking".to_string()];
        let mut slow = kinematics(0.0, 2);
        slow.speed = vec![0.5, 0.5];
        let mut fast = kinematics(0.0, 2);
        fast.speed = vec![RUN_SPEED + 1.0, RUN_SPEED + 1.0];
        let probs = [1.0, 1.0];
        let (walk, run) = (
            catalog.column("walk").unwrap(),
            catalog.column("run").unwrap(),
        );
        let (full, _) = catalog.full_probabilities(&classes, &probs, 2, &slow, &[]);
        assert!(full[walk] > full[run]);
        let (full, _) = catalog.full_probabilities(&classes, &probs, 2, &fast, &[]);
        assert!(full[run] > full[walk]);
    }
}
