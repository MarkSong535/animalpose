//! States over time. A per-frame state is `Some(true)`, `Some(false)` or `None` (unknown: the
//! inputs the rule needs are missing). Unknown frames are never guessed; they keep their own
//! value through every step here.

/// Frames covering `seconds`, at least one.
pub fn frames(seconds: f64, fps: f64) -> usize {
    ((seconds * fps).round() as usize).max(1)
}

/// A distance rule with separate enter and exit thresholds: on at `value <= enter`, off again
/// only at `value > exit`.
pub fn hysteresis(values: &[Option<f64>], enter: f64, exit: f64) -> Vec<Option<bool>> {
    let mut on = false;
    values
        .iter()
        .map(|v| {
            let v = (*v)?;
            on = if on { v <= exit } else { v <= enter };
            Some(on)
        })
        .collect()
}

/// The state changes only after the opposite value has held for `dwell` frames.
pub fn dwell(raw: &[Option<bool>], dwell: usize) -> Vec<Option<bool>> {
    let mut state: Option<bool> = None;
    let mut run = 0;
    let mut pending = None;
    raw.iter()
        .map(|r| {
            let r = (*r)?;
            match state {
                None => state = Some(r),
                Some(s) if s == r => {
                    run = 0;
                    pending = None;
                }
                Some(_) => {
                    if pending == Some(r) {
                        run += 1;
                    } else {
                        pending = Some(r);
                        run = 1;
                    }
                    if run >= dwell {
                        state = Some(r);
                        run = 0;
                        pending = None;
                    }
                }
            }
            state
        })
        .collect()
}

/// A categorical version of [`dwell`]: the label changes after `dwell` frames of another label.
pub fn dwell_label(raw: &[Option<String>], dwell: usize) -> Vec<Option<String>> {
    let mut state: Option<String> = None;
    let mut pending: Option<(String, usize)> = None;
    raw.iter()
        .map(|r| {
            let Some(r) = r else { return None };
            match &state {
                None => state = Some(r.clone()),
                Some(s) if s == r => pending = None,
                Some(_) => {
                    let count = match &pending {
                        Some((p, n)) if p == r => n + 1,
                        _ => 1,
                    };
                    if count >= dwell {
                        state = Some(r.clone());
                        pending = None;
                    } else {
                        pending = Some((r.clone(), count));
                    }
                }
            }
            state.clone()
        })
        .collect()
}

/// Runs of `true`, as `[start, end)` frame ranges.
pub fn bouts(states: &[Option<bool>]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, s) in states.iter().enumerate() {
        match (s, start) {
            (Some(true), None) => start = Some(i),
            (Some(true), Some(_)) => {}
            (_, Some(b)) => {
                out.push((b, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(b) = start {
        out.push((b, states.len()));
    }
    out
}

/// Fills gaps of at most `max_gap` frames between two bouts. Unknown frames in such a gap are
/// filled too: the bout plainly continues through them.
pub fn merge_gaps(states: &mut [Option<bool>], max_gap: usize) {
    let found = bouts(states);
    for pair in found.windows(2) {
        let (gap_start, gap_end) = (pair[0].1, pair[1].0);
        if gap_end - gap_start <= max_gap {
            for s in &mut states[gap_start..gap_end] {
                *s = Some(true);
            }
        }
    }
}

/// Turns bouts shorter than `min_len` frames off.
pub fn drop_short(states: &mut [Option<bool>], min_len: usize) {
    for (start, end) in bouts(states) {
        if end - start < min_len {
            for s in &mut states[start..end] {
                *s = Some(false);
            }
        }
    }
}

/// The usual clean-up of a behavior state: merge short gaps, then drop short bouts.
pub fn clean(
    mut states: Vec<Option<bool>>,
    fps: f64,
    merge_gap_s: f64,
    min_bout_s: f64,
) -> Vec<Option<bool>> {
    merge_gaps(&mut states, (merge_gap_s * fps).round() as usize);
    drop_short(&mut states, frames(min_bout_s, fps));
    states
}

/// One behavior per frame: the first state in priority order that is on; `unknown` when none is
/// on but a higher-priority one is unknown; `other` otherwise.
pub fn resolve(states: &[(&str, &[Option<bool>])], frames: usize) -> Vec<String> {
    (0..frames)
        .map(|t| {
            let mut unknown = false;
            for (name, s) in states {
                match s.get(t).copied().flatten() {
                    Some(true) => return (*name).to_owned(),
                    Some(false) => {}
                    None => unknown = true,
                }
            }
            if unknown { "unknown" } else { "other" }.to_owned()
        })
        .collect()
}

/// Seconds, bouts and the latency (s) to the first bout of a state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Totals {
    pub seconds: f64,
    pub bouts: usize,
    pub latency: Option<f64>,
}

pub fn totals(states: &[Option<bool>], fps: f64) -> Totals {
    let found = bouts(states);
    Totals {
        seconds: states.iter().filter(|s| **s == Some(true)).count() as f64 / fps,
        bouts: found.len(),
        latency: found.first().map(|(b, _)| *b as f64 / fps),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(s: &str) -> Vec<Option<bool>> {
        s.chars()
            .map(|c| match c {
                '1' => Some(true),
                '0' => Some(false),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn hysteresis_enters_and_leaves_at_different_distances() {
        let d = [0.2, 0.05, 0.06, 0.07, 0.08, 0.04].map(Some);
        assert_eq!(hysteresis(&d, 0.05, 0.075), b("011101"));
        assert_eq!(hysteresis(&[None, Some(0.0)], 0.05, 0.075), b("?1"));
    }

    #[test]
    fn dwell_needs_the_new_state_to_hold() {
        assert_eq!(dwell(&b("0001000110000"), 2), b("0000000011000"));
        let labels: Vec<Option<String>> = ["a", "b", "a", "b", "b", "b"]
            .iter()
            .map(|s| Some((*s).to_owned()))
            .collect();
        let out: Vec<String> = dwell_label(&labels, 2).into_iter().flatten().collect();
        assert_eq!(out, ["a", "a", "a", "a", "b", "b"]);
    }

    #[test]
    fn gaps_merge_and_short_bouts_drop() {
        let mut s = b("1100110000011");
        merge_gaps(&mut s, 2);
        assert_eq!(s, b("1111110000011"));
        drop_short(&mut s, 3);
        assert_eq!(s, b("1111110000000"));
        let mut u = b("11??11");
        merge_gaps(&mut u, 2);
        assert_eq!(u, b("111111"));
    }

    #[test]
    fn priority_and_totals() {
        let contact = b("1000?");
        let groom = b("01100");
        let out = resolve(&[("mutual_contact", &contact), ("grooming", &groom)], 5);
        assert_eq!(
            out,
            ["mutual_contact", "grooming", "grooming", "other", "unknown"]
        );
        let t = totals(&b("0011011"), 10.0);
        assert_eq!(t.bouts, 2);
        assert!((t.seconds - 0.4).abs() < 1e-12);
        assert_eq!(t.latency, Some(0.2));
    }
}
