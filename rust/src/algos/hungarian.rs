//! Hungarian (Munkres) assignment, written by hand: the crate must not pull in
//! a linear-programming dependency.

use anyhow::{ensure, Result};

/// Minimum-cost one-to-one assignment of rows to columns.
///
/// Port of `scipy.optimize.linear_sum_assignment` for square and rectangular
/// cost matrices: at most `min(rows, columns)` pairs are returned, sorted by
/// row, and the total cost is minimal.
///
/// // differs from Python: `scipy` also accepts `inf` entries as "infeasible"
/// // pairs and skips them; here every entry must be finite. The tracker's
/// // "impossible pair" marker is the finite `_INF = 1e6` (`tracker::INF`), so
/// // the same behaviour is kept. On ties the two implementations may pick
/// // different (equally optimal) assignments.
pub fn linear_sum_assignment(cost: &[Vec<f64>]) -> Result<Vec<(usize, usize)>> {
    ensure!(!cost.is_empty(), "empty cost matrix");
    let n = cost.len();
    let m = cost[0].len();
    ensure!(m > 0, "empty cost matrix");
    for row in cost {
        ensure!(row.len() == m, "cost matrix rows must have equal length");
        for v in row {
            ensure!(v.is_finite(), "cost entries must be finite, got {v}");
        }
    }
    if n <= m {
        Ok(assign_min_row_first(cost, n, m))
    } else {
        // The potential method below assumes n <= m; transpose and undo.
        let t: Vec<Vec<f64>> = (0..m)
            .map(|j| (0..n).map(|i| cost[i][j]).collect())
            .collect();
        let mut out = assign_min_row_first(&t, m, n);
        for (a, b) in &mut out {
            std::mem::swap(a, b);
        }
        out.sort_unstable();
        Ok(out)
    }
}

/// Jonker-Volgenant style potentials for `n <= m`; returns `(row, col)` pairs
/// sorted by row with 0-based indices. Column `0` of the working arrays is the
/// virtual free column the alternating path starts from.
fn assign_min_row_first(cost: &[Vec<f64>], n: usize, m: usize) -> Vec<(usize, usize)> {
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut p = vec![0usize; m + 1]; // column -> row, 0 means "column is free"
    let mut way = vec![0usize; m + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            let cost_row = &cost[i0 - 1];
            for j in 1..=m {
                if used[j] {
                    continue;
                }
                let cur = cost_row[j - 1] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out = vec![(usize::MAX, usize::MAX); n];
    for j in 1..=m {
        if p[j] != 0 {
            out[p[j] - 1] = (p[j] - 1, j - 1);
        }
    }
    out.retain(|pair| pair.0 != usize::MAX);
    out.sort_unstable();
    out
}

/// Brute-force optimum used to cross-check [`linear_sum_assignment`] in tests.
#[cfg(test)]
fn brute_force(cost: &[Vec<f64>]) -> f64 {
    let mut best = f64::INFINITY;
    let mut used = vec![false; cost[0].len()];
    search(
        cost,
        0,
        0,
        cost.len().min(cost[0].len()),
        0.0,
        &mut used,
        &mut best,
    );
    best
}

#[cfg(test)]
fn search(
    cost: &[Vec<f64>],
    row: usize,
    matched: usize,
    target: usize,
    acc: f64,
    used: &mut [bool],
    best: &mut f64,
) {
    if matched == target {
        *best = (*best).min(acc);
        return;
    }
    if row == cost.len() {
        return;
    }
    for c in 0..cost[0].len() {
        if used[c] || acc + cost[row][c] >= *best {
            continue;
        }
        used[c] = true;
        search(
            cost,
            row + 1,
            matched + 1,
            target,
            acc + cost[row][c],
            used,
            best,
        );
        used[c] = false;
    }
    if cost.len() > cost[0].len() {
        search(cost, row + 1, matched, target, acc, used, best);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost_of(rows: &[&[f64]]) -> Vec<Vec<f64>> {
        rows.iter().map(|r| r.to_vec()).collect()
    }

    fn total(cost: &[Vec<f64>], pairs: &[(usize, usize)]) -> f64 {
        pairs.iter().map(|(r, c)| cost[*r][*c]).sum()
    }

    #[test]
    fn square_assignment_matches_the_classic_example() {
        // The textbook cost matrix; scipy picks (0,1) (1,2) (2,3) (3,0) = 8.
        let cost = cost_of(&[
            &[9.0, 2.0, 7.0, 8.0],
            &[6.4, 4.0, 2.0, 9.0],
            &[2.0, 9.0, 2.0, 3.0],
            &[1.0, 3.0, 8.0, 9.0],
        ]);
        let pairs = linear_sum_assignment(&cost).unwrap();
        assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 3), (3, 0)]);
        assert_eq!(total(&cost, &pairs), 8.0);
        assert_eq!(total(&cost, &pairs), brute_force(&cost));
    }

    #[test]
    fn more_columns_than_rows_matches_brute_force() {
        let cost = cost_of(&[&[4.0, 1.0, 8.0], &[7.0, 3.0, 2.0], &[9.0, 6.0, 5.0]]);
        let pairs = linear_sum_assignment(&cost).unwrap();
        assert_eq!(pairs.len(), 3); // the optimum is a tie here, only the cost is checked
        assert_eq!(total(&cost, &pairs), brute_force(&cost));
        assert_eq!(total(&cost, &pairs), 12.0);
        let wide = cost_of(&[&[5.0, 9.0, 2.0, 7.0], &[3.0, 1.0, 8.0, 4.0]]);
        let pairs = linear_sum_assignment(&wide).unwrap();
        assert_eq!(pairs, vec![(0, 2), (1, 1)]);
        assert_eq!(total(&wide, &pairs), brute_force(&wide));
    }

    #[test]
    fn more_rows_than_columns_is_handled_by_transposing() {
        let cost = cost_of(&[&[4.0, 7.0], &[1.0, 3.0], &[8.0, 2.0], &[9.0, 6.0]]);
        let pairs = linear_sum_assignment(&cost).unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(total(&cost, &pairs), brute_force(&cost));
        assert_eq!(
            pairs,
            vec![(1, 0), (2, 1)],
            "rows must come back in increasing order"
        );
    }

    #[test]
    fn a_1x1_and_a_single_row_are_no_ops() {
        assert_eq!(
            linear_sum_assignment(&cost_of(&[&[3.0]])).unwrap(),
            vec![(0, 0)]
        );
        let row = cost_of(&[&[3.0, 1.0, 2.0]]);
        assert_eq!(linear_sum_assignment(&row).unwrap(), vec![(0, 1)]);
    }

    #[test]
    fn the_inf_marker_is_a_finite_cost_and_can_stay_expensive() {
        // tracker::INF pairs stay assignable but expensive, exactly as in Python.
        let inf = 1e6;
        let cost = cost_of(&[&[inf, 0.1], &[0.2, inf]]);
        let pairs = linear_sum_assignment(&cost).unwrap();
        assert!(
            (total(&cost, &pairs) - 0.3).abs() < 1e-12,
            "{}",
            total(&cost, &pairs)
        );
        let cost = cost_of(&[&[inf, inf], &[0.4, 0.1]]);
        let pairs = linear_sum_assignment(&cost).unwrap();
        assert!(
            total(&cost, &pairs) > inf,
            "a forbidden pair is still used when forced"
        );
    }

    #[test]
    fn ragged_or_non_finite_input_is_rejected() {
        assert!(linear_sum_assignment(&[vec![1.0], vec![2.0, 3.0]]).is_err());
        assert!(linear_sum_assignment(&[vec![]]).is_err());
        let empty: &[Vec<f64>] = &[];
        assert!(linear_sum_assignment(empty).is_err());
        assert!(linear_sum_assignment(&cost_of(&[&[1.0, f64::NAN]])).is_err());
    }

    #[test]
    fn random_matrices_match_brute_force() {
        // Deterministic pseudo-random ladder (no rand crate dependency).
        let mut seed = 12345u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) % 1000) as f64 / 1000.0
        };
        for (n, m) in [(3, 3), (2, 4), (4, 2), (3, 5), (5, 3)] {
            let cost: Vec<Vec<f64>> = (0..n).map(|_| (0..m).map(|_| next()).collect()).collect();
            let pairs = linear_sum_assignment(&cost).unwrap();
            assert_eq!(pairs.len(), n.min(m));
            let got = total(&cost, &pairs);
            let want = brute_force(&cost);
            assert!((got - want).abs() < 1e-9, "{n}x{m}: {got} vs {want}");
        }
    }
}
