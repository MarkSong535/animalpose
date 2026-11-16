//! Tests.
//!
//! Two independent kinds of check:
//!
//! * **torch fixtures** — `animalpose fixtures tensor` writes one JSON file per operation
//!   (`tests/fixtures/tensor/<name>.json`) holding the inputs, the forward output and
//!   `d(sum(output))/d(input)` for every input. Each test compares forward values and gradients with a
//!   `2e-4` absolute tolerance. When the files are absent (a fresh clone, or a CI job without torch) the
//!   test prints a line and passes.
//! * **finite differences** — a scalar loss is perturbed element by element and the tape's gradient
//!   compared with the central difference. These need nothing but this crate and are the real
//!   correctness gate: they run on every operation whose gradient is hand-derived.

use super::*;
use crate::tensor::optim::decays;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;

/// Absolute tolerance against the torch fixtures.
const TOL: f32 = 2e-4;

/// One tensor of a fixture.
#[derive(Debug, Deserialize)]
struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// One dumped torch reference: inputs, the operation's scalars, the output and its input gradients.
#[derive(Debug, Deserialize)]
struct Fixture {
    #[serde(default)]
    shape: Vec<usize>,
    inputs: BTreeMap<String, Tensor>,
    #[serde(default)]
    params: BTreeMap<String, serde_json::Value>,
    output: Tensor,
    grad_input: Tensor,
    #[serde(default)]
    grad_params: BTreeMap<String, Tensor>,
}

impl Fixture {
    /// One named input.
    fn input(&self, name: &str) -> &Tensor {
        self.inputs
            .get(name)
            .unwrap_or_else(|| panic!("fixture has no input {name}, has {:?}", self.inputs.keys()))
    }

    /// An integer scalar of the operation.
    fn usize(&self, key: &str, default: usize) -> usize {
        match self.params.get(key) {
            Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| v as usize).unwrap_or(default),
            _ => default,
        }
    }

    /// A boolean scalar of the operation.
    fn flag(&self, key: &str, default: bool) -> bool {
        match self.params.get(key) {
            Some(serde_json::Value::Bool(b)) => *b,
            _ => default,
        }
    }

    /// A float scalar of the operation.
    fn float(&self, key: &str, default: f32) -> f32 {
        match self.params.get(key) {
            Some(serde_json::Value::Number(n)) => n.as_f64().map(|v| v as f32).unwrap_or(default),
            _ => default,
        }
    }

    /// An integer scalar that may be negative (`ignore_index`).
    fn signed(&self, key: &str, default: i32) -> i32 {
        match self.params.get(key) {
            Some(serde_json::Value::Number(n)) => n.as_i64().map(|v| v as i32).unwrap_or(default),
            _ => default,
        }
    }

    /// A list of integers (axes, a permutation).
    fn list(&self, key: &str) -> Vec<usize> {
        self.params
            .get(key)
            .and_then(serde_json::Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as usize)
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// The directory the torch fixtures live in: `$ANIMALPOSE_FIXTURES` when set, otherwise the repository's
/// `tests/fixtures/tensor`, searched by walking up from the crate directory `cargo test` runs in.
fn fixtures_dir() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("ANIMALPOSE_FIXTURES") {
        let dir = PathBuf::from(dir);
        return dir.is_dir().then_some(dir);
    }
    let mut at = env::current_dir().ok()?;
    for _ in 0..6 {
        let candidate = at.join("tests").join("fixtures").join("tensor");
        if candidate.is_dir() {
            return Some(candidate);
        }
        if !at.pop() {
            break;
        }
    }
    None
}

/// One fixture, or `None` after printing when the fixtures were never dumped.
fn fixture(name: &str) -> Option<Fixture> {
    let dir = fixtures_dir()?;
    let path = dir.join(format!("{name}.json"));
    if !path.is_file() {
        println!("{name}: no fixture at {path:?} (run animalpose fixtures tensor); skipping");
        return None;
    }
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
    Some(serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {path:?}: {e}")))
}

/// Largest absolute difference between two slices of the same length.
fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(
        got.len(),
        want.len(),
        "length mismatch: {} vs {}",
        got.len(),
        want.len()
    );
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Prints the error of one comparison and asserts it is inside [`TOL`].
fn check(name: &str, what: &str, got: &[f32], want: &[f32]) {
    let err = max_abs_diff(got, want);
    println!("{name}: {what} max abs error {err:.3e}");
    assert!(
        err <= TOL,
        "{name}: {what} differs from torch by {err:.3e} > {TOL}"
    );
}

/// Runs one fixture's inputs through `build`, then compares the output, the gradient of `x` and the
/// gradients of every other input that torch recorded. Skips when the fixture file is missing.
fn compare(name: &str, build: impl Fn(&mut Graph, &Fixture, &BTreeMap<String, NodeId>) -> NodeId) {
    let Some(fx) = fixture(name) else { return };
    let mut g = Graph::new();
    let mut ids: BTreeMap<String, NodeId> = BTreeMap::new();
    for (key, tensor) in &fx.inputs {
        let id = g.from_data(tensor.data.clone(), &tensor.shape);
        ids.insert(key.clone(), id);
    }
    assert_eq!(
        fx.input("x").shape,
        fx.shape,
        "{name}: the primary input shape is `shape`"
    );
    let out = build(&mut g, &fx, &ids);
    let got = g.shape(out);
    let want = &fx.output.shape;
    let scalar = got.is_empty() && want.as_slice() == [1];
    assert!(
        got == want || scalar,
        "{name}: output shape {got:?} != {want:?}"
    );
    check(name, "output", g.value(out), &fx.output.data);
    let loss = g.sum_all(out);
    g.backward(loss).expect("a scalar loss");
    check(
        name,
        "grad_input",
        g.gradient(ids["x"]),
        &fx.grad_input.data,
    );
    for (key, want) in &fx.grad_params {
        if let Some(id) = ids.get(key) {
            check(name, &format!("grad[{key}]"), g.gradient(*id), &want.data);
        }
    }
}

/// The weight node ids of an `nn.LSTM` fixture, in the order [`Graph::lstm`] wants them.
fn lstm_ids(fx: &Fixture, ids: &BTreeMap<String, NodeId>) -> Vec<NodeId> {
    let layers = fx.usize("layers", 1);
    let bidirectional = fx.flag("bidirectional", false);
    let mut out = Vec::new();
    for l in 0..layers {
        for rev in 0..(usize::from(bidirectional) + 1) {
            let suffix = if rev == 1 { "_reverse" } else { "" };
            for which in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"] {
                out.push(ids[&format!("{which}_l{l}{suffix}")]);
            }
        }
    }
    out
}

// --------------------------------------------------------------------------- torch fixtures

#[test]
fn conv2d_matches_torch() {
    let plain = |g: &mut Graph, fx: &Fixture, ids: &BTreeMap<String, NodeId>| {
        g.conv2d(
            ids["x"],
            ids["weight"],
            Some(ids["bias"]),
            fx.usize("stride", 1),
            fx.usize("pad", 0),
            fx.usize("dilation", 1),
        )
    };
    compare("conv2d_k3_s1_p1", plain);
    compare("conv2d_dilated", plain);
}

#[test]
fn batch_norm_matches_torch() {
    compare("batch_norm_train", |g, fx, ids| {
        g.batch_norm(
            ids["x"],
            ids["gamma"],
            ids["beta"],
            fx.float("eps", 1e-5),
            BnMode::Train,
        )
    });
    compare("batch_norm_eval", |g, fx, ids| {
        let mode = BnMode::Eval {
            mean: fx.input("running_mean").data.clone(),
            var: fx.input("running_var").data.clone(),
        };
        g.batch_norm(
            ids["x"],
            ids["gamma"],
            ids["beta"],
            fx.float("eps", 1e-5),
            mode,
        )
    });
}

#[test]
fn pooling_and_upsampling_match_torch() {
    let pool = |g: &mut Graph, fx: &Fixture, ids: &BTreeMap<String, NodeId>, max: bool| {
        let (k, stride, pad) = (fx.usize("k", 2), fx.usize("stride", 2), fx.usize("pad", 0));
        if max {
            g.max_pool2d(ids["x"], k, stride, pad)
        } else {
            g.avg_pool2d(ids["x"], k, stride, pad)
        }
    };
    compare("max_pool2d", |g, fx, ids| pool(g, fx, ids, true));
    compare("avg_pool2d", |g, fx, ids| pool(g, fx, ids, false));
    compare("upsample_nearest", |g, fx, ids| {
        g.upsample_nearest(ids["x"], fx.usize("h", 1), fx.usize("w", 1))
    });
}

#[test]
fn matmul_matches_torch() {
    compare("matmul_batched", |g, _, ids| {
        g.matmul(ids["x"], ids["other"])
    });
    compare("matmul_broadcast", |g, _, ids| {
        g.matmul(ids["x"], ids["other"])
    });
    compare("matmul_2d", |g, _, ids| g.matmul(ids["x"], ids["other"]));
}

#[test]
fn shape_ops_match_torch() {
    compare("permute", |g, fx, ids| {
        g.permute(ids["x"], &fx.list("perm"))
    });
    compare("concat", |g, fx, ids| {
        g.concat(&[ids["x"], ids["other"]], fx.usize("axis", 0))
    });
    compare("slice", |g, fx, ids| {
        g.slice(
            ids["x"],
            fx.usize("axis", 0),
            fx.usize("start", 0),
            fx.usize("end", 1),
        )
    });
}

#[test]
fn reductions_match_torch() {
    for tag in ["axes1", "axes02", "all"] {
        let axes = |g: &mut Graph, fx: &Fixture, ids: &BTreeMap<String, NodeId>| {
            g.reduce_sum(ids["x"], &fx.list("axes"))
        };
        let mean = |g: &mut Graph, fx: &Fixture, ids: &BTreeMap<String, NodeId>| {
            g.reduce_mean(ids["x"], &fx.list("axes"))
        };
        compare(&format!("reduce_sum_{tag}"), axes);
        compare(&format!("reduce_mean_{tag}"), mean);
    }
}

#[test]
fn normalisations_match_torch() {
    compare("softmax", |g, fx, ids| {
        g.softmax(ids["x"], fx.usize("axis", 1))
    });
    compare("log_softmax", |g, fx, ids| {
        g.log_softmax(ids["x"], fx.usize("axis", 1))
    });
    compare("l2_normalize", |g, fx, ids| {
        g.l2_normalize(ids["x"], fx.usize("axis", 1))
    });
}

#[test]
fn losses_match_torch() {
    compare("bce_with_logits", |g, _, ids| {
        g.bce_with_logits(ids["x"], ids["target"])
    });
    compare("smooth_l1", |g, fx, ids| {
        g.weighted_loss(ids["x"], LossKind::SmoothL1, fx.float("weight", 1.0))
    });
    compare("mse", |g, fx, ids| {
        g.weighted_loss(ids["x"], LossKind::Mse, fx.float("weight", 1.0))
    });
    let ce = |g: &mut Graph, fx: &Fixture, ids: &BTreeMap<String, NodeId>| {
        let weight = ids.get("weight").copied();
        g.cross_entropy(
            ids["x"],
            ids["target"],
            weight,
            fx.float("smoothing", 0.0),
            fx.signed("ignore", -100),
        )
    };
    compare("cross_entropy", ce);
    compare("cross_entropy_weighted", ce);
}

#[test]
fn lstm_matches_torch() {
    let build = |g: &mut Graph, fx: &Fixture, ids: &BTreeMap<String, NodeId>| {
        let w = lstm_ids(fx, ids);
        g.lstm(
            ids["x"],
            &w,
            fx.usize("hidden", 4),
            fx.usize("layers", 1),
            fx.flag("bidirectional", false),
        )
    };
    compare("lstm_2layer_bidir", build);
    compare("lstm_1layer", build);
}

// ----------------------------------------------------------------------- finite differences

/// Deterministic values in `[-0.9, 0.9]`, so a gradient check needs no fixture.
fn ramp(n: usize, offset: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + offset) * 37) % 101) as f32 / 55.0 - 0.9)
        .collect()
}

/// Runs `build` on a fresh graph (optionally with input `at` replaced), returning the scalar loss and
/// the gradient of every input.
fn run(
    build: &dyn Fn(&mut Graph, &[NodeId]) -> NodeId,
    inputs: &[(&[f32], &[usize])],
    patch: Option<(usize, Vec<f32>)>,
) -> (f32, Vec<Vec<f32>>) {
    let mut g = Graph::new();
    let mut ids = Vec::with_capacity(inputs.len());
    for (i, (data, shape)) in inputs.iter().enumerate() {
        let values = match &patch {
            Some((at, values)) if *at == i => values.clone(),
            _ => data.to_vec(),
        };
        ids.push(g.from_data(values, shape));
    }
    let loss = build(&mut g, &ids);
    let value = g.value(loss)[0];
    g.backward(loss).expect("a scalar loss");
    let grads: Vec<Vec<f32>> = ids.iter().map(|id| g.gradient(*id).to_vec()).collect();
    (value, grads)
}

/// Largest difference between [`Graph::backward`]'s gradient and a central finite difference of the
/// same scalar output, over every element of every input.
fn fd_error(
    build: &dyn Fn(&mut Graph, &[NodeId]) -> NodeId,
    inputs: &[(&[f32], &[usize])],
    eps: f32,
) -> f32 {
    let (_, grads) = run(build, inputs, None);
    let mut worst = 0.0f32;
    for (i, (data, _)) in inputs.iter().enumerate() {
        for j in 0..data.len() {
            let mut up = data.to_vec();
            up[j] += eps;
            let mut down = data.to_vec();
            down[j] -= eps;
            let (up_value, _) = run(build, inputs, Some((i, up.clone())));
            let (down_value, _) = run(build, inputs, Some((i, down)));
            let numeric = (up_value - down_value) / (2.0 * eps);
            let err = (numeric - grads[i][j]).abs();
            if err > worst {
                worst = err;
                println!(
                    "fd: input {i} element {j}: tape {:.6e}, numeric {:.6e}",
                    grads[i][j], numeric
                );
            }
        }
    }
    worst
}

/// The step size finite differences use: small enough to see curvature, large enough that `f32`
/// cancellation does not dominate.
const FD_EPS: f32 = 1e-2;

/// Tolerance for a finite-difference comparison in `f32`.
const FD_TOL: f32 = 2e-3;

#[test]
fn conv2d_gradient_matches_finite_differences() {
    let x = ramp(2 * 3 * 5 * 5, 0);
    let w = ramp(2 * 3 * 3 * 3, 7);
    let b = ramp(2, 3);
    let build = |g: &mut Graph, n: &[NodeId]| {
        let out = g.conv2d(n[0], n[1], Some(n[2]), 1, 1, 1);
        g.reduce_mean(out, &[0, 1, 2, 3])
    };
    let inputs: &[(&[f32], &[usize])] = &[(&x, &[2, 3, 5, 5]), (&w, &[2, 3, 3, 3]), (&b, &[2])];
    let err = fd_error(&build, inputs, FD_EPS);
    assert!(err < FD_TOL, "conv2d gradient off by {err}");
}

#[test]
fn conv1d_pads_only_the_time_axis() {
    // `nn.Conv1d(c, c, 3, padding = dilation, dilation)` keeps the length. The expectations are
    // `F.conv1d(x, w, padding = d * (k - 1) / 2, dilation = d)` over `x = arange(L)`, printed from torch:
    // the runs outside the sequence contribute zero, and a kernel-axis padding would lengthen `y`.
    // (weight taps, sequence, dilation, expected output)
    type Case = (&'static [f32], &'static [f32], usize, &'static [f32]);
    let cases: &[Case] = &[
        (
            &[1.0, 0.0, 1.0],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            2,
            &[2.0, 3.0, 4.0, 6.0, 2.0, 3.0],
        ),
        (
            &[1.0, -1.0, 0.5],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            2,
            &[1.0, 0.5, 0.0, 0.5, -2.0, -2.0],
        ),
        (
            &[1.0, -1.0, 0.5],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            1,
            &[0.5, 0.0, 0.5, 1.0, -1.0],
        ),
        (
            &[1.0, -1.0, 0.5],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            4,
            &[2.0, -1.0, -2.0, -3.0, -4.0],
        ),
    ];
    for (w, x, dilation, want) in cases {
        let mut g = Graph::new();
        let x = g.from_data(x.to_vec(), &[1, 1, x.len()]);
        let w = g.from_data(w.to_vec(), &[1, 1, 3]);
        let y = g.conv1d(x, w, None, 3, *dilation);
        assert_eq!(g.shape(y).as_slice(), [1, 1, want.len()]);
        assert_eq!(g.value(y), *want, "dilation {dilation}");
    }
}

#[test]
fn conv1d_gradient_matches_finite_differences() {
    let x = ramp(2 * 3 * 7, 4);
    let w = ramp(2 * 3 * 3, 9);
    let build = |g: &mut Graph, n: &[NodeId]| {
        let out = g.conv1d(n[0], n[1], None, 3, 2);
        g.reduce_mean(out, &[0, 1, 2])
    };
    let inputs: &[(&[f32], &[usize])] = &[(&x, &[2, 3, 7]), (&w, &[2, 3, 3])];
    let err = fd_error(&build, inputs, FD_EPS);
    assert!(err < FD_TOL, "conv1d gradient off by {err}");
}

#[test]
fn dilated_conv2d_gradient_matches_finite_differences() {
    let x = ramp(2 * 6 * 6, 1);
    let w = ramp(2 * 2 * 3 * 3, 5);
    let build = |g: &mut Graph, n: &[NodeId]| {
        let out = g.conv2d(n[0], n[1], None, 1, 2, 2);
        g.reduce_mean(out, &[0, 1, 2, 3])
    };
    let err = fd_error(&build, &[(&x, &[1, 2, 6, 6]), (&w, &[2, 2, 3, 3])], FD_EPS);
    assert!(err < FD_TOL, "dilated conv2d gradient off by {err}");
}

#[test]
fn batch_norm_gradient_matches_finite_differences() {
    let x = ramp(2 * 3 * 4 * 4, 0);
    let gamma = ramp(3, 11);
    let beta = ramp(3, 5);
    let projection = ramp(2 * 3 * 4 * 4, 13);
    let build = |g: &mut Graph, n: &[NodeId]| {
        let normed = g.batch_norm(n[0], n[1], n[2], 1e-5, BnMode::Train);
        let weighted = g.mul(normed, n[3]);
        g.reduce_mean(weighted, &[0, 1, 2, 3])
    };
    let inputs: &[(&[f32], &[usize])] = &[
        (&x, &[2, 3, 4, 4]),
        (&gamma, &[3]),
        (&beta, &[3]),
        (&projection, &[2, 3, 4, 4]),
    ];
    let err = fd_error(&build, inputs, FD_EPS);
    assert!(err < FD_TOL, "batch_norm (train) gradient off by {err}");
}

#[test]
fn max_pool2d_gradient_matches_finite_differences() {
    let x = ramp(2 * 3 * 6 * 6, 0);
    let build = |g: &mut Graph, n: &[NodeId]| {
        let pooled = g.max_pool2d(n[0], 2, 2, 0);
        g.reduce_mean(pooled, &[0, 1, 2, 3])
    };
    let err = fd_error(&build, &[(&x, &[2, 3, 6, 6])], FD_EPS);
    assert!(err < FD_TOL, "max_pool2d gradient off by {err}");
}

#[test]
fn avg_pool2d_and_upsample_gradients_match_finite_differences() {
    let x = ramp(2 * 3 * 6 * 6, 4);
    let build = |g: &mut Graph, n: &[NodeId]| {
        let pooled = g.avg_pool2d(n[0], 2, 2, 1);
        g.reduce_mean(pooled, &[0, 1, 2, 3])
    };
    let err = fd_error(&build, &[(&x, &[2, 3, 6, 6])], FD_EPS);
    assert!(err < FD_TOL, "avg_pool2d gradient off by {err}");
    let y = ramp(2 * 3 * 4, 9);
    let up = |g: &mut Graph, n: &[NodeId]| {
        let scaled = g.upsample_nearest(n[0], 6, 8);
        g.reduce_mean(scaled, &[0, 1, 2, 3])
    };
    let err = fd_error(&up, &[(&y, &[1, 2, 3, 4])], FD_EPS);
    assert!(err < FD_TOL, "upsample_nearest gradient off by {err}");
}

#[test]
fn softmax_gradient_matches_finite_differences() {
    let x = ramp(2 * 3 * 4, 0);
    let build = |g: &mut Graph, n: &[NodeId]| {
        let p = g.softmax(n[0], 1);
        g.reduce_mean(p, &[0, 1, 2])
    };
    let err = fd_error(&build, &[(&x, &[2, 3, 4])], FD_EPS);
    assert!(err < FD_TOL, "softmax gradient off by {err}");
    let y = ramp(2 * 3 * 4, 3);
    let log = |g: &mut Graph, n: &[NodeId]| {
        let p = g.log_softmax(n[0], 1);
        g.reduce_mean(p, &[0, 1, 2])
    };
    let err = fd_error(&log, &[(&y, &[2, 3, 4])], FD_EPS);
    assert!(err < FD_TOL, "log_softmax gradient off by {err}");
}

#[test]
fn cross_entropy_gradient_matches_finite_differences() {
    let logits = ramp(4 * 5, 0);
    let target = vec![0.0f32, 3.0, 1.0, -100.0];
    let weight = vec![0.4f32, 1.0, 2.0, 0.7, 1.3];
    let plain = |g: &mut Graph, n: &[NodeId]| {
        let t = g.from_data(vec![2.0f32, 0.0, 4.0, 1.0], &[4]);
        g.cross_entropy(n[0], t, None, 0.0, -100)
    };
    let err = fd_error(&plain, &[(&logits, &[4, 5])], FD_EPS);
    assert!(err < FD_TOL, "cross_entropy gradient off by {err}");
    let smoothed = |g: &mut Graph, n: &[NodeId]| {
        let t = g.from_data(target.clone(), &[4]);
        let w = g.from_data(weight.clone(), &[5]);
        g.cross_entropy(n[0], t, Some(w), 0.2, -100)
    };
    let err = fd_error(&smoothed, &[(&logits, &[4, 5])], FD_EPS);
    assert!(
        err < FD_TOL,
        "cross_entropy (weighted, smoothed, ignored) gradient off by {err}"
    );
}

#[test]
fn matmul_gradient_matches_finite_differences() {
    let a = ramp(2 * 4 * 3, 0);
    let b = ramp(2 * 3 * 5, 6);
    let build = |g: &mut Graph, n: &[NodeId]| {
        let prod = g.matmul(n[0], n[1]);
        g.reduce_mean(prod, &[0, 1, 2])
    };
    let inputs: &[(&[f32], &[usize])] = &[(&a, &[2, 4, 3]), (&b, &[2, 3, 5])];
    let err = fd_error(&build, inputs, FD_EPS);
    assert!(err < FD_TOL, "batched matmul gradient off by {err}");
    let c = ramp(3 * 4, 2);
    let d = ramp(4 * 2, 5);
    let flat = |g: &mut Graph, n: &[NodeId]| {
        let prod = g.matmul(n[0], n[1]);
        g.reduce_mean(prod, &[0, 1])
    };
    let err = fd_error(&flat, &[(&c, &[3, 4]), (&d, &[4, 2])], FD_EPS);
    assert!(err < FD_TOL, "unbatched matmul gradient off by {err}");
    // One of the two operands broadcast over the batch dimension.
    let e = ramp(4 * 3, 1);
    let f = ramp(2 * 3 * 5, 7);
    let bc = |g: &mut Graph, n: &[NodeId]| {
        let prod = g.matmul(n[0], n[1]);
        g.reduce_mean(prod, &[0, 1, 2])
    };
    let inputs: &[(&[f32], &[usize])] = &[(&e, &[1, 4, 3]), (&f, &[2, 3, 5])];
    let err = fd_error(&bc, inputs, FD_EPS);
    assert!(err < FD_TOL, "broadcast matmul gradient off by {err}");
}

#[test]
fn layout_op_gradients_match_finite_differences() {
    let x = ramp(2 * 3 * 4, 0);
    let permute = |g: &mut Graph, n: &[NodeId]| {
        let moved = g.permute(n[0], &[2, 0, 1]);
        g.reduce_sum(moved, &[0, 1, 2])
    };
    let err = fd_error(&permute, &[(&x, &[2, 3, 4])], FD_EPS);
    assert!(err < FD_TOL, "permute gradient off by {err}");

    let a = ramp(2 * 3 * 4, 1);
    let b = ramp(2 * 3 * 2, 5);
    let concat = |g: &mut Graph, n: &[NodeId]| {
        let joined = g.concat(&[n[0], n[1]], 2);
        g.reduce_sum(joined, &[0, 1, 2])
    };
    let inputs: &[(&[f32], &[usize])] = &[(&a, &[2, 3, 4]), (&b, &[2, 3, 2])];
    let err = fd_error(&concat, inputs, FD_EPS);
    assert!(err < FD_TOL, "concat gradient off by {err}");

    let y = ramp(2 * 3 * 4, 3);
    let slice = |g: &mut Graph, n: &[NodeId]| {
        let part = g.slice(n[0], 1, 1, 3);
        g.reduce_sum(part, &[0, 1, 2])
    };
    let err = fd_error(&slice, &[(&y, &[2, 3, 4])], FD_EPS);
    assert!(err < FD_TOL, "slice gradient off by {err}");

    let z = ramp(2 * 3 * 4, 8);
    let mean = |g: &mut Graph, n: &[NodeId]| g.reduce_mean(n[0], &[0, 1, 2]);
    let err = fd_error(&mean, &[(&z, &[2, 3, 4])], FD_EPS);
    assert!(err < FD_TOL, "reduce_mean gradient off by {err}");
    let partial = |g: &mut Graph, n: &[NodeId]| {
        let half = g.reduce_mean(n[0], &[1]);
        g.reduce_sum(half, &[0, 1])
    };
    let err = fd_error(&partial, &[(&z, &[2, 3, 4])], FD_EPS);
    assert!(err < FD_TOL, "partial reduce_mean gradient off by {err}");
}

#[test]
fn l2_normalize_gradient_matches_finite_differences() {
    let x = ramp(2 * 3 * 4, 4);
    let projection = ramp(2 * 3 * 4, 17);
    let build = |g: &mut Graph, n: &[NodeId]| {
        let normed = g.l2_normalize(n[0], 1);
        let projected = g.mul(normed, n[1]);
        g.reduce_mean(projected, &[0, 1, 2])
    };
    let inputs: &[(&[f32], &[usize])] = &[(&x, &[2, 3, 4]), (&projection, &[2, 3, 4])];
    let err = fd_error(&build, inputs, FD_EPS);
    assert!(err < FD_TOL, "l2_normalize gradient off by {err}");
}

#[test]
fn elementwise_and_loss_gradients_match_finite_differences() {
    let x = ramp(6, 0);
    let y = ramp(6, 25);
    let add = |g: &mut Graph, n: &[NodeId]| {
        let squared = g.mul(n[1], n[1]);
        let sum = g.add(n[0], squared);
        g.reduce_mean(sum, &[0])
    };
    let inputs: &[(&[f32], &[usize])] = &[(&x, &[6]), (&y, &[6])];
    let err = fd_error(&add, inputs, FD_EPS);
    assert!(err < FD_TOL, "add/mul gradient off by {err}");

    let target = ramp(2 * 3, 4);
    let bce = |g: &mut Graph, n: &[NodeId]| {
        let t = g.from_data(target.clone(), &[2, 3]);
        g.bce_with_logits(n[0], t)
    };
    let err = fd_error(&bce, &[(&x, &[6])], 1e-2);
    assert!(err < FD_TOL, "bce_with_logits gradient off by {err}");
    let smooth = |g: &mut Graph, n: &[NodeId]| g.weighted_loss(n[0], LossKind::SmoothL1, 0.5);
    let err = fd_error(&smooth, &[(&x, &[6])], 1e-2);
    assert!(err < FD_TOL, "smooth_l1 gradient off by {err}");
    let l1 = |g: &mut Graph, n: &[NodeId]| g.weighted_loss(n[0], LossKind::L1, 2.0);
    let err = fd_error(&l1, &[(&x, &[6])], 1e-2);
    assert!(err < FD_TOL, "l1 gradient off by {err}");
}

#[test]
fn lstm_gradient_matches_finite_differences() {
    let x = ramp(2 * 3 * 2, 0);
    // Two bidirectional layers, hidden 2: `4 * hidden` gate rows per block.
    let mut blocks: Vec<(Vec<f32>, Vec<usize>)> = Vec::new();
    for l in 0..2 {
        let in_w = if l == 0 { 2 } else { 4 };
        for _ in 0..2 {
            let wih = ramp_scaled(8 * in_w, 30 + l * 40);
            let whh = ramp_scaled(8 * 2, 31 + l * 40);
            let bih = ramp_scaled(8, 32 + l * 40);
            let bhh = ramp_scaled(8, 33 + l * 40);
            blocks.push((wih, vec![8, in_w]));
            blocks.push((whh, vec![8, 2]));
            blocks.push((bih, vec![8]));
            blocks.push((bhh, vec![8]));
        }
    }
    let build = |g: &mut Graph, n: &[NodeId]| {
        let out = g.lstm(n[0], &n[1..], 2, 2, true);
        g.reduce_mean(out, &[0, 1, 2])
    };
    let mut inputs: Vec<(&[f32], &[usize])> = vec![(&x, &[2, 3, 2])];
    for (data, shape) in &blocks {
        inputs.push((data.as_slice(), shape.as_slice()));
    }
    let err = fd_error(&build, &inputs, FD_EPS);
    assert!(err < FD_TOL, "lstm gradient off by {err}");
}

#[test]
fn unidirectional_lstm_gradient_matches_finite_differences() {
    let x = ramp(2 * 3 * 3, 6);
    let wih = ramp_scaled(8 * 3, 71);
    let whh = ramp_scaled(8 * 2, 72);
    let bih = ramp_scaled(8, 73);
    let bhh = ramp_scaled(8, 74);
    let build = |g: &mut Graph, n: &[NodeId]| {
        let out = g.lstm(n[0], &n[1..], 2, 1, false);
        g.reduce_mean(out, &[0, 1, 2])
    };
    let inputs: &[(&[f32], &[usize])] = &[
        (&x, &[2, 3, 3]),
        (&wih, &[8, 3]),
        (&whh, &[8, 2]),
        (&bih, &[8]),
        (&bhh, &[8]),
    ];
    let err = fd_error(&build, inputs, FD_EPS);
    assert!(err < FD_TOL, "single-layer lstm gradient off by {err}");
}

/// Like [`ramp`], but small enough that the LSTM's gates stay away from saturation.
fn ramp_scaled(n: usize, offset: usize) -> Vec<f32> {
    ramp(n, offset).iter().map(|v| v * 0.35).collect()
}

// ------------------------------------------------------------------------------- the layers

#[test]
fn conv_bn_act_names_parameters_like_the_checkpoint_and_updates_running_stats() {
    let layer = conv_bn_act("stages.0.0", 3, 4, 3, 1, 1, ConvBnActOpts::default());
    assert_eq!(
        layer.params().join(","),
        "stages.0.0.0.weight,stages.0.0.1.weight,stages.0.0.1.bias"
    );
    assert_eq!(
        layer.buffers().join(","),
        "stages.0.0.1.running_mean,stages.0.0.1.running_var"
    );

    let mut g = Graph::new();
    let x = g.from_data(ramp(2 * 3 * 8 * 8, 0), &[2, 3, 8, 8]);
    let out = layer.forward(&mut g, x, &mut Ctx::new(true, 0));
    assert_eq!(g.shape(out).as_slice(), [2, 4, 8, 8]);
    // A training forward pass refreshes the running buffers; the values stay finite either way.
    let mean = layer
        .values("stages.0.0.1.running_mean")
        .expect("running mean");
    assert!(mean.iter().all(|v| v.is_finite()));
    assert_ne!(
        mean,
        vec![0.0f32; 4],
        "a train forward pass must update running_mean"
    );

    let loss = g.reduce_mean(out, &[0, 1, 2, 3]);
    g.backward(loss).unwrap();
    let grads = g.param_grads();
    let slots: Vec<usize> = grads.iter().map(|(slot, _)| *slot).collect();
    assert_eq!(slots, vec![0, 1, 2], "slots must follow params() order");
    assert!(grads
        .iter()
        .all(|(_, grad)| grad.iter().all(|v| v.is_finite())));

    // Eval mode uses the stored statistics instead of the batch's.
    let mut eval = Graph::new();
    let x = eval.from_data(ramp(2 * 3 * 8 * 8, 0), &[2, 3, 8, 8]);
    let after = layer.forward(&mut eval, x, &mut Ctx::eval());
    let before = layer
        .values("stages.0.0.1.running_mean")
        .expect("running mean");
    assert_eq!(before, mean, "eval mode must not touch the running buffers");
    assert!(eval.value(after).iter().all(|v| v.is_finite()));
}

#[test]
fn layers_load_a_checkpoint_and_run() {
    let mut model = Sequential::of(vec![
        Box::new(conv_bn_act(
            "stem.0",
            3,
            4,
            3,
            1,
            1,
            ConvBnActOpts::default(),
        )),
        Box::new(Flatten::new(1)),
        Box::new(Dense::new("head", 100, 2, true)),
    ]);
    assert_eq!(model.params().len(), 5);
    assert_eq!(model.params()[0], "stem.0.0.weight");
    assert_eq!(model.params()[4], "head.bias");

    let weight = vec![0.25f32; 4 * 3 * 3 * 3];
    assert!(model.load("stem.0.0.weight", &weight));
    assert_eq!(
        model.values("stem.0.0.weight").expect("conv weight"),
        weight
    );
    assert!(
        !model.load("stem.0.0.weight", &[1.0f32; 7]),
        "a wrong size must not load"
    );
    assert!(!model.load("nope.weight", &[1.0f32; 7]));

    let mut g = Graph::new();
    let x = g.from_data(ramp(3 * 5 * 5, 0), &[1, 3, 5, 5]);
    let out = model.forward(&mut g, x, &mut Ctx::new(true, 3));
    assert_eq!(g.shape(out).as_slice(), &[1, 2]);
    let loss = g.reduce_mean(out, &[0, 1]);
    g.backward(loss).unwrap();
    let grads = g.param_grads();
    assert_eq!(grads.len(), 5, "every parameter must get a gradient");
    assert!(grads.iter().all(|(_, grad)| grad.iter().any(|v| *v != 0.0)));
}

#[test]
fn residual_and_parallel_layers_combine_branches() {
    let block = Residual::identity(Box::new(conv_bn_act(
        "stages.1.0",
        4,
        4,
        3,
        1,
        1,
        ConvBnActOpts::default(),
    )));
    let skip = Conv2dLayer::pointwise("stages.1.0.skip", 4, 4, false);
    let with_skip = Residual::new(
        Box::new(conv_bn_act(
            "stages.2.0",
            4,
            4,
            3,
            1,
            1,
            ConvBnActOpts::default(),
        )),
        Box::new(skip),
    );
    assert_eq!(block.params().len(), 3);
    assert_eq!(with_skip.params().last().unwrap(), "stages.1.0.skip.weight");

    let mut model = Sequential::of(vec![
        Box::new(block),
        Box::new(Parallel::add(vec![
            Box::new(Conv2dLayer::pointwise("lat.0", 4, 4, true)),
            Box::new(Conv2dLayer::pointwise("lat.1", 4, 4, true)),
        ])),
        Box::new(with_skip),
    ]);
    model.assign_slots(0);
    let mut g = Graph::new();
    let x = g.from_data(ramp(4 * 5 * 5, 0), &[1, 4, 5, 5]);
    let out = model.forward(&mut g, x, &mut Ctx::new(true, 1));
    assert_eq!(g.shape(out).as_slice(), &[1, 4, 5, 5]);
    let loss = g.reduce_mean(out, &[0, 1, 2, 3]);
    g.backward(loss).unwrap();
    assert_eq!(g.param_grads().len(), model.params().len());
}

#[test]
fn lstm_layer_keeps_the_checkpoint_names() {
    let layer = Lstm::new("lstm", 5, 4, 2, true, 0.1);
    let expected = [
        "lstm.weight_ih_l0",
        "lstm.weight_hh_l0",
        "lstm.bias_ih_l0",
        "lstm.bias_hh_l0",
        "lstm.weight_ih_l0_reverse",
        "lstm.weight_hh_l0_reverse",
        "lstm.bias_ih_l0_reverse",
        "lstm.bias_hh_l0_reverse",
        "lstm.weight_ih_l1",
        "lstm.weight_hh_l1",
        "lstm.bias_ih_l1",
        "lstm.bias_hh_l1",
        "lstm.weight_ih_l1_reverse",
        "lstm.weight_hh_l1_reverse",
        "lstm.bias_ih_l1_reverse",
        "lstm.bias_hh_l1_reverse",
    ];
    assert_eq!(layer.params().to_vec(), expected.to_vec());
    assert_eq!(layer.out_width(), 8);

    let mut g = Graph::new();
    let x = g.from_data(ramp(2 * 3 * 5, 0), &[2, 3, 5]);
    let out = layer.forward(&mut g, x, &mut Ctx::new(true, 7));
    assert_eq!(g.shape(out).as_slice(), &[2, 3, 8]);
    let loss = g.reduce_mean(out, &[0, 1, 2]);
    g.backward(loss).unwrap();
    for (slot, grad) in g.param_grads() {
        assert!(
            grad.iter().all(|v| v.is_finite()),
            "slot {slot} got a non-finite gradient"
        );
    }
}

#[test]
fn dropout_is_seeded_and_off_in_eval_mode() {
    let layer = Dropout::new(0.5);
    let data = vec![1.0f32; 64];
    let mut a = Graph::new();
    let x = a.from_data(data.clone(), &[64]);
    let dropped = layer.forward(&mut a, x, &mut Ctx::new(true, 11));
    let mut b = Graph::new();
    let y = b.from_data(data, &[64]);
    let again = layer.forward(&mut b, y, &mut Ctx::new(true, 11));
    assert_eq!(
        a.value(dropped),
        b.value(again),
        "the same seed must give the same mask"
    );
    assert!(a.value(dropped).contains(&0.0));
    let mut c = Graph::new();
    let z = c.from_data(vec![1.0f32; 64], &[64]);
    let kept = layer.forward(&mut c, z, &mut Ctx::eval());
    assert_eq!(c.value(z), c.value(kept), "eval mode is the identity");
}

// ----------------------------------------------------------------------------- the optimiser

#[test]
fn adamw_converges_on_a_quadratic() {
    let names = vec!["head.weight".to_string()];
    let mut params = vec![vec![5.0f32, -2.0, 3.5]];
    let mut opt = AdamW::new(0.3, 0.0);
    let schedule = Schedule::new(300);
    for s in 0..300 {
        opt.set_scale(schedule.at(s));
        let grads: Vec<Vec<f32>> = params
            .iter()
            .map(|p| p.iter().map(|w| 2.0 * (w - 3.0)).collect())
            .collect();
        opt.step(&mut params, &grads, &names);
    }
    assert_eq!(opt.step_count(), 300);
    for w in &params[0] {
        assert!(
            (w - 3.0).abs() < 1e-3,
            "adamw did not converge to 3: {params:?}"
        );
    }
}

#[test]
fn schedule_warms_up_then_decays_to_zero() {
    let schedule = Schedule::new(1000);
    assert_eq!(schedule.warmup, 50, "warm = int(0.05 * steps)");
    assert!(
        (schedule.at(0) - 0.02).abs() < 1e-6,
        "the first step is lr / warm"
    );
    for s in 0..schedule.warmup - 1 {
        assert!(schedule.at(s) < schedule.at(s + 1), "warmup must increase");
    }
    let peak = (0..schedule.steps)
        .map(|s| schedule.at(s))
        .fold(0.0f32, f32::max);
    assert!(
        peak > 0.99 && peak <= 1.0 + 1e-6,
        "the peak after warmup is {peak}"
    );
    assert!(
        schedule.at(schedule.steps - 1) < 1e-3,
        "the last step is ~0"
    );
    assert!(
        schedule.at(schedule.steps).abs() < 1e-6,
        "past the end the factor is 0"
    );
    assert_eq!(
        Schedule::new(0).warmup,
        1,
        "a one-step warmup is at least one step"
    );
}

#[test]
fn weight_decay_skips_biases_and_normalisation_parameters() {
    let names: Vec<String> = [
        "stages.0.0.0.weight",
        "stages.0.0.1.bias",
        "blocks.0.norm.weight",
        "queries.gamma",
        "queries.beta",
        "heat_queries",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let mut params: Vec<Vec<f32>> = names.iter().map(|_| vec![1.0f32; 3]).collect();
    let grads: Vec<Vec<f32>> = names.iter().map(|_| vec![0.0f32; 3]).collect();
    let mut opt = AdamW::new(1.0, 0.1);
    adamw_step(&mut params, &grads, &names, &mut opt);
    assert!(
        params[0].iter().all(|w| (w - 0.9).abs() < 1e-6),
        "a weight must decay by lr * wd: {params:?}"
    );
    assert!(
        params[5].iter().all(|w| (w - 0.9).abs() < 1e-6),
        "a bare parameter must decay"
    );
    for row in &params[1..5] {
        assert!(
            row.iter().all(|w| (w - 1.0).abs() < 1e-6),
            "bias/norm parameters must not decay"
        );
    }
    assert!(decays("stages.0.0.0.weight"));
    assert!(!decays("stages.0.0.1.bias"));
    assert!(!decays("blocks.0.norm.weight"));
}
