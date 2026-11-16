//! Forward kernels and their gradients.
//!
//! Every kernel takes node ids, reads the operands off the [`Graph`] and returns `(shape, data)`; the
//! matching gradient is a case of [`backward`], which is handed the output node's accumulated gradient
//! and pushes into the parents'. Shapes broadcast numpy-style wherever the Python operation does.

use crate::tensor::kernels::{self, ConvShape};
use crate::tensor::{Graph, NodeId, Op, Shape};
use std::cmp::max;

/// Which reduction a pooling or reduce node performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceKind {
    /// Sum of the reduced elements.
    Sum,
    /// Sum divided by the number of reduced elements.
    Mean,
    /// Largest reduced element.
    Max,
}

/// Which pointwise loss a [`crate::tensor::Op::WeightedLoss`] node computes on its difference input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossKind {
    /// `F.l1_loss`.
    L1,
    /// `F.smooth_l1_loss` with `beta = 1`.
    SmoothL1,
    /// `F.mse_loss`.
    Mse,
}

/// Batch-normalisation statistics: the batch's own while training, fixed running values while evaluating.
#[derive(Clone, Debug)]
pub enum BnMode {
    /// `nn.BatchNorm*` in training mode: batch statistics, and the caller updates its running buffers.
    Train,
    /// `nn.BatchNorm*` in eval mode: the stored running mean and variance.
    Eval {
        /// Per-channel running mean.
        mean: Vec<f32>,
        /// Per-channel running (biased) variance.
        var: Vec<f32>,
    },
}

pub(crate) fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Row-major strides for a shape.
pub(crate) fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// The broadcast result of two shapes (numpy rules), or `None` when they do not broadcast.
pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Shape> {
    let n = max(a.len(), b.len());
    let mut out = vec![0usize; n];
    for i in 0..n {
        let x = if i + a.len() < n {
            1
        } else {
            a[i + a.len() - n]
        };
        let y = if i + b.len() < n {
            1
        } else {
            b[i + b.len() - n]
        };
        if x != y && x != 1 && y != 1 {
            return None;
        }
        out[i] = max(x, y);
    }
    Some(out)
}

/// Strides used to read `shape` while walking `out`; axes broadcast in have stride 0, so the
/// broadcast kernels can address both operands with output-rank indices.
fn bcast_strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let st = strides(shape);
    let off = out.len() - shape.len();
    let mut s = vec![0usize; out.len()];
    for i in 0..shape.len() {
        s[i + off] = if shape[i] == out[i + off] { st[i] } else { 0 };
    }
    s
}

/// Splits `axes` into the sorted unique reduction axes, the output shape and the axes that survive.
fn split_axes(shape: &[usize], axes: &[usize]) -> (Vec<usize>, Shape, Vec<usize>) {
    let mut ax = axes.to_vec();
    ax.sort_unstable();
    ax.dedup();
    for a in &ax {
        assert!(
            *a < shape.len(),
            "reduce axis {a} out of range for {shape:?}"
        );
    }
    let kept: Vec<usize> = (0..shape.len()).filter(|i| !ax.contains(i)).collect();
    let out: Shape = kept.iter().map(|i| shape[*i]).collect();
    (ax, out, kept)
}

/// The output coordinate of the input element `src` when the axes in `kept` are kept in order.
fn reduce_index(src: usize, ss: &[usize], kept: &[usize], ks: &[usize]) -> usize {
    let mut rest = src;
    let mut dst = 0usize;
    let mut at = 0usize;
    for (i, o) in ss.iter().enumerate() {
        let c = rest / o;
        rest %= o;
        if at < kept.len() && kept[at] == i {
            dst += c * ks[at];
            at += 1;
        }
    }
    dst
}

pub(crate) fn elementwise2(g: &mut Graph, a: NodeId, b: NodeId, op: Op) -> NodeId {
    let (sa, sb) = (g.shape(a).clone(), g.shape(b).clone());
    let out = broadcast_shape(&sa, &sb).expect("operands do not broadcast");
    let f: fn(f32, f32) -> f32 = match op {
        Op::Add(..) => |x, y| x + y,
        Op::Sub(..) => |x, y| x - y,
        _ => |x, y| x * y,
    };
    let (va, vb) = (g.value(a), g.value(b));
    let data = if sa == out && sb == out {
        kernels::zip_map(va, vb, f)
    } else {
        let (as_, bs) = (bcast_strides(&sa, &out), bcast_strides(&sb, &out));
        kernels::broadcast_map(&out, &as_, &bs, va, vb, f)
    };
    g.push(out, data, op)
}

/// Mirrors `torch.softmax` (numerically stabilised by subtracting the row maximum).
pub(crate) fn softmax_data(g: &Graph, a: NodeId, axis: usize) -> (Shape, Vec<f32>) {
    let shape = g.shape(a).clone();
    assert!(
        axis < shape.len(),
        "softmax axis {axis} out of range for {shape:?}"
    );
    let c = shape[axis];
    let outer = shape[..axis].iter().product::<usize>().max(1);
    let inner = shape[axis + 1..].iter().product::<usize>().max(1);
    let x = g.value(a);
    let mut out = vec![0.0f32; x.len()];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * c * inner + i;
            let mut m = f32::NEG_INFINITY;
            for k in 0..c {
                m = m.max(x[base + k * inner]);
            }
            let mut s = 0.0;
            for k in 0..c {
                let e = (x[base + k * inner] - m).exp();
                out[base + k * inner] = e;
                s += e;
            }
            let inv = 1.0 / s;
            for k in 0..c {
                out[base + k * inner] *= inv;
            }
        }
    }
    (shape, out)
}

/// Number of matrix batches in a `(..., M, K)` shape: `1` when there is no batch dimension.
fn batch_count(shape: &[usize]) -> usize {
    if shape.len() <= 2 {
        1
    } else {
        shape[..shape.len() - 2].iter().product()
    }
}

/// Mirrors `torch.matmul` for the two layouts the networks use: `(M,K) x (K,N)` and
/// `(B,M,K) x (B,K,N)`, where a batch of one broadcasts against a real batch.
pub(crate) fn matmul_data(g: &Graph, a: NodeId, b: NodeId) -> (Shape, Vec<f32>) {
    let (sa, sb) = (g.shape(a).clone(), g.shape(b).clone());
    let (m, k) = (sa[sa.len() - 2], sa[sa.len() - 1]);
    let n = sb[sb.len() - 1];
    assert_eq!(
        k,
        sb[sb.len() - 2],
        "matmul inner sizes differ: {sa:?} x {sb:?}"
    );
    let (ba, bb) = (batch_count(&sa), batch_count(&sb));
    assert!(
        ba == bb || ba == 1 || bb == 1,
        "matmul batch sizes {ba} and {sb:?} do not broadcast"
    );
    let batch = max(ba, bb);
    let out_shape = if batch == 1 {
        vec![m, n]
    } else {
        vec![batch, m, n]
    };
    let x = g.value(a);
    let y = g.value(b);
    let mut data = vec![0.0f32; numel(&out_shape)];
    let inner = m * n;
    if inner > 0 {
        kernels::for_chunks_par(&mut data, inner, batch > 1, |bi, c| {
            let oa = if ba == 1 { 0 } else { bi * m * k };
            let ob = if bb == 1 { 0 } else { bi * k * n };
            kernels::gemm(m, k, n, &x[oa..], (k, 1), &y[ob..], (n, 1), 0.0, c, (n, 1));
        });
    }
    (out_shape, data)
}

/// Mirrors `torch.nn.functional.conv2d` with square kernel, stride, padding and dilation.
pub(crate) fn conv2d_data(
    g: &Graph,
    x: NodeId,
    w: NodeId,
    b: Option<NodeId>,
    stride: usize,
    pad: usize,
    dilation: usize,
) -> (Shape, Vec<f32>) {
    let (xs, ws) = (g.shape(x).clone(), g.shape(w).clone());
    assert_eq!(xs.len(), 4, "conv2d expects (N,C,H,W), got {xs:?}");
    assert_eq!(ws.len(), 4, "conv2d expects (O,C,kH,kW), got {ws:?}");
    let (n, c, h, iw) = (xs[0], xs[1], xs[2], xs[3]);
    let (o, kc, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
    assert_eq!(kc, c, "conv2d channel mismatch: {xs:?} x {ws:?}");
    let eff = (kh - 1) * dilation + 1;
    let effw = (kw - 1) * dilation + 1;
    let (oh, ow) = (
        (h + 2 * pad - eff) / stride + 1,
        (iw + 2 * pad - effw) / stride + 1,
    );
    let xv = g.value(x);
    let wv = g.value(w);
    let bv: &[f32] = b.map(|id| g.value(id)).unwrap_or(&[]);
    assert!(
        bv.is_empty() || bv.len() == o,
        "conv2d bias has {} entries, expected {}",
        bv.len(),
        o
    );
    let geom = ConvShape {
        n,
        c,
        h,
        w: iw,
        o,
        kh,
        kw,
        stride,
        pad,
        dilation,
        oh,
        ow,
    };
    let data = kernels::conv2d_forward(xv, wv, bv, &geom);
    (vec![n, o, oh, ow], data)
}

/// Mirrors `torch.nn.functional.max_pool2d` / `avg_pool2d` (square window); the average divides by the
/// whole window, like PyTorch's `count_include_pad=True`.
pub(crate) fn pool_data(
    g: &Graph,
    x: NodeId,
    k: usize,
    stride: usize,
    pad: usize,
    kind: ReduceKind,
) -> (Shape, Vec<f32>) {
    let xs = g.shape(x).clone();
    assert_eq!(xs.len(), 4, "pooling expects (N,C,H,W), got {xs:?}");
    let (n, c, h, w) = (xs[0], xs[1], xs[2], xs[3]);
    let oh = (h + 2 * pad - k) / stride + 1;
    let ow = (w + 2 * pad - k) / stride + 1;
    let xv = g.value(x);
    let mut data = vec![0.0f32; n * c * oh * ow];
    // One task per `(n, c)` plane.
    let par = xv.len() >= kernels::PAR_MIN;
    kernels::for_chunks_par(&mut data, oh * ow, par, |plane, out| {
        let src = &xv[plane * h * w..(plane + 1) * h * w];
        for y in 0..oh {
            for xi in 0..ow {
                let mut acc = 0.0f32;
                let mut best = f32::NEG_INFINITY;
                for ky in 0..k {
                    let iy = y * stride + ky;
                    if iy < pad || iy >= h + pad {
                        continue;
                    }
                    let iy = iy - pad;
                    for kx in 0..k {
                        let ix = xi * stride + kx;
                        if ix < pad || ix >= w + pad {
                            continue;
                        }
                        let v = src[iy * w + ix - pad];
                        acc += v;
                        if v > best {
                            best = v;
                        }
                    }
                }
                out[y * ow + xi] = match kind {
                    ReduceKind::Max => best,
                    ReduceKind::Mean => acc / (k * k) as f32,
                    ReduceKind::Sum => acc,
                };
            }
        }
    });
    (vec![n, c, oh, ow], data)
}

/// Mirrors `torch.nn.functional.interpolate(size=..., mode="nearest")` for a `(N,C,H,W)` input.
pub(crate) fn upsample_nearest_data(g: &Graph, x: NodeId, h: usize, w: usize) -> (Shape, Vec<f32>) {
    let xs = g.shape(x).clone();
    assert_eq!(xs.len(), 4, "upsampling expects (N,C,H,W), got {xs:?}");
    let (n, c, ih, iw) = (xs[0], xs[1], xs[2], xs[3]);
    let xv = g.value(x);
    let mut data = vec![0.0f32; n * c * h * w];
    let cols: Vec<usize> = (0..w).map(|xi| xi * iw / w).collect();
    kernels::for_chunks(&mut data, h * w, |plane, out| {
        let src = &xv[plane * ih * iw..(plane + 1) * ih * iw];
        for y in 0..h {
            let sy = y * ih / h;
            let row = &src[sy * iw..(sy + 1) * iw];
            for (o, sx) in out[y * w..(y + 1) * w].iter_mut().zip(&cols) {
                *o = row[*sx];
            }
        }
    });
    (vec![n, c, h, w], data)
}

/// Mirrors `torch.Tensor.permute`.
pub(crate) fn permute_data(g: &Graph, a: NodeId, perm: &[usize]) -> (Shape, Vec<f32>) {
    let s = g.shape(a).clone();
    assert_eq!(perm.len(), s.len(), "permute {perm:?} does not fit {s:?}");
    let out: Shape = perm.iter().map(|p| s[*p]).collect();
    let si = strides(&s);
    let o = strides(&out);
    let xv = g.value(a);
    let mut data = vec![0.0f32; xv.len()];
    for (dst, v) in data.iter_mut().enumerate() {
        *v = xv[permute_index(dst, &o, &si, perm)];
    }
    (out, data)
}

/// The input element that output element `dst` of a permutation comes from.
fn permute_index(dst: usize, out_strides: &[usize], in_strides: &[usize], perm: &[usize]) -> usize {
    let mut rest = dst;
    let mut src = 0usize;
    for (i, oo) in out_strides.iter().enumerate() {
        let c = rest / oo;
        rest %= oo;
        src += c * in_strides[perm[i]];
    }
    src
}

/// Mirrors `torch.cat`.
pub(crate) fn concat_data(g: &Graph, parts: &[NodeId], axis: usize) -> (Shape, Vec<f32>) {
    let out = concat_shape(g, parts, axis);
    let os = strides(&out);
    let mut data = vec![0.0f32; numel(&out)];
    let mut base = 0usize;
    for p in parts {
        let ps = g.shape(*p).clone();
        let pv = g.value(*p);
        let pstr = strides(&ps);
        for (src, v) in pv.iter().enumerate() {
            data[concat_index(src, &pstr, &os, axis, base)] = *v;
        }
        base += ps[axis];
    }
    (out, data)
}

/// Output shape of a concatenation: every axis but `axis` must already agree.
fn concat_shape(g: &Graph, parts: &[NodeId], axis: usize) -> Shape {
    assert!(!parts.is_empty(), "concat needs at least one part");
    let first = g.shape(parts[0]).clone();
    assert!(
        axis < first.len(),
        "concat axis {axis} out of range for {first:?}"
    );
    let mut out = first.clone();
    let mut total = 0usize;
    for p in parts {
        let ps = g.shape(*p).clone();
        assert_eq!(
            ps.len(),
            first.len(),
            "concat parts have different ranks: {ps:?} {first:?}"
        );
        for (i, d) in first.iter().enumerate() {
            if i != axis {
                assert_eq!(
                    ps[i], *d,
                    "concat parts disagree on axis {i}: {ps:?} {first:?}"
                );
            }
        }
        total += ps[axis];
    }
    out[axis] = total;
    out
}

/// The output element that part element `src` of a concatenation lands on.
fn concat_index(src: usize, pstr: &[usize], os: &[usize], axis: usize, base: usize) -> usize {
    let mut rest = src;
    let mut at = 0usize;
    let rank = pstr.len();
    for i in 0..rank {
        let c = rest / pstr[i];
        rest %= pstr[i];
        let c = if i == axis { c + base } else { c };
        at += c * os[i];
    }
    at
}

/// Mirrors `torch.Tensor.__getitem__` with a `slice(start, end)` on one axis.
pub(crate) fn slice_data(
    g: &Graph,
    a: NodeId,
    axis: usize,
    start: usize,
    end: usize,
) -> (Shape, Vec<f32>) {
    let s = g.shape(a).clone();
    assert!(axis < s.len(), "slice axis {axis} out of range for {s:?}");
    assert!(
        end <= s[axis] && start <= end,
        "slice {start}..{end} does not fit {s:?}"
    );
    let mut out = s.clone();
    out[axis] = end - start;
    let os = strides(&s);
    let ns = strides(&out);
    let xv = g.value(a);
    let mut data = vec![0.0f32; numel(&out)];
    for dst in 0..numel(&out) {
        data[dst] = xv[slice_index(dst, &ns, &os, axis, start)];
    }
    (out, data)
}

/// The input element a sliced output element came from.
fn slice_index(dst: usize, ns: &[usize], os: &[usize], axis: usize, start: usize) -> usize {
    let mut rest = dst;
    let mut src = 0usize;
    for (i, oo) in ns.iter().enumerate() {
        let c = rest / oo;
        rest %= oo;
        let c = if i == axis { c + start } else { c };
        src += c * os[i];
    }
    src
}

/// `(outer, reduced, inner)` sizes when the sorted reduction axes `ax` form one contiguous run, so
/// the input reads as `(outer, reduced, inner)` and the output as `(outer, inner)`.
fn contiguous_axes(s: &[usize], ax: &[usize]) -> Option<(usize, usize, usize)> {
    let (&first, &last) = (ax.first()?, ax.last()?);
    if last - first + 1 != ax.len() {
        return None;
    }
    let outer = s[..first].iter().product();
    let red = s[first..=last].iter().product();
    let inner = s[last + 1..].iter().product();
    Some((outer, red, inner))
}

/// Number of elements folded into each output slot of a reduction over `ax`.
fn reduced_count(s: &[usize], ax: &[usize]) -> usize {
    ax.iter().map(|a| s[*a]).product()
}

/// Mirrors `torch.Tensor.sum` / `torch.Tensor.mean` over `axes`.
pub(crate) fn reduce_data(
    g: &Graph,
    a: NodeId,
    axes: &[usize],
    kind: ReduceKind,
) -> (Shape, Vec<f32>) {
    let s = g.shape(a).clone();
    let (ax, out, kept) = split_axes(&s, axes);
    let ss = strides(&s);
    let ks = strides(&out);
    let xv = g.value(a);
    if kind != ReduceKind::Max {
        if let Some((outer, red, inner)) = contiguous_axes(&s, &ax) {
            let mut data = if inner == 1 {
                // Each output sums one contiguous run.
                let par = xv.len() >= kernels::PAR_MIN;
                kernels::par_map_range(outer, par, |o| {
                    kernels::lane_sum(&xv[o * red..(o + 1) * red], |v| v)
                })
            } else {
                let mut data = vec![0.0f32; outer * inner];
                kernels::for_chunks(&mut data, inner, |o, out| {
                    for r in 0..red {
                        let at = (o * red + r) * inner;
                        for (d, v) in out.iter_mut().zip(&xv[at..at + inner]) {
                            *d += v;
                        }
                    }
                });
                data
            };
            if kind == ReduceKind::Mean {
                let n = reduced_count(&s, &ax).max(1) as f32;
                for v in data.iter_mut() {
                    *v /= n;
                }
            }
            return (out, data);
        }
    }
    let mut data = vec![
        if kind == ReduceKind::Max {
            f32::NEG_INFINITY
        } else {
            0.0
        };
        numel(&out)
    ];
    for (src, v) in xv.iter().enumerate() {
        let at = reduce_index(src, &ss, &kept, &ks);
        match kind {
            ReduceKind::Sum | ReduceKind::Mean => data[at] += *v,
            ReduceKind::Max => data[at] = data[at].max(*v),
        }
    }
    if kind == ReduceKind::Mean {
        let n = reduced_count(&s, &ax).max(1) as f32;
        for v in data.iter_mut() {
            *v /= n;
        }
    }
    (out, data)
}

/// Mirrors `torch.nn.functional.normalize(x, dim=axis)` (L2, with PyTorch's `eps` floor on the norm).
pub(crate) fn l2_normalize_data(g: &Graph, a: NodeId, axis: usize) -> (Shape, Vec<f32>) {
    let s = g.shape(a).clone();
    assert!(
        axis < s.len(),
        "normalize axis {axis} out of range for {s:?}"
    );
    let c = s[axis];
    let outer = s[..axis].iter().product::<usize>().max(1);
    let inner = s[axis + 1..].iter().product::<usize>().max(1);
    let xv = g.value(a);
    let mut data = xv.to_vec();
    for o in 0..outer {
        for i in 0..inner {
            let base = o * c * inner + i;
            let norm = norm_of(xv, base, inner, c);
            for k in 0..c {
                data[base + k * inner] /= norm;
            }
        }
    }
    (s, data)
}

/// The L2 norm of one row, floored at PyTorch's `F.normalize` `eps`.
fn norm_of(x: &[f32], base: usize, inner: usize, c: usize) -> f32 {
    let sum: f32 = (0..c)
        .map(|k| x[base + k * inner] * x[base + k * inner])
        .sum();
    sum.sqrt().max(1e-12)
}

pub(crate) fn dropout_data(g: &Graph, a: NodeId, p: f32, seed: u64) -> (Shape, Vec<f32>) {
    let s = g.shape(a).clone();
    let xv = g.value(a);
    let mut rng = SplitMix::new(seed);
    let keep = 1.0 / (1.0 - p);
    let data = xv
        .iter()
        .map(|v| if rng.next() < p { 0.0 } else { v * keep })
        .collect();
    (s, data)
}

/// Mirrors `torch.nn.functional.binary_cross_entropy_with_logits(..., reduction="mean")`.
pub(crate) fn bce_data(g: &Graph, logits: NodeId, target: NodeId) -> (Shape, Vec<f32>) {
    let (l, t) = (g.value(logits), g.value(target));
    assert_eq!(
        l.len(),
        t.len(),
        "bce logits {:?} and target {:?} differ",
        g.shape(logits),
        g.shape(target)
    );
    let n = l.len() as f32;
    let sum: f32 = l
        .iter()
        .zip(t)
        .map(|(z, y)| {
            let z: f32 = *z;
            z.max(0.0) - z * y + (1.0 + (-z.abs()).exp()).ln()
        })
        .sum();
    (vec![1], vec![sum / n])
}

/// Mirrors `F.l1_loss` / `F.smooth_l1_loss` / `F.mse_loss` (`reduction="mean"`) scaled by `weight`.
pub(crate) fn weighted_loss_data(
    g: &Graph,
    diff: NodeId,
    kind: LossKind,
    weight: f32,
) -> (Shape, Vec<f32>) {
    let d = g.value(diff);
    let n = d.len() as f32;
    let sum: f32 = d
        .iter()
        .map(|v| match kind {
            LossKind::L1 => v.abs(),
            LossKind::SmoothL1 => {
                let a = v.abs();
                if a < 1.0 {
                    0.5 * v * v
                } else {
                    a - 0.5
                }
            }
            LossKind::Mse => v * v,
        })
        .sum();
    (vec![1], vec![weight * sum / n])
}

/// Mirrors `torch.nn.CrossEntropyLoss(weight=..., label_smoothing=..., ignore_index=...,
/// reduction="mean")`: the weighted mean of `(1 - smoothing) * w_t * nll_t + smoothing * sum_k w_k *
/// -log p_k / sum_k w_k` over the samples that are not `ignore`.
pub(crate) fn cross_entropy_data(
    g: &Graph,
    logits: NodeId,
    target: NodeId,
    weight: Option<NodeId>,
    smoothing: f32,
    ignore: i32,
) -> (Shape, Vec<f32>) {
    let ls = g.shape(logits).clone();
    let c = *ls.last().unwrap();
    let n = numel(&ls) / c;
    let lv = g.value(logits);
    let tv = g.value(target);
    assert_eq!(
        tv.len(),
        n,
        "cross entropy has {} logit rows but {} targets",
        n,
        tv.len()
    );
    let wv = weight
        .map(|id| g.value(id).to_vec())
        .unwrap_or_else(|| vec![1.0; c]);
    let mut total = 0.0f32;
    let mut used = 0.0f32;
    for (i, &target) in tv.iter().enumerate() {
        if target as i32 == ignore {
            continue;
        }
        let t = target as usize;
        let p = probs(lv, i * c, c);
        let wt = wv.get(t).copied().unwrap_or(1.0);
        let nll = wt * -(p[t].max(1e-20)).ln();
        let loss = if smoothing > 0.0 {
            let smooth: f32 = p
                .iter()
                .enumerate()
                .map(|(k, q)| wv.get(k).copied().unwrap_or(1.0) * -(q.max(1e-20)).ln())
                .sum::<f32>()
                / c as f32;
            (1.0 - smoothing) * nll + smoothing * smooth
        } else {
            nll
        };
        total += loss;
        used += wt;
    }
    (vec![1], vec![total / used.max(1.0)])
}

/// Softmax probabilities of the row of `lv` starting at `at`.
fn probs(lv: &[f32], at: usize, c: usize) -> Vec<f32> {
    let row = &lv[at..at + c];
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|z| (*z - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

/// Sum of the class weights, defaulting to `C` when no weight vector was given.
fn weight_sum(wv: &[f32]) -> f32 {
    wv.iter().sum::<f32>().max(1e-20)
}

/// Mirrors `nn.BatchNorm2d` / `BatchNorm1d` forward: channel-wise normalisation of an `(N,C,*)` tensor.
pub(crate) fn batch_norm_data(
    g: &Graph,
    x: NodeId,
    gamma: NodeId,
    beta: NodeId,
    eps: f32,
    mode: &BnMode,
    relu: bool,
) -> (Shape, Vec<f32>) {
    let xs = g.shape(x).clone();
    assert!(xs.len() >= 2, "batch norm needs (N,C,*), got {xs:?}");
    let c = xs[1];
    let inner = xs[2..].iter().product::<usize>().max(1);
    let n = xs[0];
    let xv = g.value(x);
    let gv = g.value(gamma);
    let bv = g.value(beta);
    assert_eq!(
        gv.len(),
        c,
        "batch norm gamma has {} entries, expected {}",
        gv.len(),
        c
    );
    assert_eq!(
        bv.len(),
        c,
        "batch norm beta has {} entries, expected {}",
        bv.len(),
        c
    );
    let (mean, var) = match mode {
        BnMode::Train => batch_stats_from(xv, n, c, inner),
        BnMode::Eval { mean, var } => (mean.clone(), var.clone()),
    };
    let (sc, sh): (Vec<f32>, Vec<f32>) = (0..c)
        .map(|ch| {
            let std = (var[ch] + eps).sqrt();
            let sc = gv[ch] / std;
            (sc, bv[ch] - mean[ch] * sc)
        })
        .unzip();
    let mut data = vec![0.0f32; xv.len()];
    // One task per `(n, c)` plane.
    kernels::for_chunks(&mut data, inner, |plane, out| {
        let ch = plane % c;
        let (k, b) = (sc[ch], sh[ch]);
        let src = &xv[plane * inner..(plane + 1) * inner];
        if relu {
            for (o, x) in out.iter_mut().zip(src) {
                *o = (x * k + b).max(0.0);
            }
        } else {
            for (o, x) in out.iter_mut().zip(src) {
                *o = x * k + b;
            }
        }
    });
    (xs, data)
}

/// Momentum update of running statistics, matching `nn.BatchNorm`'s buffer update.
pub fn update_running(running: &mut [f32], batch: &[f32], momentum: f32) {
    for (r, b) in running.iter_mut().zip(batch) {
        *r = (1.0 - momentum) * *r + momentum * *b;
    }
}

/// Statistics of a batch, for [`update_running`]: `(mean, var)` per channel of an `(N,C,*)` tensor.
/// The variance is biased, like `nn.BatchNorm`'s `running_var`.
pub fn batch_stats(g: &Graph, x: NodeId) -> (Vec<f32>, Vec<f32>) {
    let xs = g.shape(x).clone();
    let (n, c) = (xs[0], xs[1]);
    let inner = xs[2..].iter().product::<usize>().max(1);
    batch_stats_from(g.value(x), n, c, inner)
}

/// Per-channel mean and biased variance of an `(n, c, inner)` layout.
fn batch_stats_from(x: &[f32], n: usize, c: usize, inner: usize) -> (Vec<f32>, Vec<f32>) {
    let m = (n * inner).max(1) as f32;
    // Two passes per channel (mean, then the centred squares), one channel per task; the sums run in
    // independent lanes so they vectorise.
    let stats = |ch: usize| {
        let plane = |ni: usize| &x[(ni * c + ch) * inner..(ni * c + ch + 1) * inner];
        let s: f32 = (0..n).map(|ni| kernels::lane_sum(plane(ni), |v| v)).sum();
        let mean = s / m;
        let v: f32 = (0..n)
            .map(|ni| {
                kernels::lane_sum(plane(ni), |v| {
                    let d = v - mean;
                    d * d
                })
            })
            .sum();
        (mean, v / m)
    };
    kernels::par_map_range(c, x.len() >= kernels::PAR_MIN, stats)
        .into_iter()
        .unzip()
}

/// A tiny deterministic generator, so a seeded dropout mask reproduces across platforms.
pub(crate) struct SplitMix(u64);

impl SplitMix {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    /// A value in `[0, 1)`.
    pub(crate) fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A value in `[-limit, limit)`, the shape of PyTorch's default `U(-1/sqrt(fan_in), ...)` init.
    pub(crate) fn symmetric(&mut self, limit: f32) -> f32 {
        (self.next() * 2.0 - 1.0) * limit
    }
}

/// The four weight blocks of one LSTM layer in one direction (`weight_ih`, `weight_hh`, `bias_ih`,
/// `bias_hh`), gate order `i, f, g, o` like `torch.nn.LSTM`.
struct LstmWeights<'a> {
    wih: &'a [f32],
    whh: &'a [f32],
    bih: &'a [f32],
    bhh: &'a [f32],
}

/// Geometry of one LSTM direction run over an `(N, T, input)` batch.
#[derive(Clone, Copy)]
struct LstmRun {
    n: usize,
    t: usize,
    input: usize,
    hidden: usize,
    reverse: bool,
    /// Columns of the layer output (`hidden * directions`).
    cols: usize,
    /// Column this direction writes.
    col: usize,
}

/// Saved activations of one direction, in the direction's own time order: the four gates, the cell
/// state, `tanh(c)` and `h`, `hidden` values per `(n, step)` (gates four times that).
struct DirState {
    gates: Vec<f32>,
    cells: Vec<f32>,
    tanh_c: Vec<f32>,
    hs: Vec<f32>,
}

impl DirState {
    fn new(n: usize, t: usize, hidden: usize) -> Self {
        Self {
            gates: vec![0.0f32; n * t * 4 * hidden],
            cells: vec![0.0f32; n * t * hidden],
            tanh_c: vec![0.0f32; n * t * hidden],
            hs: vec![0.0f32; n * t * hidden],
        }
    }
}

/// The whole LSTM stack: the output plus everything [`lstm_backward`] needs to differentiate it.
struct LstmForward {
    shape: Shape,
    out: Vec<f32>,
    /// Input of each layer (the first is `x` itself), `(N, T, in_size)` per layer.
    layer_in: Vec<Vec<f32>>,
    /// Saved activations, indexed `layer * directions + direction`.
    states: Vec<DirState>,
}

/// Runs `layers` LSTM layers, each with a forward and (when `bidirectional`) a reverse direction.
fn lstm_forward(
    g: &Graph,
    x: NodeId,
    w: &[NodeId],
    hidden: usize,
    layers: usize,
    bidirectional: bool,
) -> LstmForward {
    let xs = g.shape(x).clone();
    assert_eq!(xs.len(), 3, "lstm expects (N,T,input), got {xs:?}");
    let (n, t, input) = (xs[0], xs[1], xs[2]);
    let dir = usize::from(bidirectional) + 1;
    assert_eq!(
        w.len(),
        layers * dir * 4,
        "a {} layer (bidirectional {}) lstm needs {} weight blocks, got {}",
        layers,
        bidirectional,
        layers * dir * 4,
        w.len()
    );
    let mut layer_in = Vec::with_capacity(layers);
    let mut states = Vec::with_capacity(layers * dir);
    let mut cur = g.value(x).to_vec();
    let mut in_size = input;
    let mut out = Vec::new();
    for l in 0..layers {
        layer_in.push(cur.clone());
        out = vec![0.0f32; n * t * hidden * dir];
        // The two directions are independent: each runs on its own thread into its own `(N, T, H)`
        // buffer, interleaved into the layer output afterwards.
        let runs = kernels::par_map_range(dir, dir > 1, |d| {
            let base = (l * dir + d) * 4;
            let weights = LstmWeights {
                wih: g.value(w[base]),
                whh: g.value(w[base + 1]),
                bih: g.value(w[base + 2]),
                bhh: g.value(w[base + 3]),
            };
            let run = LstmRun {
                n,
                t,
                input: in_size,
                hidden,
                reverse: d == 1,
                cols: hidden,
                col: 0,
            };
            let mut h = vec![0.0f32; n * t * hidden];
            let mut state = DirState::new(n, t, hidden);
            lstm_dir(&cur, &weights, &run, &mut h, Some(&mut state));
            (h, state)
        });
        for (d, (h, state)) in runs.into_iter().enumerate() {
            for (row, src) in out.chunks_mut(hidden * dir).zip(h.chunks(hidden)) {
                row[d * hidden..(d + 1) * hidden].copy_from_slice(src);
            }
            states.push(state);
        }
        cur = out.clone();
        in_size = hidden * dir;
    }
    LstmForward {
        shape: vec![n, t, hidden * dir],
        out,
        layer_in,
        states,
    }
}

/// Copies `src` into `dst[at..]`.
fn put(dst: &mut [f32], at: usize, src: &[f32]) {
    dst[at..at + src.len()].copy_from_slice(src);
}

/// One direction of one LSTM layer over `(N, T, input)`, writing `h` into the `col`..`col + hidden`
/// columns of `out` and (when `state` is given) the activations used by the backward pass.
///
/// Mirrors one direction of `torch.nn.LSTM`: `c = f * c_prev + i * g`, `h = o * tanh(c)`, both states
/// starting at zero (`nn.LSTM`'s default initial state). Like PyTorch, the input projection
/// `x W_ihᵀ + b_ih + b_hh` of every step is one GEMM up front, and each step then adds `h W_hhᵀ` for
/// the whole batch at once.
fn lstm_dir(
    x: &[f32],
    w: &LstmWeights,
    r: &LstmRun,
    out: &mut [f32],
    state: Option<&mut DirState>,
) {
    let LstmRun {
        n,
        t,
        input,
        hidden,
        reverse,
        cols,
        col,
    } = *r;
    let gates_len = 4 * hidden;
    // Row `ni * t + s`: the input part of the gates of sequence `ni` at time `s`.
    let mut pre = vec![0.0f32; n * t * gates_len];
    kernels::gemm(
        n * t,
        input,
        gates_len,
        x,
        (input, 1),
        w.wih,
        (1, input),
        0.0,
        &mut pre,
        (gates_len, 1),
    );
    for row in pre.chunks_mut(gates_len) {
        for ((v, bi), bh) in row.iter_mut().zip(w.bih).zip(w.bhh) {
            *v += bi + bh;
        }
    }
    let mut gates = vec![0.0f32; n * gates_len];
    let mut h = vec![0.0f32; n * hidden];
    let mut c = vec![0.0f32; n * hidden];
    let mut state = state;
    for step in 0..t {
        let s = if reverse { t - 1 - step } else { step };
        for ni in 0..n {
            let at = (ni * t + s) * gates_len;
            gates[ni * gates_len..(ni + 1) * gates_len].copy_from_slice(&pre[at..at + gates_len]);
        }
        if step > 0 {
            // gates (n x 4H) += h (n x H) · W_hhᵀ (H x 4H)
            kernels::gemm(
                n,
                hidden,
                gates_len,
                &h,
                (hidden, 1),
                w.whh,
                (1, hidden),
                1.0,
                &mut gates,
                (gates_len, 1),
            );
        }
        for ni in 0..n {
            let g = &mut gates[ni * gates_len..(ni + 1) * gates_len];
            let (hn, cn) = (
                &mut h[ni * hidden..(ni + 1) * hidden],
                &mut c[ni * hidden..(ni + 1) * hidden],
            );
            let out_at = (ni * t + s) * cols + col;
            for j in 0..hidden {
                let ig = 1.0 / (1.0 + (-g[j]).exp());
                let fg = 1.0 / (1.0 + (-g[hidden + j]).exp());
                let cg = g[2 * hidden + j].tanh();
                let og = 1.0 / (1.0 + (-g[3 * hidden + j]).exp());
                cn[j] = fg * cn[j] + ig * cg;
                hn[j] = og * cn[j].tanh();
                g[j] = ig;
                g[hidden + j] = fg;
                g[2 * hidden + j] = cg;
                g[3 * hidden + j] = og;
                out[out_at + j] = hn[j];
            }
            if let Some(st) = state.as_mut() {
                let saved = (ni * t + step) * hidden;
                put(&mut st.gates, saved * 4, g);
                put(&mut st.cells, saved, cn);
                put(&mut st.tanh_c, saved, &tanh_of(cn));
                put(&mut st.hs, saved, hn);
            }
        }
    }
}

/// `tanh` of every element, as a temporary (the borrow checker dislikes in-place updates here).
fn tanh_of(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x.tanh()).collect()
}

/// Mirrors `torch.nn.LSTM` forward; see [`lstm_forward`] for the layout of the weight blocks.
pub(crate) fn lstm_data(
    g: &Graph,
    x: NodeId,
    w: &[NodeId],
    hidden: usize,
    layers: usize,
    bidirectional: bool,
) -> (Shape, Vec<f32>) {
    let fwd = lstm_forward(g, x, w, hidden, layers, bidirectional);
    (fwd.shape, fwd.out)
}

/// Gradients of the four weight blocks of one LSTM direction.
struct DirGrad {
    wih: Vec<f32>,
    whh: Vec<f32>,
    bih: Vec<f32>,
    bhh: Vec<f32>,
}

impl DirGrad {
    fn new(wih_len: usize, whh_len: usize, hidden: usize) -> Self {
        Self {
            wih: vec![0.0f32; wih_len],
            whh: vec![0.0f32; whh_len],
            bih: vec![0.0f32; 4 * hidden],
            bhh: vec![0.0f32; 4 * hidden],
        }
    }
}

/// Adds `from` into `to` element-wise.
fn accumulate(to: &mut [f32], from: &[f32]) {
    for (a, b) in to.iter_mut().zip(from) {
        *a += b;
    }
}

/// Truncated backpropagation through time and across the layers of [`lstm_forward`].
///
/// `grad` is the gradient of the `(N, T, hidden * directions)` output. Both directions of every layer
/// are differentiated, the top layer first, feeding the gradient down through the layer inputs to `x`
/// and to all `4 * directions` weight blocks of every layer.
fn lstm_backward(
    g: &mut Graph,
    x: NodeId,
    w: &[NodeId],
    hidden: usize,
    layers: usize,
    bidirectional: bool,
    grad: &[f32],
) {
    let fwd = lstm_forward(g, x, w, hidden, layers, bidirectional);
    let xs = g.shape(x).clone();
    let (n, t) = (xs[0], xs[1]);
    let dir = usize::from(bidirectional) + 1;
    let cols = hidden * dir;
    let weights: Vec<Vec<f32>> = w.iter().map(|id| g.value(*id).to_vec()).collect();
    let mut grad_of: Vec<Vec<f32>> = weights.iter().map(|v| vec![0.0f32; v.len()]).collect();
    // Gradient of the layer below's output; starts at the tape gradient of this node and is replaced by
    // the input gradient of the layer above as we walk down the stack.
    let mut down = grad.to_vec();
    let mut dx_bottom = vec![0.0f32; fwd.layer_in[0].len()];
    for l in (0..layers).rev() {
        let xl = &fwd.layer_in[l];
        let in_size = xl.len() / (n * t);
        let mut dx = vec![0.0f32; xl.len()];
        // Both directions in parallel, each with its own input gradient, summed afterwards.
        let grads = kernels::par_map_range(dir, dir > 1, |d| {
            let base = (l * dir + d) * 4;
            let run = LstmRun {
                n,
                t,
                input: in_size,
                hidden,
                reverse: d == 1,
                cols,
                col: d * hidden,
            };
            let mut dxd = vec![0.0f32; xl.len()];
            let dg = lstm_dir_grad(
                xl,
                &weights[base],
                &weights[base + 1],
                &fwd.states[l * dir + d],
                &run,
                &down,
                &mut dxd,
            );
            (dg, dxd)
        });
        for (d, (dg, dxd)) in grads.into_iter().enumerate() {
            let base = (l * dir + d) * 4;
            accumulate(&mut dx, &dxd);
            accumulate(&mut grad_of[base], &dg.wih);
            accumulate(&mut grad_of[base + 1], &dg.whh);
            accumulate(&mut grad_of[base + 2], &dg.bih);
            accumulate(&mut grad_of[base + 3], &dg.bhh);
        }
        dx_bottom = dx;
        down = dx_bottom.clone();
    }
    g.add_grad(x, &dx_bottom);
    for (i, id) in w.iter().enumerate() {
        g.add_grad(*id, &grad_of[i]);
    }
}

/// Gradient of one LSTM direction: `dy` (the output gradient, read at `run.col`) becomes `dx` plus the
/// four weight gradients of this direction.
///
/// The recurrence runs backwards one step at a time for the whole batch; the gate gradients of every
/// step are kept (in time order), so the input and weight gradients are three GEMMs at the end.
fn lstm_dir_grad(
    x: &[f32],
    wih: &[f32],
    whh: &[f32],
    st: &DirState,
    r: &LstmRun,
    dy: &[f32],
    dx: &mut [f32],
) -> DirGrad {
    let LstmRun {
        n,
        t,
        input,
        hidden,
        reverse,
        cols,
        col,
    } = *r;
    let gates_len = 4 * hidden;
    let mut out = DirGrad::new(wih.len(), whh.len(), hidden);
    // Row `ni * t + s`: the gate gradient of sequence `ni` at time `s`, and the `h` that step read.
    let mut dgates = vec![0.0f32; n * t * gates_len];
    let mut h_prev_all = vec![0.0f32; n * t * hidden];
    let mut dh_next = vec![0.0f32; n * hidden];
    let mut dc_next = vec![0.0f32; n * hidden];
    let mut dg = vec![0.0f32; n * gates_len];
    let zero = vec![0.0f32; hidden];
    for step in (0..t).rev() {
        let s = if reverse { t - 1 - step } else { step };
        for ni in 0..n {
            let dy_at = (ni * t + s) * cols + col;
            let save = (ni * t + step) * hidden;
            let gates = &st.gates[save * 4..save * 4 + gates_len];
            let c_prev: &[f32] = if step == 0 {
                &zero
            } else {
                &st.cells[save - hidden..save]
            };
            if step > 0 {
                let at = (ni * t + s) * hidden;
                h_prev_all[at..at + hidden].copy_from_slice(&st.hs[save - hidden..save]);
            }
            let tanh_c = &st.tanh_c[save..save + hidden];
            let dgn = &mut dg[ni * gates_len..(ni + 1) * gates_len];
            let (dhn, dcn) = (
                &mut dh_next[ni * hidden..(ni + 1) * hidden],
                &mut dc_next[ni * hidden..(ni + 1) * hidden],
            );
            for j in 0..hidden {
                let dh = dy[dy_at + j] + dhn[j];
                let (ig, fg) = (gates[j], gates[hidden + j]);
                let (cg, og) = (gates[2 * hidden + j], gates[3 * hidden + j]);
                let tc = tanh_c[j];
                let dc = dh * og * (1.0 - tc * tc) + dcn[j];
                dgn[j] = dc * cg * ig * (1.0 - ig);
                dgn[hidden + j] = dc * c_prev[j] * fg * (1.0 - fg);
                dgn[2 * hidden + j] = dc * ig * (1.0 - cg * cg);
                dgn[3 * hidden + j] = dh * tc * og * (1.0 - og);
                dcn[j] = dc * fg;
            }
            let at = (ni * t + s) * gates_len;
            dgates[at..at + gates_len].copy_from_slice(dgn);
        }
        // dh_prev (n x H) = dg (n x 4H) · W_hh (4H x H)
        kernels::gemm(
            n,
            gates_len,
            hidden,
            &dg,
            (gates_len, 1),
            whh,
            (hidden, 1),
            0.0,
            &mut dh_next,
            (hidden, 1),
        );
    }
    let rows = n * t;
    // dx (NT x in) += dG (NT x 4H) · W_ih (4H x in)
    kernels::gemm(
        rows,
        gates_len,
        input,
        &dgates,
        (gates_len, 1),
        wih,
        (input, 1),
        1.0,
        dx,
        (input, 1),
    );
    // dW_ih (4H x in) = dGᵀ (4H x NT) · x (NT x in), dW_hh = dGᵀ · h_prev
    kernels::gemm(
        gates_len,
        rows,
        input,
        &dgates,
        (1, gates_len),
        x,
        (input, 1),
        0.0,
        &mut out.wih,
        (input, 1),
    );
    kernels::gemm(
        gates_len,
        rows,
        hidden,
        &dgates,
        (1, gates_len),
        &h_prev_all,
        (hidden, 1),
        0.0,
        &mut out.whh,
        (hidden, 1),
    );
    for row in dgates.chunks(gates_len) {
        for (b, g) in out.bih.iter_mut().zip(row) {
            *b += g;
        }
    }
    out.bhh.copy_from_slice(&out.bih);
    out
}

pub(crate) fn backward(g: &mut Graph, node: usize, op: &Op, grad: &[f32]) {
    match op.clone() {
        Op::Constant | Op::Param(_) => {}
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
            let (sa, sb) = (g.shape(a).clone(), g.shape(b).clone());
            let out = g.shape(NodeId(node)).clone();
            let os = strides(&out);
            let as_ = bcast_strides(&sa, &out);
            let bs = bcast_strides(&sb, &out);
            // The gradient of each operand at full (output) size, then summed over its broadcast axes.
            let (ga, gb) = {
                let (va, vb) = (g.value(a), g.value(b));
                let ga = match op {
                    Op::Mul(..) => {
                        Some(kernels::broadcast_map(&out, &os, &bs, grad, vb, |g, y| {
                            g * y
                        }))
                    }
                    _ => None,
                };
                let gb = match op {
                    Op::Add(..) => None,
                    Op::Sub(..) => Some(kernels::map(grad, |g| -g)),
                    _ => Some(kernels::broadcast_map(&out, &os, &as_, grad, va, |g, x| {
                        g * x
                    })),
                };
                (ga, gb)
            };
            for (id, shape, st, full) in [(a, &sa, &as_, ga), (b, &sb, &bs, gb)] {
                let full: &[f32] = full.as_deref().unwrap_or(grad);
                if *shape == out {
                    g.add_grad(id, full);
                } else {
                    let d = kernels::sum_to(&out, st, full, numel(shape));
                    g.add_grad(id, &d);
                }
            }
        }
        Op::Scale(a, k) => {
            let d = kernels::map(grad, |v| v * k);
            g.add_grad(a, &d);
        }
        Op::Relu(a) => {
            let d = kernels::zip_map(grad, g.value(a), |g, x| if x > 0.0 { g } else { 0.0 });
            g.add_grad(a, &d);
        }
        Op::Sigmoid(a) => {
            let d = kernels::zip_map(grad, g.value(NodeId(node)), |g, y| g * y * (1.0 - y));
            g.add_grad(a, &d);
        }
        Op::Tanh(a) => {
            let d = kernels::zip_map(grad, g.value(NodeId(node)), |g, y| g * (1.0 - y * y));
            g.add_grad(a, &d);
        }
        Op::Softmax(a, axis) => {
            let y = g.value(NodeId(node)).to_vec();
            let s = g.shape(a).clone();
            let c = s[axis];
            let outer = s[..axis].iter().product::<usize>().max(1);
            let inner = s[axis + 1..].iter().product::<usize>().max(1);
            let mut d = vec![0.0f32; y.len()];
            for o in 0..outer {
                for i in 0..inner {
                    let base = o * c * inner + i;
                    let dot: f32 = (0..c)
                        .map(|k| grad[base + k * inner] * y[base + k * inner])
                        .sum();
                    for k in 0..c {
                        let at = base + k * inner;
                        d[at] = y[at] * (grad[at] - dot);
                    }
                }
            }
            g.add_grad(a, &d);
        }
        Op::LogSoftmax(a, axis) => {
            let y = g.value(NodeId(node)).to_vec();
            let p: Vec<f32> = y.iter().map(|v| v.exp()).collect();
            let s = g.shape(a).clone();
            let c = s[axis];
            let outer = s[..axis].iter().product::<usize>().max(1);
            let inner = s[axis + 1..].iter().product::<usize>().max(1);
            let mut d = vec![0.0f32; y.len()];
            for o in 0..outer {
                for i in 0..inner {
                    let base = o * c * inner + i;
                    let dot: f32 = (0..c).map(|k| grad[base + k * inner]).sum();
                    for k in 0..c {
                        let at = base + k * inner;
                        d[at] = grad[at] - p[at] * dot;
                    }
                }
            }
            g.add_grad(a, &d);
        }
        Op::Matmul(a, b) => {
            let (sa, sb) = (g.shape(a).clone(), g.shape(b).clone());
            let (m, k) = (sa[sa.len() - 2], sa[sa.len() - 1]);
            let n = sb[sb.len() - 1];
            let (ba, bb) = (batch_count(&sa), batch_count(&sb));
            let batch = max(ba, bb);
            let x = g.value(a);
            let y = g.value(b);
            let mut gx = vec![0.0f32; x.len()];
            let mut gy = vec![0.0f32; y.len()];
            // One batch at a time, accumulating (`beta = 1`) into a broadcast operand's gradient.
            for bi in 0..batch {
                let oa = if ba == 1 { 0 } else { bi * m * k };
                let ob = if bb == 1 { 0 } else { bi * k * n };
                let go = &grad[bi * m * n..(bi + 1) * m * n];
                // dx (m x k) += go (m x n) · yᵀ (n x k)
                kernels::gemm(
                    m,
                    n,
                    k,
                    go,
                    (n, 1),
                    &y[ob..],
                    (1, n),
                    1.0,
                    &mut gx[oa..],
                    (k, 1),
                );
                // dy (k x n) += xᵀ (k x m) · go (m x n)
                kernels::gemm(
                    k,
                    m,
                    n,
                    &x[oa..],
                    (1, k),
                    go,
                    (n, 1),
                    1.0,
                    &mut gy[ob..],
                    (n, 1),
                );
            }
            g.add_grad(a, &gx);
            g.add_grad(b, &gy);
        }
        Op::Conv2d {
            x,
            w,
            b,
            stride,
            pad,
            dilation,
        } => {
            let (xs, ws) = (g.shape(x).clone(), g.shape(w).clone());
            let (n, c, h, iw) = (xs[0], xs[1], xs[2], xs[3]);
            let (o, _, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
            let oh = g.shape(NodeId(node))[2];
            let ow = g.shape(NodeId(node))[3];
            let geom = ConvShape {
                n,
                c,
                h,
                w: iw,
                o,
                kh,
                kw,
                stride,
                pad,
                dilation,
                oh,
                ow,
            };
            let (gx, gw, gbias) = kernels::conv2d_backward(g.value(x), g.value(w), grad, &geom);
            g.add_grad(x, &gx);
            g.add_grad(w, &gw);
            if let Some(b) = b {
                g.add_grad(b, &gbias);
            }
        }
        Op::MaxPool2d { x, k, stride, pad } | Op::AvgPool2d { x, k, stride, pad } => {
            let is_max = matches!(op, Op::MaxPool2d { .. });
            let xs = g.shape(x).clone();
            let (n, c, h, w) = (xs[0], xs[1], xs[2], xs[3]);
            let oh = g.shape(NodeId(node))[2];
            let ow = g.shape(NodeId(node))[3];
            let xv = g.value(x);
            let mut gx = vec![0.0f32; n * c * h * w];
            // One task per `(n, c)` plane: a window only ever reaches into its own plane.
            let par = xv.len() >= kernels::PAR_MIN;
            kernels::for_chunks_par(&mut gx, h * w, par, |plane, gxp| {
                let src = &xv[plane * h * w..(plane + 1) * h * w];
                let gop = &grad[plane * oh * ow..(plane + 1) * oh * ow];
                for y in 0..oh {
                    for xi in 0..ow {
                        let go = gop[y * ow + xi];
                        if go == 0.0 {
                            continue;
                        }
                        let mut best = f32::NEG_INFINITY;
                        let mut best_at = 0usize;
                        let share = go / (k * k) as f32;
                        for ky in 0..k {
                            let iy = y * stride + ky;
                            if iy < pad || iy >= h + pad {
                                continue;
                            }
                            for kx in 0..k {
                                let ix = xi * stride + kx;
                                if ix < pad || ix >= w + pad {
                                    continue;
                                }
                                let at = (iy - pad) * w + ix - pad;
                                if is_max {
                                    // The first maximum wins, like the index `nn.MaxPool2d` stores.
                                    if src[at] > best {
                                        best = src[at];
                                        best_at = at;
                                    }
                                } else {
                                    gxp[at] += share;
                                }
                            }
                        }
                        if is_max {
                            gxp[best_at] += go;
                        }
                    }
                }
            });
            g.add_grad(x, &gx);
        }
        Op::UpsampleNearest { x, h, w } => {
            let xs = g.shape(x).clone();
            let (n, c, ih, iw) = (xs[0], xs[1], xs[2], xs[3]);
            let mut gx = vec![0.0f32; n * c * ih * iw];
            kernels::for_chunks(&mut gx, ih * iw, |plane, gxp| {
                let gop = &grad[plane * h * w..(plane + 1) * h * w];
                for y in 0..h {
                    let sy = y * ih / h;
                    for xi in 0..w {
                        let sx = xi * iw / w;
                        gxp[sy * iw + sx] += gop[y * w + xi];
                    }
                }
            });
            g.add_grad(x, &gx);
        }
        Op::Reshape(a, shape) => {
            debug_assert_eq!(
                numel(&shape),
                grad.len(),
                "reshaped a node to the wrong number of values"
            );
            let d = grad.to_vec();
            g.add_grad(a, &d);
        }
        Op::Permute(a, perm) => {
            let out = g.shape(NodeId(node)).clone();
            let src = g.shape(a).clone();
            let os = strides(&out);
            let ss = strides(&src);
            let mut d = vec![0.0f32; numel(&src)];
            for (dst, gv) in grad.iter().enumerate() {
                let at = permute_index(dst, &os, &ss, &perm);
                d[at] += gv;
            }
            g.add_grad(a, &d);
        }
        Op::Concat(parts, axis) => {
            let out = g.shape(NodeId(node)).clone();
            let os = strides(&out);
            let mut base = 0usize;
            for p in parts.clone() {
                let ps = g.shape(p).clone();
                let pstr = strides(&ps);
                let mut d = vec![0.0f32; numel(&ps)];
                for (src, dv) in d.iter_mut().enumerate() {
                    let at = concat_index(src, &pstr, &os, axis, base);
                    *dv += grad[at];
                }
                g.add_grad(p, &d);
                base += ps[axis];
            }
        }
        Op::Slice {
            x,
            axis,
            start,
            end,
        } => {
            let xs = g.shape(x).clone();
            debug_assert!(end <= xs[axis], "slice {start}..{end} does not fit {xs:?}");
            let mut d = vec![0.0f32; numel(&xs)];
            let ns = g.shape(NodeId(node)).clone();
            let nstr = strides(&ns);
            let xstr = strides(&xs);
            for (src, &gv) in grad.iter().enumerate() {
                let at = slice_index(src, &nstr, &xstr, axis, start);
                d[at] += gv;
            }
            g.add_grad(x, &d);
        }
        Op::ReduceSum(a, axes) | Op::ReduceMean(a, axes) => {
            let s = g.shape(a).clone();
            let (ax, out, kept) = split_axes(&s, &axes);
            let ss = strides(&s);
            let ks = strides(&out);
            let scale = match op {
                Op::ReduceMean(..) => 1.0 / reduced_count(&s, &ax).max(1) as f32,
                _ => 1.0,
            };
            let mut d = vec![0.0f32; numel(&s)];
            if let Some((_, red, inner)) = contiguous_axes(&s, &ax) {
                kernels::for_chunks(&mut d, inner, |row, dv| {
                    let o = row / red.max(1);
                    for (v, gv) in dv.iter_mut().zip(&grad[o * inner..(o + 1) * inner]) {
                        *v = gv * scale;
                    }
                });
            } else {
                for (src, dv) in d.iter_mut().enumerate() {
                    let at = reduce_index(src, &ss, &kept, &ks);
                    *dv += grad[at] * scale;
                }
            }
            g.add_grad(a, &d);
        }
        Op::BatchNorm {
            x,
            gamma,
            beta,
            eps,
            mode,
            relu,
        } => {
            // A fused ReLU passes the gradient only where the output is positive, exactly what a
            // separate `Relu` node on the normalised values does.
            let masked: Option<Vec<f32>> = relu.then(|| {
                kernels::zip_map(
                    grad,
                    g.value(NodeId(node)),
                    |g, y| if y > 0.0 { g } else { 0.0 },
                )
            });
            let grad: &[f32] = masked.as_deref().unwrap_or(grad);
            let xs = g.shape(x).clone();
            let (n, c) = (xs[0], xs[1]);
            let inner = xs[2..].iter().product::<usize>().max(1);
            let xv = g.value(x);
            let gv = g.value(gamma);
            let (mean, var) = match &mode {
                BnMode::Train => batch_stats_from(xv, n, c, inner),
                BnMode::Eval { mean, var } => (mean.clone(), var.clone()),
            };
            let train = matches!(mode, BnMode::Train);
            let std: Vec<f32> = var.iter().map(|v| (v + eps).sqrt()).collect();
            let m = (n * inner) as f32;
            let par = xv.len() >= kernels::PAR_MIN;
            // Per channel: `(sum of go, sum of go * xhat)`, i.e. the beta and gamma gradients.
            let sums: Vec<(f32, f32)> = kernels::par_map_range(c, par, |ch| {
                let (mu, inv_std) = (mean[ch], 1.0 / std[ch]);
                let mut sum_go = 0.0f32;
                let mut sum_go_xhat = 0.0f32;
                for ni in 0..n {
                    let at = (ni * c + ch) * inner;
                    let (go, x) = (&grad[at..at + inner], &xv[at..at + inner]);
                    sum_go += kernels::lane_sum(go, |v| v);
                    sum_go_xhat += kernels::lane_dot(go, x, |g, x| g * ((x - mu) * inv_std));
                }
                (sum_go, sum_go_xhat)
            });
            let mut gx = vec![0.0f32; xv.len()];
            kernels::for_chunks_par(&mut gx, inner, par, |plane, out| {
                let ch = plane % c;
                let inv = gv[ch] / std[ch];
                let (mu, inv_std) = (mean[ch], 1.0 / std[ch]);
                let (sum_go, sum_go_xhat) = sums[ch];
                let (mean_go, mean_go_xhat) = (sum_go / m, sum_go_xhat / m);
                let at = plane * inner;
                for ((o, go), x) in out
                    .iter_mut()
                    .zip(&grad[at..at + inner])
                    .zip(&xv[at..at + inner])
                {
                    *o = if train {
                        let xhat = (x - mu) * inv_std;
                        inv * (go - mean_go - xhat * mean_go_xhat)
                    } else {
                        inv * go
                    };
                }
            });
            let (gbet, ggam): (Vec<f32>, Vec<f32>) = sums.into_iter().unzip();
            g.add_grad(x, &gx);
            g.add_grad(gamma, &ggam);
            g.add_grad(beta, &gbet);
        }
        Op::L2Normalize(a, axis) => {
            let s = g.shape(a).clone();
            let c = s[axis];
            let outer = s[..axis].iter().product::<usize>().max(1);
            let inner = s[axis + 1..].iter().product::<usize>().max(1);
            let x = g.value(a).to_vec();
            let mut d = vec![0.0f32; x.len()];
            for o in 0..outer {
                for i in 0..inner {
                    let base = o * c * inner + i;
                    let norm = norm_of(&x, base, inner, c);
                    let dot: f32 = (0..c)
                        .map(|k| grad[base + k * inner] * x[base + k * inner] / norm)
                        .sum();
                    for k in 0..c {
                        let at = base + k * inner;
                        d[at] = (grad[at] - dot * x[at] / norm) / norm;
                    }
                }
            }
            g.add_grad(a, &d);
        }
        Op::Dropout { x, p, seed } => {
            let mut rng = SplitMix::new(seed);
            let keep = 1.0 / (1.0 - p);
            let d: Vec<f32> = grad
                .iter()
                .map(|v| if rng.next() < p { 0.0 } else { v * keep })
                .collect();
            g.add_grad(x, &d);
        }
        Op::BceWithLogits { logits, target } => {
            let l = g.value(logits).to_vec();
            let t = g.value(target).to_vec();
            let n = l.len() as f32;
            let go = grad[0];
            let d: Vec<f32> = l
                .iter()
                .zip(&t)
                .map(|(z, y)| go * (1.0 / (1.0 + (-z).exp()) - y) / n)
                .collect();
            g.add_grad(logits, &d);
        }
        Op::WeightedLoss { x, kind, weight } => {
            let dv = g.value(x).to_vec();
            let n = dv.len() as f32;
            let go = grad[0];
            let d: Vec<f32> = dv
                .iter()
                .map(|v| {
                    go * weight / n
                        * match kind {
                            LossKind::L1 => v.signum(),
                            LossKind::SmoothL1 => v.clamp(-1.0, 1.0),
                            LossKind::Mse => 2.0 * v,
                        }
                })
                .collect();
            g.add_grad(x, &d);
        }
        Op::CrossEntropy {
            logits,
            target,
            weight,
            smoothing,
            ignore,
        } => {
            let ls = g.shape(logits).clone();
            let c = *ls.last().unwrap();
            let n = numel(&ls) / c;
            let lv = g.value(logits).to_vec();
            let tv = g.value(target).to_vec();
            let wv = weight
                .map(|id| g.value(id).to_vec())
                .unwrap_or_else(|| vec![1.0; c]);
            let wsum = weight_sum(&wv);
            let mut used = 0.0f32;
            for &target in tv.iter() {
                if target as i32 != ignore {
                    used += wv.get(target as usize).copied().unwrap_or(1.0);
                }
            }
            let go = grad[0];
            let denom = used.max(1.0);
            let mut d = vec![0.0f32; lv.len()];
            for i in 0..n {
                if tv[i] as i32 == ignore {
                    continue;
                }
                let t = tv[i] as usize;
                let p = probs(&lv, i * c, c);
                let wt = wv.get(t).copied().unwrap_or(1.0);
                for k in 0..c {
                    let wk = wv.get(k).copied().unwrap_or(1.0);
                    let onehot = if k == t { 1.0 } else { 0.0 };
                    let gk = (1.0 - smoothing) * wt * (p[k] - onehot)
                        + smoothing * (p[k] - wk / wsum) * wsum / c as f32;
                    d[i * c + k] = go * gk / denom;
                }
            }
            g.add_grad(logits, &d);
        }
        Op::Lstm {
            x,
            w,
            hidden,
            layers,
            bidirectional,
        } => {
            lstm_backward(g, x, &w, hidden, layers, bidirectional, grad);
        }
    }
}
