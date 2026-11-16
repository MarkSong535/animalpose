//! The fast paths behind [`crate::tensor::ops`]: a multithreaded SGEMM, convolution as im2col + GEMM
//! (and its gradient as GEMM + col2im), and data-parallel helpers for the element-wise kernels.
//!
//! The matrix products go through [`matrixmultiply::sgemm`] (pure Rust, SIMD micro-kernels); every
//! loop that is worth it is spread over the cores with [`rayon`]. Each element is still produced by
//! exactly one task, so the results do not depend on the number of threads. The summation order of a
//! dot product is the GEMM's, not the naive loop's, which moves results by a few ulps; PyTorch's own
//! kernels sum in yet another order, and the torch fixtures are compared at `2e-4`.

use rayon::prelude::*;

/// Element count below which a kernel doing real work per element (pooling windows, im2col, col2im)
/// stays on the calling thread.
pub(crate) const PAR_MIN: usize = 1 << 15;

/// Element count below which a memory-bound stream (one or two loads and a store per element) stays
/// on the calling thread: one core already streams close to the memory bandwidth, so splitting a
/// smaller array only adds synchronisation (and, on a busy machine, waiting for descheduled workers).
pub(crate) const STREAM_MIN: usize = 1 << 22;

/// Chunk length of a parallel stream.
const STREAM_CHUNK: usize = 1 << 18;

/// Target multiply-adds per parallel GEMM block; a smaller product runs as one call.
const GEMM_TASK: usize = 1 << 22;

/// Column budget of one im2col buffer (`rows * cols` floats), so a large batch is convolved in groups
/// of images instead of materialising one huge buffer.
const COL_BUDGET: usize = 1 << 23;

thread_local! {
    /// Scratch buffers of the convolution kernels (im2col, its gradient, the transposed output
    /// gradient and the grouped GEMM output), kept per thread so consecutive layers reuse memory that
    /// is already mapped instead of faulting in a fresh buffer of up to [`COL_BUDGET`] floats per call.
    static SCRATCH: std::cell::RefCell<[Vec<f32>; 4]> = std::cell::RefCell::new(Default::default());
}

/// Runs `f` with four scratch buffers of at least the given lengths (contents unspecified).
fn with_scratch<R>(lens: [usize; 4], f: impl FnOnce(&mut [Vec<f32>; 4]) -> R) -> R {
    // Taken out of the cell while `f` runs, so a nested call (none today) would allocate its own.
    let mut bufs = SCRATCH.with(|s| std::mem::take(&mut *s.borrow_mut()));
    for (b, &len) in bufs.iter_mut().zip(&lens) {
        if b.len() < len {
            *b = vec![0.0f32; len];
        }
    }
    let out = f(&mut bufs);
    SCRATCH.with(|s| *s.borrow_mut() = bufs);
    out
}

/// A raw pointer that may cross threads; every use writes a region no other task touches.
#[derive(Clone, Copy)]
struct Ptr(*mut f32);

// SAFETY: the pointer is only dereferenced through `sgemm` on disjoint output blocks (see `gemm_split`).
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

impl Ptr {
    /// The pointer, taken through a method so closures capture the whole (`Send`) wrapper.
    fn get(self) -> *mut f32 {
        self.0
    }
}

#[derive(Clone, Copy)]
struct CPtr(*const f32);

// SAFETY: read-only operand pointers, shared between tasks.
unsafe impl Send for CPtr {}
unsafe impl Sync for CPtr {}

impl CPtr {
    /// The pointer, taken through a method so closures capture the whole (`Send`) wrapper.
    fn get(self) -> *const f32 {
        self.0
    }
}

/// Row and column stride of a matrix view.
pub(crate) type Strides = (usize, usize);

/// The largest offset a `rows x cols` view with strides `s` reads.
fn extent(rows: usize, cols: usize, s: Strides) -> usize {
    if rows == 0 || cols == 0 {
        0
    } else {
        (rows - 1) * s.0 + (cols - 1) * s.1 + 1
    }
}

/// `C = A B + beta C` for an `m x k` view `A`, a `k x n` view `B` and an `m x n` view `C`, all
/// strided, split over the cores when the product is large.
///
/// `beta = 0` overwrites `C` without reading it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    sa: Strides,
    b: &[f32],
    sb: Strides,
    beta: f32,
    c: &mut [f32],
    sc: Strides,
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(extent(m, k, sa) <= a.len(), "gemm: A view out of bounds");
    assert!(extent(k, n, sb) <= b.len(), "gemm: B view out of bounds");
    assert!(extent(m, n, sc) <= c.len(), "gemm: C view out of bounds");
    assert!(
        sc.0 != sc.1 || m == 1 || n == 1,
        "gemm: C strides alias elements"
    );
    // SAFETY: the views were bounds-checked above, and `gemm_split` hands every task its own block of C.
    unsafe {
        gemm_split(
            m,
            k,
            n,
            CPtr(a.as_ptr()),
            sa,
            CPtr(b.as_ptr()),
            sb,
            beta,
            Ptr(c.as_mut_ptr()),
            sc,
        );
    }
}

/// Splits `C` into a grid of blocks, about [`GEMM_TASK`] multiply-adds each (at most a few per core),
/// shaped so the operand panels every block packs stay small, and runs the blocks in parallel.
///
/// # Safety
///
/// The views must be in bounds and `C` must not alias `A` or `B`.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_split(
    m: usize,
    k: usize,
    n: usize,
    a: CPtr,
    sa: Strides,
    b: CPtr,
    sb: Strides,
    beta: f32,
    c: Ptr,
    sc: Strides,
) {
    let work = m.saturating_mul(n).saturating_mul(k.max(1));
    let tasks = (work / GEMM_TASK).min(4 * rayon::current_num_threads());
    // Rows split in multiples of the 8-row micro-kernel, columns in multiples of 8 as well.
    let (mb, nb) = (m.div_ceil(8), n.div_ceil(8));
    if tasks < 2 || mb * nb < 2 {
        sgemm_block(m, k, n, a, sa, b, sb, beta, c, sc);
        return;
    }
    // Every block packs its rows of A and its columns of B, `k * (sn * m + sm * n)` in all for an
    // `sm x sn` grid, which is smallest for `sm / sn = m / n`.
    let sm = ((tasks as f64 * m as f64 / n as f64).sqrt().round() as usize).clamp(1, mb);
    let sn = tasks.div_ceil(sm).clamp(1, nb);
    let (rm, rn) = (mb.div_ceil(sm) * 8, nb.div_ceil(sn) * 8);
    let (sm, sn) = (m.div_ceil(rm), n.div_ceil(rn));
    (0..sm * sn).into_par_iter().for_each(|t| {
        let (i, j) = (t / sn, t % sn);
        let (r0, c0) = (i * rm, j * rn);
        let (rows, cols) = (rm.min(m - r0), rn.min(n - c0));
        sgemm_block(
            rows,
            k,
            cols,
            CPtr(a.get().add(r0 * sa.0)),
            sa,
            CPtr(b.get().add(c0 * sb.1)),
            sb,
            beta,
            Ptr(c.get().add(r0 * sc.0 + c0 * sc.1)),
            sc,
        );
    });
}

/// One single-threaded [`matrixmultiply::sgemm`] call.
///
/// # Safety
///
/// As for [`gemm_split`].
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm_block(
    m: usize,
    k: usize,
    n: usize,
    a: CPtr,
    sa: Strides,
    b: CPtr,
    sb: Strides,
    beta: f32,
    c: Ptr,
    sc: Strides,
) {
    matrixmultiply::sgemm(
        m,
        k,
        n,
        1.0,
        a.0,
        sa.0 as isize,
        sa.1 as isize,
        b.0,
        sb.0 as isize,
        sb.1 as isize,
        beta,
        c.0,
        sc.0 as isize,
        sc.1 as isize,
    );
}

/// Geometry of a square-parameter 2D convolution over `(N, C, H, W)` with an `(O, C, kH, kW)` kernel.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConvShape {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub o: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub pad: usize,
    pub dilation: usize,
    pub oh: usize,
    pub ow: usize,
}

impl ConvShape {
    /// Output positions per image.
    fn p(&self) -> usize {
        self.oh * self.ow
    }

    /// Rows of the im2col matrix, `C * kH * kW`.
    fn rows(&self) -> usize {
        self.c * self.kh * self.kw
    }

    /// A `1x1`, stride-1, unpadded convolution reads the input as the GEMM operand directly.
    fn is_pointwise(&self) -> bool {
        self.kh == 1 && self.kw == 1 && self.stride == 1 && self.pad == 0
    }

    /// How many images one im2col buffer holds.
    /// Images per im2col group: as many as `budget` floats hold, balanced so the last group is not
    /// much smaller than the others.
    fn group(&self, budget: usize) -> usize {
        let n = self.n.max(1);
        let fit = (budget / (self.rows() * self.p()).max(1)).clamp(1, n);
        n.div_ceil(n.div_ceil(fit))
    }

    /// The output columns `[lo, hi)` whose input column `o * stride + k * dilation - pad` lies inside
    /// `0..size`.
    fn valid(&self, k: usize, size: usize, out: usize) -> (usize, usize) {
        let off = k * self.dilation;
        let s = self.stride;
        // ix = ox * s + off - pad >= 0  <=>  ox >= ceil((pad - off) / s)
        let lo = if off >= self.pad {
            0
        } else {
            (self.pad - off).div_ceil(s)
        };
        // ix < size  <=>  ox * s < size + pad - off
        let hi = if size + self.pad <= off {
            0
        } else {
            (size + self.pad - off).div_ceil(s)
        };
        (lo.min(out), hi.min(out).max(lo.min(out)))
    }
}

/// Writes the im2col matrix of images `n0..n0 + count`: row `(c, ky, kx)`, column `j * P + p` holds the
/// input pixel under kernel tap `(ky, kx)` at output position `p` of image `n0 + j` (zero in the padding).
fn im2col(x: &[f32], s: &ConvShape, n0: usize, count: usize, col: &mut [f32]) {
    let p = s.p();
    let cols = count * p;
    let plane = s.h * s.w;
    let fill = |(row, dst): (usize, &mut [f32])| {
        let ci = row / (s.kh * s.kw);
        let ky = (row / s.kw) % s.kh;
        let kx = row % s.kw;
        let (ylo, yhi) = s.valid(ky, s.h, s.oh);
        let (xlo, xhi) = s.valid(kx, s.w, s.ow);
        for j in 0..count {
            let src = &x[((n0 + j) * s.c + ci) * plane..((n0 + j) * s.c + ci + 1) * plane];
            let out = &mut dst[j * p..(j + 1) * p];
            out[..ylo * s.ow].fill(0.0);
            out[yhi * s.ow..].fill(0.0);
            for oy in ylo..yhi {
                let iy = oy * s.stride + ky * s.dilation - s.pad;
                let row_out = &mut out[oy * s.ow..(oy + 1) * s.ow];
                row_out[..xlo].fill(0.0);
                row_out[xhi..].fill(0.0);
                let src_row = &src[iy * s.w..(iy + 1) * s.w];
                if xhi > xlo {
                    let ix0 = xlo * s.stride + kx * s.dilation - s.pad;
                    if s.stride == 1 {
                        row_out[xlo..xhi].copy_from_slice(&src_row[ix0..ix0 + (xhi - xlo)]);
                    } else {
                        for (i, v) in row_out[xlo..xhi].iter_mut().enumerate() {
                            *v = src_row[ix0 + i * s.stride];
                        }
                    }
                }
            }
        }
    };
    let col = &mut col[..s.rows() * cols];
    if col.len() >= PAR_MIN {
        col.par_chunks_mut(cols).enumerate().for_each(fill);
    } else {
        col.chunks_mut(cols).enumerate().for_each(fill);
    }
}

/// Adds the im2col-layout gradient `col` of images `n0..n0 + count` back onto `gx` (the adjoint of
/// [`im2col`]); every `(image, channel)` plane is one task, so no two tasks write the same element.
fn col2im(col: &[f32], s: &ConvShape, n0: usize, count: usize, gx: &mut [f32]) {
    let p = s.p();
    let cols = count * p;
    let plane = s.h * s.w;
    let gx = &mut gx[n0 * s.c * plane..(n0 + count) * s.c * plane];
    let add = |(idx, dst): (usize, &mut [f32])| {
        let (j, ci) = (idx / s.c, idx % s.c);
        for ky in 0..s.kh {
            let (ylo, yhi) = s.valid(ky, s.h, s.oh);
            for kx in 0..s.kw {
                let (xlo, xhi) = s.valid(kx, s.w, s.ow);
                if xhi <= xlo {
                    continue;
                }
                let row = (ci * s.kh + ky) * s.kw + kx;
                let src = &col[row * cols + j * p..row * cols + (j + 1) * p];
                let ix0 = xlo * s.stride + kx * s.dilation - s.pad;
                for oy in ylo..yhi {
                    let iy = oy * s.stride + ky * s.dilation - s.pad;
                    let g_row = &src[oy * s.ow + xlo..oy * s.ow + xhi];
                    let d_row = &mut dst[iy * s.w..(iy + 1) * s.w];
                    if s.stride == 1 {
                        for (d, g) in d_row[ix0..ix0 + g_row.len()].iter_mut().zip(g_row) {
                            *d += g;
                        }
                    } else {
                        for (i, g) in g_row.iter().enumerate() {
                            d_row[ix0 + i * s.stride] += g;
                        }
                    }
                }
            }
        }
    };
    if gx.len() >= PAR_MIN {
        gx.par_chunks_mut(plane).enumerate().for_each(add);
    } else {
        gx.chunks_mut(plane).enumerate().for_each(add);
    }
}

/// `conv2d` forward: `out[n] = W · im2col(x[n]) + bias`, in groups of images whose im2col fits
/// [`COL_BUDGET`].
pub(crate) fn conv2d_forward(x: &[f32], w: &[f32], bias: &[f32], s: &ConvShape) -> Vec<f32> {
    conv2d_forward_in(x, w, bias, s, COL_BUDGET)
}

/// [`conv2d_forward`] with an explicit im2col budget (the tests shrink it to exercise the grouping).
pub(crate) fn conv2d_forward_in(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    s: &ConvShape,
    budget: usize,
) -> Vec<f32> {
    let (p, rows) = (s.p(), s.rows());
    let mut out = vec![0.0f32; s.n * s.o * p];
    if out.is_empty() {
        return out;
    }
    let beta = if bias.is_empty() {
        0.0
    } else {
        let fill = |(i, plane): (usize, &mut [f32])| plane.fill(bias[i % s.o]);
        if out.len() >= PAR_MIN {
            out.par_chunks_mut(p).enumerate().for_each(fill);
        } else {
            out.chunks_mut(p).enumerate().for_each(fill);
        }
        1.0
    };
    if s.is_pointwise() {
        // The input already is the GEMM operand: one (internally split) product per image.
        out.par_chunks_mut(s.o * p)
            .enumerate()
            .for_each(|(ni, on)| {
                let xn = &x[ni * s.c * p..(ni + 1) * s.c * p];
                gemm(s.o, rows, p, w, (rows, 1), xn, (p, 1), beta, on, (p, 1));
            });
        return out;
    }
    let group = s.group(budget);
    let tmp_len = if group > 1 { s.o * group * p } else { 0 };
    with_scratch([rows * group * p, tmp_len, 0, 0], |bufs| {
        let [col, tmp, _, _] = bufs;
        conv_groups(x, w, beta, s, group, col, tmp, &mut out);
    });
    out
}

/// The im2col + GEMM loop of [`conv2d_forward`] over groups of `group` images.
#[allow(clippy::too_many_arguments)]
fn conv_groups(
    x: &[f32],
    w: &[f32],
    beta: f32,
    s: &ConvShape,
    group: usize,
    col: &mut [f32],
    tmp: &mut [f32],
    out: &mut [f32],
) {
    let (p, rows) = (s.p(), s.rows());
    // A group of several images is one `O x (count * P)` product, which packs the weights once; its
    // columns are then scattered to the `(N, O, P)` layout.
    let mut n0 = 0;
    while n0 < s.n {
        let count = group.min(s.n - n0);
        let cols = count * p;
        im2col(x, s, n0, count, col);
        let outs = &mut out[n0 * s.o * p..(n0 + count) * s.o * p];
        if count == 1 {
            gemm(s.o, rows, p, w, (rows, 1), col, (p, 1), beta, outs, (p, 1));
        } else {
            let tmp = &mut tmp[..s.o * cols];
            gemm(
                s.o,
                rows,
                cols,
                w,
                (rows, 1),
                col,
                (cols, 1),
                0.0,
                tmp,
                (cols, 1),
            );
            let tmp = &*tmp;
            for_chunks(outs, p, |plane, dst| {
                let (j, oi) = (plane / s.o, plane % s.o);
                let src = &tmp[oi * cols + j * p..oi * cols + (j + 1) * p];
                if beta == 0.0 {
                    dst.copy_from_slice(src);
                } else {
                    for (d, v) in dst.iter_mut().zip(src) {
                        *d += v;
                    }
                }
            });
        }
        n0 += count;
    }
}

/// Gradients of [`conv2d_forward`]: `(dx, dw, dbias)` from the output gradient `go`.
///
/// `dw = Σ_n go[n] · im2col(x[n])ᵀ` and `dcol = Wᵀ · go[n]`, which [`col2im`] folds back onto `dx`.
pub(crate) fn conv2d_backward(
    x: &[f32],
    w: &[f32],
    go: &[f32],
    s: &ConvShape,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    conv2d_backward_in(x, w, go, s, COL_BUDGET)
}

/// [`conv2d_backward`] with an explicit im2col budget.
pub(crate) fn conv2d_backward_in(
    x: &[f32],
    w: &[f32],
    go: &[f32],
    s: &ConvShape,
    budget: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (p, rows, o) = (s.p(), s.rows(), s.o);
    let mut gx = vec![0.0f32; x.len()];
    let mut gw = vec![0.0f32; w.len()];
    let gb: Vec<f32> = if go.len() >= PAR_MIN {
        (0..o)
            .into_par_iter()
            .map(|oi| bias_grad(go, oi, s))
            .collect()
    } else {
        (0..o).map(|oi| bias_grad(go, oi, s)).collect()
    };
    if x.is_empty() || go.is_empty() {
        return (gx, gw, gb);
    }
    if s.is_pointwise() {
        for ni in 0..s.n {
            let xn = &x[ni * s.c * p..(ni + 1) * s.c * p];
            let gn = &go[ni * o * p..(ni + 1) * o * p];
            // dw (O x C) += go[n] (O x P) · x[n]ᵀ (P x C)
            gemm(o, p, rows, gn, (p, 1), xn, (1, p), 1.0, &mut gw, (rows, 1));
        }
        // dx[n] (C x P) = Wᵀ (C x O) · go[n] (O x P)
        gx.par_chunks_mut(s.c * p)
            .enumerate()
            .for_each(|(ni, gxn)| {
                let gn = &go[ni * o * p..(ni + 1) * o * p];
                gemm(rows, o, p, w, (1, rows), gn, (p, 1), 0.0, gxn, (p, 1));
            });
        return (gx, gw, gb);
    }
    let group = s.group(budget);
    let len = rows * group * p;
    with_scratch([len, len, o * group * p, 0], |bufs| {
        let [col, gcol, got, _] = bufs;
        conv_backward_groups(x, w, go, s, group, col, gcol, got, &mut gx, &mut gw);
    });
    (gx, gw, gb)
}

/// The im2col + GEMM + col2im loop of [`conv2d_backward`] over groups of `group` images.
#[allow(clippy::too_many_arguments)]
fn conv_backward_groups(
    x: &[f32],
    w: &[f32],
    go: &[f32],
    s: &ConvShape,
    group: usize,
    col: &mut [f32],
    gcol: &mut [f32],
    got: &mut [f32],
    gx: &mut [f32],
    gw: &mut [f32],
) {
    let (p, rows, o) = (s.p(), s.rows(), s.o);
    let mut n0 = 0;
    while n0 < s.n {
        let count = group.min(s.n - n0);
        let cols = count * p;
        im2col(x, s, n0, count, col);
        // The group's output gradient as one `O x (count * P)` matrix, column `j * P + q` for
        // position `q` of image `n0 + j`, so each gradient is a single GEMM over the whole group.
        let got = &mut got[..o * cols];
        for_chunks(got, cols, |oi, row| {
            for (j, dst) in row.chunks_mut(p).enumerate() {
                let at = ((n0 + j) * o + oi) * p;
                dst.copy_from_slice(&go[at..at + p]);
            }
        });
        // dw (O x rows) += got (O x cols) · colᵀ (cols x rows)
        gemm(
            o,
            cols,
            rows,
            got,
            (cols, 1),
            col,
            (1, cols),
            1.0,
            gw,
            (rows, 1),
        );
        // dcol (rows x cols) = Wᵀ (rows x O) · got (O x cols)
        gemm(
            rows,
            o,
            cols,
            w,
            (1, rows),
            got,
            (cols, 1),
            0.0,
            gcol,
            (cols, 1),
        );
        col2im(gcol, s, n0, count, gx);
        n0 += count;
    }
}

/// Sum of the output gradient over every image and position of channel `oi`.
fn bias_grad(go: &[f32], oi: usize, s: &ConvShape) -> f32 {
    let p = s.p();
    (0..s.n)
        .map(|ni| lane_sum(&go[(ni * s.o + oi) * p..(ni * s.o + oi + 1) * p], |v| v))
        .sum()
}

/// `f` applied to every element of `x`, as a new vector.
pub(crate) fn map<F>(x: &[f32], f: F) -> Vec<f32>
where
    F: Fn(f32) -> f32 + Sync + Send,
{
    if x.len() >= STREAM_MIN {
        let mut out = vec![0.0f32; x.len()];
        out.par_chunks_mut(STREAM_CHUNK)
            .zip(x.par_chunks(STREAM_CHUNK))
            .for_each(|(o, i)| {
                for (o, v) in o.iter_mut().zip(i) {
                    *o = f(*v);
                }
            });
        out
    } else {
        x.iter().map(|v| f(*v)).collect()
    }
}

/// `f(a[i], b[i])` over two equally long slices, as a new vector.
pub(crate) fn zip_map<F>(a: &[f32], b: &[f32], f: F) -> Vec<f32>
where
    F: Fn(f32, f32) -> f32 + Sync + Send,
{
    debug_assert_eq!(a.len(), b.len());
    if a.len() >= STREAM_MIN {
        let mut out = vec![0.0f32; a.len()];
        out.par_chunks_mut(STREAM_CHUNK)
            .zip(a.par_chunks(STREAM_CHUNK).zip(b.par_chunks(STREAM_CHUNK)))
            .for_each(|(o, (x, y))| {
                for ((o, x), y) in o.iter_mut().zip(x).zip(y) {
                    *o = f(*x, *y);
                }
            });
        out
    } else {
        a.iter().zip(b).map(|(x, y)| f(*x, *y)).collect()
    }
}

/// `to[i] += from[i]`.
pub(crate) fn add_assign(to: &mut [f32], from: &[f32]) {
    if to.len() >= STREAM_MIN {
        to.par_chunks_mut(STREAM_CHUNK)
            .zip(from.par_chunks(STREAM_CHUNK))
            .for_each(|(a, b)| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
            });
    } else {
        for (x, y) in to.iter_mut().zip(from) {
            *x += y;
        }
    }
}

/// Runs `f(index, chunk)` over consecutive `size`-long chunks of `data`, in parallel when `data` is
/// at least [`STREAM_MIN`] long (for memory-bound per-chunk work).
pub(crate) fn for_chunks<F>(data: &mut [f32], size: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync + Send,
{
    for_chunks_par(data, size, data.len() >= STREAM_MIN, f);
}

/// Independent accumulators of [`lane_sum`] / [`lane_dot`]: a single running `f32` sum is one long
/// dependency chain the compiler may not reorder, eight of them vectorise.
const LANES: usize = 8;

/// `Σ f(x)` over a slice, accumulated in [`LANES`] interleaved partial sums.
pub(crate) fn lane_sum(x: &[f32], f: impl Fn(f32) -> f32) -> f32 {
    let mut acc = [0.0f32; LANES];
    let chunks = x.chunks_exact(LANES);
    let tail: f32 = chunks.remainder().iter().map(|v| f(*v)).sum();
    for c in chunks {
        for (a, v) in acc.iter_mut().zip(c) {
            *a += f(*v);
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// `Σ f(a, b)` over two equally long slices, accumulated like [`lane_sum`].
pub(crate) fn lane_dot(a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; LANES];
    let (ca, cb) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let tail: f32 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(x, y)| f(*x, *y))
        .sum();
    for (x, y) in ca.zip(cb) {
        for ((s, x), y) in acc.iter_mut().zip(x).zip(y) {
            *s += f(*x, *y);
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// [`for_chunks`] that only goes parallel when `par` is set (e.g. more than one matrix in a batch).
pub(crate) fn for_chunks_par<F>(data: &mut [f32], size: usize, par: bool, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync + Send,
{
    let size = size.max(1);
    if par {
        data.par_chunks_mut(size)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
    } else {
        data.chunks_mut(size).enumerate().for_each(|(i, c)| f(i, c));
    }
}

/// `(0..n).map(f).collect()`, in parallel when `par` is set.
pub(crate) fn par_map_range<T, F>(n: usize, par: bool, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    if par {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// A broadcast walk with adjacent axes merged wherever both operands step through them as one:
/// `dims` is the merged output shape and `sa` / `sb` the operands' strides along it (0 = broadcast).
struct Walk {
    dims: Vec<usize>,
    sa: Vec<usize>,
    sb: Vec<usize>,
}

impl Walk {
    fn new(out: &[usize], sa: &[usize], sb: &[usize]) -> Self {
        let mut w = Walk {
            dims: Vec::new(),
            sa: Vec::new(),
            sb: Vec::new(),
        };
        for i in 0..out.len() {
            if out[i] == 1 {
                continue;
            }
            if let Some(&last) = w.dims.last() {
                let j = w.dims.len() - 1;
                // Axis `j` (outer) and axis `i` (inner) merge when stepping the outer one is the same as
                // stepping the inner one `out[i]` times, for both operands.
                if w.sa[j] == sa[i] * out[i] && w.sb[j] == sb[i] * out[i] {
                    w.dims[j] = last * out[i];
                    w.sa[j] = sa[i];
                    w.sb[j] = sb[i];
                    continue;
                }
            }
            w.dims.push(out[i]);
            w.sa.push(sa[i]);
            w.sb.push(sb[i]);
        }
        if w.dims.is_empty() {
            w.dims.push(1);
            w.sa.push(0);
            w.sb.push(0);
        }
        w
    }

    /// Length of the innermost run.
    fn run(&self) -> usize {
        *self.dims.last().unwrap()
    }

    /// Start offsets in both operands of run number `row`.
    fn offsets(&self, row: usize) -> (usize, usize) {
        let mut rest = row;
        let (mut oa, mut ob) = (0usize, 0usize);
        let r = self.dims.len() - 1;
        for i in (0..r).rev() {
            let c = rest % self.dims[i];
            rest /= self.dims[i];
            oa += c * self.sa[i];
            ob += c * self.sb[i];
        }
        (oa, ob)
    }
}

/// `f(a, b)` over the broadcast output shape `out`, the operands read with the (zero for a
/// broadcast axis) strides `sa` / `sb`.
pub(crate) fn broadcast_map<F>(
    out: &[usize],
    sa: &[usize],
    sb: &[usize],
    a: &[f32],
    b: &[f32],
    f: F,
) -> Vec<f32>
where
    F: Fn(f32, f32) -> f32 + Sync + Send,
{
    let walk = Walk::new(out, sa, sb);
    let n: usize = out.iter().product();
    let mut data = vec![0.0f32; n];
    if n == 0 {
        return data;
    }
    let run = walk.run();
    let (ia, ib) = (walk.sa[walk.sa.len() - 1], walk.sb[walk.sb.len() - 1]);
    for_chunks(&mut data, run, |row, dst| {
        let (oa, ob) = walk.offsets(row);
        match (ia, ib) {
            (1, 1) => {
                for ((d, x), y) in dst.iter_mut().zip(&a[oa..oa + run]).zip(&b[ob..ob + run]) {
                    *d = f(*x, *y);
                }
            }
            (1, 0) => {
                let y = b[ob];
                for (d, x) in dst.iter_mut().zip(&a[oa..oa + run]) {
                    *d = f(*x, y);
                }
            }
            (0, 1) => {
                let x = a[oa];
                for (d, y) in dst.iter_mut().zip(&b[ob..ob + run]) {
                    *d = f(x, *y);
                }
            }
            _ => {
                for (t, d) in dst.iter_mut().enumerate() {
                    *d = f(a[oa + t * ia], b[ob + t * ib]);
                }
            }
        }
    });
    data
}

/// Sums an output-shaped `full` gradient back onto an operand of `len` elements that was read with
/// the broadcast strides `st` (the adjoint of reading it in [`broadcast_map`]).
pub(crate) fn sum_to(out: &[usize], st: &[usize], full: &[f32], len: usize) -> Vec<f32> {
    let contiguous = crate::tensor::ops::strides(out);
    let walk = Walk::new(out, st, &contiguous);
    let mut d = vec![0.0f32; len];
    let n: usize = out.iter().product();
    if n == 0 {
        return d;
    }
    let run = walk.run();
    let (ia, ig) = (walk.sa[walk.sa.len() - 1], walk.sb[walk.sb.len() - 1]);
    let rows = n / run;
    for row in 0..rows {
        let (oa, og) = walk.offsets(row);
        let src = &full[og..];
        if ia == 0 {
            let mut acc = 0.0f32;
            for t in 0..run {
                acc += src[t * ig];
            }
            d[oa] += acc;
        } else {
            for t in 0..run {
                d[oa + t * ia] += src[t * ig];
            }
        }
    }
    d
}

#[cfg(test)]
mod tests {
    //! The fast kernels against straightforward reference loops (the convolution reference is the
    //! naive kernel this module replaced), on shapes that hit every special case: stride, padding
    //! wider than the kernel reach, dilation, `1x1` and `(1, k)` kernels, a bias or none, and im2col
    //! budgets small enough to split a batch into uneven groups.

    use super::*;
    use crate::tensor::ops::strides;

    /// Deterministic values in `[-1, 1)`.
    fn random(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn assert_close(what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                "{what}[{i}]: got {g}, want {w}"
            );
        }
    }

    fn shape(
        (n, c, h, w): (usize, usize, usize, usize),
        (o, kh, kw): (usize, usize, usize),
        (stride, pad, dilation): (usize, usize, usize),
    ) -> ConvShape {
        let oh = (h + 2 * pad - (kh - 1) * dilation - 1) / stride + 1;
        let ow = (w + 2 * pad - (kw - 1) * dilation - 1) / stride + 1;
        ConvShape {
            n,
            c,
            h,
            w,
            o,
            kh,
            kw,
            stride,
            pad,
            dilation,
            oh,
            ow,
        }
    }

    /// The naive direct convolution the im2col kernel replaced.
    fn naive_forward(x: &[f32], w: &[f32], bias: &[f32], s: &ConvShape) -> Vec<f32> {
        let mut out = vec![0.0f32; s.n * s.o * s.oh * s.ow];
        for ni in 0..s.n {
            for oi in 0..s.o {
                for y in 0..s.oh {
                    for xi in 0..s.ow {
                        let mut acc = 0.0f32;
                        for ci in 0..s.c {
                            for ky in 0..s.kh {
                                let iy = y * s.stride + ky * s.dilation;
                                if iy < s.pad || iy >= s.h + s.pad {
                                    continue;
                                }
                                for kx in 0..s.kw {
                                    let ix = xi * s.stride + kx * s.dilation;
                                    if ix < s.pad || ix >= s.w + s.pad {
                                        continue;
                                    }
                                    acc += x
                                        [((ni * s.c + ci) * s.h + iy - s.pad) * s.w + ix - s.pad]
                                        * w[((oi * s.c + ci) * s.kh + ky) * s.kw + kx];
                                }
                            }
                        }
                        let b = if bias.is_empty() { 0.0 } else { bias[oi] };
                        out[((ni * s.o + oi) * s.oh + y) * s.ow + xi] = acc + b;
                    }
                }
            }
        }
        out
    }

    /// The naive gradient loop the col2im kernel replaced: `(dx, dw, dbias)`.
    fn naive_backward(
        x: &[f32],
        w: &[f32],
        go: &[f32],
        s: &ConvShape,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut gx = vec![0.0f32; x.len()];
        let mut gw = vec![0.0f32; w.len()];
        let mut gb = vec![0.0f32; s.o];
        for ni in 0..s.n {
            for oi in 0..s.o {
                for y in 0..s.oh {
                    for xi in 0..s.ow {
                        let g = go[((ni * s.o + oi) * s.oh + y) * s.ow + xi];
                        gb[oi] += g;
                        for ci in 0..s.c {
                            for ky in 0..s.kh {
                                let iy = y * s.stride + ky * s.dilation;
                                if iy < s.pad || iy >= s.h + s.pad {
                                    continue;
                                }
                                for kx in 0..s.kw {
                                    let ix = xi * s.stride + kx * s.dilation;
                                    if ix < s.pad || ix >= s.w + s.pad {
                                        continue;
                                    }
                                    let xi_ =
                                        ((ni * s.c + ci) * s.h + iy - s.pad) * s.w + ix - s.pad;
                                    let wi = ((oi * s.c + ci) * s.kh + ky) * s.kw + kx;
                                    gw[wi] += g * x[xi_];
                                    gx[xi_] += g * w[wi];
                                }
                            }
                        }
                    }
                }
            }
        }
        (gx, gw, gb)
    }

    #[test]
    fn im2col_convolution_matches_the_direct_loop() {
        // ((n, c, h, w), (o, kh, kw), (stride, pad, dilation), bias)
        let cases = [
            ((2, 3, 7, 9), (4, 3, 3), (1, 1, 1), true),
            ((3, 2, 8, 8), (5, 3, 3), (2, 1, 1), false),
            ((2, 4, 9, 7), (3, 3, 3), (1, 2, 2), true),
            ((1, 5, 6, 6), (6, 1, 1), (1, 0, 1), true),
            ((3, 5, 6, 6), (2, 1, 1), (1, 0, 1), false),
            ((2, 3, 6, 7), (4, 1, 1), (2, 0, 1), false),
            ((4, 2, 1, 11), (3, 1, 3), (1, 0, 2), true),
            ((2, 3, 11, 11), (2, 7, 7), (2, 3, 1), false),
            ((3, 2, 4, 4), (2, 3, 3), (3, 4, 1), true),
            ((2, 2, 5, 5), (3, 2, 2), (1, 0, 1), true),
            ((5, 3, 5, 6), (4, 3, 3), (1, 1, 3), true),
        ];
        for (at, &(input, kernel, geom, bias)) in cases.iter().enumerate() {
            let s = shape(input, kernel, geom);
            let seed = at as u64 * 10;
            let x = random(s.n * s.c * s.h * s.w, seed + 1);
            let w = random(s.o * s.c * s.kh * s.kw, seed + 2);
            let b = if bias {
                random(s.o, seed + 3)
            } else {
                Vec::new()
            };
            let go = random(s.n * s.o * s.oh * s.ow, seed + 4);
            let want = naive_forward(&x, &w, &b, &s);
            let (gx, gw, gb) = naive_backward(&x, &w, &go, &s);
            let one = s.rows() * s.p();
            // The default budget (one group), one image per group, and two images per group (an
            // uneven last group whenever the batch is odd).
            for budget in [COL_BUDGET, 1, 2 * one] {
                let what = format!("case {at} {s:?} budget {budget}");
                let got = conv2d_forward_in(&x, &w, &b, &s, budget);
                assert_close(&format!("{what}: forward"), &got, &want);
                let (fx, fw, fb) = conv2d_backward_in(&x, &w, &go, &s, budget);
                assert_close(&format!("{what}: dx"), &fx, &gx);
                assert_close(&format!("{what}: dw"), &fw, &gw);
                assert_close(&format!("{what}: dbias"), &fb, &gb);
            }
        }
    }

    #[test]
    fn groups_are_balanced() {
        let s = shape((8, 1, 2, 2), (1, 1, 1), (1, 0, 1));
        let one = s.rows() * s.p();
        assert_eq!(s.group(COL_BUDGET), 8);
        // Six fit, but 6 + 2 needs two groups just like 4 + 4 does.
        assert_eq!(s.group(6 * one), 4);
        assert_eq!(s.group(3 * one), 3);
        assert_eq!(s.group(1), 1);
    }

    /// `C = A B + beta C` by the triple loop, `C` row-major `m x n`.
    fn naive_gemm(
        (m, k, n): (usize, usize, usize),
        (a, sa): (&[f32], Strides),
        (b, sb): (&[f32], Strides),
        beta: f32,
        c: &mut [f32],
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += a[i * sa.0 + t * sa.1] * b[t * sb.0 + j * sb.1];
                }
                let at = i * n + j;
                c[at] = acc + if beta == 0.0 { 0.0 } else { beta * c[at] };
            }
        }
    }

    #[test]
    fn split_gemm_matches_the_triple_loop() {
        // Large enough products are split into a grid of blocks; odd sizes leave ragged edges.
        for (at, &(m, k, n)) in [
            (1, 1, 1),
            (5, 0, 3),
            (13, 7, 29),
            (200, 90, 600),
            (300, 120, 133),
            (37, 300, 1001),
        ]
        .iter()
        .enumerate()
        {
            let seed = 100 + at as u64 * 10;
            for (ta, tb, beta) in [(false, false, 0.0), (true, false, 1.0), (false, true, 1.0)] {
                let a = random(m * k, seed);
                let b = random(k * n, seed + 1);
                let sa = if ta { (1, m) } else { (k, 1) };
                let sb = if tb { (1, k) } else { (n, 1) };
                let c0 = random(m * n, seed + 2);
                let (mut got, mut want) = (c0.clone(), c0);
                gemm(m, k, n, &a, sa, &b, sb, beta, &mut got, (n, 1));
                naive_gemm((m, k, n), (&a, sa), (&b, sb), beta, &mut want);
                assert_close(&format!("gemm {m}x{k}x{n} {ta} {tb}"), &got, &want);
            }
        }
    }

    /// Reads `v` at output coordinate `i` of `out` through broadcast strides `st`.
    fn at(i: usize, out: &[usize], st: &[usize]) -> usize {
        let os = strides(out);
        let mut rest = i;
        let mut src = 0;
        for (o, s) in os.iter().zip(st) {
            src += rest / o * s;
            rest %= o;
        }
        src
    }

    fn bstrides(shape: &[usize], out: &[usize]) -> Vec<usize> {
        let st = strides(shape);
        let off = out.len() - shape.len();
        let mut s = vec![0usize; out.len()];
        for i in 0..shape.len() {
            s[i + off] = if shape[i] == out[i + off] { st[i] } else { 0 };
        }
        s
    }

    #[test]
    fn broadcast_walks_match_index_arithmetic() {
        let cases: [(&[usize], &[usize]); 8] = [
            (&[2, 3, 4, 5], &[1, 3, 1, 1]),
            (&[2, 3, 4, 5], &[5]),
            (&[2, 1, 4, 1], &[1, 3, 1, 5]),
            (&[3, 1], &[1, 4]),
            (&[2, 3], &[2, 3]),
            (&[1], &[4, 1, 2]),
            (&[4, 3, 1, 2], &[3, 5, 1]),
            (&[1, 1], &[1, 1]),
        ];
        for (k, (sa, sb)) in cases.iter().enumerate() {
            let out = crate::tensor::ops::broadcast_shape(sa, sb).unwrap();
            let n: usize = out.iter().product();
            let (as_, bs) = (bstrides(sa, &out), bstrides(sb, &out));
            let a = random(sa.iter().product(), k as u64);
            let b = random(sb.iter().product(), k as u64 + 50);
            let got = broadcast_map(&out, &as_, &bs, &a, &b, |x, y| x * 3.0 - y);
            let want: Vec<f32> = (0..n)
                .map(|i| a[at(i, &out, &as_)] * 3.0 - b[at(i, &out, &bs)])
                .collect();
            assert_close(&format!("broadcast {sa:?} {sb:?}"), &got, &want);

            let full = random(n, k as u64 + 99);
            for (shape, st) in [(sa, &as_), (sb, &bs)] {
                let len: usize = shape.iter().product();
                let got = sum_to(&out, st, &full, len);
                let mut want = vec![0.0f32; len];
                for (i, v) in full.iter().enumerate() {
                    want[at(i, &out, st)] += v;
                }
                assert_close(&format!("sum_to {shape:?} from {out:?}"), &got, &want);
            }
        }
    }

    #[test]
    fn fused_batch_norm_relu_matches_the_two_nodes() {
        use crate::tensor::{BnMode, Graph};
        let (n, c, h, w) = (3, 4, 5, 6);
        let x = random(n * c * h * w, 31);
        let gamma = random(c, 32);
        let beta = random(c, 33);
        let weight = random(n * c * h * w, 34);
        let modes = [
            BnMode::Train,
            BnMode::Eval {
                mean: random(c, 35),
                var: random(c, 36).iter().map(|v| v.abs() + 0.5).collect(),
            },
        ];
        for mode in modes {
            let run = |fused: bool| {
                let mut g = Graph::new();
                let xi = g.from_data(x.clone(), &[n, c, h, w]);
                let gi = g.from_data(gamma.clone(), &[c]);
                let bi = g.from_data(beta.clone(), &[c]);
                let y = if fused {
                    g.batch_norm_relu(xi, gi, bi, 1e-5, mode.clone())
                } else {
                    let y = g.batch_norm(xi, gi, bi, 1e-5, mode.clone());
                    g.relu(y)
                };
                let wt = g.from_data(weight.clone(), &[n, c, h, w]);
                let prod = g.mul(y, wt);
                let loss = g.sum_all(prod);
                g.backward(loss).unwrap();
                [
                    g.to_vec(y),
                    g.gradient(xi).to_vec(),
                    g.gradient(gi).to_vec(),
                    g.gradient(bi).to_vec(),
                ]
            };
            let (fused, plain) = (run(true), run(false));
            for (what, (a, b)) in ["value", "dx", "dgamma", "dbeta"]
                .iter()
                .zip(fused.iter().zip(&plain))
            {
                assert_eq!(a, b, "{what} ({mode:?})");
            }
        }
    }

    #[test]
    fn streams_agree_with_their_serial_form_above_the_threshold() {
        let n = STREAM_MIN + 1234;
        let a = random(n, 7);
        let b = random(n, 8);
        let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y + 1.0).collect();
        assert_eq!(zip_map(&a, &b, |x, y| x * y + 1.0), want);
        let want: Vec<f32> = a.iter().map(|x| x.max(0.0)).collect();
        assert_eq!(map(&a, |x| x.max(0.0)), want);
        let mut acc = a.clone();
        add_assign(&mut acc, &b);
        let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert_eq!(acc, want);
    }
}
