//! `GaussianBlur` (8-bit and 32-bit float) and `medianBlur` (aperture 3, 8-bit), as
//! OpenCV 4.13 computes them for XMALab's calls (`processing/MarkerDetection.cpp`,
//! `MarkerDetection::detectionPoint`; `processing/FilterImage.cpp`, `FilterImage::run`).
//!
//! Border handling is OpenCV's default for the Gaussian (`BORDER_REFLECT_101`, reflected as
//! often as needed when the kernel is wider than the image, as `borderInterpolate` does) and
//! `BORDER_REPLICATE` for the median.
//!
//! The float blur reproduces the code OpenCV runs on x86-64 with AVX2 and FMA (the
//! `filter.simd.hpp` AVX2 dispatch), including which multiply-adds are fused; see
//! [`gaussian_blur_f32`] for what this depends on.

use crate::image::{ImageF32, ImageU8};

/// `getGaussianKernelBitExact(result, n, sigma)` in `f64`: the normalised kernel.
///
/// Ported from `modules/imgproc/src/smooth.dispatch.cpp`, `getGaussianKernelBitExact`.
/// OpenCV evaluates it in `softdouble`; this uses the platform's `exp`. The 8-bit blur rounds
/// the kernel to multiples of 1/256 and the float blur to `f32`, so a last-bit difference in
/// `exp` matters only at an exact rounding tie (none in the reference cases).
#[must_use]
pub fn gaussian_kernel(n: usize, sigma: f64) -> Vec<f64> {
    assert!(n > 0, "kernel size");
    if sigma <= 0.0 {
        let fixed: Option<&[f64]> = match n {
            1 => Some(&[1.0]),
            3 => Some(&[0.25, 0.5, 0.25]),
            5 => Some(&[0.0625, 0.25, 0.375, 0.25, 0.0625]),
            7 => Some(&[
                0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125,
            ]),
            9 => Some(&[
                4.0 / 256.0,
                13.0 / 256.0,
                30.0 / 256.0,
                51.0 / 256.0,
                60.0 / 256.0,
                51.0 / 256.0,
                30.0 / 256.0,
                13.0 / 256.0,
                4.0 / 256.0,
            ]),
            _ => None,
        };
        if let Some(f) = fixed {
            return f.to_vec();
        }
    }
    let sigma_x = if sigma > 0.0 {
        sigma
    } else {
        // mulAdd(n, 0.15, 0.35), fused in softdouble.
        (n as f64).mul_add(
            f64::from_bits(0x3fc3_3333_3333_3333),
            f64::from_bits(0x3fd6_6666_6666_6666),
        )
    };
    let scale2x = -0.125 / (sigma_x * sigma_x);
    let n2 = (n - 1) / 2;
    let mut values = Vec::with_capacity(n2);
    let mut sum = 0.0;
    let mut x = 1 - n as i64;
    for _ in 0..n2 {
        let t = ((x * x) as f64 * scale2x).exp();
        values.push(t);
        sum += t;
        x += 2;
    }
    sum *= 2.0;
    sum += 1.0;
    if n.is_multiple_of(2) {
        sum += 1.0;
    }
    let mul1 = 1.0 / sum;
    let mut result = vec![0.0; n];
    for (i, v) in values.iter().enumerate() {
        let t = v * mul1;
        result[i] = t;
        result[n - 1 - i] = t;
    }
    result[n2] = mul1;
    if n.is_multiple_of(2) {
        result[n2 + 1] = mul1;
    }
    result
}

/// `getGaussianKernelFixedPoint_ED(result, kernel, 8)`: the kernel in units of 1/256 with
/// error diffusion, the centre taking what makes the sum exactly 256. `n` odd.
///
/// Ported from `modules/imgproc/src/smooth.dispatch.cpp`, `getGaussianKernelFixedPoint_ED`.
#[must_use]
pub fn gaussian_kernel_fixed(kernel: &[f64]) -> Vec<u16> {
    let n = kernel.len();
    assert!(n % 2 == 1, "odd kernel");
    let mult = 256.0;
    let n2 = n / 2;
    let mut err = 0.0;
    let mut sum: i64 = 0;
    let mut out = vec![0u16; n];
    for i in 0..n2 {
        let adj = kernel[i] * mult + err;
        let v0 = adj.round_ties_even() as i64;
        err = adj - v0 as f64;
        out[i] = v0 as u16;
        out[n - 1 - i] = v0 as u16;
        sum += v0;
    }
    sum *= 2;
    out[n2] = (256 - sum) as u16;
    out
}

/// `borderInterpolate(p, len, BORDER_REFLECT_101)`: reflects as many times as needed.
///
/// Ported from `modules/core/src/copy.cpp`, `cv::borderInterpolate`.
///
/// The reflections repeat with period `2 (len - 1)`, so the index is reduced modulo it
/// first: the same value as OpenCV's loop for every argument that loop handles, at once,
/// and without overflow for any `p` and `len` (a length of 0 or 1 gives 0).
#[must_use]
pub fn reflect101(p: isize, len: usize) -> usize {
    reflect101_wide(p as i128, len)
}

/// [`reflect101`] for an index of any integer type up to 64 bits, widened to `i128` (exact):
/// the body of [`reflect101`] and of `crate::denoise::reflect101`.
///
/// Ported from `modules/core/src/copy.cpp`, `cv::borderInterpolate`.
pub(crate) fn reflect101_wide(p: i128, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let len = len as i128;
    let period = 2 * (len - 1);
    let q = p.rem_euclid(period);
    (if q < len { q } else { period - q }) as usize
}

/// `cv::GaussianBlur(src, dst, Size(k, k), sigma)` on an 8-bit grey image (`k` odd,
/// `BORDER_REFLECT_101`), raw row-ordered pixels in and out.
///
/// Ported from `modules/imgproc/src/smooth.dispatch.cpp`, `GaussianBlur` (the `CV_8U` branch)
/// and `smooth.simd.hpp`, `GaussianBlurFixedPoint` / `fixedSmoothInvoker`: the bit-exact path
/// every 8-bit blur takes. The kernel is rounded to 8 fraction bits
/// ([`gaussian_kernel_fixed`]); rows are filtered into 16-bit fixed point (8 fraction bits),
/// columns into 32-bit (16 fraction bits), rounded with `+ 2^15 >> 16`. Every sum is exact in
/// integers, so the order of the additions does not matter. A dimension of 1 pixel gets a
/// 1-tap kernel, as in `GaussianBlur`.
#[must_use]
pub fn gaussian_blur_u8_raw(
    src: &[u8],
    width: usize,
    height: usize,
    k: usize,
    sigma: f64,
) -> Vec<u8> {
    assert_eq!(src.len(), width * height, "image size");
    if k <= 1 || width == 0 || height == 0 {
        return src.to_vec();
    }
    let kx = if width == 1 {
        vec![256u16]
    } else {
        gaussian_kernel_fixed(&gaussian_kernel(k, sigma))
    };
    let ky = if height == 1 {
        vec![256u16]
    } else {
        gaussian_kernel_fixed(&gaussian_kernel(k, sigma))
    };
    let threads = if (width * height) <= 1 << 16 {
        // Small images (detection patches) are not worth a thread.
        1
    } else {
        cpu_count()
            .min(height)
            .min((width * height).div_ceil(1 << 16))
    };
    blur_u8_fixed(src, width, height, &kx, &ky, threads)
}

/// The number of threads the 8-bit blur may use (`available_parallelism`), asked once.
fn cpu_count() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get))
}

/// The two passes of [`gaussian_blur_u8_raw`] with the fixed-point kernels `kx` and `ky`
/// (odd lengths, symmetric, non-negative, summing to 256), on `threads` bands of rows (the
/// calling thread alone for 1). Each output pixel is computed on its own, so the result does
/// not depend on `threads`.
///
/// The kernels' symmetry halves the multiplies, as OpenCV's `hlineSmoothONa_yzy_a` and
/// `vlineSmoothONa_yzy_a` do: `k_t (x[c - t] + x[c + t])`. Every partial sum is at most the
/// full sum, which is at most `255 * 256` for a row (it fits the `u16` of the row buffer)
/// and `255 * 256 * 256` for a column, so all are exact in `u32` whatever the kernel length.
///
/// Ported from `modules/imgproc/src/smooth.simd.hpp`, `fixedSmoothInvoker` (with
/// `hlineSmoothONa_yzy_a` and `vlineSmoothONa_yzy_a`).
fn blur_u8_fixed(
    src: &[u8],
    width: usize,
    height: usize,
    kx: &[u16],
    ky: &[u16],
    threads: usize,
) -> Vec<u8> {
    let (rx, ry) = (kx.len() / 2, ky.len() / 2);
    // kc[0] the centre tap, kc[t] the taps t away on either side.
    let (kcx, kcy) = (&kx[rx..], &ky[ry..]);
    let per = height.div_ceil(threads.max(1)).max(1);
    // Rows: u8 x 8.8 fixed point into u16 (exact, at most 255 * 256).
    let mut rows = vec![0u16; width * height];
    // The pixels whose window lies inside the row: x in rx..width - rx.
    let (lo, hi) = (rx.min(width), width.saturating_sub(rx).max(rx.min(width)));
    let row_band = |first: usize, chunk: &mut [u16]| {
        let mut acc = vec![0u32; hi - lo];
        for (r, out) in chunk.chunks_mut(width).enumerate() {
            let row = &src[(first + r) * width..][..width];
            // Inside: a tap at a time over the whole run of pixels.
            let w0 = u32::from(kcx[0]);
            for (a, &v) in acc.iter_mut().zip(&row[lo..hi]) {
                *a = w0 * u32::from(v);
            }
            if lo < hi {
                for (t, &w) in kcx.iter().enumerate().skip(1) {
                    let w = u32::from(w);
                    let (left, right) = (&row[lo - t..hi - t], &row[lo + t..hi + t]);
                    for ((a, &va), &vb) in acc.iter_mut().zip(left).zip(right) {
                        *a += w * (u32::from(va) + u32::from(vb));
                    }
                }
            }
            for (o, &a) in out[lo..hi].iter_mut().zip(&acc) {
                *o = a as u16;
            }
            // The edges, reflected.
            for x in (0..lo).chain(hi..width) {
                let mut a = w0 * u32::from(row[x]);
                for (t, &w) in kcx.iter().enumerate().skip(1) {
                    let xa = reflect101(x as isize - t as isize, width);
                    let xb = reflect101(x as isize + t as isize, width);
                    a += u32::from(w) * (u32::from(row[xa]) + u32::from(row[xb]));
                }
                out[x] = a as u16;
            }
        }
    };
    if threads <= 1 {
        row_band(0, &mut rows);
    } else {
        std::thread::scope(|s| {
            for (ci, chunk) in rows.chunks_mut(per * width).enumerate() {
                let row_band = &row_band;
                s.spawn(move || row_band(ci * per, chunk));
            }
        });
    }
    // Columns: 8.8 x 8.8 into 16.16, then (v + 2^15) >> 16.
    let mut dst = vec![0u8; width * height];
    let rows = &rows;
    let col_band = |first: usize, chunk: &mut [u8]| {
        let mut acc = vec![0u32; width];
        for (r, out) in chunk.chunks_mut(width).enumerate() {
            let y = first + r;
            let w0 = u32::from(kcy[0]);
            for (a, &v) in acc.iter_mut().zip(&rows[y * width..][..width]) {
                *a = w0 * u32::from(v);
            }
            for (t, &w) in kcy.iter().enumerate().skip(1) {
                let ya = reflect101(y as isize - t as isize, height);
                let yb = reflect101(y as isize + t as isize, height);
                let (ra, rb) = (&rows[ya * width..][..width], &rows[yb * width..][..width]);
                let w = u32::from(w);
                for ((a, &va), &vb) in acc.iter_mut().zip(ra).zip(rb) {
                    *a += w * (u32::from(va) + u32::from(vb));
                }
            }
            for (o, &a) in out.iter_mut().zip(&acc) {
                *o = ((a + (1 << 15)) >> 16).min(255) as u8;
            }
        }
    };
    if threads <= 1 {
        col_band(0, &mut dst);
    } else {
        std::thread::scope(|s| {
            for (ci, chunk) in dst.chunks_mut(per * width).enumerate() {
                let col_band = &col_band;
                s.spawn(move || col_band(ci * per, chunk));
            }
        });
    }
    dst
}

/// `cv::GaussianBlur(src, dst, Size(k, k), sigma)` on an 8-bit grey image, `k` odd, default
/// border (`BORDER_REFLECT_101`). Detection calls it with `k = 3`, `sigma = 1.3`.
///
/// Ported from `modules/imgproc/src/smooth.dispatch.cpp`, `GaussianBlur` (the `CV_8U`
/// branch); see [`gaussian_blur_u8_raw`].
#[must_use]
pub fn gaussian_blur_u8(src: &ImageU8, k: usize, sigma: f64) -> ImageU8 {
    assert!(k % 2 == 1, "kernel size must be odd");
    ImageU8::from_vec(
        src.width(),
        src.height(),
        gaussian_blur_u8_raw(src.data(), src.width(), src.height(), k, sigma),
    )
}

/// Number of `f32` lanes of OpenCV's AVX2 vector code (`v_float32`, 256 bits).
const LANES: usize = 8;

/// One row of the separable float filter: `out[i]` from `ext[i .. i + k]`, where `ext` is the
/// row extended by `k / 2` reflected pixels on each side. `k` is 1, 5 or more.
///
/// Ported from `modules/imgproc/src/filter.simd.hpp`: `RowFilter<float, float, RowVec_32f>`
/// for kernels longer than 5, `SymmRowSmallFilter<float, float, SymmRowSmallVec_32f>` for 1 and
/// 5 taps (`getLinearRowFilter`). The rounding of each part of the row is described at
/// [`gaussian_blur_f32`].
fn row_filter_f32(ext: &[f32], kx: &[f32], out: &mut [f32]) {
    let width = out.len();
    let k = kx.len();
    let vec_end = width / LANES * LANES;
    let r = k / 2;
    if k > 5 {
        // Fused chain over [0, block_end): the AVX loop of RowVec_32f (s = 0, then
        // s = fma(x, k, s) for every tap), and the 4-pixel unrolled loop after it.
        let block_end = vec_end + (width - vec_end) / 4 * 4;
        // The single-pixel loop: separate multiply and add, except for the last
        // (k - 1) % 4 taps, which are fused.
        let fused_from = k - (k - 1) % 4;
        for (i, o) in out.iter_mut().enumerate() {
            let mut s = ext[i] * kx[0];
            for (t, &w) in kx.iter().enumerate().skip(1) {
                s = if i < block_end || t >= fused_from {
                    ext[i + t].mul_add(w, s)
                } else {
                    s + ext[i + t] * w
                };
            }
            *o = s;
        }
        return;
    }
    let kc = &kx[r..]; // kc[0] centre, kc[j] the tap j away
    match k {
        1 => {
            // `ksize == 1 && kx[0] == 1` copies; any other single tap multiplies.
            for (i, o) in out.iter_mut().enumerate() {
                *o = if kc[0] == 1.0 { ext[i] } else { kc[0] * ext[i] };
            }
        }
        5 => {
            // Gaussian kernels sum to 1, never the (1, 0, -2, 0, 1) special case.
            assert!(
                !(kc[0] == -2.0 && kc[1] == 0.0 && kc[2] == 1.0),
                "5-tap integer kernels are not ported"
            );
            // After the vector part, a loop over pairs of pixels runs while two remain; an
            // odd remainder leaves the last pixel to the single-pixel loop.
            let single = if (width - vec_end) % 2 == 1 {
                width - 1
            } else {
                usize::MAX
            };
            for (i, o) in out.iter_mut().enumerate() {
                let c = i + r;
                let p1 = ext[c - 1] + ext[c + 1];
                let p2 = ext[c - 2] + ext[c + 2];
                *o = if i == single {
                    // s0 = k0*S0; s0 += k1*(S1 + S-1); s0 += k2*(S2 + S-2), not fused.
                    (kc[0] * ext[c] + kc[1] * p1) + kc[2] * p2
                } else {
                    // vector: fma(S2 + S-2, k2, fma(S0, k0, (S-1 + S1) * k1)); the pair loop
                    // gives the same.
                    p2.mul_add(kc[2], ext[c].mul_add(kc[0], p1 * kc[1]))
                };
            }
        }
        _ => unreachable!("kernel of {k} taps"),
    }
}

/// `cv::GaussianBlur(src, dst, Size(k, k), sigma)` on a 32-bit float image, `k` odd, `sigmaY`
/// 0 (so equal to `sigma`), default border (`BORDER_REFLECT_101`). Detection calls it with
/// `k = 2r + 1`, `r = (int)(1.5 * masksize + 0.5)`, on patches that can be narrower than the
/// kernel: the border is then reflected repeatedly, as OpenCV's `borderInterpolate` does.
///
/// Ported from `modules/imgproc/src/smooth.dispatch.cpp`, `GaussianBlur` (the generic branch:
/// `createGaussianKernels` then `sepFilter2D`) and `modules/imgproc/src/filter.dispatch.cpp`,
/// `createSeparableLinearFilter`, with the row and column filters of
/// `modules/imgproc/src/filter.simd.hpp` (`RowVec_32f` or `SymmRowSmallVec_32f`, then
/// `SymmColumnFilter` with `SymmColumnVec_32f`) and `FilterEngine__proceed` for the borders.
///
/// The kernel is [`gaussian_kernel`] rounded to `f32`. A 1-pixel dimension gets a 1-tap
/// kernel; a 1 x 1 kernel copies.
///
/// Rounding. Rows are filtered first, then columns. Which multiply-adds are fused depends on
/// where a pixel falls in OpenCV's loops (vector part, unrolled scalar loop, single-pixel
/// loop), and the source alone does not say it: OpenCV's vector code uses fused
/// multiply-adds, and whether the compiler fuses its scalar `s += k*x` is a build property.
/// The rules below are those that reproduce the reference outputs (Python OpenCV 4.14 on
/// x86-64 with AVX2; 56 float blur cases, every value identical); `w` is the image width:
/// - rows, kernel longer than 5: for `x < 8 * (w / 8) + 4 * ((w % 8) / 4)` a fused chain
///   `fma(x_t, k_t, s)` over the taps from the left; for the remaining (at most 3) pixels the
///   same chain with separate multiply and add, except the last `(k - 1) % 4` taps, which
///   are fused (as if the compiler unrolled the tap loop by 4 and fused only its remainder);
/// - rows, 5 taps (masksize 1): `fma(S2 + S-2, k2, fma(S0, k0, (S-1 + S1) * k1))`, except
///   the last pixel of a row whose `w % 8` is odd: `(k0*S0 + k1*(S-1 + S1)) + k2*(S-2 + S2)`
///   unfused;
/// - columns: for `x < 8 * (w / 8)` `fma(centre, k0, 0)` then `fma(above_j + below_j, k_j, s)`
///   for `j = 1..r`; for the last `w % 8` pixels the same with separate multiply and add.
///
/// These are last-bit differences (about 1e-7 relative). The C++ reference build (OpenCV 4.13,
/// no IPP) is a different compilation and may round its scalar tails differently; what it
/// does is checked only through the detection outputs that depend on it.
///
/// # Panics
/// If `k` is even or 3 (OpenCV uses `SymmColumnSmallFilter` for a 3-tap float filter; XMALab
/// never asks for it: `2 * (int)(1.5 * m + 0.5) + 1` is 1, 5, 7, ...).
#[must_use]
pub fn gaussian_blur_f32(src: &ImageF32, k: usize, sigma: f64) -> ImageF32 {
    assert!(k % 2 == 1, "kernel size must be odd");
    assert!(k != 3, "a 3-tap float Gaussian is not ported");
    let (w, h) = (src.width(), src.height());
    if w == 0 || h == 0 {
        return src.clone();
    }
    let kw = if w == 1 { 1 } else { k };
    let kh = if h == 1 { 1 } else { k };
    if kw == 1 && kh == 1 {
        return src.clone();
    }
    let sigma = sigma.max(0.0);
    let kx: Vec<f32> = gaussian_kernel(kw, sigma)
        .iter()
        .map(|&v| v as f32)
        .collect();
    let ky: Vec<f32> = gaussian_kernel(kh, sigma)
        .iter()
        .map(|&v| v as f32)
        .collect();
    let (rx, ry) = (kw / 2, kh / 2);

    // Row pass on every row of the image.
    let mut rows = vec![0f32; w * h];
    let mut ext = vec![0f32; w + 2 * rx];
    for y in 0..h {
        let row = src.row(y);
        for (j, e) in ext.iter_mut().enumerate() {
            *e = row[reflect101(j as isize - rx as isize, w)];
        }
        row_filter_f32(&ext, &kx, &mut rows[y * w..(y + 1) * w]);
    }

    // Column pass: SymmColumnFilter / SymmColumnVec_32f with delta 0.
    let kc = &ky[ry..];
    let vec_end = w / LANES * LANES;
    let mut dst = ImageF32::new(w, h);
    for y in 0..h {
        let centre = &rows[y * w..(y + 1) * w];
        let out = dst.row_mut(y);
        for (o, &c) in out.iter_mut().zip(centre) {
            // fma(c, k0, 0) and k0*c + 0 round the same.
            *o = c * kc[0] + 0.0;
        }
        for (j, &kj) in kc.iter().enumerate().skip(1) {
            let ya = reflect101(y as isize - j as isize, h);
            let yb = reflect101(y as isize + j as isize, h);
            let (ra, rb) = (&rows[ya * w..(ya + 1) * w], &rows[yb * w..(yb + 1) * w]);
            for (x, ((o, &a), &b)) in out.iter_mut().zip(ra).zip(rb).enumerate() {
                // OpenCV adds src[k] (below) + src[-k] (above); addition is commutative.
                *o = if x < vec_end {
                    (b + a).mul_add(kj, *o)
                } else {
                    *o + kj * (b + a)
                };
            }
        }
    }
    dst
}

/// `cv::medianBlur(src, dst, 3)` on an 8-bit grey image: the median of each 3 x 3
/// neighbourhood, edge pixels replicated (`BORDER_REPLICATE`). In place is allowed in OpenCV
/// (it copies the source first); here the result is always a new image.
///
/// Ported from `modules/imgproc/src/median_blur.simd.hpp`, `medianBlur` and
/// `medianBlur_SortNet<MinMax8u, MinMaxVec8u>` (`m == 3`): a sorting network, so the exact
/// median; a 1-pixel-wide or -high image takes the median of 3 along its length, which is the
/// same as the replicated 3 x 3 median.
#[must_use]
pub fn median_blur3_u8(src: &ImageU8) -> ImageU8 {
    let (w, h) = (src.width(), src.height());
    let mut dst = ImageU8::new(w, h);
    if w == 0 || h == 0 {
        return dst;
    }
    for y in 0..h {
        let r0 = src.row(y.saturating_sub(1));
        let r1 = src.row(y);
        let r2 = src.row((y + 1).min(h - 1));
        let out = dst.row_mut(y);
        for (x, o) in out.iter_mut().enumerate() {
            let x0 = x.saturating_sub(1);
            let x2 = (x + 1).min(w - 1);
            let mut v = [
                r0[x0], r0[x], r0[x2], r1[x0], r1[x], r1[x2], r2[x0], r2[x], r2[x2],
            ];
            v.sort_unstable();
            *o = v[4];
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loop of `cv::borderInterpolate` (`BORDER_REFLECT_101`), as this function was
    /// written before.
    fn reflect101_loop(p: isize, len: usize) -> usize {
        let len = len as isize;
        if len == 1 {
            return 0;
        }
        let mut p = p;
        loop {
            if p < 0 {
                p = -p;
            } else if p >= len {
                p = 2 * len - 2 - p;
            } else {
                return p as usize;
            }
        }
    }

    #[test]
    fn reflect101_equals_the_loop_and_takes_any_argument() {
        for len in 1..40usize {
            for p in -300isize..300 {
                assert_eq!(reflect101(p, len), reflect101_loop(p, len), "{p} {len}");
            }
        }
        // Extremes: no overflow, no long loop, an index in range.
        for (p, len) in [
            (isize::MIN, 2usize),
            (isize::MAX, 2),
            (isize::MIN, 1024),
            (isize::MAX, usize::MAX),
            (isize::MIN, usize::MAX),
            (-1, 0),
            (5, 0),
        ] {
            let r = reflect101(p, len);
            assert!(r < len.max(1), "{p} {len} -> {r}");
        }
    }

    #[test]
    fn reflect_repeats_for_wide_kernels() {
        // 0 1 2 3 4 | 3 2 1 0 1 2 ...
        assert_eq!(reflect101(5, 5), 3);
        assert_eq!(reflect101(9, 5), 1);
        assert_eq!(reflect101(-6, 5), 2);
        assert_eq!(reflect101(-18, 21), 18);
        assert_eq!(reflect101(7, 1), 0);
    }

    /// The 8-bit blur as it was before the third review round (no symmetry, a thread
    /// scope even for one band), kept to compare with.
    fn old_blur_u8_raw(src: &[u8], width: usize, height: usize, k: usize, sigma: f64) -> Vec<u8> {
        assert_eq!(src.len(), width * height, "image size");
        if k <= 1 || width == 0 || height == 0 {
            return src.to_vec();
        }
        let kx = if width == 1 {
            vec![256u16]
        } else {
            gaussian_kernel_fixed(&gaussian_kernel(k, sigma))
        };
        let ky = if height == 1 {
            vec![256u16]
        } else {
            gaussian_kernel_fixed(&gaussian_kernel(k, sigma))
        };
        let (rx, ry) = (kx.len() / 2, ky.len() / 2);
        // Rows: u8 x 8.8 fixed point into u16 (exact, at most 255 * 256).
        let mut rows = vec![0u16; width * height];
        let threads = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(height)
            // Small images (detection patches) are not worth a thread each.
            .min((width * height).div_ceil(1 << 16));
        let per = height.div_ceil(threads.max(1)).max(1);
        std::thread::scope(|s| {
            for (ci, chunk) in rows.chunks_mut(per * width).enumerate() {
                let kx = &kx;
                s.spawn(move || {
                    for (r, out) in chunk.chunks_mut(width).enumerate() {
                        let row = &src[(ci * per + r) * width..][..width];
                        for (x, o) in out.iter_mut().enumerate() {
                            let mut acc: u32 = 0;
                            if x >= rx && x + rx < width {
                                let win = &row[x - rx..=x + rx];
                                for (&w, &v) in kx.iter().zip(win) {
                                    acc += u32::from(w) * u32::from(v);
                                }
                            } else {
                                for (t, &w) in kx.iter().enumerate() {
                                    let xi =
                                        reflect101(x as isize + t as isize - rx as isize, width);
                                    acc += u32::from(w) * u32::from(row[xi]);
                                }
                            }
                            *o = acc as u16;
                        }
                    }
                });
            }
        });
        // Columns: 8.8 x 8.8 into 16.16, then (v + 2^15) >> 16.
        let mut dst = vec![0u8; width * height];
        std::thread::scope(|s| {
            for (ci, chunk) in dst.chunks_mut(per * width).enumerate() {
                let (ky, rows) = (&ky, &rows);
                s.spawn(move || {
                    let mut acc = vec![0u32; width];
                    for (r, out) in chunk.chunks_mut(width).enumerate() {
                        let y = ci * per + r;
                        acc.fill(0);
                        for (t, &w) in ky.iter().enumerate() {
                            let yi = reflect101(y as isize + t as isize - ry as isize, height);
                            let row = &rows[yi * width..][..width];
                            let w = u32::from(w);
                            for (a, &v) in acc.iter_mut().zip(row) {
                                *a += w * u32::from(v);
                            }
                        }
                        for (o, &a) in out.iter_mut().zip(&acc) {
                            *o = ((a + (1 << 15)) >> 16).min(255) as u8;
                        }
                    }
                });
            }
        });
        dst
    }

    /// A deterministic pseudo-random image.
    fn noise(w: usize, h: usize, seed: u64) -> Vec<u8> {
        let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..w * h)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (s >> 56) as u8
            })
            .collect()
    }

    /// Every fixed-point kernel the 8-bit blur can be given (every odd length up to the
    /// visual filter's largest, 2 MAX_KRAD + 1 = 2001, and sigmas from 0 to infinity and
    /// NaN) sums to exactly 256: with `u16` taps, the bound on which the paired sums'
    /// exactness rests.
    #[test]
    fn review13_fixed_kernels_sum_to_256() {
        for k in (1..=2001usize).step_by(2) {
            for sigma in [
                0.0,
                -1.0,
                1e-300,
                0.3,
                1.3,
                10.0,
                58.9,
                1000.0,
                1e300,
                f64::INFINITY,
                f64::NAN,
            ] {
                let kf = gaussian_kernel_fixed(&gaussian_kernel(k, sigma));
                let sum: u32 = kf.iter().map(|&v| u32::from(v)).sum();
                assert_eq!(sum, 256, "k = {k}, sigma = {sigma}");
                assert!(kf.iter().zip(kf.iter().rev()).all(|(a, b)| a == b));
            }
        }
    }

    /// Finding 4 (symmetry) and finding 2 (threads): the new passes give the old bytes, on
    /// any number of threads.
    #[test]
    fn review13_blur_u8_equals_old() {
        let sizes = [
            (1, 1),
            (1, 7),
            (7, 1),
            (2, 2),
            (3, 5),
            (21, 21),
            (33, 17),
            (64, 3),
            (5, 64),
            (130, 97),
        ];
        let kernels: [(usize, f64); 9] = [
            (1, 1.0),
            (3, 1.3),
            (3, 0.0),
            (5, 0.0),
            (9, 0.0),
            (35, 10.0),
            (37, 58.9),
            (199, 30.0),
            (2001, 10.0),
        ];
        for (seed, &(w, h)) in sizes.iter().enumerate() {
            let src = noise(w, h, seed as u64);
            for &(k, sigma) in &kernels {
                let want = old_blur_u8_raw(&src, w, h, k, sigma);
                assert_eq!(
                    gaussian_blur_u8_raw(&src, w, h, k, sigma),
                    want,
                    "{w}x{h} k {k}"
                );
                if k > 1 {
                    let kx = if w == 1 {
                        vec![256u16]
                    } else {
                        gaussian_kernel_fixed(&gaussian_kernel(k, sigma))
                    };
                    let ky = if h == 1 {
                        vec![256u16]
                    } else {
                        gaussian_kernel_fixed(&gaussian_kernel(k, sigma))
                    };
                    for threads in [1, 2, 3, 7, 200] {
                        assert_eq!(
                            blur_u8_fixed(&src, w, h, &kx, &ky, threads),
                            want,
                            "{w}x{h} k {k} threads {threads}"
                        );
                    }
                }
            }
        }
        // Extremes: all 255 and all 0 with the largest kernel (the largest sums), a large
        // image split over several threads, and a 1 x 1 image with an even size (accepted
        // there: both kernels are 1 tap).
        for v in [0u8, 255] {
            let src = vec![v; 40 * 30];
            for k in [3, 35, 2001] {
                assert_eq!(
                    gaussian_blur_u8_raw(&src, 40, 30, k, 0.0),
                    old_blur_u8_raw(&src, 40, 30, k, 0.0)
                );
            }
        }
        let big = noise(300, 260, 99);
        for k in [3, 35] {
            assert_eq!(
                gaussian_blur_u8_raw(&big, 300, 260, k, 10.0),
                old_blur_u8_raw(&big, 300, 260, k, 10.0)
            );
        }
        assert_eq!(gaussian_blur_u8_raw(&[9], 1, 1, 4, 1.0), vec![9]);
        assert_eq!(old_blur_u8_raw(&[9], 1, 1, 4, 1.0), vec![9]);
        assert!(gaussian_blur_u8_raw(&[], 0, 5, 3, 1.0).is_empty());
    }

    #[test]
    fn flat_float_image_stays_nearly_flat() {
        let src = ImageF32::filled(21, 21, 100.0);
        let out = gaussian_blur_f32(&src, 37, 58.9);
        assert!(out.data().iter().all(|&v| (v - 100.0).abs() < 1e-3));
    }

    #[test]
    fn one_tap_copies() {
        let src = ImageF32::from_vec(3, 1, vec![1.0, 2.0, 3.0]);
        assert_eq!(gaussian_blur_f32(&src, 1, 2.0), src);
        let col = ImageF32::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let out = gaussian_blur_f32(&col, 7, 2.0);
        assert!((out.at(0, 1) - 2.0).abs() < 1.0);
    }

    #[test]
    fn median_of_a_spike_is_the_background() {
        let mut src = ImageU8::filled(5, 4, 10);
        src.set(2, 2, 255);
        src.set(0, 0, 0);
        let out = median_blur3_u8(&src);
        assert!(out.data().iter().all(|&v| v == 10));
    }
}
