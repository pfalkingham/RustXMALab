//! OpenCV's discrete Fourier transform in single and double precision, as `cv::matchTemplate`
//! uses it through `crossCorr` (single precision for 8-bit inputs, double precision for
//! 32-bit float inputs): the 2D real-to-CCS forward transform, `mulSpectrums` with the
//! conjugate, and the 2D CCS-to-real inverse transform, operation for operation, so that
//! the rounding of every value is OpenCV's.
//!
//! On x86-64 OpenCV's baseline includes SSE3, so `dxt.cpp` takes its SSE3 radix-2, radix-3
//! and radix-4 butterflies (`DFT_VecR2/R3/R4<float>`) in single precision. They perform the
//! same `f32` operations on the same operands as the scalar butterflies (`DFT_R2`, `DFT_R3`
//! and the scalar radix-4 loop): the sign changes they do with XOR are exact, and every sum
//! and product pairs the same values. In double precision OpenCV has no vector butterflies
//! and runs the scalar ones. The scalar forms are ported here, generic over [`Real`].
//!
//! Complex arrays are interleaved slices: element `k` is `(a[2k], a[2k + 1])`.
//! Only the code paths `crossCorr` reaches are ported: no IPP, no `DFT_ROWS`, no complex
//! input or output in 2D, and out-of-place 1D transforms (OpenCV's 2D transform always
//! works through a row buffer).

use std::f64::consts::PI;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The element type of a transform: `f32` (OpenCV's `DFT_32f`) or `f64` (`DFT_64f`).
/// Constants the C++ writes as `double` literals and casts with `(T)` go through
/// [`Real::from_f64`].
pub(crate) trait Real:
    Copy
    + Debug
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// 0.
    const ZERO: Self;
    /// 1.
    const ONE: Self;
    /// `(T)v`: rounded to nearest for `f32`, exact for `f64`.
    fn from_f64(v: f64) -> Self;
    /// `(double)v`: exact.
    fn to_f64(self) -> f64;
}

impl Real for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    fn from_f64(v: f64) -> f32 {
        v as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Real for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    fn from_f64(v: f64) -> f64 {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// The single-precision 1D transform (the tests use it by this name).
#[cfg(test)]
type Dft1d = Dft1dOf<f32>;

/// `bitrevTab` of `dxt.cpp`: the bit reversal of each byte.
fn bitrev(b: usize) -> usize {
    (b as u8).reverse_bits() as usize
}

/// `DFTTab` of `dxt.cpp`: `(cos, sin)` of `2 pi / 2^m`.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
const DFT_TAB: [[f64; 2]; 32] = [
    [1.00000000000000000, 0.00000000000000000],
    [-1.00000000000000000, 0.00000000000000000],
    [0.00000000000000000, 1.00000000000000000],
    [0.70710678118654757, 0.70710678118654746],
    [0.92387953251128674, 0.38268343236508978],
    [0.98078528040323043, 0.19509032201612825],
    [0.99518472667219693, 0.09801714032956060],
    [0.99879545620517241, 0.04906767432741802],
    [0.99969881869620425, 0.02454122852291229],
    [0.99992470183914450, 0.01227153828571993],
    [0.99998117528260111, 0.00613588464915448],
    [0.99999529380957619, 0.00306795676296598],
    [0.99999882345170188, 0.00153398018628477],
    [0.99999970586288223, 0.00076699031874270],
    [0.99999992646571789, 0.00038349518757140],
    [0.99999998161642933, 0.00019174759731070],
    [0.99999999540410733, 0.00009587379909598],
    [0.99999999885102686, 0.00004793689960307],
    [0.99999999971275666, 0.00002396844980842],
    [0.99999999992818922, 0.00001198422490507],
    [0.99999999998204725, 0.00000599211245264],
    [0.99999999999551181, 0.00000299605622633],
    [0.99999999999887801, 0.00000149802811317],
    [0.99999999999971945, 0.00000074901405658],
    [0.99999999999992983, 0.00000037450702829],
    [0.99999999999998246, 0.00000018725351415],
    [0.99999999999999567, 0.00000009362675707],
    [0.99999999999999889, 0.00000004681337854],
    [0.99999999999999978, 0.00000002340668927],
    [0.99999999999999989, 0.00000001170334463],
    [1.00000000000000000, 0.00000000585167232],
    [1.00000000000000000, 0.00000000292583616],
];

// `Constants<T>`: double literals, converted with `(T)` where they are used ([`Real::from_f64`]).
#[allow(clippy::excessive_precision)]
const SIN_120: f64 = 0.866_025_403_784_438_646_763_723_170_752_94;
#[allow(clippy::excessive_precision)]
const FFT5_2: f64 = 0.559_016_994_374_947_424_102_293_417_182_819;
#[allow(clippy::excessive_precision)]
const FFT5_3: f64 = -0.951_056_516_295_153_572_116_439_333_379_382;
#[allow(clippy::excessive_precision)]
const FFT5_4: f64 = -1.538_841_768_587_626_701_285_145_288_018_455;
#[allow(clippy::excessive_precision)]
const FFT5_5: f64 = 0.363_271_264_002_680_442_947_733_378_740_309;

/// `cv::getOptimalDFTSize`: the smallest `2^a 3^b 5^c` not below `size0` (OpenCV looks it up
/// in `optimalDFTSizeTab`, which lists exactly these numbers), or `None` past the table's
/// end (2,123,366,400).
///
/// Ported from core/src/dxt.cpp, cv::getOptimalDFTSize.
pub(crate) fn optimal_dft_size(size0: usize) -> Option<usize> {
    const LAST: usize = 2_123_366_400;
    if size0 >= LAST {
        return None;
    }
    let mut n = size0.max(1);
    loop {
        let mut m = n;
        for p in [2, 3, 5] {
            while m.is_multiple_of(p) {
                m /= p;
            }
        }
        if m == 1 {
            return Some(n);
        }
        n += 1;
    }
}

/// The factors of `n`: powers of two first as one factor, then odd factors, ordered as
/// OpenCV orders them.
///
/// Ported from core/src/dxt.cpp, DFTFactorize.
fn dft_factorize(n0: usize) -> Vec<usize> {
    let mut n = n0 as i64;
    let mut factors: Vec<i64> = Vec::new();
    if n <= 5 {
        return vec![n0];
    }
    let mut f = (((n - 1) ^ n) + 1) >> 1;
    if f > 1 {
        factors.push(f);
        n = if f == n { 1 } else { n / f };
    }
    f = 3;
    while n > 1 {
        let d = n / f;
        if d * f == n {
            factors.push(f);
            n = d;
        } else {
            f += 2;
            if f * f > n {
                break;
            }
        }
    }
    if n > 1 {
        factors.push(n);
    }
    let nf = factors.len();
    let f0 = usize::from(factors[0] & 1 == 0);
    let mut i = f0;
    while i < (nf + f0) / 2 {
        factors.swap(i, nf - i - 1 + f0);
        i += 1;
    }
    factors.into_iter().map(|v| v as usize).collect()
}

/// The digit-reversal table and the twiddle factors (`f32`, computed in `f64`).
///
/// Ported from core/src/dxt.cpp, DFTInit (single-precision `wave`).
fn dft_init<T: Real>(n0: usize, factors: &[usize], inv_itab: bool) -> (Vec<i32>, Vec<T>) {
    let nf = factors.len();
    let mut itab0 = vec![0i32; n0];
    let mut wave = vec![T::ZERO; 2 * n0];
    let mut n = factors[0];
    let mut m = 0usize;

    if n0 <= 5 {
        itab0[0] = 0;
        itab0[n0 - 1] = (n0 - 1) as i32;
        if n0 != 4 {
            for (i, v) in itab0.iter_mut().enumerate().take(n0 - 1).skip(1) {
                *v = i as i32;
            }
        } else {
            itab0[1] = 2;
            itab0[2] = 1;
        }
        if n0 == 5 {
            wave[0] = T::ONE;
            wave[1] = T::ZERO;
        }
        if n0 != 4 {
            return (itab0, wave);
        }
        m = 2;
    } else {
        let mut digits = [0usize; 34];
        let mut radix = [0usize; 34];
        assert!(nf < 34);
        radix[nf] = 1;
        digits[nf] = 0;
        for i in 0..nf {
            digits[i] = 0;
            radix[nf - i - 1] = radix[nf - i] * factors[nf - i - 1];
        }
        // With `inv_itab` the table is built in a scratch buffer (OpenCV uses `wave`) and
        // inverted into `itab0` afterwards.
        let use_tmp = inv_itab && factors[0] != factors[nf - 1];
        let mut tmp = if use_tmp { vec![0i32; n0] } else { Vec::new() };
        {
            let itab: &mut [i32] = if use_tmp { &mut tmp } else { &mut itab0 };
            if n & 1 == 0 {
                let a = radix[1];
                let na2 = (n * a) >> 1;
                let na4 = na2 >> 1;
                m = 0;
                while (1usize << m) < n {
                    m += 1;
                }
                if n <= 2 {
                    itab[0] = 0;
                    itab[1] = na2 as i32;
                } else if n <= 256 {
                    let shift = 10 - m;
                    let mut i = 0;
                    while i + 4 <= n {
                        let j = (bitrev(i >> 2) >> shift) * a;
                        itab[i] = j as i32;
                        itab[i + 1] = (j + na2) as i32;
                        itab[i + 2] = (j + na4) as i32;
                        itab[i + 3] = (j + na2 + na4) as i32;
                        i += 4;
                    }
                } else {
                    let shift = 34 - m;
                    let mut i = 0;
                    while i < n {
                        let i4 = (i >> 2) as u32;
                        let rev = ((bitrev((i4 & 255) as usize) as u64) << 24)
                            + ((bitrev(((i4 >> 8) & 255) as usize) as u64) << 16)
                            + ((bitrev(((i4 >> 16) & 255) as usize) as u64) << 8)
                            + (bitrev((i4 >> 24) as usize) as u64);
                        // `BitRev`: the 32-bit sum shifted right, then taken as int.
                        let j = ((rev as u32) >> shift) as usize * a;
                        itab[i] = j as i32;
                        itab[i + 1] = (j + na2) as i32;
                        itab[i + 2] = (j + na4) as i32;
                        itab[i + 3] = (j + na2 + na4) as i32;
                        i += 4;
                    }
                }
                digits[1] += 1;
                if nf >= 2 {
                    let mut i = n;
                    let mut j = radix[2];
                    while i < n0 {
                        for k in 0..n {
                            itab[i + k] = itab[k] + j as i32;
                        }
                        i += n;
                        if i >= n0 {
                            break;
                        }
                        j += radix[2];
                        let mut k = 1;
                        loop {
                            digits[k] += 1;
                            if digits[k] < factors[k] {
                                break;
                            }
                            digits[k] = 0;
                            j = j + radix[k + 2] - radix[k];
                            k += 1;
                        }
                    }
                }
            } else {
                let mut i = 0;
                let mut j = 0usize;
                loop {
                    itab[i] = j as i32;
                    i += 1;
                    if i >= n0 {
                        break;
                    }
                    j += radix[1];
                    let mut k = 0;
                    loop {
                        digits[k] += 1;
                        if digits[k] < factors[k] {
                            break;
                        }
                        digits[k] = 0;
                        j = j + radix[k + 2] - radix[k];
                        k += 1;
                    }
                }
            }
        }
        if use_tmp {
            itab0[0] = 0;
            let mut i = n0 & 1;
            while i < n0 {
                let k0 = tmp[i] as usize;
                let k1 = tmp[i + 1] as usize;
                itab0[k0] = i as i32;
                itab0[k1] = (i + 1) as i32;
                i += 2;
            }
        }
    }

    let (mut w_re, mut w_im, w1_re, w1_im);
    if n0 & (n0 - 1) == 0 {
        w_re = DFT_TAB[m][0];
        w_im = -DFT_TAB[m][1];
        w1_re = w_re;
        w1_im = w_im;
    } else {
        let t = -PI * 2.0 / n0 as f64;
        w_im = t.sin();
        w_re = (1.0 - w_im * w_im).sqrt();
        w1_re = w_re;
        w1_im = w_im;
    }
    n = n0.div_ceil(2);
    wave[0] = T::ONE;
    wave[1] = T::ZERO;
    if n0 & 1 == 0 {
        wave[2 * n] = -T::ONE;
        wave[2 * n + 1] = T::ZERO;
    }
    for i in 1..n {
        wave[2 * i] = T::from_f64(w_re);
        wave[2 * i + 1] = T::from_f64(w_im);
        wave[2 * (n0 - i)] = T::from_f64(w_re);
        wave[2 * (n0 - i) + 1] = T::from_f64(-w_im);
        let t = w_re * w1_re - w_im * w1_im;
        w_im = w_re * w1_im + w_im * w1_re;
        w_re = t;
    }
    (itab0, wave)
}

/// The fields of `OcvDftOptions` a transform reads.
struct Opts<'a, T> {
    n: usize,
    factors: &'a [usize],
    tab_size: usize,
    itab: &'a [i32],
    wave: &'a [T],
    inverse: bool,
    no_permute: bool,
    scale: f64,
}

#[inline]
fn cmul_re<T: Real>(ar: T, ai: T, br: T, bi: T) -> T {
    ar * br - ai * bi
}

#[inline]
fn cmul_im<T: Real>(ar: T, ai: T, br: T, bi: T) -> T {
    ar * bi + ai * br
}

/// The mixed-radix complex transform of `c.n` elements. With `src`, the input is read from
/// there (digit-reversed, and conjugated for the inverse); without it, `dst` is transformed
/// in place.
///
/// Ported from core/src/dxt.cpp, DFT<float> (with the SSE3 butterflies DFT_VecR2/R3/R4,
/// which compute the same values as the scalar ones).
#[allow(clippy::too_many_lines)]
fn dft<T: Real>(c: &Opts<T>, src: Option<&[T]>, dst: &mut [T]) {
    let wave = c.wave;
    let n0 = c.n;
    let inv = c.inverse;
    let mut dw0 = c.tab_size;
    let scale = T::from_f64(c.scale);
    let tab_step = if c.tab_size == n0 {
        1
    } else if c.tab_size == n0 * 2 {
        2
    } else {
        c.tab_size / n0
    };
    let w = |k: usize| (wave[2 * k], wave[2 * k + 1]);

    // 0. shuffle data
    if let Some(src) = src {
        assert!(!c.no_permute);
        let sign = if inv { -T::ONE } else { T::ONE };
        let mut i = 0;
        let mut it = 0;
        while i + 2 <= n0 {
            let k0 = c.itab[it] as usize;
            let k1 = c.itab[it + tab_step] as usize;
            assert!(k0 < n0 && k1 < n0);
            dst[2 * i] = src[2 * k0];
            dst[2 * i + 1] = if inv {
                -src[2 * k0 + 1]
            } else {
                src[2 * k0 + 1]
            };
            dst[2 * i + 2] = src[2 * k1];
            dst[2 * i + 3] = if inv {
                -src[2 * k1 + 1]
            } else {
                src[2 * k1 + 1]
            };
            i += 2;
            it += 2 * tab_step;
        }
        if i < n0 {
            dst[2 * (n0 - 1)] = src[2 * (n0 - 1)];
            dst[2 * (n0 - 1) + 1] = sign * src[2 * (n0 - 1) + 1];
        }
    } else {
        if !c.no_permute {
            let nf = c.factors.len();
            assert_eq!(c.factors[0], c.factors[nf - 1]);
            if nf == 1 {
                if n0 & 3 == 0 {
                    let n2 = n0 / 2;
                    let mut i = 0;
                    let mut it = 0;
                    while i < n2 {
                        let j = c.itab[it] as usize;
                        assert!(j < n2);
                        swap_c(dst, i + 1, n2 + j);
                        if j > i {
                            swap_c(dst, i, j);
                            swap_c(dst, n2 + i + 1, n2 + j + 1);
                        }
                        i += 2;
                        it += tab_step * 2;
                    }
                }
            } else {
                for i in 0..n0 {
                    let j = c.itab[i * tab_step] as usize;
                    assert!(j < n0);
                    if j > i {
                        swap_c(dst, i, j);
                    }
                }
            }
        }
        if inv {
            for i in 0..n0 {
                dst[2 * i + 1] = -dst[2 * i + 1];
            }
        }
    }

    let mut n = 1usize;
    // 1. power-2 transforms
    if c.factors[0] & 1 == 0 {
        // radix-4 transform (DFT_VecR4<float> and the scalar loop after it)
        while n * 4 <= c.factors[0] {
            let nx = n;
            n *= 4;
            dw0 /= 4;
            let mut i = 0;
            while i < n0 {
                let v0 = i;
                let v1 = v0 + nx * 2;
                let (re, im) = (|k: usize| 2 * k, |k: usize| 2 * k + 1);

                let r0 = dst[re(v1)];
                let i0 = dst[im(v1)];
                let r4 = dst[re(v1 + nx)];
                let i4 = dst[im(v1 + nx)];
                let r1 = r0 + r4;
                let i1 = i0 + i4;
                let r3 = i0 - i4;
                let i3 = r4 - r0;
                let mut r2 = dst[re(v0)];
                let mut i2 = dst[im(v0)];
                let r4 = dst[re(v0 + nx)];
                let i4 = dst[im(v0 + nx)];
                let r0 = r2 + r4;
                let i0 = i2 + i4;
                r2 -= r4;
                i2 -= i4;
                dst[re(v0)] = r0 + r1;
                dst[im(v0)] = i0 + i1;
                dst[re(v1)] = r0 - r1;
                dst[im(v1)] = i0 - i1;
                dst[re(v0 + nx)] = r2 + r3;
                dst[im(v0 + nx)] = i2 + i3;
                dst[re(v1 + nx)] = r2 - r3;
                dst[im(v1 + nx)] = i2 - i3;

                let mut dw = dw0;
                for j in 1..nx {
                    let v0 = i + j;
                    let v1 = v0 + nx * 2;
                    let (w1r, w1i) = w(dw);
                    let (w2r, w2i) = w(dw * 2);
                    let (w3r, w3i) = w(dw * 3);
                    let (xr, xi) = (dst[re(v0 + nx)], dst[im(v0 + nx)]);
                    let r2 = xr * w2r - xi * w2i;
                    let i2 = xr * w2i + xi * w2r;
                    let (xr, xi) = (dst[re(v1)], dst[im(v1)]);
                    let r0 = xr * w1i + xi * w1r;
                    let i0 = xr * w1r - xi * w1i;
                    let (xr, xi) = (dst[re(v1 + nx)], dst[im(v1 + nx)]);
                    let r3 = xr * w3i + xi * w3r;
                    let i3 = xr * w3r - xi * w3i;

                    let r1 = i0 + i3;
                    let i1 = r0 + r3;
                    let r3 = r0 - r3;
                    let i3 = i3 - i0;
                    let r4 = dst[re(v0)];
                    let i4 = dst[im(v0)];

                    let r0 = r4 + r2;
                    let i0 = i4 + i2;
                    let r2 = r4 - r2;
                    let i2 = i4 - i2;

                    dst[re(v0)] = r0 + r1;
                    dst[im(v0)] = i0 + i1;
                    dst[re(v1)] = r0 - r1;
                    dst[im(v1)] = i0 - i1;
                    dst[re(v0 + nx)] = r2 + r3;
                    dst[im(v0 + nx)] = i2 + i3;
                    dst[re(v1 + nx)] = r2 - r3;
                    dst[im(v1 + nx)] = i2 - i3;
                    dw += dw0;
                }
                i += n;
            }
        }

        // the remaining radix-2 transform (DFT_VecR2<float> = DFT_R2<float>)
        while n < c.factors[0] {
            n *= 2;
            dw0 /= 2;
            let nx = n / 2;
            let mut i = 0;
            while i < n0 {
                let v = i;
                let r0 = dst[2 * v] + dst[2 * (v + nx)];
                let i0 = dst[2 * v + 1] + dst[2 * (v + nx) + 1];
                let r1 = dst[2 * v] - dst[2 * (v + nx)];
                let i1 = dst[2 * v + 1] - dst[2 * (v + nx) + 1];
                dst[2 * v] = r0;
                dst[2 * v + 1] = i0;
                dst[2 * (v + nx)] = r1;
                dst[2 * (v + nx) + 1] = i1;
                let mut dw = dw0;
                for j in 1..nx {
                    let v = i + j;
                    let (wr, wi) = w(dw);
                    let (xr, xi) = (dst[2 * (v + nx)], dst[2 * (v + nx) + 1]);
                    let r1 = cmul_re(xr, xi, wr, wi);
                    let i1 = cmul_im(xr, xi, wr, wi);
                    let r0 = dst[2 * v];
                    let i0 = dst[2 * v + 1];
                    dst[2 * v] = r0 + r1;
                    dst[2 * v + 1] = i0 + i1;
                    dst[2 * (v + nx)] = r0 - r1;
                    dst[2 * (v + nx) + 1] = i0 - i1;
                    dw += dw0;
                }
                i += n;
            }
        }
    }

    // 2. all the other transforms
    let first = if c.factors[0] & 1 == 1 { 0 } else { 1 };
    for &factor in &c.factors[first..] {
        let nx = n;
        n *= factor;
        dw0 /= factor;
        if factor == 3 {
            radix3(dst, n0, n, dw0, wave);
        } else if factor == 5 {
            radix5(dst, n0, n, dw0, wave);
        } else {
            radix_odd(dst, n0, n, nx, factor, dw0, c.tab_size, wave);
        }
    }

    if scale != T::ONE {
        let re_scale = scale;
        let im_scale = if inv { -scale } else { scale };
        for i in 0..n0 {
            dst[2 * i] *= re_scale;
            dst[2 * i + 1] *= im_scale;
        }
    } else if inv {
        for i in 0..n0 {
            dst[2 * i + 1] = -dst[2 * i + 1];
        }
    }
}

fn swap_c<T>(a: &mut [T], i: usize, j: usize) {
    a.swap(2 * i, 2 * j);
    a.swap(2 * i + 1, 2 * j + 1);
}

/// Ported from core/src/dxt.cpp, DFT_R3<float> (and DFT_VecR3<float>, the same values).
fn radix3<T: Real>(dst: &mut [T], c_n: usize, n: usize, dw0: usize, wave: &[T]) {
    let sin_120 = T::from_f64(SIN_120);
    let half = T::from_f64(0.5);
    let nx = n / 3;
    let mut i = 0;
    while i < c_n {
        {
            let v = i;
            let (a, b) = (v + nx, v + nx * 2);
            let r1 = dst[2 * a] + dst[2 * b];
            let i1 = dst[2 * a + 1] + dst[2 * b + 1];
            let mut r0 = dst[2 * v];
            let mut i0 = dst[2 * v + 1];
            let r2 = sin_120 * (dst[2 * a + 1] - dst[2 * b + 1]);
            let i2 = sin_120 * (dst[2 * b] - dst[2 * a]);
            dst[2 * v] = r0 + r1;
            dst[2 * v + 1] = i0 + i1;
            r0 -= half * r1;
            i0 -= half * i1;
            dst[2 * a] = r0 + r2;
            dst[2 * a + 1] = i0 + i2;
            dst[2 * b] = r0 - r2;
            dst[2 * b + 1] = i0 - i2;
        }
        let mut dw = dw0;
        for j in 1..nx {
            let v = i + j;
            let (a, b) = (v + nx, v + nx * 2);
            let (wr, wi) = (wave[2 * dw], wave[2 * dw + 1]);
            let (w2r, w2i) = (wave[4 * dw], wave[4 * dw + 1]);
            let r0 = dst[2 * a] * wr - dst[2 * a + 1] * wi;
            let i0 = dst[2 * a] * wi + dst[2 * a + 1] * wr;
            let i2 = dst[2 * b] * w2r - dst[2 * b + 1] * w2i;
            let r2 = dst[2 * b] * w2i + dst[2 * b + 1] * w2r;
            let r1 = r0 + i2;
            let i1 = i0 + r2;
            let r2 = sin_120 * (i0 - r2);
            let i2 = sin_120 * (i2 - r0);
            let mut r0 = dst[2 * v];
            let mut i0 = dst[2 * v + 1];
            dst[2 * v] = r0 + r1;
            dst[2 * v + 1] = i0 + i1;
            r0 -= half * r1;
            i0 -= half * i1;
            dst[2 * a] = r0 + r2;
            dst[2 * a + 1] = i0 + i2;
            dst[2 * b] = r0 - r2;
            dst[2 * b + 1] = i0 - i2;
            dw += dw0;
        }
        i += n;
    }
}

/// Ported from core/src/dxt.cpp, DFT_R5<float>.
fn radix5<T: Real>(dst: &mut [T], c_n: usize, n: usize, dw0: usize, wave: &[T]) {
    let (fft5_2, fft5_3, fft5_4, fft5_5) = (
        T::from_f64(FFT5_2),
        T::from_f64(FFT5_3),
        T::from_f64(FFT5_4),
        T::from_f64(FFT5_5),
    );
    let quarter = T::from_f64(0.25);
    let nx = n / 5;
    let w = |k: usize| (wave[2 * k], wave[2 * k + 1]);
    let mut i = 0;
    while i < c_n {
        let mut dw = 0;
        for j in 0..nx {
            let v0 = i + j;
            let v1 = v0 + nx * 2;
            let v2 = v1 + nx * 2;
            let ld = |k: usize, d: &[T]| (d[2 * k], d[2 * k + 1]);

            let (xr, xi) = ld(v0 + nx, dst);
            let (wr, wi) = w(dw);
            let mut r3 = xr * wr - xi * wi;
            let mut i3 = xr * wi + xi * wr;
            let (xr, xi) = ld(v2, dst);
            let (wr, wi) = w(dw * 4);
            let mut r2 = xr * wr - xi * wi;
            let mut i2 = xr * wi + xi * wr;

            let mut r1 = r3 + r2;
            let mut i1 = i3 + i2;
            r3 -= r2;
            i3 -= i2;

            let (xr, xi) = ld(v1 + nx, dst);
            let (wr, wi) = w(dw * 3);
            let mut r4 = xr * wr - xi * wi;
            let mut i4 = xr * wi + xi * wr;
            let (xr, xi) = ld(v1, dst);
            let (wr, wi) = w(dw * 2);
            let mut r0 = xr * wr - xi * wi;
            let mut i0 = xr * wi + xi * wr;

            r2 = r4 + r0;
            i2 = i4 + i0;
            r4 -= r0;
            i4 -= i0;

            r0 = dst[2 * v0];
            i0 = dst[2 * v0 + 1];
            let mut r5 = r1 + r2;
            let mut i5 = i1 + i2;

            dst[2 * v0] = r0 + r5;
            dst[2 * v0 + 1] = i0 + i5;

            r0 -= quarter * r5;
            i0 -= quarter * i5;
            r1 = fft5_2 * (r1 - r2);
            i1 = fft5_2 * (i1 - i2);
            r2 = -fft5_3 * (i3 + i4);
            i2 = fft5_3 * (r3 + r4);

            i3 *= -fft5_5;
            r3 *= fft5_5;
            i4 *= -fft5_4;
            r4 *= fft5_4;

            r5 = r2 + i3;
            i5 = i2 + r3;
            r2 -= i4;
            i2 -= r4;

            r3 = r0 + r1;
            i3 = i0 + i1;
            r0 -= r1;
            i0 -= i1;

            dst[2 * (v0 + nx)] = r3 + r2;
            dst[2 * (v0 + nx) + 1] = i3 + i2;
            dst[2 * v2] = r3 - r2;
            dst[2 * v2 + 1] = i3 - i2;

            dst[2 * v1] = r0 + r5;
            dst[2 * v1 + 1] = i0 + i5;
            dst[2 * (v1 + nx)] = r0 - r5;
            dst[2 * (v1 + nx) + 1] = i0 - i5;
            dw += dw0;
        }
        i += n;
    }
}

/// The generic odd radix (7, 11, ...). `crossCorr` never reaches it, since its transform
/// sizes have no prime factor above 5; it is ported for completeness.
///
/// Ported from core/src/dxt.cpp, DFT<float> (the "radix-factor - an odd number" branch).
#[allow(clippy::too_many_arguments)]
fn radix_odd<T: Real>(
    dst: &mut [T],
    c_n: usize,
    n: usize,
    nx: usize,
    factor: usize,
    dw0: usize,
    tab_size: usize,
    wave: &[T],
) {
    let factor2 = (factor - 1) / 2;
    let dw_f = tab_size / factor;
    let mut a = vec![(T::ZERO, T::ZERO); factor2];
    let mut b = vec![(T::ZERO, T::ZERO); factor2];
    let mut i = 0;
    while i < c_n {
        let mut dw = 0usize;
        for j in 0..nx {
            let v = i + j;
            let v_0 = (dst[2 * v], dst[2 * v + 1]);
            let mut vn_0 = v_0;
            if j == 0 {
                let mut k = nx;
                for p in 1..=factor2 {
                    let (ka, kb) = (v + k, v + n - k);
                    let r0 = dst[2 * ka] + dst[2 * kb];
                    let i0 = dst[2 * ka + 1] - dst[2 * kb + 1];
                    let r1 = dst[2 * ka] - dst[2 * kb];
                    let i1 = dst[2 * ka + 1] + dst[2 * kb + 1];
                    vn_0.0 += r0;
                    vn_0.1 += i1;
                    a[p - 1] = (r0, i0);
                    b[p - 1] = (r1, i1);
                    k += nx;
                }
            } else {
                let wbase = dw * factor;
                let mut d = dw;
                let mut k = nx;
                for p in 1..=factor2 {
                    let (ka, kb) = (v + k, v + n - k);
                    let (wr, wi) = (wave[2 * d], wave[2 * d + 1]);
                    let r2 = dst[2 * ka] * wr - dst[2 * ka + 1] * wi;
                    let i2 = dst[2 * ka] * wi + dst[2 * ka + 1] * wr;
                    let wm = wbase - d;
                    let (wr, wi) = (wave[2 * wm], wave[2 * wm + 1]);
                    let r1 = dst[2 * kb] * wr - dst[2 * kb + 1] * wi;
                    let i1 = dst[2 * kb] * wi + dst[2 * kb + 1] * wr;
                    let r0 = r2 + r1;
                    let i0 = i2 - i1;
                    let r1 = r2 - r1;
                    let i1 = i2 + i1;
                    vn_0.0 += r0;
                    vn_0.1 += i1;
                    a[p - 1] = (r0, i0);
                    b[p - 1] = (r1, i1);
                    k += nx;
                    d += dw;
                }
            }
            dst[2 * v] = vn_0.0;
            dst[2 * v + 1] = vn_0.1;

            let mut k = nx;
            for p in 1..=factor2 {
                let mut s0 = v_0;
                let mut s1 = v_0;
                let dd = dw_f * p;
                let mut d = dd;
                for q in 0..factor2 {
                    let (wr, wi) = (wave[2 * d], wave[2 * d + 1]);
                    let r0 = wr * a[q].0;
                    let i0 = wi * a[q].1;
                    let r1 = wr * b[q].1;
                    let i1 = wi * b[q].0;
                    s1.0 += r0 + i0;
                    s0.0 += r0 - i0;
                    s1.1 += r1 - i1;
                    s0.1 += r1 + i1;
                    d += dd;
                    if d >= tab_size {
                        d -= tab_size;
                    }
                }
                dst[2 * (v + k)] = s0.0;
                dst[2 * (v + k) + 1] = s0.1;
                dst[2 * (v + n - k)] = s1.0;
                dst[2 * (v + n - k) + 1] = s1.1;
                k += nx;
            }
            dw += dw0;
        }
        i += n;
    }
}

/// A 1D transform of length `n` with its tables (`OcvDftBasicImpl`).
#[derive(Debug, Clone)]
struct Dft1dOf<T> {
    n: usize,
    factors: Vec<usize>,
    itab: Vec<i32>,
    wave: Vec<T>,
    inverse: bool,
    scale: f64,
}

impl<T: Real> Dft1dOf<T> {
    /// Ported from core/src/dxt.cpp, OcvDftBasicImpl::init (no IPP). `real_rows` is
    /// stage 0 of a real transform (rows); otherwise a complex transform (stage 1, columns).
    fn new(n: usize, inverse: bool, real_rows: bool, scale: f64) -> Dft1dOf<T> {
        let factors = dft_factorize(n);
        let (itab, wave) = dft_init(n, &factors, real_rows && inverse);
        Dft1dOf {
            n,
            factors,
            itab,
            wave,
            inverse,
            scale,
        }
    }

    fn opts<'a>(
        &'a self,
        factors: &'a [usize],
        n: usize,
        inverse: bool,
        no_permute: bool,
        scale: f64,
    ) -> Opts<'a, T> {
        Opts {
            n,
            factors,
            tab_size: self.n,
            itab: &self.itab,
            wave: &self.wave,
            inverse,
            no_permute,
            scale,
        }
    }

    /// The complex transform of `src` into `dst` (both `2 n` floats, different buffers).
    ///
    /// Ported from core/src/dxt.cpp, DFT_32f.
    fn complex(&self, src: &[T], dst: &mut [T]) {
        let o = self.opts(&self.factors, self.n, self.inverse, false, self.scale);
        dft(&o, Some(src), dst);
    }

    /// The factors of the half-length transform of the even real transforms: the first
    /// factor halved, dropped when it becomes 1.
    fn half_factors(&self) -> Vec<usize> {
        let f0 = self.factors[0] >> 1;
        if f0 == 1 {
            self.factors[1..].to_vec()
        } else {
            let mut f = self.factors.clone();
            f[0] = f0;
            f
        }
    }

    /// Forward transform of the real vector `src` (`n` floats) into `dst` (`2 n` floats of
    /// scratch); the CCS result is `dst[..n]` for even `n` and `dst[1..n + 1]` for odd `n`.
    ///
    /// Ported from core/src/dxt.cpp, RealDFT<float> (not `complex_output`).
    fn real_forward(&self, src: &[T], dst: &mut [T]) {
        let n = self.n;
        let scale = T::from_f64(self.scale);
        if n == 1 {
            dst[0] = src[0] * scale;
        } else if n == 2 {
            let t = (src[0] + src[1]) * scale;
            dst[1] = (src[0] - src[1]) * scale;
            dst[0] = t;
        } else if n & 1 == 1 {
            dst[0] = src[0] * scale;
            dst[1] = T::ZERO;
            let mut j = 1;
            while j < n {
                let t0 = src[self.itab[j] as usize] * scale;
                let t1 = src[self.itab[j + 1] as usize] * scale;
                dst[2 * j] = t0;
                dst[2 * j + 1] = T::ZERO;
                dst[2 * j + 2] = t1;
                dst[2 * j + 3] = T::ZERO;
                j += 2;
            }
            let o = self.opts(&self.factors, n, false, true, 1.0);
            dft(&o, None, &mut dst[..2 * n]);
            dst[1] = dst[0];
        } else {
            let scale2 = scale * T::from_f64(0.5);
            let n2 = n >> 1;
            let hf = self.half_factors();
            let o = self.opts(&hf, n2, false, false, 1.0);
            dft(&o, Some(&src[..n]), &mut dst[..n]);

            let t = dst[0] - dst[1];
            dst[0] = (dst[0] + dst[1]) * scale;
            dst[1] = t * scale;

            let t0 = dst[n2];
            let mut t = dst[n - 1];
            dst[n - 1] = dst[1];

            let mut j = 2;
            let mut wk = 1;
            while j < n2 {
                let (wr, wi) = (self.wave[2 * wk], self.wave[2 * wk + 1]);
                // calc odd
                let h2_re = scale2 * (dst[j + 1] + t);
                let h2_im = scale2 * (dst[n - j] - dst[j]);
                // calc even
                let h1_re = scale2 * (dst[j] + dst[n - j]);
                let h1_im = scale2 * (dst[j + 1] - t);
                // rotate
                let tt = h2_re * wr - h2_im * wi;
                let h2_im = h2_re * wi + h2_im * wr;
                let h2_re = tt;
                t = dst[n - j - 1];

                dst[j - 1] = h1_re + h2_re;
                dst[n - j - 1] = h1_re - h2_re;
                dst[j] = h1_im + h2_im;
                dst[n - j] = h2_im - h1_im;
                j += 2;
                wk += 1;
            }
            if j <= n2 {
                dst[n2 - 1] = t0 * scale;
                dst[n2] = -t * scale;
            }
        }
    }

    /// Inverse transform of the CCS vector `src` (`n` floats) into the real vector
    /// `dst[..n]` (`dst` has `2 n` floats of scratch), out of place.
    ///
    /// Ported from core/src/dxt.cpp, CCSIDFT<float> (not `complex_input`, `src != dst`).
    fn ccs_inverse(&self, src: &[T], dst: &mut [T]) {
        let n = self.n;
        let scale = T::from_f64(self.scale);
        if n == 1 {
            dst[0] = src[0] * scale;
        } else if n == 2 {
            let t = (src[0] + src[1]) * scale;
            dst[1] = (src[0] - src[1]) * scale;
            dst[0] = t;
        } else if n & 1 == 1 {
            // `_src = (Complex*)(src - 1)`: element j is (src[2j - 1], src[2j]).
            dst[0] = src[0];
            dst[1] = T::ZERO;
            let n2 = (n + 1) >> 1;
            for j in 1..n2 {
                let k0 = self.itab[j] as usize;
                let k1 = self.itab[n - j] as usize;
                let t0 = src[2 * j - 1];
                let t1 = src[2 * j];
                dst[2 * k0] = t0;
                dst[2 * k0 + 1] = -t1;
                dst[2 * k1] = t0;
                dst[2 * k1 + 1] = t1;
            }
            let o = self.opts(&self.factors, n, false, true, 1.0);
            dft(&o, None, &mut dst[..2 * n]);
            dst[0] *= scale;
            let mut j = 1;
            while j < n {
                let t0 = dst[j * 2] * scale;
                let t1 = dst[j * 2 + 2] * scale;
                dst[j] = t0;
                dst[j + 1] = t1;
                j += 2;
            }
        } else {
            let mut t = src[1];
            let t0 = src[0] + src[n - 1];
            let t1 = src[n - 1] - src[0];
            dst[0] = t0;
            dst[1] = t1;
            let n2 = (n + 1) >> 1;
            let mut j = 2;
            let mut wk = 1;
            while j < n2 {
                let (wr, wi) = (self.wave[2 * wk], self.wave[2 * wk + 1]);
                let h1_re = t + src[n - j - 1];
                let h1_im = src[j] - src[n - j];
                let h2_re = t - src[n - j - 1];
                let h2_im = src[j] + src[n - j];

                let tt = h2_re * wr + h2_im * wi;
                let h2_im = h2_im * wr - h2_re * wi;
                let h2_re = tt;

                t = src[j + 1];
                let t0 = h1_re - h2_im;
                let t1 = -h1_im - h2_re;
                let t2 = h1_re + h2_im;
                let t3 = h1_im - h2_re;

                let j2 = j >> 1;
                let k = self.itab[j2] as usize;
                dst[k] = t0;
                dst[k + 1] = t1;
                let k = self.itab[n2 - j2] as usize;
                dst[k] = t2;
                dst[k + 1] = t3;
                j += 2;
                wk += 1;
            }
            if j <= n2 {
                let t0 = t * T::from_f64(2.0);
                let t1 = src[n2] * T::from_f64(2.0);
                let k = self.itab[n2] as usize;
                dst[k * 2] = t0;
                dst[k * 2 + 1] = t1;
            }
            let hf = self.half_factors();
            let o = self.opts(&hf, n2, false, true, 1.0);
            dft(&o, None, &mut dst[..n]);
            let mut j = 0;
            while j < n {
                let t0 = dst[j] * scale;
                let t1 = dst[j + 1] * (-scale);
                dst[j] = t0;
                dst[j + 1] = t1;
                j += 2;
            }
        }
    }
}

/// A single-precision 2D real transform of a `width` by `height` row-major array, forward
/// (real to CCS) or inverse (CCS to real, scaled by `1 / (width height)`), in place.
///
/// Ported from core/src/dxt.cpp, OcvDftImpl (2D, `FwdRealToCCS` and `InvCCSToReal`,
/// `CV_HAL_DFT_IS_INPLACE`, no IPP).
#[derive(Debug, Clone)]
pub(crate) struct Dft2dOf<T> {
    width: usize,
    height: usize,
    inverse: bool,
    nonzero_rows: usize,
    rows: Dft1dOf<T>,
    cols: Dft1dOf<T>,
}

impl<T: Real> Dft2dOf<T> {
    /// `hal::DFT2D::create(width, height, CV_32F, 1, 1, flags, nonzero_rows)`; the inverse
    /// has `DFT_SCALE`. `nonzero_rows` 0 means all rows.
    ///
    /// Ported from core/src/dxt.cpp, OcvDftImpl::init.
    pub(crate) fn new(
        width: usize,
        height: usize,
        inverse: bool,
        nonzero_rows: usize,
    ) -> Dft2dOf<T> {
        assert!(width > 1 && height > 1, "only the 2D case is ported");
        // Forward: rows (stage 0) then columns; inverse: columns then rows. The scale goes
        // to the last stage only.
        let row_scale = if inverse {
            1.0 / (width * height) as f64
        } else {
            1.0
        };
        Dft2dOf {
            width,
            height,
            inverse,
            nonzero_rows,
            rows: Dft1dOf::new(width, inverse, true, row_scale),
            cols: Dft1dOf::new(height, inverse, false, 1.0),
        }
    }

    /// Transforms `data` (`width * height` floats, row by row) in place.
    ///
    /// Ported from core/src/dxt.cpp, OcvDftImpl::apply.
    pub(crate) fn apply(&self, data: &mut [T]) {
        assert_eq!(data.len(), self.width * self.height);
        if self.inverse {
            self.col_dft(data);
            self.row_dft(data);
        } else {
            self.row_dft(data);
            self.col_dft(data);
        }
    }

    /// Ported from core/src/dxt.cpp, OcvDftImpl::rowDft (through `tmp_bufA`).
    fn row_dft(&self, data: &mut [T]) {
        let len = self.width;
        let count = self.height;
        let mut nz = self.nonzero_rows;
        if nz == 0 || nz > count {
            nz = count;
        }
        // `dptr_offset`: an odd forward real transform leaves its CCS one float in.
        let offset = usize::from(!self.inverse && len & 1 == 1 && len > 1);
        let mut buf = vec![T::ZERO; 2 * len];
        for i in 0..nz {
            let row = &mut data[i * len..(i + 1) * len];
            if self.inverse {
                self.rows.ccs_inverse(row, &mut buf);
            } else {
                self.rows.real_forward(row, &mut buf);
            }
            row.copy_from_slice(&buf[offset..offset + len]);
        }
        for v in &mut data[nz * len..count * len] {
            *v = T::ZERO;
        }
    }

    /// Ported from core/src/dxt.cpp, OcvDftImpl::colDft (real transform, one channel in
    /// and out), with CopyColumn, CopyFrom2Columns, CopyTo2Columns and ExpandCCS.
    fn col_dft(&self, data: &mut [T]) {
        let len = self.height;
        let count = self.width;
        let step = self.width;
        let even = count & 1 == 0;
        let b = count.div_ceil(2);
        let mut buf0 = vec![T::ZERO; 2 * len];
        let mut buf1 = vec![T::ZERO; 2 * len];
        let mut out0 = vec![T::ZERO; 2 * len];
        let mut out1 = vec![T::ZERO; 2 * len];

        // The real first column and (for even widths) the real last column.
        if self.inverse {
            for r in 0..len {
                buf0[r] = data[r * step];
            }
            expand_ccs(&mut buf0, len);
            if even {
                for r in 0..len {
                    buf1[r] = data[r * step + count - 1];
                }
                expand_ccs(&mut buf1, len);
            }
        } else {
            buf0.fill(T::ZERO);
            for r in 0..len {
                buf0[2 * r] = data[r * step];
            }
            if even {
                buf1.fill(T::ZERO);
                for r in 0..len {
                    buf1[2 * r] = data[r * step + count - 1];
                }
            }
        }
        if even {
            self.cols.complex(&buf1, &mut out1);
        }
        self.cols.complex(&buf0, &mut out0);
        if self.inverse {
            // the real part of the complex vector to the first/last column
            for r in 0..len {
                data[r * step] = out0[2 * r];
            }
            if even {
                for r in 0..len {
                    data[r * step + count - 1] = out1[2 * r];
                }
            }
        } else {
            // the CCS half of the output vector to the first/last column
            out0[1] = out0[0];
            for r in 0..len {
                data[r * step] = out0[1 + r];
            }
            if even {
                out1[1] = out1[0];
                for r in 0..len {
                    data[r * step + count - 1] = out1[1 + r];
                }
            }
        }

        // The complex columns, two at a time: float columns 1 + 2 (i - 1) and the next.
        let mut i = 1;
        while i < b {
            let c0 = 1 + 2 * (i - 1);
            if i + 1 < b {
                for r in 0..len {
                    let s = r * step + c0;
                    buf0[2 * r] = data[s];
                    buf0[2 * r + 1] = data[s + 1];
                    buf1[2 * r] = data[s + 2];
                    buf1[2 * r + 1] = data[s + 3];
                }
                self.cols.complex(&buf1, &mut out1);
            } else {
                for r in 0..len {
                    let s = r * step + c0;
                    buf0[2 * r] = data[s];
                    buf0[2 * r + 1] = data[s + 1];
                }
            }
            self.cols.complex(&buf0, &mut out0);
            for r in 0..len {
                let s = r * step + c0;
                data[s] = out0[2 * r];
                data[s + 1] = out0[2 * r + 1];
                if i + 1 < b {
                    data[s + 2] = out1[2 * r];
                    data[s + 3] = out1[2 * r + 1];
                }
            }
            i += 2;
        }
    }
}

/// Unpacks a CCS vector of `n` floats into `n` complex values in place (`p` holds
/// `2 n` floats).
///
/// Ported from core/src/dxt.cpp, ExpandCCS (single precision).
fn expand_ccs<T: Real>(p: &mut [T], n0: usize) {
    let mut n = n0;
    for i in 1..n.div_ceil(2) {
        p[(n - i) * 2] = p[i * 2 - 1];
        p[(n - i) * 2 + 1] = -p[i * 2];
    }
    if n & 1 == 0 {
        p[n] = p[n - 1];
        p[n + 1] = T::ZERO;
        n -= 1;
    }
    for i in (1..n).rev() {
        p[i + 1] = p[i];
    }
    p[1] = T::ZERO;
}

/// `cv::mulSpectrums(a, b, a, 0, conjB = true)` for 2D single-channel CCS arrays: `a`
/// becomes `a * conj(b)`. Complex products are formed in `f64` and rounded to `f32`; the
/// purely real entries (first and, for even widths, last column; first and, for even
/// heights, last row of those) are multiplied in `f32`.
///
/// Ported from core/src/dxt.cpp, cv::mulSpectrums and mulSpectrums_Impl_<float, true, true>
/// (not 1D, one channel).
pub(crate) fn mul_spectrums_conj<T: Real>(a: &mut [T], b: &[T], width: usize, height: usize) {
    assert!(width > 1 && height > 1);
    assert_eq!(a.len(), width * height);
    assert_eq!(b.len(), width * height);
    let rows = height;
    let cols = width;
    let mut process_col = |c: usize| {
        let at = |r: usize| r * cols + c;
        a[at(0)] *= b[at(0)];
        let mut j = 1;
        while j + 2 <= rows {
            let (a_re, a_im) = (a[at(j)].to_f64(), a[at(j + 1)].to_f64());
            let (b_re, b_im) = (b[at(j)].to_f64(), -b[at(j + 1)].to_f64());
            let c_re = a_re * b_re - a_im * b_im;
            let c_im = a_re * b_im + a_im * b_re;
            a[at(j)] = T::from_f64(c_re);
            a[at(j + 1)] = T::from_f64(c_im);
            j += 2;
        }
        if rows & 1 == 0 {
            a[at(rows - 1)] *= b[at(rows - 1)];
        }
    };
    process_col(0);
    if cols & 1 == 0 {
        process_col(cols - 1);
    }
    let j1 = cols - usize::from(cols & 1 == 0);
    for r in 0..rows {
        let base = r * cols;
        let mut j = 1;
        while j < j1 {
            let (a_re, a_im) = (a[base + j].to_f64(), a[base + j + 1].to_f64());
            let (b_re, b_im) = (b[base + j].to_f64(), -b[base + j + 1].to_f64());
            let c_re = a_re * b_re - a_im * b_im;
            let c_im = a_re * b_im + a_im * b_re;
            a[base + j] = T::from_f64(c_re);
            a[base + j + 1] = T::from_f64(c_im);
            j += 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 1D complex transform against a direct sum in f64.
    fn naive(src: &[f32], inverse: bool) -> Vec<f64> {
        let n = src.len() / 2;
        let sgn = if inverse { 1.0 } else { -1.0 };
        let mut out = vec![0f64; 2 * n];
        for k in 0..n {
            let (mut re, mut im) = (0.0, 0.0);
            for j in 0..n {
                let ang = sgn * 2.0 * PI * ((j * k) % n) as f64 / n as f64;
                let (s, c) = ang.sin_cos();
                let (xr, xi) = (f64::from(src[2 * j]), f64::from(src[2 * j + 1]));
                re += xr * c - xi * s;
                im += xr * s + xi * c;
            }
            out[2 * k] = re;
            out[2 * k + 1] = im;
        }
        out
    }

    fn signal(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                ((x >> 8) % 1000) as f32 / 7.0 - 60.0
            })
            .collect()
    }

    #[test]
    fn factorize_like_opencv() {
        assert_eq!(dft_factorize(80), vec![16, 5]);
        assert_eq!(dft_factorize(90), vec![2, 5, 3, 3]);
        assert_eq!(dft_factorize(45), vec![5, 3, 3]);
        assert_eq!(dft_factorize(4), vec![4]);
        assert_eq!(dft_factorize(64), vec![64]);
        assert_eq!(optimal_dft_size(77), Some(80));
        assert_eq!(optimal_dft_size(87), Some(90));
        assert_eq!(optimal_dft_size(1), Some(1));
        assert_eq!(optimal_dft_size(7), Some(8));
    }

    #[test]
    fn complex_transform_is_a_dft() {
        for n in [
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 21, 25, 27, 32, 45, 49, 60, 64, 77, 80,
            81, 90, 96, 100, 120, 128, 135, 256, 300, 512,
        ] {
            for inverse in [false, true] {
                let src = signal(2 * n, n as u32);
                let d = Dft1d::new(n, inverse, false, 1.0);
                let mut out = vec![0f32; 2 * n];
                d.complex(&src, &mut out);
                let want = naive(&src, inverse);
                let mag = want.iter().fold(0f64, |m, v| m.max(v.abs()));
                for (g, w) in out.iter().zip(&want) {
                    assert!(
                        (f64::from(*g) - w).abs() <= 1e-5 * mag,
                        "n {n} inverse {inverse}: {g} vs {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn real_transforms_round_trip() {
        for n in [
            3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 20, 45, 77, 80, 81, 90, 96, 100, 135,
        ] {
            let src = signal(n, 3 * n as u32);
            let fwd = Dft1d::new(n, false, true, 1.0);
            let inv = Dft1d::new(n, true, true, 1.0 / n as f64);
            let mut buf = vec![0f32; 2 * n];
            fwd.real_forward(&src, &mut buf);
            let off = usize::from(n & 1 == 1);
            let ccs: Vec<f32> = buf[off..off + n].to_vec();
            // CCS against the direct sum.
            let cplx: Vec<f32> = src.iter().flat_map(|&v| [v, 0.0]).collect();
            let want = naive(&cplx, false);
            let mag = want.iter().fold(0f64, |m, v| m.max(v.abs()));
            let mut expect = vec![want[0]];
            for k in 1..n.div_ceil(2) {
                expect.push(want[2 * k]);
                expect.push(want[2 * k + 1]);
            }
            if n & 1 == 0 {
                expect.push(want[n]);
            }
            for (g, w) in ccs.iter().zip(&expect) {
                assert!((f64::from(*g) - w).abs() <= 1e-5 * mag, "n {n}: {g} vs {w}");
            }
            let mut back = vec![0f32; 2 * n];
            inv.ccs_inverse(&ccs, &mut back);
            for (g, w) in back[..n].iter().zip(&src) {
                assert!((g - w).abs() <= 1e-4, "n {n}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn in_place_permutation_matches_copy() {
        for n in [8, 16, 64, 9, 25, 27] {
            let src = signal(2 * n, 11);
            let d = Dft1d::new(n, false, false, 1.0);
            let mut a = vec![0f32; 2 * n];
            d.complex(&src, &mut a);
            let mut b = src.clone();
            let o = d.opts(&d.factors, n, false, false, 1.0);
            dft(&o, None, &mut b);
            assert_eq!(a, b, "n {n}");
        }
    }
}
