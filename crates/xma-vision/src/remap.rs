//! Undistorted images and the false-colour displacement maps of the undistortion grid.
//!
//! Ported from:
//!
//! - OpenCV 4.13 `modules/imgproc/src/imgwarp.cpp`: `cv::remap` with two `CV_32FC1` maps,
//!   `INTER_LANCZOS4`, `BORDER_CONSTANT` with value 0, on 8-bit images of 1 or 3 channels
//!   ([`remap_lanczos4_u8`]): the maps quantised to 1/32 pixel (`cvRound(x * 32)`, the
//!   planar-map branch of `RemapInvoker`), the 8 × 8 tap table of `initInterTab2D(
//!   INTER_LANCZOS4, true)` in 15-bit fixed point with its sum correction, and
//!   `remapLanczos4` with `FixedPtCast<int, uchar, 15>` and the constant-border branch; on
//!   16-bit images ([`remap_lanczos4_u16`]) the float table of `initInterTab2D(
//!   INTER_LANCZOS4, false)` and `remapLanczos4<Cast<float, ushort>, float, 1>`.
//!   This is what `core/UndistortionObject.cpp`, `UndistortionObject::undistort`, and
//!   `core/CalibrationSequence.cpp`, `CalibrationSequence::undistort`, call with the
//!   camera's lookup table ([`crate::undistortion::create_lookup_table`]).
//! - OpenCV 4.13 `modules/calib3d/src/undistort.dispatch.cpp` and `undistort.simd.hpp`:
//!   `cv::initUndistortRectifyMap` for `CV_32FC1` maps with no rectification and the camera
//!   matrix as the new matrix ([`lens_undistortion_maps`]), as OpenCV runs it on a
//!   processor with AVX2: the second remap the C++ applies to calibration images of a
//!   camera with a lens model (`Camera::undistort`) and to every frame of File > Undistort
//!   sequence.
//! - `core/UndistortionObject.cpp`, `computeTmpImage` and the five `getDisplacement*`
//!   functions ([`displacement_image`]), with OpenCV's `minMaxIdx`, `convertScaleAbs`,
//!   `applyColorMap` (`COLORMAP_JET`, `COLORMAP_HSV`) and `multiply`.
//!
//! # What was compared with what
//!
//! Checked for export (File > Undistort sequence writes what these functions return,
//! `xma_engine::undistort_sequence`); the displacement maps serve the display only.
//!
//! - [`remap_lanczos4_u8`] and [`remap_lanczos4_u16`]: equal, value for value, to OpenCV
//!   4.13 (the libraries XMALab 3 ships, through `oracle/py/undistort_probe/`) on the nine
//!   `remap413-*` cases (8 and 16 bits, 1 and 3 channels, grid-like maps with -1 holes,
//!   coordinates out of range, every 1/32 fraction and its rounding edges, content that
//!   saturates both ways, 12-bit data), the two `remapchain413-*` cases (a grid-like remap
//!   then the lens remap, 8 and 16 bits), and the pig's own images at full size
//!   (`undistortpig-*`, `oracle/py/gen_primitives_undistort.py`): both grid images and both
//!   calibration images through XMALab's own path, grid only and with a lens model; three
//!   video frames; 16-bit images made from a calibration image and a video frame, grid only
//!   and with a lens model: no value differs in any of the 14 images (1 048 576 pixels
//!   each). The 8-bit remap also equals OpenCV 4.14 on the six older `remap-lanczos4-*`
//!   cases and the pig grid images (`oracle/py/DISPLAY_CHECKS.md`). The 16-bit sum is a
//!   `float` sum in OpenCV's order; a build of OpenCV that contracted it into fused
//!   multiply-adds could differ by one unit at some pixels (the AppImage's does not).
//! - The Lanczos table uses the platform's `sin` and `cos` in double precision, as OpenCV
//!   does; a different maths library could change a table entry by one unit.
//! - [`lens_undistortion_maps`]: bit for bit equal to OpenCV 4.13 and 4.14 on the four
//!   `initundistortrectifymap-*` cases (zero, k1 k2, rational, skew; widths not a multiple
//!   of 8) and the two `initundistortrectifymap413-*` cases (the lens models of
//!   `rv_lens_sparse`), and through the pig images above at 1024 × 1024. It reproduces
//!   OpenCV's AVX2 path (fused multiply-adds); OpenCV on a processor without AVX2 takes
//!   its SSE2 path, whose maps can differ in the last bits of some values.
//! - [`displacement_image`]: the colour tables are OpenCV's `COLORMAP_JET` and
//!   `COLORMAP_HSV` as `applyColorMap` produces them for every grey level (taken from
//!   OpenCV 4.14, `oracle/py/display_colormaps.py`); on the pig lookup tables the
//!   distance, horizontal and vertical maps equal a Python/OpenCV port of the C++, and the
//!   two angle maps differ in at most 13 values by at most 6 (`atan2f` against numpy).
//!   Display only.

/// `INTER_BITS`: map coordinates are quantised to `1 / 2^5` pixel.
const INTER_BITS: i32 = 5;
const INTER_TAB_SIZE: i32 = 1 << INTER_BITS;
const COEF_BITS: i32 = 15;
const COEF_SCALE: i32 = 1 << COEF_BITS;

/// `interpolateLanczos4(x, coeffs)`: the eight 1-D weights for a fractional offset `x`.
fn lanczos4_1d(x: f32) -> [f32; 8] {
    // OpenCV's literal 0.70710678118654752440084436210485.
    const S45: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const CS: [[f64; 2]; 8] = [
        [1.0, 0.0],
        [-S45, -S45],
        [0.0, 1.0],
        [S45, -S45],
        [-1.0, 0.0],
        [S45, S45],
        [0.0, -1.0],
        [-S45, S45],
    ];
    let mut c = [0.0f32; 8];
    if x < f32::EPSILON {
        c[3] = 1.0;
        return c;
    }
    let mut sum = 0.0f32;
    let y0 = -f64::from(x + 3.0) * std::f64::consts::PI * 0.25;
    let (s0, c0) = (y0.sin(), y0.cos());
    for (i, ci) in c.iter_mut().enumerate() {
        let y = -f64::from(x + 3.0 - i as f32) * std::f64::consts::PI * 0.25;
        *ci = ((CS[i][0] * s0 + CS[i][1] * c0) / (y * y)) as f32;
        sum += *ci;
    }
    let sum = 1.0f32 / sum;
    for ci in &mut c {
        *ci *= sum;
    }
    c
}

/// `cvRound` of a float: round half to even.
fn round_f32(v: f32) -> i32 {
    v.round_ties_even() as i32
}

/// `initInterTab2D(INTER_LANCZOS4, true)`: for each of the 32 × 32 fractional positions,
/// 64 weights in 15-bit fixed point summing exactly to `1 << 15`.
fn lanczos4_table() -> &'static [[i16; 64]] {
    use std::sync::OnceLock;
    static TAB: OnceLock<Vec<[i16; 64]>> = OnceLock::new();
    TAB.get_or_init(|| {
        let scale = 1.0f32 / INTER_TAB_SIZE as f32;
        let tab1: Vec<[f32; 8]> = (0..INTER_TAB_SIZE)
            .map(|i| lanczos4_1d(i as f32 * scale))
            .collect();
        let mut out = Vec::with_capacity((INTER_TAB_SIZE * INTER_TAB_SIZE) as usize);
        for i in 0..INTER_TAB_SIZE as usize {
            for j in 0..INTER_TAB_SIZE as usize {
                let mut itab = [0i16; 64];
                let mut isum = 0i32;
                for k1 in 0..8 {
                    let vy = tab1[i][k1];
                    for k2 in 0..8 {
                        let v = vy * tab1[j][k2];
                        let q = round_f32(v * COEF_SCALE as f32).clamp(-32768, 32767) as i16;
                        itab[k1 * 8 + k2] = q;
                        isum += i32::from(q);
                    }
                }
                if isum != COEF_SCALE {
                    let diff = isum - COEF_SCALE;
                    let (mut mk1, mut mk2, mut big1, mut big2) = (4, 4, 4, 4);
                    for k1 in 4..6 {
                        for k2 in 4..6 {
                            if itab[k1 * 8 + k2] < itab[mk1 * 8 + mk2] {
                                mk1 = k1;
                                mk2 = k2;
                            } else if itab[k1 * 8 + k2] > itab[big1 * 8 + big2] {
                                big1 = k1;
                                big2 = k2;
                            }
                        }
                    }
                    if diff < 0 {
                        itab[big1 * 8 + big2] = (i32::from(itab[big1 * 8 + big2]) - diff) as i16;
                    } else {
                        itab[mk1 * 8 + mk2] = (i32::from(itab[mk1 * 8 + mk2]) - diff) as i16;
                    }
                }
                out.push(itab);
            }
        }
        out
    })
}

/// `borderInterpolate(p, len, BORDER_CONSTANT)`: the index, or -1 outside.
fn inside(p: i32, len: i32) -> i32 {
    if (0..len).contains(&p) { p } else { -1 }
}

/// `cv::remap(src, dst, map_x, map_y, INTER_LANCZOS4, BORDER_CONSTANT, 0)` for an 8-bit
/// image of `channels` (1 or 3) interleaved channels, `src_w × src_h`, and maps of
/// `map_w × map_h` (the output size). See the module documentation.
///
/// Panics when the buffer sizes do not match.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn remap_lanczos4_u8(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    channels: usize,
    map_x: &[f32],
    map_y: &[f32],
    map_w: usize,
    map_h: usize,
) -> Vec<u8> {
    assert_eq!(src.len(), src_w * src_h * channels, "source size");
    assert_eq!(map_x.len(), map_w * map_h, "map x size");
    assert_eq!(map_y.len(), map_w * map_h, "map y size");
    let tab = lanczos4_table();
    let mut dst = vec![0u8; map_w * map_h * channels];
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(map_h.max(1));
    let rows_per = map_h.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (chunk_i, chunk) in dst.chunks_mut(rows_per * map_w * channels).enumerate() {
            let y0 = chunk_i * rows_per;
            s.spawn(move || {
                let rows = chunk.len() / (map_w * channels);
                for dy in 0..rows {
                    for dx in 0..map_w {
                        let m = (y0 + dy) * map_w + dx;
                        let out = &mut chunk[(dy * map_w + dx) * channels..][..channels];
                        lanczos_pixel(src, src_w, src_h, channels, tab, map_x[m], map_y[m], out);
                    }
                }
            });
        }
    });
    dst
}

/// `initInterTab2D(INTER_LANCZOS4, false)`: the float table of the non-8-bit paths, each
/// weight the `float` product of the two 1-D weights, with no sum correction.
fn lanczos4_table_f32() -> &'static [[f32; 64]] {
    use std::sync::OnceLock;
    static TAB: OnceLock<Vec<[f32; 64]>> = OnceLock::new();
    TAB.get_or_init(|| {
        let scale = 1.0f32 / INTER_TAB_SIZE as f32;
        let tab1: Vec<[f32; 8]> = (0..INTER_TAB_SIZE)
            .map(|i| lanczos4_1d(i as f32 * scale))
            .collect();
        let mut out = Vec::with_capacity((INTER_TAB_SIZE * INTER_TAB_SIZE) as usize);
        for i in 0..INTER_TAB_SIZE as usize {
            for j in 0..INTER_TAB_SIZE as usize {
                out.push(std::array::from_fn(|k| tab1[i][k / 8] * tab1[j][k % 8]));
            }
        }
        out
    })
}

/// `cv::remap(src, dst, map_x, map_y, INTER_LANCZOS4, BORDER_CONSTANT, 0)` for a 16-bit image
/// (`CV_16UC1` or `CV_16UC3`), arguments as [`remap_lanczos4_u8`].
///
/// Ported from OpenCV 4.13 `modules/imgproc/src/imgwarp.cpp`, `remapLanczos4<Cast<float,
/// ushort>, float, 1>`: the same 1/32-pixel quantisation of the maps, the float table of
/// `initInterTab2D(INTER_LANCZOS4, false)`, the sum accumulated in `float` in OpenCV's order
/// (eight products added left to right per row on the inside path, one product at a time
/// on the border path), then `saturate_cast<ushort>` (rounded half to even, clamped).
///
/// Panics when the buffer sizes do not match.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn remap_lanczos4_u16(
    src: &[u16],
    src_w: usize,
    src_h: usize,
    channels: usize,
    map_x: &[f32],
    map_y: &[f32],
    map_w: usize,
    map_h: usize,
) -> Vec<u16> {
    assert_eq!(src.len(), src_w * src_h * channels, "source size");
    assert_eq!(map_x.len(), map_w * map_h, "map x size");
    assert_eq!(map_y.len(), map_w * map_h, "map y size");
    let tab = lanczos4_table_f32();
    let mut dst = vec![0u16; map_w * map_h * channels];
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(map_h.max(1));
    let rows_per = map_h.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (chunk_i, chunk) in dst.chunks_mut(rows_per * map_w * channels).enumerate() {
            let y0 = chunk_i * rows_per;
            s.spawn(move || {
                let rows = chunk.len() / (map_w * channels);
                for dy in 0..rows {
                    for dx in 0..map_w {
                        let m = (y0 + dy) * map_w + dx;
                        let out = &mut chunk[(dy * map_w + dx) * channels..][..channels];
                        lanczos_pixel_u16(
                            src, src_w, src_h, channels, tab, map_x[m], map_y[m], out,
                        );
                    }
                }
            });
        }
    });
    dst
}

/// One output pixel of `remapLanczos4<Cast<float, ushort>, float, 1>`.
#[allow(clippy::too_many_arguments)]
fn lanczos_pixel_u16(
    src: &[u16],
    src_w: usize,
    src_h: usize,
    cn: usize,
    tab: &[[f32; 64]],
    fx: f32,
    fy: f32,
    out: &mut [u16],
) {
    let sxq = round_f32(fx * INTER_TAB_SIZE as f32);
    let syq = round_f32(fy * INTER_TAB_SIZE as f32);
    let a = ((syq & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (sxq & (INTER_TAB_SIZE - 1))) as usize;
    let xi = i32::from((sxq >> INTER_BITS).clamp(-32768, 32767) as i16);
    let yi = i32::from((syq >> INTER_BITS).clamp(-32768, 32767) as i16);
    let (sx, sy) = (xi - 3, yi - 3);
    let w = &tab[a];
    let (sw, shh) = (src_w as i32, src_h as i32);
    let cast = |sum: f32| round_f32(sum).clamp(0, 65535) as u16;
    let width1 = (sw - 7).max(0) as u32;
    let height1 = (shh - 7).max(0) as u32;
    if (sx as u32) < width1 && (sy as u32) < height1 {
        for (k, o) in out.iter_mut().enumerate() {
            let mut sum = 0f32;
            for r in 0..8 {
                let row = ((sy + r) as usize * src_w + sx as usize) * cn + k;
                let s = |c: usize| f32::from(src[row + c * cn]);
                let wr = &w[r as usize * 8..r as usize * 8 + 8];
                sum += s(0) * wr[0]
                    + s(1) * wr[1]
                    + s(2) * wr[2]
                    + s(3) * wr[3]
                    + s(4) * wr[4]
                    + s(5) * wr[5]
                    + s(6) * wr[6]
                    + s(7) * wr[7];
            }
            *o = cast(sum);
        }
        return;
    }
    if sx >= sw || sx + 8 <= 0 || sy >= shh || sy + 8 <= 0 {
        out.fill(0);
        return;
    }
    let xs: [i32; 8] = std::array::from_fn(|i| inside(sx + i as i32, sw));
    let ys: [i32; 8] = std::array::from_fn(|i| inside(sy + i as i32, shh));
    for (k, o) in out.iter_mut().enumerate() {
        // cv = 0: sum = cv * 1 + sum((S - cv) * w) over the taps inside.
        let mut sum = 0f32;
        for (r, &yr) in ys.iter().enumerate() {
            if yr < 0 {
                continue;
            }
            for (c, &xc) in xs.iter().enumerate() {
                if xc >= 0 {
                    let s = f32::from(src[(yr as usize * src_w + xc as usize) * cn + k]);
                    sum += (s - 0.0) * w[r * 8 + c];
                }
            }
        }
        *o = cast(sum);
    }
}

/// One output pixel of `remapLanczos4`.
#[allow(clippy::too_many_arguments)]
fn lanczos_pixel(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    cn: usize,
    tab: &[[i16; 64]],
    fx: f32,
    fy: f32,
    out: &mut [u8],
) {
    let sxq = round_f32(fx * INTER_TAB_SIZE as f32);
    let syq = round_f32(fy * INTER_TAB_SIZE as f32);
    let a = ((syq & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (sxq & (INTER_TAB_SIZE - 1))) as usize;
    let xi = i32::from((sxq >> INTER_BITS).clamp(-32768, 32767) as i16);
    let yi = i32::from((syq >> INTER_BITS).clamp(-32768, 32767) as i16);
    let (sx, sy) = (xi - 3, yi - 3);
    let w = &tab[a];
    let (sw, shh) = (src_w as i32, src_h as i32);
    let cast = |sum: i32| ((sum + (1 << (COEF_BITS - 1))) >> COEF_BITS).clamp(0, 255) as u8;
    let width1 = (sw - 7).max(0) as u32;
    let height1 = (shh - 7).max(0) as u32;
    if (sx as u32) < width1 && (sy as u32) < height1 {
        for (k, o) in out.iter_mut().enumerate() {
            let mut sum = 0i32;
            for r in 0..8 {
                let row = ((sy + r) as usize * src_w + sx as usize) * cn + k;
                for c in 0..8 {
                    sum += i32::from(src[row + c * cn]) * i32::from(w[r as usize * 8 + c]);
                }
            }
            *o = cast(sum);
        }
        return;
    }
    if sx >= sw || sx + 8 <= 0 || sy >= shh || sy + 8 <= 0 {
        out.fill(0);
        return;
    }
    let xs: [i32; 8] = std::array::from_fn(|i| inside(sx + i as i32, sw));
    let ys: [i32; 8] = std::array::from_fn(|i| inside(sy + i as i32, shh));
    for (k, o) in out.iter_mut().enumerate() {
        // cv = 0: sum = cv * ONE + sum((S - cv) * w) over the taps inside.
        let mut sum = 0i32;
        for (r, &yr) in ys.iter().enumerate() {
            if yr < 0 {
                continue;
            }
            for (c, &xc) in xs.iter().enumerate() {
                if xc >= 0 {
                    let s = src[(yr as usize * src_w + xc as usize) * cn + k];
                    sum += i32::from(s) * i32::from(w[r * 8 + c]);
                }
            }
        }
        *o = cast(sum);
    }
}

/// Pixels per block of the vector path of `initUndistortRectifyMapComputer`: two vectors of
/// four doubles (AVX2, the build every x86-64 processor since 2013 dispatches to).
const LENS_BLOCK: usize = 8;

/// The maps of `cv::initUndistortRectifyMap(K, dist, Mat(), K, size, CV_32FC1)`: for each
/// pixel of the undistorted image, where to sample the image with lens distortion. `k` is
/// the camera matrix (row-major 3 × 3, skew allowed), `d` the eight coefficients k1, k2,
/// p1, p2, k3, k4, k5, k6 (XMALab always passes eight, `core/Camera.cpp`).
///
/// Ported from OpenCV 4.13 `modules/calib3d/src/undistort.dispatch.cpp`,
/// `initUndistortRectifyMap` (`iR = K.inv(DECOMP_LU)`, the 3 × 3 closed form of
/// [`crate::linalg::invert`]), and `undistort.simd.hpp`, `initUndistortRectifyMapComputer`,
/// as OpenCV runs it on a processor with AVX2: blocks of eight pixels on the vector path
/// (fused multiply-adds where it uses `v_muladd`, the block's start advanced by
/// `8 * iR[0]`), the remaining pixels of a row on the scalar path. `R` is the identity and
/// the thin-prism and tilt terms are zero, as XMALab passes them. Equal to OpenCV 4.13 on
/// the four `initundistortrectifymap-*` cases and the two lens-model cases of
/// `oracle/py/gen_primitives_undistort.py` (see the module documentation).
#[must_use]
pub fn lens_undistortion_maps(
    k: &[[f64; 3]; 3],
    d: &[f64; 8],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (fx, fy, u0, v0) = (k[0][0], k[1][1], k[0][2], k[1][2]);
    let km = nalgebra::Matrix3::from_fn(|r, c| k[r][c]);
    let inv = crate::linalg::invert(&km).x;
    let ir: [f64; 9] = std::array::from_fn(|i| inv[(i / 3, i % 3)]);
    let [k1, k2, p1, p2, k3, k4, k5, k6] = *d;
    let fma = f64::mul_add;
    let s_x: [f64; LENS_BLOCK] = std::array::from_fn(|i| ir[0] * i as f64);
    let s_y: [f64; LENS_BLOCK] = std::array::from_fn(|i| ir[3] * i as f64);
    let s_w: [f64; LENS_BLOCK] = std::array::from_fn(|i| ir[6] * i as f64);
    let step = LENS_BLOCK as f64;
    let mut mx = vec![0f32; width * height];
    let mut my = vec![0f32; width * height];
    for i in 0..height {
        let fi = i as f64;
        let mut cx = fi * ir[1] + ir[2];
        let mut cy = fi * ir[4] + ir[5];
        let mut cw = fi * ir[7] + ir[8];
        let row = i * width;
        let mut j = 0;
        while j + LENS_BLOCK <= width {
            for l in 0..LENS_BLOCK {
                let w = 1.0 / (cw + s_w[l]);
                let x = (cx + s_x[l]) * w;
                let y = (cy + s_y[l]) * w;
                let (x2, y2) = (x * x, y * y);
                let r2 = x2 + y2;
                let num = fma(fma(fma(k3, r2, k2), r2, k1), r2, 1.0);
                let kr = num / fma(fma(fma(k6, r2, k5), r2, k4), r2, 1.0);
                let xd2 = fma(2.0, x2, r2);
                let yd2 = fma(2.0, y2, r2);
                let two_xy = (x * y) * 2.0;
                let (xk, yk) = (x * kr, y * kr);
                let xd = fma(p1, two_xy, fma(xd2, p2, xk));
                let yd = fma(p2, two_xy, fma(yd2, p1, yk));
                // The thin-prism terms (zero), then the tilt matrix (the identity) and its
                // projection (1), kept as OpenCV evaluates them.
                let r4 = r2 * r2;
                let xd = fma(0.0, r2, fma(0.0, r4, xd));
                let yd = fma(0.0, r2, fma(0.0, r4, yd));
                let tx = fma(1.0, xd, fma(0.0, yd, 0.0));
                let ty = fma(0.0, xd, fma(1.0, yd, 0.0));
                let tz = fma(0.0, xd, fma(0.0, yd, 1.0));
                let inv_proj = if tz == 0.0 { 1.0 } else { 1.0 / tz };
                mx[row + j + l] = fma(fx * inv_proj, tx, u0) as f32;
                my[row + j + l] = fma(fy * inv_proj, ty, v0) as f32;
            }
            j += LENS_BLOCK;
            cx += step * ir[0];
            cy += step * ir[3];
            cw += step * ir[6];
        }
        while j < width {
            let w = 1.0 / cw;
            let (x, y) = (cx * w, cy * w);
            let (x2, y2) = (x * x, y * y);
            let r2 = x2 + y2;
            let two_xy = 2.0 * x * y;
            let kr =
                (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2);
            let xd = x * kr + p1 * two_xy + p2 * (r2 + 2.0 * x2);
            let yd = y * kr + p1 * (r2 + 2.0 * y2) + p2 * two_xy;
            mx[row + j] = (fx * xd + u0) as f32;
            my[row + j] = (fy * yd + v0) as f32;
            j += 1;
            cx += ir[0];
            cy += ir[3];
            cw += ir[6];
        }
    }
    (mx, my)
}

/// The false-colour images of the Undistortion workspace (`undistortionVisImage_state`
/// 2 to 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DisplacementMap {
    /// "Angle scaled": the angle in `COLORMAP_HSV`, darkened by the length.
    ScaledAngle,
    /// Angle of the displacement, `COLORMAP_HSV`.
    Angle,
    /// Length of the displacement, `COLORMAP_JET`.
    Length,
    /// Horizontal displacement, `COLORMAP_JET`.
    Horizontal,
    /// Vertical displacement, `COLORMAP_JET`.
    Vertical,
}

/// `convertScaleAbs(v - min, 255 / (max - min))` of a float plane into 8 bits.
fn scale_abs(v: &[f32]) -> Vec<u8> {
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in v {
        min = min.min(x);
        max = max.max(x);
    }
    let alpha = (255.0 / (f64::from(max) - f64::from(min))) as f32;
    v.iter()
        .map(|&x| {
            let t = ((x - min) * alpha).abs();
            if t.is_nan() {
                0
            } else {
                t.round_ties_even().clamp(0.0, 255.0) as u8
            }
        })
        .collect()
}

/// The displacement image for a lookup table (`map_x`, `map_y`, -1 where uncovered), as
/// RGBA (alpha 255).
///
/// Ported from `core/UndistortionObject.cpp`, `getDisplacementScaledAngle`,
/// `getDisplacementAngle`, `getDisplacementLength`, `getDisplacementX`,
/// `getDisplacementY`: the displacement of each pixel `(x, y)` is `(map_x - x, map_y - y)`
/// (0 where the map is -1), scaled from its minimum to its maximum onto 0 to 255 and
/// coloured.
#[must_use]
pub fn displacement_image(
    map_x: &[f32],
    map_y: &[f32],
    width: usize,
    height: usize,
    kind: DisplacementMap,
) -> Vec<u8> {
    let n = width * height;
    let mut a = vec![0f32; n];
    let mut l = vec![0f32; n];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let (mx, my) = (map_x[i], map_y[i]);
            let (dx, dy) = (mx - x as f32, my - y as f32);
            match kind {
                DisplacementMap::Horizontal => a[i] = if mx != -1.0 { dx } else { 0.0 },
                DisplacementMap::Vertical => a[i] = if my != -1.0 { dy } else { 0.0 },
                _ => {
                    if mx != -1.0 && my != -1.0 {
                        a[i] = dy.atan2(dx);
                        l[i] = (dy * dy + dx * dx).sqrt();
                    }
                }
            }
        }
    }
    let lut = match kind {
        DisplacementMap::ScaledAngle | DisplacementMap::Angle => &HSV,
        _ => &JET,
    };
    let plane = match kind {
        DisplacementMap::Length => scale_abs(&l),
        _ => scale_abs(&a),
    };
    let mut rgba = vec![255u8; n * 4];
    for (i, &v) in plane.iter().enumerate() {
        rgba[i * 4..i * 4 + 3].copy_from_slice(&lut[v as usize]);
    }
    if kind == DisplacementMap::ScaledAngle {
        // cv::multiply(gray(length), colour, colour, 1 / 255).
        let len = scale_abs(&l);
        let s = (1.0f64 / 255.0) as f32;
        for (i, &g) in len.iter().enumerate() {
            for c in 0..3 {
                let v = f32::from(rgba[i * 4 + c]);
                rgba[i * 4 + c] = (s * f32::from(g) * v).round_ties_even().clamp(0.0, 255.0) as u8;
            }
        }
    }
    rgba
}

/// OpenCV `COLORMAP_JET` as `applyColorMap` gives it for grey levels 0 to 255, in R, G, B
/// order (taken from OpenCV 4.14).
#[rustfmt::skip]
pub const JET: [[u8; 3]; 256] = [[0, 0, 128], [0, 0, 132], [0, 0, 136], [0, 0, 140], [0, 0, 144], [0, 0, 148], [0, 0, 152], [0, 0, 156], [0, 0, 160], [0, 0, 164], [0, 0, 168], [0, 0, 172], [0, 0, 176], [0, 0, 180], [0, 0, 184], [0, 0, 188], [0, 0, 192], [0, 0, 196], [0, 0, 200], [0, 0, 204], [0, 0, 208], [0, 0, 212], [0, 0, 216], [0, 0, 220], [0, 0, 224], [0, 0, 228], [0, 0, 232], [0, 0, 236], [0, 0, 240], [0, 0, 244], [0, 0, 248], [0, 0, 252], [0, 0, 255], [0, 4, 255], [0, 8, 255], [0, 12, 255], [0, 16, 255], [0, 20, 255], [0, 24, 255], [0, 28, 255], [0, 32, 255], [0, 36, 255], [0, 40, 255], [0, 44, 255], [0, 48, 255], [0, 52, 255], [0, 56, 255], [0, 60, 255], [0, 64, 255], [0, 68, 255], [0, 72, 255], [0, 76, 255], [0, 80, 255], [0, 84, 255], [0, 88, 255], [0, 92, 255], [0, 96, 255], [0, 100, 255], [0, 104, 255], [0, 108, 255], [0, 112, 255], [0, 116, 255], [0, 120, 255], [0, 124, 255], [0, 128, 255], [0, 132, 255], [0, 136, 255], [0, 140, 255], [0, 144, 255], [0, 148, 255], [0, 152, 255], [0, 156, 255], [0, 160, 255], [0, 164, 255], [0, 168, 255], [0, 172, 255], [0, 176, 255], [0, 180, 255], [0, 184, 255], [0, 188, 255], [0, 192, 255], [0, 196, 255], [0, 200, 255], [0, 204, 255], [0, 208, 255], [0, 212, 255], [0, 216, 255], [0, 220, 255], [0, 224, 255], [0, 228, 255], [0, 232, 255], [0, 236, 255], [0, 240, 255], [0, 244, 255], [0, 248, 255], [0, 252, 255], [2, 255, 254], [6, 255, 250], [10, 255, 246], [14, 255, 242], [18, 255, 238], [22, 255, 234], [26, 255, 230], [30, 255, 226], [34, 255, 222], [38, 255, 218], [42, 255, 214], [46, 255, 210], [50, 255, 206], [54, 255, 202], [58, 255, 198], [62, 255, 194], [66, 255, 190], [70, 255, 186], [74, 255, 182], [78, 255, 178], [82, 255, 174], [86, 255, 170], [90, 255, 166], [94, 255, 162], [98, 255, 158], [102, 255, 154], [106, 255, 150], [110, 255, 146], [114, 255, 142], [118, 255, 138], [122, 255, 134], [126, 255, 130], [130, 255, 126], [134, 255, 122], [138, 255, 118], [142, 255, 114], [146, 255, 110], [150, 255, 106], [154, 255, 102], [158, 255, 98], [162, 255, 94], [166, 255, 90], [170, 255, 86], [174, 255, 82], [178, 255, 78], [182, 255, 74], [186, 255, 70], [190, 255, 66], [194, 255, 62], [198, 255, 58], [202, 255, 54], [206, 255, 50], [210, 255, 46], [214, 255, 42], [218, 255, 38], [222, 255, 34], [226, 255, 30], [230, 255, 26], [234, 255, 22], [238, 255, 18], [242, 255, 14], [246, 255, 10], [250, 255, 6], [254, 255, 1], [255, 252, 0], [255, 248, 0], [255, 244, 0], [255, 240, 0], [255, 236, 0], [255, 232, 0], [255, 228, 0], [255, 224, 0], [255, 220, 0], [255, 216, 0], [255, 212, 0], [255, 208, 0], [255, 204, 0], [255, 200, 0], [255, 196, 0], [255, 192, 0], [255, 188, 0], [255, 184, 0], [255, 180, 0], [255, 176, 0], [255, 172, 0], [255, 168, 0], [255, 164, 0], [255, 160, 0], [255, 156, 0], [255, 152, 0], [255, 148, 0], [255, 144, 0], [255, 140, 0], [255, 136, 0], [255, 132, 0], [255, 128, 0], [255, 124, 0], [255, 120, 0], [255, 116, 0], [255, 112, 0], [255, 108, 0], [255, 104, 0], [255, 100, 0], [255, 96, 0], [255, 92, 0], [255, 88, 0], [255, 84, 0], [255, 80, 0], [255, 76, 0], [255, 72, 0], [255, 68, 0], [255, 64, 0], [255, 60, 0], [255, 56, 0], [255, 52, 0], [255, 48, 0], [255, 44, 0], [255, 40, 0], [255, 36, 0], [255, 32, 0], [255, 28, 0], [255, 24, 0], [255, 20, 0], [255, 16, 0], [255, 12, 0], [255, 8, 0], [255, 4, 0], [255, 0, 0], [252, 0, 0], [248, 0, 0], [244, 0, 0], [240, 0, 0], [236, 0, 0], [232, 0, 0], [228, 0, 0], [224, 0, 0], [220, 0, 0], [216, 0, 0], [212, 0, 0], [208, 0, 0], [204, 0, 0], [200, 0, 0], [196, 0, 0], [192, 0, 0], [188, 0, 0], [184, 0, 0], [180, 0, 0], [176, 0, 0], [172, 0, 0], [168, 0, 0], [164, 0, 0], [160, 0, 0], [156, 0, 0], [152, 0, 0], [148, 0, 0], [144, 0, 0], [140, 0, 0], [136, 0, 0], [132, 0, 0], [128, 0, 0]];

/// OpenCV `COLORMAP_HSV` as `applyColorMap` gives it for grey levels 0 to 255, in R, G, B
/// order (taken from OpenCV 4.14).
#[rustfmt::skip]
pub const HSV: [[u8; 3]; 256] = [[255, 0, 0], [255, 6, 0], [255, 12, 0], [255, 18, 0], [255, 24, 0], [255, 30, 0], [255, 36, 0], [255, 42, 0], [255, 48, 0], [255, 54, 0], [255, 60, 0], [255, 66, 0], [255, 72, 0], [255, 78, 0], [255, 84, 0], [255, 90, 0], [255, 96, 0], [255, 102, 0], [255, 108, 0], [255, 114, 0], [255, 120, 0], [255, 126, 0], [255, 132, 0], [255, 138, 0], [255, 144, 0], [255, 150, 0], [255, 156, 0], [255, 162, 0], [255, 168, 0], [255, 174, 0], [255, 180, 0], [255, 186, 0], [255, 192, 0], [255, 198, 0], [255, 204, 0], [255, 210, 0], [255, 216, 0], [255, 222, 0], [255, 228, 0], [255, 234, 0], [255, 240, 0], [253, 244, 0], [250, 247, 0], [247, 250, 0], [244, 253, 0], [240, 255, 0], [234, 255, 0], [228, 255, 0], [222, 255, 0], [216, 255, 0], [210, 255, 0], [204, 255, 0], [198, 255, 0], [192, 255, 0], [186, 255, 0], [180, 255, 0], [174, 255, 0], [168, 255, 0], [162, 255, 0], [156, 255, 0], [150, 255, 0], [144, 255, 0], [138, 255, 0], [132, 255, 0], [126, 255, 0], [120, 255, 0], [114, 255, 0], [108, 255, 0], [102, 255, 0], [96, 255, 0], [90, 255, 0], [84, 255, 0], [78, 255, 0], [72, 255, 0], [66, 255, 0], [60, 255, 0], [54, 255, 0], [48, 255, 0], [42, 255, 0], [36, 255, 0], [30, 255, 0], [24, 255, 0], [18, 255, 0], [12, 255, 0], [6, 255, 0], [0, 255, 0], [0, 255, 6], [0, 255, 12], [0, 255, 18], [0, 255, 24], [0, 255, 30], [0, 255, 36], [0, 255, 42], [0, 255, 48], [0, 255, 54], [0, 255, 60], [0, 255, 66], [0, 255, 72], [0, 255, 78], [0, 255, 84], [0, 255, 90], [0, 255, 96], [0, 255, 102], [0, 255, 108], [0, 255, 114], [0, 255, 120], [0, 255, 126], [0, 255, 132], [0, 255, 138], [0, 255, 144], [0, 255, 150], [0, 255, 156], [0, 255, 162], [0, 255, 168], [0, 255, 174], [0, 255, 180], [0, 255, 186], [0, 255, 192], [0, 255, 198], [0, 255, 204], [0, 255, 210], [0, 255, 216], [0, 255, 222], [0, 255, 228], [0, 255, 234], [0, 255, 240], [0, 253, 244], [0, 250, 247], [0, 247, 250], [0, 244, 253], [0, 240, 255], [0, 234, 255], [0, 228, 255], [0, 222, 255], [0, 216, 255], [0, 210, 255], [0, 204, 255], [0, 198, 255], [0, 192, 255], [0, 186, 255], [0, 180, 255], [0, 174, 255], [0, 168, 255], [0, 162, 255], [0, 156, 255], [0, 150, 255], [0, 144, 255], [0, 138, 255], [0, 132, 255], [0, 126, 255], [0, 120, 255], [0, 114, 255], [0, 108, 255], [0, 102, 255], [0, 96, 255], [0, 90, 255], [0, 84, 255], [0, 78, 255], [0, 72, 255], [0, 66, 255], [0, 60, 255], [0, 54, 255], [0, 48, 255], [0, 42, 255], [0, 36, 255], [0, 30, 255], [0, 24, 255], [0, 18, 255], [0, 12, 255], [0, 6, 255], [0, 0, 255], [6, 0, 255], [12, 0, 255], [18, 0, 255], [24, 0, 255], [30, 0, 255], [36, 0, 255], [42, 0, 255], [48, 0, 255], [54, 0, 255], [60, 0, 255], [66, 0, 255], [72, 0, 255], [78, 0, 255], [84, 0, 255], [90, 0, 255], [96, 0, 255], [102, 0, 255], [108, 0, 255], [114, 0, 255], [120, 0, 255], [126, 0, 255], [132, 0, 255], [138, 0, 255], [144, 0, 255], [150, 0, 255], [156, 0, 255], [162, 0, 255], [168, 0, 255], [174, 0, 255], [180, 0, 255], [186, 0, 255], [192, 0, 255], [198, 0, 255], [204, 0, 255], [210, 0, 255], [216, 0, 255], [222, 0, 255], [228, 0, 255], [234, 0, 255], [240, 0, 255], [244, 0, 253], [247, 0, 250], [250, 0, 247], [253, 0, 244], [255, 0, 240], [255, 0, 234], [255, 0, 228], [255, 0, 222], [255, 0, 216], [255, 0, 210], [255, 0, 204], [255, 0, 198], [255, 0, 192], [255, 0, 186], [255, 0, 180], [255, 0, 174], [255, 0, 168], [255, 0, 162], [255, 0, 156], [255, 0, 150], [255, 0, 144], [255, 0, 138], [255, 0, 132], [255, 0, 126], [255, 0, 120], [255, 0, 114], [255, 0, 108], [255, 0, 102], [255, 0, 96], [255, 0, 90], [255, 0, 84], [255, 0, 78], [255, 0, 72], [255, 0, 66], [255, 0, 60], [255, 0, 54], [255, 0, 48], [255, 0, 42], [255, 0, 36], [255, 0, 30], [255, 0, 24], [255, 0, 18], [255, 0, 12], [255, 0, 6], [255, 0, 0]];

#[cfg(test)]
mod tests {
    use super::*;

    fn case_u8(case: &xma_oracle::Case, key: &str) -> (Vec<u8>, usize, usize, usize) {
        let im = case.image(key);
        match im.data {
            xma_oracle::ImageData::U8(d) => (d, im.cols, im.rows, im.channels),
            _ => panic!("{key} is not u8"),
        }
    }

    fn case_f32(case: &xma_oracle::Case, key: &str) -> Vec<f32> {
        match case.image(key).data {
            xma_oracle::ImageData::F32(d) => d,
            _ => panic!("{key} is not f32"),
        }
    }

    #[test]
    fn remap_matches_opencv_primitives() {
        let cases = xma_oracle::cases("remap");
        assert!(!cases.is_empty(), "no remap cases");
        for case in cases {
            let (src, w, h, cn) = case_u8(&case, "src");
            let mx = case_f32(&case, "map_x");
            let my = case_f32(&case, "map_y");
            let (want, dw, dh, _) = case_u8(&case, "dst");
            let got = remap_lanczos4_u8(&src, w, h, cn, &mx, &my, dw, dh);
            let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "{}: {differ} values differ", case.name);
        }
    }

    #[test]
    fn table_rows_sum_to_one() {
        for t in lanczos4_table() {
            assert_eq!(t.iter().map(|&v| i32::from(v)).sum::<i32>(), COEF_SCALE);
        }
        // Fraction 0: the centre tap saturates at 32767 and OpenCV's sum correction adds
        // the missing unit to tap (4, 4).
        assert_eq!(lanczos4_table()[0][3 * 8 + 3], 32767);
        assert_eq!(lanczos4_table()[0][4 * 8 + 4], 1);
    }

    #[test]
    fn lens_maps_without_distortion_are_the_pixel_grid() {
        let k = [[800.0, 0.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]];
        let (mx, my) = lens_undistortion_maps(&k, &[0.0; 8], 64, 48);
        for y in 0..48 {
            for x in 0..64 {
                assert!((mx[y * 64 + x] - x as f32).abs() < 1e-3);
                assert!((my[y * 64 + x] - y as f32).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn displacement_maps_span_the_colour_table() {
        let (w, h) = (16usize, 8usize);
        let mut mx = vec![0f32; w * h];
        let mut my = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                mx[y * w + x] = x as f32 * 1.1;
                my[y * w + x] = y as f32;
            }
        }
        mx[0] = -1.0;
        let img = displacement_image(&mx, &my, w, h, DisplacementMap::Horizontal);
        // Smallest displacement (the uncovered pixel and x = 0 give 0) at JET[0], the
        // largest at JET[255].
        assert_eq!(&img[0..3], &JET[0]);
        assert_eq!(&img[(w - 1) * 4..(w - 1) * 4 + 3], &JET[255]);
        for kind in [
            DisplacementMap::ScaledAngle,
            DisplacementMap::Angle,
            DisplacementMap::Length,
            DisplacementMap::Vertical,
        ] {
            assert_eq!(displacement_image(&mx, &my, w, h, kind).len(), w * h * 4);
        }
    }

    /// The embedded colour tables equal OpenCV's (`oracle/py/display_colormaps.py`).
    #[test]
    fn colour_tables_match_opencv() {
        let path = xma_oracle::oracle_root().join("data/display/colormaps.csv");
        if !path.exists() {
            assert!(
                !xma_oracle::oracle_required(),
                "{} is missing: run oracle/py/display_colormaps.py",
                path.display()
            );
            eprintln!("SKIPPED: {} absent", path.display());
            return;
        }
        let t = xma_oracle::Table::read(&path);
        assert_eq!(t.len(), 512);
        for r in t.rows() {
            let table = if r.str("map") == "JET" { &JET } else { &HSV };
            let want = [r.i64("r"), r.i64("g"), r.i64("b")].map(|v| v as u8);
            assert_eq!(
                table[r.usize("level")],
                want,
                "{} {}",
                r.str("map"),
                r.usize("level")
            );
        }
    }
}
