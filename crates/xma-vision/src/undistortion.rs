//! Grid undistortion by local weighted mean.
//!
//! Work package: undistortion model. Ported from `processing/LocalUndistortion.cpp`
//! (`computeLWM`, `createLookupTable`, `localUndistortion_thread`) and
//! `core/UndistortionObject.cpp` (`setLWMMatrices`, `transformLWM`, `transformPoint`).
//! Bead detection and hexagonal grid matching are later packages.
//!
//! # The two directions
//!
//! `LocalUndistortion::computeLWM(detectedPts, controlPts, ...)` fits, for every point of
//! `controlPts`, a second-order polynomial in the coordinates of `controlPts` whose values
//! are the coordinates of `detectedPts`. The polynomials are therefore *centred on* the
//! second argument and *map to* the first. `localUndistortion_thread` calls it twice:
//!
//! 1. `computeLWM(distorted, references)`: centred on the reference (undistorted) lattice,
//!    mapping undistorted to distorted. The image lookup table is built from this fit
//!    (for every undistorted output pixel, where to sample the raw image), and
//!    `setLWMMatrices` stores it as the **distortion** model.
//! 2. `computeLWM(references, distorted)`: centred on the distorted grid points, mapping
//!    distorted to undistorted; stored as the **undistortion** model.
//!
//! # Threading
//!
//! The C++ runs the per-control-point fit and the per-row lookup table in
//! `cv::parallel_for_`. Each iteration writes only its own output, so the result does not
//! depend on the split; the port splits the same loops over `std::thread::scope` workers.
//! The tests check that with the hidden `fit_lwm_threads` and `create_lookup_table_threads`,
//! which take the thread count.

use std::ops::Range;
use std::sync::Arc;

use nalgebra::DMatrix;

use crate::Point2;
use crate::linalg;

/// Coordinate the C++ returns, in both x and y, for a point no control point covers.
pub const UNCOVERED: f64 = -1.0;

/// A local-weighted-mean mapping in one direction: a second-order polynomial per control
/// point, blended by distance.
///
/// The four lists are shared (`Arc<[_]>`): a copy of a transform, which every per-frame
/// context of the engine makes with its camera model, copies four pointers, not the fit.
/// They print (`Debug`) and compare (`==`, element by element, so a NaN is unequal to itself
/// as before) exactly as the `Vec`s they replace.
#[derive(Clone, Debug, PartialEq)]
pub struct LwmTransform {
    /// The points the polynomials are centred on (`controlPts` of `computeLWM`).
    pub control_points: Arc<[Point2]>,
    /// Polynomial coefficients for x at each control point, in the order
    /// 1, x, y, xy, x^2, y^2.
    pub a: Arc<[[f64; 6]]>,
    /// Polynomial coefficients for y, same order.
    pub b: Arc<[[f64; 6]]>,
    /// Radius of influence of each control point.
    pub radii: Arc<[f64]>,
    /// Which control points can reach each part of the plane, built on first use: a cache
    /// that makes [`transform_lwm`] visit only the control points near a point. Not part of
    /// the transform's value (equal for any two transforms); `LwmIndex::default()` when
    /// building a transform.
    pub index: LwmIndex,
}

/// A cache of [`LwmTransform`]: the control points whose disc of influence can meet each
/// cell of a regular grid, in index order. Built once on first use and shared by copies.
///
/// Not a port: [`transform_lwm`] adds the contributions of the control points in index
/// order, and a control point outside the candidates of a cell contributes nothing there,
/// so the sums and the result are bit for bit those of the full loop.
///
/// Equal to every other index, and printed (`Debug`) as `LwmIndex(..)` whether built or
/// not: it is not part of the transform's value, so a transform fitted again (whose index
/// is not built yet) prints and compares as the one it replaces.
#[derive(Clone, Default)]
pub struct LwmIndex(Arc<std::sync::OnceLock<Option<CellIndex>>>);

impl PartialEq for LwmIndex {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for LwmIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LwmIndex(..)")
    }
}

#[derive(Debug)]
struct CellIndex {
    min_x: f64,
    min_y: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    /// Candidates of each cell (row-major), indices ascending.
    cells: Vec<Vec<u32>>,
    /// Candidates of any point outside the grid: the control points whose radius is not a
    /// positive finite number (they can contribute anywhere).
    outside: Vec<u32>,
}

impl CellIndex {
    /// `None` when the transform has nothing to index or the grid would be too large; the
    /// full loop is used then.
    fn build(t: &LwmTransform) -> Option<CellIndex> {
        let n = t.control_points.len();
        if n < 32 || t.radii.len() != n || n > u32::MAX as usize {
            return None;
        }
        let local = |p: usize| {
            let r = t.radii[p];
            let c = t.control_points[p];
            r.is_finite() && r > 0.0 && c.x.is_finite() && c.y.is_finite()
        };
        // A disc's bounding box, widened so that rounding in the distance test can never
        // put a contributing point outside it.
        let reach = |p: usize| t.radii[p] * (1.0 + 1e-6) + 1e-6;
        let mut cell = 0.0f64;
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        let mut outside = Vec::new();
        for p in 0..n {
            if !local(p) {
                outside.push(p as u32);
                continue;
            }
            let (c, r) = (t.control_points[p], reach(p));
            cell = cell.max(r);
            x0 = x0.min(c.x - r);
            y0 = y0.min(c.y - r);
            x1 = x1.max(c.x + r);
            y1 = y1.max(c.y + r);
        }
        if cell <= 0.0 || x1 < x0 {
            return None;
        }
        let nx = ((x1 - x0) / cell).floor() as usize + 1;
        let ny = ((y1 - y0) / cell).floor() as usize + 1;
        if nx.saturating_mul(ny) > 1 << 20 {
            return None;
        }
        let mut cells: Vec<Vec<u32>> = vec![Vec::new(); nx * ny];
        let at = |v: f64, min: f64, len: usize| {
            (((v - min) / cell).floor().max(0.0) as usize).min(len - 1)
        };
        for p in 0..n {
            if !local(p) {
                for list in &mut cells {
                    list.push(p as u32);
                }
                continue;
            }
            let (c, r) = (t.control_points[p], reach(p));
            for j in at(c.y - r, y0, ny)..=at(c.y + r, y0, ny) {
                for i in at(c.x - r, x0, nx)..=at(c.x + r, x0, nx) {
                    cells[j * nx + i].push(p as u32);
                }
            }
        }
        Some(CellIndex {
            min_x: x0,
            min_y: y0,
            cell,
            nx,
            ny,
            cells,
            outside,
        })
    }

    /// The candidates for a point; `None` for a point that is not finite.
    fn candidates(&self, x: f64, y: f64) -> Option<&[u32]> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let fx = ((x - self.min_x) / self.cell).floor();
        let fy = ((y - self.min_y) / self.cell).floor();
        if fx < 0.0 || fy < 0.0 || fx >= self.nx as f64 || fy >= self.ny as f64 {
            return Some(&self.outside);
        }
        Some(&self.cells[fy as usize * self.nx + fx as usize])
    }
}

impl LwmTransform {
    /// The lookup-table cell size: `ceil` of the largest radius, as an `int`.
    ///
    /// Ported from `processing/LocalUndistortion.cpp`, `LocalUndistortion::computeLWM`
    /// (its return value): `maxRadius` starts at 0 and takes every radius greater than it.
    #[must_use]
    pub fn grid_size(&self) -> i32 {
        let mut max_radius = 0.0_f64;
        for &r in self.radii.iter() {
            if r > max_radius {
                max_radius = r;
            }
        }
        // static_cast<int>(std::ceil(maxRadius)); Rust's `as` saturates where C++ is UB.
        max_radius.ceil() as i32
    }
}

/// The fitted grid model of one camera: both directions.
#[derive(Clone, Debug, PartialEq)]
pub struct GridModel {
    /// Raw image to undistorted image: centred on the distorted grid points, mapping to
    /// the reference points (`UndistortionObject::undistortionMatrix*`).
    pub undistort: LwmTransform,
    /// Undistorted image to raw image: centred on the reference points, mapping to the
    /// distorted grid points (`UndistortionObject::distortionMatrix*`). The image lookup
    /// table is built from this one.
    pub distort: LwmTransform,
}

/// The two single-precision maps (`CV_32FC1`) of `createLookupTable`: for each pixel of the
/// undistorted image, the raw-image coordinates to sample, or -1 where no control point
/// covers the pixel. Row-major, `height` rows of `width`.
#[derive(Clone, Debug, PartialEq)]
pub struct LookupTable {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// x coordinates, `map_x[y * width + x]`.
    pub map_x: Vec<f32>,
    /// y coordinates, same layout.
    pub map_y: Vec<f32>,
}

/// Number of worker threads for the parallel loops.
fn worker_count(work: usize) -> usize {
    let n = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    n.clamp(1, work.max(1))
}

/// Splits `0..len` into `parts` contiguous ranges.
fn split_range(len: usize, parts: usize) -> Vec<Range<usize>> {
    let parts = parts.max(1);
    let chunk = len.div_ceil(parts).max(1);
    (0..len)
        .step_by(chunk)
        .map(|s| s..(s + chunk).min(len))
        .collect()
}

/// Fits a local-weighted-mean mapping.
///
/// Ported from `processing/LocalUndistortion.cpp`, `LocalUndistortion::computeLWM`
/// (`detectedPts`, `controlPts`). The polynomials are centred on `control_points` and map
/// them to `detected_points` (same index, same length). For each control point `i`:
///
/// - the Euclidean distances from it to every control point (itself included) are sorted
///   with `cv::sortIdx` ([`linalg::sort_idx`], so equal distances keep OpenCV's order);
/// - its radius is the distance of the `neighbours`-th closest (counting itself);
/// - the `6 x neighbours` matrix of monomials 1, x, y, xy, x^2, y^2 of those neighbours is
///   pseudo-inverted with `cv::invert(.., DECOMP_SVD)` ([`linalg::invert_svd`]), and the
///   coefficients are the neighbours' target coordinates times that pseudo-inverse
///   (`cv::gemm`'s plain sequential dot product).
///
/// `neighbours` is the `LocalUndistortionNeighbours` setting (default 12). For
/// `neighbours >= 25` the reference OpenCV uses LAPACK for the pseudo-inverse and agrees
/// only to rounding (measured at 25 on a 10 x 10 lattice: coefficients within 1.9e-11,
/// transformed points within 1.5e-11 pixels; the radii are identical).
///
/// Returns `None` for empty input, and also for `neighbours == 0` or fewer control points
/// than `neighbours`, where the C++ reads past the end of its index array (undefined
/// behaviour). For empty input the C++ releases `A`, `B` and the radii and returns -1 as
/// the grid size; its callers carry on with those empty matrices, which behave like a
/// transform with no control points (every point uncovered, an all -1 lookup table). See
/// [`fit_grid_model`] for what that means when a grid has no inliers.
///
/// Panics if the two slices differ in length. Tier A.
/// Oracle: `undistortion/samples.csv` (through [`transform_point`]) and the differential
/// fixtures in `tests/data/undistortion/`.
#[must_use]
pub fn fit_lwm(
    detected_points: &[Point2],
    control_points: &[Point2],
    neighbours: usize,
) -> Option<LwmTransform> {
    fit_lwm_threads(
        detected_points,
        control_points,
        neighbours,
        worker_count(control_points.len()),
    )
}

#[doc(hidden)]
/// [`fit_lwm`] with an explicit number of worker threads (at least 1), so that the tests can
/// show that the result does not depend on it. Not part of the interface.
#[must_use]
pub fn fit_lwm_threads(
    detected_points: &[Point2],
    control_points: &[Point2],
    neighbours: usize,
    threads: usize,
) -> Option<LwmTransform> {
    if detected_points.is_empty() || control_points.is_empty() {
        return None;
    }
    assert_eq!(
        detected_points.len(),
        control_points.len(),
        "fit_lwm: detected and control point lists differ in length"
    );
    let n = control_points.len();
    if neighbours == 0 || n < neighbours {
        return None;
    }

    let ranges = split_range(n, threads.clamp(1, n));
    let parts: Vec<Vec<LocalFit>> = std::thread::scope(|s| {
        let handles: Vec<_> = ranges
            .into_iter()
            .map(|range| {
                s.spawn(move || {
                    range
                        .map(|i| fit_one(detected_points, control_points, neighbours, i))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("fit_lwm worker panicked"))
            .collect()
    });

    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut radii = Vec::with_capacity(n);
    for (ai, bi, ri) in parts.into_iter().flatten() {
        a.push(ai);
        b.push(bi);
        radii.push(ri);
    }
    Some(LwmTransform {
        control_points: control_points.into(),
        a: a.into(),
        b: b.into(),
        radii: radii.into(),
        index: LwmIndex::default(),
    })
}

/// Coefficients for x, for y, and the radius of one control point.
type LocalFit = ([f64; 6], [f64; 6], f64);

/// One iteration of the `parallel_for_` body of `computeLWM`: coefficients and radius of
/// control point `i`.
///
/// Ported from `processing/LocalUndistortion.cpp`, `LocalUndistortion::computeLWM`.
fn fit_one(detected: &[Point2], control: &[Point2], neighbours: usize, i: usize) -> LocalFit {
    // tmpXY = controlPts - Scalar(curr_x, curr_y); split; multiply; add; sqrt.
    let curr_x = control[i].x;
    let curr_y = control[i].y;
    let dist: Vec<f64> = control
        .iter()
        .map(|p| {
            let dx = p.x - curr_x;
            let dy = p.y - curr_y;
            (dx * dx + dy * dy).sqrt()
        })
        .collect();
    // cv::sortIdx(.., SORT_EVERY_COLUMN + SORT_ASCENDING) of an N x 1 column.
    let idx = linalg::sort_idx(&dist, false);
    let radius = dist[idx[neighbours - 1]];

    let mut x = DMatrix::<f64>::zeros(6, neighbours);
    let mut ucp = vec![0.0; neighbours];
    let mut vcp = vec![0.0; neighbours];
    for j in 0..neighbours {
        let k = idx[j];
        let xcp = control[k].x;
        let ycp = control[k].y;
        x[(0, j)] = 1.0;
        x[(1, j)] = xcp;
        x[(2, j)] = ycp;
        x[(3, j)] = xcp * ycp;
        x[(4, j)] = xcp * xcp;
        x[(5, j)] = ycp * ycp;
        ucp[j] = detected[k].x;
        vcp[j] = detected[k].y;
    }

    // cv::invert(X, X_inv, DECOMP_SVD): X_inv is neighbours x 6.
    let (x_inv, _) = linalg::invert_svd(&x);
    (row_times(&ucp, &x_inv), row_times(&vcp, &x_inv), radius)
}

/// `ucp * X_inv` for a `1 x n` row and an `n x 6` matrix, as OpenCV's `GEMMSingleMul`
/// computes it for these shapes: each element a sequential sum from 0 (`alpha` = 1 is
/// exact). For `n == 1` OpenCV takes its "external product" path, `a * 1 * b`.
///
/// Ported from `modules/core/src/matmul.simd.hpp`, `GEMMSingleMul` (reached through
/// `cv::gemm` from `MatOp_GEMM::assign`; `m = 1` is below the LAPACK threshold).
fn row_times(row: &[f64], m: &DMatrix<f64>) -> [f64; 6] {
    let n = row.len();
    let mut out = [0.0; 6];
    for (j, o) in out.iter_mut().enumerate() {
        if n == 1 {
            *o = row[0] * 1.0 * m[(0, j)];
        } else {
            let mut s = 0.0;
            for k in 0..n {
                s += row[k] * m[(k, j)];
            }
            *o = s * 1.0;
        }
    }
    out
}

/// Fits both directions from a matched grid.
///
/// Ported from `processing/LocalUndistortion.cpp`, `LocalUndistortion::localUndistortion_thread`
/// (the two calls to `computeLWM`, with `setPointsByInlier` before them) and
/// `core/UndistortionObject.cpp`, `UndistortionObject::setLWMMatrices` (which fit is which):
/// [`GridModel::distort`] is `computeLWM(distorted, references)`, centred on the references;
/// [`GridModel::undistort`] is `computeLWM(references, distorted)`, centred on the distorted
/// points. Only points with `inlier` set take part, in their original order. Returns `None`
/// when either fit does (no inliers, or fewer than `neighbours`). Panics if the three
/// slices differ in length. Tier A.
///
/// # No inliers
///
/// When every grid point is an outlier the C++ does not stop
/// (`processing/LocalUndistortion.cpp:113-122,247-252`): both `computeLWM` calls see empty
/// matrices, release `A`, `B` and the radii and return -1; `createLookupTable` with a grid
/// size of -1 makes no cells and fills both maps with -1 (as [`create_lookup_table`] does for
/// an empty transform); the undistortion object is still marked computed and stores the
/// empty matrices; and `transformPoint` then finds no control point, so it returns every
/// point unchanged in both directions. This function returns `None` instead, so a caller
/// that must reproduce that state sets it up itself: an all -1 lookup table and an identity
/// point transform, with the camera marked as undistorted.
#[must_use]
pub fn fit_grid_model(
    grid_distorted: &[Point2],
    grid_references: &[Point2],
    inlier: &[bool],
    neighbours: usize,
) -> Option<GridModel> {
    assert!(
        grid_distorted.len() == inlier.len() && grid_references.len() == inlier.len(),
        "fit_grid_model: grid point lists and inlier flags differ in length"
    );
    let pick = |pts: &[Point2]| -> Vec<Point2> {
        pts.iter()
            .zip(inlier)
            .filter(|&(_, &keep)| keep)
            .map(|(p, _)| *p)
            .collect()
    };
    let control_pts_inverse = pick(grid_distorted);
    let control_pts = pick(grid_references);
    let distort = fit_lwm(&control_pts_inverse, &control_pts, neighbours)?;
    let undistort = fit_lwm(&control_pts, &control_pts_inverse, neighbours)?;
    Some(GridModel { undistort, distort })
}

/// Maps one point through a fitted transform.
///
/// Ported from `core/UndistortionObject.cpp`, `UndistortionObject::transformLWM`. A control
/// point contributes when `distance / radius < 1` (a zero radius never contributes); its
/// weight is `1 - 3 R^2 + 2 R^3`. Returns ([`UNCOVERED`], [`UNCOVERED`]) when the weights sum
/// to exactly 0, as the C++ does; callers test for that value. Tier A.
#[must_use]
pub fn transform_lwm(transform: &LwmTransform, point: Point2) -> Point2 {
    let candidates = transform
        .index
        .0
        .get_or_init(|| CellIndex::build(transform))
        .as_ref()
        .and_then(|ix| ix.candidates(point.x, point.y));
    match candidates {
        Some(c) => lwm_sum(transform, point, c.iter().map(|&p| p as usize)),
        None => lwm_sum(transform, point, 0..transform.control_points.len()),
    }
}

/// The body of `transformLWM` over the control points `points`, in that order.
fn lwm_sum(transform: &LwmTransform, point: Point2, points: impl Iterator<Item = usize>) -> Point2 {
    let (px, py) = (point.x, point.y);
    let xy = px * py;
    let x2 = px * px;
    let y2 = py * py;

    let mut u_numerator = 0.0;
    let mut v_numerator = 0.0;
    let mut denominator = 0.0;

    for p in points {
        let cp = transform.control_points[p];
        let dx = px - cp.x;
        let dy = py - cp.y;
        let dist_to_cp = (dx * dx + dy * dy).sqrt();

        let ri = dist_to_cp / transform.radii[p];
        if ri < 1.0 {
            let ri2 = ri * ri;
            let ri3 = ri * ri2;
            let w = 1.0 - 3.0 * ri2 + 2.0 * ri3;

            let a = &transform.a[p];
            let b = &transform.b[p];
            let u = a[0] + a[1] * px + a[2] * py + a[3] * xy + a[4] * x2 + a[5] * y2;
            let v = b[0] + b[1] * px + b[2] * py + b[3] * xy + b[4] * x2 + b[5] * y2;

            u_numerator += w * u;
            v_numerator += w * v;
            denominator += w;
        }
    }

    if denominator != 0.0 {
        Point2::new(u_numerator / denominator, v_numerator / denominator)
    } else {
        Point2::new(UNCOVERED, UNCOVERED)
    }
}

/// Euclidean distance as `transformPoint` writes it.
fn residual(pt: Point2, pt_test: Point2) -> f64 {
    ((pt.x - pt_test.x) * (pt.x - pt_test.x) + (pt.y - pt_test.y) * (pt.y - pt_test.y)).sqrt()
}

/// Undistorts or distorts one point with the grid model.
///
/// Ported from `core/UndistortionObject.cpp`, `UndistortionObject::transformPoint`:
///
/// - The point goes through [`GridModel::undistort`] or [`GridModel::distort`]; a result of
///   exactly (-1, -1) means "not covered" and the input is returned unchanged.
/// - When undistorting with `with_refine`, a damped fixed-point refinement follows against
///   the distorting transform: step `pt_out + m * (pt - distort(..))` with `m` starting at
///   0.5; a step is kept when its residual is smaller, otherwise `m *= 0.8`. The loop runs
///   while the residual exceeds 1e-6 pixels and stops after the iteration with `k == 100`
///   or when a trial point's distorted x is -1. As in the C++, the next step always uses
///   the residual of the *last trial*, kept or not, and a trial whose image is uncovered
///   (-1, -1) is still kept if its (meaningless) residual is smaller.
/// - A final result of exactly (-1, -1) is again replaced by the input.
///
/// Tier A. Oracle: `undistortion/samples.csv`.
#[must_use]
pub fn transform_point(
    model: &GridModel,
    point: Point2,
    undistort: bool,
    with_refine: bool,
) -> Point2 {
    let pt = point;
    let mut pt_out = if undistort {
        transform_lwm(&model.undistort, pt)
    } else {
        transform_lwm(&model.distort, pt)
    };

    if pt_out.x == UNCOVERED && pt_out.y == UNCOVERED {
        return pt;
    }

    if undistort && with_refine {
        let mut pt_test = transform_lwm(&model.distort, pt_out);
        let mut dist = residual(pt, pt_test);
        let mut multiplier = 0.5;
        let mut k = 0;
        while dist > 0.000_001 {
            let pt_out_tmp = Point2::new(
                pt_out.x + multiplier * (pt.x - pt_test.x),
                pt_out.y + multiplier * (pt.y - pt_test.y),
            );
            pt_test = transform_lwm(&model.distort, pt_out_tmp);
            let dist_tmp = residual(pt, pt_test);
            if dist_tmp < dist {
                dist = dist_tmp;
                pt_out = pt_out_tmp;
            } else {
                multiplier *= 0.8;
            }
            if pt_test.x == UNCOVERED || k == 100 {
                break;
            }
            k += 1;
        }
    }

    if pt_out.x == UNCOVERED && pt_out.y == UNCOVERED {
        return pt;
    }
    pt_out
}

/// The largest number of pixels of a lookup table (2^28, a 16384 x 16384 image: two maps of
/// 1 GiB each), far above any camera of this program. A larger size gives an empty table.
pub const MAX_LOOKUP_PIXELS: usize = 1 << 28;

/// The lookup table of an image of `width x height` pixels that no control point covers:
/// every entry -1 ([`UNCOVERED`]). A size of more than [`MAX_LOOKUP_PIXELS`] pixels, or one
/// whose pixel count overflows `usize`, gives an empty table (0 x 0).
#[must_use]
pub fn uncovered_table(width: usize, height: usize) -> LookupTable {
    match width.checked_mul(height) {
        Some(n) if n <= MAX_LOOKUP_PIXELS => LookupTable {
            width,
            height,
            map_x: vec![-1.0; n],
            map_y: vec![-1.0; n],
        },
        _ => LookupTable {
            width: 0,
            height: 0,
            map_x: Vec::new(),
            map_y: Vec::new(),
        },
    }
}

/// Builds the image lookup table (the two `CV_32FC1` maps) for an image of
/// `width x height` pixels from a transform, normally [`GridModel::distort`].
///
/// Ported from `processing/LocalUndistortion.cpp`, `LocalUndistortion::createLookupTable`,
/// with `gridSize` from [`LwmTransform::grid_size`] (the value `computeLWM` returns). The
/// control points are binned into square cells of `gridSize` pixels (cell index by C++
/// truncation toward zero, so points in `(-gridSize, 0)` land in cell 0). For each pixel
/// only the cells from `floor(p / g) - 1` to below `ceil(p / g) + 1` (clamped to the grid)
/// are visited, x cells outer, y cells inner, points in index order. A control point
/// contributes when `distance^2 < radius^2` (unlike [`transform_lwm`], which tests
/// `distance / radius < 1`). Sums are in `f64`, converted to `f32` at the end; uncovered
/// pixels get -1.
///
/// Departures where the C++ has undefined behaviour: the C++ has `ceil(width / gridSize)`
/// columns of cells and `ceil(height / gridSize)` rows, so a control point is addressable
/// (and binned as in the C++) while `-gridSize < x < ceil(width / gridSize) * gridSize` and
/// likewise for y, which reaches past the image edge by up to one cell. A control point at
/// or beyond `ceil(size / gridSize) * gridSize`, or at or below `-gridSize`, in either
/// coordinate has a cell index outside the grid; the C++ writes outside its cell vectors
/// there, and this port leaves the point out. Such a point lies in no cell the pixel loop
/// visits.
/// A grid size of 0 or less (all radii 0, or an empty transform) gives all -1; the C++
/// divides by zero there.
///
/// A size of more than [`MAX_LOOKUP_PIXELS`] pixels, or one whose pixel count overflows
/// `usize`, gives an empty table (0 x 0), where the C++ allocates the two maps whatever the
/// size (a project file can carry any camera size).
#[must_use]
pub fn create_lookup_table(transform: &LwmTransform, width: usize, height: usize) -> LookupTable {
    create_lookup_table_threads(transform, width, height, worker_count(height))
}

#[doc(hidden)]
/// [`create_lookup_table`] with an explicit number of worker threads (at least 1), so that
/// the tests can show that the result does not depend on it. Not part of the interface.
#[must_use]
pub fn create_lookup_table_threads(
    transform: &LwmTransform,
    width: usize,
    height: usize,
    threads: usize,
) -> LookupTable {
    let mut table = uncovered_table(width, height);
    let grid_size = transform.grid_size();
    if grid_size <= 0 || table.map_x.is_empty() {
        return table;
    }
    let (mut map_x, mut map_y) = (
        std::mem::take(&mut table.map_x),
        std::mem::take(&mut table.map_y),
    );

    let g = f64::from(grid_size);
    let gwidth = (width as f64 / g).ceil() as i64;
    let gheight = (height as f64 / g).ceil() as i64;

    // grid_pts[x][y]: indices of the control points in each cell, in index order.
    let mut grid_pts: Vec<Vec<Vec<usize>>> =
        vec![vec![Vec::new(); gheight as usize]; gwidth as usize];
    for (i, cp) in transform.control_points.iter().enumerate() {
        let xi = (cp.x / g) as i64;
        let yi = (cp.y / g) as i64;
        if (0..gwidth).contains(&xi) && (0..gheight).contains(&yi) {
            grid_pts[xi as usize][yi as usize].push(i);
        }
    }

    let inv_grid_size = 1.0 / g;
    let grid_width = (width as f64 / g).ceil();
    let grid_height = (height as f64 / g).ceil();
    let radii_squared: Vec<f64> = transform.radii.iter().map(|r| r * r).collect();

    let ctx = LutContext {
        transform,
        grid_pts: &grid_pts,
        radii_squared: &radii_squared,
        inv_grid_size,
        grid_width,
        grid_height,
        width,
    };

    let ranges = split_range(height, threads.clamp(1, height));
    std::thread::scope(|s| {
        let mut rest_x: &mut [f32] = &mut map_x;
        let mut rest_y: &mut [f32] = &mut map_y;
        for range in ranges {
            let rows = range.len();
            let (chunk_x, tail_x) = rest_x.split_at_mut(rows * width);
            let (chunk_y, tail_y) = rest_y.split_at_mut(rows * width);
            rest_x = tail_x;
            rest_y = tail_y;
            let ctx = &ctx;
            s.spawn(move || {
                for (r, y_out) in range.enumerate() {
                    let row = r * width..(r + 1) * width;
                    ctx.row(y_out, &mut chunk_x[row.clone()], &mut chunk_y[row]);
                }
            });
        }
    });

    LookupTable {
        width,
        height,
        map_x,
        map_y,
    }
}

/// Shared read-only state of the `createLookupTable` pixel loop.
struct LutContext<'a> {
    transform: &'a LwmTransform,
    grid_pts: &'a [Vec<Vec<usize>>],
    radii_squared: &'a [f64],
    inv_grid_size: f64,
    grid_width: f64,
    grid_height: f64,
    width: usize,
}

impl LutContext<'_> {
    /// One output row of `createLookupTable`'s `parallel_for_` body.
    ///
    /// Ported from `processing/LocalUndistortion.cpp`, `LocalUndistortion::createLookupTable`.
    fn row(&self, y_out: usize, out_x: &mut [f32], out_y: &mut [f32]) {
        let t = self.transform;
        let y_out_d = y_out as f64;
        let y2_cached = y_out_d * y_out_d;
        for x_out in 0..self.width {
            let x_out_d = x_out as f64;
            let xy = x_out_d * y_out_d;
            let x2 = x_out_d * x_out_d;
            let y2 = y2_cached;

            let mut u_numerator = 0.0;
            let mut v_numerator = 0.0;
            let mut denominator = 0.0;

            let xg_min = cpp_max((x_out_d * self.inv_grid_size).floor() - 1.0, 0.0);
            let xg_max = cpp_min((x_out_d * self.inv_grid_size).ceil() + 1.0, self.grid_width);
            let yg_min = cpp_max((y_out_d * self.inv_grid_size).floor() - 1.0, 0.0);
            let yg_max = cpp_min(
                (y_out_d * self.inv_grid_size).ceil() + 1.0,
                self.grid_height,
            );

            let mut xg = xg_min;
            while xg < xg_max {
                let mut yg = yg_min;
                while yg < yg_max {
                    let cell = &self.grid_pts[xg as usize][yg as usize];
                    for &i in cell {
                        let cp = t.control_points[i];
                        let dx = x_out_d - cp.x;
                        let dy = y_out_d - cp.y;
                        let dist_to_cp_sq = dx * dx + dy * dy;
                        if dist_to_cp_sq < self.radii_squared[i] {
                            let dist_to_cp = dist_to_cp_sq.sqrt();
                            let radius_i = t.radii[i];
                            let ri = dist_to_cp / radius_i;
                            let ri2 = ri * ri;
                            let ri3 = ri * ri2;
                            let w = 1.0 - 3.0 * ri2 + 2.0 * ri3;

                            let a = &t.a[i];
                            let b = &t.b[i];
                            let u = a[0]
                                + a[1] * x_out_d
                                + a[2] * y_out_d
                                + a[3] * xy
                                + a[4] * x2
                                + a[5] * y2;
                            let v = b[0]
                                + b[1] * x_out_d
                                + b[2] * y_out_d
                                + b[3] * xy
                                + b[4] * x2
                                + b[5] * y2;

                            u_numerator += w * u;
                            v_numerator += w * v;
                            denominator += w;
                        }
                    }
                    yg += 1.0;
                }
                xg += 1.0;
            }

            if denominator != 0.0 {
                out_x[x_out] = (u_numerator / denominator) as f32;
                out_y[x_out] = (v_numerator / denominator) as f32;
            } else {
                out_x[x_out] = -1.0;
                out_y[x_out] = -1.0;
            }
        }
    }
}

/// `std::max(a, b)`: `(a < b) ? b : a`.
fn cpp_max(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

/// `std::min(a, b)`: `(b < a) ? b : a`.
fn cpp_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

#[cfg(test)]
mod index_tests {
    use super::*;

    /// The indexed sum equals the full loop bit for bit, inside, at the edge of and outside
    /// the control points' discs, with radii of every kind.
    #[test]
    fn indexed_lwm_equals_the_full_loop() {
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let n = 300;
        let (mut control_points, mut a, mut b, mut radii) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for p in 0..n {
            control_points.push(Point2::new(1000.0 * rnd(), 1000.0 * rnd()));
            let mut c = [0.0; 6];
            let mut d = [0.0; 6];
            for k in 0..6 {
                c[k] = rnd() - 0.5;
                d[k] = rnd() - 0.5;
            }
            a.push(c);
            b.push(d);
            radii.push(match p {
                7 => 0.0,
                11 => -5.0,
                13 => f64::INFINITY,
                17 => f64::NAN,
                _ => 20.0 + 80.0 * rnd(),
            });
        }
        let t = LwmTransform {
            control_points: control_points.into(),
            a: a.into(),
            b: b.into(),
            radii: radii.into(),
            index: LwmIndex::default(),
        };
        let full = |q: Point2| lwm_sum(&t, q, 0..n);
        let mut queries = Vec::new();
        for _ in 0..20000 {
            queries.push(Point2::new(1400.0 * rnd() - 200.0, 1400.0 * rnd() - 200.0));
        }
        for p in 0..n {
            let (c, r) = (t.control_points[p], t.radii[p]);
            if r.is_finite() {
                queries.push(Point2::new(c.x + r, c.y));
                queries.push(Point2::new(c.x, c.y - r));
                let h = r * std::f64::consts::FRAC_1_SQRT_2;
                queries.push(Point2::new(c.x + h, c.y + h));
            }
        }
        queries.push(Point2::new(f64::NAN, 3.0));
        queries.push(Point2::new(1e300, -1e300));
        for q in queries {
            let (a, b) = (transform_lwm(&t, q), full(q));
            assert!(
                a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits(),
                "{q:?}: {a:?} against {b:?}"
            );
        }
        // Without the odd radii the index is used for every finite point.
        assert!(t.index.0.get().is_some_and(Option::is_some));
    }
}
