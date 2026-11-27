//! Hexagonal grid matching and the outlier operations of the undistortion wizard.
//!
//! Ported from `processing/LocalUndistortion.cpp` (`setupCorrespondances` and everything it
//! calls: `findNClosestPoint`, `getHexagonalGridOrientation`, `getHexagonalGridSize`,
//! `get6adjCells`, `setupHexagonalGrid`, `addNeighbours`, `checkAndAddPoint`, `contains`;
//! and `localUndistortion_thread` for the whole computation) and `core/UndistortionObject.cpp`
//! (`removeOutlier`, `findClosestPoint`, `toggleOutlier`, `setCenter`, `undistortPoints`,
//! `computeError`). Survey: `docs/cpp-calibration.md` sections 2.2, 2.3 and 2.5.
//!
//! # The computation
//!
//! [`match_grid`] is `setupCorrespondances`: from the detected beads (in detection order, as
//! [`crate::beads::detect_beads`] returns them), an optional centre and the grid image's size
//! it returns the grid points (distorted positions, reference positions, all inliers) in the
//! C++'s order, with the set-up values the reference build records (`hooks/grid_setup.csv`,
//! `grid_matches.csv`, `grid_beads.csv`). [`fit_grid`] is the rest of
//! `localUndistortion_thread` (`computeUndistortion(true)`, a refit from stored grid points),
//! and [`compute_grid_undistortion`] both (`computeUndistortion(false)`).
//!
//! Arithmetic follows the C++ operation by operation in `double`; nearest beads are chosen
//! with `cv::sortIdx`'s order ([`linalg::sort_idx`]), so ties go as in the C++ (ledger A39).
//!
//! # Defects (each `true` field of [`GridMatchCompat`] / [`OutlierCompat`] is the C++)
//!
//! - A104 ([`GridMatchCompat::six_neighbours_any_target`]): the orientation and spacing of
//!   the six nearest beads, which goes wrong when they are not the six neighbours of a
//!   lattice in their expected directions (a missing neighbour, a grid near 30 degrees).
//! - A105 ([`GridMatchCompat::lattice_site_by_coordinates`]): a lattice site is recognised by
//!   comparing accumulated `double` coordinates.
//! - A106 ([`GridMatchCompat::unchecked_reference_scale`]): the reference rescaling divides
//!   by zero when one row or column is matched.
//! - A107 ([`OutlierCompat::circle_sets_inliers`]): "remove outlier at border" sets every
//!   flag from the circle test, undoing manual toggles.
//! - A108 ([`OutlierCompat::circle_radius_signed`]): its circle radius `r - 1.5 t` is
//!   squared, so a negative radius keeps points again.
//! - Fewer than seven beads (survey CD2): the C++ reads past its index array (undefined
//!   behaviour, not reproduced); [`match_grid`] returns [`GridMatchError::TooFewBeads`] in
//!   both modes.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::f64::consts::PI;

use crate::Point2;
use crate::image::ImageU8;
use crate::imgproc::contours::PointI;
use crate::imgproc::shape::min_enclosing_circle;
use crate::linalg;
use crate::undistortion::{
    GridModel, LookupTable, create_lookup_table, fit_grid_model, transform_point, uncovered_table,
};

/// Which faults of XMALab 3's grid matching to reproduce. Each field is `true` for the C++
/// behaviour. [`Default`] and [`GridMatchCompat::FIXED`] fix all; [`GridMatchCompat::XMALAB3`]
/// is the C++.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GridMatchCompat {
    /// Ledger A104: the orientation and spacing come from the six beads nearest the centre
    /// bead, each of the six target directions taking the one whose angle is closest
    /// (`LocalUndistortion.cpp:604-628`, `:658-680`). One bead can serve two targets, a
    /// second-ring bead is among the six when a neighbour is missing, and near 30 degrees
    /// noise assigns neighbours to targets 60 degrees apart: the orientation is then off by
    /// multiples of 5 to 10 degrees. Fixed: XMALab's estimate is kept whenever it is
    /// consistent with a hexagonal lattice (six distinct picks, the six residuals within 20
    /// degrees of each other, the six distances within a ratio of 1.25; every reference
    /// grid), bit for bit; otherwise the orientation is the mean modulo 60 degrees of the
    /// first-ring neighbours and the spacing their mean distance, checked against the lattice
    /// it implies (`orientation_and_spacing`).
    pub six_neighbours_any_target: bool,
    /// Ledger A105: a lattice site counts as already matched when its accumulated `double`
    /// coordinates equal those of a matched site (`contains`, `:690-698`), but the same site
    /// reached along two paths can differ in the last bits, and then gets a second bead.
    /// Fixed: sites are compared by their integer lattice coordinates. When no site is
    /// matched twice (every reference grid), the result is the C++'s bit for bit.
    pub lattice_site_by_coordinates: bool,
    /// Ledger A106: the references are rescaled by `(max - min)` of the matched beads over
    /// `(max - min)` of the lattice sites, which is 0 / 0 or x / 0 when one column or row was
    /// matched (`:234-241`), so references become NaN or infinite. Fixed:
    /// [`match_grid`] returns [`GridMatchError::DegenerateLattice`].
    pub unchecked_reference_scale: bool,
}

impl GridMatchCompat {
    /// Every fault fixed: the default.
    pub const FIXED: GridMatchCompat = GridMatchCompat {
        six_neighbours_any_target: false,
        lattice_site_by_coordinates: false,
        unchecked_reference_scale: false,
    };

    /// Everything as XMALab 3 does it.
    pub const XMALAB3: GridMatchCompat = GridMatchCompat {
        six_neighbours_any_target: true,
        lattice_site_by_coordinates: true,
        unchecked_reference_scale: true,
    };
}

/// Why the grid cannot be matched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridMatchError {
    /// Fewer than seven beads were detected: the centre bead and its six neighbours are
    /// needed. The C++ reads an empty vector (no bead) or past the end of its sorted index
    /// array (one to six beads), which is undefined behaviour (survey CD2); both modes refuse.
    TooFewBeads {
        /// Beads detected.
        found: usize,
    },
    /// Only one column or row of the lattice was matched, so the reference points cannot be
    /// scaled onto the beads (ledger A106; default mode only).
    DegenerateLattice {
        /// Grid points matched.
        matched: usize,
    },
}

impl std::fmt::Display for GridMatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GridMatchError::TooFewBeads { found } => write!(
                f,
                "the undistortion grid needs at least 7 detected beads, {found} found"
            ),
            GridMatchError::DegenerateLattice { matched } => write!(
                f,
                "the undistortion grid matched only one row or column of beads ({matched} points)"
            ),
        }
    }
}

impl std::error::Error for GridMatchError {}

/// The values `setupCorrespondances` works with, as the reference build records them
/// (`hooks/grid_setup.csv`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridSetup {
    /// Beads detected.
    pub n_detected: usize,
    /// A centre was set by the user.
    pub center_set: bool,
    /// The centre asked for: the one set, or (0.5 · width, 0.5 · height).
    pub center_requested: Point2,
    /// The bead nearest to it (the lattice origin).
    pub center: Point2,
    /// Its index in the detected beads.
    pub center_index: usize,
    /// Grid orientation in radians (`orient`): the mean, in degrees, converted with
    /// `/ 180 * M_PI`.
    pub orientation: f64,
    /// Spacing between rows (`dY`).
    pub dy: f64,
    /// Spacing between columns, `sin(pi / 3) * dY`.
    pub dx: f64,
    /// Row offset of a neighbouring column, `0.5 * dY`.
    pub doffy: f64,
    /// Bounds of the distorted grid points (`minXDet`, `maxXDet`, `minYDet`, `maxYDet`).
    pub det_bounds: [f64; 4],
    /// Bounds of the lattice sites after rotating back (`minXCont`, ..., `maxYCont`).
    pub cont_bounds: [f64; 4],
    /// `scaleX`, `scaleY` of the final rescaling.
    pub scale: [f64; 2],
}

impl GridSetup {
    /// `orient * 180 / M_PI`, the orientation as the reference build writes it.
    #[must_use]
    pub fn orientation_deg(&self) -> f64 {
        self.orientation * 180.0 / PI
    }
}

/// One grid point as the matching found it (`hooks/grid_matches.csv`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridMatchPoint {
    /// Index of the bead in the detected beads.
    pub detected_index: usize,
    /// Lattice column: the site is `center + col · dx` in x, rotated frame.
    pub lattice_col: i32,
    /// Lattice row: the site is `center + col · doffy + row · dy` in y, rotated frame.
    pub lattice_row: i32,
    /// The bead, rotated by the grid orientation about the centre bead.
    pub rotated: Point2,
    /// The lattice site in the rotated frame, as accumulated during the growth.
    pub lattice: Point2,
}

/// The result of [`match_grid`]: XMALab's `tmpPoints_distorted`, `tmpPoints_references` and
/// `tmpPoints_inlier` (what `setGridPoints` stores), with the set-up values.
#[derive(Clone, Debug, PartialEq)]
pub struct GridMatch {
    /// Distorted grid points (the beads rotated and rotated back: equal to the detected
    /// beads only to rounding), in matching order; the first is the centre bead.
    pub distorted: Vec<Point2>,
    /// Reference points (lattice sites rotated back and rescaled), same order.
    pub references: Vec<Point2>,
    /// Inlier flags: all `true` after a matching.
    pub inlier: Vec<bool>,
    /// Per grid point: bead, lattice coordinates and rotated positions.
    pub matches: Vec<GridMatchPoint>,
    /// The set-up values.
    pub setup: GridSetup,
}

/// The fitted model and everything `localUndistortion_thread` stores after the fit.
#[derive(Clone, Debug, PartialEq)]
pub struct GridFit {
    /// Both LWM directions; `None` when there are no inliers (the C++ then stores empty
    /// matrices: every point maps to itself) or fewer inliers than `neighbours` (undefined
    /// behaviour in the C++; treated the same).
    pub model: Option<GridModel>,
    /// `computeLWM`'s return value for the distorting fit: the lookup table's cell size
    /// (`ceil` of the largest radius), -1 when `model` is `None` (`hooks/lwm_run.csv`).
    pub grid_size: i32,
    /// The lookup table (`map_x`, `map_y`); all -1 when `model` is `None`.
    pub lookup: LookupTable,
    /// Every grid point (inliers and outliers) through the undistorting transform with
    /// refinement (`points_grid_undistorted`).
    pub undistorted: Vec<Point2>,
    /// Distance of each undistorted point to its reference point (`error`).
    pub error: Vec<f64>,
}

/// The whole `computeUndistortion(false)`: the matching and the fit.
#[derive(Clone, Debug, PartialEq)]
pub struct GridUndistortion {
    /// The grid points.
    pub matching: GridMatch,
    /// The model fitted to its inliers.
    pub fit: GridFit,
}

/// Euclidean distance of every point to `pt`, as `findNClosestPoint` computes it:
/// subtract, `cv::multiply`, `cv::add`, `cv::sqrt`, all in `double`.
fn distances(pt: Point2, pts: &[Point2]) -> Vec<f64> {
    pts.iter()
        .map(|p| {
            let dx = p.x - pt.x;
            let dy = p.y - pt.y;
            (dx * dx + dy * dy).sqrt()
        })
        .collect()
}

/// Index of the point nearest to `pt`: the first index of `cv::sortIdx` on the distances
/// (`findNClosestPoint(1, ...)`). `None` for no points.
///
/// The minimum is found by a scan; only when it is not unique (or a distance is NaN) is the
/// full `sortIdx` port run, so the tie goes as in the C++.
///
/// Ported from processing/LocalUndistortion.cpp, `LocalUndistortion::findNClosestPoint`.
fn nearest_index(pt: Point2, pts: &[Point2]) -> Option<usize> {
    if pts.is_empty() {
        return None;
    }
    let d = distances(pt, pts);
    let mut best = 0;
    let mut ties = 0usize;
    let mut has_nan = false;
    for (i, &v) in d.iter().enumerate() {
        if v.is_nan() {
            has_nan = true;
        } else if v < d[best] || d[best].is_nan() {
            best = i;
            ties = 0;
        } else if v == d[best] && i != best {
            ties += 1;
        }
    }
    if has_nan || ties > 0 {
        return Some(linalg::sort_idx(&d, false)[0]);
    }
    Some(best)
}

/// The six beads nearest the centre bead, skipping the nearest (the centre itself):
/// `findNClosestPoint(6, center, ..., skipfirst = true)`. Needs at least seven beads.
///
/// Ported from processing/LocalUndistortion.cpp, `LocalUndistortion::findNClosestPoint`.
fn six_closest(center: Point2, pts: &[Point2]) -> [Point2; 6] {
    let idx = linalg::sort_idx(&distances(center, pts), false);
    std::array::from_fn(|j| pts[idx[j + 1]])
}

/// Angles in degrees from the centre to each neighbour, `(180 / M_PI) * atan2(dy, dx)`.
fn neighbour_angles(center: Point2, n: &[Point2; 6]) -> [f64; 6] {
    std::array::from_fn(|i| (180.0 / PI) * (n[i].y - center.y).atan2(n[i].x - center.x))
}

/// Index of the neighbour whose angle is closest to the direction `-s`: the first index of
/// `cv::sortIdx(abs(angles + s))` over six values.
fn closest_to_target(angles: &[f64; 6], s: f64) -> usize {
    let tmp: Vec<f64> = angles.iter().map(|a| (a + s).abs()).collect();
    linalg::sort_idx(&tmp, false)[0]
}

/// `cv::mean` of six `double` values: OpenCV's `sum_` with its four-way unrolled loop, then
/// times `1. / 6`.
///
/// Ported from OpenCV 4.13 core/src/mean.dispatch.cpp, `cv::mean`, and
/// core/src/sum.simd.hpp, `sum_` (`cn == 1`, `CV_ENABLE_UNROLLED`).
fn cv_mean6(v: &[f64; 6]) -> f64 {
    let mut s0 = 0.0f64;
    s0 += v[0] + v[1] + v[2] + v[3];
    s0 += v[4];
    s0 += v[5];
    s0 * (1.0 / 6.0)
}

/// Target directions of `getHexagonalGridOrientation`, in its order.
const ORIENT_TARGETS: [f64; 6] = [90.0, 30.0, -30.0, -90.0, -150.0, 150.0];
/// Target directions of `get6adjCells`, in its order (the last two swapped).
const SIZE_TARGETS: [f64; 6] = [90.0, 30.0, -30.0, -90.0, 150.0, -150.0];

/// Largest spread (max - min, degrees) of the six orientation residuals `angle + target`
/// for which XMALab's estimate counts as consistent (ledger A104). A consistent assignment
/// gives six residuals equal up to noise and distortion; a neighbour assigned to the wrong
/// target direction differs by about 60 degrees, a second-ring bead among the six by 30.
const CONSISTENT_RESIDUAL_SPREAD_DEG: f64 = 20.0;
/// Largest ratio of the largest to the smallest of the six neighbour distances for which
/// XMALab's estimate counts as consistent (ledger A104): a second-ring bead is at sqrt(3)
/// spacings.
const CONSISTENT_DISTANCE_RATIO: f64 = 1.25;
/// Beads around the centre bead that the corrected estimate considers (ledger A104).
const CORRECTED_CANDIDATES: usize = 12;
/// A candidate spacing takes as its first ring the beads within this fraction of it.
const CORRECTED_RING_WIDTH: f64 = 0.2;
/// A neighbour supports a corrected estimate when it lies within this fraction of the
/// spacing of a first-ring site the estimate implies.
const CORRECTED_SITE_TOLERANCE: f64 = 0.2;
/// First-ring neighbours a corrected estimate needs.
const CORRECTED_MIN_SUPPORT: usize = 4;

/// XMALab's estimate and the test of whether it is consistent with a hexagonal lattice
/// (ledger A104).
struct XmalabEstimate {
    /// `getHexagonalGridOrientation`, degrees.
    orientation: f64,
    /// `getHexagonalGridSize`.
    spacing: f64,
    /// The six target picks are six different neighbours.
    bijective: bool,
    /// Bijective, the six residuals `angle + target` lie within
    /// [`CONSISTENT_RESIDUAL_SPREAD_DEG`] of each other, and the six distances within the
    /// ratio [`CONSISTENT_DISTANCE_RATIO`].
    consistent: bool,
}

fn dist_to(center: Point2, p: Point2) -> f64 {
    ((p.x - center.x) * (p.x - center.x) + (p.y - center.y) * (p.y - center.y)).sqrt()
}

/// XMALab's estimate from the six nearest beads, with its consistency test.
///
/// Ported from processing/LocalUndistortion.cpp, `getHexagonalGridOrientation`,
/// `getHexagonalGridSize` and `get6adjCells` (both on the unrotated beads).
fn xmalab_estimate(center: Point2, closest: &[Point2; 6], angles: &[f64; 6]) -> XmalabEstimate {
    let picks: [usize; 6] = std::array::from_fn(|t| closest_to_target(angles, ORIENT_TARGETS[t]));
    let mut distinct = picks;
    distinct.sort_unstable();
    let bijective = distinct.windows(2).all(|w| w[0] != w[1]);

    // getHexagonalGridOrientation: later targets overwrite earlier ones at one index.
    let mut placed = [0.0f64; 6];
    for (t, &j) in picks.iter().enumerate() {
        placed[j] = ORIENT_TARGETS[t];
    }
    let sums: [f64; 6] = std::array::from_fn(|i| angles[i] + placed[i]);
    let orientation = cv_mean6(&sums);
    // get6adjCells / getHexagonalGridSize.
    let mut spacing = 0.0f64;
    for &s in &SIZE_TARGETS {
        spacing += dist_to(center, closest[closest_to_target(angles, s)]);
    }
    spacing /= 6.0;

    let (rmin, rmax) = sums
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let d: [f64; 6] = std::array::from_fn(|i| dist_to(center, closest[i]));
    let (dmin, dmax) = d
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let consistent = bijective
        && rmax - rmin <= CONSISTENT_RESIDUAL_SPREAD_DEG
        && dmin > 0.0
        && dmax <= CONSISTENT_DISTANCE_RATIO * dmin;
    XmalabEstimate {
        orientation,
        spacing,
        bijective,
        consistent,
    }
}

/// Orientation in degrees, in (-30, 30], of first-ring neighbours at the angles `angles`
/// (degrees): the neighbours of a lattice of orientation `o` lie at `o + 30 + 60 k`, so
/// `6 angle - 180` is `6 o` modulo 360 for each, whichever neighbour is which. The mean is
/// taken on the circle.
fn orientation_mod60(angles: &[f64]) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for &a in angles {
        let t = (6.0 * a - 180.0).to_radians();
        s += t.sin();
        c += t.cos();
    }
    let o = s.atan2(c).to_degrees() / 6.0;
    if o <= -30.0 { o + 60.0 } else { o }
}

/// The corrected estimate of ledger A104, used only where XMALab's is inconsistent.
///
/// Among the [`CORRECTED_CANDIDATES`] beads nearest the centre bead, each of their distances
/// up to `1 + `[`CORRECTED_RING_WIDTH`] times the nearest is tried as the spacing (the first
/// ring holds the nearest bead): the beads within [`CORRECTED_RING_WIDTH`] of it are taken as the
/// first ring, the orientation is their mean modulo 60 degrees ([`orientation_mod60`]) and
/// the spacing their mean distance. The estimate is then checked against the lattice it
/// implies: a neighbour supports it when it lies within [`CORRECTED_SITE_TOLERANCE`]
/// spacings of one of the six first-ring sites (each site taken once), and the estimate is
/// recomputed from its supporting neighbours only. Of the estimates with at least
/// [`CORRECTED_MIN_SUPPORT`] supporting neighbours, those within [`CORRECTED_RING_WIDTH`] of
/// the smallest spacing are kept (the second ring is a lattice too), and the one with the
/// most support (ties to the smaller summed distance to the sites) is returned as
/// (orientation in degrees, spacing); `None` when none has enough support.
fn corrected_estimate(center: Point2, pts: &[Point2]) -> Option<(f64, f64)> {
    let idx = linalg::sort_idx(&distances(center, pts), false);
    // idx[0] is the centre bead itself.
    let near: Vec<Point2> = idx
        .iter()
        .skip(1)
        .take(CORRECTED_CANDIDATES)
        .map(|&i| pts[i])
        .collect();
    let d: Vec<f64> = near.iter().map(|&p| dist_to(center, p)).collect();
    let ang: Vec<f64> = near
        .iter()
        .map(|p| (p.y - center.y).atan2(p.x - center.x).to_degrees())
        .collect();

    // The neighbours supporting (orientation, spacing), and their summed site distances.
    let support = |o: f64, s: f64| -> (Vec<usize>, f64) {
        let mut taken = [false; 6];
        let mut who = Vec::new();
        let mut cost = 0.0;
        for (i, p) in near.iter().enumerate() {
            let mut best: Option<(usize, f64)> = None;
            for (k, t) in taken.iter().enumerate() {
                if *t {
                    continue;
                }
                let a = (o + 30.0 + 60.0 * k as f64).to_radians();
                let (sx, sy) = (center.x + s * a.cos(), center.y + s * a.sin());
                let e = ((p.x - sx) * (p.x - sx) + (p.y - sy) * (p.y - sy)).sqrt();
                if e < CORRECTED_SITE_TOLERANCE * s && best.is_none_or(|(_, b)| e < b) {
                    best = Some((k, e));
                }
            }
            if let Some((k, e)) = best {
                taken[k] = true;
                who.push(i);
                cost += e;
            }
        }
        (who, cost)
    };
    let fit = |who: &[usize]| {
        let o = orientation_mod60(&who.iter().map(|&i| ang[i]).collect::<Vec<_>>());
        let s = who.iter().map(|&i| d[i]).sum::<f64>() / who.len() as f64;
        (o, s)
    };

    let mut found: Vec<(usize, f64, f64, f64)> = Vec::new();
    let nearest = d.first().copied().unwrap_or(f64::NAN);
    for &s0 in &d {
        // The first ring holds the nearest bead: a ring farther out than that is not tried.
        if !(s0 > 0.0 && s0.is_finite() && s0 <= (1.0 + CORRECTED_RING_WIDTH) * nearest) {
            continue;
        }
        let ring: Vec<usize> = (0..near.len())
            .filter(|&i| (d[i] - s0).abs() <= CORRECTED_RING_WIDTH * s0)
            .collect();
        if ring.len() < CORRECTED_MIN_SUPPORT {
            continue;
        }
        let (o, s) = fit(&ring);
        let (who, _) = support(o, s);
        if who.len() < CORRECTED_MIN_SUPPORT {
            continue;
        }
        let (o, s) = fit(&who);
        let (who, cost) = support(o, s);
        if who.len() < CORRECTED_MIN_SUPPORT || !o.is_finite() || !s.is_finite() {
            continue;
        }
        found.push((who.len(), cost, o, s));
    }
    // The first ring: the smallest spacing found (the second ring, at sqrt(3) spacings, is
    // a lattice too); among the estimates of about that spacing, the best supported.
    let s_min = found.iter().map(|f| f.3).fold(f64::INFINITY, f64::min);
    found
        .into_iter()
        .filter(|f| f.3 <= (1.0 + CORRECTED_RING_WIDTH) * s_min)
        .reduce(|a, b| {
            if b.0 > a.0 || (b.0 == a.0 && b.1 < a.1) {
                b
            } else {
                a
            }
        })
        .map(|(_, _, o, s)| (o, s))
}

/// The grid orientation in degrees and the spacing `dY` from the neighbours of the centre
/// bead.
///
/// XMALab's estimate is used whenever it is consistent with a hexagonal lattice (see
/// [`XmalabEstimate::consistent`]; the result is then the C++'s bit for bit), and always
/// in the XMALab 3 mode. Otherwise (ledger A104, default) the corrected estimate of
/// [`corrected_estimate`]. When that finds no lattice either: XMALab's estimate when its
/// six picks are distinct, else each of the six neighbours takes its own closest target
/// direction and the spacing is the mean of the six distances.
///
/// Ported from processing/LocalUndistortion.cpp, `getHexagonalGridOrientation`,
/// `getHexagonalGridSize` and `get6adjCells` (both on the unrotated beads).
fn orientation_and_spacing(center: Point2, pts: &[Point2], compat: GridMatchCompat) -> (f64, f64) {
    let closest = six_closest(center, pts);
    let angles = neighbour_angles(center, &closest);
    let x = xmalab_estimate(center, &closest, &angles);
    if x.consistent || compat.six_neighbours_any_target {
        return (x.orientation, x.spacing);
    }
    if let Some(r) = corrected_estimate(center, pts) {
        return r;
    }
    if x.bijective {
        return (x.orientation, x.spacing);
    }
    // No lattice found: each neighbour takes its own closest target direction.
    let sums: [f64; 6] = std::array::from_fn(|i| {
        let mut best = ORIENT_TARGETS[0];
        for &s in &ORIENT_TARGETS[1..] {
            if (angles[i] + s).abs() < (angles[i] + best).abs() {
                best = s;
            }
        }
        angles[i] + best
    });
    let orient = cv_mean6(&sums);
    let mut dy = 0.0f64;
    for p in &closest {
        dy += dist_to(center, *p);
    }
    dy /= 6.0;
    (orient, dy)
}

/// Rotates `p` about `c` with `x' = s·dy + co·dx`, `y' = co·dy − s·dx`, as
/// `setupCorrespondances` does (`s = sin(angle)`, `co = cos(angle)`).
fn rotate_about(p: Point2, c: Point2, s: f64, co: f64) -> Point2 {
    let x = p.x - c.x;
    let y = p.y - c.y;
    let tmpx = s * y + co * x;
    let ny = co * y - s * x;
    Point2::new(tmpx + c.x, ny + c.y)
}

/// Key of a `double` point for an exact `==` lookup: `None` when a coordinate is NaN (never
/// equal), and -0 folded onto +0 (equal under `==`).
fn point_key(p: Point2) -> Option<(u64, u64)> {
    if p.x.is_nan() || p.y.is_nan() {
        return None;
    }
    let fold = |v: f64| if v == 0.0 { 0u64 } else { v.to_bits() };
    Some((fold(p.x), fold(p.y)))
}

/// Lattice sites, matched beads and, per grid point, (bead index, column, row).
type Lattice = (Vec<Point2>, Vec<Point2>, Vec<(usize, i32, i32)>);

/// The six neighbour offsets of `addNeighbours`, in its order, as lattice steps
/// (column, row): (0, +dY), (0, −dY), (+dX, +dOffY), (−dX, +dOffY), (+dX, −dOffY),
/// (−dX, −dOffY).
const NEIGHBOUR_STEPS: [(i32, i32); 6] = [(0, 1), (0, -1), (1, 0), (-1, 1), (1, -1), (-1, 0)];

/// The breadth-first growth of the lattice from the centre bead over the rotated beads.
/// Returns, per grid point, the bead index, its lattice coordinates and the site.
///
/// Ported from processing/LocalUndistortion.cpp, `setupHexagonalGrid`, `addNeighbours`,
/// `checkAndAddPoint` and `contains`.
fn grow_lattice(
    center: Point2,
    center_index: usize,
    rotated: &[Point2],
    dy: f64,
    compat: GridMatchCompat,
) -> Lattice {
    let mut references = vec![center];
    let mut distorted = vec![center];
    let mut dy_vec = vec![dy];
    let mut info = vec![(center_index, 0i32, 0i32)];

    let mut ref_keys: HashSet<(u64, u64)> = HashSet::new();
    let mut site_keys: HashSet<(i32, i32)> = HashSet::new();
    let mut det_keys: HashSet<(u64, u64)> = HashSet::new();
    if let Some(k) = point_key(center) {
        ref_keys.insert(k);
        det_keys.insert(k);
    }
    site_keys.insert((0, 0));

    let dx = (PI / 3.0).sin() * dy;
    let doffy = 0.5 * dy;

    let mut i = 0;
    while i < dy_vec.len() {
        let centercont = references[i];
        let centerdet = distorted[i];
        let dydist = dy_vec[i];
        let (col, row) = (info[i].1, info[i].2);
        let dxdist = (PI / 3.0).sin() * dydist;
        let doffydist = 0.5 * dydist;
        let thresh = doffydist;

        let candidates: [(Point2, Point2); 6] = [
            (
                Point2::new(centercont.x, centercont.y + dy),
                Point2::new(centerdet.x, centerdet.y + dydist),
            ),
            (
                Point2::new(centercont.x, centercont.y - dy),
                Point2::new(centerdet.x, centerdet.y - dydist),
            ),
            (
                Point2::new(centercont.x + dx, centercont.y + doffy),
                Point2::new(centerdet.x + dxdist, centerdet.y + doffydist),
            ),
            (
                Point2::new(centercont.x - dx, centercont.y + doffy),
                Point2::new(centerdet.x - dxdist, centerdet.y + doffydist),
            ),
            (
                Point2::new(centercont.x + dx, centercont.y - doffy),
                Point2::new(centerdet.x + dxdist, centerdet.y - doffydist),
            ),
            (
                Point2::new(centercont.x - dx, centercont.y - doffy),
                Point2::new(centerdet.x - dxdist, centerdet.y - doffydist),
            ),
        ];

        for (k, &(cont, det)) in candidates.iter().enumerate() {
            let Some(b) = nearest_index(det, rotated) else {
                continue;
            };
            let bp = rotated[b];
            let site = (col + NEIGHBOUR_STEPS[k].0, row + NEIGHBOUR_STEPS[k].1);
            let site_present = if compat.lattice_site_by_coordinates {
                point_key(cont).is_some_and(|key| ref_keys.contains(&key))
            } else {
                site_keys.contains(&site)
            };
            if site_present {
                continue;
            }
            if point_key(bp).is_some_and(|key| det_keys.contains(&key)) {
                continue;
            }
            let ex = det.x - bp.x;
            let ey = det.y - bp.y;
            if (ex * ex + ey * ey).sqrt().partial_cmp(&thresh) != Some(Ordering::Less) {
                continue;
            }
            references.push(cont);
            distorted.push(bp);
            let cx = centerdet.x - bp.x;
            let cy = centerdet.y - bp.y;
            dy_vec.push((cx * cx + cy * cy).sqrt());
            info.push((b, site.0, site.1));
            if let Some(key) = point_key(cont) {
                ref_keys.insert(key);
            }
            site_keys.insert(site);
            if let Some(key) = point_key(bp) {
                det_keys.insert(key);
            }
        }
        i += 1;
    }
    (references, distorted, info)
}

/// Matches the detected beads of an undistortion grid to a hexagonal lattice.
///
/// `detected` are the beads in detection order; `center` the centre set by the user (C7),
/// if any; `width`, `height` the grid image's size. Steps (survey 2.2): the bead nearest the
/// centre (or the image centre); the orientation and spacing from its six neighbours; every
/// bead rotated about it; the lattice grown breadth-first; inliers all `true`; the matched
/// beads and lattice sites rotated back; the sites rescaled onto the bounding box of the
/// beads.
///
/// Ported from processing/LocalUndistortion.cpp, `LocalUndistortion::setupCorrespondances`
/// (`computeUndistortion(false)`, before the fit).
///
/// # Errors
/// [`GridMatchError::TooFewBeads`] for fewer than seven beads (both modes);
/// [`GridMatchError::DegenerateLattice`] when one row or column was matched (default mode).
pub fn match_grid(
    detected: &[Point2],
    center: Option<Point2>,
    width: usize,
    height: usize,
    compat: GridMatchCompat,
) -> Result<GridMatch, GridMatchError> {
    if detected.len() < 7 {
        return Err(GridMatchError::TooFewBeads {
            found: detected.len(),
        });
    }
    let (w, h) = (width as f64, height as f64);
    let center_requested = center.unwrap_or_else(|| Point2::new(0.5 * w, 0.5 * h));
    let center_index = nearest_index(center_requested, detected).expect("at least seven beads");
    let c = detected[center_index];

    let (orient_deg, dy) = orientation_and_spacing(c, detected, compat);
    let orient = orient_deg / 180.0 * PI;

    let (s, co) = (orient.sin(), orient.cos());
    let rotated: Vec<Point2> = detected
        .iter()
        .map(|&p| rotate_about(p, c, s, co))
        .collect();

    let (lattice, beads, info) = grow_lattice(c, center_index, &rotated, dy, compat);

    let (sb, cb) = ((-orient).sin(), (-orient).cos());
    let distorted: Vec<Point2> = beads.iter().map(|&p| rotate_about(p, c, sb, cb)).collect();
    let mut references: Vec<Point2> = lattice
        .iter()
        .map(|&p| rotate_about(p, c, sb, cb))
        .collect();

    let (mut min_x_det, mut max_x_det, mut min_y_det, mut max_y_det) = (w, -w, h, -h);
    let (mut min_x_cont, mut max_x_cont, mut min_y_cont, mut max_y_cont) = (w, -w, h, -h);
    for (d, r) in distorted.iter().zip(&references) {
        if d.x > max_x_det {
            max_x_det = d.x;
        }
        if d.x < min_x_det {
            min_x_det = d.x;
        }
        if d.y > max_y_det {
            max_y_det = d.y;
        }
        if d.y < min_y_det {
            min_y_det = d.y;
        }
        if r.x > max_x_cont {
            max_x_cont = r.x;
        }
        if r.x < min_x_cont {
            min_x_cont = r.x;
        }
        if r.y > max_y_cont {
            max_y_cont = r.y;
        }
        if r.y < min_y_cont {
            min_y_cont = r.y;
        }
    }
    let den_x = max_x_cont - min_x_cont;
    let den_y = max_y_cont - min_y_cont;
    if !compat.unchecked_reference_scale && !(den_x > 0.0 && den_y > 0.0) {
        return Err(GridMatchError::DegenerateLattice {
            matched: distorted.len(),
        });
    }
    let scale_x = (max_x_det - min_x_det) / den_x;
    let scale_y = (max_y_det - min_y_det) / den_y;
    for r in &mut references {
        r.x = scale_x * (r.x - min_x_cont) + min_x_det;
        r.y = scale_y * (r.y - min_y_cont) + min_y_det;
    }

    let matches = info
        .iter()
        .zip(beads.iter().zip(&lattice))
        .map(|(&(idx, col, row), (&b, &l))| GridMatchPoint {
            detected_index: idx,
            lattice_col: col,
            lattice_row: row,
            rotated: b,
            lattice: l,
        })
        .collect();

    Ok(GridMatch {
        inlier: vec![true; distorted.len()],
        distorted,
        references,
        matches,
        setup: GridSetup {
            n_detected: detected.len(),
            center_set: center.is_some(),
            center_requested,
            center: c,
            center_index,
            orientation: orient,
            dy,
            dx: (PI / 3.0).sin() * dy,
            doffy: 0.5 * dy,
            det_bounds: [min_x_det, max_x_det, min_y_det, max_y_det],
            cont_bounds: [min_x_cont, max_x_cont, min_y_cont, max_y_cont],
            scale: [scale_x, scale_y],
        },
    })
}

/// Fits the grid model to stored grid points and computes what the C++ stores with it: the
/// lookup table, the undistorted grid points and their errors. `computeUndistortion(true)`
/// (C2 with only inliers changed, and the refit after loading, C12).
///
/// `neighbours` is the `LocalUndistortionNeighbours` setting (default 12); `width`, `height`
/// the grid image's size (the lookup table's).
///
/// Ported from processing/LocalUndistortion.cpp, `LocalUndistortion::localUndistortion_thread`
/// (after `setupCorrespondances`), and core/UndistortionObject.cpp,
/// `UndistortionObject::undistortPoints` and `computeError`.
///
/// A size of more than [`crate::undistortion::MAX_LOOKUP_PIXELS`] pixels gives an empty
/// lookup table (0 x 0), as [`create_lookup_table`] does.
///
/// Panics if the three slices differ in length.
#[must_use]
pub fn fit_grid(
    distorted: &[Point2],
    references: &[Point2],
    inlier: &[bool],
    width: usize,
    height: usize,
    neighbours: usize,
) -> GridFit {
    let model = fit_grid_model(distorted, references, inlier, neighbours);
    let (grid_size, lookup) = match &model {
        Some(m) => (
            m.distort.grid_size(),
            create_lookup_table(&m.distort, width, height),
        ),
        None => (-1, uncovered_table(width, height)),
    };
    let undistorted: Vec<Point2> = distorted
        .iter()
        .map(|&p| match &model {
            Some(m) => transform_point(m, p, true, true),
            None => p,
        })
        .collect();
    let error = undistorted
        .iter()
        .zip(references)
        .map(|(u, r)| {
            let dx = u.x - r.x;
            let dy = u.y - r.y;
            (dx * dx + dy * dy).sqrt()
        })
        .collect();
    GridFit {
        model,
        grid_size,
        lookup,
        undistorted,
        error,
    }
}

/// The whole `computeUndistortion(false)` on detected beads: [`match_grid`], then
/// [`fit_grid`] on its points (all inliers). Used for the first computation (C1) and after a
/// new centre (C2 with a centre pending): the inliers are reset.
///
/// Ported from processing/LocalUndistortion.cpp, `LocalUndistortion::localUndistortion_thread`
/// with `hasReferences == false`.
///
/// # Errors
/// As [`match_grid`].
pub fn compute_grid_undistortion(
    detected: &[Point2],
    center: Option<Point2>,
    width: usize,
    height: usize,
    neighbours: usize,
    compat: GridMatchCompat,
) -> Result<GridUndistortion, GridMatchError> {
    let matching = match_grid(detected, center, width, height, compat)?;
    let fit = fit_grid(
        &matching.distorted,
        &matching.references,
        &matching.inlier,
        width,
        height,
        neighbours,
    );
    Ok(GridUndistortion { matching, fit })
}

// ---------------------------------------------------------------------------------------
// Outlier operations
// ---------------------------------------------------------------------------------------

/// `UndistortionObject::requiresRecalibration`: what the next "recompute undistortion" (C2)
/// does for a grid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum PendingRecompute {
    /// 0: nothing pending.
    #[default]
    None,
    /// 1: inliers changed; refit from the stored grid points ([`fit_grid`]).
    Refit,
    /// 2: a centre was set; match again from the stored beads ([`compute_grid_undistortion`]
    /// with the centre), which resets every inlier.
    Rematch,
}

impl PendingRecompute {
    /// After a toggled outlier: at least [`PendingRecompute::Refit`]
    /// (`requiresRecalibration = max(current, 1)`; `toggleOutlier`, nothing changes when no
    /// point was toggled).
    #[must_use]
    pub fn after_toggle(self) -> PendingRecompute {
        self.max(PendingRecompute::Refit)
    }

    /// After "remove outlier at border": [`PendingRecompute::Refit`], also when a centre was
    /// pending (`removeOutlier` sets 1, so a pending new centre is dropped: XMALab's
    /// behaviour).
    #[must_use]
    pub fn after_remove_outliers(self) -> PendingRecompute {
        PendingRecompute::Refit
    }

    /// After a centre is set: [`PendingRecompute::Rematch`] (`setCenter`).
    #[must_use]
    pub fn after_set_center(self) -> PendingRecompute {
        PendingRecompute::Rematch
    }
}

/// Which faults of XMALab 3's "remove outlier at border" to reproduce. `true` is the C++.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OutlierCompat {
    /// Ledger A107: every point's inlier flag is **set** from the circle test
    /// (`UndistortionObject.cpp:183-188`), so points toggled out by hand inside the circle
    /// become inliers again. Fixed: the circle test only clears flags.
    pub circle_sets_inliers: bool,
    /// Ledger A108: the circle test is `d² <= (r − 1.5 t)²`; when `r < 1.5 t` the squared
    /// radius grows again, so a larger threshold keeps more points. Fixed: a radius
    /// `r − 1.5 t` of 0 or less keeps no point.
    pub circle_radius_signed: bool,
}

impl OutlierCompat {
    /// Both faults fixed: the default.
    pub const FIXED: OutlierCompat = OutlierCompat {
        circle_sets_inliers: false,
        circle_radius_signed: false,
    };

    /// As XMALab 3.
    pub const XMALAB3: OutlierCompat = OutlierCompat {
        circle_sets_inliers: true,
        circle_radius_signed: true,
    };
}

/// The circle `removeOutlier` fits: `cv::minEnclosingCircle` of every pixel of the 8-bit
/// grid image brighter than 100, the pixels listed column by column (x outer, y inner), as
/// `cv::Point2f`. Centre and radius in `float`; `((0, 0), 0)` when no pixel is above 100
/// (OpenCV initialises them; the survey's open question 13 is settled by OpenCV's source).
///
/// The C++ passes `Point2f`, OpenCV's `CV_32F` path; [`min_enclosing_circle`] is the
/// `CV_32S` path. For whole-number coordinates below 2^24 every sum and difference the two
/// paths form is exact in `float`, so they give the same circle bit for bit.
///
/// Ported from core/UndistortionObject.cpp, `UndistortionObject::removeOutlier` (lines
/// 166-180).
#[must_use]
pub fn bright_region_circle(image: &ImageU8) -> ((f32, f32), f32) {
    let (w, h) = (image.width(), image.height());
    let data = image.data();
    let mut pts = Vec::new();
    for x in 0..w {
        for y in 0..h {
            if data[y * w + x] > 100 {
                pts.push(PointI::new(x as i32, y as i32));
            }
        }
    }
    min_enclosing_circle(&pts)
}

/// "Remove outlier at border" (C8) on one grid: points outside the bright region's circle
/// shrunk by `1.5 · threshold_circle`, and points closer than `threshold_border` to an
/// image edge (`x < b`, `y < b`, `width − x < b`, `height − y < b`), become outliers.
///
/// `image` is the 8-bit grid image (its size is the C++'s `getWidth`, `getHeight`);
/// `distorted` and `inlier` the stored grid points and flags. The dialog's two thresholds
/// are integers (0 to 100; defaults border 10, circle 25). The caller sets the pending
/// recompute with [`PendingRecompute::after_remove_outliers`]. XMALab runs this on every
/// camera with a grid, computed or not (with no grid points it changes only the pending
/// state).
///
/// Ported from core/UndistortionObject.cpp, `UndistortionObject::removeOutlier`.
///
/// Panics if `distorted` and `inlier` differ in length.
pub fn remove_outliers(
    image: &ImageU8,
    distorted: &[Point2],
    inlier: &mut [bool],
    threshold_circle: f64,
    threshold_border: f64,
    compat: OutlierCompat,
) {
    assert_eq!(
        distorted.len(),
        inlier.len(),
        "remove_outliers: grid points and inlier flags differ in length"
    );
    let ((cx, cy), radius) = bright_region_circle(image);
    let (cx, cy, radius) = (f64::from(cx), f64::from(cy), f64::from(radius));
    let r = radius - 1.5 * threshold_circle;
    for (p, flag) in distorted.iter().zip(inlier.iter_mut()) {
        let dx = p.x - cx;
        let dy = p.y - cy;
        let dist = dx * dx + dy * dy;
        let inside = if compat.circle_radius_signed {
            dist.partial_cmp(&(r * r)) != Some(Ordering::Greater)
        } else {
            r > 0.0 && dist.partial_cmp(&(r * r)) != Some(Ordering::Greater)
        };
        if compat.circle_sets_inliers {
            *flag = inside;
        } else {
            *flag = *flag && inside;
        }
    }
    let (w, h) = (image.width() as f64, image.height() as f64);
    for (p, flag) in distorted.iter().zip(inlier.iter_mut()) {
        if p.x < threshold_border
            || p.y < threshold_border
            || w - p.x < threshold_border
            || h - p.y < threshold_border
        {
            *flag = false;
        }
    }
}

/// Index of the point of `points` nearest to `(x, y)` with squared distance below
/// `max_dist_square` (strict); of equally near points the first in list order wins, as
/// only a strictly smaller distance replaces the bound.
///
/// Ported from core/UndistortionObject.cpp, `UndistortionObject::findClosestPoint`.
#[must_use]
pub fn find_closest_point(
    points: &[Point2],
    x: f64,
    y: f64,
    max_dist_square: f64,
) -> Option<usize> {
    let mut idx = None;
    let mut bound = max_dist_square;
    for (count, p) in points.iter().enumerate() {
        let xx = p.x - x;
        let yy = p.y - y;
        let d = xx * xx + yy * yy;
        if d < bound {
            idx = Some(count);
            bound = d;
        }
    }
    idx
}

/// A click in mode "toggle outlier" (C6): the nearest point of the displayed list
/// (`points`: the distorted, undistorted or reference grid points, all the same length as
/// `inlier`) within squared distance 100 has its inlier flag inverted. Returns its index;
/// the caller then sets the pending recompute with [`PendingRecompute::after_toggle`].
/// `(x, y)` is the click in image pixels (the view has already subtracted 0.5).
///
/// Ported from core/UndistortionObject.cpp, `UndistortionObject::toggleOutlier`.
///
/// Panics if the index found is outside `inlier`.
pub fn toggle_outlier(points: &[Point2], inlier: &mut [bool], x: f64, y: f64) -> Option<usize> {
    let idx = find_closest_point(points, x, y, 100.0)?;
    inlier[idx] = !inlier[idx];
    Some(idx)
}
