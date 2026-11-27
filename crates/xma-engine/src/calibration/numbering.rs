//! Ledger A189 (default mode): cameras that number a checkerboard differently.
//!
//! **What the programs do**: the numberings are compared as described below, nothing is
//! renumbered and nothing is refused; a warning names the cameras and frames found to be
//! numbered differently (after a calibration once the pair's frames are all computed, and at
//! "optimize camera setup"), and a Console line at "optimize camera setup" names pairs whose
//! frames cannot show it. An automatic renumbering built on the same analysis was not
//! trusted (three versions each failed an independent check) and was removed on 21 November
//! 2026 (the author's decision 17); it remains in the history at commit c7d10f0.
//!
//! XMALab numbers a board's corners in each camera from the corner clicked there
//! (`sortGridByReference` keeps or reverses OpenCV's order, and OpenCV fixes the numbering's
//! handedness in the image). Cameras that number the same board from different corners, or
//! with opposite handedness (a camera that sees the board from behind, as an X-ray camera
//! on the far side does), place their poses of the board in different world frames: the
//! board's frame turned about its centre. Every number that joins the cameras is then wrong:
//! "optimize camera setup", the 3D points and rigid-body poses of every trial (the cameras'
//! poses at the trial's reference calibration frame) and the MayaCams export.
//!
//! # The comparison
//!
//! - **Candidates** ([`Renumbering`]): the board's symmetries, each a rigid motion of the
//!   board that takes its corners onto its corners: kept, reversed (a half turn about the
//!   board's normal), each row reversed and the rows in reverse order (half turns about the
//!   board's middle lines: the board seen from the other side); for a square board also the
//!   two quarter turns and the two diagonal half turns.
//! - **Distance**: for a frame that cameras `a` and `b` both see, each candidate `g` of
//!   `b`'s numbering gives a transform from `a` to `b`, `pose_b(f) G_g pose_a(f)^-1`. The
//!   set of these transforms does not depend on how either camera numbers the frame now; the
//!   right candidate's is the physical transform, and every other candidate's differs from
//!   it by a rotation of 90 or 180 degrees (the symmetry, conjugated), however the board
//!   lies. Transforms are compared by the angle of the rotation between them (degrees); the
//!   translation adds nothing to tell candidates apart and is what a single camera's planar
//!   pose determines worst (depth along the line of sight, on long-focal X-ray cameras).
//! - **When the frames determine the numbering.** A wrong candidate is the right one
//!   composed with a symmetry `S` of the board; its transforms are the right ones composed
//!   with `X_f S X_f^T` (`X_f`: the board's orientation in frame `f`, in the first camera),
//!   the turn `S` about the camera-frame axis `X_f s`. It fits every shared frame exactly
//!   when that turn is the same in every frame up to the board's symmetries, which depends
//!   only on the board's motion between frames (`X_0^T X_f`). Per symmetry: a half turn
//!   about the normal ("reversed"): every frame's normal parallel to the first's (the board
//!   only spun about its normal and moved), or one frame's row or column direction along
//!   the first's normal; a half turn about the row (column) direction ("rows in reverse
//!   order", "each row reversed"): that line shared by every frame (the board turned only
//!   about it), or lying along another frame's normal or other in-plane axis (the board
//!   turned 90 degrees about the third axis); for a square board the diagonal half turns
//!   likewise with the diagonals, and the quarter turns: the normals parallel and pointing
//!   alike. [`spread`] measures, per symmetry, the largest over the frames of the angle
//!   between `X_0 S X_0^T` and the nearest `X_f S' X_f^T`: 0 exactly in that set.
//! - **Noise**: the board's motion `X_0^T X_f` as each camera's own calibration determines
//!   it: the standard deviation of that rotation from the camera's calibration covariance
//!   ([`camera_covariance`]: the reprojection error's root mean square times
//!   `(J^T J)^-1`, over the intrinsics and every frame's pose, so that intrinsics poorly
//!   determined by too few frames or a degenerate motion make the motion uncertain; doubled,
//!   since a half turn's axis moved by an angle moves the turn by twice it).
//! - **Pairs**: two board cameras that both see at least two frames. The consensus: seeds
//!   from the candidates of the first, middle and last shared frames (a bounded number:
//!   every hypothesis is among one frame's candidates), each refined (each frame's nearest
//!   candidate, their mean rotation, three times); the seed whose largest residual is
//!   smallest. The pair is decided only when (1) in both cameras every rival symmetry is
//!   excluded by some frame whose angle is at least [`SPREAD_FLOOR`] and at least
//!   [`SPREAD_FACTOR`] times that frame's noise ([`PairNumbering::margin`]); (2) the
//!   smaller spread is at least [`SPREAD_FACTOR`] times the consensus's largest residual;
//!   (3) the camera-to-camera translations agree: the frames' second-camera centres (in
//!   the first camera, with the chosen candidates) lie within [`TRANSLATION_FACTOR`] times
//!   the consensus residual (in radians, at least [`TRANSLATION_NOISE_FLOOR`] degrees) of
//!   their mean, relative to its distance. Otherwise undecided ([`Undecided`]): the frames
//!   do not determine it (1), or they disagree (2, 3). Never a guess.
//! - **Sets of cameras**: the decided pairs join the board cameras into groups. In each
//!   group a spanning tree from the lowest-index camera (each camera reached from the
//!   lowest-index camera already in the tree that it has a decided pair with) gives every
//!   camera's transform from the group's first camera.
//! - **Frames**: each frame's numbering is compared with that of the lowest-index camera of
//!   the group that sees it (its anchor): for every other camera of the group that sees the
//!   frame, the candidate whose transform from the anchor is nearest the tree's is chosen
//!   when it is clear: the second nearest at least [`FLOOR`] away and at least [`RATIO`]
//!   times as far. Otherwise the frame is undecided. Pairs outside the tree are thereby
//!   checked through the tree's transforms.
//! - **A frame that one camera sees alone** is not compared: nothing ties its board pose to
//!   another camera (the board moves freely between frames), and nothing needs it: in
//!   "optimize camera setup" its board pose is a free parameter of that frame and the
//!   board's symmetry gives every numbering the same residuals.
//! - **Messages** (`after_calibration`, `before_optimization`): a warning per camera and
//!   camera it is compared with, naming the frames whose clear candidate is not the present
//!   numbering: after a calibration only once the pair is complete (every frame in which one
//!   of the two cameras has corners has them in the other too: no frame is left to click
//!   that could change it), and before "optimize camera setup"; there also a Console line
//!   ([`is_console_only`]) per pair or frame that cannot be decided.

use xma_model::{CalibrationObject, Matrix3, Point3, Vector3};
use xma_vision::camera::{nearest_rotation_to_rodrigues, rodrigues_to_matrix};

use super::all_frames::{n_frames, object_points};
use super::checkerboard::board;
use crate::document::Document;

/// A frame's candidate is clear only when the next one is at least this many times as far
/// from the transform the cameras' rig gives ...
pub const RATIO: f64 = 3.0;
/// ... and at least this far (degrees of rotation).
pub const FLOOR: f64 = 20.0;
/// A pair is decided only when the shared frames' board orientations are at least this many
/// times the noise (the consensus's largest residual) from every orientation set under which
/// a wrong candidate would fit as well ([`PairNumbering::rival`]) ...
pub const SPREAD_FACTOR: f64 = 3.0;
/// ... and at least this far (degrees).
pub const SPREAD_FLOOR: f64 = 10.0;
/// A decided pair's camera-to-camera translations must agree: the largest distance of a
/// frame's second camera centre from their mean, divided by the mean's distance from the
/// first camera, at most this many times the noise (the consensus's largest residual, in
/// radians, at least [`TRANSLATION_NOISE_FLOOR`]).
pub const TRANSLATION_FACTOR: f64 = 3.0;
/// The least noise (degrees) the translation check assumes.
pub const TRANSLATION_NOISE_FLOOR: f64 = 2.0;

/// A renumbering of a board's corners: one of the board's symmetries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Renumbering {
    /// As detected.
    Kept,
    /// The order reversed: a half turn about the board's normal.
    Reversed,
    /// Each row reversed: the board turned over about its middle column.
    RowsReversed,
    /// The rows in reverse order: the board turned over about its middle row.
    RowOrderReversed,
    /// A square board turned a quarter turn one way.
    QuarterTurn,
    /// A square board turned a quarter turn the other way.
    ThreeQuarterTurn,
    /// A square board turned over about its main diagonal (rows and columns exchanged).
    Transposed,
    /// A square board turned over about its other diagonal.
    AntiTransposed,
}

impl Renumbering {
    /// The candidates for a board of `h` corners per row and `v` rows.
    #[must_use]
    pub fn candidates(h: usize, v: usize) -> Vec<Renumbering> {
        let mut out = vec![
            Renumbering::Kept,
            Renumbering::Reversed,
            Renumbering::RowsReversed,
            Renumbering::RowOrderReversed,
        ];
        if h == v {
            out.extend([
                Renumbering::QuarterTurn,
                Renumbering::ThreeQuarterTurn,
                Renumbering::Transposed,
                Renumbering::AntiTransposed,
            ]);
        }
        out
    }

    /// The corner whose detection the renumbered corner `(i, j)` takes (`i` along a row).
    fn source(self, i: usize, j: usize, h: usize, v: usize) -> (usize, usize) {
        match self {
            Renumbering::Kept => (i, j),
            Renumbering::Reversed => (h - 1 - i, v - 1 - j),
            Renumbering::RowsReversed => (h - 1 - i, j),
            Renumbering::RowOrderReversed => (i, v - 1 - j),
            Renumbering::QuarterTurn => (j, h - 1 - i),
            Renumbering::ThreeQuarterTurn => (h - 1 - j, i),
            Renumbering::Transposed => (j, i),
            Renumbering::AntiTransposed => (h - 1 - j, h - 1 - i),
        }
    }

    /// For each new index, the old index it takes its corner from.
    #[must_use]
    pub fn permutation(self, h: usize, v: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(h * v);
        for j in 0..v {
            for i in 0..h {
                let (a, b) = self.source(i, j, h, v);
                out.push(b * h + a);
            }
        }
        out
    }
}

/// A rigid transform as a matrix and a vector (`x -> r x + t`).
#[derive(Clone, Copy, Debug)]
struct Rt {
    r: Matrix3,
    t: Vector3,
}

impl Rt {
    fn id() -> Rt {
        Rt {
            r: Matrix3::identity(),
            t: Vector3::zeros(),
        }
    }
    /// `self` after `b`.
    fn then(&self, b: &Rt) -> Rt {
        Rt {
            r: self.r * b.r,
            t: self.r * b.t + self.t,
        }
    }
    fn inv(&self) -> Rt {
        let r = self.r.transpose();
        Rt {
            r,
            t: -(r * self.t),
        }
    }
}

/// The mean of rigid transforms: the rotation nearest the sum of the rotations, the mean
/// translation. `None` when there is none or the sum has no nearest rotation.
fn mean(ts: &[Rt]) -> Option<Rt> {
    if ts.is_empty() {
        return None;
    }
    let mut r = Matrix3::zeros();
    let mut t = Vector3::zeros();
    for x in ts {
        r += x.r;
        t += x.t;
    }
    let rv = nearest_rotation_to_rodrigues(&r)?;
    #[allow(clippy::cast_precision_loss)]
    Some(Rt {
        r: rodrigues_to_matrix(&rv),
        t: t / ts.len() as f64,
    })
}

/// The rigid motion of the board that takes each object point to the one whose detection
/// the renumbered point takes: `G(P_k) = P_source(k)`. `None` when the board is too small
/// or its points are not a grid.
fn symmetry(g: Renumbering, pts: &[Point3], h: usize, v: usize) -> Option<Rt> {
    if h < 2 || v < 2 || pts.len() != h * v {
        return None;
    }
    let at = |i: usize, j: usize| pts[j * h + i].coords;
    let o = at(0, 0);
    let a = at(1, 0) - o;
    let b = at(0, 1) - o;
    let n = a.cross(&b);
    let nn = n.norm();
    if nn == 0.0 || !nn.is_finite() {
        return None;
    }
    let src = |i: usize, j: usize| {
        let (p, q) = g.source(i, j, h, v);
        at(p, q)
    };
    let t0 = src(0, 0);
    let ga = src(1, 0) - t0;
    let gb = src(0, 1) - t0;
    let gn = ga.cross(&gb) / nn;
    let n = n / nn;
    // A = [ga gb gn] [a b n]^-1, with a, b, n orthogonal.
    let a2 = a.norm_squared();
    let b2 = b.norm_squared();
    let r = ga * (a / a2).transpose() + gb * (b / b2).transpose() + gn * n.transpose();
    Some(Rt { r, t: t0 - r * o })
}

fn pose(im: &xma_model::CalibrationImage) -> Rt {
    Rt {
        r: rodrigues_to_matrix(&im.rotation),
        t: im.translation,
    }
}

/// The angle (degrees) of the rotation between two transforms' rotations; infinite when not
/// finite.
fn angle(a: &Rt, b: &Rt) -> f64 {
    let d = a.r * b.r.transpose();
    let c = ((d.trace() - 1.0) / 2.0).clamp(-1.0, 1.0);
    let x = c.acos().to_degrees();
    if x.is_finite() { x } else { f64::INFINITY }
}

/// The board and its candidates.
struct Ctx {
    h: usize,
    v: usize,
    pts: Vec<Point3>,
    syms: Vec<(Renumbering, Rt)>,
}

fn ctx(doc: &Document) -> Option<Ctx> {
    let (h, v) = board(doc)?;
    let pts = object_points(doc);
    let n = h * v;
    if pts.len() != n || h < 2 || v < 2 {
        return None;
    }
    let syms = Renumbering::candidates(h, v)
        .into_iter()
        .filter_map(|g| symmetry(g, &pts, h, v).map(|s| (g, s)))
        .collect();
    Some(Ctx { h, v, pts, syms })
}

/// Whether frame `f` of `camera` has a pose and every corner of the board.
fn posed(doc: &Document, camera: usize, f: usize, n: usize) -> bool {
    f < n_frames(doc, camera)
        && doc.project().cameras[camera]
            .calibration_images
            .get(f)
            .is_some_and(|im| im.calibrated > 0 && im.detected.len() == n)
}

/// Whether frame `f` of `camera` has every corner of the board (posed or not yet). A frame
/// emptied by "Reset Frame", or given manual mode's fresh lists, holds `n` unset points and
/// no detection: it has no corners (review 12; the test of `frames_with_corners`).
fn has_corners(doc: &Document, camera: usize, f: usize, n: usize) -> bool {
    f < n_frames(doc, camera)
        && doc.project().cameras[camera]
            .calibration_images
            .get(f)
            .is_some_and(|im| im.detected.len() == n && !im.detected_all.is_empty())
}

/// The calibrated board cameras, by index.
fn board_cameras(doc: &Document, n: usize) -> Vec<usize> {
    (0..doc.project().cameras.len())
        .filter(|&c| {
            doc.project().cameras[c].calibrated
                && (0..n_frames(doc, c)).any(|f| posed(doc, c, f, n))
        })
        .collect()
}

/// Frame `f` in cameras `a` and `b`: per candidate of `b`'s numbering, the transform from
/// `a` to `b`.
fn frame_candidates(doc: &Document, cx: &Ctx, a: usize, b: usize, f: usize) -> Vec<Rt> {
    let cams = &doc.project().cameras;
    let pa = pose(&cams[a].calibration_images[f]);
    let pb = pose(&cams[b].calibration_images[f]);
    let pa_inv = pa.inv();
    cx.syms
        .iter()
        .map(|(_, s)| pb.then(s).then(&pa_inv))
        .collect()
}

/// Why two cameras could not be related.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Undecided {
    /// The two cameras see fewer than two frames in common (and are not related through
    /// other cameras).
    FewFrames,
    /// The shared frames do not determine the numbering: the board's orientations in them
    /// are too close to a set under which a wrong candidate fits as well as the right one
    /// (the board turned only about one of its symmetry axes, or not turned at all).
    Ambiguous,
    /// The shared frames determine the numbering but disagree among themselves: no candidate
    /// fits every frame within the noise, or the camera-to-camera translations disagree.
    NoAgreement,
}

/// Two cameras' shared frames analysed ([`Document::checkerboard_pairs`]); angles in
/// degrees.
#[derive(Clone, Debug, PartialEq)]
pub struct PairNumbering {
    pub a: usize,
    pub b: usize,
    /// The frames both cameras see with a pose and every corner.
    pub shared: Vec<usize>,
    /// Whether the transform between the two cameras is decided.
    pub decided: bool,
    pub undecided: Option<Undecided>,
    /// The noise: the consensus's largest residual (the angle of a frame's nearest
    /// candidate from it).
    pub fit: f64,
    /// The spread: how far the shared frames' board orientations are from every set under
    /// which a wrong candidate fits as well as the right one (the smaller of the two
    /// cameras').
    pub rival: f64,
    /// The translations' disagreement: the largest distance of a frame's second-camera
    /// centre (in the first camera's coordinates, with the chosen candidates) from their
    /// mean, divided by the mean's distance.
    pub translation: f64,
    /// Twice the larger of the two cameras' largest standard deviation of the board's motion
    /// from the first shared frame to another (degrees): the noise the spread is compared
    /// with.
    pub orientation_noise: f64,
    /// For the rival nearest to fitting, the largest over the frames of its angle divided by
    /// that frame's noise (0 for a frame under [`SPREAD_FLOOR`]), the smaller of the two
    /// cameras': the pair is determined when it is at least [`SPREAD_FACTOR`].
    pub margin: f64,
}

/// The analysis of two cameras' shared frames.
#[derive(Clone, Debug)]
struct Pair {
    a: usize,
    b: usize,
    shared: Vec<usize>,
    /// The transform from `a` to `b`, when decided.
    transform: Option<Rt>,
    undecided: Option<Undecided>,
    fit: f64,
    rival: f64,
    translation: f64,
    orientation_noise: f64,
    margin: f64,
    /// The shared frames farthest from the consensus (for a message).
    worst: Vec<usize>,
}

/// How far a camera's board orientations `x` (one per shared frame) are from every set
/// under which a wrong candidate fits as well as the right one. A wrong candidate composes
/// the right one with a symmetry `S` of the board; its camera-to-camera rotations are the
/// right ones composed with `X_f S X_f^T`, the turn `S` about the board axis `X_f s`, so it
/// fits every frame exactly when that turn is the same for every frame up to the board's
/// symmetries: when one camera-frame axis is the same symmetry axis of the board, with the
/// same turn, in every frame (for a half turn about the normal: the normals parallel; about
/// a row or column direction, or a diagonal of a square board: that line shared, or, since
/// the half turns about two axes at right angles compose, one frame's line along another's
/// other symmetry axis; for a quarter turn: the normals parallel and pointing alike). The
/// measure: over the symmetries `S` other than "kept", the smallest of the largest, over
/// the frames, angle (degrees) between `X_0 S X_0^T` and the nearest `X_f S' X_f^T`.
/// Taking the first frame as the reference makes it at most twice the smallest over all
/// reference frames.
fn spread(cx: &Ctx, x: &[Matrix3], noise: &[f64]) -> (f64, f64) {
    let Some(x0) = x.first() else {
        return (0.0, 0.0);
    };
    let turn = |xf: &Matrix3, s: &Rt| Rt {
        r: xf * s.r * xf.transpose(),
        t: Vector3::zeros(),
    };
    // Per rival symmetry: the largest angle over the frames, and the largest ratio of a
    // frame's angle to its noise (each at least the floor's share).
    let mut best = (f64::INFINITY, f64::INFINITY);
    for (g, s) in &cx.syms {
        if *g == Renumbering::Kept {
            continue;
        }
        let w = turn(x0, s);
        let mut worst = (0.0f64, 0.0f64);
        for (k, xf) in x.iter().enumerate() {
            let d = cx
                .syms
                .iter()
                .map(|(_, s2)| angle(&w, &turn(xf, s2)))
                .fold(f64::INFINITY, f64::min);
            let z = if d >= SPREAD_FLOOR {
                d / noise.get(k).copied().unwrap_or(f64::INFINITY).max(1e-9)
            } else {
                0.0
            };
            worst = (worst.0.max(d), worst.1.max(z));
        }
        best = (best.0.min(worst.0), best.1.min(worst.1));
    }
    best
}

/// A camera's calibration covariance, blockwise ([`camera_covariance`]).
struct CamCov {
    /// The camera's root mean square reprojection error (px).
    sigma: f64,
    /// Per frame: `V^-1` (the pose block's inverse, 6 x 6) and `M = W V^-1` (4 x 6).
    frames: Vec<Option<([f64; 36], [f64; 24])>>,
    /// `S^-1`, the inverse of the intrinsics' Schur complement (4 x 4).
    sinv: Option<Vec<f64>>,
}

impl CamCov {
    /// The covariance (unit sigma) of the rotation parts of poses `i` and `j` (3 x 3).
    fn rot(&self, i: usize, j: usize) -> Option<[f64; 9]> {
        let (vi, mi) = self.frames.get(i)?.as_ref()?;
        let (_, mj) = self.frames.get(j)?.as_ref()?;
        let sinv = self.sinv.as_ref()?;
        let mut out = [0.0f64; 9];
        for a in 0..3 {
            for b in 0..3 {
                let mut x: f64 = (0..4)
                    .map(|p| {
                        (0..4)
                            .map(|q| mi[p * 6 + a] * sinv[p * 4 + q] * mj[q * 6 + b])
                            .sum::<f64>()
                    })
                    .sum();
                if i == j {
                    x += vi[a * 6 + b];
                }
                out[a * 3 + b] = x;
            }
        }
        Some(out)
    }

    /// The standard deviation (degrees) of the rotation from pose `i` to pose `j` (the
    /// board's motion between the two frames, as this camera measures it): the largest
    /// direction of `cov_ii + cov_jj - cov_ij - cov_ji`; infinite when not available.
    fn relative(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 0.0;
        }
        let (Some(a), Some(b), Some(c)) = (self.rot(i, i), self.rot(j, j), self.rot(i, j)) else {
            return f64::INFINITY;
        };
        let mut m = [0.0f64; 9];
        for p in 0..3 {
            for q in 0..3 {
                m[p * 3 + q] = a[p * 3 + q] + b[p * 3 + q] - c[p * 3 + q] - c[q * 3 + p];
            }
        }
        if m.iter().any(|x| !x.is_finite()) {
            return f64::INFINITY;
        }
        let (w, _) = xma_vision::linalg::eigen_jacobi(&m, 3);
        let lmax = w.iter().copied().fold(0.0, f64::max);
        let x = (self.sigma * lmax.sqrt()).to_degrees();
        if x.is_finite() { x } else { f64::INFINITY }
    }
}

/// `camera`'s calibration covariance: `sigma^2 (J^T J)^-1`, where `J` holds the derivatives
/// of every posed frame's projected inlier corners (pinhole, the camera matrix) by fx, fy,
/// cx, cy and each frame's pose (a rotation applied on the left, in the camera's
/// coordinates, and the translation), and `sigma` is the camera's root mean square
/// reprojection error over those frames (undistorted corners; at least 0.05 px). The
/// intrinsics are shared by the frames, so a motion that leaves them poorly determined
/// makes the poses uncertain together; the board's motion between two frames
/// ([`CamCov::relative`]) keeps what the shared intrinsics move alike out. Blockwise (the
/// intrinsics' Schur complement), linear in the frames. The lens coefficients are taken as
/// known.
fn camera_covariance(doc: &Document, cx: &Ctx, camera: usize) -> CamCov {
    let n = cx.h * cx.v;
    let cam = &doc.project().cameras[camera];
    let k = cam.camera_matrix;
    let model = &doc.camera(camera).model;
    let frames = n_frames(doc, camera);
    let project = |r: &Matrix3, t: &Vector3, p: &Vector3| -> ([f64; 2], [f64; 2]) {
        let q = r * p + t;
        let (x, y) = (q.x / q.z, q.y / q.z);
        (
            [k[(0, 0)] * x + k[(0, 2)], k[(1, 1)] * y + k[(1, 2)]],
            [x, y],
        )
    };
    type Blocks = ([f64; 16], [f64; 24], [f64; 36]);
    let mut blocks: Vec<Option<Blocks>> = Vec::with_capacity(frames);
    let (mut ss, mut count) = (0.0, 0usize);
    for f in 0..frames {
        if !posed(doc, camera, f, n) {
            blocks.push(None);
            continue;
        }
        let im = &cam.calibration_images[f];
        let (r0, t) = (rodrigues_to_matrix(&im.rotation), im.translation);
        let (mut u, mut w, mut v) = ([0.0f64; 16], [0.0f64; 24], [0.0f64; 36]);
        let step = [1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4];
        let mut perturbed = Vec::with_capacity(6);
        for (j, h) in step.iter().enumerate() {
            let mut e = Vector3::zeros();
            let (mut t1, mut t2) = (t, t);
            let (r1, r2) = if j < 3 {
                e[j] = *h;
                (
                    rodrigues_to_matrix(&e) * r0,
                    rodrigues_to_matrix(&(-e)) * r0,
                )
            } else {
                t1[j - 3] += h;
                t2[j - 3] -= h;
                (r0, r0)
            };
            perturbed.push((r1, t1, r2, t2, *h));
        }
        for ((p, d), _) in cx
            .pts
            .iter()
            .zip(&im.detected)
            .zip(&im.inlier)
            .filter(|(_, i)| **i > 0)
        {
            let p = p.coords;
            let d = model.undistort_point(*d, true, false, true);
            let (e, xy) = project(&r0, &t, &p);
            ss += (e[0] - d.x).powi(2) + (e[1] - d.y).powi(2);
            count += 1;
            let ji = [[xy[0], 0.0, 1.0, 0.0], [0.0, xy[1], 0.0, 1.0]];
            let mut jp = [[0.0f64; 6]; 2];
            for (j, (r1, t1, r2, t2, h)) in perturbed.iter().enumerate() {
                let (a, _) = project(r1, t1, &p);
                let (c, _) = project(r2, t2, &p);
                jp[0][j] = (a[0] - c[0]) / (2.0 * h);
                jp[1][j] = (a[1] - c[1]) / (2.0 * h);
            }
            for r in 0..2 {
                for i in 0..4 {
                    for j in 0..4 {
                        u[i * 4 + j] += ji[r][i] * ji[r][j];
                    }
                    for j in 0..6 {
                        w[i * 6 + j] += ji[r][i] * jp[r][j];
                    }
                }
                for i in 0..6 {
                    for j in 0..6 {
                        v[i * 6 + j] += jp[r][i] * jp[r][j];
                    }
                }
            }
        }
        blocks.push(Some((u, w, v)));
    }
    let posed_frames = blocks.iter().filter(|b| b.is_some()).count();
    #[allow(clippy::cast_precision_loss)]
    let dof = (2 * count).saturating_sub(4 + 6 * posed_frames).max(1) as f64;
    let sigma = (ss / dof).sqrt().max(0.05);
    let finite = |m: &[f64]| m.iter().all(|x| x.is_finite());
    let mut s = [0.0f64; 16];
    let mut out_frames = Vec::with_capacity(frames);
    let mut ok = true;
    for b in &blocks {
        let Some((u, w, v)) = b else {
            out_frames.push(None);
            continue;
        };
        for i in 0..16 {
            s[i] += u[i];
        }
        if !finite(v) || !finite(w) {
            ok = false;
            out_frames.push(None);
            continue;
        }
        let vi = xma_vision::linalg::invert_eig(v, 6);
        let mut viv = [0.0f64; 36];
        viv.copy_from_slice(&vi[..36]);
        let mut m = [0.0f64; 24];
        for i in 0..4 {
            for j in 0..6 {
                m[i * 6 + j] = (0..6).map(|q| w[i * 6 + q] * viv[q * 6 + j]).sum();
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                s[i * 4 + j] -= (0..6).map(|q| m[i * 6 + q] * w[j * 6 + q]).sum::<f64>();
            }
        }
        out_frames.push(Some((viv, m)));
    }
    let sinv = (ok && finite(&s)).then(|| xma_vision::linalg::invert_eig(&s, 4));
    CamCov {
        sigma,
        frames: out_frames,
        sinv,
    }
}

fn analyse_pair(doc: &Document, cx: &Ctx, cov: &[Option<CamCov>], a: usize, b: usize) -> Pair {
    let n = cx.h * cx.v;
    let frames = n_frames(doc, a).min(n_frames(doc, b));
    let shared: Vec<usize> = (0..frames)
        .filter(|&f| posed(doc, a, f, n) && posed(doc, b, f, n))
        .collect();
    let mut out = Pair {
        a,
        b,
        shared: shared.clone(),
        transform: None,
        undecided: Some(Undecided::FewFrames),
        fit: f64::INFINITY,
        rival: 0.0,
        translation: f64::INFINITY,
        orientation_noise: f64::INFINITY,
        margin: 0.0,
        worst: Vec::new(),
    };
    if shared.len() < 2 {
        return out;
    }
    let cams = &doc.project().cameras;
    let pa: Vec<Rt> = shared
        .iter()
        .map(|&f| pose(&cams[a].calibration_images[f]))
        .collect();
    let pb: Vec<Rt> = shared
        .iter()
        .map(|&f| pose(&cams[b].calibration_images[f]))
        .collect();
    let data: Vec<Vec<Rt>> = (0..shared.len())
        .map(|k| {
            let inv = pa[k].inv();
            cx.syms
                .iter()
                .map(|(_, s)| pb[k].then(s).then(&inv))
                .collect()
        })
        .collect();
    let nearest = |hyp: &Rt, k: usize| -> (usize, f64) {
        data[k]
            .iter()
            .enumerate()
            .map(|(g, t)| (g, angle(t, hyp)))
            .fold(
                (0, f64::INFINITY),
                |acc, x| if x.1 < acc.1 { x } else { acc },
            )
    };
    // Seeds: every candidate of the first, middle and last shared frames (every hypothesis
    // is among one frame's candidates), each refined: every frame's nearest candidate, their
    // mean, three times. The consensus: the seed whose largest residual is smallest (every
    // frame must fit it).
    let mut seed_frames = vec![0, shared.len() / 2, shared.len() - 1];
    seed_frames.dedup();
    let mut best: Option<(f64, Rt, Vec<f64>)> = None;
    for &k0 in &seed_frames {
        for seed in &data[k0] {
            let mut c = *seed;
            for _ in 0..3 {
                let near: Vec<Rt> = (0..data.len()).map(|k| data[k][nearest(&c, k).0]).collect();
                if let Some(m) = mean(&near) {
                    c = m;
                }
            }
            let res: Vec<f64> = (0..data.len()).map(|k| nearest(&c, k).1).collect();
            let worst = res.iter().copied().fold(0.0, f64::max);
            if best.as_ref().is_none_or(|b| worst < b.0) {
                best = Some((worst, c, res));
            }
        }
    }
    let Some((noise, consensus, res)) = best else {
        return out;
    };
    // The degeneracy of the board's motion, in both cameras.
    let xa: Vec<Matrix3> = pa.iter().map(|p| p.r).collect();
    let xb: Vec<Matrix3> = pb.iter().map(|p| p.r).collect();
    // Each camera's noise per frame: the board's motion from the first shared frame to it,
    // as the camera's own calibration determines it (standard deviation); a half turn's
    // axis moved by an angle moves the turn by twice it.
    let motion_noise = |c: usize| -> Vec<f64> {
        let Some(cv) = cov.get(c).and_then(Option::as_ref) else {
            return vec![f64::INFINITY; shared.len()];
        };
        shared
            .iter()
            .map(|&f| 2.0 * cv.relative(shared[0], f))
            .collect()
    };
    let (nva, nvb) = (motion_noise(a), motion_noise(b));
    let ((da, za), (db, zb)) = (spread(cx, &xa, &nva), spread(cx, &xb, &nvb));
    let (na, nb) = (
        nva.iter().copied().fold(0.0, f64::max),
        nvb.iter().copied().fold(0.0, f64::max),
    );
    let rival = da.min(db);
    // The translations with each frame's chosen candidate: the second camera's centre in
    // the first camera's coordinates.
    let centres: Vec<Vector3> = (0..data.len())
        .map(|k| {
            let t = data[k][nearest(&consensus, k).0];
            -(t.r.transpose() * t.t)
        })
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let mc = centres.iter().fold(Vector3::zeros(), |acc, c| acc + c) / centres.len() as f64;
    let translation = centres.iter().map(|c| (c - mc).norm()).fold(0.0, f64::max) / mc.norm();
    let translation = if translation.is_finite() {
        translation
    } else {
        f64::INFINITY
    };
    out.fit = noise;
    out.rival = rival;
    out.translation = translation;
    let mut order: Vec<usize> = (0..res.len()).collect();
    order.sort_by(|&x, &y| res[y].total_cmp(&res[x]));
    out.worst = order
        .iter()
        .take_while(|&&k| res[k] >= 0.5 * noise)
        .map(|&k| shared[k])
        .collect();
    out.worst.sort_unstable();
    // Every rival excluded in both cameras: some frame's angle at least the floor and at
    // least the factor times that frame's noise.
    let determining = za >= SPREAD_FACTOR && zb >= SPREAD_FACTOR;
    out.margin = za.min(zb);
    // And the consensus's own residual well under the spread.
    let determined = determining && rival >= SPREAD_FACTOR * noise;
    out.orientation_noise = na.max(nb);
    let agreeing =
        translation <= TRANSLATION_FACTOR * noise.max(TRANSLATION_NOISE_FLOOR).to_radians();
    if determined && agreeing {
        out.transform = Some(consensus);
        out.undecided = None;
    } else if !determining {
        out.undecided = Some(Undecided::Ambiguous);
    } else {
        out.undecided = Some(Undecided::NoAgreement);
    }
    out
}

/// One frame of a camera compared with its anchor (the group's lowest-index camera that
/// sees it).
#[derive(Clone, Debug, PartialEq)]
pub struct FrameNumbering {
    /// The calibration frame (0-based).
    pub frame: usize,
    /// For each candidate, the angle (degrees) between its transform from the anchor and the
    /// one the group's cameras give, in the order of [`Renumbering::candidates`].
    pub distances: Vec<(Renumbering, f64)>,
    /// The candidate chosen, when it is clear.
    pub chosen: Option<Renumbering>,
}

/// A camera's numbering of the board compared with another camera's.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraNumbering {
    pub camera: usize,
    /// The camera whose numbering the frames are matched to (in the same group: the anchor
    /// of these frames), or, when the two cameras cannot be related, the other camera.
    pub reference: usize,
    /// The frames both cameras see.
    pub shared: Vec<usize>,
    /// Per frame matched to `reference`, the candidates (empty when undecided).
    pub frames: Vec<FrameNumbering>,
    /// Why the two cameras cannot be related (`None`: they are in one group).
    pub undecided: Option<Undecided>,
    /// When they cannot be related because their frames disagree: the shared frames
    /// farthest from the best transform.
    pub worst: Vec<usize>,
}

impl CameraNumbering {
    /// The frames with no clear candidate.
    #[must_use]
    pub fn undecided_frames(&self) -> Vec<usize> {
        self.frames
            .iter()
            .filter(|f| f.chosen.is_none())
            .map(|f| f.frame)
            .collect()
    }
}

/// Every board camera's numbering compared with its anchor's, frame by frame, and the pairs
/// of cameras that share frames but cannot be related.
fn plan(doc: &Document) -> Option<Vec<CameraNumbering>> {
    let cx = ctx(doc)?;
    let n = cx.h * cx.v;
    let cams = board_cameras(doc, n);
    if cams.len() < 2 {
        return Some(Vec::new());
    }
    let cov: Vec<Option<CamCov>> = (0..doc.project().cameras.len())
        .map(|c| cams.contains(&c).then(|| camera_covariance(doc, &cx, c)))
        .collect();
    let mut pairs: Vec<Pair> = Vec::new();
    for (i, &a) in cams.iter().enumerate() {
        for &b in &cams[i + 1..] {
            pairs.push(analyse_pair(doc, &cx, &cov, a, b));
        }
    }
    let pair = |a: usize, b: usize| pairs.iter().find(|p| (p.a, p.b) == (a.min(b), a.max(b)));
    // Groups and the tree: each camera's transform from its group's first camera.
    let mut rig: Vec<Option<Rt>> = vec![None; doc.project().cameras.len()];
    let mut group_of: Vec<Option<usize>> = vec![None; doc.project().cameras.len()];
    let mut groups = 0;
    for &root in &cams {
        if group_of[root].is_some() {
            continue;
        }
        let g = groups;
        groups += 1;
        group_of[root] = Some(g);
        rig[root] = Some(Rt::id());
        let mut members = vec![root];
        loop {
            // The lowest-index camera not yet in the tree that has a decided pair with a
            // camera in it, reached from the lowest-index such camera.
            let next = cams
                .iter()
                .copied()
                .filter(|&c| group_of[c].is_none())
                .find_map(|c| {
                    members
                        .iter()
                        .copied()
                        .filter(|&m| pair(m, c).is_some_and(|p| p.transform.is_some()))
                        .min()
                        .map(|m| (c, m))
                });
            let Some((c, m)) = next else { break };
            let p = pair(m, c).expect("pair");
            let t = p.transform.expect("decided");
            // `t` maps the lower camera's coordinates to the higher one's.
            let m_to_c = if p.a == m { t } else { t.inv() };
            rig[c] = rig[m].map(|r| m_to_c.then(&r));
            group_of[c] = Some(g);
            members.push(c);
        }
    }
    // Per frame: the anchor of each group, and every other camera of it matched.
    let mut entries: Vec<CameraNumbering> = Vec::new();
    let frames = cams.iter().map(|&c| n_frames(doc, c)).max().unwrap_or(0);
    for f in 0..frames {
        let seeing: Vec<usize> = cams
            .iter()
            .copied()
            .filter(|&c| posed(doc, c, f, n))
            .collect();
        for &c in &seeing {
            let Some(anchor) = seeing.iter().copied().find(|&a| group_of[a] == group_of[c]) else {
                continue;
            };
            if anchor == c {
                continue;
            }
            let (Some(ra), Some(rc)) = (rig[anchor], rig[c]) else {
                continue;
            };
            let predicted = rc.then(&ra.inv());
            let cands = frame_candidates(doc, &cx, anchor, c, f);
            let distances: Vec<(Renumbering, f64)> = cx
                .syms
                .iter()
                .zip(&cands)
                .map(|((g, _), t)| (*g, angle(t, &predicted)))
                .collect();
            let mut order: Vec<usize> = (0..distances.len()).collect();
            order.sort_by(|&x, &y| distances[x].1.total_cmp(&distances[y].1));
            let d1 = distances[order[0]].1;
            let d2 = order.get(1).map_or(f64::INFINITY, |&k| distances[k].1);
            let chosen = (d2 >= FLOOR && d2 >= RATIO * d1).then_some(distances[order[0]].0);
            let entry = match entries
                .iter_mut()
                .position(|e| e.camera == c && e.reference == anchor && e.undecided.is_none())
            {
                Some(i) => &mut entries[i],
                None => {
                    entries.push(CameraNumbering {
                        camera: c,
                        reference: anchor,
                        shared: pair(anchor, c)
                            .map(|p| p.shared.clone())
                            .unwrap_or_default(),
                        frames: Vec::new(),
                        undecided: None,
                        worst: Vec::new(),
                    });
                    entries.last_mut().expect("pushed")
                }
            };
            entry.frames.push(FrameNumbering {
                frame: f,
                distances,
                chosen,
            });
        }
    }
    // Pairs that share frames but are in different groups.
    for p in &pairs {
        if group_of[p.a] != group_of[p.b] && !p.shared.is_empty() {
            entries.push(CameraNumbering {
                camera: p.b,
                reference: p.a,
                shared: p.shared.clone(),
                frames: Vec::new(),
                undecided: Some(p.undecided.unwrap_or(Undecided::FewFrames)),
                worst: p.worst.clone(),
            });
        }
    }
    entries.sort_by_key(|e| (e.camera, e.reference, e.undecided.is_some()));
    Some(entries)
}

fn frame_list(frames: &[usize]) -> String {
    let s: Vec<String> = frames.iter().map(|f| (f + 1).to_string()).collect();
    match s.as_slice() {
        [one] => format!("frame {one}"),
        [rest @ .., last] => format!("frames {} and {last}", rest.join(", ")),
        [] => String::new(),
    }
}

/// Whether a message of a calibration edit is a Console line only (the frames cannot show
/// whether two cameras number the board alike), not one that the window also shows as an
/// error dialog.
#[must_use]
pub fn is_console_only(message: &str) -> bool {
    message.contains(NOT_SHOWN)
}

/// The words of every warning that two cameras number the board differently.
const DIFFERENTLY: &str = "numbers the checkerboard's corners differently from";
/// The words of every Console line that says the frames cannot show whether two cameras
/// number the board alike ([`is_console_only`]).
const NOT_SHOWN: &str = "do not show whether";

/// Whether a message of a calibration edit is the warning that two cameras number the
/// checkerboard differently.
#[must_use]
pub fn is_numbering_warning(message: &str) -> bool {
    message.contains(DIFFERENTLY)
}

/// The plan read, nothing changed. One warning per camera and camera it is compared with, naming the frames whose clear candidate is not the present
/// numbering (`only_complete`: only for pairs with no frame left to compute); with
/// `undecided`, one Console line per pair or frame that cannot be decided.
fn warn_only(doc: &Document, only_complete: bool, undecided: bool) -> Vec<String> {
    let Some(entries) = plan(doc) else {
        return Vec::new();
    };
    let n = board(doc).map_or(0, |(h, v)| h * v);
    let name = |c: usize| doc.project().cameras[c].name.clone();
    let mut out = Vec::new();
    for e in &entries {
        if only_complete && !complete(doc, e.camera, e.reference, n) {
            continue;
        }
        let (c, r) = (name(e.camera), name(e.reference));
        let differ: Vec<usize> = if e.undecided.is_none() {
            e.frames
                .iter()
                .filter(|x| x.chosen.is_some_and(|g| g != Renumbering::Kept))
                .map(|x| x.frame)
                .collect()
        } else {
            Vec::new()
        };
        if !differ.is_empty() {
            push_once(
                &mut out,
                format!(
                    "Warning: {c} {DIFFERENTLY} {r} in {} (another corner was clicked there, or the camera sees the board from its other side). While they differ, \"optimize camera setup\" and the 3D points of trials are wrong. Detect the board again in those frames and click the same physical corner of the board in both cameras; if the camera setup was just optimized, undo it first.",
                    frame_list(&differ)
                ),
            );
        }
        if undecided && (e.undecided.is_some() || !e.undecided_frames().is_empty()) {
            push_once(
                &mut out,
                format!(
                    "The frames that {c} and {r} both see {NOT_SHOWN} the two cameras number the checkerboard alike. If the errors are much larger after \"optimize camera setup\", undo it and click the same physical corner of the board in both cameras."
                ),
            );
        }
    }
    out
}

/// Whether every frame in which one of the two cameras has corners has them in the other:
/// no frame is left to compute that could decide the pair.
fn complete(doc: &Document, a: usize, b: usize, n: usize) -> bool {
    let frames = n_frames(doc, a).max(n_frames(doc, b));
    (0..frames).all(|f| has_corners(doc, a, f, n) == has_corners(doc, b, f, n))
}

/// Whether the project's object is a checkerboard and the default mode compares numberings.
fn active(doc: &Document) -> bool {
    !doc.compat().checkerboard_numbering_per_camera
        && matches!(
            doc.project().calibration_object,
            Some(CalibrationObject::Checkerboard { .. })
        )
}

fn push_once(out: &mut Vec<String>, m: String) {
    if !out.contains(&m) {
        out.push(m);
    }
}

/// After a board camera was calibrated: the warnings for complete pairs of cameras that
/// number the board differently ([`warn_only`]).
pub(crate) fn after_calibration(doc: &Document) -> Vec<String> {
    if !active(doc) {
        return Vec::new();
    }
    warn_only(doc, true, false)
}

/// Before "optimize camera setup": the warnings, and a Console line per pair or frame that
/// cannot be decided ([`warn_only`]). Nothing is changed and nothing is refused.
pub(crate) fn before_optimization(doc: &Document) -> Vec<String> {
    if !active(doc) {
        return Vec::new();
    }
    warn_only(doc, false, true)
}

impl Document {
    /// Ledger A189: every pair of calibrated board cameras, analysed (read only).
    #[must_use]
    pub fn checkerboard_pairs(&self) -> Vec<PairNumbering> {
        let Some(cx) = ctx(self) else {
            return Vec::new();
        };
        let cams = board_cameras(self, cx.h * cx.v);
        let cov: Vec<Option<CamCov>> = (0..self.project().cameras.len())
            .map(|c| cams.contains(&c).then(|| camera_covariance(self, &cx, c)))
            .collect();
        let mut out = Vec::new();
        for (i, &a) in cams.iter().enumerate() {
            for &b in &cams[i + 1..] {
                let p = analyse_pair(self, &cx, &cov, a, b);
                out.push(PairNumbering {
                    a,
                    b,
                    shared: p.shared,
                    decided: p.transform.is_some(),
                    undecided: p.undecided,
                    fit: p.fit,
                    rival: p.rival,
                    translation: p.translation,
                    orientation_noise: p.orientation_noise,
                    margin: p.margin,
                });
            }
        }
        out
    }

    /// Ledger A189: every calibrated board camera's numbering of the checkerboard compared,
    /// frame by frame, with the numbering of the lowest-index camera of its group that sees
    /// the frame, with every candidate renumbering; and the pairs of cameras that see frames
    /// in common but cannot be related. Empty for a project without a checkerboard or with
    /// fewer than two calibrated board cameras. Read only.
    #[must_use]
    pub fn checkerboard_numbering(&self) -> Vec<CameraNumbering> {
        plan(self).unwrap_or_default()
    }
}
