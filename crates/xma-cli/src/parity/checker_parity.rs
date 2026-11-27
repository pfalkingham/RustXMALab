//! The checkerboard calibration runs (`oracle/calibration_runs.json`, `checker_runs`; k01 to
//! k10): the named tolerances their numbers are compared within, and the comparison of
//! their hook tables (`hooks/checker_*.csv`, `hooks/calibration_all_frames*.csv`) with
//! `xma_vision`'s routines on the recorded inputs.
//!
//! # Tolerances
//!
//! A checkerboard camera is calibrated by the planar path of `cv::calibrateCamera` (no
//! intrinsic guess, 54 or 35 points per view), which agrees with the reference OpenCV to
//! rounding, not to the bit (ledger A172: the homographies' eigenvectors come from Eigen in
//! the reference build; from 50 points per view its `J^T J` goes to BLAS and the normal
//! equations to LAPACK). Everything computed from the calibration then carries that
//! rounding: the camera matrices and poses, the projections and errors, the exports, the
//! trials' projection matrices. Four tolerances (ledger A201), chosen by the state of the
//! reference (the C++ dump of that point, [`reference_state`]; never the document under
//! test) when a table is compared, each the largest difference measured on the ten runs,
//! rounded up to one significant digit:
//!
//! - [`TOLERANCE_A172_PLANAR`]: calibrations without a lens model.
//! - [`TOLERANCE_A172_RATIONAL`]: a camera with the lens model on (`CALIB_RATIONAL_MODEL`:
//!   eight coefficients that a flat board determines poorly, so the 30 iterations of the
//!   solver end at different places; A172).
//! - [`TOLERANCE_A172_MULTICAM`]: after "optimize camera setup" (levmar runs its whole
//!   iteration count: its stopping thresholds of 1e-30 are never reached, section 5 of
//!   `docs/cpp-calibration.md`, and the problem has no gauge constraint, so a starting point
//!   that differs by rounding drifts along the free directions).
//! - [`TOLERANCE_A189_MULTICAM`]: after the optimisation of cameras that number the board in
//!   opposite directions (k01, k07; ledger A189): no consistent solution, the optimisation
//!   ends at a mean error of 47 px, far from any minimum. Chosen when the mean
//!   reprojection error after an optimisation is above 1 px.
//!
//! The relative bounds go with absolute ones (`NamedTolerance::abs`) for values near 0, whose
//! relative difference says nothing: a table is within when its largest relative difference
//! or its largest absolute difference is within.
//!
//! A table gets its tolerance only when every difference in it is a number (same rows, same
//! texts) and its largest relative or absolute difference is within the bound. The hook
//! tables are compared value by value ([`hook_points`]): the detections, refinements,
//! predictions and sortings to the bit, the homographies within
//! [`TOLERANCE_A172_HOMOGRAPHY`], the calibrations over all frames within the planar
//! tolerances.

use std::path::Path;

use xma_engine::Document;

use super::{Item, NamedTolerance, Verdict};

/// Ledger A172, planar calibration without a lens model: camera matrices, poses, the
/// projections and errors computed from them, the exports and the trials' projection
/// matrices. Largest measured: 3.73e-6 (the RMS, 0.00137 px, of a calibration of k04 on the
/// homography's exact corners, replayed from the hook table); in the dumps and saved
/// projects 2.87e-6 (an entry of a rotation matrix of k05), camera matrices 1.9e-7,
/// translations 5.5e-7.
pub const TOLERANCE_A172_PLANAR: NamedTolerance = NamedTolerance {
    rule: "planar calibration (checkerboard), ledger A172 (homography eigenvectors by Eigen, \
           and from 50 points per view BLAS and LAPACK, in XMALab's OpenCV)",
    rel: 4e-6,
    abs: 0.0,
};

/// Ledger A172, planar calibration with the rational lens model (k07): relative 6e-2 or
/// absolute 3e-2 . Largest measured: relative 5.0e-2 (k4, 0.01460 against
/// 0.01537, of the first rational calibration of camera 1, from two views, replayed from the
/// hook table; its k3 30.99 against 29.83); in the dumps and saved projects relative 3.07e-3 (k3 of camera 1, five
/// views), absolute 2.18e-2 (the same coefficient, -7.10); the camera matrices 1.0e-7.
pub const TOLERANCE_A172_RATIONAL: NamedTolerance = NamedTolerance {
    rule: "planar calibration with the rational lens model, ledger A172 (eight coefficients \
           poorly determined by a flat board: the solver's 30 iterations end apart); \
           relative 6e-2 or absolute 3e-2",
    rel: 6e-2,
    abs: 3e-2,
};

/// Ledger A172 then the multi-camera optimisation of cameras that number the board alike
/// (k10): levmar's whole iteration count from starting values that differ by rounding,
/// without a gauge constraint. Relative 5e-4 or absolute 5e-3 . Largest
/// measured: relative 4.4e-4 (a pose), absolute 4.2e-3 px (a reprojection error).
pub const TOLERANCE_A172_MULTICAM: NamedTolerance = NamedTolerance {
    rule: "multi-camera optimisation after a planar calibration, ledger A172 (levmar runs \
           every iteration from starting values that differ by rounding; no gauge \
           constraint); relative 5e-4 or absolute 5e-3",
    rel: 5e-4,
    abs: 5e-3,
};

/// No tolerance: a saved project's camera files compared number by number (the item then
/// gets [`tolerate`]).
pub const NO_TOLERANCE: NamedTolerance = NamedTolerance {
    rule: "none",
    rel: 0.0,
    abs: 0.0,
};

/// Ledger A189 then the optimisation: the cameras number the board in opposite directions,
/// so the optimisation has no consistent solution (k01 and k07 end at 47 px); it runs every
/// iteration from starting values that differ by rounding and ends far apart. Relative 0.12
/// or absolute 0.4 . Largest measured: relative 0.117 (a rotation vector's
/// component of k01), absolute 0.372 px (a projected point of k07).
pub const TOLERANCE_A189_MULTICAM: NamedTolerance = NamedTolerance {
    rule: "multi-camera optimisation of cameras that number the board in opposite \
           directions, ledger A189 (no consistent solution: levmar's every iteration from \
           starting values that differ by rounding, ledger A172); relative 0.12 or absolute \
           0.4",
    rel: 0.12,
    abs: 0.4,
};

/// The state of the reference (the C++ program's) that chooses the tolerance: read from a
/// calibration dump ([`reference_state`]), never from the document under test, so that a
/// regression here cannot widen its own tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceState {
    /// A camera is optimised (`camera_state.csv`, `is_optimized`).
    pub optimized: bool,
    /// A camera has the lens model on (`camera_state.csv`, `has_model_distortion`).
    pub model_distortion: bool,
    /// The mean reprojection error (pixels) over the inliers of every calibrated image
    /// (`image_points.csv`, `error` where `inlier` > 0, of the images whose `is_calibrated`
    /// in `image_state.csv` is > 0).
    pub mean_error: f64,
}

/// The state of the C++ program in the calibration dump `dump` (its
/// `calibration/camera_state.csv`, `image_state.csv` and `image_points.csv`).
///
/// # Errors
/// A text naming the table the dump lacks or cannot be read.
pub fn reference_state(dump: &Path) -> Result<ReferenceState, String> {
    let read = |n: &str| {
        xma_oracle::Table::try_read(dump.join("calibration").join(n)).map_err(|e| {
            format!(
                "the C++ dump {} lacks a readable calibration/{n} ({e})",
                dump.display()
            )
        })
    };
    let cams = read("camera_state.csv")?;
    let images = read("image_state.csv")?;
    let points = read("image_points.csv")?;
    let e = |e: xma_oracle::OracleError| format!("{}: {e}", dump.display());
    let mut state = ReferenceState {
        optimized: false,
        model_distortion: false,
        mean_error: 0.0,
    };
    for r in cams.rows() {
        state.optimized |= r.try_i64("is_optimized").map_err(e)? != 0;
        state.model_distortion |= r.try_i64("has_model_distortion").map_err(e)? != 0;
    }
    let mut calibrated = std::collections::BTreeSet::new();
    for r in images.rows() {
        if r.try_i64("is_calibrated").map_err(e)? > 0 {
            calibrated.insert((
                r.try_i64("camera").map_err(e)?,
                r.try_i64("image").map_err(e)?,
            ));
        }
    }
    let (mut sum, mut n) = (0.0, 0usize);
    for r in points.rows() {
        let im = (
            r.try_i64("camera").map_err(e)?,
            r.try_i64("image").map_err(e)?,
        );
        let err = r.try_f64("error").map_err(e)?;
        if calibrated.contains(&im) && r.try_i64("inlier").map_err(e)? > 0 && err.is_finite() {
            sum += err;
            n += 1;
        }
    }
    state.mean_error = if n == 0 { 0.0 } else { sum / n as f64 };
    Ok(state)
}

/// The tolerance for the reference's state: after an optimisation, with a lens model, or
/// neither.
#[must_use]
pub fn tolerance_for(state: &ReferenceState) -> NamedTolerance {
    if state.optimized {
        if state.mean_error > 1.0 {
            TOLERANCE_A189_MULTICAM
        } else {
            TOLERANCE_A172_MULTICAM
        }
    } else if state.model_distortion {
        TOLERANCE_A172_RATIONAL
    } else {
        TOLERANCE_A172_PLANAR
    }
}

/// The tables computed from a calibration (relative to a dump directory, a saved project's
/// entry name, or an export file).
fn computed_from_calibration(name: &str) -> bool {
    let file = Path::new(name)
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    let file = file.rsplit(':').next().unwrap_or(&file).to_owned();
    matches!(
        file.as_str(),
        "camera_matrix.csv"
            | "calibration_images.csv"
            | "distortion.csv"
            | "camera_state.csv"
            | "image_state.csv"
            | "image_points.csv"
            | "camera_undistort_samples.csv"
            | "projection_matrices.csv"
            | "xma_metadata.xml"
    ) || file.ends_with("_CameraMatrix.csv")
        || file.ends_with("_RotationMatrix.csv")
        || file.ends_with("_TranslationVector.csv")
        || file.ends_with("_UndistortionParameter.csv")
        || (file.contains("_MayaCam") && file.ends_with(".txt"))
}

/// True when every problem of the item is a count of numbers outside the default
/// tolerance (no difference in rows, fields or texts).
fn numbers_only(item: &Item) -> bool {
    !item.problems.is_empty()
        && item.problems.iter().all(|p| {
            let mut w = p.split_whitespace();
            w.next().is_some_and(|n| n.parse::<usize>().is_ok())
                && w.next() == Some("of")
                && w.next().is_some_and(|n| n.parse::<usize>().is_ok())
                && w.next() == Some("values")
                && w.next() == Some("outside")
        })
}

/// The number of values a problem line counts as outside.
fn outside(item: &Item) -> usize {
    item.problems
        .iter()
        .filter_map(|p| p.split_whitespace().next()?.parse::<usize>().ok())
        .sum()
}

/// The tolerance of the reference's state ([`tolerance_for`]) applied to the items computed
/// from a calibration: a different item whose differences are numbers only, all within it,
/// is within tolerance. `reference` is the C++ program's state at that point, or why it
/// is not known; then nothing is tolerated and each such item says why.
pub fn tolerate(items: &mut [Item], reference: Result<&ReferenceState, &str>) {
    for item in items {
        if item.verdict != Verdict::Different
            || !computed_from_calibration(&item.name)
            || !numbers_only(item)
        {
            continue;
        }
        let t = match reference {
            Ok(s) => tolerance_for(s),
            Err(why) => {
                item.problems.push(format!(
                    "no tolerance of ledger A201 applied: the reference's state is not known \
                     here ({why})"
                ));
                continue;
            }
        };
        if item.max_rel > t.rel && item.max_abs > t.abs {
            continue;
        }
        let n = outside(item);
        let worst = item.max_rel;
        let before = item.tolerated.map_or(0, |x| x.1);
        let worst = item.tolerated.map_or(worst, |x| x.2.max(worst));
        item.notes.append(&mut item.problems);
        item.tolerated = Some((t, n + before, worst));
        item.verdict = Verdict::WithinTolerance;
    }
}

/// The rule of a reset camera's matrix (ledger A136).
const RULE_A136: &str = "the camera matrix of a camera reset by Reset Camera, Reset Frame or \
                         Delete Frame: re-created without values by the C++ (ledger A136; the \
                         identity here), not compared";

/// The rule of OpenCV's 14-coefficient output.
const RULE_14: &str = "coefficients 9 to 14 of a camera calibrated with the rational model: \
                       `cv::calibrateCamera` returns 14 (thin prism, tilt), which XMALab \
                       keeps, always 0; this program holds 8";

/// `camera_matrix.csv` and `distortion.csv` of a checkerboard run's dump, adjusted where the
/// C++ holds what the model does not (the indeterminate matrix of a reset camera, A136; the
/// six extra zero coefficients of a rational calibration), then compared as every table.
/// Returns `None` for another table.
#[must_use]
pub fn adjusted_item(
    ds: &xma_oracle::Dataset,
    rel: &str,
    mine: &str,
    doc: &Document,
) -> Option<Item> {
    if rel != "camera_matrix.csv" && rel != "distortion.csv" {
        return None;
    }
    let want = std::fs::read_to_string(ds.path(rel)).ok()?;
    let want_rows: Vec<Vec<&str>> = want
        .lines()
        .skip(1)
        .map(|l| l.split(',').collect())
        .collect();
    let mut out = String::new();
    let mut header = true;
    let mut adjusted = 0usize;
    if rel == "camera_matrix.csv" {
        for (i, line) in mine.lines().enumerate() {
            if header {
                out.push_str(line);
                out.push('\n');
                header = false;
                continue;
            }
            let f: Vec<&str> = line.split(',').collect();
            let cam: usize = f.first().and_then(|c| c.parse().ok()).unwrap_or(0);
            let reset = doc
                .project()
                .cameras
                .get(cam)
                .is_some_and(|c| !c.calibrated);
            match want_rows.get(i - 1) {
                Some(w) if reset && w.len() == f.len() && w[..3] == f[..3] && w[3] != f[3] => {
                    out.push_str(&w.join(","));
                    adjusted += 1;
                }
                _ => out.push_str(line),
            }
            out.push('\n');
        }
    } else {
        // distortion.csv: pad a camera's 8 coefficients with the C++'s rows 8 to 13, as 0.
        let mut rows: Vec<(usize, String)> = Vec::new();
        for line in mine.lines().skip(1) {
            let cam: usize = line
                .split(',')
                .next()
                .and_then(|c| c.parse().ok())
                .unwrap_or(0);
            rows.push((cam, line.to_owned()));
        }
        out.push_str(mine.lines().next().unwrap_or(""));
        out.push('\n');
        let n_cams = doc.project().cameras.len();
        for c in 0..n_cams {
            for (_, l) in rows.iter().filter(|(k, _)| *k == c) {
                out.push_str(l);
                out.push('\n');
            }
            let extra = want_rows
                .iter()
                .filter(|w| w.first().and_then(|x| x.parse::<usize>().ok()) == Some(c))
                .filter(|w| {
                    w.get(1)
                        .and_then(|x| x.parse::<usize>().ok())
                        .is_some_and(|k| k >= 8)
                })
                .count();
            for k in 8..8 + extra {
                out.push_str(&format!("{c},{k},0\n"));
                adjusted += 1;
            }
        }
    }
    let mut item = super::export_file(ds, rel, &out, super::Tier::Read);
    if adjusted > 0 {
        let (rule, note) = if rel == "camera_matrix.csv" {
            (
                RULE_A136,
                format!("{adjusted} entries of reset cameras not compared (ledger A136)"),
            )
        } else {
            (
                RULE_14,
                format!("{adjusted} coefficients beyond the 8 held here compared with 0"),
            )
        };
        if rel == "camera_matrix.csv" {
            item.excluded += adjusted;
        } else {
            item.explained += adjusted;
        }
        item.rules.push((rule.to_owned(), adjusted));
        item.notes.push(note);
        if item.verdict == Verdict::Identical {
            item.verdict = Verdict::IdenticalExplained;
        }
    }
    Some(item)
}

// ---------------------------------------------------------------------------------------
// Hook tables
// ---------------------------------------------------------------------------------------

/// Ledger A172: the homography of the manual corners (`cv::findHomography`, method 0) takes
/// an eigenvector that the reference computes with Eigen and this program with OpenCV's own
/// Jacobi method. Largest measured on the runs: 7.6e-14 (k03).
pub const TOLERANCE_A172_HOMOGRAPHY: NamedTolerance = NamedTolerance {
    rule: "manual-corner homography, ledger A172 (eigenvector by Eigen in XMALab's OpenCV)",
    rel: 8e-14,
    abs: 0.0,
};

/// The calibration images of the run's `NewProject` (camera by camera, sorted as the New
/// dataset dialog sorts them).
fn project_images(run: &super::calibration::CalibRun) -> Vec<Vec<std::path::PathBuf>> {
    let Ok(lines) = super::calibration::read_calib_script(&run.script) else {
        return Vec::new();
    };
    for l in lines {
        if let super::calibration::CalibOp::NewProjectCheckerboard(b) = l.op {
            return b
                .images
                .iter()
                .map(|im| {
                    let mut im = im.clone();
                    im.sort_by_key(|p| p.to_string_lossy().into_owned());
                    im
                })
                .collect();
        }
    }
    Vec::new()
}

/// The 8-bit image XMALab's detection reads (`getImage(image, true)`).
fn image_of(
    images: &[Vec<std::path::PathBuf>],
    camera: i64,
    image: i64,
) -> Option<xma_vision::image::ImageU8> {
    let p = images
        .get(usize::try_from(camera).ok()?)?
        .get(usize::try_from(image).ok()?)?;
    let f = xma_video::image_file::read_image(p).ok()?;
    Some(xma_engine::tracking::jobs::algorithm_image(&f))
}

type Key = (i64, i64, i64);

fn key(r: &xma_oracle::Row<'_>) -> Key {
    (r.i64("line"), r.i64("camera"), r.i64("image"))
}

/// The rows of `t` with key `k`, in order.
fn rows_of<'a>(t: &'a xma_oracle::Table, k: Key) -> Vec<xma_oracle::Row<'a>> {
    t.rows().filter(|r| key(r) == k).collect()
}

/// Every checker hook table of a run compared with `xma_vision`'s routines run on the
/// recorded inputs (the run's own images and the recorded points), and the calibrations
/// over all frames replayed with their recorded inputs.
#[must_use]
pub fn hook_points(run: &super::calibration::CalibRun) -> Vec<super::edits::Point> {
    let dir = run.output.join("hooks");
    let images = project_images(run);
    // The tables XMALab wrote (the manifest's `hook_tables`): it writes a table only when it
    // recorded a row (`oracle_hooks::take`), so a table it does not list is empty. A table it
    // lists that is absent, or one that cannot be read, is a problem.
    let listed: Option<Vec<String>> =
        xma_oracle::Manifest::try_read(run.output.join("manifest.json"))
            .ok()
            .and_then(|m| {
                m.json()
                    .get("hook_tables")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_owned))
                            .collect()
                    })
            });
    let mut tables = super::Check::new("hooks/ (tables read)", super::Tier::Read);
    let mut read = |n: &str| {
        let path = dir.join(n);
        let empty = || xma_oracle::Table::parse("line,camera,image\n", &path);
        match xma_oracle::Table::try_read(&path) {
            Ok(t) => t,
            Err(_)
                if !path.exists()
                    && listed
                        .as_ref()
                        .is_some_and(|l| !l.iter().any(|x| x == n.trim_end_matches(".csv"))) =>
            {
                empty()
            }
            Err(e) => {
                tables.problem(if path.exists() {
                    format!("{n}: cannot be read: {e}")
                } else if listed.is_some() {
                    format!("{n}: absent, though the manifest lists it as written")
                } else {
                    format!("{n}: absent, and the manifest does not list the tables written")
                });
                empty()
            }
        }
    };
    let mut items = Vec::new();
    // checker_result.csv is not compared here, but one that is absent though listed (or with
    // no list) is a problem, as for every other table.
    read("checker_result.csv");
    // checker_find.csv, checker_find_corners.csv.
    let find = read("checker_find.csv");
    let corners = read("checker_find_corners.csv");
    let mut c = super::Check::new("hooks/checker_find.csv", super::Tier::Read);
    let mut cc = super::Check::new("hooks/checker_find_corners.csv", super::Tier::Read);
    let mut n_corners = 0usize;
    for r in find.rows() {
        let k = key(&r);
        let Some(im) = image_of(&images, k.1, k.2) else {
            c.problem(format!(
                "line {}: image {} of camera {} not read",
                k.0, k.2, k.1
            ));
            continue;
        };
        let sum: u64 = im.data().iter().map(|&v| u64::from(v)).sum();
        c.int(
            format_args!("line {} image_sum", k.0),
            sum as i64,
            r.i64("image_sum"),
        );
        c.int(
            format_args!("line {} image_rows", k.0),
            im.height() as i64,
            r.i64("image_rows"),
        );
        let (cols, rows) = (
            r.i64("pattern_cols") as usize,
            r.i64("pattern_rows") as usize,
        );
        let got = xma_vision::chessboard::find_chessboard_corners(
            &im,
            cols,
            rows,
            xma_vision::chessboard::ChessboardFlags::from_bits(r.i64("flags") as i32),
        );
        c.int(
            format_args!("line {} found", k.0),
            i64::from(got.found),
            r.i64("found"),
        );
        c.int(
            format_args!("line {} n_corners", k.0),
            got.corners.len() as i64,
            r.i64("n_corners"),
        );
        let want = rows_of(&corners, k);
        n_corners += want.len();
        for (w, g) in want.iter().zip(&got.corners) {
            let i = w.i64("index");
            cc.num(
                format_args!("line {} corner {i} x", k.0),
                f64::from(g[0]),
                w.f64("x"),
            );
            cc.num(
                format_args!("line {} corner {i} y", k.0),
                f64::from(g[1]),
                w.f64("y"),
            );
        }
    }
    cc.rows(n_corners, corners.len());
    items.push(c.finish());
    items.push(cc.finish());
    // checker_homography.csv, checker_homography_points.csv, checker_predicted.csv.
    let homo = read("checker_homography.csv");
    let hpts = read("checker_homography_points.csv");
    let pred = read("checker_predicted.csv");
    let mut ch = super::Check::new("hooks/checker_homography.csv", super::Tier::Read);
    let mut cp = super::Check::new("hooks/checker_predicted.csv", super::Tier::Read);
    let mut n_predicted = 0usize;
    for r in homo.rows() {
        let k = key(&r);
        let pts = rows_of(&hpts, k);
        let grid: Vec<[f64; 2]> = pts
            .iter()
            .map(|p| [p.f64("grid_x"), p.f64("grid_y")])
            .collect();
        let clicks: Vec<[f64; 2]> = pts
            .iter()
            .map(|p| [p.f64("clicked_x"), p.f64("clicked_y")])
            .collect();
        let want_h: [f64; 9] = std::array::from_fn(|i| r.f64(&format!("h_{}{}", i / 3, i % 3)));
        match xma_vision::calib3d::homography::find_homography(&grid, &clicks) {
            Some(h) => {
                for i in 0..9 {
                    ch.num_named(
                        format_args!("line {} h_{}{}", k.0, i / 3, i % 3),
                        h[i],
                        want_h[i],
                        Some(TOLERANCE_A172_HOMOGRAPHY),
                    );
                }
            }
            None => ch.problem(format!("line {}: no homography here", k.0)),
        }
        // The prediction with XMALab's homography (the points as recorded).
        let p = rows_of(&pred, k);
        n_predicted += p.len();
        let spec: Vec<[f64; 2]> = p
            .iter()
            .map(|w| [w.f64("spec_x"), w.f64("spec_y")])
            .collect();
        let got = xma_vision::chessboard::project_with_homography(&want_h, &spec);
        for (w, g) in p.iter().zip(&got) {
            let i = w.i64("index");
            cp.num(
                format_args!("line {} point {i} x", k.0),
                f64::from(g[0]),
                w.f64("predicted_x"),
            );
            cp.num(
                format_args!("line {} point {i} y", k.0),
                f64::from(g[1]),
                w.f64("predicted_y"),
            );
        }
    }
    cp.rows(n_predicted, pred.len());
    items.push(ch.finish());
    items.push(cp.finish());
    // checker_subpix_call.csv, checker_subpix.csv: the refinement of the recorded inputs.
    let calls = read("checker_subpix_call.csv");
    let sub = read("checker_subpix.csv");
    let mut cs = super::Check::new("hooks/checker_subpix.csv", super::Tier::Read);
    let mut n_refined = 0usize;
    for r in calls.rows().filter(|r| r.i64("called") == 1) {
        let k = key(&r);
        let w = rows_of(&sub, k);
        n_refined += w.len();
        let Some(im) = image_of(&images, k.1, k.2) else {
            cs.problem(format!("line {}: image not read", k.0));
            continue;
        };
        let mut pts: Vec<[f32; 2]> = w
            .iter()
            .map(|p| [p.f64("in_x") as f32, p.f64("in_y") as f32])
            .collect();
        if xma_vision::chessboard::refine_corners_xmalab(&im, &mut pts).is_err() {
            cs.problem(format!("line {}: the refinement throws here", k.0));
            continue;
        }
        for (p, g) in w.iter().zip(&pts) {
            let i = p.i64("index");
            cs.num(
                format_args!("line {} point {i} x", k.0),
                f64::from(g[0]),
                p.f64("out_x"),
            );
            cs.num(
                format_args!("line {} point {i} y", k.0),
                f64::from(g[1]),
                p.f64("out_y"),
            );
        }
    }
    // Rows of images with no `called == 1` row are not compared: counted.
    cs.rows(n_refined, sub.len());
    items.push(cs.finish());
    // checker_sort.csv: keep or reverse, from the click and the first and last corner.
    let sort = read("checker_sort.csv");
    let mut cso = super::Check::new("hooks/checker_sort.csv", super::Tier::Read);
    for r in sort.rows() {
        let (x, y) = (r.f64("click_x"), r.f64("click_y"));
        let d = |px: f64, py: f64| (x - px) * (x - px) + (y - py) * (y - py);
        let d0 = d(r.f64("first_x"), r.f64("first_y"));
        let dl = d(r.f64("last_x"), r.f64("last_y"));
        let mut best = 10_000_000.0;
        let mut idx = 0;
        if d0 < best {
            best = d0;
        }
        if r.i64("n_all") > 1 && dl < best {
            idx = 1;
        }
        cso.int(
            format_args!("line {} reversed", r.i64("line")),
            idx,
            r.i64("reversed"),
        );
    }
    items.push(cso.finish());
    if tables.n_problems > 0 {
        items.push(tables.finish());
    }
    // calibration_all_frames*.csv: the per-camera calibrations replayed.
    if dir.join("calibration_all_frames.csv").is_file() {
        items.push(all_frames_item(&dir));
    }
    vec![super::edits::Point {
        name: "hooks".to_owned(),
        line: 0,
        items,
        skipped: None,
    }]
}

/// The calibrations over all frames (`processing/Calibration::process`) replayed with their
/// recorded inputs, within the planar tolerances of ledger A172.
fn all_frames_item(dir: &Path) -> Item {
    let t = xma_oracle::Table::read(dir.join("calibration_all_frames.csv"));
    let views = xma_oracle::Table::read(dir.join("calibration_all_frames_views.csv"));
    let pts = xma_oracle::Table::read(dir.join("calibration_all_frames_points.csv"));
    let mut c = super::Check::new("hooks/calibration_all_frames.csv", super::Tier::Read);
    for r in t.rows() {
        let (line, cam) = (r.i64("line"), r.i64("camera"));
        let n_views = r.i64("n_views") as usize;
        let mut obj = Vec::new();
        let mut img = Vec::new();
        for v in 0..n_views {
            let (o, i): (Vec<_>, Vec<_>) = pts
                .rows()
                .filter(|p| {
                    p.i64("line") == line && p.i64("camera") == cam && p.i64("view") == v as i64
                })
                .map(|p| {
                    (
                        // `cv::Point3f`, `cv::Point2f` (the values are recorded from them).
                        xma_vision::Point3::new(p.f64("x3d"), p.f64("y3d"), p.f64("z3d"))
                            .cast::<f32>(),
                        xma_vision::Point2::new(p.f64("u"), p.f64("v")).cast::<f32>(),
                    )
                })
                .unzip();
            obj.push(o);
            img.push(i);
        }
        let bits = r.i64("flags");
        let flags = xma_vision::calib3d::CalibrateFlags {
            use_intrinsic_guess: bits & 1 != 0,
            fix_principal_point: bits & 4 != 0,
            zero_tangent_dist: bits & 8 != 0,
            fix_k: [
                bits & 32 != 0,
                bits & 64 != 0,
                bits & 128 != 0,
                bits & 2048 != 0,
                bits & 4096 != 0,
                bits & 8192 != 0,
            ],
            rational_model: bits & 16384 != 0,
        };
        let tol = if flags.rational_model {
            TOLERANCE_A172_RATIONAL
        } else {
            TOLERANCE_A172_PLANAR
        };
        let mut k = xma_vision::Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                k[(i, j)] = r.f64(&format!("k_in_{i}{j}"));
            }
        }
        let dist_in: Vec<f64> = (0..8).map(|i| r.f64(&format!("dist_in_{i}"))).collect();
        let size = (r.f64("width") as usize, r.f64("height") as usize);
        let out = match xma_vision::calib3d::calibrate_camera(&obj, &img, size, &k, &dist_in, flags)
        {
            Ok(o) => o,
            Err(e) => {
                c.problem(format!("line {line} camera {cam}: refused here: {e}"));
                continue;
            }
        };
        for i in 0..3 {
            for j in 0..3 {
                c.num_named(
                    format_args!("line {line} camera {cam} k_{i}{j}"),
                    out.camera_matrix[(i, j)],
                    r.f64(&format!("k_out_{i}{j}")),
                    Some(tol),
                );
            }
        }
        for (i, d) in out.distortion.iter().enumerate() {
            c.num_named(
                format_args!("line {line} camera {cam} dist_{i}"),
                *d,
                r.f64(&format!("dist_out_{i}")),
                Some(tol),
            );
        }
        c.num_named(
            format_args!("line {line} camera {cam} rms"),
            out.rms,
            r.f64("rms"),
            Some(tol),
        );
        let vr: Vec<_> = views
            .rows()
            .filter(|v| v.i64("line") == line && v.i64("camera") == cam)
            .collect();
        for (v, row) in vr.iter().enumerate() {
            let Some(p) = out.poses.get(v) else {
                c.problem(format!("line {line} camera {cam}: view {v} missing here"));
                continue;
            };
            let vals = [
                p.rotation[0],
                p.rotation[1],
                p.rotation[2],
                p.translation[0],
                p.translation[1],
                p.translation[2],
            ];
            for (x, col) in vals.iter().zip(["rx", "ry", "rz", "tx", "ty", "tz"]) {
                c.num_named(
                    format_args!("line {line} camera {cam} view {v} {col}"),
                    *x,
                    row.f64(col),
                    Some(tol),
                );
            }
        }
    }
    c.finish()
}

#[cfg(test)]
mod review12_tests {
    //! Review of 20 November 2026, second round, findings 5 and 8 of `parity-xtask.md`.

    use super::*;
    use crate::parity::calibration::CalibRun;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("xma-review12-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("scratch folder");
        d
    }

    /// A run whose output folder holds `hooks/` with `files` and a manifest listing
    /// `listed` as written.
    fn hook_run(tag: &str, files: &[(&str, &str)], listed: &[&str]) -> CalibRun {
        let out = scratch(tag);
        std::fs::create_dir_all(out.join("hooks")).expect("hooks");
        for (n, t) in files {
            std::fs::write(out.join("hooks").join(n), t).expect("hook table");
        }
        let list: Vec<String> = listed.iter().map(|n| format!("\"{n}\"")).collect();
        std::fs::write(
            out.join("manifest.json"),
            format!("{{\"hook_tables\": [{}]}}", list.join(", ")),
        )
        .expect("manifest");
        CalibRun {
            name: tag.to_owned(),
            start_project: None,
            script: out.join("none.txt"),
            output: out,
            repeats: Vec::new(),
            unstable: Vec::new(),
            checker: true,
        }
    }

    fn verdicts(run: &CalibRun) -> Vec<(String, Verdict, Vec<String>)> {
        let p = hook_points(run);
        let _ = std::fs::remove_dir_all(&run.output);
        p.into_iter()
            .flat_map(|p| p.items)
            .map(|i| (i.name, i.verdict, i.problems))
            .collect()
    }

    const RESULT: &str =
        "line,camera,image,via_homography,n_points,expected,stored\n3,0,0,0,54,54,1\n";
    const SUBPIX: &str =
        "line,camera,image,source,index,in_x,in_y,out_x,out_y\n3,0,0,auto,0,1,1,1,1\n";

    #[test]
    fn review12_missing_listed_hook_table_is_a_problem() {
        // checker_subpix_call.csv is listed as written but absent: before, an empty table,
        // and its refinements never compared.
        let run = hook_run(
            "hk_missing",
            &[
                ("checker_result.csv", RESULT),
                ("checker_subpix.csv", SUBPIX),
            ],
            &["checker_result", "checker_subpix", "checker_subpix_call"],
        );
        let v = verdicts(&run);
        assert!(
            v.iter()
                .any(|(_, verdict, _)| *verdict == Verdict::Different),
            "{v:?}"
        );
    }

    #[test]
    fn review12_unreadable_hook_table_is_a_problem() {
        let run = hook_run(
            "hk_bad",
            &[
                ("checker_result.csv", RESULT),
                ("checker_sort.csv", "line,camera,image\n3,0\n"),
            ],
            &["checker_result", "checker_sort"],
        );
        let v = verdicts(&run);
        assert!(
            v.iter()
                .any(|(_, verdict, problems)| *verdict == Verdict::Different
                    && problems.iter().any(|p| p.contains("checker_sort.csv"))),
            "{v:?}"
        );
    }

    #[test]
    fn review12_refinements_without_a_call_are_counted() {
        // A refinement row whose image has no call row: before, never compared.
        let run = hook_run(
            "hk_rows",
            &[
                ("checker_result.csv", RESULT),
                (
                    "checker_subpix_call.csv",
                    "line,camera,image,source,called,n_points,win_w,win_h,zero_w,zero_h,criteria_type,max_iter,eps\n3,0,0,auto,0,54,11,11,-1,-1,3,30,0.1\n",
                ),
                ("checker_subpix.csv", SUBPIX),
            ],
            &["checker_result", "checker_subpix", "checker_subpix_call"],
        );
        let v = verdicts(&run);
        let sub = v.iter().find(|(n, ..)| n == "hooks/checker_subpix.csv");
        assert!(
            sub.is_some_and(|(_, verdict, _)| *verdict == Verdict::Different),
            "{v:?}"
        );
    }

    #[test]
    fn review12_a_table_not_written_is_empty() {
        // Tables the manifest does not list: XMALab wrote no row to them.
        let run = hook_run(
            "hk_absent",
            &[("checker_result.csv", RESULT)],
            &["checker_result"],
        );
        let v = verdicts(&run);
        assert!(
            v.iter()
                .all(|(_, verdict, _)| *verdict != Verdict::Different),
            "{v:?}"
        );
    }

    #[test]
    fn review12c_checker_result_not_written_is_empty() {
        // No checker_result.csv and none listed: an empty table, not a problem.
        let run = hook_run("hk_noresult", &[], &[]);
        let v = verdicts(&run);
        assert!(
            v.iter()
                .all(|(_, verdict, _)| *verdict != Verdict::Different),
            "{v:?}"
        );
    }

    #[test]
    fn review12c_checker_result_absent_without_a_list_is_a_problem() {
        let run = hook_run("hk_noresult_nolist", &[], &[]);
        std::fs::write(run.output.join("manifest.json"), "{}").expect("manifest");
        let v = verdicts(&run);
        assert!(
            v.iter()
                .any(|(_, verdict, problems)| *verdict == Verdict::Different
                    && problems.iter().any(|p| p.contains("checker_result.csv"))),
            "{v:?}"
        );
    }

    /// A different camera matrix whose numbers differ by 5 % (within A189, outside A172's
    /// multi-camera bound).
    fn item() -> Item {
        let mut c = crate::parity::Check::new("camera_matrix.csv", crate::parity::Tier::Read);
        c.num("k_00", 1.05, 1.0);
        c.finish()
    }

    fn state_dump(tag: &str, error: f64) -> std::path::PathBuf {
        let d = scratch(tag);
        std::fs::create_dir_all(d.join("calibration")).expect("calibration");
        std::fs::write(
            d.join("calibration/camera_state.csv"),
            "camera,is_calibrated,is_optimized,recalibration_required,has_model_distortion,n_calibration_images,calibration_error,calibration_n_inlier\n0,1,1,0,0,1,0,2\n",
        )
        .expect("camera_state");
        std::fs::write(
            d.join("calibration/image_state.csv"),
            "camera,image,is_calibrated,n_detected_all,n_points,n_inlier,calibration_error,filename\n0,0,1,3,3,2,0,a.tif\n0,1,0,3,3,0,0,b.tif\n",
        )
        .expect("image_state");
        // Two inliers of the calibrated image with `error`, an outlier and an image not
        // calibrated with large errors (not counted).
        std::fs::write(
            d.join("calibration/image_points.csv"),
            format!(
                "camera,image,index,inlier,detected_x,detected_y,detected_ux,detected_uy,projected_x,projected_y,projected_ux,projected_uy,error,error_undistorted\n\
                 0,0,0,1,0,0,0,0,0,0,0,0,{error},0\n\
                 0,0,1,1,0,0,0,0,0,0,0,0,{error},0\n\
                 0,0,2,0,0,0,0,0,0,0,0,0,500,0\n\
                 0,1,0,1,0,0,0,0,0,0,0,0,500,0\n"
            ),
        )
        .expect("image_points");
        d
    }

    #[test]
    fn review12_tolerance_comes_from_the_reference() {
        // The reference's mean error 0.9 px after an optimisation: A172 multi-camera, so a
        // 5 % difference is not tolerated, whatever this program's own error is.
        let d = state_dump("ref_low", 0.9);
        let s = reference_state(&d).expect("state");
        assert!(s.optimized && !s.model_distortion);
        assert!((s.mean_error - 0.9).abs() < 1e-12, "{s:?}");
        assert_eq!(tolerance_for(&s), TOLERANCE_A172_MULTICAM);
        let mut items = vec![item()];
        tolerate(&mut items, Ok(&s));
        assert_eq!(items[0].verdict, Verdict::Different);
        let _ = std::fs::remove_dir_all(&d);
        // The reference's mean error 47 px (A189): tolerated.
        let d = state_dump("ref_high", 47.0);
        let s = reference_state(&d).expect("state");
        assert_eq!(tolerance_for(&s), TOLERANCE_A189_MULTICAM);
        let mut items = vec![item()];
        tolerate(&mut items, Ok(&s));
        assert_eq!(items[0].verdict, Verdict::WithinTolerance);
        let _ = std::fs::remove_dir_all(&d);
        // No reference state: not tolerated, and said why.
        let mut items = vec![item()];
        tolerate(&mut items, Err("no dump"));
        assert_eq!(items[0].verdict, Verdict::Different);
        assert!(items[0].problems.iter().any(|p| p.contains("no dump")));
        // A dump without the state tables: an error naming what it lacks.
        let d = scratch("ref_none");
        let e = reference_state(&d).expect_err("no state tables");
        assert!(e.contains("camera_state.csv"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
