//! Marker-to-marker distances and precision info: File > Export > Marker to Marker Distances
//! and Precision Info, and the two files the C++ writes into every saved project.
//!
//! Ported from core/Trial.cpp, `Trial::saveMarkerToMarkerDistances` (`:1317-1489`) and
//! `Trial::savePrecisionInfo` (`:1491-1512`), with the statistics they read from
//! core/RigidBody.cpp (`getFramesTracked`, `getMarkerToMarkerSD`, `getError3D`) and
//! core/Marker.cpp (`getFramesTracked`, `getReprojectionError`).
//!
//! Frame ranges here are half-open, `from..to`, 0-based, as the C++ functions take them. The
//! GUI passes `from − 1` and `to` of its From/To dialog (defaults: the trial's start and end
//! frames, 1-based); the project save passes `0` and the number of frames.

use xma_model::{CppCompat, Marker, MarkerStatus, RigidBody, Trial};
use xma_vision::cmath::pow;

use super::{ExportError, ExportFile, check_half_open, check_trial, g};
use crate::cfmt::push_g17;

/// The distances between two markers over the frames `from..to` where both have a 3D point
/// (`status3D > UNDEFINED`), in frame order.
///
/// Ported from core/Trial.cpp:1336-1343: `cv::sqrt(dx·dx + dy·dy + dz·dz)` of the difference.
fn distances(a: &Marker, b: &Marker, from: usize, to: usize) -> Vec<f64> {
    (from..to)
        .filter(|&f| a.status3d[f].code() > 0 && b.status3d[f].code() > 0)
        .map(|f| {
            let d = a.points3d[f] - b.points3d[f];
            (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
        })
        .collect()
}

/// Mean and sample standard deviation of the distance between two markers over `from..to`,
/// as the distance export computes them: mean 0 with no common frame; SD over `n − 1`, and 0
/// when there is one value or none (unlike the rigid-body error SDs, ledger A20).
///
/// Ported from core/Trial.cpp:1370-1394.
#[must_use]
pub fn marker_distance_stats(a: &Marker, b: &Marker, from: usize, to: usize) -> (f64, f64) {
    let value = distances(a, b, from, to);
    let mut mean = 0.0;
    for v in &value {
        mean += v;
    }
    if !value.is_empty() {
        mean /= value.len() as f64;
    }
    let mut sd = 0.0;
    for v in &value {
        sd += pow(v - mean, 2.0);
    }
    if value.len() > 1 {
        sd = (sd / (value.len() - 1) as f64).sqrt();
    }
    (mean, sd)
}

/// `Marker <i+1> <description>`.
fn label(trial: &Trial, i: usize) -> String {
    format!("Marker {} {}", i + 1, trial.markers[i].description)
}

/// [`marker_distance_stats`] of every ordered marker pair over `from..to`, each computed
/// once and kept.
///
/// The distance is symmetric bit for bit: `a − b` is the exact negation of `b − a` and the
/// square drops the sign, so (i, j) and (j, i) give the same distances in the same order,
/// and so the same mean and SD. The one exception is a NaN coordinate in both markers at
/// one frame: the difference then carries one of the two NaNs, which may differ in sign
/// (written `nan` or `-nan`). Every distance of a pair whose mean is not NaN is a number, so
/// such a pair is shared by both orders; a pair whose mean is NaN is computed again for the
/// other order, as before.
struct PairStats<'t> {
    trial: &'t Trial,
    from: usize,
    to: usize,
    /// `(mean, sd)` of (i, j) at `i * n + j`, `None` until computed.
    cache: Vec<Option<(f64, f64)>>,
}

impl<'t> PairStats<'t> {
    fn new(trial: &'t Trial, from: usize, to: usize) -> Self {
        let n = trial.markers.len();
        Self {
            trial,
            from,
            to,
            cache: vec![None; n * n],
        }
    }

    fn get(&mut self, i: usize, j: usize) -> (f64, f64) {
        let n = self.trial.markers.len();
        if let Some(s) = self.cache[i * n + j] {
            return s;
        }
        let m = &self.trial.markers;
        let s = marker_distance_stats(&m[i], &m[j], self.from, self.to);
        self.cache[i * n + j] = Some(s);
        if !s.0.is_nan() {
            self.cache[j * n + i] = Some(s);
        }
        s
    }
}

/// One `Mean` or `SD` table over the markers `idx`: a title line, then one line per marker.
fn table(out: &mut String, stats: &mut PairStats<'_>, idx: &[usize], title: &str, sd: bool) {
    let trial = stats.trial;
    out.push_str(title);
    for &i in idx {
        out.push(',');
        out.push_str(&label(trial, i));
    }
    out.push('\n');
    for &i in idx {
        out.push_str(&label(trial, i));
        out.push(',');
        for (k, &j) in idx.iter().enumerate() {
            let (mean, s) = stats.get(i, j);
            push_g17(out, if sd { s } else { mean });
            if k != idx.len() - 1 {
                out.push(',');
            }
        }
        out.push('\n');
    }
}

/// The marker-to-marker distance file for frames `from..to`.
///
/// Ported from core/Trial.cpp, Trial::saveMarkerToMarkerDistances: the `Mean` table of every
/// marker, an empty line, the `SD` table; then for each rigid body two empty lines,
/// `<description> Mean` and the table of its markers, an empty line, `<description> SD` and
/// its table. An empty description gives a title starting with a space (` Mean`).
///
/// # Errors
/// When `to` is past the trial, or the trial's markers or bodies hold fewer frames than it
/// (undefined behaviour in the C++; see [`check_input`]).
pub fn marker_to_marker_distances(
    trial: &Trial,
    from: usize,
    to: usize,
) -> Result<String, ExportError> {
    check_input(trial, from, to)?;
    let mut out = String::new();
    let mut stats = PairStats::new(trial, from, to);
    let all: Vec<usize> = (0..trial.markers.len()).collect();
    table(&mut out, &mut stats, &all, "Mean", false);
    out.push('\n');
    table(&mut out, &mut stats, &all, "SD", true);
    for body in &trial.rigid_bodies {
        out.push_str("\n\n");
        let idx = &body.marker_indices;
        let mean_title = format!("{} Mean", body.description);
        table(&mut out, &mut stats, idx, &mean_title, false);
        out.push('\n');
        let sd_title = format!("{} SD", body.description);
        table(&mut out, &mut stats, idx, &sd_title, true);
    }
    Ok(out)
}

/// The number of frames the precision report gives for a body. XMALab 3 (`cpp`) counts
/// every frame (`poseComputed >= 0` always holds; ledger A24); fixed, the frames whose pose
/// was computed.
///
/// Ported from core/RigidBody.cpp, RigidBody::getFramesTracked (`:397-405`).
#[must_use]
pub fn body_frames_tracked(body: &RigidBody, cpp: bool) -> usize {
    if cpp {
        body.raw.valid.len()
    } else {
        body.raw.valid.iter().filter(|&&v| v).count()
    }
}

/// What [`body_pair_sums`] returns: the sum of squared deviations, the number of distances
/// and the degrees of freedom.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PairSums {
    ss: f64,
    count: usize,
    dof: usize,
}

/// The sums behind a body's marker-to-marker standard deviation over frames `start..end`:
/// the squared deviations of each marker pair's distance from that pair's own mean, summed
/// over every pair in C++ order; the number of distances; and the degrees of freedom, the
/// sum over pairs of (n − 1) for the pairs with at least one distance.
///
/// Ported from core/RigidBody.cpp, RigidBody::getMarkerToMarkerSD (`:407-445`), the loops:
/// pairs i < j of the body's markers, frames where both have a 3D point, the pair's mean
/// accumulated in frame order, `pow(err − mean, 2)` added to one running sum. The degrees of
/// freedom are not in the C++ (ledger A58). `end` is clamped to each pair's frames (the
/// Project Overview passes `usize::MAX` for the whole trial, where the C++ takes the first
/// marker's `status3D` size, the same in every project the C++ makes), and a marker index
/// that does not exist is skipped (the C++ would read out of bounds).
fn body_pair_sums(trial: &Trial, body: &RigidBody, start: usize, end: usize) -> PairSums {
    let mut sums = PairSums::default();
    let idx = &body.marker_indices;
    for i in 0..idx.len() {
        for j in i + 1..idx.len() {
            let (Some(a), Some(b)) = (trial.markers.get(idx[i]), trial.markers.get(idx[j])) else {
                continue;
            };
            let end = end
                .min(a.status3d.len())
                .min(a.points3d.len())
                .min(b.status3d.len())
                .min(b.points3d.len());
            let value = distances(a, b, start.min(end), end);
            let mut mean = 0.0;
            for v in &value {
                mean += v;
            }
            if !value.is_empty() {
                mean /= value.len() as f64;
            }
            for v in &value {
                sums.ss += pow(v - mean, 2.0);
            }
            sums.count += value.len();
            sums.dof += value.len().saturating_sub(1);
        }
    }
    sums
}

/// A body's marker-to-marker standard deviation over frames `start..end`, and the number of
/// distances it is computed from.
///
/// XMALab 3 (`compat.pooled_sd_total_count`, ledger A58): ported from core/RigidBody.cpp,
/// RigidBody::getMarkerToMarkerSD (`:407-445`): the squared deviations of each pair's
/// distance from the pair's own mean, summed over all pairs, then `sqrt(sum / (count − 1))`
/// when there is more than one distance, else the plain sum (0: no distance, or the one
/// deviation, which is 0). Fixed (ledger A58): the pooled standard deviation,
/// `sqrt(sum / dof)` with `dof` the sum over pairs of (n − 1); 0 when `dof` is 0.
///
/// Used by the precision report ([`precision_info`]) and the Project Overview dialog (with
/// `start` 0 and `end` `usize::MAX`, the whole trial).
#[must_use]
pub fn body_marker_to_marker_sd(
    trial: &Trial,
    body: &RigidBody,
    start: usize,
    end: usize,
    compat: CppCompat,
) -> (f64, usize) {
    let s = body_pair_sums(trial, body, start, end);
    let sd = if compat.pooled_sd_total_count {
        if s.count > 1 {
            (s.ss / (s.count - 1) as f64).sqrt()
        } else {
            s.ss
        }
    } else if s.dof > 0 {
        (s.ss / s.dof as f64).sqrt()
    } else {
        0.0
    };
    (sd, s.count)
}

/// A trial's marker-to-marker standard deviation over all its frames: its bodies pooled
/// (a marker pair in two bodies counts in both).
///
/// XMALab 3 (`compat.pooled_sd_total_count`, ledger A58): ported from core/Trial.cpp,
/// Trial::getMarkerToMarkerSD (`:2476-2495`): for each body with `c > 0` distances,
/// `sd += pow(s, 2) * (c − 1)` and `count += c` (with `s` the body's XMALab 3 value), then
/// `sqrt(sd / (count − 1))` when `count` is not 0. Fixed (ledger A58):
/// `sqrt(sum of the bodies' squared deviations / sum of their dof)`, 0 when that dof is 0.
#[must_use]
pub fn trial_marker_to_marker_sd(trial: &Trial, compat: CppCompat) -> f64 {
    if compat.pooled_sd_total_count {
        let mut sd = 0.0;
        let mut count = 0usize;
        for b in &trial.rigid_bodies {
            let (s, c) = body_marker_to_marker_sd(trial, b, 0, usize::MAX, compat);
            if c > 0 {
                sd += pow(s, 2.0) * (c as f64 - 1.0);
                count += c;
            }
        }
        if count != 0 {
            sd = (sd / (count as f64 - 1.0)).sqrt();
        }
        sd
    } else {
        let mut ss = 0.0;
        let mut dof = 0usize;
        for b in &trial.rigid_bodies {
            let s = body_pair_sums(trial, b, 0, usize::MAX);
            ss += s.ss;
            dof += s.dof;
        }
        if dof > 0 {
            (ss / dof as f64).sqrt()
        } else {
            0.0
        }
    }
}

/// The mean 3D error of a body over frames `start..end` that have a pose (raw or filtered);
/// 0 when none has.
///
/// Ported from core/RigidBody.cpp, RigidBody::getError3D (`:449-479`).
#[must_use]
pub fn body_error_3d(body: &RigidBody, filtered: bool, start: usize, end: usize) -> f64 {
    let track = if filtered { &body.filtered } else { &body.raw };
    let mut error = 0.0;
    let mut count = 0usize;
    for i in start..end {
        if track.valid[i] {
            count += 1;
            error += track.errors[i].mean_3d;
        }
    }
    if count != 0 {
        error /= count as f64;
    }
    error
}

/// The number of frames whose 3D status is `TRACKED` or better, over the whole trial.
///
/// Ported from core/Marker.cpp, Marker::getFramesTracked(-1) (`:1633-1652`).
#[must_use]
pub fn marker_frames_tracked(marker: &Marker) -> usize {
    marker
        .status3d
        .iter()
        .filter(|&&s| s >= MarkerStatus::Tracked)
        .count()
}

/// Mean and sample standard deviation of a marker's 2D reprojection error over frames
/// `start..end` whose 3D status is `TRACKED` or better. XMALab 3 (`cpp`) pools **every**
/// camera at those frames whatever its 2D status (ledger A25); fixed, only the cameras in
/// which the marker is set (2D status above `UNDEFINED`). Mean 0 with no value; SD 0 with
/// one or none.
///
/// Ported from core/Marker.cpp, Marker::getReprojectionError (`:1654-1691`).
#[must_use]
pub fn marker_reprojection_error(
    marker: &Marker,
    start: usize,
    end: usize,
    cpp: bool,
) -> (f64, f64) {
    let frames = || (start..end).filter(|&i| marker.status3d[i] >= MarkerStatus::Tracked);
    let used = |v: &xma_model::MarkerView, i: usize| cpp || v.status[i].has_data();
    let mut count = 0usize;
    let mut error = 0.0;
    for i in frames() {
        for v in marker.views.iter().filter(|v| used(v, i)) {
            count += 1;
            error += v.error[i];
        }
    }
    if count != 0 {
        error /= count as f64;
    }
    let mut sd = 0.0;
    count = 0;
    for i in frames() {
        for v in marker.views.iter().filter(|v| used(v, i)) {
            count += 1;
            sd += pow(v.error[i] - error, 2.0);
        }
    }
    let sd = if count <= 1 {
        0.0
    } else {
        (sd / (count - 1) as f64).sqrt()
    };
    (error, sd)
}

/// The precision report for frames `from..to`.
///
/// Ported from core/Trial.cpp, Trial::savePrecisionInfo: a header, one line per body
/// (`description,frames tracked,marker-to-marker SD,mean 3D error,mean filtered 3D error`),
/// two empty lines, a header, one line per marker (`description,frames tracked,mean
/// reprojection error,SD`). XMALab 3 writes descriptions as stored, even when empty (ledger
/// F25); fixed, an empty one is replaced by the export name (`RigidBody%03d`, `marker%03d`).
/// The frames-tracked counts ignore the range. `compat` also selects ledger A24, A25 and A58
/// ([`body_frames_tracked`], [`marker_reprojection_error`], [`body_marker_to_marker_sd`]).
///
/// # Errors
/// As [`marker_to_marker_distances`].
pub fn precision_info(
    trial: &Trial,
    from: usize,
    to: usize,
    compat: CppCompat,
) -> Result<String, ExportError> {
    check_input(trial, from, to)?;
    let mut out = String::new();
    out.push_str(
        "RigidBodies,nb tracked frames,marker to marker sd,error 3d Unfiltered,error 3D filtered\n",
    );
    for (k, rb) in trial.rigid_bodies.iter().enumerate() {
        let (average_sd, _) = body_marker_to_marker_sd(trial, rb, from, to, compat);
        let val1 = body_error_3d(rb, false, from, to);
        let val2 = body_error_3d(rb, true, from, to);
        let name = super::file_description(
            &rb.description,
            super::body_export_name(&rb.description, k),
            compat.export_details,
        );
        out.push_str(&format!(
            "{},{},{},{},{}\n",
            name,
            body_frames_tracked(rb, compat.frames_tracked_all),
            g(average_sd),
            g(val1),
            g(val2)
        ));
    }
    out.push_str("\n\nMarkers,nb tracked frames,reprojectionerror,sd\n");
    for (i, m) in trial.markers.iter().enumerate() {
        let (val, sd) =
            marker_reprojection_error(m, from, to, compat.reprojection_error_all_cameras);
        let name = super::file_description(
            &m.description,
            super::marker_export_name(&m.description, i),
            compat.export_details,
        );
        out.push_str(&format!(
            "{},{},{},{}\n",
            name,
            marker_frames_tracked(m),
            g(val),
            g(sd)
        ));
    }
    Ok(out)
}

/// The two files `ProjectFileIO::saveProject` writes at the top level of the project
/// archive for a trial: `PrecisionInfo_<trial name>.txt` then
/// `MarkerDistances_<trial name>.txt`, both over every frame (`0..n_frames`), and only when
/// the trial does not require recomputation (empty otherwise).
///
/// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::saveProject (`:190-196`). The project
/// writer calls this for each trial in order, before writing that trial's folder.
///
/// # Errors
/// As [`marker_to_marker_distances`].
pub fn project_save_files(
    trial: &Trial,
    compat: CppCompat,
) -> Result<Vec<ExportFile>, ExportError> {
    if trial.requires_recomputation {
        return Ok(Vec::new());
    }
    let n = trial.n_frames;
    Ok(vec![
        ExportFile::new(
            format!("PrecisionInfo_{}.txt", trial.name),
            precision_info(trial, 0, n, compat)?,
        ),
        ExportFile::new(
            format!("MarkerDistances_{}.txt", trial.name),
            marker_to_marker_distances(trial, 0, n)?,
        ),
    ])
}

/// What the distance and precision files would read out of bounds on in the C++: `to` past
/// the trial, a marker with fewer 3D frames than the trial or fewer 2D errors than `to` in
/// one of its cameras, a body naming a marker that does not exist, or a body whose poses
/// are shorter than `to`.
fn check_input(trial: &Trial, from: usize, to: usize) -> Result<(), ExportError> {
    check_half_open(from, to, trial.n_frames)?;
    check_trial(trial, None)?;
    if from >= to {
        return Ok(());
    }
    for (i, m) in trial.markers.iter().enumerate() {
        if m.views.iter().any(|v| v.error.len() < to) {
            return Err(ExportError::Inconsistent(format!(
                "marker {} holds fewer 2D errors than the trial has frames",
                i + 1
            )));
        }
    }
    for (k, b) in trial.rigid_bodies.iter().enumerate() {
        let short = [&b.raw, &b.filtered]
            .iter()
            .any(|t| t.valid.len() < to || t.errors.len() < to);
        if short {
            return Err(ExportError::Inconsistent(format!(
                "rigid body {} holds poses for {} frames, fewer than the {to} asked for",
                k + 1,
                b.raw.valid.len().min(b.filtered.valid.len())
            )));
        }
    }
    Ok(())
}
