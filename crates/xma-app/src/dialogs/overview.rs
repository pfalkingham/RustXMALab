//! The Project Overview window ("Project overview").
//!
//! Ported from `ProjectOverview.ui` and ui/ProjectOverview.cpp: `treeWidget`, a two-column
//! tree of the dataset (`update`): repository and creation date; the calibration trial with
//! its undistortion files, the cameras' calibration images and inlier counts, and the
//! calibration object; then each trial with its tracked range, filter frequency, frame rate,
//! errors, movies, rigid bodies and markers. `pushButton` "Update" rebuilds the tree.
//!
//! The statistics are ported from core/Trial.cpp, core/Marker.cpp, core/RigidBody.cpp,
//! core/Camera.cpp and core/CalibrationImage.cpp (named at each function). Numbers are
//! written as `QString::number` writes them (`%g`, six significant digits).
//!
//! Differences from the C++ (parity ledger):
//!
//! - The C++ lists a camera under "Calibration" only when it has an undistortion grid (the
//!   condition was copied from the undistortion list); here every camera is listed, and
//!   "Undistortion File" only for a camera that has a grid.
//! - `RigidBody::getFirstTrackedFrame`, `getLastTrackedFrame` and `getFramesTracked` test
//!   `poseComputed >= 0`, which every frame passes, so a body showed frame 1, the last frame
//!   and every frame whatever was tracked; here they count frames with a computed pose.
//! - "Camera error" and "Camera file error" are the mean distance between the detected and
//!   the reprojected calibration points (`CalibrationImage::getCalibrationError`). The
//!   reprojected points are made by the calibration step, which is ported in phase 4; until
//!   then the values are left empty.

use xma_model::{CalibrationObject, CalibrationType, Marker, MarkerStatus, RigidBody, Trial};

use xma_engine::precision::{body_marker_to_marker_sd, trial_marker_to_marker_sd};

use super::form::{TreeItem, qnum};
use super::portal::{Study, video};

/// The file name of a path (`QFileInfo::fileName`), for either separator.
#[must_use]
pub fn file_name(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or("").to_owned()
}

/// Ported from core/Marker.cpp, Marker::getFirstTrackedFrame(cam): 1-based, or -1.
fn marker_first(m: &Marker, cam: usize) -> i64 {
    m.views.get(cam).map_or(-1, |v| {
        v.status
            .iter()
            .position(|s| *s >= MarkerStatus::Tracked)
            .map_or(-1, |i| i as i64 + 1)
    })
}

/// Ported from core/Marker.cpp, Marker::getLastTrackedFrame(cam): 1-based, or -1.
fn marker_last(m: &Marker, cam: usize) -> i64 {
    m.views.get(cam).map_or(-1, |v| {
        v.status
            .iter()
            .rposition(|s| *s >= MarkerStatus::Tracked)
            .map_or(-1, |i| i as i64 + 1)
    })
}

/// Ported from core/Marker.cpp, Marker::getFramesTracked(cam).
fn marker_tracked(m: &Marker, cam: usize) -> usize {
    m.views.get(cam).map_or(0, |v| {
        v.status
            .iter()
            .filter(|s| **s >= MarkerStatus::Tracked)
            .count()
    })
}

/// Ported from core/Marker.cpp, Marker::getFramesTracked(-1): frames with a tracked 3D
/// point.
fn marker_tracked_3d(m: &Marker) -> usize {
    m.status3d
        .iter()
        .filter(|s| **s >= MarkerStatus::Tracked)
        .count()
}

/// Ported from core/Marker.cpp, Marker::getReprojectionError over the whole trial: the mean
/// 2D error of every camera at the frames with a tracked 3D point (ledger A25).
fn marker_error(m: &Marker) -> f64 {
    let mut count = 0usize;
    let mut error = 0.0;
    for (i, s) in m.status3d.iter().enumerate() {
        if *s >= MarkerStatus::Tracked {
            for v in &m.views {
                count += 1;
                error += v.error.get(i).copied().unwrap_or(0.0);
            }
        }
    }
    if count != 0 {
        error /= count as f64;
    }
    error
}

/// Ported from core/Trial.cpp, Trial::getFirstTrackedFrame / getLastTrackedFrame: per
/// camera, the first (last) tracked frame of any marker, joined with " / ".
fn trial_range(t: &Trial, n_cameras: usize, last: bool) -> String {
    (0..n_cameras)
        .map(|c| {
            let fs = t.markers.iter().map(|m| {
                if last {
                    marker_last(m, c)
                } else {
                    marker_first(m, c)
                }
            });
            let f = fs
                .filter(|f| *f >= 0)
                .reduce(|a, b| if last { a.max(b) } else { a.min(b) });
            f.map_or_else(|| "NA".to_owned(), |f| f.to_string())
        })
        .collect::<Vec<_>>()
        .join(" / ")
}

/// Ported from core/Trial.cpp, Trial::getReprojectionError: markers' errors weighted by
/// their tracked frames.
fn trial_error(t: &Trial) -> f64 {
    let mut count = 0usize;
    let mut error = 0.0;
    for m in &t.markers {
        let c = marker_tracked_3d(m);
        error += c as f64 * marker_error(m);
        count += c;
    }
    if count != 0 {
        error /= count as f64;
    }
    error
}

/// Ported from core/RigidBody.cpp, RigidBody::getError3D over the whole trial.
fn body_error(b: &RigidBody, filtered: bool) -> f64 {
    let track = if filtered { &b.filtered } else { &b.raw };
    let mut e = 0.0;
    let mut n = 0usize;
    for (i, v) in track.valid.iter().enumerate() {
        if *v {
            n += 1;
            e += track.errors.get(i).map_or(0.0, |x| x.mean_3d);
        }
    }
    if n != 0 {
        e /= n as f64;
    }
    e
}

fn na(v: f64) -> String {
    if v <= 0.0 { "NA".to_owned() } else { qnum(v) }
}

fn na_frame(v: Option<usize>) -> String {
    v.map_or_else(|| "NA".to_owned(), |f| (f + 1).to_string())
}

fn marker_item(t: &Trial, idx: usize, n_cameras: usize) -> TreeItem {
    let m = &t.markers[idx];
    let mut it = TreeItem::new("Marker", m.description.clone());
    it.add(TreeItem::new("marker id", (idx + 1).to_string()));
    let per_cam =
        |f: &dyn Fn(usize) -> String| (0..n_cameras).map(f).collect::<Vec<_>>().join(" / ");
    let na_i = |v: i64| {
        if v >= 0 {
            v.to_string()
        } else {
            "NA".to_owned()
        }
    };
    it.add(TreeItem::new(
        "first tracked frame",
        per_cam(&|c| na_i(marker_first(m, c))),
    ));
    it.add(TreeItem::new(
        "last tracked frame",
        per_cam(&|c| na_i(marker_last(m, c))),
    ));
    it.add(TreeItem::new(
        "number tracked",
        per_cam(&|c| marker_tracked(m, c).to_string()),
    ));
    it.add(TreeItem::new(
        "mean reprojection error",
        na(marker_error(m)),
    ));
    it
}

/// Ported from ui/ProjectOverview.cpp, ProjectOverview::update: the tree's items.
#[must_use]
pub fn tree(doc: &xma_engine::Document) -> Vec<TreeItem> {
    let p = doc.project();
    let compat = doc.compat();
    let n_cameras = p.cameras.len();
    let study = Study::parse_project(&p.portal_metadata);
    let mut out = vec![
        TreeItem::new("Repository", study.repository.clone()),
        TreeItem::new("Created", p.date_created.clone()),
    ];
    if p.calibration_type != CalibrationType::None {
        let name = if p.portal_metadata.is_empty() {
            format!("XMALab_name: {}", study.study_name)
        } else {
            format!("name: {}", study.study_name)
        };
        let mut item = TreeItem::new("Calibration Trial", name);
        let cam_label = |i: usize, portal: i32| {
            (
                format!("Camera {}", i + 1),
                if portal > 0 {
                    format!("(cam{portal})")
                } else {
                    String::new()
                },
            )
        };
        if p.cameras.iter().any(|c| c.undistortion.is_some()) {
            let und = item.add(TreeItem::new("Undistortion", ""));
            for (i, c) in p.cameras.iter().enumerate() {
                if let Some(u) = &c.undistortion {
                    let (a, b) = cam_label(i, c.portal_id);
                    let ci = und.add(TreeItem::new(a, b));
                    ci.add(TreeItem::new(
                        "Undistortion File",
                        file_name(&u.image_filename),
                    ));
                }
            }
        }
        let calib = item.add(TreeItem::new("Calibration", ""));
        for (i, c) in p.cameras.iter().enumerate() {
            let (a, b) = cam_label(i, c.portal_id);
            let ci = calib.add(TreeItem::new(a, b));
            // Camera::getCalibrationNbInlier: the inliers of the calibrated images.
            let inliers = |im: &xma_model::CalibrationImage| {
                if im.calibrated == 1 {
                    im.inlier.iter().filter(|v| **v > 0).count()
                } else {
                    0
                }
            };
            let total: usize = c.calibration_images.iter().map(inliers).sum();
            ci.add(TreeItem::new("Camera points", total.to_string()));
            ci.add(TreeItem::new("Camera error", ""));
            if let Some(u) = &c.undistortion {
                ci.add(TreeItem::new(
                    "Undistortion File",
                    file_name(&u.image_filename),
                ));
            }
            for im in &c.calibration_images {
                let fi = ci.add(TreeItem::new("Calib. file:", file_name(&im.filename)));
                fi.add(TreeItem::new("Camera file points", inliers(im).to_string()));
                fi.add(TreeItem::new("Camera file error", ""));
            }
        }
        let object = match &p.calibration_object {
            Some(CalibrationObject::Checkerboard {
                horizontal_squares,
                vertical_squares,
                square_size,
                ..
            }) => {
                let mut o = TreeItem::new("Calibration Object", "Checkerboard");
                o.add(TreeItem::new("Width", horizontal_squares.to_string()));
                o.add(TreeItem::new("Height", vertical_squares.to_string()));
                o.add(TreeItem::new("Size", qnum(*square_size)));
                o
            }
            Some(CalibrationObject::Frame(f)) => {
                let mut o = TreeItem::new("Calibration Object", "Framespec");
                o.add(TreeItem::new(
                    "Calib. framespec",
                    file_name(&f.frame_specification_filename),
                ));
                o.add(TreeItem::new(
                    "Calib. references",
                    file_name(&f.references_filename),
                ));
                o
            }
            None => {
                let mut o = TreeItem::new("Calibration Object", "Framespec");
                o.add(TreeItem::new("Calib. framespec", ""));
                o.add(TreeItem::new("Calib. references", ""));
                o
            }
        };
        item.add(object);
        out.push(item);
    }
    for t in &p.trials {
        if t.is_default {
            continue;
        }
        let has_study = !t.portal_metadata.is_empty();
        let tstudy = Study::parse(&t.portal_metadata);
        let mut ti = TreeItem::new(
            "Trial",
            if has_study {
                format!("name: {}", tstudy.trial_name)
            } else {
                format!("XMALab_name: {}", t.name)
            },
        );
        ti.add(TreeItem::new(
            "first tracked frame",
            trial_range(t, n_cameras, false),
        ));
        ti.add(TreeItem::new(
            "last tracked frame",
            trial_range(t, n_cameras, true),
        ));
        ti.add(TreeItem::new(
            "trial filter freq.",
            qnum(t.cutoff_frequency),
        ));
        ti.add(TreeItem::new("frame rate", qnum(t.recording_speed)));
        ti.add(TreeItem::new("mean reprojection error", na(trial_error(t))));
        ti.add(TreeItem::new(
            "intermarker dist. mean s.d.",
            na(trial_marker_to_marker_sd(t, compat)),
        ));
        let movies = ti.add(TreeItem::new("Movies", ""));
        for (m, v) in t.videos.iter().enumerate() {
            let name = if has_study {
                let portal = p.cameras.get(m).map_or(-1, |c| c.portal_id);
                video(&t.portal_metadata, m, portal).filename
            } else {
                file_name(v.filenames.first().map_or("", String::as_str))
            };
            movies.add(TreeItem::new(format!("Camera {}", m + 1), name));
        }
        let mut written = vec![false; t.markers.len()];
        for b in &t.rigid_bodies {
            let mut bi = TreeItem::new("Rigid Body name", b.description.clone());
            let first = b.raw.valid.iter().position(|v| *v);
            let last = b.raw.valid.iter().rposition(|v| *v);
            bi.add(TreeItem::new("first tracked", na_frame(first)));
            bi.add(TreeItem::new("last tracked frame", na_frame(last)));
            let n = b.raw.valid.iter().filter(|v| **v).count();
            bi.add(TreeItem::new("number tracked", n.to_string()));
            bi.add(TreeItem::new(
                "intermarker dist. mean s.d.",
                na(body_marker_to_marker_sd(t, b, 0, usize::MAX, compat).0),
            ));
            bi.add(TreeItem::new(
                "rigid body error 3D unfilt.",
                na(body_error(b, false)),
            ));
            bi.add(TreeItem::new(
                "rigid body error 3D filt.",
                na(body_error(b, true)),
            ));
            let freq = if b.override_cutoff_frequency {
                b.cutoff_frequency
            } else {
                t.cutoff_frequency
            };
            bi.add(TreeItem::new("rigid body filter freq.", qnum(freq)));
            for &idx in &b.marker_indices {
                if idx < t.markers.len() {
                    written[idx] = true;
                    bi.add(marker_item(t, idx, n_cameras));
                }
            }
            ti.add(bi);
        }
        for (idx, w) in written.iter().enumerate() {
            if !w {
                ti.add(marker_item(t, idx, n_cameras));
            }
        }
        out.push(ti);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_names() {
        assert_eq!(file_name("/a/b/c.tif"), "c.tif");
        assert_eq!(file_name("C:\\a\\c.tif"), "c.tif");
        assert_eq!(file_name("c.tif"), "c.tif");
    }
}
