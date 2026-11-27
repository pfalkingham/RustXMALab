//! Non-finite numbers (NaN, infinities) in the numbers a project file stores (parity
//! ledger F49 to F52).
//!
//! XMALab writes its data files with `std::ofstream << double` (`core/Marker.cpp`,
//! `core/Camera.cpp`, `core/CalibrationImage.cpp`, `core/RigidBody.cpp`), which gives `nan`,
//! `-nan`, `inf` and `-inf` (glibc), and reads most of them back with
//! `for (double v; in >> v; littleHelper::comma(in))`, which stops at such a word: a row
//! that starts with one is skipped (the value is lost without a word), and one with it later
//! leaves the row short, which XMALab then reads past its end. The XML attributes and the
//! `x,y,z` files are read with `QString::toDouble`, which accepts `nan` and `inf`.
//!
//! In the default mode ([`xma_model::CppCompat::nonfinite_values_kept`] off) a project never
//! holds a non-finite stored number: [`clear_non_finite`] puts the "not set" value XMALab
//! uses for each field in its place, and says where. The reader runs it on what it read
//! (each `nan` or `inf` word is read as a value first, so the row is complete), the writer
//! on a copy of what it is given, and the engine after each edit, trial update and load.
//!
//! | Stored number | Not set |
//! |---|---|
//! | 2D point (either coordinate) | the frame reset: (-2, -2), status undefined, size -1 |
//! | 2D size | -1 |
//! | 3D point | (-1000, -1000, -1000), status undefined |
//! | marker reference point | none |
//! | camera matrix | identity, camera and its calibration images not calibrated |
//! | lens distortion | zeros, lens model off |
//! | calibration image rotation or translation | zeros, image not calibrated |
//! | calibration image point (`detected`) | (-1, -1), inlier flag 0 |
//! | calibration image point (`detected_all`) | removed |
//! | undistortion grid points | grid lists emptied, grid not computed |
//! | undistortion grid centre | none |
//! | calibration-frame bead coordinate | 0 (`QString::toDouble`'s value for no number) |
//! | checkerboard square size | 0 |
//! | view scale or bias | 1, 0 |
//! | trial recording speed or cutoff frequency | 0 (a trial's initial value) |
//! | rigid-body reference point | the body's references reset (`RigidBody::resetReferences`) |
//! | virtual point (reference, or other body's point) | the virtual point removed |
//! | virtual point imported coordinate | none at that frame |
//! | rigid-body cutoff frequency | 0, override off |
//! | rigid-body mesh scale | 1 |
//! | a floating-point setting | its default |

use std::sync::Arc;

use xma_model::{
    AlgorithmSettings, CalibrationObject, Camera, ImageAdjustment, Marker, MarkerStatus, Point2,
    Point3, Project, RigidBody, SETTING_NAMES, SettingKind, SettingValue, Trial,
    VirtualPointSource,
    marker::{UNSET_3D, UNSET_SIZE},
};

use crate::project_write::{body_file, calibration_image_base, dummy_file, marker_file};

fn fin2(p: &Point2) -> bool {
    p.x.is_finite() && p.y.is_finite()
}

fn fin3(p: &Point3) -> bool {
    p.x.is_finite() && p.y.is_finite() && p.z.is_finite()
}

/// True when any number a project file stores is NaN or infinite: when
/// [`clear_non_finite`] would replace a value and say so. It reads the project and copies
/// nothing.
///
/// One case is not counted, as `clear_non_finite` replaces it without a sentence: a
/// non-finite size of a marker past the frames of its 2D points, which no project file
/// stores.
#[must_use]
pub fn has_non_finite(project: &Project) -> bool {
    project.cameras.iter().any(|c| !camera_finite(c))
        || !calibration_object_finite(project.calibration_object.as_ref())
        || !settings_finite(&project.algorithm_settings)
        || project.trials.iter().any(|t| trial_reported(t))
}

/// True unless [`clear_calibration_object`] replaces a value.
fn calibration_object_finite(object: Option<&CalibrationObject>) -> bool {
    match object {
        Some(CalibrationObject::Frame(f)) => f
            .points
            .iter()
            .all(|p| p.coords.iter().all(|v| v.is_finite())),
        Some(CalibrationObject::Checkerboard { square_size, .. }) => square_size.is_finite(),
        None => true,
    }
}

/// True unless [`clear_settings`] replaces a value.
fn settings_finite(s: &AlgorithmSettings) -> bool {
    SETTING_NAMES.iter().all(|&(name, kind)| {
        kind != SettingKind::Float
            || !matches!(s.get(name), Some(SettingValue::Float(v)) if !v.is_finite())
            || AlgorithmSettings::default().get(name).is_none()
    })
}

/// True when [`clear_trial`] would replace a value and say so.
fn trial_reported(t: &Trial) -> bool {
    !t.recording_speed.is_finite()
        || !t.cutoff_frequency.is_finite()
        || t.markers.iter().any(|m| marker_reported(m))
        || t.rigid_bodies.iter().any(|b| !body_finite(b))
}

/// True when [`clear_trial`] would replace a value of the marker and say so: as
/// [`marker_finite`], but a size counts only at a frame that has a 2D point.
fn marker_reported(m: &Marker) -> bool {
    if marker_finite(m) {
        return false;
    }
    m.reference_3d.as_ref().is_some_and(|p| !fin3(p))
        || !m.points3d.iter().all(fin3)
        || m.views.iter().any(|v| {
            v.points
                .iter()
                .zip(v.size.iter().map(Some).chain(std::iter::repeat(None)))
                .any(|(p, s)| !fin2(p) || s.is_some_and(|s| !s.is_finite()))
        })
}

/// Replaces every stored number that is NaN or infinite by its field's "not set" value
/// (module documentation) and returns one sentence per value replaced, naming the archive
/// entry and line (or `project.xml` and the attribute) where the project file stores it.
/// Derived values the file does not store (reprojections, errors, poses) are left as they
/// are. A project with only finite numbers is not touched (no `Arc` is copied).
pub fn clear_non_finite(project: &mut Project) -> Vec<String> {
    let mut out = Vec::new();
    for cam in &mut project.cameras {
        if camera_finite(cam) {
            continue;
        }
        clear_camera(Arc::make_mut(cam), &mut out);
    }
    clear_calibration_object(project, &mut out);
    clear_settings(&mut project.algorithm_settings, &mut out);
    for trial in &mut project.trials {
        if trial_finite(trial) {
            continue;
        }
        clear_trial(Arc::make_mut(trial), &mut out);
    }
    out
}

fn note(out: &mut Vec<String>, place: String, what: &str, action: &str) {
    out.push(format!("{place}: {what} is not a finite number; {action}"));
}

fn camera_finite(cam: &Camera) -> bool {
    cam.camera_matrix.iter().all(|v| v.is_finite())
        && cam.distortion.iter().all(|v| v.is_finite())
        && cam
            .image_adjustments
            .iter()
            .all(|a| a.scale.is_finite() && a.bias.is_finite())
        && cam.undistortion.as_ref().is_none_or(|g| {
            g.points_detected.iter().all(fin2)
                && g.grid_distorted.iter().all(fin2)
                && g.grid_references.iter().all(fin2)
                && g.center.as_ref().is_none_or(fin2)
        })
        && cam.calibration_images.iter().all(|im| {
            im.rotation.iter().all(|v| v.is_finite())
                && im.translation.iter().all(|v| v.is_finite())
                && im.detected.iter().all(fin2)
                && im.detected_all.iter().all(fin2)
        })
}

fn clear_camera(cam: &mut Camera, out: &mut Vec<String>) {
    let dir = cam.name.clone();
    for (v, adj) in cam.image_adjustments.iter_mut().enumerate() {
        let d = ImageAdjustment::default();
        if !adj.scale.is_finite() {
            note(
                out,
                format!("project.xml, camera {dir}, view {v} scale"),
                "the scale",
                "set to 1",
            );
            adj.scale = d.scale;
        }
        if !adj.bias.is_finite() {
            note(
                out,
                format!("project.xml, camera {dir}, view {v} bias"),
                "the bias",
                "set to 0",
            );
            adj.bias = d.bias;
        }
    }
    if let Some(row) = (0..3).find(|&r| (0..3).any(|c| !cam.camera_matrix[(r, c)].is_finite())) {
        note(
            out,
            format!("{dir}/data/{dir}_CameraMatrix.csv, line {}", row + 1),
            "a camera matrix value",
            "the camera is not calibrated",
        );
        cam.camera_matrix = xma_model::Matrix3::identity();
        cam.calibrated = false;
        for im in &mut cam.calibration_images {
            im.calibrated = 0;
        }
    }
    if let Some(k) = cam.distortion.iter().position(|v| !v.is_finite()) {
        note(
            out,
            format!("{dir}/data/{dir}_UndistortionParameter.csv, line {}", k + 1),
            "a lens distortion value",
            "the lens model is off",
        );
        cam.distortion = [0.0; 8];
        cam.has_model_distortion = false;
    }
    if let Some(g) = &mut cam.undistortion {
        let base = crate::project_write::complete_base_name(&g.image_filename).to_owned();
        let lists = [
            ("PointsDetected", &g.points_detected),
            ("GridPointsDistorted", &g.grid_distorted),
            ("GridPointsReferences", &g.grid_references),
        ];
        if let Some((name, line)) = lists
            .iter()
            .find_map(|(n, l)| l.iter().position(|p| !fin2(p)).map(|i| (*n, i + 1)))
        {
            note(
                out,
                format!("{dir}/data/{base}_{name}.csv, line {line}"),
                "an undistortion grid point",
                "the grid is not computed",
            );
            g.points_detected.clear();
            g.grid_distorted.clear();
            g.grid_references.clear();
            g.grid_inlier.clear();
            g.computed = false;
        }
        if g.center.as_ref().is_some_and(|c| !fin2(c)) {
            note(
                out,
                format!("project.xml, camera {dir}, undistortion grid centre"),
                "the centre",
                "no centre is set",
            );
            g.center = None;
        }
    }
    for index in 0..cam.calibration_images.len() {
        let base = calibration_image_base(cam, index);
        let im = &mut cam.calibration_images[index];
        if let Some(k) = im.rotation.iter().position(|v| !v.is_finite()) {
            note(
                out,
                format!("{dir}/data/{base}_RotationMatrix.csv (rotation {})", k + 1),
                "a calibration image rotation",
                "the image is not calibrated",
            );
            im.rotation = xma_model::Vector3::zeros();
            im.calibrated = 0;
        }
        if let Some(k) = im.translation.iter().position(|v| !v.is_finite()) {
            note(
                out,
                format!("{dir}/data/{base}_TranslationVector.csv, line {}", k + 1),
                "a calibration image translation",
                "the image is not calibrated",
            );
            im.translation = xma_model::Vector3::zeros();
            im.calibrated = 0;
        }
        for (i, p) in im.detected.iter_mut().enumerate() {
            if !fin2(p) {
                note(
                    out,
                    format!("{dir}/data/{base}_PointsDetected.csv, line {}", i + 1),
                    "a calibration point",
                    "set to (-1, -1), not an inlier",
                );
                *p = Point2::new(-1.0, -1.0);
                if let Some(f) = im.inlier.get_mut(i) {
                    *f = 0;
                }
            }
        }
        if let Some(i) = im.detected_all.iter().position(|p| !fin2(p)) {
            note(
                out,
                format!("{dir}/data/{base}_PointsDetectedAll.csv, line {}", i + 1),
                "a detected point",
                "removed",
            );
            im.detected_all.retain(fin2);
        }
    }
}

fn clear_calibration_object(project: &mut Project, out: &mut Vec<String>) {
    match &mut project.calibration_object {
        Some(CalibrationObject::Frame(f)) => {
            let spec = f.frame_specification_filename.clone();
            for (i, p) in f.points.iter_mut().enumerate() {
                for v in p.coords.iter_mut() {
                    if !v.is_finite() {
                        note(
                            out,
                            format!("CalibrationObject/{spec}, bead {}", i + 1),
                            "a bead coordinate",
                            "read as 0",
                        );
                        *v = 0.0;
                    }
                }
            }
        }
        Some(CalibrationObject::Checkerboard { square_size, .. }) if !square_size.is_finite() => {
            note(
                out,
                "project.xml, attribute SquareSize".to_owned(),
                "the square size",
                "set to 0",
            );
            *square_size = 0.0;
        }
        _ => {}
    }
}

fn clear_settings(s: &mut AlgorithmSettings, out: &mut Vec<String>) {
    let defaults = AlgorithmSettings::default();
    for (name, kind) in SETTING_NAMES {
        if kind != SettingKind::Float {
            continue;
        }
        if let Some(SettingValue::Float(v)) = s.get(name)
            && !v.is_finite()
            && let Some(d) = defaults.get(name)
        {
            note(
                out,
                format!("{}, {name}", crate::settings_entry::SETTINGS_ENTRY),
                "the setting",
                &format!("set to its default {d}"),
            );
            let _ = s.set(name, d);
        }
    }
}

fn trial_finite(t: &Trial) -> bool {
    t.recording_speed.is_finite()
        && t.cutoff_frequency.is_finite()
        && t.markers.iter().all(|m| marker_finite(m))
        && t.rigid_bodies.iter().all(|b| body_finite(b))
}

fn marker_finite(m: &Marker) -> bool {
    m.reference_3d.as_ref().is_none_or(fin3)
        && m.points3d.iter().all(fin3)
        && m.views
            .iter()
            .all(|v| v.points.iter().all(fin2) && v.size.iter().all(|s| s.is_finite()))
}

fn body_finite(b: &RigidBody) -> bool {
    b.cutoff_frequency.is_finite()
        && b.mesh_scale.is_finite()
        && b.reference_points.iter().all(fin3)
        && b.reference_points_original.iter().all(fin3)
        && b.virtual_points.iter().all(|v| {
            fin3(&v.reference)
                && match &v.source {
                    VirtualPointSource::OtherBody { point, .. } => fin3(point),
                    VirtualPointSource::Coordinates(c) => c.iter().flatten().all(fin3),
                }
        })
}

fn clear_trial(t: &mut Trial, out: &mut Vec<String>) {
    let name = t.name.clone();
    if !t.recording_speed.is_finite() {
        note(
            out,
            format!("project.xml, trial {name}, attribute recordingSpeed"),
            "the recording speed",
            "set to 0",
        );
        t.recording_speed = 0.0;
    }
    if !t.cutoff_frequency.is_finite() {
        note(
            out,
            format!("project.xml, trial {name}, attribute cutoffFrequency"),
            "the cutoff frequency",
            "set to 0",
        );
        t.cutoff_frequency = 0.0;
    }
    for (k, m) in t.markers.iter_mut().enumerate() {
        if marker_finite(m) {
            continue;
        }
        let m = Arc::make_mut(m);
        for (c, v) in m.views.iter_mut().enumerate() {
            for f in 0..v.points.len() {
                if !fin2(&v.points[f]) {
                    note(
                        out,
                        format!(
                            "{}, line {}, camera {}",
                            marker_file(&name, k, "points2d"),
                            f + 1,
                            c + 1
                        ),
                        "a 2D point",
                        "the point is not set",
                    );
                    v.reset(f);
                } else if v.size.get(f).is_some_and(|s| !s.is_finite()) {
                    note(
                        out,
                        format!(
                            "{}, line {}, camera {}",
                            marker_file(&name, k, "size"),
                            f + 1,
                            c + 1
                        ),
                        "a marker size",
                        "the size is not set",
                    );
                    v.size[f] = UNSET_SIZE;
                }
            }
            // Sizes past the points (not written by a well-formed project).
            for s in v.size.iter_mut().skip(v.points.len()) {
                if !s.is_finite() {
                    *s = UNSET_SIZE;
                }
            }
        }
        for f in 0..m.points3d.len() {
            if !fin3(&m.points3d[f]) {
                note(
                    out,
                    format!("{}, line {}", marker_file(&name, k, "points3d"), f + 1),
                    "a 3D point",
                    "the point is not set",
                );
                m.points3d[f] = Point3::new(UNSET_3D, UNSET_3D, UNSET_3D);
                if let Some(s) = m.status3d.get_mut(f) {
                    *s = MarkerStatus::Undefined;
                }
            }
        }
        if m.reference_3d.as_ref().is_some_and(|p| !fin3(p)) {
            note(
                out,
                format!("{}, line 1", marker_file(&name, k, "reference3Dpoint")),
                "the marker's reference point",
                "the reference point is not set",
            );
            m.reference_3d = None;
        }
    }
    for k in 0..t.rigid_bodies.len() {
        if body_finite(&t.rigid_bodies[k]) {
            continue;
        }
        let markers = t.markers.clone();
        let b = Arc::make_mut(&mut t.rigid_bodies[k]);
        clear_body(&name, k, b, &markers, out);
    }
}

fn clear_body(
    trial: &str,
    k: usize,
    b: &mut RigidBody,
    markers: &[Arc<xma_model::Marker>],
    out: &mut Vec<String>,
) {
    if !b.cutoff_frequency.is_finite() {
        note(
            out,
            format!(
                "project.xml, trial {trial}, rigid body {}, CutOffFrequency",
                k + 1
            ),
            "the cutoff frequency",
            "set to 0, override off",
        );
        b.cutoff_frequency = 0.0;
        b.override_cutoff_frequency = false;
    }
    if !b.mesh_scale.is_finite() {
        note(
            out,
            format!(
                "project.xml, trial {trial}, rigid body {}, MeshScale",
                k + 1
            ),
            "the mesh scale",
            "set to 1",
        );
        b.mesh_scale = 1.0;
    }
    let bad_ref = b.reference_points.iter().position(|p| !fin3(p));
    let bad_orig = b.reference_points_original.iter().position(|p| !fin3(p));
    if bad_ref.is_some() || bad_orig.is_some() {
        let (file, line) = match (bad_ref, b.has_optimized_coordinates) {
            (Some(i), true) => (body_file(trial, k, "ReferencePoints3d_optimized"), i + 1),
            (Some(i), false) => (body_file(trial, k, "ReferencePoints3d"), i + 1),
            (None, _) => (
                body_file(trial, k, "ReferencePoints3d"),
                bad_orig.unwrap_or(0) + 1,
            ),
        };
        note(
            out,
            format!("{file}, line {line}"),
            "a reference point",
            "the body's references are reset",
        );
        reset_references(b, markers);
    }
    let mut p = 0;
    let mut index = 0;
    while p < b.virtual_points.len() {
        let v = &mut b.virtual_points[p];
        let keep = fin3(&v.reference)
            && match &v.source {
                VirtualPointSource::OtherBody { point, .. } => fin3(point),
                VirtualPointSource::Coordinates(_) => true,
            };
        if !keep {
            note(
                out,
                format!("{}, line 2", dummy_file(trial, k, index, "PointReferences")),
                "a virtual point",
                "the virtual point is removed",
            );
            b.virtual_points.remove(p);
            index += 1;
            continue;
        }
        if let VirtualPointSource::Coordinates(c) = &mut v.source {
            for (f, slot) in c.iter_mut().enumerate() {
                if slot.as_ref().is_some_and(|q| !fin3(q)) {
                    note(
                        out,
                        format!(
                            "{}, line {}",
                            dummy_file(trial, k, index, "PointCoordinates"),
                            f + 2
                        ),
                        "a virtual point coordinate",
                        "no coordinate at that frame",
                    );
                    *slot = None;
                }
            }
        }
        p += 1;
        index += 1;
    }
}

/// `RigidBody::resetReferences` on the stored data (as the reader does for a body without
/// stored references): names cleared, points at the origin, not set, then the markers'
/// reference points taken when every marker of the body has one
/// (`setReferenceMarkerReferences`).
fn reset_references(b: &mut RigidBody, markers: &[Arc<xma_model::Marker>]) {
    let origin = Point3::new(0.0, 0.0, 0.0);
    for p in b
        .reference_points
        .iter_mut()
        .chain(b.reference_points_original.iter_mut())
    {
        *p = origin;
    }
    for n in &mut b.reference_names {
        n.clear();
    }
    b.references_set = 0;
    b.has_optimized_coordinates = false;
    let all = b
        .marker_indices
        .iter()
        .all(|&i| markers.get(i).is_some_and(|m| m.reference_3d.is_some()));
    if all {
        for (k, &i) in b.marker_indices.iter().enumerate() {
            let m = &markers[i];
            let Some(p) = m.reference_3d else { continue };
            if let Some(s) = b.reference_points.get_mut(k) {
                *s = p;
            }
            if let Some(s) = b.reference_points_original.get_mut(k) {
                *s = p;
            }
            if let Some(s) = b.reference_names.get_mut(k) {
                *s = format!("{} - FromMarker", m.description);
            }
        }
        b.references_set = 1;
    }
    b.initialised = b.references_set != 0;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project_with_trial() -> Project {
        let mut p = Project::default();
        p.cameras.push(Arc::new(Camera::new("cam1".to_owned())));
        p.cameras.push(Arc::new(Camera::new("cam2".to_owned())));
        let mut t = Trial::default_trial();
        t.name = "T".to_owned();
        t.n_frames = 4;
        t.markers.push(Arc::new(Marker::new(2, 4, 8)));
        p.trials.push(Arc::new(t));
        p
    }

    #[test]
    fn finite_project_is_untouched() {
        let mut p = project_with_trial();
        let before = p.clone();
        assert!(clear_non_finite(&mut p).is_empty());
        assert!(Arc::ptr_eq(&p.trials[0], &before.trials[0]));
        assert!(!has_non_finite(&p));
    }

    #[test]
    fn each_field_is_set_to_not_set() {
        let mut p = project_with_trial();
        {
            let t = Arc::make_mut(&mut p.trials[0]);
            t.recording_speed = f64::INFINITY;
            let m = Arc::make_mut(&mut t.markers[0]);
            m.views[1].points[2] = Point2::new(f64::NAN, 4.0);
            m.views[1].status[2] = MarkerStatus::Set;
            m.views[1].size[2] = 3.0;
            m.views[0].points[1] = Point2::new(5.0, 6.0);
            m.views[0].size[1] = f64::NEG_INFINITY;
            m.points3d[3] = Point3::new(1.0, f64::NAN, 2.0);
            m.status3d[3] = MarkerStatus::Set;
            m.reference_3d = Some(Point3::new(f64::INFINITY, 0.0, 0.0));
            let c = Arc::make_mut(&mut p.cameras[1]);
            c.camera_matrix[(1, 2)] = f64::NAN;
            c.calibrated = true;
            p.algorithm_settings.maximum_reprojection_error = f64::NAN;
        }
        assert!(has_non_finite(&p));
        let msgs = clear_non_finite(&mut p);
        assert_eq!(msgs.len(), 7, "{msgs:#?}");
        assert!(msgs[0].starts_with("cam2/data/cam2_CameraMatrix.csv, line 2"));
        assert!(
            msgs.iter()
                .any(|m| m.starts_with("T/data/Marker000points2d.csv, line 3, camera 2"))
        );
        let m = &p.trials[0].markers[0];
        assert_eq!(m.views[1].points[2], Point2::new(-2.0, -2.0));
        assert_eq!(m.views[1].status[2], MarkerStatus::Undefined);
        assert_eq!(m.views[1].size[2], UNSET_SIZE);
        assert_eq!(m.views[0].points[1], Point2::new(5.0, 6.0));
        assert_eq!(m.views[0].size[1], UNSET_SIZE);
        assert_eq!(m.points3d[3], Point3::new(UNSET_3D, UNSET_3D, UNSET_3D));
        assert_eq!(m.status3d[3], MarkerStatus::Undefined);
        assert_eq!(m.reference_3d, None);
        assert_eq!(p.trials[0].recording_speed, 0.0);
        assert!(!p.cameras[1].calibrated);
        assert_eq!(
            p.algorithm_settings.maximum_reprojection_error,
            AlgorithmSettings::default().maximum_reprojection_error
        );
        assert!(!has_non_finite(&p));
    }
}
