//! What the camera views and their info frames show in the Undistortion and Calibration
//! workspaces, as plain queries over the document.
//!
//! - [`calibration_image`]: the point sets of one calibration image as
//!   `core/CalibrationImage.cpp` holds them after loading (detected, detected
//!   undistorted, projected, projected undistorted, inlier flags, errors), with the
//!   projection `MultiCameraCalibration::reproject` makes when a project is loaded.
//! - [`calibration_info`]: the texts of `ui/CalibrationInfoFrame.cpp`.
//! - [`undistortion_grid`]: the point sets of `core/UndistortionObject.cpp`
//!   (`points_detected`, `points_grid_distorted`, `points_grid_undistorted`,
//!   `points_grid_references`, inliers, centre, errors) and the texts of
//!   `ui/UndistortionInfoFrame.cpp`.
//! - [`grid_lookup_table`]: the camera's undistortion lookup table
//!   (`LocalUndistortion::createLookupTable`), for the undistorted image and the maps.
//!
//! # Fixed C++ defect
//!
//! `UndistortionInfoFrame::getInfo` divides by `count - 1` whenever `count > 0`, so a grid
//! with a single inlier shows "nan" as its spread. Here the spread is 0 then, as
//! `CalibrationInfoFrame` computes it (`count > 1`).

use xma_model::Point2;
use xma_vision::camera::{RigidTransform, project_points_f32};
use xma_vision::undistortion::{LookupTable, create_lookup_table, transform_point};

use crate::Document;

/// A calibration image's points, all in raw (distorted) or undistorted image coordinates
/// as named, one entry per calibration-object point. A point with no position (an unset
/// detection, or, in the fixed mode, a lens undistortion with no result, ledger A19) is
/// (-1, -1); no coordinate is ever non-finite.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CalibrationImagePoints {
    /// Every point the detection found (`detectedPoints_ALL`).
    pub detected_all: Vec<Point2>,
    /// Matched detection per object point, (-1, -1) where unset.
    pub detected: Vec<Point2>,
    pub detected_undistorted: Vec<Point2>,
    /// The object points projected with the image's pose; empty when the image is not
    /// calibrated.
    pub projected: Vec<Point2>,
    pub projected_undistorted: Vec<Point2>,
    /// 1 inlier, 0 outlier, -1 not detected (`CalibrationImage::Inlier`).
    pub inlier: Vec<i32>,
    /// Distance between detected and projected, raw and undistorted (`error`,
    /// `errorUndistorted`); empty when there is no projection.
    pub error: Vec<f64>,
    pub error_undistorted: Vec<f64>,
    /// `isCalibrated()`.
    pub calibrated: i32,
}

/// The points of calibration image `image` of `camera`, as XMALab holds them: the lists the
/// calibration actions of this session left ([`crate::calibration::HeldPoints`]), or, for an
/// image no action has touched, [`calibration_image_as_loaded`].
#[must_use]
pub fn calibration_image(
    doc: &Document,
    camera: usize,
    image: usize,
) -> Option<CalibrationImagePoints> {
    let held = doc
        .camera_held_points(camera, image)
        .map(std::sync::Arc::clone);
    let Some(h) = held else {
        return calibration_image_as_loaded(doc, camera, image);
    };
    let cam = doc.project().cameras.get(camera)?;
    let im = cam.calibration_images.get(image)?;
    let fin = |l: &[Point2]| l.iter().map(|p| or_unset(Some(*p))).collect::<Vec<_>>();
    Some(CalibrationImagePoints {
        detected_all: im.detected_all.clone(),
        detected: im.detected.clone(),
        detected_undistorted: fin(&h.detected_undistorted),
        projected: fin(&h.projected),
        projected_undistorted: fin(&h.projected_undistorted),
        inlier: im.inlier.clone(),
        error: h.error.clone(),
        error_undistorted: h.error_undistorted.clone(),
        calibrated: im.calibrated,
    })
}

/// The points of calibration image `image` of `camera` as they stand after a project is
/// loaded.
///
/// Ported from `core/CalibrationImage.cpp` (`undistortPoints`, `setPointsProjectedUndistorted`,
/// `computeError`) and `processing/MultiCameraCalibration.cpp`, `reproject` (`cv::projectPoints`
/// of the frame's points as `float` with the image's pose and no distortion, then
/// `Camera::undistortPoint(pt, false)`), as they stand after a project is loaded.
#[must_use]
pub fn calibration_image_as_loaded(
    doc: &Document,
    camera: usize,
    image: usize,
) -> Option<CalibrationImagePoints> {
    let cam = doc.project().cameras.get(camera)?;
    let im = cam.calibration_images.get(image)?;
    let model = &doc.camera(camera).model;
    // Ledger A124, XMALab 3 mode: after an optimisation with lens distortion the detected
    // points were undistorted with the camera matrix from before it.
    let lens_model = doc.camera(camera).lens_points_model();
    let mut out = CalibrationImagePoints {
        detected_all: im.detected_all.clone(),
        detected: im.detected.clone(),
        detected_undistorted: im
            .detected
            .iter()
            .map(|p| or_unset(lens_model.try_undistort_point(*p, true, true, true)))
            .collect(),
        inlier: im.inlier.clone(),
        calibrated: im.calibrated,
        ..Default::default()
    };
    // The frame's beads or the checkerboard's corners.
    let object = doc
        .project()
        .calibration_object
        .as_ref()
        .map(xma_model::CalibrationObject::points)
        .filter(|p| !p.is_empty());
    if im.calibrated > 0
        && let Some(points) = object
    {
        let pts: Vec<_> = points.iter().map(|p| p.cast::<f32>()).collect();
        let pose = RigidTransform {
            rotation: im.rotation,
            translation: im.translation,
        };
        out.projected_undistorted = project_points_f32(
            &pts,
            &pose.rotation,
            &pose.translation,
            &model.intrinsics.camera_matrix,
            &[],
        )
        .into_iter()
        .map(|p| Point2::new(f64::from(p.x), f64::from(p.y)))
        .collect();
        out.projected = out
            .projected_undistorted
            .iter()
            .map(|p| or_unset(Some(model.undistort_point(*p, false, true, true))))
            .collect();
        for p in &mut out.projected_undistorted {
            *p = or_unset(Some(*p));
        }
        if out.detected.len() == out.projected.len() {
            let dist = |a: &Point2, b: &Point2| ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
            out.error = out
                .detected
                .iter()
                .zip(&out.projected)
                .map(|(a, b)| dist(a, b))
                .collect();
            out.error_undistorted = out
                .detected_undistorted
                .iter()
                .zip(&out.projected_undistorted)
                .map(|(a, b)| dist(a, b))
                .collect();
        }
    }
    Some(out)
}

/// A point, or (-1, -1), the value of an unset point, when it is absent or not finite.
fn or_unset(p: Option<Point2>) -> Point2 {
    p.filter(|p| p.x.is_finite() && p.y.is_finite())
        .unwrap_or_else(|| Point2::new(-1.0, -1.0))
}

/// The texts of the Calibration Info frame for `camera` with calibration image `image`
/// active. Empty strings where the C++ leaves a label empty.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CalibrationInfo {
    pub focal_length: String,
    pub camera_center: String,
    pub distortion: String,
    pub rotation_vector: String,
    pub translation_vector: String,
    pub error_current_dist: String,
    pub error_all_dist: String,
    pub error_current_undist: String,
    pub error_all_undist: String,
    pub frames_calibrated: String,
    pub inlier: String,
}

/// `mean +/- sd` with two decimals over `values`, the sample standard deviation when
/// there are at least two values, else 0.
fn mean_sd(values: &[f64]) -> String {
    let n = values.len();
    let mean = if n > 0 {
        values.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };
    let sd = if n > 1 {
        (values.iter().map(|v| (mean - v).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    } else {
        0.0
    };
    format!("{mean:.2} +/- {sd:.2}")
}

/// Ported from `ui/CalibrationInfoFrame.cpp`, `getCameraInfo`, `updateFrame`, `getInfoFrame`
/// and `getInfoInlier`.
#[must_use]
pub fn calibration_info(doc: &Document, camera: usize, image: usize) -> CalibrationInfo {
    let mut info = CalibrationInfo::default();
    let Some(cam) = doc.project().cameras.get(camera) else {
        return info;
    };
    let images: Vec<CalibrationImagePoints> = (0..cam.calibration_images.len())
        .filter_map(|i| calibration_image(doc, camera, i))
        .collect();
    // getInfoInlier: shown whether or not the camera is calibrated.
    let mut all = 0;
    let mut current = 0;
    for (f, im) in images.iter().enumerate() {
        if im.calibrated == 1 {
            let n = im.inlier.iter().filter(|&&v| v > 0).count();
            all += n;
            if f == image {
                current = n;
            }
        }
    }
    info.inlier = format!("{current} / {all}");
    if !cam.calibrated {
        return info;
    }
    let k = &cam.camera_matrix;
    info.focal_length = format!("{:.2},{:.2}", k[(0, 0)], k[(1, 1)]);
    info.camera_center = format!("{:.2},{:.2}", k[(0, 2)], k[(1, 2)]);
    info.distortion = cam
        .distortion
        .iter()
        .map(|d| format!("{d:.5}"))
        .collect::<Vec<_>>()
        .join(",");
    let inlier_errors = |im: &CalibrationImagePoints, undist: bool| -> Vec<f64> {
        let e = if undist {
            &im.error_undistorted
        } else {
            &im.error
        };
        e.iter()
            .enumerate()
            .filter(|(i, _)| im.inlier.get(*i).is_some_and(|&v| v > 0))
            .map(|(_, v)| *v)
            .collect()
    };
    let (mut d_all, mut u_all) = (Vec::new(), Vec::new());
    let mut frames = 0;
    for im in &images {
        if im.calibrated == 1 {
            frames += 1;
            d_all.extend(inlier_errors(im, false));
            u_all.extend(inlier_errors(im, true));
        }
    }
    info.frames_calibrated = format!("{frames} of {}", images.len());
    info.error_all_dist = mean_sd(&d_all);
    info.error_all_undist = mean_sd(&u_all);
    if let (Some(im), Some(stored)) = (images.get(image), cam.calibration_images.get(image))
        && im.calibrated == 1
    {
        info.error_current_dist = mean_sd(&inlier_errors(im, false));
        info.error_current_undist = mean_sd(&inlier_errors(im, true));
        let v3 = |v: &xma_model::Vector3| format!("{:.2},{:.2},{:.2}", v[0], v[1], v[2]);
        info.rotation_vector = v3(&stored.rotation);
        info.translation_vector = v3(&stored.translation);
    }
    info
}

/// The undistortion grid of one camera, as `core/UndistortionObject.cpp` holds and draws it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UndistortionGridPoints {
    /// Bead centres found in the grid image (`points_detected`).
    pub detected: Vec<Point2>,
    /// Matched beads in the raw image (`points_grid_distorted`).
    pub grid_distorted: Vec<Point2>,
    /// The same undistorted with the fitted grid (`points_grid_undistorted`); empty when
    /// the grid is not computed.
    pub grid_undistorted: Vec<Point2>,
    /// Ideal positions (`points_grid_references`).
    pub references: Vec<Point2>,
    pub inlier: Vec<bool>,
    /// The centre chosen by the user, if any.
    pub center: Option<Point2>,
    /// `|grid_undistorted - references|` per bead (`computeError`).
    pub error: Vec<f64>,
    /// The grid is computed (`isComputed`).
    pub computed: bool,
    /// "Nb Points :" text: "<inliers> of <beads>", empty when not computed.
    pub nb_points_text: String,
    /// "Error :" text: "<mean> +/- <sd>" over the inliers, empty when not computed.
    pub error_text: String,
}

/// The grid of `camera`, if it has one.
///
/// Ported from `core/UndistortionObject.cpp` (`undistortPoints`, `computeError`) and
/// `ui/UndistortionInfoFrame.cpp` (`update`, `getInfo`; fixed as the module documentation
/// says).
#[must_use]
pub fn undistortion_grid(doc: &Document, camera: usize) -> Option<UndistortionGridPoints> {
    let cam = doc.project().cameras.get(camera)?;
    let g = cam.undistortion.as_ref()?;
    let mut out = UndistortionGridPoints {
        detected: g.points_detected.clone(),
        grid_distorted: g.grid_distorted.clone(),
        references: g.grid_references.clone(),
        inlier: g.grid_inlier.clone(),
        center: g.center,
        computed: g.computed,
        ..Default::default()
    };
    if let Some(model) = &doc.camera(camera).model.grid
        && g.computed
    {
        out.grid_undistorted = g
            .grid_distorted
            .iter()
            .map(|p| or_unset(Some(transform_point(model, *p, true, true))))
            .collect();
        out.error = out
            .grid_undistorted
            .iter()
            .zip(&out.references)
            .map(|(u, r)| ((u.x - r.x).powi(2) + (u.y - r.y).powi(2)).sqrt())
            .collect();
        let inl: Vec<f64> = out
            .error
            .iter()
            .zip(&out.inlier)
            .filter(|(_, i)| **i)
            .map(|(e, _)| *e)
            .collect();
        out.nb_points_text = format!("{} of {}", inl.len(), out.inlier.len());
        out.error_text = mean_sd(&inl);
    }
    Some(out)
}

/// The lookup table of `camera`'s fitted grid: for each pixel of the undistorted image,
/// where to sample the raw image (-1 where not covered). `None` without a computed grid.
///
/// Ported from `processing/LocalUndistortion.cpp`, `createLookupTable`, on the distorting
/// transform, as the C++ builds `undistortionMapX/Y` after loading.
#[must_use]
pub fn grid_lookup_table(doc: &Document, camera: usize) -> Option<LookupTable> {
    let cam = doc.project().cameras.get(camera)?;
    let grid = doc.camera(camera).model.grid.as_ref()?;
    if let Some(l) = &doc.camera(camera).lookup {
        return Some(LookupTable::clone(l));
    }
    Some(create_lookup_table(
        &grid.distort,
        cam.width as usize,
        cam.height as usize,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pig() -> Option<(xma_oracle::Dataset, Document)> {
        let ds = xma_oracle::dataset("pig_clean")?;
        let doc = Document::open(&ds.project).expect("pig opens");
        Some((ds, doc))
    }

    /// Detected points and inlier flags are the C++ ones (`calibration/cube_points.csv`);
    /// the projection reproduces the detections of inliers within the calibration error.
    #[test]
    fn pig_calibration_points_match_the_dump() {
        let Some((ds, doc)) = pig() else { return };
        let t = ds.table("calibration/cube_points.csv");
        let mut checked = 0;
        for camera in 0..2 {
            let p = calibration_image(&doc, camera, 0).expect("image");
            assert_eq!(p.calibrated, 1);
            assert_eq!(p.projected.len(), p.detected.len());
            for row in t
                .rows()
                .filter(|r| r.usize("camera") == camera && r.usize("image") == 0)
            {
                let i = row.usize("index");
                assert_eq!(p.detected[i].x, row.f64("detected_x"));
                assert_eq!(p.detected[i].y, row.f64("detected_y"));
                assert_eq!(i64::from(p.inlier[i]), row.i64("inlier"));
                if p.inlier[i] == 1 {
                    assert!(
                        p.error[i] < 3.0,
                        "camera {camera} point {i}: error {}",
                        p.error[i]
                    );
                }
                checked += 1;
            }
            let info = calibration_info(&doc, camera, 0);
            assert_eq!(info.frames_calibrated, "1 of 1");
            assert!(info.error_all_dist.contains("+/-"));
            assert_eq!(info.error_all_dist, info.error_current_dist);
        }
        assert!(checked > 100, "{checked}");
    }

    /// The grid's beads and inliers are the C++ ones (`undistortion/grid_points.csv`), and
    /// undistorting the beads gives the dump's own undistortion of the same points
    /// (`undistortion/samples.csv` is checked by the vision tests; here only the texts).
    #[test]
    fn pig_grid_points_match_the_dump() {
        let Some((ds, doc)) = pig() else { return };
        let t = ds.table("undistortion/grid_points.csv");
        for camera in 0..2 {
            let g = undistortion_grid(&doc, camera).expect("grid");
            assert!(g.computed);
            let rows: Vec<_> = t.rows().filter(|r| r.usize("camera") == camera).collect();
            assert_eq!(rows.len(), g.grid_distorted.len());
            for r in &rows {
                let i = r.usize("index");
                assert_eq!(g.grid_distorted[i].x, r.f64("detected_x"));
                assert_eq!(g.references[i].y, r.f64("reference_y"));
                assert_eq!(g.inlier[i], r.i64("inlier") == 1);
            }
            let n_in = g.inlier.iter().filter(|v| **v).count();
            assert_eq!(g.nb_points_text, format!("{n_in} of {}", g.inlier.len()));
            let mean: f64 = g
                .error
                .iter()
                .zip(&g.inlier)
                .filter(|(_, i)| **i)
                .map(|(e, _)| e)
                .sum::<f64>()
                / n_in as f64;
            assert!(mean < 1.0, "mean grid error {mean}");
            let lut = grid_lookup_table(&doc, camera).expect("table");
            assert_eq!(lut.map_x.len(), 1024 * 1024);
        }
    }

    #[test]
    fn spread_of_one_value_is_zero() {
        assert_eq!(mean_sd(&[2.0]), "2.00 +/- 0.00");
        assert_eq!(mean_sd(&[]), "0.00 +/- 0.00");
        assert_eq!(mean_sd(&[1.0, 3.0]), "2.00 +/- 1.41");
    }

    /// The grid image of a pig camera, 8-bit as decoded.
    fn pig_grid_image(doc: &Document, camera: usize) -> (Vec<u8>, usize, usize) {
        let name = &doc.project().cameras[camera]
            .undistortion
            .as_ref()
            .unwrap()
            .image_filename;
        let att = doc
            .extras()
            .attachments
            .iter()
            .find(|a| a.name.ends_with(name.as_str()))
            .expect("grid image");
        let f = xma_video::image_file::decode_image(&att.data, std::path::Path::new(&att.name))
            .unwrap();
        (
            f.to_gray8().into_owned(),
            f.width as usize,
            f.height as usize,
        )
    }

    fn display_dir() -> std::path::PathBuf {
        xma_oracle::oracle_root().join("data/display/remap")
    }

    /// Writes the inputs of `oracle/py/display_remap.py` into
    /// `oracle/data/display/remap`: per pig camera the grid image and its lookup table.
    #[test]
    #[ignore = "writes the inputs of oracle/py/display_remap.py"]
    fn write_pig_remap_inputs() {
        let Some((_, doc)) = pig() else { return };
        let dir = display_dir();
        std::fs::create_dir_all(&dir).unwrap();
        for camera in 0..2 {
            let (src, _, _) = pig_grid_image(&doc, camera);
            let lut = grid_lookup_table(&doc, camera).unwrap();
            let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let w = |n: &str, b: &[u8]| {
                std::fs::write(dir.join(format!("cam{camera}_{n}")), b).unwrap()
            };
            w("src.u8", &src);
            w("map_x.f32", &f32s(&lut.map_x));
            w("map_y.f32", &f32s(&lut.map_y));
        }
    }

    /// The pig grid images undistorted here and by `cv2.remap`, and the displacement maps
    /// here and by the Python port in `oracle/py/display_remap.py`. Remap, distance,
    /// horizontal and vertical maps are identical; the two angle maps differ in at most 13
    /// of 1 048 576 pixels by at most 6 levels (measured; `atan2f` against numpy's
    /// `arctan2`).
    #[test]
    fn pig_remap_and_maps_match_opencv() {
        use xma_vision::remap::{DisplacementMap as D, displacement_image, remap_lanczos4_u8};
        let dir = display_dir();
        if !dir.join("cam0_remap.cv.u8").exists() {
            assert!(
                !xma_oracle::oracle_required(),
                "{} is missing its references: oracle/py/DISPLAY_CHECKS.md",
                dir.display()
            );
            eprintln!("SKIPPED: {} absent", dir.display());
            return;
        }
        let Some((_, doc)) = pig() else { return };
        for camera in 0..2 {
            let read = |n: &str| std::fs::read(dir.join(format!("cam{camera}_{n}"))).unwrap();
            let (src, w, h) = pig_grid_image(&doc, camera);
            let lut = grid_lookup_table(&doc, camera).unwrap();
            // The references were made from these very inputs.
            let mx: Vec<f32> = read("map_x.f32")
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            assert_eq!(
                mx, lut.map_x,
                "camera {camera}: stored lookup table differs"
            );
            assert_eq!(read("src.u8"), src);
            let got =
                remap_lanczos4_u8(&src, w, h, 1, &lut.map_x, &lut.map_y, lut.width, lut.height);
            assert_eq!(got, read("remap.cv.u8"), "camera {camera}: remap");
            for (k, n, max_count, max_diff) in [
                (D::ScaledAngle, "scaled", 13, 6),
                (D::Angle, "angle", 13, 6),
                (D::Length, "length", 0, 0),
                (D::Horizontal, "x", 0, 0),
                (D::Vertical, "y", 0, 0),
            ] {
                let rgba = displacement_image(&lut.map_x, &lut.map_y, lut.width, lut.height, k);
                let want = read(&format!("map_{n}.cv.rgb"));
                let (mut count, mut worst) = (0, 0u8);
                for (px, w3) in rgba.chunks(4).zip(want.chunks(3)) {
                    for c in 0..3 {
                        let d = px[c].abs_diff(w3[c]);
                        worst = worst.max(d);
                        count += usize::from(d > 0);
                    }
                }
                assert!(
                    count <= max_count && worst <= max_diff,
                    "camera {camera} {n}: {count} values differ, largest {worst}"
                );
            }
        }
    }
}
