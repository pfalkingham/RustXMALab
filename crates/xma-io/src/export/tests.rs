//! Unit tests on a small hand-built model, for the options and cases the reference dumps do
//! not cover. Each expected text is derived from the C++ code cited in the test, line by line.

use std::sync::Arc;
use xma_model::CppCompat;

/// The XMALab 3 mode: the tests below check the C++ output byte for byte.
const CPP: CppCompat = CppCompat::XMALAB3;

use xma_model::{
    CalibrationImage, CalibrationSequence, Camera, Event, Marker, MarkerStatus, Matrix3, Point2,
    Point3, Pose, Rgb, RigidBody, Trial, UndistortionGrid, Vector3,
};
use xma_vision::camera::{CameraModel, Intrinsics};
use xma_vision::undistortion::LookupTable;

use super::*;

/// Two cameras: 100 × 50 and 100 × 60 (different heights, to see which one `y_invert` uses).
fn cameras() -> Vec<Arc<Camera>> {
    [50, 60]
        .into_iter()
        .enumerate()
        .map(|(i, h)| {
            let mut c = Camera::new(format!("Camera {}", i + 1));
            c.width = 100;
            c.height = h;
            Arc::new(c)
        })
        .collect()
}

/// Camera models that change nothing (no grid, no lens model).
fn identity_models() -> Vec<CameraModel> {
    (0..2)
        .map(|_| CameraModel {
            compat: xma_vision::camera::ModelCompat::XMALAB3,
            intrinsics: Intrinsics {
                camera_matrix: Matrix3::identity(),
                distortion: None,
            },
            grid: None,
        })
        .collect()
}

/// Three frames, two markers, two bodies.
///
/// Marker 1 (no description):
/// - camera 1: f0 (10.5, 20.25) TRACKED, f1 (11, 21) SET, f2 undefined;
/// - camera 2: f0 (30, 40) TRACKED, f1 LOST, f2 (31.5, 41.5) MANUAL;
/// - 3D: f0 (1, 2, 3) TRACKED, f1 (1.5, −2.5, 0.25) SET, f2 LOST (placeholder point);
/// - 2D errors: camera 1 f0 0.5, f1 0.25; camera 2 f0 0.125, f2 2 (f1 stays 0).
///
/// Marker 2 "Tip": no 2D data; 3D f0 (1, 2, 7) TRACKED, f1 (1.5, −2.5, 3.25) SET.
///
/// Body 1 (no description, markers 1 and 2): raw pose f0 identity rotation with t = (1, 2, 3)
/// and 3D error 0.25, f2 identity with t = 0 and error 0.75; filtered pose f0 with
/// t = (0.5, 0, 0) and error 0.125. Body 2 "Jaw" (marker 2) overrides the cutoff with 40 Hz
/// and has no pose. Trial: 100 frames per second, cutoff 10 Hz.
fn trial() -> Trial {
    let mut t = Trial::default_trial();
    t.name = "Trial 1".to_owned();
    t.is_default = false;
    t.n_frames = 3;
    t.end_frame = 3;
    t.recording_speed = 100.0;
    t.cutoff_frequency = 10.0;

    let mut m = Marker::new(2, 3, 8);
    let v = &mut m.views[0];
    v.points[0] = Point2::new(10.5, 20.25);
    v.status[0] = MarkerStatus::Tracked;
    v.points[1] = Point2::new(11.0, 21.0);
    v.status[1] = MarkerStatus::Set;
    v.error[0] = 0.5;
    v.error[1] = 0.25;
    let v = &mut m.views[1];
    v.points[0] = Point2::new(30.0, 40.0);
    v.status[0] = MarkerStatus::Tracked;
    v.points[1] = Point2::new(99.0, 99.0);
    v.status[1] = MarkerStatus::Lost;
    v.points[2] = Point2::new(31.5, 41.5);
    v.status[2] = MarkerStatus::Manual;
    v.error[0] = 0.125;
    v.error[2] = 2.0;
    m.points3d[0] = Point3::new(1.0, 2.0, 3.0);
    m.status3d[0] = MarkerStatus::Tracked;
    m.points3d[1] = Point3::new(1.5, -2.5, 0.25);
    m.status3d[1] = MarkerStatus::Set;
    m.status3d[2] = MarkerStatus::Lost;
    t.markers.push(Arc::new(m));

    let mut m = Marker::new(2, 3, 8);
    m.description = "Tip".to_owned();
    m.points3d[0] = Point3::new(1.0, 2.0, 7.0);
    m.status3d[0] = MarkerStatus::Tracked;
    m.points3d[1] = Point3::new(1.5, -2.5, 3.25);
    m.status3d[1] = MarkerStatus::Set;
    t.markers.push(Arc::new(m));

    let mut b = RigidBody::new(3);
    b.marker_indices = vec![0, 1];
    b.raw.valid[0] = true;
    b.raw.poses[0] = Pose {
        rotation: Vector3::zeros(),
        translation: Vector3::new(1.0, 2.0, 3.0),
    };
    b.raw.errors[0].mean_3d = 0.25;
    b.raw.valid[2] = true;
    b.raw.errors[2].mean_3d = 0.75;
    b.filtered.valid[0] = true;
    b.filtered.poses[0] = Pose {
        rotation: Vector3::zeros(),
        translation: Vector3::new(0.5, 0.0, 0.0),
    };
    b.filtered.errors[0].mean_3d = 0.125;
    t.rigid_bodies.push(Arc::new(b));

    let mut b = RigidBody::new(3);
    b.description = "Jaw".to_owned();
    b.marker_indices = vec![1];
    b.override_cutoff_frequency = true;
    b.cutoff_frequency = 40.0;
    t.rigid_bodies.push(Arc::new(b));
    t
}

// ---------------------------------------------------------------------------------------
// 2D points (core/Trial.cpp, Trial::save2dPoints)
// ---------------------------------------------------------------------------------------

const HEADER_2D: &str = "marker001_cam1_X,marker001_cam1_Y,marker001_cam2_X,marker001_cam2_Y,\
Tip_cam1_X,Tip_cam1_Y,Tip_cam2_X,Tip_cam2_Y\n";

/// Defaults: the description or `marker%03d`; `X,Y` per camera; `NaN,NaN` where the 2D
/// status is not above 0 (LOST included); `%.17g` (`11`, not `11.0`).
#[test]
fn points_2d_defaults() {
    let text = points2d::points_2d(
        &trial(),
        &cameras(),
        &identity_models(),
        &points2d::Points2dOptions::default(),
    )
    .unwrap()
    .into_file();
    let want = format!(
        "{HEADER_2D}10.5,20.25,30,40,NaN,NaN,NaN,NaN\n\
         11,21,NaN,NaN,NaN,NaN,NaN,NaN\n\
         NaN,NaN,31.5,41.5,NaN,NaN,NaN,NaN\n"
    );
    assert_eq!(text, want);
}

/// Undistorted with models that change nothing gives the distorted text.
#[test]
fn points_2d_undistorted_through_the_model() {
    let opts = points2d::Points2dOptions {
        compat: CPP,
        distorted: false,
        ..Default::default()
    };
    let a = points2d::points_2d_file(&trial(), &cameras(), &identity_models(), &opts);
    let b = points2d::points_2d_file(
        &trial(),
        &cameras(),
        &identity_models(),
        &points2d::Points2dOptions::default(),
    );
    assert_eq!(a, b);
}

/// Offset columns, count from 1 and y up, one file (`:1719-1725`, `:1757-1763`,
/// `:1784-1793`): the header starts `cam0_offsetcam1_offset` with no separator (ledger F2);
/// each row starts `0,0,`; y = height − y − 1 with each camera's own height, then +1 to both.
/// Camera 1 (height 50): (10.5, 20.25) → (11.5, 29.75); (11, 21) → (12, 29).
/// Camera 2 (height 60): (30, 40) → (31, 20); (31.5, 41.5) → (32.5, 18.5).
#[test]
fn points_2d_offsets_and_inverted_y() {
    let opts = points2d::Points2dOptions {
        offset1: true,
        y_invert: true,
        offset_columns: true,
        compat: CPP,
        ..Default::default()
    };
    let text = points2d::points_2d_file(&trial(), &cameras(), &identity_models(), &opts);
    let want = format!(
        "cam0_offsetcam1_offset{HEADER_2D}0,0,11.5,29.75,31,20,NaN,NaN,NaN,NaN\n\
         0,0,12,29,NaN,NaN,NaN,NaN,NaN,NaN\n\
         0,0,NaN,NaN,32.5,18.5,NaN,NaN,NaN,NaN\n"
    );
    assert_eq!(text, want);
}

/// Ledger F2 fixed (the default): one `cam<j+1>_offset` field per camera, each followed by
/// a comma, so the header has as many fields as every data row; the rows are unchanged.
#[test]
fn points_2d_offset_header_fixed() {
    let cpp = points2d::Points2dOptions {
        offset_columns: true,
        compat: CPP,
        ..Default::default()
    };
    let fixed = points2d::Points2dOptions {
        compat: CppCompat::FIXED,
        ..cpp
    };
    let a = points2d::points_2d_file(&trial(), &cameras(), &identity_models(), &fixed);
    let b = points2d::points_2d_file(&trial(), &cameras(), &identity_models(), &cpp);
    assert!(
        a.starts_with(&format!("cam1_offset,cam2_offset,{HEADER_2D}")),
        "{a}"
    );
    let (ha, ra) = a.split_once('\n').unwrap();
    let (hb, rb) = b.split_once('\n').unwrap();
    assert_eq!(ra, rb);
    assert!(hb.starts_with("cam0_offsetcam1_offset"));
    let fields = |l: &str| l.split(',').count();
    for row in ra.lines() {
        assert_eq!(fields(ha), fields(row), "{ha} / {row}");
    }
    assert_eq!(fields(hb) + 2, fields(ra.lines().next().unwrap()));
}

/// One file per marker, no header (`:1816-1913`): `Marker%03d_<description>_points2d.csv`,
/// an empty description giving `Marker001__points2d.csv`; a marker with no data writes only
/// `NaN`.
#[test]
fn points_2d_per_marker_without_header() {
    let opts = points2d::Points2dOptions {
        compat: CPP,
        one_file: false,
        header: false,
        ..Default::default()
    };
    let files = points2d::points_2d(&trial(), &cameras(), &identity_models(), &opts)
        .unwrap()
        .into_folder();
    assert_eq!(
        files,
        vec![
            ExportFile::new(
                "Marker001__points2d.csv".into(),
                "10.5,20.25,30,40\n11,21,NaN,NaN\nNaN,NaN,31.5,41.5\n".into()
            ),
            ExportFile::new(
                "Marker002_Tip_points2d.csv".into(),
                "NaN,NaN,NaN,NaN\nNaN,NaN,NaN,NaN\nNaN,NaN,NaN,NaN\n".into()
            ),
        ]
    );
}

/// One camera (`id`, as "All Trials for External" passes it): the C++ offset header names
/// that camera's 0-based index, the marker columns its 1-based one (ledger F2); fixed, both
/// are 1-based and the offset field is followed by a comma.
#[test]
fn points_2d_one_camera() {
    let opts = points2d::Points2dOptions {
        one_file: false,
        offset_columns: true,
        compat: CPP,
        camera: Some(1),
        ..Default::default()
    };
    let text = points2d::points_2d_marker_file(&trial(), 0, &cameras(), &identity_models(), &opts);
    assert_eq!(
        text,
        "cam1_offsetmarker001_cam2_X,marker001_cam2_Y\n0,30,40\n0,NaN,NaN\n0,31.5,41.5\n"
    );
    let fixed = points2d::Points2dOptions {
        compat: CppCompat::FIXED,
        ..opts
    };
    let text = points2d::points_2d_marker_file(&trial(), 0, &cameras(), &identity_models(), &fixed);
    assert_eq!(
        text,
        "cam2_offset,marker001_cam2_X,marker001_cam2_Y\n0,30,40\n0,NaN,NaN\n0,31.5,41.5\n"
    );
}

// ---------------------------------------------------------------------------------------
// 3D points (core/Trial.cpp, Trial::save3dPoints)
// ---------------------------------------------------------------------------------------

/// Defaults: `NaN` where the 3D status is 0 or below (LOST included), no frame column.
#[test]
fn points_3d_defaults() {
    let text = points3d::points_3d_file(&trial(), &[0, 1], &Default::default()).unwrap();
    assert_eq!(
        text,
        "marker001_X,marker001_Y,marker001_Z,Tip_X,Tip_Y,Tip_Z\n\
         1,2,3,1,2,7\n\
         1.5,-2.5,0.25,1.5,-2.5,3.25\n\
         NaN,NaN,NaN,NaN,NaN,NaN\n"
    );
}

/// Frame column and range (`:1565-1608`): `Frame,` in the header, `f + 1,` in each row, only
/// frames `start..=stop`; markers in the order given.
#[test]
fn points_3d_frame_range_and_order() {
    let opts = points3d::Points3dOptions {
        compat: CPP,
        frames: Some(FrameRange { start: 1, stop: 2 }),
        ..Default::default()
    };
    let text = points3d::points_3d_file(&trial(), &[1, 0], &opts).unwrap();
    assert_eq!(
        text,
        "Frame,Tip_X,Tip_Y,Tip_Z,marker001_X,marker001_Y,marker001_Z\n\
         2,1.5,-2.5,3.25,1.5,-2.5,0.25\n\
         3,NaN,NaN,NaN,NaN,NaN,NaN\n"
    );
    // stop < start writes no row.
    let opts = points3d::Points3dOptions {
        compat: CPP,
        frames: Some(FrameRange { start: 2, stop: 1 }),
        header: false,
        ..Default::default()
    };
    assert_eq!(points3d::points_3d_file(&trial(), &[0], &opts).unwrap(), "");
}

/// Filtered, one file per marker (`:1637-1642`, `:1661-1664`): `_12.5Hz` in names and
/// headers (`QString::number`). Runs of 12 frames or fewer are copied (ledger A8) with their
/// status, so the values are the input values here.
#[test]
fn points_3d_filtered_per_marker() {
    let opts = points3d::Points3dOptions {
        compat: CPP,
        one_file: false,
        filter_frequency: 12.5,
        frames: Some(FrameRange { start: 0, stop: 2 }),
        ..Default::default()
    };
    let files = points3d::points_3d(&trial(), &[0, 1], &opts)
        .unwrap()
        .into_folder();
    assert_eq!(
        files,
        vec![
            ExportFile::new(
                "Marker001__points3d_12.5Hz.csv".into(),
                "Frame,marker001_12.5Hz_X,marker001_12.5Hz_Y,marker001_12.5Hz_Z\n\
                 1,1,2,3\n2,1.5,-2.5,0.25\n3,NaN,NaN,NaN\n"
                    .into()
            ),
            ExportFile::new(
                "Marker002_Tip_points3d_12.5Hz.csv".into(),
                "Frame,Tip_12.5Hz_X,Tip_12.5Hz_Y,Tip_12.5Hz_Z\n\
                 1,1,2,7\n2,1.5,-2.5,3.25\n3,NaN,NaN,NaN\n"
                    .into()
            ),
        ]
    );
}

/// `filterMarker` refuses a cutoff at or above Nyquist, or no recording speed (`:1126-1128`);
/// the export then writes nothing. A frequency of 0 or below means unfiltered (`:1539`).
#[test]
fn points_3d_filter_refused() {
    let opts = points3d::Points3dOptions {
        compat: CPP,
        filter_frequency: 50.0,
        ..Default::default()
    };
    let err = points3d::points_3d(&trial(), &[0, 1], &opts).unwrap_err();
    assert!(matches!(err, ExportError::FilterRefused(e) if e.marker == 0));
    let mut t = trial();
    t.recording_speed = 0.0;
    let opts = points3d::Points3dOptions {
        compat: CPP,
        filter_frequency: 10.0,
        ..Default::default()
    };
    assert!(points3d::points_3d(&t, &[1], &opts).is_err());
    let opts = points3d::Points3dOptions {
        compat: CPP,
        filter_frequency: 0.0,
        ..Default::default()
    };
    assert_eq!(
        points3d::points_3d(&t, &[0, 1], &opts).unwrap().into_file(),
        points3d::points_3d_file(&trial(), &[0, 1], &Default::default()).unwrap()
    );
}

/// A filtered marker: frames outside every run are (-1000, -1000, -1000) and UNDEFINED.
#[test]
fn filter_marker_outside_runs() {
    let t = trial();
    let (points, status) = points3d::filter_marker(
        &t.markers[0],
        10.0,
        100.0,
        3,
        xma_vision::filter::CutoffCorrection::BothPasses,
    )
    .unwrap();
    assert_eq!(status[2], MarkerStatus::Undefined);
    assert_eq!(points[2], Point3::new(-1000.0, -1000.0, -1000.0));
    assert_eq!(status[1], MarkerStatus::Set);
}

// ---------------------------------------------------------------------------------------
// Rigid-body transformations (core/Trial.cpp, Trial::saveRigidBodyTransformations)
// ---------------------------------------------------------------------------------------

fn rb_header(name: &str, suffix: &str) -> String {
    [
        "R11", "R12", "R13", "01", "R21", "R22", "R23", "02", "R31", "R32", "R33", "03", "TX",
        "TY", "TZ", "1",
    ]
    .iter()
    .map(|n| format!("{name}_{n}{suffix}"))
    .collect::<Vec<_>>()
    .join(",")
}

const NAN16: &str = "NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN";

/// Unfiltered, one file: `RigidBody001` for an empty description; identity rotation with
/// t = (1, 2, 3) gives `trans[12..15] = Rᵀ(−t)` = (−1, −2, −3) (`RigidBody.cpp:2307-2309`); a
/// zero translation gives `1·−0 + 0·−0 + 0·−0` = `-0` in each position, which `%.17g` writes
/// as `-0`; a frame without a pose writes 16 × `NaN`.
#[test]
fn transformations_unfiltered() {
    let opts = rigid_bodies::TransformationsOptions {
        compat: CPP,
        filtered: false,
        ..Default::default()
    };
    let text = rigid_bodies::transformations(&trial(), &[0, 1], &opts)
        .unwrap()
        .into_file();
    let want = format!(
        "{},{}\n\
         1,0,0,0,0,1,0,0,0,0,1,0,-1,-2,-3,1,{NAN16}\n\
         {NAN16},{NAN16}\n\
         1,0,0,0,0,1,0,0,0,0,1,0,-0,-0,-0,1,{NAN16}\n",
        rb_header("RigidBody001", ""),
        rb_header("Jaw", "")
    );
    assert_eq!(text, want);
}

/// Filtered, one file per body, frame column, no header: names
/// `RigidBody%03d_<description>_transformationFiltered_<cutoff>Hz.csv` with the body's
/// override (40) or the trial's cutoff (10) (`:1060-1066`, ledger F11).
#[test]
fn transformations_filtered_per_body() {
    let opts = rigid_bodies::TransformationsOptions {
        compat: CPP,
        one_file: false,
        header: false,
        frames: Some(FrameRange { start: 0, stop: 1 }),
        ..Default::default()
    };
    let files = rigid_bodies::transformations(&trial(), &[0, 1], &opts)
        .unwrap()
        .into_folder();
    assert_eq!(
        files,
        vec![
            ExportFile::new(
                "RigidBody001__transformationFiltered_10Hz.csv".into(),
                format!("1,1,0,0,0,0,1,0,0,0,0,1,0,-0.5,-0,-0,1\n2,{NAN16}\n")
            ),
            ExportFile::new(
                "RigidBody002_Jaw_transformationFiltered_40Hz.csv".into(),
                format!("1,{NAN16}\n2,{NAN16}\n")
            ),
        ]
    );
}

/// Filtered headers carry `_<cutoff>Hz` after each name; the frame column adds `Frame,`.
/// Unfiltered per-body names end `_transformation.csv`.
#[test]
fn transformations_headers_and_names() {
    let opts = rigid_bodies::TransformationsOptions {
        compat: CPP,
        one_file: false,
        frames: Some(FrameRange { start: 2, stop: 2 }),
        ..Default::default()
    };
    let text = rigid_bodies::transformations_body_file(&trial(), 1, &opts);
    assert_eq!(
        text,
        format!("Frame,{}\n3,{NAN16}\n", rb_header("Jaw", "_40Hz"))
    );
    assert_eq!(
        rigid_bodies::transformations_body_file_name(&trial(), 0, false, CPP),
        "RigidBody001__transformation.csv"
    );
}

/// The GUI refuses the filtered export when the trial's cutoff is 0, even if every body
/// overrides it (`ui/MainWindow.cpp:1878`, ledger F11).
#[test]
fn filtered_export_needs_the_trial_cutoff() {
    let mut t = trial();
    assert!(rigid_bodies::can_export_filtered(&t, &[0], CPP));
    t.cutoff_frequency = 0.0;
    assert!(!rigid_bodies::can_export_filtered(&t, &[0], CPP));
    t.cutoff_frequency = 10.0;
    t.recording_speed = 0.0;
    assert!(!rigid_bodies::can_export_filtered(&t, &[0], CPP));
}

// ---------------------------------------------------------------------------------------
// Reprojection errors (core/Trial.cpp, Trial::saveReprojectionErrors)
// ---------------------------------------------------------------------------------------

/// A value where the 2D status is above 0 and the 3D status is non-zero, LOST (−10) included
/// (`:2031`); `NaN` otherwise.
#[test]
fn reprojection_errors() {
    let files = reprojection::reprojection_errors(&trial(), 2, CPP).unwrap();
    assert_eq!(
        files,
        vec![
            ExportFile::new(
                "Marker001__reprojectionError.csv".into(),
                "0.5,0.125\n0.25,NaN\nNaN,2\n".into()
            ),
            ExportFile::new(
                "Marker002_Tip_reprojectionError.csv".into(),
                "NaN,NaN\nNaN,NaN\nNaN,NaN\n".into()
            ),
        ]
    );
}

// ---------------------------------------------------------------------------------------
// Distances and precision info (core/Trial.cpp)
// ---------------------------------------------------------------------------------------

/// Distances between markers 1 and 2: 4 at frame 0, 3 at frame 1; mean 3.5, SD
/// `sqrt((0.5² + 0.5²) / 1)` = `sqrt(0.5)`. A marker against itself: 0 and 0. An empty body
/// description gives ` Mean` and ` SD` (`:1407`, `:1446`). Two empty lines before each
/// body, one between its tables (`:1404-1405`, `:1444`).
#[test]
fn marker_to_marker_distances() {
    let text = precision::marker_to_marker_distances(&trial(), 0, 3).unwrap();
    let sd = "0.70710678118654757";
    let want = format!(
        "Mean,Marker 1 ,Marker 2 Tip\n\
         Marker 1 ,0,3.5\n\
         Marker 2 Tip,3.5,0\n\
         \n\
         SD,Marker 1 ,Marker 2 Tip\n\
         Marker 1 ,0,{sd}\n\
         Marker 2 Tip,{sd},0\n\
         \n\n\
         \x20Mean,Marker 1 ,Marker 2 Tip\n\
         Marker 1 ,0,3.5\n\
         Marker 2 Tip,3.5,0\n\
         \n\
         \x20SD,Marker 1 ,Marker 2 Tip\n\
         Marker 1 ,0,{sd}\n\
         Marker 2 Tip,{sd},0\n\
         \n\n\
         Jaw Mean,Marker 2 Tip\n\
         Marker 2 Tip,0\n\
         \n\
         Jaw SD,Marker 2 Tip\n\
         Marker 2 Tip,0\n"
    );
    assert_eq!(text, want);
}

/// One common frame: SD 0, not NaN (`:1394`, unlike ledger A20); no common frame: mean 0.
#[test]
fn distance_statistics_edge_cases() {
    let t = trial();
    assert_eq!(
        precision::marker_distance_stats(&t.markers[0], &t.markers[1], 1, 3),
        (3.0, 0.0)
    );
    assert_eq!(
        precision::marker_distance_stats(&t.markers[0], &t.markers[1], 2, 3),
        (0.0, 0.0)
    );
    // The pooled body SD with a single distance is the plain sum of squared deviations, 0,
    // in XMALab 3; fixed (ledger A58), no degree of freedom gives 0 too.
    for compat in [CPP, CppCompat::FIXED] {
        assert_eq!(
            precision::body_marker_to_marker_sd(&t, &t.rigid_bodies[0], 1, 3, compat),
            (0.0, 1)
        );
    }
}

/// Precision info (`:1491-1512`): body frames tracked = every frame (ledger A24); pooled SD
/// `sqrt(0.5)`; mean 3D error over frames with a pose, (0.25 + 0.75) / 2 and 0.125 filtered;
/// a body without pairs or poses gives 0s. Marker reprojection error over frames with 3D
/// status TRACKED or better (frames 0 and 1), all cameras (ledger A25): 0.5, 0.125, 0.25 and
/// the 0 of camera 2 at frame 1 → mean 0.21875, SD `sqrt(0.13671875 / 3)`. Descriptions as
/// stored, even empty.
#[test]
fn precision_info() {
    let text = precision::precision_info(&trial(), 0, 3, CPP).unwrap();
    assert_eq!(
        text,
        "RigidBodies,nb tracked frames,marker to marker sd,error 3d Unfiltered,error 3D filtered\n\
         ,3,0.70710678118654757,0.5,0.125\n\
         Jaw,3,0,0,0\n\
         \n\n\
         Markers,nb tracked frames,reprojectionerror,sd\n\
         ,2,0.21875,0.21347814095749162\n\
         Tip,2,0,0\n"
    );
}

/// The files of a project save (ui/ProjectFileIO.cpp:193-196): only when the trial does not
/// require recomputation, over every frame.
#[test]
fn project_save_files() {
    let mut t = trial();
    let files = precision::project_save_files(&t, CPP).unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].name, "PrecisionInfo_Trial 1.txt");
    assert_eq!(
        files[0].contents,
        precision::precision_info(&t, 0, 3, CPP).unwrap()
    );
    assert_eq!(files[1].name, "MarkerDistances_Trial 1.txt");
    assert_eq!(
        files[1].contents,
        precision::marker_to_marker_distances(&t, 0, 3).unwrap()
    );
    t.requires_recomputation = true;
    assert!(precision::project_save_files(&t, CPP).unwrap().is_empty());
}

// ---------------------------------------------------------------------------------------
// MayaCam 2 (core/Camera.cpp, Camera::saveMayaCamVersion2)
// ---------------------------------------------------------------------------------------

fn calibrated_camera() -> Camera {
    let mut c = Camera::new("Camera 1".to_owned());
    c.width = 100;
    c.height = 50;
    c.camera_matrix = Matrix3::new(1000.0, 0.0, 50.0, 0.0, 1000.0, 25.0, 0.0, 0.0, 1.0);
    let mut im = CalibrationImage::new("/data/cube.tif".to_owned());
    im.calibrated = 1;
    im.translation = Vector3::new(1.0, 2.0, 3.0);
    c.calibration_images.push(im);
    c.calibration_images
        .push(CalibrationImage::new("/data/cube2.tif".to_owned()));
    c
}

/// With the lens model on, a blank line, `undistortion` and 8 coefficients follow
/// (`:548-556`); the rotation is `Rodrigues` of the rotation vector.
#[test]
fn maya_cam_with_distortion() {
    let mut c = calibrated_camera();
    c.has_model_distortion = true;
    c.distortion = [0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(
        mayacam::maya_cam_v2(&c, 0),
        "image size\n100,50\n\n\
         camera matrix\n1000,0,50\n0,1000,25\n0,0,1\n\n\
         rotation\n1,0,0\n0,1,0\n0,0,1\n\n\
         translation\n1\n2\n3\n\n\
         undistortion\n0.10000000000000001\n-0.20000000000000001\n0\n0\n0\n0\n0\n0\n"
    );
}

/// Files only for calibrated images, `<base>_MayaCam<camera + 1>.txt`; a frame of a
/// calibration video is named `Frame%06d` (core/CalibrationSequence.cpp:343).
#[test]
fn maya_cam_file_names() {
    let mut c2 = calibrated_camera();
    c2.calibration_images[0].filename = String::new();
    c2.calibration_sequence = Some(CalibrationSequence {
        filename: "/data/calib.avi".to_owned(),
        n_frames: 2,
        width: 100,
        height: 50,
    });
    let cams = vec![Arc::new(calibrated_camera()), Arc::new(c2)];
    let names: Vec<String> = mayacam::maya_cams_v2(&cams, None)
        .into_iter()
        .map(|f| f.name)
        .collect();
    assert_eq!(names, ["cube_MayaCam1.txt", "Frame000000_MayaCam2.txt"]);
    assert!(mayacam::maya_cams_v2(&cams, Some(1)).is_empty());
    assert!(mayacam::maya_cams_v2(&cams, Some(7)).is_empty());
    assert_eq!(mayacam::maya_cams_v2(&cams, Some(0)).len(), 2);
}

// ---------------------------------------------------------------------------------------
// Undistortion (core/UndistortionObject.cpp, UndistortionObject::exportData)
// ---------------------------------------------------------------------------------------

/// The LUT is OpenCV's CSV of each map (`%.8g`, `", "`, a line end per row) followed by
/// `std::endl`; the point files hold the inliers only.
#[test]
fn undistortion_files() {
    let table = LookupTable {
        width: 2,
        height: 2,
        map_x: vec![0.5, 1.25, -1.0, 3.0],
        map_y: vec![0.1, 2.0, -1.0, 4.5],
    };
    let mut cam = Camera::new("Camera 1".to_owned());
    cam.undistortion = Some(UndistortionGrid {
        image_filename: "grid.tif".to_owned(),
        computed: true,
        points_detected: Vec::new(),
        grid_distorted: vec![
            Point2::new(1.0, 2.0),
            Point2::new(3.5, 4.0),
            Point2::new(5.0, 6.0),
        ],
        grid_references: vec![
            Point2::new(1.5, 2.5),
            Point2::new(3.0, 4.0),
            Point2::new(0.1, 6.0),
        ],
        grid_inlier: vec![true, false, true],
        center: None,
    });
    let mut not_computed = cam.clone();
    not_computed.undistortion.as_mut().unwrap().computed = false;
    let cams = vec![Arc::new(cam), Arc::new(not_computed)];
    let files = undistortion::export_lut(&cams, &[Some(table.clone()), Some(table)]);
    assert_eq!(
        files,
        vec![
            ExportFile::new(
                "grid_LUT.csv".into(),
                "0.5, 1.25\n-1, 3\n\n0.1, 2\n-1, 4.5\n\n".into()
            ),
            ExportFile::new("grid_INPTS.csv".into(), "1,2\n5,6\n".into()),
            ExportFile::new(
                "grid_BSPTS.csv".into(),
                "1.5,2.5\n0.10000000000000001,6\n".into()
            ),
        ]
    );
}

// ---------------------------------------------------------------------------------------
// Events (core/EventData.cpp, EventData::saveData)
// ---------------------------------------------------------------------------------------

/// `0` or `1` per line; an empty name gives `.csv`.
#[test]
fn events() {
    let mut t = trial();
    for name in ["Heel strike", ""] {
        t.events.push(Event {
            name: name.to_owned(),
            color: Rgb::new(0, 0, 0),
            draw: true,
            data: vec![true, false, true],
        });
    }
    assert_eq!(
        events::events(&t, CPP),
        vec![
            ExportFile::new("Heel strike.csv".into(), "1\n0\n1\n".into()),
            ExportFile::new(".csv".into(), "1\n0\n1\n".into()),
        ]
    );
}

// ---------------------------------------------------------------------------------------
// Input the C++ would read out of bounds on: an error, never a panic
// ---------------------------------------------------------------------------------------

/// A frame range past the trial (the 3-frame trial: `stop` 3 is the fourth frame).
#[test]
fn frame_ranges_past_the_trial_are_errors() {
    let past = Some(FrameRange { start: 1, stop: 3 });
    let want = ExportError::FrameRange {
        first: 1,
        end: 4,
        n_frames: 3,
    };
    for one_file in [true, false] {
        let opts = points3d::Points3dOptions {
            compat: CPP,
            one_file,
            frames: past,
            ..Default::default()
        };
        assert_eq!(
            points3d::points_3d(&trial(), &[0, 1], &opts).unwrap_err(),
            want
        );
        let opts = rigid_bodies::TransformationsOptions {
            compat: CPP,
            one_file,
            frames: past,
            ..Default::default()
        };
        assert_eq!(
            rigid_bodies::transformations(&trial(), &[0, 1], &opts).unwrap_err(),
            want
        );
    }
    let opts = points3d::Points3dOptions {
        compat: CPP,
        frames: past,
        ..Default::default()
    };
    assert_eq!(
        points3d::points_3d_marker_file(&trial(), 0, &opts).unwrap_err(),
        want
    );
    // The last frame is fine; an empty range (stop < start) is accepted wherever it starts.
    let opts = points3d::Points3dOptions {
        compat: CPP,
        frames: Some(FrameRange { start: 2, stop: 2 }),
        ..Default::default()
    };
    assert!(points3d::points_3d(&trial(), &[0], &opts).is_ok());
    let opts = points3d::Points3dOptions {
        compat: CPP,
        frames: Some(FrameRange { start: 9, stop: 1 }),
        ..Default::default()
    };
    assert!(points3d::points_3d(&trial(), &[0], &opts).is_ok());

    let want = ExportError::FrameRange {
        first: 0,
        end: 4,
        n_frames: 3,
    };
    assert_eq!(
        precision::marker_to_marker_distances(&trial(), 0, 4).unwrap_err(),
        want
    );
    assert_eq!(
        precision::precision_info(&trial(), 0, 4, CPP).unwrap_err(),
        want
    );
    assert!(precision::precision_info(&trial(), 3, 1, CPP).is_ok());
}

#[test]
fn indices_that_do_not_exist_are_errors() {
    let opts = points3d::Points3dOptions::default();
    assert_eq!(
        points3d::points_3d(&trial(), &[0, 2], &opts).unwrap_err(),
        ExportError::NoSuchMarker(2)
    );
    let opts = rigid_bodies::TransformationsOptions {
        compat: CPP,
        one_file: false,
        ..Default::default()
    };
    assert_eq!(
        rigid_bodies::transformations(&trial(), &[5], &opts).unwrap_err(),
        ExportError::NoSuchBody(5)
    );
    let opts = points2d::Points2dOptions {
        compat: CPP,
        camera: Some(2),
        ..Default::default()
    };
    assert_eq!(
        points2d::points_2d(&trial(), &cameras(), &identity_models(), &opts).unwrap_err(),
        ExportError::NoSuchCamera(2)
    );
    let opts = points2d::Points2dOptions {
        compat: CPP,
        distorted: false,
        ..Default::default()
    };
    assert!(points2d::points_2d(&trial(), &cameras(), &[], &opts).is_err());
}

/// A marker shorter than the trial, or with fewer cameras than the project, and a body
/// naming a marker that does not exist.
#[test]
fn inconsistent_models_are_errors() {
    let mut t = trial();
    Arc::make_mut(&mut t.markers[1]).status3d.truncate(2);
    let opts = points3d::Points3dOptions::default();
    assert!(matches!(
        points3d::points_3d(&t, &[0, 1], &opts),
        Err(ExportError::Inconsistent(_))
    ));
    assert!(precision::precision_info(&t, 0, 3, CPP).is_err());
    assert!(precision::project_save_files(&t, CPP).is_err());

    let mut t = trial();
    Arc::make_mut(&mut t.markers[0]).views.pop();
    assert!(reprojection::reprojection_errors(&t, 2, CPP).is_err());
    assert!(points2d::points_2d(&t, &cameras(), &identity_models(), &Default::default()).is_err());

    let mut t = trial();
    Arc::make_mut(&mut t.rigid_bodies[0]).marker_indices.push(7);
    assert!(precision::marker_to_marker_distances(&t, 0, 3).is_err());

    let mut t = trial();
    Arc::make_mut(&mut t.rigid_bodies[1]).init(2);
    assert!(precision::precision_info(&t, 0, 3, CPP).is_err());
    assert!(precision::precision_info(&t, 0, 2, CPP).is_ok());
}

// ---------------------------------------------------------------------------------------
// Ledger A24, A25, F11, F25 (decided 14 November 2026): fixed by default
// ---------------------------------------------------------------------------------------

const FIXED: CppCompat = CppCompat::FIXED;

/// A24: frames tracked counts the frames with a pose (body 1: frames 0 and 2; Jaw: none).
/// A25: the reprojection error pools only the cameras in which the marker is set (frame 0:
/// both cameras, 0.5 and 0.125; frame 1: camera 1, 0.25; camera 2 is LOST there).
/// F25: empty descriptions get the export names.
#[test]
fn precision_info_fixed() {
    let text = precision::precision_info(&trial(), 0, 3, FIXED).unwrap();
    let mean: f64 = (0.5 + 0.125 + 0.25) / 3.0;
    let sd = (((0.5 - mean) * (0.5 - mean)
        + (0.125 - mean) * (0.125 - mean)
        + (0.25 - mean) * (0.25 - mean))
        / 2.0)
        .sqrt();
    assert_eq!(
        text,
        format!(
            "RigidBodies,nb tracked frames,marker to marker sd,error 3d Unfiltered,error 3D filtered\n\
             RigidBody001,2,0.70710678118654757,0.5,0.125\n\
             Jaw,0,0,0,0\n\
             \n\n\
             Markers,nb tracked frames,reprojectionerror,sd\n\
             marker001,2,{},{}\n\
             Tip,2,0,0\n",
            g(mean),
            g(sd)
        )
    );
    let t = trial();
    assert_eq!(precision::body_frames_tracked(&t.rigid_bodies[0], true), 3);
    assert_eq!(precision::body_frames_tracked(&t.rigid_bodies[0], false), 2);
}

/// F11: fixed, the filtered export checks each exported body's own cutoff.
#[test]
fn filtered_export_checks_each_bodys_cutoff() {
    let mut t = trial();
    t.cutoff_frequency = 0.0;
    // Jaw overrides with 40 Hz: exportable alone; body 1 uses the trial's 0.
    assert!(rigid_bodies::can_export_filtered(&t, &[1], FIXED));
    assert!(!rigid_bodies::can_export_filtered(&t, &[0, 1], FIXED));
    assert!(!rigid_bodies::can_export_filtered(&t, &[1], CPP));
    t.cutoff_frequency = 10.0;
    Arc::make_mut(&mut t.rigid_bodies[1]).cutoff_frequency = 0.0;
    assert!(!rigid_bodies::can_export_filtered(&t, &[1], FIXED));
    assert!(rigid_bodies::can_export_filtered(&t, &[1], CPP));
    t.recording_speed = 0.0;
    assert!(!rigid_bodies::can_export_filtered(&t, &[0], FIXED));
}

/// F25: `0`, not `-0`; default names in file names; ended lines without bodies.
#[test]
fn transformations_fixed_details() {
    let opts = rigid_bodies::TransformationsOptions {
        filtered: false,
        ..Default::default()
    };
    let text = rigid_bodies::transformations(&trial(), &[0], &opts)
        .unwrap()
        .into_file();
    assert!(
        text.ends_with("1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1\n"),
        "{text}"
    );
    assert!(!text.contains("-0,"));
    assert_eq!(
        rigid_bodies::transformations_body_file_name(&trial(), 0, false, FIXED),
        "RigidBody001_RigidBody001_transformation.csv"
    );
    let framed = rigid_bodies::TransformationsOptions {
        frames: Some(FrameRange { start: 0, stop: 1 }),
        ..opts
    };
    assert_eq!(
        rigid_bodies::transformations(&trial(), &[], &framed)
            .unwrap()
            .into_file(),
        "Frame\n1\n2\n"
    );
    let cpp = rigid_bodies::TransformationsOptions {
        compat: CPP,
        ..framed
    };
    assert_eq!(
        rigid_bodies::transformations(&trial(), &[], &cpp)
            .unwrap()
            .into_file(),
        "Frame,1,2,"
    );
    assert_eq!(
        rigid_bodies::transformations(&trial(), &[], &opts)
            .unwrap()
            .into_file(),
        ""
    );
}

/// F25: per-marker file names with an empty description, and the 3D and 2D exports
/// without markers.
#[test]
fn marker_exports_fixed_details() {
    let t = trial();
    assert_eq!(
        points3d::points_3d_marker_file_name(&t, 0, -1.0, FIXED),
        "Marker001_marker001_points3d.csv"
    );
    assert_eq!(
        points3d::points_3d_marker_file_name(&t, 0, -1.0, CPP),
        "Marker001__points3d.csv"
    );
    assert_eq!(
        points2d::points_2d_marker_file_name(&t, 0, FIXED),
        "Marker001_marker001_points2d.csv"
    );
    assert_eq!(
        reprojection::reprojection_error_file_name(&t, 0, FIXED),
        "Marker001_marker001_reprojectionError.csv"
    );
    let mut empty = trial();
    empty.markers.clear();
    empty.rigid_bodies.clear();
    let framed = points3d::Points3dOptions {
        frames: Some(FrameRange { start: 1, stop: 2 }),
        ..Default::default()
    };
    assert_eq!(
        points3d::points_3d_file(&empty, &[], &framed).unwrap(),
        "Frame\n2\n3\n"
    );
    let cpp = points3d::Points3dOptions {
        compat: CPP,
        ..framed
    };
    assert_eq!(
        points3d::points_3d_file(&empty, &[], &cpp).unwrap(),
        "Frame,2,3,"
    );
    let offsets = points2d::Points2dOptions {
        offset_columns: true,
        ..Default::default()
    };
    assert_eq!(
        points2d::points_2d_file(&empty, &cameras(), &identity_models(), &offsets),
        "cam1_offset,cam2_offset\n0,0\n0,0\n0,0\n"
    );
    let cpp = points2d::Points2dOptions {
        compat: CPP,
        ..offsets
    };
    assert_eq!(
        points2d::points_2d_file(&empty, &cameras(), &identity_models(), &cpp),
        "cam0_offsetcam1_offset0,0,0,0,0,0,"
    );
}

/// F25: every event gets its own file: an empty name becomes `Event%03d`, a repeated name
/// gets `_2`, `_3`.
#[test]
fn events_fixed_names() {
    let mut t = trial();
    for name in ["Heel strike", "", "Heel strike", "heel STRIKE"] {
        t.events.push(Event {
            name: name.to_owned(),
            color: Rgb::new(0, 0, 0),
            draw: true,
            data: vec![true, false, true],
        });
    }
    let names: Vec<String> = events::events(&t, FIXED)
        .into_iter()
        .map(|f| f.name)
        .collect();
    assert_eq!(
        names,
        [
            "Heel strike.csv",
            "Event002.csv",
            "Heel strike_2.csv",
            "heel STRIKE_3.csv"
        ]
    );
    let names: Vec<String> = events::events(&t, CPP)
        .into_iter()
        .map(|f| f.name)
        .collect();
    assert_eq!(
        names,
        [
            "Heel strike.csv",
            ".csv",
            "Heel strike.csv",
            "heel STRIKE.csv"
        ]
    );
}

// ---------------------------------------------------------------------------------------
// Marker-to-marker standard deviation (ledger A58)
// ---------------------------------------------------------------------------------------

/// Four markers on the x axis over 6 frames. Marker 0 at the origin in every frame;
/// marker 1 at x = 1, 2, 3 in frames 0..3; marker 2 at x = 10, 10, 10, 10, 15 in frames
/// 0..5; marker 3 only in frame 5. Pair distances: (0, 1) 1, 2, 3 (mean 2, sum of squared
/// deviations 2); (0, 2) 10, 10, 10, 10, 15 (mean 11, 20); (1, 2) 9, 8, 7 (mean 8, 2);
/// (0, 3) one distance (0); (1, 3) and (2, 3) none.
fn sd_trial(bodies: &[&[usize]]) -> Trial {
    let mut t = Trial::default_trial();
    t.is_default = false;
    t.n_frames = 6;
    t.end_frame = 6;
    let xs: [&[(usize, f64)]; 4] = [
        &[(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0)],
        &[(0, 1.0), (1, 2.0), (2, 3.0)],
        &[(0, 10.0), (1, 10.0), (2, 10.0), (3, 10.0), (4, 15.0)],
        &[(5, 4.0)],
    ];
    for x in xs {
        let mut m = Marker::new(1, 6, 8);
        for &(f, v) in x {
            m.points3d[f] = Point3::new(v, 0.0, 0.0);
            m.status3d[f] = MarkerStatus::Tracked;
        }
        t.markers.push(Arc::new(m));
    }
    for idx in bodies {
        let mut b = RigidBody::new(6);
        b.marker_indices = idx.to_vec();
        t.rigid_bodies.push(Arc::new(b));
    }
    t
}

fn rel(a: f64, b: f64) -> f64 {
    ((a - b) / b).abs()
}

/// Ledger A58 fixed: the body SD is sqrt(sum of squared deviations / sum of (n − 1)); XMALab
/// 3 divides by the total count minus 1. With one pair the two agree.
#[test]
fn body_marker_to_marker_sd_pooled() {
    let t = sd_trial(&[&[0, 1], &[0, 2], &[0, 1, 2], &[0, 1, 2, 3]]);
    let sd = |b: usize, c: CppCompat| {
        precision::body_marker_to_marker_sd(&t, &t.rigid_bodies[b], 0, 6, c)
    };
    // One pair: 2 / 2 and 20 / 4 both ways.
    assert_eq!(sd(0, CppCompat::FIXED), (1.0, 3));
    assert_eq!(sd(0, CPP), (1.0, 3));
    assert_eq!(sd(1, CppCompat::FIXED), (5f64.sqrt(), 5));
    assert_eq!(sd(1, CPP), (5f64.sqrt(), 5));
    // Three pairs (3, 5 and 3 distances): sum 24; dof 2 + 4 + 2 = 8; total count 11.
    assert_eq!(sd(2, CppCompat::FIXED), (3f64.sqrt(), 11));
    assert_eq!(sd(2, CPP), ((24.0f64 / 10.0).sqrt(), 11));
    // With marker 3: pair (0, 3) adds one distance and no degree of freedom; (1, 3) and
    // (2, 3) none.
    assert_eq!(sd(3, CppCompat::FIXED), (3f64.sqrt(), 12));
    assert_eq!(sd(3, CPP), ((24.0f64 / 11.0).sqrt(), 12));
    // The whole trial (`end` past the frames, as the Project Overview passes it).
    for c in [CPP, CppCompat::FIXED] {
        assert_eq!(
            precision::body_marker_to_marker_sd(&t, &t.rigid_bodies[3], 0, usize::MAX, c),
            sd(3, c)
        );
    }
}

/// Ledger A58 for the trial: fixed, sqrt(sum of the bodies' squared deviations / sum of
/// their dof); XMALab 3, sum(sd² (c − 1)) / (sum c − 1) (core/Trial.cpp:2476-2495).
#[test]
fn trial_marker_to_marker_sd_pooled() {
    let t = sd_trial(&[&[0, 1], &[0, 2]]);
    assert_eq!(
        precision::trial_marker_to_marker_sd(&t, CppCompat::FIXED),
        (22.0f64 / 6.0).sqrt()
    );
    let cpp = precision::trial_marker_to_marker_sd(&t, CPP);
    // 1² · 2 + sqrt(5)² · 4 = 22, over 3 + 5 − 1; sqrt(5)² is not exactly 5.
    assert!(rel(cpp, (22.0f64 / 7.0).sqrt()) < 1e-15, "{cpp}");
    // A pair in two bodies counts in both.
    let t = sd_trial(&[&[0, 1, 2], &[0, 1]]);
    assert_eq!(
        precision::trial_marker_to_marker_sd(&t, CppCompat::FIXED),
        (26.0f64 / 10.0).sqrt()
    );
    let cpp = precision::trial_marker_to_marker_sd(&t, CPP);
    assert!(
        rel(cpp, ((2.4 * 10.0 + 2.0) / 13.0f64).sqrt()) < 1e-15,
        "{cpp}"
    );
}

/// Ledger A58, no degree of freedom: the fixed values are 0. XMALab 3 gives 0 for a body
/// with one distance, and for the trial 0 / 0 = NaN when the bodies hold one distance in
/// all (the C++ arithmetic, kept in its mode), 0 when they hold none.
#[test]
fn marker_to_marker_sd_without_degrees_of_freedom() {
    // Body [0, 3]: one distance. Body [1, 3]: none. Body []: no pair.
    let t = sd_trial(&[&[0, 3], &[1, 3], &[]]);
    for (b, count) in [(0, 1), (1, 0), (2, 0)] {
        for c in [CPP, CppCompat::FIXED] {
            assert_eq!(
                precision::body_marker_to_marker_sd(&t, &t.rigid_bodies[b], 0, 6, c),
                (0.0, count)
            );
        }
    }
    assert_eq!(
        precision::trial_marker_to_marker_sd(&t, CppCompat::FIXED),
        0.0
    );
    assert!(precision::trial_marker_to_marker_sd(&t, CPP).is_nan());
    let t = sd_trial(&[&[1, 3], &[]]);
    assert_eq!(
        precision::trial_marker_to_marker_sd(&t, CppCompat::FIXED),
        0.0
    );
    assert_eq!(precision::trial_marker_to_marker_sd(&t, CPP), 0.0);
    let t = sd_trial(&[]);
    assert_eq!(
        precision::trial_marker_to_marker_sd(&t, CppCompat::FIXED),
        0.0
    );
    assert_eq!(precision::trial_marker_to_marker_sd(&t, CPP), 0.0);
}

/// Ledger A7: the filtered 3D export designs the filter at the cutoff corrected for the two
/// passes by default, and at the entered cutoff in the XMALab 3 mode: each mode's file is
/// the unfiltered export of the points filtered by hand with that design.
#[test]
fn filtered_3d_export_cutoff_correction() {
    use xma_vision::filter::{ButterworthLowPass, CutoffCorrection, design_cutoff, filter_run};
    let mut t = Trial::default_trial();
    t.is_default = false;
    t.n_frames = 40;
    t.end_frame = 40;
    t.recording_speed = 100.0;
    let mut m = Marker::new(1, 40, 8);
    for f in 0..40 {
        let x = f as f64 * 0.9;
        m.points3d[f] = Point3::new(x.sin(), (0.5 * x).cos(), 0.1 * x);
        m.status3d[f] = MarkerStatus::Tracked;
    }
    t.markers.push(Arc::new(m));
    let mut texts = Vec::new();
    for (compat, cutoff) in [
        (
            CppCompat::FIXED,
            design_cutoff(4, 10.0, 100.0, CutoffCorrection::BothPasses),
        ),
        (CPP, 10.0),
    ] {
        assert_eq!(
            cutoff_correction(compat) == CutoffCorrection::SinglePass,
            cutoff == 10.0
        );
        let opts = points3d::Points3dOptions {
            header: false,
            filter_frequency: 10.0,
            compat,
            ..Default::default()
        };
        let text = points3d::points_3d_file(&t, &[0], &opts).unwrap();
        let filter = ButterworthLowPass::new(4, cutoff, 100.0);
        let mut by_hand = t.clone();
        let m = Arc::make_mut(&mut by_hand.markers[0]);
        let comps: Vec<Vec<f64>> = (0..3)
            .map(|k| {
                filter_run(
                    &filter,
                    &m.points3d.iter().map(|p| p[k]).collect::<Vec<_>>(),
                )
            })
            .collect();
        for (f, p) in m.points3d.iter_mut().enumerate() {
            *p = Point3::new(comps[0][f], comps[1][f], comps[2][f]);
        }
        let plain = points3d::Points3dOptions {
            filter_frequency: -1.0,
            ..opts
        };
        assert_eq!(
            text,
            points3d::points_3d_file(&by_hand, &[0], &plain).unwrap()
        );
        texts.push(text);
    }
    assert_ne!(texts[0], texts[1]);
}
