//! The calibration state of a document as the C++ calibration reference writes it at a
//! `Dump` (`oracle/FORMAT.md`, "Calibration state"): the project-level tables and
//! `calib_state.json`, the `undistortion/` tables and the `calibration/` tables, written from
//! the engine's state with the C++ driver's formats (`%.17g`, `nan`, names with commas
//! replaced). The state dump's own tables (`state.json`, `trials.csv`, `trial_state.csv`,
//! `scheduler.csv`, `trial_<t>/`) are compared by [`super::edits::compare_state`] and are
//! not written here.
//!
//! `undistortion/lwm.csv`, `maps.csv` and `transform_samples.csv` (the grid model) are
//! written when the dump asks for the model (a `Dump` without `model=false`), for each
//! camera whose grid is computed.
//!
//! Ported from oracle/xmalab-cpp/src/ui/OracleDumpDriver.cpp (`writeTopLevel`,
//! `writeLoadedTopLevel`, `writeUndistortion`, `writeCalibration`) and
//! oracle/xmalab-cpp/src/ui/CalibOracleDriver.cpp (`writeDump`, `writeExtras`).

use std::fmt::Write as _;

use xma_engine::Document;
use xma_engine::view::calibration::{calibration_image, grid_lookup_table, undistortion_grid};
use xma_io::cfmt::format_g17;
use xma_model::{CalibrationObject, CalibrationType, Point2};
use xma_vision::undistortion::{LwmTransform, transform_point};

/// `%.17g`, or `nan` for any NaN (the driver's `fmtd`).
fn d(v: f64) -> String {
    if v.is_nan() {
        "nan".to_owned()
    } else {
        format_g17(v)
    }
}

/// Text as the driver writes it: commas become `;`, line breaks spaces.
fn s(t: &str) -> String {
    t.replace(',', ";").replace(['\n', '\r'], " ")
}

fn b(v: bool) -> &'static str {
    if v { "1" } else { "0" }
}

/// A CSV table being written: header, then rows of fields.
struct Csv(String);

impl Csv {
    fn new(header: &str) -> Self {
        Csv(format!("{header}\n"))
    }
    fn row(&mut self, fields: &[String]) {
        self.0.push_str(&fields.join(","));
        self.0.push('\n');
    }
}

macro_rules! row {
    ($csv:expr, $($f:expr),* $(,)?) => {
        $csv.row(&[$($f.to_string()),*])
    };
}

/// The calibration image's file name as `CalibrationImage::getFilename` gives it: the image
/// file's name, `Frame%06d` for a frame of a calibration video.
fn image_filename(cam: &xma_model::Camera, i: usize) -> String {
    if cam.calibration_sequence.is_some() {
        format!("Frame{i:06}")
    } else {
        cam.calibration_images[i].filename.clone()
    }
}

/// The sample points of `writeUndistortion`: every 32 pixels, and the same points plus
/// (0.37, 0.61), sorted by x then y.
fn sample_points(w: u32, h: u32) -> Vec<Point2> {
    let mut pts = Vec::new();
    for y in (0..h).step_by(32) {
        for x in (0..w).step_by(32) {
            pts.push(Point2::new(f64::from(x), f64::from(y)));
            pts.push(Point2::new(f64::from(x) + 0.37, f64::from(y) + 0.61));
        }
    }
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    pts
}

/// The grid model's tables of `writeExtras` (`m_dumpModel`): the two LWM fits, the lookup
/// maps at every 8th pixel, and the transform at every 16th pixel and the same plus
/// (0.37, 0.61).
fn grid_model_tables(doc: &Document, out: &mut Vec<(String, String)>) {
    let mut lwm = Csv::new(
        "camera,direction,index,point_x,point_y,radius,a_0,a_1,a_2,a_3,a_4,a_5,b_0,b_1,b_2,b_3,b_4,b_5",
    );
    let mut maps = Csv::new("camera,x,y,map_x,map_y");
    let mut ts = Csv::new(
        "camera,x,y,undistorted_x,undistorted_y,undistorted_norefine_x,undistorted_norefine_y,redistorted_x,redistorted_y",
    );
    for (c, cam) in doc.project().cameras.iter().enumerate() {
        if !cam.undistortion.as_ref().is_some_and(|g| g.computed) {
            continue;
        }
        let model = doc.camera(c).model.grid.as_ref();
        if let Some(m) = model {
            for (name, t) in [("distort", &m.distort), ("undistort", &m.undistort)] {
                let t: &LwmTransform = t;
                for k in 0..t.control_points.len() {
                    let p = t.control_points[k];
                    let mut f = vec![
                        c.to_string(),
                        name.to_owned(),
                        k.to_string(),
                        d(p.x),
                        d(p.y),
                        d(t.radii[k]),
                    ];
                    f.extend(t.a[k].iter().map(|v| d(*v)));
                    f.extend(t.b[k].iter().map(|v| d(*v)));
                    lwm.row(&f);
                }
            }
        }
        if let Some(l) = grid_lookup_table(doc, c) {
            for y in (0..l.height).step_by(8) {
                for x in (0..l.width).step_by(8) {
                    let i = y * l.width + x;
                    row!(
                        maps,
                        c,
                        x,
                        y,
                        d(f64::from(l.map_x[i])),
                        d(f64::from(l.map_y[i]))
                    );
                }
            }
        }
        for y in (0..cam.height).step_by(16) {
            for x in (0..cam.width).step_by(16) {
                for off in 0..2 {
                    let p = if off == 1 {
                        Point2::new(f64::from(x) + 0.37, f64::from(y) + 0.61)
                    } else {
                        Point2::new(f64::from(x), f64::from(y))
                    };
                    let (u, un, r) = match model {
                        Some(m) => {
                            let u = transform_point(m, p, true, true);
                            (
                                u,
                                transform_point(m, p, true, false),
                                transform_point(m, u, false, true),
                            )
                        }
                        None => (p, p, p),
                    };
                    row!(
                        ts,
                        c,
                        d(p.x),
                        d(p.y),
                        d(u.x),
                        d(u.y),
                        d(un.x),
                        d(un.y),
                        d(r.x),
                        d(r.y)
                    );
                }
            }
        }
    }
    out.push(("undistortion/lwm.csv".to_owned(), lwm.0));
    out.push(("undistortion/maps.csv".to_owned(), maps.0));
    out.push(("undistortion/transform_samples.csv".to_owned(), ts.0));
}

/// Every table of the calibration state that the engine holds, as (path in the dump,
/// contents); with `model`, the grid model's tables too (a `Dump` without `model=false`).
#[must_use]
pub fn calibration_state_tables(doc: &Document, model: bool) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let p = doc.project();
    let cams = &p.cameras;
    let session = doc.session();

    // writeTopLevel.
    let mut t = Csv::new(
        "camera,name,width,height,is_light,is_calibrated,is_optimized,is_flipped,has_undistortion",
    );
    for (c, cam) in cams.iter().enumerate() {
        row!(
            t,
            c,
            s(&cam.name),
            cam.width,
            cam.height,
            b(cam.is_light_camera),
            b(cam.calibrated),
            b(cam.optimized),
            b(cam.flipped),
            b(cam.undistortion.is_some())
        );
    }
    out.push(("cameras.csv".to_owned(), t.0));
    let mut t = Csv::new("camera,row,col,value");
    for (c, cam) in cams.iter().enumerate() {
        for r in 0..3 {
            for k in 0..3 {
                row!(t, c, r, k, d(cam.camera_matrix[(r, k)]));
            }
        }
    }
    out.push(("camera_matrix.csv".to_owned(), t.0));
    let mut t = Csv::new("camera,index,value");
    for (c, cam) in cams.iter().enumerate() {
        for (k, v) in cam.distortion.iter().enumerate() {
            row!(t, c, k, d(*v));
        }
    }
    out.push(("distortion.csv".to_owned(), t.0));
    // writeDump's own calibration_images.csv: nan for an image not calibrated.
    let mut t = Csv::new("camera,image,is_calibrated,rx,ry,rz,tx,ty,tz");
    for (c, cam) in cams.iter().enumerate() {
        for (i, im) in cam.calibration_images.iter().enumerate() {
            let set = im.calibrated > 0;
            let v = |x: f64| d(if set { x } else { f64::NAN });
            row!(
                t,
                c,
                i,
                im.calibrated,
                v(im.rotation[0]),
                v(im.rotation[1]),
                v(im.rotation[2]),
                v(im.translation[0]),
                v(im.translation[1]),
                v(im.translation[2])
            );
        }
    }
    out.push(("calibration_images.csv".to_owned(), t.0));

    // writeLoadedTopLevel.
    let mut t = Csv::new(
        "camera,id,portal_id,has_model_distortion,calibration_scale,calibration_bias,digitization_scale,digitization_bias,detail_scale,detail_bias",
    );
    for (c, cam) in cams.iter().enumerate() {
        let a = &cam.image_adjustments;
        row!(
            t,
            c,
            c,
            cam.portal_id,
            b(cam.has_model_distortion),
            d(a[0].scale),
            d(a[0].bias),
            d(a[1].scale),
            d(a[1].bias),
            d(a[2].scale),
            d(a[2].bias)
        );
    }
    out.push(("camera_details.csv".to_owned(), t.0));
    let mut t = Csv::new("trial,is_copy_from_default,n_markers,n_rigid_bodies,n_events");
    for (i, tr) in p.trials.iter().enumerate() {
        row!(
            t,
            i,
            b(tr.is_copy_from_default),
            tr.markers.len(),
            tr.rigid_bodies.len(),
            tr.events.len()
        );
    }
    out.push(("trial_details.csv".to_owned(), t.0));
    let mut obj = Csv::new(
        "initialised,checkerboard,planar,white_blobs,n_points,n_names,n_references,horizontal_squares,vertical_squares,square_size,frame_specifications_file,references_file",
    );
    let mut pts = Csv::new("index,x,y,z,name");
    let mut refs = Csv::new("index,point_id,name");
    match &p.calibration_object {
        None => row!(obj, 0, 0, 0, 0, 0, 0, 0, 0, 0, "nan", "", ""),
        Some(CalibrationObject::Checkerboard {
            horizontal_squares: h,
            vertical_squares: v,
            square_size,
            ..
        }) => {
            let board = p.calibration_object.as_ref().expect("matched");
            let n = board.n_points();
            row!(obj, 1, 1, 1, 0, n, n, 0, h, v, d(*square_size), "", "");
            for (i, (q, name)) in board.points().iter().zip(board.point_names()).enumerate() {
                row!(pts, i, d(q.x), d(q.y), d(q.z), s(&name));
            }
        }
        Some(CalibrationObject::Frame(f)) => {
            row!(
                obj,
                1,
                0,
                b(xma_io::calibration_object::is_planar(&f.points)),
                b(f.white_blobs),
                f.points.len(),
                f.point_names.len(),
                f.reference_ids.len(),
                0,
                0,
                "nan",
                s(&f.frame_specification_filename),
                s(&f.references_filename)
            );
            for i in 0..f.points.len().max(f.point_names.len()) {
                let q = f.points.get(i);
                row!(
                    pts,
                    i,
                    d(q.map_or(f64::NAN, |q| q.x)),
                    d(q.map_or(f64::NAN, |q| q.y)),
                    d(q.map_or(f64::NAN, |q| q.z)),
                    s(f.point_names.get(i).map_or("", String::as_str))
                );
            }
            for (i, id) in f.reference_ids.iter().enumerate() {
                row!(
                    refs,
                    i,
                    id,
                    s(f.reference_names.get(i).map_or("", String::as_str))
                );
            }
        }
    }
    out.push(("calibration_object.csv".to_owned(), obj.0));
    out.push(("calibration_object_points.csv".to_owned(), pts.0));
    out.push(("calibration_object_references.csv".to_owned(), refs.0));

    // writeUndistortion.
    let mut gp = Csv::new("camera,index,detected_x,detected_y,reference_x,reference_y,inlier");
    let mut samples =
        Csv::new("camera,x,y,undistorted_x,undistorted_y,redistorted_x,redistorted_y");
    let mut cs = Csv::new("camera,x,y,out_x,out_y");
    for (c, cam) in cams.iter().enumerate() {
        let model = &doc.camera(c).model;
        if let Some(g) = &cam.undistortion {
            let n = g
                .grid_distorted
                .len()
                .max(g.grid_references.len())
                .max(g.grid_inlier.len());
            if g.grid_distorted.len() != g.grid_references.len()
                || g.grid_distorted.len() != g.grid_inlier.len()
            {
                warnings.push(format!(
                    "camera {c}: undistortion grid vectors differ in length ({}, {}, {})",
                    g.grid_distorted.len(),
                    g.grid_references.len(),
                    g.grid_inlier.len()
                ));
            }
            for i in 0..n {
                let dp = g.grid_distorted.get(i);
                let rp = g.grid_references.get(i);
                row!(
                    gp,
                    c,
                    i,
                    d(dp.map_or(f64::NAN, |q| q.x)),
                    d(dp.map_or(f64::NAN, |q| q.y)),
                    d(rp.map_or(f64::NAN, |q| q.x)),
                    d(rp.map_or(f64::NAN, |q| q.y)),
                    g.grid_inlier.get(i).map_or(-1, |&v| i32::from(v))
                );
            }
            if g.computed && model.grid.is_some() {
                for q in sample_points(cam.width, cam.height) {
                    let u = model.undistort_point(q, true, false, true);
                    let r = model.undistort_point(u, false, false, true);
                    row!(samples, c, d(q.x), d(q.y), d(u.x), d(u.y), d(r.x), d(r.y));
                }
            } else {
                warnings.push(format!(
                    "camera {c}: undistortion grid not computed, no samples written"
                ));
            }
        }
        for q in sample_points(cam.width, cam.height) {
            let o = model.undistort_point(q, true, true, true);
            row!(cs, c, d(q.x), d(q.y), d(o.x), d(o.y));
        }
    }
    out.push(("undistortion/grid_points.csv".to_owned(), gp.0));
    out.push(("undistortion/samples.csv".to_owned(), samples.0));
    out.push(("undistortion/camera_undistort_samples.csv".to_owned(), cs.0));

    // writeCalibration.
    let mut cube = Csv::new("camera,image,index,detected_x,detected_y,matched_id,inlier");
    let mut all = Csv::new("camera,image,index,x,y");
    for (c, cam) in cams.iter().enumerate() {
        for (i, im) in cam.calibration_images.iter().enumerate() {
            for (k, q) in im.detected.iter().enumerate() {
                row!(
                    cube,
                    c,
                    i,
                    k,
                    d(q.x),
                    d(q.y),
                    k,
                    im.inlier.get(k).copied().unwrap_or(-1)
                );
            }
            for (k, q) in im.detected_all.iter().enumerate() {
                row!(all, c, i, k, d(q.x), d(q.y));
            }
        }
    }
    out.push(("calibration/cube_points.csv".to_owned(), cube.0));
    out.push(("calibration/cube_points_all.csv".to_owned(), all.0));

    // writeExtras.
    let mut gs = Csv::new(
        "camera,has_undistortion,computed,center_set,center_x,center_y,recalibration_required,width,height,n_detected,n_grid,n_inlier",
    );
    let mut det = Csv::new("camera,index,x,y");
    let mut gpu = Csv::new("camera,index,undistorted_x,undistorted_y,error");
    for (c, cam) in cams.iter().enumerate() {
        let Some(g) = &cam.undistortion else {
            row!(gs, c, 0, 0, 0, "nan", "nan", 0, 0, 0, 0, 0, 0);
            continue;
        };
        let centre = g.center;
        row!(
            gs,
            c,
            1,
            b(g.computed),
            b(centre.is_some()),
            d(centre.map_or(f64::NAN, |q| q.x)),
            d(centre.map_or(f64::NAN, |q| q.y)),
            session.grid_recalibration.get(c).copied().unwrap_or(0),
            cam.width,
            cam.height,
            g.points_detected.len(),
            g.grid_distorted.len(),
            g.grid_inlier.iter().filter(|v| **v).count()
        );
        for (k, q) in g.points_detected.iter().enumerate() {
            row!(det, c, k, d(q.x), d(q.y));
        }
        if let Some(v) = undistortion_grid(doc, c) {
            let n = v.grid_undistorted.len().max(v.error.len());
            for k in 0..n {
                let u = v.grid_undistorted.get(k);
                row!(
                    gpu,
                    c,
                    k,
                    d(u.map_or(f64::NAN, |q| q.x)),
                    d(u.map_or(f64::NAN, |q| q.y)),
                    d(v.error.get(k).copied().unwrap_or(f64::NAN))
                );
            }
        }
    }
    out.push(("undistortion/grid_state.csv".to_owned(), gs.0));
    out.push(("undistortion/detected_points.csv".to_owned(), det.0));
    out.push(("undistortion/grid_points_undistorted.csv".to_owned(), gpu.0));
    if model {
        grid_model_tables(doc, &mut out);
    }

    let mut camst = Csv::new(
        "camera,is_calibrated,is_optimized,recalibration_required,has_model_distortion,n_calibration_images,calibration_error,calibration_n_inlier",
    );
    let mut imst = Csv::new(
        "camera,image,is_calibrated,n_detected_all,n_points,n_inlier,calibration_error,filename",
    );
    let mut impts = Csv::new(
        "camera,image,index,inlier,detected_x,detected_y,detected_ux,detected_uy,projected_x,projected_y,projected_ux,projected_uy,error,error_undistorted",
    );
    for (c, cam) in cams.iter().enumerate() {
        // CalibrationImage::getCalibrationNbInlier and getCalibrationError, per image.
        let mut per_image = Vec::new();
        for i in 0..cam.calibration_images.len() {
            let v = calibration_image(doc, c, i).unwrap_or_default();
            let (mut n_in, mut sum) = (0usize, 0.0);
            if v.calibrated == 1 {
                for (k, e) in v.error.iter().enumerate() {
                    if v.inlier.get(k).is_some_and(|&x| x > 0) {
                        n_in += 1;
                        sum += e;
                    }
                }
            }
            let n_inlier = if v.calibrated == 1 {
                v.inlier.iter().filter(|&&x| x > 0).count()
            } else {
                0
            };
            let err = if n_in > 0 { sum / n_in as f64 } else { 0.0 };
            per_image.push((v, n_inlier, err));
        }
        // Camera::getCalibrationError, getCalibrationNbInlier.
        let n_all: usize = per_image.iter().map(|(_, n, _)| n).sum();
        let mut mean = 0.0;
        for (_, n, e) in &per_image {
            mean += e * *n as f64;
        }
        if n_all > 0 {
            mean /= n_all as f64;
        }
        row!(
            camst,
            c,
            b(cam.calibrated),
            b(cam.optimized),
            session.camera_recalibration.get(c).copied().unwrap_or(0),
            b(cam.has_model_distortion),
            cam.calibration_images.len(),
            d(mean),
            n_all
        );
        for (i, (v, n_inlier, err)) in per_image.iter().enumerate() {
            row!(
                imst,
                c,
                i,
                v.calibrated,
                v.detected_all.len(),
                v.detected.len(),
                n_inlier,
                d(*err),
                s(&image_filename(cam, i))
            );
            let n = v
                .detected
                .len()
                .max(v.detected_undistorted.len())
                .max(v.projected.len())
                .max(v.projected_undistorted.len())
                .max(v.inlier.len());
            let pt = |l: &[Point2], k: usize, y: bool| {
                d(l.get(k).map_or(f64::NAN, |q| if y { q.y } else { q.x }))
            };
            for k in 0..n {
                row!(
                    impts,
                    c,
                    i,
                    k,
                    v.inlier.get(k).copied().unwrap_or(-9),
                    pt(&v.detected, k, false),
                    pt(&v.detected, k, true),
                    pt(&v.detected_undistorted, k, false),
                    pt(&v.detected_undistorted, k, true),
                    pt(&v.projected, k, false),
                    pt(&v.projected, k, true),
                    pt(&v.projected_undistorted, k, false),
                    pt(&v.projected_undistorted, k, true),
                    d(v.error.get(k).copied().unwrap_or(f64::NAN)),
                    d(v.error_undistorted.get(k).copied().unwrap_or(f64::NAN))
                );
            }
        }
    }
    out.push(("calibration/camera_state.csv".to_owned(), camst.0));
    out.push(("calibration/image_state.csv".to_owned(), imst.0));
    out.push(("calibration/image_points.csv".to_owned(), impts.0));

    // calib_state.json.
    let mut j = String::new();
    let _ = writeln!(j, "{{");
    let _ = writeln!(j, "  \"format_version\": 5,");
    let _ = writeln!(j, "  \"kind\": \"calib_state\",");
    let ct = match p.calibration_type {
        CalibrationType::None => 0,
        CalibrationType::Internal => 1,
        CalibrationType::External => 2,
    };
    let _ = writeln!(j, "  \"calibration_type\": {ct},");
    let _ = writeln!(j, "  \"n_cameras\": {},", cams.len());
    let _ = writeln!(
        j,
        "  \"n_calibration_images\": {},",
        session.n_calibration_images
    );
    let _ = writeln!(j, "  \"workspace\": {},", session.workspace as i32);
    let _ = writeln!(
        j,
        "  \"undistortion_state\": {},",
        i32::from(session.undistorted)
    );
    let _ = writeln!(j, "  \"active_camera\": {},", session.active_camera);
    let _ = writeln!(
        j,
        "  \"active_frame_calibration\": {},",
        session.active_frame_calibration
    );
    // Project::camerasOptimized: true for no cameras.
    let optimized = cams.iter().all(|c| c.optimized);
    let _ = writeln!(j, "  \"cameras_optimized\": {optimized},");
    let _ = writeln!(j, "  \"warnings\": [");
    for (i, w) in warnings.iter().enumerate() {
        let comma = if i + 1 < warnings.len() { "," } else { "" };
        let _ = writeln!(
            j,
            "    \"{}\"{comma}",
            w.replace('\\', "\\\\").replace('"', "\\\"")
        );
    }
    let _ = writeln!(j, "  ]");
    let _ = writeln!(j, "}}");
    out.push(("calib_state.json".to_owned(), j));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_are_sorted_like_the_driver() {
        let p = sample_points(64, 64);
        assert_eq!(p.len(), 8);
        assert_eq!((p[0].x, p[0].y), (0.0, 0.0));
        assert_eq!((p[1].x, p[1].y), (0.0, 32.0));
        assert_eq!((p[2].x, p[2].y), (0.37, 0.61));
        assert_eq!(d(f64::NAN), "nan");
    }
}
