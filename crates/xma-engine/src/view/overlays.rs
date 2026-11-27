//! What a camera view draws over a frame in the marker-tracking workspace, as plain
//! geometry in raw image coordinates (pixel centres at integers, as the C++ draws them).
//!
//! Ported from what `ui/GLCameraView.cpp`, `GLCameraView::paintGL` draws there:
//!
//! - `core/Trial.cpp`, `Trial::drawPoints`: the marker crosses (main view) or the
//!   crosshair of the active marker (detail view), the projected 3D point of the active
//!   marker and, optionally, of every marker, and the epipolar lines of the active marker
//!   from every other camera (`Marker::getEpipolarLine`, in [`xma_vision::epipolar`]);
//! - `core/Trial.cpp`, `Trial::drawRigidBodies` and `core/RigidBody.cpp`,
//!   `RigidBody::draw2D`: the rigid-body constellations (main view only);
//! - `GLCameraView::renderPointText(false)` with `Trial::getDrawTextData`: the marker
//!   numbers;
//! - the red warning when filtered data is to be drawn but the trial cannot be filtered.
//!
//! The rigid-body mesh overlay (`drawRigidBodiesMesh`, the distortion and blend shaders)
//! is in [`super::world`].
//!
//! Which items are drawn follows the settings the C++ reads ([`TrackingOptions`], named
//! after the setting keys). Colours are left to the caller ([`MarkColor`]), which maps
//! statuses to the colour settings.
//!
//! # Fixed C++ defects (parity ledger)
//!
//! - Marker numbers are written only for markers with a point in this camera at this frame
//!   (status above `UNDEFINED`). The C++ writes every marker's number at its stored position,
//!   which for an unset point is the placeholder (-2, -2), so all unset numbers pile up at
//!   the image corner.
//! - The detail view's crosshair is drawn only when the active marker has a point in this
//!   camera. The C++ draws it at the stored position, the placeholder (-2, -2) for an unset
//!   point.
//! - Shift + click selects only among markers with a point in this camera
//!   ([`marker_near`]); the C++ also considers unset points at (-2, -2).
//! - With "draw filtered data" on, a body is drawn from its filtered pose only where that
//!   pose is valid, and its virtual points use the same pose. The C++ draws a body whose
//!   raw pose is valid from the filtered arrays even where they hold no pose (zeros), and
//!   places virtual points with the raw pose.
//! - The epipolar-line fixes listed in [`xma_vision::epipolar`].

use xma_model::{MarkerStatus, Point2, Point3, Rgb, RigidBody, Trial, VirtualPointSource};
use xma_vision::camera::{
    RigidTransform, project_points_f32, projection_matrix, rodrigues_to_matrix,
};
use xma_vision::epipolar::{EpipolarCamera, EpipolarOptions, epipolar_line};

use crate::Document;

/// The display settings the overlays read, each named after its C++ setting.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct TrackingOptions {
    /// `TrialDrawHideAll`: nothing is drawn.
    pub hide_all: bool,
    /// `TrialDrawMarkers`: crosses, crosshair and projected points.
    pub markers: bool,
    /// `TrialDrawMarkerIds`: marker numbers.
    pub marker_ids: bool,
    /// `TrialDrawEpipolar`: epipolar lines of the active marker.
    pub epipolar: bool,
    /// `TrialDrawRigidBodyConstellation`: rigid-body constellations (main view).
    pub constellations: bool,
    /// `TrialDrawFiltered`: rigid bodies from their filtered poses.
    pub filtered: bool,
    /// `ShowColoredMarkerCross`: crosses in their status colour.
    pub colored_cross: bool,
    /// `ShowColoredMarkerIDs`: numbers in their status colour.
    pub colored_ids: bool,
    /// `DrawProjected2DpositionsForAllPoints`: projected points of every marker (main view).
    pub projected_all: bool,
    /// `Show3dPointDetailView`: the active marker's projected point in the detail view.
    pub detail_3d_point: bool,
    /// `ShowEpiLineDetailView`: epipolar lines in the detail view.
    pub detail_epipolar: bool,
    /// `ShowIDsInDetail`: marker numbers in the detail view.
    pub detail_ids: bool,
    /// `AdvancedCrosshairDetailView`: ticks and the marker-size box on the crosshair.
    pub detail_advanced_crosshair: bool,
    /// `EpipolarLinePrecision`: sample spacing of curved epipolar lines, in pixels.
    pub epipolar_precision: i32,
    /// Reproduce XMALab 3's epipolar-line defects ([`EpipolarOptions::xmalab3`]).
    pub xmalab3: bool,
}

impl Default for TrackingOptions {
    /// The C++ defaults (`core/Settings.cpp`).
    fn default() -> Self {
        Self {
            hide_all: false,
            markers: true,
            marker_ids: false,
            epipolar: true,
            constellations: true,
            filtered: false,
            colored_cross: false,
            colored_ids: false,
            projected_all: false,
            detail_3d_point: false,
            detail_epipolar: true,
            detail_ids: false,
            detail_advanced_crosshair: false,
            epipolar_precision: 5,
            xmalab3: false,
        }
    }
}

/// Which view is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewTarget {
    pub trial: usize,
    pub camera: usize,
    /// 0-based frame.
    pub frame: usize,
    /// The trial's active marker (`Trial::getActiveMarkerIdx`).
    pub active_marker: Option<usize>,
    /// A Detailed View panel view rather than a camera view.
    pub detail: bool,
}

/// How a cross or a number is coloured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkColor {
    /// The active marker: red (1, 0, 0).
    Active,
    /// Any other marker: green (0, 1, 0).
    Other,
    /// The colour setting of this status (`Marker::getStatusColor`).
    Status(MarkerStatus),
}

/// A marker cross: two segments of ±5 pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cross {
    pub marker: usize,
    pub at: Point2,
    pub color: MarkColor,
}

/// The crosshair of the active marker in a detail view: two red segments of ±12 pixels
/// and, with the advanced crosshair, tick marks every 2 pixels and the mean marker size
/// as a translucent square.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crosshair {
    pub at: Point2,
    pub advanced: bool,
    /// Half the side of the square (`Marker::getSize`, the mean size), when positive.
    pub size: Option<f64>,
}

/// A marker number: drawn red, green or in a status colour, 3 pixels right of the point,
/// with its baseline at the point (`renderText(x + 3, y, ...)`), 15-point font.
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    pub marker: usize,
    pub at: Point2,
    pub text: String,
    pub color: MarkColor,
}

/// An epipolar line (blue polyline) of the active marker as seen by `from_camera`.
#[derive(Clone, Debug, PartialEq)]
pub struct EpipolarPolyline {
    pub from_camera: usize,
    pub points: Vec<Point2>,
}

/// One rigid body's constellation, in the body's colour: solid segments from the
/// projected centre of its reference points to each reference point; and, when it has
/// virtual points known at the frame, stippled segments from the centre to each virtual
/// point's reference position and a ±5 pixel cross at each virtual point's own position.
#[derive(Clone, Debug, PartialEq)]
pub struct Constellation {
    pub body: usize,
    pub color: Rgb,
    pub lines: Vec<[Point2; 2]>,
    pub virtual_lines: Vec<[Point2; 2]>,
    pub virtual_crosses: Vec<Point2>,
}

/// Everything drawn over one view, in drawing order: crosses or crosshair, projected
/// points, epipolar lines, constellations, labels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrackingOverlay {
    pub crosses: Vec<Cross>,
    pub crosshair: Option<Crosshair>,
    /// Projected 3D points (cyan ×, ±5 pixels): the active marker's first, then, with
    /// [`TrackingOptions::projected_all`], every marker's (the active one again included,
    /// as the C++ draws it twice).
    pub projected: Vec<(usize, Point2)>,
    pub epipolar: Vec<EpipolarPolyline>,
    pub constellations: Vec<Constellation>,
    pub labels: Vec<Label>,
    /// "Rendering of filtered data is enabled, but framerate and cutoff are not set
    /// correctly" (red, 12-point, at the bottom centre).
    pub filtered_warning: bool,
}

/// The text of [`TrackingOverlay::filtered_warning`] (`GLCameraView.cpp:838`; the C++
/// misspells "framerate").
pub const FILTERED_WARNING: &str =
    "Rendering of filtered data is enabled, but framerate and cutoff are not set correctly";

fn has_data(status: Option<&MarkerStatus>) -> bool {
    status.is_some_and(|s| s.has_data())
}

/// The epipolar description of one camera for a trial.
struct EpiCam {
    projection: xma_vision::Matrix3x4,
    width: u32,
    height: u32,
    has_grid_object: bool,
}

fn epi_cam(doc: &Document, trial: &Trial, camera: usize) -> Option<EpiCam> {
    let cam = doc.project().cameras.get(camera)?;
    let image = cam
        .calibration_images
        .get(trial.reference_calibration_image)?;
    let pose = RigidTransform {
        rotation: image.rotation,
        translation: image.translation,
    };
    Some(EpiCam {
        projection: projection_matrix(&doc.camera(camera).model.intrinsics.camera_matrix, &pose),
        width: cam.width,
        height: cam.height,
        has_grid_object: cam.undistortion.is_some(),
    })
}

/// What the view of `target` draws.
///
/// Ported from `ui/GLCameraView.cpp`, `GLCameraView::paintGL` (the marker-tracking part)
/// with `Trial::drawPoints`, `Trial::drawRigidBodies` and `renderPointText(false)`; see the
/// module documentation. Out-of-range indices give an empty overlay.
#[must_use]
pub fn tracking_overlay(
    doc: &Document,
    target: ViewTarget,
    opts: &TrackingOptions,
) -> TrackingOverlay {
    let mut out = TrackingOverlay::default();
    let project = doc.project();
    let Some(trial) = project.trials.get(target.trial) else {
        return out;
    };
    let (cam, frame) = (target.camera, target.frame);
    if cam >= project.cameras.len() || frame >= trial.n_frames.max(1) {
        return out;
    }
    let detail = target.detail;
    // GLCameraView.cpp:835: the warning is drawn even with "hide all".
    out.filtered_warning =
        opts.filtered && (trial.cutoff_frequency <= 0.0 || trial.recording_speed <= 0.0);
    if opts.hide_all {
        return out;
    }
    let active = target.active_marker.filter(|&m| m < trial.markers.len());

    if opts.markers {
        if detail {
            if let Some(m) = active
                && let Some(p) = trial.markers[m].point2d(cam, frame)
            {
                let size = trial.markers[m].mean_size;
                out.crosshair = Some(Crosshair {
                    at: p,
                    advanced: opts.detail_advanced_crosshair,
                    size: (size > 0.0).then_some(size),
                });
            }
        } else {
            for (i, marker) in trial.markers.iter().enumerate() {
                let Some(p) = marker.point2d(cam, frame) else {
                    continue;
                };
                let color = if opts.colored_cross {
                    MarkColor::Status(marker.views[cam].status[frame])
                } else if Some(i) == active {
                    MarkColor::Active
                } else {
                    MarkColor::Other
                };
                out.crosses.push(Cross {
                    marker: i,
                    at: p,
                    color,
                });
            }
        }
        if let Some(m) = active
            && has_data(trial.markers[m].status3d.get(frame))
            && (!detail || opts.detail_3d_point)
            && let Some(p) = trial.markers[m]
                .views
                .get(cam)
                .and_then(|v| v.projected.get(frame))
        {
            out.projected.push((m, *p));
        }
        if !detail && opts.projected_all {
            for (i, marker) in trial.markers.iter().enumerate() {
                if has_data(marker.status3d.get(frame))
                    && let Some(p) = marker.views.get(cam).and_then(|v| v.projected.get(frame))
                {
                    out.projected.push((i, *p));
                }
            }
        }
    }

    if opts.epipolar
        && (!detail || opts.detail_epipolar)
        && project.calibration_type != xma_model::CalibrationType::None
        && let Some(m) = active
        && let Some(dest) = epi_cam(doc, trial, cam)
    {
        let marker = &trial.markers[m];
        let destination = EpipolarCamera {
            model: &doc.camera(cam).model,
            projection: &dest.projection,
            width: dest.width,
            height: dest.height,
            has_undistortion_grid: dest.has_grid_object,
        };
        let options = EpipolarOptions {
            precision: f64::from(opts.epipolar_precision),
            xmalab3: opts.xmalab3,
        };
        for other in 0..project.cameras.len() {
            if other == cam {
                continue;
            }
            let Some(p) = marker.point2d(other, frame) else {
                continue;
            };
            let Some(orig) = epi_cam(doc, trial, other) else {
                continue;
            };
            let origin = EpipolarCamera {
                model: &doc.camera(other).model,
                projection: &orig.projection,
                width: orig.width,
                height: orig.height,
                has_undistortion_grid: orig.has_grid_object,
            };
            out.epipolar.push(EpipolarPolyline {
                from_camera: other,
                points: epipolar_line(&origin, &destination, p, options),
            });
        }
    }

    if !detail && opts.constellations {
        for b in 0..trial.rigid_bodies.len() {
            if let Some(c) = constellation(doc, trial, b, cam, frame, opts.filtered) {
                out.constellations.push(c);
            }
        }
    }

    if (!detail || opts.detail_ids) && opts.marker_ids {
        for (i, marker) in trial.markers.iter().enumerate() {
            let Some(p) = marker.point2d(cam, frame) else {
                continue;
            };
            let color = if opts.colored_ids {
                MarkColor::Status(marker.views[cam].status[frame])
            } else if Some(i) == active {
                MarkColor::Active
            } else {
                MarkColor::Other
            };
            out.labels.push(Label {
                marker: i,
                at: Point2::new(p.x + 3.0, p.y),
                text: (i + 1).to_string(),
                color,
            });
        }
    }
    drop_non_finite(&mut out);
    out
}

fn finite(p: &Point2) -> bool {
    p.x.is_finite() && p.y.is_finite()
}

/// Leaves out every shape with a non-finite coordinate, as an unset point is left out: a
/// point with no undistorted or projected position (ledger A19, fixed mode) is not drawn.
fn drop_non_finite(out: &mut TrackingOverlay) {
    out.crosses.retain(|c| finite(&c.at));
    if out.crosshair.is_some_and(|c| !finite(&c.at)) {
        out.crosshair = None;
    }
    out.projected.retain(|(_, p)| finite(p));
    for e in &mut out.epipolar {
        e.points.retain(finite);
    }
    for c in &mut out.constellations {
        drop_non_finite_constellation(c);
    }
    out.labels.retain(|l| finite(&l.at));
}

fn drop_non_finite_constellation(c: &mut Constellation) {
    c.lines.retain(|[a, b]| finite(a) && finite(b));
    c.virtual_lines.retain(|[a, b]| finite(a) && finite(b));
    c.virtual_crosses.retain(finite);
}

/// A body's pose at a frame, raw or filtered.
fn body_pose(body: &RigidBody, frame: usize, filtered: bool) -> Option<RigidTransform> {
    let track = if filtered { &body.filtered } else { &body.raw };
    track.pose(frame).map(|p| RigidTransform {
        rotation: p.rotation,
        translation: p.translation,
    })
}

/// Body coordinates to world: `Rᵀ (x − t)` (`RigidBody::transformPoint`,
/// `RigidBody::projectToImage`).
fn to_world(pose: &RigidTransform, x: Point3) -> Point3 {
    let r = rodrigues_to_matrix(&pose.rotation);
    let v = r.transpose() * (x.coords - pose.translation);
    Point3::from(v)
}

/// One body's constellation in one camera.
///
/// Ported from `core/RigidBody.cpp`, `RigidBody::draw2D`, with `projectToImage` (the points
/// in world coordinates as `float`, `cv::projectPoints` with the camera's reference
/// calibration pose and no distortion coefficients, then `Camera::undistortPoint(pt,
/// false)`) and `transformPoint` for virtual points placed by another body. Here every pose
/// is the stored one: the C++ `transformPoint` would compute a missing pose of the other
/// body first, which the trial update has already done.
#[must_use]
pub fn constellation(
    doc: &Document,
    trial: &Trial,
    body: usize,
    camera: usize,
    frame: usize,
    filtered: bool,
) -> Option<Constellation> {
    let rb = trial.rigid_bodies.get(body)?;
    if !rb.visible || rb.references_set == 0 {
        return None;
    }
    let pose = body_pose(rb, frame, filtered)?;
    let cam = doc.project().cameras.get(camera)?;
    let image = cam
        .calibration_images
        .get(trial.reference_calibration_image)?;
    let model = &doc.camera(camera).model;
    let project = |world: &[Point3]| -> Vec<Point2> {
        if world.is_empty() {
            return Vec::new();
        }
        // The C++ builds `cv::Point3f`s: each coordinate rounded to `float`.
        let f32s: Vec<_> = world.iter().map(|p| p.cast::<f32>()).collect();
        project_points_f32(
            &f32s,
            &image.rotation,
            &image.translation,
            &model.intrinsics.camera_matrix,
            &[],
        )
        .into_iter()
        .map(|p| {
            model.undistort_point(
                Point2::new(f64::from(p.x), f64::from(p.y)),
                false,
                true,
                true,
            )
        })
        .collect()
    };
    // RigidBody::updateCenter: the mean of the reference points.
    let refs = &rb.reference_points;
    let center = if refs.is_empty() {
        Point3::origin()
    } else {
        let mut s = xma_model::Vector3::zeros();
        for p in refs {
            s += p.coords;
        }
        Point3::from(s / refs.len() as f64)
    };
    let mut world = vec![to_world(&pose, center)];
    world.extend(refs.iter().map(|p| to_world(&pose, *p)));
    let pts = project(&world);
    let lines = pts[1..].iter().map(|p| [pts[0], *p]).collect();
    let mut c = Constellation {
        body,
        color: rb.color,
        lines,
        virtual_lines: Vec::new(),
        virtual_crosses: Vec::new(),
    };
    if !rb.virtual_points.is_empty() {
        let placed: Vec<Point3> = rb
            .virtual_points
            .iter()
            .filter_map(|v| match &v.source {
                VirtualPointSource::OtherBody { body: other, point } => trial
                    .rigid_bodies
                    .get(*other)
                    .and_then(|o| body_pose(o, frame, filtered))
                    .map(|p| to_world(&p, *point)),
                VirtualPointSource::Coordinates(coords) => coords.get(frame).copied().flatten(),
            })
            .collect();
        if !placed.is_empty() {
            let mut w = vec![world[0]];
            w.extend(
                rb.virtual_points
                    .iter()
                    .map(|v| to_world(&pose, v.reference)),
            );
            let pv = project(&w);
            c.virtual_lines = pv[1..].iter().map(|p| [pv[0], *p]).collect();
            c.virtual_crosses = project(&placed);
        }
    }
    drop_non_finite_constellation(&mut c);
    Some(c)
}

/// The marker a Shift + click at `pos` selects: the nearest point of this camera at this
/// frame within 20 pixels, the first one on a tie.
///
/// Ported from `ui/WizardDigitizationFrame.cpp`, `selectDigitizationPoint`, considering
/// only markers with a point here (module documentation).
#[must_use]
pub fn marker_near(
    doc: &Document,
    trial: usize,
    camera: usize,
    frame: usize,
    pos: Point2,
) -> Option<usize> {
    let trial = doc.project().trials.get(trial)?;
    let mut best = 20.0;
    let mut idx = None;
    for (i, marker) in trial.markers.iter().enumerate() {
        let Some(p) = marker.point2d(camera, frame) else {
            continue;
        };
        let d = ((pos.x - p.x) * (pos.x - p.x) + (pos.y - p.y) * (pos.y - p.y)).sqrt();
        if d < best {
            best = d;
            idx = Some(i);
        }
    }
    idx
}

/// Where a detail view centres for `marker`: its point in this camera when set, else the
/// prediction from the two frames before (`getMarkerPrediction(.., true)`), else from the
/// two frames after; `None` leaves the view where it is.
///
/// Ported from `ui/GLCameraView.cpp`, `GLCameraView::centerViewToPoint`, and
/// `core/Marker.cpp`, `Marker::getMarkerPrediction`.
#[must_use]
pub fn detail_center(
    doc: &Document,
    trial: usize,
    camera: usize,
    frame: usize,
    marker: usize,
) -> Option<Point2> {
    let marker = doc.project().trials.get(trial)?.markers.get(marker)?;
    if let Some(p) = marker.point2d(camera, frame) {
        return Some(p);
    }
    marker_prediction(marker, camera, frame, true)
        .or_else(|| marker_prediction(marker, camera, frame, false))
}

/// The prediction the tracker uses (`crate::tracking::marker_prediction`), as a position:
/// linear extrapolation from the two previous frames in the direction of travel when both
/// are set, else the previous frame's point, else `None`.
///
/// Ported from `core/Marker.cpp`, `Marker::getMarkerPrediction`.
#[must_use]
pub fn marker_prediction(
    marker: &xma_model::Marker,
    camera: usize,
    frame: usize,
    forward: bool,
) -> Option<Point2> {
    crate::tracking::marker_prediction(marker, camera, frame, forward).1
}

#[cfg(test)]
mod tests {
    use super::*;
    use xma_model::Marker;

    #[test]
    fn prediction_matches_cpp_rules() {
        let mut m = Marker::new(1, 6, 8);
        let set = |m: &mut Marker, f: usize, x: f64| {
            m.views[0].points[f] = Point2::new(x, 2.0 * x);
            m.views[0].status[f] = MarkerStatus::Tracked;
        };
        set(&mut m, 1, 10.0);
        set(&mut m, 2, 12.0);
        // Forward from frames 2 and 1: 2*12 - 10 = 14.
        assert_eq!(
            marker_prediction(&m, 0, 3, true),
            Some(Point2::new(14.0, 28.0))
        );
        // Only frame 2 before frame 3... frame 1 before frame 2 with frame 0 unset: previous.
        assert_eq!(
            marker_prediction(&m, 0, 2, true),
            Some(Point2::new(10.0, 20.0))
        );
        // Backward from frames 1 and 2: 2*10 - 12 = 8.
        assert_eq!(
            marker_prediction(&m, 0, 0, false),
            Some(Point2::new(8.0, 16.0))
        );
        assert_eq!(marker_prediction(&m, 0, 5, true), None);
        assert_eq!(marker_prediction(&m, 0, 9, true), None);
    }

    fn pig() -> Option<(xma_oracle::Dataset, Document)> {
        let ds = xma_oracle::dataset("pig_clean")?;
        let doc = Document::open(&ds.project).expect("pig opens");
        Some((ds, doc))
    }

    fn target(camera: usize, frame: usize, active: Option<usize>, detail: bool) -> ViewTarget {
        ViewTarget {
            trial: 0,
            camera,
            frame,
            active_marker: active,
            detail,
        }
    }

    /// Crosses are the stored 2D points with status above 0, and the projected points with
    /// "all points" on are the C++ reprojections (`reprojection.csv`) at frames with a 3D
    /// point.
    #[test]
    fn pig_crosses_and_projected_points_match_the_dump() {
        let Some((ds, doc)) = pig() else { return };
        let opts = TrackingOptions {
            projected_all: true,
            ..TrackingOptions::default()
        };
        let points = ds.table("trial_0/points2d.csv");
        let reproj = ds.table("trial_0/reprojection.csv");
        let n_frames = doc.trial(0).n_frames;
        let mut crosses = xma_oracle::Comparison::new("crosses");
        let mut projected = xma_oracle::Comparison::new("projected");
        let mut n_cross = 0;
        let mut n_proj = 0;
        for camera in 0..2 {
            for frame in 0..n_frames {
                let o = tracking_overlay(&doc, target(camera, frame, Some(0), false), &opts);
                // Every stored point with status above 0 has its cross, nothing else does.
                let want: Vec<_> = points
                    .rows()
                    .filter(|r| {
                        r.usize("camera") == camera
                            && r.usize("frame") == frame
                            && r.i64("status") > 0
                    })
                    .collect();
                assert_eq!(o.crosses.len(), want.len(), "camera {camera} frame {frame}");
                for (cross, row) in o.crosses.iter().zip(&want) {
                    assert_eq!(cross.marker, row.usize("marker"));
                    crosses.add(
                        format_args!("m{} c{camera} f{frame} x", cross.marker),
                        cross.at.x,
                        row.f64("x"),
                    );
                    crosses.add(
                        format_args!("m{} c{camera} f{frame} y", cross.marker),
                        cross.at.y,
                        row.f64("y"),
                    );
                    let color = if cross.marker == 0 {
                        MarkColor::Active
                    } else {
                        MarkColor::Other
                    };
                    assert_eq!(cross.color, color);
                    n_cross += 1;
                }
                // The active marker's projection first, then every marker's.
                let all =
                    &o.projected[usize::from(doc.trial(0).markers[0].status3d[frame].has_data())..];
                for (m, p) in all {
                    let row = reproj
                        .find(&[
                            ("marker", *m as i64),
                            ("camera", camera as i64),
                            ("frame", frame as i64),
                        ])
                        .expect("row");
                    projected.add(
                        format_args!("m{m} c{camera} f{frame} x"),
                        p.x,
                        row.f64("projected_x"),
                    );
                    projected.add(
                        format_args!("m{m} c{camera} f{frame} y"),
                        p.y,
                        row.f64("projected_y"),
                    );
                    n_proj += 1;
                }
            }
        }
        assert!(
            n_cross > 1000 && n_proj > 1000,
            "{n_cross} crosses, {n_proj} projected"
        );
        println!("{crosses}\n{projected}");
        crosses.assert(xma_oracle::Tolerance::Abs(0.0));
        projected.assert(xma_oracle::Tolerance::Abs(1e-7));
    }

    /// The epipolar line of a marker's point in one camera passes, in the other camera's
    /// raw image, within the measured epipolar distance (plus the chord error of the
    /// sampled curve) of that marker's point there.
    #[test]
    fn pig_epipolar_lines_pass_by_the_corresponding_points() {
        let Some((_, doc)) = pig() else { return };
        let opts = TrackingOptions::default();
        let trial = doc.trial(0);
        let mut checked = 0;
        let mut worst: f64 = 0.0;
        for (m, marker) in trial.markers.iter().enumerate() {
            for frame in (0..trial.n_frames).step_by(7) {
                let (Some(p0), Some(p1)) = (marker.point2d(0, frame), marker.point2d(1, frame))
                else {
                    continue;
                };
                let o = tracking_overlay(&doc, target(1, frame, Some(m), false), &opts);
                assert_eq!(o.epipolar.len(), 1);
                let line = &o.epipolar[0].points;
                assert!(line.len() > 20, "{} samples", line.len());
                let d = line
                    .windows(2)
                    .map(|w| {
                        let (a, b) = (w[0], w[1]);
                        let t = ((p1 - a).dot(&(b - a)) / (b - a).norm_squared()).clamp(0.0, 1.0);
                        (a + (b - a) * t - p1).norm()
                    })
                    .fold(f64::INFINITY, f64::min);
                let e0 = epi_cam(&doc, trial, 0).unwrap();
                let e1 = epi_cam(&doc, trial, 1).unwrap();
                let c0 = EpipolarCamera {
                    model: &doc.camera(0).model,
                    projection: &e0.projection,
                    width: e0.width,
                    height: e0.height,
                    has_undistortion_grid: true,
                };
                let c1 = EpipolarCamera {
                    model: &doc.camera(1).model,
                    projection: &e1.projection,
                    ..c0
                };
                let ideal = xma_vision::epipolar::distance_to_epipolar_line(&c0, &c1, p0, p1);
                assert!(
                    d <= ideal * 1.2 + 0.25,
                    "marker {m} frame {frame}: {d} px from the line, {ideal} in the ideal image"
                );
                worst = worst.max(d);
                checked += 1;
            }
        }
        assert!(checked > 50, "{checked} checked");
        println!(
            "{checked} epipolar lines, largest distance to the corresponding point {worst:.3} px"
        );
    }

    /// A body's constellation ends at its reference points under its pose. Projected, their
    /// mean distance from the 2D points of the markers they stand for, over both cameras,
    /// is the body's 2D error that XMALab reports (`rb_errors.csv`, `error2d_mean`), which
    /// the C++ computes from the same projection.
    #[test]
    fn pig_constellations_end_at_the_fitted_positions() {
        let Some((ds, doc)) = pig() else { return };
        let trial = doc.trial(0);
        let errors = ds.table("trial_0/rb_errors.csv");
        let mut checked = 0;
        let mut worst: f64 = 0.0;
        for (b, rb) in trial.rigid_bodies.iter().enumerate() {
            for frame in 0..trial.n_frames {
                let (mut sum, mut n) = (0.0, 0);
                for camera in 0..2 {
                    let c = constellation(&doc, trial, b, camera, frame, false);
                    if !rb.visible || rb.raw.pose(frame).is_none() {
                        assert!(c.is_none());
                        continue;
                    }
                    let c = c.expect("constellation");
                    assert_eq!(c.lines.len(), rb.reference_points.len());
                    for (k, line) in c.lines.iter().enumerate() {
                        if let Some(p) = trial.markers[rb.marker_indices[k]].point2d(camera, frame)
                        {
                            sum += (line[1] - p).norm();
                            n += 1;
                        }
                    }
                }
                if n == 0 {
                    continue;
                }
                let want = errors
                    .find(&[("body", b as i64), ("frame", frame as i64)])
                    .expect("row")
                    .f64("error2d_mean");
                worst = worst.max((sum / f64::from(n) - want).abs());
                checked += 1;
            }
        }
        println!("{checked} body frames, largest difference from error2d_mean {worst:.3e} px");
        assert!(checked > 200, "{checked}");
        assert!(worst < 1e-9, "{worst}");
    }

    #[test]
    fn pig_options_switch_items_off_and_detail_view_draws_the_crosshair() {
        let Some((_, doc)) = pig() else { return };
        let trial = doc.trial(0);
        let frame = (0..trial.n_frames)
            .find(|&f| {
                trial.markers[0].point2d(0, f).is_some() && trial.markers[0].point2d(1, f).is_some()
            })
            .expect("a frame with marker 1 in both cameras");
        let full = tracking_overlay(
            &doc,
            target(0, frame, Some(0), false),
            &TrackingOptions {
                marker_ids: true,
                ..TrackingOptions::default()
            },
        );
        assert!(!full.crosses.is_empty() && !full.epipolar.is_empty() && !full.labels.is_empty());
        assert_eq!(full.labels[0].text, (full.labels[0].marker + 1).to_string());
        let hidden = tracking_overlay(
            &doc,
            target(0, frame, Some(0), false),
            &TrackingOptions {
                hide_all: true,
                ..TrackingOptions::default()
            },
        );
        assert_eq!(hidden, TrackingOverlay::default());
        let no_markers = tracking_overlay(
            &doc,
            target(0, frame, Some(0), false),
            &TrackingOptions {
                markers: false,
                ..TrackingOptions::default()
            },
        );
        assert!(no_markers.crosses.is_empty() && no_markers.projected.is_empty());
        assert!(!no_markers.epipolar.is_empty());
        let detail = tracking_overlay(
            &doc,
            target(0, frame, Some(0), true),
            &TrackingOptions::default(),
        );
        assert!(detail.crosses.is_empty() && detail.constellations.is_empty());
        assert_eq!(
            detail.crosshair.map(|c| c.at),
            trial.markers[0].point2d(0, frame)
        );
        // No active marker: no crosshair, no epipolar line.
        let none = tracking_overlay(
            &doc,
            target(0, frame, None, true),
            &TrackingOptions::default(),
        );
        assert!(none.crosshair.is_none() && none.epipolar.is_empty());
        // Selection by click: on a marker's point, and nothing far from every point.
        let p = trial.markers[0].point2d(0, frame).unwrap();
        let near = marker_near(&doc, 0, 0, frame, Point2::new(p.x + 1.0, p.y - 1.0));
        assert!(near.is_some());
        assert_eq!(
            marker_near(&doc, 0, 0, frame, Point2::new(-500.0, -500.0)),
            None
        );
        // Unset points (stored at -2, -2) are not selectable.
        assert_eq!(
            marker_near(&doc, 0, 0, frame, Point2::new(-2.0, -2.0)),
            None
        );
        assert_eq!(detail_center(&doc, 0, 0, frame, 0), Some(p));
    }
}
