//! The numbers the Plot panel draws, for each of XMALab's six plot types.
//!
//! Ported from ui/PlotWindow.cpp (`plot2D`, `plot3D`, `plotDistance`,
//! `plotReprojectionError`, `plotRigidBody`, `plotRigidBodyError`, `drawStatus`,
//! `drawEvents`, `updateMarkers`, `isFrameAboveError`, `goToNextAboveError`,
//! `goToPrevAboveError`, `saveData`) and the getters they call in core/RigidBody.cpp
//! (`getRotationEulerAngle`, `getDummyCoordinates`).
//!
//! Every query reads a [`Trial`] (or the [`Document`] where camera models are needed) and
//! returns one value per frame of a [`FrameSpan`] (the trial's start to end frame, 0-based),
//! with NaN where the C++ plots nothing. Where the C++ plots a placeholder instead (0 for a
//! missing distance, -2 for a missing error), the caller substitutes it; the queries keep
//! "no value" distinct from a value.
//!
//! Which values are held in the model and which are computed here:
//!
//! | Plot | Held in the model | Computed here |
//! |---|---|---|
//! | 2D positions | `MarkerView::points`, `status` | — |
//! | 3D positions | `Marker::points3d`, `status3d` | — |
//! | Marker to marker distance | the 3D points; virtual points' sources | the distance, its mean and SD; a virtual point's world position (`getDummyCoordinates`, through `float`) |
//! | Reprojection error | `MarkerView::error` | the mean over the chosen cameras per frame, its mean and SD |
//! | Rigid-body transformation | `PoseTrack::poses` (Rodrigues vector, translation) | the Euler angles (`getRotationEulerAngle`) and their unwrapping |
//! | Rigid-body error | `PoseTrack::errors` (per-frame mean and SD, 2D and 3D) | the mean over frames and its SD |
//!
//! Two defects of `PlotWindow.cpp` are fixed here (the C++ behaviour is described in each
//! function): the unfiltered rigid-body error's SD sums over the filtered frames
//! ([`body_error`]), and the CSV export misreads frames when the trial does not start at
//! frame 1, turns frames without a pose into ±360 after a large angle, and writes blank lines
//! and stray spaces ([`export_csv`]).

use xma_io::cfmt::{format_g17, format_qt_fixed};
use xma_model::{
    Event, Interpolation, Marker, MarkerStatus, Point3, PoseTrack, RigidBody, Trial, Vector3,
    VirtualPointSource,
};
use xma_vision::camera::RigidTransform;
use xma_vision::camera::rodrigues_to_matrix;
use xma_vision::rigid::body_to_world;

use crate::Document;

/// The frames a plot covers: the trial's start to end frame, 0-based and inclusive
/// (`getStartFrame() - 1` to `getEndFrame() - 1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameSpan {
    /// First frame, 0-based.
    pub first: usize,
    /// Last frame, 0-based; at least `first`.
    pub last: usize,
}

impl FrameSpan {
    /// The span from the 1-based start and end frames the trial view holds.
    #[must_use]
    pub fn from_one_based(start_frame: usize, end_frame: usize) -> Self {
        let first = start_frame.saturating_sub(1);
        Self {
            first,
            last: end_frame.saturating_sub(1).max(first),
        }
    }

    /// Number of frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.last - self.first + 1
    }

    /// Never empty; for clippy's `len_without_is_empty`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The frames, in order.
    pub fn frames(&self) -> std::ops::RangeInclusive<usize> {
        self.first..=self.last
    }
}

/// Mean and sample standard deviation as the plot labels show them: the mean over the
/// counted values, the SD with `count - 1`; 0 and 0 when nothing counts, SD 0 for one value.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeanSd {
    pub mean: f64,
    pub sd: f64,
    /// Values counted.
    pub count: usize,
}

impl MeanSd {
    /// Mean and SD of the non-NaN values, summed in order.
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::plotDistance (`mean`, `sd`,
    /// `countVisible`); the same arithmetic in plotReprojectionError and plotRigidBodyError.
    #[must_use]
    pub fn of(values: &[f64]) -> Self {
        let mut mean = 0.0;
        let mut count = 0usize;
        for v in values.iter().filter(|v| !v.is_nan()) {
            mean += v;
            count += 1;
        }
        if count > 0 {
            mean /= count as f64;
        }
        let mut sd = 0.0;
        for v in values.iter().filter(|v| !v.is_nan()) {
            sd += xma_vision::cmath::pow(v - mean, 2.0);
        }
        if count > 1 {
            sd = (sd / (count - 1) as f64).sqrt();
        }
        Self { mean, sd, count }
    }
}

fn status_at(s: &[MarkerStatus], frame: usize) -> MarkerStatus {
    s.get(frame).copied().unwrap_or_default()
}

// ---------------------------------------------------------------------------------------
// Combo-box contents

/// `comboBoxCamera`: "All Cameras", then each camera's name.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::updateMarkers.
#[must_use]
pub fn camera_items(doc: &Document) -> Vec<String> {
    std::iter::once("All Cameras".to_owned())
        .chain(doc.project().cameras.iter().map(|c| c.name.clone()))
        .collect()
}

/// `comboBoxMarker1`: the markers' 1-based numbers.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::updateMarkers.
#[must_use]
pub fn marker_items(trial: &Trial) -> Vec<String> {
    (1..=trial.markers.len()).map(|i| i.to_string()).collect()
}

/// What the second marker of a distance plot is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistanceTarget {
    /// A marker of the trial.
    Marker(usize),
    /// Virtual point `point` of rigid body `body` (`VP<point+1>-RB<body+1>`).
    VirtualPoint { body: usize, point: usize },
}

/// `comboBoxMarker2`: the markers' numbers, then `VP<i>-RB<j>` for every virtual point of
/// every rigid body, with what each item means.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::updateMarkers (and the parsing of the item
/// text in plotDistance).
#[must_use]
pub fn marker2_items(trial: &Trial) -> Vec<(String, DistanceTarget)> {
    let mut items: Vec<(String, DistanceTarget)> = (0..trial.markers.len())
        .map(|i| ((i + 1).to_string(), DistanceTarget::Marker(i)))
        .collect();
    for (j, rb) in trial.rigid_bodies.iter().enumerate() {
        for i in 0..rb.virtual_points.len() {
            items.push((
                format!("VP{}-RB{}", i + 1, j + 1),
                DistanceTarget::VirtualPoint { body: j, point: i },
            ));
        }
    }
    items
}

/// `comboBoxRigidBody`: `RB<i> - <description>`.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::updateMarkers.
#[must_use]
pub fn rigid_body_items(trial: &Trial) -> Vec<String> {
    trial
        .rigid_bodies
        .iter()
        .enumerate()
        .map(|(i, rb)| format!("RB{} - {}", i + 1, rb.description))
        .collect()
}

// ---------------------------------------------------------------------------------------
// 2D and 3D positions

/// A marker's 2D position in one camera: x and y per frame, NaN where the status is not
/// above `UNDEFINED`.
#[derive(Clone, Debug, PartialEq)]
pub struct Positions2d {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// Ported from ui/PlotWindow.cpp, PlotWindow::plot2D (one camera's two graphs).
#[must_use]
pub fn positions_2d(trial: &Trial, marker: usize, camera: usize, span: FrameSpan) -> Positions2d {
    let view = trial.markers.get(marker).and_then(|m| m.views.get(camera));
    let mut out = Positions2d {
        x: Vec::with_capacity(span.len()),
        y: Vec::with_capacity(span.len()),
    };
    for f in span.frames() {
        match view {
            Some(v) if status_at(&v.status, f).has_data() => {
                out.x.push(v.points[f].x);
                out.y.push(v.points[f].y);
            }
            _ => {
                out.x.push(f64::NAN);
                out.y.push(f64::NAN);
            }
        }
    }
    out
}

/// A marker's 3D position: x, y and z per frame, NaN where the 3D status is not above
/// `UNDEFINED`.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::plot3D.
#[must_use]
pub fn positions_3d(trial: &Trial, marker: usize, span: FrameSpan) -> [Vec<f64>; 3] {
    let m = trial.markers.get(marker);
    let mut out: [Vec<f64>; 3] = Default::default();
    for f in span.frames() {
        let p = m.and_then(|m| m.point3d(f));
        for (k, o) in out.iter_mut().enumerate() {
            o.push(p.map_or(f64::NAN, |p| p[k]));
        }
    }
    out
}

// ---------------------------------------------------------------------------------------
// Marker to marker distance

/// The world position of a rigid body's virtual point at a frame, as the plot reads it:
/// the other body's unfiltered pose applied to the point (`transformPoint`), or the
/// imported coordinates; returned through `cv::Point3f`, so rounded to `f32`. `None` where
/// the C++ returns (0, 0, 0), which it treats as "no point".
///
/// The C++ computes the other body's pose on demand when it has none at the frame; the
/// viewer reads the stored pose, which the trial update has computed for every frame.
///
/// Ported from core/RigidBody.cpp, RigidBody::getDummyCoordinates, and the test
/// `pt_d != cv::Point3f(0, 0, 0)` in ui/PlotWindow.cpp, PlotWindow::plotDistance.
#[must_use]
pub fn virtual_point_position(
    trial: &Trial,
    body: usize,
    point: usize,
    frame: usize,
) -> Option<Point3> {
    let vp = trial.rigid_bodies.get(body)?.virtual_points.get(point)?;
    let world = match &vp.source {
        VirtualPointSource::OtherBody { body: other, point } => {
            let pose = trial.rigid_bodies.get(*other)?.raw.pose(frame)?;
            body_to_world(
                &RigidTransform {
                    rotation: pose.rotation,
                    translation: pose.translation,
                },
                *point,
            )
        }
        VirtualPointSource::Coordinates(c) => c.get(frame).copied().flatten()?,
    };
    let f = |v: f64| f64::from(v as f32);
    let p = Point3::new(f(world.x), f(world.y), f(world.z));
    (p != Point3::new(0.0, 0.0, 0.0)).then_some(p)
}

/// The distance between a marker and a second marker or virtual point.
#[derive(Clone, Debug, PartialEq)]
pub struct Distance {
    /// Per frame; NaN where either point is missing (the C++ plots 0 there).
    pub values: Vec<f64>,
    /// Mean and SD over the frames with both points ("Mean intermarker distance").
    pub stats: MeanSd,
}

fn dist(a: Point3, b: Point3) -> f64 {
    let (dx, dy, dz) = (a.x - b.x, a.y - b.y, a.z - b.z);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Ported from ui/PlotWindow.cpp, PlotWindow::plotDistance.
#[must_use]
pub fn distance(trial: &Trial, marker: usize, target: DistanceTarget, span: FrameSpan) -> Distance {
    let m = trial.markers.get(marker);
    let values: Vec<f64> = span
        .frames()
        .map(|f| {
            let a = m.and_then(|m| m.point3d(f));
            let b = match target {
                DistanceTarget::Marker(j) => trial.markers.get(j).and_then(|m| m.point3d(f)),
                DistanceTarget::VirtualPoint { body, point } => {
                    virtual_point_position(trial, body, point, f)
                }
            };
            match (a, b) {
                (Some(a), Some(b)) => dist(a, b),
                _ => f64::NAN,
            }
        })
        .collect();
    let stats = MeanSd::of(&values);
    Distance { values, stats }
}

// ---------------------------------------------------------------------------------------
// Reprojection error

/// The reprojection error of a marker at a frame as the plot computes it: the mean of the
/// chosen cameras' errors where both the camera's status and the 3D status are above
/// `UNDEFINED`; `None` when no camera counts. `camera` `None` is "All Cameras".
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::plotReprojectionError (per frame).
#[must_use]
pub fn reprojection_at(marker: &Marker, camera: Option<usize>, frame: usize) -> Option<f64> {
    if !status_at(&marker.status3d, frame).has_data() {
        return None;
    }
    let cams = match camera {
        None => 0..marker.views.len(),
        Some(c) if c < marker.views.len() => c..c + 1,
        Some(_) => 0..0,
    };
    let mut sum = 0.0;
    let mut count = 0usize;
    for c in cams {
        let v = &marker.views[c];
        if status_at(&v.status, frame).has_data() {
            sum += v.error[frame];
            count += 1;
        }
    }
    // One camera: the C++ takes the error as it is; `sum / 1` is the same number.
    (count > 0).then(|| sum / count as f64)
}

/// A marker's reprojection error per frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Reprojection {
    /// Per frame; NaN where no camera counts (the C++ plots -2 there).
    pub values: Vec<f64>,
    /// Mean and SD over the frames with a value ("Mean reprojection error").
    pub stats: MeanSd,
}

/// Ported from ui/PlotWindow.cpp, PlotWindow::plotReprojectionError.
#[must_use]
pub fn reprojection_error(
    trial: &Trial,
    marker: usize,
    camera: Option<usize>,
    span: FrameSpan,
) -> Reprojection {
    let m = trial.markers.get(marker);
    let values: Vec<f64> = span
        .frames()
        .map(|f| {
            m.and_then(|m| reprojection_at(m, camera, f))
                .unwrap_or(f64::NAN)
        })
        .collect();
    let stats = MeanSd::of(&values);
    Reprojection { values, stats }
}

/// The error `isFrameAboveError` compares with the threshold: the reprojection error, or 0.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::isFrameAboveError(Marker*, int).
#[must_use]
pub fn marker_frame_error(marker: &Marker, camera: Option<usize>, frame: usize) -> f64 {
    reprojection_at(marker, camera, frame).unwrap_or(0.0)
}

// ---------------------------------------------------------------------------------------
// Rigid-body transformation

/// The three Euler angles of a Rodrigues rotation, in degrees, in the order
/// `getRotationEulerAngle` numbers them: 0 `atan2(R21, R22)`, 1
/// `atan2(-R20, sqrt(R21² + R22²))`, 2 `atan2(R10, R00)`.
///
/// Ported from core/RigidBody.cpp, RigidBody::getRotationEulerAngle.
#[must_use]
pub fn euler_angles(rotation: &Vector3) -> [f64; 3] {
    let r = rodrigues_to_matrix(rotation);
    let conv = 180.0 / std::f64::consts::PI;
    [
        conv * r[(2, 1)].atan2(r[(2, 2)]),
        conv * (-r[(2, 0)]).atan2((r[(2, 1)] * r[(2, 1)] + r[(2, 2)] * r[(2, 2)]).sqrt()),
        conv * r[(1, 0)].atan2(r[(0, 0)]),
    ]
}

/// Makes an angle series continuous as the plot does: over the values present (NaN
/// skipped), when a value differs from the previous present (already adjusted) one by more
/// than 270 degrees it is moved one turn towards it.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::plotRigidBody (the loop over `val`).
pub fn unwrap_angles(values: &mut [f64]) {
    let mut prev: Option<f64> = None;
    for v in values.iter_mut().filter(|v| !v.is_nan()) {
        if let Some(p) = prev
            && (p - *v).abs() > 270.0
        {
            if *v < p {
                *v += 360.0;
            } else {
                *v -= 360.0;
            }
        }
        prev = Some(*v);
    }
}

/// A rigid body's pose per frame, unfiltered or filtered.
#[derive(Clone, Debug, PartialEq)]
pub struct Transformation {
    /// True where the body has a pose (`poseComputed` or `poseFiltered`).
    pub valid: Vec<bool>,
    /// The Euler angles in degrees ([`euler_angles`]), each unwrapped ([`unwrap_angles`]);
    /// NaN where there is no pose.
    pub angles: [Vec<f64>; 3],
    /// The translation; NaN where there is no pose.
    pub translation: [Vec<f64>; 3],
}

fn track(rb: &RigidBody, filtered: bool) -> &PoseTrack {
    if filtered { &rb.filtered } else { &rb.raw }
}

/// Ported from ui/PlotWindow.cpp, PlotWindow::plotRigidBody (the values of each graph).
#[must_use]
pub fn transformation(
    trial: &Trial,
    body: usize,
    filtered: bool,
    span: FrameSpan,
) -> Transformation {
    let rb = trial.rigid_bodies.get(body);
    let mut out = Transformation {
        valid: Vec::with_capacity(span.len()),
        angles: Default::default(),
        translation: Default::default(),
    };
    for f in span.frames() {
        let pose = rb.and_then(|rb| track(rb, filtered).pose(f));
        out.valid.push(pose.is_some());
        let e = pose.map(|p| euler_angles(&p.rotation));
        for k in 0..3 {
            out.angles[k].push(e.map_or(f64::NAN, |e| e[k]));
            out.translation[k].push(pose.map_or(f64::NAN, |p| p.translation[k]));
        }
    }
    for a in &mut out.angles {
        unwrap_angles(a);
    }
    out
}

/// The cutoff frequency shown under the transformation and error plots: the body's own
/// when it overrides the trial's.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::plotRigidBody (`cutoff`).
#[must_use]
pub fn cutoff(trial: &Trial, body: usize) -> f64 {
    trial
        .rigid_bodies
        .get(body)
        .map_or(trial.cutoff_frequency, |rb| {
            crate::rigid_bodies::body_cutoff(trial, rb)
        })
}

// ---------------------------------------------------------------------------------------
// Rigid-body error

/// 3D (`comboBoxRigidBodyError` index 0) or 2D error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorDim {
    D3,
    D2,
}

/// A rigid body's fit error per frame.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyError {
    /// The per-frame mean error; NaN where there is no pose (the C++ plots -2 there).
    pub mean: Vec<f64>,
    /// The per-frame SD (the error bars); NaN where there is no pose (the C++ uses 0).
    pub sd: Vec<f64>,
    /// Mean and SD of `mean` over the frames with a pose ("Mean error unfiltered" or
    /// "filtered").
    pub stats: MeanSd,
}

/// Ported from ui/PlotWindow.cpp, PlotWindow::plotRigidBodyError.
///
/// Fixed: for the unfiltered error the C++ sums the squared deviations over the frames
/// with a *filtered* pose (`getPoseFiltered`) while dividing by the number of frames with an
/// unfiltered pose; here both run over the frames with an unfiltered pose. The two agree
/// when the filtered and unfiltered frames are the same.
#[must_use]
pub fn body_error(
    trial: &Trial,
    body: usize,
    filtered: bool,
    dim: ErrorDim,
    span: FrameSpan,
) -> BodyError {
    let rb = trial.rigid_bodies.get(body);
    let mut mean = Vec::with_capacity(span.len());
    let mut sd = Vec::with_capacity(span.len());
    for f in span.frames() {
        let t = rb.map(|rb| track(rb, filtered));
        match t.filter(|t| t.valid.get(f).copied().unwrap_or(false)) {
            Some(t) => {
                let e = t.errors[f];
                let (m, s) = match dim {
                    ErrorDim::D3 => (e.mean_3d, e.sd_3d),
                    ErrorDim::D2 => (e.mean_2d, e.sd_2d),
                };
                mean.push(m);
                sd.push(s);
            }
            None => {
                mean.push(f64::NAN);
                sd.push(f64::NAN);
            }
        }
    }
    let stats = MeanSd::of(&mean);
    BodyError { mean, sd, stats }
}

/// Which poses a rigid-body plot shows (`comboBoxRigidBodyTransType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoseChoice {
    Unfiltered,
    Filtered,
    Both,
}

impl PoseChoice {
    /// From the combo index (0, 1, 2).
    #[must_use]
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::Unfiltered,
            1 => Self::Filtered,
            _ => Self::Both,
        }
    }

    #[must_use]
    pub fn unfiltered(self) -> bool {
        self != Self::Filtered
    }

    #[must_use]
    pub fn filtered(self) -> bool {
        self != Self::Unfiltered
    }
}

/// The error `isFrameAboveError` compares with the threshold for a rigid body: the
/// unfiltered or filtered per-frame mean, or the larger of the two for "Both". Read whether
/// or not the frame has a pose, as the C++ does (a frame without one holds 0 after the
/// trial update).
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::isFrameAboveError(RigidBody*, int).
#[must_use]
pub fn body_frame_error(rb: &RigidBody, choice: PoseChoice, dim: ErrorDim, frame: usize) -> f64 {
    let get = |t: &PoseTrack| {
        t.errors.get(frame).map_or(0.0, |e| match dim {
            ErrorDim::D3 => e.mean_3d,
            ErrorDim::D2 => e.mean_2d,
        })
    };
    match choice {
        PoseChoice::Unfiltered => get(&rb.raw),
        PoseChoice::Filtered => get(&rb.filtered),
        PoseChoice::Both => get(&rb.filtered).max(get(&rb.raw)),
    }
}

/// The next (or previous) frame after `from` whose error is above `threshold`: forward
/// from `from + 1` to the last frame, backward from `from - 1` to the first.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::goToNextAboveError and goToPrevAboveError.
pub fn find_above(
    span: FrameSpan,
    from: usize,
    forward: bool,
    threshold: f64,
    error: impl Fn(usize) -> f64,
) -> Option<usize> {
    if forward {
        (from + 1..=span.last).find(|&f| error(f) > threshold)
    } else {
        (span.first..from.min(span.last + 1))
            .rev()
            .find(|&f| error(f) > threshold)
    }
}

// ---------------------------------------------------------------------------------------
// Status, interpolation and event rows

/// Per camera, the marker's status for each frame of the span (`Undefined` where the
/// marker has no data).
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::drawStatus.
#[must_use]
pub fn status_rows(
    trial: &Trial,
    marker: usize,
    n_cameras: usize,
    span: FrameSpan,
) -> Vec<Vec<MarkerStatus>> {
    let m = trial.markers.get(marker);
    (0..n_cameras)
        .map(|c| {
            span.frames()
                .map(|f| {
                    m.and_then(|m| m.views.get(c))
                        .map_or(MarkerStatus::Undefined, |v| status_at(&v.status, f))
                })
                .collect()
        })
        .collect()
}

/// The marker's interpolation method for each frame of the span.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::drawStatus (`getInterpolation`).
#[must_use]
pub fn interpolation_row(trial: &Trial, marker: usize, span: FrameSpan) -> Vec<Interpolation> {
    let m = trial.markers.get(marker);
    span.frames()
        .map(|f| {
            m.and_then(|m| m.interpolation.get(f).copied())
                .unwrap_or_default()
        })
        .collect()
}

/// The events drawn in the plot (those with drawing on), in order, with their value per
/// frame of the span.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::drawEvents.
#[must_use]
pub fn event_rows(trial: &Trial, span: FrameSpan) -> Vec<(&Event, Vec<bool>)> {
    trial
        .events
        .iter()
        .filter(|e| e.draw)
        .map(|e| {
            (
                e,
                span.frames()
                    .map(|f| e.data.get(f).copied().unwrap_or(false))
                    .collect(),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------------------
// Export

/// What "Export" writes: the plot type and the choices it reads.
#[derive(Clone, Debug, PartialEq)]
pub enum PlotExport {
    /// "2D positions": `camera` `None` is "All Cameras".
    Positions2d {
        marker: usize,
        camera: Option<usize>,
    },
    /// "3D positions".
    Positions3d { marker: usize },
    /// "Marker to Marker distance"; `target_label` is `comboBoxMarker2`'s text.
    Distance {
        marker: usize,
        target: DistanceTarget,
        target_label: String,
    },
    /// "Reprojection Error"; `camera_label` is `comboBoxCamera`'s text.
    Reprojection {
        marker: usize,
        camera: Option<usize>,
        camera_label: String,
    },
    /// "Rigid Body transformation"; `part` is `comboBoxRigidBodyTransPart`'s index.
    Transformation {
        body: usize,
        choice: PoseChoice,
        part: usize,
    },
    /// "Rigid Body Error".
    BodyError {
        body: usize,
        choice: PoseChoice,
        dim: ErrorDim,
    },
}

/// The CSV text "Export" writes for the trial's start to end frame: a header row, then one
/// row per frame starting with the 1-based frame number. Numbers are `%.17g` (the C++
/// `ofstream` with precision 17), except the rigid-body columns, which are
/// `QString::number(v, 'f', 12)`; a missing value is `NaN`.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::saveData, with these defects fixed:
/// - 3D positions and reprojection error: a frame without a value is followed by an empty
///   line in the C++ (`"NaN" << std::endl` and then `std::endl` again); one line here;
/// - 2D positions: the C++ writes `NaN,NaN,NaN, NaN` (with a space) for a missing point;
///   `NaN,NaN,NaN,NaN` here;
/// - rigid-body transformation: the C++ indexes its per-frame columns by the absolute frame,
///   so a trial not starting at frame 1 writes the wrong frames and reads past the end;
///   and its angle unwrapping treats a frame without a pose (held as `DBL_MIN`) as an angle,
///   so after an angle beyond ±270 such a frame is written as ±360 instead of `NaN`. Here
///   the columns are the plot's ([`transformation`]).
#[must_use]
pub fn export_csv(
    doc: &Document,
    trial_index: usize,
    what: &PlotExport,
    span: FrameSpan,
) -> String {
    let trial = doc.trial(trial_index);
    let mut out = String::new();
    let n_cams = doc.project().cameras.len();
    let g = format_g17;
    match what {
        PlotExport::Positions2d { marker, camera } => {
            let cams: Vec<usize> = camera.map_or_else(|| (0..n_cams).collect(), |c| vec![c]);
            let m1 = marker + 1;
            let header: Vec<String> = cams
                .iter()
                .map(|c| {
                    let c1 = c + 1;
                    format!(
                        "Cam{c1}_Marker_{m1}_x,Cam{c1}_Marker_{m1}_y,Cam{c1}_Marker_{m1}_x_undistorted,Cam{c1}_Marker_{m1}_y_undistorted"
                    )
                })
                .collect();
            out.push_str("Frame,");
            out.push_str(&header.join(","));
            out.push('\n');
            let m = trial.markers.get(*marker);
            for f in span.frames() {
                let cells: Vec<String> = cams
                    .iter()
                    .map(|&c| match m.and_then(|m| m.point2d(c, f)) {
                        Some(p) => {
                            // Ledger A19 fixed: no undistorted position is written as missing.
                            match doc.camera(c).model.try_undistort_point(p, true, true, true) {
                                Some(u) => format!("{},{},{},{}", g(p.x), g(p.y), g(u.x), g(u.y)),
                                None => format!("{},{},NaN,NaN", g(p.x), g(p.y)),
                            }
                        }
                        None => "NaN,NaN,NaN,NaN".to_owned(),
                    })
                    .collect();
                out.push_str(&format!("{},{}\n", f + 1, cells.join(",")));
            }
        }
        PlotExport::Positions3d { marker } => {
            let m1 = marker + 1;
            out.push_str(&format!(
                "Frame,Marker_{m1}_x,Marker_{m1}_y,Marker_{m1}_z\n"
            ));
            let m = trial.markers.get(*marker);
            for f in span.frames() {
                match m.and_then(|m| m.point3d(f)) {
                    Some(p) => {
                        out.push_str(&format!("{},{},{},{}\n", f + 1, g(p.x), g(p.y), g(p.z)))
                    }
                    None => out.push_str(&format!("{},NaN,NaN,NaN\n", f + 1)),
                }
            }
        }
        PlotExport::Distance {
            marker,
            target,
            target_label,
        } => {
            out.push_str(&format!(
                "Frame,Marker_{}_to_Marker_{target_label}\n",
                marker + 1
            ));
            let d = distance(trial, *marker, *target, span);
            for (f, v) in span.frames().zip(&d.values) {
                out.push_str(&format!("{},{}\n", f + 1, num_or_nan(*v, g)));
            }
        }
        PlotExport::Reprojection {
            marker,
            camera,
            camera_label,
        } => {
            out.push_str(&format!(
                "Frame,ReprojectionError_Marker_{}_{camera_label}\n",
                marker + 1
            ));
            let r = reprojection_error(trial, *marker, *camera, span);
            for (f, v) in span.frames().zip(&r.values) {
                out.push_str(&format!("{},{}\n", f + 1, num_or_nan(*v, g)));
            }
        }
        PlotExport::Transformation { body, choice, part } => {
            let b1 = body + 1;
            let unf = transformation(trial, *body, false, span);
            let fil = transformation(trial, *body, true, span);
            // (header name, unfiltered column, filtered column), in the C++ column order.
            let names = ["Yaw", "Pitch", "Roll", "X", "Y", "Z"];
            let wanted = |k: usize| -> bool {
                match part {
                    0 => true,
                    1 => k < 3,
                    2 => k >= 3,
                    p => *p == k + 3,
                }
            };
            let col = |t: &Transformation, k: usize| -> Vec<f64> {
                if k < 3 {
                    t.angles[k].clone()
                } else {
                    t.translation[k - 3].clone()
                }
            };
            let mut header = Vec::new();
            let mut cols: Vec<Vec<f64>> = Vec::new();
            for (k, name) in names.iter().enumerate() {
                if !wanted(k) {
                    continue;
                }
                if choice.unfiltered() {
                    header.push(format!("Unfiltered_{name}_RB{b1}"));
                    cols.push(col(&unf, k));
                }
                if choice.filtered() {
                    header.push(format!("Filtered_{name}_RB{b1}"));
                    cols.push(col(&fil, k));
                }
            }
            out.push_str("Frame,");
            out.push_str(&header.join(","));
            out.push('\n');
            for (i, f) in span.frames().enumerate() {
                let cells: Vec<String> = cols.iter().map(|c| num_or_nan(c[i], fixed12)).collect();
                out.push_str(&format!("{},{}\n", f + 1, cells.join(",")));
            }
        }
        PlotExport::BodyError { body, choice, dim } => {
            let b1 = body + 1;
            let d = match dim {
                ErrorDim::D3 => "3D",
                ErrorDim::D2 => "2D",
            };
            let mut header = Vec::new();
            let mut cols = Vec::new();
            if choice.unfiltered() {
                header.push(format!("Unfiltered_RigidBodyError{d}_RB{b1}_Mean"));
                header.push(format!("Unfiltered_RigidBodyError{d}_RB{b1}_SD"));
                let e = body_error(trial, *body, false, *dim, span);
                cols.push(e.mean);
                cols.push(e.sd);
            }
            if choice.filtered() {
                header.push(format!("Filtered_RigidBodyError{d}_RB{b1}_Mean"));
                header.push(format!("Filtered_RigidBodyError{d}_RB{b1}_SD"));
                let e = body_error(trial, *body, true, *dim, span);
                cols.push(e.mean);
                cols.push(e.sd);
            }
            out.push_str("Frame,");
            out.push_str(&header.join(","));
            out.push('\n');
            for (i, f) in span.frames().enumerate() {
                let cells: Vec<String> = cols.iter().map(|c| num_or_nan(c[i], fixed12)).collect();
                out.push_str(&format!("{},{}\n", f + 1, cells.join(",")));
            }
        }
    }
    out
}

fn fixed12(v: f64) -> String {
    format_qt_fixed(v, 12)
}

/// A number as the plot labels write it (`QString::number(v)`: `%g` with 6 digits, ties
/// away from zero).
#[must_use]
pub fn qt_number(v: f64) -> String {
    xma_io::cfmt::format_qt_number(v)
}

/// Reads the extra plot's data file: one row per line, cells separated by commas, each
/// read as `QString::toDouble` does (0 when it is not a number). Column `i` of the result
/// holds cell `i` of every line that has one.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::on_toolButtonExtraPlot_clicked. Fixed: the
/// C++ loop (`while (!safeGetline(fin, line).eof())`) drops the last line of a file that
/// does not end with a line break.
#[must_use]
pub fn read_extra_columns(text: &str) -> (Vec<Vec<f64>>, usize) {
    let mut cols: Vec<Vec<f64>> = Vec::new();
    let mut rows = 0;
    for line in text.lines() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        for (i, cell) in line.split(',').enumerate() {
            if cols.len() <= i {
                cols.push(Vec::new());
            }
            cols[i].push(cell.trim().parse::<f64>().unwrap_or(0.0));
        }
        rows += 1;
    }
    (cols, rows)
}

fn num_or_nan(v: f64, fmt: impl Fn(f64) -> String) -> String {
    if v.is_nan() { "NaN".to_owned() } else { fmt(v) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_sd_skips_nan_and_uses_n_minus_one() {
        let s = MeanSd::of(&[1.0, f64::NAN, 3.0]);
        assert_eq!((s.mean, s.sd, s.count), (2.0, 2f64.sqrt(), 2));
        assert_eq!(MeanSd::of(&[f64::NAN]), MeanSd::default());
        let one = MeanSd::of(&[5.0]);
        assert_eq!((one.mean, one.sd), (5.0, 0.0));
    }

    #[test]
    fn unwrap_moves_one_turn_towards_the_previous_present_value() {
        let mut v = [170.0, f64::NAN, -175.0, -170.0, 179.0];
        unwrap_angles(&mut v);
        assert_eq!(v[0], 170.0);
        assert!(v[1].is_nan());
        assert_eq!(v[2], 185.0);
        assert_eq!(v[3], 190.0);
        assert_eq!(v[4], 179.0);
    }

    #[test]
    fn euler_angles_of_the_identity_are_zero() {
        assert_eq!(euler_angles(&Vector3::new(0.0, 0.0, 0.0)), [0.0, 0.0, 0.0]);
        let a = euler_angles(&Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
        assert!((a[2] - 90.0).abs() < 1e-12 && a[0].abs() < 1e-12);
    }

    #[test]
    fn find_above_searches_after_or_before_the_frame() {
        let span = FrameSpan { first: 2, last: 9 };
        let e = |f: usize| if f == 3 || f == 8 { 1.0 } else { 0.0 };
        assert_eq!(find_above(span, 5, true, 0.5, e), Some(8));
        assert_eq!(find_above(span, 5, false, 0.5, e), Some(3));
        assert_eq!(find_above(span, 8, true, 0.5, e), None);
        assert_eq!(find_above(span, 3, false, 0.5, e), None);
    }

    #[test]
    fn span_from_one_based() {
        let s = FrameSpan::from_one_based(1, 145);
        assert_eq!((s.first, s.last, s.len()), (0, 144, 145));
    }
}
