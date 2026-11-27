//! Range fitting and placement arithmetic from `PlotWindow.cpp`, over plain data.
//!
//! The C++ recomputes the Y ranges on every redraw (every frame change, via
//! `MainWindow::redrawGL` → `PlotWindow::draw`), and the X range only on
//! "Fit plot", "Display time", a trial change or a frame-range change
//! (`resetRange`). The caller reproduces that by calling these functions and
//! writing the results into [`super::PlotView`].
//!
//! The quirks are kept: running minima start at 10000 and maxima at 0, the
//! range carried from one axis to the next in the same plot, the 0.2 or 1e-11
//! minimum range, the status-row headroom.

use super::model::FrameAxis;

/// Running minimum and maximum as XMALab keeps them: the minimum starts at
/// 10000 and the maximum at 0, so all-negative data has maximum 0 and data
/// all above 10000 has minimum 10000.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinMax {
    /// Smallest value seen, or 10000.
    pub min: f64,
    /// Largest value seen, or 0.
    pub max: f64,
}

impl Default for MinMax {
    fn default() -> Self {
        Self::new()
    }
}

impl MinMax {
    /// `max_val = 0; min_val = 10000;`
    pub const fn new() -> Self {
        Self {
            min: 10000.0,
            max: 0.0,
        }
    }

    /// `if (v > max_val) max_val = v; if (v < min_val) min_val = v;`
    pub fn add(&mut self, v: f64) {
        if v > self.max {
            self.max = v;
        }
        if v < self.min {
            self.min = v;
        }
    }

    /// Adds every value (NaN values change nothing, as in the C++ comparisons).
    pub fn extend(mut self, values: impl IntoIterator<Item = f64>) -> Self {
        for v in values {
            self.add(v);
        }
        self
    }
}

/// A centre and a range as `PlotWindow.cpp` computes them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CenteredFit {
    /// `(max_val + min_val) * 0.5`.
    pub center: f64,
    /// `max_val - min_val`, or the incoming range if that is not larger.
    pub range: f64,
}

impl CenteredFit {
    /// `center = (max + min) * 0.5; range = (max - min) > range ? (max - min) : range;`
    ///
    /// `carried` is the value of the C++ `range` variable before the line: 0.2
    /// for the first axis of a plot (1e-11 in `plotRigidBody`), and the previous
    /// axis's range for the second axis of the same plot.
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::plot2D (and the other plot functions).
    pub fn new(mm: MinMax, carried: f64) -> Self {
        let spread = mm.max - mm.min;
        Self {
            center: (mm.max + mm.min) * 0.5,
            range: if spread > carried { spread } else { carried },
        }
    }

    /// `(center - range * half, center + range * half + top)`; `half` is 0.5,
    /// or 0.55 in `plotRigidBody`; `top` is the status headroom.
    pub fn axis(&self, half: f64, top: f64) -> (f64, f64) {
        (
            self.center - self.range * half,
            self.center + self.range * half + top,
        )
    }

    /// The data span without headroom, which the cursor line and the selection
    /// rectangle span vertically.
    pub fn span(&self, half: f64) -> (f64, f64) {
        self.axis(half, 0.0)
    }
}

/// What the status rows need: the number of cameras and whether the marker has
/// any interpolation set (`Marker::getHasInterpolation`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatusLayout {
    /// `Project::getCameras().size()`.
    pub n_cameras: usize,
    /// Whether the interpolation row is shown.
    pub has_interpolation: bool,
}

/// Headroom added above the data for the status rows:
/// `range * (0.05 * (cameras + 1)) + (has_interpolation ? range * 0.07 : 0)`,
/// or 0 when "Display status" is off (`status` is `None`).
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::plot2D (`offset`, `offsetInterpolation`).
pub fn status_offset(range: f64, status: Option<StatusLayout>) -> f64 {
    match status {
        None => 0.0,
        Some(s) => {
            let interp = if s.has_interpolation {
                range * 0.07
            } else {
                0.0
            };
            range * (0.05 * (s.n_cameras as f64 + 1.0)) + interp
        }
    }
}

/// A fitted Y axis: the axis range, and the data span the cursor and the
/// selection rectangle cover.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YFit {
    /// The axis range to set.
    pub axis: (f64, f64),
    /// The span for the cursor line and the selection rectangle.
    pub span: (f64, f64),
    /// The C++ `range` variable afterwards (carried to the next axis).
    pub range: f64,
}

/// Y fit for the position plots (2D: each axis; 3D: the one axis):
/// centred, minimum range `carried`, half-range factor 0.5, plus status headroom.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::plot2D and PlotWindow::plot3D.
pub fn fit_positions(mm: MinMax, carried: f64, status: Option<StatusLayout>) -> YFit {
    let c = CenteredFit::new(mm, carried);
    YFit {
        axis: c.axis(0.5, status_offset(c.range, status)),
        span: c.span(0.5),
        range: c.range,
    }
}

/// Y fit for the 2D-position plot: left axis from the X coordinates, right
/// axis from the Y coordinates with the left range carried over (so the right
/// range is never smaller than the left one).
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::plot2D.
pub fn fit_2d(x: MinMax, y: MinMax, status: Option<StatusLayout>) -> (YFit, YFit) {
    let left = fit_positions(x, 0.2, status);
    let right = fit_positions(y, left.range, status);
    (left, right)
}

/// Y fit for the distance and rigid-body error plots: centred, minimum 0.2, no headroom.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::plotDistance and PlotWindow::plotRigidBodyError.
pub fn fit_centered(mm: MinMax) -> YFit {
    let c = CenteredFit::new(mm, 0.2);
    YFit {
        axis: c.axis(0.5, 0.0),
        span: c.span(0.5),
        range: c.range,
    }
}

/// Which parts a rigid-body transformation plot shows (`comboBoxRigidBodyTransPart`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RigidBodyParts {
    /// "Full": rotation on the left axis, translation on the right.
    Full,
    /// "Euler Angles", "Yaw", "Roll", "Pitch": rotation on the left axis.
    Rotation,
    /// "Translation", "X", "Y", "Z": translation on the left axis.
    Translation,
}

/// Y fits for the rigid-body transformation plot: `(left, right)`, the right
/// one only for "Full". Minimum range 1e-11, half-range factor 0.55. For "Full"
/// the translation axis is fitted first and its range carried to the rotation axis.
///
/// The returned span (for cursor and selection) is the left axis's.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::plotRigidBody.
pub fn fit_rigid_body(rot: MinMax, trans: MinMax, parts: RigidBodyParts) -> (YFit, Option<YFit>) {
    let fit = |mm, carried| {
        let c = CenteredFit::new(mm, carried);
        YFit {
            axis: c.axis(0.55, 0.0),
            span: c.span(0.55),
            range: c.range,
        }
    };
    match parts {
        RigidBodyParts::Full => {
            let right = fit(trans, 1e-11);
            let left = fit(rot, right.range);
            (left, Some(right))
        }
        RigidBodyParts::Rotation => (fit(rot, 1e-11), None),
        RigidBodyParts::Translation => (fit(trans, 1e-11), None),
    }
}

/// Y fit for the reprojection-error plot: from -0.5 to `max * 1.1` (or 0.2 if
/// the maximum is not above 0.2), plus status headroom. The span runs from -0.5
/// to the range without headroom.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::plotReprojectionError.
pub fn fit_reprojection(max_val: f64, status: Option<StatusLayout>) -> YFit {
    let range = if max_val > 0.2 { max_val * 1.1 } else { 0.2 };
    YFit {
        axis: (-0.5, range + status_offset(range, status)),
        span: (-0.5, range),
        range,
    }
}

/// The X range "Fit plot" sets: from the first to the last frame's x, with no
/// padding. `None` when QCustomPlot would reject it (a one-frame trial), in
/// which case the range is left as it was.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::resetRange.
pub fn fit_x(frames: &FrameAxis) -> Option<(f64, f64)> {
    let r = (frames.x_of(frames.first), frames.x_of(frames.last));
    valid_range(r).then_some(r)
}

/// The Y extents of the status rows, in left-axis coordinates, for an axis
/// range `y`: one row per camera from the top down, then the interpolation row
/// (after a gap) when the marker has interpolation.
///
/// `height = size / (1 + 0.05 * (cameras + 1) + (interp ? 0.07 : 0)) * 0.05`,
/// `y_max = upper - height / 2`; camera row c spans
/// `[y_max - (c + 1) * height + height / 2.5, y_max - c * height + height / 2.5]`;
/// the interpolation row spans `[y_max - (cameras + 1) * height, y_max - cameras * height]`.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::drawStatus.
pub fn status_rows(y: (f64, f64), status: StatusLayout) -> StatusRows {
    let size = y.1 - y.0;
    let interp = if status.has_interpolation { 0.07 } else { 0.0 };
    let n = status.n_cameras as f64;
    let height = size / (1.0 + (0.05 * (n + 1.0) + interp)) * 0.05;
    let y_max = y.1 - height / 2.0;
    let cameras = (0..status.n_cameras)
        .map(|c| {
            let c = c as f64;
            (
                y_max - (c + 1.0) * height + height / 2.5,
                y_max - c * height + height / 2.5,
            )
        })
        .collect();
    let interpolation = status
        .has_interpolation
        .then_some((y_max - (n + 1.0) * height, y_max - n * height));
    StatusRows {
        cameras,
        interpolation,
    }
}

/// Output of [`status_rows`]: `(bottom, top)` per row.
#[derive(Clone, Debug, PartialEq)]
pub struct StatusRows {
    /// One row per camera, camera 0 at the top.
    pub cameras: Vec<(f64, f64)>,
    /// The interpolation row, if shown.
    pub interpolation: Option<(f64, f64)>,
}

/// The Y extents of the event rows, in left-axis coordinates, for an axis range
/// `y` and `n_drawn` events with drawing switched on. The rows split the data
/// area (below the status headroom when status rows are shown for a marker) into
/// equal horizontal stripes, the first event at the top.
///
/// `status` is `Some` only when a marker is plotted and "Display status" is on,
/// as in the C++ (`marker && checkBoxStatus->isChecked()`).
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::drawEvents.
pub fn event_rows(y: (f64, f64), n_drawn: usize, status: Option<StatusLayout>) -> Vec<(f64, f64)> {
    if n_drawn == 0 {
        return Vec::new();
    }
    let size = y.1 - y.0;
    let height = match status {
        Some(s) => {
            let interp = if s.has_interpolation { 0.07 } else { 0.0 };
            size / (1.0 + (0.05 * (s.n_cameras as f64 + 1.0) + interp))
        }
        None => size,
    };
    let start = y.0 + height;
    let step = height / n_drawn as f64;
    (0..n_drawn)
        .map(|k| {
            let k = k as f64;
            (start - (k + 1.0) * step, start - k * step)
        })
        .collect()
}

/// Makes Euler-angle series continuous the way `plotRigidBody` does: when two
/// consecutive values differ by more than 270 degrees, the second is moved by
/// one turn towards the first. Only one turn is ever added, and the comparison
/// uses the already-adjusted previous value.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::plotRigidBody.
pub fn unwrap_angles(values: &mut [f64]) {
    for i in 1..values.len() {
        if (values[i - 1] - values[i]).abs() > 270.0 {
            if values[i] < values[i - 1] {
                values[i] += 360.0;
            } else {
                values[i] -= 360.0;
            }
        }
    }
}

/// Mean and sample standard deviation as the plot labels compute them: the
/// mean over the values where `in_mean` holds, the squared deviations summed
/// over the values where `in_sd` holds, divided by `count_mean - 1`. (In
/// `plotRigidBodyError` the two masks differ for the unfiltered error, a C++ bug
/// kept here.) Mean 0 when nothing counts; SD 0 unless more than one value counts.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::plotReprojectionError, plotDistance
/// and plotRigidBodyError.
pub fn mean_sd(
    values: &[f64],
    in_mean: impl Fn(usize) -> bool,
    in_sd: impl Fn(usize) -> bool,
) -> (f64, f64) {
    let mut mean = 0.0;
    let mut count = 0usize;
    for (i, v) in values.iter().enumerate() {
        if in_mean(i) {
            mean += v;
            count += 1;
        }
    }
    if count > 0 {
        mean /= count as f64;
    }
    let mut sd = 0.0;
    for (i, v) in values.iter().enumerate() {
        if in_sd(i) {
            sd += (v - mean).powi(2);
        }
    }
    if count > 1 {
        sd = (sd / (count - 1) as f64).sqrt();
    }
    (mean, sd)
}

// ---- QCustomPlot range arithmetic ----

/// `QCPRange::minRange`.
pub const QCP_MIN_RANGE: f64 = 1e-280;
/// `QCPRange::maxRange`.
pub const QCP_MAX_RANGE: f64 = 1e250;

/// Whether QCustomPlot accepts a range.
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPRange::validRange.
pub fn valid_range(r: (f64, f64)) -> bool {
    let (lower, upper) = r;
    lower > -QCP_MAX_RANGE
        && upper < QCP_MAX_RANGE
        && (lower - upper).abs() > QCP_MIN_RANGE
        && (lower - upper).abs() < QCP_MAX_RANGE
        && !(lower > 0.0 && (upper / lower).is_infinite())
        && !(upper < 0.0 && (lower / upper).is_infinite())
}

/// `QCPAxis::setRange(lower, upper)`: ignored if invalid, otherwise normalised.
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPAxis::setRange.
pub fn set_range(current: (f64, f64), new: (f64, f64)) -> (f64, f64) {
    if !valid_range(new) {
        return current;
    }
    if new.0 > new.1 { (new.1, new.0) } else { new }
}

/// Zooms a range by `factor` about `center`; rejected if the result is invalid.
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPAxis::scaleRange (linear).
pub fn scale_range(r: (f64, f64), factor: f64, center: f64) -> (f64, f64) {
    let new = (
        (r.0 - center) * factor + center,
        (r.1 - center) * factor + center,
    );
    set_range(r, new)
}

/// QCustomPlot's wheel zoom factor per wheel step (`mRangeZoomFactorHorz/Vert`).
pub const QCP_ZOOM_FACTOR: f64 = 0.85;

/// The factor for `steps` wheel steps (one notch = 1; positive = away from the user = zoom in).
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPAxisRect::wheelEvent.
pub fn wheel_factor(steps: f64) -> f64 {
    QCP_ZOOM_FACTOR.powf(steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minmax_keeps_cpp_sentinels() {
        let mm = MinMax::new().extend([-5.0, -3.0]);
        assert_eq!((mm.min, mm.max), (-5.0, 0.0));
        let mm = MinMax::new().extend([20000.0]);
        assert_eq!((mm.min, mm.max), (10000.0, 20000.0));
    }

    #[test]
    fn status_rows_stack_from_the_top() {
        let rows = status_rows(
            (0.0, 1.2),
            StatusLayout {
                n_cameras: 3,
                has_interpolation: false,
            },
        );
        assert_eq!(rows.cameras.len(), 3);
        assert!(rows.cameras[0].1 < 1.2 && rows.cameras[0].0 > rows.cameras[1].0);
        assert!(rows.interpolation.is_none());
    }
}
