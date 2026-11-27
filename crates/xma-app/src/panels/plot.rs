//! The Plot panel.
//!
//! Ported from `PlotWindow.ui` and ui/PlotWindow.cpp. The numbers come from
//! [`xma_engine::view::plots`]; this file holds the form, which controls show for which
//! plot type, the axis labels and ranges, and the mapping onto the plot widget
//! ([`crate::widgets::plot`]).
//!
//! # The form
//!
//! The plot (`plotWidget`, with `plotWidgetExtra` above it in a vertical splitter when
//! shown) fills the panel; below it one row, left to right: `labelPlotType` "Plot type : ",
//! `comboBoxPlotType`, then either `frameMarkerPlot` (plot types 0 to 3: `labelError`
//! "Error threshold : " and `doubleSpinBoxError` (type 3), `labelCamera` "Camera : " and
//! `comboBoxCamera` (types 0 and 3), `labelMarker1` "Marker : " and `comboBoxMarker1`,
//! `labelMarker2` "Marker : " and `comboBoxMarker2` (type 2)) or `frameRigidBodyPlot`
//! (types 4 and 5: `labelError_2` "Error threshold : " and `doubleSpinBoxErrorRB` (type 5),
//! `labelRigidBody` "Rigid Body : ", `comboBoxRigidBody`, `comboBoxRigidBodyTransType`,
//! `comboBoxRigidBodyTransPart` (type 4), `comboBoxRigidBodyError` (type 5),
//! `pushButtonUpdate` "Update"), then `checkBoxStatus` "Display status" (types 0, 1, 3),
//! `checkBoxTime` "Display time " (when the trial has a recording speed), `pushButton_Reset`
//! "Fit plot", `pushButtonSave` "Export"; and `toolButtonExtraPlot` "+" outside the row's
//! frame. The per-control status is in `crates/xma-app/ui-status/PlotWindow.md`.
//!
//! When the panel is narrow ([`PlotPanel::row_lines`]): the buttons, check boxes, labels
//! and spin boxes keep their size, and the combo boxes share what is left, each at most as
//! wide as its contents and at least 48 points, with their text cut (Qt's `Ignored` and
//! `Maximum` size policies in the form squeeze them the same way). When even that does not
//! fit, the controls continue on a second (or third) line, a label staying with the
//! control it names; Qt would squeeze the labels and check boxes as well, which can leave
//! them unreadable. A line wider than the panel on its own scrolls sideways.
//!
//! # Redraws
//!
//! The C++ redraws the plot (`PlotWindow::draw`, which refits the Y ranges) whenever the
//! camera views redraw (`MainWindow::redrawGL`: every frame change and selection change),
//! when a control changes, and after "Fit plot", "Display time" and "Display status"; it
//! refits the X range (`resetRange`) only on "Fit plot", the two check boxes, and a trial,
//! workspace or start/end-frame change (`SequenceNavigationFrame::setStartEndSequence`).
//! Here a redraw refits the Y ranges ([`PlotPanel::redraw`]); it computes the plotted data
//! and the combo lists again only when something they are computed from changed (the
//! document's content, the trial, its start and end frames, "Display time", the panel's
//! choices), so a frame step or a new selection reuses them. A wheel zoom or drag lasts
//! until the next redraw, as in the C++.
//!
//! # Fixed C++ defects
//!
//! - "Rigid Body Error" with "Both": the C++ overwrites the unfiltered graph with the
//!   threshold line (both are `graph(1)`) and attaches the unfiltered error bars to the two
//!   threshold points; here the unfiltered series, its error bars and the threshold line are
//!   all drawn.
//! - The unfiltered rigid-body error's SD (x-axis label) is summed over the frames with a
//!   filtered pose in the C++; here over the frames with an unfiltered pose
//!   ([`xma_engine::view::plots::body_error`]).
//! - "Rigid Body transformation", "Full": the C++ labels the left axis "Translation" and the
//!   right "Rotationangle" but plots the angles on the left; the labels follow the data here.
//! - "Rigid Body Error" leaves the right axis of the previous plot type visible (with its
//!   label and range) in the C++; it is hidden here.
//! - "Reprojection Error" in time mode: the label lacks ": " after "Mean reprojection error"
//!   in the C++.
//! - `updateMarkers` restores `comboBoxMarker2` to the index of `comboBoxMarker1`; here to
//!   its own.
//! - `labelError_2` stays visible without its spin box for "Rigid Body transformation"; it
//!   is hidden with the spin box here.
//! - The extra plot: the last line of a file without a final line break is dropped, and the
//!   Y range of all-negative data ends at `DBL_MIN` (just above 0) in the C++.
//!
//! # Editing
//!
//! The panel holds the selected frame range (`startFrame`, `endFrame`, `noSelection`:
//! Ctrl+click, Ctrl+drag, Shift held while the frame changes, [`FrameSelection`]) and the
//! choices the range edits act on. Each edit asks XMALab's question first (an
//! [`EditRequest::Ask`], `crate::dialogs::ask`) and changes the project only on OK:
//!
//! | Trigger | Action | What |
//! |---|---|---|
//! | Delete, Backspace (plot focused) | E44 | [`PlotPanel::range_delete_ask`] |
//! | Ctrl+U | E45 | the same, untrackable |
//! | Ctrl+I, Toolbox "Set Interpolation" | E46 | `set_interpolation`, [`PlotPanel::interpolation_ask`] |
//! | Shift+Delete, Shift+Backspace | E47 | [`PlotPanel::above_error_ask`] |
//! | "Update" | E48 | `Edit::RecomputeBodies` (no question) |
//! | O, Ctrl+O, Events "Mark frames", "Unmark frames" | E51, E52 | [`PlotPanel::event_request`] (no question) |
//!
//! The keys reach the panel as actions (`Panel::on_action`); the range is ordered and
//! clamped to the trial's start and end frames. They act in the Marker tracking workspace
//! only, where the plot is shown (the C++ shortcuts would act on the hidden plot's
//! choices elsewhere). Fixed C++ defects: D19 (no marker or body, or an index the project
//! no longer has: nothing is asked and nothing changes), the empty camera text of the
//! rigid-body question for plot type 5, a selection wholly outside the start and end frames
//! (nothing is asked), and E47 acting on the active marker while naming the plotted one.

use egui::{Align, Color32, Layout, Ui};
use xma_engine::Document;
use xma_engine::view::plots::{
    self, DistanceTarget, ErrorDim, FrameSpan, PlotExport, PoseChoice, qt_number,
};
use xma_model::{Interpolation, MarkerStatus, Trial};

use xma_engine::Edit;
use xma_engine::edit::{CameraSel, ErrorTarget, FrameRange};

use super::{AppEvent, Cx, Panel, PanelId};
use crate::actions::Action;
use crate::dialogs::ask::{Ask, AskKind, INTERPOLATION_METHODS, Then};
use crate::editing::EditRequest;
use crate::settings::parse_color;
use crate::state::Workspace;
use crate::widgets::plot::ranging::{self, MinMax, RigidBodyParts, StatusLayout, YFit};
use crate::widgets::plot::{
    BandRow, Cursor, Dash, ErrorBars, ErrorStyle, FrameAxis, FrameSelection, LineStyle, Pen,
    PlotEvent, PlotSpec, PlotStyle, PlotView, PlotWidget, ScatterShape, Series, SeriesData, YAxis,
    qt,
};

/// `comboBoxPlotType` items.
pub const PLOT_TYPES: [&str; 6] = [
    "2D positions",
    "3D positions",
    "Marker to Marker distance",
    "Reprojection Error",
    "Rigid Body transformation",
    "Rigid Body Error",
];
/// `comboBoxRigidBodyTransType` items.
pub const TRANS_TYPES: [&str; 3] = ["Unfiltered", "Filtered", "Both"];
/// `comboBoxRigidBodyTransPart` items.
pub const TRANS_PARTS: [&str; 9] = [
    "Full",
    "Euler Angles",
    "Translation",
    "Yaw",
    "Roll",
    "Pitch",
    "X",
    "Y",
    "Z",
];
/// `comboBoxRigidBodyError` items.
pub const RB_ERRORS: [&str; 2] = ["3D", "2D"];

/// The extra plot's colour cycle (`PlotWindow::updateExtraPlot`).
const EXTRA_COLOURS: [Color32; 12] = [
    Color32::from_rgb(255, 0, 0),
    Color32::from_rgb(0, 255, 0),
    Color32::from_rgb(0, 0, 255),
    Color32::from_rgb(255, 255, 0),
    Color32::from_rgb(0, 255, 255),
    Color32::from_rgb(255, 0, 255),
    Color32::from_rgb(128, 0, 0),
    Color32::from_rgb(0, 128, 0),
    Color32::from_rgb(0, 0, 128),
    Color32::from_rgb(128, 128, 0),
    Color32::from_rgb(0, 128, 128),
    Color32::from_rgb(128, 0, 128),
];

/// The combo-box contents (`PlotWindow::updateMarkers`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Items {
    pub cameras: Vec<String>,
    pub markers: Vec<String>,
    pub marker2: Vec<(String, DistanceTarget)>,
    pub bodies: Vec<String>,
}

#[derive(Clone, Debug)]
enum OwnedData {
    Frames(Vec<f64>),
    Xy(Vec<f64>, Vec<f64>),
}

#[derive(Clone, Debug)]
struct OwnedSeries {
    data: OwnedData,
    axis: YAxis,
    line: LineStyle,
    scatter: Option<(ScatterShape, f32)>,
    pen: Pen,
    errors: Option<(Vec<f64>, Pen)>,
}

impl OwnedSeries {
    fn frames(values: Vec<f64>, pen: Pen) -> Self {
        Self {
            data: OwnedData::Frames(values),
            axis: YAxis::Left,
            line: LineStyle::Line,
            scatter: None,
            pen,
            errors: None,
        }
    }

    fn xy(x: Vec<f64>, y: Vec<f64>, pen: Pen) -> Self {
        Self {
            data: OwnedData::Xy(x, y),
            ..Self::frames(Vec::new(), pen)
        }
    }

    fn line(mut self, l: LineStyle) -> Self {
        self.line = l;
        self
    }

    fn scatter(mut self, s: ScatterShape, size: f32) -> Self {
        self.scatter = Some((s, size));
        self
    }

    fn axis(mut self, a: YAxis) -> Self {
        self.axis = a;
        self
    }

    fn errors(mut self, e: Vec<f64>, pen: Pen) -> Self {
        self.errors = Some((e, pen));
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BandColors {
    Status,
    Interpolation,
    Event([Color32; 2]),
}

#[derive(Clone, Debug)]
struct OwnedBand {
    values: Vec<u8>,
    colors: BandColors,
    y: (f64, f64),
    outline: bool,
}

/// What one redraw computed: the C++ hands this to QCustomPlot and it stays until the next
/// redraw.
#[derive(Clone, Debug)]
struct Cache {
    frames: FrameAxis,
    x_label: String,
    left_label: String,
    right_label: Option<String>,
    series: Vec<OwnedSeries>,
    bands: Vec<OwnedBand>,
    /// The Y extent of the cursor line and the selection band.
    span: Option<(f64, f64)>,
    left: Option<(f64, f64)>,
    right: Option<(f64, f64)>,
}

/// A series as drawn ([`PlotPanel::series_samples`]): the 0-based frame of its first value
/// for a one-value-per-frame series (`None` for explicit x and y), and each sample's plot x
/// and y.
pub type SeriesSamples = (Option<usize>, Vec<(f64, f64)>);

/// The extra plot's data (`extraData`, `extraPos`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Extra {
    pub columns: Vec<Vec<f64>>,
    /// `rate * row`, in frames.
    pub pos: Vec<f64>,
}

/// What the series of one redraw were computed from ([`PlotPanel::data_key`]): while it
/// stays the same, a redraw reuses them and only refits the Y ranges.
#[derive(Clone, Debug, PartialEq)]
struct DataKey {
    /// [`Document::changes`]: it moves with every change of the document's content (the
    /// application keeps the interface in step by the same counter). Opening or closing a
    /// project forgets the key.
    doc_changes: u64,
    trial: usize,
    span: FrameSpan,
    frames: FrameAxis,
    /// The panel's choices the computation reads.
    choices: [usize; 9],
    thresholds: [u64; 2],
}

/// The extra plot's positions on the frame axis and its Y range, computed when the data or
/// the axis ("Display time") changes.
#[derive(Clone, Debug)]
struct ExtraDrawn {
    /// `posMultiplier` and `posOffset` the positions were computed with.
    axis: (u64, u64),
    pos: Vec<f64>,
    /// Smallest and largest value of every column.
    range: (f64, f64),
    /// [`PlotSpec::generation`] of these positions and the data.
    generation: u64,
}

/// The "values per frame" prompt of the extra plot.
#[derive(Clone, Debug)]
struct RatePrompt {
    path: std::path::PathBuf,
    rate: f64,
}

/// The Plot panel.
pub struct PlotPanel {
    plot_type: usize,
    camera: usize,
    marker1: usize,
    marker2: usize,
    body: usize,
    trans_type: usize,
    trans_part: usize,
    rb_error: usize,
    threshold: f64,
    threshold_rb: f64,
    rb_decimals: usize,
    display_status: bool,
    display_time: bool,
    items: Items,
    view: PlotView,
    extra_view: PlotView,
    selection: FrameSelection,
    redraw: bool,
    fit_x: bool,
    cache: Option<Cache>,
    extra: Option<Extra>,
    prompt: Option<RatePrompt>,
    seen_marker: Option<usize>,
    seen_body: Option<usize>,
    last_cursor_x: Option<f64>,
    last_axis_rect: egui::Rect,
    last_events: Vec<PlotEvent>,
    /// Width of each fixed control of the row when last drawn, by its text.
    row_measured: std::collections::HashMap<&'static str, f32>,
    /// The row as last laid out: per line, the combo widths and whether it scrolls.
    row_lines: Vec<(Vec<f32>, bool)>,
    /// The project may be changed now (no tracking run holds it): "Update" and Delete.
    editable: bool,
    /// How many times the series were computed ([`Self::compute_count`]).
    computes: u64,
    /// What `cache` was computed from.
    data_key: Option<DataKey>,
    /// Counts the computations, as the plot's [`PlotSpec::generation`].
    generation: u64,
    /// The extra plot's positions and range for its data and axis.
    extra_drawn: Option<ExtraDrawn>,
    /// Counts the changes of `extra_drawn`.
    extra_generation: u64,
}

impl Default for PlotPanel {
    /// The form's initial state (`PlotWindow::PlotWindow`: plot type 0, every combo at its
    /// first item, thresholds 0, both check boxes off). For screenshots of the other plot
    /// types, `XMALAB_PLOT` may hold `type[,status][,time][,trans=N][,part=N][,2d]`
    /// (type 0 to 5); it is read only here.
    fn default() -> Self {
        let mut p = Self {
            plot_type: 0,
            camera: 0,
            marker1: 0,
            marker2: 0,
            body: 0,
            trans_type: 0,
            trans_part: 0,
            rb_error: 0,
            threshold: 0.0,
            threshold_rb: 0.0,
            rb_decimals: 3,
            display_status: false,
            display_time: false,
            items: Items::default(),
            view: PlotView::default(),
            extra_view: PlotView::default(),
            selection: FrameSelection::new(),
            redraw: true,
            fit_x: true,
            cache: None,
            extra: None,
            prompt: None,
            seen_marker: None,
            seen_body: None,
            last_cursor_x: None,
            last_axis_rect: egui::Rect::NOTHING,
            last_events: Vec::new(),
            row_measured: std::collections::HashMap::new(),
            row_lines: Vec::new(),
            editable: true,
            computes: 0,
            data_key: None,
            generation: 0,
            extra_drawn: None,
            extra_generation: 0,
        };
        if let Ok(spec) = std::env::var("XMALAB_PLOT") {
            p.apply_env(&spec);
        }
        p
    }
}

impl PlotPanel {
    fn apply_env(&mut self, spec: &str) {
        for (i, part) in spec.split(',').map(str::trim).enumerate() {
            let num = |s: &str| s.parse::<usize>().ok();
            match part {
                "status" => self.display_status = true,
                "time" => self.display_time = true,
                "2d" => {
                    self.rb_error = 1;
                    self.rb_decimals = 2;
                }
                _ => {
                    if i == 0
                        && let Some(t) = num(part)
                    {
                        self.plot_type = t.min(5);
                    } else if let Some(v) = part.strip_prefix("trans=").and_then(num) {
                        self.trans_type = v.min(2);
                    } else if let Some(v) = part.strip_prefix("part=").and_then(num) {
                        self.trans_part = v.min(8);
                    } else if let Some(v) = part.strip_prefix("marker=").and_then(num) {
                        self.marker1 = v.saturating_sub(1);
                    } else if let Some(v) = part.strip_prefix("marker2=").and_then(num) {
                        self.marker2 = v.saturating_sub(1);
                    } else if let Some(v) = part.strip_prefix("body=").and_then(num) {
                        self.body = v.saturating_sub(1);
                    } else if let Some(v) = part.strip_prefix("camera=").and_then(num) {
                        self.camera = v;
                    }
                }
            }
        }
    }

    // ---- read access for tests ----

    /// `comboBoxPlotType`'s index.
    #[must_use]
    pub fn plot_type(&self) -> usize {
        self.plot_type
    }

    /// `comboBoxMarker1`, `comboBoxMarker2`, `comboBoxCamera` and `comboBoxRigidBody` indices.
    #[must_use]
    pub fn indices(&self) -> (usize, usize, usize, usize) {
        (self.marker1, self.marker2, self.camera, self.body)
    }

    /// The combo-box contents.
    #[must_use]
    pub fn items(&self) -> &Items {
        &self.items
    }

    /// The axis ranges.
    #[must_use]
    pub fn view(&self) -> &PlotView {
        &self.view
    }

    /// The selected frame range.
    #[must_use]
    pub fn selection(&self) -> &FrameSelection {
        &self.selection
    }

    /// The x of the current-frame line as last drawn.
    #[must_use]
    pub fn cursor_x(&self) -> Option<f64> {
        self.last_cursor_x
    }

    /// The plot's data area as last drawn, in screen points.
    #[must_use]
    pub fn axis_rect(&self) -> egui::Rect {
        self.last_axis_rect
    }

    /// The axis labels as last computed: x, left, right (`None`: right axis hidden).
    #[must_use]
    pub fn labels(&self) -> Option<(&str, &str, Option<&str>)> {
        self.cache.as_ref().map(|c| {
            (
                c.x_label.as_str(),
                c.left_label.as_str(),
                c.right_label.as_deref(),
            )
        })
    }

    /// Number of series in the last redraw.
    #[must_use]
    pub fn series_count(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.series.len())
    }

    /// The frame axis of the last redraw.
    #[must_use]
    pub fn frame_axis(&self) -> Option<FrameAxis> {
        self.cache.as_ref().map(|c| c.frames)
    }

    /// Series `i` of the last redraw as the widget draws it: the 0-based frame of its first
    /// value for a one-value-per-frame series (`None` for explicit x and y), and the plot x
    /// and y of every sample.
    #[must_use]
    pub fn series_samples(&self, i: usize) -> Option<SeriesSamples> {
        let c = self.cache.as_ref()?;
        let s = series_of(c.series.get(i)?, c.frames.first);
        let first = match s.data {
            SeriesData::Frames { first, .. } => Some(first),
            SeriesData::Xy { .. } => None,
        };
        let xy = (0..s.data.len())
            .map(|k| (s.data.x(k, &c.frames), s.data.y(k)))
            .collect();
        Some((first, xy))
    }

    /// Number of band rows (status, interpolation, events) in the last redraw.
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.bands.len())
    }

    /// The control row as last laid out: per line, the combo boxes' widths and whether the
    /// line scrolls.
    #[must_use]
    pub fn row_lines(&self) -> &[(Vec<f32>, bool)] {
        &self.row_lines
    }

    /// How many times the plotted series have been computed since the panel was made (for
    /// tests and measurements).
    #[must_use]
    pub fn compute_count(&self) -> u64 {
        self.computes
    }

    /// The widget events of the last frame.
    #[must_use]
    pub fn last_events(&self) -> &[PlotEvent] {
        &self.last_events
    }

    /// The extra plot's data, when shown.
    #[must_use]
    pub fn extra(&self) -> Option<&Extra> {
        self.extra.as_ref()
    }

    /// Shows the extra plot with data read from a CSV text, `rate` values per frame (what
    /// "+" does after its two dialogs).
    pub fn set_extra(&mut self, text: &str, rate: f64) {
        let (columns, rows) = plots::read_extra_columns(text);
        self.extra = Some(Extra {
            columns,
            pos: (0..rows).map(|r| rate * r as f64).collect(),
        });
        self.extra_drawn = None;
        self.redraw = true;
    }

    /// Asks for a redraw (the C++ `draw()`), for example after the project changed.
    pub fn redraw(&mut self) {
        self.redraw = true;
    }

    // ---- C++ slots ----

    /// Ported from ui/PlotWindow.cpp, PlotWindow::updateMarkers: fills the combo boxes and
    /// keeps each index when it is still in range (else the first item, as `QComboBox` does
    /// after `clear()` and `addItem`). Fixed: `comboBoxMarker2` keeps its own index.
    fn update_markers(&mut self, doc: &Document, trial: &Trial) {
        let items = Items {
            cameras: plots::camera_items(doc),
            markers: plots::marker_items(trial),
            marker2: plots::marker2_items(trial),
            bodies: plots::rigid_body_items(trial),
        };
        let keep = |i: usize, n: usize| if i < n { i } else { 0 };
        self.marker1 = keep(self.marker1, items.markers.len());
        self.marker2 = keep(self.marker2, items.marker2.len());
        self.camera = keep(self.camera, items.cameras.len());
        self.body = keep(self.body, items.bodies.len());
        self.items = items;
    }

    /// Ported from ui/PlotWindow.cpp, PlotWindow::activePointChanged.
    fn active_point_changed(&mut self, idx: Option<usize>) {
        self.selection.forget_shift();
        if let Some(i) = idx
            && self.plot_type <= 3
        {
            self.marker1 = i;
            self.redraw = true;
        }
    }

    /// Ported from ui/PlotWindow.cpp, PlotWindow::activeRigidBodyChanged.
    fn active_body_changed(&mut self, idx: Option<usize>) {
        self.selection.forget_shift();
        if let Some(i) = idx
            && self.plot_type >= 4
        {
            self.body = i;
            self.redraw = true;
        }
    }

    /// `activeTrialChanged` and `workspaceChanged`: the combo boxes, then the active marker
    /// or body (both handlers; the C++ trial handler checks only plot type 4, which leaves
    /// "Rigid Body Error" on the previous trial's body index).
    fn trial_or_workspace_changed(&mut self, cx: &Cx<'_>) {
        self.selection.forget_shift();
        let (Some(doc), Some(t)) = (cx.doc, cx.state.active_trial) else {
            return;
        };
        if t >= doc.project().trials.len() {
            return;
        }
        self.update_markers(doc, doc.trial(t));
        let view = cx.state.trial_view().copied();
        if self.plot_type >= 4 {
            self.active_body_changed(view.and_then(|v| v.active_body));
        } else {
            self.active_point_changed(view.and_then(|v| v.active_marker));
        }
        self.seen_marker = view.and_then(|v| v.active_marker);
        self.seen_body = view.and_then(|v| v.active_body);
        self.fit_x = true;
        self.redraw = true;
    }

    /// Ported from ui/PlotWindow.cpp, PlotWindow::goToNextAboveError and goToPrevAboveError
    /// (keys 4 and Ctrl+4): the next or previous frame whose reprojection error ("Reprojection
    /// Error", active marker) or rigid-body error ("Rigid Body Error") is above the threshold.
    fn go_to_above_error(&mut self, cx: &mut Cx<'_>, forward: bool) {
        let (Some(doc), Some(t)) = (cx.doc, cx.state.active_trial) else {
            return;
        };
        let Some(view) = cx.state.trial_view().copied() else {
            return;
        };
        let trial = doc.trial(t);
        let span = FrameSpan::from_one_based(view.start_frame, view.end_frame);
        let from = view.active_frame;
        let found = if self.plot_type == 3 {
            let Some(m) = view.active_marker.and_then(|m| trial.markers.get(m)) else {
                return;
            };
            let cam = self.camera.checked_sub(1);
            plots::find_above(span, from, forward, self.threshold, |f| {
                plots::marker_frame_error(m, cam, f)
            })
        } else if self.plot_type == 5 {
            let Some(rb) = trial.rigid_bodies.get(self.body) else {
                return;
            };
            let choice = PoseChoice::from_index(self.trans_type);
            let dim = self.error_dim();
            plots::find_above(span, from, forward, self.threshold_rb, |f| {
                plots::body_frame_error(rb, choice, dim, f)
            })
        } else {
            None
        };
        if let Some(f) = found {
            cx.change(|s| s.change_active_frame_trial(Some(f), false));
        }
    }

    // ---- choices, for tests (the combo boxes and spin boxes set them in the form) ----

    /// Sets `comboBoxPlotType` (0 to 5), as choosing it in the form does.
    pub fn set_plot_type(&mut self, t: usize) {
        self.plot_type = t.min(5);
        self.redraw = true;
    }

    /// Sets `comboBoxMarker1`, `comboBoxMarker2`, `comboBoxCamera` (0 "All Cameras") and
    /// `comboBoxRigidBody`.
    pub fn set_indices(&mut self, marker1: usize, marker2: usize, camera: usize, body: usize) {
        self.marker1 = marker1;
        self.marker2 = marker2;
        self.camera = camera;
        self.body = body;
        self.redraw = true;
    }

    /// Sets `doubleSpinBoxError` and `doubleSpinBoxErrorRB`.
    pub fn set_thresholds(&mut self, marker: f64, body: f64) {
        self.threshold = marker;
        self.threshold_rb = body;
        self.redraw = true;
    }

    /// Sets `comboBoxRigidBodyTransType` (0 to 2) and `comboBoxRigidBodyError` (0 3D, 1 2D).
    pub fn set_body_error(&mut self, trans_type: usize, rb_error: usize) {
        self.trans_type = trans_type.min(2);
        self.rb_error = rb_error.min(1);
        self.redraw = true;
    }

    /// The selected range, to set it as Ctrl+drag or Shift does.
    pub fn selection_mut(&mut self) -> &mut FrameSelection {
        &mut self.selection
    }

    // ---- editing (E44 to E48, E51, E52) ----

    /// The active trial, its view state, and the selection ordered and clamped to the
    /// trial's start and end frames, as every range action takes it. `None` outside Marker
    /// tracking (where the plot is not shown), without a trial, or when the selection lies
    /// wholly outside the start and end frames (the C++ then asks about a reversed range
    /// and changes nothing).
    fn edit_range(&self, cx: &Cx<'_>) -> Option<(usize, crate::state::TrialView, FrameRange)> {
        if cx.state.workspace != Workspace::MarkerTracking {
            return None;
        }
        let doc = cx.doc?;
        let t = cx.state.active_trial?;
        doc.project().trials.get(t)?;
        let view = *cx.state.trial_view()?;
        let axis = FrameAxis {
            first: view.start_frame.saturating_sub(1),
            last: view.end_frame.saturating_sub(1),
            fps: None,
        };
        let (lo, hi) = self.selection.clamped(&axis);
        if lo < 0 || lo > hi {
            return None;
        }
        Some((
            t,
            view,
            FrameRange {
                first: lo as usize,
                last: hi as usize,
            },
        ))
    }

    /// E44 (Delete, Backspace) and E45 (Ctrl+U, `untrackable`): the question and the edit
    /// for the selected range, per plot type:
    ///
    /// - 0, 3: the plot's marker, in the chosen camera ("All Cameras": every camera);
    /// - 1: the plot's marker, every camera;
    /// - 2: both markers (the second only when it is a marker, not a virtual point);
    /// - 4, 5: every marker of the plot's rigid body, every camera, then "Update" (E48), as
    ///   one undo step.
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::deleteData (`:649-699`) and
    /// PlotWindow::setUntrackable (`:3067-3117`). Fixed (cpp-editing D19): with no marker or
    /// rigid body to act on, or an index the project no longer has, nothing is asked and
    /// nothing changes (the C++ asks, then reads outside its lists); for plot type 5 the
    /// question names "All Cameras" (the C++ leaves the camera text empty).
    #[must_use]
    pub fn range_delete_ask(&self, cx: &Cx<'_>, untrackable: bool) -> Option<Ask> {
        let (t, _, frames) = self.edit_range(cx)?;
        let trial = cx.doc?.trial(t);
        let what = if untrackable {
            "delete your data and set it to untrackable"
        } else {
            "delete your data"
        };
        let (a, b) = (frames.first + 1, frames.last + 1);
        let reset = |markers: Vec<usize>, cameras: CameraSel| Edit::ResetFrames {
            trial: t,
            markers,
            cameras,
            frames,
            untrackable,
        };
        match self.plot_type {
            0..=3 => {
                let m = self.marker1;
                let m_text = self.items.markers.get(m)?;
                if m >= trial.markers.len() {
                    return None;
                }
                let (cameras, cam_text) = if matches!(self.plot_type, 0 | 3) {
                    let text = self.items.cameras.get(self.camera)?.clone();
                    match self.camera.checked_sub(1) {
                        None => (CameraSel::All, text),
                        Some(c) if c < cx.doc?.project().cameras.len() => (CameraSel::One(c), text),
                        Some(_) => return None,
                    }
                } else {
                    (CameraSel::All, "All Cameras".to_owned())
                };
                let mut markers = vec![m];
                let message = if self.plot_type == 2 {
                    let (m2_text, target) = self.items.marker2.get(self.marker2)?;
                    if let DistanceTarget::Marker(m2) = *target
                        && m2 < trial.markers.len()
                        && m2 != m
                    {
                        markers.push(m2);
                    }
                    format!(
                        "Are you sure you want to {what} for {cam_text} for Marker {m_text} and Marker {m2_text} from Frame {a} to {b}"
                    )
                } else {
                    format!(
                        "Are you sure you want to {what} for {cam_text} for Marker {m_text} from Frame {a} to {b}"
                    )
                };
                Some(Ask::confirm(
                    message,
                    Then::Request(Box::new(EditRequest::Apply(reset(markers, cameras)))),
                ))
            }
            _ => {
                let rb = trial.rigid_bodies.get(self.body)?;
                let body_text = self.items.bodies.get(self.body)?;
                let message = if untrackable {
                    format!(
                        "Are you sure you want to {what} for all markers of Rigid Body {body_text} from Frame {a} to {b}"
                    )
                } else {
                    format!(
                        "Are you sure you want to {what} for All Cameras for all marker of Rigid Body {body_text} from Frame {a} to {b}"
                    )
                };
                let first = reset(rb.marker_indices.clone(), CameraSel::All);
                let label = first.label().to_owned();
                Some(Ask::confirm(
                    message,
                    Then::Request(Box::new(EditRequest::ApplyAll {
                        label,
                        edits: vec![first, Edit::RecomputeBodies { trial: t }],
                    })),
                ))
            }
        }
    }

    /// E47 (Shift+Delete, Shift+Backspace): the question and the edit that delete every
    /// point whose error is above the threshold, over every frame of the trial (not only the
    /// start to end frames, as in the C++). Plot type 3: the plotted marker, in the chosen
    /// camera or the mean over cameras; type 5: every marker of the plotted body, by the
    /// body's error as the plot shows it (unfiltered, filtered or the larger; 3D or 2D).
    /// Other types: nothing.
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::deleteAllAboveError (`:1369-1424`). The
    /// C++ tests and resets the active marker while its question names the plot's marker;
    /// the plot's marker (the one plotted) is used for both here (the two are the same
    /// unless no marker is active, when the C++ does nothing).
    ///
    /// `ErrorTarget::Body` carries the 2D or 3D choice in `kind` (3 to 5 for 2D) until it is
    /// given its own field; this call changes then.
    #[must_use]
    pub fn above_error_ask(&self, cx: &Cx<'_>) -> Option<Ask> {
        if cx.state.workspace != Workspace::MarkerTracking {
            return None;
        }
        let doc = cx.doc?;
        let t = cx.state.active_trial?;
        let trial = doc.project().trials.get(t)?;
        let (target, message) = match self.plot_type {
            3 => {
                let m = self.marker1;
                let m_text = self.items.markers.get(m)?;
                if m >= trial.markers.len() {
                    return None;
                }
                let cam_text = self.items.cameras.get(self.camera)?;
                let cameras = match self.camera.checked_sub(1) {
                    None => CameraSel::All,
                    Some(c) if c < doc.project().cameras.len() => CameraSel::One(c),
                    Some(_) => return None,
                };
                (
                    ErrorTarget::Marker { marker: m, cameras },
                    format!(
                        "Are you sure you want to delete your data for {cam_text} for Marker {m_text} for all frames with reprojection error higher than {}",
                        qt_number(self.threshold)
                    ),
                )
            }
            5 => {
                trial.rigid_bodies.get(self.body)?;
                let body_text = self.items.bodies.get(self.body)?;
                (
                    ErrorTarget::Body {
                        body: self.body,
                        kind: self.trans_type,
                        error: if self.rb_error == 0 {
                            xma_engine::edit::BodyErrorKind::Error3d
                        } else {
                            xma_engine::edit::BodyErrorKind::Error2d
                        },
                    },
                    format!(
                        "Are you sure you want to delete your data for RigidBody {body_text} for all frames with error higher than {}",
                        qt_number(self.threshold_rb)
                    ),
                )
            }
            _ => return None,
        };
        let threshold = if self.plot_type == 3 {
            self.threshold
        } else {
            self.threshold_rb
        };
        Some(Ask::confirm(
            message,
            Then::Request(Box::new(EditRequest::Apply(Edit::ResetAboveError {
                trial: t,
                target,
                threshold,
            }))),
        ))
    }

    /// E46 (Ctrl+I, the Toolbox's "Set Interpolation"): what to ask. Outside plot types 0
    /// and 1 first whether to change to the 2D position plot (`confirmed`: the user said
    /// OK, so the plot changes and the action goes on); then, for the active marker, "No
    /// Range Selected" when the selection is one frame, else the method.
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::setInterpolation (`:3007-3065`).
    fn set_interpolation(&mut self, cx: &mut Cx<'_>, confirmed: bool) {
        if cx.state.workspace != Workspace::MarkerTracking || cx.doc.is_none() {
            return;
        }
        if self.plot_type > 1 {
            if !confirmed {
                cx.request(EditRequest::Ask(Box::new(Ask::confirm(
                    "You have to be in either the 2D positions or 3D positions plot to modify Interpolation. Do you want to change to the 2D position plot and continue with the active marker?",
                    Then::Raise(Action::Confirmed(Box::new(Action::SetInterpolationRange))),
                ))));
                return;
            }
            // dock->comboBoxPlotType->setCurrentIndex(0).
            self.plot_type = 0;
            let m = cx.state.trial_view().and_then(|v| v.active_marker);
            self.active_point_changed(m);
            self.redraw = true;
        }
        let Some(ask) = self.interpolation_ask(cx) else {
            return;
        };
        cx.request(EditRequest::Ask(Box::new(ask)));
    }

    /// The question of E46 once the plot type is 0 or 1: `None` without an active marker.
    #[must_use]
    pub fn interpolation_ask(&self, cx: &Cx<'_>) -> Option<Ask> {
        let doc = cx.doc?;
        let t = cx.state.active_trial?;
        let trial = doc.project().trials.get(t)?;
        let marker = cx.state.trial_view()?.active_marker?;
        if marker >= trial.markers.len() {
            return None;
        }
        let range = self.edit_range(cx).map(|r| r.2);
        let Some(frames) = range.filter(|r| r.first < r.last) else {
            return Some(Ask {
                title: "No Range Selected".to_owned(),
                kind: AskKind::Information {
                    text: "Please select a frame range in the plot (Shift+Drag) before setting interpolation.".to_owned(),
                },
                then: Then::Nothing,
            });
        };
        Some(Ask {
            title: "Choose interpolation method".to_owned(),
            kind: AskKind::Item {
                label: "interpolation".to_owned(),
                items: INTERPOLATION_METHODS
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect(),
                current: 0,
            },
            then: Then::SetInterpolation {
                trial: t,
                marker,
                frames,
            },
        })
    }

    /// E51 (O, "Mark frames") and E52 (Ctrl+O, "Unmark frames"): the selected range (the
    /// active frame when nothing is selected) set or cleared in every event that is drawn.
    /// `None` when no event is drawn (nothing would change).
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::setEventOn (`:3119-3136`) and
    /// PlotWindow::setEventOff (`:3138-3155`).
    #[must_use]
    pub fn event_request(&self, cx: &Cx<'_>, on: bool) -> Option<EditRequest> {
        let (t, _, frames) = self.edit_range(cx)?;
        if !cx.doc?.trial(t).events.iter().any(|e| e.draw) {
            return None;
        }
        Some(EditRequest::Apply(Edit::SetEventFrames {
            trial: t,
            frames,
            on,
        }))
    }

    /// E48 "Update": recompute and filter every rigid body of the trial.
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::on_pushButtonUpdate_clicked (`:2813-2828`).
    fn update_bodies(cx: &mut Cx<'_>) {
        if let Some(t) = cx.state.active_trial
            && cx.doc.is_some_and(|d| t < d.project().trials.len())
        {
            cx.request(EditRequest::Apply(Edit::RecomputeBodies { trial: t }));
        }
    }

    fn error_dim(&self) -> ErrorDim {
        if self.rb_error == 0 {
            ErrorDim::D3
        } else {
            ErrorDim::D2
        }
    }

    // ---- computing a redraw ----

    fn status_layout(
        &self,
        n_cameras: usize,
        trial: &Trial,
        marker: usize,
    ) -> Option<StatusLayout> {
        let m = trial.markers.get(marker)?;
        self.display_status.then(|| StatusLayout {
            n_cameras,
            has_interpolation: m.has_interpolation(),
        })
    }

    /// Status, interpolation and event rows. Ported from ui/PlotWindow.cpp,
    /// PlotWindow::drawStatus and drawEvents; `marker` is `None` for `drawStatus(-1)`.
    fn bands(
        &self,
        trial: &Trial,
        n_cameras: usize,
        span: FrameSpan,
        marker: Option<usize>,
        y_axis: (f64, f64),
    ) -> Vec<OwnedBand> {
        let mut out = Vec::new();
        let status = marker.and_then(|m| self.status_layout(n_cameras, trial, m));
        if let (Some(m), Some(st)) = (marker, status) {
            let rows = ranging::status_rows(y_axis, st);
            for (c, y) in plots::status_rows(trial, m, n_cameras, span)
                .into_iter()
                .zip(rows.cameras)
            {
                out.push(OwnedBand {
                    values: c.into_iter().map(status_index).collect(),
                    colors: BandColors::Status,
                    y,
                    outline: true,
                });
            }
            if let Some(y) = rows.interpolation {
                out.push(OwnedBand {
                    values: plots::interpolation_row(trial, m, span)
                        .into_iter()
                        .map(interpolation_index)
                        .collect(),
                    colors: BandColors::Interpolation,
                    y,
                    outline: true,
                });
            }
        }
        let events = plots::event_rows(trial, span);
        let ys = ranging::event_rows(y_axis, events.len(), status);
        for ((e, on), y) in events.into_iter().zip(ys) {
            out.push(OwnedBand {
                values: on.into_iter().map(u8::from).collect(),
                colors: BandColors::Event([
                    Color32::TRANSPARENT,
                    Color32::from_rgba_unmultiplied(e.color.r, e.color.g, e.color.b, 50),
                ]),
                y,
                outline: false,
            });
        }
        out
    }

    fn compute(&self, doc: &Document, trial: &Trial, span: FrameSpan, frames: FrameAxis) -> Cache {
        let base = if frames.is_time() {
            "Time in seconds"
        } else {
            "Frame"
        };
        let mut c = Cache {
            frames,
            x_label: base.to_owned(),
            left_label: String::new(),
            right_label: None,
            series: Vec::new(),
            bands: Vec::new(),
            span: None,
            left: None,
            right: None,
        };
        let n_cams = doc.project().cameras.len();
        let first_x = frames.x_of(span.first);
        match self.plot_type {
            0 => self.compute_2d(&mut c, trial, n_cams, span),
            1 => self.compute_3d(&mut c, trial, n_cams, span),
            2 => self.compute_distance(&mut c, trial, n_cams, span, base, first_x),
            3 => self.compute_reprojection(&mut c, trial, n_cams, span, base),
            4 => self.compute_rigid_body(&mut c, trial, n_cams, span, base),
            _ => self.compute_rigid_body_error(&mut c, trial, n_cams, span, base),
        }
        c
    }

    fn set_fit(&self, c: &mut Cache, left: YFit, right: Option<YFit>) {
        c.left = Some(left.axis);
        c.right = right.map(|r| r.axis);
        c.span = Some(left.span);
    }

    /// Ported from ui/PlotWindow.cpp, PlotWindow::plot2D.
    fn compute_2d(&self, c: &mut Cache, trial: &Trial, n_cams: usize, span: FrameSpan) {
        c.left_label = "X - Position".into();
        c.right_label = Some("Y - Position".into());
        let m = self.marker1;
        if m >= trial.markers.len() {
            return;
        }
        let cams: Vec<usize> = match self.camera.checked_sub(1) {
            None => (0..n_cams).collect(),
            Some(cam) => vec![cam],
        };
        let (mut mx, mut my) = (MinMax::new(), MinMax::new());
        for cam in cams {
            let color = match cam {
                1 => qt::BLUE,
                2 => qt::GREEN,
                3 => qt::YELLOW,
                4 => qt::CYAN,
                _ => qt::RED,
            };
            let p = plots::positions_2d(trial, m, cam, span);
            mx = mx.extend(p.x.iter().copied());
            my = my.extend(p.y.iter().copied());
            c.series.push(
                OwnedSeries::frames(p.x, Pen::solid(color))
                    .line(LineStyle::None)
                    .scatter(ScatterShape::Cross, 2.0),
            );
            c.series.push(
                OwnedSeries::frames(p.y, Pen::solid(color))
                    .line(LineStyle::None)
                    .scatter(ScatterShape::Circle, 2.0)
                    .axis(YAxis::Right),
            );
        }
        let st = self.status_layout(n_cams, trial, m);
        let (left, right) = ranging::fit_2d(mx, my, st);
        self.set_fit(c, left, Some(right));
        c.bands = self.bands(trial, n_cams, span, Some(m), left.axis);
    }

    /// Ported from ui/PlotWindow.cpp, PlotWindow::plot3D.
    fn compute_3d(&self, c: &mut Cache, trial: &Trial, n_cams: usize, span: FrameSpan) {
        c.left_label = "Position".into();
        let m = self.marker1;
        if m >= trial.markers.len() {
            return;
        }
        let p = plots::positions_3d(trial, m, span);
        let mut mm = MinMax::new();
        for (values, (color, shape)) in p.into_iter().zip([
            (qt::RED, ScatterShape::Cross),
            (qt::GREEN, ScatterShape::Circle),
            (qt::BLUE, ScatterShape::Circle),
        ]) {
            mm = mm.extend(values.iter().copied());
            c.series.push(
                OwnedSeries::frames(values, Pen::solid(color))
                    .line(LineStyle::None)
                    .scatter(shape, 2.0),
            );
        }
        let fit = ranging::fit_positions(mm, 0.2, self.status_layout(n_cams, trial, m));
        self.set_fit(c, fit, None);
        c.bands = self.bands(trial, n_cams, span, Some(m), fit.axis);
    }

    /// Ported from ui/PlotWindow.cpp, PlotWindow::plotDistance. A frame without both points
    /// is plotted at 0, as in the C++.
    fn compute_distance(
        &self,
        c: &mut Cache,
        trial: &Trial,
        n_cams: usize,
        span: FrameSpan,
        base: &str,
        first_x: f64,
    ) {
        let target = self.items.marker2.get(self.marker2);
        let (mean, sd) = if self.marker1 < trial.markers.len()
            && let Some((_, target)) = target
        {
            let d = plots::distance(trial, self.marker1, *target, span);
            let mm = MinMax::new().extend(d.values.iter().copied());
            let values: Vec<f64> = d
                .values
                .iter()
                .map(|v| if v.is_nan() { 0.0 } else { *v })
                .collect();
            c.series.push(
                OwnedSeries::frames(values, Pen::solid(qt::BLUE)).line(LineStyle::StepCenter),
            );
            c.series.push(OwnedSeries::xy(
                vec![first_x, c.frames.x_of(span.last)],
                vec![d.stats.mean; 2],
                Pen::dashed(qt::DARK_GRAY, Dash::Dash),
            ));
            let fit = ranging::fit_centered(mm);
            self.set_fit(c, fit, None);
            c.bands = self.bands(trial, n_cams, span, None, fit.axis);
            (d.stats.mean, d.stats.sd)
        } else {
            (0.0, 0.0)
        };
        c.x_label = format!(
            "{base}\nMean intermarker distance: {} +/- {}",
            qt_number(mean),
            qt_number(sd)
        );
        c.left_label = match target {
            Some((_, DistanceTarget::Marker(j))) => {
                format!("Distance Marker {} to Marker {}", self.marker1 + 1, j + 1)
            }
            Some((label, _)) => format!("Distance Marker {} to Marker {label}", self.marker1 + 1),
            None => format!(
                "Distance Marker {} to Marker {}",
                self.marker1 + 1,
                self.marker2 + 1
            ),
        };
    }

    /// Ported from ui/PlotWindow.cpp, PlotWindow::plotReprojectionError. A frame without an
    /// error is plotted at -2 (below the axis), as in the C++.
    fn compute_reprojection(
        &self,
        c: &mut Cache,
        trial: &Trial,
        n_cams: usize,
        span: FrameSpan,
        base: &str,
    ) {
        c.left_label = "Error in pixel".into();
        let m = self.marker1;
        let (mean, sd) = if m < trial.markers.len() {
            let r = plots::reprojection_error(trial, m, self.camera.checked_sub(1), span);
            let mm = MinMax::new().extend(r.values.iter().copied());
            let values: Vec<f64> = r
                .values
                .iter()
                .map(|v| if v.is_nan() { -2.0 } else { *v })
                .collect();
            c.series.push(
                OwnedSeries::frames(values, Pen::solid(qt::BLUE)).line(LineStyle::StepCenter),
            );
            c.series.push(OwnedSeries::xy(
                vec![c.frames.x_of(span.first), c.frames.x_of(span.last + 2)],
                vec![self.threshold; 2],
                Pen::solid(qt::RED),
            ));
            let fit = ranging::fit_reprojection(mm.max, self.status_layout(n_cams, trial, m));
            self.set_fit(c, fit, None);
            c.bands = self.bands(trial, n_cams, span, Some(m), fit.axis);
            (r.stats.mean, r.stats.sd)
        } else {
            (0.0, 0.0)
        };
        // Fixed: the C++ omits ": " in the time variant.
        c.x_label = format!(
            "{base}\nMean reprojection error: {} +/- {}",
            qt_number(mean),
            qt_number(sd)
        );
    }

    /// Ported from ui/PlotWindow.cpp, PlotWindow::plotRigidBody. Fixed: "Full" labels the
    /// left axis (angles) "Rotationangle" and the right (translation) "Translation".
    fn compute_rigid_body(
        &self,
        c: &mut Cache,
        trial: &Trial,
        n_cams: usize,
        span: FrameSpan,
        base: &str,
    ) {
        let b = self.body;
        if b >= trial.rigid_bodies.len() {
            // The C++ sets no label without a body; keep the last ones.
            if let Some(prev) = &self.cache {
                c.x_label = prev.x_label.clone();
                c.left_label = prev.left_label.clone();
                c.right_label = prev.right_label.clone();
            }
            return;
        }
        let part = self.trans_part;
        let choice = PoseChoice::from_index(self.trans_type);
        c.x_label = if self.trans_type > 0 {
            format!(
                "{base}\nCutoff Frequency: {}Hz",
                qt_number(plots::cutoff(trial, b))
            )
        } else {
            base.to_owned()
        };
        let parts = match part {
            0 => RigidBodyParts::Full,
            1 | 3 | 4 | 5 => RigidBodyParts::Rotation,
            _ => RigidBodyParts::Translation,
        };
        match parts {
            RigidBodyParts::Full => {
                c.left_label = "Rotationangle".into();
                c.right_label = Some("Translation".into());
            }
            RigidBodyParts::Rotation => c.left_label = "Rotationangle".into(),
            RigidBodyParts::Translation => c.left_label = "Translation".into(),
        }
        let mut unf = choice
            .unfiltered()
            .then(|| plots::transformation(trial, b, false, span));
        let mut fil = choice
            .filtered()
            .then(|| plots::transformation(trial, b, true, span));
        let (mut rot, mut trans) = (MinMax::new(), MinMax::new());
        let mut k = 0;
        for (is_rot, parts_for) in [(true, [1usize, 3, 4, 5]), (false, [2, 6, 7, 8])] {
            for z in 0..3 {
                if !(part == 0 || part == parts_for[0] || part == parts_for[z + 1]) {
                    continue;
                }
                // Filtered first, as the C++ loops `y = 0` (filtered) then `y = 1`.
                for (filtered, t) in [(true, &mut fil), (false, &mut unf)] {
                    let Some(t) = t else { continue };
                    // Each column is plotted once: moved out, not copied.
                    let values = std::mem::take(if is_rot {
                        &mut t.angles[z]
                    } else {
                        &mut t.translation[z]
                    });
                    if is_rot {
                        rot = rot.extend(values.iter().copied());
                    } else {
                        trans = trans.extend(values.iter().copied());
                    }
                    c.series.push(
                        OwnedSeries::frames(values, Pen::solid(qt::global_color(k + 7)))
                            .line(LineStyle::None)
                            .scatter(
                                if filtered {
                                    ScatterShape::Cross
                                } else {
                                    ScatterShape::Circle
                                },
                                2.0,
                            )
                            .axis(if part == 0 && !is_rot {
                                YAxis::Right
                            } else {
                                YAxis::Left
                            }),
                    );
                    k += 1;
                }
            }
        }
        let (left, right) = ranging::fit_rigid_body(rot, trans, parts);
        self.set_fit(c, left, right);
        c.bands = self.bands(trial, n_cams, span, None, left.axis);
    }

    /// Ported from ui/PlotWindow.cpp, PlotWindow::plotRigidBodyError. Fixed: "Both" draws
    /// both series with their error bars and the threshold line; the right axis is hidden;
    /// the unfiltered SD runs over the unfiltered frames. A frame without a pose is plotted
    /// at -2 with no error, as in the C++.
    fn compute_rigid_body_error(
        &self,
        c: &mut Cache,
        trial: &Trial,
        n_cams: usize,
        span: FrameSpan,
        base: &str,
    ) {
        let b = self.body;
        let dim = self.error_dim();
        let choice = PoseChoice::from_index(self.trans_type);
        let mut mean_string = String::new();
        let mut cutoff = 0.0;
        if b < trial.rigid_bodies.len() {
            cutoff = plots::cutoff(trial, b);
            c.left_label = match dim {
                ErrorDim::D3 => "3D Error",
                ErrorDim::D2 => "2D Error in pixel",
            }
            .into();
            let mut mm = MinMax::new();
            let mut get = |filtered: bool| {
                let e = plots::body_error(trial, b, filtered, dim, span);
                for (m, s) in e.mean.iter().zip(&e.sd) {
                    if !m.is_nan() {
                        mm.add(m + s);
                        mm.add(m - s);
                    }
                }
                let mean: Vec<f64> = e
                    .mean
                    .iter()
                    .map(|v| if v.is_nan() { -2.0 } else { *v })
                    .collect();
                let sd: Vec<f64> =
                    e.sd.iter()
                        .map(|v| if v.is_nan() { 0.0 } else { *v })
                        .collect();
                (mean, sd, e.stats)
            };
            let unf = choice.unfiltered().then(|| get(false));
            let fil = choice.filtered().then(|| get(true));
            let step = |values: Vec<f64>, color| {
                OwnedSeries::frames(values, Pen::solid(color))
                    .line(LineStyle::StepCenter)
                    .scatter(ScatterShape::Dot, 1.0)
            };
            let bars_only = |values: Vec<f64>, sd: Vec<f64>, color| OwnedSeries {
                line: LineStyle::None,
                ..OwnedSeries::frames(values, Pen::solid(color)).errors(sd, Pen::solid(color))
            };
            // The values and errors are moved into the series; the statistics stay for the
            // label below.
            let unf_stats = unf.as_ref().map(|u| u.2);
            let fil_stats = fil.as_ref().map(|f| f.2);
            match (unf, fil) {
                (Some(u), Some(f)) => {
                    // graph 0 (filtered), graph 1 (unfiltered), errorBars0, errorBars1.
                    c.series.push(step(f.0.clone(), qt::DARK_BLUE));
                    c.series.push(step(u.0.clone(), qt::DARK_RED));
                    c.series.push(bars_only(f.0, f.1, qt::BLUE));
                    c.series.push(bars_only(u.0, u.1, qt::RED));
                }
                (Some(u), None) => c
                    .series
                    .push(step(u.0, qt::DARK_RED).errors(u.1, Pen::solid(qt::RED))),
                (None, Some(f)) => c
                    .series
                    .push(step(f.0, qt::DARK_BLUE).errors(f.1, Pen::solid(qt::BLUE))),
                (None, None) => {}
            }
            let fit = ranging::fit_centered(mm);
            self.set_fit(c, fit, None);
            c.series.push(OwnedSeries::xy(
                vec![c.frames.x_of(span.first), c.frames.x_of(span.last + 2)],
                vec![self.threshold_rb; 2],
                Pen::solid(qt::RED),
            ));
            c.bands = self.bands(trial, n_cams, span, None, fit.axis);
            mean_string.push('\n');
            if let Some(u) = unf_stats {
                mean_string += &format!(
                    " Mean error unfiltered: {} +/- {}",
                    qt_number(u.mean),
                    qt_number(u.sd)
                );
            }
            if let Some(f) = fil_stats {
                mean_string += &format!(
                    " Mean error filtered: {} +/- {}",
                    qt_number(f.mean),
                    qt_number(f.sd)
                );
            }
        } else if let Some(prev) = &self.cache {
            c.left_label = prev.left_label.clone();
        }
        c.x_label = if self.trans_type > 0 {
            format!(
                "{base}\nCutoff Frequency: {}Hz{mean_string}",
                qt_number(cutoff)
            )
        } else {
            format!("{base}{mean_string}")
        };
    }

    /// What a computation of the series reads: the document's content, the trial, its
    /// span, the frame axis and the panel's choices (the combo lists follow from the
    /// document and the trial).
    fn data_key(&self, doc: &Document, t: usize, span: FrameSpan, frames: FrameAxis) -> DataKey {
        DataKey {
            doc_changes: doc.changes(),
            trial: t,
            span,
            frames,
            choices: [
                self.plot_type,
                self.camera,
                self.marker1,
                self.marker2,
                self.body,
                self.trans_type,
                self.trans_part,
                self.rb_error,
                usize::from(self.display_status),
            ],
            thresholds: [self.threshold.to_bits(), self.threshold_rb.to_bits()],
        }
    }

    /// Redraws the plot (`PlotWindow::draw`): recomputes the series and the combo lists when
    /// anything they are computed from changed ([`Self::data_key`]), and refits the Y
    /// ranges in every case, as the C++ does on every redraw.
    fn do_redraw(
        &mut self,
        doc: &Document,
        t: usize,
        trial: &Trial,
        span: FrameSpan,
        frames: FrameAxis,
    ) {
        if self.cache.is_none()
            || self.data_key.as_ref() != Some(&self.data_key(doc, t, span, frames))
        {
            self.update_markers(doc, trial);
            let c = self.compute(doc, trial, span, frames);
            self.computes += 1;
            self.generation += 1;
            self.cache = Some(c);
            // After update_markers: the indices it kept are what was computed.
            self.data_key = Some(self.data_key(doc, t, span, frames));
        }
        if let Some(c) = &self.cache {
            if let Some(l) = c.left {
                self.view.left = ranging::set_range(self.view.left, l);
            }
            if let Some(r) = c.right {
                self.view.right = ranging::set_range(self.view.right, r);
            }
        }
    }

    // ---- drawing ----

    /// The items of the control row for the current plot type, in the form's order: the
    /// first group is `labelPlotType`, `comboBoxPlotType` and the visible frame's controls;
    /// the second the check boxes and buttons.
    fn row_items(&self, has_speed: bool) -> (Vec<RowItem>, Vec<RowItem>) {
        use RowItem::{Button, Check, Combo, Label, Spin};
        let t = self.plot_type;
        let mut a = vec![Label("Plot type : "), Combo(ComboId::PlotType)];
        if t <= 3 {
            if t == 3 {
                a.extend([Label("Error threshold : "), Spin("doubleSpinBoxError")]);
            }
            if t == 0 || t == 3 {
                a.extend([Label("Camera : "), Combo(ComboId::Camera)]);
            }
            a.extend([Label("Marker : "), Combo(ComboId::Marker1)]);
            if t == 2 {
                a.extend([Label("Marker : "), Combo(ComboId::Marker2)]);
            }
        } else {
            if t == 5 {
                a.extend([Label("Error threshold : "), Spin("doubleSpinBoxErrorRB")]);
            }
            a.extend([
                Label("Rigid Body : "),
                Combo(ComboId::Body),
                Combo(ComboId::TransType),
            ]);
            a.push(Combo(if t == 4 {
                ComboId::TransPart
            } else {
                ComboId::RbError
            }));
            a.push(Button("Update"));
        }
        let mut b = Vec::new();
        if t <= 1 || t == 3 {
            b.push(Check("Display status"));
        }
        if has_speed {
            b.push(Check("Display time "));
        }
        b.extend([Button("Fit plot"), Button("Export")]);
        (a, b)
    }

    fn combo_items(&self, id: ComboId) -> Vec<&str> {
        match id {
            ComboId::PlotType => PLOT_TYPES.to_vec(),
            ComboId::Camera => self.items.cameras.iter().map(String::as_str).collect(),
            ComboId::Marker1 => self.items.markers.iter().map(String::as_str).collect(),
            ComboId::Marker2 => self.items.marker2.iter().map(|(s, _)| s.as_str()).collect(),
            ComboId::Body => self.items.bodies.iter().map(String::as_str).collect(),
            ComboId::TransType => TRANS_TYPES.to_vec(),
            ComboId::TransPart => TRANS_PARTS.to_vec(),
            ComboId::RbError => RB_ERRORS.to_vec(),
        }
    }

    /// The width an item had when last drawn, or an estimate from its text.
    fn item_width(&self, ui: &Ui, item: RowItem) -> f32 {
        let text_w = |s: &str, style: egui::TextStyle| {
            let font = style.resolve(ui.style());
            ui.painter()
                .layout_no_wrap(s.to_owned(), font, Color32::BLACK)
                .size()
                .x
        };
        let sp = ui.spacing();
        match item {
            RowItem::Combo(id) => {
                let longest = self
                    .combo_items(id)
                    .iter()
                    .map(|s| text_w(s, egui::TextStyle::Button))
                    .fold(0.0_f32, f32::max);
                longest + 34.0
            }
            RowItem::Label(s) | RowItem::Check(s) | RowItem::Button(s) | RowItem::Spin(s) => {
                if let Some(w) = self.row_measured.get(s) {
                    return *w;
                }
                match item {
                    RowItem::Label(s) => text_w(s, egui::TextStyle::Body),
                    RowItem::Check(s) => {
                        text_w(s, egui::TextStyle::Body) + sp.icon_width + sp.icon_spacing
                    }
                    RowItem::Button(s) => {
                        text_w(s, egui::TextStyle::Button) + 2.0 * sp.button_padding.x
                    }
                    _ => 70.0,
                }
            }
        }
    }

    /// How the row is laid out in `width` points (Qt's layout of `bottomFrame`): buttons,
    /// check boxes, labels and spin boxes keep their size; the combo boxes share what is
    /// left, never wider than their contents and never narrower than [`MIN_COMBO`]. When
    /// even that does not fit, the frame's controls and the check boxes and buttons go on
    /// two lines; a line that still does not fit scrolls sideways.
    fn row_plan(&self, ui: &Ui, width: f32, has_speed: bool) -> RowPlan {
        let gap = row_gap(ui.style());
        let (a, b) = self.row_items(has_speed);
        let fixed = |items: &[RowItem]| -> f32 {
            let n = items.len() as f32;
            items
                .iter()
                .filter(|i| !matches!(i, RowItem::Combo(_)))
                .map(|i| self.item_width(ui, *i))
                .sum::<f32>()
                + gap * (n - 1.0).max(0.0)
        };
        let naturals = |items: &[RowItem]| -> Vec<f32> {
            items
                .iter()
                .filter(|i| matches!(i, RowItem::Combo(_)))
                .map(|i| self.item_width(ui, *i))
                .collect()
        };
        let line = |items: Vec<RowItem>, avail: f32| -> RowLine {
            let nat = naturals(&items);
            let left = avail - fixed(&items);
            let widths = share(&nat, left, MIN_COMBO);
            let used = fixed(&items) + widths.iter().sum::<f32>();
            RowLine {
                items,
                combo_widths: widths,
                scroll: used > avail + 0.5,
            }
        };
        // Units that stay together: a label with the control it names, or one control.
        let mut all = a;
        all.extend(b);
        let mut units: Vec<Vec<RowItem>> = Vec::new();
        for item in all {
            match units.last_mut() {
                Some(u) if u.len() == 1 && matches!(u[0], RowItem::Label(_)) => u.push(item),
                _ => units.push(vec![item]),
            }
        }
        let min_w = |u: &[RowItem]| -> f32 {
            fixed(u)
                + u.iter().filter(|i| matches!(i, RowItem::Combo(_))).count() as f32 * MIN_COMBO
        };
        // One line when everything fits with the combo boxes at their minimum; otherwise
        // lines filled in order (Qt has no second line: it squeezes the labels and check
        // boxes too, which can leave them unreadable).
        let mut lines: Vec<Vec<RowItem>> = vec![Vec::new()];
        let mut used = 0.0;
        for u in units {
            let w = min_w(&u);
            let cur = lines.last_mut().expect("a line");
            if !cur.is_empty() && used + gap + w > width {
                lines.push(u);
                used = w;
            } else {
                used += if cur.is_empty() { w } else { gap + w };
                cur.extend(u);
            }
        }
        RowPlan {
            lines: lines.into_iter().map(|l| line(l, width)).collect(),
        }
    }

    fn bottom_row(
        &mut self,
        ui: &mut Ui,
        rect: egui::Rect,
        clip: egui::Rect,
        plan: RowPlan,
    ) -> RowOut {
        let mut out = RowOut::default();
        let frame_rect = egui::Rect::from_min_max(
            rect.min,
            egui::pos2(
                (rect.right() - PLUS_W - 4.0).max(rect.left()),
                rect.bottom(),
            ),
        );
        let plus_rect = egui::Rect::from_min_max(
            egui::pos2((rect.right() - PLUS_W).max(rect.left()), rect.top()),
            rect.max,
        );
        let mut row = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(frame_rect)
                .layout(Layout::top_down(Align::LEFT)),
        );
        row.set_clip_rect(frame_rect.intersect(clip));
        crate::qt::styled_frame(&mut row, |ui| {
            let pad = frame_pad(ui.style()).x;
            ui.set_min_width((frame_rect.width() - pad).max(0.0));
            ui.set_max_width((frame_rect.width() - pad).max(0.0));
            for (k, line) in plan.lines.into_iter().enumerate() {
                if line.scroll {
                    egui::ScrollArea::horizontal()
                        .id_salt(("plot_row_scroll", k))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| self.draw_line(ui, &line, &mut out));
                        });
                } else {
                    ui.horizontal(|ui| self.draw_line(ui, &line, &mut out));
                }
            }
        });
        // toolButtonExtraPlot, outside the row's frame.
        let mut plus = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(plus_rect)
                .layout(Layout::left_to_right(Align::Center)),
        );
        plus.set_clip_rect(plus_rect.intersect(clip));
        let text = if self.extra.is_some() { "-" } else { "+" };
        if plus
            .add(egui::Button::new(text).min_size(egui::vec2(PLUS_W, 0.0)))
            .clicked()
        {
            out.extra = true;
        }
        out
    }

    fn draw_line(&mut self, ui: &mut Ui, line: &RowLine, out: &mut RowOut) {
        let mut label = None;
        let mut combos = line.combo_widths.iter();
        for item in &line.items {
            let r = match *item {
                RowItem::Label(s) => {
                    let r = ui.label(s);
                    label = Some(r.id);
                    Some((s, r))
                }
                RowItem::Spin(name) => {
                    let l = label.unwrap_or(ui.scope_id());
                    let changed = if name == "doubleSpinBoxError" {
                        double_spin(ui, &mut self.threshold, 0.01, 2, name, l)
                    } else {
                        let step = if self.rb_decimals == 3 { 0.001 } else { 0.01 };
                        double_spin(ui, &mut self.threshold_rb, step, self.rb_decimals, name, l)
                    };
                    out.redraw |= changed.0;
                    Some((name, changed.1))
                }
                RowItem::Combo(id) => {
                    let w = combos.next().copied().unwrap_or(MIN_COMBO);
                    self.draw_combo(ui, id, w, label, out);
                    None
                }
                RowItem::Check(s) => {
                    let v = if s == "Display status" {
                        &mut self.display_status
                    } else {
                        &mut self.display_time
                    };
                    let r = ui.checkbox(v, s);
                    out.fit |= r.changed();
                    Some((s, r))
                }
                RowItem::Button("Update") => {
                    let r = ui.add_enabled(self.editable, egui::Button::new("Update"));
                    out.update |= r.clicked();
                    Some(("Update", r))
                }
                RowItem::Button(s) => {
                    let r = ui.button(s);
                    if r.clicked() {
                        if s == "Fit plot" {
                            out.fit = true;
                        } else {
                            out.export = true;
                        }
                    }
                    Some((s, r))
                }
            };
            if let Some((key, r)) = r {
                self.row_measured.insert(key, r.rect.width());
            }
        }
    }

    fn draw_combo(
        &mut self,
        ui: &mut Ui,
        id: ComboId,
        w: f32,
        label: Option<egui::Id>,
        out: &mut RowOut,
    ) {
        let (salt, selected, name) = match id {
            ComboId::PlotType => ("plot_type", self.plot_type, None),
            ComboId::Camera => ("camera", self.camera, None),
            ComboId::Marker1 => ("marker1", self.marker1, None),
            ComboId::Marker2 => ("marker2", self.marker2, None),
            ComboId::Body => ("rigid_body", self.body, None),
            ComboId::TransType => (
                "trans_type",
                self.trans_type,
                Some("comboBoxRigidBodyTransType"),
            ),
            ComboId::TransPart => (
                "trans_part",
                self.trans_part,
                Some("comboBoxRigidBodyTransPart"),
            ),
            ComboId::RbError => ("rb_error", self.rb_error, Some("comboBoxRigidBodyError")),
        };
        let label = if name.is_some() { None } else { label };
        let chosen = {
            let items = self.combo_items(id);
            sized_combo(ui, salt, Some(selected), &items, w, label, name)
        };
        let Some(i) = chosen else { return };
        match id {
            ComboId::PlotType => {
                self.plot_type = i;
                out.plot_type = true;
            }
            ComboId::Camera => {
                self.camera = i;
                out.redraw = true;
            }
            ComboId::Marker1 => {
                self.marker1 = i;
                out.select_marker = Some(i);
                out.redraw = true;
            }
            ComboId::Marker2 => {
                self.marker2 = i;
                out.redraw = true;
            }
            ComboId::Body => {
                self.body = i;
                out.select_body = Some(i);
                out.redraw = true;
            }
            ComboId::TransType => {
                self.trans_type = i;
                out.redraw = true;
            }
            ComboId::TransPart => {
                self.trans_part = i;
                out.redraw = true;
            }
            ComboId::RbError => {
                // on_comboBoxRigidBodyError_currentIndexChanged: 3 decimals and step 0.001
                // for 3D, 2 and 0.01 for 2D; the value goes back to 0.
                self.rb_error = i;
                self.rb_decimals = if i == 0 { 3 } else { 2 };
                self.threshold_rb = 0.0;
                out.redraw = true;
            }
        }
    }

    /// "Export": the application asks for the file ("Save PlotData", `PlotData.csv` in
    /// the last used folder, through its `FileChooser`) and passes it back as
    /// [`Action::PlotExportTo`], which [`Self::export_to`] writes.
    fn export(cx: &mut Cx<'_>) {
        cx.raise(Action::PlotExport);
    }

    /// Writes the plotted choice to `path`.
    fn export_to(&self, cx: &Cx<'_>, path: &std::path::Path) {
        let (Some(doc), Some(t)) = (cx.doc, cx.state.active_trial) else {
            return;
        };
        let Some(view) = cx.state.trial_view().copied() else {
            return;
        };
        let span = FrameSpan::from_one_based(view.start_frame, view.end_frame);
        let Some(what) = self.export_request() else {
            return;
        };
        let text = plots::export_csv(doc, t, &what, span);
        if let Err(e) = std::fs::write(path, text) {
            cx.log
                .error(format!("Cannot write {}: {e}", path.display()));
        }
    }

    /// What "Export" writes for the current choices.
    #[must_use]
    pub fn export_request(&self) -> Option<PlotExport> {
        let camera = self.camera.checked_sub(1);
        Some(match self.plot_type {
            0 => PlotExport::Positions2d {
                marker: self.marker1,
                camera,
            },
            1 => PlotExport::Positions3d {
                marker: self.marker1,
            },
            2 => {
                let (label, target) = self.items.marker2.get(self.marker2)?;
                PlotExport::Distance {
                    marker: self.marker1,
                    target: *target,
                    target_label: label.clone(),
                }
            }
            3 => PlotExport::Reprojection {
                marker: self.marker1,
                camera,
                camera_label: self.items.cameras.get(self.camera)?.clone(),
            },
            4 => PlotExport::Transformation {
                body: self.body,
                choice: PoseChoice::from_index(self.trans_type),
                part: self.trans_part,
            },
            _ => PlotExport::BodyError {
                body: self.body,
                choice: PoseChoice::from_index(self.trans_type),
                dim: self.error_dim(),
            },
        })
    }

    /// `on_toolButtonExtraPlot_clicked`: hide the extra plot, or ask for a file and then
    /// for the number of values per frame.
    /// The file comes back as [`Action::PlotExtraFrom`] (the application's `FileChooser`,
    /// "Select data you want to plot", "Dataset (*.csv)").
    fn extra_clicked(&mut self, cx: &mut Cx<'_>) {
        if self.extra.is_some() {
            self.extra = None;
            self.extra_drawn = None;
            return;
        }
        cx.raise(Action::PlotExtraData);
    }

    /// `QInputDialog::getDouble(this, "Enter number of data values per frame", "values per
    /// frame", 1.0, 0.0, 2147283647, 4)`.
    fn show_prompt(&mut self, ui: &mut Ui, cx: &Cx<'_>) {
        let Some(p) = self.prompt.as_mut() else {
            return;
        };
        let mut done: Option<bool> = None;
        egui::Modal::new(egui::Id::unique("plot_extra_rate")).show(ui.ctx(), |ui| {
            ui.label(egui::RichText::new("Enter number of data values per frame").strong());
            let l = ui.label("values per frame");
            ui.add(
                egui::DragValue::new(&mut p.rate)
                    .range(0.0..=2_147_283_647.0)
                    .fixed_decimals(4)
                    .speed(0.01),
            )
            .labelled_by(l.id);
            ui.horizontal(|ui| {
                if ui.button("OK").clicked() {
                    done = Some(true);
                }
                if ui.button("Cancel").clicked() {
                    done = Some(false);
                }
            });
        });
        match done {
            Some(true) => {
                let p = self.prompt.take();
                if let Some(p) = p {
                    match std::fs::read_to_string(&p.path) {
                        Ok(text) => self.set_extra(&text, p.rate),
                        Err(e) => cx
                            .log
                            .error(format!("Cannot read {}: {e}", p.path.display())),
                    }
                }
            }
            Some(false) => self.prompt = None,
            None => {}
        }
    }

    fn plot_area(
        &mut self,
        ui: &mut Ui,
        cx: &mut Cx<'_>,
        active: usize,
        status_colors: &[Color32; 9],
        interp_colors: &[Color32; 4],
    ) {
        let Some(c) = &self.cache else {
            return;
        };
        let frames = c.frames;
        let series: Vec<Series<'_>> = c
            .series
            .iter()
            .map(|s| series_of(s, frames.first))
            .collect();
        let event_colors: Vec<[Color32; 2]> = c
            .bands
            .iter()
            .map(|b| match b.colors {
                BandColors::Event(e) => e,
                _ => [Color32::TRANSPARENT; 2],
            })
            .collect();
        let bands: Vec<BandRow<'_>> = c
            .bands
            .iter()
            .zip(&event_colors)
            .map(|(b, ev)| BandRow {
                first: frames.first,
                values: &b.values,
                colors: match b.colors {
                    BandColors::Status => status_colors,
                    BandColors::Interpolation => interp_colors,
                    BandColors::Event(_) => ev,
                },
                y: b.y,
                outline: b.outline,
            })
            .collect();
        let mut spec = PlotSpec::new(frames, &c.x_label, &c.left_label);
        spec.right_label = c.right_label.as_deref();
        spec.series = series;
        spec.bands = bands;
        let cursor_x = frames.x_of(active);
        spec.style = PlotStyle::from_visuals(ui.visuals());
        spec.cursor = Some(Cursor {
            pen: Pen::solid(ui.visuals().strong_text_color()),
            ..Cursor::new(cursor_x, c.span)
        });
        spec.selection = Some(self.selection.rect(&frames, c.span));
        spec.generation = Some(self.generation);

        let extra_h = if self.extra.is_some() {
            (ui.available_height() * 0.5).floor()
        } else {
            0.0
        };
        // plotWidgetExtra is the splitter's first widget, above plotWidget.
        let extra_rect = ui.available_rect_before_wrap();
        if extra_h > 0.0 {
            ui.add_space(extra_h);
        }
        let out = egui::Frame::NONE
            .stroke(egui::Stroke::new(
                1.0,
                crate::theme::frame_line(ui.visuals()),
            ))
            .show(ui, |ui| {
                ui.set_min_size(ui.available_size());
                PlotWidget::new(&spec, &mut self.view)
                    .id_salt("plot_widget")
                    .show(ui)
            })
            .inner;
        self.last_cursor_x = Some(cursor_x);
        self.last_axis_rect = out.axis_rect;
        let margins = out.margins;
        drop(spec);

        let mut frame_change = None;
        let mut redraw = false;
        let mut delete = false;
        for ev in &out.events {
            self.selection.apply(ev);
            match *ev {
                PlotEvent::Frame(f) => frame_change = Some(f),
                PlotEvent::SelectionEnd(_) => redraw = true,
                // Delete or Backspace with the plot focused: deleteData (E44).
                PlotEvent::DeleteRange => delete = true,
                PlotEvent::SelectionStart(_) | PlotEvent::ViewChanged(_) => {}
            }
        }
        self.last_events = out.events;
        if let Some(f) = frame_change {
            cx.change(|s| s.change_active_frame_trial(Some(f), false));
        }
        if redraw {
            self.redraw = true;
        }
        if delete
            && self.editable
            && let Some(ask) = self.range_delete_ask(cx, false)
        {
            cx.request(EditRequest::Ask(Box::new(ask)));
        }

        if let Some(extra) = &self.extra {
            // PlotWindow::updateExtraPlot.
            let r =
                egui::Rect::from_min_size(extra_rect.min, egui::vec2(extra_rect.width(), extra_h));
            let axis = (frames.multiplier().to_bits(), frames.offset().to_bits());
            if self.extra_drawn.as_ref().is_none_or(|d| d.axis != axis) {
                let pos: Vec<f64> = extra
                    .pos
                    .iter()
                    .map(|p| p * frames.multiplier() + frames.offset())
                    .collect();
                let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
                for v in extra.columns.iter().flatten() {
                    lo = lo.min(*v);
                    hi = hi.max(*v);
                }
                self.extra_generation += 1;
                self.extra_drawn = Some(ExtraDrawn {
                    axis,
                    pos,
                    range: (lo, hi),
                    generation: self.extra_generation,
                });
            }
            let drawn = self.extra_drawn.as_ref().expect("computed above");
            let (pos, (lo, hi)) = (&drawn.pos, drawn.range);
            let mut spec = PlotSpec::new(frames, "", "");
            spec.generation = Some(drawn.generation);
            spec.fixed_lr_margins = Some((margins.left, margins.right));
            for (i, col) in extra.columns.iter().enumerate() {
                let n = col.len().min(pos.len());
                spec.series.push(
                    Series::new(
                        "extra",
                        SeriesData::Xy {
                            x: &pos[..n],
                            y: &col[..n],
                        },
                        Pen::solid(EXTRA_COLOURS[i % 12]),
                    )
                    .line(LineStyle::None)
                    .scatter(ScatterShape::Cross, 2.0),
                );
            }
            spec.style = PlotStyle::from_visuals(ui.visuals());
            spec.cursor = Some(Cursor {
                pen: Pen::solid(ui.visuals().strong_text_color()),
                ..Cursor::new(cursor_x, None)
            });
            self.extra_view.x = self.view.x;
            if lo.is_finite() && hi.is_finite() {
                self.extra_view.left = ranging::set_range(self.extra_view.left, (lo, hi));
            }
            let mut child = ui.new_child(egui::UiBuilder::new().max_rect(r));
            egui::Frame::NONE
                .stroke(egui::Stroke::new(
                    1.0,
                    crate::theme::frame_line(ui.visuals()),
                ))
                .show(&mut child, |ui| {
                    ui.set_min_size(ui.available_size());
                    // The C++ extra plot takes no mouse input of its own.
                    let _ = PlotWidget::new(&spec, &mut self.extra_view)
                        .id_salt("plot_widget_extra")
                        .show(ui);
                });
            self.extra_view.x = self.view.x;
        }
    }
}

/// What the bottom row asked for this frame.
#[derive(Default)]
struct RowOut {
    plot_type: bool,
    redraw: bool,
    fit: bool,
    export: bool,
    extra: bool,
    select_marker: Option<usize>,
    select_body: Option<usize>,
    /// "Update" was pressed (E48).
    update: bool,
}

/// The widget's series for `s`. A per-frame series holds one value per frame of the trial's
/// span (the engine's `FrameSpan`), so its first value belongs to `first`, the span's first
/// frame (0-based), as the C++ plots frame `i` at `i * posMultiplier + posOffset`.
fn series_of(s: &OwnedSeries, first: usize) -> Series<'_> {
    let data = match &s.data {
        OwnedData::Frames(v) => SeriesData::Frames { first, values: v },
        OwnedData::Xy(x, y) => SeriesData::Xy { x, y },
    };
    let mut out = Series::new("series", data, s.pen).line(s.line).axis(s.axis);
    if let Some((shape, size)) = s.scatter {
        out = out.scatter(shape, size);
    }
    if let Some((e, pen)) = &s.errors {
        out = out.errors(ErrorBars {
            errors: e,
            pen: *pen,
            style: ErrorStyle::QCP_BARS,
        });
    }
    out
}

/// Index into the status colours, in the order of [`status_colors`]
/// (`PlotWindow::drawStatus`'s switch; other codes use the "undefined" colour).
fn status_index(s: MarkerStatus) -> u8 {
    match s {
        MarkerStatus::Untrackable => 1,
        MarkerStatus::Interpolated => 2,
        MarkerStatus::Tracked => 3,
        MarkerStatus::TrackedAndOptimized => 4,
        MarkerStatus::Set => 5,
        MarkerStatus::SetAndOptimized => 6,
        MarkerStatus::Manual => 7,
        MarkerStatus::ManualAndOptimized => 8,
        _ => 0,
    }
}

fn interpolation_index(m: Interpolation) -> u8 {
    match m {
        Interpolation::None => 0,
        Interpolation::Repeat => 1,
        Interpolation::Linear => 2,
        Interpolation::Cubic => 3,
    }
}

fn status_colors(s: &crate::settings::StatusColors) -> [Color32; 9] {
    [
        parse_color(&s.undefined),
        parse_color(&s.untrackable),
        parse_color(&s.interpolated),
        parse_color(&s.tracked),
        parse_color(&s.tracked_and_opt),
        parse_color(&s.set),
        parse_color(&s.set_and_opt),
        parse_color(&s.manual),
        parse_color(&s.manual_and_opt),
    ]
}

fn interpolation_colors(s: &crate::settings::StatusColors) -> [Color32; 4] {
    [
        parse_color(&s.inter_none),
        parse_color(&s.inter_repeat),
        parse_color(&s.inter_linear),
        parse_color(&s.inter_cubic),
    ]
}

/// A `QComboBox` exactly `width` points wide, its text cut when it does not fit (Qt's
/// `QComboBox` elides its text when the layout gives it less than its size hint). Named by
/// its buddy label, or by `name` (its Qt object name) when it has none.
fn sized_combo(
    ui: &mut Ui,
    id: &str,
    selected: Option<usize>,
    items: &[&str],
    width: f32,
    label: Option<egui::Id>,
    name: Option<&str>,
) -> Option<usize> {
    let text = selected.and_then(|i| items.get(i)).copied().unwrap_or("");
    let h = ui.spacing().interact_size.y;
    ui.allocate_ui_with_layout(
        egui::vec2(width, h),
        Layout::left_to_right(Align::Center),
        |ui| {
            ui.set_max_width(width);
            let mut chosen = None;
            let r = egui::ComboBox::from_id_salt(id)
                .selected_text(text)
                .width(width)
                .truncate()
                .show_ui(ui, |ui| {
                    for (i, it) in items.iter().enumerate() {
                        if ui.selectable_label(Some(i) == selected, *it).clicked()
                            && Some(i) != selected
                        {
                            chosen = Some(i);
                        }
                    }
                });
            let chosen = chosen.or_else(|| {
                crate::qt::combo_wheel(ui, &r.response, selected, items.len(), |_| true)
            });
            if let Some(l) = label {
                let _ = r.response.clone().labelled_by(l);
            }
            if let Some(n) = name {
                let _ = r.response.accessible_name(n);
            }
            chosen
        },
    )
    .inner
}

/// The combo boxes of the row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComboId {
    PlotType,
    Camera,
    Marker1,
    Marker2,
    Body,
    TransType,
    TransPart,
    RbError,
}

/// One control of the row; the text is also the key of its measured width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowItem {
    Label(&'static str),
    Spin(&'static str),
    Combo(ComboId),
    Check(&'static str),
    Button(&'static str),
}

struct RowLine {
    items: Vec<RowItem>,
    combo_widths: Vec<f32>,
    scroll: bool,
}

struct RowPlan {
    lines: Vec<RowLine>,
}

/// Smallest width of a combo box: its arrow and two or three characters.
const MIN_COMBO: f32 = 48.0;
/// Horizontal spacing between the row's controls: egui's item spacing.
fn row_gap(style: &egui::Style) -> f32 {
    style.spacing.item_spacing.x
}
/// Vertical spacing between the row's lines: egui's item spacing.
fn row_gap_y(style: &egui::Style) -> f32 {
    style.spacing.item_spacing.y
}
/// The styled frame's margins and border, both sides.
fn frame_pad(style: &egui::Style) -> egui::Vec2 {
    egui::Frame::group(style).total_margin().sum()
}
/// Width of `toolButtonExtraPlot`.
const PLUS_W: f32 = 26.0;

/// Shares `total` among items with the given natural widths: each gets at most its natural
/// width and at least `min`; what one does not use goes to the others (water filling).
fn share(naturals: &[f32], total: f32, min: f32) -> Vec<f32> {
    let n = naturals.len();
    let mut out = vec![min; n];
    if n == 0 {
        return out;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| naturals[*a].total_cmp(&naturals[*b]));
    let mut left = total.max(0.0);
    let mut remaining = n;
    for i in order {
        let fair = left / remaining as f32;
        let w = naturals[i].min(fair).max(min);
        out[i] = w.floor();
        left -= out[i];
        remaining -= 1;
    }
    out
}

/// A `QDoubleSpinBox` with range 0 to 99.99 (the form leaves Qt's default range), the
/// given step and decimals, and up and down arrows; returns whether the value changed.
fn double_spin(
    ui: &mut Ui,
    value: &mut f64,
    step: f64,
    decimals: usize,
    name: &str,
    label: egui::Id,
) -> (bool, egui::Response) {
    let before = *value;
    let h = ui.spacing().interact_size.y;
    let r = ui.scope(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        let _ = ui
            .add_sized(
                [56.0, h],
                egui::DragValue::new(value)
                    .range(0.0..=99.99)
                    .speed(step)
                    .fixed_decimals(decimals),
            )
            .labelled_by(label)
            .accessible_name(name);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, h), egui::Sense::hover());
        let up = egui::Rect::from_min_max(rect.min, egui::pos2(rect.max.x, rect.center().y));
        let down = egui::Rect::from_min_max(egui::pos2(rect.min.x, rect.center().y), rect.max);
        let id = ui.scope_id();
        for (r, delta, salt) in [(up, 1.0, "up"), (down, -1.0, "down")] {
            let resp = ui.interact(r, id.with(("spin", salt)), egui::Sense::click());
            let color = ui.style().interact(&resp).fg_stroke.color;
            let c = r.center();
            let s = 3.0;
            let pts = if delta > 0.0 {
                vec![
                    egui::pos2(c.x - s, c.y + s * 0.6),
                    egui::pos2(c.x + s, c.y + s * 0.6),
                    egui::pos2(c.x, c.y - s * 0.6),
                ]
            } else {
                vec![
                    egui::pos2(c.x - s, c.y - s * 0.6),
                    egui::pos2(c.x + s, c.y - s * 0.6),
                    egui::pos2(c.x, c.y + s * 0.6),
                ]
            };
            ui.painter()
                .add(egui::Shape::convex_polygon(pts, color, egui::Stroke::NONE));
            if resp.clicked() {
                let scale = 10f64.powi(decimals as i32);
                *value = (((*value + delta * step) * scale).round() / scale).clamp(0.0, 99.99);
            }
        }
    });
    (*value != before, r.response)
}

impl Panel for PlotPanel {
    fn id(&self) -> PanelId {
        PanelId::Plot
    }

    fn ui(&mut self, ui: &mut Ui, cx: &mut Cx<'_>) {
        let (Some(doc), Some(t)) = (cx.doc, cx.state.active_trial) else {
            return;
        };
        if t >= doc.project().trials.len() || cx.state.workspace != Workspace::MarkerTracking {
            return;
        }
        let Some(view) = cx.state.trial_view().copied() else {
            return;
        };
        let trial = doc.trial(t);
        let span = FrameSpan::from_one_based(view.start_frame, view.end_frame);
        let speed = trial.recording_speed;
        let active = view.active_frame;
        // Shortcuts::eventFilter: Shift down or up, wherever the focus is.
        let shift = ui.input(|i| i.modifiers.shift);
        self.selection.update_shift(shift, active as i64);
        self.editable = !cx.state.disable_draw;

        self.show_prompt(ui, cx);

        let status_colors = status_colors(&cx.settings.colors);
        let interp_colors = interpolation_colors(&cx.settings.colors);
        // The panel's area: the plot above, the row of controls below. Children with fixed
        // rectangles, clipped, so wide contents never widen the panel.
        let full = ui.available_rect_before_wrap();
        ui.allocate_rect(full, egui::Sense::hover());
        let plan = self.row_plan(
            ui,
            (full.width() - PLUS_W - 4.0 - frame_pad(ui.style()).x).max(0.0),
            speed > 0.0,
        );
        self.row_lines = plan
            .lines
            .iter()
            .map(|l| (l.combo_widths.clone(), l.scroll))
            .collect();
        let n_lines = plan.lines.len() as f32;
        let row_h = n_lines * ui.spacing().interact_size.y
            + (n_lines - 1.0) * row_gap_y(ui.style())
            + frame_pad(ui.style()).y;
        let row_rect = egui::Rect::from_min_max(
            egui::pos2(full.left(), (full.bottom() - row_h).max(full.top())),
            full.max,
        );
        let plot_rect = egui::Rect::from_min_max(
            full.min,
            egui::pos2(full.right(), (row_rect.top() - 4.0).max(full.top())),
        );
        let clip = ui.clip_rect();
        {
            let row = self.bottom_row(ui, row_rect, clip, plan);
            if row.plot_type {
                // on_comboBoxPlotType_currentIndexChanged: the active marker for types 0 to
                // 3, the body in the combo for 4 and 5.
                if self.plot_type <= 3 {
                    self.active_point_changed(view.active_marker);
                }
                self.redraw = true;
            }
            if let Some(m) = row.select_marker {
                // PointsDockWidget::selectPoint(idx + 1).
                cx.change(|s| s.select_marker(Some(m)));
                self.seen_marker = Some(m);
                self.seen_body = None;
            }
            if let Some(b) = row.select_body {
                // PointsDockWidget::selectBody(idx + 1).
                cx.change(|s| s.select_body(Some(b)));
                self.seen_body = Some(b);
                self.seen_marker = None;
            }
            if row.redraw {
                self.redraw = true;
            }
            if row.fit {
                self.fit_x = true;
                self.redraw = true;
            }
            if row.export {
                Self::export(cx);
            }
            if row.update {
                Self::update_bodies(cx);
            }
            if row.extra {
                self.extra_clicked(cx);
            }
            let frames = FrameAxis {
                first: span.first,
                last: span.last,
                fps: (self.display_time && speed > 0.0).then_some(speed),
            };
            if self.redraw || self.cache.as_ref().is_none_or(|c| c.frames != frames) {
                self.do_redraw(doc, t, trial, span, frames);
                self.redraw = false;
            }
            if self.fit_x {
                if let Some(r) = ranging::fit_x(&frames) {
                    self.view.x = r;
                }
                self.fit_x = false;
            }
            let mut plot_ui = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(plot_rect)
                    .layout(Layout::top_down(Align::LEFT)),
            );
            plot_ui.set_clip_rect(plot_rect.intersect(clip));
            self.plot_area(&mut plot_ui, cx, active, &status_colors, &interp_colors);
        }
    }

    fn on_event(&mut self, event: &AppEvent, cx: &mut Cx<'_>) {
        match event {
            AppEvent::ProjectOpened => {
                self.cache = None;
                self.data_key = None;
                self.selection = FrameSelection::new();
                self.trial_or_workspace_changed(cx);
            }
            AppEvent::ProjectClosed => {
                self.cache = None;
                self.data_key = None;
                self.items = Items::default();
                self.selection = FrameSelection::new();
                self.seen_marker = None;
                self.seen_body = None;
            }
            AppEvent::Changed(ch) => {
                if cx.state.workspace != Workspace::MarkerTracking {
                    return;
                }
                if ch.document || ch.workspace || ch.active_trial || ch.trial_range {
                    // The series are computed again on the redraw below.
                    self.data_key = None;
                }
                if ch.workspace || ch.active_trial {
                    self.trial_or_workspace_changed(cx);
                } else if ch.document
                    && let (Some(doc), Some(t)) = (cx.doc, cx.state.active_trial)
                    && t < doc.project().trials.len()
                {
                    // After an edit, undo or redo: the lists may have changed
                    // (PlotWindow::updateMarkers after each edit in XMALab 3); the plot is
                    // redrawn below.
                    self.update_markers(doc, doc.trial(t));
                }
                if ch.trial_range {
                    // SequenceNavigationFrame::setStartEndSequence: resetRange, draw.
                    self.fit_x = true;
                }
                if ch.selection {
                    let view = cx.state.trial_view().copied();
                    let m = view.and_then(|v| v.active_marker);
                    let b = view.and_then(|v| v.active_body);
                    if m != self.seen_marker {
                        self.seen_marker = m;
                        self.active_point_changed(m);
                    }
                    if b != self.seen_body {
                        self.seen_body = b;
                        self.active_body_changed(b);
                    }
                }
                if ch.active_frame_trial
                    && let Some(f) = cx.state.active_frame_trial
                {
                    self.selection.active_frame_changed(f as i64);
                }
                // MainWindow::redrawGL → PlotWindow::draw.
                self.redraw = true;
            }
        }
    }

    fn on_action(&mut self, action: &Action, cx: &mut Cx<'_>) -> bool {
        match action {
            Action::PlotExportTo(p) => {
                self.export_to(cx, p);
                true
            }
            Action::PlotExtraFrom(p) => {
                self.prompt = Some(RatePrompt {
                    path: p.clone(),
                    rate: 1.0,
                });
                true
            }
            Action::NextAboveError => {
                self.go_to_above_error(cx, true);
                true
            }
            Action::PrevAboveError => {
                self.go_to_above_error(cx, false);
                true
            }
            Action::SetUntrackableRange | Action::DeleteAllAboveError => {
                let ask = if *action == Action::SetUntrackableRange {
                    self.range_delete_ask(cx, true)
                } else {
                    self.above_error_ask(cx)
                };
                if let Some(ask) = ask {
                    cx.request(EditRequest::Ask(Box::new(ask)));
                }
                true
            }
            Action::SetInterpolationRange => {
                self.set_interpolation(cx, false);
                true
            }
            Action::Confirmed(inner) if **inner == Action::SetInterpolationRange => {
                self.set_interpolation(cx, true);
                true
            }
            Action::SetEventOn | Action::SetEventOff => {
                if let Some(r) = self.event_request(cx, *action == Action::SetEventOn) {
                    cx.request(r);
                }
                true
            }
            _ => false,
        }
    }
}
