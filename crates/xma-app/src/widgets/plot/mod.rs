//! The plot widget: everything XMALab's `PlotWindow` draws with QCustomPlot,
//! as an egui widget that takes plain data and reports what the user did.
//!
//! The widget knows nothing about projects, markers or the engine. A panel
//! builds a [`PlotSpec`] from its data each UI frame, keeps a [`PlotView`]
//! (the axis ranges and the interaction state) between frames, and reads
//! [`PlotEvent`]s back.
//!
//! ```no_run
//! use xma_app::widgets::plot::{
//!     Cursor, FrameAxis, FrameSelection, LineStyle, Pen, PlotEvent, PlotSpec, PlotView,
//!     PlotWidget, ScatterShape, Series, SeriesData, YAxis, qt, ranging,
//! };
//!
//! struct Panel {
//!     view: PlotView,
//!     selection: FrameSelection,
//!     frame: usize,
//!     x_px: Vec<f64>, // one value per frame, NaN where untracked
//!     y_px: Vec<f64>,
//! }
//!
//! impl Panel {
//!     fn ui(&mut self, ui: &mut egui::Ui) {
//!         let frames = FrameAxis::frames(0, self.x_px.len() - 1);
//!         // Y ranges: refit on every redraw, as the C++ does (X only on "Fit plot").
//!         let mm = |v: &[f64]| ranging::MinMax::new().extend(v.iter().copied());
//!         let (left, right) = ranging::fit_2d(mm(&self.x_px), mm(&self.y_px), None);
//!         self.view.left = left.axis;
//!         self.view.right = right.axis;
//!
//!         let mut spec = PlotSpec::new(frames, "Frame", "X - Position");
//!         spec.right_label = Some("Y - Position");
//!         let data = |v| SeriesData::Frames { first: 0, values: v };
//!         spec.series.push(
//!             Series::new("x", data(&self.x_px), Pen::solid(qt::RED))
//!                 .line(LineStyle::None)
//!                 .scatter(ScatterShape::Cross, 2.0),
//!         );
//!         spec.series.push(
//!             Series::new("y", data(&self.y_px), Pen::solid(qt::RED))
//!                 .line(LineStyle::None)
//!                 .scatter(ScatterShape::Circle, 2.0)
//!                 .axis(YAxis::Right),
//!         );
//!         spec.cursor = Some(Cursor::new(frames.x_of(self.frame), Some(left.span)));
//!         spec.selection = Some(self.selection.rect(&frames, Some(left.span)));
//!
//!         let out = PlotWidget::new(&spec, &mut self.view).show(ui);
//!         for ev in &out.events {
//!             self.selection.apply(ev);
//!             if let PlotEvent::Frame(f) = *ev
//!                 && f != self.frame
//!             {
//!                 self.frame = f; // the application's "change active frame"
//!                 self.selection.active_frame_changed(f as i64);
//!             }
//!         }
//!     }
//! }
//! ```
//!
//! Parts:
//! - [`model`]: the plain-data description ([`PlotSpec`], [`Series`], [`BandRow`], ...).
//! - [`ranging`]: the C++ range fitting and row placement as functions.
//! - [`ticks`]: QCustomPlot's tick steps, sub-ticks and `%g` labels.
//! - [`decimate`]: min/max-per-pixel-column reduction that keeps gaps.
//! - [`selection`]: the highlighted frame range state machine ([`FrameSelection`]).
//! - `input`: the mouse and key bindings (table in its module docs, repeated on [`PlotWidget`]).
//!
//! Drawing uses egui_plot's items (lines with its per-pixel band aggregation)
//! and its [`egui_plot::PlotTransform`], one transform per Y axis, so the right
//! axis is a real second axis. egui_plot's `Plot` container, axes, grid and
//! interaction are not used: axes and grid are drawn here to look like
//! QCustomPlot's, and all input is handled here.

pub mod decimate;
mod input;
pub mod model;
mod paint;
pub mod ranging;
pub mod selection;
pub mod ticks;

use egui::{Rect, Response, Sense, Ui, vec2};

pub use input::Target;
pub use model::{
    BandRow, Cursor, Dash, ErrorBars, ErrorStyle, FrameAxis, LineStyle, Margins, Pen, PlotSpec,
    PlotStyle, Scatter, ScatterShape, SelectionRect, Series, SeriesData, YAxis, qt,
};
pub use paint::PaintStats;
pub use selection::FrameSelection;

/// Axis ranges and interaction state, owned by the caller and kept between
/// UI frames. The widget changes the ranges in response to wheel and drag
/// input (and reports [`PlotEvent::ViewChanged`]); the caller may set them at
/// any time, e.g. from the functions in [`ranging`].
#[derive(Clone, Debug, PartialEq)]
pub struct PlotView {
    /// Visible X range (plot units: frames or seconds).
    pub x: (f64, f64),
    /// Visible range of the left Y axis.
    pub left: (f64, f64),
    /// Visible range of the right Y axis.
    pub right: (f64, f64),
    interaction: input::Interaction,
    /// The series' shapes as last drawn ([`PlotSpec::generation`]); not part of equality.
    shapes: paint::ShapeCache,
}

impl Default for PlotView {
    /// QCustomPlot's initial axis ranges, 0 to 5.
    fn default() -> Self {
        Self::new((0.0, 5.0), (0.0, 5.0), (0.0, 5.0))
    }
}

impl PlotView {
    /// A view with the given ranges.
    pub fn new(x: (f64, f64), left: (f64, f64), right: (f64, f64)) -> Self {
        Self {
            x,
            left,
            right,
            interaction: input::Interaction::default(),
            shapes: paint::ShapeCache::default(),
        }
    }
}

/// What the user did in one UI frame, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlotEvent {
    /// Left press or left drag (without Alt) over a frame inside the trial:
    /// go to this 0-based frame. Emitted once per change of frame during a drag.
    Frame(usize),
    /// Ctrl + left press: a frame range starts at this 0-based frame (which may
    /// lie outside the trial). Precedes the [`PlotEvent::Frame`] of the same press.
    SelectionStart(i64),
    /// Ctrl + left drag: the frame range now ends at this frame.
    SelectionEnd(i64),
    /// Delete or Backspace (without Shift) while the plot has keyboard focus:
    /// the C++ deletes the data in the selected range (`PlotWindow::deleteData`).
    DeleteRange,
    /// The user changed a range with the wheel or a drag.
    ViewChanged(Target),
}

/// What [`PlotWidget::show`] returns.
pub struct PlotOutput {
    /// Events in the order they happened.
    pub events: Vec<PlotEvent>,
    /// The widget's response (whole widget).
    pub response: Response,
    /// The axis rect (data area) in screen points.
    pub axis_rect: Rect,
    /// The margins used, for lining up a second plot ([`PlotSpec::fixed_lr_margins`]).
    pub margins: Margins,
    /// Counts from the paint pass.
    pub stats: PaintStats,
}

/// The plot widget. Fills the available space.
///
/// Bindings (from `PlotWindow::eventFilter` and QCustomPlot's settings there):
///
/// | Input | Effect |
/// |---|---|
/// | Wheel | zoom X about the pointer (×0.85 per notch towards the pointer) |
/// | Ctrl + wheel | zoom the left Y axis about the pointer |
/// | Alt + wheel | zoom the right Y axis about the pointer |
/// | Wheel with only Shift or Meta | zoom the axis the previous wheel zoomed |
/// | Left press / drag | [`PlotEvent::Frame`] for the frame under the pointer, inside the trial |
/// | Ctrl + left press, drag | [`PlotEvent::SelectionStart`], then [`PlotEvent::SelectionEnd`], plus frames |
/// | Alt + left drag | pan the axis the previous right-drag panned (X at first) |
/// | Right drag | pan X |
/// | Ctrl + right drag | pan the left Y axis |
/// | Alt + right drag | pan the right Y axis |
/// | Double click | nothing beyond the presses |
/// | Delete / Backspace (focused) | [`PlotEvent::DeleteRange`] |
///
/// Shift + frame change (range selection by keys or the frame bar) is not a
/// widget binding: it is driven by [`FrameSelection::update_shift`] and
/// [`FrameSelection::active_frame_changed`] from the application.
pub struct PlotWidget<'s, 'a> {
    spec: &'s PlotSpec<'a>,
    view: &'s mut PlotView,
    id_salt: Option<egui::IdSalt>,
    min_size: egui::Vec2,
}

impl<'s, 'a> PlotWidget<'s, 'a> {
    /// A widget showing `spec` in `view`.
    pub fn new(spec: &'s PlotSpec<'a>, view: &'s mut PlotView) -> Self {
        Self {
            spec,
            view,
            id_salt: None,
            min_size: vec2(120.0, 80.0),
        }
    }

    /// Sets the id salt (needed when one `Ui` shows several plots).
    pub fn id_salt(mut self, salt: impl egui::AsIdSalt) -> Self {
        self.id_salt = Some(egui::IdSalt::new(salt));
        self
    }

    /// Minimum size in points.
    pub fn min_size(mut self, size: egui::Vec2) -> Self {
        self.min_size = size;
        self
    }

    /// Lays out, handles input and paints.
    pub fn show(self, ui: &mut Ui) -> PlotOutput {
        let Self {
            spec,
            view,
            id_salt,
            min_size,
        } = self;
        let size = ui.available_size().max(min_size);
        let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
        let id = ui.make_persistent_id(id_salt.unwrap_or_else(|| egui::IdSalt::new("xma_plot")));
        let response = ui.interact(rect, id, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let font = egui::FontId::proportional(
            spec.style
                .font_size
                .unwrap_or_else(|| egui::TextStyle::Body.resolve(ui.style()).size),
        );

        let layout = |view: &PlotView| {
            paint::Layout::compute(
                &painter,
                rect,
                spec,
                [view.x, view.left, view.right],
                font.clone(),
            )
        };
        let mut lay = layout(view);

        // ---- input ----
        let mut events = Vec::new();
        if ui.is_enabled() {
            let (raw, primary_down, secondary_down) = ui.input(|i| {
                (
                    i.raw.events.clone(),
                    i.pointer.button_down(egui::PointerButton::Primary),
                    i.pointer.button_down(egui::PointerButton::Secondary),
                )
            });
            let line_speed = ui.ctx().options(|o| o.input_options.line_scroll_speed);
            let layer = ui.layer_id();
            let ctx = ui.ctx().clone();
            let accepts = move |p: egui::Pos2| {
                rect.contains(p) && ctx.layer_id_at(p).is_none_or(|l| l == layer)
            };
            let has_focus = response.has_focus();
            let geo = input::Geometry {
                axis_rect: lay.axis_rect,
                accepts: &accepts,
                has_focus,
                points_per_line: line_speed,
            };
            let mut ranges = input::Ranges {
                x: view.x,
                left: view.left,
                right: view.right,
            };
            let out = view
                .interaction
                .process(&raw, &geo, &spec.frames, &mut ranges);
            view.interaction.sync_buttons(primary_down, secondary_down);
            events = out.events;
            for (changed, t) in out
                .changed
                .iter()
                .zip([Target::X, Target::Left, Target::Right])
            {
                if *changed {
                    events.push(PlotEvent::ViewChanged(t));
                }
            }
            view.x = ranges.x;
            view.left = ranges.left;
            view.right = ranges.right;
            if out.pressed {
                response.request_focus();
            }
            if response.has_focus() {
                // The focus is for Delete and Backspace only: the application's keys go on
                // working.
                super::view_keys::mark_focus(ui.ctx(), id);
                ui.memory_mut(|m| {
                    m.set_focus_lock_filter(
                        id,
                        egui::EventFilter {
                            tab: false,
                            horizontal_arrows: true,
                            vertical_arrows: true,
                            escape: false,
                        },
                    );
                });
            }
            if out.wheel_used {
                // Keep a surrounding ScrollArea from scrolling as well.
                ui.input_mut(|i| {
                    i.smooth_scroll_delta = egui::Vec2::ZERO;
                });
            }
            if out.changed.iter().any(|c| *c) {
                lay = layout(view);
            }
        }

        // ---- paint ----
        let stats = paint_all(ui, &painter, &lay, spec, view);
        PlotOutput {
            events,
            response,
            axis_rect: lay.axis_rect,
            margins: lay.margins,
            stats,
        }
    }
}

fn paint_all(
    ui: &Ui,
    painter: &egui::Painter,
    lay: &paint::Layout,
    spec: &PlotSpec<'_>,
    view: &mut PlotView,
) -> PaintStats {
    let ppp = ui.ctx().pixels_per_point();
    painter.rect_filled(lay.widget, 0.0, spec.style.background);
    let (tl, tr) = paint::transforms(lay.axis_rect, view.x, view.left, view.right);
    let clipped = painter.with_clip_rect(lay.axis_rect.intersect(painter.clip_rect()));
    paint::paint_grid(&clipped, lay, &tl, spec);

    let mut shapes = Vec::new();
    let mut stats = PaintStats::default();
    if let Some(c) = &spec.cursor {
        shapes.push(paint::cursor_shape(c, &tl, painter));
    }
    if let Some(s) = &spec.selection {
        paint::selection_shapes(s, &tl, painter, &mut shapes);
    }
    if let Some(b) = paint::band_shapes(&spec.bands, &tl, &spec.frames, ppp) {
        shapes.push(b);
    }
    let target = paint::SeriesTarget {
        tl: &tl,
        tr: &tr,
        ranges: [view.x, view.left, view.right],
        painter: &clipped,
    };
    view.shapes
        .series(ui, spec, &target, &mut shapes, &mut stats);
    clipped.extend(shapes);
    paint::paint_axes(painter, lay, &tl, &tr, spec);
    stats
}
