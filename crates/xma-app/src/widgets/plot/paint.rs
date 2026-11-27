//! Layout and painting: QCustomPlot's look (white background, black axes on the
//! left, bottom and right, ticks pointing inwards, dotted grid) drawn with
//! egui's painter, and the data drawn through egui_plot items with our own
//! [`PlotTransform`]s, one per Y axis.

use std::sync::Arc;

use egui::{Color32, FontId, Galley, Mesh, Painter, Pos2, Rect, Shape, Stroke, Ui, pos2, vec2};
use egui_plot::{Line, PlotBounds, PlotItem as _, PlotPoint, PlotPoints, PlotTransform};

use super::decimate::{decimate, dedupe_pixels, step_center, visible_window};
use super::model::{
    BandRow, Dash, ErrorStyle, LineStyle, Margins, Pen, PlotSpec, ScatterShape, Series, YAxis,
};
use super::ticks::{Ticks, beautify_exponent, ticks};

// QCustomPlot's axis geometry (QCPAxis constructor, QCPAxisPainterPrivate).
const AXIS_PADDING: f32 = 5.0;
const TICK_IN: f32 = 5.0;
const SUBTICK_IN: f32 = 2.0;
const LEFT_TICK_LABEL_PAD: f32 = 5.0;
const LEFT_LABEL_PAD: f32 = 10.0;
const RIGHT_TICK_LABEL_PAD: f32 = 7.0;
const RIGHT_LABEL_PAD: f32 = 12.0;
const BOTTOM_TICK_LABEL_PAD: f32 = 3.0;
const BOTTOM_LABEL_PAD: f32 = 3.0;

/// One axis's ticks and laid-out labels.
pub(crate) struct AxisLayout {
    pub ticks: Ticks,
    pub galleys: Vec<Arc<Galley>>,
    pub max_w: f32,
    pub max_h: f32,
}

/// Where everything goes this frame.
pub(crate) struct Layout {
    pub widget: Rect,
    pub axis_rect: Rect,
    pub margins: Margins,
    pub x: AxisLayout,
    pub left: AxisLayout,
    pub right: Option<AxisLayout>,
    pub x_label: Option<Arc<Galley>>,
    pub left_label: Option<Arc<Galley>>,
    pub right_label: Option<Arc<Galley>>,
}

fn axis_layout(painter: &Painter, range: (f64, f64), font: &FontId, color: Color32) -> AxisLayout {
    let ticks = ticks(range);
    let galleys: Vec<Arc<Galley>> = ticks
        .labels
        .iter()
        .map(|l| painter.layout_no_wrap(beautify_exponent(l), font.clone(), color))
        .collect();
    let max_w = galleys.iter().map(|g| g.size().x).fold(0.0, f32::max);
    let max_h = galleys
        .iter()
        .map(|g| g.size().y)
        .fold(painter.fonts_mut(|f| f.row_height(font)), f32::max);
    AxisLayout {
        ticks,
        galleys,
        max_w,
        max_h,
    }
}

fn label_galley(
    painter: &Painter,
    text: &str,
    font: &FontId,
    color: Color32,
) -> Option<Arc<Galley>> {
    if text.is_empty() {
        return None;
    }
    let mut job =
        egui::text::LayoutJob::simple(text.to_owned(), font.clone(), color, f32::INFINITY);
    job.halign = egui::Align::Center;
    Some(painter.fonts_mut(|f| f.layout_job(job)))
}

impl Layout {
    /// QCustomPlot's automatic margins: for each visible axis, padding + tick
    /// labels + label padding + label. The top margin (no visible top axis in
    /// QCustomPlot) is half a text row so the top tick label is not cut off; the
    /// right margin without a right axis leaves room for half the last X label.
    pub(crate) fn compute(
        painter: &Painter,
        widget: Rect,
        spec: &PlotSpec<'_>,
        ranges: [(f64, f64); 3],
        font: FontId,
    ) -> Self {
        let color = spec.style.axis;
        let x = axis_layout(painter, ranges[0], &font, color);
        let left = axis_layout(painter, ranges[1], &font, color);
        let right = spec
            .right_label
            .map(|_| axis_layout(painter, ranges[2], &font, color));
        let x_label = label_galley(painter, spec.x_label, &font, color);
        let left_label = label_galley(painter, spec.left_label, &font, color);
        let right_label = spec
            .right_label
            .and_then(|l| label_galley(painter, l, &font, color));

        let row_h = painter.fonts_mut(|f| f.row_height(&font));
        let mut m = Margins {
            left: AXIS_PADDING
                + left.max_w
                + LEFT_TICK_LABEL_PAD
                + left_label
                    .as_ref()
                    .map_or(0.0, |g| g.size().y + LEFT_LABEL_PAD),
            right: match &right {
                Some(r) => {
                    AXIS_PADDING
                        + r.max_w
                        + RIGHT_TICK_LABEL_PAD
                        + right_label
                            .as_ref()
                            .map_or(0.0, |g| g.size().y + RIGHT_LABEL_PAD)
                }
                None => (x.galleys.last().map_or(0.0, |g| g.size().x) * 0.5).max(AXIS_PADDING),
            },
            top: (row_h * 0.5).ceil(),
            bottom: AXIS_PADDING
                + x.max_h
                + BOTTOM_TICK_LABEL_PAD
                + x_label
                    .as_ref()
                    .map_or(0.0, |g| g.size().y + BOTTOM_LABEL_PAD),
        };
        if let Some((l, r)) = spec.fixed_lr_margins {
            m.left = l;
            m.right = r;
        }
        let mut axis_rect = Rect::from_min_max(
            pos2(widget.left() + m.left, widget.top() + m.top),
            pos2(widget.right() - m.right, widget.bottom() - m.bottom),
        );
        // Keep a usable rect in tiny widgets.
        if axis_rect.width() < 10.0 {
            axis_rect.max.x = axis_rect.min.x + 10.0;
        }
        if axis_rect.height() < 10.0 {
            axis_rect.max.y = axis_rect.min.y + 10.0;
        }
        let axis_rect = round_rect(painter, axis_rect);
        Self {
            widget,
            axis_rect,
            margins: m,
            x,
            left,
            right,
            x_label,
            left_label,
            right_label,
        }
    }
}

fn round_rect(painter: &Painter, r: Rect) -> Rect {
    let ppp = painter.pixels_per_point();
    let f = |v: f32| (v * ppp).round() / ppp;
    Rect::from_min_max(pos2(f(r.min.x), f(r.min.y)), pos2(f(r.max.x), f(r.max.y)))
}

/// Transforms for the left and right axes.
pub(crate) fn transforms(
    rect: Rect,
    x: (f64, f64),
    left: (f64, f64),
    right: (f64, f64),
) -> (PlotTransform, PlotTransform) {
    let tl = PlotTransform::new(
        rect,
        PlotBounds::from_min_max([x.0, left.0], [x.1, left.1]),
        false,
    );
    let tr = PlotTransform::new(
        rect,
        PlotBounds::from_min_max([x.0, right.0], [x.1, right.1]),
        false,
    );
    (tl, tr)
}

/// One-pixel-wide dotted line as a mesh of pixel squares (Qt's cosmetic `DotLine`:
/// one pixel on, two off).
fn dotted(mesh: &mut Mesh, a: Pos2, b: Pos2, ppp: f32, color: Color32) {
    let px = 1.0 / ppp;
    if (a.x - b.x).abs() < 1e-3 {
        let (y0, y1) = (a.y.min(b.y), a.y.max(b.y));
        let mut y = y0;
        while y < y1 {
            mesh.add_colored_rect(
                Rect::from_min_size(pos2(a.x - 0.5 * px, y), vec2(px, px)),
                color,
            );
            y += 3.0 * px;
        }
    } else {
        let (x0, x1) = (a.x.min(b.x), a.x.max(b.x));
        let mut x = x0;
        while x < x1 {
            mesh.add_colored_rect(
                Rect::from_min_size(pos2(x, a.y - 0.5 * px), vec2(px, px)),
                color,
            );
            x += 3.0 * px;
        }
    }
}

/// Grid lines of the X and left axes at major ticks, the line at zero solid.
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPGrid::drawGridLines.
pub(crate) fn paint_grid(painter: &Painter, lay: &Layout, tl: &PlotTransform, spec: &PlotSpec<'_>) {
    let ppp = painter.pixels_per_point();
    let r = lay.axis_rect;
    let mut mesh = Mesh::default();
    let zero = Stroke::new(1.0 / ppp, spec.style.zero_line);
    let b = tl.bounds();
    let x_eps = (b.max()[0] - b.min()[0]) * 1e-6;
    let y_eps = (b.max()[1] - b.min()[1]) * 1e-6;
    let x_has_zero = b.min()[0] < 0.0 && b.max()[0] > 0.0;
    let y_has_zero = b.min()[1] < 0.0 && b.max()[1] > 0.0;
    for &t in &lay.x.ticks.major {
        let x = painter.round_to_pixel_center(tl.position_from_point_x(t));
        if x_has_zero && t.abs() < x_eps {
            painter.line_segment([pos2(x, r.bottom()), pos2(x, r.top())], zero);
        } else {
            dotted(
                &mut mesh,
                pos2(x, r.top()),
                pos2(x, r.bottom()),
                ppp,
                spec.style.grid,
            );
        }
    }
    for &t in &lay.left.ticks.major {
        let y = painter.round_to_pixel_center(tl.position_from_point_y(t));
        if y_has_zero && t.abs() < y_eps {
            painter.line_segment([pos2(r.left(), y), pos2(r.right(), y)], zero);
        } else {
            dotted(
                &mut mesh,
                pos2(r.left(), y),
                pos2(r.right(), y),
                ppp,
                spec.style.grid,
            );
        }
    }
    painter.add(Shape::mesh(mesh));
}

/// Axis lines, ticks, sub-ticks, tick labels and axis labels.
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPAxisPainterPrivate::draw
/// (geometry only; Qt's text rendering differs).
pub(crate) fn paint_axes(
    painter: &Painter,
    lay: &Layout,
    tl: &PlotTransform,
    tr: &PlotTransform,
    spec: &PlotSpec<'_>,
) {
    let ppp = painter.pixels_per_point();
    let color = spec.style.axis;
    let stroke = Stroke::new(1.0 / ppp, color);
    let r = lay.axis_rect;
    let snap = |v: f32| painter.round_to_pixel_center(v);
    let (left_x, right_x, bottom_y) = (snap(r.left()), snap(r.right()), snap(r.bottom()));

    // Bottom axis.
    painter.line_segment(
        [pos2(left_x, bottom_y), pos2(snap(r.right()), bottom_y)],
        stroke,
    );
    for &t in &lay.x.ticks.minor {
        let x = snap(tl.position_from_point_x(t));
        painter.line_segment([pos2(x, bottom_y), pos2(x, bottom_y - SUBTICK_IN)], stroke);
    }
    for (t, g) in lay.x.ticks.major.iter().zip(&lay.x.galleys) {
        let x = snap(tl.position_from_point_x(*t));
        painter.line_segment([pos2(x, bottom_y), pos2(x, bottom_y - TICK_IN)], stroke);
        let pos = pos2(x - g.size().x * 0.5, r.bottom() + BOTTOM_TICK_LABEL_PAD);
        painter.galley(pos, g.clone(), color);
    }
    if let Some(g) = &lay.x_label {
        let y = r.bottom() + BOTTOM_TICK_LABEL_PAD + lay.x.max_h + BOTTOM_LABEL_PAD;
        // Laid out centred: the galley's rows are centred on the given x.
        painter.galley(pos2(r.center().x, y), g.clone(), color);
    }

    // Left axis.
    painter.line_segment(
        [pos2(left_x, snap(r.top())), pos2(left_x, bottom_y)],
        stroke,
    );
    vertical_ticks(painter, &lay.left, tl, left_x, 1.0, stroke);
    for (t, g) in lay.left.ticks.major.iter().zip(&lay.left.galleys) {
        let y = tl.position_from_point_y(*t);
        let pos = pos2(
            r.left() - LEFT_TICK_LABEL_PAD - g.size().x,
            y - g.size().y * 0.5,
        );
        painter.galley(pos, g.clone(), color);
    }
    if let Some(g) = &lay.left_label {
        // Rotated -90 degrees, centred on the axis, outside the tick labels.
        let x = r.left() - LEFT_TICK_LABEL_PAD - lay.left.max_w - LEFT_LABEL_PAD - g.size().y;
        let y = r.center().y + g.size().x * 0.5;
        painter.add(
            egui::epaint::TextShape::new(pos2(x, y), g.clone(), color)
                .with_angle(-std::f32::consts::FRAC_PI_2),
        );
    }

    // Right axis.
    if let Some(ax) = &lay.right {
        painter.line_segment(
            [pos2(right_x, snap(r.top())), pos2(right_x, bottom_y)],
            stroke,
        );
        vertical_ticks(painter, ax, tr, right_x, -1.0, stroke);
        for (t, g) in ax.ticks.major.iter().zip(&ax.galleys) {
            let y = tr.position_from_point_y(*t);
            let pos = pos2(r.right() + RIGHT_TICK_LABEL_PAD, y - g.size().y * 0.5);
            painter.galley(pos, g.clone(), color);
        }
        if let Some(g) = &lay.right_label {
            // Rotated +90 degrees: reads top to bottom.
            let x = r.right() + RIGHT_TICK_LABEL_PAD + ax.max_w + RIGHT_LABEL_PAD + g.size().y;
            let y = r.center().y - g.size().x * 0.5;
            painter.add(
                egui::epaint::TextShape::new(pos2(x, y), g.clone(), color)
                    .with_angle(std::f32::consts::FRAC_PI_2),
            );
        }
    }
}

fn vertical_ticks(
    painter: &Painter,
    ax: &AxisLayout,
    tf: &PlotTransform,
    x: f32,
    dir: f32,
    stroke: Stroke,
) {
    for &t in &ax.ticks.minor {
        let y = painter.round_to_pixel_center(tf.position_from_point_y(t));
        painter.line_segment([pos2(x, y), pos2(x + dir * SUBTICK_IN, y)], stroke);
    }
    for &t in &ax.ticks.major {
        let y = painter.round_to_pixel_center(tf.position_from_point_y(t));
        painter.line_segment([pos2(x, y), pos2(x + dir * TICK_IN, y)], stroke);
    }
}

/// Status, interpolation and event rows as one mesh: runs of equal values merged,
/// runs thinner than a pixel folded into the pixel they share.
pub(crate) fn band_shapes(
    rows: &[BandRow<'_>],
    tl: &PlotTransform,
    frames: &super::FrameAxis,
    ppp: f32,
) -> Option<Shape> {
    let mut mesh = Mesh::default();
    let b = tl.bounds();
    let (x0, x1) = (b.min()[0], b.max()[0]);
    let px = 1.0 / ppp;
    for row in rows {
        let n = row.values.len();
        if n == 0 || row.colors.is_empty() {
            continue;
        }
        let ya = tl.position_from_point_y(row.y.0);
        let yb = tl.position_from_point_y(row.y.1);
        let (mut y0, mut y1) = (ya.min(yb), ya.max(yb));
        if row.outline {
            // The C++ rectangles have a one-pixel pen in the fill colour.
            y0 -= 0.5 * px;
            y1 += 0.5 * px;
        }
        let x_of_cell = |i: usize| frames.cell(row.first + i);
        let (i0, i1) = visible_window(n, |i| x_of_cell(i).1, x0, x1);
        let mut last_end = f32::NEG_INFINITY;
        let mut i = i0;
        while i < i1 {
            let v = row.values[i];
            let start = i;
            while i < i1 && row.values[i] == v {
                i += 1;
            }
            let color = row.colors[v as usize % row.colors.len()];
            if color.a() == 0 {
                continue;
            }
            let mut xa = tl.position_from_point_x(x_of_cell(start).0);
            let mut xb = tl.position_from_point_x(x_of_cell(i - 1).1);
            if row.outline {
                xa -= 0.5 * px;
                xb += 0.5 * px;
            }
            if xb <= last_end {
                continue;
            }
            let xa = xa.max(last_end);
            let xb = xb.max(xa + px);
            mesh.add_colored_rect(Rect::from_min_max(pos2(xa, y0), pos2(xb, y1)), color);
            last_end = xb;
        }
    }
    (!mesh.is_empty()).then(|| Shape::mesh(mesh))
}

/// The C++ cursor line (`QCPItemLine`).
pub(crate) fn cursor_shape(c: &super::Cursor, tl: &PlotTransform, painter: &Painter) -> Shape {
    let r = *tl.frame();
    let x = painter.round_to_pixel_center(tl.position_from_point_x(c.x));
    let (ya, yb) = match c.y {
        Some((a, b)) => (tl.position_from_point_y(a), tl.position_from_point_y(b)),
        None => (r.bottom(), r.top()),
    };
    Shape::line_segment(
        [pos2(x, ya), pos2(x, yb)],
        Stroke::new(c.pen.width, c.pen.color),
    )
}

/// The C++ selection rectangle (`QCPItemRect`): brush and a one-pixel pen.
pub(crate) fn selection_shapes(
    s: &super::SelectionRect,
    tl: &PlotTransform,
    painter: &Painter,
    out: &mut Vec<Shape>,
) {
    let r = *tl.frame();
    let xa = tl.position_from_point_x(s.x.0);
    let xb = tl.position_from_point_x(s.x.1);
    let (ya, yb) = match s.y {
        Some((a, b)) => (tl.position_from_point_y(a), tl.position_from_point_y(b)),
        None => (r.bottom(), r.top()),
    };
    let rect = Rect::from_two_pos(pos2(xa, ya), pos2(xb, yb));
    out.push(Shape::rect_filled(rect, 0.0, s.fill));
    let px = 1.0 / painter.pixels_per_point();
    out.push(Shape::rect_stroke(
        rect,
        0.0,
        Stroke::new(px, s.pen),
        egui::StrokeKind::Middle,
    ));
}

/// Statistics of one paint pass, for the benchmark.
#[derive(Clone, Copy, Debug, Default)]
pub struct PaintStats {
    /// Points handed to line items after decimation.
    pub line_points: usize,
    /// Scatter symbols drawn after pixel de-duplication.
    pub symbols: usize,
    /// Error bars drawn (per sample or per pixel column).
    pub error_bars: usize,
}

/// What the shapes of one series depend on besides its samples: everything
/// [`series_shapes`] and the tessellator read.
#[derive(Clone, Debug, PartialEq)]
struct SeriesKey {
    /// [`PlotSpec::generation`]: the caller's promise that the samples are the same.
    generation: u64,
    /// Position of the series in [`PlotSpec::series`].
    index: usize,
    frames: super::FrameAxis,
    /// X, left and right ranges.
    ranges: [(f64, f64); 3],
    axis_rect: Rect,
    /// The `Ui`'s clip rect (egui_plot's lines cut their runs to it).
    ui_clip: Rect,
    /// The painter's clip rect, which the tessellator culls against.
    paint_clip: Rect,
    pixels_per_point: f32,
    options: egui::epaint::TessellationOptions,
    dark_mode: bool,
    axis: YAxis,
    line: LineStyle,
    scatter: super::Scatter,
    pen: Pen,
    errors: Option<(Pen, ErrorStyle, usize)>,
    len: usize,
}

/// One series as last drawn: its shapes tessellated, and its share of the statistics.
#[derive(Clone, Debug)]
struct CachedSeries {
    key: SeriesKey,
    mesh: Option<Arc<Mesh>>,
    stats: PaintStats,
}

/// The shapes of each series as last drawn, kept in [`super::PlotView`] so that a repaint
/// with nothing changed emits them without building and tessellating them again.
///
/// Only used for a spec with a [`PlotSpec::generation`]. The shapes are tessellated with
/// egui's own tessellator, with the options, pixels per point and culling rectangle egui
/// uses for them, and handed to egui as one mesh per series: egui then appends that mesh
/// where it would have appended the tessellation of each shape, so the GPU receives the
/// same vertices and indices (`tests/review13_plot.rs` compares them). Never part of the
/// view's equality.
#[derive(Clone, Default)]
pub(crate) struct ShapeCache {
    series: Vec<Option<CachedSeries>>,
}

impl PartialEq for ShapeCache {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for ShapeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShapeCache")
            .field("series", &self.series.len())
            .finish()
    }
}

/// The view ranges and the painter a series is drawn with.
pub(crate) struct SeriesTarget<'p> {
    pub tl: &'p PlotTransform,
    pub tr: &'p PlotTransform,
    pub ranges: [(f64, f64); 3],
    pub painter: &'p Painter,
}

impl ShapeCache {
    /// The series of `spec` into `shapes`: from the cache where nothing they depend on
    /// changed, else built, tessellated and kept. Without a generation, or when the painter
    /// fades or hides what it paints (egui would recolour the shapes before tessellating
    /// them), the shapes are emitted as [`series_shapes`] builds them and nothing is kept.
    pub(crate) fn series(
        &mut self,
        ui: &Ui,
        spec: &PlotSpec<'_>,
        target: &SeriesTarget<'_>,
        shapes: &mut Vec<Shape>,
        stats: &mut PaintStats,
    ) {
        let painter = target.painter;
        let Some(generation) = spec
            .generation
            .filter(|_| painter.opacity() == 1.0 && painter.is_visible())
        else {
            self.series.clear();
            for s in &spec.series {
                series_shapes(ui, s, target.tl, target.tr, &spec.frames, shapes, stats);
            }
            return;
        };
        let ppp = ui.ctx().pixels_per_point();
        let options = ui.ctx().tessellation_options(|o| *o);
        self.series.resize(spec.series.len(), None);
        for (index, (s, slot)) in spec.series.iter().zip(&mut self.series).enumerate() {
            let key = SeriesKey {
                generation,
                index,
                frames: spec.frames,
                ranges: target.ranges,
                axis_rect: *target.tl.frame(),
                ui_clip: ui.clip_rect(),
                paint_clip: painter.clip_rect(),
                pixels_per_point: ppp,
                options,
                dark_mode: ui.visuals().dark_mode,
                axis: s.axis,
                line: s.line,
                scatter: s.scatter,
                pen: s.pen,
                errors: s.errors.map(|e| (e.pen, e.style, e.errors.len())),
                len: s.data.len(),
            };
            if slot.as_ref().is_none_or(|c| c.key != key) {
                let mut own = Vec::new();
                let mut own_stats = PaintStats::default();
                series_shapes(
                    ui,
                    s,
                    target.tl,
                    target.tr,
                    &spec.frames,
                    &mut own,
                    &mut own_stats,
                );
                // Text is never among them, and their circles are not filled, so the font
                // texture's size and the prepared discs play no part.
                let mut tess = egui::epaint::Tessellator::new(ppp, options, [1, 1], Vec::new());
                tess.set_clip_rect(key.paint_clip);
                let mut mesh = Mesh::default();
                for shape in own {
                    tess.tessellate_shape(shape, &mut mesh);
                }
                *slot = Some(CachedSeries {
                    key,
                    mesh: (!mesh.is_empty()).then(|| Arc::new(mesh)),
                    stats: own_stats,
                });
            }
            let c = slot.as_ref().expect("filled above");
            if let Some(m) = &c.mesh {
                shapes.push(Shape::Mesh(m.clone()));
            }
            stats.line_points += c.stats.line_points;
            stats.symbols += c.stats.symbols;
            stats.error_bars += c.stats.error_bars;
        }
    }
}

/// Lines, symbols and error bars of one series.
pub(crate) fn series_shapes(
    ui: &Ui,
    s: &Series<'_>,
    tl: &PlotTransform,
    tr: &PlotTransform,
    frames: &super::FrameAxis,
    shapes: &mut Vec<Shape>,
    stats: &mut PaintStats,
) {
    let tf = if s.axis == YAxis::Left { tl } else { tr };
    let ppp = ui.ctx().pixels_per_point();
    let frame = *tf.frame();
    let cols = ((frame.width() * ppp) as usize).max(1);
    let b = tf.bounds();
    let (x0, x1) = (b.min()[0], b.max()[0]);
    let n = s.data.len();
    let xf = |i: usize| s.data.x(i, frames);
    let yf = |i: usize| s.data.y(i);
    let sorted = match s.data {
        super::SeriesData::Frames { .. } => true,
        super::SeriesData::Xy { x, .. } => x.windows(2).all(|w| w[0] <= w[1]),
    };
    let win = if sorted {
        visible_window(n, xf, x0, x1)
    } else {
        (0, n)
    };
    let dense = sorted && win.1 - win.0 > 2 * cols;

    if s.line != LineStyle::None && s.pen.color.a() > 0 {
        let runs = decimate(
            win,
            xf,
            yf,
            (x0, x1),
            if sorted { cols } else { usize::MAX / 4 },
        );
        for run in runs {
            let pts = if s.line == LineStyle::StepCenter && !dense {
                step_center(&run)
            } else {
                run
            };
            stats.line_points += pts.len();
            line_shapes(ui, s.name, &pts, s.pen, dense, tf, shapes);
        }
    }

    if s.scatter.shape != ScatterShape::None && s.pen.color.a() > 0 {
        let clip = frame.expand(s.scatter.size + 2.0);
        let mut pos: Vec<Pos2> = (win.0..win.1)
            .filter_map(|i| {
                let v = yf(i);
                if v.is_nan() {
                    return None;
                }
                let p = tf.position_from_point(&PlotPoint::new(xf(i), v));
                clip.contains(p).then_some(p)
            })
            .collect();
        dedupe_pixels(&mut pos, ppp);
        if pos.len() > MESH_SYMBOLS {
            // Dense: symbols closer than half their size apart merge into one
            // band anyway; keep one per half-size cell.
            dedupe_pixels(&mut pos, (2.0 / s.scatter.size.max(1.0)).min(ppp));
        }
        stats.symbols += pos.len();
        symbol_shapes(&pos, s.scatter.shape, s.scatter.size, s.pen, ppp, shapes);
    }

    if let Some(e) = &s.errors {
        error_shapes(s, e, tf, frames, win, dense, cols, shapes, stats);
    }
}

fn line_shapes(
    ui: &Ui,
    name: &str,
    pts: &[[f64; 2]],
    pen: Pen,
    dense: bool,
    tf: &PlotTransform,
    shapes: &mut Vec<Shape>,
) {
    if pts.is_empty() {
        return;
    }
    let stroke = Stroke::new(pen.width, pen.color);
    match pen.dash {
        Dash::Solid => {
            let points: Vec<PlotPoint> = pts.iter().map(|p| PlotPoint::new(p[0], p[1])).collect();
            let mut line = Line::new(name, PlotPoints::Owned(points))
                .stroke(stroke)
                // egui_plot's per-pixel band only for decimated data: it would also
                // fill the vertical joins of zoomed-in steps.
                .allow_band(dense)
                .allow_hover(false);
            line.initialize(f64::NEG_INFINITY..=f64::INFINITY);
            line.shapes(ui, tf, shapes);
        }
        Dash::Dash | Dash::Dot => {
            let screen: Vec<Pos2> = pts
                .iter()
                .map(|p| tf.position_from_point(&PlotPoint::new(p[0], p[1])))
                .collect();
            let w = pen.width.max(1.0);
            let (on, off) = if pen.dash == Dash::Dash {
                (4.0 * w, 2.0 * w)
            } else {
                (w, 2.0 * w)
            };
            Shape::dashed_line_many(&screen, stroke, on, off, shapes);
        }
    }
}

/// Above this many symbols per series they go into one mesh.
const MESH_SYMBOLS: usize = 1500;

/// A line segment `a`-`b` of half-width `hw` as a quad.
fn quad(mesh: &mut Mesh, a: Pos2, b: Pos2, hw: f32, color: Color32) {
    let d = (b - a).normalized();
    let n = vec2(-d.y, d.x) * hw;
    let i = mesh.vertices.len() as u32;
    mesh.colored_vertex(a + n, color);
    mesh.colored_vertex(a - n, color);
    mesh.colored_vertex(b + n, color);
    mesh.colored_vertex(b - n, color);
    mesh.add_triangle(i, i + 1, i + 2);
    mesh.add_triangle(i + 1, i + 2, i + 3);
}

/// A circle outline of radius `r` and half-width `hw` as an octagonal ring.
fn ring(mesh: &mut Mesh, c: Pos2, r: f32, hw: f32, color: Color32) {
    const N: u32 = 8;
    let i0 = mesh.vertices.len() as u32;
    for k in 0..N {
        let a = k as f32 * std::f32::consts::TAU / N as f32;
        let d = vec2(a.cos(), a.sin());
        mesh.colored_vertex(c + d * (r + hw), color);
        mesh.colored_vertex(c + d * (r - hw).max(0.0), color);
    }
    for k in 0..N {
        let (o0, o1) = (i0 + 2 * k, i0 + 2 * ((k + 1) % N));
        mesh.add_triangle(o0, o0 + 1, o1);
        mesh.add_triangle(o0 + 1, o1, o1 + 1);
    }
}

/// QCustomPlot's scatter shapes.
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPScatterStyle::drawShape
/// (ssDot, ssCross, ssCircle).
fn symbol_shapes(
    pos: &[Pos2],
    shape: ScatterShape,
    size: f32,
    pen: Pen,
    ppp: f32,
    shapes: &mut Vec<Shape>,
) {
    let stroke = Stroke::new(pen.width, pen.color);
    let w = size * 0.5;
    match shape {
        ScatterShape::None => {}
        ScatterShape::Dot => {
            let px = 1.0 / ppp;
            let mut mesh = Mesh::default();
            for p in pos {
                mesh.add_colored_rect(
                    Rect::from_center_size(*p, vec2(px.max(pen.width), px.max(pen.width))),
                    pen.color,
                );
            }
            shapes.push(Shape::mesh(mesh));
        }
        ScatterShape::Cross | ScatterShape::Circle if pos.len() > MESH_SYMBOLS => {
            // Many symbols: one mesh without anti-aliasing, instead of thousands
            // of tessellated shapes (they overlap at this density anyway).
            let hw = pen.width.max(1.0 / ppp) * 0.5;
            let mut mesh = Mesh::default();
            for p in pos {
                if shape == ScatterShape::Cross {
                    quad(
                        &mut mesh,
                        pos2(p.x - w, p.y - w),
                        pos2(p.x + w, p.y + w),
                        hw,
                        pen.color,
                    );
                    quad(
                        &mut mesh,
                        pos2(p.x - w, p.y + w),
                        pos2(p.x + w, p.y - w),
                        hw,
                        pen.color,
                    );
                } else {
                    ring(&mut mesh, *p, w, hw, pen.color);
                }
            }
            shapes.push(Shape::mesh(mesh));
        }
        ScatterShape::Cross => {
            for p in pos {
                shapes.push(Shape::line_segment(
                    [pos2(p.x - w, p.y - w), pos2(p.x + w, p.y + w)],
                    stroke,
                ));
                shapes.push(Shape::line_segment(
                    [pos2(p.x - w, p.y + w), pos2(p.x + w, p.y - w)],
                    stroke,
                ));
            }
        }
        ScatterShape::Circle => {
            for p in pos {
                shapes.push(Shape::circle_stroke(*p, w, stroke));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn error_shapes(
    s: &Series<'_>,
    e: &super::ErrorBars<'_>,
    tf: &PlotTransform,
    frames: &super::FrameAxis,
    win: (usize, usize),
    dense: bool,
    cols: usize,
    shapes: &mut Vec<Shape>,
    stats: &mut PaintStats,
) {
    let stroke = Stroke::new(e.pen.width, e.pen.color);
    let frame = *tf.frame();
    let x0 = frame.left();
    let col_w = frame.width() / cols as f32;
    // (screen x, value, error) of drawable samples
    let samples = (win.0..win.1).filter_map(|i| {
        let v = s.data.y(i);
        let err = e.errors.get(i).copied().unwrap_or(f64::NAN);
        if v.is_nan() || err.is_nan() {
            return None;
        }
        Some((tf.position_from_point_x(s.data.x(i, frames)), v, err))
    });
    match e.style {
        ErrorStyle::Bars {
            whisker_width,
            symbol_gap,
        } => {
            let ww = whisker_width * 0.5;
            if !dense {
                // Ported from external/QCustomPlot/qcustomplot.cpp, QCPErrorBars::getErrorBarLines.
                let gap = symbol_gap * 0.5;
                for (x, v, err) in samples {
                    let cy = tf.position_from_point_y(v);
                    let plus = tf.position_from_point_y(v + err);
                    if cy - gap > plus {
                        shapes.push(Shape::line_segment(
                            [pos2(x, cy - gap), pos2(x, plus)],
                            stroke,
                        ));
                    }
                    shapes.push(Shape::line_segment(
                        [pos2(x - ww, plus), pos2(x + ww, plus)],
                        stroke,
                    ));
                    let minus = tf.position_from_point_y(v - err);
                    if cy + gap < minus {
                        shapes.push(Shape::line_segment(
                            [pos2(x, cy + gap), pos2(x, minus)],
                            stroke,
                        ));
                    }
                    shapes.push(Shape::line_segment(
                        [pos2(x - ww, minus), pos2(x + ww, minus)],
                        stroke,
                    ));
                    stats.error_bars += 1;
                }
            } else {
                // More bars than pixel columns: one backbone per column from the
                // lowest to the highest end, whiskers at those two ends.
                for (x, lo, hi) in per_column(samples, x0, col_w) {
                    let (ya, yb) = (tf.position_from_point_y(lo), tf.position_from_point_y(hi));
                    shapes.push(Shape::line_segment([pos2(x, ya), pos2(x, yb)], stroke));
                    shapes.push(Shape::line_segment(
                        [pos2(x - ww, ya), pos2(x + ww, ya)],
                        stroke,
                    ));
                    shapes.push(Shape::line_segment(
                        [pos2(x - ww, yb), pos2(x + ww, yb)],
                        stroke,
                    ));
                    stats.error_bars += 1;
                }
            }
        }
    }
}

fn per_column(
    samples: impl Iterator<Item = (f32, f64, f64)>,
    x0: f32,
    col_w: f32,
) -> Vec<(f32, f64, f64)> {
    let mut out: Vec<(f32, f64, f64)> = Vec::new();
    let mut cur: Option<(i64, f32, f64, f64)> = None;
    for (x, v, err) in samples {
        let c = ((x - x0) / col_w).floor() as i64;
        let (lo, hi) = (v - err.abs(), v + err.abs());
        match &mut cur {
            Some((cc, _, l, h)) if *cc == c => {
                *l = l.min(lo);
                *h = h.max(hi);
            }
            _ => {
                if let Some((_, cx, l, h)) = cur.take() {
                    out.push((cx, l, h));
                }
                cur = Some((c, x0 + (c as f32 + 0.5) * col_w, lo, hi));
            }
        }
    }
    if let Some((_, cx, l, h)) = cur {
        out.push((cx, l, h));
    }
    out
}
