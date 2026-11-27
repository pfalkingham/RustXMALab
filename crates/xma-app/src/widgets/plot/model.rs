//! Plain-data description of one plot: what `PlotWindow.cpp` hands to QCustomPlot.
//!
//! Everything here borrows the caller's data for one frame of the UI; nothing
//! is copied until it is drawn, and then only the visible, decimated part.

use egui::Color32;

/// Qt's named colours, as `PlotWindow.cpp` uses them (`Qt::red`, `Qt::darkBlue`, ...).
pub mod qt {
    use egui::Color32;

    /// `Qt::black`.
    pub const BLACK: Color32 = Color32::from_rgb(0, 0, 0);
    /// `Qt::white`.
    pub const WHITE: Color32 = Color32::from_rgb(255, 255, 255);
    /// `Qt::darkGray`.
    pub const DARK_GRAY: Color32 = Color32::from_rgb(128, 128, 128);
    /// `Qt::gray`.
    pub const GRAY: Color32 = Color32::from_rgb(160, 160, 164);
    /// `Qt::lightGray`.
    pub const LIGHT_GRAY: Color32 = Color32::from_rgb(192, 192, 192);
    /// `Qt::red`.
    pub const RED: Color32 = Color32::from_rgb(255, 0, 0);
    /// `Qt::green`.
    pub const GREEN: Color32 = Color32::from_rgb(0, 255, 0);
    /// `Qt::blue`.
    pub const BLUE: Color32 = Color32::from_rgb(0, 0, 255);
    /// `Qt::cyan`.
    pub const CYAN: Color32 = Color32::from_rgb(0, 255, 255);
    /// `Qt::magenta`.
    pub const MAGENTA: Color32 = Color32::from_rgb(255, 0, 255);
    /// `Qt::yellow`.
    pub const YELLOW: Color32 = Color32::from_rgb(255, 255, 0);
    /// `Qt::darkRed`.
    pub const DARK_RED: Color32 = Color32::from_rgb(128, 0, 0);
    /// `Qt::darkGreen`.
    pub const DARK_GREEN: Color32 = Color32::from_rgb(0, 128, 0);
    /// `Qt::darkBlue`.
    pub const DARK_BLUE: Color32 = Color32::from_rgb(0, 0, 128);
    /// `Qt::darkCyan`.
    pub const DARK_CYAN: Color32 = Color32::from_rgb(0, 128, 128);
    /// `Qt::darkMagenta`.
    pub const DARK_MAGENTA: Color32 = Color32::from_rgb(128, 0, 128);
    /// `Qt::darkYellow`.
    pub const DARK_YELLOW: Color32 = Color32::from_rgb(128, 128, 0);

    /// `Qt::GlobalColor(n)` for n in 0..=19, as `QColor` resolves it
    /// (`color0`/`color1` are black/white, 19 is transparent).
    ///
    /// `PlotWindow::plotRigidBody` colours its k-th graph with `Qt::GlobalColor(k + 7)`.
    pub fn global_color(n: usize) -> Color32 {
        const TABLE: [Color32; 20] = [
            BLACK,
            WHITE,
            BLACK,
            WHITE,
            DARK_GRAY,
            GRAY,
            LIGHT_GRAY,
            RED,
            GREEN,
            BLUE,
            CYAN,
            MAGENTA,
            YELLOW,
            DARK_RED,
            DARK_GREEN,
            DARK_BLUE,
            DARK_CYAN,
            DARK_MAGENTA,
            DARK_YELLOW,
            Color32::TRANSPARENT,
        ];
        TABLE.get(n).copied().unwrap_or(BLACK)
    }
}

/// Which Y axis a series or decoration is plotted against
/// (QCustomPlot's `yAxis` on the left, `yAxis2` on the right).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum YAxis {
    /// `yAxis`.
    #[default]
    Left,
    /// `yAxis2`.
    Right,
}

/// Qt pen styles used by `PlotWindow.cpp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Dash {
    /// `Qt::SolidLine`.
    #[default]
    Solid,
    /// `Qt::DashLine`: dashes of 4 and gaps of 2 pen widths.
    Dash,
    /// `Qt::DotLine`: dots of 1 and gaps of 2 pen widths.
    Dot,
}

/// A pen: colour, width in points, dash pattern.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pen {
    /// Colour, alpha included.
    pub color: Color32,
    /// Width in points (`QPen(color)` is 1).
    pub width: f32,
    /// Dash pattern.
    pub dash: Dash,
}

impl Pen {
    /// `QPen(color)`: solid, width 1.
    pub const fn solid(color: Color32) -> Self {
        Self {
            color,
            width: 1.0,
            dash: Dash::Solid,
        }
    }

    /// A width-1 pen with a dash pattern.
    pub const fn dashed(color: Color32, dash: Dash) -> Self {
        Self {
            color,
            width: 1.0,
            dash,
        }
    }
}

/// How consecutive points of a series are joined (`QCPGraph::LineStyle`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LineStyle {
    /// `lsNone`: scatter symbols only.
    None,
    /// `lsLine`: straight lines.
    #[default]
    Line,
    /// `lsStepCenter`: horizontal steps centred on each point, vertical joins half-way.
    StepCenter,
}

/// Scatter symbol shape (`QCPScatterStyle::ScatterShape`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScatterShape {
    /// `ssNone`.
    #[default]
    None,
    /// `ssDot`: one pixel.
    Dot,
    /// `ssCross`: a diagonal cross spanning `size` in x and y.
    Cross,
    /// `ssCircle`: a circle outline of diameter `size`.
    Circle,
}

/// A scatter symbol: shape and size in points (`QCPScatterStyle(shape, size)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scatter {
    /// The shape.
    pub shape: ScatterShape,
    /// Size in points; XMALab uses 2 for crosses and circles and 1 for dots.
    pub size: f32,
}

impl Scatter {
    /// No symbols.
    pub const NONE: Self = Self {
        shape: ScatterShape::None,
        size: 0.0,
    };

    /// `QCPScatterStyle(shape, size)`.
    pub const fn new(shape: ScatterShape, size: f32) -> Self {
        Self { shape, size }
    }
}

/// Samples of a series.
///
/// A NaN value means "no sample": lines break there, and no symbol or error bar
/// is drawn. (XMALab itself sometimes plots sentinel values such as
/// -2 instead; pass them as they are to reproduce that.)
#[derive(Clone, Copy, Debug)]
pub enum SeriesData<'a> {
    /// One value per frame, starting at 0-based frame `first`; x is the frame's
    /// position on the frame axis ([`FrameAxis::x_of`]).
    Frames {
        /// 0-based frame of `values[0]`.
        first: usize,
        /// One value per frame.
        values: &'a [f64],
    },
    /// Explicit x and y (same length; x ascending for decimation to apply).
    Xy {
        /// X coordinates in plot units.
        x: &'a [f64],
        /// Y coordinates in plot units.
        y: &'a [f64],
    },
}

impl SeriesData<'_> {
    /// Number of samples.
    pub fn len(&self) -> usize {
        match self {
            Self::Frames { values, .. } => values.len(),
            Self::Xy { x, y } => x.len().min(y.len()),
        }
    }

    /// Whether there are no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Value of sample `i`.
    pub fn y(&self, i: usize) -> f64 {
        match self {
            Self::Frames { values, .. } => values[i],
            Self::Xy { y, .. } => y[i],
        }
    }

    /// Plot x of sample `i`.
    pub fn x(&self, i: usize, axis: &FrameAxis) -> f64 {
        match self {
            Self::Frames { first, .. } => axis.x_of(first + i),
            Self::Xy { x, .. } => x[i],
        }
    }
}

/// How an error range is drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ErrorStyle {
    /// `QCPErrorBars` with `etValueError`: a vertical backbone that leaves a
    /// gap of `symbol_gap` points around the data point, and a horizontal
    /// whisker `whisker_width` points wide at each end (QCustomPlot defaults 10 and 9).
    Bars {
        /// Whisker width in points.
        whisker_width: f32,
        /// Gap around the data point in points.
        symbol_gap: f32,
    },
}

impl ErrorStyle {
    /// QCustomPlot's defaults (whisker 9, gap 10).
    pub const QCP_BARS: Self = Self::Bars {
        whisker_width: 9.0,
        symbol_gap: 10.0,
    };
}

/// Symmetric errors for a series, one per sample (NaN: none).
#[derive(Clone, Copy, Debug)]
pub struct ErrorBars<'a> {
    /// One error per sample of the series.
    pub errors: &'a [f64],
    /// Pen of backbones and whiskers.
    pub pen: Pen,
    /// How the bars are drawn.
    pub style: ErrorStyle,
}

/// One data series (a `QCPGraph`, optionally with its `QCPErrorBars`).
#[derive(Clone, Copy, Debug)]
pub struct Series<'a> {
    /// Name, for debugging only (the C++ shows no legend).
    pub name: &'a str,
    /// The samples.
    pub data: SeriesData<'a>,
    /// Which axis the values belong to.
    pub axis: YAxis,
    /// Line style.
    pub line: LineStyle,
    /// Scatter symbols.
    pub scatter: Scatter,
    /// Pen for lines and symbols.
    pub pen: Pen,
    /// Error bars, drawn after the line and symbols. A series with
    /// `LineStyle::None` and no symbols draws only these, which lets a caller
    /// put error bars on their own layer, as QCustomPlot's z-order requires.
    pub errors: Option<ErrorBars<'a>>,
}

impl<'a> Series<'a> {
    /// A series with QCustomPlot's graph defaults: a solid line, no symbols.
    pub fn new(name: &'a str, data: SeriesData<'a>, pen: Pen) -> Self {
        Self {
            name,
            data,
            axis: YAxis::Left,
            line: LineStyle::Line,
            scatter: Scatter::NONE,
            pen,
            errors: None,
        }
    }

    /// Plot against the given axis.
    pub fn axis(mut self, axis: YAxis) -> Self {
        self.axis = axis;
        self
    }

    /// Set the line style.
    pub fn line(mut self, line: LineStyle) -> Self {
        self.line = line;
        self
    }

    /// Set the scatter symbol.
    pub fn scatter(mut self, shape: ScatterShape, size: f32) -> Self {
        self.scatter = Scatter::new(shape, size);
        self
    }

    /// Attach error bars.
    pub fn errors(mut self, errors: ErrorBars<'a>) -> Self {
        self.errors = Some(errors);
        self
    }
}

/// One row of per-frame coloured cells: tracking status per camera,
/// interpolation method, or an event (`QCPItemRect`s in `PlotWindow::drawStatus`
/// and `drawEvents`).
///
/// The row is placed in left-axis coordinates, so it zooms and pans with the
/// left axis, as the C++ rectangles do. [`super::ranging::status_rows`] and
/// [`super::ranging::event_rows`] compute XMALab's placement.
#[derive(Clone, Copy, Debug)]
pub struct BandRow<'a> {
    /// 0-based frame of `values[0]`.
    pub first: usize,
    /// A category per frame, an index into `colors`.
    pub values: &'a [u8],
    /// Colour per category; fully transparent categories are not drawn.
    pub colors: &'a [Color32],
    /// Bottom and top in left-axis coordinates.
    pub y: (f64, f64),
    /// Draw an outline in the fill colour, as the status rectangles do (their
    /// pen is their brush colour), which makes every cell at least one pixel
    /// wide. Events have a transparent pen.
    pub outline: bool,
}

/// The current-frame line (`frameMarker`, a `QCPItemLine` with a black pen).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cursor {
    /// Plot x of the line.
    pub x: f64,
    /// Bottom and top in left-axis coordinates; `None` spans the whole axis rect.
    pub y: Option<(f64, f64)>,
    /// Pen; QCustomPlot's default item pen is black, width 1.
    pub pen: Pen,
}

impl Cursor {
    /// A black line at `x` between `y` (left-axis coordinates).
    pub fn new(x: f64, y: Option<(f64, f64)>) -> Self {
        Self {
            x,
            y,
            pen: Pen::solid(qt::BLACK),
        }
    }
}

/// The highlighted frame range (`selectionMarker`, a `QCPItemRect`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectionRect {
    /// Left and right in plot x (either order).
    pub x: (f64, f64),
    /// Bottom and top in left-axis coordinates; `None` spans the whole axis rect.
    pub y: Option<(f64, f64)>,
    /// Brush; XMALab uses yellow with alpha 150.
    pub fill: Color32,
    /// Pen; XMALab uses yellow with alpha 50.
    pub pen: Color32,
}

impl SelectionRect {
    /// XMALab's colours (`PlotWindow.cpp:75-76`).
    pub fn new(x: (f64, f64), y: Option<(f64, f64)>) -> Self {
        Self {
            x,
            y,
            fill: Color32::from_rgba_unmultiplied(255, 255, 0, 150),
            pen: Color32::from_rgba_unmultiplied(255, 255, 0, 50),
        }
    }
}

/// Maps 0-based frames to plot x and back, in frames or in seconds.
///
/// `PlotWindow.cpp` plots frame `i` (0-based) at `i * posMultiplier + posOffset`,
/// with multiplier 1 and offset 1 in frame mode (so frame 0 is at x = 1, matching
/// the 1-based numbers on screen) and multiplier `1 / recordingSpeed` and offset 0
/// in time mode ("Display time" checked and a recording speed set).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameAxis {
    /// First frame of the trial, 0-based (`getStartFrame() - 1`).
    pub first: usize,
    /// Last frame of the trial, 0-based (`getEndFrame() - 1`).
    pub last: usize,
    /// Recording speed in frames per second when the axis shows time.
    pub fps: Option<f64>,
}

impl FrameAxis {
    /// Frame mode over `first..=last` (0-based).
    pub fn frames(first: usize, last: usize) -> Self {
        Self {
            first,
            last,
            fps: None,
        }
    }

    /// Whether the axis shows seconds (`checkBoxTime` checked and speed > 0).
    pub fn is_time(&self) -> bool {
        self.fps.is_some_and(|f| f > 0.0)
    }

    /// `posMultiplier`.
    pub fn multiplier(&self) -> f64 {
        match self.fps {
            Some(f) if f > 0.0 => 1.0 / f,
            _ => 1.0,
        }
    }

    /// `posOffset` as the plotting functions use it (1 in frame mode, 0 in time mode).
    pub fn offset(&self) -> f64 {
        if self.is_time() { 0.0 } else { 1.0 }
    }

    /// Plot x of a 0-based frame (may be outside the trial).
    ///
    /// Ported from ui/PlotWindow.cpp, `i * posMultiplier + posOffset` (e.g. PlotWindow::plot2D).
    pub fn x_of(&self, frame: usize) -> f64 {
        self.x_of_i(frame as i64)
    }

    /// [`Self::x_of`] for a frame that may be negative.
    pub fn x_of_i(&self, frame: i64) -> f64 {
        frame as f64 * self.multiplier() + self.offset()
    }

    /// The 0-based frame under plot x, as the mouse handlers compute it:
    /// `int frame = (x - posOffset) / posMultiplier + 0.5;` (truncation toward zero,
    /// so x a little left of frame 0 still gives frame 0).
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::eventFilter.
    pub fn frame_at(&self, x: f64) -> i64 {
        let v = (x - self.offset()) / self.multiplier() + 0.5;
        if v.is_nan() {
            return i64::MIN;
        }
        v as i64 // `as` truncates toward zero and saturates, like the C++ int conversion for sane values
    }

    /// Whether a 0-based frame is inside the trial.
    pub fn contains(&self, frame: i64) -> bool {
        frame >= self.first as i64 && frame <= self.last as i64
    }

    /// Left and right plot x of the cell of a 0-based frame in the status and
    /// event rows: `((f - 0.5 - posOffset) * posMultiplier, (f + 0.5 - posOffset) * posMultiplier)`
    /// with `f` 1-based and `posOffset` 1 in time mode, 0 in frame mode.
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::drawStatus.
    pub fn cell(&self, frame: usize) -> (f64, f64) {
        let f = frame as f64 + 1.0;
        let off = if self.is_time() { 1.0 } else { 0.0 };
        let m = self.multiplier();
        ((f - 0.5 - off) * m, (f + 0.5 - off) * m)
    }
}

/// Colours of the plot itself (not of the data). The defaults are QCustomPlot's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlotStyle {
    /// Background of the whole widget (QCustomPlot: white).
    pub background: Color32,
    /// Axis lines, ticks, tick labels and axis labels (QCustomPlot: black).
    pub axis: Color32,
    /// Grid lines, dotted (QCustomPlot: rgb 200, 200, 200).
    pub grid: Color32,
    /// The grid line at zero, solid (QCustomPlot: rgb 200, 200, 200).
    pub zero_line: Color32,
    /// Font size in points for tick and axis labels; `None` uses the UI's body font,
    /// as QCustomPlot uses the widget's font.
    pub font_size: Option<f32>,
}

impl Default for PlotStyle {
    fn default() -> Self {
        Self {
            background: qt::WHITE,
            axis: qt::BLACK,
            grid: Color32::from_rgb(200, 200, 200),
            zero_line: Color32::from_rgb(200, 200, 200),
            font_size: None,
        }
    }
}

impl PlotStyle {
    /// The plot's own colours from egui's current visuals, so that the plot reads in the
    /// light and the dark theme: the text-field background, axes in the text colour, the
    /// grid in the colour of egui's separators (the program's look is the toolkit's own,
    /// parity ledger U139; the data's colours are not changed).
    #[must_use]
    pub fn from_visuals(v: &egui::Visuals) -> Self {
        let grid = v.widgets.noninteractive.bg_stroke.color;
        Self {
            background: v.extreme_bg_color,
            axis: v.text_color(),
            grid,
            zero_line: grid,
            font_size: None,
        }
    }
}

/// Fixed left and right margins, in points, to line up two plots
/// (`plotWidgetExtra` copies `plotWidget`'s margins in `PlotWindow::updateExtraPlot`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Margins {
    /// Left margin.
    pub left: f32,
    /// Right margin.
    pub right: f32,
    /// Top margin.
    pub top: f32,
    /// Bottom margin.
    pub bottom: f32,
}

/// Everything one plot shows. Built by the caller each UI frame.
///
/// Draw order, bottom to top, follows the C++ item creation order: grid,
/// cursor, selection, bands (in order), series (in order), axes.
#[derive(Clone, Debug)]
pub struct PlotSpec<'a> {
    /// Frame-to-x mapping and the trial's frame range.
    pub frames: FrameAxis,
    /// X axis label; may contain `\n` (XMALab appends mean errors on a second line).
    pub x_label: &'a str,
    /// Left axis label.
    pub left_label: &'a str,
    /// Right axis label; `None` hides the right axis (`yAxis2->setVisible(false)`).
    pub right_label: Option<&'a str>,
    /// Data series, drawn in order.
    pub series: Vec<Series<'a>>,
    /// Status, interpolation and event rows, drawn in order before the series.
    pub bands: Vec<BandRow<'a>>,
    /// The current-frame line.
    pub cursor: Option<Cursor>,
    /// The highlighted frame range.
    pub selection: Option<SelectionRect>,
    /// Colours and font.
    pub style: PlotStyle,
    /// Fixed left and right margins instead of the automatic ones.
    pub fixed_lr_margins: Option<(f32, f32)>,
    /// The generation of the series' samples: when set, the widget keeps the shapes of each
    /// series in its [`super::PlotView`] and draws them again, without rebuilding them,
    /// while the generation, the series' place in [`Self::series`], its style, the frame
    /// axis, the ranges and the drawing area stay the same. The caller changes it whenever
    /// any series' samples (or error values) change. `None`: nothing is kept.
    pub generation: Option<u64>,
}

impl<'a> PlotSpec<'a> {
    /// An empty plot over the given frames with the given axis labels.
    pub fn new(frames: FrameAxis, x_label: &'a str, left_label: &'a str) -> Self {
        Self {
            frames,
            x_label,
            left_label,
            right_label: None,
            series: Vec::new(),
            bands: Vec::new(),
            cursor: None,
            selection: None,
            style: PlotStyle::default(),
            fixed_lr_margins: None,
            generation: None,
        }
    }
}
