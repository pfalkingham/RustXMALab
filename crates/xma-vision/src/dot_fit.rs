//! Fitting Gaussian dots to a small piece of a picture: a marker's dot width, several dots
//! fitted together, and two touching dots fitted apart. New in this program (XMALab has no
//! such method).
//!
//! Where two markers' dots touch in one camera's picture, the marker's detection thresholds
//! them into one shape and returns that shape's centre, which lies between the two dots. The
//! dots themselves usually stay apart, each with its own dark core (on the pig's example
//! project, marker 5 and a bead in camera 1 at frames 34 to 53: 4 to 5 px apart, each about
//! 1.4 px wide as a Gaussian). Here the picture near a dot is modelled as a plane (the
//! background) plus one Gaussian dot per bead:
//!
//! `v(x, y) = b0 + bx (x - mx) + by (y - my) + p sum_k d_k exp(-r_k^2 / (2 s_k^2))`
//!
//! with `p = -1` for dark dots and `+1` for light ones, `r_k` the distance from dot `k`'s
//! centre, `(mx, my)` the mean pixel position of the window, over the pixels within a reach
//! of the start points (the window). The plane and the depths `d_k` enter linearly and are
//! solved exactly (least squares) for each trial of the other unknowns; a depth that is not
//! positive (above the rounding of the values) refuses that trial (a dot of the other polarity
//! is no dot). The centres, and a
//! width when it is free, are found by Levenberg-Marquardt with a numerical Jacobian. Depths
//! are free, so the brightness of the background does not matter; values are whatever
//! [`GreySource::value`] gives (an image deeper than 8 bits is fitted at its own depth).
//!
//! - [`fit_dot_shape`]: one dot with centre, width and depth free (a marker's dot width).
//! - [`fit_dots`]: up to [`MAX_DOTS`] dots with given widths, centres and depths free.
//! - [`split_touching_dots`]: the procedure that decides whether the dot at a point is two
//!   touching dots, and fits them apart ([`SplitRules`] gives its numbers).
//! - [`split_joined_dots`]: the same for the centre of a shape the detection joined, whose
//!   extent is known: two dots are taken only when the shape covers both (review 19).
//! - [`WithoutDot`]: a picture with one fitted dot taken out (for measuring the other dot's
//!   contrast without its neighbour).
//!
//! Every public function is total and deterministic: any argument value (not finite, huge,
//! negative, an empty image, a start outside the image, a width of 0 or 1e9) gives a result
//! without panicking. Work and allocation are bounded: the window is the bounding box of the
//! start points widened by the reach (at most [`MAX_REACH`]) and clipped to the image, and a
//! box with a side above [`MAX_WINDOW_SIDE`] pixels is refused, so a window holds at most
//! `MAX_WINDOW_SIDE^2` pixels; the fit runs at most [`MAX_ITERATIONS`] iterations.
//!
//! The linear solves (the plane and depths, at most 3 + [`MAX_DOTS`] unknowns, and each
//! Levenberg-Marquardt step, at most `2 MAX_DOTS + 1`) are the normal equations, symmetric
//! and small, solved with [`crate::linalg::solve_eig`] (OpenCV's own Jacobi eigen solve,
//! which drops the directions a degenerate window cannot determine, as two dots fitted on top
//! of each other).
//!
//! The method was tried in Python on 25 November 2026 (the scratch
//! scripts `dotfit.py` and `stateless.py`); its numbers ([`SplitRules::default`]) were set
//! once from the reasoning given there and were not tuned on the three recordings.

use crate::image::GreySource;
use crate::linalg::solve_eig;

/// The most dots fitted together ([`fit_dots`]).
pub const MAX_DOTS: usize = 3;

/// The largest reach (pixels) of a window round a start point: a larger reach is clamped to
/// this.
pub const MAX_REACH: f64 = 24.0;

/// The largest side (pixels) of a window's bounding box: start points further apart than this
/// less twice the reach are refused (no fit), so a window holds at most this squared pixels.
pub const MAX_WINDOW_SIDE: usize = 96;

/// The most Levenberg-Marquardt iterations of a fit.
pub const MAX_ITERATIONS: usize = 40;

/// The smallest and largest width (Gaussian sigma, pixels) a fit may give a dot: a trial
/// outside is refused. Below 0.6 px a dot is narrower than a pixel (one bright or dark pixel,
/// noise); above 8 px it is no bead at the scale of these pictures, and its tails would not
/// fit in a window of [`MAX_REACH`].
pub const WIDTH_LIMITS: (f64, f64) = (0.6, 8.0);

/// The fewest pixels in a window: fewer cannot determine a plane and the dots.
const MIN_PIXELS: usize = 12;

/// The step of the numerical Jacobian (pixels), central differences.
const JACOBIAN_STEP: f64 = 0.05;

/// One fitted dot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FittedDot {
    /// Its centre (pixels, origin top-left, pixel centres at integers).
    pub centre: (f64, f64),
    /// Its width: the Gaussian's sigma (pixels).
    pub width: f64,
    /// Its depth: how much darker (dark dots) or lighter (light ones) its centre is than the
    /// background plane, in grey levels of the source. Positive.
    pub depth: f64,
}

/// The result of a fit: the dots and the root mean square residual over the window's pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct DotsFit {
    pub dots: Vec<FittedDot>,
    pub rms: f64,
}

/// The numbers of [`split_touching_dots`]. Set once from the reasoning on 25 November 2026
/// and not tuned on the recordings (each is explained at its field).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitRules {
    /// The second dot must have at least this share of the first dot's depth (in the
    /// residual search and after the two-dot fit, the smaller of the two depths against the
    /// larger): a fainter one is noise or texture, not a bead of the same kind. Default 0.35.
    pub min_depth_ratio: f64,
    /// The two-dot fit is taken only when its rms residual is at most this share of the
    /// one-dot fit's on the same pixels: two dots must explain the picture clearly better
    /// than one (two dots on top of each other explain it no better). Default 0.8.
    pub max_rms_ratio: f64,
    /// The two fitted centres must be at least this far apart (pixels), and the second dot is
    /// looked for at least this far from the first: closer, the two cannot be told apart.
    /// Default 1.5.
    pub min_apart: f64,
    /// No fitted centre may move more than this (pixels) from its start: a fit that wanders
    /// has gone onto something else. Default 4.
    pub max_move: f64,
    /// The window's reach is this many widths ... Default 3.
    pub reach_widths: f64,
    /// ... plus this many pixels round the start point: the dot to 3 sigma plus room for a
    /// neighbour 4 px away. Default 4.
    pub reach_margin: f64,
}

impl Default for SplitRules {
    fn default() -> Self {
        SplitRules {
            min_depth_ratio: 0.35,
            max_rms_ratio: 0.8,
            min_apart: 1.5,
            max_move: 4.0,
            reach_widths: 3.0,
            reach_margin: 4.0,
        }
    }
}

/// What [`split_touching_dots`] found.
#[derive(Clone, Debug, PartialEq)]
pub enum Split {
    /// Nothing to split: one dot, or the picture does not show two clearly. The caller keeps
    /// its own point.
    One,
    /// Two touching dots, fitted together. `dots[0]` is the one fitted from the start point,
    /// `dots[1]` the one found beside it. `rms_one` and `rms_two` are the one-dot and the
    /// two-dot rms residuals over the same pixels (the two-dot window).
    Two {
        dots: [FittedDot; 2],
        rms_one: f64,
        rms_two: f64,
    },
}

/// The pixels of a window: positions and values, with the mean position.
struct Window {
    xs: Vec<f64>,
    ys: Vec<f64>,
    vs: Vec<f64>,
    mx: f64,
    my: f64,
    /// The smallest depth that counts as a dot: [`DEPTH_FLOOR`] times the largest absolute
    /// value in the window (a depth below that is rounding, as on a flat picture).
    min_depth: f64,
}

/// A dot's depth must exceed this share of the largest absolute value in its window: below,
/// it is the rounding of the least-squares solve, not a dot (a flat picture fits a "dot" of
/// depth 1e-13).
const DEPTH_FLOOR: f64 = 1e-9;

/// The pixels within `reach` of any of `centres` (whole pixels inside the image, row order).
/// `None` when the image is empty, a centre or the reach is not finite, the bounding box
/// misses the image or has a side above [`MAX_WINDOW_SIDE`], or fewer than [`MIN_PIXELS`]
/// pixels are in it.
fn window<S: GreySource + ?Sized>(image: &S, centres: &[(f64, f64)], reach: f64) -> Option<Window> {
    let (w, h) = (image.width(), image.height());
    if w == 0 || h == 0 || centres.is_empty() || !(reach.is_finite() && reach > 0.0) {
        return None;
    }
    if centres
        .iter()
        .any(|c| !(c.0.is_finite() && c.1.is_finite()))
    {
        return None;
    }
    let reach = reach.min(MAX_REACH);
    let min_x = centres.iter().map(|c| c.0).fold(f64::INFINITY, f64::min) - reach;
    let max_x = centres
        .iter()
        .map(|c| c.0)
        .fold(f64::NEG_INFINITY, f64::max)
        + reach;
    let min_y = centres.iter().map(|c| c.1).fold(f64::INFINITY, f64::min) - reach;
    let max_y = centres
        .iter()
        .map(|c| c.1)
        .fold(f64::NEG_INFINITY, f64::max)
        + reach;
    if max_x - min_x > MAX_WINDOW_SIDE as f64 || max_y - min_y > MAX_WINDOW_SIDE as f64 {
        return None;
    }
    let x0 = min_x.floor().max(0.0);
    let y0 = min_y.floor().max(0.0);
    let x1 = max_x.ceil().min((w - 1) as f64);
    let y1 = max_y.ceil().min((h - 1) as f64);
    if x1 < x0 || y1 < y0 {
        return None;
    }
    let (x0, x1, y0, y1) = (x0 as usize, x1 as usize, y0 as usize, y1 as usize);
    let r2 = reach * reach;
    let mut win = Window {
        xs: Vec::new(),
        ys: Vec::new(),
        vs: Vec::new(),
        mx: 0.0,
        my: 0.0,
        min_depth: 0.0,
    };
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (fx, fy) = (x as f64, y as f64);
            if centres
                .iter()
                .any(|c| (fx - c.0).powi(2) + (fy - c.1).powi(2) <= r2)
            {
                let Some(i) = y.checked_mul(w).and_then(|i| i.checked_add(x)) else {
                    continue;
                };
                let v = f64::from(image.value(i));
                if !v.is_finite() {
                    return None;
                }
                win.xs.push(fx);
                win.ys.push(fy);
                win.vs.push(v);
            }
        }
    }
    let n = win.xs.len();
    if n < MIN_PIXELS {
        return None;
    }
    win.mx = win.xs.iter().sum::<f64>() / n as f64;
    win.my = win.ys.iter().sum::<f64>() / n as f64;
    win.min_depth = DEPTH_FLOOR * win.vs.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    Some(win)
}

/// The Gaussian profile `exp(-r^2 / (2 s^2))`.
fn gauss(dx: f64, dy: f64, s: f64) -> f64 {
    (-(dx * dx + dy * dy) / (2.0 * s * s)).exp()
}

/// Least squares of the window's values on the columns `cols` (each one value per pixel):
/// the coefficients and the residuals (prediction minus value). `None` when not finite.
fn least_squares(cols: &[Vec<f64>], vs: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let k = cols.len();
    let mut ata = vec![0.0; k * k];
    let mut atb = vec![0.0; k];
    for i in 0..k {
        for j in i..k {
            let s: f64 = cols[i].iter().zip(&cols[j]).map(|(a, b)| a * b).sum();
            ata[i * k + j] = s;
            ata[j * k + i] = s;
        }
        atb[i] = cols[i].iter().zip(vs).map(|(a, b)| a * b).sum();
    }
    if !(ata.iter().all(|v| v.is_finite()) && atb.iter().all(|v| v.is_finite())) {
        return None;
    }
    let coef = solve_eig(&ata, k, &atb, 1);
    if !coef.iter().all(|v| v.is_finite()) {
        return None;
    }
    let res: Vec<f64> = (0..vs.len())
        .map(|p| cols.iter().zip(&coef).map(|(c, a)| c[p] * a).sum::<f64>() - vs[p])
        .collect();
    Some((coef, res))
}

/// The plane's three columns over the window.
fn plane_columns(win: &Window) -> Vec<Vec<f64>> {
    vec![
        vec![1.0; win.xs.len()],
        win.xs.iter().map(|x| x - win.mx).collect(),
        win.ys.iter().map(|y| y - win.my).collect(),
    ]
}

/// The model with the given centres and widths, plane and depths solved: the residuals and
/// the depths. `None` when a width is outside [`WIDTH_LIMITS`], a depth is not positive, or
/// the solve is not finite.
fn solve_linear(
    win: &Window,
    centres: &[(f64, f64)],
    widths: &[f64],
    polarity: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if widths
        .iter()
        .any(|&s| !(s >= WIDTH_LIMITS.0 && s <= WIDTH_LIMITS.1))
    {
        return None;
    }
    let mut cols = plane_columns(win);
    for (c, &s) in centres.iter().zip(widths) {
        cols.push(
            win.xs
                .iter()
                .zip(&win.ys)
                .map(|(x, y)| polarity * gauss(x - c.0, y - c.1, s))
                .collect(),
        );
    }
    let (coef, res) = least_squares(&cols, &win.vs)?;
    let depths = coef[3..].to_vec();
    if depths.iter().any(|&d| d.is_nan() || d <= win.min_depth) {
        return None;
    }
    Some((res, depths))
}

/// Levenberg-Marquardt over the centres (and, with `free_width`, the widths) of `n` dots in
/// the window, from `starts` and `widths`. Returns the fitted dots and the rms residual.
fn levenberg_marquardt(
    win: &Window,
    starts: &[(f64, f64)],
    widths: &[f64],
    free_width: bool,
    polarity: f64,
) -> Option<DotsFit> {
    let n = starts.len();
    let mut theta: Vec<f64> = starts.iter().flat_map(|c| [c.0, c.1]).collect();
    if free_width {
        theta.extend_from_slice(widths);
    }
    let unpack = |th: &[f64]| -> (Vec<(f64, f64)>, Vec<f64>) {
        let c = (0..n).map(|k| (th[2 * k], th[2 * k + 1])).collect();
        let s = if free_width {
            th[2 * n..].to_vec()
        } else {
            widths.to_vec()
        };
        (c, s)
    };
    let eval = |th: &[f64]| {
        let (c, s) = unpack(th);
        solve_linear(win, &c, &s, polarity)
    };
    let (mut res, mut depths) = eval(&theta)?;
    let mut cost: f64 = res.iter().map(|r| r * r).sum();
    let mut lambda = 1e-2;
    let np = theta.len();
    let m = res.len();
    for _ in 0..MAX_ITERATIONS {
        // The Jacobian of the residuals by central differences.
        let mut jac = vec![0.0; m * np];
        let mut ok = true;
        for i in 0..np {
            let mut tp = theta.clone();
            tp[i] += JACOBIAN_STEP;
            let mut tm = theta.clone();
            tm[i] -= JACOBIAN_STEP;
            let (Some(a), Some(b)) = (eval(&tp), eval(&tm)) else {
                ok = false;
                break;
            };
            for p in 0..m {
                jac[p * np + i] = (a.0[p] - b.0[p]) / (2.0 * JACOBIAN_STEP);
            }
        }
        if !ok {
            break;
        }
        let mut jtj = vec![0.0; np * np];
        let mut g = vec![0.0; np];
        for p in 0..m {
            let row = &jac[p * np..(p + 1) * np];
            for i in 0..np {
                g[i] -= row[i] * res[p];
                for j in 0..np {
                    jtj[i * np + j] += row[i] * row[j];
                }
            }
        }
        if !(jtj.iter().all(|v| v.is_finite()) && g.iter().all(|v| v.is_finite())) {
            break;
        }
        let mut improved = false;
        let mut done = false;
        for _ in 0..8 {
            let mut a = jtj.clone();
            for i in 0..np {
                a[i * np + i] += lambda * (jtj[i * np + i] + 1e-9);
            }
            let step = solve_eig(&a, np, &g, 1);
            if !step.iter().all(|v| v.is_finite()) {
                lambda *= 10.0;
                continue;
            }
            let next: Vec<f64> = theta.iter().zip(&step).map(|(t, s)| t + s).collect();
            if let Some((r2, d2)) = eval(&next) {
                let c2: f64 = r2.iter().map(|r| r * r).sum();
                if c2 < cost {
                    let largest = step.iter().fold(0.0f64, |a, s| a.max(s.abs()));
                    done = cost - c2 < 1e-8 * cost || largest < 1e-3;
                    theta = next;
                    res = r2;
                    depths = d2;
                    cost = c2;
                    lambda = (lambda / 3.0).max(1e-6);
                    improved = true;
                    break;
                }
            }
            lambda *= 10.0;
        }
        if !improved || done {
            break;
        }
    }
    let (c, s) = unpack(&theta);
    let rms = (cost / m as f64).sqrt();
    if !rms.is_finite() {
        return None;
    }
    Some(DotsFit {
        dots: c
            .iter()
            .zip(&s)
            .zip(&depths)
            .map(|((&centre, &width), &depth)| FittedDot {
                centre,
                width,
                depth,
            })
            .collect(),
        rms,
    })
}

fn polarity(dark: bool) -> f64 {
    if dark { -1.0 } else { 1.0 }
}

/// One dot fitted at `start` with its centre, width and depth free (dark, or light when
/// `dark` is false), over the pixels within `reach` of `start`, the width starting at
/// `width`: how wide a marker's dot is. `None` when there is no fit (a window too small or
/// outside the image, a start width outside [`WIDTH_LIMITS`], no dot of that polarity, or
/// arguments not finite). The caller judges whether the fit is the marker's dot (how far its
/// centre moved, whether its width is plausible).
#[must_use]
pub fn fit_dot_shape<S: GreySource + ?Sized>(
    image: &S,
    start: (f64, f64),
    width: f64,
    dark: bool,
    reach: f64,
) -> Option<FittedDot> {
    let win = window(image, &[start], reach)?;
    let fit = levenberg_marquardt(&win, &[start], &[width], true, polarity(dark))?;
    fit.dots.first().copied()
}

/// `starts.len()` dots (1 to [`MAX_DOTS`]) fitted together with the given `widths` (one per
/// dot), centres and depths free, over the pixels within `reach` of any start. `None` when
/// there is no fit (as [`fit_dot_shape`]; also a count of starts outside 1 to
/// [`MAX_DOTS`] or not matching the widths, or starts so far apart that the window's side
/// exceeds [`MAX_WINDOW_SIDE`]) or when a fitted centre moved more than `max_move` from its
/// start (a `max_move` that is not a number refuses every fit; infinity allows any).
#[must_use]
pub fn fit_dots<S: GreySource + ?Sized>(
    image: &S,
    starts: &[(f64, f64)],
    widths: &[f64],
    dark: bool,
    reach: f64,
    max_move: f64,
) -> Option<DotsFit> {
    if starts.is_empty() || starts.len() > MAX_DOTS || starts.len() != widths.len() {
        return None;
    }
    let win = window(image, starts, reach)?;
    let fit = levenberg_marquardt(&win, starts, widths, false, polarity(dark))?;
    let moved_ok = fit.dots.iter().zip(starts).all(|(d, s)| {
        let m = ((d.centre.0 - s.0).powi(2) + (d.centre.1 - s.1).powi(2)).sqrt();
        m <= max_move
    });
    moved_ok.then_some(fit)
}

/// Whether the dot at `start` is two touching dots of width `width` (dark, or light when
/// `dark` is false), and if so the two fitted apart. Nothing is carried from one call to
/// another. The procedure (with `rules`, [`SplitRules`]):
///
/// 1. One dot of width `width` is fitted at `start` (centre and depth free), over the pixels
///    within the reach (`reach_widths` widths plus `reach_margin`, at most [`MAX_REACH`]).
/// 2. What it does not explain (the residual, with the plane solved again for that dot) is
///    searched for a second dot: for each pixel of the window between `min_apart` and the
///    reach less one width from the fitted centre, the depth a dot of that width would have
///    there (a matched filter on the residual). The best must reach `min_depth_ratio` of the
///    first dot's depth.
/// 3. Two dots are fitted together from `start` and that pixel (centres and depths free, both
///    of width `width`), no centre moving more than `max_move` from its start.
/// 4. The two-dot answer is taken only when both depths are positive and the smaller is at
///    least `min_depth_ratio` of the larger, the centres are at least `min_apart` apart, and
///    the two-dot rms residual is at most `max_rms_ratio` of that of the one dot of step 1
///    (its centre kept, plane and depth solved again) on the same pixels.
/// 5. `start` is the joined shape's centre: it must lie nearer the middle of the two fitted
///    centres than either of them (review 19: with the start on one dot and a separate dot 6
///    to 8 px beside it, steps 1 to 4 take two; the detection stood on a dot and there is
///    nothing to repair).
/// 6. ([`split_joined_dots`] only) both fitted centres lie within the detected shape.
/// 7. Where a third dot would be: what the two leave is searched as in step 2 (the same
///    reach of `start`, at least `min_apart` from both centres) for its strongest place.
/// 8. No third dot: three fitted together from the two centres and step 7's strongest place
///    refuse when they explain the picture clearly better than two (rms at most
///    `max_rms_ratio` of the two-dot rms on the same pixels) with the shallowest at least
///    `min_depth_ratio` of the deepest (review 19: a marker between two beads 3.5 to 4 px
///    either side; two dots fitted between the three land 2.3 to 3 px off the middle one,
///    and step 7 alone does not see the third).
///
/// No new number: every test uses the five of [`SplitRules`].
///
/// [`Split::One`] otherwise, and for any argument that is not usable (a start not finite or
/// outside the image, a width outside [`WIDTH_LIMITS`], rules not finite).
#[must_use]
pub fn split_touching_dots<S: GreySource + ?Sized>(
    image: &S,
    start: (f64, f64),
    width: f64,
    dark: bool,
    rules: &SplitRules,
) -> Split {
    split_joined_dots(image, start, width, dark, rules, None)
}

/// [`split_touching_dots`] for a joined shape whose extent is known: `shape_radius` is the
/// radius round `start` that the detected shape covers (the detection's enclosing circle).
/// Two dots are taken only when **both fitted centres lie within `shape_radius` of
/// `start`** (the shape the detection joined covers both; a dot outside it was not joined to
/// this one, and the detection's own point needs no repair). It also lets step 3 be skipped
/// when the second dot's start lies further than `shape_radius` plus `max_move` from `start`
/// (the fit cannot then end inside the shape). `None`: no such test ([`split_touching_dots`]).
/// A `shape_radius` that is not a positive finite number takes nothing.
#[must_use]
pub fn split_joined_dots<S: GreySource + ?Sized>(
    image: &S,
    start: (f64, f64),
    width: f64,
    dark: bool,
    rules: &SplitRules,
    shape_radius: Option<f64>,
) -> Split {
    if let Some(r) = shape_radius
        && !(r.is_finite() && r > 0.0)
    {
        return Split::One;
    }
    let dist = |a: (f64, f64), b: (f64, f64)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    let (w, h) = (image.width() as f64, image.height() as f64);
    if !(start.0.is_finite() && start.1.is_finite())
        || start.0 < 0.0
        || start.1 < 0.0
        || start.0 > w - 1.0
        || start.1 > h - 1.0
        || !(width >= WIDTH_LIMITS.0 && width <= WIDTH_LIMITS.1)
    {
        return Split::One;
    }
    let SplitRules {
        min_depth_ratio,
        max_rms_ratio,
        min_apart,
        max_move,
        reach_widths,
        reach_margin,
    } = *rules;
    if ![
        min_depth_ratio,
        max_rms_ratio,
        min_apart,
        max_move,
        reach_widths,
        reach_margin,
    ]
    .iter()
    .all(|v| v.is_finite())
    {
        return Split::One;
    }
    let reach = (reach_widths * width + reach_margin).min(MAX_REACH);
    if reach.is_nan() || reach <= 0.0 {
        return Split::One;
    }
    let pol = polarity(dark);
    // 1. One dot.
    let Some(one) = fit_dots(image, &[start], &[width], dark, reach, max_move) else {
        return Split::One;
    };
    let first = one.dots[0];
    let (c1, d1) = (first.centre, first.depth);
    // 2. The residual of the one dot on the window round `start`, the plane solved again.
    let Some(win) = window(image, &[start], reach) else {
        return Split::One;
    };
    let g1: Vec<f64> = win
        .xs
        .iter()
        .zip(&win.ys)
        .map(|(x, y)| gauss(x - c1.0, y - c1.1, width))
        .collect();
    let target: Vec<f64> = win
        .vs
        .iter()
        .zip(&g1)
        .map(|(v, g)| v - pol * d1 * g)
        .collect();
    let Some((_, plane_res)) = least_squares(&plane_columns(&win), &target) else {
        return Split::One;
    };
    // `plane_res` is plane - (v - pol d1 g1); the residual of the model is its negative.
    let res: Vec<f64> = plane_res.iter().map(|r| -r).collect();
    let outer = reach - width;
    let Some((amp, second)) = strongest_dot(&win, &res, width, pol, |q| {
        let d = dist(q, c1);
        d >= min_apart && d <= outer
    }) else {
        return Split::One;
    };
    if amp.is_nan() || amp < min_depth_ratio * d1 {
        return Split::One;
    }
    // With the shape's extent known: a second dot whose start lies beyond the shape by more
    // than `max_move` cannot end inside it.
    if let Some(r) = shape_radius
        && dist(second, start) > r + max_move
    {
        return Split::One;
    }
    // 3. Two dots together.
    let Some(two) = fit_dots(
        image,
        &[start, second],
        &[width, width],
        dark,
        reach,
        max_move,
    ) else {
        return Split::One;
    };
    let (a, b) = (two.dots[0], two.dots[1]);
    if !pair_acceptable(&a, &b, min_depth_ratio, min_apart) {
        return Split::One;
    }
    // 4. The one dot of step 1 (its centre kept) on the two-dot window, plane and depth
    // solved again.
    let Some(win2) = window(image, &[start, second], reach) else {
        return Split::One;
    };
    let mut cols = plane_columns(&win2);
    cols.push(
        win2.xs
            .iter()
            .zip(&win2.ys)
            .map(|(x, y)| pol * gauss(x - c1.0, y - c1.1, width))
            .collect(),
    );
    let Some((_, r1)) = least_squares(&cols, &win2.vs) else {
        return Split::One;
    };
    let rms_one = (r1.iter().map(|r| r * r).sum::<f64>() / r1.len() as f64).sqrt();
    let rms_two = two.rms;
    if !(rms_two.is_finite() && rms_one.is_finite()) || rms_two > max_rms_ratio * rms_one {
        return Split::One;
    }
    // 5. The start is the joined shape's centre: it lies nearer the middle of the two fitted
    // centres than either of them. A start on one of the two (the detection stood on one dot,
    // the other dot is a separate one beside it) has nothing to repair.
    let mid = (
        (a.centre.0 + b.centre.0) / 2.0,
        (a.centre.1 + b.centre.1) / 2.0,
    );
    let to_mid = dist(start, mid);
    if !(to_mid < dist(start, a.centre) && to_mid < dist(start, b.centre)) {
        return Split::One;
    }
    // 6. With the shape's extent known, it covers both fitted centres.
    if let Some(r) = shape_radius
        && !(dist(a.centre, start) <= r && dist(b.centre, start) <= r)
    {
        return Split::One;
    }
    // 7. Where a third dot would be: what the two leave (plane and depths solved again with
    // the two fitted centres) is searched as in step 2, in the same reach of the start and at
    // least `min_apart` from both centres. (Until round b19d a dot there of at least
    // `min_depth_ratio` of the shallower fitted depth refused the split by itself; no scene
    // was found where that decided and step 8 did not, so it is now only step 8's start.)
    let mut cols3 = plane_columns(&win);
    for d in [a, b] {
        cols3.push(
            win.xs
                .iter()
                .zip(&win.ys)
                .map(|(x, y)| pol * gauss(x - d.centre.0, y - d.centre.1, width))
                .collect(),
        );
    }
    let Some((_, r3)) = least_squares(&cols3, &win.vs) else {
        return Split::One;
    };
    let res3: Vec<f64> = r3.iter().map(|r| -r).collect();
    let third = strongest_dot(&win, &res3, width, pol, |q| {
        dist(q, a.centre) >= min_apart && dist(q, b.centre) >= min_apart && dist(q, start) <= outer
    });
    // 8. Three dots fitted together from the two centres and the residual's strongest place
    // of step 7 (centres within `max_move` of their starts, all of width `width`): when they
    // explain the picture clearly better than two (rms at most `max_rms_ratio` of the two-dot
    // rms on the same pixels) and the shallowest is at least `min_depth_ratio` of the
    // deepest, the picture is not two dots. (A depth test on step 7's residual alone does not
    // see a marker between two beads: two dots fitted between three explain the middle one's
    // tails, and the residual keeps about a fifth of a dot's depth there.)
    if let Some((_, q3)) = third
        && let Some(three) = fit_dots(
            image,
            &[a.centre, b.centre, q3],
            &[width, width, width],
            dark,
            reach,
            max_move,
        )
    {
        let d3 = three
            .dots
            .iter()
            .map(|d| d.depth)
            .fold(f64::INFINITY, f64::min);
        let deep3 = three.dots.iter().map(|d| d.depth).fold(0.0f64, f64::max);
        // The two-dot model's rms on the three-dot window.
        if let Some(win3) = window(image, &[a.centre, b.centre, q3], reach) {
            let mut c2 = plane_columns(&win3);
            for d in [a, b] {
                c2.push(
                    win3.xs
                        .iter()
                        .zip(&win3.ys)
                        .map(|(x, y)| pol * gauss(x - d.centre.0, y - d.centre.1, width))
                        .collect(),
                );
            }
            if let Some((_, r2)) = least_squares(&c2, &win3.vs) {
                let rms2 = (r2.iter().map(|r| r * r).sum::<f64>() / r2.len() as f64).sqrt();
                if d3 >= min_depth_ratio * deep3 && three.rms <= max_rms_ratio * rms2 {
                    return Split::One;
                }
            }
        }
    }
    Split::Two {
        dots: [a, b],
        rms_one,
        rms_two,
    }
}

/// Step 4's tests of the two fitted dots themselves: both depths positive, the smaller at
/// least `min_depth_ratio` of the larger, the centres at least `min_apart` apart.
fn pair_acceptable(a: &FittedDot, b: &FittedDot, min_depth_ratio: f64, min_apart: f64) -> bool {
    let sep = ((a.centre.0 - b.centre.0).powi(2) + (a.centre.1 - b.centre.1).powi(2)).sqrt();
    let (small, large) = (a.depth.min(b.depth), a.depth.max(b.depth));
    small > 0.0 && small >= min_depth_ratio * large && sep >= min_apart
}

/// The matched filter of step 2 of [`split_touching_dots`]: over the window's pixels that
/// `allowed` admits, the depth a dot of width `width` would have there in the residual `res`
/// (model minus picture, negated: picture minus model), polarity `pol`; the strongest, with its
/// pixel. First in row order on a tie.
fn strongest_dot(
    win: &Window,
    res: &[f64],
    width: f64,
    pol: f64,
    allowed: impl Fn((f64, f64)) -> bool,
) -> Option<(f64, (f64, f64))> {
    let mut best: Option<(f64, (f64, f64))> = None;
    for i in 0..win.xs.len() {
        let (px, py) = (win.xs[i], win.ys[i]);
        if !allowed((px, py)) {
            continue;
        }
        let (mut num, mut den) = (0.0, 0.0);
        for p in 0..win.xs.len() {
            let g = gauss(win.xs[p] - px, win.ys[p] - py, width);
            num += res[p] * g;
            den += g * g;
        }
        if den.is_nan() || den <= 0.0 {
            continue;
        }
        let amp = pol * num / den;
        if best.is_none_or(|b| amp > b.0) {
            best = Some((amp, (px, py)));
        }
    }
    best
}

/// A picture with one fitted dot taken out: each value is the source's with the dot's
/// Gaussian added back (dark dot) or taken away (light dot), so that a neighbouring dot's
/// contrast can be measured without it. Only what is read changes; the source is not
/// copied. A dot whose numbers are not finite takes nothing out.
pub struct WithoutDot<'a, S: GreySource + ?Sized> {
    pub source: &'a S,
    pub dot: FittedDot,
    pub dark: bool,
}

impl<S: GreySource + ?Sized> GreySource for WithoutDot<'_, S> {
    fn width(&self) -> usize {
        self.source.width()
    }
    fn height(&self) -> usize {
        self.source.height()
    }
    fn value(&self, i: usize) -> f32 {
        let v = self.source.value(i);
        let w = self.source.width();
        let d = self.dot;
        if w == 0
            || !(d.centre.0.is_finite()
                && d.centre.1.is_finite()
                && d.depth.is_finite()
                && d.width.is_finite()
                && d.width > 0.0)
        {
            return v;
        }
        let (x, y) = ((i % w) as f64, (i / w) as f64);
        let g = gauss(x - d.centre.0, y - d.centre.1, d.width);
        let delta = -polarity(self.dark) * d.depth * g;
        (f64::from(v) + delta) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, ImageF32};

    /// A deterministic noise source (a linear congruential generator), roughly uniform in
    /// [-amp, amp].
    struct Noise(u64);
    impl Noise {
        fn next(&mut self, amp: f64) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
            (2.0 * u - 1.0) * amp
        }
    }

    /// A 64 x 64 picture: `ground` plus slope (`sx`, `sy` per pixel) plus dots (x, y,
    /// signed depth: negative dark, sigma), plus uniform noise of `noise`.
    fn scene(
        ground: f64,
        slope: (f64, f64),
        dots: &[(f64, f64, f64, f64)],
        noise: f64,
        seed: u64,
    ) -> ImageF32 {
        let mut im: ImageF32 = Image::new(64, 64);
        let mut rng = Noise(seed);
        for y in 0..64 {
            for x in 0..64 {
                let (fx, fy) = (x as f64, y as f64);
                let mut v = ground + slope.0 * (fx - 32.0) + slope.1 * (fy - 32.0);
                for &(cx, cy, d, s) in dots {
                    v += d * gauss(fx - cx, fy - cy, s);
                }
                v += rng.next(noise);
                im.data_mut()[y * 64 + x] = v as f32;
            }
        }
        im
    }

    fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
    }

    #[test]
    fn one_dot_shape_is_recovered() {
        let im = scene(150.0, (0.0, 0.0), &[(30.3, 31.6, -40.0, 1.4)], 2.0, 1);
        let d = fit_dot_shape(&im, (31.0, 31.0), 1.5, true, 5.0).expect("fits");
        assert!(dist(d.centre, (30.3, 31.6)) < 0.1, "{d:?}");
        assert!((d.width - 1.4).abs() < 0.1, "{d:?}");
        assert!((d.depth - 40.0).abs() < 4.0, "{d:?}");
    }

    /// Two dots of width 1.4 (depths 40 and 30, noise of +-3 grey levels) a given distance
    /// apart, started at the pair's middle (the detection's merged centre), over 20 noise
    /// seeds: how many are split, and the largest distance of a true centre from its nearer
    /// fitted centre where split.
    fn two_dots(sep: f64) -> (usize, f64) {
        let mut split = 0;
        let mut worst = 0.0f64;
        for seed in 0..20u64 {
            let a = (30.0, 32.0);
            let b = (30.0 + sep * 0.8, 32.0 + sep * 0.6);
            let im = scene(
                160.0,
                (0.0, 0.0),
                &[(a.0, a.1, -40.0, 1.4), (b.0, b.1, -30.0, 1.4)],
                3.0,
                seed,
            );
            let mid = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
            if let Split::Two {
                dots,
                rms_one,
                rms_two,
            } = split_touching_dots(&im, mid, 1.4, true, &SplitRules::default())
            {
                assert!(rms_two <= 0.8 * rms_one);
                split += 1;
                let near = |t: (f64, f64)| {
                    dots.iter()
                        .map(|d| dist(d.centre, t))
                        .fold(f64::INFINITY, f64::min)
                };
                worst = worst.max(near(a)).max(near(b));
            }
        }
        (split, worst)
    }

    #[test]
    fn two_touching_dots_are_fitted_apart() {
        // What is achieved (25 November 2026): 4.5 and 4 px apart (3.2 and 2.9 widths), split
        // at every seed; 3.5 px, at 14 of 20; 3.25 px and closer, at none (the one-dot fit
        // takes both: its depth grows to about 51 and the residual's second dot reaches only
        // about 0.25 of it, under `min_depth_ratio`), so the caller keeps its own point.
        // Where split, every centre within 0.12 px.
        let (n, worst) = two_dots(4.5);
        assert!(n == 20 && worst < 0.15, "4.5 px: {n} {worst}");
        let (n, worst) = two_dots(4.0);
        assert!(n == 20 && worst < 0.15, "4 px: {n} {worst}");
        let (n, worst) = two_dots(3.5);
        assert!(n >= 12 && worst < 0.15, "3.5 px: {n} {worst}");
        let (n, _) = two_dots(2.5);
        assert_eq!(n, 0, "2.5 px");
    }

    #[test]
    fn one_dot_alone_gives_one() {
        for seed in 0..10 {
            let im = scene(
                150.0,
                (0.0, 0.0),
                &[(32.2, 31.7, -40.0, 1.4)],
                3.0,
                100 + seed,
            );
            let s = split_touching_dots(&im, (32.0, 32.0), 1.4, true, &SplitRules::default());
            assert_eq!(s, Split::One, "seed {seed}");
        }
    }

    #[test]
    fn dot_beside_a_straight_edge_gives_one() {
        // A dark half-plane (a bone edge) 4 px beside the dot.
        let mut im = scene(160.0, (0.0, 0.0), &[(30.0, 32.0, -40.0, 1.4)], 2.0, 7);
        for y in 0..64 {
            for x in 34..64 {
                im.data_mut()[y * 64 + x] -= 35.0;
            }
        }
        let s = split_touching_dots(&im, (30.0, 32.0), 1.4, true, &SplitRules::default());
        assert_eq!(s, Split::One);
    }

    #[test]
    fn dots_on_a_sloping_background() {
        let slope = (2.0, -1.5);
        let one = scene(150.0, slope, &[(32.0, 32.0, -40.0, 1.4)], 2.0, 3);
        let d = fit_dot_shape(&one, (32.5, 31.5), 1.5, true, 5.0).expect("fits");
        assert!(
            dist(d.centre, (32.0, 32.0)) < 0.1 && (d.width - 1.4).abs() < 0.1,
            "{d:?}"
        );
        assert_eq!(
            split_touching_dots(&one, (32.0, 32.0), 1.4, true, &SplitRules::default()),
            Split::One
        );
        let two = scene(
            150.0,
            slope,
            &[(30.0, 32.0, -40.0, 1.4), (34.0, 32.0, -35.0, 1.4)],
            2.0,
            4,
        );
        let Split::Two { dots, .. } =
            split_touching_dots(&two, (32.0, 32.0), 1.4, true, &SplitRules::default())
        else {
            panic!("not split");
        };
        let xs = {
            let mut v = [dots[0].centre.0, dots[1].centre.0];
            v.sort_by(f64::total_cmp);
            v
        };
        assert!(
            (xs[0] - 30.0).abs() < 0.25 && (xs[1] - 34.0).abs() < 0.25,
            "{dots:?}"
        );
    }

    #[test]
    fn light_dots() {
        let one = scene(60.0, (0.0, 0.0), &[(32.0, 32.0, 50.0, 1.6)], 2.0, 5);
        let d = fit_dot_shape(&one, (31.5, 32.5), 1.5, false, 5.0).expect("fits");
        assert!(
            dist(d.centre, (32.0, 32.0)) < 0.1 && (d.width - 1.6).abs() < 0.1,
            "{d:?}"
        );
        // A light dot is no dark dot.
        assert!(
            fit_dot_shape(&one, (31.5, 32.5), 1.5, true, 5.0)
                .is_none_or(|d| { dist(d.centre, (32.0, 32.0)) > 1.0 })
        );
        let two = scene(
            60.0,
            (0.0, 0.0),
            &[(30.0, 32.0, 50.0, 1.6), (30.0, 36.0, 45.0, 1.6)],
            2.0,
            6,
        );
        let Split::Two { dots, .. } =
            split_touching_dots(&two, (30.0, 34.0), 1.6, false, &SplitRules::default())
        else {
            panic!("not split");
        };
        let ys = {
            let mut v = [dots[0].centre.1, dots[1].centre.1];
            v.sort_by(f64::total_cmp);
            v
        };
        assert!(
            (ys[0] - 32.0).abs() < 0.25 && (ys[1] - 36.0).abs() < 0.25,
            "{dots:?}"
        );
    }

    #[test]
    fn sixteen_bit_values() {
        // The same scene at 256 times the scale (as a 16-bit source read at its own depth):
        // the same centres.
        let base = [(30.0, 32.0, -40.0, 1.4), (33.5, 33.0, -32.0, 1.4)];
        let deep: Vec<(f64, f64, f64, f64)> = base
            .iter()
            .map(|&(x, y, d, s)| (x, y, d * 256.0, s))
            .collect();
        let im8 = scene(160.0, (0.0, 0.0), &base, 2.0, 9);
        let im16 = scene(160.0 * 256.0, (0.0, 0.0), &deep, 2.0 * 256.0, 9);
        let r = SplitRules::default();
        let (Split::Two { dots: a, .. }, Split::Two { dots: b, .. }) = (
            split_touching_dots(&im8, (31.7, 32.5), 1.4, true, &r),
            split_touching_dots(&im16, (31.7, 32.5), 1.4, true, &r),
        ) else {
            panic!("not split");
        };
        for k in 0..2 {
            assert!(dist(a[k].centre, b[k].centre) < 1e-3, "{a:?} {b:?}");
            assert!((b[k].depth / a[k].depth - 256.0).abs() < 0.5);
        }
    }

    #[test]
    fn the_wrapper_takes_a_dot_out() {
        let dot = FittedDot {
            centre: (20.0, 20.0),
            width: 1.4,
            depth: 40.0,
        };
        let im = scene(150.0, (0.0, 0.0), &[(20.0, 20.0, -40.0, 1.4)], 0.0, 0);
        let clean = WithoutDot {
            source: &im,
            dot,
            dark: true,
        };
        for i in 0..64 * 64 {
            assert!((clean.value(i) - 150.0).abs() < 1e-3);
        }
        let light = scene(50.0, (0.0, 0.0), &[(20.0, 20.0, 40.0, 1.4)], 0.0, 0);
        let clean = WithoutDot {
            source: &light,
            dot,
            dark: false,
        };
        assert!((clean.value(20 * 64 + 20) - 50.0).abs() < 1e-3);
        let bad = WithoutDot {
            source: &im,
            dot: FittedDot {
                centre: (f64::NAN, 1.0),
                width: 0.0,
                depth: f64::INFINITY,
            },
            dark: true,
        };
        assert_eq!(bad.value(5), im.value(5));
    }

    #[test]
    fn hostile_arguments() {
        let im = scene(150.0, (0.0, 0.0), &[(32.0, 32.0, -40.0, 1.4)], 2.0, 1);
        let empty: ImageF32 = Image::new(0, 0);
        let tiny: ImageF32 = Image::filled(1, 1, 3.0);
        let mut nan = im.clone();
        nan.data_mut()[32 * 64 + 32] = f32::NAN;
        let values = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -1.0,
            0.0,
            1e-300,
            0.6,
            1.0,
            32.0,
            63.0,
            64.0,
            1e9,
            -1e300,
            1e300,
        ];
        for img in [&im, &empty, &tiny, &nan] {
            for &a in &values {
                for &b in &values {
                    let _ = fit_dot_shape(img, (a, b), b, true, a);
                    let _ = fit_dot_shape(img, (32.0, 32.0), a, false, b);
                    let _ = fit_dots(img, &[(a, b)], &[1.4], true, 6.0, a);
                    let _ = fit_dots(img, &[(32.0, 32.0), (a, b)], &[a, b], true, b, 4.0);
                    let _ = split_touching_dots(img, (a, b), 1.4, true, &SplitRules::default());
                    let _ =
                        split_touching_dots(img, (32.0, 32.0), a, b > 0.0, &SplitRules::default());
                    let r = SplitRules {
                        min_depth_ratio: a,
                        max_rms_ratio: b,
                        min_apart: a,
                        max_move: b,
                        reach_widths: a,
                        reach_margin: b,
                    };
                    let _ = split_touching_dots(img, (32.0, 32.0), 1.4, true, &r);
                }
            }
            // Counts out of range, mismatched lists, starts far apart.
            assert!(fit_dots(img, &[], &[], true, 5.0, 4.0).is_none());
            assert!(fit_dots(img, &[(1.0, 1.0); 4], &[1.4; 4], true, 5.0, 4.0).is_none());
            assert!(fit_dots(img, &[(1.0, 1.0)], &[1.4, 1.4], true, 5.0, 4.0).is_none());
            assert!(fit_dots(img, &[(0.0, 0.0), (1e6, 1e6)], &[1.4; 2], true, 5.0, 4.0).is_none());
        }
        // A huge image is read only round the start (the window's side is bounded).
        struct Huge;
        impl GreySource for Huge {
            fn width(&self) -> usize {
                usize::MAX / 2
            }
            fn height(&self) -> usize {
                usize::MAX / 2
            }
            fn value(&self, i: usize) -> f32 {
                (i % 7) as f32
            }
        }
        let _ = split_touching_dots(&Huge, (1e9, 1e9), 1.4, true, &SplitRules::default());
        let _ = fit_dot_shape(&Huge, (1e15, 3.0), 1.5, true, 1e300);
    }

    /// A lone elongated (elliptical Gaussian) dot, its long axis 1 to 3 times its short one
    /// (1.4 px), turned by 0.5 radians, its width learned by [`fit_dot_shape`] as a run
    /// learns it: never taken for two dots (20 noise seeds each). The procedure does not split
    /// a smooth elongated dot; what it splits on the guineafowl's conical markers is a shape
    /// with two dark parts.
    #[test]
    fn a_lone_elongated_dot_gives_one() {
        for r in [1.0f64, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0] {
            let mut split = 0;
            let mut ws = Vec::new();
            for seed in 0..20u64 {
                let (sa, sb, ang) = (1.4 * r, 1.4, 0.5f64);
                let mut im: ImageF32 = Image::new(64, 64);
                let mut rng = Noise(seed);
                for y in 0..64 {
                    for x in 0..64 {
                        let (dx, dy) = (x as f64 - 32.0, y as f64 - 32.0);
                        let u = dx * ang.cos() + dy * ang.sin();
                        let v = -dx * ang.sin() + dy * ang.cos();
                        let val = 160.0
                            - 40.0 * (-(u * u / (2.0 * sa * sa) + v * v / (2.0 * sb * sb))).exp()
                            + rng.next(3.0);
                        im.data_mut()[y * 64 + x] = val as f32;
                    }
                }
                let reach = 5.0f64.max(2.0 * r * 1.4 + 2.0);
                let Some(d) = fit_dot_shape(&im, (32.3, 31.8), 1.5, true, reach) else {
                    continue;
                };
                ws.push(d.width);
                if let Split::Two { .. } =
                    split_touching_dots(&im, (32.0, 32.0), d.width, true, &SplitRules::default())
                {
                    split += 1;
                }
            }
            assert_eq!(split, 0, "ratio {r}, widths {ws:?}");
            assert_eq!(ws.len(), 20, "ratio {r}");
        }
    }

    /// Step 4's tests of the two fitted dots (review 19: no test failed with either removed):
    /// the shallower must reach `min_depth_ratio` of the deeper, and the centres must be at
    /// least `min_apart` apart.
    #[test]
    fn the_fitted_pair_is_tested_for_depth_and_distance() {
        let dot = |x: f64, depth: f64| FittedDot {
            centre: (x, 10.0),
            width: 1.4,
            depth,
        };
        let r = SplitRules::default();
        let ok =
            |a: FittedDot, b: FittedDot| pair_acceptable(&a, &b, r.min_depth_ratio, r.min_apart);
        assert!(ok(dot(10.0, 40.0), dot(14.0, 30.0)));
        // Depth: 0.3 of the deeper is refused, 0.4 taken.
        assert!(!ok(dot(10.0, 40.0), dot(14.0, 12.0)));
        assert!(ok(dot(10.0, 40.0), dot(14.0, 16.0)));
        assert!(!ok(dot(10.0, 40.0), dot(14.0, 0.0)));
        // Distance: 1.0 px apart is refused, 2.0 taken.
        assert!(!ok(dot(10.0, 40.0), dot(11.0, 30.0)));
        assert!(ok(dot(10.0, 40.0), dot(12.0, 30.0)));
    }

    /// `fit_dots` refuses a fit whose centre moves more than `max_move` from its start
    /// (review 19: no test failed with the check removed): a dot 3 px from its start is found
    /// with `max_move` 4 and refused with 2.
    #[test]
    fn fit_dots_refuses_a_centre_that_moves_too_far() {
        let im = scene(160.0, (0.0, 0.0), &[(30.0, 32.0, -40.0, 1.4)], 1.0, 21);
        let start = [(30.0, 29.0)];
        let found = fit_dots(&im, &start, &[1.4], true, 8.0, 4.0).expect("found");
        assert!(dist(found.dots[0].centre, (30.0, 32.0)) < 0.2, "{found:?}");
        assert!(fit_dots(&im, &start, &[1.4], true, 8.0, 2.0).is_none());
    }

    /// With the shape's extent known ([`split_joined_dots`]), two dots are taken only when
    /// the shape covers both fitted centres: two dots 4.5 px apart, started at their middle,
    /// are taken with a shape radius of 3 px (each centre 2.25 px from the start) and not with
    /// 2 px; without a radius ([`split_touching_dots`]) they are taken. A radius that is not a
    /// positive number takes nothing.
    #[test]
    fn the_joined_shape_must_cover_both_centres() {
        let im = scene(
            160.0,
            (0.0, 0.0),
            &[(27.75, 32.0, -40.0, 1.4), (32.25, 32.0, -35.0, 1.4)],
            2.0,
            31,
        );
        let r = SplitRules::default();
        let two = |s: &Split| matches!(s, Split::Two { .. });
        assert!(two(&split_touching_dots(&im, (30.0, 32.0), 1.4, true, &r)));
        assert!(two(&split_joined_dots(
            &im,
            (30.0, 32.0),
            1.4,
            true,
            &r,
            Some(3.0)
        )));
        assert!(!two(&split_joined_dots(
            &im,
            (30.0, 32.0),
            1.4,
            true,
            &r,
            Some(2.0)
        )));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(!two(&split_joined_dots(
                &im,
                (30.0, 32.0),
                1.4,
                true,
                &r,
                Some(bad)
            )));
        }
    }

    /// No split when there is a third dot: a pair 4.5 px apart, started at its middle, with a
    /// third dot `d` px beside the middle (across the pair): taken without the third, not
    /// with it.
    #[test]
    fn a_third_dot_refuses_the_split() {
        let r = SplitRules::default();
        let pair = [(27.75, 32.0, -40.0, 1.4), (32.25, 32.0, -35.0, 1.4)];
        let im = scene(160.0, (0.0, 0.0), &pair, 2.0, 41);
        assert!(matches!(
            split_touching_dots(&im, (30.0, 32.0), 1.4, true, &r),
            Split::Two { .. }
        ));
        for d in [4.0, 5.0, 6.0] {
            let mut dots = pair.to_vec();
            dots.push((30.0, 32.0 + d, -35.0, 1.4));
            let im = scene(160.0, (0.0, 0.0), &dots, 2.0, 41);
            let s = split_touching_dots(&im, (30.0, 32.0), 1.4, true, &r);
            assert_eq!(s, Split::One, "third dot {d} px away: {s:?}");
        }
    }
}
