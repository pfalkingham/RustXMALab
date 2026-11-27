//! Reducing a series to what one pixel column can show, keeping minima,
//! maxima and gaps.
//!
//! QCustomPlot has its own "adaptive sampling" for line graphs. These
//! functions do the equivalent in front of egui_plot: the visible part of a
//! series is cut into runs at gaps (NaN), and when there are more samples
//! than pixel columns each column keeps only its first-occurring extreme and
//! the other extreme, in the order they occur. A line through the result
//! covers exactly the pixels a line through all samples covers.

/// Index window `[i0, i1)` of the samples to consider for a visible x range:
/// the visible ones plus one neighbour on each side, so lines run to the edge.
/// `x(i)` must be non-decreasing.
pub fn visible_window(n: usize, x: impl Fn(usize) -> f64, x0: f64, x1: f64) -> (usize, usize) {
    if n == 0 {
        return (0, 0);
    }
    let first_in = partition_point(n, |i| x(i) < x0);
    let after_last = partition_point(n, |i| x(i) <= x1);
    (first_in.saturating_sub(1), (after_last + 1).min(n))
}

fn partition_point(n: usize, pred: impl Fn(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pred(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Contiguous runs of `[x, y]` for samples `i0..i1`, broken at NaN values.
///
/// If the window holds more than `2 * cols` samples, each of the `cols` pixel
/// columns across `[x0, x1]` keeps at most two samples: its minimum and its
/// maximum, in sample order. Otherwise all samples are returned. A NaN always
/// ends a run, so gaps survive at any zoom.
pub fn decimate(
    (i0, i1): (usize, usize),
    x: impl Fn(usize) -> f64,
    y: impl Fn(usize) -> f64,
    (x0, x1): (f64, f64),
    cols: usize,
) -> Vec<Vec<[f64; 2]>> {
    let mut runs: Vec<Vec<[f64; 2]>> = Vec::new();
    if i1 <= i0 || cols == 0 || x1 <= x0 {
        return runs;
    }
    let mut cur: Vec<[f64; 2]> = Vec::new();
    if i1 - i0 <= 2 * cols {
        for i in i0..i1 {
            let v = y(i);
            if v.is_nan() {
                if !cur.is_empty() {
                    runs.push(std::mem::take(&mut cur));
                }
            } else {
                cur.push([x(i), v]);
            }
        }
        if !cur.is_empty() {
            runs.push(cur);
        }
        return runs;
    }

    let col_w = (x1 - x0) / cols as f64;
    struct Col {
        col: i64,
        min: (usize, f64),
        max: (usize, f64),
    }
    let flush = |cur: &mut Vec<[f64; 2]>, c: &Col| {
        let (a, b) = if c.min.0 <= c.max.0 {
            (c.min, c.max)
        } else {
            (c.max, c.min)
        };
        cur.push([x(a.0), a.1]);
        if a.0 != b.0 {
            cur.push([x(b.0), b.1]);
        }
    };
    let mut col: Option<Col> = None;
    cur.reserve(4 * cols + 8);
    for i in i0..i1 {
        let v = y(i);
        if v.is_nan() {
            if let Some(c) = col.take() {
                flush(&mut cur, &c);
            }
            if !cur.is_empty() {
                runs.push(std::mem::take(&mut cur));
            }
            continue;
        }
        let c = ((x(i) - x0) / col_w).floor() as i64;
        match &mut col {
            Some(cc) if cc.col == c => {
                if v < cc.min.1 {
                    cc.min = (i, v);
                }
                if v > cc.max.1 {
                    cc.max = (i, v);
                }
            }
            _ => {
                if let Some(cc) = col.take() {
                    flush(&mut cur, &cc);
                }
                col = Some(Col {
                    col: c,
                    min: (i, v),
                    max: (i, v),
                });
            }
        }
    }
    if let Some(c) = col.take() {
        flush(&mut cur, &c);
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    runs
}

/// Turns a run of points into QCustomPlot's `lsStepCenter` polyline: each value
/// is held from half-way after the previous point to half-way before the next;
/// the first and last points start and end the line.
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPGraph::dataToStepCenterLines.
pub fn step_center(run: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let Some(first) = run.first() else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(run.len() * 2);
    out.push(*first);
    let mut last = *first;
    for p in &run[1..] {
        let mid = (p[0] + last[0]) * 0.5;
        out.push([mid, last[1]]);
        out.push([mid, p[1]]);
        last = *p;
    }
    out.push(last);
    out
}

/// Drops symbols that would land on a grid cell already holding one: positions
/// are screen points, `cells_per_point` the grid resolution (the pixels per
/// point for pixel cells). Input sorted by x keeps memory small.
pub fn dedupe_pixels(points: &mut Vec<egui::Pos2>, cells_per_point: f32) {
    let ppp = cells_per_point;
    let mut col = i32::MIN;
    let mut rows: Vec<i32> = Vec::new();
    points.retain(|p| {
        let c = (p.x * ppp).round() as i32;
        let r = (p.y * ppp).round() as i32;
        if c != col {
            col = c;
            rows.clear();
        }
        if rows.contains(&r) {
            false
        } else {
            rows.push(r);
            true
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_center_holds_values_half_way() {
        let s = step_center(&[[1.0, 0.0], [2.0, 5.0], [3.0, 1.0]]);
        assert_eq!(
            s,
            vec![
                [1.0, 0.0],
                [1.5, 0.0],
                [1.5, 5.0],
                [2.5, 5.0],
                [2.5, 1.0],
                [3.0, 1.0]
            ]
        );
    }
}
