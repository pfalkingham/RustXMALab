//! Polygon routines on integer contours (`std::vector<cv::Point>`) that
//! `cv::findChessboardCorners` applies to each hole it finds: `approxPolyDP`,
//! `isContourConvex` and `boundingRect`, ported from OpenCV 4.13 so that the results are the
//! same point for point (checked against the 4.13 libraries, `chessops-*`).

use super::contours::PointI;

/// `cv::approxPolyDP(curve, approx, epsilon, closed)` on integer points: the
/// Douglas-Peucker simplification with OpenCV's choice of starting points, its order of
/// the output points and its final removal of points on nearly straight runs.
///
/// An empty curve, a negative `epsilon` or one not below 1e30 (where OpenCV throws) give an
/// empty result. Coordinates are combined in `i64` where OpenCV multiplies `int`s (the same
/// for coordinates up to 46340 in size, beyond which OpenCV's `int` products overflow).
///
/// Ported from OpenCV 4.13 imgproc/src/approx.cpp, `cv::approxPolyDP` and
/// `approxPolyDP_<int>`.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn approx_poly_dp(src: &[PointI], epsilon: f64, closed: bool) -> Vec<PointI> {
    if !(0.0..1e30).contains(&epsilon) {
        return Vec::new();
    }
    let count0 = src.len();
    if count0 == 0 {
        return Vec::new();
    }
    #[derive(Clone, Copy)]
    struct Range {
        start: usize,
        end: usize,
    }
    let count = count0;
    let mut dst = vec![PointI::default(); count0];
    let mut new_count = 0usize;
    let mut stack: Vec<Range> = Vec::with_capacity(count0);
    let mut init_iters = 3;
    let mut slice = Range { start: 0, end: 0 };
    let mut right_slice = Range { start: 0, end: 0 };
    let mut start_pt = PointI::new(-1_000_000, -1_000_000);
    let mut pt;
    let mut pos = 0usize;
    let mut is_closed = closed;
    let mut le_eps = false;
    let eps = epsilon * epsilon;
    let fx = |p: PointI| f64::from(p.x);
    let fy = |p: PointI| f64::from(p.y);
    let read = |pos: &mut usize| {
        let p = src[*pos];
        *pos += 1;
        if *pos >= count {
            *pos = 0;
        }
        p
    };

    if !is_closed {
        right_slice.start = count;
        let end_pt = src[0];
        start_pt = src[count - 1];
        if start_pt != end_pt {
            slice.start = 0;
            slice.end = count - 1;
            stack.push(slice);
        } else {
            is_closed = true;
            init_iters = 1;
        }
    }

    if is_closed {
        // 1. Find approximately two farthest points of the contour
        right_slice.start = 0;
        for _ in 0..init_iters {
            let mut max_dist = 0.0f64;
            pos = (pos + right_slice.start) % count;
            start_pt = read(&mut pos);
            for j in 1..count {
                pt = read(&mut pos);
                let dx = fx(pt) - fx(start_pt);
                let dy = fy(pt) - fy(start_pt);
                let dist = dx * dx + dy * dy;
                if dist > max_dist {
                    max_dist = dist;
                    right_slice.start = j;
                }
            }
            le_eps = max_dist <= eps;
        }
        // 2. initialize the stack
        if le_eps {
            dst[new_count] = start_pt;
            new_count += 1;
        } else {
            slice.start = pos % count;
            right_slice.end = slice.start;
            right_slice.start = (right_slice.start + slice.start) % count;
            slice.end = right_slice.start;
            stack.push(right_slice);
            stack.push(slice);
        }
    }

    // 3. run recursive process
    while let Some(sl) = stack.pop() {
        slice = sl;
        let end_pt = src[slice.end];
        pos = slice.start;
        start_pt = read(&mut pos);
        if pos == slice.end {
            le_eps = true;
            start_pt = src[slice.start];
        } else {
            let dx = fx(end_pt) - fx(start_pt);
            let dy = fy(end_pt) - fy(start_pt);
            let segment_len_2 = dx * dx + dy * dy;
            if dx == 0.0 && dy == 0.0 {
                // OpenCV asserts here; not reachable from a contour (module tests).
                return Vec::new();
            }
            let mut max_d = 0.0f64;
            while pos != slice.end {
                pt = read(&mut pos);
                let ix = i64::from(pt.x) - i64::from(start_pt.x);
                let iy = i64::from(pt.y) - i64::from(start_pt.y);
                #[allow(clippy::cast_precision_loss)]
                let projection = ix as f64 * dx + iy as f64 * dy;
                let d2 = if projection < 0.0 {
                    #[allow(clippy::cast_precision_loss)]
                    let q = (ix * ix + iy * iy) as f64;
                    q * segment_len_2
                } else if projection > segment_len_2 {
                    let ex = i64::from(pt.x) - i64::from(end_pt.x);
                    let ey = i64::from(pt.y) - i64::from(end_pt.y);
                    #[allow(clippy::cast_precision_loss)]
                    let q = (ex * ex + ey * ey) as f64;
                    q * segment_len_2
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    let dist = iy as f64 * dx - ix as f64 * dy;
                    dist * dist
                };
                if d2 > max_d {
                    max_d = d2;
                    right_slice.start = (pos + count - 1) % count;
                }
            }
            le_eps = max_d <= eps * segment_len_2;
        }
        if le_eps {
            dst[new_count] = start_pt;
            new_count += 1;
        } else {
            right_slice.end = slice.end;
            slice.end = right_slice.start;
            stack.push(right_slice);
            stack.push(slice);
        }
    }

    if !is_closed {
        dst[new_count] = src[count - 1];
        new_count += 1;
    }

    // last stage: do final clean-up of the approximated contour - remove extra points on
    // the [almost] straight lines.
    let is_closed = closed;
    let count = new_count;
    let read_dst = |dst: &[PointI], pos: &mut usize| {
        let p = dst[*pos];
        *pos += 1;
        if *pos >= count {
            *pos = 0;
        }
        p
    };
    pos = if is_closed { count - 1 } else { 0 };
    start_pt = read_dst(&dst, &mut pos);
    let mut wpos = pos;
    pt = read_dst(&dst, &mut pos);
    let open = usize::from(!is_closed);
    let mut i = open;
    while i < count - open && new_count > 2 {
        let end_pt = read_dst(&dst, &mut pos);
        let dx = fx(end_pt) - fx(start_pt);
        let dy = fy(end_pt) - fy(start_pt);
        let ix = i64::from(pt.x) - i64::from(start_pt.x);
        let iy = i64::from(pt.y) - i64::from(start_pt.y);
        #[allow(clippy::cast_precision_loss)]
        let dist = (ix as f64 * dy - iy as f64 * dx).abs();
        let sip = ix * (i64::from(end_pt.x) - i64::from(pt.x))
            + iy * (i64::from(end_pt.y) - i64::from(pt.y));
        if dist * dist <= 0.5 * eps * (dx * dx + dy * dy) && dx != 0.0 && dy != 0.0 && sip >= 0 {
            new_count -= 1;
            start_pt = end_pt;
            dst[wpos] = end_pt;
            wpos += 1;
            if wpos >= count {
                wpos = 0;
            }
            pt = read_dst(&dst, &mut pos);
            i += 2;
            continue;
        }
        start_pt = pt;
        dst[wpos] = pt;
        wpos += 1;
        if wpos >= count {
            wpos = 0;
        }
        pt = end_pt;
        i += 1;
    }
    if !is_closed {
        dst[wpos] = pt;
    }
    dst.truncate(new_count);
    dst
}

/// `cv::isContourConvex(contour)` on integer points: whether every turn of the closed
/// polygon goes the same way (a straight or zero-length edge counts as both ways, so it
/// is not convex). An empty contour is not convex.
///
/// Ported from OpenCV 4.13 imgproc/src/convhull.cpp, `isContourConvex_<int>` (products in
/// `i64`, the same as OpenCV's `int` products for coordinates up to 32767 in size).
#[must_use]
pub fn is_contour_convex(p: &[PointI]) -> bool {
    let n = p.len();
    if n == 0 {
        return false;
    }
    let mut prev_pt = p[(2 * n - 2) % n];
    let mut cur_pt = p[n - 1];
    let mut dx0 = i64::from(cur_pt.x) - i64::from(prev_pt.x);
    let mut dy0 = i64::from(cur_pt.y) - i64::from(prev_pt.y);
    let mut orientation = 0;
    for &q in p {
        prev_pt = cur_pt;
        cur_pt = q;
        let dx = i64::from(cur_pt.x) - i64::from(prev_pt.x);
        let dy = i64::from(cur_pt.y) - i64::from(prev_pt.y);
        let dxdy0 = dx * dy0;
        let dydx0 = dy * dx0;
        orientation |= if dydx0 > dxdy0 {
            1
        } else if dydx0 < dxdy0 {
            2
        } else {
            3
        };
        if orientation == 3 {
            return false;
        }
        dx0 = dx;
        dy0 = dy;
    }
    true
}

/// The width times the height of `cv::boundingRect(points)` for integer points (`Rect`
/// of `xmin, ymin, xmax - xmin + 1, ymax - ymin + 1`), as `Rect::area()` gives it; 0 for
/// no points.
///
/// Ported from OpenCV 4.13 imgproc/src/shapedescr.cpp, `pointSetBoundingRect`.
#[must_use]
pub fn bounding_rect_area(points: &[PointI]) -> i64 {
    let Some(first) = points.first() else {
        return 0;
    };
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (first.x, first.x, first.y, first.y);
    for p in points {
        xmin = xmin.min(p.x);
        xmax = xmax.max(p.x);
        ymin = ymin.min(p.y);
        ymax = ymax.max(p.y);
    }
    (i64::from(xmax) - i64::from(xmin) + 1) * (i64::from(ymax) - i64::from(ymin) + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(v: &[(i32, i32)]) -> Vec<PointI> {
        v.iter().map(|&(x, y)| PointI::new(x, y)).collect()
    }

    #[test]
    fn square_with_extra_points() {
        let c = pts(&[
            (0, 0),
            (0, 5),
            (0, 10),
            (5, 10),
            (10, 10),
            (10, 5),
            (10, 0),
            (5, 0),
        ]);
        let a = approx_poly_dp(&c, 1.0, true);
        assert_eq!(a.len(), 4);
        assert!(is_contour_convex(&a));
        assert_eq!(bounding_rect_area(&c), 121);
        assert!(approx_poly_dp(&[], 1.0, true).is_empty());
        assert!(approx_poly_dp(&c, -1.0, true).is_empty());
        assert!(approx_poly_dp(&c, f64::NAN, true).is_empty());
        assert_eq!(approx_poly_dp(&pts(&[(3, 4)]), 2.0, true), pts(&[(3, 4)]));
        assert!(!is_contour_convex(&[]));
        assert!(!is_contour_convex(&pts(&[(0, 0), (1, 1), (2, 2)])));
        let open = approx_poly_dp(&pts(&[(0, 0), (5, 1), (10, 0)]), 2.0, false);
        assert_eq!(open, pts(&[(0, 0), (10, 0)]));
    }
}
