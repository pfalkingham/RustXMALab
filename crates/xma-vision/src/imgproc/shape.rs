//! Shape descriptors of a contour given as integer points (`std::vector<cv::Point>` in
//! OpenCV): `minEnclosingCircle`, `moments` (the polygon form), `arcLength`,
//! `contourArea` and `convexHull` (also for float points: [`contour_area_f32`],
//! [`convex_hull_f32`]), ported from OpenCV 4.13 so that the results are the
//! same bit for bit, including OpenCV's float arithmetic where it uses `float`.

use super::contours::PointI;

/// `cv::minEnclosingCircle`'s tolerance, added to every radius it computes.
/// Ported from imgproc/src/shapedescr.cpp, `EPS`.
const EPS: f32 = 1.0e-4;

/// `cv::norm(Point2f)`: the length in double.
fn norm2f(dx: f32, dy: f32) -> f64 {
    (dx as f64 * dx as f64 + dy as f64 * dy as f64).sqrt()
}

/// Circle through three points, in float. Leaves `center` and `radius` unchanged when
/// the points are (nearly) collinear.
///
/// Ported from OpenCV 4.13 imgproc/src/shapedescr.cpp, `findCircle3pts`.
fn find_circle_3pts(pts: [(f32, f32); 3], center: &mut (f32, f32), radius: &mut f32) {
    let v1 = (pts[1].0 - pts[0].0, pts[1].1 - pts[0].1);
    let v2 = (pts[2].0 - pts[0].0, pts[2].1 - pts[0].1);
    let mid1 = ((pts[0].0 + pts[1].0) / 2.0, (pts[0].1 + pts[1].1) / 2.0);
    let c1 = mid1.0 * v1.0 + mid1.1 * v1.1;
    let mid2 = ((pts[0].0 + pts[2].0) / 2.0, (pts[0].1 + pts[2].1) / 2.0);
    let c2 = mid2.0 * v2.0 + mid2.1 * v2.1;
    let det = v1.0 * v2.1 - v1.1 * v2.0;
    if det.abs() <= EPS {
        return;
    }
    let mut cx = (c1 * v2.1 - c2 * v1.1) / det;
    let mut cy = (v1.0 * c2 - v2.0 * c1) / det;
    *center = (cx, cy);
    cx -= pts[0].0;
    cy -= pts[0].1;
    *radius = (cx * cx + cy * cy).sqrt() + EPS;
}

fn ptf(p: PointI) -> (f32, f32) {
    (p.x as f32, p.y as f32)
}

/// Ported from OpenCV 4.13 imgproc/src/shapedescr.cpp, `findThirdPoint<Point>`.
fn find_third_point(pts: &[PointI], i: usize, j: usize, center: &mut (f32, f32), radius: &mut f32) {
    center.0 = pts[j].x.wrapping_add(pts[i].x) as f32 / 2.0;
    center.1 = pts[j].y.wrapping_add(pts[i].y) as f32 / 2.0;
    let dx = pts[j].x.wrapping_sub(pts[i].x) as f32;
    let dy = pts[j].y.wrapping_sub(pts[i].y) as f32;
    *radius = norm2f(dx, dy) as f32 / 2.0 + EPS;
    for k in 0..j {
        let dx = center.0 - pts[k].x as f32;
        let dy = center.1 - pts[k].y as f32;
        if norm2f(dx, dy) < *radius as f64 {
            continue;
        }
        find_circle_3pts([ptf(pts[i]), ptf(pts[j]), ptf(pts[k])], center, radius);
    }
}

/// Ported from OpenCV 4.13 imgproc/src/shapedescr.cpp, `findSecondPoint<Point>`.
fn find_second_point(pts: &[PointI], i: usize, center: &mut (f32, f32), radius: &mut f32) {
    center.0 = pts[0].x.wrapping_add(pts[i].x) as f32 / 2.0;
    center.1 = pts[0].y.wrapping_add(pts[i].y) as f32 / 2.0;
    let dx = pts[0].x.wrapping_sub(pts[i].x) as f32;
    let dy = pts[0].y.wrapping_sub(pts[i].y) as f32;
    *radius = norm2f(dx, dy) as f32 / 2.0 + EPS;
    for j in 1..i {
        let dx = center.0 - pts[j].x as f32;
        let dy = center.1 - pts[j].y as f32;
        if norm2f(dx, dy) < *radius as f64 {
            continue;
        }
        find_third_point(pts, i, j, center, radius);
    }
}

/// The smallest circle enclosing `points` (Welzl's algorithm as OpenCV runs it, in
/// `float`), as `((cx, cy), radius)`.
///
/// No points gives `((0, 0), 0)`; one point gives that point and radius `1e-4`. Every
/// radius includes OpenCV's `+ 1e-4`, and the result depends on the order of the points.
///
/// Ported from OpenCV 4.13 imgproc/src/shapedescr.cpp, `cv::minEnclosingCircle` and
/// `findMinEnclosingCircle<Point>` (the `CV_32S` path).
pub fn min_enclosing_circle(points: &[PointI]) -> ((f32, f32), f32) {
    match points.len() {
        0 => return ((0.0, 0.0), 0.0),
        1 => return (ptf(points[0]), EPS),
        _ => {}
    }
    let pts = points;
    let mut center = (
        pts[0].x.wrapping_add(pts[1].x) as f32 / 2.0,
        pts[0].y.wrapping_add(pts[1].y) as f32 / 2.0,
    );
    let dx = pts[0].x.wrapping_sub(pts[1].x) as f32;
    let dy = pts[0].y.wrapping_sub(pts[1].y) as f32;
    let mut radius = norm2f(dx, dy) as f32 / 2.0 + EPS;
    for i in 2..pts.len() {
        let dx = pts[i].x as f32 - center.0;
        let dy = pts[i].y as f32 - center.1;
        let d = norm2f(dx, dy) as f32;
        if d < radius {
            continue;
        }
        find_second_point(pts, i, &mut center, &mut radius);
    }
    (center, radius)
}

/// Moments of a polygon (OpenCV `cv::Moments`): spatial moments `m..`, central moments
/// `mu..` and normalised central moments `nu..`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[allow(missing_docs)]
pub struct Moments {
    pub m00: f64,
    pub m10: f64,
    pub m01: f64,
    pub m20: f64,
    pub m11: f64,
    pub m02: f64,
    pub m30: f64,
    pub m21: f64,
    pub m12: f64,
    pub m03: f64,
    pub mu20: f64,
    pub mu11: f64,
    pub mu02: f64,
    pub mu30: f64,
    pub mu21: f64,
    pub mu12: f64,
    pub mu03: f64,
    pub nu20: f64,
    pub nu11: f64,
    pub nu02: f64,
    pub nu30: f64,
    pub nu21: f64,
    pub nu12: f64,
    pub nu03: f64,
}

/// Central and normalised moments from the spatial ones.
///
/// Ported from OpenCV 4.13 imgproc/src/moments.cpp, `completeMomentState`.
fn complete_moment_state(m: &mut Moments) {
    let (mut cx, mut cy, mut inv_m00) = (0.0, 0.0, 0.0);
    if m.m00.abs() > f64::EPSILON {
        inv_m00 = 1.0 / m.m00;
        cx = m.m10 * inv_m00;
        cy = m.m01 * inv_m00;
    }
    let mu20 = m.m20 - m.m10 * cx;
    let mut mu11 = m.m11 - m.m10 * cy;
    let mu02 = m.m02 - m.m01 * cy;
    m.mu20 = mu20;
    m.mu11 = mu11;
    m.mu02 = mu02;
    m.mu30 = m.m30 - cx * (3.0 * mu20 + cx * m.m10);
    mu11 += mu11;
    m.mu21 = m.m21 - cx * (mu11 + cx * m.m01) - cy * mu20;
    m.mu12 = m.m12 - cy * (mu11 + cy * m.m10) - cx * mu02;
    m.mu03 = m.m03 - cy * (3.0 * mu02 + cy * m.m01);

    let inv_sqrt_m00 = inv_m00.abs().sqrt();
    let s2 = inv_m00 * inv_m00;
    let s3 = s2 * inv_sqrt_m00;
    m.nu20 = m.mu20 * s2;
    m.nu11 = m.mu11 * s2;
    m.nu02 = m.mu02 * s2;
    m.nu30 = m.mu30 * s3;
    m.nu21 = m.mu21 * s3;
    m.nu12 = m.mu12 * s3;
    m.nu03 = m.mu03 * s3;
}

/// Moments of the polygon whose vertices are `contour` (`cv::moments(contour, false)` on
/// a point vector). The sign is made positive whatever the orientation. A polygon of
/// area at most `FLT_EPSILON` (fewer than three points, a line, a single pixel) gives all
/// zeros, so `m10 / m00` is NaN.
///
/// Ported from OpenCV 4.13 imgproc/src/moments.cpp, `contourMoments` (the `CV_32S` path).
pub fn moments(contour: &[PointI]) -> Moments {
    let mut m = Moments::default();
    let n = contour.len();
    if n == 0 {
        return m;
    }
    let (mut a00, mut a10, mut a01, mut a20, mut a11, mut a02) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut a30, mut a21, mut a12, mut a03) = (0.0, 0.0, 0.0, 0.0);
    let mut xi_1 = contour[n - 1].x as f64;
    let mut yi_1 = contour[n - 1].y as f64;
    let mut xi_12 = xi_1 * xi_1;
    let mut yi_12 = yi_1 * yi_1;
    for p in contour {
        let xi = p.x as f64;
        let yi = p.y as f64;
        let xi2 = xi * xi;
        let yi2 = yi * yi;
        let dxy = xi_1 * yi - xi * yi_1;
        let xii_1 = xi_1 + xi;
        let yii_1 = yi_1 + yi;

        a00 += dxy;
        a10 += dxy * xii_1;
        a01 += dxy * yii_1;
        a20 += dxy * (xi_1 * xii_1 + xi2);
        a11 += dxy * (xi_1 * (yii_1 + yi_1) + xi * (yii_1 + yi));
        a02 += dxy * (yi_1 * yii_1 + yi2);
        a30 += dxy * xii_1 * (xi_12 + xi2);
        a03 += dxy * yii_1 * (yi_12 + yi2);
        a21 +=
            dxy * (xi_12 * (3.0 * yi_1 + yi) + 2.0 * xi * xi_1 * yii_1 + xi2 * (yi_1 + 3.0 * yi));
        a12 +=
            dxy * (yi_12 * (3.0 * xi_1 + xi) + 2.0 * yi * yi_1 * xii_1 + yi2 * (xi_1 + 3.0 * xi));
        xi_1 = xi;
        yi_1 = yi;
        xi_12 = xi2;
        yi_12 = yi2;
    }

    if a00.abs() > f32::EPSILON as f64 {
        // The constants as the C++ writes them (rounded to double by the compiler).
        #[allow(clippy::excessive_precision)]
        let (mut d2, mut d6, mut d12, mut d24, mut d20, mut d60) = (
            0.5,
            0.16666666666666666666666666666667,
            0.083333333333333333333333333333333,
            0.041666666666666666666666666666667,
            0.05,
            0.016666666666666666666666666666667,
        );
        if a00 <= 0.0 {
            d2 = -d2;
            d6 = -d6;
            d12 = -d12;
            d24 = -d24;
            d20 = -d20;
            d60 = -d60;
        }
        m.m00 = a00 * d2;
        m.m10 = a10 * d6;
        m.m01 = a01 * d6;
        m.m20 = a20 * d12;
        m.m11 = a11 * d24;
        m.m02 = a02 * d12;
        m.m30 = a30 * d20;
        m.m21 = a21 * d60;
        m.m12 = a12 * d60;
        m.m03 = a03 * d20;
        complete_moment_state(&mut m);
    }
    m
}

/// Length of the polyline through `curve`, closed back to the first point if `closed`.
/// Each segment is computed in `float` (as OpenCV does) and summed in `double`.
///
/// Ported from OpenCV 4.13 imgproc/src/shapedescr.cpp, `cv::arcLength`.
pub fn arc_length(curve: &[PointI], closed: bool) -> f64 {
    let n = curve.len();
    if n <= 1 {
        return 0.0;
    }
    let last = if closed { n - 1 } else { 0 };
    let mut prev = ptf(curve[last]);
    let mut perimeter = 0.0f64;
    for &q in curve {
        let p = ptf(q);
        let dx = p.0 - prev.0;
        let dy = p.1 - prev.1;
        perimeter += (dx * dx + dy * dy).sqrt() as f64;
        prev = p;
    }
    perimeter
}

/// Area of the polygon `contour` by the shoelace formula; signed if `oriented` (positive
/// for points listed clockwise on screen with y down, as `findContours` lists a hole;
/// negative for an outer border), otherwise its absolute value.
///
/// Ported from OpenCV 4.13 imgproc/src/shapedescr.cpp, `cv::contourArea`.
pub fn contour_area(contour: &[PointI], oriented: bool) -> f64 {
    let n = contour.len();
    if n == 0 {
        return 0.0;
    }
    let mut prev = ptf(contour[n - 1]);
    let mut a00 = 0.0f64;
    for &q in contour {
        let p = ptf(q);
        a00 += prev.0 as f64 * p.1 as f64 - prev.1 as f64 * p.0 as f64;
        prev = p;
    }
    a00 *= 0.5;
    if !oriented {
        a00 = a00.abs();
    }
    a00
}

/// A point type `cv::convexHull` takes (`CV_32S` or `CV_32F`), with the operations of its
/// `Sklansky_` instantiation and of `CHullCmpPoints`.
trait HullPoint: Copy + PartialEq {
    /// `CHullCmpPoints` without the final comparison of addresses: x, then y.
    fn cmp_xy(a: Self, b: Self) -> std::cmp::Ordering;
    /// `a.y > b.y`.
    fn y_gt(a: Self, b: Self) -> bool;
    /// The sign of `next.y - cur.y` (`by` of `Sklansky_`).
    fn dy_sign(cur: Self, next: Self) -> i32;
    /// `Sklansky_`'s turn test at `cur`: the sign of the convexity of the edges `prev` to
    /// `cur` and `cur` to `next`, and whether the first edge vector is not zero.
    fn turn(prev: Self, cur: Self, next: Self) -> (i32, bool);
}

/// `Sklansky_<int, int64>`: integer edge vectors (wrapping, as the C++ `int` arithmetic for
/// coordinates near the `int` range), the convexity in 64 bits.
impl HullPoint for PointI {
    fn cmp_xy(a: PointI, b: PointI) -> std::cmp::Ordering {
        (a.x, a.y).cmp(&(b.x, b.y))
    }
    fn y_gt(a: PointI, b: PointI) -> bool {
        a.y > b.y
    }
    fn dy_sign(cur: PointI, next: PointI) -> i32 {
        next.y.wrapping_sub(cur.y).signum()
    }
    fn turn(prev: PointI, cur: PointI, next: PointI) -> (i32, bool) {
        let a = (cur.x.wrapping_sub(prev.x), cur.y.wrapping_sub(prev.y));
        let b = (next.x.wrapping_sub(cur.x), next.y.wrapping_sub(cur.y));
        let convexity = a.1 as i64 * b.0 as i64 - a.0 as i64 * b.1 as i64;
        (convexity.signum() as i32, a.0 != 0 || a.1 != 0)
    }
}

/// One half-hull scan over `sorted` (indices of points sorted by x then y), from `start`
/// towards `end`, writing the kept indices to `stack`; returns how many were kept.
///
/// Ported from OpenCV 4.13 imgproc/src/convhull.cpp, `Sklansky_<int, int64>` and
/// `Sklansky_<float, double>` (the turn test of each in [`HullPoint::turn`]).
fn sklansky<P: HullPoint>(
    pts: &[P],
    sorted: &[usize],
    start: usize,
    end: usize,
    stack: &mut [isize],
    nsign: i32,
    sign2: i32,
) -> usize {
    let at = |i: isize| pts[sorted[i as usize]];
    let (start, mut end) = (start as isize, end as isize);
    let incr: isize = if end > start { 1 } else { -1 };
    let mut pprev = start;
    let mut pcur = pprev + incr;
    let mut pnext = pcur + incr;
    let mut stacksize: usize = 3;

    if start == end || at(start) == at(end) {
        stack[0] = start;
        return 1;
    }
    stack[0] = pprev;
    stack[1] = pcur;
    stack[2] = pnext;
    end += incr;

    while pnext != end {
        if P::dy_sign(at(pcur), at(pnext)) != nsign {
            let (convexity, a_nonzero) = P::turn(at(pprev), at(pcur), at(pnext));
            if convexity == sign2 && a_nonzero {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack[stacksize] = pnext;
                stacksize += 1;
            } else if pprev == start {
                pcur = pnext;
                stack[1] = pcur;
                pnext += incr;
                stack[2] = pnext;
            } else {
                stack[stacksize - 2] = pnext;
                pcur = pprev;
                pprev = stack[stacksize - 4];
                stacksize -= 1;
            }
        } else {
            pnext += incr;
            stack[stacksize - 1] = pnext;
        }
    }
    stacksize - 1
}

/// The convex hull of `points`, as points in OpenCV's order (`cv::convexHull(points,
/// hull)` with the defaults `clockwise = false`, `returnPoints = true`): clockwise on
/// screen with y down (counter-clockwise in OpenCV's y-up convention), cyclically shifted
/// so that the indices of the hull points in `points` run monotonically where OpenCV
/// manages that. Collinear points on an edge and repeated points are left out.
///
/// Ported from OpenCV 4.13 imgproc/src/convhull.cpp, `cv::convexHull` (the `CV_32S` path
/// returning points).
pub fn convex_hull(points: &[PointI]) -> Vec<PointI> {
    convex_hull_indices(points)
        .into_iter()
        .map(|i| points[i])
        .collect()
}

/// The indices into `points` of the hull [`convex_hull`] returns, in the same order.
///
/// Ported from OpenCV 4.13 imgproc/src/convhull.cpp, `cv::convexHull`.
pub fn convex_hull_indices(points: &[PointI]) -> Vec<usize> {
    hull_indices(points, false)
}

/// The indices into `points` of the hull `cv::convexHull(points, hull, clockwise)` returns,
/// in its order, for either point type.
///
/// Ported from OpenCV 4.13 imgproc/src/convhull.cpp, `cv::convexHull`.
fn hull_indices<P: HullPoint>(points: &[P], clockwise: bool) -> Vec<usize> {
    let total = points.len();
    if total == 0 {
        return Vec::new();
    }
    // CHullCmpPoints: x, then y, then address (the original index).
    let mut sorted: Vec<usize> = (0..total).collect();
    sorted.sort_by(|&a, &b| P::cmp_xy(points[a], points[b]).then(a.cmp(&b)));
    let (mut miny_ind, mut maxy_ind) = (0usize, 0usize);
    for i in 1..total {
        let p = points[sorted[i]];
        if P::y_gt(points[sorted[miny_ind]], p) {
            miny_ind = i;
        }
        if P::y_gt(p, points[sorted[maxy_ind]]) {
            maxy_ind = i;
        }
    }

    if points[sorted[0]] == points[sorted[total - 1]] {
        // All points equal: the C++ stores index 0 without mapping it through the sort,
        // so the hull is the first input point.
        return vec![0];
    }
    let mut hullbuf: Vec<usize> = Vec::with_capacity(total);
    let mut tl = vec![0isize; total + 2];
    let mut tr = vec![0isize; total + 2];
    let mut tl_count = sklansky(points, &sorted, 0, maxy_ind, &mut tl, -1, 1);
    let mut tr_count = sklansky(points, &sorted, total - 1, maxy_ind, &mut tr, -1, -1);
    if !clockwise {
        std::mem::swap(&mut tl, &mut tr);
        std::mem::swap(&mut tl_count, &mut tr_count);
    }
    for &v in &tl[..tl_count.saturating_sub(1)] {
        hullbuf.push(v as usize);
    }
    for i in (1..tr_count).rev() {
        hullbuf.push(tr[i] as usize);
    }
    let stop_idx: isize = if tr_count > 2 {
        tr[1]
    } else if tl_count > 2 {
        tl[tl_count - 2]
    } else {
        -1
    };

    let mut bl = vec![0isize; total + 2];
    let mut br = vec![0isize; total + 2];
    let mut bl_count = sklansky(points, &sorted, 0, miny_ind, &mut bl, 1, -1);
    let mut br_count = sklansky(points, &sorted, total - 1, miny_ind, &mut br, 1, 1);
    if clockwise {
        std::mem::swap(&mut bl, &mut br);
        std::mem::swap(&mut bl_count, &mut br_count);
    }
    if stop_idx >= 0 {
        let check_idx: isize = if bl_count > 2 {
            bl[1]
        } else if bl_count + br_count > 2 {
            br[2 - bl_count]
        } else {
            -1
        };
        if check_idx == stop_idx
            || (check_idx >= 0
                && points[sorted[check_idx as usize]] == points[sorted[stop_idx as usize]])
        {
            bl_count = bl_count.min(2);
            br_count = br_count.min(2);
        }
    }
    for &v in &bl[..bl_count.saturating_sub(1)] {
        hullbuf.push(v as usize);
    }
    for i in (1..br_count).rev() {
        hullbuf.push(br[i] as usize);
    }

    for h in &mut hullbuf {
        *h = sorted[*h];
    }

    // Cyclic shift towards ascending or descending indices.
    let nout = hullbuf.len();
    if nout >= 3 {
        let (mut min_idx, mut max_idx, mut lt) = (0usize, 0usize, 0usize);
        for i in 1..nout {
            let idx = hullbuf[i];
            lt += usize::from(hullbuf[i - 1] < idx);
            if lt > 1 && lt + 2 <= i {
                break;
            }
            if idx < hullbuf[min_idx] {
                min_idx = i;
            }
            if idx > hullbuf[max_idx] {
                max_idx = i;
            }
        }
        let mmdist = max_idx.abs_diff(min_idx);
        if (mmdist == 1 || mmdist == nout - 1) && (lt <= 1 || lt + 2 >= nout) {
            let ascending = (max_idx + 1) % nout == min_idx;
            let i0 = if ascending { min_idx } else { max_idx };
            let mut j = i0;
            if i0 > 0 {
                let mut tmp = vec![0usize; nout];
                let mut i = 0;
                while i < nout {
                    let curr_idx = hullbuf[j];
                    tmp[i] = curr_idx;
                    let next_j = if j + 1 < nout { j + 1 } else { 0 };
                    let next_idx = hullbuf[next_j];
                    if i < nout - 1 && (ascending != (curr_idx < next_idx)) {
                        break;
                    }
                    j = next_j;
                    i += 1;
                }
                if i == nout {
                    hullbuf = tmp;
                }
            }
        }
    }
    hullbuf
}

/// Area of the polygon of float points `contour` (`std::vector<cv::Point2f>`) by the
/// shoelace formula in `double`; signed if `oriented`, otherwise its absolute value.
///
/// Ported from OpenCV 4.13 imgproc/src/shapedescr.cpp, `cv::contourArea` (the `CV_32F`
/// path).
#[must_use]
pub fn contour_area_f32(contour: &[(f32, f32)], oriented: bool) -> f64 {
    let n = contour.len();
    if n == 0 {
        return 0.0;
    }
    let mut prev = contour[n - 1];
    let mut a00 = 0.0f64;
    for &p in contour {
        a00 += f64::from(prev.0) * f64::from(p.1) - f64::from(prev.1) * f64::from(p.0);
        prev = p;
    }
    a00 *= 0.5;
    if !oriented {
        a00 = a00.abs();
    }
    a00
}

/// `cv::normalize(Vec2f)`: `v * (1 / |v|)` with the norm in `double`, each element rounded
/// to `float`; a zero vector stays zero.
fn normalize2(v: (f32, f32)) -> (f32, f32) {
    let nv = (f64::from(v.0) * f64::from(v.0) + f64::from(v.1) * f64::from(v.1)).sqrt();
    let s = if nv != 0.0 { 1.0 / nv } else { 0.0 };
    #[allow(clippy::cast_possible_truncation)]
    let r = ((f64::from(v.0) * s) as f32, (f64::from(v.1) * s) as f32);
    r
}

fn sign_f32(v: f32) -> i32 {
    i32::from(v > 0.0) - i32::from(v < 0.0)
}

fn sign_f64(v: f64) -> i32 {
    i32::from(v > 0.0) - i32::from(v < 0.0)
}

/// `Sklansky_<float, double>`: the edge vectors are normalised before the turn test.
/// `CHullCmpPoints<float>` compares with `<`, so a NaN coordinate compares equal here.
#[allow(clippy::float_cmp)]
impl HullPoint for (f32, f32) {
    fn cmp_xy(a: (f32, f32), b: (f32, f32)) -> std::cmp::Ordering {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
    fn y_gt(a: (f32, f32), b: (f32, f32)) -> bool {
        a.1 > b.1
    }
    fn dy_sign(cur: (f32, f32), next: (f32, f32)) -> i32 {
        sign_f32(next.1 - cur.1)
    }
    fn turn(prev: (f32, f32), cur: (f32, f32), next: (f32, f32)) -> (i32, bool) {
        let a = normalize2((cur.0 - prev.0, cur.1 - prev.1));
        let b = normalize2((next.0 - cur.0, next.1 - cur.1));
        let convexity = f64::from(a.1) * f64::from(b.0) - f64::from(a.0) * f64::from(b.1);
        (sign_f64(convexity), a.0 != 0.0 || a.1 != 0.0)
    }
}

/// `cv::convexHull(points, hull, clockwise)` for float points (`CV_32FC2`), returning the
/// hull's points in OpenCV's order: clockwise on screen (y down) if `clockwise`, else
/// counter-clockwise, cyclically shifted as OpenCV shifts it. Collinear and repeated points
/// are left out. Points must be finite (a NaN breaks the sort's order, as in OpenCV).
///
/// Ported from OpenCV 4.13 imgproc/src/convhull.cpp, `cv::convexHull` (the `CV_32F` path
/// returning points) and `Sklansky_<float, double>`.
#[must_use]
pub fn convex_hull_f32(points: &[(f32, f32)], clockwise: bool) -> Vec<(f32, f32)> {
    hull_indices(points, clockwise)
        .into_iter()
        .map(|i| points[i])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: i32, y: i32) -> PointI {
        PointI::new(x, y)
    }

    #[test]
    fn square() {
        let sq = [p(0, 0), p(4, 0), p(4, 4), p(0, 4)];
        assert_eq!(contour_area(&sq, true), 16.0);
        assert_eq!(contour_area(&sq[..], false), 16.0);
        assert_eq!(arc_length(&sq, true), 16.0);
        assert_eq!(arc_length(&sq, false), 12.0);
        let m = moments(&sq);
        assert_eq!((m.m00, m.m10 / m.m00, m.m01 / m.m00), (16.0, 2.0, 2.0));
        let ((cx, cy), r) = min_enclosing_circle(&sq);
        assert_eq!((cx, cy), (2.0, 2.0));
        assert!((r - 8f32.sqrt()).abs() < 1e-3);
        assert_eq!(convex_hull(&sq).len(), 4);
    }
}
