//! `cv::minAreaRect` of an integer contour, which `cv::checkChessboard` (the
//! `CALIB_CB_FAST_CHECK` test of `cv::findChessboardCorners`) uses on every outer contour.
//! Ported from OpenCV 4.13 so that the result is the same bit for bit (checked against the
//! 4.13 libraries, `chessops-*`).

use super::contours::PointI;
use super::shape::convex_hull;

/// A rotated rectangle (`cv::RotatedRect`): centre, width, height and angle in degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotatedRect {
    pub center: (f32, f32),
    pub width: f32,
    pub height: f32,
    pub angle: f32,
}

fn rotate90_ccw(v: (f32, f32)) -> (f32, f32) {
    (-v.1, v.0)
}

fn rotate90_cw(v: (f32, f32)) -> (f32, f32) {
    (v.1, -v.0)
}

fn rotate180(v: (f32, f32)) -> (f32, f32) {
    (-v.0, -v.1)
}

/// Whether the first vector is to the right (clockwise) of the second.
fn first_vec_is_right(v1: (f32, f32), v2: (f32, f32)) -> bool {
    let t = rotate90_cw(v1);
    t.0 * v2.0 + t.1 * v2.1 < 0.0
}

/// `rotatingCalipers(points, n, orientation, CALIPERS_MINAREARECT, out)`: the corner and the
/// two side vectors of the smallest rectangle around the convex polygon `points`.
///
/// Ported from OpenCV 4.13 imgproc/src/rotcalipers.cpp, `rotatingCalipers`.
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
fn rotating_calipers_min_area(points: &[(f32, f32)], orientation: f32) -> [(f32, f32); 3] {
    let n = points.len();
    let mut minarea = f32::MAX;
    let mut vect = vec![(0f32, 0f32); n];
    let mut inv_len = vec![0f32; n];
    let (mut left, mut bottom, mut right, mut top) = (0usize, 0usize, 0usize, 0usize);
    let mut pt0 = points[0];
    let (mut left_x, mut right_x, mut top_y, mut bottom_y) = (pt0.0, pt0.0, pt0.1, pt0.1);
    for i in 0..n {
        if pt0.0 < left_x {
            left_x = pt0.0;
            left = i;
        }
        if pt0.0 > right_x {
            right_x = pt0.0;
            right = i;
        }
        if pt0.1 > top_y {
            top_y = pt0.1;
            top = i;
        }
        if pt0.1 < bottom_y {
            bottom_y = pt0.1;
            bottom = i;
        }
        let pt = points[if i + 1 < n { i + 1 } else { 0 }];
        let dx = f64::from(pt.0 - pt0.0);
        let dy = f64::from(pt.1 - pt0.1);
        #[allow(clippy::cast_possible_truncation)]
        {
            vect[i] = (dx as f32, dy as f32);
            inv_len[i] = (1.0 / (dx * dx + dy * dy).sqrt()) as f32;
        }
        pt0 = pt;
    }
    // OpenCV starts the base vector at (orientation, 0); every step below replaces it, so
    // the orientation never matters for the rectangle.
    let _ = orientation;
    let mut seq = [bottom, right, top, left];
    // buffer: [seq[3] as int, base_a, width, base_b, height, seq[0] as int, area]
    let mut b_left = 0usize;
    let mut b_bottom = 0usize;
    let (mut b_a, mut b_w, mut b_b, mut b_h) = (0f32, 0f32, 0f32, 0f32);
    for _ in 0..n {
        let rot = [
            vect[seq[0]],
            rotate90_cw(vect[seq[1]]),
            rotate180(vect[seq[2]]),
            rotate90_ccw(vect[seq[3]]),
        ];
        let mut main = 0usize;
        for i in 1..4 {
            if first_vec_is_right(rot[i], rot[main]) {
                main = i;
            }
        }
        let p = seq[main];
        let lead_x = vect[p].0 * inv_len[p];
        let lead_y = vect[p].1 * inv_len[p];
        let (base_a, base_b) = match main {
            0 => (lead_x, lead_y),
            1 => (lead_y, -lead_x),
            2 => (-lead_x, -lead_y),
            _ => (-lead_y, lead_x),
        };
        seq[main] += 1;
        if seq[main] == n {
            seq[main] = 0;
        }
        let dx = points[seq[1]].0 - points[seq[3]].0;
        let dy = points[seq[1]].1 - points[seq[3]].1;
        let width = dx * base_a + dy * base_b;
        let dx = points[seq[2]].0 - points[seq[0]].0;
        let dy = points[seq[2]].1 - points[seq[0]].1;
        let height = -dx * base_b + dy * base_a;
        let area = width * height;
        if area <= minarea {
            minarea = area;
            b_left = seq[3];
            b_a = base_a;
            b_w = width;
            b_b = base_b;
            b_h = height;
            b_bottom = seq[0];
        }
    }
    let a1 = b_a;
    let b1 = b_b;
    let a2 = -b_b;
    let b2 = b_a;
    let c1 = a1 * points[b_left].0 + points[b_left].1 * b1;
    let c2 = a2 * points[b_bottom].0 + points[b_bottom].1 * b2;
    let idet = 1.0f32 / (a1 * b2 - a2 * b1);
    let px = (c1 * b2 - c2 * b1) * idet;
    let py = (a1 * c2 - a2 * c1) * idet;
    [(px, py), (a1 * b_w, b1 * b_w), (a2 * b_h, b2 * b_h)]
}

/// `cv::minAreaRect(points)` for integer points: the rotated rectangle of least area
/// around them, as OpenCV returns it (angle in `[-90, 0)` degrees). No points give a zero
/// rectangle with angle -90.
///
/// Ported from OpenCV 4.13 imgproc/src/rotcalipers.cpp, `cv::minAreaRect`.
#[must_use]
#[allow(clippy::float_cmp)]
pub fn min_area_rect(points: &[PointI]) -> RotatedRect {
    #[allow(clippy::cast_precision_loss)]
    let hull: Vec<(f32, f32)> = convex_hull(points)
        .into_iter()
        .map(|p| (p.x as f32, p.y as f32))
        .collect();
    let n = hull.len();
    let mut angle = -std::f64::consts::PI / 2.0;
    let mut r = RotatedRect {
        center: (0.0, 0.0),
        width: 0.0,
        height: 0.0,
        angle: 0.0,
    };
    if n > 2 {
        let out = rotating_calipers_min_area(&hull, 1.0);
        r.center.0 = out[0].0 + (out[1].0 + out[2].0) * 0.5;
        r.center.1 = out[0].1 + (out[1].1 + out[2].1) * 0.5;
        #[allow(clippy::cast_possible_truncation)]
        {
            r.width = (f64::from(out[2].0) * f64::from(out[2].0)
                + f64::from(out[2].1) * f64::from(out[2].1))
            .sqrt() as f32;
            r.height = (f64::from(out[1].0) * f64::from(out[1].0)
                + f64::from(out[1].1) * f64::from(out[1].1))
            .sqrt() as f32;
        }
        if out[1].0 == 0.0 && out[1].1 > 0.0 {
            std::mem::swap(&mut r.width, &mut r.height);
        } else {
            angle = -f64::from(out[1].0).atan2(f64::from(out[1].1));
        }
    } else if n == 2 {
        r.center.0 = (hull[0].0 + hull[1].0) * 0.5;
        r.center.1 = (hull[0].1 + hull[1].1) * 0.5;
        let dx = f64::from(hull[0].0 - hull[1].0);
        let dy = f64::from(hull[0].1 - hull[1].1);
        r.width = 0.0;
        #[allow(clippy::cast_possible_truncation)]
        {
            r.height = (dx * dx + dy * dy).sqrt() as f32;
        }
        if dx == 0.0 {
            std::mem::swap(&mut r.width, &mut r.height);
        } else if dy < 0.0 {
            angle = dy.atan2(dx);
            std::mem::swap(&mut r.width, &mut r.height);
        } else if dy > 0.0 {
            angle = -dx.atan2(dy);
        }
    } else if n == 1 {
        r.center = hull[0];
    }
    #[allow(clippy::cast_possible_truncation)]
    {
        r.angle = (angle * 180.0 / std::f64::consts::PI) as f32;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_aligned_and_degenerate() {
        let sq = [
            PointI::new(0, 0),
            PointI::new(4, 0),
            PointI::new(4, 2),
            PointI::new(0, 2),
        ];
        let r = min_area_rect(&sq);
        assert_eq!(r.center, (2.0, 1.0));
        assert_eq!(r.width * r.height, 8.0);
        let e = min_area_rect(&[]);
        assert_eq!(e.width, 0.0);
        let one = min_area_rect(&[PointI::new(3, 5)]);
        assert_eq!(one.center, (3.0, 5.0));
        let two = min_area_rect(&[PointI::new(0, 0), PointI::new(3, 4)]);
        assert_eq!(two.width.max(two.height), 5.0);
    }
}
