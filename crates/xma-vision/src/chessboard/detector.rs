//! The quad-based search of `cv::findChessboardCorners`: OpenCV's `ChessBoardDetector`
//! (`modules/calib3d/src/calibinit.cpp`, OpenCV 4.13), ported member by member. Quads and
//! corners live in two arrays as in OpenCV's `AutoBuffer`s; OpenCV's pointers become
//! indices into them.

// The nesting of OpenCV's conditions is kept, to compare with the source.
#![allow(clippy::collapsible_if)]

use crate::image::ImageU8;
use crate::imgproc::approx::{approx_poly_dp, bounding_rect_area, is_contour_convex};
use crate::imgproc::contours::{ChainApprox, PointI, find_contours_ccomp};
use crate::imgproc::shape::{arc_length, contour_area, contour_area_f32, convex_hull_f32};

type P2 = (f32, f32);

fn sub(a: P2, b: P2) -> P2 {
    (a.0 - b.0, a.1 - b.1)
}

fn add(a: P2, b: P2) -> P2 {
    (a.0 + b.0, a.1 + b.1)
}

/// `normL2Sqr<float>(Point2f)`: `x * x + y * y` in `float`.
fn norm_sqr(v: P2) -> f32 {
    v.0 * v.0 + v.1 * v.1
}

/// `ChessBoardCorner`.
#[derive(Clone, Copy, Debug, Default)]
struct Corner {
    pt: P2,
    row: i32,
    count: i32,
    neighbors: [Option<usize>; 4],
}

impl Corner {
    fn new(pt: P2) -> Self {
        Corner {
            pt,
            ..Corner::default()
        }
    }
}

/// `ChessBoardQuad`.
#[derive(Clone, Copy, Debug)]
struct Quad {
    count: i32,
    group_idx: i32,
    row: i32,
    col: i32,
    ordered: bool,
    edge_sqr_len: f32,
    corners: [usize; 4],
    neighbors: [Option<usize>; 4],
}

impl Quad {
    fn new(group_idx: i32) -> Self {
        Quad {
            count: 0,
            group_idx,
            row: 0,
            col: 0,
            ordered: false,
            edge_sqr_len: 0.0,
            corners: [0; 4],
            neighbors: [None; 4],
        }
    }
}

/// `ChessBoardDetector`.
pub(super) struct Detector {
    /// `pattern_size.width`, `pattern_size.height` (both at least 3; their product fits
    /// in `i32`).
    width: i32,
    height: i32,
    quads: Vec<Quad>,
    corners: Vec<Corner>,
    /// `all_quads.size()`: the allocated number of quads.
    max_quad_buf_size: usize,
}

impl Detector {
    pub(super) fn new(width: i32, height: i32) -> Self {
        Detector {
            width,
            height,
            quads: Vec::new(),
            corners: Vec::new(),
            max_quad_buf_size: 0,
        }
    }

    fn area(&self) -> i32 {
        self.width * self.height
    }

    /// `ChessBoardDetector::generateQuads`: the four-sided holes of the binary image.
    ///
    /// Ported from calibinit.cpp, `ChessBoardDetector::generateQuads`.
    pub(super) fn generate_quads(&mut self, image: &ImageU8, filter_quads: bool, dilations: i32) {
        self.quads.clear();
        self.corners.clear();
        self.max_quad_buf_size = 0;
        let min_area = 25i64;
        let (contours, hierarchy) =
            find_contours_ccomp(image, ChainApprox::Simple, PointI::default());
        if contours.is_empty() {
            return;
        }
        let mut child_counter = vec![0i32; contours.len()];
        let mut board_idx: i32 = -1;
        let mut contour_quads: Vec<([PointI; 4], i32)> = Vec::new();
        for idx in (0..contours.len()).rev() {
            let parent_idx = hierarchy[idx][3];
            if hierarchy[idx][2] != -1 || parent_idx == -1 {
                continue; // holes only (no child contours and with parent)
            }
            let contour = &contours[idx];
            if bounding_rect_area(contour) < min_area {
                continue;
            }
            let mut approx = contour.clone();
            let mut level = 1;
            while approx.len() > 4 && level <= 7 {
                approx = approx_poly_dp(&approx, f64::from(level), true);
                level += 1;
            }
            if approx.len() != 4 || !is_contour_convex(&approx) {
                continue;
            }
            let pt = [approx[0], approx[1], approx[2], approx[3]];
            if filter_quads {
                let p = arc_length(&approx, true);
                let area = contour_area(&approx, false);
                // sqrt(normL2Sqr<double>(Point)): the int dot product, in i64 here.
                #[allow(clippy::cast_precision_loss)]
                let d = |a: PointI, b: PointI| {
                    let dx = i64::from(a.x) - i64::from(b.x);
                    let dy = i64::from(a.y) - i64::from(b.y);
                    ((dx * dx + dy * dy) as f64).sqrt()
                };
                let d1 = d(pt[0], pt[2]);
                let d2 = d(pt[1], pt[3]);
                let d3 = d(pt[0], pt[1]);
                let d4 = d(pt[1], pt[2]);
                #[allow(clippy::cast_precision_loss)]
                let min_area_f = min_area as f64;
                if !(d3 * 4.0 > d4
                    && d4 * 4.0 > d3
                    && d3 * d4 < area * 1.5
                    && area > min_area_f
                    && d1 >= 0.15 * p
                    && d2 >= 0.15 * p)
                {
                    continue;
                }
            }
            let pi = parent_idx as usize;
            child_counter[pi] += 1;
            if board_idx != parent_idx
                && (board_idx < 0 || child_counter[board_idx as usize] < child_counter[pi])
            {
                board_idx = parent_idx;
            }
            contour_quads.push((pt, parent_idx));
        }

        let total = contour_quads.len();
        self.max_quad_buf_size = 2usize.max(total * 3);
        self.corners = vec![Corner::default(); self.max_quad_buf_size * 4];
        let comp = 2 * dilations;
        for (pt, parent) in &contour_quads {
            if filter_quads && *parent != board_idx {
                continue;
            }
            let quad_idx = self.quads.len();
            let mut q = Quad::new(-1);
            for (i, p) in pt.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let c = Corner::new((p.x as f32, p.y as f32));
                self.corners[quad_idx * 4 + i] = c;
                q.corners[i] = quad_idx * 4 + i;
            }
            q.edge_sqr_len = f32::MAX;
            for i in 0..4 {
                let d = norm_sqr(sub(
                    self.corners[q.corners[i]].pt,
                    self.corners[q.corners[(i + 1) & 3]].pt,
                ));
                q.edge_sqr_len = q.edge_sqr_len.min(d);
            }
            #[allow(clippy::cast_precision_loss)]
            {
                q.edge_sqr_len += 2.0 * q.edge_sqr_len.sqrt() * comp as f32 + (comp * comp) as f32;
            }
            self.quads.push(q);
        }
    }

    /// `NeighborsFinder::findCornerNeighbor`. `pts` are the quad corners as they were when
    /// the search began (OpenCV's `all_quads_pts`, the points of its k-d tree); `by_x` holds
    /// them as (x, index), sorted by x.
    #[allow(clippy::too_many_arguments)]
    fn find_corner_neighbor(
        &self,
        pts: &[P2],
        by_x: &[(f32, usize)],
        quad_idx: usize,
        corner_idx: usize,
        corner_pt: P2,
        min_sqr_dist: &mut f32,
        sqr_radius: f32,
        closest_quad_idx: &mut usize,
        closest_corner_idx: &mut usize,
        closest_corner_pt: &mut P2,
    ) -> bool {
        const THRESH_SQR_SCALE: f32 = 2.0;
        let cur = &self.quads[quad_idx];
        // radiusSearch with cvflann's KDTreeSingleIndex (exact search, L2_Simple<float>):
        // every point with squared distance below the radius, sorted by distance then
        // index (RadiusUniqueResultSet).
        // The points are taken from those whose x lies within a slightly widened radius
        // (`by_x` is sorted by x), then tested with OpenCV's float distance.
        let reach = f64::from(sqr_radius.max(0.0)).sqrt() * 1.001 + 1e-3;
        let lo_x = f64::from(corner_pt.0) - reach;
        let hi_x = f64::from(corner_pt.0) + reach;
        let first = by_x.partition_point(|&(x, _)| f64::from(x) < lo_x);
        let mut cand: Vec<(f32, usize)> = by_x[first..]
            .iter()
            .take_while(|&&(x, _)| f64::from(x) <= hi_x)
            .filter_map(|&(_, i)| {
                let p = pts[i];
                let dx = corner_pt.0 - p.0;
                let dy = corner_pt.1 - p.1;
                let d = dx * dx + dy * dy;
                (d < sqr_radius).then_some((d, i))
            })
            .collect();
        cand.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut closest_neighbor_idx: Option<usize> = None;
        let mut closest_quad: Option<usize> = None;
        let cpt = |c: usize| self.corners[c].pt;
        for &(_, neighbor_idx) in &cand {
            let k = neighbor_idx >> 2;
            if k == quad_idx {
                continue;
            }
            let q_k = &self.quads[k];
            let j = neighbor_idx & 3;
            if q_k.neighbors[j].is_some() {
                continue;
            }
            let neighbor_pt = pts[neighbor_idx];
            let sqr_dist = norm_sqr(sub(corner_pt, neighbor_pt));
            if sqr_dist <= cur.edge_sqr_len * THRESH_SQR_SCALE
                && sqr_dist <= q_k.edge_sqr_len * THRESH_SQR_SCALE
            {
                if q_k.edge_sqr_len > 16.0 * cur.edge_sqr_len
                    || cur.edge_sqr_len > 16.0 * q_k.edge_sqr_len
                {
                    continue;
                }
                let c = |o: usize| cpt(cur.corners[(corner_idx + o) & 3]);
                let half = |p: P2| (p.0 / 2.0, p.1 / 2.0);
                let mid1 = half(add(c(0), c(1)));
                let mid2 = half(add(c(2), c(3)));
                if !same_side(mid1, mid2, corner_pt, neighbor_pt) {
                    continue;
                }
                let mid3 = half(add(c(1), c(2)));
                let mid4 = half(add(c(3), c(0)));
                if !same_side(mid3, mid4, corner_pt, neighbor_pt) {
                    continue;
                }
                let diag = cpt(q_k.corners[(j + 2) & 3]);
                if !same_side(mid1, mid2, corner_pt, diag) {
                    continue;
                }
                if !same_side(mid3, mid4, corner_pt, diag) {
                    continue;
                }
                closest_neighbor_idx = Some(neighbor_idx);
                *closest_quad_idx = k;
                *closest_corner_idx = j;
                closest_quad = Some(k);
                *min_sqr_dist = sqr_dist;
                break;
            }
        }
        if let (Some(cn), Some(cq)) = (closest_neighbor_idx, closest_quad) {
            if *min_sqr_dist < f32::MAX {
                let closest = &self.quads[cq];
                if cur.count >= 4 || closest.count >= 4 {
                    return false;
                }
                *closest_corner_pt = pts[cn];
                for j in 0..4 {
                    if cur.neighbors[j] == Some(cq) {
                        return false;
                    }
                    if norm_sqr(sub(*closest_corner_pt, pts[(quad_idx << 2) + j])) < *min_sqr_dist {
                        return false;
                    }
                }
                if closest.neighbors.contains(&Some(quad_idx)) {
                    return false;
                }
                return true;
            }
        }
        false
    }

    /// `ChessBoardDetector::findQuadNeighbors`.
    fn find_quad_neighbors(&mut self) {
        let n = self.quads.len();
        let pts: Vec<P2> = (0..n * 4)
            .map(|i| self.corners[self.quads[i >> 2].corners[i & 3]].pt)
            .collect();
        let mut by_x: Vec<(f32, usize)> = pts.iter().enumerate().map(|(i, p)| (p.0, i)).collect();
        by_x.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for idx in 0..n {
            for i in 0..4 {
                if self.quads[idx].neighbors[i].is_some() {
                    continue;
                }
                let pt = pts[(idx << 2) + i];
                let mut min_sqr_dist = f32::MAX;
                let mut cq = 0usize;
                let mut cc = 0usize;
                let mut cpt = (0f32, 0f32);
                let sqr_radius = self.quads[idx].edge_sqr_len * 2.0 + 1.0;
                if !self.find_corner_neighbor(
                    &pts,
                    &by_x,
                    idx,
                    i,
                    pt,
                    &mut min_sqr_dist,
                    sqr_radius,
                    &mut cq,
                    &mut cc,
                    &mut cpt,
                ) {
                    continue;
                }
                let sqr_radius = min_sqr_dist + 1.0;
                min_sqr_dist = f32::MAX;
                let mut ccq = 0usize;
                let mut ccc = 0usize;
                let mut ccpt = (0f32, 0f32);
                if !self.find_corner_neighbor(
                    &pts,
                    &by_x,
                    cq,
                    cc,
                    cpt,
                    &mut min_sqr_dist,
                    sqr_radius,
                    &mut ccq,
                    &mut ccc,
                    &mut ccpt,
                ) {
                    continue;
                }
                if ccq != idx || ccc != i || ccpt != pt {
                    continue;
                }
                let corner = self.quads[cq].corners[cc];
                let m = add(pt, cpt);
                self.corners[corner].pt = (m.0 * 0.5, m.1 * 0.5);
                let q = &mut self.quads[idx];
                q.count += 1;
                q.neighbors[i] = Some(cq);
                q.corners[i] = corner;
                let c = &mut self.quads[cq];
                c.count += 1;
                c.neighbors[cc] = Some(idx);
            }
        }
    }

    /// `ChessBoardDetector::findConnectedQuads`.
    fn find_connected_quads(&mut self, out: &mut Vec<usize>, group_idx: i32) {
        out.clear();
        let mut stack: Vec<usize> = Vec::new();
        for i in 0..self.quads.len() {
            let q = &self.quads[i];
            if q.count <= 0 || q.group_idx >= 0 {
                continue;
            }
            stack.push(i);
            out.push(i);
            self.quads[i].group_idx = group_idx;
            self.quads[i].ordered = false;
            while let Some(q) = stack.pop() {
                for k in 0..4 {
                    if let Some(nb) = self.quads[q].neighbors[k] {
                        let n = &mut self.quads[nb];
                        if n.count > 0 && n.group_idx < 0 {
                            stack.push(nb);
                            out.push(nb);
                            n.group_idx = group_idx;
                            n.ordered = false;
                        }
                    }
                }
            }
            break;
        }
    }

    /// `ChessBoardDetector::orderQuad`: rotates the quad's corners and neighbours so that the
    /// corner at `corner`'s position has index `common`.
    fn order_quad(&mut self, quad: usize, corner: usize, common: usize) {
        let pt = self.corners[corner].pt;
        let mut tc = 0usize;
        while tc < 4 {
            if self.corners[self.quads[quad].corners[tc]].pt == pt {
                break;
            }
            tc += 1;
        }
        let q = &mut self.quads[quad];
        while tc != common {
            q.corners.rotate_right(1);
            q.neighbors.rotate_right(1);
            tc = (tc + 1) & 3;
        }
    }

    /// `ChessBoardDetector::orderFoundConnectedQuads`.
    #[allow(clippy::too_many_lines)]
    fn order_found_connected_quads(&mut self, quads: &mut Vec<usize>) -> usize {
        let max_quad_buf_size = self.max_quad_buf_size;
        let mut quad_count = quads.len();
        let Some(&start) = quads.iter().find(|&&q| self.quads[q].count == 4) else {
            return 0; // no 4-connected quad
        };
        let (mut row_min, mut col_min, mut row_max, mut col_max) = (0i32, 0i32, 0i32, 0i32);
        let mut stack = vec![start];
        {
            let s = &mut self.quads[start];
            s.row = 0;
            s.col = 0;
            s.ordered = true;
        }
        while let Some(q) = stack.pop() {
            let mut col = self.quads[q].col;
            let mut row = self.quads[q].row;
            row_max = row_max.max(row);
            row_min = row_min.min(row);
            col_max = col_max.max(col);
            col_min = col_min.min(col);
            for i in 0..4 {
                let neighbor = self.quads[q].neighbors[i];
                match i {
                    0 => {
                        row -= 1;
                        col -= 1;
                    }
                    1 => col += 2,
                    2 => row += 2,
                    _ => col -= 2,
                }
                if let Some(nb) = neighbor {
                    if !self.quads[nb].ordered && self.quads[nb].count == 4 {
                        let c = self.quads[q].corners[i];
                        self.order_quad(nb, c, (i + 2) & 3);
                        let n = &mut self.quads[nb];
                        n.ordered = true;
                        n.row = row;
                        n.col = col;
                        stack.push(nb);
                    }
                }
            }
        }

        let mut w = self.width - 1;
        let mut h = self.height - 1;
        let drow = row_max - row_min + 1;
        let dcol = col_max - col_min + 1;
        if (w > h && dcol < drow) || (w < h && drow < dcol) {
            h = self.width - 1;
            w = self.height - 1;
        }
        if dcol < w || drow < h {
            return 0; // too few inner quad rows/cols
        }

        let mut found = 0;
        for i in 0..quad_count {
            let qi = quads[i];
            if self.quads[qi].count != 4 {
                continue;
            }
            let mut col = self.quads[qi].col;
            let mut row = self.quads[qi].row;
            for j in 0..4 {
                match j {
                    0 => {
                        row -= 1;
                        col -= 1;
                    }
                    1 => col += 2,
                    2 => row += 2,
                    _ => col -= 2,
                }
                if let Some(nb) = self.quads[qi].neighbors[j] {
                    if !self.quads[nb].ordered
                        && col <= col_max
                        && col >= col_min
                        && row <= row_max
                        && row >= row_min
                    {
                        found += 1;
                        let c = self.quads[qi].corners[j];
                        self.order_quad(nb, c, (j + 2) & 3);
                        let n = &mut self.quads[nb];
                        n.ordered = true;
                        n.row = row;
                        n.col = col;
                    }
                }
            }
        }

        if found > 0 {
            let mut i = 0;
            while i < quad_count && self.quads.len() < max_quad_buf_size {
                let qi = quads[i];
                if self.quads[qi].count < 4 && self.quads[qi].ordered {
                    let added = self.add_outer_quad(qi, quads);
                    quad_count += added;
                }
                i += 1;
            }
            if self.quads.len() >= max_quad_buf_size {
                return 0;
            }
        }

        if dcol == w && drow == h {
            for i in (0..quad_count).rev() {
                let qi = quads[i];
                if !self.quads[qi].ordered {
                    let outer = self.quads[qi]
                        .neighbors
                        .iter()
                        .any(|n| n.is_some_and(|n| self.quads[n].ordered));
                    if !outer {
                        self.remove_quad_from_group(quads, qi);
                    }
                }
            }
            return quads.len();
        }
        0
    }

    /// `ChessBoardDetector::addOuterQuad`.
    fn add_outer_quad(&mut self, quad: usize, quads: &mut Vec<usize>) -> usize {
        let mut added = 0;
        let max = self.max_quad_buf_size;
        let mut i = 0;
        while i < 4 && self.quads.len() < max {
            if self.quads[quad].neighbors[i].is_none() {
                let j = (i + 2) & 3;
                let q_index = self.quads.len();
                let mut q = Quad::new(0);
                added += 1;
                quads.push(q_index);
                {
                    let src = &mut self.quads[quad];
                    src.neighbors[i] = Some(q_index);
                    src.count += 1;
                }
                let src = self.quads[quad];
                q.neighbors[j] = Some(quad);
                q.group_idx = src.group_idx;
                q.count = 1;
                q.ordered = false;
                q.edge_sqr_len = src.edge_sqr_len;
                let off = sub(
                    self.corners[src.corners[i]].pt,
                    self.corners[src.corners[j]].pt,
                );
                for k in 0..4 {
                    let pt = self.corners[src.corners[k]].pt;
                    let mut c = Corner::new(pt);
                    c.pt = add(c.pt, off);
                    self.corners[q_index * 4 + k] = c;
                    q.corners[k] = q_index * 4 + k;
                }
                q.corners[j] = src.corners[i];
                match i {
                    0 => {
                        q.col = src.col - 1;
                        q.row = src.row - 1;
                    }
                    // OpenCV gives case 2 the row of case 1 (row - 1), kept as it is.
                    1 | 2 => {
                        q.col = src.col + 1;
                        q.row = src.row - 1;
                    }
                    _ => {
                        q.col = src.col - 1;
                        q.row = src.row + 1;
                    }
                }
                self.quads.push(q);
                for k in [1usize, 3] {
                    let next_i = (i + k) % 4;
                    let prev_i = (i + k + 2) % 4;
                    let Some(qp) = self.quads[quad].neighbors[prev_i] else {
                        continue;
                    };
                    let qpq = self.quads[qp];
                    if !qpq.ordered {
                        continue;
                    }
                    let Some(qn) = qpq.neighbors[i] else {
                        continue;
                    };
                    let qnq = self.quads[qn];
                    let qq = self.quads[q_index];
                    if qnq.ordered && (qnq.col - qq.col).abs() == 1 && (qnq.row - qq.row).abs() == 1
                    {
                        let nq = &mut self.quads[q_index];
                        nq.count = 2;
                        nq.neighbors[prev_i] = Some(qn);
                        let n = &mut self.quads[qn];
                        n.neighbors[next_i] = Some(q_index);
                        n.count += 1;
                        let c = self.quads[qn].corners[next_i];
                        self.quads[q_index].corners[prev_i] = c;
                    }
                }
            }
            i += 1;
        }
        added
    }

    /// `ChessBoardDetector::removeQuadFromGroup`.
    fn remove_quad_from_group(&mut self, quads: &mut Vec<usize>, q0: usize) {
        let count = quads.len();
        let mut self_idx: Option<usize> = None;
        for i in 0..count {
            let q = quads[i];
            if q == q0 {
                self_idx = Some(i);
            }
            for j in 0..4 {
                if self.quads[q].neighbors[j] == Some(q0) {
                    self.quads[q].neighbors[j] = None;
                    self.quads[q].count -= 1;
                    for k in 0..4 {
                        if self.quads[q0].neighbors[k] == Some(q) {
                            self.quads[q0].neighbors[k] = None;
                            self.quads[q0].count -= 1;
                            break;
                        }
                    }
                    break;
                }
            }
        }
        let Some(self_idx) = self_idx else {
            return; // OpenCV asserts that the quad is in the group
        };
        if self_idx != count - 1 {
            quads[self_idx] = quads[count - 1];
        }
        quads.truncate(count - 1);
    }

    /// `ChessBoardDetector::cleanFoundConnectedQuads`: while the group has more quads than
    /// the pattern, removes the quad whose removal shrinks the convex hull of the quad
    /// centres most.
    fn clean_found_connected_quads(&mut self, group: &mut Vec<usize>) -> usize {
        let count = ((i64::from(self.width) + 1) * (i64::from(self.height) + 1) + 1) / 2;
        let mut quad_count = group.len();
        if quad_count as i64 <= count {
            return quad_count;
        }
        let mut centers: Vec<P2> = Vec::with_capacity(quad_count);
        let mut center = (0f32, 0f32);
        for &q in group.iter() {
            let c = |k: usize| self.corners[self.quads[q].corners[k]].pt;
            let s = add(add(add(c(0), c(1)), c(2)), c(3));
            let ci = (s.0 * 0.25, s.1 * 0.25);
            centers.push(ci);
            center = add(center, ci);
        }
        #[allow(clippy::cast_precision_loss)]
        let inv = 1.0f32 / quad_count as f32;
        center = (center.0 * inv, center.1 * inv);
        while quad_count as i64 > count {
            let mut min_box_area = f64::MAX;
            let mut min_idx: usize = 0;
            for skip in 0..quad_count {
                let temp = centers[skip];
                centers[skip] = center;
                let hull = convex_hull_f32(&centers[..quad_count], true);
                centers[skip] = temp;
                let hull_area = contour_area_f32(&hull, false);
                if hull_area < min_box_area {
                    min_box_area = hull_area;
                    min_idx = skip;
                }
            }
            let q0 = group[min_idx];
            for i in 0..quad_count {
                let q = group[i];
                for j in 0..4 {
                    if self.quads[q].neighbors[j] == Some(q0) {
                        self.quads[q].neighbors[j] = None;
                        self.quads[q].count -= 1;
                        for k in 0..4 {
                            if self.quads[q0].neighbors[k] == Some(q) {
                                self.quads[q0].neighbors[k] = None;
                                self.quads[q0].count -= 1;
                                break;
                            }
                        }
                        break;
                    }
                }
            }
            quad_count -= 1;
            group[min_idx] = group[quad_count];
            centers[min_idx] = centers[quad_count];
        }
        group.truncate(quad_count);
        quad_count
    }

    /// `ChessBoardDetector::checkQuadGroup`: orders the inner corners of a quad group row by
    /// row. Returns the number of corners if the group is the whole pattern, else minus the
    /// number of corners kept in `out` (which then holds them unordered).
    #[allow(clippy::too_many_lines)]
    fn check_quad_group(&mut self, group: &[usize], out: &mut Vec<usize>) -> i64 {
        const ROW1: i32 = 1_000_000;
        const ROW2: i32 = 2_000_000;
        const ROW_: i32 = 3_000_000;
        let area = i64::from(self.area());
        let mut corners: Vec<usize> = Vec::with_capacity(group.len() * 4);
        let mut result: i64 = 0;

        'finalize: {
            for &qi in group {
                let q = self.quads[qi];
                for j in 0..4 {
                    if q.neighbors[j].is_none() {
                        continue;
                    }
                    let next_j = (j + 1) & 3;
                    let (a, b) = (q.corners[j], q.corners[next_j]);
                    let row_flag = match q.count {
                        1 => ROW1,
                        2 => ROW2,
                        _ => ROW_,
                    };
                    if self.corners[a].row == 0 {
                        corners.push(a);
                        self.corners[a].row = row_flag;
                    } else if self.corners[a].row > row_flag {
                        self.corners[a].row = row_flag;
                    }
                    if q.neighbors[next_j].is_some() {
                        if self.corners[a].count >= 4 || self.corners[b].count >= 4 {
                            break 'finalize;
                        }
                        for k in 0..4 {
                            if self.corners[a].neighbors[k] == Some(b)
                                || self.corners[b].neighbors[k] == Some(a)
                            {
                                break 'finalize;
                            }
                        }
                        let ca = self.corners[a].count as usize;
                        self.corners[a].neighbors[ca] = Some(b);
                        self.corners[a].count += 1;
                        let cb = self.corners[b].count as usize;
                        self.corners[b].neighbors[cb] = Some(a);
                        self.corners[b].count += 1;
                    }
                }
            }

            let corner_count = corners.len() as i64;
            if corner_count != area {
                break 'finalize;
            }

            let mut hist = [0i64; 5];
            let mut first: Option<usize> = None;
            let mut first2: Option<usize> = None;
            for &c in &corners {
                let n = self.corners[c].count as usize;
                hist[n] += 1;
                if first.is_none() && n == 2 {
                    if self.corners[c].row == ROW1 {
                        first = Some(c);
                    } else if first2.is_none() && self.corners[c].row == ROW2 {
                        first2 = Some(c);
                    }
                }
            }
            if first.is_none() {
                first = first2;
            }
            let Some(mut cur) = first else {
                break 'finalize;
            };
            if hist[0] != 0
                || hist[1] != 0
                || hist[2] != 4
                || hist[3] != (i64::from(self.width) + i64::from(self.height)) * 2 - 8
            {
                break 'finalize;
            }

            let mut right: Option<usize> = None;
            let mut below: Option<usize> = None;
            out.clear();
            out.push(cur);
            for k in 0..4 {
                if let Some(c) = self.corners[cur].neighbors[k] {
                    if right.is_none() {
                        right = Some(c);
                    } else if below.is_none() {
                        below = Some(c);
                    }
                }
            }
            let ok = |c: Option<usize>, s: &Self| {
                c.is_some_and(|c| s.corners[c].count == 2 || s.corners[c].count == 3)
            };
            if !ok(right, self) || !ok(below, self) {
                break 'finalize;
            }
            let (mut right, mut below) = (right.unwrap_or(0), below.unwrap_or(0));
            self.corners[cur].row = 0;
            let mut first = Some(below);
            let max_wh = self.width.max(self.height) as usize;
            loop {
                self.corners[right].row = 0;
                out.push(right);
                if self.corners[right].count == 2 {
                    break;
                }
                if self.corners[right].count != 3 || out.len() >= max_wh {
                    break 'finalize;
                }
                cur = right;
                for k in 0..4 {
                    if let Some(c) = self.corners[cur].neighbors[k] {
                        if self.corners[c].row > 0 {
                            if self.corners[c].neighbors.contains(&Some(below)) {
                                below = c;
                            } else {
                                right = c;
                            }
                        }
                    }
                }
            }

            let mut width = out.len() as i64;
            let mut height = if width == i64::from(self.width) {
                i64::from(self.height)
            } else if width == i64::from(self.height) {
                i64::from(self.width)
            } else {
                break 'finalize;
            };

            let mut i: i32 = 1;
            while let Some(f) = first {
                cur = f;
                first = None;
                let mut j: i64 = 0;
                loop {
                    self.corners[cur].row = i;
                    out.push(cur);
                    if i64::from(self.corners[cur].count)
                        == 2 + i64::from(i64::from(i) < height - 1)
                        && j > 0
                    {
                        break;
                    }
                    let mut r: Option<usize> = None;
                    for k in 0..4 {
                        if let Some(c) = self.corners[cur].neighbors[k] {
                            if self.corners[c].row > i {
                                let prev_row = self.corners[c]
                                    .neighbors
                                    .iter()
                                    .any(|n| n.is_some_and(|n| self.corners[n].row == i - 1));
                                if prev_row {
                                    r = Some(c);
                                    if j > 0 {
                                        break;
                                    }
                                } else if j == 0 {
                                    first = Some(c);
                                }
                            }
                        }
                    }
                    let Some(r) = r else {
                        break 'finalize;
                    };
                    cur = r;
                    j += 1;
                }
                if j != width - 1 {
                    break 'finalize;
                }
                i += 1;
            }

            if out.len() as i64 != corner_count {
                break 'finalize;
            }

            // check if we need to transpose the board
            if width != i64::from(self.width) {
                std::mem::swap(&mut width, &mut height);
                let tmp = out.clone();
                let (w, h) = (width as usize, height as usize);
                for i in 0..h {
                    for j in 0..w {
                        out[i * w + j] = tmp[j * h + i];
                    }
                }
            }

            // check if we need to revert the order in each row
            let w = width as usize;
            let h = height as usize;
            let pw = self.width as usize;
            let p0 = self.corners[out[0]].pt;
            let p1 = self.corners[out[pw - 1]].pt;
            let p2 = self.corners[out[pw]].pt;
            if (p1.0 - p0.0) * (p2.1 - p1.1) - (p1.1 - p0.1) * (p2.0 - p1.0) < 0.0 {
                if w.is_multiple_of(2) {
                    for i in 0..h {
                        for j in 0..w / 2 {
                            out.swap(i * w + j, i * w + w - j - 1);
                        }
                    }
                } else {
                    for j in 0..w {
                        for i in 0..h / 2 {
                            out.swap(i * w + j, (h - i - 1) * w + j);
                        }
                    }
                }
            }
            result = corner_count;
        }

        // finalize:
        if result <= 0 {
            let corner_count = (corners.len() as i64).min(area) as usize;
            out.clear();
            out.extend_from_slice(&corners[..corner_count]);
            result = -(corner_count as i64);
            if result == -area {
                result = -result;
            }
        }
        result
    }

    /// `ChessBoardCorner::sumDist`.
    fn sum_dist(&self, c: usize) -> (f32, i32) {
        let me = &self.corners[c];
        let mut sum = 0f32;
        let mut n = 0;
        for nb in me.neighbors.iter().flatten() {
            sum += norm_sqr(sub(self.corners[*nb].pt, me.pt)).sqrt();
            n += 1;
        }
        (sum, n)
    }

    /// `ChessBoardDetector::checkBoardMonotony`.
    pub(super) fn check_board_monotony(&self, corners: &[P2]) -> bool {
        let (pw, ph) = (self.width as usize, self.height as usize);
        for k in 0..2 {
            let max_i = if k == 0 { ph } else { pw };
            let max_j = (if k == 0 { pw } else { ph }) - 1;
            for i in 0..max_i {
                let a = if k == 0 { corners[i * pw] } else { corners[i] };
                let b = if k == 0 {
                    corners[(i + 1) * pw - 1]
                } else {
                    corners[(ph - 1) * pw + i]
                };
                let dx0 = b.0 - a.0;
                let dy0 = b.1 - a.1;
                if dx0.abs() + dy0.abs() < f32::EPSILON {
                    return false;
                }
                let mut prevt = 0f32;
                for j in 1..max_j {
                    let c = if k == 0 {
                        corners[i * pw + j]
                    } else {
                        corners[j * pw + i]
                    };
                    let t = ((c.0 - a.0) * dx0 + (c.1 - a.1) * dy0) / (dx0 * dx0 + dy0 * dy0);
                    if t < prevt || t > 1.0 {
                        return false;
                    }
                    prevt = t;
                }
            }
        }
        true
    }

    /// `ChessBoardDetector::processQuads`: links the quads, and returns true with the
    /// pattern's corners in `out_corners` when a group is the whole pattern. Otherwise
    /// `out_corners` keeps the largest partial set met, as OpenCV leaves it.
    pub(super) fn process_quads(
        &mut self,
        out_corners: &mut Vec<P2>,
        prev_sqr_size: &mut i32,
    ) -> bool {
        out_corners.clear();
        if self.quads.is_empty() {
            return false;
        }
        self.find_quad_neighbors();
        let area = self.area();
        let mut group: Vec<usize> = Vec::new();
        let mut corner_group: Vec<usize> = Vec::new();
        let mut group_idx = 0;
        loop {
            self.find_connected_quads(&mut group, group_idx);
            group_idx += 1;
            if group.is_empty() {
                break;
            }
            let count = self.order_found_connected_quads(&mut group);
            if count == 0 {
                continue;
            }
            self.clean_found_connected_quads(&mut group);
            let count = self.check_quad_group(&group, &mut corner_group);
            let n = if count > 0 { i64::from(area) } else { -count }.min(i64::from(area)) as usize;
            let mut sum_dist = 0f32;
            let mut total = 0i32;
            for &c in corner_group.iter().take(n) {
                let (s, ni) = self.sum_dist(c);
                sum_dist += s;
                total += ni;
            }
            #[allow(clippy::cast_precision_loss)]
            {
                *prev_sqr_size = (sum_dist / total.max(1) as f32).round_ties_even() as i32;
            }
            if count > 0 || (-count > out_corners.len() as i64) {
                out_corners.clear();
                out_corners.extend(corner_group.iter().take(n).map(|&c| self.corners[c].pt));
                if count == i64::from(area) && self.check_board_monotony(out_corners) {
                    return true;
                }
            }
        }
        false
    }
}

/// `arePointsOnSameSideFromLine`: the cross products (in `double`) of the line direction with
/// the two points have a positive product.
fn same_side(l1: P2, l2: P2, p1: P2, p2: P2) -> bool {
    let d = sub(l2, l1);
    let v1 = sub(p1, l1);
    let v2 = sub(p2, l1);
    let cross = |a: P2, b: P2| f64::from(a.0) * f64::from(b.1) - f64::from(a.1) * f64::from(b.0);
    cross(d, v1) * cross(d, v2) > 0.0
}
