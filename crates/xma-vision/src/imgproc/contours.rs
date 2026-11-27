//! Contour finding: a port of OpenCV 4.13's `cv::findContours` for the two retrieval
//! modes XMALab and OpenCV's `SimpleBlobDetector` use (`RETR_EXTERNAL`, `RETR_LIST`) and
//! the two approximations (`CHAIN_APPROX_NONE`, `CHAIN_APPROX_SIMPLE`).
//!
//! OpenCV 4.13 runs the implementation in `modules/imgproc/src/contours_new.cpp` (the
//! legacy one in `contours.cpp` is only reachable as `findContours_legacy`). This module
//! follows the new one step for step, so that the order of the contours and the start
//! point and direction of each contour are OpenCV's:
//!
//! - every non-zero pixel is foreground (the image is padded with a one-pixel zero border
//!   and thresholded to 0/1 first);
//! - contours are found by a raster scan (Suzuki-Abe border following) and returned in
//!   the **reverse** order in which the scan found them: the last contour found comes
//!   first (`contourTreeToResults` walks the children of the root, and `Tree::addChild`
//!   puts each new child in front);
//! - an outer border starts at its top-most, then left-most pixel and runs
//!   counter-clockwise in image coordinates (y down) as listed; a hole (`RETR_LIST` only)
//!   starts at the pixel left of the first background pixel met in the scan;
//! - a contour of one isolated pixel has one point; with `CHAIN_APPROX_SIMPLE` a
//!   one-pixel-wide line comes back as its two ends, and with `CHAIN_APPROX_NONE` every
//!   pixel of a one-pixel-wide line appears twice (once each way) except the ends.
//!
//! For those modes the hierarchy is not computed. [`find_contours_ccomp`] adds
//! `RETR_CCOMP` (with its hierarchy), which `cv::findChessboardCorners` uses.

use crate::image::ImageU8;

/// A contour point in integer pixels (OpenCV `cv::Point`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct PointI {
    /// Column.
    pub x: i32,
    /// Row.
    pub y: i32,
}

impl PointI {
    /// A point at column `x`, row `y`.
    pub const fn new(x: i32, y: i32) -> Self {
        PointI { x, y }
    }
}

/// Which contours [`find_contours`] returns (OpenCV `RetrievalModes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetrievalMode {
    /// `RETR_EXTERNAL`: outer borders only; holes and everything inside them are skipped.
    External,
    /// `RETR_LIST`: every border, outer borders and holes, with no hierarchy.
    List,
}

/// How the points of a contour are stored (OpenCV `ContourApproximationModes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainApprox {
    /// `CHAIN_APPROX_NONE`: every border pixel.
    None,
    /// `CHAIN_APPROX_SIMPLE`: only the pixels where the chain direction changes.
    Simple,
}

/// Chain code directions: 0 is +x, counting counter-clockwise on screen (y down).
/// Ported from imgproc/src/contours_common.hpp, `chainCodeDeltas`.
const CHAIN_DELTAS: [(i32, i32); 8] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Ported from contours_new.cpp: `MASK8_RIGHT`, `MASK8_NEW`, `MASK8_FLAGS`.
const MASK8_RIGHT: i8 = 0x80u8 as i8;
const MASK8_NEW: i8 = 0x02;
const MASK8_FLAGS: i32 = 0xFEu8 as i8 as i32; // sign-extended to -2, as in the C++
/// Ported from contours_new.cpp: `MASK8_BLACK`, `MASK8_LVAL`.
const MASK8_BLACK: i8 = 0x01;
const MASK8_LVAL: i8 = 0x7F;
/// Ported from contours_common.hpp, `MAX_SIZE`.
const MAX_SIZE: i8 = 16;

/// The padded, thresholded working image (`schar` in OpenCV).
struct Work {
    data: Vec<i8>,
    /// Row length of the padded image.
    step: isize,
    width: i32,
    height: i32,
}

impl Work {
    /// `copyMakeBorder(1, 1, 1, 1, BORDER_CONSTANT 0)` and `threshold(0, 1, THRESH_BINARY)`
    /// of `image`: 1 for its non-zero pixels, inside a border of 0. `None` for an empty image.
    ///
    /// Ported from OpenCV 4.13 imgproc/src/contours_new.cpp, `cv::findContours` (the
    /// preparation of the image).
    fn padded(image: &ImageU8) -> Option<Work> {
        let (w, h) = (image.width(), image.height());
        if w == 0 || h == 0 {
            return None;
        }
        let pw = w + 2;
        let ph = h + 2;
        let mut data = vec![0i8; pw * ph];
        for y in 0..h {
            let src = image.row(y);
            let dst = &mut data[(y + 1) * pw + 1..(y + 1) * pw + 1 + w];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = i8::from(s != 0);
            }
        }
        Some(Work {
            data,
            step: pw as isize,
            width: pw as i32,
            height: ph as i32,
        })
    }

    fn at(&self, x: i32, y: i32) -> i8 {
        self.data[(y as isize * self.step + x as isize) as usize]
    }

    fn idx(&self, x: i32, y: i32) -> isize {
        y as isize * self.step + x as isize
    }

    /// Ported from contours_common.hpp, `getDelta`.
    fn delta(&self, s: i8) -> isize {
        let (dx, dy) = CHAIN_DELTAS[(s % 8) as usize];
        dx as isize + dy as isize * self.step
    }
}

/// Finds the contours of the non-zero pixels of `image`.
///
/// Returns the contours in OpenCV's order, each as its points in OpenCV's order, with
/// `offset` added to every point. An empty image, or one with no non-zero pixel, gives
/// no contours.
///
/// Ported from OpenCV 4.13 imgproc/src/contours_new.cpp, `cv::findContours` (with
/// `ContourScanner_::create`, `findNext`, `findNextX`, `contourScan`, `makeContour`,
/// `icvFetchContourEx<schar>`) and contours_common.cpp, `contourTreeToResults`, for the
/// modes `RETR_EXTERNAL`/`RETR_LIST` and methods `CHAIN_APPROX_NONE`/`CHAIN_APPROX_SIMPLE`.
pub fn find_contours(
    image: &ImageU8,
    mode: RetrievalMode,
    method: ChainApprox,
    offset: PointI,
) -> Vec<Vec<PointI>> {
    let Some(mut img) = Work::padded(image) else {
        return Vec::new();
    };
    // The scanner's offset includes (-1, -1) for the border.
    let offset = PointI::new(offset.x.wrapping_sub(1), offset.y.wrapping_sub(1));
    let is_direct = method == ChainApprox::None;

    let mut found: Vec<Vec<PointI>> = Vec::new();

    // ContourScanner_::create
    let mut pt_x = 1i32;
    let mut pt_y = 1i32;
    let mut lnbd = PointI::new(0, 1);

    // ContourScanner_::findNext, repeated until it finds nothing.
    'scan: loop {
        let mut x = pt_x;
        let mut y = pt_y;
        let width = img.width - 1;
        let height = img.height - 1;
        let mut last_pos = lnbd;
        let mut prev: i32 = img.at(x - 1, y) as i32;

        while y < height {
            let mut p: i32 = 0;
            while x < width {
                // findNextX (the SIMD branch finds the same x and p)
                while x < width {
                    p = img.at(x, y) as i32;
                    if p != prev {
                        break;
                    }
                    x += 1;
                }
                if x >= width {
                    break;
                }
                // contourScan
                if let Some(is_hole) = contour_scan(&img, mode, prev, p, &mut last_pos, x) {
                    last_pos.x = x - i32::from(is_hole);
                    let start = PointI::new(x - i32::from(is_hole), y);
                    let origin = PointI::new(start.x + offset.x, start.y + offset.y);
                    found.push(fetch_contour(&mut img, start, origin, is_hole, is_direct));
                    pt_x = x + 1;
                    pt_y = y;
                    lnbd = last_pos;
                    continue 'scan;
                }
                prev = p;
                if prev & MASK8_FLAGS != 0 {
                    last_pos.x = x;
                }
                x += 1;
            }
            last_pos = PointI::new(0, y + 1);
            x = 1;
            prev = 0;
            y += 1;
        }
        break;
    }

    // contourTreeToResults: the root's children, newest first.
    found.reverse();
    found
}

/// The test at the start of `ContourScanner_::contourScan` (8-bit, simple modes): returns
/// `Some(is_hole)` if a contour starts at `x`, updating `last_pos` as the C++ does.
fn contour_scan(
    img: &Work,
    mode: RetrievalMode,
    prev: i32,
    p: i32,
    last_pos: &mut PointI,
    x: i32,
) -> Option<bool> {
    let mut is_hole = false;
    if !(prev == 0 && p == 1) {
        if p != 0 || prev < 1 {
            return None;
        }
        if prev & MASK8_FLAGS != 0 {
            last_pos.x = x - 1;
        }
        is_hole = true;
    }
    if mode == RetrievalMode::External && (is_hole || img.at(last_pos.x, last_pos.y) > 0) {
        return None;
    }
    Some(is_hole)
}

/// Follows one border from `start` (padded coordinates), marking the pixels it visits,
/// and returns its points with `origin` as the first.
///
/// Ported from contours_new.cpp, `icvFetchContourEx<schar>` with `nbd = MASK8_NEW`.
fn fetch_contour(
    img: &mut Work,
    start: PointI,
    origin: PointI,
    is_hole: bool,
    is_direct: bool,
) -> Vec<PointI> {
    fetch_contour_ex(img, start, origin, is_hole, is_direct, MASK8_NEW).0
}

/// A bounding rectangle as `icvFetchContourEx` stores it: `x`, `y`, `width`, `height`.
#[derive(Clone, Copy, Debug, Default)]
struct BRect {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

/// Follows one border from `start` (padded coordinates), marking the pixels it visits with
/// `nbd`, and returns its points with `origin` as the first, and the bounding rectangle of
/// the points (in the coordinates of `origin`).
///
/// Ported from contours_new.cpp, `icvFetchContourEx<schar>`.
fn fetch_contour_ex(
    img: &mut Work,
    start: PointI,
    origin: PointI,
    is_hole: bool,
    is_direct: bool,
    nbd: i8,
) -> (Vec<PointI>, BRect) {
    let mut pts = Vec::new();
    let mut pt = origin;
    let mut rect = BRect {
        x: pt.x,
        y: pt.y,
        width: pt.x,
        height: pt.y,
    };
    let i0 = img.idx(start.x, start.y);
    let nonzero = |img: &Work, i: isize| img.data[i as usize] != 0;

    let mut s_end: i8 = if is_hole { 0 } else { 4 };
    let mut s = s_end;
    let mut i1;
    loop {
        s = (s - 1) & 7;
        i1 = i0 + img.delta(s);
        if nonzero(img, i1) || s == s_end {
            break;
        }
    }

    if s == s_end {
        // single pixel
        img.data[i0 as usize] = nbd | MASK8_RIGHT;
        pts.push(pt);
    } else {
        let mut i3 = i0;
        let mut prev_s = s ^ 4;
        loop {
            s_end = s;
            s = s.min(15); // clamp_direction
            let mut i4 = i3;
            while s < MAX_SIZE - 1 {
                s += 1;
                i4 = i3 + img.delta(s);
                if nonzero(img, i4) {
                    break;
                }
            }
            s &= 7;

            // "right" bound
            if ((s as i32 - 1) as u32) < (s_end as i32 as u32) {
                img.data[i3 as usize] = nbd | MASK8_RIGHT;
            } else if img.data[i3 as usize] == MASK8_BLACK {
                img.data[i3 as usize] = nbd;
            }

            if s != prev_s || is_direct {
                pts.push(pt);
            }
            if s != prev_s {
                // update bounds
                if pt.x < rect.x {
                    rect.x = pt.x;
                } else if pt.x > rect.width {
                    rect.width = pt.x;
                }
                if pt.y < rect.y {
                    rect.y = pt.y;
                } else if pt.y > rect.height {
                    rect.height = pt.y;
                }
            }
            prev_s = s;
            let (dx, dy) = CHAIN_DELTAS[s as usize];
            // OpenCV's int arithmetic; wrapping only for offsets near the int range.
            pt.x = pt.x.wrapping_add(dx);
            pt.y = pt.y.wrapping_add(dy);

            if i4 == i0 && i3 == i1 {
                break;
            }
            i3 = i4;
            s = (s + 4) & 7;
        }
    }
    rect.width = rect.width.wrapping_sub(rect.x.wrapping_sub(1));
    rect.height = rect.height.wrapping_sub(rect.y.wrapping_sub(1));
    (pts, rect)
}

/// Whether the border of the contour starting at `start` (a hole if `is_hole`) passes
/// through `end` as the last contour of that border met in a raster scan.
///
/// Ported from contours_new.cpp, `icvTraceContour<schar>`.
fn trace_contour(img: &Work, start: PointI, end: PointI, is_hole: bool) -> bool {
    let stop = img.idx(end.x, end.y);
    let i0 = img.idx(start.x, start.y);
    let nonzero = |i: isize| img.data[i as usize] != 0;
    let s_end: i8 = if is_hole { 0 } else { 4 };
    let mut s = s_end;
    let mut i1;
    loop {
        s = (s - 1) & 7;
        i1 = i0 + img.delta(s);
        if nonzero(i1) || s == s_end {
            break;
        }
    }
    let mut i3 = i0;
    if s == s_end {
        return i3 == stop;
    }
    loop {
        s = s.min(15);
        let mut i4 = i3;
        while s < MAX_SIZE - 1 {
            s += 1;
            i4 = i3 + img.delta(s);
            if nonzero(i4) {
                break;
            }
        }
        if i3 == stop {
            if img.data[i3 as usize] & MASK8_RIGHT == 0 {
                // it's the only contour
                return true;
            }
            // check if this is the last contour encountered during a raster scan
            let mut t = s;
            loop {
                t = (t - 1) & 7;
                let i5 = i3 + img.delta(t);
                if img.data[i5 as usize] != 0 {
                    break;
                }
                if t == 0 {
                    return true;
                }
            }
        }
        if i4 == i0 && i3 == i1 {
            break;
        }
        i3 = i4;
        s = (s + 4) & 7;
    }
    false
}

/// The hierarchy entry of one contour (OpenCV's `Vec4i`): the indices of the next and the
/// previous contour at the same level, of the first child and of the parent, -1 for none.
pub type Hierarchy = [i32; 4];

/// One node of the contour tree (`TreeNode<Contour>` of contours_common.hpp).
struct Node {
    parent: i32,
    first_child: i32,
    prev: i32,
    next: i32,
    ctable_next: i32,
    is_hole: bool,
    brect: BRect,
    origin: PointI,
    pts: Vec<PointI>,
}

impl Node {
    fn new(is_hole: bool) -> Self {
        Node {
            parent: -1,
            first_child: -1,
            prev: -1,
            next: -1,
            ctable_next: -1,
            is_hole,
            brect: BRect::default(),
            origin: PointI::default(),
            pts: Vec::new(),
        }
    }
}

/// `Tree::addChild`.
fn add_child(nodes: &mut [Node], parent: usize, child: usize) {
    let fc = nodes[parent].first_child;
    if fc != -1 {
        nodes[fc as usize].prev = child as i32;
        nodes[child].next = fc;
    }
    nodes[parent].first_child = child as i32;
    nodes[child].parent = parent as i32;
    nodes[child].prev = -1;
}

/// `cv::findContours(image, contours, hierarchy, RETR_CCOMP, method, offset)`: the outer
/// borders and the holes, each hole a child of the outer border around it (two levels; an
/// outer border inside a hole is again a top-level contour), in OpenCV's order, with
/// OpenCV's hierarchy (indices into the returned contours).
///
/// Ported from OpenCV 4.13 imgproc/src/contours_new.cpp, `cv::findContours` (with
/// `ContourScanner_::create`, `findNext`, `contourScan`, `findFirstBoundingContour`,
/// `makeContour`, `icvFetchContourEx<schar>`, `icvTraceContour<schar>`) and
/// contours_common.cpp, `contourTreeToResults`, for the mode `RETR_CCOMP` on an 8-bit image.
pub fn find_contours_ccomp(
    image: &ImageU8,
    method: ChainApprox,
    offset: PointI,
) -> (Vec<Vec<PointI>>, Vec<Hierarchy>) {
    let Some(mut img) = Work::padded(image) else {
        return (Vec::new(), Vec::new());
    };
    let offset = PointI::new(offset.x.wrapping_sub(1), offset.y.wrapping_sub(1));
    let is_direct = method == ChainApprox::None;

    let mut nodes: Vec<Node> = Vec::new();
    let mut root = Node::new(true);
    root.brect = BRect {
        x: 0,
        y: 0,
        width: img.width,
        height: img.height,
    };
    nodes.push(root);
    let mut ctable = [-1i32; 128];
    let mut nbd: i8 = 2;

    let mut pt_x = 1i32;
    let mut pt_y = 1i32;
    let mut lnbd = PointI::new(0, 1);

    'scan: loop {
        let mut x = pt_x;
        let mut y = pt_y;
        let width = img.width - 1;
        let height = img.height - 1;
        let mut last_pos = lnbd;
        let mut prev: i32 = img.at(x - 1, y) as i32;

        while y < height {
            let mut p: i32 = 0;
            while x < width {
                while x < width {
                    p = img.at(x, y) as i32;
                    if p != prev {
                        break;
                    }
                    x += 1;
                }
                if x >= width {
                    break;
                }
                // contourScan
                let mut is_hole = false;
                let mut starts = true;
                if !(prev == 0 && p == 1) {
                    if p != 0 || prev < 1 {
                        starts = false;
                    } else {
                        if prev & MASK8_FLAGS != 0 {
                            last_pos.x = x - 1;
                        }
                        is_hole = true;
                    }
                }
                if starts {
                    // find contour parent
                    let mut main_parent: i32;
                    if !is_hole || last_pos.x <= 0 {
                        main_parent = 0;
                    } else {
                        let lval = (img.at(last_pos.x, last_pos.y) & MASK8_LVAL) as usize;
                        main_parent =
                            find_first_bounding_contour(&img, &nodes, &ctable, last_pos, y, lval);
                        if main_parent < 0 {
                            // Not reached in OpenCV's use (it would index the tree with -1).
                            main_parent = 0;
                        }
                        let e = &nodes[main_parent as usize];
                        if e.is_hole == is_hole {
                            main_parent = if e.parent != -1 { e.parent } else { 0 };
                        }
                    }
                    last_pos.x = x - i32::from(is_hole);

                    // makeContour
                    let start = PointI::new(x - i32::from(is_hole), y);
                    let origin = PointI::new(
                        start.x.wrapping_add(offset.x),
                        start.y.wrapping_add(offset.y),
                    );
                    let lval = nbd;
                    nbd = nbd.wrapping_add(1) & MASK8_LVAL;
                    if nbd == 0 {
                        nbd = MASK8_BLACK | MASK8_NEW;
                    }
                    let (pts, mut brect) =
                        fetch_contour_ex(&mut img, start, origin, is_hole, is_direct, lval);
                    brect.x = brect.x.wrapping_sub(offset.x);
                    brect.y = brect.y.wrapping_sub(offset.y);
                    let mut node = Node::new(is_hole);
                    node.brect = brect;
                    node.origin = start;
                    node.pts = pts;
                    node.ctable_next = ctable[lval as usize];
                    let id = nodes.len();
                    ctable[lval as usize] = id as i32;
                    nodes.push(node);
                    add_child(&mut nodes, main_parent as usize, id);

                    pt_x = x + 1;
                    pt_y = y;
                    lnbd = last_pos;
                    continue 'scan;
                }
                prev = p;
                if prev & MASK8_FLAGS != 0 {
                    last_pos.x = x;
                }
                x += 1;
            }
            last_pos = PointI::new(0, y + 1);
            x = 1;
            prev = 0;
            y += 1;
        }
        break;
    }

    // contourTreeToResults: depth-first, each node's children from its first child on.
    if nodes[0].first_child == -1 {
        return (Vec::new(), Vec::new());
    }
    let mut order = Vec::with_capacity(nodes.len() - 1);
    let mut stack = vec![0usize];
    while let Some(idx) = stack.pop() {
        if idx != 0 {
            order.push(idx);
        }
        // push the children from the last sibling back to the first
        let mut cur = nodes[idx].first_child;
        if cur != -1 {
            while nodes[cur as usize].next != -1 {
                cur = nodes[cur as usize].next;
            }
            while cur != -1 {
                stack.push(cur as usize);
                cur = nodes[cur as usize].prev;
            }
        }
    }
    let mut mapping = vec![-1i32; nodes.len() + 1];
    for (i, &n) in order.iter().enumerate() {
        mapping[n + 1] = i as i32;
    }
    let map = |v: i32| mapping[(v + 1) as usize];
    let hierarchy = order
        .iter()
        .map(|&n| {
            let e = &nodes[n];
            [map(e.next), map(e.prev), map(e.first_child), map(e.parent)]
        })
        .collect();
    let contours = order
        .iter()
        .map(|&n| std::mem::take(&mut nodes[n].pts))
        .collect();
    (contours, hierarchy)
}

/// Ported from contours_new.cpp, `ContourScanner_::findFirstBoundingContour` (called with
/// `par = -1`).
fn find_first_bounding_contour(
    img: &Work,
    nodes: &[Node],
    ctable: &[i32; 128],
    last_pos: PointI,
    y: i32,
    lval: usize,
) -> i32 {
    let end_point = PointI::new(last_pos.x, y);
    let mut res: i32 = -1;
    let mut cur = ctable[lval];
    while cur != -1 {
        let e = &nodes[cur as usize];
        if last_pos.x.wrapping_sub(e.brect.x) < e.brect.width
            && last_pos.y.wrapping_sub(e.brect.y) < e.brect.height
        {
            if res != -1 {
                let r = &nodes[res as usize];
                if trace_contour(img, r.origin, end_point, r.is_hole) {
                    break;
                }
            }
            res = cur;
        }
        cur = e.ctable_next;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: usize, h: usize, px: &[(usize, usize)]) -> ImageU8 {
        let mut im = ImageU8::new(w, h);
        for &(x, y) in px {
            im.set(x, y, 255);
        }
        im
    }

    #[test]
    fn single_pixel_and_square() {
        let im = img(5, 5, &[(2, 2)]);
        let c = find_contours(
            &im,
            RetrievalMode::List,
            ChainApprox::None,
            PointI::default(),
        );
        assert_eq!(c, vec![vec![PointI::new(2, 2)]]);
        let im = img(6, 6, &[(1, 1), (2, 1), (1, 2), (2, 2)]);
        let c = find_contours(
            &im,
            RetrievalMode::External,
            ChainApprox::Simple,
            PointI::new(10, 20),
        );
        assert_eq!(
            c,
            vec![vec![
                PointI::new(11, 21),
                PointI::new(11, 22),
                PointI::new(12, 22),
                PointI::new(12, 21)
            ]]
        );
    }

    #[test]
    fn empty() {
        let im = ImageU8::new(4, 3);
        assert!(
            find_contours(
                &im,
                RetrievalMode::List,
                ChainApprox::None,
                PointI::default()
            )
            .is_empty()
        );
        let im = ImageU8::new(0, 0);
        assert!(
            find_contours(
                &im,
                RetrievalMode::List,
                ChainApprox::None,
                PointI::default()
            )
            .is_empty()
        );
    }
}
