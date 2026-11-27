//! Ports of the OpenCV image routines that XMALab's marker detection, 2D tracking and bead
//! detection call. Each is a plain function on [`crate::image::Image`], reproduces OpenCV's
//! result to its rounding on the element types XMALab uses, and names the OpenCV routine and
//! source file it was ported from. Each is tested against the primitive reference
//! (`oracle/reference/primitives/`, written by OpenCV itself).
//!
//! One file per group of routines, one work package per file:
//!
//! | Module | Routines |
//! |---|---|
//! | [`arith`] | `convertTo`, subtraction, `normalize` (min-max), `minMaxLoc`, `threshold` |
//! | [`blur`] | `GaussianBlur`, `medianBlur` |
//! | [`contours`] | `findContours` |
//! | [`shape`] | `minEnclosingCircle`, `moments`, `contourArea`, `arcLength`, `convexHull` |
//! | [`subpix`] | `getRectSubPix` (8-bit and float) |
//! | [`template`] | `matchTemplate` (8-bit and float) |
//! | [`blob`] | `SimpleBlobDetector` |
//! | [`corners`] | `goodFeaturesToTrack`, `cornerSubPix` |
//! | [`adaptive`] | `boxFilter` (normalised, 8-bit), `adaptiveThreshold`, `equalizeHist`, `mean`, the detector's border |
//! | [`morph`] | `dilate`, `erode` |
//! | [`approx`] | `approxPolyDP`, `isContourConvex`, `boundingRect` |
//! | [`rotcalipers`] | `minAreaRect` |

pub mod adaptive;
pub mod approx;
pub mod arith;
pub mod blob;
pub mod blur;
pub mod contours;
pub mod corners;
pub mod morph;
pub mod rotcalipers;
pub mod shape;
pub mod subpix;
pub mod template;
