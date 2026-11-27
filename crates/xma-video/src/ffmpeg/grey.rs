//! Reduction of what ffmpeg writes to the pipe to the 8-bit grey image the
//! C++ program holds.
//!
//! The C++ program gets frames from OpenCV's `VideoCapture` as 8-bit BGR, and
//! `core/Image.cpp`, `Image::setImage(cv::Mat&, bool)`, reduces them with
//! `cvtColor(..., COLOR_RGB2GRAY)` although the data is BGR (parity ledger V2).
//! Here the weights are chosen by [`FfmpegOptions::grey_weights`](super::FfmpegOptions):
//! correct by default, the C++'s swapped ones in XMALab 3 mode.
//!
//! OpenCV 4.13 converts with libswscale, `SWS_BICUBIC`, through
//! `sws_scale_frame` when built against libswscale 6.4.100 or later
//! (`cap_ffmpeg_impl.hpp`, `CvCapture_FFMPEG::retrieveFrame`), which takes the
//! colour range and matrix from the decoded frame, as the ffmpeg executable's
//! `scale` filter does by default. That is what is reproduced here; an OpenCV
//! built against an older FFmpeg ignores the frame's colour tags and can differ
//! on tagged colour video (proposed ledger row V9).

use std::ffi::OsString;

use super::probe::FrameIndex;
use crate::GreyWeights;
use crate::convert::gray_of;

/// How frames are reduced to 8-bit grey.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GreyConversion {
    /// ffmpeg converts to BGR as OpenCV does, then the C++ program's grey
    /// conversion runs here. Byte-identical to the C++ program for every pixel
    /// format; three times the bytes through the pipe.
    #[default]
    Exact,
    /// For 8-bit YUV video only: ffmpeg writes the luma plane and a table
    /// expands it as libswscale does. Identical to [`GreyConversion::Exact`]
    /// when the chroma is neutral (grey video), and faster. Other pixel formats
    /// use [`GreyConversion::Exact`].
    Luma,
}

/// What is done in Rust with each frame read from the pipe.
#[derive(Clone)]
pub(crate) enum Post {
    /// ffmpeg wrote 8-bit luma; map it through a table.
    Lut(Box<[u8; 256]>),
    /// ffmpeg wrote BGR; reduce as `Image.cpp` does, with the given weights.
    Bgr(GreyWeights),
}

impl Post {
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            Post::Lut(_) => 1,
            Post::Bgr(_) => 3,
        }
    }

    /// Converts one frame from the pipe into `out` (one byte per pixel).
    pub fn apply(&self, input: &[u8], out: &mut Vec<u8>) {
        match self {
            Post::Lut(t) => {
                out.clear();
                out.extend(input.iter().map(|&v| t[v as usize]));
            }
            Post::Bgr(weights) => {
                out.clear();
                out.extend(
                    input
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|&[b, g, r]| gray_of(r, g, b, *weights)),
                );
            }
        }
    }
}

/// Luma expansion of libswscale's default YUV to RGB path: limited-range luma
/// is expanded with `clip((76309 * (Y - 16) + 512) >> 16)`; full range is the
/// identity.
///
/// Ported from spike 4, `spikes/ffmpeg-scrub/src/main.rs`, `make_ff` (`ylut`).
pub fn luma_table(full_range: bool) -> [u8; 256] {
    let mut t = [0u8; 256];
    for (y, v) in t.iter_mut().enumerate() {
        *v = if full_range {
            y as u8
        } else {
            ((76309 * (y as i64 - 16) + 512) >> 16).clamp(0, 255) as u8
        };
    }
    t
}

/// 8-bit YUV pixel formats whose luma plane the [`GreyConversion::Luma`] path
/// can use.
fn is_8bit_yuv(pix_fmt: &str) -> bool {
    matches!(
        pix_fmt,
        "yuv420p"
            | "yuv422p"
            | "yuv444p"
            | "yuv440p"
            | "yuv411p"
            | "yuv410p"
            | "yuvj420p"
            | "yuvj422p"
            | "yuvj444p"
            | "yuvj440p"
            | "yuvj411p"
            | "nv12"
            | "nv21"
    )
}

/// Full-range luma: a `yuvj` pixel format or a stream tagged `pc`.
fn full_range(index: &FrameIndex) -> bool {
    index.pix_fmt.starts_with("yuvj") || index.color_range == "pc"
}

/// The output arguments of the ffmpeg command and the matching work in Rust.
pub(crate) fn plan(
    index: &FrameIndex,
    mode: GreyConversion,
    weights: GreyWeights,
) -> (Vec<OsString>, Post) {
    let s = |v: &[&str]| v.iter().map(OsString::from).collect::<Vec<_>>();
    if mode == GreyConversion::Luma && is_8bit_yuv(&index.pix_fmt) {
        let full = full_range(index);
        return (
            s(&["-vf", "extractplanes=y", "-pix_fmt", "gray"]),
            Post::Lut(Box::new(luma_table(full))),
        );
    }
    // As OpenCV: bicubic, colour range and matrix from the frame.
    (
        s(&[
            "-vf",
            "scale=flags=bicubic,format=bgr24",
            "-pix_fmt",
            "bgr24",
        ]),
        Post::Bgr(weights),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgr_weights() {
        let px = [255u8, 0, 0, 0, 0, 255, 0, 255, 0, 9, 9, 9]; // blue, red, green, grey
        let mut out = Vec::new();
        Post::Bgr(GreyWeights::Swapped).apply(&px, &mut out);
        assert_eq!(out, vec![76, 29, 150, 9]);
        Post::Bgr(GreyWeights::Correct).apply(&px, &mut out);
        assert_eq!(out, vec![29, 76, 150, 9]);
    }

    #[test]
    fn luma_table_limits() {
        let t = luma_table(false);
        assert_eq!((t[0], t[16], t[235], t[255]), (0, 0, 255, 255));
        assert_eq!(luma_table(true)[100], 100);
    }
}
