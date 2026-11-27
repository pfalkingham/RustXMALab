//! Frame sources (`.cine`, AVI, image sequences, an ffmpeg executable) behind
//! one [`FrameSource`] interface, still-image decoding, and the frame cache.
//!
//! Dependency rule: independent of `xma-model`; depends on no other workspace
//! crate (`xma-oracle` in tests only).
//!
//! Conventions: frames are 0-based; pixel (0, 0) is the top-left
//! pixel and rows are stored top to bottom.

use std::borrow::Cow;
use std::fmt;
use std::path::PathBuf;

pub mod avi;
pub mod avi_writer;
mod bytes;
pub mod cache;
pub mod cine;
pub mod convert;
pub mod ffmpeg;
pub mod image_file;
pub mod open;
pub mod sequence;

/// Name of this crate, used to check the workspace builds and links.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// Pixel values of one frame: a single channel, row by row from the top.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pixels {
    /// 8 bits per pixel.
    Gray8(Vec<u8>),
    /// More than 8 bits per pixel, held as stored in the file (not rescaled);
    /// [`Frame::bit_depth`] says how many bits carry information.
    Gray16(Vec<u16>),
}

/// One decoded frame or still image.
///
/// Colour sources are reduced to one channel when decoded (parity ledger V2;
/// the weights are chosen by [`VideoCompat`]), because XMALab works on grey
/// images. A colour still can also be decoded in colour for display
/// ([`image_file::decode_colour_image`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Bits per pixel that carry information: 8 for [`Pixels::Gray8`]; for
    /// [`Pixels::Gray16`] the depth of the data (for 16-bit stills 16, or the depth the
    /// file declares by default; `RealBPP` for 16-bit `.cine`; 10 or 12 for packed `.cine`
    /// by default), so values are below `2^bit_depth`. A value above that (a
    /// `.cine` can hold some) saturates in [`Frame::to_gray8`], as in the C++.
    pub bit_depth: u8,
    /// The image at the depth the file holds. The display uses this.
    pub pixels: Pixels,
}

impl Frame {
    /// An 8-bit frame.
    pub fn gray8(width: u32, height: u32, data: Vec<u8>) -> Frame {
        Frame {
            width,
            height,
            bit_depth: 8,
            pixels: Pixels::Gray8(data),
        }
    }

    /// A frame of 16-bit samples of which `bit_depth` bits (9 to 16) carry information.
    pub fn gray16(width: u32, height: u32, data: Vec<u16>, bit_depth: u8) -> Frame {
        debug_assert!((9..=16).contains(&bit_depth), "bit depth {bit_depth}");
        Frame {
            width,
            height,
            bit_depth,
            pixels: Pixels::Gray16(data),
        }
    }

    /// The 8-bit image every algorithm works on (parity ledger V1): the pixels
    /// themselves for an 8-bit frame, otherwise `convertTo(CV_8U, 2^-(bit_depth - 8))`
    /// as the C++ applies it, rounded half to even and saturated: 1/256 for 16-bit
    /// stills (`core/Image.cpp`), `1 / 2^(RealBPP - 8)` for `.cine` (`core/CineVideo.cpp`).
    pub fn to_gray8(&self) -> Cow<'_, [u8]> {
        match &self.pixels {
            Pixels::Gray8(data) => Cow::Borrowed(data),
            Pixels::Gray16(data) => Cow::Owned(convert::scale16_to_gray8(
                data,
                u32::from(self.bit_depth.saturating_sub(8)),
            )),
        }
    }

    /// Mirrors the frame left to right. The C++ readers apply a camera's
    /// "flip" this way while loading (`cv::flip(image, image, 1)` in
    /// `core/Image.cpp`, `core/CineVideo.cpp`, `core/AviVideo.cpp`); here the
    /// caller applies it.
    pub fn flip_horizontal(&mut self) {
        let w = self.width as usize;
        if w == 0 {
            return;
        }
        match &mut self.pixels {
            Pixels::Gray8(d) => d.chunks_exact_mut(w).for_each(<[u8]>::reverse),
            Pixels::Gray16(d) => d.chunks_exact_mut(w).for_each(<[u16]>::reverse),
        }
    }

    /// Number of pixels, `width * height`.
    pub fn len(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// True for an image with no pixels.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// How colour is reduced to grey (parity ledger V2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GreyWeights {
    /// Red weight 0.299 on red, blue weight 0.114 on blue (OpenCV's 15-bit
    /// `RGB2Gray` coefficients, applied to the right channels).
    #[default]
    Correct,
    /// As the C++: `COLOR_RGB2GRAY` on B, G, R data, so red gets 0.114 and blue
    /// 0.299.
    Swapped,
}

/// Where the readers keep XMALab 3's behaviour and where they correct it
/// (parity ledger: known C++ defects are fixed). [`VideoCompat::FIXED`] (the default)
/// corrects; [`VideoCompat::XMALAB3`] reproduces the C++ program, for checks
/// against its dumps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoCompat {
    /// Colour to grey (ledger V2): stills, uncompressed colour AVI, MJPEG and
    /// the ffmpeg reader.
    pub grey_weights: GreyWeights,
    /// MJPEG in AVI: decode with ffmpeg when it is available (identical to the
    /// C++, which decodes through OpenCV's FFmpeg) instead of natively (the
    /// same pixels on every machine, no ffmpeg needed, within a grey level or so
    /// of the C++ for grey content).
    pub mjpeg_with_ffmpeg: bool,
    /// `.cine` 16-bit with `RealBPP` outside 8 to 16: as the C++ (`RealBPP - 8`
    /// wraps when below 8 and every frame is black; above 16 the scale is
    /// `2^-(RealBPP - 8)`), instead of taking the data as 16-bit.
    pub cine_real_bpp_as_cpp: bool,
    /// 16-bit grey stills (ledger V18): as the C++, every one is 16 bits deep, whatever the
    /// file declares (OpenCV reads neither PNG `sBIT` nor TIFF `MaxSampleValue`), instead of
    /// the depth the file declares ([`image_file`]).
    pub still_depth_16: bool,
    /// Packed 10-bit `.cine` (ledger V15): as the C++, 8-bit values truncated from the
    /// samples (`>> 2`, or the lookup table `>> 4`), instead of the samples at their own
    /// depth (10 bits, or 12 through the lookup table).
    pub cine_packed_as_cpp: bool,
    /// Natively read AVI (ledger V37): as the C++, the frame count is the header's
    /// `strh.dwLength` (OpenCV's `CAP_PROP_FRAME_COUNT`), 0 when a recording cut short left
    /// it 0, and no frame at or above it is read; instead of every frame really in the file
    /// readable, found through its index or a scan of `movi`, and the count at least that.
    pub avi_count_from_header: bool,
}

impl VideoCompat {
    /// The corrected behaviour (the default).
    pub const FIXED: VideoCompat = VideoCompat {
        grey_weights: GreyWeights::Correct,
        mjpeg_with_ffmpeg: false,
        cine_real_bpp_as_cpp: false,
        still_depth_16: false,
        cine_packed_as_cpp: false,
        avi_count_from_header: false,
    };
    /// XMALab 3's behaviour.
    pub const XMALAB3: VideoCompat = VideoCompat {
        grey_weights: GreyWeights::Swapped,
        mjpeg_with_ffmpeg: true,
        cine_real_bpp_as_cpp: true,
        still_depth_16: true,
        cine_packed_as_cpp: true,
        avi_count_from_header: true,
    };
}

impl Default for VideoCompat {
    fn default() -> Self {
        VideoCompat::FIXED
    }
}

/// What kind of reader serves a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceKind {
    /// Phantom `.cine`, read natively.
    Cine,
    /// AVI with uncompressed or MJPEG frames, read natively.
    Avi,
    /// A list of still images, one per frame.
    ImageSequence,
    /// Anything else, decoded by an ffmpeg executable.
    Ffmpeg,
}

/// Fixed properties of an open source.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceInfo {
    pub kind: SourceKind,
    /// Number of frames as the C++ program reports it. For a video decoded
    /// through ffmpeg this is the container's count, which can exceed
    /// `readable_frames` (parity ledger V7).
    pub frame_count: usize,
    /// Number of frames that can actually be read; `read_frame` succeeds for
    /// indices below this.
    pub readable_frames: usize,
    pub width: u32,
    pub height: u32,
    /// [`Frame::bit_depth`] of the frames `read_frame` returns (8 to 16; the
    /// first image's for an image sequence). A video decoded through ffmpeg always
    /// reports 8, whatever the file's depth, because the C++ program receives
    /// 8-bit frames from OpenCV there.
    pub bit_depth: u8,
    /// Frames per second when the file states it.
    pub frame_rate: Option<f64>,
}

/// A video or image sequence that can return any frame by index.
pub trait FrameSource: Send {
    fn info(&self) -> &SourceInfo;

    /// Decodes frame `index` (0-based). Frames may be requested in any order.
    ///
    /// A frame that cannot be read gives an error (the viewer then shows no image);
    /// each reader's module says what the C++ shows at such a frame.
    fn read_frame(&mut self, index: usize) -> Result<Frame, VideoError>;
}

/// Why a source could not be opened or a frame could not be read.
#[derive(Debug)]
pub enum VideoError {
    /// The file could not be opened or read.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The file is of a kind, or uses a feature, that no reader here handles.
    Unsupported { path: PathBuf, reason: String },
    /// The file's structure is damaged or inconsistent.
    Corrupt { path: PathBuf, reason: String },
    /// A frame index at or beyond the number of readable frames.
    FrameOutOfRange { index: usize, readable: usize },
    /// A frame's data could not be decoded.
    Decode {
        path: PathBuf,
        frame: Option<usize>,
        reason: String,
    },
    /// The ffmpeg or ffprobe executable is missing, could not be started, or
    /// failed.
    Ffmpeg(String),
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoError::Io { path, source } => write!(f, "{}: {source}", path.display()),
            VideoError::Unsupported { path, reason } => {
                write!(f, "{}: not supported: {reason}", path.display())
            }
            VideoError::Corrupt { path, reason } => {
                write!(f, "{}: damaged file: {reason}", path.display())
            }
            VideoError::FrameOutOfRange { index, readable } => write!(
                f,
                "frame {index} requested, but only {readable} frame(s) can be read"
            ),
            VideoError::Decode {
                path,
                frame: Some(frame),
                reason,
            } => write!(f, "{}: frame {frame}: {reason}", path.display()),
            VideoError::Decode {
                path,
                frame: None,
                reason,
            } => write!(f, "{}: {reason}", path.display()),
            VideoError::Ffmpeg(reason) => write!(f, "ffmpeg: {reason}"),
        }
    }
}

impl std::error::Error for VideoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VideoError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_mirrors_each_row() {
        let mut f = Frame::gray8(3, 2, vec![1, 2, 3, 4, 5, 6]);
        f.flip_horizontal();
        assert_eq!(f.pixels, Pixels::Gray8(vec![3, 2, 1, 6, 5, 4]));
        let mut g = Frame::gray16(2, 1, vec![100, 4095], 12);
        g.flip_horizontal();
        assert_eq!(g.pixels, Pixels::Gray16(vec![4095, 100]));
    }

    #[test]
    fn to_gray8_uses_the_bit_depth() {
        // 12-bit: 2^-4, ties to even, values above 4095 saturate.
        let f = Frame::gray16(4, 1, vec![8, 24, 4095, 5000], 12);
        assert_eq!(f.to_gray8().into_owned(), vec![0, 2, 255, 255]);
        let f = Frame::gray16(2, 1, vec![128, 384], 16);
        assert_eq!(f.to_gray8().into_owned(), vec![0, 2]);
    }

    #[test]
    fn crate_name() {
        assert_eq!(super::CRATE_NAME, "xma-video");
    }
}
