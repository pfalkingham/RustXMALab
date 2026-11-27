//! Choosing the reader for a file or a camera's file list.
//!
//! - [`open_video`] picks the reader from the file's content, not its name
//!   (parity ledger V8): a Phantom `.cine` signature goes to [`CineSource`], a
//!   RIFF AVI to [`AviSource`] and, when that reader does not handle the file,
//!   to ffmpeg; anything else to ffmpeg ([`FfmpegSource`]).
//! - With [`VideoCompat::mjpeg_with_ffmpeg`] (XMALab 3 mode) an MJPEG AVI goes to
//!   ffmpeg when ffmpeg is available, which reproduces the C++ exactly; by
//!   default it is decoded natively.
//! - [`open_source`] opens what a trial's camera holds: one video file, or a
//!   list of still images, as `core/Trial.cpp` decides.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::avi::AviSource;
use crate::cine::CineSource;
use crate::ffmpeg::{FfmpegOptions, FfmpegSource, FfmpegTools};
use crate::sequence::ImageSequence;
use crate::{FrameSource, VideoCompat, VideoError};

/// Settings for opening sources.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenOptions {
    /// The `ffmpeg` executable. When `None` it is looked for through the
    /// environment variable `XMALAB_FFMPEG`, next to the running program, then
    /// on `PATH` ([`FfmpegTools::locate`]).
    pub ffmpeg: Option<PathBuf>,
    /// The `ffprobe` executable, found the same way (`XMALAB_FFPROBE`).
    pub ffprobe: Option<PathBuf>,
    /// Settings of the ffmpeg reader. Its `grey_weights` are replaced by
    /// those of `compat`.
    pub ffmpeg_options: FfmpegOptions,
    /// Corrected or XMALab 3 behaviour of every reader.
    pub compat: VideoCompat,
}

impl OpenOptions {
    /// Finds the ffmpeg executables these options name or imply.
    pub fn ffmpeg_tools(&self) -> Result<FfmpegTools, VideoError> {
        FfmpegTools::locate(self.ffmpeg.as_deref(), self.ffprobe.as_deref())
    }
}

/// What the first bytes of a file say it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSignature {
    /// Phantom `.cine`: starts with `CI`.
    Cine,
    /// `RIFF....AVI `.
    RiffAvi,
    /// A still image: PNG, JPEG, TIFF (or BigTIFF) or BMP.
    StillImage,
    /// Anything else (MP4, MOV, MKV, an unknown file).
    Other,
}

/// Reads the signature of `path` from its first bytes.
pub fn file_signature(path: &Path) -> Result<FileSignature, VideoError> {
    let mut head = [0u8; 16];
    let mut file = File::open(path).map_err(|source| VideoError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut n = 0;
    while n < head.len() {
        match file.read(&mut head[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(source) => {
                return Err(VideoError::Io {
                    path: path.to_path_buf(),
                    source,
                });
            }
        }
    }
    Ok(signature_of(&head[..n]))
}

/// The signature of a file that starts with `head`.
pub fn signature_of(head: &[u8]) -> FileSignature {
    let starts = |s: &[u8]| head.starts_with(s);
    if head.len() >= 12 && starts(b"RIFF") && &head[8..12] == b"AVI " {
        FileSignature::RiffAvi
    } else if starts(b"CI") {
        FileSignature::Cine
    } else if starts(b"\x89PNG\r\n\x1a\n")
        || starts(&[0xFF, 0xD8, 0xFF])
        || starts(b"II*\0")
        || starts(b"MM\0*")
        || starts(b"II+\0")
        || starts(b"MM\0+")
        || starts(b"BM")
    {
        FileSignature::StillImage
    } else {
        FileSignature::Other
    }
}

/// Opens a video file with the reader its content calls for (parity ledger
/// V8): `.cine` to [`CineSource`]; RIFF AVI to [`AviSource`], then to ffmpeg
/// if that reader returns [`VideoError::Unsupported`]; everything else to
/// ffmpeg. The ffmpeg executables are looked for only when ffmpeg is needed.
///
/// The C++ program chooses its reader class from the extension
/// (`core/Trial.cpp`, `Trial::Trial`) and OpenCV's FFmpeg then identifies the
/// content, so a `.avi` name holding MP4 still opens there.
pub fn open_video(path: &Path, options: &OpenOptions) -> Result<Box<dyn FrameSource>, VideoError> {
    match file_signature(path)? {
        FileSignature::Cine => Ok(Box::new(CineSource::open_with(path, &options.compat)?)),
        FileSignature::RiffAvi => match AviSource::open_with(path, &options.compat) {
            Ok(s) if s.is_mjpeg() && options.compat.mjpeg_with_ffmpeg => {
                match options.ffmpeg_tools() {
                    Ok(tools) => Ok(Box::new(FfmpegSource::open(
                        path,
                        &tools,
                        &ffmpeg_options(options),
                    )?)),
                    Err(_) => Ok(Box::new(s)),
                }
            }
            Ok(s) => Ok(Box::new(s)),
            Err(VideoError::Unsupported { .. }) => open_ffmpeg(path, options),
            Err(e) => Err(e),
        },
        FileSignature::StillImage | FileSignature::Other => open_ffmpeg(path, options),
    }
}

fn open_ffmpeg(path: &Path, options: &OpenOptions) -> Result<Box<dyn FrameSource>, VideoError> {
    let tools = options.ffmpeg_tools()?;
    Ok(Box::new(FfmpegSource::open(
        path,
        &tools,
        &ffmpeg_options(options),
    )?))
}

fn ffmpeg_options(options: &OpenOptions) -> FfmpegOptions {
    FfmpegOptions {
        grey_weights: options.compat.grey_weights,
        ..options.ffmpeg_options.clone()
    }
}

/// Opens what one camera of a trial holds.
///
/// As `core/Trial.cpp`, `Trial::Trial`: more than one file is an image
/// sequence. One file is a video, except that a still image is a sequence of
/// one frame. The C++ decides that single case by extension (`cine`, `avi`,
/// `mp4` and `mov` are videos, anything else a sequence); here it is decided
/// by content, so a video with another extension (`.mkv`) also opens.
pub fn open_source(
    filenames: &[PathBuf],
    options: &OpenOptions,
) -> Result<Box<dyn FrameSource>, VideoError> {
    match filenames {
        [] => Err(VideoError::Unsupported {
            path: PathBuf::new(),
            reason: "no files given".to_string(),
        }),
        [one] => {
            if file_signature(one)? == FileSignature::StillImage {
                Ok(Box::new(ImageSequence::open_with(
                    filenames,
                    &options.compat,
                )?))
            } else {
                open_video(one, options)
            }
        }
        _ => Ok(Box::new(ImageSequence::open_with(
            filenames,
            &options.compat,
        )?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signatures() {
        assert_eq!(
            signature_of(b"RIFF\0\0\0\0AVI LIST"),
            FileSignature::RiffAvi
        );
        assert_eq!(signature_of(b"RIFF\0\0\0\0WAVEfmt "), FileSignature::Other);
        assert_eq!(signature_of(b"CI\x2c\0\0\0"), FileSignature::Cine);
        assert_eq!(
            signature_of(b"\x89PNG\r\n\x1a\n\0"),
            FileSignature::StillImage
        );
        assert_eq!(signature_of(b"II*\0\x08\0"), FileSignature::StillImage);
        assert_eq!(signature_of(b"\0\0\0\x20ftypisom"), FileSignature::Other);
        assert_eq!(signature_of(b""), FileSignature::Other);
    }
}
