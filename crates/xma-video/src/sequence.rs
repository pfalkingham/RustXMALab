//! A list of still images, one per frame. Ported from `core/ImageSequence.cpp`.
//!
//! As in the C++:
//! - the frame count is the number of file names; nothing else is checked when opening
//!   apart from the first image, which the C++ also decodes in its constructor;
//! - each frame is decoded on its own and keeps its own size and depth, so images of
//!   different sizes give frames of different sizes ([`SourceInfo`] gives the first
//!   image's);
//! - a file that is missing or cannot be decoded only affects its own frame: `read_frame`
//!   returns the error, so the viewer shows no image. The C++ holds an empty (0 by 0)
//!   image for that frame (measured, dataset `video_seq_odd`).
//!
//! Each image is decoded with [`crate::image_file`]; colour images are reduced to grey
//! with the weights of the [`VideoCompat`] given.

use std::path::{Path, PathBuf};

use crate::image_file::read_image_with;
use crate::{Frame, FrameSource, SourceInfo, SourceKind, VideoCompat, VideoError};

/// Reader for an image sequence.
pub struct ImageSequence {
    info: SourceInfo,
    paths: Vec<PathBuf>,
    compat: VideoCompat,
}

impl ImageSequence {
    /// Opens the sequence: decodes the first image for the size and depth.
    ///
    /// Ported from core/ImageSequence.cpp, ImageSequence::ImageSequence.
    pub fn open(paths: &[PathBuf]) -> Result<ImageSequence, VideoError> {
        Self::open_with(paths, &VideoCompat::FIXED)
    }

    /// [`ImageSequence::open`] with the compatibility settings `compat`.
    ///
    /// Ported from core/ImageSequence.cpp, ImageSequence::ImageSequence.
    pub fn open_with(paths: &[PathBuf], compat: &VideoCompat) -> Result<ImageSequence, VideoError> {
        let Some(first) = paths.first() else {
            return Err(VideoError::Unsupported {
                path: PathBuf::new(),
                reason: "an image sequence needs at least one file".to_string(),
            });
        };
        let frame = read_image_with(first, compat)?;
        let bit_depth = frame.bit_depth;
        Ok(ImageSequence {
            info: SourceInfo {
                kind: SourceKind::ImageSequence,
                frame_count: paths.len(),
                readable_frames: paths.len(),
                width: frame.width,
                height: frame.height,
                bit_depth,
                frame_rate: None,
            },
            paths: paths.to_vec(),
            compat: *compat,
        })
    }

    /// The file of frame `index`.
    pub fn path(&self, index: usize) -> Option<&Path> {
        self.paths.get(index).map(PathBuf::as_path)
    }
}

impl FrameSource for ImageSequence {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    /// Ported from core/ImageSequence.cpp, ImageSequence::setActiveFrame.
    fn read_frame(&mut self, index: usize) -> Result<Frame, VideoError> {
        let path = self.paths.get(index).ok_or(VideoError::FrameOutOfRange {
            index,
            readable: self.paths.len(),
        })?;
        read_image_with(path, &self.compat)
    }
}
