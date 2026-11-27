//! The images the tracker and the detector read in the window: a
//! [`xma_engine::tracking::TrialImages`] on the frame service.
//!
//! Camera `c` at frame `f` is frame `f` of the files the views read for that camera
//! (`Session::video_files`, so a folder or file chosen after loading counts), decoded through
//! the frame service's shared cache at the depth the source holds (parity ledger V1),
//! mirrored left to right for a flipped camera, and waited for: what `VideoStream::getImage`
//! holds in the C++ after the frame is loaded. The readers are the frame service's, which
//! are those `xma_engine::tracking::TrialSources` uses for a document in the default mode, so
//! a run in the window reads exactly the pixels `xma track` reads (`tests/tracking_ui.rs`).
//!
//! The visual filter is not applied: it is a display filter only (ledger U45), as in the
//! C++, where `getSubImage` reads the unfiltered image.
//!
//! Ported from `core/VideoStream.cpp`, `VideoStream::getImage`, with `core/Image.cpp`
//! (the flip on load).

use std::collections::VecDeque;
use std::sync::Arc;

use xma_engine::tracking::TrialImages;
use xma_video::Frame;

use crate::editing::TrialFiles;
use crate::frames::Frames;

/// Mirrored frames kept, so that the start and end frames of a step are mirrored once.
const KEPT: usize = 8;

/// The images of one trial, read through the frame service.
pub struct FrameImages<'a> {
    frames: &'a mut Frames,
    files: &'a TrialFiles,
    /// Frames already mirrored: (camera, frame).
    kept: VecDeque<((usize, usize), Arc<Frame>)>,
}

impl<'a> FrameImages<'a> {
    #[must_use]
    pub fn new(frames: &'a mut Frames, files: &'a TrialFiles) -> Self {
        Self {
            frames,
            files,
            kept: VecDeque::new(),
        }
    }
}

impl TrialImages for FrameImages<'_> {
    fn image(&mut self, camera: usize, frame: usize) -> Result<Arc<Frame>, String> {
        if let Some((_, f)) = self.kept.iter().find(|(k, _)| *k == (camera, frame)) {
            return Ok(f.clone());
        }
        let files = self
            .files
            .cameras
            .get(camera)
            .ok_or_else(|| format!("no camera {}", camera + 1))?;
        if files.is_empty() {
            return Err(format!("camera {} has no video in this trial", camera + 1));
        }
        let f = self.frames.read_now(files, frame).map_err(|e| {
            format!(
                "camera {}: frame {} cannot be read: {e}",
                camera + 1,
                frame + 1
            )
        })?;
        let flip = self.files.flips.get(camera).copied().unwrap_or(false);
        if !flip {
            return Ok(f);
        }
        let mut owned = (*f).clone();
        owned.flip_horizontal();
        let owned = Arc::new(owned);
        if self.kept.len() >= KEPT {
            self.kept.pop_front();
        }
        self.kept.push_back(((camera, frame), owned.clone()));
        Ok(owned)
    }
}
