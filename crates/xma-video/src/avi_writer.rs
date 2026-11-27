//! A Motion JPEG AVI writer, and the JPEG encoding of single images.
//!
//! XMALab's File > Export > Undistorted Trial images in "avi" opens `cv::VideoWriter` with
//! the codec `-1` (`core/Trial.cpp`, `saveTrialImages`), which OpenCV 4.13's FFmpeg back
//! end refuses on Linux (`cap_ffmpeg_impl.hpp`: it lists the codecs and does not open), so
//! XMALab 3 writes no video there (parity ledger V27). This writer is what the program
//! writes instead: an AVI 1.0 file (`RIFF AVI `, `hdrl` with `avih`, one `vids` stream
//! `MJPG` with a `BITMAPINFOHEADER`, `movi` with one `00dc` chunk per frame, `idx1`), each
//! frame a baseline JPEG of R, G, B samples, as OpenCV's own MJPEG writer
//! (`cap_mjpeg_encoder.cpp`) lays the file out. The bytes are not OpenCV's.
//!
//! Frames are written to `<name>.part` beside the target as they come and the file is
//! renamed when [`MjpegAviWriter::finish`] has written the index and the sizes, so the
//! target is either absent or complete.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use image::ExtendedColorType;
use image::codecs::jpeg::JpegEncoder;

use crate::VideoError;

/// The JPEG quality `cv::imwrite` uses by default (`IMWRITE_JPEG_QUALITY` 95), also used
/// for the video's frames.
pub const JPEG_QUALITY: u8 = 95;

/// The samples of an 8-bit image to encode.
#[derive(Debug, Clone, Copy)]
pub enum Samples<'a> {
    /// Grey, one byte per pixel.
    Gray(&'a [u8]),
    /// R, G, B interleaved.
    Rgb(&'a [u8]),
}

/// Encodes an 8-bit image as a baseline JPEG of quality [`JPEG_QUALITY`] (the `image`
/// crate's encoder: the bytes are not libjpeg's, which `cv::imwrite` uses).
///
/// # Errors
/// A sample count that does not match the size, or a failed encoding.
pub fn encode_jpeg(width: u32, height: u32, samples: Samples<'_>) -> Result<Vec<u8>, VideoError> {
    let n = width as usize * height as usize;
    let (data, colour, len) = match samples {
        Samples::Gray(d) => (d, ExtendedColorType::L8, n),
        Samples::Rgb(d) => (d, ExtendedColorType::Rgb8, n * 3),
    };
    let fail = |reason: String| VideoError::Decode {
        path: PathBuf::from("(JPEG)"),
        frame: None,
        reason,
    };
    if data.len() != len {
        return Err(fail("sample count does not match the size".to_owned()));
    }
    if width == 0 || height == 0 || width > 65_535 || height > 65_535 {
        return Err(fail(format!("a JPEG cannot be {width} x {height}")));
    }
    let mut out = Vec::new();
    JpegEncoder::new_with_quality(&mut out, JPEG_QUALITY)
        .encode(data, width, height, colour)
        .map_err(|e| fail(e.to_string()))?;
    Ok(out)
}

/// An AVI file being written, one JPEG frame at a time.
#[derive(Debug)]
pub struct MjpegAviWriter {
    target: PathBuf,
    part: PathBuf,
    out: BufWriter<File>,
    width: u32,
    height: u32,
    /// Offset and size of each frame chunk's data, from the `movi` list's type field.
    index: Vec<(u32, u32)>,
    /// Bytes of the `movi` list after its size field.
    movi_len: u64,
    largest: u32,
}

/// Offsets of the fields patched at the end.
const RIFF_SIZE_AT: u64 = 4;
const AVIH_TOTAL_FRAMES_AT: u64 = 48;
const AVIH_BUFFER_AT: u64 = 60;
const STRH_LENGTH_AT: u64 = 140;
const STRH_BUFFER_AT: u64 = 144;
const MOVI_SIZE_AT: u64 = 216;

fn u32le(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

impl MjpegAviWriter {
    /// Starts `path` (through `<path>.part`) for frames of `width × height` at `fps` frames
    /// per second (30 when `fps` is not above 0, as XMALab passes it).
    ///
    /// # Errors
    /// The file cannot be created, or the size is not one a JPEG can have.
    pub fn create(path: &Path, width: u32, height: u32, fps: f64) -> Result<Self, VideoError> {
        if width == 0 || height == 0 || width > 65_535 || height > 65_535 {
            return Err(VideoError::Unsupported {
                path: path.to_path_buf(),
                reason: format!("a video frame cannot be {width} x {height}"),
            });
        }
        let fps = if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            30.0
        };
        // dwRate / dwScale: the rate in thousandths.
        let scale = 1000u32;
        let rate = (fps * f64::from(scale))
            .round()
            .clamp(1.0, f64::from(u32::MAX)) as u32;
        let micro = (1e6 / fps).round().clamp(0.0, f64::from(u32::MAX)) as u32;
        let name = path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        let part = path.with_file_name(format!(".{name}.part"));
        let io = |e: std::io::Error| VideoError::Io {
            path: path.to_path_buf(),
            source: e,
        };
        let mut out = BufWriter::new(File::create(&part).map_err(io)?);
        let mut h = Vec::with_capacity(224);
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&u32le(0)); // patched
        h.extend_from_slice(b"AVI ");
        // LIST hdrl: 4 + (8 + 56) + (8 + 4 + (8 + 56) + (8 + 40)) = 192.
        h.extend_from_slice(b"LIST");
        h.extend_from_slice(&u32le(192));
        h.extend_from_slice(b"hdrl");
        h.extend_from_slice(b"avih");
        h.extend_from_slice(&u32le(56));
        h.extend_from_slice(&u32le(micro)); // dwMicroSecPerFrame
        h.extend_from_slice(&u32le(0)); // dwMaxBytesPerSec
        h.extend_from_slice(&u32le(0)); // dwPaddingGranularity
        h.extend_from_slice(&u32le(0x10)); // dwFlags: AVIF_HASINDEX
        h.extend_from_slice(&u32le(0)); // dwTotalFrames (patched, offset 48)
        h.extend_from_slice(&u32le(0)); // dwInitialFrames
        h.extend_from_slice(&u32le(1)); // dwStreams
        h.extend_from_slice(&u32le(0)); // dwSuggestedBufferSize (patched, offset 60)
        h.extend_from_slice(&u32le(width));
        h.extend_from_slice(&u32le(height));
        h.extend_from_slice(&[0; 16]); // dwReserved
        h.extend_from_slice(b"LIST");
        h.extend_from_slice(&u32le(116));
        h.extend_from_slice(b"strl");
        h.extend_from_slice(b"strh");
        h.extend_from_slice(&u32le(56));
        h.extend_from_slice(b"vids");
        h.extend_from_slice(b"MJPG");
        h.extend_from_slice(&u32le(0)); // dwFlags
        h.extend_from_slice(&[0; 4]); // wPriority, wLanguage
        h.extend_from_slice(&u32le(0)); // dwInitialFrames
        h.extend_from_slice(&u32le(scale));
        h.extend_from_slice(&u32le(rate));
        h.extend_from_slice(&u32le(0)); // dwStart
        h.extend_from_slice(&u32le(0)); // dwLength (patched, offset 140)
        h.extend_from_slice(&u32le(0)); // dwSuggestedBufferSize (patched, offset 144)
        h.extend_from_slice(&u32le(u32::MAX)); // dwQuality: default
        h.extend_from_slice(&u32le(0)); // dwSampleSize
        h.extend_from_slice(&[0; 4]); // rcFrame left, top
        h.extend_from_slice(&(width as u16).to_le_bytes());
        h.extend_from_slice(&(height as u16).to_le_bytes());
        h.extend_from_slice(b"strf");
        h.extend_from_slice(&u32le(40));
        h.extend_from_slice(&u32le(40)); // biSize
        h.extend_from_slice(&u32le(width));
        h.extend_from_slice(&u32le(height));
        h.extend_from_slice(&1u16.to_le_bytes()); // biPlanes
        h.extend_from_slice(&24u16.to_le_bytes()); // biBitCount
        h.extend_from_slice(b"MJPG");
        h.extend_from_slice(&u32le(width.saturating_mul(height).saturating_mul(3)));
        h.extend_from_slice(&[0; 16]); // resolution, colours
        h.extend_from_slice(b"LIST");
        h.extend_from_slice(&u32le(0)); // patched (offset 216)
        h.extend_from_slice(b"movi");
        debug_assert_eq!(h.len(), 224);
        out.write_all(&h).map_err(io)?;
        Ok(MjpegAviWriter {
            target: path.to_path_buf(),
            part,
            out,
            width,
            height,
            index: Vec::new(),
            movi_len: 4,
            largest: 0,
        })
    }

    /// The frame size.
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Frames written so far.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.index.len()
    }

    /// Appends one frame (its size must be the video's).
    ///
    /// # Errors
    /// Another size, a failed encoding, a file larger than an AVI 1.0 file can be, or a
    /// write error.
    pub fn push(
        &mut self,
        width: u32,
        height: u32,
        samples: Samples<'_>,
    ) -> Result<(), VideoError> {
        if (width, height) != (self.width, self.height) {
            return Err(VideoError::Unsupported {
                path: self.target.clone(),
                reason: format!(
                    "frame of {width} x {height} in a video of {} x {}",
                    self.width, self.height
                ),
            });
        }
        let jpeg = encode_jpeg(width, height, samples)?;
        let len = u32::try_from(jpeg.len()).unwrap_or(u32::MAX);
        let padded = u64::from(len) + u64::from(len % 2);
        if 224 + self.movi_len + 8 + padded + 16 * (self.index.len() as u64 + 1) + 8
            > u64::from(u32::MAX)
        {
            return Err(VideoError::Unsupported {
                path: self.target.clone(),
                reason: "the video would exceed the 4 GB an AVI file can hold".to_owned(),
            });
        }
        let io = |e: std::io::Error| VideoError::Io {
            path: self.target.clone(),
            source: e,
        };
        let offset = self.movi_len as u32;
        self.out.write_all(b"00dc").map_err(io)?;
        self.out.write_all(&u32le(len)).map_err(io)?;
        self.out.write_all(&jpeg).map_err(io)?;
        if len % 2 == 1 {
            self.out.write_all(&[0]).map_err(io)?;
        }
        self.movi_len += 8 + padded;
        self.index.push((offset, len));
        self.largest = self.largest.max(len);
        Ok(())
    }

    /// Writes the index and the sizes and renames the file to its name; returns it.
    ///
    /// # Errors
    /// A write or rename error (the partial file is removed).
    pub fn finish(mut self) -> Result<PathBuf, VideoError> {
        let r = self.finish_inner();
        if r.is_err() {
            let _ = std::fs::remove_file(&self.part);
        }
        r.map(|()| self.target.clone())
    }

    fn finish_inner(&mut self) -> Result<(), VideoError> {
        let target = self.target.clone();
        let io = |e: std::io::Error| VideoError::Io {
            path: target.clone(),
            source: e,
        };
        let w = &mut self.out;
        w.write_all(b"idx1").map_err(io)?;
        w.write_all(&u32le(16 * self.index.len() as u32))
            .map_err(io)?;
        for &(offset, len) in &self.index {
            w.write_all(b"00dc").map_err(io)?;
            w.write_all(&u32le(0x10)).map_err(io)?; // AVIIF_KEYFRAME
            w.write_all(&u32le(offset)).map_err(io)?;
            w.write_all(&u32le(len)).map_err(io)?;
        }
        let total = 224 + self.movi_len - 4 + 8 + 16 * self.index.len() as u64;
        let n = self.index.len() as u32;
        let buffer = self.largest.saturating_add(8);
        for (at, v) in [
            (RIFF_SIZE_AT, (total - 8) as u32),
            (AVIH_TOTAL_FRAMES_AT, n),
            (AVIH_BUFFER_AT, buffer),
            (STRH_LENGTH_AT, n),
            (STRH_BUFFER_AT, buffer),
            (MOVI_SIZE_AT, self.movi_len as u32),
        ] {
            w.seek(SeekFrom::Start(at)).map_err(io)?;
            w.write_all(&u32le(v)).map_err(io)?;
        }
        w.flush().map_err(io)?;
        w.get_ref().sync_all().map_err(io)?;
        std::fs::rename(&self.part, &self.target).map_err(io)
    }
}

impl Drop for MjpegAviWriter {
    fn drop(&mut self) {
        // Not finished (cancelled, failed): no partial file is left behind.
        if self.part.exists() {
            let _ = std::fs::remove_file(&self.part);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameSource;

    #[test]
    fn written_video_reads_back() {
        let dir = std::env::temp_dir().join(format!("xma_avi_writer_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("v.avi");
        let (w, h) = (48u32, 30u32);
        let mut wr = MjpegAviWriter::create(&path, w, h, 250.0).unwrap();
        for k in 0..3u8 {
            let grey: Vec<u8> = (0..w * h)
                .map(|i| ((i % w) as u8).wrapping_mul(5) / 2 + 20 * k)
                .collect();
            let rgb: Vec<u8> = grey.iter().flat_map(|&v| [v, v, v]).collect();
            wr.push(w, h, Samples::Rgb(&rgb)).unwrap();
        }
        assert!(wr.push(w + 1, h, Samples::Gray(&[0; 49 * 30])).is_err());
        assert_eq!(wr.frames(), 3);
        let out = wr.finish().unwrap();
        assert_eq!(out, path);
        let mut src = crate::avi::AviSource::open(&path).unwrap();
        assert!(src.is_mjpeg());
        assert_eq!(src.info().frame_count, 3);
        assert_eq!((src.info().width, src.info().height), (w, h));
        let f = src.read_frame(2).unwrap();
        let g = f.to_gray8();
        let want = |i: u32| f64::from(((i % w) as u8).wrapping_mul(5) / 2 + 40);
        let worst = (0..w * h)
            .map(|i| (f64::from(g[i as usize]) - want(i)).abs())
            .fold(0.0, f64::max);
        assert!(worst <= 4.0, "JPEG error {worst}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bad_sizes_are_refused() {
        let dir = std::env::temp_dir();
        assert!(MjpegAviWriter::create(&dir.join("never.avi"), 0, 5, 30.0).is_err());
        assert!(encode_jpeg(2, 2, Samples::Gray(&[0; 3])).is_err());
        assert!(encode_jpeg(0, 0, Samples::Gray(&[])).is_err());
    }
}
