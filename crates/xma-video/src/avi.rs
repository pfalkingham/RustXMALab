//! AVI (RIFF) files with uncompressed or MJPEG frames, read natively.
//!
//! The C++ program opens every `.avi` with `cv::VideoCapture(name, CAP_FFMPEG)`
//! (`core/AviVideo.cpp`), so what it holds is FFmpeg's decode, converted to B, G, R by
//! swscale and then to grey by `core/Image.cpp` (ledger V2). This reader reproduces that
//! for the cases it handles, following FFmpeg's AVI demuxer where the file leaves a choice
//! (colour is reduced to grey with the weights [`VideoCompat::grey_weights`] chooses:
//! correct by default, the C++'s swapped ones in XMALab 3 mode):
//!
//! - frame count (ledger V37): by default `strh.dwLength` or the number of frames really in
//!   the file (the entries of the index below whose data lies wholly within the file),
//!   whichever is larger, so a header left at 0 by a recording cut short, or written before
//!   the last frames, gives the frames found; a header that declares more keeps its count and
//!   the frames past those in the file cannot be read (V7). In XMALab 3 mode `strh.dwLength` of the first video
//!   stream, which is FFmpeg's `nb_frames` and OpenCV's `CAP_PROP_FRAME_COUNT` (0 when the
//!   field is 0: measured, OpenCV estimates nothing from the duration there), and no frame at
//!   or above it is read;
//! - index: the OpenDML index (`indx` and its `ix##` chunks) when present and holding at
//!   least one entry, otherwise `idx1`, then the frames that lie in `movi` after the last
//!   indexed one (a recording cut short in a list no index covers yet, V37); with neither
//!   index usable, a scan of the `movi` lists (several `RIFF AVIX` lists are followed; nested
//!   lists are stepped into without recursion, as FFmpeg steps over `LIST` headers);
//! - uncompressed (`biCompression` 0): 8-bit with a palette (taken from the end of the
//!   `strf` data, as FFmpeg does; entries beyond it are black), 24-bit B, G, R and 32-bit
//!   B, G, R, X; rows bottom-up when `biHeight` is positive, top-down when negative;
//!   `Y800`, `Y8  ` and `GREY`: 8-bit grey, top-down. Rows are padded to 4 bytes when the
//!   frame chunk is large enough for padded rows, otherwise packed;
//! - MJPEG (`MJPG`, `mjpg`, `jpeg`, `JPEG`, `IJPG`): each frame is a JPEG image; frames
//!   without Huffman tables get the standard tables of ITU T.81 annex K.3, as FFmpeg's
//!   decoder assumes. Decoded with the `image` crate, whose output differs from
//!   FFmpeg's: by at most 1 grey level for grey content, more for colour (the tests give
//!   the distributions). In XMALab 3 mode `open::open_video` gives MJPEG to ffmpeg when
//!   it is available, which reproduces the C++ exactly.
//!
//! A frame that cannot be read (beyond the frames the index holds when `strh.dwLength`
//! declares more, ledger V7; in XMALab 3 mode also beyond `strh.dwLength`, V37; an empty
//! chunk): `read_frame` returns an
//! error, so the
//! viewer shows no image. The C++ shows frame 0 there (measured, dataset `video_avi_raw`):
//! `AviVideo::setActiveFrame` reopens the file after the failed read, which reads frame 0,
//! and the second read fails too.
//!
//! Anything else (other codecs, 1, 2, 4 and 16-bit uncompressed, 8-bit without a palette,
//! the `DIB ` tag, interlaced MJPEG with two fields per chunk) is `Unsupported`, so the
//! caller falls back to the FFmpeg reader, which is what the C++ uses for every AVI.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::bytes::{i32_at, read_at, row_starts, u16_at, u32_at};
use crate::convert::gray_of;
use crate::image_file::decode_jpeg_frame;
use crate::{Frame, FrameSource, Pixels, SourceInfo, SourceKind, VideoCompat, VideoError};

/// How the frames are coded.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Codec {
    /// Uncompressed 8-bit indices into a palette; the grey value of each entry.
    Palette8 { gray: Box<[u8; 256]> },
    /// Uncompressed grey (`Y800`).
    Gray8,
    /// Uncompressed B, G, R (24) or B, G, R, X (32).
    Bgr { bytes_per_pixel: usize },
    /// One JPEG image per frame.
    Mjpeg,
}

/// Where one frame's data is in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
    offset: u64,
    len: u32,
}

/// Reader for AVI with uncompressed or MJPEG frames.
pub struct AviSource {
    info: SourceInfo,
    path: PathBuf,
    file: File,
    codec: Codec,
    bottom_up: bool,
    index: Vec<Entry>,
    compat: VideoCompat,
}

/// One stream's header chunks: number, `strh`, `strf` and `indx` (when present).
type StreamHeaders = (usize, Vec<u8>, Vec<u8>, Option<Vec<u8>>);

/// The parts of the file's structure the reader needs.
#[derive(Debug, Default)]
struct Layout {
    /// `(stream number, strh, strf, indx)` of each stream.
    streams: Vec<StreamHeaders>,
    /// Position of each `movi` fourcc and the end of its list.
    movi: Vec<(u64, u64)>,
    /// Position and size of `idx1`'s data.
    idx1: Option<(u64, u32)>,
}

impl AviSource {
    /// Opens the source. Returns `VideoError::Unsupported` when the file is
    /// not of this kind or uses a feature this reader does not handle, so the
    /// caller can try another reader.
    ///
    /// Ported from core/AviVideo.cpp, AviVideo::AviVideo and AviVideo::reloadFile (frame
    /// count and frame rate as OpenCV reports them through FFmpeg).
    pub fn open(path: &Path) -> Result<AviSource, VideoError> {
        Self::open_with(path, &VideoCompat::FIXED)
    }

    /// [`AviSource::open`] with the compatibility settings `compat` (the grey weights of
    /// colour frames, ledger V2).
    ///
    /// Ported from core/AviVideo.cpp, AviVideo::AviVideo.
    pub fn open_with(path: &Path, compat: &VideoCompat) -> Result<AviSource, VideoError> {
        let io = |e: std::io::Error| VideoError::Io {
            path: path.to_path_buf(),
            source: e,
        };
        let unsupported = |reason: &str| VideoError::Unsupported {
            path: path.to_path_buf(),
            reason: reason.to_string(),
        };
        let mut file = File::open(path).map_err(io)?;
        let file_len = file.metadata().map_err(io)?.len();
        let mut sig = [0u8; 12];
        if file_len < 12 {
            return Err(unsupported("not a RIFF AVI file"));
        }
        file.read_exact(&mut sig).map_err(io)?;
        if &sig[0..4] != b"RIFF" || &sig[8..12] != b"AVI " {
            return Err(unsupported("not a RIFF AVI file"));
        }
        let layout = read_layout(&mut file, file_len).map_err(io)?;

        let Some((stream_no, strh, strf, indx)) = layout
            .streams
            .iter()
            .find(|(_, strh, _, _)| strh.len() >= 36 && &strh[0..4] == b"vids")
        else {
            return Err(unsupported("no video stream"));
        };
        if strf.len() < 40 {
            return Err(unsupported("video stream format too short"));
        }
        let scale = u32_at(strh, 20);
        let rate = u32_at(strh, 24);
        let length = u32_at(strh, 32);
        let width = i32_at(strf, 4);
        let height = i32_at(strf, 8);
        let bit_count = u16_at(strf, 14);
        let compression = &strf[16..20];
        if width <= 0 || height == 0 {
            return Err(unsupported("image size not handled"));
        }
        let extra = &strf[40.min(strf.len())..];

        let (codec, bottom_up) = match compression {
            [0, 0, 0, 0] => {
                let codec = match bit_count {
                    8 => {
                        if extra.is_empty() {
                            return Err(unsupported("8-bit frames without a palette"));
                        }
                        // FFmpeg (libavformat/avidec.c) takes the palette from the end of
                        // the extradata, up to 256 entries; the rest stay black.
                        let pal_size = extra.len().min(1024);
                        let pal = &extra[extra.len() - pal_size..];
                        let mut gray = Box::new([0u8; 256]);
                        for (g, q) in gray.iter_mut().zip(pal.as_chunks::<4>().0) {
                            *g = gray_of(q[2], q[1], q[0], compat.grey_weights);
                        }
                        Codec::Palette8 { gray }
                    }
                    24 => Codec::Bgr { bytes_per_pixel: 3 },
                    32 => Codec::Bgr { bytes_per_pixel: 4 },
                    _ => return Err(unsupported("uncompressed bit depth not handled")),
                };
                (codec, height > 0)
            }
            b"Y800" | b"Y8  " | b"GREY" if bit_count == 8 => (Codec::Gray8, false),
            b"MJPG" | b"mjpg" | b"jpeg" | b"JPEG" | b"IJPG" => (Codec::Mjpeg, false),
            _ => return Err(unsupported("codec not read natively")),
        };

        let index =
            build_index(&mut file, file_len, &layout, *stream_no, indx.as_deref()).map_err(io)?;
        let in_file = index
            .iter()
            .take_while(|e| e.offset.saturating_add(u64::from(e.len)) <= file_len)
            .count();
        // Ledger V37: by default every frame really in the file is readable, and the count is
        // never below them (a header left at 0 by a recording cut short, or written before the
        // last frames); a header that declares more keeps its count, as V7 decided, with the
        // frames past those in the file unreadable. In XMALab 3 mode the header's count, which
        // is OpenCV's `CAP_PROP_FRAME_COUNT` there (measured with `oracle/py/avi_cut_short.py`),
        // and no frame at or above it.
        let n = length as usize;
        let (frame_count, readable_frames) = if compat.avi_count_from_header {
            (n, in_file.min(n))
        } else {
            (n.max(in_file), in_file)
        };
        let info = SourceInfo {
            kind: SourceKind::Avi,
            frame_count,
            readable_frames,
            width: width as u32,
            height: height.unsigned_abs(),
            bit_depth: 8,
            frame_rate: (scale > 0 && rate > 0).then(|| f64::from(rate) / f64::from(scale)),
        };
        let mut index = index;
        index.truncate(info.readable_frames);
        let mut source = AviSource {
            info,
            path: path.to_path_buf(),
            file,
            codec,
            bottom_up,
            compat: *compat,
            index,
        };
        // A first frame that decodes to another size than the header's (interlaced capture:
        // two JPEG fields of half height per chunk) is left to FFmpeg, which combines the
        // fields. A first frame that does not decode opens as before (its read fails).
        if source.codec == Codec::Mjpeg
            && !source.index.is_empty()
            && let Ok(f) = source.decode_mjpeg(0)
            && (f.width, f.height) != (source.info.width, source.info.height)
        {
            return Err(unsupported(&format!(
                "MJPEG frames of {} x {} in a stream of {} x {} (interlaced fields?)",
                f.width, f.height, source.info.width, source.info.height
            )));
        }
        Ok(source)
    }

    /// True when the frames are MJPEG.
    pub fn is_mjpeg(&self) -> bool {
        self.codec == Codec::Mjpeg
    }

    /// The bytes of frame `index`'s chunk.
    fn chunk(&mut self, index: usize) -> Result<Vec<u8>, VideoError> {
        let Some(entry) = self.index.get(index).copied() else {
            return Err(VideoError::FrameOutOfRange {
                index,
                readable: self.index.len(),
            });
        };
        if entry.len == 0 {
            return Err(self.decode_err(index, "empty frame chunk"));
        }
        read_at(&mut self.file, entry.offset, entry.len as usize).map_err(|e| VideoError::Io {
            path: self.path.clone(),
            source: e,
        })
    }

    /// Frame `index` of an MJPEG stream as its JPEG decodes, whatever its size.
    fn decode_mjpeg(&mut self, index: usize) -> Result<Frame, VideoError> {
        let data = self.chunk(index)?;
        let jpeg = with_huffman_tables(&data);
        let frame = decode_jpeg_frame(&jpeg, &self.path, &self.compat).map_err(|e| match e {
            VideoError::Decode { path, reason, .. } => VideoError::Decode {
                path,
                frame: Some(index),
                reason,
            },
            other => other,
        })?;
        if let Pixels::Gray16(_) = frame.pixels {
            return Err(self.decode_err(index, "12-bit JPEG not handled"));
        }
        Ok(frame)
    }

    fn decode_err(&self, index: usize, reason: &str) -> VideoError {
        VideoError::Decode {
            path: self.path.clone(),
            frame: Some(index),
            reason: reason.to_string(),
        }
    }
}

impl FrameSource for AviSource {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    /// Ported from core/AviVideo.cpp, AviVideo::setActiveFrame (the frame OpenCV returns,
    /// reduced as `Image::setImage` reduces it).
    fn read_frame(&mut self, index: usize) -> Result<Frame, VideoError> {
        if self.codec == Codec::Mjpeg {
            let frame = self.decode_mjpeg(index)?;
            if (frame.width, frame.height) != (self.info.width, self.info.height) {
                return Err(self.decode_err(
                    index,
                    &format!(
                        "frame of {} x {} in a stream of {} x {}",
                        frame.width, frame.height, self.info.width, self.info.height
                    ),
                ));
            }
            return Ok(frame);
        }
        let data = self.chunk(index)?;
        let (w, h) = (self.info.width as usize, self.info.height as usize);
        let raw = |bits: usize| -> Option<(usize, usize)> {
            let packed = w * bits / 8;
            let padded = (w * bits).div_ceil(32) * 4;
            if data.len() >= padded * h {
                Some((padded, packed))
            } else if data.len() >= packed * h {
                Some((packed, packed))
            } else {
                None
            }
        };
        let rows = |stride: usize| row_starts(h, stride, self.bottom_up);
        let pixels = match &self.codec {
            Codec::Mjpeg => unreachable!("MJPEG frames are read above"),
            Codec::Palette8 { gray } => {
                let (stride, used) =
                    raw(8).ok_or_else(|| self.decode_err(index, "frame too short"))?;
                let mut out = Vec::with_capacity(w * h);
                for start in rows(stride) {
                    out.extend(data[start..start + used].iter().map(|&i| gray[i as usize]));
                }
                out
            }
            Codec::Gray8 => {
                let (stride, used) =
                    raw(8).ok_or_else(|| self.decode_err(index, "frame too short"))?;
                let mut out = Vec::with_capacity(w * h);
                for start in rows(stride) {
                    out.extend_from_slice(&data[start..start + used]);
                }
                out
            }
            Codec::Bgr { bytes_per_pixel } => {
                let bpp = *bytes_per_pixel;
                let (stride, used) =
                    raw(bpp * 8).ok_or_else(|| self.decode_err(index, "frame too short"))?;
                let mut out = Vec::with_capacity(w * h);
                for start in rows(stride) {
                    out.extend(
                        data[start..start + used]
                            .chunks_exact(bpp)
                            .map(|p| gray_of(p[2], p[1], p[0], self.compat.grey_weights)),
                    );
                }
                out
            }
        };
        Ok(Frame::gray8(w as u32, h as u32, pixels))
    }
}

/// Walks the RIFF lists: stream headers from `hdrl`, every `movi` list (also in `AVIX`
/// lists) and `idx1`.
///
/// A `RIFF` size or a `movi` LIST size of 0 (a writer stopped before it filled them in, a
/// recording cut short, ledger V37) is taken to reach the end of the file, or of its `RIFF`
/// list; no intact file has either. Sizes beyond the end of the file are cut at it.
fn read_layout(file: &mut File, file_len: u64) -> std::io::Result<Layout> {
    let mut layout = Layout::default();
    let mut pos = 0u64;
    while pos + 12 <= file_len {
        let head = read_at(file, pos, 12)?;
        if &head[0..4] != b"RIFF" {
            break;
        }
        let size = match u64::from(u32_at(&head, 4)) {
            0 => file_len - pos - 8,
            s => s,
        };
        let end = (pos + 8 + size).min(file_len);
        let mut p = pos + 12;
        while p + 8 <= end {
            let ch = read_at(file, p, 8)?;
            let id = [ch[0], ch[1], ch[2], ch[3]];
            let mut csize = u64::from(u32_at(&ch, 4));
            let mut cend = (p + 8 + csize).min(end);
            if &id == b"LIST" && p + 12 <= end {
                let kind = read_at(file, p + 8, 4)?;
                if csize == 0 && &kind[..] == b"movi" {
                    csize = end - p - 8;
                    cend = end;
                }
                match &kind[..] {
                    b"hdrl" => read_hdrl(file, p + 12, cend, &mut layout)?,
                    b"movi" => layout.movi.push((p + 8, cend)),
                    _ => {}
                }
            } else if &id == b"idx1" && layout.idx1.is_none() {
                layout.idx1 = Some((p + 8, csize.min(cend - (p + 8)) as u32));
            }
            p = p + 8 + csize + (csize & 1);
        }
        pos = pos + 8 + size + (size & 1);
    }
    Ok(layout)
}

fn read_hdrl(file: &mut File, start: u64, end: u64, layout: &mut Layout) -> std::io::Result<()> {
    let mut p = start;
    while p + 8 <= end {
        let ch = read_at(file, p, 8)?;
        let csize = u64::from(u32_at(&ch, 4));
        let cend = (p + 8 + csize).min(end);
        if &ch[0..4] == b"LIST" && p + 12 <= end && &read_at(file, p + 8, 4)?[..] == b"strl" {
            let n = layout.streams.len();
            let (mut strh, mut strf, mut indx) = (Vec::new(), Vec::new(), None);
            let mut q = p + 12;
            while q + 8 <= cend {
                let sub = read_at(file, q, 8)?;
                let ssize = u64::from(u32_at(&sub, 4));
                let avail = ssize.min(cend.saturating_sub(q + 8)) as usize;
                match &sub[0..4] {
                    b"strh" => strh = read_at(file, q + 8, avail)?,
                    b"strf" => strf = read_at(file, q + 8, avail)?,
                    b"indx" => indx = Some(read_at(file, q + 8, avail)?),
                    _ => {}
                }
                q = q + 8 + ssize + (ssize & 1);
            }
            layout.streams.push((n, strh, strf, indx));
        }
        p = p + 8 + csize + (csize & 1);
    }
    Ok(())
}

/// The frame entries of stream `stream`, in order: those of the index, then the frames after
/// the last of them that no index holds ([`unindexed_frames`]), or, with no usable index, those
/// a scan of `movi` finds.
fn build_index(
    file: &mut File,
    file_len: u64,
    layout: &Layout,
    stream: usize,
    indx: Option<&[u8]>,
) -> std::io::Result<Vec<Entry>> {
    let tag = format!("{stream:02}");
    let is_frame = |id: &[u8]| {
        id.len() == 4 && id[0..2] == *tag.as_bytes() && (&id[2..4] == b"dc" || &id[2..4] == b"db")
    };

    // OpenDML index.
    if let Some(indx) = indx
        && let Some(mut entries) = odml_index(file, file_len, indx)?
    {
        let more = unindexed_frames(file, file_len, layout, &entries, &is_frame)?;
        entries.extend(more);
        return Ok(entries);
    }

    // idx1.
    if let (Some((pos, size)), Some(&(movi, _))) = (layout.idx1, layout.movi.first()) {
        let data = read_at(file, pos, size as usize)?;
        let raw: Vec<(u32, u32)> = data
            .as_chunks::<16>()
            .0
            .iter()
            .filter(|e| is_frame(&e[0..4]))
            .map(|e| (u32_at(e, 8), u32_at(e, 12)))
            .collect();
        if !raw.is_empty() {
            // Offsets are relative to the `movi` fourcc, or (some writers) absolute.
            let mut fits = |base: u64| -> std::io::Result<bool> {
                let at = base + u64::from(raw[0].0);
                if at + 8 > file_len {
                    return Ok(false);
                }
                Ok(is_frame(&read_at(file, at, 4)?))
            };
            let base = if fits(movi)? {
                Some(movi)
            } else if fits(0)? {
                Some(0)
            } else {
                None
            };
            if let Some(base) = base {
                let mut entries: Vec<Entry> = raw
                    .into_iter()
                    .map(|(off, len)| Entry {
                        offset: base + u64::from(off) + 8,
                        len,
                    })
                    .collect();
                let more = unindexed_frames(file, file_len, layout, &entries, &is_frame)?;
                entries.extend(more);
                return Ok(entries);
            }
        }
    }

    // No usable index: scan the movi lists.
    let mut entries = Vec::new();
    for &(movi, end) in &layout.movi {
        scan_list(file, movi + 4, end, &is_frame, &mut entries)?;
    }
    Ok(entries)
}

/// The frame chunks that lie in the `movi` lists after the last frame of `indexed` (the frames
/// an index gave), in file order (ledger V37): the frames of a recording cut short that no
/// index holds, such as those of the last `RIFF AVIX` list when the writer stopped before it
/// added that list to the OpenDML super index (and wrote its `ix##`), or the lists after the
/// one an `idx1` covers. FFmpeg reads them too (measured with `oracle/py/avi_cut_short.py`:
/// XMALab 3 reads such a frame when it is below the header's count, as in a file FFmpeg 9.1 was
/// killed while writing, whose count is the frames of its completed lists plus one), so this
/// is done in both modes; the XMALab 3 mode still reads no frame at or above the header's
/// count. An intact file has nothing after its last indexed frame but index chunks, other
/// streams' chunks and padding, so this reads a handful of chunk headers.
///
/// The scan starts only where a chunk is known to start: the end of the last indexed frame
/// (the one furthest into the file), and only when that frame's chunk header is in the file
/// and agrees with the index (a frame of this stream, the same size); otherwise nothing is
/// added. Every frame added lies after every indexed frame, so none is counted twice.
fn unindexed_frames(
    file: &mut File,
    file_len: u64,
    layout: &Layout,
    indexed: &[Entry],
    is_frame: &dyn Fn(&[u8]) -> bool,
) -> std::io::Result<Vec<Entry>> {
    let mut out = Vec::new();
    let Some(last) = indexed.iter().max_by_key(|e| e.offset) else {
        return Ok(out);
    };
    if last.offset < 8 || last.offset > file_len {
        return Ok(out);
    }
    let head = read_at(file, last.offset - 8, 8)?;
    if !is_frame(&head[0..4]) || u32_at(&head, 4) != last.len {
        return Ok(out);
    }
    let len = u64::from(last.len);
    let start = last.offset + len + (len & 1);
    for &(movi, end) in &layout.movi {
        scan_list(file, start.max(movi + 4), end, is_frame, &mut out)?;
    }
    Ok(out)
}

/// Appends the frame chunks in `start..end`, in file order.
///
/// A `LIST` or `RIFF` header is stepped over (the scan goes on at its first child, in the same
/// loop), as FFmpeg's AVI demuxer steps over `LIST` headers (`libavformat/avidec.c`,
/// `avi_sync`), so nesting costs neither stack nor memory: a file of nested lists cannot exhaust
/// the stack. Every other chunk is stepped over by its declared size, so no chunk header is read
/// from inside a chunk's declared data; the sizes of lists are not used, which, unlike a walk
/// that clips each list at its own declared end, also finds the frames of a list whose size was
/// never filled in (0: a `rec ` list, or a `RIFF AVIX` list and its `movi` of a recording cut
/// short, V37). Where a list's children and its declared end disagree (a damaged file), the two
/// walks differ: with a list size that is wrong this walk finds the frames the clipping walk
/// loses; with a chunk size that is wrong inside a list whose size is right, the clipping walk
/// went on at the list's end and found the frames after it, which this walk does not
/// (`review14c_avi_scan.rs` counts both).
fn scan_list(
    file: &mut File,
    start: u64,
    end: u64,
    is_frame: &dyn Fn(&[u8]) -> bool,
    out: &mut Vec<Entry>,
) -> std::io::Result<()> {
    let mut p = start;
    while p + 8 <= end {
        let ch = read_at(file, p, 8)?;
        let size = u32_at(&ch, 4);
        if &ch[0..4] == b"LIST" || &ch[0..4] == b"RIFF" {
            p += 12;
            continue;
        }
        if is_frame(&ch[0..4]) {
            out.push(Entry {
                offset: p + 8,
                len: size,
            });
        }
        p = p + 8 + u64::from(size) + u64::from(size & 1);
    }
    Ok(())
}

/// Reads an OpenDML index: a super index whose entries point to `ix##` standard index
/// chunks, or a standard index directly. `None` when it is not usable; the entries collected
/// up to then are dropped, and the caller builds the index from `idx1` or the scan alone, so
/// nothing is counted twice. Memory and reads are bounded by the file's size (ledger V37):
/// super index entries that name one `ix##` chunk twice, or overlapping chunks, make it
/// unusable.
fn odml_index(file: &mut File, file_len: u64, indx: &[u8]) -> std::io::Result<Option<Vec<Entry>>> {
    if indx.len() < 24 {
        return Ok(None);
    }
    let longs_per_entry = u16_at(indx, 0);
    let sub_type = indx[2];
    let index_type = indx[3];
    let n = u32_at(indx, 4) as usize;
    let mut out = Vec::new();
    match index_type {
        // AVI_INDEX_OF_INDEXES
        0 if longs_per_entry == 4 && sub_type == 0 => {
            // The parts of the file the `ix##` chunks read so far occupy (start to end).
            let mut read: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
            for k in 0..n {
                let e = 24 + 16 * k;
                if e + 16 > indx.len() {
                    return Ok(None);
                }
                let off = u64::from_le_bytes(indx[e..e + 8].try_into().expect("8 bytes"));
                let size = u32_at(indx, e + 8) as usize;
                // An offset near u64::MAX (a damaged file) makes the index unusable too.
                if off.checked_add(8 + 24).is_none_or(|end| end > file_len) || size < 24 {
                    return Ok(None);
                }
                let avail = size.min((file_len - off - 8) as usize);
                // Ledger V37: only the header and the entries it declares are read (a chunk
                // whose entries pass `avail` is unusable, as before), and the part of the
                // file they occupy may not overlap a part another entry of the super index
                // read: a super index whose entries name one `ix##` chunk again, or chunks
                // that overlap, is unusable. So every byte of the file is read, and every
                // index entry collected, at most once: the entries number less than
                // `file_len / 8`, however many the super index declares.
                let head = read_at(file, off + 8, 24)?;
                let declared = 24 + 8 * u64::from(u32_at(&head, 4));
                let len = declared.min(avail as u64);
                let (start, end) = (off + 8, off + 8 + len);
                let before = read.range(..end).next_back();
                if before.is_some_and(|(_, &b_end)| b_end > start) {
                    return Ok(None);
                }
                read.insert(start, end);
                let chunk = read_at(file, start, len as usize)?;
                if !standard_index(&chunk, &mut out) {
                    return Ok(None);
                }
            }
        }
        // AVI_INDEX_OF_CHUNKS held in the stream header
        1 => {
            if !standard_index(indx, &mut out) {
                return Ok(None);
            }
        }
        _ => return Ok(None),
    }
    // A super index reserved but never filled (a recording cut short, ledger V37) is not
    // usable: the frames are found through `idx1` or the scan of `movi`, as FFmpeg finds them.
    if out.is_empty() {
        return Ok(None);
    }
    Ok(Some(out))
}

/// Appends the entries of one `ix##` standard index (frame indices only).
fn standard_index(chunk: &[u8], out: &mut Vec<Entry>) -> bool {
    if chunk.len() < 24 || u16_at(chunk, 0) != 2 || chunk[2] != 0 || chunk[3] != 1 {
        return false;
    }
    let n = u32_at(chunk, 4) as usize;
    let base = u64::from_le_bytes(chunk[12..20].try_into().expect("8 bytes"));
    for k in 0..n {
        let e = 24 + 8 * k;
        if e + 8 > chunk.len() {
            return false;
        }
        // A base near u64::MAX (a damaged file) makes the index unusable.
        let Some(offset) = base.checked_add(u64::from(u32_at(chunk, e))) else {
            return false;
        };
        out.push(Entry {
            offset,
            len: u32_at(chunk, e + 4) & 0x7FFF_FFFF,
        });
    }
    true
}

/// The frame with the standard Huffman tables inserted after SOI when it has none
/// before its first SOS.
fn with_huffman_tables(jpeg: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    let mut i = 2;
    while i + 4 <= jpeg.len() {
        if jpeg[i] != 0xFF {
            break;
        }
        let marker = jpeg[i + 1];
        if marker == 0xFF {
            i += 1;
            continue;
        }
        if marker == 0xC4 {
            return std::borrow::Cow::Borrowed(jpeg);
        }
        if marker == 0xDA {
            break;
        }
        let len = usize::from(u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]));
        i += 2 + len;
    }
    let mut out = Vec::with_capacity(jpeg.len() + STANDARD_DHT.len() + 4);
    out.extend_from_slice(&jpeg[..2.min(jpeg.len())]);
    out.extend_from_slice(&[0xFF, 0xC4]);
    out.extend_from_slice(&((STANDARD_DHT.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&STANDARD_DHT);
    out.extend_from_slice(&jpeg[2.min(jpeg.len())..]);
    std::borrow::Cow::Owned(out)
}

/// The DHT payload of the standard tables (ITU T.81 annex K.3: DC and AC, luminance and
/// chrominance), which MJPEG frames without tables use.
#[rustfmt::skip]
static STANDARD_DHT: [u8; 416] = [
    0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x10, 0x00, 0x02,
    0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7d, 0x01, 0x02,
    0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71,
    0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33,
    0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a,
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53,
    0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73,
    0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92,
    0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9,
    0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4,
    0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
    0x01, 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x11, 0x00, 0x02,
    0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00, 0x01, 0x02, 0x77, 0x00, 0x01,
    0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22,
    0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62,
    0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a,
    0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a,
    0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3,
    0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
];

#[cfg(test)]
#[path = "review14c_avi_scan.rs"]
mod review14c_avi_scan;
