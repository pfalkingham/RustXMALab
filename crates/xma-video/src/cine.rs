//! Phantom `.cine` files. Ported from `core/CineVideo.cpp`.
//!
//! What the C++ handles, and so what is read here:
//! - `biCompression` 256: packed 10-bit (4 pixels in 5 bytes, most significant bit
//!   first), which the C++ reduces to 8 bits by `>> 2`, or through the 10-to-12-bit table
//!   `LinLUT` and `>> 4` when `RecBPP` is 12. By default the samples are returned at their
//!   own depth ([`Pixels::Gray16`]): 10 bits, or 12 bits through `LinLUT` (ledger V15); in
//!   XMALab 3 mode ([`VideoCompat::cine_packed_as_cpp`]) as the C++'s truncated 8-bit
//!   values;
//! - otherwise `biBitCount` 16: little-endian 16-bit, reduced with
//!   `convertTo(CV_8U, 1 / 2^(RealBPP - 8))`. Returned at full depth ([`Pixels::Gray16`]
//!   with `bit_depth` = `RealBPP`) for `RealBPP` 9 to 16, so the display can use it;
//!   [`Frame::to_gray8`] applies the C++ scale. `RealBPP` 8 gives 8-bit frames (scale 1,
//!   saturating). Outside 8 to 16 the C++ scale is meaningless (below 8 the unsigned
//!   exponent wraps and every frame is black): by default the data are taken as 16-bit,
//!   in XMALab 3 mode ([`VideoCompat::cine_real_bpp_as_cpp`]) as the C++;
//! - otherwise 8-bit, `biWidth * biHeight` bytes without row padding.
//!
//! Every frame is flipped vertically (rows are stored bottom-up). The frame offsets are 4
//! bytes in version 0 files and 8 bytes otherwise; the C++ reads version 0 offsets into an
//! `unsigned long`, which is 8 bytes on Linux and macOS, and here they are read as 4 bytes,
//! as on Windows (parity ledger V3).
//!
//! `RecBPP` is read only when `SoftwareVersion` is above 720; otherwise the C++ leaves it
//! uninitialised (ledger V4) and here it is taken as "not 12" (no lookup table, 10 bits).
//! In XMALab 3 mode packed frames are returned as 8-bit: the C++ truncates (`>> 2`,
//! `LinLUT >> 4`), which [`Frame::to_gray8`]'s rounding could not reproduce from 10 or
//! 12-bit samples. Returned at full depth (the default), their [`Frame::to_gray8`] rounds
//! instead, so it can differ from the C++'s 8-bit values by one grey level.
//!
//! A frame that cannot be read (at or beyond `ImageCount`, or whose data lie beyond the
//! end of the file): `read_frame` returns an error, so the viewer shows no image. The C++
//! keeps showing the last frame it read beyond `ImageCount` (measured, dataset
//! `video_cine_b`), and shows whatever the buffer holds for data cut short.
//!
//! Not read: the file header's `Compression` and the colour fields (`CFA`,
//! `bEnableColor`) are ignored, as in the C++, so a colour (Bayer) file gives its raw
//! mosaic. Any `biBitCount` other than 8 and 16 without packing, any `biCompression`
//! other than 0 and 256, and a negative or zero size are `Unsupported`; the C++ would show
//! the data misread as 8-bit grey.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::bytes::{i32_at, read_at, row_starts, u16_at, u32_at};
use crate::convert::scale_u16_to_u8;
use crate::{Frame, FrameSource, Pixels, SourceInfo, SourceKind, VideoCompat, VideoError};

/// Offsets inside the camera SETUP block, as `CineVideo::loadCineInfo` reads it
/// (field by field without alignment, from `OffSetup`).
const SETUP_FRAME_RATE: u64 = 768;
const SETUP_SOFTWARE_VERSION: u64 = 800;
const SETUP_REAL_BPP: u64 = 896;
/// `RecBPP` is read at `OffSetup + 10112` when `SoftwareVersion > 720`.
const SETUP_REC_BPP: u64 = 10112;

/// `biCompression` of Phantom's packed 10-bit images.
const BI_PACKED: u32 = 256;

/// The masks of `CineVideo::unpackImageData`.
const MASKS: [[u8; 2]; 4] = [[0xFF, 0xC0], [0x3F, 0xF0], [0x0F, 0xFC], [0x03, 0xFF]];

/// How a frame's data is laid out and reduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// Packed 10-bit; `lut` when `RecBPP == 12`; `truncate` to return the C++'s 8-bit values.
    Packed10 { lut: bool, truncate: bool },
    /// 16 bits per pixel. `Some(bits)`: returned as stored with `bits` (9 to 16)
    /// significant bits; `None`: reduced to 8 bits by `2^-shift` here.
    Bits16 { keep: Option<u8>, shift: u32 },
    /// 8 bits per pixel.
    Bits8,
}

/// The header fields the reader uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CineHeader {
    pub version: u16,
    pub image_count: u32,
    pub off_image_header: u32,
    pub off_setup: u32,
    pub off_image_offsets: u32,
    pub bi_width: i32,
    pub bi_height: i32,
    pub bi_bit_count: u16,
    pub bi_compression: u32,
    pub bi_size_image: u32,
    pub frame_rate: u32,
    pub software_version: u32,
    pub real_bpp: u32,
    /// `None` when `SoftwareVersion <= 720` (the C++ leaves it uninitialised).
    pub rec_bpp: Option<u32>,
}

/// Reader for Phantom `.cine`.
pub struct CineSource {
    info: SourceInfo,
    path: PathBuf,
    file: File,
    header: CineHeader,
    layout: Layout,
    /// Where each frame's image data starts (after its annotation block).
    data_offsets: Vec<u64>,
    /// Bytes of image data a frame needs.
    data_len: usize,
}

impl CineSource {
    /// Opens the source. Returns `VideoError::Unsupported` when the file is
    /// not of this kind or uses a feature this reader does not handle, so the
    /// caller can try another reader.
    ///
    /// Ported from core/CineVideo.cpp, CineVideo::loadCineInfo.
    pub fn open(path: &Path) -> Result<CineSource, VideoError> {
        Self::open_with(path, &VideoCompat::FIXED)
    }

    /// [`CineSource::open`] with the compatibility settings `compat`, which decide what
    /// a `RealBPP` outside 8 to 16 means and whether packed frames are truncated to 8 bits.
    ///
    /// Ported from core/CineVideo.cpp, CineVideo::loadCineInfo.
    pub fn open_with(path: &Path, compat: &VideoCompat) -> Result<CineSource, VideoError> {
        let io = |e: std::io::Error| VideoError::Io {
            path: path.to_path_buf(),
            source: e,
        };
        let corrupt = |reason: String| VideoError::Corrupt {
            path: path.to_path_buf(),
            reason,
        };
        let unsupported = |reason: String| VideoError::Unsupported {
            path: path.to_path_buf(),
            reason,
        };
        let mut file = File::open(path).map_err(io)?;
        let file_len = file.metadata().map_err(io)?.len();

        let mut head = [0u8; 44];
        if file_len < 44 {
            return Err(unsupported("too short for a .cine header".to_string()));
        }
        file.read_exact(&mut head).map_err(io)?;
        if &head[0..2] != b"CI" {
            return Err(unsupported("no 'CI' signature".to_string()));
        }
        let version = u16_at(&head, 6);
        let image_count = u32_at(&head, 20);
        let off_image_header = u32_at(&head, 24);
        let off_setup = u32_at(&head, 28);
        let off_image_offsets = u32_at(&head, 32);

        let bih = read_at(&mut file, u64::from(off_image_header), 40)
            .map_err(|_| corrupt("BITMAPINFOHEADER outside the file".to_string()))?;
        let bi_width = i32_at(&bih, 4);
        let bi_height = i32_at(&bih, 8);
        let bi_bit_count = u16_at(&bih, 14);
        let bi_compression = u32_at(&bih, 16);
        let bi_size_image = u32_at(&bih, 20);

        let setup = u64::from(off_setup);
        let field = |file: &mut File, off: u64| -> Result<u32, VideoError> {
            let b = read_at(file, setup + off, 4)
                .map_err(|_| corrupt("camera setup outside the file".to_string()))?;
            Ok(u32_at(&b, 0))
        };
        let frame_rate = field(&mut file, SETUP_FRAME_RATE)?;
        let software_version = field(&mut file, SETUP_SOFTWARE_VERSION)?;
        let real_bpp = field(&mut file, SETUP_REAL_BPP)?;
        let rec_bpp = if software_version > 720 {
            Some(field(&mut file, SETUP_REC_BPP)?)
        } else {
            None
        };
        let header = CineHeader {
            version,
            image_count,
            off_image_header,
            off_setup,
            off_image_offsets,
            bi_width,
            bi_height,
            bi_bit_count,
            bi_compression,
            bi_size_image,
            frame_rate,
            software_version,
            real_bpp,
            rec_bpp,
        };

        if bi_width <= 0 || bi_height <= 0 {
            return Err(unsupported(format!(
                "image size {bi_width} x {bi_height} is not handled"
            )));
        }
        let pixels = bi_width as usize * bi_height as usize;
        let layout = if bi_compression == BI_PACKED {
            Layout::Packed10 {
                lut: rec_bpp == Some(12),
                truncate: compat.cine_packed_as_cpp,
            }
        } else if bi_compression != 0 {
            return Err(unsupported(format!(
                "biCompression {bi_compression} is not handled"
            )));
        } else if bi_bit_count == 16 {
            bits16_layout(real_bpp, compat)
        } else if bi_bit_count == 8 {
            Layout::Bits8
        } else {
            return Err(unsupported(format!(
                "biBitCount {bi_bit_count} is not handled"
            )));
        };
        let data_len = match layout {
            Layout::Packed10 { .. } => pixels.div_ceil(4) * 5,
            Layout::Bits16 { .. } => pixels * 2,
            Layout::Bits8 => pixels,
        };
        if (bi_size_image as usize) < data_len {
            return Err(corrupt(format!(
                "biSizeImage {bi_size_image} is smaller than the {data_len} bytes a frame needs"
            )));
        }

        // Frame offsets (ledger V3: 4 bytes in version 0, as on Windows).
        let addr_size: usize = if version == 0 { 4 } else { 8 };
        let table_len = addr_size * image_count as usize;
        let table_off = u64::from(off_image_offsets);
        let table_avail = file_len.saturating_sub(table_off).min(table_len as u64) as usize;
        let table = read_at(&mut file, table_off, table_avail).map_err(io)?;
        let addresses: Vec<u64> = table
            .chunks_exact(addr_size)
            .map(|c| {
                if addr_size == 4 {
                    u64::from(u32_at(c, 0))
                } else {
                    u64::from_le_bytes(c.try_into().expect("8 bytes"))
                }
            })
            .collect();

        // Readable frames: the leading run whose annotation and data lie in the file.
        let mut data_offsets = Vec::with_capacity(addresses.len());
        for &addr in &addresses {
            let Some(end) = addr.checked_add(4) else {
                break;
            };
            if end > file_len {
                break;
            }
            let ann = read_at(&mut file, addr, 4).map_err(io)?;
            let start = addr + u64::from(u32_at(&ann, 0));
            if start.saturating_add(data_len as u64) > file_len {
                break;
            }
            data_offsets.push(start);
        }

        // Bits carried by the frames read_frame returns.
        let bit_depth = match layout {
            Layout::Bits16 {
                keep: Some(bits), ..
            } => bits,
            Layout::Packed10 {
                lut,
                truncate: false,
            } => {
                if lut {
                    12
                } else {
                    10
                }
            }
            _ => 8,
        };
        let info = SourceInfo {
            kind: SourceKind::Cine,
            frame_count: image_count as usize,
            readable_frames: data_offsets.len(),
            width: bi_width as u32,
            height: bi_height as u32,
            bit_depth,
            frame_rate: (frame_rate > 0).then_some(f64::from(frame_rate)),
        };
        Ok(CineSource {
            info,
            path: path.to_path_buf(),
            file,
            header,
            layout,
            data_offsets,
            data_len,
        })
    }

    /// The header fields read when opening.
    pub fn header(&self) -> &CineHeader {
        &self.header
    }
}

impl FrameSource for CineSource {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    /// Ported from core/CineVideo.cpp, CineVideo::setActiveFrame.
    fn read_frame(&mut self, index: usize) -> Result<Frame, VideoError> {
        let Some(&offset) = self.data_offsets.get(index) else {
            return Err(VideoError::FrameOutOfRange {
                index,
                readable: self.data_offsets.len(),
            });
        };
        let data = read_at(&mut self.file, offset, self.data_len).map_err(|e| VideoError::Io {
            path: self.path.clone(),
            source: e,
        })?;
        let (w, h) = (self.info.width as usize, self.info.height as usize);
        Ok(Frame {
            width: w as u32,
            height: h as u32,
            bit_depth: self.info.bit_depth,
            pixels: decode(self.layout, w, h, data),
        })
    }
}

/// How 16-bit data with `RealBPP` = `real_bpp` is returned.
///
/// The C++ reduces with `convertTo(CV_8U, 1.0f / pow(2, RealBPP - 8))`, `RealBPP`
/// unsigned (core/CineVideo.cpp, CineVideo::setActiveFrame). For `RealBPP` 9 to 16 the
/// samples are kept at full depth and [`Frame::to_gray8`] applies that scale; for 8 the
/// scale is 1 (saturating). Outside 8 to 16: in XMALab 3 mode as the C++ (below 8 the
/// exponent wraps and every frame is black; above 16, `2^-(RealBPP - 8)`); otherwise the
/// data are taken as 16-bit (ledger: `.cine` RealBPP outside 8 to 16).
fn bits16_layout(real_bpp: u32, compat: &VideoCompat) -> Layout {
    match real_bpp {
        9..=16 => Layout::Bits16 {
            keep: Some(real_bpp as u8),
            shift: real_bpp - 8,
        },
        8 => Layout::Bits16 {
            keep: None,
            shift: 0,
        },
        _ if compat.cine_real_bpp_as_cpp => Layout::Bits16 {
            keep: None,
            shift: real_bpp.wrapping_sub(8),
        },
        _ => Layout::Bits16 {
            keep: Some(16),
            shift: 8,
        },
    }
}

/// The pixels of a frame of `w` x `h` (both at least 1) from its stored `data` (the
/// `data_len` bytes of `layout`), rows in reverse order (`cv::flip(image, image, 0)`: the
/// rows are stored bottom-up). The output is built once, walking the stored rows from the
/// last to the first.
///
/// Ported from core/CineVideo.cpp, CineVideo::setActiveFrame and
/// CineVideo::unpackImageData.
fn decode(layout: Layout, w: usize, h: usize, mut data: Vec<u8>) -> Pixels {
    match layout {
        // The C++ reduction to 8 bits: `>> 4` of the table's 12 bits, else `>> 2`.
        Layout::Packed10 {
            lut,
            truncate: true,
        } => Pixels::Gray8(unpack10_flipped(&data, w, h, |v| {
            if lut {
                (LIN_LUT[v as usize] >> 4) as u8
            } else {
                (v >> 2) as u8
            }
        })),
        Layout::Packed10 {
            lut,
            truncate: false,
        } => Pixels::Gray16(unpack10_flipped(&data, w, h, |v| {
            if lut { LIN_LUT[v as usize] } else { v }
        })),
        // Frame::to_gray8 applies 2^-(bits - 8), the C++ scale.
        Layout::Bits16 { keep: Some(_), .. } => Pixels::Gray16(rows16_flipped(&data, w, h, |v| v)),
        Layout::Bits16 { keep: None, shift } => {
            Pixels::Gray8(rows16_flipped(&data, w, h, |v| scale_u16_to_u8(v, shift)))
        }
        Layout::Bits8 => {
            flip_rows_in_place(&mut data, w, h);
            Pixels::Gray8(data)
        }
    }
}

/// The 4 pixels of one 5-byte group of packed 10-bit data, most significant bit first.
///
/// Ported from core/CineVideo.cpp, CineVideo::unpackImageData (without its lookup table
/// and its reduction to 8 bits).
#[inline]
fn group10(group: &[u8; 5]) -> [u16; 4] {
    std::array::from_fn(|k| {
        let hi = u16::from(group[k] & MASKS[k][0]) << 8;
        let lo = u16::from(group[k + 1] & MASKS[k][1]);
        (hi | lo) >> (2 * (3 - k))
    })
}

/// Pixel `i` (in stored order) of packed 10-bit data.
fn sample10(packed: &[u8], i: usize) -> u16 {
    let g = 5 * (i / 4);
    group10(packed[g..g + 5].try_into().expect("5 bytes"))[i % 4]
}

/// The packed 10-bit samples of a `w` x `h` frame through `f`, rows in reverse order.
/// A row need not start or end on a group: those pixels are taken one by one.
fn unpack10_flipped<T: Copy + Default>(
    packed: &[u8],
    w: usize,
    h: usize,
    f: impl Fn(u16) -> T,
) -> Vec<T> {
    let mut out = vec![T::default(); w * h];
    for (row, start) in out.chunks_exact_mut(w).zip(row_starts(h, w, true)) {
        let lead = ((4 - start % 4) % 4).min(w);
        for (j, dst) in row[..lead].iter_mut().enumerate() {
            *dst = f(sample10(packed, start + j));
        }
        let groups = (w - lead) / 4;
        let first = 5 * (start + lead) / 4;
        let (body, tail) = row[lead..].split_at_mut(4 * groups);
        for (dst, group) in body
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(packed[first..first + 5 * groups].as_chunks::<5>().0)
        {
            *dst = group10(group).map(&f);
        }
        let tail_start = start + lead + 4 * groups;
        for (j, dst) in tail.iter_mut().enumerate() {
            *dst = f(sample10(packed, tail_start + j));
        }
    }
    out
}

/// The little-endian 16-bit samples of a `w` x `h` frame through `f`, rows in reverse
/// order.
fn rows16_flipped<T>(data: &[u8], w: usize, h: usize, f: impl Fn(u16) -> T) -> Vec<T> {
    let mut out = Vec::with_capacity(w * h);
    for start in row_starts(h, 2 * w, true) {
        let row = &data[start..start + 2 * w];
        out.extend(
            row.as_chunks::<2>()
                .0
                .iter()
                .map(|c| f(u16::from_le_bytes(*c))),
        );
    }
    out
}

/// `cv::flip(image, image, 0)` on `h` rows of `w` bytes, in place.
fn flip_rows_in_place(data: &mut [u8], w: usize, h: usize) {
    for y in 0..h / 2 {
        let (top, bottom) = data.split_at_mut((h - 1 - y) * w);
        top[y * w..(y + 1) * w].swap_with_slice(&mut bottom[..w]);
    }
}

/// The 10-to-12-bit table of `core/CineVideo.cpp` (`LinLUT`), copied verbatim.
#[rustfmt::skip]
static LIN_LUT: [u16; 1024] = [
    2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
    94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
    125, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 137, 138,
    139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
    156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 167, 168, 169, 170, 171, 172,
    173, 175, 176, 177, 178, 179, 181, 182, 183, 184, 186, 187, 188, 189, 191, 192,
    193, 194, 196, 197, 198, 200, 201, 202, 204, 205, 206, 208, 209, 210, 212, 213,
    215, 216, 217, 219, 220, 222, 223, 225, 226, 227, 229, 230, 232, 233, 235, 236,
    238, 239, 241, 242, 244, 245, 247, 249, 250, 252, 253, 255, 257, 258, 260, 261,
    263, 265, 266, 268, 270, 271, 273, 275, 276, 278, 280, 281, 283, 285, 287, 288,
    290, 292, 294, 295, 297, 299, 301, 302, 304, 306, 308, 310, 312, 313, 315, 317,
    319, 321, 323, 325, 327, 328, 330, 332, 334, 336, 338, 340, 342, 344, 346, 348,
    350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374, 377, 379, 381,
    383, 385, 387, 389, 391, 394, 396, 398, 400, 402, 404, 407, 409, 411, 413, 416,
    418, 420, 422, 425, 427, 429, 431, 434, 436, 438, 441, 443, 445, 448, 450, 452,
    455, 457, 459, 462, 464, 467, 469, 472, 474, 476, 479, 481, 484, 486, 489, 491,
    494, 496, 499, 501, 504, 506, 509, 511, 514, 517, 519, 522, 524, 527, 529, 532,
    535, 537, 540, 543, 545, 548, 551, 553, 556, 559, 561, 564, 567, 570, 572, 575,
    578, 581, 583, 586, 589, 592, 594, 597, 600, 603, 606, 609, 611, 614, 617, 620,
    623, 626, 629, 632, 635, 637, 640, 643, 646, 649, 652, 655, 658, 661, 664, 667,
    670, 673, 676, 679, 682, 685, 688, 691, 694, 698, 701, 704, 707, 710, 713, 716,
    719, 722, 726, 729, 732, 735, 738, 742, 745, 748, 751, 754, 758, 761, 764, 767,
    771, 774, 777, 781, 784, 787, 790, 794, 797, 800, 804, 807, 811, 814, 817, 821,
    824, 828, 831, 834, 838, 841, 845, 848, 852, 855, 859, 862, 866, 869, 873, 876,
    880, 883, 887, 890, 894, 898, 901, 905, 908, 912, 916, 919, 923, 927, 930, 934,
    938, 941, 945, 949, 952, 956, 960, 964, 967, 971, 975, 979, 982, 986, 990, 994,
    998, 1001, 1005, 1009, 1013, 1017, 1021, 1025, 1028, 1032, 1036, 1040, 1044, 1048, 1052, 1056,
    1060, 1064, 1068, 1072, 1076, 1080, 1084, 1088, 1092, 1096, 1100, 1104, 1108, 1112, 1116, 1120,
    1124, 1128, 1132, 1137, 1141, 1145, 1149, 1153, 1157, 1162, 1166, 1170, 1174, 1178, 1183, 1187,
    1191, 1195, 1200, 1204, 1208, 1212, 1217, 1221, 1225, 1230, 1234, 1238, 1243, 1247, 1251, 1256,
    1260, 1264, 1269, 1273, 1278, 1282, 1287, 1291, 1295, 1300, 1304, 1309, 1313, 1318, 1322, 1327,
    1331, 1336, 1340, 1345, 1350, 1354, 1359, 1363, 1368, 1372, 1377, 1382, 1386, 1391, 1396, 1400,
    1405, 1410, 1414, 1419, 1424, 1428, 1433, 1438, 1443, 1447, 1452, 1457, 1462, 1466, 1471, 1476,
    1481, 1486, 1490, 1495, 1500, 1505, 1510, 1515, 1520, 1524, 1529, 1534, 1539, 1544, 1549, 1554,
    1559, 1564, 1569, 1574, 1579, 1584, 1589, 1594, 1599, 1604, 1609, 1614, 1619, 1624, 1629, 1634,
    1639, 1644, 1649, 1655, 1660, 1665, 1670, 1675, 1680, 1685, 1691, 1696, 1701, 1706, 1711, 1717,
    1722, 1727, 1732, 1738, 1743, 1748, 1753, 1759, 1764, 1769, 1775, 1780, 1785, 1791, 1796, 1801,
    1807, 1812, 1818, 1823, 1828, 1834, 1839, 1845, 1850, 1856, 1861, 1867, 1872, 1878, 1883, 1889,
    1894, 1900, 1905, 1911, 1916, 1922, 1927, 1933, 1939, 1944, 1950, 1956, 1961, 1967, 1972, 1978,
    1984, 1989, 1995, 2001, 2007, 2012, 2018, 2024, 2030, 2035, 2041, 2047, 2053, 2058, 2064, 2070,
    2076, 2082, 2087, 2093, 2099, 2105, 2111, 2117, 2123, 2129, 2135, 2140, 2146, 2152, 2158, 2164,
    2170, 2176, 2182, 2188, 2194, 2200, 2206, 2212, 2218, 2224, 2231, 2237, 2243, 2249, 2255, 2261,
    2267, 2273, 2279, 2286, 2292, 2298, 2304, 2310, 2317, 2323, 2329, 2335, 2341, 2348, 2354, 2360,
    2366, 2373, 2379, 2385, 2392, 2398, 2404, 2411, 2417, 2423, 2430, 2436, 2443, 2449, 2455, 2462,
    2468, 2475, 2481, 2488, 2494, 2501, 2507, 2514, 2520, 2527, 2533, 2540, 2546, 2553, 2559, 2566,
    2572, 2579, 2586, 2592, 2599, 2605, 2612, 2619, 2625, 2632, 2639, 2645, 2652, 2659, 2666, 2672,
    2679, 2686, 2693, 2699, 2706, 2713, 2720, 2726, 2733, 2740, 2747, 2754, 2761, 2767, 2774, 2781,
    2788, 2795, 2802, 2809, 2816, 2823, 2830, 2837, 2844, 2850, 2857, 2864, 2871, 2878, 2885, 2893,
    2900, 2907, 2914, 2921, 2928, 2935, 2942, 2949, 2956, 2963, 2970, 2978, 2985, 2992, 2999, 3006,
    3013, 3021, 3028, 3035, 3042, 3049, 3057, 3064, 3071, 3078, 3086, 3093, 3100, 3108, 3115, 3122,
    3130, 3137, 3144, 3152, 3159, 3166, 3174, 3181, 3189, 3196, 3204, 3211, 3218, 3226, 3233, 3241,
    3248, 3256, 3263, 3271, 3278, 3286, 3294, 3301, 3309, 3316, 3324, 3331, 3339, 3347, 3354, 3362,
    3370, 3377, 3385, 3393, 3400, 3408, 3416, 3423, 3431, 3439, 3447, 3454, 3462, 3470, 3478, 3486,
    3493, 3501, 3509, 3517, 3525, 3533, 3540, 3548, 3556, 3564, 3572, 3580, 3588, 3596, 3604, 3612,
    3620, 3628, 3636, 3644, 3652, 3660, 3668, 3676, 3684, 3692, 3700, 3708, 3716, 3724, 3732, 3740,
    3749, 3757, 3765, 3773, 3781, 3789, 3798, 3806, 3814, 3822, 3830, 3839, 3847, 3855, 3863, 3872,
    3880, 3888, 3897, 3905, 3913, 3922, 3930, 3938, 3947, 3955, 3963, 3972, 3980, 3989, 3997, 4006,
    4014, 4022, 4031, 4039, 4048, 4056, 4064, 4095, 4095, 4095, 4095, 4095, 4095, 4095, 4095, 4095,
];

#[cfg(test)]
mod tests {
    use super::*;

    // The reader's frame decoding before the review of 20 November 2026 (finding 8 of
    // video-model-cli.md), kept as the reference the one-pass `decode` is compared with.

    /// Unpacks `count` packed 10-bit pixels and reduces them to 8 bits, as the C++ does.
    fn unpack10(packed: &[u8], count: usize, lut: bool) -> Vec<u8> {
        unpack10_samples(packed, count, lut)
            .into_iter()
            .map(|v| if lut { (v >> 4) as u8 } else { (v >> 2) as u8 })
            .collect()
    }

    /// Unpacks `count` packed 10-bit pixels: the 10-bit samples, or with `lut` the 12-bit
    /// values of `LinLUT`.
    fn unpack10_samples(packed: &[u8], count: usize, lut: bool) -> Vec<u16> {
        let mut out = Vec::with_capacity(count);
        for group in packed.as_chunks::<5>().0 {
            for k in 0..4 {
                if out.len() == count {
                    return out;
                }
                let hi = u16::from(group[k] & MASKS[k][0]) << 8;
                let lo = u16::from(group[k + 1] & MASKS[k][1]);
                let tmp = (hi | lo) >> (2 * (3 - k));
                out.push(if lut { LIN_LUT[tmp as usize] } else { tmp });
            }
        }
        out
    }

    /// `cv::flip(image, image, 0)`: the rows in reverse order.
    fn flip_rows<T: Copy>(data: &[T], width: usize) -> Vec<T> {
        data.chunks_exact(width).rev().flatten().copied().collect()
    }

    fn old_decode(layout: Layout, w: usize, h: usize, data: &[u8]) -> Pixels {
        match layout {
            Layout::Packed10 {
                lut,
                truncate: true,
            } => Pixels::Gray8(flip_rows(&unpack10(data, w * h, lut), w)),
            Layout::Packed10 {
                lut,
                truncate: false,
            } => Pixels::Gray16(flip_rows(&unpack10_samples(data, w * h, lut), w)),
            Layout::Bits16 { keep, shift } => {
                let values: Vec<u16> = data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes(*c))
                    .collect();
                let values = flip_rows(&values, w);
                match keep {
                    Some(_) => Pixels::Gray16(values),
                    None => Pixels::Gray8(crate::convert::scale16_to_gray8(&values, shift)),
                }
            }
            Layout::Bits8 => Pixels::Gray8(flip_rows(data, w)),
        }
    }

    fn data_len(layout: Layout, pixels: usize) -> usize {
        match layout {
            Layout::Packed10 { .. } => pixels.div_ceil(4) * 5,
            Layout::Bits16 { .. } => pixels * 2,
            Layout::Bits8 => pixels,
        }
    }

    const LAYOUTS: [Layout; 10] = [
        Layout::Packed10 {
            lut: false,
            truncate: false,
        },
        Layout::Packed10 {
            lut: true,
            truncate: false,
        },
        Layout::Packed10 {
            lut: false,
            truncate: true,
        },
        Layout::Packed10 {
            lut: true,
            truncate: true,
        },
        Layout::Bits16 {
            keep: Some(12),
            shift: 4,
        },
        Layout::Bits16 {
            keep: Some(16),
            shift: 8,
        },
        Layout::Bits16 {
            keep: None,
            shift: 0,
        },
        Layout::Bits16 {
            keep: None,
            shift: 3,
        },
        Layout::Bits16 {
            keep: None,
            shift: 5u32.wrapping_sub(8),
        },
        Layout::Bits8,
    ];

    #[test]
    fn review13_cine_decode_equals_old_path_on_synthetic_frames() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        };
        let sizes = [
            (1, 1),
            (1, 2),
            (2, 1),
            (1, 9),
            (9, 1),
            (3, 5),
            (5, 3),
            (6, 7),
            (7, 6),
            (4, 4),
            (8, 3),
            (13, 11),
            (1023, 3),
            (1, 1023),
            (1024, 2),
        ];
        let mut checked = 0;
        for layout in LAYOUTS {
            for (w, h) in sizes {
                let len = data_len(layout, w * h);
                for fill in [None, Some(0u8), Some(0xFF), Some(0xAA), Some(0x55)] {
                    let data: Vec<u8> = (0..len).map(|_| fill.unwrap_or_else(&mut next)).collect();
                    let old = old_decode(layout, w, h, &data);
                    let new = decode(layout, w, h, data);
                    assert_eq!(new, old, "{layout:?} {w} x {h} fill {fill:?}");
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, LAYOUTS.len() * sizes.len() * 5);
    }

    /// Every frame of every `.cine` reference file and test fixture, in both modes.
    #[test]
    fn review13_cine_read_frame_equals_old_path_on_files() {
        let mut files: Vec<PathBuf> = Vec::new();
        for name in ["video_cine_a", "video_cine_b", "video_cine_v0"] {
            if let Some(ds) = xma_oracle::dataset(name) {
                let dir = ds.project.parent().expect("dataset folder").to_path_buf();
                files.extend(
                    std::fs::read_dir(&dir)
                        .expect("dataset folder")
                        .map(|e| e.expect("entry").path())
                        .filter(|p| p.extension().is_some_and(|e| e == "cine")),
                );
            }
        }
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        files.extend(
            std::fs::read_dir(&fixtures)
                .expect("tests/data")
                .map(|e| e.expect("entry").path())
                .filter(|p| p.extension().is_some_and(|e| e == "cine")),
        );
        files.sort();
        let mut frames = 0;
        for path in &files {
            for compat in [VideoCompat::FIXED, VideoCompat::XMALAB3] {
                let Ok(mut src) = CineSource::open_with(path, &compat) else {
                    continue;
                };
                let (w, h) = (src.info.width as usize, src.info.height as usize);
                for i in 0..src.data_offsets.len() {
                    let raw = read_at(&mut src.file, src.data_offsets[i], src.data_len)
                        .expect("raw frame");
                    let want = old_decode(src.layout, w, h, &raw);
                    let got = src.read_frame(i).expect("frame");
                    assert_eq!(got.pixels, want, "{} frame {i}", path.display());
                    frames += 1;
                }
            }
        }
        eprintln!("{} files, {frames} frames compared", files.len());
        assert!(frames > 0);
    }

    #[test]
    fn unpack_most_significant_bit_first() {
        // Pixels 1, 2, 1023, 512 packed as 40 bits.
        let v: u64 = (1 << 30) | (2 << 20) | (1023 << 10) | 512;
        let bytes: Vec<u8> = (0..5).map(|k| (v >> (8 * (4 - k))) as u8).collect();
        assert_eq!(unpack10(&bytes, 4, false), vec![0, 0, 255, 128]);
        assert_eq!(
            unpack10(&bytes, 4, true),
            vec![
                (LIN_LUT[1] >> 4) as u8,
                (LIN_LUT[2] >> 4) as u8,
                255,
                (LIN_LUT[512] >> 4) as u8
            ]
        );
        assert_eq!(unpack10_samples(&bytes, 4, false), vec![1, 2, 1023, 512]);
        assert_eq!(
            unpack10_samples(&bytes, 4, true),
            vec![LIN_LUT[1], LIN_LUT[2], 4095, LIN_LUT[512]]
        );
    }
}
