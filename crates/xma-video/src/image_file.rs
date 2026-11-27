//! Still images (TIFF, PNG, JPEG, BMP): what the C++ program holds after
//! `cv::imread(name, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH)` and the reduction of
//! `core/Image.cpp`.
//!
//! - The format is recognised from the content, as OpenCV's `findDecoder` does, not
//!   from the file name.
//! - Grey images keep their depth: 16-bit data is returned as [`crate::Pixels::Gray16`] with
//!   `bit_depth` 16, and [`Frame::to_gray8`] gives the C++ values (ledger V1). By default a
//!   16-bit grey file that declares fewer significant bits is returned at that depth (ledger
//!   V18): a PNG `sBIT` of 9 to 15 (the samples, which PNG stores scaled up to 16 bits, are
//!   shifted back down by `16 - sBIT`), or a TIFF `MaxSampleValue` below 65535 (the samples
//!   as stored, at the depth that holds that value; ignored if a sample exceeds it).
//!   Nothing is inferred from the data, and a declaration below 9 bits is ignored. In the
//!   XMALab 3 mode ([`VideoCompat::still_depth_16`]) every 16-bit still is 16 bits deep, as
//!   OpenCV reads neither field, so the reduction stays the C++'s 1/256.
//! - Colour images (RGB, RGBA, palette) are reduced to grey at once, with the weights
//!   [`VideoCompat::grey_weights`] chooses (ledger V2: correct by default, the C++'s
//!   swapped red and blue in XMALab 3 mode); 16-bit colour is reduced per sample to 8
//!   bits first, as `Image.cpp` does. Alpha is dropped, as `imread` drops it for these
//!   flags.
//! - 8-bit palette TIFF, which the `image` crate does not read, is read with the `tiff`
//!   crate and expanded as libtiff does for OpenCV (a 16-bit colour map is taken `>> 8`;
//!   one whose entries are all below 256 is taken as 8-bit).
//! - An Exif or TIFF orientation is applied, as `imread` does unless told not to.
//! - [`decode_colour_image`] gives a colour file in colour, for display: XMALab shows
//!   colour stills in colour (`Image::loadTexture`) while working on the grey reduction.

use std::io::Cursor;
use std::path::Path;

use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, RgbImage};

use crate::convert::{rgb_to_gray8, rgb16_to_gray8, scale_u16_to_u8};
use crate::{Frame, VideoCompat, VideoError};

/// A colour image for display: 8-bit R, G, B, A, row by row from the top (alpha 255;
/// the C++ drops alpha).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColourImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Decodes one still image held in memory with the default behaviour
/// ([`VideoCompat::FIXED`]). `name` is used only in error messages.
///
/// Ported from core/Image.cpp, Image::Image(QString, bool) and Image::setImage(QString, bool)
/// (the `cv::imread` call and the reduction that follows it; the horizontal flip is the
/// camera's and is left to the caller, [`Frame::flip_horizontal`]).
pub fn decode_image(bytes: &[u8], name: &Path) -> Result<Frame, VideoError> {
    decode_image_with(bytes, name, &VideoCompat::FIXED)
}

/// [`decode_image`] with the compatibility settings `compat`.
///
/// Ported from core/Image.cpp, Image::Image(QString, bool).
pub fn decode_image_with(
    bytes: &[u8],
    name: &Path,
    compat: &VideoCompat,
) -> Result<Frame, VideoError> {
    let (img, format) = decode(bytes, name, true)?;
    let declared = if compat.still_depth_16 {
        None
    } else {
        match format {
            ImageFormat::Png => png_sbit_grey(bytes).map(Declared::PngSbit),
            ImageFormat::Tiff => tiff_max_sample_value(bytes).map(Declared::TiffMax),
            _ => None,
        }
    };
    to_frame(img, format, name, compat, declared)
}

/// What a 16-bit grey file declares of its depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Declared {
    /// PNG `sBIT` of the grey channel: significant bits of samples scaled up to 16 bits.
    PngSbit(u8),
    /// TIFF `MaxSampleValue` (tag 281) of the first sample: samples stored as they are.
    TiffMax(u32),
}

/// The 16-bit samples and the depth to return them at, by what the file declares (module
/// documentation).
fn declared_depth(mut data: Vec<u16>, declared: Option<Declared>) -> (Vec<u16>, u8) {
    match declared {
        Some(Declared::PngSbit(bits @ 9..=15)) => {
            let shift = 16 - bits;
            data.iter_mut().for_each(|v| *v >>= shift);
            (data, bits)
        }
        Some(Declared::TiffMax(max @ 256..=65534)) => {
            let bits = (32 - max.leading_zeros()) as u8;
            if data.iter().all(|&v| u32::from(v) <= max) {
                (data, bits)
            } else {
                (data, 16)
            }
        }
        _ => (data, 16),
    }
}

/// The `sBIT` value of a PNG's grey channel (colour types 0 and 4), when the chunk is
/// present before the image data.
fn png_sbit_grey(bytes: &[u8]) -> Option<u8> {
    const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    if bytes.get(..8)? != SIGNATURE {
        return None;
    }
    let mut pos = 8;
    let mut colour_type = None;
    loop {
        let len = u32::from_be_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?) as usize;
        let kind = bytes.get(pos + 4..pos + 8)?;
        let data = bytes.get(pos + 8..pos + 8 + len)?;
        match kind {
            b"IHDR" => colour_type = Some(*data.get(9)?),
            b"sBIT" => {
                return match colour_type? {
                    0 | 4 => data.first().copied(),
                    _ => None,
                };
            }
            b"IDAT" | b"IEND" => return None,
            _ => {}
        }
        pos += 12 + len;
    }
}

/// The `MaxSampleValue` of a TIFF's first image (first sample), when the tag is present.
fn tiff_max_sample_value(bytes: &[u8]) -> Option<u32> {
    let mut dec = tiff::decoder::Decoder::new(Cursor::new(bytes)).ok()?;
    dec.find_tag_unsigned_vec::<u32>(tiff::tags::Tag::MaxSampleValue)
        .ok()
        .flatten()?
        .first()
        .copied()
}

/// Reads and decodes one still image file ([`decode_image`]).
///
/// Ported from core/Image.cpp, Image::Image(QString, bool).
pub fn read_image(path: &Path) -> Result<Frame, VideoError> {
    read_image_with(path, &VideoCompat::FIXED)
}

/// [`read_image`] with the compatibility settings `compat`.
pub fn read_image_with(path: &Path, compat: &VideoCompat) -> Result<Frame, VideoError> {
    decode_image_with(&read_file(path)?, path, compat)
}

/// The image in colour when the file is a colour image (RGB, RGBA or palette, 8 or 16
/// bit; 16-bit samples reduced as `convertTo(CV_8UC3, 1.0 / 256.0)`), `None` for a grey
/// file. Orientation is applied as for [`decode_image`].
///
/// Ported from core/Image.cpp, Image::Image (`image_color`, kept for display when the
/// file has more than one channel) and Image::loadTexture.
pub fn decode_colour_image(bytes: &[u8], name: &Path) -> Result<Option<ColourImage>, VideoError> {
    let (img, _) = decode(bytes, name, true)?;
    let (width, height) = (img.width(), img.height());
    let rgba = match img {
        DynamicImage::ImageRgb8(b) => expand_rgb(b.as_raw(), 3, |v| v),
        DynamicImage::ImageRgba8(b) => expand_rgb(b.as_raw(), 4, |v| v),
        DynamicImage::ImageRgb16(b) => expand_rgb(b.as_raw(), 3, |v| scale_u16_to_u8(v, 8)),
        DynamicImage::ImageRgba16(b) => expand_rgb(b.as_raw(), 4, |v| scale_u16_to_u8(v, 8)),
        _ => return Ok(None),
    };
    Ok(Some(ColourImage {
        width,
        height,
        rgba,
    }))
}

/// [`decode_colour_image`] of a file.
pub fn read_colour_image(path: &Path) -> Result<Option<ColourImage>, VideoError> {
    decode_colour_image(&read_file(path)?, path)
}

/// Decodes one JPEG frame of an MJPEG video, without any Exif orientation (FFmpeg's
/// MJPEG decoder ignores it).
pub(crate) fn decode_jpeg_frame(
    bytes: &[u8],
    name: &Path,
    compat: &VideoCompat,
) -> Result<Frame, VideoError> {
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return Err(VideoError::Decode {
            path: name.to_path_buf(),
            frame: None,
            reason: "frame is not a JPEG image".to_string(),
        });
    }
    let (img, format) = decode(bytes, name, false)?;
    to_frame(img, format, name, compat, None)
}

fn read_file(path: &Path) -> Result<Vec<u8>, VideoError> {
    std::fs::read(path).map_err(|e| VideoError::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

fn expand_rgb<T: Copy>(data: &[T], channels: usize, to8: impl Fn(T) -> u8) -> Vec<u8> {
    data.chunks_exact(channels)
        .flat_map(|p| [to8(p[0]), to8(p[1]), to8(p[2]), 255])
        .collect()
}

/// The most decoded data one image may need (2 GiB): far above any image of this program
/// (a 2048 x 2048 RGBA16 frame is 32 MiB). An image whose header asks for more is a decode
/// error, so that a damaged header cannot request an allocation of any size.
pub const MAX_DECODED_BYTES: u64 = 2 << 30;

/// Decodes to an `image` buffer, with the orientation applied when `orient`.
fn decode(
    bytes: &[u8],
    name: &Path,
    orient: bool,
) -> Result<(DynamicImage, ImageFormat), VideoError> {
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| VideoError::Io {
            path: name.to_path_buf(),
            source: e,
        })?;
    let format = match reader.format() {
        Some(f @ (ImageFormat::Png | ImageFormat::Tiff | ImageFormat::Jpeg | ImageFormat::Bmp)) => {
            f
        }
        Some(other) => {
            return Err(VideoError::Unsupported {
                path: name.to_path_buf(),
                reason: format!("image format {other:?} is not read natively"),
            });
        }
        None => {
            return Err(VideoError::Unsupported {
                path: name.to_path_buf(),
                reason: "not a TIFF, PNG, JPEG or BMP image".to_string(),
            });
        }
    };
    let decode_err = |e: image::ImageError| match e {
        image::ImageError::Unsupported(u) => VideoError::Unsupported {
            path: name.to_path_buf(),
            reason: u.to_string(),
        },
        e => VideoError::Decode {
            path: name.to_path_buf(),
            frame: None,
            reason: e.to_string(),
        },
    };
    let mut reader = reader;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_DECODED_BYTES);
    reader.limits(limits);
    let result = (|| {
        let mut decoder = reader.into_decoder().map_err(decode_err)?;
        // `from_decoder` allocates the whole image from the header's size before decoding;
        // a damaged header must not make that allocation.
        if decoder.total_bytes() > MAX_DECODED_BYTES {
            let (w, h) = decoder.dimensions();
            return Err(VideoError::Decode {
                path: name.to_path_buf(),
                frame: None,
                reason: format!(
                    "an image of {w} x {h} needs {} bytes, more than the limit of {MAX_DECODED_BYTES}",
                    decoder.total_bytes()
                ),
            });
        }
        let orientation = if orient {
            Some(decoder.orientation().map_err(decode_err)?)
        } else {
            None
        };
        let mut img = DynamicImage::from_decoder(decoder).map_err(decode_err)?;
        if let Some(o) = orientation {
            img.apply_orientation(o);
        }
        Ok(img)
    })();
    match result {
        Ok(img) => Ok((img, format)),
        Err(VideoError::Unsupported { .. }) if format == ImageFormat::Tiff => {
            match palette_tiff(bytes, orient) {
                Some(img) => Ok((img, format)),
                None => result.map(|img| (img, format)),
            }
        }
        Err(e) => Err(e),
    }
}

/// An 8-bit palette TIFF expanded to RGB, or `None` when the file is not one (or cannot
/// be read so).
///
/// Ported from libtiff `tif_getimage.c` (`checkcmap`, `cvtcmap`, `put8bitcmaptile`), which
/// OpenCV's TIFF decoder uses for palette images (`TIFFReadRGBAStrip`).
fn palette_tiff(bytes: &[u8], orient: bool) -> Option<DynamicImage> {
    let mut dec = tiff::decoder::Decoder::new(Cursor::new(bytes)).ok()?;
    let map = dec.get_tag_u16_vec(tiff::tags::Tag::ColorMap).ok()?;
    if map.len() != 3 * 256 {
        return None;
    }
    let orientation = dec
        .find_tag_unsigned::<u16>(tiff::tags::Tag::Orientation)
        .ok()
        .flatten();
    // The `tiff` crate decodes the strips of a palette image only as grey: read a copy
    // whose PhotometricInterpretation says min-is-black, which gives the indices.
    let grey = as_min_is_black(bytes)?;
    let mut dec = tiff::decoder::Decoder::new(Cursor::new(&grey[..])).ok()?;
    if dec.colortype().ok()? != tiff::ColorType::Gray(8) {
        return None;
    }
    let (w, h) = dec.dimensions().ok()?;
    let tiff::decoder::DecodingResult::U8(indices) = dec.read_image().ok()? else {
        return None;
    };
    if indices.len() != w as usize * h as usize {
        return None;
    }
    // checkcmap: a map whose entries are all below 256 is an old 8-bit map.
    let eight_bit = map.iter().all(|&v| v < 256);
    let cvt = |v: u16| if eight_bit { v as u8 } else { (v >> 8) as u8 };
    let mut rgb = Vec::with_capacity(indices.len() * 3);
    for &i in &indices {
        let i = usize::from(i);
        rgb.extend_from_slice(&[cvt(map[i]), cvt(map[256 + i]), cvt(map[512 + i])]);
    }
    let mut img = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, rgb)?);
    if orient
        && let Some(o) = orientation.and_then(|o| u8::try_from(o).ok())
        && let Some(o) = Orientation::from_exif(o)
    {
        img.apply_orientation(o);
    }
    Some(img)
}

/// A copy of a classic TIFF whose first image's PhotometricInterpretation (tag 262, one
/// SHORT) is 3 (palette), changed to 1 (min-is-black). `None` for anything else.
fn as_min_is_black(bytes: &[u8]) -> Option<Vec<u8>> {
    let le = match bytes.get(0..4)? {
        b"II*\0" => true,
        b"MM\0*" => false,
        _ => return None,
    };
    let u16_at = |o: usize| -> Option<u16> {
        let b: [u8; 2] = bytes.get(o..o + 2)?.try_into().ok()?;
        Some(if le {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        })
    };
    let u32_at = |o: usize| -> Option<u32> {
        let b: [u8; 4] = bytes.get(o..o + 4)?.try_into().ok()?;
        Some(if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    };
    let ifd = u32_at(4)? as usize;
    let n = usize::from(u16_at(ifd)?);
    for k in 0..n {
        let e = ifd + 2 + 12 * k;
        if u16_at(e)? == 262 && u16_at(e + 2)? == 3 && u32_at(e + 4)? == 1 && u16_at(e + 8)? == 3 {
            let mut out = bytes.to_vec();
            let one = if le {
                1u16.to_le_bytes()
            } else {
                1u16.to_be_bytes()
            };
            out[e + 8..e + 10].copy_from_slice(&one);
            return Some(out);
        }
    }
    None
}

/// The reduction of `core/Image.cpp` applied to a decoded image; a 16-bit grey image at the
/// depth `declared` gives (module documentation).
fn to_frame(
    img: DynamicImage,
    format: ImageFormat,
    name: &Path,
    compat: &VideoCompat,
    declared: Option<Declared>,
) -> Result<Frame, VideoError> {
    let (width, height) = (img.width(), img.height());
    let w = compat.grey_weights;
    let gray8 = |data: Vec<u8>| Ok(Frame::gray8(width, height, data));
    let gray16 = |data: Vec<u16>| {
        let (data, bits) = declared_depth(data, declared);
        Ok(Frame::gray16(width, height, data, bits))
    };
    match img {
        DynamicImage::ImageLuma8(b) => gray8(b.into_raw()),
        DynamicImage::ImageLumaA8(b) => gray8(b.into_raw().into_iter().step_by(2).collect()),
        DynamicImage::ImageLuma16(b) => gray16(b.into_raw()),
        DynamicImage::ImageLumaA16(b) => gray16(b.into_raw().into_iter().step_by(2).collect()),
        DynamicImage::ImageRgb8(b) => gray8(rgb_to_gray8(&b.into_raw(), 3, w)),
        DynamicImage::ImageRgba8(b) => gray8(rgb_to_gray8(&b.into_raw(), 4, w)),
        DynamicImage::ImageRgb16(b) => gray8(rgb16_to_gray8(&b.into_raw(), 3, w)),
        DynamicImage::ImageRgba16(b) => gray8(rgb16_to_gray8(&b.into_raw(), 4, w)),
        other => Err(VideoError::Unsupported {
            path: name.to_path_buf(),
            reason: format!(
                "{format:?} image with samples of type {:?} is not read natively",
                other.color()
            ),
        }),
    }
}

/// The samples of an image to write as TIFF ([`encode_tiff`]), row by row from the top.
#[derive(Debug, Clone, Copy)]
pub enum TiffSamples<'a> {
    /// 8-bit grey.
    Gray8(&'a [u8]),
    /// 8-bit R, G, B, interleaved.
    Rgb8(&'a [u8]),
    /// 16-bit grey with `bit_depth` (9 to 16) bits carrying information; below 16 the file
    /// states `MaxSampleValue` `2^bit_depth - 1`, which [`decode_image`] reads back as that
    /// depth (ledger V18).
    Gray16 { data: &'a [u16], bit_depth: u8 },
}

/// Encodes an image as an LZW-compressed TIFF with the horizontal predictor, as
/// `cv::imwrite` writes `.tif` by default (OpenCV's `TiffEncoder`); one strip per 8 kB or
/// so, as the `tiff` crate chooses (the bytes are not libtiff's).
///
/// Ported from the `cv::imwrite(name.tif, image)` of core/UndistortionObject.cpp,
/// UndistortionObject::undistort(Image*, QString, bool).
///
/// # Errors
/// A sample count that does not match the size, or a failed encoding.
pub fn encode_tiff(
    width: u32,
    height: u32,
    samples: TiffSamples<'_>,
) -> Result<Vec<u8>, VideoError> {
    use tiff::encoder::{Compression, Predictor, TiffEncoder, colortype};
    let n = width as usize * height as usize;
    let fail = |reason: String| VideoError::Decode {
        path: Path::new("(TIFF)").to_path_buf(),
        frame: None,
        reason,
    };
    let len = match samples {
        TiffSamples::Gray8(d) => d.len(),
        TiffSamples::Rgb8(d) => d.len() / 3,
        TiffSamples::Gray16 { data, .. } => data.len(),
    };
    if len != n || matches!(samples, TiffSamples::Rgb8(d) if d.len() % 3 != 0) {
        return Err(fail("sample count does not match the size".to_owned()));
    }
    let mut out = Cursor::new(Vec::new());
    let mut enc = TiffEncoder::new(&mut out)
        .map_err(|e| fail(e.to_string()))?
        .with_compression(Compression::Lzw)
        .with_predictor(Predictor::Horizontal);
    let r = match samples {
        TiffSamples::Gray8(d) => enc.write_image::<colortype::Gray8>(width, height, d),
        TiffSamples::Rgb8(d) => enc.write_image::<colortype::RGB8>(width, height, d),
        TiffSamples::Gray16 { data, bit_depth } => {
            let mut im = enc
                .new_image::<colortype::Gray16>(width, height)
                .map_err(|e| fail(e.to_string()))?;
            if bit_depth < 16 {
                let max = (1u32 << bit_depth) - 1;
                im.encoder()
                    .write_tag(tiff::tags::Tag::MaxSampleValue, max as u16)
                    .map_err(|e| fail(e.to_string()))?;
            }
            im.write_data(data)
        }
    };
    r.map_err(|e| fail(e.to_string()))?;
    Ok(out.into_inner())
}

/// Encodes an 8-bit grey image in the format its file name's extension names (`png`,
/// `tif`/`tiff`, `bmp`, `jpg`/`jpeg`, case-insensitive), as `cv::imwrite` chooses the
/// format by the name. The bytes are the `image` crate's, not OpenCV's (the TIFF is not
/// LZW-compressed, the JPEG quality is the crate's default).
///
/// Ported from core/Image.cpp, Image::save (the `cv::imwrite` of an 8-bit grey image).
///
/// # Errors
/// An extension that names no supported format, or a failed encoding.
pub fn encode_gray8(
    width: u32,
    height: u32,
    data: &[u8],
    name: &Path,
) -> Result<Vec<u8>, VideoError> {
    let format = name
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .and_then(|e| match e.as_str() {
            "png" => Some(ImageFormat::Png),
            "tif" | "tiff" => Some(ImageFormat::Tiff),
            "bmp" => Some(ImageFormat::Bmp),
            "jpg" | "jpeg" => Some(ImageFormat::Jpeg),
            _ => None,
        })
        .ok_or_else(|| VideoError::Unsupported {
            path: name.to_path_buf(),
            reason: "no image format for this extension".to_owned(),
        })?;
    let img = image::GrayImage::from_raw(width, height, data.to_vec()).ok_or_else(|| {
        VideoError::Decode {
            path: name.to_path_buf(),
            frame: None,
            reason: "pixel count does not match the size".to_owned(),
        }
    })?;
    let mut out = Cursor::new(Vec::new());
    DynamicImage::ImageLuma8(img)
        .write_to(&mut out, format)
        .map_err(|e| VideoError::Decode {
            path: name.to_path_buf(),
            frame: None,
            reason: e.to_string(),
        })?;
    Ok(out.into_inner())
}
