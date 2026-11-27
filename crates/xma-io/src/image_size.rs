//! Image dimensions read from file headers, without decoding pixels.
//!
//! The C++ loader decodes every calibration and grid image with `cv::imread` while loading
//! (`core/Image.cpp`, `Image::Image`), and takes the camera resolution from them
//! (`core/Camera.cpp`, `Camera::setResolutions`). Only the size is needed at load time, so
//! it is read from the header of the formats XMALab writes: TIFF, PNG, JPEG and BMP.

/// What a header says about an image's size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageSize {
    /// Width and height in pixels.
    Known(u32, u32),
    /// No data: `cv::imread` fails and the C++ image is 0 × 0.
    Missing,
    /// Data in a format not recognised here; OpenCV might still read it.
    Unrecognized,
}

impl ImageSize {
    /// The size the C++ would hold, where it is known here.
    #[must_use]
    pub fn dims(self) -> Option<(u32, u32)> {
        match self {
            ImageSize::Known(w, h) => Some((w, h)),
            ImageSize::Missing => Some((0, 0)),
            ImageSize::Unrecognized => None,
        }
    }
}

/// Reads the size from an image file's header. `None` data means the entry is absent.
#[must_use]
pub fn image_size(data: Option<&[u8]>) -> ImageSize {
    let Some(d) = data else {
        return ImageSize::Missing;
    };
    if d.is_empty() {
        return ImageSize::Missing;
    }
    tiff(d)
        .or_else(|| png(d))
        .or_else(|| jpeg(d))
        .or_else(|| bmp(d))
        .map_or(ImageSize::Unrecognized, |(w, h)| ImageSize::Known(w, h))
}

fn tiff(d: &[u8]) -> Option<(u32, u32)> {
    let le = match d.get(..4)? {
        [b'I', b'I', 42, 0] => true,
        [b'M', b'M', 0, 42] => false,
        _ => return None,
    };
    let u16_at = |p: usize| -> Option<u32> {
        let b = d.get(p..p + 2)?;
        Some(u32::from(if le {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        }))
    };
    let u32_at = |p: usize| -> Option<u32> {
        let b = d.get(p..p + 4)?;
        let a = [b[0], b[1], b[2], b[3]];
        Some(if le {
            u32::from_le_bytes(a)
        } else {
            u32::from_be_bytes(a)
        })
    };
    let ifd = usize::try_from(u32_at(4)?).ok()?;
    let n = usize::try_from(u16_at(ifd)?).ok()?;
    let (mut w, mut h) = (None, None);
    for k in 0..n {
        let e = ifd + 2 + 12 * k;
        let tag = u16_at(e)?;
        let kind = u16_at(e + 2)?;
        let value = match kind {
            3 => u16_at(e + 8)?,
            4 => u32_at(e + 8)?,
            _ => continue,
        };
        match tag {
            256 => w = Some(value),
            257 => h = Some(value),
            _ => {}
        }
    }
    Some((w?, h?))
}

fn png(d: &[u8]) -> Option<(u32, u32)> {
    if d.get(..8)? != b"\x89PNG\r\n\x1a\n" || d.get(12..16)? != b"IHDR" {
        return None;
    }
    let be = |p: usize| {
        d.get(p..p + 4)
            .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    };
    Some((be(16)?, be(20)?))
}

fn jpeg(d: &[u8]) -> Option<(u32, u32)> {
    if d.get(..2)? != [0xff, 0xd8] {
        return None;
    }
    let mut p = 2;
    loop {
        while *d.get(p)? != 0xff {
            p += 1;
        }
        while *d.get(p)? == 0xff {
            p += 1;
        }
        let marker = *d.get(p)?;
        p += 1;
        if marker == 0xd8 || (0xd0..=0xd7).contains(&marker) || marker == 0x01 {
            continue;
        }
        let len = usize::from(u16::from_be_bytes([*d.get(p)?, *d.get(p + 1)?]));
        let is_sof = (0xc0..=0xcf).contains(&marker) && !matches!(marker, 0xc4 | 0xc8 | 0xcc);
        if is_sof {
            let h = u16::from_be_bytes([*d.get(p + 3)?, *d.get(p + 4)?]);
            let w = u16::from_be_bytes([*d.get(p + 5)?, *d.get(p + 6)?]);
            return Some((u32::from(w), u32::from(h)));
        }
        if marker == 0xd9 || len < 2 {
            return None;
        }
        p += len;
    }
}

fn bmp(d: &[u8]) -> Option<(u32, u32)> {
    if d.get(..2)? != b"BM" {
        return None;
    }
    let header = u32::from_le_bytes(d.get(14..18)?.try_into().ok()?);
    if header == 12 {
        let w = u16::from_le_bytes(d.get(18..20)?.try_into().ok()?);
        let h = u16::from_le_bytes(d.get(20..22)?.try_into().ok()?);
        return Some((u32::from(w), u32::from(h)));
    }
    let w = i32::from_le_bytes(d.get(18..22)?.try_into().ok()?);
    let h = i32::from_le_bytes(d.get(22..26)?.try_into().ok()?);
    Some((w.unsigned_abs(), h.unsigned_abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers() {
        let mut tif = vec![b'I', b'I', 42, 0, 8, 0, 0, 0, 2, 0];
        tif.extend_from_slice(&[0, 1, 3, 0, 1, 0, 0, 0, 0, 4, 0, 0]); // width 1024 (SHORT)
        tif.extend_from_slice(&[1, 1, 4, 0, 1, 0, 0, 0, 0, 3, 0, 0]); // height 768 (LONG)
        assert_eq!(image_size(Some(&tif)), ImageSize::Known(1024, 768));

        let mut png_data = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
        png_data.extend_from_slice(&[0, 0, 2, 0, 0, 0, 1, 0]);
        assert_eq!(image_size(Some(&png_data)), ImageSize::Known(512, 256));

        let jpg = [
            0xff, 0xd8, 0xff, 0xe0, 0, 4, 0, 0, 0xff, 0xc0, 0, 11, 8, 0, 20, 0, 30, 1, 0, 0, 0,
        ];
        assert_eq!(image_size(Some(&jpg)), ImageSize::Known(30, 20));

        let mut bmp_data = b"BM".to_vec();
        bmp_data.resize(14, 0);
        bmp_data.extend_from_slice(&40u32.to_le_bytes());
        bmp_data.extend_from_slice(&7i32.to_le_bytes());
        bmp_data.extend_from_slice(&(-9i32).to_le_bytes());
        assert_eq!(image_size(Some(&bmp_data)), ImageSize::Known(7, 9));

        assert_eq!(image_size(None), ImageSize::Missing);
        assert_eq!(image_size(Some(b"P5 3 3 255")), ImageSize::Unrecognized);
        assert_eq!(
            image_size(Some(b"II*\0\xff\xff\xff\xff")),
            ImageSize::Unrecognized
        );
    }
}
