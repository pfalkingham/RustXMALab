//! Byte helpers shared by the native readers ([`crate::avi`], [`crate::cine`]): little-endian
//! fields, reading a block at an offset, and the order in which stored rows are taken.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

/// `len` bytes of `file` from `offset`.
pub(crate) fn read_at(file: &mut File, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

pub(crate) fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

pub(crate) fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

pub(crate) fn i32_at(b: &[u8], o: usize) -> i32 {
    u32_at(b, o) as i32
}

/// Where each of `height` stored rows, `stride` units apart, starts, in the order of a
/// top-down image: the last stored row first when the rows are stored `bottom_up`.
pub(crate) fn row_starts(
    height: usize,
    stride: usize,
    bottom_up: bool,
) -> impl Iterator<Item = usize> {
    (0..height).map(move |k| if bottom_up { height - 1 - k } else { k } * stride)
}
