//! The zip container of `.xma` project files.
//!
//! XMALab 3 reads and writes projects through QuaZip (`ui/ProjectFileIO.cpp`,
//! `ProjectFileIO::zipFromFolderToFile` and `ProjectFileIO::unzipFromFileToFolder`). This
//! module reads the same container straight from memory, with no temporary folder (parity
//! ledger F5): the central directory, stored and deflated entries with their CRC-32
//! checked, the archive comment, data descriptors and zip64 sizes. [`write_archive`]
//! writes one as QuaZip 1.4 does, field for field (only the deflate bytes differ), and
//! [`Archive::local_header`] exposes the local headers so tests can check that.
//!
//! # Entry names
//!
//! Names are kept as the raw bytes stored in the archive ([`Entry::name`]). The C++ sets
//! QuaZip's file-name codec to `IBM866` (code page 866) for both reading and writing
//! (`ui/ProjectFileIO.cpp:2315` and `:2417`, parity ledger F6). QuaZip 1.x decodes a name
//! as UTF-8 when the entry's general-purpose flag bit 11 is set, and with the codec
//! otherwise; [`Entry::decoded_name`] does the same. [`encode_cp866`] is the inverse that a
//! writer needs (characters outside code page 866 become `?`, as Qt's codec writes them).
//!
//! # Safety against hostile archives
//!
//! Every offset and size is checked against the data before use, and nothing is allocated
//! from a declared size alone: the central directory is walked within its stated bounds,
//! and inflated output grows with the data actually produced, up to the declared size. A
//! malformed archive gives a [`ZipError`], never a panic.

use std::fmt;

use miniz_oxide::inflate::TINFLStatus;
use miniz_oxide::inflate::core::{DecompressorOxide, decompress, inflate_flags};

/// Compression method of an entry. Only the two methods XMALab uses are supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Method {
    /// Method 0, no compression.
    Stored,
    /// Method 8, deflate.
    Deflated,
}

impl Method {
    /// The method number stored in the archive.
    #[must_use]
    pub const fn code(self) -> u16 {
        match self {
            Method::Stored => 0,
            Method::Deflated => 8,
        }
    }

    /// The method for a stored number, if supported.
    #[must_use]
    pub const fn from_code(code: u16) -> Option<Self> {
        match code {
            0 => Some(Method::Stored),
            8 => Some(Method::Deflated),
            _ => None,
        }
    }
}

/// An MS-DOS date and time, as zip stores it: local time, two-second resolution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DosDateTime {
    /// Bits 15-9 year since 1980, 8-5 month, 4-0 day.
    pub date: u16,
    /// Bits 15-11 hour, 10-5 minute, 4-0 seconds divided by two.
    pub time: u16,
}

impl DosDateTime {
    /// Builds a value from calendar fields. Seconds are rounded down to an even number.
    /// Returns `None` for fields outside what the format holds (years 1980 to 2107).
    #[must_use]
    pub fn from_fields(
        year: u16,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
    ) -> Option<Self> {
        if !(1980..=2107).contains(&year)
            || !(1..=12).contains(&month)
            || !(1..=31).contains(&day)
            || hour > 23
            || minute > 59
            || second > 59
        {
            return None;
        }
        Some(Self {
            date: ((year - 1980) << 9) | (u16::from(month) << 5) | u16::from(day),
            time: (u16::from(hour) << 11) | (u16::from(minute) << 5) | u16::from(second / 2),
        })
    }

    /// `(year, month, day, hour, minute, second)` as stored, without validation.
    #[must_use]
    pub fn fields(self) -> (u16, u8, u8, u8, u8, u8) {
        let d = self.date;
        let t = self.time;
        (
            1980 + (d >> 9),
            ((d >> 5) & 0x0f) as u8,
            (d & 0x1f) as u8,
            (t >> 11) as u8,
            ((t >> 5) & 0x3f) as u8,
            ((t & 0x1f) * 2) as u8,
        )
    }
}

/// General-purpose flag bit 0: the entry is encrypted.
pub const FLAG_ENCRYPTED: u16 = 1 << 0;
/// General-purpose flag bit 3: CRC and sizes follow the data in a data descriptor.
pub const FLAG_DATA_DESCRIPTOR: u16 = 1 << 3;
/// General-purpose flag bit 11: the name and comment are UTF-8.
pub const FLAG_UTF8: u16 = 1 << 11;

/// One archive entry: everything the central directory records about it.
///
/// The same type describes an entry to write, so a reader and a writer can carry entries
/// across a load and save.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The name as stored, undecoded.
    pub name: Vec<u8>,
    /// The entry comment as stored, undecoded.
    pub comment: Vec<u8>,
    pub method: Method,
    /// General-purpose bit flags ([`FLAG_UTF8`], [`FLAG_DATA_DESCRIPTOR`], ...).
    pub flags: u16,
    pub modified: DosDateTime,
    /// "Version made by": high byte the host system (3 is Unix), low byte the version.
    pub version_made_by: u16,
    /// "Version needed to extract".
    pub version_needed: u16,
    /// Internal file attributes (bit 0: text file).
    pub internal_attributes: u16,
    /// External file attributes; on Unix hosts the mode is in the high 16 bits.
    pub external_attributes: u32,
    /// Extra field of the central directory record.
    pub extra: Vec<u8>,
    pub crc32: u32,
    pub compressed_size: u64,
    pub uncompressed_size: u64,
}

impl Entry {
    /// The name as the C++ sees it: UTF-8 when flag bit 11 is set, code page 866 otherwise
    /// (QuaZip with the `IBM866` codec). Invalid UTF-8 is replaced with U+FFFD.
    #[must_use]
    pub fn decoded_name(&self) -> String {
        if self.flags & FLAG_UTF8 != 0 {
            String::from_utf8_lossy(&self.name).into_owned()
        } else {
            decode_cp866(&self.name)
        }
    }

    /// True for a directory entry (name ending in `/`).
    #[must_use]
    pub fn is_dir(&self) -> bool {
        self.name.last() == Some(&b'/')
    }
}

/// Where an entry's data starts in the archive, and the entry.
#[derive(Clone, Debug)]
struct Located {
    entry: Entry,
    local_header_offset: u64,
}

/// A zip archive read into memory.
#[derive(Clone, Debug)]
pub struct Archive<'a> {
    data: &'a [u8],
    entries: Vec<Located>,
    comment: Vec<u8>,
}

/// Why an archive could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ZipError {
    /// No end-of-central-directory record: not a zip file, or truncated.
    NotAZip,
    /// A structure extends past the end of the data or contradicts another.
    Corrupt(String),
    /// A feature this reader does not support (multi-disk archives, encryption, a
    /// compression method other than stored or deflate).
    Unsupported(String),
    /// The inflated data does not match the entry: bad deflate stream, wrong length.
    BadData(String),
    /// The CRC-32 of the data differs from the one recorded.
    CrcMismatch { expected: u32, actual: u32 },
}

impl fmt::Display for ZipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZipError::NotAZip => write!(f, "not a zip archive (no end of central directory)"),
            ZipError::Corrupt(m) => write!(f, "corrupt zip archive: {m}"),
            ZipError::Unsupported(m) => write!(f, "unsupported zip feature: {m}"),
            ZipError::BadData(m) => write!(f, "bad entry data: {m}"),
            ZipError::CrcMismatch { expected, actual } => {
                write!(
                    f,
                    "CRC-32 mismatch: recorded {expected:08x}, data {actual:08x}"
                )
            }
        }
    }
}

impl std::error::Error for ZipError {}

const SIG_LOCAL: u32 = 0x0403_4b50;
const SIG_CENTRAL: u32 = 0x0201_4b50;
const SIG_EOCD: u32 = 0x0605_4b50;
const SIG_EOCD64: u32 = 0x0606_4b50;
const SIG_EOCD64_LOCATOR: u32 = 0x0706_4b50;
const EOCD_LEN: usize = 22;
const CENTRAL_LEN: usize = 46;
const LOCAL_LEN: usize = 30;

/// Little-endian reads with bounds checks.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn at(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8], ZipError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| {
                ZipError::Corrupt(format!(
                    "structure at offset {} runs past the end",
                    self.pos
                ))
            })?;
        let out = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u16(&mut self) -> Result<u16, ZipError> {
        let b = self.bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, ZipError> {
        let b = self.bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, ZipError> {
        let b = self.bytes(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
}

fn to_usize(v: u64, what: &str) -> Result<usize, ZipError> {
    usize::try_from(v).map_err(|_| ZipError::Corrupt(format!("{what} {v} is too large")))
}

impl<'a> Archive<'a> {
    /// Reads the central directory of an archive held in memory. Entry data is read later,
    /// one entry at a time, with [`Archive::read`].
    pub fn new(data: &'a [u8]) -> Result<Self, ZipError> {
        let end = end_record(data, 0)?;
        let entries = directory(data, 0, &end)?;
        Ok(Self {
            data,
            entries,
            comment: end.comment,
        })
    }

    /// The archive comment as stored. XMALab writes `XROMM Project File`.
    #[must_use]
    pub fn comment(&self) -> &[u8] {
        &self.comment
    }

    /// The entries in central-directory order.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = &Entry> + '_ {
        self.entries.iter().map(|l| &l.entry)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry at `index`, in central-directory order.
    #[must_use]
    pub fn entry(&self, index: usize) -> Option<&Entry> {
        self.entries.get(index).map(|l| &l.entry)
    }

    /// Reads and checks the data of entry `index`: inflates it if needed, checks its length
    /// against the declared size and its CRC-32 against the recorded one.
    pub fn read(&self, index: usize) -> Result<Vec<u8>, ZipError> {
        let located = self
            .entries
            .get(index)
            .ok_or_else(|| ZipError::Corrupt(format!("no entry {index}")))?;
        let e = &located.entry;
        if e.flags & FLAG_ENCRYPTED != 0 {
            return Err(ZipError::Unsupported("encrypted entry".into()));
        }
        let start = to_usize(located.local_header_offset, "local header offset")?;
        let mut c = Cursor::at(self.data, start);
        if c.u32()? != SIG_LOCAL {
            return Err(ZipError::Corrupt(format!(
                "local header at offset {start} has a bad signature"
            )));
        }
        let mut header = Cursor::at(self.data, start + LOCAL_LEN - 4);
        let name_len = usize::from(header.u16()?);
        let extra_len = usize::from(header.u16()?);
        let data_start = start + LOCAL_LEN + name_len + extra_len;
        let csize = to_usize(e.compressed_size, "compressed size")?;
        let raw = data_start
            .checked_add(csize)
            .filter(|&end| end <= self.data.len())
            .map(|end| &self.data[data_start..end])
            .ok_or_else(|| {
                ZipError::Corrupt("entry data runs past the end of the archive".into())
            })?;
        let usize_declared = to_usize(e.uncompressed_size, "uncompressed size")?;
        let out = match e.method {
            Method::Stored => {
                if e.compressed_size != e.uncompressed_size {
                    return Err(ZipError::Corrupt(format!(
                        "stored entry with compressed size {} and uncompressed size {}",
                        e.compressed_size, e.uncompressed_size
                    )));
                }
                raw.to_vec()
            }
            Method::Deflated => inflate(raw, usize_declared)?,
        };
        let actual = crc32(&out);
        if actual != e.crc32 {
            return Err(ZipError::CrcMismatch {
                expected: e.crc32,
                actual,
            });
        }
        Ok(out)
    }
}

/// What the end-of-central-directory record (and a zip64 end record) says, with offsets
/// in the file.
#[derive(Clone, Debug)]
struct EndRecord {
    n_entries: u64,
    /// Offset of the central directory in the file.
    cd_offset: u64,
    cd_size: u64,
    /// Offset in the file where the central directory must end (the end record that
    /// follows it).
    cd_limit: u64,
    comment: Vec<u8>,
}

/// `offset` (from the start of the file) as a position in `data`, which holds the file from
/// `base` on.
fn relative(offset: u64, base: u64, what: &str) -> Result<usize, ZipError> {
    let r = offset
        .checked_sub(base)
        .ok_or_else(|| ZipError::Corrupt(format!("{what} {offset} lies before the data read")))?;
    to_usize(r, what)
}

/// Reads the end record(s) from `data`, the last bytes of a file starting at offset
/// `base` (0 for a whole archive in memory).
fn end_record(data: &[u8], base: u64) -> Result<EndRecord, ZipError> {
    let eocd = find_eocd(data)?;
    let mut c = Cursor::at(data, eocd + 4);
    let disk = c.u16()?;
    let cd_disk = c.u16()?;
    let n_disk = c.u16()?;
    let n_total = c.u16()?;
    let cd_size32 = c.u32()?;
    let cd_offset32 = c.u32()?;
    let comment_len = usize::from(c.u16()?);
    let comment = c.bytes(comment_len)?.to_vec();
    if disk != 0 || cd_disk != 0 || n_disk != n_total {
        return Err(ZipError::Unsupported("multi-disk archive".into()));
    }

    let mut n_entries = u64::from(n_total);
    let mut cd_size = u64::from(cd_size32);
    let mut cd_offset = u64::from(cd_offset32);
    let mut cd_limit = base + eocd as u64;
    // A zip64 end record, when the 32-bit fields are saturated and a locator is present.
    if (n_total == 0xffff || cd_size32 == 0xffff_ffff || cd_offset32 == 0xffff_ffff) && eocd >= 20 {
        let mut l = Cursor::at(data, eocd - 20);
        if l.u32()? == SIG_EOCD64_LOCATOR {
            let _disk = l.u32()?;
            let rec_abs = l.u64()?;
            let rec = relative(rec_abs, base, "zip64 end record offset")?;
            let mut r = Cursor::at(data, rec);
            if r.u32()? != SIG_EOCD64 {
                return Err(ZipError::Corrupt("zip64 end record signature".into()));
            }
            let _size = r.u64()?;
            let _made_by = r.u16()?;
            let _needed = r.u16()?;
            let disk64 = r.u32()?;
            let cd_disk64 = r.u32()?;
            let n_disk64 = r.u64()?;
            n_entries = r.u64()?;
            cd_size = r.u64()?;
            cd_offset = r.u64()?;
            if disk64 != 0 || cd_disk64 != 0 || n_disk64 != n_entries {
                return Err(ZipError::Unsupported("multi-disk archive".into()));
            }
            cd_limit = rec_abs;
        }
    }
    Ok(EndRecord {
        n_entries,
        cd_offset,
        cd_size,
        cd_limit,
        comment,
    })
}

/// Reads the central directory that `end` describes from `data`, which holds the file from
/// offset `base` on and must include the whole directory.
fn directory(data: &[u8], base: u64, end: &EndRecord) -> Result<Vec<Located>, ZipError> {
    let n_entries = end.n_entries;
    let cd_start = relative(end.cd_offset, base, "central directory offset")?;
    let cd_len = to_usize(end.cd_size, "central directory size")?;
    let cd_limit = relative(end.cd_limit, base, "central directory end")?;
    let cd_end = cd_start
        .checked_add(cd_len)
        .filter(|&e| e <= cd_limit && e <= data.len())
        .ok_or_else(|| ZipError::Corrupt("central directory runs past its end record".into()))?;
    // Each record is at least 46 bytes, which bounds the count we accept before reading.
    if n_entries > (cd_len / CENTRAL_LEN) as u64 {
        return Err(ZipError::Corrupt(format!(
            "{n_entries} entries cannot fit in a central directory of {cd_len} bytes"
        )));
    }
    let cd = &data[..cd_end];
    let mut c = Cursor::at(cd, cd_start);
    let mut entries = Vec::with_capacity(n_entries as usize);
    for i in 0..n_entries {
        let at = c.pos;
        if c.u32()? != SIG_CENTRAL {
            return Err(ZipError::Corrupt(format!(
                "central directory record {i} at offset {at} has a bad signature"
            )));
        }
        let version_made_by = c.u16()?;
        let version_needed = c.u16()?;
        let flags = c.u16()?;
        let method_code = c.u16()?;
        let time = c.u16()?;
        let date = c.u16()?;
        let crc32 = c.u32()?;
        let mut compressed_size = u64::from(c.u32()?);
        let mut uncompressed_size = u64::from(c.u32()?);
        let name_len = usize::from(c.u16()?);
        let extra_len = usize::from(c.u16()?);
        let comment_len = usize::from(c.u16()?);
        let disk_start = c.u16()?;
        let internal_attributes = c.u16()?;
        let external_attributes = c.u32()?;
        let mut local_header_offset = u64::from(c.u32()?);
        let name = c.bytes(name_len)?.to_vec();
        let extra = c.bytes(extra_len)?.to_vec();
        let comment = c.bytes(comment_len)?.to_vec();

        // Zip64 extended information: present fields follow the order of the
        // saturated 32-bit fields.
        let need_usize = uncompressed_size == 0xffff_ffff;
        let need_csize = compressed_size == 0xffff_ffff;
        let need_offset = local_header_offset == 0xffff_ffff;
        if need_usize || need_csize || need_offset {
            let z = find_extra(&extra, 0x0001)
                .ok_or_else(|| ZipError::Corrupt(format!("entry {i} lacks its zip64 sizes")))?;
            let mut zc = Cursor::at(z, 0);
            if need_usize {
                uncompressed_size = zc.u64()?;
            }
            if need_csize {
                compressed_size = zc.u64()?;
            }
            if need_offset {
                local_header_offset = zc.u64()?;
            }
        }
        if disk_start != 0 && disk_start != 0xffff {
            return Err(ZipError::Unsupported("multi-disk archive".into()));
        }
        let method = Method::from_code(method_code).ok_or_else(|| {
            ZipError::Unsupported(format!(
                "compression method {method_code} in entry '{}'",
                String::from_utf8_lossy(&name)
            ))
        })?;
        entries.push(Located {
            entry: Entry {
                name,
                comment,
                method,
                flags,
                modified: DosDateTime { date, time },
                version_made_by,
                version_needed,
                internal_attributes,
                external_attributes,
                extra,
                crc32,
                compressed_size,
                uncompressed_size,
            },
            local_header_offset,
        });
    }
    Ok(entries)
}

/// Finds the end-of-central-directory record, searching back over a comment of up to
/// 65535 bytes. The record whose comment length reaches exactly to the end of the data is
/// preferred; otherwise the last record found is used.
fn find_eocd(data: &[u8]) -> Result<usize, ZipError> {
    if data.len() < EOCD_LEN {
        return Err(ZipError::NotAZip);
    }
    let lowest = data.len().saturating_sub(EOCD_LEN + 0xffff);
    let mut fallback = None;
    let mut pos = data.len() - EOCD_LEN;
    loop {
        if data[pos..pos + 4] == SIG_EOCD.to_le_bytes() {
            let comment_len = usize::from(u16::from_le_bytes([data[pos + 20], data[pos + 21]]));
            if pos + EOCD_LEN + comment_len == data.len() {
                return Ok(pos);
            }
            if fallback.is_none() && pos + EOCD_LEN + comment_len <= data.len() {
                fallback = Some(pos);
            }
        }
        if pos == lowest {
            break;
        }
        pos -= 1;
    }
    fallback.ok_or(ZipError::NotAZip)
}

fn find_extra(extra: &[u8], id: u16) -> Option<&[u8]> {
    let mut pos = 0;
    while pos + 4 <= extra.len() {
        let tag = u16::from_le_bytes([extra[pos], extra[pos + 1]]);
        let len = usize::from(u16::from_le_bytes([extra[pos + 2], extra[pos + 3]]));
        let end = pos + 4 + len;
        if end > extra.len() {
            return None;
        }
        if tag == id {
            return Some(&extra[pos + 4..end]);
        }
        pos = end;
    }
    None
}

/// Inflates a raw deflate stream that must produce exactly `declared` bytes. The output
/// buffer grows with the data produced and never beyond `declared`.
fn inflate(input: &[u8], declared: usize) -> Result<Vec<u8>, ZipError> {
    let flags = inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
    let mut decompressor = Box::<DecompressorOxide>::default();
    // One byte more than declared lets an over-long stream be detected.
    let limit = declared.saturating_add(1);
    let mut out = vec![0u8; limit.min(64 * 1024)];
    let mut in_pos = 0;
    let mut out_pos = 0;
    loop {
        let (status, consumed, written) = decompress(
            &mut decompressor,
            &input[in_pos..],
            &mut out,
            out_pos,
            flags,
        );
        in_pos += consumed;
        out_pos += written;
        match status {
            TINFLStatus::Done => break,
            TINFLStatus::HasMoreOutput => {
                if out.len() >= limit {
                    return Err(ZipError::BadData(format!(
                        "inflates to more than the declared {declared} bytes"
                    )));
                }
                let grown = out.len().saturating_mul(2).min(limit);
                out.resize(grown, 0);
            }
            TINFLStatus::FailedCannotMakeProgress | TINFLStatus::NeedsMoreInput => {
                return Err(ZipError::BadData("deflate stream is truncated".into()));
            }
            other => {
                return Err(ZipError::BadData(format!(
                    "deflate stream is invalid ({other:?})"
                )));
            }
        }
    }
    if out_pos != declared {
        return Err(ZipError::BadData(format!(
            "inflates to {out_pos} bytes, but {declared} are declared"
        )));
    }
    out.truncate(out_pos);
    Ok(out)
}

const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

/// The tables of slicing-by-8: `CRC_TABLES[k][b]` is the CRC register after byte `b`
/// followed by `k` zero bytes, so eight bytes are folded in with eight lookups.
const CRC_TABLES: [[u32; 256]; 8] = {
    let mut t = [[0u32; 256]; 8];
    t[0] = CRC_TABLE;
    let mut k = 1;
    while k < 8 {
        let mut i = 0;
        while i < 256 {
            let prev = t[k - 1][i];
            t[k][i] = (prev >> 8) ^ CRC_TABLE[(prev & 0xff) as usize];
            i += 1;
        }
        k += 1;
    }
    t
};

/// CRC-32 (ISO 3309, polynomial 0xEDB88320), as zip uses.
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    crc32_update(0, data)
}

/// The CRC-32 of the data whose CRC-32 is `crc` followed by `data`: `crc32(a ++ b)` is
/// `crc32_update(crc32(a), b)`.
///
/// Eight bytes at a time (slicing-by-8), the rest one at a time; the result is the
/// byte-by-byte table routine's (test `review13_crc32_matches_bytewise`).
#[must_use]
pub fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    let t = &CRC_TABLES;
    let mut crc = !crc;
    let (chunks, rest) = data.as_chunks::<8>();
    for c in chunks {
        let one = u32::from_le_bytes([c[0], c[1], c[2], c[3]]) ^ crc;
        let two = u32::from_le_bytes([c[4], c[5], c[6], c[7]]);
        crc = t[7][(one & 0xff) as usize]
            ^ t[6][((one >> 8) & 0xff) as usize]
            ^ t[5][((one >> 16) & 0xff) as usize]
            ^ t[4][(one >> 24) as usize]
            ^ t[3][(two & 0xff) as usize]
            ^ t[2][((two >> 8) & 0xff) as usize]
            ^ t[1][((two >> 16) & 0xff) as usize]
            ^ t[0][(two >> 24) as usize];
    }
    for &b in rest {
        crc = CRC_TABLE[((crc ^ u32::from(b)) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

/// Code page 866, bytes 0x80 to 0xFF.
const CP866_HIGH: [char; 128] = [
    'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', //
    'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я', //
    'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', //
    '░', '▒', '▓', '│', '┤', '╡', '╢', '╖', '╕', '╣', '║', '╗', '╝', '╜', '╛', '┐', //
    '└', '┴', '┬', '├', '─', '┼', '╞', '╟', '╚', '╔', '╩', '╦', '╠', '═', '╬', '╧', //
    '╨', '╤', '╥', '╙', '╘', '╒', '╓', '╫', '╪', '┘', '┌', '█', '▄', '▌', '▐', '▀', //
    'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', //
    'Ё', 'ё', 'Є', 'є', 'Ї', 'ї', 'Ў', 'ў', '°', '∙', '·', '√', '№', '¤', '■', '\u{a0}',
];

/// Decodes code page 866 (IBM866), as Qt's codec does. Every byte maps to one character.
#[must_use]
pub fn decode_cp866(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|&b| {
            if b < 0x80 {
                char::from(b)
            } else {
                CP866_HIGH[usize::from(b - 0x80)]
            }
        })
        .collect()
}

/// Encodes text in code page 866. Characters the code page lacks become `?`, as Qt's
/// codec writes them.
#[must_use]
pub fn encode_cp866(text: &str) -> Vec<u8> {
    text.chars()
        .map(|ch| {
            if ch.is_ascii() {
                ch as u8
            } else {
                CP866_HIGH
                    .iter()
                    .position(|&c| c == ch)
                    .map_or(b'?', |i| 0x80 + i as u8)
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------------------
// Archives read from a file

/// The largest central directory [`FileArchive::open`] reads: 64 MiB, room for
/// [`MAX_FILE_ARCHIVE_ENTRIES`] entries with names of some 200 bytes. A larger one is refused
/// rather than read into memory (ledger F78).
pub const MAX_DIRECTORY_BYTES: u64 = 64 << 20;

/// The most entries [`FileArchive::open`] accepts: 262 144 (2^18), four times what a zip
/// without zip64 can hold (65 535). An XMA Portal download holds a few videos or image
/// sequences; an unpacked sequence of 65 535 frames fits. Refused before the directory is
/// read (ledger F78: a directory of 64 MiB of one-byte names, 1.4 million entries, held
/// 280 MB).
pub const MAX_FILE_ARCHIVE_ENTRIES: u64 = 1 << 18;

/// Bytes read from the end of a file to find the end records: the record, a comment of up
/// to 65535 bytes, the zip64 locator and the zip64 record.
const TAIL_BYTES: u64 = (EOCD_LEN + 0xffff + 20 + 56) as u64;

/// Why a file archive could not be read: the file, or the archive in it.
#[derive(Debug)]
pub enum FileArchiveError {
    Io(std::io::Error),
    Zip(ZipError),
}

impl fmt::Display for FileArchiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileArchiveError::Io(e) => write!(f, "{e}"),
            FileArchiveError::Zip(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for FileArchiveError {}

impl From<std::io::Error> for FileArchiveError {
    fn from(e: std::io::Error) -> Self {
        FileArchiveError::Io(e)
    }
}

impl From<ZipError> for FileArchiveError {
    fn from(e: ZipError) -> Self {
        FileArchiveError::Zip(e)
    }
}

/// A zip archive read from a file without loading it: the central directory is read into
/// memory (at most [`MAX_DIRECTORY_BYTES`]), and each entry is streamed from the file when
/// it is read, so an archive of any size (an XMA Portal download holding videos) needs
/// memory only for its directory and the entries asked for.
///
/// The checks are those of [`Archive`]: every offset and size is checked before use, an
/// entry inflates to exactly its declared size and its CRC-32 is checked.
#[derive(Debug)]
pub struct FileArchive {
    file: std::fs::File,
    len: u64,
    entries: Vec<Located>,
    comment: Vec<u8>,
}

fn read_at(file: &mut std::fs::File, offset: u64, len: usize) -> Result<Vec<u8>, FileArchiveError> {
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = Vec::new();
    file.take(len as u64).read_to_end(&mut buf)?;
    if buf.len() != len {
        return Err(ZipError::Corrupt(format!(
            "the file ends inside the structure at offset {offset}"
        ))
        .into());
    }
    Ok(buf)
}

impl FileArchive {
    /// Opens `path` and reads its central directory.
    pub fn open(path: &std::path::Path) -> Result<Self, FileArchiveError> {
        let mut file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        let tail_start = len.saturating_sub(TAIL_BYTES);
        let tail = read_at(
            &mut file,
            tail_start,
            to_usize(len - tail_start, "file size")?,
        )?;
        let end = end_record(&tail, tail_start)?;
        if end.n_entries > MAX_FILE_ARCHIVE_ENTRIES {
            return Err(ZipError::Unsupported(format!(
                "{} entries (at most {MAX_FILE_ARCHIVE_ENTRIES} are read)",
                end.n_entries
            ))
            .into());
        }
        if end.cd_offset > len || end.cd_limit > len {
            return Err(ZipError::Corrupt(
                "central directory lies past the end of the file".into(),
            )
            .into());
        }
        let start = end.cd_offset.min(tail_start);
        let span = len - start;
        if span > MAX_DIRECTORY_BYTES + TAIL_BYTES {
            return Err(ZipError::Unsupported(format!(
                "a central directory of {span} bytes (at most {MAX_DIRECTORY_BYTES} are read)"
            ))
            .into());
        }
        let entries = if start == tail_start {
            directory(&tail, tail_start, &end)?
        } else {
            let data = read_at(&mut file, start, to_usize(span, "central directory size")?)?;
            directory(&data, start, &end)?
        };
        Ok(Self {
            file,
            len,
            entries,
            comment: end.comment,
        })
    }

    /// The archive comment as stored.
    #[must_use]
    pub fn comment(&self) -> &[u8] {
        &self.comment
    }

    /// The entries in central-directory order.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = &Entry> + '_ {
        self.entries.iter().map(|l| &l.entry)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry at `index`, in central-directory order.
    #[must_use]
    pub fn entry(&self, index: usize) -> Option<&Entry> {
        self.entries.get(index).map(|l| &l.entry)
    }

    /// Reads entry `index` into memory, refusing one whose declared size is above `limit`
    /// bytes.
    pub fn read_limited(&mut self, index: usize, limit: u64) -> Result<Vec<u8>, FileArchiveError> {
        let declared = self
            .entry(index)
            .ok_or_else(|| ZipError::Corrupt(format!("no entry {index}")))?
            .uncompressed_size;
        if declared > limit {
            return Err(ZipError::Unsupported(format!(
                "an entry of {declared} bytes (at most {limit} are read into memory)"
            ))
            .into());
        }
        let mut out = Vec::new();
        self.copy_to(index, &mut out, &mut |_| true)?;
        Ok(out)
    }

    /// Streams entry `index` into `out`: inflated if needed, its length checked against
    /// the declared size and its CRC-32 against the recorded one (a mismatch is an error
    /// after the data was written, as with any stream). `progress` is called with the
    /// number of bytes written so far after each block; returning false stops with
    /// [`ZipError::BadData`] ("stopped").
    pub fn copy_to(
        &mut self,
        index: usize,
        out: &mut dyn std::io::Write,
        progress: &mut dyn FnMut(u64) -> bool,
    ) -> Result<u64, FileArchiveError> {
        use std::io::{Read, Seek, SeekFrom};
        let located = self
            .entries
            .get(index)
            .ok_or_else(|| ZipError::Corrupt(format!("no entry {index}")))?
            .clone();
        let e = &located.entry;
        if e.flags & FLAG_ENCRYPTED != 0 {
            return Err(ZipError::Unsupported("encrypted entry".into()).into());
        }
        let start = located.local_header_offset;
        let header = read_at(&mut self.file, start, LOCAL_LEN)?;
        let mut c = Cursor::at(&header, 0);
        if c.u32()? != SIG_LOCAL {
            return Err(ZipError::Corrupt(format!(
                "local header at offset {start} has a bad signature"
            ))
            .into());
        }
        let mut h = Cursor::at(&header, LOCAL_LEN - 4);
        let name_len = u64::from(h.u16()?);
        let extra_len = u64::from(h.u16()?);
        let data_start = start + LOCAL_LEN as u64 + name_len + extra_len;
        let csize = e.compressed_size;
        if data_start
            .checked_add(csize)
            .is_none_or(|end| end > self.len)
        {
            return Err(
                ZipError::Corrupt("entry data runs past the end of the archive".into()).into(),
            );
        }
        if e.method == Method::Stored && e.compressed_size != e.uncompressed_size {
            return Err(ZipError::Corrupt(format!(
                "stored entry with compressed size {} and uncompressed size {}",
                e.compressed_size, e.uncompressed_size
            ))
            .into());
        }
        self.file.seek(SeekFrom::Start(data_start))?;
        let mut input = (&mut self.file).take(csize);
        let declared = e.uncompressed_size;
        let mut written = 0u64;
        let mut crc = 0u32;
        let mut inbuf = vec![0u8; 64 * 1024];
        let mut emit =
            |data: &[u8], written: &mut u64, crc: &mut u32| -> Result<(), FileArchiveError> {
                *written += data.len() as u64;
                if *written > declared {
                    return Err(ZipError::BadData(format!(
                        "inflates to more than the declared {declared} bytes"
                    ))
                    .into());
                }
                *crc = crc32_update(*crc, data);
                out.write_all(data)?;
                if !progress(*written) {
                    return Err(ZipError::BadData("stopped".into()).into());
                }
                Ok(())
            };
        match e.method {
            Method::Stored => loop {
                let n = input.read(&mut inbuf)?;
                if n == 0 {
                    break;
                }
                emit(&inbuf[..n], &mut written, &mut crc)?;
            },
            Method::Deflated => {
                use miniz_oxide::inflate::stream::{InflateState, inflate};
                use miniz_oxide::{DataFormat, MZError, MZFlush, MZStatus};
                let mut state = InflateState::new_boxed(DataFormat::Raw);
                let mut outbuf = vec![0u8; 64 * 1024];
                let mut have = 0usize;
                let mut pos = 0usize;
                let mut input_done = false;
                loop {
                    if pos == have && !input_done {
                        have = input.read(&mut inbuf)?;
                        pos = 0;
                        input_done = have == 0;
                    }
                    let r = inflate(&mut state, &inbuf[pos..have], &mut outbuf, MZFlush::None);
                    pos += r.bytes_consumed;
                    if r.bytes_written > 0 {
                        emit(&outbuf[..r.bytes_written], &mut written, &mut crc)?;
                    }
                    match r.status {
                        Ok(MZStatus::StreamEnd) => break,
                        Ok(_) | Err(MZError::Buf) => {
                            if r.bytes_consumed == 0
                                && r.bytes_written == 0
                                && pos == have
                                && input_done
                            {
                                return Err(ZipError::BadData(
                                    "deflate stream is truncated".into(),
                                )
                                .into());
                            }
                        }
                        Err(other) => {
                            return Err(ZipError::BadData(format!(
                                "deflate stream is invalid ({other:?})"
                            ))
                            .into());
                        }
                    }
                }
            }
        }
        if written != declared {
            return Err(ZipError::BadData(format!(
                "inflates to {written} bytes, but {declared} are declared"
            ))
            .into());
        }
        if crc != e.crc32 {
            return Err(ZipError::CrcMismatch {
                expected: e.crc32,
                actual: crc,
            }
            .into());
        }
        Ok(written)
    }
}

// ---------------------------------------------------------------------------------------
// Local headers (for checking what a writer produced)

/// A local file header and the data descriptor after the entry's data, as stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalHeader {
    pub version_needed: u16,
    pub flags: u16,
    pub method: u16,
    pub modified: DosDateTime,
    pub crc32: u32,
    pub compressed_size: u32,
    pub uncompressed_size: u32,
    pub name: Vec<u8>,
    pub extra: Vec<u8>,
    /// The 16-byte data descriptor with its signature (`crc32`, compressed size,
    /// uncompressed size) when flag bit 3 is set and one follows the data.
    pub data_descriptor: Option<(u32, u32, u32)>,
}

impl Archive<'_> {
    /// The local header of entry `index`, and its data descriptor when flag bit 3 is set.
    pub fn local_header(&self, index: usize) -> Result<LocalHeader, ZipError> {
        let located = self
            .entries
            .get(index)
            .ok_or_else(|| ZipError::Corrupt(format!("no entry {index}")))?;
        let start = to_usize(located.local_header_offset, "local header offset")?;
        let mut c = Cursor::at(self.data, start);
        if c.u32()? != SIG_LOCAL {
            return Err(ZipError::Corrupt(format!(
                "local header at offset {start} has a bad signature"
            )));
        }
        let version_needed = c.u16()?;
        let flags = c.u16()?;
        let method = c.u16()?;
        let time = c.u16()?;
        let date = c.u16()?;
        let crc32 = c.u32()?;
        let compressed_size = c.u32()?;
        let uncompressed_size = c.u32()?;
        let name_len = usize::from(c.u16()?);
        let extra_len = usize::from(c.u16()?);
        let name = c.bytes(name_len)?.to_vec();
        let extra = c.bytes(extra_len)?.to_vec();
        let mut data_descriptor = None;
        if flags & FLAG_DATA_DESCRIPTOR != 0 {
            let csize = to_usize(located.entry.compressed_size, "compressed size")?;
            let mut d = Cursor::at(self.data, c.pos.saturating_add(csize));
            if d.u32()? == SIG_DATA_DESCRIPTOR {
                data_descriptor = Some((d.u32()?, d.u32()?, d.u32()?));
            }
        }
        Ok(LocalHeader {
            version_needed,
            flags,
            method,
            modified: DosDateTime { date, time },
            crc32,
            compressed_size,
            uncompressed_size,
            name,
            extra,
            data_descriptor,
        })
    }
}

// ---------------------------------------------------------------------------------------
// Writing

const SIG_DATA_DESCRIPTOR: u32 = 0x0807_4b50;

/// "Version made by" of every entry QuaZip writes: Unix (3), zip specification 3.0 (30).
/// Seen in archives saved by XMALab 3 on Linux (QuaZip 1.4) and on macOS.
pub const QUAZIP_VERSION_MADE_BY: u16 = 0x031e;
/// "Version needed to extract" QuaZip writes for a deflated entry (2.0).
pub const QUAZIP_VERSION_NEEDED: u16 = 20;
/// External attributes QuaZip writes for a regular file with mode 0644 (`-rw-r--r--`),
/// which is what the C++ temporary files get under the usual umask.
pub const QUAZIP_EXTERNAL_ATTRIBUTES: u32 = 0o100_644 << 16;
/// Deflate level QuaZip uses (`Z_DEFAULT_COMPRESSION`, which zlib maps to 6).
pub const QUAZIP_DEFLATE_LEVEL: u8 = 6;

/// One entry to write: its stored name (already encoded), data and time.
#[derive(Clone, Copy, Debug)]
pub struct NewEntry<'a> {
    pub name: &'a [u8],
    pub data: &'a [u8],
    pub modified: DosDateTime,
}

/// zlib's text detection (`deflate.c`, `detect_data_type`), which minizip copies into the
/// internal attributes (`zipCloseFileInZipRaw64`: bit 0 set when `data_type == Z_ASCII`).
///
/// zlib decides on the literals of the first deflate block; this looks at the whole data.
/// The two differ only for data whose first non-text byte comes after the first block,
/// which no file XMALab writes has: its text files contain no control characters other
/// than line ends, and its images (TIFF, PNG, JPEG, BMP) start with binary headers.
#[must_use]
pub fn is_text_for_zlib(data: &[u8]) -> bool {
    // Bytes 0-6, 14-25 and 28-31 are "black-listed"; 9, 10, 13 and 32-255 "white-listed";
    // 7, 8, 11, 12, 26, 27 neither.
    const BLACK: u32 = 0xf3ff_c07f;
    let mut white = false;
    for &b in data {
        if b < 32 {
            if BLACK & (1 << b) != 0 {
                return false;
            }
            white |= matches!(b, 9 | 10 | 13);
        } else {
            white = true;
        }
    }
    white
}

fn put16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn fit32(v: usize, what: &str) -> Result<u32, ZipError> {
    u32::try_from(v)
        .ok()
        .filter(|&x| x != u32::MAX)
        .ok_or_else(|| ZipError::Unsupported(format!("{what} of {v} bytes needs zip64")))
}

fn fit16(v: usize, what: &str) -> Result<u16, ZipError> {
    u16::try_from(v).map_err(|_| ZipError::Unsupported(format!("{what} {v} is too long")))
}

/// Writes an archive as XMALab 3 writes a project (`ui/ProjectFileIO.cpp`,
/// `ProjectFileIO::zipFromFolderToFile`, through QuaZip 1.4 and its copy of minizip).
///
/// Every entry is deflated at level 6 and has the layout QuaZip gives it: version made by
/// [`QUAZIP_VERSION_MADE_BY`], version needed [`QUAZIP_VERSION_NEEDED`], flag bit 3 (a data
/// descriptor with its signature follows the data, while the local header also carries the
/// CRC-32 and sizes), no extra field, no entry comment, external attributes
/// [`QUAZIP_EXTERNAL_ATTRIBUTES`], internal attribute bit 0 when zlib would call the data
/// text ([`is_text_for_zlib`]). Entries are written in the order given; the archive comment
/// follows the end record. The compressed bytes are miniz_oxide's, not zlib's, so they
/// differ from QuaZip's; what an unzip program extracts is the same.
///
/// Entries of 4 GiB or more, and more than 65535 entries, would need zip64 and are refused.
///
/// Ported from QuaZip 1.4 `zip.c`, `zipOpenNewFileInZip4_64`, `zipWriteInFileInZip`,
/// `zipCloseFileInZipRaw64` and `zipClose`, as called by `QuaZipFile`.
pub fn write_archive(entries: &[NewEntry<'_>], comment: &[u8]) -> Result<Vec<u8>, ZipError> {
    let prepared = prepare_archive(entries, comment)?;
    let mut out = Vec::with_capacity(prepared.len());
    prepared
        .write_to(&mut out)
        .map_err(|e| ZipError::Corrupt(format!("writing to memory: {e}")))?;
    Ok(out)
}

/// An archive of [`write_archive`] with every entry compressed and every size checked,
/// ready to be written with [`PreparedArchive::write_to`]: all of [`write_archive`]'s
/// errors are known before the first byte is written.
pub struct PreparedArchive<'a> {
    entries: Vec<PreparedEntry<'a>>,
    comment: &'a [u8],
    n_entries: u16,
    comment_len: u16,
    cd_offset: u32,
    cd_size: u32,
    len: usize,
}

struct PreparedEntry<'a> {
    name: &'a [u8],
    name_len: u16,
    modified: DosDateTime,
    packed: Packed,
    csize: u32,
    usize_: u32,
    offset: u32,
}

/// What compressing one entry gives: the deflated bytes, the CRC-32 and zlib's text flag
/// of the uncompressed data.
struct Packed {
    compressed: Vec<u8>,
    crc: u32,
    text: bool,
}

fn pack(e: &NewEntry<'_>) -> Packed {
    let mut compressed = miniz_oxide::deflate::compress_to_vec(e.data, QUAZIP_DEFLATE_LEVEL);
    // The buffer grows by doubling; every entry is held until the archive is written.
    compressed.shrink_to_fit();
    Packed {
        compressed,
        crc: crc32(e.data),
        text: is_text_for_zlib(e.data),
    }
}

/// Below this many bytes of data in all, entries are compressed on the calling thread.
const PARALLEL_MIN_BYTES: usize = 1 << 20;

/// The most threads compressing at once. Each holds one entry's compressor and output
/// buffer beside the entries already compressed (which are the archive being built).
const MAX_WORKERS: usize = 8;

/// The number of compressing threads, decided once per process.
fn worker_count() -> usize {
    static WORKERS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WORKERS.get_or_init(|| {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(MAX_WORKERS)
    })
}

/// Every entry compressed, in the order given. Each entry is deflated alone, so the
/// result does not depend on the number of threads or on which thread takes which entry.
fn pack_all(entries: &[NewEntry<'_>], threads: usize) -> Vec<Packed> {
    pack_all_in(entries, threads, usize::MAX)
}

/// [`pack_all`], where starting a thread fails (as when the operating system refuses
/// one) after `max_spawns` threads have started; `usize::MAX` in the program, smaller
/// only in tests. The calling thread compresses beside the `threads - 1` workers, taking
/// entries from the same shared index, so a thread that cannot be started only leaves
/// its share to the threads that did start and to the caller: the result is the same.
fn pack_all_in(entries: &[NewEntry<'_>], threads: usize, max_spawns: usize) -> Vec<Packed> {
    let threads = threads.min(entries.len());
    if threads <= 1 {
        return entries.iter().map(pack).collect();
    }
    // Largest first, so that a large entry does not start last and run alone.
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(entries[i].data.len()));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut slots: Vec<Option<Packed>> = entries.iter().map(|_| None).collect();
    let work = || {
        let mut done = Vec::new();
        loop {
            let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(&i) = order.get(k) else { break };
            done.push((i, pack(&entries[i])));
        }
        done
    };
    std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(threads - 1);
        for started in 0..threads - 1 {
            if started >= max_spawns {
                break;
            }
            match std::thread::Builder::new().spawn_scoped(scope, work) {
                Ok(w) => workers.push(w),
                // The closure is gone, but its work is still in the shared index.
                Err(_) => break,
            }
        }
        // The calling thread is the last worker.
        let mine = work();
        for (i, p) in mine {
            slots[i] = Some(p);
        }
        for w in workers {
            match w.join() {
                Ok(done) => {
                    for (i, p) in done {
                        slots[i] = Some(p);
                    }
                }
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
    });
    slots
        .into_iter()
        .map(|p| p.unwrap_or_else(|| unreachable!("every entry is taken by a worker")))
        .collect()
}

/// Compresses the entries of an archive (on several threads for a large archive) and
/// checks every size, with the errors of [`write_archive`] in the order it meets them.
///
/// # Errors
/// As [`write_archive`].
pub fn prepare_archive<'a>(
    entries: &[NewEntry<'a>],
    comment: &'a [u8],
) -> Result<PreparedArchive<'a>, ZipError> {
    let total: usize = entries.iter().map(|e| e.data.len()).sum();
    let threads = if total < PARALLEL_MIN_BYTES {
        1
    } else {
        worker_count()
    };
    prepare_archive_with(entries, comment, threads)
}

/// [`prepare_archive`] on at most `threads` threads (1: on the calling thread).
///
/// # Errors
/// As [`write_archive`].
pub fn prepare_archive_with<'a>(
    entries: &[NewEntry<'a>],
    comment: &'a [u8],
    threads: usize,
) -> Result<PreparedArchive<'a>, ZipError> {
    let n_entries = fit16(entries.len(), "entry count")?;
    let comment_len = fit16(comment.len(), "archive comment of length")?;
    let packed = pack_all(entries, threads);
    let mut prepared = Vec::with_capacity(entries.len());
    let mut offset = 0usize;
    let mut central = 0usize;
    for (e, packed) in entries.iter().zip(packed) {
        let name_len = fit16(e.name.len(), "entry name of length")?;
        let csize = fit32(packed.compressed.len(), "compressed entry")?;
        let usize_ = fit32(e.data.len(), "entry")?;
        let entry_offset = fit32(offset, "archive")?;
        offset += LOCAL_LEN + e.name.len() + packed.compressed.len() + 16;
        central += CENTRAL_LEN + e.name.len();
        prepared.push(PreparedEntry {
            name: e.name,
            name_len,
            modified: e.modified,
            packed,
            csize,
            usize_,
            offset: entry_offset,
        });
    }
    let cd_offset = fit32(offset, "archive")?;
    let cd_size = fit32(central, "central directory")?;
    Ok(PreparedArchive {
        entries: prepared,
        comment,
        n_entries,
        comment_len,
        cd_offset,
        cd_size,
        len: offset + central + EOCD_LEN + comment.len(),
    })
}

impl PreparedArchive<'_> {
    /// The length of the archive in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True for an archive of no bytes (never: the end record is always written).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Writes the archive: the bytes [`write_archive`] returns.
    ///
    /// # Errors
    /// The writer's errors.
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        let flags = FLAG_DATA_DESCRIPTOR;
        let method = Method::Deflated.code();
        let mut head = Vec::with_capacity(LOCAL_LEN + 0xffff);
        for e in &self.entries {
            head.clear();
            put32(&mut head, SIG_LOCAL);
            put16(&mut head, QUAZIP_VERSION_NEEDED);
            put16(&mut head, flags);
            put16(&mut head, method);
            put16(&mut head, e.modified.time);
            put16(&mut head, e.modified.date);
            put32(&mut head, e.packed.crc);
            put32(&mut head, e.csize);
            put32(&mut head, e.usize_);
            put16(&mut head, e.name_len);
            put16(&mut head, 0);
            head.extend_from_slice(e.name);
            w.write_all(&head)?;
            w.write_all(&e.packed.compressed)?;
            head.clear();
            put32(&mut head, SIG_DATA_DESCRIPTOR);
            put32(&mut head, e.packed.crc);
            put32(&mut head, e.csize);
            put32(&mut head, e.usize_);
            w.write_all(&head)?;
        }
        let mut central = Vec::with_capacity(self.cd_size as usize + EOCD_LEN + self.comment.len());
        for e in &self.entries {
            put32(&mut central, SIG_CENTRAL);
            put16(&mut central, QUAZIP_VERSION_MADE_BY);
            put16(&mut central, QUAZIP_VERSION_NEEDED);
            put16(&mut central, flags);
            put16(&mut central, method);
            put16(&mut central, e.modified.time);
            put16(&mut central, e.modified.date);
            put32(&mut central, e.packed.crc);
            put32(&mut central, e.csize);
            put32(&mut central, e.usize_);
            put16(&mut central, e.name_len);
            put16(&mut central, 0); // extra
            put16(&mut central, 0); // comment
            put16(&mut central, 0); // disk
            put16(&mut central, u16::from(e.packed.text));
            put32(&mut central, QUAZIP_EXTERNAL_ATTRIBUTES);
            put32(&mut central, e.offset);
            central.extend_from_slice(e.name);
        }
        put32(&mut central, SIG_EOCD);
        put16(&mut central, 0);
        put16(&mut central, 0);
        put16(&mut central, self.n_entries);
        put16(&mut central, self.n_entries);
        put32(&mut central, self.cd_size);
        put32(&mut central, self.cd_offset);
        put16(&mut central, self.comment_len);
        central.extend_from_slice(self.comment);
        w.write_all(&central)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn written_archive_reads_back() {
        let time = DosDateTime::from_fields(2026, 11, 13, 22, 24, 26).expect("valid");
        let names: [&[u8]; 3] = [b"a.txt", b"dir/b.bin", b"empty"];
        let datas: [&[u8]; 3] = [b"hello\nworld\n", &[0u8, 1, 2, 3, 255], b""];
        let entries: Vec<NewEntry<'_>> = names
            .iter()
            .zip(datas)
            .map(|(&name, data)| NewEntry {
                name,
                data,
                modified: time,
            })
            .collect();
        let bytes = write_archive(&entries, b"XROMM Project File").expect("writes");
        let archive = Archive::new(&bytes).expect("reads");
        assert_eq!(archive.comment(), b"XROMM Project File");
        assert_eq!(archive.len(), 3);
        for (i, e) in archive.entries().enumerate() {
            assert_eq!(e.name, names[i]);
            assert_eq!(e.flags, FLAG_DATA_DESCRIPTOR);
            assert_eq!(e.version_made_by, 0x031e);
            assert_eq!(e.version_needed, 20);
            assert_eq!(e.external_attributes, 0x81a4_0000);
            assert_eq!(e.modified, time);
            assert_eq!(archive.read(i).expect("data"), datas[i]);
            let local = archive.local_header(i).expect("local header");
            assert_eq!(local.flags, e.flags);
            assert_eq!(local.crc32, e.crc32);
            assert_eq!(
                local.data_descriptor,
                Some((
                    e.crc32,
                    e.compressed_size as u32,
                    e.uncompressed_size as u32
                ))
            );
        }
        let internal: Vec<u16> = archive.entries().map(|e| e.internal_attributes).collect();
        assert_eq!(internal, [1, 0, 0]);
    }

    #[test]
    fn zlib_text_detection() {
        assert!(is_text_for_zlib(b"1, 2\n"));
        assert!(is_text_for_zlib("é\r\n".as_bytes()));
        assert!(!is_text_for_zlib(b""));
        assert!(!is_text_for_zlib(b"\x07\x08"));
        assert!(!is_text_for_zlib(b"abc\x00"));
        assert!(!is_text_for_zlib(b"II*\x00"));
    }

    #[test]
    fn crc32_check_value() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn cp866_round_trip() {
        let all: Vec<u8> = (0..=255u8).collect();
        let text = decode_cp866(&all);
        assert_eq!(text.chars().count(), 256);
        assert_eq!(encode_cp866(&text), all);
        assert_eq!(
            decode_cp866(&[0x8f, 0xe0, 0xa8, 0xa2, 0xa5, 0xe2]),
            "Привет"
        );
        assert_eq!(encode_cp866("Kamera ü"), b"Kamera ?".to_vec());
    }

    #[test]
    fn dos_date_time_fields() {
        let t = DosDateTime::from_fields(2026, 3, 17, 11, 13, 21).expect("valid");
        assert_eq!(t.fields(), (2026, 3, 17, 11, 13, 20));
        assert!(DosDateTime::from_fields(1979, 1, 1, 0, 0, 0).is_none());
    }

    #[test]
    fn rejects_short_and_garbage_input() {
        assert_eq!(Archive::new(b"").unwrap_err(), ZipError::NotAZip);
        assert_eq!(Archive::new(&[0u8; 100]).unwrap_err(), ZipError::NotAZip);
    }

    #[test]
    fn inflate_rejects_wrong_lengths() {
        let data = miniz_oxide::deflate::compress_to_vec(b"hello hello hello", 6);
        assert_eq!(inflate(&data, 17).expect("inflates"), b"hello hello hello");
        assert!(matches!(inflate(&data, 16), Err(ZipError::BadData(_))));
        assert!(matches!(inflate(&data, 18), Err(ZipError::BadData(_))));
        assert!(matches!(
            inflate(&data[..data.len() - 2], 17),
            Err(ZipError::BadData(_))
        ));
        // A huge declared size allocates only what the data produces.
        assert!(matches!(
            inflate(&data, usize::MAX),
            Err(ZipError::BadData(_))
        ));
    }

    /// Review 13c: the compression of an archive's entries gives the same entries, in the
    /// same order, when starting a thread fails at once, after one worker or after
    /// several, as on one thread and as the compression before review 13 (`compress_to_vec`
    /// and the CRC-32 of each entry in turn; `review13_archive_same_bytes` in
    /// `tests/review13_io.rs` compares the one-thread archive byte for byte with the old
    /// writer).
    #[test]
    fn review13c_pack_all_when_threads_cannot_start() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut datas: Vec<Vec<u8>> = vec![
            Vec::new(),
            b"hello\n".to_vec(),
            vec![0u8; 1 << 20],
            (0..300_000).map(|_| next() as u8).collect(),
            (0..200_000).map(|i| b"0123456789,\n"[i % 12]).collect(),
        ];
        for i in 0..30 {
            let n = usize::try_from(next() % 5000).expect("small");
            datas.push((0..n).map(|_| (next() % 3) as u8 + b'a').collect());
            if i % 7 == 0 {
                datas.push(Vec::new());
            }
        }
        let time = DosDateTime::from_fields(2026, 11, 21, 10, 0, 0).expect("valid");
        let entries: Vec<NewEntry<'_>> = datas
            .iter()
            .map(|data| NewEntry {
                name: b"e",
                data,
                modified: time,
            })
            .collect();
        let as_tuples = |p: Vec<Packed>| -> Vec<(Vec<u8>, u32, bool)> {
            p.into_iter()
                .map(|p| (p.compressed, p.crc, p.text))
                .collect()
        };
        // The old compression, one entry after another.
        let old: Vec<(Vec<u8>, u32, bool)> = datas
            .iter()
            .map(|d| {
                (
                    miniz_oxide::deflate::compress_to_vec(d, QUAZIP_DEFLATE_LEVEL),
                    crc32(d),
                    is_text_for_zlib(d),
                )
            })
            .collect();
        for set in [&entries[..0], &entries[..1], &entries[..3], &entries[..]] {
            let want = &old[..set.len()];
            assert_eq!(as_tuples(pack_all(set, 1)), want);
            for threads in [2, 3, 8] {
                for max_spawns in [0, 1, 2, 5, usize::MAX] {
                    assert_eq!(
                        as_tuples(pack_all_in(set, threads, max_spawns)),
                        want,
                        "{} entries, {threads} threads, failing after {max_spawns}",
                        set.len()
                    );
                }
            }
        }
        // Through the whole archive: failing at once gives the bytes of one thread.
        let one = prepare_archive_with(&entries, b"c", 1).expect("prepares");
        let mut want = Vec::new();
        one.write_to(&mut want).expect("writes to memory");
        assert_eq!(write_archive(&entries, b"c").expect("writes"), want);
    }
}
