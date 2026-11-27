//! How the C++ XMALab reads text, reproduced site by site.
//!
//! The project loader parses numbers four different ways, and they disagree on malformed
//! input. Each has one function here:
//!
//! | C++ | Where | Function |
//! |---|---|---|
//! | `littleHelper::safeGetline` in a `while (!...eof())` loop | every CSV and text file | [`LineReader`], [`lines`] |
//! | `for (double v; in >> v; littleHelper::comma(in))` | camera, calibration, marker CSVs | [`stream_values`]; the default mode's reader also reads `nan` and `inf` ([`stream_values_nonfinite`], ledger F49) |
//! | `for (int v; in >> v; littleHelper::comma(in))` | `RigidBodies.txt` | [`stream_ints`] |
//! | `QString::toDouble`, `QString::toInt` | XML attributes, `split(",")` CSVs | [`qt_to_double`], [`qt_to_int`] |
//! | `sscanf("%i %20[0-9a-zA-Z ]s")` | calibration-object `.ref` file | [`scan_reference_line`] |
//! | `QColor(QString)` | event and rigid-body colours | [`qt_color`] |
//!
//! Stream extraction follows libstdc++ (`num_get::_M_extract_float` and `_M_extract_int`,
//! then `strtod` in the "C" locale), since the reference dumps come from a Linux build.

use xma_model::Rgb;

/// Reads lines as `littleHelper::safeGetline` does (`core/HelperFunctions.cpp:107`).
///
/// Ported from core/HelperFunctions.cpp, littleHelper::safeGetline.
///
/// A line ends at `\n`, `\r\n` or a lone `\r`. At the end of the data a non-empty last line
/// without terminator is returned; the call after it (or a call at the end with nothing
/// read) reports end of file and an empty line, which is what the C++ leaves in its
/// `line` variable.
#[derive(Clone, Debug)]
pub struct LineReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> LineReader<'a> {
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// One `safeGetline` call: the line, and whether the stream reached end of file
    /// (the C++ `eof()` after the call).
    pub fn next_line(&mut self) -> (&'a [u8], bool) {
        let start = self.pos;
        while self.pos < self.data.len() {
            let c = self.data[self.pos];
            self.pos += 1;
            match c {
                b'\n' => return (&self.data[start..self.pos - 1], false),
                b'\r' => {
                    let end = self.pos - 1;
                    if self.data.get(self.pos) == Some(&b'\n') {
                        self.pos += 1;
                    }
                    return (&self.data[start..end], false);
                }
                _ => {}
            }
        }
        let line = &self.data[start..];
        (line, line.is_empty())
    }
}

/// All lines a `while (!safeGetline(fin, line).eof())` loop sees.
#[must_use]
pub fn lines(data: &[u8]) -> Vec<&[u8]> {
    let mut r = LineReader::new(data);
    let mut out = Vec::new();
    loop {
        let (line, eof) = r.next_line();
        if eof {
            return out;
        }
        out.push(line);
    }
}

/// `QString::fromStdString(line)`: UTF-8, invalid sequences replaced.
#[must_use]
pub fn qstring(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// `isspace` in the "C" locale.
fn c_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// One `in >> double` (libstdc++ `num_get::_M_extract_float`, then `strtod`). Returns the
/// value, or `None` when the extraction sets failbit (no number, a number `strtod` does not
/// consume completely such as `1e`, or overflow).
fn extract_f64(s: &[u8], pos: &mut usize) -> Option<f64> {
    while *pos < s.len() && c_space(s[*pos]) {
        *pos += 1;
    }
    let start = *pos;
    let mut i = start;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        i += 1;
    }
    let (mut found_mantissa, mut found_dec, mut found_sci) = (false, false, false);
    while i < s.len() {
        let c = s[i];
        if c.is_ascii_digit() {
            found_mantissa = true;
        } else if c == b'.' && !found_dec && !found_sci {
            found_dec = true;
        } else if (c == b'e' || c == b'E') && !found_sci && found_mantissa {
            found_sci = true;
            if i + 1 < s.len() && (s[i + 1] == b'+' || s[i + 1] == b'-') {
                i += 1;
            }
        } else {
            break;
        }
        i += 1;
    }
    *pos = i;
    // `strtod` must consume the whole collected text; the grammar above only lets digits,
    // one point and one exponent (`e` or `E`) through, so Rust's parser agrees with it here.
    // The text is ASCII, so it is read in place.
    let v: f64 = std::str::from_utf8(&s[start..i]).ok()?.parse().ok()?;
    if v.is_infinite() {
        // libstdc++ stores the largest finite value and sets failbit.
        return None;
    }
    Some(v)
}

/// The values a `for (double value; in >> value; littleHelper::comma(in))` loop pushes for
/// one line. `littleHelper::comma` (`core/HelperFunctions.cpp:144`) skips white space and
/// requires a comma, so the loop stops at the first value that does not parse or the first
/// separator that is not a comma.
#[must_use]
pub fn stream_values(line: &[u8]) -> Vec<f64> {
    let mut pos = 0;
    let mut out = Vec::new();
    while let Some(v) = extract_f64(line, &mut pos) {
        out.push(v);
        while pos < line.len() && c_space(line[pos]) {
            pos += 1;
        }
        if line.get(pos) == Some(&b',') {
            pos += 1;
        } else {
            break;
        }
    }
    out
}

/// [`stream_values`] that also reads the words glibc's `printf` writes for non-finite
/// values, which `in >> double` does not: `nan`, `-nan`, `nan(...)`, `inf`, `-inf`,
/// `infinity` (any case, an optional sign), and a number too large for a `double`, as NaN
/// and the infinities, so that a row XMALab wrote with such a value is read whole. For the
/// default mode's reader (parity ledger F49), which then stores each such value as not set
/// (`crate::finite`) with a warning.
#[must_use]
pub fn stream_values_nonfinite(line: &[u8]) -> Vec<f64> {
    let mut pos = 0;
    let mut out = Vec::new();
    loop {
        let start = pos;
        let v = match extract_f64(line, &mut pos) {
            Some(v) => v,
            None => {
                pos = start;
                match extract_nonfinite(line, &mut pos) {
                    Some(v) => v,
                    None => break,
                }
            }
        };
        out.push(v);
        while pos < line.len() && c_space(line[pos]) {
            pos += 1;
        }
        if line.get(pos) == Some(&b',') {
            pos += 1;
        } else {
            break;
        }
    }
    out
}

/// A non-finite word or an overflowing number at `pos` (white space skipped first), for
/// [`stream_values_nonfinite`].
fn extract_nonfinite(s: &[u8], pos: &mut usize) -> Option<f64> {
    let mut i = *pos;
    while i < s.len() && c_space(s[i]) {
        i += 1;
    }
    let start = i;
    let negative = s.get(i) == Some(&b'-');
    if matches!(s.get(i), Some(b'+' | b'-')) {
        i += 1;
    }
    let rest = &s[i..];
    let starts = |w: &[u8]| rest.len() >= w.len() && rest[..w.len()].eq_ignore_ascii_case(w);
    let sign = if negative { -1.0 } else { 1.0 };
    if starts(b"nan") {
        i += 3;
        if s.get(i) == Some(&b'(')
            && let Some(close) = s[i..].iter().position(|&c| c == b')')
        {
            i += close + 1;
        }
        *pos = i;
        return Some(f64::NAN);
    }
    if starts(b"infinity") {
        *pos = i + 8;
        return Some(sign * f64::INFINITY);
    }
    if starts(b"inf") {
        *pos = i + 3;
        return Some(sign * f64::INFINITY);
    }
    // A number `strtod` reads as an overflow (libstdc++ sets failbit).
    let mut end = start;
    let _ = extract_f64(s, &mut end);
    let text = std::str::from_utf8(&s[start..end]).ok()?;
    let v: f64 = text.parse().ok()?;
    if v.is_infinite() {
        *pos = end;
        return Some(v);
    }
    None
}

/// One `in >> int` (libstdc++: a `long` through `num_get`, then range-checked to `int`).
fn extract_int(s: &[u8], pos: &mut usize) -> Option<i32> {
    while *pos < s.len() && c_space(s[*pos]) {
        *pos += 1;
    }
    let mut i = *pos;
    let negative = match s.get(i) {
        Some(b'-') => {
            i += 1;
            true
        }
        Some(b'+') => {
            i += 1;
            false
        }
        _ => false,
    };
    let digits_start = i;
    let mut value: i64 = 0;
    let mut overflow = false;
    while i < s.len() && s[i].is_ascii_digit() {
        let d = i64::from(s[i] - b'0');
        match value.checked_mul(10).and_then(|v| v.checked_add(d)) {
            Some(v) => value = v,
            None => overflow = true,
        }
        i += 1;
    }
    *pos = i;
    if i == digits_start || overflow {
        return None;
    }
    let v = if negative { -value } else { value };
    i32::try_from(v).ok()
}

/// The integers a `for (int value; in >> value; littleHelper::comma(in))` loop pushes.
#[must_use]
pub fn stream_ints(line: &[u8]) -> Vec<i32> {
    let mut pos = 0;
    let mut out = Vec::new();
    while let Some(v) = extract_int(line, &mut pos) {
        out.push(v);
        while pos < line.len() && c_space(line[pos]) {
            pos += 1;
        }
        if line.get(pos) == Some(&b',') {
            pos += 1;
        } else {
            break;
        }
    }
    out
}

/// `(int)` of a `double`, as `std::vector<int>::push_back(double)` converts. Values out of
/// the `int` range are undefined in C++; here they saturate.
#[must_use]
pub fn c_int(v: f64) -> i32 {
    v as i32
}

/// `QString::toDouble()`: leading and trailing white space ignored, `nan`, `inf` and
/// `infinity` accepted (any case; a sign only before `inf`), 0 for anything else that is
/// not a complete decimal number.
#[must_use]
pub fn qt_to_double(text: &str) -> f64 {
    let t = text.trim();
    if t.eq_ignore_ascii_case("nan") {
        return f64::NAN;
    }
    let (sign, rest) = match t.as_bytes().first() {
        Some(b'-') => (-1.0, &t[1..]),
        Some(b'+') => (1.0, &t[1..]),
        _ => (1.0, t),
    };
    if rest.eq_ignore_ascii_case("inf") || rest.eq_ignore_ascii_case("infinity") {
        return sign * f64::INFINITY;
    }
    if t.is_empty()
        || !t
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-'))
    {
        return 0.0;
    }
    t.parse().unwrap_or(0.0)
}

/// `QString::toInt()`: base 10, white space around ignored, an optional sign, 0 when the
/// text is not an integer or does not fit an `int`.
#[must_use]
pub fn qt_to_int(text: &str) -> i32 {
    let t = text.trim();
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return 0;
    }
    t.parse().unwrap_or(0)
}

/// `QStringList QString::split(",")`.
#[must_use]
pub fn split_commas(text: &str) -> Vec<&str> {
    text.split(',').collect()
}

/// `sscanf(line, "%i %20[0-9a-zA-Z ]s", &id, str) == 2`, as
/// `core/CalibrationObject.cpp:136` reads a reference line. Returns the id and name when
/// both conversions succeed.
///
/// `%i` accepts a sign and a `0x` (hexadecimal) or `0` (octal) prefix, after optional
/// white space; the name is 1 to 20 characters from letters, digits and spaces.
#[must_use]
pub fn scan_reference_line(line: &[u8]) -> Option<(i32, String)> {
    let mut i = 0;
    while i < line.len() && c_space(line[i]) {
        i += 1;
    }
    let negative = match line.get(i) {
        Some(b'-') => {
            i += 1;
            true
        }
        Some(b'+') => {
            i += 1;
            false
        }
        _ => false,
    };
    let (radix, digits_from) = if line.get(i) == Some(&b'0')
        && matches!(line.get(i + 1), Some(b'x' | b'X'))
        && line.get(i + 2).is_some_and(u8::is_ascii_hexdigit)
    {
        (16, i + 2)
    } else if line.get(i) == Some(&b'0') {
        (8, i)
    } else {
        (10, i)
    };
    let mut j = digits_from;
    // glibc's strtol saturates to `long`; the stored `int` is the low 32 bits.
    let mut value: i128 = 0;
    while j < line.len() {
        let Some(d) = char::from(line[j]).to_digit(radix) else {
            break;
        };
        value = (value * i128::from(radix) + i128::from(d)).min(i128::from(i64::MAX) + 1);
        j += 1;
    }
    if j == digits_from {
        return None;
    }
    let long = if negative {
        (-value).max(i128::from(i64::MIN))
    } else {
        value.min(i128::from(i64::MAX))
    } as i64;
    let id = long as i32;
    while j < line.len() && c_space(line[j]) {
        j += 1;
    }
    let start = j;
    while j < line.len() && j - start < 20 && (line[j].is_ascii_alphanumeric() || line[j] == b' ') {
        j += 1;
    }
    if j == start {
        return None;
    }
    Some((id, String::from_utf8_lossy(&line[start..j]).into_owned()))
}

/// `QColor(QString)` followed by `red()`, `green()`, `blue()`; see [`qt_color_rgba`].
/// Returns `None` where Qt builds an invalid colour (whose components read as 0, 0, 0).
/// The alpha channel is dropped: the model holds RGB only, and XMALab saves colours with
/// `QColor::name()`, which drops it too.
#[must_use]
pub fn qt_color(text: &str) -> Option<Rgb> {
    qt_color_rgba(text).map(|[r, g, b, _]| Rgb::new(r, g, b))
}

/// `QColor(QString)` as Qt 6 parses it (`QColor::fromString`), as red, green, blue and
/// alpha; `None` for an invalid colour.
///
/// Ported from Qt 6 qtbase, src/gui/painting/qcolor.cpp (`QColor::fromString`,
/// `get_hex_rgb`, `get_named_rgb`, `get_named_rgb_no_space`) and qcolor_p.h / qrgba64.h
/// for the 8-bit components of 12- and 16-bit hexadecimal channels. Verified against Qt
/// 6.11.2 for every name and a list of edge cases (`tests/data/qt_colors.tsv`).
///
/// - Text starting with `#` is hexadecimal: `#rgb`, `#rrggbb`, `#aarrggbb`, `#rrrgggbbb`,
///   `#rrrrggggbbbb`, nothing else; no surrounding white space is allowed.
/// - Otherwise it is a name: at most 255 UTF-16 code units; spaces and tabs (only those)
///   are removed anywhere; a character beyond Latin-1 becomes NUL in Qt's buffer, which
///   ends the name there; the rest is compared without regard to ASCII case with
///   `transparent` (alpha 0) and the 147 SVG colour names.
#[must_use]
pub fn qt_color_rgba(text: &str) -> Option<[u8; 4]> {
    if text.is_empty() {
        return None;
    }
    if let Some(hex) = text.strip_prefix('#') {
        return qt_hex_rgba(hex);
    }
    if text.encode_utf16().count() > 255 {
        return None;
    }
    let mut name = Vec::new();
    for c in text.chars() {
        match c {
            ' ' | '\t' => {}
            c if u32::from(c) > 0xff => break,
            c => name.push((u32::from(c) as u8).to_ascii_lowercase()),
        }
    }
    if name == b"transparent" {
        return Some([0, 0, 0, 0]);
    }
    let i = SVG_COLORS
        .binary_search_by(|(n, ..)| n.as_bytes().cmp(&name))
        .ok()?;
    let (_, r, g, b) = SVG_COLORS[i];
    Some([r, g, b, 255])
}

/// Ported from qcolor.cpp, `get_hex_rgb`, and the conversion to 8 bits that `red()` and
/// friends apply (`qt_div_257` of the 16-bit channel).
fn qt_hex_rgba(hex: &str) -> Option<[u8; 4]> {
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let field = |from: usize, len: usize| u32::from_str_radix(&hex[from..from + len], 16).ok();
    // A channel of `bits` bits widened to 16 bits as Qt does, then back to 8 with
    // `qt_div_257`, `(x - (x >> 8) + 0x80) >> 8`.
    let to8 = |v: u32, bits: u32| -> u8 {
        let v16 = match bits {
            4 => v * 0x1111,
            8 => v * 0x101,
            12 => (v << 4) | (v >> 8),
            _ => v,
        };
        ((v16 - (v16 >> 8) + 0x80) >> 8) as u8
    };
    let (a, r, g, b, bits) = match hex.len() {
        3 => (0xf, field(0, 1)?, field(1, 1)?, field(2, 1)?, 4),
        6 => (0xff, field(0, 2)?, field(2, 2)?, field(4, 2)?, 8),
        8 => (field(0, 2)?, field(2, 2)?, field(4, 2)?, field(6, 2)?, 8),
        9 => (0xfff, field(0, 3)?, field(3, 3)?, field(6, 3)?, 12),
        12 => (0xffff, field(0, 4)?, field(4, 4)?, field(8, 4)?, 16),
        _ => return None,
    };
    Some([to8(r, bits), to8(g, bits), to8(b, bits), to8(a, bits)])
}

/// The SVG colour names Qt knows, sorted, with their RGB values (qcolor.cpp, `rgbTbl`;
/// `transparent` is handled separately). Generated from `QColor::colorNames()` of Qt
/// 6.11.2; `tests/data/qt_colors.tsv` holds the source.
const SVG_COLORS: &[(&str, u8, u8, u8)] = &[
    ("aliceblue", 240, 248, 255),
    ("antiquewhite", 250, 235, 215),
    ("aqua", 0, 255, 255),
    ("aquamarine", 127, 255, 212),
    ("azure", 240, 255, 255),
    ("beige", 245, 245, 220),
    ("bisque", 255, 228, 196),
    ("black", 0, 0, 0),
    ("blanchedalmond", 255, 235, 205),
    ("blue", 0, 0, 255),
    ("blueviolet", 138, 43, 226),
    ("brown", 165, 42, 42),
    ("burlywood", 222, 184, 135),
    ("cadetblue", 95, 158, 160),
    ("chartreuse", 127, 255, 0),
    ("chocolate", 210, 105, 30),
    ("coral", 255, 127, 80),
    ("cornflowerblue", 100, 149, 237),
    ("cornsilk", 255, 248, 220),
    ("crimson", 220, 20, 60),
    ("cyan", 0, 255, 255),
    ("darkblue", 0, 0, 139),
    ("darkcyan", 0, 139, 139),
    ("darkgoldenrod", 184, 134, 11),
    ("darkgray", 169, 169, 169),
    ("darkgreen", 0, 100, 0),
    ("darkgrey", 169, 169, 169),
    ("darkkhaki", 189, 183, 107),
    ("darkmagenta", 139, 0, 139),
    ("darkolivegreen", 85, 107, 47),
    ("darkorange", 255, 140, 0),
    ("darkorchid", 153, 50, 204),
    ("darkred", 139, 0, 0),
    ("darksalmon", 233, 150, 122),
    ("darkseagreen", 143, 188, 143),
    ("darkslateblue", 72, 61, 139),
    ("darkslategray", 47, 79, 79),
    ("darkslategrey", 47, 79, 79),
    ("darkturquoise", 0, 206, 209),
    ("darkviolet", 148, 0, 211),
    ("deeppink", 255, 20, 147),
    ("deepskyblue", 0, 191, 255),
    ("dimgray", 105, 105, 105),
    ("dimgrey", 105, 105, 105),
    ("dodgerblue", 30, 144, 255),
    ("firebrick", 178, 34, 34),
    ("floralwhite", 255, 250, 240),
    ("forestgreen", 34, 139, 34),
    ("fuchsia", 255, 0, 255),
    ("gainsboro", 220, 220, 220),
    ("ghostwhite", 248, 248, 255),
    ("gold", 255, 215, 0),
    ("goldenrod", 218, 165, 32),
    ("gray", 128, 128, 128),
    ("green", 0, 128, 0),
    ("greenyellow", 173, 255, 47),
    ("grey", 128, 128, 128),
    ("honeydew", 240, 255, 240),
    ("hotpink", 255, 105, 180),
    ("indianred", 205, 92, 92),
    ("indigo", 75, 0, 130),
    ("ivory", 255, 255, 240),
    ("khaki", 240, 230, 140),
    ("lavender", 230, 230, 250),
    ("lavenderblush", 255, 240, 245),
    ("lawngreen", 124, 252, 0),
    ("lemonchiffon", 255, 250, 205),
    ("lightblue", 173, 216, 230),
    ("lightcoral", 240, 128, 128),
    ("lightcyan", 224, 255, 255),
    ("lightgoldenrodyellow", 250, 250, 210),
    ("lightgray", 211, 211, 211),
    ("lightgreen", 144, 238, 144),
    ("lightgrey", 211, 211, 211),
    ("lightpink", 255, 182, 193),
    ("lightsalmon", 255, 160, 122),
    ("lightseagreen", 32, 178, 170),
    ("lightskyblue", 135, 206, 250),
    ("lightslategray", 119, 136, 153),
    ("lightslategrey", 119, 136, 153),
    ("lightsteelblue", 176, 196, 222),
    ("lightyellow", 255, 255, 224),
    ("lime", 0, 255, 0),
    ("limegreen", 50, 205, 50),
    ("linen", 250, 240, 230),
    ("magenta", 255, 0, 255),
    ("maroon", 128, 0, 0),
    ("mediumaquamarine", 102, 205, 170),
    ("mediumblue", 0, 0, 205),
    ("mediumorchid", 186, 85, 211),
    ("mediumpurple", 147, 112, 219),
    ("mediumseagreen", 60, 179, 113),
    ("mediumslateblue", 123, 104, 238),
    ("mediumspringgreen", 0, 250, 154),
    ("mediumturquoise", 72, 209, 204),
    ("mediumvioletred", 199, 21, 133),
    ("midnightblue", 25, 25, 112),
    ("mintcream", 245, 255, 250),
    ("mistyrose", 255, 228, 225),
    ("moccasin", 255, 228, 181),
    ("navajowhite", 255, 222, 173),
    ("navy", 0, 0, 128),
    ("oldlace", 253, 245, 230),
    ("olive", 128, 128, 0),
    ("olivedrab", 107, 142, 35),
    ("orange", 255, 165, 0),
    ("orangered", 255, 69, 0),
    ("orchid", 218, 112, 214),
    ("palegoldenrod", 238, 232, 170),
    ("palegreen", 152, 251, 152),
    ("paleturquoise", 175, 238, 238),
    ("palevioletred", 219, 112, 147),
    ("papayawhip", 255, 239, 213),
    ("peachpuff", 255, 218, 185),
    ("peru", 205, 133, 63),
    ("pink", 255, 192, 203),
    ("plum", 221, 160, 221),
    ("powderblue", 176, 224, 230),
    ("purple", 128, 0, 128),
    ("red", 255, 0, 0),
    ("rosybrown", 188, 143, 143),
    ("royalblue", 65, 105, 225),
    ("saddlebrown", 139, 69, 19),
    ("salmon", 250, 128, 114),
    ("sandybrown", 244, 164, 96),
    ("seagreen", 46, 139, 87),
    ("seashell", 255, 245, 238),
    ("sienna", 160, 82, 45),
    ("silver", 192, 192, 192),
    ("skyblue", 135, 206, 235),
    ("slateblue", 106, 90, 205),
    ("slategray", 112, 128, 144),
    ("slategrey", 112, 128, 144),
    ("snow", 255, 250, 250),
    ("springgreen", 0, 255, 127),
    ("steelblue", 70, 130, 180),
    ("tan", 210, 180, 140),
    ("teal", 0, 128, 128),
    ("thistle", 216, 191, 216),
    ("tomato", 255, 99, 71),
    ("turquoise", 64, 224, 208),
    ("violet", 238, 130, 238),
    ("wheat", 245, 222, 179),
    ("white", 255, 255, 255),
    ("whitesmoke", 245, 245, 245),
    ("yellow", 255, 255, 0),
    ("yellowgreen", 154, 205, 50),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// `extract_f64` as it was before review 13, collecting the number into a `String`.
    fn old_extract_f64(s: &[u8], pos: &mut usize) -> Option<f64> {
        while *pos < s.len() && c_space(s[*pos]) {
            *pos += 1;
        }
        let mut text = String::new();
        let mut i = *pos;
        if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
            text.push(char::from(s[i]));
            i += 1;
        }
        let (mut found_mantissa, mut found_dec, mut found_sci) = (false, false, false);
        while i < s.len() {
            let c = s[i];
            if c.is_ascii_digit() {
                text.push(char::from(c));
                found_mantissa = true;
            } else if c == b'.' && !found_dec && !found_sci {
                text.push('.');
                found_dec = true;
            } else if (c == b'e' || c == b'E') && !found_sci && found_mantissa {
                text.push('e');
                found_sci = true;
                if i + 1 < s.len() && (s[i + 1] == b'+' || s[i + 1] == b'-') {
                    i += 1;
                    text.push(char::from(s[i]));
                }
            } else {
                break;
            }
            i += 1;
        }
        *pos = i;
        let v: f64 = text.parse().ok()?;
        if v.is_infinite() {
            return None;
        }
        Some(v)
    }

    /// Every extraction along a line, old and new: value bits, failure, and end position.
    fn same_along(line: &[u8]) {
        for start in 0..=line.len() {
            let (mut a, mut b) = (start, start);
            let new = extract_f64(line, &mut a);
            let old = old_extract_f64(line, &mut b);
            assert_eq!(
                new.map(f64::to_bits),
                old.map(f64::to_bits),
                "{:?} from {start}",
                String::from_utf8_lossy(line)
            );
            assert_eq!(a, b, "{:?} from {start}", String::from_utf8_lossy(line));
        }
    }

    /// `extract_f64` reading the number in place against the old copy into a `String`: the
    /// edge lines of review 13's finding, every start position of each, and 8 million
    /// numbers as `%.17g`, `%g` and hand-made forms write them (with `E`, signs, leading
    /// zeros, missing digits), compared by bits, failure and end position, and through
    /// `stream_values` and `stream_values_nonfinite`.
    ///
    /// `count` random numbers (8 million in the long run).
    fn extract_f64_in_place(count: usize) {
        let edges: [&[u8]; 30] = [
            b"",
            b" ",
            b"1e,2",
            b"1.,.5,+.25",
            b"1e400",
            b"-1e400",
            b".",
            b"1e+",
            b"1e-",
            b"5e-324",
            b"1e-400",
            b"000123.4500",
            b"-0",
            b"+0.0e0",
            b"1E5,2E-3,3e+7",
            b"..5",
            b"1.2.3",
            b"1e5e5",
            b"+-1",
            b"-.e1",
            b"\t\x0b\x0c\r\n 42",
            b"nan,inf,-inf",
            b"1.7976931348623157e308,1.7976931348623159e308",
            b"2.2250738585072011e-308",
            b"4.9406564584124654e-324,2.4703282292062327e-324",
            b"0x1p3",
            b"12abc",
            b"9007199254740993",
            b"-.5E+2,",
            b"1,2,3,4,5 , 6 ,x",
        ];
        for line in edges {
            same_along(line);
            let a = stream_values(line);
            assert_eq!(a.len(), {
                let mut pos = 0;
                let mut n = 0;
                while old_extract_f64(line, &mut pos).is_some() {
                    n += 1;
                    while pos < line.len() && c_space(line[pos]) {
                        pos += 1;
                    }
                    if line.get(pos) == Some(&b',') {
                        pos += 1;
                    } else {
                        break;
                    }
                }
                n
            });
            let _ = stream_values_nonfinite(line);
        }
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d)
        };
        let mut line = Vec::new();
        let mut written = 0usize;
        while written < count {
            line.clear();
            for k in 0..8 {
                if k > 0 {
                    line.extend_from_slice(if next() % 2 == 0 { b"," } else { b" , " });
                }
                let r = next();
                let v = match r % 5 {
                    0 => f64::from_bits(next()),
                    1 => (next() >> 11) as f64 / (1u64 << 53) as f64 * 2000.0 - 1000.0,
                    2 => f64::from(f32::from_bits(next() as u32)),
                    3 => (next() % 100_000) as f64 / 8.0,
                    _ => -((next() >> 11) as f64) * 1e-300,
                };
                let text = match (r >> 8) % 6 {
                    0 => crate::cfmt::format_g17(v),
                    1 => crate::cfmt::format_g(v, 6).to_uppercase(),
                    2 => format!("{v:e}"),
                    3 => format!("+00{}", v.abs()),
                    4 => format!("{v}").replace("0.", "."),
                    _ => format!("{:.3E}", v),
                };
                line.extend_from_slice(text.as_bytes());
                written += 1;
            }
            let (mut a, mut b) = (0, 0);
            loop {
                let before = a;
                let new = extract_f64(&line, &mut a);
                let old = old_extract_f64(&line, &mut b);
                assert_eq!(new.map(f64::to_bits), old.map(f64::to_bits));
                assert_eq!(a, b);
                while a < line.len() && (c_space(line[a]) || line[a] == b',') {
                    a += 1;
                }
                if a == before {
                    a += 1; // a word that is not a number (`nan`, `inf`)
                }
                b = a;
                if a >= line.len() {
                    break;
                }
            }
        }
    }

    #[test]
    fn review13_extract_f64_in_place() {
        extract_f64_in_place(800_000);
    }

    #[test]
    #[ignore = "long comparison run; cargo test -- --ignored _long"]
    fn review13_extract_f64_in_place_long() {
        extract_f64_in_place(8_000_000);
    }

    #[test]
    fn nonfinite_words_are_read_as_values() {
        let v = stream_values_nonfinite(b"1, nan,-inf ,INFINITY,-nan(ind),1e999,2");
        assert_eq!(v.len(), 7);
        assert!(v[1].is_nan() && v[4].is_nan());
        assert_eq!(
            [v[0], v[2], v[3], v[5], v[6]],
            [1.0, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY, 2.0]
        );
        // `in >> double` stops at the first of them.
        assert_eq!(stream_values(b"1, nan, 2"), vec![1.0]);
        // A word followed by something other than a comma ends the row after it.
        let v = stream_values_nonfinite(b"1, nanx, 3");
        assert_eq!(v.len(), 2);
        assert!(v[1].is_nan());
    }

    #[test]
    fn lines_follow_safe_getline() {
        assert_eq!(lines(b""), Vec::<&[u8]>::new());
        assert_eq!(lines(b"a\nb"), vec![&b"a"[..], b"b"]);
        assert_eq!(lines(b"a\r\nb\rc\n"), vec![&b"a"[..], b"b", b"c"]);
        assert_eq!(lines(b"\n\n"), vec![&b""[..], b""]);
        let mut r = LineReader::new(b"x,y,z\n");
        assert_eq!(r.next_line(), (&b"x,y,z"[..], false));
        assert_eq!(r.next_line(), (&b""[..], true));
    }

    #[test]
    fn stream_extraction_like_libstdcpp() {
        assert_eq!(stream_values(b"1, 2.5, -3e2"), vec![1.0, 2.5, -300.0]);
        assert_eq!(stream_values(b"  7 ,8"), vec![7.0, 8.0]);
        assert_eq!(stream_values(b"1 2 3"), vec![1.0]);
        assert_eq!(stream_values(b"1,NaN,3"), vec![1.0]);
        assert_eq!(stream_values(b"1.5x,2"), vec![1.5]);
        assert_eq!(stream_values(b"1e,2"), Vec::<f64>::new());
        assert_eq!(stream_values(b"1.,.5,+.25"), vec![1.0, 0.5, 0.25]);
        assert_eq!(stream_values(b"1.e2"), vec![100.0]);
        assert_eq!(stream_values(b"1e400"), Vec::<f64>::new());
        assert_eq!(stream_values(b"."), Vec::<f64>::new());
        assert_eq!(stream_values(b""), Vec::<f64>::new());
        assert_eq!(stream_values(b"0.1"), vec![0.1]);
        assert_eq!(stream_ints(b"1,2, 3]"), vec![1, 2, 3]);
        assert_eq!(stream_ints(b"name[1,2]"), Vec::<i32>::new());
        assert_eq!(stream_ints(b"3000000000"), Vec::<i32>::new());
    }

    #[test]
    fn qt_conversions() {
        assert_eq!(qt_to_double(" 0.3 "), 0.3);
        assert_eq!(qt_to_double("1,5"), 0.0);
        assert_eq!(qt_to_double(""), 0.0);
        assert!(qt_to_double("NaN").is_nan());
        assert_eq!(qt_to_double("-inf"), f64::NEG_INFINITY);
        assert_eq!(qt_to_double("1e"), 0.0);
        assert_eq!(qt_to_int(" 42"), 42);
        assert_eq!(qt_to_int("-1"), -1);
        assert_eq!(qt_to_int("1.0"), 0);
        assert_eq!(qt_to_int("3000000000"), 0);
        assert_eq!(qt_to_int("x"), 0);
    }

    #[test]
    fn reference_lines_like_sscanf() {
        assert_eq!(scan_reference_line(b"1 A"), Some((1, "A".to_owned())));
        assert_eq!(
            scan_reference_line(b"010 b c "),
            Some((8, "b c ".to_owned()))
        );
        assert_eq!(scan_reference_line(b"0x1f n"), Some((31, "n".to_owned())));
        assert_eq!(scan_reference_line(b"3 _x"), None);
        assert_eq!(scan_reference_line(b"x"), None);
        let long = scan_reference_line(b"2 abcdefghijklmnopqrstuvwxyz").expect("parses");
        assert_eq!(long.1, "abcdefghijklmnopqrst");
    }

    #[test]
    fn colours() {
        assert_eq!(qt_color("#ff8000"), Some(Rgb::new(255, 128, 0)));
        assert_eq!(qt_color("#f80"), Some(Rgb::new(255, 136, 0)));
        assert_eq!(qt_color("#80ff8000"), Some(Rgb::new(255, 128, 0)));
        assert_eq!(qt_color("red"), Some(Rgb::new(255, 0, 0)));
        assert_eq!(qt_color(" Light Sea GREEN"), Some(Rgb::new(32, 178, 170)));
        assert_eq!(qt_color_rgba("transparent"), Some([0, 0, 0, 0]));
        assert_eq!(qt_color(" #ff8000"), None);
        assert_eq!(qt_color("notacolour"), None);
        assert_eq!(qt_color(""), None);
    }
}
