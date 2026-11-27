//! C-style number formatting, byte-identical to what the C++ XMALab writes.
//!
//! Project files and exports must reproduce the C++ program's text exactly. The C++ uses
//! a handful of number styles, listed below with every place it uses them. Each style has
//! one function here. All functions are exact for every `f64`: the decimal expansion of
//! the binary value is computed exactly and rounded once, as glibc and Qt do.
//!
//! # Styles
//!
//! | Style | C++ | Function |
//! |---|---|---|
//! | S1 | `std::ofstream` with `precision(17)`, a `double` | [`format_g17`] (`printf("%.17g")`) |
//! | S2 | `std::ofstream` with the default precision, a `double` | [`format_g6`] (`printf("%.6g")`) |
//! | S3 | `std::ofstream`, an integer, `bool` or enum | [`format_int`], [`format_bool`] |
//! | S4 | `cv::format(mat, Formatter::FMT_CSV)` of a `CV_32F` matrix | [`format_opencv_f32`] (`printf("%.8g")`), [`opencv_csv_f32`] |
//! | S5 | `QString::number(double)` | [`format_qt_number`] (Qt `'g'`, precision 6) |
//! | S6 | `QString::number(double, 'f', 12)` | [`format_qt_fixed`] |
//! | S7 | `QString::number(int)` | [`format_int`] |
//! | S8 | `QString("%1").arg(n, 3, 10, QChar('0'))` | [`format_int_padded`] with width 3 |
//! | S9 | literal text for missing values: `NaN` (exports and project data), `NA` (portal metadata) | constant [`MISSING_NAN`]; `NA` is written as text where it is used |
//!
//! [`format_g`] is the general `printf("%.Pg")`, which S1, S2 and S4 call.
//!
//! ## How the C++ streams format numbers
//!
//! A `std::ostream` with no floatfield flags formats a `double` as `printf("%.Pg")`, where
//! P is the stream's precision (6 unless set). libstdc++ calls `vsnprintf` in the "C"
//! locale, so the output is glibc's: ties rounded to even, `nan`, `-nan`, `inf`, `-inf`,
//! `-0`, exponents with a sign and at least two digits (`1e+06`, `1e-05`, `1e+100`).
//! Integers, `bool` (without `boolalpha`) and enums print as plain decimal (`0`/`1` for
//! `bool`); the precision does not affect them. A `float` is promoted to `double` first.
//! Where the C++ writes a missing value it writes the literal `NaN`, never a formatted NaN.
//!
//! Other C++ standard libraries do not always agree with glibc: MSVC prints NaN as
//! `nan(ind)` or `-nan(ind)`, and macOS's libc++ prints `nan` without a sign. These only
//! matter for NaNs the program did not intend to write, and they are not reproduced.
//!
//! **Line endings.** Every writer below opens its file in text mode (`std::ofstream`
//! without `std::ios::binary`, or `QIODevice::Text`). Lines end with `std::endl` or
//! `'\n'`, which is written as `\n` on Linux and macOS and as `\r\n` on Windows. The
//! reference dumps were made on Linux and so contain `\n`.
//!
//! ## S1: `std::ofstream` with `precision(17)` → [`format_g17`]
//!
//! Separator `,` unless noted; one record per line.
//!
//! Project file data (`ui/ProjectFileIO.cpp:115` `saveProject` calls these):
//! - `core/Camera.cpp:415-419` `saveCameraMatrix`: 3 lines of 3 values, separator **`", "`**.
//! - `core/Camera.cpp:465-472` `saveUndistortionParam`: 8 lines of one value; a missing
//!   coefficient is the integer `0` (S3).
//! - `core/CalibrationImage.cpp:582-586` `savePoints` (`_PointsDetected.csv`,
//!   `_PointsDetectedAll.csv`): `x, y` per line, separator **`", "`**.
//! - `core/CalibrationImage.cpp:593-597` `savePointsInlier`: one `int` per line (S3).
//! - `core/CalibrationImage.cpp:618-624` `saveRotationMatrix`: 3×3 from `cv::Rodrigues`,
//!   separator **`", "`**.
//! - `core/CalibrationImage.cpp:631-636` `saveTranslationVector`: 3 lines of one value.
//! - `core/UndistortionObject.cpp:846-850` `savePoints` (`_PointsDetected`,
//!   `_GridPointsDistorted`, `_GridPointsReferences`): `x, y`, separator **`", "`**.
//! - `core/UndistortionObject.cpp:872-884` `saveGridPointsInlier`: `1` or `0` per line (S3).
//! - `core/CalibrationObject.cpp:188-200` `saveCoords`, frame specification: first line
//!   `white` or `x y z`, then `x,y,z,name` per point. `:204-210` the `.ref` file: `id name`
//!   per line, id 1-based (S3), separator a space. (`:205` sets the precision on the wrong
//!   stream; harmless, since only integers are written.)
//! - `core/Marker.cpp:131-133` `saveReference3DPoint`: `x,y,z`.
//! - `core/Marker.cpp:638-647` `save`, `MarkerNNNpoints2d.csv`: one line per frame,
//!   `x,y` for each camera in turn (`x1,y1,x2,y2`). Placeholders (-2) are written as numbers.
//! - `core/Marker.cpp:654-664` status2d, one line per frame, one `int` per camera (S3).
//! - `core/Marker.cpp:670-680` `MarkerNNNsize.csv`, one line per frame, one value per camera.
//! - `core/Marker.cpp:686-697` (3D points through `save`, with a 3D filename): `NaN,NaN,NaN`
//!   where status3D ≤ 0, else `x,y,z`.
//! - `core/Marker.cpp:787-791` `save3DPoints`, `MarkerNNNpoints3d.csv`: `x,y,z` for every
//!   frame, placeholders included (no `NaN` substitution). `:798-802` status3d, one `int`
//!   per line (S3).
//! - `core/Marker.cpp:943-947` `saveInterpolation`: one enum value per line (S3).
//! - `core/RigidBody.cpp:1454-1458` `RigidBodyNNNReferenceNames.csv`: one name per line.
//!   `:1461-1471` `ReferencePoints3d.csv`: `x,y,z` per point (original points if set,
//!   else the working set).
//! - `core/RigidBody.cpp:1479-1512` `saveOptimized`: with a header, one line
//!   `name_x,name_y,name_z,...` then one line of all points `x,y,z,...`; without, `x,y,z`
//!   per point.
//! - `core/RigidBody.cpp:2035-2038` `DummyMarkerNNNPointReferences.csv`: header `x,y,z`,
//!   then `x,y,z,index` (index an `int`, S3). `:2042-2053` `PointCoordinates.csv`: header
//!   `x,y,z`, then `x,y,z` or `NaN,NaN,NaN` per frame. `:2059-2062` `PointReferences2.csv`.
//! - `core/Trial.cpp:900-904` `saveMarkers` (`MarkerDescription.txt`): one description per
//!   line. `:911-921` `saveRigidBodies` (`RigidBodies.txt`): `name[1,2,3]`, indices 1-based (S3).
//! - `core/VideoStream.cpp:432-436` `save` (`<Cam>_filenames_absolute.txt`): file names only.
//! - `ui/ProjectFileIO.cpp:194-195` (in `saveProject`): `PrecisionInfo_<trial>.txt` and
//!   `MarkerDistances_<trial>.txt`, written by the same functions as the exports below.
//!
//! Exports (separator `,`, lines end `'\n'` unless noted):
//! - `core/Trial.cpp:959-1048` `saveRigidBodyTransformations` (one file): optional header
//!   (`Frame,` then `<name>_R11<filter>,...`), optional frame column `f + 1` (S3), then
//!   16 values per body or `NaN` ×16.
//! - `core/Trial.cpp:1074-1146` the same, one file per body.
//! - `core/Trial.cpp:1319-1486` `saveMarkerToMarkerDistances`: `Mean` table then `SD`
//!   table, then per-body tables; labels `Marker <i+1> <description>`; blank lines between
//!   tables. A pair with no common frame writes the initial `0.0` (as `0`).
//! - `core/Trial.cpp:1493-1510` `savePrecisionInfo`: header row, `description,frames
//!   (int),sd,error,error`, two blank lines, marker header, `description,frames,error,sd`.
//! - `core/Trial.cpp:1561-1626` `save3dPoints` (one file) and `:1644-1689` (one file per
//!   marker): `x,y,z` or `NaN,NaN,NaN`, optional frame column (S3).
//! - `core/Trial.cpp:1715-1807` `save2dPoints` (one file) and `:1819-1907` (per marker):
//!   optional header; optional offset columns write `0.0` as `0` (the header's
//!   `cam<j>_offset` names have no separator between them, a C++ bug); `x,y` or `NaN,NaN`;
//!   `x + 1`, `y + 1` and `height − y − 1` are computed in `double` before formatting.
//! - `core/Trial.cpp:2024-2046` `saveReprojectionErrors`: one line per frame, one value
//!   or `NaN` per camera.
//! - `core/RigidBody.cpp:2179-2271` `saveTransformations` (XROMM VR body files, via
//!   `core/Trial.cpp:2449`): 16 values or `NaN` ×16 per frame, lines end `std::endl`.
//! - `core/Marker.cpp:787-802` `save3DPoints` (XROMM VR marker files, via
//!   `core/Trial.cpp:2456`): as in the project.
//! - `core/Camera.cpp:516-554` `saveMayaCamVersion2`: text lines `image size`, `camera
//!   matrix`, `rotation`, `translation` and optionally `undistortion`, blank lines between
//!   blocks; the image size is two `int`s (S3).
//! - `core/Project.cpp:498-511` `exportDLT` (`MergedDlts_Frame<n>.csv`, `n` via S7): 11
//!   lines, one value per camera.
//! - `core/Project.cpp:540-546` `exportMayaCam` (`_MayaCam.csv`): 5 lines of 3 values,
//!   lines end `"\n"`; values 13 and 14 are `int` width and height converted to `double`.
//! - `core/UndistortionObject.cpp:896-917` `exportData`, `_INPTS.csv` and `_BSPTS.csv`:
//!   `x,y` per inlier.
//! - `ui/PlotWindow.cpp:137-541` `saveData` (plot data): `Frame,` header; frame `i + 1`
//!   (S3); values with S1, except the rigid-body plots, which pre-format with S6. One
//!   missing-value literal is `NaN,NaN,NaN, NaN` (with a space, `:173`), and the
//!   reprojection-error plot writes an extra blank line after each `NaN` (`:285-287`).
//!
//! ## S2: `std::ofstream` with the default precision → [`format_g6`]
//!
//! - `core/Trial.cpp:2459-2470` `saveVR`, `VRData.csv`: `Obj,<mesh>,<body>.csv,<meshScale>`,
//!   the mesh scale a `double` with six significant digits; `Marker,...,-1` (S3).
//! - `core/EventData.cpp:84-87` `saveData` (`<event>.csv`, project and export): one `bool`
//!   per line, so `0` or `1` (S3); no `double`s.
//!
//! ## S4: OpenCV CSV formatter → [`format_opencv_f32`], [`opencv_csv_f32`]
//!
//! - `core/UndistortionObject.cpp:891-893` `exportData`, `_LUT.csv`: `outfile <<
//!   cv::format(undistortionMapX, cv::Formatter::FMT_CSV) << std::endl`, then the same
//!   for `undistortionMapY`. The maps are `CV_32FC1` (`processing/LocalUndistortion.cpp:379-380`).
//!   OpenCV's formatter (`modules/core/src/out.cpp`) ignores the stream's precision and
//!   writes each element with `"%.8g"` (`prec32f` = 8), elements separated by **`", "`**,
//!   each row ending `\n`. With `std::endl` after it, each map is followed by an empty line.
//!   Verified against the pig dump: 1024 rows + blank line per map, 8 significant digits.
//!
//! ## S5: `QString::number(double)` → [`format_qt_number`]
//!
//! Qt's `'g'` with precision 6, in the C locale. Differences from `printf("%.6g")`, found by
//! testing Qt 6.12 (fixture `tests/data/cfmt_qt.tsv`): exact ties round away from zero
//! (`123456.5` → `123457`; glibc gives `123456`), `-0` is written `0`, and every NaN is
//! `nan`. This holds for Qt built with its double-conversion library (the default).
//! XML attributes, written by `QXmlStreamWriter` with auto-formatting (4-space indent,
//! `QIODevice::Text`):
//! - `project.xml` (`ui/ProjectFileIO.cpp:1456` `writeProjectFile`): doubles at
//!   `:1487-1488` (camera image adjustment `<view>Scale`, `<view>Bias`, written only when
//!   not 1 and 0), `:1509-1510` (undistortion centre `x`, `y`), `:1609` (`SquareSize`),
//!   `:1631` (`recordingSpeed`), `:1632` (`cutOffFrequency`), `:1690` (rigid body
//!   `CutOffFrequency`), `:1694` (`MeshScale`). Every other number there is an `int` or
//!   `bool` (S7): `:1470`, `:1474`, `:1480-1483`, `:1503`, `:1537`, `:1540-1541`,
//!   `:1552-1553`, `:1562-1563`, `:1568`, `:1604`, `:1607-1608`, `:1626-1630`,
//!   `:1633-1634`, `:1644`, `:1652-1657`, `:1677`, `:1687`, `:1689`, `:1693`.
//! - `xma_metadata.xml` (`ui/ProjectFileIO.cpp:1166` `writePortalFile`, portal summary,
//!   write-only): doubles at `:1234`, `:1240` (calibration `error`), `:1254` (checkerboard
//!   `size`), `:1280-1281` (`filterfrequency`, `framerate`), `:1283`, `:1285`, `:1316`,
//!   `:1318`, `:1320`, `:1362`, `:1405` (errors and SDs; `NA` when ≤ 0), `:1322`, `:1326`
//!   (`filterfrequency`); integers (S7) at `:1188`, `:1211`, `:1218-1220`, `:1231-1233`,
//!   `:1238-1239`, `:1252-1253`, `:1261`, `:1264`, `:1275`, `:1291-1293`, `:1309`,
//!   `:1311-1312` (`NA` when < 0), `:1333`, `:1376`; per-camera lists such as `1 / 1`
//!   built at `:1337-1356` and `:1380-1399` (S7 joined with `" / "`, `NA` for none).
//! - File names: `MergedDlts_Frame` + S7 (`core/Project.cpp:498`), `_MayaCam` + S7 of the
//!   1-based camera id (`core/Project.cpp:565`).
//!
//! ## S6: `QString::number(double, 'f', 12)` → [`format_qt_fixed`]
//!
//! - `ui/PlotWindow.cpp:412-438` (rigid-body angles and translations) and `:496-497`
//!   (rigid-body error mean and SD), written into the plot data export as text.
//!   Ties round away from zero; `-0` is `0.000000000000`, but a negative value that rounds
//!   to zero keeps its sign (`-0.000000000000`).
//!
//! ## S8: zero-padded integers → [`format_int_padded`]
//!
//! `QString("%1").arg(n, width, 10, QChar('0'))`: at least `width` digits, longer numbers
//! unchanged. Width 3:
//! - project file names `MarkerNNN...`, `RigidBodyNNN...`, `RigidBodyNNNDummyMarkerMMM...`,
//!   0-based (`ui/ProjectFileIO.cpp:222-258` and the matching XML attributes at
//!   `:1661-1669`, `:1680-1685`, `:1701-1703`);
//! - export names, 1-based: `RigidBody001` headers and file names (`core/Trial.cpp:974`,
//!   `:1063`, `:1070`, `:1087`), default marker names `marker001` (`:1574`, `:1655`,
//!   `:1732`, `:1834`), per-marker files `Marker001_<name>_...` (`:1637`, `:1818`, `:2023`).
//!
//! Width 4: undistorted image sequence names, 1-based (`core/Trial.cpp:1281`, `:1291`,
//! `:1308`, `ui/UndistortSequenceDialog.cpp:262`). Width 6: calibration sequence frame
//! names, 0-based (`core/CalibrationSequence.cpp:131`, `:142`, `:343`).

/// The literal the C++ writes for a missing value in exports and project data files.
pub const MISSING_NAN: &str = "NaN";

/// Rounding rule for an exact tie.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ties {
    /// glibc `printf` (in the default rounding mode). [`format_g`] takes its digits from the
    /// standard library, which rounds the same way; the tests compare the two.
    #[cfg_attr(not(test), allow(dead_code))]
    Even,
    /// Qt (double-conversion).
    AwayFromZero,
}

/// Formats like glibc `printf("%.{precision}g", value)`, byte for byte.
///
/// Precision 0 is treated as 1, as C does. Handles every `f64`: `nan` and `-nan` (by sign
/// bit), `inf`, `-inf`, `-0`, subnormals, the switch between fixed and exponent notation
/// (exponent when the decimal exponent X is below −4 or at least the precision), removal
/// of trailing zeros, and exponents written with a sign and at least two digits.
pub fn format_g(value: f64, precision: usize) -> String {
    let mut out = String::new();
    push_g(&mut out, value, precision);
    out
}

/// [`format_g`] appended to `out`, for writers that format many values into one text.
///
/// The digits come from the standard library's exact formatting (`{:.*e}`), which rounds
/// the exact binary value once with ties to even, as glibc does; they are then laid out as
/// `%g` lays them out. The big-integer expansion behind [`format_qt_number`] gives the same
/// bytes; the unit test `review13_push_g_matches_exact_expansion` compares the two.
pub fn push_g(out: &mut String, value: f64, precision: usize) {
    use std::fmt::Write as _;
    let p = precision.max(1);
    if value.is_nan() {
        out.push_str(if value.is_sign_negative() {
            "-nan"
        } else {
            "nan"
        });
        return;
    }
    if value.is_sign_negative() {
        out.push('-');
    }
    if value.is_infinite() {
        out.push_str("inf");
        return;
    }
    // `d.ddd…e<X>` with p significant digits (`de<X>` when p is 1). Zero gives `0.00…e0`.
    let mut stack = StackText::new();
    let heap;
    let sci: &[u8] = if p <= StackText::DIGITS {
        // Cannot fail: the buffer holds p + 1 characters plus `e-324`.
        let _ = write!(stack, "{:.*e}", p - 1, value.abs());
        stack.as_bytes()
    } else {
        heap = format!("{:.*e}", p - 1, value.abs());
        heap.as_bytes()
    };
    let e_at = sci.iter().position(|&b| b == b'e').unwrap_or(sci.len());
    let mantissa = &sci[..e_at];
    let x: i64 = std::str::from_utf8(&sci[(e_at + 1).min(sci.len())..])
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    // The p significant digits, without the point.
    let first = mantissa[0];
    let rest = if mantissa.len() > 2 {
        &mantissa[2..]
    } else {
        &[][..]
    };
    let digit = |i: usize| -> u8 { if i == 0 { first } else { rest[i - 1] } };
    if x < -4 || x >= p as i64 {
        // Exponent notation: d.ddd, trailing zeros removed, then e±XX.
        out.push(first as char);
        let mut n = rest.len();
        while n > 0 && rest[n - 1] == b'0' {
            n -= 1;
        }
        if n > 0 {
            out.push('.');
            for &d in &rest[..n] {
                out.push(d as char);
            }
        }
        out.push('e');
        out.push(if x < 0 { '-' } else { '+' });
        let ax = x.unsigned_abs();
        if ax < 10 {
            out.push('0');
        }
        let _ = write!(out, "{ax}");
    } else {
        // Fixed notation with p − 1 − X decimals, trailing zeros removed.
        let decpt = x + 1;
        let start = out.len();
        let has_point = if decpt <= 0 {
            out.push_str("0.");
            for _ in 0..-decpt {
                out.push('0');
            }
            for i in 0..p {
                out.push(digit(i) as char);
            }
            true
        } else {
            let decpt = decpt as usize;
            for i in 0..p {
                if i == decpt {
                    out.push('.');
                }
                out.push(digit(i) as char);
            }
            decpt < p
        };
        if has_point {
            while out.len() > start && out.ends_with('0') {
                out.pop();
            }
            if out.ends_with('.') {
                out.pop();
            }
        }
    }
}

/// A fixed buffer for the standard library's `{:.*e}` text of a short precision.
struct StackText {
    buf: [u8; StackText::DIGITS + 8],
    len: usize,
}

impl StackText {
    /// The largest precision formatted on the stack (`d.` + digits + `e-324`).
    const DIGITS: usize = 40;

    fn new() -> StackText {
        StackText {
            buf: [0; StackText::DIGITS + 8],
            len: 0,
        }
    }

    fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}

impl std::fmt::Write for StackText {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let end = self.len + s.len();
        if end > self.buf.len() {
            return Err(std::fmt::Error);
        }
        self.buf[self.len..end].copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

/// `printf("%.17g")`: style S1, every `std::ofstream` with `precision(17)`. Round-trips.
pub fn format_g17(value: f64) -> String {
    format_g(value, 17)
}

/// [`format_g17`] appended to `out`.
pub fn push_g17(out: &mut String, value: f64) {
    push_g(out, value, 17)
}

/// `printf("%.12g")`, the precision of project files written by some older XMALab builds.
pub fn format_g12(value: f64) -> String {
    format_g(value, 12)
}

/// `printf("%.6g")`: style S2, a `std::ofstream` with its default precision.
pub fn format_g6(value: f64) -> String {
    format_g(value, 6)
}

/// One element of OpenCV's `cv::format(mat, Formatter::FMT_CSV)` for a `CV_32F` matrix:
/// `printf("%.8g")` of the `float` promoted to `double` (style S4).
pub fn format_opencv_f32(value: f32) -> String {
    format_g(f64::from(value), 8)
}

/// The whole text OpenCV's CSV formatter produces for a `CV_32F` matrix of `rows` × `cols`
/// (row-major `data`, one channel): elements joined by `", "`, every row followed by `\n`
/// when `rows > 1`. A single row gets no final newline. The C++ follows it with
/// `std::endl`, which the caller adds.
pub fn opencv_csv_f32(data: &[f32], rows: usize, cols: usize) -> String {
    assert_eq!(data.len(), rows * cols, "data length is not rows × cols");
    let mut out = String::new();
    for r in 0..rows {
        for c in 0..cols {
            if c > 0 {
                out.push_str(", ");
            }
            push_g(&mut out, f64::from(data[r * cols + c]), 8);
        }
        if rows > 1 {
            out.push('\n');
        }
    }
    out
}

/// `QString::number(value)`: Qt's `'g'` format with precision 6 (style S5).
///
/// As [`format_g6`], except: exact ties round away from zero, `-0` is `0`, and every NaN
/// is `nan`.
pub fn format_qt_number(value: f64) -> String {
    format_general(value, 6, Ties::AwayFromZero, true)
}

/// `QString::number(value, 'f', decimals)`: fixed notation with exactly `decimals` digits
/// after the point, and no point when `decimals` is 0 (style S6, used with 12).
///
/// Exact ties round away from zero. `-0` is written without a sign; a negative value that
/// rounds to zero keeps it (`-0.000000000000`). NaN is `nan`, infinities `inf` and `-inf`.
pub fn format_qt_fixed(value: f64, decimals: usize) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let mut out = String::new();
    if value < 0.0 {
        out.push('-');
    }
    let (digits, decpt) = if value == 0.0 {
        (Vec::new(), 0)
    } else {
        let (d, p) = exact_decimal(value.abs());
        round_at(&d, p, i64::from(p) + decimals as i64, Ties::AwayFromZero)
    };
    // `digits` holds the significant digits of the rounded value; place the point.
    let digit = |i: i64| -> u8 {
        if i >= 0 && (i as usize) < digits.len() {
            b'0' + digits[i as usize]
        } else {
            b'0'
        }
    };
    let decpt = i64::from(decpt);
    if decpt <= 0 {
        out.push('0');
    } else {
        for i in 0..decpt {
            out.push(digit(i) as char);
        }
    }
    if decimals > 0 {
        out.push('.');
        for k in 0..decimals as i64 {
            out.push(digit(decpt + k) as char);
        }
    }
    out
}

/// An integer as plain decimal: what `std::ostream`, `QString::number(int)` and
/// `std::to_string` write (styles S3, S7).
pub fn format_int(value: i64) -> String {
    value.to_string()
}

/// A `bool` as `std::ostream` (without `boolalpha`) or `QString::number(bool)` writes it:
/// `1` or `0`.
pub fn format_bool(value: bool) -> String {
    if value { "1" } else { "0" }.to_string()
}

/// `QString("%1").arg(value, width, 10, QChar('0'))` for a non-negative value: decimal,
/// left-padded with zeros to `width` digits, never truncated (style S8, width 3).
pub fn format_int_padded(value: u64, width: usize) -> String {
    format!("{value:0width$}")
}

/// The shared implementation of `%g`-style formatting.
fn format_general(value: f64, precision: usize, ties: Ties, qt: bool) -> String {
    let p = precision.max(1);
    if value.is_nan() {
        return if !qt && value.is_sign_negative() {
            "-nan"
        } else {
            "nan"
        }
        .to_string();
    }
    let negative = value.is_sign_negative() && !(qt && value == 0.0);
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if value.is_infinite() {
        out.push_str("inf");
        return out;
    }
    let (digits, decpt) = if value == 0.0 {
        (vec![0u8; p], 1)
    } else {
        let (d, dp) = exact_decimal(value.abs());
        let (mut r, dp) = round_at(&d, dp, p as i64, ties);
        r.resize(p, 0);
        (r, dp)
    };
    let x = i64::from(decpt) - 1;
    if x < -4 || x >= p as i64 {
        // Exponent notation: d.ddd, trailing zeros removed, then e±XX.
        let mut mant: Vec<u8> = digits.clone();
        while mant.len() > 1 && mant[mant.len() - 1] == 0 {
            mant.pop();
        }
        out.push((b'0' + mant[0]) as char);
        if mant.len() > 1 {
            out.push('.');
            for &d in &mant[1..] {
                out.push((b'0' + d) as char);
            }
        }
        out.push('e');
        out.push(if x < 0 { '-' } else { '+' });
        let ax = x.unsigned_abs();
        if ax < 10 {
            out.push('0');
        }
        out.push_str(&ax.to_string());
    } else {
        // Fixed notation with p − 1 − X decimals, trailing zeros removed.
        let decpt = decpt as i64;
        let mut s = String::new();
        if decpt <= 0 {
            s.push('0');
            s.push('.');
            for _ in 0..-decpt {
                s.push('0');
            }
            for &d in &digits {
                s.push((b'0' + d) as char);
            }
        } else {
            for (i, &d) in digits.iter().enumerate() {
                if i as i64 == decpt {
                    s.push('.');
                }
                s.push((b'0' + d) as char);
            }
        }
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        out.push_str(&s);
    }
    out
}

/// Rounds the significant digits `digits` (value `0.d1d2… × 10^decpt`) to `keep` digits
/// (which may be zero or negative, meaning the rounding position is left of the first
/// digit). Returns the kept digits (possibly fewer than `keep` if `digits` is short;
/// never with a carry left over) and the new `decpt`. A result of zero is returned as
/// no digits.
fn round_at(digits: &[u8], decpt: i32, keep: i64, ties: Ties) -> (Vec<u8>, i32) {
    if keep >= digits.len() as i64 {
        return (digits.to_vec(), decpt);
    }
    if keep < 0 {
        // The value is below a tenth of the rounding unit.
        return (Vec::new(), decpt);
    }
    let keep = keep as usize;
    let mut kept = digits[..keep].to_vec();
    let first = digits[keep];
    let rest_nonzero = digits[keep + 1..].iter().any(|&d| d != 0);
    let up = match first.cmp(&5) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => {
            rest_nonzero
                || match ties {
                    Ties::AwayFromZero => true,
                    Ties::Even => kept.last().is_some_and(|d| d % 2 == 1),
                }
        }
    };
    let mut decpt = decpt;
    if up {
        let mut i = kept.len();
        loop {
            if i == 0 {
                kept.insert(0, 1);
                decpt += 1;
                kept.truncate(keep.max(1));
                break;
            }
            i -= 1;
            if kept[i] == 9 {
                kept[i] = 0;
            } else {
                kept[i] += 1;
                break;
            }
        }
    }
    (kept, decpt)
}

/// Exact decimal expansion of a finite, positive `f64`: its significant digits (first
/// digit nonzero, no trailing zeros) and `decpt`, so that the value is `0.d1d2… × 10^decpt`.
fn exact_decimal(v: f64) -> (Vec<u8>, i32) {
    debug_assert!(v.is_finite() && v > 0.0);
    let bits = v.to_bits();
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (mut m, mut e) = if exp_bits == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), exp_bits - 1075)
    };
    // Remove factors of two so the numbers stay small.
    let tz = m.trailing_zeros() as i32;
    m >>= tz;
    e += tz;

    let mut big = BigDec::from_u64(m);
    let mut decpt_shift: i32 = 0;
    if e >= 0 {
        let mut k = e;
        while k > 0 {
            let s = k.min(29);
            big.mul_small(1u32 << s);
            k -= s;
        }
    } else {
        // m · 2^e = m · 5^(−e) · 10^e.
        let mut k = -e;
        while k > 0 {
            let s = k.min(13);
            big.mul_small(5u32.pow(s as u32));
            k -= s;
        }
        decpt_shift = e;
    }
    let mut digits = big.to_digits();
    let decpt = digits.len() as i32 + decpt_shift;
    while digits.last() == Some(&0) {
        digits.pop();
    }
    (digits, decpt)
}

/// A non-negative big integer in base 10^9, least significant limb first.
struct BigDec {
    limbs: Vec<u32>,
}

const BASE: u64 = 1_000_000_000;

impl BigDec {
    fn from_u64(mut v: u64) -> BigDec {
        let mut limbs = Vec::new();
        while v > 0 {
            limbs.push((v % BASE) as u32);
            v /= BASE;
        }
        BigDec { limbs }
    }

    fn mul_small(&mut self, f: u32) {
        let mut carry: u64 = 0;
        for l in &mut self.limbs {
            let x = u64::from(*l) * u64::from(f) + carry;
            *l = (x % BASE) as u32;
            carry = x / BASE;
        }
        while carry > 0 {
            self.limbs.push((carry % BASE) as u32);
            carry /= BASE;
        }
    }

    fn to_digits(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.limbs.len() * 9);
        for (i, &l) in self.limbs.iter().rev().enumerate() {
            let s = if i == 0 {
                l.to_string()
            } else {
                format!("{l:09}")
            };
            out.extend(s.bytes().map(|b| b - b'0'));
        }
        out
    }
}

#[cfg(test)]
// The tests write values with all 17 digits on purpose, as the C++ prints them.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn g17_examples() {
        assert_eq!(format_g17(0.1), "0.10000000000000001");
        assert_eq!(format_g17(0.3), "0.29999999999999999");
        assert_eq!(format_g17(566.0), "566");
        assert_eq!(format_g17(2.2361679077100001), "2.2361679077100001");
        assert_eq!(format_g17(-2.0), "-2");
        assert_eq!(format_g17(1e23), "9.9999999999999992e+22");
        assert_eq!(format_g17(1e100), "1e+100");
        assert_eq!(format_g17(5e-324), "4.9406564584124654e-324");
        assert_eq!(format_g17(-0.0), "-0");
        assert_eq!(format_g17(0.0), "0");
        assert_eq!(format_g17(f64::NAN), "nan");
        assert_eq!(format_g17(-f64::NAN), "-nan");
        assert_eq!(format_g17(f64::INFINITY), "inf");
        assert_eq!(format_g17(f64::NEG_INFINITY), "-inf");
        assert_eq!(format_g17(0.9f32 as f64), "0.89999997615814209");
        assert_eq!(
            format_g17(-7.1054273576010019e-15),
            "-7.1054273576010019e-15"
        );
        assert_eq!(format_g17(1e16), "10000000000000000");
        assert_eq!(format_g17(1e17), "1e+17");
    }

    #[test]
    fn g6_and_g12_examples() {
        assert_eq!(format_g6(1e6), "1e+06");
        assert_eq!(format_g6(999999.0), "999999");
        assert_eq!(format_g6(999999.5), "1e+06");
        assert_eq!(format_g6(123456.5), "123456"); // tie to even
        assert_eq!(format_g6(0.0001), "0.0001");
        assert_eq!(format_g6(0.00001), "1e-05");
        assert_eq!(format_g6(0.000099999951), "0.0001");
        assert_eq!(format_g6(1.0 / 3.0), "0.333333");
        assert_eq!(format_g6(250.0), "250");
        assert_eq!(format_g6(-3.25e-7), "-3.25e-07");
        assert_eq!(format_g12(87.444340390700006), "87.4443403907");
        assert_eq!(format_g(0.5, 0), "0.5");
        assert_eq!(format_g(2.5, 1), "2");
        assert_eq!(format_g(3.5, 1), "4");
        assert_eq!(format_g(95.0, 1), "1e+02");
        assert_eq!(format_g(1e300, 3), "1e+300");
    }

    #[test]
    fn opencv_style() {
        assert_eq!(format_opencv_f32(-1.0), "-1");
        assert_eq!(format_opencv_f32(381.82788), "381.82788");
        assert_eq!(format_opencv_f32(0.1), "0.1");
        assert_eq!(
            opencv_csv_f32(&[-1.0, 1.5, 2.0, 3.25], 2, 2),
            "-1, 1.5\n2, 3.25\n"
        );
        assert_eq!(opencv_csv_f32(&[1.0, 2.0], 1, 2), "1, 2");
    }

    #[test]
    fn qt_number_examples() {
        assert_eq!(format_qt_number(123456.5), "123457");
        assert_eq!(format_qt_number(-0.0), "0");
        assert_eq!(format_qt_number(-f64::NAN), "nan");
        assert_eq!(format_qt_number(f64::NEG_INFINITY), "-inf");
        assert_eq!(format_qt_number(1e6), "1e+06");
        assert_eq!(format_qt_number(0.9f32 as f64), "0.9");
        assert_eq!(format_qt_number(250.0), "250");
        assert_eq!(format_qt_number(0.224471), "0.224471");
        assert_eq!(format_qt_number(5e-324), "4.94066e-324");
    }

    #[test]
    fn qt_fixed_examples() {
        assert_eq!(format_qt_fixed(0.0, 12), "0.000000000000");
        assert_eq!(format_qt_fixed(-0.0, 12), "0.000000000000");
        assert_eq!(format_qt_fixed(-5e-324, 12), "-0.000000000000");
        assert_eq!(format_qt_fixed(1.0 / 3.0, 12), "0.333333333333");
        assert_eq!(format_qt_fixed(3.141592653589793, 12), "3.141592653590");
        assert_eq!(format_qt_fixed(0.5, 0), "1");
        assert_eq!(format_qt_fixed(1.5, 0), "2");
        assert_eq!(format_qt_fixed(0.4, 0), "0");
        assert_eq!(format_qt_fixed(0.05, 1), "0.1");
        assert_eq!(format_qt_fixed(0.004, 2), "0.00");
        assert_eq!(format_qt_fixed(0.005, 2), "0.01"); // 0.005 is above the tie
        assert_eq!(format_qt_fixed(99.9999999999999, 12), "100.000000000000");
        assert_eq!(format_qt_fixed(99.99999999999, 12), "99.999999999990");
        assert_eq!(format_qt_fixed(999.9999999999999, 12), "1000.000000000000");
        assert_eq!(format_qt_fixed(-12.5, 0), "-13");
        assert_eq!(format_qt_fixed(f64::NAN, 12), "nan");
    }

    #[test]
    fn integers() {
        assert_eq!(format_int(-1), "-1");
        assert_eq!(format_int(145), "145");
        assert_eq!(format_bool(true), "1");
        assert_eq!(format_bool(false), "0");
        assert_eq!(format_int_padded(0, 3), "000");
        assert_eq!(format_int_padded(7, 3), "007");
        assert_eq!(format_int_padded(42, 3), "042");
        assert_eq!(format_int_padded(1234, 3), "1234");
    }

    /// The `%g` of the exact big-integer expansion, as `format_g` computed it before review
    /// 13: the reference for the standard library's digits.
    fn reference_g(value: f64, precision: usize) -> String {
        format_general(value, precision, Ties::Even, false)
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Values at the edges: zeros, NaNs of both signs and with payloads, infinities, the
    /// extremes, subnormals, exact ties and the powers of ten around the switch between
    /// fixed and exponent notation, each with its neighbours.
    fn edge_values() -> Vec<f64> {
        let mut v = vec![
            0.0,
            -0.0,
            f64::NAN,
            -f64::NAN,
            f64::from_bits(0x7ff0_0000_0000_0001),
            f64::from_bits(0xfff8_0000_dead_beef),
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::from_bits(0x000f_ffff_ffff_ffff),
            f64::EPSILON,
            0.5,
            1.5,
            2.5,
            3.5,
            123456.5,
            1234567.5,
            999999.5,
            9.5,
            0.25,
            0.125,
            0.375,
            1e23,
            8.5e-5,
            9.9999999999999995e-5,
            0.000099999951,
        ];
        for e in -330..=310 {
            let p: f64 = format!("1e{e}").parse().unwrap_or(0.0);
            for x in [p, 9.5 * p, 5.0 * p, 0.5 * p, 9.999999999999999 * p] {
                v.push(x);
                v.push(f64::from_bits(x.to_bits().wrapping_add(1)));
                v.push(f64::from_bits(x.to_bits().wrapping_sub(1)));
            }
        }
        let negated: Vec<f64> = v.iter().map(|x| -x).collect();
        v.extend(negated);
        v
    }

    fn check(value: f64, precision: usize) {
        let new = format_g(value, precision);
        let old = reference_g(value, precision);
        assert_eq!(
            new,
            old,
            "{:#018x} at precision {precision}",
            value.to_bits()
        );
    }

    /// `format_g` (the standard library's exact digits) against the big-integer expansion,
    /// byte for byte, at the precisions the crate writes (17, 12, 8, 6), on 4.2 million
    /// values: random bit patterns, uniform values, values of every magnitude, half-integers,
    /// `f32` values, and the edges. Then every precision from 0 to 60 (past the stack
    /// buffer) on fewer values.
    ///
    /// Every count of random values is `1 / shrink` of the full run's.
    fn push_g_matches_exact_expansion(shrink: usize) {
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        let mut values = edge_values();
        for _ in 0..1_000_000 / shrink {
            values.push(f64::from_bits(rng.next()));
        }
        for _ in 0..1_000_000 / shrink {
            values.push(rng.unit() * 2000.0 - 1000.0);
        }
        for _ in 0..1_000_000 / shrink {
            // Every magnitude from 1e-30 to 1e30, both signs.
            let e = rng.unit() * 60.0 - 30.0;
            let s = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
            values.push(s * 10f64.powf(e));
        }
        for _ in 0..600_000 / shrink {
            values.push((rng.next() % 4_000_001) as f64 / 2.0 - 1_000_000.0);
        }
        for _ in 0..600_000 / shrink {
            values.push(f64::from(f32::from_bits(rng.next() as u32)));
        }
        assert!(values.len() >= 4_200_000 / shrink);
        for &v in &values {
            for p in [17, 12, 8, 6] {
                check(v, p);
            }
        }
        let mut some: Vec<f64> = edge_values();
        for _ in 0..3000 / shrink {
            some.push(f64::from_bits(rng.next()));
            some.push(rng.unit() * 2000.0 - 1000.0);
            some.push((rng.next() % 2001) as f64 / 2.0);
        }
        for &v in &some {
            for p in 0..=60 {
                check(v, p);
            }
        }
    }

    #[test]
    fn review13_push_g_matches_exact_expansion() {
        push_g_matches_exact_expansion(100);
    }

    #[test]
    #[ignore = "long comparison run; cargo test -- --ignored _long"]
    fn review13_push_g_matches_exact_expansion_long() {
        push_g_matches_exact_expansion(1);
    }

    /// The appending forms add exactly what the returning forms give.
    #[test]
    fn review13_push_appends() {
        let mut s = String::from("x,");
        push_g17(&mut s, 0.1);
        s.push(',');
        push_g(&mut s, 100.0, 6);
        s.push(',');
        push_g(&mut s, 0.000012345, 3);
        s.push(',');
        push_g17(&mut s, -f64::NAN);
        assert_eq!(s, "x,0.10000000000000001,100,1.23e-05,-nan");
        // Trailing zeros are removed only from what is appended.
        let mut s = String::from("10.");
        push_g(&mut s, 0.5, 6);
        assert_eq!(s, "10.0.5");
        let mut s = String::from("1.0");
        push_g(&mut s, 100.0, 6);
        assert_eq!(s, "1.0100");
    }

    #[test]
    fn exact_expansion() {
        assert_eq!(exact_decimal(0.5), (vec![5], 0));
        assert_eq!(exact_decimal(1.0), (vec![1], 1));
        assert_eq!(exact_decimal(1024.0), (vec![1, 0, 2, 4], 4));
        assert_eq!(exact_decimal(0.125), (vec![1, 2, 5], 0));
        let (d, p) = exact_decimal(5e-324);
        assert_eq!(p, -323);
        assert_eq!(d.len(), 751);
        assert_eq!(&d[..5], &[4, 9, 4, 0, 6]);
        let (d, p) = exact_decimal(f64::MAX);
        assert_eq!((d.len(), p), (309, 309));
    }
}
