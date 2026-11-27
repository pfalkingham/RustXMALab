//! The calibration object (cube) read from the user's files: the frame specification
//! (`.csv`) and the reference points (`.ref`).
//!
//! Ported from core/CalibrationObject.cpp, CalibrationObject::loadCoords (`:69-178`), and
//! ui/NewProjectDialog.cpp, NewProjectDialog::referencesValid (`:150-171`). The project
//! reader ([`crate::project_read`]) reads the copies a project stores in
//! `CalibrationObject/` with the same functions; File > New dataset reads the user's files.
//!
//! - **Frame specification.** The first line is a header: if it contains `white`, that text
//!   is removed and the beads are white (`whiteBlobs`); the rest is split at commas, and when
//!   it has more than three fields of which the first three start with `s`, each of them
//!   with every `s` removed is a scale factor (`QString::toDouble`) for x, y and z.
//!   Otherwise the scale is (1, 1, 1). Every further line is one bead: `x,y,z[,name]`, each
//!   coordinate read with `QString::toDouble` (text or an empty field gives 0) and
//!   multiplied by its scale; the fourth field, when present, is the bead's name. The last
//!   line counts even without a final line break.
//! - **Reference file.** Every line that `sscanf("%i %20[0-9a-zA-Z ]s")` reads with two
//!   conversions gives a bead id (decimal, `0x` hexadecimal or leading-zero octal) and a
//!   name of up to 20 letters, digits and spaces; the id is stored 0-based (`id - 1`). The
//!   New dataset dialog accepts a file with exactly four such lines
//!   ([`reference_entries`]).
//! - **Planar** ([`is_planar`]): every bead has x = 0, or every bead y = 0, or every bead
//!   z = 0 (exact comparison).
//!
//! # Fixed C++ defects (both modes)
//!
//! - A bead line with fewer than three fields (a blank line, for example a blank last line)
//!   makes the C++ read past the list (`tmp[1]`, `tmp[2]` of a one-element `QStringList`:
//!   undefined behaviour, a crash in practice). Here it is an error naming the line
//!   (ledger F14).
//! - The C++ object is a singleton: `whiteBlobs`, the scale and the bead names are never
//!   reset, so they leak from one project or new dataset to the next. Here every load starts
//!   from a fresh object (ledger F16). [`load_coords_into`] keeps what the reader needs for
//!   a project file with one calibration object.

use std::fmt;

use xma_model::{CalibrationFrame, Point3};

use crate::cparse::{LineReader, lines, qstring, qt_to_double, scan_reference_line, split_commas};

/// Why a frame specification could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordsError {
    /// The 1-based line of the frame specification.
    pub line: usize,
    /// The 1-based field that is missing.
    pub column: usize,
    pub message: String,
}

impl fmt::Display for CoordsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "frame specification line {}, field {}: {}",
            self.line, self.column, self.message
        )
    }
}

impl std::error::Error for CoordsError {}

/// A calibration frame with the C++ constructor's values: no beads, no references, scale
/// (1, 1, 1), dark beads.
#[must_use]
pub fn empty_frame() -> CalibrationFrame {
    CalibrationFrame {
        points: Vec::new(),
        point_names: Vec::new(),
        reference_ids: Vec::new(),
        reference_names: Vec::new(),
        scale: Point3::new(1.0, 1.0, 1.0),
        white_blobs: false,
        frame_specification_filename: String::new(),
        references_filename: String::new(),
    }
}

/// Reads a frame specification and a reference file into a fresh calibration frame.
/// `frame_specification_name` and `references_name` are the files' names (the name, not
/// the path: `CalibrationObject/<name>` in a saved project).
///
/// Ported from core/CalibrationObject.cpp, CalibrationObject::loadCoords.
///
/// # Errors
/// A bead line with fewer than three fields (see the module documentation).
pub fn load_coords(
    frame_specification: &[u8],
    references: &[u8],
    frame_specification_name: &str,
    references_name: &str,
) -> Result<CalibrationFrame, CoordsError> {
    let mut frame = empty_frame();
    frame.frame_specification_filename = frame_specification_name.to_owned();
    frame.references_filename = references_name.to_owned();
    load_coords_into(&mut frame, frame_specification, references)?;
    Ok(frame)
}

/// [`load_coords`] into an existing frame, as the C++ singleton does within one project
/// file: the beads, the references and their names are replaced; `white_blobs` is only ever
/// set, never cleared, and the scale is kept when the header gives none.
///
/// Ported from core/CalibrationObject.cpp, CalibrationObject::loadCoords.
///
/// # Errors
/// A bead line with fewer than three fields (see the module documentation).
pub fn load_coords_into(
    frame: &mut CalibrationFrame,
    frame_specification: &[u8],
    references: &[u8],
) -> Result<(), CoordsError> {
    frame.points.clear();
    frame.point_names.clear();
    frame.reference_ids.clear();
    frame.reference_names.clear();

    let mut reader = LineReader::new(frame_specification);
    let (header, _) = reader.next_line();
    let mut header = qstring(header);
    if header.contains("white") {
        header = header.replace("white", "");
        frame.white_blobs = true;
    }
    let parts = split_commas(&header);
    if parts.len() > 3 && parts[..3].iter().all(|p| p.starts_with('s')) {
        frame.scale = Point3::new(
            qt_to_double(&parts[0].replace('s', "")),
            qt_to_double(&parts[1].replace('s', "")),
            qt_to_double(&parts[2].replace('s', "")),
        );
    }
    let mut line_no = 1;
    loop {
        let (line, eof) = reader.next_line();
        if eof {
            break;
        }
        line_no += 1;
        let text = qstring(line);
        let f = split_commas(&text);
        if f.len() < 3 {
            return Err(CoordsError {
                line: line_no,
                column: f.len() + 1,
                message: "a bead needs x, y and z".to_owned(),
            });
        }
        frame.points.push(Point3::new(
            qt_to_double(f[0]) * frame.scale.x,
            qt_to_double(f[1]) * frame.scale.y,
            qt_to_double(f[2]) * frame.scale.z,
        ));
        frame
            .point_names
            .push(f.get(3).map_or(String::new(), |s| (*s).to_owned()));
    }

    for line in lines(references) {
        if let Some((id, name)) = scan_reference_line(line) {
            frame.reference_ids.push(id.wrapping_sub(1));
            frame.reference_names.push(name);
        }
    }
    Ok(())
}

/// The number of lines of a reference file that the C++ reads as a reference (an id and a
/// name). The New dataset dialog accepts the file only when this is exactly 4.
///
/// Ported from ui/NewProjectDialog.cpp, NewProjectDialog::referencesValid.
#[must_use]
pub fn reference_entries(references: &[u8]) -> usize {
    lines(references)
        .into_iter()
        .filter(|l| scan_reference_line(l).is_some())
        .count()
}

/// Every bead in one coordinate plane: all x = 0, all y = 0 or all z = 0 (exact). True for
/// no beads, as in the C++.
///
/// Ported from core/CalibrationObject.cpp, CalibrationObject::loadCoords (`:160-172`).
#[must_use]
pub fn is_planar(points: &[Point3]) -> bool {
    (0..3).any(|a| points.iter().all(|p| p[a] == 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_scale_names_and_references() {
        let spec = b"s2,s3,s0.5,x\n1,2,4,a\n0,0,0\n7,8,9,b,c";
        let refs = b"1 circle\n0x2 rect\n010 oct\nnot a line\n5 \n";
        let f = load_coords(spec, refs, "f.csv", "r.ref").unwrap();
        assert_eq!(f.scale, Point3::new(2.0, 3.0, 0.5));
        assert_eq!(
            f.points,
            vec![
                Point3::new(2.0, 6.0, 2.0),
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(14.0, 24.0, 4.5)
            ]
        );
        assert_eq!(f.point_names, ["a", "", "b"]);
        assert_eq!(f.reference_ids, [0, 1, 7]);
        assert_eq!(f.reference_names, ["circle", "rect", "oct"]);
        assert!(!f.white_blobs);
        assert_eq!(reference_entries(refs), 3);
    }

    #[test]
    fn white_and_short_lines() {
        let f = load_coords(b"white\n1,0,0\n", b"", "f", "r").unwrap();
        assert!(f.white_blobs);
        assert!(is_planar(&f.points));
        let e = load_coords(b"x y z\n1,2,3\n\n", b"", "f", "r").unwrap_err();
        assert_eq!((e.line, e.column), (3, 2));
    }
}
