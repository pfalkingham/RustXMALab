//! Test archives built as XMALab's Portal loaders expect them (there is no real Portal
//! file in the project). The same archives are made by `oracle/py/make_portal_fixture.py`.
//!
//! Every builder takes the pig dataset's raw files (`ExampleDatasets/1 - Pig dataset`):
//! its two videos for a trial, copies of its calibration images packed as image-sequence
//! zips for a trial of images, and its calibration images, grids, frame specification and
//! reference file for a project. What the format holds beyond what the loaders read
//! (the root element's name, the other elements' order, the folder name) is inferred and
//! marked so in [`Fixture::metadata_xml`].

use std::path::Path;

use crate::zip::{DosDateTime, NewEntry, write_archive};

/// One `File` element of the metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct MetaFile {
    pub filename: Option<String>,
    pub file_id: i32,
    pub category: String,
    /// `metadata/cameraNumber`; `None` writes no element.
    pub camera: Option<String>,
    /// `metadata/FrameRate`; `None` writes no element.
    pub frame_rate: Option<String>,
}

impl MetaFile {
    /// A file of camera `cam<camera>`.
    #[must_use]
    pub fn new(filename: &str, file_id: i32, category: &str, camera: u32) -> Self {
        Self {
            filename: Some(filename.to_owned()),
            file_id,
            category: category.to_owned(),
            camera: Some(format!("cam{camera}")),
            frame_rate: None,
        }
    }

    #[must_use]
    pub fn rate(mut self, r: &str) -> Self {
        self.frame_rate = Some(r.to_owned());
        self
    }
}

/// A Portal archive to build.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fixture {
    /// The top folder's name.
    pub top: String,
    pub study_name: String,
    pub study_id: i32,
    pub trial_name: String,
    pub trial_id: i32,
    pub files: Vec<MetaFile>,
    /// Files under the top folder (relative names) and their bytes.
    pub contents: Vec<(String, Vec<u8>)>,
    /// Entries with any name (outside the top folder, climbing out, absolute).
    pub raw_entries: Vec<(String, Vec<u8>)>,
    /// Elements no loader reads, written inside the root and inside each `File`.
    pub unknown_elements: bool,
    /// No metadata file at all.
    pub omit_metadata: bool,
    /// This text instead of the generated metadata.
    pub metadata_override: Option<String>,
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

impl Fixture {
    /// The metadata file's text. The root element `XMALab_Files`, the order of the
    /// elements and the per-file fields other than `Filename`, `FileID`, `FileCategory`,
    /// `cameraNumber` and `FrameRate` are inferred (XMALab reads only element names).
    #[must_use]
    pub fn metadata_xml(&self) -> String {
        if let Some(t) = &self.metadata_override {
            return t.clone();
        }
        let mut s = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<XMALab_Files>\n");
        s += "  <Repository>XMAPortal</Repository>\n";
        s += &format!("  <StudyName>{}</StudyName>\n", esc(&self.study_name));
        s += &format!("  <StudyID>{}</StudyID>\n", self.study_id);
        s += &format!("  <TrialName>{}</TrialName>\n", esc(&self.trial_name));
        s += &format!("  <TrialID>{}</TrialID>\n", self.trial_id);
        s += "  <TrialNumber>1</TrialNumber>\n  <TrialType>Feeding</TrialType>\n";
        s +=
            "  <lab>Test lab</lab>\n  <attribcomment>made by the fixture builder</attribcomment>\n";
        s += "  <ts>2026-11-16 10:00:00</ts>\n  <trialDate>2026-11-16</trialDate>\n";
        if self.unknown_elements {
            s += "  <PortalVersion>9.9</PortalVersion>\n  <Owner><Name>X</Name></Owner>\n";
        }
        for f in &self.files {
            s += "  <File>\n";
            if let Some(n) = &f.filename {
                s += &format!("    <Filename>{}</Filename>\n", esc(n));
            }
            s += &format!("    <FileID>{}</FileID>\n", f.file_id);
            s += &format!("    <FileCategory>{}</FileCategory>\n", esc(&f.category));
            if self.unknown_elements {
                s += "    <Checksum>abc</Checksum>\n";
            }
            s += "    <metadata>\n";
            if let Some(c) = &f.camera {
                s += &format!("      <cameraNumber>{}</cameraNumber>\n", esc(c));
            }
            s += "      <Instrument>Xray</Instrument>\n";
            if let Some(r) = &f.frame_rate {
                s += &format!("      <FrameRate>{}</FrameRate>\n", esc(r));
            }
            s +=
                "      <ShutterSpeed>1/500</ShutterSpeed>\n      <kV>80</kV>\n      <EDR>1</EDR>\n";
            s += "      <mA>100</mA>\n      <SID>1200</SID>\n      <MagLevel>1</MagLevel>\n";
            s += "      <Radiationtype>continuous</Radiationtype>\n      <Pulsewidth>0</Pulsewidth>\n";
            s += "      <FileDescription>fixture</FileDescription>\n      <Lab>Test lab</Lab>\n";
            s += "    </metadata>\n  </File>\n";
        }
        s += "</XMALab_Files>\n";
        s
    }

    /// The archive's bytes: the metadata file first, then the contents, then the raw
    /// entries.
    ///
    /// # Panics
    /// Never for the archives built here (each below 4 GiB).
    #[must_use]
    pub fn to_zip(&self) -> Vec<u8> {
        let meta = self.metadata_xml();
        let mut names: Vec<(String, &[u8])> = Vec::new();
        if !self.omit_metadata {
            names.push((
                format!("{}/{}", self.top, super::METADATA_FILE),
                meta.as_bytes(),
            ));
        }
        for (n, d) in &self.contents {
            names.push((format!("{}/{n}", self.top), d));
        }
        for (n, d) in &self.raw_entries {
            names.push((n.clone(), d));
        }
        let modified = DosDateTime::from_fields(2026, 11, 16, 10, 0, 0).unwrap_or_default();
        let entries: Vec<NewEntry<'_>> = names
            .iter()
            .map(|(n, d)| NewEntry {
                name: n.as_bytes(),
                data: d,
                modified,
            })
            .collect();
        write_archive(&entries, b"").expect("a fixture archive fits a zip")
    }

    /// Writes the archive to `path`.
    ///
    /// # Errors
    /// The file cannot be written.
    pub fn write(&self, path: &Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_zip())
    }
}

fn read(dir: &Path, name: &str) -> std::io::Result<Vec<u8>> {
    std::fs::read(dir.join(name))
}

/// A zip of `images` named `frame0001.tif`, ...: an image sequence as the Portal holds one.
fn sequence_zip(image: &[u8], n: usize) -> Vec<u8> {
    let names: Vec<String> = (1..=n).map(|i| format!("frame{i:04}.tif")).collect();
    let modified = DosDateTime::from_fields(2026, 11, 16, 10, 0, 0).unwrap_or_default();
    let entries: Vec<NewEntry<'_>> = names
        .iter()
        .map(|nm| NewEntry {
            name: nm.as_bytes(),
            data: image,
            modified,
        })
        .collect();
    write_archive(&entries, b"").expect("a fixture archive fits a zip")
}

/// A trial with two cameras: the pig videos, both at 250 frames per second.
///
/// # Errors
/// The pig files cannot be read.
pub fn pig_trial(pig: &Path) -> std::io::Result<Fixture> {
    Ok(Fixture {
        top: "PigStudy_Trial01".into(),
        study_name: "Pig Study".into(),
        study_id: 49,
        trial_name: "Trial01".into(),
        trial_id: 101,
        files: vec![
            MetaFile::new("Trial01_cam1.avi", 1001, "Trial", 1).rate("250"),
            MetaFile::new("Trial01_cam2.avi", 1002, "Trial", 2).rate("250"),
        ],
        contents: vec![
            ("Trial01_cam1.avi".into(), read(pig, "Camera1.avi")?),
            ("Trial01_cam2.avi".into(), read(pig, "Camera2.avi")?),
        ],
        ..Fixture::default()
    })
}

/// A trial of image sequences: per camera a zip of three copies of its calibration image,
/// with the second camera listed first and rates that differ.
///
/// # Errors
/// The pig files cannot be read.
pub fn pig_trial_sequences(pig: &Path) -> std::io::Result<Fixture> {
    let c1 = read(pig, "Calibration_cam1.tif")?;
    let c2 = read(pig, "Calibration_cam2.tif")?;
    Ok(Fixture {
        top: "PigStudy_Trial02".into(),
        study_name: "Pig Study".into(),
        study_id: 49,
        trial_name: "Trial02".into(),
        trial_id: 102,
        files: vec![
            MetaFile::new("Images/Trial02_cam2.zip", 1004, "Trial", 2).rate("500"),
            MetaFile::new("Images/Trial02_cam1.zip", 1003, "Trial", 1).rate("250"),
        ],
        contents: vec![
            ("Images/Trial02_cam1.zip".into(), sequence_zip(&c1, 3)),
            ("Images/Trial02_cam2.zip".into(), sequence_zip(&c2, 3)),
        ],
        ..Fixture::default()
    })
}

/// A calibration project of two cameras with the cube files and both grids.
///
/// # Errors
/// The pig files cannot be read.
pub fn pig_project_cube(pig: &Path) -> std::io::Result<Fixture> {
    Ok(Fixture {
        top: "PigStudy_Calibration".into(),
        study_name: "Pig Study".into(),
        study_id: 49,
        trial_name: "Calibration01".into(),
        trial_id: 100,
        files: vec![
            MetaFile::new("Calibration_cam1.tif", 901, "XCalibObject", 1),
            MetaFile::new("Calibration_cam2.tif", 902, "CalibObject", 2),
            MetaFile::new("undistortionGrid_cam1.tif", 903, "XCalibGrid", 1),
            MetaFile::new("undistortionGrid_cam2.tif", 904, "DistortionGrid", 2),
        ],
        contents: vec![
            (
                "Calibration_cam1.tif".into(),
                read(pig, "Calibration_cam1.tif")?,
            ),
            (
                "Calibration_cam2.tif".into(),
                read(pig, "Calibration_cam2.tif")?,
            ),
            (
                "undistortionGrid_cam1.tif".into(),
                read(pig, "undistortionGrid_cam1.tif")?,
            ),
            (
                "undistortionGrid_cam2.tif".into(),
                read(pig, "undistortionGrid_cam2.tif")?,
            ),
            (
                "FrameSpecification.csv".into(),
                read(pig, "FrameSpecification.csv")?,
            ),
            (
                "Reference_points.ref".into(),
                read(pig, "Reference_points.ref")?,
            ),
        ],
        ..Fixture::default()
    })
}

/// [`pig_project_cube`] without the `.csv` and `.ref` files (XMALab chooses Checkerboard).
///
/// # Errors
/// The pig files cannot be read.
pub fn pig_project_checkerboard(pig: &Path) -> std::io::Result<Fixture> {
    let mut f = pig_project_cube(pig)?;
    f.contents
        .retain(|(n, _)| !n.ends_with(".csv") && !n.ends_with(".ref"));
    Ok(f)
}

/// [`pig_project_cube`] whose second camera has no grid (XMALab makes it a light camera).
///
/// # Errors
/// The pig files cannot be read.
pub fn pig_project_light_camera(pig: &Path) -> std::io::Result<Fixture> {
    let mut f = pig_project_cube(pig)?;
    f.files
        .retain(|m| m.filename.as_deref() != Some("undistortionGrid_cam2.tif"));
    f.contents.retain(|(n, _)| n != "undistortionGrid_cam2.tif");
    Ok(f)
}

/// The broken archives: what each is, and the archive.
///
/// - `no_metadata`: the trial without its metadata file.
/// - `camera_without_files`: the trial whose second camera's `File` has no `Filename`.
/// - `unknown_elements`: the trial with elements no loader reads (it imports as the trial).
/// - `climbing_entry`: the trial with an entry `<top>/../escaped.txt`.
/// - `absolute_entry`: the trial with an entry `/tmp/xma_portal_absolute.txt`.
/// - `malformed_metadata`: the trial whose metadata ends inside a `File` element.
/// - `empty`: a zip with no entries.
/// - `climbing_trial_name`: the trial whose `TrialName` is `../Escaped`.
///
/// # Errors
/// The pig files cannot be read.
pub fn broken(pig: &Path) -> std::io::Result<Vec<(&'static str, Fixture)>> {
    let base = pig_trial(pig)?;
    let mut no_metadata = base.clone();
    no_metadata.omit_metadata = true;
    let mut no_files = base.clone();
    no_files.files[1].filename = None;
    let mut unknown = base.clone();
    unknown.unknown_elements = true;
    let mut climbing = base.clone();
    climbing
        .raw_entries
        .push((format!("{}/../escaped.txt", base.top), b"out".to_vec()));
    let mut absolute = base.clone();
    absolute
        .raw_entries
        .push(("/tmp/xma_portal_absolute.txt".into(), b"out".to_vec()));
    let mut malformed = base.clone();
    let text = base.metadata_xml();
    let cut = text.find("</File>").unwrap_or(text.len());
    malformed.metadata_override = Some(text[..cut].to_owned());
    let empty = Fixture {
        omit_metadata: true,
        ..Fixture::default()
    };
    let mut climbing_name = base;
    climbing_name.trial_name = "../Escaped".into();
    Ok(vec![
        ("no_metadata", no_metadata),
        ("camera_without_files", no_files),
        ("unknown_elements", unknown),
        ("climbing_entry", climbing),
        ("absolute_entry", absolute),
        ("malformed_metadata", malformed),
        ("empty", empty),
        ("climbing_trial_name", climbing_name),
    ])
}
