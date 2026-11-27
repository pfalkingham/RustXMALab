//! `xma_metadata.xml`, the XMA Portal summary XMALab writes into every saved project, and
//! the Project Metadata window's tree.
//!
//! Ported from ui/ProjectFileIO.cpp, ProjectFileIO::writePortalFile (`:1166-1421`), and
//! ui/MetaDataInfo.cpp, MetaDataInfo::update (`:61-127`), with the statistics they read:
//! core/Project.cpp (`getFileID`, the study fields), core/Trial.cpp (`getFirstTrackedFrame`,
//! `getLastTrackedFrame`, `getReprojectionError`, `getMarkerToMarkerSD`), core/Marker.cpp,
//! core/RigidBody.cpp, core/Camera.cpp and core/CalibrationImage.cpp (`getCalibrationError`,
//! `getCalibrationNbInlier`), core/VideoStream.cpp (`parseXMLData`).
//!
//! Numbers are written with `QString::number` (`%g`, six significant digits). The element
//! `Camera ` of the undistortion list is written as Qt writes `writeStartElement("Camera ")`:
//! `<Camera  camera-id=...>` and `</Camera >`. That is well-formed XML (an end tag may hold
//! white space before `>`, and attributes may be separated by more than one space), read by
//! any XML reader as the element `Camera`; XMALab never reads the file back, so the bytes are
//! the same in both modes (ledger F24, F63).

use xma_io::cfmt::format_qt_number;
use xma_io::export::precision::{
    body_error_3d, body_frames_tracked, body_marker_to_marker_sd, marker_frames_tracked,
    marker_reprojection_error, trial_marker_to_marker_sd,
};
use xma_io::portal::MAX_METADATA_TOKENS;
#[cfg(test)]
use xma_io::xml::tokenize;
use xma_io::xml::{Token, XmlWriter, element_text, tokenize_lenient_limited};
use xma_model::{
    CalibrationObject, CalibrationType, CppCompat, Marker, MarkerStatus, Project, RigidBody, Trial,
};

use crate::Document;
use crate::portal_view::{Study, video};

/// `PROJECT_VERSION` of XMALab 3, which `writePortalFile` writes as `xmalab-version`.
pub const XMALAB_VERSION: &str = "3.0.0";

fn file_name(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or("").to_owned()
}

/// `Project::getFileID`: in the project's metadata (lines joined), the first `Filename`
/// whose text is `filename`, then the first `FileID` after it; -1 otherwise.
///
/// The metadata is read with the token cap of every other reader (ledger F78): past
/// `MAX_METADATA_TOKENS` tokens nothing is read, as the Project Metadata window reads it.
#[must_use]
pub fn file_id(project: &Project, filename: &str) -> i64 {
    file_id_in(&metadata_tokens(project), filename)
}

/// The project's metadata as [`file_id`] reads it: QXmlStreamReader stops at the first
/// error, so metadata held twice (ledger F60) is searched in its first copy; at most
/// `MAX_METADATA_TOKENS` tokens (ledger F78). Read once per `xma_metadata.xml`.
fn metadata_tokens(project: &Project) -> Vec<Token> {
    if project.portal_metadata.is_empty() {
        return Vec::new();
    }
    tokenize_lenient_limited(&project.portal_metadata.concat(), MAX_METADATA_TOKENS).0
}

/// [`file_id`] in the tokens of [`metadata_tokens`].
fn file_id_in(tokens: &[Token], filename: &str) -> i64 {
    for (i, t) in tokens.iter().enumerate() {
        if t.is_start("Filename") && element_text(tokens, i) == filename {
            for (k, u) in tokens.iter().enumerate().skip(i + 1) {
                if let Token::Start { name, .. } = u
                    && name == "FileID"
                {
                    return element_text(tokens, k).trim().parse().unwrap_or(0);
                }
            }
        }
    }
    -1
}

/// `Marker::getFirstTrackedFrame(cam)` (1-based) or -1.
fn marker_first(m: &Marker, cam: usize) -> i64 {
    m.views.get(cam).map_or(-1, |v| {
        v.status
            .iter()
            .position(|s| *s >= MarkerStatus::Tracked)
            .map_or(-1, |i| i as i64 + 1)
    })
}

/// `Marker::getLastTrackedFrame(cam)` (1-based) or -1.
fn marker_last(m: &Marker, cam: usize) -> i64 {
    m.views.get(cam).map_or(-1, |v| {
        v.status
            .iter()
            .rposition(|s| *s >= MarkerStatus::Tracked)
            .map_or(-1, |i| i as i64 + 1)
    })
}

/// `Marker::getFramesTracked(cam)`.
fn marker_tracked(m: &Marker, cam: usize) -> usize {
    m.views.get(cam).map_or(0, |v| {
        v.status
            .iter()
            .filter(|s| **s >= MarkerStatus::Tracked)
            .count()
    })
}

/// `Marker::getReprojectionError()` over the whole trial (ledger A25 in the XMALab 3 mode).
fn marker_error(m: &Marker, compat: CppCompat) -> f64 {
    marker_reprojection_error(
        m,
        0,
        m.status3d.len(),
        compat.reprojection_error_all_cameras,
    )
    .0
}

/// `Trial::getFirstTrackedFrame` / `getLastTrackedFrame`.
fn trial_range(t: &Trial, n_cameras: usize, last: bool) -> String {
    (0..n_cameras)
        .map(|c| {
            let f = t
                .markers
                .iter()
                .map(|m| {
                    if last {
                        marker_last(m, c)
                    } else {
                        marker_first(m, c)
                    }
                })
                .filter(|f| *f >= 0)
                .reduce(|a, b| if last { a.max(b) } else { a.min(b) });
            f.map_or_else(|| "NA".to_owned(), |f| f.to_string())
        })
        .collect::<Vec<_>>()
        .join(" / ")
}

/// `Trial::getReprojectionError`.
fn trial_error(t: &Trial, compat: CppCompat) -> f64 {
    let mut count = 0usize;
    let mut error = 0.0;
    for m in &t.markers {
        let c = marker_frames_tracked(m);
        error += c as f64 * marker_error(m, compat);
        count += c;
    }
    if count != 0 {
        error /= count as f64;
    }
    error
}

fn na(v: f64) -> String {
    if v <= 0.0 {
        "NA".to_owned()
    } else {
        format_qt_number(v)
    }
}

/// `RigidBody::getFirstTrackedFrame`, `getLastTrackedFrame` (1-based, -1 for none) and
/// `getFramesTracked`: every frame in the XMALab 3 mode (`poseComputed >= 0` always holds,
/// ledger A24, U40), else the frames with a computed pose.
fn body_tracked(b: &RigidBody, compat: CppCompat) -> (i64, i64, usize) {
    let n = body_frames_tracked(b, compat.frames_tracked_all);
    if compat.frames_tracked_all {
        let len = b.raw.valid.len() as i64;
        return if len == 0 { (-1, -1, n) } else { (1, len, n) };
    }
    let first = b
        .raw
        .valid
        .iter()
        .position(|v| *v)
        .map_or(-1, |i| i as i64 + 1);
    let last = b
        .raw
        .valid
        .iter()
        .rposition(|v| *v)
        .map_or(-1, |i| i as i64 + 1);
    (first, last, n)
}

fn marker_element(w: &mut XmlWriter, t: &Trial, idx: usize, n_cameras: usize, compat: CppCompat) {
    let m = &t.markers[idx];
    w.start_element("Marker");
    w.attribute("name", &m.description);
    w.attribute("id", &(idx + 1).to_string());
    let join = |f: &dyn Fn(usize) -> String| (0..n_cameras).map(f).collect::<Vec<_>>().join(" / ");
    let na_i = |v: i64| {
        if v >= 0 {
            v.to_string()
        } else {
            "NA".to_owned()
        }
    };
    w.attribute("firstTracked", &join(&|c| na_i(marker_first(m, c))));
    w.attribute("lastTracked", &join(&|c| na_i(marker_last(m, c))));
    w.attribute(
        "numberTracked",
        &join(&|c| marker_tracked(m, c).to_string()),
    );
    w.attribute("reprojectionerror", &na(marker_error(m, compat)));
    w.end_element();
}

/// The text of `xma_metadata.xml` for a save of `trials` (by index; the Default trial is
/// left out). `date_modified` is the save's time in ISO form with its offset
/// (`2026-11-16T10:10:45Z`, `...+01:00`).
///
/// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::writePortalFile (`:1166-1421`).
#[must_use]
pub fn xma_metadata_xml(doc: &Document, trials: &[usize], date_modified: &str) -> String {
    let p = doc.project();
    let compat = doc.compat();
    let study = Study::parse_project(&p.portal_metadata);
    let n_cameras = p.cameras.len();
    let mut w = XmlWriter::new();
    w.start_document();
    w.start_element("data");
    let leaf = |w: &mut XmlWriter, name: &str, text: &str| {
        w.start_element(name);
        w.characters(text);
        w.end_element();
    };
    leaf(&mut w, "xmalab-version", XMALAB_VERSION);
    leaf(&mut w, "repository", &study.repository);
    leaf(&mut w, "studyid", &study.study_id.to_string());
    leaf(&mut w, "studyname", &study.study_name);
    leaf(&mut w, "date-created", &p.date_created);
    leaf(&mut w, "date-modified", date_modified);
    if p.calibration_type != CalibrationType::None {
        // Read once for every `FileId` below (review 12).
        let metadata = metadata_tokens(p);
        w.start_element("CalibrationTrial");
        w.attribute("id", &study.trial_id.to_string());
        w.attribute("name", &study.trial_name);
        w.start_element("Undistortion");
        for (c, cam) in p.cameras.iter().enumerate() {
            if let Some(u) = &cam.undistortion {
                // `writeStartElement("Camera ")`, as Qt writes it.
                w.start_element("Camera ");
                w.attribute("camera-id", &c.to_string());
                w.attribute("camera-portalid", &cam.portal_id.to_string());
                w.attribute(
                    "FileId",
                    &file_id_in(&metadata, &u.image_filename).to_string(),
                );
                w.characters(&u.image_filename);
                w.end_element();
            }
        }
        w.end_element();
        w.start_element("Calibration");
        for (c, cam) in p.cameras.iter().enumerate() {
            // CalibrationImage::getCalibrationNbInlier and getCalibrationError, per image.
            let per_image: Vec<(usize, f64, String)> = (0..cam.calibration_images.len())
                .map(|i| {
                    let v =
                        crate::view::calibration::calibration_image(doc, c, i).unwrap_or_default();
                    let (mut n_in, mut sum) = (0usize, 0.0);
                    if v.calibrated == 1 {
                        for (k, e) in v.error.iter().enumerate() {
                            if v.inlier.get(k).is_some_and(|&x| x > 0) {
                                n_in += 1;
                                sum += e;
                            }
                        }
                    }
                    let n_inlier = if v.calibrated == 1 {
                        v.inlier.iter().filter(|&&x| x > 0).count()
                    } else {
                        0
                    };
                    let err = if n_in > 0 { sum / n_in as f64 } else { 0.0 };
                    (n_inlier, err, cam.calibration_images[i].filename.clone())
                })
                .collect();
            let n_all: usize = per_image.iter().map(|(n, _, _)| n).sum();
            let mut mean = 0.0;
            for (n, e, _) in &per_image {
                mean += e * *n as f64;
            }
            if n_all > 0 {
                mean /= n_all as f64;
            }
            w.start_element("Camera");
            w.attribute("camera-id", &c.to_string());
            w.attribute("camera-portalid", &cam.portal_id.to_string());
            w.attribute("points", &n_all.to_string());
            w.attribute("error", &format_qt_number(mean));
            for (n, e, f) in &per_image {
                w.start_element("File");
                w.attribute("FileId", &file_id_in(&metadata, f).to_string());
                w.attribute("points", &n.to_string());
                w.attribute("error", &format_qt_number(*e));
                w.characters(f);
                w.end_element();
            }
            w.end_element();
        }
        w.start_element("Object");
        match &p.calibration_object {
            Some(CalibrationObject::Checkerboard {
                horizontal_squares,
                vertical_squares,
                square_size,
                ..
            }) => {
                w.attribute("type", "checkerboard");
                w.attribute("width", &horizontal_squares.to_string());
                w.attribute("height", &vertical_squares.to_string());
                w.attribute("size", &format_qt_number(*square_size));
            }
            other => {
                let (csv, refs) = match other {
                    Some(CalibrationObject::Frame(f)) => (
                        file_name(&f.frame_specification_filename),
                        file_name(&f.references_filename),
                    ),
                    _ => (String::new(), String::new()),
                };
                w.attribute("type", "cube");
                w.attribute("csv", &csv);
                w.attribute("csv-FileId", &file_id_in(&metadata, &csv).to_string());
                w.attribute("ref", &refs);
                w.attribute("ref-FileId", &file_id_in(&metadata, &refs).to_string());
            }
        }
        w.end_element();
        w.end_element();
        w.end_element();
    }
    for &ti in trials {
        let Some(t) = p.trials.get(ti) else { continue };
        if t.is_default {
            continue;
        }
        let ts = Study::parse(&t.portal_metadata);
        w.start_element("Trial");
        w.attribute("id", &ts.trial_id.to_string());
        w.attribute("name", &ts.trial_name);
        w.attribute("xmalab-name", &t.name);
        w.attribute("firstTracked", &trial_range(t, n_cameras, false));
        w.attribute("lastTracked", &trial_range(t, n_cameras, true));
        w.attribute("filterfrequency", &format_qt_number(t.cutoff_frequency));
        w.attribute("framerate", &format_qt_number(t.recording_speed));
        w.attribute("reprojectionerror", &na(trial_error(t, compat)));
        w.attribute("averageSD", &na(trial_marker_to_marker_sd(t, compat)));
        w.start_element("Movies");
        for (m, v) in t.videos.iter().enumerate() {
            let portal = p.cameras.get(m).map_or(-1, |c| c.portal_id);
            let info = if t.portal_metadata.is_empty() {
                crate::portal_view::VideoInfo::default()
            } else {
                video(&t.portal_metadata, m, portal)
            };
            w.start_element("File");
            w.attribute("camera-id", &m.to_string());
            w.attribute("camera-portalid", &info.portal_id.to_string());
            w.attribute("FileId", &info.file_id.to_string());
            w.attribute("FileName", &info.filename);
            w.characters(&file_name(v.filenames.first().map_or("", String::as_str)));
            w.end_element();
        }
        w.end_element();
        let mut written = vec![false; t.markers.len()];
        for b in &t.rigid_bodies {
            w.start_element("RigidBody");
            w.attribute("name", &b.description);
            let (first, last, n) = body_tracked(b, compat);
            let na_f = |v: i64| {
                if v < 0 {
                    "NA".to_owned()
                } else {
                    v.to_string()
                }
            };
            w.attribute("firstTracked", &na_f(first));
            w.attribute("lastTracked", &na_f(last));
            w.attribute("numberTracked", &n.to_string());
            let (sd, _) = body_marker_to_marker_sd(t, b, 0, usize::MAX, compat);
            w.attribute("averageSD", &na(sd));
            let len = b.raw.valid.len();
            w.attribute("error3DUnfiltered", &na(body_error_3d(b, false, 0, len)));
            let flen = b.filtered.valid.len();
            w.attribute("error3DFiltered", &na(body_error_3d(b, true, 0, flen)));
            let freq = if b.override_cutoff_frequency {
                b.cutoff_frequency
            } else {
                t.cutoff_frequency
            };
            w.attribute("filterfrequency", &format_qt_number(freq));
            for &idx in &b.marker_indices {
                if idx < t.markers.len() {
                    written[idx] = true;
                    marker_element(&mut w, t, idx, n_cameras, compat);
                }
            }
            w.end_element();
        }
        for (idx, done) in written.iter().enumerate() {
            if !done {
                marker_element(&mut w, t, idx, n_cameras, compat);
            }
        }
        w.end_element();
    }
    w.end_element();
    w.end_document()
}

/// `date-modified` as XMALab writes it: local time with its offset from UTC, `Z` for none
/// (`QDateTime::toString(Qt::ISODate)` after `setOffsetFromUtc`).
#[must_use]
pub fn iso_with_offset(local: &str, offset_seconds: i32) -> String {
    if offset_seconds == 0 {
        return format!("{local}Z");
    }
    let sign = if offset_seconds < 0 { '-' } else { '+' };
    let a = offset_seconds.unsigned_abs();
    format!("{local}{sign}{:02}:{:02}", a / 3600, (a % 3600) / 60)
}

// ---------------------------------------------------------------------------------------
// Project Metadata

/// One row of the Project Metadata tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetaNode {
    pub label: String,
    pub value: String,
    pub children: Vec<MetaNode>,
}

impl MetaNode {
    fn new(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            children: Vec::new(),
        }
    }
}

fn study_rows(item: &mut MetaNode, s: &Study) {
    for (l, v) in [
        ("Repository:", s.repository.clone()),
        ("Studyname:", s.study_name.clone()),
        ("StudyID", s.study_id.to_string()),
        ("TrialName", s.trial_name.clone()),
        ("TrialID", s.trial_id.to_string()),
        ("Trialnumber", s.trial_number.to_string()),
        ("Trialtype", s.trial_type.clone()),
        ("Lab", s.lab.clone()),
        ("Comment", s.attribcomment.clone()),
        ("Timestamp", s.ts.clone()),
        ("TrialDate", s.trial_date.clone()),
    ] {
        item.children.push(MetaNode::new(l, v));
    }
}

/// The Project Metadata tree's top-level items.
///
/// Ported from ui/MetaDataInfo.cpp, MetaDataInfo::update (`:61-127`).
#[must_use]
pub fn metadata_tree(p: &Project) -> Vec<MetaNode> {
    let mut out = Vec::new();
    if p.calibration_type != CalibrationType::None {
        let mut item = MetaNode::new("Calibration", "");
        study_rows(&mut item, &Study::parse_project(&p.portal_metadata));
        out.push(item);
    }
    for t in &p.trials {
        if t.is_default {
            continue;
        }
        let mut it = MetaNode::new("Trial", t.name.clone());
        study_rows(&mut it, &Study::parse(&t.portal_metadata));
        for j in 0..t.videos.len() {
            let portal = p.cameras.get(j).map_or(-1, |c| c.portal_id);
            let v = if t.portal_metadata.is_empty() {
                crate::portal_view::VideoInfo::default()
            } else {
                video(&t.portal_metadata, j, portal)
            };
            let id = if v.portal_id == -1 {
                j as i64 + 1
            } else {
                v.portal_id
            };
            let mut cam = MetaNode::new(format!("Camera{id}"), "");
            for (l, val) in [
                ("Filename:", v.filename),
                ("FileID:", v.file_id.to_string()),
                ("FileCategory", v.file_category),
                ("cameraNumber", v.camera_number),
                ("Instrument", v.instrument),
                ("FrameRate", format_qt_number(v.frame_rate)),
                ("ShutterSpeed", v.shutter_speed),
                ("kV", v.kv),
                ("EDR", v.edr),
                ("mA", v.ma),
                ("SID", v.sid),
                ("MagLevel", v.mag_level),
                ("Radiationtype", v.radiation_type),
                ("Pulsewidth", v.pulse_width),
                ("FileDescription", v.file_description),
                ("Lab", v.lab),
            ] {
                cam.children.push(MetaNode::new(l, val));
            }
            it.children.push(cam);
        }
        out.push(it);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets() {
        assert_eq!(
            iso_with_offset("2026-11-16T10:00:00", 0),
            "2026-11-16T10:00:00Z"
        );
        assert_eq!(
            iso_with_offset("2026-11-16T10:00:00", 3600),
            "2026-11-16T10:00:00+01:00"
        );
        assert_eq!(
            iso_with_offset("2026-11-16T10:00:00", -16200),
            "2026-11-16T10:00:00-04:30"
        );
    }

    #[test]
    fn camera_element_with_a_space_is_well_formed() {
        let mut w = XmlWriter::new();
        w.start_document();
        w.start_element("Undistortion");
        w.start_element("Camera ");
        w.attribute("camera-id", "0");
        w.characters("g.tif");
        w.end_element();
        w.end_element();
        let text = w.end_document();
        assert!(
            text.contains("<Camera  camera-id=\"0\">g.tif</Camera >"),
            "{text}"
        );
        let toks = tokenize(&text).expect("well-formed");
        assert!(
            toks.iter()
                .any(|t| t.is_start("Camera") && t.attr("camera-id") == "0")
        );
    }
}
