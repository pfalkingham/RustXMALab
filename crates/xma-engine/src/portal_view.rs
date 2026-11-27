//! The XMA Portal study data a project or trial carries (kept as raw XML lines in the
//! model), read the way the C++ reads it: for Project Metadata, Project Overview and
//! `xma_metadata.xml` ([`crate::portal_summary`]).
//!
//! Ported from core/Project.cpp, Project::parseXMLData; core/Trial.cpp,
//! Trial::parseXMLData; core/VideoStream.cpp, VideoStream::parseXMLData. The lines are joined
//! without separators (`xml_data.join("")`) and read with the XML reader
//! ([`xma_io::xml::tokenize_lenient`]: comments, CDATA, entities and character references
//! as `QXmlStreamReader` reads them), up to the first error, as `QXmlStreamReader` stops
//! there; `readElementText()` meeting a child element is such an error
//! ([`xma_io::xml::read_element_text`]).

use xma_io::cparse::{qt_to_double, qt_to_int};
use xma_io::portal::MAX_METADATA_TOKENS;
use xma_io::xml::{Token, read_element_text, tokenize_lenient_limited};

fn tokens_of(lines: &[String]) -> Vec<Token> {
    tokenize_lenient_limited(&lines.concat(), MAX_METADATA_TOKENS).0
}

fn to_int(s: &str) -> i64 {
    i64::from(qt_to_int(s))
}

/// The study fields of a project or trial (`Project::parseXMLData`, `Trial::parseXMLData`).
/// A field appearing several times keeps its last value, as in the C++. The numbers are -1
/// without study data, as `Project::Project` and `Trial::Trial` leave them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Study {
    pub repository: String,
    pub study_name: String,
    pub study_id: i64,
    pub trial_name: String,
    pub trial_id: i64,
    pub trial_number: i64,
    pub trial_type: String,
    pub lab: String,
    pub attribcomment: String,
    pub ts: String,
    pub trial_date: String,
}

impl Default for Study {
    fn default() -> Self {
        Self {
            repository: String::new(),
            study_name: String::new(),
            study_id: -1,
            trial_name: String::new(),
            trial_id: -1,
            trial_number: -1,
            trial_type: String::new(),
            lab: String::new(),
            attribcomment: String::new(),
            ts: String::new(),
            trial_date: String::new(),
        }
    }
}

impl Study {
    /// `Trial::parseXMLData`: the study fields of a trial's metadata.
    #[must_use]
    pub fn parse(lines: &[String]) -> Self {
        Self::read(lines, false)
    }

    /// `Project::parseXMLData`: the same fields of the project's metadata, whose reading
    /// also stops at a `cameraNumber` with a child element (it reads them too).
    #[must_use]
    pub fn parse_project(lines: &[String]) -> Self {
        Self::read(lines, true)
    }

    fn read(lines: &[String], project: bool) -> Self {
        let tokens = tokens_of(lines);
        let mut s = Study::default();
        for (k, t) in tokens.iter().enumerate() {
            let Token::Start { name, .. } = t else {
                continue;
            };
            let field = match name.as_str() {
                "Repository" | "StudyName" | "StudyID" | "TrialName" | "TrialID"
                | "TrialNumber" | "TrialType" | "lab" | "attribcomment" | "ts" | "trialDate" => {
                    true
                }
                "cameraNumber" => project,
                _ => false,
            };
            if !field {
                continue;
            }
            let (text, ok) = read_element_text(&tokens, k);
            match name.as_str() {
                "Repository" => s.repository = text,
                "StudyName" => s.study_name = text,
                "StudyID" => s.study_id = to_int(&text),
                "TrialName" => s.trial_name = text,
                "TrialID" => s.trial_id = to_int(&text),
                "TrialNumber" => s.trial_number = to_int(&text),
                "TrialType" => s.trial_type = text,
                "lab" => s.lab = text,
                "attribcomment" => s.attribcomment = text,
                "ts" => s.ts = text,
                "trialDate" => s.trial_date = text,
                _ => {}
            }
            if !ok {
                break;
            }
        }
        s
    }
}

/// The portal data of one video file (`VideoStream`'s fields; empty, `FileID` -1 and
/// frame rate 0 when there is none).
#[derive(Clone, Debug, PartialEq)]
pub struct VideoInfo {
    pub filename: String,
    pub file_id: i64,
    pub file_category: String,
    pub camera_number: String,
    pub instrument: String,
    pub frame_rate: f64,
    pub shutter_speed: String,
    pub kv: String,
    pub edr: String,
    pub ma: String,
    pub sid: String,
    pub mag_level: String,
    pub radiation_type: String,
    pub pulse_width: String,
    pub file_description: String,
    pub lab: String,
    /// `VideoStream::getPortalID`: the camera number matched, or -1.
    pub portal_id: i64,
}

impl Default for VideoInfo {
    fn default() -> Self {
        Self {
            filename: String::new(),
            file_id: -1,
            file_category: String::new(),
            camera_number: String::new(),
            instrument: String::new(),
            frame_rate: 0.0,
            shutter_speed: String::new(),
            kv: String::new(),
            edr: String::new(),
            ma: String::new(),
            sid: String::new(),
            mag_level: String::new(),
            radiation_type: String::new(),
            pulse_width: String::new(),
            file_description: String::new(),
            lab: String::new(),
            portal_id: -1,
        }
    }
}

/// One `File` of the second pass of `VideoStream::parseXMLData`, with whether its reading
/// ended without error.
fn read_video_file(tokens: &[Token], start: usize) -> (VideoInfo, usize, bool) {
    let mut v = VideoInfo::default();
    let mut i = start + 1;
    while i < tokens.len() && !tokens[i].is_end("File") {
        if let Token::Start { name, .. } = &tokens[i] {
            match name.as_str() {
                "Filename" | "FileID" | "FileCategory" => {
                    let (t, ok) = read_element_text(tokens, i);
                    match name.as_str() {
                        "Filename" => v.filename = t,
                        "FileID" => v.file_id = to_int(&t),
                        _ => v.file_category = t,
                    }
                    if !ok {
                        return (v, i, false);
                    }
                }
                "metadata" => {
                    let mut k = i + 1;
                    while k < tokens.len() && !tokens[k].is_end("metadata") {
                        if let Token::Start { name, .. } = &tokens[k] {
                            let slot: Option<&mut String> = match name.as_str() {
                                "cameraNumber" => Some(&mut v.camera_number),
                                "Instrument" => Some(&mut v.instrument),
                                "ShutterSpeed" => Some(&mut v.shutter_speed),
                                "kV" => Some(&mut v.kv),
                                "EDR" => Some(&mut v.edr),
                                "mA" => Some(&mut v.ma),
                                "SID" => Some(&mut v.sid),
                                "MagLevel" => Some(&mut v.mag_level),
                                "Radiationtype" => Some(&mut v.radiation_type),
                                "Pulsewidth" => Some(&mut v.pulse_width),
                                "FileDescription" => Some(&mut v.file_description),
                                "Lab" => Some(&mut v.lab),
                                _ => None,
                            };
                            let is_rate = name == "FrameRate";
                            if slot.is_some() || is_rate {
                                let (t, ok) = read_element_text(tokens, k);
                                if is_rate {
                                    v.frame_rate = qt_to_double(&t);
                                } else if let Some(sl) = slot {
                                    *sl = t;
                                }
                                if !ok {
                                    return (v, k, false);
                                }
                            }
                        }
                        k += 1;
                    }
                    i = k;
                }
                _ => {}
            }
        }
        i += 1;
    }
    (v, i, true)
}

/// The portal data of video `id` of a trial (`VideoStream::parseXMLData`): the file whose
/// camera number is the camera's portal id, or, when the camera has none, the `id`-th
/// smallest camera number of the data's files. Every `cameraNumber` element counts, an empty
/// one as camera 0 (`QString::toInt("")`), and a `File` without one has camera 0 too (its
/// `cameraNumber_tmp` is reset to ""). The matched file's `cameraNumber` is kept without
/// "cam" (`QString::replace` changes the string it is called on).
#[must_use]
pub fn video(lines: &[String], id: usize, camera_portal_id: i32) -> VideoInfo {
    let tokens = tokens_of(lines);
    let mut portal = i64::from(camera_portal_id);
    if portal == -1 {
        // First pass: the camera numbers inside the files' `metadata`.
        let mut set = std::collections::BTreeSet::new();
        let mut i = 0;
        'pass: while i < tokens.len() {
            if tokens[i].is_start("File") {
                let mut k = i + 1;
                let mut in_meta = false;
                while k < tokens.len() && !tokens[k].is_end("File") {
                    if tokens[k].is_start("metadata") {
                        in_meta = true;
                    } else if tokens[k].is_end("metadata") {
                        in_meta = false;
                    } else if in_meta && tokens[k].is_start("cameraNumber") {
                        let (t, ok) = read_element_text(&tokens, k);
                        set.insert(to_int(&t.replace("cam", "")));
                        if !ok {
                            break 'pass;
                        }
                    }
                    k += 1;
                }
                i = k;
            }
            i += 1;
        }
        // `*std::next(s.begin(), id)`; past the end the C++ reads out of range (none here).
        if let Some(p) = set.iter().nth(id) {
            portal = *p;
        }
    }
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i].is_start("File") {
            let (mut v, end, ok) = read_video_file(&tokens, i);
            let number = to_int(&v.camera_number.replace("cam", ""));
            if number == portal {
                v.camera_number = v.camera_number.replace("cam", "");
                v.portal_id = number;
                return v;
            }
            if !ok {
                break;
            }
            i = end;
        }
        i += 1;
    }
    VideoInfo::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_study_and_files() {
        let xml = vec![
            "<?xml version=\"1.0\"?><Trial><Repository>R</Repository><StudyName>S &amp; T</StudyName>".to_owned(),
            "<StudyID>12</StudyID><File><Filename>a.cine</Filename><FileID>3</FileID><metadata><cameraNumber>cam2</cameraNumber><FrameRate>250</FrameRate></metadata></File>".to_owned(),
            "<File><Filename>b.cine</Filename><metadata><cameraNumber>cam1</cameraNumber></metadata></File></Trial>".to_owned(),
        ];
        let s = Study::parse(&xml);
        assert_eq!(s.repository, "R");
        assert_eq!(s.study_name, "S & T");
        assert_eq!(s.study_id, 12);
        let v0 = video(&xml, 0, -1);
        assert_eq!(v0.filename, "b.cine");
        assert_eq!(v0.portal_id, 1);
        let v1 = video(&xml, 1, -1);
        assert_eq!(v1.filename, "a.cine");
        assert_eq!(v1.file_id, 3);
        assert!((v1.frame_rate - 250.0).abs() < 1e-12);
        assert_eq!(video(&[], 0, -1), VideoInfo::default());
    }

    /// An empty `<cameraNumber></cameraNumber>` is camera 0 (`QString::toInt("")`), in the
    /// set of numbers and in the match; a comment holding `>`, CDATA and entities are read
    /// as `QXmlStreamReader` reads them.
    #[test]
    fn empty_camera_number_comments_cdata_entities() {
        let xml = vec![
            "<d><!-- a > b --><StudyName><![CDATA[x<y]]> &amp; &#233;</StudyName>".to_owned(),
            "<File><Filename>z.avi</Filename><metadata><cameraNumber></cameraNumber></metadata></File>".to_owned(),
            "<File><Filename>one.avi</Filename><metadata><cameraNumber>cam1</cameraNumber></metadata></File></d>".to_owned(),
        ];
        let s = Study::parse(&xml);
        assert_eq!(s.study_name, "x<y & \u{e9}");
        let v0 = video(&xml, 0, -1);
        assert_eq!(v0.filename, "z.avi");
        assert_eq!(v0.portal_id, 0);
        assert_eq!(v0.camera_number, "");
        assert_eq!(video(&xml, 1, -1).filename, "one.avi");
        // A child element in a field: Qt's readElementText stops there with an error, and
        // the reading ends (StudyID after it is not read).
        let bad = vec!["<d><StudyName>a<b/>c</StudyName><StudyID>5</StudyID></d>".to_owned()];
        let s = Study::parse(&bad);
        assert_eq!(s.study_name, "a");
        assert_eq!(s.study_id, -1);
    }
}
