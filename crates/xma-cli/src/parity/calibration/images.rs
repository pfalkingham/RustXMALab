//! The export and denoising references (`oracle/export_runs.json`,
//! `oracle/denoise_runs.json`; `oracle/FORMAT.md` "Calibration runs, version 6"): scripts
//! run through XMALab's own File > Export > Undistorted Trial images, All Trials for
//! External and the Trial dialog's "Denoise Images".
//!
//! [`run_images_parity`] replays a run's script in the XMALab 3 mode, as the calibration
//! runs are replayed ([`super::apply_op`] for the operations they share), with the
//! operations of package "images" applied here: `NewTrial` and `NewTrialNoCalibration`
//! with folders (`cam<c>dir`), `NewProject` with `flip<c>`, `SelectTrial`,
//! `CalibDistortion`, `ExportUndistortedTrialImages`, `ExportAllTrialsForExternal` and
//! `DenoiseTrial`. Every export and denoising writes into a scratch folder, which is then
//! compared with what XMALab wrote:
//!
//! - the set of folders and files, by relative path, exactly;
//! - TIFF and PNG files by their decoded size, colour type (channels and bit depth) and
//!   pixels, exactly;
//! - JPEG files (whose bytes are another encoder's, ledger V31) by their size, by the
//!   pixels before encoding wherever the reference shows them (the TIFF of the same frames
//!   and filter in the same run, the lossless copy `hooks/denoise_frames/` of every
//!   denoised frame), and by their decoded pixels within [`JPEG_BOUND`] grey levels;
//! - text files (MayaCams, 2D points, reprojection errors) byte for byte;
//! - for the denoising: which frames are copied, denoised and never written
//!   (`hooks/denoise_frames.csv`, `denoise_calls.csv`), and the parameters the jobs ran
//!   with (`denoise_params.csv`).
//!
//! Every `Dump` and `Save` is compared as in the calibration runs; in a saved project the
//! denoised files' folder is this program's scratch folder, so the trial's file names are
//! compared with that folder written as the reference's.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use xma_engine::create::{CalibrationTarget, NewCameraSpec, NewProjectSpec, NewTrialSpec};
use xma_engine::denoise::{DenoiseJob, DenoiseReport, DenoiseSettings, odd, relink_spec};
use xma_engine::tracking::TrialSources;
use xma_engine::trial_images::{AllTrialsJob, TrialImageFormat, TrialImagesJob, TrialImagesSpec};
use xma_engine::undistort_sequence::folder_images;
use xma_engine::{CalibrationEdit, Document, Edit, NoCalibrationSpec};
use xma_model::{CppCompat, SettingValue};
use xma_oracle::{JsonValue, Manifest};
use xma_vision::visual_filter::VisualFilter;

use super::super::edits::{OpRow, Point, SizeRules, compare_save};
use super::super::{Check, Item, Tier, Verdict};
use super::{
    CalibOp, CalibParityReport, CalibRun, OpStatus, ReplayState, apply_op, compare_dump, created,
    doc_of, engopt, parse_calib_line, read_ops_log, run_options, run_wizard,
};

/// The largest difference, in grey levels, accepted between a JPEG file decoded and the
/// pixels it was made from, or between two JPEG files of the same pixels decoded. It rests
/// on the encoders, not on the port: XMALab's files are libjpeg's at quality 95, this
/// program's the `image` crate's at quality 95 (ledger V31). Measured on the reference
/// frames: libjpeg's own files differ from the pixels they were made from by up to 10 grey
/// levels (x01, XMALab's JPEG against XMALab's TIFF of the same frames, filtered images),
/// the `image` crate's by up to 11, and the two encoders' files of the same pixels by up
/// to 9; the bound is the larger encoder error plus one.
pub const JPEG_BOUND: u8 = 12;

/// The rule for the queue of `ThreadScheduler` in a dump.
const RULE_SCHEDULER: &str = "not compared: the queue of XMALab's ThreadScheduler depends on      when its update jobs finish (oracle/DATASETS.md, the format 6 runs)";
/// The rule for the reference calibration XMALab's Trial dialog writes when it opens.
const RULE_U62: &str = "the reference calibration XMALab's Trial dialog writes when it opens      (-1 without a frame calibrated in every camera); this program's dialog writes nothing      on opening (ledger U62, fixed in both modes)";
/// The rule for the distortion coefficients beyond the eighth.
const RULE_DIST14: &str = "XMALab holds 14 distortion coefficients after a calibration with      the lens model, the last six 0; this program's camera holds 8 (the same values)";

/// The runs of `oracle/export_runs.json` and `oracle/denoise_runs.json`.
///
/// # Errors
/// When a file cannot be read or does not have the expected layout.
pub fn image_runs() -> Result<Vec<CalibRun>, String> {
    let root = xma_oracle::oracle_root();
    let mut out = Vec::new();
    for file in ["export_runs.json", "denoise_runs.json"] {
        let path = root.join(file);
        if !path.is_file() {
            continue;
        }
        let manifest = Manifest::try_read(&path).map_err(|e| e.to_string())?;
        let runs = manifest
            .json()
            .get("runs")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{}: no 'runs' list", path.display()))?;
        let field = |s: &JsonValue, k: &str| -> Result<String, String> {
            s.get(k)
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .ok_or_else(|| format!("{}: a run without '{k}'", path.display()))
        };
        for r in runs {
            let output = root.join(field(r, "output")?);
            let parent = output.parent().map(Path::to_path_buf).unwrap_or_default();
            let list = |k: &str, sub: Option<&str>| -> Vec<String> {
                let v = r.get(k);
                let v = match sub {
                    Some(s) => v.and_then(|x| x.get(s)),
                    None => v,
                };
                v.and_then(|x| x.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| match x {
                                JsonValue::String(s) => Some(s.clone()),
                                o => o.get("file").and_then(|f| f.as_str()).map(str::to_owned),
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            };
            out.push(CalibRun {
                name: field(r, "name")?,
                start_project: r
                    .get("start_project")
                    .and_then(|v| v.as_str())
                    .map(|p| root.join(p)),
                script: root.join(field(r, "script")?),
                repeats: list("repeats", None)
                    .into_iter()
                    .filter(|x| x != "a")
                    .map(|x| parent.join(x))
                    .collect(),
                unstable: list("repeatability", Some("differences")),
                checker: false,
                output,
            });
        }
    }
    Ok(out)
}

/// The run of that name, if one of the two files lists it.
#[must_use]
pub fn find_image_run(name: &str) -> Option<CalibRun> {
    image_runs().ok()?.into_iter().find(|r| r.name == name)
}

// ---------------------------------------------------------------------------------------
// Script
// ---------------------------------------------------------------------------------------

/// An operation of an export or denoising script.
#[derive(Clone, Debug, PartialEq)]
enum Op {
    /// An operation the calibration runs share.
    Calib(CalibOp),
    /// `NewTrial` with `cam<c>dir`: each camera's images, a folder.
    NewTrialDirs {
        dirs: Vec<PathBuf>,
        name: Option<String>,
    },
    /// `NewTrialNoCalibration` with `cam<c>dir`.
    NoCalibrationDirs {
        dirs: Vec<PathBuf>,
    },
    /// `NewProject` (cube target) with `flip<c>`.
    NewProjectFlip {
        grids: Vec<Option<PathBuf>>,
        images: Vec<Vec<PathBuf>>,
        framespec: PathBuf,
        refs: PathBuf,
        flips: Vec<bool>,
    },
    SelectTrial(usize),
    CalibDistortion {
        camera: Option<usize>,
        image: Option<usize>,
        on: bool,
    },
    ExportImages {
        dir: String,
        from: usize,
        to: usize,
        format: TrialImageFormat,
        filtered: bool,
        trial: Option<usize>,
    },
    ExportAll {
        dir: String,
    },
    Denoise {
        dir: String,
        relink: bool,
        /// search, template, temporal, strength.
        params: Option<(i32, i32, i32, f64)>,
        trial: Option<usize>,
    },
}

impl Op {
    fn name(&self) -> String {
        match self {
            Op::Calib(c) => c.name().to_owned(),
            Op::NewTrialDirs { .. } => "NewTrial".into(),
            Op::NoCalibrationDirs { .. } => "NewTrialNoCalibration".into(),
            Op::NewProjectFlip { .. } => "NewProject".into(),
            Op::SelectTrial(_) => "SelectTrial".into(),
            Op::CalibDistortion { .. } => "CalibDistortion".into(),
            Op::ExportImages { .. } => "ExportUndistortedTrialImages".into(),
            Op::ExportAll { .. } => "ExportAllTrialsForExternal".into(),
            Op::Denoise { .. } => "DenoiseTrial".into(),
        }
    }
}

/// The decoded fields of a line.
fn fields(text: &str) -> Result<(String, Vec<(String, String)>), String> {
    let mut tokens = text.split(' ').filter(|t| !t.is_empty());
    let name = tokens.next().ok_or("empty line")?.to_owned();
    let mut list = Vec::new();
    for t in tokens {
        let (k, v) = t
            .split_once('=')
            .ok_or_else(|| format!("token without '=': {t}"))?;
        list.push((k.to_owned(), xma_engine::edit::script::percent_decode(v)?));
    }
    Ok((name, list))
}

fn parse_line(text: &str, base: &Path) -> Result<Op, String> {
    let (name, list) = fields(text)?;
    let get = |k: &str| list.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
    let need = |k: &str| get(k).ok_or_else(|| format!("missing field {k}"));
    let num = |k: &str| -> Result<usize, String> {
        need(k)?
            .parse()
            .map_err(|_| format!("field {k} is not a number"))
    };
    let opt_num = |k: &str| -> Result<Option<usize>, String> { get(k).map(|_| num(k)).transpose() };
    let flag = |k: &str| -> Result<bool, String> {
        match need(k)? {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            v => Err(format!("field {k} is not true or false: {v}")),
        }
    };
    let path = |p: &str| {
        let p = PathBuf::from(p);
        if p.is_absolute() { p } else { base.join(p) }
    };
    let dirs = || -> Vec<PathBuf> {
        (0..)
            .map_while(|c| get(&format!("cam{c}dir")).map(path))
            .collect()
    };
    Ok(match name.as_str() {
        "NewTrial" if get("cam0dir").is_some() => Op::NewTrialDirs {
            dirs: dirs(),
            name: get("name").map(str::to_owned),
        },
        "NewTrialNoCalibration" if get("cam0dir").is_some() => {
            Op::NoCalibrationDirs { dirs: dirs() }
        }
        "NewProject" if list.iter().any(|(k, _)| k.starts_with("flip")) => {
            let CalibOp::NewProject {
                grids,
                images,
                framespec,
                refs,
            } = parse_calib_line(text, base)?
            else {
                return Err("flip<c> with a checkerboard target is not replayed".into());
            };
            let flips = (0..grids.len())
                .map(|c| get(&format!("flip{c}")) == Some("true"))
                .collect();
            Op::NewProjectFlip {
                grids,
                images,
                framespec,
                refs,
                flips,
            }
        }
        "SelectTrial" => Op::SelectTrial(num("trial")?),
        "CalibDistortion" => Op::CalibDistortion {
            camera: opt_num("camera")?,
            image: opt_num("image")?,
            on: flag("on")?,
        },
        "ExportUndistortedTrialImages" => Op::ExportImages {
            dir: need("dir")?.to_owned(),
            from: num("from")?,
            to: num("to")?,
            format: TrialImageFormat::parse(need("format")?)
                .ok_or_else(|| "format: jpg, tif or avi".to_owned())?,
            filtered: flag("filtered")?,
            trial: opt_num("trial")?,
        },
        "ExportAllTrialsForExternal" => Op::ExportAll {
            dir: need("dir")?.to_owned(),
        },
        "DenoiseTrial" => {
            let params = match get("search") {
                Some(_) => Some((
                    num("search")? as i32,
                    num("template")? as i32,
                    num("temporal")? as i32,
                    need("strength")?
                        .parse::<f64>()
                        .map_err(|_| "strength is not a number".to_owned())?,
                )),
                None => None,
            };
            Op::Denoise {
                dir: need("dir")?.to_owned(),
                relink: flag("relink")?,
                params,
                trial: opt_num("trial")?,
            }
        }
        _ => Op::Calib(parse_calib_line(text, base)?),
    })
}

fn read_script(path: &Path) -> Result<Vec<(usize, Op)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    let mut out = Vec::new();
    for (i, l) in text.lines().enumerate() {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let op =
            parse_line(l, base).map_err(|e| format!("{}: line {}: {e}", path.display(), i + 1))?;
        out.push((i + 1, op));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------------------

/// What the replay keeps besides the calibration replay's state.
struct Images {
    /// Where this program writes (the reference's `<out>`).
    scratch: PathBuf,
    /// The reference run's output folder.
    reference: PathBuf,
    /// The output folder XMALab wrote into (`hooks/denoise_params.csv`, `output_directory`
    /// without its last part), for the file names a relinked trial holds.
    reference_out: Option<String>,
    /// A `DenoiseTrial` ran (XMALab opened the Trial dialog, U62).
    denoised: bool,
}

/// Each camera's files of `trial` as the project stores them.
fn trial_files(doc: &Document, trial: usize) -> Vec<Vec<PathBuf>> {
    TrialSources::new(doc, trial)
        .map(|s| s.files().iter().map(|f| f.to_vec()).collect())
        .unwrap_or_default()
}

/// `CameraBoxTrial::on_toolButtonImage_clicked`: every `*.png *.tif *.bmp *.jpeg *.jpg`
/// of the folder, natural order.
fn folder_files(dirs: &[PathBuf]) -> Vec<Vec<PathBuf>> {
    dirs.iter().map(|d| folder_images(d, true)).collect()
}

fn apply(state: &mut ReplayState, im: &mut Images, op: &Op) -> (OpStatus, Option<Vec<Item>>) {
    match op {
        Op::Calib(c) => (apply_op(state, c), None),
        Op::NewTrialDirs { dirs, name } => {
            let doc = match doc_of(state) {
                Ok(d) => d,
                Err(e) => return (e, None),
            };
            let spec = NewTrialSpec {
                name: name
                    .clone()
                    .unwrap_or_else(|| format!("Trial {}", doc.project().trials.len() + 1)),
                cameras: folder_files(dirs),
            };
            (edit_status(doc.apply(&Edit::NewTrial(spec))), None)
        }
        Op::NoCalibrationDirs { dirs } => {
            let spec = NoCalibrationSpec {
                camera_names: (0..dirs.len())
                    .map(|c| format!("Camera {}", c + 1))
                    .collect(),
                trial: NewTrialSpec {
                    name: "Trial 1".to_owned(),
                    cameras: folder_files(dirs),
                },
            };
            let r = Document::new_without_calibration(&spec, &state.options);
            (created(state, r), None)
        }
        Op::NewProjectFlip {
            grids,
            images,
            framespec,
            refs,
            flips,
        } => {
            let spec = NewProjectSpec {
                cameras: images
                    .iter()
                    .zip(grids)
                    .enumerate()
                    .map(|(c, (im, g))| {
                        let mut im = im.clone();
                        im.sort_by_key(|p| p.to_string_lossy().into_owned());
                        NewCameraSpec {
                            name: format!("Camera {}", c + 1),
                            images: im,
                            light_camera: false,
                            grid: g.clone(),
                            flipped: flips.get(c).copied().unwrap_or(false),
                        }
                    })
                    .collect(),
                target: CalibrationTarget::Cube {
                    frame_specification: framespec.clone(),
                    references: refs.clone(),
                },
            };
            let r = Document::new_project(&spec, &state.options);
            (created(state, r), None)
        }
        Op::SelectTrial(t) => match doc_of(state) {
            Ok(doc) if *t < doc.project().trials.len() => {
                doc.select_trial(*t);
                (OpStatus::Ok(String::new()), None)
            }
            Ok(_) => (OpStatus::Invalid(format!("no trial {t}")), None),
            Err(e) => (e, None),
        },
        Op::CalibDistortion { camera, image, on } => {
            let doc = match doc_of(state) {
                Ok(d) => d,
                Err(e) => return (e, None),
            };
            let s = doc.session();
            let camera = camera.unwrap_or(s.active_camera);
            let image = image.unwrap_or(s.active_frame_calibration);
            doc.set_active_calibration_image(camera, image);
            let r = doc.apply(&Edit::Calibration(CalibrationEdit::SetLensModel {
                camera,
                on: *on,
            }));
            (edit_status(r), None)
        }
        Op::ExportImages {
            dir,
            from,
            to,
            format,
            filtered,
            trial,
        } => export_images(state, im, dir, (*from, *to), *format, *filtered, *trial),
        Op::ExportAll { dir } => {
            let doc = match doc_of(state) {
                Ok(d) => d,
                Err(e) => return (e, None),
            };
            let mine = im.scratch.join(dir);
            let _ = std::fs::create_dir_all(&mine);
            let d: &Document = doc;
            let job = AllTrialsJob::new(d, &mine, &|t| trial_files(d, t), VisualFilter::default());
            let r = job.run(&mut |_| {}, &|| false);
            let status = match &r.error {
                Some(e) => OpStatus::Invalid(e.clone()),
                None => OpStatus::Ok(r.problems.join(" | ")),
            };
            (
                status,
                Some(compare_tree(&im.reference.join(dir), &mine, &im.reference)),
            )
        }
        Op::Denoise {
            dir,
            relink,
            params,
            trial,
        } => denoise(state, im, dir, *relink, *params, *trial),
    }
}

fn edit_status(r: xma_engine::EditResult) -> OpStatus {
    match r {
        Ok(o) => OpStatus::Ok(o.messages.join(" | ")),
        Err(xma_engine::EditError::Rejected(m)) => OpStatus::Rejected(m),
        Err(e) => OpStatus::Invalid(e.to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
fn export_images(
    state: &mut ReplayState,
    im: &mut Images,
    dir: &str,
    (from, to): (usize, usize),
    format: TrialImageFormat,
    filtered: bool,
    trial: Option<usize>,
) -> (OpStatus, Option<Vec<Item>>) {
    let doc = match doc_of(state) {
        Ok(d) => d,
        Err(e) => return (e, None),
    };
    let trial = trial.unwrap_or(doc.project().active_trial);
    let spec = TrialImagesSpec {
        trial,
        first: from.saturating_sub(1),
        last: to.saturating_sub(1),
        format,
        filtered,
        filter: VisualFilter::default(),
        camera: None,
    };
    let files = trial_files(doc, trial);
    let mine = im.scratch.join(dir);
    let job = match TrialImagesJob::new(doc, spec.clone(), &files, &mine) {
        Ok(j) => j,
        Err(e) => return (OpStatus::Invalid(e), None),
    };
    let r = job.run(&mut |_| {}, &|| false);
    let mut items = compare_tree(&im.reference.join(dir), &mine, &im.reference);
    if format == TrialImageFormat::Jpg {
        // The pixels before encoding: the same export as TIFF, beside.
        let side = im.scratch.join("_pixels").join(dir);
        let tspec = TrialImagesSpec {
            format: TrialImageFormat::Tif,
            ..spec
        };
        if let Ok(j) = TrialImagesJob::new(doc, tspec, &files, &side) {
            j.run(&mut |_| {}, &|| false);
            items.extend(jpeg_sources(&im.reference, dir, &side, &mine));
        }
    }
    // XMALab 3 writes no video (V27): every camera's error is that, and "ok" as XMALab's.
    let errors: Vec<String> = r
        .cameras
        .iter()
        .filter_map(|c| c.error.clone())
        .filter(|e| !(format == TrialImageFormat::Avi && e.contains("codec -1")))
        .collect();
    let status = if errors.is_empty() {
        OpStatus::Ok(String::new())
    } else {
        OpStatus::Invalid(errors.join(" | "))
    };
    (status, Some(items))
}

fn denoise(
    state: &mut ReplayState,
    im: &mut Images,
    dir: &str,
    relink: bool,
    params: Option<(i32, i32, i32, f64)>,
    trial: Option<usize>,
) -> (OpStatus, Option<Vec<Item>>) {
    let doc = match doc_of(state) {
        Ok(d) => d,
        Err(e) => return (e, None),
    };
    let trial = trial.unwrap_or(doc.project().active_trial);
    let mut settings = DenoiseSettings {
        relink,
        ..DenoiseSettings::default()
    };
    if let Some((search, template, temporal, strength)) = params {
        // The dialog's spin boxes raise an even value to the next odd one.
        settings.search = odd(search);
        settings.template = odd(template);
        settings.temporal = odd(temporal);
        settings.strength = strength;
    }
    let files = trial_files(doc, trial);
    let mine = im.scratch.join(dir);
    let mut reports = Vec::new();
    let mut frames: Vec<Vec<WrittenFrame>> = Vec::new();
    let mut jobs = Vec::new();
    for c in 0..doc.project().cameras.len() {
        let f = files.get(c).cloned().unwrap_or_default();
        let job = match DenoiseJob::new(doc, trial, c, &f, &mine, &settings) {
            Ok(j) => j,
            Err(e) => return (OpStatus::Invalid(e), None),
        };
        let mut got = Vec::new();
        let base = xma_engine::undistort_sequence::complete_base_name(
            &xma_engine::undistort_sequence::file_basename(
                &f.iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect::<Vec<_>>(),
            ),
        );
        let r = job.run_with(&mut |_| {}, &|| false, &mut |fr, w, h, px| {
            got.push((fr, w, h, px.to_vec(), job.copied(fr)));
            let path = xma_engine::denoise::output_path(&mine, c, &base, fr);
            if let Some(p) = path.parent() {
                let _ = std::fs::create_dir_all(p);
            }
            xma_video::avi_writer::encode_jpeg(w, h, xma_video::avi_writer::Samples::Gray(px))
                .map_err(|e| e.to_string())
                .and_then(|b| std::fs::write(&path, b).map_err(|e| e.to_string()))
                .map(|()| path)
        });
        reports.push(r);
        frames.push(got);
        jobs.push(job);
    }
    let n_frames = jobs.first().map_or(0, DenoiseJob::frames);
    let mut notes: Vec<String> = reports
        .iter()
        .filter_map(|r| {
            r.error
                .as_ref()
                .map(|e| format!("camera {}: {e}", r.camera))
        })
        .collect();
    let status = if relink && xma_engine::denoise::relink_due(doc, &reports, n_frames) {
        match relink_spec(doc, trial, &mine, &reports) {
            Some(spec) => match edit_status(doc.apply(&Edit::ChangeTrialData { trial, spec })) {
                OpStatus::Ok(m) => {
                    notes.push(m);
                    OpStatus::Ok(notes.join(" | "))
                }
                other => other,
            },
            None => OpStatus::Invalid("the trial is gone".into()),
        }
    } else {
        OpStatus::Ok(notes.join(" | "))
    };
    let mut items = compare_tree(&im.reference.join(dir), &mine, &im.reference);
    let compat = doc.compat();
    items.extend(compare_denoise_hooks(
        &im.reference,
        &frames,
        &reports,
        &jobs,
        &settings,
        compat,
        (&im.reference.join(dir), &mine),
    ));
    if im.reference_out.is_none() {
        im.reference_out = reference_out_dir(&im.reference, dir);
    }
    im.denoised = true;
    (status, Some(items))
}

/// `output_directory` of `hooks/denoise_params.csv`: the folder XMALab wrote into, the
/// run's output folder as it was named when the reference was made, plus `dir`.
fn reference_out_dir(reference: &Path, dir: &str) -> Option<String> {
    let text = std::fs::read_to_string(reference.join("hooks/denoise_params.csv")).ok()?;
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next()?.split(',').collect();
    let col = head.iter().position(|h| *h == "output_directory")?;
    let row: Vec<&str> = lines.next()?.split(',').collect();
    let out = row.get(col)?;
    out.strip_suffix(dir)
        .map(|s| s.trim_end_matches('/').to_owned())
}

// ---------------------------------------------------------------------------------------
// Comparison of the files written
// ---------------------------------------------------------------------------------------

/// Every folder and file under `dir`, relative, folders ending in `/`.
fn tree(dir: &Path) -> BTreeSet<String> {
    fn walk(base: &Path, d: &Path, out: &mut BTreeSet<String>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            // Files written whole are renamed from `.<name>.part`; none should remain.
            let rel = p
                .strip_prefix(base)
                .map(|r| r.to_string_lossy().replace('\\', "/"))
                .unwrap_or(name);
            if p.is_dir() {
                out.insert(format!("{rel}/"));
                walk(base, &p, out);
            } else {
                out.insert(rel);
            }
        }
    }
    let mut out = BTreeSet::new();
    walk(dir, dir, &mut out);
    out
}

/// A decoded image: width, height, colour type and samples (8 or 16 bits as bytes).
type Decoded = (u32, u32, String, Vec<u8>);

/// A frame the denoising job handed its write callback: frame, width and height as the job
/// reports them, the samples, and whether it was copied rather than denoised.
type WrittenFrame = (usize, u32, u32, Vec<u8>, bool);

fn decode(path: &Path) -> Result<Decoded, String> {
    use xma_video::image_file::{read_colour_image, read_image};
    let err = |e: xma_video::VideoError| format!("{}: {e}", path.display());
    if let Some(c) = read_colour_image(path).map_err(err)? {
        let rgb: Vec<u8> = c.rgba.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        return Ok((c.width, c.height, "Rgb8".into(), rgb));
    }
    let f = read_image(path).map_err(err)?;
    Ok(match &f.pixels {
        xma_video::Pixels::Gray8(d) => (f.width, f.height, "L8".into(), d.clone()),
        xma_video::Pixels::Gray16(d) => (
            f.width,
            f.height,
            format!("L16 ({} bits)", f.bit_depth),
            d.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    })
}

/// The largest difference of two 8-bit sample arrays and where (sample index).
fn max_diff(a: &[u8], b: &[u8]) -> (u8, usize, usize) {
    let mut worst = (0u8, 0usize);
    let mut n = 0;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = x.abs_diff(*y);
        if d > 0 {
            n += 1;
        }
        if d > worst.0 {
            worst = (d, i);
        }
    }
    (worst.0, worst.1, n)
}

/// The folders and files of `mine` against `reference`, then each common file.
fn compare_tree(reference: &Path, mine: &Path, run_dir: &Path) -> Vec<Item> {
    let want = tree(reference);
    let got = tree(mine);
    let mut head = Check::new("(folders and files)".to_owned(), Tier::Read);
    let only_here: Vec<&String> = got.difference(&want).collect();
    let only_there: Vec<&String> = want.difference(&got).collect();
    for p in only_here.iter().take(6) {
        head.problem(format!("written here, not by XMALab: {p}"));
    }
    for p in only_there.iter().take(6) {
        head.problem(format!("written by XMALab, not here: {p}"));
    }
    if only_here.len() + only_there.len() > 12 {
        head.problem(format!(
            "{} only here and {} only in the reference in all",
            only_here.len(),
            only_there.len()
        ));
    }
    head.int("paths", got.len() as i64, want.len() as i64);
    head.equal_values += want.intersection(&got).count();
    let mut items = vec![head.finish()];
    for rel in want.intersection(&got).filter(|p| !p.ends_with('/')) {
        items.push(compare_file(
            &reference.join(rel),
            &mine.join(rel),
            rel,
            run_dir,
        ));
    }
    items
}

fn ext(path: &Path) -> String {
    path.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

fn compare_file(reference: &Path, mine: &Path, name: &str, _run: &Path) -> Item {
    let mut c = Check::new(name.to_owned(), Tier::Read);
    match ext(reference).as_str() {
        "tif" | "tiff" | "png" | "bmp" => match (decode(reference), decode(mine)) {
            (Ok(w), Ok(g)) => {
                compare_pixels(&mut c, &g, &w, "pixels");
                c.finish()
            }
            (Err(e), _) | (_, Err(e)) => {
                c.problem(e);
                c.finish()
            }
        },
        "jpg" | "jpeg" => match (decode(reference), decode(mine)) {
            (Ok(w), Ok(g)) => jpeg_item(c, &g, &w, "this program's JPEG and XMALab's, decoded"),
            (Err(e), _) | (_, Err(e)) => {
                c.problem(e);
                c.finish()
            }
        },
        "txt" if name.contains("_MayaCam") => {
            match (
                std::fs::read_to_string(mine),
                std::fs::read_to_string(reference),
            ) {
                (Ok(a), Ok(b)) if a == b => {
                    c.bytes_equal = Some(true);
                    c.equal_values += 1;
                }
                (Ok(a), Ok(b)) => {
                    // The numbers of a camera calibrated in this replay: within ledger A37.
                    // Text, number formatting, spacing and line ends exactly, as an export
                    // file's (the edit runs' comparison of a saved table).
                    c.bytes_equal = Some(false);
                    super::super::edits::numbers_within(
                        &mut c,
                        a.as_bytes(),
                        b.as_bytes(),
                        Some(super::super::TOLERANCE_A37_CUBE),
                    );
                }
                (Err(e), _) | (_, Err(e)) => c.problem(e.to_string()),
            }
            c.finish()
        }
        _ => {
            let (a, b) = (std::fs::read(mine), std::fs::read(reference));
            match (a, b) {
                (Ok(a), Ok(b)) => {
                    c.bytes_equal = Some(a == b);
                    if a != b {
                        let at = a.iter().zip(&b).position(|(x, y)| x != y).unwrap_or(0);
                        c.problem(format!(
                            "bytes differ (here {} bytes, XMALab {}), first at byte {at}",
                            a.len(),
                            b.len()
                        ));
                    }
                    c.equal_values += 1;
                }
                (Err(e), _) | (_, Err(e)) => c.problem(e.to_string()),
            }
            c.finish()
        }
    }
}

/// Size, colour type and every sample exactly.
fn compare_pixels(c: &mut Check, got: &Decoded, want: &Decoded, what: &str) {
    if (got.0, got.1) != (want.0, want.1) {
        c.problem(format!(
            "size {} x {}, XMALab {} x {}",
            got.0, got.1, want.0, want.1
        ));
        return;
    }
    if got.2 != want.2 {
        c.problem(format!("colour type {}, XMALab {}", got.2, want.2));
        return;
    }
    if got.3.len() != want.3.len() {
        c.problem(format!(
            "{what}: {} samples, XMALab {}",
            got.3.len(),
            want.3.len()
        ));
        return;
    }
    let (d, at, n) = max_diff(&got.3, &want.3);
    if n > 0 {
        let w = got.0.max(1) as usize;
        let cn = (got.3.len() / (got.0 as usize * got.1 as usize).max(1)).max(1);
        let px = at / cn;
        c.problem(format!(
            "{what}: {n} samples differ, by up to {d} (at x {}, y {})",
            px % w,
            px / w
        ));
    }
    c.equal_values += got.3.len() - n;
    if n == 0 {
        c.bytes_equal = Some(true);
    }
}

/// A JPEG compared within [`JPEG_BOUND`]: size and colour type exactly, the samples within
/// the bound, with the largest difference measured.
fn jpeg_item(mut c: Check, got: &Decoded, want: &Decoded, what: &str) -> Item {
    if (got.0, got.1, &got.2) != (want.0, want.1, &want.2) {
        c.problem(format!(
            "{} x {} {}, XMALab {} x {} {}",
            got.0, got.1, got.2, want.0, want.1, want.2
        ));
        return c.finish();
    }
    let (d, _, n) = max_diff(&got.3, &want.3);
    if d > JPEG_BOUND {
        c.problem(format!(
            "{what}: {n} samples differ, by up to {d} grey levels (more than {JPEG_BOUND})"
        ));
        return c.finish();
    }
    c.equal_values += got.3.len();
    c.bytes_equal = Some(true);
    let mut item = c.finish();
    if d > 0 {
        item.verdict = Verdict::WithinTolerance;
        item.max_abs = f64::from(d);
        item.max_abs_at = format!("{n} of {} samples differ", got.3.len());
        item.notes.push(format!(
            "{what}: within {JPEG_BOUND} grey levels (two JPEG encoders, ledger V31); largest \
             difference {d}, {n} of {} samples differ",
            got.3.len()
        ));
    }
    item
}

/// JPEG exports: the pixels this program encodes (the same export written as TIFF beside)
/// against the reference's TIFF of the same frames and filter when the run has one
/// (exactly), and against XMALab's JPEG decoded (within the bound).
fn jpeg_sources(reference_run: &Path, dir: &str, side: &Path, mine: &Path) -> Vec<Item> {
    let mut items = Vec::new();
    let twin = dir.replace("jpg", "tif");
    let twin_dir = reference_run.join(&twin);
    for rel in tree(side).into_iter().filter(|p| !p.ends_with('/')) {
        let Ok(px) = decode(&side.join(&rel)) else {
            continue;
        };
        let jpg_rel = rel.replace(".tif", ".jpg");
        let tif_ref = twin_dir.join(&rel);
        if tif_ref.is_file() {
            // libjpeg's own error: XMALab's JPEG against XMALab's TIFF of the same frame.
            if let (Ok(t), Ok(j)) = (
                decode(&tif_ref),
                decode(&reference_run.join(dir).join(&jpg_rel)),
            ) {
                let c = Check::new(
                    format!("{jpg_rel} (XMALab's JPEG against XMALab's TIFF {twin}/{rel})"),
                    Tier::Read,
                );
                items.push(jpeg_item(
                    c,
                    &j,
                    &t,
                    "XMALab's JPEG decoded and its TIFF of the same frame",
                ));
            }
            let mut c = Check::new(
                format!("{jpg_rel} (pixels before encoding, against {twin}/{rel})"),
                Tier::Read,
            );
            match decode(&tif_ref) {
                Ok(w) => compare_pixels(&mut c, &px, &w, "pixels before encoding"),
                Err(e) => c.problem(e),
            }
            items.push(c.finish());
        }
        if let Ok(w) = decode(&reference_run.join(dir).join(&jpg_rel)) {
            let c = Check::new(
                format!("{jpg_rel} (pixels before encoding, against XMALab's JPEG)"),
                Tier::Read,
            );
            items.push(jpeg_item(
                c,
                &px,
                &w,
                "the pixels encoded here and XMALab's JPEG decoded",
            ));
        }
        if let Ok(g) = decode(&mine.join(&jpg_rel)) {
            let c = Check::new(
                format!("{jpg_rel} (this program's JPEG against its own pixels)"),
                Tier::Read,
            );
            items.push(jpeg_item(
                c,
                &g,
                &px,
                "this program's JPEG decoded and the pixels it encoded",
            ));
        }
    }
    items
}

/// A CSV file of the reference as rows of named fields.
fn csv_rows(path: &Path) -> Vec<std::collections::BTreeMap<String, String>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut lines = text.lines();
    let Some(head) = lines.next() else {
        return Vec::new();
    };
    let head: Vec<String> = head.split(',').map(str::to_owned).collect();
    lines
        .map(|l| {
            head.iter()
                .cloned()
                .zip(l.split(',').map(str::to_owned))
                .collect()
        })
        .collect()
}

/// The denoising hooks: each frame written (mode and the pixels before `imwrite`, against
/// `hooks/denoise_frames/` and `denoise_frames.csv`), the frames never written and the
/// call that threw (`denoise_calls.csv`), the parameters (`denoise_params.csv`).
fn compare_denoise_hooks(
    reference: &Path,
    frames: &[Vec<WrittenFrame>],
    reports: &[DenoiseReport],
    jobs: &[DenoiseJob],
    settings: &DenoiseSettings,
    compat: CppCompat,
    dirs: (&Path, &Path),
) -> Vec<Item> {
    let hooks = reference.join("hooks");
    // The two encoders' own errors: each JPEG decoded against the pixels it was made from.
    let mut own = [(0u8, 0usize, 0usize), (0u8, 0usize, 0usize)];
    let mut items = Vec::new();
    // denoise_frames.csv and the lossless copies.
    let rows = csv_rows(&hooks.join("denoise_frames.csv"));
    let mut modes = Check::new(
        "hooks/denoise_frames.csv (frames written, mode)",
        Tier::Read,
    );
    let mut pixels = Check::new("hooks/denoise_frames/ (pixels before imwrite)", Tier::Read);
    let mut want_set = BTreeSet::new();
    for r in &rows {
        let (Some(c), Some(f)) = (
            r.get("camera").and_then(|v| v.parse::<usize>().ok()),
            r.get("frame").and_then(|v| v.parse::<usize>().ok()),
        ) else {
            continue;
        };
        want_set.insert((c, f));
        let Some((_, gw, gh, px, copied)) = frames.get(c).and_then(|v| v.iter().find(|x| x.0 == f))
        else {
            modes.problem(format!("camera {c} frame {f}: written by XMALab, not here"));
            continue;
        };
        let mode = if *copied { "copied" } else { "denoised" };
        modes.text(
            format!("camera {c} frame {f} mode"),
            mode,
            r.get("mode").map_or("", String::as_str),
        );
        modes.equal_values += 1;
        let png = hooks.join(r.get("png").map_or("", String::as_str));
        match decode(&png) {
            Ok((w, h, kind, want)) => {
                // The size the job gave the write callback, the reference's colour type
                // (the callback's frames are 8-bit grey, as the lossless copies).
                let got = (*gw, *gh, kind.clone(), px.clone());
                compare_pixels(
                    &mut pixels,
                    &got,
                    &(w, h, kind, want),
                    &format!("camera {c} frame {f}"),
                );
            }
            Err(e) => pixels.problem(e),
        }
        let rel: Vec<&str> = r
            .get("jpg_path")
            .map_or("", String::as_str)
            .rsplitn(3, '/')
            .collect();
        if rel.len() == 3 {
            let rel = format!("{}/{}", rel[1], rel[0]);
            let theirs = decode(&dirs.0.join(&rel));
            let ours = decode(&dirs.1.join(&rel));
            if let (Ok(t), Ok((_, _, _, png))) = (&theirs, decode(&png)) {
                let (d, _, n) = max_diff(&t.3, &png);
                own[0] = (own[0].0.max(d), own[0].1 + n, own[0].2 + png.len());
            }
            if let Ok(o) = &ours {
                let (d, _, n) = max_diff(&o.3, px);
                own[1] = (own[1].0.max(d), own[1].1 + n, own[1].2 + px.len());
            }
        }
    }
    for (c, v) in frames.iter().enumerate() {
        for (f, ..) in v {
            if !want_set.contains(&(c, *f)) {
                modes.problem(format!("camera {c} frame {f}: written here, not by XMALab"));
            }
        }
    }
    items.push(modes.finish());
    items.push(pixels.finish());
    for (k, what) in [
        "XMALab's JPEG files decoded against its pixels before imwrite (libjpeg, quality 95)",
        "this program's JPEG files decoded against its pixels before encoding (the image crate, quality 95)",
    ]
    .into_iter()
    .enumerate()
    {
        let (d, n, total) = own[k];
        let mut c = Check::new(format!("{what}: the encoder's own error"), Tier::Read);
        c.equal_values += total;
        c.bytes_equal = Some(true);
        let mut item = c.finish();
        item.verdict = if d > JPEG_BOUND {
            item.problems.push(format!("up to {d} grey levels, more than {JPEG_BOUND}"));
            Verdict::Different
        } else if d > 0 {
            Verdict::WithinTolerance
        } else {
            Verdict::Identical
        };
        item.max_abs = f64::from(d);
        item.max_abs_at = format!("{n} of {total} samples differ");
        item.notes.push(format!("largest difference {d} grey levels, {n} of {total} samples differ"));
        items.push(item);
    }
    // denoise_calls.csv: a `before` without `after` is the call OpenCV refused.
    let calls = csv_rows(&hooks.join("denoise_calls.csv"));
    let mut threw = Check::new("hooks/denoise_calls.csv (the call that threw)", Tier::Read);
    for (c, r) in reports.iter().enumerate() {
        let befores: BTreeSet<usize> = calls
            .iter()
            .filter(|x| x.get("camera").and_then(|v| v.parse().ok()) == Some(c))
            .filter(|x| x.get("event").map(String::as_str) == Some("before"))
            .filter_map(|x| x.get("frame").and_then(|v| v.parse().ok()))
            .collect();
        let afters: BTreeSet<usize> = calls
            .iter()
            .filter(|x| x.get("camera").and_then(|v| v.parse().ok()) == Some(c))
            .filter(|x| x.get("event").map(String::as_str) == Some("after"))
            .filter_map(|x| x.get("frame").and_then(|v| v.parse().ok()))
            .collect();
        let want: Option<usize> = befores.difference(&afters).next().copied();
        let got = r
            .error
            .as_ref()
            .and_then(|e| e.strip_prefix("frame "))
            .and_then(|e| e.split(':').next())
            .and_then(|n| n.parse::<usize>().ok())
            .map(|n| n - 1);
        threw.text(
            format!("camera {c}: frame refused"),
            &got.map_or("none".into(), |f| f.to_string()),
            &want.map_or("none".into(), |f| f.to_string()),
        );
        threw.int(
            format!("camera {c}: calls made"),
            (r.denoised + usize::from(got.is_some())) as i64,
            befores.len() as i64,
        );
    }
    items.push(threw.finish());
    // denoise_params.csv.
    let params = csv_rows(&hooks.join("denoise_params.csv"));
    let mut p = Check::new("hooks/denoise_params.csv (parameters)", Tier::Read);
    let h = settings.params(compat);
    for (c, job) in jobs.iter().enumerate() {
        let Some(r) = params
            .iter()
            .find(|x| x.get("camera").and_then(|v| v.parse().ok()) == Some(c))
        else {
            p.problem(format!("camera {c}: no row"));
            continue;
        };
        let v = |k: &str| r.get(k).and_then(|x| x.parse::<i64>().ok()).unwrap_or(-999);
        p.int(
            format!("camera {c} n_images"),
            job.frames() as i64,
            v("n_images"),
        );
        p.int(
            format!("camera {c} window_size"),
            i64::from((settings.temporal - 1) / 2),
            v("window_size"),
        );
        p.int(
            format!("camera {c} search_window"),
            i64::from(settings.search),
            v("search_window"),
        );
        p.int(
            format!("camera {c} template_window"),
            i64::from(settings.template),
            v("template_window"),
        );
        p.int(
            format!("camera {c} temporal_window"),
            i64::from(settings.temporal),
            v("temporal_window"),
        );
        // The strength the routine gets (an `int` in XMALab, as entered here: A190).
        let want = v("filter_strength") as f64;
        if f64::from(h.h) != want {
            p.problem(format!(
                "camera {c} filter_strength: {}, XMALab {want}",
                h.h
            ));
        }
        p.equal_values += 1;
        p.int(
            format!("camera {c} relink"),
            i64::from(settings.relink),
            v("relink"),
        );
    }
    items.push(p.finish());
    items
}

// ---------------------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------------------

/// Replays one run in the XMALab 3 mode and compares everything it writes.
///
/// # Errors
/// When the script, manifest or start project cannot be read.
pub fn run_images_parity(run: &CalibRun) -> Result<CalibParityReport, String> {
    run_images_parity_in(run, CppCompat::XMALAB3, &scratch_dir(run, "xmalab3")).map(|(r, _)| r)
}

/// The scratch folder of a replay.
#[must_use]
pub fn scratch_dir(run: &CalibRun, mode: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "xma-parity-images-{}-{mode}-{}",
        run.name,
        std::process::id()
    ))
}

/// [`run_images_parity`] in another mode, writing into `scratch` (kept for the caller, which
/// removes it); returns the report and the document at the end.
///
/// # Errors
/// As [`run_images_parity`].
pub fn run_images_parity_in(
    run: &CalibRun,
    compat: CppCompat,
    scratch: &Path,
) -> Result<(CalibParityReport, Option<Document>), String> {
    let start = Instant::now();
    let lines = read_script(&run.script)?;
    let mut options = run_options(run)?;
    options.compat = compat;
    let _ = std::fs::remove_dir_all(scratch);
    std::fs::create_dir_all(scratch).map_err(|e| format!("{}: {e}", scratch.display()))?;
    let mut state = ReplayState {
        doc: None,
        seed: None,
        options,
        out_dir: run.output.clone(),
        engopt: engopt::EngoptState::for_run(run),
        wizard: run_wizard(run),
    };
    if let Some(p) = &run.start_project {
        let mut doc =
            Document::open_with(p, &state.options).map_err(|e| format!("{}: {e}", p.display()))?;
        doc.session_mut().wizard = state.wizard;
        state.doc = Some(doc);
    }
    let recompute_when_saving =
        state.options.overrides.get("RecomputeWhenSaving") == Some(SettingValue::Bool(true));
    let cpp_log = read_ops_log(&run.output.join("ops_log.csv"));
    let mut im = Images {
        scratch: scratch.to_path_buf(),
        reference: run.output.clone(),
        reference_out: None,
        denoised: false,
    };
    let mut ops = Vec::new();
    let mut points = Vec::new();
    let mut stopped: Option<(usize, String)> = None;
    for (line, op) in &lines {
        if let Op::Calib(CalibOp::Dump { name, model }) = op {
            let items = match (&stopped, state.doc.as_ref()) {
                (None, Some(doc)) => {
                    let mut items = compare_dump(run, name, doc, *model);
                    explain_dump(&run.output.join(name), doc, &mut items, im.denoised);
                    items
                }
                _ => Vec::new(),
            };
            points.push(Point {
                name: name.clone(),
                line: *line,
                skipped: stopped
                    .as_ref()
                    .map(|(l, o)| format!("after line {l} ({o})")),
                items,
            });
            continue;
        }
        if let Op::Calib(CalibOp::Save { name }) = op {
            let items = match (&stopped, state.doc.as_mut()) {
                (None, Some(doc)) => save_items(run, name, doc, &im, recompute_when_saving),
                _ => Vec::new(),
            };
            points.push(Point {
                name: name.clone(),
                line: *line,
                skipped: stopped
                    .as_ref()
                    .map(|(l, o)| format!("after line {l} ({o})")),
                items,
            });
            continue;
        }
        let (cpp_status, cpp_messages) = cpp_log
            .iter()
            .find(|(l, ..)| l == line)
            .map(|(_, _, st, m)| (st.clone(), m.clone()))
            .unwrap_or_else(|| ("(not logged)".to_owned(), String::new()));
        if stopped.is_some() {
            ops.push(OpRow {
                line: *line,
                op: op.name(),
                cpp_status,
                cpp_messages,
                status: "not implemented".to_owned(),
                message: "not reached".to_owned(),
                agrees: false,
            });
            continue;
        }
        state.engopt.line = *line;
        let (out, items) = apply(&mut state, &mut im, op);
        points.append(&mut state.engopt.points);
        if let Some(items) = items {
            let name = match op {
                Op::ExportImages { dir, .. } | Op::ExportAll { dir } | Op::Denoise { dir, .. } => {
                    dir.clone()
                }
                _ => op.name(),
            };
            points.push(Point {
                name,
                line: *line,
                items,
                skipped: None,
            });
        }
        if out == OpStatus::NotImplemented {
            stopped = Some((*line, op.name()));
        }
        let agrees = match &out {
            OpStatus::Ok(_) => cpp_status == "ok",
            OpStatus::Rejected(m) => super::rejection_agrees(&cpp_status, &cpp_messages, m),
            OpStatus::Invalid(_) => cpp_status == "invalid",
            OpStatus::NotImplemented => false,
        };
        ops.push(OpRow {
            line: *line,
            op: op.name(),
            cpp_status,
            cpp_messages,
            status: out.label().to_owned(),
            message: out.message().to_owned(),
            agrees,
        });
    }
    let doc = state.doc.take();
    Ok((
        CalibParityReport {
            run: run.clone(),
            ops,
            points,
            stopped,
            warnings: Vec::new(),
            seconds: start.elapsed().as_secs_f64(),
        },
        doc,
    ))
}

/// A `Save`: as the calibration runs compare it, the trial's files that lie in the scratch
/// folder named as the reference names them (the folder XMALab wrote into).
fn save_items(
    run: &CalibRun,
    name: &str,
    doc: &mut Document,
    im: &Images,
    recompute_when_saving: bool,
) -> Vec<Item> {
    let rules = SizeRules {
        calibration_tolerance: Some(super::super::TOLERANCE_A37_CUBE),
        ..SizeRules::default()
    };
    let path = run.output.join(name);
    let Some(reference_out) = &im.reference_out else {
        return compare_save(&path, name, doc, recompute_when_saving, &rules);
    };
    let scratch = im.scratch.to_string_lossy().into_owned();
    let mut p = doc.project().clone();
    let mut renamed = 0;
    for t in &mut p.trials {
        let t = std::sync::Arc::make_mut(t);
        for v in &mut t.videos {
            for f in &mut v.filenames {
                if let Some(rest) = f.strip_prefix(&scratch) {
                    *f = format!("{reference_out}{rest}");
                    renamed += 1;
                }
            }
        }
    }
    let mut renamed_doc = Document::from_project(p, doc.compat());
    if let Some(dp) = doc.path() {
        renamed_doc.set_path(dp);
    }
    renamed_doc.set_workspace(doc.session().workspace, true);
    let mut items = compare_save(&path, name, &mut renamed_doc, recompute_when_saving, &rules);
    if im.denoised {
        explain_project_xml(
            &path,
            name,
            &mut renamed_doc,
            recompute_when_saving,
            &mut items,
        );
    }
    if let Some(first) = items.first_mut() {
        first.notes.push(format!(
            "{renamed} file names of the relinked trial compared with this program's scratch \
             folder written as XMALab's output folder ({reference_out})"
        ));
    }
    items
}

/// Marks an item as explained by `rule`: its problems become notes.
fn explained(item: &mut Item, rule: &str) {
    item.notes
        .extend(item.problems.drain(..).map(|p| format!("{rule}: {p}")));
    item.rules.push((rule.to_owned(), 1));
    item.explained += 1;
    item.verdict = Verdict::IdenticalExplained;
}

/// The rules of a `Dump` that are not the port's: the scheduler's queue, the reference
/// calibration of U62 (after a denoising), the 14 distortion coefficients.
fn explain_dump(dir: &Path, doc: &Document, items: &mut [Item], denoised: bool) {
    for item in items.iter_mut() {
        if item.verdict != Verdict::Different {
            continue;
        }
        match item.name.as_str() {
            "scheduler.csv" => explained(item, RULE_SCHEDULER),
            "trials.csv" if denoised => {
                // Compared again with the reference calibrations the dump holds as -1 left
                // out (U62), cell by cell; every other cell as usual.
                let ds = xma_oracle::Dataset::from_paths(
                    "trials",
                    doc.path().unwrap_or(Path::new("")),
                    dir,
                );
                *item = super::super::check_trials_u62(&ds, doc, RULE_U62);
            }
            "distortion.csv" if distortion_14(&dir.join("distortion.csv"), doc) => {
                explained(item, RULE_DIST14);
            }
            _ => {}
        }
    }
}

/// XMALab's `distortion.csv` holds this program's 8 coefficients of each camera, exactly,
/// followed by zeros only.
fn distortion_14(path: &Path, doc: &Document) -> bool {
    let rows = csv_rows(path);
    !rows.is_empty()
        && rows.iter().all(|r| {
            let (Some(c), Some(i), Some(v)) = (
                r.get("camera").and_then(|x| x.parse::<usize>().ok()),
                r.get("index").and_then(|x| x.parse::<usize>().ok()),
                r.get("value").and_then(|x| x.parse::<f64>().ok()),
            ) else {
                return false;
            };
            match doc.project().cameras.get(c) {
                Some(cam) if i < 8 => cam.distortion[i] == v,
                Some(_) => v == 0.0,
                None => false,
            }
        })
}

/// A saved project's `project.xml` that differs from XMALab's only by the reference
/// calibration XMALab's Trial dialog wrote on opening (U62).
fn explain_project_xml(
    path: &Path,
    name: &str,
    doc: &mut Document,
    recompute_when_saving: bool,
    items: &mut [Item],
) {
    let key = format!("{name}:project.xml");
    let Some(item) = items
        .iter_mut()
        .find(|i| i.name == key && i.verdict == Verdict::Different)
    else {
        return;
    };
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let Ok(cpp) = xma_io::zip::Archive::new(&bytes) else {
        return;
    };
    let names: Vec<String> = cpp.entries().map(|e| e.decoded_name()).collect();
    let Some(i) = names.iter().position(|n| n == "project.xml") else {
        return;
    };
    let Ok(theirs) = cpp.read(i) else { return };
    let theirs = String::from_utf8_lossy(&theirs).into_owned();
    let date = theirs
        .find("DateCreated=\"")
        .map(|s| s + "DateCreated=\"".len())
        .and_then(|s| theirs[s..].find('"').map(|l| theirs[s..s + l].to_owned()))
        .unwrap_or_default();
    let Some(first) = cpp.entry(0) else { return };
    let options = xma_engine::SaveOptions {
        recompute_when_saving,
        time: xma_engine::SaveTime {
            date_created: date,
            modified: first.modified,
            utc_offset_seconds: 0,
        },
        trials: None,
    };
    let Ok((entries, _)) = doc.save_entries(true, &options) else {
        return;
    };
    let Some((_, ours)) = entries.files.iter().find(|(n, _)| n == "project.xml") else {
        return;
    };
    let ours = String::from_utf8_lossy(ours);
    let fixed = theirs.clone();
    let fixed = fixed.replace("referenceCalibration=\"-1\"", "referenceCalibration=\"0\"");
    if fixed == ours {
        explained(item, RULE_U62);
    }
}

#[cfg(test)]
mod review12_tests {
    //! Review of 20 November 2026, second round, findings 6 and 7 of `parity-xtask.md`.

    use super::*;

    fn pixels(got: &Decoded, want: &Decoded) -> Item {
        let mut c = Check::new("t", Tier::Read);
        compare_pixels(&mut c, got, want, "frame");
        c.finish()
    }

    #[test]
    fn review12_pixels_with_other_sample_counts_differ() {
        let want: Decoded = (2, 1, "L8".to_owned(), vec![1, 2]);
        let short: Decoded = (2, 1, "L8".to_owned(), vec![1]);
        let long: Decoded = (2, 1, "L8".to_owned(), vec![1, 2, 3]);
        assert_eq!(pixels(&short, &want).verdict, Verdict::Different);
        assert_eq!(pixels(&long, &want).verdict, Verdict::Different);
        assert_eq!(pixels(&want, &want).verdict, Verdict::Identical);
    }

    const HEAD: &str = "trial,name,is_default,start_frame,end_frame,n_frames,interpolate_3d,\
                        reference_calibration,recording_speed,cutoff_frequency\n";

    /// A `trials.csv` of one trial `Trial 1` (reference calibration 0, recording speed 100
    /// here) with the dump's reference calibration and recording speed.
    fn u62(tag: &str, reference: i64, speed: f64) -> Item {
        let mut p = xma_model::Project::default();
        let mut t = xma_model::Trial::default_trial();
        t.name = "Trial 1".to_owned();
        t.recording_speed = 100.0;
        p.trials = vec![std::sync::Arc::new(t.clone())];
        let doc = Document::from_project(p, CppCompat::XMALAB3);
        let dir = std::env::temp_dir().join(format!("xma-review12-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch folder");
        std::fs::write(
            dir.join("trials.csv"),
            format!(
                "{HEAD}0,{},{},{},{},{},{},{reference},{speed},{}\n",
                t.name,
                i64::from(t.is_default),
                t.start_frame,
                t.end_frame,
                t.n_frames,
                i64::from(t.interpolate_3d),
                t.cutoff_frequency
            ),
        )
        .expect("scratch file");
        let ds = xma_oracle::Dataset::from_paths(tag, &dir, &dir);
        let mut items = vec![super::super::super::check_trials(&ds, &doc)];
        explain_dump(&dir, &doc, &mut items, true);
        let _ = std::fs::remove_dir_all(&dir);
        items.remove(0)
    }

    #[test]
    fn review12_u62_waives_only_the_reference_calibration() {
        // The reference calibration alone (U62): explained.
        let i = u62("u62a", -1, 100.0);
        assert_eq!(i.verdict, Verdict::IdenticalExplained, "{i:?}");
        // The recording speed differs too: different.
        let i = u62("u62b", -1, 101.0);
        assert_eq!(i.verdict, Verdict::Different, "{i:?}");
        // A reference calibration other than -1: different.
        let i = u62("u62c", 2, 100.0);
        assert_eq!(i.verdict, Verdict::Different, "{i:?}");
    }
}

#[cfg(test)]
mod review12c_tests {
    //! Review of 20 November 2026, third round: a MayaCam file of a calibration image run is
    //! compared as an export file is, not by its numbers alone.

    use super::*;

    /// Compares two MayaCam texts through [`compare_file`].
    fn mayacam(got: &str, want: &str) -> Item {
        let dir = std::env::temp_dir().join(format!(
            "xma-review12c-mayacam-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (mine, reference) = (dir.join("mine.txt"), dir.join("reference.txt"));
        std::fs::write(&mine, got).unwrap();
        std::fs::write(&reference, want).unwrap();
        let item = compare_file(&reference, &mine, "Calibration_cam1_MayaCam1.txt", &dir);
        std::fs::remove_dir_all(&dir).unwrap();
        item
    }

    const WANT: &str = "image size\n1024,1024\n\ncamera matrix\n3207.57,0,457.27\n";

    #[test]
    fn review12c_mayacam_differences_are_reported() {
        for (g, w) in [
            // Equal numbers written differently.
            ("1e-05\n", "1.0000000000000001e-05\n"),
            // Different text.
            ("camera A\n1\n", "camera B\n1\n"),
            ("image size \n1024,1024\n", "image size\n1024,1024\n"),
            // Different spacing and line ends.
            ("1024, 1024\n", "1024,1024\n"),
            ("1024,1024\r\n", "1024,1024\n"),
            ("1024,1024", "1024,1024\n"),
            // A tolerated number does not excuse the spacing around it.
            ("3207.5700000001 ,0\n", "3207.57,0\n"),
        ] {
            let i = mayacam(g, w);
            assert_eq!(i.verdict, Verdict::Different, "{g:?} against {w:?}: {i:?}");
        }
        let i = mayacam(WANT, WANT);
        assert_eq!(i.verdict, Verdict::Identical, "{i:?}");
        // A number within ledger A37's tolerance is still tolerated.
        let got = WANT.replace("3207.57,", "3207.5700000001,");
        let i = mayacam(&got, WANT);
        assert_eq!(i.verdict, Verdict::WithinTolerance, "{i:?}");
    }
}
