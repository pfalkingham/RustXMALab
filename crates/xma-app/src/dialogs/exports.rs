//! File > Export and the Points panel's two exports: the dialogs each export opens, in
//! XMALab's order, then the file or folder dialog, then [`Document::export`], which writes
//! the files the engine checks against XMALab's.
//!
//! Ported from ui/MainWindow.cpp: `on_actionExport2D_Points_triggered`, `save3DPoints`
//! (with `on_actionExport3D_Points_triggered`), `saveRigidBodies` (with
//! `on_actionRigidBodyTransformations_triggered`), `on_actionExportEvents_triggered`,
//! `on_actionMarkertoMarkerDistances_triggered`, `on_actionPrecisionInfo_triggered`,
//! `on_actionReprojection_Errors_triggered` and `on_actionMayaCams_2_0_triggered`; and
//! ui/MarkerTreeWidget.cpp, `action_Save3D_triggered`, `action_SaveTransformations_triggered`
//! (the selected markers or bodies, sorted).
//!
//! How it runs, as the Points flows ([`super::points`]): [`start`] reads the document and
//! opens the first dialog ([`Dialog::Export`]); each answer comes back as
//! [`DialogResult::Export`] to [`answer`], which opens the next dialog or, once everything
//! is answered, asks for the file or folder ([`write`]). The steps ([`begin`], [`next`])
//! are plain functions of the document and the settings. The format choices are the
//! `Export*` settings that `PointImportExportDialog`'s radio buttons write.
//!
//! Differences from XMALab 3 (interface defects fixed; ledger rows proposed by this
//! package):
//!
//! - Cancel in the frame-range dialog of the 3D and transformation exports ends the export
//!   (XMALab 3 goes on and exports every frame).
//! - Cancel in "Enter cutoff frequency (Hz)" ends the export (XMALab 3 goes on, filtering at
//!   the trial's cutoff). The prompt offers the trial's cutoff unrounded (ledger F26).
//! - Escape or the close button on a Yes/No question ("Do you want to filter 3D Points?",
//!   the MayaCams question) ends the export (XMALab 3 takes it as No and goes on).
//! - Cancel in the folder or file dialog of the 3D export shows nothing (XMALab 3 reads an
//!   uninitialised flag and may show "There has been a problem saving").
//! - The Marker-to-Marker Distances range dialog has no filter row (XMALab 3 passes its
//!   parent pointer as `withFilter`, so an "unfiltered / filtered" row that does nothing
//!   shows).
//! - MayaCams 2.0 in a project without trials exports every calibration image without
//!   asking (XMALab 3 reads the reference calibration image of a trial that does not exist
//!   when the answer is No).
//! - A file that cannot be written is reported in the error dialog and the Console (XMALab 3
//!   says nothing); each export writes one Console line naming what it wrote.

use std::path::{Path, PathBuf};

use xma_engine::Document;
use xma_engine::export::{
    Export, ExportError, ExportFailure, FrameRange, NO_RECORDING_SPEED, NO_SPEED_OR_CUTOFF,
    NOT_CALIBRATED, Points2dOptions, Points3dOptions, TransformationsOptions, can_export_filtered,
};

use super::from_to::{FromTo, FromToAnswer};
use super::point_import_export::{ImportExportKind, PointImportExportDialog};
use super::{Dialog, DialogEnv, DialogResult, form, last_dir};
use crate::XmaApp;
use crate::icons::Icon;
use crate::log::LogLevel;
use crate::settings::ImportExportSettings;

/// "Do you want to filter 3D Points?" (`MainWindow::save3DPoints`).
pub const FILTER_QUESTION: &str = "Do you want to filter 3D Points?";
/// The MayaCams question (`MainWindow::on_actionMayaCams_2_0_triggered`).
pub const MAYACAM_QUESTION: &str = "Do you want to save the mayacams for all calibration images? Click no if you only need the mayacams for the currently selected reference calibration";
/// The title of the cutoff prompt (`QInputDialog::getDouble`, ledger F26).
pub const CUTOFF_TITLE: &str = "Enter cutoff frequency (Hz)";
/// What XMALab shows when the 3D export fails (`save3dPoints` returned false).
pub const SAVE_PROBLEM: &str = "There has been a problem saving. Please ensure that the cutoff and framerate are valid in case you exported filtered data.";
/// The folder dialog's title, with XMALab's trailing space.
pub const FOLDER_TITLE: &str = "Save to Directory ";
/// The file dialogs' filter.
pub const CSV_FILTER: (&str, &[&str]) = ("Comma seperated data", &["csv"]);

/// Which export.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExportAction {
    /// File > Export > 2D Points.
    Points2d,
    /// File > Export > 3D Points (`None`: every marker), or the Points panel's "Save the 3D
    /// data of selected points" (the selected markers, sorted).
    Points3d { markers: Option<Vec<usize>> },
    /// File > Export > RigidBodyTransformations (`None`: every body), or the Points panel's
    /// "Save the transformation of selected rigid bodies".
    Transformations { bodies: Option<Vec<usize>> },
    /// File > Export > Events.
    Events,
    /// File > Export > Marker to Marker Distances (range).
    MarkerDistances,
    /// File > Export > Precision Info.
    PrecisionInfo,
    /// File > Export > Reprojection Errors.
    ReprojectionErrors,
    /// File > Export > MayaCams 2.0.
    MayaCams,
    /// File > Export > Undistorted Trial images: the range dialog with its format and
    /// filter rows, then the folder; written by a job (`app/images.rs`).
    TrialImages,
    /// File > Export > All Trials for External: the folder; written by a job.
    AllTrials,
}

/// The answers of `FromToDialog` with its format and filter rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImagesChoice {
    /// Frames, 0-based and inclusive.
    pub first: usize,
    pub last: usize,
    pub format: super::from_to::ImageFormat,
    pub filtered: bool,
}

/// Where an export stands: what it is and what has been answered so far.
#[derive(Clone, Debug, PartialEq)]
pub struct Flow {
    pub action: ExportAction,
    /// The trial exported (the active trial).
    pub trial: usize,
    /// 3D export: the cutoff in hertz, or -1 for unfiltered (the C++ `filter_frequency`).
    pub cutoff: f64,
    /// The frames of the range dialog, 0-based and inclusive.
    pub range: Option<(usize, usize)>,
    /// MayaCams: every calibration image ("Yes").
    pub all_images: bool,
    /// Undistorted Trial images: the range dialog's answers.
    pub images: Option<ImagesChoice>,
}

impl Flow {
    fn new(action: ExportAction, trial: usize) -> Self {
        Flow {
            action,
            trial,
            cutoff: -1.0,
            range: None,
            all_images: false,
            images: None,
        }
    }
}

/// The dialog a step shows.
#[derive(Clone, Debug, PartialEq)]
pub enum Form {
    /// `PointImportExportDialog` of the export's kind.
    Options(PointImportExportDialog),
    /// `ConfirmationDialog` with "Yes" and "No".
    YesNo { message: String },
    /// `QInputDialog::getDouble(this, "Enter cutoff frequency (Hz)", "", cutoff, 0.0,
    /// recordingSpeed * 0.5)`.
    Cutoff { value: f64, max: f64 },
    /// `FromToDialog` without its format and filter rows.
    FromTo(FromTo),
}

/// One step on screen.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportDialog {
    pub flow: Flow,
    pub form: Form,
}

/// The answer to a step.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    /// OK in `PointImportExportDialog`.
    Ok,
    Yes,
    No,
    /// OK in the cutoff prompt.
    Cutoff(f64),
    /// OK in the range dialog: frames, 0-based.
    Range(usize, usize),
    /// OK in the range dialog with its format and filter rows.
    Images(ImagesChoice),
}

/// A step answered.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportOutcome {
    pub flow: Flow,
    pub answer: Answer,
}

/// What follows a step.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Nothing,
    /// The error dialog with XMALab's message.
    Error(String),
    Open(ExportDialog),
    /// Everything answered: the file or folder dialog, then the export.
    Write(Flow),
}

fn open(flow: Flow, form: Form) -> Step {
    Step::Open(ExportDialog { flow, form })
}

fn range_dialog(t: &xma_model::Trial) -> Form {
    Form::FromTo(FromTo::new(
        t.start_frame,
        t.end_frame,
        t.n_frames,
        false,
        false,
    ))
}

/// The first step of an export of `trial` (`None`: no trial).
#[must_use]
pub fn begin(action: &ExportAction, doc: &Document, trial: Option<usize>) -> Step {
    let trial_data = trial.and_then(|t| doc.project().trials.get(t));
    if *action == ExportAction::AllTrials {
        // MainWindow::on_actionAll_Trials_for_External_triggered: only the folder.
        return Step::Write(Flow::new(action.clone(), trial.unwrap_or(0)));
    }
    if *action == ExportAction::MayaCams {
        if !doc.is_calibrated() {
            return Step::Error(NOT_CALIBRATED.into());
        }
        let Some(t) = trial.filter(|_| trial_data.is_some()) else {
            let mut f = Flow::new(action.clone(), 0);
            f.all_images = true;
            return Step::Write(f);
        };
        return open(
            Flow::new(action.clone(), t),
            Form::YesNo {
                message: MAYACAM_QUESTION.into(),
            },
        );
    }
    let (Some(t), Some(td)) = (trial, trial_data) else {
        return Step::Nothing;
    };
    let flow = Flow::new(action.clone(), t);
    let options = |k| Form::Options(PointImportExportDialog::new(k));
    match action {
        ExportAction::Points2d => open(flow, options(ImportExportKind::Export2d)),
        ExportAction::Points3d { .. } => open(flow, options(ImportExportKind::Export3d)),
        ExportAction::Transformations { .. } => {
            open(flow, options(ImportExportKind::ExportTransformations))
        }
        ExportAction::MarkerDistances | ExportAction::PrecisionInfo => open(flow, range_dialog(td)),
        // MainWindow::on_actionExport_Undistorted_Trial_images_for_Maya_triggered:
        // `FromToDialog(start, end, nbImages, true, true)`.
        ExportAction::TrialImages => open(
            flow,
            Form::FromTo(FromTo::new(
                td.start_frame,
                td.end_frame,
                td.n_frames,
                true,
                true,
            )),
        ),
        ExportAction::AllTrials => Step::Write(flow),
        ExportAction::Events | ExportAction::ReprojectionErrors | ExportAction::MayaCams => {
            Step::Write(flow)
        }
    }
}

/// The step after `answer`.
#[must_use]
pub fn next(flow: &Flow, answer: &Answer, doc: &Document, s: &ImportExportSettings) -> Step {
    let Some(t) = doc.project().trials.get(flow.trial) else {
        return Step::Nothing;
    };
    let mut flow = flow.clone();
    let range_or_write = |flow: Flow, offset_cols: bool| {
        if offset_cols {
            open(flow, range_dialog(t))
        } else {
            Step::Write(flow)
        }
    };
    match (&flow.action, answer) {
        (_, Answer::Images(c)) => {
            flow.images = Some(*c);
            Step::Write(flow)
        }
        (_, Answer::Range(a, b)) => {
            flow.range = Some((*a, *b));
            Step::Write(flow)
        }
        (ExportAction::Points2d, Answer::Ok) => Step::Write(flow),
        (ExportAction::Points3d { .. }, Answer::Ok) => open(
            flow,
            Form::YesNo {
                message: FILTER_QUESTION.into(),
            },
        ),
        (ExportAction::Points3d { .. }, Answer::Yes) => {
            if t.recording_speed <= 0.0 {
                return Step::Error(NO_RECORDING_SPEED.into());
            }
            let max = t.recording_speed * 0.5;
            // Qt clamps the value into the range; the trial's cutoff is offered as it is
            // (ledger F26).
            let value = t.cutoff_frequency.clamp(0.0, max);
            open(flow, Form::Cutoff { value, max })
        }
        (ExportAction::Points3d { .. }, Answer::No) => {
            flow.cutoff = -1.0;
            range_or_write(flow, s.export_3d_offset_cols)
        }
        (ExportAction::Points3d { .. }, Answer::Cutoff(v)) => {
            flow.cutoff = *v;
            range_or_write(flow, s.export_3d_offset_cols)
        }
        (ExportAction::Transformations { bodies }, Answer::Ok) => {
            let all: Vec<usize> = (0..t.rigid_bodies.len()).collect();
            let bodies = bodies.as_deref().unwrap_or(&all);
            if s.export_trans_filtered && !can_export_filtered(t, bodies, doc.compat()) {
                return Step::Error(NO_SPEED_OR_CUTOFF.into());
            }
            range_or_write(flow, s.export_trans_offset_cols)
        }
        (ExportAction::MayaCams, Answer::Yes | Answer::No) => {
            flow.all_images = *answer == Answer::Yes;
            Step::Write(flow)
        }
        _ => Step::Nothing,
    }
}

/// Where an export goes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// `QFileDialog::getExistingDirectory(this, "Save to Directory ", last folder)`; the
    /// folder becomes the last used folder when `remember`.
    Folder { remember: bool },
    /// `QFileDialog::getSaveFileName(this, title, last folder, "Comma seperated data
    /// (*.csv)")`, `.csv` appended when missing; the file's folder becomes the last used
    /// folder when `remember`.
    File { title: &'static str, remember: bool },
}

/// The export and where it goes, from the answers and the `Export*` settings; `None` when
/// neither "single files" nor "combined file" is chosen (XMALab then does nothing).
#[must_use]
pub fn plan(flow: &Flow, doc: &Document, s: &ImportExportSettings) -> Option<(Export, Target)> {
    let compat = doc.compat();
    let folder = Target::Folder { remember: true };
    let file = |title| Target::File {
        title,
        remember: true,
    };
    let choose = |single: bool, multi: bool, title| {
        if single {
            Some(folder.clone())
        } else if multi {
            Some(file(title))
        } else {
            None
        }
    };
    let frames = |on: bool| {
        flow.range
            .filter(|_| on)
            .map(|(start, stop)| FrameRange { start, stop })
    };
    Some(match &flow.action {
        ExportAction::Points2d => (
            Export::Points2d(Points2dOptions {
                one_file: s.export_2d_multi,
                distorted: s.export_2d_distorted,
                offset1: s.export_2d_count1,
                y_invert: s.export_2d_y_up,
                header: s.export_2d_header,
                offset_columns: s.export_2d_offset_cols,
                compat,
                camera: None,
            }),
            choose(s.export_2d_single, s.export_2d_multi, "Save 2D points as")?,
        ),
        ExportAction::Points3d { markers } => (
            Export::Points3d {
                markers: markers.clone(),
                options: Points3dOptions {
                    one_file: s.export_3d_multi,
                    header: s.export_3d_header,
                    filter_frequency: flow.cutoff,
                    frames: frames(s.export_3d_offset_cols),
                    compat,
                },
            },
            choose(s.export_3d_single, s.export_3d_multi, "Save 3D points as")?,
        ),
        ExportAction::Transformations { bodies } => (
            Export::RigidBodies {
                bodies: bodies.clone(),
                options: TransformationsOptions {
                    one_file: s.export_trans_multi,
                    header: s.export_trans_header,
                    filtered: s.export_trans_filtered,
                    frames: frames(s.export_trans_offset_cols),
                    compat,
                },
            },
            choose(
                s.export_trans_single,
                s.export_trans_multi,
                "Save Rigid Bodies as",
            )?,
        ),
        // The C++ passes `getFrom() - 1` and `getTo()`, and does not remember the folder.
        ExportAction::MarkerDistances | ExportAction::PrecisionInfo => {
            let (a, b) = flow.range?;
            if flow.action == ExportAction::MarkerDistances {
                (
                    Export::MarkerDistances { from: a, to: b + 1 },
                    Target::File {
                        title: "Save distances as",
                        remember: false,
                    },
                )
            } else {
                (
                    Export::PrecisionInfo { from: a, to: b + 1 },
                    Target::File {
                        title: "Save precision info as",
                        remember: false,
                    },
                )
            }
        }
        ExportAction::Events => (Export::Events, folder),
        ExportAction::ReprojectionErrors => (Export::ReprojectionErrors, folder),
        ExportAction::MayaCams => (
            Export::MayaCams {
                all: flow.all_images,
            },
            folder,
        ),
        // Written by a job ([`write`]).
        ExportAction::TrialImages | ExportAction::AllTrials => return None,
    })
}

/// `ensureExtension(fileName, {".csv"}, ".csv")`: `.csv` appended unless the name ends in
/// it (ignoring case).
#[must_use]
pub fn ensure_csv(path: PathBuf) -> PathBuf {
    if path.to_string_lossy().to_lowercase().ends_with(".csv") {
        path
    } else {
        let mut s = path.into_os_string();
        s.push(".csv");
        PathBuf::from(s)
    }
}

// ---------------------------------------------------------------------------------------
// Running in the application.

/// Starts an export (from `XmaApp::dispatch`).
pub fn start(app: &mut XmaApp, a: &ExportAction) {
    let Some(doc) = app.session.doc.as_ref() else {
        return;
    };
    let step = begin(a, doc, app.state.active_trial);
    run(app, step);
}

/// Acts on an answered step (from `XmaApp::dialog_result`).
pub fn answer(app: &mut XmaApp, o: &ExportOutcome) {
    let Some(doc) = app.session.doc.as_ref() else {
        return;
    };
    let step = next(&o.flow, &o.answer, doc, &app.settings.import_export);
    run(app, step);
}

fn run(app: &mut XmaApp, step: Step) {
    match step {
        Step::Nothing => {}
        Step::Error(m) => app.dialogs.error(m),
        Step::Open(d) => app.dialogs.open(Dialog::Export(Box::new(d))),
        Step::Write(flow) => write(app, &flow),
    }
}

/// The file or folder dialog, then the export and the files.
pub fn write(app: &mut XmaApp, flow: &Flow) {
    let Some(doc) = app.session.doc.as_ref() else {
        return;
    };
    if matches!(
        flow.action,
        ExportAction::TrialImages | ExportAction::AllTrials
    ) {
        let start = last_dir(&app.settings);
        let Some(dir) = app.file_chooser.pick_folder(FOLDER_TITLE, start.as_deref()) else {
            return;
        };
        if flow.action == ExportAction::TrialImages {
            // Settings::setLastUsedDirectory(outputPath, true) (not for All Trials).
            let abs = std::path::absolute(&dir).unwrap_or_else(|_| dir.clone());
            app.settings.last_directory_used = abs.to_string_lossy().into_owned();
            app.settings_dirty = true;
        }
        app.start_images_export(flow, &dir);
        return;
    }
    let Some((export, target)) = plan(flow, doc, &app.settings.import_export) else {
        return;
    };
    let start = last_dir(&app.settings);
    // The folder written into and the name of a one-file export.
    let (dir, name, remember_file, remember_dir) = match target {
        Target::Folder { remember } => {
            let Some(d) = app.file_chooser.pick_folder(FOLDER_TITLE, start.as_deref()) else {
                return;
            };
            (d, String::new(), None, remember)
        }
        Target::File { title, remember } => {
            let Some(p) = app
                .file_chooser
                .save_file_filtered(title, start.as_deref(), CSV_FILTER)
            else {
                return;
            };
            let p = ensure_csv(p);
            let dir = p
                .parent()
                .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            (dir, name, remember.then_some(p), false)
        }
    };
    let Some(doc) = app.session.doc.as_mut() else {
        return;
    };
    let compat = doc.compat();
    let result = doc.export(flow.trial, &export);
    // Settings::setLastUsedDirectory after the export, as the C++ does.
    if let Some(p) = &remember_file {
        app.settings.set_last_used_directory_from_file(p);
        app.settings_dirty = true;
    } else if remember_dir {
        let abs = std::path::absolute(&dir).unwrap_or_else(|_| dir.clone());
        app.settings.last_directory_used = abs.to_string_lossy().into_owned();
        app.settings_dirty = true;
    }
    let output = match result {
        Ok(o) => o,
        Err(ExportFailure::Refused(m)) => {
            app.dialogs.error(m);
            return;
        }
        Err(ExportFailure::Export(ExportError::FilterRefused(e))) => {
            app.log.error(format!("Export: {e}"));
            app.dialogs.error(SAVE_PROBLEM);
            return;
        }
        Err(e) => {
            let msg = format!("The export wrote nothing: {e}");
            app.log.error(msg.clone());
            app.dialogs.error(msg);
            return;
        }
    };
    match xma_engine::export::write_output_with(&dir, &output, &name, compat) {
        Ok(written) => {
            let what = match written.as_slice() {
                [one] => one.display().to_string(),
                many => format!("{} files in {}", many.len(), dir.display()),
            };
            app.log.write(format!("Exported {what}"), LogLevel::Output);
        }
        Err(e) => {
            let msg = format!("Could not write the export to {}: {e}", dir.display());
            app.log.error(msg.clone());
            app.dialogs.error(msg);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Drawing.

/// OK in the range dialog: the range, with the format and filter when the dialog shows them.
fn from_to_answer(f: &FromTo) -> Answer {
    let (a, b) = f.frames();
    if f.with_format {
        Answer::Images(ImagesChoice {
            first: a,
            last: b,
            format: f.format,
            filtered: f.filtered,
        })
    } else {
        Answer::Range(a, b)
    }
}

impl ExportDialog {
    /// The window title of the step.
    #[must_use]
    pub fn title(&self) -> &'static str {
        match &self.form {
            Form::Options(d) => d.kind.title(),
            Form::YesNo { .. } => "Confirmation Required",
            Form::Cutoff { .. } => CUTOFF_TITLE,
            Form::FromTo(_) => "Select Range",
        }
    }

    /// The answer of the default button (OK or Yes) with the form's values.
    #[must_use]
    pub fn default_answer(&self) -> Answer {
        match &self.form {
            Form::Options(_) => Answer::Ok,
            Form::YesNo { .. } => Answer::Yes,
            Form::Cutoff { value, max } => Answer::Cutoff(value.clamp(0.0, *max)),
            Form::FromTo(f) => from_to_answer(f),
        }
    }

    /// The result of the default button (tests: `Dialogs::accept`).
    #[must_use]
    pub fn accept(&self) -> DialogResult {
        self.result(self.default_answer())
    }

    /// The result of any answer (tests: "No").
    #[must_use]
    pub fn result(&self, answer: Answer) -> DialogResult {
        DialogResult::Export(ExportOutcome {
            flow: self.flow.clone(),
            answer,
        })
    }
}

/// What drawing a step gave.
#[derive(Debug, Default)]
pub struct Shown {
    pub result: Option<DialogResult>,
    /// Closed without an answer: the export ends.
    pub close: bool,
    pub settings_changed: bool,
}

/// Draws the step on screen as a modal dialog.
pub fn show(d: &mut ExportDialog, ctx: &egui::Context, env: &mut DialogEnv<'_>) -> Shown {
    let mut out = Shown::default();
    if let Form::Options(o) = &mut d.form {
        let f = o.show(ctx, env.settings);
        out.settings_changed = f.settings_changed;
        out.close = f.close;
        if matches!(f.result, Some(DialogResult::PointImportExport(_))) {
            out.result = Some(d.result(Answer::Ok));
        }
        return out;
    }
    let mut answer: Option<Answer> = None;
    let title = d.title();
    let icons = env.icons;
    let modal = form::modal(ctx, egui::Id::unique("xma_export_modal"), |ui| {
        if form::dialog_title(ui, title) {
            out.close = true;
        }
        match &mut d.form {
            Form::Options(_) => {}
            Form::YesNo { message } => {
                ui.set_max_width(520.0);
                ui.horizontal(|ui| {
                    ui.add(icons.image(ui.ctx(), Icon::Alert, 48.0));
                    ui.add(egui::Label::new(message.as_str()).wrap());
                });
                ui.add_space(6.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if form::button(ui, true, "No", 80.0).clicked() {
                        answer = Some(Answer::No);
                    }
                    if form::button(ui, true, "Yes", 80.0).clicked() {
                        answer = Some(Answer::Yes);
                    }
                });
            }
            Form::Cutoff { value, max } => {
                ui.set_min_width(320.0);
                ui.add(
                    egui::DragValue::new(value)
                        .range(0.0..=*max)
                        .speed(0.1)
                        .min_decimals(1),
                )
                .accessible_name(CUTOFF_TITLE);
                let v = value.clamp(0.0, *max);
                ui.add_space(6.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let ok = form::button(ui, true, "OK", 80.0);
                    let cancel = form::button(ui, true, "Cancel", 80.0);
                    if cancel.clicked() {
                        out.close = true;
                    } else if ok.clicked() {
                        answer = Some(Answer::Cutoff(v));
                    }
                });
            }
            Form::FromTo(f) => match f.ui(ui) {
                Some(FromToAnswer::Ok) => {
                    answer = Some(from_to_answer(f));
                }
                Some(FromToAnswer::Cancel) => out.close = true,
                None => {}
            },
        }
    });
    if modal.escaped() {
        out.close = true;
    }
    if let Some(a) = answer {
        out.result = Some(d.result(a));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_extension_like_ensure_extension() {
        assert_eq!(ensure_csv("a/b".into()), PathBuf::from("a/b.csv"));
        assert_eq!(ensure_csv("a/b.CSV".into()), PathBuf::from("a/b.CSV"));
        assert_eq!(ensure_csv("a/b.txt".into()), PathBuf::from("a/b.txt.csv"));
    }
}
