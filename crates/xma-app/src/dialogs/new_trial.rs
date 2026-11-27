//! `NewTrialDialog` and its `CameraBoxTrial`es: the workspace bar's "+" and the start
//! screen's "Add trial" (C55 to C59), File > New trial without calibration (C54) and the
//! Trial dialog's "Change Trial Data" (C61).
//!
//! Ported control by control from `NewTrialDialog.ui`, `CameraBoxTrial.ui`,
//! ui/NewTrialDialog.cpp and ui/CameraBoxTrial.cpp. Title "Select videopaths for new
//! trial" ("Change trial data" for C61). Grid of the form, top to bottom:
//!
//! | Row | Control | Text |
//! |---|---|---|
//! | 1 | `groupBox_2` "Name" | `lineEditTrialName`: "Trial <number of trials + 1>", or the trial's name (C61) |
//! | 2 | `frameNBCameras` (only for C54) | `label_5` "Number of cameras", `toolButtonCameraMinus` "-", `labelNbCameras`, `toolButtonCameraPlus` "+" |
//! | 3 | `groupBox` "Videopaths" | a scroll area with one `CameraBoxTrial` per camera, titled with the camera's name ("Camera 1" .. for C54) |
//! | 4 | | `pushButton_LoadXMA` "Load XMA Portal file", `pushButton_Default` "Add Default Trial", `pushButton_Cancel` "Cancel", `pushButton_OK` "Ok" |
//!
//! A `CameraBoxTrial` is a group box: "Video path", the line edit (disabled), the folder
//! button (`toolButtonImage`, icon Images.png) and the file button (`toolButtonVideo`,
//! icon video.png), and the count "(n)" in the label after a choice (the C++ writes the
//! count into the "Video path" label).
//!
//! The dialog changes nothing itself. OK runs [`NewTrialSpec::check`] (XMALab's
//! `isComplete`): a failure shows its sentence in the error dialog over the form, which
//! stays open. "Add Default Trial" is enabled unless the project has a Default trial or no
//! calibration (and never for C54).
//!
//! "Load XMA Portal file" (C60, `on_pushButton_LoadXMA_clicked`): the confirmation
//! ([`LOAD_CONFIRM`], Yes/No), the file dialog "Select dataset" (`*.xmatrial *.zip`), the
//! question [`DELETE_QUESTION`] (Yes/No); the dialog then closes with
//! [`NewTrialAnswer::LoadPortal`] and the application extracts the file on a worker thread
//! ("Load trial <file>", `crate::app::portal`) and opens the dialog again filled
//! ([`NewTrialDialog::fill_from_portal`]): the trial name, each camera's files, and the
//! metadata kept for OK, which then answers [`NewTrialAnswer::PortalTrial`] or
//! [`NewTrialAnswer::PortalChange`].
//!
//! Difference from XMALab 3 (an interface defect fixed; proposed ledger row in the appnew
//! report): cancelling the file button's dialog keeps the files chosen before (XMALab 3
//! empties the list and keeps showing the old text).

use std::path::PathBuf;

use egui::Ui;
use xma_engine::Document;
use xma_engine::create::{NewTrialSpec, NoCalibrationSpec};
use xma_model::{CalibrationType, Project};

use super::edit_forms::{FormOutcome, Prompt, PromptAnswer};
use super::file_names::{self, TRIAL_FILES, path_text};
use super::new_project::line_edit;
use super::{DialogEnv, DialogResult, form, last_dir};
use crate::icons::Icon;

/// The window title of a new trial.
pub const TITLE: &str = "Select videopaths for new trial";
/// The window title of "Change Trial Data".
pub const CHANGE_TITLE: &str = "Change trial data";
/// The folder button's dialog title (with XMALab's trailing space).
pub const FOLDER_TITLE: &str = "Load Directory ";
/// The file button's dialog title.
pub const FILES_TITLE: &str = "Open video stream movie file or images";
/// The questions of "Load XMA Portal file" (`on_pushButton_LoadXMA_clicked`).
pub use xma_engine::portal::{DELETE_QUESTION, LOAD_CONFIRM};
/// The file dialog's title.
pub const LOAD_TITLE: &str = "Select dataset";
/// The file dialog's filter, "Dataset (*.xmatrial  *.zip)".
pub const PORTAL_FILES: (&str, &[&str]) = ("Dataset", &["xmatrial", "zip"]);

/// Which question of "Load XMA Portal file" the prompt over the form is.
#[derive(Clone, Debug, PartialEq, Eq)]
enum PortalStep {
    Confirm,
    Delete(PathBuf),
}

/// What the dialog is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NewTrialMode {
    /// A new trial in the open project (C55).
    New,
    /// The first trial of a new project without calibration (C54).
    NoCalibration,
    /// "Change Trial Data" for this trial (C61).
    Change(usize),
}

/// One `CameraBoxTrial`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraBoxTrial {
    /// The group box's title: the camera's name.
    pub name: String,
    /// `imageFileNames`.
    pub files: Vec<PathBuf>,
    /// `lineEdit`: the folder, or the common prefix of the files.
    pub text: String,
    /// The count "(n)" (shown in place of "Video path" after a choice).
    pub count_text: Option<String>,
}

impl CameraBoxTrial {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            files: Vec::new(),
            text: String::new(),
            count_text: None,
        }
    }

    /// The folder button's answer: the folder's images in natural order; the line edit
    /// shows the folder. A folder with no image gives an empty list (and "(0)"), as in
    /// XMALab 3.
    ///
    /// Ported from ui/CameraBoxTrial.cpp, `CameraBoxTrial::on_toolButtonImage_clicked`.
    pub fn set_folder(&mut self, folder: &std::path::Path) {
        self.files = file_names::folder_images(folder);
        self.text = path_text(folder);
        self.count_text = Some(format!("({})", self.files.len()));
    }

    /// One file of an XMA Portal trial (`CameraBoxTrial::setFilename`): `files` as
    /// `xma_engine::portal::camera_files` lists them for `path`; the line edit shows the
    /// image folder for a zip (the path without `.zip`), else the file.
    ///
    /// Ported from ui/CameraBoxTrial.cpp, `CameraBoxTrial::setFilename`.
    pub fn set_portal_file(&mut self, path: &std::path::Path, files: Vec<PathBuf>) {
        self.text = match xma_engine::portal::nested_zip_text(path) {
            Some(t) => t,
            None => file_names::common_prefix_of(&files),
        };
        self.count_text = Some(format!("({})", files.len()));
        self.files = files;
    }

    /// The file button's answer, in natural order; the line edit shows their common
    /// prefix. No files (Cancel) changes nothing.
    ///
    /// Ported from ui/CameraBoxTrial.cpp, `CameraBoxTrial::on_toolButtonVideo_clicked`.
    pub fn set_files(&mut self, mut files: Vec<PathBuf>) {
        if files.is_empty() {
            return;
        }
        file_names::sort_natural(&mut files);
        self.text = file_names::common_prefix_of(&files);
        self.count_text = Some(format!("({})", files.len()));
        self.files = files;
    }
}

/// The answer of the dialog.
#[derive(Clone, Debug, PartialEq)]
pub enum NewTrialAnswer {
    /// OK for a new trial (C55).
    NewTrial(NewTrialSpec),
    /// OK for "Change Trial Data" (C61).
    Change { trial: usize, spec: NewTrialSpec },
    /// OK for a new project without calibration (C54); the dialog comes back after a
    /// failure.
    NoCalibration {
        spec: NoCalibrationSpec,
        dialog: Box<NewTrialDialog>,
    },
    /// "Add Default Trial" (C59).
    DefaultTrial,
    /// "Load XMA Portal file" answered (C60): the file to import, whether to delete it
    /// after a successful import, and the dialog to fill.
    LoadPortal {
        dialog: Box<NewTrialDialog>,
        file: PathBuf,
        delete: bool,
    },
    /// OK for a new trial filled from an XMA Portal file: with its metadata.
    PortalTrial {
        spec: NewTrialSpec,
        metadata: Vec<String>,
    },
    /// OK for "Change Trial Data" filled from an XMA Portal file.
    PortalChange {
        trial: usize,
        spec: NewTrialSpec,
        metadata: Vec<String>,
    },
}

/// The dialog's state.
#[derive(Clone, Debug, PartialEq)]
pub struct NewTrialDialog {
    pub mode: NewTrialMode,
    /// `lineEditTrialName`.
    pub name: String,
    pub cameras: Vec<CameraBoxTrial>,
    /// `pushButton_Default` enabled.
    pub default_enabled: bool,
    /// The error dialog over the form.
    pub prompt: Option<Prompt>,
    /// The XMA Portal metadata from "Load XMA Portal file" (`xml_metadata`), kept for OK.
    pub portal_metadata: Option<Vec<String>>,
    /// The question of "Load XMA Portal file" the prompt is.
    portal_step: Option<PortalStep>,
}

/// What a control asks for.
enum Act {
    Minus,
    Plus,
    Folder(usize),
    Files(usize),
    LoadXma,
    Default,
    Ok,
    Cancel,
}

impl NewTrialDialog {
    /// A new trial of `doc`: one box per camera, the name "Trial <n + 1>".
    ///
    /// Ported from ui/NewTrialDialog.cpp, `NewTrialDialog::NewTrialDialog` (`:53-85`).
    #[must_use]
    pub fn new(doc: &Document) -> Self {
        let p = doc.project();
        Self {
            mode: NewTrialMode::New,
            name: format!("Trial {}", p.trials.len() + 1),
            cameras: p
                .cameras
                .iter()
                .map(|c| CameraBoxTrial::new(c.name.clone()))
                .collect(),
            default_enabled: default_allowed(p),
            prompt: None,
            portal_metadata: None,
            portal_step: None,
        }
    }

    /// "Change Trial Data" for `trial`: titled "Change trial data", its name filled in, the
    /// boxes empty. `None` for a trial that does not exist.
    #[must_use]
    pub fn change(doc: &Document, trial: usize) -> Option<Self> {
        let t = doc.project().trials.get(trial)?;
        let mut d = Self::new(doc);
        d.mode = NewTrialMode::Change(trial);
        d.name.clone_from(&t.name);
        Some(d)
    }

    /// "New trial without calibration": the empty project's dialog with the camera count
    /// shown, one box "Camera 1", "Add Default Trial" disabled.
    ///
    /// Ported from ui/NewTrialDialog.cpp, `setNBCamerasVisible`.
    #[must_use]
    pub fn without_calibration() -> Self {
        Self {
            mode: NewTrialMode::NoCalibration,
            name: "Trial 1".to_owned(),
            cameras: vec![CameraBoxTrial::new("Camera 1")],
            default_enabled: false,
            prompt: None,
            portal_metadata: None,
            portal_step: None,
        }
    }

    /// The window title.
    #[must_use]
    pub fn title(&self) -> &'static str {
        match self.mode {
            NewTrialMode::Change(_) => CHANGE_TITLE,
            _ => TITLE,
        }
    }

    /// `frameNBCameras` is shown.
    #[must_use]
    pub fn count_visible(&self) -> bool {
        self.mode == NewTrialMode::NoCalibration
    }

    /// "+": a box "Camera <n>".
    pub fn plus(&mut self) {
        let n = self.cameras.len() + 1;
        self.cameras
            .push(CameraBoxTrial::new(format!("Camera {n}")));
    }

    /// "-": removes the last box, never the only one.
    pub fn minus(&mut self) {
        if self.cameras.len() > 1 {
            self.cameras.pop();
        }
    }

    /// The trial as the engine takes it.
    #[must_use]
    pub fn spec(&self) -> NewTrialSpec {
        NewTrialSpec {
            name: self.name.clone(),
            cameras: self.cameras.iter().map(|c| c.files.clone()).collect(),
        }
    }

    /// OK: the check against `project` (the empty project for C54); on failure the error
    /// is shown over the form. Returns the answer when it passes.
    ///
    /// Ported from ui/NewTrialDialog.cpp, `on_pushButton_OK_clicked` and `isComplete`.
    pub fn ok(&mut self, project: Option<&Project>) -> Option<NewTrialAnswer> {
        let spec = self.spec();
        // C54 has no project yet: the check runs against one that has the camera boxes'
        // cameras, so its sentences name them ("Camera 2 is incomplete : Missing Images").
        let mut empty = Project::new();
        if matches!(self.mode, NewTrialMode::NoCalibration) {
            empty.cameras = self
                .cameras
                .iter()
                .map(|c| std::sync::Arc::new(xma_model::Camera::new(c.name.clone())))
                .collect();
        }
        let (project, replacing) = match self.mode {
            NewTrialMode::NoCalibration => (&empty, None),
            NewTrialMode::New => (project?, None),
            NewTrialMode::Change(t) => (project?, Some(t)),
        };
        if let Err(message) = spec.check(project, replacing) {
            self.prompt = Some(Prompt::Error { message });
            return None;
        }
        Some(match (self.mode, self.portal_metadata.clone()) {
            (NewTrialMode::New, Some(metadata)) => NewTrialAnswer::PortalTrial { spec, metadata },
            (NewTrialMode::Change(trial), Some(metadata)) => NewTrialAnswer::PortalChange {
                trial,
                spec,
                metadata,
            },
            (NewTrialMode::New, None) => NewTrialAnswer::NewTrial(spec),
            (NewTrialMode::Change(trial), None) => NewTrialAnswer::Change { trial, spec },
            (NewTrialMode::NoCalibration, _) => NewTrialAnswer::NoCalibration {
                spec: NoCalibrationSpec {
                    camera_names: self.cameras.iter().map(|c| c.name.clone()).collect(),
                    trial: spec,
                },
                dialog: Box::new(self.clone()),
            },
        })
    }

    /// Draws the dialog (and the error over it).
    pub fn show(&mut self, ctx: &egui::Context, env: &mut DialogEnv<'_>) -> FormOutcome {
        let mut out = FormOutcome::default();
        let prompt_open = self.prompt.is_some();
        let mut act = None;
        let modal = egui::Modal::new(egui::Id::unique("xma_new_trial_dialog")).show(ctx, |ui| {
            if prompt_open {
                ui.disable();
            }
            act = self.form(ui, env.icons);
        });
        if let Some(p) = &mut self.prompt {
            let answer = p.show(ctx, env.icons);
            if let Some(a) = answer {
                self.prompt = None;
                let accepted = matches!(a, PromptAnswer::Accepted(_));
                match self.portal_step.take() {
                    Some(PortalStep::Confirm) if accepted => {
                        let start = last_dir(env.settings);
                        if let Some(file) =
                            env.files
                                .open_file(LOAD_TITLE, start.as_deref(), PORTAL_FILES)
                            && !file.as_os_str().is_empty()
                        {
                            self.ask(DELETE_QUESTION, PortalStep::Delete(file));
                        }
                    }
                    Some(PortalStep::Delete(file)) => {
                        out.result = Some(DialogResult::NewTrial(NewTrialAnswer::LoadPortal {
                            dialog: Box::new(self.clone()),
                            file,
                            delete: accepted,
                        }));
                        out.close = true;
                    }
                    _ => {}
                }
            }
            return out;
        }
        let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        if act.is_none() && modal.should_close() && escape {
            out.close = true;
            return out;
        }
        let start = last_dir(env.settings);
        match act {
            None => {}
            Some(Act::Minus) => self.minus(),
            Some(Act::Plus) => self.plus(),
            Some(Act::Folder(i)) => {
                if let Some(folder) = env.files.pick_folder(FOLDER_TITLE, start.as_deref())
                    && !folder.as_os_str().is_empty()
                    && let Some(b) = self.cameras.get_mut(i)
                {
                    b.set_folder(&folder);
                    // `setLastUsedDirectory(folder)`: the folder's parent.
                    env.settings.set_last_used_directory_from_file(&folder);
                    out.settings_changed = true;
                }
            }
            Some(Act::Files(i)) => {
                let files = env
                    .files
                    .open_files(FILES_TITLE, start.as_deref(), TRIAL_FILES);
                if !files.is_empty()
                    && let Some(b) = self.cameras.get_mut(i)
                {
                    b.set_files(files);
                    env.settings
                        .set_last_used_directory_from_file(std::path::Path::new(&b.text));
                    out.settings_changed = true;
                }
            }
            Some(Act::LoadXma) => self.ask(LOAD_CONFIRM, PortalStep::Confirm),
            Some(Act::Default) => {
                out.result = Some(DialogResult::NewTrial(NewTrialAnswer::DefaultTrial));
                out.close = true;
            }
            Some(Act::Ok) => {
                if let Some(a) = self.ok(env.doc.map(Document::project)) {
                    out.result = Some(DialogResult::NewTrial(a));
                    out.close = true;
                }
            }
            Some(Act::Cancel) => out.close = true,
        }
        out
    }

    /// A Yes/No confirmation over the form for a step of "Load XMA Portal file".
    fn ask(&mut self, message: &str, step: PortalStep) {
        self.prompt = Some(Prompt::Confirm {
            message: message.to_owned(),
            buttons: super::ConfirmButtons::YesNo,
        });
        self.portal_step = Some(step);
    }

    /// The dialog filled from an imported XMA Portal trial (`loadXMAPortalTrial` `:986-1007`):
    /// the trial name, each camera's files ([`xma_engine::portal::assign_trial`], boxes
    /// added when the number of cameras is the user's, C54), and the metadata. Returns the
    /// Console lines for files no camera was found for.
    pub fn fill_from_portal(
        &mut self,
        import: &xma_engine::portal::TrialImport,
        portal_ids: &[i32],
    ) -> Vec<String> {
        let a = xma_engine::portal::assign_trial(
            import,
            portal_ids,
            self.cameras.len(),
            self.count_visible(),
        );
        while self.cameras.len() < a.boxes {
            self.plus();
        }
        if let Some(n) = &a.trial_name {
            self.name.clone_from(n);
        }
        for (b, files, path) in a.cameras {
            if let Some(c) = self.cameras.get_mut(b) {
                c.set_portal_file(&path, files);
            }
        }
        self.portal_metadata = Some(import.metadata.clone());
        a.skipped
    }

    /// The form, in the `.ui` file's grid order.
    fn form(&mut self, ui: &mut Ui, icons: &crate::icons::Icons) -> Option<Act> {
        let mut act = None;
        ui.set_width(700.0);
        if form::dialog_title(ui, self.title()) {
            act = Some(Act::Cancel);
        }
        form::group_box(ui, "Name", |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.name).desired_width(f32::INFINITY))
                .accessible_name("Trial name");
        });
        if self.count_visible() {
            form::styled_frame(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Number of cameras");
                    if ui
                        .add(egui::Button::new("-").min_size(egui::vec2(22.0, 0.0)))
                        .accessible_name("Number of cameras -")
                        .clicked()
                    {
                        act = Some(Act::Minus);
                    }
                    ui.label(self.cameras.len().to_string());
                    if ui
                        .add(egui::Button::new("+").min_size(egui::vec2(22.0, 0.0)))
                        .accessible_name("Number of cameras +")
                        .clicked()
                    {
                        act = Some(Act::Plus);
                    }
                });
            });
        }
        form::group_box(ui, "Videopaths", |ui| {
            egui::ScrollArea::vertical()
                .id_salt("new_trial_cameras")
                .max_height(300.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for (i, b) in self.cameras.iter().enumerate() {
                        ui.push_id(i, |ui| {
                            if let Some(a) = camera_box(ui, icons, b, i) {
                                act = Some(a);
                            }
                        });
                    }
                });
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if form::button(ui, true, "Load XMA Portal file", 0.0).clicked() {
                act = Some(Act::LoadXma);
            }
            ui.add_space(40.0);
            if form::button(ui, self.default_enabled, "Add Default Trial", 0.0).clicked() {
                act = Some(Act::Default);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if form::button(ui, true, "Ok", 80.0).clicked() {
                    act = Some(Act::Ok);
                }
                if form::button(ui, true, "Cancel", 80.0).clicked() {
                    act = Some(Act::Cancel);
                }
            });
        });
        act
    }
}

/// `pushButton_Default` is enabled: the project has no Default trial and has a
/// calibration.
#[must_use]
pub fn default_allowed(p: &Project) -> bool {
    !p.trials.iter().any(|t| t.is_default) && p.calibration_type != CalibrationType::None
}

/// One `CameraBoxTrial`; returns the button clicked.
fn camera_box(
    ui: &mut Ui,
    icons: &crate::icons::Icons,
    b: &CameraBoxTrial,
    i: usize,
) -> Option<Act> {
    let mut act = None;
    let name = b.name.clone();
    form::group_box(ui, &name, |ui| {
        ui.horizontal(|ui| {
            ui.label(b.count_text.as_deref().unwrap_or("Video path"));
            let w = (ui.available_width() - 90.0).max(150.0);
            line_edit(ui, &b.text, w, &format!("{name}: Video path"));
            if form::icon_button(
                ui,
                icons,
                Icon::Images,
                true,
                false,
                &format!("{name}: {}", FOLDER_TITLE.trim_end()),
                30.0,
                false,
            )
            .clicked()
            {
                act = Some(Act::Folder(i));
            }
            if form::icon_button(
                ui,
                icons,
                Icon::Video,
                true,
                false,
                &format!("{name}: {FILES_TITLE}"),
                30.0,
                false,
            )
            .clicked()
            {
                act = Some(Act::Files(i));
            }
        });
    });
    act
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_sorted_naturally_and_cancel_keeps_them() {
        let mut b = CameraBoxTrial::new("Cam");
        b.set_files(vec![
            "/v/f10.png".into(),
            "/v/f2.png".into(),
            "/v/f1.png".into(),
        ]);
        assert_eq!(
            b.files,
            vec![
                PathBuf::from("/v/f1.png"),
                PathBuf::from("/v/f2.png"),
                PathBuf::from("/v/f10.png")
            ]
        );
        assert_eq!(b.text, "/v/f1");
        assert_eq!(b.count_text.as_deref(), Some("(3)"));
        b.set_files(Vec::new());
        assert_eq!(b.files.len(), 3);
    }

    #[test]
    fn without_calibration_counts_cameras() {
        let mut d = NewTrialDialog::without_calibration();
        assert!(d.count_visible());
        assert!(!d.default_enabled);
        assert_eq!(d.title(), TITLE);
        d.plus();
        d.plus();
        assert_eq!(d.cameras[2].name, "Camera 3");
        d.minus();
        d.minus();
        d.minus();
        assert_eq!(d.cameras.len(), 1);
    }
}
