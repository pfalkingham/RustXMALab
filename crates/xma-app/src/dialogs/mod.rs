//! Modal dialogs inside the main window (PLAN.md section 5: modal dialogs open inside the
//! main window; tool dialogs that stay open while you work are separate windows).
//!
//! Ported here:
//!
//! - `ErrorDialog` (ErrorDialog.ui): title "Error", the Error.png icon, the message, "OK".
//! - `ConfirmationDialog` (ConfirmationDialog.ui): title "Confirmation Required", the
//!   Alert.png icon, the message, "Yes" (default) and "Cancel".
//! - `QInputDialog::getInt` as the C++ calls it (title, label, value, range, OK/Cancel).
//! - The request for a missing video folder or file (`MainWindow::checkTrialImagePaths`).
//! - `AboutDialog` ([`about`]), `SettingsDialog` ([`settings_dialog`]) and
//!   `StatusColorDialog` ([`status_colors`]), with the message boxes About opens.
//! - `TrialSelectorDialog` (TrialSelectorDialog.ui): "Select the Trials to save", a list
//!   with multiple selection, "Ok" and "Cancel" (File > Save subdataset as).
//! - `TrialDialog` ([`trial`]): the workspace bar's trial settings.
//! - Two prompts XMALab 3 does not have (parity ledger U53, U54): "Save changes?" (Save,
//!   Don't Save, Cancel) before closing, opening or quitting with unsaved changes, and the
//!   offer to open a project's newer recovery file.
//! - `BlenderSceneDialog` ([`blender_scene`]), new in RustXMALab (ledger U176).
//!
//! The dialogs XMALab 3 opens with `show()` as non-modal windows (Help, Project Metadata,
//! Project Overview) are separate windows that stay open: [`windows`], held in
//! [`Dialogs::windows`] and drawn by [`Dialogs::show`] whether or not a modal dialog is open.
//! [`form`] has the Qt-like widgets the dialogs and the Toolbox pages share.
//!
//! A dialog is queued with [`Dialogs::open`]; one is shown at a time; [`Dialogs::show`]
//! returns what the user chose, and the application acts on it. To add a dialog: add a
//! variant to [`Dialog`], draw it in [`Dialogs::show`], and add a [`DialogResult`] if it
//! answers something.

pub mod about;
pub mod ask;
pub mod blender_scene;
pub mod camera_selector;
pub mod detection_settings;
pub mod edit_forms;
pub mod exports;
pub mod file_names;
pub mod form;
pub mod from_to;
pub mod help;
pub mod images;
pub mod import_export_points;
pub mod marker_dialog;
pub mod metadata;
pub mod new_project;
pub mod new_trial;
pub mod optimization;
pub mod overview;
pub mod point_import_export;
pub mod points;
pub mod portal;
pub mod portal_metadata;
pub mod rigid_body;
pub mod settings_dialog;
pub mod status_colors;
pub mod trial;
pub mod trial_import_delete;
pub mod undistort_sequence;
pub mod windows;

use std::collections::VecDeque;

use crate::actions::Action;
use crate::icons::{Icon, Icons};
use crate::session::MissingVideo;
use crate::settings::UiSettings;
use windows::{ToolWindow, ToolWindows};

/// The environment variable that opens a dialog at start-up, for screenshots: `about`,
/// `settings`, `settings-calibration`, `settings-tracking`, `status-colors`, `trial`, `help`,
/// `welcome`, `metadata` or `overview`; `points-marker`, `points-range`, `points-import`
/// (the Marker dialog of marker 1, "Select Range", Import/Export, for the first trial);
/// `rigid-body` (body 1 of the active trial) or `import-2d`; `new-project`,
/// `new-trial` (for the open project) or `new-trial-without-calibration`;
/// `optimization` (`OptimizationDialog`); `blender-scene` (`BlenderSceneDialog`, Marker tracking
/// defaults, the project's active trial). A
/// window it names is drawn inside the main window so
/// that the screenshot shows it. Read only by this module.
pub const SHOW_DIALOG_ENV: &str = "XMALAB_RS_SHOW_DIALOG";

/// What the dialogs read and change besides themselves.
pub struct DialogEnv<'a> {
    pub icons: &'a Icons,
    pub settings: &'a mut UiSettings,
    pub doc: Option<&'a xma_engine::Document>,
    /// The file dialogs (scripted in tests); the Rigid body dialog opens them.
    pub files: &'a mut crate::saving::FileChooser,
}

/// What an integer prompt is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntPurpose {
    /// Ctrl+5: set N, then move back.
    FrameAdvanceBackward,
    /// Ctrl+6: set N, then move forward.
    FrameAdvanceForward,
    /// The frame bar's start-frame button.
    StartFrame,
    /// The frame bar's end-frame button.
    EndFrame,
    /// File > New dataset (external calibration): "Set number of Cameras" (C53).
    ExternalCameras,
    /// "optimize camera setup" without `ShowAdvancedCalibration`: "Set number of
    /// iterations" (C37).
    OptimizeIterations,
    /// Undistortion page, "remove outlier at border": "Threshold (Border)" (C8).
    GridBorder,
    /// Undistortion page, "remove outlier at border": "Threshold (Circle)", after the
    /// border threshold was answered (C8).
    GridCircle { border: i32 },
    /// Calibration page, a reference's tool button: "Set Custom Idx for Reference n" for
    /// reference `n - 1` (C14).
    ReferenceIndex(usize),
}

/// A modal dialog.
#[derive(Clone, Debug, PartialEq)]
pub enum Dialog {
    Error {
        message: String,
    },
    Confirm {
        message: String,
        /// Raised when the user answers Yes (or OK).
        on_yes: Action,
        /// The two buttons' labels.
        buttons: ConfirmButtons,
    },
    GetInt {
        title: String,
        label: String,
        value: i32,
        min: i32,
        max: i32,
        purpose: IntPurpose,
    },
    MissingVideo(MissingVideo),
    /// `AboutDialog`.
    About,
    /// `SettingsDialog`.
    Settings(settings_dialog::SettingsDialog),
    /// `StatusColorDialog`.
    StatusColors(status_colors::StatusColorDialog),
    /// A `QMessageBox` with a title and text, shown above the dialog that opened it.
    Message {
        title: String,
        text: String,
    },
    /// `WelcomeDialog`: modal (`modal` is true in its form), opened with `show()` at
    /// start-up. `checked` is its check box.
    Welcome {
        checked: bool,
    },
    /// "Save changes?" before `then` (close, open another project, quit) while the
    /// document has unsaved changes (ledger U53).
    SaveChanges {
        /// The project's name as the title shows it.
        name: String,
        then: Action,
    },
    /// A project's recovery file is newer than the project file (ledger U54).
    Recover {
        project: std::path::PathBuf,
        recovery: std::path::PathBuf,
    },
    /// `TrialSelectorDialog`: the trials to save to `path` (File > Save subdataset as).
    TrialSelector {
        path: std::path::PathBuf,
        names: Vec<String>,
        /// Selected rows, by trial index.
        selected: std::collections::BTreeSet<usize>,
    },
    /// `TrialDialog`: the trial settings of the workspace bar.
    Trial(trial::TrialDialog),
    /// A question asked before the project changes ([`ask`]), raised by a panel with
    /// `EditRequest::Ask`.
    Ask(Box<ask::Ask>),
    /// A step of the Points panel's editing flows (input prompts, `FromToDialog`,
    /// `MarkerDialog`, `ImportExportPointsDialog`, confirmations): [`points`].
    Points(Box<points::PointsDialog>),
    /// `RigidBodyDialog` ([`rigid_body`]).
    RigidBody(Box<rigid_body::RigidBodyDialog>),
    /// `PointImportExportDialog` ([`point_import_export`]).
    PointImportExport(point_import_export::PointImportExportDialog),
    /// `TrialImportDeleteDialog` ([`trial_import_delete`]).
    TrialImportDelete(trial_import_delete::TrialImportDeleteDialog),
    /// `NewProjectDialog` ([`new_project`]): File > New dataset.
    NewProject(Box<new_project::NewProjectDialog>),
    /// `NewTrialDialog` ([`new_trial`]): a new trial, a trial without calibration, or
    /// "Change Trial Data".
    NewTrial(Box<new_trial::NewTrialDialog>),
    /// A step of a File > Export flow (`PointImportExportDialog`, a Yes/No question, the
    /// cutoff prompt, `FromToDialog`): [`exports`].
    Export(Box<exports::ExportDialog>),
    /// `CameraSelector` ([`camera_selector`]).
    CameraSelector(camera_selector::CameraSelector),
    /// `OptimizationDialog` ([`optimization`], C42).
    Optimization(optimization::OptimizationDialog),
    /// `UndistortSequenceDialog` ([`undistort_sequence`], C64): File > Undistort sequence.
    UndistortSequence(Box<undistort_sequence::UndistortSequenceDialog>),
    /// Project Metadata's "Add metadata from XMAPortal file" ([`portal_metadata`]).
    PortalMetadata(Box<portal_metadata::AddMetadataFlow>),
    /// `DenoiseDialog` ([`images`], C65): the Trial dialog's "Denoise Images".
    Denoise(Box<images::DenoiseDialog>),
    /// `VisualFilterDialog` ([`images`], C67): Display Options, beside "filter images".
    VisualFilter(images::VisualFilterDialog),
    /// `BlenderSceneDialog` ([`blender_scene`]): File > Export > Blender scene (new in
    /// RustXMALab, ledger U176).
    BlenderScene(Box<blender_scene::BlenderSceneDialog>),
}

/// The answer to "Save changes?".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveChoice {
    Save,
    DontSave,
}

/// The buttons of the confirmation dialog (`ConfirmationDialog::showConfirmationDialog`,
/// ui/ConfirmationDialog.cpp:62-72): "OK" and "Cancel" by default, "Yes" and "No" when the
/// caller passes `NoButton = true`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConfirmButtons {
    #[default]
    OkCancel,
    YesNo,
}

impl ConfirmButtons {
    /// The accepting and the rejecting button's text.
    #[must_use]
    pub fn labels(self) -> (&'static str, &'static str) {
        match self {
            ConfirmButtons::OkCancel => ("OK", "Cancel"),
            ConfirmButtons::YesNo => ("Yes", "No"),
        }
    }
}

/// What the About dialog's "Show OpenGL Info" shows: the graphics adapter wgpu uses (the
/// C++ shows OpenGL's vendor, renderer, version, shading-language version and extensions,
/// `GLSharedWidget::getInfo`). Filled by whoever holds the wgpu render state, with
/// [`Dialogs::set_graphics`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphicsInfo {
    /// Adapter name (`wgpu::AdapterInfo::name`).
    pub name: String,
    /// Backend (Vulkan, Metal, Dx12, Gl, ...).
    pub backend: String,
    /// Driver name (`driver`).
    pub driver: String,
    /// Driver version information (`driver_info`).
    pub driver_info: String,
}

/// The user's answer.
#[derive(Clone, Debug, PartialEq)]
pub enum DialogResult {
    /// Yes in a confirmation.
    Confirmed(Action),
    /// OK in an integer prompt.
    Int(IntPurpose, i32),
    /// A folder or file was chosen for a missing video.
    VideoChosen(MissingVideo, std::path::PathBuf),
    /// "Save" or "Don't Save" in "Save changes?" (Cancel gives no answer).
    SaveChanges(SaveChoice, Action),
    /// The answer to the recovery offer: open the recovery file (`recovered`) or the
    /// project file. Cancel gives no answer.
    Recover {
        project: std::path::PathBuf,
        recovery: std::path::PathBuf,
        recovered: bool,
    },
    /// "Ok" in the trial selector.
    TrialsChosen {
        path: std::path::PathBuf,
        trials: Vec<usize>,
    },
    /// OK or "Force Update Trial" in the Trial dialog: the values the user changed.
    TrialProperties {
        trial: usize,
        properties: xma_engine::edit::TrialProperties,
    },
    /// The algorithm settings the user changed in the Settings dialog (E73 to E79).
    AlgorithmSettings(xma_model::SettingsPatch),
    /// OK or Yes in an [`ask::Ask`]: the requests and actions it leads to.
    Answered(ask::Answered),
    /// A step of the Points panel's flows answered ([`points::answer`]).
    Points(points::PointsOutcome),
    /// A dialog asks to change the project (the Rigid body dialog's controls).
    Request(crate::editing::EditRequest),
    /// "OK" in `PointImportExportDialog`.
    PointImportExport(point_import_export::ImportExportKind),
    /// "Import" in `TrialImportDeleteDialog`: the trials of `path` to import, and the answer
    /// to replacing the Default trial.
    TrialsToImport {
        path: std::path::PathBuf,
        trials: Vec<String>,
        replace_default: bool,
    },
    /// "Delete", confirmed, in `TrialImportDeleteDialog`: trial indices.
    TrialsToDelete(Vec<usize>),
    /// "Ok" in `NewProjectDialog`, its check passed: the specification, and the dialog to
    /// show again if the creation fails.
    NewProject {
        spec: xma_engine::create::NewProjectSpec,
        dialog: Box<new_project::NewProjectDialog>,
    },
    /// "Ok" or "Add Default Trial" in `NewTrialDialog`.
    NewTrial(new_trial::NewTrialAnswer),
    /// A step of an export flow answered ([`exports::answer`]).
    Export(exports::ExportOutcome),
    /// OK in `CameraSelector`: each camera's visibility.
    CamerasVisible(Vec<bool>),
    /// OK in `OptimizationDialog` (C42): method 0, 1 or 2, iterations, scale factor for the
    /// initial λ.
    Optimize {
        method: i32,
        iterations: usize,
        initial: f64,
    },
    /// "Undistort" in `UndistortSequenceDialog`: the queue (C64).
    UndistortSequence(Vec<xma_engine::undistort_sequence::UndistortItem>),
    /// "Denoise Images" and `DenoiseDialog` ([`images`]).
    Images(images::ImagesAnswer),
    /// OK in `BlenderSceneDialog`: what to export (the folder is asked next).
    BlenderScene(Box<xma_engine::blender_scene::BlenderSceneSpec>),
}

/// The text of "Save changes?".
#[must_use]
pub fn save_changes_text(name: &str) -> String {
    format!("The dataset {name} has unsaved changes.\nSave changes?")
}

/// The text of the recovery offer.
#[must_use]
pub fn recover_text(project: &std::path::Path, recovery: &std::path::Path) -> String {
    format!(
        "A recovery copy of {} was saved after the dataset itself:\n{}\n\nOpen the recovered version? Its changes are not in the dataset file until you save it.",
        project.display(),
        recovery.display()
    )
}

/// The queue of modal dialogs.
#[derive(Debug, Default)]
pub struct Dialogs {
    queue: VecDeque<Dialog>,
    /// Edited value of the integer prompt on screen.
    int_value: Option<i32>,
    /// A message box over the dialog on screen (About's licence and graphics information).
    child: Option<Dialog>,
    /// The windows that stay open (Help, Welcome, Project Metadata, Project Overview).
    pub windows: ToolWindows,
    /// A dialog changed a setting since [`Dialogs::take_settings_changed`].
    settings_changed: bool,
    /// [`SHOW_DIALOG_ENV`] has been read.
    env_read: bool,
    /// The graphics adapter, once known.
    pub graphics: Option<GraphicsInfo>,
    /// [`SHOW_DIALOG_ENV`] named `detection-settings`: the application opens View >
    /// DetectionSettings inside the main window (screenshots).
    pub detection_settings_requested: bool,
}

impl Dialogs {
    pub fn open(&mut self, d: Dialog) {
        // A missing video is asked for once per trial camera.
        if let Dialog::MissingVideo(m) = &d
            && self.queue.iter().any(|q| {
                matches!(q, Dialog::MissingVideo(o) if o.trial == m.trial && o.camera == m.camera)
            })
        {
            return;
        }
        self.queue.push_back(d);
    }

    /// Number of queued dialogs.
    #[must_use]
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Answers the missing-video request on screen with `chosen` (a folder or file), as the
    /// buttons do after the operating system's file dialog. `None` when the dialog on screen
    /// is not a missing-video request. Used by tests.
    pub fn answer_missing(&mut self, chosen: std::path::PathBuf) -> Option<DialogResult> {
        let Some(Dialog::MissingVideo(m)) = self.queue.front().cloned() else {
            return None;
        };
        self.queue.pop_front();
        Some(DialogResult::VideoChosen(m, chosen))
    }

    /// Drops the queued missing-video requests for which `keep` is false (for example those
    /// already resolved, or all of them when the project is closed).
    pub fn retain_missing(&mut self, keep: impl Fn(&MissingVideo) -> bool) {
        self.queue
            .retain(|q| !matches!(q, Dialog::MissingVideo(m) if !keep(m)));
    }

    /// Shows an error (`ErrorDialog::showErrorDialog`).
    pub fn error(&mut self, message: impl Into<String>) {
        let d = Dialog::Error {
            message: message.into(),
        };
        // Over a dialog that stays open (the Rigid body dialog), as XMALab's error dialog
        // opens over it.
        if self
            .queue
            .front()
            .is_some_and(edit_forms::stays_under_errors)
        {
            self.queue.push_front(d);
        } else {
            self.open(d);
        }
    }

    #[must_use]
    pub fn is_open(&self) -> bool {
        !self.queue.is_empty()
    }

    /// The dialog on screen.
    #[must_use]
    pub fn current(&self) -> Option<&Dialog> {
        self.queue.front()
    }

    /// The dialog on screen, to change its values as its controls would (tests).
    pub fn current_mut(&mut self) -> Option<&mut Dialog> {
        self.queue.front_mut()
    }

    /// Closes the dialog on screen without an answer.
    pub fn dismiss(&mut self) {
        self.queue.pop_front();
        self.int_value = None;
        self.child = None;
    }

    /// Opens a window that stays open (Help, Project Metadata, Project Overview), or the
    /// Welcome dialog, which is modal.
    pub fn open_window(&mut self, w: ToolWindow, doc: Option<&xma_engine::Document>) {
        if w == ToolWindow::Welcome {
            if !self
                .queue
                .iter()
                .any(|d| matches!(d, Dialog::Welcome { .. }))
            {
                self.open(Dialog::Welcome { checked: false });
            }
        } else {
            self.windows.open(w, doc);
        }
    }

    /// Records the graphics adapter for the About dialog: adapter name, backend, driver and
    /// driver information, as wgpu's `AdapterInfo` gives them.
    pub fn set_graphics(&mut self, name: &str, backend: &str, driver: &str, driver_info: &str) {
        self.graphics = Some(GraphicsInfo {
            name: name.to_owned(),
            backend: backend.to_owned(),
            driver: driver.to_owned(),
            driver_info: driver_info.to_owned(),
        });
    }

    /// True once after a dialog changed a setting, so the application saves the settings.
    pub fn take_settings_changed(&mut self) -> bool {
        std::mem::take(&mut self.settings_changed)
    }

    /// The message box over the dialog on screen, if any.
    #[must_use]
    pub fn child(&self) -> Option<&Dialog> {
        self.child.as_ref()
    }

    /// Closes the Trial dialog wherever it is in the queue (after its "Delete Trial" was
    /// confirmed, `TrialDialog::on_pushButton_DeleteTrial_clicked` rejects the dialog).
    pub fn close_trial_dialog(&mut self) {
        self.queue.retain(|d| !matches!(d, Dialog::Trial(_)));
    }

    /// Opens what [`SHOW_DIALOG_ENV`] names (once).
    fn open_from_env(&mut self, doc: Option<&xma_engine::Document>) {
        if std::mem::replace(&mut self.env_read, true) {
            return;
        }
        let Ok(v) = std::env::var(SHOW_DIALOG_ENV) else {
            return;
        };
        let page = |p| {
            Dialog::Settings(settings_dialog::SettingsDialog {
                page: p,
                ..Default::default()
            })
        };
        let window = |w: ToolWindow, s: &mut Self| {
            s.windows.embedded = Some(w);
            s.windows.open(w, doc);
        };
        match v.as_str() {
            "about" => self.open(Dialog::About),
            "settings" => self.open(page(settings_dialog::Page::General)),
            "settings-calibration" => self.open(page(settings_dialog::Page::Calibration)),
            "settings-tracking" => self.open(page(settings_dialog::Page::Tracking)),
            "status-colors" => self.open(Dialog::StatusColors(Default::default())),
            "trial" => {
                if let Some(d) =
                    doc.and_then(|d| trial::TrialDialog::new(d, d.project().active_trial))
                {
                    self.open(Dialog::Trial(d));
                }
            }
            "help" => window(ToolWindow::Help, self),
            "welcome" => self.open(Dialog::Welcome { checked: false }),
            "metadata" => window(ToolWindow::Metadata, self),
            "overview" => window(ToolWindow::Overview, self),
            "detection-settings" => self.detection_settings_requested = true,
            "points-marker" | "points-range" | "points-import" => {
                if let Some(d) = doc.and_then(|d| points::for_screenshot(&v, d)) {
                    self.open(Dialog::Points(Box::new(d)));
                }
            }
            "rigid-body" => {
                if let Some(d) = doc {
                    let t = d.project().active_trial;
                    self.open(Dialog::RigidBody(Box::new(
                        rigid_body::RigidBodyDialog::new(d, t, 0, 0),
                    )));
                }
            }
            "new-project" => self.open(Dialog::NewProject(Box::default())),
            "new-trial" => {
                if let Some(d) = doc {
                    self.open(Dialog::NewTrial(Box::new(new_trial::NewTrialDialog::new(
                        d,
                    ))));
                }
            }
            "new-trial-without-calibration" => self.open(Dialog::NewTrial(Box::new(
                new_trial::NewTrialDialog::without_calibration(),
            ))),
            "optimization" => self.open(Dialog::Optimization(Default::default())),
            "undistort-sequence" => {
                if let Some(d) = doc {
                    self.open(Dialog::UndistortSequence(Box::new(
                        undistort_sequence::UndistortSequenceDialog::new(d, None),
                    )));
                }
            }
            "denoise" => self.open(Dialog::Denoise(Box::new(images::DenoiseDialog::new(
                0,
                std::path::PathBuf::new(),
            )))),
            "visual-filter" => self.open(Dialog::VisualFilter(images::VisualFilterDialog)),
            "blender-scene" => {
                if let Some(d) = doc {
                    self.open(Dialog::BlenderScene(Box::new(
                        blender_scene::BlenderSceneDialog::new(
                            d,
                            true,
                            Some(d.project().active_trial),
                            xma_engine::blender_scene::locate_blender(None),
                        ),
                    )));
                }
            }
            "trial-images" => {
                if let Some(d) = doc
                    && let exports::Step::Open(e) = exports::begin(
                        &exports::ExportAction::TrialImages,
                        d,
                        Some(d.project().active_trial),
                    )
                {
                    self.open(Dialog::Export(Box::new(e)));
                }
            }
            "import-2d" => self.open(Dialog::PointImportExport(
                point_import_export::PointImportExportDialog::new(
                    point_import_export::ImportExportKind::Import2d,
                ),
            )),
            _ => {}
        }
    }

    /// Answers the dialog on screen as if its default button were pressed (tests).
    pub fn accept(&mut self) -> Option<DialogResult> {
        let d = self.queue.pop_front()?;
        let v = self.int_value.take();
        match d {
            Dialog::Confirm { on_yes, .. } => Some(DialogResult::Confirmed(on_yes)),
            Dialog::SaveChanges { then, .. } => {
                Some(DialogResult::SaveChanges(SaveChoice::Save, then))
            }
            Dialog::Recover { project, recovery } => Some(DialogResult::Recover {
                project,
                recovery,
                recovered: true,
            }),
            Dialog::TrialSelector { path, selected, .. } => Some(DialogResult::TrialsChosen {
                path,
                trials: selected.into_iter().collect(),
            }),
            Dialog::GetInt { value, purpose, .. } => {
                Some(DialogResult::Int(purpose, v.unwrap_or(value)))
            }
            Dialog::Ask(a) => Some(DialogResult::Answered(a.accept())),
            Dialog::Points(p) => Some(p.accept()),
            Dialog::Export(e) => Some(e.accept()),
            Dialog::CameraSelector(c) => Some(DialogResult::CamerasVisible(c.checked)),
            Dialog::Optimization(o) => Some(o.result()),
            Dialog::Denoise(d) => Some(d.result()),
            Dialog::BlenderScene(d) => d.refusal().is_none().then(|| d.result()),
            _ => None,
        }
    }

    /// The question on screen, if it is an [`ask::Ask`] (tests choose an item or type a
    /// name through it before [`Dialogs::accept`]).
    pub fn ask_mut(&mut self) -> Option<&mut ask::Ask> {
        match self.queue.front_mut() {
            Some(Dialog::Ask(a)) => Some(a),
            _ => None,
        }
    }

    /// Draws the windows that stay open, then the modal dialog on screen, if any, and
    /// returns the answer when one is given.
    pub fn show(&mut self, ctx: &egui::Context, env: &mut DialogEnv<'_>) -> Option<DialogResult> {
        if env.doc.is_some() || self.env_read {
            self.open_from_env(env.doc);
        }
        self.windows.show(ctx, env.doc);
        if self.windows.take_add_metadata() && env.doc.is_some() {
            self.open(Dialog::PortalMetadata(Box::default()));
        }
        let icons = env.icons;
        if let Some(Dialog::Ask(a)) = self.queue.front_mut() {
            let mut answer = None;
            let modal = form::modal(ctx, egui::Id::unique("xma_ask_modal"), |ui| {
                answer = ask::show(ui, a, icons);
            });
            if answer.is_none() && modal.escaped() {
                // Escape rejects, except in an information box: its only button, OK, is
                // `QMessageBox`'s escape button.
                answer = Some(match a.kind {
                    ask::AskKind::Information { .. } => Some(a.accept()),
                    _ => None,
                });
            }
            let answer = answer?;
            self.queue.pop_front();
            return answer.map(DialogResult::Answered);
        }
        if self.queue.front().is_some_and(edit_forms::is_edit_form) {
            return self.show_edit_form(ctx, env);
        }
        if matches!(
            self.queue.front(),
            Some(
                Dialog::About
                    | Dialog::Settings(_)
                    | Dialog::StatusColors(_)
                    | Dialog::Welcome { .. }
                    | Dialog::Trial(_)
            )
        ) {
            return self.show_form(ctx, env);
        }
        if let Some(Dialog::Points(p)) = self.queue.front_mut() {
            let shown = points::show(p, ctx, env);
            self.settings_changed |= shown.settings_changed;
            if shown.close || shown.result.is_some() {
                self.queue.pop_front();
            }
            return shown.result;
        }
        if let Some(Dialog::Export(e)) = self.queue.front_mut() {
            let shown = exports::show(e, ctx, env);
            self.settings_changed |= shown.settings_changed;
            if shown.close || shown.result.is_some() {
                self.queue.pop_front();
            }
            return shown.result;
        }
        if let Some(Dialog::CameraSelector(c)) = self.queue.front_mut() {
            let mut answer = None;
            let mut close = false;
            let modal = form::modal(ctx, egui::Id::unique("xma_camera_selector"), |ui| {
                if form::dialog_title(ui, "Select Cameras") {
                    close = true;
                }
                answer = c.ui(ui);
            });
            if modal.escaped() {
                close = true;
            }
            let result = match answer {
                Some(camera_selector::CameraSelectorAnswer::Ok) => {
                    Some(DialogResult::CamerasVisible(c.checked.clone()))
                }
                Some(camera_selector::CameraSelectorAnswer::Cancel) => {
                    close = true;
                    None
                }
                None => None,
            };
            if close || result.is_some() {
                self.queue.pop_front();
            }
            return result;
        }
        if let Some(Dialog::Optimization(o)) = self.queue.front_mut() {
            let mut answer = None;
            let modal = form::modal(ctx, egui::Id::unique("xma_optimization"), |ui| {
                answer = o.ui(ui);
            });
            let result = match answer {
                Some(optimization::OptimizationAnswer::Ok) => Some(o.result()),
                Some(optimization::OptimizationAnswer::Cancel) => None,
                None if modal.escaped() => {
                    answer = Some(optimization::OptimizationAnswer::Cancel);
                    None
                }
                None => None,
            };
            if answer.is_some() {
                self.queue.pop_front();
            }
            return result;
        }
        let d = self.queue.front()?.clone();
        let mut close = false;
        let mut result = None;
        let title = match &d {
            Dialog::Error { .. } => "Error".to_owned(),
            Dialog::Confirm { .. } => "Confirmation Required".to_owned(),
            Dialog::GetInt { title, .. } => title.clone(),
            Dialog::MissingVideo(_) | Dialog::Recover { .. } => crate::APP_NAME.to_owned(),
            Dialog::SaveChanges { .. } => "Save changes?".to_owned(),
            Dialog::TrialSelector { .. } => "Select the Trials to save".to_owned(),
            Dialog::Message { title, .. } => title.clone(),
            Dialog::About
            | Dialog::Settings(_)
            | Dialog::StatusColors(_)
            | Dialog::Welcome { .. }
            | Dialog::Trial(_)
            | Dialog::Points(_)
            | Dialog::RigidBody(_)
            | Dialog::PointImportExport(_)
            | Dialog::TrialImportDelete(_)
            | Dialog::NewProject(_)
            | Dialog::NewTrial(_)
            | Dialog::Export(_)
            | Dialog::CameraSelector(_)
            | Dialog::Optimization(_)
            | Dialog::UndistortSequence(_)
            | Dialog::PortalMetadata(_) => String::new(),
            Dialog::Denoise(_) | Dialog::VisualFilter(_) | Dialog::BlenderScene(_) => String::new(),
            Dialog::Ask(a) => a.title.clone(),
        };
        let mut toggled: Option<usize> = None;
        let modal = form::modal(ctx, egui::Id::unique("xma_modal"), |ui| {
            ui.set_min_width(360.0);
            ui.label(egui::RichText::new(title).strong());
            ui.add_space(6.0);
            match &d {
                Dialog::Error { message } => {
                    form::message_row(ui, icons, Icon::Error, message);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("OK").clicked() {
                            close = true;
                        }
                    });
                }
                Dialog::Confirm {
                    message,
                    on_yes,
                    buttons,
                } => {
                    form::message_row(ui, icons, Icon::Alert, message);
                    let (accept, reject) = buttons.labels();
                    match form::reject_accept(ui, accept, reject) {
                        Some(true) => result = Some(DialogResult::Confirmed(on_yes.clone())),
                        Some(false) => close = true,
                        None => {}
                    }
                }
                Dialog::GetInt {
                    label,
                    value,
                    min,
                    max,
                    purpose,
                    ..
                } => {
                    let v = self.int_value.get_or_insert(*value);
                    let l = ui.label(label.as_str()).id;
                    ui.add(egui::DragValue::new(v).range(*min..=*max))
                        .labelled_by(l);
                    let v = *v;
                    ui.add_space(6.0);
                    match form::reject_accept(ui, "OK", "Cancel") {
                        Some(true) => {
                            result = Some(DialogResult::Int(*purpose, v.clamp(*min, *max)))
                        }
                        Some(false) => close = true,
                        None => {}
                    }
                }
                Dialog::Message { text, .. } => {
                    message_body(ui, text, &mut close);
                }
                Dialog::About
                | Dialog::Settings(_)
                | Dialog::StatusColors(_)
                | Dialog::Welcome { .. }
                | Dialog::Trial(_)
                | Dialog::Ask(_)
                | Dialog::Points(_)
                | Dialog::RigidBody(_)
                | Dialog::PointImportExport(_)
                | Dialog::TrialImportDelete(_)
                | Dialog::NewProject(_)
                | Dialog::NewTrial(_)
                | Dialog::Export(_)
                | Dialog::CameraSelector(_)
                | Dialog::Optimization(_)
                | Dialog::UndistortSequence(_)
                | Dialog::PortalMetadata(_) => {}
                Dialog::Denoise(_) | Dialog::VisualFilter(_) | Dialog::BlenderScene(_) => {}
                Dialog::MissingVideo(m) => {
                    missing_video_body(ui, m, env.files, &mut result, &mut close);
                }
                Dialog::SaveChanges { name, then } => {
                    form::message_row(ui, icons, Icon::Alert, &save_changes_text(name));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Cancel").clicked() {
                            close = true;
                        }
                        if ui.button("Don't Save").clicked() {
                            result = Some(DialogResult::SaveChanges(
                                SaveChoice::DontSave,
                                then.clone(),
                            ));
                        }
                        if ui.button("Save").clicked() {
                            result =
                                Some(DialogResult::SaveChanges(SaveChoice::Save, then.clone()));
                        }
                    });
                }
                Dialog::Recover { project, recovery } => {
                    form::message_row(ui, icons, Icon::Alert, &recover_text(project, recovery));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Cancel").clicked() {
                            close = true;
                        }
                        let answer = |recovered| DialogResult::Recover {
                            project: project.clone(),
                            recovery: recovery.clone(),
                            recovered,
                        };
                        if ui.button("Open saved version").clicked() {
                            result = Some(answer(false));
                        }
                        if ui.button("Open recovered version").clicked() {
                            result = Some(answer(true));
                        }
                    });
                }
                Dialog::TrialSelector {
                    path,
                    names,
                    selected,
                } => {
                    // TrialSelectorDialog.ui: listWidget (MultiSelection: a click toggles
                    // the row), then "Ok" and "Cancel" on the right.
                    ui.set_width(330.0);
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        egui::ScrollArea::vertical()
                            .id_salt("trial_selector_list")
                            .min_scrolled_height(220.0)
                            .max_height(220.0)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                for (i, n) in names.iter().enumerate() {
                                    if ui.selectable_label(selected.contains(&i), n).clicked() {
                                        toggled = Some(i);
                                    }
                                }
                            });
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Cancel").clicked() {
                            close = true;
                        }
                        if ui.button("Ok").clicked() {
                            result = Some(DialogResult::TrialsChosen {
                                path: path.clone(),
                                trials: selected.iter().copied().collect(),
                            });
                        }
                    });
                }
            }
        });
        if let (Some(i), Some(Dialog::TrialSelector { selected, .. })) =
            (toggled, self.queue.front_mut())
            && !selected.remove(&i)
        {
            selected.insert(i);
        }
        // Escape is Qt's reject(); a click outside does nothing, as with a Qt modal dialog.
        if modal.escaped() {
            close = true;
        }
        if close || result.is_some() {
            self.queue.pop_front();
            self.int_value = None;
        }
        result
    }

    /// Draws the form dialogs: About (with its message box), Settings, Status colours,
    /// Trial; returns the Settings and Trial dialogs' answers.
    fn show_form(&mut self, ctx: &egui::Context, env: &mut DialogEnv<'_>) -> Option<DialogResult> {
        let front = self.queue.front_mut()?;
        let mut result: Option<DialogResult> = None;
        let mut patch = xma_model::SettingsPatch::default();
        let mut confirm: Option<Dialog> = None;
        let mut close = false;
        let mut next: Option<Dialog> = None;
        let mut child: Option<Dialog> = None;
        let mut error: Option<String> = None;
        let mut changed = false;
        let alg_default;
        let alg = match env.doc {
            Some(d) => d.settings(),
            None => {
                // No project: this machine's defaults.
                alg_default = env.settings.algorithm_base();
                &alg_default
            }
        };
        let modal = form::modal(ctx, egui::Id::unique("xma_form_modal"), |ui| match front {
            Dialog::About => match about::ui(ui, env.icons) {
                Some(about::AboutRequest::Close) => close = true,
                Some(about::AboutRequest::OpenGlInfo) => {
                    child = Some(Dialog::Message {
                        title: "OpenGL Info".into(),
                        text: about::graphics_info(self.graphics.as_ref()),
                    });
                }
                Some(about::AboutRequest::License) => {
                    child = Some(Dialog::Message {
                        title: about::LICENSE_BUTTON.into(),
                        text: about::LICENSE_TEXT.into(),
                    });
                }
                None => {}
            },
            Dialog::Settings(sd) => match sd.ui(ui, env.settings, alg, &mut changed, &mut patch) {
                Some(settings_dialog::SettingsRequest::Close) => close = true,
                Some(settings_dialog::SettingsRequest::StatusColors) => {
                    next = Some(Dialog::StatusColors(Default::default()));
                }
                Some(settings_dialog::SettingsRequest::ChooseWorkspace) => {
                    let start = last_dir(env.settings);
                    if let Some(p) = env.files.pick_folder("Load Directory ", start.as_deref()) {
                        let p = p.to_string_lossy().into_owned();
                        env.settings.workspace_path.clone_from(&p);
                        env.settings.last_directory_used = p;
                        changed = true;
                    }
                }
                None => {}
            },
            Dialog::StatusColors(sc) => {
                if sc.ui(ui, &mut env.settings.colors, &mut changed) {
                    close = true;
                }
            }
            Dialog::Welcome { checked } => {
                ui.set_width(740.0);
                if form::dialog_title(ui, "Welcome to RustXMALab") {
                    close = true;
                }
                about::banner(ui, env.icons);
                about::text_browser(ui, "welcome_text", about::WELCOME_TEXT, 560.0);
                // WelcomeDialog::on_checkBox_clicked: checking stores the version.
                if ui
                    .checkbox(checked, "Do not show this Dialog at startup")
                    .clicked()
                    && *checked
                {
                    env.settings.welcome_dialog = about::version().to_owned();
                    changed = true;
                }
            }
            Dialog::Trial(td) => match td.ui(ui) {
                Some(trial::TrialRequest::Close) => close = true,
                Some(r @ (trial::TrialRequest::Ok | trial::TrialRequest::ForceUpdate)) => {
                    let properties = td.properties(r == trial::TrialRequest::ForceUpdate);
                    if properties != xma_engine::edit::TrialProperties::default() {
                        result = Some(DialogResult::TrialProperties {
                            trial: td.trial,
                            properties,
                        });
                    }
                    close = true;
                }
                Some(trial::TrialRequest::ChangeData) => {
                    // TrialDialog::on_pushButton_ChangeTrialData_clicked: NewTrialDialog
                    // on the trial; the Trial dialog then closes (C61).
                    next = env
                        .doc
                        .and_then(|d| new_trial::NewTrialDialog::change(d, td.trial))
                        .map(|d| Dialog::NewTrial(Box::new(d)));
                    close = next.is_none();
                }
                Some(trial::TrialRequest::Denoise) => {
                    // TrialDialog::on_pushButton_Denoise_clicked: the dialog closes, then
                    // the folder question and DenoiseDialog (C65).
                    result = Some(DialogResult::Images(images::ImagesAnswer::DenoiseAsked {
                        trial: td.trial,
                    }));
                    close = true;
                }
                Some(trial::TrialRequest::Delete) => {
                    confirm = Some(Dialog::Confirm {
                        message: trial::DELETE_TEXT.to_owned(),
                        on_yes: Action::Confirmed(Box::new(Action::DeleteTrial(td.trial))),
                        buttons: ConfirmButtons::OkCancel,
                    });
                }
                None => {}
            },
            _ => {}
        });
        let has_chooser = matches!(front, Dialog::StatusColors(sc) if sc.chooser.is_some());
        if modal.escaped() && self.child.is_none() && !has_chooser {
            close = true;
        }
        // The message box over the dialog.
        if let Some(Dialog::Message { title, text }) = &self.child {
            let mut done = false;
            let m = egui::Modal::new(egui::Id::unique("xma_form_child")).show(ctx, |ui| {
                ui.set_max_width(820.0);
                if form::dialog_title(ui, title) {
                    done = true;
                }
                message_body(ui, text, &mut done);
            });
            // Escape as read before the dialog under it was drawn.
            if done || (m.should_close() && modal.escape) {
                self.child = None;
            }
        }
        if child.is_some() {
            self.child = child;
        }
        if close && let Some(Dialog::Settings(sd)) = self.queue.front_mut() {
            // A number still being edited is written as the dialog closes.
            sd.flush(&mut patch);
        }
        if close
            && matches!(self.queue.front(), Some(Dialog::Settings(_)))
            && !settings_dialog::may_close(env.settings)
        {
            // SettingsDialog::closeEvent refuses.
            close = false;
            error = Some(settings_dialog::WORKSPACE_ERROR.to_owned());
        }
        if let Some(n) = next {
            // Settings closes, then the status colours dialog opens.
            if let Some(f) = self.queue.front_mut() {
                *f = n;
            }
        } else if close {
            self.queue.pop_front();
            self.child = None;
        }
        if let Some(e) = error {
            self.queue.push_front(Dialog::Error { message: e });
        }
        // The confirmation shows over the Trial dialog, which comes back after Cancel.
        if let Some(c) = confirm {
            self.queue.push_front(c);
        }
        self.settings_changed |= changed;
        if !patch.is_empty() {
            result = Some(DialogResult::AlgorithmSettings(patch));
        }
        result
    }
}

/// The folder the file dialogs start in (`Settings::getLastUsedDirectory`): the last used
/// folder, or none when no folder was used yet.
pub(crate) fn last_dir(settings: &UiSettings) -> Option<std::path::PathBuf> {
    (!settings.last_directory_used.is_empty())
        .then(|| std::path::PathBuf::from(&settings.last_directory_used))
}

/// A message box's text and its OK button.
fn message_body(ui: &mut egui::Ui, text: &str, close: &mut bool) {
    egui::ScrollArea::vertical()
        .id_salt("xma_message_text")
        .max_height(600.0)
        .show(ui, |ui| {
            ui.add(egui::Label::new(text).wrap());
        });
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        if ui.button("OK").clicked() {
            *close = true;
        }
    });
}

/// The request for a missing video (`MainWindow::checkTrialImagePaths`).
///
/// The C++ opens its file dialog at once, titled "<name> not found. Enter a new directory
/// for <name>" (an image sequence: a folder) or "... Enter a file for <name>" (a video
/// file). Here that text is shown with a button for the same choice, and, for an image
/// sequence, a second button to use a single video file in its place (new: a dataset moved
/// between machines often keeps the videos and not the extracted images). Cancel (or
/// Escape) closes it without a choice; the views then say the file was not found.
/// Cancelling the operating system's file dialog leaves this dialog open.
fn missing_video_body(
    ui: &mut egui::Ui,
    m: &MissingVideo,
    files: &mut crate::saving::FileChooser,
    result: &mut Option<DialogResult>,
    close: &mut bool,
) {
    ui.label(m.prompt());
    ui.add_space(6.0);
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        if ui.button("Cancel").clicked() {
            *close = true;
        }
        if m.sequence
            && ui
                .button("Use a video file instead...")
                .on_hover_text("Show this camera from one video file (AVI, CINE, MP4...) instead of the images")
                .clicked()
            && let Some(p) = files.open_file_filters(
                &format!("Choose a video file for {}", m.file_name),
                None,
                &[
                    ("Video", &["avi", "cine", "mp4", "mov", "mkv", "m4v"]),
                    ("All files", &["*"]),
                ],
            )
        {
            *result = Some(DialogResult::VideoChosen(m.clone(), p));
        }
        let what = if m.sequence {
            "Choose folder..."
        } else {
            "Choose file..."
        };
        if ui.button(what).clicked() {
            let picked = if m.sequence {
                files.pick_folder(&m.prompt(), None)
            } else {
                files.open_file_filters(&m.prompt(), None, &[])
            };
            if let Some(p) = picked {
                *result = Some(DialogResult::VideoChosen(m.clone(), p));
            }
        }
    });
}
