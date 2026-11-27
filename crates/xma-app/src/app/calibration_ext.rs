//! "optimize camera setup" (C37, C42) and the external-calibration frame's two buttons (C43,
//! C44): the application's side.
//!
//! - [`Action::OptimizeCameraSetup`] opens the Optimization dialog when the setting
//!   `ShowAdvancedCalibration` is on, otherwise "Set number of iterations"; the answer
//!   (`DialogResult::Optimize`, or `DialogResult::Int(IntPurpose::OptimizeIterations, n)`
//!   with method 0 and factor 0.01) goes to [`XmaApp::apply_optimisation`].
//! - [`Action::LoadMayaCam2`] asks for the file ("Open MayaCam 2.0 file", `*.txt`), reads it
//!   with `xma_engine::calibration::read_mayacam2` and fills that camera's frame; a file that
//!   cannot be read shows the engine's sentence in the error dialog. Nothing is applied.
//! - "APPLY CHANGES" raises `EditRequest::Apply(Edit::Calibration(ApplyExternal))`;
//!   `apply_request` passes it to [`XmaApp::apply_calibration_ext`] (`Optimize` is a long
//!   edit, [`XmaApp::apply_optimisation`]).
//!
//! Errors of these edits are shown as XMALab's error dialog: `Rejected` (XMALab's text, for
//! example the engine's refusal of an optimisation with an uncalibrated camera, CD19), `Io`,
//! and also `NotImplemented` (the engine not built yet). `Invalid` goes to the Console. A
//! successful edit is followed by the usual refresh (`document_changed`: the Calibration
//! info frame and the views read the document); the external-calibration frames refill
//! themselves from the cameras that changed, and the calibration images' projected points
//! are computed again.
//!
//! Ported from `ui/WizardCalibrationCubeFrame.cpp`, `on_pushButtonOptimize_clicked`
//! (`:964-997`), and `ui/ExternalCalibrationFrame.cpp`, `on_pushButton_MayaCam_clicked`,
//! `on_pushButton_Apply_clicked`.

use xma_engine::calibration::CalibrationEdit;
use xma_engine::{Edit, EditError};

use super::XmaApp;
use crate::actions::Action;
use crate::dialogs::Dialog;
use crate::dialogs::optimization::{OptimizationDialog, iterations_prompt};

/// The title of the MayaCam file dialog (C43).
pub const MAYACAM_TITLE: &str = "Open MayaCam 2.0 file";

/// What this part of the application remembers.
#[derive(Debug, Default)]
pub struct CalibrationExt {
    /// The last `ApplyExternal` or `Optimize` edit handed to the engine (whatever its
    /// outcome); for tests.
    pub last_edit: Option<CalibrationEdit>,
}

impl XmaApp {
    /// [`Action::OptimizeCameraSetup`] and [`Action::LoadMayaCam2`], from `dispatch`.
    /// Returns false for other actions.
    pub(super) fn dispatch_calibration_ext(&mut self, a: &Action) -> bool {
        match a {
            Action::OptimizeCameraSetup => {
                if self.settings.show_advanced_calibration {
                    self.dialogs
                        .open(Dialog::Optimization(OptimizationDialog::default()));
                } else {
                    self.dialogs.open(iterations_prompt());
                }
            }
            Action::LoadMayaCam2(camera) => self.load_mayacam2(*camera),
            _ => return false,
        }
        true
    }

    /// Applies "optimize camera setup" with the dialog's or the prompt's values. It can take
    /// seconds; XMALab shows "Optimize camera setup" meanwhile.
    pub fn apply_optimisation(&mut self, method: i32, iterations: usize, initial: f64) {
        let edit = CalibrationEdit::Optimize {
            method,
            iterations,
            initial,
        };
        // On a worker thread, so that the window is not blocked while the engine works
        // (`app/long_edit.rs`); a refusal comes back as XMALab's error dialog.
        if self.session.doc.is_none() {
            return;
        }
        self.calibration_ext.last_edit = Some(edit.clone());
        self.start_long_edit(Edit::Calibration(edit), "Optimize camera setup");
    }

    /// Applies an external-calibration or optimisation edit now (one undo step) and shows its
    /// error, as the module documentation says.
    pub fn apply_calibration_ext(&mut self, edit: CalibrationEdit) {
        if self.session.doc.is_none() {
            return;
        }
        if self.tracking.is_some() || self.refine.is_some() {
            self.log
                .error("The project cannot be changed while tracking runs; stop it first.");
            return;
        }
        self.calibration_ext.last_edit = Some(edit.clone());
        let Some(doc) = self.session.doc.as_mut() else {
            return;
        };
        match doc.apply(&Edit::Calibration(edit)) {
            // Unlike other edits, `NotImplemented` is shown in the error dialog too.
            Err(e @ EditError::NotImplemented(_)) => {
                self.log.error(e.to_string());
                self.dialogs.error(e.to_string());
            }
            res => {
                let ok = res.is_ok();
                self.edit_result(res);
                if ok {
                    // The calibration images' projected points depend on the cameras.
                    self.frames.views.calibration_points.clear();
                }
            }
        }
    }

    /// C43: the file dialog, then the file's values into `camera`'s frame.
    fn load_mayacam2(&mut self, camera: usize) {
        let start = self.last_dir();
        let Some(path) =
            self.file_chooser
                .open_file(MAYACAM_TITLE, start.as_deref(), ("Text file", &["txt"]))
        else {
            return;
        };
        match xma_engine::calibration::read_mayacam2(&path) {
            Ok(cal) => {
                let Some(doc) = self.session.doc.as_ref() else {
                    return;
                };
                let frame = self.frames.views.external.entry(camera).or_default();
                // Filled from the camera first, so that the next draw keeps the file's values.
                frame.update(doc, camera);
                frame.load(&cal);
            }
            Err(text) => {
                self.log.error(text.clone());
                self.dialogs.error(text);
            }
        }
    }
}
