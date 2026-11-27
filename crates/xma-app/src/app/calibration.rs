//! The Undistortion and Calibration workspaces' actions (`Action::Calib`), the
//! pending-changes question (C11, C39), and keeping the window's workspace, active camera
//! and calibration frame in step with the engine's session state (`xma_engine::Session`).
//!
//! Ported from ui/WizardUndistortionFrame.cpp, ui/WizardCalibrationCubeFrame.cpp,
//! ui/WizardDockWidget.cpp (`checkForPendingChanges`) and the `UNDISTORTION` and
//! `CALIBRATION` branches of ui/GLCameraView.cpp; the rules and texts are in
//! `crate::calibration_ui`, which also lists the differences from XMALab 3.
//!
//! - **Session state.** The reference clicks, manual mode and its point, the undistortion
//!   state and the pending-changes flags live in the document's session
//!   (`Document::session`), not in the project (undo and redo put back the undistortion
//!   state and the flags). The window's workspace, active camera and calibration frame
//!   ([`AppState`]) and the Settings dialog's four calibration settings go into the session
//!   at the end of every frame and before every calibration action and edit, through the
//!   engine's functions ([`XmaApp::state_to_session`]); the window takes the session's
//!   values when a project is shown and after every calibration edit
//!   ([`XmaApp::session_to_state`]), and the undistortion state after every document
//!   change, undo and redo included.
//! - **`setDialog`** is the engine's (`Document::set_active_calibration_image`,
//!   `set_workspace`, `set_manual_mode`, `calibration_dialog` and the calibration edits). A
//!   reference index set by hand is kept by the window until the project is closed and put
//!   back after each `setDialog` (the C++ loses it).
//! - **Outcomes.** `EditOutcome::messages` of a calibration edit go to the Console and,
//!   except for "optimize camera setup", to XMALab's error dialog; `EditOutcome::refused`
//!   and `EditError::Rejected` are the error dialog.
//!
//! [`AppState`]: crate::state::AppState

use xma_engine::calibration::CalibrationEdit;
use xma_engine::session::{WizardSettings, Workspace as EngineWorkspace};
use xma_engine::{Edit, EditOutcome};

use super::XmaApp;
use crate::actions::Action;
use crate::calibration_ui::{self as cui, CalibAction};
use crate::dialogs::{Dialog, IntPurpose};
use crate::editing::EditRequest;
use crate::log::LogLevel;
use crate::state::{Changes, UndistortionMouseMode, UndistortionState, Workspace};

fn to_engine(w: Workspace) -> EngineWorkspace {
    match w {
        Workspace::Undistortion => EngineWorkspace::Undistortion,
        Workspace::Calibration => EngineWorkspace::Calibration,
        Workspace::MarkerTracking => EngineWorkspace::Tracking,
    }
}

fn from_engine(w: EngineWorkspace) -> Workspace {
    match w {
        EngineWorkspace::Undistortion => Workspace::Undistortion,
        EngineWorkspace::Calibration => Workspace::Calibration,
        EngineWorkspace::Tracking => Workspace::MarkerTracking,
    }
}

impl XmaApp {
    // -----------------------------------------------------------------------------------
    // Session state.

    /// The four settings of the Settings dialog that the calibration actions read, as the
    /// engine's session holds them (`Session::wizard`).
    fn wizard_settings(&self) -> WizardSettings {
        let s = &self.settings;
        WizardSettings {
            auto_confirm_pending_changes: s.auto_confirm_pending_changes,
            auto_calib_after_reference: s.auto_calib_after_reference,
            show_advanced_calibration: s.show_advanced_calibration,
            hide_warnings_during_calibration: s.hide_warnings_during_calibration,
        }
    }

    /// The engine's session follows the window: the Settings dialog's four calibration
    /// settings are copied into `Session::wizard`; the active camera and calibration frame
    /// go through `Document::set_active_calibration_image` and the workspace through
    /// `Document::set_workspace` (Marker tracking is set directly: the window updates the
    /// active trial itself), so that the engine runs the calibration page's `setDialog`
    /// where XMALab runs it. The reference indices set by hand are put back afterwards.
    /// Run at the end of every frame, before every calibration action and edit. The
    /// undistortion state is the session's (the window only shows it).
    pub fn state_to_session(&mut self) {
        let wizard = self.wizard_settings();
        let ws = to_engine(self.state.workspace);
        let (cam, frame) = (
            self.state.active_camera,
            self.state.active_frame_calibration,
        );
        let Some(doc) = self.session.doc.as_mut() else {
            return;
        };
        doc.session_mut().wizard = wizard;
        let s = doc.session();
        let c = cam.unwrap_or(s.active_camera);
        let f = frame.unwrap_or(s.active_frame_calibration);
        // The camera and frame first, so that a workspace change in the same frame runs
        // `setDialog` on the image now active.
        doc.set_active_calibration_image(c, f);
        if doc.session().workspace != ws {
            if ws == EngineWorkspace::Tracking {
                doc.session_mut().workspace = ws;
            } else {
                doc.set_workspace(ws, false);
            }
        }
        self.restore_custom_references();
        self.follow_camera_data();
    }

    /// Takes the workspace, undistortion state, active camera and calibration frame from
    /// the document's session (after a project is shown, after a calibration edit).
    /// `force_workspace` runs the workspace handlers even when it did not change.
    pub(super) fn session_to_state(&mut self, force_workspace: bool) -> Changes {
        let Some(doc) = self.session.doc.as_ref() else {
            return Changes::NONE;
        };
        let s = doc.session().clone();
        let n_cameras = doc.project().cameras.len();
        let n_images = doc
            .project()
            .cameras
            .first()
            .map_or(0, |c| c.calibration_images.len());
        let mut ch = Changes::NONE;
        let camera = (n_cameras > 0).then(|| s.active_camera.min(n_cameras - 1));
        ch |= self.state.change_active_camera(camera, false);
        let frame = Some(s.active_frame_calibration.min(n_images.saturating_sub(1)));
        ch |= self.state.change_active_frame_calibration(frame, false);
        let u = if s.undistorted {
            UndistortionState::Undistorted
        } else {
            UndistortionState::NotUndistorted
        };
        ch |= self.state.change_undistortion(u, false);
        ch |= self
            .state
            .change_workspace(from_engine(s.workspace), force_workspace);
        ch
    }

    /// After an edit, an undo or a redo: the undistortion state is the session's (undo and
    /// redo put it back with the project), and the calibration frame is kept within the
    /// frames there are (an undone "Delete Frame" or a deleted one).
    pub(super) fn follow_session_computed(&mut self) -> Changes {
        let Some(doc) = self.session.doc.as_ref() else {
            return Changes::NONE;
        };
        let u = if doc.session().undistorted {
            UndistortionState::Undistorted
        } else {
            UndistortionState::NotUndistorted
        };
        let n_images = doc
            .project()
            .cameras
            .first()
            .map_or(0, |c| c.calibration_images.len());
        let mut ch = self.state.change_undistortion(u, false);
        if let Some(f) = self.state.active_frame_calibration
            && n_images > 0
            && f >= n_images
        {
            ch |= self
                .state
                .change_active_frame_calibration(Some(n_images - 1), false);
        }
        ch
    }

    /// The calibration page when a project is shown: the session takes the window's
    /// settings, then `setDialog` (`Document::calibration_dialog`) loads the reference
    /// indices of the reference file.
    pub(super) fn calibration_page_loaded(&mut self) {
        self.state_to_session();
        if let Some(doc) = self.session.doc.as_mut() {
            doc.calibration_dialog();
        }
        self.follow_camera_data();
    }

    /// A reference index set by hand (C14) is kept by the window until the project is
    /// closed: the engine's `setDialog` (a camera, frame or workspace change, manual mode,
    /// every calibration) reloads the reference file's indices, as XMALab 3 does, and the
    /// window puts the custom ones back after it.
    pub(super) fn restore_custom_references(&mut self) {
        let custom = self.custom_references;
        let Some(doc) = self.session.doc.as_mut() else {
            return;
        };
        for (slot, id) in custom.iter().enumerate() {
            let Some(id) = *id else { continue };
            let r = doc.session().references[slot];
            if !(r.custom && r.point_id == id) {
                doc.session_mut().set_reference_index(slot, id);
            }
        }
    }

    /// `setDialog` in manual mode can give the active image its point lists outside an
    /// edit: the views' cached calibration points are made again when a camera's data
    /// changed.
    fn follow_camera_data(&mut self) {
        let Some(doc) = self.session.doc.as_ref() else {
            return;
        };
        let ids = super::document::camera_ids(doc);
        if ids != self.seen_cameras {
            self.seen_cameras = ids;
            self.frames.calibration_changed();
        }
    }

    /// After a calibration edit was applied (at once or as a long edit): the derived images
    /// and cached points are made again, the window follows the session the edit left (the
    /// engine ran `setDialog` where XMALab runs it), and the custom reference indices are
    /// put back.
    pub(super) fn calibration_edited(&mut self) {
        self.frames.calibration_changed();
        let ch = self.session_to_state(false);
        self.changes |= ch;
        self.restore_custom_references();
    }

    /// The outcome of a calibration edit that went through: `EditOutcome::messages` (the
    /// frames that could not be calibrated, "Could not set Camera ...") go to the Console
    /// and, except for "optimize camera setup" whose message is a Console line in XMALab and
    /// the lines saying frames were renumbered (ledger A189),
    /// to one error dialog, as XMALab shows them; `EditOutcome::refused` is the error
    /// dialog.
    pub(super) fn calibration_outcome(&mut self, edit: Option<&Edit>, outcome: EditOutcome) {
        let dialog = !matches!(
            edit,
            Some(Edit::Calibration(CalibrationEdit::Optimize { .. })) | None
        );
        for m in &outcome.messages {
            self.log.write(m.clone(), LogLevel::Output);
        }
        // Ledger A189: the lines saying that the frames cannot show whether two cameras
        // number the board alike are the Console's only; the warning that cameras number it
        // differently is shown.
        let shown: Vec<&String> = outcome
            .messages
            .iter()
            .filter(|m| !xma_engine::calibration::numbering::is_console_only(m))
            .collect();
        if dialog && !shown.is_empty() {
            self.dialogs.error(
                shown
                    .iter()
                    .map(|m| m.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        if let Some(text) = outcome.refused {
            self.dialogs.error(text);
        }
    }

    // -----------------------------------------------------------------------------------
    // Calibration edit requests.

    /// An `Edit::Calibration` request: a long one on a worker thread
    /// ([`XmaApp::start_long_edit`]), an immediate one at once, followed by
    /// [`XmaApp::calibration_edited`] when it succeeds. Returns false for other edits.
    pub(super) fn apply_calibration_request(&mut self, edit: &Edit) -> bool {
        let Edit::Calibration(e) = edit else {
            return false;
        };
        let board = self.session.doc.as_ref().is_some_and(cui::is_checkerboard);
        if let Some(text) = cui::progress_text_for(e, self.state.workspace, board) {
            self.start_long_edit(edit.clone(), text);
            return true;
        }
        self.state_to_session();
        let Some(doc) = self.session.doc.as_mut() else {
            return true;
        };
        match doc.apply(edit) {
            Ok(outcome) => {
                self.calibration_outcome(Some(edit), outcome);
                self.calibration_edited();
            }
            Err(e) => self.report_edit_error(&e),
        }
        true
    }

    fn request_calibration(&mut self, e: CalibrationEdit) {
        self.edits.push(EditRequest::Apply(Edit::Calibration(e)));
    }

    // -----------------------------------------------------------------------------------
    // The pending-changes question (C11, C39).

    /// Asks the pending-changes question before `a` (a workspace change, Save, Save As)
    /// while the Undistortion workspace shows a grid with changes not recomputed (C11) or
    /// the Calibration workspace a camera (C39); returns true when it asked or started the
    /// recomputation (the caller stops; `a` follows the recomputation).
    ///
    /// Ported from ui/WizardDockWidget.cpp, checkForPendingChanges, and its callers
    /// (`WorkspaceNavigationFrame::on_comboBoxWorkspace_currentIndexChanged`,
    /// `MainWindow::on_actionSave_Project_triggered`, `on_actionSave_Project_as_triggered`).
    pub(super) fn pending_guard(&mut self, a: &Action, confirmed: bool) -> bool {
        if confirmed {
            return false;
        }
        if let Action::SetWorkspace(w) = a
            && *w == self.state.workspace
        {
            return false;
        }
        self.state_to_session();
        let Some(doc) = self.session.doc.as_ref() else {
            return false;
        };
        let s = doc.session();
        let auto = s.wizard.auto_confirm_pending_changes;
        let text = match self.state.workspace {
            Workspace::Undistortion if s.pending_undistortion() => cui::PENDING_UNDISTORTION,
            Workspace::Calibration if s.pending_calibration() => cui::PENDING_CALIBRATION,
            _ => return false,
        };
        let after_save = if matches!(a, Action::SaveProject | Action::SaveProjectAs) {
            self.after_save.take().map(Box::new)
        } else {
            None
        };
        let then = CalibAction::RecomputeThen {
            then: Box::new(a.clone()),
            after_save,
        };
        if auto {
            self.dispatch_calib(&then, true);
        } else {
            self.confirm(text.to_owned(), Action::Calib(then));
        }
        true
    }

    // -----------------------------------------------------------------------------------
    // The actions.

    /// Carries out a [`CalibAction`] (`confirmed`: its question was answered Yes).
    pub(super) fn dispatch_calib(&mut self, a: &CalibAction, confirmed: bool) {
        if self.session.doc.is_none() {
            return;
        }
        self.state_to_session();
        let camera = self.state.active_camera.unwrap_or(0);
        let image = self.state.active_frame_calibration.unwrap_or(0);
        let camera_name = self
            .facts
            .camera_names
            .get(camera)
            .cloned()
            .unwrap_or_default();
        match a {
            CalibAction::MouseMode(m) => {
                // C5: "set center" once undistorted resets the inliers: asked first. No
                // keeps the previous mode (the radio buttons show the state).
                if *m == UndistortionMouseMode::SetCenter
                    && self.state.undistortion == UndistortionState::Undistorted
                    && !confirmed
                {
                    self.confirm(cui::SET_CENTER_WARNING.into(), Action::Calib(a.clone()));
                    return;
                }
                self.changes |= self.state.change_undistortion_mouse_mode(*m, false);
            }
            CalibAction::UndistortionClick { camera, x, y } => {
                self.undistortion_click(*camera, *x, *y);
            }
            CalibAction::RemoveOutliers => {
                let (title, label, value, min, max) = cui::BORDER_PROMPT;
                self.dialogs.open(Dialog::GetInt {
                    title: title.into(),
                    label: label.into(),
                    value,
                    min,
                    max,
                    purpose: IntPurpose::GridBorder,
                });
            }
            CalibAction::ChooseReference(slot) => {
                if let Some(doc) = self.session.doc.as_mut() {
                    doc.session_mut().next_reference = (*slot).min(3);
                }
            }
            CalibAction::ReferenceIndex(slot) => {
                let Some(doc) = self.session.doc.as_ref() else {
                    return;
                };
                let n = cui::object_points(doc);
                let Some(r) = doc.session().references.get(*slot) else {
                    return;
                };
                if n == 0 {
                    return;
                }
                let n = i32::try_from(n).unwrap_or(i32::MAX);
                let value = i32::try_from(r.point_id + 1).unwrap_or(1).clamp(1, n);
                self.dialogs.open(Dialog::GetInt {
                    title: cui::reference_index_title(*slot),
                    label: "Index".into(),
                    value,
                    min: 1,
                    max: n,
                    purpose: IntPurpose::ReferenceIndex(*slot),
                });
            }
            CalibAction::CalibrationClick {
                camera,
                x,
                y,
                ctrl,
                alt,
            } => self.calibration_click(*camera, *x, *y, *ctrl, *alt),
            CalibAction::DoubleClick { camera, x, y } => {
                // C23: a calibrated image only, in the coordinates the view shows.
                let Some(doc) = self.session.doc.as_ref() else {
                    return;
                };
                if cui::image_calibrated(doc, *camera, image) == 1 {
                    let undistorted = cui::shows_undistorted(doc, &self.state, *camera);
                    self.request_calibration(CalibrationEdit::ToggleInlier {
                        camera: *camera,
                        image,
                        x: *x,
                        y: *y,
                        undistorted,
                    });
                }
            }
            CalibAction::ManualMode(on) => {
                // on_checkBoxManual_clicked: reloadManualPoints and setDialog.
                if let Some(doc) = self.session.doc.as_mut() {
                    doc.set_manual_mode(*on);
                }
                self.restore_custom_references();
                self.follow_camera_data();
            }
            CalibAction::ManualPoint(i) => {
                if let Some(doc) = self.session.doc.as_mut() {
                    doc.session_mut().manual_point = *i;
                }
            }
            CalibAction::ManualInliers(v) => {
                self.request_calibration(CalibrationEdit::SetManualInliers {
                    camera,
                    image,
                    inliers: v.clone(),
                });
            }
            CalibAction::Compute => {
                self.request_calibration(CalibrationEdit::ComputeCalibration { camera, image });
            }
            CalibAction::ResetFrame => {
                if confirmed {
                    self.request_calibration(CalibrationEdit::ResetFrame { camera, image });
                } else {
                    self.confirm(
                        cui::reset_frame_text(image, &camera_name),
                        Action::Calib(a.clone()),
                    );
                }
            }
            CalibAction::ResetCamera => {
                if confirmed {
                    self.request_calibration(CalibrationEdit::ResetCamera { camera });
                } else {
                    self.confirm(
                        cui::reset_camera_text(&camera_name),
                        Action::Calib(a.clone()),
                    );
                }
            }
            CalibAction::DeleteFrame => {
                // Only with more than one calibration frame (the C++ asks nothing else).
                if self.facts.nb_images_calibration <= 1 {
                    return;
                }
                if confirmed {
                    self.request_calibration(CalibrationEdit::DeleteFrame { image });
                } else {
                    self.confirm(cui::delete_frame_text(image), Action::Calib(a.clone()));
                }
            }
            CalibAction::LensModel(on) => {
                self.request_calibration(CalibrationEdit::SetLensModel { camera, on: *on });
            }
            CalibAction::RemoveOptimization => {
                // CD23 fixed: the removal itself, whatever the active image is.
                if confirmed {
                    self.request_calibration(CalibrationEdit::RemoveOptimization { camera, image });
                } else {
                    self.confirm(cui::REMOVE_OPTIMIZATION.into(), Action::Calib(a.clone()));
                }
            }
            CalibAction::RecomputeThen { then, after_save } => {
                let then = (**then).clone();
                let after_save = after_save.as_deref().cloned();
                let follow: super::long_edit::AfterLongEdit = Box::new(move |app: &mut XmaApp| {
                    if after_save.is_some() {
                        app.after_save = after_save;
                    }
                    app.actions.push(Action::Confirmed(Box::new(then)));
                });
                let e = CalibrationEdit::RecomputePending;
                let text = cui::progress_text(&e, self.state.workspace).unwrap_or("Recompute");
                self.start_long_then(Edit::Calibration(e), text, follow);
            }
        }
    }

    /// C6, C7: a left click in the Undistortion workspace.
    ///
    /// Ported from ui/GLCameraView.cpp, GLCameraView::mousePressEvent (`UNDISTORTION`).
    fn undistortion_click(&mut self, camera: usize, x: f64, y: f64) {
        let has_grid = self
            .session
            .doc
            .as_ref()
            .and_then(|d| d.project().cameras.get(camera))
            .is_some_and(|c| c.undistortion.is_some());
        if !has_grid {
            return;
        }
        match self.state.undistortion_mouse_mode {
            UndistortionMouseMode::None => {}
            UndistortionMouseMode::ToggleOutlier => {
                match cui::outlier_list(self.state.undistortion_vis_points) {
                    Some(list) => self.request_calibration(CalibrationEdit::ToggleGridOutlier {
                        camera,
                        list,
                        x,
                        y,
                    }),
                    None => self.dialogs.error(cui::TOGGLE_OUTLIER_ERROR),
                }
            }
            UndistortionMouseMode::SetCenter => {
                if self.state.undistortion_vis_image
                    == crate::state::UndistortionVisImage::Distorted
                {
                    self.request_calibration(CalibrationEdit::SetGridCenter { camera, x, y });
                } else {
                    self.dialogs.error(cui::SET_CENTER_ERROR);
                }
            }
        }
    }

    /// C15, C16, C19, C24, C27: a left click in the Calibration workspace.
    ///
    /// Ported from ui/GLCameraView.cpp, GLCameraView::mousePressEvent (`CALIBRATION`), and
    /// ui/WizardCalibrationCubeFrame.cpp, addCalibrationReference.
    fn calibration_click(&mut self, camera: usize, x: f64, y: f64, ctrl: bool, alt: bool) {
        let image = self.state.active_frame_calibration.unwrap_or(0);
        let Some(doc) = self.session.doc.as_ref() else {
            return;
        };
        if camera >= doc.project().cameras.len() {
            return;
        }
        let undistorted = cui::shows_undistorted(doc, &self.state, camera);
        let (x, y) = if alt {
            // C16 (ledger A130: in the view's coordinates, not rounded).
            doc.snap_calibration_click_in(camera, image, x, y, undistorted)
        } else {
            (x, y)
        };
        let calibrated = cui::image_calibrated(doc, camera, image);
        let checkerboard = cui::is_checkerboard(doc);
        let n_points = cui::object_points(doc);
        let manual = doc.session().manual_mode;
        if manual {
            // C27: the checked point takes the click; the next one is checked.
            let point = doc.session().manual_point;
            if n_points == 0 || point >= n_points {
                return;
            }
            // The engine checks the next point when the click is set.
            self.request_calibration(CalibrationEdit::SetManualPoint {
                camera,
                image,
                point,
                x,
                y,
                undistorted,
            });
        } else if calibrated != 1 {
            // C15, C40, C41: once undistorted, a cube's next reference, or a checkerboard's
            // interior upper-left corner (automatic) or next corner (manual). C19: with
            // AutoCalibAfterReference the fourth reference, or the board's corner, computes.
            if self.state.undistortion != UndistortionState::Undistorted {
                return;
            }
            if checkerboard {
                // addCalibrationReference: the text combo to "id", the points combo to
                // "detected cube points (distorted)".
                let mut ch = crate::panels::toolbox::calibration::choose_points(&mut self.state, 2);
                ch |= self
                    .state
                    .change_calibration_vis_text(crate::state::CalibrationVisText::Id, false);
                self.changes |= ch;
            }
            let Some(doc) = self.session.doc.as_mut() else {
                return;
            };
            if doc.click_calibration_reference(x, y) {
                self.request_calibration(CalibrationEdit::ComputeCalibration { camera, image });
            }
        } else if ctrl {
            // C24.
            self.request_calibration(CalibrationEdit::MovePoint {
                camera,
                image,
                x,
                y,
                undistorted,
            });
        }
    }

    /// The answer to one of the workspaces' integer prompts (C8, C14).
    pub(super) fn calibration_int(&mut self, purpose: IntPurpose, v: i32) {
        match purpose {
            IntPurpose::GridBorder => {
                let (title, label, value, min, max) = cui::CIRCLE_PROMPT;
                self.dialogs.open(Dialog::GetInt {
                    title: title.into(),
                    label: label.into(),
                    value,
                    min,
                    max,
                    purpose: IntPurpose::GridCircle { border: v },
                });
            }
            IntPurpose::GridCircle { border } => {
                self.request_calibration(CalibrationEdit::RemoveGridOutliers { border, circle: v });
            }
            IntPurpose::ReferenceIndex(slot) => {
                // Kept by the window until the project is closed (`restore_custom_references`).
                let id = usize::try_from(v - 1).unwrap_or(0);
                if let Some(c) = self.custom_references.get_mut(slot) {
                    *c = Some(id);
                }
                self.restore_custom_references();
            }
            _ => {}
        }
    }
}
