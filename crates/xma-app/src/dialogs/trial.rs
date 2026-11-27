//! The Trial dialog (modal): a trial's recording speed, filter cutoff, 3D interpolation and
//! reference calibration, "Force Update Trial" and "Delete Trial".
//!
//! Ported control by control from `TrialDialog.ui` and ui/TrialDialog.cpp, opened by the
//! workspace bar's trial settings button
//! (`WorkspaceNavigationFrame::on_toolButtonTrialSettings_clicked`). Grid of the form, top to
//! bottom:
//!
//! | Row | Control | Text |
//! |---|---|---|
//! | 0 | `groupBox` | "General Parameters": `label` "Reference Calibration :" and `comboBoxReferenceCalibration`; `label_2` "Recording Speed (fps/Hz) : " and `doubleSpinBoxRecSpeedFPS` (0 to 10000, 2 decimals) |
//! | 1 | `groupBox_2` | "Filter parameters": `label_5` "Cutoff Frequency (Hz) :" and `doubleSpinBoxCutoffFrq` (0 to 10000, 2 decimals); `checkBoxInterpolate` "interpolate in 3D" |
//! | 2 | `pushButton_DeleteTrial` | "Delete Trial" |
//! | 3 | `pushButton_ChangeTrialData` | "Change Trial Data": the New trial dialog on this trial (C61, [`super::new_trial`]) |
//! | 4 | `pushButton_Update` | "Force Update Trial" |
//! | 5 | `pushButton_Denoise` | "Denoise Images": the dialog closes, then the output folder and `DenoiseDialog` (C65, [`super::images`]) |
//! | 6 | `pushButton_Cancel`, `pushButton_OK` | "Cancel", "OK" |
//!
//! The dialog writes nothing itself. OK returns the values the user changed
//! (`Edit::SetTrialProperties`, E57 to E60), "Force Update Trial" the same with the full
//! update (E61), "Delete Trial" asks "Are you sure you really want to delete the trial?"
//! first (E62); Cancel, the close button and Escape change nothing.
//!
//! Two defects of XMALab 3 are fixed (`docs/cpp-editing.md` section 15):
//!
//! - D13: XMALab 3 stores a new reference calibration as soon as the dialog opens when the
//!   stored one is not calibrated in every camera (the first listed, or -1 with none) and
//!   starts a full update, whatever button then closes the dialog. Here opening writes
//!   nothing: a stored reference that is not listed is shown as such ("<n> (not calibrated
//!   in every camera)"), and the reference changes only when the user picks one and presses
//!   OK or "Force Update Trial".
//! - D14: the recording speed and the cutoff are shown with XMALab's two decimals, but a
//!   value is written only when the user changed it, and then as typed, so that a frame
//!   rate such as 29.97002997 is not rounded to 29.97 by pressing OK.

use egui::Ui;
use xma_engine::Document;
use xma_engine::edit::TrialProperties;
use xma_model::CalibrationType;

use super::form;

/// The confirmation "Delete Trial" asks (`TrialDialog::on_pushButton_DeleteTrial_clicked`).
pub const DELETE_TEXT: &str = "Are you sure you really want to delete the trial?";

/// The dialog's state: the values shown and what the user changed.
#[derive(Clone, Debug, PartialEq)]
pub struct TrialDialog {
    /// The trial, by index.
    pub trial: usize,
    /// Its name, for the title "Trial : <name>".
    pub name: String,
    /// The calibration images (0-based) of the first camera that are calibrated in every
    /// camera: what `comboBoxReferenceCalibration` lists, 1-based.
    pub references: Vec<usize>,
    /// The stored reference calibration image (0-based).
    pub stored_reference: usize,
    /// The reference the user picked, if any.
    pub reference: Option<usize>,
    pub stored_speed: f64,
    /// The recording speed the user entered, if any.
    pub speed: Option<f64>,
    pub stored_cutoff: f64,
    /// The cutoff the user entered, if any.
    pub cutoff: Option<f64>,
    pub stored_interpolate: bool,
    /// The check box.
    pub interpolate: bool,
    /// The dataset has no calibration (`NO_CALIBRATION`): the reference, "interpolate in 3D"
    /// and the cutoff are disabled.
    pub no_calibration: bool,
}

/// What the dialog asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrialRequest {
    /// Cancel, the close button or Escape.
    Close,
    /// OK.
    Ok,
    /// "Force Update Trial".
    ForceUpdate,
    /// "Delete Trial" (the confirmation follows).
    Delete,
    /// "Change Trial Data" (the New trial dialog on this trial follows).
    ChangeData,
    /// "Denoise Images" (the folder question and `DenoiseDialog` follow).
    Denoise,
}

impl TrialDialog {
    /// The dialog for `trial` of `doc`, as `TrialDialog::TrialDialog` fills it, without the
    /// write of D13. `None` for a trial that does not exist.
    ///
    /// Ported from ui/TrialDialog.cpp, TrialDialog::TrialDialog (`:47-101`).
    #[must_use]
    pub fn new(doc: &Document, trial: usize) -> Option<Self> {
        let p = doc.project();
        let t = p.trials.get(trial)?;
        let n_images = p.cameras.first().map_or(0, |c| c.calibration_images.len());
        let references = (0..n_images)
            .filter(|&i| {
                p.cameras.iter().all(|c| {
                    c.calibration_images
                        .get(i)
                        .is_some_and(xma_model::CalibrationImage::is_calibrated)
                })
            })
            .collect();
        Some(Self {
            trial,
            name: t.name.clone(),
            references,
            stored_reference: t.reference_calibration_image,
            reference: None,
            stored_speed: t.recording_speed,
            speed: None,
            stored_cutoff: t.cutoff_frequency,
            cutoff: None,
            stored_interpolate: t.interpolate_3d,
            interpolate: t.interpolate_3d,
            no_calibration: p.calibration_type == CalibrationType::None,
        })
    }

    /// The window title (`setWindowTitle("Trial : " + name)`).
    #[must_use]
    pub fn title(&self) -> String {
        format!("Trial : {}", self.name)
    }

    /// The reference shown: the one picked, else the stored one.
    #[must_use]
    pub fn shown_reference(&self) -> usize {
        self.reference.unwrap_or(self.stored_reference)
    }

    /// The text of the reference combo box: the image 1-based, marked when it is not
    /// calibrated in every camera (D13); empty for a dataset without calibration.
    #[must_use]
    pub fn reference_text(&self) -> String {
        if self.no_calibration {
            return String::new();
        }
        let r = self.shown_reference();
        if r == xma_model::trial::NO_REFERENCE_CALIBRATION {
            // XMALab's -1 (ledger F68).
            return "-1 (not calibrated in every camera)".to_owned();
        }
        if self.references.contains(&r) {
            (r + 1).to_string()
        } else {
            format!("{} (not calibrated in every camera)", r + 1)
        }
    }

    /// What OK (or, with `force_update`, "Force Update Trial") writes: only what the user
    /// changed (D13, D14). Ported from ui/TrialDialog.cpp, TrialDialog::isComplete
    /// (`:111-138`) and on_pushButton_Update_clicked.
    #[must_use]
    pub fn properties(&self, force_update: bool) -> TrialProperties {
        let calibrated = !self.no_calibration;
        TrialProperties {
            recording_speed: self.speed.filter(|v| *v != self.stored_speed),
            cutoff_frequency: self
                .cutoff
                .filter(|v| calibrated && *v != self.stored_cutoff),
            interpolate_3d: Some(self.interpolate)
                .filter(|v| calibrated && *v != self.stored_interpolate),
            reference_calibration: self
                .reference
                .filter(|r| calibrated && *r != self.stored_reference),
            force_update,
        }
    }

    /// Draws the dialog.
    pub fn ui(&mut self, ui: &mut Ui) -> Option<TrialRequest> {
        let mut req = None;
        ui.set_width(389.0);
        if form::dialog_title(ui, &self.title()) {
            req = Some(TrialRequest::Close);
        }
        let enabled = !self.no_calibration;
        let field_w = 100.0;
        form::group_box(ui, "General Parameters", |ui| {
            egui::Grid::new("trial_general")
                .num_columns(2)
                .show(ui, |ui| {
                    let l = ui.label("Reference Calibration :").id;
                    ui.add_enabled_ui(enabled, |ui| {
                        let mut chosen = None;
                        let shown = self.shown_reference();
                        let r = egui::ComboBox::from_id_salt("comboBoxReferenceCalibration")
                            .selected_text(self.reference_text())
                            .show_ui(ui, |ui| {
                                for &i in &self.references {
                                    if ui
                                        .selectable_label(i == shown, (i + 1).to_string())
                                        .clicked()
                                    {
                                        chosen = Some(i);
                                    }
                                }
                            });
                        let chosen = chosen.or_else(|| {
                            let at = self.references.iter().position(|&i| i == shown);
                            crate::qt::combo_wheel(
                                ui,
                                &r.response,
                                at,
                                self.references.len(),
                                |_| true,
                            )
                            .and_then(|k| self.references.get(k).copied())
                        });
                        let _ = r.response.labelled_by(l);
                        if let Some(i) = chosen {
                            self.reference = Some(i);
                        }
                    });
                    ui.end_row();
                    ui.label("Recording Speed (fps/Hz) : ");
                    if let Some(v) = form::double_spin_box(
                        ui,
                        true,
                        self.speed.unwrap_or(self.stored_speed),
                        0.0..=10000.0,
                        2,
                        field_w,
                        "Recording speed",
                    ) {
                        self.speed = Some(v);
                    }
                    ui.end_row();
                });
        });
        form::group_box(ui, "Filter parameters", |ui| {
            egui::Grid::new("trial_filter")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label("Cutoff Frequency (Hz) :");
                    if let Some(v) = form::double_spin_box(
                        ui,
                        enabled,
                        self.cutoff.unwrap_or(self.stored_cutoff),
                        0.0..=10000.0,
                        2,
                        field_w,
                        "Cutoff frequency",
                    ) {
                        self.cutoff = Some(v);
                    }
                    ui.end_row();
                });
            if let Some(v) = form::check_box(ui, enabled, self.interpolate, "interpolate in 3D") {
                self.interpolate = v;
            }
        });
        if form::wide_button(ui, true, "Delete Trial").clicked() {
            req = Some(TrialRequest::Delete);
        }
        if form::wide_button(ui, true, "Change Trial Data").clicked() {
            req = Some(TrialRequest::ChangeData);
        }
        if form::wide_button(ui, true, "Force Update Trial").clicked() {
            req = Some(TrialRequest::ForceUpdate);
        }
        if form::wide_button(ui, true, "Denoise Images").clicked() {
            req = Some(TrialRequest::Denoise);
        }
        ui.add_space(4.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if form::button(ui, true, "OK", 80.0).clicked() {
                req = Some(TrialRequest::Ok);
            }
            if form::button(ui, true, "Cancel", 80.0).clicked() {
                req = Some(TrialRequest::Close);
            }
        });
        req
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dialog() -> TrialDialog {
        TrialDialog {
            trial: 0,
            name: "T".into(),
            references: vec![0, 2],
            stored_reference: 1,
            reference: None,
            stored_speed: 29.970_029_97,
            speed: None,
            stored_cutoff: 10.0,
            cutoff: None,
            stored_interpolate: false,
            interpolate: false,
            no_calibration: false,
        }
    }

    #[test]
    fn nothing_changed_writes_nothing() {
        let d = dialog();
        assert_eq!(d.properties(false), TrialProperties::default());
        assert!(d.properties(true).force_update);
        assert_eq!(d.reference_text(), "2 (not calibrated in every camera)");
    }

    #[test]
    fn only_what_the_user_changed_is_written_as_typed() {
        let mut d = dialog();
        d.speed = Some(29.970_029_97);
        d.cutoff = Some(12.345);
        d.reference = Some(2);
        d.interpolate = true;
        let p = d.properties(false);
        assert_eq!(p.recording_speed, None, "the stored value again");
        assert_eq!(p.cutoff_frequency, Some(12.345));
        assert_eq!(p.reference_calibration, Some(2));
        assert_eq!(p.interpolate_3d, Some(true));
        assert_eq!(d.reference_text(), "3");
    }

    #[test]
    fn without_calibration_only_the_speed_is_written() {
        let mut d = dialog();
        d.no_calibration = true;
        d.speed = Some(100.0);
        d.cutoff = Some(5.0);
        d.reference = Some(0);
        d.interpolate = true;
        let p = d.properties(false);
        assert_eq!(p.recording_speed, Some(100.0));
        assert_eq!(p.cutoff_frequency, None);
        assert_eq!(p.reference_calibration, None);
        assert_eq!(p.interpolate_3d, None);
        assert_eq!(d.reference_text(), "");
    }
}
