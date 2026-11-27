//! `ImportExportPointsDialog` ("Import/Export  Markers and Rigid Bodies"), opened by the
//! Points panel's "Import/Export" button (E26; modes E67 to E71).
//!
//! Ported control by control from `ImportExportPointsDialog.ui` and
//! ui/ImportExportPointsDialog.cpp:
//!
//! - `frame` (row 0): the radio buttons "Import CSV", "Import from xma", "Copy From trial"
//!   (shown only when the project has more than one trial) and "Rename"; none is checked
//!   when the dialog opens.
//! - `groupBoxImportCSV` "Import CSV" (row 1, shown for "Import CSV" and "Rename"): the
//!   read-only `lineEditMarkersCSV` and the "..." button, which asks for a file ("Open csv
//!   file", "CSV files (*.csv)", starting in the last used folder, which it updates).
//! - `groupBoxImportXMA` "Import From XMA" (row 2, shown for "Import from xma"):
//!   `lineEditMarkersXMA` and "..." ("Open marker csv file", "Dataset (*.xma  *.zip)").
//! - `groupBoxTrial` "Trial" (row 3, shown for "Copy From trial"): `comboBoxTrial`, every
//!   other trial by name.
//! - Row 4: a spacer, "Cancel", "OK".
//!
//! OK runs the chosen mode (`on_pushButtonOK_clicked`); what follows (the prompts and the
//! edit) is the Points panel's flow (`dialogs::points`).

use std::path::PathBuf;

use egui::Ui;

use super::form;
use crate::settings::UiSettings;

/// The radio buttons, in form order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportMode {
    /// "Import CSV" (E67, E68).
    Csv,
    /// "Import from xma" (E69).
    Xma,
    /// "Copy From trial" (E70).
    Trial,
    /// "Rename" (E71).
    Rename,
}

/// The dialog's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportExportPoints {
    /// The checked radio button; `None` when the dialog opens.
    pub mode: Option<ImportMode>,
    /// `lineEditMarkersCSV`.
    pub csv: String,
    /// `lineEditMarkersXMA`.
    pub xma: String,
    /// `comboBoxTrial`: the other trials, (index in the project, name).
    pub trials: Vec<(usize, String)>,
    /// The chosen entry of `comboBoxTrial`.
    pub trial: usize,
}

/// What the buttons did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImportExportAnswer {
    Ok,
    Cancel,
}

impl ImportExportPoints {
    /// Ported from ui/ImportExportPointsDialog.cpp, the constructor: the trial list holds
    /// every trial but the active one; "Copy From trial" is shown only with more than one
    /// trial.
    #[must_use]
    pub fn new(trial_names: &[String], active: usize) -> Self {
        ImportExportPoints {
            mode: None,
            csv: String::new(),
            xma: String::new(),
            trials: trial_names
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != active)
                .map(|(i, n)| (i, n.clone()))
                .collect(),
            trial: 0,
        }
    }

    /// True when "Copy From trial" is shown (the project has more than one trial).
    #[must_use]
    pub fn trial_shown(&self) -> bool {
        !self.trials.is_empty()
    }

    /// The chosen trial of "Copy From trial", as an index in the project.
    #[must_use]
    pub fn source_trial(&self) -> Option<usize> {
        self.trials.get(self.trial).map(|(i, _)| *i)
    }

    /// Draws the form below the title row; returns the button pressed. `settings` gives
    /// and takes the last used folder.
    pub fn ui(
        &mut self,
        ui: &mut Ui,
        settings: &mut UiSettings,
        files: &mut crate::saving::FileChooser,
        changed: &mut bool,
    ) -> Option<ImportExportAnswer> {
        ui.set_min_width(520.0);
        // frame: the radio buttons (switchGroups shows the group of the checked one).
        let trial_shown = self.trial_shown();
        form::styled_frame(ui, |ui| {
            ui.horizontal(|ui| {
                let mut radio = |ui: &mut Ui, m: ImportMode, t: &str| {
                    if form::radio(ui, true, self.mode == Some(m), t) {
                        self.mode = Some(m);
                    }
                };
                radio(ui, ImportMode::Csv, "Import CSV");
                radio(ui, ImportMode::Xma, "Import from xma");
                if trial_shown {
                    radio(ui, ImportMode::Trial, "Copy From trial");
                }
                radio(ui, ImportMode::Rename, "Rename");
            });
        });
        // groupBoxImportCSV.
        if matches!(self.mode, Some(ImportMode::Csv | ImportMode::Rename)) {
            form::group_box(ui, "Import CSV", |ui| {
                if file_row(ui, &self.csv, "CSV file", "Choose CSV file")
                    && let Some(p) = pick(settings, files, "Open csv file", "CSV files", &["csv"])
                {
                    self.csv = p.to_string_lossy().into_owned();
                    *changed = true;
                }
            });
        }
        // groupBoxImportXMA.
        if self.mode == Some(ImportMode::Xma) {
            form::group_box(ui, "Import From XMA", |ui| {
                if file_row(ui, &self.xma, "Dataset file", "Choose dataset file")
                    && let Some(p) = pick(
                        settings,
                        files,
                        "Open marker csv file",
                        "Dataset",
                        &["xma", "zip"],
                    )
                {
                    self.xma = p.to_string_lossy().into_owned();
                    *changed = true;
                }
            });
        }
        // groupBoxTrial.
        if self.mode == Some(ImportMode::Trial) {
            form::group_box(ui, "Trial", |ui| {
                let shown = self.trials.get(self.trial).map_or("", |(_, n)| n.as_str());
                let mut chosen = None;
                let r = egui::ComboBox::from_id_salt("import_points_trial")
                    .selected_text(shown)
                    .width(ui.available_width() - 8.0)
                    .show_ui(ui, |ui| {
                        for (i, (_, n)) in self.trials.iter().enumerate() {
                            if ui.selectable_label(i == self.trial, n).clicked() {
                                chosen = Some(i);
                            }
                        }
                    });
                let chosen = chosen.or_else(|| {
                    crate::qt::combo_wheel(
                        ui,
                        &r.response,
                        Some(self.trial),
                        self.trials.len(),
                        |_| true,
                    )
                });
                let _ = r.response.accessible_name("Trial");
                if let Some(i) = chosen {
                    self.trial = i;
                }
            });
        }
        ui.add_space(4.0);
        let mut answer = None;
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if form::button(ui, true, "OK", 80.0).clicked() {
                answer = Some(ImportExportAnswer::Ok);
            }
            if form::button(ui, true, "Cancel", 80.0).clicked() {
                answer = Some(ImportExportAnswer::Cancel);
            }
        });
        answer
    }
}

/// A read-only line edit and its "..." tool button; true when the button is clicked.
fn file_row(ui: &mut Ui, text: &str, name: &str, button: &str) -> bool {
    let mut clicked = false;
    ui.horizontal(|ui| {
        let mut shown = text.to_owned();
        let w = (ui.available_width() - 40.0).max(120.0);
        ui.add_enabled(
            false,
            egui::TextEdit::singleline(&mut shown).desired_width(w),
        )
        .accessible_name(name);
        clicked = ui
            .button("...")
            .on_hover_text(button)
            .accessible_name(button)
            .clicked();
    });
    clicked
}

/// `QFileDialog::getOpenFileName` starting in the last used folder, which a choice updates
/// (`Settings::setLastUsedDirectory`).
/// Through the application's `FileChooser` (scripted in tests).
fn pick(
    settings: &mut UiSettings,
    files: &mut crate::saving::FileChooser,
    title: &str,
    filter: &str,
    ext: &[&str],
) -> Option<PathBuf> {
    let start = super::last_dir(settings);
    let p = files.open_file(title, start.as_deref(), (filter, ext))?;
    settings.set_last_used_directory_from_file(&p);
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn other_trials_only() {
        let names = vec!["Default".to_owned(), "a".to_owned(), "b".to_owned()];
        let d = ImportExportPoints::new(&names, 1);
        assert_eq!(
            d.trials,
            vec![(0, "Default".to_owned()), (2, "b".to_owned())]
        );
        assert_eq!(d.source_trial(), Some(0));
        assert!(d.trial_shown());
        assert!(!ImportExportPoints::new(&names[..1], 0).trial_shown());
        assert_eq!(d.mode, None);
    }
}
