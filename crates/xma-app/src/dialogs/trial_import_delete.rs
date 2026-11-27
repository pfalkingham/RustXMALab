//! `TrialImportDeleteDialog`: the trials to import from another project file (File > Import >
//! Trial from xma-file, E72) or to delete (File > Delete multiple trials, E65), and the
//! confirmation that follows.
//!
//! Ported from `TrialImportDeleteDialog.ui` and ui/TrialImportDeleteDialog.cpp: a list with
//! multiple selection (a click toggles a row) and one button, "Import" or "Delete"; the
//! title is "Select the trials to import" or "Select the trials to delete". The form has no
//! Cancel button: the window's close button and Escape reject it. The selection is returned
//! in list order.
//!
//! Then, from ui/MainWindow.cpp:
//!
//! - Delete (`on_actionDelete_multiple_trials_triggered`): "Are you sure you want to delete
//!   the following trials?" with one name per line; OK deletes.
//! - Import (`on_actionImportTrial_triggered`): when a selected trial is a Default trial and
//!   the project has one, "You are about to replace your current Default trial. Are you sure
//!   you want to update it?". XMALab asks while importing and, on Cancel, abandons the trials
//!   after it (D20); here it is asked first, and Cancel imports the other trials without the
//!   Default one (D20 fixed).

use std::collections::BTreeSet;
use std::path::PathBuf;

use super::edit_forms::FormOutcome;
use super::{DialogResult, form};
use crate::icons::{Icon, Icons};

/// The question before deleting.
pub const DELETE_CONFIRM: &str = "Are you sure you want to delete the following trials?\n";
/// The question before replacing the Default trial.
pub const REPLACE_DEFAULT: &str =
    "You are about to replace your current Default trial. Are you sure you want to update it?";

/// What the list is for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrialListUse {
    /// Trials of the project file `path`; `default[i]` marks a Default trial;
    /// `project_has_default` says whether the open project has one.
    Import {
        path: PathBuf,
        default: Vec<bool>,
        project_has_default: bool,
    },
    /// Trials of the open project.
    Delete,
}

/// The dialog's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrialImportDeleteDialog {
    pub purpose: TrialListUse,
    pub names: Vec<String>,
    /// Selected rows.
    pub selected: BTreeSet<usize>,
    /// The confirmation is on screen for these rows.
    pub confirming: Option<Vec<usize>>,
}

impl TrialImportDeleteDialog {
    #[must_use]
    pub fn new(purpose: TrialListUse, names: Vec<String>) -> Self {
        Self {
            purpose,
            names,
            selected: BTreeSet::new(),
            confirming: None,
        }
    }

    /// The window title.
    #[must_use]
    pub fn title(&self) -> &'static str {
        match self.purpose {
            TrialListUse::Import { .. } => "Select the trials to import",
            TrialListUse::Delete => "Select the trials to delete",
        }
    }

    /// The button's text.
    #[must_use]
    pub fn button(&self) -> &'static str {
        match self.purpose {
            TrialListUse::Import { .. } => "Import",
            TrialListUse::Delete => "Delete",
        }
    }

    /// The confirmation text for `rows`, if one is asked.
    #[must_use]
    pub fn question(&self, rows: &[usize]) -> Option<String> {
        match &self.purpose {
            TrialListUse::Delete => {
                let mut t = DELETE_CONFIRM.to_owned();
                for &r in rows {
                    t.push_str(&self.names[r]);
                    t.push('\n');
                }
                Some(t)
            }
            TrialListUse::Import {
                default,
                project_has_default,
                ..
            } => (*project_has_default && rows.iter().any(|&r| default[r]))
                .then(|| REPLACE_DEFAULT.to_owned()),
        }
    }

    /// The answer for `rows`; `yes` is the confirmation's answer (true when none is asked).
    fn answer(&self, rows: &[usize], yes: bool) -> Option<DialogResult> {
        match &self.purpose {
            TrialListUse::Delete => yes.then(|| DialogResult::TrialsToDelete(rows.to_vec())),
            TrialListUse::Import { path, .. } => Some(DialogResult::TrialsToImport {
                path: path.clone(),
                trials: rows.iter().map(|&r| self.names[r].clone()).collect(),
                replace_default: yes,
            }),
        }
    }

    /// Draws the list, or the confirmation after the button.
    pub fn show(&mut self, ctx: &egui::Context, icons: &Icons) -> FormOutcome {
        let mut out = FormOutcome::default();
        if let Some(rows) = self.confirming.clone() {
            let text = self.question(&rows).unwrap_or_default();
            let mut yes = None;
            let modal = form::modal(ctx, egui::Id::unique("xma_trial_list_confirm"), |ui| {
                ui.set_min_width(360.0);
                if form::dialog_title(ui, "Confirmation Required") {
                    yes = Some(false);
                }
                form::message_row(ui, icons, Icon::Alert, &text);
                if let Some(y) = form::reject_accept(ui, "OK", "Cancel") {
                    yes = Some(y);
                }
            });
            if yes.is_none() && modal.escaped() {
                yes = Some(false);
            }
            if let Some(y) = yes {
                out.result = self.answer(&rows, y);
                out.close = true;
            }
            return out;
        }
        let mut toggled = None;
        let mut pressed = false;
        let modal = form::modal(ctx, egui::Id::unique("xma_trial_list"), |ui| {
            ui.set_width(240.0);
            if form::dialog_title(ui, self.title()) {
                out.close = true;
            }
            egui::Frame::group(ui.style()).show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("trial_import_delete_list")
                    .min_scrolled_height(200.0)
                    .max_height(200.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (i, n) in self.names.iter().enumerate() {
                            if ui.selectable_label(self.selected.contains(&i), n).clicked() {
                                toggled = Some(i);
                            }
                        }
                    });
            });
            if form::wide_button(ui, true, self.button()).clicked() {
                pressed = true;
            }
        });
        if let Some(i) = toggled
            && !self.selected.remove(&i)
        {
            self.selected.insert(i);
        }
        if pressed {
            let rows: Vec<usize> = self.selected.iter().copied().collect();
            if rows.is_empty() {
                // `diag->result() == Accepted && !selected.empty()`: nothing to do.
                out.close = true;
            } else if self.question(&rows).is_some() {
                self.confirming = Some(rows);
            } else {
                out.result = self.answer(&rows, true);
                out.close = true;
            }
        } else if modal.escaped() {
            out.close = true;
        }
        out
    }
}
