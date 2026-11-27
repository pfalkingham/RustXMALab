//! The File menu's imports and trial deletion, and the Rigid body dialog's opening: what
//! `MainWindow` does around the edits (file dialogs, the dialogs that choose, what follows),
//! and keeping the per-trial interface state with its trial when trials are imported,
//! deleted, or brought back by undo.
//!
//! Ported from ui/MainWindow.cpp: `on_actionImport2D_Points_triggered` (`:2110-2144`, E66),
//! `on_actionImportTrial_triggered` (`:2146-2200`, E72),
//! `on_actionDelete_multiple_trials_triggered` (`:2202-2243`, E65); and
//! ui/MarkerTreeWidgetButton.cpp, `settingsButtonClicked` (the Rigid body dialog).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use xma_engine::{Edit, OpenOptions};

use super::XmaApp;
use super::document::TrialMove;
use crate::dialogs::Dialog;
use crate::dialogs::point_import_export::{
    ImportExportKind, PointImportExportDialog, import_options,
};
use crate::dialogs::rigid_body::RigidBodyDialog;
use crate::dialogs::trial_import_delete::{TrialImportDeleteDialog, TrialListUse};
use crate::editing::EditRequest;
use crate::jobs::JobId;
use crate::session::{MissingVideo, check_trial_image_paths};
use crate::state::{TrialView, Workspace};

/// File > Import > 2D Points between the file dialog and the import: the Progress panel
/// shows "Importing points" (`ProgressDialog::showProgressbar(0, 0, "Importing points")`)
/// for at least one frame before the import runs, as XMALab shows it during the import.
#[derive(Debug)]
pub struct PendingImport {
    job: JobId,
    request: EditRequest,
    /// The Progress panel has been drawn with the job.
    shown: bool,
}

impl XmaApp {
    /// The settings button of a rigid-body row: the Rigid body dialog for body `body` of the
    /// active trial.
    pub(super) fn open_rigid_body_dialog(&mut self, body: usize) {
        let (Some(doc), Some(t)) = (self.session.doc.as_ref(), self.state.active_trial) else {
            return;
        };
        if doc
            .project()
            .trials
            .get(t)
            .is_none_or(|tr| body >= tr.rigid_bodies.len())
        {
            return;
        }
        let frame = self.state.trial_view().map_or(0, |v| v.active_frame);
        self.dialogs
            .open(Dialog::RigidBody(Box::new(RigidBodyDialog::new(
                doc, t, body, frame,
            ))));
    }

    /// File > Import > 2D Points: the options first.
    pub(super) fn import_2d_points(&mut self) {
        self.dialogs
            .open(Dialog::PointImportExport(PointImportExportDialog::new(
                ImportExportKind::Import2d,
            )));
    }

    /// OK in `PointImportExportDialog`. For the import: "Open csv files for 2D points", then
    /// the files are imported into the active trial in one undo step. (The exports open the
    /// same form; they are wired by the export package.)
    pub(super) fn point_import_export_accepted(&mut self, kind: ImportExportKind) {
        if kind != ImportExportKind::Import2d {
            return;
        }
        let Some(trial) = self.state.active_trial else {
            return;
        };
        let start = self.last_dir();
        let files = self.file_chooser.open_files(
            "Open csv files for 2D points",
            start.as_deref(),
            ("Comma seperated data", &["csv"]),
        );
        let Some(first) = files.first() else {
            return;
        };
        self.settings.set_last_used_directory_from_file(first);
        self.settings_dirty = true;
        let options = import_options(&self.settings.import_export);
        let job = self.jobs.start("Importing points", false).id();
        self.import_points = Some(PendingImport {
            job,
            request: EditRequest::Apply(Edit::Import2dPoints {
                trial,
                files,
                options,
            }),
            shown: false,
        });
    }

    /// Runs the import once the Progress panel has shown its entry; then the entry closes
    /// (`closeProgressbar`).
    pub(super) fn step_import_points(&mut self) {
        let Some(p) = self.import_points.as_mut() else {
            return;
        };
        if !p.shown {
            p.shown = true;
            return;
        }
        let Some(p) = self.import_points.take() else {
            return;
        };
        self.apply_request(p.request);
        self.jobs.finish(p.job);
    }

    /// File > Import > Trial from xma-file: "Select dataset", then the trials to import.
    pub(super) fn import_trial(&mut self) {
        let start = self.last_dir();
        let Some(path) = self.file_chooser.open_file(
            "Select dataset",
            start.as_deref(),
            ("Dataset", &["xma", "zip"]),
        ) else {
            return;
        };
        self.settings.set_last_used_directory_from_file(&path);
        self.settings_dirty = true;
        // ProjectFileIO::readTrials: the trial names of the file.
        let loaded = match xma_engine::read_project(&path, &OpenOptions::default()) {
            Ok(l) => l,
            Err(e) => {
                let msg = format!("Could not open {}: {e}", path.display());
                self.log.error(msg.clone());
                self.dialogs.error(msg);
                return;
            }
        };
        let names = loaded
            .project
            .trials
            .iter()
            .map(|t| t.name.clone())
            .collect();
        let default = loaded.project.trials.iter().map(|t| t.is_default).collect();
        let project_has_default = self
            .session
            .doc
            .as_ref()
            .is_some_and(|d| d.project().trials.iter().any(|t| t.is_default));
        self.dialogs
            .open(Dialog::TrialImportDelete(TrialImportDeleteDialog::new(
                TrialListUse::Import {
                    path,
                    default,
                    project_has_default,
                },
                names,
            )));
    }

    /// The chosen trials of `path` imported (one undo step), then: in Marker tracking the
    /// workspace is entered again, which updates the active trial. The image paths of the
    /// new trials are checked when the trial list changes ([`XmaApp::follow_trial_list`]).
    pub(super) fn import_trials(
        &mut self,
        path: &Path,
        trials: Vec<String>,
        replace_default: bool,
    ) {
        if self.session.doc.is_none() || trials.is_empty() {
            return;
        }
        self.apply_request(EditRequest::Apply(Edit::ImportTrials {
            path: path.to_path_buf(),
            trials,
            replace_default,
        }));
        self.sync_document();
        if self.state.workspace == Workspace::MarkerTracking {
            self.changes |= self.state.change_workspace(Workspace::MarkerTracking, true);
        }
    }

    /// File > Delete multiple trials: the trials of the project to choose from.
    pub(super) fn delete_multiple_trials(&mut self) {
        let names = self.facts.trial_names.clone();
        self.dialogs
            .open(Dialog::TrialImportDelete(TrialImportDeleteDialog::new(
                TrialListUse::Delete,
                names,
            )));
    }

    /// The chosen trials deleted (one undo step), confirmed. The active trial stays the same
    /// trial when it is kept, as the C++ trial box keeps it; with no trial left the workspace
    /// is entered again (Marker tracking, as the C++ does).
    pub(super) fn delete_trials(&mut self, trials: &[usize]) {
        if self.session.doc.is_none() || trials.is_empty() {
            return;
        }
        self.apply_request(EditRequest::Apply(Edit::DeleteTrials {
            trials: trials.to_vec(),
        }));
        self.sync_document();
        if self.facts.trial_names.is_empty() {
            self.changes |= self.state.change_workspace(Workspace::MarkerTracking, true);
        }
    }

    pub(super) fn last_dir(&self) -> Option<PathBuf> {
        (!self.settings.last_directory_used.is_empty())
            .then(|| PathBuf::from(&self.settings.last_directory_used))
    }

    /// When the document's trials are no longer the ones the interface knows (an import, a
    /// deletion, their undo or redo), the per-trial state follows each trial by name: its
    /// view state (frame, active marker or body), where its videos are (kept by name for
    /// trials removed, so that undo finds them again), the queued requests for missing
    /// videos, and the active trial. Trials new to the session get their image paths checked
    /// (`MainWindow::checkTrialImagePaths`, as after an import).
    pub(super) fn follow_trial_list(&mut self) -> Option<TrialMove> {
        let doc = self.session.doc.as_ref()?;
        let new: Vec<String> = doc
            .project()
            .trials
            .iter()
            .map(|t| t.name.clone())
            .collect();
        let old = self.facts.trial_names.clone();
        if new == old {
            return None;
        }
        // Old index -> new index, matching names in order (a name used twice matches in
        // turn).
        let mut used = vec![false; new.len()];
        let map: Vec<Option<usize>> = old
            .iter()
            .map(|n| {
                let i = (0..new.len()).find(|&i| !used[i] && &new[i] == n)?;
                used[i] = true;
                Some(i)
            })
            .collect();
        let mut from: Vec<Option<usize>> = vec![None; new.len()];
        for (o, m) in map.iter().enumerate() {
            if let Some(n) = m {
                from[*n] = Some(o);
            }
        }

        // View state and the ranges seen.
        let trials = &doc.project().trials;
        let ranges = super::trial_ranges(doc);
        let mut views = Vec::with_capacity(new.len());
        let mut seen = Vec::with_capacity(new.len());
        for (i, t) in trials.iter().enumerate() {
            match from[i].and_then(|o| self.state.trials.get(o).copied().map(|v| (o, v))) {
                Some((o, v)) => {
                    views.push(v);
                    seen.push(self.seen_trials.get(o).copied().unwrap_or(ranges[i]));
                }
                None => {
                    views.push(TrialView::from_trial(t));
                    seen.push(ranges[i]);
                }
            }
        }
        self.state.trials = views;
        self.seen_trials = seen;

        // Videos: by name.
        for ((t, c), loc) in std::mem::take(&mut self.session.videos) {
            if let Some(n) = old.get(t) {
                self.session.video_stash.insert((n.clone(), c), loc);
            }
        }
        let mut videos = BTreeMap::new();
        let mut unknown = Vec::new();
        for (i, n) in new.iter().enumerate() {
            let n_cams = trials[i].videos.len();
            let mut found = false;
            for c in 0..n_cams {
                if let Some(loc) = self.session.video_stash.remove(&(n.clone(), c)) {
                    videos.insert((i, c), loc);
                    found = true;
                }
            }
            if !found && from[i].is_none() && !trials[i].is_default {
                unknown.push(i);
            }
        }
        let mut missing: Vec<MissingVideo> = Vec::new();
        if !unknown.is_empty() && !self.settings.disable_image_search {
            let project_file = doc
                .path()
                .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
            let mut found = BTreeMap::new();
            let mut all_missing = Vec::new();
            let mut messages = Vec::new();
            check_trial_image_paths(
                doc,
                &project_file,
                &mut found,
                &mut all_missing,
                &mut messages,
            );
            for ((t, c), loc) in found {
                if unknown.contains(&t) {
                    videos.insert((t, c), loc);
                }
            }
            missing = all_missing
                .into_iter()
                .filter(|m| unknown.contains(&m.trial))
                .collect();
            for m in messages {
                self.log.write(m, crate::log::LogLevel::Output);
            }
        }
        self.session.videos = videos;
        self.dialogs
            .remap_missing(|t| map.get(t).copied().flatten());
        for m in missing {
            self.dialogs.open(Dialog::MissingVideo(m));
        }

        // Where the active trial went: the same trial when kept; otherwise (deleted, or no
        // trial was active and there is one now) the caller takes the one the document made
        // active.
        match self.state.active_trial {
            Some(o) => match map.get(o).copied().flatten() {
                Some(i) if i == o => None,
                Some(i) => Some(TrialMove::Moved(i)),
                None => Some(TrialMove::Gone),
            },
            None if !new.is_empty() => Some(TrialMove::Gone),
            None => None,
        }
    }
}
