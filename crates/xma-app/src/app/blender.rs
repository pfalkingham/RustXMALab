//! File > Export > Blender scene (new in RustXMALab; parity ledger U176 for the entry and
//! the dialog, F85 for the export): the dialog (`dialogs/blender_scene.rs`), the folder
//! question, and the run.
//!
//! OK in the dialog asks for the folder to write into (`FileChooser::pick_folder`,
//! starting at and remembering the last used folder, as Undistorted Trial images does), then
//! `xma_engine::blender_scene::BlenderSceneJob` is made on the interface thread (it reads no
//! mesh file: those are read and checked on the worker thread; a refusal,
//! such as an output folder that exists and is not empty, goes to the error dialog and the
//! Console, and nothing is written) and runs on a worker thread as the cancelable job
//! "Export Blender scene" in the Progress panel, with the other exports of `app/images.rs`:
//! the window behaves as during Undistorted Trial images. The project is not changed (no
//! undo step). When the run ends, the Console says what was written and where, how many
//! rigid bodies, meshes and 3D points the scene holds, what was done other than asked (the
//! report's notes), what was left out and why, and whether Blender made the `.blend` (if not,
//! the command that makes it), as `xma export ... blender-scene` prints them; a failure goes
//! to the error dialog too.

use std::path::{Path, PathBuf};

use xma_engine::blender_scene::{
    BlenderSceneJob, BlenderSceneKind, BlenderSceneReport, BlenderSceneSpec, locate_blender,
};

use super::XmaApp;
use super::images::ExportReport;
use crate::dialogs::Dialog;
use crate::dialogs::blender_scene::{BlenderSceneDialog, FOLDER_TITLE};
use crate::jobs::JobHandle;
use crate::log::LogLevel;
use crate::state::Workspace;

/// The job's title in the Progress panel, and the Console lines' prefix.
pub const TITLE: &str = "Export Blender scene";

/// The command that makes the `.blend` from the folder by hand, the folder and the name
/// quoted for the system's shell (`xma_engine::blender_scene::by_hand_command` says how, and
/// what `cmd` on Windows cannot take).
#[must_use]
pub fn by_hand_command(folder: &Path) -> String {
    let name = folder
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    format!(
        "run: {}",
        xma_engine::blender_scene::by_hand_command(folder, &format!("{name}.blend"), false)
    )
}

impl XmaApp {
    /// File > Export > Blender scene: opens the dialog (Blender is searched for here, once).
    pub(super) fn open_blender_scene(&mut self) {
        let Some(doc) = self.session.doc.as_ref() else {
            return;
        };
        let blender = locate_blender(self.blender_search.as_deref());
        let d = BlenderSceneDialog::new(
            doc,
            self.state.workspace == Workspace::MarkerTracking,
            self.state.active_trial,
            blender,
        );
        self.dialogs.open(Dialog::BlenderScene(Box::new(d)));
    }

    /// OK in the dialog: the folder question, then the job.
    pub fn start_blender_scene(&mut self, spec: BlenderSceneSpec) {
        if self.session.doc.is_none() {
            return;
        }
        let start = self.last_dir();
        let Some(dir) = self
            .file_chooser
            .pick_folder(FOLDER_TITLE, start.as_deref())
        else {
            return;
        };
        let abs = std::path::absolute(&dir).unwrap_or_else(|_| dir.clone());
        self.settings.last_directory_used = abs.to_string_lossy().into_owned();
        self.settings_dirty = true;
        let files: Vec<Vec<PathBuf>> = match spec.kind {
            BlenderSceneKind::Trial {
                trial,
                frames: Some(_),
            } => self.trial_files(trial).cameras,
            _ => Vec::new(),
        };
        let Some(doc) = self.session.doc.as_ref() else {
            return;
        };
        let job = match BlenderSceneJob::new(doc, spec, &files, &dir) {
            Ok(j) => j,
            Err(e) => {
                let msg = format!("{TITLE}: {e}");
                self.log.error(msg.clone());
                self.dialogs.error(msg);
                return;
            }
        };
        self.start_export_run(
            TITLE,
            Box::new(move |h: &JobHandle| {
                ExportReport::Blender(job.run(
                    &mut |p| {
                        if p.blender {
                            h.set_progress(1.0);
                            h.set_message(format!("{TITLE}: Blender is making the .blend"));
                            return;
                        }
                        let done = if p.frames == 0 {
                            0.0
                        } else {
                            p.frame as f32 / p.frames as f32
                        };
                        h.set_progress((p.camera as f32 + done) / p.cameras.max(1) as f32);
                        h.set_message(format!(
                            "{TITLE}: camera {} of {}, image {} of {}",
                            p.camera + 1,
                            p.cameras,
                            (p.frame + 1).min(p.frames),
                            p.frames
                        ));
                    },
                    &|| h.is_canceled(),
                ))
            }),
        );
    }

    /// The run has ended: the Console lines, and the error dialog for a failure.
    pub(super) fn blender_scene_done(&mut self, r: &BlenderSceneReport) {
        if let Some(e) = &r.error {
            let msg = format!("{TITLE}: {e}; nothing was written.");
            self.log.error(msg.clone());
            self.dialogs.error(msg);
            return;
        }
        let Some(folder) = &r.folder else {
            if r.cancelled {
                self.log
                    .info(format!("{TITLE}: cancelled; nothing was written."));
            }
            return;
        };
        let images = r.written.iter().filter(|p| p.starts_with("images")).count();
        let meshes = r.written.iter().filter(|p| p.starts_with("meshes")).count();
        let parts = if meshes > 0 {
            format!(
                "README.txt, {images} images and {meshes} {}",
                if meshes == 1 { "mesh" } else { "meshes" }
            )
        } else {
            format!("README.txt and {images} images")
        };
        self.log.write(
            format!(
                "{TITLE}: wrote {} ({} files: build_scene.py, scene.json, {parts}, in {:.1} s)",
                folder.display(),
                r.written.len() - usize::from(r.blend.is_some()),
                r.seconds_folder
            ),
            LogLevel::Output,
        );
        if r.bodies + r.points > 0 {
            self.log.write(
                format!(
                    "{TITLE}: {} rigid bodies ({} with a mesh), {} 3D points",
                    r.bodies, r.meshes, r.points
                ),
                LogLevel::Output,
            );
        }
        for n in &r.notes {
            self.log.info(format!("{TITLE}: {n}"));
        }
        for w in &r.left_out {
            self.log.info(format!("{TITLE}: left out: {w}"));
        }
        match (&r.blend, &r.blender) {
            (Some(b), _) => self.log.write(
                format!(
                    "{TITLE}: Blender made {} ({:.1} s)",
                    b.display(),
                    r.seconds_blender
                ),
                LogLevel::Output,
            ),
            (None, Some(why)) => self.log.error(format!("{TITLE}: {why}")),
            (None, None) => self.log.info(format!(
                "{TITLE}: no .blend was made (not asked for); to make it, {}",
                by_hand_command(folder)
            )),
        }
        self.problems_box(TITLE, &r.problems);
    }
}
