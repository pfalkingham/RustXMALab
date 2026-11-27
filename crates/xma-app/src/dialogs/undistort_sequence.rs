//! `UndistortSequenceDialog` (File > Undistort sequence, C64): the queue of image folders and
//! videos to undistort with a camera's grid.
//!
//! Ported from `UndistortSequenceDialog.ui` and ui/UndistortSequenceDialog.cpp, control by
//! control: "Camera: " and its combo box (the 1-based numbers of the cameras whose grid is
//! computed), "Add Image Folder" (the folder, then the output folder), "Add Video File" (the
//! file, `Video Files (*.cine *.avi *.mp4 *.mov)`, then the output folder), the list of
//! queued items ("Camera <n> from <input> to <output>") and "Undistort". The three buttons are
//! enabled when the combo box has a camera. The input and output questions start in their own
//! last folders, both first the last used folder of the settings (which they do not change).
//!
//! Differences (ledger U121, U123): "Undistort" closes the dialog and the queue runs as a job
//! with a Progress entry and Cancel ([`crate::app`]'s `start_undistort`); XMALab processes it
//! on the interface thread with no progress and keeps the dialog open, removing each item from
//! the list when it is done. A file of another type typed into the video question is queued
//! and reported when processed; XMALab drops it and leaves its list out of step with its
//! queue. "Undistort" with an empty queue does nothing, as in XMALab.

use std::path::{Path, PathBuf};

use xma_engine::undistort_sequence::{
    UndistortInput, UndistortItem, VIDEO_TYPES, undistort_cameras,
};

use super::edit_forms::FormOutcome;
use super::{DialogEnv, DialogResult, form};
use crate::saving::FileChooser;

/// The window title.
pub const TITLE: &str = "Undistort Sequence";
/// The folder question of "Add Image Folder".
pub const FOLDER_TITLE: &str = "Image Directory ";
/// The output folder question.
pub const OUT_TITLE: &str = "Out Directory ";
/// The file question of "Add Video File".
pub const VIDEO_TITLE: &str = "Open video stream movie file";
/// Its filter.
pub const VIDEO_FILES: (&str, &[&str]) = ("Video Files", &VIDEO_TYPES);

/// The dialog's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UndistortSequenceDialog {
    /// The cameras in the combo box (0-based; shown 1-based).
    pub cameras: Vec<usize>,
    /// The combo box's current row.
    pub selected: usize,
    /// The queue (`items`), as the list shows it.
    pub items: Vec<UndistortItem>,
    /// `lastInputDir`.
    pub last_input: Option<PathBuf>,
    /// `lastOutputDir`.
    pub last_output: Option<PathBuf>,
}

impl UndistortSequenceDialog {
    /// Ported from ui/UndistortSequenceDialog.cpp, the constructor.
    #[must_use]
    pub fn new(doc: &xma_engine::Document, last_used: Option<PathBuf>) -> Self {
        Self {
            cameras: undistort_cameras(doc),
            selected: 0,
            items: Vec::new(),
            last_input: last_used.clone(),
            last_output: last_used,
        }
    }

    /// The buttons are enabled (`comboBox->count() > 0`).
    #[must_use]
    pub fn enabled(&self) -> bool {
        !self.cameras.is_empty()
    }

    fn camera(&self) -> Option<usize> {
        self.cameras.get(self.selected).copied()
    }

    /// Asks the output folder and queues `input`.
    fn queue(&mut self, files: &mut FileChooser, input: UndistortInput) {
        let Some(camera) = self.camera() else { return };
        let Some(out) = files.pick_folder(OUT_TITLE, self.last_output.as_deref()) else {
            return;
        };
        if out.as_os_str().is_empty() {
            return;
        }
        self.last_output = Some(out.clone());
        self.items.push(UndistortItem {
            camera,
            input,
            output: out,
        });
    }

    /// "Add Image Folder".
    ///
    /// Ported from ui/UndistortSequenceDialog.cpp, on_pushButtonAddFolder_clicked.
    pub fn add_folder(&mut self, files: &mut FileChooser) {
        let Some(folder) = files.pick_folder(FOLDER_TITLE, self.last_input.as_deref()) else {
            return;
        };
        if folder.as_os_str().is_empty() {
            return;
        }
        self.last_input = Some(folder.clone());
        self.queue(files, UndistortInput::Folder(folder));
    }

    /// "Add Video File".
    ///
    /// Ported from ui/UndistortSequenceDialog.cpp, on_pushButtonAddVideo_clicked.
    pub fn add_video(&mut self, files: &mut FileChooser) {
        let Some(file) = files.open_file(VIDEO_TITLE, self.last_input.as_deref(), VIDEO_FILES)
        else {
            return;
        };
        if file.as_os_str().is_empty() {
            return;
        }
        // `QFileInfo(videofile).absolutePath()`.
        self.last_input = std::path::absolute(&file)
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf));
        self.queue(files, UndistortInput::Video(file));
    }

    /// "Undistort": the queue, when it holds anything.
    ///
    /// Ported from ui/UndistortSequenceDialog.cpp, on_pushButtonUndist_clicked (the work is
    /// the application's job).
    pub fn undistort(&mut self) -> Option<DialogResult> {
        (!self.items.is_empty())
            .then(|| DialogResult::UndistortSequence(std::mem::take(&mut self.items)))
    }

    /// Draws the form.
    pub fn show(&mut self, ctx: &egui::Context, env: &mut DialogEnv<'_>) -> FormOutcome {
        enum Act {
            Folder,
            Video,
            Undistort,
        }
        let mut out = FormOutcome::default();
        let mut act = None;
        let enabled = self.enabled();
        let modal = form::modal(ctx, egui::Id::unique("xma_undistort_sequence"), |ui| {
            ui.set_width(900.0);
            if form::dialog_title(ui, TITLE) {
                out.close = true;
            }
            // Row 0: spacer, "Camera: ", comboBox, "Add Image Folder", "Add Video File", spacer.
            ui.horizontal(|ui| {
                let w = 4.0 * 130.0 + 60.0;
                ui.add_space(((ui.available_width() - w) / 2.0).max(0.0));
                let label = ui.label("Camera: ");
                let names: Vec<String> = self.cameras.iter().map(|c| (c + 1).to_string()).collect();
                let names: Vec<&str> = names.iter().map(String::as_str).collect();
                if let Some(i) = form::labelled_combo(
                    ui,
                    true,
                    label.id,
                    "undistort_sequence_camera",
                    self.selected,
                    &names,
                    Some(60.0),
                ) {
                    self.selected = i;
                }
                if form::button(ui, enabled, "Add Image Folder", 130.0).clicked() {
                    act = Some(Act::Folder);
                }
                if form::button(ui, enabled, "Add Video File", 130.0).clicked() {
                    act = Some(Act::Video);
                }
            });
            // Row 1: listWidget.
            egui::Frame::group(ui.style()).show(ui, |ui| {
                egui::ScrollArea::both()
                    .id_salt("undistort_sequence_list")
                    .min_scrolled_height(400.0)
                    .max_height(400.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for it in &self.items {
                            ui.add(egui::Label::new(it.list_text()).extend());
                        }
                    });
            });
            // Row 2: pushButtonUndist in the last column.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if form::button(ui, enabled, "Undistort", 130.0).clicked() {
                    act = Some(Act::Undistort);
                }
            });
        });
        match act {
            Some(Act::Folder) => self.add_folder(env.files),
            Some(Act::Video) => self.add_video(env.files),
            Some(Act::Undistort) => {
                if let Some(r) = self.undistort() {
                    out.result = Some(r);
                    out.close = true;
                }
            }
            None => {
                if modal.escaped() {
                    out.close = true;
                }
            }
        }
        out
    }
}
