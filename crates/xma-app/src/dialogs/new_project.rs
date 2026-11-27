//! `NewProjectDialog` and its `CameraBox`es: File > New dataset (C47 to C52).
//!
//! Ported control by control from `NewProjectDialog.ui`, `CameraBox.ui`,
//! ui/NewProjectDialog.cpp and ui/CameraBox.cpp. Title "Create a new dataset". Grid of the
//! form, top to bottom:
//!
//! | Row | Control | Text |
//! |---|---|---|
//! | 0 | `groupBox` "Cameras" | `label_5` "Number of cameras", `toolButtonCameraMinus` "-", `labelNbCameras`, `toolButtonCameraPlus` "+"; then a scroll area with one `CameraBox` per camera ("Camera 1", "Camera 2" to start) |
//! | 1 | `groupBox_Volume` "CalibrationTarget" | `radioButtonCube` "Cube" (checked), `radioButtonCheckerboard` "Checkerboard"; `frameCheckerboard` (hidden while Cube is chosen): "Nb squares horizontal", "Nb squares vertical", "Squaresize"; `frameCube`: "Frame specifications" with its line edit and "...", "Reference points" with its line edit and "..." |
//! | 2 | | `pushButton_Cancel` "Cancel", `pushButton_OK` "Ok" |
//!
//! A `CameraBox` is a group box titled with the camera's name: a frame with "XRay Camera"
//! (checked), "Light Camera" and "Flip images horizontally"; "Calibration image(s)", the
//! count "(n)", the line edit (disabled) and "..."; the "Undistortion grid" check box
//! (checked), its line edit (disabled) and "...".
//!
//! The dialog changes nothing itself. OK runs [`NewProjectSpec::check`] (XMALab's
//! `isComplete`): a failure shows its sentence in the error dialog over the form, which
//! stays open (an invalid reference file also clears the "Reference points" line edit, as
//! the C++ does); otherwise the dialog closes with the specification, and the application
//! creates the dataset on a worker thread.
//!
//! Differences from XMALab 3 (interface defects fixed; proposed ledger rows in the appnew
//! report):
//!
//! - Cancelling the images "..." file dialog keeps the files chosen before. XMALab 3
//!   empties the list but keeps showing the old text, so OK then reports "Missing Images"
//!   for a camera that shows files.
//! - The undistortion grid controls follow one rule: an X-ray camera shows the "Undistortion
//!   grid" check box, and its line edit and "..." while it is checked; a light camera shows
//!   none. XMALab 3, after Light then XRay with the box unchecked, shows the empty line edit
//!   and hides the check box, so the grid can no longer be chosen.
//!
//! "Checkerboard" shows `frameCheckerboard` and hides `frameCube`, "Cube" the reverse
//! (`on_radioButtonCheckerboard_clicked`, `on_radioButtonCube_clicked`); the spin boxes have
//! Qt's default ranges (0 to 99; 0 to 99.99 with 2 decimals). The dialog's state is the plain
//! struct [`NewProjectDialog`] with public fields, so that a caller (the XMA Portal import's
//! `setupBasedOnMissingParameters`) can fill it before it is shown.

use std::path::PathBuf;

use egui::Ui;
use xma_engine::create::{CalibrationTarget, NewCameraSpec, NewProjectSpec};

use super::edit_forms::{FormOutcome, Prompt, PromptAnswer};
use super::file_names::{
    self, CALIBRATION_FILES, FRAMESPEC_FILES, GRID_FILES, REFERENCE_FILES, path_text,
};
use super::{DialogEnv, DialogResult, form, last_dir};

/// The window title.
pub const TITLE: &str = "Create a new dataset";
/// The images "..." file dialog's title.
pub const IMAGES_TITLE: &str = "Open Calibration Images or video";
/// The grid "..." file dialog's title.
pub const GRID_TITLE: &str = "Open undistortion grid";
/// The frame specification "..." file dialog's title.
pub const FRAMESPEC_TITLE: &str = "Open frame specification of calibration object";
/// The reference points "..." file dialog's title.
pub const REFERENCES_TITLE: &str = "Open reference points file of calibration object";
/// `isComplete`'s sentence for a reference file that is not valid; the line edit is
/// cleared when it is shown.
pub const REFERENCES_INVALID: &str =
    "Reference file is invalid. Please ensure that it has 4 entries and that the format is correct";

/// One `CameraBox`.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraBox {
    /// The group box's title ("Camera 1", ...).
    pub name: String,
    /// `imageFileNames`, sorted (`QStringList::sort`).
    pub images: Vec<PathBuf>,
    /// `lineEditImages`: [`file_names::common_prefix`] of the images.
    pub images_text: String,
    /// `labelNbImages`: "(n)".
    pub count_text: String,
    /// "Light Camera" chosen (else "XRay Camera").
    pub light: bool,
    /// The "Undistortion grid" check box.
    pub grid_checked: bool,
    /// `lineEditUndistortionGrid`.
    pub grid: Option<PathBuf>,
    /// "Flip images horizontally".
    pub flipped: bool,
}

impl CameraBox {
    /// A new box: X-ray camera, grid box checked, nothing chosen.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            images: Vec::new(),
            images_text: String::new(),
            count_text: String::new(),
            light: false,
            grid_checked: true,
            grid: None,
            flipped: false,
        }
    }

    /// The "Undistortion grid" check box is shown (an X-ray camera).
    #[must_use]
    pub fn grid_check_visible(&self) -> bool {
        !self.light
    }

    /// The grid's line edit and "..." are shown (an X-ray camera with the box checked).
    #[must_use]
    pub fn grid_file_visible(&self) -> bool {
        !self.light && self.grid_checked
    }

    /// `CameraBox::hasUndistortion`: an X-ray camera with the grid box checked.
    #[must_use]
    pub fn has_undistortion(&self) -> bool {
        !self.light && self.grid_checked
    }

    /// The files chosen with "...": sorted, shown as the common prefix and the count.
    /// No files (Cancel) changes nothing.
    ///
    /// Ported from ui/CameraBox.cpp, `CameraBox::on_toolButtonImages_clicked`.
    pub fn set_images(&mut self, mut files: Vec<PathBuf>) {
        file_names::sort_plain(&mut files);
        if files.first().is_none_or(|f| f.as_os_str().is_empty()) {
            return;
        }
        self.images_text = file_names::common_prefix_of(&files);
        self.count_text = format!("({})", files.len());
        self.images = files;
    }

    /// The camera as the engine takes it. A checked grid box with no file gives an empty
    /// path, so that the check can name the missing grid.
    #[must_use]
    pub fn spec(&self) -> NewCameraSpec {
        NewCameraSpec {
            name: self.name.clone(),
            images: self.images.clone(),
            light_camera: self.light,
            grid: self
                .has_undistortion()
                .then(|| self.grid.clone().unwrap_or_default()),
            flipped: self.flipped,
        }
    }
}

/// The dialog's state.
#[derive(Clone, Debug, PartialEq)]
pub struct NewProjectDialog {
    pub cameras: Vec<CameraBox>,
    /// "Checkerboard" chosen (else "Cube").
    pub checkerboard: bool,
    /// `lineEditFrameSpecifications`.
    pub frame_specification: Option<PathBuf>,
    /// `lineEditReferencePoints`.
    pub references: Option<PathBuf>,
    /// `spinBoxHorizontalSquares` (0 to 99).
    pub horizontal_squares: i64,
    /// `spinBoxVerticalSquares` (0 to 99).
    pub vertical_squares: i64,
    /// `doubleSpinBoxSizeSquares` (0 to 99.99, 2 decimals).
    pub square_size: f64,
    /// The error dialog over the form.
    pub prompt: Option<Prompt>,
    /// The XMA Portal project the dialog was filled from (C63, `crate::app::portal`): its
    /// extracted files stay until the dialog and its answer are dropped.
    pub portal: Option<xma_engine::portal::ProjectImport>,
}

impl Default for NewProjectDialog {
    fn default() -> Self {
        Self::new()
    }
}

/// What a control asks for.
enum Act {
    Minus,
    Plus,
    Images(usize),
    Grid(usize),
    FrameSpec,
    References,
    Ok,
    Cancel,
}

impl NewProjectDialog {
    /// Two cameras, "Camera 1" and "Camera 2"; Cube.
    ///
    /// Ported from ui/NewProjectDialog.cpp, `NewProjectDialog::NewProjectDialog`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cameras: vec![CameraBox::new("Camera 1"), CameraBox::new("Camera 2")],
            checkerboard: false,
            frame_specification: None,
            references: None,
            horizontal_squares: 0,
            vertical_squares: 0,
            square_size: 0.0,
            prompt: None,
            portal: None,
        }
    }

    /// `labelNbCameras`.
    #[must_use]
    pub fn count_text(&self) -> String {
        self.cameras.len().to_string()
    }

    /// "+": a box "Camera <n>".
    pub fn plus(&mut self) {
        let n = self.cameras.len() + 1;
        self.cameras.push(CameraBox::new(format!("Camera {n}")));
    }

    /// "-": removes the last box, never the only one.
    pub fn minus(&mut self) {
        if self.cameras.len() > 1 {
            self.cameras.pop();
        }
    }

    /// What OK hands to the engine.
    #[must_use]
    pub fn spec(&self) -> NewProjectSpec {
        let target = if self.checkerboard {
            CalibrationTarget::Checkerboard {
                horizontal_squares: u32::try_from(self.horizontal_squares).unwrap_or(0),
                vertical_squares: u32::try_from(self.vertical_squares).unwrap_or(0),
                square_size: self.square_size,
            }
        } else {
            CalibrationTarget::Cube {
                frame_specification: self.frame_specification.clone().unwrap_or_default(),
                references: self.references.clone().unwrap_or_default(),
            }
        };
        NewProjectSpec {
            cameras: self.cameras.iter().map(CameraBox::spec).collect(),
            target,
        }
    }

    /// OK: the check; on failure the error is shown over the form (and an invalid
    /// reference file is cleared). Returns the specification when it passes.
    ///
    /// Ported from ui/NewProjectDialog.cpp, `on_pushButton_OK_clicked` and `isComplete`.
    pub fn ok(&mut self) -> Option<NewProjectSpec> {
        let spec = self.spec();
        match spec.check() {
            Ok(()) => Some(spec),
            Err(message) => {
                if message == REFERENCES_INVALID {
                    self.references = None;
                }
                self.prompt = Some(Prompt::Error { message });
                None
            }
        }
    }

    /// Draws the dialog (and the error over it).
    pub fn show(&mut self, ctx: &egui::Context, env: &mut DialogEnv<'_>) -> FormOutcome {
        let mut out = FormOutcome::default();
        let prompt_open = self.prompt.is_some();
        let mut act = None;
        let modal = egui::Modal::new(egui::Id::unique("xma_new_project_dialog")).show(ctx, |ui| {
            if prompt_open {
                ui.disable();
            }
            act = self.form(ui);
        });
        if let Some(p) = &mut self.prompt {
            if let Some(PromptAnswer::Accepted(_) | PromptAnswer::Rejected) = p.show(ctx, env.icons)
            {
                self.prompt = None;
            }
            return out;
        }
        let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        if act.is_none() && modal.should_close() && escape {
            out.close = true;
            return out;
        }
        if let Some(a) = act {
            self.act(a, env, &mut out);
        }
        out
    }

    fn act(&mut self, a: Act, env: &mut DialogEnv<'_>, out: &mut FormOutcome) {
        let start = last_dir(env.settings);
        match a {
            Act::Minus => self.minus(),
            Act::Plus => self.plus(),
            Act::Images(i) => {
                let files = env
                    .files
                    .open_files(IMAGES_TITLE, start.as_deref(), CALIBRATION_FILES);
                if let Some(b) = self.cameras.get_mut(i)
                    && !files.is_empty()
                {
                    b.set_images(files);
                    // `setLastUsedDirectory(lineEditImages->text())`: the folder of the
                    // shown prefix.
                    if !b.images.is_empty() && !b.images_text.is_empty() {
                        env.settings
                            .set_last_used_directory_from_file(std::path::Path::new(
                                &b.images_text,
                            ));
                        out.settings_changed = true;
                    }
                }
            }
            Act::Grid(i) => {
                if let Some(f) = env
                    .files
                    .open_file(GRID_TITLE, start.as_deref(), GRID_FILES)
                    && !f.as_os_str().is_empty()
                {
                    env.settings.set_last_used_directory_from_file(&f);
                    out.settings_changed = true;
                    if let Some(b) = self.cameras.get_mut(i) {
                        b.grid = Some(f);
                    }
                }
            }
            Act::FrameSpec => {
                if let Some(f) =
                    env.files
                        .open_file(FRAMESPEC_TITLE, start.as_deref(), FRAMESPEC_FILES)
                {
                    env.settings.set_last_used_directory_from_file(&f);
                    out.settings_changed = true;
                    self.frame_specification = Some(f);
                }
            }
            Act::References => {
                if let Some(f) =
                    env.files
                        .open_file(REFERENCES_TITLE, start.as_deref(), REFERENCE_FILES)
                {
                    env.settings.set_last_used_directory_from_file(&f);
                    out.settings_changed = true;
                    self.references = Some(f);
                }
            }
            Act::Ok => {
                if let Some(spec) = self.ok() {
                    out.result = Some(DialogResult::NewProject {
                        spec,
                        dialog: Box::new(self.clone()),
                    });
                    out.close = true;
                }
            }
            Act::Cancel => out.close = true,
        }
    }

    /// The form, in the `.ui` file's grid order.
    fn form(&mut self, ui: &mut Ui) -> Option<Act> {
        let mut act = None;
        ui.set_width(900.0);
        if form::dialog_title(ui, TITLE) {
            act = Some(Act::Cancel);
        }
        form::group_box(ui, "Cameras", |ui| {
            ui.horizontal(|ui| {
                ui.label("Number of cameras");
                ui.add_space(20.0);
                if ui
                    .add(egui::Button::new("-").min_size(egui::vec2(22.0, 0.0)))
                    .accessible_name("Number of cameras -")
                    .clicked()
                {
                    act = Some(Act::Minus);
                }
                ui.label(self.count_text());
                if ui
                    .add(egui::Button::new("+").min_size(egui::vec2(22.0, 0.0)))
                    .accessible_name("Number of cameras +")
                    .clicked()
                {
                    act = Some(Act::Plus);
                }
            });
            egui::ScrollArea::vertical()
                .id_salt("new_project_cameras")
                .max_height(360.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for (i, b) in self.cameras.iter_mut().enumerate() {
                        ui.push_id(i, |ui| {
                            if let Some(a) = camera_box(ui, b, i) {
                                act = Some(a);
                            }
                        });
                    }
                });
        });
        form::group_box(ui, "CalibrationTarget", |ui| {
            ui.horizontal(|ui| {
                if ui
                    .radio(!self.checkerboard, "Cube")
                    .accessible_name("Cube")
                    .clicked()
                {
                    self.checkerboard = false;
                }
                ui.add_space(40.0);
                if ui
                    .radio(self.checkerboard, "Checkerboard")
                    .accessible_name("Checkerboard")
                    .clicked()
                {
                    self.checkerboard = true;
                }
            });
            if self.checkerboard {
                form::styled_frame(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Nb squares horizontal");
                        if let Some(v) = form::spin_box(
                            ui,
                            true,
                            self.horizontal_squares,
                            0,
                            99,
                            60.0,
                            "Nb squares horizontal",
                        ) {
                            self.horizontal_squares = v;
                        }
                        ui.add_space(30.0);
                        ui.label("Nb squares vertical");
                        if let Some(v) = form::spin_box(
                            ui,
                            true,
                            self.vertical_squares,
                            0,
                            99,
                            60.0,
                            "Nb squares vertical",
                        ) {
                            self.vertical_squares = v;
                        }
                        ui.add_space(30.0);
                        ui.label("Squaresize");
                        if let Some(v) = form::double_spin_box(
                            ui,
                            true,
                            self.square_size,
                            0.0..=99.99,
                            2,
                            70.0,
                            "Squaresize",
                        ) {
                            self.square_size = v;
                        }
                    });
                });
            } else {
                form::styled_frame(ui, |ui| {
                    egui::Grid::new("new_project_cube")
                        .num_columns(3)
                        .show(ui, |ui| {
                            let w = 600.0;
                            if file_row(
                                ui,
                                "Frame specifications",
                                self.frame_specification.as_deref(),
                                w,
                            ) {
                                act = Some(Act::FrameSpec);
                            }
                            ui.end_row();
                            if file_row(ui, "Reference points", self.references.as_deref(), w) {
                                act = Some(Act::References);
                            }
                            ui.end_row();
                        });
                });
            }
        });
        ui.add_space(4.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if form::button(ui, true, "Ok", 80.0).clicked() {
                act = Some(Act::Ok);
            }
            if form::button(ui, true, "Cancel", 80.0).clicked() {
                act = Some(Act::Cancel);
            }
        });
        act
    }
}

/// A disabled `QLineEdit` showing `text`, named `name`.
pub(super) fn line_edit(ui: &mut Ui, text: &str, width: f32, name: &str) {
    let mut t = text;
    let h = ui.spacing().interact_size.y;
    ui.add_enabled_ui(false, |ui| {
        ui.add_sized([width, h], egui::TextEdit::singleline(&mut t))
            .accessible_name(name);
    });
}

/// A "..." tool button named `name`.
pub(super) fn dots_button(ui: &mut Ui, name: &str) -> bool {
    ui.add(egui::Button::new("...").min_size(egui::vec2(26.0, 0.0)))
        .accessible_name(name)
        .clicked()
}

/// A label, its file's line edit and "...": true when "..." is clicked.
fn file_row(ui: &mut Ui, label: &str, file: Option<&std::path::Path>, width: f32) -> bool {
    ui.label(label);
    line_edit(
        ui,
        &file.map(path_text).unwrap_or_default(),
        width.max(200.0),
        &format!("{label} file"),
    );
    dots_button(ui, &format!("{label} ..."))
}

/// One `CameraBox`; returns the "..." clicked.
fn camera_box(ui: &mut Ui, b: &mut CameraBox, i: usize) -> Option<Act> {
    let mut act = None;
    let name = b.name.clone();
    form::group_box(ui, &name, |ui| {
        form::styled_frame(ui, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .radio(!b.light, "XRay Camera")
                    .accessible_name(format!("{name}: XRay Camera"))
                    .clicked()
                {
                    b.light = false;
                }
                ui.add_space(20.0);
                if ui
                    .radio(b.light, "Light Camera")
                    .accessible_name(format!("{name}: Light Camera"))
                    .clicked()
                {
                    b.light = true;
                }
                ui.add_space(20.0);
                ui.checkbox(&mut b.flipped, "Flip images horizontally")
                    .accessible_name(format!("{name}: Flip images horizontally"));
            });
        });
        let edit_w = 600.0;
        egui::Grid::new(("camera_box", i))
            .num_columns(4)
            .min_col_width(30.0)
            .show(ui, |ui| {
                ui.label("Calibration image(s)");
                ui.label(b.count_text.as_str());
                line_edit(
                    ui,
                    &b.images_text,
                    edit_w,
                    &format!("{name}: Calibration image(s)"),
                );
                if dots_button(ui, &format!("{name}: Calibration image(s) ...")) {
                    act = Some(Act::Images(i));
                }
                ui.end_row();
                if b.grid_check_visible() {
                    ui.checkbox(&mut b.grid_checked, "Undistortion grid")
                        .accessible_name(format!("{name}: Undistortion grid"));
                    ui.label("");
                    if b.grid_file_visible() {
                        line_edit(
                            ui,
                            &b.grid.as_deref().map(path_text).unwrap_or_default(),
                            edit_w,
                            &format!("{name}: Undistortion grid file"),
                        );
                        if dots_button(ui, &format!("{name}: Undistortion grid ...")) {
                            act = Some(Act::Grid(i));
                        }
                    }
                    ui.end_row();
                }
            });
    });
    act
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_box_rules() {
        let mut b = CameraBox::new("Camera 1");
        assert!(b.grid_check_visible() && b.grid_file_visible() && b.has_undistortion());
        b.grid_checked = false;
        assert!(b.grid_check_visible() && !b.grid_file_visible());
        b.light = true;
        assert!(!b.grid_check_visible() && !b.grid_file_visible() && !b.has_undistortion());
        // Back to X-ray with the box unchecked: the check box is offered again (XMALab 3
        // hides it and shows the empty line edit).
        b.light = false;
        assert!(b.grid_check_visible() && !b.grid_file_visible());
        assert_eq!(b.spec().grid, None);
        b.grid_checked = true;
        assert_eq!(
            b.spec().grid,
            Some(PathBuf::new()),
            "checked without a file"
        );
    }

    #[test]
    fn images_sorted_and_cancel_keeps_them() {
        let mut b = CameraBox::new("Camera 1");
        b.set_images(vec!["/d/c10.tif".into(), "/d/c09.tif".into()]);
        assert_eq!(
            b.images,
            vec![PathBuf::from("/d/c09.tif"), PathBuf::from("/d/c10.tif")]
        );
        assert_eq!(b.images_text, "/d/c0");
        assert_eq!(b.count_text, "(2)");
        b.set_images(Vec::new());
        assert_eq!(b.images.len(), 2);
        assert_eq!(b.count_text, "(2)");
    }

    #[test]
    fn plus_and_minus() {
        let mut d = NewProjectDialog::new();
        assert_eq!(d.count_text(), "2");
        d.plus();
        assert_eq!(d.cameras[2].name, "Camera 3");
        d.minus();
        d.minus();
        d.minus();
        assert_eq!(d.cameras.len(), 1);
        assert_eq!(d.count_text(), "1");
    }
}
