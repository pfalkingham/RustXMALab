//! `ExternalCalibrationFrame`: one camera's external calibration, shown in place of the
//! image in the Calibration workspace when the project's calibration is external
//! (`ui/CameraViewWidget.cpp:235-275`).
//!
//! Ported from `ui/ExternalCalibrationFrame.cpp` and `ui/ui-files/ExternalCalibrationFrame.ui`
//! (actions C43 to C46, `docs/cpp-calibration.md` section 6):
//!
//! - every spin box of the form (width and height 0 to 100000; the nine camera-matrix
//!   values, the rotation matrix, the translation and the eight lens coefficients with 12
//!   decimals, range ±999999999, single step 0.01);
//! - "model undistortion" shows and hides the coefficient boxes only (C45);
//! - "Load MayaCam 2.0" raises [`Action::LoadMayaCam2`]: the application asks for the file
//!   and fills the boxes with [`ExternalCalibrationFrame::load`] (C43); nothing is applied;
//! - "APPLY CHANGES" raises `CalibrationEdit::ApplyExternal` with what the boxes hold
//!   (C44); editing a box changes nothing until then (C46);
//! - `update`: the boxes are filled again from the camera whenever the camera's values in
//!   the document differ from those the boxes were last filled from (an apply, an undo, or
//!   any other change of that camera). XMALab fills them when the Calibration workspace is
//!   shown; here values loaded or typed and not applied stay while the camera is unchanged.
//!
//! **CD7 fixed** (ledger, Interface): XMALab rebuilds the calibration from the spin boxes,
//! which hold 12 decimals, so applying without editing rounds every value. Here each box
//! keeps the full value behind it and sends it unchanged unless the user edited that box,
//! in which case the typed value is sent. Leaving a box without changing its text is not an
//! edit.

use egui::Ui;
use xma_engine::calibration::{CalibrationEdit, ExternalCalibration};
use xma_engine::{Document, Edit};

use crate::actions::Action;
use crate::editing::EditRequest;
use crate::panels::Cx;

/// Range of the double spin boxes (`ExternalCalibrationFrame.ui`).
pub const RANGE: f64 = 999_999_999.0;
/// Decimals of the double spin boxes.
pub const DECIMALS: usize = 12;
/// Single step of the double spin boxes.
pub const STEP: f64 = 0.01;
/// Maximum of the width and height spin boxes.
pub const MAX_SIZE: i64 = 100_000;

/// The interface state of one camera's frame: what its boxes hold.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExternalCalibrationFrame {
    pub width: u32,
    pub height: u32,
    /// Row by row.
    pub camera_matrix: [[f64; 3]; 3],
    /// Row by row, world to camera.
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    /// k1, k2, p1, p2, k3, k4, k5, k6: kept while "model undistortion" is unchecked.
    pub distortion: [f64; 8],
    /// "model undistortion".
    pub model_distortion: bool,
    /// The camera's values the boxes were last filled from (`None`: never filled, or the
    /// camera has no calibration image, for which `update` changes nothing).
    source: Option<ExternalCalibration>,
}

/// The values XMALab's `update` reads from `camera`: width, height, K, the rotation matrix
/// of the first calibration image's Rodrigues vector, its translation, the eight
/// coefficients (always, `distortion` is `Some`) and the lens flag (second value). `None`
/// when the camera has no calibration image (`update` returns early).
///
/// Ported from ui/ExternalCalibrationFrame.cpp, ExternalCalibrationFrame::update.
#[must_use]
pub fn camera_values(doc: &Document, camera: usize) -> Option<(ExternalCalibration, bool)> {
    let cam = doc.project().cameras.get(camera)?;
    let image = cam.calibration_images.first()?;
    let k = cam.camera_matrix;
    let r = xma_vision::camera::rodrigues_to_matrix(&image.rotation);
    let rows = |m: &xma_model::Matrix3| {
        [
            [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
            [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
            [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
        ]
    };
    Some((
        ExternalCalibration {
            width: cam.width,
            height: cam.height,
            camera_matrix: rows(&k),
            rotation: rows(&r),
            translation: [
                image.translation[0],
                image.translation[1],
                image.translation[2],
            ],
            distortion: Some(cam.distortion),
        },
        cam.has_model_distortion,
    ))
}

/// Bitwise equality, so that a NaN does not count as a change on every frame.
fn same(a: &ExternalCalibration, b: &ExternalCalibration) -> bool {
    let bits = |c: &ExternalCalibration| {
        let mut v: Vec<u64> = Vec::with_capacity(32);
        v.push(u64::from(c.width));
        v.push(u64::from(c.height));
        v.extend(c.camera_matrix.iter().flatten().map(|x| x.to_bits()));
        v.extend(c.rotation.iter().flatten().map(|x| x.to_bits()));
        v.extend(c.translation.iter().map(|x| x.to_bits()));
        v.push(u64::from(c.distortion.is_some()));
        v.extend(c.distortion.iter().flatten().map(|x| x.to_bits()));
        v
    };
    bits(a) == bits(b)
}

impl ExternalCalibrationFrame {
    /// Fills the boxes from `cal`; the check box from `cal.distortion`, the coefficients
    /// set to 0 when there are none.
    fn fill(&mut self, cal: &ExternalCalibration, model_distortion: bool) {
        self.width = cal.width;
        self.height = cal.height;
        self.camera_matrix = cal.camera_matrix;
        self.rotation = cal.rotation;
        self.translation = cal.translation;
        self.distortion = cal.distortion.unwrap_or([0.0; 8]);
        self.model_distortion = model_distortion;
    }

    /// `update`: refills the boxes when the camera's values in `doc` are not those they were
    /// last filled from. Returns true when it refilled them.
    pub fn update(&mut self, doc: &Document, camera: usize) -> bool {
        let now = camera_values(doc, camera);
        let changed = match (&now, &self.source) {
            (Some((a, _)), Some(b)) => !same(a, b),
            (None, None) => false,
            _ => true,
        };
        if !changed {
            return false;
        }
        if let Some((cal, lens)) = &now {
            self.fill(cal, *lens);
        }
        self.source = now.map(|(c, _)| c);
        true
    }

    /// C43: the values read from a MayaCam 2.0 file into the boxes ("model undistortion"
    /// checked when the file has coefficients; otherwise unchecked and the eight
    /// coefficients 0). Nothing is applied.
    ///
    /// Ported from ui/ExternalCalibrationFrame.cpp, on_pushButton_MayaCam_clicked
    /// (`:206-271`, after the file is read).
    pub fn load(&mut self, cal: &ExternalCalibration) {
        self.fill(cal, cal.distortion.is_some());
    }

    /// What the boxes hold, as `CalibrationEdit::ApplyExternal` takes it: the coefficients
    /// only when "model undistortion" is checked (`resetDistortion` otherwise).
    #[must_use]
    pub fn calibration(&self) -> ExternalCalibration {
        ExternalCalibration {
            width: self.width,
            height: self.height,
            camera_matrix: self.camera_matrix,
            rotation: self.rotation,
            translation: self.translation,
            distortion: self.model_distortion.then_some(self.distortion),
        }
    }

    /// C44, "APPLY CHANGES": the edit for `camera`.
    ///
    /// Ported from ui/ExternalCalibrationFrame.cpp, on_pushButton_Apply_clicked.
    #[must_use]
    pub fn apply_edit(&self, camera: usize) -> Edit {
        Edit::Calibration(CalibrationEdit::ApplyExternal {
            camera,
            calibration: Box::new(self.calibration()),
        })
    }
}

/// The text a box shows for `v` (12 decimals, as the `QDoubleSpinBox` does).
#[must_use]
pub fn box_text(v: f64) -> String {
    format!("{v:.DECIMALS$}")
}

/// The value the box shows for `v`: `v` rounded to 12 decimals (the text read back).
#[must_use]
pub fn shown_value(v: f64) -> f64 {
    box_text(v).parse().unwrap_or(v)
}

/// A `QDoubleSpinBox` of the frame (12 decimals, ±999999999, step 0.01) with its up and
/// down arrows ("<name> up", "<name> down"). `value` is the full value behind the box; it
/// is replaced only by an edit (typing a different number, dragging, an arrow), not by
/// leaving the box with its text unchanged (CD7). Returns true when it changed.
pub fn full_spin_box(ui: &mut Ui, value: &mut f64, width: f32, name: &str) -> bool {
    let before = *value;
    let shown = shown_value(before);
    let h = ui.spacing().interact_size.y;
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        let mut v = before;
        let field = ui
            .add_sized(
                [width, h],
                egui::DragValue::new(&mut v)
                    .range(-RANGE..=RANGE)
                    .clamp_existing_to_range(false)
                    .speed(STEP)
                    .custom_formatter(|x, _| box_text(x)),
            )
            .accessible_name(name);
        // The text read back unchanged is the rounded value: not an edit.
        if field.changed() && v.to_bits() != before.to_bits() && v != shown {
            *value = v;
        }
        if let Some(sign) = crate::qt::spin_arrows(ui, field.id, name) {
            let delta = if sign > 0 { STEP } else { -STEP };
            // QAbstractSpinBox::stepBy: the shown value plus one step, rounded to the
            // decimals and kept in range.
            *value = shown_value((shown + delta).clamp(-RANGE, RANGE));
        }
    });
    value.to_bits() != before.to_bits()
}

/// Width of a 12-decimal box.
const BOX_W: f32 = 124.0;

/// Draws the frame of `camera`, filling the view.
///
/// Ported from `ui/ExternalCalibrationFrame.cpp` (`update`, the two buttons, the check
/// box) and its form.
pub fn show(ui: &mut Ui, cx: &mut Cx<'_>, camera: usize) {
    let Some(doc) = cx.doc else { return };
    let Some(cam) = doc.project().cameras.get(camera) else {
        return;
    };
    let n = cam.name.clone();
    let frame = cx.frames.views.external.entry(camera).or_default();
    frame.update(doc, camera);
    let (load, apply) = egui::ScrollArea::both()
        .id_salt(("external_calibration_scroll", camera))
        .auto_shrink([false, false])
        .show(ui, |ui| form(ui, frame, &n, camera))
        .inner;
    let edit = apply.then(|| frame.apply_edit(camera));
    if load {
        cx.raise(Action::LoadMayaCam2(camera));
    }
    if let Some(edit) = edit {
        cx.request(EditRequest::Apply(edit));
    }
}

/// The form's controls in its layout (`gridLayout_8`); returns ("Load MayaCam 2.0",
/// "APPLY CHANGES") clicked.
fn form(ui: &mut Ui, f: &mut ExternalCalibrationFrame, n: &str, camera: usize) -> (bool, bool) {
    let mut load = false;
    let mut apply = false;
    egui::Grid::new(("external_calibration", camera))
        .num_columns(2)
        .spacing([12.0, 24.0])
        .show(ui, |ui| {
            // Row 0: label1, frame_3 (spinBox_width, spinBox_height).
            ui.label("Camera Resolution (width,height)");
            ui.horizontal(|ui| {
                for (v, what) in [(&mut f.width, "width"), (&mut f.height, "height")] {
                    let mut x = i64::from(*v);
                    if let Some(nv) = crate::dialogs::form::spin_box(
                        ui,
                        true,
                        x,
                        0,
                        MAX_SIZE,
                        80.0,
                        &format!("{n} {what}"),
                    ) {
                        x = nv;
                        *v = u32::try_from(x).unwrap_or(0);
                    }
                }
            });
            ui.end_row();
            // Row 1: label1_2, frame_2 (the camera matrix with fx, fy, px, py).
            ui.label("Camera Matrix ");
            egui::Grid::new(("external_k", camera)).show(ui, |ui| {
                let k = &mut f.camera_matrix;
                ui.label("fx");
                full_spin_box(ui, &mut k[0][0], BOX_W, &format!("{n} K11"));
                full_spin_box(ui, &mut k[0][1], BOX_W, &format!("{n} K12"));
                ui.label("px");
                full_spin_box(ui, &mut k[0][2], BOX_W, &format!("{n} K13"));
                ui.end_row();
                ui.label("");
                full_spin_box(ui, &mut k[1][0], BOX_W, &format!("{n} K21"));
                ui.horizontal(|ui| {
                    ui.label("fy");
                    full_spin_box(ui, &mut k[1][1], BOX_W, &format!("{n} K22"));
                });
                ui.label("py");
                full_spin_box(ui, &mut k[1][2], BOX_W, &format!("{n} K23"));
                ui.end_row();
                ui.label("");
                full_spin_box(ui, &mut k[2][0], BOX_W, &format!("{n} K31"));
                full_spin_box(ui, &mut k[2][1], BOX_W, &format!("{n} K32"));
                ui.label("");
                full_spin_box(ui, &mut k[2][2], BOX_W, &format!("{n} K33"));
                ui.end_row();
            });
            ui.end_row();
            // Row 2: label1_3, frame (R11 … R33).
            ui.label("Rotation Matrix (world to camera)");
            egui::Grid::new(("external_r", camera)).show(ui, |ui| {
                for (i, row) in f.rotation.iter_mut().enumerate() {
                    for (j, v) in row.iter_mut().enumerate() {
                        full_spin_box(ui, v, BOX_W, &format!("{n} R{}{}", i + 1, j + 1));
                    }
                    ui.end_row();
                }
            });
            ui.end_row();
            // Row 3: label1_4, frame_4 (TX, TY, TZ).
            ui.label("Translation Vector (world to camera)");
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                for (i, a) in ["TX", "TY", "TZ"].iter().enumerate() {
                    full_spin_box(ui, &mut f.translation[i], BOX_W, &format!("{n} {a}"));
                }
            });
            ui.end_row();
            // Row 4: checkBox_hasDistortion (C45: shows or hides frame_distortion).
            let mut checked = f.model_distortion;
            if ui
                .checkbox(&mut checked, "model undistortion")
                .accessible_name(format!("{n} model undistortion"))
                .changed()
            {
                f.model_distortion = checked;
            }
            ui.end_row();
            // Row 7: frame_distortion (label1_5, frame_5; label1_6, frame_7).
            if f.model_distortion {
                for (row, (label, names)) in [
                    ("k1,k2,p1,p2", ["k1", "k2", "p1", "p2"]),
                    ("k3,k4,k5,k6", ["k3", "k4", "k5", "k6"]),
                ]
                .into_iter()
                .enumerate()
                {
                    ui.label(label);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        for (i, a) in names.iter().enumerate() {
                            full_spin_box(
                                ui,
                                &mut f.distortion[row * 4 + i],
                                BOX_W,
                                &format!("{n} {a}"),
                            );
                        }
                    });
                    ui.end_row();
                }
            }
            // Row 9: frame_6 (pushButton_MayaCam, pushButton_Apply).
            ui.horizontal(|ui| {
                load = ui
                    .button("Load MayaCam 2.0")
                    .accessible_name(format!("{n} Load MayaCam 2.0"))
                    .clicked();
            });
            ui.horizontal(|ui| {
                apply = ui
                    .button("APPLY CHANGES")
                    .accessible_name(format!("{n} APPLY CHANGES"))
                    .clicked();
            });
            ui.end_row();
        });
    ui.add_space(24.0);
    // Row 10: label (the Matlab scripts and the OpenCV documentation, with their links).
    ui.label("Matlab scripts to convert Projectionmatrices, e.g. Easywand coefs, to xmalab");
    ui.horizontal(|ui| {
        ui.label("and to write them as Mayacam can be download");
        ui.hyperlink_to(
            "here",
            "https://bitbucket.org/xromm/xmalab/downloads/externalCalibration_Matlab_scripts.zip",
        );
    });
    ui.label("");
    ui.label("More information on the camera model can be found in the");
    ui.horizontal(|ui| {
        ui.label("Detailed Description section of the");
        ui.hyperlink_to(
            "OpenCV documentation for calibration",
            "https://docs.opencv.org/3.3.0/d9/d0c/group__calib3d.html",
        );
    });
    (load, apply)
}
