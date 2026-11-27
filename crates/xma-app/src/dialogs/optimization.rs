//! `OptimizationDialog` (C42): the options of "optimize camera setup" when the setting
//! `ShowAdvancedCalibration` is on, and the iterations prompt XMALab asks otherwise (C37).
//!
//! Ported from `ui/OptimizationDialog.cpp`, `ui/ui-files/OptimizationDialog.ui` and
//! `ui/WizardCalibrationCubeFrame.cpp`, `on_pushButtonOptimize_clicked` (`:964-991`):
//!
//! - three radio buttons: "Without Modeldistortion -  norm" (method 0, checked),
//!   "With Modeldistortion - norm" (2) and "Without Modeldistortion -  x and y seperate"
//!   (1), laid out as the form has them (two columns);
//! - "Number of iterations:" 1 to 999999999, 1000;
//! - "Scale factor for initial λ" with 8 decimals, 0.01, range 0 to 99.99 (Qt's default);
//! - OK (the answer [`DialogResult::Optimize`]) and Cancel.
//!
//! Without the setting XMALab asks `QInputDialog::getInt` "Set number of iterations",
//! "Iterations:", 10000, minimum 1, then optimises with method 0 and factor 0.01
//! ([`iterations_prompt`], answered as `IntPurpose::OptimizeIterations`). Its maximum, the
//! literal 99999999999, does not fit in an `int`; the conversion gives 1215752191. Here the
//! maximum is the largest `int`, 2147483647 (ledger, Interface: the defect fixed).

use egui::Ui;

use super::form;
use super::{Dialog, DialogResult, IntPurpose};

/// The dialog's title.
pub const TITLE: &str = "Optimization of camera setup";
/// The radio buttons' texts, by method (0, 1, 2), as the form writes them.
pub const METHODS: [&str; 3] = [
    "Without Modeldistortion -  norm",
    "Without Modeldistortion -  x and y seperate",
    "With Modeldistortion - norm",
];
/// "Number of iterations:" range.
pub const ITERATIONS_MAX: usize = 999_999_999;
/// "Scale factor for initial λ" range (Qt's default for a `QDoubleSpinBox`).
pub const INITIAL_MAX: f64 = 99.99;
/// Decimals of "Scale factor for initial λ".
pub const INITIAL_DECIMALS: usize = 8;

/// The iterations prompt's maximum: the largest `int` (XMALab 3 passes 99999999999, which
/// becomes 1215752191).
pub const PROMPT_MAX: i32 = i32::MAX;

/// What the dialog holds.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationDialog {
    /// 0, 1 or 2 (`getMethod`).
    pub method: i32,
    /// "Number of iterations:".
    pub iterations: usize,
    /// "Scale factor for initial λ".
    pub initial: f64,
}

impl Default for OptimizationDialog {
    fn default() -> Self {
        Self {
            method: 0,
            iterations: 1000,
            initial: 0.01,
        }
    }
}

/// What the form's frame gives back.
#[derive(Clone, Debug, PartialEq)]
pub enum OptimizationAnswer {
    Ok,
    Cancel,
}

impl OptimizationDialog {
    /// OK's answer.
    #[must_use]
    pub fn result(&self) -> DialogResult {
        DialogResult::Optimize {
            method: self.method,
            iterations: self.iterations,
            initial: self.initial,
        }
    }

    /// Draws the form; returns OK or Cancel when clicked (the title row's close button is
    /// Cancel).
    pub fn ui(&mut self, ui: &mut Ui) -> Option<OptimizationAnswer> {
        let mut answer = None;
        ui.set_min_width(520.0);
        if form::dialog_title(ui, TITLE) {
            answer = Some(OptimizationAnswer::Cancel);
        }
        // frame: radioButton_NoDist [r0c0], radioButton_Dist [r0c1], radioButton_NoDistXY
        // [r1c0].
        form::styled_frame(ui, |ui| {
            ui.set_width(ui.available_width());
            egui::Grid::new("optimization_methods")
                .num_columns(2)
                .spacing([24.0, 6.0])
                .show(ui, |ui| {
                    for row in [&[0_usize, 2][..], &[1][..]] {
                        for &m in row {
                            let method = i32::try_from(m).unwrap_or(0);
                            if form::radio(ui, true, self.method == method, METHODS[m]) {
                                self.method = method;
                            }
                        }
                        ui.end_row();
                    }
                });
        });
        ui.add_space(4.0);
        // frame_2: label, spinBox_Iterations; label_2, doubleSpinBox_Initial.
        form::styled_frame(ui, |ui| {
            ui.set_width(ui.available_width());
            egui::Grid::new("optimization_values")
                .num_columns(2)
                .spacing([24.0, 6.0])
                .show(ui, |ui| {
                    ui.label("Number of iterations:");
                    let max = i64::try_from(ITERATIONS_MAX).unwrap_or(i64::MAX);
                    let cur = i64::try_from(self.iterations).unwrap_or(max);
                    if let Some(v) =
                        form::spin_box(ui, true, cur, 1, max, 120.0, "Number of iterations")
                    {
                        self.iterations = usize::try_from(v).unwrap_or(1);
                    }
                    ui.end_row();
                    ui.label("Scale factor for initial λ");
                    if let Some(v) = form::double_spin_box(
                        ui,
                        true,
                        self.initial,
                        0.0..=INITIAL_MAX,
                        INITIAL_DECIMALS,
                        120.0,
                        "Scale factor for initial λ",
                    ) {
                        self.initial = v;
                    }
                    ui.end_row();
                });
        });
        ui.add_space(6.0);
        // pushButton_OK [r2c1], pushButton_Cancel [r2c2], right of a spacer.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Cancel").clicked() {
                answer = Some(OptimizationAnswer::Cancel);
            }
            if ui.button("OK").clicked() {
                answer = Some(OptimizationAnswer::Ok);
            }
        });
        answer
    }
}

/// C37 without `ShowAdvancedCalibration`: `QInputDialog::getInt(this, "Set number of
/// iterations", "Iterations:", 10000, 1, 99999999999, 1, &ok)`, maximum fixed
/// ([`PROMPT_MAX`]).
#[must_use]
pub fn iterations_prompt() -> Dialog {
    Dialog::GetInt {
        title: "Set number of iterations".into(),
        label: "Iterations:".into(),
        value: 10000,
        min: 1,
        max: PROMPT_MAX,
        purpose: IntPurpose::OptimizeIterations,
    }
}

/// The edit C37's prompt leads to: method 0, the iterations entered, factor 0.01
/// (`new MultiCameraCalibration(0, nbIterations, 0.01)`).
#[must_use]
pub fn prompt_parameters(iterations: i32) -> (i32, usize, f64) {
    (0, usize::try_from(iterations.max(1)).unwrap_or(1), 0.01)
}
