//! `DenoiseDialog` (the Trial dialog's "Denoise Images", C65) and `VisualFilterDialog`
//! (Display Options, the button beside "filter images", C67).
//!
//! Ported control by control from `DenoiseDialog.ui` with ui/DenoiseDialog.cpp, and from
//! `VisualFilterDialog.ui` with ui/VisualFilterDialog.cpp.
//!
//! **DenoiseDialog** ("Denoise Image Sequence"), a grid of six rows: "Temporal window size
//! (odd)" `spinBox_temporalWindowSize` (5), "Template window size (odd)"
//! `spinBox_templateWindowSize` (11), "Search window size (odd)" `spinBox_searchWindowSize`
//! (35), each 0 to 99 in steps of 2, an even value raised to the next odd one
//! (`on_spinBox_*_textChanged`); "Filter strenght" `doubleSpinBox_FilterStrength` (0 to 99.99,
//! step 0.1, 5); `checkBox_RelinkTrial` "relink Trial to denoised images" (checked);
//! "Cancel" and "OK". XMALab asks for the output folder before it opens the dialog
//! (`WorkspaceNavigationFrame::on_toolButtonTrialSettings_clicked`); OK starts one job per
//! camera (`app/images.rs`).
//!
//! **VisualFilterDialog** ("Visual Filter Settings"), three group boxes: "Gaussian blur"
//! with "Kernel radius (pixel))" `spinBox_krad` (1 to 99) and "Sigma"
//! `doubleSpinBox_gsigma` (0 to 99.99, step 0.1); "Blend original & blurred images" with
//! "orig. image weight" `doubleSpinBox_img_wt` (0 to 99.99) and "blur image weight"
//! `doubleSpinBox_blur_wt` (-99.99 to 99.99); "Gamma correction" with "Gamma"
//! `doubleSpinBox_gamma` (0 to 99.99). It has no buttons (closed by its title bar). Each
//! change writes its setting (`VisualFilter_krad`, `_gsigma`, `_img_wt`, `_blur_wt`,
//! `_gamma`) at once; the camera views filter with the settings of the frame they are drawn
//! in (`Frames::sync`, keyed by the filter's values), so every image on screen and in the
//! cache of filtered frames is filtered again (XMALab: `reloadTextures`, `redrawGL`).
//! Values outside the spin boxes' ranges in the settings file are shown clamped and changed
//! only when the user changes them (ledger U164).

use egui::Ui;
use xma_engine::denoise::{DenoiseSettings, odd};

use super::edit_forms::FormOutcome;
use super::{DialogEnv, DialogResult, form};
use crate::settings::UiSettings;

/// DenoiseDialog's window title.
pub const DENOISE_TITLE: &str = "Denoise Image Sequence";
/// VisualFilterDialog's window title.
pub const VISUAL_FILTER_TITLE: &str = "Visual Filter Settings";
/// The folder question before the dialog.
pub const DENOISE_FOLDER_TITLE: &str = "Save to Directory ";

/// What the two dialogs (and the Trial dialog's button) answer.
#[derive(Clone, Debug, PartialEq)]
pub enum ImagesAnswer {
    /// "Denoise Images" in the Trial dialog of `trial`: the folder question follows.
    DenoiseAsked { trial: usize },
    /// OK in DenoiseDialog.
    Denoise {
        trial: usize,
        folder: std::path::PathBuf,
        settings: DenoiseSettings,
    },
}

/// DenoiseDialog's state.
#[derive(Clone, Debug, PartialEq)]
pub struct DenoiseDialog {
    pub trial: usize,
    /// The folder chosen before the dialog.
    pub folder: std::path::PathBuf,
    pub settings: DenoiseSettings,
}

impl DenoiseDialog {
    /// The dialog with its form's defaults.
    #[must_use]
    pub fn new(trial: usize, folder: std::path::PathBuf) -> Self {
        DenoiseDialog {
            trial,
            folder,
            settings: DenoiseSettings::default(),
        }
    }

    /// OK's answer.
    #[must_use]
    pub fn result(&self) -> DialogResult {
        DialogResult::Images(ImagesAnswer::Denoise {
            trial: self.trial,
            folder: self.folder.clone(),
            settings: self.settings,
        })
    }

    /// Draws the dialog.
    pub fn show(&mut self, ctx: &egui::Context, _env: &mut DialogEnv<'_>) -> FormOutcome {
        let mut out = FormOutcome::default();
        let modal = form::modal(ctx, egui::Id::unique("xma_denoise_dialog"), |ui| {
            ui.set_width(319.0);
            if form::dialog_title(ui, DENOISE_TITLE) {
                out.close = true;
            }
            egui::Grid::new("denoise_grid")
                .num_columns(2)
                .show(ui, |ui| {
                    let s = &mut self.settings;
                    for (label, value) in [
                        ("Temporal window size (odd)", &mut s.temporal),
                        ("Template window size (odd)", &mut s.template),
                        ("Search window size (odd)", &mut s.search),
                    ] {
                        ui.label(label);
                        // singleStep 2: the arrows move by 2; a typed value is taken as
                        // typed. Either way an even value is raised to the next odd one.
                        if let Some(v) =
                            form::spin_box_step(ui, true, i64::from(*value), 0, 99, 2, 70.0, label)
                        {
                            *value = odd_window(v);
                        }
                        ui.end_row();
                    }
                    ui.label("Filter strenght");
                    if let Some(v) = form::double_spin_box(
                        ui,
                        true,
                        s.strength,
                        0.0..=99.99,
                        2,
                        70.0,
                        "Filter strenght",
                    ) {
                        s.strength = v;
                    }
                    ui.end_row();
                });
            if let Some(v) = form::check_box(
                ui,
                true,
                self.settings.relink,
                "relink Trial to denoised images",
            ) {
                self.settings.relink = v;
            }
            ui.add_space(4.0);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if form::button(ui, true, "OK", 80.0).clicked() {
                    out.result = Some(self.result());
                    out.close = true;
                }
                if form::button(ui, true, "Cancel", 80.0).clicked() {
                    out.close = true;
                }
            });
        });
        if modal.escaped() {
            out.close = true;
        }
        out
    }
}

/// The value a window-size spin box keeps after a change to `v` (an arrow step of 2 or a
/// typed value, `qt::spin_box_step`): within 0 to 99, an even value raised to the next odd
/// one, as `DenoiseDialog::on_spinBox_*_textChanged` does.
#[must_use]
pub fn odd_window(v: i64) -> i32 {
    odd(v.clamp(0, 99) as i32).min(99)
}

/// VisualFilterDialog's state: nothing beyond the settings it edits.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VisualFilterDialog;

/// The label column's width, so that the five fields line up across the three group boxes
/// (as the form's grids line them up).
const LABEL_W: f32 = 150.0;

/// The spin boxes' ranges (`VisualFilterDialog.ui`).
pub const KRAD_RANGE: (i32, i32) = (1, 99);
pub const UNSIGNED_RANGE: (f32, f32) = (0.0, 99.99);
pub const BLUR_WT_RANGE: (f32, f32) = (-99.99, 99.99);

impl VisualFilterDialog {
    /// Draws the dialog; every change goes to `settings` at once.
    pub fn show(&mut self, ctx: &egui::Context, env: &mut DialogEnv<'_>) -> FormOutcome {
        let mut out = FormOutcome::default();
        let modal = form::modal(ctx, egui::Id::unique("xma_visual_filter"), |ui| {
            ui.set_width(375.0);
            if form::dialog_title(ui, VISUAL_FILTER_TITLE) {
                out.close = true;
            }
            out.settings_changed = visual_filter_form(ui, env.settings);
        });
        if modal.escaped() {
            out.close = true;
        }
        out
    }
}

/// The dialog's three group boxes; true when a setting changed.
pub fn visual_filter_form(ui: &mut Ui, s: &mut UiSettings) -> bool {
    let mut changed = false;
    let w = 90.0;
    let clamp =
        |v: f32, (lo, hi): (f32, f32)| f64::from(if v.is_nan() { lo } else { v.clamp(lo, hi) });
    form::group_box(ui, "Gaussian blur", |ui| {
        egui::Grid::new("vf_blur")
            .num_columns(2)
            .min_col_width(LABEL_W)
            .show(ui, |ui| {
                ui.label("Kernel radius (pixel))");
                let k = s.visual_filter_krad.clamp(KRAD_RANGE.0, KRAD_RANGE.1);
                if let Some(v) = form::spin_box(
                    ui,
                    true,
                    i64::from(k),
                    i64::from(KRAD_RANGE.0),
                    i64::from(KRAD_RANGE.1),
                    w,
                    "Kernel radius (pixel))",
                ) {
                    s.visual_filter_krad = v as i32;
                    changed = true;
                }
                ui.end_row();
                ui.label("Sigma");
                if let Some(v) = form::double_spin_box(
                    ui,
                    true,
                    clamp(s.visual_filter_gsigma, UNSIGNED_RANGE),
                    0.0..=99.99,
                    2,
                    w,
                    "Sigma",
                ) {
                    s.visual_filter_gsigma = v as f32;
                    changed = true;
                }
                ui.end_row();
            });
    });
    form::group_box(ui, "Blend original & blurred images", |ui| {
        egui::Grid::new("vf_blend")
            .num_columns(2)
            .min_col_width(LABEL_W)
            .show(ui, |ui| {
                ui.label("orig. image weight");
                if let Some(v) = form::double_spin_box(
                    ui,
                    true,
                    clamp(s.visual_filter_img_wt, UNSIGNED_RANGE),
                    0.0..=99.99,
                    2,
                    w,
                    "orig. image weight",
                ) {
                    s.visual_filter_img_wt = v as f32;
                    changed = true;
                }
                ui.end_row();
                ui.label("blur image weight");
                if let Some(v) = form::double_spin_box(
                    ui,
                    true,
                    clamp(s.visual_filter_blur_wt, BLUR_WT_RANGE),
                    -99.99..=99.99,
                    2,
                    w,
                    "blur image weight",
                ) {
                    s.visual_filter_blur_wt = v as f32;
                    changed = true;
                }
                ui.end_row();
            });
    });
    form::group_box(ui, "Gamma correction", |ui| {
        egui::Grid::new("vf_gamma")
            .num_columns(2)
            .min_col_width(LABEL_W)
            .show(ui, |ui| {
                ui.label("Gamma");
                if let Some(v) = form::double_spin_box(
                    ui,
                    true,
                    clamp(s.visual_filter_gamma, UNSIGNED_RANGE),
                    0.0..=99.99,
                    2,
                    w,
                    "Gamma",
                ) {
                    s.visual_filter_gamma = v as f32;
                    changed = true;
                }
                ui.end_row();
            });
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_sizes_stay_odd() {
        // The arrows' steps of 2 from an odd value (`qt::spin_box_step`).
        assert_eq!(odd_window(7), 7);
        assert_eq!(odd_window(3), 3);
        assert_eq!(odd_window(0), 1);
        assert_eq!(odd_window(99), 99);
        // Typed values: an even one becomes the next odd one (4 over 5 gives 5).
        assert_eq!(odd_window(4), 5);
        assert_eq!(odd_window(6), 7);
        assert_eq!(odd_window(98), 99);
        assert_eq!(odd_window(20), 21);
        assert_eq!(odd_window(-3), 1);
        assert_eq!(odd_window(1000), 99);
    }
}
