//! `MarkerDialog` ("Marker"): one marker's detection and tracking settings (E29), opened
//! from the settings button of a marker row in the Points tree.
//!
//! Ported control by control from `MarkerDialog.ui` and ui/MarkerDialog.cpp:
//!
//! - `groupBox` "Detection" (row 0): "Detectionmethod" `comboBox_Method` (the seven
//!   methods, with the tool tips the constructor sets); "Marker radius in pixel (-1 for
//!   automatic)" `spinBox_MarkerRadius` (-1 to 50), which the constructor hides with its
//!   label; "Threshold offset in percent" `spinBox_ThresholdOffset` (0 to 100).
//! - `groupBox_2` "Tracking" (row 1): "Maximum penalty for prediction distance (in
//!   percent)" `spinBox_Penalty` (0 to 100).
//! - Row 2: a spacer, "Cancel", "OK".
//!
//! In a trial linked to the Default trial the method is disabled; in the Default trial the
//! three spin boxes are. OK writes the enabled values (`MarkerDialog::isComplete`), Cancel
//! nothing.
//!
//! Difference: only the values the user changed are written. XMALab 3 writes every enabled
//! control's value on OK, so a stored value outside a spin box's range (a threshold offset
//! from a `DefaultMarkerThreshold` above 100, a legacy penalty of 125) is replaced by the
//! clamped one although the user changed nothing.

use egui::Ui;
use xma_engine::edit::MarkerProperties;
use xma_model::{DetectionMethod, Marker};

use super::form;

/// The seven methods in the order of the combo box (method codes 0 to 6).
pub const METHODS: [&str; 7] = [
    "default Xray marker",
    "Blobdetection",
    "white marker",
    "corner detection",
    "no detection",
    "xray marker (center of mass)",
    "white Blobdetection",
];

/// The tool tips of the combo box items (`MarkerDialog::MarkerDialog`).
pub const METHOD_TIPS: [&str; 7] = [
    "Optimized for dark, spherical beads in X-rays",
    "Uses OpenCV SimpleBlobDetector to find circular blobs",
    "Inverts image to track bright white beads on a dark background",
    "Uses Shi-Tomasi corner finder for sharp angles (e.g. checkerboards)",
    "Turns off automatic snapping. Marker stays exactly where clicked",
    "Calculates exact center of mass of dark pixels instead of fitting a contour",
    "OpenCV SimpleBlobDetector for bright circular blobs",
];

/// The combo box's tool tip.
pub const METHOD_TIP: &str = "Select the computer vision algorithm used to snap to the marker.";

/// The dialog's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarkerDialog {
    /// The marker, 0-based.
    pub marker: usize,
    /// `comboBox_Method`, as a method code.
    pub method: usize,
    /// `spinBox_MarkerRadius` (hidden).
    pub size_override: i64,
    /// `spinBox_ThresholdOffset`.
    pub threshold_offset: i64,
    /// `spinBox_Penalty`.
    pub penalty: i64,
    /// The values shown when the dialog opened (after the spin boxes' clamping).
    initial: (usize, i64, i64, i64),
    /// The trial is the Default trial: the spin boxes are disabled.
    pub is_default: bool,
    /// The trial is linked to the Default trial: the method is disabled.
    pub is_copy: bool,
}

/// What the buttons did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerAnswer {
    Ok,
    Cancel,
}

impl MarkerDialog {
    /// Ported from ui/MarkerDialog.cpp, MarkerDialog::MarkerDialog: the controls show the
    /// marker's values, clamped to their ranges as `QSpinBox::setValue` does.
    #[must_use]
    pub fn new(marker_index: usize, m: &Marker, is_default: bool, is_copy: bool) -> Self {
        let method = usize::try_from(m.method.code()).unwrap_or(0).min(6);
        let size_override = i64::from(m.size_override).clamp(-1, 50);
        let threshold_offset = i64::from(m.threshold_offset).clamp(0, 100);
        let penalty = i64::from(m.max_penalty).clamp(0, 100);
        MarkerDialog {
            marker: marker_index,
            method,
            size_override,
            threshold_offset,
            penalty,
            initial: (method, size_override, threshold_offset, penalty),
            is_default,
            is_copy,
        }
    }

    /// What OK writes: the values of the enabled controls (`MarkerDialog::isComplete`)
    /// that differ from what the dialog showed when it opened.
    ///
    /// Ported from ui/MarkerDialog.cpp, MarkerDialog::isComplete.
    #[must_use]
    pub fn properties(&self) -> MarkerProperties {
        let (m0, s0, t0, p0) = self.initial;
        let int = |v: i64| i32::try_from(v).unwrap_or(0);
        MarkerProperties {
            method: (!self.is_copy && self.method != m0)
                .then(|| DetectionMethod::from_code(int(self.method as i64)))
                .flatten(),
            threshold_offset: (!self.is_default && self.threshold_offset != t0)
                .then(|| int(self.threshold_offset)),
            size_override: (!self.is_default && self.size_override != s0)
                .then(|| int(self.size_override)),
            max_penalty: (!self.is_default && self.penalty != p0).then(|| int(self.penalty)),
        }
    }

    /// Draws the form below the title row; returns the button pressed.
    pub fn ui(&mut self, ui: &mut Ui) -> Option<MarkerAnswer> {
        ui.set_min_width(400.0);
        // groupBox "Detection".
        form::group_box(ui, "Detection", |ui| {
            ui.horizontal(|ui| {
                let l = ui.label("Detectionmethod").id;
                ui.add_enabled_ui(!self.is_copy, |ui| {
                    if let Some(i) = method_combo(ui, l, self.method) {
                        self.method = i;
                    }
                });
            });
            // label and spinBox_MarkerRadius are hidden by the constructor.
            ui.horizontal(|ui| {
                ui.label("Threshold offset in percent");
                if let Some(v) = form::spin_box(
                    ui,
                    !self.is_default,
                    self.threshold_offset,
                    0,
                    100,
                    60.0,
                    "Threshold offset",
                ) {
                    self.threshold_offset = v;
                }
            });
        });
        // groupBox_2 "Tracking".
        form::group_box(ui, "Tracking", |ui| {
            ui.horizontal(|ui| {
                ui.label("Maximum penalty for prediction distance (in percent)");
                if let Some(v) = form::spin_box(
                    ui,
                    !self.is_default,
                    self.penalty,
                    0,
                    100,
                    60.0,
                    "Maximum penalty",
                ) {
                    self.penalty = v;
                }
            });
        });
        ui.add_space(4.0);
        let mut answer = None;
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if form::button(ui, true, "OK", 80.0).clicked() {
                answer = Some(MarkerAnswer::Ok);
            }
            if form::button(ui, true, "Cancel", 80.0).clicked() {
                answer = Some(MarkerAnswer::Cancel);
            }
        });
        answer
    }
}

/// `comboBox_Method` with its tool tips; returns the newly chosen index.
fn method_combo(ui: &mut Ui, label: egui::Id, selected: usize) -> Option<usize> {
    let mut chosen = None;
    let r = egui::ComboBox::from_id_salt(("marker_method", label))
        .selected_text(METHODS.get(selected).copied().unwrap_or(""))
        .width(220.0)
        .show_ui(ui, |ui| {
            for (i, (t, tip)) in METHODS.iter().zip(METHOD_TIPS).enumerate() {
                if ui
                    .selectable_label(i == selected, *t)
                    .on_hover_text(tip)
                    .clicked()
                    && i != selected
                {
                    chosen = Some(i);
                }
            }
        });
    let chosen = chosen.or_else(|| {
        crate::qt::combo_wheel(ui, &r.response, Some(selected), METHODS.len(), |_| true)
    });
    let _ = r.response.labelled_by(label).on_hover_text(METHOD_TIP);
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker() -> Marker {
        let mut m = Marker::new(2, 5, 8);
        m.max_penalty = 125;
        m.threshold_offset = 150;
        m
    }

    #[test]
    fn ok_writes_only_changed_enabled_values() {
        let mut d = MarkerDialog::new(0, &marker(), false, false);
        assert_eq!(
            (d.penalty, d.threshold_offset),
            (100, 100),
            "clamped as Qt does"
        );
        assert_eq!(d.properties(), MarkerProperties::default());
        d.method = 4;
        d.penalty = 30;
        let p = d.properties();
        assert_eq!(p.method, Some(DetectionMethod::NoDetection));
        assert_eq!(p.max_penalty, Some(30));
        assert_eq!(p.threshold_offset, None);
    }

    #[test]
    fn default_and_linked_trials_write_what_is_enabled() {
        let mut d = MarkerDialog::new(0, &marker(), true, false);
        d.method = 2;
        d.penalty = 10;
        let p = d.properties();
        assert_eq!(p.method, Some(DetectionMethod::WhiteMarker));
        assert_eq!(p.max_penalty, None);
        let mut d = MarkerDialog::new(0, &marker(), false, true);
        d.method = 2;
        d.penalty = 10;
        let p = d.properties();
        assert_eq!(p.method, None);
        assert_eq!(p.max_penalty, Some(10));
    }
}
