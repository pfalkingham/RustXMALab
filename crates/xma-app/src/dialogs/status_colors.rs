//! The Status colours dialog ("Marker status", modal).
//!
//! Ported control by control from `StatusColorDialog.ui` and ui/StatusColorDialog.cpp: two
//! group boxes, "MarkerStatus" (nine marker states) and "Interpolation method" (four
//! methods), each row a label and a tool button showing the colour as a swatch.
//! A click opens a colour chooser (`QColorDialog::getColor`; here egui's colour picker,
//! [`form::color_chooser`]) and, on OK, stores the colour
//! as `#rrggbb` in its setting (`setColor`); the views read the settings each frame, so the
//! new colour shows at once.

use egui::{Color32, Ui};

use super::form;
use crate::settings::{StatusColors, parse_color};

/// One row: label, setting name, field.
pub struct ColorRow {
    pub label: &'static str,
    /// The C++ setting name, also the button's accessible name.
    pub key: &'static str,
    pub get: fn(&StatusColors) -> &String,
    pub get_mut: fn(&mut StatusColors) -> &mut String,
}

macro_rules! row {
    ($label:literal, $key:literal, $f:ident) => {
        ColorRow {
            label: $label,
            key: $key,
            get: |c| &c.$f,
            get_mut: |c| &mut c.$f,
        }
    };
}

/// `groupBox` "MarkerStatus", in form order.
pub const MARKER_STATUS: [ColorRow; 9] = [
    row!("Untrackable", "ColorUntrackable", untrackable),
    row!("Undefined", "ColorUndefined", undefined),
    row!("Interpolated", "ColorInterpolated", interpolated),
    row!("Tracked", "ColorTracked", tracked),
    row!(
        "Tracked and Optimized",
        "ColorTrackedAndOpt",
        tracked_and_opt
    ),
    row!("Set from mainwindow", "ColorSet", set),
    row!("Set and Optimized", "ColorSetAndOpt", set_and_opt),
    row!("Manual from detailwindow", "ColorManual", manual),
    row!("Manual and Optimized", "ColorManualAndOpt", manual_and_opt),
];

/// `groupBox` "MarkerStatus", last row.
pub const PREDICTED: ColorRow = row!(
    "Predicted (held during an overlap)",
    "ColorPredicted",
    predicted
);

/// `groupBox_2` "Interpolation method", in form order.
pub const INTERPOLATION: [ColorRow; 4] = [
    row!("None", "ColorInterNone", inter_none),
    row!("Repeat", "ColorInterRepeat", inter_repeat),
    row!("Linear", "ColorInterLinear", inter_linear),
    row!("Cubic", "ColorInterCubic", inter_cubic),
];

/// Every row, in form order.
pub fn rows() -> impl Iterator<Item = &'static ColorRow> {
    MARKER_STATUS
        .iter()
        .chain(std::iter::once(&PREDICTED))
        .chain(INTERPOLATION.iter())
}

/// The colour chooser opened for one setting.
#[derive(Clone, Debug, PartialEq)]
pub struct Chooser {
    pub key: &'static str,
    pub color: Color32,
    /// The "HTML:" field.
    pub hex: String,
}

/// The dialog's own state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StatusColorDialog {
    pub chooser: Option<Chooser>,
}

impl StatusColorDialog {
    /// Draws the dialog; returns true when it is closed. `changed` is set when a colour
    /// setting changed.
    pub fn ui(&mut self, ui: &mut Ui, colors: &mut StatusColors, changed: &mut bool) -> bool {
        ui.set_max_width(300.0);
        let close = form::dialog_title(ui, "Marker status");
        let mut open = None;
        let mut group = |ui: &mut Ui, title: &str, rows: &[&'static ColorRow]| {
            form::group_box(ui, title, |ui| {
                egui::Grid::new(title).num_columns(2).show(ui, |ui| {
                    for r in rows {
                        ui.label(r.label);
                        let c = parse_color((r.get)(colors));
                        if form::swatch_button(ui, c, r.key).clicked() {
                            open = Some((r.key, c));
                        }
                        ui.end_row();
                    }
                });
            });
        };
        let status: Vec<&ColorRow> = MARKER_STATUS
            .iter()
            .chain(std::iter::once(&PREDICTED))
            .collect();
        group(ui, "MarkerStatus", &status);
        let inter: Vec<&ColorRow> = INTERPOLATION.iter().collect();
        group(ui, "Interpolation method", &inter);
        if let Some((key, color)) = open {
            self.chooser = Some(Chooser {
                key,
                color,
                hex: form::hex_name(color),
            });
        }
        if let Some(ch) = &mut self.chooser {
            let mut done = None;
            egui::Modal::new(egui::Id::unique("xma_color_chooser")).show(ui.ctx(), |ui| {
                // As wide as the colour picker: OK and Cancel at its bottom right.
                ui.set_max_width(form::CHOOSER_WIDTH);
                if form::dialog_title(ui, "Select Color") {
                    done = Some(false);
                }
                // QColorDialog: egui's colour picker and the "HTML:" field.
                form::color_chooser(ui, &mut ch.color, &mut ch.hex);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Cancel").clicked() {
                        done = Some(false);
                    }
                    if ui.button("OK").clicked() {
                        done = Some(true);
                    }
                });
            });
            match done {
                Some(true) => {
                    if let Some(r) = rows().find(|r| r.key == ch.key) {
                        *(r.get_mut)(colors) = form::hex_name(ch.color);
                        *changed = true;
                    }
                    self.chooser = None;
                }
                Some(false) => self.chooser = None,
                None => {}
            }
        }
        close
    }
}
