//! The Help window (non-modal): the keyboard and mouse reference drawn as keycaps.
//!
//! Ported from `HelpDialog.ui` and ui/HelpDialog.cpp:
//!
//! - `groupBox` "Manual and Tutorial Data", here "Manual": the link `label` "XMALab Manual",
//!   here "RustXMALab User Manual", to this program's wiki; `label_2` "Tutorial" is gone
//!   (parity ledger U172: it led to the tutorial data on the XMAPortal Sandbox, which no
//!   longer exists, and the manual is this program's own). The link opens in the system's
//!   browser (`crate::links`).
//! - `scrollArea` › `groupBox_2` "XMALab Shortcuts": four sections (General, Marker
//!   tracking, Plotwindow, Reprojection or Rigid Body Error plot with selected threshold),
//!   each row the keys (keycaps and the "+", "or", "Hold" between them) and what they do.
//!   The keys of every row end in the same column, so they are drawn right-aligned in a
//!   column of their own.
//!
//! Keycaps are themed as `restyleKeycaps` does (light: grey border, white; dark: #5a5a5a
//! border, #2f2f2f, #e6e6e6 text). On macOS "Ctrl" reads ⌘.
//!
//! Fixed from the C++ (parity ledger): the keycaps of the previous- and next-frame rows were
//! swapped (► for previous); "save dataset as" lacked its Ctrl key ("Shift + + S"); and
//! `restyleKeycaps` turned every short label without a space into a keycap, so "+", "or",
//! "Hold", "zoom" and "pan" were drawn as keys. Here only the labels the form styles as keys
//! are keycaps.

use egui::{RichText, Ui};

/// "Ctrl", or ⌘ on macOS (`HelpDialog::HelpDialog`).
const CTRL: &str = if cfg!(target_os = "macos") {
    "\u{2318}"
} else {
    "Ctrl"
};

/// One cell of the keys column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    /// A key (a label the form styles with a border and a white background).
    K(&'static str),
    /// Plain text: "+", "or", "Hold".
    T(&'static str),
}

use Cell::{K, T};

/// One row of the shortcut table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    Section(&'static str),
    Keys(&'static [Cell], &'static str),
}

/// The rows of `groupBox_2`, in form order.
pub const ROWS: &[Row] = &[
    Row::Section("General"),
    Row::Keys(&[K(CTRL), T("+"), K("N")], "New dataset"),
    Row::Keys(&[K(CTRL), T("+"), K("O")], "open dataset"),
    Row::Keys(&[K(CTRL), T("+"), K("X")], "close dataset"),
    Row::Keys(&[K(CTRL), T("+"), K("S")], "save dataset"),
    Row::Keys(
        &[K("Shift"), T("+"), K(CTRL), T("+"), K("S")],
        "save dataset as",
    ),
    Row::Keys(&[K("◄"), T("or"), K("Q")], "go to the previous frame"),
    Row::Keys(&[K("►"), T("or"), K("W")], "go to the next frame"),
    Row::Keys(
        &[K("5")],
        "go n frames backward  (n can be set in the Settings)",
    ),
    Row::Keys(&[K(CTRL), T("+"), K("5")], "set n and go n frames backward"),
    Row::Keys(
        &[K("6")],
        "go n frames forward (n can be set in the Settings)",
    ),
    Row::Keys(&[K(CTRL), T("+"), K("6")], "set n and go n frames forward"),
    Row::Keys(&[K("mouse wheel")], "zoom"),
    Row::Keys(&[K("hold right mouse while moving")], "pan"),
    Row::Section("Marker tracking"),
    Row::Keys(&[K("▲")], "select the previous point  in points window"),
    Row::Keys(&[K("▼")], "select the next point in points window"),
    Row::Keys(&[K("click in video")], "set location of selected point"),
    Row::Keys(
        &[K(CTRL), T("+"), K("click in video")],
        "set location for next not set point",
    ),
    Row::Keys(
        &[K(CTRL), T("+"), K("Z")],
        "undo last click if active camera, point and frame did not change",
    ),
    Row::Keys(&[K("Shift"), T("+"), K("click")], "select clicked point"),
    Row::Keys(&[K(CTRL), T("+"), K("A")], "select all points"),
    Row::Keys(&[K("1")], "track selected point to the previous frame"),
    Row::Keys(
        &[K(CTRL), T("+"), K("1")],
        "track selected points to the previous frame",
    ),
    Row::Keys(&[K("ALT"), T("+"), K("1")], "track selected point backward"),
    Row::Keys(
        &[K("ALT"), T("+"), K(CTRL), T("+"), K("1")],
        "track selected points backward",
    ),
    Row::Keys(&[K("2")], "track selected point to the next frame"),
    Row::Keys(
        &[K(CTRL), T("+"), K("2")],
        "track selected points to the next frame",
    ),
    Row::Keys(&[K("ALT"), T("+"), K("2")], "track selected point forward"),
    Row::Keys(
        &[K("ALT"), T("+"), K(CTRL), T("+"), K("2")],
        "track selected points forward",
    ),
    Row::Keys(&[K("Esc")], "stops tracking and playing"),
    Row::Keys(
        &[K("3")],
        "go to the last tracked frame of the current tracked segment",
    ),
    Row::Keys(
        &[K(CTRL), T("+"), K("3")],
        "go to the first tracked frame of the current tracked segment",
    ),
    Row::Keys(
        &[K("7")],
        "go to the next frame in which at least 1 but not all cameras are tracked",
    ),
    Row::Keys(
        &[K(CTRL), T("+"), K("7")],
        "go to the previous frame in which at least 1  but not all cameras are tracked",
    ),
    Row::Keys(&[K("I")], "update interpolation of selected point"),
    Row::Keys(
        &[K("Shift"), T("+"), K("I")],
        "update interpolation of all points",
    ),
    Row::Keys(&[K("H")], "toggles hiding of all rendered objects"),
    Row::Keys(
        &[K("C")],
        "Centers the camera views on the currently selected marker",
    ),
    Row::Section("Plotwindow"),
    Row::Keys(
        &[T("Hold"), K("Shift"), T("+"), K("change frame")],
        "select frames from the frame Shift was pressed  to the new frame",
    ),
    Row::Keys(
        &[
            T("Hold"),
            K(CTRL),
            T("+"),
            K("click and hold left mouse while dragging"),
        ],
        "select frames from the clicked to the dragged frame",
    ),
    Row::Keys(&[K(CTRL), T("+"), K("click")], "removes selection"),
    Row::Keys(
        &[K("Backspace"), T("or"), K("Del")],
        "Delete the selected data",
    ),
    Row::Keys(
        &[K(CTRL), T("+"), K("I")],
        "set Interpolation for the current selected frames",
    ),
    Row::Keys(
        &[K(CTRL), T("+"), K("U")],
        "sets point as untrackable for the current selected frames",
    ),
    Row::Keys(
        &[K("O")],
        "marks the drawn events for the current selected frames",
    ),
    Row::Keys(
        &[K(CTRL), T("+"), K("O")],
        "unmarks the drawn events for the current selected frames",
    ),
    Row::Keys(&[K("mouse wheel")], "zoom horizontal"),
    Row::Keys(
        &[T("Hold"), K(CTRL), T("+"), K("mouse wheel")],
        "zoom vertical (left y-axis)",
    ),
    Row::Keys(
        &[T("Hold"), K("Alt"), T("+"), K("mouse wheel")],
        "zoom vertical (right y-axis)",
    ),
    Row::Keys(&[K("hold right mouse while moving")], "pan horizontal"),
    Row::Keys(
        &[
            T("Hold"),
            K(CTRL),
            T("+"),
            K("hold right mouse while moving"),
        ],
        "pan vertical (left y-axis)",
    ),
    Row::Keys(
        &[
            T("Hold"),
            K("Alt"),
            T("+"),
            K("hold right mouse while moving"),
        ],
        "pan vertical (right y-axis)",
    ),
    Row::Section("Reprojection or Rigid Body Error plot with selected threshold"),
    Row::Keys(&[K("4")], "Go to the next frame above the threshold"),
    Row::Keys(
        &[K(CTRL), T("+"), K("4")],
        "Go to the previous frame above the threshold",
    ),
    Row::Keys(
        &[
            K("Shift"),
            T("+ "),
            K("Backspace"),
            T("or"),
            K("Shift"),
            T("+ "),
            K("Del"),
        ],
        "Delete all frames above the threshold",
    ),
];

/// The user manual: the wiki of this program's repository (parity ledger U172).
pub const MANUAL: &str = "https://github.com/pfalkingham/RustXMALab/wiki";
/// The manual link's label.
pub const MANUAL_LABEL: &str = "RustXMALab User Manual";

fn keycap(ui: &mut Ui, text: &str) {
    // A key cap in egui's colours: the text-field background, the frame line and the
    // strong text colour of the current theme.
    let v = ui.visuals();
    let (border, fill, fg) = (
        v.widgets.noninteractive.bg_stroke.color,
        v.extreme_bg_color,
        v.strong_text_color(),
    );
    egui::Frame::NONE
        .fill(fill)
        .stroke(egui::Stroke::new(1.0, border))
        .corner_radius(5.0)
        .inner_margin(egui::Margin::symmetric(6, 2))
        .show(ui, |ui| {
            ui.set_min_width(12.0);
            // The arrow glyphs are not in the bundled fonts: drawn as triangles.
            let dir = match text {
                "◄" => Some(0),
                "►" => Some(1),
                "▲" => Some(2),
                "▼" => Some(3),
                _ => None,
            };
            if let Some(d) = dir {
                let h = ui.text_style_height(&egui::TextStyle::Body);
                let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, h), egui::Sense::hover());
                let c = rect.center();
                let s = 4.5;
                let pts = match d {
                    0 => [
                        c + egui::vec2(-s, 0.0),
                        c + egui::vec2(s, -s),
                        c + egui::vec2(s, s),
                    ],
                    1 => [
                        c + egui::vec2(s, 0.0),
                        c + egui::vec2(-s, -s),
                        c + egui::vec2(-s, s),
                    ],
                    2 => [
                        c + egui::vec2(0.0, -s),
                        c + egui::vec2(-s, s),
                        c + egui::vec2(s, s),
                    ],
                    _ => [
                        c + egui::vec2(0.0, s),
                        c + egui::vec2(-s, -s),
                        c + egui::vec2(s, -s),
                    ],
                };
                ui.painter().add(egui::Shape::convex_polygon(
                    pts.to_vec(),
                    fg,
                    egui::Stroke::NONE,
                ));
            } else {
                ui.label(RichText::new(text).color(fg));
            }
        });
}

/// Width of the keys column.
const KEYS_W: f32 = 330.0;

/// Draws the window's contents.
pub fn ui(ui: &mut Ui) {
    super::form::group_box(ui, "Manual", |ui| {
        ui.vertical_centered(|ui| {
            ui.hyperlink_to(MANUAL_LABEL, MANUAL);
        });
    });
    egui::ScrollArea::vertical()
        .id_salt("help_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            super::form::group_box(ui, "XMALab Shortcuts", |ui| {
                for row in ROWS {
                    match row {
                        Row::Section(t) => {
                            ui.add_space(6.0);
                            ui.label(RichText::new(*t).strong());
                        }
                        Row::Keys(cells, what) => {
                            ui.horizontal(|ui| {
                                ui.allocate_ui_with_layout(
                                    egui::vec2(KEYS_W, ui.spacing().interact_size.y),
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.spacing_mut().item_spacing.x = 4.0;
                                        for c in cells.iter().rev() {
                                            match c {
                                                Cell::K(t) => keycap(ui, t),
                                                Cell::T(t) => {
                                                    ui.label(RichText::new(*t).strong());
                                                }
                                            }
                                        }
                                    },
                                );
                                ui.add_space(12.0);
                                ui.label(*what);
                            });
                        }
                    }
                }
            });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_has_keys_and_the_fixes_are_in() {
        let keys: Vec<_> = ROWS
            .iter()
            .filter_map(|r| match r {
                Row::Keys(c, w) => Some((*c, *w)),
                Row::Section(_) => None,
            })
            .collect();
        assert_eq!(keys.len(), 54);
        assert!(keys.iter().all(|(c, _)| !c.is_empty()));
        let find = |w: &str| keys.iter().find(|(_, d)| *d == w).unwrap().0;
        assert_eq!(find("go to the previous frame")[0], Cell::K("◄"));
        assert_eq!(find("go to the next frame")[0], Cell::K("►"));
        assert_eq!(
            find("save dataset as"),
            &[K("Shift"), T("+"), K(CTRL), T("+"), K("S")]
        );
    }
}
