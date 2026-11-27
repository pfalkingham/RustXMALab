//! What the editing forms of the rigid-body and File-menu package share: how the queue draws
//! them ([`Dialogs::show_edit_form`]), the small Qt dialogs they open over themselves
//! ([`Prompt`]: `ConfirmationDialog`, `QInputDialog::getText`, `getItem`, `getInt`,
//! `ErrorDialog`, `QColorDialog`), and a `QDoubleSpinBox` with arrows ([`double_spin_box`]).
//!
//! The forms are [`super::rigid_body`] (`RigidBodyDialog`), [`super::point_import_export`]
//! (`PointImportExportDialog`) and [`super::trial_import_delete`] (`TrialImportDeleteDialog`).

use egui::{Color32, Ui};
use xma_model::Rgb;

use super::form;
use super::{ConfirmButtons, Dialog, DialogEnv, DialogResult, Dialogs};
use crate::icons::{Icon, Icons};

/// True for the dialogs drawn by [`Dialogs::show_edit_form`].
#[must_use]
pub fn is_edit_form(d: &Dialog) -> bool {
    matches!(
        d,
        Dialog::RigidBody(_)
            | Dialog::PointImportExport(_)
            | Dialog::TrialImportDelete(_)
            | Dialog::NewProject(_)
            | Dialog::NewTrial(_)
            | Dialog::UndistortSequence(_)
            | Dialog::PortalMetadata(_)
            | Dialog::Denoise(_)
            | Dialog::VisualFilter(_)
            | Dialog::BlenderScene(_)
    )
}

/// True for a dialog that stays open under an error dialog raised while it is open (the
/// Rigid body dialog: the errors of its edits are shown over it, as XMALab's are).
#[must_use]
pub fn stays_under_errors(d: &Dialog) -> bool {
    matches!(d, Dialog::RigidBody(_))
}

/// What a form's frame gives back: whether it closes, and its answer.
#[derive(Debug, Default)]
pub struct FormOutcome {
    pub close: bool,
    pub result: Option<DialogResult>,
    /// A setting was changed (the import and export radio buttons).
    pub settings_changed: bool,
}

impl Dialogs {
    /// Draws the editing form on screen (the queue's front) and returns its answer.
    pub(super) fn show_edit_form(
        &mut self,
        ctx: &egui::Context,
        env: &mut DialogEnv<'_>,
    ) -> Option<DialogResult> {
        let out = match self.queue.front_mut()? {
            Dialog::RigidBody(d) => d.show(ctx, env),
            Dialog::PointImportExport(d) => d.show(ctx, env.settings),
            Dialog::TrialImportDelete(d) => d.show(ctx, env.icons),
            Dialog::NewProject(d) => d.show(ctx, env),
            Dialog::NewTrial(d) => d.show(ctx, env),
            Dialog::UndistortSequence(d) => d.show(ctx, env),
            Dialog::PortalMetadata(d) => d.show(ctx, env),
            Dialog::Denoise(d) => d.show(ctx, env),
            Dialog::VisualFilter(d) => d.show(ctx, env),
            Dialog::BlenderScene(d) => d.show(ctx, env),
            _ => return None,
        };
        if out.close {
            self.queue.pop_front();
        }
        self.settings_changed |= out.settings_changed;
        out.result
    }

    /// Renumbers the trials of the queued missing-video requests (`map(old) = new`); a
    /// request whose trial is gone is dropped. Used when trials are imported or deleted.
    pub fn remap_missing(&mut self, map: impl Fn(usize) -> Option<usize>) {
        self.queue.retain_mut(|d| match d {
            Dialog::MissingVideo(m) => match map(m.trial) {
                Some(t) => {
                    m.trial = t;
                    true
                }
                None => false,
            },
            _ => true,
        });
    }
}

// ---------------------------------------------------------------------------------------
// Prompts over a form.

/// A small modal dialog over a form.
#[derive(Clone, Debug, PartialEq)]
pub enum Prompt {
    /// `ConfirmationDialog::showConfirmationDialog` ("Confirmation Required").
    Confirm {
        message: String,
        buttons: ConfirmButtons,
    },
    /// `QInputDialog::getText`.
    Text {
        title: String,
        label: String,
        value: String,
    },
    /// `QInputDialog::getItem` (not editable).
    Item {
        title: String,
        label: String,
        items: Vec<String>,
        selected: usize,
    },
    /// `QInputDialog::getInt`.
    Int {
        title: String,
        label: String,
        value: i32,
        min: i32,
        max: i32,
    },
    /// `ErrorDialog::showErrorDialog`.
    Error { message: String },
    /// `QColorDialog`: the colour shown, and its "HTML:" field.
    Color { color: Rgb, hex: String },
}

/// The answer to a [`Prompt`].
#[derive(Clone, Debug, PartialEq)]
pub enum PromptAnswer {
    /// OK or Yes; with the value entered.
    Accepted(PromptValue),
    /// Cancel, No, Escape or the close button.
    Rejected,
}

/// The value entered in an accepted [`Prompt`].
#[derive(Clone, Debug, PartialEq)]
pub enum PromptValue {
    None,
    Text(String),
    Item(usize),
    Int(i32),
    Color(Rgb),
}

impl Prompt {
    /// A colour dialog starting at `color`.
    #[must_use]
    pub fn color(color: Rgb) -> Self {
        Prompt::Color {
            color,
            hex: form::hex_name(Color32::from_rgb(color.r, color.g, color.b)),
        }
    }

    fn title(&self) -> &str {
        match self {
            Prompt::Confirm { .. } => "Confirmation Required",
            Prompt::Text { title, .. } | Prompt::Item { title, .. } | Prompt::Int { title, .. } => {
                title
            }
            Prompt::Error { .. } => "Error",
            Prompt::Color { .. } => "Select Color",
        }
    }

    /// Draws the prompt over everything; returns the answer when one is given.
    pub fn show(&mut self, ctx: &egui::Context, icons: &Icons) -> Option<PromptAnswer> {
        let mut answer = None;
        let title = self.title().to_owned();
        let modal = form::modal(ctx, egui::Id::unique("xma_edit_form_prompt"), |ui| {
            if matches!(self, Prompt::Color { .. }) {
                // As wide as the colour picker: OK and Cancel at its bottom right.
                ui.set_max_width(form::CHOOSER_WIDTH);
            } else {
                ui.set_min_width(320.0);
            }
            if form::dialog_title(ui, &title) {
                answer = Some(PromptAnswer::Rejected);
            }
            match self {
                Prompt::Confirm { message, buttons } => {
                    form::message_row(ui, icons, Icon::Alert, message);
                    let (yes, no) = buttons.labels();
                    ok_cancel(ui, yes, no, &mut answer, || PromptValue::None);
                }
                Prompt::Error { message } => {
                    form::message_row(ui, icons, Icon::Error, message);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("OK").clicked() {
                            answer = Some(PromptAnswer::Accepted(PromptValue::None));
                        }
                    });
                }
                Prompt::Text { label, value, .. } => {
                    let l = ui.label(label.as_str()).id;
                    ui.add(egui::TextEdit::singleline(value).desired_width(300.0))
                        .labelled_by(l);
                    let v = value.clone();
                    ok_cancel(ui, "OK", "Cancel", &mut answer, || PromptValue::Text(v));
                }
                Prompt::Item {
                    label,
                    items,
                    selected,
                    ..
                } => {
                    let l = ui.label(label.as_str()).id;
                    let names: Vec<&str> = items.iter().map(String::as_str).collect();
                    if let Some(i) = crate::qt::combo_box(
                        ui,
                        "xma_prompt_item",
                        Some(*selected),
                        &names,
                        None,
                        Some(l),
                    ) {
                        *selected = i;
                    }
                    let s = *selected;
                    ok_cancel(ui, "OK", "Cancel", &mut answer, || PromptValue::Item(s));
                }
                Prompt::Int {
                    label,
                    value,
                    min,
                    max,
                    ..
                } => {
                    let l = ui.label(label.as_str()).id;
                    ui.add(egui::DragValue::new(value).range(*min..=*max))
                        .labelled_by(l)
                        .accessible_name("Frame Number value");
                    let v = (*value).clamp(*min, *max);
                    ok_cancel(ui, "OK", "Cancel", &mut answer, || PromptValue::Int(v));
                }
                Prompt::Color { color, hex } => {
                    color_body(ui, color, hex);
                    let c = *color;
                    ok_cancel(ui, "OK", "Cancel", &mut answer, || PromptValue::Color(c));
                }
            }
        });
        if answer.is_none() && modal.escaped() {
            answer = Some(PromptAnswer::Rejected);
        }
        answer
    }
}

/// The accepting and rejecting buttons, right-aligned (rejecting first from the right, as
/// the program's other dialogs place them).
fn ok_cancel(
    ui: &mut Ui,
    yes: &str,
    no: &str,
    answer: &mut Option<PromptAnswer>,
    value: impl FnOnce() -> PromptValue,
) {
    ui.add_space(6.0);
    match form::reject_accept(ui, yes, no) {
        Some(true) => *answer = Some(PromptAnswer::Accepted(value())),
        Some(false) => *answer = Some(PromptAnswer::Rejected),
        None => {}
    }
}

/// `QColorDialog`: egui's colour picker and the "HTML:" field ([`form::color_chooser`]).
fn color_body(ui: &mut Ui, color: &mut Rgb, hex: &mut String) {
    let mut c = Color32::from_rgb(color.r, color.g, color.b);
    form::color_chooser(ui, &mut c, hex);
    *color = Rgb::new(c.r(), c.g(), c.b());
}

// ---------------------------------------------------------------------------------------
// A QDoubleSpinBox.

/// A `QDoubleSpinBox` (2 decimals shown, single step 1, `min..=max`) with up and down
/// arrows named "<name> up" and "<name> down". `value` is what it shows; `edit` holds a
/// value being typed or dragged. Returns the value to write once the user has finished
/// changing it (an arrow click, the end of a drag, or the field losing focus after typing),
/// never the value it was opened with: a stored value with more decimals is shown rounded
/// and kept (parity ledger D14 fixed).
#[allow(clippy::too_many_arguments)]
pub fn double_spin_box(
    ui: &mut Ui,
    enabled: bool,
    value: f64,
    edit: &mut Option<f64>,
    min: f64,
    max: f64,
    width: f32,
    name: &str,
) -> Option<f64> {
    let mut commit = None;
    ui.add_enabled_ui(enabled, |ui| {
        let h = ui.spacing().interact_size.y;
        ui.spacing_mut().item_spacing.x = 0.0;
        let mut v = edit.unwrap_or(value);
        let field = ui
            .add_sized(
                [width, h],
                egui::DragValue::new(&mut v)
                    .range(min..=max)
                    .speed(0.05)
                    .fixed_decimals(2),
            )
            .accessible_name(name);
        if field.changed() {
            *edit = Some(v);
        }
        if edit.is_some() && !field.dragged() && !field.has_focus() {
            commit = edit.take();
        }
        if let Some(delta) = crate::qt::spin_arrows(ui, field.id, name) {
            // From what was typed, when the field gave it up in this same frame (the press
            // on the arrow ends the typing; a press and its release can share a frame).
            let from = commit.unwrap_or(value);
            commit = Some((from + f64::from(delta)).clamp(min, max));
            *edit = None;
        }
    });
    commit.filter(|v| *v != value)
}
