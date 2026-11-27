//! Questions asked before the project changes: a confirmation, an information box, a choice
//! from a list, a line of text or a colour, each with what its answer does.
//!
//! XMALab 3 asks inside its widgets' handlers (`ConfirmationDialog::showConfirmationDialog`,
//! `QMessageBox::information`, `QInputDialog::getItem` and `getText`,
//! `QColorDialog::getColor`) and writes to the model on the answer. Here a panel that must
//! ask raises `cx.request(EditRequest::Ask(Box::new(ask)))`; the application opens it as a
//! modal dialog ([`super::Dialog::Ask`]), and the answer becomes the edit requests and
//! actions of [`Answered`], handled like any other. The question is built from the panel's
//! state when it is asked; nothing else can change the project while it is open, so the
//! edit it carries still applies when it is answered.
//!
//! Used by the Plot panel (range delete, untrackable, delete above the error threshold,
//! interpolation method) and the Events panel (add, delete, colour).

use egui::Color32;
use xma_engine::Edit;
use xma_engine::edit::FrameRange;
use xma_model::{Interpolation, Rgb};

use super::{ConfirmButtons, form};
use crate::actions::Action;
use crate::editing::EditRequest;
use crate::icons::{Icon, Icons};

/// One question and what its answer does.
#[derive(Clone, Debug, PartialEq)]
pub struct Ask {
    /// The window title.
    pub title: String,
    pub kind: AskKind,
    pub then: Then,
}

/// What is asked. The values in it are the dialog's current ones (the item chosen, the
/// text typed, the colour picked); the dialog edits them while it is open.
#[derive(Clone, Debug, PartialEq)]
pub enum AskKind {
    /// `ConfirmationDialog`: the Alert icon, the message, and two buttons.
    Confirm {
        message: String,
        buttons: ConfirmButtons,
    },
    /// `QMessageBox::information`: the text and "OK"; answers nothing.
    Information { text: String },
    /// `QInputDialog::getItem` (not editable): a label and a drop-down list, OK and Cancel.
    Item {
        label: String,
        items: Vec<String>,
        current: usize,
    },
    /// `QInputDialog::getText`: a label and a line edit, OK and Cancel.
    Text { label: String, value: String },
    /// `QColorDialog::getColor`: red, green and blue, OK and Cancel.
    /// `QColorDialog`: the colour and the text of its "HTML:" field.
    Color { color: Rgb, hex: String },
}

/// What the accepting answer (OK, Yes) does.
#[derive(Clone, Debug, PartialEq)]
pub enum Then {
    /// Nothing (an information box).
    Nothing,
    /// A confirmation: this request.
    Request(Box<EditRequest>),
    /// A confirmation: this action (for example `Action::Confirmed(..)`, so that the
    /// handler goes on past the question).
    Raise(Action),
    /// `Item` over [`INTERPOLATION_METHODS`]: the method of the frames of a marker (E46).
    SetInterpolation {
        trial: usize,
        marker: usize,
        frames: FrameRange,
    },
    /// `Item` over the event names: removes the chosen event (E50).
    RemoveEvent { trial: usize },
    /// `Text`: adds an event with that name (E49); an empty name adds nothing.
    AddEvent { trial: usize },
    /// `Color`: the colour of an event (E53).
    EventColor { trial: usize, event: usize },
}

/// The items of "Choose interpolation method" (`PlotWindow::setInterpolation`).
pub const INTERPOLATION_METHODS: [&str; 4] = ["None", "Repeat", "Linear", "Cubic"];

/// What the answer asks the application to do: requests to handle, then actions to raise.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Answered {
    pub requests: Vec<EditRequest>,
    pub actions: Vec<Action>,
}

impl Ask {
    /// A confirmation with OK and Cancel ("Confirmation Required").
    #[must_use]
    pub fn confirm(message: impl Into<String>, then: Then) -> Self {
        Ask {
            title: "Confirmation Required".to_owned(),
            kind: AskKind::Confirm {
                message: message.into(),
                buttons: ConfirmButtons::OkCancel,
            },
            then,
        }
    }

    /// The text the dialog shows (the message, the information text, or the label), for
    /// tests.
    #[must_use]
    pub fn text(&self) -> &str {
        match &self.kind {
            AskKind::Confirm { message, .. } => message,
            AskKind::Information { text } => text,
            AskKind::Item { label, .. } | AskKind::Text { label, .. } => label,
            AskKind::Color { .. } => "",
        }
    }

    /// Chooses an item of an `Item` question (tests, and the drop-down list).
    pub fn choose(&mut self, index: usize) {
        if let AskKind::Item { items, current, .. } = &mut self.kind
            && index < items.len()
        {
            *current = index;
        }
    }

    /// Sets the text of a `Text` question.
    pub fn set_text(&mut self, text: &str) {
        if let AskKind::Text { value, .. } = &mut self.kind {
            *value = text.to_owned();
        }
    }

    /// Sets the colour of a `Color` question.
    pub fn set_color(&mut self, c: Rgb) {
        if let AskKind::Color { color, hex } = &mut self.kind {
            *color = c;
            *hex = crate::dialogs::form::hex_name(Color32::from_rgb(c.r, c.g, c.b));
        }
    }

    /// What OK (or Yes) does with the dialog's current values.
    #[must_use]
    pub fn accept(&self) -> Answered {
        let apply = |e: Edit| Answered {
            requests: vec![EditRequest::Apply(e)],
            actions: Vec::new(),
        };
        match (&self.then, &self.kind) {
            (Then::Nothing, _) => Answered::default(),
            (Then::Request(r), _) => Answered {
                requests: vec![(**r).clone()],
                actions: Vec::new(),
            },
            (Then::Raise(a), _) => Answered {
                requests: Vec::new(),
                actions: vec![a.clone()],
            },
            (
                Then::SetInterpolation {
                    trial,
                    marker,
                    frames,
                },
                AskKind::Item { current, .. },
            ) => {
                let method = match current {
                    0 => Interpolation::None,
                    1 => Interpolation::Repeat,
                    2 => Interpolation::Linear,
                    _ => Interpolation::Cubic,
                };
                apply(Edit::SetInterpolation {
                    trial: *trial,
                    marker: *marker,
                    frames: *frames,
                    method,
                })
            }
            (Then::RemoveEvent { trial }, AskKind::Item { items, current, .. }) => {
                if *current < items.len() {
                    apply(Edit::RemoveEvent {
                        trial: *trial,
                        event: *current,
                    })
                } else {
                    Answered::default()
                }
            }
            (Then::AddEvent { trial }, AskKind::Text { value, .. }) => {
                // `ok && !name.isEmpty()`.
                if value.is_empty() {
                    Answered::default()
                } else {
                    apply(Edit::AddEvent {
                        trial: *trial,
                        name: value.clone(),
                    })
                }
            }
            (Then::EventColor { trial, event }, AskKind::Color { color, .. }) => {
                apply(Edit::SetEventColor {
                    trial: *trial,
                    event: *event,
                    color: *color,
                })
            }
            _ => Answered::default(),
        }
    }
}

/// Draws the question inside the modal dialog. Returns `Some(Some(answer))` for OK or Yes,
/// `Some(None)` for Cancel or No, `None` while it stays open.
pub fn show(ui: &mut egui::Ui, ask: &mut Ask, icons: &Icons) -> Option<Option<Answered>> {
    if matches!(ask.kind, AskKind::Color { .. }) {
        // As wide as the colour picker: OK and Cancel at its bottom right.
        ui.set_max_width(crate::dialogs::form::CHOOSER_WIDTH);
    } else {
        ui.set_min_width(360.0);
    }
    ui.label(egui::RichText::new(ask.title.as_str()).strong());
    ui.add_space(6.0);
    let mut out = None;
    let ok_cancel = |ui: &mut egui::Ui, out: &mut Option<bool>, accept: &str, reject: &str| {
        if let Some(a) = form::reject_accept(ui, accept, reject) {
            *out = Some(a);
        }
    };
    let mut answer: Option<bool> = None;
    match &mut ask.kind {
        AskKind::Confirm { message, buttons } => {
            form::message_row(ui, icons, Icon::Alert, message);
            let (a, r) = buttons.labels();
            ok_cancel(ui, &mut answer, a, r);
        }
        AskKind::Information { text } => {
            ui.label(text.as_str());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("OK").clicked() {
                    answer = Some(true);
                }
            });
        }
        AskKind::Item {
            label,
            items,
            current,
        } => {
            let l = ui.label(label.as_str());
            let shown = items.get(*current).cloned().unwrap_or_default();
            let r = egui::ComboBox::from_id_salt("xma_ask_item")
                .selected_text(shown)
                .width(240.0)
                .show_ui(ui, |ui| {
                    for (i, it) in items.iter().enumerate() {
                        ui.selectable_value(current, i, it.as_str());
                    }
                });
            if let Some(i) =
                crate::qt::combo_wheel(ui, &r.response, Some(*current), items.len(), |_| true)
            {
                *current = i;
            }
            r.response.labelled_by(l.id);
            ui.add_space(6.0);
            ok_cancel(ui, &mut answer, "OK", "Cancel");
        }
        AskKind::Text { label, value } => {
            let l = ui.label(label.as_str());
            let r = ui
                .add(egui::TextEdit::singleline(value).desired_width(240.0))
                .labelled_by(l.id);
            if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                answer = Some(true);
            }
            ui.add_space(6.0);
            ok_cancel(ui, &mut answer, "OK", "Cancel");
        }
        AskKind::Color { color, hex } => {
            // QColorDialog: egui's colour picker and the "HTML:" field.
            let mut c = Color32::from_rgb(color.r, color.g, color.b);
            crate::dialogs::form::color_chooser(ui, &mut c, hex);
            *color = Rgb::new(c.r(), c.g(), c.b());
            ok_cancel(ui, &mut answer, "OK", "Cancel");
        }
    }
    match answer {
        Some(true) => out = Some(Some(ask.accept())),
        Some(false) => out = Some(None),
        None => {}
    }
    out
}
