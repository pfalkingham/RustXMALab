//! The Events panel.
//!
//! Ported from `EventDockWidget.ui` and ui/EventDockWidget.cpp (ui-map.md,
//! "EventDockWidget"). The form: a scroll area (row 0, both columns) holding a frame
//! (`QFrame::StyledPanel`, raised) with one row per event of the active trial, then a
//! spacer; under it "Add event" and "Delete event" (row 1), "Mark frames" and "Unmark
//! frames" (row 2).
//!
//! Each event row (`EventDockWidget::addEvent`): a check box with the event's name, checked
//! when the event is drawn (`EventData::getDraw`), and a tool button showing a 16x16 swatch
//! of the event's colour. The list is filled from the active trial in the Marker tracking
//! workspace only (`workspaceChanged`, `activeTrialChanged`, `populateActive`).
//!
//! Everything here changes the project (E49 to E54, `docs/cpp-editing.md` section 7): the
//! check box sets the event's draw flag and the swatch opens a colour chooser (both saved
//! with the project); "Add event" asks for a name ("Set event name", default
//! `Event<n+1>`), "Delete event" for the event ("Choose event to delete"); "Mark frames"
//! and "Unmark frames" raise `Action::SetEventOn` and `SetEventOff` (also O and Ctrl+O),
//! which the Plot panel carries out on its selected range for every drawn event. The
//! controls are enabled while the list shows a trial and no tracking run holds the project.
//! With no event, "Delete event" asks nothing (the C++ shows an empty list).

use egui::{Color32, Ui, vec2};
use xma_engine::Edit;
use xma_model::Rgb;

use super::{Cx, Panel, PanelId};
use crate::actions::Action;
use crate::dialogs::ask::{Ask, AskKind, Then};
use crate::editing::EditRequest;
use crate::state::Workspace;

/// The Events panel.
#[derive(Debug, Default)]
pub struct EventsPanel;

/// One event as the panel lists it: name, colour, draw flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventEntry {
    pub name: String,
    pub color: Color32,
    pub draw: bool,
}

/// The events the panel lists: the active trial's, in the Marker tracking workspace.
///
/// Ported from ui/EventDockWidget.cpp, EventDockWidget::populateActive and workspaceChanged.
#[must_use]
pub fn listed_events(cx: &Cx<'_>) -> Vec<EventEntry> {
    if cx.state.workspace != Workspace::MarkerTracking {
        return Vec::new();
    }
    let Some(trial) = cx
        .doc
        .zip(cx.state.active_trial)
        .and_then(|(d, t)| d.project().trials.get(t))
    else {
        return Vec::new();
    };
    trial
        .events
        .iter()
        .map(|e| EventEntry {
            name: e.name.clone(),
            color: Color32::from_rgb(e.color.r, e.color.g, e.color.b),
            draw: e.draw,
        })
        .collect()
}

/// What a control of the panel asked for this frame.
enum Clicked {
    Draw(usize, bool),
    Color(usize),
    Add,
    Delete,
    Mark,
    Unmark,
}

/// E49 "Add event": `QInputDialog::getText(this, "Set event name", "name:", Normal,
/// "Event<n+1>")`.
///
/// Ported from ui/EventDockWidget.cpp, EventDockWidget::on_pushButtonAdd_clicked.
#[must_use]
pub fn add_event_ask(trial: usize, n_events: usize) -> Ask {
    Ask {
        title: "Set event name".to_owned(),
        kind: AskKind::Text {
            label: "name:".to_owned(),
            value: format!("Event{}", n_events + 1),
        },
        then: Then::AddEvent { trial },
    }
}

/// E50 "Delete event": `QInputDialog::getItem(this, "Choose event to delete", "Event:",
/// names, 0, false)`. `None` without events.
///
/// Ported from ui/EventDockWidget.cpp, EventDockWidget::on_pushButtonDelete_clicked.
#[must_use]
pub fn delete_event_ask(trial: usize, names: Vec<String>) -> Option<Ask> {
    if names.is_empty() {
        return None;
    }
    Some(Ask {
        title: "Choose event to delete".to_owned(),
        kind: AskKind::Item {
            label: "Event:".to_owned(),
            items: names,
            current: 0,
        },
        then: Then::RemoveEvent { trial },
    })
}

/// E53, the colour button: `QColorDialog::getColor(current)`.
///
/// Ported from ui/EventDockWidget.cpp, EventDockWidget::changeColor.
#[must_use]
pub fn event_color_ask(trial: usize, event: usize, color: Rgb) -> Ask {
    Ask {
        title: "Select Color".to_owned(),
        kind: AskKind::Color {
            color,
            hex: crate::dialogs::form::hex_name(Color32::from_rgb(color.r, color.g, color.b)),
        },
        then: Then::EventColor { trial, event },
    }
}

impl Panel for EventsPanel {
    fn id(&self) -> PanelId {
        PanelId::Events
    }

    fn ui(&mut self, ui: &mut Ui, cx: &mut Cx<'_>) {
        let events = listed_events(cx);
        let trial = cx.state.active_trial.filter(|_| {
            cx.state.workspace == Workspace::MarkerTracking
                && cx
                    .doc
                    .zip(cx.state.active_trial)
                    .is_some_and(|(d, t)| t < d.project().trials.len())
        });
        let enabled = trial.is_some() && !cx.state.disable_draw;
        let mut clicked: Option<Clicked> = None;
        let full = ui.available_rect_before_wrap();
        let margin = 6.0;
        let inner = full.shrink(margin);
        let row_h = ui.spacing().interact_size.y;
        let gap = 6.0;
        let buttons_h = 2.0 * row_h + 2.0 * gap;
        let area = egui::Rect::from_min_max(
            inner.min,
            egui::pos2(
                inner.max.x,
                (inner.max.y - buttons_h).max(inner.min.y + 20.0),
            ),
        );
        let frame_line = crate::theme::frame_line(ui.visuals());

        // scrollArea (a framed, base-coloured box) with the event frame inside.
        ui.painter().rect(
            area,
            0.0,
            ui.visuals().extreme_bg_color,
            egui::Stroke::new(1.0, frame_line),
            egui::StrokeKind::Inside,
        );
        ui.scope_builder(egui::UiBuilder::new().max_rect(area.shrink(2.0)), |ui| {
            egui::ScrollArea::vertical()
                .id_salt("events_list")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add_space(8.0);
                    // frame (StyledPanel, Raised); its grid has no margin, so an empty list
                    // shows as a single line, as in the reference.
                    let w = ui.available_width() - 16.0;
                    let resp = ui
                        .horizontal(|ui| {
                            ui.add_space(8.0);
                            egui::Frame::NONE
                                .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                                .show(ui, |ui| {
                                    ui.set_width(w);
                                    ui.spacing_mut().item_spacing = vec2(0.0, 2.0);
                                    for (i, e) in events.iter().enumerate() {
                                        ui.horizontal(|ui| {
                                            let mut v = e.draw;
                                            if ui
                                                .add_enabled(
                                                    enabled,
                                                    egui::Checkbox::new(&mut v, e.name.as_str()),
                                                )
                                                .clicked()
                                            {
                                                clicked = Some(Clicked::Draw(i, v));
                                            }
                                            ui.with_layout(
                                                egui::Layout::right_to_left(egui::Align::Center),
                                                |ui| {
                                                    let name = format!("{} colour", e.name);
                                                    let resp = ui
                                                        .add_enabled_ui(enabled, |ui| {
                                                            crate::dialogs::form::swatch_button(
                                                                ui, e.color, &name,
                                                            )
                                                        })
                                                        .inner;
                                                    if resp.clicked() {
                                                        clicked = Some(Clicked::Color(i));
                                                    }
                                                },
                                            );
                                        });
                                    }
                                });
                        })
                        .response;
                    let _ = resp;
                });
        });

        // The buttons: rows 1 and 2, two columns.
        let bw = (inner.width() - gap) / 2.0;
        let y1 = area.max.y + gap;
        let pairs = [
            ("Add event", "Delete event"),
            ("Mark frames", "Unmark frames"),
        ];
        for (i, (a, b)) in pairs.iter().enumerate() {
            let y = y1 + i as f32 * (row_h + gap);
            for (j, label) in [a, b].iter().enumerate() {
                let r = egui::Rect::from_min_size(
                    egui::pos2(inner.min.x + j as f32 * (bw + gap), y),
                    vec2(bw, row_h),
                );
                ui.scope_builder(egui::UiBuilder::new().max_rect(r), |ui| {
                    if super::points::push_button(ui, r.size(), label, enabled).clicked() {
                        clicked = Some(match (i, j) {
                            (0, 0) => Clicked::Add,
                            (0, _) => Clicked::Delete,
                            (_, 0) => Clicked::Mark,
                            _ => Clicked::Unmark,
                        });
                    }
                });
            }
        }
        ui.advance_cursor_after_rect(full);

        let (Some(t), Some(c)) = (trial, clicked) else {
            return;
        };
        let ask = |a: Ask| EditRequest::Ask(Box::new(a));
        match c {
            // E54: EventDockWidget::checkbox_clicked.
            Clicked::Draw(event, draw) => cx.request(EditRequest::Apply(Edit::SetEventDraw {
                trial: t,
                event,
                draw,
            })),
            Clicked::Color(event) => {
                let c = events[event].color;
                cx.request(ask(event_color_ask(
                    t,
                    event,
                    Rgb::new(c.r(), c.g(), c.b()),
                )));
            }
            Clicked::Add => cx.request(ask(add_event_ask(t, events.len()))),
            Clicked::Delete => {
                if let Some(a) =
                    delete_event_ask(t, events.iter().map(|e| e.name.clone()).collect())
                {
                    cx.request(ask(a));
                }
            }
            // on_pushButtonSet_clicked, on_pushButtonUnset_clicked: PlotWindow::setEventOn/Off.
            Clicked::Mark => cx.raise(Action::SetEventOn),
            Clicked::Unmark => cx.raise(Action::SetEventOff),
        }
    }
}
