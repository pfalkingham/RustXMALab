//! Events of a trial (`docs/cpp-editing.md` section 7). Nothing is recomputed.

use std::sync::Arc;

use xma_model::{Event, Rgb};

use super::trials::check_trial;
use super::{EditError, EditOutcome, EditResult, FrameRange};
use crate::document::Document;

fn check_event(doc: &Document, trial: usize, event: usize) -> Result<(), EditError> {
    check_trial(doc, trial)?;
    let n = doc.project.trials[trial].events.len();
    if event >= n {
        return Err(EditError::Invalid(format!(
            "trial {trial} has no event {event} (of {n})"
        )));
    }
    Ok(())
}

fn event_mut(doc: &mut Document, trial: usize, event: usize) -> &mut Event {
    &mut Arc::make_mut(&mut doc.project.trials[trial]).events[event]
}

/// E49: a new event, red, not drawn, every frame off.
///
/// Ported from ui/EventDockWidget.cpp, EventDockWidget::on_pushButtonAdd_clicked
/// (`:164-185`) and core/Trial.cpp, Trial::addEvent (`:481-485`). An empty name adds
/// nothing in the C++ ([`EditError::Invalid`] here); a name already present is refused with
/// XMALab's message.
pub(super) fn add_event(doc: &mut Document, trial: usize, name: &str) -> EditResult {
    check_trial(doc, trial)?;
    if name.is_empty() {
        return Err(EditError::Invalid("an event needs a name".to_owned()));
    }
    let t = &doc.project.trials[trial];
    if t.events.iter().any(|e| e.name == name) {
        return Err(EditError::Rejected(format!(
            "Event with name {name} is already present. Please choose another name"
        )));
    }
    let n = t.n_frames;
    Arc::make_mut(&mut doc.project.trials[trial])
        .events
        .push(Event {
            name: name.to_owned(),
            color: Rgb::new(255, 0, 0),
            draw: false,
            data: vec![false; n],
        });
    Ok(EditOutcome::none())
}

/// E50.
///
/// Ported from ui/EventDockWidget.cpp, EventDockWidget::on_pushButtonDelete_clicked
/// (`:187-210`) and core/Trial.cpp, Trial::removeEvent (`:487-491`).
pub(super) fn remove_event(doc: &mut Document, trial: usize, event: usize) -> EditResult {
    check_event(doc, trial, event)?;
    Arc::make_mut(&mut doc.project.trials[trial])
        .events
        .remove(event);
    Ok(EditOutcome::none())
}

/// E51 (`on`) and E52: the frames of the range set or cleared in every event that is
/// drawn; events not drawn are untouched.
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::setEventOn (`:3119-3136`) and
/// PlotWindow::setEventOff (`:3138-3155`). The plot orders the two ends and clamps them to
/// the trial's start and end frames before calling; here the range must lie inside every
/// drawn event's frames, otherwise [`EditError::Invalid`]. `first > last` does nothing.
pub(super) fn set_event_frames(
    doc: &mut Document,
    trial: usize,
    frames: FrameRange,
    on: bool,
) -> EditResult {
    check_trial(doc, trial)?;
    if frames.first > frames.last {
        return Ok(EditOutcome::none());
    }
    let t = &doc.project.trials[trial];
    for e in t.events.iter().filter(|e| e.draw) {
        if frames.last >= e.data.len() {
            return Err(EditError::Invalid(format!(
                "frame {} is outside event {}'s {} frames",
                frames.last,
                e.name,
                e.data.len()
            )));
        }
    }
    let changes = t
        .events
        .iter()
        .any(|e| e.draw && e.data[frames.first..=frames.last].iter().any(|&d| d != on));
    if changes {
        for e in &mut Arc::make_mut(&mut doc.project.trials[trial]).events {
            if e.draw {
                for d in &mut e.data[frames.first..=frames.last] {
                    *d = on;
                }
            }
        }
    }
    Ok(EditOutcome::none())
}

/// E53.
///
/// Ported from ui/EventDockWidget.cpp, EventDockWidget::changeColor (`:222-239`):
/// `EventData::setColor` (the colour dialog's Cancel is the interface's business).
pub(super) fn set_event_color(
    doc: &mut Document,
    trial: usize,
    event: usize,
    color: Rgb,
) -> EditResult {
    check_event(doc, trial, event)?;
    if doc.project.trials[trial].events[event].color != color {
        event_mut(doc, trial, event).color = color;
    }
    Ok(EditOutcome::none())
}

/// E54.
///
/// Ported from ui/EventDockWidget.cpp, EventDockWidget::checkbox_clicked (`:241-250`):
/// `EventData::setDraw`.
pub(super) fn set_event_draw(
    doc: &mut Document,
    trial: usize,
    event: usize,
    draw: bool,
) -> EditResult {
    check_event(doc, trial, event)?;
    if doc.project.trials[trial].events[event].draw != draw {
        event_mut(doc, trial, event).draw = draw;
    }
    Ok(EditOutcome::none())
}
