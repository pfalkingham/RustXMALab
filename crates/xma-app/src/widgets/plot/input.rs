//! XMALab's plot mouse and key bindings, as a state machine over egui's raw
//! input events.
//!
//! The C++ (`PlotWindow::eventFilter`, `ui/PlotWindow.cpp:933-1125`) intercepts
//! Qt events on the QCustomPlot widget and either handles them or reconfigures
//! QCustomPlot's own range drag and zoom before passing them on. Raw egui events
//! are processed in order here, with the modifiers each event carries, so the
//! same modifier tests apply. egui_plot's own interaction is not used at all.
//!
//! | Input | Effect |
//! |---|---|
//! | Wheel | zoom X about the pointer (factor 0.85 per notch) |
//! | Ctrl + wheel | zoom the left Y axis about the pointer |
//! | Alt + wheel | zoom the right Y axis about the pointer |
//! | Wheel with other modifiers (Shift, Meta) | zoom the axis the last wheel zoomed (QCustomPlot keeps its last setting) |
//! | Left press or drag | go to the frame under the pointer, if inside the trial |
//! | Ctrl + left press, then drag | start, then extend, a frame range; the frame follows |
//! | Alt + left drag | pan the axis the last right-drag panned (QCustomPlot's own drag) |
//! | Right drag | pan X (the content follows the pointer) |
//! | Ctrl + right drag | pan the left Y axis |
//! | Alt + right drag | pan the right Y axis |
//! | Right drag with other modifiers | nothing |
//! | Delete / Backspace with focus (not Shift) | report [`PlotEvent::DeleteRange`] |
//!
//! "Ctrl" is `Modifiers::command`, so Cmd on macOS, as Qt's `ControlModifier`.

use egui::{Event, Key, Modifiers, MouseWheelUnit, PointerButton, Pos2, Rect};

use super::PlotEvent;
use super::model::FrameAxis;
use super::ranging::{scale_range, set_range, wheel_factor};

/// Which range an interaction changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Target {
    /// The X axis.
    #[default]
    X,
    /// The left Y axis.
    Left,
    /// The right Y axis.
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Drag {
    button: PointerButton,
    target: Target,
    start_pos: Pos2,
    start_range: (f64, f64),
}

/// Interaction state kept between UI frames (inside [`super::PlotView`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Interaction {
    /// QCustomPlot's `rangeZoom` setting, changed by every wheel event with
    /// no modifier, Ctrl or Alt.
    zoom_target: Target,
    /// QCustomPlot's `rangeDrag` setting, changed by right presses.
    drag_target: Target,
    drag: Option<Drag>,
    left_down: bool,
    right_down: bool,
    pointer: Option<Pos2>,
    modifiers: Modifiers,
    last_frame: Option<i64>,
    last_select: Option<i64>,
}

/// The ranges the input changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Ranges {
    pub x: (f64, f64),
    pub left: (f64, f64),
    pub right: (f64, f64),
}

impl Ranges {
    fn get(&self, t: Target) -> (f64, f64) {
        match t {
            Target::X => self.x,
            Target::Left => self.left,
            Target::Right => self.right,
        }
    }

    fn set(&mut self, t: Target, r: (f64, f64)) {
        match t {
            Target::X => self.x = r,
            Target::Left => self.left = r,
            Target::Right => self.right = r,
        }
    }
}

/// What the input pass needs to know about the widget.
pub(crate) struct Geometry<'a> {
    /// The axis rect (where the ranges map to).
    pub axis_rect: Rect,
    /// Whether a press at this position lands on the widget (not covered).
    pub accepts: &'a dyn Fn(Pos2) -> bool,
    /// Whether the widget has keyboard focus.
    pub has_focus: bool,
    /// Points per wheel line for pixel-precise wheels (`InputOptions::line_scroll_speed`).
    pub points_per_line: f32,
}

/// Result of one input pass.
#[derive(Default)]
pub(crate) struct Outcome {
    pub events: Vec<PlotEvent>,
    pub changed: [bool; 3],
    pub pressed: bool,
    pub wheel_used: bool,
}

fn no_modifier(m: Modifiers) -> bool {
    !(m.alt || m.ctrl || m.shift || m.mac_cmd || m.command)
}

impl Interaction {
    /// Processes this frame's raw events in order.
    pub(crate) fn process(
        &mut self,
        events: &[Event],
        geo: &Geometry<'_>,
        frames: &FrameAxis,
        ranges: &mut Ranges,
    ) -> Outcome {
        let mut out = Outcome::default();
        let rect = geo.axis_rect;
        let before = *ranges;
        for ev in events {
            match ev {
                Event::ModifiersChanged(m) => self.modifiers = *m,
                Event::PointerMoved(p) => {
                    self.pointer = Some(*p);
                    self.on_move(*p, rect, frames, ranges, &mut out);
                }
                Event::PointerGone => self.pointer = None,
                Event::PointerButton {
                    pos,
                    button,
                    pressed,
                    modifiers,
                } => {
                    self.modifiers = *modifiers;
                    self.pointer = Some(*pos);
                    if *pressed {
                        if (geo.accepts)(*pos) {
                            self.on_press(
                                *pos, *button, *modifiers, rect, frames, ranges, &mut out,
                            );
                        }
                    } else {
                        self.on_release(*button);
                    }
                }
                Event::MouseWheel {
                    unit,
                    delta,
                    modifiers,
                    ..
                } => {
                    self.modifiers = *modifiers;
                    if let Some(p) = self.pointer
                        && (geo.accepts)(p)
                    {
                        out.wheel_used = true;
                        self.on_wheel(*unit, *delta, *modifiers, p, rect, geo, ranges);
                    }
                }
                Event::Key {
                    key: Key::Delete | Key::Backspace,
                    pressed: true,
                    modifiers,
                    ..
                } if geo.has_focus && !modifiers.shift => {
                    // Shift+Delete / Shift+Backspace are application shortcuts
                    // ("delete all above error"), checked first in the C++.
                    out.events.push(PlotEvent::DeleteRange);
                }
                _ => {}
            }
        }
        out.changed = [
            ranges.x != before.x,
            ranges.left != before.left,
            ranges.right != before.right,
        ];
        out
    }

    /// Forgets a gesture whose button is no longer down (released outside the window).
    pub(crate) fn sync_buttons(&mut self, primary_down: bool, secondary_down: bool) {
        if !primary_down && self.left_down {
            self.on_release(PointerButton::Primary);
        }
        if !secondary_down && self.right_down {
            self.on_release(PointerButton::Secondary);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn on_press(
        &mut self,
        pos: Pos2,
        button: PointerButton,
        m: Modifiers,
        rect: Rect,
        frames: &FrameAxis,
        ranges: &Ranges,
        out: &mut Outcome,
    ) {
        match button {
            // `buttons() == Qt::LeftButton`: only the left button down.
            PointerButton::Primary if !self.right_down => {
                out.pressed = true;
                self.left_down = true;
                self.last_frame = None;
                self.last_select = None;
                if m.alt {
                    // Passed to QCustomPlot: its own range drag with the last drag setting.
                    self.drag = Some(Drag {
                        button,
                        target: self.drag_target,
                        start_pos: pos,
                        start_range: ranges.get(self.drag_target),
                    });
                    return;
                }
                let frame = frames.frame_at(x_at(pos.x, rect, ranges.x));
                if m.command {
                    self.last_select = Some(frame);
                    out.events.push(PlotEvent::SelectionStart(frame));
                }
                self.emit_frame(frame, frames, out);
            }
            PointerButton::Secondary if !self.left_down => {
                out.pressed = true;
                self.right_down = true;
                let target = if no_modifier(m) {
                    Some(Target::X)
                } else if m.command {
                    Some(Target::Left)
                } else if m.alt {
                    Some(Target::Right)
                } else {
                    None
                };
                if let Some(t) = target {
                    self.drag_target = t;
                    self.drag = Some(Drag {
                        button,
                        target: t,
                        start_pos: pos,
                        start_range: ranges.get(t),
                    });
                }
            }
            _ => {}
        }
    }

    fn on_release(&mut self, button: PointerButton) {
        match button {
            PointerButton::Primary => self.left_down = false,
            PointerButton::Secondary => self.right_down = false,
            _ => {}
        }
        if self.drag.is_some_and(|d| d.button == button) {
            self.drag = None;
        }
    }

    fn on_move(
        &mut self,
        p: Pos2,
        rect: Rect,
        frames: &FrameAxis,
        ranges: &mut Ranges,
        out: &mut Outcome,
    ) {
        if self.right_down {
            if let Some(d) = self.drag.filter(|d| d.button == PointerButton::Secondary) {
                pan(d, p, rect, ranges);
            }
            return;
        }
        if !self.left_down {
            return;
        }
        let m = self.modifiers;
        if m.alt {
            if let Some(d) = self.drag.filter(|d| d.button == PointerButton::Primary) {
                pan(d, p, rect, ranges);
            }
            return;
        }
        let frame = frames.frame_at(x_at(p.x, rect, ranges.x));
        if m.command && self.last_select != Some(frame) {
            self.last_select = Some(frame);
            out.events.push(PlotEvent::SelectionEnd(frame));
        }
        self.emit_frame(frame, frames, out);
    }

    fn emit_frame(&mut self, frame: i64, frames: &FrameAxis, out: &mut Outcome) {
        if frames.contains(frame) && self.last_frame != Some(frame) {
            self.last_frame = Some(frame);
            out.events.push(PlotEvent::Frame(frame as usize));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn on_wheel(
        &mut self,
        unit: MouseWheelUnit,
        delta: egui::Vec2,
        m: Modifiers,
        p: Pos2,
        rect: Rect,
        geo: &Geometry<'_>,
        ranges: &mut Ranges,
    ) {
        if no_modifier(m) {
            self.zoom_target = Target::X;
        } else if m.command {
            self.zoom_target = Target::Left;
        } else if m.alt {
            self.zoom_target = Target::Right;
        }
        // Some platforms turn Shift+wheel or Alt+wheel into horizontal wheel events.
        let d = if delta.y != 0.0 { delta.y } else { delta.x };
        let steps = match unit {
            MouseWheelUnit::Line => d as f64,
            MouseWheelUnit::Point => d as f64 / geo.points_per_line.max(1.0) as f64,
            MouseWheelUnit::Page => d as f64 * 3.0,
        };
        if steps == 0.0 {
            return;
        }
        let factor = wheel_factor(steps);
        let t = self.zoom_target;
        let r = ranges.get(t);
        let center = match t {
            Target::X => x_at(p.x, rect, r),
            Target::Left | Target::Right => y_at(p.y, rect, r),
        };
        ranges.set(t, scale_range(r, factor, center));
    }
}

/// QCustomPlot's linear range drag: the range at the press, shifted by the
/// coordinate difference between the press and the pointer.
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPAxisRect::mouseMoveEvent.
fn pan(d: Drag, p: Pos2, rect: Rect, ranges: &mut Ranges) {
    let r = d.start_range;
    let new = match d.target {
        Target::X => {
            let diff = x_at(d.start_pos.x, rect, r) - x_at(p.x, rect, r);
            (r.0 + diff, r.1 + diff)
        }
        Target::Left | Target::Right => {
            let diff = y_at(d.start_pos.y, rect, r) - y_at(p.y, rect, r);
            (r.0 + diff, r.1 + diff)
        }
    };
    let cur = ranges.get(d.target);
    ranges.set(d.target, set_range(cur, new));
}

/// Plot x under screen x (`QCPAxis::pixelToCoord` for the bottom axis).
pub(crate) fn x_at(px: f32, rect: Rect, r: (f64, f64)) -> f64 {
    r.0 + (px - rect.left()) as f64 / rect.width().max(1e-6) as f64 * (r.1 - r.0)
}

/// Plot y under screen y for range `r` (`QCPAxis::pixelToCoord` for a vertical axis).
pub(crate) fn y_at(py: f32, rect: Rect, r: (f64, f64)) -> f64 {
    r.0 + (rect.bottom() - py) as f64 / rect.height().max(1e-6) as f64 * (r.1 - r.0)
}
