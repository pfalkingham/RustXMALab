//! Small helpers for controls of XMALab 3's forms that egui has no exact equivalent for: a
//! spin box with up and down arrows, a combo box sized to its contents, a framed group
//! (`QFrame::StyledPanel`, drawn as egui's group frame). They
//! use egui's own look (colours, margins and sizes of the current style). Shared by the
//! bars and available to panels.

use egui::{Response, Ui};

/// `QFrame` with `StyledPanel` shape: egui's group frame around `add`.
pub fn styled_frame<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    egui::Frame::group(ui.style()).show(ui, add).inner
}

/// A `QSpinBox`: a number field with up and down arrows; returns whether the value
/// changed. `name` names it for screen readers and tests.
pub fn spin_box(ui: &mut Ui, value: &mut i64, min: i64, max: i64, width: f32, name: &str) -> bool {
    spin_box_step(ui, value, min, max, 1, width, name)
}

/// A `QSpinBox` with `singleStep` `step`: an arrow (and a drag of the field, which has no
/// Qt counterpart) moves the value by `step` per unit, within `min` to `max`; a typed value
/// is taken as typed. Returns whether the value changed. With `step` 1 this is
/// [`spin_box`].
pub fn spin_box_step(
    ui: &mut Ui,
    value: &mut i64,
    min: i64,
    max: i64,
    step: i64,
    width: f32,
    name: &str,
) -> bool {
    let before = *value;
    let h = ui.spacing().interact_size.y;
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        let field = if step == 1 {
            let mut v = *value;
            let field = ui
                .add_sized(
                    [width, h],
                    egui::DragValue::new(&mut v).range(min..=max).speed(0.2),
                )
                .accessible_name(name);
            *value = v;
            field
        } else {
            stepped_field(ui, value, min, max, step, [width, h], name)
        };
        if let Some(delta) = spin_arrows(ui, field.id, name) {
            *value = value
                .saturating_add(i64::from(delta) * step)
                .clamp(min, max);
        }
    });
    *value != before
}

/// The up and down arrows of a spin box, drawn in a 14-point column right of its field
/// (the field's height, as tall as `interact_size.y`): up in the top half, down in the
/// bottom half, named "<name> up" and "<name> down" for screen readers and tests. Returns
/// +1 when the up arrow was clicked, -1 for the down arrow. The arrows' ids are made from
/// `field_id`, the field's own (unique) id, not from the enclosing scope, which two spin
/// boxes in sibling scopes can share. Shared by [`spin_box_step`],
/// `edit_forms::double_spin_box` and the external calibration frame's boxes.
pub fn spin_arrows(ui: &mut Ui, field_id: egui::Id, name: &str) -> Option<i8> {
    let h = ui.spacing().interact_size.y;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, h), egui::Sense::hover());
    let up = egui::Rect::from_min_max(rect.min, egui::pos2(rect.max.x, rect.center().y));
    let down = egui::Rect::from_min_max(egui::pos2(rect.min.x, rect.center().y), rect.max);
    let mut step = None;
    for (r, delta, salt) in [(up, 1, "up"), (down, -1, "down")] {
        let resp = ui.interact(r, field_id.with(("spin", salt)), egui::Sense::click());
        resp.widget_info(|| {
            egui::WidgetInfo::labeled(egui::Role::Button, resp.enabled(), format!("{name} {salt}"))
        });
        let color = if ui.is_enabled() {
            ui.style().interact(&resp).fg_stroke.color
        } else {
            ui.visuals().weak_text_color()
        };
        let c = r.center();
        let s = 3.0;
        let pts = if delta > 0 {
            vec![
                egui::pos2(c.x - s, c.y + s * 0.6),
                egui::pos2(c.x + s, c.y + s * 0.6),
                egui::pos2(c.x, c.y - s * 0.6),
            ]
        } else {
            vec![
                egui::pos2(c.x - s, c.y - s * 0.6),
                egui::pos2(c.x + s, c.y - s * 0.6),
                egui::pos2(c.x, c.y + s * 0.6),
            ]
        };
        ui.painter()
            .add(egui::Shape::convex_polygon(pts, color, egui::Stroke::NONE));
        if resp.clicked() {
            step = Some(delta);
        }
    }
    step
}

/// The number field of [`spin_box_step`] with a step other than 1. It is the field of
/// [`spin_box`] (`DragValue::new` on an `i64`, written out so that its writes can be seen),
/// except while it is dragged: egui keeps the precise dragged value itself, starting from the
/// value it was given when the drag began, and writes that value rounded each frame. The
/// field's value is then the drag's start plus `step` times how far egui's value has moved
/// from it, so it moves with the drag, in multiples of `step` (review 12b). A typed value and
/// a key press are taken as written, as with step 1.
fn stepped_field(
    ui: &mut Ui,
    value: &mut i64,
    min: i64,
    max: i64,
    step: i64,
    size: [f32; 2],
    name: &str,
) -> Response {
    let before = *value;
    let mut current = before;
    let mut written = false;
    let field = ui
        .add_sized(
            size,
            egui::DragValue::from_get_set(|v: Option<f64>| {
                if let Some(v) = v {
                    // `Numeric::from_f64` for `i64`, as `DragValue::new` does.
                    current = v as i64;
                    written = true;
                }
                current as f64
            })
            .max_decimals(0)
            .range(min..=max)
            .speed(0.2),
        )
        .accessible_name(name);
    // The value egui's drag started from: egui forgets its precise value when a pointer
    // button is pressed or released and starts again from the value it is given, which is
    // this field's value before the first write after that.
    let start_id = field.id.with("spin drag start");
    let restart = ui.input(|i| i.pointer.any_pressed() || i.pointer.any_released());
    if restart || !field.dragged() {
        ui.data_mut(|d| d.remove::<i64>(start_id));
    }
    *value = if field.dragged() && written {
        let start = ui.data_mut(|d| *d.get_temp_mut_or_insert_with(start_id, || before));
        start
            .saturating_add(current.saturating_sub(start).saturating_mul(step))
            .clamp(min, max)
    } else {
        current
    };
    field
}

/// A `QComboBox` showing `items[*selected]`; returns the newly chosen index, if any.
/// `label` is the id of the label in front of it (its Qt buddy), which names it for
/// screen readers and tests.
pub fn combo_box(
    ui: &mut Ui,
    id: &str,
    selected: Option<usize>,
    items: &[&str],
    width: Option<f32>,
    label: Option<egui::Id>,
) -> Option<usize> {
    let text = selected.and_then(|i| items.get(i)).copied().unwrap_or("");
    let mut chosen = None;
    // With a label, the label's (unique) id is part of the salt, so two combo boxes given
    // the same name in sibling scopes do not share an id.
    let mut cb = match label {
        Some(l) => egui::ComboBox::from_id_salt((id, l)),
        None => egui::ComboBox::from_id_salt(id),
    }
    .selected_text(text);
    let w = width.unwrap_or_else(|| {
        // Qt's AdjustToContents: wide enough for the longest item.
        let font = egui::TextStyle::Button.resolve(ui.style());
        let longest = items
            .iter()
            .map(|s| {
                ui.painter()
                    .layout_no_wrap((*s).to_owned(), font.clone(), egui::Color32::BLACK)
                    .size()
                    .x
            })
            .fold(0.0_f32, f32::max);
        longest + 34.0
    });
    cb = cb.width(w);
    let r = cb.show_ui(ui, |ui| {
        for (i, it) in items.iter().enumerate() {
            if ui.selectable_label(Some(i) == selected, *it).clicked() && Some(i) != selected {
                chosen = Some(i);
            }
        }
    });
    let chosen = chosen.or_else(|| combo_wheel(ui, &r.response, selected, items.len(), |_| true));
    if let Some(l) = label {
        let _ = r.response.labelled_by(l);
    }
    chosen
}

// ---------------------------------------------------------------------------------------
// The scroll wheel over a closed drop-down.
//
// `QComboBox::wheelEvent` steps the current item by one per wheel event while the pointer
// is over the closed box; egui's `ComboBox` ignores the wheel. [`combo_wheel`] adds that to
// every drop-down of the program: each one calls it with the response of its box (the
// button egui draws for the closed list) after `ComboBox::show_ui`, and takes the index it
// returns exactly as an index picked from the list (the same action, the same undo step).
//
// - One step per notch. Wheel events are turned into notches (a `Line` unit is one notch,
//   a `Point` amount is `line_scroll_speed` points per notch, egui's own distance for one
//   notch, 40 points on the desktop; a `Page` is one notch) and added up per box in a
//   [`WheelAccum`]; a whole notch moves one entry. A mouse wheel's notch moves at once; a
//   touchpad's stream of small amounts moves one entry per 40 points of finger travel, the
//   distance one notch scrolls a page, so a two-finger swipe moves a few entries, not the
//   whole list. The remainder is kept for 0.6 s and dropped when the direction turns.
//   Momentum events (macOS, after the fingers are lifted) are taken but move nothing.
// - Up (away from the user) chooses the previous entry, down the next; the ends stop it (no
//   wrap-around); entries the caller says are disabled are skipped.
// - Left alone: horizontal scrolling, a wheel with Ctrl, Shift, Alt or Cmd held (the camera
//   views and the plot use those), a box that is disabled or not hovered (a modal dialog
//   over it, or a job blocking the window, leaves it unhovered or disabled), a box whose
//   list is open (the list scrolls), and a box inside a scroll area whose content is taller
//   than the area at that moment, unless the box has the focus (tabbed to, or clicked and
//   nothing else clicked or focused since, [`wheel_focus`]): there the wheel scrolls the
//   area. Qt gives the wheel to a box under the pointer
//   even in the middle of scrolling a page, which changes a setting by accident (parity
//   ledger U175).
// - When a box takes the wheel, the wheel is consumed, also at the first and last entry:
//   the frame's wheel events and smoothed scroll delta are cleared, and
//   [`combo_wheel_begin_frame`] (called first in every frame of every viewport: the main
//   window, each floating panel, each tool window and the Detection Settings window) clears
//   the rest of egui's smoothing of that notch in the frames that follow, so nothing behind
//   the box scrolls or zooms for it. The claim is kept per viewport.
// - The remembered click focus ([`WheelFocus`]) is one box in the whole program. It ends
//   with a pointer press that is not on that box (in any viewport), with another widget
//   taking the keyboard focus, and when the box is not drawn in a frame of its viewport (its
//   dialog closed, the workspace changed); a box drawn again later starts unfocused.
// ---------------------------------------------------------------------------------------

/// How long a partial notch (a touchpad's small amounts) is kept, in seconds.
const WHEEL_ACCUM_KEEP_S: f64 = 0.6;

/// Notches of wheel turn added up for one drop-down (see [`combo_wheel`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct WheelAccum {
    /// The part of a notch not yet used; positive is away from the user.
    pub notches: f32,
    /// The input time of the last amount added.
    pub last: f64,
}

impl WheelAccum {
    /// Adds `delta` notches (positive: away from the user) at time `now`; returns the whole
    /// notches made, positive away from the user. A remainder older than
    /// [`WHEEL_ACCUM_KEEP_S`], or of the other sign, is dropped first.
    pub(crate) fn add(&mut self, now: f64, delta: f32) -> i32 {
        if delta == 0.0 || !delta.is_finite() {
            return 0;
        }
        if now - self.last > WHEEL_ACCUM_KEEP_S
            || now < self.last
            || (self.notches != 0.0 && self.notches.signum() != delta.signum())
        {
            self.notches = 0.0;
        }
        self.last = now;
        self.notches += delta;
        // A little slack so that ten tenths make one notch despite rounding.
        let whole = (self.notches + self.notches.signum() * 1e-4).trunc();
        if whole != 0.0 {
            self.notches -= whole;
            // The slack can leave a speck of the other sign: no remainder then.
            if self.notches.signum() != whole.signum() {
                self.notches = 0.0;
            }
        }
        whole.clamp(-1000.0, 1000.0) as i32
    }
}

/// The entry `steps` entries after `selected` (before it when `steps` is negative) among
/// `len` entries, skipping those `enabled` refuses and stopping at the ends; `None` when that
/// is `selected` itself. A returned index is never `selected`: callers may assign it without
/// comparing (and [`combo_wheel`], which returns only what this returns, gives the same
/// guarantee). With nothing selected, a step forward chooses the first enabled
/// entry and a step back nothing (as `QComboBox::wheelEvent` from index -1).
pub(crate) fn wheel_target(
    selected: Option<usize>,
    len: usize,
    steps: i32,
    enabled: &dyn Fn(usize) -> bool,
) -> Option<usize> {
    let selected = selected.filter(|s| *s < len);
    let mut at = selected;
    for _ in 0..steps.unsigned_abs().min(len as u32) {
        let next = if steps > 0 {
            let from = at.map_or(0, |a| a + 1);
            (from..len).find(|&i| enabled(i))
        } else {
            match at {
                Some(a) => (0..a).rev().find(|&i| enabled(i)),
                None => None,
            }
        };
        match next {
            Some(n) => at = Some(n),
            None => break,
        }
    }
    at.filter(|a| Some(*a) != selected)
}

/// The id under which the time of the last wheel taken by a drop-down in viewport `vp` is
/// kept (one claim per viewport: each viewport has its own input and its own smoothing).
fn wheel_claim_id(vp: egui::ViewportId) -> egui::Id {
    egui::Id::unique("xma_combo_wheel_claim").with(vp)
}

/// The id under which the remembered click focus ([`WheelFocus`]) is kept.
fn wheel_focus_id() -> egui::Id {
    egui::Id::unique("xma_combo_wheel_focus")
}

/// The drop-down box that was clicked (or tabbed to) and keeps the focus for the wheel: one
/// box in the whole program (see [`wheel_focus`] and [`combo_wheel_begin_frame`]).
#[derive(Clone, Copy, Debug, PartialEq)]
struct WheelFocus {
    /// The box's id.
    id: egui::Id,
    /// The viewport it is drawn in.
    viewport: egui::ViewportId,
}

/// The notches (positive away from the user) of this frame's vertical wheel events, when
/// there are any and none is modified by Ctrl, Shift, Alt or Cmd. Momentum events count as
/// wheel to take but add no notch.
fn frame_notches(i: &egui::InputState, line: f32) -> Option<f32> {
    let held = |m: &egui::Modifiers| m.ctrl || m.shift || m.alt || m.mac_cmd || m.command;
    if held(&i.modifiers) {
        return None;
    }
    let line = line.max(1.0);
    let mut any = false;
    let mut notches = 0.0_f32;
    for e in &i.events {
        if let egui::Event::MouseWheel {
            unit,
            delta,
            modifiers,
            source,
            ..
        } = e
        {
            if held(modifiers) {
                return None;
            }
            if delta.y == 0.0 || !delta.y.is_finite() {
                continue;
            }
            any = true;
            if *source == egui::MouseWheelSource::Momentum {
                continue;
            }
            notches += match unit {
                egui::MouseWheelUnit::Line => delta.y,
                egui::MouseWheelUnit::Point => delta.y / line,
                egui::MouseWheelUnit::Page => delta.y.signum(),
            };
        }
    }
    any.then_some(notches)
}

/// Whether `ui` lies in a scroll area whose content was taller than the area in the last
/// frame (the content's own rectangle, which egui keeps for each `Ui`, against the height
/// of the area it is shown in).
fn in_overflowing_scroll_area(ui: &Ui) -> bool {
    ui.stack().iter().any(|s| {
        s.kind() == Some(egui::UiKind::ScrollArea)
            && ui.ctx().read_response(s.unique_id).is_some_and(|r| {
                r.rect.is_finite()
                    && s.max_rect.is_finite()
                    && r.rect.height() > s.max_rect.height() + 0.5
            })
    })
}

/// Whether the box `response` has the focus for the wheel: it has egui's keyboard focus
/// (tabbed to), or it was clicked (its list opened) and since then no pointer press landed
/// outside it and its list, and no other widget took the keyboard focus. egui gives the
/// keyboard focus only from the keyboard and drops it when an entry of the list is clicked,
/// so a click is remembered here, for one box at a time.
///
/// The memory is one [`WheelFocus`] for the whole program. A press or focus change seen
/// while the box is drawn ends it here; [`combo_wheel_begin_frame`] ends it for a press in
/// another viewport and a frame in which the box was not drawn.
fn wheel_focus(ui: &Ui, response: &Response) -> bool {
    let ctx = ui.ctx();
    let key = wheel_focus_id();
    let id = response.id;
    let mine = WheelFocus {
        id,
        viewport: ctx.viewport_id(),
    };
    if response.clicked() || response.has_focus() || egui::ComboBox::is_open(ctx, id) {
        ctx.data_mut(|d| d.insert_temp(key, mine));
        return true;
    }
    match ctx.data(|d| d.get_temp::<WheelFocus>(key)) {
        Some(f) if f.id == id && f.viewport == mine.viewport => {}
        _ => return false,
    }
    let pressed_elsewhere = ui.input(|i| i.pointer.any_pressed()) && !response.hovered();
    let other_focus = ctx.memory(|m| m.focused()).is_some_and(|f| f != id);
    if pressed_elsewhere || other_focus {
        ctx.data_mut(|d| d.remove::<WheelFocus>(key));
        return false;
    }
    true
}

/// The scroll wheel over the closed drop-down whose box is `response` (drawn in `ui`), with
/// `len` entries of which `selected` is shown and `enabled(i)` says whether entry `i` can
/// be chosen. Returns the entry the wheel chose, if it changed; the caller treats it as an
/// entry picked from the list. Consumes the wheel whenever the box takes it. See the
/// comment above [`WheelAccum`] for the rules.
pub fn combo_wheel(
    ui: &Ui,
    response: &Response,
    selected: Option<usize>,
    len: usize,
    enabled: impl Fn(usize) -> bool,
) -> Option<usize> {
    let ctx = ui.ctx();
    let focused = wheel_focus(ui, response);
    if !ui.is_enabled()
        || !response.enabled()
        || !response.hovered()
        || egui::ComboBox::is_open(ctx, response.id)
        || !ctx.memory(|m| m.allows_interaction(response.layer_id))
    {
        return None;
    }
    let line = ctx.options(|o| o.input_options.line_scroll_speed);
    let notches = ui.input(|i| frame_notches(i, line))?;
    if !focused && in_overflowing_scroll_area(ui) {
        return None;
    }
    let now = ui.input(|i| i.time);
    ctx.input_mut(|i| {
        let vertical_wheel =
            |e: &egui::Event| matches!(e, egui::Event::MouseWheel { delta, .. } if delta.y != 0.0);
        i.events.retain(|e| !vertical_wheel(e));
        i.raw.events.retain(|e| !vertical_wheel(e));
        i.smooth_scroll_delta = egui::Vec2::ZERO;
    });
    let claim = wheel_claim_id(ctx.viewport_id());
    ctx.data_mut(|d| d.insert_temp(claim, now));
    let acc_id = response.id.with("xma_combo_wheel_accum");
    let made = ctx.data_mut(|d| {
        d.get_temp_mut_or_default::<WheelAccum>(acc_id)
            .add(now, notches)
    });
    if made == 0 {
        return None;
    }
    // Away from the user (positive) is the previous entry.
    wheel_target(selected, len, -made, &enabled)
}

/// Called first in every frame of every viewport (the main window, each floating panel and
/// tool window, the Detection Settings window), with that viewport's context:
///
/// - while egui is still spreading a wheel notch that a drop-down of this viewport took over
///   the following frames (its smoothing), clears this frame's share so nothing behind the
///   box scrolls for it. A newer wheel event in this viewport ends this;
/// - ends the remembered click focus of a drop-down ([`wheel_focus`]) when a pointer button
///   is pressed in another viewport than the box's, and when the box was not drawn since
///   the last frame of its viewport (a press or another widget's focus in the box's own
///   viewport is seen by [`wheel_focus`] where the box is drawn).
pub fn combo_wheel_begin_frame(ctx: &egui::Context) {
    let vp = ctx.viewport_id();
    if let Some(t) = ctx.data(|d| d.get_temp::<f64>(wheel_claim_id(vp))) {
        let (now, since) = ctx.input(|i| (i.time, i.time_since_last_scroll()));
        // `time_since_last_scroll` is `(time - last_wheel_event) as f32`: equal when the last
        // wheel event is the one taken at `t`.
        if (now - t) as f32 == since {
            ctx.input_mut(|i| i.smooth_scroll_delta = egui::Vec2::ZERO);
        } else {
            ctx.data_mut(|d| d.remove::<f64>(wheel_claim_id(vp)));
        }
    }
    let key = wheel_focus_id();
    let Some(f) = ctx.data(|d| d.get_temp::<WheelFocus>(key)) else {
        return;
    };
    let keep = if f.viewport == vp {
        // Whether the box was drawn in the last frame: called first in the frame, egui's
        // `read_response` finds the widgets of the last pass. (Not `combo_wheel` having run
        // for it: a box whose list entry was clicked skips that call in that frame.) A press,
        // or another widget's keyboard focus, in this viewport is judged where the box is
        // drawn ([`wheel_focus`]: on the box or not; egui drops a vanished widget's focus
        // during the frame, so it is not judged here, before the frame).
        ctx.read_response(f.id).is_some()
    } else {
        !ctx.input(|i| i.pointer.any_pressed())
    };
    if !keep {
        ctx.data_mut(|d| d.remove::<WheelFocus>(key));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notches_accumulate_one_step_per_notch() {
        let mut a = WheelAccum::default();
        // A mouse notch moves at once, in either direction.
        assert_eq!(a.add(1.0, 1.0), 1);
        assert_eq!(a.add(1.1, -1.0), -1);
        assert_eq!(a.add(1.2, 3.0), 3);
        // A touchpad: ten amounts of 4.5 points (40 points a notch) make one step, with
        // the remainder kept.
        let mut a = WheelAccum::default();
        let made: i32 = (0..10)
            .map(|k| a.add(2.0 + 0.01 * k as f64, 4.5 / 40.0))
            .sum();
        assert_eq!(made, 1);
        assert!((a.notches - 0.125).abs() < 1e-4, "{}", a.notches);
        // Ten tenths make exactly one notch.
        let mut a = WheelAccum::default();
        let made: i32 = (0..10).map(|k| a.add(3.0 + 0.01 * k as f64, 0.1)).sum();
        assert_eq!(made, 1);
        // The remainder is dropped after a pause, and when the direction turns.
        let mut a = WheelAccum::default();
        assert_eq!(a.add(4.0, 0.9), 0);
        assert_eq!(a.add(5.0, 0.2), 0);
        assert_eq!(a.add(5.1, -0.9), 0);
        assert_eq!(a.add(5.2, -0.2), -1);
        // Nothing for zero, NaN or infinity; huge amounts are bounded.
        assert_eq!(a.add(6.0, 0.0), 0);
        assert_eq!(a.add(6.0, f32::NAN), 0);
        assert_eq!(a.add(6.0, f32::INFINITY), 0);
        assert_eq!(a.add(6.0, 1e9), 1000);
    }

    #[test]
    fn steps_clamp_at_the_ends_and_skip_disabled_entries() {
        let all = |_: usize| true;
        assert_eq!(wheel_target(Some(1), 4, 1, &all), Some(2));
        assert_eq!(wheel_target(Some(1), 4, -1, &all), Some(0));
        assert_eq!(wheel_target(Some(0), 4, -1, &all), None);
        assert_eq!(wheel_target(Some(3), 4, 1, &all), None);
        assert_eq!(wheel_target(Some(1), 4, 10, &all), Some(3));
        assert_eq!(wheel_target(Some(2), 4, -10, &all), Some(0));
        assert_eq!(wheel_target(None, 4, 1, &all), Some(0));
        assert_eq!(wheel_target(None, 4, -1, &all), None);
        assert_eq!(wheel_target(Some(0), 0, 1, &all), None);
        // An out-of-range selection counts as none.
        assert_eq!(wheel_target(Some(9), 4, 1, &all), Some(0));
        // Entry 2 is disabled (a separator, say) and is passed over; at the end there is no
        // enabled entry left, so nothing changes.
        let some = |i: usize| i != 2 && i != 4;
        assert_eq!(wheel_target(Some(1), 5, 1, &some), Some(3));
        assert_eq!(wheel_target(Some(3), 5, -1, &some), Some(1));
        assert_eq!(wheel_target(Some(3), 5, 1, &some), None);
        assert_eq!(wheel_target(None, 5, 1, &|i| i > 1), Some(2));
    }

    /// The guarantee the call sites in `dialogs/ask.rs` and `dialogs/rigid_body.rs` rely on:
    /// whatever the arguments, the entry returned is never the one already selected.
    #[test]
    fn the_target_is_never_the_selected_entry() {
        let filters: [&dyn Fn(usize) -> bool; 4] =
            [&|_| true, &|_| false, &|i| i % 2 == 0, &|i| i == 3];
        for len in 0..7 {
            for sel in (0..9).map(Some).chain([None]) {
                for steps in [-20, -3, -2, -1, 0, 1, 2, 3, 20, i32::MIN, i32::MAX] {
                    for en in filters {
                        if let Some(t) = wheel_target(sel, len, steps, en) {
                            assert_ne!(Some(t), sel, "{sel:?} {len} {steps}");
                            assert!(t < len);
                        }
                    }
                }
            }
        }
    }

    /// Runs one pass of `ctx` in the main viewport with `events`, calling only the frame hook.
    fn hook_pass(ctx: &egui::Context, time: f64, events: Vec<egui::Event>) {
        let input = egui::RawInput {
            time: Some(time),
            events,
            ..Default::default()
        };
        let mut out = ctx.run_ui(input, |ui| combo_wheel_begin_frame(ui.ctx()));
        out.textures_delta.clear();
    }

    fn press() -> Vec<egui::Event> {
        let pos = egui::pos2(5.0, 5.0);
        vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ]
    }

    /// The wheel claim is kept per viewport: the main window's hook neither clears nor
    /// forgets another viewport's claim, and forgets its own when a newer wheel is seen.
    #[test]
    fn the_wheel_claim_is_kept_per_viewport() {
        let ctx = egui::Context::default();
        let other = egui::ViewportId::from_hash_of("xma_test_other_viewport");
        assert_ne!(
            wheel_claim_id(other),
            wheel_claim_id(egui::ViewportId::ROOT)
        );
        ctx.data_mut(|d| {
            d.insert_temp(wheel_claim_id(other), 0.5_f64);
            d.insert_temp(wheel_claim_id(egui::ViewportId::ROOT), 0.5_f64);
        });
        hook_pass(&ctx, 1.0, Vec::new());
        hook_pass(&ctx, 1.1, Vec::new());
        // The root's claim is not the last wheel of the root (there was none): forgotten.
        assert_eq!(
            ctx.data(|d| d.get_temp::<f64>(wheel_claim_id(egui::ViewportId::ROOT))),
            None
        );
        // The other viewport's claim is untouched by the root's hook.
        assert_eq!(
            ctx.data(|d| d.get_temp::<f64>(wheel_claim_id(other))),
            Some(0.5)
        );
    }

    /// The remembered click focus of a box in another viewport ends with a press in this
    /// one, and only with a press (that box not being drawn here is no reason).
    #[test]
    fn a_press_in_another_viewport_ends_the_click_focus() {
        let ctx = egui::Context::default();
        let other = egui::ViewportId::from_hash_of("xma_test_other_viewport");
        let f = WheelFocus {
            id: egui::Id::unique("xma_test_box"),
            viewport: other,
        };
        ctx.data_mut(|d| d.insert_temp(wheel_focus_id(), f));
        hook_pass(&ctx, 1.0, Vec::new());
        hook_pass(&ctx, 1.1, Vec::new());
        assert_eq!(
            ctx.data(|d| d.get_temp::<WheelFocus>(wheel_focus_id())),
            Some(f)
        );
        hook_pass(&ctx, 1.2, press());
        assert_eq!(
            ctx.data(|d| d.get_temp::<WheelFocus>(wheel_focus_id())),
            None
        );
    }

    /// The click focus of a box in this viewport ends after a frame that did not draw it.
    #[test]
    fn a_frame_without_the_box_ends_the_click_focus() {
        let ctx = egui::Context::default();
        let f = WheelFocus {
            id: egui::Id::unique("xma_test_box"),
            viewport: egui::ViewportId::ROOT,
        };
        let draw = |time: f64, with_box: bool| {
            let input = egui::RawInput {
                time: Some(time),
                ..Default::default()
            };
            let mut out = ctx.run_ui(input, |ui| {
                combo_wheel_begin_frame(ui.ctx());
                if with_box {
                    let rect =
                        egui::Rect::from_min_size(egui::pos2(10.0, 10.0), egui::vec2(50.0, 20.0));
                    let _ = ui.interact(rect, f.id, egui::Sense::click());
                }
            });
            out.textures_delta.clear();
        };
        // The box is drawn, then clicked (remembered), then drawn again: kept.
        draw(1.0, true);
        ctx.data_mut(|d| d.insert_temp(wheel_focus_id(), f));
        draw(1.1, true);
        assert_eq!(
            ctx.data(|d| d.get_temp::<WheelFocus>(wheel_focus_id())),
            Some(f)
        );
        // A frame without the box; the hook of the frame after it ends the focus.
        draw(1.2, false);
        assert_eq!(
            ctx.data(|d| d.get_temp::<WheelFocus>(wheel_focus_id())),
            Some(f)
        );
        draw(1.3, true);
        assert_eq!(
            ctx.data(|d| d.get_temp::<WheelFocus>(wheel_focus_id())),
            None
        );
    }
}
