//! The single slim title row of a panel, and its small icon buttons.
//!
//! Ported from the docking spike's `compact` module (`spikes/docking/common/shell.rs`). A
//! docked panel's row has the title on the left and, on the right, the float button and
//! (for closable panels) the close button, as Qt's dock title bar. A floating window's row
//! has a grip (where the platform reports window positions), the dimmed title, and the dock
//! button. The row is a drag handle; double-clicking it floats or docks the panel. With the
//! layout locked the row stays but has no buttons and does not react.

use egui::{Color32, Id, Pos2, Rect, Response, Sense, Stroke, Ui, Vec2};

use crate::actions::Action;
use crate::panels::PanelId;

/// Height of a panel's title row and of a tab strip: `egui_tiles`' default tab bar height.
pub const TITLE_H: f32 = 24.0;

/// Side of the square icon buttons in title rows: egui's icon width (check boxes, radio
/// buttons) in the current style.
fn button_side(ui: &Ui) -> f32 {
    ui.spacing().icon_width + 2.0
}

/// Gap between icon buttons.
const BTN_GAP: f32 = 2.0;
/// Horizontal padding at both ends of a title row.
const PAD_X: f32 = 5.0;
/// Stroke width of the painted icons.
const ICON_STROKE: f32 = 1.2;

/// The painted icons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TitleIcon {
    Float,
    Dock,
    Close,
    Grip,
}

/// Paints an icon centred in `rect` with lines only (no font glyphs).
pub fn paint_icon(ui: &Ui, rect: Rect, icon: TitleIcon, color: Color32) {
    let p = ui.painter();
    let st = Stroke::new(ICON_STROKE, color);
    let r = rect.shrink(rect.width() * 0.22);
    match icon {
        TitleIcon::Close => {
            p.line_segment([r.left_top(), r.right_bottom()], st);
            p.line_segment([r.right_top(), r.left_bottom()], st);
        }
        TitleIcon::Float => {
            // Two overlapping windows, as Qt's float button.
            let a = Rect::from_min_max(
                r.min + Vec2::new(r.width() * 0.3, 0.0),
                r.max - Vec2::new(0.0, r.height() * 0.3),
            );
            let b = Rect::from_min_max(
                r.min + Vec2::new(0.0, r.height() * 0.3),
                r.max - Vec2::new(r.width() * 0.3, 0.0),
            );
            p.rect_stroke(a, 0.0, st, egui::StrokeKind::Inside);
            p.rect_filled(b, 0.0, ui.visuals().panel_fill);
            p.rect_stroke(b, 0.0, st, egui::StrokeKind::Inside);
        }
        TitleIcon::Dock => {
            p.rect_stroke(r, 0.0, st, egui::StrokeKind::Inside);
            let bottom = Rect::from_min_max(Pos2::new(r.min.x, r.center().y), r.max);
            p.rect_filled(bottom, 0.0, color);
        }
        TitleIcon::Grip => {
            for i in 0..3 {
                for j in 0..2 {
                    let c = Pos2::new(
                        r.center().x + (j as f32 - 0.5) * r.width() * 0.45,
                        r.min.y + r.height() * (0.15 + 0.35 * i as f32),
                    );
                    p.circle_filled(c, 1.1, color);
                }
            }
        }
    }
}

/// A small square icon button at `rect`, with a tooltip.
pub fn icon_button(ui: &mut Ui, rect: Rect, id: Id, icon: TitleIcon, tip: &str) -> Response {
    let resp = ui.interact(rect, id, Sense::click()).on_hover_text(tip);
    let v = ui.style().interact(&resp);
    if resp.hovered() {
        ui.painter().rect_filled(rect, 2.0, v.bg_fill);
    }
    let color = if resp.hovered() {
        v.fg_stroke.color
    } else {
        ui.visuals().weak_text_color()
    };
    paint_icon(ui, rect, icon, color);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, ui.is_enabled(), tip));
    resp
}

/// An icon button allocated in a horizontal layout (tab strips, right to left).
pub fn icon_button_alloc(ui: &mut Ui, id: Id, icon: TitleIcon, tip: &str) -> Response {
    let side = button_side(ui);
    let h = ui.available_height().min(TITLE_H);
    let (slot, _) = ui.allocate_exact_size(Vec2::new(side + BTN_GAP, h), Sense::hover());
    let rect = Rect::from_center_size(slot.center(), Vec2::splat(side));
    icon_button(ui, rect, id, icon, tip)
}

/// What a title row reports.
#[derive(Debug)]
pub struct TitleRow {
    pub rect: Rect,
    /// The row started being dragged (docked: hand to the docking crate).
    pub drag_started: bool,
    /// The row's response (floating windows read drags from it).
    pub response: Option<Response>,
}

/// Draws a panel's title row at the top of `ui` and pushes float, dock and close requests
/// onto `actions`.
#[allow(clippy::too_many_arguments)]
pub fn title_row(
    ui: &mut Ui,
    panel: PanelId,
    title: &str,
    floating: bool,
    locked: bool,
    grip: bool,
    actions: &mut Vec<Action>,
) -> TitleRow {
    let side = button_side(ui);
    let sense = if locked {
        Sense::hover()
    } else {
        Sense::click_and_drag()
    };
    let (rect, r) = ui.allocate_exact_size(Vec2::new(ui.available_width(), TITLE_H), sense);
    let mut x = rect.left() + PAD_X;
    let grip = grip && floating && !locked;
    if grip {
        let g = Rect::from_center_size(
            Pos2::new(x + side / 2.0, rect.center().y),
            Vec2::splat(side),
        );
        paint_icon(ui, g, TitleIcon::Grip, ui.visuals().weak_text_color());
        x += side + BTN_GAP;
    }
    let text_color = if floating {
        ui.visuals().weak_text_color()
    } else {
        ui.visuals().text_color()
    };
    let galley = ui.painter().layout_no_wrap(
        title.to_owned(),
        egui::TextStyle::Body.resolve(ui.style()),
        text_color,
    );
    let text_pos = Pos2::new(x, rect.center().y - galley.size().y / 2.0);
    ui.painter()
        .with_clip_rect(rect)
        .galley(text_pos, galley, text_color);
    r.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Label, ui.is_enabled(), title));
    let mut out = TitleRow {
        rect,
        drag_started: false,
        response: None,
    };
    if !locked {
        let mut bx = rect.right() - PAD_X / 2.0 - side;
        let mut button = |ui: &mut Ui, icon: TitleIcon, tip: &str| {
            let b = Rect::from_min_size(
                Pos2::new(bx, rect.center().y - side / 2.0),
                Vec2::splat(side),
            );
            bx -= side + BTN_GAP;
            let id = Id::unique(("title_btn", panel.slug(), tip, floating));
            icon_button(ui, b, id, icon, tip).clicked()
        };
        if panel.closable() && button(ui, TitleIcon::Close, "Close") {
            actions.push(Action::ClosePanel(panel));
        }
        let (icon, tip, act) = if floating {
            (TitleIcon::Dock, "Dock", Action::DockPanel(panel))
        } else {
            (TitleIcon::Float, "Float", Action::FloatPanel(panel))
        };
        if button(ui, icon, tip) {
            actions.push(act);
        }
        if r.double_clicked() {
            actions.push(if floating {
                Action::DockPanel(panel)
            } else {
                Action::FloatPanel(panel)
            });
        } else if !floating && r.drag_started() {
            out.drag_started = true;
        }
        if r.hovered() && (!floating || grip) {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        }
    }
    out.response = Some(r);
    out
}
