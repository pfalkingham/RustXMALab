//! Floating panels as separate operating-system windows (egui immediate viewports).
//!
//! Ported from the docking spike (`spikes/docking/common/shell.rs`, `floating_windows`),
//! with its two geometry fixes:
//!
//! - X11 (Mutter/XWayland) often reports a window's outer rectangle equal to its inner one
//!   (frame unknown), so a restored window would creep down by one title bar per run: the
//!   largest frame offset seen on any window is remembered ([`outer_pos`]).
//! - Wayland/GNOME gives child windows an inner size smaller than requested by the
//!   client-side decoration: the difference is measured in the first frames and added back
//!   when the size is saved.
//!
//! Each window has one slim row (see [`super::title`]): a grip where window positions are
//! known (dragging it onto the main window docks the panel at the nearest allowed edge), the
//! dimmed title (the operating system's title bar already shows it), and a dock button;
//! double-clicking the row docks. Closing the window closes a closable panel and docks the
//! others (Qt shows no close button on them). Under `egui_kittest` (embedded viewports) the
//! windows are drawn as windows inside the main window.

use std::collections::{BTreeMap, BTreeSet};

use egui::{Pos2, Rect, Vec2, ViewportBuilder, ViewportId};

use super::title::title_row;
use super::{Shell, ShellDraw, WinGeom};
use crate::actions::Action;
use crate::panels::{PanelId, Side};

/// Bookkeeping for the floating windows.
#[derive(Debug, Default)]
pub(crate) struct FloatState {
    /// Geometry handed to the window builder when the window was (re)created.
    initial: BTreeMap<PanelId, WinGeom>,
    /// Windows that existed last frame.
    live: BTreeSet<PanelId>,
    /// Frames each window has been shown.
    frames: BTreeMap<PanelId, u32>,
    /// Requested minus actual inner size (Wayland quirk).
    size_fix: BTreeMap<PanelId, Vec2>,
    /// Largest window-frame offset seen (X11 quirk).
    pub(crate) deco: Vec2,
    /// A floating window is being dragged by its grip.
    dock_drag: Option<(PanelId, Option<Pos2>)>,
    /// Part of the windows' viewport ids; View > Reset layout moves it, so that every
    /// floating window is made anew (a window that exists keeps its position).
    pub(crate) generation: u64,
}

impl FloatState {
    pub(crate) fn new(floating: &BTreeMap<PanelId, WinGeom>) -> Self {
        Self {
            initial: floating.clone(),
            ..Self::default()
        }
    }

    pub(crate) fn opened(&mut self, p: PanelId, g: WinGeom) {
        self.initial.insert(p, g);
        self.size_fix.remove(&p);
        self.frames.remove(&p);
    }
}

/// The outer position to save for a window (see the module documentation).
pub(crate) fn outer_pos(
    deco: &mut Vec2,
    inner: Option<Rect>,
    outer: Option<Rect>,
) -> Option<[f32; 2]> {
    let (i, o) = (inner?, outer?);
    let d = i.min - o.min;
    if d.x >= deco.x && d.y >= deco.y {
        *deco = d;
    }
    let p = i.min - *deco;
    Some([p.x, p.y])
}

/// When a floating panel is dragged over the main window: the edge it would dock to (within
/// a quarter of the window of an edge, allowed sides only).
fn drop_side(content: Rect, p: Pos2, panel: PanelId) -> Option<Side> {
    if !content.contains(p) {
        return None;
    }
    let fx = (p.x - content.min.x) / content.width();
    let fy = (p.y - content.min.y) / content.height();
    let mut cands = vec![
        (Side::Left, fx),
        (Side::Right, 1.0 - fx),
        (Side::Top, fy),
        (Side::Bottom, 1.0 - fy),
    ];
    cands.retain(|(s, d)| *d < 0.25 && panel.allowed_sides().contains(s));
    cands
        .into_iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|c| c.0)
}

fn side_band(c: Rect, s: Side) -> Rect {
    match s {
        Side::Left => Rect::from_min_max(c.min, Pos2::new(c.min.x + c.width() * 0.2, c.max.y)),
        Side::Right => Rect::from_min_max(Pos2::new(c.max.x - c.width() * 0.2, c.min.y), c.max),
        Side::Top => Rect::from_min_max(c.min, Pos2::new(c.max.x, c.min.y + c.height() * 0.2)),
        Side::Bottom => Rect::from_min_max(Pos2::new(c.min.x, c.max.y - c.height() * 0.2), c.max),
    }
}

/// The viewport id of a panel's floating window. `generation` is the shell's count of
/// View > Reset layout (0 until then).
#[must_use]
pub fn viewport_id(p: PanelId, generation: u64) -> ViewportId {
    if generation == 0 {
        ViewportId::from_hash_of(("xma_float", p.slug()))
    } else {
        ViewportId::from_hash_of(("xma_float", p.slug(), generation))
    }
}

/// Draws every shown floating panel in its own window.
pub(super) fn show_windows(
    shell: &mut Shell,
    ctx: &egui::Context,
    draw: &mut ShellDraw<'_, '_>,
    main_inner: Option<Rect>,
) {
    let embedded = ctx.embed_viewports();
    let floating: Vec<PanelId> = PanelId::ALL
        .into_iter()
        .filter(|p| shell.is_shown(*p) && shell.floating.contains_key(p))
        .collect();
    // A window (re)appearing, for example after a workspace change, opens at its last
    // geometry.
    for p in &floating {
        if !shell.float.live.contains(p)
            && let Some(g) = shell.floating.get(p).copied()
        {
            shell.float.opened(*p, g);
        }
    }
    shell.float.live = floating.iter().copied().collect();
    let locked = shell.locked;
    let (shortcuts, busy_actions) = (draw.shortcuts, draw.busy_actions);
    let mut pending: Vec<Action> = Vec::new();
    for p in floating {
        let init = shell.float.initial.get(&p).copied().unwrap_or(WinGeom {
            pos: None,
            size: [320.0, 260.0],
        });
        let title = {
            let cx = draw.parts.cx(true);
            draw.panels
                .get_mut(p)
                .dynamic_title(&cx)
                .unwrap_or_else(|| p.title().to_owned())
        };
        let mut builder = ViewportBuilder::default()
            .with_title(title.clone())
            .with_app_id(crate::APP_ID)
            .with_inner_size(init.size)
            // Qt's floating docks are tool windows.
            .with_window_type(egui::X11WindowType::Utility)
            .with_minimize_button(false)
            .with_close_button(p.closable());
        if let Some(pos) = init.pos {
            builder = builder.with_position(pos);
        }
        let disabled = p != PanelId::Progress
            && (draw.busy_window || (draw.busy_cancelable && p.disabled_by_cancelable_job()));
        let failures = &mut shell.failures;
        let size_fix = &mut shell.float.size_fix;
        let frames = &mut shell.float.frames;
        let deco = &mut shell.float.deco;
        let frame_rec = &mut shell.frame;
        let mut geom = None;
        let mut close = false;
        let mut drag: Option<(bool, Option<Pos2>)> = None;
        ctx.show_viewport_immediate(
            viewport_id(p, shell.float.generation),
            builder,
            |ui, _class| {
                let vctx = ui.ctx().clone();
                // This viewport's share of `qt::combo_wheel`'s frame hook.
                crate::qt::combo_wheel_begin_frame(&vctx);
                // The frame arrows pressed while this window has the keyboard. (Drawn
                // inside the main window, its keys are the main window's, taken already.)
                if shortcuts && !embedded {
                    let acx = crate::actions::ActionContext {
                        state: &*draw.parts.state,
                        facts: draw.parts.doc.map(|_| draw.parts.facts),
                        settings: &*draw.parts.settings,
                        busy: busy_actions,
                    };
                    pending.extend(crate::actions::collect_frame_arrows(&vctx, &acx));
                }
                let grip = vctx.input(|i| i.viewport().inner_rect.is_some());
                let resp = egui::CentralPanel::default()
                    .frame(egui::Frame::NONE.fill(ui.visuals().panel_fill))
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing = Vec2::ZERO;
                        let row = title_row(ui, p, &title, true, locked, grip, &mut pending);
                        let rect = ui.available_rect_before_wrap();
                        let mut child = ui.new_child(
                            egui::UiBuilder::new()
                                .max_rect(rect)
                                .id_salt(("float_body", p.slug()))
                                .layout(egui::Layout::top_down(egui::Align::Min)),
                        );
                        child.spacing_mut().item_spacing = vctx.global_style().spacing.item_spacing;
                        if disabled {
                            child.disable();
                        }
                        if let Some(msg) = failures.get(&Some(p)) {
                            super::failed_body(&mut child, p.title(), msg);
                        } else if rect.width() < super::MIN_BODY || rect.height() < super::MIN_BODY
                        {
                            // A window shrunk to nothing: nothing to draw.
                        } else {
                            let result = super::guard(|| {
                                let mut cx = draw.parts.cx(true);
                                draw.panels.get_mut(p).ui(&mut child, &mut cx);
                            });
                            if let Err(msg) = result {
                                draw.parts.log.error(format!(
                                    "The {} panel stopped after an error: {msg}",
                                    p.title()
                                ));
                                failures.insert(Some(p), msg);
                            }
                        }
                        ui.allocate_rect(rect, egui::Sense::hover());
                        row.response
                    })
                    .inner;
                frame_rec.drawn.insert(p);
                let info = vctx.input(|i| i.viewport().clone());
                if !embedded {
                    let s = vctx.content_rect().size();
                    let n = frames.entry(p).or_insert(0);
                    *n += 1;
                    let fix = size_fix.entry(p).or_insert(Vec2::ZERO);
                    if *n < 60 {
                        let d = Vec2::from(init.size) - s;
                        if d.x >= 0.0 && d.y >= 0.0 && d.x < 80.0 && d.y < 80.0 {
                            *fix = fix.max(d);
                        }
                    }
                    let fix = *fix;
                    geom = Some(WinGeom {
                        pos: outer_pos(deco, info.inner_rect, info.outer_rect),
                        size: [s.x + fix.x, s.y + fix.y],
                    });
                }
                close = info.close_requested();
                if let Some(r) = resp
                    && !locked
                    && (r.dragged() || r.drag_stopped())
                {
                    let global = match (info.inner_rect, vctx.input(|i| i.pointer.latest_pos())) {
                        (Some(ir), Some(lp)) => Some(ir.min + lp.to_vec2()),
                        _ => None,
                    };
                    drag = Some((r.drag_stopped(), global));
                }
                // This window's clicked links (the External calibration page's), for
                // `crate::links`.
                crate::links::collect(&vctx);
            },
        );
        if let (Some(g), Some(s)) = (geom, shell.floating.get_mut(&p)) {
            *s = g;
        }
        if close {
            // A closable panel closes; the others dock again (Qt gives them no close button;
            // winit can hide it only on some platforms).
            pending.push(if p.closable() {
                Action::ClosePanel(p)
            } else {
                Action::DockPanel(p)
            });
        }
        match drag {
            Some((false, global)) => {
                shell.float.dock_drag = Some((p, global));
                ctx.request_repaint_of(ViewportId::ROOT);
            }
            Some((true, global)) => {
                shell.float.dock_drag = None;
                if let (Some(g), Some(m)) = (global, main_inner) {
                    let content = Rect::from_min_size(Pos2::ZERO, m.size());
                    if let Some(side) = drop_side(content, g - m.min.to_vec2(), p) {
                        shell.dock_at(p, side);
                    }
                }
                ctx.request_repaint_of(ViewportId::ROOT);
            }
            None => {
                if shell.float.dock_drag.is_some_and(|(d, _)| d == p) {
                    shell.float.dock_drag = None;
                }
            }
        }
    }
    // Highlight the target edge while a floating panel is dragged over the main window.
    if let Some((p, Some(g))) = shell.float.dock_drag
        && let Some(m) = main_inner
    {
        let content = ctx.content_rect();
        if let Some(side) = drop_side(content, g - m.min.to_vec2(), p) {
            ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::unique("dockhint"),
            ))
            .rect(
                side_band(content, side),
                ctx.global_style()
                    .visuals
                    .widgets
                    .noninteractive
                    .corner_radius,
                super::tiles::drop_fill(&ctx.global_style().visuals),
                super::tiles::drop_stroke(&ctx.global_style().visuals),
                egui::StrokeKind::Inside,
            );
        }
    }
    for a in pending {
        draw.parts.actions.push(a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_side_respects_allowed_sides() {
        let c = Rect::from_min_size(Pos2::ZERO, egui::vec2(1000.0, 800.0));
        assert_eq!(
            drop_side(c, egui::pos2(10.0, 400.0), PanelId::Points),
            Some(Side::Left)
        );
        assert_eq!(
            drop_side(c, egui::pos2(10.0, 400.0), PanelId::Console),
            None
        );
        assert_eq!(
            drop_side(c, egui::pos2(500.0, 790.0), PanelId::Console),
            Some(Side::Bottom)
        );
        assert_eq!(
            drop_side(c, egui::pos2(500.0, 400.0), PanelId::Points),
            None
        );
    }
}
