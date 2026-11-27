//! The `egui_tiles` behaviour: how panes, title rows and tab strips are drawn.
//!
//! Ported from the docking spike's egui_tiles adapter (`spikes/docking/tiles_demo/src/main.rs`).

use std::cell::RefCell;
use std::collections::HashSet;

use egui_tiles::{Behavior, TileId, Tiles, UiResponse};

use super::title::{TITLE_H, TitleIcon, icon_button_alloc, title_row};
use super::{DockPane, Shell, ShellDraw, arrangement};
use crate::actions::Action;
use crate::panels::PanelId;

struct Beh<'a, 'b, 'p> {
    draw: &'a mut ShellDraw<'b, 'p>,
    frame: &'a mut super::ShellFrame,
    locked: bool,
    central: Option<TileId>,
    central_tabs: Option<TileId>,
    invisible: HashSet<TileId>,
    /// Panes whose tab group shows a tab strip this frame (filled by `is_tab_bar_visible`,
    /// which egui_tiles calls just before drawing the group's panes).
    stacked: RefCell<HashSet<TileId>>,
    pending: Vec<Action>,
    failures: &'a mut super::Failures,
    /// A dragged panel was let go this frame and the docking crate moved it.
    dropped: bool,
}

impl Beh<'_, '_, '_> {
    fn disabled(&self, p: PanelId) -> bool {
        if p == PanelId::Progress {
            return false;
        }
        self.draw.busy_window || (self.draw.busy_cancelable && p.disabled_by_cancelable_job())
    }
}

impl Behavior<DockPane> for Beh<'_, '_, '_> {
    fn pane_ui(&mut self, ui: &mut egui::Ui, tile_id: TileId, pane: &mut DockPane) -> UiResponse {
        let id = match *pane {
            DockPane::Central => {
                if self.draw.busy_window || self.draw.busy_cancelable {
                    ui.disable();
                }
                if let Some(msg) = self.failures.get(&None) {
                    super::failed_body(ui, "The camera views", msg);
                    return UiResponse::None;
                }
                let draw = &mut *self.draw;
                if let Err(msg) = super::guard(|| (draw.central)(ui, draw.parts)) {
                    draw.parts
                        .log
                        .error(format!("The camera views stopped after an error: {msg}"));
                    self.failures.insert(None, msg);
                }
                return UiResponse::None;
            }
            DockPane::Panel(id) => id,
        };
        ui.spacing_mut().item_spacing = egui::Vec2::ZERO;
        let stacked = self.stacked.borrow().contains(&tile_id);
        let mut drag = false;
        if !stacked {
            let title = {
                let cx = self.draw.parts.cx(false);
                self.draw
                    .panels
                    .get_mut(id)
                    .dynamic_title(&cx)
                    .unwrap_or_else(|| id.title().to_owned())
            };
            let row = title_row(ui, id, &title, false, self.locked, false, &mut self.pending);
            self.frame.header_rects.insert(id, row.rect);
            drag = row.drag_started;
        }
        let rect = ui.available_rect_before_wrap();
        self.frame.body_rects.insert(id, rect);
        self.frame.drawn.insert(id);
        let mut child = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(rect)
                .id_salt(("panel_body", id.slug()))
                .layout(egui::Layout::top_down(egui::Align::Min)),
        );
        child.set_clip_rect(rect.intersect(ui.clip_rect()));
        child.spacing_mut().item_spacing = ui.ctx().global_style().spacing.item_spacing;
        if self.disabled(id) {
            child.disable();
        }
        if let Some(msg) = self.failures.get(&Some(id)) {
            super::failed_body(&mut child, id.title(), msg);
        } else if rect.width() < super::MIN_BODY || rect.height() < super::MIN_BODY {
            // No room (a very small window): the panel is not asked to draw into nothing.
        } else {
            let draw = &mut *self.draw;
            let result = super::guard(|| {
                let mut cx = draw.parts.cx(false);
                draw.panels.get_mut(id).ui(&mut child, &mut cx);
            });
            if let Err(msg) = result {
                draw.parts.log.error(format!(
                    "The {} panel stopped after an error: {msg}",
                    id.title()
                ));
                self.failures.insert(Some(id), msg);
            }
        }
        ui.allocate_rect(rect, egui::Sense::hover());
        if drag {
            UiResponse::DragStarted
        } else {
            UiResponse::None
        }
    }

    fn tab_title_for_pane(&mut self, pane: &DockPane) -> egui::WidgetText {
        match pane {
            DockPane::Central => "".into(),
            DockPane::Panel(p) => p.title().into(),
        }
    }

    fn is_tab_closable(&self, _tiles: &Tiles<DockPane>, _tile_id: TileId) -> bool {
        // The close button sits at the right end of the strip instead.
        false
    }

    fn top_bar_right_ui(
        &mut self,
        tiles: &Tiles<DockPane>,
        ui: &mut egui::Ui,
        tile_id: TileId,
        tabs: &egui_tiles::Tabs,
        _scroll_offset: &mut f32,
    ) {
        if self.locked {
            return;
        }
        let Some(DockPane::Panel(p)) = tabs.active.and_then(|a| tiles.get_pane(&a)).copied() else {
            return;
        };
        // Added right to left: close is rightmost.
        if p.closable()
            && icon_button_alloc(
                ui,
                egui::Id::unique(("tab_close", tile_id)),
                TitleIcon::Close,
                "Close",
            )
            .clicked()
        {
            self.pending.push(Action::ClosePanel(p));
        }
        if icon_button_alloc(
            ui,
            egui::Id::unique(("tab_float", tile_id)),
            TitleIcon::Float,
            "Float",
        )
        .clicked()
        {
            self.pending.push(Action::FloatPanel(p));
        }
    }

    fn tab_bar_height(&self, _style: &egui::Style) -> f32 {
        TITLE_H
    }

    fn on_tab_button(
        &mut self,
        tiles: &mut Tiles<DockPane>,
        tile_id: TileId,
        resp: egui::Response,
    ) -> egui::Response {
        if let Some(DockPane::Panel(p)) = tiles.get_pane(&tile_id) {
            self.frame.tab_rects.insert(*p, resp.rect);
            if resp.double_clicked() && !self.locked {
                self.pending.push(Action::FloatPanel(*p));
            }
        }
        resp
    }

    fn is_tile_draggable(&self, _tiles: &Tiles<DockPane>, tile_id: TileId) -> bool {
        !self.locked && Some(tile_id) != self.central && Some(tile_id) != self.central_tabs
    }

    fn is_container_resizable(&self, _tiles: &Tiles<DockPane>, _tile_id: TileId) -> bool {
        !self.locked
    }

    fn is_tab_bar_visible(&self, tile_id: TileId, tabs: &egui_tiles::Tabs) -> bool {
        if Some(tile_id) == self.central_tabs {
            return false;
        }
        let visible = tabs
            .children
            .iter()
            .filter(|c| !self.invisible.contains(c))
            .count();
        let mut st = self.stacked.borrow_mut();
        for c in &tabs.children {
            if visible > 1 {
                st.insert(*c);
            } else {
                st.remove(c);
            }
        }
        visible > 1
    }

    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        simplification()
    }

    fn on_edit(&mut self, edit: egui_tiles::EditAction) {
        if edit == egui_tiles::EditAction::TileDropped {
            self.dropped = true;
        }
    }

    fn min_size(&self) -> f32 {
        super::MIN_PANEL
    }

    fn gap_width(&self, _style: &egui::Style) -> f32 {
        GAP
    }

    // The tab strip, tabs and resize lines are egui_tiles' own drawing.

    fn drag_preview_stroke(&self, visuals: &egui::Visuals) -> egui::Stroke {
        drop_stroke(visuals)
    }

    fn drag_preview_color(&self, visuals: &egui::Visuals) -> egui::Color32 {
        drop_fill(visuals)
    }

    fn dragged_overlay_color(&self, visuals: &egui::Visuals) -> egui::Color32 {
        visuals.panel_fill.gamma_multiply(0.6)
    }
}

/// How the docking crate tidies the tree before each frame.
///
/// Its defaults remove a tab group that a move left empty and a container left with
/// nothing or with one child: an empty one keeps its share of the space and shows as a
/// blank panel. A hidden panel's pane stays in its group (hidden is not removed), so its
/// place is kept.
pub(super) fn simplification() -> egui_tiles::SimplificationOptions {
    egui_tiles::SimplificationOptions {
        all_panes_must_have_tabs: true,
        // Dropping onto a pane can wrap it in a tab group inside the existing one.
        flatten_tabs_in_tabs: true,
        ..Default::default()
    }
}

/// Width of the gap between docked panels: `egui_tiles`' default (the resize handle is
/// wider, egui's `resize_grab_radius_side`).
pub const GAP: f32 = 1.0;

/// Fill of the area a dragged panel will land in: egui's selection colour, half
/// transparent, so the panels under it stay visible in either theme.
#[must_use]
pub fn drop_fill(v: &egui::Visuals) -> egui::Color32 {
    v.selection.bg_fill.gamma_multiply(0.5)
}

/// Outline of the area a dragged panel will land in: the selection's text colour, which
/// contrasts with the selection colour in both of egui's themes.
#[must_use]
pub fn drop_stroke(v: &egui::Visuals) -> egui::Stroke {
    egui::Stroke::new(2.0, v.selection.stroke.color)
}

/// Draws the docking tree into `ui`.
pub(super) fn show(shell: &mut Shell, ui: &mut egui::Ui, draw: &mut ShellDraw<'_, '_>) {
    let invisible = shell
        .tree
        .tiles
        .tile_ids()
        .filter(|t| !shell.tree.tiles.is_visible(*t))
        .collect();
    let central = shell.tree.tiles.find_pane(&DockPane::Central);
    let central_tabs = central.and_then(|c| shell.tree.tiles.parent_of(c));
    let mut frame = std::mem::take(&mut shell.frame);
    let mut failures = std::mem::take(&mut shell.failures);
    // The tree as it is while a panel is dragged, to put back if the drop moves nothing.
    let before_drop = shell
        .tree
        .dragged_id(ui.ctx())
        .is_some()
        .then(|| shell.tree.clone());
    let (pending, dropped) = {
        let mut beh = Beh {
            draw,
            frame: &mut frame,
            locked: shell.locked,
            central,
            central_tabs,
            invisible,
            stacked: RefCell::new(HashSet::new()),
            pending: Vec::new(),
            failures: &mut failures,
            dropped: false,
        };
        shell.tree.ui(&mut beh, ui);
        (beh.pending, beh.dropped)
    };
    // A panel let go where it already is (over a half of itself, or beside itself) is not
    // a move. The docking crate carries it out all the same: it takes the pane out of its
    // tab group and puts it in a new container next to that group, which loses the sizes
    // and the places of hidden panels in the group. The tree goes back to what it was.
    if dropped
        && let Some(before) = before_drop
        && arrangement::shown(&before) == arrangement::shown(&shell.tree)
    {
        shell.tree = before;
    }
    shell.frame = frame;
    shell.failures = failures;
    for a in pending {
        draw.parts.actions.push(a);
    }
}
