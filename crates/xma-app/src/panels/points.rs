//! The Points panel.
//!
//! Ported from `PointsDockWidget.ui`, ui/PointsDockWidget.cpp, ui/MarkerTreeWidget.cpp and
//! ui/MarkerTreeWidgetButton.cpp (ui-map.md, "PointsDockWidget", "Points tree", "Context
//! menus").
//!
//! The form, top to bottom (grid `gridLayout`, margin and spacing 0):
//!
//! - `treeWidgetPoints` (row 0): the tree of rigid bodies and markers. Its rows come from
//!   [`xma_engine::view::tree`]. Columns: index text (`RB<n>` or `<n>`, resized to
//!   contents), description (stretched), one status-colour swatch per camera when "Show
//!   marker status" is on, and a settings button (marker: `settings.png`; body: the icon of
//!   its reference state). Alternating row colours; extended, row-wise selection.
//! - `checkBoxShowMarkerStatus` (row 2), "Show marker status": the `ShowMarkerStates`
//!   setting.
//! - `checkBoxIsFromMaster` (row 3), "from default": shown, checked, only for a trial that is
//!   a copy of the default trial.
//! - `pushButtonApply` (row 5), "Apply to trials": shown only for the default trial.
//! - `pushButtonSetNumberMarkers`, `pushButtonSetNumberRigidBodies`, `pushButtonImportExport`
//!   (row 6): side by side when the panel is at least 450 wide, otherwise one per row
//!   (`PointsDockWidget::resizeEvent`).
//!
//! Selection (`on_treeWidgetPoints_currentItemChanged` and Qt's `ExtendedSelection`): a
//! click makes the row's marker or body the active one (clearing the other) and selects only
//! it; Ctrl (Command on macOS) toggles the row in the selection; Shift selects the shown rows
//! from the last plain or Ctrl click; a right click makes the row current and, if it is not
//! selected, selects only it, then opens the context menu. The keys are the application
//! shortcuts (Up and Down select the previous and next marker by number, Ctrl+A selects all;
//! `Shortcuts::eventFilter` takes them before the tree), handled in `app.rs`. When the
//! active marker changes from elsewhere (keys, camera views, plot), its body is expanded and
//! the row scrolled into view, as `QTreeView::scrollTo` does for `setCurrentItem`.
//!
//! Expanded bodies are remembered per trial and body (`RigidBody::setExpanded`, which the
//! C++ keeps in the body but never saves), here in the panel.
//!
//! Editing (`docs/cpp-editing.md` section 4): the context-menu entries, the drop of marker
//! rows, the three buttons, "from default", "Apply to trials" and a marker row's settings
//! button raise `Action::Points` (their prompts and edits are in `dialogs::points`); a
//! body row's settings button raises `Action::RigidBodySettings` (the rigid-body dialog).
//! In a trial linked to the Default trial dragging, "Import/Export" and both "Set Number"
//! buttons are disabled (`reloadListFromObject`).
//!
//! Drag and drop (`MarkerTreeWidget::dropEvent`, internal move): a marker row can be
//! dragged (a press on an unselected row selects it first, as in Qt); body rows cannot.
//! Released over a body row, or over a marker row inside a body, the selected markers move
//! into that body; over empty space or a free marker row they leave their bodies; released
//! outside the tree, nothing happens.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use egui::{Align2, Color32, Rect, Role, Sense, Stroke, Ui, WidgetInfo, pos2, vec2};
use xma_engine::view::tree::{
    BodyIcon, TreeItem, TreeRow, first_body_of, marker_statuses_iter, tree_rows,
    visible_row_indices,
};

use super::{AppEvent, Cx, Panel, PanelId};
use crate::actions::{Action, ActionContext};
use crate::dialogs::points::PointsAction;
use crate::icons::Icon;
use crate::settings::UiSettings;
use crate::state::{AppState, Changes, Workspace};

/// Row height: the settings tool button in the last column sets it (23 points in the
/// reference screenshot).
pub const ROW_H: f32 = 23.0;
/// Qt's `indentation` (20) for each level, the first for the root decoration.
const INDENT: f32 = 20.0;
/// Text margin of a Qt item view cell.
const CELL_MARGIN: f32 = 4.0;
/// Width of a status-colour column (an 8x16 swatch in a 16x16 icon cell, with margins).
const STATUS_COL_W: f32 = 22.0;
/// Width of the settings-button column.
const BUTTON_COL_W: f32 = 22.0;
/// Margin around the panel's contents.
const MARGIN: f32 = 9.0;
/// Panel width below which the three buttons are stacked (`PointsDockWidget::resizeEvent`).
const STACK_BELOW: f32 = 450.0;

/// A row's identity in the tree: what it shows and the body it sits under.
type RowKey = (TreeItem, Option<usize>);

/// The Points panel.
#[derive(Debug, Default)]
pub struct PointsPanel {
    /// Expanded bodies, by (trial, body). Absent means collapsed, the C++ initial state.
    expanded: HashMap<(usize, usize), bool>,
    /// Where a Shift click starts from (Qt's selection anchor).
    anchor: Option<RowKey>,
    /// Scroll the active row into view on the next draw.
    scroll_to_active: bool,
    /// Marker rows are being dragged.
    dragging: bool,
    /// The trials, and their numbers of markers and bodies, that `expanded` and `anchor`
    /// refer to; `None` until a project is drawn. Qt keeps the expanded flag on the body
    /// (`RigidBody::isExpanded`) and the anchor on a tree item that a reload drops; with
    /// rows and flags kept by index, an edit that renumbers trials, markers or bodies must
    /// move or drop them (review 12).
    shape: Option<Vec<TrialShape>>,
    /// The active trial's rows and the width of column 0, kept between repaints.
    rows: Option<RowsCache>,
}

/// What the tree draws that changes only with the trial, the expanded bodies or the font:
/// [`tree_rows`], the shown rows and the width of column 0 (Qt's `ResizeToContents`).
#[derive(Debug)]
struct RowsCache {
    /// The trial the rows were made from. While this `Weak` is alive the trial cannot
    /// change in place (`Arc::get_mut` fails and `Arc::make_mut` moves it to a new
    /// allocation), so a live pointer equal to the trial drawn means the same contents.
    trial: Weak<xma_model::Trial>,
    /// `Document::changes` when the rows were made.
    changes: u64,
    rows: Vec<TreeRow>,
    /// The expanded flag of every body of the trial when `shown` was made.
    expanded: Vec<bool>,
    /// The shown rows: indices into `rows`, and their keys.
    shown: Vec<usize>,
    shown_keys: Vec<RowKey>,
    /// The font and scale `col0` was measured in.
    font: egui::FontId,
    pixels_per_point: f32,
    col0: f32,
}

impl RowsCache {
    fn new(trial: &Arc<xma_model::Trial>, changes: u64) -> Self {
        Self {
            trial: Arc::downgrade(trial),
            changes,
            rows: tree_rows(trial),
            expanded: Vec::new(),
            shown: Vec::new(),
            shown_keys: Vec::new(),
            font: egui::FontId::default(),
            pixels_per_point: f32::NAN,
            col0: 0.0,
        }
    }

    fn is_for(&self, trial: &Arc<xma_model::Trial>, changes: u64) -> bool {
        self.changes == changes
            && self.trial.strong_count() > 0
            && std::ptr::eq(self.trial.as_ptr(), Arc::as_ptr(trial))
    }

    /// Brings the shown rows and column 0 up to date with the expanded flags (`expanded(b)`
    /// for each of the trial's `n_bodies` bodies) and the font.
    fn update_shown(
        &mut self,
        ui: &Ui,
        n_bodies: usize,
        expanded: impl Fn(usize) -> bool,
        font: &egui::FontId,
    ) {
        let ppp = ui.ctx().pixels_per_point();
        let same = self.expanded.len() == n_bodies
            && self
                .expanded
                .iter()
                .enumerate()
                .all(|(b, &v)| v == expanded(b))
            && self.font == *font
            && self.pixels_per_point.to_bits() == ppp.to_bits();
        if same {
            return;
        }
        self.expanded = (0..n_bodies).map(&expanded).collect();
        self.shown = visible_row_indices(&self.rows, &expanded);
        self.shown_keys = self
            .shown
            .iter()
            .map(|&i| (self.rows[i].item, self.rows[i].parent))
            .collect();
        // Column 0: ResizeToContents over the shown rows.
        self.col0 = self
            .shown
            .iter()
            .map(|&i| {
                let r = &self.rows[i];
                let w = ui
                    .painter()
                    .layout_no_wrap(r.id_text.clone(), font.clone(), Color32::BLACK)
                    .size()
                    .x;
                INDENT * (r.depth() as f32 + 1.0) + CELL_MARGIN + w + CELL_MARGIN
            })
            .fold(0.0_f32, f32::max);
        self.font = font.clone();
        self.pixels_per_point = ppp;
    }
}

/// The status colours of one repaint, each parsed from its setting the first time it is
/// needed ([`crate::camera_view::overlay_tracking::status_color`]).
struct StatusColorTable<'a> {
    settings: &'a UiSettings,
    /// By `(code + 30) / 5` (codes -30 to 65 in steps of 5).
    colors: [Option<Color32>; 20],
}

impl StatusColorTable<'_> {
    fn get(&mut self, status: xma_model::MarkerStatus) -> Color32 {
        let i = usize::try_from((status.code() + 30) / 5).unwrap_or(0);
        match self.colors.get_mut(i) {
            Some(Some(c)) => *c,
            Some(slot) => {
                let c = crate::camera_view::overlay_tracking::status_color(self.settings, status);
                *slot = Some(c);
                c
            }
            None => crate::camera_view::overlay_tracking::status_color(self.settings, status),
        }
    }
}

/// What [`PointsPanel::shape`] keeps of a trial.
#[derive(Debug)]
struct TrialShape {
    /// The trial, to find it again after trials before it are deleted. A `Weak` keeps the
    /// allocation, so the address is not reused by another trial.
    trial: Weak<xma_model::Trial>,
    markers: usize,
    bodies: usize,
}

fn shape_of(doc: &xma_engine::Document) -> Vec<TrialShape> {
    doc.project()
        .trials
        .iter()
        .map(|t| TrialShape {
            trial: Arc::downgrade(t),
            markers: t.markers.len(),
            bodies: t.rigid_bodies.len(),
        })
        .collect()
}

/// What a click on a row does to the selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClickKind {
    /// Plain click: select only this row.
    Plain,
    /// Ctrl (Command) click: toggle this row.
    Toggle,
    /// Shift click: select the shown rows from the anchor.
    Range,
    /// Ctrl+Shift click: add the shown rows from the anchor.
    AddRange,
    /// Right click: make current; select only it unless it is selected.
    Context,
}

impl PointsPanel {
    /// True when body `b` of `trial` is expanded.
    #[must_use]
    pub fn is_expanded(&self, trial: usize, b: usize) -> bool {
        self.expanded.get(&(trial, b)).copied().unwrap_or(false)
    }

    /// Expands or collapses a body (`MarkerTreeWidget::onItemExpanded`, `onItemCollapsed`).
    pub fn set_expanded(&mut self, trial: usize, b: usize, value: bool) {
        self.expanded.insert((trial, b), value);
    }

    /// After an edit, an undo or a redo: moves the expanded flags to their trials' new
    /// numbers (trials are only ever removed or appended, never edited, by an edit that
    /// changes their number) and drops those of trials and bodies that are gone; forgets the
    /// Shift-click anchor when the active trial's markers or bodies were added or removed,
    /// which renumbers rows. Edits that keep the numbers of markers and bodies renumber
    /// nothing (no edit reorders them), so the anchor and flags stay.
    pub fn sync_document(&mut self, doc: Option<&xma_engine::Document>, active: Option<usize>) {
        let Some(doc) = doc else {
            self.shape = None;
            return;
        };
        let new = shape_of(doc);
        let Some(old) = self.shape.replace(new) else {
            return;
        };
        let new = self.shape.as_deref().unwrap_or_default();
        let trial_of = |i: usize| -> Option<usize> {
            if old.len() == new.len() {
                Some(i)
            } else {
                let o = old.get(i)?;
                new.iter()
                    .position(|n| Weak::ptr_eq(&n.trial, &o.trial) && o.trial.strong_count() > 0)
            }
        };
        let expanded = std::mem::take(&mut self.expanded);
        for ((t, b), v) in expanded {
            if let Some(nt) = trial_of(t)
                && new.get(nt).is_some_and(|n| b < n.bodies)
            {
                self.expanded.insert((nt, b), v);
            }
        }
        let same_rows = old.len() == new.len()
            && active.is_some_and(|t| match (old.get(t), new.get(t)) {
                (Some(o), Some(n)) => o.markers == n.markers && o.bodies == n.bodies,
                _ => false,
            });
        if !same_rows {
            self.anchor = None;
        }
    }

    /// Applies a click on a shown row to the state: the active marker or body and the
    /// selection. `shown` is the list of shown rows (for Shift ranges).
    ///
    /// Ported from ui/PointsDockWidget.cpp, on_treeWidgetPoints_currentItemChanged, with the
    /// selection rules of Qt's `ExtendedSelection` (`QAbstractItemView::mousePressEvent`).
    pub fn click_row(
        &mut self,
        state: &mut AppState,
        shown: &[RowKey],
        row: RowKey,
        kind: ClickKind,
    ) -> Changes {
        let item = row.0;
        let range = |anchor: Option<RowKey>| -> Vec<TreeItem> {
            let to = shown.iter().position(|r| *r == row);
            let from = anchor.and_then(|a| shown.iter().position(|r| *r == a));
            match (from, to) {
                (Some(a), Some(b)) => shown[a.min(b)..=a.max(b)].iter().map(|r| r.0).collect(),
                _ => vec![item],
            }
        };
        let sel = &mut state.selection;
        match kind {
            ClickKind::Plain => {
                sel.markers.clear();
                sel.bodies.clear();
                insert(sel, item);
                self.anchor = Some(row);
            }
            ClickKind::Toggle => {
                if is_selected(sel, item) {
                    remove(sel, item);
                } else {
                    insert(sel, item);
                }
                self.anchor = Some(row);
            }
            ClickKind::Range | ClickKind::AddRange => {
                let items = range(self.anchor);
                if kind == ClickKind::Range {
                    sel.markers.clear();
                    sel.bodies.clear();
                }
                for it in items {
                    insert(sel, it);
                }
            }
            ClickKind::Context => {
                if !is_selected(sel, item) {
                    sel.markers.clear();
                    sel.bodies.clear();
                    insert(sel, item);
                    self.anchor = Some(row);
                }
            }
        }
        set_active(state, item);
        Changes {
            selection: true,
            ..Changes::NONE
        }
    }

    /// After the active marker changed from elsewhere: expand the body that holds it and
    /// scroll to it (`QTreeView::scrollTo` on `setCurrentItem`, which expands the parents).
    fn follow_active(&mut self, cx: &Cx<'_>) {
        let (Some(doc), Some(t)) = (cx.doc, cx.state.active_trial) else {
            return;
        };
        let Some(trial) = doc.project().trials.get(t) else {
            return;
        };
        let Some(view) = cx.state.trial_view() else {
            return;
        };
        if let Some(m) = view.active_marker {
            let rows = tree_rows(trial);
            if let Some(b) = first_body_of(&rows, m) {
                self.set_expanded(t, b, true);
            }
        }
        self.scroll_to_active = true;
    }
}

fn insert(sel: &mut crate::state::Selection, item: TreeItem) {
    match item {
        TreeItem::Marker(m) => {
            sel.markers.insert(m);
        }
        TreeItem::Body(b) => {
            sel.bodies.insert(b);
        }
    }
}

fn remove(sel: &mut crate::state::Selection, item: TreeItem) {
    match item {
        TreeItem::Marker(m) => {
            sel.markers.remove(&m);
        }
        TreeItem::Body(b) => {
            sel.bodies.remove(&b);
        }
    }
}

fn is_selected(sel: &crate::state::Selection, item: TreeItem) -> bool {
    match item {
        TreeItem::Marker(m) => sel.markers.contains(&m),
        TreeItem::Body(b) => sel.bodies.contains(&b),
    }
}

/// The current item becomes the active marker or body; the other is cleared
/// (`on_treeWidgetPoints_currentItemChanged`).
fn set_active(state: &mut AppState, item: TreeItem) {
    if let Some(v) = state.trial_view_mut() {
        match item {
            TreeItem::Marker(m) => {
                v.active_marker = Some(m);
                v.active_body = None;
            }
            TreeItem::Body(b) => {
                v.active_body = Some(b);
                v.active_marker = None;
            }
        }
    }
}

fn body_icon(i: BodyIcon) -> Icon {
    match i {
        BodyIcon::NotSet => Icon::Shape3dNotSet,
        BodyIcon::Set => Icon::Shape3d,
        BodyIcon::SetOptimized => Icon::Shape3dOptimized,
        BodyIcon::SetMarker => Icon::Shape3dSetMarker,
        BodyIcon::SetMarkerOptimized => Icon::Shape3dSetMarkerOptimized,
    }
}

/// Width of a text in a text style of `ui`'s style.
fn text_width(ui: &Ui, text: &str, style: egui::TextStyle) -> f32 {
    ui.painter()
        .layout_no_wrap(
            text.to_owned(),
            style.resolve(ui.style()),
            Color32::PLACEHOLDER,
        )
        .size()
        .x
}

/// The width the panel needs so that the check boxes and buttons below the tree show their
/// whole text in `ui`'s style (the buttons stacked, as below [`STACK_BELOW`]): the widest
/// of them plus the panel's margins.
#[must_use]
pub fn content_width(ui: &Ui) -> f32 {
    let sp = ui.spacing();
    let check = sp.icon_width + sp.icon_spacing;
    let button = 2.0 * sp.button_padding.x;
    let checks = ["Show marker status", "from default"]
        .map(|t| check + text_width(ui, t, egui::TextStyle::Button));
    let buttons = [
        "Apply to trials",
        "Set Number Markers",
        "Set Number Rigid Bodies",
        "Import/Export",
    ]
    .map(|t| button + text_width(ui, t, egui::TextStyle::Button));
    let widest = checks.into_iter().chain(buttons).fold(0.0_f32, f32::max);
    (widest + 2.0 * MARGIN).ceil() + 1.0
}

/// A `QPushButton` of a given size: egui's button, its text shortened with "…" when the
/// button is narrower than it (the full text is in the tooltip).
pub(super) fn push_button(
    ui: &mut Ui,
    size: egui::Vec2,
    text: &str,
    enabled: bool,
) -> egui::Response {
    ui.add_enabled_ui(enabled, |ui| {
        ui.add_sized(size, egui::Button::new(text).truncate())
    })
    .inner
    .on_hover_text(text)
}

/// The context-menu entries shown for a row: their text.
#[must_use]
pub fn context_entries(item: TreeItem, is_default: bool, is_copy: bool) -> Vec<&'static str> {
    context_actions(item, is_default, is_copy)
        .into_iter()
        .map(|(t, _)| t)
        .collect()
}

/// The context-menu entries shown for a row, with the action each raises. The two exports
/// raise `Action::Export3DPointsSelected` and `ExportTransformationsSelected`: File >
/// Export's flows (`MainWindow::save3DPoints`, `saveRigidBodies`) for the selected markers
/// or bodies.
///
/// Ported from ui/MarkerTreeWidget.cpp, MarkerTreeWidget::showContextMenu (entries shown by
/// the trial being the default trial or a copy of it).
#[must_use]
pub fn context_actions(
    item: TreeItem,
    is_default: bool,
    is_copy: bool,
) -> Vec<(&'static str, Action)> {
    let p = Action::Points;
    let mut e = Vec::new();
    match item {
        TreeItem::Marker(m) => {
            if !is_copy {
                e.push((
                    "Change Description",
                    p(PointsAction::ChangeDescription(item)),
                ));
                e.push((
                    "Create Rigid Body from selected",
                    p(PointsAction::CreateRigidBody),
                ));
                e.push((
                    "Add selected to Rigid Body",
                    p(PointsAction::AddToRigidBody),
                ));
            }
            if !is_default {
                e.push((
                    "Delete Data from selected points",
                    p(PointsAction::ResetPoints),
                ));
            }
            if !is_copy && !is_default {
                e.push(("Delete selected points", p(PointsAction::DeletePoints)));
            }
            if !is_copy {
                e.push((
                    "Change Detectionmethod of selected Points",
                    p(PointsAction::ChangeDetectionMethod),
                ));
            }
            if !is_default {
                e.push((
                    "Refine center of selected Points by polynomial fit",
                    p(PointsAction::RefinePoints),
                ));
                e.push((
                    "Change tracked data with data from another point",
                    p(PointsAction::ChangePoint { marker: m }),
                ));
                e.push((
                    "Save the 3D data of selected points",
                    Action::Export3DPointsSelected,
                ));
            }
        }
        TreeItem::Body(_) => {
            if !is_copy {
                e.push((
                    "Change Description",
                    p(PointsAction::ChangeDescription(item)),
                ));
            }
            if !is_default {
                e.push((
                    "Save the transformation of selected rigid bodies",
                    Action::ExportTransformationsSelected,
                ));
            }
        }
    }
    e
}

/// Whether an action is enabled now (the menu's rule).
fn enabled(cx: &Cx<'_>, a: &Action) -> bool {
    let acx = ActionContext {
        state: cx.state,
        facts: Some(cx.facts),
        settings: cx.settings,
        busy: cx.jobs.blocks_window(),
    };
    a.enabled(&acx)
}

impl Panel for PointsPanel {
    fn id(&self) -> PanelId {
        PanelId::Points
    }

    fn on_event(&mut self, event: &AppEvent, cx: &mut Cx<'_>) {
        match event {
            AppEvent::ProjectOpened | AppEvent::ProjectClosed => {
                self.expanded.clear();
                self.anchor = None;
                self.shape = None;
                self.rows = None;
            }
            AppEvent::Changed(ch) => {
                if ch.document {
                    self.sync_document(cx.doc, cx.state.active_trial);
                }
                if ch.active_trial {
                    // PointsDockWidget::activeTrialChanged -> reloadListFromObject: select the
                    // trial's active marker or body again, or forget it if it is gone.
                    self.anchor = None;
                    let doc = cx.doc;
                    cx.change(|s| reselect_active(s, doc));
                }
                if ch.selection || ch.active_trial {
                    self.follow_active(cx);
                }
            }
        }
    }

    fn min_width(&self, ui: &Ui, _cx: &Cx<'_>) -> Option<f32> {
        Some(content_width(ui))
    }

    fn ui(&mut self, ui: &mut Ui, cx: &mut Cx<'_>) {
        let doc = cx.doc;
        if self.shape.is_none()
            && let Some(d) = doc
        {
            self.shape = Some(shape_of(d));
        }
        let t = cx.state.active_trial;
        let trial = match (doc, t) {
            (Some(d), Some(t)) => d.project().trials.get(t),
            _ => None,
        };
        let is_default = trial.is_some_and(|tr| tr.is_default);
        let is_copy = trial.is_some_and(|tr| tr.is_copy_from_default);

        // gridLayout_2: Qt's default dock content margin (9) around the scroll area.
        let outer = ui.available_rect_before_wrap();
        let full = outer.shrink2(vec2(MARGIN, MARGIN / 2.0));
        let stacked = full.width() < STACK_BELOW;
        let row_h = ui.spacing().interact_size.y;
        let gap = 1.0;
        let n_rows =
            1 + usize::from(is_copy) + usize::from(is_default) + if stacked { 3 } else { 1 };
        let controls_h = n_rows as f32 * (row_h + gap);
        let tree_rect = Rect::from_min_max(
            full.min,
            pos2(full.max.x, (full.max.y - controls_h).max(full.min.y + 20.0)),
        );

        // -- treeWidgetPoints ---------------------------------------------------------------
        ui.scope_builder(egui::UiBuilder::new().max_rect(tree_rect), |ui| {
            ui.set_min_size(tree_rect.size());
            self.tree(ui, cx, trial, is_default, is_copy);
        });

        // -- the controls below -----------------------------------------------------------
        let controls = Rect::from_min_max(pos2(full.min.x, tree_rect.max.y + gap), full.max);
        ui.scope_builder(egui::UiBuilder::new().max_rect(controls), |ui| {
            ui.spacing_mut().item_spacing = vec2(0.0, gap);
            let w = controls.width();
            // checkBoxShowMarkerStatus.
            ui.allocate_ui(vec2(w, row_h), |ui| {
                ui.set_min_size(vec2(w, row_h));
                ui.centered_and_justified(|ui| {
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        let mut v = cx.settings.show_marker_states;
                        if ui.checkbox(&mut v, "Show marker status").clicked() {
                            // on_checkBoxShowMarkerStatus_clicked: the setting; the tree
                            // takes its status columns from it on the next draw.
                            cx.settings.show_marker_states = v;
                        }
                    });
                });
            });
            // checkBoxIsFromMaster: shown, checked, for a copy of the default trial.
            // Unchecking asks, then removes the link; the box stays checked until then.
            if is_copy {
                let a = Action::Points(PointsAction::UnlinkFromDefault);
                let on = enabled(cx, &a);
                ui.allocate_ui(vec2(w, row_h), |ui| {
                    ui.set_min_size(vec2(w, row_h));
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        let mut v = true;
                        if ui
                            .add_enabled(on, egui::Checkbox::new(&mut v, "from default"))
                            .clicked()
                        {
                            cx.raise(a.clone());
                        }
                    });
                });
            }
            // pushButtonApply: shown for the default trial.
            if is_default {
                let a = Action::Points(PointsAction::ApplyToTrials);
                if push_button(ui, vec2(w, row_h), "Apply to trials", enabled(cx, &a)).clicked() {
                    cx.raise(a);
                }
            }
            // Disabled in a trial linked to the Default trial.
            let buttons = [
                ("Set Number Markers", PointsAction::SetNumberMarkers),
                (
                    "Set Number Rigid Bodies",
                    PointsAction::SetNumberRigidBodies,
                ),
                ("Import/Export", PointsAction::ImportExport),
            ];
            let mut raised = None;
            let mut draw = |ui: &mut Ui, bw: f32, cx: &Cx<'_>| {
                for (l, p) in &buttons {
                    let a = Action::Points(p.clone());
                    let on = !is_copy && enabled(cx, &a);
                    if push_button(ui, vec2(bw, row_h), l, on).clicked() {
                        raised = Some(a);
                    }
                }
            };
            if stacked {
                draw(ui, w, cx);
            } else {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = gap;
                    let bw = (w - 2.0 * gap) / 3.0;
                    draw(ui, bw, cx);
                });
            }
            if let Some(a) = raised {
                cx.raise(a);
            }
        });
        ui.advance_cursor_after_rect(outer);
    }
}

/// `reloadListFromObject` after a trial change: the trial's active marker (else body) is
/// selected again if it still exists, otherwise forgotten.
fn reselect_active(s: &mut AppState, doc: Option<&xma_engine::Document>) -> Changes {
    let Some(trial) = doc
        .zip(s.active_trial)
        .and_then(|(d, t)| d.project().trials.get(t))
    else {
        return Changes::NONE;
    };
    let (nm, nb) = (trial.markers.len(), trial.rigid_bodies.len());
    let Some(v) = s.trial_view_mut() else {
        return Changes::NONE;
    };
    let item = match (v.active_marker, v.active_body) {
        (Some(m), _) if m < nm => Some(TreeItem::Marker(m)),
        (None, Some(b)) if b < nb => Some(TreeItem::Body(b)),
        _ => None,
    };
    if item.is_none() {
        v.active_marker = None;
        v.active_body = None;
    }
    s.selection = crate::state::Selection::default();
    if let Some(it) = item {
        insert(&mut s.selection, it);
        set_active(s, it);
    }
    Changes {
        selection: true,
        ..Changes::NONE
    }
}

impl PointsPanel {
    /// Draws `treeWidgetPoints`.
    #[allow(clippy::too_many_lines)]
    fn tree(
        &mut self,
        ui: &mut Ui,
        cx: &mut Cx<'_>,
        trial: Option<&Arc<xma_model::Trial>>,
        is_default: bool,
        is_copy: bool,
    ) {
        let rect = ui.max_rect();
        // The list's base and alternating row colours: egui's text-field background and its
        // striping colour.
        let base = ui.visuals().extreme_bg_color;
        let alternate = ui.visuals().faint_bg_color;
        ui.painter().rect(
            rect,
            0.0,
            base,
            Stroke::new(1.0, crate::theme::frame_line(ui.visuals())),
            egui::StrokeKind::Inside,
        );
        let inner = rect.shrink(2.0);
        let Some(trial) = trial else {
            ui.allocate_rect(rect, Sense::hover());
            return;
        };
        let t = cx.state.active_trial.unwrap_or(0);
        let changes = cx.doc.map_or(0, xma_engine::Document::changes);
        let mut cache = match self.rows.take() {
            Some(c) if c.is_for(trial, changes) => c,
            _ => RowsCache::new(trial, changes),
        };
        let font = egui::TextStyle::Body.resolve(ui.style());
        cache.update_shown(
            ui,
            trial.rigid_bodies.len(),
            |b| self.is_expanded(t, b),
            &font,
        );
        let rows = &cache.rows;
        let shown_keys = &cache.shown_keys;
        let col0 = cache.col0;

        let status_cols = if cx.settings.show_marker_states
            && cx.state.active_frame_trial.is_some()
            && cx.state.workspace == Workspace::MarkerTracking
        {
            cx.facts.n_cameras
        } else {
            0
        };
        let frame = cx.state.active_frame_trial.unwrap_or(0);

        let selection = cx.state.selection.clone();
        let active = cx.state.trial_view().and_then(|v| {
            v.active_marker
                .map(TreeItem::Marker)
                .or(v.active_body.map(TreeItem::Body))
        });
        let text_color = ui.visuals().text_color();
        let sel_fill = ui.visuals().selection.bg_fill;
        let sel_text = ui.visuals().selection.stroke.color;
        let scroll_to = std::mem::take(&mut self.scroll_to_active);
        let icons = cx.icons;
        let mut colors = StatusColorTable {
            settings: cx.settings,
            colors: [None; 20],
        };
        let mut clicked: Option<(RowKey, ClickKind)> = None;
        let mut toggle: Option<usize> = None;
        let mut raised: Option<Action> = None;
        // Every Points action has the same rule; the exports and the body dialog their own.
        let points_on = enabled(cx, &Action::Points(PointsAction::CreateRigidBody));
        let body_on = enabled(cx, &Action::RigidBodySettings(0));
        let export3d_on = enabled(cx, &Action::Export3DPoints);
        let export_rb_on = enabled(cx, &Action::ExportRigidBodyTransformations);
        let action_on = |a: &Action| match a {
            Action::Points(_) => points_on,
            Action::Export3DPoints => export3d_on,
            Action::ExportRigidBodyTransformations => export_rb_on,
            _ => false,
        };
        // MarkerTreeWidget: drag enabled except in a trial linked to the Default trial.
        let drag_enabled = !is_copy && points_on;
        let mut row_rects: Vec<(Rect, RowKey)> = Vec::new();
        let mut drop_at: Option<egui::Pos2> = None;
        let mut drag_start: Option<RowKey> = None;
        let mut drag_hover: Option<Rect> = None;
        let tree_area = inner;

        ui.scope_builder(egui::UiBuilder::new().max_rect(inner), |ui| {
            // A horizontal scroll bar appears when the fixed columns do not fit, as in Qt.
            egui::ScrollArea::both()
                .id_salt("points_tree")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = vec2(0.0, 0.0);
                    let min_w = col0 + STATUS_COL_W * status_cols as f32 + BUTTON_COL_W;
                    let w = ui.available_width().max(min_w);
                    for (i, r) in cache.shown.iter().map(|&i| &rows[i]).enumerate() {
                        let (row_rect, _) = ui.allocate_exact_size(vec2(w, ROW_H), Sense::hover());
                        let key = (r.item, r.parent);
                        let id = ui.scope_id().with(("row", key));
                        let draggable = drag_enabled && matches!(r.item, TreeItem::Marker(_));
                        let sense = if draggable {
                            Sense::click_and_drag()
                        } else {
                            Sense::click()
                        };
                        let resp = ui.interact(row_rect, id, sense);
                        row_rects.push((row_rect, key));
                        if self.dragging
                            && ui
                                .input(|i| i.pointer.hover_pos())
                                .is_some_and(|p| row_rect.contains(p))
                        {
                            drag_hover = Some(row_rect);
                        }
                        let selected = is_selected(&selection, r.item);
                        resp.widget_info(|| {
                            WidgetInfo::selected(Role::TreeItem, true, selected, &r.id_text)
                        });
                        if scroll_to
                            && Some(r.item) == active
                            && r.parent == first_parent(rows, r.item)
                        {
                            ui.scroll_to_rect(row_rect, None);
                        }
                        let painter = ui.painter_at(row_rect);
                        let fill = if selected {
                            sel_fill
                        } else if i % 2 == 1 {
                            alternate
                        } else {
                            base
                        };
                        painter.rect_filled(row_rect, 0.0, fill);
                        let fg = if selected { sel_text } else { text_color };
                        let x0 = row_rect.min.x;
                        let cy = row_rect.center().y;
                        // Branch decoration.
                        if r.n_children > 0
                            && let TreeItem::Body(b) = r.item
                        {
                            let open = self.is_expanded(t, b);
                            let c = pos2(x0 + INDENT / 2.0, cy);
                            let arrow_rect = Rect::from_center_size(c, vec2(INDENT, ROW_H));
                            let aresp =
                                ui.interact(arrow_rect, id.with("expander"), Sense::click());
                            aresp.widget_info(|| {
                                let name = format!(
                                    "{} {}",
                                    if open { "Collapse" } else { "Expand" },
                                    r.id_text
                                );
                                WidgetInfo::labeled(Role::Button, true, name)
                            });
                            let s = 3.5;
                            let pts = if open {
                                vec![
                                    pos2(c.x - s, c.y - s * 0.5),
                                    pos2(c.x + s, c.y - s * 0.5),
                                    pos2(c.x, c.y + s * 0.7),
                                ]
                            } else {
                                vec![
                                    pos2(c.x - s * 0.5, c.y - s),
                                    pos2(c.x - s * 0.5, c.y + s),
                                    pos2(c.x + s * 0.7, c.y),
                                ]
                            };
                            let arrow_col = if selected {
                                sel_text
                            } else {
                                ui.visuals().weak_text_color()
                            };
                            painter.add(egui::Shape::convex_polygon(pts, arrow_col, Stroke::NONE));
                            if aresp.clicked() {
                                toggle = Some(b);
                            }
                        }
                        // Column 0: the index text.
                        let tx = x0 + INDENT * (r.depth() as f32 + 1.0) + CELL_MARGIN;
                        let col0_rect = Rect::from_min_max(
                            pos2(x0, row_rect.min.y),
                            pos2(x0 + col0, row_rect.max.y),
                        );
                        painter.with_clip_rect(col0_rect).text(
                            pos2(tx, cy),
                            Align2::LEFT_CENTER,
                            &r.id_text,
                            font.clone(),
                            fg,
                        );
                        // The right-hand columns: status swatches, then the button.
                        let button_x = row_rect.max.x - BUTTON_COL_W;
                        let status_x = button_x - STATUS_COL_W * status_cols as f32;
                        // Column 1: the description, stretched into what is left.
                        let desc_rect = Rect::from_min_max(
                            pos2(x0 + col0, row_rect.min.y),
                            pos2(status_x, row_rect.max.y),
                        );
                        if desc_rect.width() > 0.0 {
                            painter
                                .with_clip_rect(desc_rect.shrink2(vec2(CELL_MARGIN, 0.0)))
                                .text(
                                    pos2(desc_rect.min.x + CELL_MARGIN, cy),
                                    Align2::LEFT_CENTER,
                                    &r.description,
                                    font.clone(),
                                    fg,
                                );
                        }
                        if let TreeItem::Marker(m) = r.item
                            && status_cols > 0
                        {
                            for (c, st) in marker_statuses_iter(trial, m, frame)
                                .take(status_cols)
                                .enumerate()
                            {
                                let cx0 = status_x + STATUS_COL_W * c as f32;
                                let sw = Rect::from_center_size(
                                    pos2(cx0 + STATUS_COL_W / 2.0, cy),
                                    vec2(8.0, 16.0),
                                );
                                painter.rect_filled(sw, 0.0, colors.get(st));
                            }
                        }
                        // The settings button (MarkerTreeWidgetButton).
                        let icon = r.body_icon.map_or(Icon::Settings, body_icon);
                        let brect = Rect::from_center_size(
                            pos2(button_x + BUTTON_COL_W / 2.0, cy),
                            vec2(22.0, 22.0),
                        );
                        let (baction, bon) = match r.item {
                            TreeItem::Body(b) => (Action::RigidBodySettings(b), body_on),
                            TreeItem::Marker(m) => (
                                Action::Points(PointsAction::MarkerSettings { marker: m }),
                                points_on,
                            ),
                        };
                        ui.scope_builder(egui::UiBuilder::new().max_rect(brect), |ui| {
                            ui.spacing_mut().button_padding = vec2(3.0, 3.0);
                            let img = icons.image(ui.ctx(), icon, 16.0);
                            let bresp = ui.add_enabled(
                                bon,
                                egui::Button::image(img).frame_when_inactive(false),
                            );
                            // `Response::accessible_name`, the name made only when there is
                            // an accessibility tree to put it in.
                            ui.ctx().accesskit_node_builder(bresp.id, |node| {
                                node.set_label(match r.item {
                                    TreeItem::Body(_) => format!("{} settings", r.id_text),
                                    TreeItem::Marker(_) => {
                                        format!("Marker {} settings", r.id_text)
                                    }
                                });
                            });
                            if bresp.clicked() {
                                raised = Some(baction.clone());
                            }
                        });

                        // Mouse.
                        let mods = ui.input(|i| i.modifiers);
                        if resp.double_clicked()
                            && let TreeItem::Body(b) = r.item
                            && r.n_children > 0
                        {
                            toggle = Some(b);
                        } else if resp.clicked() {
                            let kind = match (mods.command, mods.shift) {
                                (false, false) => ClickKind::Plain,
                                (true, false) => ClickKind::Toggle,
                                (false, true) => ClickKind::Range,
                                (true, true) => ClickKind::AddRange,
                            };
                            clicked = Some((key, kind));
                        } else if resp.secondary_clicked() {
                            clicked = Some((key, ClickKind::Context));
                        }
                        if resp.drag_started() {
                            drag_start = Some(key);
                        }
                        if resp.drag_stopped() {
                            drop_at = ui.input(|i| i.pointer.interact_pos());
                        }
                        resp.context_menu(|ui| {
                            for (e, a) in context_actions(r.item, is_default, is_copy) {
                                if ui
                                    .add_enabled(action_on(&a), egui::Button::new(e))
                                    .clicked()
                                {
                                    raised = Some(a);
                                    ui.close();
                                }
                            }
                        });
                    }
                });
        });

        if let Some(r) = drag_hover {
            // Qt's drop indicator: a frame around the row under the pointer.
            ui.painter().rect_stroke(
                r.shrink(0.5),
                0.0,
                Stroke::new(1.0, ui.visuals().text_color()),
                egui::StrokeKind::Inside,
            );
        }
        if let Some(b) = toggle {
            let v = self.is_expanded(t, b);
            self.set_expanded(t, b, !v);
        }
        if let Some((key, kind)) = clicked {
            cx.change(|s| self.click_row(s, shown_keys, key, kind));
        }
        if let Some(key) = drag_start {
            // A press on an unselected row selects it before the drag (Qt's mouse press).
            if !is_selected(&cx.state.selection, key.0) {
                cx.change(|s| self.click_row(s, shown_keys, key, ClickKind::Plain));
            }
            self.dragging = true;
        }
        if let Some(pos) = drop_at {
            if self.dragging
                && let Some(a) = drop_action(&row_rects, tree_area, pos)
            {
                raised = Some(a);
            }
            self.dragging = false;
        }
        if let Some(a) = raised {
            cx.raise(a);
        }
        self.rows = Some(cache);
    }
}

/// E23: the action a drop at `pos` raises, or `None` outside the tree. The rows are the
/// shown rows and their rectangles.
///
/// Ported from ui/MarkerTreeWidget.cpp, MarkerTreeWidget::dropEvent (`itemAt(pos)`).
#[must_use]
pub fn drop_action(rows: &[(Rect, RowKey)], tree: Rect, pos: egui::Pos2) -> Option<Action> {
    if !tree.contains(pos) {
        return None;
    }
    let body = match rows.iter().find(|(r, _)| r.contains(pos)) {
        None => None,
        Some((_, (TreeItem::Body(b), _))) => Some(*b),
        Some((_, (TreeItem::Marker(_), parent))) => *parent,
    };
    Some(Action::Points(PointsAction::Drop { body }))
}

/// The parent of the first row that shows `item` (the row the C++ would make current).
fn first_parent(rows: &[TreeRow], item: TreeItem) -> Option<usize> {
    rows.iter().find(|r| r.item == item).and_then(|r| r.parent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TrialView;

    fn state() -> AppState {
        AppState {
            trials: vec![TrialView {
                active_frame: 0,
                start_frame: 1,
                end_frame: 10,
                n_frames: 10,
                active_marker: None,
                active_body: None,
            }],
            active_trial: Some(0),
            ..AppState::default()
        }
    }

    const M: fn(usize) -> RowKey = |m| (TreeItem::Marker(m), None);

    #[test]
    fn clicks_follow_extended_selection() {
        let mut p = PointsPanel::default();
        let mut s = state();
        let shown: Vec<RowKey> = vec![(TreeItem::Body(0), None), M(0), M(1), M(2), M(3)];
        p.click_row(&mut s, &shown, M(1), ClickKind::Plain);
        assert_eq!(s.trials[0].active_marker, Some(1));
        assert_eq!(s.selection.markers.iter().copied().collect::<Vec<_>>(), [1]);
        p.click_row(&mut s, &shown, M(3), ClickKind::Range);
        assert_eq!(
            s.selection.markers.iter().copied().collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert_eq!(s.trials[0].active_marker, Some(3));
        p.click_row(&mut s, &shown, M(2), ClickKind::Toggle);
        assert_eq!(
            s.selection.markers.iter().copied().collect::<Vec<_>>(),
            [1, 3]
        );
        assert_eq!(
            s.trials[0].active_marker,
            Some(2),
            "Ctrl click makes the row current"
        );
        p.click_row(
            &mut s,
            &shown,
            (TreeItem::Body(0), None),
            ClickKind::AddRange,
        );
        // Ctrl+Shift adds the rows from the anchor (marker 2) up to the body.
        assert_eq!(
            s.selection.markers.iter().copied().collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert!(s.selection.bodies.contains(&0));
        assert_eq!(s.trials[0].active_body, Some(0));
        assert_eq!(s.trials[0].active_marker, None);
        // Right click on a selected row keeps the selection.
        p.click_row(&mut s, &shown, M(1), ClickKind::Context);
        assert_eq!(s.selection.markers.len(), 4);
        assert_eq!(s.trials[0].active_marker, Some(1));
        // On an unselected row it selects only that row.
        p.click_row(&mut s, &shown, M(3), ClickKind::Toggle);
        p.click_row(&mut s, &shown, M(3), ClickKind::Context);
        assert_eq!(s.selection.markers.iter().copied().collect::<Vec<_>>(), [3]);
        assert!(s.selection.bodies.is_empty());
    }

    #[test]
    fn context_menu_entries_like_cpp() {
        let all = context_entries(TreeItem::Marker(0), false, false);
        assert_eq!(all.len(), 9);
        assert_eq!(all[0], "Change Description");
        assert_eq!(all[8], "Save the 3D data of selected points");
        let default = context_entries(TreeItem::Marker(0), true, false);
        assert_eq!(
            default,
            [
                "Change Description",
                "Create Rigid Body from selected",
                "Add selected to Rigid Body",
                "Change Detectionmethod of selected Points"
            ]
        );
        let copy = context_entries(TreeItem::Marker(0), false, true);
        assert_eq!(copy.len(), 4);
        assert_eq!(copy[0], "Delete Data from selected points");
        assert_eq!(
            context_entries(TreeItem::Body(0), false, false),
            [
                "Change Description",
                "Save the transformation of selected rigid bodies"
            ]
        );
        assert!(context_entries(TreeItem::Body(0), true, true).is_empty());
    }
}
