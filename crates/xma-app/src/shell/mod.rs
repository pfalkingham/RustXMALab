//! The docking shell: where the nine panels are, around the central camera-view area.
//!
//! Production port of the docking spike (`spikes/docking/`, egui_tiles variant with compact
//! titles), following Qt's `QDockWidget` behaviour as XMALab 3 uses it:
//!
//! - **Arrangement** in an `egui_tiles::Tree` whose panes are [`DockPane`]s: the nine
//!   panels and the central area. The central area has no tab bar, cannot be dragged, and
//!   nothing may be tabbed with it. Default arrangement: [`layout::default_tree`].
//! - **One slim title row per panel** (PLAN.md section 5, "Compact panel titles"): a panel
//!   alone in its tab group draws one title row (title, float and close buttons); stacked
//!   panels show one tab strip that carries the active tab's buttons. The title row and
//!   tabs are drag handles; double-clicking them floats the panel.
//! - **Shown or hidden**: the application decides which panels are shown
//!   (`crate::app::panel_wanted`, the rules of `MainWindow::workspaceChanged`); [`Shell::sync`]
//!   hides a tile that is not shown or is floating, so it keeps its place in the tree and
//!   comes back exactly where it was (Qt's `restoreDockWidget`).
//! - **Floating** panels are separate operating-system windows ([`floating`]).
//! - **Lock UI Panes**: no dragging, resizing, floating or closing; the title rows stay.
//! - **Allowed sides** (Qt `allowedAreas`: Toolbox left or right, Console and Progress top
//!   or bottom): checked after every change; a layout that breaks a rule is reverted to the
//!   last valid one, with a message in the log ([`Shell::take_messages`]).
//! - **Save and restore**: [`Shell::to_saved`] and [`Shell::from_saved`] (JSON, kept in the
//!   settings as `UIState_XMALab`).
//!
//! Requests from the title rows (float, dock, close) are raised as actions
//! (`Action::FloatPanel` ...) and carried out by the application through [`Shell::float`],
//! [`Shell::dock`] and [`Shell::close`].

mod arrangement;
pub mod floating;
pub mod layout;
mod tiles;
pub mod title;

use std::collections::{BTreeMap, BTreeSet};

use egui::{Rect, Ui};
use egui_tiles::{Container, LinearDir, Tile, TileId, Tree};
use serde::{Deserialize, Serialize};

use crate::panels::{CxParts, PanelId, Panels, Side};

/// Version of the saved layout format. A saved layout with another version is ignored
/// (the default arrangement is used), as Qt's `restoreState` does with `UI_VERSION`.
/// Version 2: shares in points (`layout.rs`), so a version 1 layout (proportional shares)
/// is not used.
pub const LAYOUT_VERSION: u32 = 2;

/// A pane of the docking tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DockPane {
    /// The camera-view area (Qt's central widget): not a panel.
    Central,
    Panel(PanelId),
}

/// The main window's inner size in points when no geometry is saved (and after View > Reset
/// layout): 1280 × 1024 (`MainWindow.cpp:158-161`).
pub const DEFAULT_MAIN_SIZE: [f32; 2] = [1280.0, 1024.0];
/// The main window's outer position when no geometry is saved (`MainWindow.cpp:158-161`).
pub const DEFAULT_MAIN_POS: [f32; 2] = [50.0, 50.0];

/// Window geometry in points. `pos` is the outer top-left on the desktop, `None` where the
/// platform does not report it (Wayland).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WinGeom {
    pub pos: Option<[f32; 2]>,
    pub size: [f32; 2],
}

#[derive(Serialize, Deserialize)]
struct SavedLayout {
    version: u32,
    tree: Tree<DockPane>,
    floating: BTreeMap<PanelId, WinGeom>,
    main_window: Option<WinGeom>,
}

/// Rectangles from the last frame, for tests and measurements.
#[derive(Clone, Debug, Default)]
pub struct ShellFrame {
    /// Title rows of docked lone panels.
    pub header_rects: BTreeMap<PanelId, Rect>,
    /// Tab buttons of stacked panels.
    pub tab_rects: BTreeMap<PanelId, Rect>,
    /// Panel bodies (below the title row or tab strip), docked panels only.
    pub body_rects: BTreeMap<PanelId, Rect>,
    /// Panels that were drawn this frame (docked or floating).
    pub drawn: BTreeSet<PanelId>,
    /// The central area.
    pub central_rect: Option<Rect>,
}

/// What the shell needs while drawing.
pub struct ShellDraw<'a, 'p> {
    pub panels: &'a mut Panels,
    pub parts: &'a mut CxParts<'p>,
    /// A non-cancelable job runs: everything but the Progress panel is disabled.
    pub busy_window: bool,
    /// A cancelable job runs: Toolbox, Points, Plot, Detailed View, frame and workspace bars
    /// are disabled.
    pub busy_cancelable: bool,
    /// Key presses are turned into actions in this frame (no dialog is open): a floating
    /// panel's window then takes the frame arrows pressed in it
    /// (`actions::collect_frame_arrows`).
    pub shortcuts: bool,
    /// `ActionContext::busy`, for the enabled rule of those actions.
    pub busy_actions: bool,
    /// Draws the central area.
    pub central: &'a mut dyn FnMut(&mut Ui, &mut CxParts<'p>),
}

/// The docking shell. See the module documentation.
pub struct Shell {
    pub tree: Tree<DockPane>,
    /// Floating panels and their window geometry.
    pub floating: BTreeMap<PanelId, WinGeom>,
    /// The main window's geometry, saved with the layout.
    pub main_window: Option<WinGeom>,
    locked: bool,
    shown: BTreeMap<PanelId, bool>,
    last_good: Option<Tree<DockPane>>,
    prev_sig: String,
    pending: bool,
    /// The side each docked panel was last validated on.
    pub last_side: BTreeMap<PanelId, Side>,
    messages: Vec<String>,
    /// Rectangles from the last frame.
    pub frame: ShellFrame,
    pub(crate) float: floating::FloatState,
    /// Number of layouts reverted because they broke a rule.
    pub reverts: u32,
    /// The workspace the Toolbox was last sized for.
    toolbox_workspace: Option<crate::state::Workspace>,
    /// Sizes of panels squeezed below their size by a small window ([`Shell::fit_central`]).
    squeezed: std::collections::HashMap<TileId, f32>,
    /// Panels (and, under `None`, the camera views) that panicked while drawing: they show
    /// the error instead of their contents until [`Shell::clear_failures`].
    pub failures: Failures,
}

/// Panels that stopped after a panic, with the panic message.
pub type Failures = BTreeMap<Option<PanelId>, String>;

/// Runs `f`, turning a panic into its message. A panel or the camera views panicking while
/// drawing must not take the window down: the shell shows the message in its place and
/// writes it to the Console (the panic hook has already printed it on standard error).
pub(crate) fn guard(f: impl FnOnce()) -> Result<(), String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|e| panic_message(&*e))
}

/// The text of a panic payload.
#[must_use]
pub fn panic_message(e: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown error".to_owned()
    }
}

/// What a stopped panel shows instead of its contents.
pub(crate) fn failed_body(ui: &mut Ui, what: &str, msg: &str) {
    ui.add_space(6.0);
    ui.horizontal_wrapped(|ui| {
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(format!(
                "{what} stopped after an internal error and is not drawn until the dataset is \
                 reopened. The Console has the details.\n{msg}"
            ))
            .color(ui.visuals().error_fg_color),
        );
    });
}

impl Default for Shell {
    fn default() -> Self {
        Self::new_default()
    }
}

impl Shell {
    /// The default arrangement, with the first-run floating panels.
    #[must_use]
    pub fn new_default() -> Self {
        let floating = layout::FIRST_RUN_FLOATING
            .iter()
            .map(|(p, size)| {
                (
                    *p,
                    WinGeom {
                        pos: None,
                        size: *size,
                    },
                )
            })
            .collect();
        let mut s = Self::from_parts(layout::default_tree(), floating, None);
        s.last_side = [
            (PanelId::Points, Side::Left),
            (PanelId::WorldView, Side::Left),
            (PanelId::Toolbox, Side::Right),
            (PanelId::Progress, Side::Top),
            (PanelId::DetailedView, Side::Bottom),
            (PanelId::Events, Side::Bottom),
            (PanelId::DisplayOptions, Side::Bottom),
            (PanelId::Plot, Side::Bottom),
            (PanelId::Console, Side::Bottom),
        ]
        .into();
        s
    }

    fn from_parts(
        tree: Tree<DockPane>,
        floating: BTreeMap<PanelId, WinGeom>,
        main_window: Option<WinGeom>,
    ) -> Self {
        let float = floating::FloatState::new(&floating);
        let mut s = Self {
            tree,
            floating,
            main_window,
            locked: false,
            shown: BTreeMap::new(),
            last_good: None,
            prev_sig: String::new(),
            pending: true,
            last_side: BTreeMap::new(),
            messages: Vec::new(),
            frame: ShellFrame::default(),
            float,
            reverts: 0,
            toolbox_workspace: None,
            squeezed: std::collections::HashMap::new(),
            failures: Failures::new(),
        };
        // Everything hidden until the first sync.
        for p in PanelId::ALL {
            s.set_tile_visible(p, false);
        }
        s
    }

    /// View > Reset layout (parity ledger U165): the layout becomes [`Shell::new_default`],
    /// the one `--reset-layout` starts with: the default tree with its sizes, the first-run
    /// floating panels (3D world view and Console) with no saved position, no main-window
    /// geometry. Kept, because they are not part of the layout: Lock UI Panes (a setting),
    /// which panels are shown (each panel's setting and the workspace, given again by
    /// [`Shell::sync`]), the panels stopped after an error, the messages not yet logged.
    ///
    /// The floating windows are opened again as new operating-system windows on the next
    /// frame ([`floating`]), so the window manager places them, as on a fresh start: an
    /// existing window is not moved when its builder's position becomes unset. The main
    /// window is moved and sized by the application (`XmaApp::reset_layout`).
    pub fn reset_layout(&mut self) {
        let mut fresh = Self::new_default();
        fresh.locked = self.locked;
        fresh.failures = std::mem::take(&mut self.failures);
        fresh.messages = std::mem::take(&mut self.messages);
        fresh.reverts = self.reverts;
        fresh.float.deco = self.float.deco;
        fresh.float.generation = self.float.generation.wrapping_add(1);
        let shown = std::mem::take(&mut self.shown);
        *self = fresh;
        self.sync(&shown);
    }

    /// Restores a saved layout.
    ///
    /// # Errors
    /// When the JSON is not a layout of this version, or lacks a pane.
    pub fn from_saved(v: &serde_json::Value) -> Result<Self, String> {
        let saved: SavedLayout = serde_json::from_value(v.clone()).map_err(|e| e.to_string())?;
        if saved.version != LAYOUT_VERSION {
            return Err(format!(
                "layout version {} is not {LAYOUT_VERSION}",
                saved.version
            ));
        }
        let s = Self::from_parts(saved.tree, saved.floating, saved.main_window);
        for p in PanelId::ALL {
            if s.tile(p).is_none() {
                return Err(format!("saved layout has no {}", p.title()));
            }
        }
        if s.tile_of(DockPane::Central).is_none() {
            return Err("saved layout has no central area".into());
        }
        Ok(s)
    }

    /// The layout as JSON (the tree's set of hidden tiles is sorted, so equal layouts give
    /// equal JSON).
    #[must_use]
    pub fn to_saved(&self) -> serde_json::Value {
        let saved = SavedLayout {
            version: LAYOUT_VERSION,
            tree: self.tree.clone(),
            floating: self.floating.clone(),
            main_window: self.main_window,
        };
        let mut v = serde_json::to_value(&saved).unwrap_or_default();
        if let Some(serde_json::Value::Array(a)) = v.pointer_mut("/tree/tiles/invisible") {
            a.sort_by_key(serde_json::Value::as_u64);
        }
        v
    }

    fn tile_of(&self, pane: DockPane) -> Option<TileId> {
        self.tree.tiles.find_pane(&pane)
    }

    fn tile(&self, p: PanelId) -> Option<TileId> {
        self.tile_of(DockPane::Panel(p))
    }

    fn set_tile_visible(&mut self, p: PanelId, visible: bool) {
        if let Some(t) = self.tile(p)
            && self.tree.tiles.is_visible(t) != visible
        {
            self.tree.tiles.set_visible(t, visible);
        }
    }

    /// Lock UI Panes.
    pub fn set_locked(&mut self, locked: bool) {
        self.locked = locked;
    }

    #[must_use]
    pub fn locked(&self) -> bool {
        self.locked
    }

    /// Sets which panels are shown (docked or floating), as the C++ calls `show()` and
    /// `hide()` on its docks. A docked panel's tile is visible when the panel is shown and
    /// not floating.
    pub fn sync(&mut self, shown: &BTreeMap<PanelId, bool>) {
        self.shown.clone_from(shown);
        for p in PanelId::ALL {
            let want = self.is_shown(p) && !self.floating.contains_key(&p);
            self.set_tile_visible(p, want);
        }
    }

    /// Shown, docked or floating.
    #[must_use]
    pub fn is_shown(&self, p: PanelId) -> bool {
        self.shown.get(&p).copied().unwrap_or(false)
    }

    /// Shown and docked.
    #[must_use]
    pub fn is_docked(&self, p: PanelId) -> bool {
        self.tile(p).is_some_and(|t| self.tree.tiles.is_visible(t))
    }

    /// Floating (whether shown or not).
    #[must_use]
    pub fn is_floating(&self, p: PanelId) -> bool {
        self.floating.contains_key(&p)
    }

    /// Floats a panel in its own window, over where it was. `main_inner` is the main
    /// window's inner rectangle on the desktop, when known.
    pub fn float(&mut self, p: PanelId, main_inner: Option<Rect>) {
        if self.locked || self.floating.contains_key(&p) {
            return;
        }
        let r = self.panel_rect(p);
        let size = r.map_or([320.0, 260.0], |r| {
            [r.width().max(220.0), r.height().max(120.0)]
        });
        let pos = match (r, main_inner) {
            (Some(r), Some(m)) => Some([m.min.x + r.min.x + 20.0, m.min.y + r.min.y + 20.0]),
            _ => None,
        };
        let g = WinGeom { pos, size };
        self.floating.insert(p, g);
        self.float.opened(p, g);
        self.resync();
    }

    /// Docks a floating panel back where it was.
    pub fn dock(&mut self, p: PanelId) {
        if self.locked {
            return;
        }
        self.floating.remove(&p);
        self.resync();
        // Qt raises a dock put back into a tab group.
        if let Some(t) = self.tile(p) {
            self.tree.make_active(|id, _| id == t);
        }
    }

    /// Docks a floating panel at an edge of the main window (dragging a floating window's
    /// grip onto the main window).
    pub fn dock_at(&mut self, p: PanelId, side: Side) {
        if self.locked || self.floating.remove(&p).is_none() {
            return;
        }
        let (Some(pane), Some(root)) = (self.tile(p), self.tree.root) else {
            return;
        };
        if let Some(parent) = self.tree.tiles.parent_of(pane)
            && let Some(Tile::Container(c)) = self.tree.tiles.get_mut(parent)
        {
            c.remove_child(pane);
        }
        let tabs = self.tree.tiles.insert_tab_tile(vec![pane]);
        let (dir, first) = match side {
            Side::Left => (LinearDir::Horizontal, true),
            Side::Right => (LinearDir::Horizontal, false),
            Side::Top => (LinearDir::Vertical, true),
            Side::Bottom => (LinearDir::Vertical, false),
        };
        let children = if first {
            vec![tabs, root]
        } else {
            vec![root, tabs]
        };
        let new_root = self
            .tree
            .tiles
            .insert_container(Container::new_linear(dir, children));
        if let Some(Tile::Container(Container::Linear(l))) = self.tree.tiles.get_mut(new_root) {
            l.shares.set_share(tabs, 1.0);
            l.shares.set_share(root, 4.0);
        }
        self.tree.root = Some(new_root);
        self.resync();
    }

    fn resync(&mut self) {
        let shown = self.shown.clone();
        self.sync(&shown);
    }

    /// The panel's rectangle from the last frame (a hidden tab: its tab group's).
    #[must_use]
    pub fn panel_rect(&self, p: PanelId) -> Option<Rect> {
        let t = self.tile(p)?;
        if !self.tree.tiles.is_visible(t) {
            return None;
        }
        self.tree
            .tiles
            .rect(t)
            .or_else(|| self.tree.tiles.rect(self.tree.tiles.parent_of(t)?))
    }

    /// Containers of the tree with nothing in them. There should be none: the docking crate
    /// gives an empty container its share of the space, so it shows as a blank panel.
    #[must_use]
    pub fn empty_containers(&self) -> usize {
        self.tree
            .tiles
            .iter()
            .filter(|(_, t)| matches!(t, Tile::Container(c) if c.is_empty()))
            .count()
    }

    /// The central area's rectangle from the last frame.
    #[must_use]
    pub fn central_rect(&self) -> Option<Rect> {
        self.tree.tiles.rect(self.tile_of(DockPane::Central)?)
    }

    /// Checks the layout: no panel tabbed with the central area, no tab group that holds a
    /// container, every docked panel on an allowed side. Returns each docked panel's side.
    ///
    /// # Errors
    /// The rule broken, as a message for the user.
    pub fn validate(&self) -> Result<BTreeMap<PanelId, Side>, String> {
        for (_, tile) in self.tree.tiles.iter() {
            if let Tile::Container(Container::Tabs(t)) = tile
                && t.children
                    .iter()
                    .any(|c| !matches!(self.tree.tiles.get(*c), Some(Tile::Pane(_))))
            {
                return Err(
                    "A group of panels cannot be stacked as one tab; move reverted.".into(),
                );
            }
        }
        let c = self
            .tile_of(DockPane::Central)
            .ok_or("central area missing")?;
        let parent = self
            .tree
            .tiles
            .parent_of(c)
            .ok_or("central area has no tab group")?;
        match self.tree.tiles.get(parent) {
            Some(Tile::Container(Container::Tabs(t))) if t.children.len() == 1 => {}
            _ => {
                return Err("Panels cannot be tabbed into the camera views; move reverted.".into());
            }
        }
        let Some(central) = self.central_rect() else {
            return Ok(BTreeMap::new());
        };
        let mut sides = BTreeMap::new();
        for p in PanelId::ALL {
            if !self.is_docked(p) {
                continue;
            }
            let Some(r) = self.panel_rect(p) else {
                continue;
            };
            let Some(side) = side_of(r, central) else {
                return Err(format!(
                    "{} cannot share space with the camera views; move reverted.",
                    p.title()
                ));
            };
            if !p.allowed_sides().contains(&side) {
                let allowed: Vec<_> = p
                    .allowed_sides()
                    .iter()
                    .map(|s| format!("{s:?}").to_lowercase())
                    .collect();
                return Err(format!(
                    "{} can only be docked {}; move reverted.",
                    p.title(),
                    allowed.join(" or ")
                ));
            }
            sides.insert(p, side);
        }
        Ok(sides)
    }

    /// After a frame: if the layout changed, validate it once the crate has laid it out (the
    /// next frame), and revert to the last valid layout if it breaks a rule.
    fn post_frame_validate(&mut self, ctx: &egui::Context) {
        let sig = serde_json::to_string(&self.tree).unwrap_or_default();
        if sig != self.prev_sig {
            self.prev_sig = sig;
            self.pending = true;
            ctx.request_repaint();
            return;
        }
        if !self.pending || self.tree.dragged_id(ctx).is_some() {
            return;
        }
        match self.validate() {
            Ok(sides) => {
                self.last_side.extend(sides);
                self.last_good = Some(self.tree.clone());
                self.pending = false;
            }
            Err(msg) => {
                if let Some(good) = &self.last_good {
                    self.tree = good.clone();
                    self.resync();
                    self.reverts += 1;
                    self.messages.push(msg);
                } else {
                    self.pending = false;
                    self.messages.push(format!("Invalid layout: {msg}"));
                }
                ctx.request_repaint();
            }
        }
    }

    /// Tells the shell the workspace. On a change (and the first time) the Toolbox column is
    /// set to the width of the workspace's page ([`layout::toolbox_width`]), as the C++
    /// Toolbox, laid out with a fixed size, takes the size of the page it shows.
    pub fn set_workspace(&mut self, ws: crate::state::Workspace) {
        if self.toolbox_workspace == Some(ws) {
            return;
        }
        self.toolbox_workspace = Some(ws);
        let Some(t) = self.tile(PanelId::Toolbox) else {
            return;
        };
        // The Toolbox's tab group, in a horizontal container.
        let Some(group) = self.tree.tiles.parent_of(t) else {
            return;
        };
        let Some(parent) = self.tree.tiles.parent_of(group) else {
            return;
        };
        if let Some(Tile::Container(Container::Linear(l))) = self.tree.tiles.get_mut(parent)
            && l.dir == LinearDir::Horizontal
        {
            l.shares.set_share(group, layout::toolbox_width(ws));
        }
    }

    /// Gives the camera views whatever the docked panels leave, as Qt's main window does:
    /// in each container between the window and the central area, the other visible
    /// children keep their size in points (their share) and the child that holds the
    /// central area gets the rest of `rect`. When the window is too small for that, the
    /// others shrink in proportion so the central area keeps [`MIN_CENTRAL`] points.
    ///
    /// Shares smaller than [`MIN_PANEL`] (from a container the docking crate made on a drop)
    /// are read as the child's size in the last frame.
    pub fn fit_central(&mut self, rect: Rect, gap: f32) {
        let Some(central) = self.tile_of(DockPane::Central) else {
            return;
        };
        let mut path = vec![central];
        while let Some(p) = self.tree.tiles.parent_of(*path.last().unwrap_or(&central)) {
            path.push(p);
        }
        path.reverse();
        let mut size = rect.size();
        for w in path.windows(2) {
            let (cont, child) = (w[0], w[1]);
            let visible: Vec<TileId> = match self.tree.tiles.get(cont) {
                Some(Tile::Container(Container::Linear(l))) => l
                    .children
                    .iter()
                    .copied()
                    .filter(|c| self.tree.tiles.is_visible_in_layout(*c))
                    .collect(),
                _ => continue,
            };
            if !visible.contains(&child) {
                return;
            }
            let rects: Vec<Option<Rect>> =
                visible.iter().map(|c| self.tree.tiles.rect(*c)).collect();
            let Some(Tile::Container(Container::Linear(l))) = self.tree.tiles.get_mut(cont) else {
                continue;
            };
            let horizontal = l.dir == LinearDir::Horizontal;
            let along = if horizontal { size.x } else { size.y };
            let avail = (along - gap * visible.len().saturating_sub(1) as f32).max(0.0);
            let mut others: Vec<(TileId, f32)> = Vec::new();
            for (c, r) in visible.iter().zip(&rects) {
                if *c == child {
                    continue;
                }
                // A panel squeezed by a small window gets its size back when there is room.
                let mut s = self.squeezed.get(c).copied().unwrap_or(l.shares[*c]);
                if s < MIN_PANEL {
                    s = r.map_or(200.0, |r| if horizontal { r.width() } else { r.height() });
                }
                others.push((*c, s.max(MIN_PANEL)));
            }
            let sum: f32 = others.iter().map(|o| o.1).sum();
            let room = (avail - MIN_CENTRAL).max(0.0);
            let factor = if sum > room && sum > 0.0 {
                room / sum
            } else {
                1.0
            };
            let mut used = 0.0;
            for (c, s) in &others {
                if factor < 1.0 {
                    self.squeezed.entry(*c).or_insert(*s);
                } else {
                    self.squeezed.remove(c);
                }
                let s = s * factor;
                used += s;
                if (l.shares[*c] - s).abs() > 0.25 {
                    l.shares.set_share(*c, s);
                }
            }
            let rest = (avail - used).max(1.0);
            if (l.shares[child] - rest).abs() > 0.25 {
                l.shares.set_share(child, rest);
            }
            if horizontal {
                size.x = rest;
            } else {
                size.y = rest;
            }
        }
    }

    /// Keeps every docked panel that states a minimum width ([`crate::panels::Panel::min_width`],
    /// the width its texts need in egui's font) at least that wide: its share in the
    /// nearest horizontal container is raised to it. Containers that hold the camera views
    /// are left alone (they get what the others leave).
    pub fn apply_min_widths(&mut self, ui: &Ui, draw: &mut ShellDraw<'_, '_>) {
        let central = self.tile_of(DockPane::Central);
        let mut holds_central = std::collections::HashSet::new();
        let mut c = central;
        while let Some(t) = c {
            holds_central.insert(t);
            c = self.tree.tiles.parent_of(t);
        }
        for p in PanelId::ALL {
            let Some(t) = self.tile(p) else {
                continue;
            };
            if !self.tree.tiles.is_visible(t) {
                continue;
            }
            let need = {
                let cx = draw.parts.cx(false);
                draw.panels.get_mut(p).min_width(ui, &cx)
            };
            let Some(need) = need else {
                continue;
            };
            let mut child = t;
            while let Some(parent) = self.tree.tiles.parent_of(child) {
                if let Some(Tile::Container(Container::Linear(l))) = self.tree.tiles.get_mut(parent)
                    && l.dir == LinearDir::Horizontal
                {
                    if !holds_central.contains(&child) && l.shares[child] < need {
                        l.shares.set_share(child, need);
                        self.squeezed.remove(&child);
                    }
                    break;
                }
                child = parent;
            }
        }
    }

    /// Draws the stopped panels again (after a project is opened or closed).
    pub fn clear_failures(&mut self) {
        self.failures.clear();
    }

    /// Messages for the log (reverted moves).
    pub fn take_messages(&mut self) -> Vec<String> {
        std::mem::take(&mut self.messages)
    }

    /// Draws the docked panels and the central area into `ui`, then the floating windows.
    pub fn ui(&mut self, ui: &mut Ui, draw: &mut ShellDraw<'_, '_>) {
        self.frame = ShellFrame::default();
        let ctx = ui.ctx().clone();
        let main_inner = ctx.input(|i| i.viewport().inner_rect);
        if !ctx.embed_viewports() {
            let outer = ctx.input(|i| i.viewport().outer_rect);
            let content = ctx.content_rect();
            self.main_window = Some(WinGeom {
                pos: floating::outer_pos(&mut self.float.deco, main_inner, outer),
                size: [content.width(), content.height()],
            });
        }
        let gap = tiles::GAP;
        self.apply_min_widths(ui, draw);
        self.fit_central(ui.available_rect_before_wrap(), gap);
        tiles::show(self, ui, draw);
        self.frame.central_rect = self.central_rect();
        floating::show_windows(self, &ctx, draw, main_inner);
        self.post_frame_validate(&ctx);
    }
}

/// A panel body smaller than this (points) in either direction is not drawn.
pub const MIN_BODY: f32 = 24.0;
/// The smallest the camera-view area gets when the window is small (points).
pub const MIN_CENTRAL: f32 = 160.0;
/// The smallest a docked panel gets (points); also the docking crate's minimum.
pub const MIN_PANEL: f32 = 40.0;

/// Which side of the central area a rectangle is on (its centre's largest distance
/// outside the central rectangle). `None` when its centre is inside.
#[must_use]
pub fn side_of(panel: Rect, central: Rect) -> Option<Side> {
    let c = panel.center();
    let candidates = [
        (Side::Left, central.min.x - c.x),
        (Side::Right, c.x - central.max.x),
        (Side::Top, central.min.y - c.y),
        (Side::Bottom, c.y - central.max.y),
    ];
    let (side, d) = candidates.into_iter().max_by(|a, b| a.1.total_cmp(&b.1))?;
    (d > 0.0).then_some(side)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saved_layout_round_trips() {
        let s = Shell::new_default();
        let v = s.to_saved();
        let back = Shell::from_saved(&v).unwrap();
        assert_eq!(back.to_saved(), v);
        assert_eq!(back.floating, s.floating);
    }

    #[test]
    fn wrong_version_is_refused() {
        let mut v = Shell::new_default().to_saved();
        v["version"] = 99.into();
        assert!(Shell::from_saved(&v).is_err());
    }

    /// A layout saved while a panel dropped on itself still left its emptied tab group
    /// behind: the empty groups go when the tree is tidied, and the layout shows what it
    /// showed before them, with the panel at the size it had.
    #[test]
    fn empty_groups_in_a_saved_layout_are_removed() {
        let mut s = Shell::new_default();
        s.sync(&PanelId::ALL.into_iter().map(|p| (p, true)).collect());
        let shown = arrangement::shown(&s.tree);
        let points = s.tile(PanelId::Points).unwrap();
        let old_group = s.tree.tiles.parent_of(points).unwrap();
        let column = s.tree.tiles.parent_of(old_group).unwrap();
        // What three such drops left: Points in a new group, three empty groups under it.
        let Some(Tile::Container(c)) = s.tree.tiles.get_mut(old_group) else {
            panic!("the Points group")
        };
        c.remove_child(points);
        let group = s.tree.tiles.insert_tab_tile(vec![points]);
        let empties = [
            old_group,
            s.tree.tiles.insert_tab_tile(vec![]),
            s.tree.tiles.insert_tab_tile(vec![]),
        ];
        let Some(Tile::Container(Container::Linear(l))) = s.tree.tiles.get_mut(column) else {
            panic!("the Points column")
        };
        l.children.retain(|c| *c != old_group);
        l.children.insert(0, group);
        l.shares.set_share(group, 62.5);
        for (i, e) in empties.into_iter().enumerate() {
            l.children.insert(i + 1, e);
            l.shares.set_share(e, [250.0, 125.0, 62.5][i]);
        }
        assert_eq!(s.empty_containers(), 3);
        let saved = s.to_saved();

        let mut back = Shell::from_saved(&saved).unwrap();
        back.sync(&PanelId::ALL.into_iter().map(|p| (p, true)).collect());
        assert_eq!(back.empty_containers(), 3);
        back.tree.simplify(&tiles::simplification());
        assert_eq!(back.empty_containers(), 0);
        assert_eq!(arrangement::shown(&back.tree), shown);
    }

    #[test]
    fn side_of_works() {
        let central = Rect::from_min_max(egui::pos2(100.0, 0.0), egui::pos2(500.0, 400.0));
        let left = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(100.0, 400.0));
        let bottom = Rect::from_min_max(egui::pos2(0.0, 400.0), egui::pos2(600.0, 600.0));
        assert_eq!(side_of(left, central), Some(Side::Left));
        assert_eq!(side_of(bottom, central), Some(Side::Bottom));
        assert_eq!(side_of(central, central), None);
    }
}
