//! The default arrangement of the panels.
//!
//! XMALab 3 has a built-in arrangement for a first run: `Settings::getUIState`
//! (core/Settings.cpp:455-465) returns a hard-coded `QMainWindow::saveState` blob when none
//! is saved, so `restoreState` always succeeds and the "floating Toolbox" branch of the
//! main-window constructor (`MainWindow.cpp:223-233`) never runs. Decoded, that blob is:
//!
//! | Area | Content |
//! |---|---|
//! | left | Points (285 wide) |
//! | right | Toolbox |
//! | top | Progress (hidden unless a job runs) |
//! | bottom | tabs [Detailed View, Events, Display Options] (486 wide), Plot (758 wide) |
//! | floating | 3D world view (500 × 500), Console (804 × 325) |
//!
//! with the corners top-left → left, top-right → right, both bottom corners → bottom. It
//! is also the arrangement of the author's reference screenshots
//! (`docs/ui-reference/pig/MainWindow_tracking.png`), whose proportions the shares below
//! follow: Points about 117 points wide, Toolbox about 250, the bottom row about a third of
//! the height.
//!
//! Where a floating panel docks back to (its place in the tree while hidden): 3D world
//! view under Points, Console at the right end of the bottom row (the areas the blob
//! records them in).

use egui_tiles::{Container, Tile, TileId, Tiles, Tree};

use super::DockPane;
use crate::panels::PanelId;

/// The tree id (stable, so a saved layout restores to the same egui ids).
pub const TREE_ID: &str = "xma_dock_tree";

/// Panels that float on a first run, with their window size.
pub const FIRST_RUN_FLOATING: [(PanelId, [f32; 2]); 2] = [
    (PanelId::WorldView, [500.0, 500.0]),
    (PanelId::Console, [804.0, 325.0]),
];

fn set_shares(tiles: &mut Tiles<DockPane>, container: TileId, shares: &[(TileId, f32)]) {
    if let Some(Tile::Container(Container::Linear(l))) = tiles.get_mut(container) {
        for (id, s) in shares {
            l.shares.set_share(*id, *s);
        }
    }
}

/// Starting width of the Points column (the reference screenshots: 117 points). The
/// column is widened to what its texts need in egui's font on the first frame
/// (`panels::points::content_width`, applied by `Shell::apply_min_widths`).
pub const POINTS_WIDTH: f32 = 117.0;
/// Height of the bottom row (the C++ built-in state: bottom area 340 high; the reference
/// screenshots, taken in a 1127-point-high window, show 367).
pub const BOTTOM_HEIGHT: f32 = 340.0;
/// Width of the Detailed View tab group in the bottom row (C++ built-in state: 486; the
/// reference: 490). The Plot takes the rest.
pub const DETAIL_WIDTH: f32 = 486.0;
/// Width of the Plot in the bottom row (C++ built-in state: 758).
pub const PLOT_WIDTH: f32 = 758.0;
/// Height of the Progress panel above the camera views (C++ built-in state: 55).
pub const PROGRESS_HEIGHT: f32 = 55.0;

/// Width of the Toolbox column in each workspace. The C++ Toolbox lays its page out with a
/// fixed size (`WizardDockWidget`, `QLayout::SetFixedSize`), so the dock is as wide as the
/// page shown: measured from the reference screenshots (dock outer width, in points).
#[must_use]
pub fn toolbox_width(ws: crate::state::Workspace) -> f32 {
    match ws {
        crate::state::Workspace::Undistortion => 387.0,
        crate::state::Workspace::Calibration => 520.0,
        crate::state::Workspace::MarkerTracking => 252.0,
    }
}

/// The default docking tree. See the module documentation.
///
/// Shares are in points. The containers between the window and the camera views keep their
/// other children at these sizes and give the camera views the rest
/// ([`super::Shell::fit_central`]), as Qt's dock areas keep their size when the window is
/// resized; the bottom row and the Points column split their space in proportion.
#[must_use]
pub fn default_tree() -> Tree<DockPane> {
    let mut t: Tiles<DockPane> = Tiles::default();
    let tab = |t: &mut Tiles<DockPane>, p: DockPane| {
        let id = t.insert_pane(p);
        t.insert_tab_tile(vec![id])
    };
    let points = tab(&mut t, DockPane::Panel(PanelId::Points));
    let world = tab(&mut t, DockPane::Panel(PanelId::WorldView));
    let progress = tab(&mut t, DockPane::Panel(PanelId::Progress));
    let central = tab(&mut t, DockPane::Central);
    let toolbox = tab(&mut t, DockPane::Panel(PanelId::Toolbox));
    let detail = t.insert_pane(DockPane::Panel(PanelId::DetailedView));
    let events = t.insert_pane(DockPane::Panel(PanelId::Events));
    let display = t.insert_pane(DockPane::Panel(PanelId::DisplayOptions));
    let bottom_tabs = t.insert_tab_tile(vec![detail, events, display]);
    let plot = tab(&mut t, DockPane::Panel(PanelId::Plot));
    let console = tab(&mut t, DockPane::Panel(PanelId::Console));

    let left = t.insert_vertical_tile(vec![points, world]);
    set_shares(&mut t, left, &[(points, 500.0), (world, 300.0)]);
    let middle = t.insert_vertical_tile(vec![progress, central]);
    set_shares(
        &mut t,
        middle,
        &[(progress, PROGRESS_HEIGHT), (central, 600.0)],
    );
    let upper = t.insert_horizontal_tile(vec![left, middle, toolbox]);
    set_shares(
        &mut t,
        upper,
        &[
            (left, POINTS_WIDTH),
            (middle, 890.0),
            (
                toolbox,
                toolbox_width(crate::state::Workspace::MarkerTracking),
            ),
        ],
    );
    let bottom = t.insert_horizontal_tile(vec![bottom_tabs, plot, console]);
    set_shares(
        &mut t,
        bottom,
        &[
            (bottom_tabs, DETAIL_WIDTH),
            (plot, PLOT_WIDTH),
            (console, 400.0),
        ],
    );
    let root = t.insert_vertical_tile(vec![upper, bottom]);
    set_shares(&mut t, root, &[(upper, 600.0), (bottom, BOTTOM_HEIGHT)]);
    Tree::new(egui::Id::unique(TREE_ID), root, t)
}
