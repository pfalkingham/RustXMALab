//! The nine dockable panels of XMALab 3.
//!
//! Each panel is one type in its own file implementing [`Panel`]; [`Panels`] holds one
//! instance of each. The docking shell (`crate::shell`) draws the panel's single title row
//! (or the tab strip it shares with stacked panels), decides where it is and whether it is
//! shown, and calls [`Panel::ui`] with the area below the title and a [`Cx`] context.
//!
//! To write a panel: edit only its file (for example `points.rs`), keep the type name and
//! `Default`, and implement [`Panel`]. Read the project through `cx.doc`, change the
//! interface state through `cx.change(...)`, raise menu-like requests through
//! `cx.actions.push(...)`, change the project through `cx.request(...)`
//! ([`crate::editing::EditRequest`]). See `crates/xma-app/ARCHITECTURE.md`.

pub mod console;
pub mod detail_view;
pub mod display_options;
pub mod events;
pub mod plot;
pub mod points;
pub mod progress;
pub mod toolbox;
pub mod world_view;

use serde::{Deserialize, Serialize};
use xma_engine::Document;

use crate::actions::{Action, ActionQueue};
use crate::editing::{EditQueue, EditRequest};
use crate::icons::Icons;
use crate::jobs::Jobs;
use crate::log::AppLog;
use crate::settings::UiSettings;
use crate::state::{AppState, Changes, ProjectFacts};
use crate::theme::Theme;

/// The nine panels, with the object names of their C++ dock widgets in the docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PanelId {
    /// `WizardDockWidget`, title "Toolbox".
    Toolbox,
    /// `PointsDockWidget`, title "Points".
    Points,
    /// `WorldViewDockWidget`, title "3D world view".
    WorldView,
    /// `DisplayOptionsDockWidget`, title "Display Options".
    DisplayOptions,
    /// `DetailViewDockWidget`, title "Detailed View".
    DetailedView,
    /// `PlotWindow`, title "Plot".
    Plot,
    /// `EventDockWidget`, title "Events".
    Events,
    /// `ConsoleDockWidget`, title "Console".
    Console,
    /// `ProgressDockWidget` (class `ProgressDialog`); its title is the running job's.
    Progress,
}

/// A side of the central area (Qt dock areas).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Side {
    Left,
    Right,
    Top,
    Bottom,
}

pub const ALL_SIDES: [Side; 4] = [Side::Left, Side::Right, Side::Top, Side::Bottom];

impl PanelId {
    pub const ALL: [PanelId; 9] = [
        PanelId::Toolbox,
        PanelId::Points,
        PanelId::WorldView,
        PanelId::DisplayOptions,
        PanelId::DetailedView,
        PanelId::Plot,
        PanelId::Events,
        PanelId::Console,
        PanelId::Progress,
    ];

    /// The dock widget's window title (the `.ui` form's `windowTitle`).
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            PanelId::Toolbox => "Toolbox",
            PanelId::Points => "Points",
            PanelId::WorldView => "3D world view",
            PanelId::DisplayOptions => "Display Options",
            PanelId::DetailedView => "Detailed View",
            PanelId::Plot => "Plot",
            PanelId::Events => "Events",
            PanelId::Console => "Console",
            PanelId::Progress => "Progress",
        }
    }

    /// Short name for ids and file names.
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            PanelId::Toolbox => "toolbox",
            PanelId::Points => "points",
            PanelId::WorldView => "world_view",
            PanelId::DisplayOptions => "display_options",
            PanelId::DetailedView => "detailed_view",
            PanelId::Plot => "plot",
            PanelId::Events => "events",
            PanelId::Console => "console",
            PanelId::Progress => "progress",
        }
    }

    /// Has a close button: the dock features in the `.ui` forms (Toolbox, Points and Progress
    /// are movable and floatable only).
    #[must_use]
    pub fn closable(self) -> bool {
        !matches!(self, PanelId::Toolbox | PanelId::Points | PanelId::Progress)
    }

    /// Qt `allowedAreas`: Toolbox left or right; Console and Progress top or bottom.
    #[must_use]
    pub fn allowed_sides(self) -> &'static [Side] {
        match self {
            PanelId::Toolbox => &[Side::Left, Side::Right],
            PanelId::Console | PanelId::Progress => &[Side::Top, Side::Bottom],
            _ => &ALL_SIDES,
        }
    }

    /// While a job runs, the C++ disables these even for a cancelable job
    /// (ui/ProgressDialog.cpp:96-101).
    #[must_use]
    pub fn disabled_by_cancelable_job(self) -> bool {
        matches!(
            self,
            PanelId::Toolbox | PanelId::Points | PanelId::Plot | PanelId::DetailedView
        )
    }
}

/// Something that happened that panels may want to react to, besides drawing.
#[derive(Clone, Debug, PartialEq)]
pub enum AppEvent {
    /// The state changed (the C++ `State` signals).
    Changed(Changes),
    /// A project finished loading (`MainWindow::loadProjectFinished`).
    ProjectOpened,
    /// The project was closed (`MainWindow::closeProject`).
    ProjectClosed,
}

/// What a panel gets while it draws or handles an event.
pub struct Cx<'a> {
    /// The interface state. Change it through [`Cx::change`] so the change is propagated.
    pub state: &'a mut AppState,
    /// The open project, if any. Read-only: change it with [`Cx::request`].
    pub doc: Option<&'a Document>,
    /// Facts about the open project (calibrated, trials...). Default when none is open.
    pub facts: &'a ProjectFacts,
    /// Per-machine interface settings (Display Options, colours, the panel toggles...).
    pub settings: &'a mut UiSettings,
    /// Raise actions here; they are dispatched after the frame is drawn.
    pub actions: &'a mut ActionQueue,
    /// Requests to change the project ([`Cx::request`]); handled after the actions.
    pub edits: &'a mut EditQueue,
    pub icons: &'a Icons,
    /// The application log (Console panel). Writing to it is allowed from any panel.
    pub log: &'a AppLog,
    /// The running background jobs (Progress panel).
    pub jobs: &'a Jobs,
    /// True when this panel is drawn in its own floating window.
    pub floating: bool,
    /// The frame service: decoded images and the camera and detail views' state.
    pub frames: &'a mut crate::frames::Frames,
    /// Changes made through [`Cx::change`] this frame.
    pub(crate) changes: &'a mut Changes,
}

impl Cx<'_> {
    /// Changes the state and records what changed, so the application runs its handlers
    /// (and other panels get [`AppEvent::Changed`]) after the frame:
    /// `cx.change(|s| s.change_active_camera(Some(1), false))`.
    pub fn change(&mut self, f: impl FnOnce(&mut AppState) -> Changes) -> Changes {
        let ch = f(self.state);
        *self.changes |= ch;
        ch
    }

    /// Raises an action.
    pub fn raise(&mut self, a: Action) {
        self.actions.push(a);
    }

    /// Asks to change the project: `cx.request(EditRequest::Apply(edit))`. The application
    /// applies it after the frame, reports errors, and refreshes everything that shows
    /// project data (`crate::editing`).
    pub fn request(&mut self, r: EditRequest) {
        self.edits.push(r);
    }
}

/// Everything a [`Cx`] borrows, held by the application while it draws, so that a `Cx`
/// can be made for each panel in turn ([`CxParts::cx`]).
pub struct CxParts<'a> {
    pub state: &'a mut AppState,
    pub doc: Option<&'a Document>,
    pub facts: &'a ProjectFacts,
    pub settings: &'a mut UiSettings,
    pub actions: &'a mut ActionQueue,
    pub edits: &'a mut EditQueue,
    /// The theme in use. Not passed on to the panels' [`Cx`]: colours come from
    /// `ui.visuals()`.
    pub theme: &'a Theme,
    pub icons: &'a Icons,
    pub log: &'a AppLog,
    pub jobs: &'a Jobs,
    pub changes: &'a mut Changes,
    pub frames: &'a mut crate::frames::Frames,
}

impl CxParts<'_> {
    /// A context for one panel (or the central area).
    pub fn cx(&mut self, floating: bool) -> Cx<'_> {
        Cx {
            state: &mut *self.state,
            doc: self.doc,
            facts: self.facts,
            settings: &mut *self.settings,
            actions: &mut *self.actions,
            edits: &mut *self.edits,
            icons: self.icons,
            log: self.log,
            jobs: self.jobs,
            floating,
            frames: &mut *self.frames,
            changes: &mut *self.changes,
        }
    }
}

/// A dockable panel.
pub trait Panel {
    /// Which panel this is.
    fn id(&self) -> PanelId;

    /// Draws the panel's contents (everything below its title row) into `ui`, which fills
    /// the panel's area. Called only while the panel is shown.
    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut Cx<'_>);

    /// Reacts to an event (state change, project opened or closed), whether shown or not.
    /// Default: nothing.
    fn on_event(&mut self, _event: &AppEvent, _cx: &mut Cx<'_>) {}

    /// Handles an action the application passes to the panels (a shortcut that acts on a
    /// panel's own state, such as the Plot panel's next frame above the error threshold).
    /// Returns true when handled. Default: not handled.
    fn on_action(&mut self, _action: &Action, _cx: &mut Cx<'_>) -> bool {
        false
    }

    /// The title, when it is not the fixed [`PanelId::title`] (the Progress panel shows
    /// the running job's description, as `ProgressDialog::showProgressbar` sets the window
    /// title). Default: `None`.
    fn dynamic_title(&self, _cx: &Cx<'_>) -> Option<String> {
        None
    }

    /// The width in points the panel needs to show its controls' texts in full in the
    /// style of `ui` (egui's font): the docked panel is kept at least this wide
    /// (`Shell::apply_min_widths`). Default: `None` (no minimum).
    fn min_width(&self, _ui: &egui::Ui, _cx: &Cx<'_>) -> Option<f32> {
        None
    }
}

/// One instance of each panel.
pub struct Panels {
    items: Vec<Box<dyn Panel>>,
}

impl Default for Panels {
    fn default() -> Self {
        Self::new()
    }
}

impl Panels {
    /// Every panel, in [`PanelId::ALL`] order.
    #[must_use]
    // The placeholders are unit structs now; `default()` keeps this list unchanged when
    // their packages give them fields.
    #[allow(clippy::default_constructed_unit_structs)]
    pub fn new() -> Self {
        let items: Vec<Box<dyn Panel>> = vec![
            Box::new(toolbox::ToolboxPanel::default()),
            Box::new(points::PointsPanel::default()),
            Box::new(world_view::WorldViewPanel::default()),
            Box::new(display_options::DisplayOptionsPanel::default()),
            Box::new(detail_view::DetailViewPanel::default()),
            Box::new(plot::PlotPanel::default()),
            Box::new(events::EventsPanel::default()),
            Box::new(console::ConsolePanel::default()),
            Box::new(progress::ProgressPanel::default()),
        ];
        debug_assert!(items.iter().map(|p| p.id()).eq(PanelId::ALL));
        Self { items }
    }

    /// Replaces a panel (for tests, or to swap an implementation).
    pub fn replace(&mut self, panel: Box<dyn Panel>) {
        let id = panel.id();
        if let Some(slot) = self.items.iter_mut().find(|p| p.id() == id) {
            *slot = panel;
        }
    }

    pub fn get_mut(&mut self, id: PanelId) -> &mut dyn Panel {
        let i = PanelId::ALL.iter().position(|p| *p == id).unwrap_or(0);
        self.items[i].as_mut()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn Panel>> {
        self.items.iter_mut()
    }
}
