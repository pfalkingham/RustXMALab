//! The application: the main window, one frame at a time.
//!
//! Ported from ui/MainWindow.cpp (the constructor, the slots connected to `State`, the
//! menu handlers, `loadProject`/`loadProjectFinished`/`UndistortionAfterloadProjectFinished`,
//! `closeProject`, `closeEvent`). Each frame, [`XmaApp::ui`]:
//!
//! 1. applies the theme, collects a finished project load, steps playback;
//! 2. turns key presses into actions (unless a modal dialog is open);
//! 3. draws the menu bar, the workspace bar, and the docking shell (panels and central
//!    area), then the floating windows and the modal dialog;
//! 4. dispatches the queued actions ([`XmaApp::dispatch`], the one place an action does
//!    something), then the queued edit requests ([`XmaApp::apply_request`], the one place
//!    the project is changed), then one step of a running tracking run;
//! 5. notices a change of the document's revision and brings what shows project data up to
//!    date ([`XmaApp::document_changed`]);
//! 6. runs the handlers for what changed in the state ([`XmaApp::handle_changes`], the C++
//!    slots connected to the `State` signals) and passes [`AppEvent`]s to the panels;
//! 7. collects a finished save or autosave, starts a due autosave, sets the window title
//!    (with `*` for unsaved changes), handles the screenshot and quit-after options and the
//!    close request.
//!
//! Which panels are shown is computed each frame from the state ([`panel_wanted`]), the
//! rules `MainWindow::workspaceChanged` applies with `show()` and `hide()`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::actions::{Action, ActionContext, ActionQueue, collect_shortcuts};
use crate::dialogs::windows::{ToolWindow, welcome_due};
use crate::dialogs::{Dialog, DialogResult, Dialogs, IntPurpose, SaveChoice};
use crate::editing::{EditQueue, TrackingRun};
use crate::saving::{Autosave, FileChooser, SaveJob, SaveKind};

mod blender;
mod calibration;
pub mod calibration_ext;
pub mod create;
mod detection;
mod document;
mod images;
pub mod long_edit;
pub mod portal;
mod trials;
mod undistort;
use document::{camera_ids, trial_ranges};

/// The question `MainWindow::closeEvent` asks when the `ConfirmQuitXMALab` setting is on.
pub const QUIT_TEXT: &str = "You are about to quit RustXMALab. Please make sure your data have been saved. \nAre you sure you want to quit?";
/// "Save" answered in "Save changes?" while saving is not possible (a job that disables
/// the window runs).
pub const SAVE_REFUSED_TEXT: &str = "The dataset cannot be saved while the current computation runs. Nothing was saved or closed; try again when it has finished.";
use crate::frame_bar;
use crate::icons::Icons;
use crate::jobs::Jobs;
use crate::log::{AppLog, LogLevel, html_log_to_entries};
use crate::panels::{AppEvent, CxParts, PanelId, Panels};
use crate::session::{LoadResult, Session, project_name, window_title};
use crate::settings::UiSettings;
use crate::shell::{Shell, ShellDraw};
use crate::state::{AppState, Changes, ProjectFacts, Workspace, marker_status};
use crate::theme::{Theme, ThemeChoice};
use xma_model::MarkerStatus;

/// Start-up options (the command line).
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Project to open at start-up.
    pub project: Option<PathBuf>,
    /// Ignore the saved docking layout.
    pub reset_layout: bool,
    /// Theme for this run (also stored as the `Theme` setting unless taking a screenshot).
    pub theme: Option<ThemeChoice>,
    /// Write a screenshot of the main window here shortly before `quit_after`.
    pub screenshot: Option<PathBuf>,
    /// Close the window after this many seconds.
    pub quit_after: Option<f64>,
    /// Switch to this workspace once the project is loaded.
    pub workspace: Option<Workspace>,
    /// Where the settings live; `None` keeps them in memory only (tests).
    pub settings_path: Option<PathBuf>,
    /// Load the start-up project on the interface thread before the first frame (tests).
    pub load_blocking: bool,
    /// File dialogs are the operating system's (the program); otherwise every file or
    /// folder question is answered Cancel unless a test scripts answers
    /// (`FileChooser::Scripted`), so that no test opens a dialog on the desktop.
    pub native_file_dialogs: bool,
    /// Clicked links open in the system's browser or mail program (the program); otherwise
    /// their addresses are only recorded ([`crate::links::UrlOpener::Record`]), so that no
    /// test starts a browser.
    pub system_links: bool,
}

/// The application. Public fields are for tests and for the panels' packages to inspect.
pub struct XmaApp {
    pub state: AppState,
    pub settings: UiSettings,
    settings_path: Option<PathBuf>,
    /// The settings have changed since they were last saved.
    pub(crate) settings_dirty: bool,
    /// When the settings file was last written (`None`: not yet).
    settings_saved_at: Option<std::time::Instant>,
    /// How many times the settings file has been written, or a write tried (tests).
    pub settings_writes: u64,
    pub session: Session,
    pub facts: ProjectFacts,
    pub shell: Shell,
    pub panels: Panels,
    pub actions: ActionQueue,
    /// Requests to change the project, handled after the actions (`crate::editing`).
    pub edits: EditQueue,
    /// The tracking run in progress, advanced one step per frame.
    pub tracking: Option<TrackingRun>,
    /// The save running on a worker thread.
    save_job: Option<SaveJob>,
    /// `Document::changes` when the running save copied the document: with the revision,
    /// whether anything changed while it ran (`save_done`).
    save_changes: u64,
    /// The action to carry out once the running save succeeds ("Save" in "Save changes?").
    after_save: Option<Action>,
    /// "Don't Save" was answered for the document at this revision: its recovery file is
    /// removed when the document is actually closed in that state (`close_project`, or
    /// quitting), not when the answer is given, so that a cancelled next step (a file
    /// dialog, the camera count) keeps it (ledger U61).
    discard_recovery: Option<u64>,
    /// Counts the documents shown: moves whenever the open document is closed or replaced
    /// (not when a save puts its saved copy in place). Work started on a document checks it
    /// before changing the project when it ends ("Denoise Images" relink).
    doc_generation: u64,
    /// Autosave to the recovery file (`crate::saving`).
    pub autosave: Autosave,
    /// How the Save As and Save subdataset as file dialogs are answered (tests script them).
    pub file_chooser: FileChooser,
    /// Who opens the addresses of clicked links (`crate::links`; tests record them).
    pub links: crate::links::UrlOpener,
    /// The autosave timer runs (not for tests and screenshot runs, which keep no settings
    /// file; `XmaApp::autosave_now` works either way).
    pub autosave_timer: bool,
    /// The document's [`xma_engine::Document::changes`] everything shown was last brought
    /// up to date with.
    seen_revision: Option<u64>,
    /// Per trial, the model's (start frame, end frame, number of frames) at that time.
    seen_trials: Vec<(usize, usize, usize)>,
    /// The document was modified at the last frame (the autosave interval starts when it
    /// becomes modified).
    was_modified: bool,
    pub theme: Theme,
    applied_theme: Option<ThemeChoice>,
    pub icons: Icons,
    pub log: AppLog,
    pub jobs: Jobs,
    pub dialogs: Dialogs,
    title: String,
    window_title_sent: String,
    pending_workspace: Option<Workspace>,
    screenshot: Option<PathBuf>,
    screenshot_requested: bool,
    quit_after: Option<f64>,
    quit_confirmed: bool,
    started: bool,
    /// View > Reset layout from a maximised or full-screen main window: the default size
    /// and position still to be sent again once the window has left that state
    /// ([`main_reset_step`], parity ledger U165).
    main_reset: Option<MainWindowReset>,
    /// Changes not yet handled.
    changes: Changes,
    /// Decoded images for the camera and detail views, and their view state.
    pub frames: crate::frames::Frames,
    /// The polynomial refinement in progress (E21), advanced one frame per frame.
    pub refine: Option<crate::editing::RefineRun>,
    /// View > DetectionSettings.
    pub detection: crate::dialogs::detection_settings::DetectionSettingsWindow,
    /// A new dataset being created on a worker thread (`app/create.rs`).
    create: Option<create::CreateJob>,
    /// "Load XMA Portal file" running ([`portal`]).
    portal_trial: Option<portal::PortalTrialJob>,
    /// File > Import > 2D Points waiting for the Progress panel to show "Importing points"
    /// (`app/trials.rs`).
    pub import_points: Option<trials::PendingImport>,
    /// "optimize camera setup" and the external-calibration frame (`app/calibration_ext.rs`).
    pub calibration_ext: calibration_ext::CalibrationExt,
    /// A long edit (undistortion, calibration) running on a worker thread
    /// (`app/long_edit.rs`).
    long_edit: Option<long_edit::LongEdit>,
    /// File > Undistort sequence running on a worker thread (`app/undistort.rs`).
    undistort: Option<undistort::UndistortRun>,
    /// Undistorted Trial images, All Trials for External and "Denoise Images" running on
    /// worker threads (`app/images.rs`).
    images: images::ImagesRuns,
    /// Where File > Export > Blender scene looks for Blender: `None` searches
    /// (`xma_engine::blender_scene::locate_blender`); a path is taken as Blender's program
    /// (tests point it at nothing to see the dialog without Blender).
    pub blender_search: Option<PathBuf>,
    /// The reference indices set by hand ("CUSTOM INDEX", C14), per reference: kept until
    /// the project is closed and put back after every `setDialog` the engine runs
    /// (`app/calibration.rs`, `restore_custom_references`).
    custom_references: [Option<usize>; 4],
    /// Per camera, the identity of its shared data when the views last cached it
    /// (`document_changed`).
    seen_cameras: Vec<usize>,
    /// When `Some`, every edit request handled is recorded here, in order (tests).
    pub request_log: Option<Vec<crate::editing::EditRequest>>,
    /// Per (trial index, rigid body index), the last Console line written for its capped
    /// frames (ledger A210): an identical line is not written again
    /// (`write_capped_line`). Emptied when a project is shown or closed and on undo and
    /// redo.
    capped_lines_written: std::collections::HashMap<(usize, usize), String>,
}

impl XmaApp {
    /// Creates the application (`MainWindow::MainWindow`).
    #[must_use]
    pub fn new(opts: Options) -> Self {
        let log = AppLog::new();
        let mut settings = match &opts.settings_path {
            Some(p) => {
                let (s, err) = UiSettings::load(p);
                if let Some(e) = err {
                    log.error(e);
                }
                s
            }
            None => UiSettings::default(),
        };
        if let Some(t) = opts.theme {
            settings.theme = t.as_setting().to_owned();
        }
        settings.check_recent_files();
        let mut shell = if opts.reset_layout {
            Shell::new_default()
        } else {
            match settings.layout.as_ref().map(Shell::from_saved) {
                Some(Ok(s)) => s,
                Some(Err(e)) => {
                    log.error(format!("Saved panel layout ignored: {e}"));
                    Shell::new_default()
                }
                None => Shell::new_default(),
            }
        };
        shell.set_locked(settings.lock_ui_panes);
        let theme = Theme::new(ThemeChoice::from_setting(&settings.theme));
        let mut app = Self {
            state: AppState::default(),
            settings,
            // Screenshot runs leave the user's settings alone.
            settings_path: if opts.screenshot.is_some() {
                None
            } else {
                opts.settings_path.clone()
            },
            settings_dirty: false,
            settings_saved_at: None,
            settings_writes: 0,
            session: Session::default(),
            facts: ProjectFacts::default(),
            shell,
            panels: Panels::new(),
            actions: ActionQueue::default(),
            edits: EditQueue::default(),
            tracking: None,
            save_job: None,
            save_changes: 0,
            after_save: None,
            discard_recovery: None,
            doc_generation: 0,
            autosave: Autosave::default(),
            file_chooser: if opts.native_file_dialogs {
                FileChooser::Native
            } else {
                FileChooser::Scripted(Default::default())
            },
            links: if opts.system_links {
                crate::links::UrlOpener::System
            } else {
                crate::links::UrlOpener::default()
            },
            autosave_timer: opts.settings_path.is_some() && opts.screenshot.is_none(),
            seen_revision: None,
            seen_trials: Vec::new(),
            was_modified: false,
            theme,
            applied_theme: None,
            icons: Icons::default(),
            log,
            jobs: Jobs::new(),
            dialogs: Dialogs::default(),
            title: window_title(None),
            window_title_sent: String::new(),
            pending_workspace: opts.workspace,
            screenshot: opts.screenshot.clone(),
            screenshot_requested: false,
            quit_after: opts.quit_after,
            quit_confirmed: false,
            started: false,
            main_reset: None,
            changes: Changes::NONE,
            frames: crate::frames::Frames::new(),
            refine: None,
            detection: crate::dialogs::detection_settings::DetectionSettingsWindow::new(),
            create: None,
            portal_trial: None,
            import_points: None,
            calibration_ext: calibration_ext::CalibrationExt::default(),
            long_edit: None,
            undistort: None,
            images: images::ImagesRuns::default(),
            blender_search: None,
            custom_references: [None; 4],
            seen_cameras: Vec::new(),
            request_log: None,
            capped_lines_written: std::collections::HashMap::new(),
        };
        // MainWindow::MainWindow: the Welcome dialog once per version (not for tests, which
        // keep no settings file, nor in screenshot mode).
        if opts.settings_path.is_some()
            && opts.screenshot.is_none()
            && welcome_due(&app.settings.welcome_dialog, env!("CARGO_PKG_VERSION"))
        {
            app.dialogs.open_window(ToolWindow::Welcome, None);
        }
        if let Some(p) = &opts.project {
            if opts.load_blocking {
                app.begin_load(p);
                let disable = app.settings.disable_image_search;
                let r = app.session.load_blocking(p, &mut app.jobs, disable);
                app.project_loaded(r);
            } else {
                app.actions.push(Action::OpenPath(p.clone()));
            }
        }
        app
    }

    /// The window title (`MainWindow::updateWindowTitle`).
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    /// The open document.
    #[must_use]
    pub fn doc(&self) -> Option<&xma_engine::Document> {
        self.session.doc.as_ref()
    }

    fn action_context(&self) -> ActionContext<'_> {
        ActionContext {
            state: &self.state,
            facts: self.session.doc.as_ref().map(|_| &self.facts),
            settings: &self.settings,
            busy: self.jobs.blocks_window() || self.save_job.is_some(),
        }
    }

    /// Is an action enabled now?
    #[must_use]
    pub fn action_enabled(&self, a: &Action) -> bool {
        a.enabled(&self.action_context())
    }

    /// The check mark of an action in its menu, if it has one.
    #[must_use]
    pub fn action_checked(&self, a: &Action) -> Option<bool> {
        a.checked(&self.action_context())
    }

    /// The text of an action in its menu ("Undo <step>").
    #[must_use]
    pub fn action_label(&self, a: &Action) -> String {
        a.menu_label(&self.action_context())
    }

    /// Which panels are shown now.
    #[must_use]
    pub fn shown_panels(&self) -> BTreeMap<PanelId, bool> {
        let facts = self.session.doc.as_ref().map(|_| &self.facts);
        PanelId::ALL
            .into_iter()
            .map(|p| {
                (
                    p,
                    panel_wanted(p, &self.state, facts, &self.settings, &self.jobs),
                )
            })
            .collect()
    }

    /// View > Reset layout (parity ledger U165): the layout `--reset-layout` starts with
    /// ([`Shell::reset_layout`], built by the same [`Shell::new_default`]), at once. The main
    /// window gets the geometry a start without a saved one gives
    /// ([`crate::shell::DEFAULT_MAIN_SIZE`] at [`crate::shell::DEFAULT_MAIN_POS`], out of
    /// full screen and maximised), through viewport commands, since its builder applies only
    /// when the window is created; a platform that does not let a program place its windows
    /// (Wayland) ignores the position. The settings are saved with the new layout. No
    /// project change, not an undo step (a window-layout action, like docking a panel).
    ///
    /// A window that reports maximised or full screen now does not take the size at once
    /// everywhere (Wayland keeps the size it had before maximising; X11 leaving full screen
    /// ends one title bar short), so the size and position are sent again once it has left
    /// that state ([`XmaApp::step_main_reset`]). A second reset starts that wait afresh.
    pub fn reset_layout(&mut self, ctx: &egui::Context) {
        self.shell.reset_layout();
        let root = egui::ViewportId::ROOT;
        let (maximized, fullscreen) =
            ctx.input_for(root, |i| (i.viewport().maximized, i.viewport().fullscreen));
        self.main_reset =
            (maximized == Some(true) || fullscreen == Some(true)).then(|| MainWindowReset {
                since: ctx.input_for(root, |i| i.time),
                size: ctx.input_for(root, |i| i.content_rect().size()),
            });
        ctx.send_viewport_cmd_to(root, egui::ViewportCommand::Fullscreen(false));
        ctx.send_viewport_cmd_to(root, egui::ViewportCommand::Maximized(false));
        ctx.send_viewport_cmd_to(
            root,
            egui::ViewportCommand::InnerSize(crate::shell::DEFAULT_MAIN_SIZE.into()),
        );
        ctx.send_viewport_cmd_to(
            root,
            egui::ViewportCommand::OuterPosition(crate::shell::DEFAULT_MAIN_POS.into()),
        );
        self.settings_dirty = true;
        ctx.request_repaint();
    }

    /// One frame of the wait [`XmaApp::reset_layout`] starts for a maximised or full-screen
    /// main window ([`main_reset_step`]). While it waits, the main window's geometry the
    /// shell recorded this frame is the interim one and is not kept (`None`: a start
    /// without a saved geometry gives the default); once the size has been sent again, or
    /// the wait given up, the shell records the real geometry every frame, as always.
    fn step_main_reset(&mut self, ctx: &egui::Context) {
        let Some(r) = self.main_reset else {
            return;
        };
        let root = egui::ViewportId::ROOT;
        let (now, maximized, fullscreen, size) = ctx.input_for(root, |i| {
            (
                i.time,
                i.viewport().maximized,
                i.viewport().fullscreen,
                i.content_rect().size(),
            )
        });
        self.shell.main_window = None;
        match main_reset_step(&r, now, maximized, fullscreen, size) {
            MainResetStep::Wait => ctx.request_repaint(),
            MainResetStep::Apply => {
                ctx.send_viewport_cmd_to(
                    root,
                    egui::ViewportCommand::InnerSize(crate::shell::DEFAULT_MAIN_SIZE.into()),
                );
                ctx.send_viewport_cmd_to(
                    root,
                    egui::ViewportCommand::OuterPosition(crate::shell::DEFAULT_MAIN_POS.into()),
                );
                self.main_reset = None;
            }
            MainResetStep::GiveUp => self.main_reset = None,
        }
    }

    /// Whether View > Reset layout is still waiting to give the main window its default
    /// size ([`XmaApp::step_main_reset`]); for tests.
    #[must_use]
    pub fn main_reset_pending(&self) -> bool {
        self.main_reset.is_some()
    }

    /// Saves the settings (with the layout) if they have a file.
    pub fn save_settings(&mut self) {
        self.settings.layout = Some(self.shell.to_saved());
        if let Some(p) = &self.settings_path {
            if let Err(e) = self.settings.save(p) {
                self.log
                    .error(format!("Cannot save settings to {}: {e}", p.display()));
            }
            self.settings_writes += 1;
        }
        self.settings_saved_at = Some(std::time::Instant::now());
        self.settings_dirty = false;
    }

    /// Saves changed settings at the end of a frame: at once when the change is over (no
    /// pointer button or key held, no widget dragged), when a modal dialog closed this
    /// frame, or when the last save is a second old or more; otherwise (a value dragged)
    /// a frame is asked for when that second is up. Quitting saves in any case
    /// ([`XmaApp::shutdown`]).
    fn save_settings_when_due(&mut self, ctx: &egui::Context, dialog_was_open: bool) {
        if !self.settings_dirty {
            return;
        }
        let since = self.settings_saved_at.map(|t| t.elapsed());
        let due = since.is_none_or(|d| d >= SETTINGS_SAVE_INTERVAL)
            || (dialog_was_open && !self.dialogs.is_open())
            || !interaction_in_progress(ctx);
        if due {
            self.save_settings();
        } else if let Some(d) = since {
            ctx.request_repaint_after(SETTINGS_SAVE_INTERVAL.saturating_sub(d));
        }
    }

    // -----------------------------------------------------------------------------------
    // The frame.

    /// Draws one frame of the main window into `ui` (the root viewport).
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        // The rest of a wheel notch a drop-down took in an earlier frame (`qt::combo_wheel`).
        crate::qt::combo_wheel_begin_frame(&ctx);
        if !self.started {
            self.started = true;
            // A press anywhere outside a text field ends the typing in it and gives the
            // keys back to the shortcuts: egui's default waits for a click, so a drag (a
            // pan in a camera view, the frame slider) left the keyboard with the field
            // (ledger U177).
            ctx.options_mut(|o| {
                o.input_options.surrender_focus_on = egui::SurrenderFocusOn::Presses;
            });
            self.jobs.set_repaint_context(ctx.clone());
            self.frames.set_repaint_context(&ctx);
        }
        self.apply_theme(&ctx);

        // A finished load, save or autosave.
        if let Some(r) = self.session.poll(&mut self.jobs) {
            self.project_loaded(r);
        }
        if let Some(r) = self.session.take_portal_project() {
            self.portal_project_loaded(r);
        }
        self.poll_portal_trial();
        self.poll_save();
        self.poll_autosave();
        self.poll_create();
        self.poll_long_edit();
        self.poll_undistort();
        self.poll_images();
        // Playback (SequenceNavigationFrame::play_update on its timer).
        if self.state.playback != crate::state::Playback::Stopped
            && !self.jobs.blocks_window()
            && self.frames.ready_for_playback()
        {
            let ch = frame_bar::play_tick(&mut self.state, &self.facts);
            self.changes |= ch;
            ctx.request_repaint();
        }
        // Keyboard shortcuts.
        if !self.dialogs.is_open() {
            let acx = self.action_context();
            for a in collect_shortcuts(&ctx, &acx) {
                self.actions.push(a);
            }
        }

        self.frames.sync(
            self.session.doc.as_ref(),
            &self.state,
            &self.session,
            &self.settings,
            &self.log,
        );
        let shown = self.shown_panels();
        self.shell.set_workspace(self.state.workspace);
        self.shell.sync(&shown);
        // Settings a panel writes directly (Display Options, Show marker status) are saved
        // like those the actions change. The docking layout is left out of the copy: it is
        // taken from the shell when the settings are saved, and nothing writes it while
        // drawing.
        let dialog_was_open = self.dialogs.is_open();
        let layout = self.settings.layout.take();
        let settings_before = self.settings.clone();
        self.draw(ui);
        if self.settings != settings_before {
            self.settings_dirty = true;
        }
        if self.settings.layout.is_none() {
            self.settings.layout = layout;
        }
        self.step_main_reset(&ctx);

        let mut denv = crate::dialogs::DialogEnv {
            icons: &self.icons,
            settings: &mut self.settings,
            doc: self.session.doc.as_ref(),
            files: &mut self.file_chooser,
        };
        if let Some(r) = self.dialogs.show(&ctx, &mut denv) {
            self.dialog_result(r);
        }
        if self.dialogs.take_settings_changed() {
            self.settings_dirty = true;
        }
        self.detection_settings_frame(&ctx);

        // Dispatch until nothing new is raised (an action may raise others).
        for _ in 0..8 {
            let queued = self.actions.take();
            if queued.is_empty() {
                break;
            }
            for a in queued {
                self.dispatch(a, &ctx);
            }
        }
        self.handle_edit_requests();
        if self.import_points.is_some() {
            self.step_import_points();
            ctx.request_repaint();
        }
        if self.tracking.is_some() {
            self.step_tracking();
            ctx.request_repaint();
        }
        if self.refine.is_some() {
            self.step_refine();
            ctx.request_repaint();
        }
        if self.detection.dirty {
            self.update_detection_preview();
        }
        self.sync_document();
        let ch = std::mem::replace(&mut self.changes, Changes::NONE);
        self.handle_changes(ch);
        // The engine's session follows the window (with the calibration page's setDialog
        // when the active camera, calibration frame or workspace changed).
        self.state_to_session();
        self.autosave_tick(&ctx);
        self.title = self.compute_title();

        for m in self.shell.take_messages() {
            self.log.error(m);
        }
        // Clicked links (`crate::links`): those of the main window, and those the immediate
        // viewports collected while they were drawn.
        crate::links::collect(&ctx);
        for url in crate::links::take(&ctx) {
            self.links.open(&url, &self.log);
        }
        // Ledger A210: the Console lines the pose computations of this frame left in the
        // document (an edit, opening, a trial update, a point placed, a tracking step, an
        // export...); the only route by which they reach the Console. Not while a tracking
        // run or the refinement is open: their frames are noted in the document and written
        // when the run ends, one line per body listing them all, rather than one line per
        // frame stepped. Not while a long edit runs: its copy of the document holds the
        // lines pending when it started, and replaces the document when it succeeds (when it
        // fails, the document keeps them), so each is written once.
        if self.tracking.is_none()
            && self.refine.is_none()
            && self.long_edit.is_none()
            && self.write_pending_console_lines()
        {
            ctx.request_repaint();
        }
        self.save_settings_when_due(&ctx, dialog_was_open);
        if self.title != self.window_title_sent {
            self.window_title_sent = self.title.clone();
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(self.title.clone()));
        }
        self.close_request(&ctx);
        self.screenshot_and_quit(&ctx);
        if self.session.is_loading() || self.jobs.is_busy() || self.autosave.is_running() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }

    fn draw(&mut self, ui: &mut egui::Ui) {
        let busy_window = self.jobs.blocks_window();
        let busy_cancelable = self.jobs.is_busy() && !busy_window;
        // As for the shortcuts of the main window (`XmaApp::ui`, `action_context`).
        let shortcuts = !self.dialogs.is_open();
        let busy_actions = busy_window || self.save_job.is_some();
        let acx_menu = {
            let mut q = ActionQueue::default();
            let acx = self.action_context();
            egui::Panel::top("menu_bar")
                .frame(egui::Frame::side_top_panel(ui.style()))
                .show(ui, |ui| {
                    if busy_window {
                        ui.disable();
                    }
                    crate::menu::menu_bar(ui, &acx, &mut q);
                });
            q.take()
        };
        for a in acx_menu {
            self.actions.push(a);
        }

        let doc = self.session.doc.as_ref();
        let mut changes = Changes::NONE;
        let mut parts = CxParts {
            state: &mut self.state,
            doc,
            facts: &self.facts,
            settings: &mut self.settings,
            actions: &mut self.actions,
            edits: &mut self.edits,
            theme: &self.theme,
            icons: &self.icons,
            log: &self.log,
            jobs: &self.jobs,
            changes: &mut changes,
            frames: &mut self.frames,
        };
        if doc.is_some() {
            egui::Panel::top("workspace_bar")
                .frame(egui::Frame::side_top_panel(ui.style()))
                .show(ui, |ui| {
                    if busy_window || busy_cancelable {
                        ui.disable();
                    }
                    let mut cx = parts.cx(false);
                    crate::workspace_bar::ui(ui, &mut cx);
                });
        }
        let mut central = |ui: &mut egui::Ui, parts: &mut CxParts<'_>| {
            let mut cx = parts.cx(false);
            crate::central::ui(ui, &mut cx);
        };
        let panels = &mut self.panels;
        let shell = &mut self.shell;
        egui::CentralPanel::default()
            .frame(egui::Frame::central_panel(ui.style()))
            .show(ui, |ui| {
                let mut draw = ShellDraw {
                    panels,
                    parts: &mut parts,
                    busy_window,
                    busy_cancelable,
                    shortcuts,
                    busy_actions,
                    central: &mut central,
                };
                shell.ui(ui, &mut draw);
            });
        self.changes |= changes;
    }

    fn apply_theme(&mut self, ctx: &egui::Context) {
        // Which of egui's two themes is in force (for "Follow System", the operating
        // system's preference, which eframe reports every frame) is egui's to resolve;
        // everything drawn reads its colours from `ui.visuals()`.
        let choice = ThemeChoice::from_setting(&self.settings.theme);
        if self.applied_theme != Some(choice) {
            self.theme = Theme::new(choice);
            self.theme.apply(ctx);
            self.applied_theme = Some(choice);
        }
    }

    // -----------------------------------------------------------------------------------
    // Actions.

    /// Carries out an action (the C++ menu handlers and button slots). Disabled actions
    /// are ignored.
    pub fn dispatch(&mut self, a: Action, ctx: &egui::Context) {
        let (a, confirmed) = match a {
            Action::Confirmed(inner) => (*inner, true),
            a => (a, false),
        };
        if !self.action_enabled(&a) {
            // A refused save starts nothing, so nothing is to follow it (the action stored
            // by "Save" in "Save changes?" would otherwise run after a later save).
            if matches!(a, Action::SaveProject | Action::SaveProjectAs) && self.save_job.is_none() {
                self.after_save = None;
            }
            return;
        }
        // File > New and its variants, Open Dataset with different calibration, the trial
        // buttons (`app/create.rs`).
        if self.dispatch_create(&a, confirmed) {
            return;
        }
        // "optimize camera setup", "Load MayaCam 2.0" (`app/calibration_ext.rs`).
        if self.dispatch_calibration_ext(&a) {
            return;
        }
        match a {
            Action::LoadProject => {
                // XMALab 3 always asks "You are about to close your dataset..."; here only
                // unsaved changes are asked about (ledger U53).
                if self.unsaved_guard(&a, confirmed) {
                    return;
                }
                let start = self.last_dir();
                if let Some(p) = self.file_chooser.open_file(
                    "Select dataset",
                    start.as_deref(),
                    ("Dataset", &["xma", "zip"]),
                ) {
                    self.open_path(&p);
                }
            }
            Action::OpenPath(ref p) => {
                if self.unsaved_guard(&a, confirmed) {
                    return;
                }
                self.open_path(p);
            }
            Action::OpenRecentFile(i) => {
                let Some(p) = self.settings.recent_files.get(i).cloned() else {
                    return;
                };
                // The file is what was clicked: a Save As in "Save changes?" puts its file at
                // the front of the list before this goes on, so the index is not carried.
                let p = PathBuf::from(p);
                if self.unsaved_guard(&Action::OpenPath(p.clone()), confirmed) {
                    return;
                }
                self.open_path(&p);
            }
            Action::CloseProject => {
                if self.unsaved_guard(&a, confirmed) {
                    return;
                }
                self.close_project();
            }
            Action::SaveProject => {
                // C11, C39: pending undistortion or calibration changes are recomputed first.
                if self.pending_guard(&a, confirmed) {
                    return;
                }
                // MainWindow::on_actionSave_Project_triggered: Save As without a file name.
                match self.session.doc.as_ref().and_then(|d| d.path()) {
                    Some(p) => {
                        let p = p.to_path_buf();
                        self.start_save(p, SaveKind::Save);
                    }
                    None => self.dispatch(Action::SaveProjectAs, ctx),
                }
            }
            Action::SaveProjectAs if self.pending_guard(&a, confirmed) => {}
            Action::SaveProjectAs => match self.choose_save_path() {
                Some(p) => self.start_save(p, SaveKind::SaveAs),
                None => self.after_save = None,
            },
            Action::SaveSubdatasetAs => {
                // MainWindow::saveProjectAs(true): the file, then TrialSelectorDialog.
                if let Some(path) = self.choose_save_path() {
                    let names = self.facts.trial_names.clone();
                    self.dialogs.open(Dialog::TrialSelector {
                        path,
                        names,
                        selected: BTreeSet::new(),
                    });
                }
            }
            Action::Undo | Action::Redo => {
                if let Some(doc) = self.session.doc.as_mut() {
                    let _ = if a == Action::Undo {
                        doc.undo()
                    } else {
                        doc.redo()
                    };
                    // Ledger A210: after an undo or a redo the same action writes its
                    // capped-frames line again.
                    self.capped_lines_written.clear();
                }
            }
            Action::ToggleDisplayOptions => {
                self.settings.show_display_options = !self.settings.show_display_options;
                self.settings_dirty = true;
            }
            Action::ToggleDetailedView => {
                self.settings.show_detail_view = !self.settings.show_detail_view;
                self.settings_dirty = true;
            }
            Action::NextAboveError | Action::PrevAboveError => self.panel_action(&a),
            // The C key (`MainWindow::centerViews`).
            Action::CenterViews => {
                if let Some(doc) = self.session.doc.as_ref() {
                    crate::camera_view::center_on_active(&mut self.frames, doc, &self.state);
                }
            }
            // The Plot panel's range edits (E44 to E47, E51, E52): its selection and choices.
            // A confirmed one reaches the panel still wrapped, so that it goes on past its
            // question (E46).
            Action::SetInterpolationRange
            | Action::SetUntrackableRange
            | Action::DeleteAllAboveError
            | Action::SetEventOn
            | Action::SetEventOff => {
                let a = if confirmed {
                    Action::Confirmed(Box::new(a))
                } else {
                    a
                };
                self.panel_action(&a);
            }
            // E13, E14: Marker tracking only (`WizardDockWidget::interpolateActive`).
            Action::InterpolateActive | Action::InterpolateAll => {
                if self.state.workspace == crate::state::Workspace::MarkerTracking
                    && let Some(doc) = self.session.doc.as_ref()
                    && let Some(r) = crate::panels::toolbox::digitization::interpolate_request(
                        &self.state,
                        doc,
                        a == Action::InterpolateAll,
                    )
                {
                    self.edits.push(r);
                }
            }
            Action::TogglePlot => {
                self.settings.show_plot = !self.settings.show_plot;
                self.settings_dirty = true;
            }
            Action::Toggle3DWorldView => {
                self.settings.show_3d_view = !self.settings.show_3d_view;
                self.settings_dirty = true;
            }
            Action::ToggleEvents => {
                self.settings.show_events = !self.settings.show_events;
                self.settings_dirty = true;
            }
            Action::ToggleConsole => {
                self.settings.console = !self.settings.console;
                self.settings_dirty = true;
            }
            Action::LockUiPanes => {
                self.settings.lock_ui_panes = !self.settings.lock_ui_panes;
                self.shell.set_locked(self.settings.lock_ui_panes);
                self.settings_dirty = true;
            }
            Action::Theme(c) => {
                self.settings.theme = c.as_setting().to_owned();
                self.settings_dirty = true;
            }
            Action::PrevFrame => self.changes |= frame_bar::prev(&mut self.state, &self.facts),
            Action::NextFrame => self.changes |= frame_bar::next(&mut self.state, &self.facts),
            Action::GoToFrame(f) => {
                self.changes |= frame_bar::change_frame(&mut self.state, &self.facts, f as i64);
            }
            Action::StopPlayback => {
                frame_bar::stop(&mut self.state);
                // Escape also ends a tracking run (WizardDigitizationFrame: Esc unchecks
                // the running button).
                if let Some(run) = &mut self.tracking {
                    run.tracker.stop();
                }
            }
            // SequenceNavigationFrame::on_toolButtonPlay*_clicked stop tracking first.
            Action::PlayForward => {
                self.stop_tracking_run();
                frame_bar::play(&mut self.state, true);
            }
            Action::PlayBackward => {
                self.stop_tracking_run();
                frame_bar::play(&mut self.state, false);
            }
            // E8, E9: the keys 1 and 2 click their button if it is enabled
            // (`WizardDigitizationFrame::trackPointsShortcut`); a second press of a running
            // continuous button stops it.
            Action::TrackActiveNext
            | Action::TrackActivePrev
            | Action::TrackActiveForwardAll
            | Action::TrackActiveBackwardAll
            | Action::TrackSelectedNext
            | Action::TrackSelectedPrev
            | Action::TrackSelectedForwardAll
            | Action::TrackSelectedBackwardAll => {
                if let Some(b) = crate::tracking_ui::TrackButton::of_action(&a)
                    && let Some(doc) = self.session.doc.as_ref()
                    && let Some(r) =
                        crate::tracking_ui::press(&self.facts, doc, &self.state, b, false)
                {
                    self.edits.push(r);
                }
            }
            // View > DetectionSettings (`MainWindow::on_actionDetectionSettings_triggered`).
            Action::ToggleDetectionSettings => {
                let open = !self.state.detection_settings_open;
                self.set_detection_settings_open(open);
            }
            Action::MoveNFramesBackward => {
                let n = i64::from(self.settings.frame_advance);
                self.changes |= frame_bar::move_n_frames(&mut self.state, &self.facts, -n);
            }
            Action::MoveNFramesForward => {
                let n = i64::from(self.settings.frame_advance);
                self.changes |= frame_bar::move_n_frames(&mut self.state, &self.facts, n);
            }
            Action::SetAndMoveNFramesBackward | Action::SetAndMoveNFramesForward => {
                let purpose = if a == Action::SetAndMoveNFramesBackward {
                    IntPurpose::FrameAdvanceBackward
                } else {
                    IntPurpose::FrameAdvanceForward
                };
                self.dialogs.open(Dialog::GetInt {
                    title: "Set number of frames to advance".into(),
                    label: "Frames".into(),
                    value: self.settings.frame_advance,
                    min: 1,
                    max: 99,
                    purpose,
                });
            }
            Action::SetStartFrame => {
                if let Some(v) = self.state.trial_view() {
                    self.dialogs.open(Dialog::GetInt {
                        title: "Set start frame".into(),
                        label: "frame ".into(),
                        value: v.start_frame as i32,
                        min: 1,
                        max: v.end_frame as i32,
                        purpose: IntPurpose::StartFrame,
                    });
                }
            }
            Action::SetEndFrame => {
                if let Some(v) = self.state.trial_view() {
                    self.dialogs.open(Dialog::GetInt {
                        title: format!("Set end frame (Max : {})", v.n_frames),
                        label: "frame ".into(),
                        value: v.end_frame as i32,
                        min: v.start_frame as i32,
                        max: v.n_frames as i32,
                        purpose: IntPurpose::EndFrame,
                    });
                }
            }
            Action::PrevMarker | Action::NextMarker => {
                self.select_neighbour_marker(a == Action::NextMarker)
            }
            Action::SelectAllPoints => {
                if let (Some(doc), Some(t)) = (self.session.doc.as_ref(), self.state.active_trial)
                    && let Some(trial) = doc.project().trials.get(t)
                {
                    self.state.selection.markers = (0..trial.markers.len()).collect();
                    self.state.selection.bodies = (0..trial.rigid_bodies.len()).collect();
                    self.changes.selection = true;
                }
            }
            Action::GoToLastTrackedFrame => self.go_to_tracked_end(true),
            Action::GoToFirstTrackedFrame => self.go_to_tracked_end(false),
            Action::NextPartiallySetFrame => self.go_to_partially_set(true),
            Action::PrevPartiallySetFrame => self.go_to_partially_set(false),
            Action::ToggleHideAll => {
                // DisplayOptionsDockWidget::toggleHideAll.
                self.settings.trial_draw_hide_all = !self.settings.trial_draw_hide_all;
                self.settings_dirty = true;
            }
            // C11, C39: the pending-changes question before leaving the workspace.
            Action::SetWorkspace(_) if self.pending_guard(&a, confirmed) => {}
            Action::SetWorkspace(w) => self.changes |= self.state.change_workspace(w, false),
            Action::Calib(ref c) => self.dispatch_calib(c, confirmed),
            // Carried out by `dispatch_calibration_ext`, before this match.
            Action::OptimizeCameraSetup => {}
            // DisplayOptionsDockWidget::on_toolButton_VisualFilter_clicked.
            Action::VisualFilterSettings => self.dialogs.open(Dialog::VisualFilter(
                crate::dialogs::images::VisualFilterDialog,
            )),
            Action::SetDisplay(d) => self.changes |= self.state.change_display(d, false),
            Action::SetActiveTrial(t) => {
                self.changes |= self.state.change_active_trial(Some(t), false)
            }
            Action::TrialSettings => {
                // WorkspaceNavigationFrame::on_toolButtonTrialSettings_clicked: the Trial
                // dialog for the active trial (not offered for the Default trial).
                if let (Some(doc), Some(t)) = (self.session.doc.as_ref(), self.state.active_trial)
                    && doc.project().trials.get(t).is_some_and(|tr| !tr.is_default)
                    && let Some(d) = crate::dialogs::trial::TrialDialog::new(doc, t)
                {
                    self.dialogs.open(Dialog::Trial(d));
                }
            }
            Action::DeleteTrial(t) => {
                // TrialDialog::on_pushButton_DeleteTrial_clicked, confirmed: the dialog
                // closes and WorkspaceNavigationFrame deletes the trial (E62).
                self.dialogs.close_trial_dialog();
                self.edits.push(crate::editing::EditRequest::Apply(
                    xma_engine::Edit::DeleteTrials { trials: vec![t] },
                ));
            }
            Action::FloatPanel(p) => {
                let inner = ctx.input(|i| i.viewport().inner_rect);
                self.shell.float(p, inner);
            }
            Action::DockPanel(p) => self.shell.dock(p),
            Action::ResetLayout => self.reset_layout(ctx),
            // The close button (title row, tab, or the floating window's): the panel's View
            // menu entry is unchecked and its setting turned off, so the menu shows it again
            // (`closeEvent` -> `on_action*_triggered(false)` for Console, Detailed View,
            // Display Options, Events, Plot and the 3D world view,
            // ui/WorldViewDockWidget.cpp:202).
            Action::ClosePanel(p) => {
                if let Some(setting) = self.settings.panel_setting_mut(p) {
                    *setting = false;
                    self.settings_dirty = true;
                }
            }
            Action::Settings => self.dialogs.open(Dialog::Settings(Default::default())),
            Action::About => self.dialogs.open(Dialog::About),
            Action::Help => self.dialogs.open_window(ToolWindow::Help, None),
            Action::ProjectMetadata => self
                .dialogs
                .open_window(ToolWindow::Metadata, self.session.doc.as_ref()),
            Action::ProjectOverview => self
                .dialogs
                .open_window(ToolWindow::Overview, self.session.doc.as_ref()),
            Action::Import2DPoints => self.import_2d_points(),
            Action::ImportTrial => self.import_trial(),
            Action::DeleteMultipleTrials => self.delete_multiple_trials(),
            Action::UndistortSequence => self.open_undistort_sequence(),
            // File > Export > Blender scene (new, ledger U176): `app/blender.rs`.
            Action::ExportBlenderScene => self.open_blender_scene(),
            Action::RigidBodySettings(body) => self.open_rigid_body_dialog(body),
            Action::Quit => {
                if self.save_job.is_some() {
                    // Quit once the save in progress has finished, asking as for any quit:
                    // a save of a subdataset leaves the changes unsaved. A quit already
                    // answered ("Save" in "Save changes?") stays answered.
                    if self.after_save != Some(Action::Confirmed(Box::new(Action::Quit))) {
                        self.after_save = Some(Action::Quit);
                    }
                    return;
                }
                if self.quit_after.is_none() && self.unsaved_guard(&a, confirmed) {
                    return;
                }
                // MainWindow::closeEvent: the C++ question when nothing is unsaved.
                if self.settings.confirm_quit_xmalab && !confirmed && self.quit_after.is_none() {
                    self.confirm(QUIT_TEXT.into(), Action::Quit);
                    return;
                }
                self.finish_tracking();
                self.quit_confirmed = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            Action::Points(ref p) => {
                let ch = crate::dialogs::points::start(self, p);
                self.changes |= ch;
            }
            // File > Export and the Points panel's two exports (`dialogs::exports`).
            Action::Export2DPoints
            | Action::Export3DPoints
            | Action::ExportRigidBodyTransformations
            | Action::ExportEvents
            | Action::ExportMarkerToMarkerDistances
            | Action::ExportPrecisionInfo
            | Action::ExportReprojectionErrors
            | Action::ExportMayaCams2
            | Action::ExportUndistortedTrialImages
            | Action::ExportAllTrialsForExternal
            | Action::Export3DPointsSelected
            | Action::ExportTransformationsSelected => {
                use crate::dialogs::exports::{ExportAction as E, start};
                let sel = &self.state.selection;
                let e = match a {
                    Action::Export2DPoints => E::Points2d,
                    Action::Export3DPoints => E::Points3d { markers: None },
                    Action::ExportRigidBodyTransformations => E::Transformations { bodies: None },
                    Action::ExportEvents => E::Events,
                    Action::ExportMarkerToMarkerDistances => E::MarkerDistances,
                    Action::ExportPrecisionInfo => E::PrecisionInfo,
                    Action::ExportReprojectionErrors => E::ReprojectionErrors,
                    Action::ExportMayaCams2 => E::MayaCams,
                    Action::ExportUndistortedTrialImages => E::TrialImages,
                    Action::ExportAllTrialsForExternal => E::AllTrials,
                    // The selected rows, sorted (MarkerTreeWidget sorts them).
                    Action::Export3DPointsSelected => E::Points3d {
                        markers: Some(sel.markers.iter().copied().collect()),
                    },
                    _ => E::Transformations {
                        bodies: Some(sel.bodies.iter().copied().collect()),
                    },
                };
                start(self, &e);
            }
            // WorkspaceNavigationFrame::on_toolButtonCameraSettings_clicked.
            Action::CameraSettings => {
                let names = self.facts.camera_names.clone();
                self.dialogs.open(Dialog::CameraSelector(
                    crate::dialogs::camera_selector::CameraSelector::new(
                        names,
                        &self.state.camera_visible,
                    ),
                ));
            }
            // The Plot panel's two file buttons: the file dialog, then the panel.
            Action::PlotExport => {
                let start = self.last_dir();
                let start = start.map(|d| d.join("PlotData.csv"));
                if let Some(p) = self.file_chooser.save_file_filtered(
                    "Save PlotData",
                    start.as_deref(),
                    ("Comma seperated data", &["csv"]),
                ) {
                    self.panel_action(&Action::PlotExportTo(p));
                }
            }
            Action::PlotExtraData => {
                let start = self.last_dir();
                if let Some(p) = self.file_chooser.open_file(
                    "Select data you want to plot",
                    start.as_deref(),
                    ("Dataset", &["csv"]),
                ) {
                    self.panel_action(&Action::PlotExtraFrom(p));
                }
            }
            // Everything else is not implemented in this version (disabled above).
            _ => {}
        }
    }

    /// Asks "Save changes?" before `then` when the document has unsaved changes and the
    /// action was not confirmed already; returns true when it asked (the caller stops).
    /// "Save" saves and then carries out `then`, "Don't Save" carries it out at once
    /// (`XmaApp::dialog_result`). Not in XMALab 3 (ledger U53). Every action that closes the
    /// document goes through this: Close, Open, Open recent, a file given by the system,
    /// Quit, File > New and its variants, Open Dataset with different calibration.
    pub fn unsaved_guard(&mut self, then: &Action, confirmed: bool) -> bool {
        if confirmed {
            return false;
        }
        // A tracking run or a refinement is one undo step still open, which counts as a
        // change only once committed: it is ended here, so that its points are asked about.
        self.finish_tracking();
        let Some(doc) = &self.session.doc else {
            return false;
        };
        if !doc.is_modified() {
            return false;
        }
        // A new question: no earlier "Don't Save" applies to this state any more.
        self.discard_recovery = None;
        let name = doc
            .path()
            .map_or_else(|| "untitled".to_owned(), project_name);
        self.dialogs.open(Dialog::SaveChanges {
            name,
            then: then.clone(),
        });
        true
    }

    pub(crate) fn confirm(&mut self, message: String, on_yes: Action) {
        // Every confirmation the program has so far (quit, close, load) passes no `NoButton`
        // in the C++: "OK" and "Cancel".
        self.dialogs.open(Dialog::Confirm {
            message,
            on_yes: Action::Confirmed(Box::new(on_yes)),
            buttons: crate::dialogs::ConfirmButtons::OkCancel,
        });
    }

    /// Acts on the answer to a dialog (also called by tests, as the dialog buttons would).
    pub fn dialog_result(&mut self, r: DialogResult) {
        match r {
            DialogResult::Confirmed(a) => self.actions.push(a),
            DialogResult::Int(purpose, v) => match purpose {
                IntPurpose::FrameAdvanceBackward | IntPurpose::FrameAdvanceForward => {
                    // SequenceNavigationFrame::setNFrames, then moveNFrames.
                    self.settings.frame_advance = v;
                    self.settings_dirty = true;
                    let n = i64::from(v);
                    let n = if purpose == IntPurpose::FrameAdvanceBackward {
                        -n
                    } else {
                        n
                    };
                    self.changes |= frame_bar::move_n_frames(&mut self.state, &self.facts, n);
                }
                // E63, E64: the trial's start or end frame is a project edit; the view
                // follows the project, with the frame on screen moved into the new range
                // (`document_changed`, D15 fixed).
                IntPurpose::StartFrame => {
                    if let Some(t) = self.state.active_trial {
                        self.edits.push(crate::editing::EditRequest::Apply(
                            xma_engine::Edit::SetStartFrame {
                                trial: t,
                                frame: usize::try_from(v).unwrap_or(0),
                            },
                        ));
                    }
                }
                IntPurpose::ExternalCameras => self.create_external(v),
                IntPurpose::OptimizeIterations => {
                    let (m, n, f) = crate::dialogs::optimization::prompt_parameters(v);
                    self.apply_optimisation(m, n, f);
                }
                IntPurpose::GridBorder
                | IntPurpose::GridCircle { .. }
                | IntPurpose::ReferenceIndex(_) => self.calibration_int(purpose, v),
                IntPurpose::EndFrame => {
                    if let Some(t) = self.state.active_trial {
                        self.edits.push(crate::editing::EditRequest::Apply(
                            xma_engine::Edit::SetEndFrame {
                                trial: t,
                                frame: usize::try_from(v).unwrap_or(0),
                            },
                        ));
                    }
                }
            },
            DialogResult::VideoChosen(m, p) => {
                self.log.write(
                    format!("{}: using {}", m.file_name, p.display()),
                    LogLevel::Output,
                );
                self.session.resolve_missing(&m, p);
            }
            DialogResult::SaveChanges(SaveChoice::Save, then) => {
                // Saving may be refused (a job that disables the window runs): then nothing
                // is saved and nothing waits for a save.
                if !self.action_enabled(&Action::SaveProject) {
                    self.dialogs.error(SAVE_REFUSED_TEXT.to_owned());
                    return;
                }
                self.after_save = Some(Action::Confirmed(Box::new(then)));
                self.actions.push(Action::SaveProject);
            }
            DialogResult::SaveChanges(SaveChoice::DontSave, then) => {
                // "Don't Save" discards the changes, and so the recovery file (ledger U61),
                // once the document is closed (`close_project`, `shutdown`).
                self.discard_recovery = self
                    .session
                    .doc
                    .as_ref()
                    .map(xma_engine::Document::revision);
                self.actions.push(Action::Confirmed(Box::new(then)));
            }
            DialogResult::Recover {
                project,
                recovery,
                recovered,
            } => {
                // "Open saved version" discards the recovered changes (ledger U61).
                if !recovered {
                    self.remove_recovery_file(&recovery);
                }
                self.start_open(&project, recovered.then_some(recovery.as_path()));
            }
            DialogResult::TrialProperties { trial, properties } => {
                self.edits.push(crate::editing::EditRequest::Apply(
                    xma_engine::Edit::SetTrialProperties { trial, properties },
                ));
            }
            DialogResult::AlgorithmSettings(patch) => self.set_algorithm_settings(patch),
            DialogResult::TrialsChosen { path, trials } => {
                self.start_save(path, SaveKind::Subset(trials));
            }
            DialogResult::Answered(answer) => {
                for r in answer.requests {
                    self.edits.push(r);
                }
                for a in answer.actions {
                    self.actions.push(a);
                }
            }
            DialogResult::Points(o) => {
                let ch = crate::dialogs::points::answer(self, &o);
                self.changes |= ch;
            }
            DialogResult::Request(r) => self.edits.push(r),
            DialogResult::PointImportExport(kind) => self.point_import_export_accepted(kind),
            DialogResult::TrialsToImport {
                path,
                trials,
                replace_default,
            } => self.import_trials(&path, trials, replace_default),
            DialogResult::TrialsToDelete(trials) => self.delete_trials(&trials),
            DialogResult::NewProject { spec, dialog } => self.create_project(spec, dialog),
            DialogResult::NewTrial(answer) => self.new_trial_answer(answer),
            DialogResult::Export(o) => crate::dialogs::exports::answer(self, &o),
            DialogResult::Optimize {
                method,
                iterations,
                initial,
            } => self.apply_optimisation(method, iterations, initial),
            DialogResult::UndistortSequence(items) => self.start_undistort(items),
            DialogResult::BlenderScene(spec) => self.start_blender_scene(*spec),
            DialogResult::Images(a) => self.images_answer(a),
            DialogResult::CamerasVisible(v) => {
                // CameraSelector::on_pushButton_OK_clicked, then the views laid out again.
                if v.len() == self.state.camera_visible.len() {
                    self.state.camera_visible = v;
                    self.changes.cameras = true;
                }
            }
        }
    }

    // -----------------------------------------------------------------------------------
    // Project.

    pub(crate) fn begin_load(&mut self, path: &Path) {
        // MainWindow::loadProject: remember the folder and the file.
        self.state.loading = true;
        self.settings.set_last_used_directory_from_file(path);
        let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
        self.settings.add_to_recent_files(&abs.to_string_lossy());
        self.settings_dirty = true;
        // A setting the project does not store takes this machine's default.
        self.session.base_settings = self.settings.algorithm_base();
        self.session.portal_workspace = self.settings.portal_workspace();
    }

    /// Opens a project; when its recovery file is newer, asks first whether to open that
    /// (ledger U54).
    fn open_path(&mut self, path: &Path) {
        if let Some(recovery) = crate::saving::newer_recovery(path) {
            // The open project's own recovery file after "Don't Save" (the same file opened
            // again): those changes were discarded, so it is not offered.
            if self.discard_recovery.is_some()
                && self.discard_recovery == self.session.doc.as_ref().map(|d| d.revision())
                && self.recovery_file().as_deref() == Some(recovery.as_path())
            {
                self.remove_recovery_file(&recovery);
                self.start_open(path, None);
                return;
            }
            self.dialogs.open(Dialog::Recover {
                project: path.to_path_buf(),
                recovery,
            });
            return;
        }
        self.start_open(path, None);
    }

    /// Closes the open project and starts loading `path` (from `read_from` instead, for a
    /// recovery file).
    fn start_open(&mut self, path: &Path, read_from: Option<&Path>) {
        if self.session.doc.is_some() {
            self.close_project();
        }
        self.begin_load(path);
        let disable = self.settings.disable_image_search;
        self.session
            .start_load(path, read_from, &mut self.jobs, disable);
    }

    /// What follows a load: `MainWindow::loadProjectFinished`, `setupProjectUI` and
    /// `UndistortionAfterloadProjectFinished`, for the interface.
    pub fn project_loaded(&mut self, r: LoadResult) {
        let LoadResult {
            path,
            result,
            videos,
            missing,
            messages,
        } = r;
        let doc = match result {
            Ok(d) => d,
            Err(e) => {
                self.state.loading = false;
                let msg = format!("Could not open {}: {e}", path.display());
                self.log.error(msg.clone());
                self.dialogs.error(msg);
                return;
            }
        };
        // ConsoleDockWidget::afterLoad: the console shows the log saved in the project.
        let saved_log = doc
            .extras()
            .attachments
            .iter()
            .find(|a| a.name == "log.html")
            .map(|a| html_log_to_entries(&String::from_utf8_lossy(&a.data)))
            .unwrap_or_default();
        self.log.replace_with(saved_log);
        self.log
            .write(format!("Load dataset {}", path.display()), LogLevel::Output);
        for w in doc.extras().warnings.iter().chain(doc.warnings()) {
            self.log.error(w.clone());
        }
        for m in messages {
            self.log.write(m, LogLevel::Output);
        }
        self.show_document(doc, videos, missing);
        // The lines of the load-time update (ledger A210), after the saved log.
        self.write_pending_console_lines();
    }

    /// Writes to the Console the lines the document's pose computations left since the last
    /// call (`Document::take_capped_lines`, ledger A210), through [`XmaApp::write_capped_line`].
    /// Called once per frame (and after a load); the only writer of these lines, since an
    /// edit's outcome does not carry them. Returns whether a line was written. Costs a mutex
    /// lock and an empty list when nothing is pending; changes neither the project nor its
    /// modified state.
    pub fn write_pending_console_lines(&mut self) -> bool {
        let Some(doc) = self.session.doc.as_mut() else {
            return false;
        };
        let lines = doc.take_capped_lines();
        let mut wrote = false;
        for l in lines {
            wrote |= self.write_capped_line(l);
        }
        wrote
    }

    /// Writes a line about capped frames (ledger A210) to the Console (level Output),
    /// unless it is identical to the last one written for the same trial and rigid body
    /// (by index) since the project was shown or the last undo or redo, so that a point
    /// dragged across a capped frame, which recomputes the same frames at every step, does
    /// not fill the Console; a line naming other frames is written (and becomes the last).
    /// Returns whether the line was written.
    pub(crate) fn write_capped_line(&mut self, line: xma_engine::rigid_bodies::CappedLine) -> bool {
        let key = (line.trial, line.body);
        if self.capped_lines_written.get(&key) == Some(&line.text) {
            return false;
        }
        self.capped_lines_written.insert(key, line.text.clone());
        self.log.write(line.text, LogLevel::Output);
        true
    }

    /// The document becomes the open project, and the interface is set up for it
    /// (`MainWindow::setupProjectUI`, `UndistortionAfterloadProjectFinished`): after a load
    /// and after a new dataset is created (`app/create.rs`).
    fn show_document(
        &mut self,
        doc: xma_engine::Document,
        videos: BTreeMap<(usize, usize), crate::session::VideoLocation>,
        missing: Vec<crate::session::MissingVideo>,
    ) {
        self.facts = ProjectFacts::from_doc(&doc);
        self.state.reset_for_project(&doc);
        self.frames.project_opened(&doc);
        let n_trials = doc.project().trials.len();
        let active_trial = doc.project().active_trial;
        self.seen_cameras = camera_ids(&doc);
        self.seen_revision = Some(doc.changes());
        self.seen_trials = trial_ranges(&doc);
        self.was_modified = doc.is_modified();
        self.autosave.reset();
        self.doc_generation += 1;
        self.discard_recovery = None;
        self.capped_lines_written.clear();
        self.session.doc = Some(doc);
        self.session.videos = videos;
        self.title = self.compute_title();

        // ProjectFileIO::loadProject: the stored active trial, else the last one.
        let mut ch = Changes::NONE;
        if n_trials > 0 {
            let t = if active_trial < n_trials {
                active_trial
            } else {
                n_trials - 1
            };
            ch |= self.state.change_active_trial(Some(t), true);
            ch |= self
                .state
                .change_active_frame_trial(Some(self.state.trials[t].active_frame), true);
        }
        // MainWindow::setupProjectUI and UndistortionAfterloadProjectFinished: the active
        // camera and calibration frame, the undistortion state and the workspace, as the
        // document's session has them (`Session::new_project`, `Session::after_load`).
        ch |= self.session_to_state(true);
        self.custom_references = [None; 4];
        self.calibration_page_loaded();
        // The trial update these changes would start already ran in Document::open_with
        // (Document::update_after_load), so they are handled while still "loading".
        self.handle_changes(ch);
        self.state.loading = false;
        self.shell.clear_failures();
        self.notify(&AppEvent::ProjectOpened);

        for m in missing {
            self.dialogs.open(Dialog::MissingVideo(m));
        }
        if let Some(w) = self.pending_workspace.take() {
            let ch = self.state.change_workspace(w, false);
            self.handle_changes(ch);
        }
    }

    /// Ported from ui/MainWindow.cpp, MainWindow::closeProject and tearDownProjectUI.
    pub fn close_project(&mut self) {
        self.finish_tracking();
        self.drop_long_edit();
        self.custom_references = [None; 4];
        self.seen_cameras.clear();
        // The recovery file goes with a project closed without unsaved changes (ledger U54),
        // or after "Don't Save" (ledger U61).
        let discard = self.discard_recovery.take();
        if let Some(doc) = &self.session.doc
            && (!doc.is_modified() || discard == Some(doc.revision()))
        {
            self.remove_recovery(doc.path());
        }
        self.doc_generation += 1;
        self.capped_lines_written.clear();
        self.seen_revision = None;
        self.seen_trials.clear();
        self.was_modified = false;
        self.autosave.reset();
        self.session.close();
        self.frames.project_closed();
        self.facts = ProjectFacts::default();
        let ch = self.state.reset_for_close();
        self.title = window_title(None);
        self.log.clear();
        self.dialogs.windows.project_closed();
        self.handle_changes(ch);
        self.shell.clear_failures();
        // Missing-video requests belong to the closed project.
        self.dialogs.retain_missing(|_| false);
        self.notify(&AppEvent::ProjectClosed);
    }

    // -----------------------------------------------------------------------------------
    // State-change handlers (the slots connected to the State signals).

    /// Runs the handlers for what changed, repeatedly while handlers change more.
    pub fn handle_changes(&mut self, mut ch: Changes) {
        for _ in 0..8 {
            if !ch.any() {
                return;
            }
            let mut more = Changes::NONE;
            if ch.workspace {
                more |= self.on_workspace_changed();
            }
            if ch.active_trial {
                more |= self.on_active_trial_changed();
            }
            self.notify(&AppEvent::Changed(ch));
            ch = more;
        }
    }

    /// MainWindow::workspaceChanged, SequenceNavigationFrame::workspaceChanged and
    /// WorkspaceNavigationFrame::workspaceChanged, for what is not computed per frame.
    fn on_workspace_changed(&mut self) -> Changes {
        // MainWindow: panels shown per workspace from their settings (`panel_wanted`).
        // XMALab 3 also disables Undo here; the undo history here survives workspace changes
        // (ledger U52).
        // SequenceNavigationFrame: stop (playback and tracking); bring the frame into the
        // range.
        frame_bar::stop(&mut self.state);
        self.stop_tracking_run();
        let mut ch = Changes::NONE;
        if self.state.workspace == Workspace::MarkerTracking
            && let Some(v) = self.state.trial_view().copied()
        {
            ch |= frame_bar::change_frame(&mut self.state, &self.facts, v.active_frame as i64);
            // WorkspaceNavigationFrame: update the active trial (not while loading).
            if !self.state.loading
                && let (Some(t), Some(doc)) = (self.state.active_trial, self.session.doc.as_mut())
            {
                doc.update_trial(t);
            }
        }
        ch
    }

    /// MainWindow::activeTrialChanged, SequenceNavigationFrame::activeTrialChanged and
    /// WorkspaceNavigationFrame::activeTrialChanged.
    fn on_active_trial_changed(&mut self) -> Changes {
        let Some(t) = self.state.active_trial else {
            return Changes::NONE;
        };
        frame_bar::stop(&mut self.state);
        self.stop_tracking_run();
        self.state.selection = crate::state::Selection::default();
        if !self.state.loading
            && let Some(doc) = self.session.doc.as_mut()
        {
            doc.select_trial(t);
        }
        let mut ch = Changes::NONE;
        if self.state.workspace == Workspace::MarkerTracking
            && let Some(v) = self.state.trial_view().copied()
        {
            ch |= self
                .state
                .change_active_frame_trial(Some(v.active_frame), false);
        }
        ch
    }

    /// Passes an action to the panels until one handles it (`Panel::on_action`).
    fn panel_action(&mut self, a: &Action) {
        let doc = self.session.doc.as_ref();
        let mut changes = Changes::NONE;
        let mut parts = CxParts {
            state: &mut self.state,
            doc,
            facts: &self.facts,
            settings: &mut self.settings,
            actions: &mut self.actions,
            edits: &mut self.edits,
            theme: &self.theme,
            icons: &self.icons,
            log: &self.log,
            jobs: &self.jobs,
            frames: &mut self.frames,
            changes: &mut changes,
        };
        for p in self.panels.iter_mut() {
            let mut cx = parts.cx(false);
            if p.on_action(a, &mut cx) {
                break;
            }
        }
        self.changes |= changes;
    }

    fn notify(&mut self, ev: &AppEvent) {
        let doc = self.session.doc.as_ref();
        let mut changes = Changes::NONE;
        let mut parts = CxParts {
            state: &mut self.state,
            doc,
            facts: &self.facts,
            settings: &mut self.settings,
            actions: &mut self.actions,
            edits: &mut self.edits,
            theme: &self.theme,
            icons: &self.icons,
            log: &self.log,
            jobs: &self.jobs,
            changes: &mut changes,
            frames: &mut self.frames,
        };
        for p in self.panels.iter_mut() {
            let mut cx = parts.cx(false);
            p.on_event(ev, &mut cx);
        }
        self.changes |= changes;
    }

    // -----------------------------------------------------------------------------------
    // Navigation that reads marker data.

    /// Ported from ui/PointsDockWidget.cpp, selectNextPoint / selectPrevPoint.
    fn select_neighbour_marker(&mut self, next: bool) {
        let (Some(doc), Some(t)) = (self.session.doc.as_ref(), self.state.active_trial) else {
            return;
        };
        let n = doc.project().trials.get(t).map_or(0, |tr| tr.markers.len()) as i64;
        let cur = self
            .state
            .trial_view()
            .and_then(|v| v.active_marker)
            .map_or(-1, |m| m as i64);
        let idx = if next { cur + 1 } else { cur - 1 };
        if idx >= 0 && idx < n {
            self.changes |= self.state.select_marker(Some(idx as usize));
        }
    }

    /// Ported from ui/WizardDigitizationFrame.cpp, goToLastTrackedFrame (`last`) and
    /// goToFirstTrackedFrame. Where the C++ goes to the end frame itself (one past the last
    /// frame, `changeActiveFrameTrial(getEndFrame())`), this goes to the last frame.
    fn go_to_tracked_end(&mut self, last: bool) {
        let (Some(doc), Some(t)) = (self.session.doc.as_ref(), self.state.active_trial) else {
            return;
        };
        let Some(v) = self.state.trial_view().copied() else {
            return;
        };
        let Some(m) = v.active_marker else { return };
        let cur = self.state.active_frame_trial.unwrap_or(v.active_frame) as i64;
        let ncam = self.facts.n_cameras;
        let unset =
            |f: i64| (0..ncam).any(|c| (marker_status(doc, t, m, c, f as usize) as i32) <= 0);
        let target = if last {
            let mut target = Some(v.end_frame as i64 - 1);
            let mut i = cur;
            while i < v.end_frame as i64 {
                if unset(i) {
                    target = (i - 1 > cur).then_some(i - 1);
                    break;
                }
                i += 1;
            }
            target
        } else {
            let mut target = Some(v.start_frame as i64 - 1);
            let mut i = cur;
            while i >= v.start_frame as i64 - 1 {
                if unset(i) {
                    target = (i + 1 < cur).then_some(i + 1);
                    break;
                }
                i -= 1;
            }
            target
        };
        if let Some(f) = target {
            self.changes |= self
                .state
                .change_active_frame_trial(Some(f.max(0) as usize), false);
        }
    }

    /// Ported from ui/SequenceNavigationFrame.cpp, moveFrameToMissingForward / Backward:
    /// the next (previous) frame, from the current one, where the active marker is set in
    /// some cameras but not all.
    fn go_to_partially_set(&mut self, forward: bool) {
        if self.state.workspace != Workspace::MarkerTracking {
            return;
        }
        let (Some(doc), Some(t)) = (self.session.doc.as_ref(), self.state.active_trial) else {
            return;
        };
        let Some(v) = self.state.trial_view().copied() else {
            return;
        };
        let Some(m) = v.active_marker else { return };
        let ncam = self.facts.n_cameras;
        let partial = |f: usize| {
            let set = (0..ncam)
                .filter(|c| marker_status(doc, t, m, *c, f) > MarkerStatus::Undefined)
                .count();
            set > 0 && set < ncam
        };
        let mut f = v.active_frame as i64;
        let found = if forward {
            let mut r = None;
            while f < v.end_frame as i64 {
                if partial(f as usize) {
                    r = Some(f);
                    break;
                }
                f += 1;
            }
            r
        } else {
            let mut r = None;
            while f >= v.start_frame as i64 - 1 && f >= 0 {
                if partial(f as usize) {
                    r = Some(f);
                    break;
                }
                f -= 1;
            }
            r
        };
        if let Some(f) = found {
            self.changes |= frame_bar::change_frame(&mut self.state, &self.facts, f);
        }
    }

    // -----------------------------------------------------------------------------------
    // Window.

    /// Ported from ui/MainWindow.cpp, MainWindow::closeEvent: confirmation when the
    /// `ConfirmQuitXMALab` setting is on.
    ///
    /// The window's close button goes through [`Action::Quit`]: "Save changes?" when the
    /// document has unsaved changes, else the C++ question, and a save in progress finishes
    /// first.
    fn close_request(&mut self, ctx: &egui::Context) {
        let requested = ctx.input(|i| i.viewport().close_requested());
        if !requested || self.quit_confirmed || self.quit_after.is_some() {
            return;
        }
        ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
        if !self.dialogs.is_open() {
            self.actions.push(Action::Quit);
            ctx.request_repaint();
        }
    }

    /// What the program does as it exits (eframe's `on_exit`): a tracking run is closed as
    /// one undo step, a running autosave is waited for, and the recovery file is removed when
    /// nothing is unsaved.
    pub fn shutdown(&mut self) {
        self.finish_tracking();
        if let Some(out) = self.autosave.wait() {
            self.autosave_done(out);
        }
        if let Some(doc) = &self.session.doc
            && (!doc.is_modified() || self.discard_recovery == Some(doc.revision()))
        {
            let p = doc.path().map(Path::to_path_buf);
            self.remove_recovery(p.as_deref());
        }
        self.save_settings();
    }

    fn screenshot_and_quit(&mut self, ctx: &egui::Context) {
        let Some(t) = self.quit_after else { return };
        let now = ctx.input(|i| i.time);
        if let Some(path) = self.screenshot.clone()
            && !self.screenshot_requested
            && now > t - 1.0
            && !self.session.is_loading()
        {
            self.screenshot_requested = true;
            ctx.request_screenshot(move |img| {
                let size = img.size;
                let mut rgba = Vec::with_capacity(size[0] * size[1] * 4);
                for p in &img.pixels {
                    rgba.extend_from_slice(&p.to_srgba_unmultiplied());
                }
                match image::RgbaImage::from_raw(size[0] as u32, size[1] as u32, rgba) {
                    Some(im) => match im.save(&path) {
                        Ok(()) => eprintln!("screenshot written: {}", path.display()),
                        Err(e) => eprintln!("screenshot failed: {e}"),
                    },
                    None => eprintln!("screenshot failed: bad image size"),
                }
            });
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
        if now > t && (self.screenshot.is_none() || self.screenshot_requested) {
            self.quit_confirmed = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

/// The wait after View > Reset layout from a maximised or full-screen main window
/// ([`XmaApp::reset_layout`]): when it started (egui's input time, seconds) and the size of
/// the window's content then (points).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MainWindowReset {
    pub since: f64,
    pub size: egui::Vec2,
}

/// What [`main_reset_step`] decides for a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MainResetStep {
    /// The window has not left maximised or full screen yet: nothing is sent.
    Wait,
    /// It has: send the default size and position again, and stop.
    Apply,
    /// It did not leave within [`MAIN_RESET_TIMEOUT`]: stop without sending anything.
    GiveUp,
}

/// How long Reset layout waits for the main window to leave maximised or full screen
/// (seconds of egui's input time). Measured on Linux (parity ledger U165): X11 took 0.07
/// to 0.30 s, Wayland 0.02 s; macOS animates leaving full screen for about a second. A
/// window manager that keeps the window maximised or tiled (a tiling one) is not fought:
/// after this the program stops asking for frames and leaves the window as it is.
pub const MAIN_RESET_TIMEOUT: f64 = 2.0;

/// The decision for one frame of the wait: the window has left when it reports neither
/// maximised nor full screen **and** its content size differs from the size at the reset.
/// The size matters on X11, where the full-screen state reads as left in the very next
/// frame, before the window manager has resized the window; a size sent then is lost. The
/// size and position are sent once only, so a window the user maximises again, moves or
/// resizes after that is left alone; while the window is still maximised or full screen
/// nothing is sent, so a user's own maximising during the wait is not undone either.
/// After [`MAIN_RESET_TIMEOUT`] the answer is always to give up, whatever the window
/// reports (the second review of 21 November 2026).
#[must_use]
pub fn main_reset_step(
    r: &MainWindowReset,
    now: f64,
    maximized: Option<bool>,
    fullscreen: Option<bool>,
    size: egui::Vec2,
) -> MainResetStep {
    // The time limit comes first: when no frame ran for longer than the limit (a hidden
    // window gets none), nothing is sent to a window the user may have arranged since. A
    // time that is not a number ends the wait too.
    let within_limit = now - r.since < MAIN_RESET_TIMEOUT;
    if !within_limit {
        MainResetStep::GiveUp
    } else if maximized != Some(true) && fullscreen != Some(true) && size != r.size {
        MainResetStep::Apply
    } else {
        MainResetStep::Wait
    }
}

/// While a value keeps changing (a drag), the settings file is written at most this often.
const SETTINGS_SAVE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// Is the user in the middle of changing something: a pointer button or a key held in any
/// of the program's windows, or a widget being dragged?
fn interaction_in_progress(ctx: &egui::Context) -> bool {
    let mut ids: Vec<egui::ViewportId> = ctx.input(|i| i.raw.viewports.keys().copied().collect());
    if !ids.contains(&egui::ViewportId::ROOT) {
        ids.push(egui::ViewportId::ROOT);
    }
    ctx.dragged_id().is_some()
        || ids
            .into_iter()
            .any(|id| ctx.input_for(id, |i| i.pointer.any_down() || !i.keys_down.is_empty()))
}

/// Is a panel shown? The rules of ui/MainWindow.cpp (`workspaceChanged`, `closeProject`,
/// `loadProjectFinished`) and the Console and Progress panels:
///
/// | Panel | Shown when |
/// |---|---|
/// | Toolbox | a dataset is open |
/// | Points | Marker tracking, with trials, cameras calibrated (or no calibration) |
/// | 3D world view | `Show3DView`, and Calibration or as Points |
/// | Display Options, Detailed View, Plot, Events | as Points, and their setting |
/// | Console | the `Console` setting (also with no dataset) |
/// | Progress | while a job runs |
#[must_use]
pub fn panel_wanted(
    p: PanelId,
    state: &AppState,
    facts: Option<&ProjectFacts>,
    settings: &UiSettings,
    jobs: &Jobs,
) -> bool {
    let open = facts.is_some();
    let tracking_panels = facts.is_some_and(|f| {
        state.workspace == Workspace::MarkerTracking && f.n_trials() > 0 && f.calibrated_or_none()
    });
    match p {
        PanelId::Toolbox => open,
        PanelId::Points => tracking_panels,
        PanelId::WorldView => {
            settings.show_3d_view
                && ((open && state.workspace == Workspace::Calibration) || tracking_panels)
        }
        PanelId::DisplayOptions => tracking_panels && settings.show_display_options,
        PanelId::DetailedView => tracking_panels && settings.show_detail_view,
        PanelId::Plot => tracking_panels && settings.show_plot,
        PanelId::Events => tracking_panels && settings.show_events,
        PanelId::Console => settings.console,
        PanelId::Progress => jobs.is_busy(),
    }
}
