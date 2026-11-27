//! The dialogs XMALab 3 opens with `show()` as non-modal windows, which stay open while you
//! work: Help, Project Metadata and Project Overview. Each is a
//! separate operating-system window (an egui immediate viewport), drawn every frame while
//! open; closing the window closes it. Under `egui_kittest`, and for the one window the
//! screenshot variable names (`XMALAB_RS_SHOW_DIALOG`), the window is drawn inside the main
//! window instead.
//!
//! - Help: `HelpDialog.ui`, see [`super::help`].
//! - Welcome is opened with `show()` too, but its form sets `modal` to true, so it is
//!   application-modal: [`super::Dialog::Welcome`], a modal dialog. [`ToolWindow::Welcome`]
//!   only names it for [`super::Dialogs::open_window`].
//!
//! Project Metadata and Project Overview set `windowModality` to `ApplicationModal` in their
//! forms but `modal` to false after it, and `QDialog::setModal(false)` resets the modality
//! to `NonModal` (checked against Qt 6.12: `windowModality()` is 0 and `isModal()` false
//! after the two setters in the order `uic` writes them), so XMALab 3 shows them non-modal.
//! - Project Metadata and Project Overview: trees built from the document when the window
//!   opens (and, for the overview, on "Update"). Project Metadata's tree is also rebuilt
//!   when the document's metadata changes ("Add metadata from XMAPortal file" calls
//!   `MetaDataInfo::update`; undo and redo too), keeping what is expanded; its button
//!   opens [`super::portal_metadata`]. It is rebuilt only when what it is built from
//!   ([`MetadataInputs`]) changed, not on every change of the document (a tracking run
//!   changes it at every frame).

use egui::{ViewportBuilder, ViewportId};
use xma_model::{CalibrationType, Project};

use super::form::{self, Expanded, TreeItem, TreeWidth};
use super::{help, metadata, overview};

/// The four windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ToolWindow {
    Help,
    Welcome,
    Metadata,
    Overview,
}

impl ToolWindow {
    pub const ALL: [ToolWindow; 4] = [
        ToolWindow::Help,
        ToolWindow::Welcome,
        ToolWindow::Metadata,
        ToolWindow::Overview,
    ];

    /// The form's window title.
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            ToolWindow::Help => "Help",
            ToolWindow::Welcome => "Welcome to RustXMALab",
            ToolWindow::Metadata => "Project Metadata",
            ToolWindow::Overview => "Project overview",
        }
    }

    /// The form's size (Help and Welcome limited to a screen's height).
    #[must_use]
    pub fn size(self) -> [f32; 2] {
        match self {
            ToolWindow::Help => [859.0, 900.0],
            ToolWindow::Welcome => [779.0, 855.0],
            ToolWindow::Metadata | ToolWindow::Overview => [500.0, 300.0],
        }
    }

    fn viewport_id(self) -> ViewportId {
        ViewportId::from_hash_of(("xma_tool_window", self as u8))
    }
}

/// Is the Welcome dialog due (`MainWindow::MainWindow`: the `WelcomeDialog` setting is an
/// older version than this program's)? The C++ compares the two version strings as text,
/// so "3.10" counted as older than "3.9"; here they are compared as version numbers.
#[must_use]
pub fn welcome_due(setting: &str, version: &str) -> bool {
    let parse = |s: &str| -> Vec<u64> {
        s.split(['.', '-'])
            .map(|p| {
                p.chars()
                    .take_while(char::is_ascii_digit)
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0)
            })
            .collect()
    };
    let (a, b) = (parse(setting), parse(version));
    let n = a.len().max(b.len());
    let get = |v: &Vec<u64>, i: usize| v.get(i).copied().unwrap_or(0);
    for i in 0..n {
        match get(&a, i).cmp(&get(&b, i)) {
            std::cmp::Ordering::Less => return true,
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Equal => {}
        }
    }
    false
}

/// The open windows and their state.
#[derive(Debug, Default)]
pub struct ToolWindows {
    pub help: bool,
    /// The Project Metadata tree: items, the expanded ones, column 0's width.
    pub metadata: Option<(Vec<TreeItem>, Expanded, TreeWidth)>,
    /// The Project overview tree: items, the expanded ones, column 0's width.
    pub overview: Option<(Vec<TreeItem>, Expanded, TreeWidth)>,
    /// Drawn inside the main window (screenshot mode).
    pub embedded: Option<ToolWindow>,
    /// "Add metadata from XMAPortal file" was clicked ([`ToolWindows::take_add_metadata`]).
    add_metadata: bool,
    /// The document revision the metadata tree was last compared with.
    metadata_revision: Option<u64>,
    /// What the metadata tree was last built from.
    metadata_inputs: Option<MetadataInputs>,
    /// How many times the metadata tree was built (tests).
    metadata_builds: u64,
}

/// What [`metadata::tree`] reads of the project (`portal_summary::metadata_tree`): the
/// calibration type, the project's portal metadata, each trial's default flag, name, portal
/// metadata and number of videos, and the cameras' portal ids. The tree is the same as long
/// as these are.
#[derive(Debug, PartialEq)]
struct MetadataInputs {
    calibration_type: CalibrationType,
    portal_metadata: Vec<String>,
    /// Per trial: is it the Default trial, its name, its portal metadata, its video count.
    trials: Vec<(bool, String, Vec<String>, usize)>,
    portal_ids: Vec<i32>,
}

impl MetadataInputs {
    fn of(p: &Project) -> Self {
        MetadataInputs {
            calibration_type: p.calibration_type,
            portal_metadata: p.portal_metadata.clone(),
            trials: p
                .trials
                .iter()
                .map(|t| {
                    (
                        t.is_default,
                        t.name.clone(),
                        t.portal_metadata.clone(),
                        t.videos.len(),
                    )
                })
                .collect(),
            portal_ids: p.cameras.iter().map(|c| c.portal_id).collect(),
        }
    }

    /// The same as `MetadataInputs::of(p)`, without building it.
    fn matches(&self, p: &Project) -> bool {
        self.calibration_type == p.calibration_type
            && self.portal_metadata == p.portal_metadata
            && self.trials.len() == p.trials.len()
            && self.trials.iter().zip(&p.trials).all(|((d, n, m, v), t)| {
                *d == t.is_default
                    && *n == t.name
                    && *m == t.portal_metadata
                    && *v == t.videos.len()
            })
            && self.portal_ids.len() == p.cameras.len()
            && self
                .portal_ids
                .iter()
                .zip(&p.cameras)
                .all(|(id, c)| *id == c.portal_id)
    }
}

impl ToolWindows {
    /// Opens a window (or keeps it open), building its contents from the document.
    pub fn open(&mut self, w: ToolWindow, doc: Option<&xma_engine::Document>) {
        match w {
            ToolWindow::Help => self.help = true,
            ToolWindow::Welcome => {}
            ToolWindow::Metadata => {
                if let Some(d) = doc {
                    let items = self.build_metadata(d);
                    self.metadata = Some((items, Expanded::new(), TreeWidth::default()));
                }
            }
            ToolWindow::Overview => {
                if let Some(d) = doc {
                    self.overview =
                        Some((overview::tree(d), Expanded::new(), TreeWidth::default()));
                }
            }
        }
    }

    /// Closes a window.
    pub fn close(&mut self, w: ToolWindow) {
        match w {
            ToolWindow::Help => self.help = false,
            ToolWindow::Welcome => {}
            ToolWindow::Metadata => {
                self.metadata = None;
                self.metadata_inputs = None;
            }
            ToolWindow::Overview => self.overview = None,
        }
    }

    #[must_use]
    pub fn is_open(&self, w: ToolWindow) -> bool {
        match w {
            ToolWindow::Help => self.help,
            ToolWindow::Welcome => false,
            ToolWindow::Metadata => self.metadata.is_some(),
            ToolWindow::Overview => self.overview.is_some(),
        }
    }

    /// The project was closed: the windows that show it close (their C++ counterparts are
    /// parented to the main window and show a project that is gone).
    pub fn project_closed(&mut self) {
        self.metadata = None;
        self.metadata_inputs = None;
        self.overview = None;
    }

    /// How many times the Project Metadata tree was built since this was made (tests).
    #[doc(hidden)]
    #[must_use]
    pub fn metadata_builds(&self) -> u64 {
        self.metadata_builds
    }

    /// Builds the metadata tree, noting what it was built from.
    fn build_metadata(&mut self, d: &xma_engine::Document) -> Vec<TreeItem> {
        self.metadata_inputs = Some(MetadataInputs::of(d.project()));
        self.metadata_builds += 1;
        metadata::tree(d)
    }

    /// "Add metadata from XMAPortal file" was clicked since the last call.
    pub fn take_add_metadata(&mut self) -> bool {
        std::mem::take(&mut self.add_metadata)
    }

    /// Rebuilds the metadata tree when the document changed and the tree with it.
    fn follow_metadata(&mut self, doc: Option<&xma_engine::Document>) {
        let (Some(d), true) = (doc, self.metadata.is_some()) else {
            return;
        };
        let rev = d.revision();
        if self.metadata_revision == Some(rev) {
            return;
        }
        self.metadata_revision = Some(rev);
        if self
            .metadata_inputs
            .as_ref()
            .is_some_and(|i| i.matches(d.project()))
        {
            return;
        }
        let now = self.build_metadata(d);
        if let Some((items, _, width)) = &mut self.metadata
            && *items != now
        {
            *items = now;
            *width = TreeWidth::default();
        }
    }

    /// Draws the open windows.
    pub fn show(&mut self, ctx: &egui::Context, doc: Option<&xma_engine::Document>) {
        self.follow_metadata(doc);
        let mut add_metadata = false;
        for w in ToolWindow::ALL {
            if !self.is_open(w) {
                continue;
            }
            let mut close = false;
            let mut body = |ui: &mut egui::Ui| match w {
                ToolWindow::Help => help::ui(ui),
                ToolWindow::Welcome => {}
                ToolWindow::Metadata => {
                    if let Some((items, expanded, width)) = &mut self.metadata {
                        let h = ui.available_height() - ui.spacing().interact_size.y - 8.0;
                        ui.allocate_ui(egui::vec2(ui.available_width(), h.max(80.0)), |ui| {
                            form::tree(ui, "metadata_tree", items, expanded, width);
                        });
                        if form::wide_button(ui, doc.is_some(), "Add metadata from XMAPortal file")
                            .clicked()
                        {
                            add_metadata = true;
                        }
                    }
                }
                ToolWindow::Overview => {
                    let mut update = false;
                    if let Some((items, expanded, width)) = &mut self.overview {
                        let h = ui.available_height() - ui.spacing().interact_size.y - 8.0;
                        ui.allocate_ui(egui::vec2(ui.available_width(), h.max(80.0)), |ui| {
                            form::tree(ui, "overview_tree", items, expanded, width);
                        });
                        update = form::wide_button(ui, true, "Update").clicked();
                    }
                    if update && let Some(d) = doc {
                        self.overview =
                            Some((overview::tree(d), Expanded::new(), TreeWidth::default()));
                    }
                }
            };
            if self.embedded == Some(w) {
                let [sw, sh] = w.size();
                egui::Window::new(w.title())
                    .id(egui::Id::unique(("xma_tool_embedded", w as u8)))
                    .default_size([sw, sh.min(ctx.content_rect().height() - 40.0)])
                    .default_pos([40.0, 30.0])
                    .collapsible(false)
                    .show(ctx, |ui| body(ui));
            } else {
                let builder = ViewportBuilder::default()
                    .with_title(w.title())
                    .with_app_id(crate::APP_ID)
                    .with_inner_size(w.size())
                    .with_minimize_button(false);
                ctx.show_viewport_immediate(w.viewport_id(), builder, |ui, _class| {
                    // This window's share of `qt::combo_wheel`'s frame hook.
                    crate::qt::combo_wheel_begin_frame(&ui.ctx().clone());
                    egui::CentralPanel::default().show(ui, |ui| body(ui));
                    if ui.ctx().input(|i| i.viewport().close_requested()) {
                        close = true;
                    }
                    // This window's clicked links (the Help window's), for `crate::links`.
                    crate::links::collect(ui.ctx());
                });
            }
            if close {
                self.close(w);
            }
        }
        self.add_metadata |= add_metadata;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welcome_compares_versions_as_numbers() {
        assert!(welcome_due("0", "0.1.0"));
        assert!(welcome_due("3.9", "3.10"));
        assert!(!welcome_due("3.10", "3.9"));
        assert!(!welcome_due("0.1.0", "0.1.0"));
        assert!(!welcome_due("0.2", "0.1.5"));
    }
}

#[cfg(test)]
mod metadata_inputs_tests {
    use std::sync::Arc;

    use super::*;
    use xma_model::{Camera, Trial};

    fn project() -> Project {
        let mut p = Project::new();
        p.cameras.push(Arc::new(Camera::new("Camera 1".to_owned())));
        let mut t = Trial::default_trial();
        t.name = "Trial 1".into();
        t.is_default = false;
        p.trials.push(Arc::new(t));
        p
    }

    #[test]
    fn review13_each_input_of_the_metadata_tree_is_seen() {
        let p = project();
        let i = MetadataInputs::of(&p);
        assert!(i.matches(&p));
        type Change = Box<dyn Fn(&mut Project)>;
        let changes: Vec<Change> = vec![
            Box::new(|p| p.calibration_type = CalibrationType::Internal),
            Box::new(|p| p.portal_metadata.push("x".into())),
            Box::new(|p| Arc::make_mut(&mut p.trials[0]).name = "Trial 2".into()),
            Box::new(|p| Arc::make_mut(&mut p.trials[0]).is_default = true),
            Box::new(|p| {
                Arc::make_mut(&mut p.trials[0])
                    .portal_metadata
                    .push("x".into())
            }),
            Box::new(|p| {
                Arc::make_mut(&mut p.trials[0])
                    .videos
                    .push(xma_model::VideoSource {
                        filenames: vec!["a.avi".into()],
                        metadata: Vec::new(),
                    });
            }),
            Box::new(|p| {
                let t = p.trials[0].clone();
                p.trials.push(t);
            }),
            Box::new(|p| {
                p.trials.clear();
            }),
            Box::new(|p| Arc::make_mut(&mut p.cameras[0]).portal_id = 3),
            Box::new(|p| {
                let c = p.cameras[0].clone();
                p.cameras.push(c);
            }),
        ];
        for (k, change) in changes.iter().enumerate() {
            let mut q = p.clone();
            change(&mut q);
            assert!(!i.matches(&q), "change {k} not seen");
            assert!(MetadataInputs::of(&q).matches(&q));
            assert_ne!(MetadataInputs::of(&q), i);
        }
        // What the tree does not read: not a change.
        let mut q = p.clone();
        Arc::make_mut(&mut q.trials[0]).n_frames = 99;
        assert!(i.matches(&q));
    }
}
