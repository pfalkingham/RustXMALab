//! The Points panel's editing flows: what each context-menu entry, button, check box and
//! drop of the Points tree asks (input prompts, `FromToDialog`, `MarkerDialog`,
//! `ImportExportPointsDialog`, confirmations) and the edit it then requests
//! (`docs/cpp-editing.md` sections 4, 5.1 and 10.2; E15 to E29, E67 to E71).
//!
//! Ported from ui/MarkerTreeWidget.cpp (the `action_*_triggered` slots and `dropEvent`),
//! ui/PointsDockWidget.cpp (the button and check-box slots, `reloadListFromObject`),
//! ui/MarkerTreeWidgetButton.cpp (`settingsButtonClicked`) and
//! ui/ImportExportPointsDialog.cpp (`on_pushButtonOK_clicked` and the four modes).
//!
//! How it runs: the panel raises [`crate::actions::Action::Points`] with a
//! [`PointsAction`]; the application calls [`start`], which reads the document and the
//! selection and either opens the first dialog ([`Dialog::Points`]) or requests the edit.
//! Each dialog's answer comes back as [`DialogResult::Points`] and goes to [`answer`],
//! which opens the next dialog or requests the edit. The steps themselves ([`begin`],
//! [`next`]) are plain functions of the document and the selection, so they are tested
//! without a window. Cancel (or Escape, or the close button) at any step ends the flow
//! with nothing changed.
//!
//! After an edit that the C++ follows with `reloadListFromObject`, the tree's selection is
//! the active marker (or body) alone, as the C++ re-selects it; the active marker follows
//! `Trial::addMarker` (the new marker becomes active) and `Trial::removeMarker` (an active
//! index past the end becomes the last marker), the active body `Trial::removeRigidBody`.
//!
//! Differences from XMALab 3 (ledger rows proposed by this package):
//!
//! - D11: the "Import CSV" question asks "Update markers" / "Add as new markers" / Cancel
//!   (XMALab 3: OK updates, Cancel adds, so the import cannot be cancelled); "Import from
//!   xma" and "Copy From trial" say what happens (nothing is deleted by the first; the
//!   second deletes only markers and bodies beyond the source's; existing markers are
//!   renamed and keep their 2D data).
//! - "Set Number Markers" / "Set Number Rigid Bodies" with the count unchanged does
//!   nothing (XMALab 3 asks "You are about to delete Points n+1 to n").
//! - "from default": the confirmation no longer says "This step cannot be undone!" (it can
//!   be undone here).
//! - "Import from xma" with no file chosen does nothing (XMALab 3 asks its confirmation
//!   first, then does nothing).
//! - A drop counts only the selected marker rows (XMALab 3 also iterates selected body
//!   rows, reading their text as marker number 0, an out-of-range index).

use std::path::PathBuf;

use egui::Key;
use xma_engine::edit::{CameraSel, FrameRange, MarkerProperties};
use xma_engine::tracking::RefineWhich;
use xma_engine::view::tree::TreeItem;
use xma_engine::{Document, Edit};
use xma_model::DetectionMethod;

use super::from_to::{FromTo, FromToAnswer};
use super::import_export_points::{ImportExportAnswer, ImportExportPoints, ImportMode};
use super::marker_dialog::{METHODS, MarkerAnswer, MarkerDialog};
use super::{Dialog, DialogEnv, DialogResult, form};
use crate::XmaApp;
use crate::editing::EditRequest;
use crate::icons::Icon;
use crate::state::{AppState, Changes, Selection};

/// What the Points panel asks the application to do. Each names the C++ slot it ports.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PointsAction {
    /// E15 "Change Description" on the right-clicked row.
    ChangeDescription(TreeItem),
    /// E16 "Create Rigid Body from selected".
    CreateRigidBody,
    /// E17 "Add selected to Rigid Body".
    AddToRigidBody,
    /// E18 "Delete Data from selected points".
    ResetPoints,
    /// E19 "Delete selected points".
    DeletePoints,
    /// E20 "Change Detectionmethod of selected Points".
    ChangeDetectionMethod,
    /// E21 "Refine center of selected Points by polynomial fit".
    RefinePoints,
    /// E22 "Change tracked data with data from another point", on the right-clicked
    /// marker.
    ChangePoint { marker: usize },
    /// E23: the selected markers dropped. `body` is the body the drop row belongs to (a
    /// body row, or a marker row under it); `None` for empty space or a free marker row.
    Drop { body: Option<usize> },
    /// E24 "Set Number Markers".
    SetNumberMarkers,
    /// E25 "Set Number Rigid Bodies".
    SetNumberRigidBodies,
    /// E26 "Import/Export".
    ImportExport,
    /// E27 "Apply to trials".
    ApplyToTrials,
    /// E28 "from default" unchecked.
    UnlinkFromDefault,
    /// E29: the settings button of a marker row.
    MarkerSettings { marker: usize },
}

impl PointsAction {
    /// Label and C++ origin, for `Action::info`.
    #[must_use]
    pub fn info(&self) -> (&'static str, &'static str) {
        match self {
            PointsAction::ChangeDescription(_) => (
                "Change Description",
                "MarkerTreeWidget::action_ChangeDescription_triggered",
            ),
            PointsAction::CreateRigidBody => (
                "Create Rigid Body from selected",
                "MarkerTreeWidget::action_CreateRigidBody_triggered",
            ),
            PointsAction::AddToRigidBody => (
                "Add selected to Rigid Body",
                "MarkerTreeWidget::action_AddToRigidBody_triggered",
            ),
            PointsAction::ResetPoints => (
                "Delete Data from selected points",
                "MarkerTreeWidget::action_ResetPoints_triggered",
            ),
            PointsAction::DeletePoints => (
                "Delete selected points",
                "MarkerTreeWidget::action_DeletePoints_triggered",
            ),
            PointsAction::ChangeDetectionMethod => (
                "Change Detectionmethod of selected Points",
                "MarkerTreeWidget::action_ChangeDetectionMethod_triggered",
            ),
            PointsAction::RefinePoints => (
                "Refine center of selected Points by polynomial fit",
                "MarkerTreeWidget::action_RefinePointsPolynomialFit_triggered",
            ),
            PointsAction::ChangePoint { .. } => (
                "Change tracked data with data from another point",
                "MarkerTreeWidget::action_ChangePoint_triggered",
            ),
            PointsAction::Drop { .. } => ("Move markers", "MarkerTreeWidget::dropEvent"),
            PointsAction::SetNumberMarkers => (
                "Set Number Markers",
                "PointsDockWidget::on_pushButtonSetNumberMarkers_clicked",
            ),
            PointsAction::SetNumberRigidBodies => (
                "Set Number Rigid Bodies",
                "PointsDockWidget::on_pushButtonSetNumberRigidBodies_clicked",
            ),
            PointsAction::ImportExport => (
                "Import/Export",
                "PointsDockWidget::on_pushButtonImportExport_clicked",
            ),
            PointsAction::ApplyToTrials => (
                "Apply to trials",
                "PointsDockWidget::on_pushButtonApply_clicked",
            ),
            PointsAction::UnlinkFromDefault => (
                "from default",
                "PointsDockWidget::on_checkBoxIsFromMaster_clicked",
            ),
            PointsAction::MarkerSettings { .. } => (
                "Marker settings",
                "MarkerTreeWidgetButton::settingsButtonClicked",
            ),
        }
    }
}

/// Where a flow stands: which C++ handler runs and what has been answered so far.
#[derive(Clone, Debug, PartialEq)]
pub enum Flow {
    Description {
        trial: usize,
        item: TreeItem,
        current: String,
    },
    AddToBody {
        trial: usize,
        markers: Vec<usize>,
    },
    ResetRange {
        trial: usize,
        markers: Vec<usize>,
    },
    ResetConfirm {
        trial: usize,
        markers: Vec<usize>,
        frames: FrameRange,
    },
    Delete {
        trial: usize,
        markers: Vec<usize>,
    },
    Method {
        trial: usize,
        markers: Vec<usize>,
    },
    RefineWhich {
        trial: usize,
        markers: Vec<usize>,
    },
    RefineRange {
        trial: usize,
        markers: Vec<usize>,
        which: RefineWhich,
    },
    SwapOther {
        trial: usize,
        marker: usize,
        others: Vec<usize>,
    },
    SwapCamera {
        trial: usize,
        marker: usize,
        other: usize,
    },
    SwapRange {
        trial: usize,
        marker: usize,
        other: usize,
        cameras: CameraSel,
    },
    MarkerCount {
        trial: usize,
        current: usize,
    },
    MarkerCountConfirm {
        trial: usize,
        count: usize,
    },
    BodyCount {
        trial: usize,
        current: usize,
    },
    BodyCountConfirm {
        trial: usize,
        count: usize,
    },
    ImportExport {
        trial: usize,
    },
    CsvChoice {
        trial: usize,
        path: PathBuf,
    },
    XmaConfirm {
        trial: usize,
        path: PathBuf,
    },
    XmaTrial {
        trial: usize,
        path: PathBuf,
        names: Vec<String>,
    },
    CopyConfirm {
        trial: usize,
        source: usize,
    },
    Unlink {
        trial: usize,
    },
    Marker {
        trial: usize,
        marker: usize,
    },
}

/// The form a step shows.
#[derive(Clone, Debug, PartialEq)]
pub enum Form {
    /// `QInputDialog::getText`.
    Text {
        title: String,
        label: String,
        value: String,
    },
    /// `QInputDialog::getItem` (not editable).
    Item {
        title: String,
        label: String,
        items: Vec<String>,
        selected: usize,
    },
    /// `QInputDialog::getInt`.
    Int {
        title: String,
        label: String,
        value: i64,
        min: i64,
        max: i64,
    },
    /// `FromToDialog`.
    FromTo(FromTo),
    /// `ConfirmationDialog` with OK and Cancel.
    Confirm { message: String },
    /// A question with one button per choice, and Cancel.
    Choice {
        message: String,
        buttons: Vec<String>,
    },
    /// `MarkerDialog`.
    Marker(MarkerDialog),
    /// `ImportExportPointsDialog`.
    ImportExport(ImportExportPoints),
}

/// One step of a flow on screen.
#[derive(Clone, Debug, PartialEq)]
pub struct PointsDialog {
    pub flow: Flow,
    pub form: Form,
}

/// The answer to a step (its OK, or the chosen button of a [`Form::Choice`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    Text(String),
    Item(usize),
    Int(i64),
    /// Frames, 0-based.
    Range(usize, usize),
    Yes,
    Choice(usize),
    Marker(MarkerProperties),
    ImportExport(ImportExportPoints),
}

/// A step answered: what the application passes back to [`answer`].
#[derive(Clone, Debug, PartialEq)]
pub struct PointsOutcome {
    pub flow: Flow,
    pub answer: Answer,
}

/// The active marker or body after an edit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActiveRule {
    /// Left as it is.
    #[default]
    Keep,
    /// An index past the end becomes the last one (`Trial::removeMarker`,
    /// `removeRigidBody`).
    Clamp,
    /// The last marker (`Trial::addMarker`).
    Last,
}

/// What follows a successful edit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct After {
    /// `reloadListFromObject`: the selection becomes the active row alone.
    pub reload: bool,
    pub marker: ActiveRule,
    pub body: ActiveRule,
}

impl After {
    const NONE: After = After {
        reload: false,
        marker: ActiveRule::Keep,
        body: ActiveRule::Keep,
    };
    const RELOAD: After = After {
        reload: true,
        marker: ActiveRule::Keep,
        body: ActiveRule::Keep,
    };
}

/// What a step leads to.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Nothing,
    Open(PointsDialog),
    Edit { request: EditRequest, after: After },
    Error(String),
}

/// What the steps read.
pub struct Ctx<'a> {
    pub doc: &'a Document,
    pub trial: usize,
    pub selection: &'a Selection,
    pub n_cameras: usize,
}

impl Ctx<'_> {
    fn t(&self) -> &xma_model::Trial {
        &self.doc.project().trials[self.trial]
    }

    /// The selected markers in order (`selectedItems()` of type MARKER, by number).
    fn markers(&self) -> Vec<usize> {
        let n = self.t().markers.len();
        self.selection
            .markers
            .iter()
            .copied()
            .filter(|m| *m < n)
            .collect()
    }

    fn range_dialog(&self) -> FromTo {
        let t = self.t();
        FromTo::new(t.start_frame, t.end_frame, t.n_frames, false, false)
    }
}

fn open(flow: Flow, form: Form) -> Step {
    Step::Open(PointsDialog { flow, form })
}

fn edit(e: Edit, after: After) -> Step {
    Step::Edit {
        request: EditRequest::Apply(e),
        after,
    }
}

fn confirm(flow: Flow, message: impl Into<String>) -> Step {
    open(
        flow,
        Form::Confirm {
            message: message.into(),
        },
    )
}

/// The first step of an action.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn begin(a: &PointsAction, cx: &Ctx<'_>) -> Step {
    let trial = cx.trial;
    let t = cx.t();
    match a {
        // MarkerTreeWidget::action_ChangeDescription_triggered.
        PointsAction::ChangeDescription(item) => {
            let current = match *item {
                TreeItem::Marker(m) => t.markers.get(m).map(|m| m.description.clone()),
                TreeItem::Body(b) => t.rigid_bodies.get(b).map(|b| b.description.clone()),
            };
            let Some(current) = current else {
                return Step::Nothing;
            };
            open(
                Flow::Description {
                    trial,
                    item: *item,
                    current: current.clone(),
                },
                Form::Text {
                    title: "Enter description".into(),
                    label: "Description:".into(),
                    value: current,
                },
            )
        }
        // action_CreateRigidBody_triggered: one body from the selected free markers.
        PointsAction::CreateRigidBody => {
            let markers = cx.markers();
            if markers.is_empty() {
                return Step::Nothing;
            }
            edit(Edit::CreateRigidBody { trial, markers }, After::RELOAD)
        }
        // action_AddToRigidBody_triggered: "Enter ID", prefilled "RB".
        PointsAction::AddToRigidBody => open(
            Flow::AddToBody {
                trial,
                markers: cx.markers(),
            },
            Form::Text {
                title: "Enter ID".into(),
                label: "ID:".into(),
                value: "RB".into(),
            },
        ),
        // action_ResetPoints_triggered: the range first, then the confirmation.
        PointsAction::ResetPoints => open(
            Flow::ResetRange {
                trial,
                markers: cx.markers(),
            },
            Form::FromTo(cx.range_dialog()),
        ),
        // action_DeletePoints_triggered: the confirmation before anything.
        PointsAction::DeletePoints => confirm(
            Flow::Delete {
                trial,
                markers: cx.markers(),
            },
            "Are you sure you want to delete the selected Points?",
        ),
        // action_ChangeDetectionMethod_triggered.
        PointsAction::ChangeDetectionMethod => open(
            Flow::Method {
                trial,
                markers: cx.markers(),
            },
            Form::Item {
                title: "Choose method".into(),
                label: "Method:".into(),
                items: METHODS.iter().map(|s| (*s).to_owned()).collect(),
                selected: 0,
            },
        ),
        // action_RefinePointsPolynomialFit_triggered.
        PointsAction::RefinePoints => open(
            Flow::RefineWhich {
                trial,
                markers: cx.markers(),
            },
            Form::Item {
                title: "Which points do you want to refine?".into(),
                label: "Points :".into(),
                items: REFINE_ITEMS.iter().map(|s| (*s).to_owned()).collect(),
                selected: 0,
            },
        ),
        // action_ChangePoint_triggered: nothing with fewer than two markers.
        PointsAction::ChangePoint { marker } => {
            let n = t.markers.len();
            if n < 2 || *marker >= n {
                return Step::Nothing;
            }
            let others: Vec<usize> = (0..n).filter(|i| i != marker).collect();
            open(
                Flow::SwapOther {
                    trial,
                    marker: *marker,
                    others: others.clone(),
                },
                Form::Item {
                    title: "Choose marker with which to exchange tracked data".into(),
                    label: "Marker:".into(),
                    items: others.iter().map(|i| (i + 1).to_string()).collect(),
                    selected: 0,
                },
            )
        }
        // dropEvent: see `drop_edit`.
        PointsAction::Drop { body } => match drop_edit(trial, t, &cx.markers(), *body) {
            Some(e) => edit(e, After::RELOAD),
            None => Step::Nothing,
        },
        // on_pushButtonSetNumberMarkers_clicked.
        PointsAction::SetNumberMarkers => {
            let current = t.markers.len();
            open(
                Flow::MarkerCount { trial, current },
                Form::Int {
                    title: "Set Number of Markers to".into(),
                    label: String::new(),
                    value: current as i64,
                    min: 0,
                    max: 1000,
                },
            )
        }
        PointsAction::SetNumberRigidBodies => {
            let current = t.rigid_bodies.len();
            open(
                Flow::BodyCount { trial, current },
                Form::Int {
                    title: "Set Number of Rigid Bodies to".into(),
                    label: String::new(),
                    value: current as i64,
                    min: 0,
                    max: 1000,
                },
            )
        }
        // on_pushButtonImportExport_clicked.
        PointsAction::ImportExport => {
            let names: Vec<String> = cx
                .doc
                .project()
                .trials
                .iter()
                .map(|t| t.name.clone())
                .collect();
            open(
                Flow::ImportExport { trial },
                Form::ImportExport(ImportExportPoints::new(&names, trial)),
            )
        }
        // on_pushButtonApply_clicked: no question.
        PointsAction::ApplyToTrials => {
            if t.is_default {
                edit(Edit::ApplyDefaultToTrials, After::NONE)
            } else {
                Step::Nothing
            }
        }
        // on_checkBoxIsFromMaster_clicked.
        PointsAction::UnlinkFromDefault => {
            if t.is_copy_from_default {
                confirm(Flow::Unlink { trial }, UNLINK_TEXT)
            } else {
                Step::Nothing
            }
        }
        // MarkerTreeWidgetButton::settingsButtonClicked, marker row.
        PointsAction::MarkerSettings { marker } => match t.markers.get(*marker) {
            Some(m) => open(
                Flow::Marker {
                    trial,
                    marker: *marker,
                },
                Form::Marker(MarkerDialog::new(
                    *marker,
                    m,
                    t.is_default,
                    t.is_copy_from_default,
                )),
            ),
            None => Step::Nothing,
        },
    }
}

/// The items of "Which points do you want to refine?".
pub const REFINE_ITEMS: [&str; 3] = [
    "set from Mainwindow + tracked",
    "only tracked",
    "All (including set from detailwindow)",
];

/// The "from default" confirmation. XMALab 3 adds "This step cannot be undone!", which is
/// not true here.
pub const UNLINK_TEXT: &str = "Are you sure to remove the link to the default trial?";

/// The "Import CSV" question when the trial has markers (D11 fixed).
pub const CSV_TEXT: &str = "Your current trial already has markers associated with it. Do you want to update these markers with the names and positions in the csv-file (point 1 updates marker 1, and so on), or add the points of the file as new markers?";

/// Its buttons, in order, before Cancel.
pub const CSV_BUTTONS: [&str; 2] = ["Update markers", "Add as new markers"];

/// The "Import from xma" confirmation when the trial has markers or bodies (D11 fixed).
pub const XMA_TEXT: &str = "The markers and rigid bodies of the trial you choose will be applied to this trial in order: marker 1 takes the name and settings of the imported marker 1, rigid body 1 those of the imported rigid body 1, and so on. Nothing is deleted: markers keep their 2D data under their new names, and markers and rigid bodies beyond those imported stay as they are. Are you sure you want to proceed?";

/// The "Copy From trial" confirmation (D11 fixed): what really happens to this trial
/// (`dst_*`) when the source has `src_*` markers and bodies.
#[must_use]
pub fn copy_text(
    source: &str,
    src_markers: usize,
    src_bodies: usize,
    dst_markers: usize,
    dst_bodies: usize,
    links: Option<bool>,
) -> String {
    let mut s = format!(
        "The markers and rigid bodies of trial \"{source}\" will be copied to this trial in order: marker 1 takes the name and settings of its marker 1, rigid body 1 becomes a copy of its rigid body 1, and so on. Markers keep their 2D data under their new names."
    );
    let range = |what: &str, one: &str, from: usize, to: usize| {
        if from == to {
            format!(" {one} {to} will be deleted")
        } else {
            format!(" {what} {from} to {to} will be deleted")
        }
    };
    if dst_markers > src_markers {
        s += &range("Markers", "Marker", src_markers + 1, dst_markers);
        s += " with their data.";
    }
    if dst_bodies > src_bodies {
        s += &range("Rigid bodies", "Rigid body", src_bodies + 1, dst_bodies);
        s += ".";
    }
    match links {
        Some(true) => s += " This trial will be linked to the Default trial.",
        Some(false) => s += " The link to the Default trial will be removed.",
        None => {}
    }
    s += " Are you sure you want to proceed?";
    s
}

/// E23: the edit a drop makes, or `None` when it changes nothing.
///
/// Ported from ui/MarkerTreeWidget.cpp, MarkerTreeWidget::dropEvent: dropped on a body row
/// or on a marker row inside a body, every dragged marker is removed from its body and
/// added to that one; dropped on empty space or a free marker row, every dragged marker in
/// a body is removed from it and free ones stay.
#[must_use]
pub fn drop_edit(
    trial: usize,
    t: &xma_model::Trial,
    markers: &[usize],
    body: Option<usize>,
) -> Option<Edit> {
    if markers.is_empty() {
        return None;
    }
    if body.is_none()
        && !markers
            .iter()
            .any(|m| t.rigid_bodies.iter().any(|b| b.marker_indices.contains(m)))
    {
        return None;
    }
    Some(Edit::MoveMarkersToBody {
        trial,
        markers: markers.to_vec(),
        body,
    })
}

/// The step after an answer.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn next(flow: &Flow, answer: &Answer, cx: &Ctx<'_>) -> Step {
    let t = cx.t();
    match (flow, answer) {
        (
            Flow::Description {
                trial,
                item,
                current,
            },
            Answer::Text(s),
        ) => {
            if s == current {
                return Step::Nothing;
            }
            let e = match *item {
                TreeItem::Marker(marker) => Edit::SetMarkerDescription {
                    trial: *trial,
                    marker,
                    description: s.clone(),
                },
                TreeItem::Body(body) => Edit::SetBodyDescription {
                    trial: *trial,
                    body,
                    description: s.clone(),
                },
            };
            edit(e, After::NONE)
        }
        // The text must name a body row exactly ("RB2"); otherwise nothing happens.
        (Flow::AddToBody { trial, markers }, Answer::Text(s)) => {
            let body = (0..t.rigid_bodies.len()).find(|b| *s == format!("RB{}", b + 1));
            match body {
                Some(b) if !markers.is_empty() => edit(
                    Edit::MoveMarkersToBody {
                        trial: *trial,
                        markers: markers.clone(),
                        body: Some(b),
                    },
                    After::RELOAD,
                ),
                _ => Step::Nothing,
            }
        }
        (Flow::ResetRange { trial, markers }, Answer::Range(first, last)) => confirm(
            Flow::ResetConfirm {
                trial: *trial,
                markers: markers.clone(),
                frames: FrameRange {
                    first: *first,
                    last: *last,
                },
            },
            "Are you sure you want to reset the data for the selected Points?",
        ),
        (
            Flow::ResetConfirm {
                trial,
                markers,
                frames,
            },
            Answer::Yes,
        ) => {
            if markers.is_empty() || frames.first > frames.last {
                return Step::Nothing;
            }
            edit(
                Edit::ResetFrames {
                    trial: *trial,
                    markers: markers.clone(),
                    cameras: CameraSel::All,
                    frames: *frames,
                    untrackable: false,
                },
                After::NONE,
            )
        }
        (Flow::Delete { trial, markers }, Answer::Yes) => {
            if markers.is_empty() {
                return Step::Nothing;
            }
            edit(
                Edit::RemoveMarkers {
                    trial: *trial,
                    markers: markers.clone(),
                },
                After {
                    reload: true,
                    marker: ActiveRule::Clamp,
                    body: ActiveRule::Keep,
                },
            )
        }
        (Flow::Method { trial, markers }, Answer::Item(i)) => {
            let Some(method) = i32::try_from(*i).ok().and_then(DetectionMethod::from_code) else {
                return Step::Nothing;
            };
            if markers.is_empty() {
                return Step::Nothing;
            }
            edit(
                Edit::SetMarkerMethod {
                    trial: *trial,
                    markers: markers.clone(),
                    method,
                },
                After::NONE,
            )
        }
        (Flow::RefineWhich { trial, markers }, Answer::Item(i)) => {
            let which = match i {
                0 => RefineWhich::SetAndTracked,
                1 => RefineWhich::TrackedOnly,
                _ => RefineWhich::All,
            };
            open(
                Flow::RefineRange {
                    trial: *trial,
                    markers: markers.clone(),
                    which,
                },
                Form::FromTo(cx.range_dialog()),
            )
        }
        (
            Flow::RefineRange {
                trial,
                markers,
                which,
            },
            Answer::Range(first, last),
        ) => {
            if markers.is_empty() {
                return Step::Nothing;
            }
            Step::Edit {
                // Every camera, as the C++ loops over all of them.
                request: EditRequest::RefinePoints {
                    trial: *trial,
                    markers: markers.clone(),
                    first: *first,
                    last: *last,
                    which: *which,
                },
                after: After::NONE,
            }
        }
        (
            Flow::SwapOther {
                trial,
                marker,
                others,
            },
            Answer::Item(i),
        ) => {
            let Some(&other) = others.get(*i) else {
                return Step::Nothing;
            };
            let mut items = vec!["All".to_owned()];
            items.extend((1..=cx.n_cameras).map(|c| c.to_string()));
            open(
                Flow::SwapCamera {
                    trial: *trial,
                    marker: *marker,
                    other,
                },
                Form::Item {
                    title: "Which camera".into(),
                    label: "Camera:".into(),
                    items,
                    selected: 0,
                },
            )
        }
        (
            Flow::SwapCamera {
                trial,
                marker,
                other,
            },
            Answer::Item(i),
        ) => {
            let cameras = if *i == 0 {
                CameraSel::All
            } else {
                CameraSel::One(i - 1)
            };
            open(
                Flow::SwapRange {
                    trial: *trial,
                    marker: *marker,
                    other: *other,
                    cameras,
                },
                Form::FromTo(cx.range_dialog()),
            )
        }
        (
            Flow::SwapRange {
                trial,
                marker,
                other,
                cameras,
            },
            Answer::Range(first, last),
        ) => edit(
            Edit::SwapMarkerData {
                trial: *trial,
                marker: *marker,
                other: *other,
                cameras: *cameras,
                frames: FrameRange {
                    first: *first,
                    last: *last,
                },
            },
            After::NONE,
        ),
        (Flow::MarkerCount { trial, current }, Answer::Int(v)) => {
            let n = usize::try_from(*v).unwrap_or(0);
            if n > *current {
                edit(
                    Edit::SetMarkerCount {
                        trial: *trial,
                        count: n,
                    },
                    After {
                        reload: true,
                        marker: ActiveRule::Last,
                        body: ActiveRule::Keep,
                    },
                )
            } else if n < *current {
                confirm(
                    Flow::MarkerCountConfirm {
                        trial: *trial,
                        count: n,
                    },
                    format!(
                        "You are about to delete Points {} to {}. Are you sure you want to proceed?",
                        n + 1,
                        current
                    ),
                )
            } else {
                Step::Nothing
            }
        }
        (Flow::MarkerCountConfirm { trial, count }, Answer::Yes) => edit(
            Edit::SetMarkerCount {
                trial: *trial,
                count: *count,
            },
            After {
                reload: true,
                marker: ActiveRule::Clamp,
                body: ActiveRule::Keep,
            },
        ),
        (Flow::BodyCount { trial, current }, Answer::Int(v)) => {
            let n = usize::try_from(*v).unwrap_or(0);
            if n > *current {
                edit(
                    Edit::SetRigidBodyCount {
                        trial: *trial,
                        count: n,
                    },
                    After::RELOAD,
                )
            } else if n < *current {
                confirm(
                    Flow::BodyCountConfirm {
                        trial: *trial,
                        count: n,
                    },
                    format!(
                        "You are about to delete Rigid Bodies {} to {}. Are you sure you want to proceed?",
                        n + 1,
                        current
                    ),
                )
            } else {
                Step::Nothing
            }
        }
        (Flow::BodyCountConfirm { trial, count }, Answer::Yes) => edit(
            Edit::SetRigidBodyCount {
                trial: *trial,
                count: *count,
            },
            After {
                reload: true,
                marker: ActiveRule::Keep,
                body: ActiveRule::Clamp,
            },
        ),
        (Flow::ImportExport { trial }, Answer::ImportExport(d)) => import_ok(*trial, d, cx),
        (Flow::CsvChoice { trial, path }, Answer::Choice(i)) => edit(
            Edit::ImportMarkersCsv {
                trial: *trial,
                path: path.clone(),
                update_existing: *i == 0,
            },
            After::RELOAD,
        ),
        (Flow::XmaConfirm { trial, path }, Answer::Yes) => choose_xma_trial(*trial, path),
        (Flow::XmaTrial { trial, path, names }, Answer::Item(i)) => match names.get(*i) {
            Some(name) => edit(
                Edit::ImportMarkersFromProject {
                    trial: *trial,
                    path: path.clone(),
                    source_trial: name.clone(),
                },
                After::RELOAD,
            ),
            None => Step::Nothing,
        },
        (Flow::CopyConfirm { trial, source }, Answer::Yes) => edit(
            Edit::CopyMarkersFromTrial {
                trial: *trial,
                source: *source,
            },
            After {
                reload: true,
                marker: ActiveRule::Clamp,
                body: ActiveRule::Clamp,
            },
        ),
        (Flow::Unlink { trial }, Answer::Yes) => {
            edit(Edit::UnlinkFromDefault { trial: *trial }, After::RELOAD)
        }
        // MarkerDialog::isComplete: the changed values of the enabled controls.
        (Flow::Marker { trial, marker }, Answer::Marker(p)) => {
            if *p == MarkerProperties::default() {
                return Step::Nothing;
            }
            edit(
                Edit::SetMarkerProperties {
                    trial: *trial,
                    marker: *marker,
                    properties: *p,
                },
                After::NONE,
            )
        }
        _ => Step::Nothing,
    }
}

/// `ImportExportPointsDialog::on_pushButtonOK_clicked`: the checked mode.
fn import_ok(trial: usize, d: &ImportExportPoints, cx: &Ctx<'_>) -> Step {
    let t = cx.t();
    let has_markers = !t.markers.is_empty();
    let has_any = has_markers || !t.rigid_bodies.is_empty();
    match d.mode {
        None => Step::Nothing,
        // importCSV.
        Some(ImportMode::Csv) => {
            if d.csv.is_empty() {
                return Step::Nothing;
            }
            let path = PathBuf::from(&d.csv);
            if has_markers {
                open(
                    Flow::CsvChoice { trial, path },
                    Form::Choice {
                        message: CSV_TEXT.into(),
                        buttons: CSV_BUTTONS.iter().map(|s| (*s).to_owned()).collect(),
                    },
                )
            } else {
                edit(
                    Edit::ImportMarkersCsv {
                        trial,
                        path,
                        update_existing: false,
                    },
                    After::RELOAD,
                )
            }
        }
        // importXMA.
        Some(ImportMode::Xma) => {
            if d.xma.is_empty() {
                return Step::Nothing;
            }
            let path = PathBuf::from(&d.xma);
            if has_any {
                confirm(Flow::XmaConfirm { trial, path }, XMA_TEXT)
            } else {
                choose_xma_trial(trial, &path)
            }
        }
        // copyFromTrial.
        Some(ImportMode::Trial) => {
            let Some(source) = d.source_trial() else {
                return Step::Nothing;
            };
            let Some(src) = cx.doc.project().trials.get(source) else {
                return Step::Nothing;
            };
            let edit_step = edit(
                Edit::CopyMarkersFromTrial { trial, source },
                After {
                    reload: true,
                    marker: ActiveRule::Clamp,
                    body: ActiveRule::Clamp,
                },
            );
            if !has_any {
                return edit_step;
            }
            let links = (src.is_default != t.is_copy_from_default).then_some(src.is_default);
            confirm(
                Flow::CopyConfirm { trial, source },
                copy_text(
                    &src.name,
                    src.markers.len(),
                    src.rigid_bodies.len(),
                    t.markers.len(),
                    t.rigid_bodies.len(),
                    links,
                ),
            )
        }
        // rename.
        Some(ImportMode::Rename) => {
            if d.csv.is_empty() {
                return Step::Nothing;
            }
            edit(
                Edit::RenameMarkersFromCsv {
                    trial,
                    path: PathBuf::from(&d.csv),
                },
                After::RELOAD,
            )
        }
    }
}

/// `ProjectFileIO::readTrials`, then "Choose trial to import".
fn choose_xma_trial(trial: usize, path: &std::path::Path) -> Step {
    match xma_engine::document::read_project(path, &xma_engine::document::OpenOptions::default()) {
        Ok(loaded) => {
            let names: Vec<String> = loaded
                .project
                .trials
                .iter()
                .map(|t| t.name.clone())
                .collect();
            open(
                Flow::XmaTrial {
                    trial,
                    path: path.to_path_buf(),
                    names: names.clone(),
                },
                Form::Item {
                    title: "Choose trial to import".into(),
                    label: "Trial:".into(),
                    items: names,
                    selected: 0,
                },
            )
        }
        Err(e) => Step::Error(format!("Could not read {}: {e}", path.display())),
    }
}

// ---------------------------------------------------------------------------------------
// The application side.

fn ctx_of(app: &XmaApp) -> Option<Ctx<'_>> {
    let doc = app.session.doc.as_ref()?;
    let trial = app.state.active_trial?;
    doc.project().trials.get(trial)?;
    Some(Ctx {
        doc,
        trial,
        selection: &app.state.selection,
        n_cameras: doc.project().cameras.len(),
    })
}

/// Carries out a [`PointsAction`] (from `XmaApp::dispatch`); returns the state changes to
/// record.
pub fn start(app: &mut XmaApp, a: &PointsAction) -> Changes {
    let Some(cx) = ctx_of(app) else {
        return Changes::NONE;
    };
    let step = begin(a, &cx);
    run(app, step)
}

/// Acts on an answered step (from `XmaApp::dialog_result`); returns the state changes to
/// record.
pub fn answer(app: &mut XmaApp, o: &PointsOutcome) -> Changes {
    let Some(cx) = ctx_of(app) else {
        return Changes::NONE;
    };
    let step = next(&o.flow, &o.answer, &cx);
    run(app, step)
}

fn run(app: &mut XmaApp, step: Step) -> Changes {
    match step {
        Step::Nothing => Changes::NONE,
        Step::Error(msg) => {
            app.dialogs.error(msg);
            Changes::NONE
        }
        Step::Open(d) => {
            app.dialogs.open(Dialog::Points(Box::new(d)));
            Changes::NONE
        }
        Step::Edit { request, after } => {
            let before = app.session.doc.as_ref().map(Document::revision);
            app.edits.push(request);
            app.handle_edit_requests();
            let now = app.session.doc.as_ref().map(Document::revision);
            if now == before {
                return Changes::NONE;
            }
            let Some(doc) = app.session.doc.as_ref() else {
                return Changes::NONE;
            };
            apply_after(&mut app.state, doc, after)
        }
    }
}

/// The active marker and body and the selection after a successful edit.
pub fn apply_after(state: &mut AppState, doc: &Document, after: After) -> Changes {
    let Some(t) = state.active_trial else {
        return Changes::NONE;
    };
    let Some(trial) = doc.project().trials.get(t) else {
        return Changes::NONE;
    };
    let (nm, nb) = (trial.markers.len(), trial.rigid_bodies.len());
    let Some(v) = state.trials.get_mut(t) else {
        return Changes::NONE;
    };
    let rule = |cur: Option<usize>, r: ActiveRule, n: usize| match r {
        ActiveRule::Keep => cur,
        ActiveRule::Clamp => cur.and_then(|c| if c >= n { n.checked_sub(1) } else { Some(c) }),
        ActiveRule::Last => n.checked_sub(1),
    };
    let marker = rule(v.active_marker, after.marker, nm);
    let body = rule(v.active_body, after.body, nb);
    let changed = marker != v.active_marker || body != v.active_body;
    v.active_marker = marker;
    v.active_body = body;
    if after.marker == ActiveRule::Last && marker.is_some() {
        v.active_body = None;
    }
    if after.reload {
        // reloadListFromObject: selectPoint / selectBody clear the selection and select the
        // active row; the current row then clears the other active index.
        let mut sel = Selection::default();
        if let Some(m) = v.active_marker.filter(|m| *m < nm) {
            sel.markers.insert(m);
            v.active_body = None;
        } else if let Some(b) = v.active_body.filter(|b| *b < nb) {
            sel.bodies.insert(b);
        }
        state.selection = sel;
        return Changes {
            selection: true,
            ..Changes::NONE
        };
    }
    Changes {
        selection: changed,
        ..Changes::NONE
    }
}

/// The dialog [`super::SHOW_DIALOG_ENV`] names (`points-marker`, `points-range`,
/// `points-import`), for the first trial, for screenshots.
#[must_use]
pub fn for_screenshot(which: &str, doc: &Document) -> Option<PointsDialog> {
    let selection = Selection::default();
    let cx = Ctx {
        doc,
        trial: 0,
        selection: &selection,
        n_cameras: doc.project().cameras.len(),
    };
    let t = doc.project().trials.first()?;
    let a = match which {
        "points-marker" if !t.markers.is_empty() => PointsAction::MarkerSettings { marker: 0 },
        "points-range" => PointsAction::ResetPoints,
        "points-import" => PointsAction::ImportExport,
        _ => return None,
    };
    match begin(&a, &cx) {
        Step::Open(d) => Some(d),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------
// Drawing.

impl PointsDialog {
    /// The window title of the step.
    #[must_use]
    pub fn title(&self) -> String {
        match &self.form {
            Form::Text { title, .. } | Form::Item { title, .. } | Form::Int { title, .. } => {
                title.clone()
            }
            Form::FromTo(_) => "Select Range".into(),
            Form::Confirm { .. } | Form::Choice { .. } => "Confirmation Required".into(),
            Form::Marker(_) => "Marker".into(),
            Form::ImportExport(_) => "Import/Export  Markers and Rigid Bodies".into(),
        }
    }

    /// The answer of the default button (OK, or the first choice) with the form's values.
    #[must_use]
    pub fn default_answer(&self) -> Answer {
        match &self.form {
            Form::Text { value, .. } => Answer::Text(value.clone()),
            Form::Item { selected, .. } => Answer::Item(*selected),
            Form::Int {
                value, min, max, ..
            } => Answer::Int((*value).clamp(*min, *max)),
            Form::FromTo(f) => {
                let (a, b) = f.frames();
                Answer::Range(a, b)
            }
            Form::Confirm { .. } => Answer::Yes,
            Form::Choice { .. } => Answer::Choice(0),
            Form::Marker(m) => Answer::Marker(m.properties()),
            Form::ImportExport(d) => Answer::ImportExport(d.clone()),
        }
    }

    /// The result the default button gives (tests: `Dialogs::accept`).
    #[must_use]
    pub fn accept(&self) -> DialogResult {
        self.result(self.default_answer())
    }

    fn result(&self, answer: Answer) -> DialogResult {
        DialogResult::Points(PointsOutcome {
            flow: self.flow.clone(),
            answer,
        })
    }
}

/// What drawing a step gave.
#[derive(Debug, Default)]
pub struct Shown {
    pub result: Option<DialogResult>,
    /// Closed without an answer (Cancel, Escape, the close button).
    pub close: bool,
    /// The last used folder changed.
    pub settings_changed: bool,
}

/// Draws the step on screen as a modal dialog.
#[allow(clippy::too_many_lines)]
pub fn show(d: &mut PointsDialog, ctx: &egui::Context, env: &mut DialogEnv<'_>) -> Shown {
    let mut out = Shown::default();
    let mut answer: Option<Answer> = None;
    let title = d.title();
    let icons = env.icons;
    let modal = form::modal(ctx, egui::Id::unique("xma_points_modal"), |ui| {
        if form::dialog_title(ui, &title) {
            out.close = true;
        }
        match &mut d.form {
            Form::Text { label, value, .. } => {
                ui.set_min_width(320.0);
                let l = ui.label(label.as_str());
                let name = if label.is_empty() {
                    title.clone()
                } else {
                    label.clone()
                };
                let r = ui
                    .add(egui::TextEdit::singleline(value).desired_width(300.0))
                    .labelled_by(l.id)
                    .accessible_name(name);
                if r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                    answer = Some(Answer::Text(value.clone()));
                }
                ok_cancel(ui, &mut answer, &mut out.close, || {
                    Answer::Text(value.clone())
                });
            }
            Form::Item {
                label,
                items,
                selected,
                ..
            } => {
                ui.set_min_width(320.0);
                let l = ui.label(label.as_str()).id;
                let refs: Vec<&str> = items.iter().map(String::as_str).collect();
                if let Some(i) =
                    crate::qt::combo_box(ui, "points_item", Some(*selected), &refs, None, Some(l))
                {
                    *selected = i;
                }
                let s = *selected;
                ok_cancel(ui, &mut answer, &mut out.close, || Answer::Item(s));
            }
            Form::Int {
                label,
                value,
                min,
                max,
                title,
            } => {
                ui.set_min_width(320.0);
                if !label.is_empty() {
                    ui.label(label.as_str());
                }
                if let Some(v) = form::spin_box(ui, true, *value, *min, *max, 100.0, title) {
                    *value = v;
                }
                let v = (*value).clamp(*min, *max);
                ok_cancel(ui, &mut answer, &mut out.close, || Answer::Int(v));
            }
            Form::FromTo(f) => match f.ui(ui) {
                Some(FromToAnswer::Ok) => {
                    let (a, b) = f.frames();
                    answer = Some(Answer::Range(a, b));
                }
                Some(FromToAnswer::Cancel) => out.close = true,
                None => {}
            },
            Form::Confirm { message } => {
                ui.set_max_width(520.0);
                ui.horizontal(|ui| {
                    ui.add(icons.image(ui.ctx(), Icon::Alert, 48.0));
                    ui.add(egui::Label::new(message.as_str()).wrap());
                });
                ok_cancel(ui, &mut answer, &mut out.close, || Answer::Yes);
            }
            Form::Choice { message, buttons } => {
                ui.set_max_width(560.0);
                ui.horizontal(|ui| {
                    ui.add(icons.image(ui.ctx(), Icon::Alert, 48.0));
                    ui.add(egui::Label::new(message.as_str()).wrap());
                });
                ui.add_space(4.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if form::button(ui, true, "Cancel", 80.0).clicked() {
                        out.close = true;
                    }
                    for (i, b) in buttons.iter().enumerate().rev() {
                        if form::button(ui, true, b, 80.0).clicked() {
                            answer = Some(Answer::Choice(i));
                        }
                    }
                });
            }
            Form::Marker(m) => match m.ui(ui) {
                Some(MarkerAnswer::Ok) => answer = Some(Answer::Marker(m.properties())),
                Some(MarkerAnswer::Cancel) => out.close = true,
                None => {}
            },
            Form::ImportExport(dlg) => {
                match dlg.ui(ui, env.settings, env.files, &mut out.settings_changed) {
                    Some(ImportExportAnswer::Ok) => {
                        answer = Some(Answer::ImportExport(dlg.clone()));
                    }
                    Some(ImportExportAnswer::Cancel) => out.close = true,
                    None => {}
                }
            }
        }
    });
    if modal.escaped() {
        out.close = true;
    }
    if let Some(a) = answer {
        out.result = Some(d.result(a));
    }
    out
}

/// "OK" and "Cancel" on the right, OK last (`QDialogButtonBox` of `QInputDialog`).
fn ok_cancel(
    ui: &mut egui::Ui,
    answer: &mut Option<Answer>,
    close: &mut bool,
    ok: impl FnOnce() -> Answer,
) {
    ui.add_space(6.0);
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        let okr = form::button(ui, true, "OK", 80.0);
        let cancel = form::button(ui, true, "Cancel", 80.0);
        if cancel.clicked() {
            *close = true;
        } else if okr.clicked() {
            *answer = Some(ok());
        }
    });
}
