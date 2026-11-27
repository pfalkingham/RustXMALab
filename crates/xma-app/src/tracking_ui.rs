//! What the tracking buttons, their shortcuts and the clicks in the views ask for: the
//! rules of `WizardDigitizationFrame` (the buttons, `trackPointsShortcut`, `setDialog`) and
//! `GLCameraView::mousePressEvent` (the clicks of the Marker tracking workspace), as plain
//! functions of the interface state and the document. The Toolbox page, the views and the
//! application's dispatch call them and raise what they return.
//!
//! Fixed, not copied (parity ledger rows):
//!
//! - **T7**: in XMALab 3 the tracking buttons and the shortcuts between steps act on a
//!   running run (start another over it, or edit points between steps). Here, while a run
//!   is open, the only button or shortcut that does anything is a second press of the
//!   running continuous button (or its shortcut), which stops it; Escape and the frame bar's
//!   Stop stop it too. Clicks in the views place nothing while a run is open (as in XMALab 3
//!   during a step).
//! - **T5**: a run never starts outside the trial's start and end frames (the engine refuses
//!   it; the frame shown is kept inside them, ledger D15).
//!
//! New in this program (not in XMALab 3): the two multi-camera groups ("multi-camera
//! selected", "multi-camera all"), shown while "Enable multi-camera tracking" is checked
//! and usable when the trial has markers and at least two calibrated cameras are shown
//! ([`multi_unavailable`]); they request [`TrackKind::MultiSelected`] and
//! [`TrackKind::Multi`], which run like the other kinds (parity ledger U167, A218).
//!
//! Removed (parity ledger U171, A226; the author, 26 November 2026): XMALab 3's merge-aware
//! groups ("merge-aware selected", "merge-aware all", `groupBox_VolumetricSelected` and
//! `groupBox_VolumetricAll`), superseded by the multi-camera groups.
//!
//! Ported from ui/WizardDigitizationFrame.cpp (`on_toolButton_*_clicked`,
//! `trackPointsShortcut`, `setDialog`, `addDigitizationPoint`, `moveDigitizationPoint`,
//! `selectDigitizationPoint`) and ui/GLCameraView.cpp (`mousePressEvent`).

use egui::Modifiers;
use xma_engine::Document;
use xma_engine::tracking::{Direction, Placement, TrackKind, TrackRequest};
use xma_model::{MarkerStatus, Point2};

use crate::actions::Action;
use crate::editing::EditRequest;
use crate::state::{AppState, ProjectFacts, Workspace};

/// The groups of tracking buttons on the Toolbox page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackGroup {
    /// `groupBox_Point` "single" (`trackType` 1).
    Single,
    /// `groupBox_RB` "selected" (2).
    Selected,
    /// `groupBox_All` "all" (3).
    All,
    /// "multi-camera selected": multi-camera tracking of the selected markers. New in this
    /// program.
    MultiSelected,
    /// "multi-camera all": multi-camera tracking of all markers. New in this program.
    MultiAll,
}

impl TrackGroup {
    pub const ALL: [TrackGroup; 5] = [
        TrackGroup::Single,
        TrackGroup::Selected,
        TrackGroup::All,
        TrackGroup::MultiSelected,
        TrackGroup::MultiAll,
    ];

    /// True for the two multi-camera groups (new in this program).
    #[must_use]
    pub fn is_multi(self) -> bool {
        matches!(self, TrackGroup::MultiSelected | TrackGroup::MultiAll)
    }

    /// The group box's title.
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            TrackGroup::Single => "single",
            TrackGroup::Selected => "selected",
            TrackGroup::All => "all",
            TrackGroup::MultiSelected => "multi-camera selected",
            TrackGroup::MultiAll => "multi-camera all",
        }
    }
}

/// One tracking button: its group, direction and whether it runs on (the checkable
/// backward and forward buttons) or does one step (previous, next).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrackButton {
    pub group: TrackGroup,
    pub direction: Direction,
    pub continuous: bool,
}

impl TrackButton {
    #[must_use]
    pub const fn new(group: TrackGroup, direction: Direction, continuous: bool) -> Self {
        Self {
            group,
            direction,
            continuous,
        }
    }

    /// The four buttons of a group in the form's order: previous, next, backward, forward.
    #[must_use]
    pub fn of_group(group: TrackGroup) -> [TrackButton; 4] {
        [
            Self::new(group, Direction::Backward, false),
            Self::new(group, Direction::Forward, false),
            Self::new(group, Direction::Backward, true),
            Self::new(group, Direction::Forward, true),
        ]
    }

    /// The button a running request belongs to.
    #[must_use]
    pub fn of_request(r: &TrackRequest) -> Self {
        let group = match r.kind {
            TrackKind::Single(_) => TrackGroup::Single,
            TrackKind::Selected(_) => TrackGroup::Selected,
            TrackKind::All => TrackGroup::All,
            TrackKind::MultiSelected(_) => TrackGroup::MultiSelected,
            TrackKind::Multi => TrackGroup::MultiAll,
        };
        Self::new(group, r.direction, r.continuous)
    }

    /// The button a tracking shortcut clicks (`trackPointsShortcut`): 1 and 2 the single
    /// group, with Ctrl the selected group, with Alt the continuous buttons.
    #[must_use]
    pub fn of_action(a: &Action) -> Option<Self> {
        use Direction::{Backward, Forward};
        use TrackGroup::{Selected, Single};
        Some(match a {
            Action::TrackActiveNext => Self::new(Single, Forward, false),
            Action::TrackActivePrev => Self::new(Single, Backward, false),
            Action::TrackActiveForwardAll => Self::new(Single, Forward, true),
            Action::TrackActiveBackwardAll => Self::new(Single, Backward, true),
            Action::TrackSelectedNext => Self::new(Selected, Forward, false),
            Action::TrackSelectedPrev => Self::new(Selected, Backward, false),
            Action::TrackSelectedForwardAll => Self::new(Selected, Forward, true),
            Action::TrackSelectedBackwardAll => Self::new(Selected, Backward, true),
            _ => return None,
        })
    }

    /// The button's tooltip in the form, also its accessible name. The multi-camera
    /// groups' texts are new, worded as those of XMALab 3's merge-aware groups (removed,
    /// parity ledger U171).
    #[must_use]
    pub fn tip(self) -> &'static str {
        use Direction::{Backward, Forward};
        use TrackGroup::*;
        match (self.group, self.direction, self.continuous) {
            (Single, Backward, false) => "track single point to previous frame",
            (Single, Forward, false) => "track single point to next frame",
            (Single, Backward, true) => "track single point backward",
            (Single, Forward, true) => "track single point forward",
            (Selected, Backward, false) => "Track selected points to previous frame",
            (Selected, Forward, false) => "Track selected points to next frame",
            (Selected, Backward, true) => "Track selected points to backward",
            (Selected, Forward, true) => "Track selected points forward",
            (All, Backward, false) => "Track all points to previous frame",
            (All, Forward, false) => "Track all points to next frame",
            (All, Backward, true) => "Track all points backward",
            (All, Forward, true) => "Track all points to forward",
            (MultiSelected, Backward, false) => {
                "Track selected points to previous frame using all cameras together"
            }
            (MultiSelected, Forward, false) => {
                "Track selected points to next frame using all cameras together"
            }
            (MultiSelected, Backward, true) => {
                "Track selected points backward using all cameras together"
            }
            (MultiSelected, Forward, true) => {
                "Track selected points forward using all cameras together"
            }
            (MultiAll, Backward, false) => {
                "Track all points to previous frame using all cameras together"
            }
            (MultiAll, Forward, false) => {
                "Track all points to next frame using all cameras together"
            }
            (MultiAll, Backward, true) => "Track all points backward using all cameras together",
            (MultiAll, Forward, true) => "Track all points forward using all cameras together",
        }
    }
}

/// What the Toolbox page shows (`setDialog`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageState {
    /// Not calibrated: "You first have to calibrate your cameras".
    NotCalibrated,
    /// No active trial: "You first have to add a trial" and "Add Trial".
    NoTrial,
    /// A trial without markers: "You first have to add points...".
    NoMarkers,
    /// Markers to track. `single_enabled`: the "single" group is enabled.
    Tracking { single_enabled: bool },
}

/// Ported from ui/WizardDigitizationFrame.cpp, setDialog. The "single" group is enabled
/// when the active marker has a point (status above `UNDEFINED`) at the frame in any
/// camera, shown or not; with no active marker it keeps its state, enabled. While a run
/// is open the C++ leaves the group as it was (`trackDirection != 0`): here it stays
/// enabled so that a single-marker run can be stopped from its button.
#[must_use]
pub fn page_state(facts: &ProjectFacts, doc: Option<&Document>, state: &AppState) -> PageState {
    if !facts.calibrated_or_none() {
        return PageState::NotCalibrated;
    }
    let (Some(doc), Some(t)) = (doc, state.active_trial) else {
        return PageState::NoTrial;
    };
    let Some(trial) = doc.project().trials.get(t) else {
        return PageState::NoTrial;
    };
    if trial.markers.is_empty() {
        return PageState::NoMarkers;
    }
    if state.running_track.is_some() {
        return PageState::Tracking {
            single_enabled: true,
        };
    }
    let single_enabled = match (
        state.trial_view().and_then(|v| v.active_marker),
        state.active_frame_trial,
    ) {
        (Some(m), Some(f)) => trial.markers.get(m).is_some_and(|mk| {
            mk.views.iter().any(|v| {
                v.status
                    .get(f)
                    .is_some_and(|s| *s > MarkerStatus::Undefined)
            })
        }),
        (Some(_), None) => false,
        _ => true,
    };
    PageState::Tracking { single_enabled }
}

/// Why the multi-camera groups (new in this program) cannot be used on the active trial, as
/// one sentence for the check box's tooltip; `None` when they can
/// ([`xma_engine::tracking::multi_unavailable`]: markers and at least two calibrated cameras
/// that are shown, not the XMALab 3 mode, which the window never opens a project in). The
/// cameras shown are `AppState::camera_visible`, which [`press`] hands to the run.
#[must_use]
pub fn multi_unavailable(doc: Option<&Document>, state: &AppState) -> Option<String> {
    let (Some(doc), Some(t)) = (doc, state.active_trial) else {
        return Some("Multi-camera tracking needs a trial.".to_owned());
    };
    xma_engine::tracking::multi_unavailable(doc, t, &state.camera_visible)
}

/// The parameters of a multi-camera run started from the window: the method's defaults but
/// for the four set on the Toolbox's tracking page, `split_touching` ("Separate touching
/// dots", ledger U169), `stat_links`, `distance_placement` and `keep_dot_rule` (ledger U170;
/// A224, A225, A221 rule (b)), taken from `state` (the page writes them each frame). The only
/// place the window makes `MultiTrackParams`; the other kinds do not read them.
#[must_use]
pub fn multi_params(state: &AppState) -> xma_vision::joint_tracking::MultiTrackParams {
    xma_vision::joint_tracking::MultiTrackParams {
        split_touching: state.multi_split_touching,
        stat_links: state.multi_stat_links,
        distance_placement: state.multi_distance_placement,
        keep_dot_rule: state.multi_keep_dot_rule,
        ..xma_vision::joint_tracking::MultiTrackParams::default()
    }
}

/// Is the button enabled now? Its group is shown and enabled (`setDialog`). `shown` says
/// whether an optional group's check box is checked: the multi-camera groups (new) are
/// enabled only while "Enable multi-camera tracking" is and [`multi_unavailable`] says
/// nothing; the other groups ignore it. While a run is open only the running continuous button is (T7 fixed); while the
/// refinement runs, none.
#[must_use]
pub fn button_enabled(
    facts: &ProjectFacts,
    doc: Option<&Document>,
    state: &AppState,
    b: TrackButton,
    shown: bool,
) -> bool {
    if state.workspace != Workspace::MarkerTracking {
        return false;
    }
    let PageState::Tracking { single_enabled } = page_state(facts, doc, state) else {
        return false;
    };
    if let Some(running) = state.running_track {
        return running == b && b.continuous;
    }
    if state.disable_draw {
        return false;
    }
    match b.group {
        TrackGroup::Single => single_enabled,
        TrackGroup::Selected | TrackGroup::All => true,
        TrackGroup::MultiSelected | TrackGroup::MultiAll => {
            shown && multi_unavailable(doc, state).is_none()
        }
    }
}

/// Is the button shown checked? A continuous button while its run goes on
/// (`setChecked(true)` in its handler, `uncheckTrackButtons` at the end).
#[must_use]
pub fn button_checked(state: &AppState, b: TrackButton) -> bool {
    b.continuous && state.running_track == Some(b)
}

/// A press of a tracking button, or of its shortcut: what to request. `None` when the
/// button is disabled or the press does nothing (the C++ handlers' early returns: no active
/// marker for "single").
///
/// Ported from ui/WizardDigitizationFrame.cpp, the `on_toolButton_*_clicked` handlers
/// (`:1010-1277`) with `trackPointsShortcut` (`:920-1008`): a second press of a checked
/// button stops the run.
#[must_use]
pub fn press(
    facts: &ProjectFacts,
    doc: &Document,
    state: &AppState,
    b: TrackButton,
    shown: bool,
) -> Option<EditRequest> {
    if !button_enabled(facts, Some(doc), state, b, shown) {
        return None;
    }
    if state.running_track.is_some() {
        return Some(EditRequest::StopTracking);
    }
    let t = state.active_trial?;
    let view = state.trial_view()?;
    let selected = || {
        state
            .selection
            .markers
            .iter()
            .copied()
            .collect::<Vec<usize>>()
    };
    let kind = match b.group {
        TrackGroup::Single => TrackKind::Single(view.active_marker?),
        TrackGroup::Selected => TrackKind::Selected(selected()),
        TrackGroup::All => TrackKind::All,
        TrackGroup::MultiSelected => TrackKind::MultiSelected(selected()),
        TrackGroup::MultiAll => TrackKind::Multi,
    };
    Some(EditRequest::StartTracking(TrackRequest {
        trial: t,
        kind,
        direction: b.direction,
        continuous: b.continuous,
        start_frame: view.active_frame,
        end_frame: None,
        visible_cameras: state.camera_visible.clone(),
    }))
}

/// What a left click in a view of the Marker tracking workspace does.
#[derive(Clone, Debug, PartialEq)]
pub enum ClickOutcome {
    /// Change the project.
    Request(EditRequest),
    /// Make this marker the active one (Shift + click, E4).
    Select(usize),
    /// Nothing.
    Nothing,
}

/// A left click at image position `pos` in camera `camera`'s view (`detail`: the Detailed
/// View) with `modifiers`:
///
/// | Gesture | Camera view | Detailed View |
/// |---|---|---|
/// | Ctrl (Command on macOS) | E1: a point for the next marker without one here, detected | E3 |
/// | Shift | E4: the nearest marker within 20 pixels becomes active | E4 |
/// | none (or Alt) | E2: the active marker's point here, detected (status `SET`) | E3: the active marker's point exactly here (status `MANUAL`) |
///
/// Nothing while a run is open (`disable_draw`) or outside Marker tracking; E2 and E3 need
/// an active marker.
///
/// Ported from ui/GLCameraView.cpp, GLCameraView::mousePressEvent (`:211-326`), with
/// ui/WizardDigitizationFrame.cpp, addDigitizationPoint, moveDigitizationPoint and
/// selectDigitizationPoint.
#[must_use]
pub fn click(
    doc: &Document,
    state: &AppState,
    camera: usize,
    pos: (f64, f64),
    modifiers: Modifiers,
    detail: bool,
) -> ClickOutcome {
    if state.disable_draw || state.workspace != Workspace::MarkerTracking {
        return ClickOutcome::Nothing;
    }
    let (Some(trial), Some(view)) = (state.active_trial, state.trial_view()) else {
        return ClickOutcome::Nothing;
    };
    if doc.project().trials.get(trial).is_none() {
        return ClickOutcome::Nothing;
    }
    let frame = view.active_frame;
    let (x, y) = pos;
    let place = |placement| {
        ClickOutcome::Request(EditRequest::PlacePoint {
            trial,
            camera,
            frame,
            x,
            y,
            placement,
        })
    };
    if modifiers.command && !detail {
        place(Placement::AddNext {
            active: view.active_marker,
        })
    } else if modifiers.shift {
        match xma_engine::view::overlays::marker_near(doc, trial, camera, frame, Point2::new(x, y))
        {
            Some(m) => ClickOutcome::Select(m),
            None => ClickOutcome::Nothing,
        }
    } else {
        match view.active_marker {
            Some(m) if detail => place(Placement::Manual(m)),
            Some(m) => place(Placement::Detect(m)),
            None => ClickOutcome::Nothing,
        }
    }
}
