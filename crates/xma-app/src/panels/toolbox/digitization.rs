//! The Toolbox page of the Marker tracking workspace.
//!
//! Ported control by control from `WizardDigitizationFrame.ui` and
//! ui/WizardDigitizationFrame.cpp (`setDialog`). One column, a styled frame round it:
//!
//! | Row | Control | Text, icon, tooltip |
//! |---|---|---|
//! | 0 | `label` | the reason nothing can be tracked yet (hidden when markers exist) |
//! | 2 | `groupBox_Point` "single" | `toolButton_PointPrev`, `toolButton_PointNext` / `toolButton_PointBack`, `toolButton_PointForw` |
//! | 3 | `groupBox_RB` "selected" | `toolButton_RBPrev`, `toolButton_RBNext` / `toolButton_RBBack`, `toolButton_RBForw` |
//! | 4 | `groupBox_All` "all" | `toolButton_AllPrev`, `toolButton_AllNext` / `toolButton_AllBack`, `toolButton_AllForw` |
//! | 5 | `groupBox_Interpolation` "Interpolation" | `toolButton_SetInterpolation` "Set Interpolation" / `toolButton_InterpolateActive` "active", `toolButton_InterpolateAll` "all" |
//! | 6 | `pushButton` | "Add Trial" |
//! | 7 | `checkBox_VolumetricTracking` "Enable merge-aware tracking" | removed, parity ledger U171 |
//! | 8 | `label_VolumetricHint` "Resolves overlapping markers using the other cameras" | removed, parity ledger U171 |
//! | 9 | `groupBox_VolumetricSelected` "merge-aware selected" | removed, parity ledger U171 |
//! | 11 | `groupBox_VolumetricAll` "merge-aware all" | removed, parity ledger U171 |
//! | (new) | check box | "Enable multi-camera tracking" (not in XMALab 3; where row 7 was) |
//! | (new) | hint label | "Follows each marker in all cameras together" |
//! | (new) | check box | "Separate touching dots" (not in XMALab 3; under the hint) |
//! | (new) | check box | "Recognise a lost marker by its distances" (not in XMALab 3) |
//! | (new) | check box | "Place a lost marker by its distances" (not in XMALab 3) |
//! | (new) | check box | "Remove old points on another marker's dot" (not in XMALab 3) |
//! | (new) | group "multi-camera selected" | four tool buttons as above |
//! | (new) | group "multi-camera all" | four tool buttons as above |
//!
//! Icons: previous frame `playbackward_blue`, next frame `play_blue`, backward
//! `backward_blue`, forward `forward_blue`; the backward and forward buttons are checkable.
//!
//! `setDialog`: with the cameras calibrated (or no calibration) and an active trial with
//! markers, the four groups and the multi-camera check box show and the label is hidden;
//! the "single" group is enabled when the active marker has a point in any camera at the
//! frame. Without markers, the label asks for points; without a trial, it asks for a trial
//! and "Add Trial" shows; uncalibrated, it asks for a calibration.
//!
//! Removed (parity ledger U171; merge-aware tracking left the engine too, ledger A226): XMALab
//! 3's "Enable merge-aware tracking" with its hint and its groups "merge-aware selected" and
//! "merge-aware all", superseded by multi-camera tracking (the author, 26 November 2026).
//!
//! New in this program (parity ledger U167): where the merge-aware block was, and built as it
//! was, "Enable multi-camera tracking" (unchecked at start) shows its hint and the groups
//! "multi-camera selected" and "multi-camera all" when checked. It is usable when the trial has markers and at least two calibrated cameras
//! are shown ([`crate::tracking_ui::multi_unavailable`]); otherwise an unticked box is
//! disabled and its tooltip adds the sentence saying what is missing. A ticked box can
//! always be unticked (outside a run): when the method becomes unavailable while it is
//! ticked (a camera hidden, say), the sentence replaces the hint under it and the groups'
//! buttons are disabled. Most of the method's settings are not shown: the defaults are used,
//! but for four check boxes under the hint while the box above is ticked, each disabled while
//! a run is open, and all four ticked at start (parity ledger U171). "Separate touching dots"
//! (parity ledger U169) sets `MultiTrackParams::split_touching` for every multi-camera run
//! started from the window. Under it (parity ledger U170): "Recognise a lost marker by its
//! distances" (`stat_links`, ledger A224), "Place a lost marker by its distances"
//! (`distance_placement`, ledger A225) and "Remove old points on another marker's dot"
//! (`keep_dot_rule`, rule (b) of ledger A221). Each is written to its `AppState` field
//! (`multi_split_touching`, `multi_stat_links`, `multi_distance_placement`,
//! `multi_keep_dot_rule`; all true while the multi-camera box is unticked), which
//! [`crate::tracking_ui::multi_params`] reads when a run starts.
//!
//! Working: the tracking buttons (E8 to E11, [`crate::tracking_ui`]: a press requests the
//! run, the continuous buttons show checked while their run goes on and a second press stops
//! it), the multi-camera check box with its four options and the interpolation group ("Set Interpolation" raises `Action::SetInterpolationRange`, which
//! the Plot panel handles; "active" and "all" request `Edit::Interpolate`, also from the I
//! and Shift+I keys through [`interpolate_request`]). "Add Trial" raises `Action::AddTrial`
//! (the New Trial dialog, as the "+" of the workspace bar).

use crate::dialogs::form;
use crate::icons::Icon;
use crate::panels::Cx;
use crate::tracking_ui::{self, TrackButton, TrackGroup};

/// Tooltip of "Enable multi-camera tracking" (new in this program).
pub const MULTI_CAMERA_TIP: &str = "Follows all markers in all cameras together instead of each camera on its own: a point is accepted only where the cameras agree with each other and with where the marker was heading, a doubtful point is left empty, and a lost marker is looked for again at later frames. New in this program; XMALab 3 has no such method.";

/// Tooltip of "Separate touching dots" (new in this program, ledger U169; on by default,
/// ledger U171).
pub const SPLIT_TOUCHING_TIP: &str = "Where a marker's dot touches another dark dot (a bead passing in front of it or behind it) and the two are detected as one shape, they are fitted as separate dots and the point goes on the marker's own dot, not on the centre of the joined shape. For round bead markers, and for one continuous run: tracking one frame at a time gets no benefit. New in this program; on by default. Untick it for faster runs, or if points on markers that are not round are moved off their centres.";

/// Tooltip of "Recognise a lost marker by its distances" (new in this program, ledger U170;
/// `MultiTrackParams::stat_links`, ledger A224).
pub const STAT_LINKS_TIP: &str = "A marker that was lost is accepted again only where its distances to the other markers fit. Ticked (the default): each distance is compared with what this run has measured for that pair of markers so far (its average, and how much it varied). Unticked: with the distances at one earlier frame, within 2 %, which refuses the right dot when the markers are on different bones or on skin. New in this program. On the Rat recording, ticked finds more lost markers again and puts more points on a wrong dot.";

/// Tooltip of "Place a lost marker by its distances" (new in this program, ledger U170;
/// `MultiTrackParams::distance_placement`, ledger A225).
pub const DISTANCE_PLACEMENT_TIP: &str = "A lost marker is normally looked for where its nearest neighbours' movement puts it, and not at all when they do not move as one rigid body. Ticked (the default): such a marker is looked for where its average distances to four or more followed markers put it. New in this program. On the Rat recording, ticked finds more lost markers again and puts more points on a wrong dot.";

/// Tooltip of "Remove old points on another marker's dot" (new in this program, ledger U170;
/// `MultiTrackParams::keep_dot_rule`, rule (b) of ledger A221).
pub const KEEP_DOT_RULE_TIP: &str = "A multi-camera run keeps the points already in the project that it cannot confirm. Ticked (the default): such a point is removed where it lies on the dot the run gave to another marker, unless the two markers are touching there. Unticked: it is kept. A trade: ticked removes old points that had slid onto another marker's dot, and also some good points of two markers that really share a dot in that camera.";

/// The page's own state.
#[derive(Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct DigitizationPage {
    /// "Enable multi-camera tracking" (new in this program; unchecked at start).
    pub multi_camera: bool,
    /// "Separate touching dots" (new in this program, ledger U169; checked at start, ledger
    /// U171): the multi-camera runs fit touching dots apart
    /// (`MultiTrackParams::split_touching`).
    pub split_touching: bool,
    /// "Recognise a lost marker by its distances" (new, ledger U170; checked at start, U171):
    /// `MultiTrackParams::stat_links` (ledger A224).
    pub stat_links: bool,
    /// "Place a lost marker by its distances" (new, ledger U170; checked at start, U171):
    /// `MultiTrackParams::distance_placement` (ledger A225).
    pub distance_placement: bool,
    /// "Remove old points on another marker's dot" (new, ledger U170; checked at start):
    /// `MultiTrackParams::keep_dot_rule` (ledger A221, rule (b)).
    pub keep_dot_rule: bool,
}

impl Default for DigitizationPage {
    fn default() -> Self {
        Self {
            multi_camera: false,
            split_touching: true,
            stat_links: true,
            distance_placement: true,
            keep_dot_rule: true,
        }
    }
}

pub use crate::tracking_ui::PageState;

/// The prompt shown when the trial has no markers. The C++ says COMMAND on every system
/// but Windows; COMMAND is the macOS key, Ctrl is used elsewhere.
#[must_use]
pub fn no_points_text() -> &'static str {
    if cfg!(target_os = "macos") {
        "You first have to add points by using COMMAND + left Click in the image or by using the point widget"
    } else {
        "You first have to add points by using CTRL + left Click in the image or by using the point widget"
    }
}

/// Ported from ui/WizardDigitizationFrame.cpp, setDialog ([`crate::tracking_ui::page_state`]).
#[must_use]
pub fn page_state(cx: &Cx<'_>) -> PageState {
    crate::tracking_ui::page_state(cx.facts, cx.doc, cx.state)
}

/// Text of `toolButton_SetInterpolation`.
const SET_INTERPOLATION_TIP_LABEL: &str = "Set Interpolation";
/// Tooltip of `toolButton_SetInterpolation`.
const SET_INTERPOLATION_TIP: &str = "Select a missing frame range in the plot (Shift+Drag), then click here (or Ctrl+I) to assign an interpolation method to the gap.";

/// E13 ("active", I) and E14 ("all", Shift+I): fills the gaps of the active marker, or of
/// every marker of the active trial, by each frame's interpolation method. `None` when there
/// is nothing to interpolate (no trial, no active marker, no markers), as the C++ does
/// nothing then. No question is asked.
///
/// Ported from ui/WizardDigitizationFrame.cpp, on_toolButton_InterpolateActive_clicked
/// (`:1284-1292`) and on_toolButton_InterpolateAll_clicked (`:1294-1301`). The engine's
/// `Marker::interpolate` keeps "untrackable" marks (cpp-editing D3, fixed there).
#[must_use]
pub fn interpolate_request(
    state: &crate::state::AppState,
    doc: &xma_engine::Document,
    all: bool,
) -> Option<crate::editing::EditRequest> {
    let t = state.active_trial?;
    let trial = doc.project().trials.get(t)?;
    let markers: Vec<usize> = if all {
        (0..trial.markers.len()).collect()
    } else {
        let m = state.trial_view().and_then(|v| v.active_marker)?;
        if m >= trial.markers.len() {
            return None;
        }
        vec![m]
    };
    if markers.is_empty() {
        return None;
    }
    Some(crate::editing::EditRequest::Apply(
        xma_engine::Edit::Interpolate { trial: t, markers },
    ))
}

/// The four tool buttons of one tracking group: previous, next / backward, forward
/// (`crate::tracking_ui`): enabled as the C++ enables them (T7 fixed: while a run is open only
/// its own button, checked, which stops it), a press raising the start or the stop.
fn track_group(ui: &mut egui::Ui, cx: &mut Cx<'_>, group: TrackGroup, shown: bool) {
    let icons = cx.icons;
    let mut pressed = None;
    form::group_box(ui, group.title(), |ui| {
        let w = ((ui.available_width() - 6.0) / 2.0).max(40.0);
        let [prev, next, back, forw] = TrackButton::of_group(group);
        let mut button = |ui: &mut egui::Ui, b: TrackButton, icon: Icon| {
            let enabled = tracking_ui::button_enabled(cx.facts, cx.doc, cx.state, b, shown);
            let checked = tracking_ui::button_checked(cx.state, b);
            if form::icon_button(ui, icons, icon, enabled, checked, b.tip(), w, false).clicked() {
                pressed = Some(b);
            }
        };
        ui.horizontal(|ui| {
            button(ui, prev, Icon::PlayBackwardBlue);
            button(ui, next, Icon::PlayBlue);
        });
        ui.horizontal(|ui| {
            button(ui, back, Icon::BackwardBlue);
            button(ui, forw, Icon::ForwardBlue);
        });
    });
    if let Some(b) = pressed
        && let Some(doc) = cx.doc
        && let Some(r) = tracking_ui::press(cx.facts, doc, cx.state, b, shown)
    {
        cx.request(r);
    }
}

impl DigitizationPage {
    /// Draws the page.
    pub fn ui(&mut self, ui: &mut egui::Ui, cx: &mut Cx<'_>) {
        let st = page_state(cx);
        form::styled_frame(ui, |ui| {
            let label = match st {
                PageState::NotCalibrated => Some("You first have to calibrate your cameras"),
                PageState::NoTrial => Some("You first have to add a trial"),
                PageState::NoMarkers => Some(no_points_text()),
                PageState::Tracking { .. } => None,
            };
            if let Some(l) = label {
                ui.vertical_centered(|ui| {
                    ui.add(egui::Label::new(l).wrap());
                });
            }
            let PageState::Tracking { .. } = st else {
                if st == PageState::NoTrial {
                    // `on_pushButton_clicked`: the New Trial dialog
                    // (`MainWindow::on_pushButtonNewTrial_clicked`).
                    let add = crate::actions::Action::AddTrial;
                    if form::wide_button(ui, add.implemented(), "Add Trial").clicked() {
                        cx.raise(add);
                    }
                }
                return;
            };
            for g in [TrackGroup::Single, TrackGroup::Selected, TrackGroup::All] {
                track_group(ui, cx, g, false);
            }
            // groupBox_Interpolation (E13, E14 and the way to E46). Enabled whenever it shows,
            // as in the C++, except while a tracking run holds the project.
            let interp_on = !cx.state.disable_draw;
            let mut interp = None;
            form::group_box(ui, "Interpolation", |ui| {
                if form::wide_button(ui, interp_on, SET_INTERPOLATION_TIP_LABEL)
                    .on_hover_text(SET_INTERPOLATION_TIP)
                    .on_disabled_hover_text(SET_INTERPOLATION_TIP)
                    .clicked()
                {
                    // on_toolButton_SetInterpolation_clicked: PlotWindow::setInterpolation.
                    cx.raise(crate::actions::Action::SetInterpolationRange);
                }
                ui.horizontal(|ui| {
                    let w = ((ui.available_width() - 6.0) / 2.0).max(40.0);
                    let tip = "Interpolate missing frames for the active marker";
                    if form::button(ui, interp_on, "active", w)
                        .on_hover_text(tip)
                        .on_disabled_hover_text(tip)
                        .clicked()
                    {
                        interp = Some(false);
                    }
                    let tip = "Interpolate missing frames for all markers";
                    if form::button(ui, interp_on, "all", w)
                        .on_hover_text(tip)
                        .on_disabled_hover_text(tip)
                        .clicked()
                    {
                        interp = Some(true);
                    }
                });
            });
            if let Some(all) = interp
                && let Some(doc) = cx.doc
                && let Some(r) = interpolate_request(cx.state, doc, all)
            {
                cx.request(r);
            }
            // New in this program, where XMALab 3's merge-aware block was (removed, ledger
            // U171) and built as it was. Disabled while a run is open (its groups would hide
            // a running button); while the trial cannot be tracked this way an unticked box
            // is disabled and a ticked one can still be unticked.
            let unavailable = tracking_ui::multi_unavailable(cx.doc, cx.state);
            let tip = match &unavailable {
                Some(why) => format!("{MULTI_CAMERA_TIP}\n\n{why}"),
                None => MULTI_CAMERA_TIP.to_owned(),
            };
            ui.add_enabled(
                cx.state.running_track.is_none() && (unavailable.is_none() || self.multi_camera),
                egui::Checkbox::new(&mut self.multi_camera, "Enable multi-camera tracking"),
            )
            .on_hover_text(&tip)
            .on_disabled_hover_text(&tip);
            if self.multi_camera {
                // The hint, or while the method cannot be used the sentence saying why (the
                // buttons below are then disabled).
                let text = unavailable
                    .as_deref()
                    .unwrap_or("Follows each marker in all cameras together");
                ui.vertical_centered(|ui| {
                    ui.add(egui::Label::new(text).wrap());
                });
                // New (ledger U169): the one setting of the method shown. Not while a run is
                // open, as the boxes above.
                ui.add_enabled(
                    cx.state.running_track.is_none(),
                    egui::Checkbox::new(&mut self.split_touching, "Separate touching dots"),
                )
                .on_hover_text(SPLIT_TOUCHING_TIP)
                .on_disabled_hover_text(SPLIT_TOUCHING_TIP);
                // New (ledger U170): three more settings of the method, under it and built
                // the same way.
                for (value, label, tip) in [
                    (
                        &mut self.stat_links,
                        "Recognise a lost marker by its distances",
                        STAT_LINKS_TIP,
                    ),
                    (
                        &mut self.distance_placement,
                        "Place a lost marker by its distances",
                        DISTANCE_PLACEMENT_TIP,
                    ),
                    (
                        &mut self.keep_dot_rule,
                        "Remove old points on another marker's dot",
                        KEEP_DOT_RULE_TIP,
                    ),
                ] {
                    ui.add_enabled(
                        cx.state.running_track.is_none(),
                        egui::Checkbox::new(value, label),
                    )
                    .on_hover_text(tip)
                    .on_disabled_hover_text(tip);
                }
                track_group(ui, cx, TrackGroup::MultiSelected, true);
                track_group(ui, cx, TrackGroup::MultiAll, true);
            }
            // What the next run started from the window uses (only the multi-camera kinds
            // read it).
            // While "Enable multi-camera tracking" is unticked these hold the boxes' start
            // values, all true (ledger U171); the boxes keep their own state for when it is
            // ticked again.
            cx.state.multi_split_touching = !self.multi_camera || self.split_touching;
            cx.state.multi_stat_links = !self.multi_camera || self.stat_links;
            cx.state.multi_distance_placement = !self.multi_camera || self.distance_placement;
            cx.state.multi_keep_dot_rule = !self.multi_camera || self.keep_dot_rule;
        });
    }
}
