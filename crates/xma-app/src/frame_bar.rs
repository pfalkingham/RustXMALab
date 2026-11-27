//! The frame bar under the camera views, and frame navigation.
//!
//! Ported control by control from `SequenceNavigationFrame.ui` and
//! ui/SequenceNavigationFrame.cpp. One row, left to right:
//!
//! | Col | Control | Shows |
//! |---|---|---|
//! | 0 | `toolButtonPrev` | prev_blue.png: previous frame |
//! | 1 | `toolButtonPlayBackward` | playbackward_blue.png, checkable: play backward |
//! | 4 | `toolButtonStop` | stop_blue.png: stop |
//! | 5 | `toolButtonPlay` | play_blue.png, checkable: play |
//! | 6 | `toolButtonNext` | next_blue.png: next frame |
//! | 7 | `spinBoxFrame` | the frame, 1-based |
//! | 8 | `toolButtonFrameStart` | the start frame; click to set it (min width 30) |
//! | 9 | `horizontalSlider` | the frame, 0-based, page step 1 |
//! | 10 | `toolButtonFrameEnd` | the end frame; click to set it (min width 30) |
//!
//! Behaviour (`workspaceChanged`, `activeTrialChanged`, `setStartEndSequence`,
//! `activeFrameChanged`, `changeFrame`, the play timer):
//!
//! - Undistortion: hidden. Calibration: shown when there is more than one calibration
//!   image, over those images, without the play buttons and with the start and end buttons
//!   disabled. Marker tracking: shown when there are trials, over the active trial's start
//!   to end frame.
//! - Previous is disabled at the first frame of the range, Next at the last.
//! - Playback: Play and Play backward stop any playback, then step one frame per tick (the
//!   C++ uses a 1 ms timer, so it runs as fast as frames can be shown; here one step per
//!   displayed frame) until the end of the range, where playback stops. Stop unchecks both.
//!   Every workspace or trial change stops playback.

use crate::actions::Action;
use crate::icons::Icon;
use crate::panels::Cx;
use crate::state::{AppState, Changes, Playback, ProjectFacts, Workspace};

/// The frames the bar runs over, 0-based and inclusive, and what it shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameRange {
    /// First frame (0-based): the trial's start frame - 1, or 0.
    pub first: usize,
    /// Last frame (0-based).
    pub last: usize,
    /// `maxFrame`: number of frames (trial or calibration images).
    pub max_frame: usize,
    /// The current frame (0-based).
    pub current: usize,
    /// Marker tracking (play buttons shown, start and end buttons enabled).
    pub tracking: bool,
}

/// The range for the current state, or `None` when the bar is hidden.
#[must_use]
pub fn frame_range(state: &AppState, facts: &ProjectFacts) -> Option<FrameRange> {
    match state.workspace {
        Workspace::Undistortion => None,
        Workspace::Calibration => {
            let n = facts.nb_images_calibration;
            (n > 1).then(|| FrameRange {
                first: 0,
                last: n - 1,
                max_frame: n,
                current: state.active_frame_calibration.unwrap_or(0).min(n - 1),
                tracking: false,
            })
        }
        Workspace::MarkerTracking => {
            if facts.n_trials() == 0 {
                return None;
            }
            let v = state.trial_view()?;
            let first = v.start_frame.saturating_sub(1);
            let last = v.end_frame.saturating_sub(1).max(first);
            Some(FrameRange {
                first,
                last,
                max_frame: v.n_frames,
                current: v.active_frame,
                tracking: true,
            })
        }
    }
}

/// Is the frame bar shown (`MainWindow::workspaceChanged`)?
#[must_use]
pub fn visible(state: &AppState, facts: &ProjectFacts) -> bool {
    match state.workspace {
        Workspace::Undistortion => false,
        Workspace::Calibration => facts.nb_images_calibration > 1,
        Workspace::MarkerTracking => facts.n_trials() > 0,
    }
}

/// Ported from ui/SequenceNavigationFrame.cpp, SequenceNavigationFrame::changeFrame.
/// Frames outside the range are ignored.
pub fn change_frame(state: &mut AppState, facts: &ProjectFacts, f: i64) -> Changes {
    match state.workspace {
        Workspace::Calibration => {
            let n = facts.nb_images_calibration as i64;
            if (0..n).contains(&f) {
                return state.change_active_frame_calibration(Some(f as usize), true);
            }
        }
        Workspace::MarkerTracking => {
            if facts.n_trials() > 0
                && let Some(v) = state.trial_view()
                && f >= v.start_frame as i64 - 1
                && f < v.end_frame as i64
            {
                return state.change_active_frame_trial(Some(f as usize), false);
            }
        }
        Workspace::Undistortion => {}
    }
    Changes::NONE
}

/// The frame the slider shows (`horizontalSlider->value()`).
#[must_use]
pub fn current_frame(state: &AppState, facts: &ProjectFacts) -> Option<usize> {
    frame_range(state, facts).map(|r| r.current)
}

/// Ported from `on_toolButtonNext_clicked`.
pub fn next(state: &mut AppState, facts: &ProjectFacts) -> Changes {
    match current_frame(state, facts) {
        Some(c) => change_frame(state, facts, c as i64 + 1),
        None => Changes::NONE,
    }
}

/// Ported from `on_toolButtonPrev_clicked`.
pub fn prev(state: &mut AppState, facts: &ProjectFacts) -> Changes {
    match current_frame(state, facts) {
        Some(c) => change_frame(state, facts, c as i64 - 1),
        None => Changes::NONE,
    }
}

/// Ported from ui/SequenceNavigationFrame.cpp, SequenceNavigationFrame::moveNFrames: move
/// by `n`, clamped to the calibration images or the trial's range.
pub fn move_n_frames(state: &mut AppState, facts: &ProjectFacts, n: i64) -> Changes {
    match state.workspace {
        Workspace::Calibration => {
            let nb = facts.nb_images_calibration as i64;
            let f = state.active_frame_calibration.unwrap_or(0) as i64 + n;
            let f = f.max(0);
            let f = if f >= nb { nb - 1 } else { f };
            change_frame(state, facts, f)
        }
        Workspace::MarkerTracking => {
            let Some(v) = state.trial_view().copied() else {
                return Changes::NONE;
            };
            if facts.n_trials() == 0 {
                return Changes::NONE;
            }
            let f = v.active_frame as i64 + n;
            let f = f.max(v.start_frame as i64 - 1);
            let f = f.min(v.end_frame as i64 - 1);
            change_frame(state, facts, f)
        }
        Workspace::Undistortion => Changes::NONE,
    }
}

/// Ported from ui/SequenceNavigationFrame.cpp, SequenceNavigationFrame::setStartEndSequence,
/// for the active trial: sets the range (ignored when invalid) and moves the frame into it.
///
/// The C++ comparisons mix 1-based and 0-based frames (`startFrame > activeFrame`,
/// `endFrame < activeFrame`); they are kept.
pub fn set_start_end(state: &mut AppState, start: usize, end: usize) -> Changes {
    let Some(v) = state.trial_view().copied() else {
        return Changes::NONE;
    };
    if end < start || start == 0 || end > v.n_frames {
        return Changes::NONE;
    }
    if let Some(view) = state.trial_view_mut() {
        view.start_frame = start;
        view.end_frame = end;
    }
    let mut ch = Changes {
        trial_range: true,
        ..Changes::NONE
    };
    if start > v.active_frame {
        ch |= state.change_active_frame_trial(Some(start - 1), false);
    } else if end < v.active_frame {
        ch |= state.change_active_frame_trial(Some(end - 1), false);
    }
    ch
}

/// Ported from `on_toolButtonStop_clicked` (play tag 0, both play buttons unchecked).
pub fn stop(state: &mut AppState) {
    state.playback = Playback::Stopped;
}

/// Ported from `on_toolButtonPlay_clicked` / `on_toolButtonPlayBackward_clicked`.
pub fn play(state: &mut AppState, forward: bool) {
    stop(state);
    state.playback = if forward {
        Playback::Forward
    } else {
        Playback::Backward
    };
}

/// Ported from `play_update`, one timer tick: step, then stop at the end of the range.
pub fn play_tick(state: &mut AppState, facts: &ProjectFacts) -> Changes {
    let ch = match state.playback {
        Playback::Stopped => return Changes::NONE,
        Playback::Forward => next(state, facts),
        Playback::Backward => prev(state, facts),
    };
    if let Some(r) = frame_range(state, facts) {
        let at_end = match state.playback {
            Playback::Forward => r.current >= r.last,
            Playback::Backward => r.current <= r.first,
            Playback::Stopped => false,
        };
        if at_end {
            stop(state);
        }
    } else {
        stop(state);
    }
    ch
}

/// Height of the bar, frame included: one row of controls in egui's group frame.
#[must_use]
pub fn height(style: &egui::Style) -> f32 {
    style.spacing.interact_size.y + egui::Frame::group(style).total_margin().sum().y
}

/// Draws the bar.
pub fn ui(ui: &mut egui::Ui, cx: &mut Cx<'_>) {
    let Some(r) = frame_range(cx.state, cx.facts) else {
        return;
    };
    crate::qt::styled_frame(ui, |ui| {
        ui.horizontal_centered(|ui| {
            let icon = |ui: &mut egui::Ui,
                        cx: &Cx<'_>,
                        ic: Icon,
                        enabled: bool,
                        checked: bool,
                        tip: &str| {
                let img = cx.icons.image(ui.ctx(), ic, 16.0);
                ui.add_enabled(
                    enabled,
                    egui::Button::image(img)
                        .frame_when_inactive(checked)
                        .selected(checked),
                )
                .on_hover_text(tip)
                .clicked()
            };
            if icon(
                ui,
                cx,
                Icon::PrevBlue,
                r.current != r.first,
                false,
                "Previous frame",
            ) {
                cx.raise(Action::PrevFrame);
            }
            if r.tracking {
                let back = cx.state.playback == Playback::Backward;
                if icon(ui, cx, Icon::PlayBackwardBlue, true, back, "Play backward") {
                    cx.raise(Action::PlayBackward);
                }
                if icon(ui, cx, Icon::StopBlue, true, false, "Stop") {
                    cx.raise(Action::StopPlayback);
                }
                let fwd = cx.state.playback == Playback::Forward;
                if icon(ui, cx, Icon::PlayBlue, true, fwd, "Play") {
                    cx.raise(Action::PlayForward);
                }
            }
            if icon(
                ui,
                cx,
                Icon::NextBlue,
                r.current != r.last,
                false,
                "Next frame",
            ) {
                cx.raise(Action::NextFrame);
            }
            let mut v = r.current as i64 + 1;
            if crate::qt::spin_box(
                ui,
                &mut v,
                r.first as i64 + 1,
                r.last as i64 + 1,
                46.0,
                "Frame",
            ) {
                cx.raise(Action::GoToFrame((v - 1).max(0) as usize));
            }
            if ui
                .add_enabled(
                    r.tracking,
                    egui::Button::new((r.first + 1).to_string())
                        .frame_when_inactive(false)
                        .min_size(egui::vec2(30.0, 0.0)),
                )
                .on_hover_text("Set start frame")
                .clicked()
            {
                cx.raise(Action::SetStartFrame);
            }
            let end_text = (r.last + 1).to_string();
            let end_w = 30.0_f32.max(8.0 * end_text.len() as f32 + 12.0);
            let slider_w = (ui.available_width() - end_w - 8.0).max(40.0);
            let mut s = r.current;
            let slider = egui::Slider::new(&mut s, r.first..=r.last.max(r.first))
                .show_value(false)
                .step_by(1.0);
            ui.spacing_mut().slider_width = slider_w;
            if ui.add(slider).accessible_name("Frame slider").changed() && s != r.current {
                cx.raise(Action::GoToFrame(s));
            }
            if ui
                .add_enabled(
                    r.tracking,
                    egui::Button::new(end_text)
                        .frame_when_inactive(false)
                        .min_size(egui::vec2(30.0, 0.0)),
                )
                .on_hover_text("Set end frame")
                .clicked()
            {
                cx.raise(Action::SetEndFrame);
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TrialView;

    fn tracking_state() -> (AppState, ProjectFacts) {
        let state = AppState {
            workspace: Workspace::MarkerTracking,
            trials: vec![TrialView {
                active_frame: 0,
                start_frame: 1,
                end_frame: 5,
                n_frames: 10,
                active_marker: None,
                active_body: None,
            }],
            active_trial: Some(0),
            active_frame_trial: Some(0),
            ..AppState::default()
        };
        let facts = ProjectFacts {
            trial_names: vec!["T".into()],
            trial_is_default: vec![false],
            ..ProjectFacts::default()
        };
        (state, facts)
    }

    #[test]
    fn playback_runs_to_the_end_and_stops() {
        let (mut s, f) = tracking_state();
        play(&mut s, true);
        for _ in 0..10 {
            play_tick(&mut s, &f);
        }
        assert_eq!(s.trial_view().unwrap().active_frame, 4);
        assert_eq!(s.playback, Playback::Stopped);
    }

    #[test]
    fn move_n_clamps_to_range() {
        let (mut s, f) = tracking_state();
        move_n_frames(&mut s, &f, 10);
        assert_eq!(s.active_frame_trial, Some(4));
        move_n_frames(&mut s, &f, -10);
        assert_eq!(s.active_frame_trial, Some(0));
    }

    #[test]
    fn start_end_moves_frame_into_range() {
        let (mut s, f) = tracking_state();
        set_start_end(&mut s, 3, 8);
        assert_eq!(s.active_frame_trial, Some(2));
        let r = frame_range(&s, &f).unwrap();
        assert_eq!((r.first, r.last), (2, 7));
        assert!(!set_start_end(&mut s, 5, 11).trial_range);
    }
}
