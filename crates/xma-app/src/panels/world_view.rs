//! The 3D world view panel.
//!
//! Ported control by control from `WorldViewDockWidget.ui` and ui/WorldViewDockWidget.cpp:
//!
//! | Row | Control | Text, icon |
//! |---|---|---|
//! | 0 | `openGL` (`WorldViewDockGLWidget`) | the 3D view ([`crate::world_view_draw`]) |
//! | 1 | `frame` › `checkBoxEnable` | "enable": the view follows its own timeline |
//! | 1 | `toolButtonPrev`, `toolButtonPlayBackward`, `toolButtonStop`, `toolButtonPlay`, `toolButtonNext` | prev_blue, playbackward_blue (checkable), stop_blue, play_blue (checkable), next_blue; auto-raised |
//! | 1 | `spinBoxFrame`, `horizontalSlider` | the view's own frame, 1-based and 0-based |
//! | 2 | `label` "Focal plane distance", `horizontalSlider_FocusPlane` | 1 to 10000, 200 |
//!
//! Behaviour: "enable" is enabled in Marker tracking with a trial and otherwise unchecked
//! and disabled (`workspaceChanged`). While it is checked the timeline controls are enabled
//! over the active trial's start to end frame (`setTimeline`, `setStartEndSequence`);
//! previous is disabled at the first frame and next at the last (`changeFrame`); play steps
//! the view's frame each repaint and starts again at the other end (`play_update`). The
//! timeline moves only the 3D view's frame, never the main window's. Every workspace or
//! trial change stops playback.
//!
//! `setStartEndSequence` compares the trial's start frame (1-based) with the trial's active
//! frame (0-based) to decide whether to move the view's frame into the range; here the
//! view's own frame is kept inside the range.

use super::{AppEvent, Cx, Panel, PanelId};
use crate::dialogs::form;
use crate::icons::Icon;
use crate::state::Workspace;

/// The 3D world view panel.
#[derive(Debug)]
pub struct WorldViewPanel {
    /// `checkBoxEnable`.
    pub enabled: bool,
    /// The view's frame, 0-based (`horizontalSlider`).
    pub frame: usize,
    /// `play_tag`: 1 forward, -1 backward, 0 stopped.
    pub play: i8,
    /// `horizontalSlider_FocusPlane` (`WorldViewDockGLWidget::setFocalPlaneDistance`).
    pub focal_plane: i64,
}

impl Default for WorldViewPanel {
    fn default() -> Self {
        Self {
            enabled: false,
            frame: 0,
            play: 0,
            focal_plane: 200,
        }
    }
}

impl WorldViewPanel {
    /// The timeline range, 0-based inclusive: the active trial's start to end frame, when
    /// the check box may be used.
    fn range(cx: &Cx<'_>) -> Option<(usize, usize)> {
        if cx.state.workspace != Workspace::MarkerTracking || cx.facts.n_trials() == 0 {
            return None;
        }
        let v = cx.state.trial_view()?;
        Some((
            v.start_frame.saturating_sub(1),
            v.end_frame
                .saturating_sub(1)
                .max(v.start_frame.saturating_sub(1)),
        ))
    }

    /// `on_toolButtonStop_clicked`.
    pub fn stop(&mut self) {
        self.play = 0;
    }

    /// `play_update`: one step, starting again at the other end of the range.
    fn play_step(&mut self, first: usize, last: usize) {
        if self.play > 0 {
            self.frame = if self.frame >= last {
                first
            } else {
                self.frame + 1
            };
        } else if self.play < 0 {
            self.frame = if self.frame <= first {
                last
            } else {
                self.frame - 1
            };
        }
    }
}

impl Panel for WorldViewPanel {
    fn id(&self) -> PanelId {
        PanelId::WorldView
    }

    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut Cx<'_>) {
        let range = Self::range(cx);
        if range.is_none() {
            self.enabled = false;
            self.stop();
        }
        if let Some((first, last)) = range {
            self.frame = self.frame.clamp(first, last);
            if self.enabled && self.play != 0 {
                self.play_step(first, last);
                ui.ctx().request_repaint();
            }
        }
        let timeline = self.enabled && range.is_some();
        let (first, last) = range.unwrap_or((0, 98));

        let bottom_h = 2.0 * ui.spacing().interact_size.y + 20.0;
        let view_h = (ui.available_height() - bottom_h).max(40.0);
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), view_h),
            egui::Sense::hover(),
        );
        // `WorldViewDockWidget::draw`: the view's own frame while "enable" is checked, else
        // the main window's.
        let own = (self.enabled && range.is_some()).then_some(self.frame);
        crate::world_view_draw::show(
            ui,
            cx,
            rect,
            crate::world_view_draw::ViewSettings {
                own_frame: own,
                focal_plane: self.focal_plane as f64,
            },
        );

        form::styled_frame(ui, |ui| {
            ui.horizontal(|ui| {
                if form::check_box(ui, range.is_some(), self.enabled, "enable").is_some() {
                    // on_checkBoxEnable_clicked: setTimeline (which stops when disabling).
                    self.enabled = !self.enabled;
                    if !self.enabled {
                        self.stop();
                    }
                }
                ui.add_space(12.0);
                let icons = cx.icons;
                if form::icon_button(
                    ui,
                    icons,
                    Icon::PrevBlue,
                    timeline && self.frame != first,
                    false,
                    "World view previous frame",
                    24.0,
                    true,
                )
                .clicked()
                {
                    self.frame = self.frame.saturating_sub(1).max(first);
                }
                if form::icon_button(
                    ui,
                    icons,
                    Icon::PlayBackwardBlue,
                    timeline,
                    self.play < 0,
                    "World view play backward",
                    24.0,
                    true,
                )
                .clicked()
                {
                    self.play = -1;
                }
                if form::icon_button(
                    ui,
                    icons,
                    Icon::StopBlue,
                    timeline,
                    false,
                    "World view stop",
                    24.0,
                    true,
                )
                .clicked()
                {
                    self.stop();
                }
                if form::icon_button(
                    ui,
                    icons,
                    Icon::PlayBlue,
                    timeline,
                    self.play > 0,
                    "World view play",
                    24.0,
                    true,
                )
                .clicked()
                {
                    self.play = 1;
                }
                if form::icon_button(
                    ui,
                    icons,
                    Icon::NextBlue,
                    timeline && self.frame != last,
                    false,
                    "World view next frame",
                    24.0,
                    true,
                )
                .clicked()
                {
                    self.frame = (self.frame + 1).min(last);
                }
                if let Some(v) = form::spin_box(
                    ui,
                    timeline,
                    self.frame as i64 + 1,
                    first as i64 + 1,
                    last as i64 + 1,
                    50.0,
                    "World view frame",
                ) {
                    self.frame = (v - 1).max(0) as usize;
                }
                let mut s = self.frame;
                ui.spacing_mut().slider_width = (ui.available_width() - 8.0).max(40.0);
                let r = ui
                    .add_enabled(
                        timeline,
                        egui::Slider::new(&mut s, first..=last.max(first))
                            .show_value(false)
                            .step_by(1.0),
                    )
                    .accessible_name("World view frame slider");
                if r.changed() {
                    self.frame = s;
                }
            });
            ui.horizontal(|ui| {
                let l = ui.label("Focal plane distance").id;
                ui.spacing_mut().slider_width = (ui.available_width() - 8.0).max(40.0);
                ui.add(
                    egui::Slider::new(&mut self.focal_plane, 1..=10000)
                        .show_value(false)
                        .step_by(1.0),
                )
                .labelled_by(l)
                .accessible_name("Focal plane distance");
            });
        });
    }

    fn on_event(&mut self, event: &AppEvent, _cx: &mut Cx<'_>) {
        match event {
            AppEvent::Changed(ch) => {
                if ch.workspace {
                    // workspaceChanged: stop; the timeline is off until checked again.
                    self.stop();
                    self.enabled = false;
                    self.frame = 0;
                }
                if ch.active_trial {
                    // activeTrialChanged: stop (the range follows in `ui`).
                    self.stop();
                }
            }
            AppEvent::ProjectOpened | AppEvent::ProjectClosed => {
                self.stop();
                self.enabled = false;
                self.frame = 0;
            }
        }
    }
}
