//! `CameraSelector` ("Select Cameras"): which cameras are shown, from the workspace bar's
//! camera settings button.
//!
//! Ported control by control from `CameraSelector.ui` and ui/CameraSelector.cpp, with
//! ui/WorkspaceNavigationFrame.cpp, `on_toolButtonCameraSettings_clicked`:
//!
//! - `frame` (row 0, both columns): one check box per camera, named as the camera,
//!   checked when it is visible (built in the constructor).
//! - Row 1: "Cancel" (column 0), "OK" (column 1).
//!
//! OK sets each camera's visibility (`Camera::setVisible`, `MainWindow::setCameraVisible`);
//! the camera views and the Detailed View are then laid out again from
//! `AppState::camera_visible`. Visibility is view state of this session, not stored in the
//! project. Cancel, Escape and the close button change nothing.

use super::form;

/// The dialog's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraSelector {
    /// Camera names, in project order.
    pub names: Vec<String>,
    /// The check boxes.
    pub checked: Vec<bool>,
}

/// What the buttons did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraSelectorAnswer {
    Ok,
    Cancel,
}

impl CameraSelector {
    /// Ported from ui/CameraSelector.cpp, CameraSelector::CameraSelector. A camera without
    /// a visibility entry is visible.
    #[must_use]
    pub fn new(names: Vec<String>, visible: &[bool]) -> Self {
        let checked = (0..names.len())
            .map(|c| visible.get(c).copied().unwrap_or(true))
            .collect();
        Self { names, checked }
    }

    /// Draws the form below the title row; returns the button pressed.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> Option<CameraSelectorAnswer> {
        ui.set_min_width(280.0);
        form::styled_frame(ui, |ui| {
            ui.set_min_width(260.0);
            for (name, c) in self.names.iter().zip(self.checked.iter_mut()) {
                if let Some(v) = form::check_box(ui, true, *c, name) {
                    *c = v;
                }
            }
        });
        ui.add_space(4.0);
        let mut answer = None;
        ui.columns(2, |cols| {
            if form::button(&mut cols[0], true, "Cancel", 120.0).clicked() {
                answer = Some(CameraSelectorAnswer::Cancel);
            }
            if form::button(&mut cols[1], true, "OK", 120.0).clicked() {
                answer = Some(CameraSelectorAnswer::Ok);
            }
        });
        answer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxes_follow_the_visible_cameras() {
        let d = CameraSelector::new(vec!["a".into(), "b".into(), "c".into()], &[true, false]);
        assert_eq!(d.checked, [true, false, true]);
    }
}
