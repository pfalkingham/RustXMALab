//! The Toolbox panel.
//!
//! Ported from `WizardDockWidget.ui` and ui/WizardDockWidget.cpp. The form is empty: the
//! constructor adds the three wizard pages, and `workspaceChanged` shows one of them:
//!
//! | Workspace | Page |
//! |---|---|
//! | Undistortion | [`undistortion`] (`WizardUndistortionFrame`) |
//! | Calibration | [`calibration`] (`WizardCalibrationCubeFrame`), only for an internal calibration |
//! | Marker tracking | [`digitization`] (`WizardDigitizationFrame`) |
//!
//! The pages react to state changes as the C++ slots do: the undistortion page sets its
//! display choices when the project becomes undistorted (`undistortionChanged`), and the
//! calibration page's constructor sets the points display to "detected cube points
//! (distorted)".

pub mod calibration;
pub mod digitization;
pub mod undistortion;

use super::{AppEvent, Cx, Panel, PanelId};
use crate::state::{UndistortionState, Workspace};

/// The Toolbox panel.
#[derive(Debug, Default)]
pub struct ToolboxPanel {
    pub calibration: calibration::CalibrationPage,
    pub digitization: digitization::DigitizationPage,
}

impl Panel for ToolboxPanel {
    fn id(&self) -> PanelId {
        PanelId::Toolbox
    }

    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut Cx<'_>) {
        // The page inside egui's window margin, so that its frames do not touch the panel's
        // edges.
        let margin = ui.spacing().window_margin;
        egui::ScrollArea::vertical()
            .id_salt("toolbox_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Frame::NONE
                    .inner_margin(margin)
                    .show(ui, |ui| match cx.state.workspace {
                        Workspace::Undistortion => undistortion::ui(ui, cx),
                        Workspace::Calibration => self.calibration.ui(ui, cx),
                        Workspace::MarkerTracking => self.digitization.ui(ui, cx),
                    });
            });
    }

    fn on_event(&mut self, event: &AppEvent, cx: &mut Cx<'_>) {
        // WizardCalibrationCubeFrame's constructor (the dock is created with the window).
        self.calibration.construct(cx);
        if let AppEvent::Changed(ch) = event
            && ch.undistortion
            && cx.state.undistortion == UndistortionState::Undistorted
        {
            // WizardUndistortionFrame::undistortionChanged(UNDISTORTED).
            cx.change(undistortion::on_undistorted);
        }
    }
}
