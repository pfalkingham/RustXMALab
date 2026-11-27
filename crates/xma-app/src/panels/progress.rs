//! The Progress panel.
//!
//! Ported from `ProgressDockWidget.ui` and ui/ProgressDialog.cpp. Two controls in one row:
//! `progressBar` (range 0-100; busy indicator while the job gives no progress; its text is
//! hidden until the first `setProgress`) and `cancelButton` "Cancel" (shown only for a
//! cancelable job). The panel's title is the job's description (`setWindowTitle(key)`).
//! The shell shows the panel only while a job runs (`showProgressbar` / `closeProgressbar`).

use super::{Cx, Panel, PanelId};

/// The Progress panel.
#[derive(Debug, Default)]
pub struct ProgressPanel;

impl Panel for ProgressPanel {
    fn id(&self) -> PanelId {
        PanelId::Progress
    }

    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut Cx<'_>) {
        let Some(job) = cx.jobs.current() else {
            return;
        };
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            let cancel_w = if job.cancelable { 80.0 } else { 0.0 };
            let w = (ui.available_width() - cancel_w - 8.0).max(40.0);
            let bar = match job.progress() {
                // setTextVisible(true) on the first setProgress.
                Some(p) => egui::ProgressBar::new(p).show_percentage(),
                // QProgressBar with min == max == 0: busy indicator.
                None => egui::ProgressBar::new(0.3).animate(true),
            };
            ui.add_sized([w, ui.spacing().interact_size.y], bar);
            if job.cancelable {
                let cancel = ui.add_enabled(!job.is_canceled(), egui::Button::new("Cancel"));
                if cancel.clicked() {
                    // ProgressDialog::on_cancelButton_clicked.
                    job.cancel();
                }
            }
        });
    }

    fn dynamic_title(&self, cx: &Cx<'_>) -> Option<String> {
        cx.jobs.current().map(crate::jobs::JobInfo::display_title)
    }
}
