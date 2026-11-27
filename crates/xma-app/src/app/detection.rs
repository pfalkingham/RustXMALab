//! The application's side of View > DetectionSettings
//! (`crate::dialogs::detection_settings`): opening and closing it, drawing it, carrying out
//! its edits (E55, E56) and computing its preview.

use xma_engine::Edit;
use xma_engine::edit::MarkerProperties;

use super::XmaApp;
use crate::actions::Action;
use crate::dialogs::detection_settings::{DetectionEvent, Preview};
use crate::editing::{self, EditRequest};

impl XmaApp {
    /// Opens or closes View > DetectionSettings (`MainWindow::on_actionDetectionSettings_triggered`;
    /// opening holds the active marker, `DetectionSettings::showEvent`).
    pub fn set_detection_settings_open(&mut self, open: bool) {
        self.state.detection_settings_open = open;
        if open {
            self.detection.opened(&self.state);
        }
    }

    /// Once per frame: follows the active marker, draws the window when it is open, and
    /// carries out what was done in it.
    pub(super) fn detection_settings_frame(&mut self, ctx: &egui::Context) {
        if std::mem::take(&mut self.dialogs.detection_settings_requested) {
            self.detection.embedded = true;
            self.set_detection_settings_open(true);
        }
        if !self.state.detection_settings_open {
            return;
        }
        // MainWindow::workspaceChanged hides it outside Marker tracking.
        if !Action::ToggleDetectionSettings.cpp_enabled(&self.action_context()) {
            self.state.detection_settings_open = false;
            return;
        }
        self.detection.follow(&self.state);
        let events = self.detection.show(ctx, self.session.doc.as_ref());
        for e in events {
            match e {
                DetectionEvent::Close => self.state.detection_settings_open = false,
                DetectionEvent::Method(method) => {
                    if let Some((trial, marker)) = self.detection.held {
                        self.edits.push(EditRequest::Apply(Edit::SetMarkerMethod {
                            trial,
                            markers: vec![marker],
                            method,
                        }));
                        self.detection.dirty |= self.detection.last.is_some();
                    }
                }
                DetectionEvent::Threshold(v) => {
                    if let Some((trial, marker)) = self.detection.held {
                        self.edits
                            .push(EditRequest::Apply(Edit::SetMarkerProperties {
                                trial,
                                marker,
                                properties: MarkerProperties {
                                    threshold_offset: Some(v),
                                    ..MarkerProperties::default()
                                },
                            }));
                        self.detection.dirty |= self.detection.last.is_some();
                    }
                }
            }
        }
    }

    /// Computes the preview at the last clicked point with the held marker's method and
    /// threshold (`DetectionSettings::update`).
    pub fn update_detection_preview(&mut self) {
        let (Some((trial, marker)), Some((camera, frame, x, y))) =
            (self.detection.held, self.detection.last)
        else {
            self.detection.dirty = false;
            return;
        };
        let files = self.trial_files(trial);
        let Some(doc) = self.session.doc.as_ref() else {
            return;
        };
        let mut images = editing::trial_images(&mut self.frames, &files);
        let crosshair = self.detection.crosshair;
        let p = match xma_engine::tracking::detection_preview(
            doc,
            images.as_mut(),
            trial,
            marker,
            camera,
            frame,
            x,
            y,
            crosshair,
        ) {
            Ok(Some(p)) => Preview::Stages(p),
            Ok(None) => Preview::NotSupported,
            Err(e) => Preview::Failed(e),
        };
        drop(images);
        self.detection.set_preview(p);
    }
}
