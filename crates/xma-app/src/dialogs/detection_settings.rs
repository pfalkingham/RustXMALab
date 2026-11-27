//! View > DetectionSettings: the non-modal window that follows the active marker and shows
//! the stages of the detection at the last point placed by a click.
//!
//! Ported control by control from `DetectionSettings.ui` (a 381 × 345 dialog, grid layout)
//! and ui/DetectionSettings.cpp:
//!
//! | Row | Control | Text |
//! |---|---|---|
//! | 0 | `comboBox` (the stage shown) | "Input", "Blurred Orignial", "Difference (Original - Blurred)", "Median filtered Difference", "Thresholded Difference", "Blurred Thresholded", "Detected" |
//! | 3 | `label_4`, `comboBox_Method` | "Detectionmethod"; the seven methods (E55) |
//! | 4 | `label`, `spinBox_ThresholdOffset` | "Threshold offset in percent", 0 to 100 (E56) |
//! | 5 | `checkBoxCrosshair` | "draw crosshair", checked |
//! | 8 | `frame` with `label_Image` | the stage image, 3 times enlarged, nearest neighbour; "Method not supported" |
//!
//! The preview (`DetectionSettings::update`, [`xma_engine::tracking::detection_preview`]) is
//! computed when a click moves the active marker (E2, E3) while the window is open, and again
//! with the same point when the method, the threshold or the crosshair box changes. A change
//! of the active marker clears it (`activePointChanged`). Nothing in the project changes but
//! the marker's method and threshold.
//!
//! Fixed, not copied (parity ledger rows):
//!
//! - **D12**: XMALab 3 drops a marker whose method is not 0, 2 or 5 (`setMarker`), so the
//!   method box shows its method and changing it writes nothing. Here the window holds the
//!   active marker whatever its method: the method box and the threshold box write it; the
//!   preview says "Method not supported" for the methods without one.
//! - **Out-of-range stage**: choosing "Detected" when the detection found no contour reads
//!   past the end of the C++ image list (`ui/DetectionSettings.cpp:129`, `:62-63`); here the
//!   image area says "Nothing detected".

use xma_engine::tracking::DetectionPreview;
use xma_model::DetectionMethod;

use crate::state::AppState;

/// The stage combo box's items.
pub const STAGES: [&str; 7] = [
    "Input",
    "Blurred Orignial",
    "Difference (Original - Blurred)",
    "Median filtered Difference",
    "Thresholded Difference",
    "Blurred Thresholded",
    "Detected",
];

/// The tooltips of the method box's items (`DetectionSettings::DetectionSettings`).
pub const METHOD_TIPS: [&str; 7] = [
    "Optimized for dark, spherical beads in X-rays",
    "Uses OpenCV SimpleBlobDetector to find circular blobs",
    "Inverts image to track bright white beads on a dark background",
    "Uses Shi-Tomasi corner finder for sharp angles (e.g. checkerboards)",
    "Turns off automatic snapping. Marker stays exactly where clicked",
    "Calculates exact center of mass of dark pixels instead of fitting a contour",
    "OpenCV SimpleBlobDetector for bright circular blobs",
];

/// The window's title.
pub const TITLE: &str = "Detection Settings";

/// What the image area shows.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Preview {
    /// Nothing yet (no click since the window opened or the marker changed).
    #[default]
    Empty,
    /// "Method not supported".
    NotSupported,
    /// The stages of the detection.
    Stages(DetectionPreview),
    /// The image could not be read.
    Failed(String),
}

/// What the user did in the window, for the application to carry out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectionEvent {
    /// E55: the method box changed.
    Method(DetectionMethod),
    /// E56: the threshold box changed.
    Threshold(i32),
    /// The window was closed.
    Close,
}

/// The window's state; it lives in the application (`XmaApp::detection`).
#[derive(Default)]
pub struct DetectionSettingsWindow {
    /// The marker shown: (trial, marker).
    pub held: Option<(usize, usize)>,
    /// Where the last preview was computed: (camera, frame, x, y).
    pub last: Option<(usize, usize, f64, f64)>,
    /// `comboBox`: the stage shown.
    pub stage: usize,
    /// `checkBoxCrosshair` (checked in the form).
    pub crosshair: bool,
    pub preview: Preview,
    /// The preview is to be computed again (the application does it after the edits).
    pub dirty: bool,
    /// The stage image's texture and what it shows.
    texture: Option<(egui::TextureHandle, usize, u64)>,
    /// Counts previews, to know when the texture is stale.
    generation: u64,
    /// Drawn inside the main window (screenshot mode).
    pub embedded: bool,
}

impl std::fmt::Debug for DetectionSettingsWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetectionSettingsWindow")
            .field("held", &self.held)
            .field("last", &self.last)
            .field("stage", &self.stage)
            .field("crosshair", &self.crosshair)
            .finish_non_exhaustive()
    }
}

impl DetectionSettingsWindow {
    #[must_use]
    pub fn new() -> Self {
        Self {
            crosshair: true,
            ..Self::default()
        }
    }

    /// Opened (`showEvent`): holds the active marker, shows no image.
    pub fn opened(&mut self, state: &AppState) {
        self.held = active(state);
        self.clear();
    }

    fn clear(&mut self) {
        self.last = None;
        self.preview = Preview::Empty;
        self.dirty = false;
    }

    /// Each frame: follows the active marker (`activePointChanged`), clearing the preview
    /// when it changed.
    pub fn follow(&mut self, state: &AppState) {
        let now = active(state);
        if now != self.held {
            self.held = now;
            self.clear();
        }
    }

    /// A click moved marker `marker` (E2, E3) at `(x, y)` in `camera` at `frame`: the preview
    /// is computed there (`WizardDigitizationFrame::moveDigitizationPoint`).
    pub fn clicked(
        &mut self,
        state: &AppState,
        trial: usize,
        marker: usize,
        camera: usize,
        frame: usize,
        (x, y): (f64, f64),
    ) {
        if !state.detection_settings_open {
            return;
        }
        self.held = Some((trial, marker));
        self.last = Some((camera, frame, x, y));
        self.dirty = true;
    }

    /// Sets the computed preview.
    pub fn set_preview(&mut self, p: Preview) {
        self.preview = p;
        self.generation += 1;
        self.dirty = false;
    }

    /// Draws the window's contents; returns what the user did.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        doc: Option<&xma_engine::Document>,
    ) -> Vec<DetectionEvent> {
        let mut events = Vec::new();
        let marker = self.held.and_then(|(t, m)| {
            doc.and_then(|d| d.project().trials.get(t))
                .and_then(|tr| tr.markers.get(m))
        });
        let label_w = 170.0;
        // Row 0: the stage shown.
        let stage_tip =
            "Visualization tool: Preview intermediate steps of the image processing pipeline.";
        let w = ui.available_width();
        if let Some(i) = combo(
            ui,
            "xma_detection_stage",
            self.stage,
            &STAGES,
            None,
            w,
            true,
            "Stage",
            stage_tip,
        ) {
            self.stage = i;
        }
        ui.add_space(8.0);
        // Row 3: the method (E55).
        ui.horizontal(|ui| {
            ui.add_sized([label_w, 20.0], egui::Label::new("Detectionmethod"));
            let items: Vec<&str> = (0..7)
                .filter_map(|c| DetectionMethod::from_code(c).map(DetectionMethod::label))
                .collect();
            let selected = marker.map_or(0, |m| m.method.code() as usize);
            let w = ui.available_width();
            if let Some(i) = combo(
                ui,
                "xma_detection_method",
                selected,
                &items,
                Some(&METHOD_TIPS),
                w,
                marker.is_some(),
                "Detectionmethod",
                "Select the computer vision algorithm used to snap to the marker.",
            ) && let Some(m) = DetectionMethod::from_code(i as i32)
            {
                events.push(DetectionEvent::Method(m));
            }
        });
        // Row 4: the threshold offset (E56).
        ui.horizontal(|ui| {
            ui.add_sized(
                [label_w, 20.0],
                egui::Label::new("Threshold offset in percent"),
            );
            let value = marker.map_or(0, |m| i64::from(m.threshold_offset));
            if let Some(v) = crate::dialogs::form::spin_box(
                ui,
                marker.is_some(),
                value,
                0,
                100,
                50.0,
                "Threshold offset in percent",
            ) {
                events.push(DetectionEvent::Threshold(v as i32));
            }
        });
        // Row 5: the crosshair.
        if ui.checkbox(&mut self.crosshair, "draw crosshair").changed() && self.last.is_some() {
            self.dirty = true;
        }
        ui.add_space(8.0);
        // Row 8: the image.
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_min_size(egui::vec2(ui.available_width(), 120.0));
            ui.vertical_centered(|ui| self.image(ui));
        });
        events
    }

    fn image(&mut self, ui: &mut egui::Ui) {
        match &self.preview {
            Preview::Empty => {}
            Preview::NotSupported => {
                ui.label("Method not supported");
            }
            Preview::Failed(e) => {
                ui.label(e.as_str());
            }
            Preview::Stages(p) => {
                let Some(px) = p.stages.get(self.stage) else {
                    // The C++ reads past its list here.
                    ui.label("Nothing detected");
                    return;
                };
                let stale = self
                    .texture
                    .as_ref()
                    .is_none_or(|(_, s, g)| *s != self.stage || *g != self.generation);
                if stale {
                    let img = egui::ColorImage::from_gray([p.size, p.size], px);
                    let tex = ui.ctx().load_texture(
                        "xma_detection_stage",
                        img,
                        egui::TextureOptions::NEAREST,
                    );
                    self.texture = Some((tex, self.stage, self.generation));
                }
                if let Some((tex, _, _)) = &self.texture {
                    // getPixmap: 3 times, nearest neighbour.
                    let side = 3.0 * p.size as f32;
                    ui.add(
                        egui::Image::new((tex.id(), egui::vec2(side, side)))
                            .alt_text(STAGES.get(self.stage).copied().unwrap_or("")),
                    );
                }
            }
        }
    }

    /// Draws the window when open: its own operating-system window, or (embedded, and under
    /// the test harness) a window inside the main one.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        doc: Option<&xma_engine::Document>,
    ) -> Vec<DetectionEvent> {
        let mut events = Vec::new();
        let mut close = false;
        if self.embedded {
            let mut open = true;
            egui::Window::new(TITLE)
                .id(egui::Id::unique("xma_detection_settings_embedded"))
                .default_size([381.0, 345.0])
                .default_pos([60.0, 60.0])
                .collapsible(false)
                .open(&mut open)
                .show(ctx, |ui| events = self.ui(ui, doc));
            close = !open;
        } else {
            let builder = egui::ViewportBuilder::default()
                .with_title(TITLE)
                .with_app_id(crate::APP_ID)
                .with_inner_size([381.0, 345.0])
                .with_minimize_button(false)
                .with_always_on_top();
            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("xma_detection_settings"),
                builder,
                |ui, _class| {
                    // This window's share of `qt::combo_wheel`'s frame hook.
                    crate::qt::combo_wheel_begin_frame(&ui.ctx().clone());
                    egui::CentralPanel::default().show(ui, |ui| events = self.ui(ui, doc));
                    if ui.ctx().input(|i| i.viewport().close_requested()) {
                        close = true;
                    }
                    crate::links::collect(ui.ctx());
                },
            );
        }
        if close {
            events.push(DetectionEvent::Close);
        }
        events
    }
}

/// A `QComboBox` with item tooltips, named `name`; returns the newly chosen index.
#[allow(clippy::too_many_arguments)]
fn combo(
    ui: &mut egui::Ui,
    id: &str,
    selected: usize,
    items: &[&str],
    tips: Option<&[&str]>,
    width: f32,
    enabled: bool,
    name: &str,
    tip: &str,
) -> Option<usize> {
    let mut chosen = None;
    ui.add_enabled_ui(enabled, |ui| {
        let r = egui::ComboBox::from_id_salt(id)
            .selected_text(items.get(selected).copied().unwrap_or(""))
            .width(width)
            .show_ui(ui, |ui| {
                for (i, it) in items.iter().enumerate() {
                    let mut r = ui.selectable_label(i == selected, *it);
                    if let Some(t) = tips.and_then(|t| t.get(i)) {
                        r = r.on_hover_text(*t);
                    }
                    if r.clicked() && i != selected {
                        chosen = Some(i);
                    }
                }
            });
        if chosen.is_none() {
            chosen = crate::qt::combo_wheel(ui, &r.response, Some(selected), items.len(), |_| true);
        }
        let _ = r.response.on_hover_text(tip).accessible_name(name);
    });
    chosen
}

/// The active trial's active marker, if any.
fn active(state: &AppState) -> Option<(usize, usize)> {
    let t = state.active_trial?;
    Some((t, state.trial_view()?.active_marker?))
}
