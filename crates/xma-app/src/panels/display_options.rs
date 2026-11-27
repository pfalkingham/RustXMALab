//! The Display Options panel.
//!
//! Ported from `DisplayOptionsDockWidget.ui` and ui/DisplayOptionsDockWidget.cpp
//! (ui-map.md, "DisplayOptionsDockWidget"). One group box, "Drawing", with a grid of eight
//! check boxes and a tool button, each bound to the per-machine setting it reads and writes
//! (the [`UiSettings`](crate::settings::UiSettings) keys are the C++ names):
//!
//! | Control | Label | Setting |
//! |---|---|---|
//! | `checkBoxHide` | Hide all | `TrialDrawHideAll` (through [`Action::ToggleHideAll`], the H key) |
//! | `checkBoxMarkers` | Marker cross | `TrialDrawMarkers` |
//! | `checkBoxMarkerIds` | Marker Ids | `TrialDrawMarkerIds` |
//! | `checkBoxEpipolar` | Epipolar line | `TrialDrawEpipolar` |
//! | `checkBoxRigidBodyConstellation` | Rigid body constellations | `TrialDrawRigidBodyConstellation` |
//! | `checkBoxRigidBodyMeshmodels` | Meshmodels | `TrialDrawRigidBodyMeshmodels` |
//! | `checkBox_DrawFiltered` | Use filtered transformations | `TrialDrawFiltered` |
//! | `checkBox_VisualFilter` | filter images | `VisualFilterEnabled` |
//! | `toolButton_VisualFilter` | (settings icon) | opens the Visual Filter dialog |
//!
//! "Hide all" disables the seven controls below it (`toggleEnabled`). The C++ reads the
//! settings into the check boxes once, when the panel is made, and the H key keeps "Hide
//! all" in step; here every check box shows its setting on every frame, so they are always
//! in step. The camera views read the same settings.
//!
//! The rows of the grid share the group box's height equally, as Qt's grid layout spreads
//! fixed-height rows when the panel is taller than they need.

use egui::{Ui, vec2};

use super::{Cx, Panel, PanelId};
use crate::actions::Action;
use crate::icons::Icon;
use crate::settings::UiSettings;

/// The Display Options panel.
#[derive(Debug, Default)]
pub struct DisplayOptionsPanel;

/// A setting a check box writes.
pub type SettingField = fn(&mut UiSettings) -> &mut bool;

/// The check boxes below "Hide all", in form order, with the setting each writes.
pub const CHECKS: [(&str, SettingField); 7] = [
    ("Marker cross", |s| &mut s.trial_draw_markers),
    ("Marker Ids", |s| &mut s.trial_draw_marker_ids),
    ("Epipolar line", |s| &mut s.trial_draw_epipolar),
    ("Rigid body constellations", |s| {
        &mut s.trial_draw_rigid_body_constellation
    }),
    ("Meshmodels", |s| &mut s.trial_draw_rigid_body_meshmodels),
    ("Use filtered transformations", |s| {
        &mut s.trial_draw_filtered
    }),
    ("filter images", |s| &mut s.visual_filter_enabled),
];

impl Panel for DisplayOptionsPanel {
    fn id(&self) -> PanelId {
        PanelId::DisplayOptions
    }

    fn ui(&mut self, ui: &mut Ui, cx: &mut Cx<'_>) {
        let full = ui.available_rect_before_wrap();
        let margin = 6.0;
        let inner = full.shrink(margin);
        ui.scope_builder(egui::UiBuilder::new().max_rect(inner), |ui| {
            // groupBox_Interpolation_2, "Drawing": the title above a framed box.
            let title = ui.label("Drawing");
            let frame_rect = egui::Rect::from_min_max(
                egui::pos2(inner.min.x, title.rect.max.y + 4.0),
                inner.max,
            );
            ui.painter().rect(
                frame_rect,
                ui.visuals().widgets.noninteractive.corner_radius,
                egui::Color32::TRANSPARENT,
                ui.visuals().widgets.noninteractive.bg_stroke,
                egui::StrokeKind::Inside,
            );
            let grid = frame_rect.shrink2(ui.spacing().window_margin.sum() / 2.0);
            let row_h = (grid.height() / 8.0).max(ui.spacing().interact_size.y);
            let hide_all = cx.settings.trial_draw_hide_all;
            let mut row = 0usize;
            let row_rect = |r: usize| {
                egui::Rect::from_min_size(
                    egui::pos2(grid.min.x, grid.min.y + row_h * r as f32),
                    vec2(grid.width(), row_h),
                )
            };

            // checkBoxHide.
            let r = row_rect(row);
            row += 1;
            ui.scope_builder(egui::UiBuilder::new().max_rect(r), |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    let mut v = hide_all;
                    if ui.checkbox(&mut v, "Hide all").clicked() {
                        // on_checkBoxHide_clicked does what toggleHideAll (the H key) does.
                        cx.raise(Action::ToggleHideAll);
                    }
                });
            });

            for (i, (label, field)) in CHECKS.iter().enumerate() {
                let r = row_rect(row);
                row += 1;
                ui.scope_builder(egui::UiBuilder::new().max_rect(r), |ui| {
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        let mut v = *field(cx.settings);
                        let resp = ui.add_enabled(!hide_all, egui::Checkbox::new(&mut v, *label));
                        if resp.clicked() {
                            *field(cx.settings) = v;
                        }
                        // toolButton_VisualFilter, beside "filter images" (row 7, column 1).
                        if i == CHECKS.len() - 1 {
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    let img = cx.icons.image(ui.ctx(), Icon::Settings, 16.0);
                                    // Opens VisualFilterDialog (C67); disabled with
                                    // "Hide all", as the check boxes are.
                                    if ui
                                        .add_enabled(!hide_all, egui::Button::image(img))
                                        .accessible_name("Visual filter settings")
                                        .clicked()
                                    {
                                        cx.raise(Action::VisualFilterSettings);
                                    }
                                },
                            );
                        }
                    });
                });
            }
        });
        ui.advance_cursor_after_rect(full);
    }
}
