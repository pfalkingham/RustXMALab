//! The Rigid body dialog ("RB : <description>"), opened from the settings button of a
//! rigid-body row in the Points tree (`Action::RigidBodySettings`).
//!
//! Ported control by control from `RigidBodyDialog.ui` and ui/RigidBodyDialog.cpp
//! (`docs/cpp-editing.md` 5.2 and 5.3, E30 to E43 and "Export references"). As in XMALab 3,
//! every control except the correspondence combo boxes writes to the body at once and
//! Cancel does not take those writes back; here each write is its own undo step
//! (`EditRequest`), so Edit > Undo does. The combo boxes are applied with OK, and before
//! "Set Body from File" in the same undo step as the file's references.
//!
//! Differences from XMALab 3 (each in the parity ledger or the ui-status table):
//!
//! - Opening the dialog writes nothing: XMALab writes the override cutoff and the mesh scale
//!   rounded to two decimals as soon as it opens (D14). A value is written only when the
//!   user changes it.
//! - "Reset To Marker References" asks first when it would discard references set from a
//!   file or a frame, or leave the body without references (D21).
//! - Cancel in the colour dialog changes nothing (D22).
//! - A rejected reference file leaves the body as it was (D18, the engine's edit); a mesh
//!   that cannot be loaded leaves the body's mesh as it was (A74).
//! - Cancelling the driver file of a virtual point driven by a body adds nothing (XMALab
//!   adds the point with an unread file).

use std::path::{Path, PathBuf};

use egui::Ui;
use xma_engine::Edit;
use xma_engine::edit::VirtualPointInput;
use xma_engine::edit::bodies::CORRESPONDENCES_DUPLICATE;
use xma_model::{RigidBody, Trial};

use super::edit_forms::{FormOutcome, Prompt, PromptAnswer, PromptValue, double_spin_box};
use super::{ConfirmButtons, DialogEnv, DialogResult, form, last_dir};
use crate::editing::EditRequest;
use crate::icons::Icon;

/// The text of the confirmation before "Set Body from File" on a body with references.
pub const FROM_FILE_CONFIRM: &str =
    "References are already set by using a file. Are you sure you want to reset the references?";
/// "Reset To Marker References" when every marker has a reference point (D21, new).
pub const RESET_CONFIRM_MARKERS: &str = "The references of this rigid body will be replaced by the reference points of its markers. Are you sure you want to reset the references?";
/// "Reset To Marker References" when a marker has no reference point (D21, new).
pub const RESET_CONFIRM_NONE: &str = "Not every marker of this rigid body has a reference point, so the rigid body will have no references. Are you sure you want to reset the references?";
/// Step 3 of "Add virtual point".
pub const VP_ANIMATE: &str = "Do you want to animate the virtual point by using a Rigid Body? Click no if you want to import a csv of animated data instead";
/// Step 4 of "Add virtual point".
pub const VP_SECOND_FILE: &str = "Do you have different CT coordinates for the driver point, i.e. the point attached to the fully animated rigid body (such as from a scan of disarticulated bones or two scans)?";

const CSV: (&str, &[&str]) = ("CSV Files", &["csv"]);
const OBJ: (&str, &[&str]) = ("OBJ Files", &["obj"]);

/// What the open prompt is for: the step that follows its answer.
#[derive(Clone, Debug, PartialEq)]
enum Pending {
    /// An error message; nothing follows.
    Message,
    /// "Set Body from File" confirmed with this file and these correspondences.
    FromFile {
        path: PathBuf,
        markers: Vec<usize>,
    },
    FromFrame,
    Reset,
    Color,
    VpName,
    VpAnimate {
        name: String,
        reference: PathBuf,
    },
    VpChooseBody {
        name: String,
        reference: PathBuf,
    },
    VpSecondFile {
        name: String,
        reference: PathBuf,
        driver: usize,
    },
}

/// The dialog's own state. The body's values are read from the document every frame.
#[derive(Clone, Debug, PartialEq)]
pub struct RigidBodyDialog {
    pub trial: usize,
    pub body: usize,
    /// The trial's frame on screen when the dialog opened (0-based): the default of "Set
    /// Body from Frame" (the dialog is modal, so it cannot change while open).
    pub active_frame: usize,
    /// The markers the correspondence combo boxes list, in the body's order when opened.
    pub items: Vec<usize>,
    /// Per reference point, the index into `items` of the chosen marker.
    pub choice: Vec<usize>,
    cutoff_edit: Option<f64>,
    scale_edit: Option<f64>,
    prompt: Option<(Prompt, Pending)>,
    /// Controls used and not yet carried out (one is carried out per frame).
    queued: std::collections::VecDeque<Act>,
}

impl RigidBodyDialog {
    /// The dialog for body `body` of trial `trial` (`RigidBodyDialog::RigidBodyDialog`).
    #[must_use]
    pub fn new(doc: &xma_engine::Document, trial: usize, body: usize, active_frame: usize) -> Self {
        let items = doc
            .project()
            .trials
            .get(trial)
            .and_then(|t| t.rigid_bodies.get(body))
            .map(|b| b.marker_indices.clone())
            .unwrap_or_default();
        Self {
            trial,
            body,
            active_frame,
            choice: (0..items.len()).collect(),
            items,
            cutoff_edit: None,
            scale_edit: None,
            prompt: None,
            queued: std::collections::VecDeque::new(),
        }
    }

    /// The window title, "RB : <description>".
    #[must_use]
    pub fn title(rb: &RigidBody) -> String {
        format!("RB : {}", rb.description)
    }

    /// The markers the combo boxes choose, per reference point.
    #[must_use]
    pub fn chosen_markers(&self) -> Vec<usize> {
        self.choice
            .iter()
            .map(|&c| self.items.get(c).copied().unwrap_or(usize::MAX))
            .collect()
    }

    /// The open prompt's title and text (tests).
    #[must_use]
    pub fn prompt(&self) -> Option<&Prompt> {
        self.prompt.as_ref().map(|p| &p.0)
    }

    fn duplicates(&self) -> bool {
        let m = self.chosen_markers();
        m.iter().enumerate().any(|(i, a)| m[..i].contains(a))
    }

    fn apply(&self, edit: Edit) -> Option<DialogResult> {
        Some(DialogResult::Request(EditRequest::Apply(edit)))
    }

    /// Draws the dialog and its prompt; returns what to do.
    pub fn show(&mut self, ctx: &egui::Context, env: &mut DialogEnv<'_>) -> FormOutcome {
        let mut out = FormOutcome::default();
        let Some(doc) = env.doc else {
            out.close = true;
            return out;
        };
        let Some(trial) = doc.project().trials.get(self.trial) else {
            out.close = true;
            return out;
        };
        let Some(rb) = trial.rigid_bodies.get(self.body) else {
            out.close = true;
            return out;
        };
        let trial: &Trial = trial;
        let rb: &RigidBody = rb;

        let mut acts: Vec<Act> = Vec::new();
        let prompt_open = self.prompt.is_some();
        let modal = egui::Modal::new(egui::Id::unique("xma_rigid_body_dialog")).show(ctx, |ui| {
            if prompt_open {
                ui.disable();
            }
            acts = self.form(ui, env.icons, trial, rb);
        });
        // A value typed into "mesh scale" or "Cutoff frequency" is committed when the field
        // loses focus, which can be in the frame of a click on another control: the typed
        // values are carried out first (their field lost focus when the click began), then
        // the control clicked (the last one drawn, if several), one per frame, as each
        // answer is one edit.
        let typed = |a: &Act| matches!(a, Act::MeshScale(_) | Act::Cutoff(_));
        let clicked = acts.iter().rev().find(|a| !typed(a)).copied();
        self.queued
            .extend(acts.into_iter().filter(typed).chain(clicked));
        let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));

        // The prompt over the form.
        if let Some((prompt, _)) = &mut self.prompt
            && let Some(answer) = prompt.show(ctx, env.icons)
        {
            let (_, pending) = self.prompt.take().expect("a prompt is open");
            out.result = self.answered(pending, answer, env, trial, rb);
            return out;
        }
        if !prompt_open && self.queued.is_empty() && modal.should_close() && escape {
            out.close = true;
            return out;
        }
        if self.prompt.is_none()
            && let Some(a) = self.queued.pop_front()
        {
            self.act(a, env, trial, rb, &mut out);
            if !self.queued.is_empty() && !out.close {
                ctx.request_repaint();
            }
        }
        out
    }

    /// The form, in the `.ui` file's grid order; returns the controls used, in drawing
    /// order (more than one when a typed value is committed in the frame of a click).
    fn form(
        &mut self,
        ui: &mut Ui,
        icons: &crate::icons::Icons,
        trial: &Trial,
        rb: &RigidBody,
    ) -> Vec<Act> {
        let mut acts = Vec::new();
        let linked = trial.is_copy_from_default;
        let default = trial.is_default;
        // A fixed width: the group boxes fill what they are given, so an open width grows.
        ui.set_width(880.0);
        ui.horizontal(|ui| {
            ui.add(icons.image(ui.ctx(), window_icon(rb), 16.0));
            if form::dialog_title(ui, &Self::title(rb)) {
                acts.push(Act::Cancel);
            }
        });

        // groupBox "Correspondances": one row per reference point.
        ui.add_enabled_ui(!linked, |ui| {
            form::group_box(ui, "Correspondances", |ui| {
                egui::Grid::new("rb_correspondences")
                    .num_columns(3)
                    .show(ui, |ui| {
                        let texts: Vec<String> =
                            self.items.iter().map(|m| (m + 1).to_string()).collect();
                        for j in 0..self.choice.len() {
                            let name = rb.reference_names.get(j).cloned().unwrap_or_default();
                            if name.is_empty() {
                                ui.label("");
                            } else {
                                ui.label(name);
                            }
                            let sel = self.choice[j];
                            let shown = texts.get(sel).cloned().unwrap_or_default();
                            let r = egui::ComboBox::from_id_salt(("rb_corr", j))
                                .selected_text(shown)
                                .width(60.0)
                                .show_ui(ui, |ui| {
                                    for (i, t) in texts.iter().enumerate() {
                                        if ui.selectable_label(i == sel, t).clicked() {
                                            self.choice[j] = i;
                                        }
                                    }
                                });
                            if let Some(i) = crate::qt::combo_wheel(
                                ui,
                                &r.response,
                                Some(self.choice[j]),
                                texts.len(),
                                |_| true,
                            ) {
                                self.choice[j] = i;
                            }
                            let _ = r
                                .response
                                .accessible_name(format!("Reference point {} marker", j + 1));
                            let desc = self
                                .items
                                .get(self.choice[j])
                                .and_then(|m| trial.markers.get(*m))
                                .map(|m| m.description.clone())
                                .unwrap_or_default();
                            if desc.is_empty() {
                                ui.label("");
                            } else {
                                ui.label(desc);
                            }
                            ui.end_row();
                        }
                    });
            });
        });

        // groupBox_2 "Visualization".
        let has_mesh = rb.mesh_model.is_some() && rb.mesh_readable != Some(false);
        ui.add_enabled_ui(!linked, |ui| {
            form::group_box(ui, "Visualization", |ui| {
                egui::Grid::new("rb_visualization")
                    .num_columns(4)
                    .show(ui, |ui| {
                        if let Some(v) = form::check_box(ui, true, rb.visible, "Draw Rigid body") {
                            acts.push(Act::Visible(v));
                        }
                        ui.label("");
                        ui.label("");
                        let c = egui::Color32::from_rgb(rb.color.r, rb.color.g, rb.color.b);
                        if form::swatch_button(ui, c, "Rigid body colour").clicked() {
                            acts.push(Act::Color);
                        }
                        ui.end_row();
                        if let Some(v) = form::check_box(
                            ui,
                            has_mesh,
                            rb.draw_mesh_model,
                            "Draw bone mesh model",
                        ) {
                            acts.push(Act::DrawMesh(v));
                        }
                        let mesh_name = rb
                            .mesh_model
                            .as_deref()
                            .filter(|_| has_mesh)
                            .and_then(|m| Path::new(m).file_name())
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        if mesh_name.is_empty() {
                            ui.label("");
                        } else {
                            ui.label(mesh_name);
                        }
                        ui.label("");
                        if ui.button("load Obj-file").clicked() {
                            acts.push(Act::Mesh);
                        }
                        ui.end_row();
                        ui.label("");
                        ui.label("");
                        ui.label("mesh scale ");
                        if let Some(v) = double_spin_box(
                            ui,
                            has_mesh,
                            rb.mesh_scale,
                            &mut self.scale_edit,
                            0.0,
                            10000.0,
                            70.0,
                            "mesh scale",
                        ) {
                            acts.push(Act::MeshScale(v));
                        }
                        ui.end_row();
                    });
            });
        });

        // groupBox_3 "Filter".
        ui.add_enabled_ui(!linked, |ui| {
            form::group_box(ui, "Filter", |ui| {
                ui.horizontal(|ui| {
                    let on = rb.override_cutoff_frequency;
                    if let Some(v) =
                        form::check_box(ui, true, on, "Override Trial Cutoff Frequency")
                    {
                        acts.push(Act::Override(v));
                    }
                    let shown = if on {
                        rb.cutoff_frequency
                    } else {
                        trial.cutoff_frequency
                    };
                    if let Some(v) = double_spin_box(
                        ui,
                        on,
                        shown,
                        &mut self.cutoff_edit,
                        0.0,
                        10000.0,
                        70.0,
                        "Cutoff frequency",
                    ) {
                        acts.push(Act::Cutoff(v));
                    }
                });
            });
        });

        // groupBox_4 "Virtual Points".
        ui.add_enabled_ui(!linked, |ui| {
            form::group_box(ui, "Virtual Points", |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Add virtual point").clicked() {
                        acts.push(Act::AddVirtual);
                    }
                    if ui.button("Delete all virtual points").clicked() {
                        acts.push(Act::DeleteVirtual);
                    }
                });
                for vp in &rb.virtual_points {
                    ui.label(vp.name.as_str());
                }
            });
        });

        // Row 4: the reference buttons, then Cancel and OK.
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!linked, egui::Button::new("Set Body from File"))
                .clicked()
            {
                acts.push(Act::FromFile);
            }
            if ui
                .add_enabled(
                    !linked && !default,
                    egui::Button::new("Set Body from Frame"),
                )
                .clicked()
            {
                acts.push(Act::FromFrame);
            }
            if ui
                .add_enabled(!linked, egui::Button::new("Reset To Marker References"))
                .clicked()
            {
                acts.push(Act::Reset);
            }
            if let Some(v) = form::check_box(
                ui,
                !linked && !default,
                rb.has_optimized_coordinates,
                "optimized from tracked",
            ) {
                acts.push(Act::Optimized(v));
            }
            if ui
                .add_enabled(
                    !linked && !default && rb.has_optimized_coordinates,
                    egui::Button::new("Update Optimzed"),
                )
                .clicked()
            {
                acts.push(Act::Update);
            }
            if ui.button("Export references").clicked() {
                acts.push(Act::Export);
            }
            // horizontalSpacer, then Cancel and OK.
            ui.add_space(24.0);
            if ui.button("Cancel").clicked() {
                acts.push(Act::Cancel);
            }
            if ui.button("OK").clicked() {
                acts.push(Act::Ok);
            }
        });
        acts
    }

    /// Carries out a control of the form.
    fn act(
        &mut self,
        a: Act,
        env: &mut DialogEnv<'_>,
        trial: &Trial,
        rb: &RigidBody,
        out: &mut FormOutcome,
    ) {
        let (t, b) = (self.trial, self.body);
        let start = last_dir(env.settings);
        match a {
            Act::Ok => {
                // RigidBodyDialog::isComplete: duplicates keep the dialog open.
                if self.duplicates() {
                    self.message(CORRESPONDENCES_DUPLICATE);
                    return;
                }
                let markers = self.chosen_markers();
                if markers != rb.marker_indices {
                    out.result = self.apply(Edit::SetBodyCorrespondences {
                        trial: t,
                        body: b,
                        markers,
                    });
                }
                out.close = true;
            }
            Act::Cancel => out.close = true,
            Act::Visible(v) => {
                out.result = self.apply(Edit::SetBodyVisible {
                    trial: t,
                    body: b,
                    visible: v,
                });
            }
            Act::Color => self.prompt = Some((Prompt::color(rb.color), Pending::Color)),
            Act::Mesh => {
                if let Some(p) = env
                    .files
                    .open_file("Open meshfile file", start.as_deref(), OBJ)
                {
                    out.result = self.apply(Edit::SetBodyMesh {
                        trial: t,
                        body: b,
                        path: Some(p),
                    });
                }
            }
            Act::DrawMesh(v) => {
                out.result = self.apply(Edit::SetBodyDrawMesh {
                    trial: t,
                    body: b,
                    draw: v,
                });
            }
            Act::MeshScale(v) => {
                out.result = self.apply(Edit::SetBodyMeshScale {
                    trial: t,
                    body: b,
                    scale: v,
                });
            }
            Act::Override(on) => {
                // The stored cutoff, not the spin box's rounded value (D14 fixed).
                out.result = self.apply(Edit::SetBodyCutoffOverride {
                    trial: t,
                    body: b,
                    enabled: on,
                    cutoff_frequency: rb.cutoff_frequency,
                });
            }
            Act::Cutoff(v) => {
                out.result = self.apply(Edit::SetBodyCutoffOverride {
                    trial: t,
                    body: b,
                    enabled: true,
                    cutoff_frequency: v,
                });
            }
            Act::AddVirtual => {
                self.prompt = Some((
                    Prompt::Text {
                        title: "QInputDialog::getText()".into(),
                        label: "Enter a name for the virtual point:".into(),
                        value: format!("Virtual {}", rb.virtual_points.len() + 1),
                    },
                    Pending::VpName,
                ));
            }
            Act::DeleteVirtual => {
                out.result = self.apply(Edit::ClearVirtualPoints { trial: t, body: b });
            }
            Act::FromFile => {
                let Some(path) =
                    env.files
                        .open_file("Open rigid body references", start.as_deref(), CSV)
                else {
                    return;
                };
                let markers = self.chosen_markers();
                if rb.references_set != 0 {
                    self.prompt = Some((
                        Prompt::Confirm {
                            message: FROM_FILE_CONFIRM.into(),
                            buttons: ConfirmButtons::OkCancel,
                        },
                        Pending::FromFile { path, markers },
                    ));
                } else {
                    out.result = Some(self.file_references(path, markers, rb));
                }
            }
            Act::FromFrame => {
                let lo = trial.start_frame.max(1) as i32;
                let hi = trial.end_frame.max(trial.start_frame.max(1)) as i32;
                self.prompt = Some((
                    Prompt::Int {
                        title: "Frame Number".into(),
                        label: String::new(),
                        value: (self.active_frame as i32 + 1).clamp(lo, hi),
                        min: lo,
                        max: hi,
                    },
                    Pending::FromFrame,
                ));
            }
            Act::Reset => {
                let all_marker_refs = rb.marker_indices.iter().all(|&m| {
                    trial
                        .markers
                        .get(m)
                        .is_some_and(|mk| mk.reference_3d.is_some())
                });
                let unchanged =
                    rb.references_set == 0 || (rb.references_set == 1 && all_marker_refs);
                if unchanged {
                    out.result = self.apply(Edit::ResetBodyReferences { trial: t, body: b });
                } else {
                    let message = if all_marker_refs {
                        RESET_CONFIRM_MARKERS
                    } else {
                        RESET_CONFIRM_NONE
                    };
                    self.prompt = Some((
                        Prompt::Confirm {
                            message: message.into(),
                            buttons: ConfirmButtons::OkCancel,
                        },
                        Pending::Reset,
                    ));
                }
            }
            Act::Optimized(v) => {
                out.result = self.apply(Edit::SetBodyOptimized {
                    trial: t,
                    body: b,
                    optimized: v,
                });
            }
            Act::Update => {
                out.result = self.apply(Edit::UpdateBodyOptimized { trial: t, body: b });
            }
            Act::Export => {
                // RigidBodyDialog::on_pushButtonExport_clicked: start at the last folder
                // with the body's description as the name.
                let start = PathBuf::from(&env.settings.last_directory_used).join(&rb.description);
                if let Some(p) = env.files.save_file_filtered(
                    "Rigid Body references as",
                    Some(&start),
                    ("CSV", &["csv"]),
                ) {
                    env.settings.set_last_used_directory_from_file(&p);
                    out.settings_changed = true;
                    let text = xma_engine::export::rigid_body_references(
                        &rb.reference_names,
                        &rb.reference_points,
                    );
                    if let Err(e) = std::fs::write(&p, text) {
                        self.message(&format!("Could not write {}: {e}", p.display()));
                    }
                }
            }
        }
    }

    /// "Set Body from File": the correspondences and the file's references, one undo step.
    fn file_references(&self, path: PathBuf, markers: Vec<usize>, rb: &RigidBody) -> DialogResult {
        let (trial, body) = (self.trial, self.body);
        let mut edits = Vec::new();
        if markers != rb.marker_indices {
            edits.push(Edit::SetBodyCorrespondences {
                trial,
                body,
                markers,
            });
        }
        edits.push(Edit::SetBodyReferencesFromFile { trial, body, path });
        DialogResult::Request(EditRequest::ApplyAll {
            label: "Set Body from File".into(),
            edits,
        })
    }

    fn message(&mut self, text: &str) {
        self.prompt = Some((
            Prompt::Error {
                message: text.to_owned(),
            },
            Pending::Message,
        ));
    }

    /// The step after a prompt's answer.
    fn answered(
        &mut self,
        pending: Pending,
        answer: PromptAnswer,
        env: &mut DialogEnv<'_>,
        trial: &Trial,
        rb: &RigidBody,
    ) -> Option<DialogResult> {
        let (t, b) = (self.trial, self.body);
        let accepted = match &answer {
            PromptAnswer::Accepted(v) => Some(v.clone()),
            PromptAnswer::Rejected => None,
        };
        let start = last_dir(env.settings);
        match (pending, accepted) {
            (Pending::Message, _) => None,
            (Pending::FromFile { path, markers }, Some(_)) => {
                Some(self.file_references(path, markers, rb))
            }
            (Pending::FromFrame, Some(PromptValue::Int(f))) => {
                self.apply(Edit::SetBodyReferencesFromFrame {
                    trial: t,
                    body: b,
                    frame: (f.max(1) - 1) as usize,
                })
            }
            (Pending::Reset, Some(_)) => {
                self.apply(Edit::ResetBodyReferences { trial: t, body: b })
            }
            // D22 fixed: only OK applies the colour.
            (Pending::Color, Some(PromptValue::Color(c))) => (c != rb.color)
                .then(|| {
                    self.apply(Edit::SetBodyColor {
                        trial: t,
                        body: b,
                        color: c,
                    })
                })
                .flatten(),
            (Pending::VpName, Some(PromptValue::Text(name))) if !name.is_empty() => {
                let title = format!(
                    "Open CT coordinate file. (Point in the reference of RB{})",
                    b + 1
                );
                let reference = env.files.open_file(&title, start.as_deref(), CSV)?;
                remember(env, &reference);
                self.prompt = Some((
                    Prompt::Confirm {
                        message: VP_ANIMATE.into(),
                        buttons: ConfirmButtons::YesNo,
                    },
                    Pending::VpAnimate { name, reference },
                ));
                None
            }
            (Pending::VpAnimate { name, reference }, Some(_)) => {
                let items = trial
                    .rigid_bodies
                    .iter()
                    .enumerate()
                    .map(|(i, r)| format!("RB{} : {}", i + 1, r.description))
                    .collect();
                self.prompt = Some((
                    Prompt::Item {
                        title: "Choose Rigid Body".into(),
                        label: "RB:".into(),
                        items,
                        selected: 0,
                    },
                    Pending::VpChooseBody { name, reference },
                ));
                None
            }
            (Pending::VpChooseBody { name, reference }, Some(PromptValue::Item(i))) => {
                self.prompt = Some((
                    Prompt::Confirm {
                        message: VP_SECOND_FILE.into(),
                        buttons: ConfirmButtons::YesNo,
                    },
                    Pending::VpSecondFile {
                        name: format!("{name} RB{}", i + 1),
                        reference,
                        driver: i,
                    },
                ));
                None
            }
            (
                Pending::VpSecondFile {
                    name,
                    reference,
                    driver,
                },
                yes,
            ) => {
                let driver_file = if yes.is_some() {
                    let title = format!(
                        "Open CT coordinate file. (Point in the reference of RB{})",
                        driver + 1
                    );
                    // Cancel adds nothing (XMALab adds the point with no file read).
                    let p = env.files.open_file(&title, start.as_deref(), CSV)?;
                    remember(env, &p);
                    Some(p)
                } else {
                    None
                };
                self.apply(Edit::AddVirtualPoint {
                    trial: t,
                    body: b,
                    name,
                    reference_file: reference,
                    input: VirtualPointInput::Body {
                        body: driver,
                        driver_file,
                    },
                })
            }
            // "No" to animating by a body, or no body chosen: a file of tracked coordinates.
            (
                Pending::VpAnimate { name, reference } | Pending::VpChooseBody { name, reference },
                None,
            ) => {
                let file =
                    env.files
                        .open_file("Open tracked virtual data file", start.as_deref(), CSV)?;
                remember(env, &file);
                self.apply(Edit::AddVirtualPoint {
                    trial: t,
                    body: b,
                    name,
                    reference_file: reference,
                    input: VirtualPointInput::Coordinates { file },
                })
            }
            _ => None,
        }
    }
}

/// A control of the form that was used.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Act {
    Ok,
    Cancel,
    Visible(bool),
    Color,
    Mesh,
    DrawMesh(bool),
    MeshScale(f64),
    Override(bool),
    Cutoff(f64),
    AddVirtual,
    DeleteVirtual,
    FromFile,
    FromFrame,
    Reset,
    Optimized(bool),
    Update,
    Export,
}

/// The window icon (`RigidBodyDialog::updateIcon`): how the references were set, and
/// whether they are optimised.
fn window_icon(rb: &RigidBody) -> Icon {
    match (rb.references_set, rb.has_optimized_coordinates) {
        (2, true) => Icon::Shape3dOptimized,
        (2, false) => Icon::Shape3d,
        (1, true) => Icon::Shape3dSetMarkerOptimized,
        (1, false) => Icon::Shape3dSetMarker,
        _ => Icon::Shape3dNotSet,
    }
}

/// `Settings::setLastUsedDirectory(file)`.
fn remember(env: &mut DialogEnv<'_>, file: &Path) {
    env.settings.set_last_used_directory_from_file(file);
}
