//! `BlenderSceneDialog`: File > Export > Blender scene (new in RustXMALab; parity ledger
//! U176 for the entry and the dialog, F85 for the export). XMALab 3 has no such entry and no
//! such form; the controls are listed in `ui-status/BlenderSceneDialog.md`.
//!
//! The dialog ("Export Blender scene"), top to bottom:
//!
//! - **Scene**: the radio button "Cameras only" with the drop-down "Calibration image" (the
//!   images calibrated in every calibrated camera, "<number>: <file name>"; the first is
//!   chosen) and the check box "Undistorted calibration image on each camera's plane"
//!   (ticked at first; unticked, the planes are empty); the
//!   radio button "Trial" with a drop-down beside it, named by it (the project's trials,
//!   the active one chosen), the check box "Undistorted trial images" (ticked) with "from frame" and "to
//!   frame" (1-based, the whole trial at first, kept inside the trial), and the check boxes
//!   "Rigid bodies", under it "Filtered transformations and 3D points" (the engine's
//!   `filtered`, which filters both: enabled when bodies or points are ticked and the engine
//!   would not refuse it, [`filtered_refusal`], whose sentence is its hover text when
//!   disabled), "Bone meshes" (ticking it ticks "Rigid bodies") and "3D points". "Trial" is
//!   disabled, with a sentence, when the project has no
//!   trial whose reference calibration image is calibrated. In Marker tracking with a trial
//!   open "Trial" is chosen at first, otherwise "Cameras only".
//! - **Blender**: a line saying where Blender was found or that it was not
//!   ([`xma_engine::blender_scene::locate_blender`], searched once when the dialog opens; the
//!   search only looks at a few files), the check box "Create the .blend file now" (ticked
//!   when Blender was found, disabled otherwise with [`NOT_FOUND`]) and under it "Pack still
//!   images into the .blend".
//! - A line of small text saying what will be written, its folder named by the engine
//!   ([`folder_name`]: what the export writes; a name it refuses shows its sentence and OK
//!   is disabled), then OK and Cancel. OK asks for the folder (`app/blender.rs`), then the
//!   export runs as a job.
//!
//! The dialog's state becomes the engine's [`BlenderSceneSpec`] in one place,
//! [`BlenderSceneDialog::spec`].

use std::path::PathBuf;

use xma_engine::Document;
use xma_engine::blender_scene::{
    BlendOptions, BlenderSceneKind, BlenderSceneSpec, filtered_refusal, folder_name, project_stem,
};

use super::edit_forms::FormOutcome;
use super::{DialogEnv, DialogResult, form};

/// The window title.
pub const TITLE: &str = "Export Blender scene";
/// The folder question after OK.
pub const FOLDER_TITLE: &str = "Choose the folder to write the Blender scene into";
/// The sentence under "Create the .blend file now" when Blender was not found.
pub const NOT_FOUND: &str = "Blender was not found. The folder is still written; run build_scene.py in Blender to make the scene. Set XMALAB_BLENDER to Blender's program to have it found.";
/// The hover text of "Pack still images into the .blend".
pub const PACK_HOVER: &str = "Put the calibration images inside the .blend, so that the .blend can be moved on its own. Blender cannot pack an image sequence: a trial's images stay in the folder beside the .blend.";
/// The sentence shown without a calibrated camera.
pub const NO_CAMERA: &str =
    "No camera is calibrated: a Blender scene needs calibrated cameras. Calibrate first.";
/// The sentence shown when no image is calibrated in every camera.
pub const NO_COMMON_IMAGE: &str = "No calibration image is calibrated in every camera.";
/// The label of the check box that filters the bodies and the points.
pub const FILTERED: &str = "Filtered transformations and 3D points";
/// Its hover text.
pub const FILTERED_HOVER: &str = "The rigid bodies' transformations and the markers' 3D points filtered at the trial's cutoff frequency, as the filtered variants of File > Export > Rigid body transformations and 3D points.";
/// The sentence beside a disabled "Trial".
pub const NO_TRIAL: &str = "The project has no trial.";
/// The sentence when the chosen trial's reference calibration image is calibrated in no
/// camera.
pub const NO_REFERENCE: &str =
    "The trial's reference calibration image is calibrated in no camera.";

/// One trial as the dialog offers it.
#[derive(Clone, Debug, PartialEq)]
pub struct TrialChoice {
    pub name: String,
    /// Number of frames (the frame fields' upper limit).
    pub frames: usize,
    /// The Default trial: it has no images.
    pub is_default: bool,
    /// Its reference calibration image is calibrated in some camera.
    pub has_reference: bool,
    /// Why the engine refuses filtered bodies (`None`: it does not), and filtered points
    /// ([`filtered_refusal`]).
    pub filter_bodies_refused: Option<String>,
    pub filter_points_refused: Option<String>,
}

/// The dialog's state.
#[derive(Clone, Debug, PartialEq)]
pub struct BlenderSceneDialog {
    /// Some camera is calibrated.
    pub calibrated: bool,
    /// The project file's stem (the folder's name begins with it).
    pub stem: String,
    /// "Trial" is chosen (else "Cameras only").
    pub trial_kind: bool,
    /// The calibration images calibrated in every calibrated camera: (0-based index, the
    /// drop-down's text).
    pub calibration_images: Vec<(usize, String)>,
    /// The entry of `calibration_images` chosen.
    pub calibration_choice: usize,
    /// "Undistorted calibration image on each camera's plane".
    pub calibration_image_on_plane: bool,
    pub trials: Vec<TrialChoice>,
    /// The trial chosen (an index of `trials`).
    pub trial: usize,
    /// "Undistorted trial images".
    pub trial_images: bool,
    /// "from frame", "to frame", 1-based.
    pub from: i64,
    pub to: i64,
    pub rigid_bodies: bool,
    pub filtered: bool,
    pub meshes: bool,
    pub points: bool,
    /// Where Blender is, or why it was not found.
    pub blender: Result<PathBuf, String>,
    /// "Create the .blend file now".
    pub create_blend: bool,
    /// "Pack still images into the .blend".
    pub pack: bool,
}

/// The drop-down text of a calibration image: "<number>: <file name>".
fn image_text(index: usize, filename: &str) -> String {
    let name = std::path::Path::new(filename)
        .file_name()
        .map_or_else(|| filename.to_owned(), |n| n.to_string_lossy().into_owned());
    if name.is_empty() {
        format!("{}", index + 1)
    } else {
        format!("{}: {name}", index + 1)
    }
}

impl BlenderSceneDialog {
    /// The dialog for `doc`: `tracking` is true in Marker tracking, `active_trial` the trial
    /// open there, `blender` the result of the search for Blender.
    #[must_use]
    pub fn new(
        doc: &Document,
        tracking: bool,
        active_trial: Option<usize>,
        blender: Result<PathBuf, String>,
    ) -> Self {
        let p = doc.project();
        let calibrated_cams: Vec<_> = p.cameras.iter().filter(|c| c.calibrated).collect();
        let n_images = calibrated_cams
            .iter()
            .map(|c| c.calibration_images.len())
            .max()
            .unwrap_or(0);
        let calibration_images: Vec<(usize, String)> = (0..n_images)
            .filter(|&i| {
                !calibrated_cams.is_empty()
                    && calibrated_cams.iter().all(|c| {
                        c.calibration_images
                            .get(i)
                            .is_some_and(xma_model::CalibrationImage::is_calibrated)
                    })
            })
            .map(|i| {
                let name = calibrated_cams
                    .first()
                    .and_then(|c| c.calibration_images.get(i))
                    .map_or("", |ci| ci.filename.as_str());
                (i, image_text(i, name))
            })
            .collect();
        let trials: Vec<TrialChoice> = p
            .trials
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let r = t.reference_calibration_image;
                TrialChoice {
                    name: t.name.clone(),
                    frames: t.n_frames,
                    is_default: t.is_default,
                    has_reference: calibrated_cams.iter().any(|c| {
                        c.calibration_images
                            .get(r)
                            .is_some_and(xma_model::CalibrationImage::is_calibrated)
                    }),
                    filter_bodies_refused: filtered_refusal(doc, i, true, false),
                    filter_points_refused: filtered_refusal(doc, i, false, true),
                }
            })
            .collect();
        let trial = active_trial
            .filter(|&t| t < trials.len())
            .or_else(|| trials.iter().position(|t| t.has_reference))
            .unwrap_or(0);
        let frames = trials.get(trial).map_or(1, |t| t.frames.max(1)) as i64;
        let found = blender.is_ok();
        let mut d = BlenderSceneDialog {
            calibrated: !calibrated_cams.is_empty(),
            stem: project_stem(doc),
            trial_kind: false,
            calibration_images,
            calibration_choice: 0,
            calibration_image_on_plane: true,
            trials,
            trial,
            trial_images: true,
            from: 1,
            to: frames,
            rigid_bodies: false,
            filtered: false,
            meshes: false,
            points: false,
            blender,
            create_blend: found,
            pack: false,
        };
        d.trial_kind = tracking && active_trial.is_some() && d.trial_radio_enabled();
        d
    }

    /// "Trial" can be chosen: some trial's reference calibration image is calibrated.
    #[must_use]
    pub fn trial_radio_enabled(&self) -> bool {
        self.calibrated && self.trials.iter().any(|t| t.has_reference)
    }

    /// The trial chosen, if there is one.
    #[must_use]
    pub fn chosen_trial(&self) -> Option<&TrialChoice> {
        self.trials.get(self.trial)
    }

    /// The frame fields' upper limit: the chosen trial's frames.
    #[must_use]
    pub fn max_frame(&self) -> i64 {
        self.chosen_trial().map_or(1, |t| t.frames.max(1)) as i64
    }

    /// "Undistorted trial images" can be ticked: not the Default trial, which has no images.
    #[must_use]
    pub fn images_enabled(&self) -> bool {
        self.chosen_trial()
            .is_some_and(|t| !t.is_default && t.frames > 0)
    }

    /// Why "Filtered transformations and 3D points" cannot be ticked: the engine's sentence
    /// for the bodies or the points ticked, `None` when it can be ticked (or nothing it filters
    /// is ticked).
    #[must_use]
    pub fn filtered_refused(&self) -> Option<&str> {
        let t = self.chosen_trial()?;
        let bodies = self
            .rigid_bodies
            .then_some(t.filter_bodies_refused.as_deref())
            .flatten();
        let points = self
            .points
            .then_some(t.filter_points_refused.as_deref())
            .flatten();
        bodies.or(points)
    }

    /// "Filtered transformations and 3D points" can be ticked: bodies or points are ticked
    /// and the engine would filter them.
    #[must_use]
    pub fn filtered_enabled(&self) -> bool {
        (self.rigid_bodies || self.points)
            && self.chosen_trial().is_some()
            && self.filtered_refused().is_none()
    }

    /// Why OK is disabled, if it is.
    #[must_use]
    pub fn refusal(&self) -> Option<String> {
        if !self.calibrated {
            return Some(NO_CAMERA.into());
        }
        let r = if self.trial_kind {
            match self.chosen_trial() {
                None => Some(NO_TRIAL),
                Some(t) if !t.has_reference => Some(NO_REFERENCE),
                Some(_) => None,
            }
        } else if self.calibration_images.is_empty() {
            Some(NO_COMMON_IMAGE)
        } else {
            None
        };
        match (r, self.folder_name()) {
            (Some(r), _) => Some(r.into()),
            (None, Err(e)) => Some(format!("The folder's name: {e}.")),
            (None, Ok(_)) => None,
        }
    }

    /// Blender will be run.
    #[must_use]
    pub fn blend_now(&self) -> bool {
        self.create_blend && self.blender.is_ok()
    }

    /// The folder's name, as the engine names it ([`folder_name`]), or the engine's sentence
    /// refusing it.
    pub fn folder_name(&self) -> Result<String, String> {
        let trial = match (self.trial_kind, self.chosen_trial()) {
            (true, Some(t)) => Some(t.name.as_str()),
            _ => None,
        };
        folder_name(&self.stem, trial, None)
    }

    /// The scene's planes show images.
    #[must_use]
    pub fn writes_images(&self) -> bool {
        if self.trial_kind {
            self.trial_images && self.images_enabled()
        } else {
            self.calibration_image_on_plane
        }
    }

    /// The small text saying what will be written (empty when the folder's name is refused:
    /// the refusal says why).
    #[must_use]
    pub fn written_text(&self) -> String {
        let Ok(name) = self.folder_name() else {
            return String::new();
        };
        let mut s = format!(
            "A folder {name} with build_scene.py, scene.json{}",
            if self.writes_images() {
                " and the images"
            } else {
                " (no images)"
            }
        );
        if self.blend_now() {
            s.push_str(&format!(" and {name}.blend"));
        }
        s
    }

    /// Ticks or unticks "Rigid bodies" (unticking it unticks "Bone meshes").
    pub fn set_rigid_bodies(&mut self, v: bool) {
        self.rigid_bodies = v;
        if !v {
            self.meshes = false;
        }
    }

    /// Ticks or unticks "Bone meshes" (ticking it ticks "Rigid bodies").
    pub fn set_meshes(&mut self, v: bool) {
        self.meshes = v;
        if v {
            self.rigid_bodies = true;
        }
    }

    /// Chooses trial `t`: the frame fields become its whole range.
    pub fn set_trial(&mut self, t: usize) {
        if t < self.trials.len() {
            self.trial = t;
            self.from = 1;
            self.to = self.max_frame();
        }
    }

    /// Sets "from frame" (1-based), kept inside the trial; "to frame" follows when it would
    /// be before it.
    pub fn set_from(&mut self, v: i64) {
        self.from = v.clamp(1, self.max_frame());
        if self.to < self.from {
            self.to = self.from;
        }
    }

    /// Sets "to frame" (1-based), kept inside the trial; "from frame" follows when it would
    /// be after it.
    pub fn set_to(&mut self, v: i64) {
        self.to = v.clamp(1, self.max_frame());
        if self.from > self.to {
            self.from = self.to;
        }
    }

    /// What the engine exports for the dialog as it stands.
    #[must_use]
    pub fn spec(&self) -> BlenderSceneSpec {
        let kind = if self.trial_kind {
            let frames = (self.trial_images && self.images_enabled()).then(|| {
                let max = self.max_frame();
                let a = self.from.clamp(1, max);
                let b = self.to.clamp(a, max);
                ((a - 1) as usize, (b - 1) as usize)
            });
            BlenderSceneKind::Trial {
                trial: self.trial,
                frames,
            }
        } else {
            BlenderSceneKind::Cameras {
                calibration_image: self
                    .calibration_images
                    .get(self.calibration_choice)
                    .map(|(i, _)| *i),
                images: self.calibration_image_on_plane,
            }
        };
        let mut spec = BlenderSceneSpec::new(kind);
        if self.trial_kind {
            spec.rigid_bodies = self.rigid_bodies;
            spec.filtered = self.filtered && self.filtered_enabled();
            spec.meshes = self.rigid_bodies && self.meshes;
            spec.points = self.points;
        }
        spec.blend = match &self.blender {
            Ok(exe) if self.create_blend => Some(BlendOptions {
                blender: Some(exe.clone()),
                pack: self.pack,
            }),
            _ => None,
        };
        spec
    }

    /// OK's answer.
    #[must_use]
    pub fn result(&self) -> DialogResult {
        DialogResult::BlenderScene(Box::new(self.spec()))
    }

    /// Draws the dialog.
    pub fn show(&mut self, ctx: &egui::Context, _env: &mut DialogEnv<'_>) -> FormOutcome {
        let mut out = FormOutcome::default();
        let modal = form::modal(ctx, egui::Id::unique("xma_blender_scene_dialog"), |ui| {
            ui.set_width(480.0);
            if form::dialog_title(ui, TITLE) {
                out.close = true;
            }
            self.scene_group(ui);
            ui.add_space(4.0);
            self.blender_group(ui);
            ui.add_space(4.0);
            if let Some(r) = self.refusal() {
                ui.add(egui::Label::new(r).wrap());
            }
            let written = self.written_text();
            if !written.is_empty() {
                ui.add(egui::Label::new(egui::RichText::new(written).small()).wrap());
            }
            ui.add_space(4.0);
            let ok_enabled = self.refusal().is_none();
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if form::button(ui, true, "Cancel", 80.0).clicked() {
                    out.close = true;
                }
                if form::button(ui, ok_enabled, "OK", 80.0).clicked() {
                    out.result = Some(self.result());
                    out.close = true;
                }
            });
        });
        if modal.escaped() {
            out.close = true;
        }
        out
    }

    fn scene_group(&mut self, ui: &mut egui::Ui) {
        let indent = 22.0;
        form::group_box(ui, "Scene", |ui| {
            form::styled_frame(ui, |ui| {
                if form::radio(ui, self.calibrated, !self.trial_kind, "Cameras only") {
                    self.trial_kind = false;
                }
                let cams = self.calibrated && !self.trial_kind;
                ui.horizontal(|ui| {
                    ui.add_space(indent);
                    let l = ui
                        .add_enabled(cams, egui::Label::new("Calibration image"))
                        .id;
                    let items: Vec<&str> = self
                        .calibration_images
                        .iter()
                        .map(|(_, s)| s.as_str())
                        .collect();
                    let chosen = ui
                        .add_enabled_ui(cams && !items.is_empty(), |ui| {
                            crate::qt::combo_box(
                                ui,
                                "blender_calibration_image",
                                (!items.is_empty()).then_some(self.calibration_choice),
                                &items,
                                None,
                                Some(l),
                            )
                        })
                        .inner;
                    if let Some(i) = chosen {
                        self.calibration_choice = i;
                    }
                });
                ui.horizontal(|ui| {
                    ui.add_space(indent);
                    if let Some(v) = form::check_box(
                        ui,
                        cams,
                        self.calibration_image_on_plane,
                        "Undistorted calibration image on each camera's plane",
                    ) {
                        self.calibration_image_on_plane = v;
                    }
                });
                ui.add_space(4.0);
                let trial_ok = self.trial_radio_enabled();
                ui.horizontal(|ui| {
                    let radio =
                        ui.add_enabled(trial_ok, egui::RadioButton::new(self.trial_kind, "Trial"));
                    if radio.clicked() {
                        self.trial_kind = true;
                    }
                    // The radio button names the drop-down beside it.
                    let l = radio.id;
                    let tk = self.trial_kind;
                    let names: Vec<&str> = self.trials.iter().map(|t| t.name.as_str()).collect();
                    let chosen = ui
                        .add_enabled_ui(tk && !names.is_empty(), |ui| {
                            crate::qt::combo_box(
                                ui,
                                "blender_trial",
                                (!names.is_empty()).then_some(self.trial),
                                &names,
                                None,
                                Some(l),
                            )
                        })
                        .inner;
                    if let Some(t) = chosen {
                        self.set_trial(t);
                    }
                });
                if !trial_ok {
                    ui.horizontal(|ui| {
                        ui.add_space(indent);
                        let why = if self.trials.is_empty() {
                            NO_TRIAL
                        } else {
                            "No trial has a calibrated reference calibration image."
                        };
                        ui.add(egui::Label::new(why).wrap());
                    });
                }
                let tk = self.trial_kind;
                let images_ok = tk && self.images_enabled();
                ui.horizontal(|ui| {
                    ui.add_space(indent);
                    let r = ui.add_enabled(
                        images_ok,
                        egui::Checkbox::new(&mut self.trial_images, "Undistorted trial images"),
                    );
                    if tk && !self.images_enabled() {
                        r.on_disabled_hover_text("The Default trial has no images.");
                    }
                });
                ui.horizontal(|ui| {
                    ui.add_space(indent * 2.0);
                    let on = images_ok && self.trial_images;
                    let max = self.max_frame();
                    ui.add_enabled(on, egui::Label::new("from frame"));
                    if let Some(v) = form::spin_box(ui, on, self.from, 1, max, 70.0, "from frame") {
                        self.set_from(v);
                    }
                    ui.add_enabled(on, egui::Label::new("to frame"));
                    if let Some(v) = form::spin_box(ui, on, self.to, 1, max, 70.0, "to frame") {
                        self.set_to(v);
                    }
                });
                ui.horizontal(|ui| {
                    ui.add_space(indent);
                    if let Some(v) = form::check_box(ui, tk, self.rigid_bodies, "Rigid bodies") {
                        self.set_rigid_bodies(v);
                    }
                });
                ui.horizontal(|ui| {
                    ui.add_space(indent * 2.0);
                    let on = tk && self.filtered_enabled();
                    let why = self.filtered_refused().map(str::to_owned);
                    let r = ui.add_enabled(on, egui::Checkbox::new(&mut self.filtered, FILTERED));
                    let r = r.on_hover_text(FILTERED_HOVER);
                    match why {
                        Some(why) if tk => {
                            r.on_disabled_hover_text(why);
                        }
                        _ => {
                            r.on_disabled_hover_text(FILTERED_HOVER);
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.add_space(indent);
                    if let Some(v) = form::check_box(ui, tk, self.meshes, "Bone meshes") {
                        self.set_meshes(v);
                    }
                });
                ui.horizontal(|ui| {
                    ui.add_space(indent);
                    if let Some(v) = form::check_box(ui, tk, self.points, "3D points") {
                        self.points = v;
                    }
                });
            });
        });
    }

    fn blender_group(&mut self, ui: &mut egui::Ui) {
        form::group_box(ui, "Blender", |ui| {
            form::styled_frame(ui, |ui| {
                let line = match &self.blender {
                    Ok(p) => format!("Blender found: {}", p.display()),
                    Err(e) => format!("{e}."),
                };
                ui.add(egui::Label::new(line).wrap());
                let found = self.blender.is_ok();
                if let Some(v) =
                    form::check_box(ui, found, self.create_blend, "Create the .blend file now")
                {
                    self.create_blend = v;
                }
                if !found {
                    ui.add(egui::Label::new(NOT_FOUND).wrap());
                }
                ui.horizontal(|ui| {
                    ui.add_space(22.0);
                    let on = found && self.create_blend;
                    let r = ui.add_enabled(
                        on,
                        egui::Checkbox::new(&mut self.pack, "Pack still images into the .blend"),
                    );
                    r.on_hover_text(PACK_HOVER)
                        .on_disabled_hover_text(PACK_HOVER);
                });
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(name: &str, frames: usize) -> TrialChoice {
        TrialChoice {
            name: name.into(),
            frames,
            is_default: false,
            has_reference: true,
            filter_bodies_refused: Some("bodies refused".into()),
            filter_points_refused: Some("points refused".into()),
        }
    }

    fn dialog() -> BlenderSceneDialog {
        BlenderSceneDialog {
            calibrated: true,
            stem: "pig".into(),
            trial_kind: false,
            calibration_images: vec![(0, "1: a.tif".into()), (2, "3: c.tif".into())],
            calibration_choice: 0,
            calibration_image_on_plane: true,
            trials: vec![trial("T1", 100), trial("T2", 10)],
            trial: 0,
            trial_images: true,
            from: 1,
            to: 100,
            rigid_bodies: false,
            filtered: false,
            meshes: false,
            points: false,
            blender: Err("Blender not found".into()),
            create_blend: false,
            pack: false,
        }
    }

    /// The dialog's states and the spec each gives.
    #[test]
    fn spec_for_a_table_of_states() {
        let blender = PathBuf::from("/opt/blender/blender");
        type Case = (&'static str, fn(&mut BlenderSceneDialog), BlenderSceneSpec);
        let base = |k| BlenderSceneSpec::new(k);
        let cases: Vec<Case> = vec![
            (
                "cameras, first image",
                |_| {},
                base(BlenderSceneKind::Cameras {
                    calibration_image: Some(0),
                    images: true,
                }),
            ),
            (
                "cameras, second entry is image 3; trial options ignored",
                |d| {
                    d.calibration_choice = 1;
                    d.rigid_bodies = true;
                    d.points = true;
                },
                base(BlenderSceneKind::Cameras {
                    calibration_image: Some(2),
                    images: true,
                }),
            ),
            (
                "trial, whole range",
                |d| d.trial_kind = true,
                base(BlenderSceneKind::Trial {
                    trial: 0,
                    frames: Some((0, 99)),
                }),
            ),
            (
                "trial 2, frames 3 to 5, no images",
                |d| {
                    d.trial_kind = true;
                    d.set_trial(1);
                    d.set_from(3);
                    d.set_to(5);
                    d.trial_images = false;
                },
                base(BlenderSceneKind::Trial {
                    trial: 1,
                    frames: None,
                }),
            ),
            (
                "trial 2, frames 3 to 5",
                |d| {
                    d.trial_kind = true;
                    d.set_trial(1);
                    d.set_from(3);
                    d.set_to(5);
                },
                base(BlenderSceneKind::Trial {
                    trial: 1,
                    frames: Some((2, 4)),
                }),
            ),
            (
                "a Default trial has no images",
                |d| {
                    d.trial_kind = true;
                    d.trials[0].is_default = true;
                },
                base(BlenderSceneKind::Trial {
                    trial: 0,
                    frames: None,
                }),
            ),
            (
                "trial with meshes (ticks bodies), points; filtered refused",
                |d| {
                    d.trial_kind = true;
                    d.set_meshes(true);
                    d.points = true;
                    d.filtered = true;
                },
                BlenderSceneSpec {
                    rigid_bodies: true,
                    meshes: true,
                    points: true,
                    ..base(BlenderSceneKind::Trial {
                        trial: 0,
                        frames: Some((0, 99)),
                    })
                },
            ),
            (
                "filtered where the trial allows it",
                |d| {
                    d.trial_kind = true;
                    d.trials[0].filter_bodies_refused = None;
                    d.set_rigid_bodies(true);
                    d.filtered = true;
                },
                BlenderSceneSpec {
                    rigid_bodies: true,
                    filtered: true,
                    ..base(BlenderSceneKind::Trial {
                        trial: 0,
                        frames: Some((0, 99)),
                    })
                },
            ),
            (
                "unticking bodies drops meshes and filtered",
                |d| {
                    d.trial_kind = true;
                    d.trials[0].filter_bodies_refused = None;
                    d.set_meshes(true);
                    d.filtered = true;
                    d.set_rigid_bodies(false);
                },
                base(BlenderSceneKind::Trial {
                    trial: 0,
                    frames: Some((0, 99)),
                }),
            ),
            (
                "Blender not found: no .blend even if ticked",
                |d| {
                    d.create_blend = true;
                    d.pack = true;
                },
                base(BlenderSceneKind::Cameras {
                    calibration_image: Some(0),
                    images: true,
                }),
            ),
        ];
        for (what, change, want) in cases {
            let mut d = dialog();
            change(&mut d);
            assert_eq!(d.spec(), want, "{what}");
        }
        // Blender found, ticked, packed.
        let mut d = dialog();
        d.blender = Ok(blender.clone());
        d.create_blend = true;
        d.pack = true;
        let mut want = base(BlenderSceneKind::Cameras {
            calibration_image: Some(0),
            images: true,
        });
        want.blend = Some(BlendOptions {
            blender: Some(blender.clone()),
            pack: true,
        });
        assert_eq!(d.spec(), want);
        assert_eq!(
            d.written_text(),
            "A folder pig_cameras with build_scene.py, scene.json and the images and pig_cameras.blend"
        );
        d.create_blend = false;
        assert_eq!(d.spec().blend, None);
        assert_eq!(
            d.written_text(),
            "A folder pig_cameras with build_scene.py, scene.json and the images"
        );
    }

    #[test]
    fn frame_fields_stay_inside_the_trial() {
        let mut d = dialog();
        d.set_trial(1);
        assert_eq!((d.from, d.to), (1, 10));
        d.set_to(40);
        assert_eq!(d.to, 10);
        d.set_from(0);
        assert_eq!(d.from, 1);
        d.set_from(8);
        d.set_to(5);
        assert_eq!((d.from, d.to), (5, 5));
        d.set_from(9);
        assert_eq!((d.from, d.to), (9, 9));
    }

    #[test]
    fn refusals() {
        let mut d = dialog();
        assert_eq!(d.refusal(), None);
        d.calibration_images.clear();
        assert_eq!(d.refusal().as_deref(), Some(NO_COMMON_IMAGE));
        d.trial_kind = true;
        assert_eq!(d.refusal(), None);
        d.trials[0].has_reference = false;
        assert_eq!(d.refusal().as_deref(), Some(NO_REFERENCE));
        d.calibrated = false;
        assert_eq!(d.refusal().as_deref(), Some(NO_CAMERA));
        assert_eq!(image_text(4, "/a/b/cal5.tif"), "5: cal5.tif");
        // A folder's name the engine refuses: its sentence, no text of what is written.
        let mut d = dialog();
        d.stem = String::new();
        d.trial_kind = true;
        assert_eq!(d.folder_name().as_deref(), Ok("_T1"));
        d.trials[0].name = "a/b\"c".into();
        assert_eq!(d.folder_name().as_deref(), Ok("_a_b_c"));
        assert!(d.written_text().starts_with("A folder _a_b_c with"));
        d.stem = "..".into();
        d.trials[0].name = String::new();
        assert_eq!(d.folder_name().as_deref(), Ok(".._"));
        assert_eq!(image_text(0, ""), "1");
    }

    #[test]
    fn calibration_image_box_and_filtered_follow_the_engine() {
        // Cameras only without the image.
        let mut d = dialog();
        d.calibration_image_on_plane = false;
        assert_eq!(
            d.spec().kind,
            BlenderSceneKind::Cameras {
                calibration_image: Some(0),
                images: false,
            }
        );
        assert_eq!(
            d.written_text(),
            "A folder pig_cameras with build_scene.py, scene.json (no images)"
        );
        // Filtered: nothing it filters ticked, disabled.
        let mut d = dialog();
        d.trial_kind = true;
        assert!(!d.filtered_enabled());
        assert_eq!(d.filtered_refused(), None);
        // Points ticked, the engine refuses filtered points: disabled, with its sentence.
        d.points = true;
        assert!(!d.filtered_enabled());
        assert_eq!(d.filtered_refused(), Some("points refused"));
        d.filtered = true;
        assert!(!d.spec().filtered);
        // Points can be filtered: enabled, and the spec filters them without bodies.
        d.trials[0].filter_points_refused = None;
        assert!(d.filtered_enabled());
        assert!(d.spec().filtered && d.spec().points && !d.spec().rigid_bodies);
        // Bodies ticked too, which the engine refuses: disabled, the bodies' sentence.
        d.set_rigid_bodies(true);
        assert_eq!(d.filtered_refused(), Some("bodies refused"));
        assert!(!d.spec().filtered);
        d.trials[0].filter_bodies_refused = None;
        assert!(d.spec().filtered);
    }
}
