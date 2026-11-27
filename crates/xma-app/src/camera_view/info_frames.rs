//! The per-view info frames and the external-calibration form.
//!
//! - [`show`] draws, inside the camera view's `frameInfo` (shown while the view's info
//!   button is checked), the frame of the current workspace, as
//!   `CameraViewWidget::workspaceChanged` shows one of them:
//!   - `UndistortionInfoFrame` (`ui/UndistortionInfoFrame.cpp`): "Nb Points :" and
//!     "Error :" of the grid, empty while the grid is not computed;
//!   - `CalibrationInfoFrame` (`ui/CalibrationInfoFrame.cpp`): the camera's intrinsics,
//!     the active image's pose, errors and inliers, and "Render options" (scale, bias);
//!   - `DigitizationInfoFrame` (`ui/DigitizationInfoFrame.cpp`): scale, bias, the mesh
//!     transparency, "Reset" and "Draw transparent bone models".
//! - [`external_calibration`] draws `ExternalCalibrationFrame` in place of the image in the
//!   Calibration workspace when the calibration is external: the camera's resolution,
//!   camera matrix, rotation matrix and translation of its first calibration image, and
//!   lens coefficients, editable; "Load MayaCam 2.0" and "APPLY CHANGES" (C43 to C46):
//!   [`external`].
//!
//! Scale and bias: the C++ stores them in the project per camera and view
//! (`Camera::ImageAdjustment`, `CameraViewWidget::setScale` and `setBias` through
//! `activeImageAdjustment`: the Calibration info frame writes `CalibrationView`, the
//! Digitization info frame `DigitizationView`). Here the view shows
//! `crate::camera_view::CameraViewState::adjust`; every change, "Reset" included, is also
//! written to the document's camera through an edit request
//! ([`crate::editing::EditRequest::ImageAdjustment`]) so that Save stores it (parity ledger
//! U46). Undo and redo leave it alone.

pub mod external;

use std::sync::Arc;

use egui::Ui;
use xma_engine::Document;
use xma_engine::calibration::HeldPoints;
use xma_engine::view::calibration::{CalibrationInfo, calibration_info};
use xma_model::{Camera, Point2};
use xma_vision::camera::ModelCompat;
use xma_vision::undistortion::LwmTransform;

use crate::frames::ViewStates;

use crate::editing::EditRequest;
use crate::panels::Cx;
use crate::panels::detail_view::adjust_row;
use crate::widgets::image_view::ImageAdjust;
use xma_model::AdjustedView;

/// Writes a view's scale and bias to the document (`CameraViewWidget::setScale`, `setBias`).
fn write_adjustment(cx: &mut Cx<'_>, camera: usize, view: AdjustedView, a: ImageAdjust) {
    cx.request(EditRequest::ImageAdjustment {
        camera,
        view,
        scale: a.scale,
        bias: a.bias,
    });
}
use crate::state::Workspace;

/// A group box with a title (`QGroupBox`).
fn group(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui)) {
    ui.label(egui::RichText::new(title));
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(4))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
}

/// The info frame of the current workspace for `camera`.
pub fn show(ui: &mut Ui, cx: &mut Cx<'_>, camera: usize) {
    match cx.state.workspace {
        Workspace::Undistortion => undistortion(ui, cx, camera),
        Workspace::Calibration => calibration(ui, cx, camera),
        Workspace::MarkerTracking => digitization(ui, cx, camera),
    }
}

/// `UndistortionInfoFrame`.
fn undistortion(ui: &mut Ui, cx: &mut Cx<'_>, camera: usize) {
    let g = super::overlay_undistortion::grid(cx, camera);
    let (nb, err) = g.map_or((String::new(), String::new()), |g| {
        (g.nb_points_text.clone(), g.error_text.clone())
    });
    group(ui, "Undistortion Info", |ui| {
        egui::Grid::new(("undistortion_info", camera))
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Nb Points : ");
                ui.label(nb);
                ui.end_row();
                ui.label("Error : ");
                ui.label(err);
                ui.end_row();
            });
    });
}

/// `CalibrationInfoFrame`.
fn calibration(ui: &mut Ui, cx: &mut Cx<'_>, camera: usize) {
    let Some(doc) = cx.doc else { return };
    let name = doc.project().cameras[camera].name.clone();
    let info = cached_calibration_info(
        &mut cx.frames.views,
        doc,
        camera,
        cx.state.active_frame_calibration.unwrap_or(0),
    );
    group(ui, "Calibration Info", |ui| {
        egui::Grid::new(("calibration_info", camera))
            .num_columns(4)
            .show(ui, |ui| {
                ui.label("Focal length : ");
                ui.label(&info.focal_length);
                ui.label("Camera center : ");
                ui.label(&info.camera_center);
                ui.end_row();
                ui.label("Modeldistortion:");
                ui.label(&info.distortion);
                ui.end_row();
                ui.label("Rotation vector : ");
                ui.label(&info.rotation_vector);
                ui.end_row();
                ui.label("Translation vector : ");
                ui.label(&info.translation_vector);
                ui.end_row();
                ui.label("Error distorted (current frame) :");
                ui.label(&info.error_current_dist);
                ui.label("(all frames) : ");
                ui.label(&info.error_all_dist);
                ui.end_row();
                ui.label("Error undistorted (current frame) :");
                ui.label(&info.error_current_undist);
                ui.label("(all frames) : ");
                ui.label(&info.error_all_undist);
                ui.end_row();
                ui.label("Frames calibrated : ");
                ui.label(&info.frames_calibrated);
                ui.label("Inlier (current / all) : )");
                ui.label(&info.inlier);
                ui.end_row();
            });
    });
    let adj =
        &mut cx.frames.views.cameras[camera].adjust[xma_model::AdjustedView::Calibration as usize];
    let mut changed = false;
    group(ui, "Render options", |ui| {
        egui::Grid::new(("calibration_render", camera))
            .num_columns(3)
            .show(ui, |ui| {
                let p = format!("{name} calibration");
                changed |= adjust_row(
                    ui,
                    &p,
                    "Scale",
                    &mut adj.scale,
                    0.0..=3.0,
                    0.05,
                    1.0,
                    (0, 300),
                );
                ui.end_row();
                changed |= adjust_row(
                    ui,
                    &p,
                    "Bias",
                    &mut adj.bias,
                    -1.0..=1.0,
                    0.01,
                    0.0,
                    (-100, 100),
                );
                ui.end_row();
            });
    });
    if changed {
        let a = *adj;
        write_adjustment(cx, camera, AdjustedView::Calibration, a);
    }
}

/// `DigitizationInfoFrame`.
fn digitization(ui: &mut Ui, cx: &mut Cx<'_>, camera: usize) {
    let Some(doc) = cx.doc else { return };
    let name = doc.project().cameras[camera].name.clone();
    let p = format!("{name} digitization");
    let s = &mut cx.frames.views.cameras[camera];
    let mut changed = false;
    group(ui, "Digitization Info", |ui| {
        egui::Grid::new(("digitization_info", camera))
            .num_columns(3)
            .show(ui, |ui| {
                let adj = &mut s.adjust[xma_model::AdjustedView::Digitization as usize];
                changed |= adjust_row(
                    ui,
                    &p,
                    "Scale",
                    &mut adj.scale,
                    0.0..=3.0,
                    0.05,
                    1.0,
                    (0, 300),
                );
                ui.end_row();
                changed |= adjust_row(
                    ui,
                    &p,
                    "Bias",
                    &mut adj.bias,
                    -1.0..=1.0,
                    0.01,
                    0.0,
                    (-100, 100),
                );
                ui.end_row();
                // Transparency: slider 0 to 100, spin box 0 to 1 (no double-click reset).
                ui.label("Transparency");
                let mut t = (s.transparency * 100.0).round() as i32;
                if ui
                    .add(egui::Slider::new(&mut t, 0..=100).show_value(false))
                    .accessible_name(format!("{p} transparency slider"))
                    .changed()
                {
                    s.transparency = f64::from(t) / 100.0;
                }
                let mut v = s.transparency;
                if ui
                    .add_sized(
                        [48.0, ui.spacing().interact_size.y],
                        egui::DragValue::new(&mut v)
                            .range(0.0..=1.0)
                            .speed(0.01)
                            .fixed_decimals(2),
                    )
                    .accessible_name(format!("{p} transparency"))
                    .changed()
                {
                    s.transparency = (v * 100.0).round() / 100.0;
                }
                ui.end_row();
                // DigitizationInfoFrame::on_pushButtonReset_clicked: scale 1, bias 0,
                // transparency 0.5, transparent models on.
                if ui.button("Reset").clicked() {
                    changed |= *adj != ImageAdjust::default();
                    *adj = ImageAdjust::default();
                    s.transparency = 0.5;
                    s.transparent_models = true;
                }
                ui.label("");
                ui.checkbox(&mut s.transparent_models, "Draw transparent bone models");
                ui.end_row();
            });
    });
    if changed {
        let a = s.adjust[AdjustedView::Digitization as usize];
        write_adjustment(cx, camera, AdjustedView::Digitization, a);
    }
}

/// `ExternalCalibrationFrame`, filling the view ([`external`]).
pub fn external_calibration(ui: &mut Ui, cx: &mut Cx<'_>, camera: usize) {
    external::show(ui, cx, camera);
}

/// Everything [`calibration_info`] reads for one camera and image, held so that a change of
/// any of it is seen: the project's camera (its calibration images, poses, inlier flags,
/// camera matrix, lens coefficients and calibrated flag), the calibration object's points,
/// and the document's derived camera as `calibration_image` reads it (the held point lists
/// of every image, the camera model, the camera matrix the lens points use). Shared values
/// are compared by pointer while a reference to them is held here, so they cannot have
/// changed (nothing writes into a value that is shared: the document copies on write);
/// numbers are compared by their bits.
pub struct CalibrationInfoKey {
    image: usize,
    camera: Arc<Camera>,
    object: Option<Vec<[u64; 3]>>,
    held: Vec<Option<Arc<HeldPoints>>>,
    camera_matrix: [u64; 9],
    distortion: Option<[u64; 8]>,
    compat: ModelCompat,
    /// The grid model's undistorting and distorting transforms.
    grid: Option<[LwmKey; 2]>,
    lens_points_camera_matrix: Option<[u64; 9]>,
}

/// The values of an [`LwmTransform`] (its index is a cache that does not change results).
struct LwmKey {
    control_points: Arc<[Point2]>,
    a: Arc<[[f64; 6]]>,
    b: Arc<[[f64; 6]]>,
    radii: Arc<[f64]>,
}

impl LwmKey {
    fn of(t: &LwmTransform) -> Self {
        Self {
            control_points: t.control_points.clone(),
            a: t.a.clone(),
            b: t.b.clone(),
            radii: t.radii.clone(),
        }
    }

    fn same(&self, o: &Self) -> bool {
        Arc::ptr_eq(&self.control_points, &o.control_points)
            && Arc::ptr_eq(&self.a, &o.a)
            && Arc::ptr_eq(&self.b, &o.b)
            && Arc::ptr_eq(&self.radii, &o.radii)
    }
}

fn matrix_bits(m: &xma_vision::Matrix3) -> [u64; 9] {
    let mut out = [0; 9];
    for (o, v) in out.iter_mut().zip(m.iter()) {
        *o = v.to_bits();
    }
    out
}

impl CalibrationInfoKey {
    /// The key of `camera` (which must exist) and `image` in `doc`.
    fn of(doc: &Document, camera: usize, image: usize) -> Self {
        let cam = &doc.project().cameras[camera];
        let derived = doc.camera(camera);
        let model = &derived.model;
        Self {
            image,
            camera: cam.clone(),
            object: doc.project().calibration_object.as_ref().map(|o| {
                o.points()
                    .iter()
                    .map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
                    .collect()
            }),
            held: (0..cam.calibration_images.len())
                .map(|i| doc.camera_held_points(camera, i).cloned())
                .collect(),
            camera_matrix: matrix_bits(&model.intrinsics.camera_matrix),
            distortion: model.intrinsics.distortion.map(|d| d.map(f64::to_bits)),
            compat: model.compat,
            grid: model
                .grid
                .as_ref()
                .map(|g| [LwmKey::of(&g.undistort), LwmKey::of(&g.distort)]),
            lens_points_camera_matrix: derived.lens_points_camera_matrix.as_ref().map(matrix_bits),
        }
    }

    fn same(&self, o: &Self) -> bool {
        self.image == o.image
            && Arc::ptr_eq(&self.camera, &o.camera)
            && self.object == o.object
            && self.held.len() == o.held.len()
            && self.held.iter().zip(&o.held).all(|(x, y)| match (x, y) {
                (None, None) => true,
                (Some(x), Some(y)) => Arc::ptr_eq(x, y),
                _ => false,
            })
            && self.camera_matrix == o.camera_matrix
            && self.distortion == o.distortion
            && self.compat == o.compat
            && match (&self.grid, &o.grid) {
                (None, None) => true,
                (Some([u, d]), Some([ou, od])) => u.same(ou) && d.same(od),
                _ => false,
            }
            && self.lens_points_camera_matrix == o.lens_points_camera_matrix
    }
}

/// [`calibration_info`] of `camera` with calibration image `image` active, kept per camera
/// in `views` and computed again only when something it reads has changed
/// ([`CalibrationInfoKey`]): an edit, undo or redo, another image, another document. The
/// frame is drawn on every repaint, and computing it undistorts every point of every
/// calibration image of the camera (review of 20 November 2026, engine-tracking-view
/// finding 3).
pub fn cached_calibration_info(
    views: &mut ViewStates,
    doc: &Document,
    camera: usize,
    image: usize,
) -> Arc<CalibrationInfo> {
    if camera >= doc.project().cameras.len() {
        return Arc::new(calibration_info(doc, camera, image));
    }
    let key = CalibrationInfoKey::of(doc, camera, image);
    if let Some((k, info)) = views.calibration_info.get(&camera)
        && k.same(&key)
    {
        return info.clone();
    }
    let info = Arc::new(calibration_info(doc, camera, image));
    views.calibration_info.insert(camera, (key, info.clone()));
    info
}
