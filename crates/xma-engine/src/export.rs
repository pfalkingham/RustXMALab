//! The File > Export actions: what each `MainWindow` handler checks and runs before the
//! export function writes.
//!
//! Ported from ui/MainWindow.cpp (`:1702-2067`). The files themselves come from
//! [`xma_io::export`]; this module adds the GUI's refusals (with the C++ messages), the
//! recomputation the rigid-body export runs first (ledger A36), and the derived data the
//! model does not hold (camera models for undistorted 2D points, lookup tables).
//!
//! The C++ exports the **active** trial; here the trial is an argument. Selecting a trial in
//! the GUI updates it ([`Document::select_trial`]); callers that mean "the trial the GUI
//! would show" select it first.

use std::fmt;

use xma_io::export::points2d::points_2d;
use xma_io::export::points3d::points_3d;
use xma_io::export::rigid_bodies::transformations;
use xma_io::export::{events, mayacam, precision, reprojection, undistortion};
// The option types and errors of the exports, for callers that do not use `xma_io` (the
// window program).
pub use xma_io::export::points2d::Points2dOptions;
pub use xma_io::export::points3d::Points3dOptions;
pub use xma_io::export::rigid_bodies::{TransformationsOptions, can_export_filtered};
pub use xma_io::export::{ExportError, ExportFile, ExportOutput, FrameRange};
use xma_vision::undistortion::{LookupTable, create_lookup_table};

use crate::Document;

/// The C++ message when the project is not calibrated (`ui/MainWindow.cpp:1718`, `:2065`).
pub const NOT_CALIBRATED: &str = "You have to first calibrate the cameras to be able to export";
/// The C++ message for a filtered 3D export without a recording speed (`:1734`).
pub const NO_RECORDING_SPEED: &str =
    "Recording speed is not set for the trial. First set the recording speed and repeat the export";
/// The C++ message for a filtered rigid-body export without recording speed or cutoff
/// (`:1880`).
pub const NO_SPEED_OR_CUTOFF: &str = "Recording speed or Cutoff frequency are not set for the trial. First set the recording speed and Cutoff frequency and repeat the export";

/// One export, with its options. Frame ranges are 0-based, as the C++ functions take them.
#[derive(Clone, Debug, PartialEq)]
pub enum Export {
    /// File > Export 2D Points (`on_actionExport2D_Points_triggered`).
    Points2d(Points2dOptions),
    /// File > Export 3D Points (`save3DPoints`): the markers given, or every marker.
    Points3d {
        markers: Option<Vec<usize>>,
        options: Points3dOptions,
    },
    /// File > Export Rigid Body Transformations (`saveRigidBodies`): the bodies given, or
    /// every body.
    RigidBodies {
        bodies: Option<Vec<usize>>,
        options: TransformationsOptions,
    },
    /// Marker-to-marker distances over frames `from..to` (the GUI passes `From - 1` and `To`
    /// of its dialog).
    MarkerDistances { from: usize, to: usize },
    /// Precision info over frames `from..to`.
    PrecisionInfo { from: usize, to: usize },
    /// Reprojection errors, one file per marker.
    ReprojectionErrors,
    /// MayaCams 2.0: every calibration image ("Yes") or only the trial's reference
    /// calibration ("No").
    MayaCams { all: bool },
    /// The undistortion part of Export All (`Project::exportLUT`): lookup table, inlier and
    /// base points per camera with a computed grid. DLT and MayaCam version 1 are not ported.
    UndistortionLut,
    /// Events, one file per event.
    Events,
}

impl Export {
    /// The file name the reference dumps use for an export the C++ writes to a file the user
    /// names (`oracle/cpp/DEVIATIONS.md`, item 20); `None` for an export into a folder.
    #[must_use]
    pub fn default_file_name(&self) -> Option<&'static str> {
        match self {
            Self::Points2d(o) if o.one_file => Some(if o.distorted {
                "2DPoints.csv"
            } else {
                "2DPoints_undistorted.csv"
            }),
            Self::Points3d { options, .. } if options.one_file => {
                Some(if options.filter_frequency > 0.0 {
                    "3DPoints_filtered.csv"
                } else {
                    "3DPoints.csv"
                })
            }
            Self::RigidBodies { options, .. } if options.one_file => Some(if options.filtered {
                "RigidBodyTransformations_filtered.csv"
            } else {
                "RigidBodyTransformations.csv"
            }),
            Self::MarkerDistances { .. } => Some("MarkerToMarkerDistances.csv"),
            Self::PrecisionInfo { .. } => Some("PrecisionInfo.csv"),
            _ => None,
        }
    }
}

/// Why an export wrote nothing.
#[derive(Clone, Debug, PartialEq)]
pub enum ExportFailure {
    /// The trial does not exist.
    NoSuchTrial(usize),
    /// The GUI refuses the export with this message.
    Refused(&'static str),
    /// The export function refused its input (see [`ExportError`]).
    Export(ExportError),
}

impl fmt::Display for ExportFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSuchTrial(t) => write!(f, "trial {} does not exist", t + 1),
            Self::Refused(m) => write!(f, "{m}"),
            Self::Export(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ExportFailure {}

impl From<ExportError> for ExportFailure {
    fn from(e: ExportError) -> Self {
        Self::Export(e)
    }
}

impl Document {
    /// Runs one export on a trial as the GUI's handler does, and returns the file or files.
    /// MayaCams of every calibration image (`all: true`) and the lookup tables read no
    /// trial, so `trial` may be any value for them (a project without trials).
    ///
    /// Only the rigid-body export changes the document: it recomputes each exported body
    /// first (`Trial::saveRigidBodyTransformations`, `:951-955`; ledger A36), so `&mut`.
    ///
    /// # Errors
    /// [`ExportFailure::NoSuchTrial`]; [`ExportFailure::Refused`] where the GUI shows an
    /// error instead of exporting (MayaCams and Export All on an uncalibrated project, a
    /// filtered 3D export without a recording speed, a filtered rigid-body export without
    /// recording speed or cutoff); [`ExportFailure::Export`] for input the C++ would read out
    /// of bounds on (a frame range past the trial, an index that does not exist) or a cutoff
    /// `Marker::filterMarker` refuses. Nothing is changed when an error is returned.
    pub fn export(&mut self, trial: usize, export: &Export) -> Result<ExportOutput, ExportFailure> {
        // The two project-wide exports read no trial when every calibration image is
        // exported (XMALab offers MayaCams in a project without trials).
        let needs_trial = !matches!(
            export,
            Export::MayaCams { all: true } | Export::UndistortionLut
        );
        if needs_trial && trial >= self.project.trials.len() {
            return Err(ExportFailure::NoSuchTrial(trial));
        }
        match export {
            Export::Points2d(options) => {
                let models: Vec<_> = (0..self.project.cameras.len())
                    .map(|c| self.camera(c).model.clone())
                    .collect();
                let t = &self.project.trials[trial];
                // The document's mode decides (ledger F2, F25), whatever the options say.
                let options = Points2dOptions {
                    compat: self.compat(),
                    ..*options
                };
                Ok(points_2d(t, &self.project.cameras, &models, &options)?)
            }
            Export::Points3d { markers, options } => {
                let t = &self.project.trials[trial];
                // MainWindow::save3DPoints (`:1733-1736`).
                if options.filter_frequency > 0.0 && t.recording_speed <= 0.0 {
                    return Err(ExportFailure::Refused(NO_RECORDING_SPEED));
                }
                let all: Vec<usize> = (0..t.markers.len()).collect();
                let markers = markers.as_deref().unwrap_or(&all);
                let options = Points3dOptions {
                    compat: self.compat(),
                    ..*options
                };
                Ok(points_3d(t, markers, &options)?)
            }
            Export::RigidBodies { bodies, options } => {
                let t = &self.project.trials[trial];
                let options = &TransformationsOptions {
                    compat: self.compat(),
                    ..*options
                };
                let all: Vec<usize> = (0..t.rigid_bodies.len()).collect();
                let bodies = bodies.clone().unwrap_or(all);
                // MainWindow::saveRigidBodies (`:1878-1881`; ledger F11).
                if options.filtered && !can_export_filtered(t, &bodies, self.compat()) {
                    return Err(ExportFailure::Refused(NO_SPEED_OR_CUTOFF));
                }
                check_bodies_and_range(t, &bodies, options.frames)?;
                self.recompute_rigid_bodies_for_export(trial, &bodies, options.filtered);
                Ok(transformations(
                    &self.project.trials[trial],
                    &bodies,
                    options,
                )?)
            }
            Export::MarkerDistances { from, to } => Ok(ExportOutput::File(
                precision::marker_to_marker_distances(&self.project.trials[trial], *from, *to)?,
            )),
            Export::PrecisionInfo { from, to } => Ok(ExportOutput::File(
                precision::precision_info(&self.project.trials[trial], *from, *to, self.compat())?,
            )),
            Export::ReprojectionErrors => {
                Ok(ExportOutput::Folder(reprojection::reprojection_errors(
                    &self.project.trials[trial],
                    self.project.cameras.len(),
                    self.compat(),
                )?))
            }
            Export::MayaCams { all } => {
                // MainWindow::on_actionMayaCams_2_0_triggered (`:2049`).
                if !self.is_calibrated() {
                    return Err(ExportFailure::Refused(NOT_CALIBRATED));
                }
                let frame = if *all {
                    None
                } else {
                    Some(self.project.trials[trial].reference_calibration_image)
                };
                Ok(ExportOutput::Folder(mayacam::maya_cams_v2(
                    &self.project.cameras,
                    frame,
                )))
            }
            Export::UndistortionLut => {
                // MainWindow::on_actionExportAll_triggered (`:1704`).
                if !self.is_calibrated() {
                    return Err(ExportFailure::Refused(NOT_CALIBRATED));
                }
                Ok(ExportOutput::Folder(undistortion::export_lut(
                    &self.project.cameras,
                    &self.lookup_tables(),
                )))
            }
            Export::Events => Ok(ExportOutput::Folder(events::events(
                &self.project.trials[trial],
                self.compat(),
            ))),
        }
    }

    /// Each camera's undistortion lookup table, built from its fitted grid model at the
    /// camera's image size (`LocalUndistortion::createLookupTable`); `None` for a camera
    /// without a fitted grid. The table the document holds is taken when it has that size:
    /// it is the grid model's table at its size, bit for bit ([`crate::document::DerivedCamera::lookup`]), as
    /// `trial_images` and `view::calibration` take it.
    #[must_use]
    pub fn lookup_tables(&self) -> Vec<Option<LookupTable>> {
        self.project
            .cameras
            .iter()
            .enumerate()
            .map(|(c, cam)| {
                let derived = self.camera(c);
                derived.model.grid.as_ref().map(|g| {
                    let (w, h) = (cam.width as usize, cam.height as usize);
                    match &derived.lookup {
                        Some(l) if (l.width, l.height) == (w, h) => LookupTable::clone(l),
                        _ => create_lookup_table(&g.distort, w, h),
                    }
                })
            })
            .collect()
    }
}

/// The file the Rigid body dialog's "Export references" writes: a header of
/// `<name>_x,<name>_y,<name>_z` per reference point, then the (optimised) reference points
/// on one line, with 17 significant digits (`%.17g`).
///
/// Ported from core/RigidBody.cpp, RigidBody::saveOptimized (`:1477-1516`, `withHeader`),
/// as `RigidBodyDialog::on_pushButtonExport_clicked` calls it. A name missing for a point
/// is written empty.
#[must_use]
pub fn rigid_body_references(names: &[String], points: &[xma_model::Point3]) -> String {
    use xma_io::cfmt::format_g17 as g;
    let mut s = String::new();
    let n = points.len();
    for j in 0..n {
        let name = names.get(j).map_or("", String::as_str);
        s.push_str(&format!("{name}_x,{name}_y,{name}_z"));
        s.push_str(if j + 1 == n { "\n" } else { "," });
    }
    for (j, p) in points.iter().enumerate() {
        s.push_str(&format!("{},{},{}", g(p.x), g(p.y), g(p.z)));
        s.push_str(if j + 1 == n { "\n" } else { "," });
    }
    s
}

/// The checks of [`transformations`], run before the recomputation so that a refused export
/// changes nothing.
fn check_bodies_and_range(
    trial: &xma_model::Trial,
    bodies: &[usize],
    frames: Option<FrameRange>,
) -> Result<(), ExportError> {
    if let Some(&b) = bodies.iter().find(|&&b| b >= trial.rigid_bodies.len()) {
        return Err(ExportError::NoSuchBody(b));
    }
    match frames {
        Some(r) if r.start <= r.stop && r.stop >= trial.n_frames => Err(ExportError::FrameRange {
            first: r.start,
            end: r.stop + 1,
            n_frames: trial.n_frames,
        }),
        _ => Ok(()),
    }
}

/// Writes the files of an export into a folder: a one-file export as `file_name`, a folder
/// export under the names the C++ gives its files. Returns the paths written, in order.
/// The default mode's names ([`write_output_with`] with [`xma_model::CppCompat::FIXED`]).
///
/// # Errors
/// When a file cannot be written, or a name is refused ([`xma_io::export::safe_component`]).
pub fn write_output(
    dir: &std::path::Path,
    output: &ExportOutput,
    file_name: &str,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    write_output_with(dir, output, file_name, xma_model::CppCompat::FIXED)
}

/// [`write_output`] in a mode: each name of a folder export goes through
/// [`xma_io::export::safe_component`] (ledger F70): `.`, `..`, an empty name and (XMALab 3
/// mode) a name with a separator are refused rather than written outside the folder; in the
/// default mode characters no system can store are replaced.
///
/// # Errors
/// As [`write_output`].
pub fn write_output_with(
    dir: &std::path::Path,
    output: &ExportOutput,
    file_name: &str,
    compat: xma_model::CppCompat,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    let single;
    let files: &[ExportFile] = match output {
        ExportOutput::File(text) => {
            single = [ExportFile {
                name: file_name.to_owned(),
                contents: text.clone(),
            }];
            &single
        }
        ExportOutput::Folder(files) => files,
    };
    let mut written = Vec::with_capacity(files.len());
    let one_file = matches!(output, ExportOutput::File(_));
    for f in files {
        // A one-file export's name is the file the user chose.
        let name = if one_file {
            f.name.clone()
        } else {
            xma_io::export::safe_component(&f.name, compat.export_names_as_given)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?
        };
        let path = dir.join(&name);
        std::fs::write(&path, &f.contents)?;
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xma_model::Point3;

    #[test]
    fn references_are_written_as_save_optimized_writes_them() {
        let text = rigid_body_references(
            &["a".into(), "b".into()],
            &[
                Point3::new(0.1, -2.0, 1e-5),
                Point3::new(3.0, 4.5, 123_456.0),
            ],
        );
        assert_eq!(
            text,
            "a_x,a_y,a_z,b_x,b_y,b_z\n0.10000000000000001,-2,1.0000000000000001e-05,3,4.5,123456\n"
        );
        assert_eq!(rigid_body_references(&[], &[]), "");
        assert_eq!(
            rigid_body_references(&[], &[Point3::new(f64::NAN, 0.0, -0.0)]),
            "_x,_y,_z\nnan,0,-0\n"
        );
    }
}
