//! Editing a project: every change the Marker tracking workspace of XMALab 3 can make,
//! as one type ([`Edit`]) applied by one function ([`Document::apply`]).
//!
//! `docs/cpp-editing.md` lists the C++ actions (E1 to E82) and what each calls; the variants
//! below name the action they port. The interface builds an [`Edit`] and hands it to the
//! document; it never writes to the model itself. Each applied edit is one undo step
//! ([`crate::history`]).
//!
//! # Rules every edit follows
//!
//! - **Frames, markers, cameras, bodies, trials and events are 0-based indices.** The C++
//!   dialogs show frames 1-based; converting is the interface's business. Only
//!   `Trial::start_frame` and `end_frame` are 1-based, because the model stores them so.
//! - **An edit that cannot be applied changes nothing** and returns an [`EditError`]: an
//!   index out of range or an edit that does not apply is [`EditError::Invalid`]; where
//!   XMALab shows an error dialog and stops, [`EditError::Rejected`] carries XMALab's text.
//! - **No edit creates a cycle of virtual points** (rigid bodies animating each other
//!   through virtual points, round to the first; XMALab recurses without end there). Such
//!   an edit is [`EditError::Rejected`] in both modes, with a sentence naming the bodies
//!   (ledger A92). A project opened with a cycle keeps it; the computation cuts it.
//! - **Confirmations and input dialogs belong to the interface.** An edit is what happens
//!   after the user has confirmed.
//! - **Derived data is kept up to date** in the default mode: after an edit the 3D points,
//!   errors, raw poses, rotation continuity, filtered poses and filtered errors of the trial
//!   are what a full recomputation from the same 2D data and settings would give (parity
//!   ledger A31, A60). In the XMALab 3 mode ([`xma_model::CppCompat::XMALAB3`]) an edit
//!   recomputes exactly what the C++ call recomputes and no more.
//! - **Known C++ defects are fixed by default** and reproduced in the XMALab 3 mode where
//!   the C++ behaviour is defined (ledger rows named at each function).
//!
//! # Layout
//!
//! One file per group of edits, each holding the functions that [`Document::apply`]
//! calls, plus [`ops`], the ports of the C++ model calls they share (`Marker::setPoint`,
//! `Marker::reset`, `RigidBody::addPointIdx`, ...).
//!
//! Edits that need images (placing a point with detection, tracking, the polynomial
//! refinement) are not in [`Edit`]: they are functions of [`crate::tracking`], which use
//! [`ops`] and the same step mechanism.

use std::fmt;
use std::path::PathBuf;

use xma_model::{AdjustedView, DetectionMethod, Interpolation, MarkerStatus, Rgb, SettingsPatch};

use crate::document::Document;

pub mod bodies;
pub mod body_ops;
pub mod events;
pub mod import;
pub mod markers;
pub mod ops;
pub mod points;
pub mod script;
pub mod settings;
pub mod structure;
pub mod trials;

/// Frames `first..=last`, 0-based. An edit given `first > last` does nothing, as the C++
/// range loops do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameRange {
    pub first: usize,
    pub last: usize,
}

impl FrameRange {
    /// The range holding one frame.
    #[must_use]
    pub fn one(frame: usize) -> Self {
        FrameRange {
            first: frame,
            last: frame,
        }
    }
}

/// Which cameras an edit applies to (the C++ `camera == -1` for all).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraSel {
    All,
    One(usize),
}

/// Which error "delete everything above the threshold" tests (E47).
#[derive(Clone, Debug, PartialEq)]
pub enum ErrorTarget {
    /// Plot type 3: a marker's reprojection error, in one camera or the mean over cameras.
    Marker { marker: usize, cameras: CameraSel },
    /// Plot type 5: a rigid body's error. `kind` is the plot's `comboBoxRigidBodyTransType`
    /// index (0 unfiltered, 1 filtered, 2 the larger of the two); `error` the plot's
    /// `comboBoxRigidBodyError` (3D or 2D error).
    Body {
        body: usize,
        kind: usize,
        error: BodyErrorKind,
    },
}

/// Which mean error of a rigid body the plot shows (`comboBoxRigidBodyError`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BodyErrorKind {
    /// Index 0, "3D": the mean 3D error.
    #[default]
    Error3d,
    /// Index 1, "2D": the mean reprojection error.
    Error2d,
}

/// Where an added virtual point takes its position from (E42).
#[derive(Clone, Debug, PartialEq)]
pub enum VirtualPointInput {
    /// Animated by another rigid body. `driver_file` is the second CT coordinate file (the
    /// point in the driving body's frame); `None` uses the first file again.
    Body {
        body: usize,
        driver_file: Option<PathBuf>,
    },
    /// Animated by a file of tracked coordinates, one line per frame.
    Coordinates { file: PathBuf },
}

/// The options of File > Import > 2D Points (E66), as `Trial::load2dPoints` takes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Import2dOptions {
    pub distorted: bool,
    /// Coordinates count from 1.
    pub count_from_one: bool,
    pub y_up: bool,
    pub header: bool,
    /// The file has "cam offset" columns (MATLAB XROMM tools).
    pub offset_columns: bool,
    /// Imported points get status `SET`; otherwise `TRACKED`.
    pub status_set: bool,
}

/// The values of the Marker dialog (E29); `None` leaves a value as it is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MarkerProperties {
    pub method: Option<DetectionMethod>,
    pub threshold_offset: Option<i32>,
    pub size_override: Option<i32>,
    pub max_penalty: Option<i32>,
}

/// The values of the Trial dialog (E57 to E61); `None` leaves a value as it is.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrialProperties {
    pub recording_speed: Option<f64>,
    pub cutoff_frequency: Option<f64>,
    pub interpolate_3d: Option<bool>,
    /// Index of the reference calibration image.
    pub reference_calibration: Option<usize>,
    /// "Force Update Trial": recompute the whole trial whatever changed.
    pub force_update: bool,
}

/// One change to the project. The comment on each variant gives the action of
/// `docs/cpp-editing.md` it ports and the file whose function applies it.
#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    // ----- points.rs: 2D point data -------------------------------------------------
    /// Writes a 2D point and its status (`Marker::setPoint`). E3 (a `MANUAL` placement).
    SetPoint {
        trial: usize,
        marker: usize,
        camera: usize,
        frame: usize,
        x: f64,
        y: f64,
        status: MarkerStatus,
    },
    /// Writes a detected size and then the point (`Marker::setSize`, `Marker::setPoint`):
    /// what a finished detection writes.
    SetDetectedPoint {
        trial: usize,
        marker: usize,
        camera: usize,
        frame: usize,
        x: f64,
        y: f64,
        size: f64,
        status: MarkerStatus,
    },
    /// Deletes the data of the markers over a frame range, or marks it untrackable
    /// (`Marker::resetMultipleFrames`). E18, E44, E45.
    ResetFrames {
        trial: usize,
        markers: Vec<usize>,
        cameras: CameraSel,
        frames: FrameRange,
        untrackable: bool,
    },
    /// Deletes every point whose error is above a threshold (`Marker::reset` per cell). E47.
    ResetAboveError {
        trial: usize,
        target: ErrorTarget,
        threshold: f64,
    },
    /// Exchanges the tracked data of two markers over a frame range. E22.
    SwapMarkerData {
        trial: usize,
        marker: usize,
        other: usize,
        cameras: CameraSel,
        frames: FrameRange,
    },
    /// Fills gaps by each frame's interpolation method (`Marker::interpolate`). E13 (one
    /// marker), E14 (every marker of the trial).
    Interpolate { trial: usize, markers: Vec<usize> },
    /// Sets the interpolation method of a frame range; gaps are filled by
    /// [`Edit::Interpolate`]. E46.
    SetInterpolation {
        trial: usize,
        marker: usize,
        frames: FrameRange,
        method: Interpolation,
    },

    // ----- markers.rs: marker properties ---------------------------------------------
    /// E15 for a marker.
    SetMarkerDescription {
        trial: usize,
        marker: usize,
        description: String,
    },
    /// E20 (several markers), E55.
    SetMarkerMethod {
        trial: usize,
        markers: Vec<usize>,
        method: DetectionMethod,
    },
    /// The Marker dialog's OK (E29), and E56 (threshold offset alone).
    SetMarkerProperties {
        trial: usize,
        marker: usize,
        properties: MarkerProperties,
    },

    // ----- structure.rs: how many markers and bodies, and which marker is in which body
    /// `Trial::addMarker`.
    AddMarker { trial: usize },
    /// E19: removes the markers (from their bodies first), renumbering the rest.
    RemoveMarkers { trial: usize, markers: Vec<usize> },
    /// E24.
    SetMarkerCount { trial: usize, count: usize },
    /// E25.
    SetRigidBodyCount { trial: usize, count: usize },
    /// E16: one new body from the markers that are in no body yet.
    CreateRigidBody { trial: usize, markers: Vec<usize> },
    /// E17 and E23: moves markers into `body`, or with `None` out of their bodies.
    MoveMarkersToBody {
        trial: usize,
        markers: Vec<usize>,
        body: Option<usize>,
    },
    /// E15 for a rigid body.
    SetBodyDescription {
        trial: usize,
        body: usize,
        description: String,
    },

    // ----- bodies.rs: one rigid body's references, virtual points and settings ---------
    /// E30: `markers[j]` is the marker for reference point `j`.
    SetBodyCorrespondences {
        trial: usize,
        body: usize,
        markers: Vec<usize>,
    },
    /// E31 "Set Body from File".
    SetBodyReferencesFromFile {
        trial: usize,
        body: usize,
        path: PathBuf,
    },
    /// E32 "Set Body from Frame".
    SetBodyReferencesFromFrame {
        trial: usize,
        body: usize,
        frame: usize,
    },
    /// E33 "Reset To Marker References".
    ResetBodyReferences { trial: usize, body: usize },
    /// E34 "optimized from tracked".
    SetBodyOptimized {
        trial: usize,
        body: usize,
        optimized: bool,
    },
    /// E35 "Update Optimzed".
    UpdateBodyOptimized { trial: usize, body: usize },
    /// E36.
    SetBodyVisible {
        trial: usize,
        body: usize,
        visible: bool,
    },
    /// E37.
    SetBodyColor {
        trial: usize,
        body: usize,
        color: Rgb,
    },
    /// E38: `None` removes the mesh.
    SetBodyMesh {
        trial: usize,
        body: usize,
        path: Option<PathBuf>,
    },
    /// E39.
    SetBodyDrawMesh {
        trial: usize,
        body: usize,
        draw: bool,
    },
    /// E40.
    SetBodyMeshScale {
        trial: usize,
        body: usize,
        scale: f64,
    },
    /// E41: the cutoff override flag and, when on, the body's own cutoff.
    SetBodyCutoffOverride {
        trial: usize,
        body: usize,
        enabled: bool,
        cutoff_frequency: f64,
    },
    /// E42.
    AddVirtualPoint {
        trial: usize,
        body: usize,
        name: String,
        /// The CT coordinate file: the point in this body's frame.
        reference_file: PathBuf,
        input: VirtualPointInput,
    },
    /// E43.
    ClearVirtualPoints { trial: usize, body: usize },
    /// E48 "Update": recompute and filter every body of the trial.
    RecomputeBodies { trial: usize },

    // ----- events.rs --------------------------------------------------------------------
    /// E49.
    AddEvent { trial: usize, name: String },
    /// E50.
    RemoveEvent { trial: usize, event: usize },
    /// E51 (`on`) and E52: marks the frames in every event that is drawn.
    SetEventFrames {
        trial: usize,
        frames: FrameRange,
        on: bool,
    },
    /// E53.
    SetEventColor {
        trial: usize,
        event: usize,
        color: Rgb,
    },
    /// E54.
    SetEventDraw {
        trial: usize,
        event: usize,
        draw: bool,
    },

    // ----- trials.rs ----------------------------------------------------------------------
    /// E57 to E61: the Trial dialog's OK or "Force Update Trial".
    SetTrialProperties {
        trial: usize,
        properties: TrialProperties,
    },
    /// E63: the start frame, 1-based as the model stores it.
    SetStartFrame { trial: usize, frame: usize },
    /// E64: the end frame, 1-based.
    SetEndFrame { trial: usize, frame: usize },
    /// E62 and E65.
    DeleteTrials { trials: Vec<usize> },
    /// E28: removes the link to the Default trial.
    UnlinkFromDefault { trial: usize },
    /// E27 "Apply to trials": copies the Default trial's markers and bodies to every
    /// trial linked to it.
    ApplyDefaultToTrials,

    // ----- import.rs ----------------------------------------------------------------------
    /// E66.
    Import2dPoints {
        trial: usize,
        files: Vec<PathBuf>,
        options: Import2dOptions,
    },
    /// E67 (`update_existing` false: markers are added) and E68 (true: existing markers
    /// are updated in order).
    ImportMarkersCsv {
        trial: usize,
        path: PathBuf,
        update_existing: bool,
    },
    /// E69: markers and bodies of trial `source_trial` (by name) of another project file.
    ImportMarkersFromProject {
        trial: usize,
        path: PathBuf,
        source_trial: String,
    },
    /// E70.
    CopyMarkersFromTrial { trial: usize, source: usize },
    /// E71.
    RenameMarkersFromCsv { trial: usize, path: PathBuf },
    /// E72: trials (by name) of another project file. A Default trial among them replaces
    /// the project's Default trial when `replace_default` is set and is skipped otherwise.
    ImportTrials {
        path: PathBuf,
        trials: Vec<String>,
        replace_default: bool,
    },

    // ----- create.rs --------------------------------------------------------------------
    /// C55 to C58: a new trial from video or image files ([`crate::create`]).
    NewTrial(crate::create::NewTrialSpec),
    /// C59: "Add Default Trial".
    AddDefaultTrial,
    /// C61: "Change Trial Data" of a trial.
    ChangeTrialData {
        trial: usize,
        spec: crate::create::NewTrialSpec,
    },
    /// C60: a new trial from the files of an XMA Portal trial, with its metadata (the
    /// lines of `XMALab_Files-metadata.xml`) and the frame rate it gives
    /// ([`crate::portal`]).
    NewTrialFromPortal {
        spec: crate::create::NewTrialSpec,
        metadata: Vec<String>,
    },
    /// C60 within C61: "Change Trial Data" with the files of an XMA Portal trial.
    ChangeTrialDataFromPortal {
        trial: usize,
        spec: crate::create::NewTrialSpec,
        metadata: Vec<String>,
    },
    /// "Add metadata from XMAPortal file" (Project Metadata): the study metadata of the
    /// project (`trial: None`, XMALab's "Calibration") or of a trial.
    SetPortalMetadata {
        trial: Option<usize>,
        metadata: Vec<String>,
    },

    // ----- calibration.rs ---------------------------------------------------------------
    /// An undistortion or calibration action ([`crate::calibration`]).
    Calibration(crate::calibration::CalibrationEdit),

    // ----- settings.rs ------------------------------------------------------------------
    /// E73 to E79: changes algorithm settings (stored in the project, ledger A10) and
    /// recomputes what depends on them.
    SetSettings(SettingsPatch),
}

impl Edit {
    /// Ledger F51: the first number of the edit that is NaN or infinite, named for a
    /// sentence; such an edit is refused (`EditError::Invalid`) in both modes, as no control
    /// of XMALab gives one. `None` when every number is finite.
    #[must_use]
    pub fn non_finite_input(&self) -> Option<&'static str> {
        let bad = |v: &[f64]| v.iter().any(|x| !x.is_finite());
        match self {
            Edit::SetPoint { x, y, .. } if bad(&[*x, *y]) => Some("a point coordinate"),
            Edit::SetDetectedPoint { x, y, size, .. } if bad(&[*x, *y, *size]) => {
                Some("a point coordinate or size")
            }
            Edit::ResetAboveError { threshold, .. } if bad(&[*threshold]) => {
                Some("the error threshold")
            }
            Edit::SetBodyMeshScale { scale, .. } if bad(&[*scale]) => Some("the mesh scale"),
            Edit::SetBodyCutoffOverride {
                cutoff_frequency, ..
            } if bad(&[*cutoff_frequency]) => Some("the cutoff frequency"),
            Edit::SetTrialProperties { properties, .. } => {
                if properties.recording_speed.is_some_and(|v| !v.is_finite()) {
                    Some("the recording speed")
                } else if properties.cutoff_frequency.is_some_and(|v| !v.is_finite()) {
                    Some("the cutoff frequency")
                } else {
                    None
                }
            }
            Edit::SetSettings(patch) => patch.values.iter().find_map(|(_, v)| match v {
                xma_model::SettingValue::Float(f) if !f.is_finite() => Some("a setting"),
                _ => None,
            }),
            Edit::Calibration(e) => e.non_finite_input(),
            _ => None,
        }
    }

    /// The name of the undo step, as the Edit menu shows it after "Undo ".
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Edit::SetPoint { .. } | Edit::SetDetectedPoint { .. } => "Set Point",
            Edit::ResetFrames {
                untrackable: false, ..
            }
            | Edit::ResetAboveError { .. } => "Delete Data",
            Edit::ResetFrames {
                untrackable: true, ..
            } => "Set Untrackable",
            Edit::SwapMarkerData { .. } => "Exchange Tracked Data",
            Edit::Interpolate { .. } => "Interpolate",
            Edit::SetInterpolation { .. } => "Set Interpolation",
            Edit::SetMarkerDescription { .. } | Edit::SetBodyDescription { .. } => {
                "Change Description"
            }
            Edit::SetMarkerMethod { .. } => "Change Detection Method",
            Edit::SetMarkerProperties { .. } => "Change Marker Settings",
            Edit::AddMarker { .. } => "Add Marker",
            Edit::RemoveMarkers { .. } => "Delete Points",
            Edit::SetMarkerCount { .. } => "Set Number of Markers",
            Edit::SetRigidBodyCount { .. } => "Set Number of Rigid Bodies",
            Edit::CreateRigidBody { .. } => "Create Rigid Body",
            Edit::MoveMarkersToBody { .. } => "Change Rigid Body Markers",
            Edit::SetBodyCorrespondences { .. } => "Change Correspondences",
            Edit::SetBodyReferencesFromFile { .. } => "Set Body from File",
            Edit::SetBodyReferencesFromFrame { .. } => "Set Body from Frame",
            Edit::ResetBodyReferences { .. } => "Reset References",
            Edit::SetBodyOptimized { .. } | Edit::UpdateBodyOptimized { .. } => {
                "Optimize References"
            }
            Edit::SetBodyVisible { .. }
            | Edit::SetBodyColor { .. }
            | Edit::SetBodyMesh { .. }
            | Edit::SetBodyDrawMesh { .. }
            | Edit::SetBodyMeshScale { .. }
            | Edit::SetBodyCutoffOverride { .. } => "Change Rigid Body Settings",
            Edit::AddVirtualPoint { .. } => "Add Virtual Point",
            Edit::ClearVirtualPoints { .. } => "Delete Virtual Points",
            Edit::RecomputeBodies { .. } => "Update Rigid Bodies",
            Edit::AddEvent { .. } => "Add Event",
            Edit::RemoveEvent { .. } => "Delete Event",
            Edit::SetEventFrames { on: true, .. } => "Mark Frames",
            Edit::SetEventFrames { on: false, .. } => "Unmark Frames",
            Edit::SetEventColor { .. } | Edit::SetEventDraw { .. } => "Change Event",
            Edit::SetTrialProperties { .. } => "Change Trial Settings",
            Edit::SetStartFrame { .. } => "Set Start Frame",
            Edit::SetEndFrame { .. } => "Set End Frame",
            Edit::DeleteTrials { .. } => "Delete Trial",
            Edit::UnlinkFromDefault { .. } => "Remove Link to Default Trial",
            Edit::ApplyDefaultToTrials => "Apply to Trials",
            Edit::Import2dPoints { .. } => "Import 2D Points",
            Edit::ImportMarkersCsv { .. }
            | Edit::ImportMarkersFromProject { .. }
            | Edit::CopyMarkersFromTrial { .. } => "Import Markers",
            Edit::RenameMarkersFromCsv { .. } => "Rename Markers",
            Edit::ImportTrials { .. } => "Import Trials",
            Edit::SetSettings(_) => "Change Settings",
            Edit::NewTrial(_) => "New Trial",
            Edit::AddDefaultTrial => "Add Default Trial",
            Edit::ChangeTrialData { .. } => "Change Trial Data",
            Edit::NewTrialFromPortal { .. } => "New Trial",
            Edit::ChangeTrialDataFromPortal { .. } => "Change Trial Data",
            Edit::SetPortalMetadata { .. } => "Add Metadata",
            Edit::Calibration(e) => e.label(),
        }
    }
}

/// What an applied edit reports back.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EditOutcome {
    /// Lines for the Console: what XMALab would write there, and notes on anything the
    /// edit skipped (for example markers already in a body).
    pub messages: Vec<String>,
    /// XMALab's error message when, in the XMALab 3 mode, the edit reproduces an action
    /// that changes the project and then shows an error (for example ledger A76: a
    /// rejected reference file). `None` in the default mode, where such an edit is
    /// [`EditError::Rejected`] and changes nothing.
    pub refused: Option<String>,
}

impl EditOutcome {
    /// An outcome with nothing to report.
    #[must_use]
    pub fn none() -> Self {
        EditOutcome::default()
    }

    /// An outcome with one Console line.
    #[must_use]
    pub fn message(text: impl Into<String>) -> Self {
        EditOutcome {
            messages: vec![text.into()],
            refused: None,
        }
    }

    /// An outcome carrying XMALab's error message after a change ([`EditOutcome::refused`]).
    #[must_use]
    pub fn refused(text: impl Into<String>) -> Self {
        EditOutcome {
            messages: Vec::new(),
            refused: Some(text.into()),
        }
    }
}

/// Why an edit was not applied. The document is unchanged.
#[derive(Clone, Debug, PartialEq)]
pub enum EditError {
    /// An index is out of range, or the edit does not apply to this project (for example
    /// tracking data on the Default trial). The text is for a log, not for the user.
    Invalid(String),
    /// XMALab shows this text in an error dialog and changes nothing; the interface shows
    /// it the same way.
    Rejected(String),
    /// A file could not be read.
    Io { path: PathBuf, message: String },
    /// Not ported yet.
    NotImplemented(&'static str),
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditError::Invalid(text) | EditError::Rejected(text) => f.write_str(text),
            EditError::Io { path, message } => write!(f, "{}: {message}", path.display()),
            EditError::NotImplemented(what) => write!(f, "{what} is not available yet"),
        }
    }
}

impl std::error::Error for EditError {}

/// The result of applying an edit.
pub type EditResult = Result<EditOutcome, EditError>;

impl Document {
    /// Applies one edit as one undo step named [`Edit::label`]. On an error the document
    /// is unchanged.
    ///
    /// # Errors
    /// See [`EditError`].
    ///
    /// The Console lines of the poses the step computed are not in the outcome: they stay
    /// pending for [`Document::take_capped_lines`] (ledger A210); on an error the pending
    /// lines are as they were before the step.
    pub fn apply(&mut self, edit: &Edit) -> EditResult {
        let capped = self.capped_snapshot();
        let result = self.transact(edit.label(), |doc| doc.apply_in_step(edit));
        self.finish_edit_console(capped, result)
    }

    /// Applies several edits as one undo step named `label`. The first error takes back
    /// all of them.
    ///
    /// # Errors
    /// See [`EditError`].
    ///
    /// The Console lines of the poses the step computed are not in the outcome: they stay
    /// pending for [`Document::take_capped_lines`] (ledger A210); on an error the pending
    /// lines are as they were before the step.
    pub fn apply_all(&mut self, label: &str, edits: &[Edit]) -> EditResult {
        let capped = self.capped_snapshot();
        let result = self.transact(label, |doc| {
            let mut outcome = EditOutcome::none();
            for edit in edits {
                outcome.messages.extend(doc.apply_in_step(edit)?.messages);
            }
            Ok(outcome)
        });
        self.finish_edit_console(capped, result)
    }

    /// One edit inside an open step: the dispatch, then the check that no trial gained a
    /// cycle of virtual points ([`ops::refuse_new_cycles`]; a refused edit is taken back by
    /// the step).
    pub(crate) fn apply_in_step(&mut self, edit: &Edit) -> EditResult {
        if let Some(what) = edit.non_finite_input() {
            return Err(EditError::Invalid(format!("{what} is not a finite number")));
        }
        let before = self.project.trials.clone();
        let outcome = self.dispatch(edit)?;
        ops::refuse_new_cycles(&before, &self.project.trials)?;
        Ok(outcome)
    }

    /// The dispatch: one function per variant, in the file its group belongs to.
    fn dispatch(&mut self, edit: &Edit) -> EditResult {
        match edit {
            Edit::SetPoint {
                trial,
                marker,
                camera,
                frame,
                x,
                y,
                status,
            } => points::set_point(self, *trial, *marker, *camera, *frame, *x, *y, *status),
            Edit::SetDetectedPoint {
                trial,
                marker,
                camera,
                frame,
                x,
                y,
                size,
                status,
            } => points::set_detected_point(
                self, *trial, *marker, *camera, *frame, *x, *y, *size, *status,
            ),
            Edit::ResetFrames {
                trial,
                markers,
                cameras,
                frames,
                untrackable,
            } => points::reset_frames(self, *trial, markers, *cameras, *frames, *untrackable),
            Edit::ResetAboveError {
                trial,
                target,
                threshold,
            } => points::reset_above_error(self, *trial, target, *threshold),
            Edit::SwapMarkerData {
                trial,
                marker,
                other,
                cameras,
                frames,
            } => points::swap_marker_data(self, *trial, *marker, *other, *cameras, *frames),
            Edit::Interpolate { trial, markers } => points::interpolate(self, *trial, markers),
            Edit::SetInterpolation {
                trial,
                marker,
                frames,
                method,
            } => points::set_interpolation(self, *trial, *marker, *frames, *method),

            Edit::SetMarkerDescription {
                trial,
                marker,
                description,
            } => markers::set_description(self, *trial, *marker, description),
            Edit::SetMarkerMethod {
                trial,
                markers,
                method,
            } => markers::set_method(self, *trial, markers, *method),
            Edit::SetMarkerProperties {
                trial,
                marker,
                properties,
            } => markers::set_properties(self, *trial, *marker, properties),

            Edit::AddMarker { trial } => structure::add_marker(self, *trial),
            Edit::RemoveMarkers { trial, markers } => {
                structure::remove_markers(self, *trial, markers)
            }
            Edit::SetMarkerCount { trial, count } => {
                structure::set_marker_count(self, *trial, *count)
            }
            Edit::SetRigidBodyCount { trial, count } => {
                structure::set_rigid_body_count(self, *trial, *count)
            }
            Edit::CreateRigidBody { trial, markers } => {
                structure::create_rigid_body(self, *trial, markers)
            }
            Edit::MoveMarkersToBody {
                trial,
                markers,
                body,
            } => structure::move_markers_to_body(self, *trial, markers, *body),
            Edit::SetBodyDescription {
                trial,
                body,
                description,
            } => structure::set_body_description(self, *trial, *body, description),

            Edit::SetBodyCorrespondences {
                trial,
                body,
                markers,
            } => bodies::set_correspondences(self, *trial, *body, markers),
            Edit::SetBodyReferencesFromFile { trial, body, path } => {
                bodies::set_references_from_file(self, *trial, *body, path)
            }
            Edit::SetBodyReferencesFromFrame { trial, body, frame } => {
                bodies::set_references_from_frame(self, *trial, *body, *frame)
            }
            Edit::ResetBodyReferences { trial, body } => {
                bodies::reset_references(self, *trial, *body)
            }
            Edit::SetBodyOptimized {
                trial,
                body,
                optimized,
            } => bodies::set_optimized(self, *trial, *body, *optimized),
            Edit::UpdateBodyOptimized { trial, body } => {
                bodies::update_optimized(self, *trial, *body)
            }
            Edit::SetBodyVisible {
                trial,
                body,
                visible,
            } => bodies::set_visible(self, *trial, *body, *visible),
            Edit::SetBodyColor { trial, body, color } => {
                bodies::set_color(self, *trial, *body, *color)
            }
            Edit::SetBodyMesh { trial, body, path } => {
                bodies::set_mesh(self, *trial, *body, path.as_deref())
            }
            Edit::SetBodyDrawMesh { trial, body, draw } => {
                bodies::set_draw_mesh(self, *trial, *body, *draw)
            }
            Edit::SetBodyMeshScale { trial, body, scale } => {
                bodies::set_mesh_scale(self, *trial, *body, *scale)
            }
            Edit::SetBodyCutoffOverride {
                trial,
                body,
                enabled,
                cutoff_frequency,
            } => bodies::set_cutoff_override(self, *trial, *body, *enabled, *cutoff_frequency),
            Edit::AddVirtualPoint {
                trial,
                body,
                name,
                reference_file,
                input,
            } => bodies::add_virtual_point(self, *trial, *body, name, reference_file, input),
            Edit::ClearVirtualPoints { trial, body } => {
                bodies::clear_virtual_points(self, *trial, *body)
            }
            Edit::RecomputeBodies { trial } => bodies::recompute_bodies(self, *trial),

            Edit::AddEvent { trial, name } => events::add_event(self, *trial, name),
            Edit::RemoveEvent { trial, event } => events::remove_event(self, *trial, *event),
            Edit::SetEventFrames { trial, frames, on } => {
                events::set_event_frames(self, *trial, *frames, *on)
            }
            Edit::SetEventColor {
                trial,
                event,
                color,
            } => events::set_event_color(self, *trial, *event, *color),
            Edit::SetEventDraw { trial, event, draw } => {
                events::set_event_draw(self, *trial, *event, *draw)
            }

            Edit::SetTrialProperties { trial, properties } => {
                trials::set_properties(self, *trial, properties)
            }
            Edit::SetStartFrame { trial, frame } => trials::set_start_frame(self, *trial, *frame),
            Edit::SetEndFrame { trial, frame } => trials::set_end_frame(self, *trial, *frame),
            Edit::DeleteTrials { trials } => trials::delete_trials(self, trials),
            Edit::UnlinkFromDefault { trial } => trials::unlink_from_default(self, *trial),
            Edit::ApplyDefaultToTrials => trials::apply_default_to_trials(self),

            Edit::Import2dPoints {
                trial,
                files,
                options,
            } => import::import_2d_points(self, *trial, files, *options),
            Edit::ImportMarkersCsv {
                trial,
                path,
                update_existing,
            } => import::import_markers_csv(self, *trial, path, *update_existing),
            Edit::ImportMarkersFromProject {
                trial,
                path,
                source_trial,
            } => import::import_markers_from_project(self, *trial, path, source_trial),
            Edit::CopyMarkersFromTrial { trial, source } => {
                import::copy_markers_from_trial(self, *trial, *source)
            }
            Edit::RenameMarkersFromCsv { trial, path } => {
                import::rename_markers_from_csv(self, *trial, path)
            }
            Edit::ImportTrials {
                path,
                trials,
                replace_default,
            } => import::import_trials(self, path, trials, *replace_default),

            Edit::SetSettings(patch) => settings::set_settings(self, patch),
            Edit::NewTrial(spec) => crate::create::new_trial(self, spec),
            Edit::AddDefaultTrial => crate::create::add_default_trial(self),
            Edit::ChangeTrialData { trial, spec } => {
                crate::create::change_trial_data(self, *trial, spec)
            }
            Edit::NewTrialFromPortal { spec, metadata } => {
                crate::portal::new_trial(self, spec, metadata)
            }
            Edit::ChangeTrialDataFromPortal {
                trial,
                spec,
                metadata,
            } => crate::portal::change_trial_data(self, *trial, spec, metadata),
            Edit::SetPortalMetadata { trial, metadata } => {
                crate::portal::set_metadata(self, *trial, metadata)
            }
            Edit::Calibration(e) => crate::calibration::apply(self, e),
        }
    }

    /// Sets the scale and bias of one camera's view (E5, E6; ledger U46). Saved with the
    /// project and marks it modified, but is not an undo step (see [`crate::history`]).
    /// Returns false for a camera index out of range.
    pub fn set_image_adjustment(
        &mut self,
        camera: usize,
        view: AdjustedView,
        scale: f64,
        bias: f64,
    ) -> bool {
        let Some(cam) = self.project.cameras.get_mut(camera) else {
            return false;
        };
        let current = cam.image_adjustments[view as usize];
        if current.scale == scale && current.bias == bias {
            return true;
        }
        let cam = std::sync::Arc::make_mut(cam);
        cam.image_adjustments[view as usize].scale = scale;
        cam.image_adjustments[view as usize].bias = bias;
        self.mark_modified_view_only();
        true
    }
}
