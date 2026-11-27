//! The engine: the document, undo and redo, background jobs and recompute.
//! The only crate that joins the model to the algorithms; the CLI and the GUI
//! both drive it.
//!
//! Dependency rule: depends on `xma-model`, `xma-vision`, `xma-video` and
//! `xma-io`. Never on a GUI toolkit.
//!
//! # What is here
//!
//! - [`read_project`] and [`OpenOptions`]: the one way a project file is read (the C++ mesh
//!   test, the settings that apply where the project stores none, and overrides).
//! - [`Document`]: an open project, with the after-load steps of the C++ (including the
//!   poses computed while loading, ledger A54) and the trial update
//!   (`ThreadScheduler::updateTrialData` and its finalize) for markers.
//! - [`markers`]: the per-frame marker step (`Marker::reconstruct3DPoint`,
//!   `Marker::reprojectPoint`, `Marker::updateError`) and `Marker::updateMeanSize`.
//! - [`rigid_bodies`]: the per-frame rigid-body step (`RigidBody::computePose`,
//!   `transformPoint`, `updateError`), the finalize (`makeRotationsContinous`,
//!   `filterTransformations` in both modes, `Marker::filterMarker`) and
//!   `recomputeTransformations`.
//! - [`save`]: "Save" and "Save As" ([`Document::save`], [`Document::save_as`]).
//! - [`export`]: the File > Export actions, with what each `MainWindow` handler does before
//!   writing ([`Document::export`]).

pub mod blender_scene;
pub mod calibration;
pub mod create;
pub mod denoise;
pub mod document;
pub mod edit;
pub mod export;
pub mod history;
mod load_sizes;
pub mod markers;
pub mod portal;
pub mod portal_summary;
pub mod portal_view;
pub mod rigid_bodies;
pub mod save;
pub mod session;
pub mod tracking;
pub mod trial_images;
pub mod undistort_sequence;
pub mod undistorted_image;
pub mod view;

pub use calibration::{CalibrationEdit, ExternalCalibration, GridPointList};
pub use create::{
    CalibrationTarget, CreateError, NewCameraSpec, NewProjectSpec, NewTrialSpec, NoCalibrationSpec,
};
pub use document::{
    DerivedCamera, Document, LoadedExtras, OpenError, OpenOptions, Scheduler, UpdateHook,
    UpdateHookFn, read_project,
};
pub use edit::{Edit, EditError, EditOutcome, EditResult};
pub use export::Export;
pub use markers::{FrameCamera, FrameContext, MarkerFramePatch};
pub use rigid_bodies::{BodyFrameState, FrameStates, PoseContext};
pub use save::{SaveError, SaveOptions, SaveReport, SaveTime};
pub use session::{ReferenceClick, Session, WizardSettings, Workspace};
pub use tracking::images::{adjust_path_to_os, resolve_stored_path, still_bytes};

/// The marker-to-marker standard deviations shared by the precision report and the Project
/// Overview dialog (ledger A58), from `xma_io::export::precision`.
pub mod precision {
    pub use xma_io::export::precision::{body_marker_to_marker_sd, trial_marker_to_marker_sd};
}

/// Name of this crate, used to check the workspace builds and links.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// Version of the engine (the workspace version).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Names of the workspace crates the engine is built from.
pub fn component_crates() -> [&'static str; 4] {
    [
        xma_model::CRATE_NAME,
        xma_vision::CRATE_NAME,
        xma_video::CRATE_NAME,
        xma_io::CRATE_NAME,
    ]
}

#[cfg(test)]
mod tests {
    #[test]
    fn links_components() {
        assert_eq!(
            super::component_crates(),
            ["xma-model", "xma-vision", "xma-video", "xma-io"]
        );
    }
}
