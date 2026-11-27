//! Project files (`.xma` read and write), CSV and MayaCam import and export,
//! and settings.
//!
//! Dependency rule: among workspace crates, depends on `xma-model` and on `xma-vision`
//! (plain numeric functions only: rotation conversion on load and save, filtering and
//! lookup tables for exports). Never on the engine or a GUI toolkit.

pub mod calibration_object;
pub mod cfmt;
pub mod checkerboard_axis;
pub mod cparse;
pub mod export;
pub mod finite;
pub mod image_size;
pub mod obj;
pub mod portal;
pub mod project_read;
pub mod project_write;
pub mod settings_entry;
pub mod xml;
pub mod zip;

pub use project_read::{
    Attachment, FileVersion, LoadedProject, ReadError, ReadErrorKind, ReadOptions, TrialSource,
    VideoInfo, read_project, read_project_bytes, read_project_with,
};
pub use project_write::{
    BorrowedEntries, ExtraFiles, Extras, ProjectEntries, WriteError, WriteOptions, project_entries,
    project_entries_borrowed, write_project, write_project_bytes,
};

/// Name of this crate, used to check the workspace builds and links.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// Name of the model crate this crate builds on.
pub fn model_crate() -> &'static str {
    xma_model::CRATE_NAME
}

#[cfg(test)]
mod tests {
    #[test]
    fn links_model() {
        assert_eq!(super::CRATE_NAME, "xma-io");
        assert_eq!(super::model_crate(), "xma-model");
    }
}
