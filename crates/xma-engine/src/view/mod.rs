//! Read-only queries the interface draws from: what a camera view overlays on a
//! frame, the series a plot shows, the rows of the points tree, what the 3D world view and
//! the rigid-body mesh overlay draw ([`world`]).
//!
//! Everything here is computed from a [`crate::Document`] without changing it,
//! so the same queries can be tested without a window and reused by the
//! command-line program. The C++ computes these inside its widgets
//! (`GLCameraView`, `PlotWindow`, `MarkerTreeWidget`).

pub mod calibration;
pub mod overlays;
pub mod plots;
pub mod tree;
pub mod world;
