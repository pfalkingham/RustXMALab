//! Locates and reads the reference outputs ("oracle") that the Rust code is tested
//! against, and compares numbers with them.
//!
//! Used by tests in `xma-vision`, `xma-io` and `xma-engine` (as a dev-dependency), and by
//! `xma parity`. Depends on no other workspace crate. The formats are described in
//! `oracle/FORMAT.md`, `oracle/cpp/DEVIATIONS.md` and `oracle/DATASETS.md`.
//!
//! # Where things are
//!
//! - [`oracle_root`]: `$XMA_ORACLE_ROOT`, else `<workspace>/oracle`.
//! - [`dataset`]: a reference dataset from `oracle/datasets.json` (project file and C++
//!   dump). Returns `None`, after one line on stderr, when the data is absent (it is not in
//!   git); with `XMA_REQUIRE_ORACLE=1` it panics instead. An unknown name always panics.
//! - [`primitives_root`], [`cases`], [`case`]: the OpenCV primitive cases in
//!   `oracle/reference/primitives/` (these are in git).
//!
//! # Reading
//!
//! - [`Table`]: a dump table (`.csv` with a header). Per row: [`Row::f64`], [`Row::i64`],
//!   [`Row::usize`], [`Row::str`], [`Row::bool`]; per column: [`Table::column_f64`] and
//!   friends; lookup by key columns: [`Table::find`]. Get one with [`Dataset::table`].
//! - [`Cells`]: any comma-separated text without a header assumption, for the files in
//!   XMALab's own export formats under `exports/` ([`Dataset::cells`]).
//! - [`Manifest`]: a dump's `manifest.json` ([`Dataset::manifest`]): settings by name
//!   ([`Manifest::int`], [`Manifest::float`], [`Manifest::bool`]) and the `missing` list.
//! - [`Case`]: one primitive case: [`Case::matrix`] (CSV array as row-major `f64`),
//!   [`Case::table`] (by column name), [`Case::scalar`] (`name,value` files),
//!   [`Case::image`] (`.u8`/`.u16`/`.f32`), [`Case::arg`] (from `case.json`).
//! - [`parse_f64`]: the number parser used throughout; accepts what `printf("%.17g")`
//!   writes, including `nan`, `-nan`, `inf`, `-inf`, and XMALab's `NaN`.
//!
//! Every reader has a `try_*` form returning [`Result`] (for the CLI) and a plain form that
//! panics with the same one-line message (for tests). Errors ([`OracleError`]) name the
//! file and, where they apply, the row and the column.
//!
//! # Comparing
//!
//! [`Comparison`] collects `(label, got, want)` and answers: count, largest absolute and
//! relative difference and where, and pass/fail against a [`Tolerance`] (absolute,
//! relative, or either). Two NaNs are equal; a NaN against a number fails any tolerance.
//! Its `Display` is a one-line summary for parity reports; failures name the worst value
//! and print both numbers with 17 significant digits.
//!
//! # Example
//!
//! ```no_run
//! use xma_oracle::{Comparison, Tolerance};
//!
//! // A test against the C++ dump. Skips (returns) when the data is not present.
//! let Some(ds) = xma_oracle::dataset("pig_clean") else { return };
//! let manifest = ds.manifest();
//! assert_eq!(manifest.int("TriangulationMethod"), 1);
//!
//! let points = ds.table("trial_0/points2d.csv");
//! let mut cmp = Comparison::new("points2d x");
//! for row in points.rows() {
//!     if row.i64("status") <= 0 {
//!         continue;
//!     }
//!     let (m, c, f) = (row.usize("marker"), row.usize("camera"), row.usize("frame"));
//!     let got = row.f64("x"); // ...the Rust computation would go here
//!     cmp.add(format_args!("marker {m} camera {c} frame {f}"), got, row.f64("x"));
//! }
//! cmp.assert(Tolerance::Either { abs: 1e-9, rel: 1e-12 });
//! println!("{cmp}");
//!
//! // An export file in XMALab's own format (no table header rules).
//! let exported = ds.cells("trial_0/exports/3DPoints.csv");
//! let first_row: Vec<f64> = exported.line_f64(1);
//! # let _ = first_row;
//!
//! // A primitive OpenCV case.
//! for case in xma_oracle::cases("rodrigues") {
//!     if case.inputs().contains(&"rvec") {
//!         let rvec = case.matrix("rvec"); // 1×3
//!         let r = case.matrix("R"); // 3×3, row-major in r.data
//!         # let _ = (rvec, r);
//!     }
//! }
//! let solved = xma_oracle::case("solve-svd-overdet_6x3");
//! assert_eq!(solved.scalar("status", "ok"), 1.0);
//! let image = xma_oracle::case("gaussianblur-u8-3x3s1.3-binary31").image("src");
//! assert_eq!((image.rows, image.cols, image.u8().len()), (31, 31, 961));
//! ```

mod cells;
mod compare;
mod error;
mod locate;
mod manifest;
mod parse;
mod primitives;
mod table;

pub use cells::Cells;
pub use compare::{Comparison, Tolerance, Worst, diff};
pub use error::{OracleError, Result};
pub use locate::{
    Dataset, ORACLE_ROOT_ENV, REQUIRE_ORACLE_ENV, dataset, dataset_names, datasets_file,
    oracle_required, oracle_root, try_dataset_entry, try_dataset_names, workspace_root,
};
pub use manifest::{Manifest, Missing, Setting};
pub use parse::{parse_f64, parse_i64, parse_usize};
pub use primitives::{
    Case, Image, ImageData, Matrix, case, cases, cases_in, primitives_root, routine_key, try_case,
    try_case_names, try_case_names_in, try_cases, try_primitive_index,
};
/// The JSON value type of [`Manifest::json`].
pub use serde_json::Value as JsonValue;
pub use table::{Row, Table};
