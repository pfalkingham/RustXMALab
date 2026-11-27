//! `xma`: the headless RustXMALab program. The commands live in the library (`xma_cli::run`).
//!
//! Dependency rule: depends on `xma-engine`, `xma-io`, `xma-model` and `xma-oracle`. Never
//! on a GUI toolkit.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ExitCode::from(xma_cli::with_std_io(|io| xma_cli::run(&args, io)))
}
