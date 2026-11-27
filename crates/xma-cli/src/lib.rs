//! The headless RustXMALab program `xma`, as a library so that tests can run its commands
//! in-process.
//!
//! Dependency rule: depends on `xma-engine`, `xma-io`, `xma-model` and `xma-oracle` (and on
//! `xma-vision` and `xma-video` for the detection tables of `parity`). Never on a GUI
//! toolkit.
//!
//! Commands: `info`, `recompute`, `export`, `edit`, `calibrate`, `track`, `parity`,
//! `undistort`, `import-portal`, `denoise` (see
//! [`USAGE`] and each
//! command's
//! `--help`). The comparison behind `parity` is the library module [`parity`].
//!
//! Exit status: 0 success; 1 an error (a file that cannot be read or written, an export the
//! GUI would refuse); 2 a usage error; 3 `parity` found a difference.

mod args;
pub mod calibrate_cmd;
pub mod edit_cmd;
mod export_cmd;
mod images_cmd;
mod info;
pub mod parity;
mod portal_cmd;
mod recompute;
mod settings;
mod track;
mod undistort_cmd;

use std::fmt;
use std::io::Write;

pub use settings::manifest_settings;

/// The program's name.
pub const NAME: &str = "xma";
/// The program's version (the workspace version).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The program's help text.
pub const USAGE: &str = "\
xma: headless RustXMALab (reading, recomputing, editing, tracking, saving and exporting projects)

Usage: xma <command> [arguments] [options]

Commands:
  info       What a project contains
  recompute  Open a project, recompute it as XMALab does, report changes, optionally save
  export     Write one of XMALab's exports
  edit       Apply an edit script to a project
  calibrate  Build and calibrate a project from raw files with a calibration script
  track      Track markers as XMALab's track buttons do, optionally save
  undistort  Write undistorted copies of image folders and videos (File > Undistort sequence)
  import-portal  Import a file downloaded from the XMA Portal (a trial or a project archive)
  denoise    Denoise a trial's images (the Trial dialog's \"Denoise Images\")
  parity     Compare this program with the C++ reference dumps and write a report

Options:
  -h, --help     Print help (also `xma <command> --help`)
  -V, --version  Print version

Exit status: 0 success, 1 error, 2 usage error, 3 parity found a difference.";

/// An error that ends a command: a message for stderr and an exit status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliError {
    pub message: String,
    /// 1 for an error, 2 for a usage error, 3 for a parity difference.
    pub code: u8,
}

impl CliError {
    /// A usage error (status 2).
    pub fn usage(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: 2,
        }
    }

    /// An error while running (status 1).
    pub fn failed(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: 1,
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

/// Where a command writes: its normal output and its messages.
pub struct Io<'a> {
    pub out: &'a mut dyn Write,
    pub err: &'a mut dyn Write,
}

impl Io<'_> {
    /// Writes a line of normal output; a closed pipe is not an error worth a panic.
    pub(crate) fn line(&mut self, text: impl fmt::Display) {
        let _ = writeln!(self.out, "{text}");
    }

    /// Writes a line to stderr.
    pub(crate) fn note(&mut self, text: impl fmt::Display) {
        let _ = writeln!(self.err, "{text}");
    }
}

/// Calls `f` with an [`Io`] on the process's stdout and stderr, as `main` does. It holds
/// the handles, not their locks: a `Stdout` or `Stderr` locks for each write and lets it go.
/// Holding a lock for the whole run would block any other thread that writes there (a
/// worker thread's panic message on stderr, or an `eprintln!`) while the main thread waits
/// for that thread: the program would hang instead of ending with the message. Messages
/// are written in the same order as before (stderr is unbuffered; stdout is line-buffered
/// as before).
pub fn with_std_io<R>(f: impl FnOnce(&mut Io<'_>) -> R) -> R {
    let mut out = std::io::stdout();
    let mut err = std::io::stderr();
    let mut io = Io {
        out: &mut out,
        err: &mut err,
    };
    f(&mut io)
}

/// Runs the program with `args` (without the program name). Returns the exit status;
/// errors are written to `io.err`.
pub fn run(args: &[String], io: &mut Io<'_>) -> u8 {
    CAP_LINES_PRINTED.with(|m| m.borrow_mut().clear());
    let result = dispatch(args, io);
    match result {
        Ok(()) => 0,
        Err(e) => {
            let hint = if e.code == 2 {
                match args.first().map(String::as_str) {
                    Some(
                        c @ ("info" | "recompute" | "export" | "edit" | "calibrate" | "track"
                        | "parity" | "undistort" | "import-portal" | "denoise"),
                    ) => {
                        format!("\n(see `{NAME} {c} --help`)")
                    }
                    _ => format!("\n(see `{NAME} --help`)"),
                }
            } else {
                String::new()
            };
            io.note(format_args!("{NAME}: {}{hint}", e.message));
            e.code
        }
    }
}

fn dispatch(args: &[String], io: &mut Io<'_>) -> Result<(), CliError> {
    let Some(first) = args.first() else {
        io.line(USAGE);
        return Ok(());
    };
    let rest = &args[1..];
    match first.as_str() {
        "-h" | "--help" | "help" => {
            match rest.first().map(String::as_str) {
                None => io.line(USAGE),
                Some("info") => io.line(info::HELP),
                Some("recompute") => io.line(recompute::HELP),
                Some("export") => io.line(export_cmd::HELP),
                Some("edit") => io.line(edit_cmd::HELP),
                Some("calibrate") => io.line(calibrate_cmd::HELP),
                Some("track") => io.line(track::HELP),
                Some("parity") => io.line(parity::HELP),
                Some("undistort") => io.line(undistort_cmd::HELP),
                Some("import-portal") => io.line(portal_cmd::HELP),
                Some("denoise") => io.line(images_cmd::HELP),
                Some(other) => return Err(CliError::usage(format!("unknown command '{other}'"))),
            }
            Ok(())
        }
        "-V" | "--version" => {
            io.line(format_args!("{NAME} {VERSION}"));
            Ok(())
        }
        "info" => info::run(rest, io),
        "recompute" => recompute::run(rest, io),
        "export" => export_cmd::run(rest, io),
        "edit" => edit_cmd::run(rest, io),
        "calibrate" => calibrate_cmd::run(rest, io),
        "track" => track::run(rest, io),
        "parity" => parity::run(rest, io),
        "undistort" => undistort_cmd::run(rest, io),
        "import-portal" => portal_cmd::run(rest, io),
        "denoise" => images_cmd::run(rest, io),
        other if other.starts_with('-') => {
            Err(CliError::usage(format!("unknown option '{other}'")))
        }
        other => Err(CliError::usage(format!("unknown command '{other}'"))),
    }
}

/// The compatibility mode `--xmalab3-compat` selects: [`xma_model::CppCompat::XMALAB3`],
/// else the default (every known fault fixed).
pub(crate) fn compat_mode(p: &args::Parsed) -> xma_model::CppCompat {
    if p.has("xmalab3-compat") {
        xma_model::CppCompat::XMALAB3
    } else {
        xma_model::CppCompat::default()
    }
}

/// Reads a project file with the given settings in the compatibility mode `compat`, without
/// the load-time update. A missing file or one that is not a project gives a message naming
/// the file.
pub(crate) fn read_loaded(
    path: &std::path::Path,
    settings: &settings::SettingsArgs,
    compat: xma_model::CppCompat,
) -> Result<xma_io::LoadedProject, CliError> {
    if !path.exists() {
        return Err(CliError::failed(format!(
            "{}: no such file",
            path.display()
        )));
    }
    xma_engine::read_project(path, &settings.open_options(compat))
        .map_err(|e| CliError::failed(format!("{}: {e}", path.display())))
}

thread_local! {
    /// Per (trial index, rigid body index), the last cap line [`console_lines`] printed in
    /// the running command (emptied by [`run`]).
    static CAP_LINES_PRINTED: std::cell::RefCell<std::collections::HashMap<(usize, usize), String>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Writes to stderr the Console lines the document's pose computations left pending
/// (ledger A210, [`xma_engine::Document::take_capped_lines`]; every computation, an edit's
/// included, leaves them there and nowhere else), as `warning: <line>`, and forgets them.
/// Never on standard output, which scripts parse. A line identical to the last one printed
/// for the same trial and body in the same command is not printed again (the window's rule:
/// `recompute --all` recomputes the trial the load-time update computed). Commands call it
/// right after the project is opened and updated (so the load-time lines come first), after
/// their work, and on the way out of an error ([`with_console_lines`]). Nothing in the
/// XMALab 3 mode, which has no cap.
pub(crate) fn console_lines(doc: &mut xma_engine::Document, io: &mut Io<'_>) {
    for l in doc.take_capped_lines() {
        let repeat = CAP_LINES_PRINTED.with(|m| {
            let mut m = m.borrow_mut();
            let key = (l.trial, l.body);
            if m.get(&key) == Some(&l.text) {
                true
            } else {
                m.insert(key, l.text.clone());
                false
            }
        });
        if !repeat {
            io.note(format_args!("warning: {}", l.text));
        }
    }
}

/// Runs a command's work on an open document between two [`console_lines`]: the lines
/// pending from opening first, then those of the work, whether the work succeeded or ended
/// with an error (a save to a folder that does not exist), so that a late failure does not
/// lose the warning.
pub(crate) fn with_console_lines<T>(
    doc: &mut xma_engine::Document,
    io: &mut Io<'_>,
    work: impl FnOnce(&mut xma_engine::Document, &mut Io<'_>) -> Result<T, CliError>,
) -> Result<T, CliError> {
    console_lines(doc, io);
    let result = work(doc, io);
    console_lines(doc, io);
    result
}

/// Opens a project as XMALab does, with the given settings applied before the load-time
/// update, in the compatibility mode `compat` (ledger A3, A4, F2, F18). A file that is not a
/// project gives the reader's message.
pub(crate) fn open_document(
    path: &std::path::Path,
    settings: &settings::SettingsArgs,
    compat: xma_model::CppCompat,
) -> Result<xma_engine::Document, CliError> {
    let loaded = read_loaded(path, settings, compat)?;
    let mut doc = xma_engine::Document::from_loaded(loaded);
    doc.set_path(path);
    doc.update_after_load();
    Ok(doc)
}
