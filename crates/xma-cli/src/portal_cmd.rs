//! `xma import-portal`: XMALab's XMA Portal import without the window.

use std::path::{Path, PathBuf};

use xma_engine::create::{CreateError, NewTrialSpec, NoCalibrationSpec};
use xma_engine::portal::{assign_trial, import_project, import_trial, read_metadata};
use xma_engine::{Document, Edit, OpenOptions, SaveOptions};
use xma_io::portal::Destination;

use crate::args::{Opt, one_based, parse};
use crate::settings::SettingsArgs;
use crate::{CliError, Io};

pub const HELP: &str = "\
xma import-portal: import a file downloaded from the XMA Portal (a .xmatrial trial, or a
project archive), as XMALab's \"Load XMA Portal file\" and Open dataset do

Usage: xma import-portal <portal file> -o <out.xma> [--project <project.xma>]
                         [--trial <n>] [--name <trial name>] [--workspace <dir>]
                         [--metadata-only] [--delete] [--xmalab3-compat]

Without --project: a project archive (one with calibration files) becomes a new dataset,
as Open dataset with such an archive and the New dataset dialog's OK do (one camera per
Portal camera, a camera without a grid a light camera, the archive's one .csv and one .ref
as the cube, else Checkerboard); a trial archive becomes a new dataset without calibration
(File > New trial without calibration: one camera per Portal camera).
With --project: the trial is added to that project (the New trial dialog's \"Load XMA
Portal file\" and Ok), or, with --trial <n> (from 1), replaces trial n's files
(\"Change Trial Data\"). With --metadata-only nothing is extracted: the file's metadata is
set on trial n, or on the project without --trial (\"Add metadata from XMAPortal file\").

A trial is extracted to <workspace>/<StudyName>/<TrialName> with --workspace, else next to
the project file (with --project), else next to the Portal file, as <TrialName>; files there
already are kept. A project archive is extracted to a temporary folder (kept where a trial
would go when a calibration file is a video). The result is written to <out.xma>; the
project given with --project is not changed unless it is also the output.

Options:
  -o, --out <file>        The project file to write
      --project <file>    Add the trial to this project
      --trial <n>         Replace this trial's files (from 1), or its metadata
      --name <text>       The trial's name [default: the Portal TrialName]
      --workspace <dir>   The custom workspace (Settings > General)
      --metadata-only     Only set the metadata (\"Add metadata from XMAPortal file\")
      --delete            Delete the Portal file after a successful import
      --xmalab3-compat    Reproduce XMALab 3 (parity ledger F59, F60) [default: off]
  -h, --help              Print help

Writes to stdout the folder extracted to and each camera's files; messages (files kept,
the frame rate that could not be set) to stderr.

Exit status: 0 success, 1 the import or the creation failed, 2 a usage error.
";

const OPTS: [Opt; 8] = [
    Opt::value("out").short('o'),
    Opt::value("project"),
    Opt::value("trial"),
    Opt::value("name"),
    Opt::value("workspace"),
    Opt::flag("metadata-only"),
    Opt::flag("delete"),
    Opt::flag("xmalab3-compat"),
];

fn failed(e: impl std::fmt::Display) -> CliError {
    CliError::failed(e.to_string())
}

/// Saves the project; the cap lines pending before the save and those of the save are
/// written to stderr (ledger A210), also when the save fails.
fn save(doc: &mut Document, out: &Path, io: &mut Io<'_>) -> Result<(), CliError> {
    crate::with_console_lines(doc, io, |doc, io| {
        doc.save_as(out, &SaveOptions::default())
            .map_err(|e| CliError::failed(format!("{}: {e}", out.display())))?;
        io.line(format_args!("written {}", out.display()));
        Ok(())
    })
}

fn creation(e: CreateError) -> CliError {
    CliError::failed(e.to_string())
}

pub fn run(args: &[String], io: &mut Io<'_>) -> Result<(), CliError> {
    let p = parse(args, &OPTS)?;
    if p.has("help") {
        io.line(HELP);
        return Ok(());
    }
    let portal = PathBuf::from(p.positional(0, "the Portal file")?);
    p.at_most(1)?;
    let out = PathBuf::from(
        p.value("out")
            .ok_or_else(|| CliError::usage("give the output project with -o <file>"))?,
    );
    let compat = crate::compat_mode(&p);
    let options = OpenOptions {
        compat,
        ..OpenOptions::default()
    };
    let trial: Option<usize> = one_based(&p, "trial", "trials")?;
    let project_path = p.value("project").map(PathBuf::from);
    let workspace = p.value("workspace").map(PathBuf::from);
    let dest = Destination {
        workspace: workspace.as_deref(),
        project_file: project_path.as_deref(),
        portal_file: &portal,
    };
    if !portal.is_file() {
        return Err(CliError::failed(format!(
            "{}: no such file",
            portal.display()
        )));
    }

    if p.has("metadata-only") {
        let path = project_path
            .as_deref()
            .ok_or_else(|| CliError::usage("--metadata-only needs --project"))?;
        let mut doc = crate::open_document(path, &SettingsArgs::default(), compat)?;
        crate::with_console_lines(&mut doc, io, |doc, io| {
            let Some(lines) = read_metadata(&portal, compat).map_err(failed)? else {
                return Err(CliError::failed(xma_io::portal::NOT_VALID));
            };
            doc.apply(&Edit::SetPortalMetadata {
                trial,
                metadata: lines,
            })
            .map_err(failed)?;
            save(doc, &out, io)
        })?;
        return finish(&portal, &p, io);
    }

    match project_path.as_deref() {
        None => {
            let archive =
                xma_io::portal::PortalArchive::open_with(&portal, compat.portal_macos_folder_taken)
                    .map_err(failed)?;
            let is_project = !archive.metadata().project_files().is_empty();
            drop(archive);
            if is_project {
                let import =
                    import_project(&portal, &dest, compat, &mut |_| true).map_err(failed)?;
                io.line(format_args!("extracted to {}", import.folder.display()));
                let spec = import.spec();
                for (i, c) in spec.cameras.iter().enumerate() {
                    io.line(format_args!(
                        "camera {}: {} calibration file(s){}",
                        i + 1,
                        c.images.len(),
                        if c.light_camera { ", light camera" } else { "" }
                    ));
                }
                let mut doc = Document::new_project_from_portal(&spec, &import, &options)
                    .map_err(creation)?;
                for w in doc.warnings() {
                    io.note(w);
                }
                save(&mut doc, &out, io)?;
            } else {
                let imported =
                    import_trial(&portal, &dest, compat, &mut |_| true).map_err(failed)?;
                report_import(&imported, io);
                let a = assign_trial(&imported, &[], 1, true);
                let cameras = boxes(&a);
                let spec = NoCalibrationSpec {
                    camera_names: (1..=cameras.len()).map(|i| format!("Camera {i}")).collect(),
                    trial: NewTrialSpec {
                        name: name(&p, &a),
                        cameras,
                    },
                };
                let (mut doc, refused) = Document::new_without_calibration_from_portal(
                    &spec,
                    &imported.metadata,
                    &options,
                )
                .map_err(creation)?;
                if let Some(r) = refused {
                    io.note(r);
                }
                save(&mut doc, &out, io)?;
            }
        }
        Some(path) => {
            let mut doc = crate::open_document(path, &SettingsArgs::default(), compat)?;
            crate::with_console_lines(&mut doc, io, |doc, io| {
                let imported =
                    import_trial(&portal, &dest, compat, &mut |_| true).map_err(failed)?;
                report_import(&imported, io);
                let ids: Vec<i32> = doc.project().cameras.iter().map(|c| c.portal_id).collect();
                let a = assign_trial(&imported, &ids, ids.len(), false);
                for s in &a.skipped {
                    io.note(s);
                }
                let mut cameras = boxes(&a);
                cameras.resize(ids.len(), Vec::new());
                let spec = NewTrialSpec {
                    name: match trial {
                        Some(t) if p.value("name").is_none() && a.trial_name.is_none() => doc
                            .project()
                            .trials
                            .get(t)
                            .map(|t| t.name.clone())
                            .unwrap_or_default(),
                        _ => name(&p, &a),
                    },
                    cameras,
                };
                let edit = match trial {
                    None => Edit::NewTrialFromPortal {
                        spec,
                        metadata: imported.metadata.clone(),
                    },
                    Some(t) => Edit::ChangeTrialDataFromPortal {
                        trial: t,
                        spec,
                        metadata: imported.metadata.clone(),
                    },
                };
                let outcome = doc.apply(&edit).map_err(failed)?;
                for m in &outcome.messages {
                    io.note(m);
                }
                if let Some(r) = outcome.refused {
                    io.note(r);
                }
                save(doc, &out, io)
            })?;
        }
    }
    finish(&portal, &p, io)
}

/// The dialog's trial name: `--name`, else the Portal `TrialName` when the trial has a file,
/// else XMALab's default "Trial 1".
fn name(p: &crate::args::Parsed, a: &xma_engine::portal::TrialAssignment) -> String {
    p.value("name")
        .map(str::to_owned)
        .or_else(|| a.trial_name.clone())
        .unwrap_or_else(|| "Trial 1".to_owned())
}

fn boxes(a: &xma_engine::portal::TrialAssignment) -> Vec<Vec<PathBuf>> {
    let mut cams = vec![Vec::new(); a.boxes];
    for (b, files, _) in &a.cameras {
        cams[*b].clone_from(files);
    }
    cams
}

fn report_import(i: &xma_engine::portal::TrialImport, io: &mut Io<'_>) {
    io.line(format_args!("extracted to {}", i.folder.display()));
    for f in &i.files {
        io.line(format_args!(
            "cam{}: {} file(s) from {}",
            f.number,
            f.files.len(),
            f.path.display()
        ));
    }
    for m in &i.messages {
        io.note(m);
    }
}

fn finish(portal: &Path, p: &crate::args::Parsed, io: &mut Io<'_>) -> Result<(), CliError> {
    if p.has("delete") {
        std::fs::remove_file(portal)
            .map_err(|e| CliError::failed(format!("{}: {e}", portal.display())))?;
        io.line(format_args!("deleted {}", portal.display()));
    }
    Ok(())
}
