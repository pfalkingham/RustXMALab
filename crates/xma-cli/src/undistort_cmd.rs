//! `xma undistort`: XMALab's File > Undistort sequence.

use std::path::{Path, PathBuf};

use xma_engine::undistort_sequence::{UndistortInput, UndistortItem, UndistortJob};

use crate::args::{Opt, number, parse};
use crate::settings::SettingsArgs;
use crate::{CliError, Io};

pub const HELP: &str = "\
xma undistort: write undistorted copies of image folders and videos (File > Undistort sequence)

Usage: xma undistort <project.xma> --camera <n> (--folder <dir> | --video <file>) -o <dir>
                     [--camera <n>] (--folder <dir> | --video <file>) [-o <dir>] ...

Each --folder or --video adds an item, processed in the order given. --camera <n> (from 1)
applies to the items after it; -o <dir> to the items before it that have none yet, and items
after the last -o use it. The camera needs a computed undistortion grid. Each frame is
remapped with the camera's grid and, when the camera has a lens model, with the lens model,
and written as <dir>/<base>_UND.<frame, from 0001>.tif, where <base> is the video's name
without its extension, or the start all of a folder's image names share (for
img_0001.tif .. img_0100.tif: img_0). A folder's images (*.png *.tif *.tiff *.bmp *.jpeg
*.jpg) are taken in natural order. Existing files are replaced; output folders are created
if absent.

By default an image deeper than 8 bits is written at its own depth (a 16-bit TIFF) and a
flipped camera's frames are mirrored for the remap; with --xmalab3-compat both are as
XMALab 3 does them (8 bits; not mirrored), *.tiff files are not taken and a video frame
that cannot be read is written as the previous frame again (parity ledger V20 to V23).
Frames whose size differs from the camera's, or that cannot be read or written, are
skipped and listed on stderr (XMALab skips them silently).

Options:
      --camera <n>        Camera of the items that follow (from 1)
      --folder <dir>      An item: every image of a folder
      --video <file>      An item: a video file (.cine, .avi, .mp4, .mov)
  -o, --out <dir>         Output folder of the items before it that have none
      --xmalab3-compat    Write what XMALab 3 writes (see above) [default: off]
  -h, --help              Print help

Exit status: 0 when every item was processed (some frames may have been skipped), 1 when an
item could not be processed or the project cannot be read, 2 for a usage error.
";

const OPTS: [Opt; 5] = [
    Opt::value("camera"),
    Opt::value("folder"),
    Opt::value("video"),
    Opt::value("out").short('o'),
    Opt::flag("xmalab3-compat"),
];

/// One item as given: camera (1-based) and output, possibly still unset.
struct Given {
    camera: Option<usize>,
    input: UndistortInput,
    output: Option<PathBuf>,
}

/// The items in the order given (see [`HELP`]).
fn items(args: &[String]) -> Result<Vec<UndistortItem>, CliError> {
    let mut given: Vec<Given> = Vec::new();
    let mut camera: Option<usize> = None;
    let mut last_out: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        i += 1;
        if a == "--" {
            break;
        }
        let (name, inline) = match a.strip_prefix("--") {
            Some(l) => match l.split_once('=') {
                Some((n, v)) => (n.to_owned(), Some(v.to_owned())),
                None => (l.to_owned(), None),
            },
            None => match a.strip_prefix("-o") {
                Some(rest) => (
                    "out".to_owned(),
                    (!rest.is_empty()).then(|| rest.to_owned()),
                ),
                _ => continue,
            },
        };
        if !["camera", "folder", "video", "out"].contains(&name.as_str()) {
            continue;
        }
        let value = match inline {
            Some(v) => v,
            None => {
                let Some(v) = args.get(i) else {
                    return Err(CliError::usage(format!("option '--{name}' needs a value")));
                };
                i += 1;
                v.clone()
            }
        };
        match name.as_str() {
            "camera" => {
                let n: usize = number(&value, "--camera")?;
                if n == 0 {
                    return Err(CliError::usage("--camera: cameras are numbered from 1"));
                }
                camera = Some(n);
            }
            "folder" | "video" => given.push(Given {
                camera,
                input: if name == "folder" {
                    UndistortInput::Folder(PathBuf::from(value))
                } else {
                    UndistortInput::Video(PathBuf::from(value))
                },
                output: None,
            }),
            _ => {
                let out = PathBuf::from(value);
                for g in given.iter_mut().filter(|g| g.output.is_none()) {
                    g.output = Some(out.clone());
                }
                last_out = Some(out);
            }
        }
    }
    if given.is_empty() {
        return Err(CliError::usage(
            "no item: give --folder <dir> or --video <file>",
        ));
    }
    given
        .into_iter()
        .map(|g| {
            let what = g.input.path().display().to_string();
            Ok(UndistortItem {
                camera: g
                    .camera
                    .ok_or_else(|| CliError::usage(format!("{what}: no --camera before it")))?
                    - 1,
                output: g
                    .output
                    .or_else(|| last_out.clone())
                    .ok_or_else(|| CliError::usage(format!("{what}: no -o <dir>")))?,
                input: g.input,
            })
        })
        .collect()
}

pub fn run(args: &[String], io: &mut Io<'_>) -> Result<(), CliError> {
    let p = parse(args, &OPTS)?;
    if p.has("help") {
        io.line(HELP);
        return Ok(());
    }
    let path = p.positional(0, "the project file")?;
    p.at_most(1)?;
    let items = items(args)?;
    for it in &items {
        let input = it.input.path();
        let ok = match &it.input {
            UndistortInput::Folder(_) => input.is_dir(),
            UndistortInput::Video(_) => input.is_file(),
        };
        if !ok {
            return Err(CliError::failed(format!("{}: not found", input.display())));
        }
    }
    let mut doc = crate::open_document(
        Path::new(path),
        &SettingsArgs::default(),
        crate::compat_mode(&p),
    )?;
    crate::with_console_lines(&mut doc, io, |doc, io| {
        let n_cameras = doc.project().cameras.len();
        for it in &items {
            if it.camera >= n_cameras {
                return Err(CliError::failed(format!(
                    "--camera {}: the project has {n_cameras} cameras",
                    it.camera + 1
                )));
            }
            std::fs::create_dir_all(&it.output)
                .map_err(|e| CliError::failed(format!("{}: {e}", it.output.display())))?;
        }
        let job = UndistortJob::new(doc, items).map_err(CliError::failed)?;
        let report = job.run(&mut |_| {}, &|| false);
        let mut failed = false;
        for (it, r) in job.items().iter().zip(&report.items) {
            match &r.error {
                Some(e) => {
                    failed = true;
                    io.line(format_args!("{}: not processed: {e}", it.list_text()));
                }
                None => io.line(format_args!(
                    "{}: {} of {} frames written, {} skipped",
                    it.list_text(),
                    r.written.len(),
                    r.frames,
                    r.skipped.len()
                )),
            }
        }
        for line in report.problems(job.items()) {
            io.note(line);
        }
        drop(job);
        if failed {
            return Err(CliError::failed("some items could not be processed"));
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_owned()).collect()
    }

    #[test]
    fn items_in_order() {
        let it = items(&s(&[
            "p.xma",
            "--camera",
            "1",
            "--folder",
            "a",
            "-o",
            "x",
            "--camera=2",
            "--video",
            "b",
            "--out=y",
            "--folder",
            "c",
        ]))
        .unwrap();
        assert_eq!(it.len(), 3);
        assert_eq!((it[0].camera, it[0].output.as_path()), (0, Path::new("x")));
        assert_eq!((it[1].camera, it[1].output.as_path()), (1, Path::new("y")));
        assert_eq!((it[2].camera, it[2].output.as_path()), (1, Path::new("y")));
        assert_eq!(it[1].input, UndistortInput::Video(PathBuf::from("b")));
        let one = items(&s(&[
            "p", "-ox", "--camera", "2", "--folder", "a", "--folder", "b",
        ]))
        .unwrap();
        assert!(
            one.iter()
                .all(|i| i.output == Path::new("x") && i.camera == 1)
        );
        assert!(items(&s(&["p", "--folder", "a", "-o", "x"])).is_err());
        assert!(items(&s(&["p", "--camera", "1", "--folder", "a"])).is_err());
        assert!(items(&s(&["p", "--camera", "0", "--folder", "a", "-o", "x"])).is_err());
        assert!(items(&s(&["p"])).is_err());
    }
}
