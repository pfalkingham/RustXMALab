//! `xma export`: XMALab's File > Export actions.

use std::path::Path;

use xma_engine::export::Export;
use xma_io::export::FrameRange;
use xma_io::export::points2d::Points2dOptions;
use xma_io::export::points3d::Points3dOptions;
use xma_io::export::rigid_bodies::TransformationsOptions;

use crate::args::{Opt, Parsed, indices, number, parse, range};
use crate::settings::{self, SettingsArgs};
use crate::{CliError, Io};

pub const HELP: &str = "\
xma export: write one of XMALab's exports

Usage: xma export <project.xma> --out <dir> [--trial <n>] <what> [options]

The project is opened as XMALab opens it (the active trial is updated); --trial selects
another trial first, which updates it as XMALab's trial selector does. Options default to
XMALab's export dialog defaults. Exports that XMALab writes to one file chosen by the user
are written to <dir>/<name> (the names below; --name changes it); exports that XMALab writes
into a folder get the file names XMALab gives them. Frames are numbered from 1.

What:
  points2d      2D points (Export 2D Points) [2DPoints.csv; undistorted: 2DPoints_undistorted.csv]
                  --per-marker        one file per marker (default: one file)
                  --undistorted       undistorted coordinates (default: distorted)
                  --count-from-1      add 1 to x and y (default: count from 0)
                  --y-up              y = height - y - 1 (default: y down)
                  --no-header         no header row
                  --offset-columns    offset columns (one `cam<n>_offset` header field per
                                      camera; XMALab 3's header, with --xmalab3-compat,
                                      has no separators and counts from 0, ledger F2)
  points3d      3D points (Export 3D Points) [3DPoints.csv; filtered: 3DPoints_filtered.csv]
                  --per-marker        one file per marker
                  --no-header         no header row
                  --markers <list>    markers, comma-separated (default: all)
                  --filter            low-pass filter at the trial's cutoff (default: unfiltered)
                  --cutoff <hz>       filter at this cutoff instead
                  --frames <a-b>      frame column and only these frames
  rigid-bodies  Rigid-body transformations [RigidBodyTransformations_filtered.csv;
                unfiltered: RigidBodyTransformations.csv]. Each body is recomputed first,
                as XMALab does.
                  --per-body          one file per body
                  --no-header         no header row
                  --bodies <list>     bodies, comma-separated (default: all)
                  --unfiltered        unfiltered poses (default: filtered, which needs the
                                      trial's recording speed and cutoff)
                  --frames <a-b>      frame column and only these frames
  distances     Marker-to-marker distances [MarkerToMarkerDistances.csv]
                  --frames <a-b>      frames (default: the trial's start to end frame)
  precision     Precision info [PrecisionInfo.csv]
                  --frames <a-b>      frames (default: the trial's start to end frame)
  reprojection  Reprojection errors, one file per marker
  mayacams      MayaCams 2.0, the trial's reference calibration only (XMALab's \"No\")
                  --all-images        every calibrated calibration image (XMALab's \"Yes\")
  lut           The undistortion part of Export All: _LUT, _INPTS and _BSPTS per camera
                (Export All's DLT and MayaCam 1 files are not ported)
  events        Events, one file per event
  trial-images  Undistorted Trial images: each camera's frames, undistorted with its grid
                and lens model where it has them, into <dir>/<base>UND/ (one camera: into
                <dir>): <base>_UND.<frame>.<format>, or <base>.<frame>.<format> for a camera
                without undistortion
                  --frames <a-b>      frames (default: the trial's start to end frame)
                  --format <f>        jpg (default), tif or avi (a Motion JPEG video)
                  --filtered          through the visual filter (XMALab's default settings)
                  --camera <n>        only this camera, into <dir> itself
  all-trials    All Trials for External: for every trial and camera,
                <dir>/<trial>/Camera <i>/ with calib/ (MayaCams 2.0), images/ (TIFF),
                markers/ (2D points, undistorted) and reprojection_errors/. --trial is
                ignored. XMALab 3 writes every camera's images into every camera's folder,
                filtered except in Camera 0 (ledger F65; with --xmalab3-compat)
  blender-scene A folder that Blender turns into a scene with the calibrated cameras and
                their image planes (new in RustXMALab, ledger F85): <dir>/<name>/ with
                build_scene.py, scene.json, README.txt and images/camera<n>/. <name> is
                <project>_cameras or <project>_<trial> (--name changes it); a folder that
                exists and is not empty is refused. Without --trial: just cameras, at the
                poses of one calibration image, each with that image undistorted (PNG).
                With --trial: the cameras at the trial's reference calibration poses, each
                with the trial's undistorted images (JPEG sequence; Blender frame f shows
                trial frame f)
                  --calibration-image <n>  the calibration image of the poses (default: the
                                      first calibrated in every camera; not with --trial)
                  --frames <a-b>      the trial's frames whose images are written (default:
                                      the trial's start to end frame)
                  --no-images         cameras and empty planes only: without --trial, no
                                      calibration image; with --trial, no trial images
                  --plane-distance <d> the image planes' distance from their cameras, in the
                                      calibration object's units (default: 1.3 times the
                                      camera's distance from the world origin)
                  --blend             also run Blender in the background to save
                                      <name>.blend in the folder (paths relative to it).
                                      Without Blender, the folder is still complete and
                                      README.txt says how to run the script by hand
                  --pack              pack the still images into the .blend (Blender
                                      cannot pack an image sequence)
                  --blender <path>    the Blender executable (default: $XMALAB_BLENDER,
                                      then blender on PATH, then the usual install places)
                  --rigid-bodies      the trial's rigid bodies as Empties keyed with the
                                      matrices of the rigid-body export (body to world),
                                      hidden at frames without a pose (with --trial)
                  --meshes            each body's bone mesh (its OBJ file copied into
                                      meshes/), the child of its body; exports the bodies
                                      too (with --trial)
                  --points            the markers' 3D points as Empties keyed with the
                                      values of the 3D-point export, hidden at frames
                                      without one (with --trial)
                  --filtered          filtered poses and points (the trial's cutoff), as
                                      the exports' filtered variants; refused where they
                                      refuse
                Bodies and points cover the frames of --frames, or with --no-images the
                whole trial.

Options:
      --out <dir>         Folder to write into (created if absent)
      --trial <n>         Trial (default: the active trial)
      --name <file>       File name of a one-file export
      --xmalab3-compat    Reproduce the XMALab 3 faults this program fixes (the parity
                          ledger items marked \"Fix, compat\": rigid-body poses, undistortion,
                          triangulation, precision report, export details) [default: off]
  -h, --help              Print help
";

const COMMON: [Opt; 4] = [
    Opt::value("out"),
    Opt::value("trial"),
    Opt::value("name"),
    Opt::flag("xmalab3-compat"),
];

const EXPORT_OPTS: [Opt; 28] = [
    Opt::flag("per-marker"),
    Opt::flag("undistorted"),
    Opt::flag("count-from-1"),
    Opt::flag("y-up"),
    Opt::flag("no-header"),
    Opt::flag("offset-columns"),
    Opt::value("markers"),
    Opt::flag("filter"),
    Opt::value("cutoff"),
    Opt::value("frames"),
    Opt::flag("per-body"),
    Opt::value("bodies"),
    Opt::flag("unfiltered"),
    Opt::flag("all-images"),
    Opt::value("format"),
    Opt::value("camera"),
    Opt::value("calibration-image"),
    Opt::flag("no-images"),
    Opt::value("plane-distance"),
    Opt::flag("blend"),
    Opt::flag("pack"),
    Opt::value("blender"),
    Opt::flag("rigid-bodies"),
    Opt::flag("meshes"),
    Opt::flag("points"),
    // Accepted for symmetry with the GUI's radio buttons; they select the defaults.
    Opt::flag("one-file"),
    Opt::flag("distorted"),
    Opt::flag("filtered"),
];

/// The export options each kind of export accepts.
fn allowed(what: &str) -> Option<&'static [&'static str]> {
    Some(match what {
        "points2d" => &[
            "per-marker",
            "one-file",
            "undistorted",
            "distorted",
            "count-from-1",
            "y-up",
            "no-header",
            "offset-columns",
        ],
        "points3d" => &[
            "per-marker",
            "one-file",
            "no-header",
            "markers",
            "filter",
            "cutoff",
            "frames",
        ],
        "rigid-bodies" => &[
            "per-body",
            "one-file",
            "no-header",
            "bodies",
            "unfiltered",
            "filtered",
            "frames",
        ],
        "distances" | "precision" => &["frames"],
        "mayacams" => &["all-images"],
        "reprojection" | "lut" | "events" | "all-trials" => &[],
        "trial-images" => &["frames", "format", "filtered", "camera"],
        "blender-scene" => &[
            "calibration-image",
            "frames",
            "no-images",
            "plane-distance",
            "blend",
            "pack",
            "blender",
            "rigid-bodies",
            "filtered",
            "meshes",
            "points",
        ],
        _ => return None,
    })
}

pub fn run(args: &[String], io: &mut Io<'_>) -> Result<(), CliError> {
    let opts: Vec<Opt> = COMMON
        .iter()
        .chain(&EXPORT_OPTS)
        .chain(&settings::OPTIONS)
        .copied()
        .collect();
    let p = parse(args, &opts)?;
    if p.has("help") {
        io.line(format_args!("{HELP}\n{}", settings::HELP));
        return Ok(());
    }
    let path = p.positional(0, "the project file")?;
    let what = p.positional(1, "what to export (see `xma export --help`)")?;
    p.at_most(2)?;
    let accepted =
        allowed(what).ok_or_else(|| CliError::usage(format!("unknown export '{what}'")))?;
    for o in &EXPORT_OPTS {
        if p.has(o.long) && !accepted.contains(&o.long) {
            return Err(CliError::usage(format!(
                "option '--{}' does not apply to {what}",
                o.long
            )));
        }
    }
    let out = p
        .value("out")
        .ok_or_else(|| CliError::usage("missing --out <dir>"))?;
    let settings = SettingsArgs::from_parsed(&p)?;
    // Usage errors before the project is opened.
    frames(&p)?;
    for (opt, what) in [("markers", "--markers"), ("bodies", "--bodies")] {
        if let Some(v) = p.value(opt) {
            indices(v, what)?;
        }
    }
    if let Some(v) = p.value("cutoff") {
        number::<f64>(v, "--cutoff")?;
    }
    if let Some(v) = p.value("trial") {
        number::<usize>(v, "--trial")?;
    }

    let mut doc = crate::open_document(Path::new(path), &settings, crate::compat_mode(&p))?;
    crate::with_console_lines(&mut doc, io, |doc, io| {
        let n_trials = doc.project().trials.len();
        let trial = match p.value("trial") {
            Some(v) => {
                let n: usize = number(v, "--trial")?;
                if n == 0 || n > n_trials {
                    return Err(CliError::failed(format!(
                        "--trial {n}: the project has {n_trials} trials"
                    )));
                }
                if n - 1 != doc.project().active_trial {
                    doc.select_trial(n - 1);
                }
                n - 1
            }
            None => {
                let t = doc.project().active_trial;
                if t >= n_trials {
                    return Err(CliError::failed("the project has no trial"));
                }
                t
            }
        };

        if what == "blender-scene" {
            return blender_scene(&p, doc, trial, Path::new(out), io);
        }
        if what == "trial-images" || what == "all-trials" {
            return crate::images_cmd::export_images(&p, what, doc, trial, Path::new(out), io);
        }
        let export = build(&p, what, doc, trial)?;
        let output = doc
            .export(trial, &export)
            .map_err(|e| CliError::failed(format!("{what}: {e}")))?;
        let name = p
            .value("name")
            .or(export.default_file_name())
            .unwrap_or("export.csv");
        if p.has("name") && export.default_file_name().is_none() {
            return Err(CliError::usage(format!(
                "--name: {what} writes a folder of files named by XMALab"
            )));
        }
        let dir = Path::new(out);
        std::fs::create_dir_all(dir)
            .map_err(|e| CliError::failed(format!("{}: {e}", dir.display())))?;
        let written = xma_engine::export::write_output_with(dir, &output, name, doc.compat())
            .map_err(|e| CliError::failed(format!("{}: {e}", dir.display())))?;
        for w in &written {
            io.line(w.display());
        }
        if written.is_empty() {
            io.note(format!("{what}: nothing to write"));
        }
        Ok(())
    })
}

/// A 1-based inclusive `--frames` range. Whether it fits the trial is the export's check
/// (an error naming the trial's frames).
fn frames(p: &Parsed) -> Result<Option<(usize, usize)>, CliError> {
    p.value("frames").map(|v| range(v, "--frames")).transpose()
}

fn build(
    p: &Parsed,
    what: &str,
    doc: &xma_engine::Document,
    trial: usize,
) -> Result<Export, CliError> {
    let t = doc.trial(trial);
    let header = !p.has("no-header");
    let range = frames(p)?.map(|(from, to)| FrameRange {
        start: from - 1,
        stop: to - 1,
    });
    Ok(match what {
        "points2d" => Export::Points2d(Points2dOptions {
            one_file: !p.has("per-marker"),
            distorted: !p.has("undistorted"),
            offset1: p.has("count-from-1"),
            y_invert: p.has("y-up"),
            header,
            offset_columns: p.has("offset-columns"),
            // The document's mode decides (`Document::export`).
            compat: doc.compat(),
            camera: None,
        }),
        "points3d" => {
            let filter_frequency = match p.value("cutoff") {
                Some(v) => {
                    let hz: f64 = number(v, "--cutoff")?;
                    if hz.is_nan() || hz <= 0.0 {
                        return Err(CliError::usage("--cutoff: must be above 0"));
                    }
                    hz
                }
                // The dialog's default is the trial's cutoff (ledger F26: the GUI shows it
                // rounded to one decimal; this takes it unrounded, as the dumps do).
                None if p.has("filter") => {
                    if t.cutoff_frequency <= 0.0 {
                        return Err(CliError::failed(
                            "--filter: the trial has no cutoff frequency; give --cutoff",
                        ));
                    }
                    t.cutoff_frequency
                }
                None => -1.0,
            };
            Export::Points3d {
                markers: p
                    .value("markers")
                    .map(|v| indices(v, "--markers"))
                    .transpose()?,
                options: Points3dOptions {
                    one_file: !p.has("per-marker"),
                    header,
                    filter_frequency,
                    frames: range,
                    compat: doc.compat(),
                },
            }
        }
        "rigid-bodies" => Export::RigidBodies {
            bodies: p
                .value("bodies")
                .map(|v| indices(v, "--bodies"))
                .transpose()?,
            options: TransformationsOptions {
                one_file: !p.has("per-body"),
                header,
                filtered: !p.has("unfiltered"),
                frames: range,
                compat: doc.compat(),
            },
        },
        "distances" | "precision" => {
            // The From/To dialog's defaults: the trial's start and end frames; the C++ passes
            // `from - 1` and `to`.
            let (from, to) = match range {
                Some(r) => (r.start, r.stop + 1),
                None => (t.start_frame.saturating_sub(1), t.end_frame),
            };
            if what == "distances" {
                Export::MarkerDistances { from, to }
            } else {
                Export::PrecisionInfo { from, to }
            }
        }
        "reprojection" => Export::ReprojectionErrors,
        "mayacams" => Export::MayaCams {
            all: p.has("all-images"),
        },
        "lut" => Export::UndistortionLut,
        "events" => Export::Events,
        _ => return Err(CliError::usage(format!("unknown export '{what}'"))),
    })
}

/// `xma export ... blender-scene` (ledger F85).
fn blender_scene(
    p: &Parsed,
    doc: &xma_engine::Document,
    trial: usize,
    out: &Path,
    io: &mut Io<'_>,
) -> Result<(), CliError> {
    use xma_engine::blender_scene::{
        BlendOptions, BlenderSceneJob, BlenderSceneKind, BlenderSceneSpec,
    };
    let with_trial = p.has("trial");
    for (opt, ok) in [
        ("calibration-image", !with_trial),
        ("frames", with_trial),
        ("rigid-bodies", with_trial),
        ("meshes", with_trial),
        ("points", with_trial),
    ] {
        if p.has(opt) && !ok {
            return Err(CliError::usage(format!(
                "--{opt}: {} --trial",
                if with_trial { "not with" } else { "only with" }
            )));
        }
    }
    if p.has("frames") && p.has("no-images") {
        return Err(CliError::usage(
            "--frames and --no-images exclude each other",
        ));
    }
    if p.has("filtered") && !(p.has("rigid-bodies") || p.has("meshes") || p.has("points")) {
        return Err(CliError::usage(
            "--filtered goes with --rigid-bodies, --meshes or --points",
        ));
    }
    if (p.has("pack") || p.has("blender")) && !p.has("blend") {
        return Err(CliError::usage("--pack and --blender go with --blend"));
    }
    let kind = if with_trial {
        let t = doc.trial(trial);
        let frames = if p.has("no-images") {
            None
        } else {
            Some(match frames(p)? {
                Some((a, b)) => (a - 1, b - 1),
                None => (t.start_frame.saturating_sub(1), t.end_frame.max(1) - 1),
            })
        };
        BlenderSceneKind::Trial { trial, frames }
    } else {
        let calibration_image = match p.value("calibration-image") {
            Some(v) => {
                let n: usize = number(v, "--calibration-image")?;
                if n == 0 {
                    return Err(CliError::usage("--calibration-image: numbered from 1"));
                }
                Some(n - 1)
            }
            None => None,
        };
        BlenderSceneKind::Cameras {
            calibration_image,
            images: !p.has("no-images"),
        }
    };
    let plane_distance = match p.value("plane-distance") {
        Some(v) => {
            let d: f64 = number(v, "--plane-distance")?;
            if !(d.is_finite() && d > 0.0) {
                return Err(CliError::usage("--plane-distance: must be above 0"));
            }
            Some(d)
        }
        None => None,
    };
    let mut spec = BlenderSceneSpec::new(kind);
    spec.plane_distance = plane_distance;
    spec.rigid_bodies = p.has("rigid-bodies");
    spec.meshes = p.has("meshes");
    spec.points = p.has("points");
    spec.filtered = p.has("filtered");
    spec.name = p.value("name").map(str::to_owned);
    spec.blend = p.has("blend").then(|| BlendOptions {
        blender: p.value("blender").map(std::path::PathBuf::from),
        pack: p.has("pack"),
    });
    let files = xma_engine::tracking::TrialSources::new(doc, trial)
        .map(|s| s.files().iter().map(|f| f.to_vec()).collect::<Vec<_>>())
        .unwrap_or_default();
    std::fs::create_dir_all(out)
        .map_err(|e| CliError::failed(format!("{}: {e}", out.display())))?;
    let job = BlenderSceneJob::new(doc, spec, &files, out)
        .map_err(|e| CliError::failed(format!("blender-scene: {e}")))?;
    let r = job.run(&mut |_| {}, &|| false);
    for w in &r.notes {
        io.note(w);
    }
    for w in &r.left_out {
        io.note(format!("left out: {w}"));
    }
    for w in &r.problems {
        io.note(w);
    }
    if let Some(e) = r.error {
        return Err(CliError::failed(format!("blender-scene: {e}")));
    }
    if let Some(f) = &r.folder {
        io.line(format_args!(
            "{} ({} files, {:.1} s)",
            f.display(),
            r.written.len(),
            r.seconds_folder
        ));
        if r.bodies + r.points > 0 {
            io.note(format!(
                "{} rigid bodies ({} with a mesh), {} 3D points",
                r.bodies, r.meshes, r.points
            ));
        }
    }
    if let Some(b) = &r.blender {
        io.note(format!("Blender ({:.1} s): {b}", r.seconds_blender));
    }
    if let Some(b) = &r.blend {
        io.line(b.display());
    } else if p.has("blend") {
        return Err(CliError::failed(
            "blender-scene: the folder is written but no .blend was made",
        ));
    }
    Ok(())
}
