//! `xma info`: what a project contains.

use std::fmt::Write as _;
use std::path::Path;

use xma_engine::Document;
use xma_model::{CalibrationObject, CalibrationType, Camera, Trial};

use crate::args::{Opt, parse};
use crate::settings::{self, SettingsArgs};
use crate::{CliError, Io};

pub const HELP: &str = "\
xma info: what a project contains

Usage: xma info <project.xma> [options]

Opens the project as XMALab does (which updates the active trial when every camera is
calibrated) and prints: the file version; the settings that change results, each with where
it came from (project, default or command line); each camera's size, calibration,
undistortion grid and lens model; the calibration type and object; each trial's frames,
recording speed, cutoff and reference calibration; markers and rigid bodies with their
counts of tracked frames; the reader's warnings.

Counts: a marker's 2D frames are those with a 2D status above 0 in that camera, its 3D
frames those with a 3D status above 0, as held after loading. A rigid body's frames are those
with a pose, which XMALab computes only for a trial that has been updated (the active one on
load); for other trials they are shown as not computed.

Options:
      --all-settings  List every stored setting, not only those phase 1 reads
  -h, --help          Print help";

pub fn run(args: &[String], io: &mut Io<'_>) -> Result<(), CliError> {
    let opts: Vec<Opt> = std::iter::once(Opt::flag("all-settings"))
        .chain(settings::OPTIONS)
        .collect();
    let p = parse(args, &opts)?;
    if p.has("help") {
        io.line(format_args!("{HELP}\n\n{}", settings::HELP));
        return Ok(());
    }
    let path = p.positional(0, "the project file")?;
    p.at_most(1)?;
    let args = SettingsArgs::from_parsed(&p)?;
    let mut doc = crate::open_document(Path::new(path), &args, xma_model::CppCompat::default())?;
    crate::with_console_lines(&mut doc, io, |doc, io| {
        io.line(describe(doc, path, p.has("all-settings")));
        Ok(())
    })
}

fn yes(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

/// The report, as text. `all_settings` lists every stored setting, not only those phase 1
/// reads.
#[must_use]
pub fn describe(doc: &Document, path: &str, all_settings: bool) -> String {
    let p = doc.project();
    let ex = doc.extras();
    let mut s = String::new();
    let _ = writeln!(s, "Project: {path}");
    let _ = writeln!(
        s,
        "File version: {}",
        ex.file_version.text.as_deref().unwrap_or("(none)")
    );
    let _ = writeln!(
        s,
        "Created: {}",
        if p.date_created.is_empty() {
            "(not set)"
        } else {
            &p.date_created
        }
    );
    let sources = &ex.settings_sources;
    let source = |name: &str| {
        sources
            .iter()
            .find(|(n, _)| *n == name)
            .map_or("default", |(_, s)| s.label())
    };
    let _ = writeln!(
        s,
        "Settings{}:",
        if ex.stored_settings.is_empty() {
            " (the project stores none)"
        } else {
            ""
        }
    );
    let names: Vec<&str> = if all_settings {
        xma_model::SETTING_NAMES.iter().map(|(n, _)| *n).collect()
    } else {
        settings::USED_SETTINGS.to_vec()
    };
    for name in names {
        let value = doc
            .settings()
            .get(name)
            .map_or_else(String::new, |v| v.to_string());
        let _ = writeln!(s, "  {name} = {value} ({})", source(name));
    }
    let cal = match p.calibration_type {
        CalibrationType::None => "none",
        CalibrationType::Internal => "internal (calibration object)",
        CalibrationType::External => "external (imported)",
    };
    let _ = writeln!(s, "Calibration: {cal}");
    match &p.calibration_object {
        Some(CalibrationObject::Frame(f)) => {
            let _ = writeln!(
                s,
                "Calibration object: frame of {} points ({}, references {}), {} beads",
                f.points.len(),
                f.frame_specification_filename,
                f.references_filename,
                if f.white_blobs { "white" } else { "dark" }
            );
        }
        Some(CalibrationObject::Checkerboard {
            horizontal_squares,
            vertical_squares,
            square_size,
            ..
        }) => {
            let _ = writeln!(
                s,
                "Calibration object: checkerboard {horizontal_squares} x {vertical_squares}, \
                 squares of {square_size}"
            );
        }
        None => {
            let _ = writeln!(s, "Calibration object: none");
        }
    }
    let _ = writeln!(
        s,
        "Cameras: {} (all calibrated: {})",
        p.cameras.len(),
        yes(doc.is_calibrated())
    );
    for (i, cam) in p.cameras.iter().enumerate() {
        camera(&mut s, i, cam);
    }
    let _ = writeln!(
        s,
        "Trials: {} (active: {})",
        p.trials.len(),
        p.active_trial + 1
    );
    for (i, t) in p.trials.iter().enumerate() {
        trial(&mut s, i, t);
    }
    let warnings: Vec<&String> = ex.warnings.iter().chain(doc.warnings()).collect();
    if warnings.is_empty() {
        let _ = writeln!(s, "Warnings: none");
    } else {
        let _ = writeln!(s, "Warnings:");
        for w in warnings {
            let _ = writeln!(s, "  {w}");
        }
    }
    s.trim_end().to_owned()
}

fn camera(s: &mut String, i: usize, cam: &Camera) {
    let _ = writeln!(
        s,
        "  {}. {}: {} x {}, {} camera, calibrated: {}",
        i + 1,
        cam.name,
        cam.width,
        cam.height,
        if cam.is_light_camera {
            "light"
        } else {
            "X-ray"
        },
        yes(cam.calibrated)
    );
    let grid = match &cam.undistortion {
        None => "none".to_owned(),
        Some(g) => {
            let inliers = g.grid_inlier.iter().filter(|&&b| b).count();
            format!(
                "{} ({}, {} grid points, {inliers} inliers)",
                if g.computed {
                    "computed"
                } else {
                    "not computed"
                },
                g.image_filename,
                g.grid_distorted.len()
            )
        }
    };
    let _ = writeln!(s, "     undistortion grid: {grid}");
    let lens = if cam.has_model_distortion {
        let k: Vec<String> = cam.distortion.iter().map(|v| format!("{v}")).collect();
        format!("on ({})", k.join(", "))
    } else {
        "off".to_owned()
    };
    let _ = writeln!(s, "     lens model: {lens}");
    let k = &cam.camera_matrix;
    let _ = writeln!(
        s,
        "     focal length: {} x {}, principal point: ({}, {})",
        k[(0, 0)],
        k[(1, 1)],
        k[(0, 2)],
        k[(1, 2)]
    );
    let calibrated = cam
        .calibration_images
        .iter()
        .filter(|im| im.is_calibrated())
        .count();
    let _ = writeln!(
        s,
        "     calibration images: {} ({calibrated} calibrated){}",
        cam.calibration_images.len(),
        cam.calibration_sequence
            .as_ref()
            .map_or(String::new(), |q| format!(", from video {}", q.filename))
    );
}

fn trial(s: &mut String, i: usize, t: &Trial) {
    let _ = writeln!(
        s,
        "  {}. {}{}: {} frames (start {}, end {}), recording speed {} Hz, cutoff {} Hz, \
         reference calibration {}, requires recomputation: {}",
        i + 1,
        t.name,
        if t.is_default { " (default trial)" } else { "" },
        t.n_frames,
        t.start_frame,
        t.end_frame,
        t.recording_speed,
        t.cutoff_frequency,
        if t.reference_calibration_image == xma_model::trial::NO_REFERENCE_CALIBRATION {
            "none (-1)".to_owned()
        } else {
            (t.reference_calibration_image + 1).to_string()
        },
        yes(t.requires_recomputation)
    );
    let _ = writeln!(s, "     markers: {}", t.markers.len());
    for (k, m) in t.markers.iter().enumerate() {
        let cams: Vec<String> = m
            .views
            .iter()
            .enumerate()
            .map(|(c, v)| {
                format!(
                    "camera {} {}",
                    c + 1,
                    v.status.iter().filter(|s| s.has_data()).count()
                )
            })
            .collect();
        let tracked3d = m.status3d.iter().filter(|s| s.has_data()).count();
        let _ = writeln!(
            s,
            "       {}. {}: 2D frames {}; 3D frames {tracked3d}",
            k + 1,
            name(&m.description),
            cams.join(", ")
        );
    }
    let _ = writeln!(s, "     rigid bodies: {}", t.rigid_bodies.len());
    let computed = !t.requires_recomputation;
    for (k, b) in t.rigid_bodies.iter().enumerate() {
        let markers: Vec<String> = b
            .marker_indices
            .iter()
            .map(|m| (m + 1).to_string())
            .collect();
        let poses = if computed {
            format!(
                "frames with a pose {}, filtered {}",
                b.raw.valid.iter().filter(|&&v| v).count(),
                b.filtered.valid.iter().filter(|&&v| v).count()
            )
        } else {
            "poses not computed (trial not updated)".to_owned()
        };
        let _ = writeln!(
            s,
            "       {}. {}: markers {}{}; {poses}",
            k + 1,
            name(&b.description),
            markers.join(","),
            if b.virtual_points.is_empty() {
                String::new()
            } else {
                format!(", {} virtual points", b.virtual_points.len())
            }
        );
    }
    let _ = writeln!(s, "     events: {}", t.events.len());
}

fn name(description: &str) -> &str {
    if description.is_empty() {
        "(no description)"
    } else {
        description
    }
}
