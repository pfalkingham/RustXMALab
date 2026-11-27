//! `xma recompute`: open, recompute as XMALab does, report what changed, optionally save.

use std::path::Path;

use xma_engine::{Document, SaveOptions};
use xma_model::Trial;

use crate::args::{self, Opt, parse};
use crate::settings::{self, SettingsArgs};
use crate::{CliError, Io};

pub const HELP: &str = "\
xma recompute: open a project, recompute it as XMALab does, report what changed

Usage: xma recompute <project.xma> [-o <out.xma>] [options]

By default this is what XMALab does when it opens a project: the active trial is updated
when every camera is calibrated. Markers stored with RequiresRecomputation=\"0\" keep their
stored 3D points and are only reprojected (parity ledger F9). With --all, every trial is
then flagged for recomputation and updated, which recomputes every marker of every trial.

What changed is reported per trial against the project as read: markers whose 3D points or
statuses changed (with the largest change), and rigid-body frames with a pose.

Options:
  -o, --output <out.xma>   Save the result there, as XMALab's Save As does (creation date
                           and archive entry times set to now, in local time; images,
                           log.html and xma_metadata.xml carried over; PrecisionInfo and
                           MarkerDistances regenerated; the settings used stored in the
                           project's settings entry)
      --save-time <YYYY-MM-DDTHH:MM:SS>
                           Stamp the saved file with this time instead of now
      --all                Recompute every trial and every marker
      --recompute-when-saving
                           XMALab's RecomputeWhenSaving setting [default: off]
      --xmalab3-compat     Reproduce the XMALab 3 faults this program fixes (the parity
                           ledger items marked \"Fix, compat\": rigid-body poses, undistortion,
                           triangulation, meshes and 6-digit numbers on save) [default: off]
  -h, --help               Print help
";

const OPTS: [Opt; 5] = [
    Opt::value("output").short('o'),
    Opt::value("save-time"),
    Opt::flag("all"),
    Opt::flag("recompute-when-saving"),
    Opt::flag("xmalab3-compat"),
];

pub fn run(args: &[String], io: &mut Io<'_>) -> Result<(), CliError> {
    let opts: Vec<Opt> = OPTS.iter().chain(&settings::OPTIONS).copied().collect();
    let p = parse(args, &opts)?;
    if p.has("help") {
        io.line(format_args!("{HELP}\n{}", settings::HELP));
        return Ok(());
    }
    let path = p.positional(0, "the project file")?;
    p.at_most(1)?;
    let settings = SettingsArgs::from_parsed(&p)?;
    let save_time = args::save_time(&p, true)?;

    let loaded = crate::read_loaded(Path::new(path), &settings, crate::compat_mode(&p))?;
    let before: Vec<Trial> = loaded
        .project
        .trials
        .iter()
        .map(|t| t.as_ref().clone())
        .collect();
    let mut doc = Document::from_loaded(loaded);
    doc.set_path(path);
    let updated = doc.update_after_load();
    crate::with_console_lines(&mut doc, io, |doc, io| {
        io.line(format_args!(
            "{path}: opened (compatibility mode: {})",
            doc.compat().describe()
        ));
        for (name, value, source) in settings::describe(doc) {
            io.line(format_args!("  setting {name} = {value} ({source})"));
        }
        match updated {
            Some(t) => io.line(format_args!(
                "  load-time update: trial {} ({})",
                t + 1,
                doc.project().trials[t].name
            )),
            None => io.line(
                "  load-time update: none (not every camera is calibrated, or no calibration)",
            ),
        }
        if p.has("all") {
            for t in 0..doc.project().trials.len() {
                doc.set_trial_requires_recomputation(t, true);
                doc.update_trial(t);
            }
            io.line("  --all: every trial flagged and updated");
        }
        for (t, old) in before.iter().enumerate() {
            report_trial(io, t, old, doc.trial(t));
        }
        for w in doc.warnings() {
            io.line(format_args!("  warning: {w}"));
        }

        if let Some(out) = p.value("output") {
            let mut options = SaveOptions {
                recompute_when_saving: p.has("recompute-when-saving"),
                ..SaveOptions::default()
            };
            if let Some(t) = save_time {
                options.time = t;
            }
            let report = doc
                .save_as(Path::new(out), &options)
                .map_err(|e| CliError::failed(format!("{out}: {e}")))?;
            io.line(format_args!(
                "saved {} ({} entries{})",
                report.path.display(),
                report.entries,
                if report.recomputed.is_empty() {
                    String::new()
                } else {
                    format!(
                        ", recomputed before saving: trials {:?}",
                        report.recomputed.iter().map(|t| t + 1).collect::<Vec<_>>()
                    )
                }
            ));
            for n in &report.not_written {
                io.note(format_args!(
                    "note: {n} not written (neither in the model nor in the file read)"
                ));
            }
        }
        Ok(())
    })
}

/// One trial's changes against the project as read.
fn report_trial(io: &mut Io<'_>, t: usize, old: &Trial, new: &Trial) {
    let mut markers_changed = 0;
    let mut frames_changed = 0;
    let mut status_changed = 0;
    let mut worst = 0.0_f64;
    for (m_old, m_new) in old.markers.iter().zip(&new.markers) {
        let mut changed = false;
        for f in 0..m_old.points3d.len().min(m_new.points3d.len()) {
            let (a, b) = (m_old.points3d[f], m_new.points3d[f]);
            if a != b {
                frames_changed += 1;
                changed = true;
                if m_new.status3d.get(f).is_some_and(|s| s.has_data())
                    && m_old.status3d.get(f).is_some_and(|s| s.has_data())
                {
                    worst = worst.max((a - b).norm());
                }
            }
        }
        let st = m_old
            .status3d
            .iter()
            .zip(&m_new.status3d)
            .filter(|(a, b)| a != b)
            .count();
        status_changed += st;
        if changed || st > 0 {
            markers_changed += 1;
        }
    }
    io.line(format_args!(
        "  trial {} ({}): {} of {} markers changed: 3D points at {frames_changed} frames \
         (largest move between frames with data {worst:.3e}), 3D status at {status_changed} \
         frames; requires recomputation {} -> {}",
        t + 1,
        new.name,
        markers_changed,
        new.markers.len(),
        old.requires_recomputation,
        new.requires_recomputation
    ));
    for (k, b) in new.rigid_bodies.iter().enumerate() {
        io.line(format_args!(
            "    rigid body {} ({}): {} of {} frames with a pose, {} filtered",
            k + 1,
            b.description,
            b.raw.valid.iter().filter(|&&v| v).count(),
            new.n_frames,
            b.filtered.valid.iter().filter(|&&v| v).count()
        ));
    }
}
