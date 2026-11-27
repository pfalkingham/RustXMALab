//! `xma export ... trial-images | all-trials` (File > Export > Undistorted Trial images and
//! All Trials for External) and `xma denoise` (the Trial dialog's "Denoise Images").

use std::path::Path;

use xma_engine::denoise::{DenoiseJob, DenoiseReport, DenoiseSettings, odd, relink_spec};
use xma_engine::tracking::TrialSources;
use xma_engine::trial_images::{AllTrialsJob, TrialImageFormat, TrialImagesJob, TrialImagesSpec};
use xma_engine::{Document, Edit, SaveOptions};
use xma_vision::visual_filter::VisualFilter;

use crate::args::{Opt, Parsed, number, one_based, parse, range};
use crate::settings::{self, SettingsArgs};
use crate::{CliError, Io};

/// Each camera's files of `trial` as the project stores them.
fn trial_files(doc: &Document, trial: usize) -> Vec<Vec<std::path::PathBuf>> {
    TrialSources::new(doc, trial)
        .map(|s| s.files().iter().map(|f| f.to_vec()).collect())
        .unwrap_or_default()
}

/// The two image exports of `xma export`.
pub(crate) fn export_images(
    p: &Parsed,
    what: &str,
    doc: &Document,
    trial: usize,
    out: &Path,
    io: &mut Io<'_>,
) -> Result<(), CliError> {
    std::fs::create_dir_all(out)
        .map_err(|e| CliError::failed(format!("{}: {e}", out.display())))?;
    if what == "all-trials" {
        let job = AllTrialsJob::new(doc, out, &|t| trial_files(doc, t), VisualFilter::default());
        let r = job.run(&mut |_| {}, &|| false);
        for pr in &r.problems {
            io.note(pr);
        }
        io.line(format_args!(
            "{} files written under {}",
            r.written,
            out.display()
        ));
        if let Some(e) = r.error {
            return Err(CliError::failed(e));
        }
        return Ok(());
    }
    let t = doc.trial(trial);
    let (first, last) = match p.value("frames") {
        Some(v) => {
            let (a, b) = range(v, "--frames")?;
            if b > t.n_frames {
                return Err(CliError::failed(format!(
                    "--frames {v}: the trial has {} frames",
                    t.n_frames
                )));
            }
            (a - 1, b - 1)
        }
        None => (
            t.start_frame.saturating_sub(1),
            t.end_frame.saturating_sub(1),
        ),
    };
    let format = match p.value("format") {
        Some(f) => TrialImageFormat::parse(f)
            .ok_or_else(|| CliError::usage(format!("--format {f}: jpg, tif or avi")))?,
        None => TrialImageFormat::Jpg,
    };
    let camera = one_based(p, "camera", "cameras")?;
    let spec = TrialImagesSpec {
        trial,
        first,
        last,
        format,
        filtered: p.has("filtered"),
        filter: VisualFilter::default(),
        camera,
    };
    let job = TrialImagesJob::new(doc, spec, &trial_files(doc, trial), out)
        .map_err(|e| CliError::failed(format!("{what}: {e}")))?;
    let r = job.run(&mut |_| {}, &|| false);
    for c in &r.cameras {
        io.line(format_args!(
            "camera {}: {} files in {}",
            c.camera + 1,
            c.written.len(),
            c.folder.display()
        ));
    }
    for pr in r.problems() {
        io.note(pr);
    }
    if r.cameras.iter().any(|c| c.error.is_some()) {
        return Err(CliError::failed(format!("{what}: a camera wrote nothing")));
    }
    Ok(())
}

pub const HELP: &str = "\
xma denoise: denoise a trial's images (the Trial dialog's \"Denoise Images\")

Usage: xma denoise <project.xma> --out <dir> [--trial <n>] [options]

Each camera i (from 0, as XMALab names the folders) is written as
<dir>/Cam<i>/<base>.<frame, from 0001>.jpg: every frame denoised with OpenCV's non-local
means over a temporal window (cv::fastNlMeansDenoisingMulti, 8 bits), except the first and
last (temporal - 1) / 2 frames, which are copied. Window sizes that are even are raised to
the next odd number, as XMALab's spin boxes do.

Options:
      --out <dir>           Folder to write into (created if absent)
      --trial <n>           Trial [default: the project's active trial]
      --search <n>          Search window size [default: 35]
      --template <n>        Template window size [default: 11]
      --temporal <n>        Temporal window size [default: 5]
      --strength <h>        Filter strength [default: 5]
  -o, --output <out.xma>    Relink the trial to the denoised images (\"relink Trial to
                            denoised images\") and save the project there
      --xmalab3-compat      As XMALab 3: the strength truncated to an integer, the frame
                            N - w denoised with a window past the end (OpenCV refuses it:
                            the camera stops there), later frames denoised with the frames
                            already denoised, a flipped camera's files mirrored, and the
                            relink to every *.jpg of the folders (ledger A190 to A192, V30,
                            F66) [default: off]
  -h, --help                Print help
";

const OPTS: [Opt; 8] = [
    Opt::value("out"),
    Opt::value("trial"),
    Opt::value("search"),
    Opt::value("template"),
    Opt::value("temporal"),
    Opt::value("strength"),
    Opt::value("output").short('o'),
    Opt::flag("xmalab3-compat"),
];

/// Prints each camera's result, in camera order (`results[c]` is camera `c`'s job, which
/// had `frames[c]` frames). Returns every camera's report, or `None` when a job ended
/// without one (it panicked), and whether every camera wrote every frame.
fn camera_reports(
    frames: &[usize],
    results: Vec<std::thread::Result<DenoiseReport>>,
    out: &Path,
    io: &mut Io<'_>,
) -> (Option<Vec<DenoiseReport>>, bool) {
    let mut complete = results.len() == frames.len();
    let mut reports = Some(Vec::with_capacity(results.len()));
    for (c, (result, &n)) in results.into_iter().zip(frames).enumerate() {
        let Ok(r) = result else {
            io.note(format!(
                "camera {}: the denoising job stopped with an internal error",
                c + 1
            ));
            reports = None;
            complete = false;
            continue;
        };
        io.line(format_args!(
            "camera {}: {} denoised, {} copied, {} files in {}",
            r.camera + 1,
            r.denoised,
            r.copied,
            r.written.len(),
            xma_engine::denoise::camera_folder(out, r.camera).display()
        ));
        if let Some(e) = &r.error {
            io.note(format!("camera {}: {e}", r.camera + 1));
        }
        complete &= r.complete(n);
        if let Some(v) = &mut reports {
            v.push(r);
        }
    }
    (reports, complete)
}

pub fn run(args: &[String], io: &mut Io<'_>) -> Result<(), CliError> {
    let opts: Vec<Opt> = OPTS.iter().chain(&settings::OPTIONS).copied().collect();
    let p = parse(args, &opts)?;
    if p.has("help") {
        io.line(format_args!("{HELP}\n{}", settings::HELP));
        return Ok(());
    }
    let path = p.positional(0, "the project file")?;
    p.at_most(1)?;
    let out = Path::new(
        p.value("out")
            .ok_or_else(|| CliError::usage("missing --out <dir>"))?,
    );
    let int = |k: &str, d: i32| -> Result<i32, CliError> {
        p.value(k)
            .map_or(Ok(d), |v| number::<i32>(v, &format!("--{k}")))
            .map(odd)
    };
    let settings_args = SettingsArgs::from_parsed(&p)?;
    let ds = DenoiseSettings {
        search: int("search", 35)?,
        template: int("template", 11)?,
        temporal: int("temporal", 5)?,
        strength: p
            .value("strength")
            .map_or(Ok(5.0), |v| number::<f64>(v, "--strength"))?,
        relink: p.has("output"),
    };
    let mut doc = crate::open_document(Path::new(path), &settings_args, crate::compat_mode(&p))?;
    crate::with_console_lines(&mut doc, io, |doc, io| {
        let trial = match one_based(&p, "trial", "trials")? {
            Some(t) => t,
            None => doc.project().active_trial,
        };
        if trial >= doc.project().trials.len() {
            return Err(CliError::failed(format!("no trial {}", trial + 1)));
        }
        let files = trial_files(doc, trial);
        let mut jobs = Vec::new();
        for c in 0..doc.project().cameras.len() {
            let f = files.get(c).cloned().unwrap_or_default();
            jobs.push(DenoiseJob::new(doc, trial, c, &f, out, &ds).map_err(CliError::failed)?);
        }
        // One job per camera, at once, as XMALab starts them.
        let results: Vec<std::thread::Result<DenoiseReport>> = std::thread::scope(|s| {
            let hs: Vec<_> = jobs
                .iter()
                .map(|j| s.spawn(move || j.run(&mut |_| {}, &|| false)))
                .collect();
            hs.into_iter().map(|h| h.join()).collect()
        });
        let frames_of: Vec<usize> = jobs.iter().map(DenoiseJob::frames).collect();
        let (reports, complete) = camera_reports(&frames_of, results, out, io);
        let Some(reports) = reports else {
            // A camera's job ended without a report: its files are unknown, so the trial is
            // neither relinked nor saved.
            return Err(CliError::failed(
                "denoising did not finish for every camera",
            ));
        };
        let frames = jobs.first().map_or(0, DenoiseJob::frames);
        if let Some(target) = p.value("output")
            && xma_engine::denoise::relink_due(doc, &reports, frames)
        {
            // XMALab 3 mode: relinked also after OpenCV's refusal (ledger A191).
            let spec = relink_spec(doc, trial, out, &reports)
                .ok_or_else(|| CliError::failed("the trial is gone"))?;
            doc.apply(&Edit::ChangeTrialData { trial, spec })
                .map_err(|e| CliError::failed(format!("relink: {e}")))?;
            let report = doc
                .save_as(Path::new(target), &SaveOptions::default())
                .map_err(|e| CliError::failed(format!("{target}: {e}")))?;
            io.line(format_args!(
                "relinked and saved {} ({} entries)",
                report.path.display(),
                report.entries
            ));
        }
        if !complete {
            return Err(CliError::failed(
                "denoising did not finish for every camera",
            ));
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn report(camera: usize, frames: usize) -> DenoiseReport {
        DenoiseReport {
            camera,
            written: (0..frames)
                .map(|f| PathBuf::from(format!("c{camera}_{f}.png")))
                .collect(),
            denoised: frames,
            ..DenoiseReport::default()
        }
    }

    fn run_reports(
        frames: &[usize],
        results: Vec<std::thread::Result<DenoiseReport>>,
    ) -> (Option<Vec<DenoiseReport>>, bool, String, String) {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut io = Io {
            out: &mut out,
            err: &mut err,
        };
        let (r, complete) = camera_reports(frames, results, Path::new("/d"), &mut io);
        (
            r,
            complete,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    /// Review of 20 November 2026, second round (video-model-cli 7): a camera whose job
    /// panicked was dropped from the reports, the rest paired with the wrong cameras, and
    /// the trial relinked and saved without that camera's files.
    #[test]
    fn review12_denoise_panicked_camera_stops_the_relink() {
        let panicked: std::thread::Result<DenoiseReport> = Err(Box::new("job panicked"));
        let (r, complete, out, err) = run_reports(
            &[4, 4, 4],
            vec![Ok(report(0, 4)), panicked, Ok(report(2, 4))],
        );
        assert!(r.is_none(), "no reports to relink with");
        assert!(!complete);
        assert_eq!(
            out.lines().map(|l| &l[..9]).collect::<Vec<_>>(),
            ["camera 1:", "camera 3:"]
        );
        assert!(err.contains("camera 2:"), "{err}");
    }

    #[test]
    fn review12_denoise_reports_stay_in_camera_order() {
        let (r, complete, out, err) =
            run_reports(&[4, 3], vec![Ok(report(0, 4)), Ok(report(1, 2))]);
        let r = r.unwrap();
        assert_eq!(r.iter().map(|x| x.camera).collect::<Vec<_>>(), [0, 1]);
        assert!(!complete, "camera 2 wrote 2 of 3 frames");
        assert_eq!(out.lines().count(), 2);
        assert!(err.is_empty());
        let (r, complete, ..) = run_reports(&[4, 3], vec![Ok(report(0, 4)), Ok(report(1, 3))]);
        assert_eq!(r.unwrap().len(), 2);
        assert!(complete);
    }
}
