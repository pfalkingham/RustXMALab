//! `xma track`: track markers headlessly, as the Marker tracking workspace's track buttons
//! do, and optionally save the result.

use std::path::Path;

use xma_engine::SaveOptions;
use xma_engine::tracking::{
    Direction, StepReport, StopReason, TrackError, TrackKind, TrackRequest, TrackRun, TrialSources,
};
use xma_model::MarkerStatus;

use crate::args::{self, Opt, parse};
use crate::settings::{self, SettingsArgs};
use crate::{CliError, Io};

pub const HELP: &str = "\
xma track: track markers as XMALab's track buttons do, and optionally save the result

Usage: xma track <project.xma> --start <frame> [options]

Opens the project as XMALab does, then runs the tracking loop of the Marker tracking
workspace from <frame>: per step, a template tracker for each marker and camera that
qualifies, detection at the tracked points, then the validity check. Frames, markers,
cameras and trials are numbered from 1, as in XMALab. The images are read from the trial's
video files as the project stores them.

Options:
      --start <frame>          The frame to track from (the frame on screen)
      --trial <n>              The trial [default: the project's active trial]
      --kind <kind>            single (the --markers' first, \"single point\"), selected
                               (\"selected points\"), all (\"all points\"), multi or
                               multi-selected (multi-camera tracking, new in this program;
                               see below) [default: all]
      --markers <i,j,...>      The marker for single, the markers for selected and
                               multi-selected
      --backward               Track backward (the Prev and Back buttons)
      --single                 One step (the Next or Prev button)
      --to <frame>             Continuous, stopping after the step that reaches <frame>;
                               <frame> equal to --start tracks nothing
                               [default: continuous to the trial's last or first frame]
      --cameras <i,j,...>      The cameras shown; hidden cameras are not tracked
                               [default: all]
  -o, --output <out.xma>       Save the result there, as Save As does
      --save-time <YYYY-MM-DDTHH:MM:SS>
                               Stamp the saved file with this time instead of now
      --quiet                  Print only the summary, not a line per step
      --xmalab3-compat         Reproduce XMALab 3's tracking faults (ledger A1, A14, A15,
                               A62, A64, A65, A73 and the others marked \"Fix, compat\"),
                               and reduce images deeper than 8 bits to 8 bits as XMALab
                               does instead of tracking at their own depth (V1, V15, V18)
                               [default: off]
      --multi <name=value,...> Multi-camera tracking only: override parameters of the
                               method (names below)
  -h, --help                   Print help

Multi-camera tracking (--kind multi, multi-selected; new in this program, not in XMALab 3,
refused with --xmalab3-compat; needs at least two calibrated cameras that are shown, and
says so otherwise): per step all markers are predicted in 3D, several candidate dots are
found per camera (correlation maxima refined by the marker's detection), and one dot per
camera is chosen for all markers together, so that the dots agree with each other
(reprojection error) and with the prediction (a gate in every camera), no dot goes to two
markers (unless both are predicted on it: an overlap), and a camera whose dot is in doubt
is left undefined. Markers whose 3D distance has stayed constant are held to it. A marker
that is lost is looked for again round the position its neighbours give, and accepted only
when it keeps its distances to them; frames where it is not found stay undefined. The run
reads only the points at and behind its start frame (whatever their status), the points it
writes, and the points it may not overwrite (set by hand, and the others the existing kinds
keep); never a stored 3D point, nor points of a hidden camera. For a marker the run has
had, a point ahead that it may overwrite and does not find again is left as it is (and
never read), unless the run's own points contradict it: its other cameras' points written
at that frame triangulate with it to a reprojection error above max_reprojection_cap
(or not at all, as a point far outside the image), or
the run gave its dot (within same_dot) to another marker that it does not hold as
touching this one there (keep_dot_rule=0 turns this second test off: it removes good
points where two markers share a dot, and keeps points slid onto another marker's dot);
such a point is reset (keep_unconfirmed=0 resets every point the run does not confirm,
the rule before 24 November 2026). The summary gives how many earlier points were kept
and removed. Points written get status TRACKED; points it may not
overwrite are kept, and the cells left undefined because their dot was in doubt are listed
per step (\"invalid\"). The contrast test compares a dot's contrast relative to the
brightness round it (contrast_relative=1, the default since 24 November 2026: a marker
whose background brightens and darkens again is not lost; contrast_relative=0 compares
the absolute contrast). split_touching=1 (new 25 November 2026; on by default since 26
November 2026; parity ledger A223, A224, A225) fits two Gaussian dots of the marker's learned width where its
detection has joined two touching dots into one shape (the detected shape covers both
fitted centres, the detected point lies between them, and there is no third dot), and
offers both fitted centres as candidates in place of the shape's centre; such a point is written with size -1, is not
used as a clean picture of the marker afterwards, and the summary says how many points
were placed so (split_depth_ratio, split_rms_ratio, split_min_apart, split_max_move,
split_reach_widths and split_reach_margin are the fit's numbers).
stat_links=1 (new 25 November 2026; on by default since 26 November 2026;
parity ledger A223, A224, A225) recognises a marker
being found again by the run's own distance statistics to every marker whose pair with it
was measured at rigid_min_frames frames or more (mean, tolerance the larger of rigid_sigmas
spreads and rigid_min_tolerance of the mean); for such a pair this replaces the 2 % link
from one reference frame, and a pair measured at fewer frames keeps that link.
distance_placement=1 (new 25 November 2026; on by default since 26 November 2026;
parity ledger A223, A224, A225) places a lost
marker that has no rigid prediction at the point that best keeps its mean distances to four
or more markers followed in 3D, looks for it there with gate_rigid, and recognises it by
those statistics. With rigid_min_frames=0 no statistics are kept and neither does anything.
split_touching=0,stat_links=0,distance_placement=0 gives the run as it was before
26 November 2026.
Parameters (--multi), with their defaults:
  max_reprojection_error=1.5  reprojection_factor=2.5  max_reprojection_cap=3
  gate=5  gate_hold=10  gate_growth=4  gate_max=24  max_gap=3  depth_gap=300
  velocity_frames=2  velocity_window=4  search_margin=3  max_candidates=6
  min_correlation=0.3  max_detection_shift=3  min_contrast_fraction=0.6
  contrast_history=10  same_dot=1.5  single_gate=3  single_max_frames=0
  single_ambiguity=6  weight_reprojection=1  weight_prediction=1  weight_match=1
  weight_missing=1  ambiguity_margin=0.3  share_radius=1.5  contest_ratio=1
  rigid_min_frames=10  rigid_max_spread=0.02  rigid_sigmas=4  rigid_min_tolerance=0.01
  rigid_history=300  rigid_majority=2  rigid_neighbours=5  rigid_fit_tolerance=0.02
  gate_rigid=6  reacquire_links=2  reacquire_dots=1  max_combinations=4096
  gate_start=40  start_confirm_near=0  contrast_relative=1
  reacquire_exclusive=1  claim_radius=6  claim_ratio=0.5  share_needs_overlap=1
  overlap_margin=1  overlap_max_frames=30  overlap_clean=1  ray_links=1
  overlap_decision=0.5  overlap_take=1  overlap_single_checked=0  keep_unconfirmed=1
  keep_dot_rule=1  split_touching=1  split_depth_ratio=0.35  split_rms_ratio=0.8
  split_min_apart=1.5  split_max_move=4  split_reach_widths=3  split_reach_margin=4
  stat_links=1  distance_placement=1
(pixels for distances; xma_vision::joint_tracking::MultiTrackParams documents each)
";

const OPTS: [Opt; 13] = [
    Opt::value("start"),
    Opt::value("trial"),
    Opt::value("kind"),
    Opt::value("markers"),
    Opt::flag("backward"),
    Opt::flag("single"),
    Opt::value("to"),
    Opt::value("cameras"),
    Opt::value("output").short('o'),
    Opt::value("save-time"),
    Opt::flag("quiet"),
    Opt::flag("xmalab3-compat"),
    Opt::value("multi"),
];

#[allow(clippy::too_many_lines)]
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
    let start: usize = args::number(
        p.value("start")
            .ok_or_else(|| CliError::usage("missing --start"))?,
        "--start",
    )?;
    let start = start
        .checked_sub(1)
        .ok_or_else(|| CliError::usage("--start: frames are numbered from 1"))?;
    let markers = p
        .value("markers")
        .map(|v| args::indices(v, "--markers"))
        .transpose()?
        .unwrap_or_default();
    let kind_name = p.value("kind").unwrap_or("all");
    let kind = match kind_name {
        "multi" => TrackKind::Multi,
        "multi-selected" => {
            if markers.is_empty() {
                return Err(CliError::usage("--kind multi-selected needs --markers"));
            }
            TrackKind::MultiSelected(markers.clone())
        }
        "single" => TrackKind::Single(
            *markers
                .first()
                .ok_or_else(|| CliError::usage("--kind single needs --markers"))?,
        ),
        "selected" => TrackKind::Selected(markers.clone()),
        "all" => TrackKind::All,
        other => {
            return Err(CliError::usage(format!(
                "--kind: '{other}' is not single, selected, all, multi or multi-selected"
            )));
        }
    };
    let mut params = xma_vision::joint_tracking::MultiTrackParams::default();
    if let Some(v) = p.value("multi") {
        if !kind.is_multi() {
            return Err(CliError::usage(
                "--multi applies to --kind multi and multi-selected only",
            ));
        }
        set_multi_params(&mut params, v)?;
    }
    if p.has("single") && p.has("to") {
        return Err(CliError::usage("--single and --to cannot be combined"));
    }
    let end_frame = args::one_based(&p, "to", "frames")?;
    // The engine refuses an end frame behind the start in the run's direction; say which
    // option is at fault (review 11).
    if let Some(end) = end_frame {
        let behind = if p.has("backward") {
            end > start
        } else {
            end < start
        };
        if behind {
            return Err(CliError::usage(format!(
                "--to: frame {} is behind the start frame {} in the run's direction",
                end + 1,
                start + 1
            )));
        }
    }
    let save_time = args::save_time(&p, true)?;

    let mut doc = crate::open_document(Path::new(path), &settings, crate::compat_mode(&p))?;
    crate::with_console_lines(&mut doc, io, |doc, io| {
        let trial = match args::one_based(&p, "trial", "trials")? {
            Some(t) => t,
            None => doc.project().active_trial,
        };
        if trial >= doc.project().trials.len() {
            return Err(CliError::usage(format!("--trial: no trial {}", trial + 1)));
        }
        // Selecting a trial runs its update when it is flagged, as the trial selector does.
        doc.select_trial(trial);
        let n_cameras = doc.project().cameras.len();
        let visible_cameras = match p.value("cameras") {
            None => vec![true; n_cameras],
            Some(v) => {
                let shown = args::indices(v, "--cameras")?;
                if let Some(c) = shown.iter().find(|&&c| c >= n_cameras) {
                    return Err(CliError::usage(format!("--cameras: no camera {}", c + 1)));
                }
                (0..n_cameras).map(|c| shown.contains(&c)).collect()
            }
        };
        let request = TrackRequest {
            trial,
            kind,
            direction: if p.has("backward") {
                Direction::Backward
            } else {
                Direction::Forward
            },
            continuous: !p.has("single"),
            start_frame: start,
            end_frame,
            visible_cameras,
        };
        io.line(format_args!(
            "{path}: opened (compatibility mode: {})",
            doc.compat().describe()
        ));
        let to_cli = |e: TrackError| match e {
            TrackError::Invalid(t) | TrackError::Refused(t) => CliError::usage(t),
            TrackError::NotImplemented(_) => CliError::failed(e.to_string()),
        };
        let mut tracker = TrackRun::start(doc, request, params).map_err(to_cli)?;
        let mut images = TrialSources::new(doc, trial).map_err(CliError::failed)?;
        let quiet = p.has("quiet");
        let mut tracked = 0usize;
        let mut detected = 0usize;
        let mut invalid = 0usize;
        let mut kept_older = 0usize;
        let mut removed_older = 0usize;
        let mut fitted_apart = 0usize;
        let t0 = std::time::Instant::now();
        doc.begin_step("Track Points");
        let mut from = start;
        let last: StepReport = tracker.run(doc, &mut images, &mut |r| {
            tracked += r.tracked.len();
            detected += r.detected.len();
            invalid += r.invalid.len();
            kept_older += r.kept_older;
            removed_older += r.removed_older;
            fitted_apart += r.fitted_apart;
            if !quiet {
                io.line(step_line(from, r));
                for m in &r.messages {
                    io.line(format_args!("  {m}"));
                }
            }
            from = r.frame;
            true
        });
        // The last step's report (the one that ended the run) is not passed to `progress`.
        if tracker.steps_done() > 0 && last.frame != from {
            tracked += last.tracked.len();
            detected += last.detected.len();
            invalid += last.invalid.len();
            kept_older += last.kept_older;
            removed_older += last.removed_older;
            fitted_apart += last.fitted_apart;
            if !quiet {
                io.line(step_line(from, &last));
                for m in &last.messages {
                    io.line(format_args!("  {m}"));
                }
            }
        }
        doc.commit_step();
        let elapsed = t0.elapsed().as_secs_f64();
        let reason = match tracker.finished() {
            Some(StopReason::SingleStep) => "one step done".to_owned(),
            Some(StopReason::Boundary) => "the trial's first or last frame".to_owned(),
            Some(StopReason::EndFrame) => "the frame given with --to".to_owned(),
            Some(StopReason::InvalidPoint) => "the tracked point is not valid".to_owned(),
            Some(StopReason::Stopped) => "stopped".to_owned(),
            Some(StopReason::Failed(t)) => format!("failed: {t}"),
            None => "not finished".to_owned(),
        };
        io.line(format_args!(
            "{} steps, frame {} to {} ({reason}); {tracked} points tracked, {detected} detected, \
         {invalid} found invalid; {elapsed:.2} s",
            tracker.steps_done(),
            start + 1,
            tracker.frame() + 1
        ));
        if kept_older > 0 || removed_older > 0 {
            io.line(format_args!(
                "{kept_older} earlier points kept where the run found none, {removed_older} \
                 removed because they disagree with the run's own points"
            ));
        }
        if fitted_apart > 0 {
            io.line(format_args!(
                "{fitted_apart} points placed by fitting two touching dots apart"
            ));
        }
        let t = doc.trial(trial);
        for (m, mk) in t.markers.iter().enumerate() {
            let per_camera: Vec<String> = mk
                .views
                .iter()
                .map(|v| {
                    v.status
                        .iter()
                        .filter(|&&s| s == MarkerStatus::Tracked)
                        .count()
                        .to_string()
                })
                .collect();
            io.line(format_args!(
                "  marker {} ({}): tracked points per camera {}",
                m + 1,
                mk.description,
                per_camera.join(", ")
            ));
        }
        if let Some(StopReason::Failed(t)) = tracker.finished() {
            return Err(CliError::failed(t.clone()));
        }

        if let Some(out) = p.value("output") {
            let mut options = SaveOptions::default();
            if let Some(t) = save_time {
                options.time = t;
            }
            let report = doc
                .save_as(Path::new(out), &options)
                .map_err(|e| CliError::failed(format!("{out}: {e}")))?;
            io.line(format_args!(
                "saved {} ({} entries)",
                report.path.display(),
                report.entries
            ));
        }
        Ok(())
    })
}

/// Applies `--multi name=value,...` to the parameters.
fn set_multi_params(
    p: &mut xma_vision::joint_tracking::MultiTrackParams,
    list: &str,
) -> Result<(), CliError> {
    for item in list.split(',').filter(|i| !i.is_empty()) {
        let (name, value) = item
            .split_once('=')
            .ok_or_else(|| CliError::usage(format!("--multi: '{item}' is not name=value")))?;
        let f = || -> Result<f64, CliError> {
            value
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| {
                    CliError::usage(format!("--multi: {name}: '{value}' is not a number"))
                })
        };
        let n = || -> Result<usize, CliError> {
            value
                .parse::<usize>()
                .map_err(|_| CliError::usage(format!("--multi: {name}: '{value}' is not a count")))
        };
        match name {
            "max_reprojection_error" => p.max_reprojection_error = f()?,
            "gate" => p.gate = f()?,
            "gate_growth" => p.gate_growth = f()?,
            "gate_hold" => p.gate_hold = f()?,
            "gate_start" => p.gate_start = f()?,
            "start_confirm_near" => p.start_confirm_near = n()? != 0,
            "reacquire_dots" => p.reacquire_dots = n()? != 0,
            "reprojection_factor" => p.reprojection_factor = f()?,
            "max_reprojection_cap" => p.max_reprojection_cap = f()?,
            "rigid_neighbours" => p.rigid_neighbours = n()?,
            "rigid_fit_tolerance" => p.rigid_fit_tolerance = f()?,
            "gate_rigid" => p.gate_rigid = f()?,
            "reacquire_links" => p.reacquire_links = n()?,
            "gate_max" => p.gate_max = f()?,
            "max_gap" => p.max_gap = n()?,
            "velocity_frames" => p.velocity_frames = n()?,
            "depth_gap" => p.depth_gap = n()?,
            "velocity_window" => p.velocity_window = n()?,
            "search_margin" => p.search_margin = f()?,
            "max_candidates" => p.max_candidates = n()?,
            "min_correlation" => p.min_correlation = f()?,
            "max_detection_shift" => p.max_detection_shift = f()?,
            "same_dot" => p.same_dot = f()?,
            "min_contrast_fraction" => p.min_contrast_fraction = f()?,
            "contrast_history" => p.contrast_history = n()?,
            "contrast_relative" => p.contrast_relative = n()? != 0,
            "reacquire_exclusive" => p.reacquire_exclusive = n()? != 0,
            "claim_radius" => p.claim_radius = f()?,
            "claim_ratio" => p.claim_ratio = f()?,
            "share_needs_overlap" => p.share_needs_overlap = n()? != 0,
            "overlap_margin" => p.overlap_margin = f()?,
            "overlap_max_frames" => p.overlap_max_frames = n()?,
            "overlap_clean" => p.overlap_clean = n()? != 0,
            "overlap_decision" => p.overlap_decision = f()?,
            "overlap_take" => p.overlap_take = n()? != 0,
            "overlap_single_checked" => p.overlap_single_checked = n()? != 0,
            "ray_links" => p.ray_links = n()? != 0,
            "keep_unconfirmed" => p.keep_unconfirmed = n()? != 0,
            "keep_dot_rule" => p.keep_dot_rule = n()? != 0,
            "split_touching" => p.split_touching = n()? != 0,
            "split_depth_ratio" => p.split_depth_ratio = f()?,
            "split_rms_ratio" => p.split_rms_ratio = f()?,
            "split_min_apart" => p.split_min_apart = f()?,
            "split_max_move" => p.split_max_move = f()?,
            "split_reach_widths" => p.split_reach_widths = f()?,
            "split_reach_margin" => p.split_reach_margin = f()?,
            "stat_links" => p.stat_links = n()? != 0,
            "distance_placement" => p.distance_placement = n()? != 0,
            "single_gate" => p.single_gate = f()?,
            "single_max_frames" => p.single_max_frames = n()?,
            "single_ambiguity" => p.single_ambiguity = f()?,
            "weight_reprojection" => p.weight_reprojection = f()?,
            "weight_prediction" => p.weight_prediction = f()?,
            "weight_match" => p.weight_match = f()?,
            "weight_missing" => p.weight_missing = f()?,
            "ambiguity_margin" => p.ambiguity_margin = f()?,
            "max_combinations" => p.max_combinations = n()?,
            "share_radius" => p.share_radius = f()?,
            "contest_ratio" => p.contest_ratio = f()?,
            "rigid_min_frames" => p.rigid_min_frames = n()?,
            "rigid_max_spread" => p.rigid_max_spread = f()?,
            "rigid_sigmas" => p.rigid_sigmas = f()?,
            "rigid_min_tolerance" => p.rigid_min_tolerance = f()?,
            "rigid_history" => p.rigid_history = n()?,
            "rigid_majority" => p.rigid_majority = f()?,
            other => {
                return Err(CliError::usage(format!("--multi: no parameter '{other}'")));
            }
        }
    }
    Ok(())
}

/// One line per step: frames 1-based. A multi-camera step that kept or removed earlier
/// points says how many.
fn step_line(from: usize, r: &StepReport) -> String {
    let mut line = format!(
        "frame {} -> {}: {} tracked, {} detected, {} invalid",
        from + 1,
        r.frame + 1,
        r.tracked.len(),
        r.detected.len(),
        r.invalid.len()
    );
    if r.kept_older > 0 || r.removed_older > 0 {
        line.push_str(&format!(
            ", {} earlier kept, {} removed",
            r.kept_older, r.removed_older
        ));
    }
    if r.fitted_apart > 0 {
        line.push_str(&format!(", {} fitted apart", r.fitted_apart));
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use xma_vision::joint_tracking::MultiTrackParams;

    /// Every parameter the help lists can be set with `--multi`, and the defaults the help
    /// gives are the struct's.
    #[test]
    fn multi_parameters_in_help_match_the_defaults() {
        let list = HELP
            .split("Parameters (--multi), with their defaults:")
            .nth(1)
            .expect("parameter list")
            .split("(pixels")
            .next()
            .expect("list");
        let items: Vec<&str> = list.split_whitespace().collect();
        assert!(items.len() >= 30, "{items:?}");
        let mut p = MultiTrackParams::default();
        // Setting every parameter to the default the help names leaves the struct as it is.
        set_multi_params(&mut p, &items.join(",")).expect("help parameters parse");
        assert_eq!(p, MultiTrackParams::default());
        // And each one really sets its field.
        for item in &items {
            let (name, _) = item.split_once('=').expect("name=value");
            let changed = ["77", "0"].iter().any(|v| {
                let mut q = MultiTrackParams::default();
                set_multi_params(&mut q, &format!("{name}={v}")).expect("parses");
                q != MultiTrackParams::default()
            });
            assert!(changed, "{name}");
        }
        let mut q = MultiTrackParams::default();
        assert!(set_multi_params(&mut q, "nonsense=1").is_err());
        assert!(set_multi_params(&mut q, "gate=x").is_err());
        assert!(set_multi_params(&mut q, "gate=inf").is_err());
        assert!(set_multi_params(&mut q, "max_gap=-1").is_err());
    }
}
