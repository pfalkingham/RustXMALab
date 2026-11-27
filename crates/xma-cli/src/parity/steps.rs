//! The detection and tracker-step tables of a dump made with `--frame-step 1`
//! (`trial_<t>/detection_methods.csv`, `trial_<t>/tracking_steps.csv`; oracle/FORMAT.md,
//! "Added in version 3"), as `xma parity` compares them. The reference is the every-frame
//! dump of the reference project, `oracle/data/pig/filled/dump_clean_step1`.
//!
//! The project is opened in the XMALab 3 mode with the dump's settings, without any trial
//! update after loading (each row reads the stored points, statuses and mean sizes, as the
//! dump driver did), and every call is run on the 8-bit frame the dump wrote
//! (`frames/cam<c>_f<f>.png`):
//!
//! - `detection_methods.csv`: `detectionPoint` (`xma_vision::detection::detect_point`) at
//!   each stored point with the row's method, search area, mask size and threshold; the
//!   point, the size and, for the contour and blob methods, the number of candidates, one
//!   table row per method. Rows the dump flags `undefined` (method 3 with fewer than 50
//!   corners reads past its corner list in the C++, ledger A18) are not compared, and say so.
//! - `tracking_steps.csv`: the tracker job the tracking loop builds from the project
//!   ([`xma_engine::tracking::TrackerJob`]: prediction, template, penalty, search region,
//!   match, tracked point), then the detection that follows at the tracked point
//!   ([`xma_engine::tracking::DetectionJob`]: search area, mask size, threshold, point,
//!   size, number of candidates), by the marker's method. The detection is not compared on
//!   rows the dump flags `undefined`, or for method 4, which does not detect.
//!
//! Every value is compared exactly.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use xma_engine::tracking::{
    DetectionJob, TrackerJob, algorithm_image, detect_options, track_compat,
};
use xma_engine::{Document, OpenOptions};
use xma_model::{CppCompat, MarkerStatus};
use xma_oracle::Dataset;
use xma_vision::detection::{DetectionMethod, detect_point};
use xma_vision::image::ImageU8;

use super::{Check, Item, R, Tier, rf, ri, ru};

const RULE_A18: &str =
    "not compared: undefined behaviour in the C++ (method 3, fewer than 50 corners), ledger A18";
const RULE_NO_DETECTION: &str = "not compared: method 4 does not detect";

/// The tables of trial `t` this module compares, of those the dump holds.
pub(super) fn tables(ds: &Dataset, t: usize) -> Vec<String> {
    ["detection_methods.csv", "tracking_steps.csv"]
        .iter()
        .map(|f| format!("trial_{t}/{f}"))
        .filter(|r| ds.has(r))
        .collect()
}

/// Opens the project as the dump driver read it.
fn open(ds: &Dataset) -> Result<Document, String> {
    let manifest = ds.try_manifest().map_err(|e| e.to_string())?;
    let options = OpenOptions {
        overrides: crate::settings::manifest_settings(&manifest)?,
        compat: CppCompat::XMALAB3,
        ..OpenOptions::default()
    };
    let loaded = xma_engine::read_project(&ds.project, &options)
        .map_err(|e| format!("{}: {e}", ds.project.display()))?;
    let mut doc = Document::from_loaded(loaded);
    doc.set_path(&ds.project);
    Ok(doc)
}

/// The frames the dump wrote, read on demand and kept a while.
struct Frames {
    dir: std::path::PathBuf,
    cache: HashMap<(usize, usize), Arc<ImageU8>>,
}

impl Frames {
    fn get(&mut self, camera: usize, frame: usize) -> R<Arc<ImageU8>> {
        if let Some(i) = self.cache.get(&(camera, frame)) {
            return Ok(i.clone());
        }
        if self.cache.len() > 64 {
            self.cache.clear();
        }
        let path = self.dir.join(format!("frames/cam{camera}_f{frame}.png"));
        let f = xma_video::image_file::read_image(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let img = Arc::new(algorithm_image(&f));
        self.cache.insert((camera, frame), img.clone());
        Ok(img)
    }
}

fn method_name(code: i64) -> String {
    match DetectionMethod::from_code(code as i32) {
        Some(m) => format!("method {code} ({m:?})"),
        None => format!("method {code}"),
    }
}

/// Compares the tables of trial `t` (see the module text).
pub(super) fn check_step_tables(ds: &Dataset, t: usize) -> Vec<Item> {
    let rels = tables(ds, t);
    if rels.is_empty() {
        return Vec::new();
    }
    let doc = match open(ds) {
        Ok(d) => d,
        Err(e) => {
            let mut c = Check::new(rels[0].clone(), Tier::Read);
            c.problem(format!("the project cannot be opened: {e}"));
            return vec![c.finish()];
        }
    };
    let mut frames = Frames {
        dir: ds.trial_dir(t),
        cache: HashMap::new(),
    };
    let mut items = Vec::new();
    for rel in rels {
        if rel.ends_with("detection_methods.csv") {
            items.extend(detection_methods(ds, &doc, &rel, &mut frames));
        } else {
            items.extend(tracking_steps(ds, &doc, t, &rel, &mut frames));
        }
    }
    items
}

/// `detection_methods.csv`, one item per method.
fn detection_methods(ds: &Dataset, doc: &Document, rel: &str, frames: &mut Frames) -> Vec<Item> {
    let mut head = Check::new(rel, Tier::Read);
    let Some(table) = head.table(ds) else {
        return vec![head.finish()];
    };
    let options = detect_options(doc);
    // Rows by method, then by (camera, frame) so that each frame is read once per method.
    let mut by_method: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for r in table.rows() {
        match ri(&r, "method") {
            Ok(m) => by_method.entry(m).or_default().push(r.index()),
            Err(e) => head.problem(format!("row {}: {e}", r.index() + 1)),
        }
    }
    let mut items = Vec::new();
    if !head.problems.is_empty() {
        items.push(head.finish());
    }
    for (code, rows) in by_method {
        let mut c = Check::new(format!("{rel}, {}", method_name(code)), Tier::Read);
        let method = DetectionMethod::from_code(code as i32);
        let (mut undefined, mut no_detection) = (0, 0);
        let mut ordered = rows.clone();
        ordered.sort_by_key(|&i| {
            let r = table.row(i);
            (ri(&r, "camera").unwrap_or(0), ri(&r, "frame").unwrap_or(0))
        });
        for i in ordered {
            let r = table.row(i);
            let res = (|| -> R<()> {
                if ri(&r, "undefined")? != 0 {
                    undefined += 1;
                    return Ok(());
                }
                let Some(method) = method.filter(|m| *m != DetectionMethod::NoDetection) else {
                    no_detection += 1;
                    return Ok(());
                };
                let (cam, f) = (ru(&r, "camera")?, ru(&r, "frame")?);
                let image = frames.get(cam, f)?;
                let d = detect_point(
                    &image,
                    method,
                    (rf(&r, "in_x")?, rf(&r, "in_y")?),
                    ri(&r, "search_area")? as i32,
                    ri(&r, "mask_size")? as i32,
                    rf(&r, "threshold")?,
                    &options,
                );
                let l = format!("marker {} camera {cam} frame {f}", ru(&r, "marker")?);
                c.num(format_args!("{l} out_x"), d.point.0, rf(&r, "out_x")?);
                c.num(format_args!("{l} out_y"), d.point.1, rf(&r, "out_y")?);
                c.num(format_args!("{l} out_size"), d.size, rf(&r, "out_size")?);
                // The C++ fills its candidate lists for the contour and blob methods only.
                if method != DetectionMethod::Corner {
                    c.int(
                        format_args!("{l} n_found"),
                        d.all_centres.len() as i64,
                        ri(&r, "n_found")?,
                    );
                }
                Ok(())
            })();
            if let Err(e) = res {
                c.problem(format!("row {}: {e}", i + 1));
            }
        }
        let n = rows.len();
        if undefined > 0 {
            c.exclude(
                RULE_A18,
                3 * undefined,
                format!(
                    "out_x, out_y, out_size of {undefined} of {n} rows not compared: the dump flags them undefined (method \
                     3 with fewer than 50 corners reads past the end of its corner list, \
                     MarkerDetection.cpp:333; ledger A18) and gives no output"
                ),
            );
        }
        if no_detection > 0 {
            c.exclude(
                RULE_NO_DETECTION,
                3 * no_detection,
                format!("out_x, out_y, out_size of {no_detection} rows of method 4 not compared: it does not detect"),
            );
        }
        let mut item = c.finish();
        item.name = format!("{} ({n} rows)", item.name);
        items.push(item);
    }
    items
}

/// `tracking_steps.csv`: the tracker, then the detection after it by method.
#[allow(clippy::too_many_lines)]
fn tracking_steps(
    ds: &Dataset,
    doc: &Document,
    t: usize,
    rel: &str,
    frames: &mut Frames,
) -> Vec<Item> {
    let mut tracker = Check::new(format!("{rel}, tracker"), Tier::Read);
    let table = match ds.try_table(rel) {
        Ok(t) => t,
        Err(e) => {
            tracker.problem(format!("cannot read the reference: {e}"));
            return vec![tracker.finish()];
        }
    };
    let tc = track_compat(doc.compat());
    let options = detect_options(doc);
    let wraps = doc.compat().threshold_offset_wraps;
    let markers = &doc.trial(t).markers;
    let mut detection: BTreeMap<i64, (Check, usize, usize, usize)> = BTreeMap::new();
    let mut ordered: Vec<usize> = (0..table.len()).collect();
    ordered.sort_by_key(|&i| {
        let r = table.row(i);
        (
            ri(&r, "camera").unwrap_or(0),
            ri(&r, "frame_from").unwrap_or(0),
        )
    });
    for i in ordered {
        let r = table.row(i);
        let res = (|| -> R<()> {
            let (m, c) = (ru(&r, "marker")?, ru(&r, "camera")?);
            let (from, to) = (ru(&r, "frame_from")?, ru(&r, "frame_to")?);
            let forward = ri(&r, "direction")? > 0;
            let marker = markers.get(m).ok_or_else(|| format!("no marker {m}"))?;
            let job = TrackerJob::new(marker, m, c, from, to, forward, &*frames.get(c, from)?);
            let res = job.run(marker, &*frames.get(c, to)?, tc);
            let l = format!("marker {m} camera {c} frame {from} -> {to}");
            let s = res.step.ok_or_else(|| format!("{l}: no step"))?;
            let got: [(&str, f64); 12] = [
                ("prediction", f64::from(res.prediction)),
                ("predicted_x", res.predicted.x),
                ("predicted_y", res.predicted.y),
                ("template_size", f64::from(job.template.size)),
                ("penalty_used", f64::from(s.max_penalty)),
                ("off_x", f64::from(s.off_x)),
                ("off_y", f64::from(s.off_y)),
                ("match_x", s.max_loc.0 as f64),
                ("match_y", s.max_loc.1 as f64),
                ("max_value", f64::from(s.max_val)),
                ("tracked_x", res.point.x),
                ("tracked_y", res.point.y),
            ];
            for (col, v) in got {
                tracker.num(format_args!("{l} {col}"), v, rf(&r, col)?);
            }
            let code = ri(&r, "method")?;
            let e = detection.entry(code).or_insert_with(|| {
                (
                    Check::new(
                        format!("{rel}, detection after the step, {}", method_name(code)),
                        Tier::Read,
                    ),
                    0,
                    0,
                    0,
                )
            });
            e.1 += 1;
            if ri(&r, "undefined")? != 0 {
                e.2 += 1;
                return Ok(());
            }
            if code == 4 {
                e.3 += 1;
                return Ok(());
            }
            // As the loop builds it after the tracker's setPoint.
            let mut mk = marker.as_ref().clone();
            mk.views[c].points[to] = res.point;
            mk.views[c].status[to] = MarkerStatus::Tracked;
            let d = DetectionJob::new(&mk, m, c, to, mk.mean_size * 2.0, true, wraps);
            let out = d.run(&*frames.get(c, to)?, &options);
            let dc = &mut e.0;
            dc.int(
                format_args!("{l} search_area"),
                d.search_area.into(),
                ri(&r, "search_area")?,
            );
            dc.int(
                format_args!("{l} mask_size"),
                d.mask_size.into(),
                ri(&r, "mask_size")?,
            );
            dc.num(
                format_args!("{l} threshold"),
                d.threshold,
                rf(&r, "threshold")?,
            );
            dc.num(format_args!("{l} out_x"), out.point.x, rf(&r, "out_x")?);
            dc.num(format_args!("{l} out_y"), out.point.y, rf(&r, "out_y")?);
            dc.num(format_args!("{l} out_size"), out.size, rf(&r, "out_size")?);
            if code != 3
                && let Some(det) = &out.detection
            {
                dc.int(
                    format_args!("{l} n_found"),
                    det.all_centres.len() as i64,
                    ri(&r, "n_found")?,
                );
            }
            Ok(())
        })();
        if let Err(e) = res {
            tracker.problem(format!("row {}: {e}", i + 1));
        }
    }
    let mut item = tracker.finish();
    item.name = format!("{} ({} rows)", item.name, table.len());
    let mut items = vec![item];
    for (_, (mut c, n, undefined, no_detection)) in detection {
        if undefined > 0 {
            c.exclude(
                RULE_A18,
                6 * undefined,
                format!(
                    "search_area, mask_size, threshold, out_x, out_y, out_size of {undefined} of \
                     {n} steps not compared: the dump flags the detection undefined"
                ),
            );
        }
        if no_detection > 0 {
            c.exclude(
                RULE_NO_DETECTION,
                6 * no_detection,
                format!(
                    "search_area, mask_size, threshold, out_x, out_y, out_size of \
                     {no_detection} of {n} steps not compared: method 4 does not detect"
                ),
            );
        }
        let mut item = c.finish();
        item.name = format!("{} ({n} rows)", item.name);
        items.push(item);
    }
    items
}
