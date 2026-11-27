//! Start-up timing for measurements: with `XMALAB_RS_TIMING` set, the program writes to
//! standard error, once each, how long after `main` started the window drew its first frame,
//! the project was open, and every visible camera's first image was in the texture store
//! (drawn in the same frame). Nothing is written without the variable.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

/// The variable that turns the timing lines on.
pub const TIMING_ENV: &str = "XMALAB_RS_TIMING";

static START: OnceLock<Instant> = OnceLock::new();
static DONE: AtomicU8 = AtomicU8::new(0);

/// A milestone, written once.
#[derive(Clone, Copy, Debug)]
pub enum Milestone {
    FirstFrame = 1,
    ProjectOpen = 2,
    ImagesShown = 4,
}

/// Records the start (`main`). Later calls do nothing.
pub fn mark_start() {
    let _ = START.set(Instant::now());
}

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(TIMING_ENV).is_some())
}

/// Writes the milestone's time since [`mark_start`] the first time it is reached.
pub fn reached(m: Milestone) {
    if !enabled() {
        return;
    }
    let bit = m as u8;
    if DONE.fetch_or(bit, Ordering::Relaxed) & bit != 0 {
        return;
    }
    let t = START.get().map_or(0.0, |s| s.elapsed().as_secs_f64());
    eprintln!("rustxmalab timing: {m:?} at {:.3} s", t);
}

/// Checks the milestones after a frame of the main window.
pub fn after_frame(app: &crate::XmaApp) {
    if !enabled() {
        return;
    }
    reached(Milestone::FirstFrame);
    if app.doc().is_some() {
        reached(Milestone::ProjectOpen);
        let cams = &app.frames.cameras;
        if !cams.is_empty()
            && cams.iter().any(|c| c.want.is_some())
            && cams
                .iter()
                .all(|c| c.want.is_none() || c.shown == c.want || c.failed || c.message.is_some())
        {
            reached(Milestone::ImagesShown);
        }
    }
}
