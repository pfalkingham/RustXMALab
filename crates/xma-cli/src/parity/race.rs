//! The other orders of a C++ data race (parity ledger, item A56), replayed for `xma parity`.
//!
//! # The race
//!
//! `ThreadScheduler::updateTrialData` computes every frame of a trial update on worker
//! threads (`processing/UpdateTrialFrame.cpp:62-70`: every body's `poseComputed[frame]` is
//! cleared, then `computePose(frame)` runs for each body in index order). The dump driver
//! waits for an update by pumping the interface's event loop
//! (`OracleDumpDriver::recomputeTrial` and `waitIdle` call `QApplication::processEvents`
//! until the scheduler is idle), so the interface thread can draw the trial during **any**
//! update. The frame drawn is the trial's active frame (`Trial::drawRigidBodies`,
//! core/Trial.cpp:561-566), which is 0 throughout a dump: nothing in the driver changes it
//! (`writeTrialImages` moves the video streams only and restores them,
//! OracleDumpDriver.cpp:1581,1736).
//!
//! `RigidBody::draw2D` of a visible body calls `transformPoint` on the body each of its
//! virtual points refers to (core/RigidBody.cpp:2692, and again through `projectToImage`,
//! `:2405`), and `transformPoint` runs `computePose(frame)` when that body's flag is clear
//! (`:297-300`); its `computePose` does the same for the bodies its own virtual points refer
//! to (`:1099`). So every body reachable from a visible body's virtual points, directly or
//! through a chain, can be computed by both threads at frame 0 ([`raceable`]). The
//! refinement's constructor reads the held pose outside the mutex that serialises levmar
//! (`processing/RigidBodyPoseOptimization.cpp:49,187-203,255`): when one thread sets up after
//! the other has written its refined pose, it refines that pose again, and the body ends the
//! update holding the pose refined twice, with the errors of that pose or of the once-refined
//! one (`updateError` reads the held pose when it runs). A body computed after it (by index,
//! or on demand) that refers to it through a virtual point reads whichever pose is held at
//! that moment.
//!
//! With the refinement off (`OptimizeRigidBody` false) both threads compute the same values
//! from the same inputs, so the race changes nothing: an update with the refinement off has
//! no alternatives.
//!
//! # What is replayed
//!
//! For an update whose results the dump holds, a [`World`] is the document after that
//! update when, at frame 0, a set of raceable bodies (every non-empty subset, up to three
//! raceable bodies; each one alone and all together beyond that) ends with its pose refined
//! once more ([`xma_engine::rigid_bodies::refine_again`], in index order), and either no
//! other body or every body that depends on one of them through virtual points is computed
//! again from the raced poses. The replay changes the states the update computed before they
//! are written ([`xma_engine::UpdateHook`]), so the scheduler's finalize (rotation continuity,
//! filtering in both modes, filtered errors) and everything after it (the rigid-body
//! exports' recomputation, ledger A36) run on them exactly as on the unraced states.

use std::cell::{Cell, Ref, RefCell, RefMut};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use xma_engine::rigid_bodies::{compute_pose, refine_again};
use xma_engine::{Document, Export, FrameStates, UpdateHook};
use xma_model::{Trial, VirtualPointSource};

/// The frame on screen during every update of a dump (see the module documentation).
pub(super) const SHOWN_FRAME: usize = 0;

/// Raceable bodies beyond which only single bodies and the whole set are replayed.
const ALL_SUBSETS_UP_TO: usize = 3;

/// The bodies a body's virtual points refer to (in range).
fn targets(trial: &Trial, b: usize) -> impl Iterator<Item = usize> + '_ {
    let n = trial.rigid_bodies.len();
    trial.rigid_bodies[b]
        .virtual_points
        .iter()
        .filter_map(move |v| match v.source {
            VirtualPointSource::OtherBody { body, .. } if body < n => Some(body),
            _ => None,
        })
}

/// The bodies the interface thread can compute at the frame shown: those a visible body's
/// virtual point refers to, and, through their own virtual points, the bodies those refer
/// to, transitively.
#[must_use]
pub(super) fn raceable(trial: &Trial) -> Vec<usize> {
    let mut found = BTreeSet::new();
    let mut stack: Vec<usize> = (0..trial.rigid_bodies.len())
        .filter(|&b| trial.rigid_bodies[b].visible)
        .flat_map(|b| targets(trial, b).collect::<Vec<_>>())
        .collect();
    while let Some(b) = stack.pop() {
        if found.insert(b) {
            stack.extend(targets(trial, b));
        }
    }
    found.into_iter().collect()
}

/// The bodies outside `raced` that refer to one of them through virtual points, directly or
/// through other bodies.
fn consumers(trial: &Trial, raced: &[usize]) -> Vec<usize> {
    let n = trial.rigid_bodies.len();
    let mut reaches: BTreeSet<usize> = raced.iter().copied().collect();
    loop {
        let more: Vec<usize> = (0..n)
            .filter(|b| !reaches.contains(b) && targets(trial, *b).any(|x| reaches.contains(&x)))
            .collect();
        if more.is_empty() {
            break;
        }
        reaches.extend(more);
    }
    reaches.into_iter().filter(|b| !raced.contains(b)).collect()
}

/// One other order of the race in one update.
pub(super) struct World {
    /// Bodies whose pose at frame 0 was refined twice.
    pub raced: Vec<usize>,
    /// Bodies computed again from the raced poses (empty: they read the unraced ones).
    pub consumers: Vec<usize>,
    /// The document after the update in this order (and, for the update before the as-held
    /// tables, after the rigid-body exports' recomputation once [`World::exports`] is
    /// filled).
    pub doc: Document,
    /// Export files as this order writes them, by path relative to the dump.
    pub exports: BTreeMap<String, String>,
    /// Exports that failed in this order: the error, by path relative to the dump.
    pub export_errors: BTreeMap<String, String>,
    /// The cell of the dump that shows this order exactly (a raced body's own value at
    /// frame 0 that differs from this program's), when one does.
    pub established: Option<String>,
}

impl World {
    /// How the report names this order.
    pub fn describe(&self) -> String {
        let list = |v: &[usize]| {
            v.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut s = format!(
            "{} {} refined twice at frame {SHOWN_FRAME}",
            if self.raced.len() == 1 {
                "body"
            } else {
                "bodies"
            },
            list(&self.raced)
        );
        if !self.consumers.is_empty() {
            s += &format!(
                ", {} {} computed from {}",
                if self.consumers.len() == 1 {
                    "body"
                } else {
                    "bodies"
                },
                list(&self.consumers),
                if self.raced.len() == 1 { "it" } else { "them" }
            );
        }
        s
    }
}

/// Replays an update on a document.
type Step = dyn Fn(&mut Document) + Send + Sync;

/// The other orders of the race in one update, built when first needed: only when a cell
/// the race can change differs from this program's value, so a dump that agrees costs
/// nothing.
pub(super) struct Race {
    /// Which update, as the report names it.
    pub update: &'static str,
    /// The trial, the document before the update, and the step that replays it; `None` when
    /// the update cannot race.
    source: Option<(usize, Document, Box<Step>)>,
    worlds: RefCell<Option<Vec<World>>>,
    /// Whether the rigid-body exports have been replayed in every world.
    exported: Cell<bool>,
}

impl Race {
    /// No alternative: the update cannot race, or nothing it determines is compared.
    pub fn none() -> Self {
        Self {
            update: "",
            source: None,
            worlds: RefCell::new(Some(Vec::new())),
            exported: Cell::new(true),
        }
    }

    /// The other orders of the update `step` runs on trial `t`, from `base`, the document
    /// before it (with the settings the update uses). None when the refinement is off or no
    /// body of the trial is raceable.
    pub fn new(
        update: &'static str,
        base: &Document,
        t: usize,
        step: impl Fn(&mut Document) + Send + Sync + 'static,
    ) -> Self {
        if !base.settings().optimize_rigid_body
            || t >= base.project().trials.len()
            || raceable(base.trial(t)).is_empty()
        {
            return Self::none();
        }
        Self {
            update,
            source: Some((t, base.clone(), Box::new(step))),
            worlds: RefCell::new(None),
            exported: Cell::new(false),
        }
    }

    /// A race whose orders are `worlds`, already built (for tests of the comparisons).
    #[cfg(test)]
    pub fn with_worlds(base: Document, worlds: Vec<World>) -> Self {
        Self {
            update: "test",
            source: Some((0, base, Box::new(|_: &mut Document| {}))),
            worlds: RefCell::new(Some(worlds)),
            exported: Cell::new(true),
        }
    }

    /// True when the update can race (there may be worlds to build).
    pub fn possible(&self) -> bool {
        self.source.is_some()
    }

    /// The orders of the race, built on first use.
    pub fn worlds(&self) -> Ref<'_, [World]> {
        self.build();
        Ref::map(self.worlds.borrow(), |w| w.as_deref().unwrap_or_default())
    }

    /// The orders of the race, built on first use, to change.
    pub fn worlds_mut(&self) -> RefMut<'_, Vec<World>> {
        self.build();
        RefMut::map(self.worlds.borrow_mut(), |w| w.get_or_insert_with(Vec::new))
    }

    /// Replays, once, the exports that change state (`exports`, in the driver's order) in
    /// every world, keeping each file's text under its path.
    pub fn replay_exports(&self, t: usize, exports: &[(String, Export)]) {
        if self.exported.replace(true) {
            return;
        }
        let mut worlds = self.worlds_mut();
        std::thread::scope(|scope| {
            for w in worlds.iter_mut() {
                scope.spawn(move || {
                    for (rel, export) in exports {
                        match w.doc.export(t, export) {
                            Ok(out) => {
                                w.exports.insert(rel.clone(), super::file(out));
                            }
                            Err(e) => {
                                w.export_errors.insert(rel.clone(), e.to_string());
                            }
                        }
                    }
                });
            }
        });
    }

    /// The orders, among those built, whose export `rel` failed: how the report names the
    /// order, and the error.
    pub fn export_failures(&self, rel: &str) -> Vec<(String, String)> {
        let worlds = self.worlds.borrow();
        worlds
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter_map(|w| w.export_errors.get(rel).map(|e| (w.describe(), e.clone())))
            .collect()
    }

    fn build(&self) {
        if self.worlds.borrow().is_some() {
            return;
        }
        let Some((t, base, step)) = &self.source else {
            *self.worlds.borrow_mut() = Some(Vec::new());
            return;
        };
        let t = *t;
        let trial = base.trial(t);
        let r = raceable(trial);
        let mut subsets: Vec<Vec<usize>> = Vec::new();
        if r.len() <= ALL_SUBSETS_UP_TO {
            for mask in 1u32..(1 << r.len()) {
                subsets.push(
                    r.iter()
                        .enumerate()
                        .filter(|(i, _)| mask & (1 << i) != 0)
                        .map(|(_, b)| *b)
                        .collect(),
                );
            }
        } else {
            subsets.extend(r.iter().map(|b| vec![*b]));
            subsets.push(r.clone());
        }
        // Every world replays the same update from the same document, so the rigid-body
        // step's states are computed once and shared.
        let cache: Arc<Mutex<Option<Vec<FrameStates>>>> = Arc::new(Mutex::new(None));
        let mut orders = Vec::new();
        for raced in subsets {
            let cons = consumers(trial, &raced);
            orders.push((raced.clone(), Vec::new()));
            if !cons.is_empty() {
                orders.push((raced, cons));
            }
        }
        // The orders are independent: replayed in parallel.
        let worlds: Vec<World> = std::thread::scope(|scope| {
            let handles: Vec<_> = orders
                .into_iter()
                .map(|(raced, consumers)| {
                    let cache = Arc::clone(&cache);
                    scope.spawn(move || {
                        let changed = Arc::new(AtomicBool::new(false));
                        let mut doc = base.clone();
                        doc.set_update_hook(Some(hook(
                            t,
                            raced.clone(),
                            consumers.clone(),
                            Arc::clone(&changed),
                            cache,
                        )));
                        step(&mut doc);
                        doc.set_update_hook(None);
                        changed.load(Ordering::Relaxed).then(|| World {
                            raced,
                            consumers,
                            doc,
                            exports: BTreeMap::new(),
                            export_errors: BTreeMap::new(),
                            established: None,
                        })
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|h| h.join().expect("a replay of the race panicked"))
                .collect()
        });
        *self.worlds.borrow_mut() = Some(worlds);
    }
}

/// The hook that turns an update into one order of the race: at frame 0 of trial `t`, each
/// body of `raced` refined once more, then the bodies of `consumers` computed again.
fn hook(
    t: usize,
    raced: Vec<usize>,
    consumers: Vec<usize>,
    changed: Arc<AtomicBool>,
    cache: Arc<Mutex<Option<Vec<FrameStates>>>>,
) -> UpdateHook {
    UpdateHook(Arc::new(move |trial_index, ctx, trial, step| {
        if trial_index != t {
            return step();
        }
        let mut states = {
            let mut cached = cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            cached.get_or_insert_with(step).clone()
        };
        let Some(s) = states.get_mut(SHOWN_FRAME) else {
            return states;
        };
        let mut any = false;
        for &r in &raced {
            any |= refine_again(ctx, trial, s, r);
        }
        if any {
            changed.store(true, Ordering::Relaxed);
            for &c in &consumers {
                s.bodies[c].computed = false;
            }
            for &c in &consumers {
                if !s.bodies[c].computed {
                    compute_pose(ctx, trial, s, c);
                }
            }
        }
        states
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xma_model::{Point3, RigidBody, VirtualPoint};

    fn vp(to: usize) -> VirtualPoint {
        VirtualPoint {
            name: "v".to_owned(),
            reference: Point3::new(0.0, 0.0, 0.0),
            source: VirtualPointSource::OtherBody {
                body: to,
                point: Point3::new(0.0, 0.0, 0.0),
            },
        }
    }

    /// rv_vpchain's graph: 0 -> 2, 1 -> 0, 3 -> 1, and an out-of-range reference.
    fn chain(visible: bool) -> Trial {
        let mut t = Trial::default_trial();
        for refs in [vec![2], vec![0], vec![], vec![1, 9]] {
            let mut b = RigidBody::new(1);
            b.visible = visible;
            b.virtual_points = refs.into_iter().map(vp).collect();
            t.rigid_bodies.push(Arc::new(b));
        }
        t
    }

    #[test]
    fn raceable_bodies_follow_chains_from_visible_bodies() {
        assert_eq!(raceable(&chain(true)), vec![0, 1, 2]);
        assert!(raceable(&chain(false)).is_empty());
        let mut t = chain(false);
        Arc::make_mut(&mut t.rigid_bodies[1]).visible = true;
        assert_eq!(raceable(&t), vec![0, 2]);
    }

    #[test]
    fn consumers_are_the_bodies_that_reach_a_raced_one() {
        let t = chain(true);
        assert_eq!(consumers(&t, &[2]), vec![0, 1, 3]);
        assert_eq!(consumers(&t, &[0]), vec![1, 3]);
        assert_eq!(consumers(&t, &[1]), vec![3]);
        assert_eq!(consumers(&t, &[0, 1, 2]), vec![3]);
    }
}
