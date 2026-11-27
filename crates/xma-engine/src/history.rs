//! Undo, redo and the modified state of a [`Document`].
//!
//! Not a port: XMALab 3 has neither. Its Undo restores one 2D point and is cleared by any
//! change of marker, camera, frame or trial, and it never knows whether a project has
//! unsaved changes (`docs/cpp-editing.md`, sections 2.4 and 12.1; parity ledger U52, U53).
//!
//! # How it works
//!
//! The model shares its parts by reference count (`Arc`: cameras, trials, markers, rigid
//! bodies), so a copy of the whole [`Project`] costs a few pointer copies and an edit
//! duplicates only the parts it writes. A **step** is one user action. Opening a step keeps
//! a copy of the project as it was, with what goes back with it (the cameras' derived state
//! and the session values listed in [`crate::session`], ledger A159); committing it puts
//! that copy on the undo stack when the project or that state differs from the copy (the
//! scheduler's queue alone does not count), or when the step wrote the loader's extras.
//! Undo swaps the current state with the copy.
//!
//! - [`Document::transact`] runs a closure as one step and takes it back if the closure
//!   fails. Everything in [`crate::edit`] goes through it.
//! - [`Document::begin_step`], [`Document::commit_step`] and [`Document::abort_step`] are the
//!   same thing in pieces, for a step that spans many calls (a tracking run that the
//!   interface advances frame by frame).
//! - Steps nest: a step opened inside another is part of the outer one. A
//!   [`Document::transact`] inside an open step that fails puts the document back as it was
//!   when that nested call began (the outer step's earlier work stays), so a failed edit
//!   changes nothing whether or not a step is open.
//! - Committing the outermost step first brings derived data up to date
//!   ([`Document::settle`]), so an undo entry never holds half-updated poses.
//!
//! # Modified state
//!
//! Every recorded step gives the document a new revision number; a copy on the stacks
//! remembers the revision it holds. The document is modified when its revision is not the
//! one last saved, so undoing back to the saved state shows it unmodified again. A step
//! that changes only derived or session state (a grid refit, a pending flag) is recorded
//! and so marks the document modified, although the file it would save is the same
//! (ledger A159).
//!
//! # What is not on the stack
//!
//! Scale and bias of the camera views ([`Document::set_image_adjustment`]) are saved with
//! the project (ledger U46) and mark it modified, but they are view settings moved by
//! sliders: they are not undo steps, and undo and redo leave them as they are. A state
//! restored with scale and bias other than its snapshot's gets a new revision, so a
//! revision always names one set of saved values: undoing back to the saved revision after
//! a scale change leaves the document modified (review 12).
//!
//! Caches of derived state (ledger A166): a step on the undo or redo stack keeps no grid
//! lookup table, and no grid model whose source it can name, that the document does not
//! hold now (`Computed::drop_lookup_tables`, after every commit, undo and redo): the model
//! is fitted again from the camera's stored grid points it was fitted from
//! (`DerivedCamera::grid_source`, about 110 ms per camera on the pig data) and the table made
//! again from the model (about 180 ms), when the step is restored, bit for bit what the step
//! held. An undo or redo of a grid refit therefore takes about 0.3 s longer per camera; the
//! views show the restored table at once. Without this the stacks kept a table (8 MiB per
//! camera) and a model (0.8 MiB) per refit: 923.5 MiB after 200 calibration edits on the
//! pig data, 31.2 MiB now (`fixcal3_undo_memory.rs`).

use std::collections::BTreeSet;
use std::sync::Arc;

use xma_model::{Camera, Project, RigidBody, Trial};

use crate::document::{Computed, Document, DroppedTables, LoadedExtras, Scheduler};
use crate::edit::EditError;

/// Number of undo steps kept by default.
pub const DEFAULT_UNDO_LIMIT: usize = 200;

/// The state an undo or redo step returns to.
#[derive(Clone, Debug)]
struct Snapshot {
    project: Project,
    scheduler: Scheduler,
    /// The cameras' derived state and the session's recalibration flags
    /// ([`crate::document::Computed`]): they describe how the project relates to what was
    /// computed from it, so they go back with it.
    computed: Computed,
    /// The loader's extras (attachments, trial sources), kept only by a step that changed
    /// them ([`Document::extras_mut`]); they hold file contents and are costly to copy.
    extras: Option<Box<LoadedExtras>>,
    revision: u64,
    /// The cameras whose grid lookup table this step does not keep (ledger A166), with the
    /// table's size: made again when the step is restored.
    dropped: DroppedTables,
}

#[derive(Clone, Debug)]
struct Step {
    label: String,
    state: Snapshot,
}

#[derive(Clone, Debug)]
struct OpenStep {
    label: String,
    before: Snapshot,
    /// Nesting depth; 1 for the outermost step.
    depth: usize,
    /// One entry per [`Document::transact`] running inside this step, innermost last: the
    /// state when it began, put back if it fails.
    nested: Vec<NestedStart>,
}

/// What a nested [`Document::transact`] puts back when it fails: the document as it was
/// when the nested step began, without the outer step's earlier work being taken back.
#[derive(Clone, Debug)]
struct NestedStart {
    project: Project,
    scheduler: Scheduler,
    computed: Computed,
    stale: BTreeSet<(usize, usize)>,
    /// The loader's extras, kept by the first [`Document::extras_mut`] inside the nested
    /// step only.
    extras: Option<Box<LoadedExtras>>,
}

/// The undo and redo stacks and the revision counters of a document.
#[derive(Clone, Debug)]
pub struct History {
    undo: Vec<Step>,
    redo: Vec<Step>,
    open: Option<OpenStep>,
    /// Revision of the document's present state.
    revision: u64,
    /// The last revision number handed out.
    last_revision: u64,
    /// The revision last saved or loaded; `None` when no state of the document is on disk.
    saved_revision: Option<u64>,
    limit: usize,
    /// Moves with every change of the document's content ([`Document::changes`]).
    changes: u64,
}

impl Default for History {
    fn default() -> Self {
        History {
            undo: Vec::new(),
            redo: Vec::new(),
            open: None,
            revision: 0,
            last_revision: 0,
            saved_revision: Some(0),
            limit: DEFAULT_UNDO_LIMIT,
            changes: 0,
        }
    }
}

impl History {
    fn new_revision(&mut self) -> u64 {
        self.last_revision += 1;
        self.revision = self.last_revision;
        self.revision
    }
}

/// True when two lists of shared parts hold the same values. Parts that are the same
/// allocation are not compared.
fn same_parts<T: PartialEq>(a: &[Arc<T>], b: &[Arc<T>]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| Arc::ptr_eq(x, y) || **x == **y)
}

fn same_trial(a: &Arc<Trial>, b: &Arc<Trial>) -> bool {
    if Arc::ptr_eq(a, b) {
        return true;
    }
    // Every field is named so that a new field cannot be forgotten here.
    let Trial {
        name,
        videos,
        markers,
        rigid_bodies,
        events,
        n_frames,
        start_frame,
        end_frame,
        reference_calibration_image,
        recording_speed,
        cutoff_frequency,
        interpolate_3d,
        is_default,
        is_copy_from_default,
        requires_recomputation,
        portal_metadata,
    } = &**a;
    same_parts(markers, &b.markers)
        && same_parts::<RigidBody>(rigid_bodies, &b.rigid_bodies)
        && *name == b.name
        && *videos == b.videos
        && *events == b.events
        && *n_frames == b.n_frames
        && *start_frame == b.start_frame
        && *end_frame == b.end_frame
        && *reference_calibration_image == b.reference_calibration_image
        && *recording_speed == b.recording_speed
        && *cutoff_frequency == b.cutoff_frequency
        && *interpolate_3d == b.interpolate_3d
        && *is_default == b.is_default
        && *is_copy_from_default == b.is_copy_from_default
        && *requires_recomputation == b.requires_recomputation
        && *portal_metadata == b.portal_metadata
}

/// True when two projects hold the same values. Shared parts are not compared, so this is
/// cheap after an edit that touched little.
#[must_use]
pub fn same_project(a: &Project, b: &Project) -> bool {
    let Project {
        cameras,
        trials,
        calibration_type,
        calibration_object,
        algorithm_settings,
        date_created,
        active_trial,
        portal_metadata,
    } = a;
    same_parts::<Camera>(cameras, &b.cameras)
        && trials.len() == b.trials.len()
        && trials.iter().zip(&b.trials).all(|(x, y)| same_trial(x, y))
        && *calibration_type == b.calibration_type
        && *calibration_object == b.calibration_object
        && *algorithm_settings == b.algorithm_settings
        && *date_created == b.date_created
        && *active_trial == b.active_trial
        && *portal_metadata == b.portal_metadata
}

impl Document {
    fn snapshot(&self, with_extras: bool) -> Snapshot {
        Snapshot {
            project: self.project.clone(),
            scheduler: self.scheduler_state().clone(),
            computed: self.computed_state(),
            extras: with_extras.then(|| Box::new(self.extras.clone())),
            revision: self.history.revision,
            dropped: Vec::new(),
        }
    }

    /// Ledger A166: the undo and redo steps keep no grid lookup table that the document does
    /// not hold now ([`Computed::drop_lookup_tables`]); a step's tables are made again when
    /// it is restored. Without this every refit kept the previous table (8 MiB per camera
    /// on the pig data) for as long as its step stayed on a stack.
    fn drop_step_lookup_tables(&mut self) {
        let live = self.cameras.clone();
        let mut copies = std::collections::HashMap::new();
        let History { undo, redo, .. } = &mut self.history;
        for step in undo.iter_mut().chain(redo.iter_mut()) {
            let state = &mut step.state;
            state
                .computed
                .drop_lookup_tables(&live, &mut state.dropped, &mut copies);
        }
    }

    /// Puts a snapshot back. The camera views' scale and bias stay as they are now.
    fn restore(&mut self, snapshot: Snapshot) {
        let Snapshot {
            mut project,
            scheduler,
            mut computed,
            extras,
            revision,
            dropped,
        } = snapshot;
        computed.restore_lookup_tables(&dropped, &self.cameras);
        // Review 12: a state whose scale and bias are not the snapshot's is not the state
        // the snapshot's revision names (which may be the one saved), so it gets a new one.
        let mut adjusted = false;
        for (restored, current) in project.cameras.iter_mut().zip(&self.project.cameras) {
            if restored.image_adjustments != current.image_adjustments {
                Arc::make_mut(restored).image_adjustments = current.image_adjustments;
                adjusted = true;
            }
        }
        self.project = project;
        self.set_scheduler_state(scheduler);
        self.set_computed_state(computed);
        if let Some(extras) = extras {
            self.extras = *extras;
        }
        if adjusted {
            self.history.new_revision();
        } else {
            self.history.revision = revision;
        }
        self.clear_stale();
        self.sync_camera_intrinsics();
        self.note_change();
        crate::calibration::sync_manual_lists(self);
    }

    /// Opens a step named `label` (shown as "Undo <label>"). Inside an open step this only
    /// deepens the nesting: the inner step becomes part of the outer one.
    pub fn begin_step(&mut self, label: &str) {
        if let Some(open) = &mut self.history.open {
            open.depth += 1;
            return;
        }
        self.history.open = Some(OpenStep {
            label: label.to_owned(),
            before: self.snapshot(false),
            depth: 1,
            nested: Vec::new(),
        });
    }

    /// Closes the innermost open step. Closing the outermost one brings derived data up to
    /// date ([`Document::settle`]) and, when the project, the cameras' derived state or the
    /// session values undo restores differ from what they were (or the step wrote the
    /// loader's extras), records the undo step, empties the redo stack and marks the
    /// document modified. Returns whether a step was recorded.
    ///
    /// # Panics
    /// If no step is open.
    pub fn commit_step(&mut self) -> bool {
        let open = self
            .history
            .open
            .as_mut()
            .expect("commit_step without begin_step");
        if open.depth > 1 {
            open.depth -= 1;
            return false;
        }
        self.settle();
        self.clear_non_finite();
        let open = self.history.open.take().expect("open step");
        crate::calibration::sync_manual_lists(self);
        let changed = open.before.extras.is_some()
            || !same_project(&open.before.project, &self.project)
            || !open.before.computed.same_as(&self.computed_state());
        if !changed {
            // Scheduler-only differences are not user-visible state.
            return false;
        }
        self.note_change();
        self.history.new_revision();
        self.history.redo.clear();
        self.history.undo.push(Step {
            label: open.label,
            state: open.before,
        });
        let excess = self.history.undo.len().saturating_sub(self.history.limit);
        if excess > 0 {
            self.history.undo.drain(..excess);
        }
        self.drop_step_lookup_tables();
        true
    }

    /// Abandons every open step, however deeply nested, and puts the document back as it was
    /// when the outermost one was opened. Does nothing when no step is open.
    pub fn abort_step(&mut self) {
        if let Some(open) = self.history.open.take() {
            self.restore(open.before);
        }
    }

    /// True while a step is open.
    #[must_use]
    pub fn step_open(&self) -> bool {
        self.history.open.is_some()
    }

    /// Runs `f` as one step: committed when it returns `Ok`, taken back when it returns
    /// `Err`. Inside an open step `f` runs as part of it; when it fails, the document is put
    /// back as it was when this call began (the outer step's earlier work is kept) and the
    /// error is passed up, so a failed edit changes nothing in an open step either.
    ///
    /// # Errors
    /// Whatever `f` returns.
    pub fn transact<T>(
        &mut self,
        label: &str,
        f: impl FnOnce(&mut Document) -> Result<T, EditError>,
    ) -> Result<T, EditError> {
        // Nothing reads the change count while `f` runs, so it is put back when the
        // document ends as it began (a failed edit, or one that changed nothing): it moves
        // exactly when the content changed (ledger A97).
        let changes_before = self.history.changes;
        let nested = self.step_open();
        if nested {
            let start = NestedStart {
                project: self.project.clone(),
                scheduler: self.scheduler_state().clone(),
                computed: self.computed_state(),
                stale: self.stale_state().clone(),
                extras: None,
            };
            if let Some(open) = &mut self.history.open {
                open.nested.push(start);
            }
        }
        let stale_before = (!nested).then(|| self.stale_state().clone());
        self.begin_step(label);
        let result = f(self);
        let start = if nested {
            self.history
                .open
                .as_mut()
                .and_then(|open| open.nested.pop())
        } else {
            None
        };
        match result {
            Ok(value) => {
                let changed = if let Some(start) = &start {
                    start.extras.is_some()
                        || !same_project(&start.project, &self.project)
                        || !start.computed.same_as(&self.computed_state())
                } else {
                    true
                };
                let recorded = self.commit_step();
                if nested {
                    if changed {
                        if self.history.changes == changes_before {
                            self.note_change();
                        }
                    } else {
                        self.history.changes = changes_before;
                    }
                } else if !recorded {
                    // `commit_step` found the project as it was: nothing changed.
                    self.history.changes = changes_before;
                }
                Ok(value)
            }
            Err(error) => {
                if nested {
                    if let Some(start) = start {
                        self.restore_nested(start);
                    }
                    // Leave the nesting level; the outer step stays open.
                    if let Some(open) = &mut self.history.open {
                        open.depth -= 1;
                    }
                } else {
                    self.abort_step();
                    if let Some(stale) = stale_before {
                        self.set_stale_state(stale);
                    }
                }
                self.history.changes = changes_before;
                Err(error)
            }
        }
    }

    /// Puts back the state a nested step began with. The camera views' scale and bias stay
    /// as they are now, as in [`Document::undo`].
    fn restore_nested(&mut self, start: NestedStart) {
        let NestedStart {
            mut project,
            scheduler,
            computed,
            stale,
            extras,
        } = start;
        for (restored, current) in project.cameras.iter_mut().zip(&self.project.cameras) {
            if restored.image_adjustments != current.image_adjustments {
                Arc::make_mut(restored).image_adjustments = current.image_adjustments;
            }
        }
        self.project = project;
        self.set_scheduler_state(scheduler);
        self.set_computed_state(computed);
        self.set_stale_state(stale);
        if let Some(extras) = extras {
            self.extras = *extras;
        }
        self.sync_camera_intrinsics();
        self.note_change();
    }

    /// The loader's extras for writing. Inside a step, the first call keeps a copy so that
    /// undo can restore them (and so that a failed nested step can put them back).
    pub fn extras_mut(&mut self) -> &mut LoadedExtras {
        self.note_change();
        if let Some(open) = &mut self.history.open {
            let needs =
                open.before.extras.is_none() || open.nested.iter().any(|n| n.extras.is_none());
            if needs {
                let copy = Box::new(self.extras.clone());
                if open.before.extras.is_none() {
                    open.before.extras = Some(copy.clone());
                }
                for n in &mut open.nested {
                    if n.extras.is_none() {
                        n.extras = Some(copy.clone());
                    }
                }
            }
        }
        &mut self.extras
    }

    /// Undoes the last step and returns its label. `None` when there is nothing to undo or
    /// a step is open.
    pub fn undo(&mut self) -> Option<String> {
        if self.step_open() {
            return None;
        }
        let step = self.history.undo.pop()?;
        let current = self.snapshot(step.state.extras.is_some());
        self.restore(step.state);
        self.history.redo.push(Step {
            label: step.label.clone(),
            state: current,
        });
        self.drop_step_lookup_tables();
        Some(step.label)
    }

    /// Redoes the last undone step and returns its label.
    pub fn redo(&mut self) -> Option<String> {
        if self.step_open() {
            return None;
        }
        let step = self.history.redo.pop()?;
        let current = self.snapshot(step.state.extras.is_some());
        self.restore(step.state);
        self.history.undo.push(Step {
            label: step.label.clone(),
            state: current,
        });
        let excess = self.history.undo.len().saturating_sub(self.history.limit);
        if excess > 0 {
            self.history.undo.drain(..excess);
        }
        self.drop_step_lookup_tables();
        Some(step.label)
    }

    /// Label of the step [`Document::undo`] would undo.
    #[must_use]
    pub fn undo_label(&self) -> Option<&str> {
        self.history.undo.last().map(|s| s.label.as_str())
    }

    /// Label of the step [`Document::redo`] would redo.
    #[must_use]
    pub fn redo_label(&self) -> Option<&str> {
        self.history.redo.last().map(|s| s.label.as_str())
    }

    /// Number of steps that can be undone.
    #[must_use]
    pub fn undo_depth(&self) -> usize {
        self.history.undo.len()
    }

    /// Sets how many undo steps are kept (at least 1), dropping the oldest beyond it.
    pub fn set_undo_limit(&mut self, limit: usize) {
        self.history.limit = limit.max(1);
        let excess = self.history.undo.len().saturating_sub(self.history.limit);
        if excess > 0 {
            self.history.undo.drain(..excess);
        }
    }

    /// Forgets every undo and redo step (the modified state is kept).
    pub fn clear_history(&mut self) {
        self.history.undo.clear();
        self.history.redo.clear();
    }

    /// True when the document differs from what was last saved or loaded.
    #[must_use]
    pub fn is_modified(&self) -> bool {
        self.history.saved_revision != Some(self.history.revision)
    }

    /// Records that the present state is on disk. Called by a successful save.
    pub fn mark_saved(&mut self) {
        self.history.saved_revision = Some(self.history.revision);
    }

    /// Records that the state of revision `revision` is on disk: for a save written from a
    /// copy of the document (on a worker thread) while this document moved on. The document
    /// is unmodified again only if it returns to that revision (by undo or redo).
    pub fn mark_saved_at(&mut self, revision: u64) {
        self.history.saved_revision = Some(revision);
    }

    /// Records that no state of the document is on disk: for a document recovered from an
    /// autosave copy, which differs from its project file. Unlike
    /// [`Document::mark_modified`] it leaves the revision as it is.
    pub fn mark_unsaved(&mut self) {
        self.history.saved_revision = None;
    }

    /// Marks the document modified without an undo step: for a change made outside a step
    /// (the camera views' scale and bias, a recomputation that changed stored values).
    pub fn mark_modified(&mut self) {
        self.history.new_revision();
        self.note_change();
    }

    /// Marks the document modified without moving [`Document::changes`]: for view settings
    /// saved with the project that the views apply themselves (scale and bias).
    pub(crate) fn mark_modified_view_only(&mut self) {
        self.history.new_revision();
    }

    /// A counter that moves when the document's content changes: an edit (also inside an
    /// open step), undo, redo, a step taken back, a trial update or trial selection that
    /// changed stored values, a recomputation, a write of a tracking run, a change of
    /// settings, of the extras, of the compatibility mode, or a save (which sets the creation
    /// date). Those are the only ways the project changes: nothing writes it outside them
    /// (manual mode's point lists wait for the next calibration edit, ledger A164). The
    /// session's wizard input (reference clicks, manual mode and its point, the workspace)
    /// is not content and does not move it. It does not move for an edit that fails or changes nothing
    /// ([`Document::transact`] puts it back), for the commit of a step that changed nothing,
    /// or for the camera views' scale and bias ([`Document::set_image_adjustment`]), which
    /// the views apply themselves. Unlike [`Document::revision`], which names the state for
    /// the modified flag and moves only with undo steps, it also moves for changes outside
    /// a step. An interface rebuilds what it shows when it moves. It may move for a write
    /// that stores the value already there (a tracker writing the same point).
    #[must_use]
    pub fn changes(&self) -> u64 {
        self.history.changes
    }

    /// Moves [`Document::changes`].
    pub(crate) fn note_change(&mut self) {
        self.history.changes = self.history.changes.wrapping_add(1);
    }

    /// Moves [`Document::changes`] when the project differs from `before` (a copy taken
    /// before a recomputation that may leave it as it was).
    pub(crate) fn note_if_changed(&mut self, before: &Project) {
        if !same_project(before, &self.project) {
            self.note_change();
        }
    }

    /// The revision number of the present state. It changes with every committed change,
    /// undo and redo, so an interface can use it to know when to rebuild what it shows.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.history.revision
    }
}
