//! The frame cache shared by all sources of a project.
//!
//! The C++ program keeps only the frame on screen. This cache keeps decoded
//! frames up to a memory budget, reads each source on its own worker thread,
//! and reads ahead in the direction the user is moving (PLAN.md section 4,
//! "Video").
//!
//! # Use
//!
//! ```no_run
//! use std::sync::Arc;
//! use xma_video::cache::{CacheOptions, FrameCache};
//! # fn source() -> Box<dyn xma_video::FrameSource> { unimplemented!() }
//!
//! let cache = Arc::new(FrameCache::new(CacheOptions::with_budget(2 << 30)));
//! // Once, before requests: called on a worker thread when a frame asked for
//! // with `request` arrives.
//! cache.set_on_ready(Box::new(|source, frame| { let _ = (source, frame); /* repaint */ })).ok();
//! let camera = cache.add_source(source());
//!
//! // GUI: never blocks. `None` means "queued"; the callback follows.
//! match cache.request(camera, 10) {
//!     Some(Ok(frame)) => { let _ = frame; /* draw */ }
//!     Some(Err(e)) => eprintln!("{e}"),
//!     None => { /* keep showing the previous frame */ }
//! }
//!
//! // Tracking and export: blocks until the frame is there.
//! let frame = cache.get(camera, 11).expect("frame 11");
//! # let _ = frame;
//! ```
//!
//! # API
//!
//! - [`FrameCache::new`]`(`[`CacheOptions`]`)`: an empty cache with a budget in
//!   bytes and a read-ahead length. Share it as `Arc<FrameCache>`; dropping it
//!   stops every worker, drops every source (which ends any ffmpeg process),
//!   and waits for reads in progress to finish.
//! - [`FrameCache::add_source`]`(Box<dyn FrameSource>) -> `[`SourceId`]:
//!   registers a source and starts its worker thread. [`FrameCache::info`]
//!   returns its [`SourceInfo`]. [`FrameCache::remove_source`] removes it
//!   without blocking: queued reads are dropped, blocked [`FrameCache::get`]
//!   calls return [`CacheError::NoSuchSource`], its frames leave the cache,
//!   and the source is dropped once a read in progress ends.
//! - [`FrameCache::request`]`(id, frame) -> Option<`[`FrameResult`]`>`: never
//!   blocks. Returns the frame (or its cached read error) when cached;
//!   otherwise queues it ahead of everything already queued for that source,
//!   and returns `None`; the callback set with [`FrameCache::set_on_ready`]
//!   is called when it arrives.
//! - [`FrameCache::get`]`(id, frame) -> `[`FrameResult`]: blocks until the
//!   frame is read (queued like a request).
//! - [`FrameCache::peek`]: the cached frame, without queuing anything or
//!   counting a hit or a miss.
//! - [`FrameCache::set_on_ready`]: the callback, which can be set once. It runs
//!   on a worker thread with no lock held, so it may call `request` or `peek`,
//!   but should not block for long, and must not call `get` on the same source
//!   (that worker is the one that would serve it).
//! - [`FrameCache::set_budget`], [`FrameCache::stats`] ([`CacheStats`]:
//!   hits, misses, decodes, read-ahead decodes, evictions, read errors, bytes
//!   and frames held).
//!
//! # Behaviour
//!
//! - **Budget.** Frames are kept until their bytes exceed the budget, then the
//!   least recently used go. A single frame larger than the budget is still
//!   kept until the next one arrives. Frames are handed out as `Arc<Frame>`, so
//!   eviction never invalidates a frame in use; its memory is freed when the
//!   last user drops it.
//! - **Order.** Each source has one worker and reads one frame at a time. A
//!   request or get goes to the front of that source's queue, so the newest
//!   request is read next and overtakes everything queued earlier, read-ahead
//!   included. A read already in progress is not interrupted.
//! - **Read-ahead.** Each request and get (hit or miss) is remembered. When
//!   the last three requests of a source move the same way by the same step of
//!   at most 8 frames, the next [`CacheOptions::read_ahead`] frames in that
//!   direction are queued behind any requests, limited to half the budget. Any
//!   other pattern (a jump) cancels queued read-ahead.
//! - **Errors.** A failed read is cached for that frame and returned again
//!   without retrying, until the source is removed. Frames at or beyond the
//!   source's `readable_frames` fail at once without reaching the worker.
//! - **A reader that panics.** The panic is caught and the frame gets a read
//!   error saying so. The reader may be left half-way through a read, so the
//!   source is not read again: its worker ends and drops it, and every frame
//!   of it queued then or asked for later gets the same kind of error (cached
//!   like any read error). Its frames already cached are still served.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;

use crate::{Frame, FrameSource, Pixels, SourceInfo, VideoError};

/// Identifies a source registered with a [`FrameCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(pub u64);

/// Why the cache could not hand out a frame.
#[derive(Debug, Clone)]
pub enum CacheError {
    /// The source failed to read the frame. The error is cached and returned
    /// for later requests of the same frame.
    Read(Arc<VideoError>),
    /// No source has this id, or it was removed (also while a `get` waited).
    NoSuchSource(SourceId),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheError::Read(e) => e.fmt(f),
            CacheError::NoSuchSource(id) => write!(f, "frame source {} is not open", id.0),
        }
    }
}

impl std::error::Error for CacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CacheError::Read(e) => Some(e.as_ref()),
            CacheError::NoSuchSource(_) => None,
        }
    }
}

/// A frame from the cache, or why there is none.
pub type FrameResult = Result<Arc<Frame>, CacheError>;

/// Called with the source and frame index when a requested frame arrives.
pub type OnReady = Box<dyn Fn(SourceId, usize) + Send + Sync>;

/// Settings of a [`FrameCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheOptions {
    /// Memory for decoded frames, in bytes.
    pub budget_bytes: usize,
    /// Frames read ahead in the direction of travel.
    pub read_ahead: usize,
}

impl CacheOptions {
    /// A budget with the default read-ahead of 8 frames.
    pub fn with_budget(budget_bytes: usize) -> CacheOptions {
        CacheOptions {
            budget_bytes,
            read_ahead: 8,
        }
    }
}

impl Default for CacheOptions {
    /// 1 GiB, read-ahead 8.
    fn default() -> Self {
        CacheOptions::with_budget(1 << 30)
    }
}

/// Counters since the cache was made.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Requests and gets answered from the cache (frames or cached errors).
    pub hits: u64,
    /// Requests and gets that had to queue a read.
    pub misses: u64,
    /// Reads performed by the workers, read-ahead included.
    pub decodes: u64,
    /// Of `decodes`, the reads made for read-ahead.
    pub read_ahead_decodes: u64,
    /// Frames dropped to stay within the budget.
    pub evictions: u64,
    /// Reads that failed.
    pub read_errors: u64,
    /// Bytes of frame data held now.
    pub bytes: usize,
    /// Frames held now.
    pub frames: usize,
}

/// Most requests kept queued per source; older ones are dropped (the newest
/// matter, as when a slider is dragged).
const MAX_DEMAND: usize = 16;
/// Largest step between requests still taken as a direction of travel.
const MAX_STEP: i64 = 8;
/// Requests remembered per source for the direction of travel.
const HISTORY: usize = 3;

type Key = (SourceId, usize);

struct Cached {
    frame: Arc<Frame>,
    bytes: usize,
    tick: u64,
}

struct Slot {
    info: SourceInfo,
    frame_bytes: usize,
    /// Requested frames, newest first.
    demand: VecDeque<usize>,
    /// Read-ahead frames, in reading order.
    ahead: VecDeque<usize>,
    /// Frames asked for with `request` whose arrival calls the callback.
    notify: HashSet<usize>,
    in_flight: Option<usize>,
    history: VecDeque<usize>,
    /// Set when the source's reader panicked: why. The source is not read again; every
    /// frame of it not yet cached gets a read error saying so.
    panicked: Option<String>,
}

/// The read error of a frame of a source whose reader panicked.
fn panic_error(id: SourceId, n: usize, reason: &str) -> Arc<VideoError> {
    Arc::new(VideoError::Decode {
        path: format!("frame source {}", id.0).into(),
        frame: Some(n),
        reason: reason.to_string(),
    })
}

struct State {
    budget: usize,
    read_ahead: usize,
    next_id: u64,
    next_tick: u64,
    next_ticket: u64,
    frames: HashMap<Key, Cached>,
    lru: BTreeMap<u64, Key>,
    errors: HashMap<Key, Arc<VideoError>>,
    sources: HashMap<SourceId, Slot>,
    waiters: HashMap<Key, Vec<u64>>,
    delivered: HashMap<u64, FrameResult>,
    stats: CacheStats,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    /// Signalled when a frame arrives or a source goes (for `get`).
    arrived: Condvar,
    /// Signalled when work is queued or a source goes (for the workers).
    work: Condvar,
    on_ready: OnceLock<OnReady>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Decoded frames of several sources within one memory budget. See the module
/// documentation.
pub struct FrameCache {
    shared: Arc<Shared>,
    workers: Mutex<Vec<(SourceId, JoinHandle<()>)>>,
}

impl fmt::Debug for FrameCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameCache")
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

fn frame_bytes(frame: &Frame) -> usize {
    match &frame.pixels {
        Pixels::Gray8(d) => d.len(),
        Pixels::Gray16(d) => d.len() * 2,
    }
}

impl State {
    fn touch(&mut self, key: Key) {
        let tick = self.next_tick;
        if let Some(c) = self.frames.get_mut(&key) {
            self.next_tick += 1;
            self.lru.remove(&c.tick);
            c.tick = tick;
            self.lru.insert(tick, key);
        }
    }

    fn cached(&self, key: Key) -> Option<FrameResult> {
        if let Some(c) = self.frames.get(&key) {
            return Some(Ok(Arc::clone(&c.frame)));
        }
        self.errors
            .get(&key)
            .map(|e| Err(CacheError::Read(Arc::clone(e))))
    }

    fn is_done(&self, key: Key) -> bool {
        self.frames.contains_key(&key) || self.errors.contains_key(&key)
    }

    fn evict_to(&mut self, limit: usize) {
        while self.stats.bytes > limit {
            let Some((_, key)) = self.lru.pop_first() else {
                break;
            };
            if let Some(c) = self.frames.remove(&key) {
                self.stats.bytes -= c.bytes;
                self.stats.evictions += 1;
            }
        }
        self.stats.frames = self.frames.len();
    }

    fn insert(&mut self, key: Key, frame: Arc<Frame>) {
        let bytes = frame_bytes(&frame);
        if let Some(old) = self.frames.remove(&key) {
            self.lru.remove(&old.tick);
            self.stats.bytes -= old.bytes;
        }
        self.evict_to(self.budget.saturating_sub(bytes));
        let tick = self.next_tick;
        self.next_tick += 1;
        self.lru.insert(tick, key);
        self.frames.insert(key, Cached { frame, bytes, tick });
        self.stats.bytes += bytes;
        self.stats.frames = self.frames.len();
    }

    /// Remembers a request and replans the read-ahead of the source.
    fn note_request(&mut self, id: SourceId, n: usize) {
        let budget = self.budget;
        let read_ahead = self.read_ahead;
        let Some(slot) = self.sources.get_mut(&id) else {
            return;
        };
        if slot.history.back() != Some(&n) {
            slot.history.push_back(n);
            while slot.history.len() > HISTORY {
                slot.history.pop_front();
            }
        }
        slot.ahead.clear();
        let h: Vec<i64> = slot.history.iter().map(|&v| v as i64).collect();
        if h.len() < HISTORY {
            return;
        }
        let steps: Vec<i64> = h.windows(2).map(|w| w[1] - w[0]).collect();
        let step = steps[0];
        if step == 0 || step.abs() > MAX_STEP || steps.iter().any(|&s| s != step) {
            return;
        }
        let limit = (budget / 2) / slot.frame_bytes.max(1);
        let count = read_ahead.min(limit);
        let readable = slot.info.readable_frames as i64;
        let mut planned = Vec::new();
        for k in 1..=count as i64 {
            let f = n as i64 + step * k;
            if f < 0 || f >= readable {
                break;
            }
            planned.push(f as usize);
        }
        let planned: Vec<usize> = planned
            .into_iter()
            .filter(|&f| !self.frames.contains_key(&(id, f)) && !self.errors.contains_key(&(id, f)))
            .collect();
        if let Some(slot) = self.sources.get_mut(&id) {
            slot.ahead.extend(planned);
        }
    }

    /// Puts `n` at the front of the source's queue.
    fn enqueue(&mut self, id: SourceId, n: usize) {
        if let Some(slot) = self.sources.get_mut(&id) {
            slot.demand.retain(|&f| f != n);
            slot.ahead.retain(|&f| f != n);
            slot.demand.push_front(n);
            while slot.demand.len() > MAX_DEMAND {
                if let Some(old) = slot.demand.pop_back() {
                    slot.notify.remove(&old);
                }
            }
        }
    }

    /// The next frame the worker of `id` should read, and whether it is
    /// read-ahead.
    fn next_job(&mut self, id: SourceId) -> Option<(usize, bool)> {
        loop {
            let slot = self.sources.get_mut(&id)?;
            let (n, ahead) = match slot.demand.pop_front() {
                Some(n) => (n, false),
                None => (slot.ahead.pop_front()?, true),
            };
            if !self.is_done((id, n)) {
                return Some((n, ahead));
            }
        }
    }

    /// Hands `result` to every `get` waiting for `key`.
    fn deliver(&mut self, key: Key, result: &FrameResult) {
        if let Some(tickets) = self.waiters.remove(&key) {
            for t in tickets {
                self.delivered.insert(t, result.clone());
            }
        }
    }

    /// Checks the source and answers at once if it can: a cached frame or
    /// error (a hit), or a frame beyond the readable ones.
    fn immediate(&mut self, id: SourceId, n: usize) -> Option<FrameResult> {
        let Some(slot) = self.sources.get(&id) else {
            return Some(Err(CacheError::NoSuchSource(id)));
        };
        let readable = slot.info.readable_frames;
        if n >= readable {
            return Some(Err(CacheError::Read(Arc::new(
                VideoError::FrameOutOfRange { index: n, readable },
            ))));
        }
        self.note_request(id, n);
        let hit = self.cached((id, n));
        if hit.is_some() {
            self.stats.hits += 1;
            self.touch((id, n));
        } else {
            self.stats.misses += 1;
            if let Some(reason) = self.sources.get(&id).and_then(|s| s.panicked.clone()) {
                // The reader panicked earlier: answer without reading.
                let e = panic_error(id, n, &reason);
                self.errors.insert((id, n), Arc::clone(&e));
                return Some(Err(CacheError::Read(e)));
            }
        }
        hit
    }
}

impl FrameCache {
    /// An empty cache.
    pub fn new(options: CacheOptions) -> FrameCache {
        FrameCache {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    budget: options.budget_bytes,
                    read_ahead: options.read_ahead,
                    next_id: 1,
                    next_tick: 0,
                    next_ticket: 0,
                    frames: HashMap::new(),
                    lru: BTreeMap::new(),
                    errors: HashMap::new(),
                    sources: HashMap::new(),
                    waiters: HashMap::new(),
                    delivered: HashMap::new(),
                    stats: CacheStats::default(),
                    shutdown: false,
                }),
                arrived: Condvar::new(),
                work: Condvar::new(),
                on_ready: OnceLock::new(),
            }),
            workers: Mutex::new(Vec::new()),
        }
    }

    /// Sets the callback for frames that arrive after a [`FrameCache::request`]
    /// returned `None`. It can be set once; a second call returns the callback
    /// back as the error.
    pub fn set_on_ready(&self, callback: OnReady) -> Result<(), OnReady> {
        self.shared.on_ready.set(callback)
    }

    /// Registers a source and starts its worker thread.
    pub fn add_source(&self, source: Box<dyn FrameSource>) -> SourceId {
        let info = source.info().clone();
        let id = {
            let mut st = self.shared.lock();
            let id = SourceId(st.next_id);
            st.next_id += 1;
            let frame_bytes = (info.width as usize * info.height as usize)
                * if info.bit_depth > 8 { 2 } else { 1 };
            st.sources.insert(
                id,
                Slot {
                    info,
                    frame_bytes,
                    demand: VecDeque::new(),
                    ahead: VecDeque::new(),
                    notify: HashSet::new(),
                    in_flight: None,
                    history: VecDeque::new(),
                    panicked: None,
                },
            );
            id
        };
        let shared = Arc::clone(&self.shared);
        let handle = std::thread::Builder::new()
            .name(format!("xma-frames-{}", id.0))
            .spawn(move || worker(shared, id, source))
            .expect("spawning a thread");
        let mut workers = self.workers.lock().unwrap_or_else(|e| e.into_inner());
        // Workers of removed sources end on their own; forget those that have.
        workers.retain(|(_, h)| !h.is_finished());
        workers.push((id, handle));
        id
    }

    /// The properties of a registered source.
    pub fn info(&self, id: SourceId) -> Option<SourceInfo> {
        self.shared.lock().sources.get(&id).map(|s| s.info.clone())
    }

    /// Removes a source without waiting for a read in progress; see the module
    /// documentation. Returns false if there was no such source.
    pub fn remove_source(&self, id: SourceId) -> bool {
        let mut st = self.shared.lock();
        if st.sources.remove(&id).is_none() {
            return false;
        }
        let keys: Vec<Key> = st.frames.keys().filter(|k| k.0 == id).copied().collect();
        for k in keys {
            if let Some(c) = st.frames.remove(&k) {
                st.lru.remove(&c.tick);
                st.stats.bytes -= c.bytes;
            }
        }
        st.stats.frames = st.frames.len();
        st.errors.retain(|k, _| k.0 != id);
        let waiting: Vec<Key> = st.waiters.keys().filter(|k| k.0 == id).copied().collect();
        for k in waiting {
            st.deliver(k, &Err(CacheError::NoSuchSource(id)));
        }
        drop(st);
        self.shared.work.notify_all();
        self.shared.arrived.notify_all();
        true
    }

    /// The frame if it is cached, without queuing it or counting a hit.
    pub fn peek(&self, id: SourceId, frame: usize) -> Option<FrameResult> {
        self.shared.lock().cached((id, frame))
    }

    /// The frame if it is cached; otherwise queues it at the front of its
    /// source's queue and returns `None` (the callback follows). Never blocks.
    pub fn request(&self, id: SourceId, frame: usize) -> Option<FrameResult> {
        let mut st = self.shared.lock();
        if let Some(r) = st.immediate(id, frame) {
            drop(st);
            self.shared.work.notify_all();
            return Some(r);
        }
        st.enqueue(id, frame);
        if let Some(slot) = st.sources.get_mut(&id) {
            slot.notify.insert(frame);
        }
        drop(st);
        self.shared.work.notify_all();
        None
    }

    /// The frame, waiting for it to be read if it is not cached.
    pub fn get(&self, id: SourceId, frame: usize) -> FrameResult {
        let key = (id, frame);
        let mut st = self.shared.lock();
        if let Some(r) = st.immediate(id, frame) {
            drop(st);
            self.shared.work.notify_all();
            return r;
        }
        let ticket = st.next_ticket;
        st.next_ticket += 1;
        st.waiters.entry(key).or_default().push(ticket);
        st.enqueue(id, frame);
        self.shared.work.notify_all();
        loop {
            st = self
                .shared
                .arrived
                .wait(st)
                .unwrap_or_else(|e| e.into_inner());
            if let Some(r) = st.delivered.remove(&ticket) {
                return r;
            }
            let Some(slot) = st.sources.get(&id) else {
                return Err(CacheError::NoSuchSource(id));
            };
            // A queue trimmed by newer requests may have dropped this frame.
            let queued = slot.in_flight == Some(frame)
                || slot.demand.contains(&frame)
                || slot.ahead.contains(&frame);
            if !queued {
                if let Some(r) = st.cached(key) {
                    if let Some(ts) = st.waiters.get_mut(&key) {
                        ts.retain(|&t| t != ticket);
                    }
                    return r;
                }
                st.enqueue(id, frame);
                self.shared.work.notify_all();
            }
        }
    }

    /// Changes the budget, evicting at once if it shrank.
    pub fn set_budget(&self, budget_bytes: usize) {
        let mut st = self.shared.lock();
        st.budget = budget_bytes;
        st.evict_to(budget_bytes);
    }

    /// The counters.
    pub fn stats(&self) -> CacheStats {
        self.shared.lock().stats
    }
}

impl Drop for FrameCache {
    fn drop(&mut self) {
        {
            let mut st = self.shared.lock();
            st.shutdown = true;
            st.sources.clear();
        }
        self.shared.work.notify_all();
        self.shared.arrived.notify_all();
        let workers = std::mem::take(&mut *self.workers.lock().unwrap_or_else(|e| e.into_inner()));
        for (_, h) in workers {
            let _ = h.join();
        }
    }
}

/// The worker of one source: reads queued frames until the source is removed
/// or the cache is dropped. The source is dropped when it returns.
fn worker(shared: Arc<Shared>, id: SourceId, mut source: Box<dyn FrameSource>) {
    loop {
        let (n, ahead) = {
            let mut st = shared.lock();
            loop {
                if st.shutdown || !st.sources.contains_key(&id) {
                    return;
                }
                if let Some(job) = st.next_job(id) {
                    if let Some(slot) = st.sources.get_mut(&id) {
                        slot.in_flight = Some(job.0);
                    }
                    break job;
                }
                st = shared.work.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        };
        // A reader that panics is caught here, or `in_flight` would stay set and every
        // `get` of this frame wait for ever (review 11).
        let result =
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| source.read_frame(n))) {
                Ok(result) => result,
                Err(payload) => {
                    reader_panicked(&shared, id, n, payload.as_ref());
                    // The reader may be left half-way through a read: it is not used again,
                    // and is dropped here.
                    return;
                }
            };
        let key = (id, n);
        let notify = {
            let mut st = shared.lock();
            if st.shutdown || !st.sources.contains_key(&id) {
                return;
            }
            st.stats.decodes += 1;
            if ahead {
                st.stats.read_ahead_decodes += 1;
            }
            let out: FrameResult = match result {
                Ok(frame) => {
                    let frame = Arc::new(frame);
                    st.insert(key, Arc::clone(&frame));
                    Ok(frame)
                }
                Err(e) => {
                    let e = Arc::new(e);
                    st.stats.read_errors += 1;
                    st.errors.insert(key, Arc::clone(&e));
                    Err(CacheError::Read(e))
                }
            };
            st.deliver(key, &out);
            let slot = st.sources.get_mut(&id).expect("checked above");
            slot.in_flight = None;
            slot.notify.remove(&n)
        };
        shared.arrived.notify_all();
        if notify && let Some(cb) = shared.on_ready.get() {
            cb(id, n);
        }
    }
}

/// After the reader of `id` panicked while reading frame `n`: caches a read error for `n`
/// and for every frame queued or waited for, delivers them, marks the source so that no
/// later frame is read ([`State::immediate`] answers them), and calls the callback for the
/// frames asked for with `request`.
fn reader_panicked(shared: &Shared, id: SourceId, n: usize, payload: &(dyn std::any::Any + Send)) {
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned());
    let reason = match message {
        Some(m) => format!("the frame reader panicked ({m}); the source is not read again"),
        None => "the frame reader panicked; the source is not read again".to_string(),
    };
    let notify: Vec<usize> = {
        let mut st = shared.lock();
        if st.shutdown || !st.sources.contains_key(&id) {
            return;
        }
        st.stats.decodes += 1;
        st.stats.read_errors += 1;
        let slot = st.sources.get_mut(&id).expect("checked above");
        slot.in_flight = None;
        slot.panicked = Some(reason.clone());
        let mut frames: Vec<usize> = std::iter::once(n).chain(slot.demand.drain(..)).collect();
        slot.ahead.clear();
        let notify: Vec<usize> = slot.notify.drain().collect();
        frames.extend(notify.iter().copied());
        frames.extend(st.waiters.keys().filter(|k| k.0 == id).map(|k| k.1));
        frames.sort_unstable();
        frames.dedup();
        for f in frames {
            let key = (id, f);
            if st.is_done(key) {
                if let Some(r) = st.cached(key) {
                    st.deliver(key, &r);
                }
                continue;
            }
            let e = panic_error(id, f, &reason);
            st.errors.insert(key, Arc::clone(&e));
            st.deliver(key, &Err(CacheError::Read(e)));
        }
        notify
    };
    shared.arrived.notify_all();
    if let Some(cb) = shared.on_ready.get() {
        for f in notify {
            cb(id, f);
        }
    }
}
