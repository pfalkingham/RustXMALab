//! The texture store: owns the GPU textures of the images shown, keyed by an id
//! the caller chooses, and uploads new images on a worker thread.
//!
//! Upload policy (from spike 1, `spikes/camera-view/README.md`): the interface
//! thread only hands a frame over and never waits. Each key holds at most one
//! frame waiting for upload; a newer frame replaces it, and the replaced one is
//! dropped and counted. So when the uploader falls behind, frames are skipped,
//! never queued, and the newest frame is always the one that ends up shown.
//!
//! An image wider or taller than the device's largest texture is refused, as an image whose
//! data does not match its size is (wgpu would treat it as a fatal error). If the upload
//! thread panics all the same, the key it was writing is released and a new upload thread
//! takes over the waiting images (review 12c).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

use eframe::egui_wgpu::{self, wgpu};
use egui::Id;
use xma_video::image_file::ColourImage;
use xma_video::{Frame, Pixels};

use super::render::{Pipelines, TextureMeta};
use crate::frames::mirror_rows;

/// An image to display.
#[derive(Clone, Debug)]
pub enum DisplayImage {
    /// A grey frame. `bit_depth` is the number of bits that carry information
    /// (8 for `Gray8`; 8 to 16 for `Gray16`, from `SourceInfo::bit_depth`); it
    /// sets which raw value is shown as white.
    Gray { frame: Frame, bit_depth: u8 },
    /// A colour image, 4 bytes per pixel (R, G, B, unused), rows from the top.
    /// XMALab shows colour images for colour stills and for the undistortion
    /// workspace's false-colour difference images.
    Rgba8 {
        width: u32,
        height: u32,
        data: Vec<u8>,
    },
    /// A grey frame shared with its owner (the frame service's caches), as
    /// [`DisplayImage::Gray`], written mirrored left to right when `mirror` is set (a
    /// flipped camera): the copy that mirroring needs is made by the thread that writes
    /// the texture, not by the one that hands the image over.
    GrayShared {
        frame: Arc<Frame>,
        bit_depth: u8,
        mirror: bool,
    },
    /// A colour image shared with its owner, as [`DisplayImage::Rgba8`], written mirrored
    /// left to right when `mirror` is set.
    RgbaShared {
        image: Arc<ColourImage>,
        mirror: bool,
    },
}

impl DisplayImage {
    pub fn gray(frame: Frame, bit_depth: u8) -> Self {
        Self::Gray { frame, bit_depth }
    }

    /// A shared grey frame at the depth it carries, as [`From<Frame>`] gives it, mirrored
    /// when `mirror` is set ([`DisplayImage::GrayShared`]).
    #[must_use]
    pub fn gray_shared(frame: Arc<Frame>, mirror: bool) -> Self {
        let bit_depth = gray_bit_depth(&frame);
        Self::GrayShared {
            frame,
            bit_depth,
            mirror,
        }
    }

    /// The grey frame of a grey image.
    fn grey(&self) -> Option<&Frame> {
        match self {
            Self::Gray { frame, .. } => Some(frame),
            Self::GrayShared { frame, .. } => Some(frame),
            Self::Rgba8 { .. } | Self::RgbaShared { .. } => None,
        }
    }

    pub fn width(&self) -> u32 {
        match self {
            Self::Rgba8 { width, .. } => *width,
            Self::RgbaShared { image, .. } => image.width,
            Self::Gray { frame, .. } => frame.width,
            Self::GrayShared { frame, .. } => frame.width,
        }
    }

    pub fn height(&self) -> u32 {
        match self {
            Self::Rgba8 { height, .. } => *height,
            Self::RgbaShared { image, .. } => image.height,
            Self::Gray { frame, .. } => frame.height,
            Self::GrayShared { frame, .. } => frame.height,
        }
    }

    fn format(&self) -> wgpu::TextureFormat {
        match self.grey() {
            Some(frame) => match frame.pixels {
                Pixels::Gray8(_) => wgpu::TextureFormat::R8Uint,
                Pixels::Gray16(_) => wgpu::TextureFormat::R16Uint,
            },
            None => wgpu::TextureFormat::Rgba8Uint,
        }
    }

    /// The pixels as held, not mirrored (as many bytes as the mirrored ones).
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Rgba8 { data, .. } => data,
            Self::RgbaShared { image, .. } => &image.rgba,
            Self::Gray { .. } | Self::GrayShared { .. } => match self.grey().map(|f| &f.pixels) {
                Some(Pixels::Gray8(d)) => d,
                Some(Pixels::Gray16(d)) => bytemuck::cast_slice(d),
                None => &[],
            },
        }
    }

    /// The mirrored copy of a shared image whose `mirror` is set: what is written to the
    /// texture in its place. Mirrored exactly as the frame service mirrored such an image
    /// before handing it over (`Frame::flip_horizontal`, [`mirror_rows`]).
    fn mirrored(&self) -> Option<DisplayImage> {
        match self {
            Self::GrayShared {
                frame,
                bit_depth,
                mirror: true,
            } => {
                let mut owned = (**frame).clone();
                owned.flip_horizontal();
                Some(Self::Gray {
                    frame: owned,
                    bit_depth: *bit_depth,
                })
            }
            Self::RgbaShared {
                image,
                mirror: true,
            } => {
                let mut data = image.rgba.clone();
                mirror_rows(&mut data, image.width as usize, 4);
                Some(Self::Rgba8 {
                    width: image.width,
                    height: image.height,
                    data,
                })
            }
            _ => None,
        }
    }

    /// The bytes written to the texture: the pixels, mirrored for a shared image whose
    /// `mirror` is set (tests).
    #[cfg(test)]
    pub(crate) fn upload_bytes(&self) -> Vec<u8> {
        let mirrored = self.mirrored();
        mirrored.as_ref().unwrap_or(self).bytes().to_vec()
    }

    /// Format, width, height and texture metadata (tests).
    #[cfg(test)]
    pub(crate) fn upload_meta(&self) -> (wgpu::TextureFormat, u32, u32, bool, u8) {
        let m = self.meta();
        (
            self.format(),
            self.width(),
            self.height(),
            m.rgba,
            m.bit_depth,
        )
    }

    fn meta(&self) -> TextureMeta {
        match self {
            Self::Gray { frame, bit_depth } => grey_meta(frame, *bit_depth),
            Self::GrayShared {
                frame, bit_depth, ..
            } => grey_meta(frame, *bit_depth),
            Self::Rgba8 { .. } | Self::RgbaShared { .. } => TextureMeta {
                rgba: true,
                bit_depth: 8,
            },
        }
    }

    fn check(&self) -> Result<(), String> {
        let (w, h) = (self.width() as usize, self.height() as usize);
        let expected = w * h * self.format().block_copy_size(None).unwrap_or(1) as usize;
        if w == 0 || h == 0 {
            return Err("image has no pixels".into());
        }
        if self.bytes().len() != expected {
            return Err(format!(
                "image data has {} bytes, {w}x{h} needs {expected}",
                self.bytes().len()
            ));
        }
        Ok(())
    }
}

impl From<Frame> for DisplayImage {
    /// A grey frame at the depth it carries: `Frame::bit_depth` (8 for `Gray8`, the
    /// number of significant bits for `Gray16`, for example 12 for a 12-bit `.cine`).
    fn from(frame: Frame) -> Self {
        let bit_depth = gray_bit_depth(&frame);
        Self::Gray { frame, bit_depth }
    }
}

/// The texture metadata of a grey frame shown at `bit_depth`.
fn grey_meta(frame: &Frame, bit_depth: u8) -> TextureMeta {
    TextureMeta {
        rgba: false,
        bit_depth: match frame.pixels {
            Pixels::Gray8(_) => 8,
            Pixels::Gray16(_) => bit_depth.clamp(8, 16),
        },
    }
}

/// The depth a grey frame is shown at ([`From<Frame>`]).
fn gray_bit_depth(frame: &Frame) -> u8 {
    match frame.pixels {
        Pixels::Gray8(_) => 8,
        Pixels::Gray16(_) => frame.bit_depth.clamp(9, 16),
    }
}

/// What a stored texture holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureInfo {
    pub width: u32,
    pub height: u32,
    pub rgba: bool,
    pub bit_depth: u8,
    /// Number of uploads applied to this key so far.
    pub generation: u64,
    /// The caller's tag of the image the texture holds ([`ImageStore::upload_tagged`]; 0
    /// for [`ImageStore::upload`]): which frame the texture shows, for measurements and
    /// tests.
    pub tag: u64,
}

/// Upload counters for the whole store.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UploadStats {
    /// Images written to the GPU.
    pub uploaded: u64,
    /// Images replaced by a newer one before they were uploaded.
    pub dropped: u64,
    /// Bytes written to the GPU.
    pub bytes: u64,
    /// Time spent in uploads (texture creation and `write_texture`), all threads.
    pub upload_secs: f64,
    /// Duration of the most recent upload.
    pub last_upload_secs: f64,
}

#[derive(Clone)]
pub(crate) struct TextureGpu {
    texture: wgpu::Texture,
    pub bind_group: wgpu::BindGroup,
    pub info: TextureInfo,
    pub meta: TextureMeta,
    format: wgpu::TextureFormat,
}

struct Entry {
    gpu: Option<TextureGpu>,
    /// The image waiting for upload, with its tag and its hand-over number.
    pending: Option<(DisplayImage, u64, u64)>,
    /// The hand-over number ([`Inner::handed`]) of the image in `gpu`, or the store's count
    /// when the entry was made. A finished upload that is not newer is dropped: an upload
    /// the worker had already taken can end after a later `upload_now` (review 12).
    shown: u64,
}

impl Entry {
    fn new(handed: u64) -> Self {
        Self {
            gpu: None,
            pending: None,
            shown: handed,
        }
    }
}

#[derive(Default)]
struct Inner {
    entries: HashMap<Id, Entry>,
    order: VecDeque<Id>,
    busy: bool,
    /// The key whose upload the worker is writing, while `busy`.
    busy_key: Option<Id>,
    /// Images handed over so far, all keys: numbers the hand-overs in order.
    handed: u64,
    shutdown: bool,
    /// No upload thread runs (one panicked and no other could be started): images handed
    /// over are uploaded at once on the calling thread.
    worker_dead: bool,
    stats: UploadStats,
    last_error: Option<String>,
}

struct Shared {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Arc<Pipelines>,
    inner: Mutex<Inner>,
    wake: Condvar,
    idle: Condvar,
    repaint: Option<egui::Context>,
    id: usize,
    /// The device's largest texture width and height (`max_texture_dimension_2d`).
    max_texture: u32,
    /// Upload threads started after one panicked, joined when the store is dropped.
    restarted: Mutex<Vec<std::thread::JoinHandle<()>>>,
    /// Called at the start of every write (tests: the order of the writes, a panic).
    #[cfg(test)]
    write_hook: Mutex<Option<WriteHook>>,
}

#[cfg(test)]
type WriteHook = Box<dyn Fn(&DisplayImage) + Send + Sync>;

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Inner {
    /// The next hand-over number, and `key`'s entry (made if absent).
    fn hand_over(&mut self, key: Id) -> (u64, &mut Entry) {
        self.handed += 1;
        let seq = self.handed;
        (
            seq,
            self.entries
                .entry(key)
                .or_insert_with(|| Entry::new(seq - 1)),
        )
    }
}

/// Stops the upload thread and waits for it when the last [`ImageStore`] clone
/// is dropped, so that the thread's references to the wgpu device, queue and
/// textures are released before the store's owner goes on (and before the
/// process exits). A detached thread still releasing the device while the
/// process tears down the graphics driver crashed at exit.
struct WorkerHandle {
    shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl WorkerHandle {
    fn stop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.wake.notify_all();
        if let Some(t) = self.thread.take() {
            // The worker never holds a store handle, so this is never the worker.
            let _ = t.join();
        }
        // A thread that took over from a panicked one was registered by that one before it
        // ended (and none is started once `shutdown` is set).
        loop {
            let next = self
                .shared
                .restarted
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .pop();
            match next {
                Some(t) => {
                    let _ = t.join();
                }
                None => break,
            }
        }
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

static NEXT_STORE_ID: AtomicUsize = AtomicUsize::new(1);

/// Owns the textures of the displayed images. Cheap to clone; every clone is
/// the same store. Several views may show the same key (a camera view and its
/// detail view).
#[derive(Clone)]
pub struct ImageStore {
    shared: Arc<Shared>,
    _worker: Arc<WorkerHandle>,
}

impl ImageStore {
    /// A store on the device of `render_state` (eframe's
    /// `CreationContext::wgpu_render_state`), with one upload thread. Pass the
    /// egui context so that a finished upload requests a repaint.
    pub fn new(render_state: &egui_wgpu::RenderState, repaint: Option<egui::Context>) -> Self {
        let pipelines = Arc::new(Pipelines::new(
            &render_state.device,
            render_state.target_format,
        ));
        let shared = Arc::new(Shared {
            device: render_state.device.clone(),
            queue: render_state.queue.clone(),
            pipelines,
            inner: Mutex::new(Inner::default()),
            wake: Condvar::new(),
            idle: Condvar::new(),
            repaint,
            id: NEXT_STORE_ID.fetch_add(1, Ordering::Relaxed),
            max_texture: render_state.device.limits().max_texture_dimension_2d,
            restarted: Mutex::new(Vec::new()),
            #[cfg(test)]
            write_hook: Mutex::new(None),
        });
        let thread = spawn_worker(&shared).expect("spawn the image upload thread");
        let worker = Arc::new(WorkerHandle {
            shared: shared.clone(),
            thread: Some(thread),
        });
        Self {
            shared,
            _worker: worker,
        }
    }

    /// Hands `image` to the upload thread for `key` and returns at once. If an
    /// image for this key is still waiting, it is replaced (and counted as
    /// dropped). Returns true if an earlier image was dropped.
    pub fn upload(&self, key: Id, image: impl Into<DisplayImage>) -> bool {
        self.upload_tagged(key, image, 0)
    }

    /// [`ImageStore::upload`] with a tag of the caller's choosing, which
    /// [`TextureInfo::tag`] reports once the image is in the texture.
    pub fn upload_tagged(&self, key: Id, image: impl Into<DisplayImage>, tag: u64) -> bool {
        let image = image.into();
        let mut inner = self.shared.lock();
        if inner.worker_dead {
            // Nothing would ever take it: upload it now (an error is in `last_error`).
            drop(inner);
            let _ = self.upload_now_tagged(key, image, tag);
            return false;
        }
        let (seq, entry) = inner.hand_over(key);
        let dropped = entry.pending.replace((image, tag, seq)).is_some();
        if dropped {
            inner.stats.dropped += 1;
        } else {
            inner.order.push_back(key);
        }
        drop(inner);
        self.shared.wake.notify_one();
        dropped
    }

    /// Uploads `image` on the calling thread before returning (for a first image,
    /// or tests). A waiting image for the same key is discarded; an upload of the same
    /// key that the worker is writing is waited for, so that the two writes into the
    /// texture cannot end in the wrong order.
    pub fn upload_now(&self, key: Id, image: impl Into<DisplayImage>) -> Result<(), String> {
        self.upload_now_tagged(key, image, 0)
    }

    /// [`ImageStore::upload_now`] with a tag ([`ImageStore::upload_tagged`]).
    ///
    /// # Errors
    /// The image's data does not match its size, or it is larger than the device's largest
    /// texture ([`ImageStore::check`]).
    pub fn upload_now_tagged(
        &self,
        key: Id,
        image: impl Into<DisplayImage>,
        tag: u64,
    ) -> Result<(), String> {
        let image = image.into();
        let (current, seq) = {
            let mut inner = self.shared.lock();
            while inner.busy && inner.busy_key == Some(key) {
                inner = self
                    .shared
                    .idle
                    .wait(inner)
                    .unwrap_or_else(|e| e.into_inner());
            }
            let (seq, entry) = inner.hand_over(key);
            if entry.pending.take().is_some() {
                inner.order.retain(|k| *k != key);
            }
            (inner.entries.get(&key).and_then(|e| e.gpu.clone()), seq)
        };
        let result = write(&self.shared, current, &image);
        finish(&self.shared, key, result, tag, seq)
    }

    /// Whether `image` can be uploaded: its data matches its size, and it is no wider and
    /// no taller than the device's largest texture. The text of the error is the one an
    /// upload would give.
    ///
    /// # Errors
    /// What is wrong with the image.
    pub fn check(&self, image: &DisplayImage) -> Result<(), String> {
        fits(&self.shared, image)
    }

    /// Removes the texture and any waiting image for `key`.
    pub fn remove(&self, key: Id) {
        let mut inner = self.shared.lock();
        inner.entries.remove(&key);
        inner.order.retain(|k| *k != key);
    }

    /// What the texture for `key` holds, if one has been uploaded.
    pub fn info(&self, key: Id) -> Option<TextureInfo> {
        self.shared
            .lock()
            .entries
            .get(&key)
            .and_then(|e| e.gpu.as_ref().map(|g| g.info))
    }

    /// Upload counters.
    pub fn stats(&self) -> UploadStats {
        self.shared.lock().stats
    }

    /// The last upload error (for example an image whose data does not match its
    /// size), if any.
    pub fn last_error(&self) -> Option<String> {
        self.shared.lock().last_error.clone()
    }

    /// Blocks until no image is waiting and the upload thread is idle. For tests
    /// and screenshots; the interface never needs it.
    pub fn wait_idle(&self) {
        let mut inner = self.shared.lock();
        while inner.busy || (!inner.order.is_empty() && !inner.worker_dead) {
            inner = self
                .shared
                .idle
                .wait(inner)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    pub(crate) fn texture(&self, key: Id) -> Option<TextureGpu> {
        self.shared
            .lock()
            .entries
            .get(&key)
            .and_then(|e| e.gpu.clone())
    }

    /// The bind group of `key`'s texture (layout [`ImageStore::texture_layout`]), whether
    /// it is RGBA, and its bit depth: for other GPU drawings of the same image (the 3D
    /// world view's camera images, `widgets::scene3d`).
    pub(crate) fn texture_binding(&self, key: Id) -> Option<(wgpu::BindGroup, bool, u8)> {
        self.texture(key)
            .map(|g| (g.bind_group.clone(), g.meta.rgba, g.meta.bit_depth))
    }

    /// The layout of the textures' bind groups (one `texture_2d<u32>` at binding 0).
    pub(crate) fn texture_layout(&self) -> wgpu::BindGroupLayout {
        self.shared.pipelines.texture_layout.clone()
    }

    /// egui's target format.
    pub(crate) fn target_format(&self) -> wgpu::TextureFormat {
        self.shared.pipelines.target_format
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.shared.device
    }

    pub(crate) fn pipelines(&self) -> &Pipelines {
        &self.shared.pipelines
    }

    pub(crate) fn id(&self) -> usize {
        self.shared.id
    }
}

fn spawn_worker(shared: &Arc<Shared>) -> std::io::Result<std::thread::JoinHandle<()>> {
    let worker_shared = shared.clone();
    std::thread::Builder::new()
        .name("image-view upload".into())
        .spawn(move || worker(worker_shared))
}

/// Starts a new upload thread when the upload thread unwinds (a panic in wgpu, which treats
/// a validation error as fatal on the thread that made the call), so that the images handed
/// over are still uploaded. When none can be started, images are uploaded on the thread
/// that hands them over (`Inner::worker_dead`).
struct Restart<'a>(&'a Arc<Shared>);

impl Drop for Restart<'_> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            return;
        }
        let shared = self.0;
        if shared.lock().shutdown {
            return;
        }
        let started = spawn_worker(shared);
        let mut inner = shared.lock();
        match started {
            Ok(t) => {
                inner.last_error =
                    Some("The image upload thread stopped on an error and was restarted.".into());
                shared
                    .restarted
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(t);
            }
            Err(e) => {
                inner.last_error = Some(format!(
                    "The image upload thread stopped on an error and cannot be restarted \
                     ({e}); images are uploaded as they are handed over."
                ));
                inner.worker_dead = true;
            }
        }
        drop(inner);
        shared.wake.notify_all();
        shared.idle.notify_all();
    }
}

/// The upload thread. Takes ownership of its `Shared` reference so that the
/// reference is dropped when the thread returns, before `join` completes.
fn worker(shared: Arc<Shared>) {
    let _restart = Restart(&shared);
    let shared = &*shared;
    loop {
        let job = {
            let mut inner = shared.lock();
            loop {
                if inner.shutdown {
                    return;
                }
                if let Some(job) = take_job(&mut inner) {
                    break job;
                }
                if inner.order.is_empty() {
                    shared.idle.notify_all();
                    inner = shared.wake.wait(inner).unwrap_or_else(|e| e.into_inner());
                }
            }
        };
        run_job(shared, job);
        if let Some(ctx) = &shared.repaint {
            ctx.request_repaint_of(egui::ViewportId::ROOT);
        }
    }
}

/// An upload the worker has taken.
struct Job {
    key: Id,
    image: DisplayImage,
    tag: u64,
    seq: u64,
    current: Option<TextureGpu>,
}

/// Takes the next waiting image off the queue and marks the worker busy with its key.
/// `None` when the key at the front had nothing waiting (it is popped) or the queue is
/// empty.
fn take_job(inner: &mut Inner) -> Option<Job> {
    let key = inner.order.pop_front()?;
    let entry = inner.entries.get_mut(&key)?;
    let (image, tag, seq) = entry.pending.take()?;
    let current = entry.gpu.clone();
    inner.busy = true;
    inner.busy_key = Some(key);
    Some(Job {
        key,
        image,
        tag,
        seq,
        current,
    })
}

/// Marks the worker idle and wakes those waiting for it when dropped, also when the job
/// unwinds: an `upload_now` of the key must not wait for a job that will never end.
struct Idle<'a>(&'a Shared);

impl Drop for Idle<'_> {
    fn drop(&mut self) {
        {
            let mut inner = self.0.lock();
            inner.busy = false;
            inner.busy_key = None;
        }
        self.0.idle.notify_all();
    }
}

/// Writes a taken upload, stores the result and marks the worker idle.
fn run_job(shared: &Shared, job: Job) {
    let _idle = Idle(shared);
    let Job {
        key,
        image,
        tag,
        seq,
        current,
    } = job;
    let result = write(shared, current, &image);
    drop(image);
    let _ = finish(shared, key, result, tag, seq);
}

type WriteResult = Result<(TextureGpu, usize, f64), String>;

/// [`DisplayImage::check`], and the device's largest texture: a larger one is refused here,
/// before wgpu sees it.
fn fits(shared: &Shared, image: &DisplayImage) -> Result<(), String> {
    image.check()?;
    let (w, h, max) = (image.width(), image.height(), shared.max_texture);
    if w > max || h > max {
        return Err(format!(
            "The image is {w}x{h} pixels; this graphics device shows images of at most \
             {max}x{max} pixels"
        ));
    }
    Ok(())
}

/// Writes `image` into the current texture if it has the same size and format,
/// otherwise into a new texture.
fn write(shared: &Shared, current: Option<TextureGpu>, image: &DisplayImage) -> WriteResult {
    fits(shared, image)?;
    #[cfg(test)]
    if let Some(hook) = &*shared.write_hook.lock().unwrap_or_else(|e| e.into_inner()) {
        hook(image);
    }
    let start = Instant::now();
    let (w, h, format) = (image.width(), image.height(), image.format());
    let meta = image.meta();
    let reuse = current.filter(|g| g.info.width == w && g.info.height == h && g.format == format);
    let mut gpu = match reuse {
        Some(g) => g,
        None => {
            let texture = shared.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("image view texture"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let bind_group = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("image view texture"),
                layout: &shared.pipelines.texture_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                }],
            });
            TextureGpu {
                texture,
                bind_group,
                info: TextureInfo {
                    width: w,
                    height: h,
                    rgba: meta.rgba,
                    bit_depth: meta.bit_depth,
                    generation: 0,
                    tag: 0,
                },
                meta,
                format,
            }
        }
    };
    let mirrored = image.mirrored();
    let bytes = mirrored.as_ref().unwrap_or(image).bytes();
    let bpp = format.block_copy_size(None).unwrap_or(1);
    shared.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &gpu.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * bpp),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.meta = meta;
    gpu.info.rgba = meta.rgba;
    gpu.info.bit_depth = meta.bit_depth;
    Ok((gpu, bytes.len(), start.elapsed().as_secs_f64()))
}

/// Stores the texture of the upload with hand-over number `seq`, unless the key already
/// shows a later one (or was removed).
fn finish(shared: &Shared, key: Id, result: WriteResult, tag: u64, seq: u64) -> Result<(), String> {
    let mut inner = shared.lock();
    match result {
        Ok((mut gpu, bytes, secs)) => {
            inner.stats.uploaded += 1;
            inner.stats.bytes += bytes as u64;
            inner.stats.upload_secs += secs;
            inner.stats.last_upload_secs = secs;
            // A key removed while its upload ran stays removed.
            if let Some(entry) = inner.entries.get_mut(&key)
                && seq > entry.shown
            {
                entry.shown = seq;
                let generation = entry.gpu.as_ref().map_or(0, |g| g.info.generation);
                gpu.info.generation = generation + 1;
                gpu.info.tag = tag;
                entry.gpu = Some(gpu);
            }
            Ok(())
        }
        Err(e) => {
            inner.last_error = Some(e.clone());
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_state() -> Option<egui_wgpu::RenderState> {
        std::panic::catch_unwind(|| {
            egui_kittest::wgpu::create_render_state(
                egui_kittest::wgpu::default_wgpu_setup(),
                egui_wgpu::RendererOptions::PREDICTABLE,
            )
        })
        .ok()
    }

    fn image(v: u8) -> DisplayImage {
        DisplayImage::Rgba8 {
            width: 2,
            height: 2,
            data: vec![v; 16],
        }
    }

    /// Records the first byte of every image written, in the order of the writes.
    fn record_writes(store: &ImageStore) -> Arc<Mutex<Vec<u8>>> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        *store.shared.write_hook.lock().unwrap() = Some(Box::new(move |im: &DisplayImage| {
            s.lock().unwrap().push(im.bytes()[0]);
        }));
        seen
    }

    /// Review 12 (app-widgets-panels finding 10): the worker has taken key K's image A when
    /// `upload_now_tagged(K, B)` runs. However the two writes interleave, K ends showing B.
    ///
    /// Review 12c: both images have the texture's size, so both writes go into the same
    /// texture, and what K shows is the write that came last. The pixels are not read back;
    /// the order of the writes is (a hook at the start of `write`): `upload_now` waits for
    /// the worker's write of A, so B is written after it. Without the wait B is written
    /// during the sleep, before A.
    #[test]
    fn review12_upload_now_after_a_taken_upload_wins() {
        let Some(rs) = render_state() else {
            eprintln!("SKIPPED: no wgpu adapter");
            return;
        };
        let store = ImageStore::new(&rs, None);
        let key = Id::unique("review12 key");
        store.upload_now_tagged(key, image(0), 1).unwrap();
        let writes = record_writes(&store);
        // Hand A over without waking the worker, and take it as the worker does: this
        // thread now plays the worker for this one upload.
        let job = {
            let mut inner = store.shared.lock();
            let (seq, entry) = inner.hand_over(key);
            entry.pending = Some((image(10), 2, seq));
            inner.order.push_back(key);
            take_job(&mut inner).expect("job")
        };
        // The interface thread hands B over meanwhile.
        let s2 = store.clone();
        let ui = std::thread::spawn(move || s2.upload_now_tagged(key, image(20), 3));
        std::thread::sleep(std::time::Duration::from_millis(200));
        // The worker's write of A ends last.
        run_job(&store.shared, job);
        ui.join().unwrap().unwrap();
        store.wait_idle();
        let info = store.info(key).expect("texture");
        assert_eq!(info.tag, 3, "K went back to the older image");
        assert_eq!(
            *writes.lock().unwrap(),
            [10, 20],
            "B was written into K's texture before the worker's write of A ended"
        );
    }

    /// Review 12c: a job that panics in its write (wgpu panics on a validation error) still
    /// releases its key, so an `upload_now` of that key does not wait for ever.
    #[test]
    fn review12c_a_job_that_panics_releases_its_key() {
        let Some(rs) = render_state() else {
            eprintln!("SKIPPED: no wgpu adapter");
            return;
        };
        let store = ImageStore::new(&rs, None);
        let key = Id::unique("review12c panic key");
        *store.shared.write_hook.lock().unwrap() = Some(Box::new(|im: &DisplayImage| {
            assert_ne!(im.bytes()[0], 99, "a write that fails");
        }));
        // This thread plays the worker for one upload, which panics.
        let job = {
            let mut inner = store.shared.lock();
            let (seq, entry) = inner.hand_over(key);
            entry.pending = Some((image(99), 2, seq));
            inner.order.push_back(key);
            take_job(&mut inner).expect("job")
        };
        let shared = store.shared.clone();
        let panicked = std::thread::spawn(move || run_job(&shared, job)).join();
        assert!(panicked.is_err(), "the write panicked");
        let (tx, rx) = std::sync::mpsc::channel();
        let s2 = store.clone();
        std::thread::spawn(move || {
            let _ = tx.send(s2.upload_now_tagged(key, image(20), 3));
        });
        let r = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("upload_now waited for a job that had panicked");
        r.unwrap();
        assert_eq!(store.info(key).map(|i| i.tag), Some(3));
    }

    /// Review 12c: when the upload thread panics, another takes over: images handed over
    /// afterwards are uploaded, and the error says what happened.
    #[test]
    fn review12c_the_upload_thread_is_restarted_after_a_panic() {
        let Some(rs) = render_state() else {
            eprintln!("SKIPPED: no wgpu adapter");
            return;
        };
        let store = ImageStore::new(&rs, None);
        *store.shared.write_hook.lock().unwrap() = Some(Box::new(|im: &DisplayImage| {
            assert_ne!(im.bytes()[0], 99, "a write that fails");
        }));
        let bad = Id::unique("review12c bad");
        let good = Id::unique("review12c good");
        store.upload_tagged(bad, image(99), 1);
        // The key is released while the job unwinds, a moment before the thread ends and
        // starts the next one.
        let start = std::time::Instant::now();
        let err = loop {
            if let Some(e) = store.last_error() {
                break e;
            }
            assert!(start.elapsed().as_secs() < 10, "no error after the panic");
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        assert!(err.contains("restarted"), "{err}");
        assert!(store.info(bad).is_none());
        store.upload_tagged(good, image(5), 7);
        store.wait_idle();
        assert_eq!(store.info(good).map(|i| i.tag), Some(7));
        assert_eq!(store.shared.restarted.lock().unwrap().len(), 1);
        // Dropping the store joins the thread that took over.
        drop(store);
    }

    /// Review 12c: an image larger than the device's largest texture never reaches wgpu; it is
    /// refused on both paths, as an image whose data does not match its size is.
    #[test]
    fn review12c_an_image_larger_than_the_device_allows_is_refused() {
        let Some(rs) = render_state() else {
            eprintln!("SKIPPED: no wgpu adapter");
            return;
        };
        let store = ImageStore::new(&rs, None);
        let max = rs.device.limits().max_texture_dimension_2d;
        let big = |w: u32, h: u32| DisplayImage::Rgba8 {
            width: w,
            height: h,
            data: vec![0; w as usize * h as usize * 4],
        };
        let key = Id::unique("review12c big");
        let wide = store.check(&big(max + 1, 1)).unwrap_err();
        assert!(wide.contains(&format!("{}x1 pixels", max + 1)), "{wide}");
        assert!(store.check(&big(1, max + 1)).is_err());
        assert!(store.check(&big(max, 1)).is_ok());
        assert_eq!(store.upload_now(key, big(max + 1, 1)), Err(wide.clone()));
        store.upload(key, big(1, max + 1));
        store.wait_idle();
        assert!(store.last_error().unwrap().contains("this graphics device"));
        assert!(store.info(key).is_none());
        // The store still works.
        store.upload(key, image(1));
        store.wait_idle();
        assert!(store.info(key).is_some());
        assert!(
            store.shared.restarted.lock().unwrap().is_empty(),
            "nothing panicked"
        );
    }

    /// The other order: an upload the worker finishes after a newer `upload_now` result
    /// was stored is dropped.
    #[test]
    fn review12_older_result_is_dropped() {
        let Some(rs) = render_state() else {
            eprintln!("SKIPPED: no wgpu adapter");
            return;
        };
        let store = ImageStore::new(&rs, None);
        let key = Id::unique("review12 key 2");
        let (old, new) = {
            let mut inner = store.shared.lock();
            let (a, _) = inner.hand_over(key);
            let (b, _) = inner.hand_over(key);
            (a, b)
        };
        let w = |v| write(&store.shared, None, &image(v));
        finish(&store.shared, key, w(20), 3, new).unwrap();
        finish(&store.shared, key, w(10), 2, old).unwrap();
        let info = store.info(key).expect("texture");
        assert_eq!((info.tag, info.generation), (3, 1));
    }
}
