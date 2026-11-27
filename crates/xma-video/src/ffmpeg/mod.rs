//! A [`FrameSource`] for any video FFmpeg can decode, through the `ffmpeg`
//! and `ffprobe` executables (no linking; `std::process` only).
//!
//! The design is the one spike 4 recommended (`spikes/ffmpeg-scrub/README.md`,
//! "Recommendation"):
//!
//! - **Index.** On opening, one `ffprobe` run lists every packet's timestamp
//!   and keyframe flag ([`FrameIndex`]). Frame `n` is the `n`-th entry in
//!   presentation order; frames are never addressed by `n / frame rate`, which
//!   is wrong on variable-rate files.
//! - **Forward stream.** One `ffmpeg` process decodes forwards into a pipe; a
//!   reader thread converts frames and keeps a few ready. Stepping forwards and
//!   sequential reading read on from it.
//! - **Window.** The most recent decoded frames (32 by default) are kept, so
//!   stepping backwards is served from memory. A backward step that leaves the
//!   window restarts the stream that many frames earlier and reads up to the
//!   target, filling the window again.
//! - **Jumps.** Anything else restarts the stream at the target with the
//!   keyframe-anchored seek: input `-ss` just before the last keyframe at or
//!   before the target, output `-ss` just before the target. It is exact on
//!   open-GOP HEVC, where a plain input seek returns the wrong frame.
//! - **Grey.** Frames are reduced to the 8-bit grey the C++ program holds; see
//!   [`GreyConversion`]. Colour uses [`FfmpegOptions::grey_weights`]: correct by
//!   default, the C++'s swapped weights (ledger V2) in XMALab 3 mode.
//! - **Unreadable frames.** A frame the container counts but that does not decode
//!   (the pig videos declare 146 frames and decode 145, ledger V7): `read_frame`
//!   returns an error, so the viewer shows no image. The C++ shows frame 0 there
//!   (`AviVideo::setActiveFrame` reopens the file, which reads frame 0; measured on an
//!   AVI, dataset `video_avi_raw`).
//!
//! Child processes are killed and reaped when they are replaced and when the
//! source is dropped. A process that fails reports its stderr in the error.

mod grey;
mod probe;
mod tools;

use std::collections::{HashMap, VecDeque};
use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, sync_channel};
use std::thread::JoinHandle;

pub use grey::{GreyConversion, luma_table};
pub use probe::FrameIndex;
pub use tools::{FFMPEG_ENV, FFPROBE_ENV, FfmpegTools};

use grey::Post;
use tools::Running;

use crate::{Frame, FrameSource, GreyWeights, SourceInfo, SourceKind, VideoError};

/// Settings of an [`FfmpegSource`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfmpegOptions {
    /// Decoder threads (`-threads`); `None` lets ffmpeg choose (about one per
    /// core). Spike 4 found 8 threads as fast as the default with less memory.
    pub threads: Option<usize>,
    /// Number of recent decoded frames kept for stepping backwards (at least 1).
    pub window: usize,
    /// Frames the reader thread decodes ahead of the last one asked for.
    pub prefetch: usize,
    /// How frames are reduced to 8-bit grey.
    pub grey: GreyConversion,
    /// The weights of colour to grey (ledger V2) on the BGR path; correct by
    /// default, [`GreyWeights::Swapped`] as the C++.
    pub grey_weights: GreyWeights,
}

impl Default for FfmpegOptions {
    fn default() -> Self {
        FfmpegOptions {
            threads: None,
            window: 32,
            prefetch: 4,
            grey: GreyConversion::Exact,
            grey_weights: GreyWeights::Correct,
        }
    }
}

/// Counters of an [`FfmpegSource`], for tests and measurements.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FfmpegStats {
    /// ffmpeg processes started.
    pub spawns: u64,
    /// Requests served from the window.
    pub window_hits: u64,
    /// Requests served by reading on in the running stream.
    pub stream_reads: u64,
    /// Frames decoded and passed over to reach a requested one.
    pub skipped_frames: u64,
}

/// Largest gap over which a forward request reads on in the stream even when a
/// keyframe lies between (restarting would cost more than decoding a few
/// frames).
const READ_ON_GAP: usize = 8;

/// A video decoded by the `ffmpeg` executable.
pub struct FfmpegSource {
    path: PathBuf,
    tools: FfmpegTools,
    options: FfmpegOptions,
    index: FrameIndex,
    info: SourceInfo,
    out_args: Vec<OsString>,
    post: Post,
    stream: Option<Stream>,
    window: HashMap<usize, Frame>,
    /// Window frames, least recently used first.
    order: VecDeque<usize>,
    last_request: Option<usize>,
    stats: FfmpegStats,
}

impl std::fmt::Debug for FfmpegSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FfmpegSource")
            .field("path", &self.path)
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl FfmpegSource {
    /// Indexes `path` with `ffprobe` and prepares to decode it. No `ffmpeg`
    /// process starts until the first frame is read.
    pub fn open(
        path: &Path,
        tools: &FfmpegTools,
        options: &FfmpegOptions,
    ) -> Result<FfmpegSource, VideoError> {
        if !path.is_file() {
            return Err(VideoError::Io {
                path: path.to_path_buf(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"),
            });
        }
        let index = FrameIndex::probe(tools, path)?;
        Ok(Self::from_index(index, tools, options))
    }

    /// A source over an index made earlier (for example one cached on disk).
    pub fn from_index(index: FrameIndex, tools: &FfmpegTools, options: &FfmpegOptions) -> Self {
        let (out_args, post) = grey::plan(&index, options.grey, options.grey_weights);
        let fps = index.opencv_fps();
        let info = SourceInfo {
            kind: SourceKind::Ffmpeg,
            frame_count: index.cpp_frame_count(),
            readable_frames: index.len(),
            width: index.width,
            height: index.height,
            // OpenCV hands the C++ program 8-bit BGR whatever the video's depth,
            // and frames here are 8-bit for the same reason.
            bit_depth: 8,
            frame_rate: (fps > 0.0).then_some(fps),
        };
        let mut options = options.clone();
        options.window = options.window.max(1);
        options.prefetch = options.prefetch.max(1);
        FfmpegSource {
            path: index.path.clone(),
            tools: tools.clone(),
            options,
            index,
            info,
            out_args,
            post,
            stream: None,
            window: HashMap::new(),
            order: VecDeque::new(),
            last_request: None,
            stats: FfmpegStats::default(),
        }
    }

    /// The frame index made on opening.
    pub fn index(&self) -> &FrameIndex {
        &self.index
    }

    /// Counters since the source was opened.
    pub fn stats(&self) -> FfmpegStats {
        self.stats
    }

    /// The `ffmpeg` arguments that decode from frame `start` onwards.
    ///
    /// Ported from spike 4, `spikes/ffmpeg-scrub/src/source.rs`, `Ff::args`
    /// with `Seek::Key`.
    pub fn decode_args(&self, start: usize) -> Vec<OsString> {
        let idx = &self.index;
        let mut a: Vec<OsString> = ["-hide_banner", "-nostdin", "-loglevel", "error"]
            .iter()
            .map(OsString::from)
            .collect();
        if let Some(t) = self.options.threads {
            a.push("-threads".into());
            a.push(t.to_string().into());
        }
        let t = |v: f64| OsString::from(format!("{v:.6}"));
        let k = idx.keyframe_at_or_before(start);
        let tk = idx.seek_mid(k);
        if let Some(x) = tk {
            a.push("-ss".into());
            a.push(t(x));
        }
        a.push("-i".into());
        a.push(self.path.as_os_str().to_owned());
        if start != k
            && let Some(tn) = idx.seek_mid(start)
        {
            a.push("-ss".into());
            a.push(t(tn - tk.unwrap_or(0.0)));
        }
        for s in [
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-fps_mode",
            "passthrough",
        ] {
            a.push(s.into());
        }
        a.extend(self.out_args.iter().cloned());
        for s in ["-f", "rawvideo", "pipe:1"] {
            a.push(s.into());
        }
        a
    }

    fn start_stream(&mut self, start: usize) -> Result<(), VideoError> {
        // Stop the old process before starting the next one.
        self.stream = None;
        let args = self.decode_args(start);
        let running = Running::spawn(&self.tools.ffmpeg, &args)?;
        self.stats.spawns += 1;
        self.stream = Some(Stream::start(
            running,
            start,
            self.index.len(),
            (self.info.width, self.info.height),
            self.post.clone(),
            self.options.prefetch,
            &self.path,
        ));
        Ok(())
    }

    fn remember(&mut self, n: usize, frame: Frame) {
        if self.window.insert(n, frame).is_some() {
            self.order.retain(|&i| i != n);
        }
        self.order.push_back(n);
        while self.order.len() > self.options.window {
            if let Some(old) = self.order.pop_front() {
                self.window.remove(&old);
            }
        }
    }

    fn touch(&mut self, n: usize) {
        if let Some(pos) = self.order.iter().position(|&i| i == n) {
            self.order.remove(pos);
            self.order.push_back(n);
        }
    }

    /// Reads on in the running stream up to frame `n`, keeping every frame in
    /// the window.
    fn read_on(&mut self, n: usize) -> Result<Frame, VideoError> {
        loop {
            let stream = self.stream.as_mut().expect("a running stream");
            let at = stream.next;
            let got = stream.next_frame();
            match got {
                Ok(frame) => {
                    if at == n {
                        self.remember(n, frame.clone());
                        return Ok(frame);
                    }
                    self.stats.skipped_frames += 1;
                    self.remember(at, frame);
                }
                Err(e) => {
                    self.stream = None;
                    return Err(e);
                }
            }
        }
    }

    fn can_read_on(&self, n: usize) -> bool {
        match &self.stream {
            Some(s) if !s.ended && n >= s.next => {
                n - s.next <= READ_ON_GAP || self.index.keyframe_at_or_before(n) <= s.next
            }
            _ => false,
        }
    }
}

impl FrameSource for FfmpegSource {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    fn read_frame(&mut self, n: usize) -> Result<Frame, VideoError> {
        if n >= self.index.len() {
            return Err(VideoError::FrameOutOfRange {
                index: n,
                readable: self.index.len(),
            });
        }
        let last = self.last_request.replace(n);
        if let Some(f) = self.window.get(&n).cloned() {
            self.stats.window_hits += 1;
            self.touch(n);
            return Ok(f);
        }
        if self.can_read_on(n) {
            self.stats.stream_reads += 1;
            return self.read_on(n);
        }
        // A backward step: decode the window that ends at the target, so the
        // next steps back are in memory and the stream continues forwards.
        let backward = matches!(last, Some(l) if n < l && l - n <= self.options.window);
        let start = if backward {
            (n + 1).saturating_sub(self.options.window)
        } else {
            n
        };
        self.start_stream(start)?;
        self.read_on(n)
    }
}

/// One running `ffmpeg` decoding forwards, with a reader thread that converts
/// frames and keeps up to `prefetch` of them ready.
///
/// Ported from spike 4, `spikes/ffmpeg-scrub/src/source.rs`, `Persistent`.
struct Stream {
    rx: Option<Receiver<Result<Vec<u8>, String>>>,
    /// The frame the next read returns.
    next: usize,
    /// Set once the stream has failed or ended.
    ended: bool,
    width: u32,
    height: u32,
    path: PathBuf,
    // Dropped in this order: the receiver first (a reader blocked on sending
    // then stops), then the process (killed and reaped), then the thread.
    process: std::sync::Arc<std::sync::Mutex<Running>>,
    reader: Option<JoinHandle<()>>,
}

impl Stream {
    fn start(
        mut running: Running,
        start: usize,
        indexed: usize,
        (width, height): (u32, u32),
        post: Post,
        prefetch: usize,
        path: &Path,
    ) -> Stream {
        let pixels = width as usize * height as usize;
        let mut out = running.child.stdout.take().expect("piped stdout");
        let process = std::sync::Arc::new(std::sync::Mutex::new(running));
        let (tx, rx) = sync_channel(prefetch);
        let reader_process = std::sync::Arc::clone(&process);
        let reader = std::thread::Builder::new()
            .name("ffmpeg-reader".to_string())
            .spawn(move || {
                let in_bytes = pixels * post.bytes_per_pixel();
                let mut buf = vec![0u8; in_bytes];
                let mut at = start;
                loop {
                    let got = read_full(&mut out, &mut buf);
                    let msg = match got {
                        Ok(true) => {
                            let mut frame = Vec::with_capacity(pixels);
                            post.apply(&buf, &mut frame);
                            Ok(frame)
                        }
                        Ok(false) | Err(_) => {
                            drop(out);
                            let detail = reader_process
                                .lock()
                                .map(|mut p| p.finish())
                                .unwrap_or_default();
                            let what = match got {
                                Ok(false) => format!(
                                    "ffmpeg stopped before frame {at} (the index lists {indexed} \
                                     frames); exit {detail}"
                                ),
                                _ => {
                                    format!("reading frame {at} from ffmpeg failed; exit {detail}")
                                }
                            };
                            let _ = tx.send(Err(what));
                            return;
                        }
                    };
                    if tx.send(msg).is_err() {
                        return;
                    }
                    at += 1;
                }
            })
            .expect("spawning a thread");
        Stream {
            rx: Some(rx),
            next: start,
            ended: false,
            width,
            height,
            path: path.to_path_buf(),
            process,
            reader: Some(reader),
        }
    }

    fn next_frame(&mut self) -> Result<Frame, VideoError> {
        let at = self.next;
        let err = |reason: String| VideoError::Decode {
            path: self.path.clone(),
            frame: Some(at),
            reason,
        };
        let got = match &self.rx {
            Some(rx) => rx.recv(),
            None => Err(std::sync::mpsc::RecvError),
        };
        match got {
            Ok(Ok(data)) => {
                self.next += 1;
                Ok(Frame::gray8(self.width, self.height, data))
            }
            Ok(Err(reason)) => {
                self.ended = true;
                Err(err(reason))
            }
            Err(_) => {
                self.ended = true;
                Err(err("the ffmpeg reader stopped".to_string()))
            }
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // Unblock a reader waiting to send, then end the process; the reader
        // then sees the pipe close.
        drop(self.rx.take());
        // The reader holds the lock only while the process is ending.
        if let Ok(mut p) = self.process.lock() {
            p.kill();
        }
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
    }
}

/// Fills `buf` from `r`. `Ok(false)` when the stream ended cleanly before the
/// first byte; an error when it ended part-way.
fn read_full(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) => {
                return if filled == 0 {
                    Ok(false)
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "ffmpeg ended part-way through a frame",
                    ))
                };
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}
