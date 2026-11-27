//! Finding and starting the `ffmpeg` and `ffprobe` executables.

use std::ffi::{OsStr, OsString};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::VideoError;

/// Environment variable naming the `ffmpeg` executable.
pub const FFMPEG_ENV: &str = "XMALAB_FFMPEG";
/// Environment variable naming the `ffprobe` executable.
pub const FFPROBE_ENV: &str = "XMALAB_FFPROBE";

/// Most stderr kept from one ffmpeg run, in bytes.
const STDERR_LIMIT: usize = 16 * 1024;

/// The two executables the ffmpeg path runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfmpegTools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl FfmpegTools {
    /// Finds both executables. Each is looked for, in order: the explicit
    /// path given here, the environment variable (`XMALAB_FFMPEG`,
    /// `XMALAB_FFPROBE`), the directory of the running program, then `PATH`
    /// (and, on macOS, Homebrew's directories, which programs started from
    /// the Finder do not have on their `PATH`).
    ///
    /// Following the recommendation of spike 4, `spikes/ffmpeg-scrub/README.md`,
    /// "Detection, versions, missing binary".
    pub fn locate(
        ffmpeg: Option<&Path>,
        ffprobe: Option<&Path>,
    ) -> Result<FfmpegTools, VideoError> {
        Ok(FfmpegTools {
            ffmpeg: find_tool("ffmpeg", ffmpeg, FFMPEG_ENV)?,
            ffprobe: find_tool("ffprobe", ffprobe, FFPROBE_ENV)?,
        })
    }

    /// The first line of `ffmpeg -version`, for example
    /// `ffmpeg version n9.0.2 Copyright (c) 2000-2026 the FFmpeg developers`.
    pub fn version(&self) -> Result<String, VideoError> {
        let out = run_capture(&self.ffmpeg, &[OsString::from("-version")])?;
        let text = String::from_utf8_lossy(&out.stdout);
        Ok(text.lines().next().unwrap_or("").trim().to_string())
    }
}

fn exe_name(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

fn find_tool(name: &str, explicit: Option<&Path>, env: &str) -> Result<PathBuf, VideoError> {
    if let Some(p) = explicit {
        return if p.is_file() {
            Ok(p.to_path_buf())
        } else {
            Err(VideoError::Ffmpeg(format!(
                "{name} not found at the configured path {}",
                p.display()
            )))
        };
    }
    if let Some(v) = std::env::var_os(env).filter(|v| !v.is_empty()) {
        let p = PathBuf::from(v);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(VideoError::Ffmpeg(format!(
                "{name} not found at {} (set by {env})",
                p.display()
            )))
        };
    }
    let file = exe_name(name);
    let mut searched: Vec<PathBuf> = Vec::new();
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        searched.push(dir);
    }
    if let Some(path) = std::env::var_os("PATH") {
        searched.extend(std::env::split_paths(&path));
    }
    if cfg!(target_os = "macos") {
        searched.push(PathBuf::from("/opt/homebrew/bin"));
        searched.push(PathBuf::from("/usr/local/bin"));
    }
    for dir in &searched {
        let candidate = dir.join(&file);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(VideoError::Ffmpeg(format!(
        "{name} not found: it is needed to read this video. Install FFmpeg, put {file} next to \
         the program or on PATH, or set {env} to its full path"
    )))
}

/// A `Command` for `program` that opens no console window on Windows.
pub(crate) fn command(program: &Path) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

fn start_error(program: &Path, e: std::io::Error) -> VideoError {
    VideoError::Ffmpeg(format!("cannot start {}: {e}", program.display()))
}

/// Runs `program` to completion and captures its output.
pub(crate) fn run_capture(
    program: &Path,
    args: &[impl AsRef<OsStr>],
) -> Result<Output, VideoError> {
    command(program)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| start_error(program, e))
}

/// A running child process whose stderr is collected on a thread.
pub(crate) struct Running {
    pub child: Child,
    stderr: Arc<Mutex<Vec<u8>>>,
    stderr_thread: Option<JoinHandle<()>>,
}

impl Running {
    /// Starts `program` with stdout piped and stderr collected.
    pub fn spawn(program: &Path, args: &[OsString]) -> Result<Running, VideoError> {
        let mut child = command(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| start_error(program, e))?;
        let stderr = Arc::new(Mutex::new(Vec::new()));
        let stderr_thread = child.stderr.take().map(|mut pipe| {
            let sink = Arc::clone(&stderr);
            std::thread::Builder::new()
                .name("ffmpeg-stderr".to_string())
                .spawn(move || {
                    let mut buf = [0u8; 4096];
                    while let Ok(n) = pipe.read(&mut buf) {
                        if n == 0 {
                            break;
                        }
                        let mut s = sink.lock().unwrap_or_else(|e| e.into_inner());
                        let room = STDERR_LIMIT.saturating_sub(s.len());
                        s.extend_from_slice(&buf[..n.min(room)]);
                    }
                })
                .expect("spawning a thread")
        });
        Ok(Running {
            child,
            stderr,
            stderr_thread,
        })
    }

    /// Waits for the process to end and returns its exit status text and
    /// everything it wrote to stderr.
    pub fn finish(&mut self) -> String {
        let status = self.child.wait();
        if let Some(t) = self.stderr_thread.take() {
            let _ = t.join();
        }
        let err = self.stderr.lock().unwrap_or_else(|e| e.into_inner());
        let err = String::from_utf8_lossy(&err).trim().to_string();
        let status = match status {
            Ok(s) => s.to_string(),
            Err(e) => e.to_string(),
        };
        if err.is_empty() {
            status
        } else {
            format!("{status}: {err}")
        }
    }

    /// Kills the process and reaps it.
    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(t) = self.stderr_thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.kill();
    }
}
