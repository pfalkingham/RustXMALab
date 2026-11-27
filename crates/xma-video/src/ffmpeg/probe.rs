//! The one-time frame index of a video, read with `ffprobe` from the packet
//! timestamps (nothing is decoded).
//!
//! Frame `n` is the `n`-th packet in presentation order, after packets flagged
//! for discard are dropped. That is the numbering OpenCV's `VideoCapture` gives
//! when reading sequentially, which is what the C++ program's trackers see
//! (spike 4, `spikes/ffmpeg-scrub/README.md`, "Frame counts and indices").

use std::path::{Path, PathBuf};

use super::tools::{FfmpegTools, run_capture};
use crate::VideoError;

/// The presentation timestamps and keyframes of a video's first video stream,
/// plus the stream properties the C++ program reads through OpenCV.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameIndex {
    pub path: PathBuf,
    pub codec: String,
    /// Size of the decoded frames, after the rotation in the stream's display
    /// matrix (OpenCV and the ffmpeg executable both apply it).
    pub width: u32,
    pub height: u32,
    pub pix_fmt: String,
    /// `tv`, `pc` or `unknown`.
    pub color_range: String,
    /// Stream time base as numerator and denominator.
    pub time_base: (i64, i64),
    pub r_frame_rate: (i64, i64),
    pub avg_frame_rate: (i64, i64),
    /// The container's frame count (`nb_frames`), when it states one.
    pub nb_frames: Option<u64>,
    /// Container duration in seconds (`AVFormatContext::duration`).
    pub format_duration: Option<f64>,
    /// Stream duration in time-base units.
    pub duration_ts: Option<i64>,
    /// Container start time in seconds; the ffmpeg executable's `-ss` is
    /// relative to it.
    pub start_time: f64,
    /// Rotation of the display matrix in degrees, 0 when there is none.
    pub rotation: i32,
    /// Presentation timestamps in stream time-base units, ascending; frame `n`
    /// has `pts[n]`.
    pub pts: Vec<i64>,
    /// Keyframe flag of each frame, in presentation order.
    pub key: Vec<bool>,
    /// Packets flagged for discard and left out of the index.
    pub discarded_packets: usize,
}

fn ratio(s: &str) -> (i64, i64) {
    let mut it = s.split('/');
    let a = it.next().and_then(|v| v.trim().parse().ok()).unwrap_or(0);
    let b = it.next().and_then(|v| v.trim().parse().ok()).unwrap_or(1);
    (a, b)
}

fn r2d(r: (i64, i64)) -> f64 {
    if r.0 == 0 || r.1 == 0 {
        0.0
    } else {
        r.0 as f64 / r.1 as f64
    }
}

/// One line of `ffprobe -of compact`: `section|key=value|key=value...`.
struct Line<'a> {
    section: &'a str,
    fields: Vec<(&'a str, &'a str)>,
}

impl<'a> Line<'a> {
    fn parse(line: &'a str) -> Line<'a> {
        let mut parts = line.split('|');
        let section = parts.next().unwrap_or("");
        let fields = parts.filter_map(|p| p.split_once('=')).collect();
        Line { section, fields }
    }

    /// The value of the first field whose key is `rotation` or ends in
    /// `:rotation` (ffprobe nests side data as
    /// `side_datum/display_matrix:rotation`).
    fn rotation(&self) -> Option<i32> {
        self.fields
            .iter()
            .find(|(k, _)| *k == "rotation" || k.ends_with(":rotation"))
            .and_then(|(_, v)| v.parse::<f64>().ok())
            .map(|r| r.round() as i32)
    }

    fn get(&self, key: &str) -> Option<&'a str> {
        self.fields
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| *v)
            .filter(|v| !v.is_empty() && *v != "N/A")
    }
}

impl FrameIndex {
    /// Builds the index of the first video stream of `path` with one `ffprobe`
    /// run.
    ///
    /// Ported from spike 4, `spikes/ffmpeg-scrub/src/probe.rs`, `build`, with
    /// the container start time taken from the format (what `-ss` is relative
    /// to), the fields OpenCV's frame count and frame rate need, and packets
    /// without a presentation timestamp ordered by their decoding timestamp.
    pub fn probe(tools: &FfmpegTools, path: &Path) -> Result<FrameIndex, VideoError> {
        let mut args: Vec<std::ffi::OsString> = [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,color_range,time_base,r_frame_rate,\
             avg_frame_rate,nb_frames,duration_ts:stream_side_data=rotation:\
             format=start_time,duration:packet=pts,dts,flags",
            "-of",
            "compact",
        ]
        .iter()
        .map(Into::into)
        .collect();
        args.push(path.as_os_str().to_owned());
        let out = run_capture(&tools.ffprobe, &args)?;
        if !out.status.success() {
            return Err(VideoError::Ffmpeg(format!(
                "ffprobe could not read {}: {}",
                path.display(),
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        Self::parse(path, &text)
    }

    /// Parses the output of the `ffprobe` command in [`FrameIndex::probe`].
    pub fn parse(path: &Path, text: &str) -> Result<FrameIndex, VideoError> {
        let mut stream: Option<Line> = None;
        let mut format: Option<Line> = None;
        let mut rotation = 0i32;
        // (pts, dts, keyframe) in decoding order
        let mut packets: Vec<(Option<i64>, Option<i64>, bool)> = Vec::new();
        let mut discarded = 0;
        for raw in text.lines() {
            let line = Line::parse(raw.trim());
            match line.section {
                "packet" => {
                    let flags = line.get("flags").unwrap_or("");
                    if flags.contains('D') {
                        discarded += 1;
                        continue;
                    }
                    let pts = line.get("pts").and_then(|v| v.parse().ok());
                    let dts = line.get("dts").and_then(|v| v.parse().ok());
                    packets.push((pts, dts, flags.contains('K')));
                }
                "stream" => {
                    if let Some(r) = line.rotation() {
                        rotation = r;
                    }
                    if stream.is_none() {
                        stream = Some(line);
                    }
                }
                "side_data" => {
                    if let Some(r) = line.rotation() {
                        rotation = r;
                    }
                }
                "format" => format = Some(line),
                _ => {}
            }
        }
        let Some(stream) = stream else {
            return Err(VideoError::Unsupported {
                path: path.to_path_buf(),
                reason: "no video stream".to_string(),
            });
        };
        let dim = |k: &str| -> Result<u32, VideoError> {
            stream
                .get(k)
                .and_then(|v| v.parse().ok())
                .filter(|&v: &u32| v > 0)
                .ok_or_else(|| VideoError::Corrupt {
                    path: path.to_path_buf(),
                    reason: format!("ffprobe reports no frame {k}"),
                })
        };
        let (mut width, mut height) = (dim("width")?, dim("height")?);
        if rotation.rem_euclid(180) == 90 {
            std::mem::swap(&mut width, &mut height);
        }

        // Presentation order. When every packet has a presentation timestamp
        // they are sorted by it; containers that store none (some AVI and raw
        // streams) present in decoding order.
        let all_pts = packets.iter().all(|p| p.0.is_some());
        let all_dts = packets.iter().all(|p| p.1.is_some());
        let mut frames: Vec<(i64, bool)> = if all_pts {
            packets.iter().map(|p| (p.0.unwrap_or(0), p.2)).collect()
        } else if all_dts {
            packets.iter().map(|p| (p.1.unwrap_or(0), p.2)).collect()
        } else {
            return Err(VideoError::Unsupported {
                path: path.to_path_buf(),
                reason: "the video's packets carry no timestamps, so frames cannot be addressed"
                    .to_string(),
            });
        };
        frames.sort_by_key(|f| f.0);
        if let Some(first) = frames.first_mut() {
            // Decoding always starts at the first frame, so it serves as a
            // keyframe for seeking even when the container does not flag it.
            first.1 = true;
        }

        let start_time = format
            .as_ref()
            .and_then(|f| f.get("start_time"))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        let format_duration = format
            .as_ref()
            .and_then(|f| f.get("duration"))
            .and_then(|v| v.parse().ok());
        Ok(FrameIndex {
            path: path.to_path_buf(),
            codec: stream.get("codec_name").unwrap_or("").to_string(),
            width,
            height,
            pix_fmt: stream.get("pix_fmt").unwrap_or("").to_string(),
            color_range: stream.get("color_range").unwrap_or("unknown").to_string(),
            time_base: ratio(stream.get("time_base").unwrap_or("0/1")),
            r_frame_rate: ratio(stream.get("r_frame_rate").unwrap_or("0/1")),
            avg_frame_rate: ratio(stream.get("avg_frame_rate").unwrap_or("0/1")),
            nb_frames: stream.get("nb_frames").and_then(|v| v.parse().ok()),
            format_duration,
            duration_ts: stream.get("duration_ts").and_then(|v| v.parse().ok()),
            start_time,
            rotation,
            pts: frames.iter().map(|f| f.0).collect(),
            key: frames.iter().map(|f| f.1).collect(),
            discarded_packets: discarded,
        })
    }

    /// Number of frames in the index (the frames that can be read).
    pub fn len(&self) -> usize {
        self.pts.len()
    }

    /// True when the stream has no frames.
    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    fn secs(&self, ticks: f64) -> f64 {
        ticks * r2d(self.time_base)
    }

    /// Seek time half-way between frame `n - 1` and frame `n`, in seconds
    /// relative to the container start; `None` for frame 0.
    ///
    /// Ported from `spikes/ffmpeg-scrub/src/probe.rs`, `Index::seek_mid`.
    pub fn seek_mid(&self, n: usize) -> Option<f64> {
        if n == 0 {
            return None;
        }
        let mid = (self.pts[n - 1] as f64 + self.pts[n] as f64) / 2.0;
        Some(self.secs(mid) - self.start_time)
    }

    /// The last keyframe at or before frame `n`, in presentation order.
    ///
    /// Ported from `spikes/ffmpeg-scrub/src/probe.rs`,
    /// `Index::keyframe_at_or_before`.
    pub fn keyframe_at_or_before(&self, n: usize) -> usize {
        (0..=n.min(self.len().saturating_sub(1)))
            .rev()
            .find(|&i| self.key[i])
            .unwrap_or(0)
    }

    /// Indices of all keyframes.
    pub fn keyframes(&self) -> Vec<usize> {
        (0..self.len()).filter(|&i| self.key[i]).collect()
    }

    /// The frame rate OpenCV reports (`CAP_PROP_FPS`): the stream's average
    /// frame rate, else its real base frame rate (the first choice of FFmpeg's
    /// `av_guess_frame_rate`), else the inverse of the time base.
    ///
    /// Ported from OpenCV 4.13 `modules/videoio/src/cap_ffmpeg_impl.hpp`,
    /// `CvCapture_FFMPEG::get_fps`.
    pub fn opencv_fps(&self) -> f64 {
        const EPS_ZERO: f64 = 0.000025;
        let mut fps = r2d(self.avg_frame_rate);
        if fps < EPS_ZERO {
            fps = r2d(self.r_frame_rate);
        }
        if fps < EPS_ZERO {
            let tb = r2d(self.time_base);
            fps = if tb > 0.0 { 1.0 / tb } else { 0.0 };
        }
        fps
    }

    /// The frame count OpenCV reports (`CAP_PROP_FRAME_COUNT`): the container's
    /// `nb_frames`, else `floor(duration * fps + 0.5)`.
    ///
    /// Ported from OpenCV 4.13 `cap_ffmpeg_impl.hpp`,
    /// `CvCapture_FFMPEG::get_total_frames` and `get_duration_sec`.
    pub fn opencv_frame_count(&self) -> f64 {
        const EPS_ZERO: f64 = 0.000025;
        if let Some(n) = self.nb_frames.filter(|&n| n > 0) {
            return n as f64;
        }
        let mut sec = self.format_duration.unwrap_or(0.0);
        if sec < EPS_ZERO {
            sec = self.duration_ts.unwrap_or(0) as f64 * r2d(self.time_base);
        }
        (sec * self.opencv_fps() + 0.5).floor()
    }

    /// The number of frames the C++ program reports for the video:
    /// `(int)(CAP_PROP_FRAME_COUNT + 0.45)` (parity ledger V7).
    ///
    /// Ported from `core/AviVideo.cpp`, `AviVideo::reloadFile`.
    pub fn cpp_frame_count(&self) -> usize {
        let n = (self.opencv_frame_count() + 0.45) as i64;
        n.max(0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PIG: &str = "packet|pts=0|dts=-1000|flags=K__
packet|pts=3000|dts=0|flags=___
packet|pts=1000|dts=1000|flags=___
packet|pts=2000|dts=2000|flags=___
packet|pts=4000|dts=3000|flags=K__
packet|pts=5000|dts=4000|flags=_D_
stream|codec_name=h264|width=1024|height=1024|pix_fmt=yuv420p|color_range=tv|time_base=1/30000|r_frame_rate=30/1|avg_frame_rate=30/1|duration_ts=6000|nb_frames=6
format|start_time=0.000000|duration=0.200000
";

    #[test]
    fn parses_compact_output() {
        let idx = FrameIndex::parse(Path::new("x.mp4"), PIG).unwrap();
        assert_eq!(idx.pts, vec![0, 1000, 2000, 3000, 4000]);
        assert_eq!(idx.key, vec![true, false, false, false, true]);
        assert_eq!(idx.discarded_packets, 1);
        assert_eq!((idx.width, idx.height), (1024, 1024));
        assert_eq!(idx.cpp_frame_count(), 6);
        assert_eq!(idx.opencv_fps(), 30.0);
        assert_eq!(idx.keyframe_at_or_before(3), 0);
        assert_eq!(idx.keyframe_at_or_before(4), 4);
        let mid = idx.seek_mid(1).unwrap();
        assert!((mid - 0.5 / 30.0).abs() < 1e-12);
    }

    #[test]
    fn frame_count_from_duration_without_nb_frames() {
        let text = PIG.replace("|nb_frames=6", "|nb_frames=N/A");
        let idx = FrameIndex::parse(Path::new("x.mkv"), &text).unwrap();
        // 0.2 s at 30 fps
        assert_eq!(idx.cpp_frame_count(), 6);
    }

    #[test]
    fn rotation_swaps_size() {
        let text = PIG
            .replace("width=1024|height=1024", "width=640|height=480")
            .replace(
                "nb_frames=6",
                "nb_frames=6|side_datum/display_matrix:rotation=-90",
            );
        let idx = FrameIndex::parse(Path::new("x.mp4"), &text).unwrap();
        assert_eq!((idx.width, idx.height), (480, 640));
    }

    #[test]
    fn no_video_stream() {
        let err = FrameIndex::parse(Path::new("x.wav"), "format|start_time=0|duration=1\n");
        assert!(matches!(err, Err(VideoError::Unsupported { .. })));
    }
}
