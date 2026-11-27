//! File > Undistort sequence: undistorted copies of image folders and videos, one camera's
//! grid (and lens model) applied to every frame and each frame written as a TIFF.
//!
//! Ported from `ui/UndistortSequenceDialog.cpp` (`on_pushButtonUndist_clicked`,
//! `compareNames2`), `core/VideoStream.cpp` (`getFileBasename`) and
//! `core/UndistortionObject.cpp`, `UndistortionObject::undistort(Image*, QString, bool)`
//! (`docs/cpp-calibration.md`, C64). What XMALab does:
//!
//! - The cameras offered are those with a computed undistortion grid ([`undistort_cameras`]).
//! - An item is a folder of images (`*.png *.tif *.bmp *.jpeg *.jpg`, any case, sorted with
//!   `compareNames2`, the natural order of `littleHelper::compareNames`) or a video file
//!   (`*.cine`, read by `CineVideo`; `*.avi *.mp4 *.mov`, read by `AviVideo` through
//!   OpenCV's FFmpeg; another extension drops the item).
//! - Frame `i` (0-based) of an item is written as `<output>/<base>_UND.<i + 1>.tif`, the number
//!   zero-padded to four digits ([`output_path`]), where `<base>` is `QFileInfo(prefix)
//!   .completeBaseName()` of the longest start of the first file name that every file name
//!   contains ([`file_basename`]): for a video its file name without the last extension, for
//!   a folder `img_0` for `img_0001.tif` .. `img_0100.tif`. Existing files are replaced.
//! - Each frame is the image as `core/Image.cpp` holds it (8 bits; a colour still or a frame
//!   of `AviVideo` in colour, B, G, R; a `.cine` frame grey), remapped with the grid's lookup
//!   table and then, when the camera has a lens model, with the lens maps
//!   (`cv::remap`, `INTER_LANCZOS4`, `BORDER_CONSTANT` 0: [`xma_vision::remap`]), and written
//!   with `cv::imwrite`: an LZW TIFF of 8-bit samples, one channel for a grey image, three
//!   (R, G, B) for a colour one. A frame whose size differs from the grid's is skipped.
//!
//! # Differences (parity ledger V20 to V24)
//!
//! In the default mode (each kept as XMALab has it in the XMALab 3 mode, `CppCompat::XMALAB3`):
//!
//! - V20: a source deeper than 8 bits (16-bit grey stills, `.cine` above 8 bits) is remapped
//!   and written at its own depth (OpenCV's 16-bit remap, each remap saturated at the depth's
//!   largest value), a 16-bit grey TIFF stating `MaxSampleValue` below 16 bits (ledger V1);
//!   XMALab reduces it to 8 bits first.
//! - V21: a flipped camera's frames are mirrored before the remap and mirrored back after it,
//!   as the camera's grid was computed on its mirrored grid image; XMALab remaps the unmirrored
//!   frame.
//! - V22: a video frame that cannot be read is skipped and reported; XMALab writes the
//!   previous frame again under its number.
//! - V23: folders are also searched for `*.tiff`; a video of another extension is opened by
//!   its content (XMALab drops the item, U123).
//!
//! In both modes (V25): an item whose output names include one of its own inputs (an
//! output folder that is the input folder, holding an earlier run's outputs) is refused
//! before anything of it is written ([`replaced_inputs`]); XMALab overwrites the input and
//! reads the overwritten file when it reaches it.
//!
//! In both modes (V23, U122): skipped frames (another size, unreadable) and files that cannot
//! be written are reported, an empty folder is reported (XMALab crashes), files are written
//! whole (a temporary name, then renamed, on a second thread) so that a cancelled run leaves
//! complete files only, and the natural order compares numbers of any length (U100). The
//! video readers are this program's (ledger V2, V7, V10, V12), which read the frame a number
//! names (XMALab 3's seek does not on the pig's H.264 video, V22); a colour video is written
//! grey (three equal channels, XMALab's channel layout), where XMALab writes its colour (V24).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::SyncSender;

use xma_model::CppCompat;
use xma_video::image_file::{TiffSamples, decode_colour_image, decode_image_with, encode_tiff};
use xma_video::open::{OpenOptions as VideoOptions, open_video};
use xma_video::{Frame, Pixels, SourceKind, VideoCompat};
use xma_vision::remap::{lens_undistortion_maps, remap_lanczos4_u8, remap_lanczos4_u16};
use xma_vision::undistortion::{GridModel, LookupTable, create_lookup_table};

use crate::Document;

/// The image types a folder is searched for (`QDir::entryList` name filters; any case).
pub const FOLDER_IMAGE_TYPES: [&str; 5] = ["png", "tif", "bmp", "jpeg", "jpg"];
/// The video types "Add Video File" offers (`Video Files (*.cine *.avi *.mp4 *.mov)`).
pub const VIDEO_TYPES: [&str; 4] = ["cine", "avi", "mp4", "mov"];

/// What one queued item reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UndistortInput {
    /// Every image of a folder ("Add Image Folder").
    Folder(PathBuf),
    /// One video file ("Add Video File").
    Video(PathBuf),
}

impl UndistortInput {
    /// The folder or file.
    #[must_use]
    pub fn path(&self) -> &Path {
        match self {
            UndistortInput::Folder(p) | UndistortInput::Video(p) => p,
        }
    }
}

/// One queued item (`undist_item`): a camera (0-based), an input and an output folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndistortItem {
    pub camera: usize,
    pub input: UndistortInput,
    pub output: PathBuf,
}

impl UndistortItem {
    /// The dialog's list text: `Camera <n> from <input> to <output>` (`n` 1-based).
    ///
    /// Ported from ui/UndistortSequenceDialog.cpp, on_pushButtonAddFolder_clicked.
    #[must_use]
    pub fn list_text(&self) -> String {
        format!(
            "Camera {} from {} to {}",
            self.camera + 1,
            self.input.path().display(),
            self.output.display()
        )
    }
}

/// The cameras (0-based) whose undistortion grid is computed: the dialog's camera list.
///
/// Ported from ui/UndistortSequenceDialog.cpp, the constructor (`hasUndistortion() &&
/// getUndistortionObject()->isComputed()`).
#[must_use]
pub fn undistort_cameras(doc: &Document) -> Vec<usize> {
    (0..doc.project().cameras.len())
        .filter(|&c| doc.camera(c).model.grid.is_some())
        .collect()
}

/// Whether `a` comes before `b` in the dialog's order.
///
/// Ported from ui/UndistortSequenceDialog.cpp, `compareNames2` (identical to
/// core/HelperFunctions.cpp, `littleHelper::compareNames`): skip the common start (ignoring
/// case); at the first difference, if both characters are digits compare the numbers that
/// contain that position (with the digits before it), else a digit there comes last, else
/// the names compare as strings (UTF-16, case-sensitive); when one name ends there, the
/// shorter comes first. Numbers are compared at any length (the C++ `toInt` gives 0 past
/// 2147483647; ledger U100, fixed in both modes as for the trial dialogs).
#[must_use]
pub fn compare_names2(s1: &str, s2: &str) -> bool {
    names_less(&SortName::new(s1), &SortName::new(s2), 0)
}

/// A name prepared for [`compare_names2`] once, for sorting many names: its characters and
/// their lower case (`QChar::toLower`, the first character of Rust's lower case).
pub(crate) struct SortName<'a> {
    text: &'a str,
    chars: Vec<char>,
    lower: Vec<char>,
}

impl<'a> SortName<'a> {
    pub(crate) fn new(text: &'a str) -> SortName<'a> {
        let chars: Vec<char> = text.chars().collect();
        let lower = chars
            .iter()
            .map(|&c| c.to_lowercase().next().unwrap_or(c))
            .collect();
        SortName { text, chars, lower }
    }

    /// The number of leading characters `self` and `other` have in common, exactly.
    pub(crate) fn common_prefix(&self, other: &SortName<'_>) -> usize {
        self.chars
            .iter()
            .zip(&other.chars)
            .take_while(|(a, b)| a == b)
            .count()
    }
}

/// [`compare_names2`] of two prepared names. `from` is a number of leading characters the
/// two names have in common exactly (at most [`SortName::common_prefix`]; 0 is always
/// right): the skip of the common start begins there, which the comparison would pass
/// anyway.
pub(crate) fn names_less(s1: &SortName<'_>, s2: &SortName<'_>, from: usize) -> bool {
    let (a, b) = (&s1.chars, &s2.chars);
    let num = |c: char| c.is_numeric();
    let mut i = from;
    while i < a.len() && i < b.len() && s1.lower[i] == s2.lower[i] {
        i += 1;
    }
    i += 1;
    if i < a.len() && i < b.len() {
        let k = i - 1;
        if !num(a[k]) || !num(b[k]) {
            if num(a[k]) {
                return false;
            }
            if num(b[k]) {
                return true;
            }
            return s1.text.encode_utf16().lt(s2.text.encode_utf16());
        }
        let mut j = k;
        while j > 0 && num(a[j - 1]) {
            j -= 1;
        }
        let run = |v: &[char]| -> usize { v[k..].iter().take_while(|c| num(**c)).count() };
        // The number of `a` is its digits from `j`; that of `b` takes `a`'s digits before
        // `k` (`n`) and its own from `k`.
        number_less(&a[j..k], &a[k..k + run(a)], &b[k..k + run(b)])
    } else {
        a.len() < b.len()
    }
}

/// `toInt() <` on the digit strings `n + ra` and `n + rb`, at any length (non-ASCII digits
/// read as 0, as `toInt`).
fn number_less(n: &[char], ra: &[char], rb: &[char]) -> bool {
    fn value<'x>(n: &'x [char], run: &'x [char]) -> impl Iterator<Item = &'x char> + Clone {
        let digits = n.iter().chain(run).all(char::is_ascii_digit);
        n.iter()
            .chain(run)
            .take(if digits { usize::MAX } else { 0 })
            .skip_while(|c| **c == '0')
    }
    let (a, b) = (value(n, ra), value(n, rb));
    a.clone()
        .count()
        .cmp(&b.clone().count())
        .then_with(|| a.cmp(b))
        .is_lt()
}

/// A stable merge sort with a "comes before" test (the C++ `std::sort` with `compareNames2`
/// is not a strict order for every set of names; names it does not separate keep the order
/// of the plain sort before it, ledger U100).
fn stable_sort_by_less(v: &mut Vec<String>) {
    if v.len() < 2 {
        return;
    }
    let mut right = v.split_off(v.len() / 2);
    let mut left = std::mem::take(v);
    stable_sort_by_less(&mut left);
    stable_sort_by_less(&mut right);
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        if compare_names2(&right[j], &left[i]) {
            v.push(std::mem::take(&mut right[j]));
            j += 1;
        } else {
            v.push(std::mem::take(&mut left[i]));
            i += 1;
        }
    }
    v.extend(left.drain(i..));
    v.extend(right.drain(j..));
}

/// The images of a folder in the order they are numbered: files (not symbolic links) with an
/// extension of [`FOLDER_IMAGE_TYPES`] (any case; also `tiff` unless `xmalab3`, ledger V23),
/// as `<absolute folder>/<name>`, sorted with [`compare_names2`]. A folder that cannot be
/// read gives none.
///
/// Ported from ui/UndistortSequenceDialog.cpp, on_pushButtonUndist_clicked (`QDir::entryList`
/// with `QDir::Files | QDir::NoSymLinks`, then `std::sort(..., compareNames2)`).
#[must_use]
pub fn folder_images(folder: &Path, xmalab3: bool) -> Vec<PathBuf> {
    let dir = std::path::absolute(folder).unwrap_or_else(|_| folder.to_path_buf());
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let dir_text = dir.to_string_lossy().trim_end_matches('/').to_owned();
    let mut names: Vec<String> = entries
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let ext = Path::new(&name)
                .extension()
                .map(|x| x.to_string_lossy().to_lowercase())?;
            let ok = FOLDER_IMAGE_TYPES.contains(&ext.as_str()) || (!xmalab3 && ext == "tiff");
            ok.then(|| format!("{dir_text}/{name}"))
        })
        .collect();
    names.sort_by_key(|n| n.to_lowercase());
    stable_sort_by_less(&mut names);
    names.into_iter().map(PathBuf::from).collect()
}

/// The longest start of the first name that every name contains (`QString::contains`).
///
/// Ported from core/VideoStream.cpp, VideoStream::getFileBasename.
#[must_use]
pub fn file_basename(names: &[String]) -> String {
    let Some(first) = names.first() else {
        return String::new();
    };
    let chars: Vec<char> = first.chars().collect();
    let mut count = 0;
    while count < chars.len() {
        let prefix: String = chars[..=count].iter().collect();
        if names.iter().any(|n| !n.contains(&prefix)) {
            break;
        }
        count += 1;
    }
    chars[..count].iter().collect()
}

/// `QFileInfo(path).completeBaseName()`: the file name (after the last `/`) up to its last
/// dot.
#[must_use]
pub fn complete_base_name(path: &str) -> String {
    // Either separator, on every system (review 8, finding 11).
    xma_io::export::complete_base_name(path)
}

/// `<output>/<base>_UND.<frame + 1, four digits at least>.tif`.
///
/// Ported from ui/UndistortSequenceDialog.cpp, on_pushButtonUndist_clicked
/// (`QString("%1").arg(i + 1, 4, 10, QChar('0'))`).
#[must_use]
pub fn output_path(output: &Path, base: &str, frame: usize) -> PathBuf {
    output.join(format!("{base}_UND.{:04}.tif", frame + 1))
}

/// An image as it is remapped and written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UndistortPixels {
    /// 8-bit, `channels` (1 or 3: R, G, B) interleaved.
    U8 { data: Vec<u8>, channels: usize },
    /// 16-bit grey with `bit_depth` bits carrying information.
    U16 { data: Vec<u16>, bit_depth: u8 },
}

/// One frame to undistort or undistorted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndistortImage {
    pub width: usize,
    pub height: usize,
    pub pixels: UndistortPixels,
}

impl UndistortImage {
    /// A grey frame as read: kept at its depth (`full_depth`) or reduced to 8 bits as XMALab
    /// holds it (`Frame::to_gray8`).
    #[must_use]
    pub fn from_frame(frame: &Frame, full_depth: bool) -> UndistortImage {
        let pixels = match &frame.pixels {
            Pixels::Gray16(d) if full_depth => UndistortPixels::U16 {
                data: d.clone(),
                bit_depth: frame.bit_depth,
            },
            _ => UndistortPixels::U8 {
                data: frame.to_gray8().into_owned(),
                channels: 1,
            },
        };
        UndistortImage {
            width: frame.width as usize,
            height: frame.height as usize,
            pixels,
        }
    }

    /// Mirrors left to right (`cv::flip(image, image, 1)`).
    pub(crate) fn flip(&mut self) {
        let w = self.width;
        match &mut self.pixels {
            UndistortPixels::U8 { data, channels } => {
                let cn = *channels;
                for row in data.chunks_exact_mut(w * cn) {
                    for x in 0..w / 2 {
                        for k in 0..cn {
                            row.swap(x * cn + k, (w - 1 - x) * cn + k);
                        }
                    }
                }
            }
            UndistortPixels::U16 { data, .. } => {
                data.chunks_exact_mut(w).for_each(<[u16]>::reverse);
            }
        }
    }

    /// `cv::remap(image, image, map_x, map_y, INTER_LANCZOS4, BORDER_CONSTANT, 0)`.
    pub(crate) fn remap(&self, map_x: &[f32], map_y: &[f32], w: usize, h: usize) -> UndistortImage {
        let pixels = match &self.pixels {
            UndistortPixels::U8 { data, channels } => UndistortPixels::U8 {
                data: remap_lanczos4_u8(
                    data,
                    self.width,
                    self.height,
                    *channels,
                    map_x,
                    map_y,
                    w,
                    h,
                ),
                channels: *channels,
            },
            UndistortPixels::U16 { data, bit_depth } => {
                let mut out =
                    remap_lanczos4_u16(data, self.width, self.height, 1, map_x, map_y, w, h);
                // Lanczos overshoots at edges: saturate at the depth's largest value, as the
                // 8-bit path saturates at 255 (ledger V20).
                let max = u16::try_from((1u32 << *bit_depth) - 1).unwrap_or(u16::MAX);
                out.iter_mut().for_each(|v| *v = (*v).min(max));
                UndistortPixels::U16 {
                    data: out,
                    bit_depth: *bit_depth,
                }
            }
        };
        UndistortImage {
            width: w,
            height: h,
            pixels,
        }
    }

    /// The TIFF file's bytes ([`encode_tiff`]); a grey 8-bit image is written with three
    /// equal channels when `rgb`.
    ///
    /// # Errors
    /// A failed encoding.
    pub fn encode(&self, rgb: bool) -> Result<Vec<u8>, String> {
        let (w, h) = (self.width as u32, self.height as u32);
        let r = match &self.pixels {
            UndistortPixels::U8 { data, channels: 1 } if rgb => {
                let rgb: Vec<u8> = data.iter().flat_map(|&v| [v, v, v]).collect();
                encode_tiff(w, h, TiffSamples::Rgb8(&rgb))
            }
            UndistortPixels::U8 { data, channels: 1 } => {
                encode_tiff(w, h, TiffSamples::Gray8(data))
            }
            UndistortPixels::U8 { data, .. } => encode_tiff(w, h, TiffSamples::Rgb8(data)),
            UndistortPixels::U16 { data, bit_depth } => encode_tiff(
                w,
                h,
                TiffSamples::Gray16 {
                    data,
                    bit_depth: *bit_depth,
                },
            ),
        };
        r.map_err(|e| e.to_string())
    }
}

/// What a camera's frames are remapped with.
#[derive(Debug, Clone)]
pub struct CameraMaps {
    /// The grid's lookup table (`undistortionMapX/Y`); its size is the size frames must have.
    pub grid: Arc<LookupTable>,
    /// The lens maps (`Camera::undistortionMapX/Y`, `initUndistortRectifyMap` at the camera's
    /// size) when the camera has a lens model: x, y, width, height.
    pub lens: Option<(Vec<f32>, Vec<f32>, usize, usize)>,
    /// Mirror the frames before and after (a flipped camera in the default mode, V21).
    pub flip: bool,
}

impl CameraMaps {
    /// The undistorted image, or `None` when its size differs from the grid's (skipped, as
    /// XMALab skips it).
    ///
    /// Ported from core/UndistortionObject.cpp, UndistortionObject::undistort(Image*, QString,
    /// bool) (without the `filter` it is never called with here).
    #[must_use]
    pub fn undistort(&self, image: &UndistortImage) -> Option<UndistortImage> {
        let g = &self.grid;
        if image.width != g.width || image.height != g.height {
            return None;
        }
        let mut src = image.clone();
        if self.flip {
            src.flip();
        }
        let mut out = src.remap(&g.map_x, &g.map_y, g.width, g.height);
        if let Some((mx, my, w, h)) = &self.lens {
            out = out.remap(mx, my, *w, *h);
        }
        if self.flip {
            out.flip();
        }
        Some(out)
    }
}

/// What a camera's maps are built from, taken from the document so that the work can run
/// on another thread.
#[derive(Debug, Clone)]
struct CameraSource {
    grid: Option<GridModel>,
    lookup: Option<Arc<LookupTable>>,
    width: usize,
    height: usize,
    lens: Option<([[f64; 3]; 3], [f64; 8])>,
    flipped: bool,
}

/// Why a frame was not written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// Its size differs from the grid's (XMALab skips it silently).
    Size { width: usize, height: usize },
    /// It could not be read.
    Unreadable(String),
    /// Its file could not be written.
    NotWritten(String),
}

impl std::fmt::Display for SkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkipReason::Size { width, height } => {
                write!(f, "size {width} x {height} differs from the camera's")
            }
            SkipReason::Unreadable(e) => write!(f, "could not be read: {e}"),
            SkipReason::NotWritten(e) => write!(f, "could not be written: {e}"),
        }
    }
}

/// A frame that was not written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skipped {
    /// 0-based frame number within its item.
    pub frame: usize,
    /// The image file for a folder item.
    pub file: Option<PathBuf>,
    pub reason: SkipReason,
}

/// What happened to one item.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ItemReport {
    /// Frames of the item (files of a folder, the video's frame count).
    pub frames: usize,
    /// Files written, in frame order.
    pub written: Vec<PathBuf>,
    /// Frames not written.
    pub skipped: Vec<Skipped>,
    /// Why the item could not be processed at all.
    pub error: Option<String>,
    /// Every frame was attempted (false when cancelled or failed).
    pub completed: bool,
}

/// What a run did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UndistortReport {
    /// One per item processed or started, in queue order.
    pub items: Vec<ItemReport>,
    /// The run was cancelled before the end.
    pub cancelled: bool,
}

impl UndistortReport {
    /// Files written in all.
    #[must_use]
    pub fn written(&self) -> usize {
        self.items.iter().map(|i| i.written.len()).sum()
    }

    /// Frames skipped in all.
    #[must_use]
    pub fn skipped(&self) -> usize {
        self.items.iter().map(|i| i.skipped.len()).sum()
    }

    /// One line per item with an error and per skipped frame, for a message.
    #[must_use]
    pub fn problems(&self, items: &[UndistortItem]) -> Vec<String> {
        let mut out = Vec::new();
        for (r, it) in self.items.iter().zip(items) {
            if let Some(e) = &r.error {
                out.push(format!("{}: {e}", it.input.path().display()));
            }
            for s in &r.skipped {
                let what = match &s.file {
                    Some(f) => f.display().to_string(),
                    None => format!("{} frame {}", it.input.path().display(), s.frame + 1),
                };
                out.push(format!("{what}: {}", s.reason));
            }
        }
        out
    }
}

/// Where a run is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UndistortProgress {
    /// 0-based item.
    pub item: usize,
    pub items: usize,
    /// Frames of this item done.
    pub frame: usize,
    pub frames: usize,
}

/// The work of "Undistort": the queue with what it needs from the document, to run on any
/// thread ([`UndistortJob::run`]).
#[derive(Debug, Clone)]
pub struct UndistortJob {
    items: Vec<UndistortItem>,
    cameras: Vec<CameraSource>,
    xmalab3: bool,
    /// Ledger F70: the base names written as given (XMALab 3 mode).
    names_as_given: bool,
    video: VideoOptions,
}

impl UndistortJob {
    /// The queue `items` for `doc`, in the document's mode, with the default video readers.
    ///
    /// # Errors
    /// An item's camera does not exist or has no computed grid.
    pub fn new(doc: &Document, items: Vec<UndistortItem>) -> Result<UndistortJob, String> {
        let xmalab3 = doc.compat() == CppCompat::XMALAB3;
        let video = VideoOptions {
            compat: if xmalab3 {
                VideoCompat::XMALAB3
            } else {
                VideoCompat::FIXED
            },
            ..VideoOptions::default()
        };
        let ok = undistort_cameras(doc);
        for it in &items {
            if !ok.contains(&it.camera) {
                return Err(format!(
                    "Camera {} has no computed undistortion grid",
                    it.camera + 1
                ));
            }
        }
        let cameras = (0..doc.project().cameras.len())
            .map(|c| {
                let cam = &doc.project().cameras[c];
                let derived = doc.camera(c);
                let lens = cam.has_model_distortion.then(|| {
                    // XMALab 3 mode, ledger A124: the matrix the C++ built its lens maps with.
                    let k = match (&derived.lens_points_camera_matrix, xmalab3) {
                        (Some(m), true) => *m,
                        _ => cam.camera_matrix,
                    };
                    (
                        std::array::from_fn(|r| std::array::from_fn(|col| k[(r, col)])),
                        cam.distortion,
                    )
                });
                CameraSource {
                    grid: derived.model.grid.clone(),
                    lookup: derived.lookup.clone(),
                    width: cam.width as usize,
                    height: cam.height as usize,
                    lens,
                    flipped: cam.flipped && !xmalab3,
                }
            })
            .collect();
        let names_as_given = doc.compat().export_names_as_given;
        Ok(UndistortJob {
            names_as_given,
            items,
            cameras,
            xmalab3,
            video,
        })
    }

    /// Replaces the video readers' options (the ffmpeg executables); their `compat` is kept.
    #[must_use]
    pub fn with_video_options(mut self, options: VideoOptions) -> UndistortJob {
        self.video = VideoOptions {
            compat: self.video.compat,
            ..options
        };
        self
    }

    /// A base name as one safe part of the output names (ledger F70); empty stays empty.
    fn safe_base(&self, base: String) -> Result<String, String> {
        if base.is_empty() {
            return Ok(base);
        }
        xma_io::export::safe_component(&base, self.names_as_given)
    }

    /// The queue.
    #[must_use]
    pub fn items(&self) -> &[UndistortItem] {
        &self.items
    }

    /// The maps of camera `camera` (0-based) as this job applies them; `None` without a grid.
    #[must_use]
    pub fn camera_maps(&self, camera: usize) -> Option<CameraMaps> {
        let c = self.cameras.get(camera)?;
        let grid = match (&c.lookup, &c.grid) {
            (Some(l), _) => Arc::clone(l),
            (None, Some(g)) => Arc::new(create_lookup_table(&g.distort, c.width, c.height)),
            (None, None) => return None,
        };
        let lens = c.lens.map(|(k, d)| {
            let (mx, my) = lens_undistortion_maps(&k, &d, c.width, c.height);
            (mx, my, c.width, c.height)
        });
        Some(CameraMaps {
            grid,
            lens,
            flip: c.flipped,
        })
    }

    /// Undistorts every item in order, calling `progress` before each frame and after the
    /// last, and stopping before the next frame once `cancel` returns true (every file
    /// written is complete).
    ///
    /// Ported from ui/UndistortSequenceDialog.cpp, on_pushButtonUndist_clicked.
    pub fn run(
        &self,
        progress: &mut dyn FnMut(UndistortProgress),
        cancel: &dyn Fn() -> bool,
    ) -> UndistortReport {
        // Encoding and writing run on a second thread, one frame behind the remap.
        let (tx, rx) = std::sync::mpsc::sync_channel::<WriteJob>(2);
        std::thread::scope(|scope| {
            let writer = scope.spawn(move || {
                rx.into_iter()
                    .map(|j| {
                        let r = j
                            .image
                            .encode(j.rgb)
                            .and_then(|b| write_whole(&j.target, &b));
                        (j.item, j.frame, j.file, r.map(|()| j.target))
                    })
                    .collect::<Vec<_>>()
            });
            let mut report = self.run_items(&tx, progress, cancel);
            drop(tx);
            for (item, frame, file, r) in writer.join().unwrap_or_default() {
                let ir = &mut report.items[item];
                match r {
                    Ok(path) => ir.written.push(path),
                    Err(e) => ir.skipped.push(Skipped {
                        frame,
                        file,
                        reason: SkipReason::NotWritten(e),
                    }),
                }
            }
            for ir in &mut report.items {
                ir.skipped.sort_by_key(|s| s.frame);
            }
            report
        })
    }

    fn run_items(
        &self,
        tx: &SyncSender<WriteJob>,
        progress: &mut dyn FnMut(UndistortProgress),
        cancel: &dyn Fn() -> bool,
    ) -> UndistortReport {
        let mut report = UndistortReport::default();
        let mut maps: Vec<Option<CameraMaps>> = vec![None; self.cameras.len()];
        let n = self.items.len();
        for (index, item) in self.items.iter().enumerate() {
            if cancel() {
                report.cancelled = true;
                break;
            }
            progress(UndistortProgress {
                item: index,
                items: n,
                frame: 0,
                frames: 0,
            });
            if maps[item.camera].is_none() {
                maps[item.camera] = self.camera_maps(item.camera);
            }
            let Some(m) = maps[item.camera].as_ref() else {
                report.items.push(ItemReport {
                    error: Some("the camera has no computed undistortion grid".into()),
                    ..ItemReport::default()
                });
                continue;
            };
            let mut ir = ItemReport::default();
            let cx = Ctx { index, n, tx };
            let stopped = match &item.input {
                UndistortInput::Folder(dir) => {
                    self.run_folder(dir, item, m, &mut ir, &cx, progress, cancel)
                }
                UndistortInput::Video(file) => {
                    self.run_video(file, item, m, &mut ir, &cx, progress, cancel)
                }
            };
            report.items.push(ir);
            if stopped {
                report.cancelled = true;
                break;
            }
        }
        report
    }

    /// Undistorts one frame and queues it for writing; records it skipped when its size
    /// differs.
    #[allow(clippy::too_many_arguments)]
    fn write_frame(
        &self,
        maps: &CameraMaps,
        image: &UndistortImage,
        rgb: bool,
        target: PathBuf,
        frame: usize,
        file: Option<PathBuf>,
        ir: &mut ItemReport,
        cx: &Ctx<'_>,
    ) {
        let Some(out) = maps.undistort(image) else {
            ir.skipped.push(Skipped {
                frame,
                file,
                reason: SkipReason::Size {
                    width: image.width,
                    height: image.height,
                },
            });
            return;
        };
        // The receiver lives until every item is done.
        let _ = cx.tx.send(WriteJob {
            item: cx.index,
            frame,
            file,
            target,
            image: out,
            rgb,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn run_folder(
        &self,
        dir: &Path,
        item: &UndistortItem,
        maps: &CameraMaps,
        ir: &mut ItemReport,
        cx: &Ctx<'_>,
        progress: &mut dyn FnMut(UndistortProgress),
        cancel: &dyn Fn() -> bool,
    ) -> bool {
        let files = folder_images(dir, self.xmalab3);
        if files.is_empty() {
            // XMALab: `ImageSequence` reads `filenames.at(0)` of an empty list and crashes.
            ir.error = Some("the folder holds no images".into());
            return false;
        }
        let names: Vec<String> = files
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let base = match self.safe_base(complete_base_name(&file_basename(&names))) {
            Ok(b) => b,
            Err(e) => {
                ir.error = Some(e);
                return false;
            }
        };
        ir.frames = files.len();
        let outputs: Vec<PathBuf> = (0..files.len())
            .map(|i| output_path(&item.output, &base, i))
            .collect();
        if let Some(e) = replaced_inputs(&files, &outputs) {
            ir.error = Some(e);
            return false;
        }
        for (i, f) in files.iter().enumerate() {
            if cancel() {
                return true;
            }
            progress(UndistortProgress {
                item: cx.index,
                items: cx.n,
                frame: i,
                frames: files.len(),
            });
            let target = output_path(&item.output, &base, i);
            match self.read_still(f) {
                Ok(img) => self.write_frame(maps, &img, false, target, i, Some(f.clone()), ir, cx),
                // XMALab: `imread` gives an empty image, whose size differs from the map.
                Err(e) => ir.skipped.push(Skipped {
                    frame: i,
                    file: Some(f.clone()),
                    reason: SkipReason::Unreadable(e),
                }),
            }
        }
        progress(UndistortProgress {
            item: cx.index,
            items: cx.n,
            frame: files.len(),
            frames: files.len(),
        });
        ir.completed = true;
        false
    }

    /// A still as `core/Image.cpp` holds it: colour in colour (8 bits), grey at its depth in
    /// the default mode, 8 bits in the XMALab 3 mode.
    fn read_still(&self, path: &Path) -> Result<UndistortImage, String> {
        read_still_image(path, &self.video.compat, !self.xmalab3)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_video(
        &self,
        file: &Path,
        item: &UndistortItem,
        maps: &CameraMaps,
        ir: &mut ItemReport,
        cx: &Ctx<'_>,
        progress: &mut dyn FnMut(UndistortProgress),
        cancel: &dyn Fn() -> bool,
    ) -> bool {
        let ext = file
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if self.xmalab3 && !VIDEO_TYPES.contains(&ext.as_str()) {
            ir.error = Some("not a .cine, .avi, .mp4 or .mov file".into());
            return false;
        }
        let mut src = match open_video(file, &self.video) {
            Ok(s) => s,
            Err(e) => {
                ir.error = Some(e.to_string());
                return false;
            }
        };
        // `CineVideo` holds grey frames, `AviVideo` OpenCV's B, G, R frames.
        let rgb = src.info().kind != SourceKind::Cine;
        let frames = src.info().frame_count;
        let base = match self.safe_base(complete_base_name(&file.to_string_lossy())) {
            Ok(b) => b,
            Err(e) => {
                ir.error = Some(e);
                return false;
            }
        };
        ir.frames = frames;
        let outputs: Vec<PathBuf> = (0..frames)
            .map(|i| output_path(&item.output, &base, i))
            .collect();
        if let Some(e) = replaced_inputs(&[file.to_path_buf()], &outputs) {
            ir.error = Some(e);
            return false;
        }
        let mut last: Option<UndistortImage> = None;
        for i in 0..frames {
            if cancel() {
                return true;
            }
            progress(UndistortProgress {
                item: cx.index,
                items: cx.n,
                frame: i,
                frames,
            });
            let target = output_path(&item.output, &base, i);
            match src.read_frame(i) {
                Ok(f) => {
                    let img = UndistortImage::from_frame(&f, !self.xmalab3);
                    self.write_frame(maps, &img, rgb, target, i, None, ir, cx);
                    last = Some(img);
                }
                Err(e) => match (&last, self.xmalab3) {
                    // XMALab 3 (ledger V22): the reader keeps the previous image.
                    (Some(prev), true) => {
                        self.write_frame(maps, prev, rgb, target, i, None, ir, cx)
                    }
                    _ => ir.skipped.push(Skipped {
                        frame: i,
                        file: None,
                        reason: SkipReason::Unreadable(e.to_string()),
                    }),
                },
            }
        }
        progress(UndistortProgress {
            item: cx.index,
            items: cx.n,
            frame: frames,
            frames,
        });
        ir.completed = true;
        false
    }
}

/// A still as `core/Image.cpp` holds it: colour in colour (8 bits, R, G, B), grey at its
/// depth when `full_depth`, else reduced to 8 bits.
pub(crate) fn read_still_image(
    path: &Path,
    compat: &VideoCompat,
    full_depth: bool,
) -> Result<UndistortImage, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if let Some(c) = decode_colour_image(&bytes, path).map_err(|e| e.to_string())? {
        let rgb: Vec<u8> = c.rgba.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        return Ok(UndistortImage {
            width: c.width as usize,
            height: c.height as usize,
            pixels: UndistortPixels::U8 {
                data: rgb,
                channels: 3,
            },
        });
    }
    let frame = decode_image_with(&bytes, path, compat).map_err(|e| e.to_string())?;
    Ok(UndistortImage::from_frame(&frame, full_depth))
}

/// An undistorted frame on its way to the writer thread.
struct WriteJob {
    item: usize,
    frame: usize,
    file: Option<PathBuf>,
    target: PathBuf,
    image: UndistortImage,
    rgb: bool,
}

/// Where the frames of one item go.
struct Ctx<'a> {
    index: usize,
    n: usize,
    tx: &'a SyncSender<WriteJob>,
}

/// Ledger V25: the sentence refusing an item when any of its output names (`outputs`) is
/// one of its own inputs (`inputs`), `None` when none is. XMALab reads each input only when
/// it reaches it, so an output that replaces an input not yet read is read back in its
/// place (the frame undistorted twice) and the original is lost; here, in both modes, such
/// an item is refused before anything of it is written. Paths are compared by their folder
/// as the file system resolves it (symbolic links, `.` and `..`) and their file name
/// (ignoring case on Windows and macOS, whose file systems usually do).
#[must_use]
pub fn replaced_inputs(inputs: &[PathBuf], outputs: &[PathBuf]) -> Option<String> {
    let key = |p: &Path| -> Option<(PathBuf, String)> {
        let dir = std::fs::canonicalize(p.parent()?).ok()?;
        let name = p.file_name()?.to_string_lossy().into_owned();
        let name = if cfg!(any(windows, target_os = "macos")) {
            name.to_lowercase()
        } else {
            name
        };
        Some((dir, name))
    };
    let out: std::collections::HashSet<(PathBuf, String)> =
        outputs.iter().filter_map(|p| key(p)).collect();
    if out.is_empty() {
        return None;
    }
    let hit: Vec<&PathBuf> = inputs
        .iter()
        .filter(|p| key(p).is_some_and(|k| out.contains(&k)))
        .collect();
    let first = hit.first()?;
    let name = |p: &Path| {
        p.file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
    };
    Some(if hit.len() == 1 {
        format!(
            "not undistorted: its output would replace its own input {}; choose another \
             output folder",
            name(first)
        )
    } else {
        format!(
            "not undistorted: its outputs would replace {} of its own inputs ({} to {}); \
             choose another output folder",
            hit.len(),
            name(first),
            name(hit[hit.len() - 1])
        )
    })
}

/// Writes `bytes` to `path` under a temporary name in the same folder, then renames it, so
/// that `path` is either the old file or the whole new one.
pub(crate) fn write_whole(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let name = path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let tmp = path.with_file_name(format!(".{name}.part"));
    // A failed write (a full disk) removes the partial file, as a failed rename does.
    std::fs::write(&tmp, bytes).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_names_and_output_names() {
        let names: Vec<String> = ["/d/img_0001.tif", "/d/img_0002.tif", "/d/img_0010.tif"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        assert_eq!(file_basename(&names), "/d/img_00");
        assert_eq!(complete_base_name(&file_basename(&names)), "img_00");
        assert_eq!(complete_base_name("/d/trial.1."), "trial.1");
        assert_eq!(complete_base_name("/d/a/"), "");
        assert_eq!(complete_base_name("/v/Camera1.avi"), "Camera1");
        assert_eq!(
            output_path(Path::new("/o"), "Camera1", 0),
            PathBuf::from("/o/Camera1_UND.0001.tif")
        );
        assert_eq!(
            output_path(Path::new("/o"), "x", 12344),
            PathBuf::from("/o/x_UND.12345.tif")
        );
        assert_eq!(file_basename(&["/d/a.avi".to_owned()]), "/d/a.avi");
    }

    #[test]
    fn natural_order() {
        assert!(compare_names2("/d/img2.tif", "/d/img10.tif"));
        assert!(!compare_names2("/d/img10.tif", "/d/img2.tif"));
        assert!(compare_names2(
            "/d/f99999999999.tif",
            "/d/f100000000000.tif"
        ));
        let mut v: Vec<String> = ["/d/b10", "/d/b9", "/d/a1", "/d/B2"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        v.sort_by_key(|n| n.to_lowercase());
        stable_sort_by_less(&mut v);
        // Different letters compare case-sensitively (`QString::compare`): "B" before "a".
        assert_eq!(v, ["/d/B2", "/d/a1", "/d/b9", "/d/b10"]);
    }

    /// Review 12 (cross-cutting 7): a write that fails leaves no `.part` file behind.
    /// The temporary name is made a link to `/dev/full`, where every write fails.
    #[cfg(target_os = "linux")]
    #[test]
    fn review12_failed_write_removes_the_part_file() {
        let dir = std::env::temp_dir().join(format!("xma-review12-part-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("Camera1_UND.0001.tif");
        let tmp = dir.join(".Camera1_UND.0001.tif.part");
        std::os::unix::fs::symlink("/dev/full", &tmp).unwrap();
        let err = write_whole(&target, b"pixels").unwrap_err();
        assert!(err.contains("Camera1_UND.0001.tif"), "{err}");
        assert!(
            std::fs::symlink_metadata(&tmp).is_err(),
            "the partial file is left behind"
        );
        assert!(!target.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
