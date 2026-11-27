//! A calibration image (or the undistortion grid's image) undistorted as the window shows
//! it: the image as XMALab holds it (8 bits, colour when the file is, mirrored for a
//! flipped camera), remapped with the undistortion grid's lookup table and then, for a
//! calibration image, with the lens model's maps (Lanczos, border 0).
//!
//! Moved here from the window (`xma-app/src/frames.rs`, `derive`) so that the Blender-scene
//! export (ledger F85) writes the image the window shows; the window calls
//! [`undistort_held_image`]. This was display code (the project's rule: code that only affects
//! what is shown is rechecked before an export uses it): the remaps are
//! `xma_vision::remap::remap_lanczos4_u8` and `lens_undistortion_maps`, compared with
//! OpenCV as `oracle/py/DISPLAY_CHECKS.md` records; the Blender-scene tests check the
//! geometry the export relies on (the camera matrix of the undistorted image).
//!
//! Rechecked for the export: `crates/xma-engine/tests/blender_scene.rs`,
//! `blender_scene_calibration_image_is_the_undistort_sequence_image`, compares
//! [`undistorted_calibration_image`] with File › Undistort sequence's remap
//! (`undistort_sequence::CameraMaps::undistort`, the export-grade code checked against
//! XMALab's own exports; its result mirrored once more for a flipped camera, ledger V21) for
//! every calibration image of every camera with a grid of the pig, the lens reference project
//! (`x03_lens`, a lens model) and the flipped one (`x02_flipped`): the pixels are the same
//! (largest difference 0). Not covered by that comparison: a camera with a lens model and no
//! grid (the undistort sequence needs a grid); its remap is the same function with the same
//! maps.
//!
//! Ported from `core/UndistortionObject.cpp` (`undistort`, `computeTmpImage`) and
//! `core/CalibrationSequence.cpp` (`undistort`: the grid remap, then the lens-model remap
//! when the camera has one).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use xma_model::Camera;
use xma_video::open::OpenOptions;
use xma_vision::remap::{lens_undistortion_maps, remap_lanczos4_u8};
use xma_vision::undistortion::{LookupTable, LwmTransform, create_lookup_table};

use crate::Document;

/// An 8-bit image, rows top to bottom; `channels` 1 (grey) or 3 (R, G, B).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image8 {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<u8>,
}

/// A camera's lens model as the remap takes it: its camera matrix and distortion, when it
/// has one.
#[must_use]
pub fn camera_lens(cam: &Camera) -> Option<([[f64; 3]; 3], [f64; 8])> {
    cam.has_model_distortion.then(|| {
        let k = &cam.camera_matrix;
        (
            std::array::from_fn(|r| std::array::from_fn(|c| k[(r, c)])),
            cam.distortion,
        )
    })
}

/// The image as XMALab holds it (`px`, `sw` by `sh`, `cn` channels, as read from its
/// file), mirrored when `flip`, remapped with `lut` when given, then with the lens maps
/// of `lens` at the camera's size `w` by `h` when given.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn undistort_held_image(
    mut px: Vec<u8>,
    sw: usize,
    sh: usize,
    cn: usize,
    flip: bool,
    lut: Option<&LookupTable>,
    lens: Option<&([[f64; 3]; 3], [f64; 8])>,
    w: usize,
    h: usize,
) -> Image8 {
    if flip {
        let row = (sw * cn).max(1);
        for r in px.chunks_mut(row) {
            let n = r.len() / cn.max(1);
            for i in 0..n / 2 {
                for k in 0..cn {
                    r.swap(i * cn + k, (n - 1 - i) * cn + k);
                }
            }
        }
    }
    let (mut pw, mut ph) = (sw, sh);
    if let Some(lut) = lut {
        px = remap_lanczos4_u8(
            &px, pw, ph, cn, &lut.map_x, &lut.map_y, lut.width, lut.height,
        );
        pw = lut.width;
        ph = lut.height;
    }
    if let Some((k, d)) = lens {
        let (mx, my) = lens_undistortion_maps(k, d, w, h);
        px = remap_lanczos4_u8(&px, pw, ph, cn, &mx, &my, w, h);
        pw = w;
        ph = h;
    }
    Image8 {
        width: pw,
        height: ph,
        channels: cn,
        data: px,
    }
}

/// Where a camera's lookup table comes from: the one a computation left, or the grid to
/// make it from.
#[derive(Clone, Debug)]
enum Lookup {
    Table(Arc<LookupTable>),
    Grid(LwmTransform),
}

/// What [`CalibrationImageSource::undistort`] needs, taken from the document so that the
/// work can run on another thread.
#[derive(Clone, Debug)]
pub struct CalibrationImageSource {
    camera: usize,
    image: usize,
    source: Source,
    flip: bool,
    lookup: Option<Lookup>,
    lens: Option<([[f64; 3]; 3], [f64; 8])>,
    width: usize,
    height: usize,
}

#[derive(Clone, Debug)]
enum Source {
    Still { data: Arc<Vec<u8>>, path: PathBuf },
    Sequence { path: PathBuf },
}

impl CalibrationImageSource {
    /// Calibration image `image` (0-based) of `camera`.
    ///
    /// # Errors
    /// A sentence: no such camera or image, or a still image that is neither in the
    /// project nor readable.
    pub fn new(doc: &Document, camera: usize, image: usize) -> Result<Self, String> {
        let cam = doc
            .project()
            .cameras
            .get(camera)
            .ok_or_else(|| format!("no camera {}", camera + 1))?;
        let source = if let Some(seq) = &cam.calibration_sequence {
            Source::Sequence {
                path: crate::resolve_stored_path(doc.path().and_then(Path::parent), &seq.filename),
            }
        } else {
            let ci = cam.calibration_images.get(image).ok_or_else(|| {
                format!("camera {}: no calibration image {}", camera + 1, image + 1)
            })?;
            let (data, path) = crate::still_bytes(doc, camera, &ci.filename)?;
            Source::Still {
                data: Arc::new(data),
                path,
            }
        };
        let d = doc.camera(camera);
        let lookup = match (&d.lookup, &d.model.grid) {
            (Some(l), _) => Some(Lookup::Table(Arc::clone(l))),
            (None, Some(g)) => Some(Lookup::Grid(g.distort.clone())),
            (None, None) => None,
        };
        Ok(CalibrationImageSource {
            camera,
            image,
            source,
            flip: cam.flipped,
            lookup,
            lens: camera_lens(cam),
            width: cam.width as usize,
            height: cam.height as usize,
        })
    }

    /// The image is undistorted (the camera has a computed grid or a lens model).
    #[must_use]
    pub fn undistorts(&self) -> bool {
        self.lookup.is_some() || self.lens.is_some()
    }

    /// The image, undistorted as the window shows it (`Derived::UndistortedCalibration`):
    /// with the computed grid and the lens model where the camera has them, else as held
    /// (mirrored for a flipped camera).
    ///
    /// # Errors
    /// The image cannot be read, or its size is not the grid's.
    pub fn undistort(&self) -> Result<Image8, String> {
        let (camera, image) = (self.camera, self.image);
        let (px, sw, sh, cn) = match &self.source {
            Source::Sequence { path } => {
                let mut src = xma_video::open::open_source(
                    std::slice::from_ref(path),
                    &OpenOptions::default(),
                )
                .map_err(|e| format!("{}: {e}", path.display()))?;
                let f = src
                    .read_frame(image)
                    .map_err(|e| format!("{} frame {}: {e}", path.display(), image + 1))?;
                (
                    f.to_gray8().into_owned(),
                    f.width as usize,
                    f.height as usize,
                    1,
                )
            }
            Source::Still { data, path } => {
                let colour = xma_video::image_file::decode_colour_image(data, path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                match colour {
                    Some(c) => (
                        c.rgba.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect(),
                        c.width as usize,
                        c.height as usize,
                        3,
                    ),
                    None => {
                        let g = xma_video::image_file::decode_image(data, path)
                            .map_err(|e| format!("{}: {e}", path.display()))?;
                        (
                            g.to_gray8().into_owned(),
                            g.width as usize,
                            g.height as usize,
                            1,
                        )
                    }
                }
            }
        };
        let lut = match &self.lookup {
            None => None,
            Some(Lookup::Table(t)) => Some(Arc::clone(t)),
            Some(Lookup::Grid(g)) => {
                Some(Arc::new(create_lookup_table(g, self.width, self.height)))
            }
        };
        if let Some(l) = &lut
            && (l.width != sw || l.height != sh)
        {
            return Err(format!(
                "camera {}: calibration image {} is {sw} x {sh}, the undistortion grid {} x {}",
                camera + 1,
                image + 1,
                l.width,
                l.height
            ));
        }
        Ok(undistort_held_image(
            px,
            sw,
            sh,
            cn,
            self.flip,
            lut.as_deref(),
            self.lens.as_ref(),
            self.width,
            self.height,
        ))
    }
}

/// Calibration image `image` (0-based) of `camera`, undistorted as the window shows it
/// ([`CalibrationImageSource::undistort`]); the second value says whether it was
/// undistorted.
///
/// # Errors
/// A sentence: no such camera or image, or the image cannot be read.
pub fn undistorted_calibration_image(
    doc: &Document,
    camera: usize,
    image: usize,
) -> Result<(Image8, bool), String> {
    let s = CalibrationImageSource::new(doc, camera, image)?;
    Ok((s.undistort()?, s.undistorts()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_mirrors_each_row_keeping_channels() {
        let px = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let out = undistort_held_image(px, 2, 2, 3, true, None, None, 2, 2);
        assert_eq!(out.data, vec![4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9]);
        let out = undistort_held_image(vec![1, 2, 3, 4, 5, 6], 3, 2, 1, true, None, None, 3, 2);
        assert_eq!(out.data, vec![3, 2, 1, 6, 5, 4]);
    }
}
