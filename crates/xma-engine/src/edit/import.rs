//! Imports that change the project: 2D points (E66), markers from a CSV file (E67, E68),
//! markers and bodies from another project file (E69) or another trial (E70), marker names
//! (E71), and trials from another project file (E72); `docs/cpp-editing.md` section 10.
//!
//! Every file is read before anything changes, and an edit that fails is taken back, so a
//! malformed file leaves the project as it was. What is recomputed follows the mode
//! ([`super::body_ops`]).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use xma_io::cparse::{lines, qstring, qt_to_double, qt_to_int, split_commas};
use xma_model::{MarkerStatus, Point2, SETTING_NAMES, SettingsPatch, Trial};

use super::body_ops::{
    add_marker, add_point_idx, add_rigid_body, capped_penalty, check_trial, clear_point_idx,
    copy_data, cpp_recompute, keeping_bodies_current, outcome, read_file, read_reference_lines,
    reference_entry, remove_marker, remove_rigid_body, reset_references,
    set_reference_marker_references, trial_mut,
};
use super::{EditError, EditOutcome, EditResult, Import2dOptions};
use crate::document::{Document, OpenOptions, read_project, reset_references_at_load};
use crate::markers::{update_marker_frame, update_mean_size};

// ---------------------------------------------------------------------------------------
// E66: 2D points
// ---------------------------------------------------------------------------------------

/// One point of a 2D points file, as read: the new marker (0-based among the file's),
/// camera, row (frame) and the two fields.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FilePoint {
    pub marker: usize,
    pub camera: usize,
    pub frame: usize,
    pub x: f64,
    pub y: f64,
}

/// What a 2D points file holds.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Points2dFile {
    /// Markers the file adds: `(fields of the first non-empty line - offset) / 2 / cameras`.
    pub n_markers: usize,
    /// Names from the header (one per new marker), when the file has one.
    pub names: Option<Vec<String>>,
    /// The points that are not missing, in file order.
    pub points: Vec<FilePoint>,
    /// Data rows (frames) read.
    pub rows: usize,
    /// Cells the file lacks (a row shorter than the first line).
    pub missing_fields: usize,
    /// The first line (1-based) with a point whose coordinate is NaN or infinite (`nan`,
    /// `inf`: `QString::toDouble` reads them; "NaN" itself marks a missing point).
    pub non_finite_line: Option<usize>,
}

/// True when a field holds a point coordinate: not containing "NaN" and, unless
/// `empty_is_zero`, not empty (white space only).
fn present(field: Option<&str>, empty_is_zero: bool) -> Option<f64> {
    let f = field?;
    if f.contains("NaN") || (!empty_is_zero && f.trim().is_empty()) {
        return None;
    }
    Some(qt_to_double(f))
}

/// Reads a 2D points file as `Trial::load2dPoints` does: comma-separated, empty lines
/// skipped; the first non-empty line fixes the number of new markers; with a header its
/// field `i * 2 * cameras + offset` is marker `i`'s name, with "_cam1_X" removed and spaces
/// trimmed; every other line is one frame from 0, with x and y of marker `i` in camera `j`
/// at fields `i * 2 * cameras + 2 j + offset` and the next; a pair is missing when either
/// field contains "NaN".
///
/// Ported from core/Trial.cpp, Trial::load2dPoints (`:1917-2016`), with D17 fixed: a field
/// past the end of a short line (an out-of-range read in the C++) is missing, and, unless
/// `empty_is_zero` ([`xma_model::CppCompat::import_2d_empty_cell_zero`], ledger F30), so is an empty field (the C++
/// reads 0).
pub(crate) fn parse_points_2d(
    data: &[u8],
    n_cameras: usize,
    offset_columns: bool,
    header: bool,
    empty_is_zero: bool,
) -> Points2dFile {
    let offset = if offset_columns { n_cameras } else { 0 };
    let mut out = Points2dFile::default();
    let mut first = true;
    let mut read_header = header;
    for (index, line) in lines(data).into_iter().enumerate() {
        if line.is_empty() {
            continue;
        }
        let text = qstring(line);
        let list = split_commas(&text);
        if first {
            // C++ int arithmetic: (size - offset) / 2 / cameras, truncated towards zero.
            let fields = list.len() as i64 - offset as i64;
            out.n_markers = usize::try_from(fields / 2 / n_cameras as i64).unwrap_or(0);
            first = false;
        }
        if read_header {
            let names = (0..out.n_markers)
                .map(|i| {
                    let field = list.get(i * 2 * n_cameras + offset).copied().unwrap_or("");
                    field.replace("_cam1_X", "").trim_matches(' ').to_owned()
                })
                .collect();
            out.names = Some(names);
            read_header = false;
            continue;
        }
        for marker in 0..out.n_markers {
            for camera in 0..n_cameras {
                let k = marker * n_cameras * 2 + 2 * camera + offset;
                if k + 1 >= list.len() {
                    out.missing_fields += 1;
                    continue;
                }
                let (Some(x), Some(y)) = (
                    present(list.get(k).copied(), empty_is_zero),
                    present(list.get(k + 1).copied(), empty_is_zero),
                ) else {
                    continue;
                };
                if !(x.is_finite() && y.is_finite()) && out.non_finite_line.is_none() {
                    out.non_finite_line = Some(index + 1);
                }
                out.points.push(FilePoint {
                    marker,
                    camera,
                    frame: out.rows,
                    x,
                    y,
                });
            }
        }
        out.rows += 1;
    }
    out
}

/// `Marker::update()` on the given markers: a marker flagged for recomputation is
/// triangulated at every frame (and its flag cleared), another only reprojected.
///
/// Ported from core/Marker.cpp, Marker::update (`:981-998`). The rigid-body step of
/// `reconstruct3DPoint` is left out. The markers it is used on are new ones, which are
/// usually in no body; but a body can list a marker index beyond the trial's markers (a
/// project file can give one), and the import then creates that marker. Such a body is
/// brought up to date by the refresh after the edit (`keeping_bodies_current`) in the
/// default mode; in the XMALab 3 mode the points went through `setPoint`, whose own step
/// has recomputed it.
fn update_markers(doc: &mut Document, trial: usize, markers: &[usize]) {
    let ctx = doc.frame_context(trial);
    let t = trial_mut(doc, trial);
    for &m in markers {
        let marker = &t.markers[m];
        let flagged = marker.requires_recomputation;
        let n = marker.views.first().map_or(0, |v| v.points.len());
        let patches: Vec<_> = (0..n)
            .filter(|&f| f < marker.n_frames() && marker.views.iter().all(|v| f < v.points.len()))
            .map(|f| update_marker_frame(&ctx, marker, f))
            .collect();
        let marker = Arc::make_mut(&mut t.markers[m]);
        for p in &patches {
            p.apply(marker);
        }
        if flagged {
            marker.requires_recomputation = false;
        }
    }
}

/// One file of E66.
///
/// Ported from core/Trial.cpp, Trial::load2dPoints (`:1917-2016`): the new markers are
/// appended and named, each point is corrected (Count 1: minus 1; Y up: `height - y - 1`;
/// undistorted: distorted with `Camera::undistortPoint(pt, false)`) and written with
/// `Marker::setPoint` ([`super::points::set_point`]) with status `SET` or `TRACKED`, then
/// `Marker::update()` runs on each new marker (ledger A216 fixed: the points are written
/// without `setPoint`'s triangulation, which `Marker::update()` repeats; XMALab 3 mode,
/// [`xma_model::CppCompat::point_edits_follow_up_per_point`]: through `setPoint`). D17
/// fixed: rows beyond the trial's frames (written past the marker's arrays in the C++) are
/// not imported, with a Console line.
fn load_2d_points(
    doc: &mut Document,
    trial: usize,
    path: &Path,
    data: &[u8],
    options: Import2dOptions,
) -> Result<Vec<String>, EditError> {
    let n_cameras = doc.project.cameras.len();
    let empty_is_zero = doc.compat().import_2d_empty_cell_zero;
    let file = parse_points_2d(
        data,
        n_cameras,
        options.offset_columns,
        options.header,
        empty_is_zero,
    );
    // Ledger F52: a coordinate that is NaN or infinite is refused (XMALab 3 stores it).
    if let Some(line) = file.non_finite_line
        && !doc.compat().nonfinite_imports_accepted
    {
        return Err(EditError::Invalid(format!(
            "{}, line {line}: a point coordinate is not a finite number; nothing was imported",
            path.display()
        )));
    }
    let n_frames = doc.project.trials[trial].n_frames;
    let first = doc.project.trials[trial].markers.len();
    for _ in 0..file.n_markers {
        add_marker(doc, trial);
    }
    if let Some(names) = &file.names {
        let t = trial_mut(doc, trial);
        for (k, name) in names.iter().enumerate() {
            Arc::make_mut(&mut t.markers[first + k]).description = name.clone();
        }
    }
    let status = if options.status_set {
        MarkerStatus::Set
    } else {
        MarkerStatus::Tracked
    };
    // Ledger A216: XMALab 3 triangulates (and recomputes bodies) after every point, then
    // again in `update_markers`; fixed, the points are written and triangulated once.
    let per_point = {
        let compat = doc.compat();
        compat.point_edits_follow_up_per_point || compat.edit_first_body_only
    };
    for p in &file.points {
        if p.frame >= n_frames {
            continue;
        }
        let (mut x, mut y) = (p.x, p.y);
        if options.count_from_one {
            x -= 1.0;
            y -= 1.0;
        }
        if options.y_up {
            y = f64::from(doc.project.cameras[p.camera].height) - y - 1.0;
        }
        if !options.distorted {
            let q =
                doc.camera(p.camera)
                    .model
                    .undistort_point(Point2::new(x, y), false, true, true);
            x = q.x;
            y = q.y;
        }
        if per_point {
            super::points::set_point(
                doc,
                trial,
                first + p.marker,
                p.camera,
                p.frame,
                x,
                y,
                status,
            )?;
        } else {
            // Ledger A216: the point alone; `update_markers` below triangulates every
            // frame of the new markers (flagged by `add_marker`). A body that lists one of
            // them (an index beyond the trial's markers, as a project file can give) is
            // recomputed by the refresh after the edit (`keeping_bodies_current`).
            let marker = first + p.marker;
            super::ops::check_cell(doc, trial, marker, p.camera, p.frame)?;
            let v = &mut doc.marker_mut(trial, marker).views[p.camera];
            v.points[p.frame] = Point2::new(x, y);
            v.status[p.frame] = status;
        }
    }
    let new: Vec<usize> = (first..first + file.n_markers).collect();
    update_markers(doc, trial, &new);
    let mut messages = Vec::new();
    if file.rows > n_frames {
        messages.push(format!(
            "{}: {} rows beyond the trial's {n_frames} frames were not imported.",
            path.display(),
            file.rows - n_frames
        ));
    }
    if file.missing_fields > 0 {
        messages.push(format!(
            "{}: {} points were missing from short rows.",
            path.display(),
            file.missing_fields
        ));
    }
    Ok(messages)
}

/// E66 File > Import > 2D Points: each file in turn ([`load_2d_points`]) appends its
/// markers. `Invalid` for a project without cameras.
///
/// Ported from ui/MainWindow.cpp, MainWindow::on_actionImport2D_Points_triggered
/// (`:2110-2144`).
pub(super) fn import_2d_points(
    doc: &mut Document,
    trial: usize,
    files: &[PathBuf],
    options: Import2dOptions,
) -> EditResult {
    check_trial(doc, trial)?;
    if doc.project.cameras.is_empty() {
        return Err(EditError::Invalid("the project has no cameras".to_owned()));
    }
    let data = files
        .iter()
        .map(|p| read_file(p))
        .collect::<Result<Vec<_>, _>>()?;
    keeping_bodies_current(doc, trial, &[], |doc| {
        let mut messages = Vec::new();
        for (path, data) in files.iter().zip(&data) {
            messages.extend(load_2d_points(doc, trial, path, data, options)?);
        }
        Ok(outcome(messages))
    })
}

// ---------------------------------------------------------------------------------------
// E67, E68, E71: CSV files of markers
// ---------------------------------------------------------------------------------------

/// E67 (`update_existing` false: one new marker per point of the file) and E68 (true:
/// point `i` updates marker `i`, added when missing; then every body whose references come
/// from its markers takes them again and is recomputed and filtered): the description is
/// the point's name and the marker's reference point is set.
///
/// Ported from core/Trial.cpp, Trial::loadMarkersFromCSV (`:787-840`), the format of
/// `setReferenceFromFile`. The number of points is the number of names divided by 3; a
/// coordinates line shorter than that (an out-of-range read in the C++) is `Invalid`.
pub(super) fn import_markers_csv(
    doc: &mut Document,
    trial: usize,
    path: &Path,
    update_existing: bool,
) -> EditResult {
    check_trial(doc, trial)?;
    let data = read_file(path)?;
    let (names, coords) = read_reference_lines(&data);
    let n = names.len() / 3;
    if coords.len() < 3 * n {
        return Err(EditError::Invalid(format!(
            "{}: {} coordinates for {n} points",
            path.display(),
            coords.len()
        )));
    }
    // Ledger F52: a coordinate that is NaN or infinite is refused (XMALab 3 stores it).
    if !doc.compat().nonfinite_imports_accepted
        && let Some(i) = (0..n).find(|&i| {
            let p = reference_entry(&names, &coords, i).1;
            !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
        })
    {
        return Err(EditError::Invalid(format!(
            "{}, line 2: a coordinate of point {} is not a finite number",
            path.display(),
            i + 1
        )));
    }
    keeping_bodies_current(doc, trial, &[], |doc| {
        for i in 0..n {
            let (name, point) = reference_entry(&names, &coords, i);
            let m = if update_existing {
                if doc.project.trials[trial].markers.len() <= i {
                    add_marker(doc, trial);
                }
                i
            } else {
                add_marker(doc, trial)
            };
            let marker = Arc::make_mut(&mut trial_mut(doc, trial).markers[m]);
            marker.description = name;
            marker.reference_3d = Some(point);
        }
        if update_existing {
            for b in 0..doc.project.trials[trial].rigid_bodies.len() {
                if doc.project.trials[trial].rigid_bodies[b].references_set == 1 {
                    set_reference_marker_references(trial_mut(doc, trial), b);
                    cpp_recompute(doc, trial, b, true);
                }
            }
        }
        Ok(EditOutcome::none())
    })
}

/// E71 "Rename": each line `<number>,...,<name>` names marker `<number>` (1-based); numbers
/// out of range, and lines whose first field is not a number, are ignored.
///
/// Ported from core/Trial.cpp, Trial::renameMarkersFromCSV (`:768-785`).
pub(super) fn rename_markers_from_csv(doc: &mut Document, trial: usize, path: &Path) -> EditResult {
    check_trial(doc, trial)?;
    let data = read_file(path)?;
    let renames: Vec<(i64, String)> = lines(&data)
        .into_iter()
        .map(|line| {
            let text = qstring(line);
            let fields = split_commas(&text);
            let id = i64::from(qt_to_int(fields[0])) - 1;
            (id, fields.last().copied().unwrap_or("").to_owned())
        })
        .collect();
    let n = doc.project.trials[trial].markers.len();
    for (id, name) in renames {
        if let Ok(m) = usize::try_from(id)
            && m < n
            && doc.project.trials[trial].markers[m].description != name
        {
            Arc::make_mut(&mut trial_mut(doc, trial).markers[m]).description = name;
        }
    }
    Ok(EditOutcome::none())
}

// ---------------------------------------------------------------------------------------
// E69, E72: another project file
// ---------------------------------------------------------------------------------------

/// The settings patch that makes reading another project use this document's settings.
fn settings_patch(doc: &Document) -> SettingsPatch {
    let mut patch = SettingsPatch::default();
    for (name, _) in SETTING_NAMES {
        if let Some(v) = doc.settings().get(name) {
            patch.set(name, v);
        }
    }
    patch
}

/// Reads another project file with this document's settings and mode.
fn read_other(doc: &Document, path: &Path) -> Result<xma_io::LoadedProject, EditError> {
    let options = OpenOptions {
        base_settings: doc.settings().clone(),
        overrides: settings_patch(doc),
        compat: doc.compat(),
    };
    read_project(path, &options).map_err(|e| EditError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

/// E69 "Import from xma": the markers and bodies of trial `source_trial` of another
/// project file, applied over this trial's.
///
/// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::loadMarker (`:684-868`), core/Trial.cpp,
/// Trial::loadMarkers (`:843-859`) and Trial::loadRigidBodies (`:861-896`), in their order:
///
/// 1. marker `i` takes the description of the file's marker `i` (markers added as needed);
/// 2. body `i` takes the file's body `i`'s description and markers (`clearPointIdx`, then
///    `addPointIdx` without the second recomputation for each), bodies added as needed;
/// 3. per `<Marker>` element: the reference point (when the file's loads), tracking penalty
///    (`setMaxPenalty`), detection method, size override, threshold offset;
/// 4. per `<RigidBody>` element: references from the file (`RigidBody::load`, no
///    recomputation) or `resetReferences`; optimised references (`loadOptimized`); visible,
///    colour, cutoff override and cutoff; mesh, draw flag and scale.
///
/// Nothing is deleted: extra markers and bodies stay, and kept markers keep their 2D data
/// (D11; the interface's prompt is what needs changing). Virtual points are not imported
/// (the C++ reads no `DummyMarker` here). No camera data is read, so the file's project may
/// have another number of cameras (a marker added here gets this project's cameras).
///
/// The file is read by the project reader, so attributes it lacks take the reader's
/// defaults (the C++ leaves the marker's value), the version conversions of old files apply
/// to the penalty (the C++ caps the raw value), and the legacy `InterpolationMethod`
/// attribute (written by XMALab before 0.2, which the C++ applies to every frame) is not
/// applied. A mesh whose file does not load is not imported (the C++ also removes the
/// body's own mesh then).
pub(super) fn import_markers_from_project(
    doc: &mut Document,
    trial: usize,
    path: &Path,
    source_trial: &str,
) -> EditResult {
    check_trial(doc, trial)?;
    let loaded = read_other(doc, path)?;
    let Some(k) = loaded
        .project
        .trials
        .iter()
        .position(|t| t.name == source_trial)
    else {
        return Err(EditError::Invalid(format!(
            "{}: no trial named \"{source_trial}\"",
            path.display()
        )));
    };
    let src: Arc<Trial> = Arc::clone(&loaded.project.trials[k]);
    let without_element = loaded
        .trial_sources
        .get(k)
        .map(|s| s.markers_without_element.clone())
        .unwrap_or_default();
    let n_markers = doc.project.trials[trial]
        .markers
        .len()
        .max(src.markers.len());
    if let Some(m) = src
        .rigid_bodies
        .iter()
        .flat_map(|b| b.marker_indices.iter())
        .find(|&&m| m >= n_markers)
    {
        return Err(EditError::Invalid(format!(
            "{}: a rigid body of \"{source_trial}\" lists marker {} of {n_markers}",
            path.display(),
            m + 1
        )));
    }
    keeping_bodies_current(doc, trial, &[], |doc| {
        // Trial::loadMarkers.
        for (i, m) in src.markers.iter().enumerate() {
            if i >= doc.project.trials[trial].markers.len() {
                add_marker(doc, trial);
            }
            Arc::make_mut(&mut trial_mut(doc, trial).markers[i])
                .description
                .clone_from(&m.description);
        }
        // Trial::loadRigidBodies.
        for (i, b) in src.rigid_bodies.iter().enumerate() {
            if i >= doc.project.trials[trial].rigid_bodies.len() {
                add_rigid_body(doc, trial);
            }
            super::body_ops::body_mut(doc, trial, i)
                .description
                .clone_from(&b.description);
            clear_point_idx(doc, trial, i);
            for &m in &b.marker_indices {
                add_point_idx(doc, trial, i, m, false);
            }
        }
        // <Marker> elements.
        for (i, m) in src.markers.iter().enumerate() {
            if without_element.contains(&i) {
                continue;
            }
            let marker = Arc::make_mut(&mut trial_mut(doc, trial).markers[i]);
            if let Some(r) = m.reference_3d {
                marker.reference_3d = Some(r);
            }
            marker.max_penalty = capped_penalty(m.max_penalty);
            marker.method = m.method;
            marker.size_override = m.size_override;
            marker.threshold_offset = m.threshold_offset;
        }
        // <RigidBody> elements.
        for (i, b) in src.rigid_bodies.iter().enumerate() {
            if b.references_set == 2 {
                // RigidBody::load.
                let rb = super::body_ops::body_mut(doc, trial, i);
                rb.has_optimized_coordinates = false;
                let n = b.reference_points.len();
                rb.reference_points_original = if b.has_optimized_coordinates {
                    b.reference_points_original[..n.min(b.reference_points_original.len())].to_vec()
                } else {
                    b.reference_points.clone()
                };
                rb.reference_points = rb.reference_points_original.clone();
                rb.reference_names.clone_from(&b.reference_names);
                rb.references_set = 2;
                rb.initialised = true;
            } else {
                reset_references(doc, trial, i);
            }
            let rb = super::body_ops::body_mut(doc, trial, i);
            if b.has_optimized_coordinates {
                // RigidBody::loadOptimized.
                let n = rb
                    .reference_points
                    .len()
                    .min(rb.reference_points_original.len());
                let (orig, pts) = (&mut rb.reference_points_original, &rb.reference_points);
                orig[..n].copy_from_slice(&pts[..n]);
                rb.reference_points.clone_from(&b.reference_points);
                rb.has_optimized_coordinates = true;
            }
            rb.visible = b.visible;
            rb.color = b.color;
            rb.override_cutoff_frequency = b.override_cutoff_frequency;
            rb.cutoff_frequency = b.cutoff_frequency;
            if b.mesh_model.is_some() {
                rb.mesh_model.clone_from(&b.mesh_model);
                rb.mesh_readable = b.mesh_readable;
                rb.draw_mesh_model = b.draw_mesh_model;
                rb.mesh_scale = b.mesh_scale;
            }
        }
        Ok(EditOutcome::none())
    })
}

/// E70 "Copy From trial": this trial becomes linked to the Default trial when the source is
/// it; marker `i` takes the source marker's reference point (when set), description,
/// penalty, method, size override and threshold (markers added as needed); body `i` becomes
/// a copy of the source's (`RigidBody::copyData`, [`copy_data`]; bodies added as needed);
/// then extra bodies, then extra markers, are removed from the end. Kept markers keep their
/// 2D data. `Invalid` when the source is the trial itself (the dialog lists only others).
///
/// Ported from ui/ImportExportPointsDialog.cpp, ImportExportPointsDialog::copyFromTrial
/// (`:241-306`).
pub(super) fn copy_markers_from_trial(
    doc: &mut Document,
    trial: usize,
    source: usize,
) -> EditResult {
    check_trial(doc, trial)?;
    check_trial(doc, source)?;
    if source == trial {
        return Err(EditError::Invalid(
            "a trial cannot copy its markers from itself".to_owned(),
        ));
    }
    let src = Arc::clone(&doc.project.trials[source]);
    keeping_bodies_current(doc, trial, &[], |doc| {
        let mut messages = Vec::new();
        if doc.project.trials[trial].is_copy_from_default != src.is_default {
            trial_mut(doc, trial).is_copy_from_default = src.is_default;
        }
        for (i, m) in src.markers.iter().enumerate() {
            if i >= doc.project.trials[trial].markers.len() {
                add_marker(doc, trial);
            }
            let marker = Arc::make_mut(&mut trial_mut(doc, trial).markers[i]);
            if let Some(r) = m.reference_3d {
                marker.reference_3d = Some(r);
            }
            marker.description.clone_from(&m.description);
            marker.max_penalty = capped_penalty(m.max_penalty);
            marker.method = m.method;
            marker.size_override = m.size_override;
            marker.threshold_offset = m.threshold_offset;
        }
        for (i, b) in src.rigid_bodies.iter().enumerate() {
            if i >= doc.project.trials[trial].rigid_bodies.len() {
                add_rigid_body(doc, trial);
            }
            copy_data(doc, trial, i, b);
        }
        while doc.project.trials[trial].rigid_bodies.len() > src.rigid_bodies.len() {
            let last = doc.project.trials[trial].rigid_bodies.len() - 1;
            messages.extend(remove_rigid_body(doc, trial, last));
        }
        while doc.project.trials[trial].markers.len() > src.markers.len() {
            let last = doc.project.trials[trial].markers.len() - 1;
            remove_marker(doc, trial, last);
        }
        Ok(outcome(messages))
    })
}

/// E72 File > Import > Trial from xma-file: the named trials of another project file, in
/// the order given, each read as loading reads a trial (the after-load steps of
/// [`Document::from_loaded`] for it) and flagged for recomputation, then appended. A
/// Default trial, when the project has one, replaces it when `replace_default` is set; when
/// it is not, it is skipped and the trials after it are imported (ledger F31, D20, fixed in
/// both modes: XMALab 3 abandons them). After a Default trial is imported, it is applied to the trials linked to
/// it ("Apply to trials", E27). `Invalid` when a name is not in the file.
///
/// A file whose project has another number of cameras is `Rejected` in both modes, with a
/// sentence giving both counts (ledger F33): the C++ builds the trial with this project's
/// cameras and reads each marker's 2D rows for that many cameras, so with fewer cameras in
/// the file it reads past the rows (and finds no video list for the extra camera, whose
/// `filenameList.at(0)` is out of range), and with more it drops the other cameras' data.
///
/// Ported from ui/MainWindow.cpp, MainWindow::on_actionImportTrial_triggered
/// (`:2146-2200`), ui/ProjectFileIO.cpp, ProjectFileIO::loadTrials (`:388`), and
/// core/Project.cpp, Project::addTrial and Project::replaceTrial (`:264-281`). Checking the
/// image paths, binding textures and re-entering the workspace (which updates the active
/// trial) are the interface's.
pub(super) fn import_trials(
    doc: &mut Document,
    path: &Path,
    trials: &[String],
    replace_default: bool,
) -> EditResult {
    let loaded = read_other(doc, path)?;
    let (theirs, ours) = (loaded.project.cameras.len(), doc.project.cameras.len());
    if theirs != ours {
        return Err(EditError::Rejected(format!(
            "The trials of {} were recorded with {theirs} camera{}, but this project has {ours} camera{}. A trial can only be imported from a project with the same number of cameras.",
            path.display(),
            if theirs == 1 { "" } else { "s" },
            if ours == 1 { "" } else { "s" },
        )));
    }
    let mut picked = Vec::new();
    for name in trials {
        let Some(k) = loaded.project.trials.iter().position(|t| &t.name == name) else {
            return Err(EditError::Invalid(format!(
                "{}: no trial named \"{name}\"",
                path.display()
            )));
        };
        picked.push(k);
    }
    let compat = doc.compat();
    let mut messages = Vec::new();
    for k in picked {
        let mut trial: Trial = loaded.project.trials[k].as_ref().clone();
        // The after-load steps of Document::from_loaded, for this trial.
        for m in &mut trial.markers {
            update_mean_size(Arc::make_mut(m), compat.size_range_from_sum);
        }
        for b in &mut trial.rigid_bodies {
            let initialised = b.references_set != 0;
            if b.initialised != initialised {
                Arc::make_mut(b).initialised = initialised;
            }
        }
        if compat.poses_at_load {
            let mut tmp = doc.project.clone();
            tmp.trials = vec![Arc::new(trial)];
            reset_references_at_load(&mut tmp, compat);
            trial = tmp.trials[0].as_ref().clone();
        }
        let source = loaded.trial_sources.get(k).cloned();
        let existing_default = doc.project.trials.iter().position(|t| t.is_default);
        let index = match (trial.is_default, existing_default) {
            (true, Some(d)) => {
                if !replace_default {
                    messages.push(format!(
                        "The Default trial of {} was not imported.",
                        path.display()
                    ));
                    continue;
                }
                doc.project.trials[d] = Arc::new(trial);
                let n = doc.project.trials.len();
                let extras = doc.extras_mut();
                if let Some(s) = source
                    && extras.trial_sources.len() == n
                {
                    extras.trial_sources[d] = s;
                }
                d
            }
            _ => {
                let n = doc.project.trials.len();
                doc.project.trials.push(Arc::new(trial));
                let extras = doc.extras_mut();
                if let Some(s) = source
                    && extras.trial_sources.len() == n
                {
                    extras.trial_sources.push(s);
                }
                n
            }
        };
        doc.set_trial_requires_recomputation(index, true);
        if doc.project.trials[index].is_default {
            messages.extend(super::trials::apply_default_to_trials(doc)?.messages);
        }
    }
    Ok(outcome(messages))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_2d_are_read_as_the_cpp_with_d17_fixed() {
        // Two cameras, offset columns, a header, an empty line, a short row, an empty cell.
        let data =
            b"c1,c2, a_cam1_X ,a_cam1_Y,a_cam2_X,a_cam2_Y,b_cam1_X,b_cam1_Y,b_cam2_X,b_cam2_Y\n\
                     \n\
                     0,0,1,2,3,4,5,6,7,8\r\n\
                     0,0,NaN,2,3,4,,6,7,8\n\
                     0,0,1,2\n";
        let f = parse_points_2d(data, 2, true, true, false);
        assert_eq!(f.n_markers, 2);
        assert_eq!(f.names, Some(vec!["a".to_owned(), "b".to_owned()]));
        assert_eq!(f.rows, 3);
        assert_eq!(f.missing_fields, 3);
        let at = |m, c, fr| {
            f.points
                .iter()
                .find(|p| (p.marker, p.camera, p.frame) == (m, c, fr))
        };
        assert_eq!(at(0, 0, 0).map(|p| (p.x, p.y)), Some((1.0, 2.0)));
        assert_eq!(at(1, 1, 0).map(|p| (p.x, p.y)), Some((7.0, 8.0)));
        assert!(at(0, 0, 1).is_none(), "NaN");
        assert!(at(1, 0, 1).is_none(), "empty cell");
        assert_eq!(at(0, 0, 2).map(|p| (p.x, p.y)), Some((1.0, 2.0)));
        assert!(at(0, 1, 2).is_none(), "short row");
        // XMALab 3 reads the empty cell as 0.
        let f = parse_points_2d(data, 2, true, true, true);
        assert_eq!(
            f.points
                .iter()
                .find(|p| (p.marker, p.camera, p.frame) == (1, 0, 1))
                .map(|p| (p.x, p.y)),
            Some((0.0, 6.0))
        );
    }

    #[test]
    fn a_first_line_shorter_than_the_offset_adds_no_marker() {
        let f = parse_points_2d(b"1\n1,2,3\n", 2, true, false, false);
        assert_eq!(f.n_markers, 0);
        assert!(f.points.is_empty());
        assert_eq!(f.rows, 2);
    }
}
