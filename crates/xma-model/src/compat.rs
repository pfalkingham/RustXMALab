//! Compatibility with known faults of XMALab 3.
//!
//! Known faults of the C++ program are fixed, not copied (the author, 14 November
//! 2026). The Rust program fixes them by default; [`CppCompat::XMALAB3`] gives the C++
//! behaviour back, so the port can still be checked against the reference dumps of the C++
//! program (`xma parity` always uses it). Each field names its parity-ledger item.
//!
//! The mode is a property of a session (how a project is opened, computed, saved and
//! exported), not of a project: it is never stored in a project file.
//!
//! The switches of merge-aware tracking (ledger A84, A85, A96, A97, A215) went with that
//! tracking kind on 26 November 2026 (parity ledger A226).

/// Which known faults of XMALab 3 to reproduce. Each field is `true` to behave as the C++
/// does, `false` for the fixed behaviour. [`Default`] fixes everything.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[allow(clippy::struct_excessive_bools)]
pub struct CppCompat {
    /// Ledger A2: in the rigid-body pose refinement each virtual point is added once
    /// per camera for every calibrated camera (cameras x calibrated cameras
    /// observations). Fixed: one observation per calibrated camera.
    pub virtual_points_per_camera_pair: bool,
    /// Ledger A3: in the pose refinement, a virtual point whose world position is
    /// unknown at a frame (its other body has no pose, or it has no coordinates there)
    /// is added at the origin, with its reference position at the origin too. Fixed:
    /// left out, as `computePose` leaves it out.
    pub virtual_points_at_origin: bool,
    /// Ledger A4: the refinement's residual applies the lens distortion coefficients
    /// (k1, k2, p1, p2, k3) to the projection, although its targets are fully
    /// undistorted points. Fixed: plain pinhole projection.
    pub distortion_in_refinement: bool,
    /// Ledger A7: the low-pass filter is designed at the entered cutoff although it is
    /// applied twice (forwards and backwards), so the data are attenuated to 0.5 there.
    /// Fixed: the single pass is designed at a corrected cutoff so that the filter as
    /// applied has amplitude 1/sqrt(2) at the entered cutoff
    /// (`xma_vision::filter::design_cutoff`).
    pub filter_cutoff_single_pass: bool,
    /// Ledger A11: distorting a point applies the grid transform first and the lens
    /// model second, the order used to undistort. Fixed: lens model first, then the
    /// grid, the true inverse.
    pub grid_first_when_distorting: bool,
    /// Ledger A19: removing lens distortion runs 5 iterations of `cv::undistortPoints`.
    /// Fixed: iterates until converged (`xma_vision::camera::TermCriteria::CONVERGED`).
    pub five_iterations: bool,
    /// Ledger A20: a rigid-body error standard deviation with one contributing value is
    /// NaN. Fixed: 0, as for markers.
    pub error_sd_nan: bool,
    /// Ledger A23: the inputs of the rigid-body fit, the refit in Filter3DPoints mode,
    /// the virtual points of the refinement, `projectToImage` and the EPnP inputs pass
    /// through single precision. Fixed: double throughout.
    pub single_precision: bool,
    /// Ledger A24: a rigid body's frames tracked (precision report) counts every frame.
    /// Fixed: the frames whose pose was computed.
    pub frames_tracked_all: bool,
    /// Ledger A25: a marker's reprojection error (precision report) is averaged over
    /// every camera, also where the marker is not set. Fixed: only the cameras in which
    /// it is set.
    pub reprojection_error_all_cameras: bool,
    /// Ledger A26: the EPnP fallback uses the last camera with at least 4 of the body's
    /// markers, and a failed EPnP leaves a zero pose marked as computed. Fixed: the
    /// camera with the most markers (the first on a tie), and no pose when EPnP fails.
    pub epnp_last_camera: bool,
    /// Ledger A27: in the pose from 2D rays, a ray with no candidate point empties the
    /// list of combinations so that later indices no longer match their rays. Fixed:
    /// that marker is left out for the frame.
    pub rays_misaligned: bool,
    /// Ledger A28: the refinement solver's refusal or failure is ignored (a refusal
    /// replaces the pose by its own double inversion). Fixed: the pose is left exactly
    /// as it was.
    pub refine_failure_ignored: bool,
    /// Ledger A29: the refinement uses marker observations from uncalibrated cameras.
    /// Fixed: calibrated cameras only.
    pub uncalibrated_cameras_in_refinement: bool,
    /// Ledger A30: the refinement's pinhole projection ignores the skew. Fixed: the
    /// skew term is included.
    pub refinement_without_skew: bool,
    /// Ledger A34: triangulation uses uncalibrated cameras, and a degenerate result
    /// keeps the old 3D point (methods 0 and 1) or writes infinities or NaN (methods 2
    /// to 4). Fixed: only calibrated cameras, and a degenerate result gives an
    /// undefined point.
    pub triangulation_unchecked: bool,
    /// Ledger A35: a projection is written for every camera even when the 3D point is
    /// undefined. Fixed: none is written for an undefined point.
    pub project_undefined_points: bool,
    /// Ledger A54: a rigid body whose references are reset while loading computes and
    /// filters its poses at that moment, before the undistortion is fitted, and frames
    /// the trial update leaves without a pose keep them. Fixed: no poses are computed
    /// while loading; the trial update computes them.
    pub poses_at_load: bool,
    /// Ledger A58: a rigid body's marker-to-marker standard deviation divides the squared
    /// deviations, summed over the marker pairs about each pair's own mean, by the total
    /// number of distances minus 1, and the trial pools the bodies on the same basis.
    /// Fixed: the pooled standard deviation, divided by the sum over pairs of (n − 1).
    pub pooled_sd_total_count: bool,
    /// Ledger A1: the size range of a marker is the largest distance of a size from the *sum* of
    /// its sizes, and unset sizes (-1) are included, so `Marker::isValid`'s size test never
    /// rejects a point. Fixed: the largest distance from the mean size, over set sizes only.
    pub size_range_from_sum: bool,
    /// Ledger A14: detection methods 1, 3 and 6 measure a feature's distance from `searchArea + 1`,
    /// one pixel off the centre of the search patch. Fixed: from the patch centre.
    pub search_centre_plus_one: bool,
    /// Ledger A15: the detection jobs the interface starts hold a marker's threshold offset in an
    /// unsigned 8-bit value, so a negative offset wraps (-3 becomes 253). Fixed: the offset is
    /// a signed integer in every path.
    pub threshold_offset_wraps: bool,
    /// Ledger A31: after a marker point is written, only the first rigid body that lists the
    /// marker has its pose at that frame recomputed. Fixed: every body that lists it, and
    /// every body with a virtual point driven by one of those.
    pub edit_first_body_only: bool,
    /// Ledger A60: after an edit, rotations are not made continuous and filtered poses, filtered
    /// errors and the other derived values are left as they were until an explicit update.
    /// Fixed: brought up to date when the edit's step is committed (`Document::settle`).
    pub edits_not_refiltered: bool,
    /// Ledger A91: the trial update's finalize makes one body's rotations continuous and
    /// filters it before going on to the next body, so a body filtered from its virtual
    /// points (`Filter3DPoints`) reads an earlier driving body's rotations after the
    /// continuity step and a later one's before it (the same rotation, different in the last
    /// bits where the step replaced it). Fixed: every body's rotations are made continuous
    /// first, then every body is filtered.
    pub finalize_body_by_body: bool,
    /// Ledger A33: cubic interpolation of 2D points returns from the whole of
    /// `Marker::interpolate` at the first camera with fewer than 4 known frames, with no gap
    /// to fill, or whose spline fails (core/Marker.cpp:1371-1372, :1395-1401), so later
    /// cameras are not interpolated. Fixed: that camera is skipped.
    pub cubic_interpolation_stops: bool,
    /// Ledger A76 (`docs/cpp-editing.md` D18): "Set Body from File" on a body clears its
    /// references and replaces them by the mean shape before reading the file, and a file
    /// that is then rejected leaves the body so (and, for a rejected fit, `initialised`
    /// set). Fixed: a rejected file changes nothing.
    pub rejected_reference_file_keeps_reset: bool,
    /// Ledger A93: "Set Body from File" on a body whose reference points are all at one
    /// place (never set, no frame with every marker in 3D) fits every order of the file's
    /// points and keeps the one rounding favours. Fixed: the file's order, without a search.
    /// (Beyond the search's budget, from 10 markers, both modes take the file's order.)
    pub exhaustive_degenerate_reference_search: bool,
    /// Ledger A77 (`docs/cpp-editing.md` D5): `RigidBody::removePointIdx` removes the
    /// reference point, original point and name by value, every element equal to the one at
    /// the marker's position. Fixed: the entries at that position.
    pub remove_point_by_value: bool,
    /// Ledger A78 (D6): removing a rigid body leaves other bodies' virtual points with their
    /// body indices. Fixed: a point driven by the removed body is deleted, higher indices
    /// are lowered.
    pub virtual_points_keep_body_index: bool,
    /// Ledger A83: "Add virtual point" takes the driving body from the reference file's
    /// fourth value when there is one (as `saveDummy` writes it; -1 for a point animated by
    /// coordinates), whatever the dialog chose. Fixed: the dialog's choice decides and a
    /// fourth value that disagrees is ignored, with a Console line. (Where the C++ then
    /// crashes the edit is refused in both modes.)
    pub virtual_point_file_chooses_body: bool,
    /// Ledger A79: `RigidBody::clearPointIdx` clears the reference points and names but not
    /// the original reference points, which keep their old entries in front of the new
    /// ones. Fixed: cleared with the others.
    pub clear_point_idx_keeps_original: bool,
    /// Ledger A61: `Marker::interpolate` first resets every point whose status is at or below
    /// `INTERPOLATED`, which includes the negative marks `UNTRACKABLE`, `DELETED` and `LOST`.
    /// Fixed: only `PREDICTED` and `INTERPOLATED` points are reset; the marks are kept.
    pub interpolate_clears_marks: bool,
    /// Ledger A62: after a tracking step, detection (and for "selected" not validation) runs on
    /// every point whose status at the target frame is `TRACKED`, for every marker, whether or
    /// not this step tracked it. Fixed: detection and validation cover exactly the points this
    /// step tracked.
    pub redetect_all_tracked: bool,
    /// Ledger U79 (decided by the author, 16 November 2026): moving markers into a rigid body
    /// (Points panel drag and drop, "Add selected to Rigid Body") also removes and re-adds a
    /// marker already in that body, which resets the body's references and moves the marker
    /// to the end of its list. Fixed: a marker already in the target body is left out of the
    /// move; a move with nothing left changes nothing.
    pub move_to_own_body_readds: bool,
    /// Ledger A64: the tracker's distance penalty is centred half a pixel right of and below
    /// the predicted position rounded to a whole pixel (at `0.5 * size` of an odd-sized
    /// surface), whatever the prediction's fraction. Fixed: centred on the predicted position
    /// itself, sub-pixel fraction included.
    pub penalty_off_centre: bool,
    /// Ledger A65: search-window offsets are rounded with `(int)(v + 0.5)`, which rounds towards
    /// zero for negative values, so a window reaching past the left or top image edge is one
    /// pixel off. Fixed: `floor(v + 0.5)`.
    pub window_round_toward_zero: bool,
    /// Ledger A66: the polynomial point refinement computes the ellipse rotation with a
    /// mis-bracketed expression, which changes the refined radius. Fixed: the intended
    /// expression.
    pub polyfit_rotation_bracket: bool,
    /// Ledger A90: the polynomial point refinement returns a refined point and radius when its
    /// last cycle's quadratic part is not a peak (`J <= 0`) or is a minimum of the fitted
    /// quantity (a dip where the marker's polarity needs a maximum), and when the last
    /// cycle's radius is `+inf` or 0; the radius is then stored as the marker size. Fixed: no
    /// result in those cases. The same switch sets the refinement's window bound: fixed, no
    /// result once a cycle's window half-width exceeds the image's larger side; XMALab 3,
    /// once it exceeds the larger of that side and 1024, so that the C++'s results for
    /// windows larger than the image are reproduced.
    pub refine_accepts_non_peak: bool,
    /// Ledger A67: exchanging tracked data between two markers swaps points and statuses but not
    /// the detected sizes, so each marker keeps the other's sizes. Fixed: sizes are swapped
    /// with the points.
    pub swap_points_only: bool,
    /// Ledger A68: deleting a frame range resets the y of the projected point but not its x.
    /// Fixed: both.
    pub reset_keeps_projected_x: bool,
    /// Ledger A73: detection methods 1, 3 and 6 add the search patch's integer offset to the
    /// feature's `float` position in single precision, which rounds the image position (about
    /// 6e-5 pixels at x = 1000). Fixed: the sum is exact.
    pub offset_sum_f32: bool,
    /// Ledger F4: project XML attributes such as the recording speed, cutoff
    /// frequencies, mesh scale, grid centre, square size and image scale and bias are
    /// written with 6 significant digits. Fixed: `%.17g`, so a save loses nothing.
    pub six_digit_attributes: bool,
    /// Ledger F11: the filtered rigid-body export checks the trial's cutoff even for a
    /// body that overrides it. Fixed: each body's own cutoff.
    pub filtered_export_trial_cutoff: bool,
    /// Ledger F13: the tracking penalty of a project older than version 0.3 is
    /// converted after it has been capped, which gives 0. Fixed: converted first, then
    /// capped.
    pub old_penalty_zero: bool,
    /// Ledger F2: the 2D points export's offset header fields are `cam<j>_offset`,
    /// numbered from 0 and written with no separators. Fixed: `cam<j+1>_offset,` per
    /// camera, so the header has as many fields as each data row.
    pub offset_header: bool,
    /// Ledger F30 (D17): File > Import > 2D Points reads an empty cell as 0. Fixed: an
    /// empty cell is a missing value, as `NaN` is.
    pub import_2d_empty_cell_zero: bool,
    /// Ledger F18: a rigid body's mesh whose OBJ file cannot be read is dropped when
    /// the project is read (path, draw flag and scale), so the next save loses it.
    /// Fixed: kept, with a record that the file could not be read.
    pub drop_unreadable_mesh: bool,
    /// Ledger F25: export details: a zero translation is written `-0`; an empty
    /// description is used as it is in names; events with empty or equal names share a
    /// file; with no markers or bodies lines have no end. Fixed: `0`; the default name
    /// (`marker%03d`, `RigidBody%03d`); distinct file names; proper line ends.
    pub export_details: bool,
    /// Ledger F34 (CD8): File > New dataset with a camera whose images do not share one
    /// size: XMALab 3 shows "<camera> : Resolutions do not match" and still opens the
    /// project, with the cameras before that one, without it and without a calibration
    /// object. Fixed: nothing is created and the sentence is the error.
    pub partial_project_on_resolution_mismatch: bool,
    /// Ledger F35 (CD9): File > New dataset compares calibration videos by number of files,
    /// so videos of different lengths are accepted (and the project's number of
    /// calibration frames is the last camera's). Fixed: refused with the dialog's
    /// "has wrong number of images" sentence, counting frames.
    pub calibration_video_lengths_unchecked: bool,
    /// Ledger F36 (CD10): a new trial (or Change Trial Data) takes its frame count from the
    /// first camera and never compares the others. Fixed: videos of different lengths are
    /// refused with a sentence giving both counts.
    pub trial_video_lengths_unchecked: bool,
    /// Ledger F37 (CD11): Change Trial Data keeps the markers', bodies' and events'
    /// per-frame arrays at their old length and flags nothing for recomputation. Fixed:
    /// the arrays take the new length and the trial is recomputed. (A longer trial gets
    /// longer arrays in both modes: the C++ reads past them, undefined behaviour.)
    pub change_trial_data_keeps_arrays: bool,
    /// Ledger F37: Change Trial Data renames the trial before it checks the videos' size,
    /// so a refused change still renames it. Fixed: a refused change changes nothing.
    pub change_trial_data_renames_on_refusal: bool,
    /// Ledger A128: File > New dataset (external calibration) adds each camera before
    /// giving it its image, so the project's number of calibration frames is 0 until it
    /// is saved and opened again. Fixed: 1, as after reopening.
    pub external_calibration_frame_count_zero: bool,
    /// Ledger A5: the cube calibration's random search draws from one C library generator
    /// for the whole session (seeded 1 when a project is opened or created), so a
    /// calibration's result depends on every calibration run before it. Fixed: each cube
    /// calibration starts from `srand(1)`, so its result depends only on its inputs.
    pub calibration_rand_continues: bool,
    /// Ledger A104: in the grid matching one neighbour of the centre bead can serve two of
    /// the six target directions (`GridMatchCompat::six_neighbours_any_target`).
    pub grid_neighbour_for_two_targets: bool,
    /// Ledger A105: grid lattice sites are compared by their accumulated `double`
    /// coordinates (`GridMatchCompat::lattice_site_by_coordinates`).
    pub grid_sites_compared_by_coordinates: bool,
    /// Ledger A106: the grid's reference rescaling divides by zero when one row or column
    /// was matched (`GridMatchCompat::unchecked_reference_scale`).
    pub grid_reference_scale_unchecked: bool,
    /// Ledger A107: "remove outlier at border" sets every inlier flag from the circle test
    /// (`OutlierCompat::circle_sets_inliers`).
    pub outlier_circle_sets_inliers: bool,
    /// Ledger A108: its circle radius `r - 1.5 t` is squared, so a negative radius keeps
    /// points (`OutlierCompat::circle_radius_signed`).
    pub outlier_circle_radius_signed: bool,
    /// Ledger A110: a reference click counts when its undistorted position is positive
    /// (`CubeCompat::reference_from_undistorted_position`).
    pub cube_reference_from_undistorted_position: bool,
    /// Ledger A111: the single-image calibration counts as done whatever its last refit
    /// achieved (`CubeCompat::success_after_failed_refit`).
    pub cube_success_after_failed_refit: bool,
    /// Ledger A112: `FixPrincipal` fixes the principal point at `0.5 (width + 1)`,
    /// `0.5 (height + 1)`, in the single-image calibration (`CubeCompat::principal_point_one_based`)
    /// and in the calibration over all images (`processing/Calibration.cpp:267-268`).
    /// Fixed: `0.5 (width - 1)`, `0.5 (height - 1)`.
    pub principal_point_one_based: bool,
    /// Ledger A130: Alt + click in the Calibration workspace rounds the detected position to
    /// whole pixels and always detects in the distorted image, also when the view shows the
    /// undistorted one. Fixed: the detected position as found, in the view's coordinates.
    pub calibration_snap_rounded: bool,
    /// Ledger A131: "Delete Frame" leaves every trial's reference calibration image index as
    /// it was, so trials then use another frame or one past the list. Fixed: indices after
    /// the deleted frame move down by one, and one past the end moves to the last frame.
    pub delete_frame_keeps_trial_references: bool,
    /// Ledger A132: when the calibration over all images fails (OpenCV refuses it), nothing
    /// is written, no message is given, and the camera's pending recalibration is cleared,
    /// so the camera is shown as up to date. Fixed: the camera stays pending and the
    /// failure is reported.
    pub failed_recalibration_cleared: bool,
    /// Ledger A133: "Reset Camera" and the recalibration of the pending-changes question
    /// change the calibration without flagging the trials for recomputation. Fixed: every
    /// trial is flagged.
    pub recalibration_trials_not_flagged: bool,
    /// Ledger A134: unchecking "is optimized" runs the main button's action on the active
    /// calibration image, which for an image set only by the optimisation starts a
    /// reference calibration (or refuses it) instead of recalibrating. Fixed: every camera
    /// is recalibrated over all its images.
    pub remove_optimization_on_active_image: bool,
    /// Ledger A135: "compute calibration" refused for too few beads ("Not enough points
    /// found to run calibration") keeps the beads it detected. Fixed: a refused calibration
    /// changes nothing.
    pub refused_calibration_keeps_beads: bool,
    // ----- Checkerboard calibration (package p6checker) ----------------------------------
    /// Ledger A180: after a checkerboard detection `CalibrationImage::sortGridByReference`
    /// stores the detected corners, in the distorted image, as the undistorted detected
    /// points too (and as both projected lists) and leaves the error lists as they were,
    /// so the undistorted points shown and the undistorted errors are wrong until the next
    /// undistortion or reopening. Fixed: the undistorted points are the corners undistorted.
    pub checkerboard_undistorted_points_raw: bool,
    /// Ledger A181: the corners `cv::findChessboardCorners` returns are stored when there
    /// are as many as the board has, also when it reports the board not found (its last
    /// checks, for example a corner within 8 pixels of the border, reject the board but keep
    /// the corners), unrefined. Fixed: stored only when the board is found.
    pub checkerboard_unconfirmed_corners_stored: bool,
    /// Ledger A182 (survey CD25): `runCalibrationCameraAllFrames` counts the frames with
    /// detected corners on the active camera instead of the camera being calibrated.
    /// Fixed: on that camera.
    pub checkerboard_frames_counted_on_active_camera: bool,
    /// Ledger A183 (survey CD25): the manual corner placement maps the board's outer corners
    /// without the inverted axis (`CheckerboadInvertedAxis`) onto the four clicks, then
    /// applies that homography to the board's points, which carry it, so with an inverted
    /// axis every predicted corner lies outside the board. Fixed: the board's own outer
    /// corners.
    pub checkerboard_homography_ignores_inverted_axis: bool,
    /// Ledger A184: in manual corner mode "compute calibration" needs one click only and
    /// uses the unset references (-1, -1) as clicked corners. Fixed: all four corners must
    /// be clicked.
    pub checkerboard_manual_with_unset_clicks: bool,
    /// Ledger A188: with a planar object (a checkerboard) the calibration over all frames and
    /// `reproject` use every frame with a detected-point list, also one with no inlier at all
    /// (the unset points "Reset Frame" or manual mode gives a frame), so OpenCV refuses the
    /// calibration (an empty view) from then on. Fixed: a frame needs at least one inlier.
    pub planar_frames_without_points_used: bool,
    /// Ledger A202: the manual corners accept any four clicks (three coincident, a crossed
    /// or flat quadrilateral) and store the corners their homography predicts, also outside
    /// the image or not finite. Fixed: the four clicks must span a proper quadrilateral and
    /// the predicted corners must be finite and inside the image.
    pub checkerboard_degenerate_clicks_accepted: bool,
    /// Ledger A203: a detection that gives no board leaves the frame's earlier corners, which
    /// `runCalibration` then sorts by the new click and calibrates, as a success. Fixed: the
    /// failed detection is reported and the frame left as it was.
    pub checkerboard_failed_detection_reuses_corners: bool,
    /// Ledger A204: the recalibration of the pending-changes question (and every other path
    /// but `runCalibrationCameraAllFrames`) calibrates a checkerboard camera from a single
    /// frame. Fixed: two frames with corners are needed, as on that path.
    pub checkerboard_single_frame_calibrated: bool,
    /// Ledger A189: each camera numbers the board from the corner clicked in it, so two
    /// cameras can number the same board from different corners and their poses of it lie
    /// in different world frames; "optimize camera setup" then runs on them. Fixed: after a
    /// board camera is calibrated its frames are renumbered, frame by frame, to agree with
    /// the reference camera's, and the optimisation refuses cameras that still disagree.
    pub checkerboard_numbering_per_camera: bool,
    // ----- Multi-camera optimisation and external calibration (package engopt) -----------
    /// Ledger A121 (`xma_vision::multicam::MulticamCompat::chain_skips_frame_zero`): the
    /// chain that places cameras tests `f_to_use > 0`, so frame 0 never links two cameras.
    pub multicam_chain_skips_frame_zero: bool,
    /// Ledger A122 (`MulticamCompat::chain_uses_reference_image_pose`): the chain reads the
    /// placed camera's pose from its own image at the reference frame.
    pub multicam_chain_uses_reference_image_pose: bool,
    /// Ledger A123 (b) (`MulticamCompat::jacobian_nan_at_zero_difference`): the norm
    /// Jacobians are NaN where a coordinate difference is exactly 0.
    pub multicam_jacobian_nan_at_zero_difference: bool,
    /// Ledger A124 (`MulticamCompat::lens_maps_with_previous_camera_matrix`): method 2
    /// applies the new coefficients with the old camera matrix (lens maps, undistorted
    /// detected points of the calibration images).
    pub multicam_lens_maps_with_previous_camera_matrix: bool,
    /// Ledger A125 (`MulticamCompat::keeps_rational_coefficients`): method 2 keeps k4, k5,
    /// k6, which it does not optimise.
    pub multicam_keeps_rational_coefficients: bool,
    /// Ledger A126 (`MulticamCompat::trials_not_flagged`): the optimisation flags no trial
    /// for recomputation.
    pub multicam_trials_not_flagged: bool,
    /// Ledger A127 (`MulticamCompat::failure_written_back`): a refusal of the solver or
    /// non-finite parameters are written back.
    pub multicam_failure_written_back: bool,
    /// Ledger A147 (CD22): the external calibration's Apply flags no trial for
    /// recomputation. Fixed: every trial is flagged.
    pub external_calibration_trials_not_flagged: bool,
    /// Ledger A148 (CD7): the external-calibration frame's spin boxes hold 12 decimals
    /// (and +-999999999, sizes 0 to 100000), so Apply rounds every value. Fixed: the values
    /// are applied as given.
    pub external_calibration_spin_box_rounding: bool,
    /// Ledger A162: the external calibration's Apply accepts a rotation matrix that is not
    /// a rotation (`cv::Rodrigues` orthogonalises it), a width or height of 0 and a camera
    /// matrix with a focal length of 0 or less. Fixed: refused with a sentence.
    pub external_calibration_accepts_degenerate: bool,
    /// Ledger A165: "APPLY CHANGES" stores the rotation vector `cv::Rodrigues` gives for the
    /// rotation matrix, whose `sin(theta) < 1e-5` branches store a rotation within about
    /// 1e-5 rad of 0 as none and one within about 1e-5 rad of 180 degrees with an
    /// approximate axis (the stored rotation differs from the matrix by up to 2.1e-5 in an
    /// entry). Fixed: the rotation vector of the nearest rotation, exact to rounding
    /// (`xma_vision::camera::nearest_rotation_to_rodrigues`).
    pub external_rotation_cv_rodrigues: bool,
    /// Ledger A161: an undistortion refit with every grid point an outlier is applied: no
    /// grid model (every point maps to itself), the grid marked computed. Fixed: refused
    /// with a sentence, as every matching or refit that gives no model is in both modes.
    pub grid_fit_without_inliers_accepted: bool,
    /// Ledger F46: File > New dataset accepts a frame specification with no bead (0 bytes,
    /// or a header line only) and reference ids beyond the beads (`loadCoords` returns 0,
    /// which `createProject` ignores), and creates the project. Fixed: refused with a
    /// sentence. (No later code reads out of range: the cube calibration refuses the ids,
    /// ledger A114.)
    pub frame_without_beads_accepted: bool,
    /// Ledger F48: File > New dataset (external calibration) gives each camera a
    /// calibration image named "" (`loadImages({""})`), which the project file stores, and
    /// reads back, as "external". Fixed: named "external" when created, as after reopening.
    pub external_calibration_image_unnamed: bool,
    /// Ledger F49, F50: NaN and infinite values are kept where they arise (a computation,
    /// a project file), written as `nan`, `-nan`, `inf`, `-inf` (`std::ofstream`), and read
    /// back as XMALab reads them: `in >> double` stops at such a word, so a data row
    /// starting with one is skipped and a later one leaves the row short (a read past its
    /// end in the C++, an error here). Fixed: every such value is stored as its field's
    /// "not set" value (`xma_io::finite`), and a file holding one opens with a warning
    /// naming the entry and line.
    pub nonfinite_values_kept: bool,
    /// Ledger F52: an imported file's `nan` and `inf` fields (2D points, marker and rigid-body
    /// reference points, virtual-point coordinates, a frame specification) are read with
    /// `QString::toDouble` and stored as NaN and infinities. Fixed: the import is refused
    /// with a sentence naming the file and line.
    pub nonfinite_imports_accepted: bool,
    /// Ledger F59: a dataset made from an XMA Portal project numbers its cameras' portal
    /// ids (and names, "Camera <id>") from every `cameraNumber` in the metadata, trial
    /// videos and `cam0` included (`Project::parseXMLData`, `Project::addCamera`). Fixed:
    /// each camera takes the number of the calibration files it was given.
    pub portal_camera_ids_from_all_files: bool,
    /// Ledger F60: XMA Portal metadata added to a project or trial that has some is
    /// appended to it (`Project::setXMLData`, `Trial::setXMLData`), so the old and the new
    /// study data are both kept and saved. Fixed: the new metadata replaces the old.
    pub portal_metadata_appended: bool,
    /// Ledger F63: the top folder of an XMA Portal archive is the first folder by name, so
    /// the `__MACOSX` folder of a zip made on macOS is taken and the archive is "Not a
    /// valid xma file from the portal". Fixed: `__MACOSX` is skipped and the first folder
    /// that holds the metadata file is taken.
    pub portal_macos_folder_taken: bool,
    /// Ledger F64: metadata added to a project ("Add metadata from XMAPortal file") gives
    /// its cameras no portal ids until the project is opened again (only
    /// `Project::addCamera` assigns them). Fixed: the cameras get them at once.
    pub portal_ids_left_unset: bool,
    /// Ledger F73: the project file does not store which axis of a checkerboard is inverted
    /// (`CheckerboadInvertedAxis`), so opening a project builds the board with the setting
    /// in force then, which need not be the one its poses were computed with. Fixed: the
    /// writer adds the board's axis as the attribute `InvertedAxis` of `CalibrationObject`
    /// (XMALab 3 ignores it), and the reader uses it when present.
    pub checkerboard_axis_from_setting: bool,
    /// Ledger A190: "Denoise Images" reads the filter strength as an integer
    /// (`DenoiseDialog::getFilterStrength` returns `int`), so 5.5 filters at 5. Fixed: the
    /// strength as entered.
    pub denoise_strength_truncated: bool,
    /// Ledger A191: "Denoise Images" denoises frame `N - w` (`w = (temporal - 1) / 2`) with
    /// a temporal window reaching frame `N`, past the end, which OpenCV refuses with an
    /// exception that ends the camera's job there. Fixed: the last `w` frames are copied, as
    /// the first `w` are. XMALab 3 mode: the job stops at that frame with an error.
    pub denoise_window_past_end: bool,
    /// Ledger A192: "Denoise Images" writes each denoised frame into the frame it holds in
    /// memory (`out_image = images[i]` shares the pixels), so the next frames are denoised
    /// with already denoised frames in their temporal window. Fixed: every frame is
    /// denoised from the original frames.
    pub denoise_neighbours_already_denoised: bool,
    /// Ledger V26: File > Export > Undistorted Trial images writes nothing for a camera that
    /// has an undistortion grid image whose grid is not computed, or whose frames differ in
    /// size from the grid (`UndistortionObject::undistort` returns false), without a word.
    /// Fixed: an uncomputed grid is treated as no grid (the lens model, else the image as
    /// it is), and frames of another size are reported.
    pub trial_images_uncomputed_grid_writes_nothing: bool,
    /// Ledger V27: File > Export > Undistorted Trial images in "avi" opens
    /// `cv::VideoWriter` with the codec -1, which the FFmpeg back end refuses on Linux, so
    /// nothing is written. Fixed: a Motion JPEG AVI is written.
    pub trial_images_avi_not_written: bool,
    /// Ledger V28: File > Export > Undistorted Trial images writes the 8-bit image XMALab
    /// holds also for a source deeper than 8 bits. Fixed (the author's V1 decision): a TIFF
    /// at the source's depth.
    pub trial_images_eight_bit: bool,
    /// Ledger V30: "Denoise Images" writes a flipped camera's frames mirrored (as XMALab
    /// holds them), and relinking reads them mirrored again, so the trial then shows them
    /// unmirrored, unlike its points. Fixed: written as the source files hold them.
    pub denoise_writes_mirrored: bool,
    /// Ledger F65: File > Export > All Trials for External passes the camera index as
    /// `saveTrialImages`' `filter` argument and leaves `id` at -1, so every camera's
    /// folder receives the images of every camera, in `<base>UND` subfolders, unfiltered in
    /// "Camera 0" and filtered (the visual filter) in the others. Fixed: each camera's own
    /// images, unfiltered, in its `images` folder.
    pub export_all_trials_images_misplaced: bool,
    /// Ledger F66: "relink Trial to denoised images" links every `*.jpg` file in each
    /// `Cam<i>` folder, so files left there by an earlier run (or anything else) become
    /// frames of the trial. Fixed: the files this run wrote.
    pub denoise_relinks_folder_listing: bool,
    /// Ledger F67: File > Export > All Trials for External reads the video list of a
    /// Default trial, which has none (`getVideoStreams()[i]`, out of range: a crash in
    /// practice), after writing its first camera's `calib` folder. Fixed: the Default
    /// trial's folders are written without images. XMALab 3 mode: the export stops there
    /// with an error.
    pub export_all_trials_default_trial_fails: bool,
    /// Ledger F69: a project without calibration images takes its camera sizes from the
    /// first trial only when opened (`Trial::setCameraSizes`): a camera whose video there is
    /// not found stays 0 x 0, and a project without trials reads past the trial list (a
    /// crash; the XMALab 3 mode warns instead). Fixed: each camera's size from the first
    /// trial whose video for it can be read.
    pub camera_sizes_from_first_trial_only: bool,
    /// Ledger F70: the names an export builds from the project's names (trials, cameras,
    /// descriptions, video file names) are written as given: `:`, `*`, `?`, `"`, `<`, `>`,
    /// `|`, control characters, trailing dots and spaces and Windows device names kept.
    /// Fixed: replaced so that every system can create them. (`.`, `..`, an empty name and,
    /// in this mode, a name with a separator are refused in both modes.)
    pub export_names_as_given: bool,
    /// Ledger F71: Undistorted Trial images puts two cameras whose videos share a base name
    /// into the same `<base>UND` folder, the second replacing the first's files. Fixed: each
    /// such camera gets `<base>UND_cam<n>`.
    pub trial_images_same_folder_for_same_name: bool,
    /// Ledger A210: the pose from 2D rays scores every combination of ray candidates,
    /// however many there are (up to 4^n for n rays; the C++ builds the whole list and runs
    /// out of memory at about fourteen rays). Fixed: above 1,000,000 combinations the frame
    /// gets no pose from single-camera markers, and the Console names the body and frame.
    pub pose_from_2d_uncapped: bool,
    /// Ledger A216: Import 2D Points and the point edits over many cells (Reset above
    /// error, swap, interpolate) triangulate, recompute the rigid bodies and update the
    /// mean size after every single point. Fixed: the points are written first and the
    /// follow-up work is done once per marker and frame at the end.
    pub point_edits_follow_up_per_point: bool,
}

impl CppCompat {
    /// Every fault fixed: the default.
    pub const FIXED: CppCompat = CppCompat {
        virtual_points_per_camera_pair: false,
        virtual_points_at_origin: false,
        distortion_in_refinement: false,
        filter_cutoff_single_pass: false,
        grid_first_when_distorting: false,
        five_iterations: false,
        error_sd_nan: false,
        single_precision: false,
        frames_tracked_all: false,
        reprojection_error_all_cameras: false,
        epnp_last_camera: false,
        rays_misaligned: false,
        refine_failure_ignored: false,
        uncalibrated_cameras_in_refinement: false,
        refinement_without_skew: false,
        triangulation_unchecked: false,
        project_undefined_points: false,
        poses_at_load: false,
        pooled_sd_total_count: false,
        size_range_from_sum: false,
        search_centre_plus_one: false,
        threshold_offset_wraps: false,
        edit_first_body_only: false,
        edits_not_refiltered: false,
        finalize_body_by_body: false,
        cubic_interpolation_stops: false,
        rejected_reference_file_keeps_reset: false,
        exhaustive_degenerate_reference_search: false,
        remove_point_by_value: false,
        virtual_points_keep_body_index: false,
        virtual_point_file_chooses_body: false,
        clear_point_idx_keeps_original: false,
        import_2d_empty_cell_zero: false,
        interpolate_clears_marks: false,
        redetect_all_tracked: false,
        move_to_own_body_readds: false,
        penalty_off_centre: false,
        window_round_toward_zero: false,
        polyfit_rotation_bracket: false,
        refine_accepts_non_peak: false,
        swap_points_only: false,
        reset_keeps_projected_x: false,
        offset_sum_f32: false,
        six_digit_attributes: false,
        filtered_export_trial_cutoff: false,
        old_penalty_zero: false,
        offset_header: false,
        drop_unreadable_mesh: false,
        export_details: false,
        partial_project_on_resolution_mismatch: false,
        calibration_video_lengths_unchecked: false,
        trial_video_lengths_unchecked: false,
        change_trial_data_keeps_arrays: false,
        change_trial_data_renames_on_refusal: false,
        external_calibration_frame_count_zero: false,
        calibration_rand_continues: false,
        grid_neighbour_for_two_targets: false,
        grid_sites_compared_by_coordinates: false,
        grid_reference_scale_unchecked: false,
        outlier_circle_sets_inliers: false,
        outlier_circle_radius_signed: false,
        cube_reference_from_undistorted_position: false,
        cube_success_after_failed_refit: false,
        principal_point_one_based: false,
        calibration_snap_rounded: false,
        delete_frame_keeps_trial_references: false,
        failed_recalibration_cleared: false,
        recalibration_trials_not_flagged: false,
        remove_optimization_on_active_image: false,
        refused_calibration_keeps_beads: false,
        checkerboard_undistorted_points_raw: false,
        checkerboard_unconfirmed_corners_stored: false,
        checkerboard_frames_counted_on_active_camera: false,
        checkerboard_homography_ignores_inverted_axis: false,
        checkerboard_manual_with_unset_clicks: false,
        planar_frames_without_points_used: false,
        checkerboard_degenerate_clicks_accepted: false,
        checkerboard_failed_detection_reuses_corners: false,
        checkerboard_single_frame_calibrated: false,
        checkerboard_numbering_per_camera: false,
        multicam_chain_skips_frame_zero: false,
        multicam_chain_uses_reference_image_pose: false,
        multicam_jacobian_nan_at_zero_difference: false,
        multicam_lens_maps_with_previous_camera_matrix: false,
        multicam_keeps_rational_coefficients: false,
        multicam_trials_not_flagged: false,
        multicam_failure_written_back: false,
        external_calibration_trials_not_flagged: false,
        external_calibration_spin_box_rounding: false,
        external_calibration_accepts_degenerate: false,
        external_rotation_cv_rodrigues: false,
        grid_fit_without_inliers_accepted: false,
        frame_without_beads_accepted: false,
        external_calibration_image_unnamed: false,
        nonfinite_values_kept: false,
        nonfinite_imports_accepted: false,
        portal_camera_ids_from_all_files: false,
        portal_metadata_appended: false,
        portal_macos_folder_taken: false,
        portal_ids_left_unset: false,
        checkerboard_axis_from_setting: false,
        denoise_strength_truncated: false,
        denoise_window_past_end: false,
        denoise_neighbours_already_denoised: false,
        trial_images_uncomputed_grid_writes_nothing: false,
        trial_images_avi_not_written: false,
        trial_images_eight_bit: false,
        denoise_writes_mirrored: false,
        export_all_trials_images_misplaced: false,
        denoise_relinks_folder_listing: false,
        export_all_trials_default_trial_fails: false,
        camera_sizes_from_first_trial_only: false,
        export_names_as_given: false,
        trial_images_same_folder_for_same_name: false,
        pose_from_2d_uncapped: false,
        point_edits_follow_up_per_point: false,
    };

    /// Everything as XMALab 3 (commit `2967a7e`) does it: the mode `xma parity` runs in and
    /// that `--xmalab3-compat` selects.
    pub const XMALAB3: CppCompat = CppCompat {
        virtual_points_per_camera_pair: true,
        virtual_points_at_origin: true,
        distortion_in_refinement: true,
        filter_cutoff_single_pass: true,
        grid_first_when_distorting: true,
        five_iterations: true,
        error_sd_nan: true,
        single_precision: true,
        frames_tracked_all: true,
        reprojection_error_all_cameras: true,
        epnp_last_camera: true,
        rays_misaligned: true,
        refine_failure_ignored: true,
        uncalibrated_cameras_in_refinement: true,
        refinement_without_skew: true,
        triangulation_unchecked: true,
        project_undefined_points: true,
        poses_at_load: true,
        pooled_sd_total_count: true,
        size_range_from_sum: true,
        search_centre_plus_one: true,
        threshold_offset_wraps: true,
        edit_first_body_only: true,
        edits_not_refiltered: true,
        finalize_body_by_body: true,
        cubic_interpolation_stops: true,
        rejected_reference_file_keeps_reset: true,
        exhaustive_degenerate_reference_search: true,
        remove_point_by_value: true,
        virtual_points_keep_body_index: true,
        virtual_point_file_chooses_body: true,
        clear_point_idx_keeps_original: true,
        import_2d_empty_cell_zero: true,
        interpolate_clears_marks: true,
        redetect_all_tracked: true,
        move_to_own_body_readds: true,
        penalty_off_centre: true,
        window_round_toward_zero: true,
        polyfit_rotation_bracket: true,
        refine_accepts_non_peak: true,
        swap_points_only: true,
        reset_keeps_projected_x: true,
        offset_sum_f32: true,
        six_digit_attributes: true,
        filtered_export_trial_cutoff: true,
        old_penalty_zero: true,
        offset_header: true,
        drop_unreadable_mesh: true,
        export_details: true,
        partial_project_on_resolution_mismatch: true,
        calibration_video_lengths_unchecked: true,
        trial_video_lengths_unchecked: true,
        change_trial_data_keeps_arrays: true,
        change_trial_data_renames_on_refusal: true,
        external_calibration_frame_count_zero: true,
        calibration_rand_continues: true,
        grid_neighbour_for_two_targets: true,
        grid_sites_compared_by_coordinates: true,
        grid_reference_scale_unchecked: true,
        outlier_circle_sets_inliers: true,
        outlier_circle_radius_signed: true,
        cube_reference_from_undistorted_position: true,
        cube_success_after_failed_refit: true,
        principal_point_one_based: true,
        calibration_snap_rounded: true,
        delete_frame_keeps_trial_references: true,
        failed_recalibration_cleared: true,
        recalibration_trials_not_flagged: true,
        remove_optimization_on_active_image: true,
        refused_calibration_keeps_beads: true,
        checkerboard_undistorted_points_raw: true,
        checkerboard_unconfirmed_corners_stored: true,
        checkerboard_frames_counted_on_active_camera: true,
        checkerboard_homography_ignores_inverted_axis: true,
        checkerboard_manual_with_unset_clicks: true,
        planar_frames_without_points_used: true,
        checkerboard_degenerate_clicks_accepted: true,
        checkerboard_failed_detection_reuses_corners: true,
        checkerboard_single_frame_calibrated: true,
        checkerboard_numbering_per_camera: true,
        multicam_chain_skips_frame_zero: true,
        multicam_chain_uses_reference_image_pose: true,
        multicam_jacobian_nan_at_zero_difference: true,
        multicam_lens_maps_with_previous_camera_matrix: true,
        multicam_keeps_rational_coefficients: true,
        multicam_trials_not_flagged: true,
        multicam_failure_written_back: true,
        external_calibration_trials_not_flagged: true,
        external_calibration_spin_box_rounding: true,
        external_calibration_accepts_degenerate: true,
        external_rotation_cv_rodrigues: true,
        grid_fit_without_inliers_accepted: true,
        frame_without_beads_accepted: true,
        external_calibration_image_unnamed: true,
        nonfinite_values_kept: true,
        nonfinite_imports_accepted: true,
        portal_camera_ids_from_all_files: true,
        portal_metadata_appended: true,
        portal_macos_folder_taken: true,
        portal_ids_left_unset: true,
        checkerboard_axis_from_setting: true,
        denoise_strength_truncated: true,
        denoise_window_past_end: true,
        denoise_neighbours_already_denoised: true,
        trial_images_uncomputed_grid_writes_nothing: true,
        trial_images_avi_not_written: true,
        trial_images_eight_bit: true,
        denoise_writes_mirrored: true,
        export_all_trials_images_misplaced: true,
        denoise_relinks_folder_listing: true,
        export_all_trials_default_trial_fails: true,
        camera_sizes_from_first_trial_only: true,
        export_names_as_given: true,
        trial_images_same_folder_for_same_name: true,
        pose_from_2d_uncapped: true,
        point_edits_follow_up_per_point: true,
    };

    /// The ledger items and whether each is reproduced, in ledger order.
    #[must_use]
    pub fn items(&self) -> Vec<(&'static str, bool)> {
        let mut v = vec![
            (self.virtual_points_per_camera_pair, "A2"),
            (self.virtual_points_at_origin, "A3"),
            (self.distortion_in_refinement, "A4"),
            (self.filter_cutoff_single_pass, "A7"),
            (self.grid_first_when_distorting, "A11"),
            (self.five_iterations, "A19"),
            (self.error_sd_nan, "A20"),
            (self.single_precision, "A23"),
            (self.frames_tracked_all, "A24"),
            (self.reprojection_error_all_cameras, "A25"),
            (self.epnp_last_camera, "A26"),
            (self.rays_misaligned, "A27"),
            (self.refine_failure_ignored, "A28"),
            (self.uncalibrated_cameras_in_refinement, "A29"),
            (self.refinement_without_skew, "A30"),
            (self.triangulation_unchecked, "A34"),
            (self.project_undefined_points, "A35"),
            (self.poses_at_load, "A54"),
            (self.pooled_sd_total_count, "A58"),
            (self.size_range_from_sum, "A1"),
            (self.search_centre_plus_one, "A14"),
            (self.threshold_offset_wraps, "A15"),
            (self.edit_first_body_only, "A31"),
            (self.edits_not_refiltered, "A60"),
            (self.finalize_body_by_body, "A91"),
            (self.cubic_interpolation_stops, "A33"),
            (self.rejected_reference_file_keeps_reset, "A76"),
            (self.exhaustive_degenerate_reference_search, "A93"),
            (self.remove_point_by_value, "A77"),
            (self.virtual_points_keep_body_index, "A78"),
            (self.virtual_point_file_chooses_body, "A83"),
            (self.clear_point_idx_keeps_original, "A79"),
            (self.import_2d_empty_cell_zero, "F30"),
            (self.interpolate_clears_marks, "A61"),
            (self.redetect_all_tracked, "A62"),
            (self.move_to_own_body_readds, "U79"),
            (self.penalty_off_centre, "A64"),
            (self.window_round_toward_zero, "A65"),
            (self.polyfit_rotation_bracket, "A66"),
            (self.refine_accepts_non_peak, "A90"),
            (self.swap_points_only, "A67"),
            (self.reset_keeps_projected_x, "A68"),
            (self.offset_sum_f32, "A73"),
            (self.six_digit_attributes, "F4"),
            (self.filtered_export_trial_cutoff, "F11"),
            (self.old_penalty_zero, "F13"),
            (self.offset_header, "F2"),
            (self.drop_unreadable_mesh, "F18"),
            (self.export_details, "F25"),
            (self.partial_project_on_resolution_mismatch, "F34"),
            (self.calibration_video_lengths_unchecked, "F35"),
            (self.trial_video_lengths_unchecked, "F36"),
            (self.change_trial_data_keeps_arrays, "F37"),
            (self.change_trial_data_renames_on_refusal, "F37"),
            (self.external_calibration_frame_count_zero, "A128"),
            (self.calibration_rand_continues, "A5"),
            (self.grid_neighbour_for_two_targets, "A104"),
            (self.grid_sites_compared_by_coordinates, "A105"),
            (self.grid_reference_scale_unchecked, "A106"),
            (self.outlier_circle_sets_inliers, "A107"),
            (self.outlier_circle_radius_signed, "A108"),
            (self.cube_reference_from_undistorted_position, "A110"),
            (self.cube_success_after_failed_refit, "A111"),
            (self.principal_point_one_based, "A112"),
            (self.calibration_snap_rounded, "A130"),
            (self.delete_frame_keeps_trial_references, "A131"),
            (self.failed_recalibration_cleared, "A132"),
            (self.recalibration_trials_not_flagged, "A133"),
            (self.remove_optimization_on_active_image, "A134"),
            (self.refused_calibration_keeps_beads, "A135"),
            (self.checkerboard_undistorted_points_raw, "A180"),
            (self.checkerboard_unconfirmed_corners_stored, "A181"),
            (self.checkerboard_frames_counted_on_active_camera, "A182"),
            (self.checkerboard_homography_ignores_inverted_axis, "A183"),
            (self.checkerboard_manual_with_unset_clicks, "A184"),
            (self.planar_frames_without_points_used, "A188"),
            (self.checkerboard_degenerate_clicks_accepted, "A202"),
            (self.checkerboard_failed_detection_reuses_corners, "A203"),
            (self.checkerboard_single_frame_calibrated, "A204"),
            (self.checkerboard_numbering_per_camera, "A189"),
            (self.multicam_chain_skips_frame_zero, "A121"),
            (self.multicam_chain_uses_reference_image_pose, "A122"),
            (self.multicam_jacobian_nan_at_zero_difference, "A123"),
            (self.multicam_lens_maps_with_previous_camera_matrix, "A124"),
            (self.multicam_keeps_rational_coefficients, "A125"),
            (self.multicam_trials_not_flagged, "A126"),
            (self.multicam_failure_written_back, "A127"),
            (self.external_calibration_trials_not_flagged, "A147"),
            (self.external_calibration_spin_box_rounding, "A148"),
            (self.grid_fit_without_inliers_accepted, "A161"),
            (self.external_calibration_accepts_degenerate, "A162"),
            (self.external_rotation_cv_rodrigues, "A165"),
            (self.frame_without_beads_accepted, "F46"),
            (self.external_calibration_image_unnamed, "F48"),
            (self.nonfinite_values_kept, "F49"),
            (self.nonfinite_imports_accepted, "F52"),
            (self.portal_camera_ids_from_all_files, "F59"),
            (self.portal_metadata_appended, "F60"),
            (self.portal_macos_folder_taken, "F63"),
            (self.portal_ids_left_unset, "F64"),
            (self.checkerboard_axis_from_setting, "F73"),
            (self.denoise_strength_truncated, "A190"),
            (self.denoise_window_past_end, "A191"),
            (self.denoise_neighbours_already_denoised, "A192"),
            (self.trial_images_uncomputed_grid_writes_nothing, "V26"),
            (self.trial_images_avi_not_written, "V27"),
            (self.trial_images_eight_bit, "V28"),
            (self.denoise_writes_mirrored, "V30"),
            (self.export_all_trials_images_misplaced, "F65"),
            (self.denoise_relinks_folder_listing, "F66"),
            (self.export_all_trials_default_trial_fails, "F67"),
            (self.camera_sizes_from_first_trial_only, "F69"),
            (self.export_names_as_given, "F70"),
            (self.trial_images_same_folder_for_same_name, "F71"),
            (self.pose_from_2d_uncapped, "A210"),
            (self.point_edits_follow_up_per_point, "A216"),
        ];
        v.sort_by_key(|(_, k)| (k.as_bytes()[0], k[1..].parse::<u32>().unwrap_or(0)));
        v.into_iter().map(|(on, k)| (k, on)).collect()
    }

    /// A short name for reports: `fixed`, `XMALab 3 compatible`, or the list of the faults
    /// reproduced for a mixed mode.
    #[must_use]
    pub fn describe(&self) -> String {
        if *self == Self::FIXED {
            return "fixed (the default: known faults of XMALab 3 corrected)".to_owned();
        }
        if *self == Self::XMALAB3 {
            return "XMALab 3 compatible (known faults reproduced as the C++ has them)".to_owned();
        }
        let on: Vec<&str> = self
            .items()
            .into_iter()
            .filter(|(_, on)| *on)
            .map(|(k, _)| k)
            .collect();
        format!("mixed (as the C++: {})", on.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_fixes_everything() {
        assert_eq!(CppCompat::default(), CppCompat::FIXED);
        assert_ne!(CppCompat::FIXED, CppCompat::XMALAB3);
        assert!(CppCompat::FIXED.describe().starts_with("fixed"));
        assert!(CppCompat::XMALAB3.describe().starts_with("XMALab 3"));
        assert!(CppCompat::FIXED.items().iter().all(|(_, on)| !on));
        assert!(CppCompat::XMALAB3.items().iter().all(|(_, on)| *on));
        assert_eq!(CppCompat::FIXED.items().len(), 116);
        let mixed = CppCompat {
            offset_header: true,
            five_iterations: true,
            ..CppCompat::FIXED
        };
        assert_eq!(mixed.describe(), "mixed (as the C++: A19, F2)");
    }
}
