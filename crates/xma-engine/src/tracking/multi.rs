//! Multi-camera tracking: a tracking method that uses all cameras together. New in this
//! program; XMALab 3 has no such method, and it is refused in the XMALab 3 mode.
//!
//! XMALab's tracker follows each marker in each camera on its own and checks the two
//! cameras against each other only through the reprojection error, which cannot see a point
//! that slides along another camera's epipolar line. This method predicts each marker in 3D,
//! looks for several candidate dots per camera, and chooses one dot per camera for all
//! markers together, so that the chosen dots agree with each other (small reprojection
//! error) and with the prediction (a gate in every camera), no dot goes to two markers, and
//! a camera whose dot is in doubt is left undefined. A marker that is lost is looked for
//! again round the position its neighbours give and accepted only when its distances to
//! them confirm its identity.
//!
//! # What the run reads
//!
//! The run keeps its own record of trusted observations and reads the document's points
//! through nothing else (`View`): the 2D points at the start frame and behind it in the
//! run's direction, as the document holds them when the run starts; the points this run has
//! written; and points it may not overwrite (set by hand, and the others of
//! `docs/cpp-tracking.md` 3.4 that hold a position, such as optimised points when the
//! setting protects them) at any frame. Only cameras it tracks (shown, calibrated, with a
//! pose) are read; a hidden camera is neither read nor written. Its 3D positions
//! (prediction, depth, rigid statistics, rigid prediction, reprojection errors) are
//! triangulated from those observations, never taken from the document's stored 3D points
//! (which may be stale, ledger F9, or rest on points the run may not read). Its marker
//! sizes (template size, detection mask) are the mean of the sizes of those observations.
//!
//! **Points it did not detect itself are refined.** A point the run reads but did not write
//! (the points at the start frame and behind it, as far back as a template, a velocity or a
//! 2D prediction reaches: `max_gap` plus `velocity_window` frames, 7 with the defaults, but
//! never more than 16 (`REFINE_REACH`), so that a huge `max_gap` or `velocity_window` does
//! not make the first step read every image behind the start; and a protected point it
//! reaches) may lie anywhere in the marker's dark shape: a point placed without detection,
//! a click whose detection found nothing, a point imported from elsewhere (markers are not
//! always round: conical metal points, for one, and users click anywhere in the blob). The
//! run therefore works from its refined position: the marker's own detection run at the
//! stored point in that frame's image, run again until it settles (at most 6 runs, the last
//! moving no more than 1 px), accepted only when its
//! blob is no larger than twice the marker's mean size, its centre is more than `same_dot`
//! and at most 10 px away, and it lies on one dark shape with the stored point and with no
//! other marker's point the run reads there (`refine_stored_point`); otherwise the stored
//! point is used. Where the run reads the marker's points at that frame in two cameras or
//! more, the other cameras check the refinement: the largest reprojection error of the 3D
//! point from the positions the run would use must be no more than the larger of
//! `max_reprojection_error` and that from the stored points (`refinement_agrees`);
//! otherwise that frame's refinements are dropped. A point already at a detected centre
//! (within `same_dot`) is used exactly as it is. The refined position replaces the stored
//! one in the run's own record only: the templates, the contrast baseline, the 2D
//! predictions and velocities and the run's own triangulation at that frame use it. The
//! document's point is never changed, and in the
//! joint choice a protected point is used as it is at its frame. For this the run also
//! reads the image of each frame whose point it refines; an image behind the start is read
//! for the refinement only and not kept after it (an economy that no test pins, review 16f:
//! with the cap the step could hold at most 16 frames' images for each camera in any case).
//! Points further behind the start (beyond `REFINE_REACH`, the rigid statistics' history of
//! `rigid_history` frames, the depth for a lost marker) stay as stored: a start whose
//! history is off-centre learns its rigid distances from those points, so they are off by
//! about the offset in 3D (not measured).
//!
//! A point an earlier run wrote is, for a new run, a point it did not detect itself, so it
//! is refined too, with the new run's own mean size: a run of one step (the window's "next
//! frame") repeated frame by frame refines each step's start again, and so differs from one
//! continuous run more than before refinement did (review 16e, the Rat's frames 1 to 26:
//! single-step runs differ from those without refinement in 6 cells, a continuous run in 1).
//!
//! The known limit of the refinement: a marker that touches, at the start, something that
//! is not a marker of the run (a bead nobody digitised, a marker left out of
//! `multi-selected`, an edge of bone) makes one dark shape with it, and the refinement can
//! take that merged shape's centre. When the move is consistent between the cameras, or
//! lies along the other camera's epipolar line, the cross-camera check does not see it, and
//! the run works from the merged centre (about 3 px off the marker in review 16e's scene,
//! where the size check does not refuse it: its blob is 1.49 times the marker's mean size)
//! for as long as the two stay merged. With the marker's size unknown (-1) no size check
//! applies at all.
//!
//! The wider first search (3. below) also looks at the image of the frame one step beyond
//! the frame being tracked, in a camera the run tracks, to confirm a candidate, and at the
//! image of the frame of the marker's earlier point; it reads those frames' images only,
//! never a point stored there or beyond (2D or 3D, whatever its status).
//!
//! A run starts from whatever points the user gives it: at the start frame and behind it
//! every point with data is read, whatever its status (predicted and interpolated points
//! included). Ahead of the start a predicted or interpolated point is never read. Such a
//! cell is treated as any cell the run may write (6. below): it is overwritten by the run's
//! point, or, for a marker the run has had, when the run finds none, left as it is unless the
//! run's own points contradict it (`keep_unconfirmed`, the default; reset to undefined
//! without it); an interpolated point is left as it is when `TrackInterpolatedPoints` is off
//! (it may then not be overwritten, and still is not read, since it holds no position set by
//! the user).
//!
//! **A point the run leaves as it is is never read** (`keep_unconfirmed`, 6. below): it is
//! not at or behind the start, not written by the run and not protected, so it is not in the
//! run's record. The points a run writes are therefore the same whether such points are
//! there or not, and whether the parameter is on or off. A run of one step repeated (the
//! window's "next frame") starts each step from whatever the cell at its start frame holds,
//! so a point kept by one single step becomes the next single step's start point, where one
//! continuous run never reads it.
//!
//! A run tracks only if at least two calibrated cameras with a pose are shown
//! ([`tracked_cameras`]); with fewer, [`MultiTracker::new`] refuses with a sentence.
//!
//! # Which markers are the run's
//!
//! Only a marker the run has had is searched for, and only its cells at frame `e` can be
//! reset (6. below): a marker for which the run has held a trusted observation in a tracked
//! camera at some frame (at or behind the start, written by the run, or a protected point it
//! has reached). Such a marker stays the run's for the rest of the run, also while it is
//! lost: an older run's points of a marker this run lost are left as they are with
//! `keep_unconfirmed` (nothing the run wrote contradicts them) and erased without it. A
//! marker the run never had is neither searched for nor touched. A marker with a protected
//! (hand-set) point ahead and nothing at the start becomes the run's when the run reaches
//! that frame, and from there on it is searched for and its cells are handled as above.
//!
//! One step from frame `s` to the next frame `e` ([`MultiTracker::step`]); every parameter
//! named is a field of [`MultiTrackParams`], which gives its default:
//!
//! 1. **Templates and contrast.** For each marker and shown calibrated camera with a point
//!    at `s`, the template is cut there (the same square as XMALab's tracker,
//!    `2 (size + 3) + 1` pixels round the point), and the dot's contrast
//!    ([`xma_vision::joint_tracking::dot_contrast`]) and its contrast relative to the
//!    brightness round it ([`xma_vision::joint_tracking::dot_contrast_relative`]: the
//!    contrast over the mean grey level of the same ring) are added to the marker's recent
//!    contrasts in that camera (the last `contrast_history`; the median of one of the two
//!    is the test of 3. below). A marker not found at `s`
//!    keeps the template of the last frame it was found at (at the run's start, the last
//!    point within `max_gap` frames behind `s`). A template is dropped when the marker has
//!    held no point in that camera for more than `max(max_gap, depth_gap) + 1` frames: the
//!    age that counts is that of the marker's latest point there, merged or not ("Overlaps"
//!    below), so a marker still followed never loses its template, and one that has gone
//!    loses it as before. With `split_touching`, for the dot-like methods, the dot's width
//!    is learned at the same cells (the same points, merged ones excluded): one Gaussian dot
//!    fitted at the run's position with centre, width and depth free
//!    ([`xma_vision::dot_fit::fit_dot_shape`], `learn_width` gives the window and the bounds),
//!    accepted when its centre is within `same_dot` of the position; the last
//!    `contrast_history` widths are kept per marker and camera, and their median is the
//!    marker's width there.
//! 2. **Prediction.** A marker is *followed* while a camera has had it within `max_gap`
//!    frames: constant velocity from its latest 3D point and the nearest earlier one within `velocity_window`
//!    frames (extrapolated over at most `velocity_frames`), projected into every camera
//!    with its distortion; a camera whose own 2D track is more recent (the marker was seen
//!    by that camera alone) uses its 2D track, and the 3D point is moved onto that
//!    camera's viewing ray ([`xma_vision::joint_tracking::closest_point_on_ray`]). When the
//!    3D track has no known velocity (no 3D point within `velocity_window` frames before its
//!    latest, as when another camera missed the marker for a few frames) but a camera's own
//!    2D track has one (an earlier point of its own within `velocity_window`), that camera
//!    uses its own 2D prediction (constant velocity, as above) instead of the 3D track's
//!    standing position: a camera that kept the marker is not refused its dot because the
//!    other camera lost it for a while. The gate is `gate` (`gate_hold` without a known
//!    velocity, in the camera's own track or the 3D track as used), widened by `gate_growth`
//!    per frame missed ([`MultiTrackParams::gate_for`]). A marker not followed is *lost*: it is
//!    looked for round the position its nearest followed neighbours give
//!    ([`MultiTracker`]'s rigid prediction: the rigid motion of up to `rigid_neighbours` of
//!    them since the latest frame within `depth_gap` where all were seen, gate
//!    `gate_rigid`), or, in a camera that lost it while another kept it, round the other
//!    camera's ray at the depth of its latest 3D point (gate `gate_max`). Such a camera is
//!    flagged as finding the marker again, and the choice checks its identity: its 3D point
//!    must keep its rigid links (4. below) to `reacquire_links` partners accepted at `e` and
//!    break none, or to all of its partners when it has fewer than that (a marker with one
//!    partner, as on a bone with two markers, needs that one; a marker with none cannot be
//!    found again). With
//!    `single_max_frames` set (0, the default, sets no limit), so is a camera that has seen
//!    the marker alone for more than that many frames since its latest 3D point (or since
//!    the run's start).
//!    **Placed by its distances** (`distance_placement`, new 25 November 2026; on by default
//!    since 26 November 2026, the author's decision; parity ledger A223, A224, A225). Where a
//!    lost marker has no rigid prediction (no frame with three followed neighbours, or its nearest ones there do not fit one rigid motion) and no
//!    camera follows it on its own (the ray rule below is unchanged), it is placed
//!    (`place_by_distances`) at the point that best keeps its pairs' mean distances to the
//!    markers followed in 3D at this step whose pair with it the run has measured at
//!    `rigid_min_frames` frames or more (four or more of them; four distances fix a point,
//!    three leave two): [`xma_vision::joint_tracking::point_from_distances`] of their predicted
//!    positions at `e`, each distance weighted by `1 / max(spread, rigid_min_tolerance *
//!    mean)^2`, started from the marker's latest clean 3D point within `depth_gap` moved by
//!    the displacement since then of the partner nearest to it by mean distance (no clean 3D
//!    point within `depth_gap`: no placement). The placement is searched for as a rigid
//!    prediction is (gate `gate_rigid`, flagged as finding the marker again), and its identity
//!    is checked by the statistics links of 4. to the partners it was placed from, with or
//!    without `stat_links` (a placement has no reference frame).
//! 3. **Candidates.** The local maxima of the template's zero-mean normalised correlation
//!    (at least `min_correlation`) within the gate plus `search_margin`, each refined by the
//!    marker's own detection method as the existing tracker detects after tracking (search
//!    area twice the mean size, at least 10). A candidate is dropped when a contour method
//!    (methods 0, 2, 5) finds no blob there, or the detection moves more than
//!    `max_detection_shift`; for the other methods (blob detector, corner, no detection) a
//!    detection that finds nothing leaves the correlation peak as the point (size -1). For
//!    the dot-like methods (0, 1, 2, 5, 6; dark or light as the method says) a candidate is
//!    also dropped when its contrast is below `min_contrast_fraction` of the median of the
//!    marker's recent contrasts in that camera (a faint dot of noise or bone edge, not a
//!    bead); with `contrast_relative` (the default since 24 November 2026), when its
//!    relative contrast is below `min_contrast_fraction` of the median of the recent
//!    relative contrasts instead (so that a bead over which something denser passes, which
//!    darkens the picture round it and lowers its contrast in proportion, still passes; and
//!    so that a marker whose background brightens for a while, which doubles its absolute
//!    contrast, is not lost when the background returns to normal, as the example pig
//!    project's marker 5 in camera 1 was at frames 7 to 15: the baseline is not updated
//!    while a marker is lost, so it stayed lost; measured figures at
//!    `MultiTrackParams::contrast_relative`; without it the absolute test), with two guards: the candidate's
//!    absolute contrast must also reach half of `min_contrast_fraction` of the median of
//!    the recent absolute contrasts (a black dot on an almost black ground has a relative
//!    contrast near 1 and an absolute one of a few grey levels); and where another marker
//!    of the run is predicted in this camera at `e` so close that its dot may lie in or on
//!    the candidate's ring (centre distance at most the ring's outer radius, the marker's
//!    dot radius plus 3.5 px, plus the other marker's dot radius; a dot radius is the
//!    marker's mean size, 2.5 without one), the absolute test is used instead (a dark dot
//!    in the ring darkens it and lifts the relative contrast: two markers merging). Corner
//!    and no-detection markers have no contrast test and no search by contrast. The same
//!    test applies wherever a candidate is refined, including the dots of the frame beyond
//!    `e` that confirm a candidate of the wider first search and the dots of the earlier
//!    frame that refuse one (below); those use the other markers' predictions at `e` (an
//!    approximation: a neighbour moves about one frame's motion between them). At most
//!    `max_candidates` per camera.
//!    **Wider first search.** In a camera where the marker is followed without a known
//!    velocity (neither the 3D track nor the camera's own 2D track has one, 2. above: one
//!    earlier point only, as at the first step of a run from one frame; gate `gate_hold`),
//!    the search also covers `gate_start` round the prediction, when that is a finite
//!    number above the ordinary gate (anything else turns it off; at most 200 is used) and
//!    the trial has a frame `e2` one step beyond `e`. That search takes the correlation
//!    maxima beyond the ordinary search's window only, strongest first, and examines at most
//!    64 of them until `max_candidates` pass the tests (so maxima within the ordinary
//!    window, or faint dots that fail the tests, do not crowd the marker out). A candidate
//!    of that search beyond the ordinary gate is kept only when (a) the image of the
//!    marker's earlier frame (the frame of the point its velocity is measured from) shows no
//!    dot, other than the marker's own, nearer to it than the marker's earlier point is
//!    (within its distance `D` from that point, at least 2 px; found as candidates are with
//!    the template cut at the candidate in the image of `e`; the marker's own dot is a dot
//!    within its dot radius plus `same_dot` of its earlier point): each new dot belongs to
//!    its nearest earlier dot unless something says otherwise, so a dot that was nearer is
//!    the likelier owner, whatever its speed (a row of dots standing or moving together; a
//!    known limit: a real fast marker is refused when another dot was nearer to where it
//!    arrives than it was, a loss and not a wrong point), and (b) `e2`
//!    confirms it: its velocity in this camera (from the marker's earlier point there, per
//!    frame) carried on to `e2` gives a position, and the image of `e2` holds a dot within
//!    `gate` of it, found as candidates are (correlation maxima of a template refined by the
//!    detection, with the same tests; the template is the one the next step would use, cut
//!    at the candidate in the image of `e`). When at least one is kept, the camera's gate is
//!    `gate_start`, the ordinary candidates beyond the ordinary gate (every one, with
//!    `start_confirm_near`) need the same tests, and the kept ones take the places the
//!    ordinary ones leave of `max_candidates`; otherwise the camera is as without the wider
//!    search. No frame `e2` in the trial, or nothing kept: no candidate beyond the ordinary
//!    gate. The camera's ordinary gate is passed to the choice as its near gate
//!    ([`xma_vision::joint_tracking::CameraOptions::near_gate`]): the cost's distance from
//!    the prediction is measured against it, and the marker's combinations are ranked in
//!    classes (two cameras or more without a far dot, then two cameras or more with one,
//!    then one camera; a point set by hand counts as a camera). The classes apply only to a
//!    marker with such a wide cell at this step; every other marker's combinations are
//!    ranked by cost alone, as without the wider search. So the wider search never
//!    costs a point the ordinary one would have accepted in two cameras or more, and a point
//!    seen by one camera only gives way to a far pair that triangulates.
//!    For a marker being found again, the dots that stand out by contrast in the window
//!    ([`xma_vision::joint_tracking::contrast_maxima`]) are candidates too
//!    (`reacquire_dots`; their floor is `min_contrast_fraction` of the median of the
//!    absolute contrasts, with or without `contrast_relative`, and each is then refined
//!    with the test above). A marker's reprojection limit rises above
//!    `max_reprojection_error` for a marker whose recent points agree less well
//!    (`reprojection_factor` times their median error, at most `max_reprojection_cap`).
//!    **Two touching dots** (`split_touching`, new 25 November 2026; on by default since
//!    26 November 2026, the author's decision; parity ledger A223, A224, A225). Where
//!    two dots touch, the detection's threshold joins them into one shape and returns the
//!    shape's centre, between the two. With the parameter on, for a dot-like method and a
//!    marker whose width is known in that camera (1. above; none before the run has learned
//!    one, so a run that starts while the marker touches something does not split it), the
//!    detected point of every refined peak (wherever a candidate is refined: the candidates,
//!    the confirmation at the frame beyond `e`, the earlier-frame test, the dots by contrast)
//!    is examined, when the detection found a shape (size above 0; a correlation peak taken
//!    as the point is no joined shape's centre): two dots of that width are fitted there when
//!    two explain the picture clearly better than one, the detected point lies nearer their
//!    middle than either of them, **the detected shape covers both fitted centres** (each
//!    within the shape's radius of the detected point: the detected size for methods 0, 2 and
//!    5, half of it for the blob methods 1 and 6, whose size is a diameter, ledger A69), and
//!    the picture holds no third dot (three dots fitted together do not explain it clearly
//!    better) ([`xma_vision::dot_fit::split_joined_dots`] gives the
//!    procedure and `MultiTrackParams::split_rules` its numbers; since review 19, 26 November
//!    2026: before it a clean detection with a separate dot 6 to 8 px beside it was replaced
//!    too, and a marker between two beads was moved off its dot). Where the detection stands
//!    on one dot there is nothing to repair. It works from the picture alone, so a bead that
//!    is not a marker of the run is split off as well. When two are
//!    taken, both fitted centres are candidates in place of the detected point, whether or
//!    not the detected point itself was within `max_detection_shift` of the peak (since round
//!    b19d: the joined shape's centre can lie 3 to 4.5 px from a peak on the marker's own dot,
//!    and the shift test is for the point offered, not for the centre it replaces; which of the
//!    two is the marker is left to the joint choice: its prediction, its gate and the other
//!    cameras), each in the image and within `max_detection_shift` of the peak, with size -1
//!    and the correlation surface's value at its nearest pixel (the peak's where the surface
//!    does not cover it), and with the contrast test of a candidate measured on the picture
//!    with the other fitted dot taken out ([`xma_vision::dot_fit::WithoutDot`]); for that
//!    test a marker of the run predicted within its dot radius of the other fitted centre is
//!    not a neighbour in the ring (its dot is the one taken out), while any other marker of
//!    the run predicted near makes the test the absolute one as above. Otherwise the
//!    candidate is the detected point, exactly as without the parameter. Each candidate
//!    carries whether it was fitted apart to the writing (6.).
//! 4. **Rigid links.** Every pair of the run's markers whose 3D distance has been measured
//!    at `rigid_min_frames` frames or more (the run's own 3D points: the `rigid_history`
//!    frames behind the start, then each frame tracked) and whose spread is at most
//!    `rigid_max_spread` of its mean counts as rigid (on one bone), with a tolerance of the
//!    larger of `rigid_sigmas` standard deviations and `rigid_min_tolerance` of the mean. A
//!    lost marker's rigid prediction adds links to the neighbours it was predicted from
//!    (their distances at the reference frame, tolerance
//!    `max(rigid_fit_tolerance, rigid_min_tolerance)` of the distance), which check its
//!    identity when it is found again. Where such a link names a pair that also has a
//!    learned link, the partner counts once, and only as kept when both links hold (one
//!    broken link makes the partner broken), for `reacquire_links` and for
//!    `rigid_majority` alike.
//!    **Statistics links** (`stat_links`, new 25 November 2026; on by default since 26
//!    November 2026, the author's decision; parity ledger A223, A224, A225). For a
//!    marker being found again (a camera flagged so, 2. above), a link is added to every other
//!    marker of the run whose pair with it the run has measured at `rigid_min_frames` frames
//!    or more and that is not rigid (a rigid pair has its learned link, the same numbers):
//!    the pair's mean distance, tolerance the larger of `rigid_sigmas` times its spread and
//!    `rigid_min_tolerance` of the mean (`pair_statistics`); and the rigid prediction makes
//!    no reference-frame link for a pair with such statistics (a pair measured at fewer
//!    frames keeps it). The check is unchanged: `reacquire_links` partners kept (or all of
//!    them when it has fewer, at least one) and none broken. The links go to every such
//!    marker of the run, near or far; one whose partner has no point at this step counts
//!    neither as kept nor as broken, and of two markers found again at the same step the one
//!    served later counts the one served earlier. Followed markers get no such links (they
//!    are served before a marker being found again, so a link to it is not counted for them
//!    in the joint choice of 5.; the identity out of an overlap, `overlap_decision` under
//!    "Overlaps" below, counts it, as it counts a reference-frame link). With
//!    `rigid_min_frames` 0 no statistics are kept and this rule and the placement of 2. do
//!    nothing.
//! 5. **Joint choice** across markers and cameras
//!    ([`xma_vision::joint_tracking::choose_joint_overlapping`], with the run's overlaps
//!    below; [`xma_vision::joint_tracking::choose_joint_linked`] documents the rules).
//! 6. **Writing.** Accepted points are written with status `TRACKED` and their detected
//!    size, as one batch per frame (the caller brackets the run in one undo step, as for
//!    the existing kinds), and the 3D points are recomputed as after any point edit.
//!    Cells are written only where the existing kinds may write
//!    (`docs/cpp-tracking.md` 3.4): never over a point set by hand (`SET`, `MANUAL` and
//!    their optimised forms), nor over `UNTRACKABLE`, an interpolated point when
//!    `TrackInterpolatedPoints` is off, or a `TRACKED_AND_OPTIMIZED` point when
//!    `RetrackOptimizedTrackedPoints` is off. Points of those kinds that hold a position
//!    (set by hand, optimised) are observations the choice uses as they are. Every other
//!    cell at `e` of a marker the run has had, in a camera it tracks, that the run may
//!    write, holds a point and gets none from the run (an older point the run could not
//!    find again) is **left exactly as it is** (point, status, size) with `keep_unconfirmed`
//!    (the default since 24 November 2026), unless the run's own points of this step
//!    contradict it, when it is reset to undefined:
//!    (a) the run wrote a point of the same marker at `e` in at least one other camera it
//!    tracks, and the 3D point triangulated from the older point together with those written
//!    points (the run's own triangulation) has a largest reprojection error above
//!    `max_reprojection_cap`, the most the method accepts for its own points, or gives no
//!    answer at all (the older point cannot be undistorted, or the 3D point lies behind a
//!    camera: a point far outside the image, from an import or a hand-edited file; every
//!    point inside the pig's image triangulates, review 18). Only points written in this
//!    step count, not protected points; with no point of the marker written in another
//!    camera there is no evidence and (a) does not apply;
//!    (b) (`keep_dot_rule`, on by default) the run wrote a point of another marker in the
//!    same camera within `same_dot` of the older point, and does not hold the two as touching
//!    in that camera at this step (the overlap state's touching pairs given to the choice,
//!    "Overlaps" below; a pair held but not touching now is not exempt). Measured on the two
//!    Rat cases of `oracle/py/retrack.py` on 24 November 2026, and resting on those alone (on
//!    the pig and the guineafowl no point is removed with it on or off): backward over a
//!    forward run of this kind it removes 32 of the 3,359 older points the run could not find
//!    again, 29 within 2 px of the hand-finished points and none more than 5 px off (two markers that
//!    really share a dot, the other one in no camera of the run at that frame); over the plain
//!    tracker's result it removes 23, 20 more than 5 px off and 2 within 2 px, leaving 23
//!    points more than 5 px off against 42 without it. Exempting every held pair was measured
//!    and not kept (17 of the 32 still removed, 40 off over the plain tracker). On or off is a
//!    trade between the two cases and the author's choice (`MultiTrackParams::keep_dot_rule`);
//!    a cell whose stored coordinates are not finite is reset too (it cannot be compared).
//!    Without `keep_unconfirmed` every such cell is reset, as the existing kinds leave a cell
//!    whose tracked point failed (the rule until 24 November 2026: after a run, the frames it
//!    covered held its points, the protected points and gaps, never an older point that
//!    looks tracked; it erased good points of earlier runs, `MultiTrackParams::keep_unconfirmed`
//!    gives the measurements). Every cell the rule with `keep_unconfirmed` resets, the rule
//!    without it resets too, and the cells the run writes are the same with either (a kept
//!    point is never read, "What the run reads").
//!    The step's report counts the points written, in `left_empty` the cells at `e` of the
//!    markers the run has had, in the cameras it tracks, that hold no point after the step,
//!    not counting cells marked `UNTRACKABLE` (the run may never write there), in
//!    `kept_older` the older points left as they are and in `removed_older` the older points
//!    reset. A point written from a candidate fitted apart (`split_touching`) is written like
//!    any accepted point (status `TRACKED`) with size -1 (it does not enter the marker's mean
//!    size), counts as a merged point ("Overlaps" below; with `overlap_clean`, the default,
//!    no template, contrast or width is taken from it, and its frame's 3D point gives no
//!    depth to a lost marker, no rigid prediction and no rigid statistics), and is counted
//!    in `fitted_apart`.
//!
//! # Overlaps
//!
//! Two markers of a run whose dots meet in one camera (one bone passing in front of
//! another, as on the Rat) make one dark shape there, and the run can lose which is which.
//! The run keeps an **overlap state** per pair of its markers and camera (in its memory, not
//! in the document). A marker's position in a camera is its point at `s` (the run's own
//! record, refined as above), or else its prediction at `e` where it is followed (not being
//! found again). The pair enters the state when both have a position and the two lie within
//! touching distance, the two dot radii (each the marker's mean size, 2.5 px without one)
//! plus `overlap_margin`, or both are followed there and their predictions at `e` are; the
//! two then *touch*. The state is held, whatever either marker does in that camera (one may
//! lose its point), until both have points at `s` further apart than touching distance, or
//! for `overlap_max_frames` steps after they last touched. With it:
//!
//! - *Sharing* (`share_needs_overlap`): two markers share one dot in a camera
//!   (`share_radius`) only while they touch there now.
//! - *Merged points* (`overlap_clean`): a point accepted for a marker in a camera where it
//!   touches another is not a clean picture of its dot. No template or contrast is taken from
//!   it (the last clean one is kept, and does not age while the marker keeps such points: a
//!   template expires by the age of the marker's latest point in that camera, merged or
//!   not), and the 3D point of that frame gives no depth to a lost marker, no rigid
//!   prediction and no rigid statistics. With `overlap_take` off, no such point is taken at
//!   all (the choice's rule 4).
//! - *Identity out of an overlap* (`overlap_decision`): after the choice, every pair held
//!   is decided from the choice's own result (so the order of the pairs does not matter):
//!   the dots it gave them in that camera are checked against the other assignment (the two
//!   dots exchanged; or, when one of them holds a dot and the other none and they no longer
//!   touch, the dot given to the other). The assignment given is kept when the other costs at
//!   least `overlap_decision` more; within that margin neither gets a point there; when the
//!   other is that much cheaper it is taken where the markers can take it with exactly what
//!   they hold in the other cameras. The decisions are applied together, a marker asked two
//!   different things in one camera getting nothing there
//!   ([`xma_vision::joint_tracking::choose_joint_overlapping`], rule 4, gives the costs and
//!   the details).
//!
//! Two rules concern a marker being found again in a camera:
//!
//! - *Exclusive* (`reacquire_exclusive`): it never shares a dot with another marker.
//! - *Claims* (`claim_radius`, `claim_ratio`): it may not take a dot within `claim_radius` of
//!   another marker's prediction in that camera (a marker followed there, whether or not it
//!   is found at `e`), unless its own prediction is within `claim_ratio` times that distance
//!   of the dot.
//!
//! And one concerns where it is looked for (`ray_links`): in a camera that lost the marker
//! while another camera kept it, with no 3D track or rigid prediction, the search centre is
//! the point on the other camera's ray that best keeps its rigid links to markers followed in
//! 3D at the frame before
//! ([`xma_vision::joint_tracking::point_on_ray_keeping_links`], two partners or more, and
//! one place only on the ray that keeps them all), with the gate `gate_rigid`, instead of the
//! depth of its latest (clean) 3D point.
//!
//! Each rule can be turned off through its parameter. Writing a point "held" on the
//! epipolar line through an overlap (as merge-aware tracking, removed on 26 November 2026,
//! did) is not done: the cells
//! stay empty.
//!
//! Known limits (round 8, measured on the Rat): the rules know only the run's own
//! markers where the run has a position for them. A marker the run has lost in a camera
//! with no prediction there (no 3D track, no rigid prediction, no ray) is invisible to them,
//! and another marker can still walk onto its dot (most of the Rat's remaining points on
//! another marker's dot); and a marker that jumps after a one-frame gap, with the widened
//! gate, onto a dot that belongs to a marker the run does not follow is not caught.
//! Claims for followed markers too (not only for one being found again) were measured and
//! made the Rat's results worse in found and wrong points alike, and were not kept.
//!
//! Frames where a marker is not found get no point from the run (no interpolation): they stay
//! undefined, or keep an older point as step 6 says. Each step reports
//! the cells it wrote (`tracked`, `detected`) and, in `invalid` with a Console line each,
//! the cells left undefined because their dot was in doubt (another marker's, contested,
//! ambiguous, breaking rigid links, or an identity that cannot be confirmed).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use xma_model::{CalibrationType, CppCompat, DetectionMethod, MarkerStatus, Point2, Point3};
use xma_video::Frame;
use xma_vision::camera::RigidTransform;
use xma_vision::depth::on_8bit_scale;
use xma_vision::dot_fit::{
    FittedDot, Split, SplitRules, WithoutDot, fit_dot_shape, split_joined_dots,
};
use xma_vision::image::{GreySource, ImageF32};
use xma_vision::joint_tracking::{
    CameraOptions, Candidate, JointCamera, MarkerOptions, MultiTrackParams, Overlap, Pick, Refusal,
    RigidLink, choose_joint_overlapping, closest_point_on_ray, contrast_maxima, cut_ncc_template,
    dot_contrast, dot_contrast_relative, local_maxima, ncc_surface, on_one_shape,
};
use xma_vision::tracking::template_size;

use xma_vision::triangulation::{Observation, triangulate};

use super::jobs::{self, AlgorithmImage, DetectionJob};
use super::{Direction, StepReport, StopReason, TrackError, TrialImages, Write};
use crate::document::Document;
use crate::markers::FrameContext;

/// Which markers a multi-camera run tracks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MultiMarkers {
    /// Every marker of the trial (`xma track --kind multi`).
    All,
    /// The markers listed (`--kind multi-selected`).
    Selected(Vec<usize>),
}

/// A multi-camera tracking run's request. New in this program.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiRequest {
    pub trial: usize,
    pub markers: MultiMarkers,
    pub direction: Direction,
    /// False for one step, true for a continuous run.
    pub continuous: bool,
    /// The frame the run starts from, 0-based.
    pub start_frame: usize,
    /// A continuous run stops after the step that reaches this frame (0-based); `None` runs
    /// to the trial's first or last frame.
    pub end_frame: Option<usize>,
    /// Per camera, whether it is shown; hidden cameras are neither searched nor written. A
    /// camera beyond the list counts as shown.
    pub visible_cameras: Vec<bool>,
    /// The method's parameters.
    pub params: MultiTrackParams,
}

/// A template kept between steps.
#[derive(Clone, Debug)]
struct Template {
    /// The frame it was cut at.
    frame: usize,
    /// The latest frame at which the marker held a point in that camera (`frame`, or later
    /// when its points since were taken on a merged shape and gave no template,
    /// `overlap_clean`). A template expires by this frame's age, not by `frame`'s: a marker
    /// still followed keeps its template however long its points stay merged.
    held: usize,
    image: ImageF32,
}

/// A multi-camera tracking run in progress. New in this program.
#[derive(Clone, Debug)]
pub struct MultiTracker {
    request: MultiRequest,
    frame: usize,
    finished: Option<StopReason>,
    steps_done: usize,
    /// What the run remembers between steps (boxed: it is large).
    mem: Box<RunMemory>,
    /// The end frame's images of the last step, which are the next step's start images.
    kept: Vec<(usize, usize, Arc<Frame>, Arc<AlgorithmImage>)>,
}

/// What a run remembers between steps.
#[derive(Clone, Debug, Default)]
struct RunMemory {
    templates: HashMap<(usize, usize), Template>,
    /// Per (marker, camera) the contrast of its latest points and their contrast relative
    /// to the ring ([`dot_contrast_relative`]), newest last.
    contrast: HashMap<(usize, usize), std::collections::VecDeque<(f64, f64)>>,
    /// Per marker pair `(a, b)`, `a < b`: frames with both 3D points, and the mean and sum
    /// of squared deviations of their distance (Welford).
    pairs: HashMap<(usize, usize), (usize, f64, f64)>,
    /// The cells (marker, camera, frame) this run has written.
    written: HashSet<(usize, usize, usize)>,
    /// Per marker the sum and count of the sizes the run may read (its own mean size).
    sizes: HashMap<usize, (f64, usize)>,
    /// The run's own 3D points, kept between steps (`View::point3_err`).
    tri_cache: TriCache,
    /// The markers the run has had (module documentation, "Which markers are the run's").
    had: HashSet<usize>,
    /// Per (marker, camera, frame), the refined position of a point the run reads but did
    /// not detect itself ([`refine_stored_point`]), where it differs from the stored one.
    refined: HashMap<(usize, usize, usize), Point2>,
    /// The overlap state (module documentation, "Overlaps"): per (marker, marker, camera),
    /// the smaller marker first, the steps since the two last touched in that camera (1
    /// while they touch).
    overlaps: HashMap<(usize, usize, usize), usize>,
    /// The cells (marker, camera, frame) whose point the run accepted while the marker
    /// overlapped another in that camera.
    merged: HashSet<(usize, usize, usize)>,
    /// The (marker, frame)s of `merged`, with `overlap_clean` (empty without it).
    merged_frames: HashSet<(usize, usize)>,
    /// The pairs (marker, marker, camera), the smaller marker first, given to the latest
    /// step's choice as touching (for tests and diagnostics).
    touching: HashSet<(usize, usize, usize)>,
    /// Per (marker, camera) the dot widths learned at its latest clean points, newest last
    /// (`split_touching`; module documentation, step 1). Empty without it.
    widths: HashMap<(usize, usize), std::collections::VecDeque<f64>>,
}

/// An algorithm image read as grey values on the 8-bit scale.
fn with_grey<R>(image: &AlgorithmImage, f: impl FnOnce(&dyn GreySource) -> R) -> R {
    match image {
        AlgorithmImage::U8(im) => f(im),
        AlgorithmImage::Deep { image, bit_depth } => f(&on_8bit_scale(image, *bit_depth)),
    }
}

/// The text a refused multi-camera run gives in the XMALab 3 mode.
pub const REFUSED_XMALAB3: &str = "Multi-camera tracking is new in this program and XMALab 3 has \
     no such method; open the project without the XMALab 3 compatibility mode to use it.";

/// Per camera of the project, whether a multi-camera run on trial `trial` tracks it: shown
/// (per `visible`; a camera beyond the list counts as shown), calibrated, and with a pose
/// for the trial (a calibration image at the trial's reference calibration image). The one
/// definition of the cameras a run reads and writes ([`MultiTracker::step`]), counts before
/// it starts ([`MultiTracker::new`]) and the window asks about
/// ([`super::multi_unavailable`]). All false for a trial that does not exist.
#[must_use]
pub fn tracked_cameras(doc: &Document, trial: usize, visible: &[bool]) -> Vec<bool> {
    let cameras = &doc.project().cameras;
    let Some(t) = doc.project().trials.get(trial) else {
        return vec![false; cameras.len()];
    };
    cameras
        .iter()
        .enumerate()
        .map(|(c, cam)| {
            visible.get(c).copied().unwrap_or(true)
                && cam.calibrated
                && crate::document::reference_pose(cam, t.reference_calibration_image).is_some()
        })
        .collect()
}

/// The sentence for a project with fewer than two calibrated cameras (`n_cal` of them).
pub(crate) fn too_few_calibrated(n_cal: usize) -> String {
    format!(
        "Multi-camera tracking needs at least two calibrated cameras; this project has {n_cal}."
    )
}

/// The sentence for a run that would track fewer than two cameras because the others are
/// hidden: `shown` of the project's `n_cal` calibrated cameras are shown.
pub(crate) fn too_few_shown(shown: usize, n_cal: usize) -> String {
    let verb = if shown == 1 { "is" } else { "are" };
    format!(
        "Multi-camera tracking needs at least two calibrated cameras that are shown; {shown} of \
         this project's {n_cal} {verb} shown. Show at least two of them to use it."
    )
}

impl MultiTracker {
    /// Starts a run. Nothing is tracked until [`MultiTracker::step`].
    ///
    /// # Errors
    /// [`TrackError::Refused`] in the XMALab 3 mode, for a project without calibration or
    /// with fewer than two calibrated cameras, when fewer than two of them are shown (the run
    /// would track fewer than two cameras, [`tracked_cameras`]), and when no marker is given;
    /// [`TrackError::Invalid`] for an index or frame out of range.
    pub fn new(doc: &Document, mut request: MultiRequest) -> Result<MultiTracker, TrackError> {
        if doc.compat() == CppCompat::XMALAB3 {
            return Err(TrackError::Refused(REFUSED_XMALAB3.to_owned()));
        }
        let t = doc
            .project()
            .trials
            .get(request.trial)
            .ok_or_else(|| TrackError::Invalid(format!("no trial {}", request.trial)))?;
        if t.is_default {
            return Err(TrackError::Invalid(
                "the Default trial holds no tracking data".to_owned(),
            ));
        }
        if doc.project().calibration_type == CalibrationType::None {
            return Err(TrackError::Refused(
                "Multi-camera tracking needs a calibrated project.".to_owned(),
            ));
        }
        let count = |v: Vec<bool>| v.into_iter().filter(|&b| b).count();
        let n_cal = count(tracked_cameras(doc, request.trial, &[]));
        if n_cal < 2 {
            return Err(TrackError::Refused(too_few_calibrated(n_cal)));
        }
        let shown = count(tracked_cameras(
            doc,
            request.trial,
            &request.visible_cameras,
        ));
        if shown < 2 {
            return Err(TrackError::Refused(too_few_shown(shown, n_cal)));
        }
        let n_markers = t.markers.len();
        if let MultiMarkers::Selected(list) = &mut request.markers {
            if let Some(m) = list.iter().find(|&&m| m >= n_markers) {
                return Err(TrackError::Invalid(format!("no marker {m}")));
            }
            list.sort_unstable();
            list.dedup();
            if list.is_empty() {
                return Err(TrackError::Refused(
                    "Multi-camera tracking: no markers selected.".to_owned(),
                ));
            }
        } else if n_markers == 0 {
            return Err(TrackError::Refused(
                "Multi-camera tracking: the trial has no markers.".to_owned(),
            ));
        }
        let first = t.start_frame.saturating_sub(1);
        let last = t
            .end_frame
            .saturating_sub(1)
            .min(t.n_frames.saturating_sub(1));
        if request.start_frame < first || request.start_frame > last {
            return Err(TrackError::Invalid(format!(
                "frame {} is outside the trial's frames {} to {}",
                request.start_frame + 1,
                first + 1,
                last + 1
            )));
        }
        if let Some(end) = request.end_frame {
            let behind = match request.direction {
                Direction::Forward => end < request.start_frame,
                Direction::Backward => end > request.start_frame,
            };
            if behind {
                return Err(TrackError::Invalid(format!(
                    "end frame {} is behind the start frame {} in the run's direction",
                    end + 1,
                    request.start_frame + 1
                )));
            }
        }
        Ok(MultiTracker {
            frame: request.start_frame,
            request,
            finished: None,
            steps_done: 0,
            mem: Box::default(),
            kept: Vec::new(),
        })
    }

    /// The request the run was started with.
    #[must_use]
    pub fn request(&self) -> &MultiRequest {
        &self.request
    }

    /// The frame the run is at, 0-based.
    #[must_use]
    pub fn frame(&self) -> usize {
        self.frame
    }

    /// Why the run ended, once it has.
    #[must_use]
    pub fn finished(&self) -> Option<&StopReason> {
        self.finished.as_ref()
    }

    /// Whether markers `a` and `b` overlap in camera `c` in the run's overlap state (module
    /// documentation, "Overlaps") after the latest step. For tests and diagnostics.
    #[doc(hidden)]
    #[must_use]
    pub fn overlapping(&self, a: usize, b: usize, c: usize) -> bool {
        self.mem.overlaps.contains_key(&(a.min(b), a.max(b), c))
    }

    /// Whether markers `a` and `b` were given to the latest step's choice as touching in
    /// camera `c` (module documentation, "Overlaps"): the pairs whose older points the run
    /// does not hold against each other (`keep_unconfirmed`, rule (b)). For tests and
    /// diagnostics.
    #[doc(hidden)]
    #[must_use]
    pub fn touching(&self, a: usize, b: usize, c: usize) -> bool {
        self.mem.touching.contains(&(a.min(b), a.max(b), c))
    }

    /// The frame marker `m`'s template in camera `c` was cut at, when the run holds one.
    /// For tests and diagnostics.
    #[doc(hidden)]
    #[must_use]
    pub fn template_frame(&self, m: usize, c: usize) -> Option<usize> {
        self.mem.templates.get(&(m, c)).map(|t| t.frame)
    }

    /// Number of frame steps done so far.
    #[must_use]
    pub fn steps_done(&self) -> usize {
        self.steps_done
    }

    /// Ends the run before its next step.
    pub fn stop(&mut self) {
        if self.finished.is_none() {
            self.finished = Some(StopReason::Stopped);
        }
        self.kept.clear();
    }

    /// Whether marker `m` is one the run has had so far (module documentation, "Which
    /// markers are the run's"): only such a marker is searched for and can have its cells
    /// reset.
    #[must_use]
    pub fn has_had(&self, m: usize) -> bool {
        self.mem.had.contains(&m)
    }

    /// Whether the run's own record holds a point of marker `m` at frame `f` in a camera it
    /// tracks: a point at or behind its start frame (as the document holds it now), a point
    /// the run wrote, or a protected point (set by hand, and the others it may not
    /// overwrite). A point the run kept but never read (`MultiTrackParams::keep_unconfirmed`)
    /// is not in the record. Used to tell whether the run lost a marker.
    #[must_use]
    pub fn run_holds(&self, doc: &Document, m: usize, f: usize) -> bool {
        let Some(t) = doc.project().trials.get(self.request.trial) else {
            return false;
        };
        let Some(mk) = t.markers.get(m) else {
            return false;
        };
        let r_max = if doc.settings().retrack_optimized_tracked_points {
            MarkerStatus::TrackedAndOptimized
        } else {
            MarkerStatus::Tracked
        };
        let start = self.request.start_frame;
        let behind = if self.forward() {
            f <= start
        } else {
            f >= start
        };
        let usable = self.tracked_cameras(doc);
        mk.views.iter().enumerate().any(|(c, v)| {
            usable.get(c).copied().unwrap_or(false)
                && v.status.get(f).is_some_and(|st| {
                    st.has_data()
                        && (behind || self.mem.written.contains(&(m, c, f)) || *st > r_max)
                })
        })
    }

    /// Per camera, whether the run tracks it ([`tracked_cameras`] with the request's shown
    /// cameras): only these cameras are read and written.
    #[must_use]
    pub fn tracked_cameras(&self, doc: &Document) -> Vec<bool> {
        tracked_cameras(doc, self.request.trial, &self.request.visible_cameras)
    }

    fn forward(&self) -> bool {
        self.request.direction == Direction::Forward
    }

    fn at_boundary(&self, doc: &Document) -> bool {
        let t = doc.trial(self.request.trial);
        if self.forward() {
            // A trial whose end frame lies beyond its frames ends at its last frame.
            self.frame
                == t.end_frame
                    .saturating_sub(1)
                    .min(t.n_frames.saturating_sub(1))
        } else {
            self.frame == t.start_frame.saturating_sub(1)
        }
    }

    fn report(&self) -> StepReport {
        StepReport {
            frame: self.frame,
            finished: self.finished.clone(),
            messages: Vec::new(),
            tracked: Vec::new(),
            detected: Vec::new(),
            invalid: Vec::new(),
            left_empty: 0,
            kept_older: 0,
            removed_older: 0,
            fitted_apart: 0,
        }
    }

    /// One frame step (module documentation). After the run has ended this does nothing
    /// and reports the end.
    pub fn step(&mut self, doc: &mut Document, images: &mut dyn TrialImages) -> StepReport {
        if self.finished.is_some() {
            return self.report();
        }
        if self.steps_done == 0 && self.request.end_frame == Some(self.frame) {
            self.finished = Some(StopReason::EndFrame);
            return self.report();
        }
        if self.steps_done == 0 && self.at_boundary(doc) {
            self.finished = Some(StopReason::Boundary);
            return self.report();
        }
        let mut report = match self.do_step(doc, images) {
            Ok(r) => r,
            Err(msg) => {
                self.finished = Some(StopReason::Failed(msg));
                self.kept.clear();
                return self.report();
            }
        };
        if self.request.end_frame == Some(self.frame) {
            self.finished = Some(StopReason::EndFrame);
        } else if !self.request.continuous {
            self.finished = Some(StopReason::SingleStep);
        } else if self.at_boundary(doc) {
            self.finished = Some(StopReason::Boundary);
        }
        if self.finished.is_some() {
            self.kept.clear();
        }
        report.frame = self.frame;
        report.finished = self.finished.clone();
        report
    }

    /// Steps until the run ends, calling `progress` after every step but the last; the run
    /// stops when `progress` returns false. Returns the last report.
    pub fn run(
        &mut self,
        doc: &mut Document,
        images: &mut dyn TrialImages,
        progress: &mut dyn FnMut(&StepReport) -> bool,
    ) -> StepReport {
        loop {
            let report = self.step(doc, images);
            if report.finished.is_some() {
                return report;
            }
            if !progress(&report) {
                self.stop();
            }
        }
    }

    /// The frame `k` frames behind `frame` in the run's direction, within `0..n`.
    fn behind(&self, frame: usize, k: usize, n: usize) -> Option<usize> {
        behind(self.forward(), frame, k, n)
    }

    fn markers(&self, doc: &Document) -> Vec<usize> {
        match &self.request.markers {
            MultiMarkers::All => (0..doc.trial(self.request.trial).markers.len()).collect(),
            MultiMarkers::Selected(v) => v.clone(),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn do_step(
        &mut self,
        doc: &mut Document,
        images: &mut dyn TrialImages,
    ) -> Result<StepReport, String> {
        let kept = std::mem::take(&mut self.kept);
        let trial = self.request.trial;
        let compat = doc.compat();
        let s = self.frame;
        let n_frames = doc.trial(trial).n_frames;
        let e = if self.forward() {
            s + 1
        } else {
            s.checked_sub(1).ok_or("no frame before the first")?
        };
        if e >= n_frames {
            return Err(format!("frame {} is past the trial's frames", e + 1));
        }
        // The frame one step beyond `e` in the run's direction, when the trial has it: the
        // wider first search looks at its image (never at its points) to confirm a candidate.
        let e2 = {
            let t = doc.trial(trial);
            let first = t.start_frame.saturating_sub(1);
            let last = t
                .end_frame
                .saturating_sub(1)
                .min(t.n_frames.saturating_sub(1));
            if self.forward() {
                (e < last).then_some(e + 1)
            } else {
                (e > first).then(|| e - 1)
            }
        };
        let p = self.request.params;
        // The cross-camera check's limit on a refinement's reprojection error
        // (`refinement_agrees`). Not the wider `max_reprojection_cap`: measured on the
        // guineafowl with the start points moved inside their dark shapes (round 7c), the cap
        // kept a refinement of `TMTtop` that had gone onto the bar-shaped marker beside it
        // (error 1.78 px) and the run followed the bar for nine frames. This limit refuses
        // it, at the cost of the starts whose two cameras' centres disagree by 1.5 to 3 px
        // (13 of 231 refined marker-frames there).
        let refine_limit = p.max_reprojection_error;
        // Every count of frames is bounded by the trial.
        let max_gap = p.max_gap.min(n_frames);
        let depth_gap = p.depth_gap.min(n_frames);
        let velocity_window = p.velocity_window.clamp(1, n_frames.max(1));
        let rigid_history = p.rigid_history.min(n_frames);
        let markers = self.markers(doc);
        let n_cam = doc.project().cameras.len();
        let settings = doc.settings();
        let r_max = if settings.retrack_optimized_tracked_points {
            MarkerStatus::TrackedAndOptimized
        } else {
            MarkerStatus::Tracked
        };
        let track_interpolated = settings.track_interpolated_points;
        let writable = |status: MarkerStatus| {
            status > MarkerStatus::Untrackable
                && status <= r_max
                && !(status == MarkerStatus::Interpolated && !track_interpolated)
        };
        let ctx = doc.frame_context(trial);
        // The cameras the run tracks: shown, calibrated, with a pose. No other camera is
        // read or written.
        let usable: Vec<bool> = self.tracked_cameras(doc);
        let forward = self.forward();
        let merged_frames = std::mem::take(&mut self.mem.merged_frames);
        let view = View {
            t: doc.trial(trial),
            ctx: &ctx,
            usable: &usable,
            r_max,
            start: self.request.start_frame,
            forward,
            written: &self.mem.written,
            tri: RefCell::new(std::mem::take(&mut self.mem.tri_cache)),
            refined: RefCell::new(std::mem::take(&mut self.mem.refined)),
            merged: &merged_frames,
        };

        // The markers' mean sizes from the points the run may read (a size of 1 to 50, as
        // the document's mean size counts them), set up at the first step.
        if self.steps_done == 0 {
            for &m in &markers {
                let mk = &view.t.markers[m];
                let mut acc = (0.0, 0usize);
                for c in 0..n_cam {
                    for f in 0..mk.n_frames() {
                        if view.trusted(m, c, f) {
                            let z = mk.views[c].size[f];
                            if z > 1.0 && z < 50.0 {
                                acc.0 += z;
                                acc.1 += 1;
                            }
                        }
                    }
                }
                self.mem.sizes.insert(m, acc);
            }
        }
        let mean_size = |m: usize| -> f64 {
            match self.mem.sizes.get(&m) {
                Some(&(sum, n)) if n > 0 => sum / n as f64,
                _ => -1.0,
            }
        };

        // Images, converted once per (camera, frame).
        let mut cache: Vec<(usize, usize, Arc<Frame>, Arc<AlgorithmImage>)> = Vec::new();
        let mut load = |cache: &mut Vec<(usize, usize, Arc<Frame>, Arc<AlgorithmImage>)>,
                        c: usize,
                        f: usize|
         -> Result<Arc<AlgorithmImage>, String> {
            if let Some(k) = cache.iter().find(|k| (k.0, k.1) == (c, f)) {
                return Ok(Arc::clone(&k.3));
            }
            let source = images.image(c, f)?;
            let image = kept
                .iter()
                .find(|k| (k.0, k.1) == (c, f) && Arc::ptr_eq(&k.2, &source))
                .map_or_else(
                    || Arc::new(jobs::algorithm_image_for(&source, compat)),
                    |k| Arc::clone(&k.3),
                );
            cache.push((c, f, source, Arc::clone(&image)));
            Ok(image)
        };

        // Which markers are the run's: at the first step, every marker with a trusted point
        // at or behind the start; at every step, a marker whose protected point the run has
        // reached (at e). Points the run writes are of markers it already has.
        if self.steps_done == 0 {
            for &m in &markers {
                let n = view.t.markers[m].n_frames();
                let behind_start: Box<dyn Iterator<Item = usize>> = if forward {
                    Box::new(0..=s.min(n.saturating_sub(1)))
                } else {
                    Box::new(s..n)
                };
                let mut frames = behind_start;
                if frames.any(|f| (0..n_cam).any(|c| view.trusted(m, c, f))) {
                    self.mem.had.insert(m);
                }
            }
        }
        for &m in &markers {
            if (0..n_cam).any(|c| {
                usable[c] && {
                    let st = view.t.markers[m].views[c].status[e];
                    st.has_data() && st > r_max
                }
            }) {
                self.mem.had.insert(m);
            }
        }

        // The points the run reads but did not detect itself, at and behind the start (as far
        // back as a template, the velocity or a 2D prediction reaches: `max_gap` plus
        // `velocity_window` frames, and at most [`REFINE_REACH`]), are refined by the
        // marker's own detection ([`refine_stored_point`]); the run works from the refined
        // position. Older points (the rigid statistics' history, depth for a lost marker,
        // and points beyond `REFINE_REACH`) stay as stored.
        if self.steps_done == 0 {
            // Nothing has been triangulated yet at the first step, so no 3D point of the
            // run's cache rests on a stored point that is refined below.
            debug_assert!(view.tri.borrow().is_empty());
            let options_d = jobs::detect_options(doc);
            let reach = max_gap.saturating_add(velocity_window).min(REFINE_REACH);
            let shapes: Vec<(usize, f64, bool)> = markers
                .iter()
                .filter_map(|&m| {
                    let msize = mean_size(m);
                    contrast_shape(&view.t.markers[m], msize).map(|(_, dark)| (m, msize, dark))
                })
                .collect();
            for k in 0..=reach {
                let Some(f) = self.behind(s, k, n_frames) else {
                    break;
                };
                for c in 0..n_cam {
                    if !usable[c] {
                        continue;
                    }
                    let wanted =
                        |m: usize| view.trusted(m, c, f) && !self.mem.written.contains(&(m, c, f));
                    if !shapes.iter().any(|&(m, _, _)| wanted(m)) {
                        continue;
                    }
                    // The start frame's image is needed by the step anyway and is kept; an
                    // image behind the start is loaded for the refinement only and dropped
                    // after it, so the first step holds at most one such image at a time.
                    let mut only_here = Vec::new();
                    let held = if f == s { &mut cache } else { &mut only_here };
                    let Ok(image) = load(held, c, f) else {
                        continue;
                    };
                    for &(m, msize, dark) in &shapes {
                        if !wanted(m) {
                            continue;
                        }
                        let mk = &view.t.markers[m];
                        let stored = mk.views[c].points[f];
                        let others: Vec<Point2> = markers
                            .iter()
                            .filter(|&&o| o != m && view.trusted(o, c, f))
                            .map(|&o| view.t.markers[o].views[c].points[f])
                            .collect();
                        if let Some(q) = refine_stored_point(
                            mk,
                            (m, c, f),
                            msize,
                            &image,
                            stored,
                            &others,
                            dark,
                            &options_d,
                            compat.threshold_offset_wraps,
                            p.same_dot,
                        ) {
                            view.refined.borrow_mut().insert((m, c, f), q);
                        }
                    }
                }
            }
            for &(m, _, _) in &shapes {
                let mk = &view.t.markers[m];
                // The other cameras check the refinements: at each frame, the refined
                // positions must agree in 3D as well as the stored points do
                // ([`refinement_agrees`]); otherwise that frame's refinements are dropped.
                for k in 0..=reach {
                    let Some(f) = self.behind(s, k, n_frames) else {
                        break;
                    };
                    if !(0..n_cam).any(|c| view.refined.borrow().contains_key(&(m, c, f))) {
                        continue;
                    }
                    let stored: Vec<(usize, Point2)> = (0..n_cam)
                        .filter(|&c| view.trusted(m, c, f))
                        .map(|c| (c, mk.views[c].points[f]))
                        .collect();
                    let used: Vec<(usize, Point2)> = stored
                        .iter()
                        .map(|&(c, q)| {
                            (
                                c,
                                view.refined.borrow().get(&(m, c, f)).copied().unwrap_or(q),
                            )
                        })
                        .collect();
                    let e_stored = triangulate_points(&ctx, &stored).map(|x| x.1);
                    let e_used = triangulate_points(&ctx, &used).map(|x| x.1);
                    if !refinement_agrees(stored.len(), e_stored, e_used, refine_limit) {
                        for c in 0..n_cam {
                            view.refined.borrow_mut().remove(&(m, c, f));
                        }
                    }
                }
            }
        }

        // 1. Templates and contrast, from the points the run may read.
        let mut new_templates: Vec<((usize, usize), Template)> = Vec::new();
        // The cells whose point at `s` was taken on a merged shape: their template is kept and
        // counts as held there.
        let mut held_merged: Vec<((usize, usize), usize)> = Vec::new();
        let mut new_contrast: Vec<((usize, usize), (f64, f64))> = Vec::new();
        let mut new_widths: Vec<((usize, usize), f64)> = Vec::new();
        for &m in &markers {
            let mk = &view.t.markers[m];
            let msize = mean_size(m);
            let half = template_size(msize).saturating_add(3);
            for c in 0..n_cam {
                if !usable[c] {
                    continue;
                }
                let at = if view.obs(m, c, s).is_some() {
                    Some(s)
                } else if self.steps_done == 0 && !self.mem.templates.contains_key(&(m, c)) {
                    (1..=max_gap)
                        .map_while(|k| self.behind(s, k, n_frames))
                        .find(|&f| view.obs(m, c, f).is_some())
                } else {
                    None
                };
                let Some(f) = at else {
                    continue;
                };
                let Some(pt) = view.obs(m, c, f) else {
                    continue;
                };
                // A point accepted while the marker overlapped another in this camera is no
                // clean picture of its dot (`overlap_clean`): the template and contrast of
                // the last clean point are kept.
                if p.overlap_clean && self.mem.merged.contains(&(m, c, f)) {
                    held_merged.push(((m, c), f));
                    continue;
                }
                let image = load(&mut cache, c, f)?;
                if let Some(t) = with_grey(&image, |g| cut_ncc_template(g, pt.x, pt.y, half)) {
                    new_templates.push((
                        (m, c),
                        Template {
                            frame: f,
                            held: f,
                            image: t,
                        },
                    ));
                }
                if let Some((radius, dark)) = contrast_shape(mk, msize)
                    && let Some(k) =
                        with_grey(&image, |g| dot_contrast(g, (pt.x, pt.y), radius, dark))
                    && let Some(r) = with_grey(&image, |g| {
                        dot_contrast_relative(g, (pt.x, pt.y), radius, dark)
                    })
                {
                    new_contrast.push(((m, c), (k, r)));
                }
                // The dot's width at the same cells (`split_touching`).
                if p.split_touching
                    && let Some((radius, dark)) = contrast_shape(mk, msize)
                    && let Some(wd) =
                        with_grey(&image, |g| learn_width(g, pt, radius, dark, p.same_dot))
                {
                    new_widths.push(((m, c), wd));
                }
            }
        }
        for (key, t) in new_templates {
            self.mem.templates.insert(key, t);
        }
        for (key, k) in new_contrast {
            let h = self.mem.contrast.entry(key).or_default();
            h.push_back(k);
            while h.len() > p.contrast_history.clamp(1, 1000) {
                h.pop_front();
            }
        }
        for (key, wd) in new_widths {
            let h = self.mem.widths.entry(key).or_default();
            h.push_back(wd);
            while h.len() > p.contrast_history.clamp(1, 1000) {
                h.pop_front();
            }
        }
        for (key, f) in held_merged {
            if let Some(t) = self.mem.templates.get_mut(&key) {
                t.held = if forward {
                    t.held.max(f)
                } else {
                    t.held.min(f)
                };
            }
        }
        // Templates of markers that have gone are dropped: one whose marker has held no point
        // in that camera for more than `max(max_gap, depth_gap) + 1` frames.
        let gap = max_gap.max(depth_gap);
        self.mem
            .templates
            .retain(|_, t| t.held.abs_diff(e) <= gap.saturating_add(1));

        // Rigid links between the markers, from the run's own 3D positions.
        if self.steps_done == 0 && p.rigid_min_frames > 0 {
            for k in 0..=rigid_history {
                let Some(f) = self.behind(s, k, n_frames) else {
                    break;
                };
                add_pairs(&mut self.mem.pairs, &view, &markers, f);
            }
        }
        let mut links: Vec<RigidLink> = Vec::new();
        let mut extra_links: Vec<RigidLink> = Vec::new();
        if p.rigid_min_frames > 0 {
            for (i, &a) in markers.iter().enumerate() {
                for (j, &b) in markers.iter().enumerate().skip(i + 1) {
                    let Some(&(n, mean, m2)) = self.mem.pairs.get(&(a.min(b), a.max(b))) else {
                        continue;
                    };
                    if n < p.rigid_min_frames || n < 2 || mean.is_nan() || mean <= 0.0 {
                        continue;
                    }
                    let sd = (m2 / (n - 1) as f64).sqrt();
                    if sd <= p.rigid_max_spread * mean {
                        links.push(RigidLink {
                            a: i,
                            b: j,
                            distance: mean,
                            tolerance: (p.rigid_sigmas * sd).max(p.rigid_min_tolerance * mean),
                        });
                    }
                }
            }
        }

        // 2. Predictions and the cells' roles.
        struct Cell {
            marker: usize,
            camera: usize,
            prediction: Point2,
            gate: f64,
            reacquire: bool,
            /// The wider first search (`gate_start`) of a marker followed without a known
            /// velocity: the wide radius, the marker's earlier point in this camera, the
            /// frames between it and `e`, and the frame of that point.
            wide: Option<(f64, Point2, usize, usize)>,
            /// With `contrast_relative`, the other markers predicted in this camera at `e`,
            /// with their dot radii (`DotSearch::neighbours`).
            neighbours: Vec<(Point2, f64)>,
        }
        // Per marker its 3D track at e: (position, frames since its latest 3D point, whether
        // a velocity was known).
        let pred3s: Vec<Option<(Point3, usize, bool)>> = markers
            .iter()
            .map(|&m| {
                let (k, x1) = (1..=max_gap)
                    .map_while(|k| self.behind(e, k, n_frames).map(|f| (k, f)))
                    .find_map(|(k, f)| view.point3(m, f).map(|x| (k, x)))?;
                let f1 = self.behind(e, k, n_frames)?;
                let v = (1..=velocity_window)
                    .map_while(|j| self.behind(f1, j, n_frames).map(|f| (j, f)))
                    .find_map(|(j, f2)| view.point3(m, f2).map(|x2| (x1 - x2) / j as f64));
                let steps = k.min(p.velocity_frames) as f64;
                Some((
                    x1 + v.unwrap_or_else(nalgebra::Vector3::zeros) * steps,
                    k,
                    v.is_some(),
                ))
            })
            .collect();
        let mut options: Vec<MarkerOptions> = Vec::with_capacity(markers.len());
        let mut cells: Vec<Cell> = Vec::new();
        for (ki, &m) in markers.iter().enumerate() {
            let mk = &view.t.markers[m];
            let pred3 = pred3s[ki];
            // A marker the run never had is neither searched for nor touched.
            if !self.mem.had.contains(&m) {
                options.push(MarkerOptions {
                    cameras: vec![CameraOptions::default(); n_cam],
                    reprojection_limit: None,
                });
                continue;
            }
            let mut mo = MarkerOptions {
                cameras: vec![CameraOptions::default(); n_cam],
                reprojection_limit: reprojection_limit(&view, m, e, n_frames, &p),
            };
            // Per camera: whether it may be written, and its own 2D prediction (position,
            // frames since, velocity known).
            let mut pred2: Vec<Option<(Point2, usize, bool)>> = vec![None; n_cam];
            let mut open: Vec<bool> = vec![false; n_cam];
            // Frames since each camera last had the marker, within `depth_gap`.
            let mut last_seen: Vec<Option<usize>> = vec![None; n_cam];
            for c in 0..n_cam {
                if !usable[c] {
                    continue;
                }
                let v = &mk.views[c];
                let status = v.status[e];
                if !writable(status) {
                    // A protected point is an observation the choice uses as it is.
                    if status > r_max {
                        mo.cameras[c].fixed = Some(v.points[e]);
                    }
                    continue;
                }
                open[c] = true;
                last_seen[c] = (1..=max_gap.max(depth_gap))
                    .map_while(|k| self.behind(e, k, n_frames).map(|f| (k, f)))
                    .find(|&(_, f)| view.obs(m, c, f).is_some())
                    .map(|(k, _)| k);
                pred2[c] = last_seen[c].filter(|&k| k <= max_gap).and_then(|k| {
                    let f1 = self.behind(e, k, n_frames)?;
                    let p1 = view.obs(m, c, f1)?;
                    let vel = (1..=velocity_window)
                        .map_while(|j| self.behind(f1, j, n_frames).map(|f| (j, f)))
                        .find_map(|(j, f2)| view.obs(m, c, f2).map(|q| (p1 - q) / j as f64));
                    Some((
                        p1 + vel.unwrap_or_else(nalgebra::Vector2::zeros)
                            * k.min(p.velocity_frames) as f64,
                        k,
                        vel.is_some(),
                    ))
                });
            }
            // Whether a camera may follow the marker on its own: a 3D point of the marker
            // within `single_max_frames` frames, or the run started at most that many frames
            // ago (a marker with a point in one camera only at the start).
            let limit = p.single_max_frames.min(n_frames);
            let single_allowed = p.single_max_frames == 0
                || self.request.start_frame.abs_diff(e) <= limit
                || (1..=limit)
                    .map_while(|k| self.behind(e, k, n_frames))
                    .any(|f| view.point3(m, f).is_some());
            let single_ok: Vec<bool> = vec![single_allowed; n_cam];
            // Cameras whose own track is more recent than the 3D track.
            let fresh: Vec<usize> = (0..n_cam)
                .filter(|&c| match (pred2[c], pred3) {
                    (Some((_, a, _)), Some((_, b, _))) => a < b,
                    (Some(_), None) => true,
                    _ => false,
                })
                .collect();
            // Without a 3D track, the marker's position predicted from rigid neighbours.
            let from_links: Option<Point3> = if pred3.is_none() {
                rigid_prediction(
                    &view,
                    &markers,
                    ki,
                    &pred3s,
                    e,
                    n_frames,
                    depth_gap,
                    &p,
                    &self.mem.pairs,
                )
                .map(|(x, ls)| {
                    extra_links.extend(ls);
                    x
                })
            } else {
                None
            };
            // `distance_placement`: without a rigid prediction (and with no camera that
            // follows the marker on its own), the point that best keeps its mean distances
            // to the markers followed in 3D; it is searched for as a rigid prediction is, and
            // recognised by the statistics links to the partners it was placed from.
            let placed: Option<(Point3, Vec<usize>)> = if pred3.is_none()
                && from_links.is_none()
                && fresh.is_empty()
                && p.distance_placement
            {
                place_by_distances(
                    &view,
                    &markers,
                    ki,
                    &pred3s,
                    e,
                    n_frames,
                    depth_gap,
                    &p,
                    &self.mem.pairs,
                )
            } else {
                None
            };
            let from_links: Option<Point3> = from_links.or(placed.as_ref().map(|x| x.0));
            // The 3D point the other cameras are searched round: the 3D track; else the
            // rigid prediction; else the latest 3D point within `depth_gap` (only as a depth
            // on a fresh camera's ray).
            let stale: Option<Point3> = if pred3.is_none() && from_links.is_none() {
                (1..=depth_gap)
                    .map_while(|k| self.behind(e, k, n_frames))
                    .find_map(|f| view.point3_clean(m, f))
            } else {
                None
            };
            let anchor: Option<Point3> = pred3.map(|q| q.0).or(from_links).or(stale);
            // With `ray_links`, instead of the old depth: the point on the fresh camera's
            // ray that best keeps the marker's rigid links to markers followed at the frame
            // before.
            let ray_linked: Option<Point3> = match (fresh.first(), stale) {
                (Some(&a), Some(x)) if p.ray_links => {
                    let partners: Vec<(Point3, f64, f64)> = links
                        .iter()
                        .filter_map(|l| {
                            let q = if l.a == ki {
                                l.b
                            } else if l.b == ki {
                                l.a
                            } else {
                                return None;
                            };
                            let (y, k, _) = pred3s.get(q).copied().flatten()?;
                            (k == 1).then_some((y, l.distance, l.tolerance))
                        })
                        .collect();
                    ctx.cameras[a].pose.as_ref().and_then(|pose| {
                        xma_vision::joint_tracking::point_on_ray_keeping_links(
                            &ctx.cameras[a].model,
                            pose,
                            pred2[a]?.0,
                            x,
                            &partners,
                        )
                    })
                }
                _ => None,
            };
            let centre3: Option<Point3> = match (fresh.first(), anchor) {
                _ if ray_linked.is_some() => ray_linked,
                (Some(&a), Some(x)) => ctx.cameras[a].pose.as_ref().and_then(|pose| {
                    closest_point_on_ray(&ctx.cameras[a].model, pose, pred2[a]?.0, x)
                }),
                (None, Some(x)) if stale.is_none() => Some(x),
                _ => None,
            };
            for c in 0..n_cam {
                if !open[c] {
                    continue;
                }
                // `start`: followed without a known velocity (the wider first search may
                // apply).
                let (prediction, gate, reacquire, start) = if fresh.contains(&c) {
                    match pred2[c] {
                        // Seen by this camera alone for longer than `single_max_frames`:
                        // the marker must prove its identity again.
                        Some((q, k, vel)) => (q, p.gate_for(k, vel), !single_ok[c], !vel),
                        None => continue,
                    }
                } else if let (Some(x), Some(pose)) = (centre3, ctx.cameras[c].pose.as_ref()) {
                    let q = ctx.cameras[c].model.project_point(pose, x);
                    match (pred3, from_links) {
                        // Followed, without a known 3D velocity (no 3D point within
                        // `velocity_window` before the latest, as after another camera's gap),
                        // in a camera whose own 2D track has a velocity (and, not being
                        // fresh, is as recent as the 3D track): its own 2D prediction, with
                        // the gate of a known velocity.
                        (Some((_, k, false)), _)
                            if last_seen[c].is_some_and(|j| j <= k)
                                && pred2[c].is_some_and(|(_, _, v)| v) =>
                        {
                            let (q2, k2, _) = pred2[c].expect("checked");
                            (q2, p.gate_for(k2, true), false, false)
                        }
                        // Followed: the 3D track's age and velocity.
                        (Some((_, k, vel)), _) if last_seen[c].is_some_and(|j| j <= k) => {
                            (q, p.gate_for(k, vel), false, !vel)
                        }
                        // Found again from the rigid neighbours.
                        (None, Some(_)) => (q, p.gate_rigid, true, false),
                        // Found again in this camera, round a point whose depth is old:
                        // only within the camera's own short gap is it followed.
                        _ => {
                            let k = last_seen[c].unwrap_or(usize::MAX);
                            if k <= max_gap {
                                (q, p.gate_for(k, false), false, false)
                            } else if ray_linked.is_some() {
                                (q, p.gate_rigid, true, false)
                            } else {
                                (q, p.gate_max, true, false)
                            }
                        }
                    }
                } else {
                    continue;
                };
                if !(prediction.x.is_finite() && prediction.y.is_finite()) {
                    continue;
                }
                let o = &mut mo.cameras[c];
                o.prediction = Some(prediction);
                o.gate = gate;
                o.reacquire = reacquire;
                // The wider first search: a marker followed without a known velocity, with its
                // earlier point in this camera (what a candidate's velocity is measured from),
                // when `gate_start` is above the ordinary gate and the trial has a frame
                // beyond `e`.
                // Not a finite positive number: off.
                let wide_radius = if p.gate_start.is_finite() && p.gate_start > 0.0 {
                    p.gate_start.min(MAX_SEARCH_RADIUS)
                } else {
                    0.0
                };
                let wide = if start && !reacquire && wide_radius > gate && e2.is_some() {
                    last_seen[c].filter(|&j| j > 0).and_then(|j| {
                        let f1 = self.behind(e, j, n_frames)?;
                        Some((wide_radius, view.obs(m, c, f1)?, j, f1))
                    })
                } else {
                    None
                };
                if self.mem.templates.contains_key(&(m, c)) {
                    cells.push(Cell {
                        marker: m,
                        camera: c,
                        prediction,
                        gate,
                        reacquire,
                        wide,
                        neighbours: Vec::new(),
                    });
                }
            }
            // The statistics links (`stat_links`) of a marker being found again; and, from a
            // placement (`distance_placement`), to the partners it was placed from.
            let finding_again = mo.cameras.iter().any(|o| o.reacquire);
            let mut stat_to: Vec<bool> = vec![p.stat_links && finding_again; markers.len()];
            if let Some((_, used)) = &placed {
                for &j in used {
                    stat_to[j] = true;
                }
            }
            for j in (0..markers.len()).filter(|&j| j != ki && stat_to[j]) {
                if let Some((mean, tolerance, rigid)) =
                    pair_statistics(&self.mem.pairs, markers[ki], markers[j], &p)
                    && !rigid
                {
                    // A rigid pair has its learned link already (the same numbers).
                    extra_links.push(RigidLink {
                        a: ki,
                        b: j,
                        distance: mean,
                        tolerance,
                    });
                }
            }
            options.push(mo);
        }

        // 3. Candidates: correlation maxima, refined by detection.
        let mut end_images: Vec<Option<Arc<AlgorithmImage>>> = vec![None; n_cam];
        for c in 0..n_cam {
            if usable[c] && cells.iter().any(|k| k.camera == c) {
                end_images[c] = Some(load(&mut cache, c, e)?);
            }
        }
        // The images of the frame beyond `e`, for the wider first search's confirmation (its
        // image only; an image that cannot be read confirms nothing).
        let mut next_images: Vec<Option<Arc<AlgorithmImage>>> = vec![None; n_cam];
        if let Some(e2) = e2 {
            for c in 0..n_cam {
                if usable[c] && cells.iter().any(|k| k.camera == c && k.wide.is_some()) {
                    next_images[c] = load(&mut cache, c, e2).ok();
                }
            }
        }
        // The images of the frames of the markers' earlier points, for the wider first
        // search's test that a far candidate was not there already (its image only).
        let mut earlier_images: HashMap<(usize, usize), Arc<AlgorithmImage>> = HashMap::new();
        for cell in &cells {
            if let Some((_, _, _, f1)) = cell.wide
                && !earlier_images.contains_key(&(cell.camera, f1))
                && let Ok(im) = load(&mut cache, cell.camera, f1)
            {
                earlier_images.insert((cell.camera, f1), im);
            }
        }
        let options_d = jobs::detect_options(doc);
        let wraps = compat.threshold_offset_wraps;
        // Per (marker, camera) the medians of its recent contrasts and relative contrasts.
        let medians: HashMap<(usize, usize), (f64, f64)> = self
            .mem
            .contrast
            .iter()
            .filter(|(_, h)| !h.is_empty())
            .map(|(&k, h)| {
                let median = |f: fn(&(f64, f64)) -> f64| {
                    let mut v: Vec<f64> = h.iter().map(f).collect();
                    v.sort_by(f64::total_cmp);
                    v[v.len() / 2]
                };
                (k, (median(|x| x.0), median(|x| x.1)))
            })
            .collect();
        // Per (marker, camera) its dot width (`split_touching`): the median of its learned
        // widths; none without the parameter or before one is learned.
        let widths: HashMap<(usize, usize), f64> = self
            .mem
            .widths
            .iter()
            .filter(|(_, h)| p.split_touching && !h.is_empty())
            .map(|(&k, h)| {
                let mut v: Vec<f64> = h.iter().copied().collect();
                v.sort_by(f64::total_cmp);
                (k, v[v.len() / 2])
            })
            .collect();
        let msizes: HashMap<usize, f64> = markers.iter().map(|&m| (m, mean_size(m))).collect();
        // With `contrast_relative`, per cell the other markers predicted in its camera at
        // `e`, with their dot radii: a candidate whose ring one of them may lie in is tested
        // by the absolute contrast (`DotSearch::refine`).
        let neighbours: Vec<Vec<(Point2, f64)>> = cells
            .iter()
            .map(|cell| {
                if !p.contrast_relative {
                    return Vec::new();
                }
                markers
                    .iter()
                    .zip(&options)
                    .filter(|(m, _)| **m != cell.marker)
                    .filter_map(|(&m, o)| {
                        let q = o.cameras.get(cell.camera)?.prediction?;
                        let r = dot_radius(msizes.get(&m).copied().unwrap_or(-1.0));
                        (q.x.is_finite() && q.y.is_finite()).then_some((q, r))
                    })
                    .collect()
            })
            .collect();
        for (cell, n) in cells.iter_mut().zip(neighbours) {
            cell.neighbours = n;
        }
        // Per cell its candidates (with their detected sizes), the gate the choice uses and
        // the near gate (`CameraOptions::near_gate`; 0 for none).
        let found: Vec<CellFound> = {
            let earlier_images = &earlier_images;
            let t = view.t;
            let medians = &medians;
            let widths = &widths;
            let msizes = &msizes;
            let templates = &self.mem.templates;
            let end_images = &end_images;
            let next_images = &next_images;
            let options_d = &options_d;
            jobs::par_map(&cells, |cell| {
                let none = (Vec::new(), cell.gate, 0.0);
                let Some(image) = end_images[cell.camera].as_ref() else {
                    return none;
                };
                let Some(tpl) = templates.get(&(cell.marker, cell.camera)) else {
                    return none;
                };
                let mk = &t.markers[cell.marker];
                let msize = msizes.get(&cell.marker).copied().unwrap_or(-1.0);
                let search = DotSearch {
                    mk,
                    marker: cell.marker,
                    camera: cell.camera,
                    frame: e,
                    msize,
                    shape: contrast_shape(mk, msize),
                    median: medians.get(&(cell.marker, cell.camera)).map(|m| m.0),
                    median_relative: medians.get(&(cell.marker, cell.camera)).map(|m| m.1),
                    neighbours: &cell.neighbours,
                    options_d,
                    wraps,
                    p: &p,
                    split: widths.get(&(cell.marker, cell.camera)).copied(),
                    rules: p.split_rules(),
                    splits: RefCell::new(Vec::new()),
                };
                let cap = p.max_candidates.clamp(1, 16);
                let near = search.candidates(
                    image,
                    &tpl.image,
                    cell.prediction,
                    cell.gate + p.search_margin,
                    cell.reacquire,
                    cap,
                    0.0,
                );
                let Some((wide, earlier, frames, f1)) = cell.wide else {
                    return (near, cell.gate, 0.0);
                };
                // The wider first search: a candidate beyond the ordinary gate is kept only
                // when the image of the marker's earlier frame shows no dot there already
                // (`DotSearch::there_before`) and the frame beyond `e` confirms it
                // (`DotSearch::confirmed`).
                let next = next_images[cell.camera].as_ref();
                let before = earlier_images.get(&(cell.camera, f1));
                let confirm = |pt: Point2| {
                    before.is_some_and(|b| !search.there_before(image, b, pt, earlier))
                        && next.is_some_and(|im| search.confirmed(image, im, pt, earlier, frames))
                };
                let far: Vec<Found> = search
                    .candidates(
                        image,
                        &tpl.image,
                        cell.prediction,
                        wide + p.search_margin,
                        false,
                        cap,
                        cell.gate + p.search_margin,
                    )
                    .into_iter()
                    .filter(|k| (k.cand.point - cell.prediction).norm() > cell.gate)
                    .filter(|k| {
                        !near
                            .iter()
                            .any(|q| (q.cand.point - k.cand.point).norm() <= p.same_dot)
                    })
                    .filter(|k| confirm(k.cand.point))
                    .collect();
                if far.is_empty() {
                    // Nothing confirmed beyond the ordinary gate: the cell is as without the
                    // wider search.
                    return (near, cell.gate, 0.0);
                }
                // With the wide gate, a candidate of the ordinary search beyond the ordinary
                // gate needs the confirmation too (and, with `start_confirm_near`, every one).
                let mut out: Vec<Found> = near
                    .into_iter()
                    .filter(|k| {
                        let beyond = (k.cand.point - cell.prediction).norm() > cell.gate;
                        !(beyond || p.start_confirm_near) || confirm(k.cand.point)
                    })
                    .collect();
                // Confirmed candidates fill the places the ordinary ones leave.
                for k in far {
                    if out.len() >= cap {
                        break;
                    }
                    out.push(k);
                }
                (out, wide, cell.gate)
            })
        };
        // Per cell its candidates' detected sizes and whether each was fitted apart.
        let mut sizes: HashMap<(usize, usize), Vec<(f64, bool)>> = HashMap::new();
        for (cell, (list, gate, near_gate)) in cells.iter().zip(found) {
            let k = markers
                .iter()
                .position(|&m| m == cell.marker)
                .expect("listed");
            options[k].cameras[cell.camera].gate = gate;
            options[k].cameras[cell.camera].near_gate = near_gate;
            options[k].cameras[cell.camera].candidates = list.iter().map(|x| x.cand).collect();
            sizes.insert(
                (cell.marker, cell.camera),
                list.iter().map(|x| (x.size, x.apart)).collect(),
            );
        }

        // The overlap state (module documentation, "Overlaps"), per pair of the run's
        // markers and camera. A marker's position in a camera: its point at `s`, else its
        // prediction at `e` when it is followed there (not being found again). The two touch
        // when both have a position and the two are within touching distance (the two dot
        // radii plus `overlap_margin`), or both are followed there and their predictions at
        // `e` are. Entered when they touch; held while they touch and, after that, whatever
        // either marker does in that camera, until both have points at `s` further apart than
        // touching distance, or for `overlap_max_frames` steps after they last touched.
        let mut overlaps: Vec<Overlap> = Vec::new();
        {
            let margin = if p.overlap_margin.is_finite() {
                p.overlap_margin
            } else {
                0.0
            };
            let mut next: HashMap<(usize, usize, usize), usize> = HashMap::new();
            for c in 0..n_cam {
                if !usable[c] {
                    continue;
                }
                // (index, marker, point at s, followed prediction at e)
                let had: Vec<(usize, usize, Option<Point2>, Option<Point2>)> = markers
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| self.mem.had.contains(m))
                    .map(|(ki, &m)| {
                        let q = options[ki].cameras.get(c).and_then(|o| {
                            let q = o.prediction?;
                            (!o.reacquire && q.x.is_finite() && q.y.is_finite()).then_some(q)
                        });
                        (ki, m, view.obs(m, c, s), q)
                    })
                    .collect();
                for (i, &(ka, a, sa, qa)) in had.iter().enumerate() {
                    for &(kb, b, sb, qb) in &had[i + 1..] {
                        let touching = dot_radius(mean_size(a)) + dot_radius(mean_size(b)) + margin;
                        let key = (a.min(b), a.max(b), c);
                        let within = |x: Option<Point2>, y: Option<Point2>| {
                            x.zip(y).is_some_and(|(x, y)| (x - y).norm() <= touching)
                        };
                        let now = within(sa.or(qa), sb.or(qb)) || within(qa, qb);
                        let keep = match self.mem.overlaps.get(&key).copied() {
                            // Steps since they last touched.
                            Some(_) if now => Some(1),
                            Some(n) => {
                                let apart =
                                    sa.zip(sb).is_some_and(|(x, y)| (x - y).norm() > touching);
                                (!apart && n <= p.overlap_max_frames).then_some(n + 1)
                            }
                            None => now.then_some(1),
                        };
                        if let Some(n) = keep {
                            next.insert(key, n);
                            overlaps.push(Overlap {
                                a: ka,
                                b: kb,
                                camera: c,
                                touching: now,
                            });
                        }
                    }
                }
            }
            self.mem.overlaps = next;
            self.mem.touching = overlaps
                .iter()
                .filter(|o| o.touching)
                .map(|o| {
                    let (a, b) = (markers[o.a], markers[o.b]);
                    (a.min(b), a.max(b), o.camera)
                })
                .collect();
        }

        // 4. The joint choice.
        let identity = RigidTransform {
            rotation: nalgebra::Vector3::zeros(),
            translation: nalgebra::Vector3::new(0.0, 0.0, 1.0),
        };
        let jcams: Vec<JointCamera<'_>> = ctx
            .cameras
            .iter()
            .map(|fc| JointCamera {
                model: &fc.model,
                pose: fc.pose.as_ref().unwrap_or(&identity),
                projection: fc.projection.unwrap_or_else(nalgebra::Matrix3x4::zeros),
            })
            .collect();
        // A rigid prediction's link may name a pair that also has a learned link: the choice
        // counts the partner once, kept only when both hold.
        links.extend(extra_links);
        let choices = choose_joint_overlapping(&jcams, &options, &links, &overlaps, ctx.method, &p);
        let mut tri = view.tri.into_inner();
        self.mem.refined = view.refined.into_inner();

        // 5. Writing: the accepted points; every other cell at e of a marker the run has had,
        // in a tracked camera, that the run may write and that holds a point is either left as
        // it is (`keep_unconfirmed`) or reset (module documentation, step 6).
        let mut writes: Vec<Write> = Vec::new();
        // (index in `markers`, marker, camera) of the cells holding an older point the run
        // gave nothing; decided once every write of the step is known.
        let mut unconfirmed: Vec<(usize, usize, usize)> = Vec::new();
        let mut refused: Vec<(usize, usize)> = Vec::new();
        let mut messages: Vec<String> = Vec::new();
        // The cells written from a candidate fitted apart (`split_touching`).
        let mut apart: Vec<(usize, usize)> = Vec::new();
        for (k, choice) in choices.iter().enumerate() {
            let m = markers[k];
            for c in 0..n_cam {
                if !usable[c] {
                    continue;
                }
                if let Pick::Candidate(i) = choice.picks[c] {
                    let cand = options[k].cameras[c].candidates[i];
                    let found = sizes.get(&(m, c)).and_then(|v| v.get(i)).copied();
                    let size = found.map(|x| x.0);
                    if found.is_some_and(|x| x.1) {
                        apart.push((m, c));
                    }
                    writes.push(Write {
                        marker: m,
                        camera: c,
                        point: cand.point,
                        status: MarkerStatus::Tracked,
                        size,
                    });
                    continue;
                }
                let status = doc.trial(trial).markers[m].views[c].status[e];
                if writable(status) && status.has_data() && self.mem.had.contains(&m) {
                    unconfirmed.push((k, m, c));
                }
                let why = match choice.refusals[c] {
                    Some(Refusal::TakenByAnother) => "its dot went to another marker",
                    Some(Refusal::Ambiguous) => "two dots fit about as well",
                    Some(Refusal::Contested) => "another marker is predicted on the same dot",
                    Some(Refusal::Rigid) => {
                        "it would not keep its distance to a marker on the same bone"
                    }
                    Some(Refusal::Identity) => {
                        "it is being found again and its distances to its neighbours cannot \
                         confirm which marker it is"
                    }
                    Some(Refusal::Claimed) => {
                        "it is being found again and its dot is where another marker is expected"
                    }
                    Some(Refusal::Overlap) => {
                        "it overlaps another marker and which dot is which cannot be told"
                    }
                    Some(Refusal::NotFound) | None => continue,
                };
                refused.push((m, c));
                messages.push(format!(
                    "marker {} camera {} frame {}: {why}",
                    m + 1,
                    c + 1,
                    e + 1
                ));
            }
        }
        let mut resets: Vec<Write> = Vec::new();
        let mut kept_older = 0usize;
        for &(k, m, c) in &unconfirmed {
            let old = doc.trial(trial).markers[m].views[c].points[e];
            let reset = !p.keep_unconfirmed
                || contradicts(&ctx, &writes, &overlaps, &markers, (k, m, c), old, &p);
            if reset {
                resets.push(Write {
                    marker: m,
                    camera: c,
                    point: Point2::new(xma_model::marker::UNSET_2D, xma_model::marker::UNSET_2D),
                    status: MarkerStatus::Undefined,
                    size: Some(xma_model::marker::UNSET_SIZE),
                });
            } else {
                kept_older += 1;
            }
        }
        let removed_older = resets.len();
        let all: Vec<Write> = writes.iter().chain(&resets).copied().collect();
        doc.write_results(trial, e, &all)
            .map_err(|err| err.to_string())?;
        // A protected point the run has reached (used as it is in the choice at `e`) is
        // refined for what the run carries forward from it.
        {
            let options_d = jobs::detect_options(doc);
            let t = doc.trial(trial);
            for &m in &markers {
                if !self.mem.had.contains(&m) {
                    continue;
                }
                let mk = &t.markers[m];
                let msize = mean_size(m);
                let Some((_, dark)) = contrast_shape(mk, msize) else {
                    continue;
                };
                for c in 0..n_cam {
                    let st = mk.views[c].status[e];
                    if !(usable[c] && st.has_data() && st > r_max) {
                        continue;
                    }
                    let Some(image) = end_images[c]
                        .clone()
                        .or_else(|| load(&mut cache, c, e).ok())
                    else {
                        continue;
                    };
                    let others: Vec<Point2> = markers
                        .iter()
                        .filter(|&&o| {
                            let st = t.markers[o].views[c].status[e];
                            o != m
                                && st.has_data()
                                && (st > r_max
                                    || writes.iter().any(|w| (w.marker, w.camera) == (o, c)))
                        })
                        .map(|&o| t.markers[o].views[c].points[e])
                        .collect();
                    if let Some(q) = refine_stored_point(
                        mk,
                        (m, c, e),
                        msize,
                        &image,
                        mk.views[c].points[e],
                        &others,
                        dark,
                        &options_d,
                        compat.threshold_offset_wraps,
                        p.same_dot,
                    ) {
                        // The other cameras' points at `e` that the run may read (protected
                        // or written by this step) check it, as at the start.
                        let stored: Vec<(usize, Point2)> = (0..n_cam)
                            .filter(|&c2| {
                                let st = mk.views[c2].status[e];
                                usable[c2]
                                    && st.has_data()
                                    && (st > r_max
                                        || writes.iter().any(|w| (w.marker, w.camera) == (m, c2)))
                            })
                            .map(|c2| (c2, mk.views[c2].points[e]))
                            .collect();
                        let used: Vec<(usize, Point2)> = stored
                            .iter()
                            .map(|&(c2, pt)| (c2, if c2 == c { q } else { pt }))
                            .collect();
                        let e_stored = triangulate_points(&ctx, &stored).map(|x| x.1);
                        let e_used = triangulate_points(&ctx, &used).map(|x| x.1);
                        if refinement_agrees(stored.len(), e_stored, e_used, refine_limit) {
                            self.mem.refined.insert((m, c, e), q);
                        }
                    }
                }
            }
        }
        // The cells at e of the markers the run has had, in the cameras it tracks, that hold
        // no point after the step and are not marked untrackable (where the run may never
        // write).
        let left_empty = {
            let t = doc.trial(trial);
            markers
                .iter()
                .filter(|m| self.mem.had.contains(m))
                .map(|&m| {
                    (0..n_cam)
                        .filter(|&c| {
                            let st = t.markers[m].views[c].status[e];
                            usable[c] && !st.has_data() && st != MarkerStatus::Untrackable
                        })
                        .count()
                })
                .sum::<usize>()
        };
        // The points accepted while their marker overlapped another in that camera.
        let mut merged_frames = merged_frames;
        for o in overlaps.iter().filter(|o| o.touching) {
            for k in [o.a, o.b] {
                let m = markers[k];
                if writes.iter().any(|w| (w.marker, w.camera) == (m, o.camera)) {
                    self.mem.merged.insert((m, o.camera, e));
                    if p.overlap_clean {
                        merged_frames.insert((m, e));
                    }
                }
            }
        }
        // A point fitted apart from a touching dot is not a clean picture of its dot either
        // (`split_touching`): it counts as merged.
        for &(m, c) in &apart {
            self.mem.merged.insert((m, c, e));
            if p.overlap_clean {
                merged_frames.insert((m, e));
            }
        }
        for w in &writes {
            self.mem.written.insert((w.marker, w.camera, e));
            if let Some(z) = w.size
                && z > 1.0
                && z < 50.0
            {
                let acc = self.mem.sizes.entry(w.marker).or_insert((0.0, 0));
                acc.0 += z;
                acc.1 += 1;
            }
        }
        // The run's 3D positions at e change with what it wrote.
        tri.retain(|&(_, f), _| f != e);
        self.mem.tri_cache = tri;
        if p.rigid_min_frames > 0 {
            let view = View {
                t: doc.trial(trial),
                ctx: &ctx,
                usable: &usable,
                r_max,
                start: self.request.start_frame,
                forward,
                written: &self.mem.written,
                tri: RefCell::new(std::mem::take(&mut self.mem.tri_cache)),
                refined: RefCell::new(std::mem::take(&mut self.mem.refined)),
                merged: &merged_frames,
            };
            add_pairs(&mut self.mem.pairs, &view, &markers, e);
            self.mem.tri_cache = view.tri.into_inner();
            self.mem.refined = view.refined.into_inner();
        }
        self.mem.merged_frames = merged_frames;
        let tracked: Vec<(usize, usize)> = writes.iter().map(|w| (w.marker, w.camera)).collect();
        self.frame = e;
        self.steps_done += 1;
        // The images of `e` (and of the frame beyond, read for a confirmation) are the next
        // step's.
        cache.retain(|k| k.1 == e || Some(k.1) == e2);
        self.kept = cache;
        Ok(StepReport {
            frame: e,
            finished: None,
            messages,
            detected: tracked.clone(),
            tracked,
            invalid: refused,
            left_empty,
            kept_older,
            removed_older,
            fitted_apart: apart.len(),
        })
    }
}

/// Whether the run's own points at the step's frame contradict the older point `old` of
/// marker `m` (index `k` in `markers`) in camera `c`, a cell the run gave nothing
/// (`MultiTrackParams::keep_unconfirmed`, module documentation step 6): the point is not
/// finite; or (a) the run wrote points of `m` in other cameras in this step and the 3D
/// point triangulated from `old` with them gives no answer (`old` cannot be undistorted, or
/// the point lies behind a camera: an older point far outside the image) or has a largest
/// reprojection error above `max_reprojection_cap` (with no point of `m` written in another
/// camera there is no evidence, and (a) does not apply); or (b) the run wrote a point of
/// another marker in camera `c` within `same_dot` of `old`, and the two are not held as
/// touching in `c` at this step (`overlaps`, as given to the choice). Only points written in
/// this step count, never protected points.
fn contradicts(
    ctx: &FrameContext,
    writes: &[Write],
    overlaps: &[Overlap],
    markers: &[usize],
    (k, m, c): (usize, usize, usize),
    old: Point2,
    p: &MultiTrackParams,
) -> bool {
    if !(old.x.is_finite() && old.y.is_finite()) {
        return true;
    }
    // (a) Another camera of this run contradicts it.
    let mut points: Vec<(usize, Point2)> = vec![(c, old)];
    points.extend(
        writes
            .iter()
            .filter(|w| w.marker == m && w.camera != c)
            .map(|w| (w.camera, w.point)),
    );
    // The run's own point of the marker in another camera is the evidence: an older point that
    // does not even triangulate with it (review 18: far outside the image) disagrees with it.
    if points.len() >= 2 {
        match triangulate_points(ctx, &points) {
            None => return true,
            Some((_, err)) if err > p.max_reprojection_cap => return true,
            Some(_) => {}
        }
    }
    // (b) The run gave that dot to another marker (`keep_dot_rule`), unless the run holds the
    // two as touching in that camera at this step.
    if !p.keep_dot_rule {
        return false;
    }
    writes.iter().any(|w| {
        w.marker != m
            && w.camera == c
            && (w.point - old).norm() <= p.same_dot
            && !markers
                .iter()
                .position(|&o| o == w.marker)
                .is_some_and(|ko| {
                    overlaps.iter().any(|o| {
                        o.camera == c
                            && o.touching
                            && ((o.a, o.b) == (k, ko) || (o.a, o.b) == (ko, k))
                    })
                })
    })
}

/// The largest reprojection error of the 3D point the run triangulates from image points
/// `points` (camera, raw pixel position) of trial `trial` (as rule (a) of
/// `keep_unconfirmed` does), or `None` when it gives none. For tests.
#[doc(hidden)]
#[must_use]
pub fn run_reprojection_error(
    doc: &Document,
    trial: usize,
    points: &[(usize, Point2)],
) -> Option<f64> {
    if trial >= doc.project().trials.len() {
        return None;
    }
    triangulate_points(&doc.frame_context(trial), points).map(|x| x.1)
}

/// The run's own 3D points: per (marker, frame), the point and its largest reprojection
/// error, or `None` when it has none there.
type TriCache = HashMap<(usize, usize), Option<(Point3, f64)>>;

/// The frame `k` frames behind `frame` (forward: before it), within `0..n`.
fn behind(forward: bool, frame: usize, k: usize, n: usize) -> Option<usize> {
    if forward {
        frame.checked_sub(k)
    } else {
        frame.checked_add(k).filter(|&f| f < n)
    }
}

/// What the run may read (module documentation, "What the run reads"): the 2D points of the
/// cameras it tracks at the start frame and behind it, as the document held them when the
/// run started; the points the run has written; and protected points (status above what
/// may be overwritten) at any frame. Its 3D positions are triangulated from those points
/// only, never taken from the document's stored 3D points.
struct View<'a> {
    t: &'a xma_model::Trial,
    ctx: &'a FrameContext,
    usable: &'a [bool],
    r_max: MarkerStatus,
    start: usize,
    forward: bool,
    written: &'a HashSet<(usize, usize, usize)>,
    /// Per (marker, frame), the run's 3D point there and its largest reprojection error.
    tri: RefCell<TriCache>,
    /// The refined positions of points the run reads but did not detect itself
    /// (`RunMemory::refined`): `obs` gives these instead of the stored points.
    refined: RefCell<HashMap<(usize, usize, usize), Point2>>,
    /// The (marker, frame)s where the run accepted a point of the marker while it overlapped
    /// another marker in that camera (`RunMemory::merged_frames`), when such points are not
    /// clean observations (`overlap_clean`); empty otherwise.
    merged: &'a HashSet<(usize, usize)>,
}

impl View<'_> {
    /// [`View::point3`], except at a frame where the marker's point in some camera was
    /// accepted while it overlapped another marker there (`overlap_clean`): such a 3D point
    /// does not give a lost marker its depth, its rigid prediction or the rigid links'
    /// statistics.
    fn point3_clean(&self, m: usize, f: usize) -> Option<Point3> {
        if self.merged.contains(&(m, f)) {
            return None;
        }
        self.point3(m, f)
    }

    /// Whether the run may read marker `m`'s point in camera `c` at frame `f`.
    fn trusted(&self, m: usize, c: usize, f: usize) -> bool {
        if !self.usable.get(c).copied().unwrap_or(false) {
            return false;
        }
        let Some(v) = self.t.markers.get(m).and_then(|mk| mk.views.get(c)) else {
            return false;
        };
        let Some(&status) = v.status.get(f) else {
            return false;
        };
        if !status.has_data() {
            return false;
        }
        let at_or_behind_start = if self.forward {
            f <= self.start
        } else {
            f >= self.start
        };
        at_or_behind_start || self.written.contains(&(m, c, f)) || status > self.r_max
    }

    /// Marker `m`'s point in camera `c` at frame `f`, when the run may read it.
    fn obs(&self, m: usize, c: usize, f: usize) -> Option<Point2> {
        self.trusted(m, c, f).then(|| {
            self.refined
                .borrow()
                .get(&(m, c, f))
                .copied()
                .unwrap_or(self.t.markers[m].views[c].points[f])
        })
    }

    /// The run's 3D point of marker `m` at frame `f` and its largest reprojection error:
    /// triangulated from the points it may read in two cameras or more.
    fn point3_err(&self, m: usize, f: usize) -> Option<(Point3, f64)> {
        if let Some(r) = self.tri.borrow().get(&(m, f)) {
            return *r;
        }
        let r = self.triangulate(m, f);
        self.tri.borrow_mut().insert((m, f), r);
        r
    }

    fn point3(&self, m: usize, f: usize) -> Option<Point3> {
        self.point3_err(m, f).map(|x| x.0)
    }

    fn triangulate(&self, m: usize, f: usize) -> Option<(Point3, f64)> {
        let points: Vec<(usize, Point2)> = (0..self.ctx.cameras.len())
            .filter_map(|c| self.obs(m, c, f).map(|pt| (c, pt)))
            .collect();
        triangulate_points(self.ctx, &points)
    }
}

/// The 3D point of image points `points` (camera, raw pixel position) and its largest
/// reprojection error: `None` with fewer than two usable cameras (calibrated, with a pose),
/// or when a point cannot be undistorted, the triangulation fails, the point is behind a
/// camera or a value is not finite.
fn triangulate_points(ctx: &FrameContext, points: &[(usize, Point2)]) -> Option<(Point3, f64)> {
    let mut obs = Vec::new();
    let mut used = Vec::new();
    for &(c, pt) in points {
        let Some(cam) = ctx.cameras.get(c) else {
            continue;
        };
        let (Some(projection), Some(_)) = (cam.projection, cam.pose.as_ref()) else {
            continue;
        };
        let u = cam.model.try_undistort_point(pt, true, true, true)?;
        if !(u.x.is_finite() && u.y.is_finite()) {
            return None;
        }
        obs.push(Observation {
            projection,
            point: u,
        });
        used.push((c, pt));
    }
    if obs.len() < 2 {
        return None;
    }
    let x = triangulate(ctx.method, &obs)?;
    if !(x.x.is_finite() && x.y.is_finite() && x.z.is_finite()) {
        return None;
    }
    let mut err: f64 = 0.0;
    for (c, pt) in used {
        let cam = &ctx.cameras[c];
        let pose = cam.pose.as_ref()?;
        let m3 = cam.projection?;
        let w = m3[(2, 0)] * x.x + m3[(2, 1)] * x.y + m3[(2, 2)] * x.z + m3[(2, 3)];
        if w.is_nan() || w <= 0.0 {
            return None;
        }
        let q = cam.model.project_point(pose, x);
        err = err.max((q - pt).norm());
    }
    err.is_finite().then_some((x, err))
}

/// Adds frame `f`'s 3D distances of every pair of `markers` (the run's own 3D points) to
/// the pair statistics (Welford).
fn add_pairs(
    pairs: &mut HashMap<(usize, usize), (usize, f64, f64)>,
    view: &View<'_>,
    markers: &[usize],
    f: usize,
) {
    let pts: Vec<(usize, Point3)> = markers
        .iter()
        .filter_map(|&m| view.point3_clean(m, f).map(|x| (m, x)))
        .collect();
    for (i, &(a, x)) in pts.iter().enumerate() {
        for &(b, y) in &pts[i + 1..] {
            let d = (x - y).norm();
            if !d.is_finite() {
                continue;
            }
            let e = pairs.entry((a.min(b), a.max(b))).or_insert((0, 0.0, 0.0));
            e.0 += 1;
            let delta = d - e.1;
            e.1 += delta / e.0 as f64;
            e.2 += delta * (d - e.1);
        }
    }
}

/// A lost marker's 3D position at `e` from its neighbours, with the links that check its
/// identity when it is found again. The latest frame within `depth_gap` where the marker
/// and at least three markers followed now (a 3D track found at the frame before) had
/// 3D points; of those, its `rigid_neighbours` nearest there; the rigid transform from their
/// positions at that frame to their predictions at `e`, the worst-fitting dropped while any
/// misses it by more than `rigid_fit_tolerance` of its distance to the marker (at least
/// three kept); applied to the marker. The links' tolerance is
/// `max(rigid_fit_tolerance, rigid_min_tolerance)` times the distance. `None` when there is
/// no such frame or the neighbours do not move rigidly.
#[allow(clippy::too_many_arguments)]
fn rigid_prediction(
    view: &View<'_>,
    markers: &[usize],
    k: usize,
    pred3s: &[Option<(Point3, usize, bool)>],
    e: usize,
    n_frames: usize,
    depth_gap: usize,
    p: &MultiTrackParams,
    pairs: &HashMap<(usize, usize), (usize, f64, f64)>,
) -> Option<(Point3, Vec<RigidLink>)> {
    let followed: Vec<usize> = (0..markers.len())
        .filter(|&q| q != k && pred3s[q].is_some_and(|x| x.1 == 1))
        .collect();
    if followed.len() < 3 {
        return None;
    }
    let keep = p.rigid_neighbours.clamp(3, markers.len().max(3));
    for j in 1..=depth_gap {
        let f = behind(view.forward, e, j, n_frames)?;
        let Some(x) = view.point3_clean(markers[k], f) else {
            continue;
        };
        let mut near: Vec<(usize, Point3, f64)> = followed
            .iter()
            .filter_map(|&q| {
                let y = view.point3_clean(markers[q], f)?;
                Some((q, y, (y - x).norm()))
            })
            .filter(|n| n.2.is_finite() && n.2 > 0.0)
            .collect();
        if near.len() < 3 {
            continue;
        }
        near.sort_by(|a, b| a.2.total_cmp(&b.2));
        near.truncate(keep);
        loop {
            let src: Vec<Point3> = near.iter().map(|n| n.1).collect();
            let dst: Vec<Point3> = near
                .iter()
                .filter_map(|n| pred3s[n.0].map(|v| v.0))
                .collect();
            let tr = xma_vision::rigid::fit_rigid_transform(&src, &dst)?;
            let r = xma_vision::camera::rodrigues_to_matrix(&tr.rotation);
            let apply = |y: Point3| Point3::from(r * y.coords + tr.translation);
            // Each neighbour's miss relative to its distance from the marker.
            let (worst, miss) = near
                .iter()
                .zip(&dst)
                .enumerate()
                .map(|(i, (n, d))| (i, (apply(n.1) - d).norm() / n.2))
                .max_by(|a, b| a.1.total_cmp(&b.1))?;
            if miss <= p.rigid_fit_tolerance {
                let tol = p.rigid_fit_tolerance.max(p.rigid_min_tolerance);
                let links = near
                    .iter()
                    // `stat_links`: a pair with statistics is checked by them instead.
                    .filter(|n| {
                        !p.stat_links
                            || pair_statistics(pairs, markers[k], markers[n.0], p).is_none()
                    })
                    .map(|n| RigidLink {
                        a: k,
                        b: n.0,
                        distance: n.2,
                        tolerance: tol * n.2,
                    })
                    .collect();
                return Some((apply(x), links));
            }
            if near.len() <= 3 {
                return None;
            }
            near.remove(worst);
        }
    }
    None
}

/// The run's distance statistics of the pair of markers `a` and `b` (`RunMemory::pairs`),
/// when it has been measured at `rigid_min_frames` frames or more (and twice at least), with
/// a positive finite mean: `(mean, tolerance, rigid)`, the tolerance the larger of
/// `rigid_sigmas` times the spread (standard deviation) and `rigid_min_tolerance` of the mean,
/// and `rigid` whether the pair counts as rigid (spread within `rigid_max_spread` of the
/// mean: it then has a learned link with these numbers). `None` otherwise, and always when
/// `rigid_min_frames` is 0 (no statistics are kept then). New in this program (sixth round).
fn pair_statistics(
    pairs: &HashMap<(usize, usize), (usize, f64, f64)>,
    a: usize,
    b: usize,
    p: &MultiTrackParams,
) -> Option<(f64, f64, bool)> {
    if p.rigid_min_frames == 0 || a == b {
        return None;
    }
    let &(n, mean, m2) = pairs.get(&(a.min(b), a.max(b)))?;
    if n < p.rigid_min_frames || n < 2 || !mean.is_finite() || mean <= 0.0 {
        return None;
    }
    let sd = (m2 / (n - 1) as f64).sqrt();
    let tolerance = (p.rigid_sigmas * sd).max(p.rigid_min_tolerance * mean);
    (sd.is_finite() && tolerance.is_finite()).then_some((
        mean,
        tolerance,
        sd <= p.rigid_max_spread * mean,
    ))
}

/// A lost marker placed by its distances (`distance_placement`; module documentation, step
/// 2): the partners are the markers followed in 3D at `e` (a 3D track found at the frame
/// before, as for [`rigid_prediction`]) whose pair with the marker has statistics
/// (`pair_statistics`); four or more are needed. The start is the marker's latest clean 3D
/// point within `depth_gap`, moved by the displacement since that frame of the partner
/// nearest to it by mean distance among those with a clean 3D point there (none: the point
/// as it is); none within `depth_gap`: no placement. The point is
/// [`xma_vision::joint_tracking::point_from_distances`] of the partners' predicted positions
/// at `e`, the pairs' means, and spreads `max(sd, rigid_min_tolerance * mean)`. Returns the
/// point and the partners (indices into `markers`). New in this program (sixth round).
#[allow(clippy::too_many_arguments)]
fn place_by_distances(
    view: &View<'_>,
    markers: &[usize],
    k: usize,
    pred3s: &[Option<(Point3, usize, bool)>],
    e: usize,
    n_frames: usize,
    depth_gap: usize,
    p: &MultiTrackParams,
    pairs: &HashMap<(usize, usize), (usize, f64, f64)>,
) -> Option<(Point3, Vec<usize>)> {
    // (index, predicted position at e, mean, spread)
    let partners: Vec<(usize, Point3, f64, f64)> = (0..markers.len())
        .filter(|&q| q != k)
        .filter_map(|q| {
            let (y, age, _) = pred3s.get(q).copied().flatten()?;
            if age != 1 {
                return None;
            }
            let (a, b) = (markers[k], markers[q]);
            let &(n, mean, m2) = pairs.get(&(a.min(b), a.max(b)))?;
            pair_statistics(pairs, a, b, p)?;
            let sd = (m2 / (n - 1) as f64).sqrt();
            Some((q, y, mean, sd.max(p.rigid_min_tolerance * mean)))
        })
        .collect();
    if partners.len() < 4 {
        return None;
    }
    let (f0, x0) = (1..=depth_gap)
        .map_while(|j| behind(view.forward, e, j, n_frames))
        .find_map(|f| view.point3_clean(markers[k], f).map(|x| (f, x)))?;
    let shift = partners
        .iter()
        .filter_map(|&(q, y, mean, _)| {
            view.point3_clean(markers[q], f0)
                .map(|then| (mean, y - then))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map_or_else(nalgebra::Vector3::zeros, |x| x.1);
    let start = Point3::from(x0.coords + shift);
    let links: Vec<(Point3, f64, f64)> = partners.iter().map(|&(_, y, d, s)| (y, d, s)).collect();
    let x = xma_vision::joint_tracking::point_from_distances(start, &links)?;
    Some((x, partners.iter().map(|x| x.0).collect()))
}

/// One cell's search result: its candidates with their detected sizes, the gate the choice
/// uses and the near gate (`CameraOptions::near_gate`; 0 for none).
type CellFound = (Vec<Found>, f64, f64);

/// A split computed by [`DotSearch::split_at`]: the image's address, the detected point's
/// coordinates (bits) and the two dots, if taken.
type SplitMemo = (usize, u64, u64, Option<[FittedDot; 2]>);

/// A candidate found by [`DotSearch::candidates`]: the candidate, its detected size (-1 when
/// nothing was detected, or when it was fitted apart) and whether it is one of two touching
/// dots fitted apart (`split_touching`).
#[derive(Clone, Copy, Debug)]
struct Found {
    cand: Candidate,
    size: f64,
    apart: bool,
}

/// A correlation peak refined ([`DotSearch::refine`]): the point, its detected size, and
/// whether it is one of two touching dots fitted apart.
#[derive(Clone, Copy, Debug)]
struct Refined {
    point: Point2,
    size: f64,
    apart: bool,
}

/// The least width (Gaussian sigma, pixels) a learned dot width may have
/// ([`learn_width`]): narrower is about one pixel, noise rather than a bead's profile.
const WIDTH_MIN: f64 = 0.7;

/// A marker's dot width learned at the run's own position `pt` of a clean point
/// (`split_touching`; module documentation, step 1): one Gaussian dot fitted there with its
/// centre, width and depth free ([`fit_dot_shape`]), started at half the marker's dot radius
/// `radius` (1 to 4 px), over the pixels within the dot radius plus 2 px of `pt`, at least
/// 5 px (the dot and a little ground round it; 5 px as in the trial of 25 November 2026).
/// Accepted when the fitted centre is within `same_dot` (as a number, 0 to 10) of `pt` (the
/// fit is the marker's own dot, not a neighbour), and the width is at least [`WIDTH_MIN`]
/// and at most half that reach (a wider fit is the plane's work: the window does not hold
/// the dot's tails, so the width is not measured). `None` otherwise.
fn learn_width<S: GreySource + ?Sized>(
    g: &S,
    pt: Point2,
    radius: f64,
    dark: bool,
    same_dot: f64,
) -> Option<f64> {
    let start = (radius / 2.0).clamp(1.0, 4.0);
    let reach = (radius + 2.0).max(5.0);
    let d = fit_dot_shape(g, (pt.x, pt.y), start, dark, reach)?;
    let same = if same_dot.is_finite() {
        same_dot.clamp(0.0, 10.0)
    } else {
        0.0
    };
    let moved = ((d.centre.0 - pt.x).powi(2) + (d.centre.1 - pt.y).powi(2)).sqrt();
    (moved <= same && d.width >= WIDTH_MIN && d.width <= reach / 2.0).then_some(d.width)
}

/// The largest radius (pixels) a correlation search covers, whatever the gate.
const MAX_SEARCH_RADIUS: f64 = 200.0;

/// The most correlation maxima a search beyond an inner radius (the wider first search)
/// examines, strongest first, until `max_candidates` have passed the tests: each costs one
/// detection (a search area of twice the marker's size) and one contrast measure. Worst
/// case per cell: 64 detections in the wide window, plus the two tests a candidate beyond the
/// ordinary gate needs, for at most 32 candidates (16 far ones, and up to 16 of the
/// ordinary search beyond the ordinary gate, all of them with `start_confirm_near`): one
/// confirmation (a search of `gate` plus `search_margin` round one point, at most
/// `2 max_candidates` detections) and one test of the earlier frame (a search of the
/// candidate's distance from the marker's earlier point, at most `gate_start` plus
/// `search_margin`, clamped to [`MAX_SEARCH_RADIUS`], at most 32 detections). The
/// correlation work of every search is bounded by [`ncc_surface`].
const MAX_FAR_PEAKS: usize = 64;

/// With `contrast_relative`, a candidate must also have an absolute contrast of at least
/// this fraction of `min_contrast_fraction` times the median of the marker's recent
/// absolute contrasts (0.3 with the defaults): a black dot on an almost black ground has a
/// relative contrast near 1 and an absolute one of a few grey levels. Half: the true dots
/// the absolute test refused on the guineafowl, where the picture darkens round them, have
/// 0.35 to 0.59 of the median.
const RELATIVE_FLOOR: f64 = 0.5;

/// Whether another marker's dot may lie in or on the ring of a dot of radius `radius` at
/// `point`: one of `neighbours` (other markers' predicted positions with their dot radii)
/// within the ring's outer radius (`radius + 3.5`, as [`dot_contrast`] measures it) plus its
/// own dot radius. Then a candidate is tested by the absolute rule (module documentation,
/// step 3): a dark dot in the ring darkens it and lifts the relative contrast.
fn neighbour_in_ring(point: Point2, radius: f64, neighbours: &[(Point2, f64)]) -> bool {
    neighbours
        .iter()
        .any(|&(q, r)| (q - point).norm() <= radius + 3.5 + r)
}

/// The contrast test of a candidate (module documentation, step 3). With `relative` (the
/// relative rule): its relative contrast `rel` must reach `fraction` of `median_rel`, and its
/// absolute contrast `abs` must reach [`RELATIVE_FLOOR`] times `fraction` of `median_abs`.
/// Without: `abs` must reach `fraction` of `median_abs`. A test without a median (no recent
/// points measured) passes; a contrast that could not be measured fails a test that has a
/// positive median; a median that is not positive passes everything. Total for any input.
fn contrast_passes(
    relative: bool,
    fraction: f64,
    abs: Option<f64>,
    rel: Option<f64>,
    median_abs: Option<f64>,
    median_rel: Option<f64>,
) -> bool {
    let below = |k: Option<f64>, median: Option<f64>, f: f64| {
        median.is_some_and(|m| k.is_none_or(|k| m > 0.0 && k < f * m))
    };
    if relative {
        !below(rel, median_rel, fraction) && !below(abs, median_abs, RELATIVE_FLOOR * fraction)
    } else {
        !below(abs, median_abs, fraction)
    }
}

/// The radius of a marker's dot for the contrast measures: its mean size (2.5 when it has
/// none), as [`contrast_shape`] uses.
fn dot_radius(mean_size: f64) -> f64 {
    if mean_size > 0.0 && mean_size <= 50.0 {
        mean_size
    } else {
        2.5
    }
}

/// Whether the refined positions of a marker at a frame are kept: with points in fewer than
/// two cameras (`cameras`) nothing can check them and they are kept; otherwise the largest
/// reprojection error of the 3D point from the positions the run would use (`e_refined`)
/// must be at most the larger of `max_error` and that of the stored points (`e_stored`): a
/// refinement that moves a point off the marker (onto the edge of a band, say) no longer
/// agrees with the other cameras. A missing or non-finite error refuses; a `max_error` that
/// is not finite counts as 0. Total for any input.
fn refinement_agrees(
    cameras: usize,
    e_stored: Option<f64>,
    e_refined: Option<f64>,
    max_error: f64,
) -> bool {
    if cameras < 2 {
        return true;
    }
    let limit = if max_error.is_finite() {
        max_error
    } else {
        0.0
    };
    match (e_stored, e_refined) {
        (Some(s), Some(r)) if s.is_finite() && r.is_finite() => r <= limit.max(s),
        _ => false,
    }
}

/// The largest distance (pixels) from a stored point to the detected centre that refines it
/// ([`refine_stored_point`]): the offsets of interest (a click anywhere in an elongated
/// marker's dark shape, up to about 8 px from its centre on the guineafowl) plus a little,
/// and no more than the detection's own smallest search reach (10 px).
const REFINE_MAX_SHIFT: f64 = 10.0;

/// How far behind the start (frames) the first step refines stored points
/// ([`refine_stored_point`]): the reach of a template, the velocity and a 2D prediction
/// (`max_gap` plus `velocity_window`; 7 with the defaults), but never more than this, so
/// that a huge `max_gap` or `velocity_window` does not make the first step read every image
/// behind the start. Points further back are used as stored.
const REFINE_REACH: usize = 16;

/// The most times the detection is run to refine a stored point ([`refine_stored_point`]);
/// it must settle (move no more than [`REFINE_SETTLED`]) within these.
const REFINE_ROUNDS: usize = 6;

/// A refining detection has settled when its last run moved the point no more than this
/// (pixels): on an elongated shape successive runs can alternate by about a pixel.
const REFINE_SETTLED: f64 = 1.0;

/// A refined point's detected blob may be at most this many times the marker's mean size
/// ([`refine_stored_point`]): on the guineafowl, a start point moved within its shape gave
/// detected sizes of 0.96 times the mean in the median and above 2 in 5 of 376 cases.
const REFINE_MAX_SIZE: f64 = 2.0;

/// The refined position of a point the run reads but did not detect itself (a start point,
/// a point behind the start, a protected point it reaches): the marker's own detection run
/// at the stored point `stored` in that frame's `image` (search area twice the mean size,
/// at least 10, as for a candidate) and run again from its own result until it settles
/// (on an elongated shape its result depends on its seed; at most [`REFINE_ROUNDS`] runs),
/// accepted when it finds a blob that settles, is at most [`REFINE_MAX_SIZE`] times the
/// marker's mean size (when known), whose centre is more than `same_dot` and at most
/// [`REFINE_MAX_SHIFT`] from the stored point and lies on one dark shape with it
/// ([`xma_vision::joint_tracking::on_one_shape`], polarity `dark`), so that
/// the detection does not carry the point onto a separate neighbour, and no other marker's
/// point of that camera and frame that the run reads (`others`) lies on one shape with the
/// detected centre within twice [`REFINE_MAX_SHIFT`] of it (two markers on one merged
/// shape: its centre is neither marker's). `None` (the stored point is used) otherwise: a
/// point already at a detected centre (within `same_dot`), no blob, too far, not on one
/// shape, a shared shape, or a value not finite or outside the image. Where a marker
/// touches a neighbour that is not a marker of the trial (or whose point the run does not
/// read there), their merged shape is one shape and its centre may be taken (the known
/// merge limit). `marker` is (marker, camera, frame).
#[allow(clippy::too_many_arguments)]
fn refine_stored_point(
    mk: &xma_model::Marker,
    marker: (usize, usize, usize),
    msize: f64,
    image: &AlgorithmImage,
    stored: Point2,
    others: &[Point2],
    dark: bool,
    options_d: &xma_vision::detection::DetectOptions,
    wraps: bool,
    same_dot: f64,
) -> Option<Point2> {
    let (w, h) = (image.width() as f64, image_height(image) as f64);
    let inside = |q: Point2| {
        q.x.is_finite() && q.y.is_finite() && q.x >= 0.0 && q.x < w && q.y >= 0.0 && q.y < h
    };
    if mk.method == DetectionMethod::NoDetection || !inside(stored) {
        return None;
    }
    let (m, c, f) = marker;
    let detect = |at: Point2| -> Option<(Point2, f64)> {
        let mut job = DetectionJob::new(mk, m, c, f, msize * 2.0, true, wraps);
        job.mask_size = input_size(mk, msize) as i32;
        job.center = at;
        let r = job.run(image, options_d);
        (r.size >= 0.0 && inside(r.point)).then_some((r.point, r.size))
    };
    let (first, mut size) = detect(stored)?;
    let same = if same_dot.is_finite() {
        same_dot.max(0.0)
    } else {
        0.0
    };
    // A point already at a detected centre is left as it is.
    if (first - stored).norm() <= same {
        return None;
    }
    // On an elongated shape the detection's result depends on where it starts (it looks
    // within twice the size round its seed): it is run again from its own result until it
    // settles (moves no more than `REFINE_SETTLED`), at most `REFINE_ROUNDS` times.
    let mut point = first;
    let mut settled = false;
    for _ in 1..REFINE_ROUNDS {
        let (next, z) = detect(point)?;
        let moved = (next - point).norm();
        point = next;
        size = z;
        if moved <= REFINE_SETTLED {
            settled = true;
            break;
        }
    }
    // A detection that does not settle walks along something that is not a blob (a band,
    // an edge); a blob much larger than the marker is not the marker.
    if !settled || (msize > 0.0 && size > REFINE_MAX_SIZE * msize) {
        return None;
    }
    let r_point = point;
    let d = (r_point - stored).norm();
    if !(d > same && d <= REFINE_MAX_SHIFT) {
        return None;
    }
    let q = (r_point.x, r_point.y);
    with_grey(image, |g| {
        on_one_shape(g, (stored.x, stored.y), q, dark)
            && !others.iter().any(|o| {
                (*o - r_point).norm() <= 2.0 * REFINE_MAX_SHIFT
                    && on_one_shape(g, (o.x, o.y), q, dark)
            })
    })
    .then_some(r_point)
}

/// The smallest radius (pixels) of the earlier-frame test of the wider first search
/// ([`DotSearch::there_before`]): a dot that stood still is found again within a pixel or
/// so (its detection on another frame).
const STILL_RADIUS: f64 = 2.0;

/// How a marker's dots are found in one camera's image (step 3 of the module
/// documentation): correlation maxima of its template, refined by its detection, with the
/// tests of a candidate.
struct DotSearch<'a> {
    mk: &'a xma_model::Marker,
    marker: usize,
    camera: usize,
    /// The frame the detection job is made for (`e`); the job's starting point is replaced
    /// by each peak, so no stored point is read.
    frame: usize,
    msize: f64,
    shape: Option<(f64, bool)>,
    /// The median of the marker's recent contrasts in this camera.
    median: Option<f64>,
    /// The median of the marker's recent relative contrasts in this camera
    /// (`contrast_relative`).
    median_relative: Option<f64>,
    /// With `contrast_relative`: the other markers predicted in this camera at `e`, with
    /// their dot radii (empty otherwise).
    neighbours: &'a [(Point2, f64)],
    options_d: &'a xma_vision::detection::DetectOptions,
    wraps: bool,
    p: &'a MultiTrackParams,
    /// With `split_touching`, the marker's learned dot width in this camera (none before one
    /// is learned, or without the parameter: no splitting).
    split: Option<f64>,
    /// The numbers of the split (`MultiTrackParams::split_rules`).
    rules: SplitRules,
    /// The splits already computed in this search, by image and detected point (several
    /// correlation peaks on one shape give the same detected point).
    splits: RefCell<Vec<SplitMemo>>,
}

impl DotSearch<'_> {
    /// The candidates round `centre`: the strongest correlation maxima (at least
    /// `min_correlation`) whose peak lies within `reach` of it and beyond `inner` (not positive:
    /// none excluded), with, for a marker being found again (`reacquire`), the dots that
    /// stand out by contrast; each refined ([`DotSearch::refine`]); at most `cap`, none within
    /// `same_dot` of another. The search radius is at most [`MAX_SEARCH_RADIUS`], and
    /// [`ncc_surface`] bounds its work.
    #[allow(clippy::too_many_arguments)]
    fn candidates(
        &self,
        image: &AlgorithmImage,
        template: &ImageF32,
        centre: Point2,
        reach: f64,
        reacquire: bool,
        cap: usize,
        inner: f64,
    ) -> Vec<Found> {
        let p = self.p;
        let radius = if reach.is_finite() {
            reach.ceil().clamp(1.0, MAX_SEARCH_RADIUS) as i32
        } else {
            1
        };
        let Some(surface) = with_grey(image, |g| {
            ncc_surface(g, template, (centre.x, centre.y), radius)
        }) else {
            return Vec::new();
        };
        // The ordinary search takes the `2 cap` strongest maxima of its window. A search
        // beyond `inner` takes the maxima beyond `inner` only (so that the maxima within it,
        // which the ordinary search has seen, cannot crowd them out), strongest first, and
        // examines at most `MAX_FAR_PEAKS` of them.
        let mut peaks = if inner > 0.0 {
            let mut all = local_maxima(&surface.values, p.min_correlation as f32, usize::MAX);
            all.retain(|&(i, j, _)| {
                let q = Point2::new(
                    (surface.off_x + i as i64) as f64,
                    (surface.off_y + j as i64) as f64,
                );
                (q - centre).norm() > inner
            });
            all.truncate(MAX_FAR_PEAKS);
            all
        } else {
            local_maxima(&surface.values, p.min_correlation as f32, 2 * cap)
        };
        // Finding a lost marker again: also the dots of the window that stand out as much as
        // the marker's recent dots, whatever the template says.
        if reacquire
            && p.reacquire_dots
            && let Some((dr, dark)) = self.shape
            && let Some(med) = self.median
        {
            let dots = with_grey(image, |g| {
                contrast_maxima(
                    g,
                    (centre.x, centre.y),
                    radius,
                    dr,
                    dark,
                    p.min_contrast_fraction * med,
                    2 * cap,
                )
            });
            let (w0, h0) = (
                surface.values.width() as i64,
                surface.values.height() as i64,
            );
            for (x, y, _) in dots {
                let i = x as i64 - surface.off_x;
                let j = y as i64 - surface.off_y;
                if i < 0 || j < 0 || i >= w0 || j >= h0 {
                    continue;
                }
                let v = surface.values.at(i as usize, j as usize);
                if !peaks
                    .iter()
                    .any(|q| (q.0 as i64 - i).abs() <= 1 && (q.1 as i64 - j).abs() <= 1)
                {
                    peaks.push((i as usize, j as usize, v));
                }
            }
        }
        let mut out: Vec<Found> = Vec::new();
        'peaks: for (i, j, v) in peaks {
            let px = (surface.off_x + i as i64) as f64;
            let py = (surface.off_y + j as i64) as f64;
            let peak = Point2::new(px, py);
            let d = (peak - centre).norm();
            if d > reach || (inner > 0.0 && d <= inner) {
                continue;
            }
            for r in self.refine(image, peak) {
                if out
                    .iter()
                    .any(|k| (k.cand.point - r.point).norm() <= p.same_dot)
                {
                    continue;
                }
                // A centre fitted apart has the surface's correlation at its nearest pixel,
                // where the surface covers it; else the peak's.
                let correlation = if r.apart {
                    let i = r.point.x.round() as i64 - surface.off_x;
                    let j = r.point.y.round() as i64 - surface.off_y;
                    if i >= 0
                        && j >= 0
                        && (i as usize) < surface.values.width()
                        && (j as usize) < surface.values.height()
                    {
                        surface.values.at(i as usize, j as usize)
                    } else {
                        v
                    }
                } else {
                    v
                };
                out.push(Found {
                    cand: Candidate {
                        point: r.point,
                        correlation: f64::from(correlation),
                    },
                    size: r.size,
                    apart: r.apart,
                });
                if out.len() >= cap {
                    break 'peaks;
                }
            }
        }
        out
    }

    /// A correlation peak refined by the marker's own detection, as the existing tracker
    /// detects after tracking (search area twice the mean size, at least 10), with the tests
    /// of a candidate: nothing when a contour method (0, 2, 5) finds no blob, the point is
    /// outside the image, the detection moved more than `max_detection_shift`, or (dot-like
    /// methods) its contrast is below `min_contrast_fraction` of the median. Returns the
    /// point and the detected size (-1 when nothing was detected).
    ///
    /// With `split_touching` and the marker's dot width known (`split`), the detected point is
    /// then examined for two touching dots ([`split_touching_dots`]); when two are taken, the
    /// two fitted centres are returned in place of the detected point, each with size -1 and
    /// its own tests ([`DotSearch::fitted_candidate`]): none, one or both may pass.
    fn refine(&self, image: &AlgorithmImage, peak: Point2) -> Vec<Refined> {
        let p = self.p;
        let mk = self.mk;
        let (point, size) = if mk.method == DetectionMethod::NoDetection {
            (peak, -1.0)
        } else {
            let mut job = DetectionJob::new(
                mk,
                self.marker,
                self.camera,
                self.frame,
                self.msize * 2.0,
                true,
                self.wraps,
            );
            // The run's own mean size, not the document's (which counts points the run may
            // not read).
            job.mask_size = input_size(mk, self.msize) as i32;
            job.center = peak;
            let r = job.run(image, self.options_d);
            if r.size >= 0.0 {
                (r.point, r.size)
            } else if blob_contour_method(mk.method) {
                // The contour methods find no marker-like blob: not a marker.
                return Vec::new();
            } else {
                // The other methods (blob detector, corner) may find nothing on a small
                // marker; the correlation peak is the point.
                (peak, -1.0)
            }
        };
        let (w, h) = (image.width() as f64, image_height(image) as f64);
        if !(point.x >= 0.0 && point.x < w && point.y >= 0.0 && point.y < h) {
            return Vec::new();
        }
        // Two touching dots: both fitted centres, each with its own tests (among them the
        // shift from the peak). The split is tried before the detected point's own shift test:
        // where the correlation peak stands on the marker's dot and the detection returns the
        // centre of the shape it joined with a neighbour, that centre may lie more than
        // `max_detection_shift` from the peak although the marker's fitted centre does not
        // (the pig's crossing, round b19d: at 8 of its frames the detection moved 3.2 to
        // 4.5 px from a peak of correlation 0.88 to 0.93 on marker 5's dot, and the candidate
        // was dropped before the split). The point that replaces the joined centre is what
        // the shift test applies to.
        if let Some(two) = self.split_at(image, point, size) {
            return [(two[0], two[1]), (two[1], two[0])]
                .into_iter()
                .filter_map(|(dot, other)| self.fitted_candidate(image, peak, dot, other))
                .collect();
        }
        if (point - peak).norm() > p.max_detection_shift {
            return Vec::new();
        }
        // The contrast test ([`contrast_passes`]): the relative rule with `contrast_relative`,
        // unless another marker's dot may lie in or on the candidate's ring
        // ([`neighbour_in_ring`]); else the absolute rule.
        if let Some((radius, dark)) = self.shape {
            let centre = (point.x, point.y);
            let relative =
                p.contrast_relative && !neighbour_in_ring(point, radius, self.neighbours);
            let abs = with_grey(image, |g| dot_contrast(g, centre, radius, dark));
            let rel = if relative {
                with_grey(image, |g| dot_contrast_relative(g, centre, radius, dark))
            } else {
                None
            };
            if !contrast_passes(
                relative,
                p.min_contrast_fraction,
                abs,
                rel,
                self.median,
                self.median_relative,
            ) {
                return Vec::new();
            }
        }
        vec![Refined {
            point,
            size,
            apart: false,
        }]
    }

    /// The two touching dots at the detected `point` of `image`, when the procedure takes two
    /// ([`split_touching_dots`] with the marker's width and polarity): only with
    /// `split_touching`, a dot-like method and a learned width. Computed once per image and
    /// point in this search.
    fn split_at(&self, image: &AlgorithmImage, point: Point2, size: f64) -> Option<[FittedDot; 2]> {
        let width = self.split?;
        let (_, dark) = self.shape?;
        let radius = shape_radius(self.mk.method, size)?;
        let key = (
            std::ptr::from_ref(image) as usize,
            point.x.to_bits(),
            point.y.to_bits(),
        );
        if let Some(k) = self.splits.borrow().iter().find(|k| (k.0, k.1, k.2) == key) {
            return k.3;
        }
        let found = match with_grey(image, |g| {
            split_joined_dots(
                g,
                (point.x, point.y),
                width,
                dark,
                &self.rules,
                Some(radius),
            )
        }) {
            Split::Two { dots, .. } => Some(dots),
            Split::One => None,
        };
        self.splits.borrow_mut().push((key.0, key.1, key.2, found));
        found
    }

    /// A centre fitted apart (`dot`, beside `other`) as a candidate of the correlation peak
    /// `peak`: it must lie in the image and within `max_detection_shift` of the peak, and
    /// pass the contrast test ([`contrast_passes`]) measured on the picture with `other`
    /// taken out ([`WithoutDot`]: the other fitted Gaussian added back), so that neither the
    /// disk nor the ring is darkened by the neighbour. The relative rule's exception for
    /// another marker in the ring ([`neighbour_in_ring`]) leaves out the run's markers
    /// predicted within their own dot radius of `other`'s centre (that dot is the one taken
    /// out); any other marker of the run predicted near still makes the test the absolute
    /// one, as for any candidate. Size -1.
    fn fitted_candidate(
        &self,
        image: &AlgorithmImage,
        peak: Point2,
        dot: FittedDot,
        other: FittedDot,
    ) -> Option<Refined> {
        let p = self.p;
        let point = Point2::new(dot.centre.0, dot.centre.1);
        let (w, h) = (image.width() as f64, image_height(image) as f64);
        if !(point.x >= 0.0 && point.x < w && point.y >= 0.0 && point.y < h) {
            return None;
        }
        if (point - peak).norm() > p.max_detection_shift {
            return None;
        }
        if let Some((radius, dark)) = self.shape {
            let o = Point2::new(other.centre.0, other.centre.1);
            let others: Vec<(Point2, f64)> = self
                .neighbours
                .iter()
                .filter(|&&(q, r)| (q - o).norm().is_nan() || (q - o).norm() > r)
                .copied()
                .collect();
            let relative = p.contrast_relative && !neighbour_in_ring(point, radius, &others);
            let centre = (point.x, point.y);
            let (abs, rel) = with_grey(image, |g| {
                let clean = WithoutDot {
                    source: g,
                    dot: other,
                    dark,
                };
                let abs = dot_contrast(&clean, centre, radius, dark);
                let rel = if relative {
                    dot_contrast_relative(&clean, centre, radius, dark)
                } else {
                    None
                };
                (abs, rel)
            });
            if !contrast_passes(
                relative,
                p.min_contrast_fraction,
                abs,
                rel,
                self.median,
                self.median_relative,
            ) {
                return None;
            }
        }
        Some(Refined {
            point,
            size: -1.0,
            apart: true,
        })
    }

    /// Whether the image of the marker's earlier frame (`before`; no stored point of it is
    /// read) shows a dot, other than the marker's own, nearer to candidate `point` of the
    /// wider first search than the marker's earlier point `earlier` is: a dot within `D` of
    /// the candidate (`D` its distance from `earlier`, at least [`STILL_RADIUS`]), found as
    /// candidates are with the template cut at the candidate in `image` (the image of `e`),
    /// that is not the marker's own earlier dot (a dot within the marker's dot radius plus
    /// `same_dot` of `earlier` is that dot found again: its detection may move a little from
    /// the stored point). Each new dot belongs to its nearest earlier dot unless something
    /// says otherwise: if another dot was nearer to the candidate than this marker was, that
    /// dot is the likelier owner, whatever its speed (a row of dots standing or moving
    /// together: the marker's far candidate may be its neighbour's dot, and the confirmation
    /// one step further on the next neighbour's). A known limit: a real fast marker is
    /// refused when another dot was nearer to where it arrives than it was (a loss, not a
    /// wrong point). The search covers `D` plus `search_margin` round the candidate, at most
    /// [`MAX_SEARCH_RADIUS`], and refines at most 32 correlation maxima.
    fn there_before(
        &self,
        image: &AlgorithmImage,
        before: &AlgorithmImage,
        point: Point2,
        earlier: Point2,
    ) -> bool {
        let p = self.p;
        let r = (point - earlier).norm().max(STILL_RADIUS);
        let r = if r.is_finite() { r } else { STILL_RADIUS };
        let same = if p.same_dot.is_finite() {
            p.same_dot.clamp(0.0, 10.0)
        } else {
            0.0
        };
        let own = dot_radius(self.msize) + same;
        let half = template_size(self.msize).saturating_add(3);
        let Some(template) = with_grey(image, |g| cut_ncc_template(g, point.x, point.y, half))
        else {
            return false;
        };
        let margin = if p.search_margin.is_finite() {
            p.search_margin.clamp(0.0, 10.0)
        } else {
            0.0
        };
        self.candidates(before, &template, point, r + margin, false, 16, 0.0)
            .iter()
            .any(|k| (k.cand.point - point).norm() < r && (k.cand.point - earlier).norm() > own)
    }

    /// Whether the frame beyond `e` confirms candidate `point` of the wider first search:
    /// the candidate's velocity in this camera (from `earlier`, the marker's point `frames`
    /// frames before `e`) carried one frame further gives a position, and `next` (the image
    /// of that frame; no stored point of it is read) holds a dot there within `gate`, found
    /// as candidates are with the template the next step would use (cut at the candidate in
    /// `image`, the image of `e`).
    fn confirmed(
        &self,
        image: &AlgorithmImage,
        next: &AlgorithmImage,
        point: Point2,
        earlier: Point2,
        frames: usize,
    ) -> bool {
        let p = self.p;
        if frames == 0 || p.gate.is_nan() || p.gate <= 0.0 {
            return false;
        }
        let q = point + (point - earlier) / frames as f64;
        if !(q.x.is_finite() && q.y.is_finite()) {
            return false;
        }
        // The template the next step would use: cut at the candidate in the image of `e`
        // (one cut at the earlier point can be poor: at the picture's edge, half of it is
        // the edge replicated, and its correlation peak lies off the dot).
        let half = template_size(self.msize).saturating_add(3);
        let Some(template) = with_grey(image, |g| cut_ncc_template(g, point.x, point.y, half))
        else {
            return false;
        };
        self.candidates(
            next,
            &template,
            q,
            p.gate + p.search_margin,
            false,
            p.max_candidates.clamp(1, 16),
            0.0,
        )
        .iter()
        .any(|k| (k.cand.point - q).norm() <= p.gate)
    }
}

/// The radius round the detected point that the detected shape covers, from the detected
/// `size` (review 19: two dots are fitted apart only where the shape the detection joined
/// covers both). Methods 0, 2 and 5 (contours): `size` is the radius of the circle enclosing
/// the chosen contour (`minEnclosingCircle`); for 0 and 2 the point is that circle's centre,
/// for 5 the contour's centre of mass (inside the circle, not its centre, so the test is
/// round the centre of mass with the circle's radius). Methods 1 and 6 (blob detector): `size`
/// is the blob's diameter (parity ledger A69), so the radius is half of it. `None` (no split)
/// when nothing was detected (size not above 0: the correlation peak is the point, not a
/// joined shape's centre), for a size that is not finite, and for the methods without a shape
/// (corner, no detection).
fn shape_radius(method: DetectionMethod, size: f64) -> Option<f64> {
    if !(size.is_finite() && size > 0.0) {
        return None;
    }
    match method {
        DetectionMethod::XrayMarker
        | DetectionMethod::WhiteMarker
        | DetectionMethod::XrayMarkerCenterOfMass => Some(size),
        DetectionMethod::Blob | DetectionMethod::WhiteBlob => Some(size / 2.0),
        DetectionMethod::Corner | DetectionMethod::NoDetection => None,
    }
}

/// Whether a detection method looks for a blob by its contour (methods 0, 2, 5), so that
/// finding nothing means there is no marker-like blob.
fn blob_contour_method(method: DetectionMethod) -> bool {
    matches!(
        method,
        DetectionMethod::XrayMarker
            | DetectionMethod::WhiteMarker
            | DetectionMethod::XrayMarkerCenterOfMass
    )
}

/// The radius and polarity [`dot_contrast`] measures a marker's dots with: its mean size
/// (2.5 when it has none), dark unless the method looks for white markers. `None` for the
/// methods whose markers are not round dots (corner detection, no detection): for them
/// there is no contrast test and no search by contrast.
fn contrast_shape(mk: &xma_model::Marker, mean_size: f64) -> Option<(f64, bool)> {
    let dark = match mk.method {
        DetectionMethod::XrayMarker
        | DetectionMethod::XrayMarkerCenterOfMass
        | DetectionMethod::Blob => true,
        DetectionMethod::WhiteMarker | DetectionMethod::WhiteBlob => false,
        DetectionMethod::Corner | DetectionMethod::NoDetection => return None,
    };
    let radius = if mean_size > 0.0 && mean_size <= 50.0 {
        mean_size
    } else {
        2.5
    };
    Some((radius, dark))
}

/// The detection's input size for a marker with the run's mean size: the size override if
/// positive, else the mean size if positive, else 5 (as the detection after tracking).
fn input_size(mk: &xma_model::Marker, mean_size: f64) -> f64 {
    if mk.size_override > 0 {
        f64::from(mk.size_override)
    } else if mean_size > 0.0 {
        mean_size
    } else {
        5.0
    }
}

/// A marker's reprojection limit from its recent points (module documentation): the median
/// of the run's own reprojection errors at its latest `contrast_history` frames with a 3D
/// point within 50 frames behind `e`, times `reprojection_factor`, when that exceeds
/// `max_reprojection_error`, up to `max_reprojection_cap`.
fn reprojection_limit(
    view: &View<'_>,
    m: usize,
    e: usize,
    n_frames: usize,
    p: &MultiTrackParams,
) -> Option<f64> {
    let mut errs: Vec<f64> = (1..=50usize)
        .map_while(|k| behind(view.forward, e, k, n_frames))
        .filter_map(|f| view.point3_err(m, f).map(|x| x.1))
        .take(p.contrast_history.clamp(1, 1000))
        .collect();
    if errs.is_empty() {
        return None;
    }
    errs.sort_by(f64::total_cmp);
    let l = errs[errs.len() / 2] * p.reprojection_factor;
    (l > p.max_reprojection_error).then(|| l.min(p.max_reprojection_cap))
}

fn image_height(image: &AlgorithmImage) -> usize {
    match image {
        AlgorithmImage::U8(im) => im.height(),
        AlgorithmImage::Deep { image, .. } => image.height(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 120 by 120 image of ground 180 with dark elliptical Gaussian shapes: (centre x,
    /// centre y, depth, sigma along x, sigma along y).
    fn refine_scene(shapes: &[(f64, f64, f64, f64, f64)]) -> AlgorithmImage {
        let (w, h) = (120usize, 120usize);
        let mut d = vec![180.0f64; w * h];
        for &(cx, cy, depth, sx, sy) in shapes {
            for y in 0..h {
                for x in 0..w {
                    let u = (x as f64 - cx) / sx;
                    let v = (y as f64 - cy) / sy;
                    d[y * w + x] -= depth * (-(u * u + v * v) / 2.0).exp();
                }
            }
        }
        let data = d
            .iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        jobs::algorithm_image_for(&Frame::gray8(w as u32, h as u32, data), CppCompat::FIXED)
    }

    /// [`refine_stored_point`] for an X-ray marker in `image`, dark, with the defaults.
    fn refine_in(
        image: &AlgorithmImage,
        stored: Point2,
        msize: f64,
        others: &[Point2],
    ) -> Option<Point2> {
        let mk = xma_model::Marker::new(1, 1, 8);
        refine_stored_point(
            &mk,
            (0, 0, 0),
            msize,
            image,
            stored,
            others,
            true,
            &xma_vision::detection::DetectOptions::default(),
            false,
            MultiTrackParams::default().same_dot,
        )
    }

    /// Two markers of the run on one merged shape (two dots of sigma 2 px, 5 px apart), each
    /// start point 2 px outside its own dot: the detection at either takes the merged shape's
    /// centre, which is neither marker's, and the other marker's point on the same shape
    /// refuses it. Without the other marker's point the same detection is taken (so the
    /// guard is what refuses it).
    #[test]
    fn refinement_onto_a_shape_shared_with_another_marker_is_refused() {
        let image = refine_scene(&[(57.5, 60.0, 130.0, 2.0, 2.0), (62.5, 60.0, 130.0, 2.0, 2.0)]);
        let (a, b) = (Point2::new(55.5, 60.0), Point2::new(64.5, 60.0));
        assert_eq!(refine_in(&image, a, 3.0, &[b]), None);
        assert_eq!(refine_in(&image, b, 3.0, &[a]), None);
        let alone = refine_in(&image, a, 3.0, &[]).expect("refined without the other marker");
        assert!((alone - Point2::new(60.0, 60.0)).norm() < 0.5, "{alone}");
    }

    /// A small marker (sigma 1.4 px, mean size 2) touching a larger dot that is not a marker
    /// (sigma 2.5 px, 5 px away); the start point 2 px off the marker, away from the other
    /// dot. The detection takes the merged shape (detected size 4.8, more than
    /// [`REFINE_MAX_SIZE`] times 2): refused. With the size unknown (-1) no size check
    /// applies and the merged shape's centre is taken (the known limit).
    #[test]
    fn refinement_onto_a_blob_much_larger_than_the_marker_is_refused() {
        let image = refine_scene(&[(57.0, 60.0, 120.0, 1.4, 1.4), (62.0, 60.0, 140.0, 2.5, 2.5)]);
        let stored = Point2::new(55.0, 60.0);
        assert_eq!(refine_in(&image, stored, 2.0, &[]), None);
        let unknown = refine_in(&image, stored, -1.0, &[]).expect("taken with the size unknown");
        assert!(
            (unknown - Point2::new(57.0, 60.0)).norm() > 2.0,
            "{unknown}"
        );
    }

    /// A start point on a faint small dot (depth 60) with a strong dot 8 px away across
    /// bright ground: the detection there takes the strong dot, which is not on one shape
    /// with the start point ([`on_one_shape`]): refused.
    #[test]
    fn refinement_across_a_bright_gap_is_refused() {
        let image = refine_scene(&[(55.0, 60.0, 60.0, 1.5, 1.5), (63.0, 60.0, 140.0, 2.0, 2.0)]);
        assert_eq!(refine_in(&image, Point2::new(55.0, 60.0), 3.0, &[]), None);
    }

    /// A dark band 60 px long, its half-width growing from 4 to 5.2 px and its depth from 60
    /// to 120 along it; the start point in its middle (mean size 6). Each run of the
    /// detection moves its result more than [`REFINE_SETTLED`] along the band (to within
    /// [`REFINE_MAX_SHIFT`] after [`REFINE_ROUNDS`] runs): it does not settle, refused.
    #[test]
    fn refinement_that_walks_along_a_band_is_refused() {
        let (w, h) = (120usize, 120usize);
        let mut d = vec![180.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let xc = (x as f64).clamp(30.0, 90.0);
                let half = 4.0 + 0.02 * (xc - 30.0);
                let depth = 60.0 + (xc - 30.0);
                let u = (x as f64 - xc) / half;
                let v = (y as f64 - 60.0) / half;
                d[y * w + x] -= depth * (-(u * u + v * v) / 2.0).exp();
            }
        }
        let data = d
            .iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        let image =
            jobs::algorithm_image_for(&Frame::gray8(w as u32, h as u32, data), CppCompat::FIXED);
        for x in [60.0, 62.0] {
            assert_eq!(
                refine_in(&image, Point2::new(x, 60.0), 6.0, &[]),
                None,
                "{x}"
            );
        }
    }

    /// The cross-camera check of a refinement: kept with fewer than two cameras, or when the
    /// refined positions agree in 3D at least as well as the stored ones (or within the
    /// limit); refused otherwise, and for any missing or non-finite error.
    #[test]
    fn refinement_cross_camera_decision() {
        // One camera: nothing to check.
        assert!(refinement_agrees(1, None, None, 1.5));
        assert!(refinement_agrees(0, Some(f64::NAN), None, 1.5));
        // Within the limit, or no worse than the stored points.
        assert!(refinement_agrees(2, Some(0.7), Some(0.4), 1.5));
        assert!(refinement_agrees(2, Some(0.3), Some(1.4), 1.5));
        assert!(refinement_agrees(2, Some(2.6), Some(2.5), 1.5));
        // Worse than both: the Rat's marker 12 at frame 97 (2.17 stored, 6.78 refined).
        assert!(!refinement_agrees(2, Some(2.17), Some(6.78), 1.5));
        assert!(!refinement_agrees(2, Some(0.96), Some(1.86), 1.5));
        // Missing or non-finite: refused.
        assert!(!refinement_agrees(2, None, Some(0.1), 1.5));
        assert!(!refinement_agrees(2, Some(0.1), None, 1.5));
        assert!(!refinement_agrees(3, Some(f64::NAN), Some(0.1), 1.5));
        assert!(!refinement_agrees(2, Some(0.1), Some(f64::INFINITY), 1.5));
        // A limit that is not finite counts as 0.
        assert!(!refinement_agrees(2, Some(0.1), Some(0.2), f64::NAN));
        assert!(refinement_agrees(2, Some(0.3), Some(0.2), f64::INFINITY));
    }

    /// The two guards of `contrast_relative`, as pure decisions: the floor on the absolute
    /// contrast, and the absolute rule when a neighbour lies in the ring.
    #[test]
    fn contrast_decision() {
        let (f, ma, mr) = (0.6, Some(100.0), Some(0.4));
        // A dot on an almost black ground: relative contrast 0.9 (passes 0.6 of 0.4),
        // absolute 5 (below 0.3 of 100): refused with the relative rule (the floor).
        assert!(!contrast_passes(true, f, Some(5.0), Some(0.9), ma, mr));
        // Absolute 35: above the floor (30), so the relative rule passes it ...
        assert!(contrast_passes(true, f, Some(35.0), Some(0.4), ma, mr));
        // ... while the absolute rule (0.6 of 100) refuses it.
        assert!(!contrast_passes(false, f, Some(35.0), Some(0.4), ma, mr));
        // Relative contrast too low: refused.
        assert!(!contrast_passes(true, f, Some(90.0), Some(0.2), ma, mr));
        // No medians: nothing to test.
        assert!(contrast_passes(true, f, None, None, None, None));
        assert!(contrast_passes(false, f, None, None, None, None));
        // Unmeasurable contrast with a median: refused.
        assert!(!contrast_passes(false, f, None, None, ma, mr));
        // Any value: no panic.
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
            for rel in [false, true] {
                let _ = contrast_passes(rel, v, Some(v), Some(v), Some(v), Some(v));
            }
        }
        // The neighbour test: ring outer radius 3 + 3.5, plus the neighbour's radius 3.
        let at = Point2::new(100.0, 100.0);
        assert!(neighbour_in_ring(
            at,
            3.0,
            &[(Point2::new(109.5, 100.0), 3.0)]
        ));
        assert!(!neighbour_in_ring(
            at,
            3.0,
            &[(Point2::new(109.6, 100.0), 3.0)]
        ));
        assert!(!neighbour_in_ring(at, 3.0, &[]));
        assert!(!neighbour_in_ring(at, f64::NAN, &[(at, f64::NAN)]));
    }

    /// `keep_unconfirmed`'s decision without cameras (no camera can triangulate, so rule (a)
    /// resets whenever the marker was written in another camera, and does not apply
    /// otherwise): non-finite older points are reset; rule (b) resets an older point within
    /// `same_dot` of another marker's point written in the same camera unless the two are
    /// given as touching there; points of the same marker, of another camera, further than
    /// `same_dot`, or of a pair touching in another camera do not count.
    #[test]
    fn keep_unconfirmed_rule_b_and_non_finite() {
        let ctx = FrameContext {
            cameras: Vec::new(),
            no_calibration: false,
            method: xma_vision::triangulation::TriangulationMethod::default(),
            triangulation_unchecked: false,
            project_undefined_points: false,
        };
        let p = MultiTrackParams::default();
        let w = |marker: usize, camera: usize, x: f64, y: f64| Write {
            marker,
            camera,
            point: Point2::new(x, y),
            status: MarkerStatus::Tracked,
            size: Some(3.0),
        };
        let markers = [3usize, 7, 9];
        // Marker 3 (index 0), camera 1, older point at (100, 100).
        let cell = (0usize, 3usize, 1usize);
        let old = Point2::new(100.0, 100.0);
        let touch = |a: usize, b: usize, camera: usize, touching: bool| Overlap {
            a,
            b,
            camera,
            touching,
        };
        let no_ov: [Overlap; 0] = [];
        // Nothing written: kept.
        assert!(!contradicts(&ctx, &[], &no_ov, &markers, cell, old, &p));
        // Marker 7 written on the same dot (1.5 px away, the limit), not touching: reset.
        let on_dot = [w(7, 1, 101.5, 100.0)];
        assert!(contradicts(&ctx, &on_dot, &no_ov, &markers, cell, old, &p));
        // Just beyond same_dot: kept.
        let beside = [w(7, 1, 101.51, 100.0)];
        assert!(!contradicts(&ctx, &beside, &no_ov, &markers, cell, old, &p));
        // The same dot in another camera, or the same marker's own point: kept.
        let other_cam = [w(7, 0, 100.0, 100.0)];
        assert!(!contradicts(
            &ctx, &other_cam, &no_ov, &markers, cell, old, &p
        ));
        let own = [w(3, 1, 100.0, 100.0)];
        assert!(!contradicts(&ctx, &own, &no_ov, &markers, cell, old, &p));
        // Touching in that camera (either order): kept.
        for ov in [touch(0, 1, 1, true), touch(1, 0, 1, true)] {
            assert!(!contradicts(&ctx, &on_dot, &[ov], &markers, cell, old, &p));
        }
        // Held but not touching now, touching in another camera, or another pair touching:
        // reset.
        for ov in [
            touch(0, 1, 1, false),
            touch(0, 1, 0, true),
            touch(0, 2, 1, true),
        ] {
            assert!(contradicts(&ctx, &on_dot, &[ov], &markers, cell, old, &p));
        }
        // A marker not in the run's list written there (cannot be held as touching): reset.
        let stranger = [w(42, 1, 100.0, 100.0)];
        assert!(contradicts(
            &ctx,
            &stranger,
            &[touch(0, 1, 1, true)],
            &markers,
            cell,
            old,
            &p
        ));
        // Rule (a) with no answer from the triangulation (here: no camera can triangulate, as
        // for an older point that cannot be undistorted): the run's own point of the marker
        // in another camera is the evidence, so the older point is reset (review 18); with
        // no point of the marker written in another camera, (a) does not apply.
        let own_other = [w(3, 0, 100.0, 100.0)];
        assert!(contradicts(
            &ctx, &own_other, &no_ov, &markers, cell, old, &p
        ));
        let unrelated = [w(7, 0, 300.0, 300.0), w(3, 1, 300.0, 300.0)];
        assert!(!contradicts(
            &ctx, &unrelated, &no_ov, &markers, cell, old, &p
        ));
        // `keep_dot_rule` off: rule (b) never resets; a point that is not finite still is.
        let off = MultiTrackParams {
            keep_dot_rule: false,
            ..MultiTrackParams::default()
        };
        assert!(!contradicts(
            &ctx, &on_dot, &no_ov, &markers, cell, old, &off
        ));
        assert!(!contradicts(
            &ctx, &stranger, &no_ov, &markers, cell, old, &off
        ));
        assert!(contradicts(
            &ctx,
            &on_dot,
            &no_ov,
            &markers,
            cell,
            Point2::new(f64::NAN, 1.0),
            &off
        ));
        // Older points that are not finite: reset, whatever was written.
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for pt in [Point2::new(v, 1.0), Point2::new(1.0, v)] {
                assert!(contradicts(&ctx, &[], &no_ov, &markers, cell, pt, &p));
            }
        }
        // Hostile parameters and points: no panic.
        for v in [f64::NAN, f64::INFINITY, -1.0, 0.0, 1e300] {
            let q = MultiTrackParams {
                same_dot: v,
                max_reprojection_cap: v,
                ..MultiTrackParams::default()
            };
            let hostile = [w(7, 1, v, v), w(3, 0, v, 1.0)];
            let _ = contradicts(
                &ctx,
                &hostile,
                &[touch(9, 99, 1, true)],
                &markers,
                cell,
                old,
                &q,
            );
            let _ = contradicts(
                &ctx,
                &hostile,
                &no_ov,
                &markers,
                (99, 3, 99),
                Point2::new(1e300, -1e300),
                &q,
            );
        }
    }

    /// A marker's dot width learned at the run's position (`learn_width`): a dot of width
    /// 1.4 px is measured within 0.1 px from a point 0.5 px off its centre; a point 2 px off
    /// (the fitted centre moves more than `same_dot`) or a flat picture gives none; a dot
    /// beside a neighbour 7 px away is still measured within 0.15 px. Total for any value of
    /// the position, radius and `same_dot`.
    #[test]
    fn learn_width_decisions() {
        let one = refine_scene(&[(60.0, 60.0, 60.0, 1.4, 1.4)]);
        let w = with_grey(&one, |g| {
            learn_width(g, Point2::new(60.5, 60.0), 2.5, true, 1.5)
        });
        assert!(w.is_some_and(|w| (w - 1.4).abs() < 0.1), "{w:?}");
        let far = with_grey(&one, |g| {
            learn_width(g, Point2::new(62.0, 60.0), 2.5, true, 1.5)
        });
        assert_eq!(far, None);
        let flat = refine_scene(&[]);
        assert_eq!(
            with_grey(&flat, |g| learn_width(
                g,
                Point2::new(60.0, 60.0),
                2.5,
                true,
                1.5
            )),
            None
        );
        let pair = refine_scene(&[(60.0, 60.0, 60.0, 1.4, 1.4), (67.0, 60.0, 60.0, 1.4, 1.4)]);
        let w = with_grey(&pair, |g| {
            learn_width(g, Point2::new(60.0, 60.0), 2.5, true, 1.5)
        });
        assert!(w.is_some_and(|w| (w - 1.4).abs() < 0.15), "{w:?}");
        for v in [f64::NAN, f64::INFINITY, -1.0, 0.0, 1e300, -1e300] {
            let _ = with_grey(&one, |g| learn_width(g, Point2::new(v, 60.0), v, true, v));
            let _ = with_grey(&one, |g| {
                learn_width(g, Point2::new(60.0, 60.0), v, false, 1.5)
            });
            let _ = with_grey(&one, |g| {
                learn_width(g, Point2::new(60.0, 60.0), 2.5, true, v)
            });
        }
    }

    /// A centre fitted apart is tested on the picture with the other fitted dot taken out
    /// (`DotSearch::fitted_candidate`): a faint dot (depth 45) 4.5 px from a deep one (depth
    /// 85). Its contrast measured on the raw picture, where the deep dot darkens its ring, is
    /// under 0.8 of its contrast alone, and within 10 % of it with the deep dot taken out; with
    /// a median between the two the candidate passes, and fails with nothing taken out. And a marker of
    /// the run predicted on the other (taken-out) dot does not switch the relative test to the
    /// absolute one: with a median that the relative rule passes and the absolute one fails,
    /// the candidate passes with that neighbour, and fails with a neighbour predicted elsewhere
    /// near its ring.
    #[test]
    fn a_fitted_centre_is_tested_without_its_neighbour() {
        let (b, a) = ((60.0, 60.0), (64.5, 60.0));
        let deep = 85.0;
        let lone = refine_scene(&[(b.0, b.1, 45.0, 1.4, 1.4)]);
        let pair = refine_scene(&[(b.0, b.1, 45.0, 1.4, 1.4), (a.0, a.1, deep, 1.4, 1.4)]);
        let radius = 2.5;
        let c = (b.0, b.1);
        let alone = with_grey(&lone, |g| dot_contrast(g, c, radius, true)).expect("contrast");
        let alone_rel =
            with_grey(&lone, |g| dot_contrast_relative(g, c, radius, true)).expect("contrast");
        let raw = with_grey(&pair, |g| dot_contrast(g, c, radius, true)).expect("contrast");
        assert!(raw < 0.8 * alone, "raw {raw} alone {alone}");
        let Split::Two { dots, .. } = with_grey(&pair, |g| {
            xma_vision::dot_fit::split_touching_dots(
                g,
                (62.0, 60.0),
                1.4,
                true,
                &SplitRules::default(),
            )
        }) else {
            panic!("not split");
        };
        let near = |d: &FittedDot, q: (f64, f64)| {
            ((d.centre.0 - q.0).powi(2) + (d.centre.1 - q.1).powi(2)).sqrt() < 0.5
        };
        let (fb, fa) = if near(&dots[0], b) {
            (dots[0], dots[1])
        } else {
            (dots[1], dots[0])
        };
        assert!(near(&fb, b) && near(&fa, a), "{dots:?}");
        let wrapped = with_grey(&pair, |g| {
            let clean = WithoutDot {
                source: g,
                dot: fa,
                dark: true,
            };
            dot_contrast(&clean, c, radius, true)
        })
        .expect("contrast");
        assert!((wrapped / alone - 1.0).abs() < 0.1, "{wrapped} {alone}");
        // A median whose 0.6 lies between the raw contrast and the one without the neighbour.
        let median = (raw + wrapped) / 2.0 / 0.6;
        let mk = xma_model::Marker::new(1, 1, 8);
        let options_d = xma_vision::detection::DetectOptions::default();
        let absolute = MultiTrackParams {
            contrast_relative: false,
            ..MultiTrackParams::default()
        };
        let search = |p: &MultiTrackParams,
                      median: f64,
                      median_rel: f64,
                      neighbours: &[(Point2, f64)]|
         -> bool {
            let s = DotSearch {
                mk: &mk,
                marker: 0,
                camera: 0,
                frame: 0,
                msize: radius,
                shape: Some((radius, true)),
                median: Some(median),
                median_relative: Some(median_rel),
                neighbours,
                options_d: &options_d,
                wraps: false,
                p,
                split: Some(1.4),
                rules: SplitRules::default(),
                splits: RefCell::new(Vec::new()),
            };
            s.fitted_candidate(&pair, Point2::new(62.0, 60.0), fb, fa)
                .is_some_and(|r| r.apart && r.size == -1.0)
        };
        // The wrapper: passes with the deep dot taken out; with nothing taken out (a dot of
        // depth 0) the raw contrast fails.
        assert!(search(&absolute, median, alone_rel, &[]));
        let nothing = FittedDot { depth: 0.0, ..fa };
        let s = DotSearch {
            mk: &mk,
            marker: 0,
            camera: 0,
            frame: 0,
            msize: radius,
            shape: Some((radius, true)),
            median: Some(median),
            median_relative: Some(alone_rel),
            neighbours: &[],
            options_d: &options_d,
            wraps: false,
            p: &absolute,
            split: Some(1.4),
            rules: SplitRules::default(),
            splits: RefCell::new(Vec::new()),
        };
        assert!(
            s.fitted_candidate(&pair, Point2::new(62.0, 60.0), fb, nothing)
                .is_none()
        );
        // The neighbour exception: relative rule (default), a median of absolute contrasts
        // such that the relative rule's floor (0.3 of it) passes and the absolute test (0.6)
        // fails.
        let relative = MultiTrackParams::default();
        let m_abs = alone / 0.45;
        let on_other = [(Point2::new(a.0 + 0.3, a.1), 2.5)];
        let elsewhere = [(Point2::new(b.0, b.1 - 5.0), 2.5)];
        assert!(search(&relative, m_abs, alone_rel, &on_other));
        assert!(!search(&relative, m_abs, alone_rel, &elsewhere));
        // Outside the image or beyond `max_detection_shift` of the peak: none.
        let far = MultiTrackParams {
            max_detection_shift: 1.0,
            contrast_relative: false,
            ..MultiTrackParams::default()
        };
        assert!(!search(&far, median, alone_rel, &[]));
    }

    /// `learn_width`'s bounds (review 19: no test failed with them removed): a dot narrower
    /// than [`WIDTH_MIN`] (sigma 0.62 px) gives no width, nor a dot wider than half the reach
    /// (sigma 3 px with a dot radius of 2.5: reach 5, at most 2.5); dots of 0.9 and 2.2 px are
    /// learned within 0.15 px.
    #[test]
    fn learn_width_bounds() {
        let learn = |sigma: f64| {
            let im = refine_scene(&[(60.0, 60.0, 70.0, sigma, sigma)]);
            with_grey(&im, |g| {
                learn_width(g, Point2::new(60.0, 60.0), 2.5, true, 1.5)
            })
        };
        let narrow = learn(0.62);
        assert_eq!(narrow, None, "narrow {narrow:?}");
        let wide = learn(3.0);
        assert_eq!(wide, None, "wide {wide:?}");
        for sigma in [0.9, 2.2] {
            let w = learn(sigma);
            assert!(
                w.is_some_and(|w| (w - sigma).abs() < 0.15),
                "{sigma}: {w:?}"
            );
        }
    }
}
