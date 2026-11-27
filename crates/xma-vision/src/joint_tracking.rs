//! Multi-camera ("joint") marker tracking: the pure parts. New in this program (not ported
//! from XMALab, which tracks each camera on its own).
//!
//! The loop that uses these is `xma_engine::tracking::multi`, whose documentation describes
//! one step as a whole. Here:
//!
//! - candidates: [`cut_ncc_template`], [`ncc_surface`] (zero-mean normalised correlation
//!   round a prediction), [`local_maxima`], [`dot_contrast`] (how much a dot stands out
//!   from its ring, which tells a bead from noise) and [`contrast_maxima`] (dots found by
//!   contrast alone, for a marker found again);
//! - geometry: [`closest_point_on_ray`] (a 3D position for a marker seen by one camera) and
//!   [`point_on_ray_keeping_links`] (the place on that ray where its rigid links hold);
//! - the joint choice of one dot per camera for every marker: [`choose_joint`],
//!   [`choose_joint_linked`] (with [`RigidLink`]s between markers on one bone) and
//!   [`choose_joint_overlapping`] (with the markers that overlap in a camera, [`Overlap`]:
//!   who may share a dot, and identity out of an overlap);
//! - the parameters, with their defaults: [`MultiTrackParams`].
//!
//! The central point: a combination of one dot per camera is accepted only when its
//! reprojection error is small **and** every dot lies within a gate of the marker's
//! predicted position in its camera. The reprojection error does not change when a dot
//! slides along another camera's epipolar line; the distance from the prediction does.
//! A marker found again after it was lost must also prove its identity: its 3D point keeps
//! its distances to markers on the same bone ([`RigidLink`]), or it is not accepted.
//!
//! Every public function here is total: any argument value gives a result without
//! panicking, and its work is bounded: [`ncc_surface`] by [`MAX_TEMPLATE_SIDE`] and
//! [`MAX_NCC_WORK`], [`contrast_maxima`] by its clamped radii, [`choose_joint`] by at most
//! 16 candidates per camera and [`MAX_COMBINATIONS`] combinations per marker (each a
//! triangulation over the cameras) plus the dot clustering, quadratic in the number of
//! candidates of a camera; allocation is in proportion to the arguments.

use crate::camera::{CameraModel, RigidTransform};
use crate::image::{GreySource, ImageF32};
use crate::imgproc::subpix::{get_sub_image_32f, get_sub_image_at_32f, max_window_half};
use crate::triangulation::{Observation, TriangulationMethod, triangulate};
use crate::{Matrix3x4, Point2, Point3};

/// The parameters of multi-camera tracking, with their defaults. New in this program.
///
/// Pixel distances are in image pixels of the camera concerned. Costs are dimensionless;
/// a feasible combination costs roughly 0 (perfect) to a few units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiTrackParams {
    /// Largest reprojection error (pixels, in every camera of a combination) of a
    /// triangulated combination. Default 1.5.
    pub max_reprojection_error: f64,
    /// Gate radius (pixels) round the prediction for a marker found at the frame just
    /// before. Default 5.
    pub gate: f64,
    /// A marker's reprojection limit rises to this many times the median reprojection error
    /// of its recent points (the last `contrast_history`), when that is larger than
    /// `max_reprojection_error`, up to `max_reprojection_cap`. Default 2.5.
    pub reprojection_factor: f64,
    /// See `reprojection_factor`. Default 3.
    pub max_reprojection_cap: f64,
    /// A lost marker's position is predicted from its nearest followed neighbours (at most
    /// this many, at least 3) moved rigidly from a frame where all were seen. Default 5.
    pub rigid_neighbours: usize,
    /// ... when every neighbour fits that rigid motion within this fraction of its mean
    /// distance to the marker. Default 0.02.
    pub rigid_fit_tolerance: f64,
    /// When a lost marker is looked for again, the dots that stand out by contrast
    /// ([`contrast_maxima`]) are candidates too, not only the template's correlation maxima.
    /// Default true.
    pub reacquire_dots: bool,
    /// Gate radius (pixels) when the marker's velocity is not known (one earlier point only:
    /// the first step of a run from a single frame). Default 10.
    pub gate_hold: f64,
    /// A wider first search when the velocity is not known (`gate_hold`'s case: neither the
    /// 3D track nor the camera's own 2D track has one): candidates are searched within this
    /// radius (pixels) of the prediction, and a candidate further than the ordinary
    /// no-velocity gate ([`MultiTrackParams::gate_for`]) is kept only when the image of the
    /// marker's earlier frame shows no dot there already, and the next frame confirms it: at
    /// the position it implies one frame further on (constant velocity in that camera from
    /// the marker's earlier point), the image of the next frame holds a dot found as
    /// candidates are, within `gate`. The cell's gate is then this radius, and the ordinary
    /// gate its near gate ([`CameraOptions::near_gate`]): a combination within the ordinary
    /// gates always ranks first. Such candidates only fill the places `max_candidates` leaves after the
    /// ordinary ones. A value not above the ordinary no-velocity gate, or not a finite
    /// number, turns this off; at most 200 is used. Default 40.
    pub gate_start: f64,
    /// With the wider first search in use (`gate_start`), the candidates within the ordinary
    /// gate need the next frame's confirmation too. Default false.
    pub start_confirm_near: bool,
    /// Gate radius (pixels) round a position predicted from rigid neighbours. Default 6.
    pub gate_rigid: f64,
    /// A marker found again after it was lost (a camera that has not had it for more than
    /// `max_gap` frames, or a position from its rigid neighbours) is accepted only when its
    /// 3D point keeps its rigid links to at least this many partners already accepted at the
    /// frame and breaks none; a single-camera point is then refused. A partner counts once,
    /// and is kept only when every link given for the pair holds ([`RigidLink`]). A marker
    /// with fewer rigid partners than this (among the links given) needs all of its
    /// partners, at least one: a marker with one partner needs that one, a marker with none
    /// cannot be confirmed. 0 checks nothing. Default 2.
    pub reacquire_links: usize,
    /// Added to the gate for each further frame since the marker was last found.
    /// Default 4.
    pub gate_growth: f64,
    /// The largest gate (pixels). Default 24.
    pub gate_max: f64,
    /// A marker is followed by its position (constant velocity, a gate that widens) for at
    /// most this many frames after it was last found in a camera; after that it counts as
    /// lost, and it is found again only with a check of its identity (`reacquire_links`).
    /// Longer gaps find more but let one wrong pick run on (measured: 7 frames gave 251
    /// points more than 5 px off in one Rat case). Default 3.
    pub max_gap: usize,
    /// How far back (frames) an earlier frame is looked for to find a lost marker again: the
    /// frame its rigid neighbours are moved from (`rigid_neighbours`), the depth given to a
    /// marker now seen by one camera only ([`closest_point_on_ray`]); also how long its
    /// template is kept. Default 300.
    pub depth_gap: usize,
    /// When a marker was last found more than one frame ago, its velocity is extrapolated
    /// over at most this many frames. Default 2.
    pub velocity_frames: usize,
    /// The velocity is taken from the latest point and the nearest earlier one within this
    /// many frames before it (divided by their distance in frames). Default 4.
    pub velocity_window: usize,
    /// The correlation search covers the gate plus this margin (pixels). Default 3.
    pub search_margin: f64,
    /// Candidates kept per marker and camera (the strongest local maxima). Default 6,
    /// at most 16.
    pub max_candidates: usize,
    /// Smallest correlation (zero-mean normalised, -1 to 1) of a candidate. Default 0.3.
    pub min_correlation: f64,
    /// Largest move (pixels) of the detection from the correlation peak it was run at;
    /// a larger move means the detection found another dot. Default 3.
    pub max_detection_shift: f64,
    /// A candidate whose contrast ([`dot_contrast`]) is below this fraction of the median
    /// contrast of the marker's recent points in that camera is dropped. Default 0.6.
    pub min_contrast_fraction: f64,
    /// Recent points whose contrast makes the median. Default 10.
    pub contrast_history: usize,
    /// When true, the contrast test (`min_contrast_fraction`) compares a candidate's contrast
    /// relative to the brightness round it ([`dot_contrast_relative`]: the contrast over the
    /// ring's mean grey level) with the median of the same quantity at the marker's recent
    /// points, instead of the contrast itself with its median. A bead removes a fraction of
    /// the light, so when something denser passes over a marker its absolute contrast falls
    /// with the brightness round it while the relative contrast holds. Measured on the
    /// guineafowl (camera 1 darkens round both markers in the still phase), the Rat and the
    /// pig: more points found on all three, at a trade-off (points where the pig's filled
    /// project has none; single Rat cases move). Two guards apply with it (the engine's
    /// module documentation, step 3): a floor on the absolute contrast, and the absolute
    /// test where another marker's dot may lie in the candidate's ring. Default true since
    /// 24 November 2026 (false before; the author's decision, "Yes, build both"): round the
    /// example pig project's marker 5 in camera 1 the background brightens at frames 7 to 15
    /// (ring grey level 90 to 159) and the marker's absolute contrast doubles; when the
    /// background returns to normal the marker fails the absolute test and stays lost, since
    /// its baseline is not updated while it is lost. With the relative test its gaps outside
    /// its crossing with marker 9 (frames 34 to 53) disappear in both directions. Nine-run
    /// means (`oracle/py/heldout.py all --jitter`, 24 November 2026, absolute against
    /// relative): Rat 68.9 % found and 158 points more than 5 px off against 68.2 % and 174;
    /// guineafowl 74.8 % found, 0 off, 10 extra against 82.5 %, 0, 21; pig 99.2 %, 0, 55
    /// against 99.7 %, 0, 68.
    pub contrast_relative: bool,
    /// Two candidates closer than this (pixels) in one camera are the same dot. Default 1.5.
    pub same_dot: f64,
    /// When not 0, a marker seen by one camera only is followed so for at most this many
    /// frames after its latest 3D point (or after the run's start); after that it must prove
    /// its identity as a marker found again (`reacquire_links`). 0 sets no limit. Default 0:
    /// on the Rat a limit of 10 cut the points more than 5 px off from 282 to 263 over twelve
    /// cases but lost 3 % of the points (and 3 % on the pig).
    pub single_max_frames: usize,
    /// Gate (pixels) round the prediction for a point seen by one camera only. Default 3.
    pub single_gate: f64,
    /// A point seen by one camera only is refused when another candidate dot of that
    /// marker lies within this distance (pixels) of it. Default 6.
    pub single_ambiguity: f64,
    /// Weight of `(reprojection error / max_reprojection_error)^2`. Default 1.
    pub weight_reprojection: f64,
    /// Weight of the mean over cameras of `(distance from prediction / gate)^2`. Default 1.
    pub weight_prediction: f64,
    /// Weight of the mean over cameras of `1 - correlation`. Default 1.
    pub weight_match: f64,
    /// Cost of each camera left out of a combination although it had candidates. Default 1.
    pub weight_missing: f64,
    /// A camera's dot is accepted only when every other feasible combination that puts a
    /// different dot in that camera costs at least this much more. Default 0.3.
    pub ambiguity_margin: f64,
    /// A dot already given to another marker is given to this one too when it lies within
    /// this distance (pixels) of this marker's prediction in that camera (two markers that
    /// overlap in one camera). 0 never shares. Default 1.5.
    pub share_radius: f64,
    /// Rigid links ([`RigidLink`]): a pair of markers counts as rigid once both have had 3D
    /// points at this many frames together. 0 turns the links off. Default 10.
    pub rigid_min_frames: usize,
    /// ... and the standard deviation of their distance is at most this fraction of its
    /// mean. Default 0.02.
    pub rigid_max_spread: f64,
    /// The tolerance of a link: the larger of this many standard deviations and
    /// `rigid_min_tolerance` times the mean distance. Default 4.
    pub rigid_sigmas: f64,
    /// See `rigid_sigmas`. Default 0.01.
    pub rigid_min_tolerance: f64,
    /// A combination that breaks some links to accepted markers is still used when it keeps
    /// at least this many times as many. Counted per partner: a partner given several links
    /// counts once, kept when all hold and broken when any breaks ([`RigidLink`]). Default 2.
    pub rigid_majority: f64,
    /// Frames before the start whose 3D points seed the pair statistics. Default 300.
    pub rigid_history: usize,
    /// A dot that the best combinations of several markers use goes to the one whose
    /// prediction in that camera is closer than this fraction of every other's distance,
    /// and otherwise to none of them. 0 gives it to the marker served first. Default 1 (the
    /// closest wins; a tie goes to none).
    pub contest_ratio: f64,
    /// Combinations tried per marker at most (and never more than [`MAX_COMBINATIONS`]);
    /// the candidate lists are shortened (weakest first) until the product fits. Default
    /// 4096.
    pub max_combinations: usize,
    /// A camera in which a marker is being found again (`CameraOptions::reacquire`) never
    /// shares a dot with another marker: a dot another marker holds is refused to it, even
    /// within `share_radius` of its prediction, and it never shares a dot whose contest it
    /// lost. Default true.
    pub reacquire_exclusive: bool,
    /// Claims: a camera in which a marker is being found again may not take a dot within this
    /// distance (pixels) of another marker's prediction in that camera (a marker followed
    /// there, accepted or not), unless the dot is clearly its own: its own prediction is
    /// within `claim_ratio` times the other's distance of it ([`Refusal::Claimed`]). Not a
    /// finite positive number: no claims. Default 6.
    pub claim_radius: f64,
    /// See `claim_radius`. Default 0.5.
    pub claim_ratio: f64,
    /// Two markers share one dot in a camera (`share_radius`) only when the caller says they
    /// touch there now ([`Overlap::touching`], [`choose_joint_overlapping`]); without that
    /// information ([`choose_joint_linked`]) the old rule applies. Default true.
    pub share_needs_overlap: bool,
    /// The engine's overlap state (`xma_engine::tracking::multi`, "Overlaps"): two markers of
    /// a run touch in a camera when their positions there are within the two dot radii plus
    /// this margin (pixels); the overlap is held from then until both are accepted on points
    /// further apart than that, or `overlap_max_frames` steps after they last touched.
    /// Default 1.
    pub overlap_margin: f64,
    /// See `overlap_margin`: the most steps an overlap is held after the two last touched.
    /// Default 30.
    pub overlap_max_frames: usize,
    /// A point accepted while its marker touches another in that camera (the engine's overlap
    /// state) is not used as a clean observation afterwards: no template or contrast is taken
    /// from it, and the 3D point of that frame is not used for a lost marker's depth, its
    /// rigid prediction or the rigid links' statistics. Default true.
    pub overlap_clean: bool,
    /// A marker lost in a camera while another camera still has it, without a 3D track or a
    /// position from its rigid neighbours, is looked for round the point on the other
    /// camera's ray that best keeps its rigid links to markers followed at the frame before
    /// ([`point_on_ray_keeping_links`], with two such partners or more), with the gate
    /// `gate_rigid`, instead of round the depth of its latest 3D point. Default true.
    pub ray_links: bool,
    /// Identity out of an overlap ([`choose_joint_overlapping`], 4.): for two markers given
    /// as overlapping in a camera, the assignment of the dots the choice gave them there is
    /// kept only when it costs at least this much less than the other one (the two dots
    /// exchanged; or, when one holds a dot and the other none and they no longer touch, the
    /// dot given to the other), counting each marker's own cost with what it was given in the
    /// other cameras (reprojection error, distance from its prediction, correlation) plus one
    /// for each rigid partner it breaks. Within this margin the camera is left empty for
    /// them ([`Refusal::Overlap`]); when the other assignment is cheaper by more, it is taken.
    /// Not a finite positive number: off. Default 0.5.
    pub overlap_decision: f64,
    /// Points are taken while their marker touches another in that camera (an [`Overlap`]
    /// given as touching, [`choose_joint_overlapping`]). False: a marker gets no point in a
    /// camera while it touches another marker there ([`Refusal::Overlap`]; the rest of its
    /// result made consistent, rule 4), the stricter answer to a point on a merged shape than
    /// `overlap_clean`. Default true.
    pub overlap_take: bool,
    /// After the overlaps' decisions (rule 4 of [`choose_joint_linked`]), a marker left with
    /// one camera keeps that point only when it is a point on its own: a feasible
    /// single-camera combination with exactly that pick exists (within `single_gate` of its
    /// prediction, no other candidate of it within `single_ambiguity`); otherwise it gets
    /// nothing there either ([`Refusal::Overlap`]). False: the point stays without those
    /// checks (a point the choice accepted only as part of a pair, which alone may be further
    /// than `single_gate` from the prediction). Default false: the check was asked for by
    /// review 17b and measured on the Rat (nine-run means, ledger A220): it takes away about
    /// 235 points a run that are within 2 px of the hand-finished ones and no point more than 5 px off
    /// (68.7 % found against 68.9 %, 159 and 158 off), so those points are kept; the same
    /// kind of point has arisen since round 2 where rule 3 refuses a marker's other camera.
    pub overlap_single_checked: bool,
    /// A run no longer erases an earlier point it could not find again. For a marker the run
    /// has had, a cell at the step's frame, in a camera the run tracks, that the run may
    /// write, that holds a point and that gets no point from the run is left exactly as it is
    /// (point, status, size), unless the run's own points contradict it, when it is reset to
    /// undefined as before: (a) the run wrote a point of the same marker at that frame in at
    /// least one other camera, and the 3D point triangulated from the older point together
    /// with those written points has a largest reprojection error above
    /// `max_reprojection_cap`, or gives no answer (the older point cannot be undistorted, or
    /// lies behind a camera: a point far outside the image; review 18 found the first version
    /// kept such points); with no point of the marker written in another camera, (a) does
    /// not apply; or (b) the run wrote a point of
    /// another marker in the same camera within `same_dot` of the older point, and does not
    /// hold the two as touching there at this step (the engine's overlap state). A cell
    /// whose stored coordinates are not finite is reset. The run never reads a kept point:
    /// the points it writes are the same either way. False: every such cell is reset (the
    /// rule until 24 November 2026).
    ///
    /// Why (the author, 24 November 2026, on the example pig project: "it's overwriting good
    /// tracking for no reason i can see"): measured on the rule before, a forward run from
    /// frame 1 of `pigDataset_local.xma` erased 17 of marker 5's earlier points, 14 of them
    /// within 2 px of the hand-finished project, and the backward run over that result erased
    /// frames 1 to 6 (the forward run's start points) and frame 69, all of them right; on the
    /// Rat (`heldout/f1_fwd.xma` forward, then backward from 449 over the result) the
    /// backward run erased 1,228 of the forward run's points, 1,195 of them within 2 px of the
    /// hand-finished points and 3 more than 5 px off, and left fewer points (61.4 % of the
    /// hand-finished ones) than
    /// the forward run alone (65.8 %).
    ///
    /// A run of one step repeated (the window's "next frame") starts each step from whatever
    /// the cell at its start frame holds, so a kept point becomes the next single step's
    /// start point, where one continuous run never reads it. Default true.
    pub keep_unconfirmed: bool,
    /// Rule (b) of `keep_unconfirmed`: an older point the run did not find again is reset
    /// when the run wrote a point of another marker in the same camera within `same_dot` of it
    /// (the run gave that dot to the other marker), unless the run holds the two as touching
    /// there at this step. False: rule (b) is off, and only rule (a) and the reset of a point
    /// that is not finite remove a kept point. Default true.
    ///
    /// Measured on the two Rat cases of `oracle/py/retrack.py` on 24 November 2026; the choice
    /// rests on those alone (on the pig and the guineafowl no point is removed with the rule
    /// on or off):
    /// - backward over a forward run of this kind (`rat_f1`): the rule removes 32 of the 3,359
    ///   older points the run could not find again; 29 of them were within 2 px of the
    ///   hand-finished points and none more than 5 px off: two markers that really share a dot
    ///   in that camera (the hand-finished points have them 0 to 1.5 px apart), where the run gave the dot to one and
    ///   has the other in no camera at that frame. Off: none removed.
    /// - over the plain tracker's result (`rat_over_plain`): the rule removes 23, of which 20
    ///   are more than 5 px off the hand-finished points and 2 within 2 px; after the run 23 points are
    ///   more than 5 px off with the rule, 42 without it (14 when every unconfirmed point is
    ///   erased, `keep_unconfirmed` off, which also erases 252 points within 2 px).
    /// - the exemption widened to any pair the run holds in its overlap state (touching now
    ///   or not) was measured and not kept: 17 of the 32 still removed (16 within 2 px), and
    ///   40 points more than 5 px off over the plain tracker.
    ///
    /// On or off is a trade between those two cases (good points on a shared dot lost, against
    /// points slid onto another marker's dot kept), and the choice is the author's.
    pub keep_dot_rule: bool,
    /// Two touching dots are fitted apart (the engine's module documentation, step 3, "Two
    /// touching dots"; [`crate::dot_fit::split_touching_dots`]): where the marker's detection
    /// has joined two dots into one shape and returned the shape's centre (between the two),
    /// two Gaussian dots of the marker's learned width are fitted to the picture there, and
    /// when two clearly explain it better than one, both fitted centres are candidates in
    /// place of the detection's point. Only for the dot-like detection methods, and only once
    /// the run has learned the marker's dot width in that camera from its clean points. A
    /// point written from such a candidate has size -1 and counts as a merged point (no
    /// template, contrast or width is taken from it; its 3D point gives no depth, rigid
    /// prediction or rigid statistics). Default true: on by default since 26 November 2026
    /// (the author's decision; parity ledger A223, A224, A225).
    pub split_touching: bool,
    /// `split_touching`: [`crate::dot_fit::SplitRules::min_depth_ratio`]. Default 0.35.
    pub split_depth_ratio: f64,
    /// `split_touching`: [`crate::dot_fit::SplitRules::max_rms_ratio`]. Default 0.8.
    pub split_rms_ratio: f64,
    /// `split_touching`: [`crate::dot_fit::SplitRules::min_apart`] (pixels). Default 1.5.
    pub split_min_apart: f64,
    /// `split_touching`: [`crate::dot_fit::SplitRules::max_move`] (pixels). Default 4.
    pub split_max_move: f64,
    /// `split_touching`: [`crate::dot_fit::SplitRules::reach_widths`]. Default 3.
    pub split_reach_widths: f64,
    /// `split_touching`: [`crate::dot_fit::SplitRules::reach_margin`] (pixels). Default 4.
    pub split_reach_margin: f64,
    /// A marker being found again (a camera flagged as finding it again) is recognised by the
    /// run's own distance statistics (the engine's module documentation, step 4): for every
    /// other marker whose pair with it the run has measured at `rigid_min_frames` frames or
    /// more, a link with the pair's mean distance and a tolerance of the larger of
    /// `rigid_sigmas` times its spread and `rigid_min_tolerance` of the mean, whether or not
    /// the pair counts as rigid; a rigid prediction makes no reference-frame link for such a
    /// pair (a pair measured at fewer frames keeps its reference-frame link). The links go to
    /// every such marker of the run, near or far. The check is unchanged (`reacquire_links`
    /// partners kept, none broken), and the links of the markers followed are unchanged. With
    /// `rigid_min_frames` 0 no statistics are kept and this does nothing. Default true: on by default since 26 November 2026
    /// (the author's decision; parity ledger A223, A224, A225).
    pub stat_links: bool,
    /// Where a lost marker has no rigid prediction (and no camera of its own still follows
    /// it), it is placed at the point that best keeps its pairs' mean distances to four or
    /// more markers followed in 3D ([`point_from_distances`]; the engine's module
    /// documentation, step 2), searched for round that point's projection with the gate
    /// `gate_rigid`, and recognised by the links of `stat_links` to them. With
    /// `rigid_min_frames` 0 no statistics are kept and this does nothing. Default true: on by default since 26 November 2026
    /// (the author's decision; parity ledger A223, A224, A225).
    pub distance_placement: bool,
}

impl Default for MultiTrackParams {
    fn default() -> Self {
        MultiTrackParams {
            max_reprojection_error: 1.5,
            gate: 5.0,
            reprojection_factor: 2.5,
            max_reprojection_cap: 3.0,
            rigid_neighbours: 5,
            rigid_fit_tolerance: 0.02,
            reacquire_dots: true,
            gate_hold: 10.0,
            gate_start: 40.0,
            start_confirm_near: false,
            gate_rigid: 6.0,
            reacquire_links: 2,
            gate_growth: 4.0,
            gate_max: 24.0,
            max_gap: 3,
            depth_gap: 300,
            velocity_frames: 2,
            velocity_window: 4,
            search_margin: 3.0,
            max_candidates: 6,
            min_correlation: 0.3,
            max_detection_shift: 3.0,
            min_contrast_fraction: 0.6,
            contrast_history: 10,
            contrast_relative: true,
            same_dot: 1.5,
            single_max_frames: 0,
            single_gate: 3.0,
            single_ambiguity: 6.0,
            weight_reprojection: 1.0,
            weight_prediction: 1.0,
            weight_match: 1.0,
            weight_missing: 1.0,
            ambiguity_margin: 0.3,
            share_radius: 1.5,
            contest_ratio: 1.0,
            rigid_min_frames: 10,
            rigid_max_spread: 0.02,
            rigid_sigmas: 4.0,
            rigid_min_tolerance: 0.01,
            rigid_majority: 2.0,
            rigid_history: 300,
            max_combinations: 4096,
            reacquire_exclusive: true,
            claim_radius: 6.0,
            claim_ratio: 0.5,
            share_needs_overlap: true,
            overlap_margin: 1.0,
            overlap_max_frames: 30,
            overlap_clean: true,
            ray_links: true,
            overlap_decision: 0.5,
            overlap_take: true,
            overlap_single_checked: false,
            keep_unconfirmed: true,
            keep_dot_rule: true,
            split_touching: true,
            split_depth_ratio: 0.35,
            split_rms_ratio: 0.8,
            split_min_apart: 1.5,
            split_max_move: 4.0,
            split_reach_widths: 3.0,
            split_reach_margin: 4.0,
            stat_links: true,
            distance_placement: true,
        }
    }
}

impl MultiTrackParams {
    /// The gate radius for a marker last found `frames_since` frames before the frame
    /// tracked to (1 = the frame just before): `gate` (or `gate_hold` when no velocity was
    /// known) `+ gate_growth (frames_since - 1)`, at most `gate_max`. New in this program.
    #[must_use]
    pub fn gate_for(&self, frames_since: usize, velocity: bool) -> f64 {
        let base = if velocity { self.gate } else { self.gate_hold };
        let extra = frames_since.saturating_sub(1) as f64 * self.gate_growth;
        let g = base + extra;
        if g.is_nan() {
            base
        } else {
            g.min(self.gate_max)
        }
    }

    /// The numbers of the fit of two touching dots (`split_touching`).
    #[must_use]
    pub fn split_rules(&self) -> crate::dot_fit::SplitRules {
        crate::dot_fit::SplitRules {
            min_depth_ratio: self.split_depth_ratio,
            max_rms_ratio: self.split_rms_ratio,
            min_apart: self.split_min_apart,
            max_move: self.split_max_move,
            reach_widths: self.split_reach_widths,
            reach_margin: self.split_reach_margin,
        }
    }

    /// The candidate count used: `max_candidates` clamped to 1..=16.
    fn candidates_cap(&self) -> usize {
        self.max_candidates.clamp(1, 16)
    }
}

/// A correlation surface: `values` at column `i`, row `j` is the correlation of the template
/// centred on pixel `(off_x + i, off_y + j)`.
#[derive(Clone, Debug, PartialEq)]
pub struct NccSurface {
    pub off_x: i64,
    pub off_y: i64,
    pub values: ImageF32,
}

/// The template of a marker for [`ncc_surface`]: the `2 half + 1` square round `(x, y)`
/// (sub-pixel, bilinear, edges replicated), as `f32` on the image's grey scale. `None` when
/// `half` is not positive or beyond the window bound of the image, or the centre is not
/// finite. New in this program.
#[must_use]
pub fn cut_ncc_template<S: GreySource + ?Sized>(
    image: &S,
    x: f64,
    y: f64,
    half: i32,
) -> Option<ImageF32> {
    if half <= 0
        || i64::from(half) > max_window_half(image.width(), image.height())
        || image.width() == 0
        || image.height() == 0
        || !(x.is_finite() && y.is_finite())
        || x.abs() > 1e7
        || y.abs() > 1e7
    {
        return None;
    }
    let t = get_sub_image_32f(image, half, x, y);
    (!t.is_empty()).then_some(t)
}

/// The largest template side [`ncc_surface`] takes (a marker of the largest mean size, 50,
/// has a template of 107).
pub const MAX_TEMPLATE_SIDE: usize = 129;

/// The most multiply-adds one [`ncc_surface`] does (`(2 radius + 1)^2 * side^2`): 2e8.
pub const MAX_NCC_WORK: u64 = 200_000_000;

/// The most combinations [`choose_joint`] tries per marker, whatever `max_combinations`
/// says.
pub const MAX_COMBINATIONS: usize = 65_536;

/// The zero-mean normalised cross-correlation of `template` (square, odd side) with
/// `image` at every whole pixel within `radius` of `centre` (rounded): -1 to 1, 0 where
/// the image patch or the template is flat. `None` when the template is empty, not an odd
/// square or wider than [`MAX_TEMPLATE_SIDE`], `radius` is negative or beyond the window
/// bound, or the centre is not finite. The work is bounded: `radius` is clamped to the
/// image's larger side and then reduced until the surface costs at most [`MAX_NCC_WORK`]
/// multiply-adds (the surface returned is then smaller than asked for). New in this
/// program.
#[must_use]
pub fn ncc_surface<S: GreySource + ?Sized>(
    image: &S,
    template: &ImageF32,
    centre: (f64, f64),
    radius: i32,
) -> Option<NccSurface> {
    let tw = template.width();
    if tw == 0 || tw != template.height() || tw.is_multiple_of(2) || radius < 0 {
        return None;
    }
    if image.width() == 0 || image.height() == 0 || tw > MAX_TEMPLATE_SIDE {
        return None;
    }
    // Bounded work: the radius is at most the image's larger side, and is reduced until
    // `(2 radius + 1)^2 * side^2` is at most `MAX_NCC_WORK`.
    let mut radius =
        radius.min(i32::try_from(image.width().max(image.height())).unwrap_or(i32::MAX));
    while radius > 0 {
        let side = 2 * radius as u64 + 1;
        if side * side * (tw as u64) * (tw as u64) <= MAX_NCC_WORK {
            break;
        }
        radius -= 1 + radius / 8;
        radius = radius.max(0);
    }
    let half_t = (tw / 2) as i64;
    let bound = max_window_half(image.width(), image.height());
    let size = i64::from(radius) + half_t;
    if size > bound || !(centre.0.is_finite() && centre.1.is_finite()) {
        return None;
    }
    if centre.0.abs() > 1e7 || centre.1.abs() > 1e7 {
        return None;
    }
    let cx = centre.0.round() as i64;
    let cy = centre.1.round() as i64;
    let off_x = cx - size;
    let off_y = cy - size;
    let window = get_sub_image_at_32f(
        image,
        i32::try_from(size).ok()?,
        i32::try_from(off_x).ok()?,
        i32::try_from(off_y).ok()?,
    );
    if window.is_empty() {
        return None;
    }
    let n = (tw * tw) as f64;
    let tmean = template.data().iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let tz: Vec<f64> = template
        .data()
        .iter()
        .map(|&v| f64::from(v) - tmean)
        .collect();
    let tss: f64 = tz.iter().map(|v| v * v).sum();
    let ww = window.width();
    // Integral images of the window and its square.
    let mut s1 = vec![0.0f64; (ww + 1) * (ww + 1)];
    let mut s2 = vec![0.0f64; (ww + 1) * (ww + 1)];
    for y in 0..ww {
        let mut r1 = 0.0;
        let mut r2 = 0.0;
        for x in 0..ww {
            let v = f64::from(window.at(x, y));
            r1 += v;
            r2 += v * v;
            s1[(y + 1) * (ww + 1) + x + 1] = s1[y * (ww + 1) + x + 1] + r1;
            s2[(y + 1) * (ww + 1) + x + 1] = s2[y * (ww + 1) + x + 1] + r2;
        }
    }
    let rect = |s: &[f64], x0: usize, y0: usize| {
        let x1 = x0 + tw;
        let y1 = y0 + tw;
        s[y1 * (ww + 1) + x1] - s[y0 * (ww + 1) + x1] - s[y1 * (ww + 1) + x0]
            + s[y0 * (ww + 1) + x0]
    };
    let side = 2 * radius as usize + 1;
    let mut values = ImageF32::new(side, side);
    for j in 0..side {
        for i in 0..side {
            let sum = rect(&s1, i, j);
            let sq = rect(&s2, i, j);
            let var = sq - sum * sum / n;
            let mut cross = 0.0;
            for ty in 0..tw {
                let row = &window.row(j + ty)[i..i + tw];
                let trow = &tz[ty * tw..(ty + 1) * tw];
                for (a, b) in row.iter().zip(trow) {
                    cross += f64::from(*a) * b;
                }
            }
            let den = (var * tss).sqrt();
            let v = if den > 1e-9 && den.is_finite() {
                (cross / den).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            values.set(i, j, v as f32);
        }
    }
    Some(NccSurface {
        off_x: off_x + half_t,
        off_y: off_y + half_t,
        values,
    })
}

/// The local maxima of a surface (at least every one of their 8 neighbours; on a plateau
/// the first in row order) with a value of at least `min_value`, strongest first, at most
/// `max_count`. Returns `(column, row, value)`. New in this program.
#[must_use]
pub fn local_maxima(
    surface: &ImageF32,
    min_value: f32,
    max_count: usize,
) -> Vec<(usize, usize, f32)> {
    let (w, h) = (surface.width(), surface.height());
    let mut out: Vec<(usize, usize, f32)> = Vec::new();
    if max_count == 0 {
        return out;
    }
    for y in 0..h {
        for x in 0..w {
            let v = surface.at(x, y);
            if v.is_nan() || v < min_value {
                continue;
            }
            let mut is_max = true;
            'n: for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let u = surface.at(nx as usize, ny as usize);
                    // Strictly greater neighbours, and equal ones earlier in row order, win.
                    let earlier = (ny, nx) < (y as i64, x as i64);
                    if u > v || (u == v && earlier) {
                        is_max = false;
                        break 'n;
                    }
                }
            }
            if is_max {
                out.push((x, y, v));
            }
        }
    }
    out.sort_by(|a, b| b.2.total_cmp(&a.2).then((a.1, a.0).cmp(&(b.1, b.0))));
    out.truncate(max_count);
    out
}

/// How much darker (for `dark`, else lighter) a dot of radius `radius` at `centre` is than
/// its surroundings: the mean grey value of the ring from `radius + 1.5` to `radius + 3.5`
/// pixels minus the mean of the disk within `max(1, 0.75 radius)` pixels (whole pixels,
/// edges replicated), negated for a light dot. `None` when the centre or radius is not
/// finite, the radius is not positive or above 50, or the image is empty. New in this
/// program.
#[must_use]
pub fn dot_contrast<S: GreySource + ?Sized>(
    image: &S,
    centre: (f64, f64),
    radius: f64,
    dark: bool,
) -> Option<f64> {
    let (ring, disk) = ring_and_disk(image, centre, radius)?;
    let c = ring - disk;
    Some(if dark { c } else { -c })
}

/// [`dot_contrast`] divided by the mean grey value of its ring (at least 1): how much of the
/// light round the dot the dot takes away (for `dark`; for a light dot, how much it adds).
/// It stays level when the whole neighbourhood darkens or brightens by a factor, as when
/// something denser passes over a bead. `None` as for [`dot_contrast`]. Bounded as it is.
/// New in this program.
#[must_use]
pub fn dot_contrast_relative<S: GreySource + ?Sized>(
    image: &S,
    centre: (f64, f64),
    radius: f64,
    dark: bool,
) -> Option<f64> {
    let (ring, disk) = ring_and_disk(image, centre, radius)?;
    let c = (ring - disk) / ring.max(1.0);
    Some(if dark { c } else { -c })
}

/// Whether `from` lies on the dark shape at `to` (for `dark`; light otherwise), joined to
/// it: `from` is darker than the surroundings by at least a quarter of the shape's depth,
/// and along the straight segment between them no sample is brighter than `from` itself or
/// than half-way between the shape's darkness and its surroundings, whichever is brighter.
/// New in this program.
///
/// - The shape's darkness `d`: the darker of the 3 x 3 means at `from` and at `to`.
/// - Its surroundings `s`: the 75th percentile (the brighter quarter excluded) of the
///   pixels 14 to 20 px from `to`. A percentile, not a mean: round a point inside an
///   elongated marker (a conical point is up to about 28 px long) the ring crosses the
///   marker, and the ground itself is uneven.
/// - The path may be as bright as `from`: a point at the end of a cone is less dark than
///   the middle, and the path from it inwards only darkens; a bright gap between `from` and
///   a separate dot is brighter than both.
/// - Samples are 3 x 3 means at whole-pixel steps along the segment.
///
/// Measured on real images (round 7): on the guineafowl a point moved 3 to 8 px from the
/// hand-finished point within its dark shape was joined to it in 1,290 of 1,291 cases; on the
/// Rat, pairs of different markers' points were called one shape in 28 of 1,230 pairs 10 to
/// 20 px apart, 83 of 257 pairs 5 to 10 px apart and 138 of 140 pairs closer than 5 px
/// (merged blobs).
///
/// False when the shape is not darker than its surroundings by at least 5 grey levels (on
/// the 8-bit scale), when a point or value is not finite, when the segment is longer than
/// 64 px, or the image is empty. Edges are replicated. Bounded: at most 65 samples and one
/// ring of at most 1,681 pixels.
#[must_use]
pub fn on_one_shape<S: GreySource + ?Sized>(
    image: &S,
    from: (f64, f64),
    to: (f64, f64),
    dark: bool,
) -> bool {
    let (w, h) = (image.width(), image.height());
    let finite =
        |p: (f64, f64)| p.0.is_finite() && p.1.is_finite() && p.0.abs() < 1e7 && p.1.abs() < 1e7;
    if w == 0 || h == 0 || !finite(from) || !finite(to) {
        return false;
    }
    let len = (to.0 - from.0).hypot(to.1 - from.1);
    if len.is_nan() || len > 64.0 {
        return false;
    }
    let sign = if dark { 1.0 } else { -1.0 };
    let at = |x: i64, y: i64| -> f64 {
        let x = x.clamp(0, w as i64 - 1) as usize;
        let y = y.clamp(0, h as i64 - 1) as usize;
        sign * f64::from(image.value(y * w + x))
    };
    let mean3 = |p: (f64, f64)| -> f64 {
        let (cx, cy) = (p.0.round() as i64, p.1.round() as i64);
        let mut sum = 0.0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                sum += at(cx + dx, cy + dy);
            }
        }
        sum / 9.0
    };
    let d = mean3(from).min(mean3(to));
    let (cx, cy) = (to.0.round() as i64, to.1.round() as i64);
    let mut ring: Vec<f64> = Vec::with_capacity(1681);
    for dy in -20i64..=20 {
        for dx in -20i64..=20 {
            let r = (dx as f64).hypot(dy as f64);
            if (14.0..=20.0).contains(&r) {
                ring.push(at(cx + dx, cy + dy));
            }
        }
    }
    if ring.is_empty() {
        return false;
    }
    ring.sort_by(f64::total_cmp);
    let s = ring[(ring.len() * 3) / 4];
    if (s - d).is_nan() || s - d < 5.0 {
        return false;
    }
    // `from` must be on the shape (at least a quarter of its depth); the path may be as
    // bright as `from` itself (the end of a cone is less dark than its middle) or as the
    // half-way level, whichever is brighter.
    let f0 = mean3(from);
    if f0 > s - (s - d) / 4.0 {
        return false;
    }
    let level = (s - (s - d) / 2.0).max(f0);
    let n = len.ceil() as usize;
    (0..=n).all(|i| {
        let t = if n == 0 { 0.0 } else { i as f64 / n as f64 };
        mean3((from.0 + t * (to.0 - from.0), from.1 + t * (to.1 - from.1))) <= level
    })
}

/// The mean grey values of [`dot_contrast`]'s ring and disk, `None` as for it.
fn ring_and_disk<S: GreySource + ?Sized>(
    image: &S,
    centre: (f64, f64),
    radius: f64,
) -> Option<(f64, f64)> {
    let (w, h) = (image.width(), image.height());
    if w == 0
        || h == 0
        || !(centre.0.is_finite() && centre.1.is_finite())
        || !(radius > 0.0 && radius <= 50.0)
        || centre.0.abs() > 1e7
        || centre.1.abs() > 1e7
    {
        return None;
    }
    let inner = (0.75 * radius).max(1.0);
    let (r0, r1) = (radius + 1.5, radius + 3.5);
    let reach = r1.ceil() as i64;
    let (cx, cy) = (centre.0.round() as i64, centre.1.round() as i64);
    let (mut si, mut ni, mut so, mut no) = (0.0f64, 0usize, 0.0f64, 0usize);
    for dy in -reach..=reach {
        for dx in -reach..=reach {
            let x = (cx + dx).clamp(0, w as i64 - 1) as usize;
            let y = (cy + dy).clamp(0, h as i64 - 1) as usize;
            let d = ((cx + dx) as f64 - centre.0).hypot((cy + dy) as f64 - centre.1);
            let v = f64::from(image.value(y * w + x));
            if d <= inner {
                si += v;
                ni += 1;
            } else if d >= r0 && d <= r1 {
                so += v;
                no += 1;
            }
        }
    }
    if ni == 0 || no == 0 {
        return None;
    }
    Some((so / no as f64, si / ni as f64))
}

/// The dots round `centre`: the local maxima of [`dot_contrast`] (radius `dot_radius`,
/// polarity `dark`) over the whole pixels within `radius` of `centre` (rounded), at least
/// `min_value`, strongest first, at most `max_count`; as `(x, y, contrast)`. A candidate
/// source that does not depend on a template: how a marker whose appearance has changed is
/// found again. `radius` is clamped to 0..=64 and `dot_radius` to 0.5..=10, which bounds
/// the work; empty for a centre that is not finite or an empty image. New in this program.
#[must_use]
pub fn contrast_maxima<S: GreySource + ?Sized>(
    image: &S,
    centre: (f64, f64),
    radius: i32,
    dot_radius: f64,
    dark: bool,
    min_value: f64,
    max_count: usize,
) -> Vec<(f64, f64, f64)> {
    if image.width() == 0
        || image.height() == 0
        || !(centre.0.is_finite() && centre.1.is_finite())
        || centre.0.abs() > 1e7
        || centre.1.abs() > 1e7
        || max_count == 0
    {
        return Vec::new();
    }
    let r = radius.clamp(0, 64);
    let dr = if dot_radius.is_nan() {
        2.5
    } else {
        dot_radius.clamp(0.5, 10.0)
    };
    let (cx, cy) = (centre.0.round() as i64, centre.1.round() as i64);
    let side = (2 * r + 1) as usize;
    let mut surf = ImageF32::new(side, side);
    for j in 0..side {
        for i in 0..side {
            let x = (cx - i64::from(r) + i as i64) as f64;
            let y = (cy - i64::from(r) + j as i64) as f64;
            let v = dot_contrast(image, (x, y), dr, dark).unwrap_or(f64::NEG_INFINITY);
            surf.set(i, j, v as f32);
        }
    }
    let floor = if min_value.is_nan() {
        0.0
    } else {
        min_value as f32
    };
    local_maxima(&surf, floor, max_count)
        .into_iter()
        .map(|(i, j, v)| {
            (
                (cx - i64::from(r) + i as i64) as f64,
                (cy - i64::from(r) + j as i64) as f64,
                f64::from(v),
            )
        })
        .collect()
}

/// The point on the viewing ray of image point `pixel` (raw image pixels) of a camera that
/// is closest to `x`: how a marker seen by one camera only is given a 3D position, at the
/// depth of an earlier 3D point. `None` when the point cannot be undistorted, the camera
/// matrix is singular, or a value is not finite. New in this program.
#[must_use]
pub fn closest_point_on_ray(
    model: &CameraModel,
    pose: &RigidTransform,
    pixel: Point2,
    x: Point3,
) -> Option<Point3> {
    if !(finite2(pixel) && x.x.is_finite() && x.y.is_finite() && x.z.is_finite()) {
        return None;
    }
    let u = model.try_undistort_point(pixel, true, true, true)?;
    if !finite2(u) {
        return None;
    }
    let r = crate::camera::rodrigues_to_matrix(&pose.rotation);
    let k_inv = model.intrinsics.camera_matrix.try_inverse()?;
    let d = r.transpose() * (k_inv * crate::Vector3::new(u.x, u.y, 1.0));
    let centre = -(r.transpose() * pose.translation);
    let dd = d.dot(&d);
    if !(dd > 0.0 && dd.is_finite()) {
        return None;
    }
    let t = (x.coords - centre).dot(&d) / dd;
    let p = Point3::from(centre + d * t);
    (p.x.is_finite() && p.y.is_finite() && p.z.is_finite()).then_some(p)
}

/// The point on the viewing ray of image point `pixel` (raw image pixels) of a camera that
/// best keeps a marker's rigid links to partners whose 3D positions are known: each of
/// `partners` is `(position, distance, tolerance)`, and the point minimises the sum of
/// `((|x - position| - distance) / tolerance)^2` over the stretch of the ray where a point
/// can be within its distance of a partner (sampled at 400 steps, then refined; at most
/// 2e6 long, else `None`); `anchor` only places the ray's origin of that search (the point
/// closest to it). `Some` only with two partners or more, when every one of them is then
/// kept (within its tolerance), and when no point of the ray further than twice the largest
/// tolerance (and two steps) from it keeps them all too (two places the links allow): where
/// a marker seen by one camera only is, when its depth
/// from an earlier 3D point is not to be trusted. `None` also when the point cannot be
/// undistorted, the camera matrix is singular, or a value is not finite (partners with a
/// value that is not finite, or a tolerance not positive, are left out). Work is in
/// proportion to the number of partners. New in this program.
#[must_use]
pub fn point_on_ray_keeping_links(
    model: &CameraModel,
    pose: &RigidTransform,
    pixel: Point2,
    anchor: Point3,
    partners: &[(Point3, f64, f64)],
) -> Option<Point3> {
    let partners: Vec<(Point3, f64, f64)> = partners
        .iter()
        .copied()
        .filter(|(y, l, tol)| {
            y.x.is_finite()
                && y.y.is_finite()
                && y.z.is_finite()
                && l.is_finite()
                && *l >= 0.0
                && tol.is_finite()
                && *tol > 0.0
        })
        .collect();
    if partners.len() < 2 {
        return None;
    }
    let base = closest_point_on_ray(model, pose, pixel, anchor)?;
    // The ray's direction: towards the closest point from the camera centre.
    let r = crate::camera::rodrigues_to_matrix(&pose.rotation);
    let centre = -(r.transpose() * pose.translation);
    let d = base.coords - centre;
    let n = d.norm();
    if !(n.is_finite() && n > 0.0) {
        return None;
    }
    let d = d / n;
    // The stretch of the ray where a point can be within its distance of a partner: round
    // each partner's foot on the ray, its distance either side.
    let (lo, hi) =
        partners
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (y, l, _)| {
                let t = (y.coords - base.coords).dot(&d);
                (lo.min(t - l), hi.max(t + l))
            });
    if !(lo.is_finite() && hi.is_finite() && hi > lo && hi - lo <= 2e6) {
        return None;
    }
    let mid = 0.5 * (lo + hi);
    let reach = 0.5 * (hi - lo);
    let at = |t: f64| Point3::from(base.coords + d * t);
    let cost = |t: f64| -> f64 {
        let x = at(t);
        partners
            .iter()
            .map(|(y, l, tol)| {
                let e = ((x - y).norm() - l) / tol;
                e * e
            })
            .sum()
    };
    const STEPS: usize = 400;
    let step = 2.0 * reach / STEPS as f64;
    let (mut best_t, mut best) = (0.0, f64::INFINITY);
    for i in 0..=STEPS {
        let t = mid - reach + step * i as f64;
        let v = cost(t);
        if v < best {
            best = v;
            best_t = t;
        }
    }
    if !best.is_finite() {
        return None;
    }
    // Refine within one step on either side (ternary search).
    let (mut lo, mut hi) = (best_t - step, best_t + step);
    for _ in 0..60 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if cost(m1) <= cost(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    let t = 0.5 * (lo + hi);
    let x = if cost(t) <= best { at(t) } else { at(best_t) };
    let keeps = |x: Point3| {
        partners
            .iter()
            .all(|(y, l, tol)| ((x - y).norm() - l).abs() <= *tol)
    };
    // Two separate stretches of the ray that both keep every link: which is the marker's
    // cannot be told.
    let apart = 2.0 * partners.iter().map(|p| p.2).fold(0.0f64, f64::max) + 2.0 * step;
    let other = (0..=STEPS)
        .map(|i| mid - reach + step * i as f64)
        .any(|t2| (t2 - best_t).abs() > apart && keeps(at(t2)));
    (keeps(x) && !other && x.x.is_finite() && x.y.is_finite() && x.z.is_finite()).then_some(x)
}

/// The most Gauss-Newton steps [`point_from_distances`] takes.
const DISTANCE_ITERATIONS: usize = 100;

/// The most times [`point_from_distances`] halves a step that does not lower the cost.
const DISTANCE_HALVINGS: usize = 40;

/// The point that best keeps given distances to known points: each of `links` is
/// `(position, distance, spread)`, and the point minimises the sum of
/// `((|x - position| - distance) / spread)^2` (weighted least squares), found by Gauss-Newton
/// from `start` (at most [`DISTANCE_ITERATIONS`] steps, each halved while it does not lower
/// the cost, the 3 x 3 normal equations solved through [`crate::linalg`]). Four distances
/// or more are needed (three leave two points, mirror images in the partners' plane; with
/// four or more in one plane the two remain, and the one on `start`'s side is found).
///
/// `None` for fewer than four links, a value that is not finite, a distance that is negative
/// or a spread that is not positive, a start on a partner, a system that is singular (the
/// partners on one line, or coincident) or nearly so (its smallest eigenvalue below 1e-12 of
/// its largest), or no convergence (a step still above 1e-9 of the scale of the problem after
/// the last iteration). Work is in proportion to the number of links times the iterations;
/// nothing is allocated in proportion to the links. New in this program (sixth round).
#[must_use]
pub fn point_from_distances(start: Point3, links: &[(Point3, f64, f64)]) -> Option<Point3> {
    let finite3 = |p: &Point3| p.x.is_finite() && p.y.is_finite() && p.z.is_finite();
    if links.len() < 4 || !finite3(&start) {
        return None;
    }
    let mut scale: f64 = 0.0;
    for (y, d, s) in links {
        if !(finite3(y) && d.is_finite() && *d >= 0.0 && s.is_finite() && *s > 0.0) {
            return None;
        }
        scale = scale.max(*d).max((y - start).norm());
    }
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    // The cost, and the normal equations (J^T J, J^T e) at `x`; `None` on a partner or
    // where a value overflows.
    let at = |x: Point3| -> Option<(f64, [f64; 9], [f64; 3])> {
        let (mut cost, mut a, mut g) = (0.0, [0.0f64; 9], [0.0f64; 3]);
        for (y, d, s) in links {
            let v = x - y;
            let r = v.norm();
            if !(r.is_finite() && r > 0.0) {
                return None;
            }
            let j = [v.x / r / s, v.y / r / s, v.z / r / s];
            let e = (r - d) / s;
            cost += e * e;
            for i in 0..3 {
                g[i] += j[i] * e;
                for k in 0..3 {
                    a[i * 3 + k] += j[i] * j[k];
                }
            }
        }
        (cost.is_finite() && a.iter().chain(&g).all(|v| v.is_finite())).then_some((cost, a, g))
    };
    let mut x = start;
    let (mut cost, mut a, mut g) = at(x)?;
    for _ in 0..DISTANCE_ITERATIONS {
        // Nearly singular: the partners on one line (any point on a circle round it keeps
        // the distances), or coincident.
        let (w, _) = crate::linalg::eigen_jacobi(&a, 3);
        let (lo, hi) = w.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
        if !(lo.is_finite() && hi.is_finite() && hi > 0.0 && lo > 1e-12 * hi) {
            return None;
        }
        let inv = crate::linalg::invert(&nalgebra::Matrix3::from_row_slice(&a));
        if !inv.ok {
            return None;
        }
        let step = -(inv.x * nalgebra::Vector3::new(g[0], g[1], g[2]));
        let step = nalgebra::Vector3::new(step[0], step[1], step[2]);
        if !(step.x.is_finite() && step.y.is_finite() && step.z.is_finite()) {
            return None;
        }
        // Halve the step while it does not lower the cost.
        let mut t = 1.0;
        let mut next = None;
        for _ in 0..DISTANCE_HALVINGS {
            let y = Point3::from(x.coords + step * t);
            if let Some(n) = at(y)
                && n.0 <= cost
            {
                next = Some((y, n));
                break;
            }
            t *= 0.5;
        }
        let Some((y, n)) = next else {
            // No step lowers the cost: at a minimum (to rounding) when the step is tiny.
            return (step.norm() <= 1e-9 * scale).then_some(x);
        };
        let moved = (y - x).norm();
        x = y;
        (cost, a, g) = n;
        if moved <= 1e-12 * scale {
            return finite3(&x).then_some(x);
        }
    }
    None
}

/// One camera as [`choose_joint`] reads it.
#[derive(Clone, Debug)]
pub struct JointCamera<'a> {
    pub model: &'a CameraModel,
    /// The camera's pose for the trial (world to camera).
    pub pose: &'a RigidTransform,
    /// `K [R | t]` for `pose`.
    pub projection: Matrix3x4,
}

/// A candidate position of a marker in one camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// The refined position in image pixels.
    pub point: Point2,
    /// The correlation of the marker's template there (-1 to 1; 1 for a fixed point).
    pub correlation: f64,
}

/// What one marker offers in one camera.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CameraOptions {
    /// The predicted position, if any.
    pub prediction: Option<Point2>,
    /// The gate radius round the prediction (pixels).
    pub gate: f64,
    /// A point that must be used as it is (a point set by hand); `candidates` is then
    /// ignored.
    pub fixed: Option<Point2>,
    /// The marker is being found again in this camera after it was lost: a combination using
    /// it must keep `reacquire_links` rigid links ([`choose_joint_linked`]).
    pub reacquire: bool,
    /// The candidates; the camera may also be left out.
    pub candidates: Vec<Candidate>,
    /// The ordinary gate (pixels) when `gate` is a wider search's (the engine's wider first
    /// search, `MultiTrackParams::gate_start`): a dot further than this from the prediction
    /// is *far*. Feasibility uses `gate`; the cost's distance from the prediction is
    /// measured against this gate; and the marker's combinations are ranked in classes
    /// ([`choose_joint_linked`]): a combination in two cameras or more without a far dot
    /// first, then one in two cameras or more with a far dot, then one in a single camera,
    /// so a wider search never costs a point the ordinary one would have accepted in two
    /// cameras or more, and a point seen by one camera only gives way to a far pair that
    /// triangulates. Not finite, not positive or not below `gate` (the default, 0): no dot is
    /// far, and the cost is measured against `gate`.
    pub near_gate: f64,
}

/// One marker's input to [`choose_joint`]: one entry per camera (missing entries count as
/// cameras with nothing to offer).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MarkerOptions {
    pub cameras: Vec<CameraOptions>,
    /// This marker's largest reprojection error (pixels), when it is to differ from
    /// `max_reprojection_error` (a marker whose points never agree as well as others').
    /// Not finite or not positive: the parameter's.
    pub reprojection_limit: Option<f64>,
}

/// What a camera of a marker's choice holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// Nothing accepted.
    None,
    /// The fixed point.
    Fixed,
    /// Candidate `i` of that camera.
    Candidate(usize),
}

/// Why a camera of a marker was left undefined, for the report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The camera had a dot in the best combination, but it went to another marker.
    TakenByAnother,
    /// Another combination nearly as good puts a different dot in this camera.
    Ambiguous,
    /// No feasible combination uses this camera.
    NotFound,
    /// Every feasible combination breaks rigid links to markers served before.
    Rigid,
    /// The marker is being found again, and no combination keeps enough links to markers
    /// already accepted (and breaks none) to confirm which marker it is.
    Identity,
    /// Another marker's best combination uses the same dot, and neither is predicted
    /// clearly closer to it.
    Contested,
    /// The marker is being found again in this camera, and every dot it could take there is
    /// where another marker is expected (`MultiTrackParams::claim_radius`).
    Claimed,
    /// The marker overlaps another in this camera, and which of their dots is which cannot
    /// be told (`MultiTrackParams::overlap_decision`).
    Overlap,
}

/// Two markers (indices into the markers given to [`choose_joint_overlapping`]) that
/// overlap in camera `camera`: their identities there are decided together
/// (`MultiTrackParams::overlap_decision`), and, when they touch now, they may share a dot
/// there (`MultiTrackParams::share_radius`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Overlap {
    pub a: usize,
    pub b: usize,
    pub camera: usize,
    /// Their dots touch now (not only earlier in the overlap).
    pub touching: bool,
}

/// The result for one marker.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkerChoice {
    /// Per camera what is accepted.
    pub picks: Vec<Pick>,
    /// Per camera, why nothing was accepted (`None` where something was, or the camera had
    /// nothing to offer).
    pub refusals: Vec<Option<Refusal>>,
    /// Cost of the best feasible combination, `None` when there was none.
    pub cost: Option<f64>,
    /// The triangulated point of the best combination, if it had two cameras or more.
    pub point3d: Option<Point3>,
}

/// One feasible combination of a marker.
#[derive(Clone, Debug)]
struct Combo {
    picks: Vec<Pick>,
    /// Per camera the dot id (`usize::MAX` for none).
    dots: Vec<usize>,
    /// The cost the marker's own choice ranks by (distances relative to the gate, or to
    /// the near gate where a camera has one).
    cost: f64,
    /// Its rank class ([`choose_joint_linked`], 1.): 0 for every combination of a marker
    /// without a near gate in any camera; else 0 in two cameras or more without a far dot,
    /// 1 in two cameras or more with one, 2 in one camera without, 3 in one camera with.
    class: u8,
    /// The cost markers are served in order of (distances relative to the base gate, so
    /// that a marker found where it was expected goes before one found again far away).
    priority: f64,
    point3d: Option<Point3>,
}

/// A camera's near gate ([`CameraOptions::near_gate`]): the near gate when it is finite,
/// positive and below the gate, else the gate.
fn near_gate(o: &CameraOptions) -> f64 {
    if o.near_gate.is_finite() && o.near_gate > 0.0 && o.near_gate < o.gate {
        o.near_gate
    } else {
        o.gate
    }
}

/// The position a pick stands for.
fn pick_point(opt: &CameraOptions, pick: Pick) -> Option<(Point2, f64)> {
    match pick {
        Pick::None => None,
        Pick::Fixed => opt.fixed.map(|p| (p, 1.0)),
        Pick::Candidate(i) => opt.candidates.get(i).map(|c| (c.point, c.correlation)),
    }
}

fn finite2(p: Point2) -> bool {
    p.x.is_finite() && p.y.is_finite()
}

/// Dot ids per camera: every candidate and fixed point of every marker, clustered within
/// `same_dot` pixels (greedy, in marker order). Returns per marker, per camera, per option
/// (fixed first when present, then candidates) the id, and per camera the number of dots.
#[allow(clippy::type_complexity)]
fn dot_ids(
    markers: &[MarkerOptions],
    n_cam: usize,
    same: f64,
) -> (Vec<Vec<(Option<usize>, Vec<usize>)>>, Vec<usize>) {
    let mut centres: Vec<Vec<Point2>> = vec![Vec::new(); n_cam];
    let mut id = |c: usize, p: Point2| -> usize {
        if let Some(k) = centres[c].iter().position(|q| (q - p).norm() <= same) {
            return k;
        }
        centres[c].push(p);
        centres[c].len() - 1
    };
    let ids = markers
        .iter()
        .map(|m| {
            (0..n_cam)
                .map(|c| match m.cameras.get(c) {
                    None => (None, Vec::new()),
                    Some(o) => (
                        o.fixed.map(|p| id(c, p)),
                        o.candidates.iter().map(|k| id(c, k.point)).collect(),
                    ),
                })
                .collect()
        })
        .collect();
    (ids, centres.iter().map(Vec::len).collect())
}

/// Per camera the options of a marker: `Fixed`, or `None` plus the candidates (the
/// strongest `cap`), shortened until the product of the counts is at most `max_comb`.
fn camera_choices(m: &MarkerOptions, n_cam: usize, cap: usize, max_comb: usize) -> Vec<Vec<Pick>> {
    let mut lists: Vec<Vec<Pick>> = (0..n_cam)
        .map(|c| match m.cameras.get(c) {
            None => vec![Pick::None],
            Some(o) if o.fixed.is_some_and(finite2) => vec![Pick::Fixed],
            Some(o) => {
                let mut idx: Vec<usize> = (0..o.candidates.len())
                    .filter(|&i| finite2(o.candidates[i].point))
                    .collect();
                idx.sort_by(|&a, &b| {
                    o.candidates[b]
                        .correlation
                        .total_cmp(&o.candidates[a].correlation)
                        .then(a.cmp(&b))
                });
                idx.truncate(cap);
                std::iter::once(Pick::None)
                    .chain(idx.into_iter().map(Pick::Candidate))
                    .collect()
            }
        })
        .collect();
    let max_comb = max_comb.clamp(1, MAX_COMBINATIONS);
    loop {
        let product = lists
            .iter()
            .try_fold(1usize, |acc, l| acc.checked_mul(l.len()));
        if product.is_some_and(|p| p <= max_comb) {
            break;
        }
        // Shorten the longest list (its weakest candidate is last).
        let Some((k, _)) = lists
            .iter()
            .enumerate()
            .filter(|(_, l)| l.len() > 1)
            .max_by_key(|(_, l)| l.len())
        else {
            break;
        };
        lists[k].pop();
    }
    lists
}

/// Every feasible combination of one marker, cheapest first.
fn feasible_combos(
    cams: &[JointCamera<'_>],
    m: &MarkerOptions,
    ids: &[(Option<usize>, Vec<usize>)],
    method: TriangulationMethod,
    p: &MultiTrackParams,
) -> Vec<Combo> {
    let n_cam = cams.len();
    let lists = camera_choices(m, n_cam, p.candidates_cap(), p.max_combinations);
    // Cameras that had something to offer: leaving one out costs `weight_missing`.
    let offered: Vec<bool> = lists
        .iter()
        .map(|l| l.iter().any(|&k| k != Pick::None))
        .collect();
    let mut out = Vec::new();
    let mut index = vec![0usize; n_cam];
    let empty = CameraOptions::default();
    loop {
        let picks: Vec<Pick> = (0..n_cam).map(|c| lists[c][index[c]]).collect();
        if let Some(combo) = evaluate(cams, m, ids, &picks, &offered, &empty, method, p) {
            out.push(combo);
        }
        // Next index (odometer).
        let mut c = 0;
        loop {
            if c == n_cam {
                out.sort_by(|a, b| a.class.cmp(&b.class).then(a.cost.total_cmp(&b.cost)));
                return out;
            }
            index[c] += 1;
            if index[c] < lists[c].len() {
                break;
            }
            index[c] = 0;
            c += 1;
        }
    }
}

/// The cost of one combination, `None` when it is not feasible.
#[allow(clippy::too_many_arguments)]
fn evaluate(
    cams: &[JointCamera<'_>],
    m: &MarkerOptions,
    ids: &[(Option<usize>, Vec<usize>)],
    picks: &[Pick],
    offered: &[bool],
    empty: &CameraOptions,
    method: TriangulationMethod,
    p: &MultiTrackParams,
) -> Option<Combo> {
    let n_cam = cams.len();
    let mut used: Vec<(usize, Point2, f64)> = Vec::new();
    for (c, &pick) in picks.iter().enumerate() {
        let o = m.cameras.get(c).unwrap_or(empty);
        if let Some((pt, corr)) = pick_point(o, pick) {
            used.push((c, pt, corr));
        }
    }
    if used.is_empty() {
        return None;
    }
    let dots: Vec<usize> = (0..n_cam)
        .map(|c| match picks[c] {
            Pick::None => usize::MAX,
            Pick::Fixed => ids[c].0.unwrap_or(usize::MAX),
            Pick::Candidate(i) => ids[c].1.get(i).copied().unwrap_or(usize::MAX),
        })
        .collect();
    let missing = (0..n_cam)
        .filter(|&c| offered[c] && picks[c] == Pick::None)
        .count() as f64;
    let mut pred_term = 0.0;
    let mut match_term = 0.0;
    let mut n_pred = 0usize;
    let any_fixed_only = used.iter().all(|&(c, _, _)| picks[c] == Pick::Fixed);
    if any_fixed_only {
        // Nothing to decide: only points set by hand.
        return None;
    }
    // Per camera `((d / scale)^2, (d / base gate)^2, far)` with `d` the distance from the
    // prediction, `scale` the near gate when the camera has one (else the gate) and `far`
    // whether `d` is beyond it; `None` outside the gate; NaN terms without a prediction (no
    // gate).
    let gate_ok = |c: usize, pt: Point2| -> Option<(f64, f64, bool)> {
        let o = m.cameras.get(c).unwrap_or(empty);
        match o.prediction {
            Some(pr) if finite2(pr) => {
                let g = o.gate;
                let d = (pt - pr).norm();
                if g.is_nan() || g <= 0.0 || d.is_nan() || d > g {
                    return None;
                }
                let near = near_gate(o);
                let base = if p.gate > 0.0 { p.gate.min(near) } else { near };
                Some(((d / near) * (d / near), (d / base) * (d / base), d > near))
            }
            _ => Some((f64::NAN, f64::NAN, false)),
        }
    };
    let mut far = false;
    // The rank class (`Combo::class`): only for a marker with a near gate in some camera.
    let classed = m.cameras.iter().any(|o| near_gate(o) < o.gate);
    let class = |cameras: usize, far: bool| -> u8 {
        if !classed {
            0
        } else {
            u8::from(cameras < 2) * 2 + u8::from(far)
        }
    };
    let mut pred_abs = 0.0;
    if used.len() == 1 {
        let (c, pt, corr) = used[0];
        // A single camera needs a prediction and a tight gate, and no other dot near.
        let o = m.cameras.get(c).unwrap_or(empty);
        let pr = o.prediction.filter(|q| finite2(*q))?;
        let d1 = (pt - pr).norm();
        let lim = p.single_gate.min(o.gate.max(0.0));
        if d1.is_nan() || lim.is_nan() || d1 > lim {
            return None;
        }
        if picks[c] != Pick::Fixed {
            let near_other = o.candidates.iter().enumerate().any(|(i, k)| {
                Pick::Candidate(i) != picks[c]
                    && ids[c].1.get(i) != Some(&dots[c])
                    && (k.point - pt).norm() <= p.single_ambiguity
            });
            if near_other {
                return None;
            }
        }
        far = d1 > near_gate(o);
        let d = (pt - pr).norm() / p.single_gate.max(1e-9);
        let cost = p.weight_prediction * d * d
            + p.weight_match * (1.0 - corr)
            + p.weight_missing * missing;
        return cost.is_finite().then(|| Combo {
            picks: picks.to_vec(),
            dots,
            cost,
            class: class(1, far),
            priority: cost,
            point3d: None,
        });
    }
    let mut observations = Vec::with_capacity(used.len());
    for &(c, pt, corr) in &used {
        let (t, ta, f) = gate_ok(c, pt)?;
        far |= f;
        if !t.is_nan() {
            pred_term += t;
            pred_abs += ta;
            n_pred += 1;
        }
        match_term += 1.0 - corr;
        let und = cams[c].model.try_undistort_point(pt, true, true, true)?;
        if !finite2(und) {
            return None;
        }
        observations.push(Observation {
            projection: cams[c].projection,
            point: und,
        });
    }
    let limit = m
        .reprojection_limit
        .filter(|l| l.is_finite() && *l > 0.0)
        .unwrap_or(p.max_reprojection_error);
    let x = triangulate(method, &observations)?;
    if !(x.x.is_finite() && x.y.is_finite() && x.z.is_finite()) {
        return None;
    }
    let mut err_max: f64 = 0.0;
    for &(c, pt, _) in &used {
        // In front of the camera.
        let pm = &cams[c].projection;
        let w = pm[(2, 0)] * x.x + pm[(2, 1)] * x.y + pm[(2, 2)] * x.z + pm[(2, 3)];
        if w.is_nan() || w <= 0.0 {
            return None;
        }
        let q = cams[c].model.project_point(cams[c].pose, x);
        let e = (q - pt).norm();
        if e.is_nan() || limit.is_nan() || e > limit {
            return None;
        }
        err_max = err_max.max(e);
    }
    let r = err_max / limit.max(1e-9);
    let n_used = used.len() as f64;
    let (pred, pred_a) = if n_pred > 0 {
        (pred_term / n_pred as f64, pred_abs / n_pred as f64)
    } else {
        (0.0, 0.0)
    };
    let rest = p.weight_reprojection * r * r
        + p.weight_match * match_term / n_used
        + p.weight_missing * missing;
    let cost = rest + p.weight_prediction * pred;
    // A marker found again after it was lost is served after every marker followed.
    let reacquiring = used
        .iter()
        .any(|&(c, _, _)| m.cameras.get(c).is_some_and(|o| o.reacquire));
    let priority = rest + p.weight_prediction * pred_a + if reacquiring { 1000.0 } else { 0.0 };
    (cost.is_finite() && priority.is_finite()).then(|| Combo {
        picks: picks.to_vec(),
        dots,
        cost,
        class: class(used.len(), far),
        priority,
        point3d: Some(x),
    })
}

/// [`choose_joint_linked`] without rigid links. New in this program.
///
/// Work is bounded: per marker at most `params.max_combinations` combinations (at least 1),
/// each a triangulation over the cameras; the dot clustering is quadratic in the number of
/// candidates of a camera.
#[must_use]
pub fn choose_joint(
    cameras: &[JointCamera<'_>],
    markers: &[MarkerOptions],
    method: TriangulationMethod,
    params: &MultiTrackParams,
) -> Vec<MarkerChoice> {
    choose_joint_linked(cameras, markers, &[], method, params)
}

/// Two markers whose 3D distance has stayed constant (on the same bone): a combination of
/// one that puts it further than `tolerance` from `distance` to the other's accepted 3D
/// point breaks the link. `a` and `b` index the markers given to [`choose_joint_linked`].
/// The same pair may be given more than once (in either order): the partner is kept only
/// when every link given for it holds, and counts once ([`choose_joint_linked`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidLink {
    pub a: usize,
    pub b: usize,
    pub distance: f64,
    pub tolerance: f64,
}

/// Chooses, for every marker, one candidate (or nothing) per camera. New in this program.
///
/// 1. **Feasible combinations** per marker: every choice of one candidate or none per camera
///    (a fixed point is always used), at most `max_combinations`. With two cameras or more
///    the points are triangulated; feasible when the point is in front of every camera, its
///    reprojection error is at most `max_reprojection_error` in every camera used, and
///    every point lies within its camera's gate of the prediction. A point seen by one
///    camera only is feasible when it lies within `single_gate` of the prediction and no
///    other candidate of the marker in that camera lies within `single_ambiguity` of it.
///    Cost: `weight_reprojection (error / max)^2 + weight_prediction mean (d / gate)^2 +
///    weight_match mean (1 - correlation) + weight_missing (cameras left out that had
///    candidates)`, with `d` measured against a camera's [`CameraOptions::near_gate`]
///    instead of its gate when it has one. For a marker with a near gate in some camera
///    (a wider search), the combinations are ranked in classes, and by cost within a class:
///    (0) two cameras or more, no *far* dot (beyond a camera's near gate); (1) two cameras
///    or more with a far dot; (2) one camera, no far dot; (3) one camera with a far dot. A
///    fixed point counts as a camera (a fixed point in one camera and a near dot in another
///    is two cameras). A combination of a later class is no rival of the best in the
///    ambiguity test below. So the wider search never costs a point the ordinary one would
///    have accepted in two cameras or more, and a point seen by one camera only gives way
///    to a far pair that triangulates (better evidence). A marker without a near gate is
///    ranked by cost alone, as without the wider search.
/// 2. **Order.** Markers are served in order of their best combination's cost with the
///    distances measured against the base gate `gate` instead of the widened gate, so that
///    a marker found where it was expected goes before one found again far away.
/// 3. **For each marker in turn**, combinations are dropped that break [`RigidLink`]s to
///    markers served before it (with accepted 3D points), unless they keep at least
///    `rigid_majority` times as many (counted per partner: below); single-camera combinations whose dot lies within
///    `single_ambiguity` of another marker's prediction in that camera; and combinations
///    that use a camera flagged `reacquire` (the marker is being found again) unless they
///    are triangulated, keep at least `reacquire_links` partners (or as many as the marker
///    has rigid partners among `links`, when it has fewer, but at least one) and break
///    none; and, in a camera where the marker is being found again, combinations whose dot
///    there lies within `claim_radius` of another marker's prediction (a marker followed
///    there, not being found again) unless the marker's own prediction is within
///    `claim_ratio` times that distance of it ([`Refusal::Claimed`] when nothing else is
///    left). Markers found again are served after every marker followed. Then, per camera
///    of the best remaining combination, its dot is accepted
///    unless: another marker's best combination wants the same dot and this marker is not
///    predicted closer to it than `contest_ratio` times every other's distance
///    ([`Refusal::Contested`]; the contest is among the markers not served yet, each with
///    its best combination that passes the filters as things stand, so a marker that would
///    be refused does not compete); the dot went to a marker served before and is not within
///    `share_radius` of this marker's prediction ([`Refusal::TakenByAnother`]); or another
///    combination costing less than `ambiguity_margin` more puts a different free dot in
///    that camera ([`Refusal::Ambiguous`]). A camera where the marker is being found again
///    never shares (`reacquire_exclusive`): a dot another marker holds, or whose contest it
///    lost, is refused to it. A marker whose every accepted dot is shared with another
///    marker gets none of them (two markers on one point).
/// 4. **Overlaps** ([`choose_joint_overlapping`] only; [`Overlap`]). The pairs given are
///    taken as a set (out of range, a marker with itself and repeats dropped; a pair is
///    touching in a camera when any entry for it says so). With `share_needs_overlap`, two
///    markers share a dot in a camera in 3. only when they are given as touching there now.
///    Then, after 3., each pair is decided **from the result of 3.** (no pair sees another's
///    decision), giving each of its two markers in that camera an ask: keep, nothing, or a
///    given combination.
///    - With `overlap_take` off, two markers touching in a camera are asked for nothing there
///      (and no decision is made for that pair).
///    - With `overlap_decision`: a marker's cost with dot `d` in that camera is the cost of
///      its cheapest feasible combination with `d` there and, in every other camera, what 3.
///      gave it (anything where it was given nothing), plus one per rigid partner broken
///      among the markers accepted in 3. other than the two; a combination that breaks the
///      rigid majority, or (found again) its identity links, does not count. Where the two
///      hold different dots, the cost of the assignment given is compared with that of the
///      two exchanged; where one holds a dot, the other none, and they are not touching now,
///      the holder's cost with that of the other on the dot. The assignment given stands
///      (keep) when the other costs at least `overlap_decision` more; within that margin
///      both are asked for nothing there (for the one dot, the holder); when the other is
///      cheaper by more, it is taken: each marker it gives a dot is asked for its cheapest
///      combination with that dot and **exactly** what 3. gave it in the other cameras
///      (nothing where it was given nothing), and where an exchange cannot be taken so,
///      both are asked for nothing; the holder of one dot given to the other is asked for
///      nothing either way. The looser cost (anything in a camera where a marker was given
///      nothing) only weighs how well the other assignment fits; what is applied is always
///      a combination with exactly the marker's other picks. (Scoring with exactly the
///      other picks as well was measured on the Rat, nine-run means: 68.9 % found and 166
///      points more than 5 px off, against 68.7 % and 159: no clearer, so the looser cost,
///      which lets a marker that lost a camera still show that the dot fits it, stays.)
///
///    The asks are then applied together: where every pair asks the same of a marker in a
///    camera, that; where two ask different things, nothing there ([`Refusal::Overlap`]).
///    A dot given to a marker this way while another marker holds it (the two not having
///    both held it in 3.) goes to neither marker given it. Every marker changed is then made
///    consistent: its 3D point and cost are those of the feasible combination with exactly
///    its picks; with two cameras or more and no such combination, it gets nothing in any
///    camera ([`Refusal::Overlap`]). A marker left with one camera keeps that point
///    unchecked, without a 3D point (the default, `overlap_single_checked` off): it may then
///    be further than `single_gate` from its prediction, as a point may where 3. refuses a
///    marker's other camera. With `overlap_single_checked` on it keeps the point only when
///    a feasible single-camera combination with exactly that pick exists (within
///    `single_gate` of the prediction, no other candidate within `single_ambiguity`), and
///    otherwise gets nothing. The result does not depend on the order the overlaps are
///    given in.
///
/// `cameras` are the cameras in project order (a marker's `cameras[c]` is camera `c`);
/// cameras without a calibration are given with nothing to offer. Links naming a marker out
/// of range, or with a distance or tolerance that is not finite, are ignored. Links are
/// counted per partner: when a pair of markers is given more than once (in either order,
/// as the engine gives a learned link and a lost marker's rigid-prediction link for the
/// same pair), every valid link given for it must hold for the partner to count as kept,
/// and one broken link makes the partner broken; a partner counts once towards the number
/// `reacquire_links` requires and towards the kept and broken counts `rigid_majority`
/// weighs, whatever the number of links given for it.
///
/// Work is bounded as in [`choose_joint`], plus one test per link and combination; the
/// links are grouped by pair in one pass (a hash map), in proportion to their number.
#[must_use]
pub fn choose_joint_linked(
    cameras: &[JointCamera<'_>],
    markers: &[MarkerOptions],
    links: &[RigidLink],
    method: TriangulationMethod,
    params: &MultiTrackParams,
) -> Vec<MarkerChoice> {
    choose(cameras, markers, links, None, method, params)
}

/// [`choose_joint_linked`] with the markers that overlap in a camera (the engine's overlap
/// state): rule 4 of [`choose_joint_linked`] (sharing only between markers touching now,
/// `overlap_take`, identity out of an overlap). Overlaps naming a marker or camera out of
/// range are ignored; the order they are given in does not matter. New in this program.
///
/// Work is bounded as in [`choose_joint_linked`], plus: sets of the overlaps given (in
/// proportion to their number) and a look-up in them per shared dot; and, for rule 4, per
/// distinct pair, up to six scans of each of its two markers' combinations (at most
/// `max_combinations` each), every combination tested against the marker's rigid partners
/// (no copy of the accepted points is made per pair); then, per marker given a dot by a
/// decision, one pass over the other markers' picks in that camera, and per marker changed
/// one scan of its combinations.
pub fn choose_joint_overlapping(
    cameras: &[JointCamera<'_>],
    markers: &[MarkerOptions],
    links: &[RigidLink],
    overlaps: &[Overlap],
    method: TriangulationMethod,
    params: &MultiTrackParams,
) -> Vec<MarkerChoice> {
    choose(cameras, markers, links, Some(overlaps), method, params)
}

fn choose(
    cameras: &[JointCamera<'_>],
    markers: &[MarkerOptions],
    links: &[RigidLink],
    overlaps: Option<&[Overlap]>,
    method: TriangulationMethod,
    params: &MultiTrackParams,
) -> Vec<MarkerChoice> {
    let n_cam = cameras.len();
    // The pairs that may share a dot, per camera (`share_needs_overlap`): `None` when every
    // pair may (the old rule).
    let may_share: Option<std::collections::HashSet<(usize, usize, usize)>> =
        overlaps.filter(|_| params.share_needs_overlap).map(|list| {
            list.iter()
                .filter(|o| {
                    o.touching && o.a < markers.len() && o.b < markers.len() && o.camera < n_cam
                })
                .map(|o| (o.a.min(o.b), o.a.max(o.b), o.camera))
                .collect()
        });
    let pair_shares = |a: usize, b: usize, c: usize| -> bool {
        may_share
            .as_ref()
            .is_none_or(|set| set.contains(&(a.min(b), a.max(b), c)))
    };
    // Per marker one entry per partner, with every valid link given for the pair (in
    // either order): (partner, [(distance, tolerance)]). Built in one pass through a map of
    // pairs, so the work and memory are in proportion to the links given.
    let mut pairs: std::collections::HashMap<(usize, usize), Vec<(f64, f64)>> =
        std::collections::HashMap::new();
    let mut order: Vec<(usize, usize)> = Vec::new();
    for l in links {
        if l.a < markers.len()
            && l.b < markers.len()
            && l.a != l.b
            && l.distance.is_finite()
            && l.tolerance.is_finite()
            && l.tolerance >= 0.0
        {
            let key = (l.a.min(l.b), l.a.max(l.b));
            pairs
                .entry(key)
                .or_insert_with(|| {
                    order.push(key);
                    Vec::new()
                })
                .push((l.distance, l.tolerance));
        }
    }
    let mut partners: Vec<Vec<Partner>> = vec![Vec::new(); markers.len()];
    for key in order {
        let constraints = pairs.remove(&key).unwrap_or_default();
        partners[key.0].push((key.1, constraints.clone()));
        partners[key.1].push((key.0, constraints));
    }
    let mut accepted3d: Vec<Option<Point3>> = vec![None; markers.len()];
    let (ids, n_dots) = dot_ids(markers, n_cam, params.same_dot);
    let combos: Vec<Vec<Combo>> = markers
        .iter()
        .zip(&ids)
        .map(|(m, id)| feasible_combos(cameras, m, id, method, params))
        .collect();
    let mut order: Vec<usize> = (0..markers.len())
        .filter(|&k| !combos[k].is_empty())
        .collect();
    order.sort_by(|&a, &b| {
        combos[a][0]
            .priority
            .total_cmp(&combos[b][0].priority)
            .then(a.cmp(&b))
    });
    // Per camera and dot, the first marker it was given to.
    let mut taken: Vec<Vec<Option<usize>>> = n_dots.iter().map(|&n| vec![None; n]).collect();
    let dist_to = |k: usize, c: usize, combo: &Combo| -> f64 {
        markers[k]
            .cameras
            .get(c)
            .and_then(|o| Some((o.prediction?, pick_point(o, combo.picks[c])?.0)))
            .map_or(f64::INFINITY, |(pr, pt)| (pt - pr).norm())
    };
    let mut out: Vec<MarkerChoice> = markers
        .iter()
        .map(|m| {
            let refusals = (0..n_cam)
                .map(|c| {
                    m.cameras
                        .get(c)
                        .filter(|o| o.fixed.is_none() && !o.candidates.is_empty())
                        .map(|_| Refusal::NotFound)
                })
                .collect();
            MarkerChoice {
                picks: vec![Pick::None; n_cam],
                refusals,
                cost: None,
                point3d: None,
            }
        })
        .collect();
    let mut served = vec![false; markers.len()];
    // Per camera and dot, the smallest distance of a marker that lost the contest for it.
    let mut lost: Vec<Vec<f64>> = n_dots.iter().map(|&n| vec![f64::INFINITY; n]).collect();
    // Per camera and dot, the first marker that lost the contest for it.
    let mut lost_by: Vec<Vec<Option<usize>>> = n_dots.iter().map(|&n| vec![None; n]).collect();
    let ctx = Filters {
        markers,
        partners: &partners,
        params,
        n_cam,
    };
    for &k in &order {
        served[k] = true;
        let (list, why_none, claimed) = ctx.allowed(k, &combos[k], &accepted3d);
        let choice = &mut out[k];
        let Some(&best) = list.first() else {
            for r in choice.refusals.iter_mut().flatten() {
                *r = why_none;
            }
            continue;
        };
        choice.cost = Some(best.cost);
        choice.point3d = best.point3d;
        // The best combination that passes the filters of every marker not served yet, as
        // things stand now (for the contests below).
        let rival_best: Vec<(usize, &Combo)> = if params.contest_ratio > 0.0 {
            order
                .iter()
                .filter(|&&j| !served[j])
                .filter_map(|&j| {
                    let (lj, _, _) = ctx.allowed(j, &combos[j], &accepted3d);
                    lj.first().map(|b| (j, *b))
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut shared = vec![false; n_cam];
        for c in 0..n_cam {
            let pick = best.picks[c];
            if pick == Pick::None {
                continue;
            }
            let dot = best.dots[c];
            if pick == Pick::Fixed {
                choice.picks[c] = Pick::Fixed;
                choice.refusals[c] = None;
                continue;
            }
            // Within `share_radius` of its prediction, a marker may share the dot
            // (markers that overlap in this camera); never in a camera where it is being
            // found again (`reacquire_exclusive`).
            let dk = dist_to(k, c, best);
            let reacquiring = markers[k].cameras.get(c).is_some_and(|o| o.reacquire);
            let near = params.share_radius > 0.0
                && dk <= params.share_radius
                && !(reacquiring && params.reacquire_exclusive);
            // The contest: markers not served yet whose best combination that passes the
            // filters (as things stand now) puts the same dot in this camera. This marker
            // keeps the dot only when it is closer to it than `contest_ratio` times each of
            // theirs; a marker that would be refused does not compete.
            let rival_list: Vec<(usize, f64)> = rival_best
                .iter()
                .filter(|(_, bj)| matches!(bj.picks[c], Pick::Candidate(_)) && bj.dots[c] == dot)
                .map(|&(j, bj)| (j, dist_to(j, c, bj)))
                .collect();
            let rivals: Vec<f64> = rival_list.iter().map(|x| x.1).collect();
            // Markers served before that lost the contest for this dot count as rivals too
            // (a tie goes to none of them).
            let lost_before = lost[c].get(dot).copied().unwrap_or(f64::INFINITY);
            let wins = rivals
                .iter()
                .chain(std::iter::once(&lost_before))
                .all(|&dj| {
                    dj == f64::INFINITY || (dk.is_finite() && dk < params.contest_ratio * dj)
                });
            if !wins {
                if let Some(l) = lost[c].get_mut(dot) {
                    *l = l.min(dk);
                }
                // Losing the contest but predicted on the dot: it shares it, with markers
                // it overlaps in this camera only.
                let overlaps_rivals = rival_list.iter().all(|&(j, _)| pair_shares(k, j, c))
                    && lost_by[c]
                        .get(dot)
                        .and_then(|x| *x)
                        .is_none_or(|j| pair_shares(k, j, c));
                if let Some(l) = lost_by[c].get_mut(dot)
                    && l.is_none()
                {
                    *l = Some(k);
                }
                if !(near && overlaps_rivals) {
                    choice.refusals[c] = Some(Refusal::Contested);
                    continue;
                }
                shared[c] = true;
            }
            if let Some(holder) = taken[c].get(dot).copied().flatten() {
                // Two markers that overlap in this camera may share the dot when this
                // marker is predicted on it.
                if !(near && pair_shares(k, holder, c)) {
                    choice.refusals[c] = Some(Refusal::TakenByAnother);
                    continue;
                }
                shared[c] = true;
            }
            // Ambiguity: a nearly as good combination with another dot (not taken by a
            // marker served earlier) in this camera.
            // A combination of a later rank class is no rival of the best.
            let rival = list[1..]
                .iter()
                .filter(|o| o.class == best.class)
                .take_while(|o| o.cost < best.cost + params.ambiguity_margin)
                .any(|o| {
                    let d = o.dots[c];
                    d != usize::MAX && d != dot && taken[c].get(d).copied().flatten().is_none()
                });
            if rival {
                choice.refusals[c] = Some(Refusal::Ambiguous);
                continue;
            }
            choice.picks[c] = pick;
            choice.refusals[c] = None;
        }
        // A camera the best combination leaves out because the dots it could take there are
        // claimed by another marker says so.
        for c in 0..n_cam {
            if best.picks[c] == Pick::None
                && claimed.get(c).copied().unwrap_or(false)
                && choice.refusals[c] == Some(Refusal::NotFound)
            {
                choice.refusals[c] = Some(Refusal::Claimed);
            }
        }
        // Sharing in every camera it uses would put two markers on one point: one of them
        // is wrong, and this one, served later, gets nothing.
        let used: Vec<usize> = (0..n_cam)
            .filter(|&c| matches!(choice.picks[c], Pick::Candidate(_)))
            .collect();
        if !used.is_empty() && used.iter().all(|&c| shared[c]) {
            for &c in &used {
                choice.picks[c] = Pick::None;
                choice.refusals[c] = Some(Refusal::TakenByAnother);
            }
        }
        for c in 0..n_cam {
            if matches!(choice.picks[c], Pick::Candidate(_))
                && let Some(t) = taken[c].get_mut(best.dots[c])
                && t.is_none()
            {
                *t = Some(k);
            }
        }
        let n_accepted = choice.picks.iter().filter(|p| **p != Pick::None).count();
        if n_accepted >= 2 {
            accepted3d[k] = best.point3d;
        }
    }
    if let Some(list) = overlaps {
        let decide = params.overlap_decision.is_finite() && params.overlap_decision > 0.0;
        if decide || !params.overlap_take {
            decide_overlaps(list, &ids, &combos, &ctx, &accepted3d, &mut out);
        }
    }
    out
}

/// What one pair's decision asks of one of its markers in one camera
/// ([`decide_overlaps`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ask {
    /// Keep what the choice gave it.
    Keep,
    /// Nothing there ([`Refusal::Overlap`]).
    Empty,
    /// The marker's combination with this index (the dot the other assignment gives it).
    Take(usize),
}

/// The overlaps after the choice ([`choose_joint_overlapping`], 4.): `overlap_take` off, and
/// identity out of an overlap (`MultiTrackParams::overlap_decision`). Every pair is decided
/// from the same result of the choice (`out` as it comes in, `accepted` its 3D points), so
/// the order of `list` does not matter; the asks are then applied together, a marker asked
/// different things in one camera by two pairs getting nothing there. Afterwards a dot given
/// to two markers that did not both hold it before goes to neither of the markers that were
/// given it, and every marker changed is settled ([`settle`]).
///
/// Work: per distinct pair given (duplicates and order removed through a set in proportion
/// to the list), up to six scans of each of its two markers' combinations, each combination
/// tested against the marker's rigid partners; then one pass over the markers' picks per
/// camera.
fn decide_overlaps(
    list: &[Overlap],
    ids: &[Vec<(Option<usize>, Vec<usize>)>],
    combos: &[Vec<Combo>],
    ctx: &Filters<'_>,
    accepted: &[Option<Point3>],
    out: &mut [MarkerChoice],
) {
    let p = ctx.params;
    let decide = p.overlap_decision.is_finite() && p.overlap_decision > 0.0;
    let margin = p.overlap_decision;
    let n = ctx.markers.len();
    let n_cam = ctx.n_cam;
    // The distinct pairs, in a fixed order, touching when any entry for them says so.
    let mut pairs: std::collections::BTreeMap<(usize, usize, usize), bool> =
        std::collections::BTreeMap::new();
    for o in list {
        if o.a >= n || o.b >= n || o.a == o.b || o.camera >= n_cam {
            continue;
        }
        *pairs
            .entry((o.a.min(o.b), o.a.max(o.b), o.camera))
            .or_insert(false) |= o.touching;
    }
    // The dot (id) a marker holds in camera `c` in the choice's result.
    let dot = |x: usize, c: usize| -> Option<usize> {
        match out[x].picks[c] {
            Pick::Candidate(i) => ids[x].get(c).and_then(|v| v.1.get(i)).copied(),
            _ => None,
        }
    };
    let required = |x: usize| p.reacquire_links.min(ctx.partners[x].len()).max(1);
    let majority = if p.rigid_majority.is_nan() {
        1.0
    } else {
        p.rigid_majority.max(1.0)
    };
    // Marker `x`'s cost with dot `d` in camera `c`: its cheapest combination with that dot
    // there and, in every other camera, what the choice gave it (with `strict`, exactly
    // that: nothing where it was given nothing; otherwise anything there), plus one per
    // rigid partner broken among the markers accepted other than `skip`; `None` when no such
    // combination keeps the rigid majority or (found again) its identity links.
    let score = |x: usize, c: usize, d: usize, strict: bool, skip: (usize, usize)| {
        combos[x]
            .iter()
            .enumerate()
            .filter(|(_, k)| k.dots.get(c) == Some(&d))
            .filter(|(_, k)| {
                (0..n_cam)
                    .filter(|&j| j != c)
                    .all(|j| match out[x].picks[j] {
                        Pick::None => !strict || k.picks[j] == Pick::None,
                        q => k.picks[j] == q,
                    })
            })
            .filter_map(|(i, k)| {
                let reacquiring = (0..n_cam).any(|j| {
                    matches!(k.picks[j], Pick::Candidate(_))
                        && ctx.markers[x].cameras.get(j).is_some_and(|o| o.reacquire)
                });
                let (kept, broken) = match k.point3d {
                    Some(y) => ctx.link_counts_except(x, y, accepted, skip),
                    None => (0, 0),
                };
                if reacquiring
                    && p.reacquire_links > 0
                    && (k.point3d.is_none() || kept < required(x) || broken > 0)
                {
                    return None;
                }
                if broken > 0 && (kept as f64) < majority * broken as f64 {
                    return None;
                }
                let v = k.cost + broken as f64;
                v.is_finite().then_some((v, i))
            })
            .min_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)))
    };
    // Per (marker, camera) the asks of every pair, from the same starting point.
    let mut asks: std::collections::BTreeMap<(usize, usize), Vec<Ask>> =
        std::collections::BTreeMap::new();
    for (&(a, b, c), &touching) in &pairs {
        let mut ask = |x: usize, what: Ask| asks.entry((x, c)).or_default().push(what);
        if !p.overlap_take && touching {
            for x in [a, b] {
                if matches!(out[x].picks[c], Pick::Candidate(_)) {
                    ask(x, Ask::Empty);
                }
            }
            continue;
        }
        if !decide {
            continue;
        }
        let skip = (a, b);
        match (dot(a, c), dot(b, c)) {
            (Some(da), Some(db)) if da != db => {
                let (Some(ka), Some(kb)) =
                    (score(a, c, da, false, skip), score(b, c, db, false, skip))
                else {
                    continue;
                };
                let now = ka.0 + kb.0;
                let (xa, xb) = (score(a, c, db, false, skip), score(b, c, da, false, skip));
                let swapped = match (xa, xb) {
                    (Some(x), Some(y)) => x.0 + y.0,
                    _ => f64::INFINITY,
                };
                if swapped + margin < now {
                    // Exchanged: each takes the other's dot with exactly what it holds in the
                    // other cameras; when either cannot, neither keeps a dot here.
                    match (score(a, c, db, true, skip), score(b, c, da, true, skip)) {
                        (Some(ya), Some(yb)) => {
                            ask(a, Ask::Take(ya.1));
                            ask(b, Ask::Take(yb.1));
                        }
                        _ => {
                            ask(a, Ask::Empty);
                            ask(b, Ask::Empty);
                        }
                    }
                } else if swapped < now + margin {
                    ask(a, Ask::Empty);
                    ask(b, Ask::Empty);
                } else {
                    ask(a, Ask::Keep);
                    ask(b, Ask::Keep);
                }
            }
            // One dot, held by one of two markers that no longer touch (the other has lost
            // its own dot in this camera since): is it the holder's or the other's? While they
            // touch, one dot for one of them is what a merge looks like, and the choice's
            // own rules (sharing, contests) apply.
            (Some(d), None) | (None, Some(d)) if !touching => {
                let (h, o) = if dot(a, c).is_some() { (a, b) } else { (b, a) };
                let (Some(kh), Some(ko)) =
                    (score(h, c, d, false, skip), score(o, c, d, false, skip))
                else {
                    continue;
                };
                if ko.0 + margin < kh.0 {
                    // Clearly the other's: it takes the dot when it can with exactly what it
                    // holds elsewhere, and the holder gives it up.
                    ask(h, Ask::Empty);
                    if let Some(y) = score(o, c, d, true, skip) {
                        ask(o, Ask::Take(y.1));
                    }
                } else if ko.0 < kh.0 + margin {
                    ask(h, Ask::Empty);
                } else {
                    ask(h, Ask::Keep);
                }
            }
            _ => {}
        }
    }
    if asks.is_empty() {
        return;
    }
    // Applied together: one ask, or the same from every pair; different asks give nothing.
    let before: Vec<Vec<Pick>> = out.iter().map(|m| m.picks.clone()).collect();
    let mut changed = vec![false; n];
    let mut taken: Vec<(usize, usize)> = Vec::new();
    for (&(x, c), v) in &asks {
        let agreed = if v.iter().all(|w| *w == v[0]) {
            v[0]
        } else {
            Ask::Empty
        };
        match agreed {
            Ask::Keep => {}
            Ask::Empty => {
                if out[x].picks[c] != Pick::None {
                    out[x].picks[c] = Pick::None;
                    out[x].refusals[c] = Some(Refusal::Overlap);
                    changed[x] = true;
                }
            }
            Ask::Take(i) => {
                out[x].picks[c] = combos[x][i].picks[c];
                out[x].refusals[c] = None;
                changed[x] = true;
                taken.push((x, c));
            }
        }
    }
    // A dot given by a pair's decision to a marker while another marker holds it (that did
    // not hold it with this one before) goes to neither of the markers given it.
    let held = |x: usize, c: usize, picks: &Vec<Pick>| -> Option<usize> {
        match picks[c] {
            Pick::Candidate(i) => ids[x].get(c).and_then(|v| v.1.get(i)).copied(),
            _ => None,
        }
    };
    let mut clear: Vec<(usize, usize)> = Vec::new();
    for &(x, c) in &taken {
        let Some(d) = held(x, c, &out[x].picks) else {
            continue;
        };
        let clash = (0..n).any(|y| {
            y != x
                && held(y, c, &out[y].picks) == Some(d)
                && !(held(x, c, &before[x]) == Some(d) && held(y, c, &before[y]) == Some(d))
        });
        if clash {
            clear.push((x, c));
        }
    }
    for (x, c) in clear {
        out[x].picks[c] = Pick::None;
        out[x].refusals[c] = Some(Refusal::Overlap);
    }
    for x in (0..n).filter(|&x| changed[x]) {
        settle(&combos[x], &mut out[x], p.overlap_single_checked);
    }
}

/// Makes a marker's result consistent after [`decide_overlaps`] changed its picks: its 3D
/// point (and cost) are those of the feasible combination with exactly its picks (a
/// single-camera combination is feasible within `single_gate` of the prediction and with no
/// other candidate of the marker within `single_ambiguity`). Without one: with one camera
/// left and `single_checked` off, the point stays, without a 3D point (as the choice keeps a
/// camera whose other camera was refused); otherwise the marker gets nothing in any camera
/// ([`Refusal::Overlap`]).
///
/// Two cameras or more with no feasible combination of exactly those picks can arise when
/// two pairs' decisions change one marker in two cameras (each `Take` keeps the marker's
/// other picks as the choice gave them, so the two together were never costed), or with
/// three cameras or more when one is emptied and the rest are not feasible together.
fn settle(combos: &[Combo], choice: &mut MarkerChoice, single_checked: bool) {
    if let Some(k) = combos.iter().find(|k| k.picks == choice.picks) {
        choice.point3d = k.point3d;
        choice.cost = Some(k.cost);
        return;
    }
    choice.point3d = None;
    let n_picks = choice.picks.iter().filter(|q| **q != Pick::None).count();
    if n_picks == 0 || (n_picks == 1 && !single_checked) {
        // No combination stands behind what is left (nothing, or one unchecked point): no
        // cost either.
        choice.cost = None;
        return;
    }
    for c in 0..choice.picks.len() {
        if choice.picks[c] != Pick::None {
            choice.picks[c] = Pick::None;
            choice.refusals[c] = Some(Refusal::Overlap);
        }
    }
    choice.cost = None;
}

/// A marker's rigid partner and every link given for the pair, as `(distance, tolerance)`.
type Partner = (usize, Vec<(f64, f64)>);

/// The per-marker filters of [`choose_joint_linked`], which depend on the markers accepted
/// so far.
struct Filters<'a> {
    markers: &'a [MarkerOptions],
    partners: &'a [Vec<Partner>],
    params: &'a MultiTrackParams,
    n_cam: usize,
}

impl Filters<'_> {
    /// Marker `k`'s partners kept and broken by point `x` against the markers accepted: a
    /// partner is kept when every link given for the pair holds and broken when any breaks;
    /// each counts once.
    fn link_counts(&self, k: usize, x: Point3, accepted: &[Option<Point3>]) -> (usize, usize) {
        self.link_counts_except(k, x, accepted, (usize::MAX, usize::MAX))
    }

    /// [`Filters::link_counts`] leaving out the partners `skip.0` and `skip.1`.
    fn link_counts_except(
        &self,
        k: usize,
        x: Point3,
        accepted: &[Option<Point3>],
        skip: (usize, usize),
    ) -> (usize, usize) {
        self.partners[k]
            .iter()
            .filter(|(q, _)| *q != skip.0 && *q != skip.1)
            .fold((0, 0), |(kept, broken), (q, links)| match accepted[*q] {
                Some(y) => {
                    let d = (x - y).norm();
                    if links.iter().all(|&(l, tol)| (d - l).abs() <= tol) {
                        (kept + 1, broken)
                    } else {
                        (kept, broken + 1)
                    }
                }
                None => (kept, broken),
            })
    }

    /// Marker `k`'s combinations that pass the filters, cheapest first, and the reason to
    /// give when none does.
    fn allowed<'c>(
        &self,
        k: usize,
        combos: &'c [Combo],
        accepted: &[Option<Point3>],
    ) -> (Vec<&'c Combo>, Refusal, Vec<bool>) {
        let p = self.params;
        // A combination is refused when it breaks more links to accepted markers than
        // `rigid_majority` allows (one wrong partner must not refuse a right marker).
        let rigid_ok = |o: &Combo| -> bool {
            let Some(x) = o.point3d else {
                return true;
            };
            let (kept, broken) = self.link_counts(k, x, accepted);
            let majority = if p.rigid_majority.is_nan() {
                1.0
            } else {
                p.rigid_majority.max(1.0)
            };
            broken == 0 || kept as f64 >= majority * broken as f64
        };
        // A point seen by one camera only is refused near another marker's prediction in
        // that camera: it may be that marker's dot.
        let single_ok = |o: &Combo| -> bool {
            let mut used = (0..self.n_cam).filter(|&c| o.picks[c] != Pick::None);
            let (Some(c), None) = (used.next(), used.next()) else {
                return true;
            };
            let Some(pt) = self.markers[k]
                .cameras
                .get(c)
                .and_then(|opt| pick_point(opt, o.picks[c]))
                .map(|x| x.0)
            else {
                return true;
            };
            !self.markers.iter().enumerate().any(|(j, m)| {
                j != k
                    && m.cameras
                        .get(c)
                        .and_then(|opt| opt.prediction)
                        .is_some_and(|pr| (pr - pt).norm() <= p.single_ambiguity)
            })
        };
        // A marker found again must prove who it is: its 3D point keeps `reacquire_links`
        // links to markers already accepted (no more than it has rigid partners, at least
        // one) and breaks none.
        // (`partners` holds one entry per partner, whatever the links given for it.)
        let required = p.reacquire_links.min(self.partners[k].len()).max(1);
        let identity_ok = |o: &Combo| -> bool {
            let reacquiring = (0..self.n_cam).any(|c| {
                matches!(o.picks[c], Pick::Candidate(_))
                    && self.markers[k]
                        .cameras
                        .get(c)
                        .is_some_and(|opt| opt.reacquire)
            });
            if !reacquiring || p.reacquire_links == 0 {
                return true;
            }
            let Some(x) = o.point3d else {
                return false;
            };
            let (kept, broken) = self.link_counts(k, x, accepted);
            kept >= required && broken == 0
        };
        // Claims (`claim_radius`): in a camera where this marker is being found again, a dot
        // near another marker's prediction there is that marker's, unless this marker's own
        // prediction is clearly closer to it.
        let claim_fails = |o: &Combo, c: usize| -> bool {
            if !(p.claim_radius.is_finite() && p.claim_radius > 0.0) {
                return false;
            }
            let Some(opt) = self.markers[k].cameras.get(c) else {
                return false;
            };
            if !opt.reacquire || !matches!(o.picks[c], Pick::Candidate(_)) {
                return false;
            }
            let Some((pt, _)) = pick_point(opt, o.picks[c]) else {
                return false;
            };
            let own = opt
                .prediction
                .filter(|q| finite2(*q))
                .map_or(f64::INFINITY, |q| (pt - q).norm());
            self.markers.iter().enumerate().any(|(j, m)| {
                j != k
                    && m.cameras.get(c).is_some_and(|oj| {
                        !oj.reacquire
                            && oj.prediction.filter(|q| finite2(*q)).is_some_and(|q| {
                                let d = (pt - q).norm();
                                // Not clearly its own (written so: a ratio that is not a
                                // number claims, as `own > ...` would not).
                                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                                let not_own = !(own <= p.claim_ratio * d);
                                d <= p.claim_radius && not_own
                            })
                    })
            })
        };
        let mut claimed = vec![false; self.n_cam];
        let mut any_rs = false;
        let mut any_rsc = false;
        let mut any_rigid_fail = false;
        let list: Vec<&Combo> = combos
            .iter()
            .filter(|o| {
                let r = rigid_ok(o);
                any_rigid_fail |= !r;
                let rs = r && single_ok(o);
                any_rs |= rs;
                let mut rsc = rs;
                if rs {
                    for (c, flag) in claimed.iter_mut().enumerate() {
                        if claim_fails(o, c) {
                            *flag = true;
                            rsc = false;
                        }
                    }
                }
                any_rsc |= rsc;
                rsc && identity_ok(o)
            })
            .collect();
        let why = if any_rsc {
            Refusal::Identity
        } else if any_rs {
            Refusal::Claimed
        } else if any_rigid_fail {
            Refusal::Rigid
        } else {
            Refusal::NotFound
        };
        (list, why, claimed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Intrinsics, ModelCompat, projection_matrix};
    use crate::image::ImageU8;
    use crate::{Matrix3, Vector3};

    fn blob_image(w: usize, h: usize, dots: &[(f64, f64)]) -> ImageU8 {
        let mut im = ImageU8::filled(w, h, 200);
        for y in 0..h {
            for x in 0..w {
                let mut v = 200.0;
                for &(cx, cy) in dots {
                    let d2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                    v -= 120.0 * (-d2 / (2.0 * 1.5 * 1.5)).exp();
                }
                im.set(x, y, v.clamp(0.0, 255.0) as u8);
            }
        }
        im
    }

    #[test]
    fn ncc_finds_the_dots() {
        let im = blob_image(80, 80, &[(40.0, 40.0), (48.0, 37.0)]);
        let t = cut_ncc_template(&im, 40.0, 40.0, 6).unwrap();
        let s = ncc_surface(&im, &t, (42.0, 41.0), 10).unwrap();
        let peaks = local_maxima(&s.values, 0.3, 4);
        assert!(peaks.len() >= 2, "{peaks:?}");
        let pos: Vec<(i64, i64)> = peaks
            .iter()
            .map(|&(x, y, _)| (s.off_x + x as i64, s.off_y + y as i64))
            .collect();
        assert_eq!(pos[0], (40, 40));
        assert!(pos.contains(&(48, 37)), "{pos:?}");
        assert!((peaks[0].2 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn ncc_rejects_bad_arguments() {
        let im = blob_image(20, 20, &[]);
        let t = ImageF32::new(4, 4);
        assert!(ncc_surface(&im, &t, (5.0, 5.0), 3).is_none());
        let t = ImageF32::new(5, 5);
        assert!(ncc_surface(&im, &t, (f64::NAN, 5.0), 3).is_none());
        assert!(ncc_surface(&im, &t, (5.0, 5.0), -1).is_none());
        // An enormous radius is clamped to the image's larger side (20).
        let s = ncc_surface(&im, &t, (5.0, 5.0), i32::MAX).unwrap();
        assert!(s.values.width() <= 41, "{}", s.values.width());
        assert!(ncc_surface(&im, &ImageF32::new(201, 201), (5.0, 5.0), 3).is_none());
        // Flat image and template: correlation 0, no maxima above 0.3.
        let s = ncc_surface(&im, &t, (1e6, -1e6), 2).unwrap();
        assert!(s.values.data().iter().all(|&v| v == 0.0));
        assert!(cut_ncc_template(&im, 5.0, 5.0, 0).is_none());
        assert!(cut_ncc_template(&im, f64::INFINITY, 5.0, 3).is_none());
        let empty = ImageU8::new(0, 0);
        assert!(cut_ncc_template(&empty, 1.0, 1.0, 3).is_none());
        assert!(ncc_surface(&empty, &t, (1.0, 1.0), 2).is_none());
    }

    #[test]
    fn local_maxima_order_and_plateau() {
        let mut s = ImageF32::new(5, 5);
        s.set(1, 1, 0.5);
        s.set(3, 3, 0.9);
        s.set(3, 1, 0.7);
        s.set(4, 1, 0.7); // plateau with (3, 1): the first in row order wins
        let m = local_maxima(&s, 0.4, 10);
        assert_eq!(m, vec![(3, 3, 0.9), (3, 1, 0.7), (1, 1, 0.5)]);
        assert_eq!(local_maxima(&s, 0.4, 1).len(), 1);
        assert!(local_maxima(&s, 0.4, 0).is_empty());
        assert!(local_maxima(&ImageF32::new(0, 0), 0.0, 3).is_empty());
    }

    #[test]
    fn gate_grows_to_its_limit() {
        let p = MultiTrackParams::default();
        assert_eq!(p.gate_for(0, true), 5.0);
        assert_eq!(p.gate_for(1, true), 5.0);
        assert_eq!(p.gate_for(2, true), 9.0);
        assert_eq!(p.gate_for(1, false), 10.0);
        assert_eq!(p.gate_for(100, true), 24.0);
        assert_eq!(p.gate_for(usize::MAX, false), 24.0);
    }

    /// Two cameras looking at the origin from x and from y, 1000 px focal length, no
    /// distortion.
    fn two_cameras() -> Vec<(CameraModel, RigidTransform)> {
        let k = Matrix3::new(1000.0, 0.0, 512.0, 0.0, 1000.0, 512.0, 0.0, 0.0, 1.0);
        let model = CameraModel {
            intrinsics: Intrinsics {
                camera_matrix: k,
                distortion: None,
            },
            grid: None,
            compat: ModelCompat::default(),
        };
        // Camera 0 looks along +z from z = -50; camera 1 along +x from x = -50 (rotated
        // about y by -90 degrees).
        let p0 = RigidTransform {
            rotation: Vector3::zeros(),
            translation: Vector3::new(0.0, 0.0, 50.0),
        };
        let p1 = RigidTransform {
            rotation: Vector3::new(0.0, -std::f64::consts::FRAC_PI_2, 0.0),
            translation: Vector3::new(0.0, 0.0, 50.0),
        };
        vec![(model.clone(), p0), (model, p1)]
    }

    fn joint<'a>(cams: &'a [(CameraModel, RigidTransform)]) -> Vec<JointCamera<'a>> {
        cams.iter()
            .map(|(m, p)| JointCamera {
                model: m,
                pose: p,
                projection: projection_matrix(&m.intrinsics.camera_matrix, p),
            })
            .collect()
    }

    fn cand(p: Point2, corr: f64) -> Candidate {
        Candidate {
            point: p,
            correlation: corr,
        }
    }

    #[test]
    fn the_gate_stops_a_slide_along_the_epipolar_line() {
        let cams = two_cameras();
        let jc = joint(&cams);
        let truth = Point3::new(1.0, 2.0, 3.0);
        // A second dot on the same epipolar line of camera 1 seen from camera 0: same
        // camera-0 point, a different depth.
        let other = Point3::new(1.0 * 1.3, 2.0 * 1.3, 3.0 + 0.3 * 50.0 + 0.3 * 3.0);
        let q0 = cams[0].0.project_point(&cams[0].1, truth);
        let q0b = cams[0].0.project_point(&cams[0].1, other);
        assert!((q0 - q0b).norm() < 1e-6, "{q0} {q0b}");
        let q1 = cams[1].0.project_point(&cams[1].1, truth);
        let q1b = cams[1].0.project_point(&cams[1].1, other);
        assert!((q1 - q1b).norm() > 20.0);
        let m = MarkerOptions {
            reprojection_limit: None,
            cameras: vec![
                CameraOptions {
                    prediction: Some(q0 + nalgebra::Vector2::new(1.0, 0.5)),
                    gate: 5.0,
                    fixed: None,
                    reacquire: false,
                    near_gate: 0.0,
                    candidates: vec![cand(q0, 0.9)],
                },
                CameraOptions {
                    prediction: Some(q1 + nalgebra::Vector2::new(-1.0, 1.0)),
                    gate: 5.0,
                    fixed: None,
                    reacquire: false,
                    near_gate: 0.0,
                    // The wrong dot correlates better.
                    candidates: vec![cand(q1b, 0.99), cand(q1, 0.8)],
                },
            ],
        };
        let r = choose_joint(
            &jc,
            &[m],
            TriangulationMethod::default(),
            &MultiTrackParams::default(),
        );
        assert_eq!(r[0].picks, vec![Pick::Candidate(0), Pick::Candidate(1)]);
        let x = r[0].point3d.unwrap();
        assert!((x - truth).norm() < 1e-6);
    }

    /// Two markers whose dots coincide in camera 0 (points on one ray of camera 0) and are
    /// apart in camera 1: marker 0, followed, has the shared dot of camera 0 and its own dot
    /// of camera 1; marker 1 has the same dot of camera 0 (its prediction on it) and its own
    /// dot of camera 1. `reacquire` says whether marker 1 is being found again in camera 0,
    /// `pred1` moves its prediction there, and `k0` gives marker 0 no candidate in camera 0
    /// (followed there, but nothing found: "no point there").
    fn one_dot_two_markers(
        reacquire: bool,
        pred1: nalgebra::Vector2<f64>,
        k0: bool,
    ) -> (Vec<(CameraModel, RigidTransform)>, Vec<MarkerOptions>) {
        let cams = two_cameras();
        let a = Point3::new(1.0, 2.0, 3.0);
        let b = Point3::new(1.3, 2.6, 3.0 + 0.3 * 50.0 + 0.3 * 3.0);
        let pa: Vec<Point2> = cams.iter().map(|(m, p)| m.project_point(p, a)).collect();
        let pb: Vec<Point2> = cams.iter().map(|(m, p)| m.project_point(p, b)).collect();
        assert!((pa[0] - pb[0]).norm() < 1e-6 && (pa[1] - pb[1]).norm() > 20.0);
        let opt = |pred: Point2, c: Vec<Point2>, reacquire: bool, gate: f64| CameraOptions {
            prediction: Some(pred),
            gate,
            fixed: None,
            reacquire,
            near_gate: 0.0,
            candidates: c.into_iter().map(|q| cand(q, 0.9)).collect(),
        };
        let m0 = MarkerOptions {
            reprojection_limit: None,
            cameras: vec![
                opt(pa[0], if k0 { vec![pa[0]] } else { Vec::new() }, false, 5.0),
                opt(pa[1], vec![pa[1]], false, 5.0),
            ],
        };
        let m1 = MarkerOptions {
            reprojection_limit: None,
            cameras: vec![
                opt(pb[0] + pred1, vec![pb[0]], reacquire, 24.0),
                opt(pb[1], vec![pb[1]], false, 5.0),
            ],
        };
        (cams, vec![m0, m1])
    }

    fn no_identity_check() -> MultiTrackParams {
        MultiTrackParams {
            reacquire_links: 0,
            ..MultiTrackParams::default()
        }
    }

    /// `reacquire_exclusive`: a marker found again in a camera does not share the dot another
    /// marker holds there, though its prediction is on it (the old rule shares it).
    #[test]
    fn a_marker_found_again_does_not_share_a_dot() {
        let (cams, ms) = one_dot_two_markers(true, nalgebra::Vector2::new(1.0, 0.0), true);
        let jc = joint(&cams);
        // Without claims, which would refuse the dot too.
        let on = MultiTrackParams {
            claim_radius: 0.0,
            ..no_identity_check()
        };
        let r = choose_joint_linked(&jc, &ms, &[], TriangulationMethod::default(), &on);
        assert_eq!(r[0].picks, vec![Pick::Candidate(0); 2], "{r:?}");
        assert_eq!(r[1].picks[0], Pick::None, "{r:?}");
        assert_eq!(r[1].refusals[0], Some(Refusal::TakenByAnother));
        let off = MultiTrackParams {
            reacquire_exclusive: false,
            ..on
        };
        let r = choose_joint_linked(&jc, &ms, &[], TriangulationMethod::default(), &off);
        assert_eq!(r[1].picks, vec![Pick::Candidate(0); 2], "{r:?}");
    }

    /// `share_needs_overlap`: two followed markers share a dot only when the caller says
    /// they overlap in that camera; without the overlap the second is refused there. Without
    /// overlap information (`choose_joint_linked`), the old rule shares it.
    #[test]
    fn markers_share_a_dot_only_where_they_overlap() {
        let (cams, ms) = one_dot_two_markers(false, nalgebra::Vector2::new(1.0, 0.0), true);
        let jc = joint(&cams);
        let p = MultiTrackParams::default();
        let tm = TriangulationMethod::default();
        let r = choose_joint_overlapping(&jc, &ms, &[], &[], tm, &p);
        assert_eq!(r[1].picks[0], Pick::None, "{r:?}");
        assert_eq!(r[1].refusals[0], Some(Refusal::TakenByAnother));
        let both = [Overlap {
            a: 1,
            b: 0,
            camera: 0,
            touching: true,
        }];
        let r = choose_joint_overlapping(&jc, &ms, &[], &both, tm, &p);
        assert_eq!(r[1].picks, vec![Pick::Candidate(0); 2], "{r:?}");
        // Overlapping in the other camera only: not here.
        let elsewhere = [Overlap {
            a: 0,
            b: 1,
            camera: 1,
            touching: true,
        }];
        let r = choose_joint_overlapping(&jc, &ms, &[], &elsewhere, tm, &p);
        assert_eq!(r[1].picks[0], Pick::None, "{r:?}");
        let r = choose_joint_linked(&jc, &ms, &[], tm, &p);
        assert_eq!(r[1].picks, vec![Pick::Candidate(0); 2], "{r:?}");
        let off = MultiTrackParams {
            share_needs_overlap: false,
            ..p
        };
        let r = choose_joint_overlapping(&jc, &ms, &[], &[], tm, &off);
        assert_eq!(r[1].picks, vec![Pick::Candidate(0); 2], "{r:?}");
        // Out of range: ignored.
        let bad = [Overlap {
            a: 0,
            b: 9,
            camera: 7,
            touching: true,
        }];
        let r = choose_joint_overlapping(&jc, &ms, &[], &bad, tm, &p);
        assert_eq!(r[1].picks[0], Pick::None, "{r:?}");
    }

    /// Two markers in one epipolar plane of the two cameras (y = 0): their dots in camera 0
    /// lie on one row, `sep` px apart, and either camera-1 dot triangulates with either
    /// camera-0 dot without reprojection error, so only the predictions in camera 0 tell
    /// them apart. Marker 0's prediction there is `off0` px from its own dot towards the
    /// other's, marker 1's `off1` px from its own; both offer both dots in camera 0 (`both`)
    /// or only the first (marker 0's) dot.
    fn two_in_one_plane(
        off0: f64,
        off1: f64,
        both: bool,
    ) -> (Vec<(CameraModel, RigidTransform)>, Vec<MarkerOptions>, f64) {
        let cams = two_cameras();
        let a = Point3::new(1.0, 0.0, 3.0);
        let b = Point3::new(1.2, 0.0, 2.0);
        let pa: Vec<Point2> = cams.iter().map(|(m, p)| m.project_point(p, a)).collect();
        let pb: Vec<Point2> = cams.iter().map(|(m, p)| m.project_point(p, b)).collect();
        let sep = (pa[0] - pb[0]).norm();
        assert!((pa[0].y - pb[0].y).abs() < 1e-9 && sep > 3.0 && (pa[1] - pb[1]).norm() > 15.0);
        let toward = (pb[0] - pa[0]) / sep;
        let dots0: Vec<Point2> = if both {
            vec![pa[0], pb[0]]
        } else {
            vec![pa[0]]
        };
        let opt = |pred: Point2, c: Vec<Point2>| CameraOptions {
            prediction: Some(pred),
            gate: 5.0,
            fixed: None,
            reacquire: false,
            near_gate: 0.0,
            candidates: c.into_iter().map(|q| cand(q, 0.9)).collect(),
        };
        let m0 = MarkerOptions {
            reprojection_limit: None,
            cameras: vec![
                opt(pa[0] + toward * off0, dots0.clone()),
                opt(pa[1], vec![pa[1]]),
            ],
        };
        let target1 = if both { pb[0] } else { pa[0] };
        let m1 = MarkerOptions {
            reprojection_limit: None,
            cameras: vec![opt(target1 - toward * off1, dots0), opt(pb[1], vec![pb[1]])],
        };
        (cams, vec![m0, m1], sep)
    }

    /// `overlap_decision`: two overlapping markers whose dots in camera 0 only their
    /// predictions tell apart, and those barely (marker 0 predicted 1 px from its dot, marker
    /// 1 just on its side of midway). The choice gives each its own dot (its ambiguity margin
    /// is set low); the
    /// exchange costs less than `overlap_decision` more, so camera 0 is left empty for both
    /// ([`Refusal::Overlap`]), and kept with a smaller margin, with the rule off, or without
    /// the overlap.
    #[test]
    fn identity_out_of_an_overlap_is_not_guessed() {
        let (cams, ms0, sep) = two_in_one_plane(1.0, 0.0, true);
        let (_, ms, _) = two_in_one_plane(1.0, 0.5 * sep - 0.3, true);
        assert!(ms0[1].cameras[0].prediction.is_some());
        let jc = joint(&cams);
        let tm = TriangulationMethod::default();
        let over = [Overlap {
            a: 0,
            b: 1,
            camera: 0,
            touching: true,
        }];
        let p = MultiTrackParams {
            ambiguity_margin: 0.05,
            ..MultiTrackParams::default()
        };
        let r = choose_joint_overlapping(&jc, &ms, &[], &over, tm, &p);
        assert_eq!(r[0].picks[0], Pick::None, "{r:?}");
        assert_eq!(r[1].picks[0], Pick::None, "{r:?}");
        assert_eq!(r[0].refusals[0], Some(Refusal::Overlap));
        assert_eq!(r[1].refusals[0], Some(Refusal::Overlap));
        assert_eq!(r[0].picks[1], Pick::Candidate(0));
        // Without the rule, with a margin below the difference, or without the overlap:
        // each keeps its own dot.
        for (q, o) in [
            (
                MultiTrackParams {
                    overlap_decision: 0.0,
                    ..p
                },
                &over[..],
            ),
            (
                MultiTrackParams {
                    overlap_decision: 0.05,
                    ..p
                },
                &over[..],
            ),
            (p, &[][..]),
        ] {
            let r = choose_joint_overlapping(&jc, &ms, &[], o, tm, &q);
            assert_eq!(r[0].picks[0], Pick::Candidate(0), "{r:?}");
            assert_eq!(r[1].picks[0], Pick::Candidate(1), "{r:?}");
        }
        // Each predicted on its own dot: the exchange is clearly worse, and both keep theirs.
        let (_, ms, _) = two_in_one_plane(0.0, 0.0, true);
        let r = choose_joint_overlapping(&jc, &ms, &[], &over, tm, &p);
        assert_eq!(r[0].picks[0], Pick::Candidate(0), "{r:?}");
        assert_eq!(r[1].picks[0], Pick::Candidate(1), "{r:?}");
    }

    /// The pairs' decisions are applied together: marker 0 holds the only dot of camera 0
    /// that marker 2 fits about as well (that pair asks marker 0 for nothing there), and
    /// marker 1, far away on a dot of its own, makes a pair with it that asks it to keep its
    /// dot. Asked two different things, marker 0 gets nothing there; without the second
    /// pair, it keeps the dot. The same with the pairs given in either order. (Mutation: a
    /// marker given the first ask instead of nothing.)
    #[test]
    fn two_decisions_that_disagree_give_nothing() {
        let cams = two_cameras();
        let jc = joint(&cams);
        let tm = TriangulationMethod::default();
        let (_, mut ms, _) = two_in_one_plane(0.5, 0.0, false);
        let q = ms[0].cameras[0].candidates[0].point;
        ms[1].cameras[0].prediction = Some(q + nalgebra::Vector2::new(0.0, 2.0));
        let far = Point3::new(-2.0, 1.5, 2.5);
        let pf: Vec<Point2> = cams.iter().map(|(m, p)| m.project_point(p, far)).collect();
        let lone = MarkerOptions {
            reprojection_limit: None,
            cameras: pf
                .iter()
                .map(|&x| CameraOptions {
                    prediction: Some(x),
                    gate: 5.0,
                    fixed: None,
                    reacquire: false,
                    near_gate: 0.0,
                    candidates: vec![cand(x, 0.9)],
                })
                .collect(),
        };
        // Markers: 0 the holder, 1 the far one, 2 the one that fits the dot.
        let ms = vec![ms[0].clone(), lone, ms[1].clone()];
        let pair = |a: usize, b: usize| Overlap {
            a,
            b,
            camera: 0,
            touching: false,
        };
        let p = MultiTrackParams::default();
        let r = choose_joint_overlapping(&jc, &ms, &[], &[pair(0, 1)], tm, &p);
        assert_eq!(r[0].picks[0], Pick::Candidate(0), "{r:?}");
        assert_eq!(r[1].picks, vec![Pick::Candidate(0); 2], "{r:?}");
        for list in [[pair(0, 1), pair(0, 2)], [pair(2, 0), pair(1, 0)]] {
            let r = choose_joint_overlapping(&jc, &ms, &[], &list, tm, &p);
            assert_eq!(r[0].picks[0], Pick::None, "{r:?}");
            assert_eq!(r[0].refusals[0], Some(Refusal::Overlap), "{r:?}");
            assert_eq!(r[1].picks, vec![Pick::Candidate(0); 2], "{r:?}");
        }
    }

    /// `settle`'s contract, called directly (the case of two cameras or more with no feasible
    /// combination of exactly those picks is not reached by the random scenes of review 17:
    /// it needs two pairs' decisions to change one marker in two cameras, or a third camera;
    /// `settle` documents how). Exact picks: that combination's 3D point and cost. Two picks
    /// without a combination: nothing anywhere. One pick without a combination: nothing with
    /// `overlap_single_checked`, kept without a 3D point without it. (Mutations: two picks
    /// without a combination kept; a lone pick kept although checked.)
    #[test]
    fn settling_a_changed_marker() {
        let x = Point3::new(1.0, 2.0, 3.0);
        let combo = |picks: Vec<Pick>, point3d: Option<Point3>| Combo {
            dots: vec![0; picks.len()],
            picks,
            cost: 0.25,
            class: 0,
            priority: 0.25,
            point3d,
        };
        let combos = vec![
            combo(vec![Pick::Candidate(0), Pick::Candidate(0)], Some(x)),
            combo(vec![Pick::Candidate(1), Pick::None], None),
        ];
        let choice = |picks: Vec<Pick>| MarkerChoice {
            refusals: vec![None; picks.len()],
            picks,
            cost: Some(9.0),
            point3d: Some(Point3::new(9.0, 9.0, 9.0)),
        };
        // Exactly a feasible combination: its 3D point and cost.
        for checked in [true, false] {
            let mut ch = choice(vec![Pick::Candidate(0), Pick::Candidate(0)]);
            settle(&combos, &mut ch, checked);
            assert_eq!((ch.point3d, ch.cost), (Some(x), Some(0.25)));
            let mut ch = choice(vec![Pick::Candidate(1), Pick::None]);
            settle(&combos, &mut ch, checked);
            assert_eq!(ch.picks, vec![Pick::Candidate(1), Pick::None]);
            assert_eq!(ch.point3d, None);
        }
        // Two picks, no combination: nothing anywhere, either way.
        for checked in [true, false] {
            let mut ch = choice(vec![Pick::Candidate(1), Pick::Candidate(0)]);
            settle(&combos, &mut ch, checked);
            assert_eq!(ch.picks, vec![Pick::None; 2]);
            assert_eq!(ch.refusals, vec![Some(Refusal::Overlap); 2]);
            assert_eq!((ch.point3d, ch.cost), (None, None));
        }
        // One pick, no combination (no point on its own).
        let mut ch = choice(vec![Pick::None, Pick::Candidate(0)]);
        settle(&combos, &mut ch, true);
        assert_eq!(ch.picks, vec![Pick::None; 2]);
        assert_eq!(ch.refusals[1], Some(Refusal::Overlap));
        assert_eq!(ch.point3d, None);
        let mut ch = choice(vec![Pick::None, Pick::Candidate(0)]);
        settle(&combos, &mut ch, false);
        assert_eq!(ch.picks, vec![Pick::None, Pick::Candidate(0)]);
        assert_eq!((ch.point3d, ch.cost), (None, None));
    }

    /// The exchange of rule 4 is taken only with exactly the markers' other picks. Two
    /// markers in one epipolar plane, each given the other's dot in camera 0 by the choice
    /// (its prediction on the other dot); marker 0 holds no point in camera 1, marker 1 its
    /// own. The exchange is clearly cheaper. Marker 0 cannot take marker 1's dot with nothing
    /// in camera 1 (alone, the other dot 4 px away is within `single_ambiguity`), so neither
    /// takes a dot in camera 0. (Mutation, checked: the exchange taken with the looser
    /// combination, which gives marker 1 marker 0's dot while marker 0 ends with nothing.)
    #[test]
    fn an_exchange_is_taken_only_with_exactly_the_other_picks() {
        let (cams, ms, sep) = two_in_one_plane(0.0, 0.0, true);
        let mut ms = ms;
        // Predictions on the other marker's dot in camera 0.
        let d1 = ms[0].cameras[0].candidates[0].point;
        let d2 = ms[0].cameras[0].candidates[1].point;
        assert!(((d2 - d1).norm() - sep).abs() < 1e-9);
        ms[0].cameras[0].prediction = Some(d2);
        ms[1].cameras[0].prediction = Some(d1);
        let jc = joint(&cams);
        let p = MultiTrackParams::default();
        let (ids, _) = dot_ids(&ms, 2, p.same_dot);
        let combos: Vec<Vec<Combo>> = ms
            .iter()
            .zip(&ids)
            .map(|(m, id)| feasible_combos(&jc, m, id, TriangulationMethod::default(), &p))
            .collect();
        let partners: Vec<Vec<Partner>> = vec![Vec::new(); 2];
        let ctx = Filters {
            markers: &ms,
            partners: &partners,
            params: &p,
            n_cam: 2,
        };
        let given = |picks: Vec<Pick>| MarkerChoice {
            refusals: vec![None; 2],
            picks,
            cost: None,
            point3d: None,
        };
        // The choice's result to decide on: each on its own dot of camera 0 (the wrong way
        // round for the predictions); marker 0 nothing in camera 1.
        let mut out = vec![
            given(vec![Pick::Candidate(0), Pick::None]),
            given(vec![Pick::Candidate(1), Pick::Candidate(0)]),
        ];
        let over = [Overlap {
            a: 0,
            b: 1,
            camera: 0,
            touching: false,
        }];
        decide_overlaps(&over, &ids, &combos, &ctx, &[None, None], &mut out);
        assert_eq!(out[0].picks[0], Pick::None, "{out:?}");
        assert_eq!(out[1].picks[0], Pick::None, "{out:?}");
        assert_eq!(out[1].refusals[0], Some(Refusal::Overlap), "{out:?}");
        assert_eq!(out[1].picks[1], Pick::Candidate(0), "{out:?}");
        // With marker 0 holding its camera-1 point, the exchange is taken.
        let mut out = vec![
            given(vec![Pick::Candidate(0), Pick::Candidate(0)]),
            given(vec![Pick::Candidate(1), Pick::Candidate(0)]),
        ];
        decide_overlaps(&over, &ids, &combos, &ctx, &[None, None], &mut out);
        assert_eq!(
            out[0].picks,
            vec![Pick::Candidate(1), Pick::Candidate(0)],
            "{out:?}"
        );
        assert_eq!(
            out[1].picks,
            vec![Pick::Candidate(0), Pick::Candidate(0)],
            "{out:?}"
        );
        assert!(out.iter().all(|o| o.point3d.is_some()), "{out:?}");
    }

    /// `overlap_take` off: two markers touching in camera 0 get no point there, and what is
    /// left is consistent: each keeps its camera-1 point and has no 3D point (one camera);
    /// with the rule on, each keeps its own dot and its 3D point. (Mutation: the result not
    /// settled after the change, which left the 3D point of the dot taken away.)
    #[test]
    fn no_point_while_touching_leaves_a_consistent_result() {
        let (cams, ms, _) = two_in_one_plane(0.0, 0.0, true);
        let jc = joint(&cams);
        let tm = TriangulationMethod::default();
        let over = [Overlap {
            a: 0,
            b: 1,
            camera: 0,
            touching: true,
        }];
        let p = MultiTrackParams {
            overlap_take: false,
            ..MultiTrackParams::default()
        };
        let r = choose_joint_overlapping(&jc, &ms, &[], &over, tm, &p);
        for (k, ch) in r.iter().enumerate() {
            assert_eq!(ch.picks[0], Pick::None, "{r:?}");
            assert_eq!(ch.refusals[0], Some(Refusal::Overlap), "{r:?}");
            assert_eq!(ch.picks[1], Pick::Candidate(0), "marker {k}: {r:?}");
            assert!(ch.point3d.is_none(), "marker {k}: {r:?}");
        }
        let r = choose_joint_overlapping(&jc, &ms, &[], &over, tm, &MultiTrackParams::default());
        for ch in &r {
            assert!(
                ch.picks.iter().all(|q| *q != Pick::None) && ch.point3d.is_some(),
                "{r:?}"
            );
        }
    }

    /// `overlap_decision` with one dot, for two markers held in an overlap that no longer
    /// touch: marker 0 is given the only dot of camera 0 (served first), marker 1 none. When
    /// marker 1 fits the dot about as well, marker 0 does not keep it either; when marker 1
    /// fits it clearly better, the dot is marker 1's. While they touch, the choice's own
    /// result stands.
    #[test]
    fn one_dot_out_of_an_overlap_goes_to_the_marker_it_fits() {
        let cams = two_cameras();
        let jc = joint(&cams);
        let tm = TriangulationMethod::default();
        let over = [Overlap {
            a: 0,
            b: 1,
            camera: 0,
            touching: false,
        }];
        let p = MultiTrackParams::default();
        // Marker 0 predicted 0.5 px from the dot, marker 1 2 px: marker 0 is served first and
        // takes it; their costs differ by less than the margin.
        let (_, mut ms, _) = two_in_one_plane(0.5, 0.0, false);
        let q = ms[0].cameras[0].candidates[0].point;
        ms[1].cameras[0].prediction = Some(q + nalgebra::Vector2::new(0.0, 2.0));
        let r = choose_joint_linked(&jc, &ms, &[], tm, &p);
        assert_eq!(r[0].picks[0], Pick::Candidate(0), "{r:?}");
        assert_eq!(r[1].picks[0], Pick::None, "{r:?}");
        let r = choose_joint_overlapping(&jc, &ms, &[], &over, tm, &p);
        assert_eq!(r[0].picks[0], Pick::None, "{r:?}");
        assert_eq!(r[0].refusals[0], Some(Refusal::Overlap));
        assert_eq!(r[1].picks[0], Pick::None, "{r:?}");
        // Marker 1 found again in camera 0 and predicted on the dot, marker 0 predicted 4 px
        // away and with a weak match in camera 1. Marker 0 is served first (a marker found
        // again is served last) and, with no contest, takes the dot; marker 1 fits it clearly
        // better, so the dot is marker 1's.
        ms[0].cameras[0].prediction = Some(q + nalgebra::Vector2::new(0.0, 4.0));
        ms[0].cameras[1].candidates[0].correlation = 0.1;
        ms[1].cameras[0].prediction = Some(q);
        ms[1].cameras[0].reacquire = true;
        let p = MultiTrackParams {
            contest_ratio: 0.0,
            reacquire_links: 0,
            claim_radius: 0.0,
            ..p
        };
        let r0 = choose_joint_linked(&jc, &ms, &[], tm, &p);
        assert_eq!(r0[0].picks[0], Pick::Candidate(0), "{r0:?}");
        assert_eq!(r0[1].picks, vec![Pick::None, Pick::Candidate(0)], "{r0:?}");
        let r = choose_joint_overlapping(&jc, &ms, &[], &over, tm, &p);
        assert_eq!(r[0].picks[0], Pick::None, "{r:?}");
        assert_eq!(r[0].refusals[0], Some(Refusal::Overlap));
        assert_eq!(r[1].picks, vec![Pick::Candidate(0); 2], "{r:?}");
    }

    /// Claims (`claim_radius`): marker 0 is followed in camera 0 with nothing found there (no
    /// point of its own); marker 1, found again there with its prediction 8 px away, may not
    /// take the dot on marker 0's prediction ([`Refusal::Claimed`]), unless its own
    /// prediction is clearly closer (`claim_ratio`). Without claims it takes it.
    #[test]
    fn a_dot_where_another_marker_is_expected_is_claimed() {
        let away = nalgebra::Vector2::new(8.0, 0.0);
        let (cams, ms) = one_dot_two_markers(true, away, false);
        let jc = joint(&cams);
        let tm = TriangulationMethod::default();
        let on = no_identity_check();
        let r = choose_joint_linked(&jc, &ms, &[], tm, &on);
        assert_eq!(r[1].picks[0], Pick::None, "{r:?}");
        assert_eq!(r[1].refusals[0], Some(Refusal::Claimed), "{r:?}");
        let off = MultiTrackParams {
            claim_radius: 0.0,
            ..on
        };
        let r = choose_joint_linked(&jc, &ms, &[], tm, &off);
        assert_eq!(r[1].picks, vec![Pick::Candidate(0); 2], "{r:?}");
        // Its own prediction on the dot and marker 0's 6 px away: its own.
        let (cams, mut ms) = one_dot_two_markers(true, nalgebra::Vector2::zeros(), false);
        let q = ms[0].cameras[0].prediction.unwrap();
        ms[0].cameras[0].prediction = Some(q + nalgebra::Vector2::new(0.0, 5.5));
        let jc = joint(&cams);
        let r = choose_joint_linked(&jc, &ms, &[], tm, &on);
        assert_eq!(r[1].picks, vec![Pick::Candidate(0); 2], "{r:?}");
    }

    #[test]
    fn ambiguity_and_sharing() {
        let cams = two_cameras();
        let jc = joint(&cams);
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(0.0, 0.02, 0.0); // under a pixel apart in both cameras
        let pa: Vec<Point2> = cams.iter().map(|(m, p)| m.project_point(p, a)).collect();
        let pb: Vec<Point2> = cams.iter().map(|(m, p)| m.project_point(p, b)).collect();
        let opts = |pred: &[Point2], cs: [&[Point2]; 2]| MarkerOptions {
            reprojection_limit: None,
            cameras: (0..2)
                .map(|c| CameraOptions {
                    prediction: Some(pred[c]),
                    gate: 5.0,
                    fixed: None,
                    reacquire: false,
                    near_gate: 0.0,
                    candidates: cs[c].iter().map(|&p| cand(p, 0.9)).collect(),
                })
                .collect(),
        };
        // One marker, two dots 3 px apart in camera 0 on the prediction's epipolar line:
        // ambiguous there, accepted in camera 1.
        let shift = nalgebra::Vector2::new(3.0, 0.0);
        let m = opts(&pa, [&[pa[0], pa[0] + shift], &[pa[1]]]);
        let p = MultiTrackParams::default();
        let r = choose_joint(&jc, &[m], TriangulationMethod::default(), &p);
        // pa[0] + shift is not consistent with pa[1] unless along the epipolar line; whether
        // it is depends on the geometry: the camera-1 point is accepted in any case.
        assert_eq!(r[0].picks[1], Pick::Candidate(0));
        // Two markers whose best combinations use the same dots in every camera: two
        // markers on one point, so only the closer one gets them, with sharing or without.
        let m1 = opts(&pa, [&[pa[0]], &[pa[1]]]);
        let m2 = opts(&pb, [&[pa[0]], &[pa[1]]]);
        let count = |r: &[MarkerChoice]| -> usize {
            r.iter()
                .map(|c| c.picks.iter().filter(|k| **k != Pick::None).count())
                .sum()
        };
        let r = choose_joint(
            &jc,
            &[m1.clone(), m2.clone()],
            TriangulationMethod::default(),
            &p,
        );
        assert_eq!(count(&r), 2, "{r:?}");
        assert_eq!(r[1].refusals, vec![Some(Refusal::TakenByAnother); 2]);
        let mut q = p;
        q.share_radius = 0.0;
        let r = choose_joint(
            &jc,
            &[m1.clone(), m2.clone()],
            TriangulationMethod::default(),
            &q,
        );
        assert_eq!(count(&r), 2, "{r:?}");
        assert_eq!(r[0].picks, vec![Pick::Candidate(0), Pick::Candidate(0)]);
        assert_eq!(r[1].refusals, vec![Some(Refusal::TakenByAnother); 2]);
        // Equally close: neither gets the contested dots.
        let r = choose_joint(&jc, &[m1.clone(), m1], TriangulationMethod::default(), &q);
        assert_eq!(count(&r), 0, "{r:?}");
        // With contests off the marker served first takes them and the other is refused.
        q.contest_ratio = 0.0;
        let r = choose_joint(&jc, &[m2.clone(), m2], TriangulationMethod::default(), &q);
        assert_eq!(count(&r), 2, "{r:?}");
        assert!(r[1].refusals.contains(&Some(Refusal::TakenByAnother)));
    }

    #[test]
    fn dots_by_contrast() {
        let im = blob_image(60, 60, &[(30.0, 30.0), (38.0, 26.0)]);
        let d = contrast_maxima(&im, (31.0, 29.0), 12, 2.5, true, 10.0, 5);
        let pos: Vec<(f64, f64)> = d.iter().map(|x| (x.0, x.1)).collect();
        assert!(pos.contains(&(30.0, 30.0)), "{d:?}");
        assert!(pos.contains(&(38.0, 26.0)), "{d:?}");
        assert!(contrast_maxima(&im, (f64::NAN, 1.0), 5, 2.5, true, 0.0, 5).is_empty());
        assert!(contrast_maxima(&im, (5.0, 5.0), i32::MAX, f64::NAN, true, f64::NAN, 3).len() <= 3);
    }

    #[test]
    fn contrast_of_a_dot() {
        let im = blob_image(40, 40, &[(20.0, 20.0)]);
        let dark = dot_contrast(&im, (20.0, 20.0), 2.5, true).unwrap();
        assert!(dark > 40.0, "{dark}");
        assert_eq!(dot_contrast(&im, (20.0, 20.0), 2.5, false), Some(-dark));
        let flat = dot_contrast(&im, (5.0, 30.0), 2.5, true).unwrap();
        assert!(flat.abs() < 1.0, "{flat}");
        // Off the image the edge is replicated.
        assert!(dot_contrast(&im, (-100.0, 5.0), 2.5, true).is_some());
        assert!(dot_contrast(&im, (f64::NAN, 5.0), 2.5, true).is_none());
        assert!(dot_contrast(&im, (5.0, 5.0), 0.0, true).is_none());
        assert!(dot_contrast(&im, (5.0, 5.0), 1e9, true).is_none());
        assert!(dot_contrast(&ImageU8::new(0, 0), (0.0, 0.0), 2.5, true).is_none());
    }

    /// A grey image with dark gaussian shapes (centre, sigma along x, sigma along y) on a
    /// ground that rises by `slope` grey levels a pixel along x from 180 at x = 0.
    fn shapes_image(w: usize, h: usize, slope: f64, shapes: &[(f64, f64, f64, f64)]) -> ImageF32 {
        let mut im = ImageF32::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let mut v = 150.0 + slope * x as f64;
                for &(cx, cy, sx, sy) in shapes {
                    let (dx, dy) = ((x as f64 - cx) / sx, (y as f64 - cy) / sy);
                    v -= 130.0 * (-(dx * dx + dy * dy) / 2.0).exp();
                }
                im.set(x, y, v as f32);
            }
        }
        im
    }

    #[test]
    fn points_on_one_shape() {
        // An elongated shape (about 20 by 6 px): its end and edge are joined to its centre,
        // on flat and on uneven ground.
        for slope in [0.0, 1.0] {
            let im = shapes_image(100, 80, slope, &[(50.0, 40.0, 5.0, 1.5)]);
            for from in [(58.0, 40.0), (42.0, 40.0), (50.0, 42.0), (54.0, 39.0)] {
                assert!(
                    on_one_shape(&im, from, (50.0, 40.0), true),
                    "{slope} {from:?}"
                );
            }
            // A light shape is not a dark one.
            assert!(!on_one_shape(&im, (58.0, 40.0), (50.0, 40.0), false));
        }
        // Two separate dots 10 px apart with bright ground between them.
        let im = shapes_image(
            100,
            80,
            0.0,
            &[(40.0, 40.0, 1.6, 1.6), (50.0, 40.0, 1.6, 1.6)],
        );
        assert!(!on_one_shape(&im, (41.5, 40.0), (50.0, 40.0), true));
        assert!(on_one_shape(&im, (41.5, 40.0), (40.0, 40.0), true));
        // A point on the bright ground is on no shape.
        assert!(!on_one_shape(&im, (45.0, 40.0), (50.0, 40.0), true));
        // Hostile values.
        for (a, b) in [
            ((f64::NAN, 1.0), (40.0, 40.0)),
            ((40.0, 40.0), (f64::INFINITY, 0.0)),
            ((0.0, 0.0), (90.0, 70.0)),
            ((-1e9, 1e9), (40.0, 40.0)),
        ] {
            assert!(!on_one_shape(&im, a, b, true));
        }
        assert!(!on_one_shape(
            &ImageF32::new(0, 0),
            (0.0, 0.0),
            (0.0, 0.0),
            true
        ));
    }

    #[test]
    fn relative_contrast_of_a_dot() {
        let im = blob_image(40, 40, &[(20.0, 20.0)]);
        let k = dot_contrast(&im, (20.0, 20.0), 2.5, true).unwrap();
        let r = dot_contrast_relative(&im, (20.0, 20.0), 2.5, true).unwrap();
        assert!(r > 0.0 && r < 1.0, "{r}");
        // The ring's mean is the contrast over the relative contrast.
        let ring = k / r;
        assert!(ring > k, "{ring} {k}");
        assert_eq!(
            dot_contrast_relative(&im, (20.0, 20.0), 2.5, false),
            Some(-r)
        );
        // The same picture at half the brightness: the contrast halves, the relative
        // contrast does not change.
        let mut half = ImageF32::new(40, 40);
        for y in 0..40 {
            for x in 0..40 {
                half.set(x, y, im.value(y * 40 + x) * 0.5);
            }
        }
        let k2 = dot_contrast(&half, (20.0, 20.0), 2.5, true).unwrap();
        let r2 = dot_contrast_relative(&half, (20.0, 20.0), 2.5, true).unwrap();
        assert!((k2 - 0.5 * k).abs() < 1e-9, "{k2} {k}");
        assert!((r2 - r).abs() < 1e-9, "{r2} {r}");
        // A black ring: divided by 1, not by 0.
        let black = ImageF32::new(40, 40);
        assert_eq!(
            dot_contrast_relative(&black, (20.0, 20.0), 2.5, true),
            Some(0.0)
        );
        for c in [(f64::NAN, 1.0), (1e300, 1.0), (f64::INFINITY, 0.0)] {
            assert!(dot_contrast_relative(&im, c, 2.5, true).is_none());
        }
        for radius in [0.0, -1.0, f64::NAN, 1e9, f64::INFINITY] {
            assert!(dot_contrast_relative(&im, (5.0, 5.0), radius, true).is_none());
        }
        assert!(dot_contrast_relative(&im, (-1e6, 1e6), 50.0, true).is_some());
        assert!(dot_contrast_relative(&ImageU8::new(0, 0), (0.0, 0.0), 2.5, true).is_none());
    }

    #[test]
    fn rigid_links_and_single_points_near_another_marker() {
        let cams = two_cameras();
        let jc = joint(&cams);
        let proj = |x: Point3| -> Vec<Point2> {
            cams.iter().map(|(m, p)| m.project_point(p, x)).collect()
        };
        let a = Point3::new(0.0, 0.0, 0.0);
        let b_true = Point3::new(0.0, 1.0, 0.0);
        let b_wrong = Point3::new(0.0, 1.0, 0.15);
        let (pa, pt, pw) = (proj(a), proj(b_true), proj(b_wrong));
        let marker = |pred: &[Point2], cs: &[Vec<Candidate>]| MarkerOptions {
            reprojection_limit: None,
            cameras: (0..2)
                .map(|c| CameraOptions {
                    prediction: Some(pred[c]),
                    gate: 5.0,
                    fixed: None,
                    reacquire: false,
                    near_gate: 0.0,
                    candidates: cs[c].clone(),
                })
                .collect(),
        };
        let ma = marker(&pa, &[vec![cand(pa[0], 1.0)], vec![cand(pa[1], 1.0)]]);
        // B is predicted at the wrong place, where its best candidates are; the right ones
        // are within the gate but cost more.
        let mb = marker(
            &pw,
            &[
                vec![cand(pw[0], 0.95), cand(pt[0], 0.9)],
                vec![cand(pw[1], 0.95), cand(pt[1], 0.9)],
            ],
        );
        let p = MultiTrackParams {
            ambiguity_margin: 0.0,
            ..MultiTrackParams::default()
        };
        let r = choose_joint(
            &jc,
            &[ma.clone(), mb.clone()],
            TriangulationMethod::default(),
            &p,
        );
        assert_eq!(r[1].picks, vec![Pick::Candidate(0), Pick::Candidate(0)]);
        let link = RigidLink {
            a: 0,
            b: 1,
            distance: 1.0,
            tolerance: 0.005,
        };
        let r = choose_joint_linked(
            &jc,
            &[ma.clone(), mb.clone()],
            &[link],
            TriangulationMethod::default(),
            &p,
        );
        assert_eq!(r[0].picks, vec![Pick::Candidate(0), Pick::Candidate(0)]);
        // (In camera 0 the two candidates of B are the same dot.)
        assert_eq!(r[1].picks[1], Pick::Candidate(1), "{r:?}");
        assert!((r[1].point3d.unwrap() - b_true).norm() < 0.01, "{r:?}");
        // A link no triangulated combination keeps leaves B to a single camera at most (a
        // point seen by one camera has no 3D position to test), and with no single-camera
        // option, undefined.
        let far = RigidLink {
            distance: 5.0,
            ..link
        };
        let r = choose_joint_linked(
            &jc,
            &[ma.clone(), mb.clone()],
            &[far],
            TriangulationMethod::default(),
            &p,
        );
        assert_eq!(r[1].point3d, None);
        assert_eq!(r[1].picks[1], Pick::None);
        let mut strict = p;
        strict.single_gate = -1.0; // no single-camera points
        let r = choose_joint_linked(
            &jc,
            &[ma.clone(), mb.clone()],
            &[far],
            TriangulationMethod::default(),
            &strict,
        );
        assert_eq!(r[1].picks, vec![Pick::None, Pick::None]);
        assert_eq!(r[1].refusals, vec![Some(Refusal::Rigid); 2]);
        // Links out of range or not finite are ignored.
        let junk = [
            RigidLink { a: 0, b: 9, ..link },
            RigidLink {
                distance: f64::NAN,
                ..far
            },
            RigidLink {
                tolerance: -1.0,
                ..far
            },
        ];
        let r = choose_joint_linked(
            &jc,
            &[ma.clone(), mb],
            &junk,
            TriangulationMethod::default(),
            &p,
        );
        assert_eq!(r[1].picks, vec![Pick::Candidate(0), Pick::Candidate(0)]);
        // A single-camera point next to another marker's prediction is refused.
        let lone = MarkerOptions {
            reprojection_limit: None,
            cameras: vec![
                CameraOptions {
                    prediction: Some(pa[0] + nalgebra::Vector2::new(4.0, 0.0)),
                    gate: 5.0,
                    fixed: None,
                    reacquire: false,
                    near_gate: 0.0,
                    candidates: vec![cand(pa[0] + nalgebra::Vector2::new(3.0, 0.0), 0.9)],
                },
                CameraOptions::default(),
            ],
        };
        let r = choose_joint(
            &jc,
            std::slice::from_ref(&lone),
            TriangulationMethod::default(),
            &p,
        );
        assert_eq!(r[0].picks, vec![Pick::Candidate(0), Pick::None]);
        let r = choose_joint(&jc, &[lone, ma], TriangulationMethod::default(), &p);
        assert_eq!(r[0].picks, vec![Pick::None, Pick::None]);
    }

    #[test]
    fn single_camera_points() {
        let cams = two_cameras();
        let jc = joint(&cams);
        let pr = Point2::new(500.0, 500.0);
        let m = |cs: Vec<Candidate>| MarkerOptions {
            reprojection_limit: None,
            cameras: vec![
                CameraOptions {
                    prediction: Some(pr),
                    gate: 5.0,
                    fixed: None,
                    reacquire: false,
                    near_gate: 0.0,
                    candidates: cs,
                },
                CameraOptions::default(),
            ],
        };
        let p = MultiTrackParams::default();
        let near = Point2::new(501.0, 500.5);
        let r = choose_joint(
            &jc,
            &[m(vec![cand(near, 0.9)])],
            TriangulationMethod::default(),
            &p,
        );
        assert_eq!(r[0].picks, vec![Pick::Candidate(0), Pick::None]);
        // Too far for a single camera.
        let far = Point2::new(505.0, 500.0);
        let r = choose_joint(
            &jc,
            &[m(vec![cand(far, 0.9)])],
            TriangulationMethod::default(),
            &p,
        );
        assert_eq!(r[0].picks, vec![Pick::None, Pick::None]);
        // Another dot close by: ambiguous, refused.
        let r = choose_joint(
            &jc,
            &[m(vec![
                cand(near, 0.9),
                cand(Point2::new(505.0, 501.0), 0.5),
            ])],
            TriangulationMethod::default(),
            &p,
        );
        assert_eq!(r[0].picks, vec![Pick::None, Pick::None]);
    }

    /// Finding a marker again: the number of rigid links its identity needs is
    /// `reacquire_links`, but no more than the partners it has (at least one).
    #[test]
    fn identity_needs_no_more_links_than_partners() {
        let cams = two_cameras();
        let jc = joint(&cams);
        let proj = |x: Point3| -> Vec<Point2> {
            cams.iter().map(|(m, p)| m.project_point(p, x)).collect()
        };
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(0.0, 1.0, 0.0);
        let (pa, pb) = (proj(a), proj(b));
        let marker = |pts: &[Point2], reacquire: bool| MarkerOptions {
            reprojection_limit: None,
            cameras: (0..2)
                .map(|c| CameraOptions {
                    prediction: Some(pts[c]),
                    gate: 5.0,
                    fixed: None,
                    reacquire,
                    near_gate: 0.0,
                    candidates: vec![cand(pts[c], 0.95)],
                })
                .collect(),
        };
        // Marker 0 followed, marker 1 found again; markers 2 and 3 offer nothing.
        let ms = [
            marker(&pa, false),
            marker(&pb, true),
            MarkerOptions::default(),
            MarkerOptions::default(),
        ];
        let link = |a: usize, b: usize| RigidLink {
            a,
            b,
            distance: 1.0,
            tolerance: 0.01,
        };
        let p = MultiTrackParams::default();
        assert_eq!(p.reacquire_links, 2);
        let found = vec![Pick::Candidate(0), Pick::Candidate(0)];
        // One partner, and it is its only one: that one confirms it.
        let r = choose_joint_linked(&jc, &ms, &[link(0, 1)], TriangulationMethod::default(), &p);
        assert_eq!(r[1].picks, found, "{r:?}");
        // The same link given twice is still one partner.
        let r = choose_joint_linked(
            &jc,
            &ms,
            &[link(0, 1), link(1, 0)],
            TriangulationMethod::default(),
            &p,
        );
        assert_eq!(r[1].picks, found, "{r:?}");
        // Three partners of which only one is accepted at the frame: still two needed.
        let three = [link(0, 1), link(1, 2), link(1, 3)];
        let r = choose_joint_linked(&jc, &ms, &three, TriangulationMethod::default(), &p);
        assert_eq!(r[1].picks, vec![Pick::None; 2], "{r:?}");
        assert_eq!(r[1].refusals, vec![Some(Refusal::Identity); 2]);
        // No partner: it cannot be confirmed.
        let r = choose_joint_linked(&jc, &ms, &[], TriangulationMethod::default(), &p);
        assert_eq!(r[1].picks, vec![Pick::None; 2], "{r:?}");
        assert_eq!(r[1].refusals, vec![Some(Refusal::Identity); 2]);
        // The one partner broken: refused.
        let wrong = RigidLink {
            distance: 2.0,
            ..link(0, 1)
        };
        let r = choose_joint_linked(&jc, &ms, &[wrong], TriangulationMethod::default(), &p);
        assert_eq!(r[1].picks, vec![Pick::None; 2], "{r:?}");
        // `reacquire_links = 0` checks nothing, also with no partner.
        let none = MultiTrackParams {
            reacquire_links: 0,
            ..p
        };
        let r = choose_joint_linked(&jc, &ms, &[], TriangulationMethod::default(), &none);
        assert_eq!(r[1].picks, found, "{r:?}");
        // `reacquire_links = 1` with three partners: the one accepted suffices.
        let one = MultiTrackParams {
            reacquire_links: 1,
            ..p
        };
        let r = choose_joint_linked(&jc, &ms, &three, TriangulationMethod::default(), &one);
        assert_eq!(r[1].picks, found, "{r:?}");
    }

    /// The rank classes of a marker with a near gate ([`CameraOptions::near_gate`],
    /// [`choose_joint_linked`] 1.): two cameras without a far dot before two cameras with
    /// one (also when the far pair costs less, and it is then no rival in the ambiguity
    /// test); a far pair that triangulates before a dot seen by one camera only; a single
    /// near dot before a single far one; a fixed point counts as a camera.
    #[test]
    fn rank_classes_with_a_near_gate() {
        let cams = two_cameras();
        let jc = joint(&cams);
        let proj = |x: Point3| -> Vec<Point2> {
            cams.iter().map(|(m, p)| m.project_point(p, x)).collect()
        };
        let x0 = Point3::new(0.0, 0.0, 0.0);
        let pred = proj(x0);
        let y = |t: f64| proj(Point3::new(0.0, t, 0.0));
        // About 20 px a unit along y in both cameras.
        for c in 0..2 {
            assert!(((y(0.55)[c] - pred[c]).norm() - 11.0).abs() < 0.5);
        }
        let cam = |c: usize, near: f64, cands: Vec<Candidate>| CameraOptions {
            prediction: Some(pred[c]),
            gate: 40.0,
            fixed: None,
            reacquire: false,
            near_gate: near,
            candidates: cands,
        };
        let p = MultiTrackParams::default();
        let choose = |m: MarkerOptions| {
            choose_joint(&jc, &[m], TriangulationMethod::default(), &p)
                .pop()
                .expect("one marker")
        };
        // (0) before (1): a near pair 9 px off with a poor correlation against a far pair
        // 11 px off (beyond the near gate, 10) with a good one, which costs less and lies
        // within `ambiguity_margin` of it. The near pair is taken, not refused as ambiguous.
        let (near, far) = (y(0.45), y(0.55));
        let m = MarkerOptions {
            reprojection_limit: None,
            cameras: (0..2)
                .map(|c| cam(c, 10.0, vec![cand(near[c], 0.3), cand(far[c], 0.9)]))
                .collect(),
        };
        let r = choose(m.clone());
        assert_eq!(r.picks, vec![Pick::Candidate(0); 2], "{r:?}");
        // Without the near gate (cost alone) the far pair would be taken.
        let mut plain = m;
        for o in &mut plain.cameras {
            o.near_gate = 0.0;
            o.gate = 12.0;
        }
        assert_eq!(choose(plain).picks, vec![Pick::Candidate(1); 2]);
        // (1) before (2): camera 0 has a dot on the prediction (a single within
        // `single_gate`) and the far dot, camera 1 the far dot only: the far pair.
        let m = MarkerOptions {
            reprojection_limit: None,
            cameras: vec![
                cam(0, 10.0, vec![cand(pred[0], 0.95), cand(far[0], 0.9)]),
                cam(1, 10.0, vec![cand(far[1], 0.9)]),
            ],
        };
        let r = choose(m);
        assert_eq!(
            r.picks,
            vec![Pick::Candidate(1), Pick::Candidate(0)],
            "{r:?}"
        );
        // (2) before (3): near gate 2 px; camera 0 a single 1 px off with a poor correlation,
        // camera 1 a single 2.5 px off (far) with a good one, on the other side (the two do
        // not triangulate): camera 0's, though camera 1's costs less.
        let (a, b) = (y(0.05), y(-0.125));
        let m = MarkerOptions {
            reprojection_limit: None,
            cameras: vec![
                cam(0, 2.0, vec![cand(a[0], 0.2)]),
                cam(1, 2.0, vec![cand(b[1], 1.0)]),
            ],
        };
        let r = choose(m);
        assert_eq!(r.picks, vec![Pick::Candidate(0), Pick::None], "{r:?}");
        // A fixed point counts as a camera: camera 0 fixed on the prediction, camera 1 a
        // near dot (9 px, poor correlation) and a far one (11 px, good), both on camera 0's
        // viewing ray (along z): the fixed point and the near dot are two cameras, class 0.
        let z = |t: f64| proj(Point3::new(0.0, 0.0, t));
        let (zn, zf) = (z(0.45), z(0.55));
        assert!((zn[0] - pred[0]).norm() < 1e-6 && ((zn[1] - pred[1]).norm() - 9.0).abs() < 0.5);
        let mut fixed = cam(0, 10.0, Vec::new());
        fixed.fixed = Some(pred[0]);
        let m = MarkerOptions {
            reprojection_limit: None,
            cameras: vec![
                fixed,
                cam(1, 10.0, vec![cand(zn[1], 0.3), cand(zf[1], 0.9)]),
            ],
        };
        let r = choose(m);
        assert_eq!(r.picks, vec![Pick::Fixed, Pick::Candidate(0)], "{r:?}");
    }

    /// A pair given more than once: the partner counts once, and is kept only when every
    /// link given for it holds; one broken link makes it broken. The order of the links and
    /// of the two markers in a link do not matter.
    #[test]
    fn a_pair_given_twice_is_one_partner_kept_only_when_both_hold() {
        let cams = two_cameras();
        let jc = joint(&cams);
        let proj = |x: Point3| -> Vec<Point2> {
            cams.iter().map(|(m, p)| m.project_point(p, x)).collect()
        };
        let (pa, pb) = (
            proj(Point3::new(0.0, 0.0, 0.0)),
            proj(Point3::new(0.0, 1.0, 0.0)),
        );
        let marker = |pts: &[Point2], reacquire: bool| MarkerOptions {
            reprojection_limit: None,
            cameras: (0..2)
                .map(|c| CameraOptions {
                    prediction: Some(pts[c]),
                    gate: 5.0,
                    fixed: None,
                    reacquire,
                    near_gate: 0.0,
                    candidates: vec![cand(pts[c], 0.95)],
                })
                .collect(),
        };
        // Marker 0 followed and accepted, marker 1 found again (1 unit from marker 0),
        // marker 2 offers nothing.
        let ms = [
            marker(&pa, false),
            marker(&pb, true),
            MarkerOptions::default(),
        ];
        let link = |a: usize, b: usize, distance: f64| RigidLink {
            a,
            b,
            distance,
            tolerance: 0.01,
        };
        let p = MultiTrackParams::default();
        let found = vec![Pick::Candidate(0), Pick::Candidate(0)];
        let refused = vec![Pick::None; 2];
        let pick = |links: &[RigidLink]| {
            choose_joint_linked(&jc, &ms, links, TriangulationMethod::default(), &p)[1]
                .picks
                .clone()
        };
        // (first, second) link of the pair 0-1 and whether the marker is found: both hold;
        // the second breaks; the first breaks.
        for (first, second, expect) in [
            (1.0, 1.005, &found),
            (1.0, 2.0, &refused),
            (2.0, 1.0, &refused),
        ] {
            for (x, y) in [((0, 1), (1, 0)), ((1, 0), (0, 1)), ((0, 1), (0, 1))] {
                let given = [link(x.0, x.1, first), link(y.0, y.1, second)];
                assert_eq!(&pick(&given), expect, "{given:?}");
                let swapped = [given[1], given[0]];
                assert_eq!(&pick(&swapped), expect, "{swapped:?}");
            }
        }
        // Two partners (0 and 2), two required: partner 0 given twice and holding is still
        // one partner kept, so the marker is refused.
        let r = choose_joint_linked(
            &jc,
            &ms,
            &[link(0, 1, 1.0), link(1, 0, 1.005), link(1, 2, 1.0)],
            TriangulationMethod::default(),
            &p,
        );
        assert_eq!(r[1].picks, refused, "{r:?}");
        assert_eq!(r[1].refusals, vec![Some(Refusal::Identity); 2]);
    }

    /// The point on a camera's ray that keeps a marker's links: the marker at depth 3 on the
    /// ray of camera 0, its old depth 2 units off along the ray; three partners whose distances
    /// it keeps. Found within 0.01; with one partner, with a link no point of the ray keeps,
    /// or with values that are not finite, none.
    #[test]
    fn point_on_a_ray_keeping_links() {
        let cams = two_cameras();
        let x = Point3::new(1.0, 2.0, 3.0);
        let q = cams[0].0.project_point(&cams[0].1, x);
        let ys = [
            Point3::new(2.0, 2.0, 3.5),
            Point3::new(0.0, 3.0, 2.0),
            Point3::new(1.5, 0.5, 4.0),
        ];
        let partners: Vec<(Point3, f64, f64)> =
            ys.iter().map(|y| (*y, (x - y).norm(), 0.02)).collect();
        // The old depth: 2 units further along the ray.
        let ray = (x - Point3::new(0.0, 0.0, -50.0)).normalize();
        let old = x + ray * 2.0;
        let got = point_on_ray_keeping_links(&cams[0].0, &cams[0].1, q, old, &partners).unwrap();
        assert!((got - x).norm() < 0.01, "{got}");
        assert!(
            point_on_ray_keeping_links(&cams[0].0, &cams[0].1, q, old, &partners[..1]).is_none()
        );
        let mut broken = partners.clone();
        broken[2].1 += 5.0;
        assert!(point_on_ray_keeping_links(&cams[0].0, &cams[0].1, q, old, &broken).is_none());
        let nan = [
            (Point3::new(f64::NAN, 0.0, 0.0), 1.0, 0.1),
            partners[0],
            partners[1],
        ];
        let r = point_on_ray_keeping_links(&cams[0].0, &cams[0].1, q, old, &nan).unwrap();
        assert!((r - x).norm() < 0.05, "{r}");
        let huge = [(ys[0], 1e300, 1.0), (ys[1], f64::MAX, 1.0)];
        assert!(point_on_ray_keeping_links(&cams[0].0, &cams[0].1, q, old, &huge).is_none());
        assert!(
            point_on_ray_keeping_links(
                &cams[0].0,
                &cams[0].1,
                Point2::new(f64::NAN, 0.0),
                old,
                &partners
            )
            .is_none()
        );
    }

    #[test]
    fn closest_point_on_a_ray() {
        let cams = two_cameras();
        let x = Point3::new(1.0, 2.0, 3.0);
        let q = cams[0].0.project_point(&cams[0].1, x);
        // The ray of q passes through x: the closest point to a point beside the ray at the
        // same depth is x itself.
        let off = Point3::new(1.0, 2.0, 3.0) + crate::Vector3::new(0.3, -0.2, 0.0);
        let p = closest_point_on_ray(&cams[0].0, &cams[0].1, q, off).unwrap();
        let back = cams[0].0.project_point(&cams[0].1, p);
        assert!((back - q).norm() < 1e-9);
        assert!((p - x).norm() < 0.05, "{p}");
        assert!(
            closest_point_on_ray(&cams[0].0, &cams[0].1, Point2::new(f64::NAN, 0.0), x).is_none()
        );
    }

    #[test]
    fn hostile_inputs_do_not_panic() {
        let cams = two_cameras();
        let jc = joint(&cams);
        let bad = CameraOptions {
            prediction: Some(Point2::new(f64::NAN, 1.0)),
            gate: f64::INFINITY,
            fixed: Some(Point2::new(f64::INFINITY, 0.0)),
            reacquire: false,
            near_gate: 0.0,
            candidates: vec![cand(Point2::new(f64::NAN, f64::NAN), f64::NAN); 40],
        };
        let m = MarkerOptions {
            reprojection_limit: None,
            cameras: vec![bad.clone(), bad.clone(), bad],
        };
        let p = MultiTrackParams {
            max_candidates: usize::MAX,
            max_combinations: 0,
            gate: f64::NAN,
            ..MultiTrackParams::default()
        };
        let _ = choose_joint(
            &jc,
            &[m.clone(), MarkerOptions::default()],
            TriangulationMethod::default(),
            &p,
        );
        let _ = choose_joint(
            &[],
            std::slice::from_ref(&m),
            TriangulationMethod::default(),
            &p,
        );
        let _ = p.gate_for(3, true);
        // The overlaps: out of range, a marker with itself, repeated; every overlap parameter
        // at odd values.
        let over: Vec<Overlap> = [
            (0, 1, 0),
            (1, 0, 9),
            (0, 0, 0),
            (7, 1, 1),
            (usize::MAX, 0, 0),
        ]
        .iter()
        .flat_map(|&(a, b, camera)| {
            [true, false].map(|touching| Overlap {
                a,
                b,
                camera,
                touching,
            })
        })
        .collect();
        for v in [f64::NAN, f64::INFINITY, -1.0, 0.0, 1e300] {
            let q = MultiTrackParams {
                overlap_decision: v,
                claim_radius: v,
                claim_ratio: v,
                share_radius: v,
                ..p
            };
            let ms = [m.clone(), m.clone(), MarkerOptions::default()];
            let _ =
                choose_joint_overlapping(&jc, &ms, &[], &over, TriangulationMethod::default(), &q);
            let _ =
                choose_joint_overlapping(&[], &ms, &[], &over, TriangulationMethod::default(), &q);
        }
    }
}
