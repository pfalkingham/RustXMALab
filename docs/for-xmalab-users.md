# Coming from XMALab 3

What a user of XMALab 3 needs to know before using this program on real data. The full record, with XMALab's source lines and the evidence for each item, is [the parity ledger](parity-ledger.md); the row numbers below refer to it.

## Your projects

- **It opens XMALab 3 projects** (`.xma`) of every file version XMALab 3 itself opens, and recomputes them to the same numbers (checked table by table against XMALab on the reference projects).
- **XMALab 3 opens projects saved here**, with no warnings. Calibration and undistortion-grid images are stored as XMALab stores them.
- **One thing a saved project carries that XMALab's does not:** the settings that change results (triangulation method, filter options, thresholds and so on). In XMALab 3 these are per machine, so the same project can give different numbers on two machines. Here a project keeps the settings it was last used with; XMALab 3 ignores that entry (A10).
- **A value that is not a number** (NaN or infinite) is never saved: XMALab writes it as `nan` or `inf` and then cannot read its own file. Such a value is stored as "not set", and a project holding one opens with a warning naming the file and line (ledger F49 to F52); an import with one is refused.
- **Videos are referenced by path, as in XMALab.** When they are not where the project says, the program searches the project's folder as XMALab does, then asks.

## Where results differ from XMALab 3, on purpose

On unfiltered data from an existing, tracked project, the numbers are XMALab's. The differences are corrections of faults; each changes results only in the situation it names.

| What | When it matters | Size of the difference | Rows |
|---|---|---|---|
| **Filtered data** (3D points, rigid-body transformations) | Always, for the same cutoff number | The cutoff you enter is now the real cutoff of the two-pass filter; XMALab's is about 10% lower than the number entered. On the reference projects filtered 3D points move by up to 0.01 cm at 25 Hz and 0.04 cm at 10 Hz | A7 |
| **Marker-to-marker standard deviation** (precision report, project overview) | Always | 0.3% to 0.6% higher | A58 |
| **Images deeper than 8 bits** (16-bit TIFF or PNG, `.cine`) | Detection, tracking and refinement on such sources | Not yet measured on real data; XMALab reduces them to 8 bits first | V1, V15, V18 |
| **Tracking several markers at once** | "track selected" and "track all" | XMALab re-detects, and so moves, points tracked earlier, including those of unselected markers; here a step touches only the points it tracked. On the reference data mostly under a pixel | A62 and the rows marked "Fix, compat" from A60 |
| **Projects with a lens distortion model or virtual points** | Rigid-body poses and errors | Can be large: several XMALab results there were wrong | A11, A19, and the rigid-body rows of the ledger |
| **Undistortion grids** turned about 30 degrees or with a bead missing beside the centre bead | "compute undistortion" on such a grid | XMALab's orientation can be 5 to 15 degrees off, bending the undistortion by pixels; ordinary grids are matched identically | A104 |
| **"remove outlier at border"** | After toggling outliers by hand | Hand-toggled outliers stay off; a few tenths of a pixel in the maps | A107 |
| **Calibration** | "Fix principal at image center"; a calibration whose last step fails; after "optimize camera setup" or an external calibration | The principal point is the true centre; a failed calibration is reported as failed; trials are recomputed after the calibration changes | A111, A112, A126, A147 |
| **MayaCam 2.0 import** | A file with a lens model | All eight coefficients are read; XMALab reads the first only | F1 |
| **Undistort sequence** | Videos read through FFmpeg (H.264 and similar); a flipped camera; sources deeper than 8 bits | XMALab's seek can return the wrong frame for such videos (on the reference videos frame 4 for frame 1, nothing for the last two), so it writes the wrong frames; here each frame is the one asked for. A flipped camera's frames are mirrored before the grid is applied. Deeper sources are written at their own depth | V20 to V23 |
| **Checkerboard calibration** | Manual corners with an inverted axis; four clicks that do not span the board; a re-detection that fails on a frame that already has corners; "recompute" on a camera with one frame; a frame reset and then recalibrated | Ordinary boards calibrate as in XMALab, to about 4 parts in a million (not to the last digit: a flat board amplifies a rounding difference between linear-algebra libraries). In the named cases XMALab stores wrong or stale corners or fails; here the case is refused with a message or calibrated correctly | A180 to A188, A202 to A204 |
| **Checkerboard with two or more cameras** | The cameras number the board's corners differently: a different corner clicked, a board turned in one view, a camera that sees the board from behind (an X-ray camera on the far side) | The numbers are XMALab's: such cameras are joined wrongly (in the reference runs "optimize camera setup" goes from 0.05 to 47 pixels of error, and trial 3D points are wrong by 11 to 605 mm). The difference is a **warning** that names the camera and the frames numbered differently, after the calibration and at "optimize camera setup", so you can detect the board again in those frames with the same physical corner clicked. Nothing is renumbered for you | A189 |
| **Merge-aware tracking** ("Enable merge-aware tracking" on the Marker tracking page) | Always | **It is not in this program** (removed on 26 November 2026). "Enable multi-camera tracking", in the same place, replaces it (see "New since XMALab 3"). Measured before it was removed, against points finished by hand (found, then points more than 5 px from the hand-finished point): merge-aware 33 % and 2,518 on the Rat recording, 76 % and 303 on the guineafowl, 97 % and none on the pig; multi-camera 81 % and 347, 83 % and none, 99.8 % and none. At the pig's crossing of two markers, the case merge-aware tracking was made for, multi-camera tracking has 38 of the 40 points within 2 px going forward (merge-aware 30) and 31 going backward (merge-aware 2). What you lose: merge-aware tracking put a point in nearly every cell of the guineafowl, wrong ones among them, where multi-camera tracking leaves a doubtful cell empty. If you need it, track in XMALab 3; the project opens here | A226, U171 |
| **Checkerboard projects made by XMALab with an inverted axis** | Opening one with another "invert axis" setting than it was made with | XMALab mirrors the board (hundreds of pixels of error); here the axis is recognised from the stored calibration, with a warning. Projects saved here store the axis | F73, F75 |
| **Denoise Images** | Always | The filter strength is used as entered (XMALab drops the decimals); every frame is denoised from the original frames (XMALab denoises from frames it has already denoised); the last frames are written (XMALab silently loses them and shortens the trial on relinking) | A190 to A192, F66 |
| **All Trials for External** | Always | Each camera's folder holds that camera's own unfiltered images. XMALab puts every camera's images in every folder and filters all but the first camera's | F65 |
| **Undistorted Trial images** | The AVI format; cameras whose videos share a file name; sources deeper than 8 bits | An AVI is written (XMALab writes nothing on Linux, with no message); such cameras get a folder each (XMALab lets the second overwrite the first); deeper sources are written at their own depth as TIFF | V26 to V28, F71 |
| **XMA Portal import** | Metadata added a second time; camera numbers when a Portal project lists trial files too; a Mac-made archive | Metadata is replaced, not doubled; cameras are numbered from the calibration files; the `__MACOSX` folder is skipped | F59, F60, F63, F64 |
| **Rigid bodies with very many markers seen by one camera only** | A frame where the body has one or two markers seen by two cameras and about ten or more seen by one (more than a million possible combinations of their positions) | XMALab tries every combination and can hang or run out of memory. Here those markers are not used for the pose at that frame, and the Console (in `xma`, a `warning:` line) names the body and the frames | A210 |
| **Mean marker size after Change Trial Data** | A trial changed to a shorter video, then tracked without reopening the project | The mean size, which sets the tracking template's radius, is that of the frames kept, at once. XMALab keeps the old video's value until the project is reopened | F37 |
| **Deleting or importing many points at once** | "Delete above error", Delete Data, swap, interpolate, Import 2D Points | Much faster on long trials, with the same results; one exception: a marker whose detected sizes are all deleted keeps the mean size it had before | A216 |
| **AVI recordings that were cut short** | A recording stopped by a crash or power cut, whose header was never completed | It opens with every frame that was written. XMALab shows such a file with 0 frames, or only the frames its unfinished header counts | V37 |
| **Blob detector settings that cannot work** | A threshold step far too small or a maximum threshold that is enormous (only possible by editing the settings stored with a project) | XMALab freezes. Here marker detection finds nothing, and "compute undistortion" and cube calibration say in a sentence which settings to change. Maximum thresholds above 255 work as before | A217 |

If you need XMALab 3's numbers exactly, for example to continue a study already analysed with it, the command-line program has a compatibility mode: `xma recompute`, `xma export`, `xma edit`, `xma track`, `xma calibrate`, `xma undistort`, `xma denoise` and `xma import-portal` take `--xmalab3-compat`. The window has no such switch yet.

**When you report results**, say which program and version produced them, and for filtered data that the cutoff is the effective one.

## Where the interface differs

The layout, the three workspaces, the panels, the menus and the shortcuts are XMALab 3's. The deliberate differences:

- **Undo and redo for everything** (Edit menu, Ctrl+Z, Ctrl+Shift+Z). XMALab 3 can undo only a moved point.
- **Saving**: the title shows when there are unsaved changes; closing or opening asks only then; a recovery copy is written while you work and offered after a crash. A new dataset is unsaved until you save it. Closing, opening or quitting while a tracking run or a refinement is going ends it first and then asks about its points (U53, U55).
- **Long computations do not freeze the window**: tracking, refinement, undistortion, calibration and optimisation show in the Progress panel; tracking and refinement can be stopped.
- **New dataset and New trial do not close your open dataset until the new one exists**, and Cancel leaves it open.
- **Undistort sequence refuses an item whose outputs would replace its own inputs** (an output folder that holds an earlier run's outputs), which XMALab overwrites before reading them (V25).
- **Errors that XMALab passes over are shown**: a file that cannot be written, a video whose length differs between cameras, a calibration image of the wrong size, a camera that is not calibrated when you optimise.
- **Several dialogs no longer write on opening or on Cancel** (the rigid-body dialog, the colour chooser, the external-calibration frame).
- **The look is the toolkit's own, not XMALab's** (U139). Colours, fonts, spacing and standard controls are those of egui, the program's interface toolkit, in its light and dark themes; View › Theme › Follow System follows your operating system's light or dark setting as it changes. What carries meaning is kept: the marker status colours (and the Status Colours dialog), the plots' curve colours, the red, green and blue of axes, the overlay colours in the camera views, and the icons. The active camera view is marked by a grey background of the current theme.
- **Colour choosers are a colour picker**: a click on a colour swatch (rigid body, event, status colours) opens a picker with a colour field, a hue bar, R, G and B values and the `#rrggbb` field; OK applies, Cancel changes nothing.
- **Exports and "Denoise Images" run as jobs** with progress, can be stopped, and report what could not be written. "All Trials for External" appears in the File menu as soon as its setting is switched on (XMALab needs a restart).
- **Names that cannot be file names are made safe on export**: a trial, camera or video name with `/`, `\`, `:` and the like has those characters replaced by `_` in the names of exported folders and files; a name of `.` or `..` is refused.
- **A Portal import that fails or is stopped removes what it had written**, and never deletes the downloaded file unless you answered Yes to "delete after import" and the import succeeded. An archive whose entries would land outside the destination folder is refused.
- **Dropping a marker on the rigid body it already belongs to does nothing** (U79). XMALab removes and re-adds it, which resets that body's references.
- **"optimize camera setup" warns when a camera matrix has skew** (possible after an external calibration): the optimisation treats such a camera as having none, as XMALab does without saying so (A129).
- **Stacked panels have their tabs at the top.**
- **View › Reset layout** (new, last in the View menu) puts all panels and the main window back to the first-run arrangement at once. Use it when a panel has opened where you cannot see it, for example after a second monitor was unplugged (U165). Saved positions are not moved automatically.
- **View › Output is in the menu** (U174). It shows and hides the Console, the panel where the program writes its errors, warnings, what an export wrote and the summary of a multi-camera tracking run. XMALab 3 has this entry hidden, so its Console cannot be opened from the window. The Console still starts hidden; the entry works in every workspace and with no project open, and the panel opens as a window of its own that can be docked at the bottom.
- **Drop-downs follow the scroll wheel, as in XMALab** (U175): with the pointer over a drop-down, one notch moves one entry. One difference on purpose: a drop-down inside a page or list that is itself scrolling leaves the wheel to the page until you have clicked the drop-down, so scrolling a long Settings page does not change a setting by accident.
- **The Left and Right arrows and text boxes** (U177). In XMALab the arrows step frames wherever the keyboard is, also inside a number box. Here a box you are typing in keeps the arrows (and the other plain keys), so you can move in its text; press Enter, or press a mouse button anywhere else in the window (a click, or a drag such as a pan), when you are done and they step frames again. After a click in the plot all the keys work as usual, and Delete and Backspace act on the plot's selected range until you press the mouse somewhere else. In a floating panel's own window only these two arrows work; for the other keys click the main window first.
- **A camera hidden with the camera selector is not loaded while it is hidden**, so stepping and playback with hidden cameras are faster. In the 3D world view its image plane is empty, where XMALab shows the last image it had loaded (U166).
- **A project file that declares impossible sizes is refused with a sentence** (a trial of more than 10 million frames, or marker, rigid-body or event data too large to hold), where XMALab runs out of memory and closes (F84). No real project is near these limits.
- **Settings › Calibration: the tooltips of the "Use Detection Method when Alt key is held" items describe the item they are on** (U37); in XMALab five of the six show another method's text.
- **About, Welcome and Help** (U172). The About dialog says what this program is (based on the original XMALab written by Benjamin Knörlein, re-written from scratch in Rust by Peter Falkingham), names Peter Falkingham as its author and shows the build date. The user manual, the issue tracker and "Check for newer version" lead to this program's own repository (`https://github.com/pfalkingham/RustXMALab`), not to XMALab's: report this program's faults there, not to XMALab's authors. The links to XMALab's Google group and to the tutorial data on the XMAPortal Sandbox (which no longer exists) are gone. The request to cite XMALab's validation paper stays, with the paper linked by its DOI. Clicking a link opens it in your browser or mail program.
- **The program is called RustXMALab** (U173). The window's title, the Welcome dialog, the licence button ("RustXMALab license"), the question asked on quitting and two labels in Settings say RustXMALab. The program file is `rustxmalab` (the command-line program is still `xma`), so it can be installed beside XMALab. Its icon is a blue-green tile with "RX" and a rust-coloured rim, and its banner is its own, where XMALab's are a red tile and a black plate. The Linux archive holds a menu entry you can add by hand (its `INSTALL.txt` says how). Nothing saved carries the name: project files, exports and your settings (in the `RustXMALab` folder, as before) are unchanged. Where a text speaks of XMALab it means the original program, its version 3 or its file format.

## New since XMALab 3

- **Tracking with all cameras together** (21 November 2026; ledger A218). XMALab's tracking follows a marker in each camera on its own and checks the cameras against each other only afterwards, with a check that cannot see a point sliding along the other camera's line of sight; a marker that fails is dropped for the rest of the run. The new method follows each marker in 3D, takes several candidate dots per camera, and accepts the combination that fits every camera and the marker's motion; it leaves a doubtful point empty and looks again for a marker it lost. On the Rat example, started from one frame, it finds 50 to 78 % of the points finished by hand where XMALab's kinds find 16 to 59 %, with 4 to 73 points more than 5 px off where they have 29 to 484 (nine runs, each compared with the better of XMALab's kinds: more points and fewer wrong ones every time). In the window, tick "Enable multi-camera tracking" on the Marker tracking page of the Toolbox (where XMALab 3 has "Enable merge-aware tracking", which this program no longer has: ledger A226): it shows the groups "multi-camera selected" and "multi-camera all", with the same four buttons as the other groups (ledger U167); the headless program has it as `xma track --kind multi`. It needs at least two calibrated cameras that are shown: with one camera of two hidden its buttons are disabled (and `xma track --cameras` refuses it), with a sentence saying why. The window uses the method's default settings; at the end of a run the Console says how many points it wrote, how many cells of the markers it was following it left empty, and which markers it lost. XMALab's kinds are unchanged. Since 22 November 2026 (ledger A219) it also starts on a marker that moves fast: with one point set, a marker is looked for up to 40 px away, and a dot that far is taken only when the next frame shows it carrying on at that speed and the frame before shows no other dot that it could have been. A marker on a bone with only two markers is found again after it was lost (one partner at the right distance is enough for it), and a marker that one camera loses for a few frames is no longer lost in the other camera too. On the guineafowl example (two markers on one bone, the foot moving 20 to 33 px a frame as it comes in and leaves) it now finds 75 % of the points finished by hand with none more than 5 px off, where it found 40 % before; on the Rat and the pig its results are as they were. **For a marker that moves fast, set its point at two frames before tracking**: the run then knows its speed and does not need the wide first step, which can refuse a real marker (when another dot lay nearer to where it lands) and, rarely, start on a neighbour when the marker itself is hidden at that frame. **A start point need not be at the marker's centre.** Markers are not always round (the guineafowl's are conical metal points), and a point set without detection (Ctrl + click, the Detailed View), a click whose detection found nothing or stopped partway along a long marker, or a point imported from elsewhere can lie anywhere in the marker's dark shape. This kind of tracking now works from the marker's detected centre for such a point: with the guineafowl's start points moved 3 to 8 px inside their shapes it finds 63 to 80 % of the points finished by hand, where it found 14 to 56 % before. Your own point is not changed: it stays where you set it, and the points tracked from it are at the detected centre. The centre is not used when it is more than 10 px away, when the blob is more than twice the marker's size, when another marker's point lies on the same dark shape, or when the two cameras would then disagree by more than 1.5 px (and more than they did); the run then starts from your point as before. One case to know: a marker that touches a dark thing that is not one of the markers being tracked, at the frame you start from, can be followed about 3 px off its centre while the two stay merged; start from a frame where the marker stands clear. **Two markers that meet in one camera** (23 November 2026; ledger A220). This was where the method mixed markers up: after two dots had merged in one camera, one marker could come out on the other's dot and stay there, looking tracked. It now knows when two of the markers it is tracking touch in a camera. Two markers may sit on one dot only while they really touch; the marker's look is not learned from a merged shape; when the two part, which is which is decided for both together, and that camera is left empty for both when it cannot be told; and a marker being looked for again may not take a dot that lies where another marker is expected. On the Rat example the points more than 5 px from the hand-finished ones fall from 290 to 158 a run (nine-run means) with as many points found as before (68.9 %); two markers on one dot, and two markers exchanged, no longer occur; the guineafowl and the pig are as they were. What to expect: through an overlap this kind leaves cells empty rather than guess (of the Rat's 14,284 cells where two markers are within 8 px of each other in a camera it leaves 8,351 empty, 380 more than before), and it does not write a point "held" through an overlap as XMALab 3's merge-aware tracking does. At the pig's crossing (markers 5 and 9 in camera 1, frames 34 to 53) it was right on both sides and left most of the crossing empty when it was first built; with the options that are on by default since 26 November 2026 (further down) it has 38 of the crossing's 40 points within 2 px going forward and 31 going backward, and none wrong. Each rule can be switched off in the headless program (`xma track --help` lists them: `share_needs_overlap`, `overlap_clean`, `overlap_decision`, `reacquire_exclusive`, `claim_radius`, `ray_links`); the window uses the defaults. Since 24 November 2026 (ledger A221, A222, U168) two things about this kind have changed, both after the author tried it in the window on the example pig project. **A run no longer erases a point it could not find again.** Before, for every marker it was following, a run emptied each cell ahead that held a tracked point and that it did not find itself; and since this kind carries on past the frame where it loses a marker, tracking backward over a forward run's result, or over a project that already held points, removed good points (on the Rat 3,320 points within 2 px of the hand-finished ones, leaving fewer points after the second pass than after the first). Now such a point stays exactly as it was, and is removed only when the run's own points at that frame contradict it: the run found the marker in the other camera and the two do not agree (a reprojection error above 3 px), or the run put another marker on that dot and does not hold the two as touching. So tracking forward and then backward adds the two results together: on the Rat 70.3 % of the hand-finished points after both passes against 67.2 % after the forward pass, on the guineafowl 95.6 % against 76.4 %. The Console line says how many earlier points were kept and how many removed, and still names a marker the run lost even when its earlier points remain. What to know: a point the run kept is not a point the run confirmed; it is whatever was there before, another kind's mistakes included (of 273 points of XMALab's kind that a run kept on the Rat, 9 are more than 5 px off), so look at the reprojection error as after any tracking. The second exception sometimes removes a right point where two markers truly lie on one dot in a camera (32 of 3,359 on the Rat, 29 of them right), and in return removes points that XMALab's kind had slid onto another marker's dot (20 of the 23 it removed there were wrong); `xma track --multi keep_dot_rule=0` switches that exception off, and `keep_unconfirmed=0` gives the old erasing. Tracking one frame at a time, or stopping a run and starting it again, starts from whatever the frame on screen holds, a kept point included; one continuous run never reads a kept point. So where the earlier tracking of a marker is wrong, do not step through it frame by frame: delete those points first, or track in one run. **The test of whether a dot is dark enough to be a marker is now measured against the brightness round the dot** (the switch `contrast_relative` of 22 November, which was off; `--multi contrast_relative=0` gives the old test). The old test compared grey levels with the marker's last ten points, and lost a marker when the picture round it brightened and darkened again, as it does round the pig's marker 5 in camera 1 at frames 7 to 15 and 70 to 81: a forward run had almost no point of it from frame 16 to 59, a backward run none at frames 1 to 6. A bead takes away a share of the light, so the share stays level where the grey levels do not. With the new test a forward run on the example pig project leaves marker 5 empty at five frames inside the crossing of markers 5 and 9 (37, 40, 41, 47, 49), and after the backward pass at one (49). Over the three examples (nine-run means): guineafowl 82.5 % found against 74.8 %, pig 99.7 % against 99.2 %, none more than 5 px off either way; Rat 68.2 % against 68.9 % with 174 points more than 5 px off against 158, and less steady from run to run (145 to 248 against 130 to 179). **Inside a crossing the points are now mostly there, and not centred**; this replaces what is said above about the pig's crossing being left mostly empty. The two markers are 4 to 5 px apart there and make one dark shape, and about half of the points written in it lie 2 to 3 px from where the skull's other markers put the marker (none more than 5 px off). Look those frames over by hand. **An option for two dots that touch: "Separate touching dots"** (25 November 2026; ledger A223, U169). Under "Enable multi-camera tracking" there is now a second check box (unticked at start when it was built; ticked each time the program starts since 26 November 2026). Ticked, a multi-camera run looks, wherever the detection has joined a marker's dot and another dark dot into one shape, for the two dots in it: it fits them as two round dots and puts the point on the marker's own dot, where it otherwise takes the centre of the joined shape. The other dot need not be a marker: in the example pig project the bead that marker 5 passes in camera 1 at frames 34 to 53 is not one. Where the detection already stands on one dot, nothing changes. On that crossing, 19 of marker 5's 20 points are then within 2 px of where the skull's other markers put it, after a forward run and after a backward run over it (one frame, 43, stays empty), where without the option 7 are after the forward run, 8 lie 2 to 5 px off and 5 are empty. Over the three examples (nine runs each): on the Rat it finds 70.4 % of the hand-finished points with 189 a run more than 5 px off, against 68.2 % with 174; the guineafowl (82.6 % against 82.5 %) and the rest of the pig are as they were. **It was off by default when built, because it is not finished; it is on by default since 26 November 2026 (the author's decision after trying it; further down).** An independent review found two faults that are still there. On the Rat a marker (16) is lost for the rest of a forward run after it crosses another marker in one camera, in most runs from frame 1; tracking forward and then backward from frame 1 therefore ends with fewer points than without the option in six of nine runs (68.4 % of the hand-finished points on average against 69.8 %). And on the guineafowl a marker that lies on a dark band has about 30 right points a run moved 2.5 to 2.7 px along the band. Also: **use one continuous run**; tracking one frame at a time through a crossing gets no benefit from the option (each step starts afresh from the point before). Dots closer than about 3 px are not told apart, only two dots are separated, and a run that starts while the marker already touches something does not separate it: start from a frame where the marker stands clear. It is slower (the Rat's 449 frames in 31 s against 7 s). So: tick it for round bead markers that pass close to other beads, as on the pig, and look at the result as after any tracking. The Console line at the end of the run says how many points were placed this way; their size is written as -1, as for any point at which no blob was measured. In the headless program: `xma track --kind multi --multi split_touching=1`. **Two options that find a lost marker again** (25 November 2026; ledger A224, A225, U170). A count on the three examples showed where the method loses most: on the Rat two thirds of the hand-finished points it misses belong to markers that are lost and never found again. A lost marker was looked for only where its nearest neighbours' rigid movement put it, and accepted only where its distances to them matched one earlier frame within 2 %; the Rat's markers on different bones or on skin do not keep their distances that well. Under "Separate touching dots" there are now two more check boxes (unticked at start when they were built; ticked each time the program starts since 26 November 2026). **"Recognise a lost marker by its distances"** accepts a lost marker where its distances to the other markers fit what the run itself has measured for each pair so far (the average, and how much it varied). **"Place a lost marker by its distances"** looks for a lost marker whose neighbours do not move as one rigid body where its average distances to four or more followed markers put it; without the option it is not looked for at all. On the Rat (nine runs of twelve cases) the method finds 68.2 % of the hand-finished points with 174 a run more than 5 px off as it is; 75.0 % with 325 with the first box; 78.3 % with 483 with the second; 80.2 % with 530 with both; and 81.3 % with 335 with both and "Separate touching dots". Tracking forward and then backward from frame 1 ends with 86.4 % of the hand-finished points and 96 more than 5 px off with both (86.6 % and 62 with "Separate touching dots" as well), where it ends with 70.3 % and 19 as it is. The guineafowl (a bone with two markers: four partners are needed) and the pig are unchanged. **They were off by default when built, because it is a trade and not finished; they are on by default since 26 November 2026 (further down).** The markers are found again on the right dot (of 136 found again by placing, 130 within 2 px of the hand-finished point), but most of the added wrong points come soon after: the marker is found again where its dot touches another marker's dot in one camera, and within a few frames it slides onto that dot, which is free because the other marker is lost. The program has the same weakness without the options, less often. So after a run with them, look at a marker's points where it reappears beside another marker. An independent review found no defect in the code and repeated the figures; it notes that the distance test is loose for markers on different bones (by its estimate about ±16 % of the distance on the Rat). In the headless program: `xma track --kind multi --multi stat_links=1,distance_placement=1`. **A check box for an older rule** (U170): "Remove old points on another marker's dot", ticked at start, is rule (b) of what a multi-camera run does to points already in the project (ledger A221); unticked, such points are kept. Nothing changes unless it is unticked. **On by default since 26 November 2026** (ledger A223 to A225, U171; the author's decision after trying the options in the window on the Rat recording): "Separate touching dots", "Recognise a lost marker by its distances" and "Place a lost marker by its distances" start ticked, as "Remove old points on another marker's dot" always did, and `xma track --kind multi` uses them without being told. What that gives on the three recordings (one run of the 24 cases; found, then points more than 5 px off): the Rat 81.3 % and 347, where the settings before gave 68.3 % and 159; the guineafowl 82.6 % and none (before: 82.5 %); the pig 99.8 % and none (99.7 %). So more of a recording like the Rat is tracked for you, and there are more wrong points to look for: they come where a marker is found again beside another marker's dot. A run is slower: about five times on the Rat (one 449-frame run about 31 s against 6 s), 1.4 to 1.5 times on the other two. Untick the boxes (in `xma`: `--multi split_touching=0,stat_links=0,distance_placement=0`) for the method as it was. The boxes are not saved with the project or the settings: they are ticked again each time the program starts.
- **Export to Blender** (26 November 2026; ledger F85, U176). **File › Export › Blender scene** writes a folder that Blender turns into a scene: the calibrated cameras, each with its undistorted calibration image or the undistorted images of a trial on a plane in front of it, and, for a trial, optionally the rigid bodies as animated objects, their bone meshes and the 3D points. If Blender is installed the export can run it for you and leave a finished `.blend` in the folder; otherwise run the folder's `build_scene.py` in Blender. Looking through a Blender camera gives the X-ray image (checked in Blender 5.3: points land within 0.001 px of where this program puts them). A trial's images stay in the folder beside the `.blend`, because Blender cannot pack an image sequence; still images and meshes are inside it. The cameras are posed as XROMM_BlenderTools poses them from a MayaCam file. In `xma`: `xma export … blender-scene`. Only Blender 5.3 has been tried.

## What is not there yet

- **Exports:** XROMM VR, DLT, MayaCam version 1, and "XRayProject Matlab Format". Every other export is there.
- **Windows and macOS builds** are made by the repository's automatic build (Windows on x86_64, macOS on Apple Silicon) and are not covered by the project's own checks, which run on Linux: treat them as untried. Windows has a setup program and macOS a disk image with the app, neither yet signed, so each system warns once (the README says how to get past it). Linux has an AppImage, a `.deb` and an archive, built on Ubuntu 22.04: they need glibc 2.35 or later.

## What has not been checked

Everything above was checked on one machine (Linux), with XMALab 3.0.0-beta.5 as the reference. Tracking, editing, undistortion and cube calibration were checked against one real dataset family (the pig example). **Checkerboard calibration and XMA Portal import have been checked only with made-up inputs**: rendered checkerboard images, and Portal archives built to match what XMALab's loader reads, because no real checkerboard set and no real Portal download were available. A real Portal file may differ from what the loader's code suggests (the root element, the encoding, the folder names). Try both on a copy of your data first.

Not yet covered by real data either: more than two cameras, several calibration images per camera, calibration videos, images deeper than 8 bits, colour video, and marker overlaps other than the one in the pig trial. The corrections of faults have synthetic tests and independent reviews, not XMALab references, because XMALab has the faults.

**The check of the board's numbering between cameras (A189) and the hand-set-point rule in overlaps (A84) have no XMALab reference**, because XMALab has the faults they address. They were checked on made-up rigs and on the one reference trial with an overlap. The numbering check only warns: it can miss a differently numbered camera when the cameras share few frames or the board is tilted in one direction only (a Console line at "optimize camera setup" says when it cannot tell), and with three or more cameras it can name the wrong frames. Clicking the same physical corner of the board in every camera avoids the problem. An automatic renumbering was built and then removed: three versions each failed an independent check, and the author chose to keep the warning (17 November 2026).

## In detail

The sections below give the larger differences in full: what XMALab does, what this program does, and what it means for your results.

### Filtered data differs from XMALab

> **Important: for the same cutoff frequency, filtered results from this
> program are not the same as XMALab's.** This is deliberate. Unfiltered data
> is not affected.

**What XMALab does.** XMALab smooths 3D points and rigid-body transformations
with a 4th-order Butterworth low-pass filter, run forwards and then backwards
so that the result has no time lag. Each pass is designed at the cutoff
frequency you enter. Because the data goes through the filter twice, the two
passes together let through only **half** the amplitude of a signal at that
frequency. The frequency at which they let through 71% (-3 dB, the usual
meaning of "cutoff") is about 10% lower than the number entered: 27.1 Hz for
an entered 30 Hz at 250 frames per second. The common MATLAB combination of
`butter` and `filtfilt` behaves in the same way.

**What this program does.** It designs each pass slightly higher, so that
**the cutoff you enter is the -3 dB cutoff of the filter as it is actually
applied**. For a Butterworth filter of order 4 applied twice, each pass is
designed at

    f_design = (fs / pi) * atan( tan(pi * f / fs) / 0.89568 )

where `f` is the entered cutoff, `fs` the recording speed in frames per
second, and 0.89568 = (sqrt(2) - 1)^(1/8).

**What this means for you.**

- For the same entered cutoff, this program smooths a little less than
  XMALab. Entering 30 Hz here gives the filter XMALab gives for 33.1 Hz (at
  250 frames per second).
- Filtered 3D points, filtered rigid-body transformations, the errors
  computed from them, and every export or plot of filtered data differ from
  XMALab's for the same project. Measured on the pig test data (250 frames
  per second, lengths in cm), as the largest difference at any frame between
  this program and XMALab's filter:

  | Entered cutoff | Filtered 3D points | Filtered body translations | Filtered body rotations |
  |---|---|---|---|
  | 25 Hz | 0.010 cm | 0.018 cm | 0.0009 rad |
  | 10 Hz (one body at 40 Hz) | 0.043 cm | 0.034 cm | 0.0014 rad |

  These cutoffs were chosen to test the arithmetic, not as recommendations.
  The size of the difference depends on the cutoff, the motion and the noise
  in the data.
- A cutoff quoted in work done with XMALab refers to XMALab's filter. To
  reproduce it here, enter the equivalent cutoff

      f_here = (fs / pi) * atan( 0.89568 * tan(pi * f_xmalab / fs) )

  or run `xma` with `--xmalab3-compat`, which gives XMALab's filter exactly.
  At 250 frames per second:

  | Cutoff entered in XMALab (Hz) | 5 | 10 | 20 | 25 | 30 | 50 |
  |---|---|---|---|---|---|---|
  | Same filter here (Hz) | 4.48 | 8.97 | 17.99 | 22.54 | 27.12 | 45.91 |

- When you report a cutoff from this program, say that it is the -3 dB cutoff
  of the zero-lag (two-pass) filter.
- As in XMALab, a run of 12 consecutive frames or fewer is too short to
  filter; its unfiltered values appear in the filtered output.

This change (item A7 in [the ledger](parity-ledger.md)) was
decided on 14 November 2026 and may be revisited after users have been
consulted.


### Deeper images are processed at their own depth

> **Important: on images deeper than 8 bits, marker detection and tracking
> results from this program are not the same as XMALab's.** This is
> deliberate. Ordinary 8-bit images (most videos, 8-bit image sequences) are
> processed exactly as XMALab processes them, and give the same results.

**What XMALab does.** XMALab reduces every image to 8 bits (256 grey levels)
before it looks for markers. A 16-bit PNG or TIFF is divided by 256, whatever
depth its data really has, so 12-bit data stored in a 16-bit file becomes
grey levels 0 to 15. A 10, 12 or 14-bit `.cine` is reduced to its top 8 bits.

**What this program does.** It detects, tracks and refines markers on the
image at its own depth: 10, 12, 14 or 16 bits. The grey-level settings
(threshold offsets, the blob detector's thresholds) keep their meaning: they
are still given on the 0 to 255 scale. The viewer shows these images at their
full depth too. In addition:

- a packed 10-bit `.cine` is read as 10-bit samples (12-bit through its lookup
  table) instead of XMALab's 8-bit values;
- a 16-bit PNG or TIFF that declares fewer significant bits (PNG `sBIT`, TIFF
  `MaxSampleValue`) is read at that depth. A file that declares nothing is
  read as 16-bit; the program never guesses the depth from the data.

**What this means for you.** Depth helps in some cases and not in others.
The figures are from synthetic 12-bit images (600 trials per case, except the
tracked trial); "reduced" is the same image cut to 8 bits as XMALab cuts it.

- **It helps with faint markers and with noise.** Markers only 0.4 of an
  8-bit grey level darker than the background vanish when the image is cut to
  8 bits; at full depth every point of a synthetic trial was tracked to within
  1 pixel (worst 0.82 px), where the reduced images left 38 of 312 points.
  With a marker 4 grey levels deep and noise of half a grey level, the default
  detection method was off by 0.39 px on average at full depth against
  0.52 px reduced. The polynomial refinement and the tracker are better on
  faint markers (refinement of a 1-level spot: 0.03 px against 0.15 px, no
  spot missed against 63).
- **It does not help on strong markers without noise.** There the default
  and centre-of-mass methods are about as good either way: on average within
  0.02 px, and in a single scene up to 0.07 px better or worse, because the
  result on the reduced image depends on where the background level happens
  to fall between two 8-bit steps.
- **The corner detection method is worse at full depth on an unevenly lit
  background.** A 1-level marker on a gently sloping background was missed in
  585 of 600 trials at full depth and in none on the reduced image; at 4
  levels the error was 0.64 px against 0.33 px. On an evenly lit background
  the same method is about ten times more precise at full depth (0.05 px
  against 0.56 px). If you use the corner method on unevenly lit images,
  compare with `--xmalab3-compat`.
- **Blob thresholds** act half a grey level darker at full depth than on the
  reduced image.
- **The depth the file declares must be right.** A pixel above the declared
  depth's largest value is read as that value. If the largest value in your
  frames is far below that (12-bit data in a 16-bit file that declares
  nothing), the image is read as nearly black on the 0 to 255 scale: the
  default, centre-of-mass and corner methods still work, and the blob methods
  find nothing, as in XMALab.
- **When the information in the image is only 8 bits, results are the same.**
  On a 16-bit copy of the 8-bit test trial, detection gives exactly the 8-bit
  results, and a whole tracking run differs in 9 of 2748 points, by at most
  0.71 px (the tracker's template is no longer rounded to whole grey levels).
- **File > Undistort sequence writes deep images at their depth.** A 16-bit
  still or a `.cine` deeper than 8 bits is undistorted and written as a 16-bit
  TIFF (stating its depth when below 16 bits), where XMALab writes 8 bits
  (ledger V20). Undistorted 8-bit images are identical to XMALab's.
- To get XMALab's behaviour (everything reduced to 8 bits as XMALab reduces
  it), run `xma` with `--xmalab3-compat`.

This change (items V1, V15 and V18 in
[the ledger](parity-ledger.md)) was decided on 14 November
2026 and reviewed independently. It has not yet been tried on a real deep
dataset.

### Undistortion and calibration differ from XMALab in a few places

> **Important: on ordinary data the calibration is the same as XMALab's.** On
> the reference dataset it agrees to better than one part in a billion. The
> differences below are corrections of faults, and each one changes results
> only in the situation it names.

- **Grids turned about 30 degrees, or with a bead missing next to the centre
  bead.** XMALab can get the orientation of the bead grid wrong by 5 to 15
  degrees there, which bends the undistortion. This program detects that case
  and estimates the orientation in a way that does not depend on which
  neighbours it picks. Every other grid is matched exactly as XMALab matches
  it.
- **"remove outlier at border" keeps outliers you toggled by hand.** XMALab
  switches them back on when they lie inside the circle. After such a removal
  the undistortion can differ from XMALab's by a few tenths of a pixel.
- **"Fix principal at image center"** fixes it at the true centre. XMALab's
  is one pixel off in each direction.
- **The same input always gives the same calibration.** XMALab's search
  draws from a random sequence that depends on what was calibrated before in
  the session; here every calibration starts the sequence afresh.
- **A calibration whose last refinement step fails is reported as failed**,
  not stored as a calibrated camera.
- **After "optimize camera setup" or applying an external calibration, trials
  are recomputed.** XMALab leaves their 3D points and rigid-body
  transformations computed from the old calibration until something else
  triggers a recomputation.
- **MayaCam 2.0 import reads all eight lens coefficients** of files this
  program and XMALab export. XMALab reads only the first.
- **Checkerboard calibration** agrees with XMALab's to about 4 parts in a
  million on ordinary boards, not to the last digit (a flat board amplifies a
  rounding difference between linear-algebra libraries). Several faults of
  XMALab's checkerboard code are corrected: manual corners with an inverted
  axis, clicks that do not span the board, stale corners after a failed
  detection, a camera calibrated from a single frame.
  **With two or more cameras, click the same physical corner of the board in
  every camera.** As in XMALab, cameras that number the board differently are
  joined wrongly ("optimize camera setup" from 0.05 to 47 pixels of error in
  the reference runs); unlike XMALab, this program warns and names the camera
  and frames. It does not renumber them for you.

The compatibility mode (`--xmalab3-compat` in `xma`) reproduces XMALab 3 in
all of these. The full list, with XMALab's source lines, is in
[the ledger](parity-ledger.md) (rows A104 to A163, A170 to A209, F34 to F48, F73 to F79).

### The window: video, saving, settings, the Trial dialog and tracking

Uncompressed and MJPEG AVI, `.cine` files and image sequences are read
directly. H.264, HEVC and other videos need the `ffmpeg` and `ffprobe`
programs installed: they are looked for in `XMALAB_FFMPEG` and
`XMALAB_FFPROBE`, then next to `rustxmalab`, then on `PATH`. They are run as
separate programs; nothing of FFmpeg is built in.

When a trial's images or video are not where the project says, `rustxmalab`
searches the project's folder and its subfolders for them, as XMALab does
(unless "Disable searching and relinking for images" is set). If they are
still not found it asks for the folder (image sequences) or the file (a
video). For an image sequence, "Use a video file instead..." shows that camera
from one video file instead of the images. The choice is used for this session
only; the project is not changed.

**Saving differs from XMALab 3 on purpose** (parity ledger U52 to U54):

- Every change is an undo step: Edit > Undo (Ctrl+Z) and Edit > Redo
  (Ctrl+Shift+Z or Ctrl+Y). XMALab 3 can undo only the last point placed.
- The window title shows `*` after the project's name while there are unsaved
  changes. Closing, opening another project and quitting ask "Save changes?"
  (Save, Don't Save, Cancel) only then; XMALab 3 always shows the same warning.
- A save writes a temporary file next to the project and replaces the project
  file only when it is complete, so a failed save (a full disk, a folder you
  cannot write) leaves the previous file as it was and says so. XMALab 3
  deletes the old file first.
- **Autosave**: while a project has unsaved changes, a recovery copy is saved
  every 5 minutes (Settings > General, "Autosave a recovery copy ... every N
  minutes", 0 switches it off) next to the project as `<name>.recovery.xma`
  (`pig.xma` → `pig.recovery.xma`; a project never saved uses
  `untitled.recovery.xma` in the settings folder). It is removed when you save,
  when you close the project without unsaved changes, and when you choose
  "Don't Save" (the changes are discarded, and so is their copy). When you open
  a project whose recovery copy is newer, `rustxmalab` asks whether to open the
  recovered version; it opens under the project's own name, with its changes
  unsaved until you save. Choosing "Open saved version" removes the recovery
  copy. The recovery copy is an ordinary project file and can also be opened
  directly.

**Settings that change results are kept differently from XMALab 3 on
purpose** (parity ledger A10). In XMALab 3 the triangulation method, the
rigid-body options (optimize from 2D positions, no pose with fewer than three
correspondences, filter 3D points), the maximum reprojection error, the
default marker threshold and the two retracking options are per-machine
settings, so the same project gives different numbers on different machines.
In `rustxmalab`:

- each project stores these settings and is saved with them. Changing one in
  Settings > Tracking changes the open project: the project is recomputed as
  far as the setting requires, the change is one undo step ("Change
  Settings"), and it is saved with the project;
- the value you choose is also remembered as this machine's default (the
  `AlgorithmDefaults` entry of `settings.json`). A project that stores no such
  setting (a project saved by XMALab 3, or one created new) takes this
  machine's default when it is opened, then XMALab's own default for a setting
  never changed here. A project saved by `rustxmalab` or `xma` stores every
  setting and is not affected by the machine's defaults;
- with no project open, Settings > Tracking shows and changes only this
  machine's defaults;
- undoing a change takes the project back; the machine's default stays as
  last chosen.

The calibration settings on Settings > Calibration work; those that change
results are stored in the project, and each change is one undo step.

**The Trial dialog** (the settings button next to the trial list) changes the
trial's recording speed, cutoff frequency, 3D interpolation and reference
calibration, forces a full update, or deletes the trial (after a
confirmation); each is one undo step. Unlike XMALab 3 it writes nothing when
it is only opened (XMALab 3 replaces a reference calibration that is not
calibrated in every camera and recomputes the trial as soon as the dialog
opens), and the recording speed and cutoff are written only when you change
them, exactly as typed (XMALab 3 rounds them to two decimals on OK, so a frame
rate of 29.97002997 became 29.97). The frame bar's start and end frame buttons
change the trial's range in the project (an undo step); the frame on screen is
moved into the new range (XMALab 3 could leave it one frame past the end).

**Placing points and tracking** work as in XMALab 3 (Marker tracking
workspace): Ctrl+click in a camera view adds a point for the next marker
without one (Command+click on macOS), a click moves the active marker and
snaps it by detection, a click in the Detailed View places it exactly
(status "manual"), Shift+click makes the nearest marker active. The Toolbox's
single, selected and all buttons (and the keys 1 and 2, with Ctrl for the
selected markers and Alt to run on) track one frame or run on until the trial's
first or last frame; a running button shows pressed, and pressing it again,
Escape or the Stop button stops the run. A run gives exactly the points `xma
track` gives for the same request, and the whole run is one undo step.
XMALab 3's merge-aware tracking ("Enable merge-aware tracking") is not in this
program: it was removed on 26 November 2026, and "Enable multi-camera tracking"
in the same place replaces it (ledger A226, U171). Points panel > "Refine center of selected Points by
polynomial fit" runs frame by frame with a "Refining center" progress; Cancel
keeps the frames already refined, as XMALab does. View > DetectionSettings
shows the detection stages at the last point clicked and changes the active
marker's detection method and threshold. **Differences from XMALab 3 on
purpose:** while a run (or the refinement) goes on, clicks place nothing and
the other tracking buttons and keys do nothing (XMALab 3 acts on them between
the steps of a run, which can start a second run over the first); every
placement is an undo step; Detection Settings changes the method of any marker
(XMALab 3 ignores the box for blob, corner and "no detection" markers) and
says "Nothing detected" where XMALab 3 read past the end of its stage list.

Settings are kept in `settings.json` in the `RustXMALab` folder of the
platform's configuration folder (`~/.config` on Linux, `~/Library/Application
Support` on macOS, `%APPDATA%` on Windows), or in `XMALAB_RS_CONFIG_DIR` if
that is set. XMALab's own settings are not read or changed.

`rustxmalab` computes with XMALab 3's known faults fixed, as `xma` does by
default; the XMALab 3 compatibility mode (`--xmalab3-compat`) exists only on
`xma`. Where XMALab 3's interface was faulty the window also does the
corrected thing; both are listed in [the ledger](parity-ledger.md).
