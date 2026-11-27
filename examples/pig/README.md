# The pig example dataset

The standard example dataset for XMALab: a pig's head, filmed with two X-ray
cameras at 250 frames per second (1024 x 1024, 145 frames), with ten markers in
the skull and lower jaw. It was published on the XMA Portal (sandbox study 49,
trial 2) as XMALab's example data.

| File | What it is |
|---|---|
| `undistortionGrid_cam1.tif`, `undistortionGrid_cam2.tif` | Images of the undistortion grid, one per camera |
| `Calibration_cam1.tif`, `Calibration_cam2.tif` | Images of the calibration cube, one per camera |
| `FrameSpecification.csv`, `Reference_points.ref` | The calibration cube: its bead coordinates and the four reference beads |
| `Camera1.avi`, `Camera2.avi` | The trial, one video per camera |
| `CT_Markers_Skull.csv`, `CT_Markers_Jaw.csv` | Marker positions from CT, for the two rigid bodies |
| `Cranium.obj`, `LowerJaw.obj` | Bone meshes from CT, in the same coordinates |
| `PigMarker_Numbers.pdf` | Where each marker is |

To build a project from these files, follow the user manual from
[Create a Dataset](https://github.com/pfalkingham/RustXMALab/wiki/Create-a-Dataset).

The videos are H.264, so RustXMALab reads them through FFmpeg: install `ffmpeg`
and `ffprobe` first (see the manual's Getting Started page).
