# Changes

## 0.1.0

The first public release.

- Everything XMALab 3 does in daily use: building a dataset from raw files or an
  XMA Portal download, undistortion, cube and checkerboard calibration, trials,
  marker tracking and editing, rigid bodies, filtering, denoising, the plots and
  the 3D world view, and XMALab 3's exports apart from XROMM VR, DLT, MayaCam
  version 1 and the XRayProject Matlab format.
- Projects are XMALab 3's `.xma` files, readable by both programs.
- New: multi-camera tracking, undo and redo, autosaved recovery copies, settings
  stored in the project, tracking at full bit depth, and export to Blender.
- Known faults of XMALab 3 are fixed; [Coming from XMALab 3](docs/for-xmalab-users.md)
  lists every difference.
- Builds for Linux (x86_64), Windows (x86_64) and macOS (Apple Silicon). Only the
  Linux build has been tried.
