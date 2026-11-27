# Notice

## Origin

This program is a port to Rust of **XMALab** (XMA Lab), copyright (c) 2015
Brown University, Providence, RI. All rights reserved. XMALab was originally
developed by Benjamin Knörlein.

XMALab is distributed under the terms of the GNU General Public License
version 3 as published by the Free Software Foundation. This port is
distributed under the same licence (GPL-3.0-only); the full text is in
[LICENSE](LICENSE).

XMALab's licence grants use under the GPL version 3 provided that its
copyright notice appear in all copies and that **the name of Brown University
not be used in advertising or publicity pertaining to the use or distribution
of the software without specific, written prior permission from Brown
University.** That condition applies to this port as well.

XMALab's licence also states that Brown University disclaims all warranties
with regard to the software, which is provided "as is", including all implied
warranties of merchantability and fitness for any particular purpose, and that
in no event shall Brown University be liable for any special, indirect or
consequential damages arising out of or in connection with the use or
performance of the software.

## Citation

Work that uses this program must cite the original XMALab paper:

> Knörlein, B. J., Baier, D. B., Gatesy, S. M., Laurence-Chasen, J. D. and
> Brainerd, E. L. (2016). Validation of XMALab software for marker-based
> XROMM. *Journal of Experimental Biology* 219, 3701-3711.
> doi:10.1242/jeb.145383

## Licence texts in the release archive

The Linux release archive (`cargo xtask dist`) carries, beside `LICENSE` (GPL
version 3) and this file:

- `THIRD-PARTY-LICENSES.txt`: the Rust crates the programs are built from (see
  [Rust crates](#rust-crates)), written by `cargo xtask licenses`.
- `licenses/`: the copyright notices and licence texts of the works below,
  listed in `licenses/README.txt`. In the source tree the same files are in
  [`licenses/`](licenses/).

## Third-party code ported into this project

Each entry names the source project, its copyright and licence, the file of
`licenses/` that holds them, and the files or functions ported. Every ported
function also names its origin in its doc comment. Where a work is offered
under several licences or "any later version", the option used is stated. The
licences below permit use in a program distributed under GPL-3.0-only. (Two
pieces XMALab itself took from sources whose terms were not clear, its
`filtfilt` and the CodeCogs cubic spline, are no longer ported: this project's
own code, written from a mathematical specification, replaced them on 16
November 2026; see the liir entry and the one after it.)

### XMALab: the program, its interface, icons and style sheet (GPL-3.0)

Copyright (c) 2015 Brown University, Providence, RI (see Origin). Licence and
notice: `licenses/XMALab-license.txt` (XMALab's `license.txt`).

The whole program is a port of XMALab's source; its windows and dialogs are
ported control by control from XMALab's forms (`src/ui/ui-files/`).
`crates/xma-app/assets/icons/*.png` (21 files) and
`crates/xma-app/assets/dark.qss` are byte-identical copies of files in
XMALab's `src/ui/resource-files/icons/` and `src/ui/resource-files/theme/`. The
icons are embedded in the `rustxmalab` program; `dark.qss` is kept for reference,
and its colours are written into `crates/xma-app/src/theme.rs`.

### OpenCV 4.13.0 (Apache-2.0)

Copyright (C) 2000-2022 Intel Corporation; 2009-2011 Willow Garage Inc.;
2009-2016 NVIDIA Corporation; 2010-2013 Advanced Micro Devices, Inc.;
2015-2023 OpenCV Foundation; 2008-2016 Itseez Inc.; 2019-2023 Xperience AI;
2019-2022 Shenzhen Institute of Artificial Intelligence and Robotics for
Society; 2022-2023 Southern University of Science And Technology; 2023-2025
OpenCV AI; third-party copyrights are property of their respective owners
(OpenCV's `COPYRIGHT`). Licence: Apache License 2.0. Texts:
`licenses/OpenCV-LICENSE.txt` and `licenses/OpenCV-COPYRIGHT.txt` (OpenCV
4.13.0 has no top-level `NOTICE` file). The ported files carry prominent
notice of their origin in their doc comments, as Apache-2.0 section 4 asks.

- `crates/xma-vision/src/linalg.rs`: `JacobiSVDImpl_`, `SVD::compute`,
  `SVD::solveZ`, `SVBkSb`, `cv::solve`, `cv::invert`, `cv::determinant`,
  `LUImpl`, `QRImpl` and `cv::sortIdx`, from `modules/core/src/lapack.cpp`,
  `matrix_decomp.cpp`, `matrix_operations.cpp` and `hal_internal.cpp`.
- `crates/xma-vision/src/camera.rs`: `cv::Rodrigues`, `cv::projectPoints` and
  `cv::undistortPoints`, from `modules/calib3d/src/calibration_base.cpp` and
  `undistort.dispatch.cpp`.
- `crates/xma-vision/src/epnp.rs`: `cv::solvePnP` with `SOLVEPNP_EPNP`, from
  `modules/calib3d/src/solvepnp.cpp` and `epnp.cpp` (the EPnP method of
  V. Lepetit, F. Moreno-Noguer and P. Fua).
- `crates/xma-vision/src/calib3d.rs` and `calib3d/` (`extrinsic.rs`,
  `homography.rs`, `levmarq.rs`, `project.rs`, `cvcore.rs`): camera
  calibration, homography and the `CvLevMarq` solver, from
  `modules/calib3d/src/calibration.cpp`, `calibration_base.cpp`, `fundam.cpp`,
  `levmarq.cpp`, `compat_ptsetreg.cpp` and `undistort.dispatch.cpp`, and the
  matrix, norm and mean routines of `modules/core/src/` (`matmul`, `norm`,
  `mean`, `sum`, `convert_scale`).
- `crates/xma-vision/src/cube.rs`, `rigid.rs`, `triangulation.rs`,
  `undistortion.rs`, `grid_match.rs`, `filter.rs`: OpenCV matrix arithmetic as
  XMALab uses it (`matrix_expressions.cpp`, `matmul.simd.hpp`, `norm.simd.hpp`,
  `convert.dispatch.cpp`, `mean.dispatch.cpp`).
- `crates/xma-vision/src/imgproc/`: `arith.rs` (`arithm.cpp`, `convert*`,
  `minmax.cpp`, `norm.dispatch.cpp`, `thresh.cpp`); `blob.rs`
  (`cv::SimpleBlobDetector`, `modules/features2d/src/blobdetector.cpp`);
  `blur.rs` (`filter.dispatch.cpp`, `filter.simd.hpp`, `median_blur.simd.hpp`,
  `smooth.dispatch.cpp`, `copy.cpp`); `contours.rs` (`cv::findContours`,
  `contours_new.cpp`, `contours_common.cpp`, `contours.cpp`); `corners.rs`
  (`corner.cpp`, `cornersubpix.cpp`, `deriv.cpp`, `featureselect.cpp`,
  `samplers.cpp`); `shape.rs` (`cv::minEnclosingCircle`, `cv::arcLength`,
  `cv::contourArea`, `cv::moments`, `cv::convexHull`: `shapedescr.cpp`,
  `moments.cpp`, `convhull.cpp`); `subpix.rs` (`samplers.cpp`); `template.rs`
  and `template/dxt.rs` (`cv::matchTemplate` and the DFT behind it:
  `templmatch.cpp`, `dxt.cpp`, `sumpixels.dispatch.cpp`).
- `crates/xma-vision/src/tracking.rs`: `templmatch.cpp`, `samplers.cpp`.
- `crates/xma-vision/src/visual_filter.rs`: `cv::GaussianBlur` on 8-bit images
  (the bit-exact fixed-point path: `getGaussianKernelBitExact`,
  `getGaussianKernelFixedPoint_ED`, `fixedSmoothInvoker`, from
  `modules/imgproc/src/smooth.dispatch.cpp` and `smooth.simd.hpp`),
  `cv::addWeighted` and the gamma lookup table XMALab builds with OpenCV.
- `crates/xma-vision/src/denoise.rs`: `cv::fastNlMeansDenoisingMulti` for 8-bit
  single-channel images (`modules/photo/src/denoising.cpp`,
  `fast_nlmeans_multi_denoising_invoker.hpp`,
  `fast_nlmeans_denoising_invoker_commons.hpp`) and `cv::borderInterpolate` with
  `BORDER_REFLECT_101` (`modules/core/src/copy.cpp`). The photo module's source was
  read from the OpenCV 4.13.0 release archive.
- `crates/xma-vision/src/remap.rs`: `cv::remap` with `INTER_LANCZOS4` and a
  constant border (`RemapInvoker`, `initInterTab2D`, `remapLanczos4`, from
  `modules/imgproc/src/imgwarp.cpp`); `cv::initUndistortRectifyMap` without
  rectification (`modules/calib3d/src/undistort.dispatch.cpp`); `minMaxIdx`,
  `convertScaleAbs`, `multiply`, and the `COLORMAP_JET` and `COLORMAP_HSV`
  tables of `cv::applyColorMap` (values taken from OpenCV 4.14).
- `crates/xma-video/src/convert.rs`: `cv::cvtColor` with `COLOR_RGB2GRAY` on
  8-bit data (`RGB2Gray<uchar>`, `modules/imgproc/src/color_rgb.simd.hpp`) and
  `Mat::convertTo` to 8 bits with a scale factor.
- `crates/xma-video/src/ffmpeg/probe.rs` and `grey.rs`: the frame rate and
  frame count that OpenCV's FFmpeg capture reports, and the conversion it
  makes (`CvCapture_FFMPEG::get_fps`, `get_total_frames`, `get_duration_sec`,
  `retrieveFrame`, from `modules/videoio/src/cap_ffmpeg_impl.hpp`).
- `crates/xma-io/src/cfmt.rs`: the CSV layout of OpenCV's matrix formatter
  (`cv::format` with `FMT_CSV`, `modules/core/src/out.cpp`).

The OpenCV licence was checked in the OpenCV 4.13.0 source used for porting
(its `LICENSE` file).

### levmar 2.6 (GPL-2.0-or-later, used under GPL-3.0)

Copyright (C) 2004 Manolis Lourakis, Institute of Computer Science,
Foundation for Research and Technology - Hellas, Heraklion, Crete, Greece.
levmar's own `LICENSE` file is the GPL version 2, but each source file grants
"version 2 of the License, or (at your option) any later version"; the port is
used under version 3 (`LICENSE`). Texts: `licenses/levmar-NOTICE.txt` (the
copyright line, the grant and the header of `lm_core.c`) and
`licenses/levmar-LICENSE-GPL-2.0.txt` (levmar's `LICENSE`).

- `crates/xma-vision/src/levmar.rs`: `dlevmar_der` and the helpers it uses,
  from `lm_core.c`, `misc_core.c` and `Axb_core.c`.

### LAPACK and BLAS (modified BSD) and OpenBLAS 0.3.29 (BSD 3-clause)

LAPACK and the reference BLAS: Copyright (c) 1992-2023 The University of
Tennessee and The University of Tennessee Research Foundation; Copyright (c)
2000-2023 The University of California Berkeley; Copyright (c) 2006-2023 The
University of Colorado Denver. Text: `licenses/LAPACK-LICENSE.txt`.

OpenBLAS: Copyright 2011-2024 The OpenBLAS Project; 2009-2010 The University
of Texas at Austin; 2013 Martin Koehler; 2020 IBM Corporation; 2022 Arm Ltd.
Text: `licenses/OpenBLAS-LICENSE.txt` (from Debian's packaging of 0.3.29; the
upstream `LICENSE` file was not available when it was added).

- `crates/xma-vision/src/levmar.rs`: the unblocked Bunch-Kaufman
  factorisation and solve (LAPACK `dsytf2`, `dsytrs`) and the BLAS routines
  they call (`IDAMAX`, `DSYR`, `DGER`, `DGEMV`), written to reproduce the
  results of OpenBLAS 0.3.29 as the reference build uses it.

### GNU libstdc++ (GPL-3.0-or-later with the GCC Runtime Library Exception)

Copyright (C) 2001-2026 Free Software Foundation, Inc.; the file also carries
notices of Hewlett-Packard Company (1994) and Silicon Graphics Computer
Systems, Inc. (1996). Used under GPL version 3 (`LICENSE`). Texts:
`licenses/libstdc++-NOTICE.txt` (the header of `bits/stl_algo.h`) and
`licenses/GCC-RUNTIME-LIBRARY-EXCEPTION.txt`.

- `crates/xma-vision/src/linalg.rs`: `std::sort` (introsort, heap sort and
  final insertion sort from `bits/stl_algo.h` and `bits/stl_heap.h`), which
  `cv::sortIdx` relies on.

### QCustomPlot 2.1.1 (GPL-3.0-or-later)

Copyright (C) 2011-2022 Emanuel Eichhammer. Bundled with XMALab in
`src/external/QCustomPlot/`. Used under GPL version 3 (`LICENSE`; QCustomPlot
ships the same text as `GPL.txt`). Text: `licenses/QCustomPlot-NOTICE.txt`
(the licence header of `qcustomplot.cpp`).

- `crates/xma-app/src/widgets/plot/ticks.rs`: `QCPAxisTicker::generate`,
  `getTickStep`, `getSubTickCount`, `getMantissa`, `pickClosest`, `trimTicks`.
- `crates/xma-app/src/widgets/plot/ranging.rs`: `QCPRange::validRange`,
  `QCPAxis::setRange`, `QCPAxis::scaleRange`, `QCPAxisRect::wheelEvent`.
- `crates/xma-app/src/widgets/plot/input.rs`: `QCPAxisRect::mouseMoveEvent`
  (range drag).
- `crates/xma-app/src/widgets/plot/decimate.rs`:
  `QCPGraph::dataToStepCenterLines`.
- `crates/xma-app/src/widgets/plot/paint.rs`: `QCPGrid::drawGridLines`,
  `QCPAxisPainterPrivate::draw`, `QCPScatterStyle::drawShape`,
  `QCPErrorBars::getErrorBarLines`.

### LibTIFF (libtiff licence)

Copyright (c) 1988-1997 Sam Leffler; Copyright (c) 1991-1997 Silicon
Graphics, Inc. The licence permits use, copying and modification provided
the copyright notices and permission notice appear in all copies and the
names of Sam Leffler and Silicon Graphics are not used in advertising or
publicity without permission. Text: `licenses/LibTIFF-LICENSE.md`.

- `crates/xma-video/src/image_file.rs`: the expansion of 8-bit palette TIFF
  images (`checkcmap`, `cvtcmap`, `put8bitcmaptile` from `tif_getimage.c`), as
  OpenCV's TIFF decoder reads them.

### liir (GPL-2.0-or-later, used under GPL-3.0)

Copyright (C) 2007 Exstrom Laboratories LLC. XMALab's
`processing/ButterworthLowPassFilter.cpp` takes its Butterworth low-pass
coefficients from Exstrom's `liir.c`, under the GNU General Public License
version 2 or (at the user's option) any later version; this code is used under
version 3 (`LICENSE`). Text: `licenses/liir-NOTICE.txt` (the liir part of that
file's header).

- `crates/xma-vision/src/filter.rs`: `dcof_bwlp`, `ccof_bwlp`, `sf_bwlp`,
  `binomial_mult`.

### Zero-phase filtering and cubic spline: original code, not ported

The zero-phase filtering (`ButterworthLowPass::filter` and the private
functions it calls in `crates/xma-vision/src/filter.rs`) and the cubic spline
(`cubic_spline` and its private functions in the same file) are original code
of this project, under its own licence (`LICENSE`). They were written on 16
November 2026 from a mathematical specification,
[`docs/filter-spline-spec.md`](docs/filter-spline-spec.md), without reference
to the code they replace, and give the same numbers. They replace ports of
two pieces of XMALab whose terms were not clear: its `filtfilt` (in
`processing/ButterworthLowPassFilter.cpp`, credited there to a Stack Overflow
answer, question 17675053, answer 27270420, without a licence) and its cubic
spline (`processing/cubic.h`: `tridiag`, `getYD`, `ispline`, copyright
2004-2010 CodeCogs, Zyba Ltd, under "the GNU General Public License as
published by CodeCogs"). No code ported from either remains in this project.

The methods themselves are published, and the specification cites them:
zero-phase filtering with odd edge extension and a step steady-state initial
state as documented for MATLAB's `filtfilt` (The MathWorks) and SciPy's
`scipy.signal.filtfilt` and `lfilter_zi`; F. Gustafsson, "Determining the
initial states in forward-backward filtering", IEEE Transactions on Signal
Processing 44(4), 988-992, 1996; the transposed direct form II (A. V.
Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*); the
not-a-knot cubic spline in slope (Hermite) form (G. Wolberg, "Cubic spline
interpolation: a review", Columbia University technical report CUCS-389-88,
1988; C. de Boor, *A Practical Guide to Splines*); and the tridiagonal
(Thomas) algorithm (R. L. Burden and J. D. Faires, *Numerical Analysis*; W. H.
Press et al., *Numerical Recipes*). The initial state's matrix inverse uses
this project's port of OpenCV's `cv::invert` (see the OpenCV entry).

The filter and spline were tested against outputs of XMALab's own code,
compiled from the C++ original; those reference numbers are kept outside this
repository and are not distributed with the program.

### Qt 6 qtbase (GPL-3.0-only option)

Copyright (C) The Qt Company Ltd. qtbase is offered under
`LicenseRef-Qt-Commercial OR LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only`;
this code is used under GPL-3.0-only (`LICENSE`). Text:
`licenses/Qt-NOTICE.txt`.

- `crates/xma-io/src/cparse.rs`: `QColor::fromString` (`qcolor.cpp`).
- `crates/xma-io/src/export/mod.rs`: `QFileSystemEntry::completeBaseName`.
- `crates/xma-io/src/xml.rs`: `QXmlStreamWriterPrivate` (`qxmlstream.cpp`).

### GNU C Library (LGPL-2.1-or-later, used under GPL-3.0)

Copyright (C) 1991-2026 Free Software Foundation, Inc. Used under the GPL
version 3 as section 3 of the LGPL 2.1 permits. Texts:
`licenses/glibc-NOTICE.txt` and `licenses/LGPL-2.1.txt`.

- `crates/xma-vision/src/crand.rs`: `srand` and `rand` (`__srandom_r`,
  `__random_r` from `stdlib/random_r.c`).

## Rust crates

The programs are written in Rust and link statically the Rust crates they
depend on (188 crates for `rustxmalab` and `xma` together on Linux; 26 more are
used only while building them). Each crate, with its licence expression, the
licence used where it offers a choice, its authors and copyright lines and the
full text of every licence file it ships, is listed in
`THIRD-PARTY-LICENSES.txt` in the release archive (`cargo xtask licenses`
writes it). All are under licences that permit distribution with a
GPL-3.0-only program (MIT, Apache-2.0, Zlib, ISC, BSD, Unlicense, Unicode-3.0
and similar); where a crate offers a choice, a compatible option is taken (for
`self_cell`, "Apache-2.0 OR GPL-2.0-only", Apache-2.0). The fonts are below.

## Programs used at run time, not included

- **FFmpeg** (`ffmpeg` and `ffprobe`). When a video is not in a format the
  program reads itself (uncompressed or MJPEG AVI, `.cine`, image files), for
  example H.264 or HEVC, it runs the `ffmpeg` and `ffprobe` executables
  installed on the machine as separate programs (found through
  `XMALAB_FFMPEG` and `XMALAB_FFPROBE`, next to the program, or on `PATH`) and
  reads their output. Nothing of FFmpeg is linked into or shipped with this
  program. To give the same grey values as XMALab, which decodes through
  OpenCV and FFmpeg, `crates/xma-video/src/ffmpeg/grey.rs` reproduces the luma
  expansion of FFmpeg's libswscale (`luma_table`); libswscale is
  LGPL-2.1-or-later, copyright Michael Niedermayer and others
  (`licenses/FFmpeg-NOTICE.txt`).

## Fonts

The `rustxmalab` program embeds five font files of egui's `default_fonts` and
`monochrome_emoji_fonts` features (crate `epaint_default_fonts` 0.37.0, declared
`(MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0`). The fonts are data
inside the program, each under its own licence; their licence texts are in
`THIRD-PARTY-LICENSES.txt` under that crate (from its `fonts/` folder).

- **Ubuntu Light** (`Ubuntu-Light.ttf`): Copyright 2011 Canonical Ltd., Ubuntu
  Font Licence 1.0 (`UFL.txt`). The licence allows the font to be bundled,
  embedded and redistributed with software provided each copy carries the
  copyright notice and the licence, and the font itself stays under that
  licence. "Ubuntu and Canonical are registered trademarks of Canonical Ltd."
- **Hack Regular** (`Hack-Regular.ttf`): Copyright (c) 2018 Source Foundry
  Authors, MIT licence; it includes work from DejaVu (public domain) and
  Bitstream Vera Sans Mono, Copyright (c) 2003 by Bitstream, Inc., under the
  Bitstream Vera licence (`Hack-Regular.txt`), which allows the font to be sold
  as part of a larger software package but not by itself.
- **egui icons** (`egui-icons.ttf`): five glyphs taken from emoji-icon-font,
  Copyright (c) 2014 John Slegers, MIT licence
  (`emoji-icon-font-mit-license.txt`, `egui-icons.txt`).

- **Noto Emoji Regular** (`NotoEmoji-Regular.ttf`): "Copyright 2013 Google Inc.
  All Rights Reserved." (the font's own name table; "Noto is a trademark of
  Google Inc."), SIL Open Font Licence 1.1 (`OFL.txt`). The licence allows the font to be
  bundled, embedded and redistributed with software provided each copy carries
  the copyright notice and the licence; the font may not be sold by itself and
  stays under that licence.
- **emoji-icon-font** (`emoji-icon-font.ttf`): Copyright (c) 2014 John Slegers,
  MIT licence (`emoji-icon-font-mit-license.txt`).

The last two come with the `monochrome_emoji_fonts` feature, enabled on
15 November 2026 when the program's look became egui's own (parity ledger
U139): egui's colour picker labels a button with a glyph from them.
