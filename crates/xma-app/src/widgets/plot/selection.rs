//! The highlighted frame range, as `PlotWindow` keeps it.
//!
//! The range belongs to the caller (the Plot panel), not to the widget: the
//! widget only reports Ctrl-press and Ctrl-drag frames ([`super::PlotEvent`]),
//! and the Shift handling depends on application-wide key state (XMALab's
//! `Shortcuts` event filter calls `ShiftPressed` / `ShiftReleased` whatever
//! has focus). This type reproduces the C++ state machine so the panel only
//! has to feed it.

use super::PlotEvent;
use super::model::{FrameAxis, SelectionRect};

/// `startFrame`, `endFrame`, `tmpStartFrame`, `shiftPressed` and `noSelection`
/// from `PlotWindow`, with the methods that change them. Frames are 0-based and
/// may lie outside the trial, as in the C++ (the users clamp them).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameSelection {
    /// `startFrame`.
    pub start: i64,
    /// `endFrame`.
    pub end: i64,
    tmp_start: i64,
    shift: bool,
    no_selection: bool,
}

impl Default for FrameSelection {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameSelection {
    /// The state after the `PlotWindow` constructor: 0..0, no selection, Shift up.
    pub fn new() -> Self {
        Self {
            start: 0,
            end: 0,
            tmp_start: 0,
            shift: false,
            no_selection: true,
        }
    }

    /// Whether a range has been selected (the C++ `!noSelection`).
    pub fn has_selection(&self) -> bool {
        !self.no_selection
    }

    /// Shift went down while `active_frame` was current.
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::ShiftPressed.
    pub fn shift_pressed(&mut self, active_frame: i64) {
        self.tmp_start = active_frame;
        self.shift = true;
    }

    /// Shift went up.
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::ShiftReleased.
    pub fn shift_released(&mut self) {
        self.shift = false;
    }

    /// Feeds the Shift key state once per UI frame; calls [`Self::shift_pressed`]
    /// or [`Self::shift_released`] on a change. Egui has no key event for Shift
    /// itself, only modifier state, so edges are detected here.
    pub fn update_shift(&mut self, shift_down: bool, active_frame: i64) {
        if shift_down && !self.shift {
            self.shift_pressed(active_frame);
        } else if !shift_down && self.shift {
            self.shift_released();
        }
    }

    /// `shiftPressed = false;` as the C++ does on a trial, workspace, marker
    /// or rigid-body change.
    pub fn forget_shift(&mut self) {
        self.shift = false;
    }

    /// The active frame changed (from anywhere: keys, frame bar, plot).
    /// Returns true when the C++ would redraw the plot.
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::activeFrameTrialChanged.
    pub fn active_frame_changed(&mut self, frame: i64) -> bool {
        if self.shift && self.tmp_start != frame {
            self.start = self.tmp_start;
            self.end = frame;
            self.no_selection = false;
            true
        } else {
            if self.no_selection {
                self.start = frame;
                self.end = frame;
            }
            false
        }
    }

    /// Ctrl + left press at `frame`.
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::eventFilter (MouseButtonPress, Ctrl).
    pub fn ctrl_press(&mut self, frame: i64) {
        self.start = frame;
        self.end = frame;
        self.no_selection = true;
    }

    /// Ctrl + left drag over `frame`.
    ///
    /// Ported from ui/PlotWindow.cpp, PlotWindow::eventFilter (MouseMove, Ctrl).
    pub fn ctrl_drag(&mut self, frame: i64) {
        self.end = frame;
        self.no_selection = self.end == self.start;
    }

    /// Applies a widget event. Call this for every event in order; for
    /// [`PlotEvent::Frame`] the caller then changes the active frame and calls
    /// [`Self::active_frame_changed`] if it really changed (as `State` only emits on change).
    pub fn apply(&mut self, event: &PlotEvent) {
        match *event {
            PlotEvent::SelectionStart(f) => self.ctrl_press(f),
            PlotEvent::SelectionEnd(f) => self.ctrl_drag(f),
            _ => {}
        }
    }

    /// The range ordered and clamped to the trial, as `deleteData`,
    /// `setUntrackable`, `setInterpolation` and the event setters use it.
    pub fn clamped(&self, frames: &FrameAxis) -> (i64, i64) {
        let (lo, hi) = if self.end > self.start {
            (self.start, self.end)
        } else {
            (self.end, self.start)
        };
        (lo.max(frames.first as i64), hi.min(frames.last as i64))
    }

    /// The selection rectangle as the plot functions place it: from the x of
    /// `startFrame` to the x of `endFrame` (frame centres, not cell edges),
    /// spanning `span` in left-axis coordinates. Drawn even when empty, as a
    /// zero-width rectangle, as in the C++.
    pub fn rect(&self, frames: &FrameAxis, span: Option<(f64, f64)>) -> SelectionRect {
        SelectionRect::new((frames.x_of_i(self.start), frames.x_of_i(self.end)), span)
    }
}
