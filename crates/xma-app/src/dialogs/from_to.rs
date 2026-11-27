//! `FromToDialog` ("Select Range"): a frame range, with optional rows for an image format
//! and for filtered or unfiltered data.
//!
//! Ported control by control from `FromToDialog.ui` and ui/FromToDialog.cpp:
//!
//! - `frameFromTo` (row 1): "From" `spinBoxFrom`, "to" `spinBoxTo`, each 1 to `max`
//!   (the trial's number of frames), starting at the values given (the trial's start and
//!   end frame), 1-based as the C++ shows them.
//! - `frameFormat` (row 2, shown only `withFormat`): "jpg" (checked), "tif", "avi".
//! - `frameFilter` (row 3, shown only `withFilter`): "unfiltered" (checked), "filtered".
//! - Row 4: a spacer, "Cancel", "OK".
//!
//! The Points panel uses it without the two optional rows (E18, E21, E22); the format and
//! filter rows are there for the exports that use them.

use egui::Ui;

use super::form;

/// `FromToDialog::getFormat`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImageFormat {
    #[default]
    Jpg,
    Tif,
    Avi,
}

impl ImageFormat {
    /// The text `getFormat` returns.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ImageFormat::Jpg => "jpg",
            ImageFormat::Tif => "tif",
            ImageFormat::Avi => "avi",
        }
    }
}

/// The dialog's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FromTo {
    /// `spinBoxFrom`, 1-based.
    pub from: i64,
    /// `spinBoxTo`, 1-based.
    pub to: i64,
    /// The maximum of both spin boxes (the minimum is 1).
    pub max: i64,
    /// `frameFormat` shown.
    pub with_format: bool,
    /// `frameFilter` shown.
    pub with_filter: bool,
    pub format: ImageFormat,
    /// `radioFiltered` checked.
    pub filtered: bool,
}

/// What the buttons did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FromToAnswer {
    Ok,
    Cancel,
}

impl FromTo {
    /// Ported from ui/FromToDialog.cpp, FromToDialog::FromToDialog: both spin boxes range
    /// 1 to `max`, set to `from` and `to` (Qt clamps a value outside the range).
    #[must_use]
    pub fn new(from: usize, to: usize, max: usize, with_format: bool, with_filter: bool) -> Self {
        let max = i64::try_from(max).unwrap_or(i64::MAX).max(1);
        let clamp = |v: usize| i64::try_from(v).unwrap_or(i64::MAX).clamp(1, max);
        FromTo {
            from: clamp(from),
            to: clamp(to),
            max,
            with_format,
            with_filter,
            format: ImageFormat::Jpg,
            filtered: false,
        }
    }

    /// The chosen frames, 0-based (`getFrom() - 1`, `getTo() - 1`).
    #[must_use]
    pub fn frames(&self) -> (usize, usize) {
        let z = |v: i64| usize::try_from(v - 1).unwrap_or(0);
        (z(self.from), z(self.to))
    }

    /// Draws the form below the title row; returns the button pressed.
    pub fn ui(&mut self, ui: &mut Ui) -> Option<FromToAnswer> {
        ui.set_min_width(420.0);
        // frameFromTo.
        form::styled_frame(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("From");
                if let Some(v) =
                    form::spin_box(ui, true, self.from, 1, self.max, 80.0, "From frame")
                {
                    self.from = v;
                }
                ui.label("to");
                if let Some(v) = form::spin_box(ui, true, self.to, 1, self.max, 80.0, "To frame") {
                    self.to = v;
                }
            });
        });
        // frameFormat.
        if self.with_format {
            form::styled_frame(ui, |ui| {
                ui.horizontal(|ui| {
                    for (f, t) in [
                        (ImageFormat::Jpg, "jpg"),
                        (ImageFormat::Tif, "tif"),
                        (ImageFormat::Avi, "avi"),
                    ] {
                        if form::radio(ui, true, self.format == f, t) {
                            self.format = f;
                        }
                    }
                });
            });
        }
        // frameFilter.
        if self.with_filter {
            form::styled_frame(ui, |ui| {
                ui.horizontal(|ui| {
                    if form::radio(ui, true, !self.filtered, "unfiltered") {
                        self.filtered = false;
                    }
                    if form::radio(ui, true, self.filtered, "filtered") {
                        self.filtered = true;
                    }
                });
            });
        }
        ui.add_space(4.0);
        let mut answer = None;
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if form::button(ui, true, "OK", 80.0).clicked() {
                answer = Some(FromToAnswer::Ok);
            }
            if form::button(ui, true, "Cancel", 80.0).clicked() {
                answer = Some(FromToAnswer::Cancel);
            }
        });
        answer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_clamped_and_frames_zero_based() {
        let d = FromTo::new(0, 120, 100, false, false);
        assert_eq!((d.from, d.to, d.max), (1, 100, 100));
        assert_eq!(d.frames(), (0, 99));
        assert_eq!(FromTo::new(5, 10, 20, true, true).format.as_str(), "jpg");
    }
}
