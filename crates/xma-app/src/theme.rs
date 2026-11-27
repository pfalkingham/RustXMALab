//! Light, dark and system themes: egui's own.
//!
//! The author's decision (15 November 2026, parity ledger U139): the look is the GUI
//! toolkit's own. **Light** and **Dark** are `egui::Visuals::light()` and
//! `egui::Visuals::dark()` with egui's default style (spacing, margins, rounding, text
//! sizes); **Follow System** is `egui::ThemePreference::System`, which egui resolves every
//! frame from the operating system's preference (eframe passes it in), so the window
//! changes when the system does. XMALab's Qt palettes, `dark.qss` and Qt metrics are not
//! imitated.
//!
//! Colours the program draws itself are read from the visuals in force where they are
//! drawn (`ui.visuals()`), through the functions below, so that every part of the window
//! follows one theme at a time. Only colours that carry meaning in XMALab keep fixed
//! values (marker status colours, plot curve colours, axes, camera-view overlays, icons);
//! they live with the code that draws them.
//!
//! Menu labels and settings are ported from ui/MainWindow.cpp, MainWindow::createThemeMenu.

use egui::{Color32, Visuals};

/// The theme choice (`Theme` setting: `"light"`, `"dark"`, `"system"`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ThemeChoice {
    Light,
    Dark,
    #[default]
    System,
}

impl ThemeChoice {
    /// From the setting value. Anything unknown is "system", as in
    /// `MainWindow::createThemeMenu`.
    #[must_use]
    pub fn from_setting(s: &str) -> Self {
        match s {
            "light" => ThemeChoice::Light,
            "dark" => ThemeChoice::Dark,
            _ => ThemeChoice::System,
        }
    }

    /// The setting value.
    #[must_use]
    pub fn as_setting(self) -> &'static str {
        match self {
            ThemeChoice::Light => "light",
            ThemeChoice::Dark => "dark",
            ThemeChoice::System => "system",
        }
    }

    /// The menu label (View > Theme).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            ThemeChoice::Light => "Light",
            ThemeChoice::Dark => "Dark",
            ThemeChoice::System => "Follow System",
        }
    }

    /// egui's theme preference for this choice.
    #[must_use]
    pub fn preference(self) -> egui::ThemePreference {
        match self {
            ThemeChoice::Light => egui::ThemePreference::Light,
            ThemeChoice::Dark => egui::ThemePreference::Dark,
            ThemeChoice::System => egui::ThemePreference::System,
        }
    }

    pub const MENU_ORDER: [ThemeChoice; 3] =
        [ThemeChoice::Light, ThemeChoice::Dark, ThemeChoice::System];
}

/// The theme in use. Which egui theme is in force
/// (light or dark) is egui's to resolve; read it from `ui.visuals()`, never from here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Theme {
    pub choice: ThemeChoice,
}

impl Theme {
    #[must_use]
    pub fn new(choice: ThemeChoice) -> Self {
        Self { choice }
    }

    /// Applies the theme to a context: egui's default style with its own light and dark
    /// visuals, and the theme preference.
    pub fn apply(&self, ctx: &egui::Context) {
        ctx.set_style_of(egui::Theme::Light, egui::Theme::Light.default_style());
        ctx.set_style_of(egui::Theme::Dark, egui::Theme::Dark.default_style());
        ctx.set_theme(self.choice.preference());
    }
}

// ---------------------------------------------------------------------------------------
// Colours drawn by the program, derived from the visuals in force.

/// Background of the active camera view, which XMALab marks by a grey background
/// (`CameraViewWidget.cpp:293-313`): the fill egui gives a hovered button, a grey one step
/// away from the panel colour in either theme.
#[must_use]
pub fn active_camera_bg(v: &Visuals) -> Color32 {
    v.widgets.hovered.weak_bg_fill
}

/// Text on [`active_camera_bg`]: egui's strong text colour.
#[must_use]
pub fn active_camera_text(v: &Visuals) -> Color32 {
    v.strong_text_color()
}

/// Frame lines (group frames, list borders, plot frames).
#[must_use]
pub fn frame_line(v: &Visuals) -> Color32 {
    v.widgets.noninteractive.bg_stroke.color
}

/// Warning text (the workspace bar's "not optimized" label, red in XMALab).
#[must_use]
pub fn warning_text(v: &Visuals) -> Color32 {
    v.error_fg_color
}

/// Background of plots and lists (XMALab's white `Base`): egui's text-field background.
#[must_use]
pub fn plot_bg(v: &Visuals) -> Color32 {
    v.extreme_bg_color
}

/// Messages drawn over camera images (`GLCameraView.cpp:542`): red, as in XMALab. They are
/// drawn on the image, not on the window, so they do not follow the theme.
pub const VIEW_MESSAGE: Color32 = Color32::RED;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_round_trips_and_unknown_is_system() {
        for c in ThemeChoice::MENU_ORDER {
            assert_eq!(ThemeChoice::from_setting(c.as_setting()), c);
        }
        assert_eq!(ThemeChoice::from_setting("purple"), ThemeChoice::System);
    }

    #[test]
    fn drawn_colours_differ_between_themes() {
        let (l, d) = (Visuals::light(), Visuals::dark());
        for f in [active_camera_bg, active_camera_text, frame_line, plot_bg] {
            assert_ne!(f(&l), f(&d));
        }
    }

    #[test]
    fn apply_follows_egui_resolution() {
        let ctx = egui::Context::default();
        Theme::new(ThemeChoice::Light).apply(&ctx);
        assert!(!ctx.global_style().visuals.dark_mode);
        Theme::new(ThemeChoice::Dark).apply(&ctx);
        assert!(ctx.global_style().visuals.dark_mode);
        // System with no report from the platform: egui's fallback, the same for every part.
        Theme::new(ThemeChoice::System).apply(&ctx);
        assert_eq!(
            ctx.global_style().visuals.dark_mode,
            ctx.theme() == egui::Theme::Dark
        );
    }
}
