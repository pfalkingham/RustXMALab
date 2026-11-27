//! The graphical program RustXMALab (`rustxmalab`), as a library so that tests can drive it.
//!
//! Dependency rule: the only crate that may depend on the GUI toolkit
//! (`eframe`, `egui`, `wgpu`).
//!
//! The module map and how to add panels, actions and dialogs are in `ARCHITECTURE.md` next
//! to this crate's `Cargo.toml`.

pub mod actions;
pub mod app;
pub mod calibration_ui;
pub mod camera_view;
pub mod central;
/// The build date's arithmetic, compiled here for its tests (`build.rs` includes it).
#[cfg(test)]
mod civil_date;
pub mod dialogs;
pub mod editing;
pub mod frame_bar;
pub mod frames;
pub mod icons;
pub mod jobs;
pub mod links;
pub mod log;
pub mod menu;
pub mod meshes;
pub mod panels;
pub mod qt;
pub mod saving;
pub mod session;
pub mod settings;
pub mod shell;
pub mod startup;
pub mod state;
pub mod theme;
pub mod tracking_ui;
pub mod trial_images;
pub mod widgets;
pub mod workspace_bar;
pub mod world_view_draw;

pub use app::{Options, XmaApp};

/// The name the program gives itself (window title, Welcome dialog, licence button): it is
/// RustXMALab, not XMALab, so that it is not taken for an official version of the original
/// (parity ledger U173).
pub const APP_NAME: &str = "RustXMALab";

/// The application id of every window (Wayland `app_id`, X11 `WM_CLASS`), the same as the
/// program file's name and the Linux menu entry's `StartupWMClass` and `Icon`, so that a
/// desktop groups the windows under the menu entry (`packaging/linux/rustxmalab.desktop`).
pub const APP_ID: &str = "rustxmalab";

/// The eframe application: [`XmaApp`] plus saving the settings on exit.
pub struct EframeApp(pub XmaApp);

impl eframe::App for EframeApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        if self.0.frames.store().is_none()
            && let Some(rs) = frame.wgpu_render_state()
        {
            self.0.frames.set_render_state(rs, ui.ctx());
        }
        // The About dialog's graphics information (the C++ shows its OpenGL strings there).
        if self.0.dialogs.graphics.is_none()
            && let Some(rs) = frame.wgpu_render_state()
        {
            let info = rs.adapter.get_info();
            self.0.dialogs.set_graphics(
                &info.name,
                &format!("{:?}", info.backend),
                &info.driver,
                &info.driver_info,
            );
        }
        self.0.ui(ui);
        startup::after_frame(&self.0);
    }

    fn on_exit(&mut self) {
        self.0.shutdown();
    }
}

#[cfg(test)]
mod name_tests {
    /// The application id is `rustxmalab`, and the main window and the three kinds of
    /// separate window (floating panels, tool windows, Detection settings) all take it from
    /// [`super::APP_ID`].
    #[test]
    fn every_viewport_uses_the_one_application_id() {
        assert_eq!(super::APP_ID, "rustxmalab");
        assert_eq!(super::APP_NAME, "RustXMALab");
        for (file, src, id) in [
            ("main.rs", include_str!("main.rs"), "xma_app::APP_ID"),
            (
                "shell/floating.rs",
                include_str!("shell/floating.rs"),
                "crate::APP_ID",
            ),
            (
                "dialogs/windows.rs",
                include_str!("dialogs/windows.rs"),
                "crate::APP_ID",
            ),
            (
                "dialogs/detection_settings.rs",
                include_str!("dialogs/detection_settings.rs"),
                "crate::APP_ID",
            ),
        ] {
            let calls = src.matches(".with_app_id(").count();
            let ours = src.matches(&format!(".with_app_id({id})")).count();
            assert!(calls >= 1, "{file}: no application id");
            assert_eq!(calls, ours, "{file}: an application id other than {id}");
        }
    }
}
