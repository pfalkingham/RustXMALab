//! Per-machine interface settings.
//!
//! Ported from core/Settings.cpp, `Settings::Settings`: every entry registered there that
//! is not an algorithm setting (those travel with the project, `xma_model::AlgorithmSettings`,
//! parity ledger A10), plus what the C++ keeps in `QSettings` outside the registered list
//! (`lastDirectoryUsed`, `UIGeometry_XMALab`, `UIState_XMALab`) as the docking layout.
//!
//! Kept as JSON in the platform's configuration directory, under `RustXMALab/settings.json`
//! ([`config_dir`]). XMALab's own `XMALab.ini` is never read or written. The JSON keys are
//! the C++ setting names, so a reader can match them with `core/Settings.cpp`.
//!
//! Unknown keys are ignored and missing keys take the C++ default, so the file survives
//! version changes in both directions.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The environment variable that overrides the configuration directory (tests, portable
/// installs). The settings file is `<dir>/settings.json`.
pub const CONFIG_DIR_ENV: &str = "XMALAB_RS_CONFIG_DIR";

/// Maximum number of recent files (`Settings::addToRecentFiles`).
pub const MAX_RECENT_FILES: usize = 10;

/// The interface settings. Field names follow Rust conventions; the serialised names are the
/// C++ ones.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UiSettings {
    #[serde(rename = "RecentFiles")]
    pub recent_files: Vec<String>,

    // -- UI -------------------------------------------------------------------------------
    #[serde(rename = "ShowDetailView")]
    pub show_detail_view: bool,
    #[serde(rename = "ShowDisplayOptions")]
    pub show_display_options: bool,
    #[serde(rename = "ShowPlot")]
    pub show_plot: bool,
    #[serde(rename = "ShowEvents")]
    pub show_events: bool,
    #[serde(rename = "Show3DView")]
    pub show_3d_view: bool,
    /// Console panel shown (`View > Output`; hidden in XMALab 3's menu, shown here, parity
    /// ledger U174). Off by default.
    #[serde(rename = "Console")]
    pub console: bool,
    #[serde(rename = "LockUIPanes")]
    pub lock_ui_panes: bool,
    /// Version for which the welcome dialog was last shown.
    #[serde(rename = "WelcomeDialog")]
    pub welcome_dialog: String,
    /// `"light"`, `"dark"` or `"system"`.
    #[serde(rename = "Theme")]
    pub theme: String,

    // -- General --------------------------------------------------------------------------
    #[serde(rename = "AutoConfirmPendingChanges")]
    pub auto_confirm_pending_changes: bool,
    #[serde(rename = "ConfirmQuitXMALab")]
    pub confirm_quit_xmalab: bool,
    #[serde(rename = "CustomWorkspacePath")]
    pub custom_workspace_path: bool,
    #[serde(rename = "WorkspacePath")]
    pub workspace_path: String,
    /// Frames moved by the 5 and 6 keys.
    #[serde(rename = "FrameAdvance")]
    pub frame_advance: i32,
    #[serde(rename = "ExportAllEnabled")]
    pub export_all_enabled: bool,
    /// Do not search for a trial's missing video files after loading.
    #[serde(rename = "DisableImageSearch")]
    pub disable_image_search: bool,
    #[serde(rename = "RecomputeWhenSaving")]
    pub recompute_when_saving: bool,
    /// Minutes between autosaves of a modified project to its recovery file; 0 switches
    /// autosave off. Not a XMALab 3 setting (parity ledger U54); `crate::saving`.
    #[serde(rename = "AutosaveMinutes")]
    pub autosave_minutes: u32,

    // -- Undistortion ---------------------------------------------------------------------
    #[serde(rename = "UndistortNamingPattern")]
    pub undistort_naming_pattern: String,

    // -- Calibration (interface behaviour only) --------------------------------------------
    #[serde(rename = "AutoCalibAfterReference")]
    pub auto_calib_after_reference: bool,
    #[serde(rename = "ShowAdvancedCalibration")]
    pub show_advanced_calibration: bool,
    #[serde(rename = "HideWarningsDuringCalibration")]
    pub hide_warnings_during_calibration: bool,

    // -- Digitizing (interface behaviour only) ---------------------------------------------
    #[serde(rename = "CenterDetailView")]
    pub center_detail_view: bool,
    #[serde(rename = "AdvancedCrosshairDetailView")]
    pub advanced_crosshair_detail_view: bool,
    #[serde(rename = "DrawProjected2DpositionsForAllPoints")]
    pub draw_projected_2d_positions_for_all_points: bool,
    #[serde(rename = "Show3dPointDetailView")]
    pub show_3d_point_detail_view: bool,
    #[serde(rename = "ShowEpiLineDetailView")]
    pub show_epi_line_detail_view: bool,
    #[serde(rename = "ShowIDsInDetail")]
    pub show_ids_in_detail: bool,

    // -- Colours ----------------------------------------------------------------------------
    /// Marker-status colours as `#rrggbb`, as `QColor::name` writes them.
    #[serde(flatten)]
    pub colors: StatusColors,
    #[serde(rename = "ShowColoredMarkerIDs")]
    pub show_colored_marker_ids: bool,
    #[serde(rename = "ShowColoredMarkerCross")]
    pub show_colored_marker_cross: bool,

    // -- Detail view filter -----------------------------------------------------------------
    #[serde(rename = "VisualFilterEnabled")]
    pub visual_filter_enabled: bool,
    #[serde(rename = "VisualFilter_krad")]
    pub visual_filter_krad: i32,
    #[serde(rename = "VisualFilter_gsigma")]
    pub visual_filter_gsigma: f32,
    #[serde(rename = "VisualFilter_img_wt")]
    pub visual_filter_img_wt: f32,
    #[serde(rename = "VisualFilter_blur_wt")]
    pub visual_filter_blur_wt: f32,
    #[serde(rename = "VisualFilter_gamma")]
    pub visual_filter_gamma: f32,

    // -- Import and export dialogs (the radio buttons they remember) ----------------------
    #[serde(flatten)]
    pub import_export: ImportExportSettings,

    // -- Display Options (trial drawing) -----------------------------------------------------
    #[serde(rename = "TrialDrawHideAll")]
    pub trial_draw_hide_all: bool,
    #[serde(rename = "TrialDrawMarkers")]
    pub trial_draw_markers: bool,
    #[serde(rename = "TrialDrawMarkerIds")]
    pub trial_draw_marker_ids: bool,
    #[serde(rename = "TrialDrawEpipolar")]
    pub trial_draw_epipolar: bool,
    #[serde(rename = "TrialDrawRigidBodyConstellation")]
    pub trial_draw_rigid_body_constellation: bool,
    #[serde(rename = "TrialDrawRigidBodyMeshmodels")]
    pub trial_draw_rigid_body_meshmodels: bool,
    #[serde(rename = "TrialDrawFiltered")]
    pub trial_draw_filtered: bool,

    /// Points panel: "Show marker status".
    #[serde(rename = "ShowMarkerStates")]
    pub show_marker_states: bool,
    #[serde(rename = "EpipolarLinePrecision")]
    pub epipolar_line_precision: i32,

    // -- Kept by the C++ in QSettings outside the registered list ---------------------------
    /// `lastDirectoryUsed`: where file dialogs start.
    #[serde(rename = "lastDirectoryUsed")]
    pub last_directory_used: String,
    /// The docking layout and window geometry (`UIState_XMALab`, `UIGeometry_XMALab`);
    /// owned by [`crate::shell`]. `None` until the first save: the default arrangement.
    #[serde(rename = "UIState_XMALab")]
    pub layout: Option<serde_json::Value>,

    // -- Algorithm settings: this machine's defaults (not how XMALab 3 keeps them) -----------
    /// The settings that change computed results which the user set in the Settings dialog,
    /// by their C++ names (`xma_model::SETTING_NAMES`). They are this machine's defaults:
    /// [`UiSettings::algorithm_base`] gives them to a project that stores no value of its
    /// own (`xma_engine::OpenOptions::base_settings`). An open project keeps its own values
    /// (parity ledger A10). Settings never set here take XMALab's defaults. Entries with an
    /// unknown name or a value of the wrong kind are ignored.
    #[serde(rename = "AlgorithmDefaults")]
    pub algorithm_defaults: std::collections::BTreeMap<String, serde_json::Value>,
}

impl Default for UiSettings {
    /// The C++ defaults (`Settings::Settings`).
    fn default() -> Self {
        Self {
            recent_files: Vec::new(),
            show_detail_view: true,
            show_display_options: true,
            show_plot: true,
            show_events: true,
            show_3d_view: true,
            console: false,
            lock_ui_panes: false,
            welcome_dialog: "0".into(),
            theme: "system".into(),
            auto_confirm_pending_changes: false,
            confirm_quit_xmalab: true,
            custom_workspace_path: false,
            workspace_path: String::new(),
            frame_advance: 10,
            export_all_enabled: false,
            disable_image_search: false,
            recompute_when_saving: false,
            autosave_minutes: crate::saving::DEFAULT_AUTOSAVE_MINUTES,
            undistort_naming_pattern: "%NAMEBASE%%NUMBER%".into(),
            auto_calib_after_reference: true,
            show_advanced_calibration: false,
            hide_warnings_during_calibration: false,
            center_detail_view: false,
            advanced_crosshair_detail_view: false,
            draw_projected_2d_positions_for_all_points: false,
            show_3d_point_detail_view: false,
            show_epi_line_detail_view: true,
            show_ids_in_detail: false,
            colors: StatusColors::default(),
            show_colored_marker_ids: false,
            show_colored_marker_cross: false,
            visual_filter_enabled: true,
            visual_filter_krad: 17,
            visual_filter_gsigma: 10.0,
            visual_filter_img_wt: 3.6,
            visual_filter_blur_wt: -2.9,
            visual_filter_gamma: 0.4,
            import_export: ImportExportSettings::default(),
            trial_draw_hide_all: false,
            trial_draw_markers: true,
            trial_draw_marker_ids: false,
            trial_draw_epipolar: true,
            trial_draw_rigid_body_constellation: true,
            trial_draw_rigid_body_meshmodels: true,
            trial_draw_filtered: false,
            show_marker_states: false,
            epipolar_line_precision: 5,
            last_directory_used: String::new(),
            layout: None,
            algorithm_defaults: std::collections::BTreeMap::new(),
        }
    }
}

impl UiSettings {
    /// The setting that shows a panel with a View menu entry (`Show3DView`, `ShowPlot`,
    /// ...), which its close button turns off; `None` for the panels without one (Toolbox,
    /// Points, Progress).
    pub fn panel_setting_mut(&mut self, p: crate::panels::PanelId) -> Option<&mut bool> {
        use crate::panels::PanelId;
        match p {
            PanelId::DisplayOptions => Some(&mut self.show_display_options),
            PanelId::DetailedView => Some(&mut self.show_detail_view),
            PanelId::Plot => Some(&mut self.show_plot),
            PanelId::WorldView => Some(&mut self.show_3d_view),
            PanelId::Events => Some(&mut self.show_events),
            PanelId::Console => Some(&mut self.console),
            PanelId::Toolbox | PanelId::Points | PanelId::Progress => None,
        }
    }

    /// This machine's algorithm settings: XMALab's defaults with the values remembered in
    /// [`UiSettings::algorithm_defaults`]. What a project that stores no settings is opened
    /// with, and what the Settings dialog shows with no project open.
    #[must_use]
    pub fn algorithm_base(&self) -> xma_model::AlgorithmSettings {
        let mut s = xma_model::AlgorithmSettings::default();
        for (name, value) in &self.algorithm_defaults {
            if let Some(v) = setting_from_json(name, value) {
                // A value the settings refuse (out of range) is ignored like a bad entry.
                let _ = s.set(name, v);
            }
        }
        s
    }

    /// Remembers every value of `patch` as this machine's default.
    pub fn remember_algorithm_settings(&mut self, patch: &xma_model::SettingsPatch) {
        for (name, value) in &patch.values {
            let json = match *value {
                xma_model::SettingValue::Bool(b) => serde_json::Value::from(b),
                xma_model::SettingValue::Int(i) => serde_json::Value::from(i),
                xma_model::SettingValue::Float(f) => serde_json::Value::from(f),
            };
            self.algorithm_defaults.insert(name.clone(), json);
        }
    }
}

/// A remembered algorithm setting as the kind `xma_model::setting_kind` gives its name;
/// `None` for an unknown name or a value of another kind.
fn setting_from_json(name: &str, v: &serde_json::Value) -> Option<xma_model::SettingValue> {
    use xma_model::{SettingKind, SettingValue};
    match xma_model::setting_kind(name)? {
        SettingKind::Bool => v.as_bool().map(SettingValue::Bool),
        SettingKind::Int => v.as_i64().map(SettingValue::Int),
        SettingKind::Float => v.as_f64().map(SettingValue::Float),
    }
}

/// The marker-status and interpolation colours (core/Settings.cpp:110-124), as `#rrggbb`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StatusColors {
    #[serde(rename = "ColorUntrackable")]
    pub untrackable: String,
    #[serde(rename = "ColorInterpolated")]
    pub interpolated: String,
    #[serde(rename = "ColorManual")]
    pub manual: String,
    #[serde(rename = "ColorManualAndOpt")]
    pub manual_and_opt: String,
    #[serde(rename = "ColorSet")]
    pub set: String,
    #[serde(rename = "ColorSetAndOpt")]
    pub set_and_opt: String,
    #[serde(rename = "ColorTracked")]
    pub tracked: String,
    #[serde(rename = "ColorTrackedAndOpt")]
    pub tracked_and_opt: String,
    #[serde(rename = "ColorUndefined")]
    pub undefined: String,
    #[serde(rename = "ColorPredicted")]
    pub predicted: String,
    #[serde(rename = "ColorInterNone")]
    pub inter_none: String,
    #[serde(rename = "ColorInterRepeat")]
    pub inter_repeat: String,
    #[serde(rename = "ColorInterLinear")]
    pub inter_linear: String,
    #[serde(rename = "ColorInterCubic")]
    pub inter_cubic: String,
}

impl Default for StatusColors {
    fn default() -> Self {
        Self {
            untrackable: "#000000".into(),
            interpolated: "#ffcd32".into(),
            manual: "#c8001e".into(),
            manual_and_opt: "#ff002a".into(),
            set: "#00c864".into(),
            set_and_opt: "#00ff80".into(),
            tracked: "#0064c8".into(),
            tracked_and_opt: "#0080ff".into(),
            undefined: "#d2d2d2".into(),
            predicted: "#ff8c00".into(),
            inter_none: "#ffffff".into(),
            inter_repeat: "#e68080".into(),
            inter_linear: "#b8e6b8".into(),
            inter_cubic: "#aaaae6".into(),
        }
    }
}

/// Parses `#rrggbb` (as `QColor::name` writes it). Returns black for anything else, as
/// `QColor` gives an invalid (black) colour for a bad name.
#[must_use]
pub fn parse_color(name: &str) -> egui::Color32 {
    let hex = name.strip_prefix('#').unwrap_or(name);
    if hex.len() == 6
        && let Ok(v) = u32::from_str_radix(hex, 16)
    {
        return egui::Color32::from_rgb((v >> 16) as u8, (v >> 8) as u8, v as u8);
    }
    egui::Color32::BLACK
}

/// The radio-button choices the import and export dialogs remember
/// (core/Settings.cpp:163-203).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[allow(clippy::struct_excessive_bools)]
pub struct ImportExportSettings {
    #[serde(rename = "Import2DCount0")]
    pub import_2d_count0: bool,
    #[serde(rename = "Import2DCount1")]
    pub import_2d_count1: bool,
    #[serde(rename = "Import2DYDown")]
    pub import_2d_y_down: bool,
    #[serde(rename = "Import2DYUp")]
    pub import_2d_y_up: bool,
    #[serde(rename = "Import2DNoHeader")]
    pub import_2d_no_header: bool,
    #[serde(rename = "Import2DHeader")]
    pub import_2d_header: bool,
    #[serde(rename = "Import2DDistorted")]
    pub import_2d_distorted: bool,
    #[serde(rename = "Import2DUndistorted")]
    pub import_2d_undistorted: bool,
    #[serde(rename = "Import2DNoCols")]
    pub import_2d_no_cols: bool,
    #[serde(rename = "Import2DOffsetCols")]
    pub import_2d_offset_cols: bool,
    #[serde(rename = "ImportStatusSet")]
    pub import_status_set: bool,
    #[serde(rename = "ImportStatusTracked")]
    pub import_status_tracked: bool,
    #[serde(rename = "Export2DMulti")]
    pub export_2d_multi: bool,
    #[serde(rename = "Export2DSingle")]
    pub export_2d_single: bool,
    #[serde(rename = "Export2DCount0")]
    pub export_2d_count0: bool,
    #[serde(rename = "Export2DCount1")]
    pub export_2d_count1: bool,
    #[serde(rename = "Export2DYDown")]
    pub export_2d_y_down: bool,
    #[serde(rename = "Export2DYUp")]
    pub export_2d_y_up: bool,
    #[serde(rename = "Export2DNoHeader")]
    pub export_2d_no_header: bool,
    #[serde(rename = "Export2DHeader")]
    pub export_2d_header: bool,
    #[serde(rename = "Export2DDistorted")]
    pub export_2d_distorted: bool,
    #[serde(rename = "Export2DUndistorted")]
    pub export_2d_undistorted: bool,
    #[serde(rename = "Export2DNoCols")]
    pub export_2d_no_cols: bool,
    #[serde(rename = "Export2DOffsetCols")]
    pub export_2d_offset_cols: bool,
    #[serde(rename = "Export3DMulti")]
    pub export_3d_multi: bool,
    #[serde(rename = "Export3DSingle")]
    pub export_3d_single: bool,
    #[serde(rename = "Export3DNoHeader")]
    pub export_3d_no_header: bool,
    #[serde(rename = "Export3DHeader")]
    pub export_3d_header: bool,
    #[serde(rename = "Export3DNoCols")]
    pub export_3d_no_cols: bool,
    #[serde(rename = "Export3DOffsetCols")]
    pub export_3d_offset_cols: bool,
    #[serde(rename = "ExportTransMulti")]
    pub export_trans_multi: bool,
    #[serde(rename = "ExportTransSingle")]
    pub export_trans_single: bool,
    #[serde(rename = "ExportTransNoHeader")]
    pub export_trans_no_header: bool,
    #[serde(rename = "ExportTransHeader")]
    pub export_trans_header: bool,
    #[serde(rename = "ExportTransUnfiltered")]
    pub export_trans_unfiltered: bool,
    #[serde(rename = "ExportTransFiltered")]
    pub export_trans_filtered: bool,
    #[serde(rename = "ExportTransNoCols")]
    pub export_trans_no_cols: bool,
    #[serde(rename = "ExportTransOffsetCols")]
    pub export_trans_offset_cols: bool,
}

impl Default for ImportExportSettings {
    fn default() -> Self {
        Self {
            import_2d_count0: true,
            import_2d_count1: false,
            import_2d_y_down: true,
            import_2d_y_up: false,
            import_2d_no_header: false,
            import_2d_header: true,
            import_2d_distorted: true,
            import_2d_undistorted: false,
            import_2d_no_cols: true,
            import_2d_offset_cols: false,
            import_status_set: true,
            import_status_tracked: false,
            export_2d_multi: true,
            export_2d_single: false,
            export_2d_count0: true,
            export_2d_count1: false,
            export_2d_y_down: true,
            export_2d_y_up: false,
            export_2d_no_header: false,
            export_2d_header: true,
            export_2d_distorted: true,
            export_2d_undistorted: false,
            export_2d_no_cols: true,
            export_2d_offset_cols: false,
            export_3d_multi: true,
            export_3d_single: false,
            export_3d_no_header: false,
            export_3d_header: true,
            export_3d_no_cols: true,
            export_3d_offset_cols: false,
            export_trans_multi: true,
            export_trans_single: false,
            export_trans_no_header: false,
            export_trans_header: true,
            export_trans_unfiltered: false,
            export_trans_filtered: true,
            export_trans_no_cols: true,
            export_trans_offset_cols: false,
        }
    }
}

impl UiSettings {
    /// Puts a file at the top of the recent-file list.
    ///
    /// Ported from core/Settings.cpp, Settings::addToRecentFiles: insert at the front,
    /// remove duplicates (keeping the first), and drop the last entry when there are more
    /// than ten. (The C++ drops one entry only, so a list that was already longer stays
    /// longer; it never is.)
    pub fn add_to_recent_files(&mut self, file: &str) {
        self.recent_files.insert(0, file.to_owned());
        let mut seen = std::collections::HashSet::new();
        self.recent_files.retain(|f| seen.insert(f.clone()));
        if self.recent_files.len() > MAX_RECENT_FILES {
            self.recent_files.pop();
        }
    }

    /// Removes recent files that no longer exist.
    ///
    /// Ported from core/Settings.cpp, Settings::checkRecenFiles.
    pub fn check_recent_files(&mut self) {
        self.recent_files.retain(|f| Path::new(f).is_file());
    }

    /// The folder XMA Portal data is extracted to when "Use custom workspace to save data
    /// imported from XMA Portal" is on (`CustomWorkspacePath`, `WorkspacePath`); `None` when
    /// it is off or its folder is empty (`ProjectFileIO::loadXMAPortalTrial`).
    #[must_use]
    pub fn portal_workspace(&self) -> Option<PathBuf> {
        (self.custom_workspace_path && !self.workspace_path.is_empty())
            .then(|| PathBuf::from(&self.workspace_path))
    }

    /// Remembers the folder of `path` for the next file dialog.
    ///
    /// Ported from core/Settings.cpp, Settings::setLastUsedDirectory (`directory` false).
    pub fn set_last_used_directory_from_file(&mut self, path: &Path) {
        if let Some(dir) = path.parent() {
            let dir = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
            self.last_directory_used = dir.to_string_lossy().into_owned();
        }
    }

    /// Reads the settings file. A missing file gives the defaults; an unreadable one gives
    /// the defaults and an error message for the log.
    ///
    /// A file that is a JSON object is read key by key: each key whose value the settings
    /// accept is kept, each that they refuse (a wrong type, a number out of range) takes its
    /// default and is named in the message, so one bad value does not reset the others.
    /// Whenever something is ignored, or the file is not JSON, the file is first copied to
    /// `settings.json.bad` beside it (replacing an older copy), since the next save writes
    /// over it.
    #[must_use]
    pub fn load(path: &Path) -> (Self, Option<String>) {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (Self::default(), None),
            Err(e) => {
                return (
                    Self::default(),
                    Some(format!("Settings file {} unreadable: {e}", path.display())),
                );
            }
        };
        let value = match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(serde_json::Value::Object(map)) => map,
            Ok(_) => {
                let kept = keep_bad_copy(path);
                return (
                    Self::default(),
                    Some(format!(
                        "Settings file {} ignored: not a JSON object{kept}",
                        path.display()
                    )),
                );
            }
            Err(e) => {
                let kept = keep_bad_copy(path);
                return (
                    Self::default(),
                    Some(format!(
                        "Settings file {} ignored: {e}{kept}",
                        path.display()
                    )),
                );
            }
        };
        // The whole file at once when it is valid (the usual case).
        if let Ok(mut s) = serde_json::from_value::<Self>(serde_json::Value::Object(value.clone()))
        {
            s.drop_removed_settings();
            return (s, None);
        }
        let Ok(serde_json::Value::Object(mut merged)) = serde_json::to_value(Self::default())
        else {
            return (Self::default(), None);
        };
        let mut ignored = Vec::new();
        for (key, v) in value {
            let old = merged.insert(key.clone(), v);
            if serde_json::from_value::<Self>(serde_json::Value::Object(merged.clone())).is_err() {
                match old {
                    Some(o) => merged.insert(key.clone(), o),
                    None => merged.remove(&key),
                };
                ignored.push(key);
            }
        }
        let mut s =
            serde_json::from_value::<Self>(serde_json::Value::Object(merged)).unwrap_or_default();
        s.drop_removed_settings();
        let kept = keep_bad_copy(path);
        (
            s,
            Some(format!(
                "Settings file {}: the values of {} are not valid and take their defaults{kept}",
                path.display(),
                ignored.join(", ")
            )),
        )
    }

    /// Forgets the remembered algorithm settings of methods this program no longer has
    /// ([`REMOVED_ALGORITHM_SETTINGS`]), silently, so that the next save does not write them.
    fn drop_removed_settings(&mut self) {
        self.algorithm_defaults
            .retain(|name, _| !REMOVED_ALGORITHM_SETTINGS.contains(&name.as_str()));
    }

    /// Writes the settings file, creating its folder.
    ///
    /// # Errors
    /// When the folder or file cannot be written.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let text = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        // Write to a temporary file first so a crash cannot leave half a file; its name has
        // the process id, so that two running copies of the program do not share it.
        let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
        // On failure the temporary file is removed and the original error returned.
        std::fs::write(&tmp, text)
            .and_then(|()| std::fs::rename(&tmp, path))
            .inspect_err(|_| {
                let _ = std::fs::remove_file(&tmp);
            })
    }
}

/// Algorithm settings that a settings file written by an earlier version can hold and this
/// program no longer has: the five thresholds of merge-aware tracking, removed with it
/// (parity ledger U171, A226). [`UiSettings::load`] drops them without a message.
pub const REMOVED_ALGORITHM_SETTINGS: [&str; 5] = [
    "MergeEpipolarThreshold",
    "MergeAuditThreshold",
    "MergeMaxPendingFrames",
    "MergeMaxHoldFrames",
    "MergeMaxOverlapFrames",
];

/// Copies a settings file that could not be read whole to `<name>.bad` beside it; returns
/// the words for the message.
fn keep_bad_copy(path: &Path) -> String {
    let mut bad = path.as_os_str().to_owned();
    bad.push(".bad");
    let bad = PathBuf::from(bad);
    match std::fs::copy(path, &bad) {
        Ok(_) => format!(" (the file is kept as {})", bad.display()),
        Err(e) => format!(" (could not keep a copy as {}: {e})", bad.display()),
    }
}

/// The folder that holds `settings.json`: `$XMALAB_RS_CONFIG_DIR` if set, otherwise
/// `RustXMALab/` in the platform's configuration directory:
///
/// - Linux and other Unix: `$XDG_CONFIG_HOME`, else `$HOME/.config`;
/// - macOS: `$HOME/Library/Application Support`;
/// - Windows: `%APPDATA%`.
///
/// `None` when none of these variables is set.
#[must_use]
pub fn config_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os(CONFIG_DIR_ENV).filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(dir));
    }
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
    }?;
    Some(base.join("RustXMALab"))
}

/// `<config_dir>/settings.json`.
#[must_use]
pub fn settings_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("settings.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip_through_json_with_cpp_names() {
        let s = UiSettings::default();
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["ShowDetailView"], true);
        assert_eq!(v["FrameAdvance"], 10);
        assert_eq!(v["ColorManual"], "#c8001e");
        assert_eq!(v["Export2DHeader"], true);
        let back: UiSettings = serde_json::from_value(v).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn missing_keys_take_defaults() {
        let s: UiSettings = serde_json::from_str(r#"{"Console": true, "Unknown": 3}"#).unwrap();
        assert!(s.console);
        assert_eq!(s.frame_advance, 10);
    }

    #[test]
    fn recent_files_like_cpp() {
        let mut s = UiSettings::default();
        for i in 0..12 {
            s.add_to_recent_files(&format!("f{i}"));
        }
        assert_eq!(s.recent_files.len(), 10);
        assert_eq!(s.recent_files[0], "f11");
        s.add_to_recent_files("f5");
        assert_eq!(s.recent_files[0], "f5");
        assert_eq!(s.recent_files.iter().filter(|f| *f == "f5").count(), 1);
    }

    #[test]
    fn algorithm_defaults_round_trip_and_ignore_bad_entries() {
        use xma_model::{SettingValue, SettingsPatch};
        let mut s = UiSettings::default();
        assert_eq!(s.algorithm_base(), xma_model::AlgorithmSettings::default());
        let mut p = SettingsPatch::default();
        p.set("TriangulationMethod", SettingValue::Int(3));
        p.set("Filter3DPoints", SettingValue::Bool(true));
        p.set("MaximumReprojectionError", SettingValue::Float(2.5));
        s.remember_algorithm_settings(&p);
        let text = serde_json::to_string(&s).unwrap();
        let mut back: UiSettings = serde_json::from_str(&text).unwrap();
        let base = back.algorithm_base();
        assert_eq!(base.triangulation_method, 3);
        assert!(base.filter_3d_points);
        assert_eq!(base.maximum_reprojection_error, 2.5);
        // Unknown names, wrong kinds and refused values are ignored.
        back.algorithm_defaults
            .insert("NoSuchSetting".into(), serde_json::Value::from(1));
        back.algorithm_defaults
            .insert("OptimizeRigidBody".into(), serde_json::Value::from(7));
        back.algorithm_defaults
            .insert("TriangulationMethod".into(), serde_json::Value::from(9));
        let base = back.algorithm_base();
        assert!(base.optimize_rigid_body);
        assert_eq!(base.triangulation_method, 1);
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xma_settings_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Review of 20 November 2026 (app-core 7): one wrong value keeps every other key.
    #[test]
    fn one_bad_value_keeps_the_other_keys() {
        let dir = scratch("one_bad");
        let p = dir.join("settings.json");
        let text = r##"{"FrameAdvance": "x", "RecentFiles": ["/a", "/b"], "Console": true,
            "AutosaveMinutes": -1, "ColorManual": "#010203", "Unknown": 3}"##;
        std::fs::write(&p, text).unwrap();
        let (s, msg) = UiSettings::load(&p);
        assert_eq!(s.recent_files, vec!["/a".to_owned(), "/b".to_owned()]);
        assert!(s.console);
        assert_eq!(s.colors.manual, "#010203");
        let d = UiSettings::default();
        assert_eq!(s.frame_advance, d.frame_advance);
        assert_eq!(s.autosave_minutes, d.autosave_minutes);
        let msg = msg.expect("a message");
        assert!(msg.contains("FrameAdvance"), "{msg}");
        assert!(msg.contains("AutosaveMinutes"), "{msg}");
        assert!(!msg.contains("RecentFiles"), "{msg}");
        assert!(!msg.contains("Unknown"), "{msg}");
        // The file as it was, beside it.
        let bad = dir.join("settings.json.bad");
        assert_eq!(std::fs::read_to_string(&bad).unwrap(), text);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_that_is_not_json_gives_defaults_and_a_bad_copy() {
        let dir = scratch("not_json");
        let p = dir.join("settings.json");
        let bad = dir.join("settings.json.bad");
        std::fs::write(&bad, "an older copy").unwrap();
        std::fs::write(&p, "{ not json").unwrap();
        let (s, msg) = UiSettings::load(&p);
        assert_eq!(s, UiSettings::default());
        assert!(msg.is_some());
        assert_eq!(std::fs::read_to_string(&bad).unwrap(), "{ not json");
        // A valid file: no message, no copy made.
        let _ = std::fs::remove_file(&bad);
        UiSettings::default().save(&p).unwrap();
        let (s, msg) = UiSettings::load(&p);
        assert_eq!(s, UiSettings::default());
        assert!(msg.is_none());
        assert!(!bad.exists());
        // A missing file: the defaults, nothing said.
        let (_, msg) = UiSettings::load(&dir.join("none.json"));
        assert!(msg.is_none());
        // `save` leaves no temporary file.
        let names: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, vec![std::ffi::OsString::from("settings.json")]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Review of 20 November 2026 (cross-cutting 7): a failed save removes its temporary
    /// file and returns the error of the step that failed.
    #[test]
    fn failed_save_leaves_no_temporary_file() {
        let dir = scratch("failed_save");
        // The rename fails: the settings path is a folder that is not empty.
        let p = dir.join("settings.json");
        std::fs::create_dir_all(p.join("inside")).unwrap();
        let err = UiSettings::default()
            .save(&p)
            .expect_err("the rename fails");
        assert_ne!(err.kind(), std::io::ErrorKind::NotFound, "{err}");
        let tmp = p.with_extension(format!("json.{}.tmp", std::process::id()));
        assert!(!tmp.exists(), "temporary file left at {}", tmp.display());
        assert!(p.join("inside").is_dir());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn colors_parse() {
        assert_eq!(parse_color("#c8001e"), egui::Color32::from_rgb(200, 0, 30));
        assert_eq!(parse_color("bad"), egui::Color32::BLACK);
    }
}
