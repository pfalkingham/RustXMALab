//! The menu bar: File, Edit, View, with the items, order and separators of XMALab 3.
//!
//! Ported from `MainWindow.ui` (`menubar`) and ui/MainWindow.cpp (ui-map "Menu bar"):
//!
//! - File > Recent Files is rebuilt from the `RecentFiles` setting
//!   (`MainWindow::updateRecentFiles`): one item per path, the full path as text.
//! - View > Theme (Light, Dark, Follow System, exclusive) is appended after Lock UI Panes
//!   (`MainWindow::createThemeMenu`).
//! - View > Reset layout, after a separator at the end of the View menu, is an addition
//!   (parity ledger U165): the default panel layout and main-window geometry at once, as
//!   `--reset-layout` gives on a fresh start.
//! - View > Output (`actionConsole`) is shown, a checkable entry like the other panel
//!   toggles, enabled in every workspace and with no project, so the Console panel can be
//!   opened from the window (parity ledger U174; XMALab 3 hides it at start-up and never shows
//!   it, `MainWindow.cpp:131`). File > Export > XROMM-VR stays hidden (`:132`); File > Export >
//!   All Trials for External is shown only with the `ExportAllEnabled` setting (`:133`).
//! - File > Export > Blender scene, after a separator at the end of the exports (before
//!   the hidden XROMM-VR and the setting-dependent All Trials for External), is new (parity
//!   ledger U176).
//! - Separators collapse as Qt's do (`QMenu::separatorsCollapsible`, on by default): no
//!   separator at the start or end of a menu, never two in a row, so the separators around a
//!   hidden item do not double up.
//! - The roughly 45 actions of `MainWindow.ui` that no menu contains are left out
//!   (ui-map "Actions not in any menu").

use crate::actions::{Action, ActionContext, ActionQueue};
use crate::settings::UiSettings;
use crate::theme::ThemeChoice;

/// An entry of a menu.
#[derive(Debug)]
pub enum MenuItem {
    Action(Action),
    Separator,
    Submenu(&'static str, &'static [MenuItem]),
    /// File > Recent Files, built from the settings.
    RecentFiles,
    /// View > Theme.
    Theme,
}

use Action as A;
use MenuItem::{Action as Act, Separator as Sep};

static EXPORT: [MenuItem; 18] = [
    Act(A::Export3DPoints),
    Act(A::Export2DPoints),
    Act(A::ExportRigidBodyTransformations),
    Act(A::ExportEvents),
    Act(A::ExportMarkerToMarkerDistances),
    Act(A::ExportPrecisionInfo),
    Act(A::ExportReprojectionErrors),
    Sep,
    Act(A::ExportUndistortedTrialImages),
    Act(A::ExportMayaCams2),
    Sep,
    Act(A::ExportXRayProjectMatlab),
    // Not in XMALab 3 (parity ledger U176): after the exports, before the hidden and
    // setting-dependent entries.
    Sep,
    Act(A::ExportBlenderScene),
    Sep,
    Act(A::ExportXrommVr),
    Sep,
    Act(A::ExportAllTrialsForExternal),
];

static IMPORT: [MenuItem; 2] = [Act(A::Import2DPoints), Act(A::ImportTrial)];

static FILE: [MenuItem; 17] = [
    Act(A::NewProject),
    Act(A::NewDatasetExternalCalibration),
    Act(A::NewTrialWithoutCalibration),
    Act(A::LoadProject),
    Act(A::LoadProjectWithDifferentCalibration),
    Act(A::CloseProject),
    MenuItem::RecentFiles,
    Sep,
    Act(A::SaveProject),
    Act(A::SaveProjectAs),
    Act(A::SaveSubdatasetAs),
    Sep,
    MenuItem::Submenu("Export", &EXPORT),
    MenuItem::Submenu("Import", &IMPORT),
    Sep,
    Act(A::UndistortSequence),
    Act(A::DeleteMultipleTrials),
];

/// Edit: Undo as in XMALab 3, and Redo, an addition (parity ledger U52).
static EDIT: [MenuItem; 2] = [Act(A::Undo), Act(A::Redo)];

static VIEW: [MenuItem; 18] = [
    Act(A::Settings),
    Act(A::About),
    Act(A::Help),
    Sep,
    Act(A::ToggleDisplayOptions),
    Act(A::ToggleDetailedView),
    Act(A::TogglePlot),
    Act(A::Toggle3DWorldView),
    Act(A::ToggleEvents),
    Act(A::ProjectMetadata),
    Act(A::ProjectOverview),
    Act(A::ToggleDetectionSettings),
    Act(A::ToggleConsole),
    Sep,
    Act(A::LockUiPanes),
    MenuItem::Theme,
    // Not in XMALab 3 (parity ledger U165).
    Sep,
    Act(A::ResetLayout),
];

/// The menus in order.
pub static MENUS: [(&str, &[MenuItem]); 3] = [("File", &FILE), ("Edit", &EDIT), ("View", &VIEW)];

/// Is a menu action shown at all (not just enabled)?
#[must_use]
pub fn action_visible(a: &Action, settings: &UiSettings) -> bool {
    match a {
        // View > Output (`ToggleConsole`) is shown, unlike XMALab 3 (parity ledger U174).
        Action::ExportXrommVr => false,
        Action::ExportAllTrialsForExternal => settings.export_all_enabled,
        _ => true,
    }
}

/// A menu's entries as they appear: hidden items removed, separators collapsed.
#[must_use]
pub fn visible_items<'a>(items: &'a [MenuItem], settings: &UiSettings) -> Vec<&'a MenuItem> {
    let mut out: Vec<&MenuItem> = Vec::new();
    for it in items {
        match it {
            MenuItem::Separator => {
                if out
                    .last()
                    .is_some_and(|l| !matches!(l, MenuItem::Separator))
                {
                    out.push(it);
                }
            }
            MenuItem::Action(a) if !action_visible(a, settings) => {}
            _ => out.push(it),
        }
    }
    while out.last().is_some_and(|l| matches!(l, MenuItem::Separator)) {
        out.pop();
    }
    out
}

/// The menu structure as text, one line per entry: `Menu/Sub/Label [Shortcut]`, `Menu/---`
/// for a separator. Used by the tests to compare with ui-map.md.
#[must_use]
pub fn menu_lines(settings: &UiSettings) -> Vec<String> {
    fn walk(prefix: &str, items: &[MenuItem], settings: &UiSettings, out: &mut Vec<String>) {
        for it in visible_items(items, settings) {
            match it {
                MenuItem::Action(a) => {
                    let mut line = format!("{prefix}/{}", a.info().label);
                    if let Some(s) = a.shortcut_text() {
                        line.push_str(&format!(" [{s}]"));
                    }
                    out.push(line);
                }
                MenuItem::Separator => out.push(format!("{prefix}/---")),
                MenuItem::Submenu(name, sub) => {
                    out.push(format!("{prefix}/{name} >"));
                    walk(&format!("{prefix}/{name}"), sub, settings, out);
                }
                MenuItem::RecentFiles => {
                    out.push(format!("{prefix}/Recent Files >"));
                    for f in &settings.recent_files {
                        out.push(format!("{prefix}/Recent Files/{f}"));
                    }
                }
                MenuItem::Theme => {
                    out.push(format!("{prefix}/Theme >"));
                    for c in ThemeChoice::MENU_ORDER {
                        out.push(format!("{prefix}/Theme/{}", c.label()));
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    for (name, items) in &MENUS {
        walk(name, items, settings, &mut out);
    }
    out
}

/// Draws the menu bar and queues the chosen action.
pub fn menu_bar(ui: &mut egui::Ui, acx: &ActionContext<'_>, queue: &mut ActionQueue) {
    // Qt's menu bar is about 22 points high.
    ui.spacing_mut().interact_size.y = 20.0;
    egui::MenuBar::new().ui(ui, |ui| {
        for (name, items) in &MENUS {
            ui.menu_button(*name, |ui| {
                draw_items(ui, items, acx, queue);
            });
        }
    });
}

fn draw_items(
    ui: &mut egui::Ui,
    items: &[MenuItem],
    acx: &ActionContext<'_>,
    queue: &mut ActionQueue,
) {
    for it in visible_items(items, acx.settings) {
        match it {
            MenuItem::Separator => {
                ui.separator();
            }
            MenuItem::Action(a) => draw_action(ui, a, acx, queue),
            MenuItem::Submenu(name, sub) => {
                ui.menu_button(*name, |ui| draw_items(ui, sub, acx, queue));
            }
            MenuItem::RecentFiles => {
                ui.menu_button("Recent Files", |ui| {
                    for (i, f) in acx.settings.recent_files.iter().enumerate() {
                        let a = Action::OpenRecentFile(i);
                        if ui
                            .add_enabled(a.enabled(acx), egui::Button::new(f.as_str()))
                            .clicked()
                        {
                            queue.push(a);
                            ui.close();
                        }
                    }
                });
            }
            MenuItem::Theme => {
                ui.menu_button("Theme", |ui| {
                    for c in ThemeChoice::MENU_ORDER {
                        let a = Action::Theme(c);
                        let checked = a.checked(acx).unwrap_or(false);
                        if ui.radio(checked, c.label()).clicked() {
                            queue.push(a);
                            ui.close();
                        }
                    }
                });
            }
        }
    }
}

fn draw_action(ui: &mut egui::Ui, a: &Action, acx: &ActionContext<'_>, queue: &mut ActionQueue) {
    let label = a.menu_label(acx);
    let enabled = a.enabled(acx);
    let clicked = if let Some(checked) = a.checked(acx) {
        let mut c = checked;
        ui.add_enabled(enabled, egui::Checkbox::new(&mut c, label))
            .clicked()
    } else {
        let mut b = egui::Button::new(label);
        if let Some(s) = a.shortcut_text() {
            b = b.shortcut_text(s);
        }
        ui.add_enabled(enabled, b).clicked()
    };
    if clicked {
        queue.push(a.clone());
        ui.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separators_collapse_around_hidden_items() {
        let s = UiSettings::default();
        let lines = menu_lines(&s);
        assert!(!lines.iter().any(|l| l.contains("XROMM-VR")));
        assert!(!lines.iter().any(|l| l.contains("All Trials for External")));
        // View > Output is shown (parity ledger U174), after DetectionSettings.
        let i = lines
            .iter()
            .position(|l| l == "View/Output")
            .expect("View/Output");
        assert_eq!(lines[i - 1], "View/DetectionSettings");
        // No two separators in a row, none at the end of a menu.
        for w in lines.windows(2) {
            assert!(!(w[0].ends_with("/---") && w[1].ends_with("/---")), "{w:?}");
        }
        let export: Vec<_> = lines
            .iter()
            .filter(|l| l.starts_with("File/Export/"))
            .collect();
        // Blender scene (new, ledger U176) ends the shown exports, after a separator.
        assert_eq!(export.last().unwrap().as_str(), "File/Export/Blender scene");
        assert_eq!(
            export[export.len() - 3].as_str(),
            "File/Export/XRayProject Matlab Format"
        );
        assert_eq!(export[export.len() - 2].as_str(), "File/Export/---");
    }

    #[test]
    fn export_all_setting_shows_the_item() {
        let s = UiSettings {
            export_all_enabled: true,
            ..UiSettings::default()
        };
        let lines = menu_lines(&s);
        assert!(lines.contains(&"File/Export/All Trials for External".to_owned()));
    }
}
