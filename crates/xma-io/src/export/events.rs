//! Events: File > Export > Events, and the event files of a saved project.
//!
//! Ported from core/EventData.cpp, `EventData::saveData` (`:80-91`), and
//! `ui/MainWindow.cpp`, `MainWindow::on_actionExportEvents_triggered` (`:1856-1869`).

use xma_model::{CppCompat, Event, Trial};

use super::ExportFile;

/// One event's file: one line per frame, `0` or `1` (a `bool` through `std::ostream`), no
/// header.
///
/// Ported from core/EventData.cpp, EventData::saveData. The same text is written into the
/// project as `<trial>/data/<event name>.csv` (ui/ProjectFileIO.cpp:218).
#[must_use]
pub fn event_file(event: &Event) -> String {
    let mut out = String::with_capacity(event.data.len() * 2);
    for &v in &event.data {
        out.push(if v { '1' } else { '0' });
        out.push('\n');
    }
    out
}

/// One file per event, named `<event name>.csv`, in event order.
///
/// Ported from ui/MainWindow.cpp, MainWindow::on_actionExportEvents_triggered. In XMALab 3
/// (`compat.export_details`, ledger F25) an empty name gives `.csv` (the guard in
/// `saveData` tests the whole path, which is never empty), and events with the same name
/// share a file: written in order, the last one wins. Fixed: an empty name is replaced by
/// `Event%03d` (1-based), and a name already used gets `_2`, `_3`, ... before `.csv`, so
/// every event has its own file. Names are compared ignoring case (Unicode lowercase), as
/// the default file systems of Windows and macOS do, so `Walk` and `walk` do not share a
/// file there either.
#[must_use]
pub fn events(trial: &Trial, compat: CppCompat) -> Vec<ExportFile> {
    let mut used = std::collections::BTreeSet::new();
    trial
        .events
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let name = if compat.export_details {
                e.name.clone()
            } else {
                let base = if e.name.is_empty() {
                    format!("Event{}", super::padded(i))
                } else {
                    e.name.clone()
                };
                let mut name = base.clone();
                let mut n = 2;
                while used.contains(&name.to_lowercase()) {
                    name = format!("{base}_{n}");
                    n += 1;
                }
                used.insert(name.to_lowercase());
                name
            };
            ExportFile::new(format!("{name}.csv"), event_file(e))
        })
        .collect()
}
