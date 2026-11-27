//! `PointImportExportDialog`: the format options of File > Import > 2D Points and of the
//! three exports that open the same form in XMALab 3 (File > Export > 2D Points, 3D Points,
//! Rigid Body Transformations).
//!
//! Ported control by control from `PointImportExportDialog.ui` and
//! ui/PointImportExportDialog.cpp. Each kind shows its frames (the constructor's `hide()`
//! calls) and its title; each radio button writes its setting at once (`Import2D*`,
//! `Export2D*`, `Export3D*`, `ExportTrans*`, `ImportStatus*`), even when the dialog is then
//! cancelled, as in XMALab. OK returns the kind; the caller reads the settings.

use egui::Ui;

use super::edit_forms::FormOutcome;
use super::{DialogResult, form};
use crate::settings::{ImportExportSettings, UiSettings};

/// Which use of the form (`ImportExportType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportExportKind {
    /// `IMPORT2D`: File > Import > 2D Points (E66).
    Import2d,
    /// `EXPORT2D`.
    Export2d,
    /// `EXPORT3D`.
    Export3d,
    /// `EXPORTTRANS`.
    ExportTransformations,
}

impl ImportExportKind {
    /// The window title. `EXPORTTRANS` sets none in XMALab, so the form's "Dialog" shows.
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            Self::Import2d => "Import 2D points",
            Self::Export2d => "Export 2D points",
            Self::Export3d => "Export 3D points",
            Self::ExportTransformations => "Dialog",
        }
    }
}

/// The dialog's state: only its kind; the choices are settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointImportExportDialog {
    pub kind: ImportExportKind,
}

/// One frame of two radio buttons: their texts and the two settings they write.
struct Pair {
    left: &'static str,
    right: &'static str,
    get: fn(&ImportExportSettings) -> (bool, bool),
    set: fn(&mut ImportExportSettings, bool),
}

/// The getter and setter of one pair of settings.
type Accessors = (
    fn(&ImportExportSettings) -> (bool, bool),
    fn(&mut ImportExportSettings, bool),
);

/// Sets one pair of settings: `right` chosen or not.
macro_rules! pair_fns {
    ($l:ident, $r:ident) => {
        (
            (|s: &ImportExportSettings| (s.$l, s.$r)) as fn(&ImportExportSettings) -> (bool, bool),
            (|s: &mut ImportExportSettings, right: bool| {
                s.$l = !right;
                s.$r = right;
            }) as fn(&mut ImportExportSettings, bool),
        )
    };
}

impl PointImportExportDialog {
    #[must_use]
    pub fn new(kind: ImportExportKind) -> Self {
        Self { kind }
    }

    /// The frames shown for this kind, in form order (rows 0 to 8).
    fn frames(&self) -> Vec<Pair> {
        use ImportExportKind as K;
        let k = self.kind;
        let mut v = Vec::new();
        let mk = |left, right, (get, set): Accessors| Pair {
            left,
            right,
            get,
            set,
        };
        // frame_File: single files / combined file (hidden for import).
        match k {
            K::Import2d => {}
            K::Export2d => v.push(mk(
                "single files",
                "combined file",
                pair_fns!(export_2d_single, export_2d_multi),
            )),
            K::Export3d => v.push(mk(
                "single files",
                "combined file",
                pair_fns!(export_3d_single, export_3d_multi),
            )),
            K::ExportTransformations => v.push(mk(
                "single files",
                "combined file",
                pair_fns!(export_trans_single, export_trans_multi),
            )),
        }
        // frame_Header.
        let header = match k {
            K::Import2d => pair_fns!(import_2d_no_header, import_2d_header),
            K::Export2d => pair_fns!(export_2d_no_header, export_2d_header),
            K::Export3d => pair_fns!(export_3d_no_header, export_3d_header),
            K::ExportTransformations => pair_fns!(export_trans_no_header, export_trans_header),
        };
        v.push(mk("no header row", "header Row", header));
        // frame_Distorted, frame_Start, frame_YAxis: 2D only.
        match k {
            K::Import2d => {
                v.push(mk(
                    "distorted coordinates",
                    "undistorted coordinates",
                    pair_fns!(import_2d_distorted, import_2d_undistorted),
                ));
                v.push(mk(
                    "start coordinates at 0",
                    "start coordinates at 1 (Matlab)",
                    pair_fns!(import_2d_count0, import_2d_count1),
                ));
                v.push(mk(
                    "y axis down",
                    "y axis up (Matlab)",
                    pair_fns!(import_2d_y_down, import_2d_y_up),
                ));
            }
            K::Export2d => {
                v.push(mk(
                    "distorted coordinates",
                    "undistorted coordinates",
                    pair_fns!(export_2d_distorted, export_2d_undistorted),
                ));
                v.push(mk(
                    "start coordinates at 0",
                    "start coordinates at 1 (Matlab)",
                    pair_fns!(export_2d_count0, export_2d_count1),
                ));
                v.push(mk(
                    "y axis down",
                    "y axis up (Matlab)",
                    pair_fns!(export_2d_y_down, export_2d_y_up),
                ));
            }
            K::Export3d | K::ExportTransformations => {}
        }
        // frame_Filtered: transformations only.
        if k == K::ExportTransformations {
            v.push(mk(
                "unfiltered",
                "filtered",
                pair_fns!(export_trans_unfiltered, export_trans_filtered),
            ));
        }
        // frame_Columns, with the text the constructor sets.
        let (cols_text, cols) = match k {
            K::Import2d => (
                "cam offset columns(Matlab xromm tools)",
                pair_fns!(import_2d_no_cols, import_2d_offset_cols),
            ),
            K::Export2d => (
                "cam offset columns(Matlab xromm tools)",
                pair_fns!(export_2d_no_cols, export_2d_offset_cols),
            ),
            K::Export3d => (
                "Frame Number (allows saving of subsequence)",
                pair_fns!(export_3d_no_cols, export_3d_offset_cols),
            ),
            K::ExportTransformations => (
                "Frame Number (allows saving of subsequence)",
                pair_fns!(export_trans_no_cols, export_trans_offset_cols),
            ),
        };
        v.push(mk("no leading colums", cols_text, cols));
        // frame_Status: import only.
        if k == K::Import2d {
            v.push(mk(
                "import as SET",
                "import as TRACKED",
                pair_fns!(import_status_set, import_status_tracked),
            ));
        }
        v
    }

    /// Draws the dialog. OK answers [`DialogResult::PointImportExport`]; Cancel, Escape and
    /// the close button close it without an answer.
    pub fn show(&mut self, ctx: &egui::Context, settings: &mut UiSettings) -> FormOutcome {
        let mut out = FormOutcome::default();
        let frames = self.frames();
        let modal = form::modal(ctx, egui::Id::unique("xma_point_import_export"), |ui| {
            ui.set_width(460.0);
            if form::dialog_title(ui, self.kind.title()) {
                out.close = true;
            }
            for p in &frames {
                row(
                    ui,
                    p,
                    &mut settings.import_export,
                    &mut out.settings_changed,
                );
                ui.add_space(4.0);
            }
            ui.add_space(8.0);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("OK").clicked() {
                    out.result = Some(DialogResult::PointImportExport(self.kind));
                    out.close = true;
                }
                if ui.button("Cancel").clicked() {
                    out.close = true;
                }
            });
        });
        if modal.escaped() {
            out.close = true;
        }
        out
    }
}

/// One frame (`QFrame` StyledPanel) with its two radio buttons.
fn row(ui: &mut Ui, p: &Pair, s: &mut ImportExportSettings, changed: &mut bool) {
    form::styled_frame(ui, |ui| {
        ui.horizontal(|ui| {
            let (l, r) = (p.get)(s);
            if form::radio(ui, true, l, p.left) && !l {
                (p.set)(s, false);
                *changed = true;
            }
            ui.add_space(24.0);
            if form::radio(ui, true, r, p.right) && !r {
                (p.set)(s, true);
                *changed = true;
            }
        });
    });
}

/// The import options the settings hold (`MainWindow::on_actionImport2D_Points_triggered`).
#[must_use]
pub fn import_options(s: &ImportExportSettings) -> xma_engine::edit::Import2dOptions {
    xma_engine::edit::Import2dOptions {
        distorted: s.import_2d_distorted,
        count_from_one: s.import_2d_count1,
        y_up: s.import_2d_y_up,
        header: s.import_2d_header,
        offset_columns: s.import_2d_offset_cols,
        status_set: s.import_status_set,
    }
}
