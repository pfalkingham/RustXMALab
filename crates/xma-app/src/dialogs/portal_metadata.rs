//! Project Metadata's "Add metadata from XMAPortal file": its questions, the file dialog,
//! and the edit.
//!
//! Ported from ui/MetaDataInfo.cpp, MetaDataInfo::on_pushButtonAdd_clicked (`:129-193`) and
//! ui/ProjectFileIO.cpp, ProjectFileIO::addMetaData (`:1423-1448`):
//!
//! 1. "Are you sure you want to manually add metadata from the portal? ..." (OK, Cancel);
//! 2. "Select the trial you want to add xml data", "Trial ": "Calibration" (when the
//!    project has a calibration) and every trial's name;
//! 3. the file: "Select XMA file" (`XMA-files (*.xma)`) for Calibration, "Select XMATRIAL
//!    file" (`XMATRIAL-files (*.xmatrial)`) for a trial;
//! 4. when that one has study data: "There is already data associated with the trial
//!    <item>. Are you sure you want to override it?" (OK, Cancel);
//! 5. `Edit::SetPortalMetadata` with the file's metadata (one undo step "Add Metadata");
//!    the window's tree is rebuilt when the document changes (`MetaDataInfo::update`).
//!
//! A file that is not a Portal archive adds nothing, as in XMALab 3; a file that cannot be
//! read, or whose metadata is malformed, shows the error (XMALab 3 crashes on an archive
//! with no folder, ledger F53, and hangs on malformed metadata, F56).

use std::path::PathBuf;

use xma_engine::Edit;

use super::edit_forms::{FormOutcome, Prompt, PromptAnswer, PromptValue};
use super::last_dir;
use super::{ConfirmButtons, DialogEnv, DialogResult};
use crate::editing::EditRequest;

pub use xma_engine::portal::{
    CALIBRATION_ITEM, METADATA_CONFIRM as CONFIRM_TEXT, METADATA_ITEM_LABEL as ITEM_LABEL,
    METADATA_ITEM_TITLE as ITEM_TITLE, override_text,
};

#[derive(Clone, Debug, PartialEq)]
enum Step {
    Confirm,
    Item(Vec<String>),
    Override { trial: Option<usize>, file: PathBuf },
    Error,
}

/// The flow, one prompt at a time.
#[derive(Clone, Debug, PartialEq)]
pub struct AddMetadataFlow {
    step: Step,
    /// The prompt on screen.
    pub prompt: Prompt,
}

impl Default for AddMetadataFlow {
    fn default() -> Self {
        Self::new()
    }
}

impl AddMetadataFlow {
    /// The first question.
    #[must_use]
    pub fn new() -> Self {
        Self {
            step: Step::Confirm,
            prompt: Prompt::Confirm {
                message: CONFIRM_TEXT.to_owned(),
                buttons: ConfirmButtons::OkCancel,
            },
        }
    }

    /// Draws the prompt and goes on with its answer.
    pub fn show(&mut self, ctx: &egui::Context, env: &mut DialogEnv<'_>) -> FormOutcome {
        let mut out = FormOutcome::default();
        let Some(answer) = self.prompt.show(ctx, env.icons) else {
            return out;
        };
        let PromptAnswer::Accepted(value) = answer else {
            out.close = true;
            return out;
        };
        if let Some(r) = self.next(value, env) {
            out.result = r;
            out.close = true;
        }
        out
    }

    /// The accepted answer: `Some(result)` ends the flow (`None` inside: nothing to do).
    fn next(
        &mut self,
        value: PromptValue,
        env: &mut DialogEnv<'_>,
    ) -> Option<Option<DialogResult>> {
        let doc = env.doc?;
        let p = doc.project();
        match std::mem::replace(&mut self.step, Step::Error) {
            Step::Confirm => {
                let items = xma_engine::portal::metadata_items(p);
                self.prompt = Prompt::Item {
                    title: ITEM_TITLE.to_owned(),
                    label: ITEM_LABEL.to_owned(),
                    items: items.clone(),
                    selected: 0,
                };
                self.step = Step::Item(items);
                None
            }
            Step::Item(items) => {
                let PromptValue::Item(i) = value else {
                    return Some(None);
                };
                let item = items.get(i)?.clone();
                // Ledger U155: the item by its place, not its text.
                let trial = xma_engine::portal::metadata_target(p, i)?;
                let start = last_dir(env.settings);
                let file = match trial {
                    None => env.files.open_file(
                        "Select XMA file",
                        start.as_deref(),
                        ("XMA-files", &["xma"]),
                    ),
                    Some(_) => env.files.open_file(
                        "Select XMATRIAL file",
                        start.as_deref(),
                        ("XMATRIAL-files", &["xmatrial"]),
                    ),
                };
                let file = file.filter(|f| !f.as_os_str().is_empty())?;
                let has = match trial {
                    None => !p.portal_metadata.is_empty(),
                    Some(t) => p
                        .trials
                        .get(t)
                        .is_some_and(|t| !t.portal_metadata.is_empty()),
                };
                if has {
                    self.prompt = Prompt::Confirm {
                        message: override_text(&item),
                        buttons: ConfirmButtons::OkCancel,
                    };
                    self.step = Step::Override { trial, file };
                    return None;
                }
                self.read(trial, &file)
            }
            Step::Override { trial, file } => self.read(trial, &file),
            Step::Error => Some(None),
        }
    }

    /// `addMetaData`: the file's metadata as the edit; an error is shown as the next
    /// prompt (`None`: the flow goes on to it).
    fn read(
        &mut self,
        trial: Option<usize>,
        file: &std::path::Path,
    ) -> Option<Option<DialogResult>> {
        match xma_engine::portal::read_metadata(file, xma_model::CppCompat::default()) {
            Ok(Some(metadata)) => Some(Some(DialogResult::Request(EditRequest::Apply(
                Edit::SetPortalMetadata { trial, metadata },
            )))),
            Ok(None) => Some(None),
            Err(e) => {
                self.prompt = Prompt::Error {
                    message: format!("{}: {e}", file.display()),
                };
                self.step = Step::Error;
                None
            }
        }
    }
}
