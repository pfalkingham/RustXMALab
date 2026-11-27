//! Saving a document: what `MainWindow`'s save handlers do around `ProjectFileIO::saveProject`.
//!
//! Ported from ui/MainWindow.cpp, MainWindow::saveProject (`:971-995`),
//! MainWindow::saveProjectAs (`:997-1053`) and MainWindow::updateBeforeSaveProject
//! (`:958-969`), and from the part of ui/ProjectFileIO.cpp, ProjectFileIO::saveProject that
//! writes files the model does not hold (`:190-196`, `PrecisionInfo_<trial>.txt` and
//! `MarkerDistances_<trial>.txt`). The archive itself is [`xma_io::project_write`]'s.
//!
//! What a save does here, in the C++ order:
//!
//! 1. The creation date: "Save" sets it only when the project has none (`:977-980`); "Save
//!    As" always resets it to the time of saving (`:1010`; ledger F22).
//! 2. `RecomputeWhenSaving` (off by default, `core/Settings.cpp:70`): when on, every saved
//!    trial that requires recomputation is updated ([`Document::update_trial`]) before
//!    writing. The C++ then busy-waits on `ThreadScheduler::isRunning`; here the update is
//!    synchronous.
//! 3. The archive: the model's files, written by [`xma_io::project_write::project_entries`];
//!    `PrecisionInfo_<trial>.txt` and `MarkerDistances_<trial>.txt` regenerated from the
//!    model for every saved trial that does not require recomputation
//!    ([`xma_io::export::precision::project_save_files`]); images, `log.html` and
//!    `log.html` carried over from the file the project was read from (the C++ re-encodes
//!    the images and regenerates the log; ledger F24); `xma_metadata.xml` written as
//!    `writePortalFile` writes it ([`crate::portal_summary`]). Copies of the two
//!    text files carried from the old file are dropped, so a stale one is never written.

use std::fmt;
use std::path::{Path, PathBuf};

use xma_io::WriteError;
use xma_io::export::ExportError;
use xma_io::export::precision::project_save_files;
use xma_io::project_write::{
    ExtraFiles, ProjectEntries, WriteOptions, project_entries, project_entries_borrowed,
    write_borrowed_entries,
};
use xma_io::zip::DosDateTime;

use crate::Document;

/// The time a save is stamped with: the project's `DateCreated` (when the save sets it) and
/// the modification time of every archive entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaveTime {
    /// `QDateTime::currentDateTime().toString(Qt::ISODate)`: `YYYY-MM-DDTHH:MM:SS`
    /// (`core/Project.cpp:148-152`).
    pub date_created: String,
    /// The zip entries' time.
    pub modified: DosDateTime,
    /// The local time's offset from UTC in seconds (`xma_metadata.xml`'s `date-modified`).
    pub utc_offset_seconds: i32,
}

impl SaveTime {
    /// A fixed time, from calendar fields. Returns `None` outside the zip format's years
    /// (1980 to 2107) or for invalid fields.
    #[must_use]
    pub fn at(year: u16, month: u8, day: u8, hour: u8, minute: u8, second: u8) -> Option<Self> {
        let modified = DosDateTime::from_fields(year, month, day, hour, minute, second)?;
        Some(Self {
            date_created: format!(
                "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}"
            ),
            modified,
            utc_offset_seconds: 0,
        })
    }

    /// The current time in the machine's local time zone, as the C++ stamps both values
    /// (ledger F27, decided 14 November 2026).
    ///
    /// Ported from core/Project.cpp, Project::set_date_created (`:148-152`):
    /// `QDateTime::currentDateTime().toString(Qt::ISODate)` is local time with no zone
    /// suffix and no milliseconds, `YYYY-MM-DDTHH:MM:SS`. The zip entries' time is what
    /// QuaZip's `QuaZipNewInfo(name, file)` takes from the temporary file
    /// (`QFileInfo::lastModified()`, local time), written moments before
    /// (ui/ProjectFileIO.cpp, ProjectFileIO::zipFromFolderToFile, `:2355`); here it is the
    /// same instant as the creation date. The zone comes from the system (`TZ`, then
    /// `/etc/localtime` on Unix, the system setting on Windows) through `jiff`; where it
    /// cannot be found `jiff` uses UTC. A time outside the zip format's years (1980 to
    /// 2107) is clamped to its first or last second.
    #[must_use]
    pub fn now() -> Self {
        Self::from_zoned(&jiff::Zoned::now())
    }

    /// The calendar fields of a zoned time as a save time ([`SaveTime::now`]).
    #[must_use]
    pub fn from_zoned(time: &jiff::Zoned) -> Self {
        let year = time.year();
        let fields = if year < 1980 {
            (1980, 1, 1, 0, 0, 0)
        } else if year > 2107 {
            (2107, 12, 31, 23, 59, 59)
        } else {
            (
                u16::try_from(year).unwrap_or(1980),
                u8::try_from(time.month()).unwrap_or(1),
                u8::try_from(time.day()).unwrap_or(1),
                u8::try_from(time.hour()).unwrap_or(0),
                u8::try_from(time.minute()).unwrap_or(0),
                u8::try_from(time.second()).unwrap_or(0),
            )
        };
        let (y, mo, d, h, mi, s) = fields;
        let mut t = Self::at(y, mo, d, h, mi, s)
            .unwrap_or_else(|| Self::at(1980, 1, 1, 0, 0, 0).expect("the zip epoch"));
        t.utc_offset_seconds = time.offset().seconds();
        t
    }

    /// Parses a fixed time given as `YYYY-MM-DDTHH:MM:SS` (the form the creation date is
    /// written in, a space also accepted for the `T`), for tests and the command line.
    /// Returns `None` for any other text, or outside the zip format's years.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        let b = text.as_bytes();
        if b.len() != 19
            || b[4] != b'-'
            || b[7] != b'-'
            || !(b[10] == b'T' || b[10] == b' ')
            || b[13] != b':'
            || b[16] != b':'
        {
            return None;
        }
        let num = |r: std::ops::Range<usize>| -> Option<u16> {
            let s = &text[r];
            s.bytes()
                .all(|c| c.is_ascii_digit())
                .then(|| s.parse().ok())
                .flatten()
        };
        let small = |r| num(r).and_then(|v| u8::try_from(v).ok());
        Self::at(
            num(0..4)?,
            small(5..7)?,
            small(8..10)?,
            small(11..13)?,
            small(14..16)?,
            small(17..19)?,
        )
    }
}

/// Options of a save.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaveOptions {
    /// The C++ setting `RecomputeWhenSaving` (default off).
    pub recompute_when_saving: bool,
    /// The time the save is stamped with.
    pub time: SaveTime,
    /// The trials to save, by index, in order. `None` saves every trial ("Save", "Save
    /// As"); `Some` is "Save subset of trials" (`saveProjectAs(true)`), where only the
    /// chosen trials are updated by `RecomputeWhenSaving`.
    pub trials: Option<Vec<usize>>,
}

impl Default for SaveOptions {
    /// `RecomputeWhenSaving` off, the current local time ([`SaveTime::now`]), every trial.
    fn default() -> Self {
        Self {
            recompute_when_saving: false,
            time: SaveTime::now(),
            trials: None,
        }
    }
}

/// Why a project was not saved.
#[derive(Debug)]
pub enum SaveError {
    /// "Save" on a document that has no file yet (the C++ turns it into "Save As").
    NoPath,
    /// A trial index in [`SaveOptions::trials`] that does not exist.
    NoSuchTrial(usize),
    /// A precision or distance file could not be produced (an inconsistent model).
    Export {
        /// The trial's name.
        trial: String,
        error: ExportError,
    },
    /// The archive could not be built or written.
    Write(WriteError),
}

impl fmt::Display for SaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPath => write!(f, "the project has no file name yet; use Save As"),
            Self::NoSuchTrial(t) => write!(f, "trial {} does not exist", t + 1),
            Self::Export { trial, error } => write!(f, "trial '{trial}': {error}"),
            Self::Write(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SaveError {}

impl From<WriteError> for SaveError {
    fn from(e: WriteError) -> Self {
        Self::Write(e)
    }
}

/// What a save did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaveReport {
    /// The file written.
    pub path: PathBuf,
    /// Trials updated by `RecomputeWhenSaving`, in order.
    pub recomputed: Vec<usize>,
    /// Number of archive entries.
    pub entries: usize,
    /// Files the C++ writes that could not be produced (see
    /// [`ProjectEntries::not_written`]).
    pub not_written: Vec<String>,
    /// Files carried from the old archive that are not part of the project as saved.
    pub unused_extras: Vec<String>,
    /// Files the C++ would leave out of the archive (see [`ProjectEntries::dropped`]).
    pub dropped: Vec<String>,
}

/// True for the two files `saveProject` writes at the top of the archive for each trial.
fn is_trial_summary(name: &str) -> bool {
    !name.contains(['/', '\\'])
        && name.ends_with(".txt")
        && (name.starts_with("PrecisionInfo_") || name.starts_with("MarkerDistances_"))
}

impl Document {
    /// "Save": writes the project to its file ([`Document::path`]).
    ///
    /// Ported from ui/MainWindow.cpp, MainWindow::saveProject: the creation date is set only
    /// when empty, then `updateBeforeSaveProject`, then `ProjectFileIO::saveProject` with
    /// every trial.
    ///
    /// # Errors
    /// [`SaveError::NoPath`] when the document has no file, and the errors of
    /// [`Document::save_entries`] and of writing.
    pub fn save(&mut self, options: &SaveOptions) -> Result<SaveReport, SaveError> {
        let path = self.path.clone().ok_or(SaveError::NoPath)?;
        let report = self.write_to(&path, false, options)?;
        if options.trials.is_none() {
            self.mark_saved();
        }
        Ok(report)
    }

    /// "Save As" (or "Save subset of trials" when [`SaveOptions::trials`] is set): writes the
    /// project to `path`, which becomes the document's file.
    ///
    /// Ported from ui/MainWindow.cpp, MainWindow::saveProjectAs: the creation date is reset
    /// to the time of saving (ledger F22), then `updateBeforeSaveProject` on the trials
    /// saved, then `ProjectFileIO::saveProject`. The file dialog's `.xma` default extension
    /// is the caller's business.
    ///
    /// # Errors
    /// As [`Document::save_entries`], and when the file cannot be written.
    pub fn save_as(&mut self, path: &Path, options: &SaveOptions) -> Result<SaveReport, SaveError> {
        let report = self.write_to(path, true, options)?;
        self.path = Some(path.to_path_buf());
        // A subset of trials ("Save subdataset as") does not hold the whole document.
        if options.trials.is_none() {
            self.mark_saved();
        }
        Ok(report)
    }

    fn write_to(
        &mut self,
        path: &Path,
        save_as: bool,
        options: &SaveOptions,
    ) -> Result<SaveReport, SaveError> {
        let (trials, recomputed) = self.prepare_save(save_as, options)?;
        let extras = self.save_extras(&trials, options)?;
        let entries =
            project_entries_borrowed(&self.project, &extras, &self.write_options(options))?;
        write_borrowed_entries(path, &entries, options.time.modified)?;
        Ok(SaveReport {
            path: path.to_path_buf(),
            recomputed,
            entries: entries.files.len(),
            not_written: entries.not_written,
            unused_extras: entries.unused_extras,
            dropped: entries.dropped,
        })
    }

    /// Everything a save does except writing the file: the creation date ("Save As" when
    /// `save_as`), `RecomputeWhenSaving`, and the archive's entries in order. Returns the
    /// entries and the trials recomputed.
    ///
    /// # Errors
    /// [`SaveError::NoSuchTrial`] for a trial index out of range, [`SaveError::Export`] when
    /// a trial's precision files cannot be produced, [`SaveError::Write`] when the model
    /// cannot be written (an image missing, an inconsistent model).
    pub fn save_entries(
        &mut self,
        save_as: bool,
        options: &SaveOptions,
    ) -> Result<(ProjectEntries, Vec<usize>), SaveError> {
        let (trials, recomputed) = self.prepare_save(save_as, options)?;
        let extras = self.save_extras(&trials, options)?;
        let entries = project_entries(&self.project, &extras, &self.write_options(options))?;
        Ok((entries, recomputed))
    }

    /// The steps of a save that change the document: the creation date, the trials'
    /// `RecomputeWhenSaving`, and the non-finite values cleared. Returns the trials to save
    /// and the trials recomputed.
    fn prepare_save(
        &mut self,
        save_as: bool,
        options: &SaveOptions,
    ) -> Result<(Vec<usize>, Vec<usize>), SaveError> {
        let trials: Vec<usize> = match &options.trials {
            Some(list) => list.clone(),
            None => (0..self.project.trials.len()).collect(),
        };
        if let Some(&t) = trials.iter().find(|&&t| t >= self.project.trials.len()) {
            return Err(SaveError::NoSuchTrial(t));
        }

        self.note_change();
        // Save: `if (date_created.isEmpty()) set_date_created()`; Save As: always.
        if save_as || self.project.date_created.is_empty() {
            self.project.date_created = options.time.date_created.clone();
        }

        // MainWindow::updateBeforeSaveProject.
        let mut recomputed = Vec::new();
        if options.recompute_when_saving {
            for &t in &trials {
                if self.project.trials[t].requires_recomputation {
                    self.update_trial(t);
                    recomputed.push(t);
                }
            }
        }

        // Ledger F50: what is written is what the document then holds.
        self.clear_non_finite();
        Ok((trials, recomputed))
    }

    /// The archive entries the model does not hold: the attachments carried from the file
    /// (borrowed from the document, not copied), each trial's precision files and the
    /// portal summary.
    fn save_extras(
        &self,
        trials: &[usize],
        options: &SaveOptions,
    ) -> Result<ExtraFiles<'_>, SaveError> {
        let mut extras = ExtraFiles::new();
        for a in self.extras.attachments.iter() {
            if !a.entry.is_dir() && !is_trial_summary(&a.name) {
                extras.insert_borrowed(&a.name, &a.data);
            }
        }
        for &t in trials {
            let trial = &self.project.trials[t];
            let files =
                project_save_files(trial, self.compat()).map_err(|error| SaveError::Export {
                    trial: trial.name.clone(),
                    error,
                })?;
            for f in files {
                extras.insert(&f.name, f.contents.into_bytes());
            }
        }
        // ProjectFileIO::writePortalFile, at every save (ledger F24).
        let modified = crate::portal_summary::iso_with_offset(
            &options.time.date_created,
            options.time.utc_offset_seconds,
        );
        extras.insert(
            "xma_metadata.xml",
            crate::portal_summary::xma_metadata_xml(self, trials, &modified).into_bytes(),
        );
        Ok(extras)
    }

    fn write_options(&self, options: &SaveOptions) -> WriteOptions {
        WriteOptions {
            modified: options.time.modified,
            trials: options.trials.clone(),
            compat: self.compat(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_times() {
        let t = SaveTime::at(2026, 11, 13, 9, 5, 7).expect("valid");
        assert_eq!(t.date_created, "2026-11-13T09:05:07");
        assert_eq!(t.modified.fields(), (2026, 11, 13, 9, 5, 6));
        assert!(SaveTime::at(1979, 1, 1, 0, 0, 0).is_none());
        let now = SaveTime::now();
        assert_eq!(now.date_created.len(), 19);
        assert_eq!(SaveTime::parse(&now.date_created), Some(now));
    }

    /// Ledger F27: the fields are the local ones of the zone, not UTC's.
    #[test]
    fn save_time_is_local() {
        let instant: jiff::Timestamp = "2026-11-14T23:30:15Z".parse().expect("timestamp");
        let tokyo = instant.in_tz("Asia/Tokyo").expect("zone");
        let t = SaveTime::from_zoned(&tokyo);
        assert_eq!(t.date_created, "2026-11-15T08:30:15");
        assert_eq!(t.modified.fields(), (2026, 11, 15, 8, 30, 14));
        let ny = instant.in_tz("America/New_York").expect("zone");
        assert_eq!(
            SaveTime::from_zoned(&ny).date_created,
            "2026-11-14T18:30:15"
        );
        let utc = instant.in_tz("UTC").expect("zone");
        assert_eq!(
            SaveTime::from_zoned(&utc).date_created,
            "2026-11-14T23:30:15"
        );
        // Outside the zip years: clamped.
        let old: jiff::Timestamp = "1975-06-01T00:00:00Z".parse().expect("timestamp");
        let t = SaveTime::from_zoned(&old.in_tz("UTC").expect("zone"));
        assert_eq!(t.date_created, "1980-01-01T00:00:00");
        // `now` is the system zone's local time.
        let before = jiff::Zoned::now();
        let now = SaveTime::now();
        let after = jiff::Zoned::now();
        let fmt = |z: &jiff::Zoned| z.strftime("%Y-%m-%dT%H:%M:%S").to_string();
        assert!(
            now.date_created == fmt(&before) || now.date_created == fmt(&after),
            "{} vs {} / {}",
            now.date_created,
            fmt(&before),
            fmt(&after)
        );
    }

    #[test]
    fn parse_fixed_times() {
        assert_eq!(
            SaveTime::parse("2026-11-13T09:05:07"),
            SaveTime::at(2026, 11, 13, 9, 5, 7)
        );
        assert_eq!(
            SaveTime::parse("2026-11-13 09:05:07"),
            SaveTime::at(2026, 11, 13, 9, 5, 7)
        );
        for bad in [
            "2026-11-13",
            "2026-13-13T09:05:07",
            "1979-11-13T09:05:07",
            "2026-11-13T09:05:0x",
            "+026-11-13T09:05:07",
        ] {
            assert_eq!(SaveTime::parse(bad), None, "{bad}");
        }
    }

    #[test]
    fn trial_summaries() {
        assert!(is_trial_summary("PrecisionInfo_Trial 1.txt"));
        assert!(is_trial_summary("MarkerDistances_x.txt"));
        assert!(!is_trial_summary("Trial 1/PrecisionInfo_x.txt"));
        assert!(!is_trial_summary("log.html"));
    }
}
