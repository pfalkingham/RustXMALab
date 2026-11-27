//! The application log shown in the Console panel.
//!
//! Ported from ui/ConsoleDockWidget.cpp: `writeLog(message, level)` with the four levels and
//! colours of the C++ (0 black, 1 dark green, 2 dark blue for standard output, 3 dark red
//! for standard error), `clear` (on closing a project) and `afterLoad` (after loading, the
//! console shows the log the project was saved with, `log.html` in the project file).
//!
//! The C++ captures the process's `stdout` and `stderr` into the console every 500 ms; here
//! code that wants a message in the console calls [`AppLog::write`] (or the shorthands)
//! instead. The log is shareable between threads ([`AppLog::clone`] gives a handle to the
//! same log), so a worker can write to it.

use std::sync::{Arc, Mutex};

/// The four levels of `ConsoleDockWidget::writeLog`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LogLevel {
    /// 0: black.
    Normal,
    /// 1: rgb(0, 100, 0).
    Success,
    /// 2: rgb(0, 0, 100); the C++ uses it for standard output.
    Output,
    /// 3: rgb(100, 0, 0); the C++ uses it for standard error (warnings and errors).
    Error,
}

impl LogLevel {
    /// The colour the C++ writes this level in (`ConsoleDockWidget::writeLog`). On a dark
    /// background the C++ keeps these colours, which makes level 0 invisible; the console
    /// uses the theme's text colour for level 0 there instead (see [`LogLevel::color_on`]).
    #[must_use]
    pub fn cpp_color(self) -> egui::Color32 {
        match self {
            LogLevel::Normal => egui::Color32::from_rgb(0, 0, 0),
            LogLevel::Success => egui::Color32::from_rgb(0, 100, 0),
            LogLevel::Output => egui::Color32::from_rgb(0, 0, 100),
            LogLevel::Error => egui::Color32::from_rgb(100, 0, 0),
        }
    }

    /// The colour to draw this level with in the given visuals: level 0 in egui's text
    /// colour (the look is the toolkit's own, ledger U139); the others in the C++ colour in
    /// light mode and lightened colours in dark mode.
    #[must_use]
    pub fn color_on(self, visuals: &egui::Visuals) -> egui::Color32 {
        if self == LogLevel::Normal {
            return visuals.text_color();
        }
        if !visuals.dark_mode {
            return self.cpp_color();
        }
        match self {
            LogLevel::Normal => visuals.text_color(),
            LogLevel::Success => egui::Color32::from_rgb(110, 200, 110),
            LogLevel::Output => egui::Color32::from_rgb(130, 150, 230),
            LogLevel::Error => egui::Color32::from_rgb(230, 110, 110),
        }
    }
}

/// One line (or block) of the log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogEntry {
    pub level: LogLevel,
    pub text: String,
}

#[derive(Debug, Default)]
struct Inner {
    entries: Vec<LogEntry>,
    /// Incremented on every change, so a view can tell when to scroll to the end.
    generation: u64,
}

/// The application log. Cloning gives another handle to the same log.
#[derive(Clone, Debug, Default)]
pub struct AppLog {
    inner: Arc<Mutex<Inner>>,
}

impl AppLog {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        // A panic while holding the lock leaves the log usable.
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Ported from ui/ConsoleDockWidget.cpp, ConsoleDockWidget::writeLog.
    pub fn write(&self, text: impl Into<String>, level: LogLevel) {
        let mut g = self.lock();
        g.entries.push(LogEntry {
            level,
            text: text.into(),
        });
        g.generation += 1;
    }

    /// Level 0.
    pub fn info(&self, text: impl Into<String>) {
        self.write(text, LogLevel::Normal);
    }

    /// Level 3 (what the C++ prints to `stderr`).
    pub fn error(&self, text: impl Into<String>) {
        self.write(text, LogLevel::Error);
    }

    /// Ported from ui/ConsoleDockWidget.cpp, ConsoleDockWidget::clear.
    pub fn clear(&self) {
        let mut g = self.lock();
        g.entries.clear();
        g.generation += 1;
    }

    /// Ported from ui/ConsoleDockWidget.cpp, ConsoleDockWidget::afterLoad: replaces the log
    /// with the one stored in the project (`log.html`, converted by [`html_log_to_entries`]).
    pub fn replace_with(&self, entries: Vec<LogEntry>) {
        let mut g = self.lock();
        g.entries = entries;
        g.generation += 1;
    }

    /// A copy of the entries.
    #[must_use]
    pub fn entries(&self) -> Vec<LogEntry> {
        self.lock().entries.clone()
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Changes whenever the log changes.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.lock().generation
    }

    /// Runs `f` on the entries without copying them.
    pub fn with_entries<R>(&self, f: impl FnOnce(&[LogEntry]) -> R) -> R {
        f(&self.lock().entries)
    }
}

/// Converts the console log a project stores (`log.html`, written by `QTextEdit::toHtml`)
/// into entries: one per paragraph, with the level taken from the span colour the C++
/// wrote (`#640000` error, `#000064` output, `#006400` success, else normal). Empty
/// paragraphs are dropped. Only the HTML that `QTextEdit` writes is understood.
#[must_use]
pub fn html_log_to_entries(html: &str) -> Vec<LogEntry> {
    let mut out = Vec::new();
    let body = html.find("<body").map_or(html, |i| &html[i..]);
    for para in body.split("<p").skip(1) {
        let end = para.find("</p>").unwrap_or(para.len());
        let para = &para[..end];
        if para.contains("-qt-paragraph-type:empty") {
            continue;
        }
        let lower = para.to_ascii_lowercase();
        let level = if lower.contains("#640000") {
            LogLevel::Error
        } else if lower.contains("#000064") {
            LogLevel::Output
        } else if lower.contains("#006400") {
            LogLevel::Success
        } else {
            LogLevel::Normal
        };
        // Strip the tags, then decode the few entities QTextEdit writes.
        let mut text = String::new();
        let mut in_tag = false;
        // The first '>' closes the <p ...> tag itself.
        let inner = para.find('>').map_or(para, |i| &para[i + 1..]);
        for c in inner.chars() {
            match c {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => text.push(c),
                _ => {}
            }
        }
        let text = text
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&nbsp;", " ")
            .replace("&amp;", "&");
        let text = text.trim_end().to_owned();
        if !text.is_empty() {
            out.push(LogEntry { level, text });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_log_parses_qtextedit_output() {
        let html = r#"<html><body style="x"><p style="a"><span style=" color:#640000;">Inlier 16</span></p>
<p style="-qt-paragraph-type:empty; margin-top:0px;"><br /></p>
<p style="b"><span style=" color:#000064;">Load trial &quot;T1&quot;</span></p></body></html>"#;
        let e = html_log_to_entries(html);
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].level, LogLevel::Error);
        assert_eq!(e[0].text, "Inlier 16");
        assert_eq!(e[1].level, LogLevel::Output);
        assert_eq!(e[1].text, "Load trial \"T1\"");
    }

    #[test]
    fn log_is_shared_between_handles() {
        let a = AppLog::new();
        let b = a.clone();
        b.error("x");
        assert_eq!(a.len(), 1);
        a.clear();
        assert!(b.is_empty());
    }
}
