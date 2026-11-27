//! Links that open in the system's browser or mail program.
//!
//! egui's `hyperlink_to` puts an [`egui::OutputCommand::OpenUrl`] in the frame's output,
//! and eframe opens it only with its `links` feature (the `webbrowser` crate), which this
//! workspace leaves off (`eframe` without default features). Without it eframe only logs
//! "Cannot open url", so a click on a link did nothing (parity ledger U172). Here the
//! application takes those commands itself, in one place:
//!
//! - [`collect`] moves the open-URL commands of the current viewport into a list kept in
//!   the context's memory. [`crate::XmaApp::ui`] calls it for the main window, and each
//!   immediate viewport (tool windows, floating panels, Detection Settings) calls it at the
//!   end of its closure.
//! - [`crate::XmaApp::ui`] then hands each address to its [`UrlOpener`]: the system's opener
//!   only when `main` asks for it ([`crate::Options::system_links`]), as
//!   [`crate::saving::FileChooser`] is the system dialog only then; otherwise (tests) a
//!   recording opener, so that no test starts a browser.
//!
//! The system opener starts `xdg-open` (Linux and other Unix), `/usr/bin/open` (macOS) or
//! `<system root>\System32\rundll32.exe url.dll,FileProtocolHandler` (Windows) with the
//! address as one argument, never through a shell, and does not wait for it (a thread reaps
//! the child). Only `http://`, `https://` and `mailto:` addresses without spaces or control
//! characters are opened.
//!
//! On macOS and Windows the program is named by its full system path (parity ledger U172,
//! review of 26 November 2026): on Windows `std::process::Command` looks for a bare name in
//! the program's own folder before the system folder, so a `rundll32.exe` lying beside
//! `rustxmalab.exe` (an unpacked download folder) would be run on a click; on macOS a bare
//! `open` would be taken from `PATH`. The Windows system root is read from `SystemRoot`,
//! then `windir`, when it is an absolute path, else `C:\Windows` ([`windows_system_root`]).
//! On Linux and the other Unix systems `xdg-open` is still found through `PATH`: where it is
//! installed differs between distributions, and on every one it is on `PATH`.

use std::ffi::{OsStr, OsString};
use std::process::{Command, Stdio};

use crate::log::AppLog;

/// The schemes that are opened (compared without regard to case).
const SCHEMES: [&str; 3] = ["http://", "https://", "mailto:"];

/// Whether `url` may be handed to the system's opener: it begins with `http://`, `https://`
/// or `mailto:`, has something after the scheme, and has no whitespace, control character
/// or double quote (so it is one argument on every system and cannot start with `-`).
#[must_use]
pub fn acceptable(url: &str) -> bool {
    let scheme = SCHEMES.iter().find(|s| {
        url.get(..s.len())
            .is_some_and(|p| p.eq_ignore_ascii_case(s))
    });
    let Some(scheme) = scheme else {
        return false;
    };
    url.len() > scheme.len()
        && !url.starts_with('-')
        && !url
            .chars()
            .any(|c| c.is_whitespace() || c.is_control() || c == '"')
}

/// The operating systems the opener knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenerOs {
    /// Linux and the other Unix systems: `xdg-open`.
    Unix,
    MacOs,
    Windows,
}

impl OpenerOs {
    /// The system this program was built for.
    #[must_use]
    pub fn current() -> Self {
        if cfg!(target_os = "macos") {
            OpenerOs::MacOs
        } else if cfg!(windows) {
            OpenerOs::Windows
        } else {
            OpenerOs::Unix
        }
    }
}

/// The program and arguments that open `url` on `os`, or `None` when `url` is refused
/// ([`acceptable`]). The address is the last argument, whole. The program is `xdg-open`
/// by name on Unix (found through `PATH`), `/usr/bin/open` on macOS, and
/// `<system root>\System32\rundll32.exe` on Windows, the system root chosen by
/// [`windows_system_root`] from this process's `SystemRoot` and `windir` (parity ledger
/// U172: never a bare name where the system's own program has a fixed place).
#[must_use]
pub fn opener_command(os: OpenerOs, url: &str) -> Option<(OsString, Vec<OsString>)> {
    let (system_root, windir) = if os == OpenerOs::Windows {
        (std::env::var_os("SystemRoot"), std::env::var_os("windir"))
    } else {
        (None, None)
    };
    opener_command_with(os, url, system_root.as_deref(), windir.as_deref())
}

/// [`opener_command`] with the values of `SystemRoot` and `windir` given (used on Windows
/// only), so that the choice can be tested without touching the process environment.
fn opener_command_with(
    os: OpenerOs,
    url: &str,
    system_root: Option<&OsStr>,
    windir: Option<&OsStr>,
) -> Option<(OsString, Vec<OsString>)> {
    if !acceptable(url) {
        return None;
    }
    let url = OsString::from(url);
    Some(match os {
        OpenerOs::Unix => (OsString::from("xdg-open"), vec![url]),
        OpenerOs::MacOs => (OsString::from("/usr/bin/open"), vec![url]),
        OpenerOs::Windows => {
            let root = windows_system_root(system_root, windir);
            (
                OsString::from(format!("{root}\\System32\\rundll32.exe")),
                vec![OsString::from("url.dll,FileProtocolHandler"), url],
            )
        }
    })
}

/// The Windows system folder used for `rundll32.exe`, without a trailing separator: the
/// value of `SystemRoot` when it is an absolute Windows path, else that of `windir`, else
/// `C:\Windows` (a root drive `C:\` gives `C:`). An absolute path is a drive letter, a
/// colon and a separator (`C:\...`, `C:/...`) or a UNC path (`\\server\share`); an empty, relative or non-Unicode value is
/// passed over. The path is handled as a string (not with [`std::path::Path`], whose rules
/// depend on the system it runs on), so that this means the same when tested on Linux.
#[must_use]
pub fn windows_system_root(system_root: Option<&OsStr>, windir: Option<&OsStr>) -> String {
    fn absolute(v: &str) -> bool {
        let b = v.as_bytes();
        let drive = b.len() >= 3
            && b[0].is_ascii_alphabetic()
            && b[1] == b':'
            && (b[2] == b'\\' || b[2] == b'/');
        drive || v.starts_with("\\\\")
    }
    [system_root, windir]
        .into_iter()
        .flatten()
        .filter_map(OsStr::to_str)
        .filter(|v| absolute(v) && !v.contains('\0'))
        // A root drive "C:\" becomes "C:", to which "\System32\rundll32.exe" is added;
        // a bare "\\" (no server) is passed over.
        .map(|v| v.trim_end_matches(['\\', '/']))
        .find(|v| !v.is_empty())
        .map_or_else(|| "C:\\Windows".to_owned(), str::to_owned)
}

/// Who opens the addresses of clicked links.
#[derive(Debug)]
pub enum UrlOpener {
    /// The system's opener (the program, when `main` asks for it).
    System,
    /// Records the accepted addresses and opens nothing (tests, screenshots).
    Record(Vec<String>),
}

impl Default for UrlOpener {
    fn default() -> Self {
        UrlOpener::Record(Vec::new())
    }
}

impl UrlOpener {
    /// The addresses a [`UrlOpener::Record`] was handed (empty for the system opener).
    #[must_use]
    pub fn recorded(&self) -> &[String] {
        match self {
            UrlOpener::System => &[],
            UrlOpener::Record(v) => v,
        }
    }

    /// Opens `url`, or writes on the Console why not. Never waits for the opener.
    pub fn open(&mut self, url: &str, log: &AppLog) {
        if !acceptable(url) {
            log.error(format!(
                "Not opening the link \"{}\": only http, https and mailto addresses are opened.",
                url.escape_debug()
            ));
            return;
        }
        match self {
            UrlOpener::Record(v) => v.push(url.to_owned()),
            UrlOpener::System => {
                let Some((program, args)) = opener_command(OpenerOs::current(), url) else {
                    return;
                };
                let spawned = Command::new(&program)
                    .args(&args)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn();
                match spawned {
                    Ok(mut child) => {
                        // Reaped on a thread of its own, so that the interface never waits
                        // and no zombie is left.
                        let reaper = std::thread::Builder::new()
                            .name("xma-link-opener".into())
                            .spawn(move || {
                                let _ = child.wait();
                            });
                        if reaper.is_err() {
                            log.error("Could not start a thread to wait for the link opener.");
                        }
                    }
                    Err(e) => log.error(format!(
                        "Could not open the link {url} with {}: {e}",
                        program.to_string_lossy()
                    )),
                }
            }
        }
    }
}

/// The memory key of the addresses collected this frame.
fn pending_id() -> egui::Id {
    egui::Id::unique("xma_links_pending")
}

/// Moves the open-URL commands of the current viewport's output into the list
/// [`take`] returns. Called at the end of every viewport's drawing.
pub fn collect(ctx: &egui::Context) {
    let urls: Vec<String> = ctx.output_mut(|o| {
        let mut urls = Vec::new();
        o.commands.retain(|c| match c {
            egui::OutputCommand::OpenUrl(u) => {
                urls.push(u.url.clone());
                false
            }
            _ => true,
        });
        urls
    });
    if !urls.is_empty() {
        ctx.data_mut(|d| {
            d.get_temp_mut_or_default::<Vec<String>>(pending_id())
                .extend(urls);
        });
    }
}

/// The addresses [`collect`] has gathered since the last call, in order.
#[must_use]
pub fn take(ctx: &egui::Context) -> Vec<String> {
    ctx.data_mut(|d| d.remove_temp::<Vec<String>>(pending_id()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_web_and_mail_addresses_are_accepted() {
        for ok in [
            "https://github.com/pfalkingham/RustXMALab/wiki",
            "http://example.org/content/219",
            "mailto:someone@example.org",
            "HTTPS://example.org",
        ] {
            assert!(acceptable(ok), "{ok}");
        }
        for bad in [
            "",
            "javascript:alert(1)",
            "file:///etc/passwd",
            "-https://example.org",
            "--help",
            "https://example.org/\nrm",
            "https://example.org/\r",
            "https://exa mple.org",
            "https://example.org/\"x",
            "https://",
            "mailto:",
            "ftp://example.org",
            "xmalab://x",
        ] {
            assert!(!acceptable(bad), "{bad:?}");
        }
    }

    #[test]
    fn the_command_has_the_address_as_one_argument() {
        let url = "https://github.com/pfalkingham/RustXMALab/issues?q=is%3Aopen&x=1;y";
        let (p, a) = opener_command(OpenerOs::Unix, url).unwrap();
        assert_eq!(
            (p, a),
            (OsString::from("xdg-open"), vec![OsString::from(url)])
        );
        let (p, a) = opener_command(OpenerOs::MacOs, url).unwrap();
        assert_eq!(
            (p, a),
            (OsString::from("/usr/bin/open"), vec![OsString::from(url)])
        );
        let (p, a) = opener_command_with(
            OpenerOs::Windows,
            url,
            Some(OsStr::new("C:\\Windows")),
            None,
        )
        .unwrap();
        assert_eq!(
            (p, a),
            (
                OsString::from("C:\\Windows\\System32\\rundll32.exe"),
                vec![
                    OsString::from("url.dll,FileProtocolHandler"),
                    OsString::from(url)
                ]
            )
        );
        // Whatever this process's environment holds, the Windows program is a full path.
        let (p, a) = opener_command(OpenerOs::Windows, url).unwrap();
        let p = p.to_str().unwrap().to_owned();
        assert!(p.ends_with("\\System32\\rundll32.exe"), "{p}");
        assert!(p.as_bytes()[1] == b':' || p.starts_with("\\\\"), "{p}");
        assert_eq!(a.len(), 2);
        // The std Command built from it carries the same arguments (not run).
        let (p, a) = opener_command(OpenerOs::current(), url).unwrap();
        let mut c = Command::new(p);
        c.args(&a);
        let got: Vec<_> = c.get_args().collect();
        assert_eq!(got.last().copied(), Some(std::ffi::OsStr::new(url)));
        assert_eq!(got.len(), a.len());
        for os in [OpenerOs::Unix, OpenerOs::MacOs, OpenerOs::Windows] {
            for bad in ["javascript:x", "file:///x", "", "-x", "https://a\nb"] {
                assert!(opener_command(os, bad).is_none(), "{os:?} {bad:?}");
            }
        }
    }

    #[test]
    fn the_windows_program_is_under_the_system_root() {
        let root = |sr: Option<&str>, wd: Option<&str>| {
            windows_system_root(sr.map(OsStr::new), wd.map(OsStr::new))
        };
        let program = |sr: Option<&str>, wd: Option<&str>| {
            let (p, a) = opener_command_with(
                OpenerOs::Windows,
                "mailto:someone@example.org",
                sr.map(OsStr::new),
                wd.map(OsStr::new),
            )
            .unwrap();
            assert_eq!(
                a,
                [
                    OsString::from("url.dll,FileProtocolHandler"),
                    OsString::from("mailto:someone@example.org")
                ]
            );
            p.into_string().unwrap()
        };
        const FALLBACK: &str = "C:\\Windows\\System32\\rundll32.exe";
        assert_eq!(program(Some("C:\\Windows"), None), FALLBACK);
        assert_eq!(
            program(Some("C:\\Windows"), Some("D:\\WINNT")),
            FALLBACK,
            "SystemRoot comes first"
        );
        assert_eq!(
            program(None, Some("D:\\WINNT")),
            "D:\\WINNT\\System32\\rundll32.exe"
        );
        assert_eq!(program(None, None), FALLBACK);
        for bad in [
            "",
            "Windows",
            "WINNT\\x",
            ".\\Windows",
            "\\Windows",
            "C:Windows",
            "C:",
            "\\\\",
        ] {
            assert_eq!(program(Some(bad), None), FALLBACK, "{bad:?}");
            assert_eq!(program(Some(bad), Some(bad)), FALLBACK, "{bad:?}");
            assert_eq!(
                program(Some(bad), Some("D:\\WINNT")),
                "D:\\WINNT\\System32\\rundll32.exe",
                "{bad:?}"
            );
        }
        assert_eq!(root(Some("e:/win/"), None), "e:/win");
        assert_eq!(root(Some("C:\\"), None), "C:");
        assert_eq!(
            root(Some("\\\\srv\\share\\Windows\\"), None),
            "\\\\srv\\share\\Windows"
        );
        assert_eq!(program(Some("C:\\"), None), "C:\\System32\\rundll32.exe");
    }

    #[test]
    fn a_refused_address_is_reported_and_not_recorded() {
        let log = AppLog::new();
        let mut o = UrlOpener::default();
        o.open("javascript:alert(1)", &log);
        assert!(o.recorded().is_empty());
        assert_eq!(log.len(), 1);
        o.open("https://example.org", &log);
        assert_eq!(o.recorded(), ["https://example.org".to_owned()]);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn collect_takes_the_open_url_commands_only() {
        let ctx = egui::Context::default();
        let mut taken = Vec::new();
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
            ui.ctx()
                .open_url(egui::OpenUrl::same_tab("https://example.org/a"));
            ui.ctx().copy_text("kept".into());
            ui.ctx()
                .open_url(egui::OpenUrl::new_tab("https://example.org/b"));
            collect(ui.ctx());
            taken = take(ui.ctx());
        });
        assert_eq!(taken, ["https://example.org/a", "https://example.org/b"]);
        assert_eq!(out.platform_output.commands.len(), 1);
        // The font atlas the pass made is not drawn here.
        out.textures_delta.clear();
        assert!(take(&ctx).is_empty());
    }
}
