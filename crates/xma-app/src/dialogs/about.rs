//! The About dialog (modal) and the Welcome dialog's shared parts.
//!
//! Ported from `AboutDialog.ui` and ui/AboutDialog.cpp:
//!
//! | Row | Control | Content |
//! |---|---|---|
//! | 3 | `label` | the banner (RustXMALab's own, 436 × 100 points) |
//! | 4 | `textBrowser` | welcome text with links (manual, issue tracker, new issue, paper) |
//! | 7 | `label_3`, `version_label` | "Version:", the program version |
//! | 8 | `label_4`, `date_label` | "Date:", the build date |
//! | 9-10 | `label_5`, `label_6`, `label_7` | "Author:", the author, his e-mail address (link) |
//! | 11 | `label_9` | link "Check for newer version" |
//! | 12 | `pushButtonOpenGLInfo` | "Show OpenGL Info": a message with the graphics information |
//! | 13 | `pushButtonLicense` | "RustXMALab license": a message with the licence |
//!
//! Different from XMALab's form (parity ledger U172): this program is a rewrite with its own
//! author and repository, so the texts are its own. The About and Welcome texts say that it
//! is based on the original XMALab by Benjamin Knörlein and was re-written in Rust by Peter
//! Falkingham; the manual, issue tracker, new-issue and "Check for newer version" links point
//! into this program's repository ([`REPOSITORY`]); the paragraphs and links of XMALab's
//! Google group (the original program's user group) and of the tutorial data on the XMAPortal
//! Sandbox (which no longer exists) are gone. "Author:" names Peter Falkingham alone, with
//! one e-mail address. "Date:" is the build date that `build.rs` sets ([`build_date`]), no
//! longer "unknown". The validation paper is still the original's, linked by its DOI
//! ([`PAPER`]; XMALab's `jeb.biologists.org` address is refused now), and the licence text
//! ([`LICENSE_TEXT`]) keeps the original's copyright notice, as its licence requires.
//! Clicked links open in the system's browser (`crate::links`).
//!
//! Different from XMALab's form (parity ledger U173): the program calls itself RustXMALab,
//! so that it is not taken for an official version of XMALab. The banner is its own picture
//! (`RustXMALab-Banner.png`), the licence button is "RustXMALab license", and the licence
//! text starts with one sentence saying that this program is a port of XMALab under the
//! same licence.

use egui::{RichText, Ui};

use super::form;
use crate::icons::{Icon, Icons};

/// One paragraph of the rich text the About and Welcome dialogs show.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Para {
    /// The bold italic heading.
    Heading(&'static str),
    Text(&'static str),
    Link(&'static str, &'static str),
    Blank,
}

/// The program's repository; every link but the paper and the e-mail address points into it.
pub const REPOSITORY: &str = "https://github.com/pfalkingham/RustXMALab";
const WIKI: &str = "https://github.com/pfalkingham/RustXMALab/wiki";
const ISSUES: &str = "https://github.com/pfalkingham/RustXMALab/issues";
const NEW_ISSUE: &str = "https://github.com/pfalkingham/RustXMALab/issues/new";
/// "Check for newer version".
pub const RELEASES: &str = "https://github.com/pfalkingham/RustXMALab/releases";
/// The validation paper of the original XMALab (its DOI).
pub const PAPER: &str = "https://doi.org/10.1242/jeb.145383";
/// The author, as the About dialog's grid names him.
pub const AUTHOR: &str = "Peter Falkingham";
/// The author's e-mail address (the grid's one link under "Author:").
pub const AUTHOR_EMAIL: &str = "p.l.falkingham@ljmu.ac.uk";

/// The paragraph that names the original program and this rewrite.
const BASED_ON: &str = "RustXMALab is based on the original XMALab written by Benjamin Knörlein, and was re-written from scratch in Rust by Peter Falkingham.";
const MANUAL_TEXT: &str = "The user manual is on the wiki of the program's GitHub repository.";
const ISSUES_TEXT: &str = "Another useful resource is the GitHub issue page. Feel free to open an issue whenever you encounter a crash or have an idea for a feature.";
const CITE_TEXT: &str = "If you publish research done with this program, please cite the validation paper of the original XMALab, published in the Journal of Experimental Biology.";
const PAPER_TITLE: &str = "Validation of XMALab software for marker-based XROMM";

/// `AboutDialog.ui`, `textBrowser`, with this program's texts and links (parity ledger U172).
pub const ABOUT_TEXT: &[Para] = &[
    Para::Heading("Welcome to RustXMALab"),
    Para::Text(BASED_ON),
    Para::Text(MANUAL_TEXT),
    Para::Link("RustXMALab User Manual", WIKI),
    Para::Blank,
    Para::Text(ISSUES_TEXT),
    Para::Link("RustXMALab issue tracker", ISSUES),
    Para::Link("Report a bug or request a feature", NEW_ISSUE),
    Para::Blank,
    Para::Text(CITE_TEXT),
    Para::Link(PAPER_TITLE, PAPER),
];

/// `WelcomeDialog.ui`, `textBrowser`: the About text without "Report a bug or request a
/// feature", and two closing lines (parity ledger U172).
pub const WELCOME_TEXT: &[Para] = &[
    Para::Heading("Welcome to RustXMALab"),
    Para::Text(BASED_ON),
    Para::Text(MANUAL_TEXT),
    Para::Link("RustXMALab User Manual", WIKI),
    Para::Blank,
    Para::Text(ISSUES_TEXT),
    Para::Link("RustXMALab issue tracker", ISSUES),
    Para::Blank,
    Para::Text(CITE_TEXT),
    Para::Link(PAPER_TITLE, PAPER),
    Para::Blank,
    Para::Text(
        "You can also find most of the information in this dialog in the Help and About dialogs.",
    ),
    Para::Text("We hope you enjoy working with RustXMALab."),
];

/// The links of the About dialog's grid, in order: the author's e-mail address and "Check
/// for newer version" (label, address).
pub const GRID_LINKS: &[(&str, &str)] = &[
    (AUTHOR_EMAIL, "mailto:p.l.falkingham@ljmu.ac.uk"),
    ("Check for newer version", RELEASES),
];

/// A `QTextBrowser` showing `text` (centred paragraphs, links open in the browser), in
/// egui's body text.
pub fn text_browser(ui: &mut Ui, id: &str, text: &[Para], height: f32) {
    egui::Frame::NONE
        .fill(ui.visuals().extreme_bg_color)
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .inner_margin(egui::Margin::same(6))
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt(id)
                .max_height(height)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for p in text {
                        match p {
                            Para::Heading(t) => {
                                ui.label(RichText::new(*t).strong().italics());
                            }
                            Para::Text(t) => {
                                ui.vertical_centered(|ui| {
                                    ui.add(egui::Label::new(RichText::new(*t)).wrap());
                                });
                            }
                            Para::Link(t, url) => {
                                ui.vertical_centered(|ui| {
                                    ui.hyperlink_to(RichText::new(*t), *url);
                                });
                            }
                            Para::Blank => {
                                ui.add_space(16.0);
                            }
                        }
                        ui.add_space(12.0);
                    }
                });
        });
}

/// The width the banner is drawn at, in points, whatever the picture's pixel size.
pub const BANNER_WIDTH: f32 = 436.0;

/// The size in points the banner is drawn at: [`BANNER_WIDTH`] wide, the height from the
/// picture's proportions (436 × 100 for the 1308 × 300 picture, and for a 436 × 100 one). A
/// picture drawn at a higher resolution therefore stays the same size on screen and is
/// sharper on a high-density display. A picture with no pixels gives 436 × 100.
#[must_use]
pub fn banner_size(pixels: egui::Vec2) -> egui::Vec2 {
    let h = if pixels.x >= 1.0 && pixels.y >= 1.0 {
        BANNER_WIDTH * pixels.y / pixels.x
    } else {
        100.0
    };
    egui::vec2(BANNER_WIDTH, h)
}

/// The banner (`label` with the banner pixmap, here `RustXMALab-Banner.png`), centred, at
/// [`banner_size`].
pub fn banner(ui: &mut Ui, icons: &Icons) {
    let tex = icons.texture(ui.ctx(), Icon::Banner);
    let size = banner_size(tex.size_vec2());
    ui.vertical_centered(|ui| {
        ui.add(egui::Image::new(egui::load::SizedTexture::new(
            tex.id(),
            size,
        )));
    });
}

/// The version shown in the About dialog and the window title.
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The build date (`PROJECT_BUILD_TIME`), `YYYY-MM-DD`, set by the crate's `build.rs`:
/// `XMALAB_BUILD_DATE` from the build's environment when that is set (unchanged), else the
/// UTC date of `SOURCE_DATE_EPOCH` when that is set, else the UTC date of the build. Cargo
/// reruns `build.rs` only when it, a file under the crate's `src/` folder or one of those two
/// variables changes, so without them the date is the day of the last build in which this
/// crate's own sources changed (a build in which only another workspace crate changed keeps
/// the earlier date).
#[must_use]
pub fn build_date() -> &'static str {
    env!("XMALAB_BUILD_DATE")
}

/// The licence button's label and the title of the message it opens
/// (`pushButtonLicense`, "XMALab license" in XMALab 3; parity ledger U173).
pub const LICENSE_BUTTON: &str = "RustXMALab license";

/// The licence text (`on_pushButtonLicense_clicked`): one sentence naming this program and
/// its licence, then XMALab's notice. The list of libraries names what this program uses,
/// not the C++ libraries.
pub const LICENSE_TEXT: &str = "RustXMALab is a port to Rust of XMALab and is distributed under the same licence, the GNU General Public License version 3 (GPL-3.0-only). XMALab's notice follows.\n\nXMALab -- Copyright \u{a9} 2015, Brown University, Providence, RI\n\nAll Rights Reserved\n\nUse of the XMALab software is provided under the terms of the GNU General Public License version 3 as published by the Free Software Foundation at http ://www.gnu.org/licenses/gpl-3.0.html, provided that this copyright notice appear in all copies and that the name of Brown University not be used in advertising or publicity pertaining to the use or distribution of the software without specific, written prior permission from Brown University.\n\nThis program uses code ported from, and libraries acquired under open source licenses as follows :\n   OpenCV 4.13.0 (ported routines) - Apache 2.0\n   Levmar 2.6 (ported routines) - GPL 2.0\n   egui, eframe, wgpu - MIT or Apache 2.0\n   See NOTICE.md for the full list.\n\nBROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE WHICH IS PROVIDED \"AS IS\", INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR FOR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, OR ANY OTHER LEGAL THEORY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.";

/// The graphics information (`on_pushButtonOpenGLInfo_clicked`, which shows OpenGL's
/// vendor, renderer, version, shading-language version and extensions, one per line). This
/// program draws with wgpu: the adapter name, backend, driver and driver information, one
/// per line, once the render state has passed them in ([`super::Dialogs::set_graphics`]).
#[must_use]
pub fn graphics_info(info: Option<&super::GraphicsInfo>) -> String {
    match info {
        Some(g) => format!("{}\n{}\n{}\n{}", g.name, g.backend, g.driver, g.driver_info),
        None => "Renderer: wgpu (eframe). The graphics adapter is not known yet.".to_owned(),
    }
}

/// What the About dialog asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AboutRequest {
    Close,
    OpenGlInfo,
    License,
}

/// Draws the About dialog's contents.
pub fn ui(ui: &mut Ui, icons: &Icons) -> Option<AboutRequest> {
    let mut req = None;
    ui.set_width(640.0);
    if form::dialog_title(ui, "About") {
        req = Some(AboutRequest::Close);
    }
    banner(ui, icons);
    text_browser(ui, "about_text", ABOUT_TEXT, 420.0);
    egui::Grid::new("about_grid").num_columns(2).show(ui, |ui| {
        ui.label("Version:");
        ui.label(version());
        ui.end_row();
        ui.label("Date:");
        ui.label(build_date());
        ui.end_row();
        ui.label("Author:");
        ui.label(AUTHOR);
        ui.end_row();
        ui.label("");
        let (email, mailto) = GRID_LINKS[0];
        ui.hyperlink_to(email, mailto);
        ui.end_row();
    });
    ui.vertical_centered(|ui| {
        let (label, url) = GRID_LINKS[1];
        ui.hyperlink_to(label, url);
    });
    if form::wide_button(ui, true, "Show OpenGL Info").clicked() {
        req = Some(AboutRequest::OpenGlInfo);
    }
    if form::wide_button(ui, true, LICENSE_BUTTON).clicked() {
        req = Some(AboutRequest::License);
    }
    req
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The banner is drawn 436 × 100 points whatever its pixel size.
    #[test]
    fn banner_is_436_by_100_points() {
        for px in [egui::vec2(1308.0, 300.0), egui::vec2(436.0, 100.0)] {
            assert_eq!(banner_size(px), egui::vec2(436.0, 100.0), "{px:?}");
        }
        assert_eq!(banner_size(egui::Vec2::ZERO), egui::vec2(436.0, 100.0));
    }
}
