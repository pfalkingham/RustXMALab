//! The icons of XMALab 3, embedded in the program.
//!
//! Copied from ~/Projects/xmalab/src/ui/resource-files/icons/ (GPL-3, like this project)
//! into `crates/xma-app/assets/icons/`; the list is ui-map.md, "Resources". The banner and
//! the program's icon are not XMALab's: `RustXMALab-Banner.png` and `RustXMALab-Icon.png` are
//! this program's own (parity ledger U173). Each icon is
//! decoded once per context and kept as a texture; [`Icons::get`] gives an `egui::Image`
//! source for it.

use std::collections::HashMap;
use std::sync::Mutex;

/// Every icon the interface uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Icon {
    About,
    Alert,
    ArrowOutLonger,
    BackwardBlue,
    Error,
    ForwardBlue,
    Images,
    NextBlue,
    PlayBlue,
    PlayBackwardBlue,
    PrevBlue,
    Settings,
    Shape3d,
    Shape3dNotSet,
    Shape3dOptimized,
    Shape3dSetMarker,
    Shape3dSetMarkerOptimized,
    StopBlue,
    Video,
    Banner,
    AppIcon,
}

impl Icon {
    /// The PNG data.
    #[must_use]
    pub fn bytes(self) -> &'static [u8] {
        macro_rules! png {
            ($f:literal) => {
                include_bytes!(concat!("../assets/icons/", $f, ".png"))
            };
        }
        match self {
            Icon::About => png!("About"),
            Icon::Alert => png!("Alert"),
            Icon::ArrowOutLonger => png!("arrow_out_longer"),
            Icon::BackwardBlue => png!("backward_blue"),
            Icon::Error => png!("Error"),
            Icon::ForwardBlue => png!("forward_blue"),
            Icon::Images => png!("Images"),
            Icon::NextBlue => png!("next_blue"),
            Icon::PlayBlue => png!("play_blue"),
            Icon::PlayBackwardBlue => png!("playbackward_blue"),
            Icon::PrevBlue => png!("prev_blue"),
            Icon::Settings => png!("settings"),
            Icon::Shape3d => png!("shape_3d"),
            Icon::Shape3dNotSet => png!("shape_3d_notSet"),
            Icon::Shape3dOptimized => png!("shape_3d_optimized"),
            Icon::Shape3dSetMarker => png!("shape_3d_setMarker"),
            Icon::Shape3dSetMarkerOptimized => png!("shape_3d_setMarker_optimized"),
            Icon::StopBlue => png!("stop_blue"),
            Icon::Video => png!("video"),
            Icon::Banner => png!("RustXMALab-Banner"),
            Icon::AppIcon => png!("RustXMALab-Icon"),
        }
    }

    /// The file name in `assets/icons/` (the C++ resources' name, but for the banner and the
    /// program's icon).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Icon::About => "About.png",
            Icon::Alert => "Alert.png",
            Icon::ArrowOutLonger => "arrow_out_longer.png",
            Icon::BackwardBlue => "backward_blue.png",
            Icon::Error => "Error.png",
            Icon::ForwardBlue => "forward_blue.png",
            Icon::Images => "Images.png",
            Icon::NextBlue => "next_blue.png",
            Icon::PlayBlue => "play_blue.png",
            Icon::PlayBackwardBlue => "playbackward_blue.png",
            Icon::PrevBlue => "prev_blue.png",
            Icon::Settings => "settings.png",
            Icon::Shape3d => "shape_3d.png",
            Icon::Shape3dNotSet => "shape_3d_notSet.png",
            Icon::Shape3dOptimized => "shape_3d_optimized.png",
            Icon::Shape3dSetMarker => "shape_3d_setMarker.png",
            Icon::Shape3dSetMarkerOptimized => "shape_3d_setMarker_optimized.png",
            Icon::StopBlue => "stop_blue.png",
            Icon::Video => "video.png",
            Icon::Banner => "RustXMALab-Banner.png",
            Icon::AppIcon => "RustXMALab-Icon.png",
        }
    }

    /// Decodes the PNG into RGBA.
    ///
    /// # Panics
    /// Never for the embedded files (checked by a test).
    #[must_use]
    pub fn decode(self) -> egui::ColorImage {
        let mut img = image::load_from_memory_with_format(self.bytes(), image::ImageFormat::Png)
            .expect("embedded icon decodes")
            .to_rgba8();
        if self == Icon::Settings {
            clear_white_backdrop(&mut img);
        }
        if self.is_symbolic() {
            // White strokes, so that the tint ([`Icon::tint`]) gives them the text colour.
            for p in img.pixels_mut() {
                p.0[0] = 255;
                p.0[1] = 255;
                p.0[2] = 255;
            }
        }
        let size = [img.width() as usize, img.height() as usize];
        egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw())
    }

    /// An icon drawn only in black strokes (`video.png`): it is drawn in the text colour of
    /// the theme, black in the light theme and white in the dark one. In XMALab 3's dark
    /// theme it stays black on the dark button face and cannot be seen.
    #[must_use]
    pub fn is_symbolic(self) -> bool {
        matches!(self, Icon::Video)
    }

    /// The tint an icon is drawn with: the text colour for a symbolic icon, else white
    /// (the icon's own colours).
    #[must_use]
    pub fn tint(self, visuals: &egui::Visuals) -> egui::Color32 {
        if self.is_symbolic() {
            visuals.text_color()
        } else {
            egui::Color32::WHITE
        }
    }

    /// The window icon (`RustXMALab-Icon.png`, scaled down to 64 × 64).
    #[must_use]
    pub fn window_icon() -> egui::IconData {
        let img =
            image::load_from_memory_with_format(Icon::AppIcon.bytes(), image::ImageFormat::Png)
                .expect("embedded icon decodes")
                .resize(64, 64, image::imageops::FilterType::Triangle)
                .to_rgba8();
        egui::IconData {
            width: img.width(),
            height: img.height(),
            rgba: img.into_raw(),
        }
    }
}

/// `settings.png` has no transparency: its gear sits on an opaque white square, which
/// shows as a white box on the dark theme's bars and buttons (as in XMALab 3's dark theme).
/// The pure white pixels joined to the border (4-connected) become transparent; the gear's
/// own light pixels are kept.
fn clear_white_backdrop(img: &mut image::RgbaImage) {
    let (w, h) = img.dimensions();
    let white = |p: &image::Rgba<u8>| p.0 == [255, 255, 255, 255];
    let mut stack: Vec<(u32, u32)> = Vec::new();
    for x in 0..w {
        stack.push((x, 0));
        stack.push((x, h - 1));
    }
    for y in 0..h {
        stack.push((0, y));
        stack.push((w - 1, y));
    }
    while let Some((x, y)) = stack.pop() {
        if !white(img.get_pixel(x, y)) {
            continue;
        }
        img.put_pixel(x, y, image::Rgba([255, 255, 255, 0]));
        if x > 0 {
            stack.push((x - 1, y));
        }
        if x + 1 < w {
            stack.push((x + 1, y));
        }
        if y > 0 {
            stack.push((x, y - 1));
        }
        if y + 1 < h {
            stack.push((x, y + 1));
        }
    }
}

/// Texture cache for the icons, one per egui context.
#[derive(Default)]
pub struct Icons {
    textures: Mutex<HashMap<Icon, egui::TextureHandle>>,
}

impl std::fmt::Debug for Icons {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Icons")
    }
}

impl Icons {
    /// The texture of an icon, loaded on first use.
    pub fn texture(&self, ctx: &egui::Context, icon: Icon) -> egui::TextureHandle {
        let mut map = self
            .textures
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        map.entry(icon)
            .or_insert_with(|| {
                ctx.load_texture(
                    format!("icon:{}", icon.name()),
                    icon.decode(),
                    egui::TextureOptions::LINEAR,
                )
            })
            .clone()
    }

    /// An image of the icon at `size` points.
    pub fn image(&self, ctx: &egui::Context, icon: Icon, size: f32) -> egui::Image<'static> {
        let tex = self.texture(ctx, icon);
        egui::Image::new(egui::load::SizedTexture::new(
            tex.id(),
            egui::vec2(size, size),
        ))
        .tint(icon.tint(&ctx.style_of(ctx.theme()).visuals))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_icon_decodes() {
        for icon in [
            Icon::About,
            Icon::Alert,
            Icon::ArrowOutLonger,
            Icon::BackwardBlue,
            Icon::Error,
            Icon::ForwardBlue,
            Icon::Images,
            Icon::NextBlue,
            Icon::PlayBlue,
            Icon::PlayBackwardBlue,
            Icon::PrevBlue,
            Icon::Settings,
            Icon::Shape3d,
            Icon::Shape3dNotSet,
            Icon::Shape3dOptimized,
            Icon::Shape3dSetMarker,
            Icon::Shape3dSetMarkerOptimized,
            Icon::StopBlue,
            Icon::Video,
            Icon::Banner,
        ] {
            let img = icon.decode();
            assert!(img.size[0] > 0, "{}", icon.name());
        }
    }

    /// The settings gear's white square is transparent; the gear itself is opaque.
    #[test]
    fn settings_icon_has_no_white_square() {
        let img = Icon::Settings.decode();
        let [w, h] = img.size;
        let at = |x: usize, y: usize| img.pixels[y * w + x];
        for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] {
            assert_eq!(at(x, y).a(), 0, "corner ({x}, {y})");
        }
        // The centre row crosses the gear: mostly opaque.
        let opaque = (0..w).filter(|&x| at(x, h / 2).a() == 255).count();
        assert!(opaque >= w / 2, "{opaque}");
        let transparent = img.pixels.iter().filter(|p| p.a() == 0).count();
        assert!(transparent > 0 && transparent < w * h / 2, "{transparent}");
    }

    /// The video icon is black strokes only: it is drawn in the theme's text colour, so it
    /// can be seen on the dark theme's buttons (black on #2d2d2d in XMALab 3).
    #[test]
    fn symbolic_icon_follows_the_text_colour() {
        let img = Icon::Video.decode();
        assert!(
            img.pixels
                .iter()
                .all(|p| p.a() == 0 || (p.r() == p.a() && p.g() == p.a()))
        );
        assert!(img.pixels.iter().any(|p| p.a() > 0));
        let dark = egui::Visuals::dark();
        let light = egui::Visuals::light();
        assert_eq!(Icon::Video.tint(&dark), dark.text_color());
        assert_eq!(Icon::Video.tint(&light), light.text_color());
        assert_ne!(Icon::Video.tint(&dark), Icon::Video.tint(&light));
        // Coloured icons keep their colours in both themes.
        for i in [Icon::Images, Icon::Settings, Icon::About] {
            assert_eq!(i.tint(&dark), egui::Color32::WHITE);
            assert_eq!(i.tint(&light), egui::Color32::WHITE);
        }
    }
}
