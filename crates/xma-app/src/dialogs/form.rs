//! Qt form widgets as the Toolbox pages and the dialogs use them: group boxes, styled
//! frames, full-width push buttons, icon tool buttons, labelled combo boxes, spin boxes,
//! colour swatches and the colour chooser, and a two-column tree (`QTreeWidget` without a
//! header). They are drawn with egui's own look: the frames are egui's group frame, the
//! colour chooser is egui's colour picker, colours and sizes come from the current style.
//!
//! Every interactive widget drawn here gets an accessible name (its text, or the name
//! passed in), as the `egui_kittest` harness requires.

use std::collections::BTreeSet;

use egui::{Color32, Response, Ui};

use crate::icons::{Icon, Icons};

/// `QGroupBox` with a title: the title on its own line, then egui's group frame holding
/// the contents. An empty title draws the frame only.
pub fn group_box<R>(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.vertical(|ui| {
        if !title.is_empty() {
            ui.label(title);
        }
        styled_frame(ui, add)
    })
    .inner
}

/// `QFrame` with the `StyledPanel` shape: egui's group frame, as wide as the space given.
pub fn styled_frame<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    egui::Frame::group(ui.style())
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            add(ui)
        })
        .inner
}

/// A `QPushButton` filling the available width.
pub fn wide_button(ui: &mut Ui, enabled: bool, text: &str) -> Response {
    let h = ui.spacing().interact_size.y;
    let w = ui.available_width();
    ui.add_enabled_ui(enabled, |ui| ui.add_sized([w, h], egui::Button::new(text)))
        .inner
}

/// A `QPushButton` of width `w`.
/// At least `w` wide; wider when the text needs it.
pub fn button(ui: &mut Ui, enabled: bool, text: &str, w: f32) -> Response {
    let h = ui.spacing().interact_size.y;
    let font = egui::TextStyle::Button.resolve(ui.style());
    let tw = ui
        .painter()
        .layout_no_wrap(text.to_owned(), font, Color32::BLACK)
        .size()
        .x
        + 2.0 * ui.spacing().button_padding.x;
    ui.add_enabled_ui(enabled, |ui| {
        ui.add_sized([w.max(tw), h], egui::Button::new(text))
    })
    .inner
}

/// A `QToolButton` showing an icon, `w` wide; `tip` is its tooltip and accessible name.
#[allow(clippy::too_many_arguments)]
pub fn icon_button(
    ui: &mut Ui,
    icons: &Icons,
    icon: Icon,
    enabled: bool,
    checked: bool,
    tip: &str,
    w: f32,
    auto_raise: bool,
) -> Response {
    let img = icons.image(ui.ctx(), icon, 16.0);
    let h = ui.spacing().interact_size.y;
    let mut b = egui::Button::image(img)
        .selected(checked)
        .min_size(egui::vec2(w, h));
    if auto_raise {
        b = b.frame_when_inactive(checked);
    }
    ui.add_enabled(enabled, b)
        .on_hover_text(tip)
        .on_disabled_hover_text(tip)
        .accessible_name(tip)
}

/// A check box, enabled or not, that shows `checked` and returns the new value when the
/// user changes it.
pub fn check_box(ui: &mut Ui, enabled: bool, checked: bool, text: &str) -> Option<bool> {
    let mut v = checked;
    let r = ui.add_enabled(enabled, egui::Checkbox::new(&mut v, text));
    (r.changed() && v != checked).then_some(v)
}

/// A radio button; returns true when clicked.
pub fn radio(ui: &mut Ui, enabled: bool, checked: bool, text: &str) -> bool {
    ui.add_enabled(enabled, egui::RadioButton::new(checked, text))
        .clicked()
}

/// A label followed by a combo box it names (Qt buddy); returns the newly chosen index.
pub fn labelled_combo(
    ui: &mut Ui,
    enabled: bool,
    label_id: egui::Id,
    id: &str,
    selected: usize,
    items: &[&str],
    width: Option<f32>,
) -> Option<usize> {
    ui.add_enabled_ui(enabled, |ui| {
        match width {
            // A fixed width: the shown item is truncated when it does not fit, as a Qt
            // combo box in a narrow dock elides it.
            Some(w) => {
                let text = items.get(selected).copied().unwrap_or("");
                let mut chosen = None;
                let r = egui::ComboBox::from_id_salt(id)
                    .selected_text(text)
                    .width(w)
                    .truncate()
                    .show_ui(ui, |ui| {
                        for (i, it) in items.iter().enumerate() {
                            if ui.selectable_label(i == selected, *it).clicked() && i != selected {
                                chosen = Some(i);
                            }
                        }
                    });
                let chosen = chosen.or_else(|| {
                    crate::qt::combo_wheel(ui, &r.response, Some(selected), items.len(), |_| true)
                });
                let _ = r.response.labelled_by(label_id);
                chosen
            }
            None => crate::qt::combo_box(ui, id, Some(selected), items, None, Some(label_id)),
        }
    })
    .inner
}

/// A `QSpinBox`; returns the new value when changed.
pub fn spin_box(
    ui: &mut Ui,
    enabled: bool,
    value: i64,
    min: i64,
    max: i64,
    width: f32,
    name: &str,
) -> Option<i64> {
    spin_box_step(ui, enabled, value, min, max, 1, width, name)
}

/// [`spin_box`] with `singleStep` `step` ([`crate::qt::spin_box_step`]).
#[allow(clippy::too_many_arguments)]
pub fn spin_box_step(
    ui: &mut Ui,
    enabled: bool,
    value: i64,
    min: i64,
    max: i64,
    step: i64,
    width: f32,
    name: &str,
) -> Option<i64> {
    let mut v = value;
    // `qt::spin_box` names its arrows after the enclosing scope, so two spin boxes in
    // sibling scopes would share ids: each gets its own id here.
    let changed = ui
        .push_id(name, |ui| {
            ui.add_enabled_ui(enabled, |ui| {
                crate::qt::spin_box_step(ui, &mut v, min, max, step, width, name)
            })
            .inner
        })
        .inner;
    changed.then_some(v)
}

/// A `QDoubleSpinBox` with `decimals` decimals; returns the new value when changed.
pub fn double_spin_box(
    ui: &mut Ui,
    enabled: bool,
    value: f64,
    range: std::ops::RangeInclusive<f64>,
    decimals: usize,
    width: f32,
    name: &str,
) -> Option<f64> {
    let mut v = value;
    let h = ui.spacing().interact_size.y;
    let r = ui
        .add_enabled_ui(enabled, |ui| {
            ui.add_sized(
                [width, h],
                egui::DragValue::new(&mut v)
                    .range(range)
                    .speed(0.01)
                    .fixed_decimals(decimals),
            )
            .accessible_name(name)
        })
        .inner;
    (r.changed() && v != value).then_some(v)
}

/// A colour swatch button (a `QToolButton` whose icon is filled with the colour); a click
/// opens the colour chooser ([`color_chooser`]). Drawn as an egui button holding the colour;
/// `name` is its accessible name.
pub fn swatch_button(ui: &mut Ui, color: Color32, name: &str) -> Response {
    let h = ui.spacing().interact_size.y;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(h * 2.0, h), egui::Sense::click());
    let vis = ui.style().interact(&resp);
    ui.painter().rect(
        rect.expand(vis.expansion),
        vis.corner_radius,
        vis.weak_bg_fill,
        vis.bg_stroke,
        egui::StrokeKind::Inside,
    );
    let sw = rect.shrink2(ui.spacing().button_padding.max(egui::vec2(3.0, 3.0)));
    egui::color_picker::show_color_at(ui.painter(), color, sw);
    let enabled = ui.is_enabled();
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, enabled, name));
    resp.on_hover_text("Click to choose a colour")
}

/// `#rrggbb`, as `QColor::name` writes it.
#[must_use]
pub fn hex_name(c: Color32) -> String {
    format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b())
}

/// Width of the colour chooser's contents: the slider width egui gives its own colour-picker
/// popup (`color_edit_button_*`). The "Select Color" windows are kept this wide, so that
/// their right-aligned OK and Cancel do not stretch them.
pub const CHOOSER_WIDTH: f32 = 275.0;

/// The body of the "Select Color" dialog (`QColorDialog`): egui's colour picker, then the
/// "HTML:" field (`#rrggbb`). `hex` is the field's text; both are kept in step.
pub fn color_chooser(ui: &mut Ui, color: &mut Color32, hex: &mut String) {
    ui.spacing_mut().slider_width = CHOOSER_WIDTH;
    if egui::color_picker::color_picker_color32(ui, color, egui::color_picker::Alpha::Opaque) {
        *hex = hex_name(*color);
    }
    ui.horizontal(|ui| {
        let l = ui.label("HTML:").id;
        if ui
            .add(egui::TextEdit::singleline(hex).desired_width(90.0))
            .labelled_by(l)
            .changed()
        {
            let t = hex.trim();
            if t.len() == 7 && t.starts_with('#') && t[1..].chars().all(|x| x.is_ascii_hexdigit()) {
                let p = |k: usize| u8::from_str_radix(&t[k..k + 2], 16).unwrap_or(0);
                *color = Color32::from_rgb(p(1), p(3), p(5));
            }
        }
    });
}

/// One row of a two-column tree: text in each column and the child rows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TreeItem {
    pub col0: String,
    pub col1: String,
    pub children: Vec<TreeItem>,
}

impl TreeItem {
    #[must_use]
    pub fn new(col0: impl Into<String>, col1: impl Into<String>) -> Self {
        Self {
            col0: col0.into(),
            col1: col1.into(),
            children: Vec::new(),
        }
    }

    /// Adds a child and returns it (`QTreeWidgetItem::addChild`).
    pub fn add(&mut self, child: TreeItem) -> &mut TreeItem {
        self.children.push(child);
        self.children.last_mut().expect("just pushed")
    }
}

/// The expanded items of a [`tree`], by path of child indices.
pub type Expanded = BTreeSet<Vec<usize>>;

/// The width of a [`tree`]'s column 0, kept with the tree's items so that it is measured
/// once, not on every frame: it depends only on the items and on the font and scale the
/// texts are laid out with, and is measured again when either changes. Whoever replaces
/// the items replaces this with a new (empty) one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TreeWidth {
    /// The font, the points-to-pixels scale and the width measured with them.
    measured: Option<(egui::FontId, f32, f32)>,
}

impl TreeWidth {
    /// Column 0's width for `items` drawn with `font` in `ui`, measured on the first call
    /// and again when the font or the scale changes.
    fn get(&mut self, ui: &Ui, font: &egui::FontId, items: &[TreeItem], indent: f32) -> f32 {
        let ppp = ui.ctx().pixels_per_point();
        if let Some((f, p, w)) = &self.measured
            && f == font
            && p.to_bits() == ppp.to_bits()
        {
            return *w;
        }
        let w = col0_width(ui, font, items, indent);
        self.measured = Some((font.clone(), ppp, w));
        w
    }
}

/// Column 0's width: its widest text with every item expanded, plus the indent of its depth
/// and a margin; at least 40.
fn col0_width(ui: &Ui, font: &egui::FontId, items: &[TreeItem], indent: f32) -> f32 {
    let mut col0_w: f32 = 40.0;
    fn widest(
        ui: &Ui,
        font: &egui::FontId,
        items: &[TreeItem],
        depth: usize,
        indent: f32,
        w: &mut f32,
    ) {
        for it in items {
            let tw = ui
                .painter()
                .layout_no_wrap(it.col0.clone(), font.clone(), Color32::BLACK)
                .size()
                .x;
            *w = w.max(tw + indent * (depth as f32 + 1.0) + 8.0);
            widest(ui, font, &it.children, depth + 1, indent, w);
        }
    }
    widest(ui, font, items, 0, indent, &mut col0_w);
    col0_w
}

/// A `QTreeWidget` with two columns and no header. Column 0 is as wide as its widest text
/// with every item expanded (`expandAll`, `resizeColumnToContents(0)`, `collapseAll`, as
/// the C++ forms do after filling the tree), measured once and kept in `width`; items start
/// collapsed. A click on the arrow or the row expands or collapses the item.
pub fn tree(
    ui: &mut Ui,
    id: &str,
    items: &[TreeItem],
    expanded: &mut Expanded,
    width: &mut TreeWidth,
) {
    let font = egui::TextStyle::Body.resolve(ui.style());
    let indent = 18.0;
    let col0_w = width.get(ui, &font, items, indent);
    egui::Frame::NONE
        .fill(ui.visuals().extreme_bg_color)
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .inner_margin(ui.spacing().button_padding.x)
        .show(ui, |ui| {
            egui::ScrollArea::both()
                .id_salt(id)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 0.0;
                    let mut path = Vec::new();
                    rows(ui, items, 0, &mut path, expanded, col0_w, indent);
                });
        });
}

fn rows(
    ui: &mut Ui,
    items: &[TreeItem],
    depth: usize,
    path: &mut Vec<usize>,
    expanded: &mut Expanded,
    col0_w: f32,
    indent: f32,
) {
    for (i, it) in items.iter().enumerate() {
        path.push(i);
        let open = expanded.contains(path);
        let h = ui.spacing().interact_size.y + 2.0;
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width().max(col0_w + 200.0), h),
            if it.children.is_empty() {
                egui::Sense::hover()
            } else {
                egui::Sense::click()
            },
        );
        let label = if it.col1.is_empty() {
            it.col0.clone()
        } else {
            format!("{}: {}", it.col0, it.col1)
        };
        resp.widget_info(|| {
            if it.children.is_empty() {
                egui::WidgetInfo::labeled(egui::Role::Label, true, &label)
            } else {
                egui::WidgetInfo::labeled(egui::Role::TreeItem, true, &label)
            }
        });
        if resp.clicked() {
            if open {
                expanded.remove(path);
            } else {
                expanded.insert(path.clone());
            }
        }
        let p = ui.painter();
        let text_color = ui.visuals().text_color();
        let x0 = rect.min.x + indent * depth as f32;
        if !it.children.is_empty() {
            let c = egui::pos2(x0 + indent / 2.0, rect.center().y);
            let s = 4.0;
            let pts = if open {
                vec![
                    egui::pos2(c.x - s, c.y - s * 0.5),
                    egui::pos2(c.x + s, c.y - s * 0.5),
                    egui::pos2(c.x, c.y + s * 0.6),
                ]
            } else {
                vec![
                    egui::pos2(c.x - s * 0.5, c.y - s),
                    egui::pos2(c.x - s * 0.5, c.y + s),
                    egui::pos2(c.x + s * 0.6, c.y),
                ]
            };
            p.add(egui::Shape::convex_polygon(
                pts,
                ui.visuals().weak_text_color(),
                egui::Stroke::NONE,
            ));
        }
        let font = egui::TextStyle::Body.resolve(ui.style());
        p.text(
            egui::pos2(x0 + indent, rect.center().y),
            egui::Align2::LEFT_CENTER,
            &it.col0,
            font.clone(),
            text_color,
        );
        p.text(
            egui::pos2(rect.min.x + col0_w, rect.center().y),
            egui::Align2::LEFT_CENTER,
            &it.col1,
            font,
            text_color,
        );
        if open {
            rows(ui, &it.children, depth + 1, path, expanded, col0_w, indent);
        }
        path.pop();
    }
}

/// `QString::number(double)`: `%g` with six significant digits.
#[must_use]
pub fn qnum(v: f64) -> String {
    crate::widgets::plot::ticks::format_g(v, 6)
}

/// The icon and text row of a message dialog (`ErrorDialog`, `ConfirmationDialog`): the
/// 48-point `icon`, then `message`.
pub fn message_row(ui: &mut Ui, icons: &Icons, icon: Icon, message: &str) {
    ui.horizontal(|ui| {
        ui.add(icons.image(ui.ctx(), icon, 48.0));
        ui.label(message);
    });
}

/// The rejecting and accepting buttons of a small dialog, right-aligned, the rejecting one
/// rightmost. Returns `Some(true)` when `accept` is clicked, `Some(false)` for `reject`.
pub fn reject_accept(ui: &mut Ui, accept: &str, reject: &str) -> Option<bool> {
    let mut out = None;
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        if ui.button(reject).clicked() {
            out = Some(false);
        }
        if ui.button(accept).clicked() {
            out = Some(true);
        }
    });
    out
}

/// A modal dialog drawn by [`modal`], with whether Escape was pressed in this frame, read
/// before egui's `ModalResponse::should_close` consumes the key.
pub struct ModalShown<R> {
    pub response: egui::ModalResponse<R>,
    /// Escape was pressed (read before the dialog was drawn).
    pub escape: bool,
}

impl<R> ModalShown<R> {
    /// Escape closes the dialog (Qt's `reject()`): `should_close()`, which also consumes
    /// the key, and Escape pressed. A click outside does nothing, as with a Qt modal dialog
    /// (`should_close` alone is also true for a click on the backdrop).
    pub fn escaped(&self) -> bool {
        self.response.should_close() && self.escape
    }
}

/// Shows `add` as an egui modal dialog with the id `id`, reading Escape first.
pub fn modal<R>(
    ctx: &egui::Context,
    id: egui::Id,
    add: impl FnOnce(&mut Ui) -> R,
) -> ModalShown<R> {
    let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));
    let response = egui::Modal::new(id).show(ctx, add);
    ModalShown { response, escape }
}

/// The title row of a modal dialog: the window title and a close button (the title bar the
/// operating system draws for a Qt dialog). Returns true when the close button is clicked.
pub fn dialog_title(ui: &mut Ui, title: &str) -> bool {
    let mut close = false;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(title).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::click());
            let name = format!("Close {title}");
            resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, &name));
            let vis = ui.style().interact(&resp);
            if resp.hovered() {
                ui.painter().rect_filled(rect, 2.0, vis.weak_bg_fill);
            }
            let r = rect.shrink(5.0);
            let st = egui::Stroke::new(1.5, vis.fg_stroke.color);
            ui.painter()
                .line_segment([r.left_top(), r.right_bottom()], st);
            ui.painter()
                .line_segment([r.right_top(), r.left_bottom()], st);
            if resp.clicked() {
                close = true;
            }
        });
    });
    ui.add_space(4.0);
    close
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items() -> Vec<TreeItem> {
        let mut root = TreeItem::new("Trial", "Trial 1");
        let m = root.add(TreeItem::new("Marker", "a"));
        m.add(TreeItem::new("mean reprojection error", "NA"));
        m.add(TreeItem::new("a much longer label of the third level", ""));
        vec![root, TreeItem::new("Repository", "")]
    }

    /// One frame; `f` gets the width kept in `width` and the width measured afresh in the
    /// same frame, as the code before 21 November 2026 measured it on every frame.
    fn frame(ctx: &egui::Context, items: &[TreeItem], width: &mut TreeWidth) -> (f32, f32) {
        let mut got = (0.0, 0.0);
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
            let font = egui::TextStyle::Body.resolve(ui.style());
            got = (
                width.get(ui, &font, items, 18.0),
                col0_width(ui, &font, items, 18.0),
            );
        });
        out.textures_delta.clear();
        got
    }

    #[test]
    fn review13_tree_width_kept_is_the_width_measured() {
        let ctx = egui::Context::default();
        let items = items();
        let mut width = TreeWidth::default();
        let (kept, fresh) = frame(&ctx, &items, &mut width);
        assert_eq!(kept.to_bits(), fresh.to_bits());
        assert!(kept > 40.0);
        for _ in 0..3 {
            let (k, f) = frame(&ctx, &items, &mut width);
            assert_eq!(k.to_bits(), kept.to_bits());
            assert_eq!(f.to_bits(), kept.to_bits());
        }
        // Another scale: measured again.
        ctx.set_pixels_per_point(1.75);
        let (k, f) = frame(&ctx, &items, &mut width);
        assert_eq!(k.to_bits(), f.to_bits());
        // Another font size: measured again.
        ctx.all_styles_mut(|s| {
            s.text_styles
                .insert(egui::TextStyle::Body, egui::FontId::proportional(21.0));
        });
        let (k, f) = frame(&ctx, &items, &mut width);
        assert_eq!(k.to_bits(), f.to_bits());
        assert!(k > kept);
        // No items: the minimum.
        let (k, f) = frame(&ctx, &[], &mut TreeWidth::default());
        assert_eq!((k, f), (40.0, 40.0));
    }
}
