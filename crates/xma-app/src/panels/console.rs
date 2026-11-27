//! The Console panel.
//!
//! Ported from `ConsoleDockWidget.ui` and ui/ConsoleDockWidget.cpp. The form has one
//! control, `console` (a `QTextEdit` filling the panel) that shows the application log
//! ([`crate::log::AppLog`]): one paragraph per message in the colour of its level, scrolled
//! to the end whenever a message arrives (`writeLog`). The text can be selected and copied,
//! not edited. Closing the panel turns the `Console` setting off
//! (`ConsoleDockWidget::closeEvent` calls `on_actionConsole_triggered(false)`), which the
//! shell does for this panel's close button.

use super::{Cx, Panel, PanelId};

/// The Console panel.
#[derive(Debug, Default)]
pub struct ConsolePanel {
    /// Log generation last drawn, to scroll to the end when it changes.
    seen_generation: u64,
}

impl Panel for ConsolePanel {
    fn id(&self) -> PanelId {
        PanelId::Console
    }

    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut Cx<'_>) {
        let generation = cx.log.generation();
        let scroll_to_end = generation != self.seen_generation;
        self.seen_generation = generation;
        // QTextEdit: a sunken white (Base) box.
        egui::Frame::NONE
            .fill(ui.visuals().extreme_bg_color)
            .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
            .inner_margin(egui::Margin::same(3))
            .show(ui, |ui| {
                ui.set_min_size(ui.available_size());
                let visuals = ui.visuals().clone();
                // Copied out under the log's lock and drawn without it, so a thread writing
                // to the log does not wait for the layout. Each label takes its entry's text.
                let entries = cx.log.entries();
                egui::ScrollArea::both()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.style_mut().interaction.selectable_labels = true;
                        ui.spacing_mut().item_spacing.y = 0.0;
                        for e in entries {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(e.text).color(e.level.color_on(&visuals)),
                                )
                                .wrap_mode(egui::TextWrapMode::Extend),
                            );
                        }
                        if scroll_to_end {
                            ui.scroll_to_cursor(Some(egui::Align::BOTTOM));
                        }
                    });
            });
    }
}
