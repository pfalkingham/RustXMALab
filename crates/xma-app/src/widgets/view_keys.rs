//! What a view tells whoever turns key presses into commands: that the keyboard focus it
//! holds is not a text field's.
//!
//! A text field that has the keyboard keeps the plain keys and the arrows (the
//! application's shortcuts leave them to it, `actions::collect_shortcuts`). A view that
//! takes the focus for a key of its own (the plot, for Delete and Backspace) must not: it
//! says so here in each frame it has the focus, and the application's keys go on working.
//!
//! The mark lives in the context's memory, one per window, as egui's focus is (a floating
//! panel is a window of its own).

use egui::{Context, Id};

fn focus(ctx: &Context) -> Id {
    Id::unique(("xma view keys: focused view", ctx.viewport_id()))
}

/// A view that holds the keyboard focus for keys of its own calls this, with its id, in
/// each frame it has the focus.
pub fn mark_focus(ctx: &Context, id: Id) {
    // The key first: it reads the context, which must not be locked then.
    let key = focus(ctx);
    ctx.data_mut(|d| d.insert_temp(key, id));
}

/// Whether the widget that has the keyboard focus in this window is a view (and so not a
/// text field).
#[must_use]
pub fn view_has_focus(ctx: &Context) -> bool {
    let focused = ctx.memory(|m| m.focused());
    let key = focus(ctx);
    focused.is_some() && focused == ctx.data(|d| d.get_temp::<Id>(key))
}
