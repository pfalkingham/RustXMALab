//! What a docking tree shows, without how it is stored.
//!
//! Two trees can show the same thing and differ in sizes, in where hidden panels are kept,
//! and in containers that show nothing or hold a single child. [`shown`] reduces a tree to
//! the visible panes and how they are split and stacked, so the shell can tell a drop that
//! moved a panel from one that left every panel where it was (a panel let go over a half of
//! itself, or next to itself). The docking crate carries out both the same way: it takes the
//! pane out of its tab group and puts it in a new container beside that group.

use egui_tiles::{Container, LinearDir, Tile, TileId, Tree};

use super::DockPane;

/// The visible part of a docking tree.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Shown {
    Pane(DockPane),
    /// Stacked panes with a tab strip, in tab order, and the open tab.
    Tabs(Vec<Shown>, Option<DockPane>),
    /// Side by side (horizontal) or one above the other (vertical), in order.
    Split(LinearDir, Vec<Shown>),
    Grid(Vec<Shown>),
}

/// What `tree` shows, `None` when it shows nothing.
///
/// A container that shows one thing counts as that thing, a split inside a split of the
/// same direction and a tab group inside a tab group count as one, and hidden panes and
/// empty containers are left out: what the docking crate's simplification does, and what
/// the window looks like.
pub(super) fn shown(tree: &Tree<DockPane>) -> Option<Shown> {
    node(tree, tree.root?)
}

fn node(tree: &Tree<DockPane>, id: TileId) -> Option<Shown> {
    if !tree.tiles.is_visible(id) {
        return None;
    }
    let container = match tree.tiles.get(id)? {
        Tile::Pane(p) => return Some(Shown::Pane(*p)),
        Tile::Container(c) => c,
    };
    let children = container.children().filter_map(|c| node(tree, *c));
    let mut out = Vec::new();
    match container {
        Container::Tabs(tabs) => {
            // The open tab, when it is one of the visible panes.
            let mut active = tabs
                .active
                .filter(|a| tree.tiles.is_visible(*a))
                .and_then(|a| tree.tiles.get_pane(&a).copied());
            for c in children {
                match c {
                    Shown::Tabs(inner, inner_active) => {
                        active = active.or(inner_active);
                        out.extend(inner);
                    }
                    other => out.push(other),
                }
            }
            one_or(out, |v| Shown::Tabs(v, active))
        }
        Container::Linear(l) => {
            for c in children {
                match c {
                    Shown::Split(dir, inner) if dir == l.dir => out.extend(inner),
                    other => out.push(other),
                }
            }
            one_or(out, |v| Shown::Split(l.dir, v))
        }
        Container::Grid(_) => {
            out.extend(children);
            one_or(out, Shown::Grid)
        }
    }
}

/// Nothing, the only member itself, or the members joined by `join`.
fn one_or(mut members: Vec<Shown>, join: impl FnOnce(Vec<Shown>) -> Shown) -> Option<Shown> {
    match members.len() {
        0 => None,
        1 => members.pop(),
        _ => Some(join(members)),
    }
}

#[cfg(test)]
mod tests {
    use egui_tiles::Tiles;

    use super::*;
    use crate::panels::PanelId;

    fn pane(p: PanelId) -> Shown {
        Shown::Pane(DockPane::Panel(p))
    }

    /// Points alone in a tab group above a hidden 3D world view, beside the camera views.
    /// With `dropped`, as the docking crate leaves it when Points is let go over the lower
    /// half of its own group: the group emptied, in a new container with the pane below it.
    fn small(dropped: bool) -> Tree<DockPane> {
        let mut t: Tiles<DockPane> = Tiles::default();
        let points = t.insert_pane(DockPane::Panel(PanelId::Points));
        let group = if dropped {
            let emptied = t.insert_tab_tile(vec![]);
            t.insert_vertical_tile(vec![emptied, points])
        } else {
            t.insert_tab_tile(vec![points])
        };
        let world = t.insert_pane(DockPane::Panel(PanelId::WorldView));
        let world_group = t.insert_tab_tile(vec![world]);
        t.set_visible(world, false);
        let left = t.insert_vertical_tile(vec![group, world_group]);
        let central = t.insert_pane(DockPane::Central);
        let central_group = t.insert_tab_tile(vec![central]);
        let root = t.insert_horizontal_tile(vec![left, central_group]);
        Tree::new(egui::Id::unique("t"), root, t)
    }

    #[test]
    fn hidden_panes_and_lone_containers_are_left_out() {
        assert_eq!(
            shown(&small(false)),
            Some(Shown::Split(
                LinearDir::Horizontal,
                vec![pane(PanelId::Points), Shown::Pane(DockPane::Central)]
            ))
        );
    }

    #[test]
    fn a_pane_beside_its_emptied_group_shows_the_same() {
        assert_eq!(shown(&small(true)), shown(&small(false)));
    }

    #[test]
    fn tab_order_and_the_open_tab_count() {
        let mut t: Tiles<DockPane> = Tiles::default();
        let a = t.insert_pane(DockPane::Panel(PanelId::DetailedView));
        let b = t.insert_pane(DockPane::Panel(PanelId::Events));
        let group = t.insert_tab_tile(vec![a, b]);
        let mut tree = Tree::new(egui::Id::unique("t"), group, t);
        let first = shown(&tree);
        tree.make_active(|id, _| id == b);
        let second = shown(&tree);
        assert_ne!(first, second);
        let Some(Tile::Container(Container::Tabs(tabs))) = tree.tiles.get_mut(group) else {
            panic!("group")
        };
        tabs.children.reverse();
        assert_ne!(shown(&tree), second);
    }
}
