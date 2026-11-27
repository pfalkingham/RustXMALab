//! The rows of the points tree (the Points panel's `MarkerTreeWidget`).
//!
//! Ported from ui/PointsDockWidget.cpp, `PointsDockWidget::reloadListFromObject` and
//! `updateColor`, and ui/MarkerTreeWidgetButton.cpp, `setButtonIcon`: the C++ builds the
//! tree inside the widget; here the rows are a plain query over a [`Trial`], so the
//! interface draws them and tests check them without a window.
//!
//! What the C++ builds:
//!
//! - every marker as a top-level row, text `"<n>"` (1-based) and its description;
//! - then, for each rigid body in order, a row `"RB<n>"` with its description, inserted
//!   above the markers (so the bodies come first, in body order), and each marker the body
//!   lists moved under it, in the order the body lists them;
//! - the markers no body takes stay top-level, after the bodies, in index order.
//!
//! One difference, a fixed C++ defect: a marker listed by two bodies appears under each
//! body here. The C++ finds the marker's row by its text and can only move a top-level row,
//! so a marker already under an earlier body is left there and the later body shows
//! without it (rv_vpchain's third and fourth bodies show no markers at all). A marker
//! listed twice by the same body, or an index past the last marker, is shown once or not
//! at all, as in the C++.

use xma_model::{MarkerStatus, RigidBody, Trial};

/// What a row stands for: a rigid body or a marker, by index in the trial (0-based).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TreeItem {
    /// C++ item type `RIGID_BODY`.
    Body(usize),
    /// C++ item type `MARKER`.
    Marker(usize),
}

/// The icon of a body row's settings button, from the reference state of the body.
///
/// Ported from ui/MarkerTreeWidgetButton.cpp, MarkerTreeWidgetButton::setButtonIcon. The
/// names are the icon files.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BodyIcon {
    /// `shape_3d_notSet.png`: no reference (`isReferencesSet()` neither 1 nor 2).
    NotSet,
    /// `shape_3d.png`: references set (2), not optimised.
    Set,
    /// `shape_3d_optimized.png`: references set (2) and optimised.
    SetOptimized,
    /// `shape_3d_setMarker.png`: references from the markers (1), not optimised.
    SetMarker,
    /// `shape_3d_setMarker_optimized.png`: references from the markers (1), optimised.
    SetMarkerOptimized,
}

impl BodyIcon {
    /// Ported from ui/MarkerTreeWidgetButton.cpp, MarkerTreeWidgetButton::setButtonIcon.
    #[must_use]
    pub fn of(body: &RigidBody) -> Self {
        match (body.references_set, body.has_optimized_coordinates) {
            (2, false) => BodyIcon::Set,
            (2, true) => BodyIcon::SetOptimized,
            (1, false) => BodyIcon::SetMarker,
            (1, true) => BodyIcon::SetMarkerOptimized,
            _ => BodyIcon::NotSet,
        }
    }
}

/// One row of the tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeRow {
    pub item: TreeItem,
    /// The body a marker row sits under; `None` for a body row and a free marker.
    pub parent: Option<usize>,
    /// Column 0: `"RB<n>"` for a body, `"<n>"` for a marker (both 1-based).
    pub id_text: String,
    /// Column 1: the description.
    pub description: String,
    /// For a body row: the icon of its settings button. Marker rows show `settings.png`.
    pub body_icon: Option<BodyIcon>,
    /// For a body row: how many marker rows sit under it (the expander is drawn only when
    /// there are some, as Qt does).
    pub n_children: usize,
}

impl TreeRow {
    /// The row's depth: 1 for a marker under a body, 0 otherwise.
    #[must_use]
    pub fn depth(&self) -> usize {
        usize::from(self.parent.is_some())
    }
}

/// Every row of the tree in display order (each body followed by its markers, then the
/// free markers), whether or not the bodies are expanded.
///
/// Ported from ui/PointsDockWidget.cpp, PointsDockWidget::reloadListFromObject, with the
/// fix described in the module documentation.
#[must_use]
pub fn tree_rows(trial: &Trial) -> Vec<TreeRow> {
    let n_markers = trial.markers.len();
    let mut in_body = vec![false; n_markers];
    let mut rows = Vec::with_capacity(n_markers + trial.rigid_bodies.len());
    for (b, body) in trial.rigid_bodies.iter().enumerate() {
        let mut children: Vec<usize> = Vec::with_capacity(body.marker_indices.len());
        for &m in &body.marker_indices {
            if m < n_markers && !children.contains(&m) {
                children.push(m);
            }
        }
        rows.push(TreeRow {
            item: TreeItem::Body(b),
            parent: None,
            id_text: format!("RB{}", b + 1),
            description: body.description.clone(),
            body_icon: Some(BodyIcon::of(body)),
            n_children: children.len(),
        });
        for m in children {
            in_body[m] = true;
            rows.push(marker_row(trial, m, Some(b)));
        }
    }
    for (m, taken) in in_body.iter().enumerate() {
        if !taken {
            rows.push(marker_row(trial, m, None));
        }
    }
    rows
}

fn marker_row(trial: &Trial, m: usize, parent: Option<usize>) -> TreeRow {
    TreeRow {
        item: TreeItem::Marker(m),
        parent,
        id_text: (m + 1).to_string(),
        description: trial.markers[m].description.clone(),
        body_icon: None,
        n_children: 0,
    }
}

/// The rows shown: every row except the markers of a collapsed body. `expanded(b)` says
/// whether body `b` is expanded (`RigidBody::isExpanded` in the C++, interface state here).
#[must_use]
pub fn visible_rows(rows: &[TreeRow], expanded: impl Fn(usize) -> bool) -> Vec<&TreeRow> {
    visible_row_indices(rows, expanded)
        .into_iter()
        .map(|i| &rows[i])
        .collect()
}

/// [`visible_rows`] as indices into `rows`.
#[must_use]
pub fn visible_row_indices(rows: &[TreeRow], expanded: impl Fn(usize) -> bool) -> Vec<usize> {
    (0..rows.len())
        .filter(|&i| rows[i].parent.is_none_or(&expanded))
        .collect()
}

/// The first body whose row holds `marker`: the row the C++ finds and makes current when it
/// selects the marker (`PointsDockWidget::selectPoint`), expanding that body.
#[must_use]
pub fn first_body_of(rows: &[TreeRow], marker: usize) -> Option<usize> {
    rows.iter()
        .find(|r| r.item == TreeItem::Marker(marker))
        .and_then(|r| r.parent)
}

/// The status of a marker in each camera at `frame`, the colour swatches the tree shows
/// with "Show marker status" on (`PointsDockWidget::updateColor`, `Marker::getStatusColor`).
/// A frame past the end of a camera's data gives [`MarkerStatus::Undefined`].
#[must_use]
pub fn marker_statuses(trial: &Trial, marker: usize, frame: usize) -> Vec<MarkerStatus> {
    marker_statuses_iter(trial, marker, frame).collect()
}

/// [`marker_statuses`] without collecting them.
pub fn marker_statuses_iter(
    trial: &Trial,
    marker: usize,
    frame: usize,
) -> impl Iterator<Item = MarkerStatus> + '_ {
    trial
        .markers
        .get(marker)
        .into_iter()
        .flat_map(|m| m.views.iter())
        .map(move |v| v.status.get(frame).copied().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use xma_model::Marker;

    fn trial(n_markers: usize, bodies: &[&[usize]]) -> Trial {
        let mut t = Trial::default_trial();
        t.markers = (0..n_markers)
            .map(|i| {
                let mut m = Marker::new(2, 3, 8);
                m.description = format!("m{i}");
                Arc::new(m)
            })
            .collect();
        t.rigid_bodies = bodies
            .iter()
            .map(|idx| {
                let mut b = RigidBody::new(3);
                b.marker_indices = idx.to_vec();
                Arc::new(b)
            })
            .collect();
        t
    }

    fn texts(rows: &[TreeRow]) -> Vec<String> {
        rows.iter()
            .map(|r| format!("{}{}", "  ".repeat(r.depth()), r.id_text))
            .collect()
    }

    #[test]
    fn bodies_first_then_free_markers() {
        let t = trial(6, &[&[2, 0], &[4]]);
        let rows = tree_rows(&t);
        assert_eq!(
            texts(&rows),
            ["RB1", "  3", "  1", "RB2", "  5", "2", "4", "6"]
        );
        assert_eq!(rows[1].description, "m2");
        assert_eq!(rows[0].n_children, 2);
        assert_eq!(rows[0].body_icon, Some(BodyIcon::NotSet));
    }

    #[test]
    fn shared_markers_shown_under_each_body_and_bad_indices_dropped() {
        let t = trial(3, &[&[0, 1], &[1, 1, 7], &[]]);
        let rows = tree_rows(&t);
        assert_eq!(
            texts(&rows),
            ["RB1", "  1", "  2", "RB2", "  2", "RB3", "3"]
        );
        assert_eq!(first_body_of(&rows, 1), Some(0));
        assert_eq!(first_body_of(&rows, 2), None);
        let vis = visible_rows(&rows, |b| b == 1);
        let v: Vec<_> = vis.iter().map(|r| r.id_text.as_str()).collect();
        assert_eq!(v, ["RB1", "RB2", "2", "RB3", "3"]);
    }

    #[test]
    fn body_icons() {
        let mut b = RigidBody::new(1);
        for (set, opt, icon) in [
            (0, false, BodyIcon::NotSet),
            (0, true, BodyIcon::NotSet),
            (1, false, BodyIcon::SetMarker),
            (1, true, BodyIcon::SetMarkerOptimized),
            (2, false, BodyIcon::Set),
            (2, true, BodyIcon::SetOptimized),
            (3, false, BodyIcon::NotSet),
        ] {
            b.references_set = set;
            b.has_optimized_coordinates = opt;
            assert_eq!(BodyIcon::of(&b), icon);
        }
    }

    #[test]
    fn status_colours_like_cpp() {
        let t = trial(1, &[]);
        assert_eq!(marker_statuses(&t, 0, 99), vec![MarkerStatus::Undefined; 2]);
        assert!(marker_statuses(&t, 5, 0).is_empty());
    }
}
