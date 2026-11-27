//! The Project Metadata window ("Project Metadata").
//!
//! Ported from `MetaDataInfo.ui` and ui/MetaDataInfo.cpp: `treeWidget`, a two-column tree of
//! the project's XMA Portal data (`update`, built by the engine's
//! [`xma_engine::portal_summary::metadata_tree`]), and `pushButtonAdd` "Add metadata from
//! XMAPortal file", whose questions and edit are [`super::portal_metadata`].

use super::form::TreeItem;
use xma_engine::portal_summary::{MetaNode, metadata_tree};

fn item(n: &MetaNode) -> TreeItem {
    let mut it = TreeItem::new(n.label.clone(), n.value.clone());
    for c in &n.children {
        it.add(item(c));
    }
    it
}

/// Ported from ui/MetaDataInfo.cpp, MetaDataInfo::update: the tree's items.
#[must_use]
pub fn tree(doc: &xma_engine::Document) -> Vec<TreeItem> {
    metadata_tree(doc.project()).iter().map(item).collect()
}
