use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Add;
use std::process::id;
use generational_arena::{Arena, Index};
use xrpl_consensus_core::{Ledger, LedgerIndex};
use crate::ledger_trie::LedgerTrie;
use crate::span::{Span, SpanTip};

pub struct Node<T: Ledger> {
    idx: Option<Index>,
    span: Option<Span<T>>,
    tip_support: u32,
    branch_support: u32,
    children: Vec<Index>,
    parent: Option<Index>
}

impl<T: Ledger> Node<T> {
    pub fn new(ledger: T) -> Self {
        Node {
            idx: None,
            span: Some(Span::new(ledger)),
            tip_support: 1,
            branch_support: 1,
            children: vec![],
            parent: None,
        }
    }

    pub fn with_index(idx: Index) -> Self {
        Node {
            idx: Some(idx),
            span: None,
            tip_support: 0,
            branch_support: 0,
            children: vec![],
            parent: None,
        }
    }

    pub fn from_span(span: Span<T>, idx: Index) -> Self {
        Node {
            idx: Some(idx),
            span: Some(span),
            tip_support: 0,
            branch_support: 0,
            children: vec![],
            parent: None,
        }
    }

}

pub struct ArenaLedgerTrie<T: Ledger> {
    arena: Arena<Node<T>>,
    seq_support: HashMap<LedgerIndex, u32>,
}

impl<T: Ledger> LedgerTrie<T> for ArenaLedgerTrie<T> {

    fn insert(&mut self, ledger: &T, count: Option<u32>) {
        // Find the ID of the node with the longest common ancestry with `ledger`
        // and the sequence of the first ledger difference
        let (loc_idx, diff_seq) = self.find(ledger);

        let mut inc_node_idx = Some(loc_idx);

        // Insert a new, basically empty, Node and save its Index.
        let new_node_idx = self.arena.insert_with(|idx| {
            let new_node = Node::with_index(idx);
            new_node
        });

        // Get a mutable reference to both the loc node and new node we inserted above.
        // We have to do it this way because we need a mutable reference to both, but
        // cannot cannot call self.arena.get_mut twice without having two simultaneous
        // mutable borrows of self.arena, which would break Rust's ownership rules.
        let (loc, new_node) = self.arena.get2_mut(loc_idx, new_node_idx);


        // We know they exist because we got them in self.find and we just inserted new_node
        let loc = loc.unwrap();
        let new_node = new_node.unwrap();

        // loc->span has the longest common prefix with Span{ledger} of all
        // existing nodes in the trie. The optional<Span>'s below represent
        // the possible common suffixes between loc->span and Span{ledger}.
        //
        // loc->span
        //  a b c  | d e f
        //  prefix | oldSuffix
        //
        // Span{ledger}
        //  a b c  | g h i
        //  prefix | newSuffix
        let prefix = loc.span.as_ref().unwrap().before(diff_seq);
        let old_suffix = loc.span.as_ref().unwrap().from(diff_seq);
        let new_suffix = Span::new(*ledger).from(diff_seq);

        if let Some(old_suffix) = old_suffix {
            // Have
            //   abcdef -> ....
            // Inserting
            //   abc
            // Becomes
            //   abc -> def -> ...

            // Set new_node's span to old_suffix and take tip_support and branch_support
            // from loc so that new_node takes over loc. new_node will be loc's child.
            new_node.span = Some(old_suffix);
            new_node.tip_support = loc.tip_support;
            new_node.branch_support = loc.branch_support;
            new_node.parent = loc.idx;

            // Replace loc's children Vec with an empty vector because we will move
            // loc's children into new_node's children. However, we need to clone
            // the children Vec into new_node.children because we later need to
            // iterate through the children, get a mutable reference to the Node
            // the child Index points to and update each child Node's parent idx to
            // point to new_node. If we simply moved loc.children into new_node.children,
            // we'd need to keep the mutable reference to new_node alive which would
            // prevent us from getting mutable references to each child Node.
            let loc_children = std::mem::replace(&mut loc.children, vec![]);
            new_node.children = loc_children.clone();

            // loc truncates to prefix and new_node is its child
            loc.span = prefix;
            loc.children.push(new_node_idx);
            loc.tip_support = 0;

            // Update each child node's parent field to point to new_node.
            loc_children.iter()
                .for_each(|child_idx| {
                    self.arena.get_mut(*child_idx).unwrap().parent = Some(new_node_idx)
                })
        } else if let Some(new_suffix) = new_suffix {
            // Have
            //  abc -> ...
            // Inserting
            //  abcdef-> ...
            // Becomes
            //  abc -> ...
            //     \-> def

            new_node.parent = loc.idx;
            inc_node_idx = Some(new_node_idx);
            loc.children.push(new_node_idx);
        }

        let count = count.unwrap_or(1);
        self.arena.get_mut(inc_node_idx.unwrap()).unwrap().tip_support += 1;
        while inc_node_idx.is_some() {
            let inc_node = self.arena.get_mut(inc_node_idx.unwrap()).unwrap();
            inc_node.branch_support += count;
            inc_node_idx = inc_node.parent;
        }

        // Update seq support by adding count, or insert a new entry
        match self.seq_support.entry(ledger.seq()) {
            Entry::Occupied(mut entry) => {
                *entry.get_mut() += count;
            }
            Entry::Vacant(entry) => {
                entry.insert(count);
            }
        }
    }

    fn get_preferred(&self, largest_issued: LedgerIndex) -> Option<SpanTip<T>> {
        todo!()
    }
}

impl<T: Ledger> ArenaLedgerTrie<T> {
    pub fn find(&self, ledger: &T) -> (Index, LedgerIndex) {
        todo!()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foo() {
        let mut map = HashMap::<u32, u32>::new();
        match map.entry(1) {
            Entry::Occupied(mut e) => {
                *e.get_mut() += 2;
            }
            Entry::Vacant(e) => {
                e.insert(2);
            }
        };

        println!("map: {:?}", map);
        match map.entry(1) {
            Entry::Occupied(mut e) => {
                *e.get_mut() += 2;
            }
            Entry::Vacant(e) => {
                e.insert(2);
            }
        };

        println!("map: {:?}", map)
    }
}

