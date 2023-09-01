use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::DerefMut;
use std::rc::Rc;
use xrpl_consensus_core::{Ledger, LedgerIndex};
use crate::ledger_trie::LedgerTrie;
use crate::span::{Span, SpanTip};

pub type NodePointer<T> = Rc<RefCell<Node<T>>>;

pub struct Node<T: Ledger> {
    span: Span<T>,
    tip_support: u32,
    branch_support: u32,
    children: Vec<NodePointer<T>>,
    parent: Option<NodePointer<T>>,
}

impl<T: Ledger> Default for Node<T> {
    fn default() -> Self {
        Node {
            span: Span::default(),
            tip_support: 0,
            branch_support: 0,
            children: vec![],
            parent: None,
        }
    }
}

impl<T: Ledger> Node<T> {
    pub fn new(ledger: T) -> Self {
        Node {
            span: Span::new(ledger),
            tip_support: 1,
            branch_support: 1,
            children: vec![],
            parent: None,
        }
    }

    pub fn from_span(
        span: Span<T>,
        tip_support: u32,
        branch_support: u32,
        children: Vec<NodePointer<T>>,
        parent: Option<NodePointer<T>>
    ) -> Self {
        Node {
            span,
            tip_support,
            branch_support,
            children,
            parent,
        }
    }

    pub fn erase(&mut self, child: &NodePointer<T>) {
        // This will find the index of the given child node such that the memory address
        // of the node equals the memory address of the Node in the children Vec.
        let index = self.children.iter()
            .position(|node| Rc::ptr_eq(node, child)).unwrap();
        self.children.swap_remove(index);
    }

    pub fn set_parent(&mut self, parent: NodePointer<T>) {
        self.parent = Some(parent);
    }
}

pub struct NodeLedgerTrie<T: Ledger> {
    root: Box<Node<T>>,
    seq_support: HashMap<LedgerIndex, u32>
}

impl<T: Ledger> LedgerTrie<T> for NodeLedgerTrie<T> {

    fn insert(&mut self, ledger: &T, count: Option<u32>) {
        let (mut loc, diff_seq) = self.find_mut(ledger);

        // loc.span has the longest common prefix with Span{ledger} of all
        // existing nodes in the trie. The Option<Span>'s below represent
        // the possible common suffixes between loc.span and Span{ledger}.
        //
        // loc.span
        //  a b c  | d e f
        //  prefix | oldSuffix
        //
        // Span{ledger}
        //  a b c  | g h i
        //  prefix | newSuffix

        let prefix = loc.borrow().span.before(diff_seq);
        let old_suffix = loc.borrow().span.from(diff_seq);
        let new_suffix = Span::new(*ledger).from(diff_seq);

        if let Some(old_suffix) = old_suffix {
            // Have
            //   abcdef -> ....
            // Inserting
            //   abc
            // Becomes
            //   abc -> def -> ...

            // Create old_suffix node that takes over loc
            let mut loc_ref_mut = loc.borrow_mut();
            let mut new_node = Rc::new(RefCell::new(
                Node::from_span(
                    old_suffix,
                    loc.borrow().tip_support,
                    loc.borrow().branch_support,
                    std::mem::replace(&mut loc_ref_mut.children, vec![]),
                    Some(loc.clone())
                )
            ));

            new_node.borrow_mut().children.iter_mut()
                .for_each(|child| child.borrow_mut().parent = Some(new_node.clone()));

            loc_ref_mut.span = prefix.unwrap();
            loc_ref_mut.children.push(new_node);
            loc_ref_mut.tip_support = 0;

            // TODO finish implementing
        }
    }

    fn get_preferred(&self, largest_issued: LedgerIndex) -> Option<SpanTip<T>> {
        todo!()
    }
}

impl<T: Ledger> NodeLedgerTrie<T> {
    fn find_mut(&mut self, ledger: &T) -> (Rc<RefCell<Node<T>>>, LedgerIndex) {
        let mut curr = self.root.deref_mut();

        // Note: This is different than C++ code. In C++, the loop below keeps updating curr.
        //  We cannot do that in Rust because of the borrow checker, so instead we keep track
        //  of the index of the child node that we are trying to find.
        let mut child_index = 0;
        let mut pos = curr.span.diff(ledger);

        let mut done = false;

        // Continue searching for a better span as long as the current position
        // matches the entire span
        while !done && pos == curr.span.end() {
            done = true;

            // Find the child with the longest ancestry match
            for child in 0..curr.children.len() {
                let child_pos = curr.children[child].borrow().span.diff(ledger);
                if child_pos > pos {
                    done = false;
                    pos = child_pos;
                    child_index = child;
                    break;
                }
            }
        }

        ((curr.children[child_index].clone()), pos)
    }
}
