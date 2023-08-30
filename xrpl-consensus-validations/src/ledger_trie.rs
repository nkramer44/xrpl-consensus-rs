use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::rc::Rc;
use std::sync::Arc;
use xrpl_consensus_core::{Ledger, LedgerIndex};
use crate::span::{Span, SpanTip};

pub trait LedgerTrie<T: Ledger> {
    type NodePointer;

    fn insert(&mut self, ledger: &T, count: Option<u32>);
    fn get_preferred(&self, largest_issued: LedgerIndex) -> Option<SpanTip<T>>;
    fn find_mut(&mut self, ledger: &T) -> (Self::NodePointer, LedgerIndex);

}

