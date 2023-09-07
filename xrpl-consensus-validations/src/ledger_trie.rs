use xrpl_consensus_core::{Ledger, LedgerIndex};

use crate::span::SpanTip;

pub trait LedgerTrie<T: Ledger> {

    fn insert(&mut self, ledger: &T, count: Option<u32>);
    fn remove(&mut self, ledger: &T, count: Option<u32>) -> bool;
    fn get_preferred(&self, largest_issued: LedgerIndex) -> Option<SpanTip<T>>;
    fn tip_support(&self, ledger: &T) -> u32;
    fn branch_support(&self, ledger: &T) -> u32;
}

