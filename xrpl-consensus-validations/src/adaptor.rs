use std::hash::Hash;
use std::time::SystemTime;
use xrpl_consensus_core::{Ledger, LedgerId, Validation};

pub trait Adaptor {
    type ValidationType: Validation<LedgerIdType = Self::LedgerIdType>;
    type LedgerType: Ledger<IdType = Self::LedgerIdType>;
    type LedgerIdType: LedgerId;
    type NodeIdType: Eq + PartialEq + Hash + Copy + Clone;
    type NodeKeyType;

    fn now(&self) -> SystemTime;

    fn acquire(&mut self, ledger_id: &Self::LedgerIdType) -> Option<Self::LedgerType>;
}