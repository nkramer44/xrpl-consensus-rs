use std::hash::Hash;
use std::time::SystemTime;
use async_trait::async_trait;
use xrpl_consensus_core::{Ledger, LedgerId, NetClock, Validation};

#[async_trait]
pub trait Adaptor {
    type ValidationType: Validation<LedgerIdType = Self::LedgerIdType>;
    type LedgerType: Ledger<IdType = Self::LedgerIdType>;
    type LedgerIdType: LedgerId;
    type NodeIdType: Eq + PartialEq + Hash + Copy + Clone;
    type NodeKeyType;
    type ClockType: NetClock;

    fn now(&self) -> SystemTime;

    async fn acquire(&mut self, ledger_id: &Self::LedgerIdType) -> Option<Self::LedgerType>;
}