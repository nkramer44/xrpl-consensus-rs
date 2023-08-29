use xrpl_consensus_core::{Ledger, LedgerId};

pub trait Adaptor {
    type ValidationType;
    type LedgerType: Ledger<IdType = Self::LedgerIdType>;
    type LedgerIdType: LedgerId;
    type NodeIdType;
    type NodeKeyType;
}