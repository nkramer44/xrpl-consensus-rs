use xrpl_consensus_core::Ledger;

pub trait Adaptor {
    type ValidationType;
    type LedgerType: Ledger<IdType = Self::LedgerIdType>;
    type LedgerIdType;
    type NodeIdType;
    type NodeKeyType;
}