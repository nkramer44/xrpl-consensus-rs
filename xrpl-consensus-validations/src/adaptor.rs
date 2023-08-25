
pub trait Adaptor {
    type ValidationType;
    type LedgerType;
    type LedgerIdType;
    type NodeIdType;
    type NodeKeyType;
}