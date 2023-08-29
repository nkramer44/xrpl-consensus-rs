
pub trait Ledger {
    type IdType;

    fn id(&self) -> Self::IdType;
}