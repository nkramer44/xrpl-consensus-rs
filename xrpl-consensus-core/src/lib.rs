pub type LedgerIndex = u32;

pub trait Ledger: Clone {
    type IdType: Eq + PartialEq + Ord + PartialOrd + Copy + Clone;

    fn id(&self) -> Self::IdType;

    fn seq(&self) -> LedgerIndex;

    fn get_ancestor(&self, seq: LedgerIndex) -> Self::IdType;

    fn make_genesis() -> Self;

    fn mismatch(&self, other: &Self) -> LedgerIndex;
}

pub trait LedgerId: Eq + PartialEq + Ord + PartialOrd + Copy + Clone {

}