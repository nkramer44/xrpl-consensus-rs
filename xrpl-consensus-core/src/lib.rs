use std::ops::{Index, IndexMut};

pub type LedgerIndex = u32;

pub trait Ledger {
    type IdType: LedgerId;

    fn id(&self) -> Self::IdType;

    fn seq(&self) -> LedgerIndex;

    fn get_ancestor(&self, seq: LedgerIndex) -> Self::IdType;
}

pub trait LedgerId: Eq + PartialEq + Ord + PartialOrd + Copy + Clone {

}