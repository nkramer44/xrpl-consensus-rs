pub mod aged_unordered_map;

use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::time::SystemTime;
use serde::Serialize;

pub type LedgerIndex = u32;

pub trait Ledger: Clone {
    type IdType: LedgerId;

    fn id(&self) -> Self::IdType;

    fn seq(&self) -> LedgerIndex;

    fn get_ancestor(&self, seq: LedgerIndex) -> Self::IdType;

    fn make_genesis() -> Self;

    fn mismatch(&self, other: &Self) -> LedgerIndex;
}

pub trait LedgerId: Eq + PartialEq + Ord + PartialOrd + Copy + Clone + Hash + Serialize + Debug + Display {

}

pub trait Validation: Copy + Clone {
    type LedgerIdType: LedgerId;

    fn seq(&self) -> LedgerIndex;
    fn ledger_id(&self) -> Self::LedgerIdType;
    fn sign_time(&self) -> SystemTime;
    fn seen_time(&self) -> SystemTime;
    fn cookie(&self) -> u64;
    fn trusted(&self) -> bool;
    fn full(&self) -> bool;
    fn load_fee(&self) -> Option<u32>;
}

pub trait NetClock {
    fn now(&self) -> SystemTime;
}

pub struct WallNetClock;

impl NetClock for WallNetClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}