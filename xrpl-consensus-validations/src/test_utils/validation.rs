use std::ops::{Add, AddAssign, Sub};
use std::time::SystemTime;

use derivative::Derivative;

use xrpl_consensus_core::{Ledger, LedgerIndex, Validation};

use crate::test_utils::ledgers::SimulatedLedger;

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub(crate) struct PeerId(pub u32);

impl Add for PeerId {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        PeerId(self.0 + rhs.0)
    }
}

impl AddAssign for PeerId {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

impl AddAssign<u32> for PeerId {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs
    }
}

impl Sub for PeerId {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        PeerId(self.0 - rhs.0)
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct PeerKey(pub PeerId, pub usize);

#[derive(Derivative)]
#[derivative(Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Debug)]
pub(crate) struct TestValidation {
    ledger_id: <SimulatedLedger as Ledger>::IdType,
    seq: LedgerIndex,
    sign_time: SystemTime,
    seen_time: SystemTime,
    key: PeerKey,
    node_id: PeerId,
    load_fee: Option<u32>,
    full: bool,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Ord = "ignore")]
    #[derivative(PartialOrd = "ignore")]
    trusted: bool,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Ord = "ignore")]
    #[derivative(PartialOrd = "ignore")]
    cookie: u64
}

impl TestValidation {
    pub fn new(
        ledger_id: <SimulatedLedger as Ledger>::IdType,
        seq: LedgerIndex,
        sign_time: SystemTime,
        seen_time: SystemTime,
        key: PeerKey,
        node_id: PeerId,
        trusted: bool,
        full: bool,
        load_fee: Option<u32>,
        cookie: Option<u64>
    ) -> Self {
        TestValidation {
            ledger_id,
            seq,
            sign_time,
            seen_time,
            key,
            node_id,
            trusted,
            full,
            load_fee,
            cookie: cookie.unwrap_or(0),
        }
    }

    pub(crate) fn node_id(&self) -> &PeerId {
        &self.node_id
    }

    pub(crate) fn key(&self) -> &PeerKey {
        &self.key
    }

    pub(crate) fn set_trusted(&mut self) {
        self.trusted = true;
    }

    pub(crate) fn set_untrusted(&mut self) {
        self.trusted = false;
    }

    pub(crate) fn set_seen(&mut self, seen: SystemTime) {
        self.seen_time = seen;
    }
}

impl Validation for TestValidation {
    type LedgerIdType = <SimulatedLedger as Ledger>::IdType;

    fn seq(&self) -> LedgerIndex {
        self.seq
    }

    fn ledger_id(&self) -> Self::LedgerIdType {
        self.ledger_id
    }

    fn sign_time(&self) -> SystemTime {
        self.sign_time
    }

    fn seen_time(&self) -> SystemTime {
        self.seen_time
    }

    fn cookie(&self) -> u64 {
        self.cookie
    }

    fn trusted(&self) -> bool {
        self.trusted
    }

    fn full(&self) -> bool {
        self.full
    }

    fn load_fee(&self) -> Option<u32> {
        self.load_fee
    }
}