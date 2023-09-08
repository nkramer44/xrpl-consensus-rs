use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::ops::{Add, Div, Sub};
use std::rc::Rc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bimap::BiMap;
use derivative::Derivative;
use once_cell::sync::Lazy;

use xrpl_consensus_core::{Ledger, LedgerIndex};

pub(crate) type TxSetType = Vec<Tx>;
pub(crate) type TxId = u32;

#[derive(Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Hash, Debug)]
pub(crate) struct LedgerId(u32);

#[derive(Ord, PartialOrd, Eq, PartialEq, Clone, Copy, Hash, Debug)]
pub(crate) struct Tx {
    id: TxId,
}

impl Tx {
    pub fn new(id: TxId) -> Self {
        Tx { id }
    }

    pub fn id(&self) -> TxId {
        self.id
    }
}

#[derive(Derivative)]
#[derivative(Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Debug)]
pub(crate) struct LedgerInstance {
    seq: LedgerIndex,
    txs: TxSetType,
    close_time_resolution: Duration,
    close_time: SystemTime,
    close_time_agree: bool,
    parent_id: LedgerId,
    parent_close_time: Option<SystemTime>,
    /// IDs of this ledgers ancestors. Since each ledger already has unique
    /// ancestors based on the parent_id, this member is not needed for any
    /// operations such as Hash or ordering.
    #[derivative(Hash = "ignore")]
    #[derivative(PartialEq = "ignore")]
    #[derivative(Ord = "ignore")]
    #[derivative(PartialOrd = "ignore")]
    ancestors: Vec<LedgerId>,
}

impl LedgerInstance {
    pub fn new(
        txs: TxSetType,
        ancestors: Vec<LedgerId>,
    ) -> Self {
        LedgerInstance {
            seq: 0,
            txs,
            close_time_resolution: Duration::from_secs(30),
            close_time: SystemTime::now(),
            close_time_agree: true,
            parent_id: LedgerId(0),
            parent_close_time: None,
            ancestors,
        }
    }

    fn genesis() -> LedgerInstance {
        LedgerInstance::new(
            vec![],
            vec![],
        )
    }
}

/// A ledger is a set of observed transactions and a sequence number
///     identifying the ledger.
///
/// Peers in the consensus process are trying to agree on a set of transactions
/// to include in a ledger. For simulation, each transaction is a single
/// integer and the ledger is the set of observed integers. This means future
/// ledgers have prior ledgers as subsets, e.g.
///
/// Ledger 0 :  {}
/// Ledger 1 :  {1,4,5}
/// Ledger 2 :  {1,2,4,5,10}
/// ....
///
/// Ledgers are immutable value types. All ledgers with the same sequence
/// number, transactions, close time, etc. will have the same ledger ID. The
/// LedgerOracle struct below manges ID assignments for a simulation and is the
/// only way to close and create a new ledger. Since the parent ledger ID is
/// part of type, this also means ledgers with distinct histories will have
/// distinct ids, even if they have the same set of transactions, sequence
/// number and close time.
#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Debug)]
pub(crate) struct SimulatedLedger {
    instance: Rc<LedgerInstance>,
    id: LedgerId,
}

pub(crate) static GENISIS: Lazy<LedgerInstance> = Lazy::new(|| {
    LedgerInstance::genesis()
});

impl SimulatedLedger {
    pub fn genesis() -> Self {
        SimulatedLedger {
            instance: Rc::new(LedgerInstance::genesis()),
            id: LedgerId(0),
        }
    }

    pub fn new(id: LedgerId, instance: Rc<LedgerInstance>) -> Self {
        SimulatedLedger {
            id,
            instance,
        }
    }

    pub fn is_ancestor(&self, ancestor: &SimulatedLedger) -> bool {
        if ancestor.seq() < self.seq() {
            return self.get_ancestor(ancestor.seq()) == ancestor.id();
        }
        false
    }

    pub fn close_time_resolution(&self) -> Duration {
        self.instance.close_time_resolution
    }

    pub fn close_agree(&self) -> bool {
        self.instance.close_time_agree
    }

    pub fn close_time(&self) -> SystemTime {
        self.instance.close_time
    }

    pub fn parent_close_time(&self) -> Option<SystemTime> {
        self.instance.parent_close_time
    }

    pub fn parent_id(&self) -> LedgerId {
        self.instance.parent_id
    }

    pub fn txs(&self) -> &TxSetType {
        &self.instance.txs
    }
}

impl Ledger for SimulatedLedger {
    type IdType = LedgerId;

    fn id(&self) -> Self::IdType {
        self.id
    }

    fn seq(&self) -> LedgerIndex {
        self.instance.seq
    }

    fn get_ancestor(&self, seq: LedgerIndex) -> Self::IdType {
        if seq > self.seq() {
            panic!("BLAH")
        }
        if seq == self.seq() {
            return self.id();
        }

        *self.instance.ancestors.get(seq as usize).unwrap()
    }

    fn make_genesis() -> Self {
        SimulatedLedger::genesis()
    }

    fn mismatch(&self, other: &Self) -> LedgerIndex {
        let mut start = 0;
        let end = std::cmp::min(self.seq() + 1, other.seq() + 1);

        let mut count = end - start;
        while count > 0 {
            let step = count / 2;
            let mut curr = start + step;
            if self.get_ancestor(curr) == other.get_ancestor(curr) {
                curr += 1;
                start = curr;
                count -= step + 1;
            } else {
                count = step;
            }
        }

        start
    }
}

/// Oracle maintaining unique ledgers for a simulation.
#[derive(Debug)]
pub(crate) struct LedgerOracle {
    instances: BiMap<Rc<LedgerInstance>, LedgerId>,
}

impl LedgerOracle {
    pub fn new() -> Self {
        let mut instances = BiMap::new();
        instances.insert(Rc::new(LedgerInstance::genesis()), LedgerId(0));
        LedgerOracle {
            instances
        }
    }

    pub fn lookup(&self, id: LedgerId) -> Option<SimulatedLedger> {
        self.instances.get_by_right(&id)
            .map(|entry| SimulatedLedger::new(id, entry.clone()))
    }

    pub fn accept_with_times(
        &mut self,
        parent: &SimulatedLedger,
        txs: &TxSetType,
        close_time_resolution: Duration,
        consensus_close_time: &SystemTime,
    ) -> Rc<SimulatedLedger> {
        let mut next_txs = parent.txs().clone();
        next_txs.extend_from_slice(txs.as_slice());
        let close_time_agree = parent.close_time() != UNIX_EPOCH;
        let mut next_ancestors = parent.instance.ancestors.clone();
        next_ancestors.push(parent.id());
        let next = Rc::new(
            LedgerInstance {
                seq: parent.seq() + 1,
                txs: next_txs,
                close_time_resolution,
                close_time: if close_time_agree {
                    effective_close_time(consensus_close_time, close_time_resolution, &parent.close_time())
                } else {
                    parent.close_time() + Duration::from_secs(1)
                },
                close_time_agree,
                parent_id: parent.id(),
                parent_close_time: Some(parent.close_time()),
                ancestors: next_ancestors,
            }
        );

        let id = if self.instances.contains_left(&next) {
            *self.instances.get_by_left(&next).unwrap()
        } else {
            let id = self.next_id();
            let inserted = self.instances.insert(next.clone(), id);
            id
        };
        return Rc::new(SimulatedLedger::new(
            id,
            next.clone(),
        ));
    }

    pub fn accept(
        &mut self,
        curr: &SimulatedLedger,
        tx: Tx,
    ) -> Rc<SimulatedLedger> {
        self.accept_with_times(
            curr,
            &vec![tx],
            curr.close_time_resolution(),
            &curr.close_time().add(Duration::from_secs(1)),
        )
    }

    pub fn branches(&self, ledgers: &HashSet<SimulatedLedger>) -> usize {
        // Tips always maintains the Ledgers with largest sequence number
        // along all known chains.
        let mut tips: Vec<SimulatedLedger> = Vec::with_capacity(ledgers.len());

        for ledger in ledgers {
            // Three options,
            //  1. ledger is on a new branch
            //  2. ledger is on a branch that we have seen tip for
            //  3. ledger is the new tip for a branch

            let mut found = false;

            for idx in 0..tips.len() {
                let idx_earlier = tips[idx].seq() < ledger.seq();
                let (earlier, later) = if idx_earlier {
                    (&tips[idx], ledger)
                } else {
                    (ledger, &tips[idx])
                };

                if later.is_ancestor(earlier) {
                    tips[idx] = later.clone();
                    found = true;
                    break;
                }
            }

            if !found {
                tips.push(ledger.clone());
            }
        }

        tips.len()
    }

    pub fn next_id(&self) -> LedgerId {
        LedgerId(self.instances.len() as u32)
    }
}

pub fn effective_close_time(consensus_close_time: &SystemTime, close_time_resolution: Duration, prior_close_time: &SystemTime) -> SystemTime {
    if prior_close_time == &UNIX_EPOCH {
        return *prior_close_time;
    }

    return std::cmp::max(
        round_close_time(prior_close_time, close_time_resolution),
        prior_close_time.add(Duration::from_secs(1)),
    );
}

fn round_close_time(close_time: &SystemTime, close_resolution: Duration) -> SystemTime {
    if close_time == &UNIX_EPOCH {
        return *close_time;
    }

    let close_time = close_time.add(close_resolution.div(2));
    return close_time.sub(
        // FIXME: C++ does close_time.time_since_epoch % close_resolution but rust Durations dont have % operator.
        Duration::from_nanos((close_time.duration_since(UNIX_EPOCH).unwrap().as_nanos() % close_resolution.as_nanos()) as u64)
    );
}

#[derive(Debug)]
pub(crate) struct LedgerHistoryHelper {
    oracle: LedgerOracle,
    next_tx: TxId,
    ledgers: HashMap<&'static str, Rc<SimulatedLedger>>,
    seen: HashSet<char>,
}

impl LedgerHistoryHelper {
    pub fn new() -> Self {
        let mut ledgers = HashMap::new();
        ledgers.insert("", Rc::new(SimulatedLedger::genesis()));
        LedgerHistoryHelper {
            oracle: LedgerOracle::new(),
            next_tx: 0,
            ledgers,
            seen: HashSet::new(),
        }
    }

    pub fn get_or_create(&mut self, s: &'static str) -> Rc<SimulatedLedger> {
        if let Some(ledger) = self.ledgers.get(s) {
            return ledger.clone();
        }

        assert!(self.seen.insert(s.chars().last().unwrap()));
        let parent = self.get_or_create(&s[0..s.len() - 1]);
        self.next_tx += 1;
        let new_ledger = self.oracle.accept(&parent, Tx::new(self.next_tx));
        self.ledgers.insert(s, new_ledger.clone());
        new_ledger.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_ledgers() {
        let mut helper = LedgerHistoryHelper::new();
        let ledger = helper.get_or_create("abc");
        assert_eq!(helper.ledgers.len(), 4);
        assert!(helper.ledgers.contains_key(""));
        assert!(helper.ledgers.contains_key("a"));
        assert!(helper.ledgers.contains_key("ab"));
        assert!(helper.ledgers.contains_key("abc"));

        let ledger = helper.get_or_create("abc");
        assert_eq!(helper.ledgers.len(), 4);
        assert!(helper.ledgers.contains_key(""));
        assert!(helper.ledgers.contains_key("a"));
        assert!(helper.ledgers.contains_key("ab"));
        assert!(helper.ledgers.contains_key("abc"));


        let ledger = helper.get_or_create("abcdef");
        assert_eq!(helper.ledgers.len(), 7);
        assert!(helper.ledgers.contains_key(""));
        assert!(helper.ledgers.contains_key("a"));
        assert!(helper.ledgers.contains_key("ab"));
        assert!(helper.ledgers.contains_key("abc"));
        assert!(helper.ledgers.contains_key("abcd"));
        assert!(helper.ledgers.contains_key("abcde"));
        assert!(helper.ledgers.contains_key("abcdef"));
    }
}