use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::Add;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;
use xrpl_consensus_core::{Ledger, LedgerIndex};
use crate::adaptor::Adaptor;
use crate::ledger_trie::{LedgerTrie, SpanTip};
use crate::seq_enforcer::SeqEnforcer;
use crate::validation_params::ValidationParams;

struct AgedUnorderedMap<K, V> {
    // TODO: This should be a port/replica of beast::aged_unordered_map
    v: PhantomData<K>,
    k: PhantomData<V>
}

struct KeepRange {
    pub low: LedgerIndex,
    pub high: LedgerIndex,
}

pub struct Validations<T: Adaptor> {
    /// Manages concurrent access to members
    // TODO: Do we need a Mutex here, or should we create another struct that has one field, a Mutex<Validations>?
    //   I think the latter. see https://stackoverflow.com/questions/57256035/how-to-lock-a-rust-struct-the-way-a-struct-is-locked-in-go
    // mutex: Mutex<()>,
    /// Validations from currently listed and trusted nodes (partial and full)
    current: HashMap<T::NodeIdType, T::ValidationType>,
    /// Used to enforce the largest validation invariant for the local node
    local_seq_enforcer: SeqEnforcer,
    /// Sequence of the largest validation received from each node
    seq_enforcers: HashMap<T::NodeIdType, SeqEnforcer>,
    /// Validations from listed nodes, indexed by ledger id (partial and full)
    by_ledger: AgedUnorderedMap<T::LedgerIdType, HashMap<T::NodeIdType, T::ValidationType>>,
    /// Partial and full validations indexed by sequence
    by_sequence: AgedUnorderedMap<LedgerIndex, HashMap<T::NodeIdType, T::ValidationType>>,
    /// A range [low, high) of validations to keep from expire
    to_keep: Option<KeepRange>,
    /// Represents the ancestry of validated ledgers
    trie: LedgerTrie<T::LedgerType>,
    /// Last (validated) ledger successfully acquired. If in this map, it is
    /// accounted for in the trie.
    last_ledger: HashMap<T::NodeIdType, T::LedgerType>,
    /// Set of ledgers being acquired from the network
    acquiring: HashMap<(LedgerIndex, T::LedgerIdType), HashSet<T::NodeIdType>>,
    /// Parameters to determine validation staleness
    params: ValidationParams,
    /// Adaptor instance
    /// Is NOT managed by the Mutex above
    adaptor: T,
}

impl<T: Adaptor> Validations<T> {
    pub fn adaptor(&self) -> &T {
        &self.adaptor
    }

    pub fn params(&self) -> &ValidationParams {
        &self.params
    }

    /// Return whether the local node can issue a validation for the given
    /// sequence number.
    ///
    /// # Params
    /// - s: The [`LedgerIndex`] of the ledger the node wants to validate
    ///
    /// # Return
    /// A bool indicating whether the validation satisfies the invariant, updating the
    /// largest sequence number seen accordingly.
    pub fn can_validate_seq(&mut self, seq: LedgerIndex) -> bool {
        self.local_seq_enforcer.advance_ledger(Instant::now(), seq, &self.params)
    }

    pub fn add(&mut self, node_id: &T::NodeIdType, validation: &T::ValidationType) -> ValidationStatus {
        todo!()
    }

    pub fn set_seq_to_keep(&mut self, low: &LedgerIndex, high: &LedgerIndex) {
        todo!()
    }

    pub fn expire(&mut self) {
        todo!()
    }

    pub fn trust_changed(
        &mut self,
        added: &HashSet<T::NodeIdType>,
        removed: &HashSet<T::NodeIdType>,
    ) {
        todo!()
    }

    /// Return the sequence number and ID of the preferred working ledger.
    ///
    /// A ledger is preferred if it has more support amongst trusted validators and is **not**
    /// an ancestor of the current working ledger; otherwise it remains the current working ledger.
    ///
    /// # Params
    /// - curr: The local node's current working ledger.
    ///
    /// # Returns
    /// The sequence and id of the preferred working ledger, or `None` if no trusted validations
    /// are available to determine the preferred ledger.
    pub fn get_preferred(&mut self, curr: &T::LedgerType) -> Option<(LedgerIndex, T::LedgerIdType)> {
        let seq = self.local_seq_enforcer.largest();
        let preferred = self._with_trie(|trie| {
            trie.get_preferred(seq)
        });

        match preferred {
            // No trusted validations to determine branch
            None => {
                // fall back to majority over acquiring ledgers
                self.acquiring.iter()
                    .max_by(|a, b| {
                        let a_key = a.0;
                        let a_size: usize = a.1.len();
                        let b_key = b.0;
                        let b_size: usize = b.1.len();

                        // order by number of trusted peers validating that ledger
                        // break ties with ledger ID
                        match a_size.cmp(&b_size) {
                            Ordering::Less => Ordering::Less,
                            Ordering::Greater => Ordering::Greater,
                            Ordering::Equal => {
                                let a_ledger_id = &a_key.1;
                                let b_ledger_id = &b_key.1;
                                a_ledger_id.cmp(b_ledger_id)
                            }
                        }
                    })
                    .map(|entry| *entry.0)
            }
            Some(preferred) => {
                // If we are the parent of the preferred ledger, stick with our
                // current ledger since we might be about to generate it
                if preferred.seq() == curr.seq() + 1 &&
                    preferred.ancestor(curr.seq()) == curr.id() {
                    return Some((curr.seq(), curr.id()))
                }

                // A ledger ahead of us is preferred regardless of whether it is
                // a descendant of our working ledger or it is on a different chain
                if preferred.seq() > curr.seq() {
                    return Some((preferred.seq(), preferred.id()));
                }

                // Only switch to earlier or same sequence number
                // if it is a different chain.
                if curr.get_ancestor(preferred.seq()) != preferred.id() {
                    return Some((preferred.seq(), preferred.id()))
                }

                // Stick with current ledger
                return Some((curr.seq(), curr.id()));
            }
        }
    }

    /// Return the ID of the preferred working ledger that exceeds a minimum valid ledger sequence
    /// number.
    ///
    /// A ledger is preferred if it has more support amongst trusted validators and is **not**
    /// an ancestor of the current working ledger; otherwise it remains the current working ledger.
    ///
    /// # Params
    /// - curr: The local node's current working ledger.
    /// - min_valid_seq: Minimum allowed sequence number.
    ///
    /// # Returns
    /// The ID of the preferred working ledger, or `curr` if the preferred ledger is not valid.
    pub fn get_preferred_id(
        &mut self,
        curr: &T::LedgerType,
        min_valid_seq: LedgerIndex,
    ) -> T::LedgerIdType {
        self.get_preferred(curr)
            .filter(|preferred| preferred.0 >= min_valid_seq)
            .map_or_else(
                || curr.id(),
                |preferred| preferred.1
            )
    }

    pub fn get_prefered_lcl(
        &mut self,
        lcl: &T::LedgerType,
        min_seq: LedgerIndex,
        peer_counts: HashMap<T::LedgerIdType, u32>,
    ) -> T::LedgerIdType {
        todo!()
    }

    pub fn get_nodes_after(&mut self, ledger: &T::LedgerType, ledger_id: T::LedgerIdType) -> usize {
        todo!()
    }

    pub fn current_trusted(&mut self) -> Vec<T::ValidationType> {
        todo!()
    }

    pub fn get_current_node_ids(&mut self) -> HashSet<T::NodeIdType> {
        todo!()
    }

    pub fn num_trusted_for_ledger(&mut self, ledger_id: &T::LedgerIdType) -> usize {
        todo!()
    }

    pub fn get_trusted_for_ledger(
        &mut self,
        ledger_id: &T::LedgerIdType,
        seq: LedgerIndex,
    ) -> Vec<T::ValidationType> {
        todo!()
    }

    pub fn fees(
        &mut self,
        ledger_id: &T::LedgerIdType,
        base_fee: u32,
    ) -> Vec<u32> {
        todo!()
    }

    pub fn flush(&mut self) {
        todo!()
    }

    pub fn laggards(&mut self, seq: LedgerIndex, trusted_keys: HashSet<&mut T::NodeKeyType>) -> usize {
        todo!()
    }


    ///// Private functions
    fn _remove_trie(
        &mut self,
        lock: MutexGuard<()>,
        node_id: &T::NodeIdType,
        validation: &T::ValidationType,
    ) {
        todo!()
    }

    fn _check_acquired(&self, lock: MutexGuard<()>) {
        todo!()
    }

    fn _update_trie(
        &mut self,
        lock: MutexGuard<()>,
        node_id: &T::NodeIdType,
        ledger: T::LedgerType,
    ) {
        todo!()
    }

    fn _process_validation(
        &mut self,
        lock: MutexGuard<()>,
        node_id: &T::NodeIdType,
        validation: &T::ValidationType,
        prior: Option<(LedgerIndex, T::LedgerIdType)>,
    ) {
        todo!()
    }

    fn _with_trie<F: FnMut(&mut LedgerTrie<T::LedgerType>) -> R, R>(
        &mut self,
        // TODO: Maybe reinstate this
        // lock: MutexGuard<()>,
        f: F,
    ) -> R {
        todo!()
    }

    fn _current<Pre, F>(
        &mut self,
        lock: MutexGuard<()>,
        pre: Pre,
        f: F,
    ) where
        Pre: FnOnce(usize),
        F: Fn(&T::NodeIdType, &T::ValidationType) {
        todo!()
    }

    fn _by_ledger<Pre, F>(
        &mut self,
        lock: MutexGuard<()>,
        ledger_id: &T::LedgerIdType,
        pre: Pre,
        f: F
    ) where
        Pre: FnOnce(usize),
        F: Fn(&T::NodeIdType, &T::ValidationType) {
        todo!()
    }
}


pub enum ValidationStatus {
    Current,
    Stale,
    BadSeq,
    Multiple,
    Conflicting,
}