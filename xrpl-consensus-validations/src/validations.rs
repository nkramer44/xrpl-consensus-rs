use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::marker::PhantomData;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use xrpl_consensus_core::{Ledger, LedgerIndex, Validation};

use crate::adaptor::Adaptor;
use crate::ledger_trie::LedgerTrie;
use crate::seq_enforcer::SeqEnforcer;
use crate::validation_params::ValidationParams;

struct AgedUnorderedMap<K, V> {
    // TODO: This should be a port/replica of beast::aged_unordered_map
    v: PhantomData<K>,
    k: PhantomData<V>,
}

impl<K, V> AgedUnorderedMap<K, V> {
    pub fn now(&self) -> Instant {
        todo!()
    }

    pub fn get_or_insert(&mut self, k: K) -> &V {
        todo!()
    }

    pub fn get_or_insert_mut(&mut self, k: K) -> &mut V {
        todo!()
    }
}

struct KeepRange {
    pub low: LedgerIndex,
    pub high: LedgerIndex,
}

pub struct Validations<A: Adaptor, T: LedgerTrie<A::LedgerType>> {
    /// Manages concurrent access to members
    // TODO: Do we need a Mutex here, or should we create another struct that has one field, a Mutex<Validations>?
    //   I think the latter. see https://stackoverflow.com/questions/57256035/how-to-lock-a-rust-struct-the-way-a-struct-is-locked-in-go
    // mutex: Mutex<()>,
    /// Validations from currently listed and trusted nodes (partial and full)
    current: HashMap<A::NodeIdType, A::ValidationType>,
    /// Used to enforce the largest validation invariant for the local node
    local_seq_enforcer: SeqEnforcer,
    /// Sequence of the largest validation received from each node
    seq_enforcers: HashMap<A::NodeIdType, SeqEnforcer>,
    /// Validations from listed nodes, indexed by ledger id (partial and full)
    by_ledger: AgedUnorderedMap<A::LedgerIdType, HashMap<A::NodeIdType, A::ValidationType>>,
    /// Partial and full validations indexed by sequence
    by_sequence: AgedUnorderedMap<LedgerIndex, HashMap<A::NodeIdType, A::ValidationType>>,
    /// A range [low, high) of validations to keep from expire
    to_keep: Option<KeepRange>,
    /// Represents the ancestry of validated ledgers
    trie: T,
    /// Last (validated) ledger successfully acquired. If in this map, it is
    /// accounted for in the trie.
    last_ledger: HashMap<A::NodeIdType, A::LedgerType>,
    /// Set of ledgers being acquired from the network
    acquiring: HashMap<(LedgerIndex, A::LedgerIdType), HashSet<A::NodeIdType>>,
    /// Parameters to determine validation staleness
    params: ValidationParams,
    /// Adaptor instance
    /// Is NOT managed by the Mutex above
    adaptor: A,
}

impl<A: Adaptor, T: LedgerTrie<A::LedgerType>> Validations<A, T> {
    pub fn adaptor(&self) -> &A {
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

    pub fn add(&mut self, node_id: &A::NodeIdType, validation: &A::ValidationType) -> ValidationStatus {
        if !Self::_is_current(
            self.params(),
            &self.adaptor().now(),
            &validation.sign_time(),
            &validation.seen_time()
        ) {
            return ValidationStatus::Stale;
        }

        // Check that validation sequence is greater than any non-expired
        // validations sequence from that validator; if it's not, perform
        // additional work to detect Byzantine validations
        let now = self.by_ledger.now();

        let inserted = match self.by_sequence.get_or_insert_mut(validation.seq()).entry(*node_id) {
            Entry::Occupied(mut e) => {
                // Check if the entry we're already tracking was signed
                // long enough ago that we can disregard it.
                let diff = e.get().sign_time().max(validation.sign_time())
                    .duration_since(e.get().sign_time().min(validation.sign_time()))
                    .unwrap();
                if diff > self.params.validation_current_wall() &&
                    validation.sign_time() > e.get().sign_time() {
                    e.insert(*validation);
                }

                e.into_mut()
            }
            Entry::Vacant(e) => {
                e.insert(*validation)
            }
        };

        // Enforce monotonically increasing sequences for validations
        // by a given node, and run the active Byzantine detector:
        let enforcer = self.seq_enforcers.entry(*node_id).or_insert(SeqEnforcer::new());
        if !enforcer.advance_ledger(now, validation.seq(), &self.params) {
            // If the validation is for the same sequence as one we are
            // tracking, check it closely:
            if inserted.seq() == validation.seq() {
                // Two validations for the same sequence but for different
                // ledgers. This could be the result of misconfiguration
                // but it can also mean a Byzantine validator.
                if inserted.ledger_id() != validation.ledger_id() {
                    return ValidationStatus::Conflicting;
                }

                // Two validations for the same sequence and for the same
                // ledger with different sign times. This could be the
                // result of a misconfiguration but it can also mean a
                // Byzantine validator.
                if inserted.sign_time() != validation.sign_time() {
                    return ValidationStatus::Conflicting;
                }

                // Two validations for the same sequence but with different
                // cookies. This is probably accidental misconfiguration.
                if inserted.cookie() != validation.cookie() {
                    return ValidationStatus::Multiple;
                }
            }

            return ValidationStatus::BadSeq;
        }

        self.by_ledger.get_or_insert_mut(validation.ledger_id()).insert(*node_id, *validation);

        match self.current.entry(*node_id) {
            Entry::Occupied(mut e) => {
                // Replace existing only if this one is newer
                if validation.sign_time() > e.get().sign_time() {
                    let old = (e.get().seq(), e.get().ledger_id());
                    e.insert(*validation);
                    if validation.trusted() {
                        self._process_validation(node_id, validation, Some(old));
                    }
                } else {
                    return ValidationStatus::Stale;
                }
            }
            Entry::Vacant(e) => {
                e.insert(*validation);
                if validation.trusted() {
                    self._process_validation(node_id, validation, None);
                }
            }
        }
        return ValidationStatus::Current;
    }

    pub fn set_seq_to_keep(&mut self, low: &LedgerIndex, high: &LedgerIndex) {
        todo!()
    }

    pub fn expire(&mut self) {
        todo!()
    }

    pub fn trust_changed(
        &mut self,
        added: &HashSet<A::NodeIdType>,
        removed: &HashSet<A::NodeIdType>,
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
    pub fn get_preferred(&mut self, curr: &A::LedgerType) -> Option<(LedgerIndex, A::LedgerIdType)> {
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
                    return Some((curr.seq(), curr.id()));
                }

                // A ledger ahead of us is preferred regardless of whether it is
                // a descendant of our working ledger or it is on a different chain
                if preferred.seq() > curr.seq() {
                    return Some((preferred.seq(), preferred.id()));
                }

                // Only switch to earlier or same sequence number
                // if it is a different chain.
                if curr.get_ancestor(preferred.seq()) != preferred.id() {
                    return Some((preferred.seq(), preferred.id()));
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
        curr: &A::LedgerType,
        min_valid_seq: LedgerIndex,
    ) -> A::LedgerIdType {
        self.get_preferred(curr)
            .filter(|preferred| preferred.0 >= min_valid_seq)
            .map_or_else(
                || curr.id(),
                |preferred| preferred.1,
            )
    }

    pub fn get_prefered_lcl(
        &mut self,
        lcl: &A::LedgerType,
        min_seq: LedgerIndex,
        peer_counts: HashMap<A::LedgerIdType, u32>,
    ) -> A::LedgerIdType {
        todo!()
    }

    pub fn get_nodes_after(&mut self, ledger: &A::LedgerType, ledger_id: A::LedgerIdType) -> usize {
        todo!()
    }

    pub fn current_trusted(&mut self) -> Vec<A::ValidationType> {
        todo!()
    }

    pub fn get_current_node_ids(&mut self) -> HashSet<A::NodeIdType> {
        todo!()
    }

    pub fn num_trusted_for_ledger(&mut self, ledger_id: &A::LedgerIdType) -> usize {
        todo!()
    }

    pub fn get_trusted_for_ledger(
        &mut self,
        ledger_id: &A::LedgerIdType,
        seq: LedgerIndex,
    ) -> Vec<A::ValidationType> {
        todo!()
    }

    pub fn fees(
        &mut self,
        ledger_id: &A::LedgerIdType,
        base_fee: u32,
    ) -> Vec<u32> {
        todo!()
    }

    pub fn flush(&mut self) {
        todo!()
    }

    pub fn laggards(&mut self, seq: LedgerIndex, trusted_keys: HashSet<&mut A::NodeKeyType>) -> usize {
        todo!()
    }


    ///// Private functions
    fn _remove_trie(
        trie: &mut T,
        node_id: &A::NodeIdType,
        validation: &A::ValidationType,
    ) {
        todo!()
    }

    fn _update_trie(
        &mut self,
        node_id: &A::NodeIdType,
        ledger: A::LedgerType,
    ) {
        todo!()
    }

    fn _process_validation(
        &mut self,
        node_id: &A::NodeIdType,
        validation: &A::ValidationType,
        prior: Option<(LedgerIndex, A::LedgerIdType)>,
    ) {
        todo!()
    }

    /// Use the trie for a calculation.
    ///
    /// Accessing the trie through this helper ensures acquiring validations are checked
    /// and any stale validations are flushed from the trie.
    fn _with_trie<F: FnMut(&mut T) -> R, R>(
        &mut self,
        mut f: F,
    ) -> R {
        // Call current to flush any stale validations
        self._current(|_| {}, |_, _| {});
        f(&mut self.trie)
    }

    /// Iterate through current validations, flushing any which are stale.
    ///
    /// # Params
    /// **pre**: A `FnOnce(usize)` called prior to iterating.
    /// **f**: A `Fn` to call on each iteration.
    fn _current<Pre, F>(
        &mut self,
        pre: Pre,
        f: F,
    ) where
        Pre: FnOnce(usize),
        F: Fn(&A::NodeIdType, &A::ValidationType) {
        let now = self.adaptor.now();
        pre(self.current.len());

        let (current, trie) = (&mut self.current, &mut self.trie);
        current.retain(|node_id, val| {
            let is_current = Self::_is_current(&self.params, &now, &val.sign_time(), &val.seen_time());
            if !is_current {
                Self::_remove_trie(trie, node_id, val);
            } else {
                f(node_id, val);
            }
            is_current
        });
    }

    fn _is_current(
        p: &ValidationParams,
        now: &SystemTime,
        sign_time: &SystemTime,
        seen_time: &SystemTime
    ) -> bool {
        return (sign_time > &(*now - p.validation_current_early())) &&
            (sign_time < &(*now + p.validation_current_wall())) &&
            ((seen_time == &UNIX_EPOCH) || (seen_time < &(*now + p.validation_current_local())));
    }

    fn _by_ledger<Pre, F>(
        &mut self,
        ledger_id: &A::LedgerIdType,
        pre: Pre,
        f: F,
    ) where
        Pre: FnOnce(usize),
        F: Fn(&A::NodeIdType, &A::ValidationType) {
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