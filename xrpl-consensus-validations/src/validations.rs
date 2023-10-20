use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::ops::Range;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use xrpl_consensus_core::{Ledger, LedgerIndex, NetClock, Validation, WallNetClock};
use xrpl_consensus_core::aged_unordered_map::AgedUnorderedMap;

use crate::adaptor::Adaptor;
use crate::ledger_trie::LedgerTrie;
use crate::seq_enforcer::SeqEnforcer;
use crate::validation_params::ValidationParams;

struct KeepRange {
    pub low: LedgerIndex,
    pub high: LedgerIndex,
}

/// Maintains current and recent ledger validations.
///
/// Manages storage and queries related to validations received on the network.
/// Stores the most current validation from nodes and sets of recent
/// validations grouped by ledger identifier.
///
/// Stored validations are not necessarily from trusted nodes, so clients
/// and implementations should take care to use `trusted` member functions or
/// check the validation's trusted status.
///
/// This class uses a generic interface to allow adapting Validations for
/// specific applications. The Adaptor trait implements a set of helper
/// functions and type definitions.
pub struct Validations<A: Adaptor, T: LedgerTrie<A::LedgerType>, C: NetClock = WallNetClock> {
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
    by_ledger: AgedUnorderedMap<A::LedgerIdType, HashMap<A::NodeIdType, A::ValidationType>, C>,
    /// Partial and full validations indexed by sequence
    by_sequence: AgedUnorderedMap<LedgerIndex, HashMap<A::NodeIdType, A::ValidationType>, C>,
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

impl<A: Adaptor, T: LedgerTrie<A::LedgerType>, C: NetClock> Validations<A, T, C> {
    pub fn new(params: ValidationParams, adaptor: A, clock: Arc<RwLock<C>>) -> Self {
        Validations {
            current: Default::default(),
            local_seq_enforcer: SeqEnforcer::new(),
            seq_enforcers: Default::default(),
            by_ledger: AgedUnorderedMap::new(clock.clone()),
            by_sequence: AgedUnorderedMap::new(clock),
            to_keep: None,
            trie: T::default(),
            last_ledger: Default::default(),
            acquiring: Default::default(),
            params,
            adaptor,
        }
    }
}

impl<A: Adaptor, T: LedgerTrie<A::LedgerType>, C: NetClock> Validations<A, T, C> {
    pub fn adaptor(&self) -> &A {
        &self.adaptor
    }

    pub fn adaptor_mut(&mut self) -> &mut A {
        &mut self.adaptor
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
        self.local_seq_enforcer.advance_ledger(self.by_ledger.now(), seq, &self.params)
    }

    /// Attempt to add a new validation.
    ///
    /// # Params
    /// - **node_id**: The identity of the node issuing this validation.
    /// - **validation**: The validation to store.
    ///
    /// Returns `Ok(())` if the validation was added, or an `Err(ValidationError)` if not.
    pub async fn try_add(&mut self, node_id: &A::NodeIdType, validation: &A::ValidationType) -> Result<(), ValidationError<A::ValidationType>> {
        if !Self::_is_current(
            self.params(),
            &self.adaptor().now(),
            &validation.sign_time(),
            &validation.seen_time(),
        ) {
            return Err(ValidationError::Stale);
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
                    return Err(ValidationError::ConflictingLedgerId);
                }

                // Two validations for the same sequence and for the same
                // ledger with different sign times. This could be the
                // result of a misconfiguration but it can also mean a
                // Byzantine validator.
                if inserted.sign_time() != validation.sign_time() {
                    return Err(ValidationError::ConflictingSignTime(inserted.clone()));
                }

                // Two validations for the same sequence but with different
                // cookies. This is probably accidental misconfiguration.
                if inserted.cookie() != validation.cookie() {
                    return Err(ValidationError::Multiple);
                }
            }

            return Err(ValidationError::BadSeq);
        }

        self.by_ledger.get_or_insert_mut(validation.ledger_id()).insert(*node_id, *validation);

        match self.current.entry(*node_id) {
            Entry::Occupied(mut e) => {
                // Replace existing only if this one is newer
                if validation.sign_time() > e.get().sign_time() {
                    let old = (e.get().seq(), e.get().ledger_id());
                    e.insert(*validation);
                    if validation.trusted() {
                        self._process_validation(node_id, validation, Some(old)).await;
                    }
                } else {
                    return Err(ValidationError::Stale);
                }
            }
            Entry::Vacant(e) => {
                e.insert(*validation);
                if validation.trusted() {
                    self._process_validation(node_id, validation, None).await;
                }
            }
        }
        return Ok(());
    }

    /// Set the range of validations to keep from expiring.
    pub fn set_seq_to_keep(&mut self, range: Range<LedgerIndex>) {
        todo!()
    }

    /// Expire old validation sets. Removes validation sets that were accessed more than
    /// this `Validations`' `ValidationParams.validation_set_expires()` ago and were not asked
    /// to keep around.
    pub fn expire(&mut self) {
        todo!()
    }

    /// Update the trust status of validations.
    ///
    /// Updates the trusted status of known validations to account for nodes that have been added or
    /// removed from the UNL. This also updates the trie to ensure only currently trusted nodes'
    /// validations are used.
    ///
    /// # Params
    /// - **added**: A `&HashSet` of identifiers of nodes that are not trusted.
    /// - **removed**: A `&HashSet` of identifiers of nodes that are no longer trusted.
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

    pub fn get_preferred_lcl(
        &mut self,
        lcl: &A::LedgerType,
        min_seq: LedgerIndex,
        peer_counts: &HashMap<A::LedgerIdType, u32>,
    ) -> A::LedgerIdType {
        let preferred = self.get_preferred(lcl);

        // Trusted validations exist, but stick with local preferred ledger if
        // preferred is in the past
        if let Some(preferred) = preferred {
            return if preferred.0 > min_seq {
                preferred.1
            } else {
                lcl.id()
            };
        }

        peer_counts.into_iter()
            .max_by(|(id, count), (id2, count2)| {
                (count, id).cmp(&(count2, id2))
            })
            .map_or_else(
                || lcl.id(),
                |(max_id, max_count)| *max_id,
            )
    }

    /// Count the number of current trusted validators working on a ledger after the specified
    /// ledger.
    ///
    /// # Params
    /// - **ledger**: The working ledger.
    /// - **ledger_id**: The preferred ledger ID.
    ///
    /// Returns the number of current trusted validators working on a descendant of the preferred
    /// ledger.
    pub fn get_nodes_after(&mut self, ledger: &A::LedgerType, ledger_id: A::LedgerIdType) -> usize {
        // Use trie if ledger is the right one
        if ledger.id() == ledger_id {
            return self._with_trie(|trie| {
                let branch_support = trie.branch_support(ledger);
                let tip_support = trie.tip_support(ledger);
                branch_support - tip_support
            }) as usize;
        }

        // Otherwise count parent ledgers as a fallback
        return self.last_ledger.values().filter(|curr| {
            curr.seq() > 0 && curr.get_ancestor(curr.seq() - 1) == ledger_id
        }).count();
    }

    /// Get the currently trusted full validations.
    pub fn current_trusted(&mut self) -> Vec<A::ValidationType> {
        let mut ret = Vec::with_capacity(self.current.len());
        self._current(
            |node_id, val| {
                if val.trusted() && val.full() {
                    ret.push(*val);
                }
            }
        );

        ret
    }

    /// Get the set of the node IDs associated with current validations.
    pub fn get_current_node_ids(&mut self) -> HashSet<A::NodeIdType> {
        let mut ret = HashSet::with_capacity(self.current.len());
        self._current(
            |node_id, _| {
                ret.insert(*node_id);
            }
        );

        ret
    }

    /// Count the number of trusted full validations for the given ledger.
    pub fn num_trusted_for_ledger(&mut self, ledger_id: &A::LedgerIdType) -> usize {
        self._by_ledger(
            ledger_id,
            |_| { 0 },
            |node_id, val, ret| {
                if val.trusted() && val.full() {
                    *ret += 1;
                }
            },
        )
    }

    /// Get a `Vec` of trusted full validations for a specific ledger.
    pub fn get_trusted_for_ledger(
        &mut self,
        ledger_id: &A::LedgerIdType,
        seq: &LedgerIndex,
    ) -> Vec<A::ValidationType> {
        self._by_ledger(
            ledger_id,
            |num_validations| {
                Vec::with_capacity(num_validations)
            },
            |node_id, val, ret| {
                if val.trusted() && val.full() && &val.seq() == seq {
                    ret.push(*val);
                }
            },
        )
    }

    /// Returns fees reported by trusted full validators in the given ledger, as a `Vec<u32>`.
    pub fn fees(
        &mut self,
        ledger_id: &A::LedgerIdType,
        base_fee: u32,
    ) -> Vec<u32> {
        self._by_ledger(
            ledger_id,
            |num_validations| { Vec::with_capacity(num_validations) },
            |node_id, val, ret| {
                if val.trusted() && val.full() {
                    match val.load_fee() {
                        None => { ret.push(base_fee) }
                        Some(load_fee) => { ret.push(load_fee) }
                    }
                }
            },
        )
    }

    /// Flush all current validations.
    pub fn flush(&mut self) {
        self.current.clear();
    }

    /// Return the quantity of lagging proposers, and remove online proposers for purposes of
    /// evaluating whether to pause.
    ///
    /// Laggards are the trusted proposers whose sequence number is lower than the sequence number
    /// from which our current pending proposal is based. Proposers from whom we have not
    /// received a validation for awhile are considered offline.
    pub fn laggards(&mut self, seq: LedgerIndex, trusted_keys: HashSet<&mut A::NodeKeyType>) -> usize {
        todo!()
    }


    ///// Private functions
    fn _remove_trie(
        trie: &mut T,
        acquiring: &mut HashMap<(LedgerIndex, A::LedgerIdType), HashSet<A::NodeIdType>>,
        last_ledger: &mut HashMap<A::NodeIdType, A::LedgerType>,
        node_id: &A::NodeIdType,
        validation: &A::ValidationType,
    ) {
        let acquiring_entry = acquiring.entry((validation.seq(), validation.ledger_id()))
            .and_modify(|e| { e.remove(node_id); });
        if let Entry::Occupied(e) = acquiring_entry {
            if e.get().is_empty() {
                e.remove();
            }
        }

        if let Entry::Occupied(e) = last_ledger.entry(*node_id) {
            if e.get().id() == validation.ledger_id() {
                trie.remove(e.get(), None);
                last_ledger.remove(node_id);
            }
        }
    }

    fn _update_trie(
        trie: &mut T,
        last_ledger: &mut HashMap<A::NodeIdType, A::LedgerType>,
        node_id: &A::NodeIdType,
        ledger: A::LedgerType,
    ) {
        let ledger_copy = ledger.clone();
        match last_ledger.entry(*node_id) {
            Entry::Occupied(mut e) => {
                trie.remove(e.get(), None);
                e.insert(ledger_copy);
            }
            Entry::Vacant(e) => {
                e.insert(ledger_copy);
            }
        }

        trie.insert(&ledger, None);
    }

    async fn _process_validation(
        &mut self,
        node_id: &A::NodeIdType,
        validation: &A::ValidationType,
        prior: Option<(LedgerIndex, A::LedgerIdType)>,
    ) {
        // Clear any prior acquiring ledger for this node
        if let Some((seq, id)) = prior {
            if let Entry::Occupied(e) = self.acquiring.entry((seq, id))
                .and_modify(|e| {
                    e.remove(node_id);
                }) {
                if e.get().len() == 0 {
                    e.remove();
                }
            }
        }

        self.check_acquired().await;

        match self.acquiring.entry((validation.seq(), validation.ledger_id())) {
            Entry::Occupied(mut e) => {
                e.get_mut().insert(*node_id);
            }
            Entry::Vacant(e) => {
                match self.adaptor.acquire(&validation.ledger_id()).await {
                    None => {
                        self.acquiring.entry((validation.seq(), validation.ledger_id())).or_default()
                            .insert(*node_id);
                    }
                    Some(ledger) => {
                        Self::_update_trie(&mut self.trie, &mut self.last_ledger, node_id, ledger);
                    }
                }
            }
        }
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
        self._current(|_, _| {});
        f(&mut self.trie)
    }

    /// Iterate through current validations, flushing any which are stale.
    ///
    /// # Params
    /// **f**: A `FnMut` to call on each iteration.
    fn _current<F>(
        &mut self,
        mut f: F,
    ) where
        F: FnMut(&A::NodeIdType, &A::ValidationType) {
        let now = self.adaptor.now();

        let (current, acquiring, last_ledger, trie) = (
            &mut self.current, &mut self.acquiring, &mut self.last_ledger, &mut self.trie
        );
        current.retain(|node_id, val| {
            let is_current = Self::_is_current(&self.params, &now, &val.sign_time(), &val.seen_time());
            if !is_current {
                Self::_remove_trie(trie, acquiring, last_ledger, node_id, val);
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
        seen_time: &SystemTime,
    ) -> bool {
        return (sign_time > &(*now - p.validation_current_early())) &&
            (sign_time < &(*now + p.validation_current_wall())) &&
            ((seen_time == &UNIX_EPOCH) || (seen_time < &(*now + p.validation_current_local())));
    }

    fn _by_ledger<Pre, PreR, F>(
        &mut self,
        ledger_id: &A::LedgerIdType,
        pre: Pre,
        mut f: F,
    ) -> PreR where
        Pre: FnOnce(usize) -> PreR,
        F: FnMut(&A::NodeIdType, &A::ValidationType, &mut PreR), {
        if let Some(vals) = self.by_ledger.get(ledger_id) {
            self.by_ledger.touch(ledger_id);
            let mut pre_ret = pre(vals.len());
            vals.iter()
                .for_each(|(node_id, val)| {
                    f(node_id, val, &mut pre_ret)
                });
            pre_ret
        } else {
            pre(0)
        }
    }

    async fn check_acquired(&mut self) {
        let (trie, last_ledger) = (&mut self.trie, &mut self.last_ledger);
        let mut to_remove = vec![];
        for ((seq, id), node_ids) in &self.acquiring {
            if let Some(ledger) = self.adaptor.acquire(&id).await {
                for node_id in node_ids {
                    Self::_update_trie(trie, last_ledger, node_id, ledger.clone())
                }

                to_remove.push((*seq, *id));
            }
        }

        to_remove.iter().for_each(|r| { self.acquiring.remove(r); })
    }
}

/// Errors related to validations we received.
#[derive(Eq, PartialEq, Debug)]
pub enum ValidationError<T> {
    /// Validation is not current or was older than the current validation from this node.
    Stale,
    /// A validation violates the increasing sequence requirement.
    BadSeq,
    /// Multiple validations by a validator for the same ledger.
    Multiple,
    /// Multiple validations by a validator for different ledgers.
    ConflictingLedgerId,
    ConflictingSignTime(T),
}


#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, RwLock};
    use std::time::{Duration, SystemTime};

    use async_trait::async_trait;

    use xrpl_consensus_core::{Ledger, LedgerIndex, NetClock, Validation};

    use crate::adaptor::Adaptor;
    use crate::arena_ledger_trie::ArenaLedgerTrie;
    use crate::test_utils::ledgers::{LedgerHistoryHelper, LedgerId, LedgerOracle, SimulatedLedger};
    use crate::test_utils::ManualClock;
    use crate::test_utils::validation::{PeerId, PeerKey, TestValidation};
    use crate::validation_params::ValidationParams;
    use crate::validations::{ValidationError, Validations};

    #[tokio::test]
    async fn test_add_validations() {
        let mut h = LedgerHistoryHelper::new();
        let a = h.get_or_create("a");
        let ab = h.get_or_create("ab");
        let az = h.get_or_create("az");
        let abc = h.get_or_create("abc");
        let abcd = h.get_or_create("abcd");
        let abcde = h.get_or_create("abcde");

        let mut harness = TestHarness::new(h.oracle_mut());
        let mut node = harness.make_node();
        let v = node.validate_ledger(&a);
        assert_eq!(harness.try_add(&v).await, Ok(()));

        // Re-adding violates the increasing seq requirement for full
        // validations
        assert_eq!(harness.try_add(&v).await, Err(ValidationError::BadSeq));

        harness.advance_time(Duration::from_secs(1));

        assert_eq!(harness.try_add(&node.validate_ledger(&ab)).await, Ok(()));

        // Test the node changing signing key

        // Confirm old ledger on hand, but not new ledger
        // let validations_mut: &mut TestValidations = harness.validations_mut();
        assert_eq!(harness.validations.num_trusted_for_ledger(&ab.id()), 1);
        assert_eq!(harness.validations.num_trusted_for_ledger(&abc.id()), 0);


        // Rotate signing keys
        node.advance_key();

        harness.advance_time(Duration::from_secs(1));

        // Cannot re-do the same full validation sequence
        assert_eq!(harness.try_add(&node.validate_ledger(&ab)).await, Err(ValidationError::Conflicting));

        // Cannot send the same partial validation sequence
        assert_eq!(harness.try_add(&node.partial(&ab)).await, Err(ValidationError::Conflicting));

        // Now trusts the newest ledger too
        harness.advance_time(Duration::from_secs(1));
        assert_eq!(harness.try_add(&node.validate_ledger(&abc)).await, Ok(()));
        assert_eq!(harness.validations.num_trusted_for_ledger(&ab.id()), 1);
        assert_eq!(harness.validations.num_trusted_for_ledger(&abc.id()), 1);

        // Processing validations out of order should ignore the older
        // validation
        harness.advance_time(Duration::from_secs(2));
        let val_abcde = node.validate_ledger(&abcde);
        harness.advance_time(Duration::from_secs(4));
        let val_abcd = node.validate_ledger(&abcd);

        assert_eq!(harness.try_add(&val_abcd).await, Ok(()));
        assert_eq!(harness.try_add(&val_abcde).await, Err(ValidationError::Stale));
    }

    #[tokio::test]
    async fn test_add_validations_out_of_order_with_shifted_times() {
        let mut h = LedgerHistoryHelper::new();
        let a = h.get_or_create("a");
        let ab = h.get_or_create("ab");
        let az = h.get_or_create("az");
        let abc = h.get_or_create("abc");
        let abcd = h.get_or_create("abcd");
        let abcde = h.get_or_create("abcde");

        let mut harness = TestHarness::new(h.oracle_mut());
        let node = harness.make_node();

        // Establish a new current validation
        assert_eq!(harness.try_add(&node.validate_ledger(&a)).await, Ok(()));

        // Process a validation that has "later" seq but early sign time
        assert_eq!(harness.try_add(&node.validate_full(
            &ab,
            DurationOffset::Minus(Duration::from_secs(1)),
            DurationOffset::Minus(Duration::from_secs(1)),
        )).await, Err(ValidationError::Stale));

        // Process a validation that has a later seq and later sign
        // time
        assert_eq!(harness.try_add(&node.validate_full(
            &abc,
            DurationOffset::Plus(Duration::from_secs(1)),
            DurationOffset::Plus(Duration::from_secs(1)),
        )).await, Ok(()));
    }

    #[tokio::test]
    async fn test_add_validations_stale_on_arrival() {
        let mut h = LedgerHistoryHelper::new();
        let a = h.get_or_create("a");
        let ab = h.get_or_create("ab");
        let az = h.get_or_create("az");
        let abc = h.get_or_create("abc");
        let abcd = h.get_or_create("abcd");
        let abcde = h.get_or_create("abcde");

        let mut harness = TestHarness::new(h.oracle_mut());
        let node = harness.make_node();
        assert_eq!(
            harness.try_add(&node.validate_full(
                &a,
                DurationOffset::Minus(harness.params().validation_current_early()),
                DurationOffset::Zero,
            )).await,
            Err(ValidationError::Stale)
        );

        assert_eq!(
            harness.try_add(&node.validate_full(
                &a,
                DurationOffset::Plus(harness.params().validation_current_wall()),
                DurationOffset::Zero,
            )).await,
            Err(ValidationError::Stale)
        );

        assert_eq!(
            harness.try_add(&node.validate_full(
                &a,
                DurationOffset::Zero,
                DurationOffset::Plus(harness.params().validation_current_local()),
            )).await,
            Err(ValidationError::Stale)
        );
    }

    #[tokio::test]
    async fn test_add_validations_full_or_partials_cannot_be_sent_for_older_seqs_unless_timeout() {
        let mut h = LedgerHistoryHelper::new();
        let a = h.get_or_create("a");
        let ab = h.get_or_create("ab");
        let az = h.get_or_create("az");
        let abc = h.get_or_create("abc");
        let abcd = h.get_or_create("abcd");
        let abcde = h.get_or_create("abcde");

        do_test(true, &ab, &abc, &az, &mut h).await;
        do_test(false, &ab, &abc, &az, &mut h).await;
    }

    async fn do_test(do_full: bool, ab: &SimulatedLedger, abc: &SimulatedLedger, az: &SimulatedLedger, h: &mut LedgerHistoryHelper) {
        let mut harness = TestHarness::new(h.oracle_mut());
        let node = harness.make_node();

        /*let process = |harness: &mut TestHarness, ledger: &SimulatedLedger| {
            if do_full {
                harness.try_add(&node.validate_ledger(ledger)).await
            } else {
                harness.try_add(&node.partial(ledger)).await
            }
        };*/

        assert_eq!(process(do_full, &node, &mut harness, &abc).await, Ok(()));
        harness.advance_time(Duration::from_secs(1));
        assert!(ab.seq() < abc.seq());
        assert_eq!(process(do_full, &node, &mut harness, &ab).await, Err(ValidationError::BadSeq));

        // If we advance far enough for AB to expire, we can fully
        // validate or partially validate that sequence number again
        assert_eq!(process(do_full, &node, &mut harness, &az).await, Err(ValidationError::Conflicting));
        harness.advance_time(harness.params().validation_set_expires() + Duration::from_millis(1));
        assert_eq!(process(do_full, &node, &mut harness, &az).await, Ok(()));
    }

    async fn process(do_full: bool, node: &TestNode, harness: &mut TestHarness<'_>, ledger: &SimulatedLedger) -> Result<(), ValidationError> {
        if do_full {
            harness.try_add(&node.validate_ledger(ledger)).await
        } else {
            harness.try_add(&node.partial(ledger)).await
        }
    }

    /// Verify validation becomes stale based solely on time passing, but
    /// use different functions to trigger the check for staleness
    #[tokio::test]
    async fn test_on_stale() {
        let mut h = LedgerHistoryHelper::new();
        let a = h.get_or_create("a");
        let ab = h.get_or_create("ab");

        let genesis = SimulatedLedger::genesis();

        let triggers: Vec<Box<dyn Fn(&mut TestValidations)>> = vec![
            Box::new(|vals: &mut TestValidations| { vals.current_trusted(); }),
            // Box::new(|vals: &mut TestValidations| { vals.get_current_node_ids(); }),
            // Box::new(|vals: &mut TestValidations| { vals.get_preferred(&genesis); }),
            // Box::new(|vals: &mut TestValidations| { vals.current_trusted(); }),
        ];

        for trigger in triggers {
            run_test_for_trigger(trigger, &a, &ab, &mut h).await;
        }
    }

    async fn run_test_for_trigger(trigger: Box<dyn Fn(&mut TestValidations)>, a: &SimulatedLedger, ab: &SimulatedLedger, h: &mut LedgerHistoryHelper) {
        let mut harness = TestHarness::new(h.oracle_mut());
        let node = harness.make_node();

        assert_eq!(harness.try_add(&node.validate_ledger(&ab)).await, Ok(()));
        trigger(&mut harness.validations);
        assert_eq!(harness.validations.get_nodes_after(&a, a.id()), 1);
        assert_eq!(
            harness.validations.get_preferred(&SimulatedLedger::genesis()),
            Some((ab.seq(), ab.id()))
        );

        harness.advance_time(harness.params().validation_current_local());

        trigger(&mut harness.validations);

        assert_eq!(harness.validations.get_nodes_after(&a, a.id()), 0);
        assert_eq!(harness.validations.get_preferred(&SimulatedLedger::genesis()), None);
    }

    /// Test getting number of nodes working on a validation descending
    /// a prescribed one. This count should only be for trusted nodes, but
    /// includes partial and full validations.
    #[tokio::test]
    async fn test_get_nodes_after() {
        let mut h = LedgerHistoryHelper::new();
        let a = h.get_or_create("a");
        let ab = h.get_or_create("ab");
        let abc = h.get_or_create("abc");
        let ad = h.get_or_create("ad");

        let mut harness = TestHarness::new(h.oracle_mut());
        let a_node = harness.make_node();
        let b_node = harness.make_node();
        let mut c_node = harness.make_node();
        let d_node = harness.make_node();
        c_node.untrust();

        assert_eq!(harness.try_add(&a_node.validate_ledger(&a)).await, Ok(()));
        assert_eq!(harness.try_add(&b_node.validate_ledger(&a)).await, Ok(()));
        assert_eq!(harness.try_add(&c_node.validate_ledger(&a)).await, Ok(()));
        assert_eq!(harness.try_add(&d_node.partial(&a)).await, Ok(()));

        vec![&a, &ab, &abc, &ad].into_iter().for_each(|ledger| {
            assert_eq!(harness.validations.get_nodes_after(ledger, ledger.id()), 0);
        });

        harness.advance_time(Duration::from_secs(5));

        assert_eq!(harness.try_add(&a_node.validate_ledger(&ab)).await, Ok(()));
        assert_eq!(harness.try_add(&b_node.validate_ledger(&abc)).await, Ok(()));
        assert_eq!(harness.try_add(&c_node.validate_ledger(&ab)).await, Ok(()));
        assert_eq!(harness.try_add(&d_node.partial(&abc)).await, Ok(()));

        assert_eq!(harness.validations.get_nodes_after(&a, a.id()), 3);
        assert_eq!(harness.validations.get_nodes_after(&ab, ab.id()), 2);
        assert_eq!(harness.validations.get_nodes_after(&abc, abc.id()), 0);
        assert_eq!(harness.validations.get_nodes_after(&ad, ad.id()), 0);

        // If given a ledger inconsistent with the id, is still able to check
        // using slower method
        assert_eq!(harness.validations.get_nodes_after(&ad, a.id()), 1);
        assert_eq!(harness.validations.get_nodes_after(&ad, ab.id()), 2);
    }

    #[tokio::test]
    async fn test_current_trusted() {
        let mut h = LedgerHistoryHelper::new();
        let a = h.get_or_create("a");
        let b = h.get_or_create("b");
        let ac = h.get_or_create("ac");

        let mut harness = TestHarness::new(h.oracle_mut());
        let a_node = harness.make_node();
        let mut b_node = harness.make_node();
        b_node.untrust();

        assert_eq!(harness.try_add(&a_node.validate_ledger(&a)).await, Ok(()));
        assert_eq!(harness.try_add(&b_node.validate_ledger(&b)).await, Ok(()));

        // Only a_node is trusted
        assert_eq!(harness.validations.current_trusted().len(), 1);
        assert_eq!(harness.validations.current_trusted()[0].ledger_id(), a.id());
        assert_eq!(harness.validations.current_trusted()[0].seq(), a.seq());

        harness.advance_time(Duration::from_secs(3));

        assert_eq!(harness.try_add(&a_node.validate_ledger(&ac)).await, Ok(()));
        assert_eq!(harness.try_add(&b_node.validate_ledger(&ac)).await, Ok(()));

        // New validation for a_node
        assert_eq!(harness.validations.current_trusted().len(), 1);
        assert_eq!(harness.validations.current_trusted()[0].ledger_id(), ac.id());
        assert_eq!(harness.validations.current_trusted()[0].seq(), ac.seq());

        // Pass enough time for it to go stale
        harness.advance_time(harness.params.validation_current_local());
        assert!(harness.validations.current_trusted().is_empty());
    }

    #[tokio::test]
    async fn test_get_current_public_keys() {
        let mut h = LedgerHistoryHelper::new();
        let a = h.get_or_create("a");
        let ac = h.get_or_create("ac");

        let mut harness = TestHarness::new(h.oracle_mut());
        let mut a_node = harness.make_node();
        let mut b_node = harness.make_node();
        b_node.untrust();

        assert_eq!(harness.try_add(&a_node.validate_ledger(&a)).await, Ok(()));
        assert_eq!(harness.try_add(&b_node.validate_ledger(&a)).await, Ok(()));

        assert_eq!(harness.validations.get_current_node_ids(), HashSet::from([a_node.node_id(), b_node.node_id()]));

        harness.advance_time(Duration::from_secs(3));

        // Change keys and issue partials
        a_node.advance_key();
        b_node.advance_key();

        assert_eq!(harness.try_add(&a_node.partial(&ac)).await, Ok(()));
        assert_eq!(harness.try_add(&b_node.partial(&ac)).await, Ok(()));

        assert_eq!(harness.validations.get_current_node_ids(), HashSet::from([a_node.node_id(), b_node.node_id()]));

        harness.advance_time(harness.params.validation_current_wall());
        assert!(harness.validations.get_current_node_ids().is_empty());
    }

    /// Test the Validations functions that calculate a value by ledger ID.
    ///
    /// Several Validations functions return a set of values associated
    /// with trusted ledgers sharing the same ledger ID.  The tests below
    /// exercise this logic by saving the set of trusted Validations, and
    /// verifying that the Validations member functions all calculate the
    /// proper transformation of the available ledgers.
    #[tokio::test]
    async fn test_trusted_by_ledger() {
        let mut h = LedgerHistoryHelper::new();

        let a = h.get_or_create("a");
        let b = h.get_or_create("b");
        let ac = h.get_or_create("ac");

        let mut harness = TestHarness::new(h.oracle_mut());
        let mut a_node = harness.make_node();
        let mut b_node = harness.make_node();
        let mut c_node = harness.make_node();
        let d_node = harness.make_node();
        let mut e_node = harness.make_node();
        c_node.untrust();

        a_node.set_load_fee(12);
        b_node.set_load_fee(1);
        c_node.set_load_fee(12);
        e_node.set_load_fee(12);

        let mut trusted_validations: HashMap<(LedgerId, LedgerIndex), Vec<TestValidation>> = HashMap::new();

        // Add a dummy ID to cover unknown ledger identifiers
        trusted_validations.insert((LedgerId::new(100), 100), vec![]);

        // first round a,b,c agree
        for node in vec![&a_node, &b_node, &c_node] {
            let val = node.validate_ledger(&a);
            assert_eq!(harness.try_add(&val).await, Ok(()));
            if val.trusted() {
                trusted_validations.entry((val.ledger_id(), val.seq())).or_default()
                    .push(val);
            }
        }

        // d disagrees
        let val = d_node.validate_ledger(&b);
        assert_eq!(harness.try_add(&val).await, Ok(()));
        trusted_validations.entry((val.ledger_id(), val.seq())).or_default().push(val);

        // e only issued partials
        assert_eq!(harness.try_add(&e_node.partial(&a)).await, Ok(()));

        harness.advance_time(Duration::from_secs(5));
        // second round, a,b,c move to ledger 2
        for node in vec![&a_node, &b_node, &c_node] {
            let val = node.validate_ledger(&ac);
            assert_eq!(harness.try_add(&val).await, Ok(()));
            if val.trusted() {
                trusted_validations.entry((val.ledger_id(), val.seq())).or_default()
                    .push(val);
            }
        }

        // d now thinks ledger 1, but cannot re-issue a previously used seq
        // and attempting it should generate a conflict.
        assert_eq!(harness.try_add(&d_node.partial(&a)).await, Err(ValidationError::Conflicting));

        // e only issues partials
        assert_eq!(harness.try_add(&e_node.partial(&ac)).await, Ok(()));

        trusted_validations.iter_mut().for_each(|((id, seq), vals)| {
            assert_eq!(harness.validations.num_trusted_for_ledger(id), vals.len());
            let mut trusted = harness.validations.get_trusted_for_ledger(id, seq);
            trusted.sort();
            vals.sort();
            assert_eq!(
                &mut trusted,
                vals
            );

            let base_fee = 0;
            let mut expected_fees: Vec<u32> = vals.iter()
                .map(|val| val.load_fee().unwrap_or(base_fee))
                .collect();
            expected_fees.sort();
            let mut fees = harness.validations.fees(id, base_fee);
            fees.sort();
            assert_eq!(
                fees,
                expected_fees
            );
        })
    }

    #[test]
    fn test_expire() {
        // TODO: Implement this if we ever implement expire()
    }

    #[test]
    fn test_flush() {
        // TODO: Implement this if we ever implement flush()
    }

    #[tokio::test]
    async fn test_get_preferred_ledger() {
        let mut h = LedgerHistoryHelper::new();
        let a = h.get_or_create("a");
        let b = h.get_or_create("b");
        let ac = h.get_or_create("ac");
        let acd = h.get_or_create("acd");

        let mut harness = TestHarness::new(h.oracle_mut());
        let a_node = harness.make_node();
        let b_node = harness.make_node();
        let mut c_node = harness.make_node();
        let d_node = harness.make_node();
        c_node.untrust();

        // Empty, no ledgers
        assert!(harness.validations.get_preferred(&a).is_none());

        // Single ledger
        assert_eq!(harness.try_add(&a_node.validate_ledger(&b)).await, Ok(()));
        assert_eq!(harness.validations.get_preferred(&a), Some((b.seq(), b.id())));
        assert_eq!(harness.validations.get_preferred(&b), Some((b.seq(), b.id())));

        // Minimum valid sequence
        assert_eq!(harness.validations.get_preferred_id(&a, 10), a.id());

        // Untrusted doesn't impact preferred ledger
        // (ledger a has tie-break over ledger b)
        assert_eq!(harness.try_add(&b_node.validate_ledger(&a)).await, Ok(()));
        assert_eq!(harness.try_add(&c_node.validate_ledger(&a)).await, Ok(()));
        assert!(b.id() > a.id());
        assert_eq!(harness.validations.get_preferred(&a), Some((b.seq(), b.id())));
        assert_eq!(harness.validations.get_preferred(&b), Some((b.seq(), b.id())));

        // Partial does break ties
        assert_eq!(harness.try_add(&d_node.partial(&a)).await, Ok(()));
        assert_eq!(harness.validations.get_preferred(&a), Some((a.seq(), a.id())));
        assert_eq!(harness.validations.get_preferred(&b), Some((a.seq(), a.id())));

        harness.advance_time(Duration::from_secs(5));

        // Parent of preferred-> stick with ledger
        for node in vec![&a_node, &b_node, &c_node, &d_node] {
            assert_eq!(harness.try_add(&node.validate_ledger(&ac)).await, Ok(()));
        };

        // Parent of preferred stays put
        assert_eq!(harness.validations.get_preferred(&a), Some((a.seq(), a.id())));
        // Earlier different chain, switch
        assert_eq!(harness.validations.get_preferred(&b), Some((ac.seq(), ac.id())));
        // Later on chain, stays where it is
        assert_eq!(harness.validations.get_preferred(&acd), Some((acd.seq(), acd.id())));

        // Any later grandchild or different chain is preferred
        harness.advance_time(Duration::from_secs(5));
        for node in vec![&a_node, &b_node, &c_node, &d_node] {
            assert_eq!(harness.try_add(&node.validate_ledger(&acd)).await, Ok(()));
        };

        vec![&a, &b, &acd].into_iter().for_each(|ledger| {
            assert_eq!(harness.validations.get_preferred(ledger), Some((acd.seq(), acd.id())));
        })
    }

    #[tokio::test]
    async fn test_get_preferred_lcl() {
        let mut h = LedgerHistoryHelper::new();
        let a = h.get_or_create("a");
        let b = h.get_or_create("b");
        let c = h.get_or_create("c");

        let mut harness = TestHarness::new(h.oracle_mut());
        let a_node = harness.make_node();

        let mut peer_counts = HashMap::new();

        // No trusted validations or counts sticks with current ledger
        assert_eq!(harness.validations.get_preferred_lcl(&a, 0, &peer_counts), a.id());

        peer_counts.insert(b.id(), 1);

        // No trusted validations, rely on peer counts
        assert_eq!(harness.validations.get_preferred_lcl(&a, 0, &peer_counts), b.id());

        peer_counts.insert(c.id(), 1);

        // No trusted validations, tied peers goes with larger ID
        assert!(c.id() > b.id());

        assert_eq!(harness.validations.get_preferred_lcl(&a, 0, &peer_counts), c.id());

        *peer_counts.get_mut(&c.id()).unwrap() += 1000;

        // Single trusted always wins over peer counts
        assert_eq!(harness.try_add(&a_node.validate_ledger(&a)).await, Ok(()));
        assert_eq!(harness.validations.get_preferred_lcl(&a, 0, &peer_counts), a.id());
        assert_eq!(harness.validations.get_preferred_lcl(&b, 0, &peer_counts), a.id());
        assert_eq!(harness.validations.get_preferred_lcl(&c, 0, &peer_counts), a.id());

        // Stick with current ledger if trusted validation ledger has too old
        // of a sequence
        assert_eq!(harness.validations.get_preferred_lcl(&b, 2, &peer_counts), b.id());
    }

    #[test]
    fn test_acquire_validated_ledger() {
        todo!()
    }

    #[test]
    fn test_trust_changed() {
        todo!()
    }

    pub enum DurationOffset {
        Plus(Duration),
        Minus(Duration),
        Zero,
    }

    impl DurationOffset {
        pub fn apply_to(&self, sys_time: SystemTime) -> SystemTime {
            match self {
                DurationOffset::Plus(d) => sys_time + *d,
                DurationOffset::Minus(d) => sys_time - *d,
                DurationOffset::Zero => sys_time
            }
        }
    }


    struct TestNode {
        node_id: PeerId,
        trusted: bool,
        sign_idx: usize,
        load_fee: Option<u32>,
        clock: Arc<RwLock<ManualClock>>,
    }

    impl TestNode {
        pub fn new(node_id: PeerId, clock: Arc<RwLock<ManualClock>>) -> Self {
            TestNode {
                node_id,
                trusted: true,
                sign_idx: 1,
                load_fee: None,
                clock,
            }
        }

        pub fn untrust(&mut self) {
            self.trusted = false;
        }

        pub fn trust(&mut self) {
            self.trusted = true;
        }

        pub fn set_load_fee(&mut self, fee: u32) {
            self.load_fee = Some(fee);
        }

        pub fn node_id(&self) -> PeerId {
            self.node_id
        }

        pub fn advance_key(&mut self) {
            self.sign_idx += 1;
        }

        pub fn curr_key(&self) -> PeerKey {
            PeerKey(self.node_id, self.sign_idx)
        }

        pub fn master_key(&self) -> PeerKey {
            PeerKey(self.node_id, 0)
        }

        pub fn now(&self) -> SystemTime {
            self.clock.read().unwrap().now()
        }

        pub fn validate(
            &self,
            id: <SimulatedLedger as Ledger>::IdType,
            seq: LedgerIndex,
            sign_offset: DurationOffset,
            seen_offset: DurationOffset,
            full: bool,
        ) -> TestValidation {
            TestValidation::new(
                id,
                seq,
                sign_offset.apply_to(self.now()),
                seen_offset.apply_to(self.now()),
                self.curr_key(),
                self.node_id,
                self.trusted,
                full,
                self.load_fee,
                None,
            )
        }

        pub fn validate_full(
            &self,
            ledger: &SimulatedLedger,
            sign_offset: DurationOffset,
            seen_offset: DurationOffset,
        ) -> TestValidation {
            self.validate(
                ledger.id(),
                ledger.seq(),
                sign_offset,
                seen_offset,
                true,
            )
        }

        pub fn validate_ledger(&self, ledger: &SimulatedLedger) -> TestValidation {
            self.validate(
                ledger.id(),
                ledger.seq(),
                DurationOffset::Zero,
                DurationOffset::Zero,
                true,
            )
        }

        pub fn partial(&self, ledger: &SimulatedLedger) -> TestValidation {
            self.validate(
                ledger.id(),
                ledger.seq(),
                DurationOffset::Zero,
                DurationOffset::Zero,
                false,
            )
        }
    }

    struct TestAdaptor<'a> {
        oracle: &'a mut LedgerOracle,
        clock: Arc<RwLock<ManualClock>>,
    }

    impl<'a> TestAdaptor<'a> {
        fn new(oracle: &'a mut LedgerOracle, clock: Arc<RwLock<ManualClock>>) -> Self {
            TestAdaptor {
                oracle,
                clock,
            }
        }
    }

    #[async_trait]
    impl<'a> Adaptor for TestAdaptor<'a> {
        type ValidationType = TestValidation;
        type LedgerType = SimulatedLedger;
        type LedgerIdType = <Self::LedgerType as Ledger>::IdType;
        type NodeIdType = PeerId;
        type NodeKeyType = PeerKey;
        type ClockType = ManualClock;

        fn now(&self) -> SystemTime {
            self.clock.read().unwrap().now()
        }

        async fn acquire(&mut self, ledger_id: &Self::LedgerIdType) -> Option<Self::LedgerType> {
            self.oracle.lookup(ledger_id)
        }
    }

    type TestValidations<'a> = Validations<TestAdaptor<'a>, ArenaLedgerTrie<SimulatedLedger>, ManualClock>;

    struct TestHarness<'a> {
        params: ValidationParams,
        pub validations: TestValidations<'a>,
        next_node_id: PeerId,
        clock: Arc<RwLock<ManualClock>>,
    }

    impl<'a> TestHarness<'a> {
        pub fn new(oracle: &'a mut LedgerOracle) -> Self {
            let clock = Arc::new(RwLock::new(ManualClock::new()));
            TestHarness {
                params: Default::default(),
                validations: Validations::new(
                    ValidationParams::default(),
                    TestAdaptor::new(oracle, clock.clone()),
                    clock.clone(),
                ),
                next_node_id: PeerId(0),
                clock,
            }
        }

        pub async fn try_add(&mut self, v: &TestValidation) -> Result<(), ValidationError> {
            self.validations.try_add(&v.node_id(), v).await
        }

        pub fn make_node(&mut self) -> TestNode {
            self.next_node_id += 1;
            TestNode::new(self.next_node_id, self.clock.clone())
        }

        pub fn params(&self) -> &ValidationParams {
            &self.params
        }

        pub fn advance_time(&mut self, dur: Duration) {
            self.clock.write().unwrap().advance(dur);
        }
    }
}