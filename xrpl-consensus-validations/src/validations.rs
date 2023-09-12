use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::ops::Add;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use xrpl_consensus_core::{Ledger, LedgerIndex, Validation};

use crate::adaptor::Adaptor;
use crate::ledger_trie::LedgerTrie;
use crate::seq_enforcer::SeqEnforcer;
use crate::validation_params::ValidationParams;

struct AgedUnorderedMap<K, V>
    where
        K: Eq + PartialEq + Hash,
        V: Default {
    // TODO: This should be a port/replica of beast::aged_unordered_map. The only
    //  difference between a HashMap and beast::aged_unordered_map is that you can
    //  expire entries, but for simplicity's sake, we can just not expire entries.
    inner: HashMap<K, V>,
}

impl<K: Eq + PartialEq + Hash, V: Default> Default for AgedUnorderedMap<K, V> {
    fn default() -> Self {
        AgedUnorderedMap {
            inner: HashMap::default()
        }
    }
}

impl<K: Eq + PartialEq + Hash, V: Default> AgedUnorderedMap<K, V> {
    pub fn now(&self) -> Instant {
        Instant::now()
    }

    pub fn get_or_insert(&mut self, k: K) -> &V {
        self.get_or_insert_mut(k)
    }

    pub fn get_or_insert_mut(&mut self, k: K) -> &mut V {
        self.inner.entry(k).or_default()
    }

    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
        where
            K: Borrow<Q>,
            Q: Hash + Eq,
    {
        self.inner.get(k)
    }

    fn touch(&self, k: &K) {
        // TODO: Update the timestamp on the entry
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
    pub fn new(params: ValidationParams, adaptor: A) -> Self {
        Validations {
            current: Default::default(),
            local_seq_enforcer: SeqEnforcer::new(),
            seq_enforcers: Default::default(),
            by_ledger: AgedUnorderedMap::default(),
            by_sequence: AgedUnorderedMap::default(),
            to_keep: None,
            trie: T::default(),
            last_ledger: Default::default(),
            acquiring: Default::default(),
            params,
            adaptor,
        }
    }
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
            &validation.seen_time(),
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
        let mut count = 0;
        self._by_ledger(
            ledger_id,
            |_| {},
            |node_id, val| {
                if val.trusted() && val.full() {
                    count += 1;
                }
            },
        );
        count
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
        match self.last_ledger.entry(*node_id) {
            Entry::Occupied(mut e) => {
                self.trie.remove(e.get(), None);
                e.insert(ledger);
            }
            Entry::Vacant(e) => {
                e.insert(ledger);
            }
        }
    }

    fn _process_validation(
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

        match self.acquiring.entry((validation.seq(), validation.ledger_id())) {
            Entry::Occupied(mut e) => {
                e.get_mut().insert(*node_id);
            }
            Entry::Vacant(e) => {
                match self.adaptor.acquire(&validation.ledger_id()) {
                    None => {
                        self.acquiring.entry((validation.seq(), validation.ledger_id())).or_default()
                            .insert(*node_id);
                    }
                    Some(ledger) => {
                        self._update_trie(node_id, ledger);
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
        seen_time: &SystemTime,
    ) -> bool {
        return (sign_time > &(*now - p.validation_current_early())) &&
            (sign_time < &(*now + p.validation_current_wall())) &&
            ((seen_time == &UNIX_EPOCH) || (seen_time < &(*now + p.validation_current_local())));
    }

    fn _by_ledger<Pre, F>(
        &mut self,
        ledger_id: &A::LedgerIdType,
        pre: Pre,
        mut f: F,
    ) where
        Pre: FnOnce(usize),
        F: FnMut(&A::NodeIdType, &A::ValidationType) {
        if let Some(vals) = self.by_ledger.get(ledger_id) {
            self.by_ledger.touch(ledger_id);
            pre(vals.len());
            vals.iter()
                .for_each(|(node_id, val)| {
                    f(node_id, val)
                });
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
pub enum ValidationStatus {
    Current,
    Stale,
    BadSeq,
    Multiple,
    Conflicting,
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::{Duration, SystemTime};

    use xrpl_consensus_core::{Ledger, LedgerIndex};

    use crate::adaptor::Adaptor;
    use crate::arena_ledger_trie::ArenaLedgerTrie;
    use crate::test_utils::ledgers::{LedgerHistoryHelper, LedgerOracle, SimulatedLedger};
    use crate::test_utils::validation::{PeerId, PeerKey, TestValidation};
    use crate::validation_params::ValidationParams;
    use crate::validations::{Validations, ValidationStatus};

    #[test]
    fn test_add_validations() {
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
        assert_eq!(harness.add(&v), ValidationStatus::Current);

        // Re-adding violates the increasing seq requirement for full
        // validations
        assert_eq!(harness.add(&v), ValidationStatus::BadSeq);

        assert_eq!(harness.add(&node.validate_ledger(&ab)), ValidationStatus::Current);

        // Test the node changing signing key

        // Confirm old ledger on hand, but not new ledger
        // let validations_mut: &mut TestValidations = harness.validations_mut();
        assert_eq!(harness.validations.num_trusted_for_ledger(&ab.id()), 1);
        assert_eq!(harness.validations.num_trusted_for_ledger(&abc.id()), 0);


        // Rotate signing keys
        node.advance_key();

        // Cannot re-do the same full validation sequence
        assert_eq!(harness.add(&node.validate_ledger(&ab)), ValidationStatus::Conflicting);

        // Cannot send the same partial validation sequence
        assert_eq!(harness.add(&node.partial(&ab)), ValidationStatus::Conflicting);

        // Now trusts the newest ledger too
        assert_eq!(harness.add(&node.validate_ledger(&abc)), ValidationStatus::Current);
        assert_eq!(harness.validations.num_trusted_for_ledger(&ab.id()), 1);
        assert_eq!(harness.validations.num_trusted_for_ledger(&abc.id()), 1);

        // Processing validations out of order should ignore the older
        // validation
        let val_abcde = node.validate_ledger(&abcde);
        let val_abcd = node.validate_ledger(&abcd);

        assert_eq!(harness.add(&val_abcd), ValidationStatus::Current);
        assert_eq!(harness.add(&val_abcde), ValidationStatus::Stale);
    }

    #[test]
    fn test_add_validations_out_of_order_with_shifted_times() {
        let mut h = LedgerHistoryHelper::new();
        let a = h.get_or_create("a");
        let ab = h.get_or_create("ab");
        let az = h.get_or_create("az");
        let abc = h.get_or_create("abc");
        let abcd = h.get_or_create("abcd");
        let abcde = h.get_or_create("abcde");

        let mut harness = TestHarness::new(h.oracle_mut());
        let mut node = harness.make_node();

        // Establish a new current validation
        assert_eq!(harness.add(&node.validate_ledger(&a)), ValidationStatus::Current);

        // Process a validation that has "later" seq but early sign time
        assert_eq!(harness.add(&node.validate_full(
            &ab,
            DurationOffset::Minus(Duration::from_secs(1)),
            DurationOffset::Minus(Duration::from_secs(1)),
        )), ValidationStatus::Stale);

        // Process a validation that has a later seq and later sign
        // time
        assert_eq!(harness.add(&node.validate_full(
            &abc,
            DurationOffset::Plus(Duration::from_secs(1)),
            DurationOffset::Plus(Duration::from_secs(1)),
        )), ValidationStatus::Current);
    }

    #[test]
    fn test_add_validations_stale_on_arrival() {
        let mut h = LedgerHistoryHelper::new();
        let a = h.get_or_create("a");
        let ab = h.get_or_create("ab");
        let az = h.get_or_create("az");
        let abc = h.get_or_create("abc");
        let abcd = h.get_or_create("abcd");
        let abcde = h.get_or_create("abcde");

        let mut harness = TestHarness::new(h.oracle_mut());
        let mut node = harness.make_node();
        assert_eq!(
            harness.add(&node.validate_full(
                &a,
                DurationOffset::Minus(harness.params().validation_current_early()),
                DurationOffset::Zero,
            )),
            ValidationStatus::Stale
        );

        assert_eq!(
            harness.add(&node.validate_full(
                &a,
                DurationOffset::Plus(harness.params().validation_current_wall()),
                DurationOffset::Zero,
            )),
            ValidationStatus::Stale
        );

        assert_eq!(
            harness.add(&node.validate_full(
                &a,
                DurationOffset::Zero,
                DurationOffset::Plus(harness.params().validation_current_local()),
            )),
            ValidationStatus::Stale
        );
    }

    #[test]
    fn test_add_validations_full_or_partials_cannot_be_sent_for_older_seqs_unless_timeout() {
        let mut h = LedgerHistoryHelper::new();
        let a = h.get_or_create("a");
        let ab = h.get_or_create("ab");
        let az = h.get_or_create("az");
        let abc = h.get_or_create("abc");
        let abcd = h.get_or_create("abcd");
        let abcde = h.get_or_create("abcde");

        let do_test = |do_full: bool| {
            let mut harness = TestHarness::new(h.oracle_mut());
            let node = harness.make_node();

            let mut process = |ledger: &SimulatedLedger| {
                if do_full {
                    harness.add(&node.validate_ledger(ledger))
                } else {
                    harness.add(&node.partial(ledger))
                }
            };

            assert_eq!(process(&abc), ValidationStatus::Current);
            assert!(ab.seq() < abc.seq());
            assert_eq!(process(&ab), ValidationStatus::BadSeq);

            // If we advance far enough for AB to expire, we can fully
            // validate or partially validate that sequence number again
        };
    }

    pub enum DurationOffset {
        Plus(Duration),
        Minus(Duration),
        Zero
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
    }

    impl TestNode {
        pub fn new(node_id: PeerId) -> Self {
            TestNode {
                node_id,
                trusted: true,
                sign_idx: 1,
                load_fee: None,
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

        // pub fn now()

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
                sign_offset.apply_to(SystemTime::now()),
                seen_offset.apply_to(SystemTime::now()),
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
    }

    impl<'a> TestAdaptor<'a> {
        fn new(oracle: &'a mut LedgerOracle) -> Self {
            TestAdaptor {
                oracle,
            }
        }
    }

    impl<'a> Adaptor for TestAdaptor<'a> {
        type ValidationType = TestValidation;
        type LedgerType = SimulatedLedger;
        type LedgerIdType = <Self::LedgerType as Ledger>::IdType;
        type NodeIdType = PeerId;
        type NodeKeyType = PeerKey;

        fn now(&self) -> SystemTime {
            SystemTime::now()
        }

        fn acquire(&mut self, ledger_id: &Self::LedgerIdType) -> Option<Self::LedgerType> {
            self.oracle.lookup(ledger_id)
        }
    }

    type TestValidations<'a> = Validations<TestAdaptor<'a>, ArenaLedgerTrie<SimulatedLedger>>;

    struct TestHarness<'a> {
        params: ValidationParams,
        pub validations: TestValidations<'a>,
        next_node_id: PeerId,
    }

    impl<'a> TestHarness<'a> {
        pub fn new(oracle: &'a mut LedgerOracle) -> Self {
            TestHarness {
                params: Default::default(),
                validations: Validations::new(
                    ValidationParams::default(),
                    TestAdaptor::new(oracle),
                ),
                next_node_id: PeerId(0),
            }
        }

        pub fn add(&mut self, v: &TestValidation) -> ValidationStatus {
            self.validations.add(&v.node_id(), v)
        }

        pub fn make_node(&mut self) -> TestNode {
            self.next_node_id += 1;
            TestNode::new(self.next_node_id)
        }

        pub fn params(&self) -> &ValidationParams {
            &self.params
        }
    }
}