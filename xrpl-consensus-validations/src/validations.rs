use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard};
use crate::adaptor::Adaptor;
use crate::ledger_trie::LedgerTrie;
use crate::validation_params::ValidationParams;

type LedgerIndex = u32;

struct AgedUnorderedMap<K, V> {
    // TODO: This should be a port/replica of beast::aged_unordered_map
    v: PhantomData<K>,
    k: PhantomData<V>
}

struct SeqEnforcer {
    // TODO
}

struct KeepRange {
    pub low: LedgerIndex,
    pub high: LedgerIndex,
}

pub struct Validations<T: Adaptor> {
    /// Manages concurrent access to members
    mutex: Mutex<()>,
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

    pub fn can_validate_seq(&self, seq: LedgerIndex) -> bool {
        todo!()
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

    pub fn get_preferred(&mut self, curr: &T::LedgerType) -> Option<(LedgerIndex, T::LedgerIdType)> {
        todo!()
    }

    pub fn get_preferred_id(
        &mut self,
        curr: &T::LedgerType,
        min_valid_seq: LedgerIndex,
    ) -> T::LedgerIdType {
        todo!()
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

    fn _with_trie<F: FnMut(&mut T::ValidationType)>(
        &mut self,
        lock: MutexGuard<()>,
        f: F,
    ) {
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