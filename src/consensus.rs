#![allow(dead_code)] // FIXME: Remove this eventually
#![allow(unused_variables)] // FIXME: Remove this eventually

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ops::{Add, Sub};
use std::time::{Duration, Instant};
use crate::adaptor::Adaptor;
use crate::{ConsensusMode, ConsensusPhase};
use crate::consensus_times::{ConsensusCloseTimes, ConsensusTimer};

/// Wrapper struct to ensure the Adaptor is notified whenever the ConsensusMode changes
struct MonitoredMode(ConsensusMode);

impl MonitoredMode {
    fn get(&self) -> ConsensusMode {
        self.0
    }

    fn set<T: Adaptor>(&mut self, mode: ConsensusMode, adaptor: &mut T) {
        adaptor.on_mode_change(self.0, mode);
        self.0 = mode;
    }
}

pub struct Consensus<'a, T: Adaptor> {
    adaptor: &'a mut T,
    phase: ConsensusPhase,
    mode: MonitoredMode,
    first_round: bool,
    have_close_time_consensus: bool,
    // clock: Instant,
    converge_percent: u32,
    open_time: ConsensusTimer,
    close_resolution: Duration,
    prev_round_time: Option<Duration>,
    now: Option<Instant>,
    previous_close_time: Option<Instant>,
    prev_ledger_id: Option<T::LedgerIdType>,
    previous_ledger: Option<T::LedgerType>,
    acquired: HashMap<T::TxSetIdType, T::TxSetType>,
    result: Option<T::ResultType>,
    raw_close_times: Option<ConsensusCloseTimes>,
    curr_peer_positions: HashMap<T::NodeIdType, T::PeerPositionType>,
    recent_peer_positions: HashMap<T::NodeIdType, VecDeque<T::PeerPositionType>>,
    prev_proposers: usize,
    dead_nodes: HashSet<T::NodeIdType>
}

impl <'a, T: Adaptor> Consensus<'a, T> {
    pub fn new(adaptor: &'a mut T) -> Consensus<T> {
        Consensus {
            adaptor,
            phase: ConsensusPhase::Accepted,
            mode: MonitoredMode(ConsensusMode::Observing),
            first_round: true,
            have_close_time_consensus: false,
            converge_percent: 0,
            open_time: ConsensusTimer::default(),
            close_resolution: Duration::from_secs(30), // Taken from LedgerTiming.h ledgerDefaultTimeResolution
            prev_round_time: None,
            now: None,
            previous_close_time: None,
            prev_ledger_id: None,
            previous_ledger: None,
            acquired: HashMap::new(),
            result: None,
            raw_close_times: None,
            curr_peer_positions: HashMap::new(),
            recent_peer_positions: HashMap::new(),
            prev_proposers: 0,
            dead_nodes: HashSet::new(),
        }
    }

    pub fn start_round(
        &mut self,
        now: Instant,
        prev_ledger_id: &T::LedgerIdType,
        prev_ledger: T::LedgerType,
        now_untrusted: &HashSet<T::NodeIdType>,
        proposing: bool
    ) {
        todo!()
    }

    pub fn peer_proposal(&mut self,now: Instant, new_proposal: T::PeerPositionType) -> bool {
        todo!()
    }

    pub fn timer_entry(&mut self, now: Instant) {
        todo!()
    }

    pub fn got_tx_set(&mut self, now: Instant, tx_set: &T::TxSetType) {
        todo!()
    }

    pub fn simulate(&mut self, now: Instant, consensus_delay: Option<Duration>) {
        todo!()
    }

    pub fn prev_ledger_id(&self) -> &T::LedgerIdType {
        todo!()
    }

    pub fn phase(&self) -> ConsensusPhase {
        todo!()
    }

    // TODO: Serde instead of rippled getJson method?
}

impl <'a, T: Adaptor> Consensus<'a, T> {
    fn _start_round(
        &mut self,
        now: &Instant,
        prev_ledger_id: &T::LedgerIdType,
        prev_ledger: &T::LedgerType,
        mode: ConsensusMode
    ) {
        todo!()
    }

    fn _handle_wrong_ledger(&mut self, ledger_id: T::LedgerIdType) {
        todo!()
    }

    fn _check_ledger(&mut self) {
        todo!()
    }

    fn _playback_proposals(&mut self) {
        todo!()
    }

    fn _peer_proposal(
        &mut self,
        now: &Instant,
        new_proposal: &T::PeerPositionType
    ) -> bool {
        todo!()
    }

    fn _phase_open(&mut self) {
        todo!()
    }

    fn _phase_establish(&mut self) {
        todo!()
    }

    fn _should_pause(&self) -> bool {
        todo!()
    }

    fn _close_ledger(&mut self) {
        todo!()
    }

    fn _update_our_positions(&mut self) {
        todo!()
    }

    fn _have_consensus(&mut self) -> bool {
        todo!()
    }

    fn _create_disputes(&mut self, tx_set: &T::TxSetType) {
        todo!()
    }

    fn _update_disputes(&mut self, node: &T::NodeIdType, other: &T::TxSetType) {
        todo!()
    }

    fn _leave_consensus(&mut self) {
        todo!()
    }

    fn _as_close_time(&self) -> Instant {
        todo!()
    }
}