use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use crate::adaptor::{Adaptor, ConsensusCloseTimes};
use crate::ConsensusPhase;

pub struct ConsensusTimer; // TODO
pub struct MonitoredMode; // TODO

pub struct Consensus<T: Adaptor> {
    adaptor: T,
    phase: ConsensusPhase,
    mode: MonitoredMode,
    first_round: bool,
    have_close_time_consensus: bool,
    clock: Instant,
    converge_percent: u32,
    open_time: ConsensusTimer,
    close_resolution: Duration,
    prev_round_time: Duration,
    now: Instant,
    previous_close_time: Instant,
    prev_ledger_id: T::LedgerIdType,
    previous_ledger: T::LedgerType,
    acquired: HashMap<T::TxSetIdType, T::TxSetType>,
    result: Option<T::ResultType>,
    raw_close_times: ConsensusCloseTimes,
    curr_peer_positions: HashMap<T::NodeIdType, T::PeerPositionType>,
    recent_peer_positions: HashMap<T::NodeIdType, VecDeque<T::PeerPositionType>>,
    prev_proposers: usize,
    dead_nodes: HashSet<T::NodeIdType>
}