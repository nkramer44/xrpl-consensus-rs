use std::time::Duration;

pub struct ConsensusParams {
    pub validation_valid_wall: Duration,
    pub validation_valid_local: Duration,
    pub validation_valid_early: Duration,
    pub propose_freshness: Duration,
    pub propose_interval: Duration,
    pub min_consensus_percentage: u8,
    pub ledger_idle_interval: Duration,
    pub ledger_min_consensus: Duration,
    pub ledger_max_consensus: Duration,
    pub ledger_min_close: Duration,
    pub ledger_granularity: Duration,
    pub av_min_consensus_time: Duration,
    pub av_init_consensus_percentage: u8,
    pub av_mid_consensus_time: u8,
    pub av_mid_consensus_percentage: u8,
    pub av_late_consensus_time: u8,
    pub av_late_consensus_percentage: u8,
    pub av_stuck_consensus_time: u8,
    pub av_stuck_consensus_percentage: u8,
    pub av_ct_consensus_percentage: u8,
}

impl Default for ConsensusParams {
    fn default() -> Self {
        ConsensusParams {
            validation_valid_wall: Duration::from_secs(5 * 60),
            validation_valid_local: Duration::from_secs(3 * 60),
            validation_valid_early: Duration::from_secs(3 * 60),
            propose_freshness: Duration::from_secs(20),
            propose_interval: Duration::from_secs(12),
            min_consensus_percentage: 80,
            ledger_idle_interval: Duration::from_secs(15),
            ledger_min_consensus: Duration::from_millis(1950),
            ledger_max_consensus: Duration::from_secs(10),
            ledger_min_close: Duration::from_secs(2),
            ledger_granularity: Duration::from_secs(1),
            av_min_consensus_time: Duration::from_secs(5),
            av_init_consensus_percentage: 50,
            av_mid_consensus_time: 50,
            av_mid_consensus_percentage: 65,
            av_late_consensus_time: 85,
            av_late_consensus_percentage: 70,
            av_stuck_consensus_time: 200,
            av_stuck_consensus_percentage: 95,
            av_ct_consensus_percentage: 75,
        }
    }
}