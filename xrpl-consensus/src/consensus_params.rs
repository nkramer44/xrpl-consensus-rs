use std::time::Duration;

/// Consensus algorithm parameters
///
/// Parameters which control the consensus algorithm.  These are not meant to be changed arbitrarily.
#[derive(Debug, PartialEq)]
pub struct ConsensusParams {
    // TODO: These may not be auto-converted in c++ due to the `explicit` keyword.

    /// Duration a validation remains current after first observed.
    ///
    /// The duration a validation remains current after its ledger's close time.
    /// This is a safety to protect against very old validations and the time it takes to adjust the
    /// close time accuracy window.
    validation_valid_wall: Duration,

    /// The duration a validation remains current after the time we first saw it.
    ///
    /// This provides faster recovery in very rare cases where the number of validations produced by
    /// the network is lower than normal
    validation_valid_local: Duration,

    /// Duration pre-close in which validations are acceptable.
    ///
    /// The number of seconds before a close time that we consider a validation acceptable. This
    /// protects against extreme clock errors
    validation_valid_early: Duration,

    /// How long we consider a proposal to be fresh.
    propose_freshness: Duration,

    /// How often we force generating a new proposal to keep ours fresh
    propose_interval: Duration,

    /// The percentage threshold above which we can declare consensus.
    min_consensus_pct: usize,

    /// The duration a ledger may remain idle before closing
    ledger_idle_interval: Duration,

    /// The number of seconds we wait minimum to ensure participation
    ledger_min_consensus: Duration,

    /// The maximum amount of time to spend pausing for laggards.
    ///
    /// This should be sufficiently less than validationFRESHNESS so that validators don't appear to
    /// be offline that are merely waiting for laggards.
    ledger_max_consensus: Duration,

    /// Minimum number of seconds to wait to ensure others have computed the LCL
    ledger_min_close: Duration,

    /// How often we check state or change positions
    ledger_granularity: Duration,

    /// The minimum amount of time to consider the previous round to have taken.
    ///
    /// The minimum amount of time to consider the previous round to have taken. This ensures that
    /// there is an opportunity for a round at each avalanche threshold even if the previous
    /// consensus was very fast. This should be at least twice the interval between proposals (0.7s)
    /// divided by the interval between mid and late consensus ([85-50]/100).
    av_min_consensus_time: Duration,

    //------------------------------------------------------------------------------
    // Avalanche tuning
    // As a function of the percent this round's duration is of the prior round, we increase the
    // threshold for yes votes to add a transaction to our position.

    /// Percentage of nodes on our UNL that must vote yes
    av_init_consensus_pct: usize,

    /// Percentage of previous round duration before we advance
    av_mid_consensus_time: usize,

    /// Percentage of nodes that most vote yes after advancing
    av_mid_consensus_pct: usize,

    /// Percentage of previous round duration before we advance
    av_late_consensus_time: usize,

    /// Percentage of nodes that most vote yes after advancing
    av_late_consensus_pct: usize,

    /// Percentage of previous round duration before we are stuck
    av_stuck_consensus_time: usize,

    /// Percentage of nodes that must vote yes after we are stuck
    av_stuck_consensus_pct: usize,

    /// Percentage of nodes required to reach agreement on ledger close time
    av_ct_consensus_pct: usize,
}

impl Default for ConsensusParams {
    // TODO: Consider _derivative_ to define default values
    // (See https://mcarton.github.io/rust-derivative/latest/index.html)
    fn default() -> Self {
        ConsensusParams {
            validation_valid_wall: Duration::from_secs(5 * 60),
            validation_valid_local: Duration::from_secs(3 * 60),
            validation_valid_early: Duration::from_secs(5 * 60),
            propose_freshness: Duration::from_secs(20),
            propose_interval: Duration::from_secs(12),
            min_consensus_pct: 80,
            ledger_idle_interval: Duration::from_secs(15),
            ledger_min_consensus: Duration::from_millis(1950),
            ledger_max_consensus: Duration::from_secs(10),
            ledger_min_close: Duration::from_secs(2),
            ledger_granularity: Duration::from_secs(1),
            av_min_consensus_time: Duration::from_secs(5),
            av_init_consensus_pct: 50,
            av_mid_consensus_time: 50,
            av_mid_consensus_pct: 65,
            av_late_consensus_time: 85,
            av_late_consensus_pct: 70,
            av_stuck_consensus_time: 200,
            av_stuck_consensus_pct: 95,
            av_ct_consensus_pct: 75,
        }
    }
}

impl ConsensusParams {
    pub fn validation_valid_wall(&self) -> &Duration {
        &self.validation_valid_wall
    }
    pub fn validation_valid_local(&self) -> &Duration {
        &self.validation_valid_local
    }
    pub fn validation_valid_early(&self) -> &Duration {
        &self.validation_valid_early
    }
    pub fn propose_freshness(&self) -> &Duration {
        &self.propose_freshness
    }
    pub fn propose_interval(&self) -> &Duration {
        &self.propose_interval
    }
    pub fn min_consensus_pct(&self) -> usize {
        self.min_consensus_pct
    }
    pub fn ledger_idle_interval(&self) -> &Duration {
        &self.ledger_idle_interval
    }
    pub fn ledger_min_consensus(&self) -> &Duration {
        &self.ledger_min_consensus
    }
    pub fn ledger_max_consensus(&self) -> &Duration {
        &self.ledger_max_consensus
    }
    pub fn ledger_min_close(&self) -> &Duration {
        &self.ledger_min_close
    }
    pub fn ledger_granularity(&self) -> &Duration {
        &self.ledger_granularity
    }
    pub fn av_min_consensus_time(&self) -> &Duration {
        &self.av_min_consensus_time
    }
    pub fn av_init_consensus_pct(&self) -> usize {
        self.av_init_consensus_pct
    }
    pub fn av_mid_consensus_time(&self) -> usize {
        self.av_mid_consensus_time
    }
    pub fn av_mid_consensus_pct(&self) -> usize {
        self.av_mid_consensus_pct
    }
    pub fn av_late_consensus_time(&self) -> usize {
        self.av_late_consensus_time
    }
    pub fn av_late_consensus_pct(&self) -> usize {
        self.av_late_consensus_pct
    }
    pub fn av_stuck_consensus_time(&self) -> usize {
        self.av_stuck_consensus_time
    }
    pub fn av_stuck_consensus_pct(&self) -> usize {
        self.av_stuck_consensus_pct
    }
    pub fn av_ct_consensus_pct(&self) -> usize {
        self.av_ct_consensus_pct
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    use crate::consensus_params::ConsensusParams;

    //noinspection ALL
    #[test]
    fn test_defaults() {
        let expected = ConsensusParams {
            validation_valid_wall: Duration::from_secs(5 * 60),
            validation_valid_local: Duration::from_secs(3 * 60),
            validation_valid_early: Duration::from_secs(5 * 60),
            propose_freshness: Duration::from_secs(20),
            propose_interval: Duration::from_secs(12),
            min_consensus_pct: 80,
            ledger_idle_interval: Duration::from_secs(15),
            ledger_min_consensus: Duration::from_millis(1950),
            ledger_max_consensus: Duration::from_secs(10),
            ledger_min_close: Duration::from_secs(2),
            ledger_granularity: Duration::from_secs(1),
            av_min_consensus_time: Duration::from_secs(5),
            av_init_consensus_pct: 50,
            av_mid_consensus_time: 50,
            av_mid_consensus_pct: 65,
            av_late_consensus_time: 85,
            av_late_consensus_pct: 70,
            av_stuck_consensus_time: 200,
            av_stuck_consensus_pct: 95,
            av_ct_consensus_pct: 75,
        };

        let default: ConsensusParams = ConsensusParams::default();
        assert_eq!(expected, default);
    }
}