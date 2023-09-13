use std::time::{Duration, SystemTime};
use xrpl_consensus_core::NetClock;

#[cfg(test)]
pub(crate) mod ledgers;
pub(crate) mod validation;

pub struct ManualClock {
    now: SystemTime
}

impl ManualClock {
    pub fn new() -> Self {
        ManualClock {
            now: SystemTime::now()
        }
    }

    pub fn advance(&mut self, dur: Duration) {
        self.now += dur
    }
}

impl NetClock for ManualClock {
    fn now(&self) -> SystemTime {
        self.now
    }
}