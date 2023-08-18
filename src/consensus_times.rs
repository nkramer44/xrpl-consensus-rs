use std::collections::BTreeMap;
use std::ops::{Add, Sub};
use std::time::{Duration, Instant};

pub struct ConsensusCloseTimes {
    /// Close time estimates, keep ordered for predictable traverse
    peers: BTreeMap<Instant, u64>,
    ours: Instant
}

impl ConsensusCloseTimes {
    // TODO
}

pub struct ConsensusTimer {
    start: Instant,
    duration: Duration
}

impl Default for ConsensusTimer {
    fn default() -> Self {
        ConsensusTimer {
            start: Instant::now(),
            duration: Duration::ZERO
        }
    }
}

impl ConsensusTimer {
    pub fn read(&self) -> &Duration {
        &self.duration
    }

    pub fn tick_duration(&mut self, fixed: Duration) {
        self.duration = self.duration.add(fixed);
    }

    pub fn reset(&mut self, time_point: Instant) {
        self.start = time_point;
        self.duration = Duration::ZERO;
    }

    pub fn tick_fixed(&mut self, time_point: Instant) {
        self.duration = time_point.sub(self.start);
    }
}