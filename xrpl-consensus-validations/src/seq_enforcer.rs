use std::ops::Add;
use std::time::SystemTime;

use xrpl_consensus_core::LedgerIndex;

use crate::validation_params::ValidationParams;

/// Enforce validation increasing sequence requirement.
///
/// Helper struct for enforcing that a validation must be larger than all
/// unexpired validation sequence numbers previously issued by the validator
/// tracked by the instance of this struct.
pub(crate) struct SeqEnforcer {
    seq: LedgerIndex,
    when: SystemTime
}

impl SeqEnforcer {
    pub fn new() -> Self {
        SeqEnforcer {
            seq: 0,
            // TODO: Not clear what this should be. In c++, this is a
            //  std::chrono::steady_clock::time_point, which may default to the start of the Unix epoch
            when: SystemTime::now(),
        }
    }

    /// Try advancing the largest observed validation ledger sequence.
    ///
    /// Try setting the largest validation sequence observed, but return false
    /// if it violates the invariant that a validation must be larger than all
    /// unexpired validation sequence numbers.
    ///
    /// # Params
    /// - now: The current time
    /// - s: The sequence number we want to validate
    /// - p: Validation parameters
    ///
    /// # Returns
    /// A bool indicating whether the validation satisfies the invariant.
    pub fn advance_ledger(&mut self, now: SystemTime, seq: LedgerIndex, params: &ValidationParams) -> bool {
        if now > self.when.add(params.validation_set_expires()) {
            self.seq = 0;
        }
        if seq <= self.seq {
            return false;
        }

        self.seq = seq;
        self.when = now;
        true
    }


    pub fn largest(&self) -> LedgerIndex {
        self.seq
    }

    pub fn when(&self) -> SystemTime {
        self.when
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn advance_resets_seq_when_validations_expire_and_s_is_0() {
        let params = ValidationParams::default();
        let mut enforcer = SeqEnforcer::new();
        let when_before_advance = enforcer.when();
        let advanced = enforcer.advance_ledger(
            SystemTime::now().add(params.validation_set_expires().add(Duration::from_secs(1))),
            0,
            &params
        );

        assert_eq!(enforcer.largest(), 0);
        assert_eq!(enforcer.when, when_before_advance);
        assert!(!advanced)
    }

    #[test]
    fn advance_does_not_reset_seq_when_validations_expire_and_s_gt_0() {
        let params = ValidationParams::default();
        let mut enforcer = SeqEnforcer::new();
        let when_before_advance = enforcer.when;
        let now = SystemTime::now().add(params.validation_set_expires().add(Duration::from_secs(1)));
        let advanced = enforcer.advance_ledger(
            now,
            10,
            &params
        );

        assert_eq!(enforcer.largest(), 10);
        assert_eq!(enforcer.when, now);
        assert_ne!(when_before_advance, now);
        assert!(advanced)
    }

    #[test]
    fn advance_returns_false_when_s_less_than_seq() {
        let params = ValidationParams::default();
        let mut enforcer = SeqEnforcer::new();
        let now = SystemTime::now();
        let advanced = enforcer.advance_ledger(
            now,
            10,
            &params
        );

        assert!(advanced);
        let advanced2 = enforcer.advance_ledger(
            SystemTime::now(),
            10,
            &params
        );

        assert!(!advanced2);

        let advanced3 = enforcer.advance_ledger(
            SystemTime::now(),
            9,
            &params
        );

        assert!(!advanced3);

        assert_eq!(enforcer.largest(), 10);
        assert_eq!(enforcer.when, now);
    }

    #[test]
    fn advance_succeeds() {
        let params = ValidationParams::default();
        let mut enforcer = SeqEnforcer::new();
        let now = SystemTime::now();
        let advanced = enforcer.advance_ledger(
            now,
            10,
            &params
        );

        assert!(advanced);
        assert_eq!(enforcer.largest(), 10);
        assert_eq!(enforcer.when, now);
    }
}