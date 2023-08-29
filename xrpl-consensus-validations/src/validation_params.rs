use std::time::Duration;

/// Timing parameters to control validation staleness and expiration.
///
/// ## Note
/// These are protocol level parameters that should not be changed without
/// careful consideration.  They are *not* implemented as static
/// to allow simulation code to test alternate parameter settings.
pub struct ValidationParams {
    /// The number of seconds a validation remains current after its ledger's
    /// close time.
    ///
    /// This is a safety to protect against very old validations and the time
    /// it takes to adjust the close time accuracy window.
    validation_current_wall: Duration,
    /// Duration a validation remains current after first observed.
    ///
    /// The number of seconds a validation remains current after the time we
    /// first saw it. This provides faster recovery in very rare cases where the
    /// number of validations produced by the network is lower than normal.
    validation_current_local: Duration,
    /// Duration pre-close in which validations are acceptable.
    ///
    /// The number of seconds before a close time that we consider a validation
    /// acceptable. This protects against extreme clock errors.
    validation_currency_early: Duration,
    /// Duration a set of validations for a given ledger hash remain valid.
    ///
    /// The number of seconds before a set of validations for a given ledger
    /// hash can expire.  This keeps validations for recent ledgers available
    /// for a reasonable interval.
    validation_set_expires: Duration,
    /// How long we consider a validation fresh.
    ///
    /// The number of seconds since a validation has been seen for it to
    /// be considered to accurately represent a live proposer's most recent
    /// validation. This value should be sufficiently higher than
    /// ledgerMAX_CONSENSUS such that validators who are waiting for
    /// laggards are not considered offline.
    validation_freshness: Duration
}

impl Default for ValidationParams {
    fn default() -> Self {
        ValidationParams {
            validation_current_wall: Duration::from_secs(5 * 60),
            validation_current_local: Duration::from_secs(3 * 60),
            validation_currency_early: Duration::from_secs(3 * 60),
            validation_set_expires: Duration::from_secs(10 * 60),
            validation_freshness: Duration::from_secs(20),
        }
    }
}

impl ValidationParams {
    pub fn validation_current_wall(&self) -> Duration {
        self.validation_current_wall
    }
    pub fn validation_current_local(&self) -> Duration {
        self.validation_current_local
    }
    pub fn validation_currency_early(&self) -> Duration {
        self.validation_currency_early
    }
    pub fn validation_set_expires(&self) -> Duration {
        self.validation_set_expires
    }
    pub fn validation_freshness(&self) -> Duration {
        self.validation_freshness
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let params = ValidationParams::default();
        assert_eq!(params.validation_current_wall(), Duration::from_secs(5 * 60));
        assert_eq!(params.validation_current_local(), Duration::from_secs(3 * 60));
        assert_eq!(params.validation_currency_early(), Duration::from_secs(3 * 60));
        assert_eq!(params.validation_set_expires(), Duration::from_secs(10 * 60));
        assert_eq!(params.validation_freshness(), Duration::from_secs(20));
    }
}