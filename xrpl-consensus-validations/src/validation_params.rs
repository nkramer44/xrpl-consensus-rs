use std::time::Duration;

pub struct ValidationParams {
    validation_current_wall: Duration,
    validation_current_local: Duration,
    validation_currency_early: Duration,
    validation_set_expires: Duration,
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