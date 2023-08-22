mod adaptor;
mod consensus;
mod consensus_params;

/// Represents how a node currently participates in Consensus.
///
/// A node participates in consensus in varying modes, depending on how
/// the node was configured by its operator and how well it stays in sync
/// with the network during consensus.
///
///
/// Proposing               Observing
///    \                       /
///     \---> wrongLedger <---/
///                ^
///                |
///                |
///                v
///          SwitchedLedger
///
///
/// We enter the round Proposing or Observing. If we detect we are working
/// on the wrong prior ledger, we go to WrongLedger and attempt to acquire
/// the right one. Once we acquire the right one, we go to the SwitchedLedger
/// mode.  It is possible we fall behind again and find there is a new better
/// ledger, moving back and forth between WrongLedger and SwitchLedger as
/// we attempt to catch up.
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum ConsensusMode {
    /// We are normal participant in consensus and propose our position
    Proposing,
    /// We are observing peer positions, but not proposing our position
    Observing,
    /// We have the wrong ledger and are attempting to acquire it
    WrongLedger,
    /// We switched ledgers since we started this consensus round but are now
    /// running on what we believe is the correct ledger.  This mode is as
    /// if we entered the round observing, but is used to indicate we did
    /// have the wrongLedger at some point.
    SwitchedLedger
}

pub enum ConsensusPhase {
    Open,
    Establish,
    Accepted
}