use std::time::Duration;
use crate::ConsensusMode;

pub struct ConsensusCloseTimes {
    // TODO
}

pub struct ConsensusProposal {
    // TODO
}

pub struct ConsensusParams {

}

pub trait Adaptor {
    type LedgerType;
    type LedgerIdType;
    type TxSetType;
    type TxSetIdType;
    type ResultType;
    type NodeIdType;
    type PeerPositionType;
    type ProposalType;
    type TxType;

    /// Attempt to acquire a specific ledger.
    fn acquire_ledger(&mut self, ledger_id: &Self::LedgerIdType) -> Option<Self::LedgerType>;

    /// Acquire the transaction set associated with a proposed position.
    fn acquire_tx_set(&mut self, set_id: &Self::TxSetIdType) -> Option<Self::TxSetType>;

    /// Whether any transactions are in the open ledger
    fn has_open_transactions(&self) -> bool;

    /// Number of proposers that have validated the given ledger
    fn proposers_validated(&self, prev_ledger: &Self::LedgerIdType) -> usize;

    /// Number of proposers that have validated a ledger descended from the
    /// given ledger; if prev_ledger.id() != prev_ledger_id, use prev_ledger_id
    /// for the determination
    fn proposers_finished(
        &self,
        prev_ledger: &Self::LedgerType,
        prev_ledger_id: &Self::LedgerIdType,
    ) -> usize;

    fn get_prev_ledger(
        &mut self,
        prev_ledger_id: &Self::LedgerIdType,
        prev_ledger: &Self::LedgerType,
        mode: ConsensusMode,
    ) -> Self::LedgerIdType;

    fn on_mode_change(
        &mut self,
        before: ConsensusMode,
        after: ConsensusMode,
    );

    fn on_close(
        &mut self,
        ledger: &Self::LedgerType,
        prev: &Self::LedgerType,
        mode: ConsensusMode,
    ) -> Self::ResultType;

    fn on_accept(
        &mut self,
        result: &Self::ResultType,
        prev_ledger: &Self::LedgerType,
        close_resolution: Duration,
        raw_close_times: &ConsensusCloseTimes,
        mode: ConsensusMode,
    );

    fn on_force_accept(
        &mut self,
        result: &Self::ResultType,
        prev_ledger: &Self::LedgerType,
        close_resolution: Duration,
        raw_close_times: &ConsensusCloseTimes,
        mode: ConsensusMode,
    );

    fn propose(
        &mut self,
        position: &Self::ProposalType
    );

    fn share_peer_position(&mut self, proposal: &Self::PeerPositionType);

    fn share_tx(&mut self, tx: &Self::TxType);

    fn share_tx_set(&mut self, tx: &Self::TxSetType);

    fn params(&self) -> &ConsensusParams;
}