use xrpl_consensus_core::{Ledger, LedgerIndex};

pub struct SpanTip<T: Ledger> {
    /// The sequence number of the tip ledger.
    seq: LedgerIndex,
    /// The ID of the tip ledger.
    id: T::IdType,
    ledger: T,
}
impl<T: Ledger> SpanTip<T> {
    pub(crate) fn new(seq: LedgerIndex, id: T::IdType, ledger: T) -> Self {
        SpanTip {
            seq,
            id,
            ledger,
        }
    }

    pub(crate) fn id(&self) -> T::IdType {
        self.id
    }

    pub(crate) fn seq(&self) -> LedgerIndex {
        self.seq
    }

    pub(crate) fn ancestor(&self, seq: LedgerIndex) -> T::IdType {
        self.ledger.get_ancestor(seq)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Span<T: Ledger> {
    start: LedgerIndex,
    end: LedgerIndex,
    ledger: T
}

impl<T: Ledger> Span<T> {
    pub fn new(ledger: T) -> Span<T> {
        Span {
            start: 0,
            end: ledger.seq() + 1,
            ledger
        }
    }

    pub fn start(&self) -> LedgerIndex {
        self.start
    }

    pub fn end(&self) -> LedgerIndex {
        self.end
    }

    pub fn from(&self, spot: LedgerIndex) -> Option<Span<T>> {
        todo!()
    }

    pub fn before(&self, spot: LedgerIndex) -> Option<Span<T>> {
        todo!()
    }

    pub fn start_id(&self) -> T::IdType {
        todo!()
    }

    pub fn diff(&self, other: &T) -> LedgerIndex {
        todo!()
    }

    pub fn tip(&self) -> SpanTip<T> {
        todo!()
    }

    fn _clamp(&self, seq: LedgerIndex) -> LedgerIndex {
        todo!()
    }

    fn _sub(&self, from: LedgerIndex, to: LedgerIndex) -> Option<Span<T>> {
        todo!()
    }

    fn merge(a: &Span<T>, b: Span<T>) -> Span<T> {
        todo!()
    }
}

impl<T: Ledger> Default for Span<T> {
    fn default() -> Self {
        Span {
            start: 0,
            end: 1,
            ledger: T::make_genesis()
        }
    }
}