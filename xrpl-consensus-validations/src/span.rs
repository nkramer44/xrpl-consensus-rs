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

impl<T: Ledger> From<T> for Span<T> {
    fn from(value: T) -> Span<T> {
        Span {
            start: 0,
            end: value.seq() + 1,
            ledger: value
        }
    }
}

impl<T: Ledger> Span<T> {

    fn _new(start: LedgerIndex, end: LedgerIndex, ledger: T) -> Self {
        Span {
            start,
            end,
            ledger,
        }
    }

    pub fn start(&self) -> LedgerIndex {
        self.start
    }

    pub fn end(&self) -> LedgerIndex {
        self.end
    }

    pub fn after(&self, spot: LedgerIndex) -> Option<Span<T>> {
        self._sub(spot, self.end)
    }

    pub fn before(&self, spot: LedgerIndex) -> Option<Span<T>> {
        self._sub(self.start, spot)
    }

    pub fn start_id(&self) -> T::IdType {
        self.ledger.get_ancestor(self.start)
    }

    pub fn diff(&self, other: &T) -> LedgerIndex {
        self._clamp(self.ledger.mismatch(other))
    }

    pub fn tip(&self) -> SpanTip<T> {
        let tip_seq = self.end - 1;
        SpanTip::new(tip_seq, self.ledger.get_ancestor(tip_seq), self.ledger.clone())
    }

    fn _clamp(&self, seq: LedgerIndex) -> LedgerIndex {
        std::cmp::min(std::cmp::max(self.start, seq), self.end)
    }

    fn _sub(&self, from: LedgerIndex, to: LedgerIndex) -> Option<Span<T>> {
        let new_from = self._clamp(from);
        let new_to = self._clamp(to);
        if new_from < new_to {
            return Some(Span::_new(new_from, new_to, self.ledger.clone()));
        }
        None
    }

    pub fn merge(a: &Span<T>, b: &Span<T>) -> Span<T> {
        // Return combined span, using ledger_ from higher sequence span
        if a.end < b.end {
            return Span::_new(
                std::cmp::min(a.start, b.start),
                b.end,
                b.ledger.clone()
            );
        }

        return Span::_new(
            std::cmp::min(a.start, b.start),
            a.end,
            a.ledger.clone()
        )
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