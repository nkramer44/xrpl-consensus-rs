use std::marker::PhantomData;
use xrpl_consensus_core::{Ledger, LedgerIndex};

pub struct LedgerTrie<T: Ledger> {
    t: PhantomData<T>,
}

impl<T: Ledger> LedgerTrie<T> {
    pub fn get_preferred(&self, largest_issued: LedgerIndex) -> Option<SpanTip<T>>{
        todo!()
    }
}


pub struct SpanTip<T: Ledger> {
    t: PhantomData<T>,
}

impl<T: Ledger> SpanTip<T> {
    pub(crate) fn id(&self) -> T::IdType {
        todo!()
    }
}

impl<T: Ledger> SpanTip<T> {
    pub(crate) fn ancestor(&self, seq: LedgerIndex) -> T::IdType {
        todo!()
    }
}

impl<T: Ledger> SpanTip<T> {
    pub(crate) fn seq(&self) -> LedgerIndex {
        todo!()
    }
}