use std::marker::PhantomData;

pub struct LedgerTrie<T> {
    t: PhantomData<T>,
}