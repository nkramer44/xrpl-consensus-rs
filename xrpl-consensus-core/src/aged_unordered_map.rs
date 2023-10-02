use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;
use crate::NetClock;

pub struct AgedUnorderedMap<K, V, C>
    where
        K: Eq + PartialEq + Hash,
        V: Default,
        C: NetClock {
    // TODO: This should be a port/replica of beast::aged_unordered_map. The only
    //  difference between a HashMap and beast::aged_unordered_map is that you can
    //  expire entries, but for simplicity's sake, we can just not expire entries.
    inner: HashMap<K, V>,
    clock: Arc<RwLock<C>>,
}

impl<K: Eq + PartialEq + Hash, V: Default, C: NetClock> AgedUnorderedMap<K, V, C> {
    pub fn new(clock: Arc<RwLock<C>>) -> Self {
        AgedUnorderedMap {
            inner: Default::default(),
            clock,
        }
    }

    pub fn now(&self) -> SystemTime {
        self.clock.read().unwrap().now()
    }

    pub fn get_or_insert(&mut self, k: K) -> &V {
        self.get_or_insert_mut(k)
    }

    pub fn get_or_insert_mut(&mut self, k: K) -> &mut V {
        self.inner.entry(k).or_default()
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        self.inner.get(k)
    }

    pub fn touch(&self, k: &K) {
        // TODO: Update the timestamp on the entry
    }
}