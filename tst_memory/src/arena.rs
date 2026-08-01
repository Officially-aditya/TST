use std::collections::BTreeSet;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Arena<T> {
    items: Vec<T>,
    free_list: BTreeSet<u32>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            free_list: BTreeSet::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            free_list: BTreeSet::new(),
        }
    }

    /// Allocates an item and returns its index.
    pub fn alloc(&mut self, value: T) -> u32 {
        if let Some(idx) = self.free_list.pop_first() {
            self.items[idx as usize] = value;
            idx
        } else {
            let idx = u32::try_from(self.items.len()).expect("arena index space exhausted");
            self.items.push(value);
            idx
        }
    }

    /// Borrows the item at index.
    pub fn get(&self, idx: u32) -> Option<&T> {
        if self.free_list.contains(&idx) {
            return None;
        }
        self.items.get(idx as usize)
    }

    /// Mutably borrows the item at index.
    pub fn get_mut(&mut self, idx: u32) -> Option<&mut T> {
        if self.free_list.contains(&idx) {
            return None;
        }
        self.items.get_mut(idx as usize)
    }

    /// Marks an active index as reusable. Invalid and repeated frees are no-ops.
    pub fn free(&mut self, idx: u32) -> bool {
        if idx as usize >= self.items.len() {
            return false;
        }
        self.free_list.insert(idx)
    }

    pub fn len(&self) -> usize {
        self.items.len().saturating_sub(self.free_list.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn slot_count(&self) -> usize {
        self.items.len()
    }

    pub fn active_indices(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.items.len()).filter_map(|idx| {
            let idx = u32::try_from(idx).ok()?;
            (!self.free_list.contains(&idx)).then_some(idx)
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.items.len() > u32::MAX as usize {
            return Err("arena exceeds the u32 index space".to_string());
        }
        if let Some(idx) = self
            .free_list
            .iter()
            .find(|idx| **idx as usize >= self.items.len())
        {
            return Err(format!("arena free-list index {idx} is out of range"));
        }
        Ok(())
    }
}

pub trait DummyItem {
    fn dummy() -> Self;
}

impl<T: DummyItem> Arena<T> {
    /// Frees an index and overwrites it with a dummy (tombstone) value.
    pub fn free_with_tombstone(&mut self, idx: u32) -> bool {
        if let Some(item) = self.get_mut(idx) {
            *item = T::dummy();
            return self.free_list.insert(idx);
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_get() {
        let mut arena = Arena::new();
        let i1 = arena.alloc(10);
        let i2 = arena.alloc(20);

        assert_eq!(i1, 0);
        assert_eq!(i2, 1);

        assert_eq!(arena.get(i1), Some(&10));
        assert_eq!(arena.get(i2), Some(&20));
        assert_eq!(arena.len(), 2);
    }

    #[test]
    fn test_free_and_reuse() {
        let mut arena = Arena::new();
        let i1 = arena.alloc(10);
        let i2 = arena.alloc(20);

        assert!(arena.free(i1));
        assert_eq!(arena.len(), 1);

        let i3 = arena.alloc(30);
        assert_eq!(i3, i1); // Should reuse the freed index
        assert_eq!(arena.get(i3), Some(&30));
        assert_eq!(arena.get(i2), Some(&20));
        assert_eq!(arena.len(), 2);
    }

    #[test]
    fn invalid_and_double_free_are_safe() {
        let mut arena = Arena::new();
        let idx = arena.alloc(10);
        assert!(!arena.free(99));
        assert!(arena.free(idx));
        assert!(!arena.free(idx));
        assert_eq!(arena.len(), 0);
        assert_eq!(arena.get(idx), None);

        let reused = arena.alloc(20);
        assert_eq!(reused, idx);
        assert_eq!(arena.len(), 1);
        assert_eq!(arena.get(reused), Some(&20));
    }
}
