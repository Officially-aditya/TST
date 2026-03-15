#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Arena<T> {
    items: Vec<T>,
    free_list: Vec<u32>,
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            free_list: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            free_list: Vec::new(),
        }
    }

    /// Allocates an item and returns its index.
    pub fn alloc(&mut self, value: T) -> u32 {
        if let Some(idx) = self.free_list.pop() {
            self.items[idx as usize] = value;
            idx
        } else {
            let idx = self.items.len() as u32;
            self.items.push(value);
            idx
        }
    }

    /// Borrows the item at index.
    pub fn get(&self, idx: u32) -> Option<&T> {
        self.items.get(idx as usize)
    }

    /// Mutably borrows the item at index.
    pub fn get_mut(&mut self, idx: u32) -> Option<&mut T> {
        self.items.get_mut(idx as usize)
    }

    /// Marks the given index as free for future allocations.
    /// Note: `Arena` does not automatically clear the value.
    pub fn free(&mut self, idx: u32) {
        self.free_list.push(idx);
    }

    pub fn len(&self) -> usize {
        self.items.len() - self.free_list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait DummyItem {
    fn dummy() -> Self;
}

impl<T: DummyItem> Arena<T> {
    /// Frees an index and overwrites it with a dummy (tombstone) value.
    pub fn free_with_tombstone(&mut self, idx: u32) {
        if let Some(item) = self.get_mut(idx) {
            *item = T::dummy();
            self.free_list.push(idx);
        }
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
        
        arena.free(i1);
        assert_eq!(arena.len(), 1);
        
        let i3 = arena.alloc(30);
        assert_eq!(i3, i1); // Should reuse the freed index
        assert_eq!(arena.get(i3), Some(&30));
        assert_eq!(arena.get(i2), Some(&20));
        assert_eq!(arena.len(), 2);
    }
}
