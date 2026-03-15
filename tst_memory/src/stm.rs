use std::collections::HashMap;
use crate::types::Timestamp;

pub const STM_DEFAULT_CAPACITY: usize = 256;
pub const DECAY_BETA: f32 = 0.98;

#[derive(Debug, Clone)]
pub struct STMEntry {
    pub entry_id: u32,
    pub created_ts: Timestamp,
    pub last_access_ts: Timestamp,
    pub key_ref: u32, // Used as key_hash
    pub payload_ref: u32,
    pub reinforcement_score: f32,
    pub flags: u8,
}

pub struct ShortTermMemory {
    pub buffer: Vec<Option<STMEntry>>,
    pub capacity: usize,
    pub head: usize,
    pub index: HashMap<u32, usize>,
    pub promotion_threshold: f32,
}

impl ShortTermMemory {
    pub fn new(capacity: usize, promotion_threshold: f32) -> Self {
        Self {
            buffer: vec![None; capacity],
            capacity,
            head: 0,
            index: HashMap::with_capacity(capacity),
            promotion_threshold,
        }
    }

    pub fn default() -> Self {
        Self::new(STM_DEFAULT_CAPACITY, 10.0)
    }

    pub fn insert(&mut self, entry: STMEntry) {
        let key_hash = entry.key_ref;

        if let Some(old_entry) = &self.buffer[self.head] {
            self.index.remove(&old_entry.key_ref);
        }

        self.index.insert(key_hash, self.head);
        self.buffer[self.head] = Some(entry);
        self.head = (self.head + 1) % self.capacity;
    }

    pub fn lookup_mut(&mut self, key_hash: u32) -> Option<&mut STMEntry> {
        if let Some(&slot) = self.index.get(&key_hash) {
            self.buffer[slot].as_mut()
        } else {
            None
        }
    }

    pub fn decay_tick(&mut self) {
        for entry_opt in self.buffer.iter_mut() {
            if let Some(entry) = entry_opt {
                entry.reinforcement_score *= DECAY_BETA;
            }
        }
    }

    pub fn check_promotion(&self) -> Vec<STMEntry> {
        let mut promoted = Vec::new();
        for entry_opt in self.buffer.iter() {
            if let Some(entry) = entry_opt {
                if entry.reinforcement_score >= self.promotion_threshold {
                    promoted.push(entry.clone());
                }
            }
        }
        promoted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stm_insert_lookup() {
        let mut stm = ShortTermMemory::new(4, 10.0);
        let entry = STMEntry {
            entry_id: 1,
            created_ts: Timestamp(0),
            last_access_ts: Timestamp(0),
            key_ref: 12345,
            payload_ref: 1,
            reinforcement_score: 5.0,
            flags: 0,
        };
        stm.insert(entry);

        let retrieved = stm.lookup_mut(12345).unwrap();
        assert_eq!(retrieved.entry_id, 1);
        
        // Update score
        retrieved.reinforcement_score = 15.0;

        let promoted = stm.check_promotion();
        assert_eq!(promoted.len(), 1);
        assert_eq!(promoted[0].entry_id, 1);
    }

    #[test]
    fn test_stm_decay() {
        let mut stm = ShortTermMemory::new(4, 10.0);
        let entry = STMEntry {
            entry_id: 1,
            created_ts: Timestamp(0),
            last_access_ts: Timestamp(0),
            key_ref: 12345,
            payload_ref: 1,
            reinforcement_score: 100.0,
            flags: 0,
        };
        stm.insert(entry);
        stm.decay_tick();

        let retrieved = stm.lookup_mut(12345).unwrap();
        assert!((retrieved.reinforcement_score - 98.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stm_eviction() {
        let mut stm = ShortTermMemory::new(2, 10.0);
        
        // Insert 3 items into a 2-slot buffer
        for i in 0..3 {
            let entry = STMEntry {
                entry_id: i,
                created_ts: Timestamp(i as u64),
                last_access_ts: Timestamp(i as u64),
                key_ref: i + 100, // hash
                payload_ref: i * 2,
                reinforcement_score: 1.0,
                flags: 0,
            };
            stm.insert(entry);
        }

        // The first item (key_ref 100) should have been evicted and overwritten
        assert!(stm.lookup_mut(100).is_none());
        assert!(stm.lookup_mut(101).is_some());
        assert!(stm.lookup_mut(102).is_some());
    }
}
