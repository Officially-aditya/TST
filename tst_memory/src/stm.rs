use std::collections::HashMap;

use crate::types::Timestamp;

pub const STM_DEFAULT_CAPACITY: usize = 256;

/// Kept for the legacy stress harness. Runtime decay uses elapsed wall time.
pub const DECAY_BETA: f32 = 0.98;

pub type EntryFlags = u8;
pub const FLAG_PROMOTED: EntryFlags = 1 << 0;

#[derive(Debug, Clone)]
pub struct STMEntry {
    pub entry_id: u64,
    pub key: Box<[u8]>,
    pub payload_ref: u32,
    pub created_ts: Timestamp,
    pub last_access_ts: Timestamp,
    pub access_count: u32,
    pub reinforcement_score: f32,
    pub flags: EntryFlags,
}

/// A bounded ring buffer with a collision-safe hash index.
///
/// The index deliberately stores buckets rather than mapping a hash directly to
/// one slot. Every lookup verifies the complete key before returning an entry.
pub struct ShortTermMemory {
    pub buffer: Vec<Option<STMEntry>>,
    pub capacity: usize,
    pub head: usize,
    index: HashMap<u32, Vec<usize>>,
    decay_checkpoints: Vec<Timestamp>,
    pub promotion_threshold: f32,
    pub evictions: u64,
}

impl Default for ShortTermMemory {
    fn default() -> Self {
        Self::new(STM_DEFAULT_CAPACITY, 10.0)
    }
}

impl ShortTermMemory {
    pub fn new(capacity: usize, promotion_threshold: f32) -> Self {
        let capacity = capacity.max(1);
        Self {
            buffer: vec![None; capacity],
            capacity,
            head: 0,
            index: HashMap::with_capacity(capacity),
            decay_checkpoints: vec![Timestamp(0); capacity],
            promotion_threshold,
            evictions: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.iter().filter(|entry| entry.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn index_bucket_count(&self) -> usize {
        self.index.len()
    }

    pub fn hash_key(key: &[u8]) -> u32 {
        let mut hash: u32 = 0;
        for &byte in key {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }

    fn slot_for_key(&self, key: &[u8]) -> Option<usize> {
        let hash = Self::hash_key(key);
        self.index.get(&hash)?.iter().copied().find(|slot| {
            self.buffer[*slot]
                .as_ref()
                .is_some_and(|entry| entry.key.as_ref() == key)
        })
    }

    fn remove_index_slot(&mut self, hash: u32, slot: usize) {
        let remove_bucket = if let Some(bucket) = self.index.get_mut(&hash) {
            bucket.retain(|candidate| *candidate != slot);
            bucket.is_empty()
        } else {
            false
        };
        if remove_bucket {
            self.index.remove(&hash);
        }
    }

    /// Inserts or replaces an entry and returns the displaced entry, if any.
    /// Replacements do not advance the ring; new keys use the next ring slot.
    pub fn insert(&mut self, entry: STMEntry) -> Option<STMEntry> {
        if let Some(slot) = self.slot_for_key(&entry.key) {
            self.decay_checkpoints[slot] = entry.last_access_ts;
            return self.buffer[slot].replace(entry);
        }

        let slot = (0..self.capacity)
            .map(|offset| (self.head + offset) % self.capacity)
            .find(|slot| self.buffer[*slot].is_none())
            .unwrap_or(self.head);
        let displaced = self.buffer[slot].take();
        if let Some(old_entry) = &displaced {
            self.remove_index_slot(Self::hash_key(&old_entry.key), slot);
            self.evictions = self.evictions.saturating_add(1);
        }

        let hash = Self::hash_key(&entry.key);
        self.decay_checkpoints[slot] = entry.last_access_ts;
        self.buffer[slot] = Some(entry);
        self.index.entry(hash).or_default().push(slot);
        self.head = (self.head + 1) % self.capacity;
        displaced
    }

    pub fn lookup(&self, key: &[u8]) -> Option<&STMEntry> {
        let slot = self.slot_for_key(key)?;
        self.buffer[slot].as_ref()
    }

    pub fn lookup_mut(&mut self, key: &[u8]) -> Option<&mut STMEntry> {
        let slot = self.slot_for_key(key)?;
        self.buffer[slot].as_mut()
    }

    /// Compatibility-only hash lookup for old benchmarks. Production reads
    /// must call `lookup`/`lookup_mut` with the complete key.
    pub fn lookup_hash_mut(&mut self, key_hash: u32) -> Option<&mut STMEntry> {
        let slot = self.index.get(&key_hash)?.first().copied()?;
        self.buffer[slot].as_mut()
    }

    pub fn remove(&mut self, key: &[u8]) -> Option<STMEntry> {
        let slot = self.slot_for_key(key)?;
        let entry = self.buffer[slot].take()?;
        self.decay_checkpoints[slot] = Timestamp(0);
        self.remove_index_slot(Self::hash_key(&entry.key), slot);
        Some(entry)
    }

    fn decay_slot(&mut self, slot: usize, now: Timestamp, half_life_seconds: f64) {
        let Some(entry) = self.buffer[slot].as_mut() else {
            return;
        };
        let checkpoint = self.decay_checkpoints[slot];
        if now.0 <= checkpoint.0 {
            return;
        }

        let elapsed_seconds = (now.0 - checkpoint.0) as f64 / 1_000.0;
        let factor = if half_life_seconds <= 0.0 {
            0.0
        } else {
            (-std::f64::consts::LN_2 * elapsed_seconds / half_life_seconds).exp()
        };
        entry.reinforcement_score *= factor as f32;
        self.decay_checkpoints[slot] = now;
    }

    pub fn reinforce(
        &mut self,
        key: &[u8],
        now: Timestamp,
        increment: f32,
        half_life_seconds: f64,
    ) -> Option<f32> {
        self.apply_reinforcement(key, now, increment, half_life_seconds, true)
    }

    pub fn reinforce_write(
        &mut self,
        key: &[u8],
        now: Timestamp,
        increment: f32,
        half_life_seconds: f64,
    ) -> Option<f32> {
        self.apply_reinforcement(key, now, increment, half_life_seconds, false)
    }

    fn apply_reinforcement(
        &mut self,
        key: &[u8],
        now: Timestamp,
        increment: f32,
        half_life_seconds: f64,
        count_access: bool,
    ) -> Option<f32> {
        let slot = self.slot_for_key(key)?;
        self.decay_slot(slot, now, half_life_seconds);
        let entry = self.buffer[slot].as_mut()?;
        if count_access {
            entry.access_count = entry.access_count.saturating_add(1);
        }
        entry.last_access_ts = now;
        entry.reinforcement_score += increment.max(0.0);
        Some(entry.reinforcement_score)
    }

    pub fn decay_at(&mut self, now: Timestamp, half_life_seconds: f64) {
        for slot in 0..self.capacity {
            self.decay_slot(slot, now, half_life_seconds);
        }
    }

    pub fn remove_decayed(
        &mut self,
        now: Timestamp,
        half_life_seconds: f64,
        minimum_score: f32,
    ) -> Vec<STMEntry> {
        self.decay_at(now, half_life_seconds);
        let expired_keys: Vec<Box<[u8]>> = self
            .buffer
            .iter()
            .filter_map(|entry| {
                entry.as_ref().and_then(|entry| {
                    (entry.reinforcement_score < minimum_score).then(|| entry.key.clone())
                })
            })
            .collect();
        expired_keys
            .into_iter()
            .filter_map(|key| self.remove(&key))
            .collect()
    }

    pub fn promotion_candidates_at(
        &mut self,
        now: Timestamp,
        half_life_seconds: f64,
    ) -> Vec<STMEntry> {
        self.decay_at(now, half_life_seconds);
        self.buffer
            .iter()
            .filter_map(|entry| {
                entry.as_ref().and_then(|entry| {
                    (entry.flags & FLAG_PROMOTED == 0
                        && entry.reinforcement_score >= self.promotion_threshold)
                        .then(|| entry.clone())
                })
            })
            .collect()
    }

    /// Legacy cycle-based decay retained only for benchmark compatibility.
    pub fn decay_tick(&mut self) {
        for entry in self.buffer.iter_mut().flatten() {
            entry.reinforcement_score *= DECAY_BETA;
        }
    }

    pub fn check_promotion(&self) -> Vec<STMEntry> {
        self.buffer
            .iter()
            .filter_map(|entry| {
                entry.as_ref().and_then(|entry| {
                    (entry.flags & FLAG_PROMOTED == 0
                        && entry.reinforcement_score >= self.promotion_threshold)
                        .then(|| entry.clone())
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: u64, key: &[u8], score: f32, timestamp: u64) -> STMEntry {
        STMEntry {
            entry_id: id,
            key: key.into(),
            payload_ref: id as u32,
            created_ts: Timestamp(timestamp),
            last_access_ts: Timestamp(timestamp),
            access_count: 0,
            reinforcement_score: score,
            flags: 0,
        }
    }

    #[test]
    fn insert_lookup_and_promotion() {
        let mut stm = ShortTermMemory::new(4, 10.0);
        stm.insert(entry(1, b"alpha", 5.0, 0));
        let retrieved = stm.lookup_mut(b"alpha").unwrap();
        assert_eq!(retrieved.entry_id, 1);
        retrieved.reinforcement_score = 15.0;
        assert_eq!(stm.check_promotion()[0].entry_id, 1);
    }

    #[test]
    fn elapsed_decay_uses_half_life() {
        let mut stm = ShortTermMemory::new(4, 10.0);
        stm.insert(entry(1, b"alpha", 100.0, 1_000));
        stm.decay_at(Timestamp(11_000), 10.0);
        let score = stm.lookup(b"alpha").unwrap().reinforcement_score;
        assert!((score - 50.0).abs() < 0.001);

        // A second pass at the same timestamp must not decay twice.
        stm.decay_at(Timestamp(11_000), 10.0);
        assert!((stm.lookup(b"alpha").unwrap().reinforcement_score - 50.0).abs() < 0.001);
    }

    #[test]
    fn eviction_returns_displaced_entry_and_cleans_index() {
        let mut stm = ShortTermMemory::new(2, 10.0);
        stm.insert(entry(1, b"one", 1.0, 0));
        stm.insert(entry(2, b"two", 1.0, 0));
        let evicted = stm.insert(entry(3, b"three", 1.0, 0)).unwrap();
        assert_eq!(evicted.key.as_ref(), b"one");
        assert!(stm.lookup(b"one").is_none());
        assert!(stm.lookup(b"two").is_some());
        assert!(stm.lookup(b"three").is_some());
        assert_eq!(stm.evictions, 1);
    }

    #[test]
    fn full_key_verification_survives_hash_collision() {
        // "Aa" and "BB" collide under the 31-based compatibility hash.
        assert_eq!(
            ShortTermMemory::hash_key(b"Aa"),
            ShortTermMemory::hash_key(b"BB")
        );
        let mut stm = ShortTermMemory::new(4, 10.0);
        stm.insert(entry(1, b"Aa", 1.0, 0));
        stm.insert(entry(2, b"BB", 1.0, 0));
        assert_eq!(stm.lookup(b"Aa").unwrap().entry_id, 1);
        assert_eq!(stm.lookup(b"BB").unwrap().entry_id, 2);
    }

    #[test]
    fn insertion_reuses_a_removed_hole_before_evicting() {
        let mut stm = ShortTermMemory::new(3, 10.0);
        stm.insert(entry(1, b"one", 1.0, 0));
        stm.insert(entry(2, b"two", 1.0, 0));
        stm.insert(entry(3, b"three", 1.0, 0));
        assert!(stm.remove(b"two").is_some());

        assert!(stm.insert(entry(4, b"four", 1.0, 0)).is_none());
        assert_eq!(stm.evictions, 0);
        assert!(stm.lookup(b"one").is_some());
        assert!(stm.lookup(b"three").is_some());
        assert!(stm.lookup(b"four").is_some());
    }
}
