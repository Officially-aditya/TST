use crate::payload::{Payload, PayloadArena};
use crate::tst::TernarySearchTrie;
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct LongTermMemory {
    pub trie: TernarySearchTrie,
    pub payloads: PayloadArena,
    /// Searchable UTF-8 keys. The TST remains the exact-lookup index.
    #[serde(default)]
    pub keys: BTreeMap<String, u32>,
}

impl Default for LongTermMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl LongTermMemory {
    pub fn new() -> Self {
        Self {
            trie: TernarySearchTrie::new(),
            payloads: PayloadArena::new(),
            keys: BTreeMap::new(),
        }
    }

    pub fn with_capacity(node_cap: usize, payload_cap: usize) -> Self {
        Self {
            trie: TernarySearchTrie::with_capacity(node_cap),
            payloads: PayloadArena::with_capacity(payload_cap),
            keys: BTreeMap::new(),
        }
    }

    pub fn read(&self, key: &[u8]) -> Option<&Payload> {
        let payload_idx = self.trie.lookup(key)?;
        self.payloads.get(payload_idx)
    }

    pub fn read_mut(&mut self, key: &[u8]) -> Option<&mut Payload> {
        let payload_idx = self.trie.lookup(key)?;
        self.payloads.get_mut(payload_idx)
    }

    pub fn write(&mut self, key: &[u8], payload: Payload) {
        if let Some(existing_idx) = self.trie.lookup(key) {
            // Update existing or overwrite
            if let Some(existing) = self.payloads.get_mut(existing_idx) {
                *existing = payload;
            }
            if let Ok(key) = std::str::from_utf8(key) {
                self.keys.insert(key.to_string(), existing_idx);
            }
        } else {
            // New insert
            let payload_idx = self.payloads.alloc(payload);
            self.trie.insert(key, payload_idx);
            if let Ok(key) = std::str::from_utf8(key) {
                self.keys.insert(key.to_string(), payload_idx);
            }
        }
    }

    pub fn delete(&mut self, key: &[u8]) -> bool {
        let Some(payload_idx) = self.trie.lookup(key) else {
            return false;
        };
        if !self.trie.delete(key) {
            return false;
        }
        self.payloads.free(payload_idx);
        if let Ok(key) = std::str::from_utf8(key) {
            self.keys.remove(key);
        }
        true
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Payload)> {
        self.keys.iter().filter_map(|(key, payload_idx)| {
            self.payloads
                .get(*payload_idx)
                .map(|payload| (key.as_str(), payload))
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        self.trie.validate()?;
        self.payloads.validate()?;
        if self.keys.len() != self.payloads.len() {
            return Err("LTM key and payload counts differ".to_string());
        }
        let mut payload_indices = HashSet::new();
        for (key, expected_idx) in &self.keys {
            let actual_idx = self
                .trie
                .lookup(key.as_bytes())
                .ok_or_else(|| format!("LTM key {key} is missing from the trie"))?;
            if actual_idx != *expected_idx {
                return Err(format!(
                    "LTM key {key} points to inconsistent payload indices"
                ));
            }
            if !payload_indices.insert(actual_idx) {
                return Err("multiple LTM keys share one payload index".to_string());
            }
            self.payloads
                .get(actual_idx)
                .ok_or_else(|| format!("LTM key {key} references a missing payload"))?;
        }
        let leaf_indices: HashSet<u32> = self.trie.leaf_payload_indices()?.into_iter().collect();
        if leaf_indices != payload_indices {
            return Err("LTM key map does not match trie leaves".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::{PayloadData, PayloadHeader};
    use crate::types::Timestamp;

    #[test]
    fn test_ltm_read_write() {
        let mut ltm = LongTermMemory::new();
        let payload = Payload {
            header: PayloadHeader {
                payload_type: 1,
                version: 1,
                created_ts: Timestamp(0),
                last_access_ts: Timestamp(0),
                access_count: 0,
            },
            data: PayloadData::Preference {
                key: "test".to_string(),
                value: "1".to_string(),
                weight: 1.0,
            },
        };

        ltm.write(b"my_key", payload);
        let retrieved = ltm.read(b"my_key").unwrap();

        if let PayloadData::Preference { key, .. } = &retrieved.data {
            assert_eq!(key, "test");
        } else {
            panic!("Wrong data type");
        }
    }

    #[test]
    fn test_ltm_delete_releases_payload() {
        let mut ltm = LongTermMemory::new();
        let payload = Payload {
            header: PayloadHeader {
                payload_type: 1,
                version: 1,
                created_ts: Timestamp(0),
                last_access_ts: Timestamp(0),
                access_count: 0,
            },
            data: PayloadData::Preference {
                key: "editor".to_string(),
                value: "vim".to_string(),
                weight: 1.0,
            },
        };
        ltm.write(b"user:editor", payload);
        assert_eq!(ltm.len(), 1);
        assert!(ltm.delete(b"user:editor"));
        assert!(ltm.read(b"user:editor").is_none());
        assert_eq!(ltm.len(), 0);
        assert!(!ltm.delete(b"user:editor"));
    }

    #[test]
    fn delete_reinsert_churn_reuses_payload_slots_without_tombstones() {
        let mut ltm = LongTermMemory::new();
        for index in 0..100 {
            let payload = Payload {
                header: PayloadHeader {
                    payload_type: 1,
                    version: 1,
                    created_ts: Timestamp(index),
                    last_access_ts: Timestamp(index),
                    access_count: 0,
                },
                data: PayloadData::Preference {
                    key: "editor".to_string(),
                    value: format!("value-{index}"),
                    weight: 1.0,
                },
            };
            ltm.write(b"user:editor", payload);
            assert!(ltm.delete(b"user:editor"));
        }
        assert_eq!(ltm.payloads.arena.slot_count(), 1);
        assert!(ltm.is_empty());
        ltm.validate().unwrap();
    }
}
