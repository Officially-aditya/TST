use crate::tst::TernarySearchTrie;
use crate::payload::{PayloadArena, Payload};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct LongTermMemory {
    pub trie: TernarySearchTrie,
    pub payloads: PayloadArena,
}

impl LongTermMemory {
    pub fn new() -> Self {
        Self {
            trie: TernarySearchTrie::new(),
            payloads: PayloadArena::new(),
        }
    }

    pub fn with_capacity(node_cap: usize, payload_cap: usize) -> Self {
        Self {
            trie: TernarySearchTrie::with_capacity(node_cap),
            payloads: PayloadArena::with_capacity(payload_cap),
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
        } else {
            // New insert
            let payload_idx = self.payloads.alloc(payload);
            self.trie.insert(key, payload_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::{PayloadHeader, PayloadData};
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
}
