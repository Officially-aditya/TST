use crate::arena::{Arena, DummyItem};
use crate::types::Timestamp;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayloadHeader {
    pub payload_type: u16,
    pub version: u16,
    pub created_ts: Timestamp,
    pub last_access_ts: Timestamp,
    pub access_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PayloadData {
    TokenStats {
        canonical_form: String,
        frequency: u32,
        decay_score: f32,
        preferred_tokenizer_origin: Option<String>,
    },
    PhraseMeta {
        canonical_phrase: String,
        usage_count: u32,
        domain_mask: u16,
    },
    ConceptAnchor {
        concept_id: u32,
        related_tokens: Vec<String>,
        strength: f32,
    },
    StructurePattern {
        pattern_id: u32,
        steps: Vec<u8>,
        success_score: f32,
    },
    Preference {
        key: String,
        value: String,
        weight: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Payload {
    pub header: PayloadHeader,
    pub data: PayloadData,
}

impl DummyItem for Payload {
    fn dummy() -> Self {
        Self {
            header: PayloadHeader {
                payload_type: 0,
                version: 0,
                created_ts: Timestamp(0),
                last_access_ts: Timestamp(0),
                access_count: 0,
            },
            data: PayloadData::Preference {
                key: String::new(),
                value: String::new(),
                weight: 0.0,
            },
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct PayloadArena {
    pub arena: Arena<Payload>,
}

impl PayloadArena {
    pub fn new() -> Self {
        Self {
            arena: Arena::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            arena: Arena::with_capacity(capacity),
        }
    }

    pub fn alloc(&mut self, payload: Payload) -> u32 {
        self.arena.alloc(payload)
    }

    pub fn get(&self, idx: u32) -> Option<&Payload> {
        self.arena.get(idx)
    }

    pub fn get_mut(&mut self, idx: u32) -> Option<&mut Payload> {
        self.arena.get_mut(idx)
    }

    pub fn free(&mut self, idx: u32) {
        self.arena.free_with_tombstone(idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_arena() {
        let mut arena = PayloadArena::new();
        let payload = Payload {
            header: PayloadHeader {
                payload_type: 1,
                version: 1,
                created_ts: Timestamp(100),
                last_access_ts: Timestamp(100),
                access_count: 1,
            },
            data: PayloadData::TokenStats {
                canonical_form: "test".to_string(),
                frequency: 1,
                decay_score: 1.0,
                preferred_tokenizer_origin: None,
            },
        };

        let idx = arena.alloc(payload);
        assert_eq!(idx, 0);

        let retrieved = arena.get(idx).unwrap();
        assert_eq!(retrieved.header.access_count, 1);
        
        if let PayloadData::TokenStats { canonical_form, .. } = &retrieved.data {
            assert_eq!(canonical_form, "test");
        } else {
            panic!("Wrong payload data type");
        }

        arena.free(idx);
        let idx2 = arena.alloc(Payload::dummy());
        assert_eq!(idx2, 0); // Should reuse index 0
    }
}
