use crate::arena::{Arena, DummyItem};
use crate::types::Timestamp;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadHeader {
    pub payload_type: u16,
    pub version: u16,
    pub created_ts: Timestamp,
    pub last_access_ts: Timestamp,
    pub access_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
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
    /// Metadata-rich v0.2 memory representation. `payload_type` records the
    /// planner's original semantic hint (`preference` or `token_stats`).
    MemoryRecord {
        payload_type: String,
        key: String,
        value: String,
        memory_type: String,
        source_text: String,
        created_at: u64,
        updated_at: u64,
        confidence: f32,
        tags: Vec<String>,
        source: String,
        layer: String,
        reinforcement_score: f32,
        deleted: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    pub header: PayloadHeader,
    pub data: PayloadData,
}

impl Payload {
    pub fn is_deleted(&self) -> bool {
        matches!(self.data, PayloadData::MemoryRecord { deleted: true, .. })
    }

    pub fn category(&self) -> &str {
        match &self.data {
            PayloadData::TokenStats { .. } => "token_stats",
            PayloadData::PhraseMeta { .. } => "phrase_meta",
            PayloadData::ConceptAnchor { .. } => "concept_anchor",
            PayloadData::StructurePattern { .. } => "structure_pattern",
            PayloadData::Preference { .. } => "preference",
            PayloadData::MemoryRecord { memory_type, .. } => memory_type,
        }
    }

    pub fn reinforcement_score(&self) -> f32 {
        match &self.data {
            PayloadData::MemoryRecord {
                reinforcement_score,
                ..
            } => *reinforcement_score,
            _ => 0.0,
        }
    }

    pub fn confidence(&self) -> f32 {
        match &self.data {
            PayloadData::MemoryRecord { confidence, .. } => *confidence,
            _ => 1.0,
        }
    }

    pub fn updated_timestamp(&self) -> u64 {
        match &self.data {
            PayloadData::MemoryRecord { updated_at, .. } => *updated_at,
            _ => self.header.last_access_ts.0.max(self.header.created_ts.0),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let valid = match &self.data {
            PayloadData::TokenStats { decay_score, .. } => decay_score.is_finite(),
            PayloadData::PhraseMeta { .. } => true,
            PayloadData::ConceptAnchor { strength, .. } => strength.is_finite(),
            PayloadData::StructurePattern { success_score, .. } => success_score.is_finite(),
            PayloadData::Preference { weight, .. } => weight.is_finite(),
            PayloadData::MemoryRecord {
                key,
                memory_type,
                source_text,
                created_at,
                updated_at,
                confidence,
                tags,
                source,
                layer,
                reinforcement_score,
                payload_type,
                ..
            } => {
                matches!(payload_type.as_str(), "preference" | "token_stats")
                    && !key.is_empty()
                    && !memory_type.is_empty()
                    && !source_text.is_empty()
                    && !source.is_empty()
                    && matches!(layer.as_str(), "stm" | "ltm")
                    && created_at <= updated_at
                    && tags.len() <= 1_000
                    && confidence.is_finite()
                    && (0.0..=1.0).contains(confidence)
                    && reinforcement_score.is_finite()
                    && *reinforcement_score >= 0.0
            }
        };
        if valid {
            Ok(())
        } else {
            Err("payload contains an invalid numeric value".to_string())
        }
    }

    pub fn validate_record_integrity(
        &self,
        outer_key: &str,
        outer_layer: &str,
    ) -> Result<(), String> {
        let PayloadData::MemoryRecord {
            payload_type,
            key,
            layer,
            ..
        } = &self.data
        else {
            return Ok(());
        };
        if !matches!(payload_type.as_str(), "preference" | "token_stats") {
            return Err(format!(
                "unknown memory-record payload type: {payload_type}"
            ));
        }
        if key != outer_key {
            return Err("payload data key must match the request key".to_string());
        }
        if layer != outer_layer {
            return Err("payload data layer must match the request layer".to_string());
        }
        Ok(())
    }

    pub fn preserve_update_metadata(&mut self, previous: &Payload, now: Timestamp) {
        self.header.created_ts = previous.header.created_ts;
        self.header.last_access_ts = now;
        self.header.access_count = previous.header.access_count;
        if let (
            PayloadData::MemoryRecord {
                created_at,
                reinforcement_score,
                ..
            },
            PayloadData::MemoryRecord {
                created_at: previous_created_at,
                reinforcement_score: previous_reinforcement,
                ..
            },
        ) = (&mut self.data, &previous.data)
        {
            *created_at = *previous_created_at;
            *reinforcement_score = reinforcement_score.max(*previous_reinforcement);
        }
    }

    pub fn sync_runtime_metadata(
        &mut self,
        created_ts: Timestamp,
        last_access_ts: Timestamp,
        access_count: u32,
        reinforcement_score: f32,
        layer: &str,
    ) {
        self.header.created_ts = created_ts;
        self.header.last_access_ts = last_access_ts;
        self.header.access_count = access_count;
        if let PayloadData::MemoryRecord {
            created_at,
            updated_at,
            reinforcement_score: record_reinforcement,
            layer: record_layer,
            ..
        } = &mut self.data
        {
            if *created_at == 0 {
                *created_at = created_ts.0;
            }
            *updated_at = last_access_ts.0;
            *record_reinforcement = reinforcement_score.max(0.0);
            *record_layer = layer.to_string();
        }
    }
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

impl Default for PayloadArena {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn free(&mut self, idx: u32) -> bool {
        self.arena.free_with_tombstone(idx)
    }

    pub fn len(&self) -> usize {
        self.arena.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    pub fn validate(&self) -> Result<(), String> {
        self.arena.validate()?;
        for idx in self.arena.active_indices() {
            self.arena
                .get(idx)
                .ok_or_else(|| format!("payload arena index {idx} is unavailable"))?
                .validate()
                .map_err(|error| format!("payload {idx}: {error}"))?;
        }
        Ok(())
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

        assert!(arena.free(idx));
        assert!(arena.get(idx).is_none());
        assert!(!arena.free(idx));
        assert!(!arena.free(u32::MAX));
        let idx2 = arena.alloc(Payload::dummy());
        assert_eq!(idx2, 0); // Should reuse index 0
    }
}
