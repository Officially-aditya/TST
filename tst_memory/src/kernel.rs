use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::bias::{ModelConfig, compute_bias};
use crate::errors::{KernelError, KernelResult};
use crate::ltm::LongTermMemory;
use crate::payload::{Payload, PayloadArena};
use crate::stm::{STMEntry, ShortTermMemory};
use crate::tokenizer::TokenizerCache;
use crate::tree::{TreeEvent, TreeMemory};
use crate::types::Timestamp;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryLayer {
    STM,
    LTM,
    Tree,
}

impl MemoryLayer {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::STM => "stm",
            Self::LTM => "ltm",
            Self::Tree => "tree",
        }
    }
}

pub struct WriteProposal {
    pub layer: MemoryLayer,
    pub key: Vec<u8>,
    pub payload: Option<Payload>,
    pub tree_event: Option<TreeEvent>,
}

#[derive(Debug, Clone)]
pub struct STMConfig {
    pub capacity: usize,
    pub half_life_seconds: f64,
    pub promotion_threshold: f32,
    pub read_reinforcement: f32,
    pub write_reinforcement: f32,
    pub expiry_score: f32,
}

impl Default for STMConfig {
    fn default() -> Self {
        Self {
            capacity: 256,
            half_life_seconds: 1_800.0,
            promotion_threshold: 10.0,
            read_reinforcement: 1.0,
            write_reinforcement: 2.0,
            expiry_score: 0.01,
        }
    }
}

impl STMConfig {
    pub fn validate(&self) -> KernelResult<()> {
        if self.capacity == 0 {
            return Err(KernelError::invalid_params("STM capacity must be positive"));
        }
        if !self.half_life_seconds.is_finite() || self.half_life_seconds <= 0.0 {
            return Err(KernelError::invalid_params(
                "STM half-life must be a finite positive number",
            ));
        }
        for (name, value) in [
            ("promotion threshold", self.promotion_threshold),
            ("read reinforcement", self.read_reinforcement),
            ("write reinforcement", self.write_reinforcement),
            ("expiry score", self.expiry_score),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(KernelError::invalid_params(format!(
                    "STM {name} must be finite and non-negative"
                )));
            }
        }
        if self.promotion_threshold <= self.expiry_score {
            return Err(KernelError::invalid_params(
                "STM promotion threshold must exceed the expiry score",
            ));
        }
        if self.write_reinforcement < self.read_reinforcement {
            return Err(KernelError::invalid_params(
                "STM write reinforcement must be at least the read reinforcement",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct KernelMetrics {
    pub stm_evictions: u64,
    pub stm_expirations: u64,
    pub stm_promotions: u64,
    pub retrieval_hits: u64,
    pub retrieval_misses: u64,
    pub retrieval_searches: u64,
    pub retrieval_result_count: u64,
    pub protocol_errors: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemorySearchMatch {
    pub key: String,
    pub payload: Payload,
    pub score: f32,
    pub layer: &'static str,
}

pub struct Kernel {
    pub stm: ShortTermMemory,
    pub stm_payloads: PayloadArena,
    pub ltm: LongTermMemory,
    pub tree: TreeMemory,
    pub tokenizer: TokenizerCache,
    pub model_config: ModelConfig,
    pub stm_config: STMConfig,
    pub metrics: KernelMetrics,
    next_stm_entry_id: u64,
}

impl Default for Kernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel {
    pub fn new() -> Self {
        Self::with_stm_config(STMConfig::default())
    }

    pub fn with_stm_config(stm_config: STMConfig) -> Self {
        assert!(stm_config.validate().is_ok(), "invalid STM configuration");
        Self {
            stm: ShortTermMemory::new(stm_config.capacity, stm_config.promotion_threshold),
            stm_payloads: PayloadArena::new(),
            ltm: LongTermMemory::new(),
            tree: TreeMemory::new(),
            tokenizer: TokenizerCache::default(),
            model_config: ModelConfig::default(),
            stm_config,
            metrics: KernelMetrics::default(),
            next_stm_entry_id: 1,
        }
    }

    pub fn now() -> Timestamp {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(u64::MAX as u128) as u64;
        Timestamp(millis)
    }

    fn validate_key(key: &[u8]) -> KernelResult<()> {
        if key.is_empty() {
            return Err(KernelError::invalid_params("memory key must not be empty"));
        }
        if key.len() > 4_096 {
            return Err(KernelError::invalid_params(
                "memory key exceeds the 4096-byte limit",
            ));
        }
        for prefix in [b"system:".as_slice(), b"kernel:", b"tst:internal:"] {
            if key.starts_with(prefix) {
                return Err(KernelError::new(
                    "reserved_key",
                    "access denied for reserved memory-key prefix",
                ));
            }
        }
        Ok(())
    }

    fn maintain_stm(&mut self, now: Timestamp) {
        let expired = self.stm.remove_decayed(
            now,
            self.stm_config.half_life_seconds,
            self.stm_config.expiry_score,
        );
        for entry in expired {
            self.stm_payloads.free(entry.payload_ref);
            self.metrics.stm_expirations = self.metrics.stm_expirations.saturating_add(1);
        }
        self.metrics.stm_evictions = self.stm.evictions;
    }

    fn promote_ready(&mut self, now: Timestamp) -> usize {
        let candidates = self
            .stm
            .promotion_candidates_at(now, self.stm_config.half_life_seconds);
        let mut promoted = 0;
        for candidate in candidates {
            let Some(mut payload) = self.stm_payloads.get(candidate.payload_ref).cloned() else {
                continue;
            };
            payload.sync_runtime_metadata(
                candidate.created_ts,
                candidate.last_access_ts,
                candidate.access_count,
                candidate.reinforcement_score,
                "ltm",
            );
            // Canonical-key upsert makes retrying a promotion idempotent.
            self.ltm.write(&candidate.key, payload);
            if let Some(removed) = self.stm.remove(&candidate.key) {
                self.stm_payloads.free(removed.payload_ref);
                self.metrics.stm_promotions = self.metrics.stm_promotions.saturating_add(1);
                promoted += 1;
            }
        }
        promoted
    }

    pub fn maintenance(&mut self) -> usize {
        let now = Self::now();
        self.maintain_stm(now);
        self.promote_ready(now)
    }

    pub fn store_memory(
        &mut self,
        layer: MemoryLayer,
        key: &[u8],
        mut payload: Payload,
    ) -> KernelResult<bool> {
        Self::validate_key(key)?;
        payload.validate().map_err(KernelError::invalid_params)?;
        if payload.is_deleted() {
            return Err(KernelError::invalid_params(
                "deleted records must use memory.delete instead of memory.store",
            ));
        }
        payload
            .validate_record_integrity(std::str::from_utf8(key).unwrap_or(""), layer.as_str())
            .map_err(KernelError::invalid_params)?;
        let now = Self::now();
        if payload.header.created_ts.0 == 0 {
            payload.header.created_ts = now;
        }
        payload.header.last_access_ts = now;

        match layer {
            MemoryLayer::STM => {
                self.maintain_stm(now);
                let previous = self.stm.lookup(key).cloned();
                let previous_payload = previous
                    .as_ref()
                    .and_then(|entry| self.stm_payloads.get(entry.payload_ref))
                    .cloned();
                let (entry_id, created_ts, access_count, score) = if previous.is_some() {
                    let _ = self.stm.reinforce_write(
                        key,
                        now,
                        self.stm_config.write_reinforcement,
                        self.stm_config.half_life_seconds,
                    );
                    let existing = self.stm.lookup(key).expect("entry existed");
                    (
                        existing.entry_id,
                        existing.created_ts,
                        existing.access_count,
                        existing.reinforcement_score,
                    )
                } else {
                    let id = self.next_stm_entry_id;
                    self.next_stm_entry_id = self.next_stm_entry_id.saturating_add(1);
                    (
                        id,
                        payload.header.created_ts,
                        0,
                        self.stm_config.write_reinforcement,
                    )
                };
                if let Some(previous_payload) = &previous_payload {
                    payload.preserve_update_metadata(previous_payload, now);
                }
                payload.sync_runtime_metadata(created_ts, now, access_count, score, "stm");
                let payload_ref = self.stm_payloads.alloc(payload);
                let displaced = self.stm.insert(STMEntry {
                    entry_id,
                    key: key.into(),
                    payload_ref,
                    created_ts,
                    last_access_ts: now,
                    access_count,
                    reinforcement_score: score,
                    flags: 0,
                });
                if let Some(displaced) = displaced {
                    self.stm_payloads.free(displaced.payload_ref);
                }
                self.metrics.stm_evictions = self.stm.evictions;
                Ok(self.promote_ready(now) > 0)
            }
            MemoryLayer::LTM => {
                if let Some(previous) = self.ltm.read(key).cloned() {
                    let created_ts = previous.header.created_ts;
                    let access_count = previous.header.access_count;
                    let reinforcement =
                        previous.reinforcement_score() + self.stm_config.write_reinforcement;
                    payload.preserve_update_metadata(&previous, now);
                    payload.sync_runtime_metadata(
                        created_ts,
                        now,
                        access_count,
                        reinforcement,
                        "ltm",
                    );
                } else {
                    let reinforcement = payload.reinforcement_score();
                    payload.sync_runtime_metadata(
                        payload.header.created_ts,
                        now,
                        payload.header.access_count,
                        reinforcement,
                        "ltm",
                    );
                }
                self.ltm.write(key, payload);
                Ok(false)
            }
            MemoryLayer::Tree => Err(KernelError::invalid_params(
                "tree is not a valid layer for memory.store",
            )),
        }
    }

    pub fn update_memory(
        &mut self,
        layer: MemoryLayer,
        key: &[u8],
        payload: Payload,
    ) -> KernelResult<Option<bool>> {
        let exists = match layer {
            MemoryLayer::STM => self.stm.lookup(key).is_some(),
            MemoryLayer::LTM => self.ltm.read(key).is_some(),
            MemoryLayer::Tree => false,
        };
        if !exists {
            return Ok(None);
        }
        self.store_memory(layer, key, payload).map(Some)
    }

    pub fn read_memory(&mut self, layer: MemoryLayer, key: &[u8]) -> KernelResult<Option<Payload>> {
        Self::validate_key(key)?;
        let now = Self::now();
        let result = match layer {
            MemoryLayer::STM => {
                self.maintain_stm(now);
                let score = self.stm.reinforce(
                    key,
                    now,
                    self.stm_config.read_reinforcement,
                    self.stm_config.half_life_seconds,
                );
                let metadata = self.stm.lookup(key).map(|entry| {
                    (
                        entry.payload_ref,
                        entry.created_ts,
                        entry.last_access_ts,
                        entry.access_count,
                        entry.reinforcement_score,
                    )
                });
                let result = metadata.and_then(|metadata| {
                    let (payload_ref, created, accessed, count, reinforcement) = metadata;
                    self.stm_payloads.get_mut(payload_ref).map(|payload| {
                        payload.sync_runtime_metadata(
                            created,
                            accessed,
                            count,
                            reinforcement,
                            "stm",
                        );
                        payload.clone()
                    })
                });
                if score.is_some() {
                    self.promote_ready(now);
                }
                result
            }
            MemoryLayer::LTM => self.ltm.read_mut(key).map(|payload| {
                payload.header.access_count = payload.header.access_count.saturating_add(1);
                payload.header.last_access_ts = now;
                payload.clone()
            }),
            MemoryLayer::Tree => {
                return Err(KernelError::invalid_params(
                    "tree is not a valid layer for memory.get",
                ));
            }
        };

        if result.is_some() {
            self.metrics.retrieval_hits = self.metrics.retrieval_hits.saturating_add(1);
        } else {
            self.metrics.retrieval_misses = self.metrics.retrieval_misses.saturating_add(1);
        }
        Ok(result)
    }

    pub fn delete_memory(&mut self, layer: MemoryLayer, key: &[u8]) -> KernelResult<bool> {
        Self::validate_key(key)?;
        match layer {
            MemoryLayer::STM => {
                let Some(entry) = self.stm.remove(key) else {
                    return Ok(false);
                };
                self.stm_payloads.free(entry.payload_ref);
                Ok(true)
            }
            MemoryLayer::LTM => Ok(self.ltm.delete(key)),
            MemoryLayer::Tree => Err(KernelError::invalid_params(
                "tree is not a valid layer for memory.delete",
            )),
        }
    }

    /// Immutable compatibility read. New callers should use `read_memory` so
    /// reads reinforce STM entries and update access metadata.
    pub fn route_read(&self, key: &[u8]) -> Option<Payload> {
        if let Some(entry) = self.stm.lookup(key)
            && let Some(payload) = self.stm_payloads.get(entry.payload_ref)
        {
            return Some(payload.clone());
        }
        self.ltm.read(key).cloned()
    }

    pub fn search_memory(
        &mut self,
        layer: Option<MemoryLayer>,
        query: &str,
        prefix: Option<&str>,
        limit: usize,
    ) -> KernelResult<Vec<MemorySearchMatch>> {
        if matches!(layer, Some(MemoryLayer::Tree)) {
            return Err(KernelError::invalid_params(
                "tree is not a valid layer for memory.search",
            ));
        }
        let now = Self::now();
        self.maintain_stm(now);
        let mut matches = Vec::new();

        if layer.is_none() || layer == Some(MemoryLayer::STM) {
            for entry in self.stm.buffer.iter().flatten() {
                let Ok(key) = std::str::from_utf8(&entry.key) else {
                    continue;
                };
                if prefix.is_some_and(|prefix| !key.starts_with(prefix)) {
                    continue;
                }
                if let Some(payload) = self.stm_payloads.get(entry.payload_ref) {
                    if payload.is_deleted() {
                        continue;
                    }
                    let score = lexical_score(query, key, payload);
                    if score > 0.0 || query.is_empty() || prefix.is_some() {
                        matches.push(MemorySearchMatch {
                            key: key.to_string(),
                            payload: payload.clone(),
                            score: score
                                + normalized_reinforcement(entry.reinforcement_score) * 0.15,
                            layer: "stm",
                        });
                    }
                }
            }
        }

        if layer.is_none() || layer == Some(MemoryLayer::LTM) {
            for (key, payload) in self.ltm.iter() {
                if payload.is_deleted() {
                    continue;
                }
                if prefix.is_some_and(|prefix| !key.starts_with(prefix)) {
                    continue;
                }
                let score = lexical_score(query, key, payload);
                if score > 0.0 || query.is_empty() || prefix.is_some() {
                    matches.push(MemorySearchMatch {
                        key: key.to_string(),
                        payload: payload.clone(),
                        score,
                        layer: "ltm",
                    });
                }
            }
        }

        matches.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    right
                        .payload
                        .header
                        .updated_timestamp()
                        .cmp(&left.payload.header.updated_timestamp())
                })
                .then_with(|| left.key.cmp(&right.key))
        });
        matches.dedup_by(|left, right| left.key == right.key);
        matches.truncate(limit.clamp(1, 1_000));
        self.metrics.retrieval_searches = self.metrics.retrieval_searches.saturating_add(1);
        self.metrics.retrieval_result_count = self
            .metrics
            .retrieval_result_count
            .saturating_add(matches.len() as u64);
        if matches.is_empty() {
            self.metrics.retrieval_misses = self.metrics.retrieval_misses.saturating_add(1);
        } else {
            self.metrics.retrieval_hits = self.metrics.retrieval_hits.saturating_add(1);
        }
        Ok(matches)
    }

    pub fn validate_and_commit(&mut self, proposal: WriteProposal) -> Result<(), &'static str> {
        if let Some(event) = proposal.tree_event {
            if proposal.layer != MemoryLayer::Tree {
                return Err("tree events require the tree layer");
            }
            self.tree.process_event(event);
            return Ok(());
        }
        let Some(payload) = proposal.payload else {
            return Err("missing payload");
        };
        self.store_memory(proposal.layer, &proposal.key, payload)
            .map(|_| ())
            .map_err(|error| match error.code.as_str() {
                "reserved_key" => "Access denied: reserved prefix",
                _ => "Kernel validation failed",
            })
    }

    pub fn get_logit_bias(&self, frequency: u32, decay_score: f32) -> f32 {
        compute_bias(frequency, decay_score, &self.model_config)
    }

    pub fn resolve_tokens(&mut self, model: &str, string: &str) -> Vec<u32> {
        self.tokenizer.resolve_tokens(model, string)
    }

    pub fn hash_key(key: &[u8]) -> u32 {
        ShortTermMemory::hash_key(key)
    }
}

trait PayloadHeaderTimestamp {
    fn updated_timestamp(&self) -> u64;
}

impl PayloadHeaderTimestamp for crate::payload::PayloadHeader {
    fn updated_timestamp(&self) -> u64 {
        self.last_access_ts.0.max(self.created_ts.0)
    }
}

fn normalized_reinforcement(score: f32) -> f32 {
    score.max(0.0) / (score.max(0.0) + 10.0)
}

fn tokens(text: &str) -> HashSet<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn lexical_score(query: &str, key: &str, payload: &Payload) -> f32 {
    let query_tokens = tokens(query);
    if query_tokens.is_empty() {
        return 0.0;
    }
    let searchable = format!(
        "{} {}",
        key.replace([':', '_'], " "),
        serde_json::to_string(payload).unwrap_or_default()
    );
    let candidate_tokens = tokens(&searchable);
    let overlap = query_tokens.intersection(&candidate_tokens).count() as f32;
    if overlap == 0.0 {
        return 0.0;
    }
    let union = query_tokens.union(&candidate_tokens).count().max(1) as f32;
    let lexical = overlap / union;
    let prefix = if query_tokens
        .iter()
        .any(|token| key.to_lowercase().contains(token))
    {
        1.0
    } else {
        0.0
    };
    lexical * 0.65 + prefix * 0.20 + 0.10
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::{PayloadData, PayloadHeader};

    fn preference(value: &str) -> Payload {
        Payload {
            header: PayloadHeader {
                payload_type: 1,
                version: 1,
                created_ts: Timestamp(0),
                last_access_ts: Timestamp(0),
                access_count: 0,
            },
            data: PayloadData::Preference {
                key: "language".to_string(),
                value: value.to_string(),
                weight: 1.0,
            },
        }
    }

    fn memory_record(key: &str, value: &str, layer: &str, created_at: u64) -> Payload {
        Payload {
            header: PayloadHeader {
                payload_type: 5,
                version: 1,
                created_ts: Timestamp(created_at),
                last_access_ts: Timestamp(created_at),
                access_count: 0,
            },
            data: PayloadData::MemoryRecord {
                payload_type: "preference".to_string(),
                key: key.to_string(),
                value: value.to_string(),
                memory_type: "preference".to_string(),
                source_text: format!("I prefer {value}"),
                created_at,
                updated_at: created_at,
                confidence: 1.0,
                tags: vec!["language".to_string()],
                source: "user".to_string(),
                layer: layer.to_string(),
                reinforcement_score: 0.0,
                deleted: false,
            },
        }
    }

    #[test]
    fn ltm_routing_and_reserved_keys() {
        let mut kernel = Kernel::new();
        kernel
            .store_memory(
                MemoryLayer::LTM,
                b"user:preference:language",
                preference("TypeScript"),
            )
            .unwrap();
        assert!(kernel.route_read(b"user:preference:language").is_some());
        assert!(
            kernel
                .store_memory(MemoryLayer::LTM, b"system:config", preference("bad"))
                .is_err()
        );
    }

    #[test]
    fn stm_reads_reinforce_and_promote_idempotently() {
        let config = STMConfig {
            promotion_threshold: 4.0,
            read_reinforcement: 1.0,
            write_reinforcement: 2.0,
            ..STMConfig::default()
        };
        let mut kernel = Kernel::with_stm_config(config);
        kernel
            .store_memory(MemoryLayer::STM, b"session:atlas", preference("Atlas"))
            .unwrap();
        for _ in 0..10 {
            if kernel.stm.lookup(b"session:atlas").is_none() {
                break;
            }
            assert!(
                kernel
                    .read_memory(MemoryLayer::STM, b"session:atlas")
                    .unwrap()
                    .is_some()
            );
        }
        assert!(kernel.stm.lookup(b"session:atlas").is_none());
        assert!(kernel.ltm.read(b"session:atlas").is_some());
        assert_eq!(kernel.metrics.stm_promotions, 1);
        assert!(
            kernel
                .read_memory(MemoryLayer::LTM, b"session:atlas")
                .unwrap()
                .is_some()
        );
        assert_eq!(kernel.metrics.stm_promotions, 1);
    }

    #[test]
    fn eviction_frees_transient_payload() {
        let config = STMConfig {
            capacity: 1,
            promotion_threshold: 100.0,
            ..STMConfig::default()
        };
        let mut kernel = Kernel::with_stm_config(config);
        kernel
            .store_memory(MemoryLayer::STM, b"one", preference("one"))
            .unwrap();
        assert_eq!(kernel.stm_payloads.arena.len(), 1);
        kernel
            .store_memory(MemoryLayer::STM, b"two", preference("two"))
            .unwrap();
        assert_eq!(kernel.stm_payloads.arena.len(), 1);
        assert!(kernel.route_read(b"one").is_none());
        assert!(kernel.route_read(b"two").is_some());
    }

    #[test]
    fn lexical_search_finds_paraphrased_preference() {
        let mut kernel = Kernel::new();
        kernel
            .store_memory(
                MemoryLayer::LTM,
                b"user:default:preference:programming_language",
                preference("TypeScript"),
            )
            .unwrap();
        let matches = kernel
            .search_memory(None, "Which language do I usually use?", None, 3)
            .unwrap();
        assert_eq!(
            matches[0].key,
            "user:default:preference:programming_language"
        );
    }

    #[test]
    fn structured_stm_metadata_survives_reads_updates_and_promotion() {
        let key = "session:test:context:language";
        let config = STMConfig {
            promotion_threshold: 100.0,
            ..STMConfig::default()
        };
        let mut kernel = Kernel::with_stm_config(config);
        kernel
            .store_memory(
                MemoryLayer::STM,
                key.as_bytes(),
                memory_record(key, "Rust", "stm", 100),
            )
            .unwrap();
        let initial_created = kernel.route_read(key.as_bytes()).unwrap().header.created_ts;
        let read = kernel
            .read_memory(MemoryLayer::STM, key.as_bytes())
            .unwrap()
            .unwrap();
        assert_eq!(read.header.access_count, 1);
        assert!((read.reinforcement_score() - 3.0).abs() < 0.001);

        kernel
            .update_memory(
                MemoryLayer::STM,
                key.as_bytes(),
                memory_record(key, "TypeScript", "stm", 999),
            )
            .unwrap();
        let updated = kernel.route_read(key.as_bytes()).unwrap();
        assert_eq!(updated.header.created_ts, initial_created);
        assert_eq!(updated.header.access_count, 1);
        assert!((updated.reinforcement_score() - 5.0).abs() < 0.001);

        kernel.stm.promotion_threshold = 4.9;
        assert_eq!(kernel.maintenance(), 1);
        let promoted = kernel.ltm.read(key.as_bytes()).unwrap();
        assert_eq!(promoted.header.created_ts, initial_created);
        assert_eq!(promoted.header.access_count, 1);
        assert!((promoted.reinforcement_score() - 5.0).abs() < 0.001);
        match &promoted.data {
            PayloadData::MemoryRecord {
                created_at,
                layer,
                value,
                ..
            } => {
                assert_eq!(*created_at, 100);
                assert_eq!(layer, "ltm");
                assert_eq!(value, "TypeScript");
            }
            _ => panic!("expected structured memory record"),
        }
    }

    #[test]
    fn ltm_update_preserves_creation_and_adds_write_reinforcement() {
        let key = "user:default:preference:language";
        let mut kernel = Kernel::new();
        kernel
            .store_memory(
                MemoryLayer::LTM,
                key.as_bytes(),
                memory_record(key, "Rust", "ltm", 123),
            )
            .unwrap();
        kernel
            .update_memory(
                MemoryLayer::LTM,
                key.as_bytes(),
                memory_record(key, "TypeScript", "ltm", 999),
            )
            .unwrap();
        let updated = kernel.ltm.read(key.as_bytes()).unwrap();
        assert_eq!(updated.header.created_ts, Timestamp(123));
        assert_eq!(updated.reinforcement_score(), 2.0);
        match &updated.data {
            PayloadData::MemoryRecord {
                created_at, value, ..
            } => {
                assert_eq!(*created_at, 123);
                assert_eq!(value, "TypeScript");
            }
            _ => panic!("expected structured memory record"),
        }
    }

    #[test]
    fn stm_configuration_rejects_non_operational_values() {
        assert!(
            STMConfig {
                capacity: 0,
                ..STMConfig::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            STMConfig {
                half_life_seconds: f64::NAN,
                ..STMConfig::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            STMConfig {
                read_reinforcement: 3.0,
                write_reinforcement: 2.0,
                ..STMConfig::default()
            }
            .validate()
            .is_err()
        );
    }
}
