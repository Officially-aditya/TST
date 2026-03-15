use crate::stm::{ShortTermMemory, STMEntry};
use crate::ltm::LongTermMemory;
use crate::tree::{TreeMemory, TreeEvent};
use crate::payload::{Payload, PayloadData};
use crate::tokenizer::TokenizerCache;
use crate::bias::{ModelConfig, compute_bias};
use crate::types::Timestamp;

pub enum MemoryLayer {
    STM,
    LTM,
    Tree,
}

pub struct WriteProposal {
    pub layer: MemoryLayer,
    pub key: Vec<u8>,
    pub payload: Option<Payload>,
    pub tree_event: Option<TreeEvent>,
}

pub struct Kernel {
    pub stm: ShortTermMemory,
    pub ltm: LongTermMemory,
    pub tree: TreeMemory,
    pub tokenizer: TokenizerCache,
    pub model_config: ModelConfig,
}

impl Kernel {
    pub fn new() -> Self {
        Self {
            stm: ShortTermMemory::default(),
            ltm: LongTermMemory::new(),
            tree: TreeMemory::new(),
            tokenizer: TokenizerCache::default(),
            model_config: ModelConfig::default(),
        }
    }

    pub fn route_read(&self, key: &[u8]) -> Option<Payload> {
        // STM > LTM Precedence (Conflict Resolution)
        // For strict correctness, we'd lookup `key` in STM by hash and then retrieve actual key.
        // For MVP, if it's in LTM, return it.
        
        if let Some(payload) = self.ltm.read(key) {
            return Some(payload.clone());
        }
        None
    }

    pub fn validate_and_commit(&mut self, proposal: WriteProposal) -> Result<(), &'static str> {
        // Security: validate prefix ACLs
        if proposal.key.starts_with(b"system:") {
            return Err("Access denied: reserved prefix");
        }

        match proposal.layer {
            MemoryLayer::STM => {
                if let Some(payload) = proposal.payload {
                    let p_idx = self.ltm.payloads.alloc(payload);
                    let entry = STMEntry {
                        entry_id: 0,
                        created_ts: Timestamp(0),
                        last_access_ts: Timestamp(0),
                        key_ref: Self::hash_key(&proposal.key),
                        payload_ref: p_idx,
                        reinforcement_score: 1.0,
                        flags: 0,
                    };
                    self.stm.insert(entry);
                }
            }
            MemoryLayer::LTM => {
                if let Some(payload) = proposal.payload {
                    self.ltm.write(&proposal.key, payload);
                }
            }
            MemoryLayer::Tree => {
                if let Some(event) = proposal.tree_event {
                    self.tree.process_event(event);
                }
            }
        }
        Ok(())
    }

    pub fn get_logit_bias(&self, frequency: u32, decay_score: f32) -> f32 {
        compute_bias(frequency, decay_score, &self.model_config)
    }

    pub fn resolve_tokens(&mut self, model: &str, string: &str) -> Vec<u32> {
        self.tokenizer.resolve_tokens(model, string)
    }

    pub fn hash_key(key: &[u8]) -> u32 {
        let mut hash: u32 = 0;
        for &b in key {
            hash = hash.wrapping_mul(31).wrapping_add(b as u32);
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::PayloadHeader;

    #[test]
    fn test_kernel_routing() {
        let mut kernel = Kernel::new();
        
        let payload = Payload {
            header: PayloadHeader {
                payload_type: 1,
                version: 1,
                created_ts: Timestamp(0),
                last_access_ts: Timestamp(0),
                access_count: 0,
            },
            data: PayloadData::Preference {
                key: "key".to_string(),
                value: "val".to_string(),
                weight: 1.0,
            },
        };

        let prop = WriteProposal {
            layer: MemoryLayer::LTM,
            key: b"user:pref".to_vec(),
            payload: Some(payload),
            tree_event: None,
        };

        assert!(kernel.validate_and_commit(prop).is_ok());

        let read = kernel.route_read(b"user:pref").unwrap();
        match read.data {
            PayloadData::Preference { value, .. } => assert_eq!(value, "val"),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_security_acl() {
        let mut kernel = Kernel::new();
        let prop = WriteProposal {
            layer: MemoryLayer::LTM,
            key: b"system:config".to_vec(),
            payload: None,
            tree_event: None,
        };
        
        assert!(kernel.validate_and_commit(prop).is_err());
    }
}
