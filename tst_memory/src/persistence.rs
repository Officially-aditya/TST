use crate::kernel::Kernel;
use crate::ltm::LongTermMemory;
use serde::{Deserialize, Serialize};
use std::fs;

/// The data written to disk. Only LTM persists across restarts.
/// STM is explicitly transient; Tree is rebuilt from source at startup.
#[derive(Serialize, Deserialize)]
struct SnapshotData {
    version: u32,
    ltm: LongTermMemory,
}

pub struct PersistenceHandler {
    snapshot_path: String,
}

impl PersistenceHandler {
    pub fn new(path: &str) -> Self {
        Self {
            snapshot_path: path.to_string(),
        }
    }

    /// Serialise LTM to disk atomically (write to .tmp then rename).
    pub fn save_snapshot(&self, kernel: &Kernel) -> Result<(), String> {
        let data = SnapshotData {
            version: 1,
            ltm: kernel.ltm.clone(),
        };

        let json = serde_json::to_string(&data)
            .map_err(|e| format!("Serialization failed: {}", e))?;

        let temp_path = format!("{}.tmp", self.snapshot_path);
        fs::write(&temp_path, &json)
            .map_err(|e| format!("Write failed: {}", e))?;
        fs::rename(&temp_path, &self.snapshot_path)
            .map_err(|e| format!("Atomic rename failed: {}", e))?;

        Ok(())
    }

    /// Deserialise LTM from disk and restore it into the kernel.
    pub fn load_snapshot(&self, kernel: &mut Kernel) -> Result<(), String> {
        let json = fs::read_to_string(&self.snapshot_path)
            .map_err(|e| format!("Read failed: {}", e))?;

        let data: SnapshotData = serde_json::from_str(&json)
            .map_err(|e| format!("Deserialization failed: {}", e))?;

        if data.version != 1 {
            return Err(format!("Unsupported snapshot version: {}", data.version));
        }

        kernel.ltm = data.ltm;
        Ok(())
    }

    /// Returns the byte size of the snapshot file on disk, or 0 if absent.
    pub fn snapshot_size_bytes(&self) -> u64 {
        fs::metadata(&self.snapshot_path)
            .map(|m| m.len())
            .unwrap_or(0)
    }

    pub fn compact(&self, _kernel: &mut Kernel, _threshold: f32) {
        // Compaction: traverse leaf list, evict entries below threshold, rebuild trie.
        // Full implementation deferred — requires decay_score access from PayloadData variants.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{MemoryLayer, WriteProposal};
    use crate::payload::{Payload, PayloadData, PayloadHeader};
    use crate::types::Timestamp;

    fn make_pref(key: &str, value: &str) -> Payload {
        Payload {
            header: PayloadHeader {
                payload_type: 2,
                version: 1,
                created_ts: Timestamp(0),
                last_access_ts: Timestamp(0),
                access_count: 1,
            },
            data: PayloadData::Preference {
                key: key.to_string(),
                value: value.to_string(),
                weight: 1.0,
            },
        }
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let path = "/tmp/tst_snapshot_roundtrip_test.json";
        let handler = PersistenceHandler::new(path);

        // Session 1: write 10 preferences
        let mut kernel1 = Kernel::new();
        for i in 0..10 {
            let key = format!("pref_{:03}", i);
            let val = format!("value_{}", i);
            let _ = kernel1.validate_and_commit(WriteProposal {
                layer: MemoryLayer::LTM,
                key: key.into_bytes(),
                payload: Some(make_pref(&format!("pref_{:03}", i), &val)),
                tree_event: None,
            });
        }
        handler.save_snapshot(&kernel1).expect("save failed");

        // Session 2: fresh kernel, load snapshot
        let mut kernel2 = Kernel::new();
        handler.load_snapshot(&mut kernel2).expect("load failed");

        // Verify all 10 entries present with correct values
        for i in 0..10 {
            let key = format!("pref_{:03}", i);
            let payload = kernel2.route_read(key.as_bytes()).expect(&format!("missing: {}", key));
            match payload.data {
                PayloadData::Preference { value, .. } => {
                    assert_eq!(value, format!("value_{}", i));
                }
                _ => panic!("Wrong payload type for {}", key),
            }
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_atomic_snapshot() {
        let kernel = Kernel::new();
        let path = "/tmp/tst_snapshot_test.json";
        let handler = PersistenceHandler::new(path);
        assert!(handler.save_snapshot(&kernel).is_ok());
        let _ = std::fs::remove_file(path);
    }
}
