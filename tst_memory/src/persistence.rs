use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::kernel::Kernel;
use crate::ltm::LongTermMemory;

const SNAPSHOT_VERSION: u32 = 1;
const MAX_SNAPSHOT_BYTES: u64 = 256 * 1024 * 1024;

/// Only LTM persists. STM is session-scoped and Tree is rebuilt from source.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapshotData {
    version: u32,
    checksum: String,
    ltm: LongTermMemory,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecoverySource {
    Empty,
    Primary,
    Previous,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecoveryReport {
    pub source: RecoverySource,
    pub warning: Option<String>,
    pub preserved_corrupt_path: Option<String>,
}

pub struct PersistenceHandler {
    snapshot_path: PathBuf,
    previous_path: PathBuf,
}

impl PersistenceHandler {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let snapshot_path = path.as_ref().to_path_buf();
        let previous_path = PathBuf::from(format!("{}.previous", snapshot_path.display()));
        Self {
            snapshot_path,
            previous_path,
        }
    }

    pub fn snapshot_path(&self) -> &Path {
        &self.snapshot_path
    }

    pub fn previous_path(&self) -> &Path {
        &self.previous_path
    }

    fn ensure_parent(&self) -> Result<(), String> {
        if let Some(parent) = self
            .snapshot_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create snapshot directory: {error}"))?;
        }
        Ok(())
    }

    /// Serializes LTM and atomically replaces the primary snapshot. Before the
    /// replacement, the last good primary is copied to `.previous`.
    pub fn save_snapshot(&self, kernel: &Kernel) -> Result<(), String> {
        self.ensure_parent()?;
        kernel
            .ltm
            .validate()
            .map_err(|error| format!("refusing to save invalid LTM state: {error}"))?;
        let bytes = encode_snapshot(&kernel.ltm)?;
        if bytes.len() as u64 > MAX_SNAPSHOT_BYTES {
            return Err(format!(
                "snapshot exceeds the {} byte limit",
                MAX_SNAPSHOT_BYTES
            ));
        }

        let temp_path = temporary_path(&self.snapshot_path);
        prepare_temporary_path(&temp_path)?;
        let write_result = (|| {
            let mut temp_file = open_private_new(&temp_path)?;
            temp_file
                .write_all(&bytes)
                .map_err(|error| format!("snapshot temporary-file write failed: {error}"))?;
            temp_file
                .sync_all()
                .map_err(|error| format!("snapshot temporary-file sync failed: {error}"))?;
            Ok::<(), String>(())
        })();
        if let Err(error) = write_result {
            let _ = fs::remove_file(&temp_path);
            return Err(error);
        }

        if self.snapshot_path.exists() {
            reject_symlink(&self.snapshot_path)?;
            match read_snapshot(&self.snapshot_path) {
                Ok(_) => {
                    let previous_temp = temporary_path(&self.previous_path);
                    prepare_temporary_path(&previous_temp)?;
                    fs::copy(&self.snapshot_path, &previous_temp)
                        .map_err(|error| format!("snapshot backup copy failed: {error}"))?;
                    sync_file(&previous_temp)?;
                    fs::rename(&previous_temp, &self.previous_path)
                        .map_err(|error| format!("snapshot backup rename failed: {error}"))?;
                }
                Err(_) => {
                    let corrupt_path = corrupt_path(&self.snapshot_path);
                    fs::rename(&self.snapshot_path, &corrupt_path).map_err(|error| {
                        format!("failed to preserve corrupt primary before save: {error}")
                    })?;
                }
            }
        }

        fs::rename(&temp_path, &self.snapshot_path)
            .map_err(|error| format!("snapshot atomic rename failed: {error}"))?;
        if let Some(parent) = self
            .snapshot_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            // Directory fsync makes the rename durable on Unix. Some platforms
            // do not support it, so failure here is intentionally non-fatal.
            let _ = File::open(parent).and_then(|directory| directory.sync_all());
        }
        Ok(())
    }

    fn load_path(&self, path: &Path, kernel: &mut Kernel) -> Result<(), String> {
        kernel.ltm = read_snapshot(path)?;
        Ok(())
    }

    pub fn load_snapshot(&self, kernel: &mut Kernel) -> Result<(), String> {
        self.load_path(&self.snapshot_path, kernel)
    }

    /// Loads the primary snapshot, falling back to `.previous` when the latest
    /// snapshot is invalid. A corrupt primary is preserved for diagnosis.
    pub fn recover(&self, kernel: &mut Kernel) -> Result<RecoveryReport, String> {
        if !self.snapshot_path.exists() {
            if self.previous_path.exists() {
                self.load_path(&self.previous_path, kernel)?;
                return Ok(RecoveryReport {
                    source: RecoverySource::Previous,
                    warning: Some(
                        "primary snapshot was absent; recovered previous snapshot".to_string(),
                    ),
                    preserved_corrupt_path: None,
                });
            }
            return Ok(RecoveryReport {
                source: RecoverySource::Empty,
                warning: None,
                preserved_corrupt_path: None,
            });
        }

        match self.load_path(&self.snapshot_path, kernel) {
            Ok(()) => Ok(RecoveryReport {
                source: RecoverySource::Primary,
                warning: None,
                preserved_corrupt_path: None,
            }),
            Err(primary_error) => {
                let corrupt_path = corrupt_path(&self.snapshot_path);
                fs::rename(&self.snapshot_path, &corrupt_path).map_err(|error| {
                    format!(
                        "{primary_error}; additionally failed to preserve corrupt snapshot: {error}"
                    )
                })?;

                if !self.previous_path.exists() {
                    return Err(format!(
                        "{primary_error}; corrupt snapshot preserved at {}, but no previous snapshot exists",
                        corrupt_path.display()
                    ));
                }
                self.load_path(&self.previous_path, kernel).map_err(|previous_error| {
                    format!(
                        "{primary_error}; previous snapshot recovery also failed: {previous_error}"
                    )
                })?;
                Ok(RecoveryReport {
                    source: RecoverySource::Previous,
                    warning: Some(primary_error),
                    preserved_corrupt_path: Some(corrupt_path.display().to_string()),
                })
            }
        }
    }

    pub fn snapshot_size_bytes(&self) -> u64 {
        fs::metadata(&self.snapshot_path)
            .map(|metadata| metadata.len())
            .unwrap_or(0)
    }

    pub fn snapshot_age_seconds(&self) -> Option<u64> {
        let modified = fs::metadata(&self.snapshot_path).ok()?.modified().ok()?;
        SystemTime::now()
            .duration_since(modified)
            .ok()
            .map(|duration| duration.as_secs())
    }

    pub fn compact(&self, _kernel: &mut Kernel, _threshold: f32) {
        // Compaction remains a future schema-level operation.
    }
}

fn sync_file(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("snapshot sync failed for {}: {error}", path.display()))
}

fn encode_snapshot(ltm: &LongTermMemory) -> Result<Vec<u8>, String> {
    let checksum = ltm_checksum(ltm)?;
    serde_json::to_vec(&SnapshotData {
        version: SNAPSHOT_VERSION,
        checksum,
        ltm: ltm.clone(),
    })
    .map_err(|error| format!("snapshot serialization failed: {error}"))
}

fn read_snapshot(path: &Path) -> Result<LongTermMemory, String> {
    reject_symlink(path)?;
    let metadata = fs::metadata(path)
        .map_err(|error| format!("snapshot metadata failed for {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "snapshot path is not a regular file: {}",
            path.display()
        ));
    }
    if metadata.len() > MAX_SNAPSHOT_BYTES {
        return Err(format!(
            "snapshot {} exceeds the {} byte limit",
            path.display(),
            MAX_SNAPSHOT_BYTES
        ));
    }
    let bytes = fs::read(path)
        .map_err(|error| format!("snapshot read failed for {}: {error}", path.display()))?;
    let data: SnapshotData = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "snapshot deserialization failed for {}: {error}",
            path.display()
        )
    })?;
    if data.version != SNAPSHOT_VERSION {
        return Err(format!(
            "unsupported snapshot version {} in {}",
            data.version,
            path.display()
        ));
    }
    let actual_checksum = ltm_checksum(&data.ltm)?;
    if data.checksum != actual_checksum {
        return Err(format!(
            "snapshot integrity check failed for {}",
            path.display()
        ));
    }
    data.ltm.validate().map_err(|error| {
        format!(
            "snapshot structure is invalid for {}: {error}",
            path.display()
        )
    })?;
    Ok(data.ltm)
}

fn ltm_checksum(ltm: &LongTermMemory) -> Result<String, String> {
    let bytes = serde_json::to_vec(ltm)
        .map_err(|error| format!("snapshot checksum serialization failed: {error}"))?;
    let mut checksum = 0xcbf29ce484222325_u64;
    for byte in bytes {
        checksum ^= u64::from(byte);
        checksum = checksum.wrapping_mul(0x100000001b3);
    }
    Ok(format!("{checksum:016x}"))
}

fn temporary_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.tmp.{}", path.display(), std::process::id()))
}

fn corrupt_path(path: &Path) -> PathBuf {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    PathBuf::from(format!(
        "{}.corrupt.{millis}.{}",
        path.display(),
        std::process::id()
    ))
}

fn reject_symlink(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(format!(
            "refusing to use symlink snapshot path: {}",
            path.display()
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "snapshot path metadata failed for {}: {error}",
            path.display()
        )),
    }
}

fn prepare_temporary_path(path: &Path) -> Result<(), String> {
    reject_symlink(path)?;
    if path.exists() {
        let metadata = fs::metadata(path)
            .map_err(|error| format!("temporary snapshot metadata failed: {error}"))?;
        if !metadata.is_file() {
            return Err(format!(
                "temporary snapshot path is not a regular file: {}",
                path.display()
            ));
        }
        fs::remove_file(path)
            .map_err(|error| format!("stale temporary snapshot removal failed: {error}"))?;
    }
    Ok(())
}

fn open_private_new(path: &Path) -> Result<File, String> {
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(path)
        .map_err(|error| format!("snapshot temporary-file open failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::MemoryLayer;
    use crate::payload::{Payload, PayloadData, PayloadHeader};
    use crate::types::Timestamp;

    fn temp_snapshot(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "tst-{name}-{}-{nonce}.snapshot",
            std::process::id()
        ))
    }

    fn preference(value: &str) -> Payload {
        Payload {
            header: PayloadHeader {
                payload_type: 2,
                version: 1,
                created_ts: Timestamp(0),
                last_access_ts: Timestamp(0),
                access_count: 1,
            },
            data: PayloadData::Preference {
                key: "preference".to_string(),
                value: value.to_string(),
                weight: 1.0,
            },
        }
    }

    #[test]
    fn snapshot_roundtrip_and_previous_backup() {
        let path = temp_snapshot("roundtrip");
        let handler = PersistenceHandler::new(&path);
        let mut kernel = Kernel::new();
        kernel
            .store_memory(MemoryLayer::LTM, b"preference", preference("first"))
            .unwrap();
        handler.save_snapshot(&kernel).unwrap();
        kernel
            .store_memory(MemoryLayer::LTM, b"preference", preference("second"))
            .unwrap();
        handler.save_snapshot(&kernel).unwrap();
        assert!(handler.previous_path().exists());

        let mut recovered = Kernel::new();
        let report = handler.recover(&mut recovered).unwrap();
        assert_eq!(report.source, RecoverySource::Primary);
        match recovered.route_read(b"preference").unwrap().data {
            PayloadData::Preference { value, .. } => assert_eq!(value, "second"),
            _ => panic!("wrong payload type"),
        }

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(handler.previous_path());
    }

    #[test]
    fn corrupt_primary_falls_back_to_previous() {
        let path = temp_snapshot("recovery");
        let handler = PersistenceHandler::new(&path);
        let mut kernel = Kernel::new();
        kernel
            .store_memory(MemoryLayer::LTM, b"preference", preference("safe"))
            .unwrap();
        handler.save_snapshot(&kernel).unwrap();
        handler.save_snapshot(&kernel).unwrap();
        fs::write(&path, b"not-json").unwrap();

        let mut recovered = Kernel::new();
        let report = handler.recover(&mut recovered).unwrap();
        assert_eq!(report.source, RecoverySource::Previous);
        assert!(
            report
                .preserved_corrupt_path
                .as_ref()
                .is_some_and(|path| Path::new(path).exists())
        );
        assert!(recovered.route_read(b"preference").is_some());

        if let Some(corrupt_path) = report.preserved_corrupt_path {
            let _ = fs::remove_file(corrupt_path);
        }
        let _ = fs::remove_file(handler.previous_path());
    }

    #[test]
    fn checksum_detects_parseable_tampering_and_uses_backup() {
        let path = temp_snapshot("checksum");
        let handler = PersistenceHandler::new(&path);
        let mut kernel = Kernel::new();
        kernel
            .store_memory(MemoryLayer::LTM, b"preference", preference("safe"))
            .unwrap();
        handler.save_snapshot(&kernel).unwrap();
        handler.save_snapshot(&kernel).unwrap();

        let tampered = fs::read_to_string(&path).unwrap().replace("safe", "evil");
        fs::write(&path, tampered).unwrap();
        let mut recovered = Kernel::new();
        let report = handler.recover(&mut recovered).unwrap();
        assert_eq!(report.source, RecoverySource::Previous);
        assert!(
            report
                .warning
                .as_deref()
                .is_some_and(|warning| warning.contains("integrity"))
        );
        match recovered.route_read(b"preference").unwrap().data {
            PayloadData::Preference { value, .. } => assert_eq!(value, "safe"),
            _ => panic!("wrong payload type"),
        }

        if let Some(corrupt_path) = report.preserved_corrupt_path {
            let _ = fs::remove_file(corrupt_path);
        }
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(handler.previous_path());
    }

    #[test]
    fn interrupted_temporary_write_does_not_replace_primary() {
        let path = temp_snapshot("temporary-interruption");
        let handler = PersistenceHandler::new(&path);
        let mut kernel = Kernel::new();
        kernel
            .store_memory(MemoryLayer::LTM, b"preference", preference("safe"))
            .unwrap();
        handler.save_snapshot(&kernel).unwrap();
        let temp = temporary_path(&path);
        fs::write(&temp, b"partial").unwrap();

        let mut recovered = Kernel::new();
        let report = handler.recover(&mut recovered).unwrap();
        assert_eq!(report.source, RecoverySource::Primary);
        assert!(recovered.route_read(b"preference").is_some());

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(temp);
    }

    #[cfg(unix)]
    #[test]
    fn snapshot_symlinks_are_rejected_without_touching_the_target() {
        use std::os::unix::fs::symlink;

        let path = temp_snapshot("symlink");
        let target = temp_snapshot("symlink-target");
        fs::write(&target, b"do-not-overwrite").unwrap();
        symlink(&target, &path).unwrap();
        let handler = PersistenceHandler::new(&path);
        let kernel = Kernel::new();

        assert!(handler.save_snapshot(&kernel).is_err());
        assert_eq!(fs::read(&target).unwrap(), b"do-not-overwrite");

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&target);
        let _ = fs::remove_file(temporary_path(&path));
    }
}
