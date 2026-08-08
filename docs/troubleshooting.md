# Troubleshooting Guide

This guide covers common issues, error messages, and diagnostic steps for TST.

## Diagnostic Tool: `tst doctor`

Always start troubleshooting with `tst doctor`. It inspects your environment and reports system health.

```bash
# Human-readable summary
tst doctor

# Detailed JSON (use in bug reports)
tst doctor --json
```

### What `tst doctor` Checks

- Python version (requires 3.10+)
- Installed packages and optional extras (`[analysis]`, `[router]`, `[models]`)
- Rust toolchain (`cargo`, `rustc`, edition 2024 support)
- Kernel binary presence, version, and execution permissions
- Active kernel execution strategy (`source_build` vs `env_bin` vs `bundled`)
- System platform and architecture
- Snapshot directory write permissions

---

## Common Issues & Solutions

### 1. Kernel Binary Not Found

**Symptom**: `FileNotFoundError: Kernel binary not found at tst_memory/target/release/server`

**Cause**: TST normal startup **never** runs Cargo automatically. The kernel binary must be built explicitly before first run.

**Solution**:
```bash
# Explicit release build
tst kernel build

# Verify build succeeded
ls -la tst_memory/target/release/server
```

**Alternative (Prebuilt binary)**:
If using a prebuilt binary or custom path:
```bash
export TST_KERNEL_BIN=/path/to/custom/server
tst doctor
```

---

### 2. Kernel Starts but Never Becomes `READY`

**Symptom**: Process hangs during startup, client times out waiting for `READY` signal.

**Cause**: The kernel recovers LTM from snapshot before emitting `READY`. If the snapshot file is corrupt, unreadable, or locked by another process, recovery hangs or fails.

**Diagnostic Steps**:
1. Check snapshot permissions and directory:
   ```bash
   ls -la .tst/
   ```
2. Inspect captured kernel stderr (if running in debug mode):
   ```bash
   RUST_LOG=tst_memory=debug tst doctor
   ```

**Recovery Procedure**:
```bash
# 1. Back up existing snapshots (DO NOT delete immediately)
cp .tst/ltm.snapshot .tst/ltm.snapshot.bak.$(date +%s)
cp .tst/ltm.snapshot.previous .tst/ltm.snapshot.previous.bak.$(date +%s)

# 2. Check if primary snapshot is corrupt
# The kernel automatically preserves corrupt primary as .tst/ltm.snapshot.corrupt.<timestamp>
# and falls back to .tst/ltm.snapshot.previous.

# 3. If both are corrupt and non-recoverable, reset LTM:
rm .tst/ltm.snapshot .tst/ltm.snapshot.previous
# Kernel will start with clean LTM
```

---

### 3. `tst chat` Fails with Import Errors

**Symptom**: `ImportError: PyTorch or Transformers required for chat`

**Cause**: The core TST installation deliberately excludes large model dependencies (~2 GB+) to stay lightweight.

**Solution**:
```bash
# Install model extra
python -m pip install -e '.[models]'

# On Apple Silicon (macOS arm64), PyTorch MPS support is automatically used
# On Linux, ensure CUDA drivers match PyTorch build if using GPU
```

---

### 4. JavaScript/TypeScript/Rust Reports `structural-fallback` Parser

**Symptom**: Code graph works, but metadata shows `parser: "structural_fallback"` instead of `"tree_sitter"`.

**Cause**: Native Tree-sitter grammar wheels are not installed. Core install uses standard Python `ast` for Python, but structural regex for JS/TS/Rust.

**Solution**:
```bash
# Install analysis extra (includes tree-sitter grammars)
python -m pip install -e '.[analysis]'

# Verify parser in scan output
tst analyze . --json | grep parser
```

**Platform Fallback**: If prebuilt Tree-sitter wheels are unavailable for your specific platform/Python version, structural fallback continues to work safely without crashing.

---

### 5. Source File Missing from Code Graph Analysis

**Symptom**: A file exists in your project but does not appear in `tst analyze` results.

**Cause**: Scanner safety rules excluded the file (size limit, secret pattern, ignored directory, binary content, etc.).

**Diagnostic Steps**:
```bash
# Check skipped files in scan report
tst analyze path/to/repo --json | jq '.report.skipped'
```

**Common Skip Reasons & Workarounds**:

| Reason | Explanation | Fix/Workaround |
|--------|-------------|----------------|
| `dependency_directory` | Matches `node_modules`, `target`, `venv`, etc. | Expected behavior. Do not analyze dependencies. |
| `secret_filename` | Matches `.env`, `id_rsa`, `*.pem`, `secrets*` | Security exclusion. Rename file if it's not a secret. |
| `file_too_large` | Exceeds `max_file_size` (1 MB default) | Increase limit: `ScanConfig(max_file_size=5_000_000)` |
| `non_utf8_binary` | File contains non-UTF-8 bytes | Convert to UTF-8 or exclude. |
| `symlink_skipped` | Symlinks not followed by default | Enable in config if safe: `ScanConfig(follow_symlinks=True)` |
| `outside_root` | Path escapes scan root directory | Scan from higher directory root. |

---

### 6. Code Review Finding Not Displayed

**Symptom**: Model generated a code review finding, but it doesn't appear in output.

**Cause**: `ReviewValidator` withheld the finding during validation pipeline.

**Diagnostic Steps**:
Enable debug mode to inspect validation failures:

```python
from tst.worker import ReviewWorker

worker = ReviewWorker(..., debug=True)
result = worker.review(files, code_graph)

# Inspect why findings were rejected
print("Parse failures:", result.parse_failures)
print("Validation failures:", result.validation_failures)
print("Low confidence:", result.flagged_low_confidence)
```

**Common Validation Rejections**:

| Rejection | Cause | Fix |
|-----------|-------|-----|
| `stale_content` | File changed after model generated finding (hash mismatch) | Re-run review on fresh file content |
| `file_not_found` | Model hallucinated path not in project | Check prompt / repository index |
| `line_range_invalid` | Line numbers outside file bounds | Model hallucinated line numbers |
| `unknown_symbol` | Related symbol doesn't exist in file | Model referenced symbol from another file |
| `confidence_low` | Confidence score < threshold (default 0.5) | Lower threshold or tune model prompt |

---

### 7. Protocol Error: `INVALID_ENVELOPE` or Timeout

**Symptom**: `ProtocolError: Kernel response timeout after 30s` or `INVALID_ENVELOPE: Unknown field 'foo'`

**Cause**:
- **Timeout**: Kernel deadlock, infinite loop in Rust, or heavy memory load.
- **Envelope error**: Client sent fields not in protocol v1 schema.

**Solution**:
1. Check protocol version compatibility (must be version 1).
2. Run protocol contract tests:
   ```bash
   python -m pytest -m protocol_contract
   ```
3. Restart kernel client (automatically handled by Python process wrapper on timeout).

---

### 8. LTM Persistence Disk Full / Permission Denied

**Symptom**: `PERSISTENCE_ERROR: Failed to write snapshot: Permission denied (os error 13)`

**Cause**: Snapshot path directory (`.tst/`) is not writable or disk space exhausted.

**Solution**:
```bash
# Check directory permissions
mkdir -p .tst
chmod 755 .tst

# Change snapshot location via environment variable
export TST_SNAPSHOT_PATH=/tmp/my_project_ltm.snapshot
```

---

## Getting Help

If an issue persists:

1. Run `tst doctor --json > doctor_report.json`
2. Gather kernel stderr (set `RUST_LOG=tst_memory=debug`)
3. Open an issue with the doctor report and reproduction steps.