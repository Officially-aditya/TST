# Configuration Reference

This reference documents all configuration options, environment variables, CLI flags, and default settings across the TST framework.

## Complete Environment Variable List

### 1. Kernel Process Configuration

| Variable | Default Value | Valid Range | Description |
|----------|---------------|-------------|-------------|
| `TST_KERNEL_BIN` | `tst_memory/target/release/server` | Valid file path | Path to the Rust kernel release binary. |
| `TST_KERNEL_TIMEOUT` | `30.0` | `1.0` – `300.0` | Timeout in seconds for kernel STDIO requests. |
| `TST_KERNEL_MAX_RESPONSE_BYTES` | `8388608` | `1024` – `67108864` | Maximum allowed response size in bytes (default 8 MiB). |
| `RUST_LOG` | `warn` | `error`, `warn`, `info`, `debug`, `trace` | Kernel logging verbosity (controlled by `env_logger`). |

---

### 2. Short-Term Memory (STM) Configuration

| Variable | Default Value | Valid Range | Description |
|----------|---------------|-------------|-------------|
| `TST_STM_CAPACITY` | `256` | `1` – `100000` | Maximum number of concurrent entries in STM ring buffer. |
| `TST_STM_HALF_LIFE_SECONDS` | `1800` | `1.0` – `864000.0` | Half-life in seconds for wall-clock exponential score decay (30 min). |
| `TST_STM_PROMOTION_THRESHOLD` | `10.0` | `0.1` – `1000.0` | Score threshold triggering automatic promotion from STM to LTM. |
| `TST_STM_READ_REINFORCEMENT` | `1.0` | `0.0` – `100.0` | Reinforcement score added to record on read access (`memory.get`). |
| `TST_STM_WRITE_REINFORCEMENT` | `2.0` | `0.0` – `100.0` | Reinforcement score added to record on store/update (`memory.store`). |
| `TST_STM_EXPIRY_SCORE` | `0.01` | `0.0001` – `1.0` | Score threshold below which unused STM records expire and free slots. |

---

### 3. Long-Term Memory (LTM) & Persistence

| Variable | Default Value | Valid Range | Description |
|----------|---------------|-------------|-------------|
| `TST_SNAPSHOT_PATH` | `.tst/ltm.snapshot` | Valid file path | File system path for LTM snapshot persistence. |
| `TST_SNAPSHOT_DEBOUNCE_MS` | `5000` | `100` – `600000` | Debounce delay in milliseconds before flushing dirty LTM state to disk. |

---

### 4. Code Graph & Repository Scanner

| Variable | Default Value | Valid Range | Description |
|----------|---------------|-------------|-------------|
| `TST_SCAN_MAX_FILE_SIZE` | `1048576` | `1024` – `50000000` | Maximum file size in bytes (1 MB default) to analyze. |
| `TST_SCAN_MAX_TOTAL_BYTES` | `104857600` | `1024` – `1000000000` | Maximum total bytes (100 MB default) across all analyzed files. |
| `TST_SCAN_MAX_FILES` | `10000` | `1` – `1000000` | Maximum file count limit for scanner. |
| `TST_SCAN_FOLLOW_SYMLINKS` | `false` | `true`, `false` | Whether to follow directory symlinks during scan. |

---

### 5. Model Router & Inference Settings

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `TST_ROUTER_TIER1_MODEL` | `google/gemma-2-2b-it` | Model identifier for fast Tier 1 routing decisions. |
| `TST_ROUTER_TIER2_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Model identifier for complex Tier 2 routing decisions. |
| `TST_ROUTER_DEVICE` | `auto` | Device placement (`auto`, `cpu`, `cuda`, `mps`). |
| `TST_ROUTER_CONFIDENCE_THRESHOLD` | `0.80` | Minimum confidence score to accept model decision over fallback. |

---

## Configuration Files

TST supports reading configuration from `.env` files in the working directory when launched via Python.

### Example `.env` File

```env
# Kernel Process Settings
TST_KERNEL_BIN=tst_memory/target/release/server
TST_KERNEL_TIMEOUT=15.0

# Memory Tuning
TST_STM_CAPACITY=512
TST_STM_HALF_LIFE_SECONDS=3600
TST_STM_PROMOTION_THRESHOLD=15.0

# Persistence
TST_SNAPSHOT_PATH=.tst/ltm.snapshot
TST_SNAPSHOT_DEBOUNCE_MS=2000

# Scanner Settings
TST_SCAN_MAX_FILE_SIZE=2097152
```

---

## Validation & Error Behavior

- **Strict Validation**: Invalid environment variable values (e.g. `TST_STM_CAPACITY=-5` or non-numeric strings) cause startup validation failures.
- **Fail-Fast**: The kernel refuses to start if configuration parameters are internally inconsistent (e.g. `PROMOTION_THRESHOLD < EXPIRY_SCORE`).
- **Doctor Check**: `tst doctor` validates current environment configuration and reports warnings or errors.