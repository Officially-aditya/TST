# Structured Code-Review Validation

Conversational text and code-review findings have different safety needs. A review worker must generate a JSON object matching `CodeReviewOutput`; prose surrounding the object is treated as a parse failure.

## Output Schema

### `CodeReviewOutput`

```python
@dataclass
class CodeReviewOutput:
    issues: List[CodeIssue]
    metadata: ReviewMetadata
```

### `CodeIssue` (Required Fields)

| Field | Type | Description |
|-------|------|-------------|
| `file_path` | string | Root-relative path (e.g., `router/server.py`) |
| `line_start` | integer | 1-based, inclusive |
| `line_end` | integer | 1-based, inclusive, ≥ `line_start` |
| `content_hash` | string | SHA-256 of file content at generation time |
| `severity` | enum | `critical` \| `major` \| `minor` \| `info` |
| `category` | enum | `correctness` \| `security` \| `performance` \| `maintainability` \| `style` \| `documentation` |
| `title` | string | One-line summary (max 120 chars) |
| `explanation` | string | Detailed reasoning |
| `suggested_fix` | string | Concrete code change or remediation |
| `confidence` | float | 0.0–1.0 |
| `related_symbols` | string[] | Optional qualified symbol names |

### `ReviewMetadata`

```python
@dataclass
class ReviewMetadata:
    model: str                    # Model identifier used
    prompt_version: str           # Prompt template version
    generation_ms: int            # Model inference time
    tokens_input: int
    tokens_output: int
```

---

## Validation Pipeline

`ReviewValidator` processes raw model output through strict stages:

### Stage 1: JSON + Schema Parse

```python
def parse_output(raw: str) -> CodeReviewOutput:
    # 1. Extract JSON object (handles markdown code fences)
    # 2. Validate against CodeReviewOutput schema
    # 3. Return typed object or raise ValidationError
```

**Failure modes**: Invalid JSON, schema mismatch, extra fields → `parse_failure` counter + rejected

### Stage 2: File Existence

```python
def validate_file(issue: CodeIssue, project_root: Path) -> ValidationResult:
    path = project_root / issue.file_path
    if not path.is_file():
        return ValidationError("file_not_found", f"File not found: {issue.file_path}")
    if not path.is_relative_to(project_root):
        return ValidationError("path_traversal", "Path escapes project root")
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    if issue.line_start < 1 or issue.line_end > len(lines):
        return ValidationError("line_range_invalid", f"Lines {issue.line_start}-{issue.line_end} invalid for {len(lines)}-line file")
    return ValidatedIssue(actual_hash=sha256(content))
```

### Stage 3: Content Hash Verification

```python
def validate_hash(issue: CodeIssue, actual_hash: str) -> ValidationResult:
    if issue.content_hash != actual_hash:
        return ValidationError("stale_content", 
            f"Content hash mismatch: expected {issue.content_hash[:8]}, got {actual_hash[:8]}. File changed after generation.")
    return OK
```

**Purpose**: Prevents stale model output from being shown after file changes.

### Stage 4: Symbol Validation (Optional)

```python
def validate_symbols(issue: CodeIssue, code_graph: CodeGraph) -> ValidationResult:
    for symbol in issue.related_symbols:
        if not code_graph.has_symbol(symbol, issue.file_path):
            return ValidationError("unknown_symbol", 
                f"Related symbol '{symbol}' not found in {issue.file_path}")
    return OK
```

**Only runs** when code graph is supplied to validator.

### Stage 5: Deduplication

```python
def deduplicate(issues: List[ValidatedIssue]) -> List[ValidatedIssue]:
    seen = set()
    unique = []
    for issue in issues:
        identity = (issue.file_path, issue.line_start, issue.line_end, issue.title)
        if identity in seen:
            continue  # Duplicate, increment duplicate counter
        seen.add(identity)
        unique.append(issue)
    return unique
```

**Identity key**: `(file_path, line_start, line_end, title)` — same location + same title = duplicate

### Stage 6: Confidence Threshold

```python
def filter_confidence(issues: List[ValidatedIssue], threshold: float = 0.5) -> SplitResult:
    safe = [i for i in issues if i.confidence >= threshold]
    flagged = [i for i in issues if i.confidence < threshold]
    return SplitResult(safe_to_display=safe, flagged_low_confidence=flagged)
```

**Default threshold**: 0.5 (configurable)

---

## Worker Discipline

`ReviewWorker` enforces generation constraints:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `temperature` | 0.0 | Deterministic output |
| `do_sample` | false | No sampling variance |
| `response_format` | JSON schema | Structured output enforcement |
| `max_tokens` | 2048 | Bounded output |

**Prompt injection**: JSON schema for `CodeReviewOutput` included in system prompt.

**Counters exposed**:
- `generation_calls`
- `parse_failures`
- `accepted_issues`
- `rejected_issues`
- `low_confidence_issues`

---

## Error Handling Philosophy

**Parse failures and rejected findings are data, not exceptions.**

```python
@dataclass
class ReviewResult:
    safe_to_display: List[CodeIssue]      # Passed all validation, confidence ≥ threshold
    flagged_low_confidence: List[CodeIssue]  # Passed validation, confidence < threshold
    parse_failures: List[ParseFailure]    # JSON/schema errors
    validation_failures: List[ValidationFailure]  # File/line/hash/symbol errors
    duplicates: List[CodeIssue]           # Removed by dedup
    raw_output_debug: Optional[str]       # Only in explicit debug mode
```

**Display rule**: Only `safe_to_display` shown to user. Others available in debug/telemetry.

---

## Debug Mode

```python
worker = ReviewWorker(..., debug=True)
result = worker.review(files, code_graph)
# result.raw_output_debug contains raw model output (size-bounded: max 64 KB)
```

**Warning**: Raw output can contain source code or memory context. **Never write to ordinary logs.** Use only for explicit debugging sessions.

---

## Example: Valid Finding

```json
{
  "file_path": "router/server.py",
  "line_start": 42,
  "line_end": 45,
  "content_hash": "sha256:a1b2c3d4...",
  "severity": "major",
  "category": "correctness",
  "title": "Missing timeout on external HTTP call",
  "explanation": "The httpx.AsyncClient call at line 42 has no timeout configured. Network issues can cause indefinite hangs.",
  "suggested_fix": "Add timeout: `async with httpx.AsyncClient(timeout=30.0) as client:`",
  "confidence": 0.92,
  "related_symbols": ["run_route"]
}
```

---

## Example: Rejected Finding

```json
{
  "validation_failures": [
    {
      "issue": { "file_path": "router/server.py", "line_start": 42, ... },
      "error": "stale_content",
      "message": "Content hash mismatch: expected a1b2c3d4, got e5f6g7h8. File changed after generation."
    }
  ]
}
```

---

## Configuration

```python
ReviewValidator(
    project_root=Path("/path/to/repo"),
    code_graph=code_graph,           # Optional, enables symbol validation
    confidence_threshold=0.5,        # Default 0.5
    max_raw_debug_bytes=65536        # Default 64 KB
)
```

---

## Non-Goals (v0.2)

- Automatic fix execution (findings are display-only)
- Cross-file refactoring suggestions
- PR/MR integration (GitHub/GitLab/Bitbucket)
- Custom severity/category schemas
- Human-in-the-loop approval workflows