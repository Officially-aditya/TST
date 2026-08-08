# Repository Scanner and Code Graph

`tst.analysis.IncrementalIndexer` turns a selected source root into a line-aware graph without importing or executing project code.

## Parser Selection

| Language | File Suffixes | Primary Parser | Dependency-Light Behavior |
|----------|---------------|----------------|---------------------------|
| Python | `.py` | Tree-sitter | Standard-library `ast` (always active) |
| JavaScript/JSX | `.js`, `.jsx`, `.mjs`, `.cjs` | Tree-sitter | Line-aware structural fallback |
| TypeScript/TSX | `.ts`, `.tsx` | Tree-sitter | Line-aware structural fallback |
| Rust | `.rs` | Tree-sitter | Line-aware structural fallback |

### Parser Loading

- Tree-sitter grammars are loaded **lazily** via `ParserRegistry` when the `[analysis]` extra is installed.
- The core install retains precise Python AST parsing via standard-library `ast`.
- Grammar wheels: `tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-typescript`, `tree-sitter-rust`.
- If native grammar wheels are absent or fail on a platform, parsers fall back to safe structural line-aware regex extraction without throwing runtime exceptions.

### Parser Output Models (`tst.analysis.models`)

Each parser produces a normalized `ParsedFile`:

```python
@dataclass(slots=True)
class ParsedFile:
    path: str                            # Root-relative POSIX path
    language: str                        # "python" | "javascript" | "typescript" | "tsx" | "rust"
    content_hash: str                    # SHA-256 digest of source bytes
    parser: str                          # "tree_sitter" | "ast" | "structural_fallback"
    symbols: list[ParsedSymbol]          # Extracted functions, classes, methods, structs, etc.
    imports: list[ParsedImport]          # Raw import statements before repository resolution
    errors: list[str] = field(default_factory=list)  # Parser warnings/syntax errors
    line_count: int = 1                  # Total line count
    failed: bool = False                 # Set to True if file could not be parsed safely
```

**Symbol fields (`ParsedSymbol`)**:
- `name`: Local symbol identifier (e.g. `run_route`)
- `qualified_name`: Qualified module name (e.g. `tst.routing.router.run_route`)
- `kind`: `SymbolKind` enum (`MODULE`, `CLASS`, `FUNCTION`, `METHOD`, `STRUCT`, `ENUM`, `TRAIT`, `INTERFACE`, `VARIABLE`, `SYMBOL`)
- `span`: `SourceSpan` (1-based positive `start_line`, `end_line`, `start_column`, `end_column`)
- `signature`: Optional full signature string
- `parent_qualified_name`: Enclosing class/module qualified name
- `references`: List of `ParsedReference` (`name`, `kind`: `CALL` | `REFERENCE` | `INHERITS` | `IMPLEMENTS`, `line`)
- `bases`: Tuple of base class names
- `implements`: Tuple of implemented interface/trait names
- `content_hash`: Optional symbol-level content hash

**Import fields (`ParsedImport`)**:
- `module`: Imported module string (without leading dots)
- `names`: Tuple of imported symbol names (`("run_route",)` or empty)
- `aliases`: Tuple of `(name, alias)` pairs
- `level`: Relative import depth (0 = absolute, 1 = current package `.`, 2 = `..`)
- `line`: 1-based line number

---

## Nodes and Edges (`tst.analysis.graph_builder`)

### Node Representation (`GraphNode`)

Graph nodes use BLAKE2b 64-bit integer IDs (`node_id: int`) derived deterministically from candidate string identities (`_stable_id`).

```python
@dataclass(slots=True)
class GraphNode:
    node_id: int                         # BLAKE2b 64-bit integer stable hash
    node_type: str                       # "project" | "file" | "function" | "class" | "method" | ...
    name: str                            # Short display label
    qualified_name: str                  # Unique qualified name (e.g., "tst.cli.main")
    file_path: str | None = None         # Root-relative path
    start_line: int | None = None        # 1-based line start
    end_line: int | None = None          # 1-based line end
    signature: str | None = None        # Function/method signature
    content_hash: str | None = None      # File or symbol content SHA-256
    parent: int | None = None            # Parent node_id in hierarchy
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Edge Representation (`GraphEdge`)

Edges maintain directed connections between node IDs with explicit confidence ratings (0.0 to 1.0).

```python
class EdgeKind(str, Enum):
    CONTAINS = "contains"      # Hierarchy (project -> file, file -> symbol, class -> method)
    DEFINES = "defines"        # File defines symbol
    IMPORTS = "imports"        # Import relationship
    CALLS = "calls"            # Call relationship
    REFERENCES = "references"  # Non-call reference
    INHERITS = "inherits"      # Subclassing / trait implementation
    IMPLEMENTS = "implements"  # Interface implementation
    TESTS = "tests"            # Test function -> production symbol

@dataclass(slots=True)
class GraphEdge:
    source: int                # Source node_id
    target: int                # Target node_id
    kind: EdgeKind             # Relationship type
    confidence: float = 1.0    # Confidence rating between 0.0 and 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Symbol & Import Resolution (`tst.analysis.resolver`)

`SymbolResolver` performs multi-pass repository-wide symbol linking:

1. **Hierarchy Linking**: `CONTAINS` edges connect project → file and parent symbol → child symbol. `DEFINES` edges link files to all defined symbols.
2. **Import Resolution**:
   - **Python**: Resolves relative imports (`level > 0`) against package directory structures. Handles `src/` layout and module init files (`__init__.py`).
   - **JavaScript/TypeScript**: Resolves relative paths (`./`, `../`), index modules (`index.ts`), and path aliases.
   - **Rust**: Resolves `crate::`, `super::`, `self::` paths, mapping `lib.rs` and `main.rs` to crate roots.
3. **Call & Reference Resolution**: Matches parsed references to in-graph symbol qualified names. High-confidence exact matches create `CALLS`, `REFERENCES`, `INHERITS`, or `IMPLEMENTS` edges.
4. **Test Linkage**: Links test files/functions matching `test_*` or `*_test` to imported target symbols via `TESTS` edges.
5. **Unresolved Dependencies**: External libraries or missing local symbols produce explicit `external:<name>` nodes with `node_type="external"`.

---

## Graph Traversal & Queries

### `CodeGraph.query()`

Traverses the graph using a breadth-first queue with edge priority and strict budget enforcement:

```python
graph_slice = graph.query(
    start="tst.routing.router.run_route",  # String name/qualified_name or integer node_id
    depth=1,                               # Graph traversal depth
    max_nodes=40,                          # Node budget
    token_budget=2000                      # Estimated token budget
)
```

**Edge Traversal Priority**:
1. `CALLS` (priority 0)
2. `TESTS` (priority 1)
3. `IMPORTS` (priority 2)
4. `INHERITS` / `IMPLEMENTS` (priority 3)
5. `REFERENCES` (priority 4)
6. `CONTAINS` (priority 5)
7. `DEFINES` (priority 6)

**Budget Enforcement**:
- `max_nodes`: Maximum allowed nodes in `GraphSlice`.
- `token_budget`: Token estimation calculated as `max(1, (len(qualified_name) + len(signature)) // 4)`. Traversal halts and sets `truncated=True` if budgets are reached.

**QueryResult (`GraphSlice`)**:
```python
@dataclass(slots=True)
class GraphSlice:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool = False
    estimated_tokens: int = 0
```

### `CodeGraph.find()`

Locates graph nodes matching a string query with exact-match and suffix scoring:

```python
matches = graph.find("run_route", file_path="tst/routing/router.py")
```

---

## Incremental Indexing (`tst.analysis.incremental`)

`IncrementalIndexer` manages content-hash tracking so unchanged files bypass parsing entirely:

```python
indexer = IncrementalIndexer(root="/path/to/repo")
report = indexer.index()
```

### Index Report (`IndexReport`)

```python
@dataclass(slots=True)
class IndexReport:
    added: list[str]                       # Newly discovered file paths
    changed: list[str]                     # Modified file paths (hash changed)
    removed: list[str]                     # Deleted file paths
    unchanged: list[str]                   # Bypassed file paths (hash matched)
    parse_errors: dict[str, list[str]]     # Non-fatal parse/graph errors
    skipped: list[tuple[str, str]]         # (path, reason) pairs skipped by scanner
    parsed_files: int = 0                  # Count of files re-parsed
    elapsed_ms: float = 0.0                # Total indexing duration in ms
```

### Incremental Guarantees

- **Unchanged files**: Zero re-parsing or re-tokenization. The previous `ParsedFile` and subtree remain in memory.
- **Transactional updates**: If a file update or graph resolution fails, the staging builder discards the change and retains the prior valid code graph state.
- **Deletion handling**: Deleting a file cascades and removes all defined symbols and connected edges, leaving zero dangling edge references.

---

## Scanner Safety Limits (`tst.analysis.scanner`)

`ProjectScanner` enforces path safety, secret exclusion, and size limits:

```python
@dataclass(frozen=True, slots=True)
class ScanConfig:
    languages: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LANGUAGES))
    excluded_directories: frozenset[str] = DEFAULT_EXCLUDED_DIRECTORIES
    secret_patterns: tuple[str, ...] = DEFAULT_SECRET_PATTERNS
    max_file_bytes: int = 2 * 1024 * 1024       # 2 MB per file
    max_repository_bytes: int = 64 * 1024 * 1024  # 64 MB total across repository
    max_files: int = 10_000                      # Max total files
    skip_hidden_directories: bool = True
    skip_all_symlinks: bool = True
```

### Excluded Directories (`DEFAULT_EXCLUDED_DIRECTORIES`)
`.git`, `.hg`, `.svn`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `__pycache__`, `bower_components`, `build`, `coverage`, `dist`, `node_modules`, `site-packages`, `target`, `vendor`, `venv`.

### Secret Patterns (`DEFAULT_SECRET_PATTERNS`)
`.env`, `.env.*`, `*.key`, `*.pem`, `*.p12`, `*.pfx`, `*credentials*.json`, `*credentials*.yaml`, `*secret*.json`, `id_rsa*`, `id_ed25519*`, etc.

### Path Traversal Protection (`resolve_project_path`)
- All target paths must be strictly contained within `root.resolve()`.
- Outward-pointing relative paths (`../`) or symlinks escaping the project root raise `UnsafePathError`.
- Directory symlinks are **never** followed.
- Binary files containing NULL bytes (`\x00`) or non-UTF-8 sequences are skipped with reason `"binary content"` or `"unreadable UTF-8 source"`.

---

## CLI Integration

```bash
# Analyze current workspace and report graph summary
tst analyze .

# Query context for a specific symbol
tst analyze . --symbol run_route

# Output complete graph and index report as JSON
tst analyze . --json
```