# Repository scanner and code graph

`tst.analysis.IncrementalIndexer` turns a selected source root into a
line-aware graph without importing or executing project code.

## Parser selection

| Language | Primary parser | Dependency-light behavior |
|---|---|---|
| Python | Tree-sitter | standard-library `ast` |
| JavaScript/JSX | Tree-sitter | structural fallback |
| TypeScript/TSX | Tree-sitter | structural fallback |
| Rust | Tree-sitter | structural fallback |

Tree-sitter grammars are loaded lazily from the `analysis` extra. A core-only
install retains precise Python AST parsing. Parser
results contain qualified names, one-based start/end lines, signatures,
imports, inheritance names and calls. Syntax errors are recorded in the index
report instead of crashing or causing source execution.

## Nodes and edges

Project, file and symbol nodes have deterministic identities derived from
their path and qualified name. Symbol nodes retain file path, source range,
signature and file content hash. Supported edges are:

- `contains` and `defines` for hierarchy;
- `imports` for local or unresolved external modules/symbols;
- `calls` and `references` for symbol use;
- `inherits` and `implements` for type relationships;
- `tests` from test modules to imported production files or symbols.

Resolution normalizes Python relative imports, JS/TS relative modules and
common Rust `crate`, `self` and `super` paths. Ambiguous global name matches
are not promoted to high-confidence edges. Unresolved dependencies remain
explicit external nodes.

`CodeGraph.query()` begins at a matching file or symbol and walks callers,
callees, imports, types, tests, parents and children. Both node count and an
estimated token budget are enforced. Removing a node removes all incoming and
outgoing edges, so deleted files cannot leave dangling references.

## Incremental behavior

The scanner computes SHA-256 for each accepted file. A subsequent scan:

- skips parsing when the path and hash are unchanged;
- stages and transactionally rebuilds a changed file subtree;
- removes deleted subtrees and all connected edges;
- adds new files normally;
- reruns local symbol resolution only after a graph-changing scan.

If a changed file fails parsing or graph resolution, the previous valid graph
and content hash remain active so a transient syntax error cannot erase useful
repository context.

Tree graph persistence is intentionally outside v0.2 scope.

## Safety limits

Scanning is rooted at a resolved directory and rejects traversal outside it.
It does not follow directory symlinks and skips all file symlinks by default.
It also skips VCS metadata, dependencies, caches, build output, hidden
directories, common secret filenames, binary content and non-UTF-8 source.
Per-file, repository-byte and file-count limits are configurable through
`ScanConfig`.

The scanner is an indexing boundary, not a sandbox for executing code. TST
never evaluates imports, invokes build hooks or runs analyzed files.
