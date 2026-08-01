# Structured code-review validation

Conversational text and code-review findings have different safety needs. A
review worker must generate a JSON object matching `CodeReviewOutput`; prose
surrounding the object is treated as a parse failure.

Each `CodeIssue` requires:

- a root-relative file path and ordered one-based line range;
- the current source SHA-256 when emitted by `ReviewWorker`;
- severity and category enums;
- a title, explanation and suggested fix;
- confidence between zero and one;
- optional related symbol names.

`ReviewValidator` retains raw output only in explicit, size-bounded debug mode and verifies
that the file resolves inside the selected project, exists and is readable,
and that its line range exists. When a code graph is supplied, related symbols
must exist in the named file. Duplicate findings are rejected. Findings below
the configured confidence threshold are returned separately as flagged and
are not included in `safe_to_display`.

`ReviewWorker` always sends `do_sample=false`, `temperature=0`, and the JSON
schema to its injected generation adapter. Accepted generated findings must
carry the current content hash, preventing stale model output from being shown
after a file changes. Worker counters expose generation/parse failures and
accepted, rejected, and low-confidence issue counts.

Parse failures and rejected findings are data, not exceptions that can fall
through to unvalidated display. Model output cannot become executable edits;
automatic fix execution is outside v0.2 scope.
