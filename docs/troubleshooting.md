# Troubleshooting

Start with `tst doctor`; use `tst doctor --json` in bug reports or automated
setup checks.

## Kernel binary not found

Normal startup never runs Cargo. Build explicitly with `tst kernel build`, or
point `TST_KERNEL_BIN` at a compatible v0.2 server. The default source-checkout
path is `tst_memory/target/release/server`.

## Kernel starts but never becomes ready

`READY` is emitted only after persistence recovery. Inspect captured kernel
stderr and verify that the snapshot directory is writable. A corrupt latest
snapshot should be preserved while the previous snapshot is tried. Do not
delete either snapshot until diagnostic copies have been made.

## Chat imports Torch or Transformers errors

The core installation deliberately excludes model runtimes. Install with
`python -m pip install -e '.[models]'`. Model downloads and accelerator support
are environment-specific; `tst analyze`, `tst doctor` and kernel commands do
not require this extra.

## JS, TS or Rust reports `structural-fallback`

Install `python -m pip install -e '.[analysis]'`. If native grammar wheels are
not available for the current Python/platform combination, Python AST analysis
still works and the other languages retain conservative structural indexing.
The parser name is included in each file node's metadata.

## A source file is missing from analysis

Run `tst analyze PATH --json` and inspect `report.skipped`. Files are omitted
when they are outside the selected root, symlinks, hidden or dependency/build
content, potential secrets, binary/non-UTF-8 data, unsupported suffixes, or
over configured size/count limits.

## A review finding was not displayed

Inspect the structured validation result in debug mode. Findings are withheld
for invalid JSON/schema, paths outside the project, missing files or line
ranges, unknown related symbols, duplicate identity or confidence below the
display threshold. Raw worker output is absent by default and retained only in
explicit, size-bounded debug mode. It should not be written to ordinary logs
because it can contain source or memory context.
