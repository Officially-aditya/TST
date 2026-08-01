"""Content-hash based incremental repository indexing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .graph_builder import CodeGraph, GraphBuilder
from .parsers import ParserRegistry
from .scanner import ProjectScanner, ScanConfig


@dataclass(slots=True)
class IndexReport:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    parse_errors: dict[str, list[str]] = field(default_factory=dict)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    parsed_files: int = 0
    elapsed_ms: float = 0.0

    @property
    def changed_graph(self) -> bool:
        return bool(self.added or self.changed or self.removed)


class IncrementalIndexer:
    def __init__(
        self,
        root: str | Path,
        *,
        scan_config: ScanConfig | None = None,
        registry: ParserRegistry | None = None,
        project_name: str | None = None,
    ) -> None:
        self.scanner = ProjectScanner(root, scan_config)
        self.registry = registry or ParserRegistry()
        self.builder = GraphBuilder(project_name or self.scanner.root.name)
        self.hashes: dict[str, str] = {}

    @property
    def graph(self) -> CodeGraph:
        return self.builder.graph

    def index(self, include_paths: set[str] | None = None) -> IndexReport:
        started = time.perf_counter()
        report = IndexReport()
        sources = {source.relative_path: source for source in self.scanner.scan(include_paths)}
        report.skipped = list(self.scanner.skipped)

        known_scope = (
            set(self.hashes)
            if include_paths is None
            else set(self.hashes) & {Path(path).as_posix() for path in include_paths}
        )
        report.removed = sorted(known_scope - set(sources))
        parsed_updates = {}

        for path, source in sorted(sources.items()):
            previous = self.hashes.get(path)
            if previous == source.content_hash:
                report.unchanged.append(path)
                continue
            if previous is None:
                report.added.append(path)
            else:
                report.changed.append(path)
            try:
                parsed = self.registry.parse(source)
            except Exception as exc:
                report.parse_errors[path] = [f"parser failed safely: {type(exc).__name__}: {exc}"]
                continue
            report.parsed_files += 1
            if parsed.errors:
                report.parse_errors[path] = list(parsed.errors)
            if parsed.failed:
                continue
            parsed_updates[path] = parsed

        if report.removed or parsed_updates:
            desired = dict(self.builder.parsed_files)
            for path in report.removed:
                desired.pop(path, None)
            desired.update(parsed_updates)

            staged = GraphBuilder(self.builder.project_name)
            try:
                for parsed in sorted(desired.values(), key=lambda item: item.path):
                    staged.add_parsed_file(parsed)
                staged.resolve()
            except Exception as exc:
                report.parse_errors["<graph>"] = [
                    f"graph update failed safely: {type(exc).__name__}: {exc}"
                ]
            else:
                self.builder = staged
                for path in report.removed:
                    self.hashes.pop(path, None)
                for path in parsed_updates:
                    self.hashes[path] = sources[path].content_hash
        report.elapsed_ms = (time.perf_counter() - started) * 1_000
        return report

    def clear(self) -> None:
        self.builder = GraphBuilder(self.builder.project_name)
        self.hashes.clear()
