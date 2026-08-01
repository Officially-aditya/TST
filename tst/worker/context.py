"""Bounded, source-grounded context assembly for code review."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tst.analysis.graph_builder import CodeGraph
from tst.analysis.scanner import resolve_project_path

from .schemas import CodeReviewOutput


@dataclass(frozen=True, slots=True)
class ReviewContext:
    prompt: str
    target_path: str
    content_hash: str
    line_count: int


class ReviewContextBuilder:
    def __init__(
        self,
        project_root: str | Path,
        *,
        graph: CodeGraph | None = None,
        max_prompt_characters: int = 80_000,
        max_target_characters: int = 40_000,
        max_related_files: int = 8,
        max_related_characters: int = 20_000,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve(strict=True)
        if not self.project_root.is_dir():
            raise ValueError("review project root must be a directory")
        if (
            min(
                max_prompt_characters,
                max_target_characters,
                max_related_files,
                max_related_characters,
            )
            <= 0
        ):
            raise ValueError("review context budgets must be positive")
        self.graph = graph
        self.max_prompt_characters = max_prompt_characters
        self.max_target_characters = max_target_characters
        self.max_related_files = max_related_files
        self.max_related_characters = max_related_characters

    def build(
        self,
        task: str,
        target_file: str | Path,
        *,
        related_files: tuple[str | Path, ...] = (),
        conventions: tuple[str, ...] = (),
    ) -> ReviewContext:
        target_path = resolve_project_path(self.project_root, target_file)
        if not target_path.is_file():
            raise ValueError("review target must be a file")
        relative = target_path.relative_to(self.project_root).as_posix()
        target_bytes = target_path.read_bytes()
        target_text = target_bytes.decode("utf-8")
        content_hash = hashlib.sha256(target_bytes).hexdigest()
        line_count = len(target_text.splitlines())

        sections = [
            self._section("Task", task.strip()),
            self._section(
                "Target code",
                self._source_block(
                    relative,
                    content_hash,
                    target_text,
                    self.max_target_characters,
                ),
            ),
            self._section("Relevant symbol graph", self._graph_context(relative)),
            self._section(
                "Related source excerpts",
                self._related_context(related_files),
            ),
            self._section(
                "Retrieved project conventions",
                self._convention_context(conventions),
            ),
            self._section(
                "Output schema",
                json.dumps(CodeReviewOutput.model_json_schema(), sort_keys=True, indent=2),
            ),
        ]
        prompt = "\n\n".join(sections)
        if len(prompt) > self.max_prompt_characters:
            raise ValueError(
                "assembled review context exceeds max_prompt_characters; reduce source budgets"
            )
        return ReviewContext(prompt, relative, content_hash, line_count)

    @staticmethod
    def _section(name: str, content: str) -> str:
        return f"## {name}\n{content or '(none)'}"

    @staticmethod
    def _bounded(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        marker = "\n... [truncated by context budget]"
        return value[: max(0, limit - len(marker))] + marker

    def _source_block(self, path: str, content_hash: str, text: str, limit: int) -> str:
        numbered = "\n".join(
            f"{line_number:>6} | {line}"
            for line_number, line in enumerate(text.splitlines(), start=1)
        )
        body = self._bounded(numbered, limit)
        return f"path: {path}\nsha256: {content_hash}\n```text\n{body}\n```"

    def _graph_context(self, relative: str) -> str:
        if self.graph is None:
            return "(none)"
        candidates = sorted(
            (node for node in self.graph.nodes.values() if node.file_path == relative),
            key=lambda node: (node.start_line or 0, node.qualified_name, node.node_id),
        )
        if not candidates:
            return "(none)"
        start = next((node for node in candidates if node.node_type == "file"), candidates[0])
        graph_slice = self.graph.query(start.node_id, depth=1, max_nodes=40, token_budget=2_000)
        document = {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "qualified_name": node.qualified_name,
                    "file_path": node.file_path,
                    "start_line": node.start_line,
                    "end_line": node.end_line,
                }
                for node in graph_slice.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "kind": edge.kind.value,
                    "confidence": edge.confidence,
                }
                for edge in graph_slice.edges
            ],
            "truncated": graph_slice.truncated,
        }
        return self._bounded(json.dumps(document, sort_keys=True, indent=2), 12_000)

    def _related_context(self, related_files: tuple[str | Path, ...]) -> str:
        blocks: list[str] = []
        remaining = self.max_related_characters
        seen: set[str] = set()
        for requested in sorted(map(str, related_files))[: self.max_related_files]:
            path = resolve_project_path(self.project_root, requested)
            relative = path.relative_to(self.project_root).as_posix()
            if relative in seen or not path.is_file():
                continue
            seen.add(relative)
            text = path.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            block = self._source_block(relative, content_hash, text, min(8_000, remaining))
            if len(block) > remaining:
                break
            blocks.append(block)
            remaining -= len(block)
        return "\n\n".join(blocks) or "(none)"

    @staticmethod
    def _convention_context(conventions: tuple[str, ...]) -> str:
        cleaned = sorted({item.strip() for item in conventions if item.strip()})
        return "\n".join(f"- {item}" for item in cleaned) or "(none)"


__all__ = ["ReviewContext", "ReviewContextBuilder"]
