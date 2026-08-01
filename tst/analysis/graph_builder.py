"""Line-aware repository graph construction and budgeted retrieval."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from .models import ParsedFile


class EdgeKind(str, Enum):
    CONTAINS = "contains"
    IMPORTS = "imports"
    CALLS = "calls"
    REFERENCES = "references"
    DEFINES = "defines"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    TESTS = "tests"


@dataclass(slots=True)
class GraphNode:
    node_id: int
    node_type: str
    name: str
    qualified_name: str
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    signature: str | None = None
    content_hash: str | None = None
    parent: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdge:
    source: int
    target: int
    kind: EdgeKind
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[int, int, EdgeKind]:
        return (self.source, self.target, self.kind)


@dataclass(slots=True)
class GraphSlice:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool = False
    estimated_tokens: int = 0


class CodeGraph:
    """An in-memory graph with referential integrity on every mutation."""

    _EDGE_PRIORITY = {
        EdgeKind.CALLS: 0,
        EdgeKind.TESTS: 1,
        EdgeKind.IMPORTS: 2,
        EdgeKind.INHERITS: 3,
        EdgeKind.IMPLEMENTS: 3,
        EdgeKind.REFERENCES: 4,
        EdgeKind.CONTAINS: 5,
        EdgeKind.DEFINES: 6,
    }

    def __init__(self) -> None:
        self.nodes: dict[int, GraphNode] = {}
        self._edges: dict[tuple[int, int, EdgeKind], GraphEdge] = {}

    @property
    def edges(self) -> list[GraphEdge]:
        return list(self._edges.values())

    def add_node(self, node: GraphNode) -> None:
        existing = self.nodes.get(node.node_id)
        if existing is not None and (
            existing.node_type,
            existing.qualified_name,
            existing.file_path,
        ) != (node.node_type, node.qualified_name, node.file_path):
            raise ValueError(f"graph node ID collision for {node.node_id}")
        if node.parent is not None and node.parent not in self.nodes:
            raise KeyError(f"missing parent node {node.parent}")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise KeyError("both graph edge endpoints must exist")
        if not 0.0 <= edge.confidence <= 1.0:
            raise ValueError("edge confidence must be between zero and one")
        existing = self._edges.get(edge.key)
        if existing is None or edge.confidence > existing.confidence:
            self._edges[edge.key] = edge

    def remove_edges(self, predicate: Callable[[GraphEdge], bool]) -> int:
        keys = [key for key, edge in self._edges.items() if predicate(edge)]
        for key in keys:
            del self._edges[key]
        return len(keys)

    def descendants(self, node_id: int) -> set[int]:
        found: set[int] = set()
        pending = [node_id]
        while pending:
            parent = pending.pop()
            children = [node.node_id for node in self.nodes.values() if node.parent == parent]
            for child in children:
                if child not in found:
                    found.add(child)
                    pending.append(child)
        return found

    def remove_node(self, node_id: int, *, cascade: bool = True) -> set[int]:
        if node_id not in self.nodes:
            return set()
        removing = {node_id}
        if cascade:
            removing.update(self.descendants(node_id))
        elif any(node.parent == node_id for node in self.nodes.values()):
            raise ValueError("cannot remove a graph node that still owns children")
        self.remove_edges(lambda edge: edge.source in removing or edge.target in removing)
        for target in removing:
            self.nodes.pop(target, None)
        return removing

    def incoming(self, node_id: int, kinds: set[EdgeKind] | None = None) -> list[GraphEdge]:
        return [
            edge
            for edge in self._edges.values()
            if edge.target == node_id and (kinds is None or edge.kind in kinds)
        ]

    def outgoing(self, node_id: int, kinds: set[EdgeKind] | None = None) -> list[GraphEdge]:
        return [
            edge
            for edge in self._edges.values()
            if edge.source == node_id and (kinds is None or edge.kind in kinds)
        ]

    def find(self, query: str, *, file_path: str | None = None) -> list[GraphNode]:
        needle = query.casefold().strip()
        scored: list[tuple[int, int, GraphNode]] = []
        for node in self.nodes.values():
            if file_path is not None and node.file_path != file_path:
                continue
            name = node.name.casefold()
            qualified = node.qualified_name.casefold()
            if qualified == needle:
                score = 0
            elif name == needle:
                score = 1
            elif qualified.endswith(f".{needle}"):
                score = 2
            elif needle in qualified:
                score = 3
            else:
                continue
            scored.append((score, len(qualified), node))
        return [
            item[2] for item in sorted(scored, key=lambda item: (item[0], item[1], item[2].node_id))
        ]

    def query(
        self,
        start: int | str,
        *,
        depth: int = 1,
        max_nodes: int = 40,
        token_budget: int = 2_000,
    ) -> GraphSlice:
        if isinstance(start, str):
            matches = self.find(start)
            if not matches:
                return GraphSlice([], [])
            start_id = matches[0].node_id
        else:
            start_id = start
        if start_id not in self.nodes:
            return GraphSlice([], [])
        if depth < 0 or max_nodes <= 0 or token_budget <= 0:
            raise ValueError("graph query budgets must be positive")

        selected: list[GraphNode] = []
        visited: set[int] = set()
        pending: deque[tuple[int, int]] = deque([(start_id, 0)])
        estimated_tokens = 0
        truncated = False

        while pending:
            node_id, current_depth = pending.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = self.nodes[node_id]
            cost = max(1, (len(node.qualified_name) + len(node.signature or "")) // 4)
            if selected and (len(selected) >= max_nodes or estimated_tokens + cost > token_budget):
                truncated = True
                continue
            selected.append(node)
            estimated_tokens += cost
            if current_depth >= depth:
                continue

            connected = self.incoming(node_id) + self.outgoing(node_id)
            connected.sort(
                key=lambda edge: (
                    self._EDGE_PRIORITY.get(edge.kind, 99),
                    -edge.confidence,
                    edge.source,
                    edge.target,
                )
            )
            neighbor_ids: list[int] = []
            if node.parent is not None:
                neighbor_ids.append(node.parent)
            neighbor_ids.extend(
                edge.source if edge.target == node_id else edge.target for edge in connected
            )
            neighbor_ids.extend(
                child.node_id for child in self.nodes.values() if child.parent == node_id
            )
            for neighbor in dict.fromkeys(neighbor_ids):
                if neighbor not in visited:
                    pending.append((neighbor, current_depth + 1))

        selected_ids = {node.node_id for node in selected}
        edges = [
            edge
            for edge in self._edges.values()
            if edge.source in selected_ids and edge.target in selected_ids
        ]
        return GraphSlice(selected, edges, truncated, estimated_tokens)

    def validate(self) -> list[str]:
        errors: list[str] = []
        for node in self.nodes.values():
            if node.parent is not None and node.parent not in self.nodes:
                errors.append(f"node {node.node_id} has missing parent {node.parent}")
        for edge in self._edges.values():
            if edge.source not in self.nodes:
                errors.append(f"edge {edge.key} has missing source")
            if edge.target not in self.nodes:
                errors.append(f"edge {edge.key} has missing target")
        return errors


def module_name(path: str) -> str:
    pure = PurePosixPath(path)
    parts = list(pure.with_suffix("").parts)
    if parts and parts[-1] in {"__init__", "mod", "index"}:
        parts.pop()
    return ".".join(parts) or pure.stem


def _stable_id(identity: str) -> int:
    value = int.from_bytes(hashlib.blake2b(identity.encode("utf-8"), digest_size=8).digest(), "big")
    return value or 1


class GraphBuilder:
    def __init__(self, project_name: str) -> None:
        self.graph = CodeGraph()
        self.project_name = project_name
        self.project_id = self._make_id(f"project:{project_name}")
        self.graph.add_node(
            GraphNode(
                node_id=self.project_id,
                node_type="project",
                name=project_name,
                qualified_name=project_name,
            )
        )
        self.parsed_files: dict[str, ParsedFile] = {}
        self.file_nodes: dict[str, int] = {}
        self.symbol_nodes: dict[tuple[str, str], int] = {}

    def _make_id(self, identity: str) -> int:
        candidate = _stable_id(identity)
        salt = 0
        while candidate in self.graph.nodes:
            salt += 1
            candidate = _stable_id(f"{identity}:{salt}")
        return candidate

    def add_parsed_file(self, parsed: ParsedFile) -> int:
        if parsed.path in self.file_nodes:
            self.remove_file(parsed.path)
        file_qname = module_name(parsed.path)
        pure_path = PurePosixPath(parsed.path)
        if parsed.language == "rust" and pure_path.stem in {"lib", "main"}:
            file_qname = ".".join(pure_path.parent.parts) or pure_path.stem
        file_id = self._make_id(f"file:{parsed.path}")
        file_node = GraphNode(
            node_id=file_id,
            node_type="file",
            name=PurePosixPath(parsed.path).name,
            qualified_name=file_qname,
            file_path=parsed.path,
            content_hash=parsed.content_hash,
            parent=self.project_id,
            metadata={
                "language": parsed.language,
                "parser": parsed.parser,
                "parse_errors": list(parsed.errors),
            },
        )
        self.graph.add_node(file_node)
        self.graph.add_edge(GraphEdge(self.project_id, file_id, EdgeKind.CONTAINS))
        self.parsed_files[parsed.path] = parsed
        self.file_nodes[parsed.path] = file_id

        local_nodes: dict[str, int] = {}
        ordered = sorted(
            parsed.symbols, key=lambda item: (item.qualified_name.count("."), item.span.start_line)
        )
        for symbol in ordered:
            parent_id = local_nodes.get(symbol.parent_qualified_name or "", file_id)
            qualified = (
                f"{file_qname}.{symbol.qualified_name}" if file_qname else symbol.qualified_name
            )
            node_id = self._make_id(f"symbol:{parsed.path}:{qualified}:{symbol.kind.value}")
            node = GraphNode(
                node_id=node_id,
                node_type=symbol.kind.value,
                name=symbol.name,
                qualified_name=qualified,
                file_path=parsed.path,
                start_line=symbol.span.start_line,
                end_line=symbol.span.end_line,
                signature=symbol.signature,
                content_hash=symbol.content_hash or parsed.content_hash,
                parent=parent_id,
                metadata={
                    **symbol.metadata,
                    "local_qualified_name": symbol.qualified_name,
                    "bases": list(symbol.bases),
                    "implements": list(symbol.implements),
                    "references": [
                        {"name": ref.name, "kind": ref.kind.value, "line": ref.line}
                        for ref in symbol.references
                    ],
                },
            )
            self.graph.add_node(node)
            self.graph.add_edge(GraphEdge(parent_id, node_id, EdgeKind.CONTAINS))
            self.graph.add_edge(GraphEdge(file_id, node_id, EdgeKind.DEFINES))
            local_nodes[symbol.qualified_name] = node_id
            self.symbol_nodes[(parsed.path, symbol.qualified_name)] = node_id
        return file_id

    def remove_file(self, path: str) -> bool:
        node_id = self.file_nodes.pop(path, None)
        if node_id is None:
            return False
        removing = self.graph.remove_node(node_id, cascade=True)
        self.parsed_files.pop(path, None)
        self.symbol_nodes = {
            key: value for key, value in self.symbol_nodes.items() if value not in removing
        }
        return True

    def external_node(self, name: str, *, kind: str = "external") -> int:
        qualified = f"external:{name}"
        matches = [
            node.node_id
            for node in self.graph.nodes.values()
            if node.node_type == kind and node.qualified_name == qualified
        ]
        if matches:
            return matches[0]
        node_id = self._make_id(f"{kind}:{name}")
        self.graph.add_node(
            GraphNode(
                node_id=node_id,
                node_type=kind,
                name=name,
                qualified_name=qualified,
                parent=self.project_id,
                metadata={"external": True},
            )
        )
        self.graph.add_edge(GraphEdge(self.project_id, node_id, EdgeKind.CONTAINS))
        return node_id

    def resolve(self) -> None:
        from .resolver import SymbolResolver

        SymbolResolver(self).resolve()
