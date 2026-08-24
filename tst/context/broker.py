"""Context retrieval across global, project, session, and code-graph layers."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tst.analysis import IncrementalIndexer
from tst.analysis.graph_builder import CodeGraph, GraphNode
from tst.memory.retrieval import memory_record_from_payload
from tst.scope.broker import layer_for_scope, scope_prefix
from tst.scope.kernel_manager import KernelManager
from tst.scope.models import Project, Scope
from tst.scope.registry import ProjectRegistry

from .explain import estimate_tokens, memory_reason, tree_reason
from .models import ContextBudget, ContextItem, ContextPack
from .ranking import ContextRanker

IndexerFactory = Callable[[Project], IncrementalIndexer]


class ContextBroker:
    """Merge existing kernel retrieval and repository graph intelligence."""

    def __init__(
        self,
        kernel_manager: KernelManager,
        *,
        registry: ProjectRegistry | None = None,
        indexer_factory: IndexerFactory | None = None,
        ranker: ContextRanker | None = None,
    ) -> None:
        self.kernel_manager = kernel_manager
        self.registry = registry or kernel_manager.registry
        self.indexer_factory = indexer_factory or (
            lambda project: IncrementalIndexer(project.root, project_name=project.name)
        )
        self.ranker = ranker or ContextRanker()
        self._indexers: dict[str, IncrementalIndexer] = {}
        self._tree_sync_tokens: dict[str, int] = {}

    def retrieve(
        self,
        query: str,
        project: Project | str | Path,
        session_id: str,
        budget: ContextBudget | int | None = None,
    ) -> ContextPack:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("context query cannot be empty")
        selected_project = self._resolve_project(project)
        limits = ContextBudget.coerce(budget)
        self.index_project(selected_project)

        candidates: list[ContextItem] = []
        global_client = self.kernel_manager.global_client()
        candidates.extend(
            self._memory_items(
                global_client,
                query,
                Scope.GLOBAL,
                selected_project,
                session_id,
            )
        )
        project_client = self.kernel_manager.project_client(selected_project)
        candidates.extend(
            self._memory_items(
                project_client,
                query,
                Scope.PROJECT,
                selected_project,
                session_id,
            )
        )
        candidates.extend(
            self._memory_items(
                project_client,
                query,
                Scope.SESSION,
                selected_project,
                session_id,
            )
        )
        candidates.extend(self._tree_items(query, selected_project, project_client))
        items = self.ranker.select(candidates, budget=limits)
        return ContextPack(
            query=query,
            project=selected_project.name,
            items=items,
            estimated_tokens=sum(estimate_tokens(item.content) for item in items),
        )

    def index_project(self, project: Project | str | Path) -> Any:
        selected_project = self._resolve_project(project)
        indexer = self._indexer(selected_project)
        report = indexer.index()
        client = self.kernel_manager.project_client(selected_project)
        token = client.pid if client.pid is not None else id(client)
        if self._tree_sync_tokens.get(selected_project.id) != token:
            self._sync_tree(selected_project, indexer.graph)
            self._tree_sync_tokens[selected_project.id] = token
        elif report.changed_graph:
            self._sync_tree(selected_project, indexer.graph)
        return report

    def graph(self, project: Project | str | Path) -> CodeGraph:
        selected_project = self._resolve_project(project)
        self.index_project(selected_project)
        return self._indexer(selected_project).graph

    def _memory_items(
        self,
        client: Any,
        query: str,
        scope: Scope,
        project: Project,
        session_id: str,
    ) -> list[ContextItem]:
        layer = layer_for_scope(scope)
        prefix = scope_prefix(scope, project, session_id)
        result = client.search(query, layer=layer, prefix=prefix, limit=100)
        matches = result.get("matches", []) if isinstance(result, dict) else []
        items: list[ContextItem] = []
        for match in matches:
            if not isinstance(match, dict):
                continue
            key = str(match.get("key", ""))
            payload = match.get("payload")
            record = memory_record_from_payload(payload, key=key, layer=layer)
            content = record.value if record is not None else _payload_value(payload)
            if not key or not content:
                continue
            raw_score = match.get("score", 0.0)
            try:
                score = max(0.0, min(float(raw_score), 1.0))
            except (TypeError, ValueError):
                score = 0.35
            if score == 0.0:
                score = 0.35
            items.append(
                ContextItem(
                    source="memory",
                    scope=scope,
                    key=key,
                    content=content,
                    score=score,
                    reason=memory_reason(query, key, content),
                    layer=layer,
                    metadata={
                        "memory_type": record.memory_type if record else "unknown",
                        "tags": record.tags if record else [],
                    },
                )
            )
        return items

    def _tree_items(self, query: str, project: Project, client: Any) -> list[ContextItem]:
        indexer = self._indexer(project)
        graph = indexer.graph
        terms = [query.strip(), *_query_terms(query)]
        found: dict[int, tuple[GraphNode, float, str]] = {}
        for term in dict.fromkeys(term for term in terms if term):
            for position, node in enumerate(graph.find(term)):
                score = max(0.55, 0.96 - position * 0.08)
                reason = tree_reason(term, node.qualified_name)
                previous = found.get(node.node_id)
                if previous is None or score > previous[1]:
                    found[node.node_id] = (node, score, reason)
                if len(found) >= 24:
                    break
        items: list[ContextItem] = []
        for node, score, reason in found.values():
            if node.node_type == "project":
                continue
            items.append(self._node_item(node, project, score, reason))

            # One bounded relationship hop makes the provenance useful without
            # turning a context request into a full graph dump.
            related = graph.query(node.node_id, depth=1, max_nodes=8, token_budget=500)
            for related_node in related.nodes[1:]:
                if related_node.node_id == node.node_id or related_node.node_type == "project":
                    continue
                if any(item.symbol == related_node.qualified_name for item in items):
                    continue
                items.append(
                    self._node_item(
                        related_node,
                        project,
                        max(0.45, score - 0.16),
                        tree_reason(query, related_node.qualified_name, related=True),
                    )
                )
        if not items:
            try:
                for node in client.tree_find(query, limit=10):
                    if isinstance(node, dict):
                        items.append(self._wire_node_item(node))
            except Exception:
                # Graph retrieval remains useful when a kernel has no tree yet.
                pass
        return items

    @staticmethod
    def _node_item(node: GraphNode, project: Project, score: float, reason: str) -> ContextItem:
        content = " ".join(
            part
            for part in [
                f"{node.node_type} {node.qualified_name}",
                node.signature or "",
                f"defined at {node.file_path}:{node.start_line}"
                if node.file_path and node.start_line
                else "",
            ]
            if part
        )
        return ContextItem(
            source="tree",
            scope=Scope.PROJECT,
            content=content,
            score=max(0.0, min(score, 1.0)),
            reason=reason,
            symbol=node.qualified_name,
            file=node.file_path,
            metadata={"node_id": node.node_id, "node_type": node.node_type},
        )

    @staticmethod
    def _wire_node_item(node: dict[str, Any]) -> ContextItem:
        file_path = node.get("file_path")
        qualified = str(node.get("qualified_name") or node.get("name") or "node")
        return ContextItem(
            source="tree",
            scope=Scope.PROJECT,
            content=f"{node.get('node_type', 'node')} {qualified}",
            score=0.72,
            reason="symbol_match",
            symbol=qualified,
            file=str(file_path) if file_path else None,
            metadata={"node_id": node.get("node_id")},
        )

    def _sync_tree(self, project: Project, graph: CodeGraph) -> None:
        client = self.kernel_manager.project_client(project)
        client.tree_clear()
        mapping: dict[int, int] = {}
        pending = set(graph.nodes)
        while pending:
            progressed = False
            for node_id in sorted(pending):
                node = graph.nodes[node_id]
                if node.parent is not None and node.parent not in mapping:
                    continue
                parent = mapping.get(node.parent)
                kwargs: dict[str, Any] = {
                    "qualified_name": node.qualified_name,
                    "file_path": node.file_path,
                    "start_line": node.start_line,
                    "end_line": node.end_line,
                    "signature": node.signature,
                    "content_hash": node.content_hash,
                }
                mapping[node_id] = client.tree_insert(
                    _kernel_node_type(node.node_type),
                    node.name,
                    parent,
                    **{key: value for key, value in kwargs.items() if value is not None},
                )
                pending.remove(node_id)
                progressed = True
            if not progressed:
                raise ValueError("code graph contains an unresolved parent cycle")
        for edge in graph.edges:
            if edge.source in mapping and edge.target in mapping:
                client.tree_link(
                    mapping[edge.source],
                    mapping[edge.target],
                    edge_type=edge.kind.value,
                    confidence=edge.confidence,
                )

    def _indexer(self, project: Project) -> IncrementalIndexer:
        if project.id not in self._indexers:
            self._indexers[project.id] = self.indexer_factory(project)
        return self._indexers[project.id]

    def _resolve_project(self, project: Project | str | Path) -> Project:
        if isinstance(project, Project):
            return project
        found = self.registry.find(project)
        if found is not None:
            return found
        return self.registry.ensure(project)


def _query_terms(query: str) -> list[str]:
    split = re.sub(r"([a-z])([A-Z])", r"\1 \2", query)
    return [term for term in re.findall(r"[A-Za-z_][A-Za-z0-9_+#.-]*", split) if len(term) > 2]


def _payload_value(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return str(payload) if payload is not None else None
    data = payload.get("data", payload)
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        data = data["data"]
    if isinstance(data, dict):
        for name in ("MemoryRecord", "memory_record", "Preference", "preference"):
            candidate = data.get(name)
            if isinstance(candidate, dict) and candidate.get("value") is not None:
                return str(candidate["value"])
        if data.get("value") is not None:
            return str(data["value"])
    return None


def _kernel_node_type(value: str) -> str:
    """Collapse parser-specific kinds into the Rust tree's stable enum."""

    return {
        "method": "function",
        "variable": "symbol",
    }.get(value.casefold(), value)
