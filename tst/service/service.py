"""Single application boundary used by CLI, HTTP, UI, and integrations."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from tst.context import ContextBroker, ContextBudget, ContextPack
from tst.kernel.process import KernelProcessConfig
from tst.memory.keys import CanonicalKeyBuilder, InvalidCanonicalKey
from tst.memory.planner import MemoryRecord
from tst.memory.retrieval import memory_record_from_payload
from tst.scope.broker import (
    key_for_scope,
    layer_for_scope,
    remap_key,
    scope_for_key,
    scope_prefix,
)
from tst.scope.kernel_manager import KernelManager
from tst.scope.models import Project, Scope
from tst.scope.project import find_project_root
from tst.scope.registry import ProjectRegistry

from .errors import (
    IntegrationError,
    MemoryNotFoundError,
    ProjectNotFoundError,
    ScopeMovementError,
)
from .events import ActivityEvent, EventBus
from .models import IntegrationStatus, MemoryMatch


class TSTService:
    """Own project state and expose model-neutral control-plane operations."""

    def __init__(
        self,
        project: Project | str | Path | None = None,
        *,
        registry: ProjectRegistry | None = None,
        kernel_manager: KernelManager | None = None,
        context_broker: ContextBroker | None = None,
        session_id: str | None = None,
        actor: str = "TST",
        event_bus: EventBus | None = None,
    ) -> None:
        self.registry = registry or ProjectRegistry()
        self.project = self._resolve_or_create_project(project)
        self.session_id = session_id or uuid4().hex
        self.actor = actor
        self.events = event_bus or EventBus()
        self.kernel_manager = kernel_manager or KernelManager(self.registry)
        self.context_broker = context_broker or ContextBroker(
            self.kernel_manager,
            registry=self.registry,
        )
        self._closed = False

    def status(self) -> dict[str, Any]:
        """Return local health and scope counts without exposing memory values."""

        started = time.perf_counter()
        request_id = uuid4().hex
        counts: dict[str, int] = {scope.value: 0 for scope in Scope}
        errors: list[str] = []
        for scope in (Scope.GLOBAL, Scope.PROJECT, Scope.SESSION):
            try:
                counts[scope.value] = len(self.list_memories(scope))
            except Exception as exc:
                errors.append(f"{scope.value}: {type(exc).__name__}")
        kernel_status = self.kernel_manager.status()
        result = {
            "project": self.project.to_dict(),
            "session_id": self.session_id,
            "kernel": kernel_status,
            "memory_counts": counts,
            "errors": errors,
            "healthy": not errors,
        }
        self._publish(
            "status",
            request_id,
            started,
            metadata={"healthy": result["healthy"], "error_count": len(errors)},
        )
        return result

    def projects(self) -> list[Project]:
        return self.registry.list()

    def switch_project(self, project: Project | str | Path) -> Project:
        selected = self.registry.find(project)
        if selected is None:
            candidate = Path(project).expanduser()
            if candidate.exists() or candidate.is_absolute():
                selected = self.registry.ensure(find_project_root(candidate))
            else:
                raise ProjectNotFoundError(f"project is not registered: {project}")
        started = time.perf_counter()
        request_id = uuid4().hex
        self.kernel_manager.project_client(selected)
        self.project = selected
        self._publish(
            "project.switch",
            request_id,
            started,
            scope=Scope.PROJECT,
            metadata={"project_id": selected.id, "project_name": selected.name},
        )
        return selected

    def retrieve_context(
        self,
        query: str,
        *,
        budget: ContextBudget | int | None = None,
        session_id: str | None = None,
    ) -> ContextPack:
        started = time.perf_counter()
        request_id = uuid4().hex
        pack = self.context_broker.retrieve(
            query,
            self.project,
            session_id or self.session_id,
            budget,
        )
        self._publish(
            "context.retrieve",
            request_id,
            started,
            scope=Scope.PROJECT,
            metadata={"item_count": len(pack.items), "estimated_tokens": pack.estimated_tokens},
        )
        return pack

    def context_preview(
        self,
        query: str,
        *,
        budget: ContextBudget | int | None = None,
    ) -> ContextPack:
        return self.retrieve_context(query, budget=budget)

    def list_memories(
        self,
        scope: Scope | str | None = None,
        *,
        limit: int = 1_000,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or limit > 1_000:
            raise ValueError("memory list limit must be between 1 and 1000")
        scopes = [Scope.coerce(scope)] if scope is not None else [Scope.SESSION, Scope.PROJECT, Scope.GLOBAL]
        records: list[dict[str, Any]] = []
        for selected in scopes:
            client = self._client_for_scope(selected)
            layer = layer_for_scope(selected)
            result = client.search(
                "",
                layer=layer,
                prefix=scope_prefix(selected, self.project, self.session_id),
                limit=limit,
            )
            for match in result.get("matches", []):
                if not isinstance(match, dict):
                    continue
                key = str(match.get("key", ""))
                record = memory_record_from_payload(match.get("payload"), key=key, layer=layer)
                if record is None:
                    continue
                records.append({**record.model_dump(), "scope": selected.value})
        return records[:limit]

    def search_memories(
        self,
        query: str,
        scope: Scope | str | None = None,
        *,
        limit: int = 20,
    ) -> list[MemoryMatch]:
        if not query.strip():
            raise ValueError("memory search query cannot be empty")
        if limit <= 0 or limit > 1_000:
            raise ValueError("memory search limit must be between 1 and 1000")
        scopes = [Scope.coerce(scope)] if scope is not None else [Scope.SESSION, Scope.PROJECT, Scope.GLOBAL]
        matches: list[MemoryMatch] = []
        for selected in scopes:
            client = self._client_for_scope(selected)
            layer = layer_for_scope(selected)
            result = client.search(
                query,
                layer=layer,
                prefix=scope_prefix(selected, self.project, self.session_id),
                limit=limit,
            )
            for match in result.get("matches", []):
                if not isinstance(match, dict):
                    continue
                key = str(match.get("key", ""))
                record = memory_record_from_payload(match.get("payload"), key=key, layer=layer)
                if record is None:
                    continue
                score = _bounded_score(match.get("score", 0.0))
                matches.append(
                    MemoryMatch(
                        key=key,
                        scope=selected,
                        layer=layer,
                        value=record.value,
                        score=score,
                        source_text=record.source_text,
                        tags=record.tags,
                    )
                )
        matches.sort(key=lambda item: (-item.score, item.scope.value, item.key))
        return matches[:limit]

    def store_memory(
        self,
        value: str,
        *,
        scope: Scope | str = Scope.PROJECT,
        key: str | None = None,
        subject: str | None = None,
        memory_type: str = "fact",
        source_text: str | None = None,
        confidence: float = 1.0,
        tags: Iterable[str] = (),
        actor: str | None = None,
    ) -> dict[str, Any]:
        selected = Scope.coerce(scope)
        memory_key = self._memory_key(selected, key, subject, memory_type)
        record = self._new_record(
            memory_key,
            value,
            memory_type=memory_type,
            source_text=source_text,
            confidence=confidence,
            tags=tags,
            scope=selected,
        )
        started = time.perf_counter()
        request_id = uuid4().hex
        kernel_result = self._client_for_scope(selected).store(
            layer_for_scope(selected),
            memory_key,
            _record_payload(record),
        )
        self._publish(
            "memory.store",
            request_id,
            started,
            actor=actor,
            scope=selected,
            metadata={"key": memory_key, "memory_type": memory_type},
        )
        return {
            "stored": True,
            "scope": selected.value,
            "key": memory_key,
            "record": record.model_dump(mode="json"),
            "kernel": kernel_result,
        }

    def update_memory(
        self,
        key: str,
        value: str,
        *,
        scope: Scope | str | None = None,
        memory_type: str | None = None,
        source_text: str | None = None,
        confidence: float | None = None,
        tags: Iterable[str] | None = None,
        actor: str | None = None,
    ) -> dict[str, Any]:
        selected = self._resolve_key_scope(key, scope)
        client = self._client_for_scope(selected)
        layer = layer_for_scope(selected)
        existing = self._get_record(client, layer, key)
        if existing is None:
            raise MemoryNotFoundError(f"memory does not exist: {key}")
        record = self._new_record(
            key,
            value,
            memory_type=memory_type or existing.memory_type,
            source_text=source_text or existing.source_text,
            confidence=existing.confidence if confidence is None else confidence,
            tags=existing.tags if tags is None else tags,
            scope=selected,
            created_at=existing.created_at,
        )
        started = time.perf_counter()
        request_id = uuid4().hex
        result = client.update(layer, key, _record_payload(record))
        self._publish(
            "memory.update",
            request_id,
            started,
            actor=actor,
            scope=selected,
            metadata={"key": key},
        )
        return {"updated": True, "scope": selected.value, "key": key, "record": record.model_dump(mode="json"), "kernel": result}

    def forget_memory(
        self,
        key: str,
        *,
        scope: Scope | str | None = None,
        actor: str | None = None,
    ) -> dict[str, Any]:
        selected = self._resolve_key_scope(key, scope)
        started = time.perf_counter()
        request_id = uuid4().hex
        result = self._client_for_scope(selected).delete(layer_for_scope(selected), key)
        self._publish(
            "memory.delete",
            request_id,
            started,
            actor=actor,
            scope=selected,
            metadata={"key": key},
        )
        return {"deleted": bool(result.get("deleted", False)), "scope": selected.value, "key": key, "kernel": result}

    def move_memory(
        self,
        key: str,
        target_scope: Scope | str,
        *,
        source_scope: Scope | str | None = None,
        actor: str | None = None,
    ) -> dict[str, Any]:
        target = Scope.coerce(target_scope)
        source = self._resolve_key_scope(key, source_scope)
        source_client = self._client_for_scope(source)
        source_layer = layer_for_scope(source)
        original = self._get_record(source_client, source_layer, key)
        if original is None:
            raise MemoryNotFoundError(f"memory does not exist: {key}")
        target_key = remap_key(key, target, self.project, self.session_id)
        target_layer = layer_for_scope(target)
        destination = original.model_copy(
            update={
                "key": target_key,
                "layer": target_layer,
                "updated_at": max(original.updated_at, int(time.time() * 1000)),
            }
        )
        started = time.perf_counter()
        request_id = uuid4().hex
        target_client = self._client_for_scope(target)
        try:
            target_client.store(target_layer, target_key, _record_payload(destination))
            verified = self._get_record(target_client, target_layer, target_key)
            if verified is None or verified.value != destination.value:
                raise ScopeMovementError("destination verification failed; source was preserved")
        except Exception as exc:
            if isinstance(exc, ScopeMovementError):
                raise
            raise ScopeMovementError(f"could not write destination scope: {exc}") from exc
        if source is not target or key != target_key:
            try:
                source_client.delete(source_layer, key)
            except Exception as exc:
                raise ScopeMovementError(
                    f"destination was verified but source cleanup failed: {exc}"
                ) from exc
        self._publish(
            "memory.move",
            request_id,
            started,
            actor=actor,
            scope=target,
            metadata={"source_scope": source.value, "target_scope": target.value, "key": key},
        )
        return {
            "moved": True,
            "source_scope": source.value,
            "target_scope": target.value,
            "source_key": key,
            "key": target_key,
            "record": destination.model_dump(mode="json"),
        }

    def find_symbol(self, name: str, *, file_path: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        graph = self.context_broker.graph(self.project)
        return [_graph_node_dict(node) for node in graph.find(name, file_path=file_path)[:limit]]

    def query_symbol(
        self,
        symbol: str | int,
        *,
        depth: int = 1,
        max_nodes: int = 40,
        token_budget: int = 2_000,
    ) -> dict[str, Any]:
        graph = self.context_broker.graph(self.project)
        result = graph.query(symbol, depth=depth, max_nodes=max_nodes, token_budget=token_budget)
        return {
            "nodes": [_graph_node_dict(node) for node in result.nodes],
            "edges": [_graph_edge_dict(edge) for edge in result.edges],
            "truncated": result.truncated,
            "estimated_tokens": result.estimated_tokens,
        }

    def index_project(self) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = uuid4().hex
        report = self.context_broker.index_project(self.project)
        document = asdict(report)
        self._publish(
            "project.index",
            request_id,
            started,
            scope=Scope.PROJECT,
            metadata={
                "added": len(report.added),
                "changed": len(report.changed),
                "removed": len(report.removed),
                "parsed_files": report.parsed_files,
            },
        )
        return document

    def integrations(self) -> list[IntegrationStatus]:
        root = self.project.root
        claude = root / ".claude" / "skills" / "tst-context" / "SKILL.md"
        codex = root / ".agents" / "skills" / "tst-context" / "SKILL.md"
        opencode = root / ".opencode" / "plugins" / "tst_context.js"
        return [
            IntegrationStatus(
                name="Claude Code",
                status="connected" if claude.is_file() else "available",
                location=str(claude.parent.parent) if claude.is_file() else str(root / ".claude"),
            ),
            IntegrationStatus(
                name="Codex",
                status="connected" if codex.is_file() else "available",
                location=str(codex.parent.parent) if codex.is_file() else str(root / ".agents"),
            ),
            IntegrationStatus(
                name="OpenCode",
                status="connected" if opencode.is_file() else "available",
                location=str(opencode.parent.parent) if opencode.is_file() else str(root / ".opencode"),
            ),
            IntegrationStatus(name="MCP", status="available", details={"command": "tst mcp serve"}),
            IntegrationStatus(name="Python SDK", status="available"),
        ]

    def install_integration(self, provider: str, *, force: bool = False) -> dict[str, Any]:
        selected = provider.strip().lower()
        if selected == "claude":
            from tst.integrations.claude import install
        elif selected == "codex":
            from tst.integrations.codex import install
        elif selected == "opencode":
            from tst.integrations.opencode import install
        else:
            raise IntegrationError(f"unsupported integration: {provider}")
        result = install(self.project.root, force=force)
        self._publish(
            f"integration.connect.{selected}",
            uuid4().hex,
            time.perf_counter(),
            metadata={"written": len(result)},
        )
        return result

    def uninstall_integration(self, provider: str, *, force: bool = False) -> dict[str, Any]:
        selected = provider.strip().lower()
        if selected == "claude":
            from tst.integrations.claude import uninstall
        elif selected == "codex":
            from tst.integrations.codex import uninstall
        elif selected == "opencode":
            from tst.integrations.opencode import uninstall
        else:
            raise IntegrationError(f"unsupported integration: {provider}")
        result = uninstall(self.project.root, force=force)
        self._publish(
            f"integration.disconnect.{selected}",
            uuid4().hex,
            time.perf_counter(),
            metadata={"removed": sum(status == "removed" for status in result.values())},
        )
        return result

    def doctor(self) -> dict[str, Any]:
        config = KernelProcessConfig()
        binary = config.resolved_binary()
        checks = [
            {"name": "python", "ok": sys.version_info >= (3, 10), "detail": sys.version.split()[0]},
            {"name": "kernel", "ok": binary.is_file(), "detail": str(binary)},
            {"name": "global-storage", "ok": self.registry.global_dir.is_dir(), "detail": str(self.registry.global_dir)},
            {"name": "project-storage", "ok": self.project.metadata_path.is_file(), "detail": str(self.project.metadata_path)},
        ]
        return {"ok": all(item["ok"] for item in checks), "checks": checks}

    def close(self) -> None:
        if not self._closed:
            self.kernel_manager.close()
            self._closed = True

    def subscribe_events(self):
        return self.events.subscribe()

    def _resolve_or_create_project(self, project: Project | str | Path | None) -> Project:
        if isinstance(project, Project):
            return self.registry.register(project)
        root = find_project_root(project or Path.cwd())
        if project is not None:
            found = self.registry.find(project)
            if found is not None:
                return found
        return self.registry.ensure(root)

    def _client_for_scope(self, scope: Scope) -> Any:
        if scope is Scope.GLOBAL:
            return self.kernel_manager.global_client()
        return self.kernel_manager.project_client(self.project)

    def _memory_key(
        self,
        scope: Scope,
        key: str | None,
        subject: str | None,
        memory_type: str,
    ) -> str:
        if key:
            CanonicalKeyBuilder.validate(key)
            if scope_for_key(key) is not scope:
                raise InvalidCanonicalKey(f"key {key!r} does not belong to {scope.value} scope")
            return key
        if not subject:
            raise ValueError("a memory key or subject is required")
        return key_for_scope(scope, self.project, self.session_id, subject, memory_type=memory_type)

    def _resolve_key_scope(self, key: str, scope: Scope | str | None) -> Scope:
        CanonicalKeyBuilder.validate(key)
        inferred = scope_for_key(key)
        if scope is None:
            return inferred
        selected = Scope.coerce(scope)
        if selected is not inferred:
            raise InvalidCanonicalKey(f"key {key!r} does not belong to {selected.value} scope")
        return selected

    def _new_record(
        self,
        key: str,
        value: str,
        *,
        memory_type: str,
        source_text: str | None,
        confidence: float,
        tags: Iterable[str],
        scope: Scope,
        created_at: int | None = None,
    ) -> MemoryRecord:
        now = int(time.time() * 1000)
        return MemoryRecord(
            key=key,
            value=value,
            memory_type=memory_type,
            source_text=source_text or value,
            created_at=created_at if created_at is not None else now,
            updated_at=max(now, created_at or now),
            confidence=confidence,
            tags=list(tags),
            source="user",
            layer="stm" if scope is Scope.SESSION else "ltm",
        )

    @staticmethod
    def _get_record(client: Any, layer: str, key: str) -> MemoryRecord | None:
        result = client.get(layer, key)
        if not result.get("found"):
            return None
        return memory_record_from_payload(result.get("payload"), key=key, layer=layer)

    def _publish(
        self,
        operation: str,
        request_id: str,
        started: float,
        *,
        actor: str | None = None,
        scope: Scope | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActivityEvent:
        return self.events.publish(
            operation,
            actor=actor or self.actor,
            scope=scope,
            request_id=request_id,
            duration_ms=(time.perf_counter() - started) * 1000,
            metadata=metadata,
        )


def _record_payload(record: MemoryRecord) -> dict[str, Any]:
    return {
        "type": "preference" if record.memory_type == "preference" else "token_stats",
        "data": record.model_dump(mode="json"),
    }


def _bounded_score(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _graph_node_dict(node: Any) -> dict[str, Any]:
    value = asdict(node)
    return value


def _graph_edge_dict(edge: Any) -> dict[str, Any]:
    value = asdict(edge)
    value["kind"] = edge.kind.value
    return value
