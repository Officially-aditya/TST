from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tst.commands import CommandRegistry, CommandSpec, parse_command, suggest_commands
from tst.integrations.claude import install as install_claude
from tst.integrations.codex import install as install_codex
from tst.integrations.mcp.schemas import TOOL_DEFINITIONS
from tst.scope.broker import key_for_scope, remap_key, scope_for_key
from tst.scope.models import Project, Scope
from tst.service.events import EventBus


def test_scope_key_translation_and_registry_commands(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "repo", project_id="demo")
    project.root.mkdir()
    key = key_for_scope(Scope.PROJECT, project, "session", "auth", memory_type="fact")
    assert scope_for_key(key) is Scope.PROJECT
    moved = remap_key(key, Scope.GLOBAL, project, "session")
    assert moved == "user:default:context:auth"

    calls: list[str] = []
    registry = CommandRegistry(
        [CommandSpec("context", "inspect", lambda command: calls.append(command.name) or SimpleNamespace())]
    )
    assert parse_command('/context "auth middleware"').args == ("auth middleware",)
    assert suggest_commands("/con", registry.specs())[0].name == "context"
    registry.execute("/context auth")
    assert calls == ["context"]


def test_event_redaction_preserves_operation_metadata() -> None:
    bus = EventBus()
    event = bus.publish(
        "memory.store",
        metadata={"key": "project:demo:fact:auth", "value": "secret memory", "query": "private prompt", "count": 1},
    )
    assert event.metadata["key"] == "project:demo:fact:auth"
    assert event.metadata["count"] == 1
    assert "value" not in event.metadata
    assert "query" not in event.metadata


def test_agent_skill_installers_are_project_local_and_non_destructive(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    claude = install_claude(root)
    codex = install_codex(root)
    assert len(claude) == 5
    assert len(codex) == 5
    target = root / ".claude" / "skills" / "tst-context" / "SKILL.md"
    original = target.read_text(encoding="utf-8")
    target.write_text("user content", encoding="utf-8")
    assert install_claude(root)[str(target)] == "exists"
    assert target.read_text(encoding="utf-8") == "user content"
    assert "tst_context" in original
    assert (root / ".mcp.json").is_file()
    assert (root / ".codex" / "config.toml").is_file()
    assert {item["name"] for item in TOOL_DEFINITIONS} == {
        "tst_status",
        "tst_context",
        "tst_memory_search",
        "tst_memory_store",
        "tst_memory_update",
        "tst_memory_forget",
        "tst_tree_find",
        "tst_tree_query",
    }
