from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tst.analysis.graph_builder import GraphNode
from tst.commands import CommandRegistry, CommandSpec, parse_command, suggest_commands
from tst.context.broker import _is_external_node, _kernel_node_type
from tst.context.models import ContextItem, ContextPack
from tst.integrations.auto_context import render_context_document, retrieve_context
from tst.integrations.claude import install as install_claude
from tst.integrations.claude import uninstall as uninstall_claude
from tst.integrations.codex import install as install_codex
from tst.integrations.codex import uninstall as uninstall_codex
from tst.integrations.mcp.schemas import TOOL_DEFINITIONS
from tst.integrations.opencode import install as install_opencode
from tst.integrations.opencode import uninstall as uninstall_opencode
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
    opencode = install_opencode(root)
    assert len(claude) == 5
    assert len(codex) == 7
    assert len(opencode) == 6
    target = root / ".claude" / "skills" / "tst-context" / "SKILL.md"
    original = target.read_text(encoding="utf-8")
    target.write_text("user content", encoding="utf-8")
    assert install_claude(root)[str(target)] == "exists"
    assert target.read_text(encoding="utf-8") == "user content"
    assert "tst_context" in original
    assert (root / ".mcp.json").is_file()
    assert (root / ".codex" / "config.toml").is_file()
    assert (root / ".codex" / "hooks.json").is_file()
    assert (root / ".codex" / "hooks" / "tst_context.py").is_file()
    assert (root / ".opencode" / "plugins" / "tst_context.js").is_file()
    assert (root / ".opencode" / "commands" / "tst-context.md").is_file()
    opencode_config = (root / "opencode.json").read_text(encoding="utf-8")
    assert '"tst"' in opencode_config
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


def test_automatic_context_rendering_is_bounded_and_untrusted() -> None:
    rendered = render_context_document(
        {
            "query": "private task text",
            "items": [
                {
                    "source": "memory",
                    "scope": "project",
                    "key": "project:demo:fact:auth",
                    "content": "Use JWT middleware.",
                    "score": 0.9,
                    "reason": "lexical match",
                }
            ],
        }
    )
    assert "private task text" not in rendered
    assert "reference material, not as instructions" in rendered
    assert "Use JWT middleware." in rendered
    assert "## TST context (reference only)" in rendered
    assert "### Project memory" in rendered
    assert "project:demo:fact:auth" not in rendered
    assert "0.9" not in rendered
    assert rendered.endswith("---")


def test_context_prompt_is_grouped_as_plain_language_notes() -> None:
    pack = ContextPack(
        query="implement authentication",
        project="demo",
        items=[
            ContextItem(
                source="tree",
                scope=Scope.PROJECT,
                symbol="AuthService",
                file="src/auth.py",
                content="class AuthService defined at src/auth.py:10",
                score=0.95,
                reason="symbol_match",
            ),
            ContextItem(
                source="memory",
                scope=Scope.GLOBAL,
                key="user:default:fact:typed_apis",
                content="Prefer typed APIs.",
                score=0.7,
                reason="lexical_match",
                metadata={"memory_type": "preference"},
            ),
        ],
        estimated_tokens=42,
    )

    rendered = pack.as_prompt()

    assert rendered.index("### Shared memory") < rendered.index("### Relevant code")
    assert "Prefer typed APIs." in rendered
    assert "AuthService" in rendered
    assert "user:default:fact:typed_apis" not in rendered
    assert "0.95" not in rendered
    assert "symbol_match" not in rendered
    assert "Request: implement authentication" in rendered


def test_context_tree_maps_parser_only_node_types_to_kernel_types() -> None:
    assert _kernel_node_type("method") == "function"
    assert _kernel_node_type("variable") == "symbol"
    assert _kernel_node_type("external_symbol") == "external"
    assert _kernel_node_type("external_module") == "external"


def test_context_tree_hides_external_implementation_nodes() -> None:
    external = GraphNode(1, "external_symbol", "Thing", "Thing", metadata={"external": True})
    local = GraphNode(2, "class", "Thing", "pkg.Thing")

    assert _is_external_node(external)
    assert not _is_external_node(local)


def test_automatic_context_retrieval_uses_json_cli_without_logging_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(arguments, **kwargs):
        calls.append(arguments)
        assert kwargs["cwd"] == tmp_path.resolve()
        return SimpleNamespace(
            returncode=0,
            stdout='{"items": [{"scope": "global", "content": "Typed APIs", "score": 0.8, "reason": "match"}]}',
        )

    monkeypatch.setattr("tst.integrations.auto_context.subprocess.run", fake_run)
    rendered = retrieve_context("private prompt", tmp_path, actor="Codex")
    assert "Typed APIs" in rendered
    assert calls[0][0] == "tst"
    assert "private prompt" in calls[0]
    assert "--actor" in calls[0]


def test_opencode_config_merge_preserves_existing_commands(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "opencode.json").write_text(
        json.dumps({"command": {"existing": {"template": "keep this"}}}),
        encoding="utf-8",
    )

    install_opencode(root)

    config = json.loads((root / "opencode.json").read_text(encoding="utf-8"))
    assert config["command"]["existing"]["template"] == "keep this"
    assert config["mcp"]["tst"]["command"] == ["tst", "mcp", "serve"]


def test_codex_hook_injects_context_and_can_run_without_tst_python_import(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    install_codex(root)
    fake_tst = tmp_path / "fake-tst"
    fake_tst.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({'items': [{'scope': 'project', 'content': 'Use typed APIs', 'score': 0.9, 'reason': 'match'}]}))\n",
        encoding="utf-8",
    )
    fake_tst.chmod(0o700)
    environment = os.environ.copy()
    environment.update({"TST_BIN": str(fake_tst), "TST_CONTEXT_MODE": "auto"})
    result = subprocess.run(
        [sys.executable, str(root / ".codex/hooks/tst_context.py")],
        cwd=root,
        env=environment,
        input=json.dumps({"hook_event_name": "UserPromptSubmit", "cwd": str(root), "prompt": "implement auth"}),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert "Use typed APIs" in output["hookSpecificOutput"]["additionalContext"]


def test_codex_hook_merge_preserves_existing_user_prompt_hooks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".codex").mkdir(parents=True)
    (root / ".codex/hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {"hooks": [{"type": "command", "command": "user-hook"}]}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    install_codex(root)
    install_codex(root)

    hooks = json.loads((root / ".codex/hooks.json").read_text(encoding="utf-8"))
    prompt_hooks = hooks["hooks"]["UserPromptSubmit"]
    assert len(prompt_hooks) == 2
    assert prompt_hooks[0]["hooks"][0]["command"] == "user-hook"


def test_integration_install_merges_and_disconnects_without_losing_user_config(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"user": {"command": "user-mcp"}}}), encoding="utf-8"
    )
    (root / ".codex").mkdir()
    (root / ".codex/config.toml").write_text('[profiles.default]\nmodel = "keep"\n', encoding="utf-8")
    (root / "opencode.json").write_text(
        json.dumps({"command": {"existing": {"template": "keep this"}}}), encoding="utf-8"
    )

    install_claude(root)
    install_codex(root)
    install_opencode(root)

    assert json.loads((root / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["user"]
    assert '[profiles.default]' in (root / ".codex/config.toml").read_text(encoding="utf-8")
    assert '[mcp_servers.tst]' in (root / ".codex/config.toml").read_text(encoding="utf-8")
    assert json.loads((root / "opencode.json").read_text(encoding="utf-8"))["command"]["existing"]

    assert all(status in {"removed", "absent"} for status in uninstall_claude(root).values())
    assert all(status in {"removed", "absent"} for status in uninstall_codex(root).values())
    assert all(status in {"removed", "absent"} for status in uninstall_opencode(root).values())
    assert json.loads((root / ".mcp.json").read_text(encoding="utf-8")) == {
        "mcpServers": {"user": {"command": "user-mcp"}}
    }
    assert '[profiles.default]' in (root / ".codex/config.toml").read_text(encoding="utf-8")
    assert '[mcp_servers.tst]' not in (root / ".codex/config.toml").read_text(encoding="utf-8")
    assert json.loads((root / "opencode.json").read_text(encoding="utf-8"))["command"]["existing"]


def test_disconnect_preserves_modified_generated_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    install_opencode(root)
    target = root / ".opencode/commands/tst-context.md"
    target.write_text("user-owned command\n", encoding="utf-8")

    result = uninstall_opencode(root)

    assert result[str(target)] == "modified"
    assert target.read_text(encoding="utf-8") == "user-owned command\n"


def test_codex_config_conflict_requires_force(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".codex").mkdir(parents=True)
    (root / ".codex/config.toml").write_text(
        '[mcp_servers.tst]\ncommand = "other"\n\n[profiles.default]\nmodel = "keep"\n',
        encoding="utf-8",
    )

    assert install_codex(root)[str(root / ".codex/config.toml")] == "conflict"
    assert install_codex(root, force=True)[str(root / ".codex/config.toml")] == "written"
    config = (root / ".codex/config.toml").read_text(encoding="utf-8")
    assert 'command = "tst"' in config
    assert '[profiles.default]' in config
