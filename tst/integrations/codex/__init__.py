"""Codex project skill installer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tst.integrations.installer import (
    install_skills,
    merge_json_file,
    merge_text_block,
    remove_json_list_item,
    remove_text_block,
    remove_text_file,
    write_text_file,
)
from tst.integrations.skills import SKILLS

HOOK_SOURCE = Path(__file__).with_name("tst_context.py").read_text(encoding="utf-8")
HOOKS: dict[str, Any] = {
    "description": "TST automatic context retrieval for Codex.",
    "hooks": {
        "UserPromptSubmit": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": 'python3 "$(git rev-parse --show-toplevel)/.codex/hooks/tst_context.py"',
                        "statusMessage": "Retrieving TST context",
                        "timeout": 30,
                        "additionalContextLimit": 3000,
                    }
                ]
            }
        ]
    },
}
HOOK_ENTRY = HOOKS["hooks"]["UserPromptSubmit"][0]
CODEX_CONFIG = '[mcp_servers.tst]\ncommand = "tst"\nargs = ["mcp", "serve"]\n'
CODEX_CONFIG_MARKER = "TST CODEX MCP"


def install(root: str | Path, *, force: bool = False) -> dict[str, str]:
    result = install_skills(root, ".agents/skills", force=force)
    config_path = Path(root).expanduser().resolve() / ".codex" / "config.toml"
    if config_path.exists():
        config_status = merge_text_block(
            root,
            ".codex/config.toml",
            CODEX_CONFIG,
            CODEX_CONFIG_MARKER,
            force=force,
        )
    else:
        config_status = write_text_file(root, ".codex/config.toml", CODEX_CONFIG, force=force)
    result[str(config_path)] = config_status
    result[str(Path(root).expanduser().resolve() / ".codex" / "hooks" / "tst_context.py")] = write_text_file(
        root,
        ".codex/hooks/tst_context.py",
        HOOK_SOURCE,
        force=force,
    )
    result[str(Path(root).expanduser().resolve() / ".codex" / "hooks.json")] = merge_json_file(
        root,
        ".codex/hooks.json",
        HOOKS,
        force=force,
        append_lists={("hooks", "UserPromptSubmit")},
    )
    return result


def uninstall(root: str | Path, *, force: bool = False) -> dict[str, str]:
    project_root = Path(root).expanduser().resolve()
    result: dict[str, str] = {}
    for name, content in SKILLS.items():
        relative = f".agents/skills/{name}/SKILL.md"
        result[str(project_root / relative)] = remove_text_file(root, relative, content, force=force)
    config_relative = ".codex/config.toml"
    result[str(project_root / config_relative)] = remove_text_block(
        root,
        config_relative,
        CODEX_CONFIG,
        CODEX_CONFIG_MARKER,
        force=force,
    )
    hook_relative = ".codex/hooks/tst_context.py"
    result[str(project_root / hook_relative)] = remove_text_file(
        root,
        hook_relative,
        HOOK_SOURCE,
        force=force,
    )
    result[str(project_root / ".codex/hooks.json")] = remove_json_list_item(
        root,
        ".codex/hooks.json",
        ("hooks", "UserPromptSubmit"),
        HOOK_ENTRY,
        force=force,
    )
    return result


__all__ = ["install", "uninstall"]
