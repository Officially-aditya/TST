"""Claude Code project skill installer."""

from __future__ import annotations

from pathlib import Path

from tst.integrations.installer import (
    install_skills,
    merge_json_file,
    remove_json_key,
    remove_text_file,
)
from tst.integrations.skills import SKILLS

MCP_SERVER = {"command": "tst", "args": ["mcp", "serve"]}


def install(root: str | Path, *, force: bool = False) -> dict[str, str]:
    result = install_skills(root, ".claude/skills", force=force)
    result[str(Path(root).expanduser().resolve() / ".mcp.json")] = merge_json_file(
        root,
        ".mcp.json",
        {"mcpServers": {"tst": MCP_SERVER}},
        force=force,
    )
    return result


def uninstall(root: str | Path, *, force: bool = False) -> dict[str, str]:
    project_root = Path(root).expanduser().resolve()
    result = {
        str(project_root / ".mcp.json"): remove_json_key(
            root,
            ".mcp.json",
            ("mcpServers", "tst"),
            MCP_SERVER,
            force=force,
        )
    }
    for name, content in SKILLS.items():
        relative = f".claude/skills/{name}/SKILL.md"
        result[str(project_root / relative)] = remove_text_file(root, relative, content, force=force)
    return result


__all__ = ["install", "uninstall"]
