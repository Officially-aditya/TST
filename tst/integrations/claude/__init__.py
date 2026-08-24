"""Claude Code project skill installer."""

from __future__ import annotations

from pathlib import Path

from tst.integrations.installer import install_skills, write_json_file


def install(root: str | Path, *, force: bool = False) -> dict[str, str]:
    result = install_skills(root, ".claude/skills", force=force)
    result[str(Path(root).expanduser().resolve() / ".mcp.json")] = write_json_file(
        root,
        ".mcp.json",
        {"mcpServers": {"tst": {"command": "tst", "args": ["mcp", "serve"]}}},
        force=force,
    )
    return result


__all__ = ["install"]
