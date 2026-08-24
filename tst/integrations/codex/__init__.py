"""Codex project skill installer."""

from __future__ import annotations

from pathlib import Path

from tst.integrations.installer import install_skills, write_text_file


def install(root: str | Path, *, force: bool = False) -> dict[str, str]:
    result = install_skills(root, ".agents/skills", force=force)
    result[str(Path(root).expanduser().resolve() / ".codex" / "config.toml")] = write_text_file(
        root,
        ".codex/config.toml",
        '[mcp_servers.tst]\ncommand = "tst"\nargs = ["mcp", "serve"]\n',
        force=force,
    )
    return result


__all__ = ["install"]
