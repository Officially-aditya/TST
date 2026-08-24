"""Shared skill text used by Claude Code and Codex installers."""

from __future__ import annotations

SKILLS: dict[str, str] = {
    "tst-context": """---
name: tst-context
---

Use the TST MCP tool `tst_context` with the user's task. Present the returned
items grouped by scope and retain the explanation/reason metadata. Do not
invent memories or bypass TST's scope rules.
""",
    "tst-memory": """---
name: tst-memory
---

Use the TST MCP tools `tst_memory_search`, `tst_memory_store`,
`tst_memory_update`, and `tst_memory_forget`. Ask before storing information
unless the user explicitly requested it. Never promote project memory to
global scope automatically.
""",
    "tst-status": """---
name: tst-status
---

Call the TST MCP tool `tst_status` and summarize the project and health fields.
Do not expose memory values in general status output.
""",
    "tst-tree": """---
name: tst-tree
---

Use `tst_tree_find` for symbol discovery and `tst_tree_query` for bounded
relationships. Keep file paths and line locations from TST's result intact.
""",
}
