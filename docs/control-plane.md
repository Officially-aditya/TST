# TST Control Plane

TST v0.3 adds a Python-owned visible context layer above the existing Rust
kernel. The Rust process still speaks only versioned NDJSON over STDIO. Python
owns project discovery, scope selection, retrieval explanation, indexing,
HTTP, UI, and agent integrations.

## Scopes and Storage

The application maps user-facing scopes to kernel layers:

| Scope | Kernel | Storage |
|---|---|---|
| Global | LTM | `~/.tst/global/ltm.snapshot` |
| Project | LTM | `<repo>/.tst/ltm.snapshot` |
| Session | STM | active project process only |

`ProjectRegistry` stores repository identities in `~/.tst/projects.json` and
each repository stores its identity in `.tst/project.json`. `TST_HOME` can be
set for an isolated local installation or tests.

`KernelManager` keeps one global child and one active-project child. Switching
projects closes the old project child before starting the new one. The manager
always overrides `TST_SNAPSHOT_PATH` with an absolute path, so separate
repositories cannot accidentally share a relative snapshot.

## Context Packs

`ContextBroker.retrieve(query, project, session_id, budget)` merges:

- global LTM records
- project LTM records
- session STM records
- the safe Python `CodeGraph` and its bounded relationships

It returns a `ContextPack` of `ContextItem` values. Each item retains source,
scope, key or symbol, score, reason, and layer metadata. `ContextPack.as_prompt`
renders a provider-neutral prompt fragment; no model provider is required.

## Service and Local API

All clients call `TSTService`, never the kernel client directly:

```bash
tst init
tst context --query "implement authentication middleware" --json
tst status --json
tst serve
tst ui
```

The API binds to loopback by default:

```text
GET    /api/v1/status
GET    /api/v1/projects
POST   /api/v1/projects/select
POST   /api/v1/context/preview
GET    /api/v1/memories
POST   /api/v1/memories
PATCH  /api/v1/memories/{key}
DELETE /api/v1/memories/{key}
POST   /api/v1/memories/{key}/move
POST   /api/v1/index
GET    /api/v1/tree/find
POST   /api/v1/tree/query
GET    /api/v1/integrations
GET    /api/v1/events/stream
```

Set `TST_REQUIRE_UI_TOKEN=1` to require the ephemeral `X-TST-Session` token
returned by `/api/v1/status` on subsequent API requests. Prompt and memory
values are redacted from the bounded in-memory activity bus.

## Agent Integrations

The universal integration is MCP:

```bash
tst mcp serve
```

The stdio adapter exposes `tst_status`, `tst_context`, memory CRUD tools, and
bounded tree tools. Claude and Codex project skills are generated explicitly:

```bash
tst connect claude
tst connect codex
```

Claude files live under `.claude/skills`; Codex files live under
`.agents/skills`. Existing files are not overwritten unless `--force` is
provided. The generated skills instruct each agent to call the same MCP
surface instead of implementing memory behavior independently.

## UI

The source UI is a React/TypeScript/Vite application under `ui/`. Build it with
`npm install && npm run build`; the server prefers `ui/dist` when present and
otherwise serves a bundled static fallback. The first UI includes Project,
Context, Memory, Tree, Activity, and Connections screens.
