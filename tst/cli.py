"""Lightweight installed command surface for TST.

Model dependencies are imported only by ``tst chat``. Commands such as
``doctor``, ``kernel build``, and ``analyze`` remain usable from a core install.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path

from tst.kernel.process import KernelProcessConfig, build_kernel


def _doctor(as_json: bool) -> int:
    config = KernelProcessConfig()
    crate = config.crate_dir
    binary = config.resolved_binary()
    source_available = (crate / "Cargo.toml").is_file()
    checks: list[dict[str, object]] = []

    def add(name: str, ok: bool, detail: str, remedy: str | None = None) -> None:
        item: dict[str, object] = {"name": name, "ok": ok, "detail": detail}
        if remedy and not ok:
            item["remedy"] = remedy
        checks.append(item)

    add(
        "python",
        sys.version_info >= (3, 10),
        sys.version.split()[0],
        "Install Python 3.10 or newer.",
    )
    cargo = shutil.which("cargo")
    add(
        "cargo",
        cargo is not None or binary.is_file(),
        cargo
        or ("not required for configured prebuilt kernel" if binary.is_file() else "not found"),
        "Install Rust with rustup, then run `tst kernel build`.",
    )
    add(
        "kernel-source",
        source_available or binary.is_file(),
        str(crate) if source_available else "not present in this installation",
        "Use a source checkout to build, or set TST_KERNEL_BIN to a compatible prebuilt server.",
    )
    add(
        "kernel",
        binary.is_file(),
        str(binary),
        "Run `tst kernel build`.",
    )
    for module, extra in (
        ("pydantic", "core"),
        ("fastapi", "router"),
        ("torch", "models"),
        ("tree_sitter_language_pack", "analysis"),
    ):
        available = find_spec(module) is not None
        required = module == "pydantic"
        add(
            f"python:{module}",
            available or not required,
            "installed" if available else f"optional {extra} extra not installed",
            "Reinstall the core package." if required else None,
        )

    document = {"ok": all(bool(item["ok"]) for item in checks), "checks": checks}
    if as_json:
        print(json.dumps(document, indent=2))
    else:
        for item in checks:
            marker = "ok" if item["ok"] else "FAIL"
            print(f"[{marker:>4}] {item['name']}: {item['detail']}")
            if item.get("remedy"):
                print(f"       {item['remedy']}")
    return 0 if document["ok"] else 1


def _kernel_build() -> int:
    config = KernelProcessConfig(build_kernel=True)
    binary = build_kernel(config)
    print(f"Built kernel: {binary}")
    return 0


def _service(project: str | None = None, *, actor: str = "TST"):
    from tst.service.service import TSTService

    return TSTService(project, actor=actor)


def _context(
    path_value: str | None,
    query: str,
    *,
    budget: int,
    actor: str = "TST",
    as_json: bool,
) -> int:
    service = _service(path_value, actor=actor)
    try:
        pack = service.retrieve_context(query, budget=budget)
        document = pack.model_dump(mode="json")
        if as_json:
            print(json.dumps(document, indent=2, default=str))
        else:
            print(pack.as_prompt() or "No matching context.")
            print(f"\nEstimated tokens: {pack.estimated_tokens}")
        return 0
    finally:
        service.close()


def _init(path_value: str | None, *, as_json: bool, no_index: bool, integrations: list[str]) -> int:
    service = _service(path_value)
    try:
        index_report: dict[str, object] | None = None
        index_error: str | None = None
        if not no_index:
            try:
                index_report = service.index_project()
            except Exception as exc:
                index_error = f"{type(exc).__name__}: {exc}"
        installed: dict[str, object] = {}
        for provider in integrations:
            try:
                installed[provider] = service.install_integration(provider)
            except Exception as exc:
                installed[provider] = {"error": str(exc)}
        integration_error = any(
            isinstance(result, dict)
            and (
                "error" in result
                or any(status in {"conflict", "invalid"} for status in result.values())
            )
            for result in installed.values()
        )
        detected = {
            "claude": (service.project.root / ".claude").is_dir(),
            "codex": (service.project.root / ".agents").is_dir(),
            "opencode": (service.project.root / ".opencode").is_dir(),
        }
        document = {
            "project": service.project.to_dict(),
            "index": index_report,
            "index_error": index_error,
            "detected_agents": detected,
            "integrations": installed,
        }
        if as_json:
            print(json.dumps(document, indent=2, default=str))
        else:
            print("TST initialized")
            print(f"\nProject\n{service.project.name}")
            print(f"\nRepository\n{service.project.root}")
            if index_report is not None:
                print(f"\nIndexed\n{index_report.get('parsed_files', 0)} changed files")
            elif index_error:
                print(f"\nIndex warning\n{index_error}")
            for provider, result in installed.items():
                print(f"{provider}: {result}")
        return 0 if index_error is None and not integration_error else 2
    finally:
        service.close()


def _status(path_value: str | None, *, as_json: bool) -> int:
    service = _service(path_value)
    try:
        document = service.status()
        if as_json:
            print(json.dumps(document, indent=2, default=str))
        else:
            project = document["project"]
            print(f"{project['name']}  {project['root']}")
            print(f"Kernel: {document['kernel']}")
            print(f"Memory: {document['memory_counts']}")
            if document["errors"]:
                print(f"Errors: {', '.join(document['errors'])}")
        return 0 if document["healthy"] else 1
    finally:
        service.close()


def _projects(*, as_json: bool) -> int:
    from tst.scope.registry import ProjectRegistry

    projects = [project.to_dict() for project in ProjectRegistry().list()]
    if as_json:
        print(json.dumps(projects, indent=2))
    else:
        for project in projects:
            print(f"{project['name']:<24} {project['root']}")
    return 0


def _serve(path_value: str | None, *, host: str, port: int, open_browser: bool) -> int:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print("TST server only binds to loopback addresses.", file=sys.stderr)
        return 2
    from tst.server.app import run

    run(project=path_value, host=host, port=port, open_browser=open_browser)
    return 0


def _mcp(path_value: str | None) -> int:
    from tst.integrations.mcp.server import serve

    serve(project=path_value)
    return 0


def _connect(path_value: str | None, provider: str, *, force: bool, as_json: bool) -> int:
    service = _service(path_value)
    try:
        document = service.install_integration(provider, force=force)
        if as_json:
            print(json.dumps(document, indent=2))
        else:
            for target, status in document.items():
                print(f"{status}: {target}")
        return 2 if any(status in {"conflict", "invalid"} for status in document.values()) else 0
    finally:
        service.close()


def _disconnect(path_value: str | None, provider: str, *, force: bool, as_json: bool) -> int:
    service = _service(path_value)
    try:
        document = service.uninstall_integration(provider, force=force)
        if as_json:
            print(json.dumps(document, indent=2))
        else:
            for target, status in document.items():
                print(f"{status}: {target}")
        return 2 if any(status in {"modified", "invalid"} for status in document.values()) else 0
    finally:
        service.close()


def _analyze(path_value: str, *, as_json: bool, symbol: str | None) -> int:
    from tst.analysis import IncrementalIndexer

    requested = Path(path_value).expanduser().resolve(strict=True)
    root = requested if requested.is_dir() else requested.parent
    indexer = IncrementalIndexer(root)
    include_paths = None
    if requested.is_file():
        include_paths = {requested.relative_to(root).as_posix()}
    report = indexer.index(include_paths)
    graph = indexer.graph

    if symbol:
        symbol_slice = graph.query(symbol, depth=1)
        nodes = symbol_slice.nodes
        edges = symbol_slice.edges
    elif requested.is_file():
        relative = requested.relative_to(root).as_posix()
        node_id = indexer.builder.file_nodes.get(relative)
        file_slice = graph.query(node_id, depth=1) if node_id is not None else None
        nodes = file_slice.nodes if file_slice else []
        edges = file_slice.edges if file_slice else []
    else:
        nodes = list(graph.nodes.values())
        edges = graph.edges

    document = {
        "root": str(root),
        "report": asdict(report),
        "graph": {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "nodes": [asdict(node) for node in nodes],
            "edges": [
                {
                    **asdict(edge),
                    "kind": edge.kind.value,
                }
                for edge in edges
            ],
        },
    }
    if as_json:
        print(json.dumps(document, indent=2, default=str))
    else:
        print(
            f"Indexed {len(graph.nodes)} nodes and {len(graph.edges)} edges "
            f"({report.parsed_files} parsed, {len(report.unchanged)} unchanged)."
        )
        for node in nodes[:50]:
            location = (
                f"{node.file_path}:{node.start_line}-{node.end_line}"
                if node.file_path and node.start_line
                else node.file_path or ""
            )
            print(f"  {node.node_type:<12} {node.qualified_name} {location}".rstrip())
        if len(nodes) > 50:
            print(f"  ... {len(nodes) - 50} more nodes")
        for path, errors in sorted(report.parse_errors.items()):
            for error in errors:
                print(f"  parser warning {path}: {error}")
    return 0 if not report.parse_errors else 2


def _chat(args: argparse.Namespace) -> int:
    forwarded: list[str] = []
    if args.no_kernel:
        forwarded.append("--no-kernel")
    if args.no_worker:
        forwarded.append("--no-worker")
    if args.build_kernel:
        forwarded.append("--build-kernel")
    try:
        legacy_cli = importlib.import_module("cli")
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "transformers", "accelerate"}:
            print(
                "Chat model dependencies are not installed. "
                "Install the project with the models extra.",
                file=sys.stderr,
            )
            return 2
        raise
    previous = sys.argv
    try:
        sys.argv = ["tst chat", *forwarded]
        result = legacy_cli.main()
    finally:
        sys.argv = previous
    return int(result or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tst", description="TST local memory framework")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor", help="check the local installation")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    kernel = subcommands.add_parser("kernel", help="manage the Rust kernel")
    kernel_commands = kernel.add_subparsers(dest="kernel_command", required=True)
    kernel_commands.add_parser("build", help="build the release kernel explicitly")

    analyze = subcommands.add_parser("analyze", help="index a source repository safely")
    analyze.add_argument("path")
    analyze.add_argument("--symbol")
    analyze.add_argument("--json", action="store_true", dest="as_json")

    context = subcommands.add_parser("context", help="retrieve explainable project context")
    context.add_argument("query", nargs="?")
    context.add_argument("--query", dest="query_option")
    context.add_argument("--project", dest="path_value")
    context.add_argument("--budget", type=int, default=2_000)
    context.add_argument("--actor", default="TST")
    context.add_argument("--json", action="store_true", dest="as_json")

    init = subcommands.add_parser("init", help="initialize and register a project")
    init.add_argument("path", nargs="?", default=None)
    init.add_argument("--no-index", action="store_true")
    init.add_argument("--connect", action="append", choices=["claude", "codex", "opencode"], default=[])
    init.add_argument("--json", action="store_true", dest="as_json")

    status = subcommands.add_parser("status", help="show TST project and kernel status")
    status.add_argument("--project", dest="path_value")
    status.add_argument("--json", action="store_true", dest="as_json")

    projects = subcommands.add_parser("projects", help="list registered projects")
    projects.add_argument("--json", action="store_true", dest="as_json")

    serve = subcommands.add_parser("serve", help="run the local TST API")
    serve.add_argument("--project", dest="path_value")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    ui = subcommands.add_parser("ui", help="run the local API and open the TST UI")
    ui.add_argument("--project", dest="path_value")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8000)

    mcp = subcommands.add_parser("mcp", help="serve TST over MCP")
    mcp_commands = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_serve = mcp_commands.add_parser("serve", help="run the stdio MCP server")
    mcp_serve.add_argument("--project", dest="path_value")

    connect = subcommands.add_parser("connect", help="install a coding-agent integration")
    connect.add_argument("provider", choices=["claude", "codex", "opencode"])
    connect.add_argument("--project", dest="path_value")
    connect.add_argument("--force", action="store_true")
    connect.add_argument("--json", action="store_true", dest="as_json")

    disconnect = subcommands.add_parser("disconnect", help="remove a coding-agent integration")
    disconnect.add_argument("provider", choices=["claude", "codex", "opencode"])
    disconnect.add_argument("--project", dest="path_value")
    disconnect.add_argument("--force", action="store_true", help="remove modified generated files")
    disconnect.add_argument("--json", action="store_true", dest="as_json")

    chat = subcommands.add_parser("chat", help="start the local model-backed chat REPL")
    chat.add_argument("--no-kernel", action="store_true")
    chat.add_argument("--no-worker", action="store_true")
    chat.add_argument("--build-kernel", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.as_json)
    if args.command == "kernel" and args.kernel_command == "build":
        return _kernel_build()
    if args.command == "analyze":
        return _analyze(args.path, as_json=args.as_json, symbol=args.symbol)
    if args.command == "context":
        query = args.query_option or args.query
        if not query:
            parser.error("context requires a query")
        return _context(args.path_value, query, budget=args.budget, actor=args.actor, as_json=args.as_json)
    if args.command == "init":
        return _init(args.path, as_json=args.as_json, no_index=args.no_index, integrations=args.connect)
    if args.command == "status":
        return _status(args.path_value, as_json=args.as_json)
    if args.command == "projects":
        return _projects(as_json=args.as_json)
    if args.command == "serve":
        return _serve(args.path_value, host=args.host, port=args.port, open_browser=False)
    if args.command == "ui":
        return _serve(args.path_value, host=args.host, port=args.port, open_browser=True)
    if args.command == "mcp" and args.mcp_command == "serve":
        return _mcp(args.path_value)
    if args.command == "connect":
        return _connect(args.path_value, args.provider, force=args.force, as_json=args.as_json)
    if args.command == "disconnect":
        return _disconnect(args.path_value, args.provider, force=args.force, as_json=args.as_json)
    if args.command == "chat":
        return _chat(args)
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
