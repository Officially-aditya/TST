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
    if args.command == "chat":
        return _chat(args)
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
