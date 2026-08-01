"""Best-effort local import, call, inheritance, and test-edge resolution."""

from __future__ import annotations

import posixpath
from collections import defaultdict
from pathlib import PurePosixPath

from .graph_builder import EdgeKind, GraphBuilder, GraphEdge, GraphNode
from .models import ParsedImport, ReferenceKind

_BUILTIN_CALLS = {
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "open",
    "print",
    "range",
    "set",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "zip",
}


class SymbolResolver:
    def __init__(self, builder: GraphBuilder) -> None:
        self.builder = builder
        self.graph = builder.graph
        self._module_files: dict[str, int] = {}
        self._file_modules: dict[str, str] = {}
        self._symbols_by_name: dict[str, list[GraphNode]] = defaultdict(list)
        self._symbols_by_qualified: dict[str, list[GraphNode]] = defaultdict(list)
        self._default_exports: dict[str, GraphNode] = {}

    def resolve(self) -> None:
        dynamic = {
            EdgeKind.IMPORTS,
            EdgeKind.CALLS,
            EdgeKind.REFERENCES,
            EdgeKind.INHERITS,
            EdgeKind.IMPLEMENTS,
            EdgeKind.TESTS,
        }
        self.graph.remove_edges(lambda edge: edge.kind in dynamic)
        external_ids = [
            node.node_id for node in self.graph.nodes.values() if node.metadata.get("external")
        ]
        for node_id in external_ids:
            self.graph.remove_node(node_id)

        self._build_indexes()
        aliases_by_file: dict[str, dict[str, str]] = {}
        for path, parsed in self.builder.parsed_files.items():
            aliases_by_file[path] = self._resolve_imports(path, parsed.imports)
        for path, parsed in self.builder.parsed_files.items():
            aliases = aliases_by_file[path]
            for symbol in parsed.symbols:
                source_id = self.builder.symbol_nodes.get((path, symbol.qualified_name))
                if source_id is None:
                    continue
                for base in symbol.bases:
                    target, confidence = self._resolve_reference(
                        path, symbol.qualified_name, base, aliases
                    )
                    if target is None:
                        target = self.builder.external_node(base, kind="external_symbol")
                        confidence = 0.3
                    self.graph.add_edge(
                        GraphEdge(source_id, target, EdgeKind.INHERITS, confidence, {"name": base})
                    )
                for implemented in symbol.implements:
                    target, confidence = self._resolve_reference(
                        path, symbol.qualified_name, implemented, aliases
                    )
                    if target is None:
                        target = self.builder.external_node(implemented, kind="external_symbol")
                        confidence = 0.3
                    self.graph.add_edge(
                        GraphEdge(
                            source_id,
                            target,
                            EdgeKind.IMPLEMENTS,
                            confidence,
                            {"name": implemented},
                        )
                    )
                for reference in symbol.references:
                    target, confidence = self._resolve_reference(
                        path, symbol.qualified_name, reference.name, aliases
                    )
                    if target is None:
                        if reference.kind == ReferenceKind.CALL or (
                            "." in reference.name
                            and reference.name.split(".", 1)[0] not in {"self", "cls"}
                        ):
                            target = self.builder.external_node(
                                reference.name, kind="external_symbol"
                            )
                            confidence = 0.3
                        else:
                            continue
                    edge_kind = (
                        EdgeKind.CALLS
                        if reference.kind == ReferenceKind.CALL
                        else EdgeKind.REFERENCES
                    )
                    self.graph.add_edge(
                        GraphEdge(
                            source_id,
                            target,
                            edge_kind,
                            confidence,
                            {"line": reference.line, "name": reference.name},
                        )
                    )

    def _build_indexes(self) -> None:
        self._module_files.clear()
        self._file_modules.clear()
        self._symbols_by_name.clear()
        self._symbols_by_qualified.clear()
        self._default_exports.clear()
        for path, file_id in self.builder.file_nodes.items():
            module = self.graph.nodes[file_id].qualified_name
            self._file_modules[path] = module
            aliases = {module, path.rsplit(".", 1)[0].replace("/", ".")}
            if module.startswith("src."):
                aliases.add(module.removeprefix("src."))
            for alias in aliases:
                if alias:
                    self._module_files.setdefault(alias, file_id)
        for node in self.graph.nodes.values():
            if node.file_path and node.node_type not in {"file", "project"}:
                self._symbols_by_name[node.name].append(node)
                self._symbols_by_qualified[node.qualified_name].append(node)
                if node.metadata.get("default_export"):
                    module = self._file_modules[node.file_path]
                    self._default_exports[module] = node

    def _normalize_module(self, path: str, imported: ParsedImport) -> str:
        language = self.builder.parsed_files[path].language
        raw = imported.module.strip().strip("'\"")
        if language == "python":
            if imported.level:
                current = self._file_modules[path]
                package = (
                    current
                    if PurePosixPath(path).name == "__init__.py"
                    else current.rpartition(".")[0]
                )
                parts = [part for part in package.split(".") if part]
                trim = max(0, imported.level - 1)
                if trim:
                    parts = parts[:-trim] if trim <= len(parts) else []
                if raw:
                    parts.extend(part for part in raw.split(".") if part)
                return ".".join(parts)
            return raw
        if language in {"javascript", "typescript", "tsx"} and raw.startswith("."):
            parent = PurePosixPath(path).parent.as_posix()
            joined = posixpath.normpath(posixpath.join(parent, raw))
            for suffix in (".d.ts", ".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs"):
                if joined.endswith(suffix):
                    joined = joined[: -len(suffix)]
                    break
            return joined.replace("/", ".")
        if language == "rust":
            current = self._file_modules[path]
            raw = raw.replace("::", ".").replace("{", "").replace("}", "")
            raw = raw.split(",", 1)[0].strip().rstrip(".*")
            if imported.metadata.get("module_declaration"):
                pure = PurePosixPath(path)
                parent = current if pure.name == "mod.rs" else ".".join(pure.parent.parts)
                return f"{parent}.{raw}".strip(".")
            if raw.startswith("crate."):
                pure = PurePosixPath(path)
                parts = list(pure.parent.parts)
                if "src" in parts:
                    parts = parts[: parts.index("src") + 1]
                crate_root = ".".join(parts)
                return f"{crate_root}.{raw.removeprefix('crate.')}".strip(".")
            if raw.startswith("self."):
                return f"{current.rpartition('.')[0]}.{raw.removeprefix('self.')}".strip(".")
            if raw.startswith("super."):
                parent = current.rpartition(".")[0].rpartition(".")[0]
                return f"{parent}.{raw.removeprefix('super.')}".strip(".")
        return raw.replace("/", ".")

    def _module_target(self, module: str) -> int | None:
        candidates = [module]
        # JS/TS imports often omit an index file or extension.
        candidates.extend((f"{module}.index", f"{module}.__init__", f"src.{module}"))
        if module.endswith(".index"):
            candidates.append(module.removesuffix(".index"))
        for candidate in candidates:
            target = self._module_files.get(candidate)
            if target is not None:
                return target
        return None

    def _resolve_imports(self, path: str, imports: list[ParsedImport]) -> dict[str, str]:
        aliases: dict[str, str] = {}
        source_id = self.builder.file_nodes[path]
        is_test = (
            PurePosixPath(path).name.startswith("test_") or "tests" in PurePosixPath(path).parts
        )
        for imported in imports:
            module = self._normalize_module(path, imported)
            target_id = self._module_target(module)
            if target_id is None and imported.metadata.get("rust_leaf"):
                symbol = self._find_by_qualified_suffix(module)
                if symbol is not None:
                    target_id = symbol.node_id
            if target_id is None:
                target_id = self.builder.external_node(
                    module or imported.raw, kind="external_module"
                )
                confidence = 0.5
            else:
                confidence = 1.0
            self.graph.add_edge(
                GraphEdge(
                    source_id,
                    target_id,
                    EdgeKind.IMPORTS,
                    confidence,
                    {"line": imported.line, "module": module},
                )
            )
            if is_test and self.graph.nodes[target_id].file_path:
                self.graph.add_edge(
                    GraphEdge(
                        source_id, target_id, EdgeKind.TESTS, confidence, {"line": imported.line}
                    )
                )

            explicit_aliases = dict(imported.aliases)
            for original, alias in explicit_aliases.items():
                if original in {"*", imported.module, module}:
                    aliases[alias] = module
                elif original == "default" and module in self._default_exports:
                    aliases[alias] = self._default_exports[module].qualified_name
                else:
                    aliases[alias] = f"{module}.{original}".strip(".")
            for name in imported.names:
                if name == "*":
                    continue
                alias = explicit_aliases.get(name, name)
                default_export = self._default_exports.get(module) if name == "default" else None
                aliases[alias] = (
                    default_export.qualified_name
                    if default_export is not None
                    else f"{module}.{name}".strip(".")
                )
                symbol = self._find_by_qualified_suffix(aliases[alias])
                named_module_id = self._module_target(aliases[alias])
                imported_target = symbol.node_id if symbol is not None else named_module_id
                if imported_target is not None:
                    self.graph.add_edge(
                        GraphEdge(
                            source_id,
                            imported_target,
                            EdgeKind.IMPORTS,
                            1.0,
                            {"line": imported.line},
                        )
                    )
                    if is_test:
                        self.graph.add_edge(
                            GraphEdge(
                                source_id,
                                imported_target,
                                EdgeKind.TESTS,
                                1.0,
                                {"line": imported.line},
                            )
                        )
            if not imported.names:
                binding = module.split(".")[-1] if module else imported.module.split(".")[-1]
                if binding:
                    aliases.setdefault(binding, module)
                root_binding = imported.module.split(".")[0] if imported.module else ""
                if root_binding:
                    aliases.setdefault(root_binding, imported.module)
        return aliases

    def _find_by_qualified_suffix(self, qualified: str) -> GraphNode | None:
        exact = self._symbols_by_qualified.get(qualified)
        if exact:
            return exact[0]
        matches = [
            node
            for name, nodes in self._symbols_by_qualified.items()
            if name == qualified or name.endswith(f".{qualified}")
            for node in nodes
        ]
        return matches[0] if len(matches) == 1 else None

    def _resolve_reference(
        self,
        path: str,
        local_qualified: str,
        raw_name: str,
        aliases: dict[str, str],
    ) -> tuple[int | None, float]:
        name = raw_name.strip()
        if not name or name in _BUILTIN_CALLS:
            return None, 0.0
        pieces = name.split(".")
        local_parent = local_qualified.rpartition(".")[0]
        if pieces[0] in {"self", "cls", "this"} and len(pieces) > 1:
            candidate = f"{local_parent}.{'.'.join(pieces[1:])}".strip(".")
            local_target_id = self.builder.symbol_nodes.get((path, candidate))
            if local_target_id is not None:
                return local_target_id, 1.0
        if pieces[0] in aliases:
            expanded = ".".join([aliases[pieces[0]], *pieces[1:]]).strip(".")
            if matched := self._find_by_qualified_suffix(expanded):
                return matched.node_id, 1.0
            if module_id := self._module_target(expanded):
                return module_id, 0.9
            return self.builder.external_node(expanded, kind="external_symbol"), 0.3
        if len(pieces) == 1:
            same_file = [
                node for node in self._symbols_by_name.get(name, []) if node.file_path == path
            ]
            if len(same_file) == 1:
                return same_file[0].node_id, 0.95
            if name in aliases:
                if matched := self._find_by_qualified_suffix(aliases[name]):
                    return matched.node_id, 1.0
            global_matches = self._symbols_by_name.get(name, [])
            if len(global_matches) == 1:
                return global_matches[0].node_id, 0.7
        if matched := self._find_by_qualified_suffix(name):
            return matched.node_id, 0.75
        return None, 0.0
