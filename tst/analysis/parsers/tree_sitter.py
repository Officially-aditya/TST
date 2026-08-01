"""Tree-sitter adapters for Python, JavaScript, TypeScript/TSX, and Rust.

The loader supports the aggregate language packages as well as individually
installed grammar wheels. Tree-sitter remains optional for core installations.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from ..models import (
    ParsedFile,
    ParsedImport,
    ParsedReference,
    ParsedSymbol,
    ReferenceKind,
    SourceSpan,
    SymbolKind,
)
from ..scanner import SourceFile

_JAVASCRIPT_LANGUAGES = {"javascript", "typescript", "tsx"}
_LOAD_ERRORS = (ImportError, AttributeError, LookupError, TypeError, RuntimeError, ValueError)


def _load_direct_parser(language: str) -> Any:
    from tree_sitter import Language, Parser

    module_names = {
        "python": ("tree_sitter_python", "language"),
        "javascript": ("tree_sitter_javascript", "language"),
        "typescript": ("tree_sitter_typescript", "language_typescript"),
        "tsx": ("tree_sitter_typescript", "language_tsx"),
        "rust": ("tree_sitter_rust", "language"),
    }
    module_name, factory_name = module_names[language]
    module = __import__(module_name, fromlist=[factory_name])
    grammar = getattr(module, factory_name)()
    tree_language = grammar if isinstance(grammar, Language) else Language(grammar)
    try:
        return Parser(tree_language)
    except TypeError:
        parser = Parser()
        parser.language = tree_language
        return parser


def _load_parser(language: str) -> Any:
    parser_name = "typescript" if language == "typescript" else language
    last_error: Exception | None = None
    for module_name in ("tree_sitter_language_pack", "tree_sitter_languages"):
        try:
            module = __import__(module_name, fromlist=["get_parser"])
            return module.get_parser(parser_name)
        except _LOAD_ERRORS as exc:
            last_error = exc
    try:
        return _load_direct_parser(language)
    except _LOAD_ERRORS as exc:
        last_error = exc
    raise ImportError(f"no Tree-sitter grammar is available for {language}") from last_error


def _coordinate(point: Any, field: str) -> int:
    if hasattr(point, field):
        return int(getattr(point, field))
    return int(point[0 if field == "row" else 1])


def _span(node: Any) -> SourceSpan:
    start_row = _coordinate(node.start_point, "row")
    start_column = _coordinate(node.start_point, "column")
    end_row = _coordinate(node.end_point, "row")
    end_column = _coordinate(node.end_point, "column")
    end_line = end_row + 1 if end_column or end_row == start_row else end_row
    return SourceSpan(start_row + 1, max(start_row + 1, end_line), start_column, end_column)


def _split_top_level(value: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    quote = ""
    escaped = False
    for index, character in enumerate(value):
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = ""
            continue
        if character in {"'", '"', "`"}:
            quote = character
        elif character in pairs:
            stack.append(pairs[character])
        elif stack and character == stack[-1]:
            stack.pop()
        elif character == delimiter and not stack:
            part = value[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    final = value[start:].strip()
    if final:
        parts.append(final)
    return parts


def _content_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _clean_type_name(value: str) -> str:
    cleaned = re.sub(r"^(?:&(?:'\w+\s*)?|mut\s+|dyn\s+|impl\s+)+", "", value.strip())
    cleaned = cleaned.split("<", 1)[0].strip()
    return cleaned.rsplit("::", 1)[-1].rsplit(".", 1)[-1]


def _parse_python_import(raw: str, line: int) -> list[ParsedImport]:
    compact = re.sub(r"\s+", " ", raw.replace("\\\n", " ")).strip()
    from_match = re.match(r"from\s+([.\w]*)\s+import\s+(.+)$", compact)
    if from_match:
        dotted = from_match.group(1)
        level = len(dotted) - len(dotted.lstrip("."))
        module = dotted[level:]
        imported_names = from_match.group(2).strip()
        if imported_names.startswith("(") and imported_names.endswith(")"):
            imported_names = imported_names[1:-1]
        names: list[str] = []
        aliases: list[tuple[str, str]] = []
        for item in _split_top_level(imported_names):
            item = item.split("#", 1)[0].strip()
            if not item:
                continue
            pieces = re.split(r"\s+as\s+", item, maxsplit=1)
            name = pieces[0].strip()
            names.append(name)
            if len(pieces) == 2:
                aliases.append((name, pieces[1].strip()))
        return [
            ParsedImport(
                module=module,
                names=tuple(names),
                aliases=tuple(aliases),
                level=level,
                line=line,
                raw=raw.strip(),
            )
        ]

    import_match = re.match(r"import\s+(.+)$", compact)
    if not import_match:
        return []
    imports: list[ParsedImport] = []
    for item in _split_top_level(import_match.group(1)):
        pieces = re.split(r"\s+as\s+", item, maxsplit=1)
        module = pieces[0].strip()
        import_aliases = ((module, pieces[1].strip()),) if len(pieces) == 2 else ()
        imports.append(
            ParsedImport(module=module, aliases=import_aliases, line=line, raw=raw.strip())
        )
    return imports


def _parse_javascript_import(raw: str, line: int) -> list[ParsedImport]:
    source_match = re.search(r"(?:\bfrom\s*)?['\"]([^'\"]+)['\"]\s*;?\s*$", raw, re.S)
    if not source_match:
        return []
    module = source_match.group(1)
    prefix = raw[: source_match.start()].strip()
    prefix = re.sub(r"^import\s+(?:type\s+)?", "", prefix).strip()
    if not prefix or prefix == "import":
        return [ParsedImport(module=module, line=line, raw=raw.strip())]

    names: list[str] = []
    aliases: list[tuple[str, str]] = []
    named_match = re.search(r"\{(.*)\}", prefix, re.S)
    namespace_match = re.search(r"\*\s+as\s+([A-Za-z_$][\w$]*)", prefix)
    before_special = re.split(r"[,<{*]", prefix, maxsplit=1)[0].strip()
    if before_special and re.match(r"^[A-Za-z_$][\w$]*$", before_special):
        names.append("default")
        aliases.append(("default", before_special))
    if named_match:
        for item in _split_top_level(named_match.group(1)):
            item = re.sub(r"^(?:type\s+)", "", item.strip())
            pieces = re.split(r"\s+as\s+", item, maxsplit=1)
            name = pieces[0].strip()
            if not name:
                continue
            names.append(name)
            if len(pieces) == 2:
                aliases.append((name, pieces[1].strip()))
    if namespace_match:
        names.append("*")
        aliases.append(("*", namespace_match.group(1)))
    return [
        ParsedImport(
            module=module,
            names=tuple(names),
            aliases=tuple(aliases),
            line=line,
            raw=raw.strip(),
        )
    ]


def _expand_rust_use(value: str, *, grouped: bool = False) -> list[tuple[str, str, bool]]:
    depth = 0
    opening = -1
    closing = -1
    for index, character in enumerate(value):
        if character == "{":
            if depth == 0:
                opening = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                closing = index
                break
    if opening >= 0 and closing > opening:
        base = value[:opening].rstrip(":").strip()
        expanded: list[tuple[str, str, bool]] = []
        for item in _split_top_level(value[opening + 1 : closing]):
            joined = f"{base}::{item}" if base else item
            expanded.extend(_expand_rust_use(joined, grouped=True))
        return expanded
    pieces = re.split(r"\s+as\s+", value.strip(), maxsplit=1)
    return [(pieces[0].strip(), pieces[1].strip() if len(pieces) == 2 else "", grouped)]


def _parse_rust_imports(raw: str, line: int) -> list[ParsedImport]:
    value = re.sub(r"^\s*(?:pub(?:\([^)]*\))?\s+)?use\s+", "", raw).strip().rstrip(";")
    imports: list[ParsedImport] = []
    for path, alias, grouped in _expand_rust_use(value):
        if grouped:
            module, separator, name = path.rpartition("::")
            if name == "self":
                imports.append(
                    ParsedImport(
                        module=module,
                        aliases=((module, alias),) if alias else (),
                        line=line,
                        raw=raw.strip(),
                    )
                )
            elif separator:
                imports.append(
                    ParsedImport(
                        module=module,
                        names=(name,),
                        aliases=((name, alias),) if alias else (),
                        line=line,
                        raw=raw.strip(),
                    )
                )
            continue
        imports.append(
            ParsedImport(
                module=path,
                aliases=((path, alias),) if alias else (),
                line=line,
                raw=raw.strip(),
                metadata={"rust_leaf": True},
            )
        )
    return imports


@dataclass(frozen=True, slots=True)
class _Scope:
    qualified_name: str
    kind: SymbolKind


@dataclass(slots=True)
class _SymbolDetails:
    name: str
    kind: SymbolKind
    declaration: Any
    signature_start: Any
    body: Any | None = None
    bases: tuple[str, ...] = ()
    implements: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class TreeSitterParser:
    name = "tree-sitter"

    def __init__(self, language: str) -> None:
        if language not in {"python", "javascript", "typescript", "tsx", "rust"}:
            raise LookupError(f"unsupported Tree-sitter adapter language: {language}")
        self.language = language
        self._parser = _load_parser(language)

    def parse(self, source: SourceFile, text: str | None = None) -> ParsedFile:
        contents = source.read_text() if text is None else text
        encoded = contents.encode("utf-8")
        tree = self._parser.parse(encoded)
        root = tree.root_node
        result = ParsedFile(
            path=source.relative_path,
            language=source.language,
            content_hash=source.content_hash,
            parser=f"{self.name}:{self.language}",
            line_count=max(1, len(contents.splitlines())),
        )
        if getattr(root, "has_error", False):
            result.errors.append("Tree-sitter found one or more syntax errors")
            result.failed = True
            return result

        pending_implements: list[tuple[str, str]] = []

        def node_text(node: Any | None) -> str:
            if node is None:
                return ""
            return encoded[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

        def child_field(node: Any, field_name: str) -> Any | None:
            try:
                return node.child_by_field_name(field_name)
            except (AttributeError, TypeError):
                return None

        def is_field(parent: Any, child: Any, field_name: str) -> bool:
            field = child_field(parent, field_name)
            return field is not None and field == child

        def header_parts(raw: str, keyword: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
            header = raw.split("{", 1)[0]
            bases: tuple[str, ...] = ()
            implements: tuple[str, ...] = ()
            extends_match = re.search(r"\bextends\s+(.+?)(?=\s+implements\b|$)", header, re.S)
            if extends_match:
                value = extends_match.group(1).strip()
                delimiter = "," if keyword == "interface" else ","
                bases = tuple(item.strip() for item in _split_top_level(value, delimiter) if item)
            implements_match = re.search(r"\bimplements\s+(.+)$", header, re.S)
            if implements_match:
                implements = tuple(
                    item.strip()
                    for item in _split_top_level(implements_match.group(1))
                    if item.strip()
                )
            return bases, implements

        def symbol_details(node: Any, scope: tuple[_Scope, ...]) -> _SymbolDetails | None:
            node_type = node.type
            name_node = child_field(node, "name")
            body = child_field(node, "body")
            declaration = node
            signature_start = node
            metadata: dict[str, Any] = {"node_type": node_type}
            bases: tuple[str, ...] = ()
            implements: tuple[str, ...] = ()
            kind: SymbolKind | None = None

            if self.language == "python":
                if node_type not in {"class_definition", "function_definition"}:
                    return None
                if node_type == "class_definition":
                    kind = SymbolKind.CLASS
                    superclasses = child_field(node, "superclasses")
                    if superclasses is not None:
                        raw_bases = node_text(superclasses).strip()[1:-1]
                        bases = tuple(_split_top_level(raw_bases))
                else:
                    kind = (
                        SymbolKind.METHOD
                        if scope and scope[-1].kind == SymbolKind.CLASS
                        else SymbolKind.FUNCTION
                    )
                parent = getattr(node, "parent", None)
                if parent is not None and parent.type == "decorated_definition":
                    declaration = parent
                    decorators = [
                        node_text(child).strip()
                        for child in parent.named_children
                        if child.type == "decorator"
                    ]
                    metadata["decorators"] = decorators
            elif self.language in _JAVASCRIPT_LANGUAGES:
                kinds = {
                    "class_declaration": SymbolKind.CLASS,
                    "abstract_class_declaration": SymbolKind.CLASS,
                    "function_declaration": SymbolKind.FUNCTION,
                    "generator_function_declaration": SymbolKind.FUNCTION,
                    "method_definition": SymbolKind.METHOD,
                    "method_signature": SymbolKind.METHOD,
                    "interface_declaration": SymbolKind.INTERFACE,
                }
                kind = kinds.get(node_type)
                if node_type == "variable_declarator":
                    value = child_field(node, "value")
                    if value is None or value.type not in {
                        "arrow_function",
                        "function_expression",
                        "generator_function",
                    }:
                        return None
                    kind = SymbolKind.FUNCTION
                    body = child_field(value, "body")
                    parent = getattr(node, "parent", None)
                    if parent is not None:
                        declaration = parent
                        signature_start = parent
                if kind is None:
                    return None
                raw = node_text(node)
                if kind in {SymbolKind.CLASS, SymbolKind.INTERFACE}:
                    bases, implements = header_parts(
                        raw, "interface" if kind == SymbolKind.INTERFACE else "class"
                    )
                parent = getattr(node, "parent", None)
                if parent is not None and parent.type == "export_statement":
                    export_header = node_text(parent).lstrip()
                    metadata["exported"] = True
                    metadata["default_export"] = bool(
                        re.match(r"export\s+default\b", export_header)
                    )
            else:
                kinds = {
                    "function_item": SymbolKind.FUNCTION,
                    "function_signature_item": SymbolKind.METHOD,
                    "struct_item": SymbolKind.STRUCT,
                    "enum_item": SymbolKind.ENUM,
                    "trait_item": SymbolKind.TRAIT,
                    "mod_item": SymbolKind.MODULE,
                }
                kind = kinds.get(node_type)
                if kind is None:
                    return None
                if node_type in {"function_item", "function_signature_item"} and scope:
                    if scope[-1].kind in {
                        SymbolKind.CLASS,
                        SymbolKind.STRUCT,
                        SymbolKind.ENUM,
                        SymbolKind.TRAIT,
                        SymbolKind.INTERFACE,
                    }:
                        kind = SymbolKind.METHOD
                if node_type == "trait_item":
                    header = node_text(node).split("{", 1)[0]
                    supertraits = re.search(r"\btrait\s+\w+(?:<.*?>)?\s*:\s*(.+)$", header)
                    if supertraits:
                        bases = tuple(
                            item.strip()
                            for item in _split_top_level(supertraits.group(1).replace(" + ", ","))
                            if item.strip()
                        )

            if kind is None:
                return None

            name = node_text(name_node).strip()
            if not name and self.language in _JAVASCRIPT_LANGUAGES:
                parent = getattr(node, "parent", None)
                if parent is not None and re.match(r"export\s+default\b", node_text(parent)):
                    name = "default"
                    metadata["exported"] = True
                    metadata["default_export"] = True
            if not name:
                return None
            return _SymbolDetails(
                name=name,
                kind=kind,
                declaration=declaration,
                signature_start=signature_start,
                body=body,
                bases=bases,
                implements=implements,
                metadata=metadata,
            )

        def add_reference(symbol: ParsedSymbol, name: str, kind: ReferenceKind, line: int) -> None:
            normalized = name.strip().replace("?.", ".").replace("::", ".")
            if not normalized:
                return
            key = (normalized, kind, line)
            if any((item.name, item.kind, item.line) == key for item in symbol.references):
                return
            symbol.references.append(ParsedReference(normalized, kind, line))

        def declaration_identifier(node: Any) -> bool:
            parent = getattr(node, "parent", None)
            if parent is None:
                return True
            if parent.type in {
                "import_statement",
                "import_from_statement",
                "import_declaration",
                "use_declaration",
            }:
                return True
            ancestor = parent
            while ancestor is not None:
                if ancestor.type in {
                    "import_statement",
                    "import_from_statement",
                    "import_declaration",
                    "use_declaration",
                }:
                    return True
                if ancestor.type in {
                    "class_heritage",
                    "extends_clause",
                    "extends_type_clause",
                    "implements_clause",
                    "superclass",
                }:
                    return True
                if ancestor.type in {
                    "function_definition",
                    "function_declaration",
                    "method_definition",
                    "function_item",
                }:
                    break
                ancestor = getattr(ancestor, "parent", None)
            if parent.type in {
                "formal_parameters",
                "parameters",
                "self_parameter",
                "object_pattern",
                "array_pattern",
            }:
                return True
            if parent.type in {
                "required_parameter",
                "optional_parameter",
                "parameter",
                "typed_parameter",
                "default_parameter",
            } and (is_field(parent, node, "pattern") or not child_field(parent, "type")):
                return True
            if parent.type in {
                "class_definition",
                "class_declaration",
                "abstract_class_declaration",
                "interface_declaration",
                "function_definition",
                "function_declaration",
                "method_definition",
                "method_signature",
                "function_item",
                "function_signature_item",
                "struct_item",
                "enum_item",
                "trait_item",
                "mod_item",
                "variable_declarator",
                "property_signature",
            } and is_field(parent, node, "name"):
                return True
            if parent.type in {"assignment", "assignment_expression", "let_declaration"} and (
                is_field(parent, node, "left") or is_field(parent, node, "pattern")
            ):
                return True
            if parent.type in {"attribute", "member_expression", "scoped_identifier"}:
                return True
            return False

        def visit(
            node: Any,
            scope: tuple[_Scope, ...],
            current: ParsedSymbol | None,
        ) -> None:
            raw = node_text(node)
            line = _coordinate(node.start_point, "row") + 1

            if self.language == "python" and node.type in {
                "import_statement",
                "import_from_statement",
            }:
                result.imports.extend(_parse_python_import(raw, line))
                return
            if self.language in _JAVASCRIPT_LANGUAGES and node.type == "import_statement":
                result.imports.extend(_parse_javascript_import(raw, line))
                return
            if self.language in _JAVASCRIPT_LANGUAGES and node.type == "export_statement":
                if child_field(node, "source") is not None:
                    import_raw = re.sub(r"^\s*export\s+", "import ", raw, count=1)
                    result.imports.extend(_parse_javascript_import(import_raw, line))
            if self.language == "rust" and node.type == "use_declaration":
                result.imports.extend(_parse_rust_imports(raw, line))
                return

            if self.language in _JAVASCRIPT_LANGUAGES and node.type == "variable_declarator":
                value = child_field(node, "value")
                if value is not None and value.type == "call_expression":
                    function = child_field(value, "function")
                    if node_text(function).strip() == "require":
                        module_match = re.search(r"require\s*\(\s*['\"]([^'\"]+)['\"]", raw)
                        name_node = child_field(node, "name")
                        if module_match and name_node is not None:
                            binding = node_text(name_node).strip()
                            result.imports.append(
                                ParsedImport(
                                    module=module_match.group(1),
                                    aliases=(("*", binding),),
                                    names=("*",),
                                    line=line,
                                    raw=raw.strip(),
                                )
                            )

            if self.language == "rust" and node.type == "impl_item":
                type_node = child_field(node, "type")
                trait_node = child_field(node, "trait")
                target_name = _clean_type_name(node_text(type_node))
                if not target_name:
                    return
                target_qualified = (
                    f"{scope[-1].qualified_name}.{target_name}" if scope else target_name
                )
                trait_name = _clean_type_name(node_text(trait_node))
                if trait_name:
                    pending_implements.append((target_qualified, trait_name))
                impl_scope = (*scope, _Scope(target_qualified, SymbolKind.STRUCT))
                body = child_field(node, "body")
                if body is not None:
                    for child in body.named_children:
                        visit(child, impl_scope, current)
                return

            details = symbol_details(node, scope)
            symbol: ParsedSymbol | None = None
            active_scope = scope
            active_current = current
            if details is not None:
                qualified = f"{scope[-1].qualified_name}.{details.name}" if scope else details.name
                signature_end = (
                    details.body.start_byte
                    if details.body is not None
                    else details.signature_start.end_byte
                )
                signature_raw = encoded[details.signature_start.start_byte : signature_end].decode(
                    "utf-8", errors="replace"
                )
                signature = re.sub(r"\s+", " ", signature_raw).strip().rstrip(";{").strip()
                declaration_raw = node_text(details.declaration)
                symbol = ParsedSymbol(
                    name=details.name,
                    qualified_name=qualified,
                    parent_qualified_name=scope[-1].qualified_name if scope else None,
                    kind=details.kind,
                    span=_span(details.declaration),
                    signature=signature or None,
                    bases=details.bases,
                    implements=details.implements,
                    content_hash=_content_hash(declaration_raw),
                    metadata=details.metadata,
                )
                result.symbols.append(symbol)
                active_scope = (*scope, _Scope(qualified, details.kind))
                active_current = symbol
                if self.language == "rust" and node.type == "mod_item":
                    if child_field(node, "body") is None:
                        result.imports.append(
                            ParsedImport(
                                module=details.name,
                                line=line,
                                raw=raw.strip(),
                                metadata={"module_declaration": True},
                            )
                        )

            call_types = (
                {"call"}
                if self.language == "python"
                else {"call_expression", "new_expression"}
                if self.language in _JAVASCRIPT_LANGUAGES
                else {"call_expression"}
            )
            call_function: Any | None = None
            if node.type in call_types and active_current is not None:
                call_function = child_field(node, "function") or child_field(node, "constructor")
                if call_function is None and node.named_children:
                    call_function = node.named_children[0]
                call_name = node_text(call_function).strip()
                if call_name:
                    add_reference(active_current, call_name, ReferenceKind.CALL, line)

            composite_types = {"attribute", "member_expression", "scoped_identifier"}
            if node.type in composite_types and active_current is not None:
                parent = getattr(node, "parent", None)
                is_call_target = bool(
                    parent is not None
                    and parent.type in call_types
                    and (
                        is_field(parent, node, "function") or is_field(parent, node, "constructor")
                    )
                )
                if not is_call_target:
                    add_reference(active_current, raw, ReferenceKind.REFERENCE, line)
                return

            if (
                node.type in {"identifier", "type_identifier"}
                and active_current is not None
                and not declaration_identifier(node)
            ):
                add_reference(active_current, raw, ReferenceKind.REFERENCE, line)

            for child in node.named_children:
                if call_function is not None and child == call_function:
                    continue
                visit(child, active_scope, active_current)

        visit(root, (), None)

        for target, trait in pending_implements:
            matches = [
                symbol
                for symbol in result.symbols
                if symbol.qualified_name == target
                or symbol.qualified_name.endswith(f".{target}")
                or symbol.name == target.rsplit(".", 1)[-1]
            ]
            if len(matches) == 1 and trait not in matches[0].implements:
                matches[0].implements = (*matches[0].implements, trait)
        return result
