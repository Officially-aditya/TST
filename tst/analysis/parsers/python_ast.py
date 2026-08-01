"""Python parser backed by the standard library concrete source AST."""

from __future__ import annotations

import ast
import hashlib
from collections import defaultdict

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


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _dotted_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Subscript):
        return _dotted_name(node.value)
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return ""


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(base) for base in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({args}){returns}"


def _source_hash(contents: str, span: SourceSpan) -> str:
    lines = contents.splitlines(keepends=True)
    selected = lines[span.start_line - 1 : span.end_line]
    if not selected:
        return hashlib.sha256(b"").hexdigest()
    selected[0] = selected[0].encode("utf-8")[span.start_column :].decode("utf-8", errors="ignore")
    if span.start_line == span.end_line:
        width = max(0, span.end_column - span.start_column)
        selected[-1] = selected[-1].encode("utf-8")[:width].decode("utf-8", errors="ignore")
    elif span.end_column:
        selected[-1] = (
            selected[-1].encode("utf-8")[: span.end_column].decode("utf-8", errors="ignore")
        )
    return hashlib.sha256("".join(selected).encode("utf-8")).hexdigest()


class _PythonVisitor(ast.NodeVisitor):
    def __init__(self, contents: str) -> None:
        self.contents = contents
        self.imports: list[ParsedImport] = []
        self.symbols: list[ParsedSymbol] = []
        self._scope: list[ParsedSymbol] = []
        self._references: dict[str, list[ParsedReference]] = defaultdict(list)

    @property
    def current(self) -> ParsedSymbol | None:
        return self._scope[-1] if self._scope else None

    def _qualified(self, name: str) -> str:
        if not self._scope:
            return name
        return f"{self._scope[-1].qualified_name}.{name}"

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ParsedImport(
                    module=alias.name,
                    aliases=((alias.name, alias.asname),) if alias.asname else (),
                    line=node.lineno,
                    raw=ast.unparse(node),
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        aliases = tuple(
            (alias.name, alias.asname) for alias in node.names if alias.asname is not None
        )
        self.imports.append(
            ParsedImport(
                module=node.module or "",
                names=tuple(alias.name for alias in node.names),
                aliases=aliases,
                level=node.level,
                line=node.lineno,
                raw=ast.unparse(node),
            )
        )

    def _visit_symbol(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> None:
        parent = self.current
        if isinstance(node, ast.ClassDef):
            kind = SymbolKind.CLASS
            bases = tuple(filter(None, (_dotted_name(base) for base in node.bases)))
        else:
            kind = (
                SymbolKind.METHOD
                if parent and parent.kind == SymbolKind.CLASS
                else SymbolKind.FUNCTION
            )
            bases = ()
        decorators = list(node.decorator_list)
        start_line = min((item.lineno for item in decorators), default=node.lineno)
        start_column = (
            min(item.col_offset for item in decorators)
            if decorators and start_line < node.lineno
            else node.col_offset
        )
        span = SourceSpan(
            start_line=start_line,
            end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
            start_column=start_column,
            end_column=getattr(node, "end_col_offset", 0) or 0,
        )
        symbol = ParsedSymbol(
            name=node.name,
            qualified_name=self._qualified(node.name),
            parent_qualified_name=parent.qualified_name if parent else None,
            kind=kind,
            span=span,
            signature=_signature(node),
            bases=bases,
            content_hash=_source_hash(self.contents, span),
            metadata={
                "decorators": [ast.unparse(item) for item in decorators],
                "node_type": type(node).__name__,
            },
        )
        self.symbols.append(symbol)
        self._scope.append(symbol)
        self.generic_visit(node)
        self._scope.pop()
        symbol.references.extend(self._references.pop(symbol.qualified_name, []))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_symbol(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_symbol(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_symbol(node)

    def visit_Call(self, node: ast.Call) -> None:
        current = self.current
        name = _dotted_name(node.func)
        if current is not None and name:
            self._references[current.qualified_name].append(
                ParsedReference(name=name, kind=ReferenceKind.CALL, line=node.lineno)
            )
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node: ast.Name) -> None:
        current = self.current
        if current is not None and isinstance(node.ctx, ast.Load):
            self._references[current.qualified_name].append(
                ParsedReference(name=node.id, kind=ReferenceKind.REFERENCE, line=node.lineno)
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        current = self.current
        name = _dotted_name(node)
        if current is not None and isinstance(node.ctx, ast.Load) and name:
            self._references[current.qualified_name].append(
                ParsedReference(name=name, kind=ReferenceKind.REFERENCE, line=node.lineno)
            )
        self.visit(node.value)


class PythonAstParser:
    name = "python-ast"

    def parse(self, source: SourceFile, text: str | None = None) -> ParsedFile:
        contents = source.read_text() if text is None else text
        result = ParsedFile(
            path=source.relative_path,
            language=source.language,
            content_hash=source.content_hash,
            parser=self.name,
            line_count=max(1, len(contents.splitlines())),
        )
        try:
            tree = ast.parse(contents, filename=source.relative_path, type_comments=True)
        except (SyntaxError, ValueError) as exc:
            line = getattr(exc, "lineno", None)
            location = f" at line {line}" if line else ""
            result.errors.append(f"{type(exc).__name__}{location}: {exc}")
            result.failed = True
            return result

        visitor = _PythonVisitor(contents)
        visitor.visit(tree)
        result.imports.extend(visitor.imports)
        result.symbols.extend(visitor.symbols)
        return result
