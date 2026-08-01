"""Line-aware structural fallback for platforms without Tree-sitter wheels.

This adapter is intentionally conservative. It exists to retain basic indexing
for unsupported languages; supported Python always uses ``ast`` and the JS/TS/
Rust analysis extra upgrades these languages to Tree-sitter.
"""

from __future__ import annotations

import hashlib
import re

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

_CALL_EXCLUSIONS = {
    "catch",
    "class",
    "def",
    "else",
    "for",
    "function",
    "if",
    "match",
    "return",
    "sizeof",
    "switch",
    "while",
}


def _brace_end(lines: list[str], start: int) -> int:
    depth = 0
    opened = False
    for index in range(start, len(lines)):
        # This is deliberately a fallback heuristic, not the primary parser.
        code = lines[index].split("//", 1)[0]
        depth += code.count("{")
        if "{" in code:
            opened = True
        depth -= code.count("}")
        if opened and depth <= 0:
            return index + 1
    return start + 1


class StructuralFallbackParser:
    name = "structural-fallback"

    def __init__(self, language: str) -> None:
        self.language = language

    def parse(self, source: SourceFile, text: str | None = None) -> ParsedFile:
        contents = source.read_text() if text is None else text
        lines = contents.splitlines()
        result = ParsedFile(
            path=source.relative_path,
            language=source.language,
            content_hash=source.content_hash,
            parser=self.name,
            line_count=max(1, len(lines)),
        )
        if self.language in {"javascript", "typescript", "tsx"}:
            self._parse_javascript(lines, result)
        elif self.language == "rust":
            self._parse_rust(lines, result)
        else:
            result.errors.append(f"no syntax parser is registered for {self.language}")
            result.failed = True
        self._assign_parents(result.symbols)
        self._collect_calls(lines, result.symbols)
        return result

    def _parse_javascript(self, lines: list[str], result: ParsedFile) -> None:
        import_re = re.compile(
            r"(?:import\s+(?P<names>.*?)\s+from\s+|import\s+|require\s*\()"
            r"['\"](?P<module>[^'\"]+)['\"]"
        )
        declarations = (
            (
                re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\("),
                SymbolKind.FUNCTION,
            ),
            (re.compile(r"^\s*(?:export\s+)?class\s+(\w+)"), SymbolKind.CLASS),
            (re.compile(r"^\s*(?:export\s+)?interface\s+(\w+)"), SymbolKind.INTERFACE),
            (
                re.compile(
                    r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*"
                    r"(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>"
                ),
                SymbolKind.FUNCTION,
            ),
        )
        for index, line in enumerate(lines):
            if match := import_re.search(line):
                names = tuple(
                    item.strip().split(" as ")[0]
                    for item in (match.groupdict().get("names") or "").strip("{} ").split(",")
                    if item.strip()
                )
                result.imports.append(
                    ParsedImport(
                        module=match.group("module"),
                        names=names,
                        line=index + 1,
                        raw=line.strip(),
                    )
                )
            for pattern, kind in declarations:
                match = pattern.search(line)
                if match:
                    end = _brace_end(lines, index)
                    span = SourceSpan(index + 1, max(index + 1, end))
                    result.symbols.append(
                        ParsedSymbol(
                            name=match.group(1),
                            qualified_name=match.group(1),
                            kind=kind,
                            span=span,
                            signature=line.strip().split("{", 1)[0].strip(),
                            content_hash=self._content_hash(lines, span),
                            metadata={"node_type": "fallback_declaration"},
                        )
                    )
                    break

    def _parse_rust(self, lines: list[str], result: ParsedFile) -> None:
        use_re = re.compile(r"^\s*use\s+([^;]+)")
        declarations = (
            (
                re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)"),
                SymbolKind.FUNCTION,
            ),
            (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)"), SymbolKind.STRUCT),
            (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?enum\s+(\w+)"), SymbolKind.ENUM),
            (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?trait\s+(\w+)"), SymbolKind.TRAIT),
            (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)"), SymbolKind.MODULE),
        )
        for index, line in enumerate(lines):
            if match := use_re.search(line):
                result.imports.append(
                    ParsedImport(module=match.group(1).strip(), line=index + 1, raw=line.strip())
                )
            for pattern, kind in declarations:
                match = pattern.search(line)
                if match:
                    end = _brace_end(lines, index)
                    span = SourceSpan(index + 1, max(index + 1, end))
                    result.symbols.append(
                        ParsedSymbol(
                            name=match.group(1),
                            qualified_name=match.group(1),
                            kind=kind,
                            span=span,
                            signature=line.strip().split("{", 1)[0].strip(),
                            content_hash=self._content_hash(lines, span),
                            metadata={"node_type": "fallback_declaration"},
                        )
                    )
                    break

    def _assign_parents(self, symbols: list[ParsedSymbol]) -> None:
        containers = sorted(symbols, key=lambda item: (item.span.start_line, -item.span.end_line))
        for symbol in containers:
            candidates = [
                parent
                for parent in containers
                if parent is not symbol
                and parent.span.start_line <= symbol.span.start_line
                and parent.span.end_line >= symbol.span.end_line
                and parent.kind
                in {SymbolKind.CLASS, SymbolKind.STRUCT, SymbolKind.TRAIT, SymbolKind.MODULE}
            ]
            if candidates:
                parent = min(candidates, key=lambda item: item.span.end_line - item.span.start_line)
                symbol.parent_qualified_name = parent.qualified_name
                symbol.qualified_name = f"{parent.qualified_name}.{symbol.name}"
                if (
                    parent.kind in {SymbolKind.CLASS, SymbolKind.STRUCT, SymbolKind.TRAIT}
                    and symbol.kind == SymbolKind.FUNCTION
                ):
                    symbol.kind = SymbolKind.METHOD

    def _collect_calls(self, lines: list[str], symbols: list[ParsedSymbol]) -> None:
        call_re = re.compile(r"\b([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(")
        for symbol in symbols:
            for line_no in range(symbol.span.start_line, min(symbol.span.end_line, len(lines)) + 1):
                for match in call_re.finditer(lines[line_no - 1]):
                    name = match.group(1)
                    if name.rsplit(".", 1)[-1] not in _CALL_EXCLUSIONS and name != symbol.name:
                        symbol.references.append(
                            ParsedReference(name=name, kind=ReferenceKind.CALL, line=line_no)
                        )

    def _content_hash(self, lines: list[str], span: SourceSpan) -> str:
        contents = "\n".join(lines[span.start_line - 1 : span.end_line])
        return hashlib.sha256(contents.encode("utf-8")).hexdigest()
