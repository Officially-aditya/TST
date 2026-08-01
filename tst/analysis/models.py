"""Transport-neutral models emitted by source parser adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SymbolKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    STRUCT = "struct"
    ENUM = "enum"
    TRAIT = "trait"
    INTERFACE = "interface"
    VARIABLE = "variable"
    SYMBOL = "symbol"


class ReferenceKind(str, Enum):
    CALL = "call"
    REFERENCE = "reference"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"


@dataclass(frozen=True, slots=True)
class SourceSpan:
    start_line: int
    end_line: int
    start_column: int = 0
    end_column: int = 0

    def __post_init__(self) -> None:
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("source spans use positive, ordered line numbers")
        if self.start_column < 0 or self.end_column < 0:
            raise ValueError("source span columns cannot be negative")
        if self.start_line == self.end_line and self.end_column < self.start_column:
            raise ValueError("single-line source span columns must be ordered")


@dataclass(frozen=True, slots=True)
class ParsedReference:
    name: str
    kind: ReferenceKind
    line: int


@dataclass(frozen=True, slots=True)
class ParsedImport:
    """An import before repository-local resolution.

    ``module`` excludes leading dots. ``level`` follows Python AST semantics:
    zero means absolute, one means the current package, and so on.
    """

    module: str
    names: tuple[str, ...] = ()
    aliases: tuple[tuple[str, str], ...] = ()
    level: int = 0
    line: int = 1
    raw: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedSymbol:
    name: str
    qualified_name: str
    kind: SymbolKind
    span: SourceSpan
    signature: str | None = None
    parent_qualified_name: str | None = None
    references: list[ParsedReference] = field(default_factory=list)
    bases: tuple[str, ...] = ()
    implements: tuple[str, ...] = ()
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedFile:
    path: str
    language: str
    content_hash: str
    parser: str
    symbols: list[ParsedSymbol] = field(default_factory=list)
    imports: list[ParsedImport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    line_count: int = 1
    failed: bool = False
