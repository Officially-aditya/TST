"""Parser adapter protocol and dependency-aware registry."""

from __future__ import annotations

from typing import Protocol

from ..models import ParsedFile
from ..scanner import SourceFile


class ParserAdapter(Protocol):
    name: str

    def parse(self, source: SourceFile, text: str | None = None) -> ParsedFile: ...


class ParserRegistry:
    """Choose the most precise parser available without mandatory native deps."""

    def __init__(self) -> None:
        self._parsers: dict[str, ParserAdapter] = {}
        self._fallbacks: dict[str, ParserAdapter] = {}

        from .fallback import StructuralFallbackParser
        from .python_ast import PythonAstParser

        self._fallbacks["python"] = PythonAstParser()
        for language in ("javascript", "typescript", "tsx", "rust"):
            self._fallbacks[language] = StructuralFallbackParser(language)

        # Tree-sitter is an analysis extra. Importing TST must continue to work
        # when native parser wheels are unavailable for a platform.
        try:
            from .tree_sitter import TreeSitterParser

            for language in ("python", "javascript", "typescript", "tsx", "rust"):
                try:
                    self.register(language, TreeSitterParser(language))
                except (ImportError, LookupError, RuntimeError):
                    pass
        except ImportError:
            pass

    def register(self, language: str, parser: ParserAdapter) -> None:
        self._parsers[language] = parser

    def get(self, language: str) -> ParserAdapter:
        parser = self._parsers.get(language) or self._fallbacks.get(language)
        if parser is None:
            from .fallback import StructuralFallbackParser

            parser = StructuralFallbackParser(language)
            self._fallbacks[language] = parser
        return parser

    def parse(self, source: SourceFile, text: str | None = None) -> ParsedFile:
        parser = self.get(source.language)
        fallback = self._fallbacks.get(source.language)
        try:
            return parser.parse(source, text)
        except Exception as exc:
            if fallback is None or fallback is parser:
                raise
            parsed = fallback.parse(source, text)
            parsed.errors.insert(
                0,
                f"{parser.name} failed; used {fallback.name}: {type(exc).__name__}: {exc}",
            )
            return parsed
