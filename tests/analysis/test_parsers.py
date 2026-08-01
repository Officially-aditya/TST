from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from tst.analysis.models import ReferenceKind, SymbolKind
from tst.analysis.parsers.fallback import StructuralFallbackParser
from tst.analysis.parsers.python_ast import PythonAstParser
from tst.analysis.scanner import SourceFile


def source_file(root: Path, name: str, language: str, text: str) -> SourceFile:
    path = root / name
    path.write_text(text, encoding="utf-8")
    encoded = text.encode()
    return SourceFile(path, name, language, len(encoded), hashlib.sha256(encoded).hexdigest())


class ParserTests(unittest.TestCase):
    def test_python_ast_tracks_qualified_symbols_lines_imports_and_calls(self) -> None:
        code = """from .helpers import work as do_work
import json

class Service(Base):
    async def run(self, value: int) -> str:
        return do_work(value)

def top():
    return Service()
"""
        with tempfile.TemporaryDirectory() as directory:
            source = source_file(Path(directory), "service.py", "python", code)
            parsed = PythonAstParser().parse(source)

        symbols = {item.qualified_name: item for item in parsed.symbols}
        self.assertEqual(parsed.parser, "python-ast")
        self.assertEqual(symbols["Service"].span.start_line, 4)
        self.assertEqual(symbols["Service"].span.end_line, 6)
        self.assertEqual(symbols["Service.run"].kind, SymbolKind.METHOD)
        self.assertEqual(symbols["Service.run"].signature, "async def run(self, value: int) -> str")
        self.assertEqual(symbols["Service"].bases, ("Base",))
        self.assertIn(
            ("do_work", ReferenceKind.CALL, 6),
            {(ref.name, ref.kind, ref.line) for ref in symbols["Service.run"].references},
        )
        self.assertEqual(parsed.imports[0].module, "helpers")
        self.assertEqual(parsed.imports[0].level, 1)
        self.assertEqual(parsed.imports[0].aliases, (("work", "do_work"),))

    def test_python_syntax_errors_are_reported_without_partial_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = source_file(Path(directory), "broken.py", "python", "def broken(:\n")
            parsed = PythonAstParser().parse(source)
        self.assertEqual(parsed.symbols, [])
        self.assertIn("SyntaxError", parsed.errors[0])

    def test_structural_fallback_is_line_aware(self) -> None:
        code = """import { helper } from './helper';
export function run(value) {
  return helper(value);
}
const later = (x) => {
  return run(x);
};
"""
        with tempfile.TemporaryDirectory() as directory:
            source = source_file(Path(directory), "app.ts", "typescript", code)
            parsed = StructuralFallbackParser("typescript").parse(source)
        symbols = {item.name: item for item in parsed.symbols}
        self.assertEqual(parsed.parser, "structural-fallback")
        self.assertEqual((symbols["run"].span.start_line, symbols["run"].span.end_line), (2, 4))
        self.assertEqual((symbols["later"].span.start_line, symbols["later"].span.end_line), (5, 7))
        self.assertEqual(parsed.imports[0].module, "./helper")


if __name__ == "__main__":
    unittest.main()
