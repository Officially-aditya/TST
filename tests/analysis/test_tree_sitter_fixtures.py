from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tst.analysis.graph_builder import EdgeKind
from tst.analysis.incremental import IncrementalIndexer
from tst.analysis.models import ReferenceKind
from tst.analysis.parsers.tree_sitter import TreeSitterParser
from tst.analysis.scanner import SourceFile

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "analysis"


def _parser(language: str) -> TreeSitterParser:
    try:
        return TreeSitterParser(language)
    except ImportError:
        pytest.skip(f"Tree-sitter {language} grammar is not installed")


def _source(relative: str, language: str) -> SourceFile:
    path = FIXTURES / relative
    contents = path.read_bytes()
    return SourceFile(
        path,
        relative,
        language,
        len(contents),
        hashlib.sha256(contents).hexdigest(),
    )


@pytest.mark.parametrize(
    ("relative", "language", "expected_symbols"),
    [
        ("python/pkg/service.py", "python", {"Service", "Service.run"}),
        ("javascript/app.js", "javascript", {"run"}),
        ("typescript/app.ts", "typescript", {"Runnable", "Service", "Service.run"}),
        ("tsx/component.tsx", "tsx", {"Props", "Card"}),
        ("rust/src/lib.rs", "rust", {"Runnable", "Service", "Service.run"}),
    ],
)
def test_real_language_fixtures_use_tree_sitter(
    relative: str, language: str, expected_symbols: set[str]
) -> None:
    parsed = _parser(language).parse(_source(relative, language))
    assert parsed.failed is False, parsed.errors
    assert parsed.parser == f"tree-sitter:{language}"
    assert expected_symbols <= {symbol.qualified_name for symbol in parsed.symbols}
    assert all(symbol.content_hash for symbol in parsed.symbols)

    if language == "javascript":
        imports = parsed.imports[0]
        assert ("default", "helper") in imports.aliases
        assert ("twice", "double") in imports.aliases
        run = next(symbol for symbol in parsed.symbols if symbol.name == "run")
        calls = {ref.name for ref in run.references if ref.kind == ReferenceKind.CALL}
        assert {"helper", "double"} <= calls
    if language == "typescript":
        service = next(symbol for symbol in parsed.symbols if symbol.name == "Service")
        assert service.bases == ("Widget",)
        assert service.implements == ("Runnable",)
    if language == "rust":
        service = next(symbol for symbol in parsed.symbols if symbol.name == "Service")
        assert service.implements == ("Runnable",)


def test_cross_file_aliases_inheritance_and_trait_edges_resolve() -> None:
    for language in ("python", "javascript", "typescript", "tsx", "rust"):
        _parser(language)

    indexer = IncrementalIndexer(FIXTURES)
    report = indexer.index()
    assert not report.parse_errors, report.parse_errors
    graph = indexer.graph

    ts_run = graph.find("typescript.app.Service.run")[0]
    helper = graph.find("typescript.helper.helper")[0]
    formatter = graph.find("typescript.util.format")[0]
    ts_calls = {edge.target for edge in graph.outgoing(ts_run.node_id, {EdgeKind.CALLS})}
    assert {helper.node_id, formatter.node_id} <= ts_calls

    service = graph.find("typescript.app.Service")[0]
    widget = graph.find("typescript.helper.Widget")[0]
    runnable = graph.find("typescript.app.Runnable")[0]
    assert any(
        edge.target == widget.node_id
        for edge in graph.outgoing(service.node_id, {EdgeKind.INHERITS})
    )
    assert any(
        edge.target == runnable.node_id
        for edge in graph.outgoing(service.node_id, {EdgeKind.IMPLEMENTS})
    )

    rust_run = graph.find("rust.src.Service.run")[0]
    worker_run = graph.find("rust.src.worker.run")[0]
    assert any(
        edge.target == worker_run.node_id
        for edge in graph.outgoing(rust_run.node_id, {EdgeKind.CALLS})
    )
    assert graph.validate() == []
