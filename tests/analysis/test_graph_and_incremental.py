from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tst.analysis.graph_builder import EdgeKind
from tst.analysis.incremental import IncrementalIndexer
from tst.analysis.parsers.base import ParserRegistry


class GraphAndIncrementalTests(unittest.TestCase):
    def test_import_call_and_test_edges_are_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pkg").mkdir()
            (root / "tests").mkdir()
            (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (root / "pkg" / "worker.py").write_text(
                "def work(value: int) -> int:\n    return value + 1\n", encoding="utf-8"
            )
            (root / "pkg" / "service.py").write_text(
                "from .worker import work\n\ndef run():\n    return work(1)\n", encoding="utf-8"
            )
            (root / "tests" / "test_service.py").write_text(
                "from pkg.service import run\n\ndef test_run():\n    assert run() == 2\n",
                encoding="utf-8",
            )
            indexer = IncrementalIndexer(root)
            report = indexer.index()

            self.assertEqual(report.parsed_files, 4)
            self.assertEqual(indexer.graph.validate(), [])
            run = indexer.graph.find("pkg.service.run")[0]
            work = indexer.graph.find("pkg.worker.work")[0]
            calls = indexer.graph.outgoing(run.node_id, {EdgeKind.CALLS})
            self.assertTrue(any(edge.target == work.node_id for edge in calls))
            self.assertTrue(any(edge.kind == EdgeKind.TESTS for edge in indexer.graph.edges))

            context = indexer.graph.query(work.node_id, depth=1, max_nodes=10)
            self.assertIn("pkg.service.run", {node.qualified_name for node in context.nodes})

    def test_unchanged_files_are_not_reparsed_and_deletion_has_no_dangling_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.py"
            second = root / "second.py"
            first.write_text(
                "from second import helper\n\ndef run():\n    return helper()\n", encoding="utf-8"
            )
            second.write_text("def helper():\n    return 1\n", encoding="utf-8")
            indexer = IncrementalIndexer(root)

            initial = indexer.index()
            unchanged = indexer.index()
            self.assertEqual(initial.parsed_files, 2)
            self.assertEqual(unchanged.parsed_files, 0)
            self.assertEqual(unchanged.unchanged, ["first.py", "second.py"])

            second.write_text("def helper():\n    return 2\n", encoding="utf-8")
            changed = indexer.index()
            self.assertEqual(changed.changed, ["second.py"])
            self.assertEqual(changed.parsed_files, 1)

            second.unlink()
            removed = indexer.index()
            self.assertEqual(removed.removed, ["second.py"])
            self.assertEqual(indexer.graph.validate(), [])
            self.assertFalse(
                any(node.file_path == "second.py" for node in indexer.graph.nodes.values())
            )
            self.assertFalse(
                any(
                    edge.target not in indexer.graph.nodes or edge.source not in indexer.graph.nodes
                    for edge in indexer.graph.edges
                )
            )

    def test_failed_changed_parse_keeps_last_good_file_transactionally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "app.py"
            source.write_text("def run():\n    return 1\n", encoding="utf-8")
            indexer = IncrementalIndexer(root)
            indexer.index()
            previous_hash = indexer.hashes["app.py"]
            previous_node = indexer.graph.find("app.run")[0].node_id

            source.write_text("def run(:\n", encoding="utf-8")
            failed = indexer.index()

            self.assertIn("app.py", failed.parse_errors)
            self.assertEqual(indexer.hashes["app.py"], previous_hash)
            self.assertEqual(indexer.graph.find("app.run")[0].node_id, previous_node)
            self.assertEqual(indexer.graph.validate(), [])
            self.assertEqual(indexer.index().changed, ["app.py"])

    def test_runtime_primary_parser_failure_retries_safe_fallback(self) -> None:
        class BrokenParser:
            name = "broken-primary"

            def parse(self, source, text=None):
                raise RuntimeError("parser runtime failed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
            registry = ParserRegistry()
            registry.register("python", BrokenParser())
            indexer = IncrementalIndexer(root, registry=registry)
            report = indexer.index()

            self.assertEqual(report.parsed_files, 1)
            self.assertIn("broken-primary failed", report.parse_errors["app.py"][0])
            self.assertTrue(indexer.graph.find("app.run"))

    def test_query_always_includes_start_and_defines_nested_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text(
                "class Service:\n    def very_long_method_name(self):\n        return 1\n",
                encoding="utf-8",
            )
            indexer = IncrementalIndexer(root)
            indexer.index()
            method = indexer.graph.find("app.Service.very_long_method_name")[0]
            file_node = next(
                node
                for node in indexer.graph.nodes.values()
                if node.file_path == "app.py" and node.node_type == "file"
            )

            context = indexer.graph.query(method.node_id, depth=1, max_nodes=1, token_budget=1)
            self.assertEqual([node.node_id for node in context.nodes], [method.node_id])
            self.assertTrue(context.truncated)
            self.assertTrue(
                any(
                    edge.source == file_node.node_id
                    and edge.target == method.node_id
                    and edge.kind == EdgeKind.DEFINES
                    for edge in indexer.graph.edges
                )
            )


if __name__ == "__main__":
    unittest.main()
