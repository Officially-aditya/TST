from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

PYDANTIC_V2 = False
if importlib.util.find_spec("pydantic") is not None:
    import pydantic

    PYDANTIC_V2 = int(pydantic.__version__.split(".", 1)[0]) >= 2


@unittest.skipUnless(PYDANTIC_V2, "Pydantic v2 is not installed")
class ReviewValidatorTests(unittest.TestCase):
    def issue(self, **updates):
        value = {
            "file_path": "app.py",
            "line_start": 1,
            "line_end": 1,
            "severity": "high",
            "category": "logic",
            "title": "Wrong result",
            "explanation": "The function returns the wrong constant.",
            "suggested_fix": "Return the expected value.",
            "confidence": 0.9,
            "related_symbols": [],
        }
        value.update(updates)
        return value

    def test_validates_ranges_deduplicates_and_flags_low_confidence(self) -> None:
        from tst.worker.validator import ReviewValidator

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("def run():\n    return 0\n", encoding="utf-8")
            validator = ReviewValidator(root, confidence_threshold=0.6)
            first = self.issue()
            duplicate = self.issue()
            outside_range = self.issue(title="Imaginary line", line_start=9, line_end=9)
            low = self.issue(title="Minor concern", confidence=0.2)
            result = validator.validate({"issues": [first, duplicate, outside_range, low]})

        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(len(result.flagged_low_confidence), 1)
        self.assertEqual(
            {item.reason for item in result.rejected},
            {"duplicate finding", "line range ends at 9, but the file has 2 lines"},
        )
        self.assertIsNone(result.parse_error)

    def test_rejects_unstructured_output_and_traversal(self) -> None:
        from tst.worker.validator import ReviewValidator

        with tempfile.TemporaryDirectory() as directory:
            validator = ReviewValidator(directory)
            prose = validator.validate("I found a bug on line one.")
            traversal = validator.validate({"issues": [self.issue(file_path="../app.py")]})
        self.assertIsNotNone(prose.parse_error)
        self.assertIsNotNone(traversal.parse_error)
        self.assertEqual(prose.accepted, [])

    def test_graph_can_verify_related_symbols(self) -> None:
        from tst.analysis.incremental import IncrementalIndexer
        from tst.worker.validator import ReviewValidator

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("def run():\n    return 0\n", encoding="utf-8")
            indexer = IncrementalIndexer(root)
            indexer.index()
            validator = ReviewValidator(root, graph=indexer.graph)
            result = validator.validate(
                {
                    "issues": [
                        self.issue(related_symbols=["run"]),
                        self.issue(title="Invented symbol", related_symbols=["missing"]),
                    ]
                }
            )
        self.assertEqual(len(result.accepted), 1)
        self.assertIn("missing", result.rejected[0].reason)


if __name__ == "__main__":
    unittest.main()
