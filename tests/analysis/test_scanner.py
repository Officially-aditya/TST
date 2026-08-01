from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tst.analysis.scanner import (
    ProjectScanner,
    ScanConfig,
    UnsafePathError,
    resolve_project_path,
)


class ProjectScannerTests(unittest.TestCase):
    def test_excludes_dependencies_secrets_binaries_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
            (root / "node_modules").mkdir()
            (root / "node_modules" / "ignored.js").write_text(
                "function nope() {}", encoding="utf-8"
            )
            (root / ".env.py").write_text("PASSWORD = 'private'", encoding="utf-8")
            (root / "binary.py").write_bytes(b"ok\x00not-source")
            (root / "late-binary.py").write_bytes(b"#" * 9_000 + b"\x00")
            (root / "late-invalid.py").write_bytes(b"#" * 9_000 + b"\xff")
            outside_file = Path(outside) / "outside.py"
            outside_file.write_text("def secret(): pass", encoding="utf-8")
            try:
                (root / "linked.py").symlink_to(outside_file)
            except OSError:
                pass

            scanner = ProjectScanner(root)
            files = scanner.scan()

            self.assertEqual([item.relative_path for item in files], ["src/app.py"])
            skipped = dict(scanner.skipped)
            self.assertIn("node_modules", skipped)
            self.assertIn(".env.py", skipped)
            self.assertIn("binary.py", skipped)
            self.assertEqual(skipped["late-binary.py"], "binary content")
            self.assertIn("unreadable UTF-8 source", skipped["late-invalid.py"])
            if (root / "linked.py").exists():
                self.assertIn("linked.py", skipped)

    def test_path_resolution_blocks_traversal_and_outward_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            nested = root / "src" / "app.py"
            nested.parent.mkdir()
            nested.write_text("pass\n", encoding="utf-8")
            self.assertEqual(resolve_project_path(root, "src/app.py"), nested.resolve())
            with self.assertRaises(UnsafePathError):
                resolve_project_path(root, Path(outside) / "missing.py", must_exist=False)

            target = Path(outside) / "target.py"
            target.write_text("pass\n", encoding="utf-8")
            link = root / "out.py"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlinks are unavailable")
            with self.assertRaises(UnsafePathError):
                resolve_project_path(root, link)

    def test_limits_are_applied_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("a = 1\n", encoding="utf-8")
            (root / "b.py").write_text("b = 2\n", encoding="utf-8")
            scanner = ProjectScanner(root, ScanConfig(max_files=1))
            files = scanner.scan()
            self.assertEqual([item.relative_path for item in files], ["a.py"])
            self.assertEqual(dict(scanner.skipped)["b.py"], "repository file-count limit")


if __name__ == "__main__":
    unittest.main()
