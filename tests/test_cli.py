from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tst import cli
from tst.kernel.process import KernelProcessConfig, build_kernel


def test_analyze_single_file_does_not_index_siblings(tmp_path: Path, capsys) -> None:
    target = tmp_path / "target.py"
    target.write_text("def target():\n    return 1\n", encoding="utf-8")
    (tmp_path / "sibling.py").write_text("def sibling():\n    return 2\n", encoding="utf-8")

    assert cli._analyze(str(target), as_json=True, symbol=None) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["report"]["parsed_files"] == 1
    paths = {node["file_path"] for node in document["graph"]["nodes"] if node["file_path"]}
    assert paths == {"target.py"}


def test_doctor_honors_configured_prebuilt_kernel(tmp_path: Path, monkeypatch, capsys) -> None:
    binary = tmp_path / "server"
    binary.write_text("binary", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("TST_KERNEL_BIN", str(binary))
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)

    assert cli._doctor(as_json=True) == 0
    document = json.loads(capsys.readouterr().out)
    checks = {item["name"]: item for item in document["checks"]}
    assert checks["kernel"]["ok"] is True
    assert checks["kernel"]["detail"] == str(binary.resolve())
    assert checks["cargo"]["ok"] is True


def test_chat_forwards_all_runtime_flags(monkeypatch) -> None:
    observed: list[str] = []

    def legacy_main() -> int:
        observed.extend(sys.argv)
        return 7

    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda _: SimpleNamespace(main=legacy_main),
    )
    args = argparse.Namespace(no_kernel=True, no_worker=True, build_kernel=True)
    previous = list(sys.argv)
    assert cli._chat(args) == 7
    assert observed == ["tst chat", "--no-kernel", "--no-worker", "--build-kernel"]
    assert sys.argv == previous


def test_kernel_build_is_locked_and_env_binary_has_precedence(tmp_path: Path, monkeypatch) -> None:
    crate = tmp_path / "crate"
    (crate / "target" / "release").mkdir(parents=True)
    (crate / "Cargo.toml").write_text(
        "[package]\nname='fixture'\nversion='0.1.0'\n", encoding="utf-8"
    )
    binary = crate / "target" / "release" / ("server.exe" if os.name == "nt" else "server")
    binary.write_text("binary", encoding="utf-8")
    observed: list[str] = []

    def run(command, **kwargs):
        observed.extend(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("tst.kernel.process.subprocess.run", run)
    assert build_kernel(KernelProcessConfig(crate_dir=crate)) == binary.resolve()
    assert observed == ["cargo", "build", "--locked", "--release", "--bin", "server"]

    configured = tmp_path / "configured-server"
    monkeypatch.setenv("TST_KERNEL_BIN", str(configured))
    assert KernelProcessConfig(crate_dir=crate).resolved_binary() == configured.resolve()
