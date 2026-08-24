"""Stage a compiled Rust kernel in the Python package before wheel creation."""

from __future__ import annotations

import argparse
import platform
import shutil
from pathlib import Path


def platform_key(system: str | None = None, machine: str | None = None) -> str:
    selected_system = (system or platform.system()).lower()
    selected_machine = (machine or platform.machine()).lower()
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    return f"{selected_system}-{aliases.get(selected_machine, selected_machine)}"


def stage_kernel(binary: Path, package_root: Path, *, system: str | None = None, machine: str | None = None) -> Path:
    source = binary.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"kernel binary does not exist: {source}")
    selected_system = (system or platform.system()).lower()
    target_name = "server.exe" if selected_system == "windows" else "server"
    target = package_root / "bin" / platform_key(system, machine) / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if target_name != "server.exe":
        target.chmod(target.stat().st_mode | 0o111)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tst",
    )
    parser.add_argument("--system")
    parser.add_argument("--machine")
    args = parser.parse_args()
    target = stage_kernel(
        args.binary,
        args.package_root.expanduser().resolve(),
        system=args.system,
        machine=args.machine,
    )
    print(f"Staged kernel: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
