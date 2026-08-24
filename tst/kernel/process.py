"""Kernel binary discovery and opt-in developer builds."""

from __future__ import annotations

import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tst.protocol.errors import KernelStartError


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_kernel_binary(crate_dir: Path | None = None) -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    system = platform.system().lower()
    machine = (
        platform.machine()
        .lower()
        .replace("amd64", "x86_64")
        .replace("x64", "x86_64")
        .replace("aarch64", "arm64")
    )
    bundled = Path(__file__).resolve().parents[1] / "bin" / f"{system}-{machine}" / f"server{suffix}"
    if (crate_dir is None or not (crate_dir / "Cargo.toml").is_file()) and bundled.is_file():
        return bundled
    crate = crate_dir or project_root() / "tst_memory"
    return crate / "target" / "release" / f"server{suffix}"


@dataclass(frozen=True, slots=True)
class KernelProcessConfig:
    """Configuration for one local kernel process.

    ``build_kernel`` is deliberately false by default. Normal application
    startup must use an existing binary; developer builds are explicit.
    ``command`` exists primarily for contract tests and prebuilt wrappers.
    """

    crate_dir: Path = field(default_factory=lambda: project_root() / "tst_memory")
    binary_path: Path | None = None
    command: Sequence[str] | None = None
    build_kernel: bool = False
    startup_timeout: float = 10.0
    request_timeout: float = 5.0
    shutdown_timeout: float = 3.0
    stderr_history: int = 200
    max_response_bytes: int = 8 * 1024 * 1024
    env: Mapping[str, str] | None = None
    working_directory: Path | None = None

    def __post_init__(self) -> None:
        if min(self.startup_timeout, self.request_timeout, self.shutdown_timeout) <= 0:
            raise ValueError("kernel process timeouts must be positive")
        if self.stderr_history <= 0 or self.max_response_bytes <= 0:
            raise ValueError("kernel process buffer limits must be positive")
        if self.command is not None and not self.command:
            raise ValueError("kernel command cannot be empty")

    def resolved_binary(self) -> Path:
        configured = self.binary_path
        if configured is None:
            from_env = os.environ.get("TST_KERNEL_BIN")
            configured = Path(from_env) if from_env else default_kernel_binary(self.crate_dir)
        return configured.expanduser().resolve()

    def resolved_command(self) -> list[str]:
        if self.command:
            return [str(part) for part in self.command]
        return [str(self.resolved_binary())]

    def resolved_working_directory(self) -> Path:
        """Return a valid child-process directory for source and wheel installs."""

        if self.working_directory is not None:
            return self.working_directory.expanduser().resolve()
        if self.crate_dir.is_dir():
            return self.crate_dir.resolve()
        return Path.cwd().resolve()


def build_kernel(config: KernelProcessConfig) -> Path:
    """Build the release server after an explicit developer opt-in."""

    if not (config.crate_dir / "Cargo.toml").is_file():
        raise KernelStartError(
            "Rust kernel source is not present in this installation; use a source checkout "
            "or configure a prebuilt binary with TST_KERNEL_BIN"
        )
    try:
        completed = subprocess.run(
            ["cargo", "build", "--locked", "--release", "--bin", "server"],
            cwd=config.crate_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise KernelStartError(f"could not execute Cargo: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-2000:]
        raise KernelStartError(f"kernel build failed: {detail}")
    binary = config.resolved_binary()
    if not binary.is_file():
        raise KernelStartError(f"cargo succeeded but kernel binary is missing: {binary}")
    return binary
