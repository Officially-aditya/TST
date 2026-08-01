"""Shared lifecycle and STDIO client for the Rust kernel."""

from .client import StdioKernelClient
from .process import KernelProcessConfig, default_kernel_binary

__all__ = ["KernelProcessConfig", "StdioKernelClient", "default_kernel_binary"]
