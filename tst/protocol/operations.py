"""Operation and layer names supported by protocol version 1."""

from __future__ import annotations

from enum import Enum


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class MemoryLayer(_StringEnum):
    STM = "stm"
    LTM = "ltm"
    TREE = "tree"
    NONE = "none"


class ProtocolOperation(_StringEnum):
    KERNEL_PING = "kernel.ping"
    KERNEL_STATUS = "kernel.status"
    KERNEL_SHUTDOWN = "kernel.shutdown"

    MEMORY_STORE = "memory.store"
    MEMORY_GET = "memory.get"
    MEMORY_UPDATE = "memory.update"
    MEMORY_DELETE = "memory.delete"
    MEMORY_SEARCH = "memory.search"

    TREE_CLEAR = "tree.clear"
    TREE_INSERT = "tree.insert"
    TREE_REMOVE = "tree.remove"
    TREE_LINK = "tree.link"
    TREE_UNLINK = "tree.unlink"
    TREE_QUERY = "tree.query"
    TREE_FIND = "tree.find"

    PERSISTENCE_SAVE = "persistence.save"
    PERSISTENCE_STATUS = "persistence.status"
