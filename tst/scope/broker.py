"""Scope-to-kernel and canonical-key translation helpers."""

from __future__ import annotations

from tst.memory.keys import CanonicalKeyBuilder, InvalidCanonicalKey

from .models import Project, Scope


def scope_for_key(key: str) -> Scope:
    """Infer the application scope from a canonical key namespace."""

    try:
        prefix = key.split(":", 1)[0].lower()
    except AttributeError as exc:
        raise InvalidCanonicalKey("memory key must be a string") from exc
    if prefix == "user":
        return Scope.GLOBAL
    if prefix == "project":
        return Scope.PROJECT
    if prefix == "session":
        return Scope.SESSION
    raise InvalidCanonicalKey(f"memory key has no known scope: {key!r}")


def layer_for_scope(scope: Scope | str) -> str:
    return "stm" if Scope.coerce(scope) is Scope.SESSION else "ltm"


def scope_prefix(scope: Scope | str, project: Project, session_id: str) -> str:
    selected = Scope.coerce(scope)
    if selected is Scope.GLOBAL:
        return "user:default:"
    if selected is Scope.PROJECT:
        return f"project:{CanonicalKeyBuilder.escape(project.id)}:"
    return f"session:{CanonicalKeyBuilder.escape(session_id)}:"


def key_for_scope(
    scope: Scope | str,
    project: Project,
    session_id: str,
    subject: str,
    *,
    memory_type: str = "fact",
) -> str:
    selected = Scope.coerce(scope)
    category = "preference" if memory_type == "preference" else "context"
    if selected is Scope.GLOBAL:
        category = "preference" if memory_type == "preference" else "fact"
        return CanonicalKeyBuilder.build("user", "default", category, subject)
    if selected is Scope.PROJECT:
        return CanonicalKeyBuilder.build("project", project.id, category, subject)
    return CanonicalKeyBuilder.build("session", session_id, "context", subject)


def remap_key(
    key: str,
    target: Scope | str,
    project: Project,
    session_id: str,
) -> str:
    """Move a key's semantic suffix into a destination namespace."""

    CanonicalKeyBuilder.validate(key)
    parts = key.split(":")
    suffix = parts[2:] if len(parts) > 2 else ["context", "value"]
    selected = Scope.coerce(target)
    if selected is Scope.GLOBAL:
        prefix = ["user", "default"]
    elif selected is Scope.PROJECT:
        prefix = ["project", project.id]
    else:
        prefix = ["session", session_id]
    return CanonicalKeyBuilder.build(*prefix, *suffix)


class ScopeBroker:
    """Small stateless façade useful to callers that only need scope mapping."""

    @staticmethod
    def scope_for_key(key: str) -> Scope:
        return scope_for_key(key)

    @staticmethod
    def layer_for_scope(scope: Scope | str) -> str:
        return layer_for_scope(scope)
