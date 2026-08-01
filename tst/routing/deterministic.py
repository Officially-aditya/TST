"""Conservative deterministic rules for explicit user intent."""

from __future__ import annotations

import re
from pathlib import Path

from .decision import RouteDecision

_SPACE = re.compile(r"\s+")
_FILE = re.compile(
    r"(?P<path>(?:[\w.-]+/)*[\w.-]+\.(?:py|js|jsx|ts|tsx|rs))\b",
    re.IGNORECASE,
)
_MUTATION_PREFIXES = {
    "store": re.compile(
        r"^(?:(?:please|kindly)\s+|(?:can|could|would)\s+you\s+|"
        r"i\s+(?:want|need)\s+you\s+to\s+)?"
        r"(?:remember\b|store\b|save\b|record\b|keep\s+(?:this|that)\b|"
        r"note\s+(?:this|that)\b)|^for\s+(?:this|the)\s+(?:conversation|session)\b",
        re.IGNORECASE,
    ),
    "update": re.compile(
        r"^(?:(?:please|kindly)\s+|(?:can|could|would)\s+you\s+)?"
        r"(?:update\b|change\b|replace\b|correct\b)",
        re.IGNORECASE,
    ),
    "forget": re.compile(
        r"^(?:(?:please|kindly)\s+|(?:can|could|would)\s+you\s+)?"
        r"(?:forget\b|delete\b|remove\b|erase\b)",
        re.IGNORECASE,
    ),
}


def _clean(value: str) -> str:
    return _SPACE.sub(" ", value.strip().strip(".?!"))


def _terms(value: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_+#.-]+", value.lower())
    ignored = {
        "a",
        "an",
        "about",
        "did",
        "do",
        "for",
        "i",
        "is",
        "it",
        "me",
        "my",
        "say",
        "said",
        "that",
        "the",
        "this",
        "to",
        "what",
    }
    return list(dict.fromkeys(word for word in words if word not in ignored))


def deterministic_route(text: str, payload: str | None = None) -> RouteDecision | None:
    """Return a decision only when the input carries explicit, safe intent.

    Ambiguous phrasing deliberately returns ``None`` so a model tier can decide.
    Mutations require a direct imperative phrase; merely mentioning words such
    as "remember" or "forget" inside an explanation cannot trigger a write.
    """

    query = _clean(text)
    if not query:
        return None

    analyze = re.fullmatch(r"/analyze\s+(.+)", query, re.IGNORECASE)
    if analyze:
        requested = analyze.group(1).strip()
        return RouteDecision(
            operation="analyze_code",
            layer="tree",
            subject=Path(requested).name,
            file_path=requested,
            search_terms=_terms(requested),
            confidence=1.0,
            source="deterministic",
        )

    temporary = re.fullmatch(
        r"for (?:this|the) (?:conversation|session)\s*[:,]?\s*(.+)",
        query,
        re.IGNORECASE,
    )
    if temporary:
        value = _clean(payload or temporary.group(1))
        value = re.sub(r"^remember\s+that\s+", "", value, flags=re.IGNORECASE)
        return RouteDecision(
            operation="store",
            layer="stm",
            subject=_infer_subject(value),
            payload=value,
            search_terms=_terms(value),
            confidence=1.0,
            source="deterministic",
        )

    temporary_remember = re.fullmatch(
        r"until (?:this|the) session ends\s*[:,]?\s*remember\s+(.+)",
        query,
        re.IGNORECASE,
    )
    temporary_use = re.fullmatch(r"temporarily\s+(?:use|remember)\s+(.+)", query, re.IGNORECASE)
    temporary_match = temporary_remember or temporary_use
    if temporary_match:
        value = _clean(payload or temporary_match.group(1))
        return RouteDecision(
            operation="store",
            layer="stm",
            subject=_infer_subject(value),
            payload=value,
            search_terms=_terms(value),
            confidence=0.99,
            source="deterministic",
        )

    remember = re.fullmatch(r"(?:please\s+)?remember\s+that\s+(.+)", query, re.IGNORECASE)
    if remember:
        value = _clean(payload or remember.group(1))
        return RouteDecision(
            operation="store",
            layer="ltm",
            subject=_infer_subject(value),
            payload=value,
            search_terms=_terms(value),
            confidence=1.0,
            source="deterministic",
        )

    persistent_remember = re.fullmatch(
        r"please remember permanently\s*:\s*(.+)", query, re.IGNORECASE
    )
    save_preference = re.fullmatch(
        r"save this preference(?:\s*#\d+)?\s*:\s*(.+)", query, re.IGNORECASE
    )
    persistent_match = persistent_remember or save_preference
    if persistent_match:
        value = _clean(payload or persistent_match.group(1))
        return RouteDecision(
            operation="store",
            layer="ltm",
            subject=_infer_subject(value),
            payload=value,
            search_terms=_terms(value),
            confidence=0.99,
            source="deterministic",
        )

    just_said = re.fullmatch(
        r"what did i (?:just|recently) say|what (?:was|is) the (?:last|recent) thing i said",
        query,
        re.IGNORECASE,
    )
    if just_said:
        return RouteDecision(
            operation="retrieve",
            layer="stm",
            subject="recent_context",
            search_terms=["recent", "context"],
            confidence=1.0,
            source="deterministic",
        )

    recent_subject = re.fullmatch(
        r"what did i just say about\s+(.+)|"
        r"what is the temporary\s+(.+?)\s+for this chat(?:\s*\(\d+\))?|"
        r"recall our current-session\s+(.+?)(?:\s*,\s*reference\s+\d+)?",
        query,
        re.IGNORECASE,
    )
    if recent_subject:
        subject = _clean(next(group for group in recent_subject.groups() if group is not None))
        return RouteDecision(
            operation="retrieve",
            layer="stm",
            subject=_infer_subject(subject),
            search_terms=_terms(subject),
            confidence=0.99,
            source="deterministic",
        )

    if re.fullmatch(
        r"what is the service called(?: in (?:this|the) (?:session|conversation))?",
        query,
        re.IGNORECASE,
    ):
        return RouteDecision(
            operation="retrieve",
            layer="stm",
            subject="service_name",
            search_terms=["service", "name"],
            confidence=1.0,
            source="deterministic",
        )

    said_about = re.fullmatch(r"what did i (?:say|tell you) about\s+(.+)", query, re.IGNORECASE)
    if said_about:
        subject = _clean(said_about.group(1))
        return RouteDecision(
            operation="retrieve",
            layer="ltm",
            subject=_infer_subject(subject),
            search_terms=_terms(subject),
            confidence=1.0,
            source="deterministic",
        )

    saved_preference = re.fullmatch(
        r"which\s+(.+?)\s+do i usually use(?:\s*\(\d+\))?|"
        r"recall my saved\s+(.+?)\s+preference(?:\s*,\s*item\s+\d+)?",
        query,
        re.IGNORECASE,
    )
    if saved_preference:
        subject = _clean(next(group for group in saved_preference.groups() if group is not None))
        return RouteDecision(
            operation="retrieve",
            layer="ltm",
            subject=_infer_subject(subject),
            search_terms=_terms(subject),
            confidence=0.99,
            source="deterministic",
        )

    direct_preference = re.fullmatch(
        r"which\s+(.+?)\s+do i use|"
        r"which (?:programming )?language should (?:we|i) use(?:\s+for\s+.+)?",
        query,
        re.IGNORECASE,
    )
    if direct_preference:
        subject = _clean(direct_preference.group(1) or "programming language")
        return RouteDecision(
            operation="retrieve",
            layer="ltm",
            subject=_infer_subject(subject),
            search_terms=_terms(subject),
            confidence=0.99,
            source="deterministic",
        )

    forget = re.fullmatch(
        r"(?:please\s+)?forget\s+(?:that\s+|what i (?:said|told you) about\s+)?(.+)",
        query,
        re.IGNORECASE,
    )
    if forget:
        subject = _clean(forget.group(1))
        return RouteDecision(
            operation="forget",
            layer="ltm",
            subject=_infer_subject(subject),
            search_terms=_terms(subject),
            confidence=1.0,
            source="deterministic",
        )

    delete_saved = re.fullmatch(
        r"delete what you remember about\s+(.+?)(?:\s*,\s*request\s+\d+)?",
        query,
        re.IGNORECASE,
    )
    if delete_saved:
        subject = _clean(delete_saved.group(1))
        return RouteDecision(
            operation="forget",
            layer="ltm",
            subject=_infer_subject(subject),
            search_terms=_terms(subject),
            confidence=0.99,
            source="deterministic",
        )

    update = re.fullmatch(r"(?:update|change)\s+(?:my\s+)?(.+?)\s+to\s+(.+)", query, re.IGNORECASE)
    if update:
        subject = _clean(update.group(1))
        value = _clean(payload or update.group(2))
        return RouteDecision(
            operation="update",
            layer="ltm",
            subject=_infer_subject(subject),
            payload=value,
            search_terms=_terms(subject),
            confidence=0.98,
            source="deterministic",
        )

    revised = re.fullmatch(
        r"change the saved\s+(.+?)\s*;\s*it is now\s+(.+?)(?:\s*,\s*revision\s+\d+)?",
        query,
        re.IGNORECASE,
    )
    if revised:
        subject = _clean(revised.group(1))
        value = _clean(payload or revised.group(2))
        return RouteDecision(
            operation="update",
            layer="ltm",
            subject=_infer_subject(subject),
            payload=value,
            search_terms=_terms(subject),
            confidence=0.99,
            source="deterministic",
        )

    # A concrete code path plus an analysis verb is safe to route without a model.
    file_match = _FILE.search(query)
    if file_match and re.search(
        r"\b(?:analy[sz]e|debug|inspect|review|trace|index|cover|which tests|"
        r"what calls|find (?:callers?|references))\b",
        query,
        re.IGNORECASE,
    ):
        file_path = file_match.group("path")
        return RouteDecision(
            operation="analyze_code",
            layer="tree",
            subject=Path(file_path).name,
            file_path=file_path,
            search_terms=_terms(query),
            confidence=0.99,
            source="deterministic",
        )

    return None


def mutation_is_authorized(text: str, operation: str) -> bool:
    """Require an explicit user-directed verb before accepting a model mutation."""

    pattern = _MUTATION_PREFIXES.get(operation)
    return pattern is not None and pattern.search(_clean(text)) is not None


def _infer_subject(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\b(?:typescript|javascript|python|rust|java|language)\b", lowered):
        return "programming_language"
    if re.search(r"\b(?:vim|neovim|emacs|vscode|editor|ide)\b", lowered):
        return "editor"
    if re.search(r"\b(?:theme|dark mode|light mode)\b", lowered):
        return "theme"
    if re.search(
        r"\b(?:call|called|name)\b.*\bservice\b|\bservice\b.*\b(?:call|called|name)\b", lowered
    ):
        return "service_name"
    if re.search(r"\b(?:framework|react|vue|svelte|django|flask)\b", lowered):
        return "framework"
    # Prefer a stable, short noun phrase over the entire utterance.
    cleaned = re.sub(
        r"^(?:i\s+)?(?:always\s+)?(?:prefer|use|want|like)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return "_".join(_terms(cleaned)[:6]) or "general"
