from typing import Any

from router.tools import ActionHandlers
from tst.memory.context import ContextRanker
from tst.memory.planner import MemoryPlanner, MemoryRecord
from tst.memory.retrieval import LexicalMemoryRetriever
from tst.protocol.models import KernelResponse
from tst.protocol.operations import ProtocolOperation
from tst.routing.decision import RouteDecision
from tst.routing.deterministic import deterministic_route


def _record(
    key: str,
    value: str,
    source_text: str,
    tags: list[str],
    *,
    updated_at: int = 1_000_000,
    deleted: bool = False,
) -> MemoryRecord:
    return MemoryRecord(
        key=key,
        value=value,
        memory_type="preference",
        source_text=source_text,
        created_at=1,
        updated_at=updated_at,
        confidence=1.0,
        tags=tags,
        source="user",
        layer="ltm",
        deleted=deleted,
    )


def test_planner_generates_canonical_preference_payload():
    decision = deterministic_route("Remember that I prefer TypeScript over JavaScript.")
    plan = MemoryPlanner().plan(decision, "Remember that I prefer TypeScript")
    assert plan.protocol_operation.value == "memory.store"
    assert plan.canonical_key == "user:default:preference:programming_language"
    assert plan.params["payload"]["data"]["value"] == "TypeScript"


def test_retrieval_question_is_get_not_store():
    decision = deterministic_route("What did I say about my editor?")
    plan = MemoryPlanner().plan(decision, "What did I say about my editor?")
    assert plan.protocol_operation.value == "memory.get"
    assert plan.params["key"] == "user:default:preference:editor"


def test_paraphrased_language_query_beats_distractors():
    records = [
        _record(
            "user:default:preference:programming_language",
            "TypeScript",
            "I prefer TypeScript over JavaScript",
            ["typescript", "programming-language"],
        ),
        _record(
            "user:default:preference:editor",
            "Vim",
            "I use Vim",
            ["vim", "editor"],
        ),
        _record(
            "user:default:preference:theme",
            "Dark mode",
            "I prefer dark mode",
            ["theme", "dark-mode"],
        ),
    ]
    ranked = LexicalMemoryRetriever().search(
        "Which language do I usually use?", records, top_k=3, now_ms=1_000_000
    )
    assert ranked[0].record.value == "TypeScript"


def test_new_update_wins_and_deleted_memory_never_leaks():
    key = "user:default:preference:editor"
    records = [
        _record(key, "Vim", "I use Vim", ["editor", "vim"], updated_at=10),
        _record(key, "VS Code", "I use VS Code", ["editor", "vscode"], updated_at=20),
    ]
    ranked = LexicalMemoryRetriever().search("my editor", records, now_ms=20)
    assert ranked[0].record.value == "VS Code"

    records.append(_record(key, "", "deleted", ["editor"], updated_at=30, deleted=True))
    assert LexicalMemoryRetriever().search("my editor", records, now_ms=30) == []


def test_context_budget_selects_ranked_values():
    record = _record(
        "user:default:preference:programming_language",
        "TypeScript",
        "I prefer TypeScript",
        ["language"],
    )
    ranked = LexicalMemoryRetriever().search("language", [record], now_ms=1_000_000)
    context = ContextRanker().select(ranked, max_characters=80)
    assert context.text == "[preference] TypeScript"
    assert context.characters == len(context.text)


def _response(request_id: str, result: dict[str, Any]) -> KernelResponse:
    return KernelResponse(
        protocol_version=1,
        request_id=request_id,
        ok=True,
        result=result,
        metrics={"kernel_ms": 0.1},
    )


def _wire_record(record: MemoryRecord) -> dict[str, Any]:
    return {"header": {}, "data": {"MemoryRecord": record.model_dump()}}


class _RetrievalClient:
    def __init__(self, wanted: MemoryRecord, distractor: MemoryRecord) -> None:
        self.wanted = wanted
        self.distractor = distractor
        self.calls: list[tuple[ProtocolOperation | str, dict[str, Any]]] = []

    def request(self, operation: ProtocolOperation | str, params: dict[str, Any]) -> KernelResponse:
        self.calls.append((operation, params))
        if operation == ProtocolOperation.MEMORY_GET:
            return _response("get", {"found": False, "key": params["key"], "payload": None})
        matches = [
            {
                "key": self.distractor.key,
                "layer": "ltm",
                "payload": _wire_record(self.distractor),
            },
            {
                "key": self.wanted.key,
                "layer": "ltm",
                "payload": _wire_record(self.wanted),
            },
            # A cross-layer duplicate must not leak into persistent retrieval.
            {
                "key": self.wanted.key,
                "layer": "stm",
                "payload": _wire_record(self.wanted.model_copy(update={"layer": "stm"})),
            },
        ]
        return _response("search", {"matches": matches, "count": len(matches)})


def test_action_handler_uses_shared_read_only_retrieval_pipeline():
    wanted = _record(
        "user:default:preference:programming_language",
        "TypeScript",
        "I prefer TypeScript",
        ["programming-language", "typescript"],
    )
    distractor = _record(
        "user:default:preference:editor",
        "Vim",
        "I use Vim",
        ["editor", "vim"],
    )
    client = _RetrievalClient(wanted, distractor)
    decision = RouteDecision(
        operation="retrieve",
        layer="ltm",
        subject="programming_language",
        search_terms=["language"],
        confidence=1.0,
        source="deterministic",
    )

    result = ActionHandlers(client).dispatch(decision, "Which language do I usually use?")

    assert result["context"][0] == "TypeScript"
    assert result["matches"][0]["key"] == wanted.key
    assert len({match["key"] for match in result["matches"]}) == len(result["matches"])
    assert client.calls[0][0] == ProtocolOperation.MEMORY_GET
    assert all(
        operation in {ProtocolOperation.MEMORY_GET, ProtocolOperation.MEMORY_SEARCH}
        for operation, _ in client.calls
    )


def test_context_budget_is_enforced_for_exact_results():
    wanted = _record(
        "user:default:preference:programming_language",
        "TypeScript",
        "I prefer TypeScript",
        ["language"],
    )

    class ExactClient:
        def request(self, operation, params):
            assert operation == ProtocolOperation.MEMORY_GET
            return _response(
                "exact",
                {"found": True, "key": wanted.key, "payload": _wire_record(wanted)},
            )

    decision = RouteDecision(
        operation="retrieve",
        layer="ltm",
        subject="programming_language",
        search_terms=["language"],
        confidence=1.0,
        source="deterministic",
    )
    result = ActionHandlers(ExactClient(), context_max_characters=10).dispatch(
        decision, "What did I say about programming language?"
    )
    assert result["context"] == []
    assert result["context_text"] == ""


def test_unconfigured_external_dispatch_degrades_to_local_no_memory():
    decision = RouteDecision(
        operation="escalate_external",
        layer="none",
        confidence=0.8,
        source="qwen",
    )
    result = ActionHandlers(None).dispatch(decision, "hard question")
    assert result["operation"] == "answer_without_memory"
    assert result["escalate"] is False
