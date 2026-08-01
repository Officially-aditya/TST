import json

import pytest

from tst.routing.parser import (
    RouteParseError,
    parse_functiongemma_output,
    parse_json_tool_output,
)
from tst.routing.router import ActionRouter


def test_valid_functiongemma_action_call():
    raw = (
        "<start_function_call>call:store_memory{"
        "layer:<escape>ltm<escape>,subject:<escape>editor<escape>,"
        "payload:<escape>I prefer Vim<escape>}<end_function_call>"
    )
    decision = parse_functiongemma_output(raw, query="remember my editor")
    assert decision.operation == "store"
    assert decision.layer == "ltm"
    assert decision.source == "functiongemma"


def test_valid_qwen_action_call():
    raw = json.dumps(
        {
            "tool_calls": [
                {
                    "name": "retrieve_memory",
                    "args": {"layer": "ltm", "subject": "editor"},
                }
            ]
        }
    )
    decision = parse_json_tool_output(raw, query="Which editor?")
    assert decision.operation == "retrieve"
    assert not decision.mutates_memory


@pytest.mark.parametrize(
    "raw",
    [
        "I think store_memory would be appropriate.",
        'Explanation first {"tool_calls":[{"name":"store_memory","args":{}}]}',
        '{"tool_calls":[],"note":"store_memory"}',
        '{"tool_calls":[{"name":"store_memory","args":{"layer":"ltm"}}]}',
    ],
)
def test_invalid_output_cannot_become_mutation(raw):
    with pytest.raises(RouteParseError):
        if raw.lstrip().startswith("{"):
            parse_json_tool_output(raw, query="hello")
        else:
            parse_functiongemma_output(raw, query="hello")


def test_parser_failure_falls_back_to_no_memory():
    router = ActionRouter(
        functiongemma=lambda *_: "store_memory appears in this explanation",
        qwen=lambda *_: '{"tool_calls": []}',
    )
    decision = router.route("Tell me something general")
    assert decision.operation == "answer_without_memory"
    assert decision.layer == "none"
    assert router.parse_failures == 2


def test_valid_model_mutation_requires_explicit_user_authorization():
    call = (
        "call:store_memory{layer:<escape>ltm<escape>,"
        "subject:<escape>editor<escape>,payload:<escape>Vim<escape>}"
    )
    router = ActionRouter(functiongemma=lambda *_: call)
    decision = router.route("Explain how an editor stores files")
    assert decision.operation == "answer_without_memory"
    assert router.parse_failures == 1

    authorized = router.route("Store this editor preference", "Vim")
    assert authorized.operation == "store"


def test_inference_failure_and_unconfigured_external_provider_are_safe():
    external = json.dumps({"tool_calls": [{"name": "escalate_external", "args": {}}]})
    router = ActionRouter(
        functiongemma=lambda *_: (_ for _ in ()).throw(RuntimeError("model failed")),
        qwen=lambda *_: external,
    )
    decision = router.route("Answer this difficult question")
    assert decision.operation == "answer_without_memory"
    assert router.inference_failures == 1
    assert router.parse_failures == 1

    configured = ActionRouter(
        qwen=lambda *_: external,
        external_provider_configured=True,
    )
    assert configured.route("Answer this difficult question").operation == "escalate_external"


def test_schema_required_arguments_match_parser_contract():
    with pytest.raises(RouteParseError, match="missing arguments"):
        parse_json_tool_output(
            json.dumps({"tool_calls": [{"name": "retrieve_memory", "args": {"layer": "ltm"}}]}),
            query="recall it",
        )
