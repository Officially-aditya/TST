from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tst.protocol.models import KernelRequest, KernelResponse
from tst.protocol.operations import ProtocolOperation

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "protocol"
pytestmark = pytest.mark.protocol_contract


def load(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.protocol_contract
@pytest.mark.parametrize("stem", ["ping", "store-ltm", "invalid-layer", "unknown-operation"])
def test_fixture_responses_echo_request_identity(stem: str) -> None:
    request = load(f"{stem}.request.json")
    response = KernelResponse.model_validate(load(f"{stem}.response.json"))
    assert response.protocol_version == request["protocol_version"]
    assert response.request_id == request["request_id"]


@pytest.mark.protocol_contract
@pytest.mark.parametrize("stem", ["ping", "store-ltm", "invalid-layer"])
def test_known_request_envelopes_are_valid(stem: str) -> None:
    request = KernelRequest.model_validate(load(f"{stem}.request.json"))
    assert request.protocol_version == 1


@pytest.mark.protocol_contract
def test_unknown_operation_fixture_is_rejected_by_python_contract() -> None:
    with pytest.raises(ValidationError):
        KernelRequest.model_validate(load("unknown-operation.request.json"))


@pytest.mark.protocol_contract
def test_envelopes_reject_unknown_top_level_fields() -> None:
    request = load("ping.request.json")
    request["silent_default"] = True
    with pytest.raises(ValidationError):
        KernelRequest.model_validate(request)


@pytest.mark.protocol_contract
def test_operation_catalog_covers_every_v1_operation_once() -> None:
    catalog = load("operation-catalog.json")
    assert isinstance(catalog, list)
    requests = [KernelRequest.model_validate(item) for item in catalog]
    assert len({request.request_id for request in requests}) == len(requests)
    assert {request.operation for request in requests} == set(ProtocolOperation)
