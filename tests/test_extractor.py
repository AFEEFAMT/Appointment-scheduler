"""Tests for extraction parsing, retries, fallbacks and caching."""

import json
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors
from pydantic import SecretStr

from app.config import settings
from app.schemas import ExtractedEntities
from app.services import extractor


APPOINTMENT_TEXT = "Book dentist next Friday at 3pm"

ENTITY_DATA = {
    "date_phrase": "next Friday",
    "time_phrase": "3pm",
    "department": "Dentistry",
    "ambiguity_reason": None,
}


def _valid_response():
    """Create a structured response without using the Gemini API."""

    return SimpleNamespace(
        parsed=dict(ENTITY_DATA),
        text=None,
    )


def _api_error(code):
    """Construct an SDK error for exercising status-code handling."""

    return errors.APIError(
        code=code,
        response_json={
            "error": {
                "code": code,
                "message": "Simulated API failure",
            }
        },
    )


class StubClient:
    """Local Gemini client substitute with controlled responses."""

    def __init__(self, state):
        self.state = state
        self.models = self
        self.state.created += 1

    def generate_content(self, **kwargs):
        self.state.requests.append(kwargs)

        if not self.state.outcomes:
            pytest.fail("Unexpected additional Gemini request")

        outcome = self.state.outcomes.pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        return outcome

    def close(self):
        self.state.closed += 1


@pytest.fixture(autouse=True)
def gemini_stub(monkeypatch):
    """Replace client creation and waiting with local test substitutes."""

    state = SimpleNamespace(
        outcomes=[],
        requests=[],
        sleeps=[],
        created=0,
        closed=0,
    )

    monkeypatch.setattr(
        settings,
        "gemini_api_key",
        SecretStr("test-only-not-a-real-key"),
    )
    monkeypatch.setattr(
        settings,
        "gemini_primary_model",
        "test-primary",
    )
    monkeypatch.setattr(
        settings,
        "gemini_fallback_models",
        "test-fallback",
    )
    monkeypatch.setattr(
        settings,
        "llm_max_retries",
        1,
    )
    monkeypatch.setattr(
        settings,
        "llm_retry_base_seconds",
        1.0,
    )
    monkeypatch.setattr(
        settings,
        "cache_ttl_seconds",
        600,
    )

    monkeypatch.setattr(
        extractor.genai,
        "Client",
        lambda **kwargs: StubClient(state),
    )
    monkeypatch.setattr(
        extractor.time,
        "sleep",
        state.sleeps.append,
    )
    monkeypatch.setattr(
        extractor.random,
        "uniform",
        lambda minimum, maximum: 0.0,
    )

    extractor.clear_extraction_cache()

    try:
        yield state
    finally:
        extractor.clear_extraction_cache()


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(
            parsed=dict(ENTITY_DATA),
            text=None,
        ),
        SimpleNamespace(
            parsed=ExtractedEntities(**ENTITY_DATA),
            text=None,
        ),
        SimpleNamespace(
            parsed=None,
            text=json.dumps(ENTITY_DATA),
        ),
    ],
)
def test_accept_supported_response_formats(gemini_stub, response):
    gemini_stub.outcomes.append(response)

    result = extractor.extract_entities(APPOINTMENT_TEXT)

    assert result.entities.department == "Dentistry"
    assert result.entities.date_phrase == "next Friday"
    assert result.entities.time_phrase == "3pm"
    assert result.model_used == "test-primary"
    assert gemini_stub.created == 1
    assert gemini_stub.closed == 1


@pytest.mark.parametrize("text", ["", " \n\t ", None, 123])
def test_invalid_input_rejected_before_client_creation(
    gemini_stub,
    text,
):
    with pytest.raises(extractor.EntityExtractionError):
        extractor.extract_entities(text)

    assert gemini_stub.created == 0
    assert gemini_stub.requests == []


def test_overlong_input_rejected_before_client_creation(gemini_stub):
    with pytest.raises(
        extractor.EntityExtractionError,
        match="too long",
    ):
        extractor.extract_entities("x" * 10_001)

    assert gemini_stub.created == 0


@pytest.mark.parametrize("code", [400, 404])
def test_unsupported_model_moves_to_fallback(gemini_stub, code):
    gemini_stub.outcomes.extend(
        [_api_error(code), _valid_response()]
    )

    result = extractor.extract_entities(APPOINTMENT_TEXT)

    assert result.model_used == "test-fallback"
    assert [
        request["model"]
        for request in gemini_stub.requests
    ] == ["test-primary", "test-fallback"]
    assert gemini_stub.sleeps == []
    assert gemini_stub.closed == 1


def test_temporary_server_failure_retries_primary(gemini_stub):
    gemini_stub.outcomes.extend(
        [_api_error(503), _valid_response()]
    )

    result = extractor.extract_entities(APPOINTMENT_TEXT)

    assert result.model_used == "test-primary"
    assert [
        request["model"]
        for request in gemini_stub.requests
    ] == ["test-primary", "test-primary"]
    assert gemini_stub.sleeps == [1.0]
    assert gemini_stub.closed == 1


def test_network_failure_retries_primary(gemini_stub):
    gemini_stub.outcomes.extend(
        [
            httpx.ConnectError("Simulated connection failure"),
            _valid_response(),
        ]
    )

    result = extractor.extract_entities(APPOINTMENT_TEXT)

    assert result.model_used == "test-primary"
    assert len(gemini_stub.requests) == 2
    assert gemini_stub.sleeps == [1.0]
    assert gemini_stub.closed == 1


def test_timeout_moves_directly_to_fallback(gemini_stub):
    gemini_stub.outcomes.extend(
        [
            httpx.ReadTimeout("Simulated read timeout"),
            _valid_response(),
        ]
    )

    result = extractor.extract_entities(APPOINTMENT_TEXT)

    assert result.model_used == "test-fallback"
    assert [
        request["model"]
        for request in gemini_stub.requests
    ] == ["test-primary", "test-fallback"]
    assert gemini_stub.sleeps == []
    assert gemini_stub.closed == 1


@pytest.mark.parametrize("code", [401, 403])
def test_authentication_failure_stops_attempts(gemini_stub, code):
    gemini_stub.outcomes.append(_api_error(code))

    with pytest.raises(extractor.GeminiConfigurationError):
        extractor.extract_entities(APPOINTMENT_TEXT)

    assert len(gemini_stub.requests) == 1
    assert gemini_stub.sleeps == []
    assert gemini_stub.closed == 1


def test_quota_failure_stops_attempts(gemini_stub):
    gemini_stub.outcomes.append(_api_error(429))

    with pytest.raises(extractor.GeminiQuotaError):
        extractor.extract_entities(APPOINTMENT_TEXT)

    assert len(gemini_stub.requests) == 1
    assert gemini_stub.sleeps == []
    assert gemini_stub.closed == 1


def test_invalid_structured_response_uses_fallback(
    monkeypatch,
    gemini_stub,
):
    monkeypatch.setattr(settings, "llm_max_retries", 0)

    gemini_stub.outcomes.extend(
        [
            SimpleNamespace(
                parsed={"unexpected_field": True},
                text=None,
            ),
            _valid_response(),
        ]
    )

    result = extractor.extract_entities(APPOINTMENT_TEXT)

    assert result.model_used == "test-fallback"
    assert len(gemini_stub.requests) == 2
    assert gemini_stub.closed == 1


def test_empty_responses_retry_then_use_fallback(gemini_stub):
    gemini_stub.outcomes.extend(
        [
            SimpleNamespace(parsed=None, text=""),
            SimpleNamespace(parsed=None, text=None),
            _valid_response(),
        ]
    )

    result = extractor.extract_entities(APPOINTMENT_TEXT)

    assert result.model_used == "test-fallback"
    assert [
        request["model"]
        for request in gemini_stub.requests
    ] == [
        "test-primary",
        "test-primary",
        "test-fallback",
    ]
    assert gemini_stub.sleeps == [1.0]
    assert gemini_stub.closed == 1


def test_all_model_failures_raise_service_error(gemini_stub):
    gemini_stub.outcomes.extend(
        [
            TimeoutError("Primary timed out"),
            TimeoutError("Fallback timed out"),
        ]
    )

    with pytest.raises(
        extractor.GeminiServiceError,
        match="All configured Gemini models failed",
    ):
        extractor.extract_entities(APPOINTMENT_TEXT)

    assert len(gemini_stub.requests) == 2
    assert gemini_stub.closed == 1


def test_repeated_input_uses_cache(gemini_stub):
    gemini_stub.outcomes.append(_valid_response())

    first = extractor.extract_entities(APPOINTMENT_TEXT)
    second = extractor.extract_entities(
        "  Book dentist   next Friday at 3pm  "
    )

    assert first == second
    assert len(gemini_stub.requests) == 1
    assert gemini_stub.created == 1
    assert gemini_stub.closed == 1


def test_expired_cache_triggers_new_request(
    monkeypatch,
    gemini_stub,
):
    clock = [100.0]

    monkeypatch.setattr(
        extractor.time,
        "monotonic",
        lambda: clock[0],
    )
    monkeypatch.setattr(
        settings,
        "cache_ttl_seconds",
        10,
    )

    gemini_stub.outcomes.extend(
        [_valid_response(), _valid_response()]
    )

    extractor.extract_entities(APPOINTMENT_TEXT)
    clock[0] = 111.0
    extractor.extract_entities(APPOINTMENT_TEXT)

    assert len(gemini_stub.requests) == 2
    assert gemini_stub.created == 2
    assert gemini_stub.closed == 2


def test_mutating_result_does_not_change_cache(gemini_stub):
    gemini_stub.outcomes.append(_valid_response())

    first = extractor.extract_entities(APPOINTMENT_TEXT)
    first.entities.department = "Cardiology"

    second = extractor.extract_entities(APPOINTMENT_TEXT)

    assert second.entities.department == "Dentistry"
    assert len(gemini_stub.requests) == 1


def test_model_configuration_change_invalidates_cache_key(
    monkeypatch,
    gemini_stub,
):
    gemini_stub.outcomes.extend(
        [_valid_response(), _valid_response()]
    )

    extractor.extract_entities(APPOINTMENT_TEXT)

    monkeypatch.setattr(
        settings,
        "gemini_primary_model",
        "test-new-primary",
    )

    result = extractor.extract_entities(APPOINTMENT_TEXT)

    assert result.model_used == "test-new-primary"
    assert len(gemini_stub.requests) == 2


def test_zero_ttl_disables_cache_reuse(monkeypatch, gemini_stub):
    monkeypatch.setattr(
        settings,
        "cache_ttl_seconds",
        0,
    )

    gemini_stub.outcomes.extend(
        [_valid_response(), _valid_response()]
    )

    extractor.extract_entities(APPOINTMENT_TEXT)
    extractor.extract_entities(APPOINTMENT_TEXT)

    assert len(gemini_stub.requests) == 2


def test_missing_key_rejected_before_client_creation(
    monkeypatch,
    gemini_stub,
):
    monkeypatch.setattr(
        settings,
        "gemini_api_key",
        SecretStr(""),
    )

    with pytest.raises(extractor.GeminiConfigurationError):
        extractor.extract_entities(APPOINTMENT_TEXT)

    assert gemini_stub.created == 0
    assert gemini_stub.requests == []


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({}, 0.0),
        ({"department": "Dentistry"}, 0.333),
        (
            {
                "department": "Dentistry",
                "date_phrase": "tomorrow",
            },
            0.667,
        ),
        (ENTITY_DATA, 1.0),
    ],
)
def test_completeness_score_counts_present_fields(data, expected):
    entities = ExtractedEntities(**data)

    assert extractor._calculate_confidence(entities) == expected


def test_ambiguity_caps_completeness_score():
    entities = ExtractedEntities(
        **{
            **ENTITY_DATA,
            "ambiguity_reason": "Multiple appointment options",
        }
    )

    assert extractor._calculate_confidence(entities) == 0.65