"""Regression tests for early OCR-confidence guardrails."""

import logging

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.schemas import (
    ClarificationResponse,
    EntityExtractionResult,
    ExtractedEntities,
    OCRResult,
    SuccessResponse,
)
from app.services import pipeline


APPOINTMENT_TEXT = "Book dentist next Friday at 3pm"
REFERENCE_DATETIME = "2026-09-18T12:00:00+05:30"

client = TestClient(app)


def _unexpected_call(*args, **kwargs):
    """Fail if a blocked processing stage is invoked."""

    pytest.fail("An early guardrail failed to prevent processing")


@pytest.fixture(autouse=True)
def block_external_extraction(monkeypatch):
    """Prevent all real Gemini calls during these tests."""

    monkeypatch.setattr(
        pipeline,
        "extract_entities",
        _unexpected_call,
    )


def _stub_image_ocr(monkeypatch, confidence):
    """Provide a controlled OCR result without running Tesseract."""

    result = OCRResult(
        raw_text=APPOINTMENT_TEXT,
        confidence=confidence,
        engine="tesseract",
        preprocessing_variant="test",
    )

    monkeypatch.setattr(
        pipeline,
        "extract_text_from_image",
        lambda image_bytes, content_type: result,
    )


def _allow_stub_extraction(monkeypatch):
    """Provide validated entities and track extraction invocations."""

    calls = []

    def fake_extract_entities(text):
        calls.append(text)

        return EntityExtractionResult(
            entities=ExtractedEntities(
                department="Dentistry",
                date_phrase="next Friday",
                time_phrase="3pm",
                ambiguity_reason=None,
            ),
            confidence=1.0,
            model_used="test-model",
        )

    monkeypatch.setattr(
        pipeline,
        "extract_entities",
        fake_extract_entities,
    )

    return calls


@pytest.mark.parametrize("confidence", [0.0, 0.49, 0.59])
def test_low_confidence_skips_extraction(
    monkeypatch,
    confidence,
):
    monkeypatch.setattr(
        settings,
        "ocr_confidence_threshold",
        0.60,
    )
    _stub_image_ocr(monkeypatch, confidence)

    result = pipeline.process_image(
        image_bytes=b"test-image",
        content_type="image/png",
        reference_datetime=REFERENCE_DATETIME,
    )

    assert isinstance(result.response, ClarificationResponse)
    assert "clearer image" in result.response.message
    assert result.trace.ocr.confidence == confidence
    assert result.trace.extraction is None
    assert result.trace.models_attempted == []
    assert result.trace.normalized is None
    assert result.trace.normalization_confidence is None


@pytest.mark.parametrize("confidence", [0.60, 0.61])
def test_threshold_boundary_allows_extraction(
    monkeypatch,
    confidence,
):
    monkeypatch.setattr(
        settings,
        "ocr_confidence_threshold",
        0.60,
    )
    _stub_image_ocr(monkeypatch, confidence)
    calls = _allow_stub_extraction(monkeypatch)

    result = pipeline.process_image(
        image_bytes=b"test-image",
        content_type="image/png",
        reference_datetime=REFERENCE_DATETIME,
    )

    assert calls == [APPOINTMENT_TEXT]
    assert isinstance(result.response, SuccessResponse)
    assert result.response.appointment.date.isoformat() == "2026-09-25"
    assert result.response.appointment.time == "15:00"
    assert result.trace.extraction is not None


def test_higher_configured_threshold_is_enforced(monkeypatch):
    monkeypatch.setattr(
        settings,
        "ocr_confidence_threshold",
        0.90,
    )
    _stub_image_ocr(monkeypatch, 0.85)

    result = pipeline.process_image(
        image_bytes=b"test-image",
        content_type="image/png",
        reference_datetime=REFERENCE_DATETIME,
    )

    assert isinstance(result.response, ClarificationResponse)
    assert result.trace.extraction is None
    assert result.trace.models_attempted == []


def test_lower_configured_threshold_is_respected(monkeypatch):
    monkeypatch.setattr(
        settings,
        "ocr_confidence_threshold",
        0.55,
    )
    _stub_image_ocr(monkeypatch, 0.56)
    calls = _allow_stub_extraction(monkeypatch)

    result = pipeline.process_image(
        image_bytes=b"test-image",
        content_type="image/png",
        reference_datetime=REFERENCE_DATETIME,
    )

    assert calls == [APPOINTMENT_TEXT]
    assert isinstance(result.response, SuccessResponse)


def test_typed_text_does_not_require_image_ocr(monkeypatch):
    monkeypatch.setattr(
        settings,
        "ocr_confidence_threshold",
        1.0,
    )
    monkeypatch.setattr(
        pipeline,
        "extract_text_from_image",
        _unexpected_call,
    )
    calls = _allow_stub_extraction(monkeypatch)

    result = pipeline.process_text(
        text=APPOINTMENT_TEXT,
        reference_datetime=REFERENCE_DATETIME,
    )

    assert calls == [APPOINTMENT_TEXT]
    assert isinstance(result.response, SuccessResponse)
    assert result.trace.ocr.engine == "typed_text"


def test_api_handles_and_logs_skipped_extraction(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(
        settings,
        "ocr_confidence_threshold",
        0.60,
    )
    _stub_image_ocr(monkeypatch, 0.59)

    with caplog.at_level(logging.INFO, logger="app.main"):
        response = client.post(
            "/api/v1/appointments/image",
            files={
                "image": (
                    "appointment.png",
                    b"test-image",
                    "image/png",
                )
            },
            data={
                "reference_datetime": REFERENCE_DATETIME,
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "needs_clarification"
    assert "clearer image" in response.json()["message"]
    assert "model=not_called" in caplog.text