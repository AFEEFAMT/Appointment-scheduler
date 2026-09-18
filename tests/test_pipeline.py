"""Tests for the complete appointment-processing pipeline."""

from datetime import date

from app.schemas import (
    ClarificationResponse,
    EntityExtractionResult,
    ExtractedEntities,
    OCRResult,
    SuccessResponse,
)
from app.services import pipeline


REFERENCE_DATETIME = "2025-09-19T00:00:00+05:30"


def _complete_extraction() -> EntityExtractionResult:
    return EntityExtractionResult(
        entities=ExtractedEntities(
            date_phrase="next Friday",
            time_phrase="3pm",
            department="dentist",
            ambiguity_reason=None,
        ),
        confidence=1.0,
        model_used="test-model",
    )


def test_process_text_success(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "extract_entities",
        lambda _text: _complete_extraction(),
    )

    result = pipeline.process_text(
        text="Book dentist next Friday at 3pm",
        reference_datetime=REFERENCE_DATETIME,
    )

    assert isinstance(result.response, SuccessResponse)
    assert result.response.appointment.department == "Dentistry"
    assert result.response.appointment.date == date(2025, 9, 26)
    assert result.response.appointment.time == "15:00"
    assert result.response.appointment.tz == "Asia/Kolkata"
    assert result.trace.models_attempted == ["test-model"]
    assert result.trace.normalization_confidence == 1.0


def test_missing_time_returns_clarification(monkeypatch):
    extraction = EntityExtractionResult(
        entities=ExtractedEntities(
            date_phrase="tomorrow",
            time_phrase=None,
            department="Dentistry",
            ambiguity_reason="The appointment time is missing",
        ),
        confidence=0.667,
        model_used="test-model",
    )

    monkeypatch.setattr(
        pipeline,
        "extract_entities",
        lambda _text: extraction,
    )

    result = pipeline.process_text(
        text="Book a dentist tomorrow",
        reference_datetime=REFERENCE_DATETIME,
    )

    assert isinstance(result.response, ClarificationResponse)
    assert "time" in result.response.message.casefold()
    assert result.trace.normalized is None


def test_ambiguous_date_returns_clarification(monkeypatch):
    extraction = EntityExtractionResult(
        entities=ExtractedEntities(
            date_phrase="next week",
            time_phrase="3pm",
            department="Dentistry",
            ambiguity_reason=None,
        ),
        confidence=1.0,
        model_used="test-model",
    )

    monkeypatch.setattr(
        pipeline,
        "extract_entities",
        lambda _text: extraction,
    )

    result = pipeline.process_text(
        text="Book a dentist next week at 3pm",
        reference_datetime=REFERENCE_DATETIME,
    )

    assert isinstance(result.response, ClarificationResponse)
    assert "date" in result.response.message.casefold()
    assert result.trace.normalized is None


def test_past_appointment_returns_clarification(monkeypatch):
    extraction = EntityExtractionResult(
        entities=ExtractedEntities(
            date_phrase="today",
            time_phrase="9am",
            department="Dentistry",
            ambiguity_reason=None,
        ),
        confidence=1.0,
        model_used="test-model",
    )

    monkeypatch.setattr(
        pipeline,
        "extract_entities",
        lambda _text: extraction,
    )

    result = pipeline.process_text(
        text="Book a dentist today at 9am",
        reference_datetime="2025-09-19T10:00:00+05:30",
    )

    assert isinstance(result.response, ClarificationResponse)
    assert "time" in result.response.message.casefold()
    assert result.trace.normalized is None


def test_process_image_success(monkeypatch):
    ocr_result = OCRResult(
        raw_text="Book dentist next Friday at 3pm",
        confidence=0.95,
        engine="tesseract",
        preprocessing_variant="enhanced-psm6",
    )

    monkeypatch.setattr(
        pipeline,
        "extract_text_from_image",
        lambda image_bytes, content_type: ocr_result,
    )

    monkeypatch.setattr(
        pipeline,
        "extract_entities",
        lambda _text: _complete_extraction(),
    )

    result = pipeline.process_image(
        image_bytes=b"test-image",
        content_type="image/png",
        reference_datetime=REFERENCE_DATETIME,
    )

    assert isinstance(result.response, SuccessResponse)
    assert result.response.appointment.department == "Dentistry"
    assert result.response.appointment.date == date(2025, 9, 26)
    assert result.response.appointment.time == "15:00"
    assert result.trace.ocr.engine == "tesseract"


def test_low_ocr_confidence_requests_confirmation(monkeypatch):
    ocr_result = OCRResult(
        raw_text="Book dentist next Friday at 3pm",
        confidence=0.4,
        engine="tesseract",
        preprocessing_variant="adaptive-psm6",
    )

    monkeypatch.setattr(
        pipeline,
        "extract_text_from_image",
        lambda image_bytes, content_type: ocr_result,
    )

    monkeypatch.setattr(
        pipeline,
        "extract_entities",
        lambda _text: _complete_extraction(),
    )

    result = pipeline.process_image(
        image_bytes=b"test-image",
        content_type="image/png",
        reference_datetime=REFERENCE_DATETIME,
    )

    assert isinstance(result.response, ClarificationResponse)
    assert "confirm" in result.response.message.casefold()