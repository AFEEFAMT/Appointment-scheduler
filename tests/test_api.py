"""Tests for the appointment scheduling API."""

from datetime import date
from typing import Literal

from fastapi.testclient import TestClient

import app.main as main_module
from app.schemas import (
    Appointment,
    ClarificationResponse,
    EntityExtractionResult,
    ExtractedEntities,
    NormalizedSchedule,
    OCRResult,
    PipelineTrace,
    SuccessResponse,
)
from app.services.extractor import GeminiServiceError
from app.services.ocr import OCRProcessingError
from app.services.pipeline import PipelineResult


client = TestClient(main_module.app)


def _success_result(
    engine: Literal[
        "typed_text",
        "tesseract",
    ] = "typed_text",
) -> PipelineResult:
    """Create a successful pipeline result for API tests."""

    ocr_result = OCRResult(
        raw_text="Book dentist next Friday at 3pm",
        confidence=1.0,
        engine=engine,
        preprocessing_variant=(
            "none"
            if engine == "typed_text"
            else "enhanced-psm6"
        ),
    )

    extraction_result = EntityExtractionResult(
        entities=ExtractedEntities(
            date_phrase="next Friday",
            time_phrase="3pm",
            department="Dentistry",
            ambiguity_reason=None,
        ),
        confidence=1.0,
        model_used="test-model",
    )

    normalized = NormalizedSchedule(
        date=date(2025, 9, 26),
        time="15:00",
        tz="Asia/Kolkata",
    )

    trace = PipelineTrace(
        ocr=ocr_result,
        extraction=extraction_result,
        normalized=normalized,
        normalization_confidence=1.0,
        models_attempted=["test-model"],
    )

    response = SuccessResponse(
        appointment=Appointment(
            department="Dentistry",
            date=date(2025, 9, 26),
            time="15:00",
            tz="Asia/Kolkata",
        )
    )

    return PipelineResult(
        response=response,
        trace=trace,
    )


def _clarification_result() -> PipelineResult:
    """Create a clarification pipeline result for API tests."""

    ocr_result = OCRResult(
        raw_text="Book a dentist appointment",
        confidence=1.0,
        engine="typed_text",
        preprocessing_variant="none",
    )

    extraction_result = EntityExtractionResult(
        entities=ExtractedEntities(
            date_phrase=None,
            time_phrase=None,
            department="Dentistry",
            ambiguity_reason=(
                "The appointment date and time "
                "are missing"
            ),
        ),
        confidence=0.333,
        model_used="test-model",
    )

    trace = PipelineTrace(
        ocr=ocr_result,
        extraction=extraction_result,
        normalized=None,
        normalization_confidence=None,
        models_attempted=["test-model"],
    )

    response = ClarificationResponse(
        message=(
            "Please provide the appointment date "
            "and appointment time."
        )
    )

    return PipelineResult(
        response=response,
        trace=trace,
    )


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["timezone"] == (
        "Asia/Kolkata"
    )


def test_text_appointment_success(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "process_text",
        lambda text, reference_datetime=None: (
            _success_result()
        ),
    )

    response = client.post(
        "/api/v1/appointments/text",
        json={
            "text": (
                "Book dentist next Friday at 3pm"
            ),
            "reference_datetime": (
                "2025-09-19T00:00:00+05:30"
            ),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "appointment": {
            "department": "Dentistry",
            "date": "2025-09-26",
            "time": "15:00",
            "tz": "Asia/Kolkata",
        },
        "status": "ok",
    }


def test_text_appointment_needs_clarification(
    monkeypatch,
):
    monkeypatch.setattr(
        main_module,
        "process_text",
        lambda text, reference_datetime=None: (
            _clarification_result()
        ),
    )

    response = client.post(
        "/api/v1/appointments/text",
        json={
            "text": "Book a dentist appointment",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == (
        "needs_clarification"
    )
    assert "date" in (
        response.json()["message"].casefold()
    )
    assert "time" in (
        response.json()["message"].casefold()
    )


def test_reject_empty_text():
    response = client.post(
        "/api/v1/appointments/text",
        json={"text": ""},
    )

    assert response.status_code == 422
    assert response.json()["status"] == "error"


def test_reject_missing_text():
    response = client.post(
        "/api/v1/appointments/text",
        json={},
    )

    assert response.status_code == 422
    assert response.json()["status"] == "error"


def test_image_appointment_success(monkeypatch):
    captured: dict[str, object] = {}

    def fake_process_image(
        image_bytes: bytes,
        content_type: str,
        reference_datetime: str | None = None,
    ) -> PipelineResult:
        captured["image_bytes"] = image_bytes
        captured["content_type"] = content_type
        captured["reference_datetime"] = (
            reference_datetime
        )

        return _success_result(
            engine="tesseract"
        )

    monkeypatch.setattr(
        main_module,
        "process_image",
        fake_process_image,
    )

    response = client.post(
        "/api/v1/appointments/image",
        files={
            "image": (
                "appointment.png",
                b"fake-image-content",
                "image/png",
            )
        },
        data={
            "reference_datetime": (
                "2025-09-19T00:00:00+05:30"
            ),
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert captured["image_bytes"] == (
        b"fake-image-content"
    )
    assert captured["content_type"] == "image/png"
    assert captured["reference_datetime"] == (
        "2025-09-19T00:00:00+05:30"
    )


def test_reject_missing_image():
    response = client.post(
        "/api/v1/appointments/image",
    )

    assert response.status_code == 422
    assert response.json()["status"] == "error"


def test_ocr_failure_returns_422(monkeypatch):
    def fake_process_image(
        image_bytes: bytes,
        content_type: str,
        reference_datetime: str | None = None,
    ) -> PipelineResult:
        raise OCRProcessingError(
            "The uploaded image could not be decoded."
        )

    monkeypatch.setattr(
        main_module,
        "process_image",
        fake_process_image,
    )

    response = client.post(
        "/api/v1/appointments/image",
        files={
            "image": (
                "corrupted.png",
                b"not-an-image",
                "image/png",
            )
        },
    )

    assert response.status_code == 422
    assert response.json()["status"] == "error"


def test_unsupported_image_returns_415(
    monkeypatch,
):
    def fake_process_image(
        image_bytes: bytes,
        content_type: str,
        reference_datetime: str | None = None,
    ) -> PipelineResult:
        raise OCRProcessingError(
            "Unsupported image type: application/pdf"
        )

    monkeypatch.setattr(
        main_module,
        "process_image",
        fake_process_image,
    )

    response = client.post(
        "/api/v1/appointments/image",
        files={
            "image": (
                "appointment.pdf",
                b"pdf-content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 415
    assert response.json()["status"] == "error"


def test_gemini_failure_returns_503(monkeypatch):
    def fake_process_text(
        text: str,
        reference_datetime: str | None = None,
    ) -> PipelineResult:
        raise GeminiServiceError(
            "All configured Gemini models failed."
        )

    monkeypatch.setattr(
        main_module,
        "process_text",
        fake_process_text,
    )

    response = client.post(
        "/api/v1/appointments/text",
        json={
            "text": (
                "Book dentist tomorrow at 3pm"
            ),
        },
    )

    assert response.status_code == 503
    assert response.json()["status"] == (
        "service_unavailable"
    )
    assert "temporarily unavailable" in (
        response.json()["message"].casefold()
    )