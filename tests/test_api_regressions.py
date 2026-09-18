"""Regression tests for request validation and typed OCR errors."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import settings
from app.services import ocr, pipeline


client = TestClient(main_module.app)

IMAGE_ENDPOINT = "/api/v1/appointments/image"
TEXT_ENDPOINT = "/api/v1/appointments/text"

INVALID_REFERENCES = [
    "string",
    "not-a-date",
    "2026-99-18T12:00:00+05:30",
]


def _unexpected_processing(*args, **kwargs):
    """Fail if rejected input reaches an expensive processing stage."""

    pytest.fail("Invalid input reached the processing pipeline")


@pytest.fixture(autouse=True)
def prohibit_external_extraction(monkeypatch):
    """Ensure regression tests never call the Gemini API."""

    monkeypatch.setattr(
        pipeline,
        "extract_entities",
        _unexpected_processing,
    )


@pytest.mark.parametrize(
    "content_type",
    [
        "application/pdf",
        "text/plain",
        "application/octet-stream",
    ],
)
def test_real_unsupported_upload_returns_415(
    monkeypatch,
    content_type,
):
    monkeypatch.setattr(
        main_module,
        "process_image",
        _unexpected_processing,
    )

    response = client.post(
        IMAGE_ENDPOINT,
        files={
            "image": (
                "unsupported.bin",
                b"unsupported-content",
                content_type,
            )
        },
    )

    assert response.status_code == 415
    assert response.json()["status"] == "error"
    assert "Only PNG, JPEG and WebP" in response.json()["message"]


def test_real_corrupted_image_returns_422(monkeypatch):
    # Decoder validation does not need an installed OCR engine.
    monkeypatch.setattr(
        ocr,
        "tesseract_is_available",
        lambda: True,
    )

    response = client.post(
        IMAGE_ENDPOINT,
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
    assert "not a readable image" in response.json()["message"]


def test_real_empty_image_returns_422():
    response = client.post(
        IMAGE_ENDPOINT,
        files={
            "image": (
                "empty.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 422
    assert response.json()["status"] == "error"
    assert "empty" in response.json()["message"].casefold()


def test_real_oversized_upload_returns_413(monkeypatch):
    monkeypatch.setattr(
        settings,
        "max_image_size_mb",
        1,
    )

    response = client.post(
        IMAGE_ENDPOINT,
        files={
            "image": (
                "oversized.png",
                b"x" * (1024 * 1024 + 1),
                "image/png",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["status"] == "error"
    assert "size limit" in response.json()["message"]


def test_whitespace_text_rejected_before_processing(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "process_text",
        _unexpected_processing,
    )

    response = client.post(
        TEXT_ENDPOINT,
        json={"text": "   \n\t   "},
    )

    assert response.status_code == 422
    assert response.json()["status"] == "error"
    assert "blank" in response.json()["message"]


@pytest.mark.parametrize("reference", INVALID_REFERENCES)
def test_invalid_text_reference_rejected_before_processing(
    monkeypatch,
    reference,
):
    monkeypatch.setattr(
        main_module,
        "process_text",
        _unexpected_processing,
    )

    response = client.post(
        TEXT_ENDPOINT,
        json={
            "text": "Book dentist tomorrow at 3pm",
            "reference_datetime": reference,
        },
    )

    assert response.status_code == 422
    assert response.json()["status"] == "error"
    assert "reference_datetime" in response.json()["message"]
    assert "specific appointment time" not in response.json()["message"]


@pytest.mark.parametrize("reference", INVALID_REFERENCES)
def test_invalid_image_reference_rejected_before_processing(
    monkeypatch,
    reference,
):
    monkeypatch.setattr(
        main_module,
        "process_image",
        _unexpected_processing,
    )

    response = client.post(
        IMAGE_ENDPOINT,
        files={
            "image": (
                "appointment.png",
                b"placeholder-content",
                "image/png",
            )
        },
        data={"reference_datetime": reference},
    )

    assert response.status_code == 422
    assert response.json()["status"] == "error"
    assert "reference_datetime" in response.json()["message"]


@pytest.mark.parametrize(
    ("error_type", "expected_code", "expected_status"),
    [
        (ocr.UnsupportedImageTypeError, 415, "error"),
        (ocr.ImageSizeLimitError, 413, "error"),
        (ocr.OCRUnavailableError, 503, "service_unavailable"),
    ],
)
def test_error_status_does_not_depend_on_message(
    monkeypatch,
    error_type,
    expected_code,
    expected_status,
):
    def failing_process_image(*args, **kwargs):
        raise error_type("A generic processing failure.")

    monkeypatch.setattr(
        main_module,
        "process_image",
        failing_process_image,
    )

    response = client.post(
        IMAGE_ENDPOINT,
        files={
            "image": (
                "appointment.png",
                b"placeholder-content",
                "image/png",
            )
        },
    )

    assert response.status_code == expected_code
    assert response.json()["status"] == expected_status


def test_missing_ocr_engine_returns_503(monkeypatch):
    monkeypatch.setattr(
        ocr,
        "tesseract_is_available",
        lambda: False,
    )

    response = client.post(
        IMAGE_ENDPOINT,
        files={
            "image": (
                "appointment.png",
                b"placeholder-content",
                "image/png",
            )
        },
    )

    assert response.status_code == 503
    assert response.json()["status"] == "service_unavailable"
    assert "OCR service" in response.json()["message"]


def test_real_pixel_limit_returns_413(monkeypatch):
    monkeypatch.setattr(
        ocr,
        "tesseract_is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        ocr,
        "MAX_IMAGE_PIXELS",
        1,
    )

    image_path = (
        Path(__file__).parent
        / "fixtures"
        / "images"
        / "appointment_clean.png"
    )

    response = client.post(
        IMAGE_ENDPOINT,
        files={
            "image": (
                image_path.name,
                image_path.read_bytes(),
                "image/png",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["status"] == "error"
    assert "dimensions" in response.json()["message"]