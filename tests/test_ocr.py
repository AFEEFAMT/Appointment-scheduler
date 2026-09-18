from pathlib import Path

import pytest

from app.config import settings
from app.services.ocr import (
    OCRProcessingError,
    create_typed_text_result,
    extract_text_from_image,
    tesseract_is_available,
)


IMAGE_DIRECTORY = (
    Path(__file__).parent / "fixtures" / "images"
)


def test_tesseract_is_available():
    assert tesseract_is_available() is True


def test_create_typed_text_result():
    result = create_typed_text_result(
        "  Book dentist   next Friday at 3pm  "
    )

    assert result.raw_text == (
        "Book dentist next Friday at 3pm"
    )
    assert result.confidence == 1.0
    assert result.engine == "typed_text"
    assert result.preprocessing_variant is None


def test_reject_empty_typed_text():
    with pytest.raises(OCRProcessingError):
        create_typed_text_result("   ")


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("appointment_clean.png", "image/png"),
        ("appointment_rotated.png", "image/png"),
        ("appointment_noisy.png", "image/png"),
        ("appointment_compressed.jpg", "image/jpeg"),
    ],
)
def test_extract_appointment_from_images(
    filename,
    content_type,
):
    image_path = IMAGE_DIRECTORY / filename

    result = extract_text_from_image(
        image_path.read_bytes(),
        content_type,
    )

    normalized_text = result.raw_text.lower()

    assert "dentist" in normalized_text
    assert "friday" in normalized_text
    assert "3pm" in normalized_text
    assert result.engine == "tesseract"
    assert result.confidence >= 0.60
    assert 0.0 <= result.confidence <= 1.0
    assert result.preprocessing_variant is not None


def test_reject_unsupported_file_type():
    with pytest.raises(
        OCRProcessingError,
        match="Only PNG, JPEG and WebP",
    ):
        extract_text_from_image(
            b"not-a-pdf",
            "application/pdf",
        )


def test_reject_empty_image():
    with pytest.raises(
        OCRProcessingError,
        match="Uploaded image is empty",
    ):
        extract_text_from_image(
            b"",
            "image/png",
        )


def test_reject_oversized_image():
    oversized_data = b"x" * (
        settings.max_image_size_mb * 1024 * 1024 + 1
    )

    with pytest.raises(
        OCRProcessingError,
        match="Image exceeds",
    ):
        extract_text_from_image(
            oversized_data,
            "image/png",
        )


def test_reject_corrupted_image():
    with pytest.raises(
        OCRProcessingError,
        match="not a readable image",
    ):
        extract_text_from_image(
            b"this-is-not-an-image",
            "image/png",
        )