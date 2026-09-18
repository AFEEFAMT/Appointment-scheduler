"""Coordinate OCR, extraction, validation and normalization."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.schemas import (
    Appointment,
    ClarificationResponse,
    EntityExtractionResult,
    NormalizedSchedule,
    OCRResult,
    PipelineTrace,
    SuccessResponse,
)
from app.services.extractor import extract_entities
from app.services.normalizer import (
    NormalizationError,
    normalize_schedule,
)
from app.services.ocr import (
    create_typed_text_result,
    extract_text_from_image,
)
from app.services.validator import validate_entities


PipelineResponse = SuccessResponse | ClarificationResponse


@dataclass(frozen=True)
class PipelineResult:
    """Public response and internal processing trace."""

    response: PipelineResponse
    trace: PipelineTrace


def _create_trace(
    ocr_result: OCRResult,
    extraction_result: EntityExtractionResult | None = None,
    normalized: NormalizedSchedule | None = None,
    normalization_confidence: float | None = None,
) -> PipelineTrace:
    """Record completed stages without fabricating skipped results."""

    return PipelineTrace(
        ocr=ocr_result,
        extraction=extraction_result,
        normalized=normalized,
        normalization_confidence=normalization_confidence,
        models_attempted=(
            [extraction_result.model_used]
            if extraction_result is not None
            else []
        ),
    )


def _normalization_clarification_message(
    error: NormalizationError,
) -> str:
    """Turn normalization failures into relevant clarification messages."""

    message = str(error).strip().rstrip(".")
    lowered_message = message.casefold()

    if "reference datetime" in lowered_message:
        return (
            "Please provide a valid ISO-8601 reference_datetime "
            "or omit it to use the current date and time."
        )

    if (
        "past" in lowered_message
        or "not in the future" in lowered_message
    ):
        return (
            f"{message}. Please provide a future "
            "appointment date and time."
        )

    if "time" in lowered_message:
        return (
            f"{message}. Please provide a specific appointment time, "
            "such as 3pm or 15:00."
        )

    if "date" in lowered_message:
        return (
            f"{message}. Please provide a specific "
            "future appointment date."
        )

    return (
        f"{message}. Please provide a specific future date and time."
    )


def _run_pipeline(
    ocr_result: OCRResult,
    reference_datetime: str | None = None,
) -> PipelineResult:
    """Run shared processing stages with an early image-confidence gate."""

    if (
        ocr_result.engine != "typed_text"
        and ocr_result.confidence
        < settings.ocr_confidence_threshold
    ):
        return PipelineResult(
            response=ClarificationResponse(
                message=(
                    "The image text could not be read reliably. "
                    "Please upload a clearer image or confirm the "
                    "appointment details using typed text."
                )
            ),
            trace=_create_trace(
                ocr_result=ocr_result,
            ),
        )

    extraction_result = extract_entities(
        ocr_result.raw_text
    )

    combined_confidence = min(
        ocr_result.confidence,
        extraction_result.confidence,
    )

    validation_result = validate_entities(
        entities=extraction_result.entities,
        extraction_confidence=combined_confidence,
    )

    if not validation_result.is_valid:
        return PipelineResult(
            response=ClarificationResponse(
                message=(
                    validation_result.clarification_question
                    or "Please clarify the appointment details."
                )
            ),
            trace=_create_trace(
                ocr_result=ocr_result,
                extraction_result=extraction_result,
            ),
        )

    entities = validation_result.entities

    if (
        entities.department is None
        or entities.date_phrase is None
        or entities.time_phrase is None
    ):
        return PipelineResult(
            response=ClarificationResponse(
                message=(
                    "Please provide the department, date, and time."
                )
            ),
            trace=_create_trace(
                ocr_result=ocr_result,
                extraction_result=extraction_result,
            ),
        )

    try:
        normalized, normalization_confidence = normalize_schedule(
            date_phrase=entities.date_phrase,
            time_phrase=entities.time_phrase,
            reference_datetime=reference_datetime,
        )
    except NormalizationError as error:
        return PipelineResult(
            response=ClarificationResponse(
                message=_normalization_clarification_message(
                    error
                )
            ),
            trace=_create_trace(
                ocr_result=ocr_result,
                extraction_result=extraction_result,
            ),
        )

    appointment = Appointment(
        department=entities.department,
        date=normalized.date,
        time=normalized.time,
        tz=normalized.tz,
    )

    return PipelineResult(
        response=SuccessResponse(
            appointment=appointment,
        ),
        trace=_create_trace(
            ocr_result=ocr_result,
            extraction_result=extraction_result,
            normalized=normalized,
            normalization_confidence=normalization_confidence,
        ),
    )


def process_text(
    text: str,
    reference_datetime: str | None = None,
) -> PipelineResult:
    """Process a typed appointment request."""

    return _run_pipeline(
        ocr_result=create_typed_text_result(text),
        reference_datetime=reference_datetime,
    )


def process_image(
    image_bytes: bytes,
    content_type: str,
    reference_datetime: str | None = None,
) -> PipelineResult:
    """Process an appointment request contained in an image."""

    ocr_result = extract_text_from_image(
        image_bytes=image_bytes,
        content_type=content_type,
    )

    return _run_pipeline(
        ocr_result=ocr_result,
        reference_datetime=reference_datetime,
    )