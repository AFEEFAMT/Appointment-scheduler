"""Coordinate OCR, extraction, validation, and normalization."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Response and diagnostic information produced by the pipeline."""

    response: PipelineResponse
    trace: PipelineTrace


def _create_trace(
    ocr_result: OCRResult,
    extraction_result: EntityExtractionResult,
    normalized: NormalizedSchedule | None = None,
    normalization_confidence: float | None = None,
) -> PipelineTrace:
    """Create a trace describing the completed pipeline stages."""

    return PipelineTrace(
        ocr=ocr_result,
        extraction=extraction_result,
        normalized=normalized,
        normalization_confidence=normalization_confidence,
        models_attempted=[extraction_result.model_used],
    )


def _normalization_clarification_message(
    error: NormalizationError,
) -> str:
    """Convert normalization failures into useful clarification questions."""

    message = str(error).strip().rstrip(".")
    lowered_message = message.casefold()

    if "time" in lowered_message:
        return (
            f"{message}. Please provide a specific appointment time, "
            "such as 3pm or 15:00."
        )

    if "date" in lowered_message or "past" in lowered_message:
        return (
            f"{message}. Please provide a specific future appointment date."
        )

    return (
        f"{message}. Please provide a specific future date and time."
    )


def _run_pipeline(
    ocr_result: OCRResult,
    reference_datetime: str | None = None,
) -> PipelineResult:
    """Run the common processing stages for text and image inputs."""

    extraction_result = extract_entities(ocr_result.raw_text)

    combined_confidence = min(
        ocr_result.confidence,
        extraction_result.confidence,
    )

    validation_result = validate_entities(
        entities=extraction_result.entities,
        extraction_confidence=combined_confidence,
    )

    if not validation_result.is_valid:
        trace = _create_trace(
            ocr_result=ocr_result,
            extraction_result=extraction_result,
        )

        return PipelineResult(
            response=ClarificationResponse(
                message=validation_result.clarification_question
                or "Please clarify the appointment details."
            ),
            trace=trace,
        )

    entities = validation_result.entities

    if (
        entities.department is None
        or entities.date_phrase is None
        or entities.time_phrase is None
    ):
        trace = _create_trace(
            ocr_result=ocr_result,
            extraction_result=extraction_result,
        )

        return PipelineResult(
            response=ClarificationResponse(
                message="Please provide the department, date, and time."
            ),
            trace=trace,
        )

    try:
        normalized, normalization_confidence = normalize_schedule(
            date_phrase=entities.date_phrase,
            time_phrase=entities.time_phrase,
            reference_datetime=reference_datetime,
        )
    except NormalizationError as error:
        trace = _create_trace(
            ocr_result=ocr_result,
            extraction_result=extraction_result,
        )

        return PipelineResult(
            response=ClarificationResponse(
                message=_normalization_clarification_message(error)
            ),
            trace=trace,
        )

    appointment = Appointment(
        department=entities.department,
        date=normalized.date,
        time=normalized.time,
        tz=normalized.tz,
    )

    trace = _create_trace(
        ocr_result=ocr_result,
        extraction_result=extraction_result,
        normalized=normalized,
        normalization_confidence=normalization_confidence,
    )

    return PipelineResult(
        response=SuccessResponse(
            appointment=appointment,
        ),
        trace=trace,
    )


def process_text(
    text: str,
    reference_datetime: str | None = None,
) -> PipelineResult:
    """Process a typed appointment request."""

    ocr_result = create_typed_text_result(text)

    return _run_pipeline(
        ocr_result=ocr_result,
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