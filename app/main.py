"""FastAPI application for the appointment scheduling service."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.schemas import (
    ClarificationResponse,
    HealthResponse,
    ServiceErrorResponse,
    SuccessResponse,
)
from app.services.extractor import (
    EntityExtractionError,
    GeminiConfigurationError,
    GeminiQuotaError,
    GeminiServiceError,
)
from app.services.normalizer import (
    NormalizationError,
    parse_reference_datetime,
)
from app.services.ocr import (
    ImageSizeLimitError,
    OCRProcessingError,
    validate_image_input,
)
from app.services.pipeline import (
    PipelineResult,
    process_image,
    process_text,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


class APIRequestError(Exception):
    """Raised when a request fails application validation."""

    def __init__(
        self,
        message: str,
        status_code: int = 422,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


app = FastAPI(
    title="AI-Powered Appointment Scheduler",
    description=(
        "Converts typed or image-based appointment requests "
        "into validated scheduling data."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def _error_response(
    message: str,
    status_code: int,
) -> JSONResponse:
    """Create a consistent JSON error response."""

    response = ServiceErrorResponse(
        status=(
            "service_unavailable"
            if status_code == 503
            else "error"
        ),
        message=message,
    )

    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(mode="json"),
    )


def _validate_reference_datetime(
    value: str | None,
) -> str | None:
    """Validate an optional reference before invoking the pipeline."""

    if value is None:
        return None

    cleaned_value = value.strip()

    if not cleaned_value:
        return None

    try:
        parse_reference_datetime(cleaned_value)
    except NormalizationError as error:
        raise APIRequestError(
            message=(
                "reference_datetime must be a valid ISO-8601 datetime, "
                "such as 2026-09-18T12:00:00+05:30. "
                "Omit it to use the current date and time."
            ),
            status_code=422,
        ) from error

    return cleaned_value


def _log_pipeline_result(
    input_type: str,
    result: PipelineResult,
) -> None:
    """Log processing metadata without storing appointment text."""

    extraction = result.trace.extraction

    model_used = (
        extraction.model_used
        if extraction is not None
        else "not_called"
    )

    logger.info(
        "appointment_processed input_type=%s status=%s "
        "ocr_engine=%s ocr_confidence=%.3f model=%s",
        input_type,
        result.response.status,
        result.trace.ocr.engine,
        result.trace.ocr.confidence,
        model_used,
    )


@app.exception_handler(APIRequestError)
async def api_request_error_handler(
    _request: Request,
    error: APIRequestError,
) -> JSONResponse:
    """Handle invalid application inputs."""

    return _error_response(
        message=error.message,
        status_code=error.status_code,
    )


@app.exception_handler(OCRProcessingError)
async def ocr_error_handler(
    _request: Request,
    error: OCRProcessingError,
) -> JSONResponse:
    """Map typed OCR errors to their declared HTTP status codes."""

    if error.status_code == 503:
        logger.error("OCR service unavailable: %s", error)

        return _error_response(
            message=(
                "The OCR service is temporarily unavailable. "
                "Please try again later."
            ),
            status_code=503,
        )

    return _error_response(
        message=str(error),
        status_code=error.status_code,
    )


@app.exception_handler(GeminiConfigurationError)
async def gemini_configuration_error_handler(
    _request: Request,
    error: GeminiConfigurationError,
) -> JSONResponse:
    """Handle missing or invalid Gemini configuration."""

    logger.error("Gemini configuration error: %s", error)

    return _error_response(
        message=(
            "The AI extraction service is not configured correctly."
        ),
        status_code=503,
    )


@app.exception_handler(GeminiQuotaError)
async def gemini_quota_error_handler(
    _request: Request,
    error: GeminiQuotaError,
) -> JSONResponse:
    """Handle temporary Gemini quota exhaustion."""

    logger.warning("Gemini quota error: %s", error)

    return _error_response(
        message=(
            "The AI extraction service is temporarily unavailable. "
            "Please try again later."
        ),
        status_code=503,
    )


@app.exception_handler(GeminiServiceError)
async def gemini_service_error_handler(
    _request: Request,
    error: GeminiServiceError,
) -> JSONResponse:
    """Handle failures across configured Gemini models."""

    logger.error("Gemini service error: %s", error)

    return _error_response(
        message=(
            "The AI extraction service is temporarily unavailable. "
            "Please try again."
        ),
        status_code=503,
    )


@app.exception_handler(EntityExtractionError)
async def extraction_error_handler(
    _request: Request,
    error: EntityExtractionError,
) -> JSONResponse:
    """Handle entity extraction errors without exposing SDK details."""

    logger.warning("Entity extraction failed: %s", error)

    return _error_response(
        message=(
            "The appointment request could not be processed. "
            "Please check the input or try again."
        ),
        status_code=422,
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    _request: Request,
    _error: RequestValidationError,
) -> JSONResponse:
    """Handle malformed or incomplete API requests."""

    return _error_response(
        message=(
            "The request body is missing or contains invalid fields."
        ),
        status_code=422,
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(
    _request: Request,
    error: Exception,
) -> JSONResponse:
    """Keep unexpected internal errors out of API responses."""

    logger.error(
        "Unexpected application error",
        exc_info=(type(error), error, error.__traceback__),
    )

    return _error_response(
        message="An unexpected server error occurred.",
        status_code=500,
    )


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Provide documentation and health links."""

    return {
        "service": "AI-Powered Appointment Scheduler",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Report whether the API process is running."""

    return HealthResponse(
        service="appointment-scheduler",
        timezone=settings.app_timezone,
    )


@app.post(
    "/api/v1/appointments/text",
    response_model=SuccessResponse | ClarificationResponse,
    responses={
        422: {
            "model": ServiceErrorResponse,
            "description": "Invalid request",
        },
        503: {
            "model": ServiceErrorResponse,
            "description": "AI service unavailable",
        },
    },
    tags=["Appointments"],
)
async def schedule_from_text(
    text: Annotated[
        str,
        Body(
            min_length=1,
            max_length=10_000,
            description="Natural-language appointment request.",
        ),
    ],
    reference_datetime: Annotated[
        str | None,
        Body(
            description=(
                "Optional ISO-8601 reference datetime. "
                "Omit to use the current date and time."
            ),
        ),
    ] = None,
) -> SuccessResponse | ClarificationResponse:
    """Create scheduling data from typed text."""

    if not text.strip():
        raise APIRequestError(
            "Appointment text must not be blank."
        )

    reference_datetime = _validate_reference_datetime(
        reference_datetime
    )

    result = await run_in_threadpool(
        process_text,
        text=text,
        reference_datetime=reference_datetime,
    )

    _log_pipeline_result("text", result)

    return result.response


@app.post(
    "/api/v1/appointments/image",
    response_model=SuccessResponse | ClarificationResponse,
    responses={
        413: {
            "model": ServiceErrorResponse,
            "description": "Image exceeds processing limits",
        },
        415: {
            "model": ServiceErrorResponse,
            "description": "Unsupported image type",
        },
        422: {
            "model": ServiceErrorResponse,
            "description": "Invalid request or unreadable image",
        },
        503: {
            "model": ServiceErrorResponse,
            "description": "OCR or AI service unavailable",
        },
    },
    tags=["Appointments"],
)
async def schedule_from_image(
    image: Annotated[
        UploadFile,
        File(
            description=(
                "PNG, JPEG, or WebP appointment request image."
            ),
        ),
    ],
    reference_datetime: Annotated[
        str | None,
        Form(
            description=(
                "Optional ISO-8601 reference datetime. "
                "Leave blank to use the current date and time."
            ),
        ),
    ] = None,
) -> SuccessResponse | ClarificationResponse:
    """Create scheduling data from an uploaded image."""

    max_size_bytes = (
        settings.max_image_size_mb * 1024 * 1024
    )

    content_type = (
        image.content_type
        or "application/octet-stream"
    )

    try:
        image_bytes = await image.read(
            max_size_bytes + 1
        )
    finally:
        await image.close()

    if len(image_bytes) > max_size_bytes:
        raise ImageSizeLimitError(
            "The uploaded image exceeds the "
            f"{settings.max_image_size_mb} MB size limit."
        )

    validate_image_input(
        image_bytes=image_bytes,
        content_type=content_type,
    )

    reference_datetime = _validate_reference_datetime(
        reference_datetime
    )

    result = await run_in_threadpool(
        process_image,
        image_bytes=image_bytes,
        content_type=content_type,
        reference_datetime=reference_datetime,
    )

    _log_pipeline_result("image", result)

    return result.response