"""Pydantic models shared across the application."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class StrictSchema(BaseModel):
    """Base model that rejects unexpected fields."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class OCRResult(StrictSchema):
    """Text and confidence produced by input processing."""

    raw_text: str = Field(
        min_length=1,
        max_length=10_000,
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    engine: Literal[
        "typed_text",
        "tesseract",
        "gemini_vision",
    ]
    preprocessing_variant: str | None = None


class ExtractedEntities(StrictSchema):
    """Appointment entities extracted from input text."""

    date_phrase: str | None = None
    time_phrase: str | None = None
    department: str | None = None
    ambiguity_reason: str | None = None

    @field_validator(
        "date_phrase",
        "time_phrase",
        "department",
        "ambiguity_reason",
        mode="before",
    )
    @classmethod
    def convert_blank_strings_to_none(
        cls,
        value: object,
    ) -> object:
        """Treat blank model output as a missing value."""

        if isinstance(value, str) and not value.strip():
            return None

        return value


class EntityExtractionResult(StrictSchema):
    """Validated output from the entity extraction stage."""

    entities: ExtractedEntities
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    model_used: str = Field(
        min_length=1,
        max_length=200,
    )


class NormalizedSchedule(StrictSchema):
    """Normalized appointment date and time."""

    date: date
    time: str = Field(
        pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$",
    )
    tz: Literal["Asia/Kolkata"] = "Asia/Kolkata"


class Appointment(StrictSchema):
    """Final structured appointment."""

    department: str = Field(
        min_length=1,
        max_length=100,
    )
    date: date
    time: str = Field(
        pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$",
    )
    tz: Literal["Asia/Kolkata"] = "Asia/Kolkata"


class SuccessResponse(StrictSchema):
    """Response returned for a valid appointment."""

    appointment: Appointment
    status: Literal["ok"] = "ok"


class ClarificationResponse(StrictSchema):
    """Response returned when more information is required."""

    status: Literal["needs_clarification"] = (
        "needs_clarification"
    )
    message: str = Field(
        min_length=1,
        max_length=500,
    )


class ServiceErrorResponse(StrictSchema):
    """Response returned for request and service failures."""

    status: Literal[
        "error",
        "service_unavailable",
    ] = "error"
    message: str = Field(
        min_length=1,
        max_length=500,
    )


class HealthResponse(StrictSchema):
    """Health information returned by the API."""

    status: Literal["ok"] = "ok"
    service: str
    timezone: str


class PipelineTrace(StrictSchema):
    """Internal diagnostic information for a pipeline execution."""

    ocr: OCRResult
    extraction: EntityExtractionResult
    normalized: NormalizedSchedule | None = None
    normalization_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    models_attempted: list[str] = Field(
        default_factory=list,
    )