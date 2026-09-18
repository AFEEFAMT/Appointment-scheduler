"""Tests for appointment entity validation."""

import pytest

from app.schemas import ExtractedEntities
from app.services.validator import (
    resolve_department,
    validate_entities,
)


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        ("dentist", "Dentistry"),
        ("dental", "Dentistry"),
        ("skin doctor", "Dermatology"),
        ("cardiologist", "Cardiology"),
        ("ENT", "ENT"),
        ("eye doctor", "Ophthalmology"),
        ("bone doctor", "Orthopedics"),
        ("pediatrician", "Pediatrics"),
        ("gynaecologist", "Gynecology"),
        ("oncologist", "Oncology"),
    ],
)
def test_resolve_known_department_aliases(input_value, expected):
    department, confidence = resolve_department(input_value)

    assert department == expected
    assert confidence >= 0.8


def test_resolve_minor_ocr_error():
    department, confidence = resolve_department("dentstry")

    assert department == "Dentistry"
    assert confidence >= 0.8


def test_accept_clear_unknown_specialty():
    department, confidence = resolve_department("Endocrinology")

    assert department == "Endocrinology"
    assert confidence == 0.8


@pytest.mark.parametrize(
    "input_value",
    [
        None,
        "",
        "doctor",
        "specialist",
        "clinic",
        "dentist or dermatologist",
    ],
)
def test_reject_missing_or_ambiguous_department(input_value):
    department, confidence = resolve_department(input_value)

    assert department is None
    assert confidence == 0.0


def test_valid_complete_entities():
    entities = ExtractedEntities(
        date_phrase="next Friday",
        time_phrase="3pm",
        department="dentist",
    )

    result = validate_entities(
        entities,
        extraction_confidence=1.0,
    )

    assert result.is_valid is True
    assert result.entities.department == "Dentistry"
    assert result.clarification_question is None
    assert result.missing_fields == ()


def test_missing_department_requests_clarification():
    entities = ExtractedEntities(
        date_phrase="tomorrow",
        time_phrase="10am",
        department=None,
    )

    result = validate_entities(entities)

    assert result.is_valid is False
    assert result.missing_fields == ("department",)
    assert result.clarification_question == (
        "Which medical department would you like to book?"
    )


def test_missing_date_requests_clarification():
    entities = ExtractedEntities(
        date_phrase=None,
        time_phrase="10am",
        department="Cardiology",
    )

    result = validate_entities(entities)

    assert result.is_valid is False
    assert result.missing_fields == ("date",)
    assert result.clarification_question == (
        "What date would you like the appointment?"
    )


def test_missing_time_requests_clarification():
    entities = ExtractedEntities(
        date_phrase="tomorrow",
        time_phrase=None,
        department="Cardiology",
    )

    result = validate_entities(entities)

    assert result.is_valid is False
    assert result.missing_fields == ("time",)
    assert result.clarification_question == (
        "What time would you like the appointment?"
    )


def test_multiple_missing_fields():
    entities = ExtractedEntities(
        date_phrase=None,
        time_phrase=None,
        department="dentist",
    )

    result = validate_entities(entities)

    assert result.is_valid is False
    assert result.missing_fields == ("date", "time")
    assert result.clarification_question == (
        "Please provide the appointment date and appointment time."
    )


def test_model_ambiguity_reason_is_respected():
    entities = ExtractedEntities(
        date_phrase="next week",
        time_phrase="morning",
        department="Dentistry",
        ambiguity_reason="The requested date and time are not specific",
    )

    result = validate_entities(entities)

    assert result.is_valid is False
    assert result.clarification_question == (
        "Please clarify the appointment request: "
        "The requested date and time are not specific."
    )


def test_low_extraction_confidence_requests_confirmation():
    entities = ExtractedEntities(
        date_phrase="tomorrow",
        time_phrase="10am",
        department="Dentistry",
    )

    result = validate_entities(
        entities,
        extraction_confidence=0.4,
    )

    assert result.is_valid is False
    assert result.clarification_question == (
        "Please confirm the department, date, and time for the appointment."
    )