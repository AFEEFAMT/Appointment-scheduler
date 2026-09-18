"""Validate and normalize extracted appointment entities."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from app.schemas import ExtractedEntities


@dataclass(frozen=True)
class ValidationResult:
    """Result produced after validating extracted appointment entities."""

    is_valid: bool
    entities: ExtractedEntities
    clarification_question: str | None = None
    missing_fields: tuple[str, ...] = ()


DEPARTMENT_ALIASES: dict[str, set[str]] = {
    "Cardiology": {
        "cardiology",
        "cardiologist",
        "heart",
        "heart doctor",
    },
    "Dentistry": {
        "dentistry",
        "dentist",
        "dental",
        "orthodontist",
        "oral health",
    },
    "Dermatology": {
        "dermatology",
        "dermatologist",
        "skin",
        "skin doctor",
    },
    "ENT": {
        "ent",
        "ear nose throat",
        "ear nose and throat",
        "otolaryngology",
        "otolaryngologist",
    },
    "Gastroenterology": {
        "gastroenterology",
        "gastroenterologist",
        "stomach",
        "digestive",
    },
    "General Medicine": {
        "general medicine",
        "general physician",
        "physician",
        "family medicine",
        "primary care",
    },
    "General Surgery": {
        "general surgery",
        "surgeon",
        "surgery",
    },
    "Gynecology": {
        "gynecology",
        "gynaecology",
        "gynecologist",
        "gynaecologist",
        "women's health",
        "womens health",
    },
    "Neurology": {
        "neurology",
        "neurologist",
        "brain",
        "nerve specialist",
    },
    "Oncology": {
        "oncology",
        "oncologist",
        "cancer specialist",
    },
    "Ophthalmology": {
        "ophthalmology",
        "ophthalmologist",
        "eye",
        "eye doctor",
    },
    "Orthopedics": {
        "orthopedics",
        "orthopaedics",
        "orthopedic",
        "orthopaedic",
        "bone",
        "bone doctor",
    },
    "Pediatrics": {
        "pediatrics",
        "paediatrics",
        "pediatrician",
        "paediatrician",
        "child specialist",
        "children's doctor",
        "childrens doctor",
    },
    "Psychiatry": {
        "psychiatry",
        "psychiatrist",
        "mental health",
    },
    "Pulmonology": {
        "pulmonology",
        "pulmonologist",
        "lung",
        "lung specialist",
    },
    "Urology": {
        "urology",
        "urologist",
    },
}


AMBIGUOUS_DEPARTMENT_TERMS = {
    "appointment",
    "checkup",
    "clinic",
    "consultation",
    "doctor",
    "hospital",
    "medical",
    "specialist",
}


ALIAS_TO_DEPARTMENT = {
    alias: department
    for department, aliases in DEPARTMENT_ALIASES.items()
    for alias in aliases
}


def _clean_department(value: str) -> str:
    """Prepare a department phrase for matching."""

    cleaned = value.casefold().strip()
    cleaned = re.sub(r"[^a-z0-9\s'-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def _format_department(value: str) -> str:
    """Format an unknown but valid specialty name consistently."""

    uppercase_names = {
        "ent": "ENT",
    }

    if value in uppercase_names:
        return uppercase_names[value]

    return value.title()


def resolve_department(
    department: str | None,
) -> tuple[str | None, float]:
    """
    Resolve aliases and minor OCR errors to a canonical department name.

    Unknown but clear specialty names are accepted instead of being restricted
    to a fixed department catalogue.
    """

    if department is None or not department.strip():
        return None, 0.0

    cleaned = _clean_department(department)

    if not cleaned:
        return None, 0.0

    if cleaned in ALIAS_TO_DEPARTMENT:
        return ALIAS_TO_DEPARTMENT[cleaned], 1.0

    if cleaned in AMBIGUOUS_DEPARTMENT_TERMS:
        return None, 0.0

    if " or " in cleaned or "/" in department:
        return None, 0.0

    match = process.extractOne(
        cleaned,
        ALIAS_TO_DEPARTMENT.keys(),
        scorer=fuzz.ratio,
    )

    if match is not None:
        matched_alias, score, _ = match

        if score >= 88:
            canonical_name = ALIAS_TO_DEPARTMENT[matched_alias]
            return canonical_name, round(score / 100, 3)

    if not re.fullmatch(r"[a-z][a-z\s'-]{1,59}", cleaned):
        return None, 0.0

    return _format_department(cleaned), 0.8


def _build_clarification_question(
    missing_fields: list[str],
    ambiguity_reason: str | None,
) -> str:
    """Build one concise clarification question."""

    if missing_fields:
        field_set = set(missing_fields)

        if field_set == {"department"}:
            return "Which medical department would you like to book?"

        if field_set == {"date"}:
            return "What date would you like the appointment?"

        if field_set == {"time"}:
            return "What time would you like the appointment?"

        readable_fields = {
            "department": "medical department",
            "date": "appointment date",
            "time": "appointment time",
        }

        details = [
            readable_fields[field]
            for field in ("department", "date", "time")
            if field in field_set
        ]

        if len(details) == 2:
            joined_details = " and ".join(details)
        else:
            joined_details = ", ".join(details[:-1])
            joined_details += f", and {details[-1]}"

        return f"Please provide the {joined_details}."

    if ambiguity_reason:
        reason = ambiguity_reason.strip().rstrip(".")
        return f"Please clarify the appointment request: {reason}."

    return "Please confirm the department, date, and time for the appointment."


def validate_entities(
    entities: ExtractedEntities,
    extraction_confidence: float = 1.0,
) -> ValidationResult:
    """Validate extracted fields before date and time normalization."""

    department, department_confidence = resolve_department(
        entities.department
    )

    validated_entities = entities.model_copy(
        update={"department": department}
    )

    missing_fields: list[str] = []

    if department is None:
        missing_fields.append("department")

    if not entities.date_phrase:
        missing_fields.append("date")

    if not entities.time_phrase:
        missing_fields.append("time")

    needs_clarification = bool(
        missing_fields
        or entities.ambiguity_reason
        or extraction_confidence < 0.5
        or department_confidence < 0.5
    )

    if needs_clarification:
        question = _build_clarification_question(
            missing_fields=missing_fields,
            ambiguity_reason=entities.ambiguity_reason,
        )

        return ValidationResult(
            is_valid=False,
            entities=validated_entities,
            clarification_question=question,
            missing_fields=tuple(missing_fields),
        )

    return ValidationResult(
        is_valid=True,
        entities=validated_entities,
    )