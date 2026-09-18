"""Regression coverage for weekday semantics and a known OCR abbreviation."""

from datetime import date
from types import SimpleNamespace

import pytest

from app.config import settings
from app.schemas import ExtractedEntities, SuccessResponse
from app.services import extractor, pipeline
from app.services.grounding import ground_entities
from app.services.normalizer import (
    NormalizationError,
    normalize_schedule,
)


FRIDAY_NOON = "2026-09-18T12:00:00+05:30"


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("Friday", date(2026, 9, 18)),
        ("this Friday", date(2026, 9, 18)),
        ("next Friday", date(2026, 9, 25)),
        ("nxt Friday", date(2026, 9, 25)),
        ("this Sunday", date(2026, 9, 20)),
        ("next Monday", date(2026, 9, 21)),
    ],
)
def test_weekday_policy(phrase, expected):
    schedule, _ = normalize_schedule(
        phrase,
        "3pm",
        FRIDAY_NOON,
    )

    assert schedule.date == expected
    assert schedule.time == "15:00"


@pytest.mark.parametrize(
    "phrase",
    ["Friday", "this Friday"],
)
def test_elapsed_time_does_not_roll_forward_a_week(phrase):
    with pytest.raises(
        NormalizationError,
        match="not in the future",
    ):
        normalize_schedule(
            phrase,
            "11am",
            FRIDAY_NOON,
        )


@pytest.mark.parametrize(
    "phrase",
    ["this Monday", "this Thursday"],
)
def test_this_weekday_in_the_past_is_rejected(phrase):
    with pytest.raises(
        NormalizationError,
        match="past",
    ):
        normalize_schedule(
            phrase,
            "3pm",
            FRIDAY_NOON,
        )


def test_this_week_uses_local_timezone():
    schedule, _ = normalize_schedule(
        "this Friday",
        "3pm",
        "2026-09-18T06:30:00+00:00",
    )

    assert schedule.date == date(2026, 9, 18)
    assert schedule.tz == "Asia/Kolkata"


@pytest.mark.parametrize(
    "phrase",
    ["next Friday", "nxt Friday"],
)
def test_known_ocr_alias_is_supported(phrase):
    entities = ExtractedEntities(
        department="Dentistry",
        date_phrase=phrase,
        time_phrase="3pm",
    )

    checked = ground_entities(
        "book dentist nxt Friday @ 3 pm",
        entities,
    )

    assert checked.date_phrase == phrase
    assert checked.time_phrase == "3pm"
    assert checked.ambiguity_reason is None


@pytest.mark.parametrize(
    "phrase",
    ["Friday", "this Friday", "next Monday"],
)
def test_ocr_alias_does_not_allow_changed_date_meaning(phrase):
    entities = ExtractedEntities(
        department="Dentistry",
        date_phrase=phrase,
        time_phrase="3pm",
    )

    checked = ground_entities(
        "book dentist nxt Friday @ 3 pm",
        entities,
    )

    assert checked.date_phrase is None
    assert checked.ambiguity_reason


def test_ocr_alias_does_not_allow_invented_time():
    entities = ExtractedEntities(
        department="Dentistry",
        date_phrase="next Friday",
        time_phrase="4pm",
    )

    checked = ground_entities(
        "book dentist nxt Friday @ 3 pm",
        entities,
    )

    assert checked.time_phrase is None
    assert checked.ambiguity_reason


@pytest.mark.parametrize(
    "phrase",
    ["next Friday", "nxt Friday"],
)
def test_assignment_noisy_example_through_real_pipeline(
    monkeypatch,
    phrase,
):
    client = SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(
        extractor,
        "_get_gemini_client",
        lambda: client,
    )
    monkeypatch.setattr(
        settings,
        "gemini_primary_model",
        "test-model",
    )
    monkeypatch.setattr(
        settings,
        "gemini_fallback_models",
        "",
    )
    monkeypatch.setattr(
        extractor,
        "call_model",
        lambda **kwargs: ExtractedEntities(
            department="Dentistry",
            date_phrase=phrase,
            time_phrase="3pm",
        ),
    )

    extractor.clear_extraction_cache()

    try:
        result = pipeline.process_text(
            "book dentist nxt Friday @ 3 pm",
            FRIDAY_NOON,
        )

        assert isinstance(result.response, SuccessResponse)
        assert result.response.appointment.date == date(2026, 9, 25)
        assert result.response.appointment.time == "15:00"
        assert result.response.appointment.department == "Dentistry"
    finally:
        extractor.clear_extraction_cache()