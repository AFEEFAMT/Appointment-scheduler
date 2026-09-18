"""Regression tests for source-grounded schedule extraction."""

import pytest

from app.schemas import ExtractedEntities
from app.services.grounding import ground_entities


@pytest.mark.parametrize(
    ("source", "date_phrase", "time_phrase"),
    [
        ("Book dentist next Friday at 3pm", "next Friday", "3pm"),
        ("Book dentist NEXT FRIDAY at 3 PM", "next Friday", "3pm"),
        ("Book dentist tomorrow at 10:30 AM", "tomorrow", "10:30 am"),
        ("Book dentist tomorrow at 3 p.m.", "tomorrow", "3pm"),
    ],
)
def test_accept_source_supported_phrases(source, date_phrase, time_phrase):
    entities = ExtractedEntities(
        department="Dentistry",
        date_phrase=date_phrase,
        time_phrase=time_phrase,
    )

    result = ground_entities(source, entities)

    assert result.date_phrase == date_phrase
    assert result.time_phrase == time_phrase
    assert result.ambiguity_reason is None


@pytest.mark.parametrize(
    ("source", "date_phrase", "time_phrase", "missing"),
    [
        ("Book dentist at 3pm", "tomorrow", "3pm", "date_phrase"),
        ("Book dentist tomorrow", "tomorrow", "3pm", "time_phrase"),
        (
            "Book dentist next Friday at 3pm",
            "Friday",
            "3pm",
            "date_phrase",
        ),
        (
            "Book dentist tomorrow around 3pm",
            "tomorrow",
            "3pm",
            "time_phrase",
        ),
        ("Book dentist sometime at 3pm", "May", "3pm", "date_phrase"),
    ],
)
def test_remove_unsupported_or_shortened_phrases(
    source, date_phrase, time_phrase, missing
):
    entities = ExtractedEntities(
        department="Dentistry",
        date_phrase=date_phrase,
        time_phrase=time_phrase,
    )

    result = ground_entities(source, entities)

    assert getattr(result, missing) is None
    assert result.ambiguity_reason
    assert entities.date_phrase == date_phrase
    assert entities.time_phrase == time_phrase