from datetime import date, time

import pytest

from app.services.normalizer import (
    NormalizationError,
    normalize_date_phrase,
    normalize_schedule,
    normalize_time_phrase,
    parse_reference_datetime,
)


REFERENCE = parse_reference_datetime(
    "2025-09-19T10:00:00+05:30"
)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("today", date(2025, 9, 19)),
        ("tomorrow", date(2025, 9, 20)),
        ("day after tomorrow", date(2025, 9, 21)),
        ("next Friday", date(2025, 9, 26)),
        ("next Monday", date(2025, 9, 22)),
        ("2026-09-25", date(2026, 9, 25)),
        ("25 Sep 2026", date(2026, 9, 25)),
        ("September 25, 2026", date(2026, 9, 25)),
    ],
)
def test_normalize_valid_dates(phrase, expected):
    result, confidence = normalize_date_phrase(
        phrase,
        REFERENCE,
    )

    assert result == expected
    assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("3pm", time(15, 0)),
        ("3 pm", time(15, 0)),
        ("3:30pm", time(15, 30)),
        ("10:15 AM", time(10, 15)),
        ("15:00", time(15, 0)),
        ("09:00", time(9, 0)),
        ("noon", time(12, 0)),
        ("midnight", time(0, 0)),
    ],
)
def test_normalize_valid_times(phrase, expected):
    result, confidence = normalize_time_phrase(phrase)

    assert result == expected
    assert confidence == 1.0


@pytest.mark.parametrize(
    "phrase",
    [
        "next week",
        "this week",
        "sometime",
        "soon",
        "09/10/2026",
        "2024-09-25",
        "",
    ],
)
def test_reject_ambiguous_or_invalid_dates(phrase):
    with pytest.raises(NormalizationError):
        normalize_date_phrase(phrase, REFERENCE)


@pytest.mark.parametrize(
    "phrase",
    [
        "3",
        "3:00",
        "around 3pm",
        "morning",
        "evening",
        "after lunch",
        "",
    ],
)
def test_reject_ambiguous_times(phrase):
    with pytest.raises(NormalizationError):
        normalize_time_phrase(phrase)


def test_complete_plum_example():
    schedule, confidence = normalize_schedule(
        date_phrase="next Friday",
        time_phrase="3pm",
        reference_datetime="2025-09-19T00:00:00+05:30",
    )

    assert schedule.date == date(2025, 9, 26)
    assert schedule.time == "15:00"
    assert schedule.tz == "Asia/Kolkata"
    assert confidence == 1.0


def test_reject_appointment_in_the_past_today():
    with pytest.raises(NormalizationError):
        normalize_schedule(
            date_phrase="today",
            time_phrase="15:00",
            reference_datetime="2025-09-19T16:00:00+05:30",
        )


def test_reference_datetime_timezone_conversion():
    result = parse_reference_datetime(
        "2025-09-19T04:30:00+00:00"
    )

    assert result.isoformat() == "2025-09-19T10:00:00+05:30"


def test_reject_invalid_reference_datetime():
    with pytest.raises(NormalizationError):
        parse_reference_datetime("not-a-date")