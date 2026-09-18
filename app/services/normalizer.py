"""Normalize appointment dates and times using explicit scheduling rules."""

import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import dateparser

from app.config import settings
from app.schemas import NormalizedSchedule


class NormalizationError(ValueError):
    """Raised when a date or time cannot be safely normalized."""


WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

AMBIGUOUS_DATE_PHRASES = {
    "next week",
    "this week",
    "sometime",
    "later",
    "soon",
    "weekend",
    "next weekend",
}

AMBIGUOUS_TIME_WORDS = {
    "around",
    "about",
    "approximately",
    "morning",
    "afternoon",
    "evening",
    "night",
    "lunch",
    "later",
    "sometime",
}


def clean_phrase(value: str) -> str:
    """Normalize spaces and casing without changing meaning."""

    return " ".join(value.strip().lower().split())


def parse_reference_datetime(
    value: str | datetime | None = None,
) -> datetime:
    """Parse a reference time in the application timezone."""

    timezone = ZoneInfo(settings.app_timezone)

    if value is None:
        return datetime.now(timezone)

    if isinstance(value, datetime):
        parsed = value
    else:
        normalized_value = value.strip().replace("Z", "+00:00")

        try:
            parsed = datetime.fromisoformat(normalized_value)
        except ValueError as error:
            raise NormalizationError(
                "Invalid reference datetime"
            ) from error

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone)

    return parsed.astimezone(timezone)


def normalize_date_phrase(
    date_phrase: str,
    reference: datetime,
) -> tuple[date, float]:
    """Convert an explicit or relative date phrase into a date."""

    if not date_phrase or not date_phrase.strip():
        raise NormalizationError("Appointment date is missing")

    phrase = clean_phrase(date_phrase)
    phrase = re.sub(r"^(?:on|for)\s+", "", phrase)

    if phrase in AMBIGUOUS_DATE_PHRASES:
        raise NormalizationError("Appointment date is ambiguous")

    relative_dates = {
        "today": 0,
        "tomorrow": 1,
        "day after tomorrow": 2,
    }

    if phrase in relative_dates:
        result = reference.date() + timedelta(
            days=relative_dates[phrase]
        )
        return result, 1.0

    weekday_match = re.fullmatch(
        r"(?:(next|this|nxt)\s+)?"
        r"(monday|tuesday|wednesday|thursday|friday|"
        r"saturday|sunday)",
        phrase,
    )

    if weekday_match:
        modifier = weekday_match.group(1)
        target_weekday = WEEKDAYS[weekday_match.group(2)]

        if modifier == "this":
            # The current week runs from Monday through Sunday.
            days_ahead = target_weekday - reference.weekday()

            if days_ahead < 0:
                raise NormalizationError(
                    "Appointment date is in the past"
                )
        else:
            days_ahead = (
                target_weekday - reference.weekday()
            ) % 7

            # "Next" excludes today; an unqualified weekday may mean today.
            if modifier in {"next", "nxt"} and days_ahead == 0:
                days_ahead = 7

        return (
            reference.date() + timedelta(days=days_ahead),
            1.0,
        )

    try:
        iso_date = date.fromisoformat(phrase)
    except ValueError:
        iso_date = None

    if iso_date is not None:
        if iso_date < reference.date():
            raise NormalizationError(
                "Appointment date is in the past"
            )

        return iso_date, 1.0

    numeric_date = re.fullmatch(
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})",
        phrase,
    )

    if numeric_date:
        first = int(numeric_date.group(1))
        second = int(numeric_date.group(2))

        if first <= 12 and second <= 12:
            raise NormalizationError(
                "Numeric date format is ambiguous"
            )

    if not re.search(
        r"\d|jan|feb|mar|apr|may|jun|jul|aug|"
        r"sep|oct|nov|dec",
        phrase,
    ):
        raise NormalizationError("Appointment date is ambiguous")

    parsed = dateparser.parse(
        phrase,
        languages=["en"],
        settings={
            "RELATIVE_BASE": reference,
            "PREFER_DATES_FROM": "future",
            "DATE_ORDER": "DMY",
            "PREFER_LOCALE_DATE_ORDER": False,
        },
    )

    if parsed is None:
        raise NormalizationError(
            "Appointment date could not be parsed"
        )

    parsed_date = parsed.date()

    if parsed_date < reference.date():
        raise NormalizationError(
            "Appointment date is in the past"
        )

    return parsed_date, 0.90


def normalize_time_phrase(
    time_phrase: str,
) -> tuple[time, float]:
    """Convert an explicit time phrase into a time value."""

    if not time_phrase or not time_phrase.strip():
        raise NormalizationError("Appointment time is missing")

    phrase = clean_phrase(time_phrase)
    phrase = re.sub(r"^(?:at|@)\s*", "", phrase)

    if phrase == "noon":
        return time(12, 0), 1.0

    if phrase == "midnight":
        return time(0, 0), 1.0

    # Complete-word matching prevents "night" matching "midnight".
    if any(
        re.search(rf"\b{re.escape(word)}\b", phrase)
        for word in AMBIGUOUS_TIME_WORDS
    ):
        raise NormalizationError("Appointment time is ambiguous")

    phrase = phrase.replace(".", "")

    twelve_hour_match = re.fullmatch(
        r"(0?[1-9]|1[0-2])"
        r"(?::([0-5]\d))?\s*(am|pm)",
        phrase,
    )

    if twelve_hour_match:
        hour = int(twelve_hour_match.group(1))
        minute = int(twelve_hour_match.group(2) or 0)
        period = twelve_hour_match.group(3)

        if period == "am" and hour == 12:
            hour = 0
        elif period == "pm" and hour != 12:
            hour += 12

        return time(hour, minute), 1.0

    twenty_four_hour_match = re.fullmatch(
        r"(\d{2}):([0-5]\d)",
        phrase,
    )

    if twenty_four_hour_match:
        hour = int(twenty_four_hour_match.group(1))
        minute = int(twenty_four_hour_match.group(2))

        if hour > 23:
            raise NormalizationError(
                "Invalid appointment time"
            )

        return time(hour, minute), 1.0

    raise NormalizationError("Appointment time is ambiguous")


def normalize_schedule(
    date_phrase: str,
    time_phrase: str,
    reference_datetime: str | datetime | None = None,
) -> tuple[NormalizedSchedule, float]:
    """Normalize and validate a complete appointment schedule."""

    reference = parse_reference_datetime(reference_datetime)

    normalized_date, date_confidence = normalize_date_phrase(
        date_phrase,
        reference,
    )

    normalized_time, time_confidence = normalize_time_phrase(
        time_phrase
    )

    appointment_datetime = datetime.combine(
        normalized_date,
        normalized_time,
    ).replace(tzinfo=ZoneInfo(settings.app_timezone))

    if appointment_datetime <= reference:
        raise NormalizationError(
            "Appointment date and time are not in the future"
        )

    schedule = NormalizedSchedule(
        date=normalized_date,
        time=normalized_time.strftime("%H:%M"),
        tz=settings.app_timezone,
    )

    confidence = min(date_confidence, time_confidence)

    return schedule, confidence