"""Check extracted date and time phrases against the source text."""

from __future__ import annotations

import re
import unicodedata

from app.schemas import ExtractedEntities


_TOKEN_PATTERN = re.compile(r"[^\W\d_]+|\d+|[:/+\-]", re.UNICODE)

_DATE_MODIFIERS = {
    "next", "this", "last", "following", "coming",
    "after", "before",
}

_TIME_MODIFIERS = {
    "around", "about", "approximately", "after", "before",
    "between", "from", "by",
}


def _tokens(value: str) -> tuple[str, ...]:
    """Ignore casing and harmless spacing without calculating dates."""

    cleaned = unicodedata.normalize("NFKC", value).casefold()
    cleaned = re.sub(r"\ba\s*\.\s*m\s*\.?", "am", cleaned)
    cleaned = re.sub(r"\bp\s*\.\s*m\s*\.?", "pm", cleaned)
    return tuple(_TOKEN_PATTERN.findall(cleaned))


def _is_supported(
    source: tuple[str, ...],
    phrase: str,
    modifiers: set[str],
) -> bool:
    """Require a contiguous source span without dropping a leading modifier."""

    candidate = _tokens(phrase)

    if not candidate:
        return False

    length = len(candidate)

    for start in range(len(source) - length + 1):
        if source[start:start + length] != candidate:
            continue

        # "Friday" must not silently replace "next Friday", for example.
        if start > 0 and source[start - 1] in modifiers:
            continue

        return True

    return False


def ground_entities(
    source_text: str,
    entities: ExtractedEntities,
) -> ExtractedEntities:
    """Remove unsupported schedule phrases and require clarification."""

    source = _tokens(source_text)
    checked = entities.model_copy(deep=True)
    unsupported: list[str] = []

    if checked.date_phrase and not _is_supported(
        source,
        checked.date_phrase,
        _DATE_MODIFIERS,
    ):
        checked.date_phrase = None
        unsupported.append("date")

    if checked.time_phrase and not _is_supported(
        source,
        checked.time_phrase,
        _TIME_MODIFIERS,
    ):
        checked.time_phrase = None
        unsupported.append("time")

    if unsupported:
        fields = " and ".join(unsupported)
        checked.ambiguity_reason = (
            f"The appointment {fields} could not be verified against "
            "the supplied text. Please restate the appointment "
            "date and time explicitly."
        )

    return checked