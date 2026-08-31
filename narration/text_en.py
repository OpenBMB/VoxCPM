"""Prepare raw English prose for a TTS engine.

The English twin of :mod:`narration.text_fr`, and it exists for the same reason:
a model reads what it is given, and ``1789``, ``Mr. Dupont``, ``chapter XIV`` or
``$1,250`` are not words. One number read wrong in the middle of a chapter is
enough to break the spell.

What differs from the French module is not the shape but the language's own
awkwardness:

* **Years are said, not counted.** 1789 is "seventeen eighty-nine", not "one
  thousand seven hundred and eighty-nine", and 2005 is "two thousand five". A
  four-digit number in prose is far more often a year than a quantity, so that
  is the default — and a number carrying a thousands separator (``1,789``) is
  never one, which is what tells them apart.
* **The scale words are large and regular** — thousand, million, billion — and
  invariable, so none of the agreement rules that make French numbers hard.
* **Ordinal suffixes are written**: ``1st``, ``2nd``, ``21st``. Their spelling
  is decided by the last two digits, which is why eleventh is not "eleven-first".
* **Titles keep their period.** ``Mr.`` and ``St.`` end in one that is not a
  sentence end, and consuming it would silently glue two sentences together and
  destroy a pause.

Pure text-in / text-out with no dependencies, so it is cheap to test
exhaustively.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional

from .text_fr import Pronunciation, _apply_lexicon, _clean_typography, _strip_markdown

__all__ = [
    "DEFAULT_ROMAN_TRIGGERS_EN",
    "cardinal_en",
    "normalize_english",
    "ordinal_en",
    "year_en",
]

_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = {
    2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
    6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety",
}
_SCALES = ((10**9, "billion"), (10**6, "million"), (10**3, "thousand"))

#: Ordinals whose written form is irregular; everything else takes ``th``.
_ORDINAL_WORDS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}

#: Words after which a Roman numeral is unambiguous. Without a trigger, ``I``
#: and ``C`` are an initial and a letter far more often than they are numbers.
DEFAULT_ROMAN_TRIGGERS_EN = (
    "chapter", "chapters", "part", "parts", "book", "books", "volume", "volumes",
    "act", "acts", "scene", "scenes", "section", "sections", "appendix", "annex",
    "episode", "episodes", "title", "article", "articles", "figure", "plate",
    "lesson", "canto", "world war",
)

_ABBREVIATIONS: tuple[tuple[str, str], ...] = (
    # The period is matched by lookahead and left in place: it may also be the
    # end of the sentence, and eating it would merge two sentences into one.
    # A title is always followed by a name, so its period is never a sentence
    # end and is consumed with it. Left behind, it invents a full stop in the
    # middle of "Mister. Dupont" and the segmentation splits the sentence there.
    (r"\bMrs\.(?=\s+[A-Z])", "Missus"),
    (r"\bMr\.(?=\s+[A-Z])", "Mister"),
    (r"\bMs\.(?=\s+[A-Z])", "Miz"),
    (r"\bDr\.(?=\s+[A-Z])", "Doctor"),
    (r"\bProf\.(?=\s+[A-Z])", "Professor"),
    (r"\bSt\.(?=\s+[A-Z])", "Saint"),
    (r"\bMt\.(?=\s+[A-Z])", "Mount"),
    # This one genuinely can end a sentence, so its period stays.
    (r"\betc(?=\.)", "et cetera"),
    (r"\be\.\s*g\.", "for example"),
    (r"\bi\.\s*e\.", "that is"),
    (r"\bvs\.?(?!\w)", "versus"),
    (r"\bNo\.(?=\s*\d)", "number"),
    (r"\bpp?\.(?=\s*\d)", "page"),
    (r"\bA\.?M\.(?!\w)", "A M"),
    (r"\bP\.?M\.(?!\w)", "P M"),
)

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_STRICT = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")

#: symbol -> (singular, plural, name of the hundredth part)
_CURRENCIES = {
    "$": ("dollar", "dollars", "cents"),
    "£": ("pound", "pounds", "pence"),
    "€": ("euro", "euros", "cents"),
}


def _below_100(n: int) -> str:
    if n < 20:
        return _UNITS[n]
    tens, unit = divmod(n, 10)
    return _TENS[tens] if unit == 0 else f"{_TENS[tens]}-{_UNITS[unit]}"


def _below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    if hundreds == 0:
        return _below_100(rest)
    head = f"{_UNITS[hundreds]} hundred"
    # "and" after the hundreds is the British reading and the one a narrator
    # uses; American English drops it, but never wrongly.
    return head if rest == 0 else f"{head} and {_below_100(rest)}"


def cardinal_en(n: int) -> str:
    """Spell out an integer in English."""
    if n < 0:
        return f"minus {cardinal_en(-n)}"
    if n < 1000:
        return _below_1000(n)

    parts: list[str] = []
    remainder = n
    for value, name in _SCALES:
        count, remainder = divmod(remainder, value)
        if count:
            parts.append(f"{_below_1000(count)} {name}")
    if remainder:
        parts.append(_below_1000(remainder))
    return " ".join(parts)


def ordinal_en(n: int) -> str:
    """Spell out an ordinal: 1 -> first, 21 -> twenty-first, 1000 -> thousandth."""
    words = cardinal_en(n)
    head, separator, last = words.rpartition("-")
    if not separator:
        head, separator, last = words.rpartition(" ")
    if last in _ORDINAL_WORDS:
        last = _ORDINAL_WORDS[last]
    elif last.endswith("y"):
        last = f"{last[:-1]}ieth"
    else:
        last = f"{last}th"
    return f"{head}{separator}{last}"


def year_en(n: int) -> str:
    """Read a year the way it is said rather than counted.

    1789 is "seventeen eighty-nine". The exceptions are the ones a reader makes
    without thinking: whole centuries ("nineteen hundred"), the years either
    side of a millennium ("two thousand five"), and anything with a zero in the
    tens where the pairing would produce "nineteen oh five" — which is right,
    and is what this returns.
    """
    if not 1000 <= n <= 2999:
        return cardinal_en(n)
    high, low = divmod(n, 100)
    if 2000 <= n < 2010:
        return f"two thousand {_UNITS[low]}" if low else "two thousand"
    if low == 0:
        return f"{_below_100(high)} hundred"
    if low < 10:
        return f"{_below_100(high)} oh {_UNITS[low]}"
    return f"{_below_100(high)} {_below_100(low)}"


def roman_to_int(s: str) -> Optional[int]:
    """Value of a well-formed Roman numeral, or None. Strict, so initials survive."""
    s = (s or "").strip().upper()
    if not s or not _ROMAN_STRICT.match(s):
        return None
    total, previous = 0, 0
    for char in reversed(s):
        value = _ROMAN_VALUES[char]
        total += value if value >= previous else -value
        previous = max(previous, value)
    return total or None


def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)
    return text


def _expand_roman(text: str, triggers: Iterable[str]) -> str:
    words = "|".join(re.escape(t) for t in triggers)
    if not words:
        return text

    def replace(match: re.Match) -> str:
        value = roman_to_int(match.group("roman"))
        return match.group(0) if value is None else f"{match.group('trigger')} {cardinal_en(value)}"

    return re.sub(
        rf"(?P<trigger>\b(?:{words}))\s+(?P<roman>[IVXLCDM]+)\b",
        replace,
        text,
        flags=re.IGNORECASE,
    )


def _expand_times(text: str) -> str:
    def replace(match: re.Match) -> str:
        hour, minute = int(match.group(1)), int(match.group(2))
        if minute == 0:
            return f"{cardinal_en(hour)} o'clock"
        if minute < 10:
            return f"{cardinal_en(hour)} oh {_UNITS[minute]}"
        return f"{cardinal_en(hour)} {cardinal_en(minute)}"

    return re.sub(r"\b(\d{1,2}):(\d{2})\b", replace, text)


def _expand_currency(text: str) -> str:
    symbols = "".join(re.escape(s) for s in _CURRENCIES)

    def replace(match: re.Match) -> str:
        singular, plural, subunit = _CURRENCIES[match.group("symbol")]
        whole = int(match.group("whole").replace(",", ""))
        cents = match.group("cents")
        words = f"{cardinal_en(whole)} {singular if whole == 1 else plural}"
        if cents and int(cents):
            # Named, or "one thousand two hundred and fifty dollars fifty"
            # leaves the listener wondering what the fifty was.
            words += f" and {cardinal_en(int(cents))} {subunit}"
        return words

    return re.sub(
        rf"(?P<symbol>[{symbols}])\s?(?P<whole>\d[\d,]*)(?:\.(?P<cents>\d{{2}}))?",
        replace,
        text,
    )


def _expand_percent(text: str) -> str:
    return re.sub(
        r"\b(\d[\d,]*)\s?%",
        lambda m: f"{cardinal_en(int(m.group(1).replace(',', '')))} percent",
        text,
    )


def _expand_ordinal_marks(text: str) -> str:
    return re.sub(
        r"\b(\d+)(?:st|nd|rd|th)\b",
        lambda m: ordinal_en(int(m.group(1))),
        text,
        flags=re.IGNORECASE,
    )


def _expand_numbers(text: str, read_years: bool = True) -> str:
    def replace(match: re.Match) -> str:
        raw = match.group(0)
        digits = raw.replace(",", "")
        value = int(digits)
        # A separator marks a quantity, never a year: "1,789 men" is counted.
        if read_years and "," not in raw and len(digits) == 4 and 1000 <= value <= 2999:
            return year_en(value)
        return cardinal_en(value)

    return re.sub(r"\b\d[\d,]*\b", replace, text)


def normalize_english(
    text: str,
    *,
    lexicon: Optional[Mapping[str, object]] = None,
    expand_roman: bool = True,
    roman_triggers: Iterable[str] = DEFAULT_ROMAN_TRIGGERS_EN,
    strip_markdown: bool = True,
    read_years: bool = True,
) -> str:
    """Rewrite English prose into the words a narrator would speak.

    The passes run in a fixed order because several compete for the same digits:
    times, currency, percentages and ordinal marks each claim their pattern
    before the generic number rule can reach it.
    """
    if not text or not text.strip():
        return ""

    text = _clean_typography(text)
    if strip_markdown:
        text = _strip_markdown(text)
    if lexicon:
        text = _apply_lexicon(text, lexicon)
    text = _expand_abbreviations(text)
    if expand_roman:
        text = _expand_roman(text, roman_triggers)
    text = _expand_times(text)
    text = _expand_currency(text)
    text = _expand_percent(text)
    text = _expand_ordinal_marks(text)
    text = _expand_numbers(text, read_years=read_years)
    return re.sub(r"[^\S\n]{2,}", " ", text).strip()
