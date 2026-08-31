"""Opening and closing credits, the way distributors require them.

A finished audiobook is not just the book read aloud. Every distributor — ACX
and Audible, and behind them Amazon, Apple Books, Kobo, Google Play — requires
the recording to *announce itself*: the first file opens on the title, the
author and the narrator, and the last one closes by naming them again. A
submission without them is rejected at quality review before anyone listens to
a word of the prose.

The rules this module encodes:

* **Opening credits** carry title, subtitle when there is one, author, narrator.
  They are the very first thing heard.
* **Closing credits** name the work and its author again, then the narrator, and
  may carry production and rights information.
* **Synthetic narration is disclosed.** Where no human narrator is named, the
  credit says the reading is a synthetic voice. Audible distributes such titles
  through a separate programme and labels them; claiming a machine reading as a
  human performance is what gets an account closed, so the disclosure is the
  default and switching it off has to be a deliberate act.

Only the *text* lives here. It is narrated by the same voice, with the same
seed, through the same pipeline as the book — which is exactly why the credits
sound like the same narrator rather than a bolted-on announcement.

The wording is French, like the rest of the narration this fork produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

__all__ = [
    "CLOSING_TITLE",
    "OPENING_TITLE",
    "BookCredits",
    "titles_for",
    "SYNTHETIC_DISCLOSURE",
]

#: Chapter titles used for the two credit files, and their marker names in the
#: assembled M4B.
OPENING_TITLE = "Générique de début"
CLOSING_TITLE = "Générique de fin"

#: Said when no human narrator is named. Not a disclaimer bolted on for safety:
#: distributors require synthetic narration to be identified as such.
SYNTHETIC_DISCLOSURE = "une voix de synthèse"

#: Everything the credits say, per language. Kept as data rather than as
#: branches in the methods, so adding a language is adding an entry.
_WORDS = {
    "fr": {
        "opening_title": OPENING_TITLE,
        "closing_title": CLOSING_TITLE,
        "synthetic": SYNTHETIC_DISCLOSURE,
        "by": "de",
        "read_by": "Lu par {narrator}",
        "you_heard": "Vous venez d'écouter {work}",
        "read_by_inline": ", lu par {narrator}",
        "produced_by_year": "Enregistrement produit par {publisher}, {year}",
        "produced_by": "Enregistrement produit par {publisher}",
        "recorded_in": "Enregistrement réalisé en {year}",
        "public_domain": "Texte du domaine public",
        "untitled": "Ce livre",
        "missing_title": "le titre",
        "missing_author": "l'auteur",
        "missing_narrator": "le narrateur (ou la mention de voix de synthèse)",
    },
    "en": {
        "opening_title": "Opening credits",
        "closing_title": "Closing credits",
        "synthetic": "a synthetic voice",
        "by": "by",
        "read_by": "Narrated by {narrator}",
        "you_heard": "You have been listening to {work}",
        "read_by_inline": ", narrated by {narrator}",
        "produced_by_year": "Produced by {publisher}, {year}",
        "produced_by": "Produced by {publisher}",
        "recorded_in": "Recorded in {year}",
        "public_domain": "This text is in the public domain",
        "untitled": "This book",
        "missing_title": "the title",
        "missing_author": "the author",
        "missing_narrator": "the narrator (or the synthetic voice disclosure)",
    },
}

# A title already carrying its author ("Autour de la Lune, par Jules Verne")
# would otherwise be announced as "…, par Jules Verne, de Jules Verne".
_AUTHOR_PREPOSITIONS = frozenset({"par", "de", "by"})


def _clean(text: str) -> str:
    """One line, single spaces, no trailing punctuation of its own."""
    collapsed = " ".join((text or "").split())
    return collapsed.rstrip(" .;,:")


def _sentence(text: str) -> str:
    """End a credit line on a full stop, so the narrator lands it."""
    text = _clean(text)
    return f"{text}." if text else ""


@dataclass(frozen=True)
class BookCredits:
    """What the recording says about itself, at its two ends."""

    title: str
    author: str = ""
    #: Human narrator. Left empty for a synthetic reading, which is then
    #: disclosed rather than passed off as a performance.
    narrator: str = ""
    #: Name given to the synthetic voice — "Aurore Cabonet", "Gabriel Adam".
    #: A catalogue read by the same voice deserves to credit it by name, the
    #: way a publisher credits a virtual voice. It does **not** replace the
    #: disclosure: the credit says the name *and* that the voice is synthetic,
    #: because a name alone would present a machine as a performer, which is
    #: exactly what distributors require not to happen.
    voice_name: str = ""
    subtitle: str = ""
    publisher: str = ""
    year: str = ""
    #: Public-domain works are worth saying so: it answers the rights question
    #: a distributor asks about every uploaded recording.
    public_domain: bool = False
    #: Turning this off is a deliberate act — see the module docstring.
    disclose_synthetic: bool = True
    #: "fr" or "en". Anything else falls back to French, which is what this
    #: fork narrates by default.
    language: str = "fr"

    @property
    def _words(self) -> dict:
        return _WORDS.get(self.language, _WORDS["fr"])

    @property
    def narrator_credit(self) -> str:
        """Who the recording says read it.

        Three cases rather than two. A human narrator is named and that is all.
        A synthetic voice with a name is named *and* disclosed — "Aurore
        Cabonet, une voix de synthèse" — because the name alone would credit a
        performance that never happened. A synthetic voice without a name is
        disclosed as before.
        """
        narrator = _clean(self.narrator)
        if narrator:
            return narrator

        voice_name = _clean(self.voice_name)
        synthetic = self._words["synthetic"] if self.disclose_synthetic else ""
        if voice_name and synthetic:
            return f"{voice_name}, {synthetic}"
        return voice_name or synthetic

    def _work(self) -> str:
        """« Title », by Author — the phrase both credits are built around."""
        words = self._words
        title = _clean(self.title) or words["untitled"]
        author = _clean(self.author)
        # French quotes in both languages: they are heard as a pause rather than
        # read as characters, and they keep a title made of ordinary words from
        # dissolving into the sentence around it.
        piece = f"« {title} »"
        if self.subtitle:
            piece += f", {_clean(self.subtitle)}"
        if author and not _names_the_author(title, author):
            piece += f", {words['by']} {author}"
        return piece

    def opening(self) -> str:
        """The first thing heard: title, subtitle, author, narrator."""
        lines: List[str] = [_sentence(self._work())]
        narrator = self.narrator_credit
        if narrator:
            lines.append(_sentence(self._words["read_by"].format(narrator=narrator)))
        return "\n\n".join(line for line in lines if line)

    def closing(self) -> str:
        """The last thing heard: the work named again, then the production."""
        words = self._words
        narrator = self.narrator_credit
        first = words["you_heard"].format(work=self._work())
        if narrator:
            first += words["read_by_inline"].format(narrator=narrator)
        lines: List[str] = [_sentence(first)]

        publisher = _clean(self.publisher)
        year = _clean(self.year)
        if publisher and year:
            lines.append(
                _sentence(words["produced_by_year"].format(publisher=publisher, year=year))
            )
        elif publisher:
            lines.append(_sentence(words["produced_by"].format(publisher=publisher)))
        elif year:
            lines.append(_sentence(words["recorded_in"].format(year=year)))

        if self.public_domain:
            lines.append(_sentence(words["public_domain"]))
        return "\n\n".join(line for line in lines if line)

    def missing_for_distribution(self) -> List[str]:
        """What a distributor would send this recording back for.

        Reported rather than raised: a draft narration is a perfectly reasonable
        thing to produce, and the gaps only matter on the day it is uploaded.
        """
        words = self._words
        missing: List[str] = []
        if not _clean(self.title):
            missing.append(words["missing_title"])
        if not _clean(self.author):
            missing.append(words["missing_author"])
        if not self.narrator_credit:
            missing.append(words["missing_narrator"])
        return missing


def titles_for(language: str) -> tuple:
    """The two chapter titles, in the language the credits are spoken in."""
    words = _WORDS.get(language, _WORDS["fr"])
    return words["opening_title"], words["closing_title"]


def _names_the_author(title: str, author: str) -> bool:
    """Whether the title already ends by naming the author.

    Matched from the end rather than by searching forwards: "Autour **de** la
    Lune, par Jules Verne" has a ``de`` long before the one that matters.
    """
    title_key = _clean(title).casefold()
    author_key = _clean(author).casefold()
    if not author_key or not title_key.endswith(author_key):
        return False
    head = title_key[: -len(author_key)].strip()
    if head.endswith((",", "-", "—", "–", ":")):
        return True
    words = head.rstrip(" ,-—–:").split()
    return bool(words) and words[-1] in _AUTHOR_PREPOSITIONS
