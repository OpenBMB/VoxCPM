"""Prepare raw French prose for a TTS engine.

A TTS model reads what it is given. Left alone it will stumble on ``1789``,
``M. Dupont``, ``XIVe siècle`` or ``14h30`` — and a single mispronounced number
in the middle of a chapter is enough to break the spell of an audiobook. This
module rewrites those forms as the words a human narrator would actually say,
before the text is ever segmented or synthesized.

Everything here is pure text-in / text-out with no dependencies, so it is cheap
to test exhaustively — which matters, because French number agreement has more
edge cases than it looks (``quatre-vingts`` but ``quatre-vingt-un``, ``deux
cents`` but ``deux cent mille``).

Typical use::

    from narration.text_fr import normalize_french, load_lexicon
    spoken = normalize_french(raw_chapter, lexicon=load_lexicon("conf/pronunciation_fr.json"))

Ordering inside :func:`normalize_french` is significant: currency, times,
percentages and ordinal marks each consume their digits before the generic
number rule can reach them, and abbreviations are expanded before segmentation
so that ``M.`` no longer looks like the end of a sentence.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

__all__ = [
    "normalize_french",
    "cardinal",
    "ordinal",
    "roman_to_int",
    "Pronunciation",
    "load_lexicon",
    "DEFAULT_ROMAN_TRIGGERS",
]


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

_UNITS = [
    "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
    "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
    "dix-sept", "dix-huit", "dix-neuf",
]
_TENS = {2: "vingt", 3: "trente", 4: "quarante", 5: "cinquante", 6: "soixante"}

# Ordered high to low. ``mille`` is invariable; ``million``/``milliard`` are
# nouns and take a plural s.
_SCALES = ((10**9, "milliard"), (10**6, "million"), (10**3, "mille"))


def _below_100(n: int, final: bool) -> str:
    """``final`` is False when another numeral word follows, which suppresses the
    plural s of ``quatre-vingts`` (``quatre-vingt mille``, not ``quatre-vingts mille``)."""
    if n < 20:
        return _UNITS[n]
    if n < 70:
        tens, unit = divmod(n, 10)
        word = _TENS[tens]
        if unit == 0:
            return word
        if unit == 1:
            return f"{word} et un"
        return f"{word}-{_UNITS[unit]}"
    if n < 80:
        rest = n - 60  # 10..19
        if rest == 11:
            return "soixante et onze"
        return f"soixante-{_UNITS[rest]}"
    rest = n - 80  # 0..19
    if rest == 0:
        return "quatre-vingts" if final else "quatre-vingt"
    return f"quatre-vingt-{_UNITS[rest]}"


def _below_1000(n: int, final: bool) -> str:
    hundreds, rest = divmod(n, 100)
    if hundreds == 0:
        return _below_100(rest, final)
    head = "cent" if hundreds == 1 else f"{_UNITS[hundreds]} cent"
    if rest == 0:
        # ``cent`` agrees only when it is multiplied and ends the number.
        return f"{head}s" if (hundreds > 1 and final) else head
    return f"{head} {_below_100(rest, final)}"


def cardinal(n: int, final: bool = True) -> str:
    """Spell out an integer in French.

    ``final=False`` suppresses the plural s on ``cent`` and ``quatre-vingt``, as
    required when a numeral word follows (``deux cent mille``).
    """
    if n < 0:
        return f"moins {cardinal(-n, final)}"
    if n < 1000:
        return _below_1000(n, final)

    parts: list[str] = []
    remainder = n
    for value, name in _SCALES:
        count, remainder = divmod(remainder, value)
        if count == 0:
            continue
        if name == "mille":
            # ``mille`` is invariable and drops its ``un``: 1000 is just "mille".
            parts.append("mille" if count == 1 else f"{_below_1000(count, False)} mille")
        else:
            plural = "s" if count > 1 else ""
            parts.append(f"{_below_1000(count, True)} {name}{plural}")
    if remainder:
        parts.append(_below_1000(remainder, final))
    return " ".join(parts)


def ordinal(n: int, feminine: bool = False) -> str:
    """Spell out an ordinal: ``1`` -> premier/première, ``21`` -> vingt et unième."""
    if n == 1:
        return "première" if feminine else "premier"
    # Built with final=False so that 80 yields "quatre-vingt" -> "quatre-vingtième".
    base = cardinal(n, final=False)
    if base.endswith("un"):
        base = f"{base[:-2]}unième"
    elif base.endswith("cinq"):
        base = f"{base[:-4]}cinquième"
    elif base.endswith("neuf"):
        base = f"{base[:-4]}neuvième"
    elif base.endswith("e"):
        base = f"{base[:-1]}ième"
    else:
        base = f"{base}ième"
    return base


# --------------------------------------------------------------------------
# Roman numerals
# --------------------------------------------------------------------------

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_STRICT = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def roman_to_int(s: str) -> Optional[int]:
    """Return the value of a well-formed Roman numeral, or None if malformed.

    Strict on purpose: loose parsing would happily read an initial or an acronym
    as a number.
    """
    s = (s or "").strip().upper()
    if not s or not _ROMAN_STRICT.match(s):
        return None
    total = 0
    previous = 0
    for char in reversed(s):
        value = _ROMAN_VALUES[char]
        total += value if value >= previous else -value
        previous = max(previous, value)
    return total or None


#: Words after which a Roman numeral is unambiguous. Expanding Roman numerals
#: everywhere would wreck initials and acronyms, so a trigger is required.
DEFAULT_ROMAN_TRIGGERS = (
    "chapitre", "chapitres", "partie", "parties", "livre", "livres", "tome", "tomes",
    "acte", "actes", "scène", "scènes", "section", "sections", "annexe", "annexes",
    "appendice", "volume", "volumes", "épisode", "épisodes", "titre", "article",
    "articles", "figure", "planche", "leçon", "chant",
)


# --------------------------------------------------------------------------
# Abbreviations
# --------------------------------------------------------------------------

# (pattern, replacement), applied in order. Entries that could collide with an
# ordinary word require a following capitalised token.
_ABBREVIATIONS: tuple[tuple[str, str], ...] = (
    # These end in a period that may also be the end of the sentence, so the
    # period is matched by lookahead and left in place — consuming it would
    # silently merge two sentences and destroy a pause.
    (r"\bav\.\s*J\.-?\s*C(?=\.)", "avant Jésus-Christ"),
    (r"\bapr\.\s*J\.-?\s*C(?=\.)", "après Jésus-Christ"),
    (r"\betc(?=\.)", "et cetera"),
    (r"\bc\.-à-d\.", "c'est-à-dire"),
    (r"\bp\.\s*ex\.", "par exemple"),
    (r"\bMM\.(?=\s)", "Messieurs"),
    (r"\bM\.(?=\s+[A-ZÀ-Þ])", "Monsieur"),
    (r"\bMmes\.?(?!\w)", "Mesdames"),
    (r"\bMme\.?(?!\w)", "Madame"),
    (r"\bMlles\.?(?!\w)", "Mesdemoiselles"),
    (r"\bMlle\.?(?!\w)", "Mademoiselle"),
    (r"\bDr\.?(?=\s+[A-ZÀ-Þ])", "Docteur"),
    (r"\bPr\.?(?=\s+[A-ZÀ-Þ])", "Professeur"),
    (r"\bMe\.?(?=\s+[A-ZÀ-Þ])", "Maître"),
    (r"\bStes\.?(?=\s+[A-ZÀ-Þ])", "Saintes"),
    (r"\bSts\.?(?=\s+[A-ZÀ-Þ])", "Saints"),
    (r"\bSte\.?(?=\s+[A-ZÀ-Þ])", "Sainte"),
    (r"\bSt\.?(?=\s+[A-ZÀ-Þ])", "Saint"),
    (r"\bchap\.", "chapitre"),
    (r"\bvol\.", "volume"),
    (r"\béd\.", "édition"),
    (r"\benv\.(?=\s)", "environ"),
    (r"\bcf\.", "voir"),
    (r"\bart\.(?=\s*\d)", "article"),
    (r"\bpp\.(?=\s*\d)", "pages"),
    (r"\bp\.(?=\s*\d)", "page"),
    (r"\bn[°º]\s*(?=\d)", "numéro "),
    (r"[°º](?=\s|$)", " degrés"),
    # « 20 h » est déjà lu par la règle des heures, « 5 min » ne l'était par
    # personne : 106 fois dans les vingt et un livres, dit « min ». Le singulier
    # s'écrit ici en toutes lettres — « 1 minute » serait rendu « un minute »
    # par la passe des nombres, qui ne connaît pas le genre du mot qui suit.
    # (Le défaut subsiste au-delà de un : « 51 min », une fois dans le
    # catalogue, se dit « cinquante et un minutes ».)
    (r"\b1\s*min\b(?!ute)", "une minute"),
    (r"(?<=\d)\s*min\b(?!ute)", " minutes"),
)


# --------------------------------------------------------------------------
# Compiled patterns for the rewriting passes
# --------------------------------------------------------------------------

# Thousands may be grouped with a plain, non-breaking or narrow no-break space.
# The pattern is anchored on groups of exactly three digits: a looser class such
# as ``\d[\d ]*`` would swallow the space *after* a number, turning "12 et" into
# a match on "12 " and gluing the next word to the spelled-out number.
_SEP = "    "
_NUM = rf"\d{{1,3}}(?:[{_SEP}]\d{{3}})+|\d+"

# A single-letter Roman ordinal is restricted to I, V and X. L, C, D and M would
# otherwise turn the extremely common "Le", "Ce", "De" and "Me" into ordinals —
# "Le manuscrit" read aloud as "cinquantième manuscrit".
_RE_ROMAN_ORDINAL = re.compile(r"\b([IVX]|[IVXLCDM]{2,15})(?:e|è?me|ᵉ)\b")
# The minutes carry their own separator: with the space outside the optional
# group, "9h du matin" matched "9h " and came back as "neuf heuresdu matin".
# The trailing guard is what keeps "35ha" and "9h305" out — a bare `\b` would
# let the first of them through as "trente-cinq heures a".
_RE_TIME = re.compile(r"\b(\d{1,2})\s*[hH](?:\s*(\d{2}))?(?!\w)")
_RE_CURRENCY = re.compile(rf"({_NUM})(?:,(\d{{1,2}}))?\s*([€$£])")
_RE_CURRENCY_PREFIX = re.compile(rf"([€$£])\s*({_NUM})(?:,(\d{{1,2}}))?")
_RE_PERCENT = re.compile(rf"({_NUM}(?:,\d+)?)\s*%")
_RE_ORDINAL_MARK = re.compile(r"\b(\d+)(ers|er|res|re|èmes|ème|es|e)\b")
_RE_DECIMAL = re.compile(rf"\b({_NUM}),(\d+)\b")
_RE_INTEGER = re.compile(rf"\b(?:{_NUM})\b")
_RE_GROUPED = re.compile(rf"[{_SEP}]")

_CURRENCY_NAMES = {
    "€": ("euro", "euros", "centime", "centimes"),
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("livre", "livres", "penny", "pennies"),
}

#: Ordinal suffixes that mark a feminine ordinal (``1re``, ``1res``).
_FEMININE_SUFFIXES = {"re", "res"}


def _digits(raw: str) -> int:
    """Parse an integer that may carry French thousands separators."""
    return int(_RE_GROUPED.sub("", raw))


def _spell_decimal(whole: str, frac: str) -> str:
    return f"{cardinal(_digits(whole))} virgule {' '.join(_UNITS[int(d)] for d in frac)}"


# --------------------------------------------------------------------------
# Individual passes
# --------------------------------------------------------------------------


#: Le blanc d'un formulaire : un trait à remplir au stylo.
_BLANK = re.compile(r"_{2,}")

#: Ce qu'on efface après le blanc s'arrête à la ponctuation forte : au-delà,
#: ce n'est plus l'unité qui accompagnait le trait, c'est la phrase suivante.
_BLANK_TAIL = re.compile(r"_{2,}[^(.!?;\n]*")

#: Un blanc long est un trait de formulaire ; une queue longue est de la prose
#: qui se trouvait derrière. Sur les trois livres concernés, la plus longue
#: queue légitime fait 28 caractères (« ___ fois par jour en moyenne »).
_MAX_TAIL = 40

#: Séparateurs qu'un champ vidé laisse pendre à ses extrémités.
_ORPHAN_EDGE = re.compile(r"^[\s:;,+/–—-]+|[\s:;,+/–—-]+$")


def _form_fields(line: str) -> list[str]:
    """Découper une ligne de formulaire sur ses barres obliques de premier rang.

    Une parenthèse en protège une : « Temps réel mesuré (Screen Time /
    Bien-être numérique) » est un seul champ, pas deux.
    """
    fields: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in line:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "/" and depth == 0:
            fields.append("".join(current))
            current = []
        else:
            current.append(ch)
    fields.append("".join(current))
    return fields


def _strip_form_blanks(text: str) -> str:
    """Rendre lisible une ligne à remplir, ou la faire taire.

    Effacer le seul trait ne suffisait pas, et c'est ce qui a été livré :
    « Jour 5: ___ minutes (objectif: 10 min) / Ressenti: ___ » devenait
    « Jour 5 : minutes (objectif : 10 min) / Ressenti : », que le moteur a
    narré tel quel — l'audit l'a relu en « Jour 5, minute objectif, 10 mines,
    essenci », sept fois de suite, dans un livre déjà livré.

    Ce qui reste après le trait n'a de sens qu'avec lui : « ___ heures ___
    minutes » énonce des unités sans grandeur, « ___ h / ___ h / ___ h » trois
    fois rien. On efface donc le trait **et sa queue**, jusqu'à la parenthèse
    ou la barre oblique suivante — la parenthèse porte souvent la seule vraie
    information de la ligne (« objectif : 10 min ») et doit survivre.

    L'intitulé, lui, reste toujours : c'est une consigne que l'auditeur peut
    suivre. « Application la plus consultée : ______ » devient « Application la
    plus consultée. » Une ligne qui n'était qu'un trait disparaît, faute
    d'avoir jamais rien dit.

    Trois livres de la file portaient ces lignes — 68, 11 et 8 — et aucun
    n'aurait dû les faire entendre.
    """
    if "__" not in text:
        return text

    def deblank(field: str) -> str:
        def cut(m: re.Match) -> str:
            # Une queue trop longue n'est pas une unité, c'est une phrase :
            # on se contente alors d'ôter le trait, sans l'emporter avec lui.
            if len(m.group(0)) <= _MAX_TAIL:
                return ""
            return _BLANK.sub(" ", m.group(0))

        return _BLANK_TAIL.sub(cut, field)

    def rewrite(line: str) -> str:
        if not _BLANK.search(line):
            return line
        kept = []
        for field in _form_fields(line):
            field = _ORPHAN_EDGE.sub("", deblank(field)).strip()
            # « Jour 5 : (objectif : 10 min) » — le deux-points a perdu sa
            # valeur, la parenthèse la porte désormais seule.
            field = re.sub(r"\s*:\s*(?=\()", " ", field)
            if re.search(r"[^\W\d_]|\d", field):
                kept.append(field)
        if not kept:
            return ""
        rebuilt = ". ".join(kept)
        return rebuilt if rebuilt[-1] in ".!?…:;" else rebuilt + "."

    return "\n".join(rewrite(line) for line in text.split("\n"))


def _clean_symbols(text: str) -> str:
    """Traduire en mots les signes qu'un manuscrit garde et qu'on ne dit pas.

    Un manuscrit n'est pas que de la prose : il porte des restes de mise en
    page, des cases à cocher, des flèches, de l'écriture inclusive, des
    commandes de traitement de texte. Le moteur les lit — ou pire, il en lit
    une partie : ``
ewpage`` perdait sa barre oblique au nettoyage markdown et
    devenait « ewpage », prononcé tel quel au milieu d'un chapitre.

    Chaque règle vient d'un relevé sur les vingt et un livres de la file, pas
    d'une liste imaginée : 571 lignes à remplir, 163 appels de note, 107 points
    médians, 55 commandes LaTeX, 47 degrés, 26 esperluettes.
    """
    # D'abord les commandes de traitement de texte : le nettoyage markdown
    # mangerait la barre oblique et laisserait « ewpage », lu tel quel.
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\{[^}]*\})*", " ", text)

    # Unités collées à un nombre. L'ordre compte : sans la règle Celsius avant
    # la règle générale, « 18,5 °C » deviendrait « 18,5 degrésC ».
    text = re.sub(r"\s*°\s*C(?![a-zà-ÿ])", " degrés Celsius", text)
    text = re.sub(r"\s*°\s*F(?![a-zà-ÿ])", " degrés Fahrenheit", text)
    text = re.sub(r"(\d)\s*°", r"\1 degrés", text)

    # Un intervalle chiffré. Le trait d'union se dit « à », et ne pas le dire
    # ne laisse pas un silence : il colle les deux nombres l'un à l'autre et la
    # passe des nombres les fond en un seul. « La pandémie de 2020-2022 » se
    # narrait « deux mille vingt-deux mille vingt-deux ». 726 intervalles dans
    # les vingt et un livres de la file.
    #
    # Les bornes acceptent un horaire (« 14h-15h30 »), et les deux gardes
    # interdisent qu'une chaîne plus longue soit prise pour un intervalle :
    # « 4-7-8 » est une respiration, pas « quatre à sept à huit », et
    # « 5-4-3-2-1 » un exercice d'ancrage.
    text = re.sub(
        r"(?<![\d\-–])(\d{1,4}(?:h\d{0,2})?)\s*[-–]\s*(\d{1,4}(?:h\d{0,2})?)(?![\d\-–])",
        r"\1 à \2",
        text,
    )

    # Signes mathématiques au fil d'une phrase.
    text = re.sub(r"\s*×\s*", " fois ", text)
    text = re.sub(r"(?<=[\w)])\s*=\s*(?=[\w(])", " égale ", text)
    text = re.sub(r"\s*&\s*", " et ", text)

    # Flèches et chemins d'interface : « Réglages > Temps d'écran ».
    text = re.sub(r"\s*[→⟶➜]\s*", " puis ", text)
    text = re.sub(r"(?<=[a-zà-ÿ0-9])\s*>\s*(?=[A-ZÀ-Þa-zà-ÿ])", " puis ", text)

    # Écriture inclusive : « conjoint·e » se dit « conjoint ou conjointe ». Il
    # faut le mot entier pour reconstruire la forme accordée ; la terminaison
    # seule ne suffit pas, et « conjoint ou e » ne veut rien dire.
    text = re.sub(
        r"([a-zà-ÿ]{2,})[·‧∙]([a-zà-ÿ]{1,3})(?![a-zà-ÿ])",
        lambda m: f"{m.group(1)} ou {m.group(1)}{m.group(2)}",
        text,
    )

    # Restes de formulaire : lignes à remplir, cases à cocher, appels de note.
    text = _strip_form_blanks(text)
    text = re.sub(r"[☐☑✓✗▢]", " ", text)
    text = re.sub(r"(?<=[a-zà-ÿ])\*(?=[\s,.;:)])", "", text)

    return re.sub(r"[  ]{2,}", " ", text)


#: Lettres modificatives en exposant, telles qu'un traitement de texte les
#: produit pour « 5ᵉ » ou « 1ʳᵉ ». Elles ressemblent à leurs équivalents
#: ordinaires et n'en sont pas : la règle des ordinaux ne les voit pas, « 5ᵉ »
#: traverse la normalisation intact, et le normaliseur interne du moteur meurt
#: dessus — assert len(input) > 0, après quarante et une minutes de narration.
_EXPOSANTS = {
    "ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e", "ᶠ": "f", "ᵍ": "g",
    "ʰ": "h", "ⁱ": "i", "ʲ": "j", "ᵏ": "k", "ˡ": "l", "ᵐ": "m", "ⁿ": "n",
    "ᵒ": "o", "ᵖ": "p", "ʳ": "r", "ˢ": "s", "ᵗ": "t", "ᵘ": "u", "ᵛ": "v",
    "ʷ": "w", "ˣ": "x", "ʸ": "y", "ᶻ": "z",
}
_EXPOSANTS_RE = re.compile("|".join(map(re.escape, _EXPOSANTS)))


def _clean_typography(text: str) -> str:
    """Normalise Unicode punctuation to forms the engine handles predictably."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    # Avant tout le reste : « 5ᵉ » doit redevenir « 5e » pour que la règle des
    # ordinaux le lise, sinon il arrive intact jusqu'au moteur.
    text = _EXPOSANTS_RE.sub(lambda m: _EXPOSANTS[m.group(0)], text)
    text = re.sub(r"[   ]", " ", text)
    text = re.sub(r"\.{3,}", "…", text)
    return text


def _strip_markdown(text: str) -> str:
    """Remove markup that would otherwise be read aloud as punctuation noise."""
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)            # ATX headings
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)                  # block quotes
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                  # bold
    text = re.sub(r"(?<!\w)[*_](\S(?:[^*_\n]*\S)?)[*_](?!\w)", r"\1", text)
    text = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", text)            # code spans
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)        # links / images
    return text


@dataclass(frozen=True)
class Pronunciation:
    """How a written form should be said, and when that applies.

    Without ``after`` and ``before`` this is the plain substitution the lexicon
    has always done. With them it becomes the only thing that can handle a
    French homograph: *il est* and *à l'est* are the same three letters and two
    different words, so a rule that fires on the word alone must either break
    one of them or do nothing.
    """

    spoken: str
    #: Regex that must match immediately before the word — "à l'|dans l'".
    after: str = ""
    #: Regex that must match immediately after it.
    before: str = ""

    @classmethod
    def parse(cls, value) -> Optional["Pronunciation"]:
        """Read either form from the lexicon file, or None if it makes no sense.

        A malformed entry is dropped rather than raised on: an optional override
        file with a typo in it must not take a nine-hour narration down.
        """
        if isinstance(value, str):
            return cls(value) if value else None
        if isinstance(value, Mapping):
            spoken = str(value.get("prononcer", "")).strip()
            if not spoken:
                return None
            return cls(
                spoken=spoken,
                after=str(value.get("après", value.get("apres", ""))),
                before=str(value.get("avant", "")),
            )
        return None


def _apply_lexicon(text: str, lexicon: Mapping[str, object]) -> str:
    """Apply user pronunciation overrides, longest key first so that multi-word
    entries win over their own prefixes."""
    for source in sorted(lexicon, key=len, reverse=True):
        entry = Pronunciation.parse(lexicon[source])
        if entry is None:
            continue

        word = rf"(?<!\w){re.escape(source)}(?!\w)"
        if entry.after:
            # The preceding context is captured and put back rather than looked
            # behind: Python's lookbehind must be fixed width, and "à l'|dans l'"
            # is exactly the kind of alternation that is not.
            pattern = re.compile(rf"(?P<before>{entry.after})(?P<gap>\s*){word}", re.IGNORECASE)
            replacement = "\\g<before>\\g<gap>" + entry.spoken.replace("\\", "\\\\")
        else:
            pattern = re.compile(word + (rf"(?=\s*(?:{entry.before}))" if entry.before else ""),
                                 re.IGNORECASE)
            replacement = entry.spoken.replace("\\", "\\\\")

        if entry.after:
            text = pattern.sub(replacement, text)
        else:
            # La correspondance ignore la casse, donc « Ces » en tête de phrase
            # tomberait sur « cés » en minuscule. Rendre la majuscule quand le
            # mot d'origine en portait une : une phrase qui commence en
            # minuscule est une anomalie gratuite, et la règle vaut pour toutes
            # les entrées, pas seulement celle qui l'a révélée.
            def _garder_la_casse(m: re.Match) -> str:
                trouve = m.group(0)
                if trouve[:1].isupper() and entry.spoken[:1].islower():
                    return entry.spoken[:1].upper() + entry.spoken[1:]
                return entry.spoken

            text = pattern.sub(_garder_la_casse, text)
    return text


def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)
    return text


def _expand_roman(text: str, triggers: Iterable[str]) -> str:
    def _ordinal_sub(match: re.Match) -> str:
        value = roman_to_int(match.group(1))
        return ordinal(value) if value is not None else match.group(0)

    # "XXe siècle" -> "vingtième siècle". Runs first: it is the most specific form.
    text = _RE_ROMAN_ORDINAL.sub(_ordinal_sub, text)

    trigger_group = "|".join(sorted((re.escape(t) for t in triggers), key=len, reverse=True))
    if trigger_group:
        def _after_trigger(match: re.Match) -> str:
            value = roman_to_int(match.group(2))
            return f"{match.group(1)}{cardinal(value)}" if value is not None else match.group(0)

        # The trigger is matched case-insensitively via an inline group, but the
        # numeral itself stays case-sensitive on purpose: with a global
        # IGNORECASE the perfectly ordinary "chapitre dix" parses as the Roman
        # numeral DIX and is read back as "cinq cent neuf".
        text = re.sub(
            rf"\b((?:(?i:{trigger_group}))\s+)([IVXLCDM]{{1,15}})\b",
            _after_trigger,
            text,
        )

    # A line containing nothing but a Roman numeral is a chapter heading.
    def _heading(match: re.Match) -> str:
        value = roman_to_int(match.group(1))
        return cardinal(value) if value is not None else match.group(0)

    return re.sub(r"(?m)^[ \t]*([IVXLCDM]{1,15})[ \t]*\.?[ \t]*$", _heading, text)


def _expand_times(text: str) -> str:
    def _sub(match: re.Match) -> str:
        hours = int(match.group(1))
        if hours > 23:
            return match.group(0)
        minutes = match.group(2)
        hour_word = "une" if hours == 1 else cardinal(hours)
        unit = "heure" if hours in (0, 1) else "heures"
        if not minutes or int(minutes) == 0:
            return f"{hour_word} {unit}"
        return f"{hour_word} {unit} {cardinal(int(minutes))}"

    return _RE_TIME.sub(_sub, text)


def _expand_currency(text: str) -> str:
    def _format(amount: str, cents: Optional[str], symbol: str) -> str:
        singular, plural, cent_singular, cent_plural = _CURRENCY_NAMES[symbol]
        value = _digits(amount)
        words = f"{cardinal(value)} {singular if value <= 1 else plural}"
        if cents:
            cent_value = int(cents.ljust(2, "0"))
            if cent_value:
                unit = cent_singular if cent_value == 1 else cent_plural
                words += f" {cardinal(cent_value)} {unit}"
        return words

    text = _RE_CURRENCY.sub(lambda m: _format(m.group(1), m.group(2), m.group(3)), text)
    return _RE_CURRENCY_PREFIX.sub(lambda m: _format(m.group(2), m.group(3), m.group(1)), text)


def _expand_percent(text: str) -> str:
    def _sub(match: re.Match) -> str:
        raw = match.group(1)
        if "," in raw:
            whole, frac = raw.split(",", 1)
            return f"{_spell_decimal(whole, frac)} pour cent"
        return f"{cardinal(_digits(raw))} pour cent"

    return _RE_PERCENT.sub(_sub, text)


def _expand_ordinal_marks(text: str) -> str:
    def _sub(match: re.Match) -> str:
        suffix = match.group(2)
        word = ordinal(_digits(match.group(1)), feminine=suffix in _FEMININE_SUFFIXES)
        return f"{word}s" if suffix.endswith("s") and not word.endswith("s") else word

    return _RE_ORDINAL_MARK.sub(_sub, text)


def _expand_numbers(text: str) -> str:
    text = _RE_DECIMAL.sub(lambda m: _spell_decimal(m.group(1), m.group(2)), text)
    return _RE_INTEGER.sub(lambda m: cardinal(_digits(m.group(0))), text)


def _clean_dialogue(text: str, strip_quotes: bool) -> str:
    """Dialogue dashes and guillemets carry no sound of their own — the pause
    around them is what a listener actually hears."""
    text = re.sub(r"(?m)^[ \t]*[—–-]{1,2}[ \t]+", "", text)
    text = re.sub(r"\s*[—–]\s*", ", ", text)
    if strip_quotes:
        text = text.replace("«", "").replace("»", "").replace('"', "")
    return text


_RE_PARENTHETICAL = re.compile(r"\(([^()]{0,400})\)")


def _flatten_parentheses(text: str) -> str:
    """Turn a parenthetical aside into the apposition a narrator would speak.

    Parentheses make no sound of their own, but the model treats an opening one
    as a bracket it must close, and on a long enumeration it gives up partway.
    Across the first twenty books, 86% of the truncated segments held a
    parenthesis against 29% of segments overall — and 30% of segments of the
    same length, so it is the construction that costs, not the length. "Les
    approches alternatives (keynésienne, institutionnaliste, marxiste,
    écologique)" came out as 2.1 seconds of audio for 260 characters of text.

    Commas read aloud the same way. Short asides used to be left alone — a
    date, a source — on the grounds that they had never truncated. Measured
    again on a book narrated *with* this pass in place, that exemption is what
    was left of the defect: of sixty-nine segments transcribed back, the eight
    that stopped early all carried a parenthesis, and all eight carried one the
    exemption had spared — « (REM) », « (N3) », « (chapitre dix) »,
    « (urgences pédiatriques) ». Not one of the fifty-one segments without a
    parenthesis stopped early. Length was never the trigger; the bracket was.
    """

    def replace(match: "re.Match[str]") -> str:
        inner = match.group(1).strip()
        if not inner:
            return " "
        return f", {inner}, "

    text = _RE_PARENTHETICAL.sub(replace, text)
    # The apposition's closing comma lands on whatever punctuation ended the
    # host sentence: "…marxiste, ." Nothing in French wants a comma there.
    text = re.sub(r",\s*([.;:!?…])", r"\1", text)
    # An aside that ended the line has nothing to lean its comma against —
    # "le rapport, deux mille huit," — and a trailing comma is exactly the
    # unclosed construction this pass exists to remove.
    return re.sub(r"(?m),[ \t]*$", "", text)


def _tidy_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([,.;:!?…])", r"\1", text)
    text = re.sub(r",(\s*,)+", ",", text)
    text = re.sub(r"(?m)^[ \t]+|[ \t]+$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def normalize_french(
    text: str,
    *,
    lexicon: Optional[Mapping[str, str]] = None,
    expand_roman: bool = True,
    roman_triggers: Iterable[str] = DEFAULT_ROMAN_TRIGGERS,
    strip_markdown: bool = True,
    strip_quotes: bool = True,
) -> str:
    """Rewrite French prose into the words a narrator would speak.

    The passes run in a fixed order because several of them compete for the same
    digits: times, currency, percentages and ordinal marks each claim their
    pattern before the generic number rule sees it.
    """
    if not text or not text.strip():
        return ""

    text = _clean_typography(text)
    text = _clean_symbols(text)
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
    text = _expand_numbers(text)
    # After the markdown pass, so that a link's "(url)" is already gone.
    text = _flatten_parentheses(text)
    text = _clean_dialogue(text, strip_quotes)
    return _tidy_whitespace(text)


def load_lexicon(path: str | Path) -> Dict[str, str]:
    """Load a user pronunciation lexicon (``{"écrit": "prononcé"}``).

    A missing or malformed file is never fatal — a whole narration run must not
    die because an optional override file has a typo — it yields an empty
    lexicon instead.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Keys starting with "_" are comments — JSON has no other way to carry one.
    # Values are kept as they were written — a string for a plain substitution,
    # an object for one that only applies in context — and interpreted later by
    # Pronunciation.parse, which drops whatever it cannot make sense of.
    return {
        str(k): v
        for k, v in data.items()
        if str(k).strip() and not str(k).startswith("_") and Pronunciation.parse(v)
    }
