"""Turn a Markdown manuscript into text a narrator can read aloud.

A manuscript is written to be *seen*. Its first pages carry an ISBN, a
copyright notice, a table of contents and a web address — all of which a
narrator would never read out, and all of which a TTS engine reads out
happily. Its body carries asterisks for emphasis and hashes for headings,
which are silent on a page and absurd in an ear.

Three traps, in order of how much damage they do:

1. **``---`` means two different things.** In the manuscript it is a
   horizontal rule, sprinkled through the front matter. To ``narrate_book.py``
   it is *the* chapter separator. Converting naively cuts the book at the
   copyright page. So every rule is removed, and ``---`` is re-emitted only at
   the boundaries this script decides.
2. **Front matter is not narration.** Everything before the first content
   heading goes, except the blocks worth keeping (a disclaimer, a dedication),
   which are named rather than guessed.
3. **Tables cannot be read.** A Markdown table spoken aloud is a stream of
   pipes. They are dropped, and counted.

Nothing is removed silently: ``--report`` prints every dropped block, because
text taken out of a book has to be reported back to whoever asked for it.

    python scripts/prepare_manuscript.py manuscrit_complet.md -o livre.txt --report
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from narration import adresse_audio  # noqa: E402  (après l'ajout au chemin)

# A heading that opens something a narrator actually reads. Anything before the
# first of these is front matter: title page, copyright, ISBN, contents.
CONTENT_HEADING = re.compile(
    r"^\s*(introduction|chapitre|partie|prologue|pr[ée]face|avant[- ]propos"
    r"|conclusion|[ée]pilogue|annexe|postface)\b",
    re.IGNORECASE,
)

# Front-matter blocks worth keeping anyway. A disclaimer carries legal weight
# and a dedication is read in most audiobooks; a copyright page is neither.
KEEP_IN_FRONT = re.compile(r"^\s*(avertissement|d[ée]dicace|note de l['’]auteur)\b", re.IGNORECASE)

# Blocks to drop even when they sit in the body.
DROP_ALWAYS = re.compile(r"^\s*(table des mati[èe]res|sommaire|remerciements?|bibliographie|index)\b", re.IGNORECASE)

# A part divider carries no prose of its own — it must not become a chapter of
# two words, it belongs to the chapter that follows.
PART_HEADING = re.compile(r"^\s*partie\b", re.IGNORECASE)

#: Un en-tête qui ne dit que son rang : « Chapitre 4 », « Introduction ». Le
#: manuscrit met la vraie formule juste en dessous, en niveau 2 — et comme le
#: marqueur du lecteur audio est la première ligne du chapitre, le sommaire
#: n'affichait que « Chapitre 4 ».
BARE_HEADING = re.compile(
    r"^\s*(chapitre\s+[0-9IVXLC]+|introduction|conclusion|[ée]pilogue|prologue"
    r"|avant[- ]propos|pr[ée]face|annexes?)\s*$",
    re.IGNORECASE,
)


def join_subtitle(chapter: str) -> str:
    """« Chapitre 1 » + « Le grand malentendu » = un titre qui dit quelque chose.

    Utile deux fois : le sommaire devient lisible sur un téléphone, et
    l'annonce sonne juste, parce qu'un narrateur lit le titre entier plutôt que
    son numéro seul.
    """
    blocs = chapter.split("\n\n")
    if len(blocs) < 2 or not BARE_HEADING.match(blocs[0].strip()):
        return chapter
    suite = blocs[1].strip()
    # Un sous-titre est court et ne se termine pas ; un paragraphe fait les deux.
    if not suite or len(suite) > 90 or suite.endswith((".", "!", "?", "…")):
        return chapter
    # « La fin du culte du charisme — Pourquoi l'extraversion a cessé… » porte
    # déjà son propre tiret : recoller tel quel donnerait deux tirets et un
    # marqueur tronqué à 80 caractères. Le titre est ce qui précède.
    for coupure in (" — ", " – ", " : "):
        if coupure in suite:
            suite = suite.split(coupure, 1)[0].strip()
            break
    return "\n\n".join([f"{blocs[0].strip()} — {suite}"] + blocs[2:])


@dataclass
class Block:
    """A heading and the prose under it, down to the next heading of any level."""

    level: int
    title: str
    lines: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


#: A bracketed direction asking the narrator to stop. It is not a word, it is a
#: silence — so it becomes one, rather than being read out as "PAUSE".
PAUSE_MARKER = re.compile(r"\[\s*pause\s*\]", re.IGNORECASE)

#: Stage directions in a transcribed testimony. They tell a reader what
#: happened in the room; spoken aloud they say that the narrator laughed.
STAGE_DIRECTION = re.compile(
    r"\[\s*(rire|rires|hésitation|h[ée]sitations|silence(?:\s+prolong[ée])?|soupir|soupirs"
    r"|pleurs(?:\s+contenus)?|larmes|blanc|sanglots?|se l[èe]ve[^\]]*|s'?arr[êe]te[^\]]*)\s*\]",
    re.IGNORECASE,
)


def unbracket(text: str) -> tuple[str, list[str]]:
    """Deal with the square brackets a manuscript carries, by what they mean.

    A corpus of twenty books held 993 of them in 24 forms, and they are not one
    thing. ``[PAUSE]``, 969 times over, is an instruction to stop talking.
    ``[rire]`` is a stage direction in a transcribed testimony. But
    ``[nom du département]`` and ``[ton mari / ta femme]`` are the sentence
    itself — a blank the reader fills — and deleting them leaves a hole where
    the meaning was.

    So: a pause becomes a paragraph break, a stage direction goes, and anything
    else keeps its words and loses only its brackets. Brackets are never
    spoken; what is inside them sometimes is.
    """
    removed: list[str] = []

    def note(kind: str, m: re.Match) -> str:
        removed.append(f"{kind} : {m.group(0)}")
        return ""

    text = PAUSE_MARKER.sub(lambda m: note("pause", m) or "\n\n", text)
    text = STAGE_DIRECTION.sub(lambda m: note("didascalie", m), text)

    def keep_inside(m: re.Match) -> str:
        inner = m.group(1).strip()
        removed.append(f"crochets retirés : {m.group(0)}")
        return inner

    text = re.sub(r"\[([^\]\n]{1,80})\]", keep_inside, text)
    return text, removed


#: Mots dont la forme sans accent n'est pas un mot français : la restaurer ne
#: peut donc pas créer d'ambiguïté. Volontairement court et vérifié à la main —
#: « cote », « tache », « sur », « mure », « pecheur » ont tous deux lectures
#: légitimes et n'ont rien à faire ici.
#:
#: Pourquoi c'est nécessaire : le manuscrit écrit « degres », le moteur lit
#: « degre », et l'auditeur entend une faute que personne n'a commise à la
#: synthèse. Mesuré sur vingt-deux fichiers sur vingt-quatre.
ACCENTS_PERDUS = {
    "degre": "degré", "degres": "degrés", "maniere": "manière",
    "difference": "différence", "differences": "différences",
    "plongee": "plongée", "societe": "société", "societes": "sociétés",
    "desir": "désir", "desirs": "désirs", "dedicace": "dédicace",
    "annee": "année", "annees": "années", "realite": "réalité",
    "realites": "réalités", "probleme": "problème", "problemes": "problèmes",
    "systeme": "système", "systemes": "systèmes", "modele": "modèle",
    "modeles": "modèles", "premiere": "première", "premieres": "premières",
    "derniere": "dernière", "dernieres": "dernières", "matiere": "matière",
    "matieres": "matières", "experience": "expérience",
    "experiences": "expériences", "etre": "être", "etait": "était",
    "etaient": "étaient", "meme": "même", "memes": "mêmes", "tres": "très",
    "apres": "après", "present": "présent", "presente": "présente",
    "reponse": "réponse", "reponses": "réponses", "resultat": "résultat",
    "resultats": "résultats", "periode": "période", "periodes": "périodes",
    "sante": "santé", "verite": "vérité", "verites": "vérités",
}
_ACCENTS_RE = re.compile(
    r"(?<![\w'’])(" + "|".join(sorted(ACCENTS_PERDUS, key=len, reverse=True)) + r")(?![\w'’])",
    re.IGNORECASE,
)


def restore_accents(text: str) -> tuple[str, list[str]]:
    """Rendre aux mots l'accent que le manuscrit leur a pris.

    Le moteur lit ce qui est écrit : « degres » se dit « degre », et l'auditeur
    entend une faute de prononciation là où il y a une faute d'orthographe. La
    corriger dans le texte est la seule réparation juste — un lexique
    remplacerait un mot mal écrit par une graphie inventée, ce qui empile deux
    approximations au lieu d'en retirer une.
    """
    trouves: list[str] = []

    def remplacer(m: re.Match) -> str:
        mot = m.group(0)
        juste = ACCENTS_PERDUS[mot.lower()]
        trouves.append(f"accent rendu : {mot} → {juste}")
        return juste[:1].upper() + juste[1:] if mot[:1].isupper() else juste

    return _ACCENTS_RE.sub(remplacer, text), trouves


def strip_inline(text: str) -> str:
    """Remove the marks that are silent on a page and spoken by an engine."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # images: nothing to say
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)      # links: keep the words
    text = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", text)        # inline code
    text = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", text)
    text = re.sub(r"^\s{0,3}>\s?", "", text)                  # blockquote marker
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text)              # bullet
    text = re.sub(r"^\s{0,3}\d+[.)]\s+", "", text)            # numbered item
    return text.strip()


def parse(md: str) -> tuple[List[Block], List[str]]:
    """Split Markdown into heading-led blocks, dropping what cannot be spoken."""
    removed: List[str] = []
    blocks: List[Block] = [Block(0, "")]
    in_fence = False
    in_table = False

    for raw in md.splitlines():
        if raw.strip().startswith("```"):
            in_fence = not in_fence
            if in_fence:
                removed.append("bloc de code")
            continue
        if in_fence:
            continue

        # A table row, and the ---|--- rule under it, are unreadable aloud.
        if re.match(r"^\s*\|", raw) or re.match(r"^\s*\|?[\s:-]*\|[\s:|-]*$", raw) and "|" in raw:
            if not in_table:
                removed.append("tableau")
                in_table = True
            continue
        in_table = False

        heading = re.match(r"^(#{1,6})\s+(.*)$", raw)
        if heading:
            blocks.append(Block(len(heading.group(1)), strip_inline(heading.group(2))))
            continue

        # Horizontal rules only ever meant "new visual section"; the chapter
        # boundaries this script emits are decided from the headings instead.
        if re.match(r"^\s*([-*_])\1{2,}\s*$", raw):
            continue

        blocks[-1].lines.append(strip_inline(raw))

    return [b for b in blocks if b.title or b.text], removed


def to_chapters(blocks: List[Block]) -> tuple[List[str], List[str]]:
    """Group blocks into chapters, and say what was left out."""
    removed: List[str] = []

    start = next((i for i, b in enumerate(blocks) if CONTENT_HEADING.match(b.title)), None)
    if start is None:
        # No recognisable structure — narrate the whole thing as one chapter
        # rather than refuse. Better a long chapter than no book.
        body = "\n\n".join(b.text for b in blocks if b.text)
        return ([body] if body else []), removed

    kept_front = []
    for b in blocks[:start]:
        if KEEP_IN_FRONT.match(b.title):
            kept_front.append(b)
        elif b.title or b.text:
            removed.append(f"liminaire : {b.title or b.text[:40]}…")

    chapters: List[str] = []
    current: List[str] = []

    for b in kept_front:
        chapters.append("\n\n".join(x for x in (b.title, b.text) if x))

    for b in blocks[start:]:
        if DROP_ALWAYS.match(b.title):
            removed.append(f"section : {b.title}")
            continue
        # A new chapter opens on a level-1 heading — except a part divider,
        # which introduces the chapter after it rather than standing alone.
        if b.level == 1 and not PART_HEADING.match(b.title):
            if current:
                chapters.append("\n\n".join(current).strip())
            current = []
        piece = "\n\n".join(x for x in (b.title, b.text) if x)
        if piece:
            current.append(piece)

    if current:
        chapters.append("\n\n".join(current).strip())

    chapters = [join_subtitle(c) for c in chapters]
    return [c for c in chapters if c.strip()], removed


def main() -> int:
    # The report names chapters, so it carries whatever the book does — and a
    # Windows console defaults to cp1252, where an em dash or an arrow raises
    # rather than prints. Never let the summary kill a conversion that worked.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manuscript", help="fichier Markdown")
    ap.add_argument("-o", "--output", help="fichier .txt de sortie (défaut : à côté du manuscrit)")
    ap.add_argument("--report", action="store_true", help="détailler ce qui a été retiré")
    ap.add_argument("--no-adresse-audio", action="store_true",
                    help="ne pas adapter « lire ce livre » en « écouter ce livre audio »")
    ap.add_argument("--rapport-adresse", metavar="FICHIER",
                    help="écrire le détail de l'adaptation à l'écoute, et les "
                         "passages laissés à décider à la main")
    args = ap.parse_args()

    src = pathlib.Path(args.manuscript)
    if not src.is_file():
        print(f"introuvable : {src}", file=sys.stderr)
        return 1

    md = src.read_text(encoding="utf-8", errors="replace")
    # Avant tout découpage : un « [PAUSE] » devenu saut de paragraphe doit
    # pouvoir séparer deux paragraphes, ce que le parseur lira ensuite.
    md, removed_brackets = unbracket(md)
    md, removed_accents = restore_accents(md)
    blocks, removed_parse = parse(md)
    removed_parse = removed_parse + removed_brackets + removed_accents
    chapters, removed_struct = to_chapters(blocks)

    if not chapters:
        print("aucun texte narrable trouvé", file=sys.stderr)
        return 1

    # Le manuscrit s'adresse à un lecteur ; l'audio s'adresse à un auditeur.
    # L'adaptation vient en dernier, sur le texte déjà nettoyé : elle raisonne
    # sur des phrases, et les phrases n'existent qu'une fois les blocs de mise
    # en forme retirés.
    adaptations, signalements = [], []
    if not args.no_adresse_audio:
        adaptes = []
        # Le nom du dossier identifie le livre : un livre dont le sujet est la
        # différence entre lire et écouter échappe à la règle sur « lecteur ».
        slug = src.parent.name
        for c in chapters:
            neuf, ch, sig = adresse_audio.adapter(c, slug=slug)
            adaptes.append(neuf)
            adaptations.extend(ch)
            signalements.extend(sig)
        chapters = adaptes

    out = pathlib.Path(args.output) if args.output else src.with_suffix(".narration.txt")
    body = "\n\n---\n\n".join(chapters)
    # The separator must be unambiguous: it is the one thing narrate_book.py
    # keys on, so no stray rule may survive anywhere else in the file.
    assert body.count("\n---\n") == len(chapters) - 1, "séparateur ambigu"
    out.write_text(body + "\n", encoding="utf-8")

    chars = sum(len(c) for c in chapters)
    print(f"{len(chapters)} chapitre(s) · {chars} caractères · ~{chars/15/60:.0f} min → {out.name}")
    for i, c in enumerate(chapters, 1):
        print(f"  {i:>3}. {c.splitlines()[0][:62]:<62} {len(c):>7} car.")

    if adaptations or signalements:
        print(f"\nAdressé à l'auditeur : {len(adaptations)} adaptation(s), "
              f"{len(signalements)} passage(s) à décider à la main")
        for c in adaptations:
            print(f"  [{c.regle}] {c.avant} → {c.apres}")
    if args.rapport_adresse:
        pathlib.Path(args.rapport_adresse).write_text(
            adresse_audio.rapport_texte(adaptations, signalements), encoding="utf-8")

    counts: dict[str, int] = {}
    for r in removed_parse:
        # Grouper par nature : 969 lignes « pause : [PAUSE] » n'apprennent rien
        # de plus qu'une seule ligne disant 969.
        cle = r.split(" : ")[0] if " : " in r else r
        counts[cle] = counts.get(cle, 0) + 1
    if counts or removed_struct:
        print("\nRetiré :")
        for k, n in sorted(counts.items()):
            print(f"  {n:>3} × {k}")
        if args.report:
            for r in removed_struct:
                print(f"      {r}")
        else:
            print(f"  {len(removed_struct)} bloc(s) liminaire(s)/section(s) — --report pour le détail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
