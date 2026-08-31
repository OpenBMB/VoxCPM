#!/usr/bin/env python3
"""Parler à quelqu'un qui écoute, quand le manuscrit s'adresse à quelqu'un qui lit.

Un manuscrit dit « comment lire ce livre », « vous lisez ces pages », « cher
lecteur ». À l'oreille, chacune de ces phrases est fausse : l'auditeur n'est
pas en train de lire, et l'entendre dire le contraire le sort du livre.

**Ce module ne réécrit pas le manuscrit.** Il agit sur le texte de narration,
celui qu'on peut relire et comparer avant de payer la synthèse. Le fichier
d'origine reste ce qu'il est.

Le piège est de tout convertir. Mesuré sur quarante-six livres, « lecteur »
apparaît 163 fois, et une majorité ne désigne pas le public :

    …payer par simple contact de la main avec un lecteur…   (une puce NFC)
    …l'étude du cerveau lecteur…                            (un concept)
    …lui-même, lecteur assidu depuis l'enfance…             (un personnage)
    …se sont formées en lisant cet ouvrage…                 (un AUTRE livre)

D'où le partage retenu ici. Ce qui nomme explicitement *ce* livre est converti
tout seul, parce que la référence lève l'ambiguïté. Tout le reste est
seulement **signalé**, pour qu'une décision soit prise en la voyant plutôt
qu'en la devinant. Un module qui convertirait les 163 aurait tort 92 fois.

Certaines phrases enfin ne parlent pas de lire mais du support — « sur une
liseuse », « crayon à la main » — et aucun verbe ne les répare. Elles sont
signalées comme telles : il faut les récrire ou les couper.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Iterable

__all__ = ["Changement", "Signalement", "adapter", "rapport_texte"]


@dataclasses.dataclass(frozen=True)
class Changement:
    """Une substitution faite, avec de quoi la relire."""
    regle: str
    avant: str
    apres: str
    contexte: str


@dataclasses.dataclass(frozen=True)
class Signalement:
    """Un passage douteux, laissé intact et porté à l'attention."""
    motif: str
    extrait: str
    pourquoi: str


#: Le livre se désigne lui-même : la référence lève l'ambiguïté, on peut convertir.
#: « cet ouvrage » est volontairement absent — il désigne souvent un livre cité.
CE_LIVRE = r"(?:ce livre|ce chapitre|ces pages|ce guide|cette page|ce volume)"

#: Conjugaisons rencontrées dans le corpus, et leur équivalent à l'écoute.
#: « relire » devient « réécouter » : revenir en arrière existe aussi en audio.
VERBES = {
    "lire": "écouter", "relire": "réécouter",
    "lis": "écoute", "lit": "écoute", "lisez": "écoutez", "lisons": "écoutons",
    "lisent": "écoutent", "lisait": "écoutait", "lisaient": "écoutaient",
    "lisant": "écoutant", "relisant": "réécoutant", "relisez": "réécoutez",
    "lirez": "écouterez", "lira": "écoutera", "liront": "écouteront",
    "lirait": "écouterait", "liriez": "écouteriez", "lirons": "écouterons",
    "lu": "écouté", "lue": "écoutée", "lus": "écoutés", "lues": "écoutées",
    "relu": "réécouté", "relue": "réécoutée",
}
_VERBE_RX = "|".join(sorted(VERBES, key=len, reverse=True))

#: Verbe puis référence, ou référence puis verbe : les deux ordres existent
#: (« lire ce livre », mais aussi « ce livre se lit d'une traite »).
_AVANT = re.compile(rf"\b({_VERBE_RX})\b([^.!?\n]{{0,45}}?\b{CE_LIVRE}\b)", re.IGNORECASE)
_APRES = re.compile(rf"\b({CE_LIVRE}\b[^.!?\n]{{0,45}}?)\b({_VERBE_RX})\b", re.IGNORECASE)

#: Les intertitres : c'est là que la faute s'entend le plus, parce qu'elle est
#: annoncée. Traités à part pour ajouter « audio », qu'une phrase ordinaire
#: n'a pas besoin de répéter.
_TITRE = re.compile(
    rf"^(\s*)Comment\s+(?:lire|aborder|utiliser)\s+({CE_LIVRE})([^\n]*)$",
    re.IGNORECASE | re.MULTILINE)

#: Le support de lecture, pas l'acte : convertir le verbe y produirait une
#: phrase pire que l'originale — « écouter ce livre sur une liseuse ». Une
#: phrase qui porte un de ces mots est donc laissée telle quelle ET signalée.
#:
#: « crayon » et « stylo » n'en sont pas : un auditeur peut très bien écrire
#: en écoutant, et ces mots appartiennent aux exercices, pas au support.
_SUPPORT_LECTURE = re.compile(
    r"\b(?:liseuse|kindle|sur papier|version papier|surlign\w+|dans la marge|"
    r"note de bas de page|coin de la page|en haut de la page|imprim\w+|"
    r"tourner la page|feuillet\w*)\b", re.IGNORECASE)

#: Signalé plus largement que suppressif : ces mots méritent un regard sans
#: pour autant bloquer une conversion juste.
_SUPPORT_SIGNALE = re.compile(
    rf"{_SUPPORT_LECTURE.pattern}|\b(?:stylo|crayon)\b", re.IGNORECASE)

#: Bornes de phrase, pour savoir de quelle phrase relève une occurrence.
_FIN_PHRASE = re.compile(r"[.!?\n]")


def _phrase_autour(texte: str, debut: int, fin: int) -> str:
    """La phrase qui contient l'occurrence — l'unité où le sens se décide."""
    d = 0
    for m in _FIN_PHRASE.finditer(texte, 0, debut):
        d = m.end()
    m = _FIN_PHRASE.search(texte, fin)
    return texte[d: m.start() if m else len(texte)]

#: « lecteur » : converti, mais jamais aveuglément.
#:
#: Le relevé des 163 occurrences du corpus a renversé mon hypothèse de départ.
#: Je pensais la majorité ambiguë ; elle ne l'est pas — la plupart désignent
#: bien le public, et les laisser produirait un livre qui parle à quelqu'un
#: d'absent. Un livre dit même déjà « ce livre peut s'écouter de deux manières,
#: selon le type de lecteur que vous êtes » : ne pas convertir laisserait la
#: phrase se contredire elle-même.
#:
#: Restent des exceptions franches, relevées une à une dans le corpus. Elles
#: ne sont pas devinées : chacune a été lue dans sa phrase.
_LECTEUR = re.compile(r"\blect(?:eur|rice)s?\b", re.IGNORECASE)

LECTEUR_VERS_AUDITEUR = {
    "lecteur": "auditeur", "lecteurs": "auditeurs",
    "lectrice": "auditrice", "lectrices": "auditrices",
}

#: Ce qui porte le mot « lecteur » sans désigner le public. Motifs relevés
#: dans le corpus, avec le livre où ils apparaissent.
_LECTEUR_GARDE = re.compile(
    r"lecteurs? d'écran"                       # accessibilité (livre-06)
    r"|cerveau lecteurs?"                      # concept de Maryanne Wolf (livre-01)
    r"|lecteurs? (?:assidus?|de Proust|de romans?|de journaux|de presse)"
    r"|lecteurs? numériques"                   # sujets d'une étude (livre-01)
    r"|(?:annonceurs|abonnés) et lecteurs"     # économie de la presse (livre-08)
    r"|magazines? [^.\n]{0,40}lecteurs"        # lectorat d'un magazine (livre-21)
    r"|journ\w+ [^.\n]{0,60}lecteurs"          # lecteurs d'un journal (livre-09)
    r"|auditeur ou (?:un )?lecteur"            # la phrase oppose déjà les deux
    r"|lecteur (?:et|ou) (?:un )?auditeur"
    r"|contact [^.\n]{0,30}avec un lecteur"    # une puce NFC (livre-02)
    r"|lecteurs? de livres"                    # les lecteurs d'autres livres
    , re.IGNORECASE)

#: Un livre entier échappe à la règle. « L'EFFET PODCAST » compare page après
#: page ce que la lecture fait et ce que l'écoute fait — « le format du livre
#: demande à son lecteur la même chose que le podcast demande à son auditeur ».
#: Y remplacer lecteur par auditeur détruirait l'argument du livre.
LIVRES_SANS_CONVERSION_LECTEUR = frozenset({"livre-21-podcast"})

#: « cet ouvrage » : peut désigner ce livre-ci comme un livre cité.
_OUVRAGE = re.compile(rf"\b(?:{_VERBE_RX})\b[^.!?\n]{{0,45}}?\bcet ouvrage\b",
                      re.IGNORECASE)


#: Tous les remplacements commencent par une voyelle, ce que « lire » ne
#: faisait pas : « se lit » doit devenir « s'écoute », jamais « se écoute ».
_ELIDABLES = ("se", "ne", "je", "me", "te", "le", "la", "de", "que", "ce")
_ELISION = re.compile(rf"\b({'|'.join(_ELIDABLES)})(\s+)$", re.IGNORECASE)


def _accorder(source: str, cible: str) -> str:
    """Rend la casse du mot d'origine : « Lisez » ne doit pas devenir « écoutez »."""
    if source.isupper():
        return cible.upper()
    if source[:1].isupper():
        return cible[:1].upper() + cible[1:]
    return cible


def _elider(gauche: str) -> str:
    """Élide le mot qui précède, s'il le demande. Rend le texte de gauche corrigé.

    Tous les mots concernés finissent par une voyelle — « se », « que »,
    « la » — donc l'élision consiste à retirer cette dernière lettre et à la
    remplacer par l'apostrophe : « se » devient « s' », « que » devient « qu' ».
    """
    m = _ELISION.search(gauche)
    if not m:
        return gauche
    return gauche[:m.start()] + m.group(1)[:-1] + "'"


def _extrait(texte: str, debut: int, fin: int, marge: int = 70) -> str:
    d = max(0, debut - marge)
    f = min(len(texte), fin + marge)
    return " ".join(texte[d:f].split())


def adapter(texte: str, *, slug: str = "") -> tuple[str, list[Changement], list[Signalement]]:
    """Adapte le texte à l'écoute. Rend le texte, ce qui a changé, ce qui inquiète.

    ``slug`` nomme le livre : un livre dont le sujet *est* la différence entre
    lire et écouter ne peut pas subir la règle sur « lecteur ».
    """
    changements: list[Changement] = []
    signalements: list[Signalement] = []

    def _titre(m: re.Match) -> str:
        blanc, cible, reste = m.group(1), m.group(2), m.group(3)
        # « ce chapitre » reste un chapitre ; seul le livre gagne « audio »,
        # et une seule fois, là où l'auditeur comprend ce qu'il écoute.
        neuf = "ce livre audio" if cible.lower() == "ce livre" else cible.lower()
        remplacement = f"{blanc}Comment écouter {neuf}{reste}"
        changements.append(Changement(
            "intertitre", m.group(0).strip(), remplacement.strip(),
            _extrait(texte, m.start(), m.end())))
        return remplacement

    texte = _TITRE.sub(_titre, texte)

    # Le remplacement peut réclamer une élision du mot d'avant, qui se trouve
    # hors de la correspondance : on reconstruit donc le texte à la main
    # plutôt que d'utiliser sub(), qui ne sait pas revenir en arrière.
    def _passe(rx: re.Pattern, groupe_verbe: int) -> None:
        nonlocal texte
        morceaux: list[str] = []
        fin_precedente = 0
        for m in rx.finditer(texte):
            # Une phrase qui parle du livre papier ne se répare pas en
            # changeant son verbe : « écouter ce livre sur une liseuse » est
            # pire que l'original. On la laisse et on la signale.
            phrase = _phrase_autour(texte, m.start(), m.end())
            if _SUPPORT_LECTURE.search(phrase):
                signalements.append(Signalement(
                    "support", " ".join(phrase.split())[:180],
                    "parle du livre papier : conversion refusée, à récrire à la main"))
                continue
            verbe = m.group(groupe_verbe)
            neuf = _accorder(verbe, VERBES[verbe.lower()])
            avant_verbe = texte[fin_precedente:m.start(groupe_verbe)]
            morceaux.append(_elider(avant_verbe))
            morceaux.append(neuf)
            fin_precedente = m.end(groupe_verbe)
            changements.append(Changement(
                "verbe", verbe, neuf, _extrait(texte, m.start(), m.end())))
        if morceaux:
            morceaux.append(texte[fin_precedente:])
            texte = "".join(morceaux)

    _passe(_AVANT, 1)
    _passe(_APRES, 2)

    if slug in LIVRES_SANS_CONVERSION_LECTEUR:
        for m in _LECTEUR.finditer(texte):
            signalements.append(Signalement(
                "lecteur", _extrait(texte, m.start(), m.end()),
                f"livre exclu de la règle ({slug}) : son propos oppose lire et écouter"))
    else:
        morceaux: list[str] = []
        fin_precedente = 0
        for m in _LECTEUR.finditer(texte):
            phrase = _phrase_autour(texte, m.start(), m.end())
            if _LECTEUR_GARDE.search(phrase):
                signalements.append(Signalement(
                    "lecteur", " ".join(phrase.split())[:180],
                    "ne désigne pas le public : laissé intact"))
                continue
            mot = m.group(0)
            neuf = _accorder(mot, LECTEUR_VERS_AUDITEUR[mot.lower()])
            # « auditeur » commence par une voyelle, « lecteur » non :
            # « le lecteur » doit donner « l'auditeur », pas « le auditeur ».
            morceaux.append(_elider(texte[fin_precedente:m.start()]))
            morceaux.append(neuf)
            fin_precedente = m.end()
            changements.append(Changement(
                "lecteur", mot, neuf, _extrait(texte, m.start(), m.end())))
        if morceaux:
            morceaux.append(texte[fin_precedente:])
            texte = "".join(morceaux)
    for m in _OUVRAGE.finditer(texte):
        signalements.append(Signalement(
            "cet ouvrage", _extrait(texte, m.start(), m.end()),
            "peut désigner ce livre-ci ou un livre cité en référence"))
    deja = {s.extrait for s in signalements if s.motif == "support"}
    for m in _SUPPORT_SIGNALE.finditer(texte):
        extrait = " ".join(_phrase_autour(texte, m.start(), m.end()).split())[:180]
        if extrait not in deja:
            signalements.append(Signalement(
                "support", extrait,
                "mentionne le support ou l'écrit : à vérifier à l'oreille"))

    return texte, changements, signalements


def rapport_texte(changements: Iterable[Changement],
                  signalements: Iterable[Signalement]) -> str:
    """Un rapport qu'on lit avant de payer la narration, pas après."""
    ch, si = list(changements), list(signalements)
    lignes = [f"{len(ch)} adaptation(s), {len(si)} passage(s) à décider à la main", ""]
    if ch:
        lignes.append("--- adapté ---")
        for c in ch:
            lignes.append(f"  [{c.regle}] {c.avant} → {c.apres}")
            lignes.append(f"      {c.contexte}")
    if si:
        lignes.append("")
        lignes.append("--- signalé, laissé intact ---")
        par_motif: dict[str, list[Signalement]] = {}
        for s in si:
            par_motif.setdefault(s.motif, []).append(s)
        for motif, groupe in par_motif.items():
            lignes.append(f"  {motif} ({len(groupe)}) — {groupe[0].pourquoi}")
            for s in groupe:
                lignes.append(f"      {s.extrait}")
    return "\n".join(lignes)
