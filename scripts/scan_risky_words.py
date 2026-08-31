"""Repérer, dans les textes à narrer, les mots qu'un moteur français dira mal.

On ne peut pas corriger ce qu'on n'a pas entendu, et écouter quatre-vingt-dix
heures pour trouver dix mots n'est pas une méthode. Ce script fait l'inverse :
il cherche dans le texte les formes qui *mettent un moteur en défaut*, les
classe par nombre d'occurrences, et laisse l'oreille trancher.

Le classement par fréquence est le cœur du tri. Un sigle lu quatre-vingts fois
dans un catalogue coûte quatre-vingts fautes ; un nom propre lu deux fois n'en
coûte que deux. À temps d'écoute égal, on corrige le premier.

Quatre familles, par ordre de dégât :

* **Sigles** — « TDAH », « ADN », « IA ». Le moteur hésite entre épeler et lire
  comme un mot, et se trompe dans les deux sens.
* **Mots étrangers** — « burnout », « mindfulness ». Lus avec des règles
  françaises, ils deviennent méconnaissables.
* **Noms propres** — capitale en milieu de phrase. Aucun moteur ne les connaît
  tous, et un nom d'auteur écorché à chaque citation s'entend.
* **Restes typographiques** — chiffres romains, symboles, unités collées.

Rien n'est corrigé ici. Le rapport nourrit `try_pronunciation.py`, qui fait
entendre les candidats, et seul ce qui a été entendu entre dans le lexique.

    python scripts/scan_risky_words.py queue/*.txt --top 40
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Un sigle : au moins deux capitales d'affilée, éventuellement pointées.
SIGLE = re.compile(r"\b(?:[A-ZÀ-Þ]\.){2,}|\b[A-ZÀ-Þ]{2,6}\b")

# Une capitale en milieu de phrase : nom propre probable. On exclut le début de
# phrase, où la capitale ne dit rien.
NOM_PROPRE = re.compile(r"(?<![.!?…]\s)(?<!^)\b([A-ZÀ-Þ][a-zà-ÿ]{2,})\b", re.MULTILINE)

# Suites de lettres qui ne se prononcent pas en français : marqueurs d'emprunt.
ETRANGER = re.compile(
    r"\b\w*(?:ing|ness|ment\b(?=\s*(?:de|of)\b)|ough|augh|eigh|oo[kdmt]|ee[dnpt]"
    r"|ck\b|sh\b|th|wh|ay\b|ey\b|ss\b)\w*\b",
    re.IGNORECASE,
)

CHIFFRE_ROMAIN = re.compile(r"\b(?=[MDCLXVI]{2,})M*(?:C[MD]|D?C{0,3})(?:X[CL]|L?X{0,3})(?:I[XV]|V?I{0,3})\b")

# Mots français fréquents qu'on ne veut pas voir remonter comme noms propres.
BANALS = {
    "Les", "Des", "Une", "Elle", "Nous", "Vous", "Cette", "Mais", "Pour", "Dans",
    "Avec", "Sans", "Sous", "Leur", "Tout", "Tous", "Plus", "Bien", "Cela", "Quand",
    "Comme", "Alors", "Puis", "Ainsi", "Donc", "Car", "Que", "Qui", "Est", "Son",
    "Ses", "Ces", "Ils", "Elles", "Chaque", "Certains", "Aujourd", "Enfin", "Depuis",
    "Pourtant", "Cependant", "Toutefois", "Parce", "Lorsque", "Chapitre", "Partie",
}


def scanner(textes: dict[str, str]) -> dict[str, collections.Counter]:
    trouve = {k: collections.Counter() for k in ("sigles", "étrangers", "noms propres", "romains")}
    for texte in textes.values():
        for m in SIGLE.findall(texte):
            if m.upper() not in {"OK"}:
                trouve["sigles"][m] += 1
        for m in NOM_PROPRE.findall(texte):
            if m not in BANALS:
                trouve["noms propres"][m] += 1
        for m in ETRANGER.findall(texte):
            trouve["étrangers"][m.lower()] += 1
        for m in CHIFFRE_ROMAIN.findall(texte):
            trouve["romains"][m] += 1
    # Un sigle est déjà compté comme sigle ; qu'il ressorte en nom propre est du bruit.
    for s in list(trouve["sigles"]):
        trouve["noms propres"].pop(s, None)
    return trouve


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fichiers", nargs="+", help="textes préparés (.txt)")
    ap.add_argument("--top", type=int, default=30, help="entrées par famille (défaut : 30)")
    ap.add_argument("--min", type=int, default=3, help="occurrences minimales (défaut : 3)")
    ap.add_argument("--json", help="écrire le rapport ici")
    args = ap.parse_args()

    textes = {}
    for entree in args.fichiers:
        chemin = pathlib.Path(entree)
        # Un shell POSIX développe déjà le motif, un cmd.exe non : accepter les
        # deux plutôt que de dépendre de qui appelle.
        chemins = [chemin] if chemin.is_file() else sorted(
            chemin.parent.glob(chemin.name) if chemin.parent != chemin else []
        )
        for p in chemins:
            if p.is_file():
                textes[p.name] = p.read_text(encoding="utf-8", errors="replace")
    if not textes:
        print("aucun fichier lu", file=sys.stderr)
        return 1

    total = sum(len(t) for t in textes.values())
    print(f"{len(textes)} texte(s), {total} caractères\n")

    trouve = scanner(textes)
    rapport = {}
    for famille, compteur in trouve.items():
        retenu = [(m, n) for m, n in compteur.most_common() if n >= args.min][: args.top]
        rapport[famille] = retenu
        if not retenu:
            continue
        print(f"=== {famille.upper()} ({len(retenu)} retenu(s), seuil {args.min}) ===")
        for mot, n in retenu:
            print(f"  {n:>5} × {mot}")
        print()

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"rapport écrit dans {args.json}")

    print("Rien n'a été corrigé. Passez les candidats à try_pronunciation.py,")
    print("écoutez, et n'inscrivez au lexique que ce qui sonne faux.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
