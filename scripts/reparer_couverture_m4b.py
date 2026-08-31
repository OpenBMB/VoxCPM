#!/usr/bin/env python3
"""Remplace la couverture embarquée d'un M4B déjà assemblé.

Pourquoi ce script existe. La règle de choix de couverture de
``construire_file.py`` retenait autrefois le premier fichier image assez gros,
sans regarder l'image : trois livres sont partis en narration avec leur
vignette ebook en portrait, et l'ont embarquée dans leur M4B. La règle est
corrigée, mais les livres, eux, sont narrés — et une couverture n'est pas une
raison de redépenser trois heures de GPU par livre.

Le remplacement est un remuxage : l'audio est recopié tel quel, octet pour
octet, et seuls l'image et l'index des chapitres sont réécrits. Le fichier
d'origine n'est jamais écrasé en place — on écrit à côté, on vérifie, puis on
permute. Un M4B à demi réécrit est indistinguable d'un M4B valide tant qu'on
ne l'ouvre pas, et il ne s'ouvre qu'au dépôt.

    python scripts/reparer_couverture_m4b.py livre.m4b couverture.jpg
    python scripts/reparer_couverture_m4b.py --lot dossier_livres --catalogue queue/queue_catalogue.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

#: Côté minimal exigé par Audible (ACX). Une couverture qui ne l'atteint pas
#: n'a aucune raison d'entrer dans un M4B : elle serait refusée au dépôt.
COTE_MINIMAL = 2400


def ffoutil(nom: str) -> str:
    """ffmpeg est installé mais absent du PATH des shells déjà ouverts."""
    trouve = shutil.which(nom)
    if trouve:
        return trouve
    base = pathlib.Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    for exe in base.rglob(f"{nom}.exe"):
        return str(exe)
    print(f"{nom} introuvable", file=sys.stderr)
    raise SystemExit(2)


def dimensions_image(chemin: pathlib.Path) -> tuple[int, int] | None:
    r = subprocess.run(
        [ffoutil("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(chemin)],
        capture_output=True, text=True, encoding="utf-8")
    ligne = r.stdout.strip().splitlines()
    if not ligne:
        return None
    try:
        l, h = (int(x) for x in ligne[0].split(",")[:2])
    except ValueError:
        return None
    return l, h


def couverture_du_m4b(m4b: pathlib.Path) -> tuple[int, int] | None:
    return dimensions_image(m4b)


def nb_chapitres(m4b: pathlib.Path) -> int:
    r = subprocess.run(
        [ffoutil("ffprobe"), "-v", "error", "-show_chapters", "-of", "json", str(m4b)],
        capture_output=True, text=True, encoding="utf-8")
    try:
        return len(json.loads(r.stdout or "{}").get("chapters", []))
    except json.JSONDecodeError:
        return -1


def duree(m4b: pathlib.Path) -> float:
    r = subprocess.run(
        [ffoutil("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(m4b)],
        capture_output=True, text=True, encoding="utf-8")
    try:
        return float(r.stdout.strip())
    except ValueError:
        return -1.0


def remplacer(m4b: pathlib.Path, cover: pathlib.Path, *, verbeux: bool = True) -> bool:
    """Vrai si la couverture a été remplacée et le résultat vérifié."""
    dim = dimensions_image(cover)
    if dim is None:
        print(f"  couverture illisible : {cover}", file=sys.stderr)
        return False
    if dim[0] != dim[1] or dim[0] < COTE_MINIMAL:
        print(f"  couverture refusée ({dim[0]}x{dim[1]}) : il faut un carré "
              f"d'au moins {COTE_MINIMAL} px", file=sys.stderr)
        return False

    avant_chap, avant_duree = nb_chapitres(m4b), duree(m4b)
    # Le fichier de travail garde l'extension .m4b : ffmpeg choisit son
    # conteneur d'après elle, et un « .m4b.nouveau » le laisse sans format.
    # Le suffixe reste hors de « *_complet.m4b », pour qu'un fichier oublié
    # après un plantage ne soit jamais repris pour un livre par le mode --lot.
    tmp = m4b.with_name(f"{m4b.stem}.reparation.m4b")

    # -map_chapters recrée l'index à partir de la source : la piste de données
    # d'origine n'est donc pas remappée, elle est régénérée. -map_metadata
    # garde titre, auteur et album, que le dépôt lit avant d'ouvrir l'audio.
    cmd = [ffoutil("ffmpeg"), "-y", "-loglevel", "error",
           "-i", str(m4b), "-i", str(cover),
           "-map", "0:a", "-map", "1:v",
           "-c", "copy", "-map_metadata", "0", "-map_chapters", "0",
           "-disposition:v:0", "attached_pic",
           str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(f"  ffmpeg a échoué : {r.stderr.strip()[:400]}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    # Vérifier avant de permuter : la couverture est la raison du remuxage,
    # les chapitres et la durée en sont le prix à ne pas payer.
    apres_dim = couverture_du_m4b(tmp)
    apres_chap, apres_duree = nb_chapitres(tmp), duree(tmp)
    if apres_dim != dim:
        print(f"  la nouvelle couverture ne s'est pas embarquée ({apres_dim})", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    if apres_chap != avant_chap:
        print(f"  chapitres perdus : {avant_chap} → {apres_chap}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    if abs(apres_duree - avant_duree) > 1.0:
        print(f"  durée modifiée : {avant_duree:.1f} s → {apres_duree:.1f} s", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(m4b)
    if verbeux:
        print(f"  {apres_dim[0]}x{apres_dim[1]}, {apres_chap} chapitres, "
              f"{apres_duree/3600:.2f} h — vérifié")
    return True


def slug_du_m4b(m4b: pathlib.Path) -> str:
    """book_livre_01_titre_complet.m4b → livre-01-titre."""
    return m4b.stem.replace("book_", "").removesuffix("_complet").replace("_", "-")


def main() -> int:
    # Le script écrit des filets et des flèches : sur une console cp1252 il
    # tombait dessus, après avoir déjà remplacé la couverture de deux livres.
    for flux in (sys.stdout, sys.stderr):
        try:
            flux.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("m4b", nargs="?", type=pathlib.Path)
    ap.add_argument("couverture", nargs="?", type=pathlib.Path)
    ap.add_argument("--lot", type=pathlib.Path, metavar="DOSSIER",
                    help="parcourt les *_complet.m4b et ne répare que ceux dont "
                         "la couverture embarquée n'est pas un carré conforme")
    ap.add_argument("--catalogue", type=pathlib.Path, default=pathlib.Path("queue/queue_catalogue.json"),
                    help="où lire la couverture de chaque livre, en mode --lot")
    ap.add_argument("--racine", type=pathlib.Path, default=pathlib.Path("."),
                    help="racine à laquelle les chemins du catalogue sont relatifs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.lot:
        catalogue = {e["slug"]: e for e in json.loads(
            args.catalogue.read_text(encoding="utf-8"))}
        a_reparer, sans_source, ok = [], [], 0
        # ``rglob`` et non ``glob`` : un livre rapatrié est un *dossier*
        # (M4B + export ACX + rapport), pas un fichier posé à plat. Avec un
        # glob plat, ce mode parcourait zéro fichier et annonçait tout de même
        # « 0 à réparer » — le rapport d'un lot sain et celui d'un lot jamais
        # regardé étaient le même.
        m4bs = sorted(args.lot.rglob("*_complet.m4b"))
        if not m4bs:
            print(f"aucun *_complet.m4b sous {args.lot} — rien n'a été examiné")
            return 1
        for m4b in m4bs:
            dim = couverture_du_m4b(m4b)
            if dim and dim[0] == dim[1] and dim[0] >= COTE_MINIMAL:
                ok += 1
                continue
            slug = slug_du_m4b(m4b)
            e = catalogue.get(slug)
            cover = (args.racine / e["cover"]) if e else None
            if cover is None or not cover.exists():
                sans_source.append((m4b.name, slug))
                continue
            a_reparer.append((m4b, cover, dim))

        print(f"{ok} livre(s) déjà conformes, {len(a_reparer)} à réparer, "
              f"{len(sans_source)} sans couverture de rechange")
        for nom, slug in sans_source:
            print(f"  ⚠ {nom} : rien dans le catalogue pour « {slug} »")

        echecs = 0
        for m4b, cover, dim in a_reparer:
            # ``dim`` est None quand le M4B n'a aucune image : c'est le cas de
            # `livre-rebatir-intimite`, et le formatage l'ignorait.
            actuel = f"{dim[0]}x{dim[1]}" if dim else "sans couverture"
            print(f"── {m4b.name}  ({actuel} → {cover.name})")
            if args.dry_run:
                continue
            if not remplacer(m4b, cover):
                echecs += 1
        return 1 if (echecs or sans_source) else 0

    if not args.m4b or not args.couverture:
        ap.error("donner un M4B et une couverture, ou --lot DOSSIER")
    print(f"── {args.m4b.name}")
    if args.dry_run:
        print(f"  {couverture_du_m4b(args.m4b)} → {dimensions_image(args.couverture)}")
        return 0
    return 0 if remplacer(args.m4b, args.couverture) else 1


if __name__ == "__main__":
    raise SystemExit(main())
