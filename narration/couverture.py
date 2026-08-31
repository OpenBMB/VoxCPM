"""La règle de couverture d'un livre audio, en un seul endroit.

Un distributeur ne regarde pas le nom du fichier ni le dossier d'où il vient :
il regarde l'image. Audible (ACX) exige une couverture **carrée** d'au moins
2400 pixels de côté, et les autres plateformes reprennent ce seuil. Tout le
reste — jaquette imprimée, rabat complet, vignette ebook en portrait — est une
autre image du même livre, pour un autre produit.

Cette règle vivait dans ``scripts/construire_file.py``, c'est-à-dire au moment
où la file se construit. Elle y arrivait trop tard une fois sur deux : un livre
narré à la main, ou repris par ``relancer_renarration.sh``, ne passe pas par le
constructeur de file et embarquait ce qu'on lui donnait. Elle est ici pour que
le constructeur de file *et* le narrateur posent la même question à la même
image, et pour qu'un M4B sans couverture cesse d'être un fichier qu'on découvre
au dépôt.

Rien n'est décodé : seuls les premiers octets sont lus, donc mesurer cent
couvertures coûte le prix d'un ``ls`` et n'ajoute aucune dépendance.
"""
from __future__ import annotations

import pathlib
import struct
from typing import NamedTuple, Optional

#: Extensions qu'un distributeur accepte comme couverture.
IMAGES = (".jpg", ".jpeg", ".png", ".webp")

#: Côté minimal d'une couverture audio, en pixels : c'est le seuil d'Audible
#: (ACX), et il est repris tel quel par les autres distributeurs.
COTE_MINIMAL = 2400

#: En deçà, un fichier image est une icône ou une vignette, pas une couverture.
TAILLE_MINIMALE = 20_000


def dimensions(f: pathlib.Path) -> Optional[tuple[int, int]]:
    """Largeur et hauteur d'une image, lues dans son en-tête.

    Sans dépendance : le catalogue tient dans trois formats et leurs en-têtes
    tiennent en trente lignes. Rien n'est décodé, seuls les premiers octets
    sont lus.
    """
    try:
        with f.open("rb") as fh:
            tete = fh.read(32)
            if tete[:8] == b"\x89PNG\r\n\x1a\n":
                l, h = struct.unpack(">II", tete[16:24])
                return int(l), int(h)
            if tete[:4] == b"RIFF" and tete[8:12] == b"WEBP":
                fh.seek(0)
                d = fh.read(40)
                if d[12:16] == b"VP8X":
                    return (int.from_bytes(d[24:27], "little") + 1,
                            int.from_bytes(d[27:30], "little") + 1)
                if d[12:16] == b"VP8 ":
                    return (int.from_bytes(d[26:28], "little") & 0x3FFF,
                            int.from_bytes(d[28:30], "little") & 0x3FFF)
                if d[12:16] == b"VP8L":
                    b = int.from_bytes(d[21:25], "little")
                    return (b & 0x3FFF) + 1, ((b >> 14) & 0x3FFF) + 1
                return None
            if tete[:2] == b"\xff\xd8":
                # JPEG : sauter de marqueur en marqueur jusqu'au SOFn, seul
                # segment qui porte les dimensions. Les SOF 4, 8 et 12 sont
                # des marqueurs de table, pas des cadres — d'où l'exclusion.
                fh.seek(2)
                while True:
                    octet = fh.read(1)
                    if not octet:
                        return None
                    if octet != b"\xff":
                        continue
                    while octet == b"\xff":
                        octet = fh.read(1)
                    marqueur = octet[0]
                    if 0xC0 <= marqueur <= 0xCF and marqueur not in (0xC4, 0xC8, 0xCC):
                        fh.read(3)
                        h, l = struct.unpack(">HH", fh.read(4))
                        return int(l), int(h)
                    taille = struct.unpack(">H", fh.read(2))[0]
                    fh.seek(taille - 2, 1)
    except (OSError, struct.error, IndexError):
        return None
    return None


class Verdict(NamedTuple):
    """Ce qu'on peut dire d'une couverture avant de lancer trois heures de GPU.

    ``raison`` est vide quand elle est conforme, et rédigée pour être lue dans
    un journal détaché — c'est souvent la seule trace qu'il en restera.
    """

    conforme: bool
    dimensions: Optional[tuple[int, int]]
    raison: str

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.conforme


def inspecter(chemin: pathlib.Path | str | None) -> Verdict:
    """Cette image peut-elle être déposée comme couverture de livre audio ?

    Répond aussi — et surtout — quand il n'y a pas d'image du tout : c'est le
    cas qui est passé inaperçu, parce qu'un chemin absent ne déclenche aucune
    des vérifications qu'on écrit pour un chemin présent.
    """
    if chemin is None or str(chemin) == "":
        return Verdict(False, None, "aucune couverture n'a été fournie")
    f = pathlib.Path(chemin)
    if not f.is_file():
        return Verdict(False, None, f"fichier introuvable : {f}")
    if f.suffix.lower() not in IMAGES:
        return Verdict(False, None,
                       f"format non accepté : {f.suffix or '(sans extension)'} "
                       f"(attendu {', '.join(IMAGES)})")
    dim = dimensions(f)
    if dim is None:
        return Verdict(False, None, f"image illisible ou tronquée : {f.name}")
    l, h = dim
    if l != h:
        return Verdict(False, dim,
                       f"couverture non carrée : {l}×{h} — les distributeurs "
                       f"exigent un carré")
    if l < COTE_MINIMAL:
        return Verdict(False, dim,
                       f"couverture trop petite : {l}×{h} — minimum "
                       f"{COTE_MINIMAL}×{COTE_MINIMAL}")
    return Verdict(True, dim, "")


def choisir(d: pathlib.Path) -> Optional[pathlib.Path]:
    """La couverture *audio* d'un dossier de livre : carrée, ≥ 2400 px.

    Un dossier de livre contient plusieurs couvertures qui ne servent pas au
    même produit. Prendre la première venue passe inaperçu jusqu'au dépôt ; la
    contrainte du distributeur est donc la règle de choix, au lieu d'un ordre
    de dossiers qui ne la connaissait pas. À égalité, le fichier nommé pour
    l'audio l'emporte, puis le plus grand.

    Le classement précède la mesure, et on s'arrête à la première conforme :
    le catalogue vit sur OneDrive, où lire le moindre octet d'un fichier le
    fait descendre en entier. Mesurer les dix images d'un livre pour n'en
    garder qu'une rapatriait des gigaoctets.
    """
    vues: list[pathlib.Path] = []
    for sous in ("_covers_v3", "_covers_v2", "_covers", "couverture", "formats"):
        rep = d / sous
        if rep.is_dir():
            vues += sorted(rep.rglob("*"))
    vues += sorted(d.glob("*"))

    candidates = [f for f in vues
                  if f.suffix.lower() in IMAGES and f.is_file()
                  and f.stat().st_size > TAILLE_MINIMALE]
    candidates.sort(key=lambda f: (0 if "audio" in f.name.lower() else 1,
                                   -f.stat().st_size))
    for f in candidates:
        dim = dimensions(f)
        if dim is not None and dim[0] == dim[1] and dim[0] >= COTE_MINIMAL:
            return f
    return None
