"""Tests du choix de couverture de scripts/construire_file.py.

Le cas qui a motivé ce fichier : un dossier de livre porte plusieurs
couvertures — jaquette imprimée, rabat complet, vignette ebook en portrait,
couverture audio carrée. La règle d'origine parcourait des dossiers dans un
ordre fixe et retenait le premier fichier assez gros, sans jamais regarder
l'image. Trois livres sont ainsi partis en narration avec une couverture
ebook 1600×2560, embarquée dans leur M4B ; Audible ne l'aurait refusée qu'au
dépôt, une fois les trois heures de GPU dépensées.

D'où le sens des tests qui suivent : ce n'est pas l'emplacement du fichier
qui décide, c'est sa forme.
"""
from __future__ import annotations

import importlib.util
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narration import couverture as regle_couverture

spec = importlib.util.spec_from_file_location(
    "construire_file", ROOT / "scripts" / "construire_file.py"
)
construire_file = importlib.util.module_from_spec(spec)
sys.modules["construire_file"] = construire_file
spec.loader.exec_module(construire_file)


def _png(chemin: Path, largeur: int, hauteur: int, octets: int = 30_000) -> Path:
    """Un PNG dont l'en-tête est vrai et le contenu quelconque.

    Les dimensions sont lues dans le IHDR, donc l'image n'a pas besoin d'être
    décodable — mais elle doit peser plus que le seuil de vignette, sinon elle
    est écartée avant d'être mesurée.
    """
    ihdr = struct.pack(">II", largeur, hauteur) + bytes([8, 2, 0, 0, 0])
    bloc = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr
    bloc += struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    bourrage = b"\x00" * max(0, octets - len(bloc) - 8)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_bytes(b"\x89PNG\r\n\x1a\n" + bloc + bourrage)
    return chemin


def _jpeg(chemin: Path, largeur: int, hauteur: int, octets: int = 30_000) -> Path:
    """Un JPEG dont le SOF0 porte les vraies dimensions.

    Un segment APP0 le précède, comme dans tout fichier réel : c'est lui qui
    vérifie que le parcours saute bien de marqueur en marqueur au lieu de
    lire à un décalage fixe.
    """
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
    sof0 = b"\xff\xc0" + struct.pack(">H", 11) + bytes([8]) + struct.pack(">HH", hauteur, largeur) + bytes([1, 1, 17, 0])
    corps = b"\xff\xd8" + app0 + sof0 + b"\xff\xda"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_bytes(corps + b"\x00" * max(0, octets - len(corps)))
    return chemin


class TestDimensions:
    """Mesurer sans décoder — sinon le catalogue réclame une dépendance."""

    def test_png(self, tmp_path):
        f = _png(tmp_path / "c.png", 3000, 3000)
        assert construire_file.dimensions(f) == (3000, 3000)

    def test_jpeg_passe_par_dessus_les_segments_dentete(self, tmp_path):
        f = _jpeg(tmp_path / "c.jpg", 1600, 2560)
        assert construire_file.dimensions(f) == (1600, 2560)

    def test_un_fichier_qui_nest_pas_une_image_ne_leve_pas(self, tmp_path):
        f = tmp_path / "c.png"
        f.write_bytes(b"ceci n'est pas une image")
        assert construire_file.dimensions(f) is None


class TestCouverture:
    def test_la_carree_lemporte_sur_la_premiere_venue(self, tmp_path):
        """Le cas réel : « couverture/ » vient avant dans l'alphabet et dans
        l'ancien ordre des dossiers, mais son contenu est un portrait."""
        _jpeg(tmp_path / "couverture" / "couverture_front.jpg", 1600, 2560)
        carree = _jpeg(tmp_path / "_covers_v3" / "audio_cover.jpg", 3000, 3000)
        assert construire_file.couverture(tmp_path) == carree

    def test_une_carree_trop_petite_ne_compte_pas(self, tmp_path):
        """1600×1600 est carré et refusé quand même : le seuil du distributeur
        n'est pas la forme, c'est la forme *et* la taille."""
        _jpeg(tmp_path / "_covers_v3" / "audio_cover.jpg", 1600, 1600)
        assert construire_file.couverture(tmp_path) is None

    def test_aucune_carree_vaut_aucune_couverture(self, tmp_path):
        """Un livre sans couverture carrée est écarté avec sa raison plutôt
        que narré pour rien — trois heures de GPU sont en jeu."""
        _jpeg(tmp_path / "couverture" / "couverture_front.jpg", 1600, 2560)
        _png(tmp_path / "couverture" / "couverture_wrap.png", 3200, 2000)
        assert construire_file.couverture(tmp_path) is None

    def test_a_forme_egale_le_nom_audio_lemporte(self, tmp_path):
        """Deux carrés valides : celui qui se nomme pour l'audio gagne, même
        plus léger que la version imprimée."""
        _png(tmp_path / "_covers_v3" / "front_print.png", 3000, 3000, octets=90_000)
        audio = _jpeg(tmp_path / "_covers_v3" / "audio_cover.jpg", 3000, 3000, octets=40_000)
        assert construire_file.couverture(tmp_path) == audio

    def test_a_defaut_de_nom_la_plus_grande_lemporte(self, tmp_path):
        grande = _png(tmp_path / "_covers_v3" / "b.png", 3000, 3000, octets=90_000)
        _png(tmp_path / "_covers_v3" / "a.png", 2400, 2400, octets=30_000)
        assert construire_file.couverture(tmp_path) == grande

    def test_la_racine_du_dossier_est_lue_en_dernier_recours(self, tmp_path):
        seule = _jpeg(tmp_path / "cover.jpg", 2400, 2400)
        assert construire_file.couverture(tmp_path) == seule

    def test_on_ne_mesure_pas_ce_quon_ne_gardera_pas(self, tmp_path, monkeypatch):
        """Le catalogue vit sur OneDrive : lire un octet d'une image la fait
        descendre en entier. Mesurer les dix couvertures d'un livre pour n'en
        garder qu'une rapatriait des gigaoctets — la conformité se vérifie
        donc dans l'ordre de préférence, et s'arrête au premier succès."""
        _png(tmp_path / "_covers_v3" / "full_wrap.png", 6000, 4000, octets=200_000)
        _png(tmp_path / "_covers_v3" / "back_cover.png", 3000, 3000, octets=150_000)
        _jpeg(tmp_path / "_covers_v3" / "audio_cover.jpg", 3000, 3000, octets=40_000)

        mesurees = []
        vraie = regle_couverture.dimensions

        def compter(f):
            mesurees.append(f.name)
            return vraie(f)

        # La règle vit dans narration.couverture ; construire_file n'en est
        # plus qu'un appelant, comme narrate_book.
        monkeypatch.setattr(regle_couverture, "dimensions", compter)
        choix = construire_file.couverture(tmp_path)

        assert choix.name == "audio_cover.jpg"
        assert mesurees == ["audio_cover.jpg"]

    def test_une_vignette_reste_ignoree(self, tmp_path):
        """Le seuil de taille précède la mesure : une miniature carrée de
        quelques kilo-octets n'est pas une couverture."""
        _jpeg(tmp_path / "_covers_v3" / "audio_cover.jpg", 3000, 3000, octets=5_000)
        assert construire_file.couverture(tmp_path) is None
