"""La couverture est vérifiée avant la narration, pas après.

Deux défauts réels sont derrière ce fichier. Trois livres sont partis avec une
vignette ebook 1600×2560 embarquée dans leur M4B, refusée au dépôt seulement —
c'est la règle de forme, corrigée d'abord dans le constructeur de file. Puis
``livre-rebatir-intimite`` est sorti **sans aucune couverture** : son entrée de
file n'en portait pas, ``narrate_book`` s'en accommodait par un ``print`` à
l'assemblage, et le pré-vol de la file ne pouvait rien voir puisqu'il relançait
une commande réduite, sans ``--assemble`` ni ``--cover``.

D'où les trois choses tenues ici : le verdict dit *pourquoi*, la narration
refuse de démarrer avant la première minute de GPU, et le pré-vol vole le même
plan que la vraie prise.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narration import couverture

SR = 24000
BASE_SEED = 4242


def _png(chemin: Path, largeur: int, hauteur: int, octets: int = 30_000) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    ihdr = struct.pack(">II", largeur, hauteur) + b"\x08\x02\x00\x00\x00"
    bloc = b"\x00\x00\x00\rIHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    chemin.write_bytes(b"\x89PNG\r\n\x1a\n" + bloc + b"\x00" * octets)
    return chemin


def _jpeg(chemin: Path, largeur: int, hauteur: int, octets: int = 30_000) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, hauteur, largeur, 3) + b"\x00" * 9
    chemin.write_bytes(b"\xff\xd8" + sof + b"\x00" * octets)
    return chemin


class TestInspecter:
    """Le verdict doit être lisible dans un journal détaché : c'est souvent
    tout ce qu'il en restera."""

    def test_carree_et_grande_est_conforme(self, tmp_path):
        v = couverture.inspecter(_jpeg(tmp_path / "audio.jpg", 3000, 3000))
        assert v.conforme and v.dimensions == (3000, 3000) and v.raison == ""
        assert bool(v) is True

    def test_absente_est_dite_absente(self):
        """Le cas qui a coûté un livre : un chemin vide ne déclenchait aucune
        des vérifications écrites pour un chemin présent."""
        v = couverture.inspecter(None)
        assert not v.conforme and "aucune couverture" in v.raison
        assert not couverture.inspecter("")

    def test_introuvable_nomme_le_fichier(self, tmp_path):
        v = couverture.inspecter(tmp_path / "pas_la.jpg")
        assert not v.conforme and "pas_la.jpg" in v.raison

    def test_portrait_refuse_avec_ses_dimensions(self, tmp_path):
        """1600×2560 : la vignette ebook, exactement celle des trois livres."""
        v = couverture.inspecter(_jpeg(tmp_path / "ebook.jpg", 1600, 2560))
        assert not v.conforme
        assert v.dimensions == (1600, 2560)
        assert "carrée" in v.raison and "1600" in v.raison

    def test_carree_mais_trop_petite(self, tmp_path):
        v = couverture.inspecter(_png(tmp_path / "petite.png", 1400, 1400))
        assert not v.conforme and "trop petite" in v.raison
        assert str(couverture.COTE_MINIMAL) in v.raison

    def test_format_non_accepte(self, tmp_path):
        f = tmp_path / "couverture.pdf"
        f.write_bytes(b"%PDF-1.7" + b"\x00" * 30_000)
        assert "format non accept" in couverture.inspecter(f).raison

    def test_illisible(self, tmp_path):
        f = tmp_path / "tronquee.jpg"
        f.write_bytes(b"\xff\xd8\xff")
        assert "illisible" in couverture.inspecter(f).raison


# --------------------------------------------------------------------------
# narrate_book : le refus arrive avant le modèle, donc aussi en --dry-run.
# --------------------------------------------------------------------------
class StubDemo:
    def __init__(self, **_kwargs) -> None:
        pass

    def generate_tts_audio(self, *, text_input, seed=None, **_kwargs):
        secondes = max(0.5, len((text_input or "").strip()) / 17.0)
        n = int(SR * secondes)
        rng = np.random.default_rng(1)
        return SR, (rng.normal(0, 0.2, n)).astype(np.float32), None


app_stub = types.ModuleType("app")
app_stub.PRESET_VOICES = [
    {"name": "Voix de test", "description": "voix française de test", "seed": BASE_SEED}
]
app_stub._PRESET_BY_NAME = {"Voix de test": app_stub.PRESET_VOICES[0]}
app_stub._OUTPUT_DIR = ROOT / "output"
app_stub._sanitize_filename = lambda name: name
app_stub.VoxCPMDemo = StubDemo
sys.modules.setdefault("app", app_stub)

spec = importlib.util.spec_from_file_location("narrate_book", ROOT / "scripts" / "narrate_book.py")
narrate_book = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(narrate_book)


@pytest.fixture
def livre(tmp_path):
    chemin = tmp_path / "livre.txt"
    chemin.write_text(
        "Chapitre premier. Une phrase de longueur raisonnable pour un segment.\n",
        encoding="utf-8",
    )
    return chemin


def _lancer(monkeypatch, livre, outdir, *extra) -> int:
    monkeypatch.setattr(sys, "argv", [
        "narrate_book.py", str(livre), "--voice", "Voix de test",
        "--outdir", str(outdir), "--no-credits", "--dry-run", *extra,
    ])
    return narrate_book.main()


class TestRefusAuPreVol:
    def test_sans_couverture_le_livre_ne_part_pas(self, monkeypatch, livre, tmp_path, capsys):
        code = _lancer(monkeypatch, livre, tmp_path / "out", "--assemble", "m4b")
        assert code == 2
        assert "REFUS" in capsys.readouterr().out

    def test_couverture_portrait_refusee(self, monkeypatch, livre, tmp_path, capsys):
        cover = _jpeg(tmp_path / "ebook.jpg", 1600, 2560)
        code = _lancer(monkeypatch, livre, tmp_path / "out", "--assemble", "m4b",
                       "--cover", str(cover))
        assert code == 2
        assert "carr" in capsys.readouterr().out

    def test_couverture_conforme_passe_et_se_dit(self, monkeypatch, livre, tmp_path, capsys):
        cover = _jpeg(tmp_path / "audio.jpg", 3000, 3000)
        code = _lancer(monkeypatch, livre, tmp_path / "out", "--assemble", "m4b",
                       "--cover", str(cover))
        assert code == 0
        assert "3000×3000" in capsys.readouterr().out

    def test_absence_assumee_est_permise(self, monkeypatch, livre, tmp_path, capsys):
        """``--no-cover`` reste une décision qu'on peut prendre — elle est
        seulement dite, au lieu d'être subie."""
        code = _lancer(monkeypatch, livre, tmp_path / "out", "--assemble", "m4b",
                       "--no-cover")
        assert code == 0
        assert "aucune (--no-cover)" in capsys.readouterr().out

    def test_sans_assemblage_rien_nest_exige(self, monkeypatch, livre, tmp_path):
        """Une narration qui ne produit pas de fichier à déposer n'a pas de
        couverture à porter."""
        assert _lancer(monkeypatch, livre, tmp_path / "out") == 0


# --------------------------------------------------------------------------
# narrate_queue : toute la file est examinée avant le premier livre.
# --------------------------------------------------------------------------
spec_q = importlib.util.spec_from_file_location(
    "narrate_queue", ROOT / "scripts" / "narrate_queue.py"
)
narrate_queue = importlib.util.module_from_spec(spec_q)
assert spec_q.loader is not None
spec_q.loader.exec_module(narrate_queue)


class TestFile:
    def test_la_file_nomme_les_livres_sans_couverture(self, tmp_path):
        bonne = _jpeg(tmp_path / "audio.jpg", 3000, 3000)
        manquantes = narrate_queue.couvertures_manquantes([
            {"slug": "livre-a", "cover": str(bonne)},
            {"slug": "livre-rebatir-intimite"},
            {"slug": "livre-c", "cover": str(_jpeg(tmp_path / "ebook.jpg", 1600, 2560))},
        ])
        assert [slug for slug, _ in manquantes] == ["livre-rebatir-intimite", "livre-c"]

    def test_la_file_reelle_du_onze_aout_aurait_ete_refusee(self):
        """Le témoin : ``queue/queue.json`` telle qu'elle a tourné. Un seul
        livre y manque de couverture, et c'est celui dont le M4B n'en a
        aucune."""
        books = json.loads((ROOT / "queue" / "queue.json").read_text(encoding="utf-8"))
        assert [slug for slug, _ in narrate_queue.couvertures_manquantes(books)] == [
            "livre-rebatir-intimite"
        ]

    def test_le_catalogue_courant_est_conforme(self):
        books = json.loads(
            (ROOT / "queue" / "queue_catalogue.json").read_text(encoding="utf-8")
        )
        assert narrate_queue.couvertures_manquantes(books) == []

    def test_le_pre_vol_vole_le_meme_plan(self, tmp_path):
        """Le pré-vol relance cette commande avec ``--dry-run``. S'il en
        construisait une autre — sans ``--assemble`` ni ``--cover`` — il
        validerait un livre que la vraie prise assemble autrement, ce qui est
        exactement ce qui a laissé passer le livre sans couverture."""
        args = types.SimpleNamespace(device="cuda", qc_retries="2",
                                     no_synthetic_disclosure=False)
        cmd = narrate_queue.commande(
            {"slug": "livre-a", "voice": "Alex Somerset", "cover": "queue/covers/x.jpg",
             "title": "TITRE", "author": "Auteur"},
            tmp_path / "livre.txt", tmp_path / "out", args,
        )
        assert "--assemble" in cmd and "--export-acx" in cmd
        assert cmd[cmd.index("--cover") + 1] == "queue/covers/x.jpg"
        assert cmd[cmd.index("--title") + 1] == "TITRE"
