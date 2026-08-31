"""Le code de sortie doit dire « le livrable existe », pas « rien n'a levé ».

La file s'y fie pour marquer un livre ``done``, et ``done`` ne se rejoue pas :
``state.json`` saute ces livres-là pour toujours. Quatre chemins rendaient
pourtant zéro sans qu'aucun fichier déposable existe — aucun chapitre à
assembler, ffmpeg absent à l'assemblage, ffmpeg absent à l'export, fichiers
ACX hors norme. C'est le même motif que la couverture manquante, et que le
mode ``--lot`` qui parcourait zéro fichier : **le rapport d'un travail réussi
et celui d'un travail jamais fait étaient le même.**

Le motif vaut aussi pour ce qu'on consomme, d'où la dernière classe : un
lexique nommé qui ne charge rien se narrait en silence.
"""
from __future__ import annotations

import importlib.util
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SR = 24000
BASE_SEED = 4242


class StubDemo:
    def __init__(self, **_kwargs) -> None:
        pass

    def generate_tts_audio(self, *, text_input, seed=None, **_kwargs):
        secondes = max(0.5, len((text_input or "").strip()) / 17.0)
        rng = np.random.default_rng(1)
        return SR, rng.normal(0, 0.2, int(SR * secondes)).astype(np.float32), None


app_stub = types.ModuleType("app")
app_stub.PRESET_VOICES = [
    {"name": "Voix de test", "description": "voix française de test", "seed": BASE_SEED}
]
app_stub._PRESET_BY_NAME = {"Voix de test": app_stub.PRESET_VOICES[0]}
app_stub._OUTPUT_DIR = ROOT / "output"
app_stub._sanitize_filename = lambda name: name
app_stub.VoxCPMDemo = StubDemo
sys.modules.setdefault("app", app_stub)


def _charger(nom: str):
    spec = importlib.util.spec_from_file_location(nom, ROOT / "scripts" / f"{nom}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


narrate_book = _charger("narrate_book")
narrate_queue = _charger("narrate_queue")


def _cover(chemin: Path, cote: int = 3000) -> Path:
    """Une couverture conforme : le pré-vol en exige une pour assembler."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, cote, cote, 3) + b"\x00" * 9
    chemin.write_bytes(b"\xff\xd8" + sof + b"\x00" * 30_000)
    return chemin


@pytest.fixture
def livre(tmp_path):
    chemin = tmp_path / "livre.txt"
    chemin.write_text(
        "Chapitre premier. Une phrase de longueur raisonnable pour un segment.\n"
        "---\n"
        "Chapitre second. Une autre phrase, de longueur comparable au premier.\n",
        encoding="utf-8",
    )
    return chemin


def _narrer(monkeypatch, livre, tmp_path, *extra, couverture=True) -> int:
    """Une narration complète, assemblage compris.

    La couverture de ces tests n'a qu'un en-tête : elle suffit au pré-vol, qui
    ne lit que ça, mais pas à ffmpeg, qui la décode. Les cas qui vont jusqu'au
    M4B passent donc ``--no-cover`` — ce qu'ils mesurent est le code de sortie,
    pas l'image.
    """
    cover = (["--cover", str(_cover(tmp_path / "audio.jpg"))] if couverture
             else ["--no-cover"])
    monkeypatch.setattr(sys, "argv", [
        "narrate_book.py", str(livre), "--voice", "Voix de test",
        "--outdir", str(tmp_path / "out"), "--no-credits",
        "--assemble", "m4b", *cover, *extra,
    ])
    return narrate_book.main()


class TestNarrateBook:
    def test_un_livre_assemble_rend_zero(self, monkeypatch, livre, tmp_path):
        assert _narrer(monkeypatch, livre, tmp_path, couverture=False) == 0
        assert list((tmp_path / "out").glob("*_complet.m4b"))

    def test_un_m4b_qui_reste_a_encoder_nest_pas_un_m4b(
        self, monkeypatch, livre, tmp_path, capsys
    ):
        """Sans ffmpeg, l'assemblage écrit la commande à lancer plus tard. Elle
        est utile à qui la lit ; elle ne remplace pas le fichier."""
        monkeypatch.setattr(narrate_book.assembly, "find_ffmpeg", lambda: None)
        code = _narrer(monkeypatch, livre, tmp_path)
        sortie = capsys.readouterr().out
        assert code == 1
        assert "Incomplet" in sortie and "M4B" in sortie

    def test_un_export_acx_en_echec_nest_pas_une_remarque(
        self, monkeypatch, livre, tmp_path, capsys
    ):
        """L'export sort non nul quand des fichiers sont hors norme ou n'ont
        jamais été encodés. C'était imprimé, puis oublié."""
        vrai_run = narrate_book.subprocess.run

        def run_stub(cmd, *args, **kwargs):
            if "export_acx.py" in " ".join(str(c) for c in cmd):
                return types.SimpleNamespace(returncode=1)
            return vrai_run(cmd, *args, **kwargs)

        monkeypatch.setattr(narrate_book.subprocess, "run", run_stub)
        code = _narrer(monkeypatch, livre, tmp_path, "--export-acx", couverture=False)
        sortie = capsys.readouterr().out
        assert code == 1
        # Le M4B, lui, a bien été écrit : la cause nommée est la bonne.
        derniere = sortie.strip().splitlines()[-1]
        assert derniere == "Incomplet : il manque un export ACX déposable."

    def test_rien_a_assembler_nest_pas_un_succes(self, monkeypatch, livre, tmp_path, capsys):
        """Le cas qui a coûté un volume saturé : lancé depuis le mauvais
        dossier, le script ne trouvait aucun chapitre, sortait zéro, et la
        purge des WAV — conditionnée à ce compte — ne se déclenchait pas."""
        monkeypatch.setattr(
            narrate_book.Path, "glob",
            lambda self, motif: iter(()) if motif == "chapitre_*.wav"
            else Path.glob(self, motif),
        )
        code = _narrer(monkeypatch, livre, tmp_path)
        assert code == 1
        assert "Rien à assembler" in capsys.readouterr().out


class TestLivrablesDeLaFile:
    """La file regarde les fichiers, pas le code de sortie de qui les fabrique."""

    def _livre_complet(self, d: Path) -> Path:
        d.mkdir(parents=True, exist_ok=True)
        (d / "chapitre_001.wav").write_bytes(b"RIFF")
        (d / f"{d.name}_complet.m4b").write_bytes(b"\x00" * 10)
        (d / "acx").mkdir()
        (d / "acx" / "001 - Chapitre.mp3").write_bytes(b"\x00" * 10)
        return d

    def test_un_livre_complet_ne_manque_de_rien(self, tmp_path):
        d = self._livre_complet(tmp_path / "book_livre_x")
        assert narrate_queue.livrables_absents(d, wavs=1) == []

    def test_sans_m4b(self, tmp_path):
        d = self._livre_complet(tmp_path / "book_livre_x")
        next(d.glob("*_complet.m4b")).unlink()
        assert narrate_queue.livrables_absents(d, wavs=1) == ["M4B"]

    def test_un_dossier_acx_sans_mp3_ne_compte_pas(self, tmp_path):
        """Sans ffmpeg, l'export laisse des WAV et un `encoder.txt` : de quoi
        encoder, pas de quoi déposer. Tester l'existence du dossier aurait
        accepté ça."""
        d = self._livre_complet(tmp_path / "book_livre_x")
        next((d / "acx").glob("*.mp3")).unlink()
        (d / "acx" / "001 - Chapitre.wav").write_bytes(b"RIFF")
        (d / "acx" / "encoder.txt").write_text("ffmpeg ...", encoding="utf-8")
        assert narrate_queue.livrables_absents(d, wavs=1) == ["export ACX (MP3)"]

    def test_zero_chapitre_narre(self, tmp_path):
        d = self._livre_complet(tmp_path / "book_livre_x")
        assert "chapitres narrés" in narrate_queue.livrables_absents(d, wavs=0)

    def test_un_dossier_absent_manque_de_tout(self, tmp_path):
        absents = narrate_queue.livrables_absents(tmp_path / "jamais_cree", wavs=0)
        assert len(absents) == 3


class TestLexique:
    """`load_lexicon` est tolérant par dessein : un fichier d'appoint mal formé
    ne doit pas tuer neuf heures de narration. Mais la file passe désormais le
    lexique commun à *chaque* livre, et celui-ci porte `ce` → `çe` — 12 498
    occurrences dans le catalogue, validé à l'oreille. Un chemin faux et trois
    cents livres se narrent avec la mauvaise prononciation du mot le plus
    fréquent. La tolérance reste ; c'est le silence qui s'en va."""

    def _prevol(self, monkeypatch, livre, tmp_path, *extra) -> int:
        monkeypatch.setattr(sys, "argv", [
            "narrate_book.py", str(livre), "--voice", "Voix de test",
            "--outdir", str(tmp_path / "out"), "--no-credits", "--dry-run", *extra,
        ])
        return narrate_book.main()

    def test_un_lexique_nomme_et_introuvable_arrete_tout(
        self, monkeypatch, livre, tmp_path, capsys
    ):
        code = self._prevol(monkeypatch, livre, tmp_path,
                            "--lexicon", str(tmp_path / "jamais_ecrit.json"))
        assert code == 2
        assert "REFUS" in capsys.readouterr().out

    def test_un_lexique_nomme_et_illisible_aussi(self, monkeypatch, livre, tmp_path, capsys):
        """Une virgule en trop suffit : le fichier existe, il ne dit rien."""
        casse = tmp_path / "lexique.json"
        casse.write_text('{"ce": "çe",}', encoding="utf-8")
        assert self._prevol(monkeypatch, livre, tmp_path, "--lexicon", str(casse)) == 2
        assert "illisible" in capsys.readouterr().out

    def test_un_lexique_qui_charge_passe(self, monkeypatch, livre, tmp_path, capsys):
        bon = tmp_path / "lexique.json"
        bon.write_text('{"ce": "çe", "ACE": "A C E"}', encoding="utf-8")
        assert self._prevol(monkeypatch, livre, tmp_path, "--lexicon", str(bon)) == 0
        assert "2 entrée(s) de lexique" in capsys.readouterr().out

    def test_le_lexique_commun_du_depot_charge(self, monkeypatch, livre, tmp_path):
        """Le témoin : c'est ce fichier que la file passe à chaque livre."""
        assert self._prevol(monkeypatch, livre, tmp_path,
                            "--lexicon", "conf/pronunciation_fr.json") == 0

    def test_le_defaut_implicite_manquant_est_dit_sans_refuser(
        self, monkeypatch, livre, tmp_path, capsys
    ):
        """Narré à la main depuis un autre dossier, le chemin par défaut ne
        résout pas. Ce n'est pas une faute de frappe, donc pas un refus — mais
        la ligne le dit, au lieu de laisser croire à une préparation."""
        monkeypatch.chdir(tmp_path)
        code = self._prevol(monkeypatch, livre, tmp_path)
        assert code == 0
        assert "aucune prononciation n'est appliquée" in capsys.readouterr().out
