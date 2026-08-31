"""Tests de scripts/prepare_manuscript.py.

Le cas qui a motivé ce fichier : un corpus de vingt livres portait 993
marqueurs entre crochets, dont « [PAUSE] » 969 fois. Envoyés au moteur tels
quels, ils se lisent à voix haute. Aucun n'avait encore été narré — d'où
l'urgence de figer le comportement avant qu'ils ne le soient.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "prepare_manuscript", ROOT / "scripts" / "prepare_manuscript.py"
)
prepare_manuscript = importlib.util.module_from_spec(spec)
sys.modules["prepare_manuscript"] = prepare_manuscript
spec.loader.exec_module(prepare_manuscript)


class TestBrackets:
    """Trois natures de crochets, trois traitements — ils ne sont pas une chose."""

    def test_a_pause_becomes_a_silence_not_a_word(self):
        out, _ = prepare_manuscript.unbracket("Il se tut. [PAUSE] Puis reprit.")
        assert "PAUSE" not in out
        assert "\n\n" in out

    @pytest.mark.parametrize("marqueur", ["[rire]", "[hésitation]", "[silence prolongé]",
                                          "[pleurs contenus]", "[soupir]"])
    def test_stage_directions_are_removed(self, marqueur):
        out, _ = prepare_manuscript.unbracket(f"Elle parla. {marqueur} Puis se tut.")
        assert "[" not in out and "]" not in out
        assert marqueur.strip("[]") not in out

    @pytest.mark.parametrize("champ", ["[nom du département]", "[ton mari / ta femme]", "[date]"])
    def test_a_blank_to_fill_keeps_its_words(self, champ):
        # Supprimer ceux-là laisserait un trou à la place du sens.
        out, _ = prepare_manuscript.unbracket(f"Écrivez {champ} ici.")
        assert "[" not in out
        assert champ.strip("[]") in out

    def test_everything_removed_is_reported(self):
        _, removed = prepare_manuscript.unbracket("A [PAUSE] B [rire] C [date] D")
        assert len(removed) == 3
        assert any(r.startswith("pause") for r in removed)
        assert any(r.startswith("didascalie") for r in removed)

    def test_text_without_brackets_is_untouched(self):
        texte = "Une phrase parfaitement ordinaire, sans rien de particulier."
        out, removed = prepare_manuscript.unbracket(texte)
        assert out == texte
        assert removed == []


class TestMarkdownIsNotSpoken:
    def test_emphasis_and_headings_leave_no_trace(self):
        assert prepare_manuscript.strip_inline("**gras** et *italique*") == "gras et italique"

    def test_a_link_keeps_its_words_and_loses_its_target(self):
        assert prepare_manuscript.strip_inline("voir [le site](https://x.fr)") == "voir le site"

    def test_an_image_says_nothing(self):
        assert prepare_manuscript.strip_inline("![couverture](img.png)") == ""


class TestAccentsRestored:
    """Le moteur lit ce qui est écrit, y compris les fautes du manuscrit.

    Edwin a signalé « degré » prononcé « degre ». Ce n'était pas la synthèse :
    le manuscrit écrit « degres » sans accent, et le moteur avait raison. Vingt-
    deux fichiers sur vingt-quatre portent ce défaut — 31 « maniere », 17
    « difference », 7 « degre(s) ».
    """

    @pytest.mark.parametrize("faute,juste", [
        ("degres", "degrés"), ("maniere", "manière"), ("difference", "différence"),
        ("plongee", "plongée"), ("societe", "société"), ("dedicace", "dédicace"),
    ])
    def test_a_lost_accent_comes_back(self, faute, juste):
        out, notes = prepare_manuscript.restore_accents(f"une {faute} ici")
        assert juste in out
        assert notes

    def test_capitalisation_survives(self):
        out, _ = prepare_manuscript.restore_accents("Dedicace au lecteur")
        assert out.startswith("Dédicace")

    @pytest.mark.parametrize("ambigu", ["cote", "tache", "sur", "mure", "pecheur"])
    def test_words_with_two_readings_are_left_alone(self, ambigu):
        # « la cote atlantique » et « la côte » sont deux mots : corriger à
        # l'aveugle remplacerait une faute par une autre.
        texte = f"voici le mot {ambigu} dans sa phrase"
        out, notes = prepare_manuscript.restore_accents(texte)
        assert out == texte
        assert notes == []

    def test_a_clean_text_is_untouched(self):
        texte = "Un texte déjà correctement accentué, avec ses différences."
        out, notes = prepare_manuscript.restore_accents(texte)
        assert out == texte and notes == []
