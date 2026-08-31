"""Tests for segmentation and the pause plan."""
import pytest

from narration.chunking import (
    PauseProfile,
    split_chapters,
    split_into_segments,
    split_text_into_chunks,
)


class TestSplitTextIntoChunks:
    def test_empty(self):
        assert split_text_into_chunks("") == []
        assert split_text_into_chunks("   ") == []

    def test_short_text_is_one_chunk(self):
        assert split_text_into_chunks("Bonjour le monde.") == ["Bonjour le monde."]

    def test_sentences_are_packed_up_to_the_limit(self):
        text = "Un. Deux. Trois. Quatre."
        chunks = split_text_into_chunks(text, max_chars=12)
        assert all(len(c) <= 12 for c in chunks)
        assert " ".join(chunks) == text

    def test_an_overlong_sentence_is_cut_rather_than_sent_whole(self):
        # Cette règle a été inversée le 2026-08-08, mesures à l'appui. Le
        # raisonnement d'origine — une coupe au milieu d'une proposition
        # s'entend plus qu'un segment un peu long — supposait que le moteur
        # lise le segment long en entier. Il ne le fait pas : il le tronque.
        # Sur « Le Lundi de Trop », le seul segment de 679 caractères est
        # revenu en 16,2 s au lieu des 34 s nécessaires, moitié de phrase
        # perdue. Mieux vaut une virgule devenue respiration.
        long_sentence = "mot " * 200
        chunks = split_text_into_chunks(long_sentence.strip(), max_chars=50)
        assert len(chunks) > 1
        assert all(len(c) <= 50 for c in chunks)
        assert " ".join(chunks).split() == long_sentence.split()

    def test_every_word_survives(self):
        text = "Première phrase ici. Deuxième phrase là. Troisième enfin."
        assert " ".join(split_text_into_chunks(text, max_chars=25)) == text


class TestSplitIntoSegments:
    def test_pause_is_longer_after_a_paragraph_than_after_a_sentence(self):
        segments = split_into_segments("Une phrase. Une autre.\n\nNouveau paragraphe.", max_chars=15)
        pauses = [s.pause_after for s in segments]
        profile = PauseProfile()
        assert profile.paragraph in pauses
        assert profile.sentence in pauses
        assert max(pauses) == profile.paragraph

    def test_a_split_that_lands_mid_sentence_gets_the_shortest_pause(self):
        # A line break inside a paragraph (verse, dialogue, an address) is a
        # split point that is not a sentence end.
        segments = split_into_segments("Première ligne\nDeuxième ligne.", max_chars=20)
        assert [s.text for s in segments] == ["Première ligne", "Deuxième ligne."]
        assert segments[0].pause_after == PauseProfile().clause

    def test_segments_never_span_a_paragraph(self):
        segments = split_into_segments("Court.\n\nAussi court.", max_chars=500)
        assert [s.text for s in segments] == ["Court.", "Aussi court."]
        assert [s.paragraph for s in segments] == [0, 1]

    @pytest.mark.parametrize("ending", ["Vraiment?", "Incroyable!", "Et alors…"])
    def test_question_and_exclamation_end_sentences(self, ending):
        # A second sentence keeps the first one away from the paragraph end,
        # where the longer paragraph pause would apply instead.
        segments = split_into_segments(f"{ending} Puis il partit.", max_chars=12)
        assert segments[0].pause_after == PauseProfile().sentence

    def test_closing_quote_after_the_full_stop_still_ends_the_sentence(self):
        segments = split_into_segments('Il dit "oui." Puis il partit.', max_chars=14)
        assert segments[0].pause_after == PauseProfile().sentence

    def test_the_last_segment_of_a_paragraph_gets_the_paragraph_pause(self):
        segments = split_into_segments("Une phrase.")
        assert segments[0].pause_after == PauseProfile().paragraph

    def test_custom_profile_is_honoured(self):
        profile = PauseProfile(clause=0.1, sentence=0.2, paragraph=0.3)
        segments = split_into_segments("Une phrase.\n\nUne autre.", profile=profile)
        assert segments[0].pause_after == 0.3

    def test_empty_text(self):
        assert split_into_segments("") == []


class TestSplitChapters:
    def test_default_separator(self):
        assert split_chapters("Un\n\n---\n\nDeux") == ["Un", "Deux"]

    def test_text_without_a_separator_is_a_single_chapter(self):
        assert split_chapters("Un seul chapitre.") == ["Un seul chapitre."]

    def test_empty_text_yields_nothing(self):
        assert split_chapters("") == []

    def test_blank_chapters_are_dropped(self):
        assert split_chapters("Un\n---\n\n\n---\nDeux") == ["Un", "Deux"]

    def test_custom_pattern(self):
        chapters = split_chapters("A\nCHAPITRE\nB", pattern=r"(?m)^CHAPITRE$")
        assert chapters == ["A", "B"]

    @pytest.mark.parametrize("separator", ["---", "  ---  ", "---   "])
    def test_separator_tolerates_surrounding_whitespace(self, separator):
        assert split_chapters(f"Un\n{separator}\nDeux") == ["Un", "Deux"]


class TestFragmentsAreAbsorbed:
    """Un fragment de deux caractères ne doit jamais partir seul au moteur.

    Mesuré sur « Rebâtir l'Intimité Après Divorce » : un chapitre EPUB finissait
    sur ``-e``, reliquat d'un mot coupé à la frontière de fichier. Le moteur,
    devant deux caractères, a produit 0,6 s de babil là où 0,1 s était attendue,
    que le contrôle qualité a classé fatal. Régénérer n'y change rien — le
    second essai a donné 1,6 s de babil au lieu de 0,6.
    """

    def test_trailing_debris_joins_the_sentence_before_it(self):
        segments = split_into_segments("Nous entrerons dans celui du corps.\n\n-e")
        assert len(segments) == 1
        assert segments[0].text.endswith("-e")

    def test_leading_debris_joins_the_sentence_after_it(self):
        segments = split_into_segments("»\n\nElle entra sans frapper.")
        assert len(segments) == 1
        assert segments[0].text.startswith("»")

    def test_a_short_real_sentence_survives(self):
        segments = split_into_segments("Oui.\n\nElle répondit enfin.")
        assert [s.text for s in segments] == ["Oui.", "Elle répondit enfin."]

    def test_a_lone_fragment_is_kept_rather_than_lost(self):
        # Rien à quoi le rattacher : mieux vaut un défaut signalé qu'un texte
        # silencieusement supprimé du livre.
        assert [s.text for s in split_into_segments("-e")] == ["-e"]

    def test_the_pause_of_the_absorbed_tail_is_the_one_kept(self):
        segments = split_into_segments("Une phrase complète ici.\n\n-e")
        assert segments[0].pause_after > 0


class TestOverlongSentencesAreCut:
    """Une phrase trop longue est tronquée par le moteur, pas lue lentement.

    Mesuré sur « Le Lundi de Trop » : sous 300 caractères le taux de défaut est
    de 4 à 8 %, il passe à 20 % entre 300 et 400, et le seul segment de 679
    caractères est revenu en 16,2 s là où 34 s étaient nécessaires — la moitié
    de la phrase manquait. Une virgule devenue respiration coûte moins cher.
    """

    def test_a_long_sentence_is_cut_at_its_commas(self):
        sentence = ("Elle avança dans le couloir, " * 12).strip().rstrip(",") + "."
        chunks = split_text_into_chunks(sentence, 300)
        assert len(chunks) > 1
        assert all(len(c) <= 300 for c in chunks)

    def test_every_word_survives_the_cut(self):
        sentence = ("un mot de plus, " * 40).strip().rstrip(",") + "."
        chunks = split_text_into_chunks(sentence, 200)
        assert " ".join(chunks).split() == sentence.split()

    def test_a_sentence_without_any_boundary_falls_back_to_words(self):
        sentence = "mot " * 200
        chunks = split_text_into_chunks(sentence.strip(), 150)
        assert all(len(c) <= 150 for c in chunks)
        assert " ".join(chunks).split() == sentence.split()

    def test_a_short_sentence_is_left_alone(self):
        assert split_text_into_chunks("Elle entra.", 300) == ["Elle entra."]

    def test_semicolons_are_preferred_over_commas(self):
        left = "a, " * 40
        sentence = f"{left}; {left}".strip()
        chunks = split_text_into_chunks(sentence, 300)
        assert all(len(c) <= 300 for c in chunks)
