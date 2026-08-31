"""Tests for the opening and closing credits.

What is checked is what a distributor checks: that the title, the author and
the narrator are actually said, at both ends, and that a synthetic reading says
so rather than passing for a performance.
"""
import pytest

from narration.credits import CLOSING_TITLE, OPENING_TITLE, SYNTHETIC_DISCLOSURE, BookCredits


class TestOpening:
    def test_it_names_the_work_and_its_author(self):
        opening = BookCredits(title="Autour de la Lune", author="Jules Verne").opening()
        assert "Autour de la Lune" in opening
        assert "Jules Verne" in opening

    def test_a_human_narrator_is_named(self):
        opening = BookCredits(
            title="Le Livre", author="Une Autrice", narrator="Edwin Osayamwen"
        ).opening()
        assert "Lu par Edwin Osayamwen." in opening

    def test_a_subtitle_is_announced(self):
        opening = BookCredits(
            title="Le Livre", subtitle="une histoire vraie", author="Une Autrice"
        ).opening()
        assert "une histoire vraie" in opening

    def test_an_author_already_in_the_title_is_not_said_twice(self):
        opening = BookCredits(
            title="Autour de la Lune, par Jules Verne", author="Jules Verne"
        ).opening()
        assert opening.count("Jules Verne") == 1


class TestClosing:
    def test_it_names_the_work_again(self):
        closing = BookCredits(title="Le Livre", author="Une Autrice").closing()
        assert closing.startswith("Vous venez d'écouter")
        assert "Le Livre" in closing
        assert "Une Autrice" in closing

    def test_production_is_credited_when_given(self):
        closing = BookCredits(
            title="Le Livre", author="Une Autrice", publisher="Studio X", year="2026"
        ).closing()
        assert "Studio X" in closing and "2026" in closing

    def test_a_year_alone_still_reads_as_a_sentence(self):
        closing = BookCredits(title="Le Livre", year="2026").closing()
        assert "réalisé en 2026." in closing

    def test_public_domain_is_stated(self):
        closing = BookCredits(
            title="Autour de la Lune", author="Jules Verne", public_domain=True
        ).closing()
        assert "domaine public" in closing


class TestSyntheticDisclosure:
    """Claiming a machine reading as a human performance is what closes accounts."""

    def test_a_synthetic_reading_says_so_at_both_ends(self):
        credits = BookCredits(title="Le Livre", author="Une Autrice")
        assert SYNTHETIC_DISCLOSURE in credits.opening()
        assert SYNTHETIC_DISCLOSURE in credits.closing()

    def test_a_named_narrator_replaces_the_disclosure(self):
        credits = BookCredits(title="Le Livre", narrator="Edwin")
        assert SYNTHETIC_DISCLOSURE not in credits.opening()
        assert "Edwin" in credits.opening()

    def test_it_is_on_unless_deliberately_turned_off(self):
        assert BookCredits(title="x").disclose_synthetic is True
        silent = BookCredits(title="x", disclose_synthetic=False)
        assert silent.narrator_credit == ""
        assert "Lu par" not in silent.opening()


class TestDistributionReadiness:
    def test_a_complete_set_of_credits_is_ready(self):
        credits = BookCredits(title="Le Livre", author="Une Autrice", narrator="Edwin")
        assert credits.missing_for_distribution() == []

    def test_a_synthetic_reading_counts_as_credited(self):
        """The disclosure *is* the narrator credit."""
        assert BookCredits(title="Le Livre", author="Une Autrice").missing_for_distribution() == []

    def test_what_is_missing_is_named(self):
        missing = BookCredits(title="", author="").missing_for_distribution()
        assert any("titre" in item for item in missing)
        assert any("auteur" in item for item in missing)

    def test_an_undisclosed_synthetic_reading_is_reported_as_uncredited(self):
        credits = BookCredits(title="Le Livre", author="Une Autrice", disclose_synthetic=False)
        assert any("narrateur" in item for item in credits.missing_for_distribution())


class TestShape:
    def test_credits_are_narratable_prose(self):
        """No markup, no lists — this text goes straight to the engine."""
        credits = BookCredits(title="Le Livre", author="Une Autrice", publisher="Studio X")
        for text in (credits.opening(), credits.closing()):
            assert text.strip() == text
            assert "<" not in text and "*" not in text
            for paragraph in text.split("\n\n"):
                assert paragraph.endswith(".")

    def test_a_book_with_no_metadata_still_says_something(self):
        opening = BookCredits(title="").opening()
        assert opening.strip()

    def test_the_two_files_have_stable_names(self):
        assert OPENING_TITLE and CLOSING_TITLE
        assert OPENING_TITLE != CLOSING_TITLE


class TestNamedSyntheticVoice:
    """Nommer la voix ne dispense pas de dire qu'elle est synthétique.

    Un catalogue lu par la même voix mérite qu'on la crédite — les éditeurs le
    font pour leurs voix virtuelles. Mais le nom seul créditerait une
    interprétation qui n'a pas eu lieu, et c'est précisément ce que les
    plateformes exigent d'éviter. Les deux se disent, jamais l'un à la place
    de l'autre.
    """

    def test_the_name_and_the_disclosure_are_both_said(self):
        c = BookCredits(title="Un livre", author="Un auteur", voice_name="Aurore Cabonet")
        assert "Aurore Cabonet" in c.opening()
        assert "voix de synthèse" in c.opening()

    def test_the_closing_credit_says_both_too(self):
        c = BookCredits(title="Un livre", author="Un auteur", voice_name="Gabriel Adam")
        closing = c.closing()
        assert "Gabriel Adam" in closing
        assert "voix de synthèse" in closing

    def test_a_human_narrator_still_wins_and_stands_alone(self):
        c = BookCredits(title="Un livre", narrator="Jean Dupont", voice_name="Gabriel Adam")
        assert "Jean Dupont" in c.opening()
        assert "Gabriel Adam" not in c.opening()
        assert "voix de synthèse" not in c.opening()

    def test_without_a_name_the_disclosure_is_unchanged(self):
        c = BookCredits(title="Un livre")
        assert "voix de synthèse" in c.opening()

    def test_disabling_the_disclosure_leaves_the_name_alone(self):
        # Le désactiver reste un acte délibéré, documenté comme tel.
        c = BookCredits(title="Un livre", voice_name="Gabriel Adam", disclose_synthetic=False)
        assert c.narrator_credit == "Gabriel Adam"
