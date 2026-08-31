"""Tests for the French text preprocessor.

Number agreement and the Roman-numeral rules carry most of the risk here: they
are the parts that silently produce a *plausible but wrong* reading, which is
exactly what nobody notices until they hear it in a finished chapter.
"""
import json

import pytest

from narration.text_fr import (
    cardinal,
    load_lexicon,
    normalize_french,
    ordinal,
    roman_to_int,
)


class TestCardinal:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0, "zéro"),
            (1, "un"),
            (16, "seize"),
            (17, "dix-sept"),
            (20, "vingt"),
            (21, "vingt et un"),
            (22, "vingt-deux"),
            (31, "trente et un"),
            (70, "soixante-dix"),
            (71, "soixante et onze"),
            (72, "soixante-douze"),
            (79, "soixante-dix-neuf"),
            (90, "quatre-vingt-dix"),
            (91, "quatre-vingt-onze"),
            (99, "quatre-vingt-dix-neuf"),
        ],
    )
    def test_the_awkward_tens(self, value, expected):
        assert cardinal(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (80, "quatre-vingts"),
            (81, "quatre-vingt-un"),
            (100, "cent"),
            (101, "cent un"),
            (180, "cent quatre-vingts"),
            (200, "deux cents"),
            (201, "deux cent un"),
            (280, "deux cent quatre-vingts"),
        ],
    )
    def test_plural_agreement_when_the_number_ends(self, value, expected):
        assert cardinal(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (200_000, "deux cent mille"),      # not "deux cents mille"
            (80_000, "quatre-vingt mille"),    # not "quatre-vingts mille"
            (200_000_000, "deux cents millions"),  # million is a noun: agreement returns
        ],
    )
    def test_agreement_is_suppressed_before_another_numeral(self, value, expected):
        assert cardinal(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (1000, "mille"),
            (1001, "mille un"),
            (1234, "mille deux cent trente-quatre"),
            (1789, "mille sept cent quatre-vingt-neuf"),
            (1_000_000, "un million"),
            (2_000_000, "deux millions"),
            (1_000_000_000, "un milliard"),
        ],
    )
    def test_scales(self, value, expected):
        assert cardinal(value) == expected

    def test_mille_never_takes_un_or_an_s(self):
        assert cardinal(1000) == "mille"
        assert "un mille" not in cardinal(1500)

    def test_negative(self):
        assert cardinal(-5) == "moins cinq"


class TestOrdinal:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1, "premier"),
            (2, "deuxième"),
            (4, "quatrième"),
            (5, "cinquième"),
            (9, "neuvième"),
            (11, "onzième"),
            (20, "vingtième"),
            (21, "vingt et unième"),
            (80, "quatre-vingtième"),
            (100, "centième"),
            (1000, "millième"),
        ],
    )
    def test_ordinals(self, value, expected):
        assert ordinal(value) == expected

    def test_feminine_first(self):
        assert ordinal(1, feminine=True) == "première"


class TestRoman:
    @pytest.mark.parametrize("text,value", [("XIV", 14), ("MCMXCIV", 1994), ("IX", 9), ("i", 1)])
    def test_valid(self, text, value):
        assert roman_to_int(text) == value

    @pytest.mark.parametrize("text", ["IIII", "VV", "", "  ", "ABC", "XIIX"])
    def test_malformed_is_rejected(self, text):
        assert roman_to_int(text) is None


class TestNormalizeFrench:
    def test_empty(self):
        assert normalize_french("") == ""
        assert normalize_french("   \n  ") == ""

    def test_numbers_are_spelled_out(self):
        assert "mille sept cent quatre-vingt-neuf" in normalize_french("En 1789, tout bascula.")

    def test_thousands_separator_does_not_swallow_the_following_space(self):
        # A greedy digits-and-spaces pattern used to match "12 " and glue the
        # spelled-out number to the next word ("douzeet").
        assert normalize_french("Il y a 12 et 3 000 raisons.") == (
            "Il y a douze et trois mille raisons."
        )

    def test_common_words_are_not_roman_numerals(self):
        # "Le" is L + e; without a guard it reads as the 50th.
        for word in ("Le", "Ce", "De", "Me"):
            result = normalize_french(f"{word} manuscrit")
            assert result == f"{word} manuscrit", result

    def test_genuine_roman_ordinals_still_expand(self):
        assert "dix-neuvième siècle" in normalize_french("au XIXe siècle")
        assert "cinquième République" in normalize_french("la Ve République")
        assert "vingtième" in normalize_french("le XXème anniversaire")

    def test_roman_numeral_after_a_trigger_word(self):
        assert "chapitre quatorze" in normalize_french("Voir chapitre XIV pour la suite.")

    def test_a_spelled_out_number_after_a_trigger_is_left_alone(self):
        # "dix" parses as the Roman numeral DIX (509) under a case-insensitive
        # match — the numeral side must stay case-sensitive.
        assert normalize_french("Au chapitre dix, rien.") == "Au chapitre dix, rien."

    def test_roman_numeral_alone_on_a_line_is_a_heading(self):
        assert normalize_french("XIV\n\nIl arriva.").startswith("quatorze")

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("M. Dupont arriva", "Monsieur Dupont arriva"),
            ("Mme Leblanc", "Madame Leblanc"),
            ("le Dr Martin", "le Docteur Martin"),
            ("de Me Durand", "de Maître Durand"),
            ("MM. Dupont et Durand", "Messieurs Dupont et Durand"),
        ],
    )
    def test_abbreviations(self, source, expected):
        assert normalize_french(source) == expected

    def test_etc_keeps_its_sentence_ending_period(self):
        # Consuming the period would merge two sentences and lose the pause.
        assert normalize_french("des pommes, etc. Il partit.") == (
            "des pommes, et cetera. Il partit."
        )

    def test_before_christ_keeps_its_period(self):
        assert normalize_french("En 52 av. J.-C. Les Romains.") == (
            "En cinquante-deux avant Jésus-Christ. Les Romains."
        )

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("14h30", "quatorze heures trente"),
            ("2h", "deux heures"),
            ("1h", "une heure"),
            ("8h00", "huit heures"),
        ],
    )
    def test_times(self, source, expected):
        assert normalize_french(source) == expected

    def test_an_hour_like_number_is_not_a_time(self):
        assert normalize_french("il y a 2 hommes") == "il y a deux hommes"

    @pytest.mark.parametrize(
        "source,expected",
        [
            # The space after a whole hour used to be eaten with the minutes
            # that were not there: "neuf heuresdu matin".
            ("Il est 9h du matin.", "Il est neuf heures du matin."),
            ("Vers 20h il rentre.", "Vers vingt heures il rentre."),
            ("de 9h à 17h", "de neuf heures à dix-sept heures"),
            # Punctuation hid the defect, and must keep working.
            ("À 8h, il partit.", "À huit heures, il partit."),
            ("Il est 9h.", "Il est neuf heures."),
            # Minutes still read, spaced or not.
            ("à 14 h 30", "à quatorze heures trente"),
            ("10h00 pile", "dix heures pile"),
        ],
    )
    def test_a_whole_hour_keeps_the_space_after_it(self, source, expected):
        assert normalize_french(source) == expected

    @pytest.mark.parametrize("source", ["un champ de 35ha", "9h305 n'est pas une heure"])
    def test_what_only_looks_like_an_hour_is_left_alone(self, source):
        """A letter or a digit right after the h means it was never a time."""
        assert normalize_french(source) == source

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("1 250 €", "mille deux cent cinquante euros"),
            ("1 €", "un euro"),
            ("3,50 €", "trois euros cinquante centimes"),
            ("$5", "cinq dollars"),
        ],
    )
    def test_currency(self, source, expected):
        assert normalize_french(source) == expected

    def test_percentages(self):
        assert normalize_french("3,5 %") == "trois virgule cinq pour cent"
        assert normalize_french("50 %") == "cinquante pour cent"

    def test_ordinal_marks(self):
        assert normalize_french("la 1re fois") == "la première fois"
        assert normalize_french("le 1er jour") == "le premier jour"
        assert normalize_french("la 2e chance") == "la deuxième chance"
        assert normalize_french("les 3es places") == "les troisièmes places"

    def test_decimals(self):
        assert normalize_french("3,5 litres") == "trois virgule cinq litres"

    def test_dialogue_dash_is_removed_and_guillemets_dropped(self):
        assert normalize_french("— Bonjour, dit-il.") == "Bonjour, dit-il."
        assert normalize_french("Il dit « bonjour ».") == "Il dit bonjour."

    def test_markdown_is_stripped(self):
        assert normalize_french("## Titre\n\nUn **mot** important.") == (
            "Titre\n\nUn mot important."
        )

    def test_french_spacing_before_punctuation_is_removed(self):
        assert normalize_french("Vraiment ?") == "Vraiment?"

    def test_lexicon_overrides_are_applied(self):
        result = normalize_french("La SNCF annonce", lexicon={"SNCF": "S N C F"})
        assert result == "La S N C F annonce"

    def test_lexicon_matches_whole_words_only(self):
        assert normalize_french("chat chatte", lexicon={"chat": "minou"}) == "minou chatte"

    def test_roman_expansion_can_be_disabled(self):
        assert "XIV" in normalize_french("chapitre XIV", expand_roman=False)


class TestContextualLexicon:
    """A homograph cannot be fixed by a rule that fires on the word alone."""

    EAST = {"est": {"prononcer": "èsste", "après": "à l'|dans l'|vers l'|l'"}}

    def test_it_fires_in_context(self):
        assert "èsste" in normalize_french("Le vent vient de l'est.", lexicon=self.EAST)

    def test_it_leaves_the_other_word_alone(self):
        """`il est` must survive a rule aimed at `à l'est`."""
        out = normalize_french("Il est tard et elle est partie.", lexicon=self.EAST)
        assert "èsste" not in out
        assert "est tard" in out

    def test_the_context_itself_is_kept(self):
        out = normalize_french("Il regarde vers l'est.", lexicon=self.EAST)
        assert "vers l'" in out

    def test_a_following_context_works_too(self):
        lexicon = {"plus": {"prononcer": "pluss", "avant": "de|que"}}
        out = normalize_french("Il y en a plus de dix, plus tard.", lexicon=lexicon)
        assert "pluss de dix" in out
        assert "plus tard" in out

    def test_a_plain_string_entry_still_works(self):
        assert "S N C F" in normalize_french("La SNCF.", lexicon={"SNCF": "S N C F"})

    def test_case_does_not_matter(self):
        assert "èsste" in normalize_french("À L'EST, la mer.", lexicon=self.EAST)

    def test_the_capital_of_a_sentence_start_is_kept(self):
        """« Ce » en tête de phrase doit devenir « Çe », pas « ce » ni « Ce ».

        C'est le cas le plus lourd du lexique et rien ne le testait. Mesuré sur
        dix phrases réelles du corpus, voix Alex Somerset, relues par Whisper :
        « Ce » brut est mal dit **10 fois sur 10** — le moteur épelle les
        lettres, l'ASR écrit « Point C E » ou « Ces E- ». Avec la cédille,
        10/10 sont justes, et 0 des 27 segments de production contenant « Çe »
        n'était fautif. La correction ne vaut donc que si elle survit à la
        majuscule : une substitution qui rendrait « ce » en minuscule, ou qui
        laisserait « Ce » intact, ramène un défaut audible toutes les deux
        minutes — 2 989 des 12 498 « ce » du corpus ouvrent une phrase.
        """
        assert normalize_french("Ce livre.", lexicon={"ce": "çe"}) == "Çe livre."

    def test_the_capital_is_kept_after_a_full_stop_too(self):
        out = normalize_french("Voici ce livre. Ce chapitre parle.", lexicon={"ce": "çe"})
        assert out == "Voici çe livre. Çe chapitre parle."

    def test_a_capital_is_not_invented_where_there_was_none(self):
        assert normalize_french("dans ce livre", lexicon={"ce": "çe"}) == "dans çe livre"

    def test_a_malformed_entry_is_ignored_not_fatal(self):
        lexicon = {"est": {"pas_la_bonne_clef": "x"}, "SNCF": "S N C F"}
        out = normalize_french("La SNCF est là.", lexicon=lexicon)
        assert "S N C F" in out and "est là" in out

    def test_a_replacement_containing_a_backslash_is_literal(self):
        out = normalize_french("Voir SNCF.", lexicon={"SNCF": r"S\N"})
        assert r"S\N" in out


class TestLoadLexicon:
    def test_missing_file_yields_empty(self, tmp_path):
        assert load_lexicon(tmp_path / "nope.json") == {}

    def test_malformed_file_yields_empty_rather_than_raising(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{ not json", encoding="utf-8")
        assert load_lexicon(path) == {}

    def test_comment_keys_are_ignored(self, tmp_path):
        path = tmp_path / "lex.json"
        path.write_text(
            json.dumps({"_comment": "note", "SNCF": "S N C F"}, ensure_ascii=False),
            encoding="utf-8",
        )
        assert load_lexicon(path) == {"SNCF": "S N C F"}

    def test_a_json_list_is_not_a_lexicon(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2]", encoding="utf-8")
        assert load_lexicon(path) == {}


class TestSuperscriptLetters:
    """« 5ᵉ » n'est pas « 5e », et cette différence a tué une narration.

    Un traitement de texte produit une lettre modificative en exposant (U+1D49)
    qui ressemble à un « e » sans en être un. La règle des ordinaux ne la voit
    pas, le fragment traverse la normalisation intact, et le normaliseur interne
    du moteur meurt dessus — assert len(input) > 0 — après quarante et une
    minutes de narration. Dix-huit occurrences dans neuf fichiers de la file.
    """

    def test_a_superscript_ordinal_is_spoken(self):
        assert normalize_french("la 5ᵉ édition") == "la cinquième édition"

    def test_a_two_digit_superscript_ordinal(self):
        assert normalize_french("la 11ᵉ édition") == "la onzième édition"

    def test_a_feminine_first(self):
        assert normalize_french("la 1ʳᵉ fois") == "la première fois"

    def test_the_plain_form_still_works(self):
        assert normalize_french("la 5e édition") == "la cinquième édition"


class TestSymbolsAManuscriptKeeps:
    """Un manuscrit n'est pas que de la prose.

    Relevé sur les vingt et un livres de la file : 571 lignes à remplir, 163
    appels de note, 107 points médians, 55 commandes LaTeX, 47 degrés, 26
    esperluettes, 21 chemins d'interface, 20 flèches, 8 cases à cocher. Chacun
    se lit à voix haute, ou pire : « \newpage » perdait sa barre oblique au
    nettoyage markdown et devenait « ewpage », prononcé tel quel.
    """

    def test_a_word_processor_command_leaves_nothing_behind(self):
        # La barre oblique doit être littérale : écrite « \n », elle devient un
        # saut de ligne et le test reproduit le défaut qu'il vérifie.
        assert "ewpage" not in normalize_french("gratuits. " + chr(92) + "newpage Voici")

    def test_degrees_celsius_are_spoken(self):
        assert normalize_french("réglé à 18,5 °C") == "réglé à dix-huit virgule cinq degrés Celsius"

    def test_the_celsius_rule_wins_over_the_general_one(self):
        # Sans l'ordre, « 18,5 °C » deviendrait « 18,5 degrésC ».
        assert "degrésC" not in normalize_french("réglé à 18,5 °C")

    def test_an_ampersand_becomes_a_word(self):
        assert normalize_french("Sparrow, Liu & Wegner") == "Sparrow, Liu et Wegner"

    def test_inclusive_writing_is_read_in_full(self):
        assert normalize_french("votre conjoint·e") == "votre conjoint ou conjointe"

    @pytest.mark.parametrize("source,attendu", [
        ("Réglages > Temps", "Réglages puis Temps"),
        ("Tête → visage", "Tête puis visage"),
    ])
    def test_arrows_and_interface_paths_become_puis(self, source, attendu):
        assert normalize_french(source) == attendu

    def test_form_leftovers_are_removed(self):
        assert "_" not in normalize_french("Date: ___ fin: ___")
        assert "☐" not in normalize_french("☐ Je consulte")

    def test_a_footnote_marker_is_not_spoken(self):
        assert normalize_french("Jamie*, trente ans") == "Jamie, trente ans"

    def test_ordinary_prose_is_untouched(self):
        texte = "Une phrase parfaitement ordinaire, sans aucun signe particulier."
        assert normalize_french(texte) == texte


class TestParentheses:
    """Une parenthèse ne s'entend pas — et une longue énumération entre
    parenthèses est ce qui a tronqué 86 % des segments défectueux."""

    def test_an_enumeration_becomes_an_apposition(self):
        assert normalize_french(
            "Les approches alternatives (keynésienne, marxiste) existent."
        ) == "Les approches alternatives, keynésienne, marxiste, existent."

    def test_no_comma_is_left_against_the_full_stop(self):
        # « …aux médias. » et non « …aux médias, . »
        rendu = normalize_french("Ils ont des moyens (financements, accès aux médias).")
        assert rendu == "Ils ont des moyens, financements, accès aux médias."

    def test_a_short_aside_is_flattened_too(self):
        # L'exemption des incises courtes a été mesurée fausse : sur un livre
        # narré avec l'aplatissement en place, les huit segments tronqués
        # portaient tous une parenthèse, et tous une parenthèse courte.
        assert normalize_french("le rapport (2008)") == "le rapport, deux mille huit"
        assert normalize_french("sur France Culture (Paris)") == "sur France Culture, Paris"

    def test_the_sigla_that_truncated_a_chapter(self):
        # Mesuré : 236 caractères sortis en 6,9 s, la transcription s'arrêtant
        # à « Le sommeil paradoxal » — 22 % du texte.
        rendu = normalize_french(
            "Le sommeil paradoxal (REM) représente cinquante pour cent du temps."
        )
        assert "(" not in rendu
        assert rendu == "Le sommeil paradoxal, REM, représente cinquante pour cent du temps."

    def test_a_long_aside_without_a_comma_is_flattened_too(self):
        rendu = normalize_french(
            "un effet rebond (la consommation augmente avec l'efficacité) connu"
        )
        assert "(" not in rendu
        assert "la consommation augmente" in rendu

    def test_the_parenthesis_characters_never_reach_the_model(self):
        rendu = normalize_french("des ressources (a, b, c) et (d, e) ailleurs")
        assert "(" not in rendu and ")" not in rendu

    def test_an_empty_parenthesis_disappears(self):
        assert "(" not in normalize_french("un mot () suivant")


class TestFormBlanks:
    """Une ligne à remplir se lit des yeux et ne se dit pas.

    Effacer le seul trait laissait un résidu que le moteur a narré tel quel :
    « Jour 5 : minutes (objectif : 10 min) / Ressenti : ». L'audit l'a relu en
    « Jour 5, minute objectif, 10 mines, essenci » — sept fois de suite, dans
    un livre déjà livré.
    """

    def test_the_label_survives_its_blank(self):
        assert normalize_french(
            "Application la plus consultée : ___________________________"
        ) == "Application la plus consultée."

    def test_orphaned_units_go_with_the_blank(self):
        # « heures » et « minutes » n'énoncent plus rien sans leur grandeur.
        assert normalize_french(
            "Estimation de mon temps d'écran : ____ heures ____ minutes"
        ) == "Estimation de mon temps d'écran."

    def test_a_line_that_was_only_a_rule_disappears(self):
        assert normalize_french("____________________________").strip() == ""

    def test_the_parenthesis_holds_the_only_real_content(self):
        # C'est le programme du livre : il doit survivre au nettoyage.
        assert normalize_french(
            "Jour 5: ___ minutes (objectif: 10 min) / Ressenti: ___"
        ) == "Jour cinq, objectif: dix minutes. Ressenti."

    def test_repeated_fields_collapse_instead_of_leaving_slashes(self):
        assert normalize_french(
            "Mes trois créneaux quotidiens: ___ h / ___ h / ___ h"
        ) == "Mes trois créneaux quotidiens."

    def test_a_slash_inside_a_parenthesis_is_not_a_field_separator(self):
        # L'incise est ensuite aplatie par la passe des parenthèses ; ce que
        # ce test protège, c'est qu'elle soit restée d'un seul tenant.
        assert normalize_french(
            "Temps réel mesuré (Screen Time / Bien-être numérique) : ____ heures"
        ) == "Temps réel mesuré, Screen Time / Bien-être numérique."

    def test_prose_behind_a_blank_is_not_eaten(self):
        # Le garde-fou : au-delà d'une queue courte, ce n'est plus une unité,
        # c'est la phrase — on ôte le trait et on lui laisse ses mots.
        assert normalize_french(
            "Elle note ___ dans la marge puis referme le carnet et sort."
        ) == "Elle note dans la marge puis referme le carnet et sort."

    def test_a_line_without_a_blank_keeps_its_slashes(self):
        assert normalize_french(
            "Il gagne trois mille euros / mois."
        ) == "Il gagne trois mille euros / mois."


class TestMinutesAbbreviation:
    """« 20 h » était lu, « 5 min » ne l'était par personne : 106 fois dans les
    vingt et un livres de la file, dit « min »."""

    def test_minutes_are_spoken(self):
        assert normalize_french("une séance de 5 min") == "une séance de cinq minutes"

    def test_one_minute_stays_singular(self):
        assert normalize_french("après 1 min") == "après une minute"

    def test_an_already_spelled_minute_is_left_alone(self):
        assert normalize_french("après 5 minutes") == "après cinq minutes"

    def test_a_word_beginning_with_min_is_untouched(self):
        assert normalize_french("le minimum de 3 minutes") == "le minimum de trois minutes"


class TestNumericRanges:
    """Un trait d'union entre deux nombres se dit « à ».

    Ne pas le dire ne laisse pas un silence : il colle les deux nombres et la
    passe des nombres les fond en un seul. « La pandémie de 2020-2022 » se
    narrait « deux mille vingt-deux mille vingt-deux ». 726 intervalles dans
    les vingt et un livres de la file.
    """

    def test_a_year_range_is_two_years(self):
        assert normalize_french("la pandémie de 2020-2022") == (
            "la pandémie de deux mille vingt à deux mille vingt-deux"
        )

    def test_a_quantity_range(self):
        assert normalize_french("dormir 7-8 heures") == "dormir sept à huit heures"

    def test_an_hour_range_keeps_its_minutes(self):
        assert normalize_french("de 14h-15h30") == "de quatorze heures à quinze heures trente"

    def test_a_breathing_exercise_is_not_a_range(self):
        # « 4-7-8 » est une respiration, « 5-4-3-2-1 » un exercice d'ancrage :
        # trois nombres à dire l'un après l'autre, pas un intervalle.
        assert "à" not in normalize_french("la respiration 4-7-8")
        assert "à" not in normalize_french("exercice 5-4-3-2-1")

    def test_a_hyphenated_name_with_a_number_is_untouched(self):
        assert normalize_french("le COVID-19") == "le COVID-dix-neuf"
