"""Tests for the English text preparation.

The interesting cases are the ones where English is quietly irregular: a year is
said rather than counted, an ordinal suffix depends on the last two digits, and
a title's period is not the end of a sentence.
"""
import pytest

from narration.text_en import cardinal_en, normalize_english, ordinal_en, year_en


class TestCardinals:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0, "zero"),
            (7, "seven"),
            (13, "thirteen"),
            (21, "twenty-one"),
            (40, "forty"),
            (100, "one hundred"),
            (101, "one hundred and one"),
            (999, "nine hundred and ninety-nine"),
            (1000, "one thousand"),
            (1_000_000, "one million"),
            (-5, "minus five"),
        ],
    )
    def test_it_spells_them_out(self, value, expected):
        assert cardinal_en(value) == expected

    def test_a_large_number_reads_in_scale_order(self):
        assert cardinal_en(1_234_567).startswith("one million two hundred and thirty-four thousand")


class TestOrdinals:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1, "first"), (2, "second"), (3, "third"), (5, "fifth"),
            (8, "eighth"), (9, "ninth"), (11, "eleventh"), (12, "twelfth"),
            (20, "twentieth"), (21, "twenty-first"), (22, "twenty-second"),
            (100, "one hundredth"), (1000, "one thousandth"),
        ],
    )
    def test_the_irregular_ones_are_right(self, value, expected):
        assert ordinal_en(value) == expected

    def test_written_suffixes_are_expanded(self):
        assert "twenty-first of May" in normalize_english("21st of May")
        assert "second" in normalize_english("2nd")


class TestYears:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1789, "seventeen eighty-nine"),
            (1066, "ten sixty-six"),
            (1900, "nineteen hundred"),
            (1905, "nineteen oh five"),
            (2000, "two thousand"),
            (2005, "two thousand five"),
            (2026, "twenty twenty-six"),
        ],
    )
    def test_a_year_is_said_not_counted(self, value, expected):
        assert year_en(value) == expected

    def test_prose_reads_four_digits_as_a_year(self):
        assert "seventeen eighty-nine" in normalize_english("It began in 1789.")

    def test_a_thousands_separator_means_a_quantity(self):
        """`1,789 men` is counted; only a bare 1789 is a year."""
        out = normalize_english("There were 1,789 men.")
        assert "one thousand seven hundred and eighty-nine" in out

    def test_it_can_be_switched_off(self):
        out = normalize_english("In 1789.", read_years=False)
        assert "one thousand seven hundred and eighty-nine" in out


class TestAbbreviations:
    def test_a_title_does_not_end_the_sentence(self):
        """Leaving the period would split "Mister. Dupont" in two."""
        out = normalize_english("Mr. Dupont met Dr. Smith at St. Paul.")
        assert "Mister Dupont" in out
        assert "Doctor Smith" in out
        assert "Saint Paul" in out
        assert "Mister." not in out

    def test_etc_keeps_its_period_because_it_may_end_one(self):
        assert normalize_english("And so on, etc. The end.").count(".") == 2

    def test_latin_shorthand_is_spoken(self):
        assert "for example" in normalize_english("Fruit, e.g. apples.")
        assert "that is" in normalize_english("One, i.e. the first.")


class TestQuantities:
    def test_money_names_its_parts(self):
        out = normalize_english("It cost $1,250.50.")
        assert "one thousand two hundred and fifty dollars and fifty cents" in out

    def test_pounds_have_pence(self):
        assert "five pence" in normalize_english("£3.05")

    def test_one_of_something_is_singular(self):
        assert "one dollar" in normalize_english("$1")

    def test_percentages(self):
        assert "three percent" in normalize_english("3%")

    def test_times(self):
        assert "fourteen thirty" in normalize_english("at 14:30")
        assert "nine o'clock" in normalize_english("at 9:00")
        assert "nine oh five" in normalize_english("at 9:05")


class TestRoman:
    def test_a_trigger_word_is_required(self):
        assert "chapter fourteen" in normalize_english("chapter XIV")

    def test_an_initial_is_not_a_number(self):
        """Without a trigger, `I` and `C` are letters far more often."""
        out = normalize_english("I met C. D. Lewis.")
        assert out.startswith("I met")

    def test_it_can_be_switched_off(self):
        assert "XIV" in normalize_english("chapter XIV", expand_roman=False)


class TestShape:
    def test_empty_input(self):
        assert normalize_english("") == ""
        assert normalize_english("   ") == ""

    def test_markdown_is_stripped(self):
        assert "**" not in normalize_english("A **bold** claim.")

    def test_the_lexicon_applies(self):
        assert "N A S A" in normalize_english("The NASA report.", lexicon={"NASA": "N A S A"})

    def test_a_contextual_lexicon_entry_works_here_too(self):
        lexicon = {"read": {"prononcer": "red", "après": "have|has|had"}}
        out = normalize_english("I have read it. I read daily.", lexicon=lexicon)
        assert "have red it" in out
        assert "I read daily" in out
