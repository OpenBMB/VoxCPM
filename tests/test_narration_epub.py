"""Tests for EPUB import.

The fixtures build real .epub archives rather than mocking ``zipfile``: the
whole point of this module is that it copes with how books are actually laid
out, so the tests exercise the two structures in the wild (EPUB 3 with a nav
document, EPUB 2 with an NCX) plus the malformed cases that must fail loudly.
"""
import zipfile

import pytest

from narration import epub


CONTAINER = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="{opf}" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""


def document(body: str, title: str = "") -> str:
    """A minimal XHTML content document."""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
        f"<title>{title}</title><style>p {{ margin: 0 }}</style></head>"
        f"<body>{body}</body></html>"
    )


def build_epub(
    path,
    documents,
    *,
    title="Le Livre",
    author="Une Autrice",
    nav=None,
    ncx=None,
    spine_extra="",
    encrypted=False,
    prefix="OEBPS/",
):
    """Write a working .epub made of ``documents`` — a list of (name, xhtml)."""
    opf_path = f"{prefix}content.opf"
    manifest_items = []
    spine_items = []
    for index, (name, _) in enumerate(documents, 1):
        manifest_items.append(
            f'<item id="c{index}" href="{name}" media-type="application/xhtml+xml"/>'
        )
        spine_items.append(f'<itemref idref="c{index}"/>')
    if nav is not None:
        manifest_items.append(
            '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" '
            'properties="nav"/>'
        )
        spine_items.insert(0, '<itemref idref="nav"/>')
    if ncx is not None:
        manifest_items.append(
            '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        )

    opf = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:title>{title}</dc:title><dc:creator>{author}</dc:creator>"
        "</metadata>"
        f"<manifest>{''.join(manifest_items)}</manifest>"
        f'<spine{" toc=\"ncx\"" if ncx is not None else ""}>'
        f"{''.join(spine_items)}{spine_extra}</spine>"
        "</package>"
    )

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", CONTAINER.format(opf=opf_path))
        if encrypted:
            archive.writestr("META-INF/encryption.xml", "<encryption/>")
        archive.writestr(opf_path, opf)
        for name, content in documents:
            archive.writestr(f"{prefix}{name}", content)
        if nav is not None:
            archive.writestr(f"{prefix}nav.xhtml", nav)
        if ncx is not None:
            archive.writestr(f"{prefix}toc.ncx", ncx)
    return path


LONG = "Il faisait un temps splendide sur la ville endormie, et personne ne bougeait. " * 3


@pytest.fixture
def simple_book(tmp_path):
    return build_epub(
        tmp_path / "livre.epub",
        [
            ("ch1.xhtml", document(f"<h1>Premier chapitre</h1><p>{LONG}</p>")),
            ("ch2.xhtml", document(f"<h1>Deuxième chapitre</h1><p>{LONG}</p>")),
        ],
    )


class TestReadEpub:
    def test_reads_metadata_and_chapters(self, simple_book):
        book = epub.read_epub(simple_book)
        assert book.title == "Le Livre"
        assert book.author == "Une Autrice"
        assert len(book.chapters) == 2
        assert book.characters > 0

    def test_titles_come_from_the_headings(self, simple_book):
        book = epub.read_epub(simple_book)
        assert [c.title for c in book.chapters] == ["Premier chapitre", "Deuxième chapitre"]

    def test_markup_becomes_paragraphs(self, tmp_path):
        path = build_epub(
            tmp_path / "p.epub",
            [("ch1.xhtml", document(f"<p>{LONG}</p><p>Deuxième paragraphe entier.</p>"))],
        )
        text = epub.read_epub(path).chapters[0].text
        assert "\n\n" in text
        assert "Deuxième paragraphe entier." in text
        # Inline styling must not survive, nor leave its tags in the prose.
        assert "<" not in text

    def test_style_and_script_are_not_narrated(self, tmp_path):
        body = f"<script>var x = 'NEPASLIRE';</script><p>{LONG}</p>"
        path = build_epub(tmp_path / "s.epub", [("ch1.xhtml", document(body))])
        assert "NEPASLIRE" not in epub.read_epub(path).chapters[0].text

    def test_entities_are_decoded(self, tmp_path):
        body = f"<p>{LONG}</p><p>L&#8217;h&ocirc;te &amp; l&rsquo;invit&eacute;.</p>"
        path = build_epub(tmp_path / "e.epub", [("ch1.xhtml", document(body))])
        text = epub.read_epub(path).chapters[0].text
        assert "hôte & l" in text
        assert "&" not in text.replace("hôte & l", "")

    def test_br_breaks_a_line_without_ending_the_paragraph(self, tmp_path):
        body = f"<p>{LONG}</p><p>Premier vers<br/>Second vers</p>"
        path = build_epub(tmp_path / "br.epub", [("ch1.xhtml", document(body))])
        text = epub.read_epub(path).chapters[0].text
        assert "Premier vers\nSecond vers" in text


class TestReadingOrder:
    def test_order_is_the_spine_not_the_file_names(self, tmp_path):
        """Chapter 10 must not be narrated before chapter 2."""
        documents = [
            ("ch10.xhtml", document(f"<h1>Chapitre dix</h1><p>{LONG}</p>")),
            ("ch2.xhtml", document(f"<h1>Chapitre deux</h1><p>{LONG}</p>")),
        ]
        path = build_epub(tmp_path / "order.epub", documents)
        book = epub.read_epub(path)
        assert [c.title for c in book.chapters] == ["Chapitre dix", "Chapitre deux"]

    def test_non_linear_items_are_left_out(self, tmp_path):
        documents = [
            ("ch1.xhtml", document(f"<h1>Le chapitre</h1><p>{LONG}</p>")),
            ("notes.xhtml", document(f"<h1>Notes</h1><p>{LONG}</p>")),
        ]
        path = build_epub(tmp_path / "nl.epub", documents)
        # Rewrite the spine so the notes are marked as outside the reading flow.
        with zipfile.ZipFile(path) as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
        opf = entries["OEBPS/content.opf"].decode()
        entries["OEBPS/content.opf"] = opf.replace(
            '<itemref idref="c2"/>', '<itemref idref="c2" linear="no"/>'
        ).encode()
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in entries.items():
                archive.writestr(name, data)

        book = epub.read_epub(path)
        assert [c.title for c in book.chapters] == ["Le chapitre"]


class TestTableOfContents:
    def test_nav_supplies_titles_when_headings_do_not(self, tmp_path):
        nav = document(
            '<nav epub:type="toc"><ol>'
            '<li><a href="ch1.xhtml">Le manuscrit trouvé</a></li>'
            '<li><a href="ch2.xhtml">La traversée</a></li>'
            "</ol></nav>"
        )
        documents = [
            ("ch1.xhtml", document(f"<div class='t'>Titre stylé</div><p>{LONG}</p>")),
            ("ch2.xhtml", document(f"<div class='t'>Autre</div><p>{LONG}</p>")),
        ]
        path = build_epub(tmp_path / "nav.epub", documents, nav=nav)
        book = epub.read_epub(path)
        assert [c.title for c in book.chapters] == ["Le manuscrit trouvé", "La traversée"]

    def test_the_nav_document_is_not_narrated(self, tmp_path):
        nav = document('<nav epub:type="toc"><ol><li><a href="ch1.xhtml">Un</a></li></ol></nav>')
        path = build_epub(
            tmp_path / "nav2.epub",
            [("ch1.xhtml", document(f"<p>{LONG}</p>"))],
            nav=nav,
        )
        book = epub.read_epub(path)
        assert len(book.chapters) == 1
        assert "nav.xhtml" not in book.chapters[0].href

    def test_ncx_titles_are_read_for_epub2_books(self, tmp_path):
        ncx = (
            '<?xml version="1.0"?>'
            '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><navMap>'
            '<navPoint id="n1" playOrder="1"><navLabel><text>Ouverture</text></navLabel>'
            '<content src="ch1.xhtml"/></navPoint>'
            "</navMap></ncx>"
        )
        path = build_epub(
            tmp_path / "ncx.epub",
            [("ch1.xhtml", document(f"<div>Titre stylé</div><p>{LONG}</p>"))],
            ncx=ncx,
        )
        assert epub.read_epub(path).chapters[0].title == "Ouverture"

    def test_a_broken_toc_costs_titles_not_the_book(self, tmp_path):
        path = build_epub(
            tmp_path / "badtoc.epub",
            [("ch1.xhtml", document(f"<div>x</div><p>{LONG}</p>"))],
            ncx="<ncx><navMap><navPoint>  unclosed",
        )
        book = epub.read_epub(path)
        assert len(book.chapters) == 1

    def test_titles_survive_percent_encoded_hrefs(self, tmp_path):
        nav = document('<nav><ol><li><a href="ch%201.xhtml">Le titre</a></li></ol></nav>')
        path = build_epub(
            tmp_path / "enc.epub",
            [("ch 1.xhtml", document(f"<div>x</div><p>{LONG}</p>"))],
            nav=nav,
        )
        assert epub.read_epub(path).chapters[0].title == "Le titre"

    def test_nested_content_directories_resolve(self, tmp_path):
        """Hrefs are relative to the document that writes them, not to the root."""
        nav = document('<nav><ol><li><a href="ch1.xhtml">Le titre</a></li></ol></nav>')
        path = build_epub(
            tmp_path / "deep.epub",
            [("ch1.xhtml", document(f"<div>x</div><p>{LONG}</p>"))],
            nav=nav,
            prefix="EPUB/text/",
        )
        book = epub.read_epub(path)
        assert book.chapters[0].title == "Le titre"
        assert book.chapters[0].href == "EPUB/text/ch1.xhtml"


class TestHeadingSplit:
    """Books packed several chapters to a file — Gutenberg's layout."""

    def test_a_file_holding_several_chapters_becomes_several_chapters(self, tmp_path):
        body = (
            f"<h2>I Le départ</h2><p>{LONG}</p>"
            f"<h2>II La traversée</h2><p>{LONG}</p>"
            f"<h2>III Le retour</h2><p>{LONG}</p>"
        )
        path = build_epub(tmp_path / "packed.epub", [("all.xhtml", document(body))])
        book = epub.read_epub(path)
        assert [c.title for c in book.chapters] == [
            "I Le départ",
            "II La traversée",
            "III Le retour",
        ]

    def test_scene_headings_do_not_split_a_chapter(self, tmp_path):
        """One file per chapter: the chapter level repeats across the book.

        Each file holds a single ``<h1>`` and several ``<h3>`` scene headings.
        Counted per file the ``<h1>`` looks unique and the ``<h3>`` looks like
        the chapter level; counted across the book the ``<h1>`` is the one that
        repeats, and each file stays one chapter.
        """
        chapter = (
            "<h1>{title}</h1>"
            f"<p>{LONG}</p><h3>Première scène</h3><p>{LONG}</p>"
            f"<h3>Seconde scène</h3><p>{LONG}</p>"
        )
        documents = [
            ("ch1.xhtml", document(chapter.format(title="Le départ"))),
            ("ch2.xhtml", document(chapter.format(title="La traversée"))),
        ]
        path = build_epub(tmp_path / "scenes.epub", documents)
        book = epub.read_epub(path)
        assert [c.title for c in book.chapters] == ["Le départ", "La traversée"]
        assert "Première scène" in book.chapters[0].text

    def test_a_title_page_does_not_prevent_the_split(self, tmp_path):
        """The book title in the first file must not decide the cut level.

        The shape Project Gutenberg produces: a title page carrying the only
        ``<h1>``, then fixed-size files that open straight on chapter headings.
        """
        documents = [
            (
                "f1.xhtml",
                document(
                    f"<h1>Le titre du livre</h1><p>{LONG}</p>"
                    f"<h3>I Le départ</h3><p>{LONG}</p>"
                ),
            ),
            ("f2.xhtml", document(f"<h3>II La traversée</h3><p>{LONG}</p>")),
            ("f3.xhtml", document(f"<h3>III Le retour</h3><p>{LONG}</p>")),
            ("f4.xhtml", document(f"<h3>IV L'arrivée</h3><p>{LONG}</p>")),
        ]
        path = build_epub(tmp_path / "titled.epub", documents)
        titles = [c.title for c in epub.read_epub(path).chapters]
        assert titles[-4:] == [
            "I Le départ",
            "II La traversée",
            "III Le retour",
            "IV L'arrivée",
        ]
        # The title page keeps its own chapter rather than opening chapter one.
        assert titles[0] == "Le titre du livre"

    def test_a_one_file_book_is_left_whole(self, tmp_path):
        """Known limit, pinned deliberately.

        With a single document there is nothing to compare it against, so a
        title above repeated subheadings is read as one chapter with sections —
        the same thing a .txt without separators does. Splitting it is the
        user's call, by inserting `---`.
        """
        body = (
            "<h1>Le titre du livre</h1>"
            f"<h3>I Le départ</h3><p>{LONG}</p>"
            f"<h3>II La traversée</h3><p>{LONG}</p>"
        )
        path = build_epub(tmp_path / "onefile.epub", [("all.xhtml", document(body))])
        book = epub.read_epub(path)
        assert len(book.chapters) == 1
        assert "II La traversée" in book.chapters[0].text

    def test_the_split_can_be_turned_off(self, tmp_path):
        body = f"<h2>I Le départ</h2><p>{LONG}</p><h2>II La traversée</h2><p>{LONG}</p>"
        path = build_epub(tmp_path / "off.epub", [("all.xhtml", document(body))])
        assert len(epub.read_epub(path, split_on_headings=False).chapters) == 1

    def test_text_before_the_first_heading_continues_the_previous_chapter(self, tmp_path):
        """A file that starts mid-chapter must not open a chapter of its own."""
        documents = [
            ("f1.xhtml", document(f"<h2>I Le départ</h2><p>{LONG}</p>")),
            (
                "f2.xhtml",
                document(f"<p>SUITE DU PREMIER.</p><h2>II La traversée</h2><p>{LONG}</p>"),
            ),
        ]
        path = build_epub(tmp_path / "midchapter.epub", documents)
        book = epub.read_epub(path)
        assert [c.title for c in book.chapters] == ["I Le départ", "II La traversée"]
        assert "SUITE DU PREMIER." in book.chapters[0].text

    def test_front_matter_of_the_first_file_stays_its_own_chapter(self, tmp_path):
        """With nothing to continue, an opening section is front matter."""
        body = (
            f"<p>{LONG}</p>"  # title page, before any heading
            f"<h2>I Le départ</h2><p>{LONG}</p>"
            f"<h2>II La traversée</h2><p>{LONG}</p>"
        )
        path = build_epub(tmp_path / "front.epub", [("all.xhtml", document(body))])
        book = epub.read_epub(path)
        assert len(book.chapters) == 3
        assert book.chapters[0].titled is False

    def test_a_section_too_short_to_stand_alone_is_folded_in(self, tmp_path):
        body = (
            f"<h2>I Le départ</h2><p>{LONG}</p>"
            "<h2>Interlude</h2><p>Trois mots.</p>"
            f"<h2>II La traversée</h2><p>{LONG}</p>"
        )
        path = build_epub(tmp_path / "fold.epub", [("all.xhtml", document(body))])
        book = epub.read_epub(path)
        assert [c.title for c in book.chapters] == ["I Le départ", "II La traversée"]
        # Folded, not dropped: the words are still there to be narrated.
        assert "Trois mots." in book.chapters[0].text
        assert "Interlude" in book.chapters[0].text


class TestFrontMatter:
    def test_short_documents_are_dropped(self, tmp_path):
        documents = [
            ("cover.xhtml", document("<p>Couverture</p>")),
            ("ch1.xhtml", document(f"<h1>Chapitre</h1><p>{LONG}</p>")),
        ]
        path = build_epub(tmp_path / "fm.epub", documents)
        book = epub.read_epub(path)
        assert len(book.chapters) == 1
        assert book.skipped == ["OEBPS/cover.xhtml"]

    def test_a_short_book_is_still_a_book(self, tmp_path):
        """Dropping front matter must not empty a book that is simply short."""
        path = build_epub(
            tmp_path / "short.epub",
            [("ch1.xhtml", document("<h1>Un</h1><p>Très court.</p>"))],
        )
        book = epub.read_epub(path)
        assert len(book.chapters) == 1
        assert "Très court." in book.chapters[0].text

    def test_the_floor_is_adjustable(self, tmp_path):
        documents = [
            ("cover.xhtml", document("<p>Couverture</p>")),
            ("ch1.xhtml", document(f"<h1>Chapitre</h1><p>{LONG}</p>")),
        ]
        path = build_epub(tmp_path / "floor.epub", documents)
        assert len(epub.read_epub(path, min_chars=1).chapters) == 2


class TestBoilerplate:
    """The apparatus around a book: Gutenberg wrappers, contents pages."""

    HEADER = (
        "<p>The Project Gutenberg eBook of Le Livre</p>"
        "<p>This eBook is for the use of anyone anywhere at no cost.</p>"
        "<p>*** START OF THE PROJECT GUTENBERG EBOOK LE LIVRE ***</p>"
    )
    FOOTER = (
        "<p>*** END OF THE PROJECT GUTENBERG EBOOK LE LIVRE ***</p>"
        "<p>THE FULL PROJECT GUTENBERG LICENSE — Section 1. General Terms of Use.</p>"
    )

    def test_the_english_notice_before_the_book_is_removed(self, tmp_path):
        body = f"{self.HEADER}<h2>I Le départ</h2><p>{LONG}</p>"
        path = build_epub(tmp_path / "gh.epub", [("ch1.xhtml", document(body))])
        book = epub.read_epub(path)
        assert "Project Gutenberg" not in book.chapters[0].text
        assert LONG.strip()[:40] in book.chapters[0].text

    def test_the_licence_after_the_book_is_removed(self, tmp_path):
        body = f"<h2>I Le départ</h2><p>{LONG}</p>{self.FOOTER}"
        path = build_epub(tmp_path / "gf.epub", [("ch1.xhtml", document(body))])
        book = epub.read_epub(path)
        assert "FULL PROJECT GUTENBERG LICENSE" not in book.chapters[-1].text
        assert LONG.strip()[:40] in book.chapters[-1].text

    def test_a_whole_licence_chapter_is_dropped(self, tmp_path):
        documents = [
            ("ch1.xhtml", document(f"<h2>I Le départ</h2><p>{LONG}</p>{self.FOOTER}")),
            ("ch2.xhtml", document(f"<h2>Licence</h2><p>{LONG}</p>")),
        ]
        path = build_epub(tmp_path / "gc.epub", documents)
        book = epub.read_epub(path)
        assert [c.title for c in book.chapters] == ["I Le départ"]

    def test_the_opening_chapter_is_retitled_from_what_is_left(self, tmp_path):
        """Its title came from a heading inside the notice that was removed."""
        body = (
            "<h2>The Project Gutenberg eBook of Le Livre</h2>"
            "<p>*** START OF THE PROJECT GUTENBERG EBOOK LE LIVRE ***</p>"
            f"<p>LE LIVRE</p><p>par une autrice</p><p>{LONG}</p>"
        )
        path = build_epub(tmp_path / "gt.epub", [("ch1.xhtml", document(body))])
        book = epub.read_epub(path)
        assert book.chapters[0].title == "LE LIVRE"

    def test_nothing_is_removed_silently(self, tmp_path):
        body = f"{self.HEADER}<h2>I Le départ</h2><p>{LONG}</p>{self.FOOTER}"
        path = build_epub(tmp_path / "gr.epub", [("ch1.xhtml", document(body))])
        book = epub.read_epub(path)
        assert len(book.removed) == 2
        assert any("en-tête" in note for note in book.removed)
        assert any("licence" in note for note in book.removed)

    def test_stripping_can_be_turned_off(self, tmp_path):
        body = f"{self.HEADER}<h2>I Le départ</h2><p>{LONG}</p>{self.FOOTER}"
        path = build_epub(tmp_path / "gk.epub", [("ch1.xhtml", document(body))])
        book = epub.read_epub(path, strip_boilerplate=False)
        assert "Project Gutenberg" in book.chapters[0].text
        assert book.removed == []

    def test_a_book_without_the_markers_is_untouched(self, simple_book):
        book = epub.read_epub(simple_book)
        assert book.removed == []
        assert len(book.chapters) == 2

    def test_a_contents_page_is_dropped(self, tmp_path):
        contents = (
            "<h2>Table des matières</h2>"
            "<p>I Le départ</p><p>II. La traversée</p><p>III Le retour</p>"
            "<p>IV L'arrivée</p><p>V La fin</p>"
        )
        documents = [
            ("toc.xhtml", document(contents)),
            ("c1.xhtml", document(f"<h2>I Le départ</h2><p>{LONG}</p>")),
            ("c2.xhtml", document(f"<h2>II La traversée</h2><p>{LONG}</p>")),
            ("c3.xhtml", document(f"<h2>III Le retour</h2><p>{LONG}</p>")),
            ("c4.xhtml", document(f"<h2>IV L'arrivée</h2><p>{LONG}</p>")),
            ("c5.xhtml", document(f"<h2>V La fin</h2><p>{LONG}</p>")),
        ]
        path = build_epub(tmp_path / "toc.epub", documents, prefix="OEBPS/")
        book = epub.read_epub(path, min_chars=1)
        assert "Table des matières" not in [c.title for c in book.chapters]
        assert any("table des matières" in note for note in book.removed)

    def test_punctuation_does_not_hide_a_contents_page(self, tmp_path):
        """`CHAPITRE II.` in the list, `CHAPITRE II` in the heading."""
        assert epub._toc_key("CHAPITRE II.") == epub._toc_key("Chapitre II")

    def test_ordinary_prose_is_never_taken_for_a_contents_page(self, tmp_path):
        """The rule must not be able to eat a chapter of the actual book."""
        documents = [
            ("c1.xhtml", document(f"<h2>I Le départ</h2><p>{LONG}</p>")),
            ("c2.xhtml", document(f"<h2>II La traversée</h2><p>{LONG}</p>")),
        ]
        path = build_epub(tmp_path / "prose.epub", documents)
        book = epub.read_epub(path)
        assert len(book.chapters) == 2
        assert book.removed == []


class TestFailures:
    def test_missing_file(self, tmp_path):
        with pytest.raises(epub.EpubError, match="No such file"):
            epub.read_epub(tmp_path / "absent.epub")

    def test_not_a_zip(self, tmp_path):
        path = tmp_path / "fake.epub"
        path.write_text("ceci n'est pas une archive", encoding="utf-8")
        with pytest.raises(epub.EpubError, match="ZIP"):
            epub.read_epub(path)

    def test_drm_is_refused_with_a_reason(self, tmp_path):
        path = build_epub(
            tmp_path / "drm.epub",
            [("ch1.xhtml", document(f"<p>{LONG}</p>"))],
            encrypted=True,
        )
        with pytest.raises(epub.EpubError, match="DRM"):
            epub.read_epub(path)

    def test_a_book_with_no_text_says_so(self, tmp_path):
        path = build_epub(tmp_path / "scan.epub", [("ch1.xhtml", document("<img src='p1.jpg'/>"))])
        with pytest.raises(epub.EpubError, match="No readable text"):
            epub.read_epub(path)

    def test_container_without_an_opf(self, tmp_path):
        path = tmp_path / "noopf.epub"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("META-INF/container.xml", "<container><rootfiles/></container>")
        with pytest.raises(epub.EpubError, match="no OPF"):
            epub.read_epub(path)

    def test_missing_container(self, tmp_path):
        path = tmp_path / "empty.epub"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("mimetype", "application/epub+zip")
        with pytest.raises(epub.EpubError, match="container"):
            epub.read_epub(path)


class TestToBookText:
    def test_chapters_are_separated_the_way_the_pipeline_expects(self, simple_book):
        from narration import chunking

        text = epub.to_book_text(epub.read_epub(simple_book))
        assert len(chunking.split_chapters(text)) == 2

    def test_each_chapter_opens_with_its_title(self, tmp_path):
        nav = document('<nav><ol><li><a href="ch1.xhtml">Le grand départ</a></li></ol></nav>')
        path = build_epub(
            tmp_path / "t.epub",
            [("ch1.xhtml", document(f"<div>x</div><p>{LONG}</p>"))],
            nav=nav,
        )
        text = epub.to_book_text(epub.read_epub(path))
        assert text.split("\n", 1)[0] == "Le grand départ"

    def test_a_title_already_in_the_text_is_not_repeated(self, simple_book):
        text = epub.to_book_text(epub.read_epub(simple_book))
        assert text.count("Premier chapitre") == 1

    def test_a_heading_broken_over_two_lines_is_not_announced_twice(self, tmp_path):
        """A number above a title reads as one line in the TOC, two in the text."""
        body = f"<h2>VII<br/>Un moment d'ivresse</h2><p>{LONG}</p>"
        path = build_epub(tmp_path / "twoline.epub", [("ch1.xhtml", document(body))])
        text = epub.to_book_text(epub.read_epub(path))
        assert text.count("Un moment d'ivresse") == 1
        # …and the whole title lands on the first line, which downstream turns
        # into the chapter marker. A marker reading "VII" would be useless.
        assert text.split("\n", 1)[0] == "VII Un moment d'ivresse"

    def test_an_invented_title_is_never_read_aloud(self, tmp_path):
        """A file name makes a fine chapter marker and a terrible first sentence."""
        path = build_epub(
            tmp_path / "untitled.epub",
            [("ch1.xhtml", document(f"<p>{LONG}</p>"))],
        )
        book = epub.read_epub(path)
        assert book.chapters[0].titled is False
        assert not epub.to_book_text(book).startswith(book.chapters[0].title)

    def test_a_separator_inside_the_book_does_not_split_a_chapter(self, tmp_path):
        body = f"<p>{LONG}</p><p>---</p><p>{LONG}</p>"
        path = build_epub(tmp_path / "sep.epub", [("ch1.xhtml", document(body))])

        from narration import chunking

        text = epub.to_book_text(epub.read_epub(path))
        assert len(chunking.split_chapters(text)) == 1


class TestCover:
    """A cover is found by any of the three routes real books use."""

    COVER = b"\x89PNG\r\n\x1a\n" + b"pixels"

    def with_cover(self, tmp_path, name, manifest_extra, meta="", image="cover.png",
                   media_type="image/png"):
        """An EPUB whose cover is declared the way ``manifest_extra`` says."""
        path = build_epub(
            tmp_path / name,
            [("ch1.xhtml", document(f"<p>{LONG}</p>"))],
        )
        with zipfile.ZipFile(path) as archive:
            entries = {entry: archive.read(entry) for entry in archive.namelist()}
        opf = entries["OEBPS/content.opf"].decode()
        opf = opf.replace("</manifest>", manifest_extra + "</manifest>")
        opf = opf.replace("</metadata>", meta + "</metadata>")
        entries["OEBPS/content.opf"] = opf.encode()
        entries[f"OEBPS/{image}"] = self.COVER
        with zipfile.ZipFile(path, "w") as archive:
            for entry, data in entries.items():
                archive.writestr(entry, data)
        return path

    def test_epub3_marks_it_with_a_property(self, tmp_path):
        path = self.with_cover(
            tmp_path, "c3.epub",
            '<item id="cov" href="cover.png" media-type="image/png" properties="cover-image"/>',
        )
        out = epub.extract_cover(path, tmp_path / "out")
        assert out is not None and out.read_bytes() == self.COVER
        assert out.suffix == ".png"

    def test_epub2_points_at_it_from_the_metadata(self, tmp_path):
        path = self.with_cover(
            tmp_path, "c2.epub",
            '<item id="cov" href="cover.png" media-type="image/png"/>',
            meta='<meta name="cover" content="cov"/>',
        )
        assert epub.extract_cover(path, tmp_path / "out2") is not None

    def test_a_book_that_declares_nothing_is_found_by_name(self, tmp_path):
        path = self.with_cover(
            tmp_path, "c1.epub",
            '<item id="img" href="cover.png" media-type="image/png"/>',
        )
        assert epub.extract_cover(path, tmp_path / "out3") is not None

    def test_the_extension_follows_the_declared_type(self, tmp_path):
        path = self.with_cover(
            tmp_path, "cj.epub",
            '<item id="cov" href="cover.bin" media-type="image/jpeg" properties="cover-image"/>',
            image="cover.bin",
        )
        out = epub.extract_cover(path, tmp_path / "out4")
        assert out is not None and out.suffix == ".jpg"

    def test_a_book_without_a_cover_is_not_an_error(self, simple_book, tmp_path):
        assert epub.extract_cover(simple_book, tmp_path / "out5") is None

    def test_an_unreadable_file_is_not_an_error_either(self, tmp_path):
        broken = tmp_path / "broken.epub"
        broken.write_text("pas une archive", encoding="utf-8")
        assert epub.extract_cover(broken, tmp_path / "out6") is None
        assert epub.extract_cover(tmp_path / "absent.epub", tmp_path / "out7") is None

    def test_an_explicit_path_is_honoured(self, tmp_path):
        path = self.with_cover(
            tmp_path, "cp.epub",
            '<item id="cov" href="cover.png" media-type="image/png" properties="cover-image"/>',
        )
        target = tmp_path / "ailleurs" / "image.png"
        assert epub.extract_cover(path, target) == target
        assert target.read_bytes() == self.COVER


class TestHelpers:
    def test_is_epub(self, tmp_path):
        assert epub.is_epub("livre.epub")
        assert epub.is_epub("LIVRE.EPUB")
        assert not epub.is_epub("livre.txt")

    def test_load_book_text_returns_both(self, simple_book):
        text, book = epub.load_book_text(simple_book)
        assert book.title == "Le Livre"
        assert "Premier chapitre" in text

    def test_summary_lists_every_chapter(self, simple_book):
        summary = epub.summarize(epub.read_epub(simple_book))
        assert "Le Livre" in summary
        assert "Premier chapitre" in summary
        assert "Deuxième chapitre" in summary
