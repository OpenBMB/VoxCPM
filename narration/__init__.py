"""Audiobook narration toolkit built on top of the VoxCPM engine.

This package deliberately depends only on the standard library, ``numpy``,
``scipy`` and ``soundfile`` — never on ``torch`` or ``gradio``. That keeps every stage of the
production chain (text preparation, segmentation, audio mastering, assembly)
importable and unit-testable without loading a multi-gigabyte model, which
matters a lot on a CPU-only machine where model load alone takes minutes.

Stages, in pipeline order::

    epub       read an .epub into the plain chapters everything else expects
    credits    the opening and closing credits distributors require
    voices     the catalogue of narration voices, designed or cloned
    text_fr    prepare raw French prose for a TTS engine
    text_en    the same for English — years, ordinals, titles
    chunking   cut prepared text into engine-sized segments + pause plan
    cache      content-addressed store so an interrupted run resumes per chunk
    quality    flag the segments the engine got wrong, and re-roll those only
    audio      trim, master and stitch the generated segments
    polish     high-pass, de-ess, compress, limit; and measure LUFS
    repair     re-roll one segment and restitch its chapter, from a saved plan
    assemble   join chapters into a single MP3/M4B with chapter markers
    delivery   cut, sample and encode the files a distributor accepts
"""

__all__ = [
    "assemble",
    "audio",
    "cache",
    "chunking",
    "credits",
    "delivery",
    "epub",
    "polish",
    "quality",
    "repair",
    "text_en",
    "text_fr",
    "voices",
]
