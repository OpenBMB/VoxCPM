"""Narrate a whole book / long script to per-chapter WAV files — robustly.

Designed for long-form content (books, guided meditations, podcast scripts) where
a single generation call is not possible (the engine errors above ~8192 tokens)
and holding the whole audio in memory is wasteful.

The pipeline
------------
0. **Read** — a ``.txt`` splits into chapters on lines containing only ``---``;
   an ``.epub`` is read in spine order, chapters and their titles taken from the
   book's own table of contents (``--no-epub-split``, ``--epub-min-chars``).
1. **Prepare** — the text goes through the French normalizer, so ``1789``,
   ``M. Dupont``, ``XIVe siècle`` and ``14h30`` are read as a narrator would say
   them (``--no-text-prep`` to disable, ``--lexicon`` for your own proper nouns).
2. **Segment** — each chapter is cut on sentence boundaries so every call stays
   well under the engine's token limit, and each segment carries how long the
   pause after it should be: longer after a paragraph than after a full stop.
3. **Synthesize** — with the SAME seed throughout, so the voice stays identical.
   Every segment is cached by content, so an interrupted run resumes at the
   segment it died on rather than restarting the chapter.
4. **Master** — segments are trimmed, de-clicked, stitched with their pauses and
   normalised once per chapter to the audiobook loudness target.
5. **Assemble** (optional, ``--assemble``) — chapters are joined into a single
   M4B/MP3 with chapter markers.

**Memory-safe:** only one chapter is held in memory at a time, never the book.
**Resumable:** finished chapters are skipped, and within an unfinished chapter
every already-generated segment comes from the cache.
The denoiser is never loaded (narration uses no reference audio), so startup is
fast and does not touch ModelScope.

Examples
--------
  # Preview segmentation and prepared text without generating anything:
  ./.venv/Scripts/python.exe scripts/narrate_book.py livre.txt --voice "Narrateur profond & calme" --dry-run

  # Narrate with a preset voice, then assemble an M4B:
  ./.venv/Scripts/python.exe scripts/narrate_book.py livre.txt --voice "Narrateur profond & calme" --assemble m4b

  # Custom voice (description + seed):
  ./.venv/Scripts/python.exe scripts/narrate_book.py livre.txt --description "Voix ..." --seed 123

  # Straight from an EPUB (chapters and titles come from the book):
  ./.venv/Scripts/python.exe scripts/narrate_book.py livre.epub --voice "Narrateur profond & calme" --dry-run

  # On a CUDA GPU (far faster):
  ./.venv/Scripts/python.exe scripts/narrate_book.py livre.txt --voice "..." --device cuda
"""
import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Make the repo root importable so we can reuse app.py's helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402
from narration import assemble as assembly  # noqa: E402
from narration import audio as audio_tools  # noqa: E402
from narration import cache as cache_tools  # noqa: E402
from narration import chunking, couverture, credits, epub, quality, repair, text_en, text_fr  # noqa: E402

#: Rough characters-per-second of finished narration, used only to estimate how
#: long a book will run before committing hours of CPU to it.
_CHARS_PER_SECOND = 14.0


def resolve_voice(args) -> tuple[str, int | None]:
    """Return (description, seed) from a preset name or explicit --description/--seed.

    A cloned preset also fills in --reference-audio and --reference-text, unless
    the command line gave its own: what is typed now beats what was configured
    once.
    """
    if args.voice:
        preset = app._PRESET_BY_NAME.get(args.voice)
        if preset is None:
            names = ", ".join(repr(v["name"]) for v in app.PRESET_VOICES)
            raise SystemExit(f"Unknown voice {args.voice!r}. Available presets: {names}")
        if preset.get("reference") and not args.reference_audio:
            args.reference_audio = preset["reference"]
            args.reference_text = args.reference_text or preset.get("reference_text", "")
        return preset["description"], preset["seed"]
    return (args.description or ""), args.seed


def describe_reference(path: str, text: str) -> str:
    """One line on the state of a cloning recording, for the pre-flight summary.

    Printed before the model is even loaded — and so during ``--dry-run`` too,
    which is where it earns its keep: a recording whose transcript does not
    cover it truncates every segment of the book, and the symptom appears
    minutes of CPU away from the cause.
    """
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as error:  # noqa: BLE001 - the engine will fail on it too, more obscurely
        return f"référence illisible ({error})"

    report = quality.inspect_reference(wav, sr, text)
    if report.ok:
        return f"référence saine ({report.speech_sec:.1f}s de parole, {report.chars_per_second:.0f} car/s)"
    marks = {quality.FATAL: "/!\\", quality.SUSPECT: "(!)"}
    return " ; ".join(f"{marks.get(i.severity, '')} {i.detail}" for i in report.issues)


def chapter_title(chapter: str, index: int) -> str:
    """First non-empty line of a chapter, used as its marker title."""
    for line in chapter.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:80]
    return f"Chapitre {index}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Path to the .txt or .epub file to narrate")

    voice = parser.add_argument_group("voix")
    voice.add_argument("--voice", help="Preset voice name (see conf/preset_voices.json)")
    voice.add_argument("--description", help="Custom voice description (if not using --voice)")
    voice.add_argument("--seed", type=int, help="Seed for the custom voice (fixes the voice identity)")
    voice.add_argument("--cfg", type=float, default=2.0, help="CFG guidance scale (default: 2.0)")
    voice.add_argument("--steps", type=int, default=10, help="Diffusion steps (default: 10)")
    voice.add_argument("--reference-audio", metavar="WAV",
                       help="Clone a voice from this recording instead of describing one. "
                            "A short, clean take beats a long noisy one — the denoiser is "
                            "off during narration, so what is in the file is what is copied")
    voice.add_argument("--reference-text",
                       help="Exact transcript of --reference-audio. Optional, and worth "
                            "giving: the engine matches the words to the audio and clones "
                            "more faithfully with it")

    text = parser.add_argument_group("texte")
    text.add_argument("--language", choices=["fr", "en"], default="fr",
                      help="Language of the book: picks the text preparation and the "
                           "wording of the credits (default: fr)")
    text.add_argument("--no-text-prep", action="store_true",
                      help="Skip French normalization (numbers, abbreviations, Roman numerals)")
    text.add_argument("--lexicon", action="append", metavar="FICHIER",
                      help="Lexique de prononciation JSON. Répétable : les fichiers "
                           "s'empilent et le dernier gagne, donc un lexique propre à un "
                           "livre s'ajoute au lexique général plutôt que de le remplacer. "
                           "Défaut : conf/pronunciation_fr.json")
    text.add_argument("--no-normalize", action="store_true", help="Disable the engine's own text normalization")
    text.add_argument("--chapter-regex", help="Regex (MULTILINE) that separates chapters (default: '^---$')")
    text.add_argument("--epub-min-chars", type=int, default=epub.DEFAULT_MIN_CHARS,
                      help="EPUB: below this many characters a document is front matter, "
                           f"not a chapter (default: {epub.DEFAULT_MIN_CHARS})")
    text.add_argument("--no-epub-split", action="store_true",
                      help="EPUB: keep one chapter per file instead of cutting files that "
                           "hold several chapters at their headings")
    text.add_argument("--keep-boilerplate", action="store_true",
                      help="EPUB: keep the Project Gutenberg header and licence, and any "
                           "contents page, instead of removing them")
    text.add_argument("--chunk-max-chars", type=int, default=chunking.DEFAULT_MAX_CHARS,
                      help=f"Max characters per segment (default: {chunking.DEFAULT_MAX_CHARS})")

    pauses = parser.add_argument_group("pauses et mastering")
    defaults = chunking.PauseProfile()
    pauses.add_argument("--pause-clause", type=float, default=defaults.clause,
                        help=f"Pause after a mid-sentence split (default: {defaults.clause}s)")
    pauses.add_argument("--pause-sentence", type=float, default=defaults.sentence,
                        help=f"Pause after a sentence (default: {defaults.sentence}s)")
    pauses.add_argument("--pause-paragraph", type=float, default=defaults.paragraph,
                        help=f"Pause after a paragraph (default: {defaults.paragraph}s)")
    pauses.add_argument("--silence", type=float,
                        help="Force one uniform pause everywhere, overriding the three above")
    pauses.add_argument("--target-rms", type=float, default=audio_tools.MasteringSettings().target_rms_db,
                        help="Loudness target in dBFS (ACX window is -23..-18, default: -20)")
    pauses.add_argument("--no-master", action="store_true",
                        help="Skip trimming, de-clicking and loudness normalization")
    pauses.add_argument("--no-polish", action="store_true",
                        help="Skip the studio chain (high-pass, de-esser, compressor, limiter) "
                             "applied to each chapter before its level is set")

    run = parser.add_argument_group("exécution")
    run.add_argument("--outdir", help="Output directory (default: output/book_<filename>)")
    run.add_argument("--device", default="cpu", help="auto, cpu, mps, cuda, or cuda:N (default: cpu)")
    run.add_argument("--model-id", default="openbmb/VoxCPM2", help="Model path or HF repo id")
    run.add_argument("--force", action="store_true", help="Regenerate chapters even if their .wav exists")
    run.add_argument("--no-cache", action="store_true", help="Do not cache or reuse generated segments")
    run.add_argument("--dry-run", action="store_true", help="Show the plan, generate nothing")
    run.add_argument("--assemble", nargs="?", const="m4b", choices=["m4b", "m4a", "mp3", "wav"],
                     help="Assemble the chapters into one chaptered file when done")
    run.add_argument("--cover", help="Cover image for the assembled file "
                                     "(default: the EPUB's own, when there is one)")
    run.add_argument("--no-cover", action="store_true",
                     help="Do not embed a cover in the assembled file")
    run.add_argument("--assemble-bitrate",
                     help="Bitrate of the assembled file, e.g. 96, 128k "
                          "(default: 64k AAC, 128k MP3)")
    run.add_argument("--export-acx", action="store_true",
                     help="Prepare the folder a distributor accepts (one file per chapter, "
                          "192 kbps CBR MP3, retail sample) once the narration is done")
    run.add_argument("--title", default="", help="Book title (assembled file, and credits)")
    run.add_argument("--author", default="", help="Author (assembled file, and credits)")

    story = parser.add_argument_group("generique")
    story.add_argument("--no-synthetic-disclosure", action="store_true",
                       help="Ne pas dire « une voix de synthèse » dans les génériques. "
                            "ACX, Apple Books et Findaway l'exigent pour une narration "
                            "générée : le retirer est un choix d'éditeur, pas un réglage "
                            "de confort, et il expose au rejet ou au retrait.")
    story.add_argument("--voice-name", default="",
                       help="Nom donné à la voix de synthèse dans les génériques "
                            "(« Aurore Cabonet »). Le générique dit alors le nom ET "
                            "qu'il s'agit d'une voix de synthèse : le nom seul "
                            "créditerait une interprétation qui n'a pas eu lieu.")
    story.add_argument("--narrator", default="",
                       help="Human narrator named in the credits. Left empty, the credits "
                            "disclose a synthetic voice, as distributors require")
    story.add_argument("--publisher", default="", help="Production credited at the end")
    story.add_argument("--year", default="", help="Year credited at the end")
    story.add_argument("--public-domain", action="store_true",
                       help="State in the closing credits that the text is public domain")
    story.add_argument("--no-credits", action="store_true",
                       help="Do not add the opening and closing credits distributors require")
    run.add_argument("--continuity", action="store_true",
                     help="EXPERIMENTAL: chain each segment from the previous one (prompt-cache "
                          "continuation) for smoother joins, instead of same-seed only. Slower; "
                          "resets at each chapter boundary. Tune on a GPU (slow to iterate on CPU).")

    qc = parser.add_argument_group("contrôle qualité")
    qc.add_argument("--no-qc", action="store_true",
                    help="Do not inspect generated segments for defects")
    qc.add_argument("--qc-retries", type=int, default=1, metavar="N",
                    help="Re-roll a fatally defective segment up to N times with a derived "
                         "seed (default: 1; 0 to report defects without regenerating)")
    qc.add_argument("--qc-strict", action="store_true",
                    help="Exit non-zero if any segment is still defective at the end")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.voice and not args.description and not args.reference_audio:
        raise SystemExit(
            "Provide either --voice <preset name>, --description <text> [--seed N], "
            "or --reference-audio <wav> to clone a voice."
        )
    if args.reference_audio and not Path(args.reference_audio).is_file():
        raise SystemExit(f"Reference audio not found: {args.reference_audio}")


    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input file not found: {in_path}")
    if epub.is_epub(in_path):
        try:
            raw_text, book = epub.load_book_text(
                in_path,
                min_chars=args.epub_min_chars,
                split_on_headings=not args.no_epub_split,
                strip_boilerplate=not args.keep_boilerplate,
            )
        except epub.EpubError as error:
            raise SystemExit(str(error))
        print(epub.summarize(book))
        print()
    else:
        book = None
        raw_text = in_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise SystemExit(f"Input file is empty: {in_path}")

    description, seed = resolve_voice(args)
    # A preset may have just supplied one, so the file is checked again here.
    if args.reference_audio and not Path(args.reference_audio).is_file():
        raise SystemExit(f"Reference audio not found: {args.reference_audio}")
    outdir = Path(args.outdir) if args.outdir else app._OUTPUT_DIR / f"book_{app._sanitize_filename(in_path.stem)}"

    # ---- prepare -------------------------------------------------------
    raw_chapters = chunking.split_chapters(raw_text, args.chapter_regex)
    titles = [chapter_title(chapter, i) for i, chapter in enumerate(raw_chapters, 1)]

    # Credits are chapters like any other, deliberately: they then go through
    # the same French preparation, the same voice and seed, the same mastering
    # and the same cache as the book, so they sound like the narrator rather
    # than an announcement bolted on afterwards.
    book_credits = credits.BookCredits(
        title=args.title or (book.title if book else "") or in_path.stem,
        author=args.author or (book.author if book else ""),
        narrator=args.narrator,
        voice_name=args.voice_name,
        disclose_synthetic=not args.no_synthetic_disclosure,
        publisher=args.publisher,
        year=args.year,
        public_domain=args.public_domain,
        language=args.language,
    )
    if not args.no_credits:
        opening_title, closing_title = credits.titles_for(args.language)
        raw_chapters = [book_credits.opening()] + raw_chapters + [book_credits.closing()]
        titles = [opening_title] + titles + [closing_title]

    lexicon = {}
    #: Les lexiques nommés qui n'ont rien donné. `load_lexicon` est tolérant
    #: par dessein — un fichier d'appoint mal formé ne doit pas tuer une
    #: narration — mais la tolérance était devenue silence : le lexique commun
    #: porte `ce` → `çe`, 12 498 occurrences dans le catalogue, validé à
    #: l'oreille. Un chemin faux, une virgule en trop, et trois cents livres se
    #: narrent avec la mauvaise prononciation du mot le plus fréquent sans
    #: qu'une ligne le dise. La correction, elle, invalide le cache de segments
    #: de *tous* les livres : elle coûte le catalogue, pas un fichier.
    lexiques_vides: list[str] = []
    if not args.no_text_prep:
        # Empiler plutôt que remplacer : un livre qui définit son abréviation
        # maison ne doit pas perdre au passage les sigles communs.
        demandes = list(args.lexicon or [])
        for chemin in (demandes or ["conf/pronunciation_fr.json"]):
            entrees = text_fr.load_lexicon(chemin)
            lexicon.update(entrees)
            if not entrees:
                raison = ("introuvable" if not Path(chemin).is_file()
                          else "illisible ou sans entrée utilisable")
                # Un lexique explicitement demandé qui ne donne rien est une
                # faute de frappe, pas un choix : c'est ce cas qu'on refuse.
                # Le défaut implicite, lui, peut légitimement manquer quand on
                # narre à la main depuis ailleurs — on le dit sans refuser.
                if demandes:
                    lexiques_vides.append(f"{chemin} ({raison})")
                else:
                    print(f"Lexique     : {chemin} {raison} — aucune "
                          f"prononciation n'est appliquée")
        prepare = (
            text_en.normalize_english if args.language == "en" else text_fr.normalize_french
        )
        chapters = [prepare(chapter, lexicon=lexicon) for chapter in raw_chapters]
    else:
        chapters = raw_chapters

    if args.silence is not None:
        profile = chunking.PauseProfile(clause=args.silence, sentence=args.silence, paragraph=args.silence)
    else:
        profile = chunking.PauseProfile(
            clause=args.pause_clause, sentence=args.pause_sentence, paragraph=args.pause_paragraph
        )

    plan = [
        (index, chunking.split_into_segments(chapter, args.chunk_max_chars, profile))
        for index, chapter in enumerate(chapters, 1)
    ]
    total_segments = sum(len(segments) for _, segments in plan)
    total_chars = sum(chunking.total_characters(segments) for _, segments in plan)

    print(f"Entrée      : {in_path}")
    if args.reference_audio:
        print(f"Voix        : clonée de {Path(args.reference_audio).name}"
              + (" (avec transcription)" if args.reference_text else " (sans transcription)"))
        print(f"              {describe_reference(args.reference_audio, args.reference_text or '')}")
    else:
        print(f"Voix        : {args.voice or '(personnalisée)'} | seed={seed}")
    print(f"Préparation : {'désactivée' if args.no_text_prep else f'française ({len(lexicon)} entrée(s) de lexique)'}")
    print(f"Chapitres   : {len(chapters)} | segments : {total_segments} | caractères : {total_chars}")
    print(f"Durée estimée : ~{total_chars / _CHARS_PER_SECOND / 60:.0f} min de narration")
    if args.no_credits:
        print("Générique   : aucun (les distributeurs en exigent un au début et à la fin)")
    else:
        missing = book_credits.missing_for_distribution()
        print("Générique   : début et fin ajoutés"
              + (f" — manque encore {', '.join(missing)}" if missing else ""))
    print(f"Sortie      : {outdir}")
    for index, segments in plan:
        print(f"  chapitre {index:03d}: {len(segments)} segment(s)  « {titles[index - 1][:50]} »")

    # ---- lexique -------------------------------------------------------
    if lexiques_vides:
        print(f"Lexique     : REFUS — {', '.join(lexiques_vides)}")
        print("              Un lexique nommé qui ne donne rien est une faute "
              "de frappe, pas un choix ; et le corriger après coup invalide "
              "le cache de segments de tous les livres.")
        return 2

    # ---- couverture ----------------------------------------------------
    # Vérifiée ici, avant que le modèle ne se charge — et donc aussi en
    # --dry-run. Elle ne l'était qu'à l'assemblage, c'est-à-dire trois heures
    # de GPU plus tard, où une couverture absente ou non carrée ne produisait
    # qu'une ligne de journal que personne ne relit : `livre-rebatir-intimite`
    # est ainsi sorti sans aucune couverture, et trois autres livres avec une
    # vignette ebook en portrait. Une couverture ne coûte rien à corriger
    # avant la narration et coûte la narration entière après.
    #: Ce qui a été demandé et n'a pas été produit. Vide vaut zéro en sortie.
    manque: list[str] = []

    cover_path = None
    if args.assemble:
        cover_path = Path(args.cover) if args.cover else None
        origine = "fournie"
        if cover_path is None and not args.no_cover and epub.is_epub(in_path):
            # En dry-run on extrait ailleurs : un plan ne crée pas la sortie.
            dest = Path(tempfile.mkdtemp()) if args.dry_run else outdir
            dest.mkdir(parents=True, exist_ok=True)
            cover_path = epub.extract_cover(in_path, dest)
            origine = "tirée de l'EPUB"
        if args.no_cover:
            print("Couverture  : aucune (--no-cover) — les distributeurs en exigent une")
        else:
            verdict = couverture.inspecter(cover_path)
            if verdict.conforme:
                largeur, hauteur = verdict.dimensions
                print(f"Couverture  : {Path(cover_path).name} "
                      f"({largeur}×{hauteur}, {origine})")
            else:
                print(f"Couverture  : REFUS — {verdict.raison}")
                print("              Corrigez-la avant de dépenser la narration, "
                      "ou assumez l'absence avec --no-cover.")
                return 2

    if args.dry_run:
        if plan and plan[0][1]:
            print("\nPremier segment après préparation du texte :")
            print(f"  « {plan[0][1][0].text} »")
        print("\nDry run — rien n'a été généré.")
        return 0

    # ---- synthesize ----------------------------------------------------
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "titles.txt").write_text("\n".join(titles) + "\n", encoding="utf-8")

    voice_spec = cache_tools.VoiceSpec(
        description=description,
        seed=seed,
        cfg=args.cfg,
        steps=args.steps,
        normalize=not args.no_normalize,
        model_id=args.model_id,
        # Hashed by content: the cache must not serve a segment spoken by a
        # different recording that happened to live at the same path.
        reference=cache_tools.VoiceSpec.hash_reference(args.reference_audio),
        reference_text=(args.reference_text or "").strip(),
    )
    cache = cache_tools.ChunkCache(outdir / ".cache", enabled=not args.no_cache)
    mastering = audio_tools.MasteringSettings(
        target_rms_db=args.target_rms, polish=not args.no_polish
    )

    # The plan is what makes a later repair possible: without it, which cache
    # entry holds which sentence is lost the moment this run ends. Written
    # before any audio, so a narration interrupted after nine hours is still
    # repairable — which is exactly the narration worth repairing rather than
    # running again.
    repair.BookPlan(
        voice=dataclasses.asdict(voice_spec),
        mastering=dataclasses.asdict(mastering),
        chapters=tuple(
            repair.PlannedChapter(
                index=index,
                title=titles[index - 1],
                segments=tuple(
                    repair.PlannedSegment(segment.text, segment.pause_after)
                    for segment in segments
                ),
            )
            for index, segments in plan
        ),
    ).save(outdir)

    qc = not args.no_qc
    thresholds = quality.QualityThresholds()
    qc_reports: list[tuple[str, quality.SegmentReport]] = []

    demo = app.VoxCPMDemo(model_id=args.model_id, device=args.device, load_denoiser=False)
    print(f"\nDébut de la narration à {time.strftime('%H:%M:%S')} (device={args.device}). "
          f"C'est lent sur CPU.\n", flush=True)

    for index, segments in plan:
        out = outdir / f"chapitre_{index:03d}.wav"
        if out.is_file() and not args.force:
            print(f"[chapitre {index:03d}/{len(plan)}] déjà présent, ignoré -> {out.name}", flush=True)
            continue

        print(f"[chapitre {index:03d}/{len(plan)}] {len(segments)} segment(s) …", flush=True)
        rendered: list[tuple[np.ndarray, float]] = []
        sample_rate = None
        # Continuity chains each segment to the immediately previous one only
        # (bounded window, so the KV cache never overflows). Reset per chapter.
        previous_wav_path: str | None = None
        previous_text: str | None = None
        previous_key: str | None = None
        temporaries: list[str] = []
        try:
            for position, segment in enumerate(segments):
                key = cache.key(segment.text, voice_spec, parent=previous_key if args.continuity else None)
                label = f"ch{index:03d}/seg{position + 1:03d}"

                def render(current_seed, _segment=segment):
                    """One generation of this segment at a given seed."""
                    if args.continuity and previous_wav_path is not None:
                        # The voice now comes from the running audio, so the
                        # control text is dropped.
                        sr, wav_out, _ = demo.generate_tts_audio(
                            text_input=_segment.text,
                            control_instruction="",
                            reference_wav_path_input=previous_wav_path,
                            prompt_text=previous_text,
                            cfg_value_input=args.cfg,
                            do_normalize=not args.no_normalize,
                            inference_timesteps=args.steps,
                            seed=current_seed,
                        )
                    else:
                        sr, wav_out, _ = demo.generate_tts_audio(
                            text_input=_segment.text,
                            control_instruction=description,
                            reference_wav_path_input=args.reference_audio,
                            prompt_text=(args.reference_text or ""),
                            cfg_value_input=args.cfg,
                            do_normalize=not args.no_normalize,
                            inference_timesteps=args.steps,
                            denoise=False,
                            seed=current_seed,
                        )
                    return sr, wav_out

                cached = cache.get(key)
                if cached is not None:
                    sample_rate, wav = cached
                    status = "cache"
                    # Inspected too: a segment cached by a run that predates the
                    # quality pass, or one that was kept as the least-bad
                    # attempt, should still be reported rather than pass silently.
                    report = quality.inspect_segment(wav, sample_rate, segment.text, thresholds) if qc else None
                elif qc:
                    result = quality.render_checked(
                        segment.text,
                        render,
                        base_seed=seed,
                        max_attempts=max(1, args.qc_retries + 1),
                        thresholds=thresholds,
                    )
                    sample_rate, wav, report = result.sample_rate, result.wav, result.report
                    cache.put(key, sample_rate, wav, text=segment.text)
                    if result.attempts == 1:
                        status = "généré"
                    elif result.unrepairable:
                        status = f"généré, DÉFECTUEUX après {result.attempts} essais"
                    else:
                        status = f"régénéré ({result.attempts} essais)"
                else:
                    sample_rate, wav = render(seed)
                    cache.put(key, sample_rate, wav, text=segment.text)
                    status, report = "généré", None

                if report is not None:
                    qc_reports.append((label, report))
                    if not report.ok:
                        status += f" — {report.describe()}"

                rendered.append((wav, segment.pause_after))
                previous_key = key
                if args.continuity:  # stash this segment as the prompt for the next
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
                        temporaries.append(handle.name)
                    sf.write(temporaries[-1], wav, sample_rate)
                    previous_wav_path, previous_text = temporaries[-1], segment.text
                print(f"    segment {position + 1}/{len(segments)} {status}", flush=True)
        finally:
            for path in temporaries:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        if not rendered or sample_rate is None:
            print(f"[chapitre {index:03d}/{len(plan)}] vide, ignoré", flush=True)
            continue

        if args.no_master:
            chapter_audio = audio_tools.concatenate(
                (wav for wav, _ in rendered), sample_rate, gap_sec=profile.sentence
            )
        else:
            chapter_audio = audio_tools.stitch(rendered, sample_rate, mastering)

        sf.write(str(out), chapter_audio, sample_rate, subtype="PCM_16")
        report = audio_tools.acx_report(chapter_audio, sample_rate)
        print(
            f"[chapitre {index:03d}/{len(plan)}] écrit -> {out.name} "
            f"({report['duration_sec'] / 60:.1f} min, RMS {report['rms_db']:.1f} dBFS, "
            f"crête {report['peak_db']:.1f} dBFS)",
            flush=True,
        )

    print(f"\nTerminé. Chapitres dans : {outdir}")
    print(cache.stats.describe())

    # ---- quality report -------------------------------------------------
    defective = 0
    if qc_reports:
        summary = quality.summarize(qc_reports)
        defective = summary["fatal"]
        report_path = outdir / "qc_report.json"
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if summary["flagged"]:
            codes = ", ".join(f"{code}×{count}" for code, count in sorted(summary["by_code"].items()))
            print(
                f"Contrôle qualité : {summary['flagged']}/{summary['segments']} segment(s) signalé(s) "
                f"({codes}) — détail dans {report_path.name}"
            )
            for detail in summary["details"][:10]:
                print(f"  {detail['segment']}: {', '.join(i['code'] for i in detail['issues'])}")
            if len(summary["details"]) > 10:
                print(f"  … et {len(summary['details']) - 10} autre(s), voir {report_path.name}")
        else:
            print(f"Contrôle qualité : {summary['segments']}/{summary['segments']} segment(s) sains")

    # ---- assemble ------------------------------------------------------
    if args.assemble:
        chapter_files = sorted(p for p in outdir.glob("chapitre_*.wav"))
        if not chapter_files:
            # Sortir zéro ici disait « terminé » d'un livre qui n'existe pas.
            # C'est ce qui a laissé la file annoncer « terminé — 0 chapitre(s) »
            # sur un livre pourtant complet, lancé depuis le mauvais dossier :
            # rien n'était trouvé, tout allait bien, et la purge des WAV — qui
            # dépend de ce compte — ne se déclenchait pas. Un volume saturé
            # trois livres plus loin, pour une cause étrangère à la narration.
            print(f"Rien à assembler : aucun chapitre_*.wav dans {outdir}")
            return 1
        # La couverture a été choisie et vérifiée au pré-vol : on ne la
        # redécide pas ici, sinon la vérification ne porterait pas sur ce qui
        # est réellement embarqué.
        cover_path = Path(cover_path) if cover_path else None

        target = outdir / f"{outdir.name}_complet.{args.assemble}"
        print(f"\nAssemblage de {len(chapter_files)} chapitre(s) -> {target.name}")
        result = assembly.assemble(
            chapter_files,
            target,
            title=args.title or in_path.stem,
            author=args.author,
            titles=titles,
            bitrate=args.assemble_bitrate,
            cover_path=cover_path,
        )
        print(f"Durée totale : {result.duration_sec / 60:.1f} min")
        print(result.message)
        if result.pending_command:
            print("À exécuter une fois ffmpeg installé :")
            print("  " + subprocess.list2cmdline(result.pending_command))
            # La commande est utile à qui la lit ; elle ne remplace pas le
            # fichier. Tant qu'il manque, le livre n'est pas assemblé, et la
            # file ne doit pas le marquer terminé — elle ne le reprendrait
            # jamais.
            manque.append("le M4B (ffmpeg absent ou en échec)")
    else:
        print("Astuce : ajoutez --assemble m4b pour produire un fichier unique avec chapitres, "
              "ou lancez scripts/assemble_audiobook.py plus tard.")

    # ---- deliver -------------------------------------------------------
    if args.export_acx:
        # Run as a subprocess rather than imported: the exporter is a script in
        # its own right, and a book that narrated for nine hours must not lose
        # its chapters to an exception raised while preparing the delivery.
        print()
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("export_acx.py")), str(outdir)]
        )
        if result.returncode:
            print("Export : des fichiers sont hors norme ou n'ont pas été "
                  "encodés, voir ci-dessus.")
            manque.append("un export ACX déposable")

    if args.qc_strict and defective:
        print(f"--qc-strict : {defective} segment(s) toujours défectueux.")
        return 1

    # Le code de sortie dit une seule chose, et il faut qu'il la dise bien :
    # le livrable demandé existe-t-il et peut-il être déposé ? La file s'y fie
    # pour marquer un livre « terminé », et « terminé » ne se rejoue pas.
    if manque:
        print()
        print(f"Incomplet : il manque {', '.join(manque)}.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
