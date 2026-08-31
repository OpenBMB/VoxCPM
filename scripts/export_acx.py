"""Prepare the folder an audiobook distributor accepts, from finished chapters.

``narrate_book.py`` produces WAV chapters and, with ``--assemble``, one M4B to
listen to. Neither is what ACX, Audible, Amazon, Apple Books or Kobo take. They
take **one file per chapter**, encoded to a fixed specification, plus a retail
sample — and they reject a submission over details that have nothing to do with
how the narration sounds.

This script turns the one into the other::

    output/book_mon_livre/          ->      output/book_mon_livre/acx/
      chapitre_001.wav                        001 - Generique de debut.mp3
      chapitre_002.wav                        002 - Chapitre premier.mp3
      ...                                     ...
      titles.txt                              extrait_commercial.mp3
                                              rapport_acx.json

What it does
------------
1. **Checks every chapter** against the whole specification — loudness, peak,
   noise floor, room tone at both ends, duration, file size — and says which
   ones would come back, with the reason in plain French.
2. **Splits what is too long.** A chapter over the duration or size limit is cut
   into parts, in a pause rather than mid-word, each part shaped like a file of
   its own.
3. **Extracts a retail sample** of 1 to 5 minutes from the book's introduction
   when it has one, else from its first real chapter. Never the credits, never
   the title page, never the dedication: a sample is what a buyer decides on,
   and what decides them is the argument of the book, not its name read out.
4. **Encodes to 192 kbps CBR MP3 at 44.1 kHz**, which needs ffmpeg.

Without ffmpeg the first three steps still run, the WAVs are written, and the
exact commands to encode them later are printed. Hours of synthesis must never
be held hostage to a missing binary.

Examples
--------
  # Check without producing anything:
  ./.venv/Scripts/python.exe scripts/export_acx.py output/book_mon_livre --check

  # Full export:
  ./.venv/Scripts/python.exe scripts/export_acx.py output/book_mon_livre

  # A longer sample, taken further into the chapter:
  ./.venv/Scripts/python.exe scripts/export_acx.py output/book_mon_livre \
      --sample-seconds 240 --sample-start 60
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from narration import assemble as assembly  # noqa: E402
from narration import audio as audio_tools  # noqa: E402
from narration import credits as credits_tools  # noqa: E402
from narration import delivery  # noqa: E402

#: Where the delivery files land inside the book directory.
DEFAULT_SUBDIR = "acx"
SAMPLE_STEM = "extrait_commercial"
REPORT_NAME = "rapport_acx.json"


def ascii_filename(text: str, fallback: str) -> str:
    """A file name that survives every upload form.

    Distribution portals are not reliably at ease with accents in file names,
    and a rejected upload over "Générique" is a silly way to lose an evening.
    """
    stripped = unicodedata.normalize("NFKD", text or "")
    stripped = stripped.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^\w\s-]", "", stripped).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:60] or fallback


def read_titles(directory: Path, count: int) -> List[str]:
    """Chapter titles from titles.txt, padded to the number of chapters."""
    path = directory / "titles.txt"
    titles: List[str] = []
    if path.is_file():
        # utf-8-sig, because a titles.txt edited on Windows arrives with a byte
        # order mark glued to its first title — enough to stop the opening
        # credits being recognised as credits, and so to sample them.
        titles = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    titles = [title for title in titles if title]
    while len(titles) < count:
        titles.append(f"Chapitre {len(titles) + 1}")
    return titles[:count]


def is_credit(title: str) -> bool:
    """Whether a chapter is one of the two credit files."""
    return title.strip() in (credits_tools.OPENING_TITLE, credits_tools.CLOSING_TITLE)


#: What opens a book by explaining it. A buyer deciding on a sample wants the
#: argument of the book, not its first anecdote and not its title page.
_OPENING_MATTER = re.compile(
    r"^\s*(introduction|avant[- ]propos|pr[ée]face|prologue|pr[ée]ambule)\b",
    re.IGNORECASE,
)

#: Front matter that is read aloud but says nothing about the book: the title
#: page, the dedication, the disclaimer.
_FRONT_MATTER = re.compile(
    r"^\s*(d[ée]dicace|avertissement|copyright|mentions)\b",
    re.IGNORECASE,
)


def sample_chapter_index(titles: Sequence[str], book_title: str = "") -> Optional[int]:
    """Which chapter the retail sample should come from, 1-based.

    Taking the first non-credit chapter is what this did, and it picked the
    title page: a buyer heard the book's own name read out and learnt nothing.
    A sample has to let someone grasp what the book argues without giving the
    book away, and that is exactly what an introduction is for.

    So: the introduction if the book has one, otherwise the first chapter that
    is neither a credit, nor front matter, nor the title page repeated.
    """
    ranked = list(enumerate(titles, 1))
    for index, title in ranked:
        if _OPENING_MATTER.match(title or ""):
            return index

    normalised = (book_title or "").strip().casefold()
    for index, title in ranked:
        clean = (title or "").strip()
        if is_credit(clean) or _FRONT_MATTER.match(clean):
            continue
        if normalised and clean.casefold() == normalised:
            continue
        return index

    return next((i for i, t in ranked if not is_credit(t or "")), None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", help="Directory holding the chapter .wav files")
    parser.add_argument("--out", help=f"Output directory (default: <directory>/{DEFAULT_SUBDIR})")
    parser.add_argument("--pattern", default="chapitre_*.wav",
                        help="Glob selecting the chapters (default: chapitre_*.wav)")
    parser.add_argument("--check", action="store_true",
                        help="Report compliance and write nothing")
    parser.add_argument("--no-sample", action="store_true",
                        help="Do not extract a retail sample")
    parser.add_argument("--sample-seconds", type=float, default=delivery.SAMPLE_TARGET_SEC,
                        help=f"Retail sample length, 60-300 s (default: {delivery.SAMPLE_TARGET_SEC:.0f})")
    parser.add_argument("--sample-start", type=float, default=0.0,
                        help="Seconds into the chapter the sample starts (default: 0)")
    parser.add_argument("--sample-chapter", type=int,
                        help="1-based chapter to sample (default: the introduction, else the first real chapter)")
    parser.add_argument("--keep-wav", action="store_true",
                        help="Keep the intermediate WAV of each delivered file")
    return parser


def describe(report: dict) -> str:
    """One line saying whether a file passes, and why not when it does not."""
    if report["compliant"]:
        return (
            f"OK — {report['duration_sec'] / 60:.1f} min, "
            f"RMS {report['rms_db']:.1f} dBFS, "
            f"~{report['encoded_bytes'] / 1e6:.0f} Mo"
        )
    return "HORS NORME — " + " ; ".join(delivery.failures(report))


def encode(wav_path: Path, out_path: Path, ffmpeg: Optional[str]) -> Tuple[bool, List[str]]:
    """Encode one file, or hand back the command when ffmpeg is missing."""
    command = delivery.encode_command(wav_path, out_path)
    if not ffmpeg:
        return False, command
    command[0] = ffmpeg
    # Comme à l'assemblage : ffmpeg renvoie les métadonnées du livre sur sa
    # sortie d'erreur, et un shell sans LANG fait retomber Python sur l'ASCII.
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"    échec de l'encodage : {result.stderr.strip().splitlines()[-1:]}")
        return False, command
    return True, command


def main() -> int:
    args = build_parser().parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")

    chapter_paths = sorted(p for p in directory.glob(args.pattern) if p.is_file())
    if not chapter_paths:
        raise SystemExit(f"No chapter files matching {args.pattern!r} in {directory}")

    titles = read_titles(directory, len(chapter_paths))
    outdir = Path(args.out) if args.out else directory / DEFAULT_SUBDIR
    ffmpeg = assembly.find_ffmpeg()
    profile = delivery.ACX_PROFILE
    limit = delivery.max_seconds_for(profile)

    print(f"Source      : {directory}")
    print(f"Chapitres   : {len(chapter_paths)}")
    print(f"Norme       : {profile.name} — MP3 {profile.bitrate_kbps} kbps CBR, "
          f"{profile.sample_rate} Hz, {profile.channels} canal")
    print(f"Limite      : {limit / 60:.0f} min par fichier "
          f"({profile.max_bytes / 1e6:.0f} Mo)")
    if not args.check:
        print(f"Sortie      : {outdir}")
    if not ffmpeg and not args.check:
        print("ffmpeg absent : les WAV et les commandes d'encodage seront produits.")
    print()

    if not args.check:
        outdir.mkdir(parents=True, exist_ok=True)

    entries: List[dict] = []
    commands: List[List[str]] = []
    failures = 0
    sample_source: Optional[Tuple[np.ndarray, int, str]] = None
    preferred_sample = sample_chapter_index(titles)

    for index, path in enumerate(chapter_paths, 1):
        title = titles[index - 1]
        data, sample_rate = sf.read(str(path), dtype="float32")
        data = audio_tools.as_float_mono(data)
        parts = delivery.split_for_delivery(data, sample_rate, profile)

        wanted = args.sample_chapter == index if args.sample_chapter else index == preferred_sample
        if sample_source is None and wanted:
            sample_source = (data, sample_rate, title)

        for part_number, part in enumerate(parts, 1):
            suffix = f" (partie {part_number})" if len(parts) > 1 else ""
            label = f"{len(entries) + 1:03d} - {ascii_filename(title + suffix, f'Chapitre {index}')}"
            report = delivery.check_delivered(part, sample_rate, profile)
            print(f"  {label}: {describe(report)}")
            if not report["compliant"]:
                failures += 1

            entry = {
                "file": label + profile.suffix,
                "source": path.name,
                "title": title + suffix,
                **{key: value for key, value in report.items() if key != "profile"},
            }
            entries.append(entry)

            if args.check:
                continue

            wav_path = outdir / (label + ".wav")
            sf.write(str(wav_path), part, sample_rate, subtype="PCM_16")
            done, command = encode(wav_path, outdir / (label + profile.suffix), ffmpeg)
            if done:
                encoded = (outdir / (label + profile.suffix)).stat().st_size
                entry.update(
                    delivery.check_delivered(
                        part, sample_rate, profile, encoded_bytes=encoded
                    )
                )
                entry["file"] = label + profile.suffix
                if not args.keep_wav:
                    wav_path.unlink(missing_ok=True)
            else:
                commands.append(command)

    if not args.no_sample and sample_source is not None:
        data, sample_rate, title = sample_source
        sample = delivery.retail_sample(
            data,
            sample_rate,
            target_seconds=args.sample_seconds,
            start_seconds=args.sample_start,
        )
        report = delivery.check_delivered(sample, sample_rate, profile)
        print(f"\n  {SAMPLE_STEM}: {sample.size / sample_rate / 60:.1f} min, tiré de « {title} » — "
              f"{describe(report)}")
        entries.append({"file": SAMPLE_STEM + profile.suffix, "title": "Extrait commercial", **report})
        if not args.check:
            wav_path = outdir / (SAMPLE_STEM + ".wav")
            sf.write(str(wav_path), sample, sample_rate, subtype="PCM_16")
            done, command = encode(wav_path, outdir / (SAMPLE_STEM + profile.suffix), ffmpeg)
            if done and not args.keep_wav:
                wav_path.unlink(missing_ok=True)
            elif not done:
                commands.append(command)

    if not args.check:
        (outdir / REPORT_NAME).write_text(
            json.dumps(
                {"profile": profile.name, "limit_seconds": limit, "files": entries},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print()
    if failures:
        print(f"{failures} fichier(s) hors norme — voir les raisons ci-dessus.")
    else:
        print(f"Les {len(entries)} fichier(s) satisfont la norme {profile.name}.")

    if commands:
        print(f"\nffmpeg absent : {len(commands)} fichier(s) restent à encoder. Par exemple :")
        print("  " + " ".join(commands[0]))
        script = outdir / "encoder.txt"
        if not args.check:
            script.write_text(
                "\n".join(" ".join(command) for command in commands) + "\n", encoding="utf-8"
            )
            print(f"Toutes les commandes sont dans {script}")

    # Des commandes qui restent, ce sont des MP3 qui n'existent pas : le
    # dossier ne contient alors que des WAV et un `encoder.txt`, et rien ne
    # peut être déposé. Sortir zéro là-dessus, c'est dire « livré » d'un
    # dossier vide — sauf en --check, qui n'écrit rien par contrat.
    reste_a_encoder = bool(commands) and not args.check
    return 1 if (failures or reste_a_encoder) else 0


if __name__ == "__main__":
    raise SystemExit(main())
