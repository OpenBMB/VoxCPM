"""Assemble per-chapter WAVs into one chaptered audiobook file.

Run this after ``narrate_book.py`` has produced a directory of chapter WAVs.
Chapters are ordered by filename, which is why ``narrate_book.py`` zero-pads
them (``chapitre_001.wav``, ``chapitre_002.wav``, ...).

Chapter titles come from ``--titles`` if given, otherwise from a ``titles.txt``
next to the WAVs (one title per line), otherwise from the filenames.

Examples
--------
  # M4B with chapter markers (needs ffmpeg on PATH):
  ./.venv/Scripts/python.exe scripts/assemble_audiobook.py output/book_mon_livre \\
      --title "Mon Livre" --author "Edwin" --format m4b

  # No ffmpeg installed? This still produces the full WAV plus the chapter file,
  # and prints the exact command to run once ffmpeg is available:
  ./.venv/Scripts/python.exe scripts/assemble_audiobook.py output/book_mon_livre

  # Check the assembled book against audiobook loudness limits:
  ./.venv/Scripts/python.exe scripts/assemble_audiobook.py output/book_mon_livre --check
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from narration import assemble as assembly  # noqa: E402


def _read_titles(directory: Path, explicit: str | None) -> list[str] | None:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"Titles file not found: {path}")
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    default = directory / "titles.txt"
    if default.is_file():
        return [line.strip() for line in default.read_text(encoding="utf-8").splitlines() if line.strip()]
    return None


def _report_levels(wav_path: Path) -> None:
    """Measure the finished book against the ACX limits."""
    import soundfile as sf

    from narration import audio

    data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    report = audio.acx_report(data, sample_rate)
    print("\nNiveaux (norme ACX / livre audio) :")
    print(f"  durée        : {report['duration_sec'] / 60:.1f} min")
    print(
        f"  RMS          : {report['rms_db']:.1f} dBFS "
        f"[{audio.ACX_RMS_MIN_DB:.0f} .. {audio.ACX_RMS_MAX_DB:.0f}] "
        f"{'OK' if report['rms_ok'] else 'HORS NORME'}"
    )
    print(
        f"  crête        : {report['peak_db']:.1f} dBFS "
        f"[<= {audio.ACX_PEAK_CEILING_DB:.0f}] {'OK' if report['peak_ok'] else 'HORS NORME'}"
    )
    print(
        f"  bruit de fond: {report['noise_floor_db']:.1f} dBFS "
        f"[<= {audio.ACX_NOISE_FLOOR_DB:.0f}] {'OK' if report['noise_floor_ok'] else 'HORS NORME'}"
    )
    print(f"  conforme     : {'oui' if report['compliant'] else 'non'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", help="Directory holding the chapter .wav files")
    parser.add_argument("--out", help="Output file (default: <directory>/<name>.<format>)")
    parser.add_argument(
        "--format",
        default="m4b",
        choices=["m4b", "m4a", "mp3", "wav"],
        help="Container for the finished book (default: m4b)",
    )
    parser.add_argument("--title", default="", help="Book title")
    parser.add_argument("--author", default="", help="Author / narrator")
    parser.add_argument("--titles", help="File with one chapter title per line")
    parser.add_argument("--cover", help="Cover image embedded in the finished file")
    parser.add_argument("--bitrate", help="Audio bitrate, e.g. 96, 128k (default: 64k AAC, 128k MP3)")
    parser.add_argument(
        "--gap",
        type=float,
        default=assembly.DEFAULT_CHAPTER_GAP_SEC,
        help=f"Silence between chapters in seconds (default: {assembly.DEFAULT_CHAPTER_GAP_SEC})",
    )
    parser.add_argument("--pattern", default="*.wav", help="Glob selecting chapter files (default: *.wav)")
    parser.add_argument("--check", action="store_true", help="Report loudness against the ACX limits")
    parser.add_argument(
        "--run-pending",
        action="store_true",
        help="If ffmpeg was missing, try running the encode command anyway",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    chapters = sorted(p for p in directory.glob(args.pattern) if p.is_file())
    # Never fold a previous assembly back into the book.
    chapters = [p for p in chapters if not p.stem.endswith("_complet")]
    if not chapters:
        raise SystemExit(f"No chapter files matching {args.pattern!r} in {directory}")

    book_name = args.title or directory.name.replace("book_", "").replace("_", " ").strip() or "livre"
    out_path = Path(args.out) if args.out else directory / f"{directory.name}_complet.{args.format}"

    print(f"Chapitres : {len(chapters)}")
    for path in chapters:
        print(f"  {path.name}")

    result = assembly.assemble(
        chapters,
        out_path,
        title=book_name,
        author=args.author,
        titles=_read_titles(directory, args.titles),
        gap_sec=args.gap,
        cover_path=args.cover,
        bitrate=args.bitrate,
    )

    print(f"\nDurée totale : {result.duration_sec / 60:.1f} min ({len(result.chapters)} chapitres)")
    print(f"WAV          : {result.wav_path}")
    if result.metadata_path:
        print(f"Marqueurs    : {result.metadata_path}")
    print(result.message)

    if result.pending_command and args.run_pending:
        print("\nExécution de la commande ffmpeg…")
        completed = subprocess.run(result.pending_command)
        if completed.returncode == 0:
            result.output_path = out_path
            result.pending_command = None
            print(f"Écrit : {out_path}")
    if result.pending_command:
        print("\nÀ exécuter une fois ffmpeg installé :")
        print("  " + subprocess.list2cmdline(result.pending_command))
    elif result.output_path:
        print(f"Livre audio  : {result.output_path}")

    if args.check:
        _report_levels(result.wav_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
