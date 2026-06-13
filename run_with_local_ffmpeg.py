import os
import runpy
import sys
from pathlib import Path


def prepend_local_ffmpeg_to_path() -> None:
    project_ffmpeg = Path(__file__).resolve().parent / ".local-ffmpeg" / "ffmpeg.exe"
    if project_ffmpeg.exists():
        os.environ["PATH"] = str(project_ffmpeg.parent) + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", str(project_ffmpeg))
        return

    try:
        import imageio_ffmpeg
    except ImportError:
        return

    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not ffmpeg.exists():
        return

    os.environ["PATH"] = str(ffmpeg.parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", str(ffmpeg))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python run_with_local_ffmpeg.py <script.py> [args...]")
        return 2

    prepend_local_ffmpeg_to_path()

    target = Path(sys.argv[1])
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
