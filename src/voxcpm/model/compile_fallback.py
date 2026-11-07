"""Proactive toolchain detection for ``torch.compile`` fallbacks.

``torch.compile`` is lazy: it returns immediately and Triton only looks for a
C compiler (``$CC`` / ``cc`` / ``gcc`` / ``clang`` / ``cl`` / ``clang-cl``,
or ``triton.knobs.build.impl``)
when the compiled artifact is first built.  Wrapping the *construction* of the
compiled callable in ``try/except`` therefore never catches a missing compiler
and the failure surfaces as an ``InductorError`` during the first ``generate()``.

Detecting the toolchain up front lets callers skip ``torch.compile`` entirely
and stay in eager mode with a warning.  This module is intentionally
stdlib-only so it can be used (and tested) without importing torch.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional


def _triton_build_impl() -> Optional[str]:
    """Return the configured Triton build backend command, if any."""
    try:
        import triton  # type: ignore[import-not-found]  # optional dependency

        knobs = getattr(triton, "knobs", None)
        build = getattr(knobs, "build", None) if knobs is not None else None
        impl = getattr(build, "impl", None) if build is not None else None
        return str(impl) if impl else None
    except Exception:
        return None


def compiler_available() -> bool:
    """True when a C compiler usable by Triton/Inductor can be located.

    Honors, in order: ``triton.knobs.build.impl``, the ``CC`` environment
    variable, and the common compiler names on ``PATH``.
    """
    candidates = []
    impl = _triton_build_impl()
    if impl:
        candidates.append(impl.split()[0])
    cc = os.environ.get("CC")
    if cc:
        candidates.append(cc.split()[0])
    # POSIX names first, then the MSVC / clang-cl driver names used on Windows.
    candidates.extend(["cc", "gcc", "clang", "cl", "clang-cl"])
    return any(shutil.which(c) for c in candidates)
