"""
Shared helpers for routing the stdout/stderr of a ``__main__`` entrypoint into
a log file with a clear section header.

Two capture helpers are provided:

* ``capture_to_extended_log`` -- writes into the shared project-wide
  ``output_extended.txt``. This is the legacy path used by ``main.py`` as the
  canonical "fresh-run" entrypoint.

* ``capture_to_test_log`` -- derives a per-test filename from the caller's
  ``__file__`` (e.g. ``test_foo.py`` -> ``output_test_foo.txt``) and truncates
  it on each run so each test produces its own standalone report. This is the
  recommended helper for every ``test_*.py`` module.

Typical usage in a test module:

    if __name__ == "__main__":
        from test_output import capture_to_test_log
        with capture_to_test_log(__file__):
            unittest.main(verbosity=2, exit=False)
"""

from __future__ import annotations

import contextlib
import datetime
import os
import sys
from pathlib import Path
from typing import Iterator, Union

HEADER_RULE: str = "=" * 80
REPO_ROOT: Path = Path(__file__).resolve().parent
OUTPUT_EXTENDED_PATH: Path = REPO_ROOT / "output_extended.txt"


def _resolve_section_label(section: Union[str, "os.PathLike[str]"]) -> str:
    """
    Normalize a section identifier (``__file__`` or free-form string) into a
    compact, human-readable label for the log header.

    Args:
        section: Either a filesystem path (typically ``__file__``) or a
            descriptive string.

    Returns:
        A short label such as ``"test_global_operations.py"``.
    """
    section_str = os.fspath(section)
    if os.sep in section_str or section_str.endswith(".py"):
        return Path(section_str).name
    return section_str


def _resolve_per_test_log_path(section: Union[str, "os.PathLike[str]"]) -> Path:
    """
    Derive the per-test log path from the caller's ``__file__`` (or any label).

    Examples:
        ``.../test_global_operations.py`` -> ``REPO_ROOT / "output_test_global_operations.txt"``
        ``"combined_warp"``              -> ``REPO_ROOT / "output_combined_warp.txt"``

    Args:
        section: Either a filesystem path (typically ``__file__``) or a
            descriptive string.

    Returns:
        Absolute path to the per-test log file inside the repo root.
    """
    label = _resolve_section_label(section)
    stem = Path(label).stem if label.endswith(".py") else label
    return REPO_ROOT / f"output_{stem}.txt"


@contextlib.contextmanager
def _capture_stdout_streams_to(
    log_path: Path,
    *,
    section: Union[str, "os.PathLike[str]"],
    truncate: bool,
    announce_on_real_stdout: bool,
    announce_verb: str,
) -> Iterator[Path]:
    """
    Shared implementation: route stdout/stderr into ``log_path`` for the
    duration of the ``with`` block, bracketed by a labeled section header.
    """
    mode = "w" if truncate else "a"
    label = _resolve_section_label(section)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    with log_path.open(mode, encoding="utf-8") as log_file:
        log_file.write(f"\n{HEADER_RULE}\n")
        log_file.write(f"== {label}  ({timestamp})\n")
        log_file.write(f"{HEADER_RULE}\n\n")
        log_file.flush()

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = log_file
        sys.stderr = log_file
        try:
            yield log_path
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    if announce_on_real_stdout:
        print(f"{label} output {announce_verb} {log_path}")


@contextlib.contextmanager
def capture_to_extended_log(
    section: Union[str, "os.PathLike[str]"],
    *,
    truncate: bool = False,
    announce_on_real_stdout: bool = True,
) -> Iterator[Path]:
    """
    Redirect stdout/stderr into ``output_extended.txt`` for the duration of
    the ``with`` block, bracketed by a labeled section header.

    This is the legacy shared log used by ``main.py``. Test modules should
    prefer :func:`capture_to_test_log`, which writes to a per-test file.

    Args:
        section: Free-form label or ``__file__``-style path identifying the
            caller. The filename (or full string) is rendered into the header.
        truncate: When ``True``, overwrite the log. Defaults to ``False``
            (append), which keeps multiple runs composed into a single report.
        announce_on_real_stdout: When ``True``, print a short confirmation to
            the real stdout after the block exits so interactive users know
            where the output went.

    Yields:
        The absolute path to the log file being written.
    """
    with _capture_stdout_streams_to(
        OUTPUT_EXTENDED_PATH,
        section=section,
        truncate=truncate,
        announce_on_real_stdout=announce_on_real_stdout,
        announce_verb="appended to" if not truncate else "written to",
    ) as path:
        yield path


@contextlib.contextmanager
def capture_to_test_log(
    section: Union[str, "os.PathLike[str]"],
    *,
    truncate: bool = True,
    announce_on_real_stdout: bool = True,
) -> Iterator[Path]:
    """
    Redirect stdout/stderr into a per-test log file derived from ``section``.

    The log file is named ``output_<stem>.txt`` in the repo root, where
    ``<stem>`` comes from the caller's ``__file__`` (``test_foo.py`` ->
    ``output_test_foo.txt``). By default the file is truncated on each run
    so every invocation produces a fresh standalone report for that test.

    Args:
        section: Free-form label or ``__file__``-style path identifying the
            caller.
        truncate: When ``True`` (default), overwrite the per-test log.
        announce_on_real_stdout: When ``True``, print a short confirmation to
            the real stdout after the block exits.

    Yields:
        The absolute path to the per-test log file being written.
    """
    log_path = _resolve_per_test_log_path(section)
    with _capture_stdout_streams_to(
        log_path,
        section=section,
        truncate=truncate,
        announce_on_real_stdout=announce_on_real_stdout,
        announce_verb="written to" if truncate else "appended to",
    ) as path:
        yield path
