"""Opt-in wall-clock profiler for the eval loop.

Enable with ``ROBODOJO_PROFILE=1``. When disabled, ``span`` is a cheap no-op so
the instrumentation can stay in hot paths.
"""

from collections import defaultdict
from contextlib import contextmanager
import json
import os
import time

ENABLED = os.environ.get("ROBODOJO_PROFILE") == "1"
CPROFILE_ENABLED = os.environ.get("ROBODOJO_CPROFILE") == "1"

_totals: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)


@contextmanager
def span(name: str):
    """Accumulate the wall-clock time spent inside the block under ``name``."""
    if not ENABLED:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        _totals[name] += time.perf_counter() - start
        _counts[name] += 1


@contextmanager
def cprofile(path: str):
    """Run the block under cProfile and append stats to ``path`` (``ROBODOJO_CPROFILE=1``)."""
    if not CPROFILE_ENABLED:
        yield
        return
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            stats = pstats.Stats(path)
            stats.add(profiler)
        else:
            stats = pstats.Stats(profiler)
        stats.dump_stats(path)


def record(name: str, seconds: float) -> None:
    """Add an externally measured duration under ``name``."""
    if ENABLED:
        _totals[name] += seconds
        _counts[name] += 1


def snapshot() -> dict[str, dict[str, float]]:
    """Return spans sorted by total time, largest first."""
    return {
        name: {
            "total_s": round(total, 4),
            "count": _counts[name],
            "mean_ms": round(total / _counts[name] * 1000.0, 3) if _counts[name] else 0.0,
        }
        for name, total in sorted(_totals.items(), key=lambda item: -item[1])
    }


def reset() -> None:
    _totals.clear()
    _counts.clear()


def dump(path: str) -> None:
    """Write the current snapshot to ``path`` when profiling is enabled."""
    if not ENABLED:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(snapshot(), fp, indent=2)
    os.replace(tmp_path, path)
