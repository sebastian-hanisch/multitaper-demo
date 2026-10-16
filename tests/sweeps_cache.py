"""Die Experimente je Prozess nur einmal rechnen (feste Ströme: das Ergebnis ist immer dasselbe)."""

import functools

import mtp_evaluation as ev


@functools.lru_cache(maxsize=None)
def get(name: str):
    return getattr(ev, "sweep_" + name)()
