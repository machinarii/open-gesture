"""Lazy backend registry. Importing this module imports no model library."""

from __future__ import annotations

from typing import Callable

_FACTORIES: dict[str, Callable[[], object]] = {}


def register(key: str, factory: Callable[[], object]) -> None:
    _FACTORIES[key] = factory


def all_keys() -> list[str]:
    return sorted(_FACTORIES)


def get(key: str):
    if key not in _FACTORIES:
        raise KeyError(f"unknown backend {key!r}; known backends: {', '.join(all_keys())}")
    return _FACTORIES[key]()


def availability() -> list[tuple[str, bool, str]]:
    """(key, available, reason) for every backend. Never raises."""
    rows = []
    for key in all_keys():
        try:
            available, reason = get(key).available()
        except Exception as exc:  # an import error is an unavailability, not a crash
            available, reason = False, f"{type(exc).__name__}: {exc}"
        rows.append((key, available, reason))
    return rows


def _register_builtin_backends() -> None:
    """Register lazily: each lambda imports its module only when called."""

    def _lazy(module: str, cls: str):
        def factory():
            import importlib

            return getattr(importlib.import_module(module), cls)()

        return factory

    pkg = "open_gesture_annotate.backends"
    register("face", _lazy(f"{pkg}.face_uniface", "UniFaceBackend"))
    register("aus", _lazy(f"{pkg}.affect_pyfeat", "PyFeatBackend"))
    register("va", _lazy(f"{pkg}.affect_hsemotion", "HSEmotionBackend"))
    register("pose", _lazy(f"{pkg}.pose_mediapipe", "MediaPipePoseBackend"))
    register("embed", _lazy(f"{pkg}.embed_clip", "ClipEmbedBackend"))
    register("wholebody", _lazy(f"{pkg}.wholebody_rtmw", "RTMWBackend"))


_register_builtin_backends()
