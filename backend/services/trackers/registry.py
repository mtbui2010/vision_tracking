"""Tracker registry — single place to construct a tracker by name.

Used by the eval script, the comparison API, and the frontend dropdowns.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import BaseTracker


_REGISTRY: dict[str, Callable[..., BaseTracker]] = {}


def register(name: str, factory: Callable[..., BaseTracker]) -> None:
    _REGISTRY[name] = factory


def available() -> list[str]:
    return sorted(_REGISTRY.keys())


def build(name: str, **kwargs) -> BaseTracker:
    if name not in _REGISTRY:
        raise KeyError(f"unknown tracker '{name}'. Available: {available()}")
    return _REGISTRY[name](**kwargs)


def _autoload() -> None:
    from .bytetrack import ByteTrack
    from .custom import CustomTracker
    from .deepsort import DeepSORT
    from .sort import SORT

    register("sort", SORT)
    register("deepsort", DeepSORT)
    register("bytetrack", ByteTrack)
    register("custom", CustomTracker)


_autoload()
