"""Worker-local LRU for Stage II clips/distances payloads and phoneme token ids.

Metadata files are treated as immutable for a run; rebuild the Dataset if they
change. The cache lives on the dataset instance (copied per worker), not in a
process-global dict.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np

DEFAULT_METADATA_CACHE_MAX_ENTRIES = 128
DEFAULT_METADATA_CACHE_MAX_BYTES = 33554432  # 32 MiB
_ALLOWED_METADATA_CACHE_KEYS = frozenset({"max_entries", "max_bytes"})
_CONTAINER_TYPES = (list, tuple, set, frozenset, dict, Mapping, np.ndarray)


def _require_plain_int(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"stage2.metadata_cache.{field} must be an integer >= {minimum}, "
            f"got {value!r}"
        )
    if value < minimum:
        raise ValueError(
            f"stage2.metadata_cache.{field} must be an integer >= {minimum}, "
            f"got {value!r}"
        )
    return value


def resolve_metadata_cache_limits(
    config: Mapping[str, Any] | None,
) -> tuple[int, int]:
    """Return ``(max_entries, max_bytes)``. ``None`` disables the LRU."""
    if config is None:
        return 0, DEFAULT_METADATA_CACHE_MAX_BYTES
    if not isinstance(config, Mapping):
        raise ValueError("stage2.metadata_cache must be a mapping")
    unknown = sorted(set(config) - _ALLOWED_METADATA_CACHE_KEYS)
    if unknown:
        raise ValueError(
            "Unknown stage2.metadata_cache fields: " + ", ".join(unknown)
        )
    max_entries = _require_plain_int(
        config.get("max_entries", DEFAULT_METADATA_CACHE_MAX_ENTRIES),
        field="max_entries",
        minimum=0,
    )
    max_bytes = _require_plain_int(
        config.get("max_bytes", DEFAULT_METADATA_CACHE_MAX_BYTES),
        field="max_bytes",
        minimum=1,
    )
    return max_entries, max_bytes


def estimate_nbytes(obj: Any, *, _seen: set[int] | None = None) -> int:
    """Recursive payload size with identity dedup; not just ``ndarray.nbytes``."""
    if _seen is None:
        _seen = set()
    if not isinstance(obj, _CONTAINER_TYPES):
        return sys.getsizeof(obj)
    obj_id = id(obj)
    if obj_id in _seen:
        return 0
    _seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, np.ndarray):
        if obj.base is None:
            # CPython/numpy getsizeof already includes an owned buffer; only
            # add nbytes when the header-only report would miss the payload.
            n_bytes = int(obj.nbytes)
            if size < n_bytes:
                size += n_bytes
        else:
            size += estimate_nbytes(obj.base, _seen=_seen)
        if obj.dtype.hasobject:
            for item in obj.flat:
                size += estimate_nbytes(item, _seen=_seen)
        return size
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            size += estimate_nbytes(key, _seen=_seen)
            size += estimate_nbytes(value, _seen=_seen)
        return size
    for item in obj:
        size += estimate_nbytes(item, _seen=_seen)
    return size


class MetadataLRUCache:
    """Dual-capped LRU. ``max_entries=0`` never stores; oversized items are skipped."""

    def __init__(self, *, max_entries: int, max_bytes: int) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._store: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self._nbytes = 0

    def get(self, key: Hashable) -> Any | None:
        if self.max_entries <= 0 or key not in self._store:
            return None
        self._store.move_to_end(key)
        value, _size = self._store[key]
        return value

    def put(self, key: Hashable, value: Any) -> bool:
        if self.max_entries <= 0:
            return False
        size = estimate_nbytes(key) + estimate_nbytes(value)
        if key in self._store:
            self._discard(key)
        if size > self.max_bytes:
            return False
        while self._store and (
            len(self._store) >= self.max_entries or self._nbytes + size > self.max_bytes
        ):
            self._evict()
        if len(self._store) >= self.max_entries or self._nbytes + size > self.max_bytes:
            return False
        self._store[key] = (value, size)
        self._nbytes += size
        return True

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def _discard(self, key: Hashable) -> None:
        _value, size = self._store.pop(key)
        self._nbytes -= size

    def _evict(self) -> None:
        _key, (_value, size) = self._store.popitem(last=False)
        self._nbytes -= size
