"""Storage backends for prompt-cache."""

from prompt_cache.backends.base import BaseBackend
from prompt_cache.backends.memory import MemoryBackend

try:
    from prompt_cache.backends.sqlite import SQLiteBackend
except ImportError:
    SQLiteBackend = None  # type: ignore

try:
    from prompt_cache.backends.redis import RedisBackend
except ImportError:
    RedisBackend = None  # type: ignore

__all__ = [
    "BaseBackend",
    "MemoryBackend",
    "SQLiteBackend",
    "RedisBackend",
]
