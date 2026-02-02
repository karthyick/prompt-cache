"""
prompt-cache: Semantic caching for LLM API calls.

Cut LLM costs 30% with one decorator.
"""

__version__ = "0.1.0"
__author__ = "Karthick Raja M"
__license__ = "MIT"

# Core exports
from prompt_cache.config import CacheConfig
from prompt_cache.core import (
    CacheContext,
    CachedLLM,
    cache,
    get_default_backend,
    set_default_backend,
)
from prompt_cache.exceptions import (
    CacheBackendError,
    CacheNotFoundError,
    CacheSerializationError,
    PromptCacheError,
)
from prompt_cache.stats import CacheStats, clear_cache, get_stats, invalidate
from prompt_cache.storage import StorageBackend

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core API
    "cache",
    "CacheContext",
    "CachedLLM",
    "get_default_backend",
    "set_default_backend",
    # Storage
    "StorageBackend",
    # Statistics
    "CacheStats",
    "get_stats",
    "clear_cache",
    "invalidate",
    # Configuration
    "CacheConfig",
    # Exceptions
    "PromptCacheError",
    "CacheBackendError",
    "CacheSerializationError",
    "CacheNotFoundError",
]
