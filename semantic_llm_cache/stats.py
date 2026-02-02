"""Statistics and analytics for llm-semantic-cache."""

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Optional

from semantic_llm_cache.backends import MemoryBackend
from semantic_llm_cache.backends.base import BaseBackend


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    hits: int = 0
    misses: int = 0
    total_saved_ms: float = 0.0
    estimated_savings_usd: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate between 0 and 1
        """
        total = self.hits + self.misses
        return self.hits / max(total, 1)

    @property
    def total_requests(self) -> int:
        """Get total number of requests.

        Returns:
            Total hits + misses
        """
        return self.hits + self.misses

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of stats
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "total_requests": self.total_requests,
            "total_saved_ms": self.total_saved_ms,
            "estimated_savings_usd": self.estimated_savings_usd,
        }

    def __iadd__(self, other: "CacheStats") -> "CacheStats":
        """Add another stats object to this one.

        Args:
            other: Other CacheStats to add

        Returns:
            self
        """
        self.hits += other.hits
        self.misses += other.misses
        self.total_saved_ms += other.total_saved_ms
        self.estimated_savings_usd += other.estimated_savings_usd
        return self


# Global stats manager
class _StatsManager:
    """Thread-safe manager for global cache statistics."""

    def __init__(self) -> None:
        """Initialize stats manager."""
        self._stats: dict[str, CacheStats] = {}
        self._lock = Lock()
        self._default_backend: Optional[BaseBackend] = None

    def get_backend(self) -> BaseBackend:
        """Get default backend for cache operations.

        Returns:
            Default backend (creates if needed)
        """
        if self._default_backend is None:
            self._default_backend = MemoryBackend()
        return self._default_backend

    def set_backend(self, backend: BaseBackend) -> None:
        """Set default backend for cache operations.

        Args:
            backend: Backend to use as default
        """
        with self._lock:
            self._default_backend = backend

    def record_hit(
        self,
        namespace: str,
        latency_saved_ms: float = 0.0,
        saved_cost: float = 0.0,
    ) -> None:
        """Record a cache hit.

        Args:
            namespace: Cache namespace
            latency_saved_ms: Latency saved by cache hit (ms)
            saved_cost: Estimated cost savings (USD)
        """
        with self._lock:
            if namespace not in self._stats:
                self._stats[namespace] = CacheStats()

            stats = self._stats[namespace]
            stats.hits += 1
            stats.total_saved_ms += latency_saved_ms
            stats.estimated_savings_usd += saved_cost

    def record_miss(self, namespace: str) -> None:
        """Record a cache miss.

        Args:
            namespace: Cache namespace
        """
        with self._lock:
            if namespace not in self._stats:
                self._stats[namespace] = CacheStats()

            self._stats[namespace].misses += 1

    def get_stats(self, namespace: Optional[str] = None) -> CacheStats:
        """Get statistics for namespace or all.

        Args:
            namespace: Optional namespace filter

        Returns:
            CacheStats object
        """
        with self._lock:
            if namespace is not None:
                return self._stats.get(namespace, CacheStats())

            # Aggregate all namespaces
            total = CacheStats()
            for stats in self._stats.values():
                total += stats
            return total

    def clear_stats(self, namespace: Optional[str] = None) -> None:
        """Clear statistics for namespace or all.

        Args:
            namespace: Optional namespace to clear (None = all)
        """
        with self._lock:
            if namespace is None:
                self._stats.clear()
            elif namespace in self._stats:
                del self._stats[namespace]


# Global stats manager instance
_stats_manager = _StatsManager()


def get_stats(namespace: Optional[str] = None) -> dict[str, Any]:
    """Get cache statistics.

    Args:
        namespace: Optional namespace to filter by

    Returns:
        Dictionary with cache statistics

    Examples:
        >>> get_stats()
        {
            'hits': 1547,
            'misses': 892,
            'hit_rate': 0.634,
            'total_requests': 2439,
            'total_saved_ms': 773500.0,
            'estimated_savings_usd': 3.09
        }
    """
    stats = _stats_manager.get_stats(namespace)
    return stats.to_dict()


def clear_cache(namespace: Optional[str] = None) -> int:
    """Clear all cached entries.

    Args:
        namespace: Optional namespace to clear (None = all)

    Returns:
        Number of entries cleared

    Examples:
        >>> clear_cache()
        152
        >>> clear_cache(namespace="myapp")
        42
    """
    backend = _stats_manager.get_backend()

    if namespace is None:
        # Get current size before clearing
        stats = backend.get_stats()
        size = stats.get("size", 0)
        backend.clear()
        _stats_manager.clear_stats()
        return size

    # Clear specific namespace
    entries = backend.iterate(namespace=namespace)
    count = len(entries)

    for key, _ in entries:
        backend.delete(key)

    _stats_manager.clear_stats(namespace)
    return count


def invalidate(
    pattern: str,
    namespace: Optional[str] = None,
) -> int:
    """Invalidate cache entries matching pattern.

    Args:
        pattern: String pattern to match in prompts
        namespace: Optional namespace filter

    Returns:
        Number of entries invalidated

    Examples:
        >>> invalidate("Python")
        23
        >>> invalidate("deprecated", namespace="api")
        5
    """
    backend = _stats_manager.get_backend()
    entries = backend.iterate(namespace=namespace)
    count = 0

    pattern_lower = pattern.lower()

    for key, entry in entries:
        if pattern_lower in entry.prompt.lower():
            backend.delete(key)
            count += 1

    return count


def warm_cache(
    prompts: list[str],
    llm_func: Callable[[str], Any],
    namespace: str = "default",
) -> int:
    """Pre-populate cache with prompts.

    Args:
        prompts: List of prompts to cache
        llm_func: LLM function to call for each prompt
        namespace: Cache namespace to use

    Returns:
        Number of prompts cached

    Examples:
        >>> def ask_gpt(prompt):
        ...     return openai.chat.completions.create(...)
        >>> warm_cache(["What is Python?", "Explain AI"], ask_gpt)
        2
    """
    from semantic_llm_cache.core import cache

    # Create a temporary cached function
    cached_func: Callable[[str], Any] = cache(namespace=namespace)(llm_func)

    for prompt in prompts:
        try:
            cached_func(prompt)
        except Exception:
            # Skip failed prompts
            pass

    return len(prompts)


def export_cache(
    namespace: Optional[str] = None,
    filepath: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Export cache entries for analysis.

    Args:
        namespace: Optional namespace filter
        filepath: Optional file path to save export (JSON)

    Returns:
        List of cache entry dictionaries

    Examples:
        >>> data = export_cache()
        >>> export_cache(filepath="cache_export.json")
    """
    import json
    from datetime import datetime

    backend = _stats_manager.get_backend()
    entries = backend.iterate(namespace=namespace)

    export_data = []
    for key, entry in entries:
        export_data.append({
            "key": key,
            "prompt": entry.prompt,
            "response": str(entry.response)[:1000],  # Truncate large responses
            "namespace": entry.namespace,
            "hit_count": entry.hit_count,
            "created_at": datetime.fromtimestamp(entry.created_at).isoformat(),
            "ttl": entry.ttl,
            "input_tokens": entry.input_tokens,
            "output_tokens": entry.output_tokens,
        })

    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

    return export_data
