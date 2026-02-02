"""Core cache decorator and API for llm-semantic-cache."""

import functools
import time
from typing import Any, Callable, Optional, ParamSpec, TypeVar

from semantic_llm_cache.backends import MemoryBackend
from semantic_llm_cache.backends.base import BaseBackend
from semantic_llm_cache.config import CacheConfig, CacheEntry
from semantic_llm_cache.exceptions import PromptCacheError
from semantic_llm_cache.similarity import (
    EmbeddingCache,
)
from semantic_llm_cache.stats import _stats_manager
from semantic_llm_cache.utils import hash_prompt, normalize_prompt

P = ParamSpec("P")
R = TypeVar("R")


class CacheContext:
    """Context manager for cache configuration.

    Allows temporary override of cache settings within a scope.

    Examples:
        >>> with CacheContext(similarity=0.9) as ctx:
        ...     result = llm_call("prompt")
        ...     print(ctx.stats)
    """

    def __init__(
        self,
        similarity: Optional[float] = None,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Initialize cache context.

        Args:
            similarity: Similarity threshold (overrides default)
            ttl: Time-to-live in seconds (overrides default)
            namespace: Cache namespace (overrides default)
            enabled: Whether caching is enabled (overrides default)
        """
        self._config = CacheConfig(
            similarity_threshold=similarity if similarity is not None else 1.0,
            ttl=ttl,
            namespace=namespace if namespace is not None else "default",
            enabled=enabled if enabled is not None else True,
        )
        self._stats: dict[str, Any] = {"hits": 0, "misses": 0}

    def __enter__(self) -> "CacheContext":
        """Enter context.

        Returns:
            self
        """
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context.

        Args:
            *args: Exception info (unused)
        """
        pass

    @property
    def stats(self) -> dict[str, Any]:
        """Get context statistics.

        Returns:
            Dictionary with hits and misses
        """
        return self._stats.copy()

    @property
    def config(self) -> CacheConfig:
        """Get context configuration.

        Returns:
            CacheConfig object
        """
        return self._config


class CachedLLM:
    """Wrapper class for LLM calls with automatic caching.

    Provides a simple class-based API for caching LLM interactions.

    Examples:
        >>> llm = CachedLLM(provider="openai", similarity=0.9)
        >>> response = llm.chat("What is Python?")
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        similarity: float = 1.0,
        ttl: Optional[int] = 3600,
        backend: Optional[BaseBackend] = None,
        namespace: str = "default",
        enabled: bool = True,
    ) -> None:
        """Initialize cached LLM wrapper.

        Args:
            provider: LLM provider name (for reference)
            model: Model name (for reference)
            similarity: Similarity threshold for semantic matching
            ttl: Time-to-live in seconds
            backend: Storage backend (None = in-memory)
            namespace: Cache namespace
            enabled: Whether caching is enabled
        """
        self._provider = provider
        self._model = model
        self._backend = backend or MemoryBackend()
        self._embedding_cache = EmbeddingCache()
        self._config = CacheConfig(
            similarity_threshold=similarity,
            ttl=ttl,
            namespace=namespace,
            enabled=enabled,
        )

    def _call_llm(self, prompt: str, **kwargs: Any) -> Any:
        """Make actual LLM API call.

        Args:
            prompt: Prompt to send
            **kwargs: Additional arguments

        Returns:
            LLM response

        Raises:
            NotImplementedError: Subclasses should implement this
        """
        raise NotImplementedError(
            "Subclasses should implement _call_llm or use chat() method directly"
        )

    def chat(
        self,
        prompt: str,
        llm_func: Optional[Callable[[str], Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Get response with caching.

        Args:
            prompt: Input prompt
            llm_func: LLM function to call on cache miss
            **kwargs: Additional arguments for llm_func

        Returns:
            LLM response (cached or fresh)
        """
        if llm_func is None:
            llm_func = self._call_llm

        # Create cached function on-the-fly
        @cache(
            similarity=self._config.similarity_threshold,
            ttl=self._config.ttl,
            backend=self._backend,
            namespace=self._config.namespace,
            enabled=self._config.enabled,
        )
        def _cached_call(p: str) -> Any:
            return llm_func(p, **kwargs)

        return _cached_call(prompt)


def cache(
    similarity: float = 1.0,
    ttl: Optional[int] = 3600,
    backend: Optional[BaseBackend] = None,
    namespace: str = "default",
    enabled: bool = True,
    key_func: Optional[Callable[..., str]] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for caching LLM function responses.

    Supports exact match (similarity=1.0) and semantic matching (similarity<1.0).

    Args:
        similarity: Cosine similarity threshold (1.0=exact, 0.9=semantic)
        ttl: Time-to-live in seconds (None=forever)
        backend: Storage backend (None=in-memory)
        namespace: Cache namespace for isolation
        enabled: Whether caching is enabled
        key_func: Custom cache key function

    Returns:
        Decorated function with caching

    Examples:
        >>> @cache()
        ... def ask_gpt(prompt: str) -> str:
        ...     return openai.chat.completions.create(...)

        >>> @cache(similarity=0.9, ttl=3600)
        ... def ask_gpt(prompt: str) -> str:
        ...     return call_openai(prompt)
    """
    # Set up backend and embedding cache
    if backend is None:
        backend = MemoryBackend()

    embedding_cache = EmbeddingCache()

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        """Apply caching to function.

        Args:
            func: Function to decorate

        Returns:
            Decorated function with caching
        """
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Wrapper with caching logic.

            Args:
                *args: Positional arguments
                **kwargs: Keyword arguments

            Returns:
                Cached or fresh result
            """
            if not enabled:
                return func(*args, **kwargs)

            start_time = time.time()

            # Extract prompt from args or kwargs
            prompt: str
            if args and isinstance(args[0], str):
                prompt = args[0]
            elif "prompt" in kwargs:
                prompt = str(kwargs["prompt"])
            else:
                # Use all arguments as key
                prompt = str(args) + str(sorted(kwargs.items()))

            # Normalize prompt
            normalized = normalize_prompt(prompt)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = hash_prompt(normalized, namespace)

            # Try exact match first
            entry = backend.get(cache_key)

            if entry is not None:
                latency_ms = (time.time() - start_time) * 1000
                _stats_manager.record_hit(
                    namespace,
                    latency_saved_ms=latency_ms,
                    saved_cost=entry.estimate_cost(
                        backend.get_stats().get("input_cost_per_1k", 0.001),
                        backend.get_stats().get("output_cost_per_1k", 0.002),
                    ),
                )
                return entry.response

            # Try semantic match if similarity < 1.0
            if similarity < 1.0:
                query_embedding = embedding_cache.encode(normalized)

                result = backend.find_similar(
                    query_embedding,
                    threshold=similarity,
                    namespace=namespace,
                )

                if result is not None:
                    key, entry, sim_score = result
                    latency_ms = (time.time() - start_time) * 1000
                    _stats_manager.record_hit(
                        namespace,
                        latency_saved_ms=latency_ms,
                        saved_cost=entry.estimate_cost(0.001, 0.002),
                    )
                    return entry.response

            # Cache miss - call function
            _stats_manager.record_miss(namespace)

            try:
                response = func(*args, **kwargs)
            except Exception as e:
                raise PromptCacheError(f"LLM function call failed: {e}") from e

            # Store in cache
            embedding = None
            if similarity < 1.0:
                embedding = embedding_cache.encode(normalized)

            cache_entry = CacheEntry(
                prompt=normalized,
                response=response,
                embedding=embedding,
                created_at=time.time(),
                ttl=ttl,
                namespace=namespace,
                hit_count=0,
                input_tokens=len(normalized) // 4,  # Rough estimate
                output_tokens=len(str(response)) // 4,  # Rough estimate
            )

            backend.set(cache_key, cache_entry)

            return response

        return wrapper

    return decorator


# Global default backend for utility functions
_default_backend: Optional[BaseBackend] = None


def get_default_backend() -> BaseBackend:
    """Get default storage backend.

    Returns:
        Default backend (creates if needed)
    """
    global _default_backend
    if _default_backend is None:
        _default_backend = MemoryBackend()
    return _default_backend


def set_default_backend(backend: BaseBackend) -> None:
    """Set default storage backend.

    Args:
        backend: Backend to use as default
    """
    global _default_backend
    _default_backend = backend
    _stats_manager.set_backend(backend)


__all__ = [
    "cache",
    "CacheContext",
    "CachedLLM",
    "get_default_backend",
    "set_default_backend",
]
