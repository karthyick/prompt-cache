# prompt-cache

**Semantic caching for LLM API calls - reduce costs with one decorator.**

[![PyPI](https://img.shields.io/pypi/v/prompt-cache)](https://pypi.org/project/prompt-cache/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/prompt-cache)](https://pypi.org/project/prompt-cache/)

## Overview

LLM API calls are expensive and slow. In production applications, **20-40% of prompts are semantically identical** but get charged as separate API calls. `prompt-cache` solves this with a simple decorator that:

- ✅ **Caches semantically similar prompts** (not just exact matches)
- ✅ **Reduces API costs by 20-40%**
- ✅ **Returns cached responses in <10ms**
- ✅ **Works with any LLM provider** (OpenAI, Anthropic, local models)
- ✅ **Zero behavior change** - drop-in decorator

## Installation

```bash
# Core (exact match only)
pip install prompt-cache

# With semantic similarity
pip install prompt-cache[semantic]

# With Redis backend
pip install prompt-cache[redis]

# With everything
pip install prompt-cache[all]
```

## Quick Start

### Basic Caching (Exact Match)

```python
from prompt_cache import cache

@cache()
def ask_gpt(prompt: str) -> str:
    return openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

# First call - API hit
ask_gpt("What is Python?")  # $0.002

# Second call - cache hit
ask_gpt("What is Python?")  # FREE, <10ms
```

### Semantic Matching

Match semantically similar prompts (requires `pip install prompt-cache[semantic]`):

```python
from prompt_cache import cache

@cache(similarity=0.90)
def ask_gpt(prompt: str) -> str:
    return call_openai(prompt)

ask_gpt("What is Python?")   # API call
ask_gpt("What's Python?")    # Cache hit (95% similar)
ask_gpt("Explain Python")    # Cache hit (91% similar)
ask_gpt("What is Rust?")     # API call (different topic)
```

### TTL Expiration

```python
from prompt_cache import cache

@cache(ttl=3600)  # 1 hour
def ask_gpt(prompt: str) -> str:
    return call_openai(prompt)
```

### Cache Statistics

```python
from prompt_cache import get_stats

stats = get_stats()
# {
#     "hits": 1547,
#     "misses": 892,
#     "hit_rate": 0.634,
#     "estimated_savings_usd": 3.09,
#     "latency_saved_ms": 773500
# }
```

### Cache Management

```python
from prompt_cache import clear_cache, invalidate

# Clear all cached entries
clear_cache()

# Invalidate specific pattern
invalidate(pattern="Python")
```

## Advanced Usage

### Multiple Cache Backends

```python
from prompt_cache import cache
from prompt_cache.backends import RedisBackend

# Use Redis for distributed caching
backend = RedisBackend(url="redis://localhost:6379")

@cache(backend=backend)
def ask_gpt(prompt: str) -> str:
    return call_openai(prompt)
```

### Context Manager

```python
from prompt_cache import CacheContext

with CacheContext(similarity=0.9) as ctx:
    result1 = any_llm_call("prompt 1")
    result2 = any_llm_call("prompt 2")

print(ctx.stats)  # {"hits": 1, "misses": 1}
```

### Wrapper Class

```python
from prompt_cache import CachedLLM

llm = CachedLLM(
    provider="openai",
    similarity=0.9,
    ttl=3600
)

response = llm.chat("What is Python?")
```

## API Reference

### `@cache()` Decorator

```python
@cache(
    similarity: float = 1.0,      # 1.0 = exact match, 0.9 = semantic
    ttl: int = 3600,              # seconds, None = forever
    backend: Backend = None,      # None = in-memory
    namespace: str = "default",   # isolate different use cases
    enabled: bool = True,         # toggle for debugging
    key_func: Callable = None,    # custom cache key
)
def my_llm_function(prompt: str) -> str:
    ...
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `similarity` | `float` | `1.0` | Cosine similarity threshold (1.0 = exact, 0.9 = semantic) |
| `ttl` | `int \| None` | `3600` | Time-to-live in seconds (None = never expires) |
| `backend` | `Backend` | `None` | Storage backend (None = in-memory) |
| `namespace` | `str` | `"default"` | Isolate different use cases |
| `enabled` | `bool` | `True` | Enable/disable caching |
| `key_func` | `Callable` | `None` | Custom cache key function |

### Utility Functions

```python
from prompt_cache import (
    get_stats,      # Get cache statistics
    clear_cache,    # Clear all cached entries
    invalidate,     # Invalidate by pattern
    warm_cache,     # Pre-populate cache
    export_cache,   # Export for analysis
)
```

## Backends

| Backend | Description | Installation |
|---------|-------------|--------------|
| `MemoryBackend` | In-memory (default) | Built-in |
| `SQLiteBackend` | Persistent storage | Built-in |
| `RedisBackend` | Distributed caching | `pip install prompt-cache[redis]` |

## Performance

| Metric | Value |
|--------|-------|
| Cache hit latency | <10ms |
| Cache miss overhead | <50ms (embedding) |
| Typical hit rate | 25-40% |
| Cost reduction | 20-40% |

## Requirements

- Python >= 3.9
- numpy >= 1.24.0

### Optional Dependencies

- `sentence-transformers >= 2.2.0` (for semantic matching)
- `redis >= 4.0.0` (for Redis backend)
- `openai >= 1.0.0` (for OpenAI embeddings)

## License

MIT License - see [LICENSE](LICENSE) file.

## Author

**Karthick Raja M** ([@karthyick](https://github.com/karthyick))

## Related Packages

- [distill-json](https://pypi.org/project/distill-json/) - JSON compression for LLMs

---

**Cut LLM costs 30% with one decorator.** `pip install prompt-cache`
