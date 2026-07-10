"""Optional Langfuse tracing for the LLM/embed calls (sponsor awards teams using it).

Design: a thin, best-effort wrapper that is a *silent no-op* unless both
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present. When the keys are
absent (the default, and every test — there is no network here) nothing is
imported from langfuse and every hook returns immediately, so tracing adds zero
overhead and can never raise an ImportError offline.

The langfuse import lives *inside this module only* (deferred to first use) — call
sites (`llm_parse`, `embeddings`) import this module, never langfuse, so a missing
or broken langfuse install can't break a query. Every emit path is wrapped in a
blanket try/except: a tracing failure is logged at DEBUG and swallowed, never
surfaced to the request path (HARD RULE: tracing must not break the demo).
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"

# langfuse 4.x observation "as_type" values we use: an LLM parse is a generation,
# an embed batch is an embedding.
_GENERATION = "generation"
_EMBEDDING = "embedding"

_client: Any = None  # cached ONLY on successful init; disabled state is never cached


def enabled() -> bool:
    """True iff both Langfuse keys are set. Cheap; re-checked on every hook so the
    disabled path never imports langfuse and never caches an enabled decision."""
    return bool(os.environ.get(PUBLIC_KEY_ENV) and os.environ.get(SECRET_KEY_ENV))


def _get_client() -> Any:
    """Return a cached Langfuse client, or None when disabled/unavailable. The
    langfuse import is deferred here so importing this module (and the call sites)
    stays free offline."""
    global _client
    if _client is not None:
        return _client
    if not enabled():
        return None
    try:
        from langfuse import Langfuse  # deferred: only imported when keys are present

        _client = Langfuse()  # reads LANGFUSE_* from the environment
    except Exception:  # noqa: BLE001 - a broken langfuse must not break a query
        logger.debug("Langfuse init failed; tracing disabled.", exc_info=True)
        return None
    return _client


class _Handle:
    """Wraps a langfuse observation so call sites can attach output/metadata via
    `.update(...)`. The no-op instance (`_NOOP`) swallows every call."""

    __slots__ = ("_span",)

    def __init__(self, span: Any = None) -> None:
        self._span = span

    def update(self, **kwargs: Any) -> None:
        if self._span is None:
            return
        try:
            self._span.update(**kwargs)
        except Exception:  # noqa: BLE001
            logger.debug("Langfuse span.update failed", exc_info=True)


_NOOP = _Handle(None)


@contextmanager
def traced(
    name: str,
    *,
    kind: str = _GENERATION,
    model: str | None = None,
    input: Any = None,
    metadata: dict | None = None,
) -> Iterator[_Handle]:
    """Emit one Langfuse observation around a block, or a zero-overhead no-op when
    tracing is disabled. Yields a handle; call `handle.update(output=..., metadata=...)`
    to attach results. Records wall-clock latency in metadata on exit. Never raises."""
    client = _get_client()
    if client is None:
        yield _NOOP
        return
    start = time.perf_counter()
    cm = None
    handle = _NOOP
    try:
        cm = client.start_as_current_observation(
            name=name, as_type=kind, model=model, input=input, metadata=metadata
        )
        handle = _Handle(cm.__enter__())
    except Exception:  # noqa: BLE001
        logger.debug("Langfuse observation start failed", exc_info=True)
        cm = None
    try:
        yield handle
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        handle.update(metadata={"latency_ms": latency_ms})
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                logger.debug("Langfuse observation exit failed", exc_info=True)


def flush() -> None:
    """Flush buffered traces on shutdown. Best effort; no-op when disabled."""
    client = _get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:  # noqa: BLE001
        logger.debug("Langfuse flush failed", exc_info=True)
