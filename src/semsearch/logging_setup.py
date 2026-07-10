"""Minimal, level-configurable logging for the search service (autoplan reframe).

Deliberately small: ONE structured INFO record per request (emitted by the API layer),
never per-POI inside the ranking loop — so it cannot regress the warm-p95 gate (G4). There
is no SSE stream and no runtime mutable log endpoint; verbosity is set by the standard
`logging` config or the `SEMSEARCH_LOG_LEVEL` env var (DEBUG/INFO/WARNING/ERROR). This gives
the "how the pipeline ran" transparency the feature asked for without production-observability
machinery the CEO review flagged as low-value for the demo.
"""
from __future__ import annotations

import logging
import os

LOGGER_NAME = "semsearch"
_DEFAULT_LEVEL = "INFO"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def configure_logging(level: str | int | None = None) -> logging.Logger:
    """Configure and return the `semsearch` logger. Level precedence: explicit arg >
    SEMSEARCH_LOG_LEVEL env var > INFO. Idempotent — safe to call once per app boot; a
    single stream handler is attached and reused."""
    resolved = level if level is not None else os.environ.get("SEMSEARCH_LOG_LEVEL", _DEFAULT_LEVEL)
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False  # don't double-emit through the root logger
    logger.setLevel(resolved)
    return logger
