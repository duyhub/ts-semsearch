"""Tests for the level-configurable per-request logging (T6)."""
from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from semsearch.api import create_app
from semsearch.logging_setup import configure_logging, get_logger


def test_level_from_explicit_arg():
    assert configure_logging("WARNING").level == logging.WARNING


def test_level_from_env(monkeypatch):
    monkeypatch.setenv("SEMSEARCH_LOG_LEVEL", "DEBUG")
    assert configure_logging().level == logging.DEBUG


def test_single_handler_and_no_propagate():
    configure_logging("INFO")
    lg = get_logger()
    before = len(lg.handlers)
    configure_logging("INFO")  # idempotent — must not stack handlers
    assert len(lg.handlers) == before
    assert lg.propagate is False


def test_search_emits_one_info_record():
    configure_logging("INFO")
    lg = get_logger()
    captured: list[logging.LogRecord] = []
    sink = logging.Handler()
    sink.emit = captured.append  # type: ignore[method-assign]
    lg.addHandler(sink)
    try:
        client = TestClient(create_app(prewarm=False))
        client.get("/v1/search", params={"q": "cà phê"})
        assert any("search q=" in r.getMessage() for r in captured)
    finally:
        lg.removeHandler(sink)
