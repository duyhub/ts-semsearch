"""Stable UX contracts for the single-file demo UI.

These tests intentionally inspect semantic hooks and user-visible copy instead of
pixel values.  The UI is a dependency-free HTML/JS document, so keeping the checks
in pytest gives us useful regression coverage without introducing a browser toolchain.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


UI_PATH = Path(__file__).resolve().parents[1] / "ui" / "index.html"


@pytest.fixture(scope="module")
def ui() -> str:
    return UI_PATH.read_text(encoding="utf-8")


def _has_id(ui: str, element_id: str) -> bool:
    return re.search(rf'\bid=["\']{re.escape(element_id)}["\']', ui) is not None


def test_location_is_explicit_and_recoverable(ui):
    """Opportunistic permission remains visible, actionable, and recoverable."""
    assert _has_id(ui, "locStatus")
    assert 'role="status"' in ui and 'aria-live="polite"' in ui
    assert "Bật vị trí" in ui
    assert "Mặc định:" in ui and "Bật vị trí" in ui
    assert _has_id(ui, "locEnable") or _has_id(ui, "locButton")
    # The same visible button may serve as both initial enable and retry.
    assert _has_id(ui, "locRetry") or _has_id(ui, "locButton") or _has_id(ui, "locEnable")
    assert "geolocation.getCurrentPosition" in ui
    assert re.search(r"(?:locEnable|locButton).*addEventListener\s*\(\s*['\"]click", ui, re.S)


def test_default_map_focus_never_enters_ranking_request(ui):
    """Denied geolocation changes only the map focus, never API lat/lon."""
    assert "DEFAULT_MAP_FOCUS" in ui and "displayFallback" in ui
    location_query = re.search(
        r"function locationQuery\(\)\s*\{(?P<body>.*?)\n\}", ui, re.S
    )
    assert location_query
    body = location_query.group("body")
    assert "locationState.coords" in body
    assert "displayFallback" not in body and "DEFAULT_MAP_FOCUS" not in body
    assert "locationState.coords=null" in ui
    assert "locationState.displayFallback={...DEFAULT_MAP_FOCUS}" in ui
    assert "const displayFocus=" in ui
    assert "chỉ là tâm hiển thị" in ui
    assert "kind==='default'" not in ui  # no fallback target/anchor marker


def test_no_location_map_has_an_actionable_empty_state(ui):
    """The map must explain why proximity is absent instead of showing empty chrome."""
    assert _has_id(ui, "mapEmpty")
    assert re.search(r"(?:Bật|Cho phép).{0,30}vị trí.{0,50}(?:bản đồ|gần bạn)", ui, re.I | re.S)
    assert re.search(r"mapEmpty.{0,300}(?:hidden|display|classList)", ui, re.S)


def test_proximity_summary_and_result_distances_are_visible(ui):
    assert _has_id(ui, "proximitySummary")
    assert "distanceMeters" in ui
    assert re.search(r"function\s+(?:formatDistance|fmtDistance)\s*\(", ui)
    assert re.search(r"(?:gần bạn|quanh bạn|từ vị trí)", ui, re.I)
    # Both search lanes receive exactly the same request-scoped location focus.
    assert ui.count("${loc}") >= 2


def test_comparison_mode_explains_rank_changes(ui):
    """Comparison should say what improved and label reordered results."""
    assert _has_id(ui, "modeCompare")
    assert _has_id(ui, "compareSummary") or "comparison-summary" in ui
    assert "rank-change" in ui or "rankDelta" in ui
    assert re.search(r"(?:tăng|giảm|lên|xuống).{0,12}(?:hạng|bậc)", ui, re.I)
    assert re.search(r"(?:AI|ngữ nghĩa).{0,60}(?:từ khoá|BM25)", ui, re.I | re.S)


def test_explanations_progress_from_reason_to_score_detail(ui):
    """Plain-language reasons lead; the weighted signal table is secondary detail."""
    assert "Vì sao phù hợp" in ui
    assert "Xem cách tính điểm" in ui
    assert "Tín hiệu" in ui and "Trọng số" in ui and "Đóng góp" in ui
    assert "aria-expanded" in ui


def test_corrected_query_is_visible_only_when_returned(ui):
    assert _has_id(ui, "qhint")
    assert "correctedQuery" in ui
    assert "Đã hiểu: «" in ui
    assert "state.corrected ? 'Đã hiểu: «' + state.corrected + '»' : ''" in ui


def test_result_cards_offer_directions_and_save_actions(ui):
    assert "directions-btn" in ui and "Chỉ đường" in ui
    assert "save-btn" in ui and "Lưu" in ui
    assert re.search(r"(?:google\.[^'\"` ]+/maps|maps\.google|dir_action=navigate)", ui, re.I)
    assert "localStorage" in ui
    # Card selection must ignore action clicks instead of unexpectedly moving the map.
    assert re.search(r"closest\([^\n]+(?:directions-btn|result-actions)", ui)


def test_mobile_list_map_controls_are_accessible(ui):
    assert "@media (max-width:900px)" in ui
    assert 'role="group" aria-label="Danh sách hoặc bản đồ"' in ui
    assert _has_id(ui, "viewList") and _has_id(ui, "viewMap")
    assert 'aria-pressed="true">Danh sách' in ui
    assert 'aria-pressed="false">Bản đồ' in ui
    assert "function setView(view)" in ui
    assert "data-view" in ui
