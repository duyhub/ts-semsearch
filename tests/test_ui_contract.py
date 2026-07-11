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
    # The denied/unavailable status is just the action, never a fake "default
    # location" claim — the camera frames results, not DEFAULT_MAP_FOCUS.
    assert "Mặc định:" not in ui
    assert "setLocationStatus('default','Bật vị trí')" in ui
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


def test_map_bounds_prefer_results_over_display_fallback(ui):
    """The camera fits the result markers; the display-only fallback is a last resort.

    Regression guard: the no-anchor bounds ladder must reach for result-derived
    points before ever parking on `displayFocus` (the sticky 'Trung tâm Hà Nội'
    display center). If the fallback wins, result pins fall off-viewport.
    The pts branch frames the <=25 km cluster around the AI's #1 result
    (`framePts`) and must keep the whole-pts guard as its own fallback.
    """
    bounds_expr = re.search(r"let bounds =(?P<expr>.*?);", ui, re.S)
    assert bounds_expr, "bounds ternary not found in renderMap"
    expr = bounds_expr.group("expr")
    assert "pts.length ? (framePts.length ? framePts : pts)" in expr
    # result-derived points must be preferred strictly before the displayFocus fallback branch.
    assert "displayFocus" in expr
    assert expr.index("pts.length") < expr.index("displayFocus")
    # framePts is the cluster around the top result, cut with the same 25 km haversine
    # pattern the anchor branch uses — never a fabricated anchor marker.
    frame = re.search(r"const framePts=(?P<def>[^\n]*)", ui)
    assert frame, "framePts cluster filter not found in renderMap"
    assert "haversineKm" in frame.group("def") and "<=25" in frame.group("def")


def test_map_empty_state_accounts_for_results(ui):
    """The '#mapEmpty' card can never show while results with coordinates exist.

    The empty-state / marker-strip early return keys off `hasFocus`; that flag must
    fold in whether any result carries coordinates, so a geolocation-pending window
    (no anchor, no display fallback yet) still renders the result pins.
    """
    assert "const hasResults=results.some(r=>r.coordinates)" in ui
    has_focus = re.search(r"const hasFocus=Boolean\((?P<args>[^)]*)\)", ui)
    assert has_focus, "hasFocus computation not found in renderMap"
    assert "hasResults" in has_focus.group("args")
    # The empty card is still driven by the negation of that same flag.
    assert re.search(r"#mapEmpty['\"]\)\.classList\.toggle\(\s*['\"]show['\"]\s*,\s*!hasFocus", ui)


def test_default_focus_legend_is_truthful(ui):
    """The 'bản đồ mặc định' legend appears only when the fallback drives the camera.

    When result points frame the map, the display fallback is not the camera focus,
    so the legend must stay silent about it (matching the pre-regression behavior).
    """
    assert re.search(
        r"else if\(displayFocus\s*&&\s*!pts\.length\)\s*\{", ui
    ), "default-focus legend must be guarded by an empty result set"


# --------------------------------------------------------------------------- #
# Demo-day polish invariants (added with the wow pass — extend, never weaken)  #
# --------------------------------------------------------------------------- #
def test_guided_tour_chips_carry_capability_tags(ui):
    """Each guided-tour chip renders a capability tag above the query text."""
    assert 'class="chiptag"' in ui and 'class="chipq"' in ui
    assert re.search(r"CHIPS\s*=\s*\[", ui)
    assert re.search(r"\btag\s*:", ui) and re.search(r"\bq\s*:", ui)
    # the flagship no-diacritics hunger query is a first-class chip (fires the correction beat)
    assert "minh doi bung qua" in ui
    # autoload + empty-state chip read the query field, not the whole object
    assert "CHIPS[0].q" in ui


def test_corrected_query_diff_marks_repaired_tokens(ui):
    """After the (contract-pinned) hint line, changed tokens get <mark class="fix">."""
    assert 'mark class="fix"' in ui          # the per-token diff span
    assert "mark.fix" in ui                   # its accent-underline styling
    assert re.search(r"raw\.length===cor\.length", ui)  # equal-length fallback guard


def test_compare_rank_badges_are_directional(ui):
    """Compare cards carry a directional rank-move badge (up/down/new/same)."""
    assert "data-move=" in ui
    assert "rank-change.up" in ui and "rank-change.new" in ui and "rank-change.down" in ui
    assert "AI tìm thêm" in ui
    # the human-readable delta fragments survive (also guarded by the rank-change regex above)
    assert "Lên ${rankDelta} hạng" in ui and "Xuống ${Math.abs(rankDelta)} hạng" in ui


def test_product_mode_top_pick_ribbon(ui):
    """Product mode crowns the AI's #1 pick; gated on product mode + first index."""
    assert "toppick" in ui and "AI đề xuất" in ui
    assert re.search(r"state\.mode==='product'\s*&&\s*i===0", ui)


def test_signal_bars_grow_on_disclosure_and_crown_lead(ui):
    """Signal bars render empty then sweep to their value on open; top row is crowned."""
    assert re.search(r'sigfill"\s+data-w=', ui)   # target width stashed, bar starts empty
    assert 'style="width:0' in ui
    assert ".sigc.ct.lead" in ui                   # crowned biggest-contribution cell
    assert "f.dataset.w" in ui                      # swept in via JS on disclosure


def test_latency_badge_counts_up(ui):
    assert "function countUp(" in ui
    assert "countUp($('#latVal')" in ui


def test_mobile_app_shell_and_touch_targets(ui):
    """At <=560px the chrome pins (page-scroll overrides removed) with >=44px targets + safe-area."""
    # app-shell revert: the page-scroll overrides are gone, base 100dvh/hidden shell takes over
    assert "min-height:66vh" not in ui
    assert "height:auto; min-height:100dvh; overflow:auto" not in ui
    assert "height:100dvh; overflow:hidden" in ui
    # thumb targets + notch/home-indicator safe areas
    assert "min-height:44px" in ui
    assert "env(safe-area-inset-top)" in ui and "env(safe-area-inset-bottom)" in ui


def test_mobile_compare_is_swipeable(ui):
    """Compare mode on a phone is a horizontal scroll-snap swipe track (AI lane leads)."""
    assert re.search(r'#app\[data-mode="compare"\]\{[^}]*scroll-snap-type:x mandatory', ui)
    assert re.search(r'#app\[data-mode="compare"\] \.pane\{[^}]*scroll-snap-align', ui)


def test_reduced_motion_neutralizes_new_staggers(ui):
    """The reduced-motion guard zeros the new per-item stagger delays, not only durations."""
    guard = re.search(r"@media \(prefers-reduced-motion:reduce\)\{(?P<b>.*?)\n\s*\}", ui, re.S)
    assert guard
    body = guard.group("b")
    assert "animation-delay:0" in body and "transition-delay:0" in body


def test_offline_map_texture_and_loading_beat(ui):
    """Tile errors paint a deliberate offline texture; loading reinforces intent-understanding."""
    assert "#mapCanvas.offline" in ui
    assert "classList.add('offline')" in ui
    assert "Đang hiểu ý định" in ui


def test_denied_location_proximity_query_is_acknowledged(ui):
    """A 'near me' query with no coords and no anchor must say so, not silently go nationwide."""
    assert "Bật vị trí để xếp theo khoảng cách gần bạn" in ui
    # gated on: location not used, no resolved anchor, and the query actually asking for proximity
    assert re.search(r"!state\.usedLocation\s*&&\s*!\(state\.intent\|\|\{\}\)\.anchor", ui)
    assert re.search(r"gần đây\|quanh đây\|gần bạn", ui)
    # no chip tag may promise personal proximity that denied-location cannot deliver
    assert "gần bạn 24/7" not in ui


def test_mobile_compare_swipe_affordance(ui):
    """The hidden keyword lane is hinted (dismissed after the first swipe), never unhinted."""
    assert _has_id(ui, "swipeHint")
    assert "vuốt để so sánh" in ui
    assert re.search(r'#app\[data-mode="compare"\] \.swipehint\{[^}]*display:inline-flex', ui)
    assert re.search(r"scrollLeft>\d+.*swipeHint.*classList\.add\(['\"]gone", ui)


def test_desktop_chip_rail_is_mouse_scrollable_and_compact(ui):
    """Wide viewports compact the rail (all 7 chips at 1440) and wheel scrolls it horizontally."""
    assert "@media (min-width:901px)" in ui
    wheel = re.search(r"addEventListener\(\s*['\"]wheel['\"]\s*,(?P<b>.*?)\{passive:true\}\)", ui, re.S)
    assert wheel, "wheel-to-horizontal handler missing on the chip rail"
    assert "scrollLeft" in wheel.group("b") and "deltaY" in wheel.group("b")
    # only when the rail actually overflows
    assert "scrollWidth<=bar.clientWidth" in wheel.group("b")


def test_score_panel_weight_column_is_normalized(ui):
    """Displayed weights are normalized shares, consistent with the contribution math."""
    assert "${f2(wt/totalW)}" in ui
    # raw (un-normalized) weight must not be rendered
    assert "${f2(wt)}" not in ui
