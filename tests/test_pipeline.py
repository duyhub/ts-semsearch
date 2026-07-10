"""Full-pipeline behavior tests (SPEC §5-6)."""
from __future__ import annotations

import pytest

from semsearch.data import content_tokens, load_pois
from semsearch.pipeline import FullPipeline


@pytest.fixture(scope="module")
def pipe():
    return FullPipeline(load_pois())


def test_anchor_gate_keeps_near_anchor_on_top(pipe):
    # "gần hồ gươm" resolves to a Hà Nội anchor; the top results must be near it,
    # never an other-city café 600+ km away (the Phase-9 distance-gate fix).
    _, results = pipe.search("cafe có wifi gần hồ gươm", k=5)
    assert results
    assert all(r.poi.city == "Hà Nội" for r in results[:3])


def test_no_anchor_query_unaffected(pipe):
    # a query with no location anchor still returns quiet work cafés on top
    _, results = pipe.search("quán cà phê yên tĩnh để làm việc", k=3)
    assert all(r.poi.category == "Quán cà phê" for r in results)


def test_category_word_dominates_lineup(pipe):
    # "quán cà phê" + a District-1 location must return coffee shops, not the
    # malls / gas stations / restaurants that share the location tokens
    # ("quan 1 tp hcm"). Before the category signal, non-cafés interleaved the
    # top-3 (mall #2, gas station #5). Runs under DEFAULT_WEIGHTS (the API path).
    _, results = pipe.search("quan ca phe o q1 tphcm", k=5)
    assert results
    assert results[0].poi.category == "Quán cà phê"
    assert all(r.poi.category == "Quán cà phê" for r in results[:3])


def test_pure_category_returns_only_that_category(pipe):
    # "cafe" is a pure category query -> results are ONLY coffee shops, no mall/gas/restaurant.
    _, results = pipe.search("cafe", k=10)
    assert results
    assert all(r.poi.category == "Quán cà phê" for r in results)


def test_pure_location_returns_only_that_district(pipe):
    # "quan 1 tphcm" is a pure location query -> ONLY District 1 / TP.HCM, nothing else.
    _, results = pipe.search("quan 1 tphcm", k=10)
    assert results
    assert all(r.poi.district == "Quận 1" and r.poi.city == "TP.HCM" for r in results)


def test_subject_term_isolates_bun_cha(pipe):
    # "quán bún chả ..." -> only bún-chả places; R003 is the only POI whose text has it.
    _, results = pipe.search("quán bún chả cho khách du lịch", k=10)
    assert results
    assert results[0].poi.poi_id == "R003"
    assert all({"bun", "cha"} <= content_tokens(r.poi) for r in results)


def test_p055_mall_not_banished_by_category(pipe):
    # mis-parse (category -> Nhà hàng) must NOT hard-filter; location (Quận 1) keeps M001.
    _, results = pipe.search("nơi mua sắm có nhiều nhà hàng gần quận 1", k=10)
    ids = [r.poi.poi_id for r in results]
    assert "M001" in ids
    assert all(r.poi.district == "Quận 1" for r in results)  # location constraint honored


def test_superlative_does_not_hijack_to_proper_name(pipe):
    # "quan an ngon nhat" (best restaurant): 'nhất' is rare in POI names (only in
    # "Công viên Thống Nhất"), so BM25 ranks that park #1 and the old subject filter
    # pinned results to it. Dense ranks the park ~45th, so corroboration drops the
    # spurious subject and the category (Nhà hàng) drives the lineup.
    _, results = pipe.search("quan an ngon nhat", k=5)
    assert results
    assert results[0].poi.category == "Nhà hàng"
    assert all(r.poi.category == "Nhà hàng" for r in results)


def test_superlative_on_other_category_not_hijacked(pipe):
    # Same spurious-'nhất' collision, different category: gas stations, not the park.
    _, results = pipe.search("cay xang gan nhat", k=5)
    assert results
    assert all(r.poi.category == "Trạm xăng" for r in results)


def test_genuine_proper_name_query_still_returns_it(pipe):
    # The corroboration gate must NOT harm a real proper-name search: when the user
    # actually wants "Thống Nhất", dense ranks the park top, so it still comes first.
    _, results = pipe.search("công viên thống nhất", k=5)
    assert results
    assert results[0].poi.name == "Công viên Thống Nhất"


def test_cheapest_cafe_ranks_above_priciest(pipe):
    # "cafe rẻ nhất": among the returned cafés, a cheaper one must outrank a pricier one
    # (the price signal honors the affordability intent). Also: no park hijack.
    _, results = pipe.search("cafe rẻ nhất", k=10)
    assert results
    assert all(r.poi.category == "Quán cà phê" for r in results)
    prices = [r.poi.price_level for r in results]
    # the top result is not the most expensive tier present
    assert results[0].poi.price_level <= min(prices) + 1
    # a level-1 café outranks a level-4 café when both are returned
    levels = {r.poi.price_level: i for i, r in enumerate(results)}  # first index per level
    if 1 in levels and 4 in levels:
        assert levels[1] < levels[4]


def test_expensive_intent_prefers_pricier(pipe):
    _, results = pipe.search("nhà hàng sang trọng", k=10)
    assert results
    top_levels = [r.poi.price_level for r in results[:3]]
    assert max(top_levels) >= 3  # pricey restaurants surface for an upscale intent


def test_no_price_word_ranking_unaffected(pipe):
    # constant-neutral property: with no price intent, zeroing the price weight must
    # not change the ranking (price_signal is 0.5 for every POI).
    ids_on = pipe.rank_ids("cafe yên tĩnh để làm việc")
    saved = pipe.ranker.weights
    pipe.ranker.weights = {**saved, "price": 0.0}
    try:
        ids_off = pipe.rank_ids("cafe yên tĩnh để làm việc")
    finally:
        pipe.ranker.weights = saved
    assert ids_on == ids_off


def test_impossible_strict_query_still_nonempty(pipe):
    # a district that doesn't exist -> relaxation keeps the result non-empty (G5).
    assert pipe.rank_ids("cafe quận 99 nowhere")


def test_gibberish_still_nonempty(pipe):
    assert pipe.rank_ids("zzzz qwerty asdf")  # full-corpus ranking never empty
