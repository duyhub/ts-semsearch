"""Adversarial input set for the robustness sweep (SPEC §11; PRD NFR-2, gate G5).

Kept in one place so the sweep script and the test use the identical list.
"""
from __future__ import annotations

ADVERSARIAL: list[tuple[str, str]] = [
    ("empty", ""),
    ("emoji", "🍜🍕🎉"),
    ("all_caps_no_diacritics", "CAFE WIFI YEN TINH QUAN 1"),
    ("rambling_200_char", (
        "tôi đang tìm một nơi nào đó thật sự yên tĩnh và thoải mái để có thể ngồi làm việc "
        "cả buổi chiều có wifi mạnh nhiều ổ cắm điện cà phê ngon giá hợp lý gần trung tâm "
        "thành phố và không quá đông người vào cuối tuần")),
    ("pure_english", "a quiet place to work with good coffee and wifi near the city center"),
    ("pure_address", "27 Ngô Đức Kế, Quận 1"),
    ("coordinate_only", "10.7738, 106.704"),
    ("unknown_city", "quán ăn ngon ở Cần Thơ"),
]
