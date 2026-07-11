#!/usr/bin/env python3
"""Deterministic synthetic-POI stress corpus generator (Worker D).

Builds a 1000-POI xlsx = the official 111 POIs (VERBATIM, same ids) + 889 seeded
synthetic distractors (ids ``SYN0001``..``SYN0889``), written with the SAME sheet
name and columns as the official ``POI_Dataset`` so ``semsearch.data.load_pois``
reads it UNCHANGED.

Determinism: driven ONLY by ``random.Random(seed)`` — no global ``random``, no
``time``, no network. Same seed -> identical logical rows (values AND order).

Integrity (CLAUDE.md / NFR-6): NO eval-query text and NO query->POI mapping is
embedded here. Distractors are plausible because they follow the official data's
DISTRIBUTIONS (geography, categories, attributes, hours), never because they were
engineered to answer any specific evaluation query.

Realism is grounded in the official rows read at runtime (per-city lat/lon boxes,
district lists, city mix, category shape) plus curated Vietnamese name/description
pools below. Synthetic attributes are drawn ONLY from the closed 10-attribute
taxonomy; the official 111 rows keep their free-form attributes verbatim.

    uv run python scripts/gen_synth_pois.py            # -> data/synth/synth_dataset.xlsx
    uv run python scripts/gen_synth_pois.py --n 1000 --seed 20260711
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd

OFFICIAL_XLSX = Path("data/raw/ai_maps_track2_dataset_participants.xlsx")
POI_SHEET = "POI_Dataset"
TAX_SHEET = "Attribute_Taxonomy"

# 12 canonical categories = parse.CATEGORY_KEYWORDS values. Fixed generation order.
CANONICAL_CATEGORIES = [
    "Quán cà phê", "Nhà hàng", "Khách sạn", "Trung tâm thương mại", "ATM", "Trạm xăng",
    "Điểm tham quan", "Bệnh viện", "Rạp phim", "Công viên", "Nhà thuốc", "Trạm sạc điện",
]

# The closed 10-attribute taxonomy -> Vietnamese descriptive phrase. Descriptions MUST
# mention a POI's real attributes (the review signal + dense retrieval read this text).
ATTR_PHRASE: dict[str, str] = {
    "yên tĩnh": "không gian yên tĩnh",
    "wifi": "wifi mạnh và ổn định",
    "phù hợp làm việc": "phù hợp để ngồi làm việc",
    "phù hợp gia đình": "phù hợp cho gia đình có trẻ nhỏ",
    "lãng mạn": "không gian lãng mạn",
    "mở khuya": "mở cửa đến khuya",
    "gần biển": "vị trí gần biển",
    "bãi đỗ xe": "có bãi đỗ xe rộng rãi",
    "check-in": "nhiều góc check-in đẹp",
    "24/7": "phục vụ 24/7",
}

# Coastal-only attribute: never assigned outside the coastal city (no inland "gần biển").
COASTAL_CITY = "Đà Nẵng"

# Vietnamese common cat-noun for descriptions (title-case; lowered on the fly when mid-sentence).
CAT_NOUN: dict[str, str] = {
    "Quán cà phê": "Quán cà phê", "Nhà hàng": "Nhà hàng", "Khách sạn": "Khách sạn",
    "Trung tâm thương mại": "Trung tâm thương mại", "ATM": "Cây ATM", "Trạm xăng": "Trạm xăng",
    "Điểm tham quan": "Điểm tham quan", "Bệnh viện": "Bệnh viện", "Rạp phim": "Rạp chiếu phim",
    "Công viên": "Công viên", "Nhà thuốc": "Nhà thuốc", "Trạm sạc điện": "Trạm sạc xe điện",
}

STREETS = [
    "Nguyễn Trãi", "Lê Lợi", "Trần Hưng Đạo", "Hai Bà Trưng", "Lý Thường Kiệt",
    "Phan Chu Trinh", "Điện Biên Phủ", "Cách Mạng Tháng Tám", "Võ Văn Tần", "Pasteur",
    "Nguyễn Đình Chiểu", "Nguyễn Thị Minh Khai", "Trần Phú", "Bạch Đằng", "Hùng Vương",
    "Lê Duẩn", "Nguyễn Văn Cừ", "Hoàng Diệu", "Phạm Ngũ Lão", "Trần Quốc Toản",
    "Nguyễn Công Trứ", "Lê Thánh Tôn", "Trương Định", "Nguyễn Du", "Phan Đình Phùng",
]

VIBES = [
    "thoáng đãng", "sạch sẽ", "thân thiện", "gần gũi", "hiện đại", "ấm cúng",
    "rộng rãi", "tiện lợi", "dễ tìm", "đông vui", "yên bình", "trẻ trung",
]

# Shared description templates (>= 8 available to every category family), varied phrasing so
# dense embeddings don't collapse. {p1}/{p2} are the POI's actual attribute phrases.
DESC_TEMPLATES = [
    "{cat} nằm ở {district}, {city}, {p1}.",
    "{cat} tại {district} với {p1} và {p2}.",
    "Một {catl} ở {district} được nhiều người ghé, {p1}.",
    "{cat} khu {district} — {p1}, {p2}.",
    "Địa điểm ở {district}, {city}; {p1} và không gian {vibe}.",
    "{cat} khá {vibe} tại {district}, {p2}.",
    "Nằm giữa {district}, {catl} này {p1}.",
    "{cat} quen thuộc của khu {district}, {p1} và {p2}.",
    "Ghé {catl} ở {district} để trải nghiệm {p1}.",
    "{cat} tại {city} — {p1}, không gian {vibe}.",
    "{catc} ở {district} {vibe}, {p2}.",
    "{cat} phục vụ khách khu {district}, {p1}.",
    "{cat} ở {district} nổi bật với {p1} và {p2}.",
    "Điểm đến {vibe} tại {district}, {city}: {p1}.",
]

SECOND_SENTENCES = [
    "Ngoài ra {catl} còn {p}.",
    "Điểm cộng là {p}.",
    "Nơi đây {p}, rất tiện cho khách khu {district}.",
    "Bên cạnh đó, quán {p}.",
]

# Per-category attribute plausibility weights (drawn ONLY from the 10 taxonomy attrs; weights
# derived from official co-occurrence / taxonomy applicable_categories — no contradictions).
ATTR_WEIGHTS: dict[str, dict[str, float]] = {
    "Quán cà phê": {"wifi": 0.85, "yên tĩnh": 0.55, "phù hợp làm việc": 0.55,
                    "check-in": 0.45, "lãng mạn": 0.3, "bãi đỗ xe": 0.35, "mở khuya": 0.25},
    "Nhà hàng": {"phù hợp gia đình": 0.6, "bãi đỗ xe": 0.5, "lãng mạn": 0.4,
                 "gần biển": 0.35, "mở khuya": 0.3, "wifi": 0.25},
    "Khách sạn": {"wifi": 0.85, "phù hợp gia đình": 0.5, "yên tĩnh": 0.5,
                  "gần biển": 0.4, "bãi đỗ xe": 0.45, "lãng mạn": 0.3},
    "Trung tâm thương mại": {"bãi đỗ xe": 0.9, "phù hợp gia đình": 0.6, "check-in": 0.4, "wifi": 0.4},
    "ATM": {"24/7": 0.95, "bãi đỗ xe": 0.4},
    "Trạm xăng": {"24/7": 0.85, "bãi đỗ xe": 0.4},
    "Điểm tham quan": {"check-in": 0.85, "phù hợp gia đình": 0.6, "bãi đỗ xe": 0.3, "gần biển": 0.25},
    "Bệnh viện": {"24/7": 0.9, "bãi đỗ xe": 0.45},
    "Rạp phim": {"bãi đỗ xe": 0.6, "phù hợp gia đình": 0.5, "check-in": 0.3, "mở khuya": 0.3},
    "Công viên": {"phù hợp gia đình": 0.7, "check-in": 0.5, "bãi đỗ xe": 0.35, "yên tĩnh": 0.3},
    "Nhà thuốc": {"mở khuya": 0.55, "24/7": 0.45},
    "Trạm sạc điện": {"bãi đỗ xe": 0.85, "24/7": 0.35, "wifi": 0.25},
}

# Per-category sub_category pools (free-text field; official values + plausible extras).
SUBS: dict[str, dict[str, float]] = {
    "Quán cà phê": {"Coffee": 0.4, "Specialty Coffee": 0.15, "Coffee Chain": 0.15,
                    "Book Cafe": 0.1, "View Cafe": 0.1, "Garden Cafe": 0.1},
    "Nhà hàng": {"Restaurant": 0.35, "Vietnamese Restaurant": 0.2, "Seafood Restaurant": 0.15,
                 "Hotpot": 0.1, "Family Restaurant": 0.1, "Vegetarian Restaurant": 0.05,
                 "Late Night Food": 0.05},
    "Khách sạn": {"Hotel": 0.4, "City Hotel": 0.2, "Business Hotel": 0.15, "Resort": 0.1,
                  "Homestay": 0.1, "Beach Hotel": 0.05},
    "Trung tâm thương mại": {"Mall": 1.0},
    "ATM": {"ATM": 1.0},
    "Trạm xăng": {"Fuel Station": 1.0},
    "Điểm tham quan": {"Landmark": 0.25, "Museum": 0.2, "Temple": 0.2, "Square": 0.15,
                       "Theme Park": 0.12, "Zoo": 0.08},
    "Bệnh viện": {"Hospital": 0.75, "General Hospital": 0.25},
    "Rạp phim": {"Cinema": 1.0},
    "Công viên": {"Park": 0.7, "Theme Park": 0.3},
    "Nhà thuốc": {"Pharmacy": 1.0},
    "Trạm sạc điện": {"EV Charging": 1.0},
}

# Coastal-gated subs: only allowed where geographically plausible.
_COASTAL_SUBS = {"Beach Hotel": {COASTAL_CITY}, "Resort": {COASTAL_CITY, "Đà Lạt"}}

# Name pools: chain brands (reused/real VN chains), standalone cores (brand = the core), and
# per-category name templates using {brand}/{loc}. chain_prob = share of chain-branded names.
NAME_CONFIG: dict[str, dict[str, Any]] = {
    "Quán cà phê": {
        "chain_prob": 0.35,
        "chains": ["Highlands Coffee", "Cộng Cà Phê", "The Coffee House", "Katinat",
                   "Phúc Long", "Trung Nguyên Legend", "Aha Cafe"],
        "templates": ["{brand} {loc}", "{brand} Coffee {loc}"],
        "cores": ["Cà Phê Nắng", "Cà Phê Sân Vườn", "Mộc Coffee", "An Nhiên Cafe",
                  "Góc Phố Coffee", "Cà Phê Cũ", "Nhà Của Mây", "Cà Phê Sách", "Lặng Coffee",
                  "Cà Phê Đỏ", "Thềm Xưa Cafe", "Cà Phê Vườn", "Bụi Coffee", "Cà Phê Yên",
                  "Sương Mai Coffee", "Cà Phê Gió", "Là Cafe", "Cà Phê Thơ", "Cỏ Cây Coffee"],
    },
    "Nhà hàng": {
        "chain_prob": 0.3,
        "chains": ["Pizza 4P's", "Món Huế", "Wrap & Roll", "Sushi House", "King BBQ",
                   "Hotpot Story", "Golden Gate"],
        "templates": ["{brand} {loc}", "Nhà Hàng {brand} {loc}"],
        "cores": ["Nhà Hàng Ngon", "Cơm Tấm Cô Ba", "Bún Chả Hàng Mành", "Phở Gánh",
                  "Lẩu Dê Sáu Miền", "Bếp Nhà Quê", "Nhà Hàng Sen Vàng", "Quán Nướng Lá Chuối",
                  "Cơm Niêu Chợ Cũ", "Hải Sản Biển Đông", "Nhà Hàng Đồng Quê", "Quán Ăn Ba Miền",
                  "Bếp Việt Xưa", "Nhà Hàng Hương Sen", "Quán Chay Tịnh Tâm", "Lẩu Nấm Rừng",
                  "Nhà Hàng Phố Biển", "Cơm Gà Xối Mỡ", "Quán Ốc Cô Tư", "Nhà Hàng Làng Nướng"],
    },
    "Khách sạn": {
        "chain_prob": 0.3,
        "chains": ["Mường Thanh", "Vinpearl", "Fusion", "A25 Hotel", "Grand Hotel"],
        "templates": ["{brand} {loc}", "Khách Sạn {brand} {loc}"],
        "cores": ["Khách Sạn Ánh Dương", "Khách Sạn Phương Đông", "Ngọc Lan Hotel",
                  "Khách Sạn Sao Mai", "Á Đông Hotel", "Khách Sạn Bình Minh", "Hương Biển Resort",
                  "Khách Sạn Hoàng Gia", "Thái Bình Hotel", "Khách Sạn Đông Đô", "Sen Homestay",
                  "Khách Sạn Ban Mai", "Nhà Nghỉ Cỏ May", "Khách Sạn Thủy Tiên", "Hoa Đăng Hotel",
                  "Khách Sạn Long Biên", "Mộc Miên Homestay", "Khách Sạn Hải Âu"],
    },
    "Trung tâm thương mại": {
        "chain_prob": 0.8,
        "chains": ["Vincom", "Aeon Mall", "Lotte", "Parkson", "Gigamall", "Crescent Mall",
                   "Vạn Hạnh Mall", "Sense City", "Takashimaya"],
        "templates": ["{brand} {loc}", "{brand} Plaza {loc}"],
        "cores": ["Trung Tâm Thương Mại Sài Gòn", "Chợ Lớn Plaza", "Đông Đô Plaza",
                  "Trung Tâm Mua Sắm Thủ Đô"],
    },
    "ATM": {
        "chain_prob": 1.0,
        "chains": ["Vietcombank", "Techcombank", "VPBank", "BIDV", "Agribank", "ACB",
                   "Sacombank", "MB Bank", "VietinBank", "TPBank", "Đông Á Bank", "SHB"],
        "templates": ["ATM {brand} {loc}", "{brand} {loc}", "ATM {brand} - {loc}"],
        "cores": ["Cây ATM Trung Tâm"],
    },
    "Trạm xăng": {
        "chain_prob": 1.0,
        "chains": ["Petrolimex", "Mipec", "Shell Việt", "PV Oil", "Comeco", "Saigon Petro",
                   "Nam Sông Hậu"],
        "templates": ["Trạm Xăng {brand} {loc}", "Cây Xăng {brand} {loc}"],
        "cores": ["Trạm Xăng Trung Tâm"],
    },
    "Điểm tham quan": {
        "chain_prob": 0.35,
        "chains": ["Sun World", "Vinpearl Land"],
        "templates": ["{brand} {loc}", "{brand} - {loc}"],
        "cores": ["Bảo Tàng Mỹ Thuật", "Chùa Linh Ẩn", "Thác Voi Xanh", "Đồi Chè Cầu Đất",
                  "Vườn Hoa Thành Phố", "Làng Gốm Cổ", "Khu Du Lịch Sinh Thái Suối Mơ",
                  "Bảo Tàng Lịch Sử", "Chùa Cổ Am", "Đền Ngọc Hoàng", "Nhà Thờ Núi",
                  "Vườn Quốc Gia Xanh", "Khu Sinh Thái Rừng Dừa", "Đồi Thông Reo"],
    },
    "Bệnh viện": {
        "chain_prob": 0.5,
        "chains": ["Vinmec", "Hoàn Mỹ", "Thu Cúc", "Medlatec", "Tâm Anh"],
        "templates": ["Bệnh Viện {brand} {loc}", "{brand} {loc}"],
        "cores": ["Bệnh Viện Đa Khoa Sài Gòn", "Bệnh Viện An Sinh", "Bệnh Viện Nhi Đồng",
                  "Bệnh Viện Hồng Ngọc", "Bệnh Viện Đại Học Y", "Bệnh Viện Quốc Tế Vinh",
                  "Bệnh Viện Đa Khoa Tâm Trí", "Bệnh Viện Phương Nam", "Bệnh Viện Ánh Sáng"],
    },
    "Rạp phim": {
        "chain_prob": 0.85,
        "chains": ["CGV", "Galaxy Cinema", "Lotte Cinema", "BHD Star", "Beta Cinemas",
                   "Cinestar", "Mega GS"],
        "templates": ["{brand} {loc}", "{brand} - {loc}"],
        "cores": ["Rạp Phim Thành Phố", "Rạp Chiếu Bóng Hòa Bình"],
    },
    "Công viên": {
        "chain_prob": 0.0,
        "chains": [],
        "templates": ["{brand} {loc}"],
        "cores": ["Công Viên Hòa Bình", "Công Viên Tao Đàn", "Công Viên Gia Định",
                  "Công Viên Lê Văn Tám", "Công Viên Yên Sở", "Công Viên Thủ Lệ",
                  "Công Viên Cầu Giấy", "Công Viên Bách Thảo", "Công Viên Lê Thị Riêng",
                  "Công Viên Hoàng Văn Thụ", "Công Viên Thanh Niên", "Vườn Hoa Lý Tự Trọng",
                  "Công Viên Cây Xanh"],
    },
    "Nhà thuốc": {
        "chain_prob": 0.7,
        "chains": ["Pharmacity", "Long Châu", "An Khang", "Trung Sơn", "Phano", "Nhà Thuốc Việt"],
        "templates": ["Nhà Thuốc {brand} {loc}", "{brand} {loc}"],
        "cores": ["Nhà Thuốc Minh Châu", "Hiệu Thuốc Ngọc Anh", "Nhà Thuốc Đức Tâm",
                  "Nhà Thuốc Hồng Phúc"],
    },
    "Trạm sạc điện": {
        "chain_prob": 0.8,
        "chains": ["VinFast", "EV One", "EVN", "EBOOST", "Selex", "VuPhong"],
        "templates": ["Trạm Sạc {brand} {loc}", "{brand} {loc}"],
        "cores": ["Trạm Sạc Xe Điện Xanh", "Trạm Sạc Năng Lượng Mới"],
    },
}

# Per-category tag pools (style-matched to official: short vi/en, hyphenated, ;-joined).
TAG_POOL: dict[str, list[str]] = {
    "Quán cà phê": ["coffee", "cafe", "wifi", "yên-tĩnh", "làm-việc", "view", "check-in",
                    "chill", "sân-vườn", "acoustic", "rang-xay", "take-away"],
    "Nhà hàng": ["nhà-hàng", "ẩm-thực", "gia-đình", "hải-sản", "món-việt", "đặc-sản", "nhóm",
                 "lẩu", "nướng", "buffet", "đặt-bàn", "ngon"],
    "Khách sạn": ["hotel", "khách-sạn", "nghỉ-dưỡng", "gần-biển", "hồ-bơi", "spa", "view-đẹp",
                  "gia-đình", "công-tác", "giá-tốt"],
    "Trung tâm thương mại": ["mall", "mua-sắm", "ăn-uống", "rạp-phim", "bãi-đỗ-xe", "siêu-thị",
                             "thời-trang", "giải-trí", "trung-tâm"],
    "ATM": ["atm", "bank", "24h", "rút-tiền", "gần-trung-tâm", "chuyển-khoản", "ngân-hàng"],
    "Trạm xăng": ["fuel", "xăng-dầu", "24h", "toilet", "bơm-lốp", "cửa-hàng-tiện-lợi", "rửa-xe"],
    "Điểm tham quan": ["tham-quan", "du-lịch", "check-in", "gia-đình", "ngoài-trời", "văn-hóa",
                       "chụp-ảnh", "free", "nổi-tiếng"],
    "Bệnh viện": ["hospital", "bệnh-viện", "cấp-cứu", "24h", "khám-bệnh", "đa-khoa", "bảo-hiểm"],
    "Rạp phim": ["cinema", "rạp-phim", "giải-trí", "phim", "imax", "gia-đình", "đặt-vé"],
    "Công viên": ["park", "công-viên", "cây-xanh", "đi-bộ", "gia-đình", "trẻ-em", "ngoài-trời",
                  "thể-dục", "hồ-nước"],
    "Nhà thuốc": ["pharmacy", "nhà-thuốc", "sức-khỏe", "thuốc", "mở-muộn", "tư-vấn"],
    "Trạm sạc điện": ["ev", "charging", "sạc-điện", "bãi-đỗ-xe", "nhanh", "xe-điện", "trạm-sạc"],
}

# Per-category price_level plausibility.
PRICE_WEIGHTS: dict[str, dict[int, float]] = {
    "Quán cà phê": {1: 0.2, 2: 0.4, 3: 0.3, 4: 0.1},
    "Nhà hàng": {1: 0.15, 2: 0.35, 3: 0.3, 4: 0.2},
    "Khách sạn": {1: 0.1, 2: 0.3, 3: 0.3, 4: 0.3},
    "Trung tâm thương mại": {2: 0.2, 3: 0.4, 4: 0.4},
    "ATM": {1: 0.6, 2: 0.3, 3: 0.1},
    "Trạm xăng": {1: 0.4, 2: 0.4, 3: 0.2},
    "Điểm tham quan": {1: 0.4, 2: 0.4, 3: 0.1, 4: 0.1},
    "Bệnh viện": {2: 0.5, 3: 0.2, 4: 0.3},
    "Rạp phim": {2: 0.8, 3: 0.2},
    "Công viên": {1: 1.0},
    "Nhà thuốc": {1: 0.8, 2: 0.2},
    "Trạm sạc điện": {1: 0.3, 2: 0.7},
}

# Per-category review_count log-uniform band (min, max) — thin-traffic POIs get fewer reviews.
REVIEW_BAND: dict[str, tuple[int, int]] = {
    "Quán cà phê": (200, 9000), "Nhà hàng": (200, 9000), "Khách sạn": (150, 6000),
    "Trung tâm thương mại": (500, 12000), "ATM": (120, 1200), "Trạm xăng": (120, 1500),
    "Điểm tham quan": (300, 8000), "Bệnh viện": (300, 5000), "Rạp phim": (400, 6000),
    "Công viên": (300, 5000), "Nhà thuốc": (120, 1500), "Trạm sạc điện": (120, 900),
}

# Per-category NORMAL opening-hours pools (no 24/7 / overnight — those are attribute-driven).
NORMAL_HOURS: dict[str, dict[str, float]] = {
    "Quán cà phê": {"07:00-22:00": 0.4, "07:00-23:00": 0.2, "06:30-22:30": 0.15,
                    "08:00-22:00": 0.15, "07:30-23:00": 0.1},
    "Nhà hàng": {"10:00-22:00": 0.35, "11:00-23:00": 0.25, "09:00-22:00": 0.2,
                 "10:00-22:30": 0.1, "06:00-14:00": 0.1},
    "Khách sạn": {"24/7": 0.7, "06:00-23:00": 0.3},
    "Trung tâm thương mại": {"09:30-22:00": 0.5, "10:00-22:00": 0.35, "09:00-22:00": 0.15},
    "ATM": {"07:00-22:00": 0.6, "06:00-23:00": 0.4},
    "Trạm xăng": {"05:00-22:00": 0.4, "06:00-23:00": 0.35, "07:00-22:00": 0.25},
    "Điểm tham quan": {"07:00-17:30": 0.35, "08:00-17:00": 0.3, "06:00-18:00": 0.2,
                       "07:00-22:00": 0.15},
    "Bệnh viện": {"07:00-17:30": 0.5, "07:00-20:00": 0.5},
    "Rạp phim": {"09:00-23:00": 0.4, "09:30-23:30": 0.35, "10:00-23:30": 0.25},
    "Công viên": {"05:00-22:00": 0.4, "06:00-21:30": 0.3, "04:30-22:00": 0.15, "24/7": 0.15},
    "Nhà thuốc": {"07:00-22:00": 0.5, "08:00-22:00": 0.35, "07:30-21:30": 0.15},
    "Trạm sạc điện": {"06:00-22:00": 0.5, "07:00-23:00": 0.5},
}

# Overnight/late-close pools used when the "mở khuya" (late-night) attribute is present.
LATE_HOURS: dict[str, list[str]] = {
    "Quán cà phê": ["17:00-01:00", "18:00-00:00", "16:00-23:30"],
    "Nhà hàng": ["18:00-03:00", "17:00-02:00", "16:00-01:00"],
    "Nhà thuốc": ["08:00-00:00", "07:00-23:30", "08:30-00:30"],
    "Rạp phim": ["12:00-00:30", "13:00-01:00"],
}


def weighted_choice(rng: Random, weights: dict) -> Any:
    """Deterministic weighted pick over a {key: weight} mapping (fixed key order)."""
    keys = list(weights)
    total = sum(weights.values())
    r = rng.random() * total
    upto = 0.0
    for k in keys:
        upto += weights[k]
        if r <= upto:
            return k
    return keys[-1]


def sample_attributes(rng: Random, category: str, city: str) -> list[str]:
    """2-5 taxonomy attributes, weighted-without-replacement; 'gần biển' only in coastal city."""
    pool = {a: w for a, w in ATTR_WEIGHTS[category].items()
            if not (a == "gần biển" and city != COASTAL_CITY)}
    n = rng.randint(2, min(5, len(pool)))
    chosen: list[str] = []
    avail = dict(pool)
    for _ in range(n):
        if not avail:
            break
        k = weighted_choice(rng, avail)
        chosen.append(k)
        del avail[k]
    return chosen


def sample_tags(rng: Random, category: str) -> list[str]:
    pool = TAG_POOL[category]
    n = rng.randint(2, min(4, len(pool)))
    return rng.sample(pool, n)


def build_name(rng: Random, category: str, district: str) -> tuple[str, str]:
    """Return (name, brand). Chain-branded with prob chain_prob (brand = chain); otherwise a
    standalone core (brand = the core, mirroring the official 'name-core is the brand' style)."""
    cfg = NAME_CONFIG[category]
    loc = rng.choice([district, rng.choice(STREETS)])
    if cfg["chains"] and rng.random() < cfg["chain_prob"]:
        brand = rng.choice(cfg["chains"])
        name = rng.choice(cfg["templates"]).format(brand=brand, loc=loc)
        return name, brand
    core = rng.choice(cfg["cores"])
    name = f"{core} {loc}" if rng.random() < 0.4 else core
    return name, core


def pick_sub_category(rng: Random, category: str, city: str) -> str:
    sub = weighted_choice(rng, SUBS[category])
    if sub in _COASTAL_SUBS and city not in _COASTAL_SUBS[sub]:
        sub = "City Hotel" if category == "Khách sạn" else weighted_choice(rng, SUBS[category])
    return sub


def pick_hours(rng: Random, category: str, attrs: list[str]) -> str:
    if "24/7" in attrs:
        return "24/7"
    if "mở khuya" in attrs and category in LATE_HOURS:
        return rng.choice(LATE_HOURS[category])
    return weighted_choice(rng, NORMAL_HOURS[category])


def build_description(rng: Random, category: str, district: str, city: str,
                      attrs: list[str]) -> str:
    phrases = [ATTR_PHRASE[a] for a in attrs]
    p1 = phrases[0]
    p2 = phrases[1] if len(phrases) > 1 else f"không gian {rng.choice(VIBES)}"
    vibe = rng.choice(VIBES)
    cat = CAT_NOUN[category]
    catl = cat[0].lower() + cat[1:]
    text = rng.choice(DESC_TEMPLATES).format(
        cat=cat, catl=catl, catc=cat, district=district, city=city, p1=p1, p2=p2, vibe=vibe
    )
    if len(phrases) >= 3 and rng.random() < 0.5:
        text += " " + rng.choice(SECOND_SENTENCES).format(
            p=phrases[2], catl=catl, district=district
        )
    return text


def apportion(total: int, weights: dict[str, int], floor: int) -> dict[str, int]:
    """Split `total` across categories: a flat floor each, then the remainder by official
    share (largest-remainder, deterministic key-order tie-break). Guarantees each >= floor."""
    keys = list(weights)
    if total < floor * len(keys):
        raise ValueError(f"total {total} too small for floor {floor} over {len(keys)} categories")
    counts = {k: floor for k in keys}
    remaining = total - floor * len(keys)
    wsum = sum(weights.values())
    exact = {k: remaining * weights[k] / wsum for k in keys}
    base = {k: int(exact[k]) for k in keys}
    for k in keys:
        counts[k] += base[k]
    leftover = remaining - sum(base.values())
    order = sorted(keys, key=lambda k: (exact[k] - base[k], k), reverse=True)
    for k in order[:leftover]:
        counts[k] += 1
    return counts


def _city_boxes(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Derive per-city lat/lon bounding boxes + district lists from the official rows."""
    boxes: dict[str, dict[str, Any]] = {}
    for city, g in df.groupby("city"):
        boxes[str(city)] = {
            "lat": (float(g["latitude"].min()), float(g["latitude"].max())),
            "lon": (float(g["longitude"].min()), float(g["longitude"].max())),
            "districts": sorted(g["district"].astype(str).unique().tolist()),
            "n": int(len(g)),
        }
    return boxes


def generate_synthetic(df_official: pd.DataFrame, n_synth: int, seed: int) -> list[dict]:
    """Build `n_synth` synthetic POI row-dicts (keys = official columns). Deterministic in seed."""
    rng = Random(seed)
    boxes = _city_boxes(df_official)
    cities = list(boxes)
    city_weights = {c: boxes[c]["n"] for c in cities}  # ~ official city mix
    cat_official = df_official["category"].value_counts().to_dict()
    cat_weights = {c: int(cat_official.get(c, 1)) for c in CANONICAL_CATEGORIES}
    per_cat = apportion(n_synth, cat_weights, floor=15)

    rows: list[dict] = []
    idx = 0
    for category in CANONICAL_CATEGORIES:
        for _ in range(per_cat[category]):
            idx += 1
            city = weighted_choice(rng, city_weights)
            box = boxes[city]
            district = rng.choice(box["districts"])
            lat = round(rng.uniform(*box["lat"]), 6)
            lon = round(rng.uniform(*box["lon"]), 6)
            sub = pick_sub_category(rng, category, city)
            attrs = sample_attributes(rng, category, city)
            name, brand = build_name(rng, category, district)
            tags = sample_tags(rng, category)
            rating = round(min(4.7, max(3.8, rng.gauss(4.3, 0.25))), 1)
            lo, hi = REVIEW_BAND[category]
            review_count = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
            popularity = int(round(min(98, max(50, rng.gauss(72, 13)))))
            price = weighted_choice(rng, PRICE_WEIGHTS[category])
            hours = pick_hours(rng, category, attrs)
            desc = build_description(rng, category, district, city, attrs)
            street_no = rng.randint(1, 320)
            rows.append({
                "poi_id": f"SYN{idx:04d}",
                "poi_name": name,
                "brand": brand,
                "category": category,
                "sub_category": sub,
                "city": city,
                "district": district,
                "address": f"{street_no} {rng.choice(STREETS)}, {district}",
                "latitude": lat,
                "longitude": lon,
                "rating": rating,
                "review_count": review_count,
                "popularity_score": popularity,
                "price_level": int(price),
                "opening_hours": hours,
                "attributes": ";".join(attrs),
                "tags": ";".join(tags),
                "description": desc,
            })
    return rows


def build_dataframe(official_xlsx: Path, n_total: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (poi_dataset_df, attribute_taxonomy_df). First rows are the official 111 verbatim."""
    df_official = pd.read_excel(official_xlsx, sheet_name=POI_SHEET)
    df_tax = pd.read_excel(official_xlsx, sheet_name=TAX_SHEET)
    n_synth = n_total - len(df_official)
    if n_synth < 0:
        raise ValueError(f"--n {n_total} < official row count {len(df_official)}")
    synth_rows = generate_synthetic(df_official, n_synth, seed)
    df_synth = pd.DataFrame(synth_rows, columns=list(df_official.columns))
    df_out = pd.concat([df_official, df_synth], ignore_index=True)
    return df_out, df_tax


def write_xlsx(out: Path, df_poi: pd.DataFrame, df_tax: pd.DataFrame) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_poi.to_excel(writer, sheet_name=POI_SHEET, index=False)
        df_tax.to_excel(writer, sheet_name=TAX_SHEET, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic synthetic-POI stress corpus generator")
    ap.add_argument("--out", type=Path, default=Path("data/synth/synth_dataset.xlsx"))
    ap.add_argument("--n", type=int, default=1000, help="total POIs (official + synthetic)")
    ap.add_argument("--seed", type=int, default=20260711)
    ap.add_argument("--official", type=Path, default=OFFICIAL_XLSX)
    args = ap.parse_args()

    df_poi, df_tax = build_dataframe(args.official, args.n, args.seed)
    write_xlsx(args.out, df_poi, df_tax)
    n_synth = int(df_poi["poi_id"].astype(str).str.startswith("SYN").sum())
    print(f"wrote {args.out} — {len(df_poi)} POIs ({n_synth} synthetic), seed={args.seed}")


if __name__ == "__main__":
    main()
