"""Export the OpenAPI spec to openapi.json (SPEC §9; submission deliverable).

  python scripts/export_openapi.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsearch.api import create_app  # noqa: E402


def main() -> None:
    app = create_app(prewarm=False, mode="local")
    spec = app.openapi()
    out = Path("openapi.json")
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}  ({len(spec.get('paths', {}))} paths)")


if __name__ == "__main__":
    main()
