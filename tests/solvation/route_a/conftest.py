from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = PROJECT_ROOT / "docs" / "solvation" / "route-a"
FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)
