#!/usr/bin/env python3
"""Replay the SLIC/CDC Supporting Information table-level accuracy audit.

The script does not run SLIC/CDC. It extracts the published SI twice with
Poppler ``pdftotext`` and independently recomputes the Table S10 statistics
used by the Route 1 source audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

_ROW = re.compile(r"^\s*(\S.*?)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)(?:\s|$)")
_WRAPPED_NAME_EXCLUSIONS = {
    "cav",
    "comb",
    "disp",
    "es",
    "hb",
    "kcal",
    "mol",
    "tot",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def artifact_content_sha256(artifact: dict[str, Any]) -> str:
    """Return the canonical hash while excluding the hash field itself."""

    payload = {key: value for key, value in artifact.items() if key != "content_sha256"}
    return _sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    )


def normalize_name(name: str) -> str:
    """Normalize only punctuation variants needed for table-name matching."""

    return name.strip().lower().replace("-", "")


def parse_training_name_rows(layout_text: str) -> list[list[str]]:
    """Parse the 21-by-3 Table S4 training-name grid from layout text."""

    marker = "Table S4: Training-set used in SLIC/CDC calculations"
    start = layout_text.index(marker) + len(marker)
    end = layout_text.index("Table S5:", start)
    rows: list[list[str]] = []
    for line in layout_text[start:end].splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit():
            continue
        columns = [
            value.strip() for value in re.split(r"\s{2,}", stripped) if value.strip()
        ]
        if len(columns) == 3:
            rows.append(columns)
    return rows


def parse_training_names(layout_text: str) -> list[str]:
    """Flatten the Table S4 layout grid in source row and column order."""

    return [name for row in parse_training_name_rows(layout_text) for name in row]


def parse_table_s10(text: str) -> list[dict[str, float | str]]:
    """Parse compound name, experimental total, and calculated total."""

    start = text.index("Table S10:")
    end = text.index("Table S11:", start)
    rows: list[dict[str, float | str]] = []
    pending_index: int | None = None

    for line in text[start:end].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("rmse"):
            break

        match = _ROW.match(line)
        if match:
            rows.append(
                {
                    "name": match.group(1).strip(),
                    "experimental": float(match.group(2)),
                    "calculated": float(match.group(3)),
                }
            )
            pending_index = len(rows) - 1
            continue

        if pending_index is None:
            continue
        is_wrapped_name_suffix = (
            len(stripped) <= 30
            and " " not in stripped
            and re.fullmatch(r"[A-Za-z0-9,'+-]+", stripped) is not None
            and not stripped.isdigit()
            and not re.search(r"\d+\.\d+", stripped)
            and not any(character in stripped for character in "[]()=∗∆")
            and lowered not in _WRAPPED_NAME_EXCLUSIONS
            and not lowered.startswith(
                (
                    "calc.",
                    "expt.",
                    "figure",
                    "neutral compound",
                    "reference",
                    "table s10",
                )
            )
        )
        if is_wrapped_name_suffix:
            rows[pending_index]["name"] = str(rows[pending_index]["name"]) + stripped

    return rows


def _metrics(rows: list[dict[str, float | str]]) -> dict[str, float | int]:
    errors = [float(row["calculated"]) - float(row["experimental"]) for row in rows]
    count = len(errors)
    return {
        "n": count,
        "mae_kcal_per_mol": sum(abs(error) for error in errors) / count,
        "rmse_kcal_per_mol": math.sqrt(sum(error * error for error in errors) / count),
        "mean_signed_error_kcal_per_mol": sum(errors) / count,
        "maximum_absolute_error_kcal_per_mol": max(map(abs, errors)),
    }


def _extract(pdf: Path, mode: str, destination: Path) -> str:
    subprocess.run(
        ["pdftotext", mode, str(pdf), str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    return destination.read_text(encoding="utf-8")


def replay(pdf: Path) -> dict[str, Any]:
    """Extract the SI and return a source-bound, table-level replay report."""

    version = subprocess.run(
        ["pdftotext", "-v"],
        check=True,
        capture_output=True,
        text=True,
    )
    version_line = (version.stderr or version.stdout).splitlines()[0]

    with tempfile.TemporaryDirectory(prefix="slic-cdc-si-") as tmp:
        directory = Path(tmp)
        layout_path = directory / "layout.txt"
        raw_path = directory / "raw.txt"
        layout_text = _extract(pdf, "-layout", layout_path)
        raw_text = _extract(pdf, "-raw", raw_path)
        layout_sha256 = _sha256_file(layout_path)
        raw_sha256 = _sha256_file(raw_path)

    layout_rows = parse_table_s10(layout_text)
    raw_rows = parse_table_s10(raw_text)
    if layout_rows != raw_rows:
        raise ValueError("Layout and raw Table S10 extractions do not match.")

    training_name_rows = parse_training_name_rows(layout_text)
    training_names = [name for row in training_name_rows for name in row]
    if len(training_name_rows) != 21 or any(
        len(row) != 3 for row in training_name_rows
    ):
        raise ValueError("Table S4 is not the expected 21-by-3 training-name grid.")
    training_set = {normalize_name(name) for name in training_names}
    table_names = {normalize_name(str(row["name"])) for row in layout_rows}
    missing_training_names = sorted(training_set - table_names)
    if missing_training_names:
        raise ValueError(
            "Training names missing from Table S10: "
            + ", ".join(missing_training_names)
        )

    training_rows = [
        row for row in layout_rows if normalize_name(str(row["name"])) in training_set
    ]
    remaining_rows = [
        row
        for row in layout_rows
        if normalize_name(str(row["name"])) not in training_set
    ]

    normalized_counts: dict[str, int] = {}
    normalized_display: dict[str, list[str]] = {}
    for row in layout_rows:
        normalized = normalize_name(str(row["name"]))
        normalized_counts[normalized] = normalized_counts.get(normalized, 0) + 1
        normalized_display.setdefault(normalized, []).append(str(row["name"]))
    duplicates = sorted(
        min(displays, key=len)
        for normalized, displays in normalized_display.items()
        if normalized_counts[normalized] > 1
    )

    ranked = sorted(
        layout_rows,
        key=lambda row: abs(float(row["calculated"]) - float(row["experimental"])),
        reverse=True,
    )
    largest_errors = []
    for row in ranked[:3]:
        error = float(row["calculated"]) - float(row["experimental"])
        largest_errors.append(
            {
                "name": row["name"],
                "experimental": row["experimental"],
                "calculated": row["calculated"],
                "error": error,
            }
        )

    row_payload = json.dumps(
        layout_rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    training_name_payload = json.dumps(
        training_names,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "route1-slic-cdc-si-table-replay",
        "claim_scope": (
            "Source-bound transcription-level replay of published SLIC/CDC "
            "Supporting Information Tables S4 and S10. Table S4 is parsed from "
            "the layout-preserving extraction; Table S10 rows are required to "
            "match across layout and raw extraction modes. This is not a model "
            "execution, independent data source, independent FreeSolv calculation, "
            "or product benchmark."
        ),
        "source": {
            "filename": pdf.name,
            "pdf_sha256": _sha256_file(pdf),
            "pdftotext_version": version_line,
            "layout_text_sha256": layout_sha256,
            "raw_text_sha256": raw_sha256,
        },
        "extraction": {
            "layout_and_raw_rows_identical": True,
            "parsed_rows": len(layout_rows),
            "row_payload_sha256": _sha256_bytes(row_payload),
            "normalized_unique_names": len(normalized_counts),
            "normalized_duplicate_names": duplicates,
            "training_names_listed": len(training_names),
            "training_table_s4": {
                "extraction_mode": "layout",
                "row_count": len(training_name_rows),
                "column_count": 3,
                "name_payload_sha256": _sha256_bytes(training_name_payload),
            },
        },
        "metrics": {
            "all_parsed_rows": _metrics(layout_rows),
            "listed_training_rows": _metrics(training_rows),
            "remaining_rows_after_excluding_listed_training_names": _metrics(
                remaining_rows
            ),
            "largest_absolute_errors_kcal_per_mol": largest_errors,
        },
    }
    report["content_sha256"] = artifact_content_sha256(report)
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    report = replay(arguments.pdf.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if arguments.output is None:
        print(rendered, end="")
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
