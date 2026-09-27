"""Project sealed native numeric cavity cases into a portable regression fixture.

No upstream source/binary or experimental labels are copied. Requires the
already-generated local pinned native panel/translation assets; no QM or
native executable is run. Geometry, rmin and expected cavity scalar only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / ".omx/benchmarks/route1-torch-amber-exact-sp"
INPUTS = ROOT / ".omx/benchmarks/route1-torch-full-cha/panel30/continuum_inputs"
PINNED_FILES = {
    "input_manifest": (
        INPUTS / "manifest.json",
        "344711992c25d89f8413306208ce0a1dcaf9910f02042aa98200364361a9ae9b",
    ),
    "native_manifest": (
        BENCHMARK / "native-oracle30-v1/summary.json",
        "71c1b483ba3c171b037df83caab469cfd72f7d58031d46c7249ff021b885bf47",
    ),
    "translations": (
        BENCHMARK / "cavity-translation-oracle-v1.json",
        "cb5901517b92929c3e969a774bbea2e501f0a4e42299406595dfefee0d4fef81",
    ),
}
# Existing chemistry pilots, followed by three pre-existing geometric probes.
CASES = ("mobley_9055303", "mobley_3053621", "mobley_1636752")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pinned(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Frozen numeric source drift: {path.name}")
    return json.loads(path.read_bytes())


def freeze(output):
    if output.exists():
        raise FileExistsError(output)
    inputs, native, shifts = (
        read_pinned(*PINNED_FILES[key])
        for key in ("input_manifest", "native_manifest", "translations")
    )
    by_id = {row["compound_id"]: row for row in inputs["records"]}
    records = []
    for case in CASES:
        item = by_id[case]
        topology = read_pinned(
            INPUTS / case / "topology.json", item["topology_file_sha256"]
        )
        coords = read_pinned(
            INPUTS / case / "coordinates.json", item["coordinate_file_sha256"]
        )
        reference = read_pinned(
            BENCHMARK / "native-oracle30-v1/records" / case / "record.json",
            native["record_file_sha256"][case],
        )
        records.append(
            {
                "case_id": case,
                "positions_angstrom": coords["positions_angstrom"],
                "lj_rmin_angstrom": topology["lj_rmin_angstrom"],
                "native_cavity_kcal_mol": reference["components_kcal_mol"]["cavity"],
                "native_prmtop_sha256": reference["prmtop_sha256"],
                "native_coordinate_input_sha256": reference["inpcrd_sha256"],
                "native_pbsa_binary_sha256": reference["executables_sha256"]["pbsa"],
                "native_pbsa_input_sha256": reference["pbsa_input_sha256"],
                "native_pbsa_output_sha256": reference["pbsa_output_sha256"],
            }
        )
    for index in (1, 2, 3):
        item = shifts["rows"][index]
        records.append(
            {
                "case_id": f"methane-translation-{index}",
                "positions_angstrom": item["positions_angstrom"],
                "lj_rmin_angstrom": shifts["lj_rmin_angstrom"],
                "native_cavity_kcal_mol": item["native_cavity_kcal_mol"],
                "native_prmtop_sha256": shifts["prmtop_sha256"],
                "native_pbsa_binary_sha256": shifts["pbsa_executable_sha256"],
                "native_translation_artifact_sha256": PINNED_FILES["translations"][1],
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "pbsa-discrete-cavity-numeric-oracle-v1",
        "scope": "cavity scalar only; not full CHA, forces, or hydration accuracy",
        "label_reads": False,
        "new_qm": False,
        "upstream_source_included": False,
        "generator_sha256": sha(__file__),
        "source_file_sha256": {key: value[1] for key, value in PINNED_FILES.items()},
        "cases": records,
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output, payload["content_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    freeze(parser.parse_args().output)
