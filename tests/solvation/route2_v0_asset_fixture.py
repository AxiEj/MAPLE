"""Source-faithful synthetic Route-2 V0 solvent-asset fixture."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.route2_v0_rism_generation_source import (
    V0_RISM_GENERATION_SOURCE_CONSTRUCTION,
    V0_RISM_GENERATION_SOURCE_STATUS,
    normalized_rism1d_xvv_sha256,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_rism_molecular_source import (
    AMBER_ELECTROSTATIC_CHARGE_SCALE,
    AVOGADRO_PER_ANGSTROM3_PER_MOLAR,
    BOLTZMANN_KCAL_MOL_K,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_rism_short_range_source import (
    V0_RISM_SHORT_RANGE_DERIVATION,
    V0_RISM_SHORT_RANGE_SOURCE_CONSTRUCTION,
    V0_RISM_SHORT_RANGE_SOURCE_SCOPE,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_solvent_asset import (
    V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _xvv() -> str:
    thermal_energy = BOLTZMANN_KCAL_MOL_K * 298.0
    oxygen_charge_amber = -0.8 * AMBER_ELECTROSTATIC_CHARGE_SCALE
    hydrogen_charge_amber = 0.4 * AMBER_ELECTROSTATIC_CHARGE_SCALE
    return f"""%VERSION  VERSION_STAMP = V0001.001  DATE = 01:01:00 00:00:01
%FLAG POINTERS
%FORMAT(10I8)
       4       2       1
%FLAG THERMO
%FORMAT(1P5E24.16)
  2.9800000000000000E+02  7.8497000000000000E+01  0.0  1.0 2.0000000000000000E+00 1.0000000000000000E+00
%FLAG ATOM_NAME
%FORMAT(20A4)
O
H1
%FLAG MTV
%FORMAT(10I8)
       1       2
%FLAG NVSP
%FORMAT(10I8)
       2
%FLAG MASS
%FORMAT(1P5E24.16)
  1.5999000000000000E+01  1.0080000000000000E+00
%FLAG RHOV
%FORMAT(1P5E24.16)
  3.3000000000000000E-02  6.6000000000000000E-02
%FLAG RHOSP
%FORMAT(1P5E24.16)
  3.3000000000000000E-02
%FLAG QV
%FORMAT(1P5E24.16)
 {oxygen_charge_amber / math.sqrt(thermal_energy):.16E}  {hydrogen_charge_amber / math.sqrt(thermal_energy):.16E}
%FLAG EPSV
%FORMAT(1P5E24.16)
 {0.1553 / thermal_energy:.16E}  {0.01553 / thermal_energy:.16E}
%FLAG RMIN2V
%FORMAT(1P5E24.16)
  1.7767000000000000E+00  6.5423795200000000E-01
%FLAG COORD
%FORMAT(1P3E24.16)
  0.0000000000000000E+00  0.0000000000000000E+00  0.0000000000000000E+00
  1.0000000000000000E+00  0.0000000000000000E+00  0.0000000000000000E+00
 -3.3331400000000000E-01  9.4281600000000000E-01  0.0000000000000000E+00
%FLAG XVV
%FORMAT(1P5E24.16)
  0.0
"""


def _cvv() -> str:
    thermal_energy = BOLTZMANN_KCAL_MOL_K * 298.0
    charges = (
        -0.8 * AMBER_ELECTROSTATIC_CHARGE_SCALE / math.sqrt(thermal_energy),
        0.4 * AMBER_ELECTROSTATIC_CHARGE_SCALE / math.sqrt(thermal_energy),
    )
    rows = [
        "#RISM1D ATOM-ATOM INTERACTIONS: DIRECT CORRELATION VS. SEPARATION [A]",
        "#    SEPARATION          H1:O             O:O             H1:H1",
    ]
    for radius in (0.0, 2.0, 4.0, 6.0):
        kernel = (
            2.0 / math.sqrt(math.pi) if radius == 0.0 else math.erf(radius) / radius
        )
        rows.append(
            " ".join(
                f"{value:.16E}"
                for value in (
                    radius,
                    -charges[1] * charges[0] * kernel,
                    -charges[0] * charges[0] * kernel,
                    -charges[1] * charges[1] * kernel,
                )
            )
        )
    return "\n".join(rows) + "\n"


def _mdl() -> str:
    return f"""%VERSION  VERSION_STAMP = V0001.000
%FLAG TITLE
%FORMAT(20a4)
cSPCE
%FLAG POINTERS
%FORMAT(10I8)
       3       2
%FLAG ATMTYP
%FORMAT(10I8)
       1       2
%FLAG ATMNAME
%FORMAT(20a4)
O   H1
%FLAG MASS
%FORMAT(5e16.8)
  1.59990000e+01  1.00800000e+00
%FLAG CHG
%FORMAT(5e16.8)
 {-0.8 * AMBER_ELECTROSTATIC_CHARGE_SCALE:.8e}  {0.4 * AMBER_ELECTROSTATIC_CHARGE_SCALE:.8e}
%FLAG LJEPSILON
%FORMAT(5e16.8)
  1.55300000e-01  1.55300000e-02
%FLAG LJSIGMA
%FORMAT(5e16.8)
  1.77670000e+00  6.54237952e-01
%FLAG MULTI
%FORMAT(10I8)
       1       2
%FLAG COORD
%FORMAT(5e16.8)
  0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
  0.00000000e+00 -3.33314000e-01  9.42816000e-01  0.00000000e+00
"""


def _rism1d_input() -> str:
    density_molar = 0.033 / AVOGADRO_PER_ANGSTROM3_PER_MOLAR
    return f"""&PARAMETERS
  THEORY='DRISM', CLOSURE='PSE3',
  NR=4, DR=2.0,
  SMEAR=1.0, TEMPERATURE=298.0, DIEPS=78.497, NSP=1,
  SELFTEST=-1, OUTLIST='xc', MAXSTEP=100,
  TOLERANCE=1.e-12
/
&SPECIES
  DENSITY={density_molar:.16E},
  UNITS='M'
  MODEL='cSPCE.mdl'
/
"""


def _thermodynamic_output() -> str:
    return """NET CHARGE NEUTRALITY [sqrt(kT A)]
  ZERO Input from MDL          0.0000000000000000E+00
  ZERO Sum of excess charges   4.0000000000000000E-08
TOTAL EXCESS FREE ENERGY PER UNIT VOLUME [kT/A^3]
  Free energy                           -3.8000000000000000E-01
Relative difference [kT/A^3]
  ZERO ExChem (SM) - ExP to Free Energy  1.0000000000000000E-12
Pressure [kT/A^3]
  Pressure (free energy)   1.2000000000000000E-01
  Pressure (virial)        5.2000000000000000E-01
Relative difference [kT/A^3]
  ZERO Pressure           -3.3333333333333333E+00
"""


def _transcript(run_number: int) -> str:
    return f"""reading input data file: cSPCE.inp
independent run marker: {run_number}
relaxing RISM:
step=   1     Res=  5.0000000000000000E-13     MDIIS=  1
done.
relaxing RISM DT:
step=   1     Res=  4.0000000000000000E-13     MDIIS=  1
outputting Xvv(K) to file: cSPCE.xvv
"""


def _alternate_xvv_first_line(first_line: str) -> str:
    alternate, count = re.subn(
        r"(DATE\s*=\s*)\S+\s+\S+\s*$",
        r"\g<1>01:01:00 00:00:02",
        first_line,
    )
    if count != 1:
        raise ValueError("Synthetic XVV first line needs one DATE field.")
    return alternate


def _entry(path: Path, *, root: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}


def _write_generation_provenance(tmp_path: Path) -> Path:
    xvv = tmp_path / "bulk/cSPCE.xvv"
    cvv = tmp_path / "bulk/cSPCE.cvv"
    thermodynamic = tmp_path / "bulk/cSPCE.thermo"
    transcripts = (_transcript(1), _transcript(2))
    xvv_first_line = xvv.read_bytes().partition(b"\n")[0].decode("utf-8")
    xvv_body = xvv.read_bytes().partition(b"\n")[2]
    first_lines = (xvv_first_line, _alternate_xvv_first_line(xvv_first_line))
    runs = [
        {
            "raw_xvv_sha256": hashlib.sha256(
                first_line.encode("utf-8") + b"\n" + xvv_body
            ).hexdigest(),
            "normalized_xvv_sha256": normalized_rism1d_xvv_sha256(xvv),
            "xvv_first_line": first_line,
            "cvv_sha256": sha256_file(cvv),
            "thermodynamic_output_sha256": sha256_file(thermodynamic),
            "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
            "transcript": transcript,
        }
        for first_line, transcript in zip(first_lines, transcripts, strict=True)
    ]
    payload = {
        "construction": V0_RISM_GENERATION_SOURCE_CONSTRUCTION,
        "status": V0_RISM_GENERATION_SOURCE_STATUS,
        "claim_boundary": "Synthetic source fixture; no physical or accuracy claim.",
        "source_literature": [
            {
                "role": "test source",
                "title": "Synthetic source-faithful fixture",
                "doi": "10.0000/test",
                "url": "https://example.invalid/test",
            }
        ],
        "generator": {
            "package": "test-rism1d",
            "version": "1",
            "build": "test",
            "channel_url": "https://example.invalid/package",
            "package_sha256": "0" * 64,
            "license_expression": "test-only",
            "rism1d_binary_sha256": "1" * 64,
            "site_model_package_path": "test/data/cSPCE.mdl",
            "command": "rism1d cSPCE",
        },
        "no_target_policy": {
            "post_training": False,
            "fine_tuning": False,
            "experimental_solvation_fit": False,
            "map_or_uq_calibration": False,
            "target_solvation_labels_used": False,
            "excluded_target_label_sets": [
                "mnsol",
                "freesolv",
                "development",
                "confirmation",
                "blind",
            ],
        },
        "frozen_source_sha256": {
            "site_model": sha256_file(tmp_path / "model/cSPCE.mdl"),
            "rism1d_input": sha256_file(tmp_path / "bulk/cSPCE.inp"),
            "xvv": sha256_file(xvv),
            "cvv": sha256_file(cvv),
            "thermodynamic_output": sha256_file(thermodynamic),
        },
        "residual_tolerance": 1.0e-12,
        "runs": runs,
        "not_claimed": ["physical liquid", "solvation accuracy"],
    }
    path = tmp_path / "provenance/cSPCE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_route2_v0_test_manifest(
    tmp_path: Path,
    *,
    model_identifier: str = "cSPCE-test-control",
) -> tuple[Path, dict]:
    """Write one complete synthetic source-bound registry manifest."""

    for name, contents in {
        "model/cSPCE.mdl": _mdl(),
        "bulk/cSPCE.inp": _rism1d_input(),
        "bulk/cSPCE.xvv": _xvv(),
        "bulk/cSPCE.cvv": _cvv(),
        "bulk/cSPCE.thermo": _thermodynamic_output(),
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    provenance_path = _write_generation_provenance(tmp_path)
    short_range_path = tmp_path / "short-range/cSPCE-source.json"
    short_range_path.parent.mkdir(parents=True, exist_ok=True)
    short_range_path.write_text(
        json.dumps(
            {
                "construction": V0_RISM_SHORT_RANGE_SOURCE_CONSTRUCTION,
                "source_scope": V0_RISM_SHORT_RANGE_SOURCE_SCOPE,
                "derivation": V0_RISM_SHORT_RANGE_DERIVATION,
                "closure": "PSE3",
                "temperature_kelvin": 298.0,
                "pressure_bar": 1.0,
                "coulomb_tail_start_angstrom": 6.0,
                "coulomb_tail_tolerance_dimensionless": 1.0e-12,
                "target_solvation_labels_used": False,
                "source_sha256": {
                    "site_model": sha256_file(tmp_path / "model/cSPCE.mdl"),
                    "rism1d_input": sha256_file(tmp_path / "bulk/cSPCE.inp"),
                    "xvv": sha256_file(tmp_path / "bulk/cSPCE.xvv"),
                    "cvv": sha256_file(tmp_path / "bulk/cSPCE.cvv"),
                    "thermodynamic_output": sha256_file(tmp_path / "bulk/cSPCE.thermo"),
                    "provenance_statement": sha256_file(provenance_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def entry(name: str) -> dict[str, str]:
        return _entry(tmp_path / name, root=tmp_path)

    payload = {
        "protocol_id": V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION,
        "schema_version": 2,
        "assets": [
            {
                "solvent_id": "water",
                "model": {
                    "family": "ambertools-rism1d",
                    "identifier": model_identifier,
                },
                "state": {"temperature_kelvin": 298.0, "pressure_bar": 1.0},
                "liquid_convention": {
                    "closure": "PSE3",
                    "standard_state": "1M solute / pure-liquid solvent",
                    "pressure_definition": "source bulk thermodynamic pressure",
                    "partial_molar_volume_definition": "source functional derivative",
                    "sign_convention": "subtract P times partial molar volume",
                },
                "bulk_correlation": {
                    "xvv": entry("bulk/cSPCE.xvv"),
                    "cvv": entry("bulk/cSPCE.cvv"),
                    "coulomb_tail_start_angstrom": 6.0,
                    "coulomb_tail_tolerance_dimensionless": 1.0e-12,
                },
                "source_files": {
                    "site_model": entry("model/cSPCE.mdl"),
                    "rism1d_input": entry("bulk/cSPCE.inp"),
                    "thermodynamic_output": entry("bulk/cSPCE.thermo"),
                    "short_range_interaction": entry("short-range/cSPCE-source.json"),
                    "provenance_statement": entry("provenance/cSPCE.json"),
                },
                "molecular_reference": {
                    "atomic_numbers": [8, 1, 1],
                    "site_charges_e": [-0.8, 0.4, 0.4],
                    "reference_positions_bohr": [
                        [0.0, 0.0, 0.0],
                        [1.0 / Bohr, 0.0, 0.0],
                        [-0.333314 / Bohr, 0.942816 / Bohr, 0.0],
                    ],
                    "rism_site_type_names": ["O", "H1", "H1"],
                    "site_model_sha256": entry("model/cSPCE.mdl")["sha256"],
                    "target_total_charge_e": 0.0,
                },
                "provenance": {
                    "target_solvation_labels_used": False,
                    "excluded_target_label_sets": [
                        "mnsol",
                        "freesolv",
                        "development",
                        "confirmation",
                        "blind",
                    ],
                },
            }
        ],
    }
    manifest = tmp_path / "route2-v0-solvent-assets.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest, payload


def refresh_route2_v0_test_source_bindings(
    tmp_path: Path,
    payload: dict,
) -> None:
    """Refresh every synthetic content-addressing edge after a source mutation."""

    paths = {
        "site_model": tmp_path / "model/cSPCE.mdl",
        "rism1d_input": tmp_path / "bulk/cSPCE.inp",
        "xvv": tmp_path / "bulk/cSPCE.xvv",
        "cvv": tmp_path / "bulk/cSPCE.cvv",
        "thermodynamic_output": tmp_path / "bulk/cSPCE.thermo",
    }
    source_hashes = {role: sha256_file(path) for role, path in paths.items()}

    provenance_path = tmp_path / "provenance/cSPCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["frozen_source_sha256"] = dict(source_hashes)
    normalized_xvv = normalized_rism1d_xvv_sha256(paths["xvv"])
    xvv_content = paths["xvv"].read_bytes()
    xvv_first_line = xvv_content.partition(b"\n")[0].decode("utf-8")
    xvv_body = xvv_content.partition(b"\n")[2]
    first_lines = (xvv_first_line, _alternate_xvv_first_line(xvv_first_line))
    for run, first_line in zip(provenance["runs"], first_lines, strict=True):
        run["raw_xvv_sha256"] = hashlib.sha256(
            first_line.encode("utf-8") + b"\n" + xvv_body
        ).hexdigest()
        run["normalized_xvv_sha256"] = normalized_xvv
        run["xvv_first_line"] = first_line
        run["cvv_sha256"] = source_hashes["cvv"]
        run["thermodynamic_output_sha256"] = source_hashes["thermodynamic_output"]
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    short_range_path = tmp_path / "short-range/cSPCE-source.json"
    short_range = json.loads(short_range_path.read_text(encoding="utf-8"))
    short_range["source_sha256"] = {
        **source_hashes,
        "provenance_statement": sha256_file(provenance_path),
    }
    short_range_path.write_text(
        json.dumps(short_range, indent=2),
        encoding="utf-8",
    )

    asset = payload["assets"][0]
    asset["bulk_correlation"]["xvv"]["sha256"] = source_hashes["xvv"]
    asset["bulk_correlation"]["cvv"]["sha256"] = source_hashes["cvv"]
    for role in ("site_model", "rism1d_input", "thermodynamic_output"):
        asset["source_files"][role]["sha256"] = source_hashes[role]
    asset["source_files"]["provenance_statement"]["sha256"] = sha256_file(
        provenance_path
    )
    asset["source_files"]["short_range_interaction"]["sha256"] = sha256_file(
        short_range_path
    )
    asset["molecular_reference"]["site_model_sha256"] = source_hashes["site_model"]


__all__ = [
    "refresh_route2_v0_test_source_bindings",
    "sha256_file",
    "write_route2_v0_test_manifest",
]
