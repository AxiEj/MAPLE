"""Packaged, source-bound OBC-II/mbondi2 profile.

This module deliberately verifies shipped data rather than inspecting an
installed OpenMM distribution.  OpenMM remains the attributed upstream source
and an optional validation oracle, never a runtime dependency of this path.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib import resources
import json
from pathlib import Path
import re
from typing import Any, Mapping


OBC2_FORMULA_MANIFEST_ID = "openmm-8.5.2-obc2-mbondi2-v1"
OPENMM_CUSTOMGBFORCES_SHA256 = (
    "f96a49a9e9ef0528e937481b1c1331b43ad03e6ec3018b478c642b301f07391a"
)
OBC2_CANONICAL_PAYLOAD_SHA256 = (
    "6387f1a16e10569d91ea8dc03a90c56035cf4a52888df54c18a1bd5a4bf8c30f"
)
OPENMM_CUSTOMGBFORCES_LICENSE_SHA256 = (
    "fd9a41c274520658277e570f2347afaf95868bc93299d8e9d453d6671387b33d"
)
_DATA_PACKAGE = f"{__package__}.data"


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _resource_text(name: str) -> str:
    return resources.files(_DATA_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def load_obc2_profile_manifest() -> dict[str, Any]:
    """Return an ownership-independent copy of the packaged source manifest."""
    value = json.loads(_resource_text("obc2_profile_manifest.json"))
    if not isinstance(value, dict):
        raise ValueError("Packaged OBC-II source manifest must be a JSON object.")
    return deepcopy(value)


def load_obc2_profile_license() -> str:
    """Return the verbatim permission notice shipped with the derived code."""
    return _resource_text("OPENMM_CUSTOMGBFORCES_LICENSE.txt")


def verify_obc2_profile_manifest(
    *,
    manifest: Mapping[str, Any] | None = None,
    license_text: str | None = None,
) -> dict[str, Any]:
    """Fail closed if packaged rules, source identity, or license drift.

    Optional values exist solely for developer/tests to exercise the verifier;
    production calls read the package resources directly.
    """
    candidate = load_obc2_profile_manifest() if manifest is None else deepcopy(dict(manifest))
    notice = load_obc2_profile_license() if license_text is None else license_text
    if candidate.get("schema_version") != 1:
        raise ValueError("Unsupported OBC-II source-manifest schema.")
    if candidate.get("manifest_id") != OBC2_FORMULA_MANIFEST_ID:
        raise ValueError("OBC-II formula/source manifest identity changed.")
    upstream = candidate.get("upstream")
    if not isinstance(upstream, Mapping):
        raise ValueError("OBC-II source manifest is missing upstream provenance.")
    if (
        upstream.get("project") != "OpenMM"
        or upstream.get("version") != "8.5.2"
        or upstream.get("source_sha256") != OPENMM_CUSTOMGBFORCES_SHA256
    ):
        raise ValueError("OBC-II upstream source identity changed.")
    if candidate.get("runtime_dependency_on_openmm") is not False:
        raise ValueError("Independent OBC-II manifest must not require runtime OpenMM.")
    if candidate.get("supported_scalar_nonpolar_selectors") != ["ace", "none"]:
        raise ValueError("OBC-II scalar selectors must remain exactly ace and none.")

    payload = candidate.get("canonical_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("OBC-II source manifest is missing its canonical payload.")
    payload_sha = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if (
        payload_sha != OBC2_CANONICAL_PAYLOAD_SHA256
        or candidate.get("canonical_payload_sha256") != payload_sha
    ):
        raise ValueError("OBC-II canonical payload SHA256 mismatch.")
    if payload.get("formula_source_profile") != OBC2_FORMULA_MANIFEST_ID:
        raise ValueError("OBC-II canonical payload profile identity changed.")

    license_record = candidate.get("license")
    if not isinstance(license_record, Mapping):
        raise ValueError("OBC-II source manifest is missing the license record.")
    license_sha = hashlib.sha256(notice.encode("utf-8")).hexdigest()
    if (
        license_sha != OPENMM_CUSTOMGBFORCES_LICENSE_SHA256
        or license_record.get("sha256") != license_sha
    ):
        raise ValueError("OBC-II packaged license SHA256 mismatch.")
    return {
        "manifest_id": OBC2_FORMULA_MANIFEST_ID,
        "canonical_payload_sha256": payload_sha,
        "source_sha256": OPENMM_CUSTOMGBFORCES_SHA256,
        "license_sha256": license_sha,
        "upstream_project": "OpenMM",
        "upstream_version": "8.5.2",
        "runtime_openmm_required": False,
    }


def obc2_profile_payload() -> dict[str, Any]:
    """Return verified local constants/rules as a fresh mutable mapping."""
    verify_obc2_profile_manifest()
    return deepcopy(load_obc2_profile_manifest()["canonical_payload"])


def extract_openmm_obc2_profile(source_path: str | Path) -> dict[str, Any]:
    """Developer-only extraction from the frozen upstream audit source.

    The extractor imports no OpenMM code.  It independently parses the bounded
    source ranges and returns the same canonical rule payload used at runtime.
    It never rewrites accepted package data.
    """
    path = Path(source_path)
    source_bytes = path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if source_sha != OPENMM_CUSTOMGBFORCES_SHA256:
        raise ValueError(
            "Frozen OpenMM customgbforces.py SHA256 mismatch during profile extraction."
        )
    text = source_bytes.decode("utf-8")
    lines = text.splitlines(keepends=True)
    manifest = load_obc2_profile_manifest()
    source_ranges = manifest["upstream"]["source_ranges"]
    for name, record in source_ranges.items():
        start = int(record["start_line"])
        end = int(record["end_line"])
        digest = hashlib.sha256("".join(lines[start - 1 : end]).encode("utf-8")).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Frozen OpenMM source range {name!r} changed.")

    radii_block = "".join(lines[300:334])
    screen_block = "".join(lines[394:410])
    energy_block = "".join(lines[351:392])
    particle_block = "".join(lines[427:450])
    obc2_block = "".join(lines[650:690])
    if (
        "strip_unit(params[self.RADIUS_ARG_POSITION], u.nanometer) - self.OFFSET"
        not in particle_block
        or "params[self.SCREEN_POSITION] *= params[self.RADIUS_ARG_POSITION]"
        not in particle_block
    ):
        raise ValueError("Frozen OpenMM particle radius/screen transformation changed.")

    element_names = {
        "carbon": 6,
        "nitrogen": 7,
        "oxygen": 8,
        "fluorine": 9,
        "silicon": 14,
        "phosphorus": 15,
        "sulfur": 16,
        "chlorine": 17,
    }
    radii: dict[str, float] = {}
    for name, number in element_names.items():
        if name == "carbon":
            value = _extract_float(
                radii_block,
                r"elif\s+element\s+is\s+E\.carbon:\s*radii\[i\]\s*=\s*([0-9.]+)",
                "mbondi2 carbon radius",
            )
        else:
            value = _extract_float(
                radii_block,
                rf"E\.{name}:\s*([0-9.]+)",
                f"mbondi2 {name} radius",
            )
        radii[str(number)] = value
    screens: dict[str, float] = {}
    screen_elements = {
        "hydrogen": 1,
        "carbon": 6,
        "nitrogen": 7,
        "oxygen": 8,
        "fluorine": 9,
        "phosphorus": 15,
        "sulfur": 16,
    }
    for name, number in screen_elements.items():
        screens[str(number)] = _extract_float(
            screen_block,
            rf"E\.{name}\s*:\s*\(\s*([-0-9.]+)",
            f"normal {name} screen factor",
        )
    if "_SCREEN_PARAMETERS[E.deuterium] = _SCREEN_PARAMETERS[E.hydrogen]" not in screen_block:
        raise ValueError("Frozen OpenMM deuterium screen alias changed.")

    payload = {
        "schema_version": 1,
        "formula_source_profile": OBC2_FORMULA_MANIFEST_ID,
        "radius_profile": "obc2-mbondi2",
        "offset_nm": _extract_float(
            text, r"OFFSET\s*=\s*([0-9.]+)", "OBC radius offset"
        ),
        "solute_dielectric": _extract_float(
            obc2_block, r"soluteDielectric=([0-9.]+)", "OBC-II solute dielectric"
        ),
        "solvent_dielectric": _extract_float(
            obc2_block, r"solventDielectric=([0-9.]+)", "OBC-II solvent dielectric"
        ),
        "salt_kappa_inverse_nm": _extract_float(
            obc2_block, r"kappa=([0-9.]+)\):", "OBC-II kappa"
        ),
        "obc2_coefficients": {
            "alpha": 1.0,
            "beta": _extract_float(
                obc2_block, r"tanh\(psi-([0-9.]+)\*psi\^2", "OBC-II beta"
            ),
            "gamma": _extract_float(
                obc2_block, r"\+([0-9.]+)\*psi\^3", "OBC-II gamma"
            ),
        },
        "coulomb_kj_mol_nm_e2": _extract_float(
            energy_block, r"-0\.5\*([0-9.]+)\*", "OBC Coulomb constant"
        ),
        "ace_kj_mol_nm2": _extract_float(
            energy_block, r'addEnergyTerm\("([0-9.]+)\*\(radius', "ACE constant"
        ),
        "hydrogen_radii_angstrom": {
            "bonded_to_nitrogen": _extract_float(
                radii_block,
                r"bondeds\[0\]\.element\s+is\s+E\.nitrogen:\s*radii\[i\]\s*=\s*([0-9.]+)",
                "nitrogen-bound hydrogen radius",
            ),
            "otherwise": _extract_float(
                radii_block, r"else:\s*radii\[i\]\s*=\s*([0-9.]+)", "other hydrogen radius"
            ),
        },
        "radii_angstrom_by_atomic_number": radii,
        "default_radius_angstrom": _extract_float(
            radii_block, r"default_radius\s*=\s*([0-9.]+)", "default mbondi2 radius"
        ),
        "screen_factors_by_atomic_number": screens,
        "default_screen_factor": _extract_float(
            screen_block, r"None\s*:\s*\(\s*([-0-9.]+)", "default normal screen factor"
        ),
    }
    observed_payload_sha = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if observed_payload_sha != OBC2_CANONICAL_PAYLOAD_SHA256:
        raise ValueError("Extracted OpenMM OBC-II canonical payload SHA256 mismatch.")
    return {
        "source_sha256": source_sha,
        "source_ranges": deepcopy(source_ranges),
        "canonical_payload": payload,
        "canonical_payload_sha256": observed_payload_sha,
    }


def _extract_float(text: str, pattern: str, label: str) -> float:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None or match.lastindex is None:
        raise ValueError(f"Could not extract {label} from frozen OpenMM source.")
    return float(match.group(1))
