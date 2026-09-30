#!/usr/bin/env python3
"""Build a wheelhouse-backed, fresh-venv proof for independent OBC-II.

The harness never resolves from an index and never exposes the source checkout
to the installed-package probe.  Wheelhouse preparation is explicit so the
release receipt binds every cached archive and the candidate MAPLE wheel.
"""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import compat32
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
OMX_ROOT = (ROOT / ".omx").resolve()
ISOLATION_ROOT = OMX_ROOT / "benchmarks/route1-rewrite-20260930/isolation"
DEFAULT_WHEELHOUSE = ISOLATION_ROOT / "wheelhouse"
DEFAULT_VENV = ISOLATION_ROOT / "fresh-venv"
DEFAULT_RECEIPT = ISOLATION_ROOT / "fresh-install-receipt.json"
DEFAULT_RUNTIME_ROOT = ISOLATION_ROOT / "fresh-runtime"
DEFAULT_MANIFEST = ISOLATION_ROOT / "offline-wheel-selection.json"
DEFAULT_CANDIDATE = ISOLATION_ROOT / "dist/maple-0.1.4-py3-none-any.whl"
PORTABLE_FIXTURE = ROOT / "tests/solvation/data/obc2_openmm_8_5_2_reference_v1.json"
INDEPENDENT_TORCH_VERSION = "2.7.1+cpu"

EXPECTED_EXTRAS = {
    "implicit": {"openmm": "==8.5.2"},
    "implicit-gb": {"openmm": "==8.5.2"},
    "implicit-ddlpb": {"openmm": "==8.5.2", "pyddx": "==0.8.0"},
    "implicit-free-energy": {"openmm": "==8.5.2", "pymbar": "==4.0.3"},
    "implicit-obc2-torch": {"torch": ""},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_omx(path: Path, *, role: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(OMX_ROOT)
    except ValueError as exc:
        raise ValueError(f"{role} must remain under {OMX_ROOT}: {resolved}") from exc
    return resolved


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _wheel_metadata(path: Path) -> tuple[object, object, list[str]]:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a complete wheel archive: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_members = [
            name
            for name in names
            if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
        ]
        wheel_members = [
            name
            for name in names
            if name.count("/") == 1 and name.endswith(".dist-info/WHEEL")
        ]
        if len(metadata_members) != 1 or len(wheel_members) != 1:
            raise ValueError(f"Wheel metadata is ambiguous: {path}")
        metadata = BytesParser(policy=compat32).parsebytes(
            archive.read(metadata_members[0])
        )
        wheel = BytesParser(policy=compat32).parsebytes(archive.read(wheel_members[0]))
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Wheel CRC failed for {path}: {bad_member}")
    return metadata, wheel, names


def _candidate_source_binding(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        wheel_sources = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("maple/") and not name.endswith("/")
        }
    checkout_sources = {
        source.relative_to(ROOT).as_posix(): _sha256(source)
        for source in ROOT.glob("maple/**/*.py")
        if source.is_file()
    }
    required_data = (
        ROOT
        / "maple/function/calculator/extra_correction/implicit/data/obc2_profile_manifest.json",
        ROOT
        / "maple/function/calculator/extra_correction/implicit/data/OPENMM_CUSTOMGBFORCES_LICENSE.txt",
    )
    checkout_sources.update(
        {
            source.relative_to(ROOT).as_posix(): _sha256(source)
            for source in required_data
            if source.is_file()
        }
    )
    missing = sorted(set(checkout_sources) - set(wheel_sources))
    mismatched = sorted(
        name
        for name in set(checkout_sources).intersection(wheel_sources)
        if checkout_sources[name] != wheel_sources[name]
    )
    matches = not missing and not mismatched
    if not matches:
        raise ValueError(
            "Candidate wheel is stale or incomplete relative to the checkout: "
            f"missing={missing}, mismatched={mismatched}"
        )
    payload = {
        "checkout_sources": checkout_sources,
        "wheel_sources": wheel_sources,
    }
    return {
        "matches_checkout": True,
        "source_count": len(checkout_sources),
        "aggregate_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _validate_candidate(path: Path) -> dict[str, object]:
    metadata, wheel, _names = _wheel_metadata(path)
    if canonicalize_name(str(metadata["Name"])) != "maple":
        raise ValueError("Candidate wheel is not MAPLE")
    if str(metadata["Version"]) != "0.1.4":
        raise ValueError("Candidate wheel must be maple==0.1.4")
    provided = set(metadata.get_all("Provides-Extra") or [])
    if provided != set(EXPECTED_EXTRAS):
        raise ValueError(f"Candidate extra set changed: {sorted(provided)}")
    requirements = [
        Requirement(value) for value in metadata.get_all("Requires-Dist") or []
    ]
    observed: dict[str, dict[str, str]] = {}
    for extra in sorted(EXPECTED_EXTRAS):
        selected: dict[str, str] = {}
        for requirement in requirements:
            if requirement.marker is None:
                continue
            if requirement.marker.evaluate({"extra": extra}):
                selected[canonicalize_name(requirement.name)] = str(
                    requirement.specifier
                )
        observed[extra] = selected
    if observed != EXPECTED_EXTRAS:
        raise ValueError(f"Candidate extra requirements changed: {observed}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "name": str(metadata["Name"]),
        "version": str(metadata["Version"]),
        "tags": wheel.get_all("Tag") or [],
        "extras": observed,
        "source_binding": _candidate_source_binding(path),
    }


def _wheel_filename(record: dict[str, object]) -> str:
    distribution = re.sub(r"[-.]+", "_", str(record["name"]))
    version = str(record["version"]).replace("-", "_")
    compatible = {str(tag) for tag in sys_tags()}
    tags = [str(tag) for tag in record["tags"] if str(tag) in compatible]
    if not tags:
        raise ValueError(f"No compatible wheel tag for {record['name']}")
    tag = tags[0]
    return f"{distribution}-{version}-{tag}.whl"


def _prepare_wheelhouse(
    wheelhouse: Path, manifest_path: Path, candidate_path: Path
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("openmm_selected") is not False:
        raise ValueError("Offline manifest must explicitly exclude OpenMM")
    if manifest.get("missing_requested") or manifest.get(
        "unsatisfied_runtime_requirements"
    ):
        raise ValueError("Offline dependency manifest is incomplete")
    selected = manifest.get("selected", {})
    torch_record = selected.get("torch", {})
    if torch_record.get("version") != INDEPENDENT_TORCH_VERSION:
        raise ValueError("Offline manifest is not the frozen CPU independence profile")
    if wheelhouse.exists():
        shutil.rmtree(wheelhouse)
    wheelhouse.mkdir(parents=True)
    installed: list[dict[str, object]] = []
    for key, record in sorted(selected.items()):
        source = Path(str(record["path"])).resolve()
        if _sha256(source) != record["sha256"]:
            raise ValueError(f"Cached wheel bytes changed: {source}")
        metadata, wheel, _names = _wheel_metadata(source)
        if canonicalize_name(str(metadata["Name"])) != canonicalize_name(key):
            raise ValueError(f"Cached wheel name mismatch: {source}")
        if str(metadata["Version"]) != str(record["version"]):
            raise ValueError(f"Cached wheel version mismatch: {source}")
        if not set(wheel.get_all("Tag") or []).intersection(record["tags"]):
            raise ValueError(f"Cached wheel tag mismatch: {source}")
        destination = wheelhouse / _wheel_filename(record)
        shutil.copyfile(source, destination)
        if _sha256(destination) != record["sha256"]:
            raise ValueError(f"Materialized wheel hash mismatch: {destination}")
        installed.append(
            {
                "name": str(metadata["Name"]),
                "version": str(metadata["Version"]),
                "path": str(destination),
                "sha256": str(record["sha256"]),
            }
        )
    candidate = _validate_candidate(candidate_path)
    candidate_destination = wheelhouse / candidate_path.name
    shutil.copyfile(candidate_path, candidate_destination)
    if _sha256(candidate_destination) != candidate["sha256"]:
        raise ValueError("Materialized candidate wheel hash mismatch")
    candidate["path"] = str(candidate_destination)
    result = {
        "wheelhouse": str(wheelhouse),
        "dependency_count": len(installed),
        "dependencies": installed,
        "candidate": candidate,
        "openmm_in_wheelhouse": any(
            canonicalize_name(item["name"]) == "openmm" for item in installed
        ),
    }
    _atomic_json(wheelhouse / "wheelhouse-receipt.json", result)
    return result


PROBE = r"""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import sysconfig
from unittest.mock import patch

import ase
import maple
import numpy as np
import scipy
import torch

checkout = Path(sys.argv[1]).resolve()
fixture_path = Path(sys.argv[2]).resolve()
mol2_path = Path(sys.argv[3]).resolve()
site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
base_site_packages = (Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages").resolve()

def inside(path, parent):
    try:
        Path(path).resolve().relative_to(parent)
        return True
    except ValueError:
        return False

for entry in sys.path:
    if entry and inside(entry, checkout) and not inside(entry, site_packages):
        raise AssertionError(f"source checkout leaked onto sys.path: {entry}")
if sys.prefix == sys.base_prefix:
    raise AssertionError("probe did not execute inside a venv")
if os.environ.get("PYTHONPATH") is not None:
    raise AssertionError("PYTHONPATH leaked into installed-package probe")
if any(Path(entry).resolve() == base_site_packages for entry in sys.path if entry):
    raise AssertionError("base environment site-packages leaked into fresh venv")
origins = {}
for module in (maple, ase, np, scipy, torch):
    origin = Path(module.__file__).resolve()
    if not inside(origin, site_packages):
        raise AssertionError(f"module outside fresh venv: {module.__name__}: {origin}")
    origins[module.__name__] = str(origin)

openmm_spec = importlib.util.find_spec("openmm")
if openmm_spec is not None:
    raise AssertionError(f"OpenMM import unexpectedly exists: {openmm_spec}")
try:
    openmm_distribution = importlib.metadata.distribution("openmm")
except importlib.metadata.PackageNotFoundError:
    openmm_distribution = None
if openmm_distribution is not None:
    raise AssertionError("OpenMM distribution metadata unexpectedly exists")

fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
recorded_hash = fixture["content_sha256"]
payload = dict(fixture)
payload.pop("content_sha256")
observed_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
if observed_hash != recorded_hash:
    raise AssertionError("portable fixture self-hash mismatch")
row = next(row for row in fixture["records"] if (row["case"], row["nonpolar"], row["geometry"]) == ("methanol", "ace", "base"))
if hashlib.sha256(mol2_path.read_bytes()).hexdigest() != row["source_mol2_sha256"]:
    raise AssertionError("portable MOL2 hash mismatch")

from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.function.calculator.extra_correction.implicit.independent_obc2 import IndependentOBC2

atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
atoms.set_positions(np.asarray(row["coordinates_angstrom"], dtype=np.float64))
charges = np.asarray(row["ordered_charges_e"], dtype=np.float64)
atoms.set_initial_charges(charges)
provider = IndependentOBC2(atoms, charges, nonpolar="ace", device="cpu")
result = provider.evaluate(atoms, need_forces=True)
hessian = np.asarray(provider.hessian(atoms), dtype=np.float64)
direction = np.random.default_rng(2026093001).normal(size=3 * len(atoms))
direction /= np.linalg.norm(direction)

def forbidden(*_args, **_kwargs):
    raise AssertionError("direct HVP materialized a dense Hessian")

with patch.object(provider.kernel, "hessian", side_effect=forbidden), patch.object(provider, "hessian", side_effect=forbidden):
    directional = provider.directional_derivatives(atoms, direction)
direct_hvp = np.asarray(directional.hvp_hartree_per_angstrom2, dtype=np.float64)
dense_hvp = hessian @ direction
force_difference = np.asarray(result.forces_hartree_per_angstrom) - np.asarray(row["forces_hartree_per_angstrom"])
energy_error = abs(float(result.energy_hartree) - float(row["total_hartree"]))
energy_limit = 2.0e-10 + 1.0e-9 * abs(float(row["total_hartree"]))
polar_error = abs(float(result.components_hartree["polar"]) - float(row["polar_hartree"]))
nonpolar_error = abs(float(result.components_hartree["nonpolar"]) - float(row["nonpolar_hartree"]))
force_max = float(np.max(np.abs(force_difference), initial=0.0))
force_rms = float(np.sqrt(np.mean(np.square(force_difference))))
hvp_error = float(np.max(np.abs(direct_hvp - dense_hvp), initial=0.0))
hvp_limit = float(1.0e-9 + 1.0e-9 * np.max(np.abs(dense_hvp), initial=0.0))
portable_passed = bool(energy_error <= energy_limit and polar_error <= 2.0e-10 and nonpolar_error <= 2.0e-10 and force_max <= 2.0e-7 and force_rms <= 5.0e-8)
if not portable_passed or hvp_error > hvp_limit:
    raise AssertionError("installed independent OBC-II parity gate failed")
print(json.dumps({
    "public_operations": ["energy", "forces", "hessian", "direct_hvp"],
    "portable_parity_passed": portable_passed,
    "dense_hessian_forbidden_during_direct_hvp": True,
    "provider": provider.provenance["provider"],
    "versions": {"maple": importlib.metadata.version("maple"), "numpy": np.__version__, "scipy": scipy.__version__, "ase": ase.__version__, "torch_distribution": importlib.metadata.version("torch"), "torch_runtime": torch.__version__},
    "origins": origins,
    "site_packages": str(site_packages),
    "cwd": str(Path.cwd().resolve()),
    "metrics": {"energy_abs_hartree": energy_error, "polar_abs_hartree": polar_error, "nonpolar_abs_hartree": nonpolar_error, "force_max_hartree_per_angstrom": force_max, "force_rms_hartree_per_angstrom": force_rms, "hvp_dense_max_hartree_per_angstrom2": hvp_error},
    "openmm": {"import_spec": None, "distribution_metadata": None},
    "isolation": {"venv_active": sys.prefix != sys.base_prefix, "pythonpath": os.environ.get("PYTHONPATH"), "base_site_packages_on_sys_path": any(Path(entry).resolve() == base_site_packages for entry in sys.path if entry), "sys_prefix": sys.prefix, "sys_base_prefix": sys.base_prefix, "sys_path": sys.path},
}, sort_keys=True))
"""


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        PIP_NO_INDEX="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_NO_CACHE_DIR="1",
        PYTHONNOUSERSITE="1",
        http_proxy="http://127.0.0.1:9",
        https_proxy="http://127.0.0.1:9",
        HTTP_PROXY="http://127.0.0.1:9",
        HTTPS_PROXY="http://127.0.0.1:9",
        NO_PROXY="",
        no_proxy="",
    )
    return environment


def _run(
    command: list[str], *, cwd: Path, environment: dict[str, str], timeout: int
) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    result = {
        "command": command,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def _description(args: argparse.Namespace) -> dict[str, object]:
    return {
        "network_allowed": False,
        "system_site_packages": False,
        "pythonpath_allowed": False,
        "source_checkout_on_probe_sys_path": False,
        "candidate_extra": "implicit-obc2-torch",
        "wheelhouse": str(args.wheelhouse),
        "venv": str(args.venv),
        "receipt": str(args.receipt),
        "runtime_root": str(args.runtime_root),
        "independent_profile": {
            "torch": INDEPENDENT_TORCH_VERSION,
            "replaces_model_qualification": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--prepare-wheelhouse", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--wheelhouse", type=Path, default=DEFAULT_WHEELHOUSE)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-wheel", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--portable-fixture", type=Path, default=PORTABLE_FIXTURE)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.wheelhouse = _inside_omx(args.wheelhouse, role="wheelhouse")
    args.venv = _inside_omx(args.venv, role="venv")
    args.receipt = _inside_omx(args.receipt, role="receipt")
    args.runtime_root = _inside_omx(args.runtime_root, role="runtime root")
    args.manifest = _inside_omx(args.manifest, role="cache manifest")
    args.candidate_wheel = _inside_omx(args.candidate_wheel, role="candidate wheel")
    if args.describe:
        print(json.dumps(_description(args), indent=2, sort_keys=True))
        return

    receipt: dict[str, object] = _description(args)
    if args.prepare_wheelhouse:
        receipt["wheelhouse_preparation"] = _prepare_wheelhouse(
            args.wheelhouse, args.manifest, args.candidate_wheel
        )
    if args.prepare_only:
        receipt["passed"] = True
        _atomic_json(args.receipt, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    candidates = sorted(args.wheelhouse.glob("maple-0.1.4-*.whl"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one MAPLE 0.1.4 wheel in {args.wheelhouse}")
    receipt["candidate"] = _validate_candidate(candidates[0])
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    assets = args.runtime_root / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir()
    fixture = json.loads(args.portable_fixture.read_text(encoding="utf-8"))
    row = next(
        row
        for row in fixture["records"]
        if (row["case"], row["nonpolar"], row["geometry"])
        == ("methanol", "ace", "base")
    )
    source_mol2 = ROOT / row["source_mol2_relative_path"]
    fixture_copy = assets / args.portable_fixture.name
    mol2_copy = assets / "methanol.mol2"
    shutil.copyfile(args.portable_fixture, fixture_copy)
    shutil.copyfile(source_mol2, mol2_copy)
    runtime_cwd = args.runtime_root / "cwd"
    if runtime_cwd.exists():
        shutil.rmtree(runtime_cwd)
    runtime_cwd.mkdir()

    if args.venv.exists():
        shutil.rmtree(args.venv)
    venv.EnvBuilder(with_pip=True, clear=True, system_site_packages=False).create(
        args.venv
    )
    python = args.venv / "bin/python"
    environment = _clean_environment()
    install = _run(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-cache-dir",
            f"--find-links={args.wheelhouse}",
            "maple[implicit-obc2-torch]==0.1.4",
        ],
        cwd=runtime_cwd,
        environment=environment,
        timeout=600,
    )
    check = _run(
        [str(python), "-I", "-m", "pip", "check"],
        cwd=runtime_cwd,
        environment=environment,
        timeout=60,
    )
    probe_run = _run(
        [
            str(python),
            "-I",
            "-c",
            PROBE,
            str(ROOT),
            str(fixture_copy),
            str(mol2_copy),
        ],
        cwd=runtime_cwd,
        environment=environment,
        timeout=300,
    )
    probe = json.loads(str(probe_run["stdout"]).strip())
    receipt.update(
        passed=True,
        install=install,
        pip_check=check,
        probe=probe,
        openmm=probe["openmm"],
    )
    _atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
