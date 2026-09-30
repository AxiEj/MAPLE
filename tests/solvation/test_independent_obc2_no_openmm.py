"""Guard the independent import/runtime against all OpenMM dependency paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPOSITORY_ROOT / "tests" / "solvation" / "run_no_openmm_install.py"


def test_project_extras_keep_legacy_pins_and_add_one_no_openmm_torch_path():
    import tomllib

    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text("utf-8"))
    extras = project["project"]["optional-dependencies"]

    assert extras["implicit-gb"] == ["openmm==8.5.2"]
    assert extras["implicit"] == ["openmm==8.5.2"]
    assert extras["implicit-ddlpb"] == ["openmm==8.5.2", "pyddx==0.8.0"]
    assert extras["implicit-free-energy"] == ["openmm==8.5.2", "pymbar==4.0.3"]
    assert extras["implicit-obc2-torch"] == ["torch"]
    assert all(
        "openmm" not in dependency.lower()
        for dependency in extras["implicit-obc2-torch"]
    )


def test_guarded_subprocess_runs_efhhvp_without_import_or_metadata_lookup(
    water_mol2,
):
    script = textwrap.dedent("""
        import importlib.abc
        import importlib.metadata
        import json
        import sys

        class BlockOpenMM(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "openmm" or fullname.startswith("openmm."):
                    raise AssertionError(f"forbidden OpenMM import: {fullname}")
                return None

        original_version = importlib.metadata.version
        original_distribution = importlib.metadata.distribution

        def guarded_version(name):
            if str(name).lower() == "openmm":
                raise AssertionError("forbidden OpenMM metadata version lookup")
            return original_version(name)

        def guarded_distribution(name):
            if str(name).lower() == "openmm":
                raise AssertionError("forbidden OpenMM distribution lookup")
            return original_distribution(name)

        importlib.metadata.version = guarded_version
        importlib.metadata.distribution = guarded_distribution
        sys.meta_path.insert(0, BlockOpenMM())

        from maple.function.read.filereader.mol2_reader import MOL2Reader
        from maple.function.calculator.extra_correction.implicit.independent_obc2 import IndependentOBC2

        atoms = MOL2Reader(sys.argv[1], charge=0, mult=1)
        provider = IndependentOBC2(atoms, atoms.get_initial_charges(), nonpolar="ace", device="cpu")
        result = provider.evaluate(atoms, need_forces=True)
        hessian = provider.hessian(atoms)
        direction = [1.0] * (3 * len(atoms))
        directional = provider.directional_derivatives(atoms, direction)
        assert result.forces_hartree_per_angstrom.shape == (len(atoms), 3)
        assert hessian.shape == (3 * len(atoms), 3 * len(atoms))
        assert directional.hvp_hartree_per_angstrom2.shape == (3 * len(atoms),)
        assert not any(name == "openmm" or name.startswith("openmm.") for name in sys.modules)
        print(json.dumps({"provider": provider.provenance["provider"], "ok": True}))
        """)
    guarded_script = (
        f"import sys; sys.path.insert(0, {str(REPOSITORY_ROOT)!r})\n" + script
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", guarded_script, str(water_mol2)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"provider": "torch-obc2"' in completed.stdout


def test_fresh_install_harness_declares_an_offline_omx_scoped_environment():
    completed = subprocess.run(
        [sys.executable, str(HARNESS), "--describe"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    description = json.loads(completed.stdout)
    assert description["network_allowed"] is False
    assert description["system_site_packages"] is False
    assert description["pythonpath_allowed"] is False
    assert description["source_checkout_on_probe_sys_path"] is False
    assert description["candidate_extra"] == "implicit-obc2-torch"
    assert description["independent_profile"]["torch"] == "2.7.1+cpu"
    assert description["independent_profile"]["replaces_model_qualification"] is False
    for key in ("wheelhouse", "venv", "receipt", "runtime_root"):
        path = Path(description[key]).resolve()
        path.relative_to((REPOSITORY_ROOT / ".omx").resolve())


@pytest.mark.skipif(
    os.environ.get("MAPLE_RUN_ISOLATED_INSTALL") != "1",
    reason="authoritative offline wheel-install proof is an explicit release gate",
)
def test_authoritative_fresh_install_harness_is_run_explicitly():
    """The release gate must supply a truly empty venv and cached offline wheels.

    This marker prevents the ordinary unit suite from silently substituting an
    import blocker for the required absence of both ``openmm`` and its dist-info.
    The execution harness populates this test after building the candidate wheel.
    """
    assert HARNESS.is_file(), "missing authoritative fresh-install harness"
    arguments = [sys.executable, str(HARNESS)]
    environment_options = {
        "MAPLE_ISOLATION_WHEELHOUSE": "--wheelhouse",
        "MAPLE_ISOLATION_VENV": "--venv",
        "MAPLE_ISOLATION_RECEIPT": "--receipt",
        "MAPLE_ISOLATION_CANDIDATE_WHEEL": "--candidate-wheel",
    }
    for variable, option in environment_options.items():
        if os.environ.get(variable):
            arguments.extend((option, os.environ[variable]))
    completed = subprocess.run(
        arguments,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["passed"] is True
    assert receipt["network_allowed"] is False
    assert receipt["openmm"]["import_spec"] is None
    assert receipt["openmm"]["distribution_metadata"] is None
    assert receipt["probe"]["public_operations"] == [
        "energy",
        "forces",
        "hessian",
        "direct_hvp",
    ]
    assert receipt["probe"]["portable_parity_passed"] is True
    assert receipt["probe"]["isolation"]["venv_active"] is True
    assert receipt["probe"]["isolation"]["pythonpath"] is None
    assert receipt["probe"]["isolation"]["base_site_packages_on_sys_path"] is False
    assert receipt["candidate"]["source_binding"]["matches_checkout"] is True
