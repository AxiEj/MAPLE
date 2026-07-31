from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from ase import Atoms
from ase.units import Hartree

from maple.function.calculator.calculator_base import CalcABC
from maple.function.calculator.extra_correction.implicit.correction import (
    ImplicitSolvationCorrection,
)
from maple.function.calculator.extra_correction.implicit.result import (
    Route2EnergyLedger,
    SolvationResult,
)
from maple.function.calculator.extra_correction.implicit.route2_engine import (
    Route2ContinuumEngine,
)
from maple.function.route2_energy_ledger import (
    LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
    PCM_HALF_COUPLING_ONLY_V1,
    route2_energy_composition_description,
)
from maple.function.route2_smd_profiles import (
    DDPCM_SMD_DIRECT_PCM_PROFILE,
    PCMSOLVER_INTRINSIC_EXACT_GTO_DIRECT_PCM_PROFILE,
    route2_smd_profile_spec,
)


def _components() -> dict[str, float]:
    return {
        "solute_polarization": -0.20,
        "pcm_polarization": -0.30,
        "electrostatic": -0.50,
        "cds": 0.04,
        "standard_state": 0.01,
        "delta_g_solv": -0.45,
    }


def _canonical_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_route2_energy_ledger_separates_leaves_from_checked_totals():
    result = SolvationResult(
        energy_hartree=-0.45,
        components_hartree=_components(),
        forces_hartree_per_angstrom=np.array([[0.1, 0.2, 0.3]]),
        provenance={"provider": "test"},
    )

    assert dict(result.leaf_components_hartree) == {
        "solute_polarization": -0.20,
        "pcm_polarization": -0.30,
        "cds": 0.04,
        "standard_state": 0.01,
    }
    assert dict(result.derived_totals_hartree) == {
        "electrostatic": -0.50,
        "delta_g_solv": -0.45,
    }
    assert result.components_hartree["delta_g_solv"] == result.energy_hartree
    assert result.forces_hartree_per_angstrom.flags.writeable is False
    with pytest.raises(TypeError):
        result.leaf_components_hartree["cds"] = 1.0
    with pytest.raises(TypeError):
        result.derived_totals_hartree["delta_g_solv"] = 1.0
    with pytest.raises(ValueError):
        result.forces_hartree_per_angstrom[0, 0] = 1.0


@pytest.mark.parametrize(
    ("components", "energy_hartree", "message"),
    [
        (
            {**_components(), "electrostatic": -0.49},
            -0.45,
            "electrostatic does not close",
        ),
        (
            _components(),
            -0.44,
            "energy_hartree does not close",
        ),
        (
            {key: value for key, value in _components().items() if key != "cds"},
            -0.45,
            "missing=cds",
        ),
        (
            {**_components(), "total_summed_twice": -0.90},
            -0.45,
            "unknown=total_summed_twice",
        ),
    ],
)
def test_route2_energy_ledger_rejects_ambiguous_or_nonclosing_components(
    components,
    energy_hartree,
    message,
):
    with pytest.raises(ValueError, match=message):
        SolvationResult(
            energy_hartree=energy_hartree,
            components_hartree=components,
        )


def test_route2_energy_ledger_can_derive_totals_from_declared_leaves():
    leaves = {
        key: value
        for key, value in _components().items()
        if key not in {"electrostatic", "delta_g_solv"}
    }

    ledger = Route2EnergyLedger.from_components(leaves)

    assert ledger.derived_totals_hartree["electrostatic"] == pytest.approx(-0.50)
    assert ledger.derived_totals_hartree["delta_g_solv"] == pytest.approx(-0.45)


def test_calcabc_publishes_separate_route2_leaf_and_total_ledgers():
    class _Correction:
        supported_properties = {"energy"}

        @staticmethod
        def evaluate(_atoms, *, need_forces=False, calculator=None):
            del need_forces, calculator
            return SolvationResult(
                energy_hartree=-0.45,
                components_hartree=_components(),
                provenance={"provider": "test"},
            )

    calculator = CalcABC()
    calculator.solvent_correction = _Correction()
    calculator._finalize_results(
        Atoms("H"),
        energy=-1.0,
        unit="hartree",
    )

    solvation = calculator.results["solvation"]
    assert solvation["leaf_components_hartree"] == {
        "solute_polarization": -0.20,
        "pcm_polarization": -0.30,
        "cds": 0.04,
        "standard_state": 0.01,
    }
    assert solvation["derived_totals_hartree"] == {
        "electrostatic": -0.50,
        "delta_g_solv": -0.45,
    }
    assert solvation["components_hartree"] == {
        **solvation["leaf_components_hartree"],
        **solvation["derived_totals_hartree"],
    }


def test_calcabc_does_not_retry_internal_provider_type_errors():
    class _Correction:
        provider_api_version = 1
        supported_properties = {"energy"}

        def __init__(self):
            self.calls = 0

        def evaluate(self, _atoms, *, need_forces=False, calculator=None):
            del need_forces, calculator
            self.calls += 1
            raise TypeError("synthetic provider implementation failure")

    correction = _Correction()
    calculator = CalcABC()
    calculator.solvent_correction = correction

    with pytest.raises(TypeError, match="synthetic provider implementation failure"):
        calculator._finalize_results(
            Atoms("H"),
            energy=-1.0,
            unit="hartree",
        )
    assert correction.calls == 1


def test_correction_writes_the_checked_public_result_ledger(monkeypatch, tmp_path):
    import maple.function.calculator.extra_correction.implicit.correction as module

    class _Provider:
        supported_properties = frozenset({"energy"})

        def __init__(self, _atoms, _options, *, audit_dir):
            self.audit_dir = audit_dir
            self.provenance = {
                "provider": "pcmsolver",
                "profile": "smd-iefpcm",
            }

        @staticmethod
        def evaluate(_atoms, *, need_forces=False, calculator=None):
            del need_forces, calculator
            return SolvationResult(
                energy_hartree=-0.45,
                components_hartree=_components(),
                provenance={
                    "provider": "pcmsolver",
                    "profile": "smd-iefpcm",
                },
            )

    monkeypatch.setattr(module, "SMDImplicitSolvation", _Provider)
    initial_atoms = Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=np.diag([12.0, 13.0, 14.0]),
        pbc=[False, False, False],
    )
    correction = ImplicitSolvationCorrection(
        initial_atoms,
        {},
        {
            "method": "smd",
            "implicit": "water",
            "provider": "pcmsolver",
            "profile": "smd-iefpcm",
            "experimental": True,
        },
        output=tmp_path / "job.out",
    )

    first_atoms = initial_atoms.copy()
    second_atoms = initial_atoms.copy()
    second_atoms.positions[0, 0] = 0.25
    correction.evaluate(first_atoms)
    correction.evaluate(second_atoms)

    audit_dir = tmp_path / "job.out.implicit"
    manifest = json.loads((audit_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["initial_geometry"]["positions_angstrom"] == [[0.0, 0.0, 0.0]]
    assert manifest["initial_geometry_sha256"] == _canonical_sha256(
        manifest["initial_geometry"]
    )
    static_manifest = dict(manifest)
    static_manifest.pop("manifest_sha256")
    assert manifest["manifest_sha256"] == _canonical_sha256(static_manifest)
    assert manifest["source_receiver_contract"] == {
        "contract_version": "route2-source-receiver-contract-v1",
        "profile": "smd-iefpcm",
        "solute_source": "point-multipole-l1",
        "reaction_field_receiver": "local-jet",
        "continuum_pairing_status": "point-multipole-local-jet-dual",
        "continuum_pairing_established": True,
        "common_stationary_electronic_functional_established": False,
        "public_capability": "experimental-energy-only",
        "next_required_physical_gate": (
            "common-electronic-scalar-energy-density-conjugacy-and-kkt"
        ),
        "prohibited_claims": [
            "variational SCRF",
            "common-energy stationary electronic state",
            "solution-phase PES",
            "analytic solution-phase forces",
        ],
    }

    payload = json.loads(
        (tmp_path / "job.out.implicit" / "route2-public-result-ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["schema_version"] == 2
    assert payload["energy_hartree"] == pytest.approx(-0.45)
    assert payload["leaf_components_hartree"] == {
        "solute_polarization": -0.20,
        "pcm_polarization": -0.30,
        "cds": 0.04,
        "standard_state": 0.01,
    }
    assert payload["derived_totals_hartree"] == {
        "electrostatic": -0.50,
        "delta_g_solv": -0.45,
    }
    assert payload["evaluation_geometry"]["positions_angstrom"] == [
        [0.25, 0.0, 0.0]
    ]
    assert payload["base_manifest_sha256"] == manifest["manifest_sha256"]
    assert payload["source_receiver_contract"] == manifest[
        "source_receiver_contract"
    ]
    content = dict(payload)
    for key in (
        "result_content_sha256",
        "evaluation_manifest_path",
        "evaluation_manifest_sha256",
    ):
        content.pop(key)
    assert payload["result_content_sha256"] == _canonical_sha256(content)

    evaluation_manifest_path = audit_dir / payload["evaluation_manifest_path"]
    evaluation_manifest = json.loads(
        evaluation_manifest_path.read_text(encoding="utf-8")
    )
    manifest_content = dict(evaluation_manifest)
    manifest_content.pop("evaluation_manifest_sha256")
    assert evaluation_manifest["evaluation_manifest_sha256"] == _canonical_sha256(
        manifest_content
    )
    assert payload["evaluation_manifest_sha256"] == evaluation_manifest[
        "evaluation_manifest_sha256"
    ]
    assert evaluation_manifest["result_content_sha256"] == payload[
        "result_content_sha256"
    ]

    result_records = sorted(
        path
        for path in (audit_dir / "route2-public-results").glob("*.json")
        if not path.name.endswith(".manifest.json")
    )
    assert len(result_records) == 2
    records = [json.loads(path.read_text(encoding="utf-8")) for path in result_records]
    assert len({record["run_id"] for record in records}) == 2
    assert len({record["geometry_sha256"] for record in records}) == 2


def test_pcm_half_coupling_ledger_excludes_field_conditioned_mace_delta():
    common = {
        "gas_energy_ev": -100.0,
        "solvent_energy_ev": -99.5,
        "polarization_energy_hartree": -0.020,
        "cds_energy_hartree": 0.004,
    }

    legacy = Route2ContinuumEngine.compose_energy_components(
        **common,
        electrostatic_energy_ledger=LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
    )
    direct_pcm = Route2ContinuumEngine.compose_energy_components(
        **common,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    )

    field_conditioned_delta = 0.5 / Hartree
    assert legacy["solute_polarization"] == pytest.approx(
        field_conditioned_delta
    )
    assert direct_pcm["solute_polarization"] == 0.0
    assert direct_pcm["pcm_polarization"] == legacy["pcm_polarization"]
    assert direct_pcm["cds"] == legacy["cds"]
    assert direct_pcm["delta_g_solv"] == pytest.approx(-0.016)
    assert legacy["delta_g_solv"] == pytest.approx(
        direct_pcm["delta_g_solv"] + field_conditioned_delta
    )


@pytest.mark.parametrize(
    "profile",
    (
        DDPCM_SMD_DIRECT_PCM_PROFILE,
        PCMSOLVER_INTRINSIC_EXACT_GTO_DIRECT_PCM_PROFILE,
    ),
)
def test_direct_pcm_profiles_are_explicit_and_versioned(profile):
    spec = route2_smd_profile_spec(profile)

    assert spec.electrostatic_energy_ledger == PCM_HALF_COUPLING_ONLY_V1
    description = route2_energy_composition_description(
        spec.electrostatic_energy_ledger,
        continuum_symbol="PCM",
    )
    assert "0.5*<c_MACE-POLAR, f_reac_PCM>" in description
    assert "excluded" in description
