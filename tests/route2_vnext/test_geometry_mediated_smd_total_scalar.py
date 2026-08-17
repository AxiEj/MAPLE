from __future__ import annotations

import hashlib
from types import SimpleNamespace

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.api import (
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_PROFILE_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_SMDCDS_PROFILE_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_SMDCDS_V1,
    PROFILE_REGISTRY,
    SCALAR_REGISTRY,
)
from maple.solvation.api.profiles import AIMNET2_FROZEN_CHARGE_MODEL_PROFILE_ID
from maple.solvation.continuum.harmonic_point_ddpcm_torch_functional import (
    build_water_aimnet2_frozen_charge_harmonic_ddpcm_candidate,
)
from maple.solvation.coupling.geometry_mediated import (
    GeometryMediatedElectrostaticScalar,
)
from maple.solvation.coupling.geometry_mediated_smd import (
    GeometryMediatedSMDTotalScalar,
    PYSCF_WATER_SMD_CDS_NONPOLAR_PROFILE_ID,
    PySCFSMDCDSNonpolarFunctional,
)
from maple.solvation.models import (
    AIMNet2CheckpointContract,
    AIMNet2GeometryMediatedModelAdapter,
)

_CHARGE_SLOPE = 0.04


class _FakeFloat64AIMNet2:
    model_name = "aimnet2"
    _supports_atom_charges = True
    _coulomb_method = "simple"
    solvent_correction = None
    inference_dtype = "float64"
    device = "cpu"
    cutoff = 5.0
    cutoff_lr = float("inf")

    def __init__(self, model_path):
        self.model_path = str(model_path)

    @staticmethod
    def _charges(atoms):
        x = np.asarray(atoms.positions, dtype=float)[:, 0]
        base = np.asarray([-0.65, 0.325, 0.325])
        return base + _CHARGE_SLOPE * (x - float(np.mean(x)))

    def charge_state(self, atoms):
        return SimpleNamespace(
            energy_ev=0.5 * float(np.vdot(atoms.positions, atoms.positions)),
            charges_e=self._charges(atoms),
            requested_total_charge_e=0.0,
            model_name="aimnet2",
        )

    def charge_position_response(self, atoms, charge_cotangent_ev_per_e):
        cotangent = np.asarray(charge_cotangent_ev_per_e, dtype=float)
        charge_vjp = np.zeros_like(atoms.positions)
        charge_vjp[:, 0] = _CHARGE_SLOPE * (cotangent - float(np.mean(cotangent)))
        return SimpleNamespace(
            charge_state=self.charge_state(atoms),
            charge_cotangent_ev_per_e=cotangent.copy(),
            intrinsic_energy_gradient_ev_per_angstrom=np.array(
                atoms.positions, copy=True
            ),
            charge_position_vjp_ev_per_angstrom=charge_vjp,
        )


def _atoms() -> Atoms:
    return Atoms(
        "OHH",
        positions=[
            [0.0000000000, 0.0000000000, 0.0000000000],
            [0.9572000000, 0.0000000000, 0.0000000000],
            [-0.2399872000, 0.9272970000, 0.0000000000],
        ],
        info={"charge": 0, "mult": 1},
    )


def _total_scalar(tmp_path) -> GeometryMediatedSMDTotalScalar:
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyscf")
    checkpoint = tmp_path / "aimnet2.pt"
    checkpoint.write_bytes(b"synthetic frozen-charge water checkpoint\n")
    contract = AIMNet2CheckpointContract(
        provider_id="maple.route2.model.test-aimnet2-frozen-water.impl.v1",
        model_profile_id=AIMNET2_FROZEN_CHARGE_MODEL_PROFILE_ID,
        checkpoint_identifier="synthetic-test",
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_size_bytes=checkpoint.stat().st_size,
        model_name="aimnet2",
        coulomb_method="simple",
        inference_dtype="float64",
        supported_atomic_numbers=(1, 6, 7, 8),
        upstream_repository="test",
        publication_doi="10.1039/D4SC08572H",
        upstream_version="test",
        upstream_commit="test",
        checkpoint_origin_status="synthetic-test-only",
    )
    model = AIMNet2GeometryMediatedModelAdapter(
        _FakeFloat64AIMNet2(checkpoint), contract
    )
    continuum = build_water_aimnet2_frozen_charge_harmonic_ddpcm_candidate(
        tuple(_atoms().get_chemical_symbols()),
        dtype=torch.float64,
        device="cpu",
    )
    electrostatic = GeometryMediatedElectrostaticScalar(
        model,
        continuum,
        scalar_id=(
            CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1
        ),
        profile_id=(
            CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_PROFILE_V1
        ),
    )
    return GeometryMediatedSMDTotalScalar(
        electrostatic,
        PySCFSMDCDSNonpolarFunctional(),
    )


def test_total_smd_scalar_is_registered_disabled_and_component_bound(tmp_path):
    scalar = _total_scalar(tmp_path)
    definition = SCALAR_REGISTRY[
        CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_SMDCDS_V1
    ]
    profile = PROFILE_REGISTRY[
        CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_SMDCDS_PROFILE_V1
    ]
    assert definition.enabled is False
    assert profile.enabled is False
    assert definition.admitted_capabilities.enabled_tiers == ()
    assert profile.capabilities.enabled_tiers == ()
    assert profile.nonpolar_profile == PYSCF_WATER_SMD_CDS_NONPOLAR_PROFILE_ID
    assert "pyscf_2p13p1_water_smd_cds_energy" in definition.included_components
    assert "electronic_scf_iteration" in definition.excluded_components
    assert len(scalar.fingerprint_sha256()) == 64


def test_total_smd_scalar_energy_and_gradient_ledgers_match_directional_fd(tmp_path):
    scalar = _total_scalar(tmp_path)
    atoms = _atoms()
    result = scalar.evaluate(atoms)
    direction = np.asarray(
        [
            [0.12, -0.07, 0.03],
            [-0.05, 0.11, -0.02],
            [-0.07, -0.04, -0.01],
        ]
    )
    direction /= np.linalg.norm(direction)
    step = 2.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * direction
    minus.positions -= step * direction
    finite_difference = (
        scalar.evaluate_energy(plus) - scalar.evaluate_energy(minus)
    ) / (2.0 * step)
    analytic = float(np.vdot(result.total_gradient_eV_per_A, direction))

    assert result.energy.total_energy_eV == pytest.approx(
        result.energy.vacuum_energy_eV
        + result.energy.continuum_energy_eV
        + result.energy.nonpolar_energy_eV,
        abs=1.0e-12,
    )
    np.testing.assert_allclose(
        result.total_gradient_eV_per_A,
        result.electrostatic_total_gradient_eV_per_A
        + result.nonpolar_gradient_eV_per_A,
        atol=0.0,
        rtol=0.0,
    )
    assert analytic == pytest.approx(finite_difference, abs=1.0e-6)
    assert np.linalg.norm(result.nonpolar_gradient_eV_per_A) > 0.0
    assert result.nonpolar.runtime_provenance["pyscf_version"] == "2.13.1"
    assert result.total_gradient_eV_per_A.flags.writeable is False
    assert result.forces_eV_per_A.flags.writeable is False


def test_total_smd_scalar_hvp_fails_closed(tmp_path):
    scalar = _total_scalar(tmp_path)
    with pytest.raises(NotImplementedError, match="no admitted same-scalar HVP"):
        scalar.hessian_vector_product(_atoms(), np.ones((3, 3)))
