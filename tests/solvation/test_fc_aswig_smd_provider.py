"""Public wiring and fail-closed derivative evidence for the FC-aSWIG profile."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.correction import (
    ImplicitSolvationCorrection,
)
from maple.function.calculator.extra_correction.implicit.fc_aswig_smd import (
    FC_ASWIG_DERIVATIVE_EVIDENCE_ONLY_ERROR,
    FC_ASWIG_SCF_SOLVER,
    FixedTopologyASWIGAqueousSMDImplicitSolvation,
)
from maple.function.calculator.extra_correction.implicit.route2_fixed_point import (
    SAFEGUARDED_ANDERSON_SOLVER,
)
from maple.function.route2_smd_profiles import (
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE,
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE,
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE,
    MACEPOL_MOLECULAR_REALSPACE_PROFILE,
)


def _atoms() -> Atoms:
    atoms = Atoms(
        "CONH",
        positions=np.asarray(
            [
                [-1.30, -0.30, 0.20],
                [-0.10, 0.60, -0.30],
                [1.05, 0.10, 0.25],
                [1.55, 0.85, 0.55],
            ]
        ),
    )
    atoms.info["charge"] = 0
    atoms.info["mult"] = 1
    return atoms


def _options(**overrides):
    return {
        "method": "smd",
        "implicit": "water",
        "provider": "fc-aswig",
        "profile": FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE,
        "response": "scf",
        "standard_state": "1m",
        **overrides,
    }


class _ZeroResponse:
    def __init__(self, atom_count: int):
        self.shape = (atom_count, 4)

    def jvp(self, _field_direction):
        return np.zeros(self.shape)

    def vjp(self, _density_cotangent):
        return np.zeros(self.shape)


class _ZeroMACEPolarCalculator:
    """Minimal exact fixed point used only to exercise provider composition."""

    def __init__(self, atoms: Atoms):
        self.atoms = atoms.copy()
        self.route2_mace_geometry_frame_policy = "jgp94-d2-canonical-v1"
        self.long_range_evaluator_profile = MACEPOL_MOLECULAR_REALSPACE_PROFILE
        density = np.zeros((len(atoms), 4))
        # Deliberately distinct from the zero and negated-gas seeds used by
        # the root-study evidence.  The field response below is exactly zero,
        # so all three starts still converge to the same zero root.
        density[:, 0] = np.asarray([0.12, -0.05, -0.07, 0.0])
        forces = np.zeros((len(atoms), 3))
        self._gas_state = SimpleNamespace(
            energy_ev=-10.0,
            density_coefficients=density,
            fixed_field_forces_ev_per_angstrom=forces,
        )
        self._last_polar_state = self._gas_state

    def cached_polar_state(self, _atoms, *, require_forces=False):
        del require_forces
        return self._gas_state

    def polar_state(
        self,
        atoms,
        *,
        node_potential_ev=None,
        node_gradient_ev_per_angstrom=None,
        compute_forces=False,
        **_kwargs,
    ):
        del node_potential_ev, node_gradient_ev_per_angstrom, compute_forces
        return (
            SimpleNamespace(
                energy_ev=-10.0,
                density_coefficients=np.zeros((len(atoms), 4)),
                fixed_field_forces_ev_per_angstrom=np.zeros((len(atoms), 3)),
            ),
            {},
        )

    def linearize_density_response(self, atoms, **_kwargs):
        return _ZeroResponse(len(atoms))

    def density_position_vjp(self, atoms, **_kwargs):
        return np.zeros((len(atoms), 3))


def test_fixed_topology_aqueous_provider_exposes_energy_and_labelled_derivative_evidence():
    pytest.importorskip("pyscf")
    atoms = _atoms()
    calculator = _ZeroMACEPolarCalculator(atoms)
    provider = FixedTopologyASWIGAqueousSMDImplicitSolvation(
        atoms,
        _options(),
    )
    result = provider.evaluate(atoms, calculator=calculator)
    assert result.forces_hartree_per_angstrom is None
    assert result.provenance["electrostatics_model"] == "cpcm"
    assert result.provenance["forces_available"] is False
    assert result.provenance["scf_solver"] == FC_ASWIG_SCF_SOLVER
    assert result.provenance["scf_solver"] == SAFEGUARDED_ANDERSON_SOLVER
    assert result.components_hartree["solute_polarization"] == 0.0
    assert result.components_hartree["pcm_polarization"] == 0.0

    with pytest.raises(NotImplementedError, match="derivative evidence only"):
        provider.evaluate(atoms, calculator=calculator, need_forces=True)

    evidence = provider.evaluate_single_point_derivative_evidence(
        atoms,
        calculator=calculator,
    )
    assert evidence.provenance["research_derivative_evidence"] is True
    assert evidence.provenance["forces_available"] is False
    assert evidence.provenance["multi_start_root_agreement"]["agreed"] is True
    certificate = evidence.provenance["force_admission"]
    assert certificate["release_admitted"] is False
    assert "component-resolved-force-finite-difference-unverified" in (
        certificate["failure_reasons"]
    )


def test_fixed_topology_provider_rejects_unregistered_mixtures():
    atoms = _atoms()
    with pytest.raises(ValueError, match="supports water only"):
        FixedTopologyASWIGAqueousSMDImplicitSolvation(
            atoms,
            _options(implicit="methanol"),
        )
    with pytest.raises(ValueError, match="requires response='scf'"):
        FixedTopologyASWIGAqueousSMDImplicitSolvation(
            atoms,
            _options(response="frozen"),
        )
    with pytest.raises(ValueError, match="requires profile"):
        FixedTopologyASWIGAqueousSMDImplicitSolvation(
            atoms,
            _options(profile="smd-ddpcm-l15-n1202-v1"),
        )
    assert "fixed-topology" in FC_ASWIG_DERIVATIVE_EVIDENCE_ONLY_ERROR


def test_fixed_topology_provider_accepts_the_versioned_canonical_mace_operator():
    atoms = _atoms()
    provider = FixedTopologyASWIGAqueousSMDImplicitSolvation(
        atoms,
        _options(profile=FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE),
    )
    assert provider.profile == FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE
    assert provider.provenance["mace_geometry_frame_policy"] == (
        "jgp94-d2-canonical-v1"
    )
    assert provider.supported_properties == frozenset({"energy"})


def test_force_v3_is_the_only_public_fixed_topology_force_profile():
    pytest.importorskip("pyscf")
    atoms = _atoms()
    calculator = _ZeroMACEPolarCalculator(atoms)
    provider = FixedTopologyASWIGAqueousSMDImplicitSolvation(
        atoms,
        _options(profile=FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE),
    )

    assert provider.supported_properties == frozenset({"energy", "forces"})
    result = provider.evaluate(atoms, calculator=calculator, need_forces=True)

    assert result.forces_hartree_per_angstrom is not None
    assert result.provenance["forces_available"] is True
    assert result.provenance["force_admission"]["release_admitted"] is True


def test_force_v3_rejects_a_noncanonical_mace_transform():
    pytest.importorskip("pyscf")
    atoms = _atoms()
    calculator = _ZeroMACEPolarCalculator(atoms)
    calculator.route2_mace_geometry_frame_policy = "laboratory-v1"
    provider = FixedTopologyASWIGAqueousSMDImplicitSolvation(
        atoms,
        _options(profile=FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE),
    )

    with pytest.raises(TypeError, match="JGP94-D2"):
        provider.evaluate(atoms, calculator=calculator, need_forces=True)


def test_force_v3_opens_the_common_route2_correction_boundary(tmp_path):
    pytest.importorskip("pyscf")
    atoms = _atoms()
    calculator = _ZeroMACEPolarCalculator(atoms)
    correction = ImplicitSolvationCorrection(
        atoms,
        {},
        _options(
            profile=FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE,
            experimental=True,
        ),
        output=tmp_path / "route2-force.out",
    )

    assert correction.supported_properties == {"energy", "forces"}
    result = correction.evaluate(atoms, calculator=calculator, need_forces=True)

    assert result.forces_hartree_per_angstrom is not None
    assert result.provenance["force_admission"]["release_admitted"] is True
