from __future__ import annotations

import re

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    SurfaceChargeState,
)
from maple.function.mlip_cosmo_rs import (
    MLIPCOSMORSSurface,
    OPEN_COSMORS_24A_AUDITED_RADII_BOHR,
    open_cosmors_24a_cavity_radii,
    render_mlip_orcacosmo,
)


class _FakeConductorResponse:
    runtime_provenance = {
        "provider": "fake-conductor",
        "conductor_limit": True,
    }
    cavity_radii_angstrom = np.asarray([1.30, 1.72])
    surface_points_bohr = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.8, 0.0, 0.0],
            [1.8, 1.0, 0.0],
        ]
    )
    surface_areas_bohr2 = np.asarray(
        [
            0.2 / Bohr**2,
            0.005 / Bohr**2,
            0.3 / Bohr**2,
            0.4 / Bohr**2,
        ]
    )
    surface_parent_atom_indices = np.asarray([0, 0, 1, 1])

    @staticmethod
    def solve(surface_potential_hartree_per_e):
        potential = np.asarray(surface_potential_hartree_per_e, dtype=float)
        charge = np.asarray([0.10, -0.02, -0.04, -0.04])
        return SurfaceChargeState(
            surface_potential_hartree_per_e=potential,
            direct_surface_charge_e=charge,
            adjoint_surface_charge_e=charge,
            energy_conjugate_surface_charge_e=charge,
            polarization_energy_hartree=0.5 * float(np.dot(potential, charge)),
        )


def _profile() -> MLIPCOSMORSSurface:
    return MLIPCOSMORSSurface.from_fixed_multipoles(
        _FakeConductorResponse(),
        ("H", "O"),
        np.asarray([[-0.5, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        np.asarray(
            [
                [0.2, 0.0, 0.0, 0.0],
                [-0.2, 0.0, 0.0, 0.0],
            ]
        ),
        source_id="aimnet2-fixed-l0",
    )


def test_open_cosmors_24a_radii_are_bounded_and_explicit():
    np.testing.assert_allclose(
        open_cosmors_24a_cavity_radii(("H", "C", "O", "Cl", "Br")),
        [1.30, 2.00, 1.72, 2.05, 2.16],
    )
    with pytest.raises(ValueError, match="No audited"):
        open_cosmors_24a_cavity_radii(("Xe",))
    assert OPEN_COSMORS_24A_AUDITED_RADII_BOHR["C"] == 3.779452268


def test_mlip_cosmors_surface_filters_small_segments_and_closes_charge():
    profile = _profile()

    assert profile.surface_charge_e.size == 3
    assert profile.discarded_segment_count == 1
    assert profile.pre_correction_surface_charge_e == pytest.approx(0.02)
    assert float(np.sum(profile.surface_charge_e)) == pytest.approx(
        0.0,
        abs=1.0e-14,
    )
    assert profile.conductor_energy_hartree - profile.gas_energy_hartree == (
        pytest.approx(profile.polarization_energy_hartree)
    )
    assert profile.cavity_volume_bohr3 > 0.0
    manifest = profile.as_manifest()
    assert manifest["qm_solute_calculation_required"] is False
    assert manifest["strict_open_cosmors_24a_parameterization_equivalence"] is (
        False
    )


def test_mlip_orcacosmo_renderer_emits_the_upstream_minimum_contract():
    profile = _profile()

    rendered = render_mlip_orcacosmo(profile, name="water-canary")

    assert rendered.startswith(
        "water-canary : MLIP_aimnet2-fixed-l0_CPCM_CONDUCTOR\n"
    )
    assert "#ENERGY\nFINAL SINGLE POINT ENERGY" in rendered
    assert "#XYZ_FILE\n2\n" in rendered
    assert "# SURFACE POINTS (A.U.)" in rendered
    assert "#COSMO_corrected" in rendered
    assert "2.456643974 1\n" in rendered
    assert "3.250328950 8\n" in rendered
    corrected = rendered.split("C-PCM corrected charges:\n", maxsplit=1)[1]
    corrected = corrected.split(
        "##################################################",
        maxsplit=1,
    )[0]
    corrected_charges = [
        float(value)
        for value in corrected.splitlines()
        if value.strip()
    ]
    assert len(corrected_charges) == profile.surface_charge_e.size
    assert sum(corrected_charges) == pytest.approx(0.0, abs=2.0e-14)
    assert re.search(r"\n3 # Number of surface points\n", rendered)

    with pytest.raises(ValueError, match="safe line"):
        render_mlip_orcacosmo(profile, name="bad:name")


def test_mlip_cosmors_bridge_rejects_nonconductor_response():
    response = _FakeConductorResponse()
    response.runtime_provenance = {"conductor_limit": False}

    with pytest.raises(ValueError, match="conductor response"):
        MLIPCOSMORSSurface.from_fixed_multipoles(
            response,
            ("H", "O"),
            np.asarray([[-0.5, 0.0, 0.0], [0.8, 0.0, 0.0]]),
            np.zeros((2, 4)),
            source_id="test",
        )


def test_mlip_cosmors_bridge_rejects_non_parameterized_radii():
    response = _FakeConductorResponse()
    response.cavity_radii_angstrom = np.asarray([1.31, 1.72])

    with pytest.raises(ValueError, match="audited openCOSMO-RS 24a"):
        MLIPCOSMORSSurface.from_fixed_multipoles(
            response,
            ("H", "O"),
            np.asarray([[-0.5, 0.0, 0.0], [0.8, 0.0, 0.0]]),
            np.zeros((2, 4)),
            source_id="test",
        )
