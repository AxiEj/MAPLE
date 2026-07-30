from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_gfn2_molden_permanent_source import (
    V0_GFN2_MOLDEN_PERMANENT_SOURCE_CONSTRUCTION,
    V0_GFN2_MOLDEN_PERMANENT_SOURCE_SCOPE,
    load_route2_v0_gfn2_effective_core_model,
    load_route2_v0_gfn2_molden_ao_density,
    load_route2_v0_gfn2_molden_permanent_reference,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _parameter_file(path: Path) -> Path:
    path.write_text(
        """$info
name GFN2-xTB
$end
$Z= 1
 ao=1s
$end
$Z= 6
 ao=2s2p
$end
$Z= 8
 ao=2s2p
$end
$Z= 16
 ao=3s3p3d
$end
""",
        encoding="utf-8",
    )
    return path


def _molden_file(
    path: Path,
    *,
    position: tuple[float, float, float] = (0.7, -0.2, 0.4),
    coefficient_scale: float = 1.0,
    shell_scale: float = 1.0,
    spin: str = "Alpha",
) -> Path:
    primitive_coefficient = coefficient_scale * (2.0 / math.pi) ** 0.75
    path.write_text(
        f"""[Molden Format]
[Title]
[Atoms] AU
H 1 1 {position[0]:.16f} {position[1]:.16f} {position[2]:.16f}
[GTO]
1 0
s 1 {shell_scale:.16f}
1.0 {primitive_coefficient:.16f}

[MO]
Sym= 1a
Ene= -0.5
Spin= {spin}
Occup= 2.0
1 1.0
""",
        encoding="utf-8",
    )
    return path


def _xtbout_file(path: Path, *, dipole: np.ndarray, command: str | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "method": "GFN2-xTB",
                "xtb version": "6.7.1 (test)",
                "number of molecular orbitals": 1,
                "number of electrons": 2,
                "number of unpaired electrons": 0,
                "dipole / a.u.": dipole.tolist(),
                "total energy": -0.5,
                "program call": (
                    "xtb molecule.xyz --gfn 2 --molden --json"
                    if command is None
                    else command
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _stdout_file(path: Path) -> Path:
    path.write_text(
        """ :  GBSA solvation                  false          :
 :  PC potential                    false          :
 *** convergence criteria satisfied after 1 iterations ***
""",
        encoding="utf-8",
    )
    return path


def test_gfn2_molden_closed_shell_source_round_trips_metric_count_and_dipole(
    tmp_path: Path,
):
    position = np.array([0.7, -0.2, 0.4])
    molden = _molden_file(tmp_path / "molden.input", position=tuple(position))
    parameter = _parameter_file(tmp_path / "param_gfn2-xtb.txt")
    xtbout = _xtbout_file(tmp_path / "xtbout.json", dipole=-position)
    stdout = _stdout_file(tmp_path / "stdout.txt")

    density = load_route2_v0_gfn2_molden_ao_density(molden)
    core_model = load_route2_v0_gfn2_effective_core_model(parameter)
    reference = load_route2_v0_gfn2_molden_permanent_reference(
        molden_path=molden,
        xtbout_json_path=xtbout,
        stdout_path=stdout,
        parameter_file_path=parameter,
        expected_net_charge_e=-1.0,
    )

    assert density.construction == V0_GFN2_MOLDEN_PERMANENT_SOURCE_CONSTRUCTION
    assert density.scope == V0_GFN2_MOLDEN_PERMANENT_SOURCE_SCOPE
    assert density.ao_count == 1
    assert density.mo_metric_error < 2.0e-14
    assert density.valence_electron_count_e == pytest.approx(2.0, abs=2.0e-14)
    np.testing.assert_allclose(
        density.valence_electronic_dipole_e_bohr(), 2.0 * position, atol=2.0e-14
    )
    assert core_model.charges_for(np.array([1])).tolist() == [1.0]
    assert reference.total_effective_charge_e == pytest.approx(-1.0, abs=2.0e-14)
    np.testing.assert_allclose(
        reference.reconstructed_total_dipole_e_bohr, -position, atol=2.0e-14
    )
    assert reference.dipole_round_trip_error_e_bohr < 2.0e-14

    grid = RegularCartesianGrid(
        origin_bohr=np.array([-1.6, -1.7, -1.5]),
        spacing_bohr=np.array([0.42, 0.42, 0.42]),
        shape=(10, 10, 10),
    )
    projection = reference.grid_projection(grid, maximum_points_per_block=37)
    state = projection.evaluate(density.density_matrix)
    potential = np.linspace(-0.1, 0.2, grid.point_count).reshape(grid.shape)
    assert (
        projection.density_potential_pairing_error_hartree(
            density.density_matrix, potential
        )
        < 2.0e-14
    )
    assert state.ao_electron_count_e == pytest.approx(2.0, abs=2.0e-14)


def test_gfn2_effective_core_model_is_derived_from_parameter_shells(tmp_path: Path):
    core_model = load_route2_v0_gfn2_effective_core_model(
        _parameter_file(tmp_path / "param_gfn2-xtb.txt")
    )
    np.testing.assert_allclose(
        core_model.charges_for(np.array([1, 6, 8, 16])),
        np.array([1.0, 4.0, 6.0, 6.0]),
        atol=0.0,
    )
    with pytest.raises(ValueError, match="no effective core declaration"):
        core_model.charges_for(np.array([17]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"coefficient_scale": 1.1}, "MO coefficients do not satisfy"),
        ({"shell_scale": 1.1}, "scale must equal one"),
        ({"spin": "Beta"}, "requires closed-shell Alpha"),
    ],
)
def test_gfn2_molden_loader_fails_closed_on_ambiguous_or_nonmetric_exports(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        load_route2_v0_gfn2_molden_ao_density(
            _molden_file(tmp_path / "molden.input", **kwargs)
        )


def test_gfn2_molden_permanent_reference_rejects_a_solvent_or_field_command(
    tmp_path: Path,
):
    molden = _molden_file(tmp_path / "molden.input")
    parameter = _parameter_file(tmp_path / "param_gfn2-xtb.txt")
    xtbout = _xtbout_file(
        tmp_path / "xtbout.json",
        dipole=np.array([-0.7, 0.2, -0.4]),
        command="xtb molecule.xyz --gfn 2 --molden --json --cosmo water",
    )
    with pytest.raises(ValueError, match="forbidden field or solvent"):
        load_route2_v0_gfn2_molden_permanent_reference(
            molden_path=molden,
            xtbout_json_path=xtbout,
            stdout_path=_stdout_file(tmp_path / "stdout.txt"),
            parameter_file_path=parameter,
            expected_net_charge_e=-1.0,
        )


def test_gfn2_molden_permanent_reference_rejects_a_dipole_that_is_not_source_bound(
    tmp_path: Path,
):
    molden = _molden_file(tmp_path / "molden.input")
    parameter = _parameter_file(tmp_path / "param_gfn2-xtb.txt")
    with pytest.raises(ValueError, match="does not reproduce the xTB total dipole"):
        load_route2_v0_gfn2_molden_permanent_reference(
            molden_path=molden,
            xtbout_json_path=_xtbout_file(
                tmp_path / "xtbout.json", dipole=np.array([1.0, 2.0, 3.0])
            ),
            stdout_path=_stdout_file(tmp_path / "stdout.txt"),
            parameter_file_path=parameter,
            expected_net_charge_e=-1.0,
        )
