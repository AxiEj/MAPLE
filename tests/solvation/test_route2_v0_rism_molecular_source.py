from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.route2_v0_rism_molecular_source import (
    V0_RISM_MOLECULAR_SOURCE_CONSTRUCTION,
    parse_route2_v0_rism1d_input,
    parse_route2_v0_rism_molecular_source,
    same_labelled_distance_geometry,
    same_labelled_rigid_geometry,
)

# Source-faithful compact fixtures copied from the AmberTools 26 cSPC/E model,
# input, and generated XVV metadata.  The correlation array is intentionally
# omitted because this module audits source identity rather than Cvv/Xvv data.
_CSPCE_MDL = """%VERSION  VERSION_STAMP = V0001.000  DATE = 05/12/09  17:44:15
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
  1.60000000e+01  1.00800000e+00
%FLAG CHG
%FORMAT(5e16.8)
 -1.54452215e+01  7.72261074e+00
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

_CSPCE_INPUT = """&PARAMETERS
    exchem_sc=1
    exchem_sm=1
    exchem_pr=1
    selftest=-1
    THEORY='DRISM', CLOSURE='PSE3',
    NR=16384, DR=0.025,
    OUTLIST='xc', ROUT=0, KOUT=0,
    MDIIS_NVEC=20, MDIIS_DEL=0.3, TOLERANCE=1.e-12,
    KSAVE=-1,
    MAXSTEP=10000,
    SMEAR=1, ADBCOR=0.5,
    TEMPERATURE=298, DIEPS=78.497, NSP=1
/
&SPECIES
    DENSITY=55.345,
    UNITS='M'
    MODEL="cSPCE.mdl"
/
"""

_CSPCE_XVV_METADATA = """%VERSION  VERSION_STAMP = V0001.001
%FLAG POINTERS
%FORMAT(10I8)
   16384       2       1
%FLAG THERMO
%FORMAT(1P5E24.16)
  2.9800000000000000E+02  7.8497000000000000E+01  0.0000000000000000E+00  2.6116114190532951E+00  2.5000000000000001E-02
  1.0000000000000000E+00
%FLAG ATOM_NAME
%FORMAT(20A4)
O   H1
%FLAG MTV
%FORMAT(10I8)
       1       2
%FLAG NVSP
%FORMAT(10I8)
       2
%FLAG MASS
%FORMAT(1P5E24.16)
  1.6000000000000000E+01  1.0080000000000000E+00
%FLAG RHOV
%FORMAT(1P5E24.16)
  3.3329515566149999E-02  6.6659031132299998E-02
%FLAG RHOSP
%FORMAT(1P5E24.16)
  3.3329515566149999E-02
%FLAG QV
%FORMAT(1P5E24.16)
 -2.0071094037563302E+01  1.0035547018781651E+01
%FLAG EPSV
%FORMAT(1P5E24.16)
  2.6224676781260281E-01  2.6224676781260278E-02
%FLAG RMIN2V
%FORMAT(1P5E24.16)
  1.7766999999999997E+00  6.5423795200000012E-01
%FLAG COORD
%FORMAT(1P3E24.16)
 -1.6024920012659376E-17 -2.2662169277074149E-17 -2.8867934551721919E-01
 -8.1649069689310616E-01  1.3877787807814457E-16  2.8867924486401886E-01
  8.1649069689310605E-01 -8.3266726846886741E-17  2.8867944617041941E-01
"""


def _parse_source(
    *,
    mdl: str = _CSPCE_MDL,
    rism1d_input: str = _CSPCE_INPUT,
    xvv: str = _CSPCE_XVV_METADATA,
    filename: str = "cSPCE.mdl",
):
    return parse_route2_v0_rism_molecular_source(
        mdl_text=mdl,
        rism1d_input=rism1d_input,
        xvv_text=xvv,
        site_model_filename=filename,
    )


def test_rism1d_input_parses_real_two_block_amber_convention():
    control = parse_route2_v0_rism1d_input(_CSPCE_INPUT)

    assert control.theory == "DRISM"
    assert control.closure == "PSE3"
    assert control.nr == 16384
    assert control.dr == pytest.approx(0.025)
    assert control.temperature_kelvin == pytest.approx(298.0)
    assert control.component_count == 1
    assert control.dielectric == pytest.approx(78.497)
    assert control.density == pytest.approx(0.03332953803622)
    assert control.coulomb_smear_angstrom == pytest.approx(1.0)
    assert control.residual_tolerance == pytest.approx(1.0e-12)
    assert control.self_test == -1
    assert control.output_list == "xc"
    assert control.maximum_steps == 10000
    assert control.model == "cSPCE.mdl"


def test_molecular_source_accepts_real_cspce_serialization():
    source = _parse_source()

    assert source.construction == V0_RISM_MOLECULAR_SOURCE_CONSTRUCTION
    np.testing.assert_array_equal(source.atomic_numbers, [8, 1, 1])
    assert source.atom_site_names == ("O", "H1", "H1")
    serialized_charges = np.asarray(
        [-15.4452215 / 18.2223, 7.72261074 / 18.2223, 7.72261074 / 18.2223]
    )
    np.testing.assert_allclose(
        source.serialized_site_charges_e,
        serialized_charges,
        rtol=0.0,
        atol=1.0e-14,
    )
    projected_charges = serialized_charges - np.sum(serialized_charges) / 3.0
    projected_charges[-1] -= np.sum(projected_charges)
    np.testing.assert_array_equal(source.site_charges_e, projected_charges)
    assert float(np.sum(source.site_charges_e)) == 0.0
    np.testing.assert_allclose(
        source.reference_positions_bohr,
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-0.333314, 0.942816, 0.0],
            ]
        )
        / Bohr,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert not source.atomic_numbers.flags.writeable
    assert not source.site_charges_e.flags.writeable
    assert not source.serialized_site_charges_e.flags.writeable
    assert not source.reference_positions_bohr.flags.writeable


def test_molecular_source_rejects_mdl_xvv_multiplicity_mismatch():
    xvv = _CSPCE_XVV_METADATA.replace(
        "       1       2\n%FLAG NVSP",
        "       2       1\n%FLAG NVSP",
    )
    xvv = xvv.replace(
        "3.3329515566149999E-02  6.6659031132299998E-02",
        "6.6659031132299998E-02  3.3329515566149999E-02",
    )
    xvv = xvv.replace(
        "-2.0071094037563302E+01  1.0035547018781651E+01",
        "1.0035547018781651E+01 -2.0071094037563302E+01",
    )
    with pytest.raises(ValueError, match="MULTI and XVV MTV"):
        _parse_source(xvv=xvv)


def test_molecular_source_rejects_qv_unit_mismatch():
    xvv = _CSPCE_XVV_METADATA.replace(
        "-2.0071094037563302E+01  1.0035547018781651E+01",
        "-1.0035547018781651E+01  5.0177735093908255E+00",
    )
    with pytest.raises(ValueError, match="CHG and XVV QV"):
        _parse_source(xvv=xvv)


def test_molecular_source_rejects_xvv_mass_lj_and_geometry_mismatch():
    mass = _CSPCE_XVV_METADATA.replace(
        "1.6000000000000000E+01", "1.5000000000000000E+01", 1
    )
    with pytest.raises(ValueError, match="site masses"):
        _parse_source(xvv=mass)

    epsilon = _CSPCE_XVV_METADATA.replace(
        "2.6224676781260281E-01", "3.6224676781260281E-01"
    )
    with pytest.raises(ValueError, match="LJEPSILON"):
        _parse_source(xvv=epsilon)

    geometry = _CSPCE_XVV_METADATA.replace(
        "8.1649069689310616E-01", "7.1649069689310616E-01"
    )
    with pytest.raises(ValueError, match="geometries"):
        _parse_source(xvv=geometry)


def test_molecular_source_rejects_virtual_and_united_atom_sites():
    virtual = _CSPCE_MDL.replace("O   H1", "EP1 H1")
    with pytest.raises(ValueError, match="virtual-site"):
        _parse_source(mdl=virtual)

    united_atom = _CSPCE_MDL.replace("1.60000000e+01", "1.90000000e+01")
    with pytest.raises(ValueError, match="united-atom"):
        _parse_source(mdl=united_atom)


def test_molecular_source_resolves_two_letter_elements_by_name_and_mass():
    mdl = _CSPCE_MDL.replace("O   H1", "CL1 H1")
    mdl = mdl.replace("1.60000000e+01", "3.54500000e+01")
    xvv = _CSPCE_XVV_METADATA.replace("O   H1", "CL1 H1")
    xvv = xvv.replace("1.6000000000000000E+01", "3.5450000000000000E+01")

    source = _parse_source(mdl=mdl, xvv=xvv)

    np.testing.assert_array_equal(source.atomic_numbers, [17, 1, 1])


def test_molecular_source_binds_rism1d_model_filename_and_state():
    with pytest.raises(ValueError, match="hash-bound MDL"):
        _parse_source(filename="other.mdl")

    density = _CSPCE_INPUT.replace("DENSITY=55.345", "DENSITY=50.000")
    with pytest.raises(ValueError, match="density disagree"):
        _parse_source(rism1d_input=density)

    with pytest.raises(ValueError, match="without a path"):
        _parse_source(filename="models/cSPCE.mdl")

    malformed = _CSPCE_INPUT.replace("KSAVE=-1,", "KSAVE=-1, unparsed-token,")
    with pytest.raises(ValueError, match="unparsed content"):
        _parse_source(rism1d_input=malformed)

    with pytest.raises(ValueError, match="unparsed content"):
        _parse_source(rism1d_input="orphan-token\n" + _CSPCE_INPUT)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("selftest=-1", "selftest=0", "SELFTEST=-1"),
        ("OUTLIST='xc'", "OUTLIST='x'", "request XVV and Cvv"),
        ("MAXSTEP=10000", "MAXSTEP=0", "positive integer"),
    ],
)
def test_rism1d_input_rejects_missing_generation_controls(old, new, message):
    with pytest.raises(ValueError, match=message):
        parse_route2_v0_rism1d_input(_CSPCE_INPUT.replace(old, new))


def test_labelled_geometry_rejects_a_chiral_mirror_but_accepts_proper_rotation():
    geometry = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    angle = 0.73
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translated = geometry @ rotation + np.asarray([2.0, -3.0, 4.0])
    labels = ("A", "B", "C", "D")

    assert same_labelled_rigid_geometry(geometry, labels, translated, labels)

    mirrored = translated.copy()
    mirrored[:, 0] *= -1.0
    # XVV inertia-axis signs are output conventions, so the MDL-to-XVV
    # source-identity boundary must accept this distance-equivalent reflection.
    assert same_labelled_distance_geometry(geometry, labels, mirrored, labels)
    # The canonical manifest-to-MDL boundary remains chirality preserving.
    assert not same_labelled_rigid_geometry(geometry, labels, mirrored, labels)
