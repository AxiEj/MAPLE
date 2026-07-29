from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from maple.function.calculator.extra_correction.implicit.route2_v0_rism_bulk import (
    V0_RISM_BULK_DIRECT_CORRELATION_CONSTRUCTION,
    load_rism1d_bulk_direct_correlation,
    parse_rism1d_direct_correlation,
    parse_rism1d_xvv_metadata,
)

RADII = np.array([0.0, 0.5, 1.0, 1.5])
CHARGES = np.array([-2.0, 1.0])
SHORT_RANGE = np.array(
    [
        [[7.0, 0.3, 0.0, 0.0], [5.0, -0.1, 0.0, 0.0]],
        [[5.0, -0.1, 0.0, 0.0], [3.0, 0.2, 0.0, 0.0]],
    ]
)


def _xvv(*, spacing: float = 0.5) -> str:
    return f"""%VERSION  VERSION_STAMP = V0001.001
%FLAG POINTERS
%FORMAT(10I8)
       4       2       1
%FLAG THERMO
%FORMAT(1P5E24.16)
  2.9800000000000000E+02  7.8497000000000000E+01  0.0  1.0 {spacing:.16E} 1.0000000000000000E+00
%FLAG ATOM_NAME
%FORMAT(20A4)
O
H1
%FLAG MTV
%FORMAT(10I8)
       1       2
%FLAG RHOV
%FORMAT(1P5E24.16)
  3.3000000000000000E-02  6.6000000000000000E-02
%FLAG QV
%FORMAT(1P5E24.16)
 -2.0000000000000000E+00  1.0000000000000000E+00
%FLAG XVV
%FORMAT(1P5E24.16)
  0.0
"""


def _direct_table(
    *,
    radii: np.ndarray = RADII,
    short_range: np.ndarray = SHORT_RANGE,
) -> str:
    raw = np.array(short_range, copy=True)
    kernel = np.empty_like(radii)
    positive = radii > 0.0
    kernel[positive] = erf(radii[positive]) / radii[positive]
    kernel[~positive] = 2.0 / np.sqrt(np.pi)
    raw -= np.multiply.outer(CHARGES, CHARGES)[..., None] * kernel
    lines = [
        "#RISM1D ATOM-ATOM INTERACTIONS: DIRECT CORRELATION VS. SEPARATION [A]",
        "#    SEPARATION          H1:O             O:O             H1:H1",
    ]
    for index, radius in enumerate(radii):
        lines.append(
            " ".join(
                f"{value:.16E}"
                for value in (
                    radius,
                    raw[1, 0, index],
                    raw[0, 0, index],
                    raw[1, 1, index],
                )
            )
        )
    return "\n".join(lines) + "\n"


def test_rism_bulk_parser_binds_cvv_pairs_to_xvv_metadata_and_files(tmp_path):
    xvv = _xvv()
    cvv = _direct_table()
    metadata = parse_rism1d_xvv_metadata(xvv)
    asset = parse_rism1d_direct_correlation(cvv, metadata=metadata)

    assert asset.construction == V0_RISM_BULK_DIRECT_CORRELATION_CONSTRUCTION
    assert asset.metadata.site_names == ("O", "H1")
    np.testing.assert_array_equal(asset.metadata.site_multiplicity, [1, 2])
    np.testing.assert_allclose(
        asset.metadata.bulk_number_density_angstrom3, [0.033, 0.066]
    )
    assert asset.metadata.coulomb_smear_angstrom == pytest.approx(1.0)
    np.testing.assert_allclose(
        asset.values_dimensionless, np.swapaxes(asset.values_dimensionless, 0, 1)
    )
    np.testing.assert_allclose(asset.radii_angstrom, RADII)

    xvv_path = tmp_path / "solvent.xvv"
    cvv_path = tmp_path / "solvent.cvv"
    xvv_path.write_text(xvv, encoding="utf-8")
    cvv_path.write_text(cvv, encoding="utf-8")
    loaded = load_rism1d_bulk_direct_correlation(
        xvv_path=xvv_path,
        cvv_path=cvv_path,
    )
    np.testing.assert_allclose(loaded.values_dimensionless, asset.values_dimensionless)


def test_rism_bulk_smooth_coulomb_split_recovers_the_full_radial_short_range_part():
    asset = parse_rism1d_direct_correlation(
        _direct_table(),
        metadata=parse_rism1d_xvv_metadata(_xvv()),
    )

    short_range = asset.split_coulomb_long_range()
    np.testing.assert_allclose(short_range.radii_angstrom, RADII)
    np.testing.assert_allclose(short_range.values_dimensionless, SHORT_RANGE)
    tail = asset.smoothed_coulomb_tail_dimensionless()
    assert np.all(np.isfinite(tail))
    assert tail[0, 0, 0] == pytest.approx(
        -(CHARGES[0] ** 2) * 2.0 / np.sqrt(np.pi),
    )


def test_rism_bulk_bare_tail_residual_remains_an_asymptotic_diagnostic():
    radii = np.array([0.0, 10.0, 20.0, 30.0])
    asset = parse_rism1d_direct_correlation(
        _direct_table(
            radii=radii,
            short_range=np.zeros((2, 2, radii.size)),
        ),
        metadata=parse_rism1d_xvv_metadata(_xvv(spacing=10.0)),
    )

    assert asset.coulomb_tail_residual(minimum_radius_angstrom=10.0) == pytest.approx(
        0.0,
        abs=1.0e-14,
    )


def test_rism_bulk_rejects_incompatible_cvv_grid_and_incomplete_site_pairs():
    metadata = parse_rism1d_xvv_metadata(_xvv())
    with pytest.raises(ValueError, match="spacing"):
        parse_rism1d_direct_correlation(
            _direct_table(radii=np.array([0.0, 0.4, 0.8, 1.2])),
            metadata=metadata,
        )

    incomplete = """# SEPARATION O:O H1:H1
0.0 1.0 2.0
0.5 1.0 2.0
1.0 1.0 2.0
1.5 1.0 2.0
"""
    with pytest.raises(ValueError, match="complete known site-pair"):
        parse_rism1d_direct_correlation(incomplete, metadata=metadata)


def test_rism_bulk_rejects_xvv_metadata_with_a_site_count_mismatch():
    malformed = _xvv().replace("O\nH1", "O")
    with pytest.raises(ValueError, match="ATOM_NAME count"):
        parse_rism1d_xvv_metadata(malformed)


def test_rism_bulk_rejects_a_nonneutral_multiplicity_weighted_site_charge():
    nonneutral = _xvv().replace(
        "-2.0000000000000000E+00  1.0000000000000000E+00",
        "-2.0000000000000000E+00  9.0000000000000000E-01",
    )
    with pytest.raises(ValueError, match="neutral after multiplicity"):
        parse_rism1d_xvv_metadata(nonneutral)


def test_rism_bulk_rejects_an_absent_or_nonpositive_source_smear():
    missing = _xvv().replace(
        " 1.0000000000000000E+00\n%FLAG ATOM_NAME", "\n%FLAG ATOM_NAME"
    )
    with pytest.raises(ValueError, match="SMEAR"):
        parse_rism1d_xvv_metadata(missing)

    zero = _xvv().replace(
        " 1.0000000000000000E+00\n%FLAG ATOM_NAME",
        " 0.0000000000000000E+00\n%FLAG ATOM_NAME",
    )
    with pytest.raises(ValueError, match="SMEAR"):
        parse_rism1d_xvv_metadata(zero)
