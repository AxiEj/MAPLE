from __future__ import annotations

import numpy as np
import pytest

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
  2.9800000000000000E+02  7.8497000000000000E+01  0.0  1.0 {spacing:.16E}
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


def _direct_table(*, radii: np.ndarray = RADII) -> str:
    raw = np.array(SHORT_RANGE, copy=True)
    positive = radii > 0.0
    raw[..., positive] -= (
        np.multiply.outer(CHARGES, CHARGES)[..., None] / radii[positive]
    )
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


def test_rism_bulk_coulomb_split_recovers_only_positive_radius_short_range_part():
    asset = parse_rism1d_direct_correlation(
        _direct_table(),
        metadata=parse_rism1d_xvv_metadata(_xvv()),
    )

    assert asset.coulomb_tail_residual(minimum_radius_angstrom=1.0) == pytest.approx(
        0.0,
        abs=1.0e-14,
    )
    short_range = asset.split_coulomb_long_range()
    np.testing.assert_allclose(short_range.radii_angstrom, RADII[1:])
    np.testing.assert_allclose(short_range.values_dimensionless, SHORT_RANGE[..., 1:])
    assert short_range.radii_angstrom[0] > 0.0


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
