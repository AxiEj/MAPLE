from __future__ import annotations

import textwrap

import pytest


@pytest.fixture
def water_mol2(tmp_path):
    path = tmp_path / "water.mol2"
    path.write_text(
        textwrap.dedent(
            """\
            @<TRIPOS>MOLECULE
            WATER
             3 2 1 0 0
            SMALL
            USER_CHARGES

            @<TRIPOS>ATOM
                  1 O1          0.000000    0.000000    0.000000 O.3       1 WAT       -0.834000
                  2 H1          0.957200    0.000000    0.000000 H         1 WAT        0.417000
                  3 H2         -0.239987    0.927297    0.000000 H         1 WAT        0.417000
            @<TRIPOS>BOND
                 1    1    2 1
                 2    1    3 1
            @<TRIPOS>SUBSTRUCTURE
                 1 WAT         1 TEMP              0 ****  ****    0 ROOT
            """
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def methanol_mol2(tmp_path):
    path = tmp_path / "methanol.mol2"
    path.write_text(
        textwrap.dedent(
            """\
            @<TRIPOS>MOLECULE
            METHANOL
             6 5 1 0 0
            SMALL
            USER_CHARGES

            @<TRIPOS>ATOM
                  1 C1          0.000000    0.000000    0.000000 C.3       1 MOL       -0.200000
                  2 O1          1.430000    0.000000    0.000000 O.3       1 MOL       -0.600000
                  3 H1         -0.360000    1.030000    0.000000 H         1 MOL        0.100000
                  4 H2         -0.360000   -0.515000    0.892000 H         1 MOL        0.100000
                  5 H3         -0.360000   -0.515000   -0.892000 H         1 MOL        0.100000
                  6 H4          1.790000    0.900000    0.000000 H         1 MOL        0.500000
            @<TRIPOS>BOND
                 1    1    2 1
                 2    1    3 1
                 3    1    4 1
                 4    1    5 1
                 5    2    6 1
            @<TRIPOS>SUBSTRUCTURE
                 1 MOL         1 TEMP              0 ****  ****    0 ROOT
            """
        ),
        encoding="utf-8",
    )
    return path
