"""Independent chemical-domain contracts for Route 2."""

from __future__ import annotations

import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.route2_domain import (
    validate_route2_domain,
)


def test_neutral_odd_electron_system_cannot_be_declared_closed_shell_singlet():
    atoms = Atoms("NO", positions=[[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]])
    atoms.info.update(charge=0, mult=1)

    with pytest.raises(ValueError, match=r"electron.*parity|odd.*electron|multiplicity"):
        validate_route2_domain(atoms)
