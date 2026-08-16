from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_RUNNER = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "route2_release"
    / "run_hybrid_smd_water_derivative_canary.py"
)
_SPEC = importlib.util.spec_from_file_location("hybrid_smd_water_canary", _RUNNER)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@dataclass(frozen=True)
class _ElectrostaticState:
    primal_residual_ev: float = 0.0


@dataclass(frozen=True)
class _State:
    root_sha256: str
    total_energy_eV: float
    solvation_energy_eV: float
    polarization_energy_eV: float
    cds_energy_eV: float
    electrostatic_state: _ElectrostaticState = _ElectrostaticState()


@dataclass(frozen=True)
class _ElectrostaticForce:
    adjoint_residual_ev: float = 0.0


@dataclass(frozen=True)
class _Force:
    total_forces_eV_per_A: np.ndarray
    evaluation_sha256: str = "f" * 64
    electrostatic_evaluation: _ElectrostaticForce = _ElectrostaticForce()


class _InvariantQuadraticPES:
    spring = 0.73

    @staticmethod
    def _centered(geometry: object) -> np.ndarray:
        positions = np.asarray(geometry.get_positions(), dtype=float)
        return positions - np.mean(positions, axis=0, keepdims=True)

    def solve(self, geometry: object) -> _State:
        centered = self._centered(geometry)
        energy = 0.5 * self.spring * float(np.vdot(centered, centered))
        return _State(
            root_sha256="r" * 64,
            total_energy_eV=energy,
            solvation_energy_eV=0.3 * energy,
            polarization_energy_eV=0.2 * energy,
            cds_energy_eV=0.1 * energy,
        )

    def evaluate_forces(
        self,
        geometry: object,
        *,
        central_state: _State | None = None,
    ) -> _Force:
        del central_state
        return _Force(-self.spring * self._centered(geometry))

    def get_potential_energy(self, geometry: object) -> float:
        return self.solve(geometry).total_energy_eV

    def configuration_sha256(self) -> str:
        return "p" * 64


def test_canary_geometry_helpers_remove_translation_and_define_so3_rotation() -> None:
    direction = _MODULE._direction(20260816, 7)
    rotation = _MODULE._rotation(20260817)

    np.testing.assert_allclose(np.sum(direction, axis=0), 0.0, atol=2.0e-16)
    assert np.linalg.norm(direction) == 1.0
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=5.0e-16)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=3.0e-16)


def test_canary_run_uses_frozen_profile_and_closes_all_scalar_checks(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    mdp = SimpleNamespace(checkpoint_sha256="m" * 64)
    radial = SimpleNamespace(
        provenance=SimpleNamespace(checkpoint_sha256="q" * 64)
    )
    hybrid = SimpleNamespace(
        configuration_sha256=lambda: "h" * 64,
        provenance_sha256="v" * 64,
    )

    monkeypatch.setattr(_MODULE, "_configure_torch", lambda: None)
    monkeypatch.setattr(
        _MODULE,
        "_git",
        lambda _repo, *args: "" if args[0] == "status" else "a" * 40,
    )
    monkeypatch.setattr(
        _MODULE,
        "build_mace_mdp_moment_adapter",
        lambda **kwargs: (captured.setdefault("mdp", kwargs), mdp)[1],
    )
    monkeypatch.setattr(
        _MODULE,
        "build_official_mace_polar_1_m_radial_gto_adapter",
        lambda **kwargs: (captured.setdefault("polar", kwargs), radial)[1],
    )
    monkeypatch.setattr(
        _MODULE,
        "MACEPolarOriginalSourceNativeFieldAdapter",
        lambda value: value,
    )
    monkeypatch.setattr(
        _MODULE,
        "build_mace_mdp_anchored_mace_polar_hybrid",
        lambda **kwargs: (captured.setdefault("hybrid", kwargs), hybrid)[1],
    )

    def _build_pes(*args, **kwargs):
        captured["pes_args"] = args
        captured["pes_kwargs"] = kwargs
        return _InvariantQuadraticPES()

    monkeypatch.setattr(
        _MODULE,
        "build_smd_mace_mdp_polar_hybrid_ddx_pes",
        _build_pes,
    )
    payload = _MODULE.run(
        argparse.Namespace(
            mdp_checkpoint=Path("/tmp/mdp.model"),
            polar_checkpoint=Path("/tmp/polar.model"),
        )
    )

    assert payload["status"] == "pass"
    assert all(payload["gates"].values())
    assert captured["mdp"] == {
        "checkpoint_path": Path("/tmp/mdp.model"),
        "device": "cpu",
    }
    assert captured["polar"]["device"] == "cpu"
    assert captured["pes_kwargs"] == {
        "solvent": "water",
        "continuum_model": "pcm",
        "lmax": 15,
        "n_lebedev": 1202,
        "solver_tolerance": 1.0e-12,
        "eta": 0.1,
        "n_proc": 1,
    }
    assert payload["configuration"]["energy_ledger"] == (
        "vacuum + ddX polarization + PySCF SMD-CDS"
    )
    assert payload["force"]["finite_difference"][-1][
        "absolute_error_eV_per_A"
    ] < 1.0e-10
