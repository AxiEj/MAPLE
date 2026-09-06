from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator.calculator_base import HARTREE2EV
from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.dispatcher.ts.algorithm.neb import NEB
from maple.function.utility import Molecules


class _DoubleWellCalculator(Calculator):
    """Conservative H2 bond double well; public ASE units are eV and eV/A."""

    implemented_properties = ("energy", "forces")

    def __init__(
        self,
        *,
        solvent: str = "water",
        pes_identity=None,
        reject_nebts: bool = False,
        require_stationary_endpoints: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = "manufactured-double-well"
        self.solvent = solvent
        self._pes_identity = pes_identity or (self.model_name, solvent, "v1")
        self._reject_nebts = reject_nebts
        self.MAPLE_NEB_REQUIRE_STATIONARY_ENDPOINTS = require_stationary_endpoints
        self.requested_properties: list[tuple[str, ...]] = []

    def maple_neb_pes_identity(self, _atoms):
        return self._pes_identity

    def validate_maple_job(self, *, jobtype, params=None):
        if (
            self._reject_nebts
            and str(jobtype).lower() == "ts"
            and (params or {}).get("refine") == "nebts"
        ):
            raise NotImplementedError("nebts requires Hessian certification")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        requested = tuple(properties or ("energy",))
        self.requested_properties.append(requested)
        assert not ({"hessian", "hvp", "frequencies"} & set(requested))

        positions = np.asarray(atoms.get_positions(), dtype=float)
        delta = positions[1] - positions[0]
        distance = float(np.linalg.norm(delta))
        # Minima at 1 and 2 A, maximum at 1.5 A along the direct path.
        left = distance - 1.0
        right = distance - 2.0
        energy = left * left * right * right
        derivative = 2.0 * left * right * (left + right)
        unit = delta / distance
        forces = np.zeros((2, 3), dtype=float)
        forces[0] = derivative * unit
        forces[1] = -derivative * unit
        self.results = {"energy": energy, "forces": forces}

    def get_hessian(self, *_args, **_kwargs):  # pragma: no cover - must stay unused
        raise AssertionError("E/F-only NEB requested a Hessian")

    def get_hvp(self, *_args, **_kwargs):  # pragma: no cover - must stay unused
        raise AssertionError("E/F-only NEB requested an HVP")


class _TriatomicDoubleWellCalculator(Calculator):
    """Double-well reaction coordinate plus a perpendicular harmonic mode."""

    implemented_properties = ("energy", "forces")

    def __init__(self) -> None:
        super().__init__()
        self.model_name = "manufactured-triatomic-double-well"
        self.requested_properties: list[tuple[str, ...]] = []

    def maple_neb_pes_identity(self, _atoms):
        return self.model_name, "v1"

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        requested = tuple(properties or ("energy",))
        self.requested_properties.append(requested)
        assert not ({"hessian", "hvp", "frequencies"} & set(requested))

        positions = np.asarray(atoms.get_positions(), dtype=float)
        delta = positions[1] - positions[0]
        distance = float(np.linalg.norm(delta))
        unit = delta / distance
        left = distance - 1.0
        right = distance - 2.0
        double_well = left * left * right * right
        derivative = 2.0 * left * right * (left + right)

        # This Cartesian restraint is deliberately perpendicular to the
        # endpoint interpolation tangent for the manufactured aligned path.
        displacement = positions[2, 0] - 0.5 * (positions[0, 0] + positions[1, 0])
        spring = 10.0
        energy = double_well + 0.5 * spring * displacement * displacement
        forces = np.zeros((3, 3), dtype=float)
        forces[0] += derivative * unit
        forces[1] -= derivative * unit
        forces[2, 0] -= spring * displacement
        forces[0, 0] += 0.5 * spring * displacement
        forces[1, 0] += 0.5 * spring * displacement
        self.results = {"energy": energy, "forces": forces}

    def get_hessian(self, *_args, **_kwargs):  # pragma: no cover - must stay unused
        raise AssertionError("E/F-only NEB requested a Hessian")

    def get_hvp(self, *_args, **_kwargs):  # pragma: no cover - must stay unused
        raise AssertionError("E/F-only NEB requested an HVP")


class _ForceConsistentDoubleWellCalculator(_DoubleWellCalculator):
    implemented_properties = ("energy", "free_energy", "forces")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        force_consistent_energy = self.results["energy"]
        self.results["energy"] = force_consistent_energy + 10.0
        self.results["free_energy"] = force_consistent_energy


class _ConstantComponentForceCalculator(Calculator):
    implemented_properties = ("energy", "forces")
    MAPLE_NEB_REQUIRE_STATIONARY_ENDPOINTS = True

    def maple_neb_pes_identity(self, _atoms):
        return "constant-component-force", "v1"

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.full((len(atoms), 3), 4.0e-4 * HARTREE2EV),
        }


class _Command:
    def __init__(self, params: dict) -> None:
        self.params = params


def _endpoint(
    distance: float,
    *,
    solvent: str = "water",
    require_stationary_endpoints: bool = False,
) -> tuple[Atoms, _DoubleWellCalculator]:
    atoms = Atoms(
        "H2",
        positions=[[-0.5 * distance, 0.0, 0.0], [0.5 * distance, 0.0, 0.0]],
    )
    calculator = _DoubleWellCalculator(
        solvent=solvent,
        require_stationary_endpoints=require_stationary_endpoints,
    )
    atoms.calc = calculator
    return atoms, calculator


def _triatomic_image(distance: float, *, perpendicular_displacement: float = 0.0):
    atoms = Atoms(
        "H3",
        positions=[
            [-0.5 * distance, 0.0, 0.0],
            [0.5 * distance, 0.0, 0.0],
            [perpendicular_displacement, 1.0, 0.0],
        ],
    )
    calculator = _TriatomicDoubleWellCalculator()
    atoms.calc = calculator
    return atoms, calculator


def _run(tmp_path, *, refine=None, max_iter=5, n_images=3):
    reactant, reactant_calc = _endpoint(1.0)
    product, product_calc = _endpoint(2.0)
    output = tmp_path / "neb.out"
    params = {
        "method": "neb",
        "n_images": n_images,
        "ifidpp": 0,
        "use_dynamic_k": False,
        "max_iter": max_iter,
        "cineb_f_max_th": 1.0e-6,
        "cineb_f_rms_th": 1.0e-6,
    }
    if refine is not None:
        params["refine"] = refine
    Dispatcher()(_Command(params), "ts", [reactant, product], str(output))
    return output, reactant_calc, product_calc


def _run_direct(tmp_path, *, refine=None):
    reactant, reactant_calc = _endpoint(1.0)
    product, product_calc = _endpoint(2.0)
    output = tmp_path / "direct-neb.out"
    params = {
        "method": "neb",
        "n_images": 3,
        "ifidpp": 0,
        "use_dynamic_k": False,
        "max_iter": 5,
        "cineb_f_max_th": 1.0e-6,
        "cineb_f_rms_th": 1.0e-6,
    }
    if refine is not None:
        params["refine"] = refine
    NEB(str(output), Molecules([reactant, product]), params).run()
    assert reactant.calc is reactant_calc
    assert product.calc is product_calc
    return output, reactant_calc, product_calc


def test_plain_neb_runs_real_ef_driver_without_automatic_refinement(tmp_path):
    output, reactant_calc, product_calc = _run(tmp_path)

    text = output.read_text(encoding="utf-8")
    assert "NEB already converged at initial geometry (iteration 0)." in text
    assert "HIGHEST ENERGY IMAGE (TS CANDIDATE; NOT A CERTIFIED SADDLE)" in text
    assert "Starting CINEB refinement" not in text
    assert "SADDLE POINT" not in text
    assert not (tmp_path / "neb_cineb_mep.xyz").exists()
    # 0.0625 eV at r=1.5 A must cross the legacy job boundary before logging.
    assert "Energy                                    ....   0.00229683 Eh" in text
    requested = reactant_calc.requested_properties + product_calc.requested_properties
    assert requested
    assert all(set(item) <= {"energy", "forces"} for item in requested)


def test_direct_neb_and_dispatcher_share_the_same_hartree_job_boundary(tmp_path):
    dispatched, _, _ = _run(tmp_path)
    direct, _, _ = _run_direct(tmp_path)

    expected = "Energy                                    ....   0.00229683 Eh"
    assert expected in dispatched.read_text(encoding="utf-8")
    direct_text = direct.read_text(encoding="utf-8")
    assert expected in direct_text
    assert "Energy                                    ....   0.06250000 Eh" not in direct_text


def test_neb_prefers_declared_force_consistent_energy(tmp_path):
    reactant, _ = _endpoint(1.0)
    product, _ = _endpoint(2.0)
    reactant.calc = _ForceConsistentDoubleWellCalculator()
    product.calc = _ForceConsistentDoubleWellCalculator()
    output = tmp_path / "force-consistent-neb.out"

    NEB(
        str(output),
        Molecules([reactant, product]),
        {
            "method": "neb",
            "n_images": 3,
            "ifidpp": 0,
            "use_dynamic_k": False,
            "max_iter": 5,
        },
    ).run()

    text = output.read_text(encoding="utf-8")
    assert "Energy                                    ....   0.00229683 Eh" in text
    assert "Energy                                    ....   0.36979015 Eh" not in text


def test_direct_cineb_is_ef_only_and_does_not_automatically_run_prfo(tmp_path):
    output, reactant_calc, product_calc = _run_direct(tmp_path, refine="cineb")

    text = output.read_text(encoding="utf-8")
    assert "Starting CINEB refinement" in text
    assert "CINEB converged after 0 iterations." in text
    assert "REFINED TS STRUCTURE" not in text
    assert not (tmp_path / "direct-neb_nebts_ts.xyz").exists()
    requested = reactant_calc.requested_properties + product_calc.requested_properties
    assert requested
    assert all(set(item) <= {"energy", "forces"} for item in requested)


def test_cineb_refinement_is_still_ef_only_and_reports_candidate_status(tmp_path):
    output, reactant_calc, product_calc = _run(tmp_path, refine="cineb")

    text = output.read_text(encoding="utf-8")
    assert "Starting CINEB refinement" in text
    assert "CINEB converged after 0 iterations." in text
    assert "CLIMBING IMAGE (TS CANDIDATE; NOT A CERTIFIED SADDLE)" in text
    assert "CERTIFIED SADDLE POINT" not in text
    assert (tmp_path / "neb_cineb_mep.xyz").exists()
    requested = reactant_calc.requested_properties + product_calc.requested_properties
    assert requested
    assert all(set(item) <= {"energy", "forces"} for item in requested)


def test_unconverged_neb_path_is_not_reported_as_a_ts_candidate(tmp_path):
    reactant, _ = _endpoint(1.0)
    middle, _ = _endpoint(1.6)
    product, _ = _endpoint(2.0)
    output = tmp_path / "unconverged-neb.out"

    Dispatcher()(
        _Command(
            {
                "method": "neb",
                "n_images": 1,
                "ifidpp": 0,
                "use_dynamic_k": False,
                "max_iter": 0,
            }
        ),
        "ts",
        [reactant, middle, product],
        str(output),
    )

    text = output.read_text(encoding="utf-8")
    assert "NEB optimization reached maximum iterations." in text
    assert "UNCONVERGED PATH IMAGE; SADDLE STATUS UNASSESSED" in text
    assert "(TS CANDIDATE; NOT A CERTIFIED SADDLE)" not in text


def test_flagged_neb_requires_stationary_endpoints(tmp_path):
    reactant, _ = _endpoint(1.2, require_stationary_endpoints=True)
    product, _ = _endpoint(2.0, require_stationary_endpoints=True)

    with pytest.raises(ValueError, match="requires stationary endpoints"):
        Dispatcher()(
            _Command(
                {
                    "method": "neb",
                    "n_images": 1,
                    "ifidpp": 0,
                    "max_iter": 0,
                    # The calculator's stationarity contract cannot be loosened.
                    "endpoint_f_max_th": 1.0,
                    "endpoint_f_rms_th": 1.0,
                }
            ),
            "ts",
            [reactant, product],
            str(tmp_path / "high-endpoint-force.out"),
        )


def test_flagged_neb_accepts_stationary_ef_endpoints(tmp_path):
    reactant, _ = _endpoint(1.0, require_stationary_endpoints=True)
    product, _ = _endpoint(2.0, require_stationary_endpoints=True)
    output = tmp_path / "stationary-endpoints.out"

    Dispatcher()(
        _Command({"method": "neb", "n_images": 1, "ifidpp": 0, "max_iter": 0}),
        "ts",
        [reactant, product],
        str(output),
    )

    assert "NEB already converged at initial geometry" in output.read_text(
        encoding="utf-8"
    )


def test_endpoint_gate_uses_optimizer_cartesian_component_metrics(tmp_path):
    reactant, _ = _endpoint(1.0)
    product, _ = _endpoint(2.0)
    reactant.calc = _ConstantComponentForceCalculator()
    product.calc = _ConstantComponentForceCalculator()
    output = tmp_path / "component-metrics.out"

    Dispatcher()(
        _Command({"method": "neb", "n_images": 1, "ifidpp": 0, "max_iter": 0}),
        "ts",
        [reactant, product],
        str(output),
    )

    text = output.read_text(encoding="utf-8")
    assert "RMS(F)          ....    0.000400 Eh/Angstrom" in text
    assert "MAX(|F|)        ....    0.000400 Eh/Angstrom" in text


def test_failed_flagged_initial_endpoint_optimization_does_not_continue(tmp_path):
    reactant, _ = _endpoint(1.2, require_stationary_endpoints=True)
    product, _ = _endpoint(2.0, require_stationary_endpoints=True)
    output = tmp_path / "failed-endpoint-opt.out"

    with pytest.raises(ValueError, match="requires stationary endpoints"):
        Dispatcher()(
            _Command(
                {
                    "method": "neb",
                    "n_images": 1,
                    "ifidpp": 0,
                    "initial_opt": True,
                    "max_iter": 0,
                }
            ),
            "ts",
            [reactant, product],
            str(output),
        )

    text = output.read_text(encoding="utf-8")
    assert "Initial endpoint optimization did not satisfy" in text
    assert "Initial endpoint optimization converged" not in text
    assert "Starting NEB iterations" not in text
    assert not (tmp_path / "failed-endpoint-opt_reactant_min.xyz").exists()
    assert not (tmp_path / "failed-endpoint-opt_product_min.xyz").exists()
    assert (tmp_path / "failed-endpoint-opt_reactant_endpoint_unconverged.xyz").exists()
    assert (tmp_path / "failed-endpoint-opt_product_endpoint_unconverged.xyz").exists()
    assert "Unconverged reactant endpoint written" in text
    assert "Optimized reactant written" not in text


def test_neb_moves_a_displaced_internal_image_and_converges_with_ef_only(tmp_path):
    reactant, reactant_calc = _triatomic_image(1.0)
    middle, middle_calc = _triatomic_image(1.5, perpendicular_displacement=0.3)
    product, product_calc = _triatomic_image(2.0)
    output = tmp_path / "moving-neb.out"

    Dispatcher()(
        _Command(
            {
                "method": "neb",
                "n_images": 1,
                "ifidpp": 0,
                "use_dynamic_k": False,
                "max_iter": 200,
            }
        ),
        "ts",
        [reactant, middle, product],
        str(output),
    )

    text = output.read_text(encoding="utf-8")
    assert "NEB already converged at initial geometry" not in text
    assert "NEB optimization converged after" in text
    iteration_lines = [line for line in text.splitlines() if "LBFGS" in line]
    assert iteration_lines
    fields = iteration_lines[-1].split()
    assert int(fields[1]) > 0
    assert float(fields[4]) < 9.5e-3
    assert float(fields[5]) < 5.0e-3
    requested = (
        reactant_calc.requested_properties
        + middle_calc.requested_properties
        + product_calc.requested_properties
    )
    assert requested
    assert all(set(item) <= {"energy", "forces"} for item in requested)


def test_unconverged_cineb_image_is_not_reported_as_a_ts_candidate(tmp_path):
    output, _, _ = _run(tmp_path, refine="cineb", max_iter=0, n_images=2)

    text = output.read_text(encoding="utf-8")
    assert "NEB already converged at initial geometry (iteration 0)." in text
    assert "CINEB refinement reached maximum iterations." in text
    assert "UNCONVERGED CLIMBING IMAGE; SADDLE STATUS UNASSESSED" in text
    # The converged plain path may be a candidate; the climbing image is not.
    assert "CLIMBING IMAGE (TS CANDIDATE" not in text


def test_neb_rejects_endpoint_atom_order_mismatch_before_calculation(tmp_path):
    reactant = Atoms("HCl", positions=[[0, 0, 0], [1, 0, 0]])
    product = Atoms("ClH", positions=[[0, 0, 0], [1, 0, 0]])
    reactant.calc = _DoubleWellCalculator()
    product.calc = _DoubleWellCalculator()

    with pytest.raises(ValueError, match="identical atomic numbers and atom order"):
        Dispatcher()(
            _Command({"method": "neb", "n_images": 1, "ifidpp": 0}),
            "ts",
            [reactant, product],
            str(tmp_path / "mismatch.out"),
        )
    assert reactant.calc.requested_properties == []
    assert product.calc.requested_properties == []


def test_neb_requires_both_endpoint_calculators(tmp_path):
    reactant, _ = _endpoint(1.0)
    product = Atoms("H2", positions=[[-1.0, 0, 0], [1.0, 0, 0]])

    with pytest.raises(ValueError, match="reactant and product endpoints require calculators"):
        Dispatcher()(
            _Command({"method": "neb", "n_images": 1, "ifidpp": 0}),
            "ts",
            [reactant, product],
            str(tmp_path / "missing-calc.out"),
        )


def test_neb_rejects_explicit_image_without_calculator_without_leaking_view(tmp_path):
    reactant, reactant_calc = _endpoint(1.0)
    middle = Atoms("H2", positions=[[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]])
    product, product_calc = _endpoint(2.0)

    with pytest.raises(ValueError, match="Every explicitly supplied NEB image"):
        Dispatcher()(
            _Command({"method": "neb", "n_images": 1, "ifidpp": 0}),
            "ts",
            [reactant, middle, product],
            str(tmp_path / "missing-middle-calc.out"),
        )

    assert reactant.calc is reactant_calc
    assert middle.calc is None
    assert product.calc is product_calc


def test_neb_rejects_inconsistent_calculator_profiles(tmp_path):
    reactant, _ = _endpoint(1.0, solvent="water")
    product, _ = _endpoint(2.0, solvent="ethanol")

    with pytest.raises(ValueError, match="same calculator model/profile"):
        Dispatcher()(
            _Command({"method": "neb", "n_images": 1, "ifidpp": 0}),
            "ts",
            [reactant, product],
            str(tmp_path / "profile.out"),
        )


def test_neb_rejects_same_named_calculators_with_different_explicit_pes_identity(
    tmp_path,
):
    reactant, _ = _endpoint(1.0)
    product, _ = _endpoint(2.0)
    reactant.calc = _DoubleWellCalculator(pes_identity=("checkpoint-sha256", "aaa"))
    product.calc = _DoubleWellCalculator(pes_identity=("checkpoint-sha256", "bbb"))

    with pytest.raises(ValueError, match="same calculator model/profile"):
        Dispatcher()(
            _Command({"method": "neb", "n_images": 1, "ifidpp": 0}),
            "ts",
            [reactant, product],
            str(tmp_path / "pes-identity.out"),
        )


def test_direct_neb_cannot_bypass_calculator_workflow_validator(tmp_path):
    reactant, _ = _endpoint(1.0)
    product, _ = _endpoint(2.0)
    reactant.calc = _DoubleWellCalculator(reject_nebts=True)
    product.calc = _DoubleWellCalculator(reject_nebts=True)
    reactant_calc = reactant.calc
    product_calc = product.calc
    job = NEB(
        str(tmp_path / "direct.out"),
        Molecules([reactant, product]),
        {"method": "neb", "refine": "nebts", "n_images": 1, "ifidpp": 0},
    )

    with pytest.raises(NotImplementedError, match="requires Hessian certification"):
        job.run()
    assert reactant.calc is reactant_calc
    assert product.calc is product_calc
    assert reactant.calc.requested_properties == []
    assert product.calc.requested_properties == []
