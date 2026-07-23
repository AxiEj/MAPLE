"""One composition boundary for MAPLE gas energies plus solvent corrections."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .charges import (
    QEQ_EXPERIMENTAL_PROVENANCE,
    ChargeResult,
    QEqGTO,
    prepare_charges,
)
from .openmm_gb import EV_PER_HARTREE, OpenMMGB
from .result import SolvationResult


class ImplicitSolvationCorrection:
    """Prepare charge/provider state once and evaluate additive solvent terms."""

    def __init__(
        self,
        atoms,
        charge_options: dict[str, Any],
        solvation_options: dict[str, Any],
        *,
        output: str | os.PathLike[str] | None = None,
    ):
        self.atoms = atoms
        self.charge_options = dict(charge_options)
        self.solvation_options = dict(solvation_options)
        if self.solvation_options.get("experimental") is not True:
            raise ValueError(
                "Implicit-solvation providers have not passed MAPLE's public scientific benchmark gate; "
                "set experimental=true explicitly."
            )
        method = str(self.solvation_options["method"]).lower()
        if (
            method == "pb"
            and str(self.solvation_options.get("provider", "apbs")).lower()
            == "amber-pbsa"
        ):
            raise NotImplementedError(
                "The amber-pbsa/abcg2-pbsa-2023 profile remains evidence-gated: the paper's "
                "optimized atom-type radii and nonpolar parameter artifact must be supplied and "
                "reviewed before MAPLE can execute it. Use "
                "provider=apbs,profile=generic-mbondi2,experimental=true now."
            )
        output_path = Path(output).resolve() if output else Path.cwd() / "maple.out"
        self.audit_dir = output_path.with_suffix(output_path.suffix + ".implicit")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.charge_result: ChargeResult = prepare_charges(
            atoms, self.charge_options, self.audit_dir
        )
        if (
            self.charge_result.provider_positions is not None
            and self.charge_options.get("geometry", "keep") == "provider"
        ):
            atoms.set_positions(self.charge_result.provider_positions)
        atoms.set_initial_charges(self.charge_result.charges)

        self.method = method
        self.mode = str(self.charge_options.get("mode", "fixed")).lower()
        if self.mode not in {"fixed", "polarizable"}:
            raise ValueError("Charge mode must be 'fixed' or 'polarizable'.")
        if method == "gb":
            self.provider = OpenMMGB(
                atoms,
                self.charge_result.charges,
                model=self.solvation_options.get("model", "obc2"),
                nonpolar=self.solvation_options.get("nonpolar", "ace"),
                platform=self.solvation_options.get("platform", "Reference"),
            )
        elif method == "pb":
            from .apbs_pb import APBSLPB

            self.provider = APBSLPB(
                atoms,
                self.charge_result.charges,
                executable=self.solvation_options.get("executable", "apbs"),
                grid_spacing=self.solvation_options.get("grid_spacing", 0.33),
                grid_points=self.solvation_options.get("grid_points", 97),
                probe_radius=self.solvation_options.get("probe_radius", 1.4),
                surface_tension=self.solvation_options.get("surface_tension", 0.105),
                pressure=self.solvation_options.get("pressure", 0.0),
                timeout=self.solvation_options.get("timeout", 3600.0),
                audit_dir=self.audit_dir,
            )
        else:
            raise ValueError(f"Unsupported implicit-solvation method: {method!r}.")

        if self.mode == "polarizable" and method != "gb":
            raise ValueError("QEq-GTO polarizable mode is currently supported only with GB.")
        if self.mode == "polarizable" and self.charge_result.method != "qeq-gto":
            raise ValueError("mode=polarizable requires #charge(source=maple,method=qeq-gto).")
        self.supported_properties = set(self.provider.supported_properties)
        self._write_audit_manifest()

    def _write_audit_manifest(self) -> None:
        manifest = {
            "schema_version": 1,
            "charge": self.charge_result.provenance,
            "solvation": getattr(self.provider, "provenance", None)
            if not callable(getattr(self.provider, "provenance", None))
            else self.provider.provenance(),
            "charge_options": self.charge_options,
            "solvation_options": self.solvation_options,
            "energy_composition": "E_MAPLE_gas + delta_G_solv",
            "fixed_charge_lifecycle": (
                "reference-geometry-once"
                if self.mode == "fixed"
                else "variational-cqeq-gto-gb-each-geometry"
            ),
        }
        (self.audit_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )

    def evaluate(
        self,
        atoms,
        *,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        if self.mode != "polarizable":
            return self.provider.evaluate(atoms, need_forces=need_forces, calculator=calculator)
        return self._evaluate_polarizable_cqeq_gb(atoms, need_forces=need_forces)

    def _evaluate_polarizable_cqeq_gb(
        self, atoms, *, need_forces: bool
    ) -> SolvationResult:
        qeq = QEqGTO()
        total_charge = float(atoms.info.get("charge", 0.0))
        q_vac = qeq.solve_variational(atoms, total_charge=total_charge)
        vacuum_diagnostics = {
            "iterations": qeq.last_variational_iterations,
            "kkt_residual_ev": qeq.last_variational_kkt_residual,
            "min_projected_hessian_eigenvalue_ev": qeq.last_variational_min_eigenvalue,
        }
        gb_hessian = self.provider.polar_charge_hessian_ev(atoms)
        q_solv = qeq.solve_variational(
            atoms,
            total_charge=total_charge,
            extra_hessian=gb_hessian,
            initial_charges=q_vac,
        )
        solvent_diagnostics = {
            "iterations": qeq.last_variational_iterations,
            "kkt_residual_ev": qeq.last_variational_kkt_residual,
            "min_projected_hessian_eigenvalue_ev": qeq.last_variational_min_eigenvalue,
        }
        gb_result = self.provider.evaluate(atoms, need_forces=need_forces, charges=q_solv)
        qeq_vac_ev = qeq.energy_ev(atoms, q_vac)
        qeq_solv_ev = qeq.energy_ev(atoms, q_solv)
        polarization_cost_ha = (qeq_solv_ev - qeq_vac_ev) / EV_PER_HARTREE
        forces = gb_result.forces_hartree_per_angstrom
        if need_forces:
            _, qeq_vac_force = qeq.energy_and_forces_ev_angstrom(atoms, q_vac)
            _, qeq_solv_force = qeq.energy_and_forces_ev_angstrom(atoms, q_solv)
            forces = np.asarray(forces) + (qeq_solv_force - qeq_vac_force) / EV_PER_HARTREE
        components = dict(gb_result.components_hartree)
        components["qeq_polarization"] = polarization_cost_ha
        components["polar"] = components.get("polar", 0.0) + polarization_cost_ha
        return SolvationResult(
            energy_hartree=gb_result.energy_hartree + polarization_cost_ha,
            forces_hartree_per_angstrom=forces,
            components_hartree=components,
            provenance={
                **gb_result.provenance,
                "charge_mode": "polarizable",
                "charge_method": "cqeq-gto",
                **QEQ_EXPERIMENTAL_PROVENANCE,
                "variational_model": "consistent-qeq-gto-plus-gb",
                "variational_force": "envelope-theorem",
                "citation": (
                    "Ogawa et al., Consistent Charge Equilibration Method Combined "
                    "with Universal Force Field, DOI:10.1273/cbij.3.78"
                ),
                "q_vac": q_vac.tolist(),
                "q_solv": q_solv.tolist(),
                "vacuum_solver": vacuum_diagnostics,
                "solvent_solver": solvent_diagnostics,
            },
        )
