"""One composition boundary for MAPLE gas energies plus solvent corrections."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from maple.function.read.filereader.mol2_reader import (
    MOL2_ATOM_ID_ARRAY,
    MOL2_IDENTITY_SHA256_KEY,
    mol2_identity_sha256,
)

from .charges import (
    QEQ_EXPERIMENTAL_PROVENANCE,
    ChargeResult,
    QEqGTO,
    prepare_charges,
)
from .common import EV_PER_HARTREE
from .openmm_gb import DEFAULT_OPENMM_PLATFORM, OpenMMGB
from .result import SolvationResult

_ATOM_IDENTITY_ARRAY = "_maple_implicit_atom_identity"


def _mol2_topology_signature(atoms) -> str | None:
    metadata = atoms.info.get("mol2")
    if not isinstance(metadata, dict):
        return None
    topology = {
        key: metadata.get(key)
        for key in (
            "atom_ids",
            "atom_names",
            "atom_types",
            "subst_ids",
            "subst_names",
            "bonds",
            "component_ids",
            "component_count",
        )
    }
    return json.dumps(
        topology,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


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
        mol2_metadata = atoms.info.get("mol2")
        mol2_atom_identity = atoms.arrays.get(MOL2_ATOM_ID_ARRAY)
        if isinstance(mol2_metadata, dict):
            if (
                mol2_metadata.get(MOL2_IDENTITY_SHA256_KEY)
                != mol2_identity_sha256(mol2_metadata)
            ):
                raise ValueError(
                    "MOL2 atom/type/substructure/topology metadata changed "
                    "after MOL2Reader; refusing implicit-solvation setup."
                )
            mol2_atom_ids = tuple(
                int(value) for value in mol2_metadata.get("atom_ids", [])
            )
            if (
                mol2_atom_identity is None
                or len(mol2_atom_ids) != len(atoms)
                or tuple(int(value) for value in mol2_atom_identity)
                != mol2_atom_ids
            ):
                raise ValueError(
                    "Current ASE atom order no longer matches the source MOL2 "
                    "atom IDs; reload the structure before preparing implicit "
                    "solvation."
                )
        atom_identity = atoms.arrays.get(_ATOM_IDENTITY_ARRAY)
        if atom_identity is None:
            atom_identity = (
                np.asarray(mol2_atom_identity, dtype=np.int64)
                if mol2_atom_identity is not None
                else np.arange(len(atoms), dtype=np.int64)
            )
            atoms.new_array(_ATOM_IDENTITY_ARRAY, atom_identity)
        else:
            atom_identity = np.asarray(atom_identity)
            if (
                atom_identity.shape != (len(atoms),)
                or not np.issubdtype(atom_identity.dtype, np.integer)
                or len(np.unique(atom_identity)) != len(atoms)
            ):
                raise ValueError(
                    "The reserved implicit-solvation atom identity array must "
                    "contain one unique integer token per atom."
                )
        self._frozen_atom_identity = tuple(int(token) for token in atom_identity)
        self._frozen_atomic_numbers = tuple(
            int(number) for number in atoms.get_atomic_numbers()
        )
        self._frozen_mol2_topology_signature = _mol2_topology_signature(atoms)
        self.charge_options = dict(charge_options)
        self.solvation_options = dict(solvation_options)
        self.provider: Any
        inner = self.solvation_options.get("inner")
        self.inner_mode = None if inner is None else str(inner).lower()
        if self.inner_mode is not None:
            if self.inner_mode != "prebuilt":
                raise ValueError(
                    "The explicit-inner/implicit-outer runtime supports "
                    "inner=prebuilt only."
                )
            if (
                self.charge_options.get("source") != "mol2"
                or self.charge_options.get("mode", "fixed") != "fixed"
                or self.charge_options.get("geometry", "keep") != "keep"
            ):
                raise ValueError(
                    "inner=prebuilt requires fixed #charge(source=mol2) charges "
                    "with geometry=keep for the complete cluster."
                )
            metadata = atoms.info.get("mol2", {})
            if metadata.get("component_count", 1) < 2:
                raise ValueError(
                    "inner=prebuilt requires a MOL2 cluster with at least two "
                    "connected components."
                )
        if self.solvation_options.get("experimental") is not True:
            raise ValueError(
                "Implicit-solvation providers have not passed MAPLE's public scientific benchmark gate; "
                "set experimental=true explicitly."
            )
        method = str(self.solvation_options["method"]).lower()
        provider_name = str(
            self.solvation_options.get(
                "provider",
                "openmm" if method == "gb" else "apbs",
            )
        ).lower()
        nonpolar_name = str(
            self.solvation_options.get(
                "nonpolar",
                "ace" if method == "gb" else provider_name,
            )
        ).lower()
        if self.inner_mode == "prebuilt" and (
            method != "gb"
            or provider_name != "openmm"
            or nonpolar_name not in {"ace", "lcpo"}
        ):
            raise ValueError(
                "inner=prebuilt is validated only with OpenMM GB and "
                "nonpolar=ACE or LCPO."
            )
        if method == "pb" and provider_name == "amber-pbsa":
            raise NotImplementedError(
                "The amber-pbsa/abcg2-pbsa-2023 profile remains evidence-gated: the paper's "
                "optimized atom-type radii and nonpolar parameter artifact must be supplied and "
                "reviewed before MAPLE can execute it. Use "
                "provider=apbs,profile=generic-mbondi2,experimental=true now."
            )
        gb_provider = provider_name
        if method == "gb" and gb_provider == "ambertools":
            if (
                self.inner_mode is not None
                or self.charge_options.get("source") != "maple"
                or str(self.charge_options.get("method", "")).lower() != "am1bcc"
                or str(self.charge_options.get("mode", "fixed")).lower() != "fixed"
                or str(self.charge_options.get("geometry", "keep")).lower() != "keep"
                or str(self.solvation_options.get("model", "chagb")).lower() != "chagb"
                or str(
                    self.solvation_options.get("profile", "chagb-bondi-pbsa-inp2")
                ).lower()
                != "chagb-bondi-pbsa-inp2"
                or str(
                    self.solvation_options.get("nonpolar", "cavity-dispersion")
                ).lower()
                != "cavity-dispersion"
            ):
                raise ValueError(
                    "The validated AmberTools CHA-GB profile requires fixed "
                    "#charge(source=maple,method=am1bcc,geometry=keep), "
                    "model=chagb, profile=chagb-bondi-pbsa-inp2, "
                    "nonpolar=cavity-dispersion, and no inner=prebuilt."
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
            provider = gb_provider
            if provider == "openmm":
                self.provider = OpenMMGB(
                    atoms,
                    self.charge_result.charges,
                    model=self.solvation_options.get("model", "obc2"),
                    nonpolar=self.solvation_options.get("nonpolar", "ace"),
                    platform=self.solvation_options.get(
                        "platform", DEFAULT_OPENMM_PLATFORM
                    ),
                )
            elif provider == "ambertools":
                from .amber_chagb import AmberToolsChaGB

                self.provider = AmberToolsChaGB(
                    atoms,
                    self.charge_result.charges,
                    executable=self.solvation_options.get("executable", "gbnsr6"),
                    timeout=self.solvation_options.get("timeout", 3600.0),
                    audit_dir=self.audit_dir,
                )
            else:
                raise ValueError(f"Unsupported implicit-GB provider: {provider!r}.")
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
            raise ValueError(
                "QEq-GTO polarizable mode is currently supported only with GB."
            )
        if self.mode == "polarizable" and self.charge_result.method != "qeq-gto":
            raise ValueError(
                "mode=polarizable requires #charge(source=maple,method=qeq-gto)."
            )
        self.supported_properties = set(self.provider.supported_properties)
        self._write_audit_manifest()

    def _write_audit_manifest(self) -> None:
        fixed_charge = self.mode == "fixed"
        if fixed_charge:
            route_name = "Additive fixed-charge PB/GB implicit solvation"
            route_role = "Baseline/Product Route"
            route_formula = (
                "E_solution(R) = E_MLIP,gas(R) + " "G_polar(R,q_fixed) + G_nonpolar(R)"
            )
        else:
            route_name = "Variational CQEq-GTO/GB research profile"
            route_role = "Research control"
            route_formula = (
                "E_solution(R) = E_MLIP,gas(R) + " "DeltaG_variational_CQEq-GB(R)"
            )
        route = {
            "name": route_name,
            "role": route_role,
            "formula": route_formula,
            "fixed_charge": fixed_charge,
            "product_contract": fixed_charge,
            "prohibited_terms": {
                "gas_phase_mm_energy": False,
                "retraining": False,
                "hydration_label_residual": False,
            },
        }
        if self.inner_mode == "prebuilt":
            route.update(
                {
                    "composition_mode": "prebuilt-explicit-inner/implicit-outer",
                    "coordinate_scope": "entire-prebuilt-cluster",
                    "thermodynamic_quantity": (
                        "fixed-shell cluster-continuum configurational potential"
                    ),
                    "absolute_solvation_free_energy_claim": False,
                    "cluster_component_count": int(
                        self.atoms.info["mol2"]["component_count"]
                    ),
                    "terms_not_computed": [
                        "cluster_formation_or_occupancy_free_energy",
                        "standard_state_conversion",
                        "solvent_cluster_reference_free_energy",
                        "cluster_conformer_ensemble",
                    ],
                }
            )
        provider_provenance = getattr(self.provider, "provenance", None)
        manifest = {
            "schema_version": 1,
            "route": route,
            "charge": self.charge_result.provenance,
            "solvation": (
                provider_provenance()
                if callable(provider_provenance)
                else provider_provenance
            ),
            "charge_options": self.charge_options,
            "solvation_options": self.solvation_options,
            "energy_composition": route["formula"],
            "fixed_charge_lifecycle": (
                "reference-geometry-once"
                if self.mode == "fixed"
                else "variational-cqeq-gto-gb-each-geometry"
            ),
        }
        (self.audit_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def evaluate(
        self,
        atoms,
        *,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        current_atom_identity = atoms.arrays.get(_ATOM_IDENTITY_ARRAY)
        if (
            current_atom_identity is None
            or tuple(int(token) for token in current_atom_identity)
            != self._frozen_atom_identity
        ):
            raise ValueError(
                "Implicit solvation is bound to the frozen atom identity and "
                "order used to prepare charges, radii, and topology."
            )
        current_atomic_numbers = tuple(
            int(number) for number in atoms.get_atomic_numbers()
        )
        if current_atomic_numbers != self._frozen_atomic_numbers:
            raise ValueError(
                "Implicit solvation is bound to the frozen atom identity and "
                "order used to prepare charges, radii, and topology."
            )
        if (
            self._frozen_mol2_topology_signature is not None
            and _mol2_topology_signature(atoms) != self._frozen_mol2_topology_signature
        ):
            raise ValueError(
                "Implicit solvation is bound to the frozen MOL2 topology used "
                "to prepare charges, radii, and connectivity."
            )
        if self.mode != "polarizable":
            result = self.provider.evaluate(
                atoms, need_forces=need_forces, calculator=calculator
            )
            if self.inner_mode == "prebuilt":
                return SolvationResult(
                    energy_hartree=result.energy_hartree,
                    forces_hartree_per_angstrom=(result.forces_hartree_per_angstrom),
                    components_hartree=dict(result.components_hartree),
                    provenance={
                        **result.provenance,
                        "composition_mode": ("prebuilt-explicit-inner/implicit-outer"),
                        "thermodynamic_quantity": (
                            "fixed-shell cluster-continuum configurational potential"
                        ),
                        "absolute_solvation_free_energy_claim": False,
                    },
                )
            return result
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
        gb_result = self.provider.evaluate(
            atoms, need_forces=need_forces, charges=q_solv
        )
        qeq_vac_ev = qeq.energy_ev(atoms, q_vac)
        qeq_solv_ev = qeq.energy_ev(atoms, q_solv)
        polarization_cost_ha = (qeq_solv_ev - qeq_vac_ev) / EV_PER_HARTREE
        forces = gb_result.forces_hartree_per_angstrom
        if need_forces:
            _, qeq_vac_force = qeq.energy_and_forces_ev_angstrom(atoms, q_vac)
            _, qeq_solv_force = qeq.energy_and_forces_ev_angstrom(atoms, q_solv)
            forces = (
                np.asarray(forces) + (qeq_solv_force - qeq_vac_force) / EV_PER_HARTREE
            )
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
