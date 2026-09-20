"""One composition boundary for MAPLE gas energies plus solvent corrections."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from maple.function.read.filereader.mol2_reader import (
    MOL2_ATOM_ID_ARRAY,
    MOL2_IDENTITY_SHA256_KEY,
    mol2_identity_sha256,
)

from .charges import ChargeResult, prepare_charges
from .openmm_gb import DEFAULT_OPENMM_PLATFORM, OpenMMGB
from .result import SolvationResult

_ATOM_IDENTITY_ARRAY = "_maple_implicit_atom_identity"

_NUMERICAL_FORCE_KEYS = {
    "force_step_angstrom",
    "force_check_step_angstrom",
    "curvature_step_angstrom",
    "max_scalar_evaluations",
    "max_raw_records",
    "max_audit_bytes",
}


def _positive_finite(value: Any, *, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _task_contract(task_context: Any) -> tuple[str, str, Any]:
    if task_context is None:
        return "sp", "", None
    if isinstance(task_context, str):
        value = task_context.lower()
        if value in {"prfo", "dimer"}:
            return "ts", value, None
        return value, "", None
    if not isinstance(task_context, dict):
        raise TypeError("task_context must be a task name or mapping")
    task = str(task_context.get("task", "sp")).lower()
    method = str(task_context.get("method", "")).lower()
    if task in {"prfo", "dimer"}:
        task, method = "ts", task
    return task, method, task_context.get("delta")


def _validate_direct_contract(
    charge_options: dict[str, Any],
    solvation_options: dict[str, Any],
    task_context: Any,
) -> tuple[str, str, str | None, str, str, Any]:
    """Pure compatibility validation run before charge/tool/audit side effects."""
    if str(charge_options.get("mode", "fixed")).lower() != "fixed":
        raise ValueError("Charge mode must be 'fixed'.")
    if solvation_options.get("experimental") is not True:
        raise ValueError(
            "Implicit-solvation providers have not passed MAPLE's public scientific benchmark gate; "
            "set experimental=true explicitly."
        )
    method = str(solvation_options.get("method", "")).lower()
    if method not in {"gb", "pb"}:
        raise ValueError(f"Unsupported implicit-solvation method: {method!r}.")
    provider = str(
        solvation_options.get("provider", "openmm" if method == "gb" else "apbs")
    ).lower()
    mode_value = solvation_options.get("mode")
    force_mode = None if mode_value is None else str(mode_value).lower()
    if force_mode not in {None, "native", "numerical"}:
        raise ValueError("Solvation mode must be native or numerical.")
    supplied = _NUMERICAL_FORCE_KEYS.intersection(solvation_options)
    native_provider = (method, provider) in {("gb", "openmm"), ("pb", "ddx")}
    numerical_provider = (method, provider) in {
        ("gb", "ambertools"),
        ("pb", "apbs"),
    }
    if native_provider and (force_mode == "numerical" or supplied):
        raise ValueError(
            f"provider={provider} supplies native forces and rejects numerical-force options"
        )
    if native_provider and force_mode not in {None, "native"}:
        raise ValueError(f"provider={provider} supports only native forces")
    if numerical_provider and force_mode == "native":
        raise ValueError(
            f"provider={provider} has no admitted native force"
        )
    if numerical_provider and (force_mode == "numerical" or supplied):
        raise ValueError(
            f"provider={provider} numerical forces are an internal diagnostic, "
            "not a runtime mode; use an admitted native-force provider for "
            "derivative workflows"
        )
    if not native_provider and not numerical_provider and (force_mode is not None or supplied):
        raise ValueError(f"provider={provider} does not support numerical-force options")

    task, task_method, task_delta = _task_contract(task_context)
    return method, provider, force_mode, task, task_method, task_delta


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
        model_device=None,
        task_context: Any = None,
    ):
        charge_options = dict(charge_options)
        solvation_options = dict(solvation_options)
        (
            method,
            provider_name,
            force_mode,
            self.task,
            self.task_method,
            self.task_delta,
        ) = _validate_direct_contract(
            charge_options, solvation_options, task_context
        )
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
        self.charge_options = charge_options
        self.solvation_options = solvation_options
        if force_mode is not None:
            self.solvation_options["mode"] = force_mode
        self.force_mode = force_mode
        self.model_device = str(model_device) if model_device is not None else None
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
        execution_keys = {"platform", "precision", "device_index", "opencl_platform_index"}
        if (method, provider_name) != ("gb", "openmm") and execution_keys.intersection(self.solvation_options):
            raise ValueError("OpenMM execution options require method=gb, provider=openmm; CPU native/external solvers do not use them.")
        nonpolar_name = str(
            self.solvation_options.get(
                "nonpolar",
                "ace" if method == "gb" else ("none" if provider_name == "ddx" else provider_name),
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
        kappa_key = "solvent_kappa_inverse_angstrom"
        if kappa_key in self.solvation_options and (method, provider_name) != ("pb", "ddx"):
            raise ValueError(f"{kappa_key} is only valid with implicit PB provider=ddx.")
        if method == "pb" and provider_name not in {"apbs", "amber-pbsa", "ddx"}:
            raise ValueError(f"Unsupported implicit-PB provider: {provider_name!r}.")
        if method == "pb" and provider_name == "ddx":
            allowed = {"method", "implicit", "provider", "model", "profile",
                       "nonpolar", "experimental", "mode", kappa_key}
            conflicts = sorted(set(self.solvation_options) - allowed)
            if conflicts:
                raise ValueError("ddX reference does not support options: " + ", ".join(conflicts))
            if kappa_key not in self.solvation_options:
                raise ValueError(f"ddX reference requires explicit {kappa_key}.")
            if (
                str(self.solvation_options.get("implicit", "water")).lower() != "water"
                or str(self.solvation_options.get("model", "lpb")).lower() != "lpb"
                or str(self.solvation_options.get("profile", "ddlpb-union-mbondi2-v1")).lower()
                != "ddlpb-union-mbondi2-v1"
                or nonpolar_name != "none"
            ):
                raise ValueError(
                    "ddX reference requires model=lpb, profile=ddlpb-union-mbondi2-v1, "
                    "and nonpolar=none."
                )
            self.solvation_options.update(
                provider="ddx", model="lpb", profile="ddlpb-union-mbondi2-v1", nonpolar="none"
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
                    model_device=model_device,
                    precision=self.solvation_options.get("precision"),
                    device_index=self.solvation_options.get("device_index"),
                    opencl_platform_index=self.solvation_options.get("opencl_platform_index"),
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
            if provider_name == "ddx":
                from .ddx_lpb import DDXLPB

                self.provider = DDXLPB(
                    atoms,
                    self.charge_result.charges,
                    solvent_kappa_inverse_angstrom=self.solvation_options[kappa_key],
                    audit_dir=self.audit_dir,
                )
            else:  # APBS only: unknown names and Amber's gated profile were rejected above.
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

        self.underlying_provider = self.provider
        self.supported_properties = set(self.provider.supported_properties)
        self._write_audit_manifest()

    def _write_audit_manifest(self) -> None:
        route_formula = (
            "E_solution(R) = E_MLIP,gas(R) + G_polar(R,q_fixed) + G_nonpolar(R)"
        )
        route = {
            "name": "Additive fixed-charge PB/GB implicit solvation",
            "role": "Baseline/Product Route",
            "formula": route_formula,
            "fixed_charge": True,
            "product_contract": True,
            "prohibited_terms": {
                "gas_phase_mm_energy": False,
                "retraining": False,
                "hydration_label_residual": False,
            },
        }
        if self.method == "pb" and self.solvation_options.get("provider") == "ddx":
            route.update(
                name="Fixed-charge numerical LPB reference",
                role="Reference",
                formula="E_solution(R) = E_MLIP,gas(R) + G_ddLPB,polar(R,q_fixed)",
                product_contract=False,
                nonpolar="none",
                absolute_solvation_free_energy_claim=False,
                independent_crosscheck_complete=False,
            )
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
        underlying_supported = sorted(
            set(getattr(self.underlying_provider, "supported_properties", ()))
        )
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
            "fixed_charge_lifecycle": "reference-geometry-once",
            "capabilities": {
                "wrapper_supported_properties": sorted(self.supported_properties),
                "underlying_supported_properties": underlying_supported,
                "testing_only": False,
                "native_force": "forces" in underlying_supported,
                "numerical_force": False,
                "production_admitted": False,
                "accuracy_certified": False,
            },
            "execution": {
                "model_device": self.model_device,
                "solvent_platform": getattr(self.underlying_provider, "platform", "CPU"),
                "solver_scope": "OpenMM platform" if isinstance(self.underlying_provider, OpenMMGB) else "CPU native/external solver",
            },
        }
        (self.audit_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def resolve_outer_curvature_step(
        self,
        *,
        task_delta: float | None = None,
        backend_hint: float | None = None,
        task: str | None = None,
    ) -> float:
        """Resolve and record the authoritative complete-force displacement."""
        selected_task = str(task or self.task or "").lower()
        if task_delta is not None:
            step = _positive_finite(task_delta, name="task-local delta")
            source = "task-local-delta"
        elif backend_hint is not None:
            step = _positive_finite(backend_hint, name="backend numerical derivative step")
            source = "backend-recommendation"
        else:
            step = 0.002
            source = "maple-default"
        self.last_outer_curvature_resolution = {
            "task": selected_task,
            "selected_step_angstrom": step,
            "selected_source": source,
            "task_delta_angstrom": task_delta,
            "backend_hint_angstrom": backend_hint,
        }
        return step

    @staticmethod
    def _detached_at_positions(atoms, positions: np.ndarray):
        detached = atoms.copy()
        detached.calc = None
        detached.set_constraint()
        detached.set_positions(np.asarray(positions, dtype=np.float64))
        return detached

    def validate_outer_displacements(self, atoms, displaced_pairs) -> list[dict[str, Any]]:
        """Run provider-aware outer-center preflight without scalar launches."""
        validate = getattr(self.underlying_provider, "validate_outer_displacements", None)
        if not callable(validate):
            return []
        center = self._detached_at_positions(atoms, atoms.get_positions())
        pairs = [
            (
                self._detached_at_positions(atoms, minus),
                self._detached_at_positions(atoms, plus),
            )
            for minus, plus in displaced_pairs
        ]
        records = validate(center, pairs)
        self.last_outer_displacement_preflight = records
        return records

    def preflight_cartesian_outer_displacements(
        self, atoms, step_angstrom: float
    ) -> list[dict[str, Any]]:
        step = _positive_finite(step_angstrom, name="outer curvature step")
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        flat = positions.reshape(-1)
        pairs = []
        for coordinate in range(flat.size):
            minus = flat.copy()
            plus = flat.copy()
            minus[coordinate] -= step
            plus[coordinate] += step
            pairs.append((minus.reshape(positions.shape), plus.reshape(positions.shape)))
        return self.validate_outer_displacements(atoms, pairs)

    def preflight_directional_outer_displacements(
        self, atoms, direction: np.ndarray, step_angstrom: float
    ) -> list[dict[str, Any]]:
        step = _positive_finite(step_angstrom, name="outer curvature step")
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        vector = np.asarray(direction, dtype=np.float64).reshape(positions.shape)
        if not np.isfinite(vector).all():
            raise ValueError("outer displacement direction must be finite")
        return self.validate_outer_displacements(
            atoms,
            [(positions - step * vector, positions + step * vector)],
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
        result = self.provider.evaluate(
            atoms, need_forces=need_forces, calculator=calculator
        )
        if self.inner_mode == "prebuilt":
            return SolvationResult(
                energy_hartree=result.energy_hartree,
                forces_hartree_per_angstrom=result.forces_hartree_per_angstrom,
                components_hartree=dict(result.components_hartree),
                provenance={
                    **result.provenance,
                    "composition_mode": "prebuilt-explicit-inner/implicit-outer",
                    "thermodynamic_quantity": (
                        "fixed-shell cluster-continuum configurational potential"
                    ),
                    "absolute_solvation_free_energy_claim": False,
                },
            )
        return result
