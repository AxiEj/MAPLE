from __future__ import annotations

import ctypes
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal


def _preload_conda_libstdcpp() -> None:
    """Prefer the active Conda C++ runtime before importing torch/MACE.

    Some Conda MACE stacks otherwise resolve the system libstdc++ first and
    fail later while importing compiled dependencies (for example matplotlib).
    Loading the environment copy up front is a no-op outside Linux/Conda.
    """

    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return
    candidate = Path(prefix) / "lib" / "libstdc++.so.6"
    if not candidate.is_file():
        return
    try:
        ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        # The subsequent torch/MACE import will emit the actionable loader
        # error.  Do not mask it with an optional compatibility preload.
        pass


_preload_conda_libstdcpp()

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from ..calculator_base import (
    CalcABC,
    EV2HARTREE,
    ROUTE2_SMD_CALCULATOR_PROFILE,
    hessian_via_double_autograd,
    register_calculator,
)
from ...route2_smd_profiles import (
    MACEPOL_MOLECULAR_REALSPACE_PROFILE,
    validate_route2_smd_profile,
)
from ..extra_correction.implicit.gto_field_projection import (
    EXACT_GTO_GRAPH_LONGRANGE_VERSION,
    MACEPolarGTOFieldProjectionSpec,
)
from ..extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_MODEL_FEATURE_FIELD_INDICES,
)
from ._macepol_long_range import MACEPolarLongRangeEvaluator


_MACEPOL_FOUNDATION_NAMES = {
    "macepols": "polar-1-s",
    "macepolm": "polar-1-m",
    "macepoll": "polar-1-l",
}
_ROUTE2_MACE_TORCH_VERSION = "0.3.16"
_LOCAL_REACTION_FIELD_LOCK = threading.RLock()


@dataclass(frozen=True)
class PolarState:
    """One MACE-POLAR electronic state in model-native units."""

    energy_ev: float
    density_coefficients: np.ndarray
    dipole_e_angstrom: np.ndarray
    fixed_field_forces_ev_per_angstrom: np.ndarray | None = None


class _LocalReactionFieldProjector(torch.nn.Module):
    """Inject a non-uniform local potential into MACE-POLAR's field features.

    Upstream MACE-POLAR exposes a uniform graph-level field.  The trained model
    already consumes l=0/l=1 GTO projections internally, so Route 2 supplies
    the reaction potential and its gradient at every atom through that same
    projection matrix without changing any learned weight.
    """

    def __init__(self, upstream: torch.nn.Module):
        super().__init__()
        self.upstream = upstream
        self._node_potential_gradient: torch.Tensor | None = None
        self._model_field_features: torch.Tensor | None = None

    def set_node_potential_gradient(self, values: torch.Tensor | None) -> None:
        self._node_potential_gradient = values

    @contextmanager
    def use_node_potential_gradient(self, values: torch.Tensor | None):
        """Temporarily install one field without leaking across nested calls."""

        with _LOCAL_REACTION_FIELD_LOCK:
            if values is not None and self._model_field_features is not None:
                raise RuntimeError(
                    "Preprojected MACE-POLAR field features are already installed."
                )
            previous = self._node_potential_gradient
            self._node_potential_gradient = values
            try:
                yield
            finally:
                self._node_potential_gradient = previous

    @contextmanager
    def use_model_field_features(self, values: torch.Tensor | None):
        """Temporarily install checkpoint-native GTO field features."""

        with _LOCAL_REACTION_FIELD_LOCK:
            if values is not None and self._node_potential_gradient is not None:
                raise RuntimeError(
                    "A local MACE-POLAR node field is already installed."
                )
            previous = self._model_field_features
            self._model_field_features = values
            try:
                yield
            finally:
                self._model_field_features = previous

    def forward(self, batch, positions, field):
        features = self._model_field_features
        if features is not None:
            expected_features = int(self.upstream.matrix.shape[0])
            if (
                features.ndim != 2
                or features.shape[1] != expected_features
            ):
                raise ValueError(
                    "MACE-POLAR preprojected reaction-field features must "
                    f"have shape (n_atoms, {expected_features})."
                )
            if features.shape[0] != batch.shape[0]:
                raise ValueError(
                    "MACE-POLAR preprojected reaction-field atom count does "
                    "not match the model graph."
                )
            return features.to(
                device=positions.device,
                dtype=positions.dtype,
            )

        values = self._node_potential_gradient
        if values is None:
            return self.upstream(batch, positions, field)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError(
                "MACE-POLAR local reaction field must have shape (n_atoms, 4) "
                "for [V, dV/dx, dV/dy, dV/dz]."
            )
        if values.shape[0] != batch.shape[0]:
            raise ValueError(
                "MACE-POLAR local reaction field atom count does not match the model graph."
            )
        node_fields = values.to(device=positions.device, dtype=positions.dtype)
        # Match graph_longrange.GTOInternalFieldtoFeaturesBlock and MACE's
        # Cartesian-to-e3nn convention exactly.
        node_fields = node_fields[
            :,
            list(MACE_POLAR_MODEL_FEATURE_FIELD_INDICES),
        ]
        return torch.einsum("pf,nf->np", self.upstream.matrix, node_fields)


def _integer_info(atoms, key: str, default: int) -> int:
    value = atoms.info.get(key, default)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"MACE-POLAR requires integer atoms.info['{key}']; got {value!r}."
        ) from exc
    if not numeric_value.is_integer():
        raise ValueError(
            f"MACE-POLAR requires integer atoms.info['{key}']; got {value!r}."
        )
    return int(numeric_value)


@register_calculator
class MACEPolCalculator(CalcABC):
    """ASE calculator backed by the official MACE-POLAR foundation models.

    The official MACE cache/download path is used when ``model_path`` is not
    provided.  In addition to ordinary gas-phase energy/force evaluation, the
    calculator exposes the atom-centred GTO charge density and a Route-2-only
    response hook for a non-uniform PCM reaction potential.
    """

    implemented_properties = ["energy", "forces", "free_energy", "hessian"]

    MODEL_NAMES = ("macepols", "macepolm", "macepoll")
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("analytic", "numerical")
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ()
    MODEL_PATH_OPTION = "model_path"

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        if resolved_model_path is not None:
            kwargs["model_path"] = resolved_model_path
        return kwargs

    @classmethod
    def build_implicit_solvent_kwargs(cls, solvation_options):
        provider = str(
            solvation_options.get("provider", "pcmsolver")
        ).strip().lower()
        profile = str(
            solvation_options.get("profile", "smd-iefpcm")
        ).strip().lower()
        spec = validate_route2_smd_profile(provider, profile)
        return {
            "long_range_evaluator_profile": (
                spec.mace_long_range_evaluator
            )
        }

    def __init__(
        self,
        device: torch.device | str,
        model: str = "macepolm",
        model_path: str | None = None,
        implicit: Literal["smd", "gb", "pb", "none"] = "none",
        solvent: str = "none",
        long_range_evaluator_profile: str = (
            MACEPOL_MOLECULAR_REALSPACE_PROFILE
        ),
    ):
        super().__init__()
        route2_smd = str(implicit).strip().lower() == "smd"
        self._long_range_evaluator = (
            MACEPolarLongRangeEvaluator.from_profile(
                long_range_evaluator_profile
            )
        )
        if not route2_smd and not self._long_range_evaluator.is_default:
            raise ValueError(
                "The forced reciprocal MACE-POLAR evaluator is available "
                "only through its explicit Route-2 SMD profile."
            )

        try:
            mace_version = version("mace-torch")
        except PackageNotFoundError as exc:
            raise ImportError("MACE-POLAR requires mace-torch.") from exc
        if mace_version != _ROUTE2_MACE_TORCH_VERSION:
            raise RuntimeError(
                "MAPLE Route 2 is pinned to mace-torch "
                f"{_ROUTE2_MACE_TORCH_VERSION} because it uses the release's "
                "MACE-POLAR graph_longrange density/field API; found "
                f"{mace_version}."
            )
        self.mace_torch_version = mace_version
        try:
            self.graph_longrange_version = version("graph-longrange")
        except PackageNotFoundError as exc:
            raise ImportError(
                "MACE-POLAR requires graph-longrange."
            ) from exc
        self.route2_smd_profile = (
            ROUTE2_SMD_CALCULATOR_PROFILE if route2_smd else None
        )

        try:
            from mace.calculators import mace_polar
        except Exception as exc:
            raise ImportError(
                "MACE-POLAR requires the official mace-torch runtime with "
                "graph_longrange support. Install a CUDA/CPU-compatible MACE "
                "release that provides mace.calculators.mace_polar."
            ) from exc

        if route2_smd and (model != "macepolm" or model_path is not None):
            raise ValueError(
                "Route 2 requires the unmodified official MACE-POLAR-1-M "
                "checkpoint through MACE's upstream cache."
            )
        model_source = (
            str(Path(model_path).expanduser())
            if model_path is not None
            else _MACEPOL_FOUNDATION_NAMES[model]
        )
        try:
            self._mace = mace_polar(
                model=model_source,
                device=str(device),
                # Route 2 subtracts large absolute MLIP energies to obtain a
                # small polarization response.  The upstream-recommended
                # float64 mode avoids quantizing that difference in float32.
                # Preserve the existing float32 gas-only calculator behavior.
                default_dtype="float64" if route2_smd else "float32",
                return_raw_model=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load official MACE-POLAR model {model_source!r}. "
                "The upstream cache/download failed; no MAPLE-owned weight "
                "fallback is permitted."
            ) from exc

        if len(self._mace.models) != 1:
            raise ValueError("Route 2 requires exactly one MACE-POLAR model.")
        self.model = self._mace.models[0]
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._long_range_evaluator.configure_model(self.model)

        projector = getattr(self.model, "external_field_contribution", None)
        if projector is None or not hasattr(projector, "matrix"):
            raise RuntimeError(
                "The loaded model is not a compatible MACE-POLAR checkpoint: "
                "its GTO external-field projector is missing."
            )
        self._reaction_projector = _LocalReactionFieldProjector(projector)
        self.model.external_field_contribution = self._reaction_projector

        self.device = self._mace.device
        self.dtype = next(self.model.parameters()).dtype
        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]
        self.long_range_evaluator_profile = (
            self._long_range_evaluator.profile
        )
        self.long_range_evaluator_provenance = (
            self._long_range_evaluator.provenance
        )
        self.hessian = "analytic"
        self._last_polar_state: PolarState | None = None
        self._last_polar_state_numbers: np.ndarray | None = None
        self._last_polar_state_positions: np.ndarray | None = None

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    @property
    def last_polar_state(self) -> PolarState | None:
        return self._last_polar_state

    @property
    def last_density_coefficients(self) -> np.ndarray | None:
        if self._last_polar_state is None:
            return None
        return self._last_polar_state.density_coefficients.copy()

    def route2_gto_field_projection_spec(
        self,
    ) -> MACEPolarGTOFieldProjectionSpec:
        """Return the immutable receiver basis of the loaded checkpoint."""

        graph_longrange_version = getattr(
            self,
            "graph_longrange_version",
            None,
        )
        if (
            graph_longrange_version
            != EXACT_GTO_GRAPH_LONGRANGE_VERSION
        ):
            raise RuntimeError(
                "The exact Route-2 GTO projector is pinned to "
                f"graph-longrange {EXACT_GTO_GRAPH_LONGRANGE_VERSION} "
                "because its receiver-row ordering is part of the scientific "
                f"contract; found {graph_longrange_version!r}."
            )
        descriptor = getattr(
            self.model,
            "electric_potential_descriptor",
            None,
        )
        basis = getattr(descriptor, "feature_basis", None)
        if basis is None:
            raise RuntimeError(
                "The loaded MACE-POLAR checkpoint does not expose its "
                "receiver GTO basis."
            )
        upstream = getattr(self._reaction_projector, "upstream", None)
        matrix = getattr(upstream, "matrix", None)
        if matrix is None or not torch.is_tensor(matrix):
            raise RuntimeError(
                "The loaded MACE-POLAR checkpoint does not expose its "
                "external-field projection matrix."
            )
        return MACEPolarGTOFieldProjectionSpec(
            receiver_sigmas_angstrom=tuple(basis.sigmas),
            receiver_max_l=int(basis.max_l),
            receiver_normalization=str(basis.normalize),
            upstream_matrix=np.asarray(
                matrix.detach().cpu(),
                dtype=float,
            ),
            graph_longrange_version=graph_longrange_version,
        )

    def cached_polar_state(
        self,
        atoms,
        *,
        require_forces: bool = False,
    ) -> PolarState | None:
        """Return the last public gas state only for the exact same geometry."""

        state = self._last_polar_state
        if (
            state is None
            or self._last_polar_state_numbers is None
            or self._last_polar_state_positions is None
            or not np.array_equal(
                self._last_polar_state_numbers,
                np.asarray(atoms.numbers, dtype=int),
            )
            or not np.array_equal(
                self._last_polar_state_positions,
                np.asarray(atoms.get_positions(), dtype=float),
            )
            or (
                require_forces
                and state.fixed_field_forces_ev_per_angstrom is None
            )
        ):
            return None
        return state

    @staticmethod
    def _atoms_for_mace(atoms):
        model_atoms = atoms.copy()
        multiplicity = _integer_info(atoms, "mult", 1)
        model_atoms.info["charge"] = float(atoms.info.get("charge", 0.0))
        # Upstream MACE-POLAR names this field "spin", but its public contract
        # uses 1 for a singlet, 2 for a doublet, i.e. the multiplicity.
        model_atoms.info["spin"] = multiplicity
        model_atoms.info["external_field"] = [0.0, 0.0, 0.0]
        return model_atoms

    def _batch_dict(self, atoms) -> dict[str, torch.Tensor]:
        batch = self._mace._atoms_to_batch(self._atoms_for_mace(atoms))
        model_dtype = next(self.model.parameters()).dtype
        result = batch.to_dict()
        for key, value in tuple(result.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                result[key] = value.to(dtype=model_dtype)
        return self._long_range_evaluator.prepare_batch(
            result,
            r_max=self.r_max,
        )

    def _model_forward(self, batch, **kwargs):
        return self._long_range_evaluator.forward_model(
            self.model,
            batch,
            **kwargs,
        )

    @staticmethod
    def _polar_state_from_output(output) -> PolarState:
        density = output.get("density_coefficients")
        dipole = output.get("dipole")
        if density is None or dipole is None:
            raise RuntimeError(
                "MACE-POLAR did not return density_coefficients and dipole; "
                "Route 2 cannot fall back to atom charges."
            )
        density_np = density.detach().cpu().numpy().astype(float, copy=True)
        if density_np.ndim != 2 or density_np.shape[1] != 4:
            raise RuntimeError(
                "Route 2 requires the MACE-POLAR l<=1 four-coefficient GTO "
                f"density; received shape {density_np.shape}."
            )
        dipole_np = np.asarray(dipole.detach().cpu(), dtype=float).reshape(-1, 3)[0]
        energy_ev = float(output["energy"].sum().detach().cpu())
        forces = output.get("forces")
        fixed_field_forces = None
        if forces is not None:
            fixed_field_forces = np.asarray(
                forces.detach().cpu(),
                dtype=float,
            ).copy()
            expected_shape = (density_np.shape[0], 3)
            if fixed_field_forces.shape != expected_shape or not np.all(
                np.isfinite(fixed_field_forces)
            ):
                raise RuntimeError(
                    "MACE-POLAR fixed-local-field forces must be finite with "
                    f"shape {expected_shape}; received {fixed_field_forces.shape}."
                )
        return PolarState(
            energy_ev=energy_ev,
            density_coefficients=density_np,
            dipole_e_angstrom=dipole_np,
            fixed_field_forces_ev_per_angstrom=fixed_field_forces,
        )

    def polar_output_torch(
        self,
        atoms,
        *,
        node_potential_ev: torch.Tensor | None = None,
        node_gradient_ev_per_angstrom: torch.Tensor | None = None,
        model_field_features: torch.Tensor | None = None,
        compute_forces: bool = False,
        compute_hessian: bool = False,
    ) -> dict:
        """Return raw MACE-POLAR output without detaching the local-field graph.

        This Route-2 derivative interface preserves autograd connectivity from
        the model energy and density outputs back to the supplied atom-centred
        reaction potential (eV/e) and gradient (eV/(e Å)).  It does not assume
        that either derivative is conjugate to the returned density; that
        relation must be established separately before constructing forces.
        """

        local_values = None
        feature_values = None
        if model_field_features is not None and (
            node_potential_ev is not None
            or node_gradient_ev_per_angstrom is not None
        ):
            raise ValueError(
                "Supply either local reaction potential/gradient or "
                "preprojected model-field features, not both."
            )
        if model_field_features is not None:
            if not torch.is_tensor(model_field_features):
                raise TypeError(
                    "Graph-preserving model-field features must be a torch "
                    "tensor."
                )
            if (
                model_field_features.ndim != 2
                or model_field_features.shape[0] != len(atoms)
            ):
                raise ValueError(
                    "Model-field features must have shape "
                    "(n_atoms, n_features)."
                )
            if not torch.is_floating_point(model_field_features):
                raise TypeError(
                    "Model-field features must use a floating-point torch "
                    "dtype."
                )
            feature_values = model_field_features.to(
                dtype=self.dtype,
                device=self.device,
            )
        if node_potential_ev is not None or node_gradient_ev_per_angstrom is not None:
            if node_potential_ev is None or node_gradient_ev_per_angstrom is None:
                raise ValueError(
                    "Both local reaction potential and gradient are required."
                )
            if not torch.is_tensor(node_potential_ev) or not torch.is_tensor(
                node_gradient_ev_per_angstrom
            ):
                raise TypeError(
                    "The graph-preserving local reaction potential and gradient "
                    "must be torch tensors."
                )
            if node_potential_ev.shape != (len(atoms),) or (
                node_gradient_ev_per_angstrom.shape != (len(atoms), 3)
            ):
                raise ValueError(
                    "Local reaction potential/gradient shapes must be "
                    "(n_atoms,) and (n_atoms, 3)."
                )
            if not torch.is_floating_point(
                node_potential_ev
            ) or not torch.is_floating_point(node_gradient_ev_per_angstrom):
                raise TypeError(
                    "The local reaction potential and gradient must use a "
                    "floating-point torch dtype."
                )
            potential = node_potential_ev.to(
                dtype=self.dtype,
                device=self.device,
            )
            gradient = node_gradient_ev_per_angstrom.to(
                dtype=self.dtype,
                device=self.device,
            )
            local_values = torch.cat((potential[:, None], gradient), dim=1)

        batch = self._batch_dict(atoms)
        context = (
            self._reaction_projector.use_model_field_features(feature_values)
            if feature_values is not None
            else self._reaction_projector.use_node_potential_gradient(
                local_values
            )
        )
        with context:
            return self._model_forward(
                batch,
                compute_force=compute_forces,
                compute_stress=False,
                compute_hessian=compute_hessian,
            )

    def polar_state(
        self,
        atoms,
        *,
        node_potential_ev: np.ndarray | None = None,
        node_gradient_ev_per_angstrom: np.ndarray | None = None,
        model_field_features: np.ndarray | None = None,
        compute_forces: bool = False,
        compute_hessian: bool = False,
    ) -> tuple[PolarState, dict]:
        """Evaluate a gas or locally field-polarized MACE-POLAR state.

        ``node_potential_ev`` is the electrostatic potential energy per unit
        charge in eV/e; the gradient is in eV/(e Å).  The returned model energy
        deliberately excludes the explicit ``<rho,V>`` coupling for a local
        field. Route 2 instead takes the PCMSolver polarization work directly
        as ``0.5*<rho,V>`` and uses the full coupling only as a reciprocity
        diagnostic.  When ``compute_forces`` is true, the state carries
        ``-partial E_intrinsic/partial R`` in eV/Å with the supplied atom-indexed
        potential and gradient samples held fixed.  This is a MACE-side partial
        derivative, not a total implicit-solvent force.
        """

        potential_tensor = None
        gradient_tensor = None
        feature_tensor = None
        if model_field_features is not None and (
            node_potential_ev is not None
            or node_gradient_ev_per_angstrom is not None
        ):
            raise ValueError(
                "Supply either local reaction potential/gradient or "
                "preprojected model-field features, not both."
            )
        if model_field_features is not None:
            features = np.asarray(model_field_features, dtype=float)
            if (
                features.ndim != 2
                or features.shape[0] != len(atoms)
                or not np.all(np.isfinite(features))
            ):
                raise ValueError(
                    "Model-field features must be finite with shape "
                    "(n_atoms, n_features)."
                )
            feature_tensor = torch.tensor(
                features,
                dtype=self.dtype,
                device=self.device,
            )
        if node_potential_ev is not None or node_gradient_ev_per_angstrom is not None:
            if node_potential_ev is None or node_gradient_ev_per_angstrom is None:
                raise ValueError(
                    "Both local reaction potential and gradient are required."
                )
            potential = np.asarray(node_potential_ev, dtype=float)
            gradient = np.asarray(node_gradient_ev_per_angstrom, dtype=float)
            if potential.shape != (len(atoms),) or gradient.shape != (len(atoms), 3):
                raise ValueError(
                    "Local reaction potential/gradient shapes must be "
                    "(n_atoms,) and (n_atoms, 3)."
                )
            potential_tensor = torch.tensor(
                potential,
                dtype=self.dtype,
                device=self.device,
            )
            gradient_tensor = torch.tensor(
                gradient,
                dtype=self.dtype,
                device=self.device,
            )

        output = self.polar_output_torch(
            atoms,
            node_potential_ev=potential_tensor,
            node_gradient_ev_per_angstrom=gradient_tensor,
            model_field_features=feature_tensor,
            compute_forces=compute_forces,
            compute_hessian=compute_hessian,
        )
        return self._polar_state_from_output(output), output

    def intrinsic_energy_field_gradient(
        self,
        atoms,
        *,
        node_potential_ev: np.ndarray,
        node_gradient_ev_per_angstrom: np.ndarray,
    ) -> np.ndarray:
        """Differentiate the intrinsic model energy with respect to node field.

        The result uses external Cartesian order
        ``[V, dV/dx, dV/dy, dV/dz]``.  Its columns are conjugate to the input
        units eV/e and eV/(e Angstrom), respectively.  This is an exact
        autograd derivative of the MACE-POLAR intrinsic energy; it is not
        assumed equal to the model's returned density coefficients.
        """

        potential = np.asarray(node_potential_ev, dtype=float)
        gradient = np.asarray(node_gradient_ev_per_angstrom, dtype=float)
        if potential.shape != (len(atoms),) or gradient.shape != (len(atoms), 3):
            raise ValueError(
                "Local reaction potential/gradient shapes must be "
                "(n_atoms,) and (n_atoms, 3)."
            )
        if not np.all(np.isfinite(potential)) or not np.all(np.isfinite(gradient)):
            raise ValueError("Local reaction potential/gradient must be finite.")

        potential_tensor = torch.tensor(
            potential,
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        gradient_tensor = torch.tensor(
            gradient,
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        output = self.polar_output_torch(
            atoms,
            node_potential_ev=potential_tensor,
            node_gradient_ev_per_angstrom=gradient_tensor,
        )
        energy = output.get("energy")
        if energy is None or not torch.is_tensor(energy):
            raise RuntimeError(
                "MACE-POLAR did not return a differentiable intrinsic energy."
            )
        if not bool(torch.isfinite(energy).all()):
            raise RuntimeError("MACE-POLAR intrinsic energy is non-finite.")
        if not energy.requires_grad:
            raise RuntimeError(
                "MACE-POLAR intrinsic energy is disconnected from the local "
                "reaction-field autograd graph."
            )
        potential_gradient, spatial_gradient = torch.autograd.grad(
            energy.sum(),
            (potential_tensor, gradient_tensor),
            create_graph=False,
            allow_unused=True,
        )
        if potential_gradient is None or spatial_gradient is None:
            raise RuntimeError(
                "MACE-POLAR intrinsic energy is not differentiable with "
                "respect to both local reaction-potential inputs."
            )
        result = np.concatenate(
            (
                np.asarray(
                    potential_gradient.detach().cpu(),
                    dtype=float,
                )[:, None],
                np.asarray(
                    spatial_gradient.detach().cpu(),
                    dtype=float,
                ),
            ),
            axis=1,
        )
        expected_shape = (len(atoms), 4)
        if result.shape != expected_shape or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "MACE-POLAR intrinsic-energy field gradient must be finite "
                f"with shape {expected_shape}; received {result.shape}."
            )
        return result

    def density_position_vjp(
        self,
        atoms,
        *,
        node_potential_ev: np.ndarray,
        node_gradient_ev_per_angstrom: np.ndarray,
        density_cotangent: np.ndarray,
    ) -> np.ndarray:
        """Differentiate a density pairing with respect to atom positions.

        The supplied atom-indexed node potential and gradient samples are held
        fixed.  The result is
        ``d <density_cotangent, density_coefficients> / dR`` in the
        cotangent's units per Angstrom and contains only the direct MACE
        coordinate response.  Any coordinate dependence of the PCM field must
        be contracted separately.
        """

        potential = np.asarray(node_potential_ev, dtype=float)
        gradient = np.asarray(node_gradient_ev_per_angstrom, dtype=float)
        cotangent = np.asarray(density_cotangent, dtype=float)
        expected_density_shape = (len(atoms), 4)
        if potential.shape != (len(atoms),) or gradient.shape != (len(atoms), 3):
            raise ValueError(
                "Local reaction potential/gradient shapes must be "
                "(n_atoms,) and (n_atoms, 3)."
            )
        if cotangent.shape != expected_density_shape:
            raise ValueError(
                "Density cotangent must have shape "
                f"{expected_density_shape}; received {cotangent.shape}."
            )
        if (
            not np.all(np.isfinite(potential))
            or not np.all(np.isfinite(gradient))
            or not np.all(np.isfinite(cotangent))
        ):
            raise ValueError(
                "Local reaction field and density cotangent must be finite."
            )

        potential_tensor = torch.as_tensor(
            potential,
            dtype=self.dtype,
            device=self.device,
        )
        gradient_tensor = torch.as_tensor(
            gradient,
            dtype=self.dtype,
            device=self.device,
        )
        local_values = torch.cat(
            (potential_tensor[:, None], gradient_tensor),
            dim=1,
        )
        batch = self._batch_dict(atoms)
        positions = batch.get("positions")
        if (
            not torch.is_tensor(positions)
            or positions.shape != (len(atoms), 3)
            or not torch.is_floating_point(positions)
        ):
            received = None if positions is None else tuple(positions.shape)
            raise RuntimeError(
                "MACE-POLAR coordinate VJP requires floating-point batch "
                f"positions with shape ({len(atoms)}, 3); received {received}."
            )
        positions_required_grad = positions.requires_grad
        positions.requires_grad_(True)
        try:
            with self._reaction_projector.use_node_potential_gradient(
                local_values
            ):
                output = self._model_forward(
                    batch,
                    compute_force=False,
                    compute_stress=False,
                    compute_hessian=False,
                )

            density = output.get("density_coefficients")
            if density is None or density.shape != expected_density_shape:
                received = None if density is None else tuple(density.shape)
                raise RuntimeError(
                    "MACE-POLAR density coordinate response must have shape "
                    f"{expected_density_shape}; received {received}."
                )
            if not bool(torch.isfinite(density).all()):
                raise RuntimeError(
                    "MACE-POLAR density coordinate response is non-finite."
                )
            if not density.requires_grad:
                raise RuntimeError(
                    "MACE-POLAR density is disconnected from the coordinate "
                    "autograd graph."
                )

            cotangent_tensor = torch.as_tensor(
                cotangent,
                dtype=self.dtype,
                device=self.device,
            )
            (position_gradient,) = torch.autograd.grad(
                density,
                (positions,),
                grad_outputs=cotangent_tensor,
                create_graph=False,
                allow_unused=True,
            )
            if position_gradient is None:
                raise RuntimeError(
                    "MACE-POLAR density is disconnected from the coordinate "
                    "autograd graph."
                )
            result = np.asarray(
                position_gradient.detach().cpu(),
                dtype=float,
            ).copy()
            expected_position_shape = (len(atoms), 3)
            if result.shape != expected_position_shape or not np.all(
                np.isfinite(result)
            ):
                raise RuntimeError(
                    "MACE-POLAR density position VJP must be finite with shape "
                    f"{expected_position_shape}; received {result.shape}."
                )
            return result
        finally:
            if not positions_required_grad:
                positions.requires_grad_(False)

    def linearize_density_response(
        self,
        atoms,
        *,
        node_potential_ev: np.ndarray,
        node_gradient_ev_per_angstrom: np.ndarray,
    ):
        """Return the fixed-geometry field-to-density JVP/VJP interface.

        The external node-field order is ``[V, dV/dx, dV/dy, dV/dz]`` in
        eV/e and eV/(e Å).  The returned density uses MACE-POLAR's native
        ``(n_atoms, 4)`` monopole/real-spherical convention.  This linearizes
        the learned response only; PCM and nuclear-coordinate derivatives are
        outside this interface.
        """

        return _MACEPolarDensityResponseLinearization(
            self,
            atoms,
            node_potential_ev=node_potential_ev,
            node_gradient_ev_per_angstrom=node_gradient_ev_per_angstrom,
        )

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        needs_forces = "forces" in properties
        needs_hessian = "hessian" in properties
        state, output = self.polar_state(
            atoms,
            compute_forces=needs_forces,
            compute_hessian=needs_hessian,
        )
        self._last_polar_state = state
        self._last_polar_state_numbers = np.asarray(
            atoms.numbers,
            dtype=int,
        ).copy()
        self._last_polar_state_positions = np.asarray(
            atoms.get_positions(),
            dtype=float,
        ).copy()

        forces_np = None
        if needs_forces:
            forces_np = state.fixed_field_forces_ev_per_angstrom
            if forces_np is None:
                raise RuntimeError("MACE-POLAR did not return requested forces.")

        hessian_np = None
        if needs_hessian:
            if self.solvent_correction is not None:
                raise NotImplementedError(
                    "Hessian calculation with implicit solvent is not implemented."
                )
            hessian = output.get("hessian")
            if hessian is None:
                hessian_np = self.get_hessian(atoms)
            else:
                hessian_np = (
                    hessian.detach().cpu().numpy().astype(float, copy=False)
                    * EV2HARTREE
                )

        self._finalize_results(
            atoms,
            energy=state.energy_ev,
            forces=forces_np,
            hessian=hessian_np,
        )

    def _analytic_hessian(self, atoms) -> np.ndarray:
        # Keep a fallback for upstream checkpoints that do not expose a Hessian
        # tensor from their standard forward.
        batch = self._batch_dict(atoms)
        positions = batch["positions"]
        positions.requires_grad_(True)

        def energy_fn():
            output = self._model_forward(
                batch,
                compute_force=False,
                compute_stress=False,
                compute_hessian=False,
            )
            return output["energy"].sum() * EV2HARTREE

        return hessian_via_double_autograd(energy_fn, positions)


class _MACEPolarDensityResponseLinearization:
    """Autograd JVP/VJP for MACE-POLAR density at one geometry and node field."""

    def __init__(
        self,
        calculator: MACEPolCalculator,
        atoms,
        *,
        node_potential_ev: np.ndarray,
        node_gradient_ev_per_angstrom: np.ndarray,
    ):
        potential = np.asarray(node_potential_ev, dtype=float)
        gradient = np.asarray(node_gradient_ev_per_angstrom, dtype=float)
        if potential.shape != (len(atoms),) or gradient.shape != (len(atoms), 3):
            raise ValueError(
                "Local reaction potential/gradient shapes must be "
                "(n_atoms,) and (n_atoms, 3)."
            )
        if not np.all(np.isfinite(potential)) or not np.all(np.isfinite(gradient)):
            raise ValueError("Local reaction potential/gradient must be finite.")
        self._calculator = calculator
        self._atoms = atoms.copy()
        self._potential = potential.copy()
        self._gradient = gradient.copy()
        self._shape = (len(atoms), 4)

    def _base_tensors(self, *, requires_grad: bool):
        potential = torch.tensor(
            self._potential,
            dtype=self._calculator.dtype,
            device=self._calculator.device,
            requires_grad=requires_grad,
        )
        gradient = torch.tensor(
            self._gradient,
            dtype=self._calculator.dtype,
            device=self._calculator.device,
            requires_grad=requires_grad,
        )
        return potential, gradient

    def _density_tensor(
        self,
        potential: torch.Tensor,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        output = self._calculator.polar_output_torch(
            self._atoms,
            node_potential_ev=potential,
            node_gradient_ev_per_angstrom=gradient,
        )
        density = output.get("density_coefficients")
        if density is None or density.shape != self._shape:
            received = None if density is None else tuple(density.shape)
            raise RuntimeError(
                "MACE-POLAR density response must have shape "
                f"{self._shape}; received {received}."
            )
        if not bool(torch.isfinite(density).all()):
            raise RuntimeError("MACE-POLAR density response is non-finite.")
        return density

    def _validated_block(self, values: np.ndarray, *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape != self._shape or not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} must be finite with shape {self._shape}; "
                f"received {array.shape}."
            )
        return array

    @staticmethod
    def _to_numpy(values: torch.Tensor) -> np.ndarray:
        return np.asarray(values.detach().cpu(), dtype=float).copy()

    def jvp(self, field_direction: np.ndarray) -> np.ndarray:
        """Apply the node-field-to-density Jacobian in external field order."""

        direction = self._validated_block(
            field_direction,
            name="field_direction",
        )
        potential, gradient = self._base_tensors(requires_grad=False)
        potential_direction = torch.as_tensor(
            direction[:, 0],
            dtype=self._calculator.dtype,
            device=self._calculator.device,
        )
        gradient_direction = torch.as_tensor(
            direction[:, 1:],
            dtype=self._calculator.dtype,
            device=self._calculator.device,
        )
        _, density_direction = torch.autograd.functional.jvp(
            self._density_tensor,
            (potential, gradient),
            (potential_direction, gradient_direction),
            create_graph=False,
            strict=True,
        )
        return self._to_numpy(density_direction)

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray:
        """Apply the discrete density-to-node-field adjoint by autograd."""

        cotangent = self._validated_block(
            density_cotangent,
            name="density_cotangent",
        )
        potential, gradient = self._base_tensors(requires_grad=True)
        density = self._density_tensor(potential, gradient)
        cotangent_tensor = torch.as_tensor(
            cotangent,
            dtype=self._calculator.dtype,
            device=self._calculator.device,
        )
        potential_cotangent, gradient_cotangent = torch.autograd.grad(
            density,
            (potential, gradient),
            grad_outputs=cotangent_tensor,
            create_graph=False,
        )
        return np.concatenate(
            (
                self._to_numpy(potential_cotangent)[:, None],
                self._to_numpy(gradient_cotangent),
            ),
            axis=1,
        )
