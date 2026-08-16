"""Frozen MACE-MDP moment coefficients for an independent polarization model.

MACE-MDP predicts permanent atomic charges/dipoles and a molecular
polarizability, but it does not expose a field-conditioned scalar energy.
This module therefore does **not** label the checkpoint variational and does
not admit any Route-2 capability.  It only provides an immutable,
content-addressed coefficient state that can be screened as the permanent and
linear-response input of a separately declared quadratic polarization scalar.

Torch and MACE are imported only by the runtime builder so dependency-light
Route-2 contract imports remain clean.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from maple.function.calculator.extra_correction.implicit.gto_density import (
    external_field_to_density_order,
)
from maple.solvation.api.capabilities import CapabilityStatus
from .base import model_charge_and_multiplicity, model_input_sha256

MACE_MDP_MOMENT_MODEL_PROFILE_ID = (
    "mace-mdp-route2-independent-variational-polarization-candidate-v1"
)
MACE_MDP_MOMENT_PROVIDER_ID = "maple.route2.model.mace-mdp-moments.impl.v1"
MACE_MDP_MODEL_TYPE = "DipolePolarizabilityMACE"
MACE_MDP_EXPECTED_CHECKPOINT_SHA256 = (
    "126f8d1602549e6fa0df775c701a5119ddeb0e3738202af8e7aa736de6c2b692"
)

# CODATA constants frozen by the earlier source-bound MACE-MDP response audit.
_BOHR_RADIUS_METERS = 5.29177210903e-11
_BOHR_RADIUS_ANGSTROM = 0.529177210903
_HARTREE_JOULE = 4.3597447222071e-18
_ELEMENTARY_CHARGE_COULOMB = 1.602176634e-19
_ATOMIC_FIELD_VOLT_PER_ANGSTROM = (
    _HARTREE_JOULE / (_ELEMENTARY_CHARGE_COULOMB * _BOHR_RADIUS_METERS) / 1.0e10
)
POLARIZABILITY_BOHR3_PER_EANGSTROM2_PER_VOLT = (
    _ATOMIC_FIELD_VOLT_PER_ANGSTROM / _BOHR_RADIUS_ANGSTROM
)


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA256 digest.") from exc
    return value.lower()


def _readonly(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if (
        (shape is not None and array.shape != shape)
        or array.size == 0
        or not np.all(np.isfinite(array))
    ):
        expected = "a nonempty finite array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must be finite with {expected}.")
    contiguous = np.ascontiguousarray(array, dtype=float)
    result = np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(
        contiguous.shape
    )
    return result


def _positive_atomic_numbers(value: object, *, atom_count: int) -> np.ndarray:
    raw = np.asarray(value)
    numeric = np.asarray(raw, dtype=float)
    if (
        numeric.shape != (atom_count,)
        or not np.all(np.isfinite(numeric))
        or not np.array_equal(numeric, np.rint(numeric))
        or np.any(numeric < 1.0)
    ):
        raise ValueError("atomic_numbers must contain positive integers.")
    contiguous = np.ascontiguousarray(numeric, dtype=np.int64)
    result = np.frombuffer(contiguous.tobytes(), dtype=np.int64)
    return result


def _relative_norm(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(right)), np.finfo(float).tiny)
    )


def _dipole_reconstruction_matches(
    *,
    charges: np.ndarray,
    positions: np.ndarray,
    atomic_dipoles: np.ndarray,
    public_dipole: np.ndarray,
) -> bool:
    """Compare two equivalent reductions with a cancellation-safe error bound.

    The public checkpoint dipole and the NumPy reconstruction sum the same
    atomic terms, but Torch scatter reduction and NumPy reduction need not
    round a nearly cancelled result identically.  A relative error is therefore
    undefined at molecular dipoles close to zero.  The standard floating-point
    summation bound uses the magnitude of the unreduced atomic terms instead.
    The factor two covers the two independently rounded reductions.
    """

    charge_terms = charges[:, None] * positions
    contributions = charge_terms + atomic_dipoles
    reconstructed = np.sum(contributions, axis=0)
    epsilon = np.finfo(np.float64).eps
    operation_count = len(charges) + 1
    gamma = (operation_count * epsilon) / (1.0 - operation_count * epsilon)
    component_scale = np.sum(
        np.abs(charge_terms) + np.abs(atomic_dipoles),
        axis=0,
    )
    roundoff_bound = 2.0 * gamma * component_scale
    return bool(np.all(np.abs(reconstructed - public_dipole) <= roundoff_bound))


@dataclass(frozen=True, slots=True)
class MACE_MDPMomentState:
    """One immutable permanent-moment and polarizability coefficient state."""

    provider_id: str
    model_profile_id: str
    configuration_sha256: str
    model_input_sha256: str
    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    charges_e: np.ndarray
    atomic_dipoles_eangstrom: np.ndarray
    atomic_polarizabilities_eangstrom2_per_volt: np.ndarray
    atomic_dipole_weights: np.ndarray
    public_dipole_eangstrom: np.ndarray
    public_polarizability_eangstrom2_per_volt: np.ndarray
    source4_raw_l1: np.ndarray
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if self.provider_id != MACE_MDP_MOMENT_PROVIDER_ID:
            raise ValueError("MACE-MDP provider identity is invalid.")
        if self.model_profile_id != MACE_MDP_MOMENT_MODEL_PROFILE_ID:
            raise ValueError("MACE-MDP model-profile identity is invalid.")
        configuration = _digest(self.configuration_sha256, name="configuration_sha256")
        model_input = _digest(self.model_input_sha256, name="model_input_sha256")
        positions = _readonly(self.positions_angstrom, name="positions_angstrom")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions_angstrom must have shape (N,3).")
        atom_count = positions.shape[0]
        numbers = _positive_atomic_numbers(self.atomic_numbers, atom_count=atom_count)
        charges = _readonly(self.charges_e, name="charges_e", shape=(atom_count,))
        dipoles = _readonly(
            self.atomic_dipoles_eangstrom,
            name="atomic_dipoles_eangstrom",
            shape=(atom_count, 3),
        )
        atomic_alpha = _readonly(
            self.atomic_polarizabilities_eangstrom2_per_volt,
            name="atomic_polarizabilities_eangstrom2_per_volt",
            shape=(atom_count, 3, 3),
        )
        weights = _readonly(
            self.atomic_dipole_weights,
            name="atomic_dipole_weights",
            shape=(atom_count, 3, 3),
        )
        public_dipole = _readonly(
            self.public_dipole_eangstrom,
            name="public_dipole_eangstrom",
            shape=(3,),
        )
        public_alpha = _readonly(
            self.public_polarizability_eangstrom2_per_volt,
            name="public_polarizability_eangstrom2_per_volt",
            shape=(3, 3),
        )
        source = _readonly(
            self.source4_raw_l1,
            name="source4_raw_l1",
            shape=(atom_count, 4),
        )

        scale = max(1.0, float(np.linalg.norm(public_alpha, ord=2)))
        if (
            float(np.linalg.norm(public_alpha - public_alpha.T, ord=2))
            > 1.0e-10 * scale
        ):
            raise ValueError("MACE-MDP molecular polarizability is not reciprocal.")
        if float(np.min(np.linalg.eigvalsh(public_alpha))) <= 1.0e-12 * scale:
            raise ValueError(
                "MACE-MDP molecular polarizability is not positive definite."
            )
        if _relative_norm(np.sum(atomic_alpha, axis=0), public_alpha) > 1.0e-10:
            raise ValueError("Atomic polarizabilities do not reconstruct the total.")
        if (
            float(np.linalg.norm(np.sum(weights, axis=0) - np.eye(3), ord="fro"))
            > 1.0e-10
        ):
            raise ValueError("Atomic dipole weights do not preserve molecular dipole.")

        expected_source = external_field_to_density_order(
            np.concatenate((charges[:, None], dipoles), axis=1)
        )
        if not np.allclose(source, expected_source, rtol=0.0, atol=2.0e-15):
            raise ValueError("Raw-l1 source does not match the Cartesian moments.")
        if not _dipole_reconstruction_matches(
            charges=charges,
            positions=positions,
            atomic_dipoles=dipoles,
            public_dipole=public_dipole,
        ):
            raise ValueError("Atomic moments do not reconstruct the public dipole.")

        payload = {
            "provider_id": self.provider_id,
            "model_profile_id": self.model_profile_id,
            "configuration_sha256": configuration,
            "model_input_sha256": model_input,
            "atomic_numbers": numbers.tolist(),
            "positions_angstrom": positions.tolist(),
            "charges_e": charges.tolist(),
            "atomic_dipoles_eangstrom": dipoles.tolist(),
            "atomic_polarizabilities_eangstrom2_per_volt": atomic_alpha.tolist(),
            "atomic_dipole_weights": weights.tolist(),
            "public_dipole_eangstrom": public_dipole.tolist(),
            "public_polarizability_eangstrom2_per_volt": public_alpha.tolist(),
            "source4_raw_l1": source.tolist(),
        }
        expected_hash = _hash(payload)
        if self.state_sha256 and self.state_sha256 != expected_hash:
            raise ValueError("MACE-MDP state_sha256 does not match its content.")
        object.__setattr__(self, "configuration_sha256", configuration)
        object.__setattr__(self, "model_input_sha256", model_input)
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "positions_angstrom", positions)
        object.__setattr__(self, "charges_e", charges)
        object.__setattr__(self, "atomic_dipoles_eangstrom", dipoles)
        object.__setattr__(
            self, "atomic_polarizabilities_eangstrom2_per_volt", atomic_alpha
        )
        object.__setattr__(self, "atomic_dipole_weights", weights)
        object.__setattr__(self, "public_dipole_eangstrom", public_dipole)
        object.__setattr__(
            self, "public_polarizability_eangstrom2_per_volt", public_alpha
        )
        object.__setattr__(self, "source4_raw_l1", source)
        object.__setattr__(self, "state_sha256", expected_hash)

    @property
    def total_charge_e(self) -> float:
        return float(np.sum(self.charges_e))

    @property
    def polarizability_bohr3(self) -> np.ndarray:
        result = np.asarray(
            self.public_polarizability_eangstrom2_per_volt
            * POLARIZABILITY_BOHR3_PER_EANGSTROM2_PER_VOLT,
            dtype=float,
        )
        result.setflags(write=False)
        return result


def build_mace_mdp_moment_state(
    *,
    configuration_sha256: str,
    model_input_sha256_value: str,
    atomic_numbers: object,
    positions_angstrom: object,
    charges_e: object,
    atomic_dipoles_eangstrom: object,
    atomic_polarizabilities_eangstrom2_per_volt: object,
    public_dipole_eangstrom: object,
    public_polarizability_eangstrom2_per_volt: object,
) -> MACE_MDPMomentState:
    """Build and cross-check one pure-NumPy MACE-MDP coefficient state."""

    alpha = np.asarray(atomic_polarizabilities_eangstrom2_per_volt, dtype=float)
    total = np.asarray(public_polarizability_eangstrom2_per_volt, dtype=float)
    try:
        weights = alpha @ np.linalg.inv(total)
    except np.linalg.LinAlgError as exc:
        raise ValueError("MACE-MDP molecular polarizability is singular.") from exc
    charges = np.asarray(charges_e, dtype=float)
    dipoles = np.asarray(atomic_dipoles_eangstrom, dtype=float)
    source = external_field_to_density_order(
        np.concatenate((charges[:, None], dipoles), axis=1)
    )
    return MACE_MDPMomentState(
        provider_id=MACE_MDP_MOMENT_PROVIDER_ID,
        model_profile_id=MACE_MDP_MOMENT_MODEL_PROFILE_ID,
        configuration_sha256=configuration_sha256,
        model_input_sha256=model_input_sha256_value,
        atomic_numbers=np.asarray(atomic_numbers),
        positions_angstrom=np.asarray(positions_angstrom),
        charges_e=charges,
        atomic_dipoles_eangstrom=dipoles,
        atomic_polarizabilities_eangstrom2_per_volt=alpha,
        atomic_dipole_weights=weights,
        public_dipole_eangstrom=np.asarray(public_dipole_eangstrom),
        public_polarizability_eangstrom2_per_volt=total,
        source4_raw_l1=source,
    )


class MACE_MDPMomentAdapter:
    """Sealed runtime adapter for the frozen MACE-MDP coefficient checkpoint."""

    __slots__ = (
        "_calculator",
        "_checkpoint_sha256",
        "_configuration_sha256",
        "_runtime_sources",
        "_sealed",
    )

    provider_id = MACE_MDP_MOMENT_PROVIDER_ID
    model_profile_id = MACE_MDP_MOMENT_MODEL_PROFILE_ID
    capabilities = CapabilityStatus()

    def __init__(
        self,
        *,
        calculator: object,
        checkpoint_sha256: str,
        runtime_sources: tuple[tuple[str, str], ...],
    ) -> None:
        for name in ("get_property", "_atoms_to_batch", "_clone_batch"):
            if not callable(getattr(calculator, name, None)):
                raise TypeError(f"MACE-MDP calculator requires {name}().")
        models = getattr(calculator, "models", None)
        if not isinstance(models, (list, tuple)) or len(models) != 1:
            raise TypeError("MACE-MDP calculator must contain exactly one model.")
        checkpoint = _digest(checkpoint_sha256, name="checkpoint_sha256")
        normalized_sources = tuple(sorted(runtime_sources))
        for name, digest in normalized_sources:
            if not isinstance(name, str) or not name:
                raise ValueError("Runtime source names must be nonempty.")
            _digest(digest, name=f"runtime source {name}")
        configuration = _hash(
            {
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "model_type": MACE_MDP_MODEL_TYPE,
                "checkpoint_sha256": checkpoint,
                "runtime_sources": [list(item) for item in normalized_sources],
                "coefficient_semantics": (
                    "permanent atomic q/p plus molecular-alpha atom partition"
                ),
                "energy_role": "coefficients-only-no-field-energy",
            }
        )
        object.__setattr__(self, "_calculator", calculator)
        object.__setattr__(self, "_checkpoint_sha256", checkpoint)
        object.__setattr__(self, "_runtime_sources", normalized_sources)
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACE_MDPMomentAdapter is sealed.")
        object.__setattr__(self, name, value)

    def configuration_sha256(self) -> str:
        return self._configuration_sha256

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def runtime_sources(self) -> tuple[tuple[str, str], ...]:
        return self._runtime_sources

    def evaluate(self, atoms: object) -> MACE_MDPMomentState:
        charge, multiplicity = model_charge_and_multiplicity(atoms)
        if (charge, multiplicity) != (0, 1):
            raise ValueError(
                "Frozen MACE-MDP candidate supports neutral singlets only."
            )
        positions = np.asarray(getattr(atoms, "positions"), dtype=float)
        numbers = np.asarray(getattr(atoms, "numbers"))
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or len(numbers) != len(positions)
        ):
            raise ValueError("atoms must expose finite matching positions/numbers.")

        calculator = self._calculator
        public_dipole = np.asarray(
            calculator.get_property("dipole", atoms), dtype=float
        )
        public_alpha = np.asarray(
            calculator.get_property("polarizability", atoms), dtype=float
        )
        model = calculator.models[0]
        readouts = tuple(getattr(model, "readouts", ()))
        if not readouts:
            raise RuntimeError("Frozen MACE-MDP model has no readout blocks.")
        captured: list[Any] = []

        def capture(_module: object, _inputs: object, output: object) -> None:
            if not hasattr(output, "detach"):
                raise RuntimeError("MACE-MDP readout did not return a tensor.")
            captured.append(output.detach().clone())

        handles = [readout.register_forward_hook(capture) for readout in readouts]
        try:
            batch = calculator._atoms_to_batch(atoms)
            output = model(
                calculator._clone_batch(batch).to_dict(),
                compute_dielectric_derivatives=False,
                training=False,
            )
        finally:
            for handle in handles:
                handle.remove()
        if not isinstance(output, dict) or len(captured) != len(readouts):
            raise RuntimeError("MACE-MDP forward/readout sequence did not close.")

        def array(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
            if not hasattr(value, "detach"):
                raise RuntimeError(f"{name} is not tensor-like.")
            result = np.asarray(value.detach().cpu().numpy(), dtype=float)
            if result.shape != shape or not np.all(np.isfinite(result)):
                raise RuntimeError(f"{name} must be finite with shape {shape}.")
            return result

        atom_count = len(positions)
        direct_dipole = array(output.get("dipole"), name="direct dipole", shape=(1, 3))[
            0
        ]
        direct_alpha = array(
            output.get("polarizability"),
            name="direct polarizability",
            shape=(1, 3, 3),
        )[0]
        if _relative_norm(direct_dipole, public_dipole) > 1.0e-12:
            raise RuntimeError("Public/direct MACE-MDP dipoles differ.")
        if _relative_norm(direct_alpha, public_alpha) > 1.0e-12:
            raise RuntimeError("Public/direct MACE-MDP polarizabilities differ.")
        charges = array(
            output.get("charges"), name="atomic charges", shape=(atom_count,)
        )
        dipoles = array(
            output.get("atomic_dipoles"),
            name="atomic dipoles",
            shape=(atom_count, 3),
        )

        torch = __import__("torch")
        vectors: list[Any] = []
        for raw in captured:
            node_output = raw.squeeze(-1)
            if tuple(node_output.shape) != (atom_count, 10):
                raise RuntimeError("Frozen MACE-MDP readout layout changed.")
            vectors.append(torch.cat((node_output[:, 1:2], node_output[:, 5:]), dim=-1))
        atomic_alpha_sh = torch.stack(vectors, dim=-1).sum(dim=-1)
        atomic_alpha = torch.einsum(
            "ijk,ai->ajk", model.change_of_basis, atomic_alpha_sh
        )
        atomic_alpha_array = array(
            atomic_alpha,
            name="atomic polarizabilities",
            shape=(atom_count, 3, 3),
        )
        return build_mace_mdp_moment_state(
            configuration_sha256=self.configuration_sha256(),
            model_input_sha256_value=model_input_sha256(atoms),
            atomic_numbers=numbers,
            positions_angstrom=positions,
            charges_e=charges,
            atomic_dipoles_eangstrom=dipoles,
            atomic_polarizabilities_eangstrom2_per_volt=atomic_alpha_array,
            public_dipole_eangstrom=public_dipole,
            public_polarizability_eangstrom2_per_volt=public_alpha,
        )

    def source_position_vjp(
        self, atoms: object, source_cotangent: object
    ) -> np.ndarray:
        """Return ``d<c(R),bar_c>/dR`` from the native MACE-MDP graph.

        The source coordinates use the repository's raw real-spherical order
        ``[q,y,z,x]``.  Positions remain in Angstrom, so a cotangent carrying
        the dual source units produces an ``(N,3)`` coordinate gradient per
        Angstrom.  No finite differences or reconstructed receiver model are
        used here.
        """

        charge, multiplicity = model_charge_and_multiplicity(atoms)
        if (charge, multiplicity) != (0, 1):
            raise ValueError(
                "Frozen MACE-MDP candidate supports neutral singlets only."
            )
        positions = np.asarray(getattr(atoms, "positions"), dtype=float)
        cotangent = np.asarray(source_cotangent, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or cotangent.shape != (len(positions), 4)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(cotangent))
        ):
            raise ValueError(
                "MACE-MDP source cotangent must be finite with shape (N,4)."
            )

        torch = __import__("torch")
        calculator = self._calculator
        batch = calculator._atoms_to_batch(atoms)
        data = calculator._clone_batch(batch).to_dict()
        position_tensor = data["positions"].detach().clone().requires_grad_(True)
        data["positions"] = position_tensor
        output = calculator.models[0](
            data,
            compute_dielectric_derivatives=False,
            training=False,
        )
        charges = output.get("charges")
        dipoles = output.get("atomic_dipoles")
        if (
            charges is None
            or dipoles is None
            or tuple(charges.shape) != (len(positions),)
            or tuple(dipoles.shape) != (len(positions), 3)
        ):
            raise RuntimeError(
                "MACE-MDP did not return differentiable atomic q/p values."
            )
        # Cartesian [x,y,z] -> raw real-spherical [y,z,x].
        raw_source = torch.cat((charges[:, None], dipoles[:, (1, 2, 0)]), dim=1)
        cotangent_tensor = torch.as_tensor(
            cotangent,
            dtype=raw_source.dtype,
            device=raw_source.device,
        )
        contraction = torch.sum(raw_source * cotangent_tensor)
        (gradient,) = torch.autograd.grad(
            contraction,
            position_tensor,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        result = np.asarray(gradient.detach().cpu().numpy(), dtype=float)
        if result.shape != positions.shape or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "MACE-MDP source position VJP must be finite with shape (N,3)."
            )
        return result.copy()


def build_mace_mdp_moment_adapter(
    *,
    checkpoint_path: str | Path,
    device: str = "cpu",
    expected_checkpoint_sha256: str = MACE_MDP_EXPECTED_CHECKPOINT_SHA256,
) -> MACE_MDPMomentAdapter:
    """Load the frozen coefficient checkpoint under its audited CPU policy."""

    if device != "cpu":
        raise ValueError("The frozen MACE-MDP coefficient profile is CPU/float64 only.")
    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint_sha = _sha256_file(checkpoint)
    if checkpoint_sha != _digest(
        expected_checkpoint_sha256, name="expected_checkpoint_sha256"
    ):
        raise ValueError("MACE-MDP checkpoint SHA256 does not match the frozen model.")
    mace_module = __import__("mace.calculators", fromlist=["MACECalculator"])
    calculator_type = getattr(mace_module, "MACECalculator", None)
    if calculator_type is None:
        raise RuntimeError("Installed MACE runtime has no MACECalculator.")
    calculator = calculator_type(
        model_paths=str(checkpoint),
        model_type=MACE_MDP_MODEL_TYPE,
        default_dtype="float64",
        device=device,
    )
    sources: dict[str, str] = {}
    for label, value in (
        ("mace_calculator", calculator_type),
        ("mace_model", type(calculator.models[0])),
    ):
        raw = inspect.getsourcefile(value)
        if raw is not None:
            path = Path(raw).resolve()
            if path.is_file():
                sources[label] = _sha256_file(path)
    sources["maple_mace_mdp_adapter"] = _sha256_file(Path(__file__))
    return MACE_MDPMomentAdapter(
        calculator=calculator,
        checkpoint_sha256=checkpoint_sha,
        runtime_sources=tuple(sources.items()),
    )


__all__ = [
    "MACE_MDP_EXPECTED_CHECKPOINT_SHA256",
    "MACE_MDP_MODEL_TYPE",
    "MACE_MDP_MOMENT_MODEL_PROFILE_ID",
    "MACE_MDP_MOMENT_PROVIDER_ID",
    "POLARIZABILITY_BOHR3_PER_EANGSTROM2_PER_VOLT",
    "MACE_MDPMomentAdapter",
    "MACE_MDPMomentState",
    "build_mace_mdp_moment_adapter",
    "build_mace_mdp_moment_state",
]
