"""QM-source-supervised permanent multipoles on a frozen MACE-MDP backbone.

The original MACE-MDP atomwise charge-like and local-dipole contributions are
identified only through molecular observables.  They are therefore not a
quantitative near-field PCM source.  This research provider replaces that
latent partition by two small readouts trained exclusively against independent
SPICE MBIS atomic multipoles.  It then performs one differentiable metric
projection that preserves the frozen checkpoint's total molecular charge and
dipole exactly.

This module does not read solvation labels, does not fit a solvent correction,
and does not admit energy or force capability.  Torch and MACE remain lazy
runtime dependencies so dependency-light contract imports stay clean.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE

from .base import atom_count, model_charge_and_multiplicity, model_input_sha256
from .mace_mdp import (
    MACE_MDP_EXPECTED_CHECKPOINT_SHA256,
    MACE_MDP_MODEL_TYPE,
)

MACE_MDP_MBIS_SOURCE_PROVIDER_ID = (
    "maple.route2.model.mace-mdp-frozen-mbis-source-head.impl.v1"
)
MACE_MDP_MBIS_SOURCE_MODEL_PROFILE_ID = (
    "mace-mdp-frozen-backbone-mbis-atomic-source-head-prototype-v1"
)
MACE_MDP_MBIS_SOURCE_HEAD_EXPECTED_SHA256 = (
    "ee476dda6d30191fbaa025aa4e857a7687e797af9d1c9ad75ba5552968282d32"
)
MACE_MDP_MBIS_SOURCE_HEAD_CONTRACT = "mace-mdp-mbis-source-head-runtime-v1"

SUPPORTED_ATOMIC_NUMBERS = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
PRODUCT_LAYER_COUNT = 2
PRODUCT_WIDTH = 1152
SCALAR_CHANNELS_PER_LAYER = 128
VECTOR_CHANNELS_PER_LAYER = 128
SCALAR_FEATURE_COUNT = PRODUCT_LAYER_COUNT * SCALAR_CHANNELS_PER_LAYER
VECTOR_FEATURE_COUNT = PRODUCT_LAYER_COUNT * VECTOR_CHANNELS_PER_LAYER

_HEAD_KEYS = frozenset(
    {
        "scalar_mean",
        "scalar_scale",
        "vector_scale",
        "charge_weights",
        "dipole_weights",
        "charge_sigma",
        "dipole_sigma",
        "atomic_numbers",
    }
)


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


def _metadata_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _readonly(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    positive: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.shape != shape
        or not np.all(np.isfinite(array))
        or (positive and np.any(array <= 0.0))
    ):
        qualifier = "finite positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier} with shape {shape}.")
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(shape)


@dataclass(frozen=True, slots=True)
class MACE_MDPMBISSourceHeadAsset:
    """Immutable numeric readout asset loaded from a content-addressed NPZ."""

    sha256: str
    scalar_mean: np.ndarray
    scalar_scale: np.ndarray
    vector_scale: np.ndarray
    charge_weights: np.ndarray
    dipole_weights: np.ndarray
    charge_sigma_e: float
    dipole_sigma_eangstrom: float
    atomic_numbers: tuple[int, ...]

    def __post_init__(self) -> None:
        digest = _digest(self.sha256, name="source-head sha256")
        scalar_mean = _readonly(
            self.scalar_mean,
            name="scalar_mean",
            shape=(SCALAR_FEATURE_COUNT,),
        )
        scalar_scale = _readonly(
            self.scalar_scale,
            name="scalar_scale",
            shape=(SCALAR_FEATURE_COUNT,),
            positive=True,
        )
        vector_scale = _readonly(
            self.vector_scale,
            name="vector_scale",
            shape=(VECTOR_FEATURE_COUNT,),
            positive=True,
        )
        charge_weights = _readonly(
            self.charge_weights,
            name="charge_weights",
            shape=(SCALAR_FEATURE_COUNT + len(SUPPORTED_ATOMIC_NUMBERS),),
        )
        dipole_weights = _readonly(
            self.dipole_weights,
            name="dipole_weights",
            shape=(VECTOR_FEATURE_COUNT,),
        )
        charge_sigma = float(self.charge_sigma_e)
        dipole_sigma = float(self.dipole_sigma_eangstrom)
        numbers = tuple(int(value) for value in self.atomic_numbers)
        if (
            not np.isfinite(charge_sigma)
            or charge_sigma <= 0.0
            or not np.isfinite(dipole_sigma)
            or dipole_sigma <= 0.0
        ):
            raise ValueError("Projection metric scales must be finite and positive.")
        if numbers != SUPPORTED_ATOMIC_NUMBERS:
            raise ValueError("Source-head atomic-number order is incompatible.")
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "scalar_mean", scalar_mean)
        object.__setattr__(self, "scalar_scale", scalar_scale)
        object.__setattr__(self, "vector_scale", vector_scale)
        object.__setattr__(self, "charge_weights", charge_weights)
        object.__setattr__(self, "dipole_weights", dipole_weights)
        object.__setattr__(self, "charge_sigma_e", charge_sigma)
        object.__setattr__(self, "dipole_sigma_eangstrom", dipole_sigma)
        object.__setattr__(self, "atomic_numbers", numbers)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        expected_sha256: str = MACE_MDP_MBIS_SOURCE_HEAD_EXPECTED_SHA256,
    ) -> "MACE_MDPMBISSourceHeadAsset":
        source = Path(path).expanduser().resolve(strict=True)
        actual = _sha256_file(source)
        if actual != _digest(expected_sha256, name="expected source-head sha256"):
            raise ValueError("MACE-MDP MBIS source-head SHA256 does not match.")
        with np.load(source, allow_pickle=False) as payload:
            if frozenset(payload.files) != _HEAD_KEYS:
                raise ValueError("MACE-MDP MBIS source-head schema is incompatible.")
            return cls(
                sha256=actual,
                scalar_mean=payload["scalar_mean"],
                scalar_scale=payload["scalar_scale"],
                vector_scale=payload["vector_scale"],
                charge_weights=payload["charge_weights"],
                dipole_weights=payload["dipole_weights"],
                charge_sigma_e=float(payload["charge_sigma"]),
                dipole_sigma_eangstrom=float(payload["dipole_sigma"]),
                atomic_numbers=tuple(int(v) for v in payload["atomic_numbers"]),
            )


@dataclass(frozen=True, slots=True)
class MACE_MDPMBISSourceState:
    """Geometry-bound projected source and its diagnostic components."""

    configuration_sha256: str
    model_input_sha256: str
    total_charge_e: float
    positions_angstrom: np.ndarray
    raw_charges_e: np.ndarray
    raw_dipoles_eangstrom: np.ndarray
    latent_mdp_charges_e: np.ndarray
    latent_mdp_dipoles_eangstrom: np.ndarray
    charges_e: np.ndarray
    dipoles_eangstrom: np.ndarray
    public_molecular_dipole_eangstrom: np.ndarray
    source4_raw_l1: np.ndarray
    projection_condition_number: float
    state_sha256: str = ""

    def __post_init__(self) -> None:
        configuration = _digest(self.configuration_sha256, name="configuration_sha256")
        input_digest = _digest(self.model_input_sha256, name="model_input_sha256")
        total_charge = float(self.total_charge_e)
        if not np.isfinite(total_charge) or total_charge != 0.0:
            raise ValueError("The prototype source state is neutral-only.")
        positions = np.asarray(self.positions_angstrom, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
            raise ValueError("positions_angstrom must have shape (N,3).")
        count = len(positions)
        positions = _readonly(positions, name="positions_angstrom", shape=(count, 3))
        raw_q = _readonly(self.raw_charges_e, name="raw_charges_e", shape=(count,))
        raw_p = _readonly(
            self.raw_dipoles_eangstrom,
            name="raw_dipoles_eangstrom",
            shape=(count, 3),
        )
        latent_q = _readonly(
            self.latent_mdp_charges_e,
            name="latent_mdp_charges_e",
            shape=(count,),
        )
        latent_p = _readonly(
            self.latent_mdp_dipoles_eangstrom,
            name="latent_mdp_dipoles_eangstrom",
            shape=(count, 3),
        )
        charges = _readonly(self.charges_e, name="charges_e", shape=(count,))
        dipoles = _readonly(
            self.dipoles_eangstrom,
            name="dipoles_eangstrom",
            shape=(count, 3),
        )
        molecular = _readonly(
            self.public_molecular_dipole_eangstrom,
            name="public_molecular_dipole_eangstrom",
            shape=(3,),
        )
        source = _readonly(
            self.source4_raw_l1,
            name="source4_raw_l1",
            shape=(count, 4),
        )
        expected_source = np.concatenate(
            (charges[:, None], dipoles[:, (1, 2, 0)]), axis=1
        )
        if not np.allclose(source, expected_source, rtol=0.0, atol=2.0e-15):
            raise ValueError("Raw-l1 source does not match Cartesian multipoles.")
        reconstructed = np.sum(charges[:, None] * positions + dipoles, axis=0)
        latent_reconstructed = np.sum(
            latent_q[:, None] * positions + latent_p,
            axis=0,
        )
        if abs(float(np.sum(charges)) - total_charge) > 2.0e-11:
            raise ValueError("Projected source does not close total charge.")
        if not np.allclose(reconstructed, molecular, rtol=0.0, atol=2.0e-11):
            raise ValueError("Projected source does not close the molecular dipole.")
        if not np.allclose(
            latent_reconstructed,
            molecular,
            rtol=0.0,
            atol=2.0e-11,
        ):
            raise ValueError("Latent MDP source does not close the molecular dipole.")
        condition = float(self.projection_condition_number)
        if not np.isfinite(condition) or condition <= 0.0:
            raise ValueError("Projection condition number must be finite and positive.")
        payload = {
            "contract": MACE_MDP_MBIS_SOURCE_HEAD_CONTRACT,
            "configuration_sha256": configuration,
            "model_input_sha256": input_digest,
            "total_charge_e": total_charge,
            "positions_angstrom": positions.tolist(),
            "raw_charges_e": raw_q.tolist(),
            "raw_dipoles_eangstrom": raw_p.tolist(),
            "latent_mdp_charges_e": latent_q.tolist(),
            "latent_mdp_dipoles_eangstrom": latent_p.tolist(),
            "charges_e": charges.tolist(),
            "dipoles_eangstrom": dipoles.tolist(),
            "public_molecular_dipole_eangstrom": molecular.tolist(),
            "source4_raw_l1": source.tolist(),
            "projection_condition_number": condition,
        }
        expected_hash = _metadata_sha256(payload)
        if self.state_sha256 and self.state_sha256 != expected_hash:
            raise ValueError("MACE-MDP MBIS source state hash does not match content.")
        object.__setattr__(self, "configuration_sha256", configuration)
        object.__setattr__(self, "model_input_sha256", input_digest)
        object.__setattr__(self, "total_charge_e", total_charge)
        object.__setattr__(self, "positions_angstrom", positions)
        object.__setattr__(self, "raw_charges_e", raw_q)
        object.__setattr__(self, "raw_dipoles_eangstrom", raw_p)
        object.__setattr__(self, "latent_mdp_charges_e", latent_q)
        object.__setattr__(self, "latent_mdp_dipoles_eangstrom", latent_p)
        object.__setattr__(self, "charges_e", charges)
        object.__setattr__(self, "dipoles_eangstrom", dipoles)
        object.__setattr__(self, "public_molecular_dipole_eangstrom", molecular)
        object.__setattr__(self, "source4_raw_l1", source)
        object.__setattr__(self, "projection_condition_number", condition)
        object.__setattr__(self, "state_sha256", expected_hash)


def _project_source_torch(
    *,
    torch: Any,
    positions: Any,
    raw_charges: Any,
    raw_dipoles: Any,
    total_charge: int,
    molecular_dipole: Any,
    charge_sigma: float,
    dipole_sigma: float,
) -> tuple[Any, Any, Any]:
    """Return the exact metric projection using a differentiable 4x4 solve."""

    count = int(raw_charges.shape[0])
    q_variance = raw_charges.new_tensor(charge_sigma**2)
    p_variance = raw_charges.new_tensor(dipole_sigma**2)
    one = raw_charges.new_ones((count,))
    constraint = raw_charges.new_zeros((4, 4 * count))
    constraint[0, :count] = one
    constraint[1:, :count] = positions.T
    identity = torch.eye(3, dtype=positions.dtype, device=positions.device)
    for atom_index in range(count):
        start = count + 3 * atom_index
        constraint[1:, start : start + 3] = identity
    source = torch.cat((raw_charges, raw_dipoles.reshape(-1)))
    covariance = torch.cat(
        (
            q_variance.expand(count),
            p_variance.expand(3 * count),
        )
    )
    target = torch.cat(
        (
            raw_charges.new_tensor([float(total_charge)]),
            molecular_dipole.reshape(3),
        )
    )
    residual = target - constraint @ source
    schur = (constraint * covariance.unsqueeze(0)) @ constraint.T
    multipliers = torch.linalg.solve(schur, residual)
    projected = source + covariance * (constraint.T @ multipliers)
    return projected[:count], projected[count:].reshape(count, 3), schur


class MACE_MDPMBISSourceAdapter:
    """Sealed differentiable permanent-source provider for the new profile."""

    __slots__ = (
        "_asset",
        "_calculator",
        "_checkpoint_sha256",
        "_configuration_sha256",
        "_device",
        "_runtime_identity",
        "_runtime_sources",
        "_sealed",
    )

    provider_id = MACE_MDP_MBIS_SOURCE_PROVIDER_ID
    model_profile_id = MACE_MDP_MBIS_SOURCE_MODEL_PROFILE_ID
    source_space = ATOMIC_L1_SOURCE_SPACE
    capabilities = CapabilityStatus()
    variational_functional_admitted = False

    def __init__(
        self,
        *,
        calculator: object,
        asset: MACE_MDPMBISSourceHeadAsset,
        checkpoint_sha256: str,
        device: str,
        runtime_identity: tuple[tuple[str, str], ...],
        runtime_sources: tuple[tuple[str, str], ...],
    ) -> None:
        for name in ("_atoms_to_batch", "_clone_batch"):
            if not callable(getattr(calculator, name, None)):
                raise TypeError(f"MACE-MDP source adapter requires {name}().")
        models = getattr(calculator, "models", None)
        if not isinstance(models, (list, tuple)) or len(models) != 1:
            raise TypeError("MACE-MDP source adapter requires exactly one model.")
        model = models[0]
        products = tuple(getattr(model, "products", ()))
        if len(products) != PRODUCT_LAYER_COUNT:
            raise RuntimeError("MACE-MDP product-layer count changed.")
        model_numbers = tuple(int(value) for value in model.atomic_numbers.tolist())
        if model_numbers != SUPPORTED_ATOMIC_NUMBERS:
            raise RuntimeError("MACE-MDP element order changed.")
        checkpoint = _digest(checkpoint_sha256, name="checkpoint_sha256")
        if not isinstance(asset, MACE_MDPMBISSourceHeadAsset):
            raise TypeError("asset must be MACE_MDPMBISSourceHeadAsset.")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a nonempty string.")
        sources = tuple(sorted(runtime_sources))
        for name, digest in sources:
            if not isinstance(name, str) or not name:
                raise ValueError("Runtime source names must be nonempty.")
            _digest(digest, name=f"runtime source {name}")
        identity = tuple(sorted(runtime_identity))
        if not identity or any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or not value
            for name, value in identity
        ):
            raise ValueError("Runtime identity must contain nonempty string pairs.")
        configuration = _metadata_sha256(
            {
                "contract": MACE_MDP_MBIS_SOURCE_HEAD_CONTRACT,
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "checkpoint_sha256": checkpoint,
                "source_head_sha256": asset.sha256,
                "device": device,
                "dtype": "float64",
                "runtime_identity": [list(item) for item in identity],
                "runtime_sources": [list(item) for item in sources],
                "source_space_sha256": self.source_space.metadata_hash(),
                "projection_target": (
                    "frozen checkpoint total charge and molecular dipole"
                ),
                "solvation_labels_used": False,
                "capabilities": "none",
            }
        )
        object.__setattr__(self, "_asset", asset)
        object.__setattr__(self, "_calculator", calculator)
        object.__setattr__(self, "_checkpoint_sha256", checkpoint)
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_runtime_identity", identity)
        object.__setattr__(self, "_runtime_sources", sources)
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACE_MDPMBISSourceAdapter is sealed.")
        object.__setattr__(self, name, value)

    def configuration_sha256(self) -> str:
        return self._configuration_sha256

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def source_head_sha256(self) -> str:
        return self._asset.sha256

    @property
    def device(self) -> str:
        return self._device

    @property
    def runtime_identity(self) -> tuple[tuple[str, str], ...]:
        return self._runtime_identity

    def _forward(
        self, atoms: object, *, need_position_gradient: bool
    ) -> tuple[Any, ...]:
        charge, multiplicity = model_charge_and_multiplicity(atoms)
        if (charge, multiplicity) != (0, 1):
            raise ValueError(
                "MACE-MDP MBIS source prototype supports neutral singlets only."
            )
        numbers = np.asarray(getattr(atoms, "numbers", ()), dtype=np.int64)
        count = atom_count(atoms)
        if numbers.shape != (count,) or not set(numbers).issubset(
            SUPPORTED_ATOMIC_NUMBERS
        ):
            raise ValueError("Geometry contains unsupported atomic numbers.")
        torch = __import__("torch")
        calculator = self._calculator
        batch = calculator._atoms_to_batch(atoms)
        data = calculator._clone_batch(batch).to_dict()
        positions = data["positions"].detach().clone()
        positions.requires_grad_(need_position_gradient)
        data["positions"] = positions
        model = calculator.models[0]
        captured: list[Any] = []

        def capture(_module: object, _inputs: object, output: object) -> None:
            captured.append(output)

        handles = [product.register_forward_hook(capture) for product in model.products]
        try:
            output = model(
                data,
                compute_dielectric_derivatives=False,
                training=False,
            )
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != PRODUCT_LAYER_COUNT or any(
            tuple(value.shape) != (count, PRODUCT_WIDTH) for value in captured
        ):
            raise RuntimeError("MACE-MDP hidden representation changed.")
        scalar = torch.cat(
            [value[:, :SCALAR_CHANNELS_PER_LAYER] for value in captured], dim=1
        )
        vector = torch.cat(
            [
                value[
                    :,
                    SCALAR_CHANNELS_PER_LAYER : SCALAR_CHANNELS_PER_LAYER
                    + 3 * VECTOR_CHANNELS_PER_LAYER,
                ].reshape(count, VECTOR_CHANNELS_PER_LAYER, 3)
                for value in captured
            ],
            dim=1,
        )
        dtype = scalar.dtype
        device = scalar.device
        asset = self._asset
        # The asset arrays intentionally use immutable byte-backed storage.
        # ``torch.tensor`` copies them, avoiding a writable alias to that audit
        # boundary while preserving the differentiable graph downstream.
        scalar_mean = torch.tensor(asset.scalar_mean, dtype=dtype, device=device)
        scalar_scale = torch.tensor(asset.scalar_scale, dtype=dtype, device=device)
        vector_scale = torch.tensor(asset.vector_scale, dtype=dtype, device=device)
        charge_weights = torch.tensor(asset.charge_weights, dtype=dtype, device=device)
        dipole_weights = torch.tensor(asset.dipole_weights, dtype=dtype, device=device)
        number_tensor = torch.as_tensor(numbers, dtype=torch.long, device=device)
        supported = torch.as_tensor(
            SUPPORTED_ATOMIC_NUMBERS, dtype=torch.long, device=device
        )
        one_hot = (number_tensor[:, None] == supported[None, :]).to(dtype=dtype)
        q_features = torch.cat(((scalar - scalar_mean) / scalar_scale, one_hot), dim=1)
        raw_charges = q_features @ charge_weights
        normalized_vector = vector / vector_scale[None, :, None]
        raw_dipoles = torch.einsum("adc,d->ac", normalized_vector, dipole_weights)
        molecular = output.get("dipole")
        latent_charges = output.get("charges")
        latent_dipoles = output.get("atomic_dipoles")
        if molecular is None or tuple(molecular.shape) != (1, 3):
            raise RuntimeError("MACE-MDP molecular dipole output changed.")
        if (
            latent_charges is None
            or tuple(latent_charges.shape) != (count,)
            or latent_dipoles is None
            or tuple(latent_dipoles.shape) != (count, 3)
        ):
            raise RuntimeError("MACE-MDP latent atomic source output changed.")
        charges, dipoles, schur = _project_source_torch(
            torch=torch,
            positions=positions,
            raw_charges=raw_charges,
            raw_dipoles=raw_dipoles,
            total_charge=charge,
            molecular_dipole=molecular[0],
            charge_sigma=asset.charge_sigma_e,
            dipole_sigma=asset.dipole_sigma_eangstrom,
        )
        raw_source = torch.cat((charges[:, None], dipoles[:, (1, 2, 0)]), dim=1)
        return (
            torch,
            positions,
            raw_charges,
            raw_dipoles,
            charges,
            dipoles,
            molecular[0],
            raw_source,
            schur,
            latent_charges,
            latent_dipoles,
        )

    def evaluate_state(self, atoms: object) -> MACE_MDPMBISSourceState:
        with __import__("torch").no_grad():
            (
                torch,
                positions,
                raw_charges,
                raw_dipoles,
                charges,
                dipoles,
                molecular,
                source,
                schur,
                latent_charges,
                latent_dipoles,
            ) = self._forward(atoms, need_position_gradient=False)

        def array(value: Any) -> np.ndarray:
            return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)

        condition = float(torch.linalg.cond(schur).detach().cpu())
        if not np.isfinite(condition):
            raise RuntimeError("Source projection Schur matrix is singular.")
        return MACE_MDPMBISSourceState(
            configuration_sha256=self.configuration_sha256(),
            model_input_sha256=model_input_sha256(atoms),
            total_charge_e=0.0,
            positions_angstrom=array(positions),
            raw_charges_e=array(raw_charges),
            raw_dipoles_eangstrom=array(raw_dipoles),
            latent_mdp_charges_e=array(latent_charges),
            latent_mdp_dipoles_eangstrom=array(latent_dipoles),
            charges_e=array(charges),
            dipoles_eangstrom=array(dipoles),
            public_molecular_dipole_eangstrom=array(molecular),
            source4_raw_l1=array(source),
            projection_condition_number=condition,
        )

    def evaluate_source(self, geometry: object) -> np.ndarray:
        return self.source_space.validate(
            self.evaluate_state(geometry).source4_raw_l1,
            atom_count=atom_count(geometry),
            name="MACE-MDP MBIS permanent source",
        )

    def source_position_vjp(
        self, geometry: object, source_cotangent: object
    ) -> np.ndarray:
        count = atom_count(geometry)
        cotangent = self.source_space.validate(
            source_cotangent,
            atom_count=count,
            name="MACE-MDP MBIS source cotangent",
        )
        torch, positions, *_, source, _schur, _latent_q, _latent_p = self._forward(
            geometry, need_position_gradient=True
        )
        cotangent_tensor = torch.as_tensor(
            cotangent, dtype=source.dtype, device=source.device
        )
        contraction = torch.sum(source * cotangent_tensor)
        (gradient,) = torch.autograd.grad(
            contraction,
            positions,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        result = np.asarray(gradient.detach().cpu().numpy(), dtype=np.float64)
        if result.shape != (count, 3) or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "MBIS source position VJP must be finite with shape (N,3)."
            )
        return result.copy()


def build_mace_mdp_mbis_source_adapter(
    *,
    checkpoint_path: str | Path,
    source_head_path: str | Path,
    device: str = "cuda",
    expected_checkpoint_sha256: str = MACE_MDP_EXPECTED_CHECKPOINT_SHA256,
    expected_source_head_sha256: str = MACE_MDP_MBIS_SOURCE_HEAD_EXPECTED_SHA256,
) -> MACE_MDPMBISSourceAdapter:
    """Load the frozen MDP backbone and content-addressed MBIS readout."""

    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint_sha = _sha256_file(checkpoint)
    if checkpoint_sha != _digest(
        expected_checkpoint_sha256, name="expected_checkpoint_sha256"
    ):
        raise ValueError("MACE-MDP checkpoint SHA256 does not match.")
    asset = MACE_MDPMBISSourceHeadAsset.from_npz(
        source_head_path, expected_sha256=expected_source_head_sha256
    )
    torch = __import__("torch")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA source-head runtime was requested but is unavailable.")
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
    sources: dict[str, str] = {
        "maple_mace_mdp_mbis_adapter": _sha256_file(Path(__file__))
    }
    for label, value in (
        ("mace_calculator", calculator_type),
        ("mace_model", type(calculator.models[0])),
    ):
        raw = inspect.getsourcefile(value)
        if raw is not None:
            path = Path(raw).resolve()
            if path.is_file():
                sources[label] = _sha256_file(path)
    return MACE_MDPMBISSourceAdapter(
        calculator=calculator,
        asset=asset,
        checkpoint_sha256=checkpoint_sha,
        device=device,
        runtime_identity=(
            ("python", platform.python_version()),
            ("numpy", np.__version__),
            ("torch", str(torch.__version__)),
            ("cuda", str(torch.version.cuda)),
            ("mace_torch", importlib.metadata.version("mace-torch")),
        ),
        runtime_sources=tuple(sources.items()),
    )


__all__ = [
    "MACE_MDP_MBIS_SOURCE_HEAD_CONTRACT",
    "MACE_MDP_MBIS_SOURCE_HEAD_EXPECTED_SHA256",
    "MACE_MDP_MBIS_SOURCE_MODEL_PROFILE_ID",
    "MACE_MDP_MBIS_SOURCE_PROVIDER_ID",
    "MACE_MDPMBISSourceAdapter",
    "MACE_MDPMBISSourceHeadAsset",
    "MACE_MDPMBISSourceState",
    "build_mace_mdp_mbis_source_adapter",
]
