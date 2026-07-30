"""Pinned AniSolv compact scalar additive-solvent correction adapter.

The upstream ``model1_compact`` checkpoint predicts a geometry-level
``E_solv - E_gas`` correction and an autograd force.  MAPLE exposes only the
scalar correction because the exact upstream Euler/Jd path is not
rotation-covariant at pole-aligned geometries.  The exact vacuum output gate is
useful for defining a paired energy, but is not by itself evidence for an
absolute solvation free-energy protocol.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ...backends import SolvationBackend
from ...model_capabilities import SolvationCapabilities, load_model_provenance_card
from .result import SolvationResult

ANISOLV_SOURCE_REVISION = "66f36dc7d4134c78b718255bb2abfed14482acb0"
ANISOLV_COMPACT_CHECKPOINT_SHA256 = (
    "b79be343c55a7cc06579aa4778753156bc71f99b3a00f046a270bc3427c1498f"
)
ANISOLV_COMPACT_REFERENCE_CARD = (
    "maple/function/calculator/model_cards/anisolv-compact.yaml"
)
ANISOLV_MODEL_CARD_ROOT = Path(__file__).resolve().parents[2] / "model_cards"
EV_PER_HARTREE = 27.211386245988


@dataclass(frozen=True)
class AniSolvSolventSpec:
    """One literature-validated compact-checkpoint solvent identifier."""

    upstream_name: str


# The upstream descriptor table contains additional entries, but the project
# documents validation only for these 21 solvents.  Keep the executable
# surface to that released scope rather than turning descriptor availability
# into an unsupported extrapolation claim.
_ANISOLV_VALIDATED_SOLVENTS = {
    "water": "water",
    "acetone": "acetone",
    "acetonitrile": "acetonitrile",
    "aniline": "aniline",
    "benzaldehyde": "benzaldehyde",
    "benzene": "benzene",
    "ch2cl2": "dichloromethane",
    "dichloromethane": "dichloromethane",
    "chcl3": "chloroform",
    "chloroform": "chloroform",
    "cs2": "carbon disulfide",
    "carbon disulfide": "carbon disulfide",
    "dioxane": "1,4-dioxane",
    "1,4-dioxane": "1,4-dioxane",
    "dmf": "N,N-dimethylformamide",
    "n,n-dimethylformamide": "N,N-dimethylformamide",
    "dimethylformamide": "N,N-dimethylformamide",
    "dmso": "dimethyl sulfoxide (DMSO)",
    "dimethyl sulfoxide": "dimethyl sulfoxide (DMSO)",
    "dimethyl sulfoxide (dmso)": "dimethyl sulfoxide (DMSO)",
    "ether": "diethyl ether",
    "diethyl ether": "diethyl ether",
    "ethylacetate": "ethyl ethanoate",
    "ethyl acetate": "ethyl ethanoate",
    "ethyl ethanoate": "ethyl ethanoate",
    "hexadecane": "n-hexadecane",
    "n-hexadecane": "n-hexadecane",
    "hexane": "n-hexane",
    "n-hexane": "n-hexane",
    "methanol": "methanol",
    "nitromethane": "nitromethane",
    "octanol": "1-octanol",
    "1-octanol": "1-octanol",
    "thf": "tetrahydrofuran",
    "tetrahydrofuran": "tetrahydrofuran",
    "toluene": "toluene",
}
SUPPORTED_ANISOLV_SOLVENTS = frozenset(_ANISOLV_VALIDATED_SOLVENTS.values())


def _solvent_token(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def resolve_anisolv_solvent(solvent: str) -> AniSolvSolventSpec:
    """Resolve a user spelling to one official, validated upstream identifier."""

    token = _solvent_token(solvent)
    try:
        return AniSolvSolventSpec(_ANISOLV_VALIDATED_SOLVENTS[token])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported AniSolv compact solvent request: {solvent!r}; "
            "only the 21 solvents explicitly validated by the upstream model card are enabled."
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AniSolvCompactResult:
    """Raw upstream correction in AniSolv's documented eV/Angstrom units."""

    energy_ev: float
    forces_ev_per_angstrom: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


class _UpstreamAniSolvCompactRuntime:
    """Lazy bridge to the official, torch-only ``anisolv`` inference package."""

    def __init__(self, *, model_path: str, device: str) -> None:
        try:
            module = importlib.import_module("anisolv")
        except ImportError as exc:
            raise ImportError(
                "AniSolv compact runtime is unavailable. Install the official "
                "Ant-on-knee/anisolv inference package or inject runtime_factory."
            ) from exc
        predict = getattr(module, "predict_solvation_energy", None)
        if not callable(predict):
            raise ImportError(
                "Installed AniSolv package does not expose predict_solvation_energy."
            )
        self._predict = predict
        self._model_path = model_path
        self._device = device

    def evaluate(
        self,
        atomic_numbers: np.ndarray,
        positions_angstrom: np.ndarray,
        *,
        charge: int,
        multiplicity: int,
        solvent: AniSolvSolventSpec,
        need_forces: bool,
    ) -> AniSolvCompactResult:
        energy_ev, forces_ev_per_angstrom = self._predict(
            (atomic_numbers.tolist(), positions_angstrom.tolist()),
            charge=charge,
            spin=multiplicity,
            solvent=solvent.upstream_name,
            checkpoint=self._model_path,
            device=self._device,
            inference_settings="default",
        )
        return AniSolvCompactResult(
            energy_ev=float(energy_ev),
            forces_ev_per_angstrom=(
                np.asarray(forces_ev_per_angstrom, dtype=float) if need_forces else None
            ),
        )


RuntimeFactory = Callable[..., Any]


@dataclass
class AniSolvCompactBackend(SolvationBackend):
    """Verified public AniSolv compact scalar correction for single points.

    This backend deliberately does not expose vacuum as a selectable solvent:
    its zero correction belongs to the paired gas potential, not a solvent
    state that users may confuse with an absolute free-energy endpoint.  The
    upstream force path is also deliberately disabled because it is not
    rotation-covariant at Euler-chart pole orientations.
    """

    model_path: str
    solvent: str
    device: str = "cpu"
    runtime_factory: RuntimeFactory | None = None
    checkpoint_sha256: str = ANISOLV_COMPACT_CHECKPOINT_SHA256
    execution_task: str | None = None

    capabilities = SolvationCapabilities(
        energy=True,
        forces=False,
        conservative_forces=False,
        hessian="none",
        supports_pbc=False,
        multisolvent=True,
        energy_reference="relative",
        supports_absolute_solvation=False,
    )
    supported_properties = {"energy"}
    conservative_forces = False
    supports_pbc = False
    multisolvent = True
    energy_reference = "relative"
    supports_absolute_solvation = False

    def __post_init__(self) -> None:
        provenance = load_model_provenance_card(
            "anisolv-compact", ANISOLV_MODEL_CARD_ROOT
        )
        if self.execution_task is not None:
            provenance.validate_task(self.execution_task)
        provenance.validate_device(self.device, task=self.execution_task)
        self.solvent_spec = resolve_anisolv_solvent(self.solvent)
        checkpoint = Path(str(self.model_path).strip()).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"AniSolv compact model file not found: {checkpoint}"
            )
        expected = str(self.checkpoint_sha256).strip().lower()
        if expected != ANISOLV_COMPACT_CHECKPOINT_SHA256:
            raise ValueError(
                "AniSolv compact backend is pinned to the official checkpoint SHA256 "
                f"{ANISOLV_COMPACT_CHECKPOINT_SHA256}."
            )
        actual = _sha256(checkpoint)
        if actual != expected:
            raise ValueError(
                f"AniSolv compact checkpoint checksum mismatch: expected {expected}, got {actual}."
            )
        self.model_path = str(checkpoint)
        factory = self.runtime_factory or _UpstreamAniSolvCompactRuntime
        self._runtime = factory(model_path=self.model_path, device=self.device)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "provider": "anisolv-compact",
            "source_revision": ANISOLV_SOURCE_REVISION,
            "checkpoint_sha256": ANISOLV_COMPACT_CHECKPOINT_SHA256,
            "model_path": self.model_path,
            "solvent": self.solvent_spec.upstream_name,
            "units": {"energy": "eV", "forces": "eV/angstrom"},
            "energy_reference": "relative",
            "supports_absolute_solvation": False,
            "task_scope": "single_point_scalar_solution_correction_only",
            "forces_exposed": False,
            "model_card": ANISOLV_COMPACT_REFERENCE_CARD,
        }

    @staticmethod
    def _charge_and_multiplicity(atoms) -> tuple[int, int]:
        info = getattr(atoms, "info", {})
        if not isinstance(info, dict) or "charge" not in info or "mult" not in info:
            raise ValueError(
                "AniSolv compact requires explicit atoms.info['charge'] and atoms.info['mult']; "
                "MAPLE will not infer electronic-state metadata."
            )
        try:
            charge = float(info["charge"])
            multiplicity = float(info["mult"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "AniSolv compact charge and multiplicity must be finite integers."
            ) from exc
        if not np.isfinite(charge) or not charge.is_integer():
            raise ValueError("AniSolv compact requires a finite integer total charge.")
        if (
            not np.isfinite(multiplicity)
            or not multiplicity.is_integer()
            or int(multiplicity) < 1
        ):
            raise ValueError(
                "AniSolv compact requires a positive integer spin multiplicity."
            )
        charge = int(charge)
        multiplicity = int(multiplicity)
        if charge != 0 or multiplicity != 1:
            raise ValueError(
                "AniSolv compact is currently limited to neutral singlet inputs: "
                "the released compact model has no auditable charged/open-shell coverage."
            )
        return charge, multiplicity

    def evaluate(
        self,
        atoms,
        *,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        del calculator
        if need_forces:
            raise NotImplementedError(
                "AniSolv compact forces are disabled: the exact upstream Euler/Jd "
                "path is not rotation-covariant at pole-aligned geometries. "
                "Only scalar single-point correction energy is admissible."
            )
        if bool(np.any(getattr(atoms, "pbc", False))):
            raise NotImplementedError(
                "AniSolv compact is non-periodic; PBC is unsupported."
            )
        atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        if atomic_numbers.ndim != 1 or len(atomic_numbers) != len(atoms):
            raise ValueError("AniSolv compact atomic numbers must have shape (N,).")
        if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
            raise ValueError(
                "AniSolv compact positions must be finite with shape (N, 3)."
            )
        charge, multiplicity = self._charge_and_multiplicity(atoms)
        raw = self._runtime.evaluate(
            atomic_numbers,
            positions,
            charge=charge,
            multiplicity=multiplicity,
            solvent=self.solvent_spec,
            need_forces=False,
        )
        if isinstance(raw, AniSolvCompactResult):
            result = raw
        elif isinstance(raw, Mapping):
            result = AniSolvCompactResult(
                energy_ev=float(raw["energy_ev"]),
                forces_ev_per_angstrom=(
                    None
                    if raw.get("forces_ev_per_angstrom") is None
                    else np.asarray(raw["forces_ev_per_angstrom"], dtype=float)
                ),
                provenance=dict(raw.get("provenance", {})),
            )
        else:
            raise TypeError(
                "AniSolv compact runtime must return AniSolvCompactResult or an eV-unit mapping."
            )
        if not np.isfinite(result.energy_ev):
            raise ValueError("AniSolv compact energy must be finite.")
        authoritative_provenance = {
            **self.provenance,
            "charge": charge,
            "multiplicity": multiplicity,
        }
        runtime_provenance = dict(result.provenance)
        reserved = sorted(runtime_provenance.keys() & authoritative_provenance.keys())
        if reserved:
            raise ValueError(
                "AniSolv compact runtime returned reserved provenance fields: "
                + ", ".join(reserved)
            )
        return SolvationResult(
            energy_hartree=float(result.energy_ev) / EV_PER_HARTREE,
            forces_hartree_per_angstrom=None,
            components_hartree={
                "anisolv_compact_delta": float(result.energy_ev) / EV_PER_HARTREE
            },
            provenance={
                **runtime_provenance,
                **authoritative_provenance,
            },
        )
