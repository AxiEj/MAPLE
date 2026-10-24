"""MAPLE adapter for the converted pure-PyTorch ConSolv potential."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.data import covalent_radii

from ..calculator_base import CalcABC, register_calculator
from ..model_capabilities import ModelProvenanceCard, load_model_provenance_card
from ._torch_model import load_consolv_torch_checkpoint

CONSOLV_SOURCE_REVISION = "06801dd5f53182ed86c31eee1fa03cc011c05b5a"
CONSOLV_TORCH_CHECKPOINT_SHA256 = (
    "534114cd0a45d97308eb28506ccfe731df143ae5fb0c6d611b6c6c0cb1656767"
)
MODEL_CARD_ROOT = Path(__file__).resolve().parents[1] / "model_cards"
CONSOLV_CARD = MODEL_CARD_ROOT / "consolv.yaml"
CONSOLV_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "model/consolv/consolv-torch-v1.pt"
)
CONSOLV_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})

# Official ``dataset.choose_task`` IDs resolved through the frozen Solv@TUM order.
CONSOLV_SOLVENTS = {
    "(±)-2-butanol": 2,
    "1,4-dioxane": 3,
    "1-methyl-2-piperidinone": 95,
    "1-octanol": 4,
    "1-propanol": 5,
    "2,2,4-trimethylpentane": 6,
    "2-ethoxyethanol": 63,
    "4-formylmorpholine": 80,
    "5,8,11,14-tetraoxaoctadecane": 97,
    "acetone": 7,
    "acetophenone": 86,
    "benzonitrile": 50,
    "bromobenzene": 8,
    "butanone": 65,
    "carbon tetrachloride": 9,
    "chlorobenzene": 10,
    "chloroform": 11,
    "decan-1-ol": 14,
    "decane": 15,
    "dibutyl ether": 67,
    "dichloroethane": 16,
    "dichloromethane": 17,
    "diethylene glycol dibutyl ether": 103,
    "diiodomethane": 52,
    "dimethylformamide": 53,
    "ethanol": 19,
    "ethyl acetate": 61,
    "ethylbenzene": 69,
    "formamide": 88,
    "heptane": 23,
    "hexadecane": 24,
    "hexane": 25,
    "isobutanol": 28,
    "isopropanol": 29,
    "m-xylene": 30,
    "methanol": 31,
    "methoxyethanol": 71,
    "methyl acetate": 72,
    "n,n-dibutylformamid": 107,
    "n,n-dimethylacetamide": 54,
    "n-butanol": 33,
    "n-ethylacetamide": 109,
    "n-methylacetamide": 34,
    "n-methylformamide": 55,
    "n-methylpyrrolidone": 56,
    "octane": 38,
    "pentanol": 41,
    "phenylamine": 59,
    "pyridine": 74,
    "sulfolane": 78,
    "tert-butanol": 42,
    "tetraethylene glycol dimethyl ether": 43,
    "thf": 44,
    "toluene": 45,
    "tributyl phosphate": 58,
    "propylene carbonate": 57,
    "1-hexadecene": 83,
    "1-chlorobutane": 93,
    "butyl acetate": 66,
    "dodecane": 18,
    "cyclohexanone": 12,
    "pentane": 40,
    "ethylene glycol": 20,
    "undecane": 46,
    "acetic acid": 49,
    "p-xylene": 39,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_card() -> tuple[dict, ModelProvenanceCard]:
    payload = json.loads(CONSOLV_CARD.read_text(encoding="utf-8"))
    provenance = load_model_provenance_card("consolv", MODEL_CARD_ROOT)
    capabilities = provenance.capabilities
    if not (
        capabilities.energy
        and capabilities.forces
        and capabilities.conservative_forces
        and capabilities.hessian == "none"
        and capabilities.solvation_mode == "native"
        and not capabilities.supports_md
        and not capabilities.supports_pbc
        and not capabilities.supports_absolute_solvation
        and not capabilities.supports_alchemical_lambda
    ):
        raise ValueError(
            "ConSolv model card does not preserve the native SP-only contract."
        )
    return payload, provenance


@register_calculator
class ConSolvCalculator(CalcABC):
    """Neutral, closed-shell ConSolv solution potential on CPU/float64."""

    implemented_properties = ["energy", "forces", "free_energy"]
    MODEL_NAMES = ("consolv",)
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ()
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False
    OPTION_KEYS = ("native_solvent", "model_path")
    MODEL_PATH_OPTION = "model_path"
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False

    @classmethod
    def build_kwargs_from_options(
        cls, model, model_options, *, resolved_model_path=None
    ):
        del model
        if model_options.get("d4", False):
            raise ValueError(
                "ConSolv is a complete native potential and rejects D4 composition."
            )
        kwargs = {"native_solvent": model_options.get("native_solvent")}
        if resolved_model_path is not None:
            kwargs["model_path"] = resolved_model_path
        elif "model_path" in model_options:
            kwargs["model_path"] = model_options["model_path"]
        return kwargs

    @property
    def device(self):
        return self._consolv_device

    @property
    def model_path(self):
        return self._consolv_model_path

    @property
    def native_solvent(self):
        return self._consolv_native_solvent

    @property
    def solvent_index(self):
        return self._consolv_solvent_index

    def set(self, **kwargs):
        immutable = {"native_solvent", "solvent_index", "model_path", "device"}
        changed = sorted(immutable.intersection(kwargs))
        if changed:
            raise AttributeError(
                "ConSolv configuration is constructor-immutable: " + ", ".join(changed)
            )
        return super().set(**kwargs)

    def __init__(
        self,
        device,
        model="consolv",
        implicit="none",
        solvent="none",
        native_solvent=None,
        model_path=CONSOLV_DEFAULT_MODEL_PATH,
        execution_task=None,
    ):
        super().__init__()
        if str(model).strip().lower() != "consolv":
            raise ValueError("ConSolv requires model='consolv'.")
        if str(implicit).strip().lower() not in {"", "none"} or str(
            solvent
        ).strip().lower() not in {"", "none"}:
            raise ValueError("ConSolv is a native solution potential; remove #solv.")
        if not native_solvent:
            raise ValueError(
                "ConSolv requires native_solvent=<one of the frozen 66 solvents>."
            )
        normalized = str(native_solvent).strip().casefold()
        if normalized not in CONSOLV_SOLVENTS:
            raise ValueError(
                "Unsupported ConSolv solvent; only the frozen 66 trained solvents are accepted."
            )
        self.model_card, self.model_provenance = _load_card()
        task = execution_task or "sp"
        self.model_provenance.validate_task(task)
        self.model_provenance.validate_device(device, task=task)
        if str(device).strip().lower() not in {"cpu", "none"}:
            raise ValueError("ConSolv is admitted on CPU only.")
        if not isinstance(model_path, (str, os.PathLike)):
            raise TypeError("ConSolv model_path must be a non-empty path string.")
        if not str(model_path).strip():
            raise ValueError("ConSolv model_path must be a non-empty path string.")
        checkpoint = Path(model_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ConSolv Torch checkpoint not found: {checkpoint}")
        actual = _sha256(checkpoint)
        if actual != CONSOLV_TORCH_CHECKPOINT_SHA256:
            raise ValueError(
                "ConSolv Torch checkpoint checksum mismatch: expected "
                f"{CONSOLV_TORCH_CHECKPOINT_SHA256}, got {actual}."
            )
        self._consolv_device = "cpu"
        self._consolv_model_path = checkpoint
        self._consolv_native_solvent = normalized
        self._consolv_solvent_index = CONSOLV_SOLVENTS[normalized]
        self.execution_task = task
        self._model = load_consolv_torch_checkpoint(
            checkpoint, device="cpu", dtype=torch.float64, expected_state="solution"
        )
        if self._model.metadata.get("source_revision") != CONSOLV_SOURCE_REVISION:
            raise ValueError("ConSolv converted checkpoint source revision mismatch.")
        self.capabilities = self.model_provenance.capabilities
        self.solvent_correction = None

    @property
    def provenance(self):
        return {
            "provider": "consolv",
            "source_revision": CONSOLV_SOURCE_REVISION,
            "checkpoint_sha256": CONSOLV_TORCH_CHECKPOINT_SHA256,
            "model_path": str(self.model_path),
            "native_solvent": self.native_solvent,
            "solvent_index": self.solvent_index,
            "runtime": "pure_torch",
            "compute_dtype": "float64",
            "runtime_scope": "native_solution_potential_sp_mechanics",
            "license": self.model_card["license"],
        }

    def _validate_system(self, atoms: Atoms | None) -> Atoms:
        if atoms is None or len(atoms) == 0:
            raise ValueError("ConSolv requires a non-empty ASE Atoms object.")
        if np.asarray(atoms.get_pbc(), dtype=bool).any():
            raise ValueError("ConSolv is non-periodic.")
        positions = np.asarray(atoms.get_positions(), dtype=float)
        if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
            raise ValueError("ConSolv positions must be a finite (N, 3) array.")
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        unsupported = sorted(set(numbers).difference(CONSOLV_ELEMENTS))
        if unsupported:
            raise ValueError(
                f"ConSolv received unsupported atomic numbers: {unsupported}."
            )
        charge = float(atoms.info.get("charge", 0.0))
        multiplicity = float(atoms.info.get("mult", 1.0))
        if charge != 0.0 or multiplicity != 1.0:
            raise ValueError("ConSolv is restricted to neutral closed-shell molecules.")
        if len(atoms) > 1:
            adjacency = [set() for _ in atoms]
            for left in range(len(atoms)):
                for right in range(left + 1, len(atoms)):
                    cutoff = 1.25 * (
                        covalent_radii[numbers[left]] + covalent_radii[numbers[right]]
                    )
                    if np.linalg.norm(positions[left] - positions[right]) <= cutoff:
                        adjacency[left].add(right)
                        adjacency[right].add(left)
            visited, pending = {0}, [0]
            while pending:
                new = adjacency[pending.pop()].difference(visited)
                visited.update(new)
                pending.extend(new)
            if len(visited) != len(atoms):
                raise ValueError("ConSolv requires one geometry-connected molecule.")
        return atoms

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol=tol)
        if self.solvent_correction is not None:
            changes.append("solvent_correction")
        return changes

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        self.results = {}
        if self.solvent_correction is not None:
            raise ValueError("ConSolv is native and rejects any attached correction.")
        properties = tuple(self._normalize_properties(properties))
        unsupported = sorted(
            set(properties).difference({"energy", "free_energy", "forces"})
        )
        if unsupported:
            raise ValueError(
                "ConSolv received unsupported properties: " + ", ".join(unsupported)
            )
        calculated = self._validate_system(
            super().calculate(atoms, properties, system_changes)
        )
        numbers = torch.as_tensor(calculated.get_atomic_numbers(), dtype=torch.long)
        positions = torch.as_tensor(calculated.get_positions(), dtype=torch.float64)
        if "forces" in properties:
            energy, forces = self._model.energy_and_forces_ev(
                numbers, positions, self.solvent_index
            )
            forces_value = forces.detach().cpu().numpy()
        else:
            energy = self._model.energy_ev(numbers, positions, self.solvent_index)
            forces_value = None
        if not torch.isfinite(energy):
            raise ValueError("ConSolv energy must be finite.")
        if forces_value is not None and not np.isfinite(forces_value).all():
            raise ValueError("ConSolv forces must be finite.")
        self._finalize_results(
            calculated, energy=float(energy.detach()), forces=forces_value, unit="eV"
        )
        self.results["model_metadata"] = self.provenance
