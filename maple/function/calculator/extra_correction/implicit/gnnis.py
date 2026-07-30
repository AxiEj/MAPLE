"""Official-reference OpenFF + GNNIS potential adapter.

GNNIS was trained and deployed together with an OpenFF vacuum Hamiltonian.
This module therefore exposes that exact reference composition as one complete
potential backend.  It intentionally does not expose ``arbitrary MLIP +
GNNIS`` because that substitution has not been validated.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ...model_capabilities import ModelCapabilities, load_model_provenance_card
from .topology import CanonicalTopology, canonicalize_topology

GNNIS_SOURCE_REVISION = "5f849bab475570aeaf129ec2e5672f4afbef6bbe"
GNNIS_CHECKPOINT_SHA256 = (
    "304a6cb2e1f804d30dcc4e1b135be1aa768074b4cb268ba26ee34acaa513e5f6"
)
GNNIS_REFERENCE_CARD = "maple/function/calculator/model_cards/gnnis-reference.yaml"
GNNIS_MODEL_CARD_ROOT = Path(__file__).resolve().parents[2] / "model_cards"
KJ_PER_MOL_PER_HARTREE = 2625.4996394799
GNNIS_TOTAL_CHARGE_TOLERANCE = 1.0e-4
GNNIS_PARTIAL_CHARGE_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class GNNISSolventSpec:
    upstream_key: str
    solvent_id: int
    dielectric: float


# Exact 39-entry mapping from upstream Simulation/solvents.yml at
# GNNIS_SOURCE_REVISION.  Keys remain upstream identifiers so the adapter never
# invents a new model condition.
_UPSTREAM_SOLVENT_ROWS = (
    ("tip3p", 0, 78.5),
    ("Chloroform", 1, 4.81),
    ("Methanol", 2, 33.0),
    ("DMSO", 3, 47.24),
    ("DMPU", 4, 36.12),
    ("Diethylether", 5, 4.27),
    ("Ethanol", 6, 25.3),
    ("DMF", 7, 38.25),
    ("DCM", 8, 8.93),
    ("Toluol", 9, 2.38),
    ("Benzol", 10, 2.28),
    ("Hexan", 11, 1.88),
    ("acetonitrile", 12, 36.64),
    ("acetone", 13, 21.01),
    ("aceticacid", 14, 6.20),
    ("14dioxane", 15, 2.22),
    ("nitrobenzol", 16, 35.6),
    ("HMPA", 17, 29.6),
    ("MTBE", 18, 4.5),
    ("IPA", 19, 20.18),
    ("Hexafluorobenzene", 20, 2.03),
    ("pyridine", 21, 13.26),
    ("THF", 22, 7.52),
    ("Ethylacetate", 23, 6.20),
    ("Sulfolane", 24, 43.3),
    ("nitromethane", 25, 37.27),
    ("Butylformate", 26, 6.10),
    ("NMP", 27, 32.55),
    ("Octanol", 28, 10.3),
    ("cyclohexane", 29, 2.024),
    ("glycerin", 30, 46.53),
    ("carbontetrachloride", 31, 2.24),
    ("DME", 32, 7.30),
    ("2Nitropropane", 33, 26.74),
    ("Trifluorotoluene", 34, 9.22),
    ("hexafluroacetone", 35, 2.104),
    ("Propionitrile", 36, 29.7),
    ("Benzonitrile", 37, 25.9),
    ("oxylol", 38, 2.56),
)


def _solvent_token(value: str) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


_SOLVENT_BY_TOKEN = {
    _solvent_token(key): GNNISSolventSpec(key, solvent_id, dielectric)
    for key, solvent_id, dielectric in _UPSTREAM_SOLVENT_ROWS
}
_SOLVENT_ALIASES = {
    "water": "tip3p",
    "toluene": "Toluol",
    "benzene": "Benzol",
    "hexane": "Hexan",
    "dichloromethane": "DCM",
    "diethylether": "Diethylether",
    "ethylacetate": "Ethylacetate",
    "isopropanol": "IPA",
    "14dioxane": "14dioxane",
    "xylene": "oxylol",
}
for alias, upstream_key in _SOLVENT_ALIASES.items():
    _SOLVENT_BY_TOKEN[_solvent_token(alias)] = _SOLVENT_BY_TOKEN[
        _solvent_token(upstream_key)
    ]

SUPPORTED_GNNIS_SOLVENTS = frozenset(
    spec.upstream_key
    for spec in {row.upstream_key: row for row in _SOLVENT_BY_TOKEN.values()}.values()
)


def resolve_gnnis_solvent(solvent: str) -> GNNISSolventSpec:
    token = _solvent_token(solvent)
    if not token or token not in _SOLVENT_BY_TOKEN:
        raise ValueError(
            f"Unsupported GNNIS solvent request: {solvent!r}; "
            "use one of the 39 identifiers pinned from upstream solvents.yml."
        )
    return _SOLVENT_BY_TOKEN[token]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GNNISReferenceResult:
    energy_hartree: float
    forces_hartree_per_angstrom: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


def _canonical_topology_to_rdkit(atoms, topology: CanonicalTopology):
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
    except ImportError as exc:
        raise ImportError(
            "GNNIS reference mode requires the upstream RDKit/OpenFF/OpenMM stack."
        ) from exc

    molecule = Chem.RWMol()
    for index, symbol in enumerate(topology.symbols):
        atom = Chem.Atom(symbol)
        formal_charge = topology.formal_charges[index]
        if formal_charge is not None:
            atom.SetFormalCharge(int(formal_charge))
        molecule.AddAtom(atom)
    for i, j, order in topology.bonds:
        is_aromatic = order == 1.5
        if order == 1.5:
            bond_type = Chem.BondType.AROMATIC
        elif order == 1.0:
            bond_type = Chem.BondType.SINGLE
        elif order == 2.0:
            bond_type = Chem.BondType.DOUBLE
        elif order == 3.0:
            bond_type = Chem.BondType.TRIPLE
        else:
            raise ValueError(
                f"GNNIS RDKit conversion does not support bond order {order!r}."
            )
        molecule.AddBond(i, j, bond_type)
        if is_aromatic:
            molecule.GetAtomWithIdx(i).SetIsAromatic(True)
            molecule.GetAtomWithIdx(j).SetIsAromatic(True)
            molecule.GetBondBetweenAtoms(i, j).SetIsAromatic(True)
    molecule = molecule.GetMol()
    conformer = Chem.Conformer(topology.natoms)
    for index, xyz in enumerate(np.asarray(atoms.get_positions(), dtype=float)):
        conformer.SetAtomPosition(index, Point3D(*map(float, xyz)))
    molecule.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(molecule)
    return molecule


class _UpstreamOpenFFGNNISRuntime:
    """Lazy bridge to the reference implementation's OpenMM System."""

    def __init__(
        self,
        *,
        atoms,
        charges: np.ndarray,
        topology: CanonicalTopology,
        model_path: str,
        solvent: GNNISSolventSpec,
        device: str,
        forcefield: str,
    ) -> None:
        if str(device).strip().lower() != "cpu":
            raise ValueError(
                "GNNIS original runtime is CPU-only until upstream accelerator "
                "selection has frozen no-loss parity evidence."
            )
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "GNNIS reference mode requires the upstream PyTorch runtime."
            ) from exc
        if torch.cuda.is_available():
            raise RuntimeError(
                "GNNIS original runtime is disabled in this process: the pinned "
                "upstream helper auto-selects CUDA when visible, but no frozen "
                "CPU/GPU parity evidence is admitted."
            )
        try:
            helper = importlib.import_module("Simulation.helper_functions")
        except ImportError:
            try:
                helper = importlib.import_module(
                    "GNNImplicitSolvent.Simulation.helper_functions"
                )
            except ImportError as exc:
                raise ImportError(
                    "GNNIS reference runtime is unavailable. Install the official "
                    "rinikerlab/GNNImplicitSolvent environment or inject runtime_factory."
                ) from exc

        molecule = _canonical_topology_to_rdkit(atoms, topology)
        solvent_dict = getattr(helper, "SOLVENT_DICT", None)
        if not isinstance(solvent_dict, Mapping):
            solvent_dict = {
                key: {
                    "solvent_id": solvent_id,
                    "dielectric": dielectric,
                }
                for key, solvent_id, dielectric in _UPSTREAM_SOLVENT_ROWS
            }
        get_gnn_sim = getattr(helper, "get_gnn_sim", None)
        if not callable(get_gnn_sim):
            raise ImportError(
                "Installed GNNIS runtime does not expose Simulation.helper_functions.get_gnn_sim."
            )
        self._simulation = get_gnn_sim(
            mol=molecule,
            solvent=solvent.upstream_key,
            model_path=model_path,
            solvent_dict=solvent_dict,
            partial_charges=charges,
            forcefield=forcefield,
            constraints=None,
            num_confs=1,
        )._simulation
        try:
            platform_name = (
                self._simulation.context.getPlatform().getName().strip().lower()
            )
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "GNNIS could not prove the active OpenMM platform; execution is "
                "blocked rather than assuming CPU precision."
            ) from exc
        if platform_name not in {"cpu", "reference"}:
            raise RuntimeError(
                "GNNIS selected an unverified accelerator OpenMM platform "
                f"{platform_name!r}; only CPU/Reference execution is admitted."
            )
        self.platform_name = platform_name

    def evaluate(self, positions_angstrom, *, need_forces: bool):
        from openmm import unit

        self._simulation.context.setPositions(
            np.asarray(positions_angstrom, dtype=float) * 0.1 * unit.nanometer
        )
        state = self._simulation.context.getState(
            getEnergy=True,
            getForces=need_forces,
        )
        energy_kj_mol = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        forces = None
        if need_forces:
            forces_kj_mol_nm = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            forces = np.asarray(forces_kj_mol_nm, dtype=float) / (
                KJ_PER_MOL_PER_HARTREE * 10.0
            )
        return GNNISReferenceResult(
            energy_hartree=float(energy_kj_mol) / KJ_PER_MOL_PER_HARTREE,
            forces_hartree_per_angstrom=forces,
        )


RuntimeFactory = Callable[..., Any]


@dataclass
class GNNISReferenceBackend:
    """Complete upstream reference Hamiltonian: OpenFF-2.0.0 + GNNIS."""

    atoms: Any
    charges: np.ndarray
    topology: CanonicalTopology
    model_path: str
    solvent: str | None = None
    device: str = "cpu"
    forcefield: str = "openff-2.0.0"
    runtime_factory: RuntimeFactory | None = None
    checkpoint_sha256: str = GNNIS_CHECKPOINT_SHA256
    execution_task: str | None = None

    capabilities = ModelCapabilities(
        energy=True,
        forces=True,
        conservative_forces=True,
        hessian="finite_difference",
        supports_md=True,
        supports_pbc=False,
        supports_charge=False,
        supports_multiplicity=False,
        supports_multifragment=False,
        requires_topology=True,
        requires_partial_charges=True,
        solvation_mode="native",
        energy_reference="unknown",
        supports_absolute_solvation=False,
        supports_alchemical_lambda=False,
    )

    def __post_init__(self) -> None:
        provenance = load_model_provenance_card(
            "gnnis-reference", GNNIS_MODEL_CARD_ROOT
        )
        provenance.validate_device(self.device, task=self.execution_task)
        if self.solvent is None or not str(self.solvent).strip():
            raise ValueError(
                "GNNIS reference mode requires an explicit solvent identifier."
            )
        self.solvent_spec = resolve_gnnis_solvent(self.solvent)
        if not self.model_path or not str(self.model_path).strip():
            raise ValueError("GNNIS reference mode requires an explicit model_path.")
        checkpoint = Path(str(self.model_path).strip()).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"GNNIS model file not found: {checkpoint}")
        actual = _sha256(checkpoint)
        expected = str(self.checkpoint_sha256).strip().lower()
        if expected != GNNIS_CHECKPOINT_SHA256:
            raise ValueError(
                "GNNIS reference backend is pinned to the official checkpoint SHA256 "
                f"{GNNIS_CHECKPOINT_SHA256}."
            )
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise ValueError(
                "GNNIS checkpoint_sha256 must be a 64-hex-character SHA256."
            )
        if actual != expected:
            raise ValueError(
                f"GNNIS checkpoint checksum mismatch: expected {expected}, got {actual}."
            )
        self.model_path = str(checkpoint)

        self.charges = np.asarray(self.charges, dtype=np.float64)
        if self.charges.shape != (len(self.atoms),):
            raise ValueError(
                "GNNIS reference mode requires one partial charge per atom."
            )
        if not np.isfinite(self.charges).all():
            raise ValueError("GNNIS reference charges must be finite numbers.")
        if self.topology is None:
            raise ValueError("GNNIS reference mode requires canonical topology input.")
        if self.topology.natoms != len(self.atoms):
            raise ValueError(
                "GNNIS topology atom count does not match atoms: "
                f"{self.topology.natoms} != {len(self.atoms)}"
            )
        info = getattr(self.atoms, "info", {})
        if not isinstance(info, dict) or "charge" not in info or "mult" not in info:
            raise ValueError(
                "GNNIS reference mode requires explicit atoms.info['charge'] and "
                "atoms.info['mult']; MAPLE will not assume neutral singlet metadata."
            )
        declared_charge_raw = info["charge"]
        multiplicity_raw = info["mult"]
        declared_charge = float(declared_charge_raw)
        if not np.isfinite(declared_charge) or not declared_charge.is_integer():
            raise ValueError(
                "GNNIS reference mode requires a finite integer total charge."
            )
        declared_charge = float(int(declared_charge))
        multiplicity = float(multiplicity_raw)
        if (
            not np.isfinite(multiplicity)
            or not multiplicity.is_integer()
            or int(multiplicity) != 1
        ):
            raise ValueError("GNNIS reference backend supports multiplicity=1 only.")
        partial_charge_sum = float(np.sum(self.charges))
        if abs(partial_charge_sum - declared_charge) > GNNIS_TOTAL_CHARGE_TOLERANCE:
            raise ValueError(
                "GNNIS reference total charge mismatch: partial-charge sum "
                f"does not match declared charge ({partial_charge_sum:.8f} != {declared_charge:.8f})."
            )
        formal_sum = self.topology.formal_charge_sum
        if (
            formal_sum is not None
            and abs(float(formal_sum) - declared_charge) > GNNIS_TOTAL_CHARGE_TOLERANCE
        ):
            raise ValueError(
                "GNNIS reference total charge mismatch: topology formal charges "
                f"do not match declared charge ({formal_sum:.8f} != {declared_charge:.8f})."
            )
        if abs(declared_charge) > GNNIS_TOTAL_CHARGE_TOLERANCE:
            raise ValueError(
                "GNNIS reference backend is neutral-only in the current release; non-zero total charge inputs are rejected."
            )
        self._declared_charge = int(declared_charge)
        self._multiplicity = int(multiplicity)
        self.topology.validate_atoms(self.atoms)

        factory = self.runtime_factory or _UpstreamOpenFFGNNISRuntime
        self._runtime = factory(
            atoms=self.atoms,
            charges=self.charges.copy(),
            topology=self.topology,
            model_path=self.model_path,
            solvent=self.solvent_spec,
            device=self.device,
            forcefield=self.forcefield,
        )

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "provider": "gnnis-reference",
            "reference_hamiltonian": f"{self.forcefield}+GNNIS",
            "source_revision": GNNIS_SOURCE_REVISION,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_path": self.model_path,
            "solvent": self.solvent_spec.upstream_key,
            "solvent_id": self.solvent_spec.solvent_id,
            "solvent_dielectric": self.solvent_spec.dielectric,
            "solvation_mode": "native-reference-composition",
            "energy_reference": "unknown",
            "supports_absolute_solvation": False,
            "model_card": GNNIS_REFERENCE_CARD,
        }

    def evaluate(
        self,
        atoms,
        *,
        need_forces: bool = False,
        need_hessian: bool = False,
    ) -> GNNISReferenceResult:
        self.topology.validate_atoms(atoms)
        info = getattr(atoms, "info", {})
        if not isinstance(info, dict) or "charge" not in info or "mult" not in info:
            raise ValueError(
                "GNNIS runtime requires explicit atoms.info['charge'] and "
                "atoms.info['mult']; metadata may not disappear after initialization."
            )
        charge_raw = info["charge"]
        multiplicity_raw = info["mult"]
        try:
            charge = float(charge_raw)
            multiplicity = float(multiplicity_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "GNNIS runtime charge and multiplicity must be finite integers."
            ) from exc
        if (
            not np.isfinite(charge)
            or not charge.is_integer()
            or int(charge) != self._declared_charge
        ):
            raise ValueError(
                "GNNIS runtime total charge differs from the audited neutral reference input."
            )
        if (
            not np.isfinite(multiplicity)
            or not multiplicity.is_integer()
            or int(multiplicity) != self._multiplicity
        ):
            raise ValueError(
                "GNNIS runtime multiplicity differs from the audited singlet reference input."
            )
        has_array = getattr(atoms, "has", None)
        if callable(has_array) and atoms.has("initial_charges"):
            current_charges = np.asarray(atoms.get_initial_charges(), dtype=np.float64)
            if (
                current_charges.shape != self.charges.shape
                or not np.isfinite(current_charges).all()
                or not np.allclose(
                    current_charges,
                    self.charges,
                    rtol=0.0,
                    atol=GNNIS_PARTIAL_CHARGE_TOLERANCE,
                )
            ):
                raise ValueError(
                    "GNNIS runtime partial charges differ from the audited reference input."
                )
        if bool(np.any(getattr(atoms, "pbc", False))):
            raise NotImplementedError(
                "GNNIS reference mode is non-periodic; periodic cells are unsupported."
            )
        if need_hessian:
            raise NotImplementedError(
                "GNNIS reference Hessian is available only through finite differences of forces."
            )
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
            raise ValueError(
                "GNNIS reference positions must be finite with shape (N, 3)."
            )
        raw = self._runtime.evaluate(
            positions,
            need_forces=need_forces,
        )
        if isinstance(raw, GNNISReferenceResult):
            result = raw
        elif isinstance(raw, Mapping):
            result = GNNISReferenceResult(
                energy_hartree=float(raw["energy_hartree"]),
                forces_hartree_per_angstrom=(
                    None
                    if raw.get("forces_hartree_per_angstrom") is None
                    else np.asarray(raw["forces_hartree_per_angstrom"], dtype=float)
                ),
            )
        else:
            raise TypeError(
                "GNNIS runtime must return GNNISReferenceResult or a Hartree-unit mapping."
            )
        if need_forces and result.forces_hartree_per_angstrom is None:
            raise NotImplementedError("GNNIS reference runtime did not return forces.")
        energy = float(result.energy_hartree)
        if not np.isfinite(energy):
            raise ValueError("GNNIS reference energy must be finite.")
        forces = result.forces_hartree_per_angstrom
        if forces is not None:
            forces = np.asarray(forces, dtype=float)
            if forces.shape != (len(atoms), 3):
                raise ValueError("GNNIS reference forces must have shape (N, 3).")
            if not np.isfinite(forces).all():
                raise ValueError("GNNIS reference forces must be finite.")
        return GNNISReferenceResult(
            energy_hartree=energy,
            forces_hartree_per_angstrom=forces,
            provenance={**self.provenance, **dict(result.provenance)},
        )


def build_gnnis_reference_adapter(*, mode: str = "reference", **kwargs):
    if mode != "reference":
        raise NotImplementedError(
            "Arbitrary MLIP + GNNIS correction is intentionally disabled; "
            "use the verified OpenFF + GNNIS reference Hamiltonian."
        )
    return GNNISReferenceBackend(**kwargs)


def canonicalize_gnnis_topology(topology_source: Any) -> CanonicalTopology:
    return canonicalize_topology(topology_source)
