from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Dict, Literal

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from .._batch_types import BatchResult
from .._batch_utils import (
    atom_counts,
    atoms_list_has_pbc,
    empty_batch_result,
    normalize_energy_forces_request,
    sequential_calculate_many,
    split_atomwise_array,
)
from ..calculator_base import EV2HARTREE, CalcABC, register_calculator
from ..analytic_protocol import (
    AnalyticIdentityError,
    AnalyticOperationResult,
    CheckpointFileIdentity,
    NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM,
    NEIGHBOR_CUTOFF_GUARD_POLICY,
    ScalarCellIdentity,
    ScalarPointIdentity,
    freeze_identity_options,
    file_stat_identity,
    geometry_sha256,
    named_array_sha256,
    require_no_amp_tf32,
    torch_model_runtime_identity,
    torch_response_settings,
)


_AIMNET2_ANALYTIC_CHECKPOINT_SHA256 = (
    "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
)


@dataclass(frozen=True)
class _AIMNetAnalyticContext:
    data: Dict[str, torch.Tensor]
    atom_count: int
    identity: ScalarPointIdentity


# --------------------------------------------
# Build dense neighbor list (N+1, M) sentinel padded
# --------------------------------------------
def nblist_dense_padded(coord: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Brute-force dense neighbor list for single molecule.
    Returns: (N+1, M) int32 tensor, sentinel row = N
    """
    device = coord.device
    N = coord.shape[0]
    if N == 0:
        return torch.full((1, 1), 0, dtype=torch.int32, device=device)

    diff = coord[:, None, :] - coord[None, :, :]
    dist2 = torch.sum(diff**2, dim=-1)
    dist2[torch.eye(N, dtype=torch.bool, device=device)] = float("inf")
    mask = dist2 <= cutoff**2
    M = max(int(mask.sum(dim=1).max().item()), 1)

    nbmat = torch.full((N + 1, M), N, dtype=torch.int32, device=device)
    for i in range(N):
        nb_i = torch.nonzero(mask[i], as_tuple=False).flatten()
        if nb_i.numel() > 0:
            nbmat[i, : min(nb_i.numel(), M)] = nb_i[: min(nb_i.numel(), M)]
    return nbmat


def nblist_dense_padded_multi(
    coord: torch.Tensor,
    mol_idx: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """Build a dense neighbor list without cross-molecule edges."""
    device = coord.device
    N = coord.shape[0]
    if N == 0:
        return torch.full((1, 1), 0, dtype=torch.int32, device=device)

    diff = coord[:, None, :] - coord[None, :, :]
    dist2 = torch.sum(diff**2, dim=-1)
    same_molecule = mol_idx[:, None] == mol_idx[None, :]
    self_edges = torch.eye(N, dtype=torch.bool, device=device)
    mask = (dist2 <= cutoff**2) & same_molecule & (~self_edges)
    width = max(int(mask.sum(dim=1).max().item()), 1)

    neighbors = torch.full(
        (N + 1, width),
        N,
        dtype=torch.int32,
        device=device,
    )
    for index in range(N):
        row = torch.nonzero(mask[index], as_tuple=False).flatten()
        if row.numel():
            neighbors[index, : row.numel()] = row.to(torch.int32)
    return neighbors


# --------------------------------------------
# Pad helpers
# --------------------------------------------
def pad_dim0(a: torch.Tensor, value=0.0) -> torch.Tensor:
    """
    Pad one row along dim0.
    For (N, C) -> (N+1, C), (N,) -> (N+1,)
    """
    pad_shape = list(a.shape)
    pad_shape[0] = 1
    pad_row = torch.full(pad_shape, value, dtype=a.dtype, device=a.device)
    return torch.cat([a, pad_row], dim=0)


def maybe_pad_dim0(a: torch.Tensor, N: int, value=0.0) -> torch.Tensor:
    """
    If a.shape[0] == N, return as is.
    If a.shape[0] == N-1, pad one row to length N.
    """
    diff = N - a.shape[0]
    assert diff in (0, 1), f"Invalid pad: {a.shape[0]} vs target {N}"
    if diff == 1:
        a = pad_dim0(a, value=value)
    return a


# ==========================================================
# AIMNet2 Calculator (single-molecule minimal version)
# ==========================================================
@register_calculator
class AIMNet2Calculator(CalcABC):
    implemented_properties = ["energy", "forces", "free_energy", "hessian"]

    MODEL_NAMES = ("aimnet2", "aimnet2nse")
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("analytic", "numerical")
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False
    SUPPORTS_IMPLICIT_SOLVATION = True
    SUPPORTED_COULOMB_METHODS = ("simple", "dsf")
    CHECKPOINT_FILENAME = {"aimnet2": "aimnet2.pt", "aimnet2nse": "aimnet2nse.pt"}
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ("coulomb_method",)
    MODEL_PATH_OPTION = "model_path"
    supports_batch_energy_forces = True

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        coulomb_method = options.get("coulomb_method")
        if coulomb_method is not None:
            kwargs["coulomb_method"] = str(coulomb_method).lower()
        if resolved_model_path is not None:
            kwargs["model_path"] = resolved_model_path
        return kwargs

    def __init__(
        self,
        device: torch.device,
        model: str = "aimnet2",
        model_path: str = None,
        coulomb_method: str = "simple",
        implicit: Literal["gbsa", "none"] = "none",
        solvent: str = "none",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.model_name = str(model)

        # Load model
        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            model_path = os.path.join(model_dir, "model", f"{model}.pt")
        self._checkpoint_path = os.path.realpath(model_path)
        loaded_checkpoint_stat = file_stat_identity(
            os.stat(self._checkpoint_path)
        )
        self.model = torch.jit.load(
            self._checkpoint_path,
            map_location=self.device,
        ).eval()
        self._checkpoint_identity = CheckpointFileIdentity.bind(
            self._checkpoint_path,
            expected_stat_identity=loaded_checkpoint_stat,
        )
        self._checkpoint_sha256 = self._checkpoint_identity.sha256
        self._loaded_model_object_id = id(self.model)

        implemented_species = getattr(self.model, "implemented_species", None)
        if implemented_species is None:
            # The pinned upstream TorchScript checkpoint preserves this typo.
            implemented_species = getattr(self.model, "impemented_species", None)
        if implemented_species is None:
            raise RuntimeError(
                f"AIMNet2 checkpoint '{model}' does not expose its implemented species."
            )
        self.atomic_numbers = sorted(
            {int(value) for value in implemented_species.detach().cpu().tolist()}
        )
        if not self.atomic_numbers:
            raise RuntimeError(
                f"AIMNet2 checkpoint '{model}' exposes an empty element domain."
            )

        self.cutoff = float(getattr(self.model, "cutoff"))
        self.cutoff_lr = float(getattr(self.model, "cutoff_lr", float("inf")))
        self._checkpoint_declared_cutoff = self.cutoff
        self._checkpoint_declared_cutoff_lr = self.cutoff_lr
        # Always provide nbmat_lr to avoid TorchScript KeyError; method routing
        # handles cutoff_lr.
        self.lr = True
        self.hessian: str = "analytic"

        self._set_lrcoulomb_method(coulomb_method)
        self._analytic_bound_cutoff = self.cutoff
        self._analytic_bound_cutoff_lr = self.cutoff_lr
        self._analytic_bound_lrcoulomb_methods = tuple(
            (name, str(getattr(module, "method", "")))
            for name, module in self.model.named_modules()
            if name == "lrcoulomb" or name.endswith(".lrcoulomb")
        )
        self._analytic_model_runtime_identity = torch_model_runtime_identity(
            self.model
        )

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _set_lrcoulomb_method(
        self, method: str, cutoff: float = 15.0, dsf_alpha: float = 0.2
    ):
        """
        Configure the long-range Coulomb interaction method if the model contains a 'lrcoulomb' submodule.
        method: 'simple' or 'dsf'. The historical 'ewald' selector is rejected
        until this wrapper carries validated cell/PBC/MIC inputs.
        cutoff: cutoff distance for long-range interactions
        dsf_alpha: DSF damping parameter (if used)
        """
        method = str(method).lower()
        if method == "ewald":
            raise NotImplementedError(
                "AIMNet2 coulomb_method='ewald' requires validated PBC/cell/MIC support; "
                "use 'simple' or 'dsf'."
            )
        if method not in self.SUPPORTED_COULOMB_METHODS:
            raise ValueError(
                f"Invalid coulomb_method: {method!r}; expected one of 'simple', 'dsf'."
            )

        def _iter_lrcoulomb_mods(model):
            for name, mod in model.named_modules():
                if name == "lrcoulomb":
                    yield mod

        for mod in _iter_lrcoulomb_mods(self.model):
            mod.method = method
            if method == "dsf" and hasattr(mod, "dsf_alpha"):
                mod.dsf_alpha = dsf_alpha

        self.cutoff_lr = float("inf") if method == "simple" else float(cutoff)
        self._coulomb_method = method

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        if (
            ("hessian" in properties or self.analytic_composition_admission is not None)
            and self.model_name == "aimnet2"
            and self._checkpoint_sha256 == _AIMNET2_ANALYTIC_CHECKPOINT_SHA256
            and self._coulomb_method == "simple"
        ):
            gas_energy_forces = self._analytic_energy_forces_operation(
                atoms,
                need_forces="forces" in properties,
            )
            gas_hessian = (
                self._analytic_hessian_operation(atoms)
                if "hessian" in properties
                else None
            )
            self._finalize_results(
                atoms,
                gas_energy_forces=gas_energy_forces,
                gas_hessian=gas_hessian,
            )
            return

        needs_grad = "forces" in properties or "hessian" in properties
        coord = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=needs_grad,
        )
        data = self._build_data(coord, atoms)

        # Pure model energy in eV; _finalize_results handles eV→Ha + solvent.
        energy_eV = self._forward_energy(data)

        if "forces" in properties:
            grad_full = torch.autograd.grad(
                energy_eV, data["coord"], create_graph=("hessian" in properties)
            )[0]
            forces_eV = -grad_full[: coord.shape[0]]
            forces_np = forces_eV.detach().cpu().numpy()
        else:
            forces_np = None

        hessian = None
        if "hessian" in properties:
            if self.solvent_correction is not None:
                raise NotImplementedError(
                    "Hessian calculation with implicit solvent is not implemented yet."
                )
            hessian = self.get_hessian(atoms)

        self._finalize_results(
            atoms, energy=energy_eV.item(), forces=forces_np, hessian=hessian
        )

    def calculate_many(
        self,
        atoms_list,
        properties=("energy", "forces"),
    ) -> BatchResult:
        """Evaluate variable-size molecules in one AIMNet2 ``mol_idx`` batch."""
        _, want_energy, want_forces, request = normalize_energy_forces_request(
            properties
        )
        if not request:
            return BatchResult()

        atoms_list = list(atoms_list)
        if not atoms_list:
            return empty_batch_result(want_energy, want_forces)
        if atoms_list_has_pbc(atoms_list) or self.solvent_correction is not None:
            return sequential_calculate_many(
                self,
                atoms_list,
                request,
                want_energy,
                want_forces,
            )

        counts = atom_counts(atoms_list)
        batch_size = len(atoms_list)
        coordinates = np.concatenate(
            [atoms.get_positions() for atoms in atoms_list],
            axis=0,
        )
        atomic_numbers = np.concatenate(
            [atoms.get_atomic_numbers() for atoms in atoms_list],
            axis=0,
        )
        molecule_indices = np.concatenate(
            [
                np.full(len(atoms), index, dtype=np.int32)
                for index, atoms in enumerate(atoms_list)
            ],
            axis=0,
        )

        coord = torch.tensor(
            coordinates,
            dtype=self.dtype,
            device=self.device,
            requires_grad=want_forces,
        )
        numbers = torch.tensor(
            atomic_numbers,
            dtype=torch.int32,
            device=self.device,
        )
        mol_idx = torch.tensor(
            molecule_indices,
            dtype=torch.int32,
            device=self.device,
        )
        charges = [float(atoms.info.get("charge", 0.0)) for atoms in atoms_list]
        multiplicities = [float(atoms.info.get("mult", 1.0)) for atoms in atoms_list]
        long_range_cutoff = (
            self.cutoff_lr if np.isfinite(self.cutoff_lr) else self.cutoff
        )

        data: Dict[str, torch.Tensor] = {
            "coord": pad_dim0(coord, value=0.0),
            "numbers": pad_dim0(numbers, value=0),
            "charge": torch.tensor(
                charges + [0.0],
                dtype=self.dtype,
                device=self.device,
            ),
            "mult": torch.tensor(
                multiplicities + [1.0],
                dtype=self.dtype,
                device=self.device,
            ),
            "mol_idx": pad_dim0(mol_idx, value=batch_size),
            "nbmat": nblist_dense_padded_multi(
                coord,
                mol_idx,
                self.cutoff,
            ),
            "nbmat_lr": nblist_dense_padded_multi(
                coord,
                mol_idx,
                long_range_cutoff,
            ),
            "cutoff_lr": torch.tensor(
                long_range_cutoff,
                device=self.device,
            ),
        }

        with torch.jit.optimized_execution(False):
            if want_forces:
                output = self.model(data)
            else:
                with torch.no_grad():
                    output = self.model(data)
        energy_vector_eV = self._energy_vector_from_output(
            output["energy"],
            batch_size,
        )
        energies = (
            energy_vector_eV.detach().cpu().numpy().astype(np.float64) * EV2HARTREE
            if want_energy
            else None
        )

        forces_list = None
        if want_forces:
            forces_eV = -torch.autograd.grad(
                energy_vector_eV.sum(),
                data["coord"],
            )[
                0
            ][: coord.shape[0]]
            forces_hartree = (
                forces_eV.detach().cpu().numpy().astype(np.float64) * EV2HARTREE
            )
            forces_list = split_atomwise_array(forces_hartree, counts)

        return BatchResult(energies=energies, forces=forces_list)

    @staticmethod
    def _energy_vector_from_output(
        energy_output: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Return the documented padded per-molecule AIMNet2 energies."""
        energy_vector = energy_output.reshape(-1)
        expected_size = batch_size + 1
        if energy_vector.numel() != expected_size:
            raise RuntimeError(
                "AIMNet2 batch output must contain one energy per molecule "
                f"plus the padded molecule ({expected_size} values); got "
                f"shape {tuple(energy_output.shape)}."
            )
        return energy_vector[:-1]

    def _build_data(self, coord: torch.Tensor, atoms) -> Dict[str, torch.Tensor]:
        Z = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device
        )
        mol_idx = torch.zeros(coord.shape[0], dtype=torch.int32, device=self.device)
        N = coord.shape[0]
        charge_val = float(atoms.info.get("charge", 0.0))
        mult_val = float(atoms.info.get("mult", 1.0))

        nbmat = nblist_dense_padded(coord, self.cutoff)
        data: Dict[str, torch.Tensor] = {
            "coord": pad_dim0(coord, value=0.0),
            "numbers": pad_dim0(Z, value=0),
            "charge": torch.tensor([charge_val], dtype=self.dtype, device=self.device),
            "mult": torch.tensor([mult_val], dtype=self.dtype, device=self.device),
            "mol_idx": pad_dim0(mol_idx, value=mol_idx[-1].item() if N > 0 else 0),
            "nbmat": nbmat,
        }

        lr_cutoff = self.cutoff_lr if np.isfinite(self.cutoff_lr) else self.cutoff
        data["nbmat_lr"] = nblist_dense_padded(coord, lr_cutoff)
        data["cutoff_lr"] = torch.tensor(lr_cutoff, device=self.device)
        return data

    def _forward_energy(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pure model forward; returns energy in eV (model's native unit)."""
        with torch.jit.optimized_execution(False):
            out = self.model(data)
        return out["energy"].sum()

    def prepare_analytic_derivatives(self) -> None:
        """Retain native precision; exact qualification is factory-owned."""
        self._checkpoint_identity.require_unchanged()
        require_no_amp_tf32(self.device)

    def _require_analytic_candidate(self, atoms) -> None:
        """Fail closed outside the one source-bound AIMNet2 candidate cell."""

        self._checkpoint_identity.require_unchanged()
        if id(self.model) != self._loaded_model_object_id:
            raise AnalyticIdentityError(
                "The AIMNet2 model object changed after checkpoint binding."
            )
        if (
            torch_model_runtime_identity(self.model)
            != self._analytic_model_runtime_identity
        ):
            raise AnalyticIdentityError(
                "The AIMNet2 live parameter/buffer identity changed after binding."
            )
        if self.model_name != "aimnet2":
            raise AnalyticIdentityError(
                "Only the separately measured AIMNet2 checkpoint is an analytic "
                "candidate; AIMNet2-NSE requires its own evidence."
            )
        if self._checkpoint_sha256 != _AIMNET2_ANALYTIC_CHECKPOINT_SHA256:
            raise AnalyticIdentityError(
                "The AIMNet2 checkpoint SHA256 is not the frozen analytic candidate."
            )
        if self._coulomb_method != "simple":
            raise AnalyticIdentityError(
                "Only AIMNet2 coulomb_method='simple' is in the measured analytic "
                "candidate cell."
            )
        if self.cutoff != self._analytic_bound_cutoff or (
            math.isfinite(self.cutoff_lr)
            != math.isfinite(self._analytic_bound_cutoff_lr)
        ) or (
            math.isfinite(self.cutoff_lr)
            and self.cutoff_lr != self._analytic_bound_cutoff_lr
        ):
            raise AnalyticIdentityError(
                "The AIMNet2 short/long-range cutoff configuration changed after "
                "checkpoint binding."
            )
        lrcoulomb_methods = tuple(
            (name, str(getattr(module, "method", "")))
            for name, module in self.model.named_modules()
            if name == "lrcoulomb" or name.endswith(".lrcoulomb")
        )
        if (
            not lrcoulomb_methods
            or lrcoulomb_methods != self._analytic_bound_lrcoulomb_methods
            or any(method != "simple" for _name, method in lrcoulomb_methods)
        ):
            raise AnalyticIdentityError(
                "The AIMNet2 live long-range Coulomb module configuration drifted."
            )
        if self.dtype is not torch.float32:
            raise AnalyticIdentityError(
                "The AIMNet2 analytic candidate must retain native torch.float32."
            )
        charge = float(atoms.info.get("charge", 0.0))
        multiplicity = float(atoms.info.get("mult", 1.0))
        if charge != 0.0 or multiplicity != 1.0:
            raise AnalyticIdentityError(
                "The measured AIMNet2 analytic candidate is neutral singlet only."
            )
        require_no_amp_tf32(self.device)

    def _analytic_cell_identity(self) -> ScalarCellIdentity:
        floating_dtypes = sorted(
            {
                str(tensor.dtype)
                for tensor in list(self.model.parameters())
                + list(self.model.buffers())
                if tensor.is_floating_point()
            }
        )
        return ScalarCellIdentity(
            component="gas",
            implementation="maple-aimnet2-torchscript-v1",
            model=self.model_name,
            checkpoint_path=self._checkpoint_path,
            checkpoint_sha256=self._checkpoint_sha256,
            device=str(self.device),
            effective_dtype=str(self.dtype),
            d4=False,
            source_energy_unit="eV",
            cutoff_guard_policy=NEIGHBOR_CUTOFF_GUARD_POLICY,
            cutoff_guard_margin_angstrom=(
                NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM
            ),
            options=freeze_identity_options(
                {
                    "coulomb_method": self._coulomb_method,
                    "checkpoint_declared_long_range_cutoff": (
                        "infinity"
                        if not math.isfinite(self._checkpoint_declared_cutoff_lr)
                        else self._checkpoint_declared_cutoff_lr
                    ),
                    "effective_long_range_neighbor_cutoff": (
                        self.cutoff_lr
                        if math.isfinite(self.cutoff_lr)
                        else self.cutoff
                    ),
                    "long_range_neighbor_semantics": (
                        "existing-wrapper-infinite-falls-back-to-short-range"
                    ),
                    "lrcoulomb_modules": self._analytic_bound_lrcoulomb_methods,
                    "model_floating_dtypes": floating_dtypes,
                    "neighbor_builder": "aimnet-dense-directed-padded-v1",
                    "pbc": False,
                    "torch_jit_optimized_execution": False,
                }
            ),
        )

    def _analytic_response_settings(self):
        return torch_response_settings(self.device)

    @staticmethod
    def _minimum_cutoff_gap(
        coord: torch.Tensor,
        cutoffs: tuple[tuple[str, float], ...],
        *,
        enforce_guard: bool,
    ) -> float | None:
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise AnalyticIdentityError(
                "AIMNet2 analytic coordinates must have shape (N, 3)."
            )
        if not torch.isfinite(coord).all():
            raise AnalyticIdentityError(
                "AIMNet2 analytic coordinates must be finite."
            )
        atom_count = int(coord.shape[0])
        if atom_count < 2:
            return float("inf")
        pair_indices = torch.triu_indices(
            atom_count,
            atom_count,
            offset=1,
            device=coord.device,
        )
        pair_distances = torch.linalg.vector_norm(
            (coord[pair_indices[0]] - coord[pair_indices[1]]).detach(),
            dim=1,
        )
        if not torch.isfinite(pair_distances).all():
            raise AnalyticIdentityError(
                "AIMNet2 analytic pair distances must be finite."
            )
        if enforce_guard and torch.any(pair_distances == 0):
            raise AnalyticIdentityError(
                f"{NEIGHBOR_CUTOFF_GUARD_POLICY} rejects coincident atom pairs."
            )

        finite_cutoffs = [
            float(cutoff)
            for _name, cutoff in cutoffs
            if math.isfinite(float(cutoff))
        ]
        if not finite_cutoffs:
            return None
        if any(cutoff <= 0.0 for cutoff in finite_cutoffs):
            raise AnalyticIdentityError(
                "AIMNet2 analytic neighbor cutoffs must be positive."
            )
        gap_tensors = [
            torch.abs(pair_distances - cutoff) for cutoff in finite_cutoffs
        ]
        minimum_gap = float(
            torch.min(torch.stack([gaps.min() for gaps in gap_tensors]))
            .detach()
            .cpu()
        )
        if enforce_guard and minimum_gap <= (
            NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM + 1.0e-12
        ):
            raise AnalyticIdentityError(
                f"{NEIGHBOR_CUTOFF_GUARD_POLICY} rejects a geometry with "
                f"minimum |r_ij-r_cut|={minimum_gap:.17g} Angstrom at or below "
                f"the frozen {NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM:.17g} "
                "Angstrom margin."
            )
        return minimum_gap

    def _build_analytic_context(
        self,
        atoms,
        *,
        enforce_cutoff_guard: bool,
    ) -> _AIMNetAnalyticContext:
        self._require_analytic_candidate(atoms)
        coord = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        data = self._build_data(coord, atoms)
        effective_lr_cutoff = (
            self.cutoff_lr if math.isfinite(self.cutoff_lr) else self.cutoff
        )
        active_cutoffs = tuple(
            sorted(
                (
                    ("short_range", float(self.cutoff)),
                    ("long_range", float(effective_lr_cutoff)),
                )
            )
        )
        minimum_gap = self._minimum_cutoff_gap(
            coord,
            active_cutoffs,
            enforce_guard=enforce_cutoff_guard,
        )
        graph_sha256 = named_array_sha256(
            {
                "nbmat": data["nbmat"].detach().cpu().numpy(),
                "nbmat_lr": data["nbmat_lr"].detach().cpu().numpy(),
            }
        )
        identity = ScalarPointIdentity(
            cell=self._analytic_cell_identity(),
            geometry_sha256=geometry_sha256(atoms),
            atomic_numbers=tuple(
                int(value) for value in atoms.get_atomic_numbers()
            ),
            charge=float(atoms.info.get("charge", 0.0)),
            multiplicity=int(atoms.info.get("mult", 1)),
            atom_topology_sha256=None,
            neighbor_graph_sha256=graph_sha256,
            active_cutoffs_angstrom=active_cutoffs,
            minimum_cutoff_gap_angstrom=minimum_gap,
            response_settings=self._analytic_response_settings(),
        )
        return _AIMNetAnalyticContext(
            data=data,
            atom_count=len(atoms),
            identity=identity,
        )

    def _analytic_energy_forces_operation(
        self,
        atoms,
        *,
        need_forces: bool,
    ) -> AnalyticOperationResult:
        with torch.enable_grad():
            context = self._build_analytic_context(
                atoms,
                enforce_cutoff_guard=False,
            )
            energy_eV = self._forward_energy(context.data)
            forces = None
            if need_forces:
                gradient_eV = torch.autograd.grad(
                    energy_eV,
                    context.data["coord"],
                )[0][: context.atom_count]
                forces = (
                    -gradient_eV * EV2HARTREE
                ).detach().cpu().numpy()
            return AnalyticOperationResult.energy_forces(
                identity=context.identity,
                energy_hartree=(
                    float(energy_eV.detach().cpu().item()) * EV2HARTREE
                ),
                forces_hartree_per_angstrom=forces,
            )

    def _analytic_hessian_operation(self, atoms) -> AnalyticOperationResult:
        with torch.enable_grad():
            context = self._build_analytic_context(
                atoms,
                enforce_cutoff_guard=True,
            )
            energy_hartree = self._forward_energy(context.data) * EV2HARTREE
            gradient = torch.autograd.grad(
                energy_hartree,
                context.data["coord"],
                create_graph=True,
            )[0][: context.atom_count]
            rows = [
                torch.autograd.grad(
                    component,
                    context.data["coord"],
                    retain_graph=True,
                )[0][: context.atom_count].reshape(-1)
                for component in gradient.reshape(-1).unbind()
            ]
            hessian = torch.stack(rows).detach().cpu().numpy()
        return AnalyticOperationResult.hessian(
            identity=context.identity,
            hessian_hartree_per_angstrom2=hessian,
        )

    def _analytic_hvp_operation(
        self,
        atoms,
        direction,
    ) -> AnalyticOperationResult:
        direction_array = np.asarray(direction, dtype=np.float64).reshape(-1)
        expected_size = 3 * len(atoms)
        if direction_array.shape != (expected_size,) or not np.isfinite(
            direction_array
        ).all():
            raise ValueError(
                f"AIMNet2 HVP direction must be a finite ({expected_size},) array."
            )
        with torch.enable_grad():
            context = self._build_analytic_context(
                atoms,
                enforce_cutoff_guard=True,
            )
            energy_hartree = self._forward_energy(context.data) * EV2HARTREE
            gradient = torch.autograd.grad(
                energy_hartree,
                context.data["coord"],
                create_graph=True,
            )[0][: context.atom_count]
            direction_tensor = torch.as_tensor(
                direction_array,
                dtype=self.dtype,
                device=self.device,
            ).reshape(context.atom_count, 3)
            hvp = torch.autograd.grad(
                torch.sum(gradient * direction_tensor),
                context.data["coord"],
            )[0][: context.atom_count]
            forces = -gradient
        return AnalyticOperationResult.hvp(
            identity=context.identity,
            direction=direction_array,
            hvp_hartree_per_angstrom2=(
                hvp.detach().cpu().numpy().reshape(-1)
            ),
            forces_hartree_per_angstrom=forces.detach().cpu().numpy(),
            energy_hartree=float(energy_hartree.detach().cpu().item()),
        )

    def _analytic_hessian(self, atoms) -> np.ndarray:
        """Analytic Hessian via autograd. Returns (3N, 3N) np.ndarray in Hartree/Å²."""
        if (
            self.model_name == "aimnet2"
            and self._checkpoint_sha256 == _AIMNET2_ANALYTIC_CHECKPOINT_SHA256
            and self._coulomb_method == "simple"
            and float(atoms.info.get("charge", 0.0)) == 0.0
            and float(atoms.info.get("mult", 1.0)) == 1.0
        ):
            return np.asarray(
                self._analytic_hessian_operation(
                    atoms
                ).hessian_hartree_per_angstrom2
            )
        coord = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        N = coord.shape[0]
        data = self._build_data(coord, atoms)

        energy = self._forward_energy(data) * EV2HARTREE

        forces_full = torch.autograd.grad(energy, data["coord"], create_graph=True)[0]
        forces = -forces_full[:N]

        hessian = -torch.stack(
            [
                torch.autograd.grad(f, data["coord"], retain_graph=True)[0]
                for f in forces.flatten().unbind()
            ]
        ).view(-1, 3, N + 1, 3)[:, :, :N, :]

        return hessian.detach().cpu().numpy().reshape(3 * N, 3 * N)
