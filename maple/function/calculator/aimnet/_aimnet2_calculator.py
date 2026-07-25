from __future__ import annotations

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
        self.device = device
        self.dtype = torch.float32

        # Load model
        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            model_path = os.path.join(model_dir, "model", f"{model}.pt")
        self.model = torch.jit.load(model_path, map_location=device).eval()

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
        # Always provide nbmat_lr to avoid TorchScript KeyError; method routing
        # handles cutoff_lr.
        self.lr = True
        self.hessian: str = "analytic"

        self._set_lrcoulomb_method(coulomb_method)

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

    def _analytic_hessian(self, atoms) -> np.ndarray:
        """Analytic Hessian via autograd. Returns (3N, 3N) np.ndarray in Hartree/Å²."""
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
