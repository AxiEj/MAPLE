import os
import torch
import numpy as np
from typing import Dict, Literal
from ase.calculators.calculator import Calculator, all_changes
from ..calculator_base import CalcABC
from .._batch_types import BatchResult
from .._batch_utils import (
    atom_counts,
    atoms_list_has_pbc,
    empty_batch_result,
    normalize_energy_forces_request,
    sequential_calculate_many,
    split_atomwise_array,
)
from .._autograd_hessian import hessian_batched_vjp, hessian_loop


EV2HARTREE = 1.0 / 27.211386245988

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
    dist2 = torch.sum(diff ** 2, dim=-1)
    dist2[torch.eye(N, dtype=torch.bool, device=device)] = float('inf')
    mask = dist2 <= cutoff ** 2
    M = max(int(mask.sum(dim=1).max().item()), 1)

    nbmat = torch.full((N + 1, M), N, dtype=torch.int32, device=device)
    for i in range(N):
        nb_i = torch.nonzero(mask[i], as_tuple=False).flatten()
        if nb_i.numel() > 0:
            nbmat[i, :min(nb_i.numel(), M)] = nb_i[:min(nb_i.numel(), M)]
    return nbmat


def nblist_dense_padded_multi(coord: torch.Tensor, mol_idx: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Dense sentinel-padded neighbor list for concatenated molecules."""
    device = coord.device
    N = coord.shape[0]
    if N == 0:
        return torch.full((1, 1), 0, dtype=torch.int32, device=device)

    diff = coord[:, None, :] - coord[None, :, :]
    dist2 = torch.sum(diff ** 2, dim=-1)
    same = mol_idx[:, None] == mol_idx[None, :]
    eye = torch.eye(N, dtype=torch.bool, device=device)
    mask = (dist2 <= cutoff ** 2) & same & (~eye)
    M = max(int(mask.sum(dim=1).max().item()), 1)

    nbmat = torch.full((N + 1, M), N, dtype=torch.int32, device=device)
    for i in range(N):
        nb_i = torch.nonzero(mask[i], as_tuple=False).flatten()
        if nb_i.numel() > 0:
            nbmat[i, :min(nb_i.numel(), M)] = nb_i[:min(nb_i.numel(), M)].to(torch.int32)
    return nbmat

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
class AIMNet2Calculator(CalcABC):
    implemented_properties = ["energy", "forces", "hessian", "free_energy"]
    supported_hessian_modes = ("analytic", "numerical")
    supports_batch_energy_forces = True
    supports_analytic_hessian = True
    batch_memory_model = "concat_dense_neighbor"
    auto_batch_hard_cap = 8

    def __init__(self, device: torch.device, 
                model: str = "aimnet2", 
                coulomb_method: str = "simple",
                implicit: Literal["gbsa", "none"] = "gbsa",
                solvent: str = 'none',
                ):
        super().__init__()
        self.device = device

        # Load model
        model_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.dirname(model_dir)
        model_path = os.path.join(model_dir, "model", f"{model}.pt")
        self.model = torch.jit.load(model_path, map_location=device).eval()

        self.cutoff = float(getattr(self.model, "cutoff"))
        # CHANGED: do not rely on hasattr(model, 'cutoff_lr') to decide LR; model may still need nbmat_lr
        self.cutoff_lr = float(getattr(self.model, "cutoff_lr", float("inf")))
        # keep a flag (optional); but we will always provide nbmat_lr anyway
        self.lr = True  # CHANGED: force-true to avoid conditional omission
        self.hessian: str = 'analytic'  # 'analytic' or 'numerical'

        # Coulomb settings (keep original behavior)
        self._set_lrcoulomb_method(coulomb_method)

        # Initialize implicit solvent
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _set_lrcoulomb_method(self, method: str, cutoff: float = 15.0, dsf_alpha: float = 0.2):
            """
            Configure the long-range Coulomb interaction method if the model contains a 'lrcoulomb' submodule.
            method: 'simple', 'dsf', or 'ewald'
            cutoff: cutoff distance for long-range interactions
            dsf_alpha: DSF damping parameter (if used)
            """
            assert method in ("simple", "dsf", "ewald"), f"Invalid method: {method}"

            # recursively look for 'lrcoulomb' submodules
            def _iter_lrcoulomb_mods(model):
                for name, mod in model.named_modules():
                    if name == "lrcoulomb":
                        yield mod

            for mod in _iter_lrcoulomb_mods(self.model):
                mod.method = method
                if method == "dsf" and hasattr(mod, "dsf_alpha"):
                    mod.dsf_alpha = dsf_alpha

            # update cutoff_lr based on the chosen method
            self.cutoff_lr = float("inf") if method == "simple" else float(cutoff)
            self._coulomb_method = method

    # ------------------------ calculate ------------------------
    def calculate(self, atoms=None, properties=["energy", "forces", "free_energy", "hessian"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        coord = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=("forces" in properties or "hessian" in properties)
        )
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device)
        mol_idx = torch.zeros(coord.shape[0], dtype=torch.int32, device=self.device)

        N = coord.shape[0]

        charge_val = float(self.atoms.info.get("charge", 0.0))
        mult_val = float(self.atoms.info.get("mult", 1.0))

        nbmat = nblist_dense_padded(coord, self.cutoff)
        data: Dict[str, torch.Tensor] = {
            "coord": pad_dim0(coord, value=0.0),         # (N+1, 3)
            "numbers": pad_dim0(Z, value=0),             # (N+1,)
            "charge": torch.tensor([charge_val], dtype=torch.float32, device=self.device),
            "mult": torch.tensor([mult_val], dtype=torch.float32, device=self.device),
            "mol_idx": pad_dim0(mol_idx, value=mol_idx[-1].item() if N > 0 else 0),
            "nbmat": nbmat,
        }

        # CHANGED: ALWAYS provide nbmat_lr + cutoff_lr to avoid KeyError inside TorchScript
        lr_cutoff = self.cutoff_lr if np.isfinite(self.cutoff_lr) else self.cutoff
        data["nbmat_lr"] = nblist_dense_padded(coord, lr_cutoff)
        data["cutoff_lr"] = torch.tensor(lr_cutoff, device=self.device)

        energy = self.get_energy(data)
        energy = energy * EV2HARTREE  # to Hartree

        if self.solvent_correction:
            solvent_energy = self.implicit_solv_energy(atoms)
            energy += solvent_energy

        self.results["energy"] = float(energy.item())
        self.results["free_energy"] = float(energy.item())

        if "forces" in properties:
            grad_full = torch.autograd.grad(
                energy, data["coord"], create_graph=("hessian" in properties)
            )[0]                      # (N+1, 3)
            forces = -grad_full[:N]   # (N, 3)
            if self.solvent_correction:
                solvent_energy, solvent_force = self.implicit_solv_energy_and_force(atoms)
                forces += solvent_force

            self.results["forces"] = forces.detach().cpu().numpy()
            
        if "hessian" in properties:
            if self.solvent_correction:
                raise NotImplementedError("Hessian calculation with implicit solvent is not implemented yet.")
            self.results["hessian"] = self.get_hessian(atoms)

    def calculate_many(self, atoms_list, properties=("energy", "forces")) -> BatchResult:
        """Evaluate AIMNet2 structures in one concatenated ``mol_idx`` batch.

        AIMNet2's upstream interface supports both dense ``(B, N, 3)`` inputs
        and concatenated atom lists keyed by ``mol_idx``.  The latter naturally
        covers variable-size molecules and the equal-size finite-difference
        Hessian workload without padding forces back through the model.
        """
        _, want_energy, want_forces, request = normalize_energy_forces_request(properties)
        if not request:
            return BatchResult()

        atoms_list = list(atoms_list)
        if not atoms_list:
            return empty_batch_result(want_energy, want_forces)

        if atoms_list_has_pbc(atoms_list):
            return sequential_calculate_many(self, atoms_list, request, want_energy, want_forces)

        if self.solvent_correction:
            return sequential_calculate_many(self, atoms_list, request, want_energy, want_forces)

        counts = atom_counts(atoms_list)
        B = len(atoms_list)
        coords_np = np.concatenate([at.get_positions() for at in atoms_list], axis=0)
        numbers_np = np.concatenate([at.get_atomic_numbers() for at in atoms_list], axis=0)
        mol_idx_np = np.concatenate(
            [np.full(len(at), i, dtype=np.int32) for i, at in enumerate(atoms_list)],
            axis=0,
        )

        coord = torch.tensor(
            coords_np,
            dtype=torch.float32,
            device=self.device,
            requires_grad=want_forces,
        )
        numbers = torch.tensor(numbers_np, dtype=torch.int32, device=self.device)
        mol_idx = torch.tensor(mol_idx_np, dtype=torch.int32, device=self.device)
        sentinel_mol = B

        charges = [float(at.info.get("charge", 0.0)) for at in atoms_list]
        mults = [float(at.info.get("mult", 1.0)) for at in atoms_list]
        nbmat = nblist_dense_padded_multi(coord, mol_idx, self.cutoff)
        lr_cutoff = self.cutoff_lr if np.isfinite(self.cutoff_lr) else self.cutoff

        data: Dict[str, torch.Tensor] = {
            "coord": pad_dim0(coord, value=0.0),
            "numbers": pad_dim0(numbers, value=0),
            "charge": torch.tensor(charges + [0.0], dtype=torch.float32, device=self.device),
            "mult": torch.tensor(mults + [1.0], dtype=torch.float32, device=self.device),
            "mol_idx": pad_dim0(mol_idx, value=sentinel_mol),
            "nbmat": nbmat,
            "nbmat_lr": nblist_dense_padded_multi(coord, mol_idx, lr_cutoff),
            "cutoff_lr": torch.tensor(lr_cutoff, device=self.device),
        }

        with torch.jit.optimized_execution(False):
            out = self.model(data)
        energy_vec = self._energy_vector_from_output(out["energy"], B, coord.shape[0], mol_idx)
        energy_vec = energy_vec * EV2HARTREE

        energies = (
            energy_vec.detach().cpu().numpy().astype(np.float64)
            if want_energy else None
        )

        forces_list = None
        if want_forces:
            grad_full = torch.autograd.grad(energy_vec.sum(), data["coord"])[0]
            forces_all = -grad_full[:coord.shape[0]]
            forces_list = split_atomwise_array(
                forces_all.detach().cpu().numpy().astype(np.float64),
                counts,
            )

        return BatchResult(energies=energies, forces=forces_list)

    def _energy_vector_from_output(
        self,
        energy_out: torch.Tensor,
        batch_size: int,
        n_atoms_total: int,
        mol_idx: torch.Tensor,
    ) -> torch.Tensor:
        energy_vec = energy_out.reshape(-1)
        if energy_vec.numel() in (batch_size + 1, n_atoms_total + 1):
            energy_vec = energy_vec[:-1]

        if energy_vec.numel() == batch_size:
            return energy_vec
        if energy_vec.numel() == n_atoms_total:
            return torch.zeros(
                batch_size,
                dtype=energy_vec.dtype,
                device=energy_vec.device,
            ).scatter_add(0, mol_idx.to(torch.long), energy_vec)

        raise RuntimeError(f"Unexpected AIMNet2 energy shape {tuple(energy_out.shape)}")

    # ------------------------ get_energy ------------------------
    def get_energy(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.jit.optimized_execution(False):
            out = self.model(data)
        return out["energy"].sum()

    # ------------------------ get_hessian ------------------------
    def get_hessian(
        self, 
        atoms,
        delta: float = 0.002,
    ) -> np.ndarray:
        """
        Compute the Hessian matrix using either analytic or numerical method.
        
        Method is determined by self.hessian:
        - 'analytic': Use automatic differentiation (faster, exact)
        - 'numerical': Use finite-difference forces (slower, approximate)
        
        Returns a (3N, 3N) numpy array.
        
        Args:
            atoms: ASE Atoms object
            delta: Step size for numerical differentiation (only used if method='numerical')
        """
        if self.hessian == 'analytic':
            return self._get_hessian_analytic(atoms)
        elif self.hessian == 'numerical':
            return self._get_hessian_numerical(atoms, delta)
        else:
            raise ValueError(f"Unknown hessian method: {self.hessian}. Must be 'analytic' or 'numerical'")


    def _get_hessian_analytic(self, atoms) -> np.ndarray:
        """
        Compute Hessian using automatic differentiation.
        Fast and exact, but requires energy to be differentiable w.r.t. coordinates.
        """
        data, n_atoms = self._build_single_molecule_hessian_data(atoms)

        energy = self.get_energy(data)
        energy = energy * EV2HARTREE

        try:
            hessian = self._hessian_from_energy_batched(
                data["coord"],
                energy,
                n_atoms=n_atoms,
                batch_size=getattr(self, "hessian_batch_size", getattr(self, "batch_size", None)),
            )
        except (RuntimeError, TypeError):
            # Some PyTorch/TorchScript operator combinations do not support
            # batched VJPs. Keep the original row-by-row path as the exact
            # compatibility fallback rather than silently switching to FD.
            data, n_atoms = self._build_single_molecule_hessian_data(atoms)
            energy = self.get_energy(data)
            energy = energy * EV2HARTREE
            hessian = self._hessian_from_energy_loop(
                data["coord"],
                energy,
                n_atoms=n_atoms,
            )

        return hessian.detach().cpu().numpy().reshape(3 * n_atoms, 3 * n_atoms)

    def _build_single_molecule_hessian_data(self, atoms) -> tuple[Dict[str, torch.Tensor], int]:
        """Build AIMNet2 padded input tensors for one-molecule analytic Hessian."""
        coord = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device)
        mol_idx = torch.zeros(coord.shape[0], dtype=torch.int32, device=self.device)

        N = coord.shape[0]
        nbmat = nblist_dense_padded(coord, self.cutoff)
        charge_val = float(atoms.info.get("charge", 0.0))
        mult_val = float(atoms.info.get("mult", 1.0))

        data = {
            "coord": pad_dim0(coord, value=0.0),
            "numbers": pad_dim0(Z, value=0),
            "charge": torch.tensor([charge_val], dtype=torch.float32, device=self.device),
            "mult": torch.tensor([mult_val], dtype=torch.float32, device=self.device),
            "mol_idx": pad_dim0(mol_idx, value=mol_idx[-1].item() if N > 0 else 0),
            "nbmat": nbmat,
        }
        
        # ALWAYS provide nbmat_lr + cutoff_lr
        lr_cutoff = self.cutoff_lr if np.isfinite(self.cutoff_lr) else self.cutoff
        data["nbmat_lr"] = nblist_dense_padded(coord, lr_cutoff)
        data["cutoff_lr"] = torch.tensor(lr_cutoff, device=self.device)

        return data, N

    @staticmethod
    def _hessian_from_energy_batched(
        coord_padded: torch.Tensor,
        energy: torch.Tensor,
        *,
        n_atoms: int,
        batch_size=None,
    ) -> torch.Tensor:
        """Analytic Hessian via one batched vector-Jacobian product.

        This keeps the exact autograd Hessian but avoids the Python loop over
        3N force components on PyTorch builds where batched VJPs are supported.
        AIMNet2 has one sentinel-padded coordinate row; the physical Hessian is
        the top-left 3N x 3N block.
        """
        n3 = 3 * n_atoms
        return hessian_batched_vjp(
            energy,
            coord_padded,
            output_dof=n3,
            input_dof=n3,
            batch_size=batch_size,
            warn_on_fallback=True,
        )

    @staticmethod
    def _hessian_from_energy_loop(
        coord_padded: torch.Tensor,
        energy: torch.Tensor,
        *,
        n_atoms: int,
    ) -> torch.Tensor:
        """Original row-by-row analytic Hessian assembly."""
        n3 = 3 * n_atoms
        return hessian_loop(
            energy,
            coord_padded,
            output_dof=n3,
            input_dof=n3,
        )


    def _get_hessian_numerical(
        self,
        atoms,
        delta: float = 0.002,
    ) -> np.ndarray:
        """Central-difference numerical Hessian via batched displacement.

        Delegates the 2 * 3 * N_movable force evaluations to
        ``FDHessianEvaluator``, which routes through ``calc.calculate_many``
        (sequential fallback in ``CalcABC`` by default; subclasses can
        override for true batched evaluation). FixAtoms respected upstream.
        """
        from .._batch_eval import FDHessianEvaluator

        return FDHessianEvaluator(
            self,
            fd_batch_size=getattr(self, "fd_batch_size", None),
        ).hessian(atoms, delta=delta)
