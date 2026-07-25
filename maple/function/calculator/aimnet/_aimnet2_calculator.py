from __future__ import annotations

import hashlib
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
from ..calculator_base import CalcABC, register_calculator
from ..calculator_base import EV2HARTREE


AIMNET2_PADDED_PER_MOLECULE_LAYOUT = "padded_per_molecule_v1"
AIMNET2_BATCH_LAYOUT_BY_SHA256 = {
    # Packaged MAPLE checkpoints validated by the batch parity suite.
    "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d":
        AIMNET2_PADDED_PER_MOLECULE_LAYOUT,
    "924570119cdb7a5f83ee204d72842620357ba482d7a41f900aed04f229e174b5":
        AIMNET2_PADDED_PER_MOLECULE_LAYOUT,
}


def identify_aimnet2_batch_layout(model_path: str) -> str | None:
    """Return the validated batch output schema for a known checkpoint."""
    digest = hashlib.sha256()
    try:
        with open(model_path, "rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return AIMNET2_BATCH_LAYOUT_BY_SHA256.get(digest.hexdigest())


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


def nblist_dense_padded_multi(
    coord: torch.Tensor,
    mol_idx: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """Dense sentinel-padded neighbor list for concatenated molecules.

    The same-molecule mask is the critical scientific guard: without it,
    unrelated NEB/path images concatenated into one tensor could form artificial
    cross-image edges whenever two atoms happen to be close in Cartesian space.
    """
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
            k = min(nb_i.numel(), M)
            nbmat[i, :k] = nb_i[:k].to(torch.int32)
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
@register_calculator
class AIMNet2Calculator(CalcABC):
    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian']

    MODEL_NAMES = ('aimnet2', 'aimnet2nse')
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False
    SUPPORTED_COULOMB_METHODS = ('simple', 'dsf')
    CHECKPOINT_FILENAME = {'aimnet2': 'aimnet2.pt', 'aimnet2nse': 'aimnet2nse.pt'}
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ('coulomb_method', 'batch_size', 'path_batch_size')
    MODEL_PATH_OPTION = 'model_path'
    supports_batch_energy_forces = False
    supports_analytic_hessian = True
    batch_memory_model = 'concat_dense_neighbor'
    auto_batch_hard_cap = 8
    auto_path_batch_cap = 8
    BATCH_ENERGY_LAYOUT = None

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        coulomb_method = options.get('coulomb_method')
        if coulomb_method is not None:
            kwargs['coulomb_method'] = str(coulomb_method).lower()
        if options.get('batch_size') is not None:
            kwargs['batch_size'] = options.get('batch_size')
        if options.get('path_batch_size') is not None:
            kwargs['path_batch_size'] = options.get('path_batch_size')
        if resolved_model_path is not None:
            kwargs['model_path'] = resolved_model_path
        return kwargs

    def __init__(self, device: torch.device,
                model: str = 'aimnet2',
                model_path: str = None,
                coulomb_method: str = 'simple',
                batch_size=None,
                path_batch_size='auto',
                implicit: Literal['gbsa', 'none'] = 'none',
                solvent: str = 'none',
                ):
        super().__init__()
        self.device = device

        # Load model
        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            model_path = os.path.join(model_dir, 'model', f'{model}.pt')
        self.model_path = os.path.realpath(model_path)
        self._batch_energy_layout = identify_aimnet2_batch_layout(self.model_path)
        self.BATCH_ENERGY_LAYOUT = self._batch_energy_layout
        self.supports_batch_energy_forces = (
            self._batch_energy_layout == AIMNET2_PADDED_PER_MOLECULE_LAYOUT
        )
        self.model = torch.jit.load(model_path, map_location=device).eval()

        self.cutoff = float(getattr(self.model, 'cutoff'))
        self.cutoff_lr = float(getattr(self.model, 'cutoff_lr', float('inf')))
        # Always provide nbmat_lr to avoid TorchScript KeyError; method routing
        # handles cutoff_lr.
        self.lr = True
        self.hessian: str = 'analytic'
        self.batch_size = batch_size
        self.path_batch_size = path_batch_size

        self._set_lrcoulomb_method(coulomb_method)

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _set_lrcoulomb_method(self, method: str, cutoff: float = 15.0, dsf_alpha: float = 0.2):
        """
        Configure the long-range Coulomb interaction method if the model contains a 'lrcoulomb' submodule.
        method: 'simple' or 'dsf'. The historical 'ewald' selector is rejected
        until this wrapper carries validated cell/PBC/MIC inputs.
        cutoff: cutoff distance for long-range interactions
        dsf_alpha: DSF damping parameter (if used)
        """
        method = str(method).lower()
        if method == 'ewald':
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
                if name == 'lrcoulomb':
                    yield mod

        for mod in _iter_lrcoulomb_mods(self.model):
            mod.method = method
            if method == 'dsf' and hasattr(mod, 'dsf_alpha'):
                mod.dsf_alpha = dsf_alpha

        self.cutoff_lr = float('inf') if method == 'simple' else float(cutoff)
        self._coulomb_method = method

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        needs_grad = ('forces' in properties or 'hessian' in properties)
        coord = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=needs_grad,
        )
        data = self._build_data(coord, atoms)

        # Pure model energy in eV; _finalize_results handles eV→Ha + solvent.
        energy_eV = self._forward_energy(data)

        if 'forces' in properties:
            grad_full = torch.autograd.grad(
                energy_eV, data['coord'], create_graph=('hessian' in properties)
            )[0]
            forces_eV = -grad_full[: coord.shape[0]]
            forces_np = forces_eV.detach().cpu().numpy()
        else:
            forces_np = None

        hessian = None
        if 'hessian' in properties:
            if self.solvent_correction is not None:
                raise NotImplementedError(
                    'Hessian calculation with implicit solvent is not implemented yet.'
                )
            hessian = self.get_hessian(atoms)

        self._finalize_results(atoms, energy=energy_eV.item(), forces=forces_np, hessian=hessian)

    def calculate_many(self, atoms_list, properties=("energy", "forces")) -> BatchResult:
        """Evaluate AIMNet2 structures as one ``mol_idx``-keyed batch.

        AIMNet2 can represent a batch as a concatenated atom list plus
        per-atom molecule indices.  This preserves variable molecule sizes and
        avoids padding the model input, while ``nblist_dense_padded_multi``
        prevents any cross-structure neighbor edges.
        """
        _, want_energy, want_forces, request = normalize_energy_forces_request(
            properties
        )
        if not request:
            return BatchResult()

        atoms_list = list(atoms_list)
        if not atoms_list:
            return empty_batch_result(want_energy, want_forces)

        if atoms_list_has_pbc(atoms_list):
            return sequential_calculate_many(
                self, atoms_list, request, want_energy, want_forces
            )

        if getattr(self, 'solvent_correction', None) is not None:
            return sequential_calculate_many(
                self, atoms_list, request, want_energy, want_forces
            )
        if not self.supports_batch_energy_forces:
            return sequential_calculate_many(
                self, atoms_list, request, want_energy, want_forces
            )

        counts = atom_counts(atoms_list)
        batch_size = len(atoms_list)
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

        charges = [float(at.info.get('charge', 0.0)) for at in atoms_list]
        mults = [float(at.info.get('mult', 1.0)) for at in atoms_list]
        lr_cutoff = self.cutoff_lr if np.isfinite(self.cutoff_lr) else self.cutoff

        data: Dict[str, torch.Tensor] = {
            'coord': pad_dim0(coord, value=0.0),
            'numbers': pad_dim0(numbers, value=0),
            'charge': torch.tensor(charges + [0.0], dtype=torch.float32, device=self.device),
            'mult': torch.tensor(mults + [1.0], dtype=torch.float32, device=self.device),
            'mol_idx': pad_dim0(mol_idx, value=batch_size),
            'nbmat': nblist_dense_padded_multi(coord, mol_idx, self.cutoff),
            'nbmat_lr': nblist_dense_padded_multi(coord, mol_idx, lr_cutoff),
            'cutoff_lr': torch.tensor(lr_cutoff, device=self.device),
        }

        with torch.jit.optimized_execution(False):
            out = self.model(data)
        energy_vec_eV = self._energy_vector_from_output(
            out['energy'], batch_size, self._batch_energy_layout
        )
        energy_vec_ha = energy_vec_eV * EV2HARTREE

        energies = (
            energy_vec_ha.detach().cpu().numpy().astype(np.float64)
            if want_energy else None
        )

        forces_list = None
        if want_forces:
            grad_full = torch.autograd.grad(energy_vec_ha.sum(), data['coord'])[0]
            forces_all = -grad_full[: coord.shape[0]]
            forces_list = split_atomwise_array(
                forces_all.detach().cpu().numpy().astype(np.float64),
                counts,
            )

        return BatchResult(
            energies=energies,
            forces=forces_list,
        ).validate_against(atoms_list, request)

    @staticmethod
    def _energy_vector_from_output(
        energy_out: torch.Tensor,
        batch_size: int,
        layout: str | None,
    ) -> torch.Tensor:
        """Read the checkpoint's versioned padded-per-molecule energy layout.

        MAPLE's supported ``aimnet2.pt`` and ``aimnet2nse.pt`` checkpoints
        return one energy per molecule followed by the sentinel molecule
        introduced by ``pad_dim0``. Atom-wise and unpadded layouts are not
        guessed here because their lengths can be numerically ambiguous.
        """
        if layout != AIMNET2_PADDED_PER_MOLECULE_LAYOUT:
            raise RuntimeError(
                "AIMNet2 native batching requires a checkpoint with the "
                f"validated {AIMNET2_PADDED_PER_MOLECULE_LAYOUT!r} schema"
            )
        energy_vec = energy_out.reshape(-1)
        expected = batch_size + 1
        if energy_vec.numel() != expected:
            raise RuntimeError(
                "AIMNet2 checkpoint must return the padded per-molecule energy "
                f"layout with {expected} entries for batch size {batch_size}; "
                f"got shape {tuple(energy_out.shape)} ({energy_vec.numel()} entries)"
            )
        if not bool(torch.isfinite(energy_vec).all().item()):
            raise FloatingPointError(
                "AIMNet2 returned non-finite padded per-molecule energies"
            )
        return energy_vec[:-1]

    def _build_data(self, coord: torch.Tensor, atoms) -> Dict[str, torch.Tensor]:
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device)
        mol_idx = torch.zeros(coord.shape[0], dtype=torch.int32, device=self.device)
        N = coord.shape[0]
        charge_val = float(atoms.info.get('charge', 0.0))
        mult_val = float(atoms.info.get('mult', 1.0))

        nbmat = nblist_dense_padded(coord, self.cutoff)
        data: Dict[str, torch.Tensor] = {
            'coord': pad_dim0(coord, value=0.0),
            'numbers': pad_dim0(Z, value=0),
            'charge': torch.tensor([charge_val], dtype=torch.float32, device=self.device),
            'mult': torch.tensor([mult_val], dtype=torch.float32, device=self.device),
            'mol_idx': pad_dim0(mol_idx, value=mol_idx[-1].item() if N > 0 else 0),
            'nbmat': nbmat,
        }

        lr_cutoff = self.cutoff_lr if np.isfinite(self.cutoff_lr) else self.cutoff
        data['nbmat_lr'] = nblist_dense_padded(coord, lr_cutoff)
        data['cutoff_lr'] = torch.tensor(lr_cutoff, device=self.device)
        return data

    def _forward_energy(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pure model forward; returns energy in eV (model's native unit)."""
        with torch.jit.optimized_execution(False):
            out = self.model(data)
        return out['energy'].sum()

    def _analytic_hessian(self, atoms) -> np.ndarray:
        """Analytic Hessian via autograd. Returns (3N, 3N) np.ndarray in Hartree/Å²."""
        from ..calculator_base import EV2HARTREE

        coord = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        N = coord.shape[0]
        data = self._build_data(coord, atoms)

        energy = self._forward_energy(data) * EV2HARTREE

        forces_full = torch.autograd.grad(energy, data['coord'], create_graph=True)[0]
        forces = -forces_full[:N]

        hessian = -torch.stack([
            torch.autograd.grad(f, data['coord'], retain_graph=True)[0]
            for f in forces.flatten().unbind()
        ]).view(-1, 3, N + 1, 3)[:, :, :N, :]

        return hessian.detach().cpu().numpy().reshape(3 * N, 3 * N)
