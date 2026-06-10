from __future__ import annotations

import os

import ase
import numpy as np

from ..calculator_base import (
    CalcABC,
    hessian_via_double_autograd,
    parse_bool_option,
    register_calculator,
)
from .._batch_types import BatchResult
from .._batch_utils import (
    atoms_list_has_pbc,
    empty_batch_result,
    grouped_indices_by_numbers,
    normalize_energy_forces_request,
    sequential_calculate_many,
)


@register_calculator
class ANICalculator(CalcABC):
    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian']

    MODEL_NAMES = ('ani2x', 'ani1x', 'ani1ccx', 'ani1xnr')
    # ANI's TorchScript checkpoints already return Hartree; no eV→Ha conversion.
    MODEL_ENERGY_UNIT = 'hartree'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = {
        'ani2x': 'ani2x.pt',
        'ani1x': 'ani1x.pt',
        'ani1ccx': 'ani1ccx.pt',
        'ani1xnr': 'ani1xnr.pt',
    }
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ('d4', 'batch_size', 'path_batch_size')
    MODEL_PATH_OPTION = 'model_path'
    supports_batch_energy_forces = True
    supports_analytic_hessian = True
    batch_memory_model = 'dense_same_shape'
    auto_path_batch_cap = 8

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {'d4': parse_bool_option(options.get('d4', False), name='d4')}
        if options.get('batch_size') is not None:
            kwargs['batch_size'] = options.get('batch_size')
        if options.get('path_batch_size') is not None:
            kwargs['path_batch_size'] = options.get('path_batch_size')
        if resolved_model_path is not None:
            kwargs['model_path'] = resolved_model_path
        return kwargs

    def __init__(self, device,
        model: str = 'ani2x',
        model_path: str = None,
        overwrite=False,
        d4=False,
        batch_size=None,
        path_batch_size='auto',
        implicit: str = 'none',
        solvent: str = 'none',
        ):
        import torch

        super().__init__()

        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            model_path = os.path.join(model_dir, 'model', f'{model}.pt')

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = torch.float32
        self.overwrite = overwrite
        self.d4 = d4
        self.batch_size = batch_size
        self.path_batch_size = path_batch_size
        self.hessian: str = 'analytic'

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        import torch

        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        needs_forces = 'forces' in properties
        coordinates = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=needs_forces,
        ).unsqueeze(0)

        energy = self._forward_energy(atoms, coordinates)

        if needs_forces:
            forces = -torch.autograd.grad(energy, coordinates)[0]
            forces_np = forces.squeeze(0).cpu().numpy()
        else:
            forces_np = None

        hessian = None
        if 'hessian' in properties:
            if self.solvent_correction is not None:
                raise NotImplementedError(
                    'Hessian calculation with implicit solvent is not implemented yet.'
                )
            hessian = self.get_hessian(atoms)

        self._finalize_results(atoms, energy=energy.item(), forces=forces_np, hessian=hessian)

    def calculate_many(self, atoms_list, properties=("energy", "forces")) -> BatchResult:
        """Evaluate same-composition ANI structures in one TorchScript batch.

        ANI checkpoints accept species as ``(B, N)`` and coordinates as
        ``(B, N, 3)``.  Grouping structures by identical atomic-number order lets
        NEB/path evaluators fetch all image energies and forces with model-level
        batches while preserving MAPLE's per-image force/energy contract.

        D4, implicit solvent, and PBC deliberately fall back to the validated
        single-structure path because their batched derivative semantics are not
        part of the current production contract.
        """
        import torch

        _, want_energy, want_forces, request = normalize_energy_forces_request(properties)
        if not request:
            return BatchResult()

        atoms_list = list(atoms_list)
        if not atoms_list:
            return empty_batch_result(want_energy, want_forces)

        if atoms_list_has_pbc(atoms_list):
            return sequential_calculate_many(self, atoms_list, request, want_energy, want_forces)

        if self.d4 or getattr(self, 'solvent_correction', None) is not None:
            return sequential_calculate_many(self, atoms_list, request, want_energy, want_forces)

        energies = np.empty(len(atoms_list), dtype=np.float64) if want_energy else None
        forces_out = [None] * len(atoms_list) if want_forces else None

        for numbers, indices in grouped_indices_by_numbers(atoms_list):
            group_atoms = [atoms_list[i] for i in indices]
            species = torch.tensor(
                [numbers] * len(group_atoms),
                dtype=torch.long,
                device=self.device,
            )
            coords_np = np.stack([at.get_positions() for at in group_atoms], axis=0)
            coordinates = torch.tensor(
                coords_np,
                dtype=self.dtype,
                device=self.device,
                requires_grad=want_forces,
            )

            if want_forces:
                energy_vec = self.model(species, coordinates)[0].reshape(-1)
                force_tensor = -torch.autograd.grad(energy_vec.sum(), coordinates)[0]
            else:
                with torch.no_grad():
                    energy_vec = self.model(species, coordinates)[0].reshape(-1)
                force_tensor = None

            if want_energy:
                energy_np = energy_vec.detach().cpu().numpy().astype(np.float64)
                for out_i, val in zip(indices, energy_np):
                    energies[out_i] = float(val)

            if want_forces:
                force_np = force_tensor.detach().cpu().numpy().astype(np.float64)
                for out_i, val in zip(indices, force_np):
                    forces_out[out_i] = val

        return BatchResult(energies=energies, forces=forces_out)

    def _forward_energy(self, atoms, coordinates):
        import torch

        species = torch.tensor(
            atoms.get_atomic_numbers(),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        energy = self.model(species, coordinates)[0]
        if self.d4:
            energy = energy + self.dftd4(species, coordinates)

        return energy

    def _analytic_hessian(self, atoms) -> np.ndarray:
        import torch

        coordinates = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        ).unsqueeze(0)

        # ANI's TorchScript model is Hartree-native, so energy_fn returns Hartree
        # directly (no EV2HARTREE) and the shared helper yields Hartree/Å².
        return hessian_via_double_autograd(
            lambda: self._forward_energy(atoms, coordinates), coordinates
        )

    def dftd4(self, species, coordinates):
        import torch
        import tad_dftd4 as d4

        charge = torch.tensor(0.0, device=self.device)
        param = {
            's6': coordinates.new_tensor(1.0),
            's8': coordinates.new_tensor(0.34783580),
            's9': coordinates.new_tensor(1.0),
            'a1': coordinates.new_tensor(0.57488291),
            'a2': coordinates.new_tensor(6.41921802),
        }
        bohr_coords = coordinates[0] * 1.8897261245864
        return torch.sum(d4.dftd4(species[0], bohr_coords, charge, param))

    def get_hvp(self, atoms, n: np.ndarray):
        """Hessian-vector product Hn via autograd for ANI's (species, coords) forward.

        Returns (Hn, forces, energy) as torch tensors, consumed by Dimer-mode TS.
        """
        if getattr(self, 'solvent_correction', None) is not None:
            raise NotImplementedError(
                'ANI HVP with implicit solvent is not supported; solvent HVP would be omitted.'
            )

        import torch

        coords = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        ).unsqueeze(0)
        species = torch.tensor(
            atoms.get_atomic_numbers(),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        energy = self.model(species, coords)[0]
        if self.d4:
            energy = energy + self.dftd4(species, coords)

        grad = torch.autograd.grad(energy, coords, create_graph=True)[0].squeeze(0)
        grad_vec = grad.view(-1)

        n_tensor = torch.tensor(n, dtype=self.dtype, device=self.device)
        hvp = torch.autograd.grad(
            grad_vec @ n_tensor, coords, retain_graph=True
        )[0].squeeze(0).view(-1)

        forces = -grad_vec
        return hvp, forces, energy
