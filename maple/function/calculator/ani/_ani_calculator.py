
import os

import torch
import numpy as np

import ase

from ..calculator_base import CalcABC
from .._batch_types import BatchResult
from .._batch_utils import (
    empty_batch_result,
    grouped_indices_by_numbers,
    normalize_energy_forces_request,
    sequential_calculate_many,
)
from .._autograd_hessian import hessian_loop


class ANICalculator(CalcABC):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']
    supported_hessian_modes = ("analytic", "numerical")
    supports_batch_energy_forces = True
    supports_analytic_hessian = True

    def __init__(self, device: torch.device,
        model:str = 'ani2x',
        overwrite=False,
        d4=False,
        implicit: str = 'none',
        solvent: str = 'none',
        ):
        """
        Initialize the ANICalculator.

        Args:
            device (torch.device): The device to run the model on.
            model (str, optional): The model to use. Defaults to 'ani-2x'.
            overwrite (bool, optional): Whether to overwrite existing models. Defaults to False.
            d4 (bool, optional): Whether to use D4 dispersion correction. Defaults to False.
        """
        super().__init__()
        info_message = [f"\nLoading the Machine Learning Potential Model...\n"]
        
        
        model_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.dirname(model_dir)
        model_path = os.path.join(model_dir, 'model', f'{model}.pt')
        
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        
        
        info_message.append(f'Loading model ({model}) successfully.\n')
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = torch.float32
        self.overwrite = overwrite
        self.d4 = d4
        self.hessian: str = 'analytic'  # 'analytic' or 'numerical'

        # Initialize implicit solvent
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super().calculate(atoms, properties, system_changes)

        coordinates = torch.tensor(atoms.get_positions(), dtype=self.dtype, device=self.device, requires_grad='forces' in properties).unsqueeze(0)
        
        energy = self.get_energy(atoms, coordinates)

        if self.solvent_correction:
            solvent_energy = self.implicit_solv_energy(atoms)
            energy += solvent_energy

        self.results['energy'] = energy.item()
        self.results['free_energy'] = energy.item()

        if 'forces' in properties:
            forces = -torch.autograd.grad(energy, coordinates, retain_graph='stress' in properties)[0]
            if self.solvent_correction:
                solvent_energy, solvent_force = self.implicit_solv_energy_and_force(atoms)
                forces += solvent_force
                
            self.results['forces'] = forces.squeeze(0).cpu().numpy()

    def calculate_many(self, atoms_list, properties=("energy", "forces")) -> BatchResult:
        """Evaluate same-composition ANI structures in native torch batches.

        The scripted ANI wrapper accepts ``species`` as ``(B, N)`` and
        coordinates as ``(B, N, 3)``.  Finite-difference Hessian displacements
        are exactly this case, so grouping by atomic-number sequence turns
        ``2 * 3N`` calculator calls into one model call per composition.
        D4 and implicit-solvent corrections remain on the sequential path
        until their batched force semantics are validated.
        """
        _, want_energy, want_forces, request = normalize_energy_forces_request(properties)
        if not request:
            return BatchResult()

        atoms_list = list(atoms_list)
        if not atoms_list:
            return empty_batch_result(want_energy, want_forces)

        if self.d4 or self.solvent_correction:
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

    def get_energy(self, atoms, coordinates):
        
        species = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device).unsqueeze(0)

        energy = self.model(species, coordinates)[0]
        if self.d4:
            energy += self.dftd4(species, coordinates)
        
        return energy

    @staticmethod
    def compute_hessian(coords, energy):
    
        num_atoms = coords.shape[1]
        return hessian_loop(energy, coords, output_dof=3 * num_atoms, input_dof=3 * num_atoms)

    def get_hessian(
        self,
        atoms: ase.Atoms,
        delta: float = 0.002,
    ) -> torch.Tensor:
        """
        Compute the Hessian matrix using either analytic or numerical method.
        
        Method is determined by self.hessian:
        - 'analytic': Use automatic differentiation (faster, exact)
        - 'numerical': Use finite-difference forces (slower, approximate)
        
        Returns a (3N, 3N) torch.Tensor on self.device.
        
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


    def _get_hessian_analytic(self, atoms: ase.Atoms) -> torch.Tensor:
        """
        Compute Hessian using automatic differentiation.
        Fast and exact, but requires energy to be differentiable w.r.t. coordinates.
        """
        coordinates = torch.tensor(
            atoms.get_positions(), 
            dtype=self.dtype, 
            device=self.device, 
            requires_grad=True
        ).unsqueeze(0)
        
        if self.d4:
            energy = self.get_energy(atoms, coordinates)
        else:
            species = torch.tensor(
                atoms.get_atomic_numbers(),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
            energy = self.model(species, coordinates)[0]
        
        return self.compute_hessian(coordinates, energy)


    def _get_hessian_numerical(
        self,
        atoms: ase.Atoms,
        delta: float = 0.002,
    ) -> torch.Tensor:
        """Central-difference numerical Hessian via batched displacement.

        Routes the 2 * 3 * N_movable force evaluations through
        ``calc.calculate_many`` so a batched backend (or the sequential
        fallback in ``CalcABC.calculate_many``) handles dispatch, instead of
        a hand-rolled per-DOF Python loop. FixAtoms is respected upstream.
        Returns a ``(3N, 3N)`` tensor on ``self.device`` to preserve the
        original return-type contract.
        """
        from .._batch_eval import FDHessianEvaluator

        H_np = FDHessianEvaluator(
            self,
            fd_batch_size=getattr(self, "fd_batch_size", None),
        ).hessian(atoms, delta=delta)
        return torch.as_tensor(H_np, dtype=self.dtype, device=self.device)


    def dftd4(self, species, coordinates):
        import tad_dftd4 as d4
        charge = torch.tensor(0.0, device=self.device)
        param = {
            "s6": coordinates.new_tensor(1.0),
            "s8": coordinates.new_tensor(0.34783580),
            "s9": coordinates.new_tensor(1.0),
            "a1": coordinates.new_tensor(0.57488291),
            "a2": coordinates.new_tensor(6.41921802),
        }
        # Å → Bohr 转换
        bohr_coords = coordinates[0] * 1.8897261245864
        return torch.sum(d4.dftd4(species[0], bohr_coords, charge, param))


    
