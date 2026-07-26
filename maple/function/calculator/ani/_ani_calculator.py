from __future__ import annotations

import os

import ase
import numpy as np

from .._batch_types import BatchResult
from .._batch_utils import (
    atoms_list_has_pbc,
    empty_batch_result,
    grouped_indices_by_numbers,
    normalize_energy_forces_request,
    sequential_calculate_many,
)
from ..calculator_base import (
    CalcABC,
    hessian_via_double_autograd,
    parse_bool_option,
    register_calculator,
)


@register_calculator
class ANICalculator(CalcABC):
    implemented_properties = ["energy", "forces", "free_energy", "hessian"]

    MODEL_NAMES = ("ani2x", "ani1x", "ani1ccx", "ani1xnr")
    # ANI's TorchScript checkpoints already return Hartree; no eV→Ha conversion.
    MODEL_ENERGY_UNIT = "hartree"
    SUPPORTED_HESSIAN_MODES = ("analytic", "numerical")
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False
    SUPPORTS_IMPLICIT_SOLVATION = True
    CHECKPOINT_FILENAME = {
        "ani2x": "ani2x.pt",
        "ani1x": "ani1x.pt",
        "ani1ccx": "ani1ccx.pt",
        "ani1xnr": "ani1xnr.pt",
    }
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ("d4",)
    MODEL_PATH_OPTION = "model_path"
    supports_batch_energy_forces = True

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {"d4": parse_bool_option(options.get("d4", False), name="d4")}
        if resolved_model_path is not None:
            kwargs["model_path"] = resolved_model_path
        return kwargs

    def __init__(
        self,
        device,
        model: str = "ani2x",
        model_path: str = None,
        overwrite=False,
        d4=False,
        implicit: str = "none",
        solvent: str = "none",
    ):
        import torch

        super().__init__()

        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            model_path = os.path.join(model_dir, "model", f"{model}.pt")

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        converter = dict(self.model.named_buffers()).get(
            "model.species_converter.conv_tensor"
        )
        if converter is None:
            raise RuntimeError(
                f"ANI checkpoint '{model}' does not expose its "
                "species-conversion table."
            )
        self.atomic_numbers = [
            index
            for index, species_index in enumerate(converter.detach().cpu().tolist())
            if int(species_index) >= 0
        ]
        if not self.atomic_numbers:
            raise RuntimeError(
                f"ANI checkpoint '{model}' exposes an empty element domain."
            )

        self.device = device
        self.dtype = torch.float32
        self.overwrite = overwrite
        self.d4 = d4
        self.hessian: str = "analytic"

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(
        self,
        atoms=None,
        properties=["energy"],
        system_changes=ase.calculators.calculator.all_changes,
    ):
        import torch

        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        needs_forces = "forces" in properties
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
        if "hessian" in properties:
            if self.solvent_correction is not None:
                raise NotImplementedError(
                    "Hessian calculation with implicit solvent is not implemented yet."
                )
            hessian = self.get_hessian(atoms)

        self._finalize_results(
            atoms, energy=energy.item(), forces=forces_np, hessian=hessian
        )

    def calculate_many(
        self,
        atoms_list,
        properties=("energy", "forces"),
    ) -> BatchResult:
        """Batch ANI structures grouped by identical element ordering.

        D4, PBC, and attached implicit-solvent corrections remain on the
        validated sequential path because their batched semantics are outside
        this contract.
        """
        import torch

        _, want_energy, want_forces, request = normalize_energy_forces_request(
            properties
        )
        if not request:
            return BatchResult()

        atoms_list = list(atoms_list)
        if not atoms_list:
            return empty_batch_result(want_energy, want_forces)
        if (
            atoms_list_has_pbc(atoms_list)
            or self.d4
            or self.solvent_correction is not None
        ):
            return sequential_calculate_many(
                self,
                atoms_list,
                request,
                want_energy,
                want_forces,
            )

        energies = np.empty(len(atoms_list), dtype=np.float64) if want_energy else None
        forces_list = [None] * len(atoms_list) if want_forces else None

        for atomic_numbers, indices in grouped_indices_by_numbers(atoms_list):
            group = [atoms_list[index] for index in indices]
            species = torch.tensor(
                [atomic_numbers] * len(group),
                dtype=torch.long,
                device=self.device,
            )
            coordinates = torch.tensor(
                np.stack(
                    [atoms.get_positions() for atoms in group],
                    axis=0,
                ),
                dtype=self.dtype,
                device=self.device,
                requires_grad=want_forces,
            )

            if want_forces:
                energy_vector = self.model(species, coordinates)[0].reshape(-1)
                force_tensor = -torch.autograd.grad(
                    energy_vector.sum(),
                    coordinates,
                )[0]
            else:
                with torch.no_grad():
                    energy_vector = self.model(
                        species,
                        coordinates,
                    )[
                        0
                    ].reshape(-1)
                force_tensor = None

            if energy_vector.numel() != len(group):
                raise RuntimeError(
                    "ANI batch model returned "
                    f"{energy_vector.numel()} energies for {len(group)} structures."
                )
            if want_forces and force_tensor.shape != coordinates.shape:
                raise RuntimeError(
                    "ANI batch force shape does not match the input coordinates: "
                    f"{tuple(force_tensor.shape)} != {tuple(coordinates.shape)}."
                )

            if want_energy:
                energy_values = energy_vector.detach().cpu().numpy().astype(np.float64)
                for output_index, value in zip(indices, energy_values):
                    energies[output_index] = float(value)
            if want_forces:
                force_values = force_tensor.detach().cpu().numpy().astype(np.float64)
                for output_index, value in zip(indices, force_values):
                    forces_list[output_index] = value

        return BatchResult(energies=energies, forces=forces_list)

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
            "s6": coordinates.new_tensor(1.0),
            "s8": coordinates.new_tensor(0.34783580),
            "s9": coordinates.new_tensor(1.0),
            "a1": coordinates.new_tensor(0.57488291),
            "a2": coordinates.new_tensor(6.41921802),
        }
        bohr_coords = coordinates[0] * 1.8897261245864
        return torch.sum(d4.dftd4(species[0], bohr_coords, charge, param))

    def get_hvp(self, atoms, n: np.ndarray):
        """Hessian-vector product Hn via autograd for ANI's (species, coords) forward.

        Returns (Hn, forces, energy) as torch tensors, consumed by Dimer-mode TS.
        """
        if getattr(self, "solvent_correction", None) is not None:
            raise NotImplementedError(
                "ANI HVP with implicit solvent is not supported; solvent HVP would be omitted."
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
        hvp = (
            torch.autograd.grad(grad_vec @ n_tensor, coords, retain_graph=True)[0]
            .squeeze(0)
            .view(-1)
        )

        forces = -grad_vec
        return hvp, forces, energy
