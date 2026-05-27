import os
import torch
import numpy as np
from typing import Dict, Union, Sequence, Optional
from ase.calculators.calculator import all_changes
from ..calculator_base import CalcABC
from .._batch_types import BatchResult
from .._batch_utils import (
    atoms_list_has_pbc,
    empty_batch_result,
    normalize_energy_forces_request,
    sequential_calculate_many,
    split_atomwise_array,
)
from .._autograd_hessian import hessian_loop
from ._batch_graph import build_mace_data_dict_batch, energy_vector_from_output
from typing import Literal
EV2HARTREE = 1.0 / 27.211386245988

# ------------------------ Basic helpers ------------------------

_SYMBOL2Z = {
    "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9, "Ne":10,
    "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18,
    "K":19, "Ca":20, "Sc":21, "Ti":22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27, "Ni":28, "Cu":29, "Zn":30
}

def _symbols_to_Z(symbols: Sequence[Union[str,int]]) -> list:
    """Convert element symbols to atomic numbers."""
    out = []
    for s in symbols:
        if isinstance(s, int):
            out.append(int(s))
        else:
            z = _SYMBOL2Z.get(str(s))
            if z is None:
                raise ValueError(f"Unknown element symbol: {s}")
            out.append(z)
    return out

def _one_hot_node_attrs(Z: torch.Tensor, atomic_number_table: list, dtype=torch.float64) -> torch.Tensor:
    """Convert atomic numbers into one-hot vectors aligned with atomic_number_table."""
    table = torch.tensor(atomic_number_table, dtype=torch.long, device=Z.device)
    eq = (Z[:, None] == table[None, :])
    if not torch.all(eq.any(dim=1)):
        miss = Z[~eq.any(dim=1)].unique().tolist()
        raise ValueError(f"Atomic number(s) {miss} not in AtomicNumberTable {atomic_number_table}")
    return eq.to(dtype)

def _radius_graph_no_pbc(positions: torch.Tensor, r_max: float):
    """Construct a simple O(N^2) radius graph without periodic boundaries."""
    N = positions.size(0)
    rij = positions[:, None, :] - positions[None, :, :]
    d2 = (rij * rij).sum(dim=-1)
    mask = torch.ones((N, N), dtype=torch.bool, device=positions.device)
    mask.fill_diagonal_(False)
    mask &= (d2 <= (r_max + 1e-12) ** 2)
    iu, ju = torch.nonzero(torch.triu(mask), as_tuple=True)
    src = torch.cat([iu, ju], dim=0)
    dst = torch.cat([ju, iu], dim=0)
    edge_index = torch.stack([src, dst], dim=0).to(torch.long)
    shifts = torch.zeros((edge_index.size(1), 3), dtype=positions.dtype, device=positions.device)
    return edge_index, shifts

def _model_float_dtype(model) -> torch.dtype:
    """Return the floating dtype used by a scripted MACE wrapper."""
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_floating_point():
            return tensor.dtype
    return torch.float64


_MACE_OFF_SIZE = {
    "maceoff23s": "small",
    "maceoff23m": "medium",
    "maceoff23l": "large",
}

_RAW_UPSTREAM_MODELS = {"maceoff23s", "maceoff23l", "maceomol"}


def build_data_from_atoms(atoms, model, device="cpu", dtype: torch.dtype | None = None):
    """Build a data_dict for Wrapper.forward() from an ASE Atoms object."""
    device = torch.device(device)
    if dtype is None:
        dtype = _model_float_dtype(model)
    pos = torch.tensor(atoms.get_positions(), dtype=dtype, device=device)
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    r_max = float(model.r_max)
    atomic_number_table = [int(z) for z in model.atomic_numbers]

    node_attrs = _one_hot_node_attrs(Z, atomic_number_table)
    edge_index, shifts = _radius_graph_no_pbc(pos, r_max)

    N = pos.size(0)
    batch = torch.zeros(N, dtype=torch.int64, device=device)
    cell = torch.zeros(3, 3, dtype=dtype, device=device)
    charge = torch.zeros(N, dtype=dtype, device=device)
    dipole = torch.zeros(1, 3, dtype=dtype, device=device)
    energy = torch.tensor([0.0], dtype=dtype, device=device)
    energy_weight = torch.tensor([0.0], dtype=dtype, device=device)
    force = torch.zeros(N, 3, dtype=dtype, device=device)
    forces_weight = torch.tensor([0.0], dtype=dtype, device=device)
    ptr = torch.tensor([0, N], dtype=torch.int64, device=device)
    stress = torch.zeros(1, 3, 3, dtype=dtype, device=device)
    stress_weight = torch.tensor([0.0], dtype=dtype, device=device)
    unit_shifts = torch.zeros(edge_index.size(1), 3, dtype=dtype, device=device)
    virials = torch.zeros(1, 3, 3, dtype=dtype, device=device)
    virials_weight = torch.tensor([0.0], dtype=dtype, device=device)
    weight = torch.tensor([1.0], dtype=dtype, device=device)

    data_dict = {
        'batch': batch,
        'cell': cell,
        'charges': charge,
        'dipole': dipole,
        'edge_index': edge_index,
        'energy': energy,
        'energy_weight': energy_weight,
        'forces': force,
        'forces_weight': forces_weight,
        'node_attrs': node_attrs,
        'positions': pos,
        'ptr': ptr,
        'shifts': shifts,
        'stress': stress,
        'stress_weight': stress_weight,
        'unit_shifts': unit_shifts,
        'virials': virials,
        'virials_weight': virials_weight,
        'weight': weight
    }

    local_or_ghost = torch.ones(N, dtype=dtype, device=device)
    return data_dict, local_or_ghost


# ------------------------ Calculator ------------------------

class MACECalculator(CalcABC):
    """ASE-style calculator wrapping a scripted Wrapper MACE model."""

    implemented_properties = ['energy', 'forces', 'free_energy']
    supported_hessian_modes = ("analytic", "numerical")
    supports_batch_energy_forces = True

    def __init__(self, 
        device: torch.device, 
        model: str = 'maceoff23s', 
        model_path: Optional[str] = None,
        overwrite: bool = False,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = 'none',
        ):

        """
        Args:
            device (torch.device): Torch device.
            model (str): Name of the model (expects `<model>.pt` under `model/`).
            model_path (str, optional): Explicit path to the scripted model file.
            overwrite (bool): Whether to overwrite existing models (unused).
        """
        super().__init__()
        self._raw_mace_model = False
        if model in _RAW_UPSTREAM_MODELS and (
            model_path is None or model == "maceomol"
        ):
            # MACE-OFF23 small/large and MACE-OMOL are not shipped as MAPLE
            # TorchScript wrappers in the HF bundle.  Use the official
            # upstream raw checkpoints for those variants while preserving the
            # existing MAPLE TorchScript paths where they exist.
            if model == "maceomol":
                from mace.calculators import mace_omol

                self.model = mace_omol(
                    model=model_path or "extra_large",
                    device=str(device),
                    default_dtype="float64",
                    return_raw_model=True,
                )
            else:
                from mace.calculators import mace_off

                self.model = mace_off(
                    model=_MACE_OFF_SIZE[model],
                    device=str(device),
                    default_dtype="float64",
                    return_raw_model=True,
                )
            self.model = self.model.to(device)
            self._raw_mace_model = True
        else:
            if model_path is None:
                model_dir = os.path.dirname(os.path.realpath(__file__))
                model_dir = os.path.dirname(model_dir)
                model_path = os.path.join(model_dir, 'model', f'{model}.pt')

            # Load the scripted wrapper model
            self.model = torch.jit.load(model_path, map_location=device)

        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = _model_float_dtype(self.model)
        self.overwrite = overwrite

        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]
        self.hessian = "analytic"

        # Initialize implicit solvent
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _augment_raw_data_dict(self, data_dict, atoms_list) -> None:
        """Populate optional graph-level fields expected by raw MACE models."""
        if not self._raw_mace_model:
            return

        atoms_list = list(atoms_list)
        n_graphs = len(atoms_list)
        data_dict["head"] = torch.zeros(
            n_graphs,
            dtype=torch.long,
            device=self.device,
        )

        embedding_specs = getattr(self.model, "embedding_specs", {})
        if "total_charge" in embedding_specs:
            data_dict["total_charge"] = torch.tensor(
                [float(at.info.get("charge", 0.0)) for at in atoms_list],
                dtype=self.dtype,
                device=self.device,
            )
        if "total_spin" in embedding_specs:
            data_dict["total_spin"] = torch.tensor(
                [
                    float(at.info.get("spin", at.info.get("mult", 1.0)))
                    for at in atoms_list
                ],
                dtype=self.dtype,
                device=self.device,
            )

    def _forward_raw_model(self, data_dict, *, compute_force: bool = False, compute_hessian: bool = False):
        return self.model(
            data_dict,
            training=False,
            compute_force=compute_force or compute_hessian,
            compute_virials=False,
            compute_stress=False,
            compute_hessian=compute_hessian,
        )

    def _forward_script_model(self, data_dict, local_or_ghost):
        return self.model.forward(
            data=data_dict,
            local_or_ghost=local_or_ghost,
            compute_virials=False,
        )

    def _energy_vector(self, energy_out, *, batch_size: int, n_atoms_total: int, batch: torch.Tensor) -> torch.Tensor:
        return energy_vector_from_output(
            energy_out,
            batch_size=batch_size,
            n_atoms_total=n_atoms_total,
            batch=batch,
        )

    def calculate(self, atoms=None, properties=['energy','forces'], system_changes=all_changes):
        """Main ASE calculation entry point."""
        super().calculate(atoms, properties, system_changes)

        want_forces = 'forces' in properties
        positions = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=want_forces and not self._raw_mace_model,
        )
        data_dict, local_or_ghost = self._build_graph_inputs(
            atoms, positions=positions
        )
        self._augment_raw_data_dict(data_dict, [atoms])

        # Forward pass. Raw upstream MACE models can return forces directly;
        # the MAPLE scripted wrapper returns energy only, so keep its original
        # exact autograd force path.
        if self._raw_mace_model:
            out = self._forward_raw_model(data_dict, compute_force=want_forces)
            energy_vec = self._energy_vector(
                out["energy"],
                batch_size=1,
                n_atoms_total=data_dict["positions"].shape[0],
                batch=data_dict["batch"],
            )
            ml_energy = energy_vec.sum() * EV2HARTREE
        else:
            total_energy_local = self._forward_script_model(data_dict, local_or_ghost)
            ml_energy = total_energy_local.sum() * EV2HARTREE
        energy = ml_energy

        solvent_force = None
        if self.solvent_correction:
            if want_forces:
                solvent_energy, solvent_force = self.implicit_solv_energy_and_force(atoms)
            else:
                solvent_energy = self.implicit_solv_energy(atoms)
            energy += solvent_energy

        self.results['energy'] = energy.item()
        self.results['free_energy'] = energy.item()

        # Compute forces if requested
        if want_forces:
            if self._raw_mace_model:
                forces = out["forces"] * EV2HARTREE
            else:
                forces = -torch.autograd.grad(
                    ml_energy,
                    data_dict['positions'],
                    create_graph=False,
                    retain_graph=False
                )[0]

            if self.solvent_correction:
                forces += solvent_force

            self.results['forces'] = forces.detach().cpu().numpy()

        if "hessian" in properties:
            if self.solvent_correction:
                raise NotImplementedError("Hessian calculation with implicit solvent is not implemented yet.")
            self.results["hessian"] = self.get_hessian(atoms)

    def calculate_many(self, atoms_list, properties=("energy", "forces")) -> BatchResult:
        """Evaluate independent MACE structures as one disconnected graph batch."""
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

        data_dict, local_or_ghost, counts = build_mace_data_dict_batch(
            atoms_list,
            atomic_numbers=self.atomic_numbers,
            r_max=self.r_max,
            device=self.device,
            dtype=self.dtype,
            requires_grad=want_forces and not self._raw_mace_model,
        )
        self._augment_raw_data_dict(data_dict, atoms_list)

        if self._raw_mace_model:
            out = self._forward_raw_model(data_dict, compute_force=want_forces)
            total_energy_local = out["energy"]
        elif want_forces:
            total_energy_local = self._forward_script_model(data_dict, local_or_ghost)
        else:
            with torch.no_grad():
                total_energy_local = self._forward_script_model(data_dict, local_or_ghost)

        energy_vec = self._energy_vector(
            total_energy_local,
            batch_size=len(atoms_list),
            n_atoms_total=data_dict["positions"].shape[0],
            batch=data_dict["batch"],
        ) * EV2HARTREE

        energies = (
            energy_vec.detach().cpu().numpy().astype(np.float64)
            if want_energy else None
        )

        forces_list = None
        if want_forces:
            if self._raw_mace_model:
                forces = out["forces"] * EV2HARTREE
            else:
                forces = -torch.autograd.grad(
                    energy_vec.sum(),
                    data_dict["positions"],
                    create_graph=False,
                    retain_graph=False,
                )[0]
            forces_list = split_atomwise_array(
                forces.detach().cpu().numpy().astype(np.float64),
                counts,
            )

        return BatchResult(energies=energies, forces=forces_list)

    def get_energy(self, atoms) -> torch.Tensor:
        """Compute total energy as a torch scalar in eV."""
        data_dict, local_or_ghost = build_data_from_atoms(
            atoms, self.model, device=self.device, dtype=self.dtype
        )
        if self._raw_mace_model:
            self._augment_raw_data_dict(data_dict, [atoms])
            out = self._forward_raw_model(data_dict)
            energy = self._energy_vector(
                out["energy"],
                batch_size=1,
                n_atoms_total=data_dict["positions"].shape[0],
                batch=data_dict["batch"],
            )
            return energy.sum()
        total_energy_local = self._forward_script_model(data_dict, local_or_ghost)
        return total_energy_local.sum()

    @staticmethod
    def compute_hessian(
        coords: torch.Tensor,
        energy: torch.Tensor,
        batch_size=None,
    ) -> torch.Tensor:
        """Compute the Hessian matrix (3N x 3N) by second derivatives."""
        num_atoms = coords.shape[0]
        return hessian_loop(
            energy,
            coords,
            output_dof=3 * num_atoms,
            input_dof=3 * num_atoms,
            batch_size=batch_size,
        )

    def _build_graph_inputs(self, atoms, positions: Optional[torch.Tensor] = None):
        if positions is None:
            positions = torch.tensor(
                atoms.get_positions(), dtype=self.dtype, device=self.device
            )
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        node_attrs = _one_hot_node_attrs(Z, self.atomic_numbers, dtype=self.dtype)
        edge_index, shifts = _radius_graph_no_pbc(positions, self.r_max)
        N = positions.size(0)
        batch = torch.zeros(N, dtype=torch.int64, device=self.device)
        cell = torch.zeros(3, 3, dtype=self.dtype, device=self.device)
        charge = torch.zeros(N, dtype=self.dtype, device=self.device)
        dipole = torch.zeros(1, 3, dtype=self.dtype, device=self.device)
        energy = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        energy_weight = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        force = torch.zeros(N, 3, dtype=self.dtype, device=self.device)
        forces_weight = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        ptr = torch.tensor([0, N], dtype=torch.int64, device=self.device)
        stress = torch.zeros(1, 3, 3, dtype=self.dtype, device=self.device)
        stress_weight = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        unit_shifts = torch.zeros(edge_index.size(1), 3, dtype=self.dtype, device=self.device)
        virials = torch.zeros(1, 3, 3, dtype=self.dtype, device=self.device)
        virials_weight = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        weight = torch.tensor([1.0], dtype=self.dtype, device=self.device)

        data_dict = {
            'batch': batch,
            'cell': cell,
            'charges': charge,
            'dipole': dipole,
            'edge_index': edge_index,
            'energy': energy,
            'energy_weight': energy_weight,
            'forces': force,
            'forces_weight': forces_weight,
            'node_attrs': node_attrs,
            'positions': positions,
            'ptr': ptr,
            'shifts': shifts,
            'stress': stress,
            'stress_weight': stress_weight,
            'unit_shifts': unit_shifts,
            'virials': virials,
            'virials_weight': virials_weight,
            'weight': weight
        }
        local_or_ghost = torch.ones(N, dtype=self.dtype, device=self.device)
        return data_dict, local_or_ghost

    def _get_hessian_analytic(self, atoms=None) -> np.ndarray:
        """Compute the Hessian matrix for an ASE Atoms object."""
        batch_size = getattr(self, "hessian_batch_size", getattr(self, "batch_size", None))
        if self._raw_mace_model and batch_size is None:
            data_dict, _ = self._build_graph_inputs(atoms)
            self._augment_raw_data_dict(data_dict, [atoms])
            out = self._forward_raw_model(data_dict, compute_hessian=True)
            hessian = out["hessian"].reshape(3 * len(atoms), 3 * len(atoms))
            return (hessian * EV2HARTREE).detach().cpu().numpy()

        positions = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True
        )

        data_dict, local_or_ghost = self._build_graph_inputs(atoms, positions=positions)
        if self._raw_mace_model:
            self._augment_raw_data_dict(data_dict, [atoms])
            out = self._forward_raw_model(data_dict)
            energy = self._energy_vector(
                out["energy"],
                batch_size=1,
                n_atoms_total=data_dict["positions"].shape[0],
                batch=data_dict["batch"],
            ).sum() * EV2HARTREE
        else:
            total_energy_local = self._forward_script_model(data_dict, local_or_ghost)
            energy = total_energy_local.sum() * EV2HARTREE

        hessian = self.compute_hessian(positions, energy, batch_size=batch_size)
        return hessian.detach().cpu().numpy()

    def _get_hessian_numerical(self, atoms, delta: float = 0.002) -> np.ndarray:
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

    def get_hessian(self, atoms=None, delta: float = 0.002) -> np.ndarray:
        if self.hessian == "analytic":
            return self._get_hessian_analytic(atoms)
        if self.hessian == "numerical":
            return self._get_hessian_numerical(atoms, delta)
        raise ValueError(
            f"Unknown hessian method: {self.hessian}. Must be 'analytic' or 'numerical'"
        )
