from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Optional

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from .._batch_types import BatchResult
from .._batch_utils import (
    atoms_list_has_pbc,
    empty_batch_result,
    normalize_energy_forces_request,
    sequential_calculate_many,
    split_atomwise_array,
)
from ..calculator_base import (
    EV2HARTREE,
    CalcABC,
    hessian_via_double_autograd,
    register_calculator,
)
from ..analytic_protocol import (
    AnalyticIdentityError,
    AnalyticOperationResult,
    CheckpointFileIdentity,
    ScalarCellIdentity,
    ScalarPointIdentity,
    file_stat_identity,
    freeze_identity_options,
    geometry_sha256,
    named_array_sha256,
    require_no_amp_tf32,
    torch_model_runtime_identity,
    torch_response_settings,
)
from ._batch_graph import build_mace_data_dict_batch, energy_vector_from_output
from ._common import (
    NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM,
    NEIGHBOR_CUTOFF_GUARD_POLICY,
    guarded_radius_graph_no_pbc,
    model_float_dtype,
    one_hot_node_attrs,
    radius_graph_no_pbc,
)


_MACEOFF23M_ANALYTIC_CHECKPOINT_SHA256 = (
    "ac172fdf9b5173fef4c64667739dbd06f230b0167b4ae2c67e8c08033254c9ee"
)


@dataclass(frozen=True)
class _MACEAnalyticContext:
    data: dict[str, torch.Tensor]
    local_or_ghost: torch.Tensor
    identity: ScalarPointIdentity


# ------------------------ Data builder ------------------------

def build_data_from_atoms(
    atoms,
    model,
    device='cpu',
    positions: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    edge_index: Optional[torch.Tensor] = None,
    shifts: Optional[torch.Tensor] = None,
):
    """Build a data_dict for Wrapper.forward() from an ASE Atoms object.

    Pass an explicit ``positions`` tensor (e.g. with ``requires_grad=True``) to
    reuse the graph for autograd forces/Hessian; otherwise positions are read
    from ``atoms``.
    """
    device = torch.device(device)
    dtype = dtype or model_float_dtype(model)
    if positions is None:
        pos = torch.tensor(atoms.get_positions(), dtype=dtype, device=device)
    else:
        pos = positions
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    r_max = float(model.r_max)
    atomic_number_table = [int(z) for z in model.atomic_numbers]

    dtype = pos.dtype
    node_attrs = one_hot_node_attrs(Z, atomic_number_table, dtype=dtype)
    if (edge_index is None) != (shifts is None):
        raise ValueError('edge_index and shifts must be supplied together.')
    if edge_index is None:
        edge_index, shifts = radius_graph_no_pbc(pos, r_max)

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

@register_calculator
class MACECalculator(CalcABC):
    """ASE-style calculator wrapping a scripted Wrapper MACE model."""

    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian']

    MODEL_NAMES = ('maceoff23s', 'maceoff23m', 'maceoff23l', 'egret')
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False
    SUPPORTS_IMPLICIT_SOLVATION = True
    # Only the auto-downloaded variants. maceoff23s and maceoff23l are
    # local-only (REQUIRES_LOCAL_MODEL_FILE) — the factory falls back to
    # _require_local_model_file when CHECKPOINT_FILENAME has no entry.
    CHECKPOINT_FILENAME = {'maceoff23m': 'maceoff23m.pt', 'egret': 'egret1s.pt'}
    REQUIRES_LOCAL_MODEL_FILE = True
    OPTION_KEYS = ()
    MODEL_PATH_OPTION = 'model_path'
    supports_batch_energy_forces = True

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        if resolved_model_path is not None:
            kwargs['model_path'] = resolved_model_path
        return kwargs

    def __init__(self,
        device: torch.device,
        model: str = 'maceoff23s',
        model_path: Optional[str] = None,
        overwrite: bool = False,
        implicit: Literal['gbsa', 'none'] = 'none',
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
        self.model_name = str(model)
        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            model_path = os.path.join(model_dir, 'model', f'{model}.pt')

        self.device = torch.device(device)
        self._checkpoint_path = os.path.realpath(model_path)
        loaded_checkpoint_stat = file_stat_identity(
            os.stat(self._checkpoint_path)
        )
        self.model = torch.jit.load(
            self._checkpoint_path,
            map_location=self.device,
        )
        self.model.eval()
        self._checkpoint_identity = CheckpointFileIdentity.bind(
            self._checkpoint_path,
            expected_stat_identity=loaded_checkpoint_stat,
        )
        self._checkpoint_sha256 = self._checkpoint_identity.sha256
        self._loaded_model_object_id = id(self.model)

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.dtype = model_float_dtype(self.model)
        self.overwrite = overwrite

        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]
        self._analytic_bound_r_max = self.r_max
        self._analytic_bound_atomic_numbers = tuple(self.atomic_numbers)
        self._analytic_model_runtime_identity = torch_model_runtime_identity(
            self.model
        )
        self.hessian = 'analytic'

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Main ASE calculation entry point."""
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        if (
            ('hessian' in properties or self.analytic_composition_admission is not None)
            and self.model_name == 'maceoff23m'
            and self._checkpoint_sha256
            == _MACEOFF23M_ANALYTIC_CHECKPOINT_SHA256
        ):
            gas_energy_forces = self._analytic_energy_forces_operation(
                atoms,
                need_forces='forces' in properties,
            )
            gas_hessian = (
                self._analytic_hessian_operation(atoms)
                if 'hessian' in properties
                else None
            )
            self._finalize_results(
                atoms,
                gas_energy_forces=gas_energy_forces,
                gas_hessian=gas_hessian,
            )
            return

        # Single forward; positions carry grad only when forces are requested.
        needs_forces = 'forces' in properties
        positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device,
            requires_grad=needs_forces,
        )
        data_dict, local_or_ghost = build_data_from_atoms(
            atoms, self.model, device=self.device, positions=positions, dtype=self.dtype
        )
        total_energy_local = self.model.forward(
            data=data_dict, local_or_ghost=local_or_ghost, compute_virials=False
        )
        energy_eV = total_energy_local.sum()

        forces_np = None
        if needs_forces:
            forces = -torch.autograd.grad(energy_eV, positions)[0]
            forces_np = forces.detach().cpu().numpy()

        hessian = None
        if 'hessian' in properties:
            if self.solvent_correction is not None:
                raise NotImplementedError('Hessian calculation with implicit solvent is not implemented yet.')
            hessian = self.get_hessian(atoms)

        self._finalize_results(atoms, energy=energy_eV.item(), forces=forces_np, hessian=hessian)

    def calculate_many(
        self,
        atoms_list,
        properties=("energy", "forces"),
    ) -> BatchResult:
        """Evaluate independent molecules as one disconnected MACE graph.

        Each structure's radius graph is built separately before concatenation,
        so different conformers cannot acquire artificial cross-image edges.
        PBC and attached implicit-solvent corrections retain the validated
        sequential path.
        """
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

        data_dict, local_or_ghost, counts = build_mace_data_dict_batch(
            atoms_list,
            atomic_numbers=self.atomic_numbers,
            r_max=self.r_max,
            device=self.device,
            dtype=self.dtype,
            requires_grad=want_forces,
        )
        if want_forces:
            total_energy_local = self.model.forward(
                data=data_dict,
                local_or_ghost=local_or_ghost,
                compute_virials=False,
            )
        else:
            with torch.no_grad():
                total_energy_local = self.model.forward(
                    data=data_dict,
                    local_or_ghost=local_or_ghost,
                    compute_virials=False,
                )

        energy_vector_eV = energy_vector_from_output(
            total_energy_local,
            batch_size=len(atoms_list),
            n_atoms_total=data_dict['positions'].shape[0],
            batch=data_dict['batch'],
        )
        energies = (
            energy_vector_eV.detach().cpu().numpy().astype(np.float64)
            * EV2HARTREE
            if want_energy
            else None
        )

        forces_list = None
        if want_forces:
            forces_eV = -torch.autograd.grad(
                energy_vector_eV.sum(),
                data_dict['positions'],
            )[0]
            forces_hartree = (
                forces_eV.detach().cpu().numpy().astype(np.float64)
                * EV2HARTREE
            )
            forces_list = split_atomwise_array(forces_hartree, counts)

        return BatchResult(energies=energies, forces=forces_list)

    def prepare_analytic_derivatives(self) -> None:
        """Retain native precision; exact qualification is factory-owned."""
        self._checkpoint_identity.require_unchanged()
        require_no_amp_tf32(self.device)

    def _require_analytic_candidate(self, atoms) -> None:
        """Fail closed outside the measured MACE-OFF23m wrapper cell."""

        self._checkpoint_identity.require_unchanged()
        if id(self.model) != self._loaded_model_object_id:
            raise AnalyticIdentityError(
                'The MACE model object changed after checkpoint binding.'
            )
        if (
            torch_model_runtime_identity(self.model)
            != self._analytic_model_runtime_identity
        ):
            raise AnalyticIdentityError(
                'The MACE live parameter/buffer identity changed after binding.'
            )
        if self.model_name != 'maceoff23m':
            raise AnalyticIdentityError(
                'Only MACE-OFF23m is the measured analytic wrapper candidate; '
                'other MACE checkpoints require separate evidence.'
            )
        if self._checkpoint_sha256 != _MACEOFF23M_ANALYTIC_CHECKPOINT_SHA256:
            raise AnalyticIdentityError(
                'The MACE-OFF23m checkpoint SHA256 is not the frozen candidate.'
            )
        if self.dtype is not torch.float64:
            raise AnalyticIdentityError(
                'The MACE-OFF23m analytic candidate must retain native '
                'torch.float64.'
            )
        if self.r_max != self._analytic_bound_r_max:
            raise AnalyticIdentityError(
                'The MACE radius-graph cutoff changed after checkpoint binding.'
            )
        live_atomic_numbers = tuple(int(z) for z in self.model.atomic_numbers)
        if (
            tuple(self.atomic_numbers) != self._analytic_bound_atomic_numbers
            or live_atomic_numbers != self._analytic_bound_atomic_numbers
        ):
            raise AnalyticIdentityError(
                'The MACE atomic-number table changed after checkpoint binding.'
            )
        charge = float(atoms.info.get('charge', 0.0))
        multiplicity = float(atoms.info.get('mult', 1.0))
        if charge != 0.0 or multiplicity != 1.0:
            raise AnalyticIdentityError(
                'The MACE-OFF23m wrapper candidate is neutral singlet only.'
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
            component='gas',
            implementation='maple-mace-scripted-wrapper-v1',
            model=self.model_name,
            checkpoint_path=self._checkpoint_path,
            checkpoint_sha256=self._checkpoint_sha256,
            device=str(self.device),
            effective_dtype=str(self.dtype),
            d4=False,
            source_energy_unit='eV',
            cutoff_guard_policy=NEIGHBOR_CUTOFF_GUARD_POLICY,
            cutoff_guard_margin_angstrom=(
                NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM
            ),
            options=freeze_identity_options(
                {
                    'atomic_numbers': self.atomic_numbers,
                    'charge_multiplicity_support': False,
                    'compute_virials': False,
                    'model_floating_dtypes': floating_dtypes,
                    'neighbor_builder': 'maple-radius-graph-no-pbc-v1',
                    'pbc': False,
                }
            ),
        )

    def _analytic_response_settings(self):
        return torch_response_settings(self.device)

    def _build_analytic_context(
        self,
        atoms,
        *,
        enforce_cutoff_guard: bool,
    ) -> _MACEAnalyticContext:
        self._require_analytic_candidate(atoms)
        positions = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        if enforce_cutoff_guard:
            graph = guarded_radius_graph_no_pbc(positions, self.r_max)
            edge_index = graph.edge_index
            shifts = graph.shifts
            minimum_gap = graph.minimum_cutoff_gap_angstrom
        else:
            edge_index, shifts = radius_graph_no_pbc(positions, self.r_max)
            # Identity still records the actual distance from this fixed graph's
            # cutoff; EF construction itself does not apply the analytic guard.
            atom_count = int(positions.shape[0])
            minimum_gap = float('inf')
            if atom_count > 1:
                pair_indices = torch.triu_indices(
                    atom_count,
                    atom_count,
                    offset=1,
                    device=positions.device,
                )
                distances = torch.linalg.vector_norm(
                    (
                        positions[pair_indices[0]]
                        - positions[pair_indices[1]]
                    ).detach(),
                    dim=1,
                )
                if not torch.isfinite(distances).all():
                    raise AnalyticIdentityError(
                        'MACE analytic pair distances must be finite.'
                    )
                minimum_gap = float(
                    torch.min(torch.abs(distances - self.r_max)).detach().cpu()
                )
        data_dict, local_or_ghost = build_data_from_atoms(
            atoms,
            self.model,
            device=self.device,
            positions=positions,
            dtype=self.dtype,
            edge_index=edge_index,
            shifts=shifts,
        )
        graph_sha256 = named_array_sha256(
            {
                'edge_index': edge_index.detach().cpu().numpy(),
                'shifts': shifts.detach().cpu().numpy(),
                'unit_shifts': data_dict['unit_shifts'].detach().cpu().numpy(),
            }
        )
        identity = ScalarPointIdentity(
            cell=self._analytic_cell_identity(),
            geometry_sha256=geometry_sha256(atoms),
            atomic_numbers=tuple(
                int(value) for value in atoms.get_atomic_numbers()
            ),
            charge=float(atoms.info.get('charge', 0.0)),
            multiplicity=int(atoms.info.get('mult', 1)),
            atom_topology_sha256=None,
            neighbor_graph_sha256=graph_sha256,
            active_cutoffs_angstrom=(('radius_graph', float(self.r_max)),),
            minimum_cutoff_gap_angstrom=minimum_gap,
            response_settings=self._analytic_response_settings(),
        )
        return _MACEAnalyticContext(
            data=data_dict,
            local_or_ghost=local_or_ghost,
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
            total_energy_local = self.model.forward(
                data=context.data,
                local_or_ghost=context.local_or_ghost,
                compute_virials=False,
            )
            energy_eV = total_energy_local.sum()
            forces = None
            if need_forces:
                gradient_eV = torch.autograd.grad(
                    energy_eV,
                    context.data['positions'],
                )[0]
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
            total_energy_local = self.model.forward(
                data=context.data,
                local_or_ghost=context.local_or_ghost,
                compute_virials=False,
            )
            energy_hartree = total_energy_local.sum() * EV2HARTREE
            gradient = torch.autograd.grad(
                energy_hartree,
                context.data['positions'],
                create_graph=True,
            )[0]
            rows = [
                torch.autograd.grad(
                    component,
                    context.data['positions'],
                    retain_graph=True,
                )[0].reshape(-1)
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
                f'MACE HVP direction must be a finite ({expected_size},) array.'
            )
        with torch.enable_grad():
            context = self._build_analytic_context(
                atoms,
                enforce_cutoff_guard=True,
            )
            total_energy_local = self.model.forward(
                data=context.data,
                local_or_ghost=context.local_or_ghost,
                compute_virials=False,
            )
            energy_hartree = total_energy_local.sum() * EV2HARTREE
            gradient = torch.autograd.grad(
                energy_hartree,
                context.data['positions'],
                create_graph=True,
            )[0]
            direction_tensor = torch.as_tensor(
                direction_array,
                dtype=self.dtype,
                device=self.device,
            ).reshape_as(gradient)
            hvp = torch.autograd.grad(
                torch.sum(gradient * direction_tensor),
                context.data['positions'],
            )[0]
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
            self.model_name == 'maceoff23m'
            and self._checkpoint_sha256
            == _MACEOFF23M_ANALYTIC_CHECKPOINT_SHA256
            and float(atoms.info.get('charge', 0.0)) == 0.0
            and float(atoms.info.get('mult', 1.0)) == 1.0
        ):
            return np.asarray(
                self._analytic_hessian_operation(
                    atoms
                ).hessian_hartree_per_angstrom2
            )
        positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device, requires_grad=True
        )
        data_dict, local_or_ghost = build_data_from_atoms(
            atoms, self.model, device=self.device, positions=positions, dtype=self.dtype
        )

        def energy_fn():
            total_energy_local = self.model.forward(
                data=data_dict, local_or_ghost=local_or_ghost, compute_virials=False
            )
            return total_energy_local.sum() * EV2HARTREE

        return hessian_via_double_autograd(energy_fn, positions)
