from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Literal

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC, register_calculator


AIMNET2_RAW_CHARGE_TOLERANCE_E = 1.0e-4
AIMNET2_PADDED_SENTINEL_TOLERANCE_E = 1.0e-6


@dataclass(frozen=True)
class AIMNet2ChargeState:
    """One audited AIMNet2 atom-centred point-charge prediction.

    AIMNet2 applies neural charge equilibration internally.  The checkpoint
    used by MAPLE emits float32 atomic charges, so a tiny uniform affine
    projection removes only floating-point residue from the declared molecular
    charge.  A material mismatch is rejected rather than silently normalized.
    """

    energy_ev: float
    raw_charges_e: np.ndarray
    charges_e: np.ndarray
    requested_total_charge_e: float
    raw_charge_residual_e: float
    charge_projection_per_atom_e: float
    model_name: str

    def __post_init__(self) -> None:
        raw = np.array(self.raw_charges_e, dtype=float, copy=True)
        charges = np.array(self.charges_e, dtype=float, copy=True)
        if (
            raw.ndim != 1
            or raw.size == 0
            or charges.shape != raw.shape
            or not np.all(np.isfinite(raw))
            or not np.all(np.isfinite(charges))
        ):
            raise ValueError(
                "AIMNet2 charge arrays must be finite non-empty vectors with "
                "matching shapes."
            )
        scalars = (
            self.energy_ev,
            self.requested_total_charge_e,
            self.raw_charge_residual_e,
            self.charge_projection_per_atom_e,
        )
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("AIMNet2 charge-state scalars must be finite.")
        if not self.model_name:
            raise ValueError("AIMNet2 charge state requires a model name.")

        expected_residual = (
            float(np.sum(raw)) - float(self.requested_total_charge_e)
        )
        if abs(expected_residual - self.raw_charge_residual_e) > 1.0e-12:
            raise ValueError(
                "AIMNet2 raw charge residual is inconsistent with the stored "
                "atomic charges."
            )
        expected = raw + float(self.charge_projection_per_atom_e)
        if not np.allclose(charges, expected, rtol=0.0, atol=2.0e-15):
            raise ValueError(
                "AIMNet2 projected charges do not match the declared affine "
                "charge correction."
            )
        if abs(float(np.sum(charges)) - self.requested_total_charge_e) > 1.0e-12:
            raise ValueError(
                "AIMNet2 projected charges do not conserve the requested "
                "molecular charge."
            )

        raw.setflags(write=False)
        charges.setflags(write=False)
        object.__setattr__(self, "raw_charges_e", raw)
        object.__setattr__(self, "charges_e", charges)

    @property
    def raw_charge_sum_e(self) -> float:
        return float(np.sum(self.raw_charges_e))

    @property
    def projected_charge_sum_e(self) -> float:
        return float(np.sum(self.charges_e))

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "charge_model": self.model_name,
            "charge_output": "aimnet2-nqe-atom-centred-point-charges",
            "raw_charge_residual_e": self.raw_charge_residual_e,
            "raw_charge_tolerance_e": AIMNET2_RAW_CHARGE_TOLERANCE_E,
            "charge_projection_per_atom_e": (
                self.charge_projection_per_atom_e
            ),
            "charge_projection": "uniform-affine-float-residue-only",
            "requested_total_charge_e": self.requested_total_charge_e,
        }


@dataclass(frozen=True)
class AIMNet2ChargePositionResponse:
    """Coordinate derivatives for one AIMNet2 energy/charge evaluation.

    ``charge_position_vjp_ev_per_angstrom`` is the derivative of
    ``sum_i q_i * v_i`` for the supplied charge cotangent ``v``.  The tiny
    affine total-charge projection is kept inside the autograd graph so this
    response remains in the fixed-total-charge tangent space.
    """

    charge_state: AIMNet2ChargeState
    charge_cotangent_ev_per_e: np.ndarray
    intrinsic_energy_gradient_ev_per_angstrom: np.ndarray
    charge_position_vjp_ev_per_angstrom: np.ndarray

    def __post_init__(self) -> None:
        atom_count = self.charge_state.charges_e.size
        cotangent = np.asarray(
            self.charge_cotangent_ev_per_e,
            dtype=float,
        ).copy()
        intrinsic = np.asarray(
            self.intrinsic_energy_gradient_ev_per_angstrom,
            dtype=float,
        ).copy()
        charge_vjp = np.asarray(
            self.charge_position_vjp_ev_per_angstrom,
            dtype=float,
        ).copy()
        if cotangent.shape != (atom_count,) or not np.all(
            np.isfinite(cotangent)
        ):
            raise ValueError(
                "AIMNet2 charge cotangent must be finite with shape "
                f"{(atom_count,)}."
            )
        expected_gradient_shape = (atom_count, 3)
        for name, values in (
            ("intrinsic energy gradient", intrinsic),
            ("charge-position VJP", charge_vjp),
        ):
            if values.shape != expected_gradient_shape or not np.all(
                np.isfinite(values)
            ):
                raise ValueError(
                    f"AIMNet2 {name} must be finite with shape "
                    f"{expected_gradient_shape}."
                )
        cotangent.setflags(write=False)
        intrinsic.setflags(write=False)
        charge_vjp.setflags(write=False)
        object.__setattr__(
            self,
            "charge_cotangent_ev_per_e",
            cotangent,
        )
        object.__setattr__(
            self,
            "intrinsic_energy_gradient_ev_per_angstrom",
            intrinsic,
        )
        object.__setattr__(
            self,
            "charge_position_vjp_ev_per_angstrom",
            charge_vjp,
        )


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
    implemented_properties = [
        'energy',
        'forces',
        'free_energy',
        'hessian',
        'charges',
    ]

    MODEL_NAMES = ('aimnet2', 'aimnet2nse')
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False
    SUPPORTED_COULOMB_METHODS = ('simple', 'dsf')
    CHECKPOINT_FILENAME = {'aimnet2': 'aimnet2.pt', 'aimnet2nse': 'aimnet2nse.pt'}
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ('coulomb_method',)
    MODEL_PATH_OPTION = 'model_path'

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        coulomb_method = options.get('coulomb_method')
        if coulomb_method is not None:
            kwargs['coulomb_method'] = str(coulomb_method).lower()
        if resolved_model_path is not None:
            kwargs['model_path'] = resolved_model_path
        return kwargs

    def __init__(self, device: torch.device,
                model: str = 'aimnet2',
                model_path: str = None,
                coulomb_method: str = 'simple',
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
        self.model_name = str(model)
        self.model_path = os.fspath(model_path)
        self._supports_atom_charges = self.model_name == 'aimnet2'
        if not self._supports_atom_charges:
            self.implemented_properties = [
                name
                for name in type(self).implemented_properties
                if name != 'charges'
            ]
        self.model = torch.jit.load(model_path, map_location=device).eval()

        self.cutoff = float(getattr(self.model, 'cutoff'))
        self.cutoff_lr = float(getattr(self.model, 'cutoff_lr', float('inf')))
        # Always provide nbmat_lr to avoid TorchScript KeyError; method routing
        # handles cutoff_lr.
        self.lr = True
        self.hessian: str = 'analytic'
        self._last_charge_state: AIMNet2ChargeState | None = None

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
        output = self._forward_output(data)
        energy_eV = self._energy_from_output(output)
        charge_state = None
        if 'charges' in properties:
            self._validate_charge_output_domain(atoms)
            charge_state = self._charge_state_from_output(
                output,
                atom_count=len(atoms),
                requested_total_charge_e=self._total_charge_from_atoms(atoms),
            )

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
        if charge_state is None:
            self._last_charge_state = None
        else:
            self._last_charge_state = charge_state
            self.results['charges'] = charge_state.charges_e.copy()

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

    def _forward_output(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return one raw checkpoint output without changing its tensors."""

        with torch.jit.optimized_execution(False):
            return self.model(data)

    @staticmethod
    def _energy_from_output(
        output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        energy = output.get('energy')
        if energy is None or not torch.isfinite(energy).all():
            raise RuntimeError(
                "AIMNet2 did not return a finite energy tensor."
            )
        return energy.sum()

    def _forward_energy(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pure model forward; returns energy in eV (model's native unit)."""

        return self._energy_from_output(self._forward_output(data))

    def _charge_state_from_output(
        self,
        output: Dict[str, torch.Tensor],
        *,
        atom_count: int,
        requested_total_charge_e: float,
    ) -> AIMNet2ChargeState:
        if atom_count <= 0:
            raise ValueError(
                "AIMNet2 charge prediction requires at least one atom."
            )
        raw_tensor = output.get('charges')
        if raw_tensor is None:
            raise RuntimeError(
                "This AIMNet2 checkpoint does not expose atom-centred charges."
            )
        if raw_tensor.ndim != 1 or raw_tensor.numel() not in {
            atom_count,
            atom_count + 1,
        }:
            raise RuntimeError(
                "AIMNet2 charges must be a one-dimensional tensor containing "
                f"{atom_count} atoms and at most one padded sentinel; received "
                f"shape {tuple(raw_tensor.shape)}."
            )
        raw_with_padding = (
            raw_tensor.detach().to(device='cpu', dtype=torch.float64).numpy()
        )
        if not np.all(np.isfinite(raw_with_padding)):
            raise RuntimeError("AIMNet2 returned non-finite atomic charges.")
        if raw_with_padding.size == atom_count + 1:
            sentinel = float(raw_with_padding[-1])
            if abs(sentinel) > AIMNET2_PADDED_SENTINEL_TOLERANCE_E:
                raise RuntimeError(
                    "AIMNet2 returned a nonzero padded sentinel charge "
                    f"({sentinel:.6e} e)."
                )
            raw_charges = raw_with_padding[:-1].copy()
        else:
            raw_charges = raw_with_padding.copy()

        requested = float(requested_total_charge_e)
        if not np.isfinite(requested):
            raise ValueError("The requested molecular charge must be finite.")
        residual = float(np.sum(raw_charges)) - requested
        if abs(residual) > AIMNET2_RAW_CHARGE_TOLERANCE_E:
            raise RuntimeError(
                "AIMNet2 atom-centred charges violate the requested "
                "total-charge constraint "
                f"(sum={float(np.sum(raw_charges)):.9f} e, "
                f"target={requested:.9f} e, residual={residual:.3e} e)."
            )
        correction = -residual / atom_count
        charges = raw_charges + correction

        return AIMNet2ChargeState(
            energy_ev=float(self._energy_from_output(output).detach().cpu()),
            raw_charges_e=raw_charges,
            charges_e=charges,
            requested_total_charge_e=requested,
            raw_charge_residual_e=residual,
            charge_projection_per_atom_e=correction,
            model_name=self.model_name,
        )

    def _validate_charge_output_domain(self, atoms) -> None:
        if not self._supports_atom_charges:
            raise NotImplementedError(
                "MAPLE has not validated the AIMNet2-NSE two-channel charge "
                "representation as a scalar continuum source; use the "
                "closed-shell aimnet2 checkpoint."
            )
        multiplicity = float(atoms.info.get('mult', 1.0))
        if multiplicity != 1.0:
            raise NotImplementedError(
                "The AIMNet2 point-charge-l0 baseline is restricted to "
                "closed-shell multiplicity 1; open-shell charge channels "
                "require a separately validated AIMNet2-NSE adapter."
            )

    def charge_state(self, atoms) -> AIMNet2ChargeState:
        """Evaluate gas-phase AIMNet2 energy and NQE point charges once."""

        self._reject_unsupported_pbc(atoms)
        self._validate_charge_output_domain(atoms)
        coord = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
        )
        data = self._build_data(coord, atoms)
        with torch.no_grad():
            output = self._forward_output(data)
        state = self._charge_state_from_output(
            output,
            atom_count=len(atoms),
            requested_total_charge_e=self._total_charge_from_atoms(atoms),
        )
        self._last_charge_state = state
        return state

    def charge_position_response(
        self,
        atoms,
        charge_cotangent_ev_per_e: np.ndarray,
    ) -> AIMNet2ChargePositionResponse:
        """Differentiate AIMNet2 energy and NQE charges with respect to geometry.

        This is a fixed-geometry response primitive, not an implicit-solvent
        model by itself.  A continuum caller supplies the reaction potential
        as the charge cotangent and remains responsible for the continuum's
        explicit cavity/source coordinate derivative.
        """

        self._reject_unsupported_pbc(atoms)
        self._validate_charge_output_domain(atoms)
        atom_count = len(atoms)
        cotangent = np.array(
            charge_cotangent_ev_per_e,
            dtype=float,
            copy=True,
        )
        if cotangent.shape != (atom_count,) or not np.all(
            np.isfinite(cotangent)
        ):
            raise ValueError(
                "AIMNet2 charge cotangent must be finite with shape "
                f"{(atom_count,)}."
            )

        coord = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        data = self._build_data(coord, atoms)
        output = self._forward_output(data)
        charge_state = self._charge_state_from_output(
            output,
            atom_count=atom_count,
            requested_total_charge_e=self._total_charge_from_atoms(atoms),
        )
        raw_tensor = output.get("charges")
        if raw_tensor is None or not raw_tensor.requires_grad:
            raise RuntimeError(
                "AIMNet2 charge-position response requires differentiable "
                "checkpoint charge outputs."
            )
        raw_atomic = raw_tensor[:atom_count]
        target_charge = torch.as_tensor(
            charge_state.requested_total_charge_e,
            dtype=raw_atomic.dtype,
            device=raw_atomic.device,
        )
        differentiable_charges = raw_atomic + (
            target_charge - raw_atomic.sum()
        ) / atom_count

        energy_ev = self._energy_from_output(output)
        if not energy_ev.requires_grad:
            raise RuntimeError(
                "AIMNet2 charge-position response requires a differentiable "
                "checkpoint energy output."
            )
        intrinsic_gradient = torch.autograd.grad(
            energy_ev,
            data["coord"],
            retain_graph=True,
        )[0][:atom_count]
        cotangent_tensor = torch.as_tensor(
            cotangent,
            dtype=differentiable_charges.dtype,
            device=differentiable_charges.device,
        )
        charge_pairing_ev = torch.sum(
            differentiable_charges * cotangent_tensor
        )
        charge_position_vjp = torch.autograd.grad(
            charge_pairing_ev,
            data["coord"],
        )[0][:atom_count]

        response = AIMNet2ChargePositionResponse(
            charge_state=charge_state,
            charge_cotangent_ev_per_e=cotangent,
            intrinsic_energy_gradient_ev_per_angstrom=(
                intrinsic_gradient.detach().cpu().numpy()
            ),
            charge_position_vjp_ev_per_angstrom=(
                charge_position_vjp.detach().cpu().numpy()
            ),
        )
        self._last_charge_state = charge_state
        return response

    @property
    def last_charge_state(self) -> AIMNet2ChargeState | None:
        return self._last_charge_state

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
