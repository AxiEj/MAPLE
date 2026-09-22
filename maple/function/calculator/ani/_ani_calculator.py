from __future__ import annotations

import hashlib
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
    ANALYTIC_IMPLICIT_HESSIAN_MODELS = ("ani2x",)
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
    OPTION_KEYS = ("d4", "dtype")
    MODEL_PATH_OPTION = "model_path"
    supports_batch_energy_forces = True

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {
            "d4": parse_bool_option(options.get("d4", False), name="d4"),
            "dtype": options.get("dtype", "auto"),
            "hessian": options.get("hessian", "analytic"),
        }
        if resolved_model_path is not None:
            kwargs["model_path"] = resolved_model_path
        return kwargs

    def __init__(
        self,
        device,
        model: str = "ani2x",
        model_path: str | None = None,
        overwrite=False,
        d4=False,
        implicit: str = "none",
        solvent: str = "none",
        *,
        dtype: str = "auto",
        hessian: str = "analytic",
    ):
        import torch

        super().__init__()

        if dtype not in {"auto", "float32", "float64"}:
            raise ValueError(
                f"Unsupported ANI dtype={dtype!r}; expected one of "
                "'auto', 'float32', or 'float64'."
            )
        if hessian not in self.SUPPORTED_HESSIAN_MODES:
            supported = ", ".join(self.SUPPORTED_HESSIAN_MODES)
            raise ValueError(
                f"Unsupported ANI hessian={hessian!r}; supported modes: {supported}."
            )
        if dtype == "float32" and hessian == "numerical":
            raise ValueError(
                "ANI float32 cannot be used for numerical curvature; use "
                "dtype='auto' (automatic promotion) or dtype='float64'."
            )

        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            model_path = os.path.join(model_dir, "model", f"{model}.pt")

        self._original_checkpoint_path = os.path.realpath(model_path)
        checkpoint_stat = os.stat(self._original_checkpoint_path)
        self._loaded_checkpoint_identity = self._stat_identity(checkpoint_stat)
        self._original_checkpoint_sha256 = None
        self._requested_dtype = dtype
        self._numerical_curvature_prepared = False
        self._recommended_numerical_step = None
        self._precision_cast_reason = "checkpoint_native_precision"

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
        self.hessian = hessian

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

        if dtype == "float64":
            self._record_checkpoint_sha256()
            self._promote_to_float64(reason="explicit_dtype_float64")
        if hessian == "numerical":
            self.prepare_numerical_derivatives()

    @staticmethod
    def _stat_identity(stat_result) -> tuple[int, int, int, int]:
        return (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_size,
            stat_result.st_mtime_ns,
        )

    def _record_checkpoint_sha256(self) -> None:
        if self._original_checkpoint_sha256 is not None:
            return
        digest = hashlib.sha256()
        with open(self._original_checkpoint_path, "rb") as handle:
            identity_before = self._stat_identity(os.fstat(handle.fileno()))
            if identity_before != self._loaded_checkpoint_identity:
                raise RuntimeError(
                    "ANI checkpoint file identity changed after the model was loaded; "
                    "refusing to report a SHA for different bytes."
                )
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            identity_after = self._stat_identity(os.fstat(handle.fileno()))
        if identity_after != self._loaded_checkpoint_identity:
            raise RuntimeError(
                "ANI checkpoint file changed while its SHA256 was being computed."
            )
        self._original_checkpoint_sha256 = digest.hexdigest()

    def _promote_to_float64(self, *, reason: str) -> None:
        import torch

        if self.dtype is torch.float64:
            return
        self.model.to(dtype=torch.float64)
        self.dtype = torch.float64
        self._precision_cast_reason = reason
        # Coordinate dtype is part of the evaluated PES. Clear ASE's cached
        # atoms/results whenever it changes so no float32 result can survive
        # into the promoted execution mode.
        self.reset()

    def prepare_numerical_derivatives(self) -> float:
        """Select the validated ANI precision and displacement for curvature."""
        if self._requested_dtype == "float32":
            raise ValueError(
                "ANI float32 cannot be used for numerical curvature; use "
                "dtype='auto' (automatic promotion) or dtype='float64'."
            )
        self._record_checkpoint_sha256()
        self._promote_to_float64(reason="numerical_curvature_requires_float64")
        self._numerical_curvature_prepared = True
        self._recommended_numerical_step = 0.0005
        if self._requested_dtype == "auto":
            self._precision_cast_reason = "numerical_curvature_requires_float64"
        return self._recommended_numerical_step

    def prepare_analytic_derivatives(self) -> None:
        """Promote only an explicitly requested implicit analytic workflow."""
        if self._requested_dtype == "float32":
            raise ValueError(
                "ANI float32 cannot be used for analytic implicit curvature; "
                "use dtype='auto' or dtype='float64'."
            )
        self._record_checkpoint_sha256()
        self._promote_to_float64(
            reason="analytic_implicit_curvature_requires_float64"
        )
        if self._requested_dtype == "auto":
            self._precision_cast_reason = (
                "analytic_implicit_curvature_requires_float64"
            )
        self.analytic_implicit_derivatives_admitted = True

    @property
    def inference_precision_provenance(self) -> dict:
        import torch

        return {
            "requested_dtype": self._requested_dtype,
            "effective_dtype": (
                "float64" if self.dtype is torch.float64 else "float32"
            ),
            "numerical_curvature_prepared": self._numerical_curvature_prepared,
            "recommended_step_angstrom": self._recommended_numerical_step,
            "original_checkpoint_path": self._original_checkpoint_path,
            "original_checkpoint_sha256": self._original_checkpoint_sha256,
            "in_memory_cast_only": self.dtype is torch.float64,
            "reason": self._precision_cast_reason,
        }

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
        import tad_dftd4 as d4
        import torch

        charge = coordinates.new_tensor(0.0)
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
        correction = getattr(self, "solvent_correction", None)
        if correction is not None:
            if self.analytic_implicit_derivatives_admitted is not True:
                raise NotImplementedError(
                    "ANI analytic implicit-solvent derivatives were not prepared."
                )
            directional_fn = getattr(
                correction, "get_directional_derivatives", None
            )
            if not callable(directional_fn):
                raise NotImplementedError(
                    "The attached solvent correction has no admitted directional "
                    "derivative backend; use Dimer use_hvp=false."
                )
            solvent = directional_fn(atoms, np.asarray(n, dtype=np.float64))
            hvp = hvp + torch.as_tensor(
                solvent.hvp_hartree_per_angstrom2,
                dtype=self.dtype,
                device=self.device,
            )
            forces = forces + torch.as_tensor(
                solvent.forces_hartree_per_angstrom.reshape(-1),
                dtype=self.dtype,
                device=self.device,
            )
            energy = energy + torch.as_tensor(
                solvent.energy_hartree,
                dtype=self.dtype,
                device=self.device,
            )
            self.last_hvp_provenance = {
                "composition": "ani-gas-analytic+torch-obc2-directional",
                "solvent": dict(solvent.provenance),
            }
        else:
            self.last_hvp_provenance = {"composition": "ani-gas-analytic"}
        return hvp, forces, energy
