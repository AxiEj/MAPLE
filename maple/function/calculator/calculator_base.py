import torch
import numpy as np

import ase.calculators.calculator

from ._batch_types import BatchResult


class CalcABC(ase.calculators.calculator.Calculator):
    # ----- batch capability flags -----
    # Subclasses that ship a true batched `calculate_many` should flip this
    # to True. The default (False) keeps the sequential fallback below.
    supports_batch_energy_forces: bool = False
    # Subclasses that expose an analytic Hessian via `get_hessian` (and have
    # `self.hessian == 'analytic'` selected) should advertise it here so
    # callers know they can avoid finite-difference paths entirely.
    supports_analytic_hessian: bool = False
    # `CalcABC.get_hvp` below is ANI-style (model(species, coords), self.d4,
    # self.dtype, self.device). It is not a safe generic contract for MACE,
    # UMA, AIMNet2, or MACEPol, so HVP is opt-in only after model-specific
    # validation.
    supports_hvp: bool = False

    # FD Hessian context mode for calculators that override
    # `make_fd_context`. The base class has no reusable graph/neighbor
    # context, so it validates the knob and returns None.
    fd_context_mode: str = "safe"

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Unified batched E / F / H interface (Phase 1 default fallback)
    # ------------------------------------------------------------------
    def calculate_many(self, atoms_list, properties=("energy", "forces")) -> BatchResult:
        """Sequential fallback evaluation of a list of structures.

        This default implementation simply loops over `atoms_list` and calls
        `self.calculate(...)` per structure, capturing the per-call results
        before the next iteration overwrites `self.results`. Subclasses with
        native batched evaluation (e.g. ANI's `(B,N,3)` forward, MACE's
        disconnected-graph batch, AIMNet2's `mol_idx`-keyed batch) override
        this method and set `supports_batch_energy_forces = True`.

        Notes
        -----
        * Return-value driven: the caller never has to read `self.results`.
          This keeps batched paths from clobbering ASE's single-structure
          cache that other modules (frequency, irc, scan) still rely on.
        * Properties default to ``("energy", "forces")`` because that is
          what every Phase 1 caller (FDHessianEvaluator, energy_forces_one,
          PathEvaluator) actually needs.
        """
        props = tuple(properties)
        want_energy = "energy" in props
        want_forces = "forces" in props
        # `hessian` is intentionally not part of the fallback: numerical
        # Hessian assembly is the FDHessianEvaluator's job, not the
        # calculator's. Subclasses that expose batched analytic Hessians
        # can override this method.
        if "hessian" in props:
            raise NotImplementedError(
                "calculate_many fallback does not assemble Hessians; "
                "use FDHessianEvaluator for numerical Hessian or override "
                "calculate_many in a subclass for analytic batch Hessian."
            )

        energies = [] if want_energy else None
        forces_list = [] if want_forces else None

        request = [p for p in props if p in ("energy", "forces")]
        if not request:
            return BatchResult()

        for at in atoms_list:
            self.calculate(
                at,
                properties=list(request),
                system_changes=ase.calculators.calculator.all_changes,
            )
            if want_energy:
                if "free_energy" in self.results:
                    energies.append(float(self.results["free_energy"]))
                else:
                    energies.append(float(self.results["energy"]))
            if want_forces:
                forces_list.append(
                    np.asarray(self.results["forces"], dtype=np.float64)
                )

        return BatchResult(
            energies=np.asarray(energies, dtype=np.float64) if energies is not None else None,
            forces=forces_list,
        )

    def make_fd_context(
        self,
        atoms,
        *,
        delta: float | None = None,
        fd_context_mode: str | None = None,
    ):
        """Optional fixed-topology force-evaluation context for FD Hessians.

        Subclasses can override this factory in Phase 2A model-specific PRs
        to cache reference-geometry invariants (species tensors, graph
        metadata, neighbor lists with cutoff skins, etc.) and expose a
        ``force_at(positions)`` method. Returning ``None`` keeps the Phase 1
        sequential ``calculate_many`` fallback.
        """
        mode = fd_context_mode or getattr(self, "fd_context_mode", "safe")
        if mode not in ("safe", "fast"):
            raise ValueError(
                "fd_context_mode must be 'safe' or 'fast', "
                f"got {mode!r}"
            )
        return None

    def log_error(self, error_message: str) -> None:
        """
        Logs error messages to the output file.

        Args:
            error_message: The error message to log.
        """
        with open(self.output, 'a') as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """
        Logs info messages to the output file.

        Args:
            info_message: The info message to log.
        """
        with open(self.output, 'a') as file:
            for info in info_message:   
                file.write(f"{info}")

    def get_hvp(self, atoms, n: np.ndarray):
        """
        Compute Hessian-vector product Hn for the given atoms and direction n using autograd.
        Args:
            atoms (ase.Atoms): system
            n (np.ndarray): direction vector, shape (3N,)
        Returns:
            Hn (torch.Tensor): Hessian-vector product (3N,) on same device/dtype
            forces (torch.Tensor): forces (3N,) on same device/dtype
            energy (torch.Tensor): scalar total energy
        """
        # 1. prepare coordinates with grad enabled
        coords = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True
        ).unsqueeze(0)

        # 2. atomic numbers
        species = torch.tensor(
            atoms.get_atomic_numbers(),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        # 3. forward pass → energy
        energy = self.model(species, coords)[0]
        if self.d4:
            energy += self.dftd4(species, coords)

        # 4. compute gradient (forces = -grad V)
        grad = torch.autograd.grad(energy, coords, create_graph=True)[0].squeeze(0)  # shape (N,3)
        grad_vec = grad.view(-1)  # (3N,)

        # 5. Hessian-vector product: grad(grad·n)
        n_tensor = torch.tensor(n, dtype=self.dtype, device=self.device)
        hvp = torch.autograd.grad(
            grad_vec @ n_tensor, coords, retain_graph=True
        )[0].squeeze(0).view(-1)  # (3N,)

        # 6. Forces (already computed, negative gradient)
        forces = -grad_vec

        return hvp, forces, energy

    def implicit_solv_init(self, implicit: str, solvent: str):

        if implicit == "gbsa" and solvent != 'none':

            # GBSA solvent correction and QEq charge calculator
            from .extra_correction import GBSA
            from .extra_correction import QEqTorch

            self.solvent_correction = GBSA(solvent=solvent, device=self.device)
        
            self.chargecalc = QEqTorch(device=self.device)
        else:
            self.solvent_correction = None
    
    def implicit_solv_energy(self, atoms: ase.Atoms) -> torch.Tensor:
        """
        Compute implicit solvent correction energy if applicable.

        Args:
            atoms (ase.Atoms): Atomic structure.

        Returns:
            torch.Tensor: Implicit solvent correction energy in Hartree.
        """
        atoms.atomic_charges = self.chargecalc(atoms)
        solvent_energy,_ = self.solvent_correction.get_energy(atoms)
        return solvent_energy

    def implicit_solv_energy_and_force(self, atoms: ase.Atoms) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute implicit solvent correction energy and forces if applicable.

        Args:
            atoms (ase.Atoms): Atomic structure.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Implicit solvent correction energy in Hartree and forces in Hartree/Å.
        """
        atoms.atomic_charges = self.chargecalc(atoms)
        solvent_energy, solvent_forces = self.solvent_correction.get_energy_and_force(atoms)
        return solvent_energy, solvent_forces


class CalcBatchABC(ase.calculators.calculator.Calculator):
    def __init__(self):
        super().__init__()


    def log_error(self, error_message: str) -> None:
        """
        Logs error messages to the output file.

        Args:
            error_message: The error message to log.
        """
        with open(self.output, 'a') as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """
        Logs info messages to the output file.

        Args:
            info_message: The info message to log.
        """
        with open(self.output, 'a') as file:
            for info in info_message:   
                file.write(f"{info}")
