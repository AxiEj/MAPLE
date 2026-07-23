"""Rappe--Goddard charge equilibration with a Gaussian STO approximation.

The original QEq method uses Slater charge densities and treats hydrogen
self-consistently: both its idempotential and its Slater exponent depend on
the current hydrogen charge.  This module uses the documented single-Gaussian
approximation to the Slater Coulomb integral, but retains both hydrogen
updates.  Open Babel's fixed-hydrogen Gaussian shortcut is intentionally not
used.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch


BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988
COULOMB_EV_ANGSTROM = HARTREE_TO_EV * BOHR_TO_ANGSTROM
QEQ_CHARGE_TOL = 1.0e-10
HYDROGEN_CHARGE_MIN = -1.0
HYDROGEN_CHARGE_MAX = 1.0
SUPPORTED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})

_GTO_FIT_COEFFICIENT = {
    1: 0.270917,
    2: 0.098800,
    3: 0.055600,
    4: 0.039100,
    5: 0.029600,
}


class QEqGTO:
    """QEq with GTO Coulomb integrals and full hydrogen self-consistency.

    ``lambda_scale=0.5`` and the atomic parameters follow Rappe and Goddard,
    J. Phys. Chem. 95, 3358 (1991), DOI 10.1021/j100161a070.  The GTO mapping
    follows the analytic single-Gaussian approximation used by Caltech
    ``cheq``.  Unlike ``cheq`` 0.5.1 and Open Babel 3.x, every SCF iteration
    updates both the hydrogen idempotential and all Coulomb integrals that
    contain hydrogen, because the original method defines
    ``zeta_H(q) = zeta_H(0) + q_H``.
    """

    coulomb_constant = COULOMB_EV_ANGSTROM

    def __init__(
        self,
        data_file: str | os.PathLike[str] | None = None,
        *,
        tolerance: float = 1.0e-8,
        max_iterations: int = 2000,
        lambda_scale: float = 0.5,
        damping: float = 0.4,
        hydrogen_scf: bool = True,
        coulomb_scale: float = 1.0,
        supported_elements: frozenset[str] | None = SUPPORTED_ELEMENTS,
    ):
        if data_file is None:
            data_file = Path(__file__).parent / "data" / "qeq.dat"
        if tolerance <= 0:
            raise ValueError("QEq tolerance must be positive.")
        if max_iterations < 1:
            raise ValueError("QEq max_iterations must be at least one.")
        if lambda_scale <= 0:
            raise ValueError("QEq lambda_scale must be positive.")
        if not 0.0 < damping <= 1.0:
            raise ValueError("QEq damping must be in (0, 1].")
        if coulomb_scale <= 0:
            raise ValueError("QEq Coulomb scaling must be positive.")

        self.data_file = Path(data_file)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.lambda_scale = float(lambda_scale)
        self.damping = float(damping)
        self.hydrogen_scf = bool(hydrogen_scf)
        self.coulomb_scale = float(coulomb_scale)
        self.supported_elements = supported_elements
        self.params = self._load_params(self.data_file)
        self.last_iterations: int | None = None
        self.last_max_delta: float | None = None
        self.last_kkt_residual: float | None = None
        self.last_variational_iterations: int | None = None
        self.last_variational_kkt_residual: float | None = None
        self.last_variational_min_eigenvalue: float | None = None

    @staticmethod
    def _load_params(path: Path) -> dict[str, tuple[float, float, float, int]]:
        params: dict[str, tuple[float, float, float, int]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 5:
                    raise ValueError(
                        "Canonical QEq parameter rows require element, chi, hardness, "
                        "covalent radius, and principal quantum number."
                    )
                symbol, chi, hardness, radius, principal_n = fields[:5]
                params[symbol] = (
                    float(chi),
                    float(hardness),
                    float(radius),
                    int(principal_n),
                )
        if not params:
            raise ValueError(f"QEq parameter table is empty: {path}")
        return params

    @staticmethod
    def _fit_coefficient(principal_n: int) -> float:
        if principal_n < 1:
            raise ValueError("QEq principal quantum number must be positive.")
        return _GTO_FIT_COEFFICIENT.get(
            principal_n, 0.27 * float(principal_n) ** -1.35
        )

    @classmethod
    def gaussian_exponent(cls, slater_exponent: float, principal_n: int) -> float:
        """Return the fitted Gaussian exponent in bohr^-2."""
        if slater_exponent <= 0:
            raise ValueError("QEq Slater exponent must be positive.")
        return (
            cls._fit_coefficient(principal_n)
            * slater_exponent
            * slater_exponent
            / float(principal_n)
        )

    @classmethod
    def coulomb_kernel(
        cls,
        distance: float,
        radius_i: float,
        principal_n_i: int,
        radius_j: float,
        principal_n_j: int,
        *,
        lambda_scale: float = 0.5,
        slater_exponent_i: float | None = None,
        slater_exponent_j: float | None = None,
    ) -> float:
        """Evaluate the fitted GTO Coulomb integral in eV/e^2."""
        if distance <= 0:
            raise ValueError("QEq-GTO pair distance must be positive.")
        if radius_i <= 0 or radius_j <= 0:
            raise ValueError("QEq covalent radii must be positive.")

        radius_i_bohr = radius_i / BOHR_TO_ANGSTROM
        radius_j_bohr = radius_j / BOHR_TO_ANGSTROM
        zeta_i = slater_exponent_i
        if zeta_i is None:
            zeta_i = lambda_scale * (2.0 * principal_n_i + 1.0) / (2.0 * radius_i_bohr)
        zeta_j = slater_exponent_j
        if zeta_j is None:
            zeta_j = lambda_scale * (2.0 * principal_n_j + 1.0) / (2.0 * radius_j_bohr)
        kernel, _, _ = cls._pair_kernel_and_zeta_derivatives(
            distance,
            float(zeta_i),
            principal_n_i,
            float(zeta_j),
            principal_n_j,
        )
        return kernel

    @classmethod
    def _pair_kernel_and_zeta_derivatives(
        cls,
        distance: float,
        zeta_i: float,
        principal_n_i: int,
        zeta_j: float,
        principal_n_j: int,
    ) -> tuple[float, float, float]:
        """Return J_ij, dJ_ij/dzeta_i, and dJ_ij/dzeta_j in eV."""
        alpha_i = cls.gaussian_exponent(zeta_i, principal_n_i)
        alpha_j = cls.gaussian_exponent(zeta_j, principal_n_j)
        beta = math.sqrt(2.0 * alpha_i * alpha_j / (alpha_i + alpha_j))
        distance_bohr = distance / BOHR_TO_ANGSTROM
        kernel = HARTREE_TO_EV * math.erf(beta * distance_bohr) / distance_bohr
        common = (
            HARTREE_TO_EV
            * 2.0
            / math.sqrt(math.pi)
            * math.exp(-(beta * distance_bohr) ** 2)
        )
        d_beta_d_zeta_i = beta * alpha_j / (zeta_i * (alpha_i + alpha_j))
        d_beta_d_zeta_j = beta * alpha_i / (zeta_j * (alpha_i + alpha_j))
        return (
            kernel,
            common * d_beta_d_zeta_i,
            common * d_beta_d_zeta_j,
        )

    def _arrays(
        self, symbols: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        unsupported = sorted(set(symbols).difference(self.params))
        if self.supported_elements is not None:
            unsupported = sorted(
                set(unsupported).union(set(symbols).difference(self.supported_elements))
            )
        if unsupported:
            supported = (
                ", ".join(sorted(self.supported_elements))
                if self.supported_elements is not None
                else "the parameter table"
            )
            raise ValueError(
                f"QEq-GTO supports {supported}; unsupported elements: "
                f"{', '.join(unsupported)}."
            )
        values = [self.params[symbol] for symbol in symbols]
        chi = np.asarray([value[0] for value in values], dtype=np.float64)
        hardness = np.asarray([value[1] for value in values], dtype=np.float64)
        radii = np.asarray([value[2] for value in values], dtype=np.float64)
        principal_n = np.asarray([value[3] for value in values], dtype=np.int64)
        return chi, hardness, radii, principal_n

    def _slater_exponents(
        self,
        symbols: list[str],
        radii: np.ndarray,
        principal_n: np.ndarray,
        charges: np.ndarray,
    ) -> np.ndarray:
        radii_bohr = radii / BOHR_TO_ANGSTROM
        zeta = self.lambda_scale * (2.0 * principal_n + 1.0) / (2.0 * radii_bohr)
        if self.hydrogen_scf:
            for index, symbol in enumerate(symbols):
                if symbol == "H":
                    zeta[index] += charges[index]
                    if zeta[index] <= 0:
                        raise ValueError(
                            "QEq-GTO hydrogen screening exponent became non-positive; "
                            "the SCF left the published hydrogen charge domain."
                        )
        return zeta

    def interaction_matrix(
        self, atoms, charges: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        symbols = atoms.get_chemical_symbols()
        chi, hardness, radii, principal_n = self._arrays(symbols)
        if charges is None:
            q = np.zeros(len(atoms), dtype=np.float64)
        else:
            q = self._validate_charges(atoms, charges)
        zeta = self._slater_exponents(symbols, radii, principal_n, q)

        matrix = np.diag(hardness.copy())
        if self.hydrogen_scf:
            for index, symbol in enumerate(symbols):
                if symbol == "H":
                    zeta_zero = zeta[index] - q[index]
                    matrix[index, index] = hardness[index] * (
                        1.0 + q[index] / zeta_zero
                    )
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                distance = float(np.linalg.norm(positions[i] - positions[j]))
                if distance < 1.0e-10:
                    raise ValueError("QEq-GTO cannot evaluate coincident atoms.")
                value = self.coulomb_scale * self.coulomb_kernel(
                    distance,
                    radii[i],
                    int(principal_n[i]),
                    radii[j],
                    int(principal_n[j]),
                    lambda_scale=self.lambda_scale,
                    slater_exponent_i=float(zeta[i]),
                    slater_exponent_j=float(zeta[j]),
                )
                matrix[i, j] = matrix[j, i] = value
        return chi, matrix

    @staticmethod
    def _validate_charges(atoms, charges: np.ndarray) -> np.ndarray:
        q = np.asarray(charges, dtype=np.float64)
        if q.shape != (len(atoms),):
            raise ValueError(
                f"QEq-GTO charges must have shape ({len(atoms)},), received {q.shape}."
            )
        if not np.isfinite(q).all():
            raise ValueError("QEq-GTO charges must be finite.")
        return q

    @staticmethod
    def _solve_constrained(
        chi: np.ndarray, matrix: np.ndarray, total_charge: float
    ) -> np.ndarray:
        n_atoms = len(chi)
        augmented = np.zeros((n_atoms + 1, n_atoms + 1), dtype=np.float64)
        rhs = np.zeros(n_atoms + 1, dtype=np.float64)
        augmented[:n_atoms, :n_atoms] = matrix
        augmented[n_atoms, :n_atoms] = 1.0
        augmented[:n_atoms, n_atoms] = 1.0
        rhs[:n_atoms] = -chi
        rhs[n_atoms] = total_charge
        try:
            solution = np.linalg.solve(augmented, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "QEq-GTO solve failed; the constrained idempotential matrix is singular."
            ) from exc
        charges = solution[:n_atoms]
        if not np.isfinite(charges).all():
            raise ValueError("QEq-GTO produced non-finite charges.")
        return charges

    @staticmethod
    def _validate_hydrogen_domain(atoms, charges: np.ndarray) -> None:
        for symbol, charge in zip(atoms.get_chemical_symbols(), charges):
            if symbol == "H" and not HYDROGEN_CHARGE_MIN < charge < HYDROGEN_CHARGE_MAX:
                raise ValueError(
                    "QEq-GTO left the published hydrogen domain -1 < q_H < 1; "
                    f"received q_H={charge:.8f}."
                )

    def _validate_solution(self, atoms, charges: np.ndarray, total_charge: float) -> np.ndarray:
        if abs(float(charges.sum()) - total_charge) > QEQ_CHARGE_TOL:
            raise ValueError("QEq-GTO failed its total-charge constraint.")
        if self.hydrogen_scf:
            self._validate_hydrogen_domain(atoms, charges)
        chi, matrix = self.interaction_matrix(atoms, charges)
        chemical_potential = matrix @ charges + chi
        self.last_kkt_residual = float(
            np.max(np.abs(chemical_potential - chemical_potential.mean()))
        )
        residual_limit = max(1.0e-8, 100.0 * self.tolerance)
        if self.last_kkt_residual > residual_limit:
            raise ValueError(
                "QEq-GTO failed its self-consistent KKT residual: "
                f"{self.last_kkt_residual:.3e} eV > {residual_limit:.3e} eV."
            )
        return charges

    def solve(
        self,
        atoms,
        total_charge: float = 0.0,
        extra_hessian: np.ndarray | None = None,
    ) -> np.ndarray:
        if len(atoms) == 0:
            raise ValueError("QEq-GTO requires at least one atom.")
        if extra_hessian is not None:
            return self.solve_variational(
                atoms,
                total_charge=total_charge,
                extra_hessian=extra_hessian,
            )

        target = float(total_charge)
        symbols = atoms.get_chemical_symbols()
        charges = np.full(len(atoms), target / len(atoms), dtype=np.float64)
        has_hydrogen_scf = self.hydrogen_scf and "H" in symbols
        if not has_hydrogen_scf:
            chi, matrix = self.interaction_matrix(atoms, charges)
            result = self._solve_constrained(chi, matrix, target)
            self.last_iterations = 1
            self.last_max_delta = 0.0
            return self._validate_solution(atoms, result, target)

        last_delta = math.inf
        for iteration in range(1, self.max_iterations + 1):
            chi, matrix = self.interaction_matrix(atoms, charges)
            new_charges = self._solve_constrained(chi, matrix, target)
            self._validate_hydrogen_domain(atoms, new_charges)
            last_delta = float(np.max(np.abs(new_charges - charges)))
            charges = (1.0 - self.damping) * charges + self.damping * new_charges
            self._validate_hydrogen_domain(atoms, charges)
            if last_delta < self.tolerance:
                self.last_iterations = iteration
                self.last_max_delta = last_delta
                return self._validate_solution(atoms, charges, target)

        self.last_iterations = self.max_iterations
        self.last_max_delta = last_delta
        raise ValueError(
            "QEq-GTO hydrogen SCF did not converge after "
            f"{self.max_iterations} iterations (max delta={last_delta:.3e} e)."
        )

    def energy_ev(self, atoms, charges: np.ndarray) -> float:
        """Evaluate the published QEq energy at fixed charges.

        The original hydrogen SCF neglects derivatives of charge-dependent
        exponents while updating its linear equations.  Consequently this
        energy is useful for diagnostics, but the SCF charges must not be
        advertised as a strict variational minimum of this function.
        """
        q = self._validate_charges(atoms, charges)
        symbols = atoms.get_chemical_symbols()
        chi, hardness, radii, principal_n = self._arrays(symbols)
        zeta = self._slater_exponents(symbols, radii, principal_n, q)
        radii_bohr = radii / BOHR_TO_ANGSTROM
        zeta_zero = self.lambda_scale * (2.0 * principal_n + 1.0) / (2.0 * radii_bohr)
        energy = float(chi @ q + 0.5 * hardness @ (q * q))
        if self.hydrogen_scf:
            for index, symbol in enumerate(symbols):
                if symbol == "H":
                    energy += 0.5 * hardness[index] * q[index] ** 3 / zeta_zero[index]
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                distance = float(np.linalg.norm(positions[i] - positions[j]))
                energy += q[i] * q[j] * self.coulomb_scale * self.coulomb_kernel(
                    distance,
                    radii[i],
                    int(principal_n[i]),
                    radii[j],
                    int(principal_n[j]),
                    lambda_scale=self.lambda_scale,
                    slater_exponent_i=float(zeta[i]),
                    slater_exponent_j=float(zeta[j]),
                )
        return float(energy)

    def charge_gradient_ev(
        self,
        atoms,
        charges: np.ndarray,
        extra_hessian: np.ndarray | None = None,
    ) -> np.ndarray:
        """Differentiate the published QEq-GTO energy with respect to charges.

        This is the consistent-QEq derivative: it includes the charge
        derivative of every hydrogen-dependent pair integral rather than using
        the cheaper original QEq fixed-point approximation.
        """
        q = self._validate_charges(atoms, charges)
        symbols = atoms.get_chemical_symbols()
        chi, idempotential, radii, principal_n = self._arrays(symbols)
        zeta = self._slater_exponents(symbols, radii, principal_n, q)
        radii_bohr = radii / BOHR_TO_ANGSTROM
        zeta_zero = self.lambda_scale * (2.0 * principal_n + 1.0) / (
            2.0 * radii_bohr
        )
        gradient = chi + idempotential * q
        if self.hydrogen_scf:
            for index, symbol in enumerate(symbols):
                if symbol == "H":
                    gradient[index] += (
                        1.5 * idempotential[index] * q[index] ** 2 / zeta_zero[index]
                    )

        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                distance = float(np.linalg.norm(positions[i] - positions[j]))
                if distance < 1.0e-10:
                    raise ValueError("QEq-GTO cannot evaluate coincident atoms.")
                kernel, d_kernel_i, d_kernel_j = self._pair_kernel_and_zeta_derivatives(
                    distance,
                    float(zeta[i]),
                    int(principal_n[i]),
                    float(zeta[j]),
                    int(principal_n[j]),
                )
                kernel *= self.coulomb_scale
                d_kernel_i *= self.coulomb_scale
                d_kernel_j *= self.coulomb_scale
                gradient[i] += q[j] * kernel
                gradient[j] += q[i] * kernel
                if self.hydrogen_scf and symbols[i] == "H":
                    gradient[i] += q[i] * q[j] * d_kernel_i
                if self.hydrogen_scf and symbols[j] == "H":
                    gradient[j] += q[i] * q[j] * d_kernel_j

        if extra_hessian is not None:
            extra = np.asarray(extra_hessian, dtype=np.float64)
            if extra.shape != (len(atoms), len(atoms)) or not np.isfinite(extra).all():
                raise ValueError("QEq extra Hessian has the wrong shape or contains non-finite values.")
            if not np.allclose(extra, extra.T, atol=1.0e-10):
                raise ValueError("QEq extra Hessian must be symmetric.")
            gradient = gradient + extra @ q
        return np.asarray(gradient, dtype=np.float64)

    def solve_variational(
        self,
        atoms,
        total_charge: float = 0.0,
        extra_hessian: np.ndarray | None = None,
        initial_charges: np.ndarray | None = None,
    ) -> np.ndarray:
        """Minimize the consistent QEq energy under the charge constraint.

        This nonlinear path is used for polarizable continuum coupling.  It is
        distinct from the original QEq fixed-point SCF used to generate fixed
        charges, because it retains every charge derivative required by the
        published CQEq energy.
        """
        from scipy.linalg import null_space
        from scipy.optimize import minimize

        if len(atoms) == 0:
            raise ValueError("QEq-GTO requires at least one atom.")
        target = float(total_charge)
        extra = None
        if extra_hessian is not None:
            extra = np.asarray(extra_hessian, dtype=np.float64)
            if extra.shape != (len(atoms), len(atoms)) or not np.isfinite(extra).all():
                raise ValueError("QEq extra Hessian has the wrong shape or contains non-finite values.")
            if not np.allclose(extra, extra.T, atol=1.0e-10):
                raise ValueError("QEq extra Hessian must be symmetric.")
        if initial_charges is None:
            initial = self.solve(atoms, total_charge=target)
        else:
            initial = self._validate_charges(atoms, initial_charges).copy()
            if abs(float(initial.sum()) - target) > QEQ_CHARGE_TOL:
                raise ValueError("Initial QEq charges do not satisfy the total-charge constraint.")

        def objective(q: np.ndarray) -> float:
            value = self.energy_ev(atoms, q)
            if extra is not None:
                value += 0.5 * float(q @ extra @ q)
            return value

        def jacobian(q: np.ndarray) -> np.ndarray:
            return self.charge_gradient_ev(atoms, q, extra)

        bounds = [
            (HYDROGEN_CHARGE_MIN + 1.0e-8, HYDROGEN_CHARGE_MAX - 1.0e-8)
            if symbol == "H"
            else (None, None)
            for symbol in atoms.get_chemical_symbols()
        ]
        result = minimize(
            objective,
            initial,
            jac=jacobian,
            method="SLSQP",
            bounds=bounds,
            constraints={
                "type": "eq",
                "fun": lambda q: float(np.sum(q) - target),
                "jac": lambda q: np.ones_like(q),
            },
            options={"ftol": 1.0e-12, "maxiter": self.max_iterations},
        )
        if not result.success:
            raise ValueError(f"Variational QEq-GTO solve failed: {result.message}")
        charges = self._validate_solution_domain_only(atoms, result.x, target)
        gradient = jacobian(charges)
        self.last_variational_iterations = int(result.nit)
        self.last_variational_kkt_residual = float(
            np.max(np.abs(gradient - gradient.mean()))
        )
        if self.last_variational_kkt_residual > 2.0e-6:
            raise ValueError(
                "Variational QEq-GTO failed its KKT residual: "
                f"{self.last_variational_kkt_residual:.3e} eV."
            )

        tangent = null_space(np.ones((1, len(atoms)), dtype=np.float64))
        if tangent.size:
            step = 1.0e-5
            hessian = np.column_stack(
                [
                    (
                        jacobian(charges + step * tangent[:, column])
                        - jacobian(charges - step * tangent[:, column])
                    )
                    / (2.0 * step)
                    for column in range(tangent.shape[1])
                ]
            )
            projected = tangent.T @ hessian
            projected = 0.5 * (projected + projected.T)
            self.last_variational_min_eigenvalue = float(
                np.min(np.linalg.eigvalsh(projected))
            )
            if self.last_variational_min_eigenvalue < -1.0e-5:
                raise ValueError(
                    "Variational QEq-GTO stationary point is not a local minimum; "
                    f"projected Hessian eigenvalue={self.last_variational_min_eigenvalue:.3e} eV."
                )
        else:
            self.last_variational_min_eigenvalue = math.inf
        return charges

    def _validate_solution_domain_only(
        self, atoms, charges: np.ndarray, total_charge: float
    ) -> np.ndarray:
        q = self._validate_charges(atoms, charges)
        if abs(float(q.sum()) - total_charge) > QEQ_CHARGE_TOL:
            raise ValueError("QEq-GTO failed its total-charge constraint.")
        if self.hydrogen_scf:
            self._validate_hydrogen_domain(atoms, q)
        return q

    def energy_and_forces_ev_angstrom(
        self, atoms, charges: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Evaluate the QEq energy and coordinate force at fixed charges."""
        symbols = atoms.get_chemical_symbols()
        q_np = self._validate_charges(atoms, charges)
        chi_np, hardness_np, radii_np, principal_n_np = self._arrays(symbols)
        zeta_np = self._slater_exponents(symbols, radii_np, principal_n_np, q_np)
        radii_bohr_np = radii_np / BOHR_TO_ANGSTROM
        zeta_zero_np = self.lambda_scale * (2.0 * principal_n_np + 1.0) / (
            2.0 * radii_bohr_np
        )
        alpha_np = np.asarray(
            [
                self.gaussian_exponent(float(zeta), int(principal_n))
                for zeta, principal_n in zip(zeta_np, principal_n_np)
            ],
            dtype=np.float64,
        )

        coords = torch.tensor(
            np.asarray(atoms.get_positions()), dtype=torch.float64, requires_grad=True
        )
        q = torch.tensor(q_np, dtype=torch.float64)
        chi = torch.tensor(chi_np, dtype=torch.float64)
        hardness = torch.tensor(hardness_np, dtype=torch.float64)
        alpha = torch.tensor(alpha_np, dtype=torch.float64)
        energy = torch.dot(chi, q) + 0.5 * torch.dot(hardness, q * q)
        if self.hydrogen_scf:
            for index, symbol in enumerate(symbols):
                if symbol == "H":
                    energy = energy + (
                        0.5
                        * hardness[index]
                        * q[index] ** 3
                        / float(zeta_zero_np[index])
                    )
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                distance_bohr = torch.linalg.vector_norm(coords[i] - coords[j]) / BOHR_TO_ANGSTROM
                beta = torch.sqrt(2.0 * alpha[i] * alpha[j] / (alpha[i] + alpha[j]))
                kernel = HARTREE_TO_EV * torch.erf(beta * distance_bohr) / distance_bohr
                energy = energy + self.coulomb_scale * q[i] * q[j] * kernel
        forces = -torch.autograd.grad(energy, coords)[0]
        return float(energy.detach().numpy()), forces.detach().numpy()


class QEqTorch:
    """Compatibility wrapper for MAPLE's legacy experimental GBSA path."""

    def __init__(self, data_file="./data/qeq.dat", device="cpu", eps0=1.0):
        data_path = Path(__file__).parent / data_file
        self.device = torch.device(device)
        self.solver = QEqGTO(
            data_file=data_path,
            coulomb_scale=1.0 / float(eps0),
            supported_elements=None,
        )

    @staticmethod
    def _infer_total_charge(atoms) -> float:
        charge = getattr(atoms, "info", {}).get("charge")
        if charge is not None:
            return float(charge)
        if hasattr(atoms, "get_initial_charges"):
            initial = np.asarray(atoms.get_initial_charges(), dtype=float)
            if initial.size and np.all(np.isfinite(initial)):
                return float(initial.sum())
        if hasattr(atoms, "get_initial_charge"):
            try:
                return float(atoms.get_initial_charge())
            except Exception:
                pass
        return 0.0

    def forward(self, atoms, total_charge=None):
        target = self._infer_total_charge(atoms) if total_charge is None else float(total_charge)
        charges = self.solver.solve(atoms, total_charge=target)
        return torch.as_tensor(charges, dtype=torch.float64).cpu()

    __call__ = forward
