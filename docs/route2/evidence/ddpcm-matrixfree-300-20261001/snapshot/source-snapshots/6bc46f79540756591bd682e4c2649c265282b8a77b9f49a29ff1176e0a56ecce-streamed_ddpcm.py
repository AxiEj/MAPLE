"""Unadmitted, value-only exact matrix-free ddPCM research prototype.

The unchanged DDPCMResponseOperators supplies constants and conventions only.
No dense build, derivative assembler, public calculator or domain gate is used.
Iterative solves below are explicitly diagnostics WITHOUT the dense SVD stability
certificate. They must not be used to claim admitted energies or derivatives.
"""

from dataclasses import asdict
import hashlib
import math
import time

import numpy as np
import torch
from ase.units import Bohr

from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.continuum.ddpcm_response_operators import DDPCMResponseOperators
from maple.solvation.continuum.harmonic_torch_primitives import _torch_real_harmonic_design
from maple.solvation.continuum.response_topology import (
    OneSidedNodeProof, RESPONSE_TOPOLOGY_POLICY, ResponseTopologyCertificate,
    _hash_arrays,
)


class StreamedDDPCM:
    """CPU FP64 operator actions with one target and bounded source tiles."""

    def __init__(self, symbols, radii_angstrom, positions, *, dielectric,
                 lmax=15, n_lebedev=1202, eta=0.1, source_tile=8,
                 topology_margin=1e-8):
        if isinstance(source_tile, bool) or not isinstance(source_tile, int) or source_tile < 1:
            raise ValueError("source_tile must be a positive integer.")
        self.parameters = DDPCMResponseOperators(
            symbols, radii_angstrom, dielectric=dielectric, lmax=lmax,
            n_lebedev=n_lebedev, eta=eta, topology_margin=topology_margin,
        )
        self.n = len(symbols)
        self.q = (lmax + 1) ** 2
        self.g = n_lebedev
        self.dimension = self.n * self.q
        self.tile = source_tile
        self.lmax = lmax
        self.epsilon = float(dielectric)
        self.eta = float(eta)
        self.margin = topology_margin
        self._validate(positions, (self.n, 3), "positions")
        self.positions = positions.detach().clone()
        self.centres = self.positions / Bohr
        p = self.parameters
        self.nodes = self.centres[:, None, :] + p._radii[:, None, None] * p._directions
        self.fi, self.ui, self.gi, self.topology = self._geometry()
        self.action_calls = 0
        self.action_seconds = 0.0

    @staticmethod
    def _validate(value, shape, name):
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} requires shape {shape}.")
        if value.dtype != torch.float64 or value.device.type != "cpu":
            raise ValueError(f"{name} requires CPU FP64.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite.")

    def _row_geometry(self, i):
        delta = self.nodes[i, :, None, :] - self.centres[None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        if bool((distance <= 1e-13).any()):
            raise RuntimeError("a point source coincides with a ddPCM surface node.")
        ratio = distance / self.parameters._radii[None, :]
        # Preserve the old scalar's operation order, including owner exclusion.
        a = 1.0 - (ratio - 0.5 * self.eta)
        z = a / self.eta
        polynomial = z**3 * (z * (6.0 * z - 15.0) + 10.0)
        chi = torch.where(a <= 0, 0.0, torch.where(a >= self.eta, 1.0, polynomial))
        chi[:, i] = 0.0
        return delta, distance, ratio, chi

    @torch.no_grad()
    def _geometry(self):
        """Stream the existing topology decisions and byte-identical hash order."""
        fi = torch.empty((self.n, self.g), dtype=torch.float64)
        digest = hashlib.sha256(RESPONSE_TOPOLOGY_POLICY.encode())
        digest.update(str((self.n, self.g, self.n)).encode())
        digest.update(b"int8")
        minimum_switch = math.inf
        minimum_active = math.inf
        proofs = []
        active_count = buried_count = transition_count = 0
        lower, upper = 1 - self.eta / 2, 1 + self.eta / 2
        for i in range(self.n):
            _, _, ratio, chi = self._row_geometry(i)
            nonself = torch.ones(self.n, dtype=torch.bool)
            nonself[i] = False
            f = chi.sum(dim=-1)
            ui = torch.clamp(1 - f, min=0)
            if (not bool(torch.isfinite(ratio[:, nonself]).all())
                    or bool((ratio[:, nonself] <= 0).any())
                    or not bool(torch.isfinite(chi).all())
                    or bool(((chi < 0) | (chi > 1)).any())
                    or not bool(torch.isfinite(f).all())):
                raise RuntimeError("Invalid radial values for ddPCM response topology proof.")
            endpoint = torch.minimum((ratio - lower).abs(), (ratio - upper).abs())
            if self.n > 1:
                local_min = float(endpoint[:, nonself].min())
                minimum_switch = min(minimum_switch, local_min)
                if local_min <= self.margin:
                    raise RuntimeError("ddPCM response switch endpoint is within the certified margin.")
            one = (ratio < lower) & nonself
            interior = (ratio > lower) & (ratio < upper) & nonself
            zero = (ratio > upper) & nonself
            if bool(((one & (chi != 1)) | (zero & (chi != 0))).any()):
                raise RuntimeError("ddPCM radial support and plateau values disagree.")
            n_one, n_interior = one.sum(-1), interior.sum(-1)
            transition = n_interior > 0
            near = (f - 1).abs() <= self.margin
            if bool((near & transition & (f == 1)).any()):
                raise RuntimeError("ddPCM response computed equality has an active interior switch.")
            upper_proof = (n_one == 1) & (f > 1)
            lower_proof = (n_one == 0) & (n_interior == 1) & (f < 1)
            if bool((near & transition & ~(upper_proof | lower_proof)).any()):
                raise RuntimeError("ddPCM response has an unproved near-one branch crossing.")
            support = torch.where(one, 2, torch.where(interior, 1, 0)).to(torch.int8)
            digest.update(support.numpy().tobytes())
            for node in (near & transition).nonzero().flatten().tolist():
                is_upper = bool(upper_proof[node])
                proofs.append(OneSidedNodeProof(
                    owner=i, lebedev_index=node, f=float(f[node]),
                    side="upper" if is_upper else "lower",
                    proof="stable-plateau-one" if is_upper else "single-interior",
                    plateau_one_count=int(n_one[node]), interior_count=int(n_interior[node]),
                    plateau_zero_count=int(zero[node].sum()),
                    nonowner_radial_branches=tuple(int(v) for v in support[node, nonself]),
                    minimum_switch_margin=float(endpoint[node, nonself].min()),
                ))
            if bool(transition.any()):
                minimum_active = min(minimum_active, float((f[transition] - 1).abs().min()))
            active_count += int((ui > 0).sum())
            buried_count += int(((f == 1) & ~transition).sum())
            transition_count += int(transition.sum())
            fi[i] = f
        ui, gi = torch.clamp(1 - fi, min=0), torch.clamp(fi, min=1)
        for mask in (ui > 0, fi > 1):
            array = mask.numpy()
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        certificate = ResponseTopologyCertificate(
            contract=RESPONSE_TOPOLOGY_POLICY, topology_sha256=digest.hexdigest(),
            positions_sha256=_hash_arrays("positions-angstrom-v1", (self.positions,)),
            active_node_count=active_count, buried_plateau_count=buried_count,
            transition_node_count=transition_count,
            minimum_switch_margin=minimum_switch if math.isfinite(minimum_switch) else None,
            minimum_active_f_margin=minimum_active if math.isfinite(minimum_active) else None,
            one_sided_nodes=tuple(proofs),
        )
        return fi, ui, gi, certificate

    def _tiles(self, i, kind, delta, distance, ratio, chi):
        p = self.parameters
        indices = [j for j in range(self.n) if j != i and
                   (kind != "L" or bool((chi[:, j] != 0).any()))]
        for start in range(0, len(indices), self.tile):
            js = indices[start:start + self.tile]
            unit = delta[:, js, :].permute(1, 0, 2) / distance[:, js].T[:, :, None]
            design = _torch_real_harmonic_design(unit.reshape(-1, 3), lmax=self.lmax)
            design = design.reshape(len(js), self.g, self.q)
            r = ratio[:, js].T
            if kind == "L":
                kernel = -design * (
                    (chi[:, js].T / self.gi[i])[:, :, None]
                    * r[:, :, None] ** p._ell[None, None, :]
                    * p._single_scale[None, None, :]
                )
            else:
                kernel = design * (p._double_scale[None, None, :]
                                   * r[:, :, None] ** (-(p._ell[None, None, :] + 1)))
            yield js, kernel

    @torch.no_grad()
    def apply(self, kind, vector, *, transpose=False):
        if kind not in {"L", "D", "A", "C"}:
            raise ValueError("Unknown ddPCM operator.")
        if not torch.is_tensor(vector) or vector.ndim not in (1, 2):
            raise ValueError("operator input must be a vector or RHS block.")
        if vector.shape[0] != self.dimension:
            raise ValueError("operator dimension mismatch.")
        self._validate(vector, vector.shape, "operator vector")
        if kind in {"A", "C"}:
            scale = (2 * math.pi * (self.epsilon + 1) / (self.epsilon - 1)
                     if kind == "A" else 2 * math.pi)
            return scale * vector - self.apply("D", vector, transpose=transpose)
        begun = time.perf_counter()
        was_vector = vector.ndim == 1
        x = vector.reshape(self.n, self.q, -1)
        result = torch.zeros_like(x)
        p = self.parameters
        y, w = p._design, p._weights
        for i in range(self.n):
            delta, distance, ratio, chi = self._row_geometry(i)
            if kind == "L":
                result[i] += p._single_scale[:, None] * x[i]
                target_weight = w
            else:
                target_weight = w * self.ui[i]
                if transpose:
                    result[i] += (-0.5 * p._single_scale[:, None]
                                 * (y.T @ (target_weight[:, None] * (y @ x[i]))))
                else:
                    result[i] += y.T @ (target_weight[:, None] *
                                       (y @ (-0.5 * p._single_scale[:, None] * x[i])))
            if transpose:
                left = target_weight[:, None] * (y @ x[i])
                for js, kernel in self._tiles(i, kind, delta, distance, ratio, chi):
                    result[js] += torch.einsum("sgq,gk->sqk", kernel, left)
            else:
                field = torch.zeros((self.g, x.shape[-1]), dtype=torch.float64)
                for js, kernel in self._tiles(i, kind, delta, distance, ratio, chi):
                    field += torch.einsum("sgq,sqk->gk", kernel, x[js])
                result[i] += y.T @ (target_weight[:, None] * field)
        self.action_calls += 1
        self.action_seconds += time.perf_counter() - begun
        output = result.reshape(self.dimension, -1)
        return output[:, 0] if was_vector else output

    @torch.no_grad()
    def rhs_psi(self, source):
        self._validate(source, (self.n, 4), "source")
        p = self.parameters
        rhs = torch.empty((self.n, self.q), dtype=torch.float64)
        dipoles = source[:, (3, 1, 2)] / Bohr
        for i in range(self.n):
            delta, distance, _, _ = self._row_geometry(i)
            phi = (source[None, :, 0] / distance
                   + (delta * dipoles[None, :, :]).sum(-1) / distance**3).sum(-1)
            rhs[i] = -(p._design.T @ (p._weights * self.ui[i] * phi))
        psi = torch.zeros_like(rhs)
        psi[:, 0] = math.sqrt(4 * math.pi) * source[:, 0]
        psi[:, 1:4] = ((4 * math.pi / 3)
                      * source[:, 1:4] / (Bohr * math.sqrt(4 * math.pi / 3))
                      / p._radii[:, None])
        return rhs.flatten(), psi.flatten()

    @torch.no_grad()
    def diagonal(self, kind):
        p = self.parameters
        if kind == "L":
            return p._single_scale.repeat(self.n)
        if kind == "A":
            diag_d = -0.5 * p._single_scale[None, :] * (
                (self.ui * p._weights[None, :]) @ p._design.square())
            return (2 * math.pi * (self.epsilon + 1) / (self.epsilon - 1) - diag_d).flatten()
        raise ValueError("Jacobi diagonal is implemented for L and A only.")

    def retained_tensors(self):
        geometry = [(name, getattr(self, name)) for name in
                    ("positions", "centres", "nodes", "fi", "ui", "gi")]
        return geometry + list(self.parameters._constant_tensors())

    def storage_report(self):
        tensors = self.retained_tensors()
        return {
            "atom_count": self.n, "basis_dimension": self.dimension,
            "retained_tensor_bytes": sum(t.numel() * t.element_size() for _, t in tensors),
            "dense_operator_bytes": 0, "retained_pair_grid_bytes": 0,
            "source_tile": self.tile,
            "kernel_tile_bytes": min(self.tile, max(0, self.n - 1)) * self.g * self.q * 8,
            "one_target_pair_scalar_bytes": self.g * self.n * 8,
            "counts_are_not_measured_process_peak": True,
        }


def solve_diagnostic(operator, source, *, tolerance=1e-12, restart=40,
                     maxiter=30, include_state=False):
    """Uncertified continuum-only E/adjoint probe; never a public PES API."""
    from scipy.sparse.linalg import LinearOperator, gmres

    if not 0 < tolerance <= 1e-12:
        raise ValueError("Diagnostic solve tolerance may not exceed 1e-12.")
    records = []

    def solve(kind, rhs, transpose=False):
        def action(x):
            return operator.apply(kind, torch.as_tensor(x, dtype=torch.float64),
                                  transpose=transpose).numpy()

        matrix = LinearOperator((operator.dimension, operator.dimension), matvec=action,
                                dtype=np.float64)
        diagonal = operator.diagonal(kind).numpy()
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal == 0):
            raise RuntimeError("Invalid diagnostic Jacobi preconditioner.")
        preconditioner = LinearOperator(matrix.shape, matvec=lambda x: x / diagonal,
                                        dtype=np.float64)
        trace = []
        vector, info = gmres(matrix, np.asarray(rhs), M=preconditioner,
                             rtol=tolerance, atol=tolerance, restart=restart,
                             maxiter=maxiter, callback=trace.append,
                             callback_type="pr_norm")
        residual = float(np.linalg.norm(action(vector) - rhs) / max(1., np.linalg.norm(rhs)))
        if info != 0 or not np.isfinite(residual) or residual > tolerance:
            raise RuntimeError(f"{kind} did not converge: info={info}, true residual={residual:.3e}")
        records.append({"operator": kind, "transpose": transpose, "iterations": len(trace),
                        "true_relative_residual": residual,
                        "preconditioned_residual_trace": [float(v) for v in trace]})
        return torch.from_numpy(vector)

    rhs, psi = operator.rhs_psi(source)
    y = solve("A", operator.apply("C", rhs).numpy())
    z = solve("L", y.numpy())
    mu = solve("L", ((HARTREE_TO_EV / 2) * psi).numpy(), transpose=True)
    lam = solve("A", mu.numpy(), transpose=True)
    result = {"energy_eV": float((HARTREE_TO_EV / 2) * psi.dot(z)),
              "scientific_admitted": False, "stability_certified": False,
              "scope": "continuum-only iterative diagnostic; no MACE/CDS/domain/public admission",
              "solves": records, "storage": operator.storage_report(),
              "topology": asdict(operator.topology),
              "action_calls": operator.action_calls,
              "action_seconds": operator.action_seconds}
    if include_state:
        result["state"] = {"rhs": rhs, "psi": psi, "y": y, "z": z,
                           "mu": mu, "lam": lam}
    return result
