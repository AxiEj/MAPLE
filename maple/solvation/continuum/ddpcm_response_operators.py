"""Exact ddPCM values and tiled analytic operator contractions.

The discretization is the frozen ddX 0.8 equation used by ``TorchDDPCM``.
Only its execution changes: numerical values and local solid-harmonic jets
replace the retained all-pair autograd tape. These private workspaces belong
to one evaluation, never to a long-lived PES cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
from ase.units import Bohr

from maple.solvation.surfaces.lebedev import ordered_lebedev_grid

from .harmonic_torch_primitives import _torch_real_harmonic_design
from .solid_harmonic_response import (
    solid_harmonic_contracted_hessian,
    solid_harmonic_jets,
)
from .torch_ddpcm import _immutable_float64
from .response_topology import RESPONSE_TOPOLOGY_POLICY, certify_response_topology
from .response_tensor_binding import TensorStateBinding

_CONSTANT_TENSORS = (
    "_directions",
    "_weights",
    "_radii",
    "_ell",
    "_single_scale",
    "_double_scale",
    "_design",
    "_identity3",
)


@dataclass(frozen=True, slots=True)
class _Geometry:
    positions: torch.Tensor
    nodes: torch.Tensor
    delta: torch.Tensor
    distance: torch.Tensor
    chi: torch.Tensor
    fi: torch.Tensor
    ui: torch.Tensor
    gi: torch.Tensor
    radial_first: torch.Tensor
    radial_second: torch.Tensor
    chi_gradient: torch.Tensor
    f_jacobian: torch.Tensor
    u_jacobian: torch.Tensor
    g_jacobian: torch.Tensor
    topology: Any


@dataclass(frozen=True, slots=True)
class _PrimalData:
    L: torch.Tensor
    D: torch.Tensor
    rhs: torch.Tensor
    psi: torch.Tensor
    geometry: _Geometry
    source_active: torch.Tensor
    phi: torch.Tensor


@dataclass(frozen=True, slots=True)
class _DerivativeData:
    rhs_jacobian: torch.Tensor
    psi_jacobian: torch.Tensor
    l_action_jacobian: torch.Tensor
    d_action_jacobian: torch.Tensor
    lt_action_jacobian: torch.Tensor
    dt_action_jacobian: torch.Tensor
    rhs_contraction_hessian: torch.Tensor | None
    l_contraction_hessian: torch.Tensor | None
    d_contraction_hessian: torch.Tensor | None


def _relative_jacobian(gradient, target: int, source: int, dimension: int):
    result = gradient.new_zeros((len(gradient), dimension))
    if target != source:
        result[:, 3 * target : 3 * target + 3] = gradient
        result[:, 3 * source : 3 * source + 3] = -gradient
    return result


def _add_relative_hessian(result, block, target: int, source: int):
    if target == source:
        return
    i = slice(3 * target, 3 * target + 3)
    j = slice(3 * source, 3 * source + 3)
    result[i, i] += block
    result[j, j] += block
    result[i, j] -= block
    result[j, i] -= block


def _add_local_action_jacobian(result, block, target: int, source: int):
    if target != source:
        result[:, 3 * target : 3 * target + 3] += block
        result[:, 3 * source : 3 * source + 3] -= block


class DDPCMResponseOperators:
    """Private immutable parameters and constant quadrature for exact kernels.

    Mutable numerical tensors are private implementation buffers. Public
    radius data are bytes-backed, read-only, and detached from input arrays.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        radii_angstrom,
        *,
        dielectric: float,
        lmax: int = 15,
        n_lebedev: int = 1202,
        eta: float = 0.1,
        device="cpu",
        topology_margin: float = 1e-8,
    ):
        object.__setattr__(self, "_sealed", False)
        self._symbols = tuple(symbols)
        if not self._symbols:
            raise ValueError("ddPCM requires a nonempty ordered atom list.")
        self._radii_angstrom = _immutable_float64(radii_angstrom, (len(self._symbols),))
        if np.any(self._radii_angstrom <= 0):
            raise ValueError("ddPCM radii must be positive.")
        if isinstance(lmax, bool) or not isinstance(lmax, int) or lmax < 1:
            raise ValueError("lmax must be a positive integer.")
        self.lmax = lmax
        self.n_lebedev = n_lebedev
        self.dielectric = float(dielectric)
        self._eta = float(eta)
        self._topology_margin = float(topology_margin)
        if not math.isfinite(self.dielectric) or self.dielectric <= 1:
            raise ValueError("dielectric must be finite and greater than one.")
        if not math.isfinite(self._eta) or not 0 < self._eta <= 1:
            raise ValueError("eta must lie in (0, 1].")
        if not math.isfinite(self._topology_margin) or self._topology_margin <= 0:
            raise ValueError("topology margin must be positive and finite.")
        self.device = torch.device(device)
        self.dtype = torch.float64
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("ddPCM response supports CPU or CUDA.")
        grid = ordered_lebedev_grid(n_lebedev)
        self._grid_sha256 = grid.sha256
        self._directions = torch.tensor(
            grid.directions.copy(), dtype=self.dtype, device=self.device
        )
        self._weights = torch.tensor(
            grid.weights.copy(), dtype=self.dtype, device=self.device
        )
        self._radii = torch.tensor(
            self._radii_angstrom.copy() / Bohr,
            dtype=self.dtype,
            device=self.device,
        )
        self._ell = torch.tensor(
            np.concatenate([np.full(2 * l + 1, l) for l in range(lmax + 1)]),
            dtype=self.dtype,
            device=self.device,
        )
        self._single_scale = 4 * math.pi / (2 * self._ell + 1)
        self._double_scale = self._ell * self._single_scale
        self._design = _torch_real_harmonic_design(self._directions, lmax=lmax)
        self._identity3 = torch.eye(3, dtype=self.dtype, device=self.device)
        self._constant_binding = TensorStateBinding.capture(self._constant_tensors())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("ddPCM response operator parameters are immutable.")
        object.__setattr__(self, name, value)

    @property
    def symbols(self):
        return self._symbols

    @property
    def radii_angstrom(self):
        return self._radii_angstrom

    @property
    def eta(self):
        return self._eta

    @property
    def n_atoms(self):
        return len(self._symbols)

    @property
    def n_basis(self):
        return (self.lmax + 1) ** 2

    @property
    def basis_dimension(self):
        return self.n_atoms * self.n_basis

    def configuration_payload(self):
        self._constant_binding.validate(self._constant_tensors())
        return {
            "topology_policy": RESPONSE_TOPOLOGY_POLICY,
            "symbols": self.symbols,
            "radii_angstrom": self.radii_angstrom.tolist(),
            "dielectric": self.dielectric,
            "lmax": self.lmax,
            "n_lebedev": self.n_lebedev,
            "eta": self.eta,
            "topology_margin": self._topology_margin,
            "grid_sha256": self._grid_sha256,
            "device": str(self.device),
            "dtype": "torch.float64",
            "equations": "ddx-0.8.0-exact-point-l1",
        }

    def _constant_tensors(self):
        return tuple((name, getattr(self, name)) for name in _CONSTANT_TENSORS)

    def _validate(self, positions, source):
        for name, value, shape in (
            ("positions", positions, (self.n_atoms, 3)),
            ("active source", source, (self.n_atoms, 4)),
        ):
            if not torch.is_tensor(value) or value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")
            if value.dtype != self.dtype or value.device != self._directions.device:
                raise ValueError(f"{name} must use configured FP64/device.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite.")

    @torch.no_grad()
    def _geometry(self, positions):
        self._constant_binding.validate(self._constant_tensors())
        positions = positions.detach().clone()
        centres = positions / Bohr
        nodes = centres[:, None, :] + self._radii[:, None, None] * self._directions
        delta = nodes[:, :, None, :] - centres[None, None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        owner = torch.eye(self.n_atoms, dtype=torch.bool, device=positions.device)[
            :, None, :
        ]
        nonself = ~owner.expand_as(distance)
        safe = torch.where(owner, torch.ones_like(distance), distance)
        if bool((safe[nonself] <= 1e-13).any()):
            raise RuntimeError(
                "a ddPCM grid node coincides with a distinct sphere centre."
            )
        ratio = safe / self._radii[None, None, :]
        # Identical operation order to the reviewed switch value path.
        shifted = ratio - 0.5 * self.eta
        a = 1.0 - shifted
        z = a / self.eta
        polynomial = z**3 * (z * (6.0 * z - 15.0) + 10.0)
        chi = torch.where(
            a <= 0,
            torch.zeros_like(a),
            torch.where(a >= self.eta, torch.ones_like(a), polynomial),
        )
        chi = torch.where(owner, torch.zeros_like(chi), chi)
        fi = chi.sum(dim=-1)
        ui, gi = torch.clamp(1 - fi, min=0), torch.clamp(fi, min=1)
        cert_ratio = torch.where(
            owner, torch.ones_like(distance), distance / self._radii[None, None, :]
        )
        topology = certify_response_topology(
            positions,
            cert_ratio,
            chi,
            fi,
            ui,
            eta=self.eta,
            margin=self._topology_margin,
        )
        if bool((distance <= 1e-13).any()):
            raise RuntimeError("a point source coincides with a ddPCM surface node.")
        interior = (a > 0) & (a < self.eta) & nonself
        first = torch.where(
            interior,
            -30 * z**2 * (z - 1) ** 2 / (self.eta * self._radii[None, None, :]),
            0.0,
        )
        second = torch.where(
            interior,
            60
            * z
            * (2 * z**2 - 3 * z + 1)
            / (self.eta * self._radii[None, None, :]) ** 2,
            0.0,
        )
        grad = first[..., None] * delta / distance[..., None] / Bohr
        jf = -grad.reshape(self.n_atoms, self.n_lebedev, 3 * self.n_atoms).clone()
        for i in range(self.n_atoms):
            jf[i, :, 3 * i : 3 * i + 3] += grad[i].sum(dim=1)
        ju = -jf * (fi <= 1)[..., None]
        jg = (fi >= 1)[..., None] * jf
        return _Geometry(
            positions,
            nodes,
            delta,
            distance,
            chi,
            fi,
            ui,
            gi,
            first,
            second,
            grad,
            jf,
            ju,
            jg,
            topology,
        )

    def _chi_hessian(self, geometry, i, j):
        radius = geometry.distance[i, :, j]
        unit = geometry.delta[i, :, j] / radius[:, None]
        outer = unit[:, :, None] * unit[:, None, :]
        return (
            geometry.radial_second[i, :, j, None, None] * outer
            + (geometry.radial_first[i, :, j] / radius)[:, None, None]
            * (self._identity3 - outer)
        ) / Bohr**2

    def _potential(self, geometry, source):
        delta, distance = geometry.delta, geometry.distance
        dipoles = source[:, (3, 1, 2)] / Bohr
        phi = (
            source[None, None, :, 0] / distance
            + (delta * dipoles[None, None, :, :]).sum(dim=-1) / distance**3
        ).sum(dim=-1)
        return phi

    @torch.no_grad()
    def build(self, positions, source_active):
        self._validate(positions, source_active)
        source = source_active.detach().clone()
        geometry = self._geometry(positions)
        q, b = self.n_basis, self.basis_dimension
        L = positions.new_zeros((b, b))
        D = positions.new_zeros((b, b))
        Y, w, scale = self._design, self._weights, self._single_scale
        for i in range(self.n_atoms):
            rows = slice(i * q, (i + 1) * q)
            target_weight = w * geometry.ui[i]
            L[rows, rows] = torch.diag(scale)
            D[rows, rows] = Y.T @ (target_weight[:, None] * (-0.5 * Y * scale[None, :]))
            for j in range(self.n_atoms):
                if i == j:
                    continue
                columns = slice(j * q, (j + 1) * q)
                distance = geometry.distance[i, :, j]
                unit = geometry.delta[i, :, j] / distance[:, None]
                design = _torch_real_harmonic_design(unit, lmax=self.lmax)
                ratio = distance / self._radii[j]
                overlap = geometry.chi[i, :, j] / geometry.gi[i]
                lp = -design * (
                    overlap[:, None]
                    * ratio[:, None] ** self._ell[None, :]
                    * scale[None, :]
                )
                L[rows, columns] = Y.T @ (w[:, None] * lp)
                ds = self._double_scale * ratio[:, None] ** (-(self._ell[None, :] + 1))
                D[rows, columns] = Y.T @ (target_weight[:, None] * design * ds)
        phi = self._potential(geometry, source)
        rhs = torch.cat(
            [-(Y.T @ (w * geometry.ui[i] * phi[i])) for i in range(self.n_atoms)]
        )
        psi = positions.new_zeros((self.n_atoms, q))
        psi[:, 0] = math.sqrt(4 * math.pi) * source[:, 0]
        psi[:, 1:4] = (
            (4 * math.pi / 3)
            * (source[:, 1:4] / (Bohr * math.sqrt(4 * math.pi / 3)))
            / self._radii[:, None]
        )
        return _PrimalData(L, D, rhs, psi.flatten(), geometry, source, phi)

    def _rhs_derivatives(self, primal, alpha, order):
        geometry, source, phi = primal.geometry, primal.source_active, primal.phi
        n, q = self.n_atoms, self.n_basis
        m, d = 3 * n, 7 * n
        delta, distance = geometry.delta, geometry.distance
        p = source[:, (3, 1, 2)] / Bohr
        dot = (delta * p[None, None, :, :]).sum(dim=-1)
        grad = (
            -source[None, None, :, 0, None] * delta / distance[..., None] ** 3
            + p[None, None, :, :] / distance[..., None] ** 3
            - 3 * delta * dot[..., None] / distance[..., None] ** 5
        ) / Bohr
        for i in range(n):
            grad[i, :, i, :] = 0
        jr_phi = -grad.reshape(n, self.n_lebedev, m).clone()
        for i in range(n):
            jr_phi[i, :, 3 * i : 3 * i + 3] += grad[i].sum(dim=1)
        js_phi = torch.cat(
            (
                1 / distance[..., None],
                delta[..., (1, 2, 0)] / (Bohr * distance[..., None] ** 3),
            ),
            dim=-1,
        ).reshape(n, self.n_lebedev, 4 * n)
        jphi = torch.cat((jr_phi, js_phi), dim=-1)
        jr = source.new_zeros((n * q, d))
        jp = torch.zeros_like(jr)
        h = source.new_zeros((d, d)) if order == 2 else None
        for i in range(n):
            rows = slice(i * q, (i + 1) * q)
            ju = source.new_zeros((self.n_lebedev, d))
            ju[:, :m] = geometry.u_jacobian[i]
            jr[rows] = -(
                self._design.T
                @ (
                    self._weights[:, None]
                    * (geometry.ui[i, :, None] * jphi[i] + phi[i, :, None] * ju)
                )
            )
            jp[i * q, m + 4 * i] = math.sqrt(4 * math.pi)
            for k in range(3):
                jp[i * q + 1 + k, m + 4 * i + 1 + k] = (
                    (4 * math.pi / 3)
                    / (Bohr * math.sqrt(4 * math.pi / 3))
                    / self._radii[i]
                )
            if order != 2:
                continue
            w = -self._weights * (self._design @ alpha[rows])
            cross = ju.T @ (w[:, None] * jphi[i])
            h += cross + cross.T
            w_phi = w * geometry.ui[i]
            w_u = -w * phi[i] * (geometry.fi[i] <= 1)
            for j in range(n):
                if i == j:
                    continue
                x, radius = delta[i, :, j], distance[i, :, j]
                outer = x[:, :, None] * x[:, None, :]
                pdot = dot[i, :, j]
                h_phi = source[j, 0] * (
                    3 * outer / radius[:, None, None] ** 5
                    - self._identity3 / radius[:, None, None] ** 3
                )
                h_phi += (
                    -3
                    * (
                        p[j, :, None] * x[:, None, :]
                        + x[:, :, None] * p[j, None, :]
                        + pdot[:, None, None] * self._identity3
                    )
                    / radius[:, None, None] ** 5
                )
                h_phi += 15 * pdot[:, None, None] * outer / radius[:, None, None] ** 7
                block = (w_phi[:, None, None] * h_phi).sum(dim=0) / Bohr**2
                block += (w_u[:, None, None] * self._chi_hessian(geometry, i, j)).sum(
                    dim=0
                )
                _add_relative_hessian(h, block, i, j)
                mixed_q = -x / (Bohr * radius[:, None] ** 3)
                mixed_p = (
                    self._identity3 / radius[:, None, None] ** 3
                    - 3 * outer / radius[:, None, None] ** 5
                ) / Bohr**2
                mixed = torch.cat((mixed_q[..., None], mixed_p[..., (1, 2, 0)]), dim=-1)
                mixed = (w_phi[:, None, None] * mixed).sum(dim=0)
                ss = slice(m + 4 * j, m + 4 * j + 4)
                ii, jj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
                h[ii, ss] += mixed
                h[ss, ii] += mixed.T
                h[jj, ss] -= mixed
                h[ss, jj] -= mixed.T
        return jr, jp, h

    @torch.no_grad()
    def derivatives(self, primal, *, mu, z, lam, v0, alpha, order=1):
        """Return contractions at fixed central state and adjoint vectors.

        In particular ``v0=y0-r(theta0)`` is constant here. Its RHS response
        is supplied separately by Jr and the explicit mixed off-shell terms.
        """
        self._constant_binding.validate(self._constant_tensors())
        if order not in (1, 2):
            raise ValueError("operator derivative order must be one or two.")
        n, q, b = self.n_atoms, self.n_basis, self.basis_dimension
        m = 3 * n
        for value in (mu, z, lam, v0, alpha):
            if value.shape != (b,) or not bool(torch.isfinite(value).all()):
                raise ValueError(
                    "Contraction vectors must be finite central B-vectors."
                )
        geometry = primal.geometry
        jr, jp, hr = self._rhs_derivatives(primal, alpha, order)
        j_l = z.new_zeros((b, m))
        j_d = torch.zeros_like(j_l)
        j_lt = torch.zeros_like(j_l)
        j_dt = torch.zeros_like(j_l)
        h_l = z.new_zeros((m, m)) if order == 2 else None
        h_d = z.new_zeros((m, m)) if order == 2 else None
        Y, w = self._design, self._weights
        for i in range(n):
            rows = slice(i * q, (i + 1) * q)
            g, u = geometry.gi[i], geometry.ui[i]
            jg, ju = geometry.g_jacobian[i], geometry.u_jacobian[i]
            ym, yl = Y @ mu[rows], Y @ lam[rows]
            numerator = z.new_zeros(self.n_lebedev)
            jnumerator = z.new_zeros((self.n_lebedev, m))
            field = -0.5 * (Y @ (self._single_scale * v0[rows]))
            jfield = torch.zeros_like(jnumerator)
            # Only projected local 3x3 Hessians survive a tile, not basis tapes.
            local_second = []
            j_dt[rows] += (
                -0.5 * self._single_scale[:, None] * (Y.T @ ((w * yl)[:, None] * ju))
            )
            for j in range(n):
                if i == j:
                    continue
                columns = slice(j * q, (j + 1) * q)
                vectors = geometry.delta[i, :, j] / self._radii[j]
                radius_A = float(self._radii_angstrom[j])
                rc = self._single_scale * z[columns]
                dc = self._double_scale * v0[columns]
                if order == 2:
                    regular = solid_harmonic_contracted_hessian(
                        vectors, self.lmax, "regular", rc
                    )
                    irregular = solid_harmonic_contracted_hessian(
                        vectors, self.lmax, "irregular", dc
                    )
                    regular_values = regular.basis_values
                    irregular_values = irregular.basis_values
                    regular_gradient = regular.basis_gradient
                    irregular_gradient = irregular.basis_gradient
                else:
                    regular = solid_harmonic_jets(
                        vectors, self.lmax, "regular", derivative_order=order
                    )
                    irregular = solid_harmonic_jets(
                        vectors, self.lmax, "irregular", derivative_order=order
                    )
                    regular_values = regular.values
                    irregular_values = irregular.values
                    regular_gradient = regular.gradient
                    irregular_gradient = irregular.gradient
                f_l, f_d = regular_values @ rc, irregular_values @ dc
                gl = torch.einsum("gpa,p->ga", regular_gradient, rc) / radius_A
                gd = torch.einsum("gpa,p->ga", irregular_gradient, dc) / radius_A
                chi = geometry.chi[i, :, j]
                gc = geometry.chi_gradient[i, :, j]
                numerator += chi * f_l
                jnumerator += _relative_jacobian(
                    chi[:, None] * gl + f_l[:, None] * gc, i, j, m
                )
                field += f_d
                jfield += _relative_jacobian(gd, i, j, m)
                jc = _relative_jacobian(gc, i, j, m)
                overlap = chi / g
                jo = jc / g[:, None] - (chi / g**2)[:, None] * jg
                j_lt[columns] -= self._single_scale[:, None] * (
                    regular_values.T @ ((w * ym)[:, None] * jo)
                )
                local_l = (
                    -self._single_scale[:, None]
                    * torch.einsum("gpa,g->pa", regular_gradient, w * ym * overlap)
                    / radius_A
                )
                _add_local_action_jacobian(j_lt[columns], local_l, i, j)
                j_dt[columns] += self._double_scale[:, None] * (
                    irregular_values.T @ ((w * yl)[:, None] * ju)
                )
                local_d = (
                    self._double_scale[:, None]
                    * torch.einsum("gpa,g->pa", irregular_gradient, w * yl * u)
                    / radius_A
                )
                _add_local_action_jacobian(j_dt[columns], local_d, i, j)
                if order == 2:
                    hc = self._chi_hessian(geometry, i, j)
                    hl = regular.contracted_hessian / radius_A**2
                    hd = irregular.contracted_hessian / radius_A**2
                    hn = (
                        chi[:, None, None] * hl
                        + f_l[:, None, None] * hc
                        + gc[:, :, None] * gl[:, None, :]
                        + gl[:, :, None] * gc[:, None, :]
                    )
                    local_second.append((j, hc, hn, hd))
                del regular, irregular
            j_l[rows] = -(
                Y.T
                @ (
                    w[:, None]
                    * (jnumerator / g[:, None] - (numerator / g**2)[:, None] * jg)
                )
            )
            j_d[rows] = Y.T @ (w[:, None] * (u[:, None] * jfield + field[:, None] * ju))
            if order == 2:
                wl, wd = -w * ym, w * yl
                cross_l = -jnumerator.T @ ((wl / g**2)[:, None] * jg)
                h_l += (
                    cross_l
                    + cross_l.T
                    + 2 * jg.T @ ((wl * numerator / g**3)[:, None] * jg)
                )
                cross_d = ju.T @ (wd[:, None] * jfield)
                h_d += cross_d + cross_d.T
                for j, hc, hn, hd in local_second:
                    hlocal_l = ((wl / g)[:, None, None] * hn).sum(dim=0)
                    hlocal_l -= (
                        (wl * numerator / g**2 * (geometry.fi[i] >= 1))[:, None, None]
                        * hc
                    ).sum(dim=0)
                    _add_relative_hessian(h_l, hlocal_l, i, j)
                    hlocal_d = ((wd * u)[:, None, None] * hd).sum(dim=0)
                    hlocal_d -= (
                        (wd * field * (geometry.fi[i] <= 1))[:, None, None] * hc
                    ).sum(dim=0)
                    _add_relative_hessian(h_d, hlocal_d, i, j)
        return _DerivativeData(jr, jp, j_l, j_d, j_lt, j_dt, hr, h_l, h_d)


__all__ = ["DDPCMResponseOperators"]
