"""Shared limited-memory BFGS driver for reaction-path optimizers."""

from __future__ import annotations

import numpy as np


class LBFGSDriver:
    """L-BFGS optimizer for projected gradients without a line search."""

    def __init__(
        self,
        m=5,
        curvature=70.0,
        maxstep=0.2,
        *,
        step_limit_in_place=True,
    ):
        self.m = m
        self.H0 = 1.0 / curvature
        self.maxstep = maxstep
        self.step_limit_in_place = step_limit_in_place
        self.S, self.Y, self.rhos = [], [], []

    def two_loop(self, grad: np.ndarray) -> np.ndarray:
        q = grad.copy()
        alpha = []
        for s, y, rho in reversed(list(zip(self.S, self.Y, self.rhos))):
            a = rho * np.dot(s, q)
            alpha.append(a)
            q -= a * y

        gamma = self.H0 if not self.Y else np.dot(
            self.Y[-1], self.S[-1]
        ) / (np.dot(self.Y[-1], self.Y[-1]) + 1e-20)
        z = gamma * q

        for (s, y, rho), a in zip(zip(self.S, self.Y, self.rhos), reversed(alpha)):
            b = rho * np.dot(y, z)
            z += s * (a - b)

        return -z

    def update(self, s: np.ndarray, y: np.ndarray):
        rho = 1.0 / (np.dot(y, s) + 1e-20)
        self.S.append(s.copy())
        self.Y.append(y.copy())
        self.rhos.append(rho)
        if len(self.S) > self.m:
            self.S.pop(0)
            self.Y.pop(0)
            self.rhos.pop(0)

    def step_limit(self, step: np.ndarray) -> np.ndarray:
        if not self.step_limit_in_place:
            step = step.copy()
        max_disp = np.max(np.abs(step))
        if max_disp > self.maxstep:
            step *= self.maxstep / max_disp
        return step

    def should_stop(self, grad, fmax_th=1e-3, frms_th=5e-4) -> bool:
        max_f = np.max(np.abs(grad))
        rms_f = np.sqrt(np.mean(grad**2))
        return (max_f < fmax_th) and (rms_f < frms_th)

    def ci_should_stop(self, regular_forces, climbing_force):
        if regular_forces.size == 0:
            max_regular = 0.0
            rms_regular = 0.0
        else:
            max_regular = float(np.max(np.abs(regular_forces)))
            rms_regular = float(np.sqrt(np.mean(regular_forces**2)))

        if climbing_force.size == 0:
            max_climbing = 0.0
            rms_climbing = 0.0
        else:
            max_climbing = float(np.max(np.abs(climbing_force)))
            rms_climbing = float(np.sqrt(np.mean(climbing_force**2)))

        return (
            max_regular < self.fmax_reg
            and rms_regular < self.frms_reg
            and max_climbing < self.fmax_ci
            and rms_climbing < self.frms_ci
        )
