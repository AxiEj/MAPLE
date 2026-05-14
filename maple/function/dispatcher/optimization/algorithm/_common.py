# -*- coding: utf-8 -*-
"""Shared utilities for OPT algorithms."""
import os
from typing import List, Optional

import numpy as np
from ase import Atoms


def write_xyz(filename: str, atoms_list: List[Atoms],
              energies: Optional[List[float]] = None,
              mode: str = "w",
              start_index: int = 0) -> None:
    """Write one or more structures in XYZ format."""
    blocks = []
    for i, at in enumerate(atoms_list):
        pos = at.get_positions()
        symbols = at.get_chemical_symbols()
        image_index = start_index + i

        lines = [f"{len(symbols)}\n"]
        if energies is not None:
            lines.append(f"Image {image_index}  Energy = {energies[i]:.10f}\n")
        else:
            lines.append(f"Image {image_index}\n")
        lines.extend(
            f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n"
            for s, (x, y, z) in zip(symbols, pos)
        )
        blocks.append("".join(lines))

    with open(filename, mode) as f:
        f.writelines(blocks)



def finalize_optimization_output(
    job,
    *,
    energy: float,
    summary: str,
    opt_traj_file: str,
) -> None:
    """Write ``<output_stem>_opt.xyz``, dump last iteration info if silent, and log summary.

    Shared by LBFGS / SDCG / any future OPT algorithm. The ``job`` argument is
    a JobABC subclass exposing ``self.output``, ``self.atoms``, ``self.log_info``,
    ``self.params.verbose``, and ``self._last_iter_info``.
    """
    base, _ = os.path.splitext(job.output)
    opt_file = base + "_opt.xyz"
    write_xyz(opt_file, [job.atoms], energies=[energy])
    if job.params.verbose != 1 and job._last_iter_info is not None:
        job.log_info(job._last_iter_info)
    job.log_info([
        f"\n{summary}\n"
        f"Final frame written to {opt_file}\n"
        f"Optimization trajectory written to {opt_traj_file}\n"
    ])


from maple.function.utility.numeric import to_numpy_f64, vec1d  # noqa: F401  (re-exported)


def compute_metrics(atoms, step_cart, forces) -> None:
    """Set max_dp, rms_dp, max_f, rms_f on atoms from step and force arrays."""
    atoms.max_dp = np.abs(step_cart).max()
    atoms.rms_dp = np.sqrt((step_cart ** 2).sum() / step_cart.size)
    atoms.max_f = np.abs(forces).max()
    atoms.rms_f = np.sqrt((forces ** 2).sum() / forces.size)


def is_converged(atoms) -> bool:
    """Test geometry-optimization convergence against atoms.*_th thresholds."""
    return (
        atoms.max_f <= atoms.f_max_th
        and atoms.rms_f <= atoms.f_rms_th
        and atoms.max_dp <= atoms.dp_max_th
        and atoms.rms_dp <= atoms.dp_rms_th
    )
