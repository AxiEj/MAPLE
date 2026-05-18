# -*- coding: utf-8 -*-
"""Shared utilities for OPT algorithms."""
import os
from typing import List, Optional

import numpy as np
from ase import Atoms

from maple.function.utility.numeric import to_numpy_f64, vec1d
from maple.function.utility.xyz_io import write_xyz as _write_xyz


def write_xyz(filename: str, atoms_list: List[Atoms],
              energies: Optional[List[float]] = None,
              mode: str = "w",
              start_index: int = 0) -> None:
    """Write one or more structures through the shared extxyz writer."""
    _write_xyz(
        filename,
        atoms_list,
        energies=energies,
        mode=mode,
        start_index=start_index,
    )


def finalize_optimization_output(
    job,
    *,
    energy: float,
    summary: str,
    opt_traj_file: str,
    final_path_label: Optional[str] = None,
) -> None:
    """Write the final optimizer XYZ frame and closing summary."""
    base, _ = os.path.splitext(job.output)
    opt_file = base + "_opt.xyz"
    write_xyz(opt_file, [job.atoms], energies=[energy])
    if job.params.verbose != 1 and job._last_iter_info is not None:
        job.log_info(job._last_iter_info)

    info = [f"\n{summary}\n"]
    if job.params.log_final_paths:
        if final_path_label is None:
            info.extend([
                f"Final frame written to {opt_file}\n",
                f"Optimization trajectory written to {opt_traj_file}\n",
            ])
        else:
            info.extend([
                f"\n{final_path_label} {opt_file}\n",
                f"Optimization trajectory written to: {opt_traj_file}\n",
            ])
    job.log_info(info)


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
