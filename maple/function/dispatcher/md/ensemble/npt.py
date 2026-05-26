"""
NPT (isothermal-isobaric) ensemble implementation.

Supports two combinations:
    thermostat: 'langevin' | 'v-rescale'  (default: v-rescale)
    barostat:   'berendsen' | 'c-rescale' (default: c-rescale)

Recommended combination for production MLP runs:
    thermostat=v-rescale + barostat=c-rescale
    → Bernetti-Bussi reversible-Euler stochastic cell rescaling with explicit
      N_P multiple-time-step barostat propagation, split V-rescale thermostat,
      and force refresh after every volume move.

Berendsen variants are suitable for rapid pre-equilibration but suppress
pressure/temperature fluctuations and do not generate correct ensemble averages.

Integration order each step:
    - Langevin: LFMiddle carried-velocity sequence, barostat pressure from
      synchronized standard velocity, then barostat scaling of carried state.
    - V-rescale + c-rescale: on steps divisible by N_P, advance √V over
      N_P·dt and scale positions/momenta; then refresh forces, apply a
      V-rescale half step, run full Velocity Verlet, and apply the second
      V-rescale half step.
    - V-rescale + Berendsen: full Velocity Verlet step, thermostat, then weak
      pressure scaling (equilibration-only).

Requirements:
    - Atoms object must have a full three-dimensional periodic cell
      (atoms.pbc must be [True, True, True])
    - Calculator should support stress tensor evaluation for accurate pressure

References:
    Berendsen et al., J. Chem. Phys. 81, 3684 (1984).
    Bussi, Donadio & Parrinello, J. Chem. Phys. 126, 014101 (2007).
    Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from ase import Atoms

from ...jobABC import JobABC
from maple.function.timer import timer

from ..integrator.velocity_verlet import VelocityVerlet
from ..evaluator import evaluate_md_properties
from ..thermostat.langevin import LangevinThermostat
from ..thermostat.vrescale import VRescaleThermostat, ZERO_KE_THRESHOLD_HA
from ..barostat.berendsen import BerendsenBarostat
from ..barostat.crescale import CRescaleBarostat, MDBarostatClampError
from ..utils import (
    EV_PER_ANG3_TO_BAR,
    HARTREE_TO_EV,
    VELOCITY_REPR_LFMIDDLE_CARRIED,
    VELOCITY_REPR_STANDARD,
    apply_runtime_motion_projection,
    calculate_temperature,
    calculate_kinetic_energy,
    condition_input_velocities,
    get_atoms_velocity_representation,
    initialize_velocities,
    forces_au,
    lfmiddle_carried_to_standard,
    normalize_velocities_to_standard,
    pbc_com_default_note,
    set_atoms_velocity_representation,
    standard_to_lfmiddle_carried,
    FS_TO_AU,
    validate_md_parameter_ranges,
)
from ..semantics import resolve_md_dof_policy, validate_md_admission_state
from ..provenance import build_run_context
from ..rst_io import get_rng_state_hex, restore_rng_from_hex
from ..logger import MDLogger


@dataclass
class NPTParams:
    """
    Parameters for NPT (isothermal-isobaric) ensemble simulation.

    NPT is used for density equilibration and computing thermodynamic
    properties at constant pressure (e.g., liquid density, compressibility).
    Defaults follow published standards for ML potentials.

    Thermostat default: v-rescale (Bussi et al. 2007 JCP 126, 014101)
      — correct canonical ensemble; less perturbative than Langevin.
    Barostat default: c-rescale (Bernetti & Bussi 2020 JCP 153, 114107)
      — stochastic isotropic log-volume cell rescaling; pressure analogue of
        v-rescale and preferred over Berendsen for volume fluctuations.

    Recommended production combination: thermostat=v-rescale + barostat=c-rescale.
    Langevin remains available as an explicitly requested damped-equilibration
    path with synchronized pressure velocities, but MAPLE does not implement
    the Monte Carlo/Langevin-piston/MTTK barostats used by older MD engines for
    formally stronger Langevin-NPT treatments.
    Berendsen variants are suitable for rapid pre-equilibration only.
    """
    # ------------------------------------------------------------------
    # Timestep: 0.1 fs
    # Smaller timestep for ML potentials improves energy conservation.
    # Refs: Zhang et al. (2018) Phys. Rev. Lett. 120, 143001 (DeePMD);
    #       Batatia et al. (2022) NeurIPS 35, 11423 (MACE).
    # ------------------------------------------------------------------
    timestep:        float = 0.1          # fs

    # ------------------------------------------------------------------
    # Total steps: 100000 × 0.1 fs = 10 ps
    # Standard default simulation length for ML-MD runs.
    # Refs: GROMACS Lemkul tutorial; AMBER Tutorial 1.
    # ------------------------------------------------------------------
    steps:           int   = 100000       # steps (= 10 ps at 0.1 fs/step)

    temperature:     float = 300.0        # K
    pressure:        float = 1.0          # bar

    # ------------------------------------------------------------------
    # Thermostat: v-rescale (default for NPT)
    # V-rescale produces correct canonical KE distribution while perturbing
    # dynamics less than Langevin, making it better suited for NPT where
    # both thermostat and barostat act each step.
    # Ref: GROMACS default since v4.5 (Bussi et al. 2007).
    # ------------------------------------------------------------------
    thermostat:      str   = 'v-rescale'  # [GROMACS default; Bussi 2007]

    # ------------------------------------------------------------------
    # Barostat: c-rescale (default for NPT)
    # C-rescale advances the log-volume strain with a stochastic term
    # (Bernetti & Bussi 2020). Unlike Berendsen, it preserves volume
    # fluctuations and is suitable for production-style NPT in this module.
    # ------------------------------------------------------------------
    barostat:        str   = 'c-rescale'  # [Bernetti & Bussi 2020 JCP 153, 114107]

    # Langevin-specific (only used when thermostat='langevin')
    friction:        float = 0.001        # 1/fs = 1 ps⁻¹

    # ------------------------------------------------------------------
    # V-rescale τ_T: 200 fs
    # Slightly larger than NVT default (100 fs) to avoid over-coupling
    # when both thermostat and barostat act each step.
    # Ref: GROMACS NPT tutorial: tau_t = 0.1 ps = 100 fs;
    #      CHARMM-GUI NPT protocol: tau_t = 1 ps (conservative).
    # ------------------------------------------------------------------
    tau_t:           float = 200.0        # fs  [GROMACS NPT tutorial]

    # ------------------------------------------------------------------
    # Barostat τ_P: 2000 fs = 2 ps
    # Larger than classical MD defaults (0.5–1 ps) to account for ML
    # potential noise in instantaneous pressure.  Noisy pressure → large
    # τ_P needed to avoid volume instability.
    # Ref: GROMACS Lemkul NPT tutorial: tau_p = 2.0 ps;
    #      Bernetti & Bussi 2020 §III: τ_P ≥ 1 ps recommended.
    # ------------------------------------------------------------------
    tau_p:           float = 2000.0       # fs  [GROMACS Lemkul; Bernetti 2020]

    # ------------------------------------------------------------------
    # Barostat multiple-time-step stride N_P (Bernetti & Bussi 2020).
    #
    # When N_P > 1, the reversible-Euler c-rescale operator is applied only on
    # MD steps divisible by N_P, but its λ = √V SDE is advanced over the full
    # N_P·dt interval.  This is not a skipped one-step update; it matches the
    # paper's multiple-time-step barostat cost model (1 + 1/N_P force refreshes
    # on average).  Keep N_P=1 for the tightest coupling/validation default.
    # ------------------------------------------------------------------
    barostat_stride: int = 1

    # ------------------------------------------------------------------
    # Isothermal compressibility: 4.5e-5 1/bar (liquid water, 300 K, 1 bar)
    # Used by both Berendsen and C-rescale barostats as a scaling prefactor.
    # The barostat dynamics are not very sensitive to this value; using water
    # as a default is standard practice for biomolecular systems.
    # Ref: CRC Handbook of Chemistry and Physics; GROMACS mdp default.
    # ------------------------------------------------------------------
    compressibility: float = 4.5e-5       # 1/bar  [CRC Handbook; GROMACS default]

    # ------------------------------------------------------------------
    # Output frequencies
    #
    # ML potentials are ~1000–3000× slower than classical FFs.
    # A typical ML-NPT run is 10–50 ps; dense output is needed to monitor
    # density convergence and detect volume instabilities early.
    #
    # Target: 100–1000 frames per 10 ps.
    #   traj_every = 100 steps × 0.1 fs/step = 10 fs = 0.01 ps/frame
    #   10 ps → 1000 frames  ✓
    #
    # Refs: Stocker et al. (2022) Mach. Learn.: Sci. Technol. 3, 045010 —
    #         GNN-MD benchmarks, typical run 10–100 ps with dense output.
    #       Kovács et al. (2023) J. Chem. Phys. 159, 044118 — MACE evaluation
    #         with per-step monitoring of thermodynamic convergence.
    # ------------------------------------------------------------------
    traj_every:      int   = 100          # steps (= 10 fs = 0.01 ps at 0.1 fs/step)
    log_every:       int   = 100          # steps (= 10 fs)

    # ------------------------------------------------------------------
    # Trajectory format: xyz (text) or dcd (binary)
    # DCD binary format is ~3-4x smaller than XYZ and faster to read/write.
    # Ref: CHARMM documentation; VMD molfile plugin.
    # ------------------------------------------------------------------
    traj_format:     str   = "xyz"        # "xyz" (text, default) or "dcd" (binary)

    verbose:         int   = 1
    debug:           bool  = False
    init_velocities: bool  = True
    restart:          bool  = False
    load_state:       bool  = False
    rst_file:         str   = ""           # Path to RST checkpoint file (explicit source for restart/load_state)
    rst_every:        int   = 1000
    remove_com:       bool  = True   # initialization-only COM removal
    remove_rotation:  bool  = False  # legacy alias path; prefer remove_angular
    remove_angular:   bool  = False  # initialization-only COM + rotation; parallel to remove_com
    remove_com_every: int   = 100    # runtime-only COM removal
    remove_angular_every: int = 0    # runtime-only COM + rotation; parallel to remove_com_every
    allow_partial_pbc: bool = False  # WS0-C: NPT requires full 3-D PBC; field kept for param-key uniformity
    # Berendsen is equilibration-only (suppresses volume fluctuations -> wrong
    # ensemble).  It is hard-rejected for production unless this escape hatch is
    # set, mirroring the allow_partial_pbc per-concern opt-in idiom.
    allow_equilibration_only_barostat: bool = False
    allow_unknown_cutoff: bool = False  # run PBC MD without a declared neighbor cutoff (manifest-recorded)
    allow_barostat_clamp: bool = False  # continue an EXPERIMENTAL run when the c-rescale stability clamp fires (manifest-recorded)
    validation_artifact_id: str = ""  # release-harness acceptance artifact id (manifest traceability)
    random_seed: Optional[int] = None


class NPT(JobABC):
    """
    NPT (isothermal-isobaric) ensemble simulation.

    Integrates with the MAPLE dispatcher via JobABC.
    """

    _THERMOSTAT_CHOICES = {'langevin', 'v-rescale'}
    _BAROSTAT_CHOICES   = {'berendsen', 'c-rescale'}

    def __init__(self, output: str, atoms: Atoms, paras: Optional[dict] = None):
        super().__init__(output)

        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator attached")

        self.atoms = atoms
        self.params = self._init_params(NPTParams, paras, ("md", "MD", "npt", "NPT"))
        self._apply_barostat_stride_aliases(paras or {})
        _com_note = pbc_com_default_note(
            self.atoms, self.params,
            "remove_com_every" in self._lower_keys(
                self._select_subdict(paras or {}, ("md", "MD", "npt", "NPT"))
            ),
        )
        if _com_note:
            self.log_info([f"\nNOTE: {_com_note}\n"])

        if self.params.thermostat not in self._THERMOSTAT_CHOICES:
            raise ValueError(
                f"Unknown thermostat '{self.params.thermostat}'. "
                f"Choose from: {self._THERMOSTAT_CHOICES}"
            )
        if self.params.barostat not in self._BAROSTAT_CHOICES:
            raise ValueError(
                f"Unknown barostat '{self.params.barostat}'. "
                f"Choose from: {self._BAROSTAT_CHOICES}"
            )
        validate_md_parameter_ranges(self.params, "npt")
        self.params.barostat_stride = int(self.params.barostat_stride)
        self._validate_admission_state("startup")

        # Warn if user set Langevin-specific params but chose v-rescale (or vice versa)
        if self.params.thermostat == 'v-rescale' and paras and 'friction' in (paras or {}):
            self.log_info([
                "\n*** WARNING: 'friction' parameter was specified but thermostat is 'v-rescale'.\n"
                "    The friction parameter is only used by the Langevin thermostat.\n"
                "    If you intended Langevin dynamics, add: thermostat=langevin\n\n"
            ])
        if self.params.thermostat == 'langevin' and paras and 'tau_t' in (paras or {}):
            self.log_info([
                "\n*** WARNING: 'tau_t' parameter was specified but thermostat is 'langevin'.\n"
                "    The tau_t parameter is only used by the V-rescale thermostat.\n\n"
            ])
        if self.params.barostat == 'berendsen':
            if not self.params.allow_equilibration_only_barostat:
                raise ValueError(
                    "barostat=berendsen is equilibration-only: it suppresses volume "
                    "fluctuations and does not generate a correct production NPT "
                    "ensemble. Use barostat=c-rescale for production-style isotropic "
                    "NPT, or set allow_equilibration_only_barostat=true to run an "
                    "EXPERIMENTAL (non-production) Berendsen equilibration."
                )
            self.log_info([
                "\n*** WARNING: barostat=berendsen is equilibration-only.\n"
                "    It suppresses volume fluctuations and does not generate a correct production NPT ensemble.\n"
                "    Use barostat=c-rescale for production-style isotropic NPT.\n\n"
            ])
        if self.params.barostat_stride != 1 and not (
            self.params.thermostat == 'v-rescale' and self.params.barostat == 'c-rescale'
        ):
            raise ValueError(
                "barostat_stride (Bernetti-Bussi N_P) is implemented only for the "
                "thermostat=v-rescale + barostat=c-rescale reversible-Euler path. "
                "Use barostat_stride=1 for Langevin or Berendsen paths."
            )

        self._rng = (np.random.default_rng(self.params.random_seed)
                     if self.params.random_seed is not None
                     else np.random.default_rng())

        self._dof_policy = resolve_md_dof_policy(self.atoms, self.params, "npt")
        for warning in self._dof_policy.warnings:
            self.log_info([f"\n*** WARNING: {warning}\n"])
        if self.params.remove_angular:
            self.log_info(["\n*** WARNING: remove_angular is ignored for NPT/PBC systems; only initialization COM removal remains active.\n"])
        self._runtime_n_dof = self._dof_policy.runtime_n_dof
        self._runtime_dof_description = self._dof_policy.runtime_description

        if self.params.thermostat == 'langevin':
            self.thermostat = LangevinThermostat(
                atoms,
                temperature=self.params.temperature,
                friction=self.params.friction,
                timestep=self.params.timestep,
                rng=self._rng,
            )
        else:  # v-rescale
            self.thermostat = VRescaleThermostat(
                atoms,
                temperature=self.params.temperature,
                tau_t=self.params.tau_t,
                timestep=self.params.timestep,
                rng=self._rng,
                n_dof=self._runtime_n_dof,
            )

        if self.params.barostat == 'berendsen':
            self.barostat = BerendsenBarostat(
                atoms,
                pressure=self.params.pressure,
                tau_p=self.params.tau_p,
                timestep=self.params.timestep,
                compressibility=self.params.compressibility,
            )
        else:  # c-rescale
            self.barostat = CRescaleBarostat(
                atoms,
                pressure=self.params.pressure,
                temperature=self.params.temperature,
                tau_p=self.params.tau_p,
                timestep=self.params.timestep,
                compressibility=self.params.compressibility,
                rng=self._rng,
            )

        self.logger = MDLogger(
            output_path=output,
            log_every=self.params.log_every,
            traj_every=self.params.traj_every,
            traj_format=self.params.traj_format,
            verbose=self.params.verbose,
            debug=self.params.debug,
        )

    def _apply_barostat_stride_aliases(self, paras: dict) -> None:
        """Accept explicit N_P spelling without making a broad parser feature.

        ``barostat_stride`` is the canonical MAPLE parameter name; ``barostat_np``
        and ``barostat_n_p`` are narrow aliases for users following the notation
        in Bernetti & Bussi.  Ambiguous conflicting aliases fail at startup.
        """
        sub = self._lower_keys(self._select_subdict(paras, ("md", "MD", "npt", "NPT")))
        aliases = {
            key: sub[key]
            for key in ("barostat_stride", "barostat_np", "barostat_n_p")
            if key in sub
        }
        if not aliases:
            return
        values = {str(value) for value in aliases.values()}
        if len(values) > 1:
            raise ValueError(
                "Conflicting C-rescale N_P aliases supplied: "
                + ", ".join(f"{key}={value!r}" for key, value in aliases.items())
            )
        self.params.barostat_stride = next(iter(aliases.values()))

    def _validate_admission_state(self, context: str) -> None:
        for advisory in validate_md_admission_state(
            self.atoms, "npt", self.params, context=context
        ):
            self.log_info([advisory])
            print(advisory, end="", flush=True)

    def _validate_runtime_cutoff_after_barostat(self, abs_step: int) -> None:
        """Fail before force/stress evaluation if NPT shrinkage breaks MIC cutoff."""
        if not any(self.atoms.pbc):
            return

        from maple.function.calculator.set_calculator import (
            _calculator_neighbor_cutoff_A,
            _minimum_image_radius_A,
            validate_pbc_neighbor_cutoff,
        )

        try:
            validate_pbc_neighbor_cutoff(
                self.atoms,
                self.atoms.calc,
                allow_unknown_cutoff=bool(self.params.allow_unknown_cutoff),
                require_known_cutoff=True,
            )
        except ValueError as exc:
            cutoff = _calculator_neighbor_cutoff_A(self.atoms.calc)
            try:
                radius, shortest = _minimum_image_radius_A(self.atoms)
            except ValueError:
                radius, shortest = float("nan"), float("nan")
            cell = np.asarray(self.atoms.get_cell(), dtype=float)
            volume = float(self.atoms.get_volume()) if all(self.atoms.pbc) else float("nan")
            msg = (
                f"NPT runtime cutoff/MIC guard failed after barostat scaling at step "
                f"{abs_step}: neighbor_cutoff_A={cutoff!r}, "
                f"minimum_image_radius_A={radius:.8g}, "
                f"shortest_lattice_vector_A={shortest:.8g}, volume_A3={volume:.8g}, "
                f"cell_A={cell.tolist()}. The current cell no longer satisfies the "
                "minimum-image convention before the next force/stress evaluation. "
                f"Original error: {exc}"
            )
            self.logger.abort_simulation(reason=msg)
            raise ValueError(msg) from exc

    def run(self):
        """Execute NPT simulation."""
        with timer("MD Simulation (NPT)"):
            self._log_parameters()

            if self.params.load_state:
                if self.params.init_velocities and self.params.debug:
                    self.log_info([
                        "\nload_state=True: ignoring init_velocities and using coordinates/velocities from RST.\n"
                    ])
                result = self.logger.restart_simulation(
                    ensemble='npt',
                    timestep=self.params.timestep,
                    n_steps=self.params.steps,
                    temperature=self.params.temperature,
                    atoms=self.atoms,
                    pressure=self.params.pressure,
                    rst_file=self.params.rst_file if self.params.rst_file else None,
                    load_state=True,
                )
                if result is None:
                    return
                self.atoms, velocities, step_offset = result
                self._validate_admission_state("load_state")
                velocity_representation = self.logger.resumed_velocity_representation
                resumed_timestep_au = (
                    self.logger.resumed_timestep * FS_TO_AU
                    if self.logger.resumed_timestep is not None else None
                )
                if self.logger.resumed_rng_state is not None:
                    restore_rng_from_hex(self._rng, self.logger.resumed_rng_state)
                remaining = self.params.steps
            elif self.params.restart:
                if self.params.init_velocities and self.params.debug:
                    self.log_info([
                        "\nrestart=True: ignoring init_velocities and using coordinates/velocities from RST.\n"
                    ])
                result = self.logger.restart_simulation(
                    ensemble='npt',
                    timestep=self.params.timestep,
                    n_steps=self.params.steps,
                    temperature=self.params.temperature,
                    atoms=self.atoms,
                    pressure=self.params.pressure,
                    rst_file=self.params.rst_file if self.params.rst_file else None,
                    load_state=False,
                )
                if result is None:   # already completed
                    return
                self.atoms, velocities, step_offset = result
                self._validate_admission_state("restart")
                velocity_representation = self.logger.resumed_velocity_representation
                # Resume continues at the same timestep; pass the RST's own recorded
                # timestep (not None) so the shared normalize gate never has to guess.
                resumed_timestep_au = (
                    self.logger.resumed_timestep * FS_TO_AU
                    if self.logger.resumed_timestep is not None else None
                )
                # Restore RNG state for deterministic continuation
                if self.logger.resumed_rng_state is not None:
                    restore_rng_from_hex(self._rng, self.logger.resumed_rng_state)
                remaining = self.params.steps - step_offset
            else:
                if 'velocities' in self.atoms.arrays and self.params.init_velocities:
                    velocity_representation = get_atoms_velocity_representation(self.atoms)
                    velocities, summary = condition_input_velocities(
                        atoms=self.atoms,
                        velocities=self.atoms.arrays['velocities'],
                        temperature=self.params.temperature,
                        remove_com=self.params.remove_com,
                        remove_rotation=self.params.remove_rotation,
                        # NPT requires full PBC; rigid-body rotation is undefined.
                        remove_angular=False,
                        target_n_dof=self._dof_policy.init_n_dof,
                    )
                    self.log_info([
                        "\nVelocities loaded from input file and conditioned as "
                        "the initialization state: "
                        f"T {summary['temperature_before']:.2f} -> "
                        f"{summary['temperature_after']:.2f} K "
                        f"({self._dof_policy.init_description}); "
                        f"projected_com={summary['projected_com']}.\n"
                    ])
                elif self.params.init_velocities:
                    velocities = self._initialize_velocities()
                    velocity_representation = VELOCITY_REPR_STANDARD
                else:
                    if 'velocities' not in self.atoms.arrays:
                        raise ValueError(
                            "init_velocities=False, "
                            "but no velocities found in atoms.arrays"
                        )
                    velocities = np.asarray(self.atoms.arrays['velocities'], dtype=float).copy()
                    velocity_representation = get_atoms_velocity_representation(self.atoms)
                resumed_timestep_au = None
                step_offset = 0
                remaining   = self.params.steps
                source = "input_xyz" if 'velocities' in self.atoms.arrays and not self.params.init_velocities else ("input_xyz" if 'velocities' in self.atoms.arrays and self.params.init_velocities else "init_velocities")
                self.logger.log_debug_initial_state(
                    self.atoms,
                    velocities,
                    mode=source,
                    effective_step=step_offset,
                    velocity_representation=velocity_representation,
                )

            final_velocities, final_representation = self._run_simulation(
                velocities,
                velocity_representation=velocity_representation,
                step_offset=step_offset,
                n_steps=remaining,
                source_timestep_au=resumed_timestep_au,
            )
            self.atoms.arrays['velocities'] = final_velocities
            set_atoms_velocity_representation(self.atoms, final_representation)

    def _log_parameters(self):
        """Log NPT parameters to output."""
        lines = [
            "\n" + "=" * 80 + "\n",
            f"{'NPT MD PARAMETERS':^80}\n",
            "=" * 80 + "\n",
            f"Ensemble:              NPT (isothermal-isobaric)\n",
            f"Thermostat:            {self.params.thermostat}\n",
            f"Barostat:              {self.params.barostat}\n",
            f"Timestep:              {self.params.timestep:.3f} fs\n",
            f"Total steps:           {self.params.steps}\n",
            f"Temperature:           {self.params.temperature:.2f} K\n",
            f"Target pressure:       {self.params.pressure:.2f} bar\n",
        ]
        if self.params.thermostat == 'langevin':
            lines.append(f"Friction (γ):          {self.params.friction:.4f} 1/fs\n")
        else:
            lines.append(f"τ_T:                   {self.params.tau_t:.1f} fs\n")
        lines += [
            f"τ_P:                   {self.params.tau_p:.1f} fs\n",
            f"Barostat stride N_P:   {self.params.barostat_stride} step(s)\n",
            f"Compressibility:       {self.params.compressibility:.2e} 1/bar\n",
            f"\nOutput frequencies:\n",
            f"  Log every:           {self.params.log_every} steps\n",
            f"  Traj every:          {self.params.traj_every} steps\n",
            f"\nVelocity init:         {self.params.init_velocities}\n",
            f"Restart mode:          {self.params.restart}\n",
            f"Load-state mode:       {self.params.load_state}\n",
            f"RST every:             {self.params.rst_every} steps\n",
            f"Remove COM:            {self.params.remove_com} (initialization-only)\n",
            f"Remove angular:        {self.params.remove_angular} (initialization-only; ignored under PBC)\n",
            f"Remove COM every:      {self.params.remove_com_every} (runtime-only)\n",
            f"Remove angular ev.:    {self.params.remove_angular_every} (runtime-only; ignored under PBC)\n",
        ]
        if self.params.random_seed is not None:
            lines.append(f"Random seed:           {self.params.random_seed}\n")
        lines.append("=" * 80 + "\n")
        self.log_info(lines)

    def _initialize_velocities(self) -> np.ndarray:
        """Initialize velocities from Maxwell-Boltzmann distribution."""
        self.log_info([f"\nInitializing velocities at {self.params.temperature:.2f} K...\n"])
        # PBC: rotation is undefined, so initialization projects COM only; the
        # rescale target is the init basis (init_n_dof), not the runtime basis.
        velocities = initialize_velocities(
            atoms=self.atoms,
            temperature=self.params.temperature,
            remove_com=self.params.remove_com,
            remove_rotation=self.params.remove_rotation,
            remove_angular=False,
            target_n_dof=self._dof_policy.init_n_dof,
            rng=self._rng,
        )
        t_init = calculate_temperature(
            self.atoms, velocities, n_dof=self._dof_policy.init_n_dof
        )
        self.log_info([
            f"Initial temperature: {t_init:.2f} K "
            f"({self._dof_policy.init_description})\n"
        ])
        if self._dof_policy.runtime_n_dof != self._dof_policy.init_n_dof:
            t_runtime = calculate_temperature(
                self.atoms, velocities, n_dof=self._dof_policy.runtime_n_dof
            )
            self.log_info([
                f"  Runtime basis: {t_runtime:.2f} K "
                f"({self._dof_policy.runtime_description}); the thermostat "
                f"repopulates the init-projected COM during the run.\n"
            ])
        return velocities

    def _run_simulation(self, velocities: np.ndarray,
                        velocity_representation: str,
                        step_offset: int = 0, n_steps: int = None,
                        source_timestep_au: Optional[float] = None) -> tuple[np.ndarray, str]:
        """
        Run NPT simulation.

        Langevin uses LF-Middle carried velocities internally; the barostat
        pressure decision uses a synchronized standard velocity so the kinetic
        pressure term is not computed from the half-step carried state.
        For the v-rescale + c-rescale production path, the driver follows the
        Bernetti-Bussi reversible-Euler ordering with explicit N_P support:
        on absolute steps divisible by N_P, propagate √V over the full N_P·dt
        interval, scale positions/momenta, refresh forces at the scaled
        geometry, then run a split V-rescale half step / full Velocity Verlet /
        V-rescale half step.  Non-barostat steps still use the same split
        thermostat + Hamiltonian ordering; they do not perform a hidden one-step
        pressure update.  This path reports H̃ = K + U + P_0·V − Σ ΔW_ext
        (thermostat + barostat + projection work) as an effective-energy
        integration diagnostic.
        """
        if n_steps is None:
            n_steps = self.params.steps

        is_langevin = self.params.thermostat == 'langevin'
        # Two-step restart-velocity contract (shared across ensembles): (1) normalize
        # any checkpoint representation to standard with the source-geometry forces and
        # the RST's own timestep — an unknown label or a carried checkpoint with no
        # source timestep is rejected, not guessed; (2) re-derive the LF-Middle carried
        # velocity the Langevin integrator carries internally, at the *current* timestep.
        force_for_conversion = (
            forces_au(self.atoms)
            if velocity_representation != VELOCITY_REPR_STANDARD or is_langevin
            else None
        )
        velocities, velocity_representation = normalize_velocities_to_standard(
            self.atoms, velocities, velocity_representation,
            force_for_conversion, source_timestep_au,
        )
        if not is_langevin:
            active_ke = calculate_kinetic_energy(self.atoms, velocities)
            if active_ke <= ZERO_KE_THRESHOLD_HA:
                raise ValueError(
                    "NPT with thermostat=v-rescale requires non-zero active kinetic "
                    f"energy after restart/load_state velocity normalization; got "
                    f"{active_ke:.3e} Ha. Provide finite initial velocities or use "
                    "thermostat=langevin for zero-velocity heating."
                )
        if is_langevin:
            velocities = standard_to_lfmiddle_carried(
                self.atoms, velocities, force_for_conversion, self.thermostat.timestep,
            )
            velocity_representation = VELOCITY_REPR_LFMIDDLE_CARRIED

        write_sync_thermo = bool(
            is_langevin and velocity_representation == VELOCITY_REPR_LFMIDDLE_CARRIED
        )
        # Reversible c-rescale reports an effective-energy diagnostic
        # H̃ = K + U + P_0·V − Σ ΔW_ext.  It is only meaningful for the stochastic
        # v-rescale + c-rescale pair (Langevin has no conserved energy; Berendsen
        # is equilibration-only), so the bookkeeping below is gated on that pair.
        track_conserved = (not is_langevin) and self.params.barostat == 'c-rescale'
        # Target-pressure work term P_0·V wants P_0 in Ha/Å³ (bar → eV/Å³ → Ha/Å³).
        p0_ha_per_a3 = self.params.pressure / EV_PER_ANG3_TO_BAR / HARTREE_TO_EV

        manifest_context = build_run_context(
            params=self.params, dof_policy=self._dof_policy, ensemble="npt",
            rng_state_hex=get_rng_state_hex(self._rng),
        )
        self.logger.start_simulation(
            ensemble='npt',
            timestep=self.params.timestep,
            n_steps=n_steps,
            temperature=self.params.temperature,
            atoms=self.atoms,
            pressure=self.params.pressure,
            step_offset=step_offset,
            velocity_representation=velocity_representation,
            n_dof=self._runtime_n_dof,
            dof_description=self._runtime_dof_description,
            write_sync_thermo=write_sync_thermo,
            write_conserved_energy=track_conserved,
            manifest_context=manifest_context,
        )
        self.logger.log_main([
            f"\nStarting NPT simulation "
            f"({self.params.thermostat} + {self.params.barostat})...\n\n"
        ])

        integrator = VelocityVerlet(self.atoms, self.params.timestep)
        v = velocities.copy()

        # Running external-work ledger for the effective-energy diagnostic
        # (v-rescale + c-rescale only); accumulates thermostat, barostat and
        # projection work.
        w_bath = 0.0

        # Cache forces at t=0; the Langevin LFMiddle path reuses the same initial
        # forces for the standard→carried conversion and for the first kick.
        forces = force_for_conversion if force_for_conversion is not None else (
            forces_au(self.atoms)
        )  # Ha/Å → a.u.
        for step in range(1, n_steps + 1):
            abs_step = step_offset + step
            reversible_crescale_path = (
                (not is_langevin) and self.params.barostat == 'c-rescale'
            )
            barostat_due = (
                reversible_crescale_path
                and abs_step % int(self.params.barostat_stride) == 0
            )
            pressure_pre = None
            volume_pre = None
            baro_work = 0.0

            if reversible_crescale_path:
                if barostat_due:
                    # Bernetti-Bussi Reversible Euler, N_P stride:
                    # 1. At the scheduled barostat step, propagate sqrt(V) over
                    #    Δt_barostat = N_P·dt and scale positions/momenta.
                    # 2. Immediately refresh forces at the scaled geometry.
                    # 3. Continue every MD step with split thermostat +
                    #    Velocity-Verlet + split thermostat.
                    volume_pre = self.atoms.get_volume()
                    ke_pre_baro = calculate_kinetic_energy(self.atoms, v)
                    u_pre_baro = evaluate_md_properties(
                        self.atoms, need_stress=True
                    ).energy_ha
                    try:
                        pressure_pre, v = self.barostat.apply(
                            v,
                            timestep_multiplier=float(self.params.barostat_stride),
                        )
                    except MDBarostatClampError as exc:
                        msg = f"C-rescale barostat failed at step {abs_step}: {exc}"
                        self.logger.abort_simulation(reason=msg)
                        raise
                    if getattr(self.barostat, "last_clamped", False) and not self.params.allow_barostat_clamp:
                        msg = (
                            f"C-rescale stability clamp fired at step {abs_step}: the "
                            "per-step volume ratio left the [0.125, 8.0] bound, so the stochastic-"
                            "cell-rescaling NPT ensemble is truncated. Aborting now (set "
                            "allow_barostat_clamp=true to continue an EXPERIMENTAL equilibration; "
                            "the clamp count is recorded in the run manifest)."
                        )
                        self.logger.abort_simulation(reason=msg)
                        raise MDBarostatClampError(msg)
                    self._validate_runtime_cutoff_after_barostat(abs_step)
                    volume_after_baro = self.atoms.get_volume()
                    forces = forces_au(self.atoms)
                    ke_post_baro = calculate_kinetic_energy(self.atoms, v)
                    u_post_baro = evaluate_md_properties(self.atoms).energy_ha
                    if track_conserved:
                        baro_work = (
                            (ke_post_baro - ke_pre_baro)
                            + (u_post_baro - u_pre_baro)
                            + p0_ha_per_a3 * (volume_after_baro - volume_pre)
                        )

                v, delta_w = self.thermostat.apply(v, timestep_fraction=0.5)
                if track_conserved:
                    w_bath += delta_w
                v, forces = integrator.step(v, forces)
                v, delta_w = self.thermostat.apply(v, timestep_fraction=0.5)
                if track_conserved:
                    w_bath += delta_w
                pressure_velocities = None

            elif is_langevin:
                # LFMiddle sequence with carried velocities, then barostat.
                v = integrator.lfmiddle_full_kick(v, forces)
                integrator.half_step_r(v)
                v = self.thermostat.apply(v)
                v, forces = integrator.lfmiddle_post_thermostat(v)
                # LFMiddle stores half-step/carried velocities.  Following the
                # pressure-control lesson from mature MD engines, keep that
                # propagated state intact but compute the kinetic pressure term
                # from a velocity synchronized to the current coordinates.
                pressure_velocities = lfmiddle_carried_to_standard(
                    self.atoms,
                    v,
                    forces,
                    integrator.timestep,
                )
            else:
                # Berendsen remains a sequential equilibration-only coupling:
                # full Velocity Verlet first, then V-rescale, then weak pressure
                # scaling.  It is not advertised as a production NPT integrator.
                v, forces = integrator.step(v, forces)
                v, delta_w = self.thermostat.apply(v)
                pressure_velocities = None

            if not reversible_crescale_path:
                # Barostat decision uses the pre-rescale pressure/volume pair;
                # keep them only as a labeled diagnostic.  The primary
                # thermodynamic record below is the post-rescale state.
                volume_pre = self.atoms.get_volume()
                try:
                    pressure_pre, v = self.barostat.apply(
                        v,
                        pressure_velocities=pressure_velocities,
                    )
                except MDBarostatClampError as exc:
                    msg = f"C-rescale barostat failed at step {abs_step}: {exc}"
                    self.logger.abort_simulation(reason=msg)
                    raise
                if getattr(self.barostat, "last_clamped", False) and not self.params.allow_barostat_clamp:
                    msg = (
                        f"C-rescale stability clamp fired at step {abs_step}: the "
                        "per-step volume ratio left the [0.125, 8.0] bound, so the stochastic-"
                        "cell-rescaling NPT ensemble is truncated. Aborting now (set "
                        "allow_barostat_clamp=true to continue an EXPERIMENTAL equilibration; "
                        "the clamp count is recorded in the run manifest)."
                    )
                    self.logger.abort_simulation(reason=msg)
                    raise MDBarostatClampError(msg)
                self._validate_runtime_cutoff_after_barostat(abs_step)
                forces = forces_au(self.atoms)
                if track_conserved:
                    w_bath += delta_w
                    ke_post_baro = calculate_kinetic_energy(self.atoms, v)
                    u_post_baro = evaluate_md_properties(self.atoms).energy_ha
                    volume_after_baro = self.atoms.get_volume()

            ke_pre_projection = calculate_kinetic_energy(self.atoms, v)
            v, _projection = apply_runtime_motion_projection(
                self.atoms,
                v,
                step=step,
                remove_com_every=self.params.remove_com_every,
                remove_angular_every=self.params.remove_angular_every,
            )
            if not reversible_crescale_path:
                # Sequential paths scale the cell after dynamics, so refresh the
                # force cache once at the post-rescale geometry.  The reversible
                # path keeps its returned Velocity-Verlet force cache instead.
                forces = forces_au(self.atoms)

            current_time     = abs_step * self.params.timestep
            volume_post      = self.atoms.get_volume()
            temperature      = calculate_temperature(self.atoms, v, n_dof=self._runtime_n_dof)
            kinetic_energy   = calculate_kinetic_energy(self.atoms, v)
            # Determine the velocity used for the post-rescale kinetic pressure
            # term.  Langevin carries half-step velocities, so synchronize them to
            # the current coordinates first (matching the pre-rescale barostat
            # decision and the sync-corrected T/KE); v-rescale uses v directly.
            temperature_sync = None
            kinetic_energy_sync = None
            total_energy_sync = None
            if write_sync_thermo:
                v_sync = lfmiddle_carried_to_standard(
                    self.atoms,
                    v,
                    forces,
                    integrator.timestep,
                )
                temperature_sync = calculate_temperature(
                    self.atoms,
                    v_sync,
                    n_dof=self._runtime_n_dof,
                )
                kinetic_energy_sync = calculate_kinetic_energy(self.atoms, v_sync)
                pressure_velocity_post = v_sync
            else:
                pressure_velocity_post = v

            # Single property entry point for the logged post-rescale state: the
            # potential energy and the fresh post-rescale pressure (a real stress
            # evaluation at the post-rescale cell, paired with volume_post) come
            # from one evaluator call.  forces_au() above already populated
            # energy+forces at this geometry, so this adds only the stress pass
            # (no extra backend call; the loop-routing test pins the per-step count).
            props = evaluate_md_properties(
                self.atoms, need_stress=True, velocities_au=pressure_velocity_post
            )
            potential_energy = props.energy_ha   # Ha
            pressure_post = props.pressure_bar
            if write_sync_thermo:
                total_energy_sync = kinetic_energy_sync + potential_energy

            # Effective-energy diagnostic H̃ = K + U + P_0·V − Σ ΔW_ext.  The barostat
            # injects ΔW_baro = Δ(K + U + P_0·V) across its volume/momentum
            # rescale; the runtime projection injects ΔW_proj = ΔK.  Both join
            # the same ledger as the thermostat work, so H̃ drift is tracked as
            # an integration-quality diagnostic.
            conserved = None
            if track_conserved:
                proj_work = kinetic_energy - ke_pre_projection
                w_bath += baro_work + proj_work
                conserved = (
                    kinetic_energy + potential_energy
                    + p0_ha_per_a3 * volume_post - w_bath
                )

            self.logger.log_step(
                step=abs_step,
                time=current_time,
                temperature=temperature,
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                total_energy=kinetic_energy + potential_energy,
                atoms=self.atoms,
                velocities=v,
                pressure=pressure_post,
                volume=volume_post,
                pressure_pre=pressure_pre,
                volume_pre=volume_pre,
                rng_state=get_rng_state_hex(self._rng),
                rst_every=self.params.rst_every,
                conserved_energy=conserved,
                velocity_representation=velocity_representation,
                temperature_sync=temperature_sync,
                kinetic_energy_sync=kinetic_energy_sync,
                total_energy_sync=total_energy_sync,
            )

        # Record the barostat stability-clamp summary into the manifest run block
        # before it is written. This is the same dict the logger holds; build_md_manifest
        # snapshots it at end_simulation. count=0 (the production expectation) is recorded
        # too, so the manifest always states the clamp status of a c-rescale run.
        manifest_context["barostat_clamps"] = {
            "count": int(getattr(self.barostat, "clamp_count", 0)),
            "max_abs_log_excursion": float(getattr(self.barostat, "max_abs_log_excursion", 0.0)),
            "allowed": bool(self.params.allow_barostat_clamp),
        }
        self.logger.end_simulation(
            atoms=self.atoms,
            final_velocities=v,
            rng_state=get_rng_state_hex(self._rng),
            velocity_representation=velocity_representation,
        )
        self.logger.log_main(["\nNPT simulation completed successfully.\n"])
        return v, velocity_representation
