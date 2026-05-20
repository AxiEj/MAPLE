"""MD physics-semantics policy (WS0).

This module concentrates the *parameter-dependent* MD admission logic that must
run after the ensemble parameters are resolved, keeping it out of ``utils.py``
(physics calculators / unit conversions / velocity tools) and out of the
calculator-capability gate (``validate_md_capabilities``, which is
parameter-independent).

Three concerns live here:

* WS0-A — :func:`resolve_md_dof_policy` resolves the active number of degrees of
  freedom in an *operator-aware* way.  The deciding question for an
  init-projected COM/rotation mode is not the ensemble or the removal cadence
  alone, but whether the operator acting each step re-excites that mode:

  ======================  ============================  ===========================
  Operator (per step)     Re-excites init-projected     DOF rule for that mode
                          COM/rotation mode?
  ======================  ============================  ===========================
  NVE deterministic VV    No (P, L conserved)           subtract if projected or
  v-rescale (alpha*v)     No (global scalar)            removed at any cadence
  c-rescale (v/mu)        No (global scalar)            (it stays at zero)
  Langevin + every>1      Yes, repopulated betw. resets don't subtract
  Langevin + init-only    Yes, repopulated every step   don't subtract
  Langevin + every==1     pinned at zero each step      subtract
  ======================  ============================  ===========================

  Re-excitation is the deciding question, not the cadence alone: a non-re-exciting
  operator cannot move a zero mode off zero, so projecting/removing it subtracts;
  Langevin repopulates it, so only an every-step runtime projection subtracts.

  ``init_n_dof`` (the basis the initial velocity draw was projected onto) and
  ``runtime_n_dof`` (the basis temperature/thermostat/summary use during
  production) may legitimately differ — most importantly for Langevin, where the
  initial draw removes COM/rotation but per-atom noise repopulates them.

* WS0-B — :func:`validate_md_semantics` hard-rejects any ASE constraint on the
  atoms.  MAPLE's hand-written Velocity Verlet has no SHAKE/RATTLE/SETTLE, no
  constraint-aware velocity projection, and no constraint DOF accounting, so
  constrained MD would silently produce fake kinetic energy / temperature.

* WS0-C — :func:`validate_md_semantics` rejects partial periodicity for
  production runs.  ``allow_partial_pbc`` is a development/experimental escape
  hatch only: it does not pass production validation and the run banner is
  marked accordingly.
"""

from dataclasses import dataclass
from typing import List, Tuple

from ase import Atoms

from .utils import (
    get_initialization_dof_policy,
    get_n_dof_from_policy,
    get_runtime_dof_policy,
    is_linear_molecule,
)


@dataclass(frozen=True)
class MDDOFPolicy:
    """Resolved degree-of-freedom policy for one MD run.

    ``init_n_dof`` scales the initial velocity draw (it must match the
    projection actually applied at initialization).  ``runtime_n_dof`` drives
    runtime temperature, the thermostat target, and the summary/report.  They
    differ when an init-only projection is later re-excited (Langevin).
    """

    init_n_dof: int
    runtime_n_dof: int
    init_description: str
    runtime_description: str
    warnings: Tuple[str, ...]


def _mode_subtracted(*, init_projected: bool, runtime_every: int, operator_reexcites: bool) -> bool:
    """Decide whether one COM/rotation mode is held at ~zero kinetic energy.

    Re-excitation by the per-step operator is the deciding question (the old
    cadence-only rule was both too coarse and operator-blind — it treated
    v-rescale like Langevin).  The decision is taken at the moment temperature is
    measured (after any runtime projection):

    * Re-exciting operator (Langevin, per-atom noise): the mode is repopulated
      every step, so only an every-step runtime projection (``== 1``) pins it at
      zero by logging time.  An init-only projection or an intermittent
      (``> 1``) removal does not reduce the active DOF.
    * Non-re-exciting operator (NVE deterministic VV, v-rescale / c-rescale
      global scalars): the operator cannot move a zero mode off zero, so once the
      mode is projected at init *or* removed at runtime (any cadence) it stays at
      zero and is a genuine constraint → subtract.
    """
    if operator_reexcites:
        return runtime_every == 1
    return bool(init_projected) or runtime_every > 0


def _operator_reexcites(params) -> bool:
    """Return True when the per-step operator repopulates init-projected modes.

    Only Langevin (per-atom Ornstein-Uhlenbeck noise) creates new velocity
    directions.  v-rescale (``alpha * v``) and c-rescale (``v / mu``) are global
    scalars and NVE is deterministic, so none of them can re-excite a mode that
    was projected to zero.
    """
    return str(getattr(params, "thermostat", "") or "").lower() == "langevin"


def resolve_md_dof_policy(atoms: Atoms, params, ensemble: str) -> MDDOFPolicy:
    """Resolve the init/runtime DOF policy for an MD run (WS0-A)."""
    n_atoms = len(atoms)
    is_pbc = any(atoms.pbc)

    # Initialization projection flags, mirroring ``initialize_velocities``:
    # remove_angular (or the legacy remove_rotation) implies remove_com.
    remove_rotation = bool(getattr(params, "remove_rotation", False))
    remove_angular = bool(getattr(params, "remove_angular", False)) or remove_rotation
    remove_com = bool(getattr(params, "remove_com", True)) or remove_angular
    remove_com_every = int(getattr(params, "remove_com_every", 0) or 0)
    remove_angular_every = int(getattr(params, "remove_angular_every", 0) or 0)
    operator_reexcites = _operator_reexcites(params)

    # init_n_dof is taken straight from the projection applied at init so the
    # initial velocity rescaling target is self-consistent with that draw.
    init_policy = get_initialization_dof_policy(
        atoms, remove_com=remove_com, remove_angular=remove_angular
    )
    init_n_dof = get_n_dof_from_policy(init_policy, n_atoms=n_atoms)

    # Reuse the runtime policy only to harvest its PBC advisory warnings; the
    # DOF count itself is recomputed below with operator awareness.
    runtime_warn_policy = get_runtime_dof_policy(
        atoms,
        remove_com_every=remove_com_every,
        remove_angular_every=remove_angular_every,
    )

    # COM translation (3 DOF). Under PBC the runtime angular cadence is ignored,
    # but COM removal still applies.
    com_subtracted = _mode_subtracted(
        init_projected=remove_com,
        runtime_every=remove_com_every,
        operator_reexcites=operator_reexcites,
    )

    # Rigid-body rotation is only defined for non-periodic systems.
    rotation_dof = 0
    rotation_subtracted = False
    if not is_pbc:
        rotation_subtracted = _mode_subtracted(
            init_projected=remove_angular,
            runtime_every=remove_angular_every,
            operator_reexcites=operator_reexcites,
        )
        if rotation_subtracted:
            rotation_dof = 2 if is_linear_molecule(atoms) else 3

    runtime_n_dof = 3 * n_atoms
    if com_subtracted:
        runtime_n_dof -= 3
    runtime_n_dof -= rotation_dof
    runtime_n_dof = max(runtime_n_dof, 1)

    warnings: List[str] = []
    for warning in list(init_policy["warnings"]) + list(runtime_warn_policy["warnings"]):
        if warning not in warnings:
            warnings.append(warning)

    return MDDOFPolicy(
        init_n_dof=init_n_dof,
        runtime_n_dof=runtime_n_dof,
        init_description=_describe(is_pbc, init_n_dof, n_atoms, basis="init draw"),
        runtime_description=_describe(
            is_pbc, runtime_n_dof, n_atoms, basis=_runtime_reason(ensemble, params, operator_reexcites)
        ),
        warnings=tuple(warnings),
    )


def _runtime_reason(ensemble: str, params, operator_reexcites: bool) -> str:
    if operator_reexcites:
        return "Langevin re-excites projected modes"
    thermostat = str(getattr(params, "thermostat", "") or "").lower()
    if thermostat in {"v-rescale"}:
        return "v-rescale keeps projected modes at zero"
    return f"{str(ensemble).lower()} runtime"


def _describe(is_pbc: bool, n_dof: int, n_atoms: int, *, basis: str) -> str:
    removed = 3 * n_atoms - n_dof
    label = "PBC" if is_pbc else "isolated"
    if removed <= 0:
        return f"{label}: 3N = {n_dof} ({basis})"
    return f"{label}: 3N - {removed} = {n_dof} ({basis})"


# Roadmap pointer reused by the constraint rejection message.
_CONSTRAINED_MD_ROADMAP = (
    "Constrained MD (SHAKE/RATTLE/SETTLE for FixInternals; velocity zeroing + "
    "DOF/KE exclusion + barostat scaling + RST persistence for FixAtoms) is a "
    "roadmap item and is not implemented in this MD engine."
)

_PARTIAL_PBC_BANNER = (
    "\n"
    "  ┌─ PARTIAL-PBC: EXPERIMENTAL — NOT PRODUCTION VALIDATED ────────────────┐\n"
    "  │  This system mixes periodic and non-periodic directions (e.g. a slab).│\n"
    "  │  Long-range electrostatics, stress, neighbor cutoffs and the          │\n"
    "  │  minimum-image convention are NOT validated for partial periodicity.  │\n"
    "  │  allow_partial_pbc=true is a development escape hatch only; it does    │\n"
    "  │  not pass production validation.  Use full 3-D PBC for production.     │\n"
    "  └───────────────────────────────────────────────────────────────────────┘\n"
)


def validate_md_semantics(atoms: Atoms, params, ensemble: str) -> List[str]:
    """Parameter-dependent MD admission checks (WS0-B, WS0-C).

    Returns a list of advisory messages (e.g. the partial-PBC experimental
    banner) for the caller to log.  Raises ``ValueError`` on a hard reject.
    """
    advisories: List[str] = []

    # WS0-B — constraints are a fake-simulation entry point until constrained MD
    # is actually implemented; reject hard, with no escape hatch.
    constraints = getattr(atoms, "constraints", None)
    if constraints:
        kinds = sorted({type(c).__name__ for c in constraints})
        raise ValueError(
            f"{str(ensemble).upper()} MD does not support ASE constraints "
            f"({', '.join(kinds)}). " + _CONSTRAINED_MD_ROADMAP + " Remove the "
            "constraints (C/B/A/D commands) to run MD, or use opt/scan/ts/irc "
            "for constrained workflows."
        )

    # WS0-C — partial periodicity has no production policy.
    pbc = [bool(flag) for flag in atoms.pbc]
    if any(pbc) and not all(pbc):
        if not bool(getattr(params, "allow_partial_pbc", False)):
            raise ValueError(
                f"{str(ensemble).upper()} MD received partial periodicity "
                f"(pbc={pbc}). Production MD supports full three-dimensional PBC "
                "only; long-range/stress/cutoff/minimum-image assumptions are "
                "unvalidated for slab/partial-PBC systems. Set allow_partial_pbc="
                "true to run an EXPERIMENTAL (non-production) partial-PBC trajectory."
            )
        advisories.append(_PARTIAL_PBC_BANNER)

    return advisories
