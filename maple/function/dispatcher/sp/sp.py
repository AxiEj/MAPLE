import math
from typing import Union, List, Optional
from dataclasses import dataclass

from ase import Atoms

from ..jobABC import JobABC
from maple.function.timer import timer

@dataclass
class SPParams:
    """Parameters for Single Point calculation."""
    verbose: int = 0  # 0=coordinates+energy+charge/mult, 1=+gradients

class SinglePoint(JobABC):

    def __init__(self, output: str, atoms: Union[Atoms, List[Atoms]],
                 paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.is_trajectory = isinstance(atoms, list)

        # Initialize params
        self.params = self._init_params(SPParams, paras, ("sp", "SP"))
        self.verbose = self.params.verbose

    def run(self):
        if self.is_trajectory:
            self._run_trajectory()
        else:
            self._run_single()

    def _run_single(self):
        """Original single-point calculation logic."""
        with timer("Single Point Energy Calculation"):
            energy = self.atoms.get_potential_energy()
            self.log_info(self._single_energy_lines(energy))

    def _single_energy_lines(self, energy: float) -> list:
        """Return single-structure SP result lines for the selected verbosity."""
        lines = ["\n"]
        lines.extend(self._charge_mult_lines(self.atoms))
        lines.append(f"Energy: {energy:.10f} Hartree\n")
        lines.extend(self._solvation_lines(self.atoms))
        if self.verbose >= 1:
            lines.extend(self._gradient_lines(self.atoms))
        return lines

    @staticmethod
    def _solvation_lines(atoms: Atoms) -> list:
        results = getattr(getattr(atoms, "calc", None), "results", {})
        if not isinstance(results, dict) or "solvation" not in results:
            return []
        solvation = results["solvation"]
        required = {
            "energy_hartree",
            "gas_energy_hartree",
            "combined_energy_hartree",
            "components_hartree",
            "provenance",
            "ase_free_energy_is_thermochemical_gibbs",
        }
        if not isinstance(solvation, dict) or not required.issubset(solvation):
            raise ValueError("Malformed solvation result: required core fields are missing.")
        if (
            not isinstance(solvation["components_hartree"], dict)
            or not isinstance(solvation["provenance"], dict)
            or solvation["ase_free_energy_is_thermochemical_gibbs"] is not False
        ):
            raise ValueError("Malformed solvation result: core field types are invalid.")
        gas = solvation["gas_energy_hartree"]
        combined = solvation["combined_energy_hartree"]
        provenance = solvation["provenance"]
        if provenance.get("reference_only") is True:
            delta = solvation.get("reference_correction_hartree")
            if delta is None:
                raise ValueError(
                    "Malformed solvation result: reference correction is missing."
                )
            return [
                f"Gas-phase MLIP energy: {float(gas):.10f} Hartree\n",
                f"Numerical reference correction: {float(delta):.10f} Hartree\n",
                f"Combined E_MLIP(gas)+reference correction: {float(combined):.10f} Hartree\n",
                "This reference result is not a complete solvation free energy.\n",
            ]
        if provenance.get("absolute_solvation_free_energy_claim") is False:
            correction = solvation.get("cluster_continuum_correction_hartree")
            if correction is None:
                raise ValueError(
                    "Malformed solvation result: cluster-continuum correction is missing."
                )
            return [
                f"Gas-phase MLIP cluster energy: {float(gas):.10f} Hartree\n",
                (
                    "Outer PB/GB correction for the prebuilt cluster: "
                    f"{float(correction):.10f} Hartree\n"
                ),
                (
                    "Combined fixed-shell cluster-continuum configurational "
                    f"potential: {float(combined):.10f} Hartree\n"
                ),
                (
                    "This value is not an absolute solvation free energy; "
                    "cluster formation/occupancy, standard-state, solvent-cluster "
                    "reference, and ensemble terms are not computed.\n"
                ),
            ]
        delta = solvation.get("delta_g_solv_hartree")
        if delta is None:
            raise ValueError("Malformed solvation result: Delta G_solv is missing.")
        standard_state = provenance.get("standard_state", "provider-defined")
        lines = [
            f"Gas-phase MLIP energy: {float(gas):.10f} Hartree\n",
            (
                "Solvation free-energy correction (Delta G_solv, "
                f"{standard_state}): {float(delta):.10f} Hartree\n"
            ),
            (
                "Combined E_MLIP(gas)+Delta G_solv: "
                f"{float(combined):.10f} Hartree\n"
            ),
            (
                "ASE free_energy is the combined electronic-plus-solvation "
                "value, not a thermochemical Gibbs free energy.\n"
            ),
        ]
        lines.extend(SinglePoint._private_chagb_torch_lines(solvation, provenance))
        return lines

    @staticmethod
    def _private_chagb_torch_lines(solvation: dict, provenance: dict) -> list:
        """Render the current dual-device private CHA runtime contract."""
        if provenance.get("runtime_provider") != "private-torch-cha":
            return []
        def malformed(detail: str) -> None:
            raise ValueError(f"Malformed private CHA Torch SP provenance: {detail}.")

        engine = provenance.get("engine")
        api_version = engine.get("api_version") if isinstance(engine, dict) else None
        profile = provenance.get("arithmetic_profile")
        if (api_version, profile) in {
            (2, "cpu-fp32-exact-sp-v1"),
            (3, "cuda-fp32-sp-v1"),
        }:
            # These exact historical engine/profile pairs remain archival
            # evidence and retain their original basic SP output.
            return []
        if profile == "torch-fp32-sp-v1" or api_version == 4:
            if not isinstance(engine, dict) or api_version != 4:
                malformed("torch-fp32-sp-v1 requires an API 4 engine identity")
        else:
            malformed("engine and arithmetic_profile are not a recognized pair")
        if profile != "torch-fp32-sp-v1":
            malformed("arithmetic_profile must be torch-fp32-sp-v1")
        if provenance.get("supports_forces") is not False:
            malformed("supports_forces must be false")

        components = solvation.get("components_hartree")
        runtime = provenance.get("runtime")
        if not isinstance(components, dict) or not isinstance(runtime, dict):
            malformed("components_hartree and runtime must be mappings")
        required_components = ("polar", "cavity", "dispersion")
        required_runtime = (
            "device",
            "dtype",
            "workspace_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        )
        if any(name not in components for name in required_components) or any(
            name not in runtime for name in required_runtime
        ):
            malformed("required component or runtime fields are missing")

        device = provenance.get("device")
        if not isinstance(device, str) or runtime["device"] != device:
            malformed("runtime device must match the selected device")
        if device == "cpu":
            expected_platform = "CPU"
            if runtime["peak_allocated_bytes"] is not None or runtime["peak_reserved_bytes"] is not None:
                malformed("CPU CUDA peak-memory fields must be null")
        else:
            kind, separator, index = device.partition(":")
            if (
                kind != "cuda"
                or separator != ":"
                or not index.isdigit()
                or str(int(index)) != index
            ):
                malformed("device must be cpu or canonical cuda:N")
            expected_platform = "CUDA"
            if any(
                type(runtime[name]) is not int or runtime[name] < 0
                for name in ("peak_allocated_bytes", "peak_reserved_bytes")
            ):
                malformed("CUDA peak-memory fields must be non-negative bytes")
        if provenance.get("platform") != expected_platform:
            malformed("platform must match the selected device")
        if runtime["dtype"] != "torch.float32":
            malformed("dtype must be torch.float32")
        if type(runtime["workspace_bytes"]) is not int or runtime["workspace_bytes"] < 0:
            malformed("workspace must be non-negative bytes")
        try:
            component_values = {
                name: float(components[name]) for name in required_components
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Malformed private CHA Torch SP provenance: components must be numeric."
            ) from exc
        if not all(math.isfinite(value) for value in component_values.values()):
            malformed("components must be finite")

        lines = [
            (
                "Private CHA Torch runtime: "
                f"device={device}, platform={expected_platform}, "
                f"dtype={runtime['dtype']}, "
                f"profile={provenance['arithmetic_profile']}\n"
            ),
            f"Private CHA polar component: {component_values['polar']:.10f} Hartree\n",
            f"Private CHA cavity component: {component_values['cavity']:.10f} Hartree\n",
            f"Private CHA dispersion component: {component_values['dispersion']:.10f} Hartree\n",
            f"Private CHA workspace: {runtime['workspace_bytes']} bytes\n",
        ]
        if expected_platform == "CUDA":
            lines.append(
                "Private CHA CUDA memory: "
                f"peak allocated={runtime['peak_allocated_bytes']} bytes, "
                f"peak reserved={runtime['peak_reserved_bytes']} bytes\n"
            )
        else:
            lines.append("Private CHA CUDA memory: not applicable on CPU\n")
        lines.append("Private CHA capability: supports_forces=false (energy-only)\n")
        return lines

    def _charge_mult_lines(self, atoms: Atoms) -> list:
        """Return charge and multiplicity metadata lines for SP output."""
        charge = atoms.info.get('charge', 0)
        mult = atoms.info.get('mult', 1)
        return [f"Charge: {charge}, Multiplicity: {mult}\n"]

    def _gradient_lines(self, atoms: Atoms) -> list:
        """Return per-atom energy gradients for detailed SP output."""
        forces = atoms.get_forces()
        gradients = -forces
        symbols = atoms.get_chemical_symbols()
        lines = [
            "\nGradients (Hartree/Angstrom):\n",
            "  Gradient = -Force\n",
            "  Atom  El"
            "        Gx              Gy              Gz\n",
        ]
        for i, (sym, gradient) in enumerate(zip(symbols, gradients), start=1):
            lines.append(
                f"  {i:<4} {sym:<2}"
                f" {gradient[0]:>15.8f} {gradient[1]:>15.8f} {gradient[2]:>15.8f}\n"
            )
        return lines

    def _trajectory_frame_lines(self, idx: int, atoms_frame: Atoms, energy_hartree: float) -> list:
        """Return trajectory-frame SP result lines for the selected verbosity."""
        lines = [
            f"\n{('Frame ' + str(idx)):=^80}\n",
        ]
        lines.extend(self._charge_mult_lines(atoms_frame))
        lines.extend([
            f"Energy: {energy_hartree:.10f} Hartree\n\n",
            "Coordinates (Angstrom):\n",
        ])
        symbols = atoms_frame.get_chemical_symbols()
        positions = atoms_frame.get_positions()
        for i, (sym, pos) in enumerate(zip(symbols, positions), start=1):
            lines.append(f"  {i:<4} {sym:<2} {pos[0]:>15.8f} {pos[1]:>15.8f} {pos[2]:>15.8f}\n")
        if self.verbose >= 1:
            lines.extend(self._gradient_lines(atoms_frame))
        lines.append("=" * 80 + "\n")
        return lines

    def _run_trajectory(self):
        """Process multiple structures sequentially."""
        with timer("Single Point Energy Calculation (Trajectory)"):
            n_frames = len(self.atoms)
            self.log_info([f"\nProcessing {n_frames} structures from trajectory...\n"])
            self.log_info(["=" * 80 + "\n"])

            energies_hartree = []

            for idx, atoms_frame in enumerate(self.atoms, start=1):
                # Calculate energy
                energy_hartree = atoms_frame.get_potential_energy()
                energies_hartree.append(energy_hartree)

                self.log_info(self._trajectory_frame_lines(idx, atoms_frame, energy_hartree))

            # Summary (always shown)
            self.log_info([f"\n{' SUMMARY ':=^80}\n"])
            self.log_info([f"Total frames processed: {n_frames}\n"])
            self.log_info([f"Energy range: {min(energies_hartree):.10f} to {max(energies_hartree):.10f} Hartree\n"])
            energy_span = max(energies_hartree) - min(energies_hartree)
            self.log_info([f"Energy span: {energy_span:.10f} Hartree ({energy_span * 627.509:.4f} kcal/mol)\n"])
            self.log_info(["=" * 80 + "\n"])
