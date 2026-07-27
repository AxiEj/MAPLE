from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "maple").is_dir() and (parent / "pyproject.toml").is_file()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.calculator.extra_correction.implicit.correction import ImplicitSolvationCorrection
from maple.function.read.command_control import CommandControl
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.function.route2_smd_profiles import (
    PCMSOLVER_INTRINSIC_CAVITY_PROFILE,
    PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE,
)
OUT = Path(os.environ["OUT_DIR"]).resolve()
SELECTION_PATH = Path(__file__).with_name("selection.json")
BASE = ROOT / ".omx/benchmarks/route2-macepolar-smd"
selection = json.loads(SELECTION_PATH.read_text())
prepared = json.loads((BASE / "prepared.json").read_text())
by_id = {item["compound_id"]: item for item in prepared["candidates"]}
device = "cuda" if torch.cuda.is_available() else "cpu"
profiles = {
    "local_jet": PCMSOLVER_INTRINSIC_CAVITY_PROFILE,
    "exact_gto": PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE,
}
def params(profile: str):
    return CommandControl.from_settings([
        "#model=macepol-m",
        "#sp",
        "#solv(implicit=water,method=smd,provider=pcmsolver,"
        f"profile={profile},response=scf,standard_state=1m,experimental=true)",
    ]).as_dict()
def sync():
    if device == "cuda":
        torch.cuda.synchronize()
first = by_id[selection["records"][0]["compound_id"]]
first_atoms = MOL2Reader(str(BASE / first["mol2_relative_path"]), charge=0, mult=1)
base_params = params(PCMSOLVER_INTRINSIC_CAVITY_PROFILE)
started = time.perf_counter()
calculator = SetCalculator(
    device,
    base_params["model"],
    str(OUT / "model.out"),
    atoms=first_atoms,
    implicit="smd",
    solvent="water",
    model_options=base_params.get("model_options"),
    solvation_options=base_params["solv"],
    charge_options={},
).set_calculator()
sync()
model_load_seconds = time.perf_counter() - started
records = []
panel_started = time.perf_counter()
for index, selected in enumerate(selection["records"]):
    candidate = by_id[selected["compound_id"]]
    if candidate["partition"] != "development":
        raise RuntimeError(f"{candidate['compound_id']} is not development.")
    if candidate["name"].lower() != selected["name"].lower():
        raise RuntimeError(f"{candidate['compound_id']} name drifted.")
    atoms = MOL2Reader(str(BASE / candidate["mol2_relative_path"]), charge=0, mult=1)
    order = ["local_jet", "exact_gto"] if index % 2 == 0 else ["exact_gto", "local_jet"]
    for label in order:
        profile = profiles[label]
        route_params = params(profile)
        target = OUT / candidate["compound_id"] / label / "maple.out"
        target.parent.mkdir(parents=True)
        calculator.solvent_correction = ImplicitSolvationCorrection(
            atoms, {}, route_params["solv"], output=str(target)
        )
        calculator.reset()
        atoms.calc = calculator
        sync()
        started = time.perf_counter()
        atoms.get_potential_energy()
        sync()
        elapsed = time.perf_counter() - started
        solvation = calculator.results["solvation"]
        predicted = float(solvation["delta_g_solv_hartree"]) * 627.5094740631
        experimental = float(candidate["experimental_kcal_mol"])
        audit = json.loads(
            (Path(calculator.solvent_correction.audit_dir) / "route2-result.json").read_text()
        )
        record = {
            "selection_index": index,
            "compound_id": candidate["compound_id"],
            "name": candidate["name"],
            "class": selected["class"],
            "functional_groups": candidate["functional_groups"],
            "profile_order": order,
            "label": label,
            "profile": profile,
            "experimental_kcal_mol": experimental,
            "experimental_uncertainty_kcal_mol": float(
                candidate["experimental_uncertainty_kcal_mol"]
            ),
            "predicted_kcal_mol": predicted,
            "signed_error_kcal_mol": predicted - experimental,
            "absolute_error_kcal_mol": abs(predicted - experimental),
            "wall_seconds": elapsed,
            "iterations": int(audit["iterations"]),
            "density_residual_inf_e": float(
                audit["fixed_point"]["density_residual_inf_e"]
            ),
            "cavity_tesserae": int(audit["cavity_stability"]["attempts"][0]["cavity_tesserae"]),
            "native_warning_count": int(
                audit["pcmsolver_diagnostics"]["native_stderr_warning_count"]
            ),
            "pedra_warning_count": int(
                audit["pcmsolver_diagnostics"]["pedra_warning_count"]
            ),
            "components_kcal_mol": {
                key: float(value) * 627.5094740631
                for key, value in solvation["components_hartree"].items()
            },
        }
        records.append(record)
        print(
            f"{index + 1:02d}/10 {candidate['name']:<15} {label:<9} "
            f"pred={predicted:8.3f} exp={experimental:7.3f} "
            f"abs={abs(predicted - experimental):6.3f} "
            f"t={elapsed:5.2f}s it={audit['iterations']}",
            flush=True,
        )
def metrics(label: str):
    subset = [record for record in records if record["label"] == label]
    errors = np.asarray([record["signed_error_kcal_mol"] for record in subset])
    times = np.asarray([record["wall_seconds"] for record in subset])
    return {
        "record_count": len(subset),
        "mean_absolute_error_kcal_mol": float(np.mean(np.abs(errors))),
        "root_mean_square_error_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "mean_signed_error_kcal_mol": float(np.mean(errors)),
        "maximum_absolute_error_kcal_mol": float(np.max(np.abs(errors))),
        "mean_wall_seconds": float(np.mean(times)),
        "total_wall_seconds": float(np.sum(times)),
        "mean_iterations": float(np.mean([record["iterations"] for record in subset])),
    }
aggregate = {label: metrics(label) for label in profiles}
pairs = []
for selected in selection["records"]:
    local = next(
        record for record in records
        if record["compound_id"] == selected["compound_id"] and record["label"] == "local_jet"
    )
    exact = next(
        record for record in records
        if record["compound_id"] == selected["compound_id"] and record["label"] == "exact_gto"
    )
    pairs.append({
        "compound_id": selected["compound_id"],
        "name": selected["name"],
        "class": selected["class"],
        "local_absolute_error_kcal_mol": local["absolute_error_kcal_mol"],
        "exact_absolute_error_kcal_mol": exact["absolute_error_kcal_mol"],
        "exact_minus_local_absolute_error_kcal_mol": (
            exact["absolute_error_kcal_mol"] - local["absolute_error_kcal_mol"]
        ),
        "winner": (
            "exact_gto"
            if exact["absolute_error_kcal_mol"] < local["absolute_error_kcal_mol"]
            else "local_jet"
        ),
    })
gate_spec = selection["decision_gates"]
acetone = next(pair for pair in pairs if pair["compound_id"] == "mobley_3867265")
gates = {
    "candidate_panel_mae_below_1": (
        aggregate["exact_gto"]["mean_absolute_error_kcal_mol"]
        < gate_spec["candidate_panel_mae_kcal_mol_max"]
    ),
    "candidate_mae_not_above_control": (
        aggregate["exact_gto"]["mean_absolute_error_kcal_mol"]
        <= aggregate["local_jet"]["mean_absolute_error_kcal_mol"]
    ),
    "candidate_acetone_below_1": (
        acetone["exact_absolute_error_kcal_mol"]
        < gate_spec["candidate_acetone_absolute_error_kcal_mol_max"]
    ),
    "candidate_max_error_below_2": (
        aggregate["exact_gto"]["maximum_absolute_error_kcal_mol"]
        < gate_spec["candidate_maximum_absolute_error_kcal_mol_max"]
    ),
    "all_native_warning_counts_zero": all(
        record["native_warning_count"] == 0 for record in records
    ),
}
diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
payload = {
    "artifact": "route2-intrinsic-exact-gto-ten-panel-v1",
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "selection_path": str(SELECTION_PATH),
    "selection_sha256": hashlib.sha256(SELECTION_PATH.read_bytes()).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "device": device,
    "model_load_seconds": model_load_seconds,
    "actual_panel_wall_seconds": time.perf_counter() - panel_started,
    "records": sorted(records, key=lambda record: (record["selection_index"], record["label"])),
    "pairs": pairs,
    "aggregate": aggregate,
    "gates": gates,
    "pass": all(gates.values()),
    "claim_boundary": selection["claim_boundary"],
}
(OUT / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "aggregate": aggregate,
    "gates": gates,
    "pass": all(gates.values()),
    "wins": {label: sum(pair["winner"] == label for pair in pairs) for label in profiles},
}, indent=2))
print("RESULT_PATH=" + str(OUT / "results.json"))
