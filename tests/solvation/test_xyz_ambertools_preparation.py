"""Cartesian preparation is an explicit, audited, atom-preserving boundary."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

from ase import Atoms
import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit import ambertools, charges, correction
from maple.function.read.charge_options import normalize_charge_options
from maple.function.read.filereader.mol2_reader import MOL2Reader, MOL2_ATOM_ID_ARRAY
from maple.function.read.input_reader import InputReader

POSITIONS = np.array(
    [
        [0.000000123456789, 0, 0],
        [0.957200123456789, 0, 0],
        [-0.239987076543211, 0.927297123456789, 0],
    ]
)


def water():
    atoms = Atoms("OH2", positions=POSITIONS)
    atoms.info.update(charge=0, mult=1)
    return atoms


def read_job(tmp_path, body, charge="#charge(source=maple)", extra=""):
    inp = tmp_path / "job.inp"
    inp.write_text(
        f"#model=ani2x\n#sp\n{charge}\n"
        "#solv(implicit=water,method=gb,experimental=true)\n"
        f"{extra}\n{body}\n"
    )
    return InputReader()(str(inp), str(tmp_path / "job.out"))


def test_source_default_is_coherent_without_overriding_explicit_mol2():
    assert (
        normalize_charge_options({"executable": "/custom/antechamber"})["source"]
        == "maple"
    )
    assert normalize_charge_options({"source": "mol2"})["source"] == "mol2"
    with pytest.raises(ValueError, match="does not run a charge executable"):
        normalize_charge_options(
            {"source": "mol2", "executable": "/custom/antechamber"}
        )


@pytest.mark.parametrize("inline", [False, True])
def test_reader_accepts_single_cartesian_without_running_tools(
    tmp_path, monkeypatch, inline
):
    monkeypatch.setattr(
        ambertools.subprocess, "run", lambda *a, **k: pytest.fail("reader ran tool")
    )
    coords = "\n".join(
        f"{s} " + " ".join(map(repr, p.tolist()))
        for s, p in zip(["O", "H", "H"], POSITIONS)
    )
    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nwater\n" + coords + "\n")
    body = "0 1\n" + (coords if inline else f"XYZ {xyz}")
    atoms = read_job(tmp_path, body, "#charge(executable=/custom/antechamber)")
    np.testing.assert_array_equal(atoms.positions, POSITIONS)
    assert "mol2" not in atoms.info
    assert atoms.info["_maple_charge_options"]["source"] == "maple"
    assert atoms.info["_maple_cartesian_source"]["sha256"]


@pytest.mark.parametrize(
    "body, match",
    [
        ("O 0 0 0\nH .96 0 0\nH -.24 .93 0", "explicit '0 1'"),
        ("1 1\nO 0 0 0", "neutral closed-shell"),
        ("0 2\nO 0 0 0", "neutral closed-shell"),
        ("0 1\nO 0 0 0\n&\n0 1\nO 4 0 0", "exactly one"),
    ],
)
def test_reader_retains_explicit_domain(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        read_job(tmp_path, body)


def test_xyz_never_overrides_source_mol2(tmp_path):
    with pytest.raises(ValueError, match="source=mol2.*MOL2"):
        read_job(tmp_path, "0 1\nO 0 0 0", "#charge(source=mol2)")


@pytest.mark.parametrize(
    "contents",
    [
        "3\nwater\nO 0 0 0\nH .96 0 0\n",  # truncated
        "1\nframe1\nO 0 0 0\n1\nframe2\nO 2 0 0\n",
        "1\nwater\nO 0 0 0 trailing\n",
    ],
)
def test_implicit_xyz_requires_one_exact_frame(tmp_path, contents):
    xyz = tmp_path / "bad.xyz"
    xyz.write_text(contents)
    with pytest.raises(
        ValueError, match="single.*XYZ|XYZ.*single|XYZ.*count|XYZ.*columns"
    ):
        read_job(tmp_path, f"0 1\nXYZ {xyz}")


def test_reader_rejects_periodic_xyz(tmp_path):
    xyz = tmp_path / "periodic.xyz"
    xyz.write_text('1\nLattice="10 0 0 0 10 0 0 0 10" pbc="T T T"\nO 0 0 0\n')
    with pytest.raises(ValueError, match="non-periodic"):
        read_job(tmp_path, f"0 1\nXYZ {xyz}")


def fake_ambertools(monkeypatch, water_mol2, *, corrupt=None):
    calls = []
    monkeypatch.setattr(
        ambertools.shutil, "which", lambda name: "/opt/amber/bin/antechamber"
    )

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("-o") + 1])
        if "-fi" in command and command[command.index("-fi") + 1] == "gcrt":
            assert "-c" not in command, "topology inference must not compute charges"
            assert command[command.index("-j") + 1] == "4"
            text = water_mol2.read_text()
            text = text.replace("0.957200", "0.957000").replace(
                "-0.239987", "-0.240000"
            )
            text = text.replace("0.927297", "0.927000")
            if corrupt:
                text = corrupt(text)
            output.write_text(text)
        elif "-fo" in command and command[command.index("-fo") + 1] == "sqmcrt":
            output.write_text(
                "Run semi-empirical minimization\n &qmmm\n"
                " qm_theory='AM1', grms_tol=0.0005,\n"
                " scfconv=1.d-10, ndiis_attempts=700, qmcharge=0,\n /\n"
                "8 O1 0 0 0\n1 H1 .9572 0 0\n1 H2 -.239987 .927297 0\n\n"
            )
        elif Path(command[0]).name == "sqm":
            source = Path(command[command.index("-i") + 1])
            rows = source.read_text().split("/\n")[1].splitlines()
            actual = np.array(
                [list(map(float, r.split()[2:])) for r in rows if r.strip()]
            )
            np.testing.assert_array_equal(actual, POSITIONS)
            output.write_text(
                " ... geometry converged !\n"
                " Atom Element Mulliken Charge\n"
                "1 O -0.834\n2 H 0.417\n3 H 0.417\n"
                " Total Mulliken Charge = 0.000\n"
                " Final Structure\n"
                " QMMM: QM Region Cartesian Coordinates (*=link atom)\n"
                " QMMM: QM_NO. MM_NO. ATOM X Y Z\n"
                " QMMM: 1 1 O 0.0100 0.0000 0.0000\n"
                " QMMM: 2 2 H 0.9672 0.0000 0.0000\n"
                " QMMM: 3 3 H -0.2300 0.9273 0.0000\n\n"
                " --------- Calculation Completed ----------\n"
            )
        else:
            assert command[command.index("-fi") + 1] == "sqmout"
            assert command[command.index("-c") + 1] in {"bcc", "abcg2"}
            source = Path(command[command.index("-a") + 1])
            typed = MOL2Reader(str(source), charge=0, mult=1)
            np.testing.assert_array_equal(typed.positions, POSITIONS)
            fields = []
            ac = []
            section = ""
            for line in source.read_text().splitlines():
                if line.startswith("@<TRIPOS>"):
                    section = line[9:]
                elif section == "ATOM" and line.strip():
                    row = line.split()
                    xyz = [f"{float(token):.3f}" for token in row[2:5]]
                    row[2:5] = [f"{float(token):.4f}" for token in xyz]
                    row[8] = "-0.8340" if int(row[0]) == 1 else "0.4170"
                    ac.append(
                        f"ATOM {row[0]} {row[1]} MOL 1 {' '.join(xyz)} {row[8]} {row[5]}"
                    )
                    line = " ".join(row)
                fields.append(line)
            output.write_text("\n".join(fields) + "\n")
            (Path(kwargs["cwd"]) / "ANTECHAMBER_AM1BCC_PRE.AC").write_text(
                "\n".join(ac) + "\n"
            )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(ambertools.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("method", ["am1bcc", "abcg2"])
def test_infer_then_charge_once_and_save_exact_geometry(
    tmp_path, water_mol2, monkeypatch, method
):
    calls = fake_ambertools(monkeypatch, water_mol2)
    atoms = water()
    before = deepcopy(atoms.info)
    environment = dict(os.environ)
    result = charges.prepare_charges(
        atoms, {"source": "maple", "method": method}, tmp_path / "audit"
    )
    assert len(calls) == 4
    assert atoms.info == before
    assert dict(os.environ) == environment
    assert MOL2_ATOM_ID_ARRAY not in atoms.arrays
    np.testing.assert_array_equal(atoms.positions, POSITIONS)
    sqm = result.provenance["cartesian_sqm"]
    assert sqm["sqm_execution_count"] == 1
    assert sqm["second_sqm_generation_detected"] is False
    assert (
        sqm["charge_evaluation_geometry"] == "AM1-optimized-not-original-MLIP-geometry"
    )
    assert sqm["source_to_bcc_serialized_maximum_error_angstrom"] > 5e-5
    assert not np.array_equal(
        sqm["optimized_geometry"]["positions_angstrom"], POSITIONS
    )
    assert charges.MAX_ANTECHAMBER_COORDINATE_HALF_WIDTH_ANGSTROM == 5e-5
    saved = Path(result.provenance["prepared_charged_mol2"])
    prepared = MOL2Reader(str(saved), charge=0, mult=1, validate_charge=True)
    np.testing.assert_array_equal(prepared.positions, POSITIONS)
    np.testing.assert_allclose(
        prepared.get_initial_charges(), result.charges, atol=1e-15
    )
    assert result.prepared_atoms.info["mol2"]["path"] == str(saved)
    prep = result.provenance["preparation"]
    assert prep["original_coordinates_restored"] is True
    assert prep["warning"]
    assert prep["source_geometry_sha256"]
    assert (
        json.loads((tmp_path / "audit" / "preparation.json").read_text())["status"]
        == "complete"
    )


@pytest.mark.parametrize(
    "corrupt, match",
    [
        (lambda s: s.replace("2 H1", "4 H1"), "bond atom id|atom ID"),
        (lambda s: s.replace("2 H1", "3 H1"), "Duplicate.*atom id"),
        (
            lambda s: s.replace("0.957000", "0.960000"),
            "topology.*coordinate|coordinate.*topology",
        ),
        (lambda s: s.replace("H         1", "C.3       1", 1), "element"),
        (lambda s: s.replace("2    1    3 1", "2    2    3 1"), "hydrogen|Hydrogen"),
    ],
)
def test_inference_rejection_preserves_caller_and_failure_audit(
    tmp_path, water_mol2, monkeypatch, corrupt, match
):
    calls = fake_ambertools(monkeypatch, water_mol2, corrupt=corrupt)
    atoms = water()
    before = deepcopy(atoms.info)
    with pytest.raises(ValueError, match=match):
        charges.prepare_charges(atoms, {"source": "maple"}, tmp_path / "audit")
    assert len(calls) == 1
    assert atoms.info == before
    assert MOL2_ATOM_ID_ARRAY not in atoms.arrays
    np.testing.assert_array_equal(atoms.positions, POSITIONS)
    audit = json.loads((tmp_path / "audit" / "preparation.json").read_text())
    assert audit["status"] == "failed"
    assert audit["error"]
    assert (tmp_path / "audit" / "topology.stdout.log").is_file()


@pytest.mark.parametrize("options", [{"geometry": "provider"}, {"source": "mol2"}])
def test_cartesian_rejects_incompatible_source_or_geometry_without_tools(
    tmp_path, monkeypatch, options
):
    monkeypatch.setattr(
        ambertools.subprocess, "run", lambda *a, **k: pytest.fail("tool reached")
    )
    with pytest.raises(ValueError):
        charges.prepare_charges(
            water(), {"source": "maple", **options}, tmp_path / "audit"
        )
    assert not (tmp_path / "audit").exists()


def test_correction_adopts_only_validated_preparation_before_freezing(
    tmp_path, water_mol2, monkeypatch
):
    calls = fake_ambertools(monkeypatch, water_mol2)
    atoms = water()
    out = tmp_path / "job.out"
    runtime = correction.ImplicitSolvationCorrection(
        atoms, {"source": "maple"}, {"method": "gb", "experimental": True}, output=out
    )
    assert len(calls) == 4
    assert MOL2_ATOM_ID_ARRAY in atoms.arrays
    runtime.evaluate(atoms)
    for _ in range(2):
        force = runtime.evaluate(atoms, need_forces=True).forces_hartree_per_angstrom
        hessian = runtime.get_hessian(atoms)
        assert force is not None and np.isfinite(force).all()
        assert hessian.shape == (9, 9) and np.isfinite(hessian).all()
    assert len(calls) == 4, "force/Hessian evaluation must not rerun preparation"
    manifest = json.loads((runtime.audit_dir / "manifest.json").read_text())
    assert manifest["charge"]["preparation"]["warning"]
    assert "inferred" in out.read_text().lower()
    assert manifest["charge"]["prepared_charged_mol2"] in out.read_text()
    atoms.info["mol2"]["bonds"][0][2] = "2"
    with pytest.raises(ValueError, match="frozen MOL2 topology"):
        runtime.evaluate(atoms)


def test_bcc_coordinate_corruption_is_rejected_not_restored_into_false_success(
    tmp_path, water_mol2, monkeypatch
):
    calls = fake_ambertools(monkeypatch, water_mol2)
    run = ambertools.subprocess.run

    def rounded(command, **kwargs):
        completed = run(command, **kwargs)
        if "-c" in command:
            path = Path(command[command.index("-o") + 1])
            path.write_text(path.read_text().replace("0.9570", "0.9580"))
        return completed

    monkeypatch.setattr(ambertools.subprocess, "run", rounded)
    atoms = water()
    with pytest.raises(ValueError, match="changed coordinates under geometry=keep"):
        charges.prepare_charges(atoms, {"source": "maple"}, tmp_path / "audit")
    assert len(calls) == 4
    assert "mol2" not in atoms.info
    assert MOL2_ATOM_ID_ARRAY not in atoms.arrays
    np.testing.assert_array_equal(atoms.positions, POSITIONS)
    assert not (tmp_path / "audit" / "prepared-am1bcc.mol2").exists()
    audit = json.loads((tmp_path / "audit" / "preparation.json").read_text())
    assert audit["status"] == "failed"
    assert audit["charge_geometry_validated"] is False


@pytest.mark.parametrize("stage", ["topology", "sqm-template", "sqm", "charges"])
@pytest.mark.parametrize("failure", ["nonzero", "timeout", "startup"])
def test_failed_tool_attempts_preserve_raw_audit_and_caller(
    tmp_path, water_mol2, monkeypatch, stage, failure
):
    fake_ambertools(monkeypatch, water_mol2)
    run = ambertools.subprocess.run

    def failing(command, **kwargs):
        actual_stage = (
            "charges"
            if "-c" in command
            else (
                "sqm"
                if Path(command[0]).name == "sqm"
                else (
                    "sqm-template"
                    if command[command.index("-fo") + 1] == "sqmcrt"
                    else "topology"
                )
            )
        )
        reached = actual_stage == stage
        if not reached:
            return run(command, **kwargs)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                command, 1, output=b"partial stdout", stderr=b"partial stderr"
            )
        if failure == "startup":
            raise OSError("cannot launch")
        return subprocess.CompletedProcess(command, 1, "failed stdout", "failed stderr")

    monkeypatch.setattr(ambertools.subprocess, "run", failing)
    atoms = water()
    before = deepcopy(atoms.info)
    with pytest.raises((RuntimeError, OSError, subprocess.TimeoutExpired)):
        charges.prepare_charges(atoms, {"source": "maple"}, tmp_path / "audit")
    assert atoms.info == before
    assert set(atoms.arrays) == {"numbers", "positions"}
    stem = "charges-am1bcc" if stage == "charges" else stage
    command_record = json.loads(
        (tmp_path / "audit" / f"{stem}.command.json").read_text()
    )
    assert command_record["status"] == "failed"
    assert command_record["error"]
    assert (tmp_path / "audit" / f"{stem}.stdout.log").is_file()
    assert (tmp_path / "audit" / f"{stem}.stderr.log").is_file()
    if failure == "timeout":
        assert (
            tmp_path / "audit" / f"{stem}.stderr.log"
        ).read_text() == "partial stderr"
    assert (
        json.loads((tmp_path / "audit" / "preparation.json").read_text())["status"]
        == "failed"
    )


def test_disconnected_inference_is_rejected_before_charge(
    tmp_path, water_mol2, monkeypatch
):
    # Fixture spacing is deliberately retained by dedent.
    text = water_mol2.read_text()
    rows = text.splitlines()
    rows[rows.index("WATER") + 1] = "3 1 1 0 0"
    rows = [row for row in rows if row.split() != ["2", "1", "3", "1"]]
    calls = fake_ambertools(
        monkeypatch, water_mol2, corrupt=lambda _: "\n".join(rows) + "\n"
    )
    with pytest.raises(ValueError, match="one connected molecule"):
        charges.prepare_charges(water(), {"source": "maple"}, tmp_path / "audit")
    assert len(calls) == 1


def test_precharged_mol2_never_discovers_or_runs_ambertools(
    tmp_path, water_mol2, monkeypatch
):
    monkeypatch.setattr(
        charges, "resolve_antechamber", lambda *a, **k: pytest.fail("discovery reached")
    )
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1, validate_charge=True)
    result = charges.prepare_charges(
        atoms, {"source": "mol2", "label": "frozen"}, tmp_path / "audit"
    )
    np.testing.assert_array_equal(result.charges, atoms.get_initial_charges())
    assert result.prepared_atoms is None
    assert result.method == "frozen"
    assert not (tmp_path / "audit").exists()


def test_invalid_correction_configuration_rejects_before_preparation_side_effects(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        charges, "resolve_antechamber", lambda *a, **k: pytest.fail("discovery reached")
    )
    atoms = water()
    before = deepcopy(atoms.info)
    with pytest.raises(ValueError, match="experimental=true"):
        correction.ImplicitSolvationCorrection(
            atoms, {"source": "maple"}, {"method": "gb"}, output=tmp_path / "bad.out"
        )
    assert atoms.info == before
    assert set(atoms.arrays) == {"numbers", "positions"}
    assert not list(tmp_path.iterdir())


def test_source_file_drift_is_rejected_before_inference(
    tmp_path, water_mol2, monkeypatch
):
    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nwater\nO 0 0 0\nH .9572 0 0\nH -.239987 .927297 0\n")
    atoms = read_job(tmp_path, f"0 1\nXYZ {xyz}")
    xyz.write_text(xyz.read_text() + "\n")
    calls = fake_ambertools(monkeypatch, water_mol2)
    with pytest.raises(ValueError, match="source file changed"):
        charges.prepare_charges(
            atoms, atoms.info["_maple_charge_options"], tmp_path / "audit"
        )
    assert not calls
    assert "mol2" not in atoms.info


def test_new_prepared_mol2_keeps_validated_charge_provider_atom_types(
    tmp_path, water_mol2, monkeypatch
):
    fake_ambertools(monkeypatch, water_mol2)
    run = ambertools.subprocess.run

    def retyped(command, **kwargs):
        completed = run(command, **kwargs)
        if "-c" in command:
            path = Path(command[command.index("-o") + 1])
            path.write_text(path.read_text().replace(" O.3 1 MOL", " oh 1 MOL"))
        return completed

    monkeypatch.setattr(ambertools.subprocess, "run", retyped)
    result = charges.prepare_charges(water(), {"source": "maple"}, tmp_path / "audit")
    assert result.prepared_atoms.info["mol2"]["atom_types"][0] == "oh"


@pytest.mark.parametrize("pbc", ['pbc="T F F"', "pbc='T F F'", "pbc=T F F"])
def test_pbc_only_xyz_comment_is_not_silently_ignored(tmp_path, pbc):
    xyz = tmp_path / "periodic.xyz"
    xyz.write_text(f"1\n{pbc}\nO 0 0 0\n")
    with pytest.raises(ValueError, match="non-periodic"):
        read_job(tmp_path, f"0 1\nXYZ {xyz}")


def test_empty_mol2_metadata_never_falls_back_to_cartesian_inference(
    tmp_path, monkeypatch
):
    atoms = water()
    atoms.info["mol2"] = {}
    monkeypatch.setattr(ambertools.shutil, "which", lambda _: "/opt/amber/bin/antechamber")
    monkeypatch.setattr(
        ambertools.subprocess, "run", lambda *a, **k: pytest.fail("inference fallback")
    )
    with pytest.raises(ValueError, match="requires a MOL2"):
        charges.prepare_charges(atoms, {"source": "maple"}, tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


@pytest.mark.parametrize(
    "stage, corrupt, match",
    [
        (
            "sqm-template",
            lambda s: s.replace("grms_tol=0.0005", "grms_tol=0.005"),
            "default AM1 protocol",
        ),
        (
            "sqm",
            lambda s: s.replace("geometry converged !", "maximum steps exceeded"),
            "converged AM1",
        ),
        (
            "sqm",
            lambda s: s.replace("Calculation Completed", "Calculation Failed"),
            "converged AM1",
        ),
        ("sqm", lambda s: s.replace("2 H 0.417", "1 H 0.417"), "Mulliken atom"),
        ("sqm", lambda s: s.replace("2 H 0.417", "2 C 0.417"), "Mulliken atom"),
        ("sqm", lambda s: s.replace("2 H 0.417", "2 H nan"), "Mulliken atom"),
        ("sqm", lambda s: s.replace("QMMM: 2 2 H", "QMMM: 3 3 H"), "final geometry"),
        ("sqm", lambda s: s.replace("QMMM: 2 2 H", "QMMM: 2 2 C"), "final geometry"),
        ("charges", lambda s: s.replace("-0.8340", "-0.834"), "charge token.*coarser"),
        (
            "charges",
            lambda s: s.replace("0.9570", "0.957"),
            "coordinate token.*coarser",
        ),
        ("charges", lambda s: s.replace("-0.8340", "-0.8000"), "charge residual"),
        ("charges", lambda s: s.replace(" H1 ", " H9 "), "atom name"),
        ("charges", lambda s: s.replace("1 1 2 1", "1 1 2 2"), "labeled bond"),
    ],
)
def test_native_chain_guard_failures_never_mutate_caller(
    tmp_path, water_mol2, monkeypatch, stage, corrupt, match
):
    fake_ambertools(monkeypatch, water_mol2)
    run = ambertools.subprocess.run

    def corrupted(command, **kwargs):
        completed = run(command, **kwargs)
        actual = (
            "charges"
            if "-c" in command
            else (
                "sqm"
                if Path(command[0]).name == "sqm"
                else (
                    "sqm-template"
                    if command[command.index("-fo") + 1] == "sqmcrt"
                    else "topology"
                )
            )
        )
        if actual == stage:
            path = Path(command[command.index("-o") + 1])
            path.write_text(corrupt(path.read_text()))
        return completed

    monkeypatch.setattr(ambertools.subprocess, "run", corrupted)
    atoms = water()
    before = deepcopy(atoms.info)
    with pytest.raises(ValueError, match=match):
        charges.prepare_charges(atoms, {"source": "maple"}, tmp_path / "audit")
    assert atoms.info == before
    assert set(atoms.arrays) == {"numbers", "positions"}
    assert atoms.positions.tobytes() == POSITIONS.tobytes()
    assert not (tmp_path / "audit" / "prepared-am1bcc.mol2").exists()
    assert (
        json.loads((tmp_path / "audit" / "preparation.json").read_text())["status"]
        == "failed"
    )


@pytest.mark.parametrize("change", ["new-sqm", "input", "completed-output"])
def test_bcc_cannot_regenerate_or_modify_completed_sqm(
    tmp_path, water_mol2, monkeypatch, change
):
    fake_ambertools(monkeypatch, water_mol2)
    run = ambertools.subprocess.run

    def changed(command, **kwargs):
        completed = run(command, **kwargs)
        if "-c" in command:
            work = Path(kwargs["cwd"])
            path = (
                work
                / {
                    "new-sqm": "sqm.in",
                    "input": "sqm-exact.in",
                    "completed-output": "sqm-exact.out",
                }[change]
            )
            path.write_text("unexpected second computation or modified artifact")
        return completed

    monkeypatch.setattr(ambertools.subprocess, "run", changed)
    atoms = water()
    with pytest.raises(ValueError, match="regenerated SQM|modified.*QM"):
        charges.prepare_charges(atoms, {"source": "maple"}, tmp_path / "audit")
    assert "mol2" not in atoms.info
    assert not (tmp_path / "audit" / "prepared-am1bcc.mol2").exists()


@pytest.mark.parametrize(
    "change", ["order", "ac-name", "ac-position", "ac-token", "missing-ac", "qm-input"]
)
def test_native_identity_and_geometry_boundary_guards(
    tmp_path, water_mol2, monkeypatch, change
):
    fake_ambertools(monkeypatch, water_mol2)
    run = ambertools.subprocess.run

    def corrupted(command, **kwargs):
        completed = run(command, **kwargs)
        if change == "qm-input" and Path(command[0]).name == "sqm":
            path = Path(command[command.index("-i") + 1])
            path.write_text(path.read_text() + "changed by QM tool\n")
        if "-c" in command:
            path = Path(kwargs["cwd"]) / "ANTECHAMBER_AM1BCC_PRE.AC"
            if change == "order":
                output = Path(command[command.index("-o") + 1])
                rows = output.read_text().splitlines()
                start = rows.index("@<TRIPOS>ATOM") + 1
                rows[start + 1], rows[start + 2] = rows[start + 2], rows[start + 1]
                output.write_text("\n".join(rows) + "\n")
            elif change == "ac-name":
                path.write_text(path.read_text().replace(" H1 ", " H9 "))
            elif change == "ac-position":
                path.write_text(path.read_text().replace("0.957", "0.958"))
            elif change == "ac-token":
                path.write_text(path.read_text().replace("0.957", "0.9570"))
            elif change == "missing-ac":
                path.unlink()
        return completed

    monkeypatch.setattr(ambertools.subprocess, "run", corrupted)
    atoms = water()
    before, environment = deepcopy(atoms.info), dict(os.environ)
    with pytest.raises(
        ValueError,
        match="atom order|IDs/names|AC geometry|serialization tokens|AC geometry audit|modified its exact",
    ):
        charges.prepare_charges(atoms, {"source": "maple"}, tmp_path / "audit")
    assert atoms.info == before and set(atoms.arrays) == {"numbers", "positions"}
    assert atoms.positions.tobytes() == POSITIONS.tobytes()
    assert dict(os.environ) == environment
    assert not (tmp_path / "audit" / "prepared-am1bcc.mol2").exists()


def test_exact_sqm_refuses_native_line_truncation(tmp_path, water_mol2, monkeypatch):
    calls = fake_ambertools(monkeypatch, water_mol2)
    typed = MOL2Reader(str(water_mol2), charge=0, mult=1)
    typed.positions[:] = -1.2345678901234567e-100
    typed.info["mol2"]["atom_names"][0] = "O1234567"
    tool = charges.resolve_antechamber()
    with pytest.raises(ValueError, match="native 80-character/8-character-name"):
        charges.run_cartesian_sqm_bcc(
            typed, "am1bcc", water_mol2, tmp_path, tool, {}, 1.0
        )
    assert len(calls) == 1, "truncated coordinates must never reach native SQM"


def test_failed_correction_preparation_does_not_install_caller_identity(
    tmp_path, water_mol2, monkeypatch
):
    fake_ambertools(
        monkeypatch,
        water_mol2,
        corrupt=lambda text: text.replace("0.957000", "0.960000"),
    )
    atoms = water()
    before = deepcopy(atoms.info)
    with pytest.raises(ValueError, match="topology coordinates"):
        correction.ImplicitSolvationCorrection(
            atoms,
            {"source": "maple"},
            {"method": "gb", "experimental": True},
            output=tmp_path / "rejected.out",
        )
    assert atoms.info == before
    assert set(atoms.arrays) == {"numbers", "positions"}
    np.testing.assert_array_equal(atoms.positions, POSITIONS)
    assert (
        json.loads(
            (tmp_path / "rejected.out.implicit" / "preparation.json").read_text()
        )["status"]
        == "failed"
    )
