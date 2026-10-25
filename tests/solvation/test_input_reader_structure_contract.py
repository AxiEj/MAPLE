"""Public input parsing controls; no calculator or provider is constructed."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from maple.function.read.input_reader import InputReader
from maple.function.utility import Molecules


@pytest.fixture
def structure_files(tmp_path):
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    write(tmp_path / "water.xyz", atoms)
    write(tmp_path / "frames.xyz", [atoms, atoms.copy()])
    write(tmp_path / "water.pdb", atoms)
    write(tmp_path / "frames.pdb", [atoms, atoms.copy()])
    return tmp_path


def read_input(tmp_path, coordinates, settings="#model=ani2x\n#sp"):
    inp = tmp_path / "job.inp"
    inp.write_text(f"{settings}\n\n{coordinates}\n")
    reader = InputReader()
    return reader, reader(str(inp), str(tmp_path / "job.out"))


def frames(result):
    return result.multiatoms if isinstance(result, Molecules) else [result]


@pytest.mark.parametrize(
    "reference,count",
    [
        ("XYZ water.xyz", 1),
        ("XYZTRAJ frames.xyz", 2),
        ("PDB water.pdb", 1),
        ("PDB frames.pdb", 2),
        ("XYZ water.xyz\nXYZ water.xyz", 2),
        ("PDB water.pdb\nPDB water.pdb", 2),
    ],
)
@pytest.mark.parametrize("state", ["0 1", "-1 2"])
def test_block_state_precedes_external_dispatch(
    structure_files, reference, count, state
):
    _, result = read_input(structure_files, f"{state}\n{reference}")
    assert len(frames(result)) == count
    charge, mult = map(int, state.split())
    for atoms in frames(result):
        assert (atoms.info["charge"], atoms.info["mult"], atoms.info["spin"]) == (
            charge,
            mult,
            (mult - 1) / 2,
        )
        assert atoms.get_chemical_symbols() == ["O", "H", "H"]


@pytest.mark.parametrize(
    "reference",
    [
        "XYZ -1 2 water.xyz",
        "XYZ water.xyz -1 2",
        "XYZTRAJ -1 2 frames.xyz",
        "XYZTRAJ frames.xyz -1 2",
        "PDB -1 2 water.pdb",
        "PDB -1 2 frames.pdb",
    ],
)
def test_existing_reference_state_api(structure_files, reference):
    _, result = read_input(structure_files, reference)
    for atoms in frames(result):
        assert (atoms.info["charge"], atoms.info["mult"]) == (-1, 2)


def test_separate_blocks_do_not_share_state(structure_files):
    _, result = read_input(
        structure_files, "0 1\nXYZ water.xyz\n&\n-1 2\nPDB water.pdb\n\nH 0 0 0"
    )
    first, second, third = frames(result)
    assert (first.info["charge"], second.info["charge"]) == (0, -1)
    assert "charge" not in third.info


@pytest.mark.parametrize("reference", ["XYZ -1 2 water.xyz", "PDB -1 2 water.pdb"])
def test_conflicting_state_is_not_silently_overwritten(structure_files, reference):
    with pytest.raises(ValueError, match="[Cc]onflicting.*charge/multiplicity"):
        read_input(structure_files, f"0 1\n{reference}")


@pytest.mark.parametrize(
    "reference", ["XYZ water.xyz", "PDB water.pdb", "MOL2 missing.mol2"]
)
def test_file_inline_ambiguity_rejected(structure_files, reference):
    with pytest.raises(ValueError, match="[Mm]ixed.*[Ss]ame group"):
        read_input(structure_files, f"{reference}\nH 0 0 0")


def test_mixed_external_formats_require_separate_groups(structure_files):
    with pytest.raises(ValueError, match="[Mm]ixed.*[Ss]ame group"):
        read_input(structure_files, "XYZ water.xyz\nPDB water.pdb")


def test_xyz_and_xyztraj_share_existing_group(structure_files):
    _, result = read_input(structure_files, "0 1\nXYZ water.xyz\nXYZTRAJ frames.xyz")
    assert len(frames(result)) == 3


@pytest.mark.parametrize(
    "body", ["0 0\nXYZ water.xyz", "0 1", "0 1\n0 1\nXYZ water.xyz"]
)
def test_invalid_or_repeated_block_state(structure_files, body):
    with pytest.raises(ValueError):
        read_input(structure_files, body)


def test_parmfit_pdb_remains_deferred(structure_files, monkeypatch):
    from maple.function.read.filereader import PDBReader

    def forbidden(*args, **kwargs):
        raise AssertionError("parmfit must not parse PDB coordinates in InputReader")

    monkeypatch.setattr(PDBReader, "_read_coordinate_frames", forbidden)
    reader, result = read_input(
        structure_files, "0 1\nPDB water.pdb", "#model=ani2x\n#parmfit"
    )
    assert isinstance(result, Atoms)
    assert reader.command_control.params["pdb"] == str(structure_files / "water.pdb")
    assert result.get_chemical_symbols() == ["H"]
    assert result.info["mult"] == 1


@pytest.mark.parametrize(
    "body", ["PDB water.pdb\nPDB water.pdb", "PDB water.pdb\n&\nPDB water.pdb"]
)
def test_parmfit_rejects_multiple_pdb_inputs(structure_files, body):
    with pytest.raises(ValueError, match="exactly one PDB"):
        read_input(structure_files, body, "#model=ani2x\n#parmfit")


@pytest.mark.parametrize(
    "reference", ["XYZ water.xyz", "XYZTRAJ frames.xyz", "PDB frames.pdb", "H 0 0 0"]
)
def test_gas_pbc_is_preserved(structure_files, reference):
    _, result = read_input(
        structure_files, f"0 1\n{reference}", "#model=ani2x\n#sp\n#pbc(10,11,12)"
    )
    for atoms in frames(result):
        assert atoms.pbc.all()
        np.testing.assert_array_equal(atoms.cell.lengths(), [10, 11, 12])


def test_metadata_is_copied_on_every_frame(structure_files):
    reader, result = read_input(
        structure_files,
        "0 1\nXYZTRAJ frames.xyz",
        "#model=ani2x\n#sp\n#charge(source=maple,method=qeq-gto)",
    )
    first, second = frames(result)
    expected = deepcopy(reader.command_control.params["charge"])
    assert (
        first.info["_maple_charge_options"]
        == second.info["_maple_charge_options"]
        == expected
    )
    assert (
        first.info["_maple_solvation_options"]
        == second.info["_maple_solvation_options"]
        == {}
    )
    first.info["_maple_charge_options"]["source"] = "changed"
    first.info["_maple_solvation_options"]["method"] = "changed"
    assert (
        second.info["_maple_charge_options"]
        == reader.command_control.params["charge"]
        == expected
    )
    assert second.info["_maple_solvation_options"] == {}


@pytest.mark.parametrize("state", ["", "-1 1\n", "0 2\n"])
def test_legacy_mol2_state_gate(water_mol2, tmp_path, state):
    with pytest.raises(ValueError, match="explicit '0 1'"):
        read_input(
            tmp_path,
            f"{state}MOL2 {water_mol2.name}",
            "#model=ani2x\n#charge(source=mol2)\n#solv(implicit=water,method=gb,experimental=true)",
        )


@pytest.mark.parametrize(
    "body", ["0 1\nH 0 0 0", "0 1\nPDB missing.pdb", "0 1\nXYZTRAJ missing.xyz"]
)
def test_legacy_rejects_untyped_before_reading(structure_files, body):
    with pytest.raises(ValueError, match="requires a MOL2"):
        read_input(
            structure_files,
            body,
            "#model=ani2x\n#charge(source=maple,method=qeq-gto)\n#solv(implicit=water,method=gb,experimental=true)",
        )


def test_modern_anysolv_accepts_inline_and_copies_metadata(tmp_path):
    _, result = read_input(
        tmp_path,
        "0 1\nH 0 0 0",
        "#model=ani2x\n#device=cpu\n#sp(verbose=0)\n#solv(implicit=water,method=anysolv,experimental=true,source_root=/pinned/anysolv)",
    )
    assert isinstance(result, Atoms)
    assert result.info["_maple_solvation_options"]["method"] == "anysolv"


def test_native_consolv_accepts_xyz(structure_files):
    _, result = read_input(
        structure_files,
        "0 1\nXYZ water.xyz",
        "#model=consolv(native_solvent=ethanol)\n#device=cpu\n#sp",
    )
    assert isinstance(result, Atoms)
    assert result.get_chemical_symbols() == ["O", "H", "H"]


@pytest.mark.parametrize(
    "kind,name",
    [
        ("XYZ", "water.xyz"),
        ("XYZTRAJ", "frames.xyz"),
        ("PDB", "frames.pdb"),
        ("MOL2", "water.mol2"),
    ],
)
def test_classifier_dispatch_agree_on_case_and_whitespace(
    structure_files, water_mol2, kind, name
):
    _, result = read_input(structure_files, f"0 1\n{kind.lower()}\t{name}")
    for atoms in frames(result):
        assert atoms.get_chemical_symbols() == ["O", "H", "H"]
        assert atoms.info["_maple_charge_options"] == {}
        assert atoms.info["_maple_solvation_options"] == {}


def test_mol2_public_flow_is_lossless_for_multiple_files(water_mol2, tmp_path):
    from maple.function.read.filereader import MOL2Reader

    original_bytes = water_mol2.read_bytes()
    expected = MOL2Reader(str(water_mol2), charge=0, mult=1)
    _, result = read_input(
        tmp_path,
        f"0 1\nMOL2 {water_mol2.name}\nMOL2 {water_mol2.name}",
        "#model=ani2x\n#charge(source=mol2)\n#solv(implicit=water,method=gb,experimental=true)",
    )
    first, second = frames(result)
    for atoms in (first, second):
        np.testing.assert_array_equal(atoms.positions, expected.positions)
        np.testing.assert_array_equal(
            atoms.get_initial_charges(), expected.get_initial_charges()
        )
        assert atoms.info["mol2"] == expected.info["mol2"]
        assert atoms.info["_maple_solvation_options"]["method"] == "gb"
    first.info["_maple_solvation_options"]["model"] = "changed"
    assert second.info["_maple_solvation_options"]["model"] == "obc2"
    assert water_mol2.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "method,model,charge",
    [
        ("gb", "ani2x", "#charge(source=mol2)\n"),
        ("pb", "ani2x", "#charge(source=mol2)\n"),
        ("smd", "macepol-m", ""),
    ],
)
@pytest.mark.parametrize(
    "body,message",
    [
        ("0 1\nXYZ missing.xyz", "requires a MOL2"),
        ("MOL2 missing.mol2", "explicit '0 1'"),
    ],
)
def test_all_legacy_methods_gate_before_file_io(
    tmp_path, method, model, charge, body, message
):
    with pytest.raises(ValueError, match=message):
        read_input(
            tmp_path,
            body,
            f"#model={model}\n#sp\n{charge}#solv(implicit=water,method={method},experimental=true)",
        )


def test_solvent_pbc_remains_rejected(water_mol2, tmp_path):
    with pytest.raises(ValueError, match="non-periodic"):
        read_input(
            tmp_path,
            f"0 1\nMOL2 {water_mol2.name}",
            "#model=ani2x\n#pbc(10,11,12)\n#charge(source=mol2)\n#solv(implicit=water,method=gb,experimental=true)",
        )


def test_mol2_gas_pbc_remains_supported(water_mol2, tmp_path):
    _, atoms = read_input(
        tmp_path, f"0 1\nMOL2 {water_mol2.name}", "#model=ani2x\n#pbc(10,11,12)"
    )
    assert isinstance(atoms, Atoms)
    assert atoms.pbc.all()
    np.testing.assert_array_equal(atoms.cell.lengths(), [10, 11, 12])


@pytest.mark.parametrize(
    "reference", ["XYZ water.xyz", "PDB frames.pdb", "XYZTRAJ frames.xyz"]
)
def test_anysolv_multiframe_metadata_without_mol2_only_gate(structure_files, reference):
    reader, result = read_input(
        structure_files,
        f"0 1\n{reference}",
        "#model=ani2x\n#device=cpu\n#sp(verbose=0)\n#solv(implicit=toluene,method=anysolv,experimental=true,source_root=/pinned/anysolv)",
    )
    for atoms in frames(result):
        assert (
            atoms.info["_maple_solvation_options"]
            == reader.command_control.params["solv"]
        )
        assert atoms.info["mult"] == 1


def test_mol2_and_inline_groups_and_postprocessing_remain_distinct(
    water_mol2, tmp_path
):
    _, result = read_input(
        tmp_path, f"0 1\nMOL2 {water_mol2.name}\n&\n0 1\nH 0 0 0\n\nC 1"
    )
    assert len(frames(result)) == 2
    for atoms in frames(result):
        np.testing.assert_array_equal(atoms.constraints[0].index, [0])
        assert atoms.info["mult"] == 1
        assert atoms.info["_maple_charge_options"] == {}


def test_inline_velocity_warning_preserved(tmp_path):
    _, atoms = read_input(tmp_path, "0 1\nH 0 0 0 1 2 3")
    assert isinstance(atoms, Atoms)
    assert "momenta" not in atoms.arrays
    assert "These velocities are ignored" in (tmp_path / "job.out").read_text()


def test_scan_postprocessing_not_misclassified_as_atom(tmp_path):
    reader, atoms = read_input(
        tmp_path, "0 1\nH 0 0 0\nH 0 0 .74\n\nS 1 2 .1 5", "#model=ani2x\n#scan"
    )
    assert isinstance(atoms, Atoms)
    assert len(atoms) == 2
    assert reader.scan_constraints == [[1, 2, 0.1, 5]]


@pytest.mark.parametrize("reference_state", ["-1 1", "0 2", "-1 2"])
def test_parmfit_rejects_conflicting_reference_state_before_loading(
    structure_files, monkeypatch, reference_state
):
    from maple.function.read.filereader import PDBReader

    def forbidden(*args, **kwargs):
        raise AssertionError("deferred parmfit must not read PDB coordinates")

    monkeypatch.setattr(PDBReader, "_read_coordinate_frames", forbidden)
    with pytest.raises(ValueError, match="Conflicting.*charge/multiplicity"):
        read_input(
            structure_files,
            f"0 1\nPDB {reference_state} water.pdb",
            "#model=ani2x\n#parmfit",
        )


@pytest.mark.parametrize(
    "block_state,reference_state", [("0 1\n", "0 1"), ("-1 2\n", "-1 2"), ("", "-1 2")]
)
def test_parmfit_matching_or_reference_only_state_retains_deferral(
    structure_files, monkeypatch, block_state, reference_state
):
    from maple.function.read.filereader import PDBReader

    # Geometry belongs to parmfit, not InputReader. Only existence is needed here.
    (structure_files / "water.pdb").write_text("not a coordinate record\n")
    resolutions = []
    original_resolve = PDBReader.resolve_path

    def resolve_once(*args, **kwargs):
        resolutions.append(args[0])
        return original_resolve(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("deferred parmfit must not read PDB coordinates")

    monkeypatch.setattr(PDBReader, "resolve_path", resolve_once)
    monkeypatch.setattr(PDBReader, "_read_coordinate_frames", forbidden)
    reader, atoms = read_input(
        structure_files,
        f"{block_state}PDB {reference_state} water.pdb",
        "#model=ani2x\n#parmfit",
    )
    assert isinstance(atoms, Atoms)
    charge, mult = map(int, reference_state.split())
    assert (atoms.info["charge"], atoms.info["mult"], atoms.info["spin"]) == (
        charge,
        mult,
        (mult - 1) / 2,
    )
    assert reader.command_control.params["pdb"] == str(structure_files / "water.pdb")
    assert len(resolutions) == 1
    assert atoms.get_chemical_symbols() == ["H"]


@pytest.mark.parametrize("kind", ["INLINE", "XYZ", "XYZTRAJ", "PDB", "MOL2"])
@pytest.mark.parametrize("token", ["1e999", "-1e999", "nan", "inf", "-inf"])
def test_nonfinite_coordinates_rejected_per_format(
    structure_files, water_mol2, kind, token
):
    if kind == "INLINE":
        body = f"H {token} 0 0"
    elif kind in {"XYZ", "XYZTRAJ"}:
        (structure_files / "bad.xyz").write_text(f"1\nnonfinite\nH {token} 0 0\n")
        body = f"{kind} bad.xyz"
    elif kind == "PDB":
        line = next(
            line
            for line in (structure_files / "water.pdb").read_text().splitlines()
            if line[:6].strip() in {"ATOM", "HETATM"}
        )
        (structure_files / "bad.pdb").write_text(
            f"{line[:30]}{token:>8}{line[38:]}\nEND\n"
        )
        body = "PDB bad.pdb"
    else:
        water_mol2.write_text(water_mol2.read_text().replace("0.957200", token, 1))
        body = f"MOL2 {water_mol2.name}"
    # Literal nan/inf is already outside the inline/XYZ numeric grammar. Overflow
    # and PDB float tokens reach numeric preparation and must fail as nonfinite.
    message = "finite" if kind in {"PDB", "MOL2"} or "999" in token else None
    with pytest.raises(ValueError, match=message):
        read_input(structure_files, f"0 1\n{body}")


@pytest.mark.parametrize("kind", ["XYZ", "XYZTRAJ", "PDB", "MOL2"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_common_boundary_rejects_nonfinite_reader_results(
    tmp_path, monkeypatch, kind, value
):
    from maple.function.read import input_reader

    valid = Atoms("H", positions=[[0, 0, 0]])
    invalid = Atoms("H", positions=[[value, 0, 0]])
    result = Molecules([valid, invalid]) if kind in {"XYZTRAJ", "PDB"} else invalid
    monkeypatch.setattr(
        input_reader,
        {
            "XYZ": "XYZReader",
            "XYZTRAJ": "XYZTrajReader",
            "PDB": "PDBReader",
            "MOL2": "MOL2Reader",
        }[kind],
        lambda *args, **kwargs: result,
    )
    with pytest.raises(ValueError, match="coordinates must be finite"):
        read_input(tmp_path, f"0 1\n{kind} supplied-by-reader")


def test_trajectory_overflow_in_later_frame_is_rejected(tmp_path):
    (tmp_path / "bad.xyz").write_text("1\nvalid\nH 0 0 0\n1\ninvalid\nH 1e999 0 0\n")
    with pytest.raises(ValueError, match="coordinates must be finite"):
        read_input(tmp_path, "0 1\nXYZTRAJ bad.xyz")
