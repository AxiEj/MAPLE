from __future__ import annotations

from maple.function.read.input_reader import InputReader


def test_input_reader_accepts_charge_mult_then_mol2(water_mol2, tmp_path):
    inp = tmp_path / "job.inp"
    out = tmp_path / "job.out"
    inp.write_text(
        "\n".join(
            [
                "#model=ani2x",
                "#sp(verbose=1)",
                "#charge(source=mol2,label=resp2)",
                "#solv(implicit=water,method=gb,model=obc2,nonpolar=ace,experimental=true)",
                "",
                "0 1",
                f"MOL2 {water_mol2}",
                "",
            ]
        )
    )
    atoms = InputReader()(str(inp), str(out))
    assert atoms.info["charge"] == 0
    assert atoms.info["mult"] == 1
    assert atoms.info["_maple_charge_options"]["source"] == "mol2"
    assert atoms.info["_maple_solvation_options"]["model"] == "obc2"


def test_implicit_input_rejects_xyz_even_when_charge_method_can_use_geometry(tmp_path):
    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nwater\nO 0 0 0\nH .96 0 0\nH -.24 .93 0\n")
    inp = tmp_path / "job.inp"
    inp.write_text(
        "\n".join(
            [
                "#model=ani2x",
                "#charge(source=maple,method=qeq-gto)",
                "#solv(implicit=water,method=gb,experimental=true)",
                "",
                "0 1",
                f"XYZ {xyz}",
            ]
        )
    )
    try:
        InputReader()(str(inp), str(tmp_path / "job.out"))
    except ValueError as exc:
        assert "requires a MOL2" in str(exc)
    else:
        raise AssertionError("Implicit XYZ input should fail closed")


def test_implicit_mol2_requires_explicit_neutral_closed_shell_line(water_mol2, tmp_path):
    inp = tmp_path / "job.inp"
    inp.write_text(
        "\n".join(
            [
                "#model=ani2x",
                "#charge(source=maple,method=qeq-gto)",
                "#solv(implicit=water,method=gb,experimental=true)",
                "",
                f"MOL2 {water_mol2}",
            ]
        )
    )
    try:
        InputReader()(str(inp), str(tmp_path / "job.out"))
    except ValueError as exc:
        assert "explicit '0 1'" in str(exc)
    else:
        raise AssertionError("Implicit MOL2 without charge/multiplicity should fail closed")
