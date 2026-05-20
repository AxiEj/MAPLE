from maple.function.read.filereader.xyz_reader import XYZReader


def test_xyz_reader_keeps_multiplicity_as_source_of_truth_without_spin_alias(tmp_path):
    xyz = tmp_path / "mol.xyz"
    xyz.write_text(
        "\n".join(
            [
                "2",
                "water fragment",
                "O 0.0 0.0 0.0",
                "H 0.9 0.0 0.0",
            ]
        ),
        encoding="utf-8",
    )

    atoms = XYZReader(f"XYZ -1 3 {xyz}")

    assert atoms.info["charge"] == -1
    assert atoms.info["mult"] == 3
    assert "spin" not in atoms.info
