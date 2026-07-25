from maple.function.engine import engine


def test_explicit_first_shell_is_built_before_calculator_initialization(tmp_path):
    job = engine()

    job._input_reader(
        "examples/solvation/route3/gbsa_first_shell.inp",
        str(tmp_path / "route3.out"),
    )

    assert len(job.atoms) == 18  # six solute atoms + four three-atom waters
    assert not job.atoms.get_pbc().any()
    assert set(job.atoms.arrays["maple_molecule_id"]) == {-1, 0, 1, 2, 3}
    assert job.commandcontrol.params["solv"]["implicit"] == "water"
