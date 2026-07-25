from __future__ import annotations

import numpy as np
import pytest

from .conftest import FIXTURE_DIR, PROJECT_ROOT, load_json

MODEL_FIX = FIXTURE_DIR / "models" / "real_model_probes_v1.json"
RADIUS_PATH = PROJECT_ROOT / "docs" / "solvation" / "route-a" / "radius-profiles" / "bondi-mantina-v1.json"

def _assert_float64_array(value, expected):
    arr = np.asarray(value, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    assert arr.dtype == np.float64
    assert exp.dtype == np.float64
    assert arr.shape == exp.shape
    assert np.array_equal(arr, exp)


def _assert_model_record(role: str, model: dict):
    expected = {
        "off24": {
            "role": "periodic sampler",
            "provider": "MACEOff24Provider",
            "pbc": True,
            "source_ref": "v0.2",
            "source_commit": "91a78c5a9c300d1104700d9352c8bfe449227737",
            "source_url": "https://raw.githubusercontent.com/ACEsuit/mace-off/v0.2/mace_off24/MACE-OFF24_medium.model",
            "sha256": "e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69",
            "size_bytes": 18375569,
            "energy_eV": -4162.237085927501,
            "max_force_eV_A": 0.47028084589104147,
            "forces_eV_A": [
                [-0.2822910242714707, 0.20694251015432724, 0.0],
                [0.023502803560100094, -0.47028084589104147, 0.0],
                [-0.17143034535680068, 0.0211413168824785, 0.0],
                [0.06757533639056822, 0.17552992856623145, 0.0],
                [0.12886830683019954, -0.12455365053248127, 0.0],
                [0.23377492284740348, 0.19122074082048554, 0.0],
            ],
            "stress": [
                -0.0007728586364225908,
                -0.00011395989526723694,
                0.0,
                0.0,
                0.0,
                -3.345598360723563e-05,
            ],
        },
        "omol0": {
            "role": "nonperiodic target scorer",
            "provider": "MACEOMOLProvider",
            "pbc": False,
            "source_ref": "mace_omol_0",
            "source_commit": "6009f59095a1c471e77fa8e77b989b1bb81435b3",
            "source_url": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/MACE-omol-0-extra-large-1024.model",
            "sha256": "9b64b4fd5153ca578c694abc57806d8111050de6ff652e695c9b525bc4d36469",
            "size_bytes": 422242640,
            "energy_eV": -4159.718571495262,
            "max_force_eV_A": 0.4887109118211197,
            "forces_eV_A": [
                [-0.38706764902755286, 0.09570616208091787, 0.0],
                [0.18865266613545328, -0.4887109118211197, 0.0],
                [-0.22245860299135153, 0.146049668731907, 0.0],
                [-0.008953762765481627, 0.011990194397120381, 0.0],
                [0.24289919235291152, -0.13798085532172125, 0.0],
                [0.1869281562960212, 0.37294574193289576, 0.0],
            ],
            "stress": None,
        },
        "polar1m": {
            "role": "outer SMD density provider",
            "provider": "SMDPolarDeltaProvider",
            "pbc": False,
            "source_ref": "mace_polar_1",
            "source_commit": "2ee7e58beac3a9a12ad3644de1c3591c1ff11c2c",
            "source_url": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-M.model",
            "sha256": "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a",
            "size_bytes": 68133235,
            "energy_eV": -2079.86392931896,
            "max_force_eV_A": 0.24601890962074524,
            "forces_eV_A": [
                [-0.1458899213508661, -0.14301995452892796, 0.00017483042688030753],
                [0.24601890962074524, -0.048879362300277895, -8.730839143431297e-05],
                [-0.10012898826987564, 0.19189931682920208, -8.752203544599453e-05],
            ],
            "stress": None,
        },
    }

    cfg = expected[role]
    for key in ("role", "provider", "pbc", "source_ref", "source_commit", "source_url", "sha256", "size_bytes"):
        assert model[key] == cfg[key]
    assert isinstance(model["energy_eV"], float)
    assert isinstance(model["max_force_eV_A"], float)
    assert model["energy_eV"] == pytest.approx(cfg["energy_eV"], rel=0, abs=0)
    assert model["max_force_eV_A"] == pytest.approx(cfg["max_force_eV_A"], rel=0, abs=0)
    _assert_float64_array(model["forces_eV_A"], cfg["forces_eV_A"])

    if cfg["stress"] is None:
        assert "stress_eV_A3_voigt" not in model
    else:
        _assert_float64_array(model["stress_eV_A3_voigt"], cfg["stress"])


def test_model_probe_metadata_and_environment_status_are_exact():
    probe = load_json(MODEL_FIX)

    assert probe["fixture_name"] == "route-a-real-model-probes"
    assert probe["fixture_version"] == "1.0.0"
    assert probe["protocol_version"] == "1.0.0"
    assert probe["engineering_evidence_only"] is True
    assert probe["scientific_status"] == "engineering-evidence-only"
    assert "not solvation free energies" in probe["warning"]

    env = probe["environment"]
    assert env["dtype"] == "float64"
    assert env["python"] == "3.11.14"
    assert env["torch"] == "2.12.0+cu130"
    assert env["torch_device"] == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert env["cuda_available"] is True
    assert env["ase"] == "3.27.0"
    assert env["mace_torch"] == "0.3.16"
    assert env["cuequivariance"] is False

    geom = probe["geometry"]
    assert geom["name"] == "water-dimer-angstrom"
    pos = np.asarray(geom["positions_A"], dtype=np.float64)
    assert pos.shape == (6, 3)
    assert pos.dtype == np.float64
    assert geom["symbols"] == ["O", "H", "H", "O", "H", "H"]


def test_model_probe_off24_sampler_omol_target_and_polar_outer_records_exact_values():
    probe = load_json(MODEL_FIX)
    _assert_model_record("off24", probe["models"]["off24"])
    _assert_model_record("omol0", probe["models"]["omol0"])
    _assert_model_record("polar1m", probe["models"]["polar1m"])


def test_model_probe_and_radius_profile_are_mutually_consistent():
    probe = load_json(MODEL_FIX)
    radius_profile = load_json(RADIUS_PATH)

    assert radius_profile["doi"] == "10.1021/j100785a001"
    assert radius_profile["source_doi"] == "10.1021/j100785a001"
    assert radius_profile["radii"]["O"] == 1.52

    for item in probe["models"].values():
        assert item["license"] == "Academic Software Licence"

    for symbol in radius_profile["radii"]:
        assert radius_profile["value_provenance"][symbol]["doi"] == "10.1021/j100785a001"
        if symbol == "O":
            assert "Mantina" in radius_profile["value_provenance"][symbol]["notes"]

    # mantle line must be lineage-only, not a changed oxygen value
    assert "Mantina et al." in radius_profile["mantina_lineage"]
    assert radius_profile["mantina_doi"] == "10.1021/jp8111556"
    
