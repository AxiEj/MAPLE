import pytest

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "production_validation.py"
_SPEC = importlib.util.spec_from_file_location("production_validation", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
_parse_model_options = _MODULE._parse_model_options


def test_model_option_parser_coerces_supported_scalar_types():
    opts = _parse_model_options([
        "dispersion=true",
        "cutoff=12.0",
        "ewald_accuracy=1e-6",
        "max_steps=25",
        "head=omat",
    ])
    assert opts == {
        "dispersion": True,
        "cutoff": 12.0,
        "ewald_accuracy": 1e-6,
        "max_steps": 25,
        "head": "omat",
    }


def test_model_option_parser_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate --model-option key: cutoff"):
        _parse_model_options(["cutoff=12.0", "cutoff=10.0"])
