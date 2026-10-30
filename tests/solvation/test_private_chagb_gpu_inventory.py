"""The shared Torch engine has one active installed release identity."""

import hashlib
from pathlib import Path

from maple.function.calculator import set_calculator


def test_installed_inventory_binds_only_current_torch_release():
    inventory = set_calculator._installed_runtime_source_inventory()
    root = Path(set_calculator.__file__).resolve().parents[3]
    relative = "maple/function/calculator/data/private_cha_torch_fp32_release_v1.json"
    assert inventory[relative] == hashlib.sha256((root / relative).read_bytes()).hexdigest()
    for retired in ("private_cha_fp32_release_v1.json", "private_cha_gpu_fp32_release_v1.json"):
        assert f"maple/function/calculator/data/{retired}" not in inventory
