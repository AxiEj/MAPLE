#!/usr/bin/env python3
"""Run the MAPLE real-backend release gate locally.

This is the production gate for model-backed batch contracts when the
project wants a local, explicit release command instead of an external runner.
It deliberately sets both flags:

* MAPLE_REAL_BACKEND_SMOKE=1 loads real ANI/AIMNet2/MACE/UMA assets.
* MAPLE_REAL_BACKEND_REQUIRED=1 turns missing modules/weights/cache files into
  test failures instead of skips.

Extra CLI arguments are forwarded to pytest after the smoke test path.
"""
from __future__ import annotations

import os
import subprocess
import sys


TEST_PATH = "maple/function/calculator/test_real_backend_hessian_smoke.py"


def main(argv: list[str]) -> int:
    env = os.environ.copy()
    env["MAPLE_REAL_BACKEND_SMOKE"] = "1"
    env["MAPLE_REAL_BACKEND_REQUIRED"] = "1"
    cmd = [sys.executable, "-m", "pytest", "-q", TEST_PATH, *argv]
    print("$ " + " ".join(cmd))
    print("MAPLE_REAL_BACKEND_SMOKE=1")
    print("MAPLE_REAL_BACKEND_REQUIRED=1")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
