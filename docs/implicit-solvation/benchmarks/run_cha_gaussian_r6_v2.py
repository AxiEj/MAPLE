#!/usr/bin/env python3
"""Thin CLI for the closed Gaussian-CHA R6 derivative-v2 campaign context."""

from __future__ import annotations

from run_cha_gaussian_opt import V2_RUN_CONTEXT, main

if __name__ == "__main__":
    raise SystemExit(main(context=V2_RUN_CONTEXT))
